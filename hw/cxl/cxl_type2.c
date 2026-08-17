/*
 * CXL Type 2 Device (Accelerator with Coherent Memory)
 * GPU Passthrough Forwarder with CPU-GPU Coherency
 *
 * This implements a CXL Type 2 device that combines:
 * - Type 1 cache coherency (CXL.cache protocol)
 * - Type 3 device-attached memory (CXL.mem protocol)
 * - GPU passthrough via VFIO
 * - Full coherency protocol between CPU and GPU memory
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"

#include "hw/cxl/cxl_type2.h"
#include "exec/target_page.h"
#include "hw/core/cpu.h"
#include "hw/cxl/cxl.h"
#include "hw/cxl/cxl_cdat.h"
#include "hw/cxl/cxl_component.h"
#include "hw/cxl/cxl_device.h"
#include "hw/cxl/cxl_hetgpu.h"
#include "hw/cxl/cxl_pci.h"
#include "hw/cxl/cxl_type2_coherency.h"
#include "hw/cxl/cxl_type2_cuda_contract.h"
#include "hw/cxl/cxl_type2_gpu_cmd.h"
#include "hw/cxl/cxl_type2_model_alias_gate.h"
#include "hw/pci/msix.h"
#include "hw/pci/pci.h"
#include "hw/pci/pcie.h"
#include "hw/qdev-properties-system.h"
#include "hw/qdev-properties.h"
#include "hw/virtio/vhost-user-fs.h"
#include "io/channel-socket.h"
#include "migration/vmstate.h"
#include "qapi/error.h"
#include "qemu/cutils.h"
#include "qemu/error-report.h"
#include "qemu/log.h"
#include "qemu/module.h"
#include "qemu/plugin.h"
#include "qemu/range.h"
#include "qemu/rcu.h"
#include "qemu/sockets.h"
#include "qemu/timer.h"
#include "qemu/units.h"
#include "qobject/qdict.h"
#include "qobject/qjson.h"
#include "qobject/qlist.h"
#include "qobject/qnum.h"
#include "system/hw_accel.h"
#include "system/memory.h"
#include "system/runstate.h"
#include <fcntl.h>
#include <linux/vfio.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zstd.h>

extern int LZ4_decompress_safe(const char *src, char *dst, int compressed_size,
                               int dst_capacity);

#ifndef DEFAULT_HETGPU_LIB_PATH
#define DEFAULT_HETGPU_LIB_PATH NULL
#endif

#define CXL_OP_READ 0
#define CXL_OP_WRITE 1
#define CXL_OP_FENCE 5
#define CXL_OP_BI_ENABLE 14
#define CXL_OP_BI_DISABLE 15
#define CXL_OP_BI_INVALIDATE 16
#define CXL_OP_BI_WRITEBACK 17
#define CXL_OP_BI_QUERY 18
#define CXL_OP_RANGE_READ 19
#define CXL_OP_RANGE_WRITE 20

/* A CXL.mem response is one fixed-size message. A missing response must
 * release the QEMU device path instead of consuming the outer CNB timeout. */
#define CXL_MEMSIM_RESPONSE_TIMEOUT_NS (1000LL * 1000 * 1000)

#define CXL_CUDA_MEMHOSTREGISTER_PORTABLE 0x01U
#define CXL_CUDA_MEMHOSTREGISTER_DEVICEMAP 0x02U
#define CXL_CUDA_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY 19

static int64_t cxl_type2_host_monotonic_ns(void) {
  struct timespec now;

  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
    return 0;
  }
  return now.tv_sec * NANOSECONDS_PER_SECOND + now.tv_nsec;
}

typedef struct QEMU_PACKED CXLMemSimRequest {
  uint8_t op_type;
  uint64_t addr;
  uint64_t size;
  uint64_t timestamp;
  uint64_t value;
  uint64_t expected;
  uint8_t data[64];
} CXLMemSimRequest;

typedef struct QEMU_PACKED CXLMemSimResponse {
  uint8_t status;
  uint64_t latency_ns;
  uint64_t old_value;
  uint8_t data[64];
} CXLMemSimResponse;

/* Forward declarations for hetGPU coherency integration */
static void cxl_type2_hetgpu_coherency_callback(void *opaque, uint64_t addr,
                                                uint64_t size, bool invalidate);
static bool cxl_type2_memsim_request_ext(CXLType2State *ct2d, uint8_t op_type,
                                         uint64_t addr, uint64_t size,
                                         const uint8_t *data, uint64_t value,
                                         uint64_t expected,
                                         CXLMemSimResponse *resp);
static bool cxl_type2_memsim_request(CXLType2State *ct2d, uint8_t op_type,
                                     uint64_t addr, uint64_t size,
                                     const uint8_t *data,
                                     CXLMemSimResponse *resp);
static bool cxl_type2_fabric_access_allowed(CXLType2State *ct2d, uint64_t addr,
                                            uint64_t size, bool is_write,
                                            bool is_atomic);
static bool cxl_type2_register_gpu_handle(void ***handles, size_t *capacity,
                                          uint32_t *count, void *handle,
                                          uint32_t *id);

static bool cxl_type2_range_contains(uint64_t outer_offset, uint64_t outer_size,
                                     uint64_t inner_offset,
                                     uint64_t inner_size) {
  return outer_size && inner_size && inner_offset >= outer_offset &&
         inner_offset - outer_offset <= outer_size &&
         inner_size <= outer_size - (inner_offset - outer_offset);
}

static bool cxl_type2_bulk_memsim_access(CXLType2State *ct2d, uint8_t op_type,
                                         uint64_t addr, size_t size,
                                         uint8_t *data) {
  uint8_t range_op;

  if (ct2d->memsim.use_shm) {
    /* Range authorization is currently owned by the TCP server. */
    return false;
  }
  if (!ct2d->memsim.connected || !size || !data) {
    return false;
  }
  range_op = op_type == CXL_OP_WRITE ? CXL_OP_RANGE_WRITE : CXL_OP_RANGE_READ;
  return cxl_type2_memsim_request(ct2d, range_op, addr, size, NULL, NULL);
}

static uint8_t *cxl_type2_bar4_host_ptr(CXLType2State *ct2d, uint64_t offset,
                                        size_t size) {
  if (offset >= ct2d->coherent_pool.base_offset &&
      offset - ct2d->coherent_pool.base_offset <= ct2d->coherent_pool.size &&
      size <= ct2d->coherent_pool.size -
                  (offset - ct2d->coherent_pool.base_offset)) {
    uint8_t *pool = memory_region_get_ram_ptr(&ct2d->coherent_pool_mem);
    return pool ? pool + offset - ct2d->coherent_pool.base_offset : NULL;
  }

  uint8_t *mem = memory_region_get_ram_ptr(&ct2d->device_mem);
  return mem && offset <= ct2d->device_mem_size &&
                 size <= ct2d->device_mem_size - offset
             ? mem + offset
             : NULL;
}

/* ========================================================================
 * Coherency Protocol Implementation
 * ======================================================================== */

void cxl_type2_coherency_init(CXLType2State *ct2d) {
  qemu_mutex_init(&ct2d->coherency.lock);
  ct2d->coherency.cache_lines =
      g_hash_table_new_full(g_int64_hash, g_int64_equal, g_free, g_free);
  ct2d->coherency.cache_hits = 0;
  ct2d->coherency.cache_misses = 0;
  ct2d->coherency.coherency_ops = 0;
  ct2d->coherency.snoops = 0;
  ct2d->coherency.coherency_enabled = true;

  qemu_log("CXL Type2: Coherency protocol initialized\n");
}

void cxl_type2_coherency_cleanup(CXLType2State *ct2d) {
  if (ct2d->coherency.cache_lines) {
    g_hash_table_destroy(ct2d->coherency.cache_lines);
    ct2d->coherency.cache_lines = NULL;
  }
  qemu_mutex_destroy(&ct2d->coherency.lock);

  qemu_log("CXL Type2: Coherency stats - Hits: %lu, Misses: %lu, Ops: %lu, "
           "Snoops: %lu\n",
           ct2d->coherency.cache_hits, ct2d->coherency.cache_misses,
           ct2d->coherency.coherency_ops, ct2d->coherency.snoops);
}

CXLCacheLine *cxl_type2_cache_lookup(CXLType2State *ct2d, uint64_t addr) {
  /* Align to cache line boundary */
  uint64_t cache_line_addr = addr & ~0x3F;
  CXLCacheLine *line;

  qemu_mutex_lock(&ct2d->coherency.lock);
  line = g_hash_table_lookup(ct2d->coherency.cache_lines, &cache_line_addr);

  if (line) {
    ct2d->coherency.cache_hits++;
  } else {
    ct2d->coherency.cache_misses++;
  }
  qemu_mutex_unlock(&ct2d->coherency.lock);

  return line;
}

void cxl_type2_cache_insert(CXLType2State *ct2d, uint64_t addr,
                            const uint8_t *data, CXLCoherencyState state) {
  uint64_t cache_line_addr = addr & ~0x3F;
  CXLCacheLine *line = g_new0(CXLCacheLine, 1);
  uint64_t *key = g_new(uint64_t, 1);

  *key = cache_line_addr;
  line->tag = cache_line_addr;
  line->state = state;
  line->dirty = false;
  line->timestamp = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
  if (data) {
    memcpy(line->data, data, 64);
  }

  qemu_mutex_lock(&ct2d->coherency.lock);
  g_hash_table_insert(ct2d->coherency.cache_lines, key, line);
  ct2d->coherency.coherency_ops++;
  qemu_mutex_unlock(&ct2d->coherency.lock);
}

void cxl_type2_cache_invalidate(CXLType2State *ct2d, uint64_t addr) {
  uint64_t cache_line_addr = addr & ~0x3F;

  qemu_mutex_lock(&ct2d->coherency.lock);
  if (g_hash_table_remove(ct2d->coherency.cache_lines, &cache_line_addr)) {
    ct2d->coherency.coherency_ops++;
    qemu_log_mask(LOG_TRACE, "CXL Type2: Cache invalidate at 0x%lx\n",
                  cache_line_addr);
  }
  qemu_mutex_unlock(&ct2d->coherency.lock);

  if (ct2d->bi_enabled) {
    cxl_type2_memsim_request(ct2d, CXL_OP_BI_INVALIDATE, cache_line_addr, 64,
                             NULL, NULL);
  }
}

void cxl_type2_cache_writeback(CXLType2State *ct2d, uint64_t addr) {
  uint64_t cache_line_addr = addr & ~0x3F;
  CXLCacheLine *line;

  qemu_mutex_lock(&ct2d->coherency.lock);
  line = g_hash_table_lookup(ct2d->coherency.cache_lines, &cache_line_addr);

  if (line && line->dirty) {
    /* Write back to device memory */
    if (cache_line_addr < ct2d->device_mem_size) {
      uint8_t *mem_ptr = memory_region_get_ram_ptr(&ct2d->device_mem);
      if (mem_ptr) {
        memcpy(mem_ptr + cache_line_addr, line->data, 64);
      }
    }

    if (ct2d->bi_enabled) {
      cxl_type2_memsim_request(ct2d, CXL_OP_BI_WRITEBACK, cache_line_addr, 64,
                               line->data, NULL);
    } else {
      cxl_type2_memsim_request(ct2d, CXL_OP_WRITE, cache_line_addr, 64,
                               line->data, NULL);
    }

    line->dirty = false;
    ct2d->coherency.coherency_ops++;

    qemu_log_mask(LOG_TRACE, "CXL Type2: Cache writeback at 0x%lx\n",
                  cache_line_addr);
  }
  qemu_mutex_unlock(&ct2d->coherency.lock);
}

static bool cxl_type2_cache_fill_line(CXLType2State *ct2d,
                                      uint64_t cache_line_addr,
                                      uint8_t cache_data[64]) {
  HetGPUState *hetgpu = &ct2d->gpu_info.hetgpu_state;
  uint8_t *mem_ptr;

  memset(cache_data, 0, 64);

  if (hetgpu->initialized) {
    HetGPUError err;

    err = hetgpu_memcpy_dtoh(hetgpu, cache_data, cache_line_addr, 64);
    if (err == HETGPU_SUCCESS) {
      return true;
    }
  }

  mem_ptr = memory_region_get_ram_ptr(&ct2d->device_mem);
  if (mem_ptr && cache_line_addr < ct2d->device_mem_size) {
    size_t copy_size = MIN(64, ct2d->device_mem_size - cache_line_addr);

    memcpy(cache_data, mem_ptr + cache_line_addr, copy_size);
    return true;
  }

  return false;
}

static void cxl_type2_cache_prefetch(CXLType2State *ct2d, uint64_t addr,
                                     uint64_t size, bool write_intent) {
  uint64_t start;
  uint64_t last;

  if (!ct2d->coherency.coherency_enabled || size == 0) {
    return;
  }

  start = addr & ~0x3FULL;
  if (addr + size - 1 < addr) {
    last = UINT64_MAX & ~0x3FULL;
  } else {
    last = (addr + size - 1) & ~0x3FULL;
  }

  for (uint64_t line_addr = start;; line_addr += 64) {
    CXLCacheLine *line = cxl_type2_cache_lookup(ct2d, line_addr);

    if (line && line->state != CXL_COHERENCY_INVALID) {
      if (write_intent) {
        qemu_mutex_lock(&ct2d->coherency.lock);
        line->state = CXL_COHERENCY_EXCLUSIVE;
        line->timestamp = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
        qemu_mutex_unlock(&ct2d->coherency.lock);
      }
    } else {
      uint8_t cache_data[64];
      CXLCoherencyState state =
          write_intent ? CXL_COHERENCY_EXCLUSIVE : CXL_COHERENCY_SHARED;

      if (cxl_type2_cache_fill_line(ct2d, line_addr, cache_data)) {
        cxl_type2_cache_insert(ct2d, line_addr, cache_data, state);
        cxl_type2_memsim_request(ct2d, CXL_OP_READ, line_addr, 64, NULL, NULL);
        ct2d->stats.read_ops++;
      }
    }

    if (line_addr == last || line_addr > UINT64_MAX - 64) {
      break;
    }
  }

  ct2d->coherency.coherency_ops++;

  qemu_log_mask(LOG_TRACE,
                "CXL Type2: Cache prefetch addr=0x%" PRIx64 " size=0x%" PRIx64
                " write_intent=%d\n",
                addr, size, write_intent);
}

bool cxl_type2_snoop_request(CXLType2State *ct2d, uint64_t addr,
                             bool invalidate) {
  uint64_t cache_line_addr = addr & ~0x3F;
  CXLCacheLine *line;
  bool hit = false;

  qemu_mutex_lock(&ct2d->coherency.lock);
  ct2d->coherency.snoops++;

  line = g_hash_table_lookup(ct2d->coherency.cache_lines, &cache_line_addr);
  if (line) {
    hit = true;

    /* If dirty, write back before invalidation */
    if (line->dirty) {
      qemu_mutex_unlock(&ct2d->coherency.lock);
      cxl_type2_cache_writeback(ct2d, addr);
      qemu_mutex_lock(&ct2d->coherency.lock);
      line = g_hash_table_lookup(ct2d->coherency.cache_lines, &cache_line_addr);
    }

    if (invalidate && line) {
      g_hash_table_remove(ct2d->coherency.cache_lines, &cache_line_addr);
    } else if (line) {
      /* Downgrade to shared */
      line->state = CXL_COHERENCY_SHARED;
    }
  }

  qemu_mutex_unlock(&ct2d->coherency.lock);

  qemu_log_mask(LOG_TRACE,
                "CXL Type2: Snoop request at 0x%lx, hit=%d, invalidate=%d\n",
                cache_line_addr, hit, invalidate);

  return hit;
}

/* ========================================================================
 * GPU Passthrough Implementation - VFIO Helpers
 * ======================================================================== */

/* Open and setup VFIO container */
static int cxl_type2_vfio_container_init(CXLType2State *ct2d, Error **errp) {
  int container_fd, version;

  /* Open VFIO container */
  container_fd = open("/dev/vfio/vfio", O_RDWR);
  if (container_fd < 0) {
    error_setg(errp, "Failed to open /dev/vfio/vfio: %s", strerror(errno));
    return -1;
  }

  /* Check VFIO API version */
  version = ioctl(container_fd, VFIO_GET_API_VERSION);
  if (version != VFIO_API_VERSION) {
    error_setg(errp, "VFIO API version mismatch: expected %d, got %d",
               VFIO_API_VERSION, version);
    close(container_fd);
    return -1;
  }

  /* Check VFIO extension support */
  if (!ioctl(container_fd, VFIO_CHECK_EXTENSION, VFIO_TYPE1_IOMMU) &&
      !ioctl(container_fd, VFIO_CHECK_EXTENSION, VFIO_TYPE1v2_IOMMU)) {
    error_setg(errp, "VFIO does not support TYPE1 or TYPE1v2 IOMMU");
    close(container_fd);
    return -1;
  }

  ct2d->gpu_info.vfio_container = GINT_TO_POINTER(container_fd);
  qemu_log("CXL Type2: VFIO container initialized (fd=%d)\n", container_fd);

  return container_fd;
}

/* Setup VFIO group for GPU device */
static int cxl_type2_vfio_group_init(CXLType2State *ct2d, const char *pci_addr,
                                     Error **errp) {
  char group_path[256];
  char iommu_group_path[512];
  char *group_name;
  int group_fd, container_fd;
  ssize_t len;
  struct vfio_group_status group_status = {.argsz = sizeof(group_status)};

  /* Get IOMMU group for the device */
  snprintf(iommu_group_path, sizeof(iommu_group_path),
           "/sys/bus/pci/devices/%s/iommu_group", pci_addr);

  len = readlink(iommu_group_path, group_path, sizeof(group_path) - 1);
  if (len < 0) {
    error_setg(errp, "Failed to read IOMMU group for %s: %s", pci_addr,
               strerror(errno));
    return -1;
  }
  group_path[len] = '\0';

  /* Extract group number from path */
  group_name = basename(group_path);

  /* Open VFIO group */
  snprintf(group_path, sizeof(group_path), "/dev/vfio/%s", group_name);
  group_fd = open(group_path, O_RDWR);
  if (group_fd < 0) {
    error_setg(errp, "Failed to open VFIO group %s: %s", group_path,
               strerror(errno));
    return -1;
  }

  /* Check group is viable */
  if (ioctl(group_fd, VFIO_GROUP_GET_STATUS, &group_status) < 0) {
    error_setg(errp, "Failed to get VFIO group status: %s", strerror(errno));
    close(group_fd);
    return -1;
  }

  if (!(group_status.flags & VFIO_GROUP_FLAGS_VIABLE)) {
    error_setg(errp,
               "VFIO group is not viable (not all devices bound to VFIO)");
    close(group_fd);
    return -1;
  }

  /* Add group to container */
  container_fd = GPOINTER_TO_INT(ct2d->gpu_info.vfio_container);
  if (ioctl(group_fd, VFIO_GROUP_SET_CONTAINER, &container_fd) < 0) {
    error_setg(errp, "Failed to add group to container: %s", strerror(errno));
    close(group_fd);
    return -1;
  }

  /* Enable IOMMU for the container */
  if (ioctl(container_fd, VFIO_SET_IOMMU, VFIO_TYPE1v2_IOMMU) < 0) {
    if (ioctl(container_fd, VFIO_SET_IOMMU, VFIO_TYPE1_IOMMU) < 0) {
      error_setg(errp, "Failed to set IOMMU type: %s", strerror(errno));
      close(group_fd);
      return -1;
    }
  }

  ct2d->gpu_info.vfio_group = GINT_TO_POINTER(group_fd);
  qemu_log("CXL Type2: VFIO group %s initialized (fd=%d)\n", group_name,
           group_fd);

  return group_fd;
}

/* Open and setup VFIO device */
static int cxl_type2_vfio_device_init(CXLType2State *ct2d, const char *pci_addr,
                                      Error **errp) {
  int group_fd, device_fd;
  struct vfio_device_info device_info = {.argsz = sizeof(device_info)};
  struct vfio_region_info region_info = {.argsz = sizeof(region_info)};

  group_fd = GPOINTER_TO_INT(ct2d->gpu_info.vfio_group);

  /* Get device FD */
  device_fd = ioctl(group_fd, VFIO_GROUP_GET_DEVICE_FD, pci_addr);
  if (device_fd < 0) {
    error_setg(errp, "Failed to get device FD for %s: %s", pci_addr,
               strerror(errno));
    return -1;
  }

  /* Get device info */
  if (ioctl(device_fd, VFIO_DEVICE_GET_INFO, &device_info) < 0) {
    error_setg(errp, "Failed to get device info: %s", strerror(errno));
    close(device_fd);
    return -1;
  }

  qemu_log("CXL Type2: GPU device %s has %d regions, %d IRQs\n", pci_addr,
           device_info.num_regions, device_info.num_irqs);

  /* Get BAR0 region info (GPU memory typically in BAR0 or BAR1) */
  region_info.index = VFIO_PCI_BAR0_REGION_INDEX;
  if (ioctl(device_fd, VFIO_DEVICE_GET_REGION_INFO, &region_info) < 0) {
    error_setg(errp, "Failed to get BAR0 region info: %s", strerror(errno));
    close(device_fd);
    return -1;
  }

  /* Map GPU memory region */
  if (region_info.size > 0) {
    void *bar_mem = mmap(NULL, region_info.size, PROT_READ | PROT_WRITE,
                         MAP_SHARED, device_fd, region_info.offset);

    if (bar_mem == MAP_FAILED) {
      error_setg(errp, "Failed to mmap GPU BAR0: %s", strerror(errno));
      close(device_fd);
      return -1;
    }

    ct2d->gpu_info.gpu_mem_base = region_info.offset;
    ct2d->gpu_info.gpu_mem_size = region_info.size;

    qemu_log("CXL Type2: Mapped GPU BAR0 at offset 0x%llx, size %llu MB\n",
             region_info.offset, region_info.size / (1024 * 1024));
  }

  ct2d->gpu_info.vfio_device_fd = device_fd;

  return device_fd;
}

static int cxl_type2_vfio_dma_map_range(CXLType2State *ct2d, uint64_t offset,
                                        uint64_t size, Error **errp) {
  int container_fd = GPOINTER_TO_INT(ct2d->gpu_info.vfio_container);
  uint8_t *mem_ptr = memory_region_get_ram_ptr(&ct2d->device_mem);
  struct vfio_iommu_type1_dma_map dma_map = {
      .argsz = sizeof(dma_map),
      .flags = VFIO_DMA_MAP_FLAG_READ | VFIO_DMA_MAP_FLAG_WRITE,
      .vaddr = (uint64_t)(mem_ptr + offset),
      .iova = offset,
      .size = size,
  };

  if (ioctl(container_fd, VFIO_IOMMU_MAP_DMA, &dma_map) < 0) {
    error_setg(errp,
               "Failed to map DMA region IOVA=0x%" PRIx64 " size=0x%" PRIx64
               ": %s",
               offset, size, strerror(errno));
    return -1;
  }

  qemu_log("CXL Type2: DMA mapping configured - IOVA: 0x%" PRIx64
           ", size: 0x%" PRIx64 "\n",
           offset, size);
  return 0;
}

static void cxl_type2_vfio_dma_unmap_range(CXLType2State *ct2d, uint64_t offset,
                                           uint64_t size) {
  int container_fd = GPOINTER_TO_INT(ct2d->gpu_info.vfio_container);
  struct vfio_iommu_type1_dma_unmap dma_unmap = {
      .argsz = sizeof(dma_unmap),
      .iova = offset,
      .size = size,
  };

  if (size && ioctl(container_fd, VFIO_IOMMU_UNMAP_DMA, &dma_unmap) < 0) {
    qemu_log("CXL Type2: Warning - Failed to unmap DMA IOVA=0x%" PRIx64
             " size=0x%" PRIx64 ": %s\n",
             offset, size, strerror(errno));
  }
}

size_t cxl_type2_vfio_dma_ranges(const CXLType2State *ct2d,
                                 CXLType2MemoryRange ranges[2]) {
  uint64_t aperture_end;
  size_t count = 0;

  if (!ct2d->model_aperture.size) {
    ranges[count++] = (CXLType2MemoryRange){
        .size = ct2d->device_mem_size,
    };
    return count;
  }

  aperture_end = ct2d->model_aperture.offset + ct2d->model_aperture.size;
  if (ct2d->model_aperture.offset) {
    ranges[count++] = (CXLType2MemoryRange){
        .size = ct2d->model_aperture.offset,
    };
  }
  if (aperture_end < ct2d->device_mem_size) {
    ranges[count++] = (CXLType2MemoryRange){
        .offset = aperture_end,
        .size = ct2d->device_mem_size - aperture_end,
    };
  }

  return count;
}

/*
 * The model aperture is serviced through the file-backed CXL path. Mapping it
 * through the ordinary BAR4 DMA backing would create a second physical owner.
 */
static int cxl_type2_vfio_dma_map(CXLType2State *ct2d, Error **errp) {
  CXLType2MemoryRange ranges[2];
  size_t count;

  if (!memory_region_get_ram_ptr(&ct2d->device_mem)) {
    error_setg(errp, "Failed to get device memory pointer");
    return -1;
  }

  count = cxl_type2_vfio_dma_ranges(ct2d, ranges);
  for (size_t i = 0; i < count; i++) {
    if (cxl_type2_vfio_dma_map_range(ct2d, ranges[i].offset, ranges[i].size,
                                     errp) < 0) {
      while (i) {
        i--;
        cxl_type2_vfio_dma_unmap_range(ct2d, ranges[i].offset, ranges[i].size);
      }
      return -1;
    }
  }

  return 0;
}

/* Interrupt handling thread for GPU events */
static void *cxl_type2_irq_thread(void *opaque) {
  CXLType2State *ct2d = opaque;
  PCIDevice *pci_dev = PCI_DEVICE(ct2d);
  int device_fd = ct2d->gpu_info.vfio_device_fd;
  struct vfio_irq_info irq_info = {.argsz = sizeof(irq_info)};
  int *event_fds = NULL;
  int num_irqs = 0;
  int ret, i;

  /* Get IRQ info for MSI-X */
  irq_info.index = VFIO_PCI_MSIX_IRQ_INDEX;
  if (ioctl(device_fd, VFIO_DEVICE_GET_IRQ_INFO, &irq_info) < 0) {
    qemu_log("CXL Type2: Failed to get MSI-X IRQ info: %s\n", strerror(errno));
    return NULL;
  }

  num_irqs = irq_info.count;
  if (num_irqs == 0) {
    qemu_log("CXL Type2: No MSI-X interrupts available\n");
    return NULL;
  }

  qemu_log("CXL Type2: Setting up %d MSI-X interrupts\n", num_irqs);

  /* Create event FDs for each interrupt */
  event_fds = g_new0(int, num_irqs);
  for (i = 0; i < num_irqs; i++) {
    event_fds[i] = eventfd(0, EFD_NONBLOCK);
    if (event_fds[i] < 0) {
      qemu_log("CXL Type2: Failed to create event FD %d: %s\n", i,
               strerror(errno));
      goto cleanup;
    }
  }

  /* Set up VFIO IRQ forwarding */
  struct vfio_irq_set *irq_set;
  size_t irq_set_size = sizeof(*irq_set) + num_irqs * sizeof(int);
  irq_set = g_malloc0(irq_set_size);
  irq_set->argsz = irq_set_size;
  irq_set->flags = VFIO_IRQ_SET_DATA_EVENTFD | VFIO_IRQ_SET_ACTION_TRIGGER;
  irq_set->index = VFIO_PCI_MSIX_IRQ_INDEX;
  irq_set->start = 0;
  irq_set->count = num_irqs;
  memcpy(irq_set->data, event_fds, num_irqs * sizeof(int));

  if (ioctl(device_fd, VFIO_DEVICE_SET_IRQS, irq_set) < 0) {
    qemu_log("CXL Type2: Failed to set IRQs: %s\n", strerror(errno));
    g_free(irq_set);
    goto cleanup;
  }

  g_free(irq_set);
  qemu_log("CXL Type2: MSI-X interrupt forwarding configured\n");

  /* Monitor interrupts and forward to guest */
  fd_set rfds;
  int max_fd = 0;
  for (i = 0; i < num_irqs; i++) {
    if (event_fds[i] > max_fd) {
      max_fd = event_fds[i];
    }
  }

  while (ct2d->gpu_info.passthrough_enabled) {
    FD_ZERO(&rfds);
    for (i = 0; i < num_irqs; i++) {
      FD_SET(event_fds[i], &rfds);
    }

    struct timeval tv = {.tv_sec = 1, .tv_usec = 0};
    ret = select(max_fd + 1, &rfds, NULL, NULL, &tv);

    if (ret < 0) {
      if (errno == EINTR) {
        continue;
      }
      qemu_log("CXL Type2: IRQ select failed: %s\n", strerror(errno));
      break;
    }

    if (ret == 0) {
      /* Timeout, check if still enabled */
      continue;
    }

    /* Check which interrupt fired */
    for (i = 0; i < num_irqs; i++) {
      if (FD_ISSET(event_fds[i], &rfds)) {
        uint64_t count;
        ssize_t s = read(event_fds[i], &count, sizeof(count));
        if (s == sizeof(count)) {
          qemu_log_mask(LOG_TRACE, "CXL Type2: GPU interrupt %d triggered\n",
                        i);

          /* Notify guest via MSI-X */
          if (msix_enabled(pci_dev)) {
            msix_notify(pci_dev, i);
          }

          /* Update statistics */
          ct2d->stats.gpu_accesses++;
        }
      }
    }
  }

cleanup:
  /* Clean up event FDs */
  for (i = 0; i < num_irqs; i++) {
    if (event_fds[i] >= 0) {
      close(event_fds[i]);
    }
  }
  g_free(event_fds);

  qemu_log("CXL Type2: IRQ thread exiting\n");
  return NULL;
}

/* ========================================================================
 * hetGPU Backend Implementation
 * ======================================================================== */

/* Coherency callback for hetGPU operations */
static void cxl_type2_hetgpu_coherency_callback(void *opaque, uint64_t addr,
                                                uint64_t size,
                                                bool invalidate) {
  CXLType2State *ct2d = opaque;

  if (!ct2d || !ct2d->coherency.coherency_enabled) {
    return;
  }

  /* Notify enhanced BAR coherency layer of GPU access */
  if (ct2d->bar_coherency.enabled) {
    cxl_bar_notify_gpu_access(&ct2d->bar_coherency, addr, size, invalidate);
  }

  if (invalidate) {
    /* GPU is writing - invalidate CPU cache lines */
    if (addr && size) {
      uint64_t aligned_addr = addr & ~0x3F;
      uint64_t end_addr = (addr + size + 63) & ~0x3F;
      for (uint64_t a = aligned_addr; a < end_addr; a += 64) {
        cxl_type2_cache_invalidate(ct2d, a);
      }
    }
  } else {
    /* GPU is reading - write back CPU cache lines */
    if (addr && size) {
      uint64_t aligned_addr = addr & ~0x3F;
      uint64_t end_addr = (addr + size + 63) & ~0x3F;
      for (uint64_t a = aligned_addr; a < end_addr; a += 64) {
        cxl_type2_cache_writeback(ct2d, a);
      }
    }
  }

  ct2d->coherency.coherency_ops++;
}

int cxl_type2_hetgpu_init(CXLType2State *ct2d, Error **errp) {
  HetGPUState *hetgpu = &ct2d->gpu_info.hetgpu_state;
  HetGPUError err;
  const char *lib_path;

  fprintf(stderr, "cxl_type2_hetgpu_init: ENTERED\n");
  fflush(stderr);

  /* Determine library path - try multiple locations */
  lib_path = ct2d->gpu_info.hetgpu_lib_path;
  fprintf(stderr, "cxl_type2_hetgpu_init: hetgpu_lib_path = '%s'\n",
          lib_path ? lib_path : "(null)");
  fflush(stderr);
  if (!lib_path || lib_path[0] == '\0') {
    lib_path = getenv("HETGPU_LIB_PATH");
    fprintf(stderr, "cxl_type2_hetgpu_init: env HETGPU_LIB_PATH = '%s'\n",
            lib_path ? lib_path : "(null)");
    fflush(stderr);
  }
  if (!lib_path || lib_path[0] == '\0') {
    /* Try system CUDA library first for real GPU passthrough */
    lib_path = DEFAULT_HETGPU_LIB_PATH;
  }
  if (!lib_path || lib_path[0] == '\0') {
    lib_path = "/usr/lib/x86_64-linux-gnu/libcuda.so";
  }

  fprintf(stderr, "cxl_type2_hetgpu_init: Using library: %s\n", lib_path);
  fprintf(stderr, "cxl_type2_hetgpu_init: backend=%d, device_index=%d\n",
          ct2d->gpu_info.hetgpu_backend, ct2d->gpu_info.hetgpu_device_index);
  fflush(stderr);

  /* Initialize hetGPU */
  if (ct2d->paired_case.required) {
    err = hetgpu_init_formal(hetgpu, ct2d->gpu_info.hetgpu_backend,
                             ct2d->gpu_info.hetgpu_device_index, lib_path);
  } else if (ct2d->model_alias_gate.output) {
    err = hetgpu_init_capability(hetgpu, ct2d->gpu_info.hetgpu_backend,
                                 ct2d->gpu_info.hetgpu_device_index, lib_path);
  } else {
    err = hetgpu_init(hetgpu, ct2d->gpu_info.hetgpu_backend,
                      ct2d->gpu_info.hetgpu_device_index, lib_path);
  }
  if (err != HETGPU_SUCCESS) {
    error_setg(errp, "hetGPU initialization failed: %s",
               hetgpu_get_error_string(err));
    return -1;
  }

  /* Create context */
  err = hetgpu_create_context(hetgpu);
  if (err != HETGPU_SUCCESS) {
    error_setg(errp, "hetGPU context creation failed: %s",
               hetgpu_get_error_string(err));
    hetgpu_cleanup(hetgpu);
    return -1;
  }

  /* Set up coherency callback */
  hetgpu_set_coherency_callback(hetgpu, cxl_type2_hetgpu_coherency_callback,
                                ct2d);

  ct2d->gpu_info.passthrough_enabled = true;
  ct2d->gpu_info.gpu_mem_size = hetgpu->props.total_memory;
  cxl_type2_cuda_allocation_table_reset(&ct2d->cuda_allocations);

  qemu_log("CXL Type2: hetGPU initialized - Backend: %s, Device: %s\n",
           hetgpu_get_backend_name(hetgpu->backend), hetgpu->props.name);
  qemu_log("CXL Type2: GPU Memory: %lu MB, Compute: %d.%d\n",
           hetgpu->props.total_memory / (1024 * 1024),
           hetgpu->props.compute_capability_major,
           hetgpu->props.compute_capability_minor);

  return 0;
}

void cxl_type2_hetgpu_cleanup(CXLType2State *ct2d) {
  HetGPUState *hetgpu = &ct2d->gpu_info.hetgpu_state;

  if (!hetgpu->initialized) {
    return;
  }

  qemu_log("CXL Type2: Cleaning up hetGPU backend\n");
  hetgpu_cleanup(hetgpu);
}

int cxl_type2_hetgpu_load_ptx(CXLType2State *ct2d, const char *ptx_source,
                              void **module) {
  HetGPUState *hetgpu = &ct2d->gpu_info.hetgpu_state;
  HetGPUError err;

  if (!hetgpu->initialized) {
    return -1;
  }

  err = hetgpu_load_ptx(hetgpu, ptx_source, (HetGPUModule *)module);
  if (err != HETGPU_SUCCESS) {
    qemu_log("CXL Type2: Failed to load PTX: %s\n",
             hetgpu_get_error_string(err));
    return -1;
  }

  return 0;
}

int cxl_type2_hetgpu_launch_kernel(CXLType2State *ct2d, void *function,
                                   uint32_t grid_x, uint32_t grid_y,
                                   uint32_t grid_z, uint32_t block_x,
                                   uint32_t block_y, uint32_t block_z,
                                   uint32_t shared_mem, void **args,
                                   size_t num_args) {
  HetGPUState *hetgpu = &ct2d->gpu_info.hetgpu_state;
  HetGPULaunchConfig config;
  HetGPUError err;

  if (!hetgpu->initialized) {
    return -1;
  }

  config.grid_dim[0] = grid_x;
  config.grid_dim[1] = grid_y;
  config.grid_dim[2] = grid_z;
  config.block_dim[0] = block_x;
  config.block_dim[1] = block_y;
  config.block_dim[2] = block_z;
  config.shared_mem_bytes = shared_mem;
  config.stream = NULL;

  err = hetgpu_launch_kernel(hetgpu, function, &config, args, num_args);
  if (err != HETGPU_SUCCESS) {
    qemu_log("CXL Type2: Kernel launch failed: %s\n",
             hetgpu_get_error_string(err));
    return -1;
  }

  ct2d->stats.gpu_accesses++;
  return 0;
}

int cxl_type2_hetgpu_malloc(CXLType2State *ct2d, size_t size,
                            uint64_t *dev_ptr) {
  HetGPUState *hetgpu = &ct2d->gpu_info.hetgpu_state;
  HetGPUError err;

  if (!hetgpu->initialized || !dev_ptr) {
    return -1;
  }

  err = hetgpu_malloc(hetgpu, size, HETGPU_MEM_HOST_MAPPED,
                      (HetGPUDevicePtr *)dev_ptr);
  if (err != HETGPU_SUCCESS) {
    qemu_log("CXL Type2: Memory allocation failed: %s\n",
             hetgpu_get_error_string(err));
    return -1;
  }

  return 0;
}

int cxl_type2_hetgpu_free(CXLType2State *ct2d, uint64_t dev_ptr) {
  HetGPUState *hetgpu = &ct2d->gpu_info.hetgpu_state;
  HetGPUError err;

  if (!hetgpu->initialized) {
    return -1;
  }

  err = hetgpu_free(hetgpu, dev_ptr);
  if (err != HETGPU_SUCCESS) {
    qemu_log("CXL Type2: Memory free failed: %s\n",
             hetgpu_get_error_string(err));
    return -1;
  }

  return 0;
}

int cxl_type2_hetgpu_memcpy_htod(CXLType2State *ct2d, uint64_t dst,
                                 const void *src, size_t size) {
  HetGPUState *hetgpu = &ct2d->gpu_info.hetgpu_state;
  HetGPUError err;

  if (!hetgpu->initialized) {
    return -1;
  }

  /* Invalidate cache before GPU write */
  cxl_type2_cache_invalidate(ct2d, dst);

  err = hetgpu_memcpy_htod(hetgpu, dst, src, size);
  if (err != HETGPU_SUCCESS) {
    qemu_log("CXL Type2: HtoD memcpy failed: %s\n",
             hetgpu_get_error_string(err));
    return -1;
  }

  ct2d->stats.gpu_accesses++;
  return 0;
}

int cxl_type2_hetgpu_memcpy_dtoh(CXLType2State *ct2d, void *dst, uint64_t src,
                                 size_t size) {
  HetGPUState *hetgpu = &ct2d->gpu_info.hetgpu_state;
  HetGPUError err;

  if (!hetgpu->initialized) {
    return -1;
  }

  /* Writeback cache before GPU read */
  cxl_type2_cache_writeback(ct2d, src);

  err = hetgpu_memcpy_dtoh(hetgpu, dst, src, size);
  if (err != HETGPU_SUCCESS) {
    qemu_log("CXL Type2: DtoH memcpy failed: %s\n",
             hetgpu_get_error_string(err));
    return -1;
  }

  ct2d->stats.gpu_accesses++;
  return 0;
}

int cxl_type2_hetgpu_sync(CXLType2State *ct2d) {
  HetGPUState *hetgpu = &ct2d->gpu_info.hetgpu_state;
  HetGPUError err;

  if (!hetgpu->initialized) {
    return -1;
  }

  err = hetgpu_synchronize(hetgpu);
  if (err != HETGPU_SUCCESS) {
    qemu_log("CXL Type2: Sync failed: %s\n", hetgpu_get_error_string(err));
    return -1;
  }

  return 0;
}

/* ========================================================================
 * GPU Passthrough Implementation - Main Functions
 * ======================================================================== */

int cxl_type2_gpu_init(CXLType2State *ct2d, Error **errp) {
  Error *local_err = NULL;
  int ret;

  /* Determine GPU mode if auto */
  if (ct2d->gpu_info.mode == CXL_TYPE2_GPU_MODE_AUTO) {
    if (ct2d->gpu_info.vfio_device && ct2d->gpu_info.vfio_device[0]) {
      ct2d->gpu_info.mode = CXL_TYPE2_GPU_MODE_VFIO;
    } else {
      /* Default to hetGPU mode - will use system libcuda.so for real GPU */
      ct2d->gpu_info.mode = CXL_TYPE2_GPU_MODE_HETGPU;
    }
  }

  /* Initialize based on mode */
  fprintf(stderr,
          "CXL Type2: GPU mode = %d (NONE=0, VFIO=1, HETGPU=2, AUTO=3)\n",
          ct2d->gpu_info.mode);
  fflush(stderr);
  switch (ct2d->gpu_info.mode) {
  case CXL_TYPE2_GPU_MODE_HETGPU:
    fprintf(stderr, "CXL Type2: Initializing hetGPU backend...\n");
    fflush(stderr);
    ret = cxl_type2_hetgpu_init(ct2d, errp);
    fprintf(stderr, "CXL Type2: cxl_type2_hetgpu_init returned %d\n", ret);
    fflush(stderr);
    if (ret == 0) {
      fprintf(stderr, "CXL Type2: hetGPU backend initialized successfully\n");
      fflush(stderr);
      return 0;
    }
    if (ct2d->paired_case.required || ct2d->model_alias_gate.output) {
      qemu_log("CXL Type2: strict mode rejects hetGPU "
               "initialization failure\n");
      return -1;
    }
    /* Fall through to VFIO or simulation if hetGPU fails */
    fprintf(stderr, "CXL Type2: hetGPU init failed, trying fallback\n");
    fflush(stderr);
    /* fall through */

  case CXL_TYPE2_GPU_MODE_VFIO:
    if (!ct2d->gpu_info.vfio_device || !ct2d->gpu_info.vfio_device[0]) {
      qemu_log("CXL Type2: No VFIO device specified\n");
      ct2d->gpu_info.mode = CXL_TYPE2_GPU_MODE_NONE;
      /* fall through */
    } else {
      qemu_log("CXL Type2: Initializing GPU passthrough for device %s\n",
               ct2d->gpu_info.vfio_device);
      break; /* Continue with VFIO init below */
    }
    /* fall through */

  case CXL_TYPE2_GPU_MODE_NONE:
  default:
    qemu_log("CXL Type2: GPU passthrough not configured (simulation mode)\n");
    ct2d->gpu_info.gpu_mem_base = 0;
    ct2d->gpu_info.gpu_mem_size = ct2d->device_mem_size;
    ct2d->gpu_info.passthrough_enabled = false;
    return 0;
  }

  /* VFIO passthrough initialization continues here */
  qemu_log("CXL Type2: Initializing VFIO passthrough for device %s\n",
           ct2d->gpu_info.vfio_device);

  /* Step 1: Open VFIO container */
  ret = cxl_type2_vfio_container_init(ct2d, &local_err);
  if (ret < 0) {
    error_propagate(errp, local_err);
    return -1;
  }

  /* Step 2: Setup VFIO group */
  ret = cxl_type2_vfio_group_init(ct2d, ct2d->gpu_info.vfio_device, &local_err);
  if (ret < 0) {
    error_propagate(errp, local_err);
    goto err_close_container;
  }

  /* Step 3: Get device FD and map BARs */
  ret =
      cxl_type2_vfio_device_init(ct2d, ct2d->gpu_info.vfio_device, &local_err);
  if (ret < 0) {
    error_propagate(errp, local_err);
    goto err_close_group;
  }

  /* Step 4: Setup DMA mapping for coherent access */
  ret = cxl_type2_vfio_dma_map(ct2d, &local_err);
  if (ret < 0) {
    error_propagate(errp, local_err);
    goto err_close_device;
  }

  ct2d->gpu_info.passthrough_enabled = true;

  /* Step 5: Start IRQ forwarding thread */
  ct2d->gpu_info.irq_thread_running = true;
  qemu_thread_create(&ct2d->gpu_info.irq_thread, "cxl-type2-irq",
                     cxl_type2_irq_thread, ct2d, QEMU_THREAD_JOINABLE);

  qemu_log("CXL Type2: GPU passthrough successfully initialized\n");
  qemu_log("CXL Type2: GPU memory: base=0x%lx, size=%lu MB\n",
           ct2d->gpu_info.gpu_mem_base,
           ct2d->gpu_info.gpu_mem_size / (1024 * 1024));

  return 0;

err_close_device:
  if (ct2d->gpu_info.vfio_device_fd >= 0) {
    close(ct2d->gpu_info.vfio_device_fd);
    ct2d->gpu_info.vfio_device_fd = -1;
  }

err_close_group:
  if (ct2d->gpu_info.vfio_group) {
    close(GPOINTER_TO_INT(ct2d->gpu_info.vfio_group));
    ct2d->gpu_info.vfio_group = NULL;
  }

err_close_container:
  if (ct2d->gpu_info.vfio_container) {
    close(GPOINTER_TO_INT(ct2d->gpu_info.vfio_container));
    ct2d->gpu_info.vfio_container = NULL;
  }

  ct2d->gpu_info.passthrough_enabled = false;
  return -1;
}

void cxl_type2_gpu_cleanup(CXLType2State *ct2d) {
  if (!ct2d->gpu_info.passthrough_enabled) {
    return;
  }

  /* Handle hetGPU cleanup */
  if (ct2d->gpu_info.mode == CXL_TYPE2_GPU_MODE_HETGPU) {
    cxl_type2_hetgpu_cleanup(ct2d);
    ct2d->gpu_info.passthrough_enabled = false;
    return;
  }

  /* VFIO cleanup */
  /* Stop IRQ forwarding thread */
  if (ct2d->gpu_info.irq_thread_running) {
    ct2d->gpu_info.irq_thread_running = false;
    ct2d->gpu_info.passthrough_enabled = false; /* Signal thread to exit */
    qemu_thread_join(&ct2d->gpu_info.irq_thread);
    qemu_log("CXL Type2: IRQ thread stopped\n");
  }

  /* Unmap the same BAR4 ranges that initialization admitted. */
  if (ct2d->gpu_info.vfio_container) {
    CXLType2MemoryRange ranges[2];
    size_t count = cxl_type2_vfio_dma_ranges(ct2d, ranges);

    for (size_t i = 0; i < count; i++) {
      cxl_type2_vfio_dma_unmap_range(ct2d, ranges[i].offset, ranges[i].size);
    }
  }

  /* Disable VFIO interrupts */
  if (ct2d->gpu_info.vfio_device_fd >= 0) {
    struct vfio_irq_set irq_set = {
        .argsz = sizeof(irq_set),
        .flags = VFIO_IRQ_SET_DATA_NONE | VFIO_IRQ_SET_ACTION_TRIGGER,
        .index = VFIO_PCI_MSIX_IRQ_INDEX,
        .start = 0,
        .count = 0,
    };
    ioctl(ct2d->gpu_info.vfio_device_fd, VFIO_DEVICE_SET_IRQS, &irq_set);
  }

  /* Close VFIO device */
  if (ct2d->gpu_info.vfio_device_fd >= 0) {
    close(ct2d->gpu_info.vfio_device_fd);
    ct2d->gpu_info.vfio_device_fd = -1;
  }

  /* Close VFIO group */
  if (ct2d->gpu_info.vfio_group) {
    int group_fd = GPOINTER_TO_INT(ct2d->gpu_info.vfio_group);
    close(group_fd);
    ct2d->gpu_info.vfio_group = NULL;
  }

  /* Close VFIO container */
  if (ct2d->gpu_info.vfio_container) {
    int container_fd = GPOINTER_TO_INT(ct2d->gpu_info.vfio_container);
    close(container_fd);
    ct2d->gpu_info.vfio_container = NULL;
  }

  qemu_log("CXL Type2: GPU passthrough cleanup complete\n");
}

int cxl_type2_gpu_read(CXLType2State *ct2d, uint64_t offset, void *buf,
                       size_t size) {
  ssize_t ret;

  if (!ct2d->gpu_info.passthrough_enabled) {
    return -1;
  }

  /* If VFIO device is configured, read from actual GPU BAR */
  if (ct2d->gpu_info.vfio_device_fd >= 0) {
    uint64_t bar_offset = ct2d->gpu_info.gpu_mem_base + offset;

    ret = pread(ct2d->gpu_info.vfio_device_fd, buf, size, bar_offset);
    if (ret < 0) {
      qemu_log("CXL Type2: GPU read failed at offset 0x%lx: %s\n", offset,
               strerror(errno));
      return -1;
    }

    if (ret != size) {
      qemu_log(
          "CXL Type2: GPU read incomplete at offset 0x%lx: %zd/%zu bytes\n",
          offset, ret, size);
      return -1;
    }

    ct2d->stats.gpu_accesses++;
    return 0;
  }

  /* Fallback to device memory for simulation mode */
  if (offset + size <= ct2d->device_mem_size) {
    uint8_t *mem_ptr = memory_region_get_ram_ptr(&ct2d->device_mem);
    if (mem_ptr) {
      memcpy(buf, mem_ptr + offset, size);
      ct2d->stats.gpu_accesses++;
      return 0;
    }
  }

  return -1;
}

int cxl_type2_gpu_write(CXLType2State *ct2d, uint64_t offset, const void *buf,
                        size_t size) {
  ssize_t ret;

  if (!ct2d->gpu_info.passthrough_enabled) {
    return -1;
  }

  /* If VFIO device is configured, write to actual GPU BAR */
  if (ct2d->gpu_info.vfio_device_fd >= 0) {
    uint64_t bar_offset = ct2d->gpu_info.gpu_mem_base + offset;

    ret = pwrite(ct2d->gpu_info.vfio_device_fd, buf, size, bar_offset);
    if (ret < 0) {
      qemu_log("CXL Type2: GPU write failed at offset 0x%lx: %s\n", offset,
               strerror(errno));
      return -1;
    }

    if (ret != size) {
      qemu_log(
          "CXL Type2: GPU write incomplete at offset 0x%lx: %zd/%zu bytes\n",
          offset, ret, size);
      return -1;
    }

    /* Invalidate cache line if present for coherency */
    cxl_type2_cache_invalidate(ct2d, offset);

    cxl_type2_memsim_request(ct2d, CXL_OP_WRITE, offset, size, buf, NULL);

    ct2d->stats.gpu_accesses++;
    return 0;
  }

  /* Fallback to device memory for simulation mode */
  if (offset + size <= ct2d->device_mem_size) {
    uint8_t *mem_ptr = memory_region_get_ram_ptr(&ct2d->device_mem);
    if (mem_ptr) {
      memcpy(mem_ptr + offset, buf, size);

      /* Invalidate cache line if present */
      cxl_type2_cache_invalidate(ct2d, offset);

      ct2d->stats.gpu_accesses++;
      return 0;
    }
  }

  return -1;
}

/* ========================================================================
 * CXLMemSim Communication
 * ======================================================================== */

static void cxlmemsim_drop_locked(CXLType2State *ct2d) {
  ct2d->memsim.connected = false;

  if (ct2d->memsim.socket) {
    qio_channel_close(QIO_CHANNEL(ct2d->memsim.socket), NULL);
    object_unref(OBJECT(ct2d->memsim.socket));
    ct2d->memsim.socket = NULL;
  }
}

static void cxlmemsim_connect(CXLType2State *ct2d) {
  Error *err = NULL;
  SocketAddress addr;

  if (ct2d->memsim.connected) {
    return;
  }

  /* Check if using shared memory mode */
  const char *transport_mode = getenv("CXL_TRANSPORT_MODE");
  if (!transport_mode || !transport_mode[0]) {
    transport_mode = getenv("CXL_MEMSIM_TRANSPORT");
  }

  qemu_log("CXL Type2: Transport mode = %s\n",
           transport_mode ? transport_mode : "(not set)");

  if (transport_mode && (strcmp(transport_mode, "shm") == 0 ||
                         strcmp(transport_mode, "pgas") == 0)) {
    ct2d->memsim.use_shm = true;
    qemu_log(
        "CXL Type2: Using shared memory transport - skipping TCP connection\n");
    /* Type3 device handles SHM connection */
    return;
  }

  qemu_log("CXL Type2: Using TCP transport mode\n");

  /* TCP connection to CXLMemSim */
  addr.type = SOCKET_ADDRESS_TYPE_INET;
  addr.u.inet.host = ct2d->memsim.server_addr;
  addr.u.inet.port = g_strdup_printf("%u", ct2d->memsim.server_port);

  ct2d->memsim.socket = qio_channel_socket_new();
  if (qio_channel_socket_connect_sync(ct2d->memsim.socket, &addr, &err) < 0) {
    qemu_log("Warning: Failed to connect to CXLMemSim at %s:%s: %s\n",
             addr.u.inet.host, addr.u.inet.port, error_get_pretty(err));
    error_free(err);
    g_free(addr.u.inet.port);
    object_unref(OBJECT(ct2d->memsim.socket));
    ct2d->memsim.socket = NULL;
    return;
  }

  ct2d->memsim.connected = true;
  g_free(addr.u.inet.port);

  qemu_log("CXL Type2: Connected to CXLMemSim at %s:%u\n",
           ct2d->memsim.server_addr, ct2d->memsim.server_port);
}

static void cxlmemsim_disconnect(CXLType2State *ct2d) {
  qemu_mutex_lock(&ct2d->memsim.lock);

  cxlmemsim_drop_locked(ct2d);

  if (ct2d->memsim.use_shm && ct2d->memsim.shm_base) {
    munmap(ct2d->memsim.shm_base, ct2d->memsim.shm_size);
    ct2d->memsim.shm_base = NULL;
  }

  qemu_mutex_unlock(&ct2d->memsim.lock);
}

static bool cxl_type2_memsim_read_response(CXLType2State *ct2d,
                                           const CXLMemSimRequest *req,
                                           CXLMemSimResponse *resp,
                                           Error **errp) {
  QIOChannelSocket *socket = ct2d->memsim.socket;
  uint8_t *cursor = (uint8_t *)resp;
  size_t remaining = sizeof(*resp);
  int64_t deadline =
      g_get_monotonic_time() * 1000 + CXL_MEMSIM_RESPONSE_TIMEOUT_NS;

  while (remaining > 0) {
    ssize_t received = recv(socket->fd, cursor, remaining, MSG_DONTWAIT);
    if (received > 0) {
      cursor += received;
      remaining -= received;
      continue;
    }
    if (received == 0) {
      error_setg(errp,
                 "CXLMemSim closed response for op=%u addr=0x%" PRIx64
                 " size=%" PRIu64,
                 req->op_type, req->addr, req->size);
      return false;
    }
    if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
      error_setg_errno(
          errp, errno,
          "CXLMemSim response read failed for op=%u addr=0x%" PRIx64
          " size=%" PRIu64,
          req->op_type, req->addr, req->size);
      return false;
    }

    int64_t remaining_ns = deadline - g_get_monotonic_time() * 1000;
    if (remaining_ns <= 0) {
      error_setg(errp,
                 "timed out waiting for CXLMemSim response for op=%u"
                 " addr=0x%" PRIx64 " size=%" PRIu64,
                 req->op_type, req->addr, req->size);
      return false;
    }

    GPollFD poll_fd = {
        .fd = socket->fd,
        .events = G_IO_IN | G_IO_HUP | G_IO_ERR,
        .revents = 0,
    };
    int poll_result = qemu_poll_ns(&poll_fd, 1, remaining_ns);
    if (poll_result == 0) {
      error_setg(errp,
                 "timed out waiting for CXLMemSim response for op=%u"
                 " addr=0x%" PRIx64 " size=%" PRIu64,
                 req->op_type, req->addr, req->size);
      return false;
    }
    if (poll_result < 0) {
      if (errno == EINTR) {
        continue;
      }
      error_setg_errno(
          errp, errno,
          "CXLMemSim response poll failed for op=%u addr=0x%" PRIx64
          " size=%" PRIu64,
          req->op_type, req->addr, req->size);
      return false;
    }
    if (poll_fd.revents & (G_IO_HUP | G_IO_ERR | G_IO_NVAL)) {
      error_setg(errp,
                 "CXLMemSim response channel failed for op=%u addr=0x%" PRIx64
                 " size=%" PRIu64,
                 req->op_type, req->addr, req->size);
      return false;
    }
  }

  return true;
}

static void
cxl_type2_record_memsim_request(CXLType2State *ct2d, uint8_t op_type,
                                uint64_t logical_bytes, int64_t begin_host_ns,
                                int64_t end_host_ns, bool response_received,
                                const CXLMemSimResponse *response) {
  uint64_t qemu_wall_ns = end_host_ns >= begin_host_ns
                              ? (uint64_t)(end_host_ns - begin_host_ns)
                              : 0;

  if (ct2d->paired_case.active_case == CXL_GPU_CASE_NONE) {
    return;
  }

  ct2d->paired_case.active_cxl_request_count++;
  if (op_type == CXL_OP_READ || op_type == CXL_OP_RANGE_READ) {
    ct2d->paired_case.active_cxl_read_count++;
  } else if (op_type == CXL_OP_WRITE) {
    ct2d->paired_case.active_cxl_write_count++;
  } else if (op_type == CXL_OP_RANGE_WRITE) {
    ct2d->paired_case.active_cxl_write_count++;
  }
  ct2d->paired_case.active_cxl_logical_bytes += logical_bytes;
  ct2d->paired_case.active_cxl_qemu_wall_ns += qemu_wall_ns;
  ct2d->paired_case.active_cxl_wire_bytes += sizeof(CXLMemSimRequest);
  if (op_type == CXL_OP_RANGE_READ || op_type == CXL_OP_RANGE_WRITE) {
    ct2d->paired_case.active_cxl_range_requests++;
    ct2d->paired_case.active_cxl_range_bytes += logical_bytes;
  }
  if (response_received && response) {
    ct2d->paired_case.active_cxl_response_count++;
    ct2d->paired_case.active_cxl_wire_bytes += sizeof(CXLMemSimResponse);
    ct2d->paired_case.active_cxl_server_reported_latency_ns +=
        response->latency_ns;
    if (response->status != 0) {
      ct2d->paired_case.active_cxl_request_failures++;
    }
  } else {
    ct2d->paired_case.active_cxl_request_failures++;
  }
}

static bool cxl_type2_memsim_request_ext(CXLType2State *ct2d, uint8_t op_type,
                                         uint64_t addr, uint64_t size,
                                         const uint8_t *data, uint64_t value,
                                         uint64_t expected,
                                         CXLMemSimResponse *resp) {
  CXLMemSimRequest req;
  CXLMemSimResponse local_resp;
  Error *err = NULL;
  bool ok = false;
  bool request_sent = false;
  bool response_received = false;
  int64_t request_begin_host_ns = 0;
  int64_t request_end_host_ns = 0;

  if (!ct2d->memsim.connected || ct2d->memsim.use_shm) {
    return false;
  }

  memset(&req, 0, sizeof(req));
  memset(&local_resp, 0, sizeof(local_resp));
  req.op_type = op_type;
  req.addr = addr;
  req.size = size;
  req.timestamp = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
  req.value = value;
  req.expected = expected;
  if (data) {
    memcpy(req.data, data, MIN(size, sizeof(req.data)));
  }

  qemu_mutex_lock(&ct2d->memsim.lock);
  if (!ct2d->memsim.connected || !ct2d->memsim.socket) {
    goto out;
  }

  request_begin_host_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
  if (qio_channel_write_all(QIO_CHANNEL(ct2d->memsim.socket),
                            (const char *)&req, sizeof(req), &err) < 0) {
    error_report("CXL Type2: Failed to send request to CXLMemSim: %s",
                 error_get_pretty(err));
    error_free(err);
    cxlmemsim_drop_locked(ct2d);
    goto out;
  }
  request_sent = true;

  if (!cxl_type2_memsim_read_response(ct2d, &req, &local_resp, &err)) {
    error_report("CXL Type2: Failed to receive response from CXLMemSim: %s",
                 error_get_pretty(err));
    error_free(err);
    cxlmemsim_drop_locked(ct2d);
    goto out;
  }
  response_received = true;
  request_end_host_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);

  if (resp) {
    *resp = local_resp;
  }
  ok = local_resp.status == 0;

out:
  if (request_sent) {
    if (request_end_host_ns == 0) {
      request_end_host_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
    }
    cxl_type2_record_memsim_request(ct2d, op_type, size, request_begin_host_ns,
                                    request_end_host_ns, response_received,
                                    response_received ? &local_resp : NULL);
  }
  qemu_mutex_unlock(&ct2d->memsim.lock);
  return ok;
}

static bool cxl_type2_memsim_request(CXLType2State *ct2d, uint8_t op_type,
                                     uint64_t addr, uint64_t size,
                                     const uint8_t *data,
                                     CXLMemSimResponse *resp) {
  return cxl_type2_memsim_request_ext(ct2d, op_type, addr, size, data, 0, 0,
                                      resp);
}

bool cxl_type2_model_aperture_access(CXLType2State *ct2d, uint64_t offset,
                                     uint64_t size, bool is_write,
                                     uint64_t *latency_ns,
                                     const char **reason) {
  CXLMemSimResponse response;

  if (latency_ns) {
    *latency_ns = 0;
  }
  if (reason) {
    *reason = NULL;
  }
  if (!ct2d || !latency_ns || !reason || !size || size > 64) {
    if (reason) {
      *reason = "model-aperture-request-invalid";
    }
    return false;
  }
  if (!cxl_type2_range_contains(ct2d->model_aperture.offset,
                                ct2d->model_aperture.size, offset, size)) {
    *reason = "model-aperture-range-rejected";
    return false;
  }
  if (is_write) {
    *reason = "model-aperture-readonly";
    return false;
  }
  if (!cxl_type2_fabric_access_allowed(ct2d, offset, size, false, false)) {
    *reason = "model-aperture-fabric-rejected";
    return false;
  }
  if (!cxl_type2_memsim_request(ct2d, CXL_OP_READ, offset, size, NULL,
                                &response)) {
    *reason = "model-aperture-cxl-read-failed";
    return false;
  }
  *latency_ns = response.latency_ns;
  return true;
}

/* ========================================================================
 * Memory Access Handlers with Coherency
 * ======================================================================== */

/* Forward declaration for GPU command handler */
static uint64_t cxl_type2_gpu_cmd_read(void *opaque, hwaddr addr,
                                       unsigned size);
static void cxl_type2_gpu_cmd_write(void *opaque, hwaddr addr, uint64_t value,
                                    unsigned size);

static uint64_t cxl_type2_cache_read(void *opaque, hwaddr addr, unsigned size) {
  CXLType2State *ct2d = opaque;
  CXLCacheLine *line;
  uint64_t value = 0;
  uint64_t cache_line_addr = addr & ~0x3F;
  size_t offset = addr & 0x3F;

  /* Check if this is a GPU command register access */
  if (addr < CXL_GPU_CMD_REG_SIZE) {
    return cxl_type2_gpu_cmd_read(opaque, addr, size);
  }

  ct2d->stats.cpu_accesses++;

  /* Track with enhanced BAR coherency */
  if (ct2d->bar_coherency.enabled) {
    cxl_bar_coherency_request(&ct2d->bar_coherency, CXL_COH_REQ_RD_SHARED, addr,
                              size, CXL_DOMAIN_CPU, NULL);
  }

  /* Check coherency protocol */
  line = cxl_type2_cache_lookup(ct2d, addr);

  if (line && line->state != CXL_COHERENCY_INVALID) {
    /* Cache hit */
    memcpy(&value, &line->data[offset], MIN(size, 64 - offset));

    qemu_log_mask(LOG_TRACE, "CXL Type2: Cache read hit at 0x%lx = 0x%lx\n",
                  addr, value);
  } else {
    /* Cache miss - fetch from device memory */
    uint8_t *mem_ptr = memory_region_get_ram_ptr(&ct2d->device_mem);
    if (mem_ptr && addr < ct2d->device_mem_size) {
      memcpy(&value, mem_ptr + addr, size);

      /* Insert into cache */
      uint8_t cache_data[64];
      memcpy(cache_data, mem_ptr + cache_line_addr, 64);
      cxl_type2_cache_insert(ct2d, addr, cache_data, CXL_COHERENCY_SHARED);
    }

    cxl_type2_memsim_request(ct2d, CXL_OP_READ, addr, size, NULL, NULL);

    qemu_log_mask(LOG_TRACE, "CXL Type2: Cache read miss at 0x%lx = 0x%lx\n",
                  addr, value);
  }

  ct2d->stats.read_ops++;
  return value;
}

static void cxl_type2_cache_write(void *opaque, hwaddr addr, uint64_t value,
                                  unsigned size) {
  CXLType2State *ct2d = opaque;
  CXLCacheLine *line;
  uint64_t cache_line_addr = addr & ~0x3F;
  size_t offset = addr & 0x3F;

  /* Check if this is a GPU command register access */
  if (addr < CXL_GPU_CMD_REG_SIZE) {
    cxl_type2_gpu_cmd_write(opaque, addr, value, size);
    return;
  }

  ct2d->stats.cpu_accesses++;
  ct2d->stats.write_ops++;

  /* Track with enhanced BAR coherency - write requires exclusive access */
  if (ct2d->bar_coherency.enabled) {
    cxl_bar_coherency_request(&ct2d->bar_coherency, CXL_COH_REQ_WR_INV, addr,
                              size, CXL_DOMAIN_CPU, NULL);
  }

  /* Check if we have the cache line */
  line = cxl_type2_cache_lookup(ct2d, addr);

  if (!line || line->state == CXL_COHERENCY_INVALID) {
    /* Need to fetch cache line first */
    uint8_t *mem_ptr = memory_region_get_ram_ptr(&ct2d->device_mem);
    if (mem_ptr && cache_line_addr < ct2d->device_mem_size) {
      uint8_t cache_data[64];
      memcpy(cache_data, mem_ptr + cache_line_addr, 64);
      cxl_type2_cache_insert(ct2d, addr, cache_data, CXL_COHERENCY_MODIFIED);
      line = cxl_type2_cache_lookup(ct2d, addr);
    }
  }

  if (line) {
    /* Update cache line */
    qemu_mutex_lock(&ct2d->coherency.lock);
    memcpy(&line->data[offset], &value, MIN(size, 64 - offset));
    line->state = CXL_COHERENCY_MODIFIED;
    line->dirty = true;
    line->timestamp = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
    qemu_mutex_unlock(&ct2d->coherency.lock);

    /* Write through to device memory */
    uint8_t *mem_ptr = memory_region_get_ram_ptr(&ct2d->device_mem);
    if (mem_ptr && addr < ct2d->device_mem_size) {
      memcpy(mem_ptr + addr, &value, size);
    }

    cxl_type2_memsim_request(ct2d, CXL_OP_WRITE, addr, size,
                             (const uint8_t *)&value, NULL);
  }
}

static uint64_t cxl_type2_device_mem_read(void *opaque, hwaddr addr,
                                          unsigned size) {
  CXLType2State *ct2d = opaque;

  if (!cxl_type2_fabric_access_allowed(ct2d, addr, size, false, false)) {
    return 0;
  }

  /* Forward all device memory reads through the cache coherency layer */
  return cxl_type2_cache_read(opaque, addr, size);
}

static void cxl_type2_device_mem_write(void *opaque, hwaddr addr,
                                       uint64_t value, unsigned size) {
  CXLType2State *ct2d = opaque;

  if (!cxl_type2_fabric_access_allowed(ct2d, addr, size, true, false)) {
    return;
  }

  /* Forward all device memory writes through the cache coherency layer */
  cxl_type2_cache_write(opaque, addr, value, size);
}

static const MemoryRegionOps cxl_type2_cache_ops = {
    .read = cxl_type2_cache_read,
    .write = cxl_type2_cache_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid =
        {
            .min_access_size = 1,
            .max_access_size = 8,
        },
    .impl =
        {
            .min_access_size = 1,
            .max_access_size = 8,
        },
};

static const MemoryRegionOps cxl_type2_device_mem_ops = {
    .read = cxl_type2_device_mem_read,
    .write = cxl_type2_device_mem_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid =
        {
            .min_access_size = 1,
            .max_access_size = 8,
        },
    .impl =
        {
            .min_access_size = 1,
            .max_access_size = 8,
        },
};

/* ========================================================================
 * DCD / GFAM / MH-SLD Fabric Memory Models
 * ======================================================================== */

static uint64_t cxl_type2_dcd_align(CXLType2State *ct2d, uint64_t value) {
  uint64_t granularity = ct2d->dcd.granularity;

  if (!granularity) {
    granularity = CXL_TYPE2_DCD_DEFAULT_GRANULARITY;
  }

  return (value + granularity - 1) & ~(granularity - 1);
}

static bool cxl_type2_range_valid(CXLType2State *ct2d, uint64_t base,
                                  uint64_t size) {
  return size && base < ct2d->device_mem_size &&
         size <= ct2d->device_mem_size - base;
}

static bool cxl_type2_dcd_overlaps_locked(CXLType2State *ct2d, uint64_t base,
                                          uint64_t size) {
  uint64_t end = base + size;

  for (int i = 0; i < CXL_TYPE2_MAX_DCD_EXTENTS; i++) {
    CXLType2DCDExtent *extent = &ct2d->dcd.extents[i];
    uint64_t extent_end;

    if (!extent->active) {
      continue;
    }

    extent_end = extent->base + extent->size;
    if (base < extent_end && extent->base < end) {
      return true;
    }
  }

  return false;
}

static bool cxl_type2_dcd_allocated_locked(CXLType2State *ct2d, uint64_t base,
                                           uint64_t size) {
  uint64_t end = base + size;

  if (!ct2d->dcd.enabled) {
    return true;
  }

  for (int i = 0; i < CXL_TYPE2_MAX_DCD_EXTENTS; i++) {
    CXLType2DCDExtent *extent = &ct2d->dcd.extents[i];

    if (!extent->active) {
      continue;
    }
    if (base >= extent->base && end <= extent->base + extent->size) {
      return true;
    }
  }

  return false;
}

static bool cxl_type2_dcd_allocated(CXLType2State *ct2d, uint64_t base,
                                    uint64_t size) {
  bool allocated;

  qemu_mutex_lock(&ct2d->dcd.lock);
  allocated = cxl_type2_dcd_allocated_locked(ct2d, base, size);
  qemu_mutex_unlock(&ct2d->dcd.lock);

  return allocated;
}

static uint64_t cxl_type2_dcd_active_extents_locked(CXLType2State *ct2d) {
  uint64_t count = 0;

  for (int i = 0; i < CXL_TYPE2_MAX_DCD_EXTENTS; i++) {
    if (ct2d->dcd.extents[i].active) {
      count++;
    }
  }

  return count;
}

static bool cxl_type2_dcd_find_free_locked(CXLType2State *ct2d, uint64_t size,
                                           uint64_t *base) {
  uint64_t candidate = 0;

  while (cxl_type2_range_valid(ct2d, candidate, size)) {
    bool moved = false;

    for (int i = 0; i < CXL_TYPE2_MAX_DCD_EXTENTS; i++) {
      CXLType2DCDExtent *extent = &ct2d->dcd.extents[i];

      if (!extent->active) {
        continue;
      }
      if (candidate < extent->base + extent->size &&
          extent->base < candidate + size) {
        candidate = cxl_type2_dcd_align(ct2d, extent->base + extent->size);
        moved = true;
        break;
      }
    }
    if (!moved) {
      *base = candidate;
      return true;
    }
  }

  return false;
}

static int cxl_type2_dcd_add(CXLType2State *ct2d, uint64_t requested_base,
                             uint64_t size, uint64_t tag, uint64_t *out_base,
                             uint64_t *out_size, uint64_t *out_tag) {
  uint64_t base = requested_base;
  uint64_t aligned_size = cxl_type2_dcd_align(ct2d, size);
  int slot = -1;

  if (!ct2d->dcd.enabled) {
    return -EOPNOTSUPP;
  }

  qemu_mutex_lock(&ct2d->dcd.lock);
  ct2d->dcd.add_requests++;

  if (!aligned_size || aligned_size > ct2d->device_mem_size) {
    ct2d->dcd.failed_requests++;
    qemu_mutex_unlock(&ct2d->dcd.lock);
    return -EINVAL;
  }

  if (requested_base == UINT64_MAX) {
    if (!cxl_type2_dcd_find_free_locked(ct2d, aligned_size, &base)) {
      ct2d->dcd.failed_requests++;
      qemu_mutex_unlock(&ct2d->dcd.lock);
      return -ENOSPC;
    }
  } else if (base != cxl_type2_dcd_align(ct2d, base)) {
    ct2d->dcd.failed_requests++;
    qemu_mutex_unlock(&ct2d->dcd.lock);
    return -EINVAL;
  }

  if (!cxl_type2_range_valid(ct2d, base, aligned_size) ||
      cxl_type2_dcd_overlaps_locked(ct2d, base, aligned_size)) {
    ct2d->dcd.failed_requests++;
    qemu_mutex_unlock(&ct2d->dcd.lock);
    return -ENOSPC;
  }

  for (int i = 0; i < CXL_TYPE2_MAX_DCD_EXTENTS; i++) {
    if (!ct2d->dcd.extents[i].active) {
      slot = i;
      break;
    }
  }
  if (slot < 0) {
    ct2d->dcd.failed_requests++;
    qemu_mutex_unlock(&ct2d->dcd.lock);
    return -ENOSPC;
  }

  if (!tag) {
    tag = ct2d->dcd.next_tag++;
  }

  ct2d->dcd.extents[slot] = (CXLType2DCDExtent){
      .base = base,
      .size = aligned_size,
      .tag = tag,
      .active = true,
  };
  ct2d->dcd.allocated += aligned_size;

  if (out_base) {
    *out_base = base;
  }
  if (out_size) {
    *out_size = aligned_size;
  }
  if (out_tag) {
    *out_tag = tag;
  }

  qemu_mutex_unlock(&ct2d->dcd.lock);

  qemu_log("CXL Type2 DCD: add base=0x%" PRIx64 " size=0x%" PRIx64
           " tag=%" PRIu64 "\n",
           base, aligned_size, tag);
  return 0;
}

static int cxl_type2_dcd_release(CXLType2State *ct2d, uint64_t base,
                                 uint64_t size, uint64_t tag) {
  uint64_t aligned_size = cxl_type2_dcd_align(ct2d, size);

  if (!ct2d->dcd.enabled) {
    return -EOPNOTSUPP;
  }

  qemu_mutex_lock(&ct2d->dcd.lock);
  ct2d->dcd.release_requests++;

  for (int i = 0; i < CXL_TYPE2_MAX_DCD_EXTENTS; i++) {
    CXLType2DCDExtent *extent = &ct2d->dcd.extents[i];

    if (!extent->active || extent->base != base ||
        extent->size != aligned_size) {
      continue;
    }
    if (tag && extent->tag != tag) {
      continue;
    }

    extent->active = false;
    ct2d->dcd.allocated -= extent->size;
    qemu_mutex_unlock(&ct2d->dcd.lock);

    qemu_log("CXL Type2 DCD: release base=0x%" PRIx64 " size=0x%" PRIx64
             " tag=%" PRIu64 "\n",
             base, aligned_size, tag);
    return 0;
  }

  ct2d->dcd.failed_requests++;
  qemu_mutex_unlock(&ct2d->dcd.lock);
  return -ENOENT;
}

static void cxl_type2_gfam_revoke_range(CXLType2State *ct2d, uint64_t base,
                                        uint64_t size) {
  uint64_t end = base + size;

  qemu_mutex_lock(&ct2d->gfam.lock);
  for (int i = 0; i < CXL_TYPE2_MAX_GFAM_MAPPINGS; i++) {
    CXLType2GFAMMapping *mapping = &ct2d->gfam.mappings[i];
    uint64_t mapping_end;

    if (!mapping->active) {
      continue;
    }

    mapping_end = mapping->base + mapping->size;
    if (base < mapping_end && mapping->base < end) {
      mapping->active = false;
    }
  }
  qemu_mutex_unlock(&ct2d->gfam.lock);
}

static int cxl_type2_gfam_grant(CXLType2State *ct2d, uint32_t host_id,
                                uint64_t base, uint64_t size,
                                uint32_t permissions) {
  int slot = -1;

  if (!ct2d->gfam.enabled) {
    return -EOPNOTSUPP;
  }
  if (host_id >= ct2d->gfam.num_hosts || !permissions ||
      !cxl_type2_dcd_allocated(ct2d, base, size)) {
    return -EINVAL;
  }

  qemu_mutex_lock(&ct2d->gfam.lock);
  for (int i = 0; i < CXL_TYPE2_MAX_GFAM_MAPPINGS; i++) {
    CXLType2GFAMMapping *mapping = &ct2d->gfam.mappings[i];

    if (mapping->active && mapping->host_id == host_id &&
        mapping->base == base && mapping->size == size) {
      mapping->permissions = permissions;
      qemu_mutex_unlock(&ct2d->gfam.lock);
      return 0;
    }
    if (!mapping->active && slot < 0) {
      slot = i;
    }
  }

  if (slot < 0) {
    qemu_mutex_unlock(&ct2d->gfam.lock);
    return -ENOSPC;
  }

  ct2d->gfam.mappings[slot] = (CXLType2GFAMMapping){
      .host_id = host_id,
      .permissions = permissions,
      .base = base,
      .size = size,
      .active = true,
  };

  qemu_mutex_unlock(&ct2d->gfam.lock);
  return 0;
}

static int cxl_type2_gfam_revoke(CXLType2State *ct2d, uint32_t host_id,
                                 uint64_t base, uint64_t size) {
  if (!ct2d->gfam.enabled) {
    return -EOPNOTSUPP;
  }

  qemu_mutex_lock(&ct2d->gfam.lock);
  for (int i = 0; i < CXL_TYPE2_MAX_GFAM_MAPPINGS; i++) {
    CXLType2GFAMMapping *mapping = &ct2d->gfam.mappings[i];

    if (mapping->active && mapping->host_id == host_id &&
        mapping->base == base && mapping->size == size) {
      mapping->active = false;
      qemu_mutex_unlock(&ct2d->gfam.lock);
      return 0;
    }
  }
  qemu_mutex_unlock(&ct2d->gfam.lock);
  return -ENOENT;
}

static uint64_t cxl_type2_gfam_active_mappings_locked(CXLType2State *ct2d) {
  uint64_t count = 0;

  for (int i = 0; i < CXL_TYPE2_MAX_GFAM_MAPPINGS; i++) {
    if (ct2d->gfam.mappings[i].active) {
      count++;
    }
  }

  return count;
}

static bool cxl_type2_gfam_allowed(CXLType2State *ct2d, uint32_t host_id,
                                   uint64_t base, uint64_t size, bool is_write,
                                   bool is_atomic) {
  uint32_t required = is_write ? CXL_DCD_PERM_WRITE : CXL_DCD_PERM_READ;
  uint64_t end = base + size;

  if (!ct2d->gfam.enabled) {
    return true;
  }
  if (is_atomic) {
    required |= CXL_DCD_PERM_ATOMIC;
  }

  qemu_mutex_lock(&ct2d->gfam.lock);
  for (int i = 0; i < CXL_TYPE2_MAX_GFAM_MAPPINGS; i++) {
    CXLType2GFAMMapping *mapping = &ct2d->gfam.mappings[i];

    if (!mapping->active || mapping->host_id != host_id) {
      continue;
    }
    if (base >= mapping->base && end <= mapping->base + mapping->size &&
        (mapping->permissions & required) == required) {
      ct2d->gfam.allowed_accesses++;
      ct2d->gfam.total_latency_ns += ct2d->gfam.fabric_latency_ns;
      qemu_mutex_unlock(&ct2d->gfam.lock);
      return true;
    }
  }

  ct2d->gfam.denied_accesses++;
  qemu_mutex_unlock(&ct2d->gfam.lock);
  return false;
}

static uint32_t cxl_type2_popcount32(uint32_t value) {
  uint32_t count = 0;

  while (value) {
    count += value & 1;
    value >>= 1;
  }

  return count;
}

static void cxl_type2_mhsld_record(CXLType2State *ct2d, uint64_t addr,
                                   bool is_write, bool is_atomic) {
  CXLType2MHSLDLine *line;
  uint64_t line_addr;
  uint32_t head;
  uint32_t head_mask;
  uint32_t other_sharers;
  uint32_t index;

  if (!ct2d->mhsld.enabled) {
    return;
  }

  line_addr = addr & CXL_CACHE_LINE_MASK;
  index = (line_addr >> 6) % CXL_TYPE2_MAX_MHSLD_LINES;
  head = ct2d->mhsld.local_head_id;
  head_mask = head < 32 ? 1u << head : 0;

  qemu_mutex_lock(&ct2d->mhsld.lock);
  line = &ct2d->mhsld.lines[index];
  if (!line->valid || line->line_addr != line_addr) {
    *line = (CXLType2MHSLDLine){
        .line_addr = line_addr,
        .owner_head = head,
        .sharer_mask = head_mask,
        .valid = true,
        .modified = false,
    };
  }

  if (is_write || is_atomic) {
    other_sharers = line->sharer_mask & ~head_mask;
    if ((line->modified && line->owner_head != head) || other_sharers) {
      ct2d->mhsld.conflicts++;
      ct2d->mhsld.invalidations += cxl_type2_popcount32(other_sharers);
    }
    line->owner_head = head;
    line->sharer_mask = head_mask;
    line->modified = true;
    ct2d->mhsld.writes++;
    if (is_atomic) {
      ct2d->mhsld.atomics++;
    }
  } else {
    if (line->modified && line->owner_head != head) {
      ct2d->mhsld.conflicts++;
      line->modified = false;
    }
    line->sharer_mask |= head_mask;
    ct2d->mhsld.reads++;
  }

  qemu_mutex_unlock(&ct2d->mhsld.lock);
}

static bool cxl_type2_fabric_access_allowed(CXLType2State *ct2d, uint64_t addr,
                                            uint64_t size, bool is_write,
                                            bool is_atomic) {
  if (!cxl_type2_range_valid(ct2d, addr, size)) {
    return false;
  }

  if (!cxl_type2_dcd_allocated(ct2d, addr, size)) {
    qemu_log_mask(LOG_GUEST_ERROR,
                  "CXL Type2 DCD: access outside allocated capacity "
                  "addr=0x%" PRIx64 " size=0x%" PRIx64 "\n",
                  addr, size);
    return false;
  }

  if (!cxl_type2_gfam_allowed(ct2d, ct2d->gfam.local_host_id, addr, size,
                              is_write, is_atomic)) {
    qemu_log_mask(LOG_GUEST_ERROR,
                  "CXL Type2 GFAM: denied host=%u addr=0x%" PRIx64
                  " size=0x%" PRIx64 " write=%d\n",
                  ct2d->gfam.local_host_id, addr, size, is_write);
    return false;
  }

  cxl_type2_mhsld_record(ct2d, addr, is_write, is_atomic);
  return true;
}

static void cxl_type2_fabric_features_init(CXLType2State *ct2d) {
  bool use_dcd = ct2d->dcd.enabled || ct2d->gfam.enabled;
  uint64_t initial_size;

  qemu_mutex_init(&ct2d->dcd.lock);
  qemu_mutex_init(&ct2d->gfam.lock);
  qemu_mutex_init(&ct2d->mhsld.lock);

  if (!ct2d->dcd.granularity ||
      (ct2d->dcd.granularity & (ct2d->dcd.granularity - 1))) {
    ct2d->dcd.granularity = CXL_TYPE2_DCD_DEFAULT_GRANULARITY;
  }
  ct2d->dcd.next_tag = 1;

  if (ct2d->gfam.num_hosts == 0) {
    ct2d->gfam.num_hosts = 1;
  }
  if (ct2d->gfam.num_hosts > CXL_TYPE2_MAX_GFAM_HOSTS) {
    ct2d->gfam.num_hosts = CXL_TYPE2_MAX_GFAM_HOSTS;
  }
  if (ct2d->gfam.local_host_id >= ct2d->gfam.num_hosts) {
    ct2d->gfam.local_host_id = 0;
  }
  if (!ct2d->gfam.default_permissions) {
    ct2d->gfam.default_permissions = CXL_DCD_PERM_ALL;
  }
  if (!ct2d->gfam.fabric_latency_ns) {
    ct2d->gfam.fabric_latency_ns = 150;
  }
  if (!ct2d->gfam.bandwidth_mbps) {
    ct2d->gfam.bandwidth_mbps = 32768;
  }

  if (!ct2d->mhsld.num_heads) {
    ct2d->mhsld.num_heads = ct2d->gfam.num_hosts;
  }
  if (ct2d->mhsld.num_heads > 32) {
    ct2d->mhsld.num_heads = 32;
  }
  if (ct2d->mhsld.local_head_id >= ct2d->mhsld.num_heads) {
    ct2d->mhsld.local_head_id = 0;
  }
  if (!ct2d->mhsld.coherency_latency_ns) {
    ct2d->mhsld.coherency_latency_ns = 200;
  }

  if (!use_dcd) {
    return;
  }

  ct2d->dcd.enabled = true;
  initial_size = ct2d->dcd.initial_size;
  if (initial_size == CXL_TYPE2_DCD_INIT_AUTO) {
    initial_size = ct2d->device_mem_size;
  }
  if (initial_size) {
    uint64_t base;
    uint64_t size;
    uint64_t tag;

    if (cxl_type2_dcd_add(ct2d, 0, initial_size, 1, &base, &size, &tag) == 0 &&
        ct2d->gfam.enabled) {
      for (uint32_t host = 0; host < ct2d->gfam.num_hosts; host++) {
        cxl_type2_gfam_grant(ct2d, host, base, size,
                             ct2d->gfam.default_permissions);
      }
    }
  }

  qemu_log("CXL Type2 fabric: DCD=%d GFAM=%d MHSLD=%d initial=%" PRIu64
           "MB hosts=%u heads=%u\n",
           ct2d->dcd.enabled, ct2d->gfam.enabled, ct2d->mhsld.enabled,
           initial_size / MiB, ct2d->gfam.num_hosts, ct2d->mhsld.num_heads);
}

static void cxl_type2_fabric_features_cleanup(CXLType2State *ct2d) {
  qemu_mutex_destroy(&ct2d->mhsld.lock);
  qemu_mutex_destroy(&ct2d->gfam.lock);
  qemu_mutex_destroy(&ct2d->dcd.lock);
}

/* ========================================================================
 * Coherent Pool Allocator
 * ======================================================================== */

/* First-fit allocator from free list, page-aligned (4KB) */
static int64_t cxl_coherent_pool_alloc(CXLType2State *ct2d, uint64_t size) {
  uint64_t aligned_size = (size + 0xFFF) & ~0xFFFULL; /* 4KB page align */
  CXLCohFreeBlock **prev = &ct2d->coherent_pool.free_list;
  CXLCohFreeBlock *blk = ct2d->coherent_pool.free_list;

  qemu_mutex_lock(&ct2d->coherent_pool.lock);

  while (blk) {
    if (blk->size >= aligned_size) {
      uint64_t alloc_offset = blk->offset;

      if (blk->size == aligned_size) {
        /* Exact fit - remove block */
        *prev = blk->next;
        g_free(blk);
      } else {
        /* Split block */
        blk->offset += aligned_size;
        blk->size -= aligned_size;
      }

      /* Track allocation */
      uint64_t *key = g_new(uint64_t, 1);
      CXLCohAllocation *val = g_new0(CXLCohAllocation, 1);
      *key = alloc_offset;
      val->size = aligned_size;
      g_hash_table_insert(ct2d->coherent_pool.allocations, key, val);
      ct2d->coherent_pool.used += aligned_size;

      qemu_mutex_unlock(&ct2d->coherent_pool.lock);

      qemu_log("CXL Type2: Coherent pool alloc: offset=0x%lx size=%lu\n",
               (unsigned long)alloc_offset, (unsigned long)aligned_size);
      return (int64_t)alloc_offset;
    }
    prev = &blk->next;
    blk = blk->next;
  }

  qemu_mutex_unlock(&ct2d->coherent_pool.lock);
  qemu_log("CXL Type2: Coherent pool alloc FAILED: size=%lu (used=%lu/%lu)\n",
           (unsigned long)size, (unsigned long)ct2d->coherent_pool.used,
           (unsigned long)ct2d->coherent_pool.size);
  return -1;
}

/* Free allocation and coalesce with adjacent free blocks */
static int cxl_coherent_pool_free(CXLType2State *ct2d, uint64_t offset) {
  qemu_mutex_lock(&ct2d->coherent_pool.lock);

  CXLCohAllocation *allocation =
      g_hash_table_lookup(ct2d->coherent_pool.allocations, &offset);
  if (!allocation || allocation->host_registered) {
    qemu_mutex_unlock(&ct2d->coherent_pool.lock);
    qemu_log("CXL Type2: Coherent pool free FAILED: offset=0x%lx"
             " missing-or-mapped\n",
             (unsigned long)offset);
    return -1;
  }

  uint64_t size = allocation->size;
  bool retire = allocation->ever_mapped;
  g_hash_table_remove(ct2d->coherent_pool.allocations, &offset);
  if (retire) {
    ct2d->coherent_pool.retired_bytes += size;
    ct2d->coherent_pool.retired_allocations++;
    qemu_mutex_unlock(&ct2d->coherent_pool.lock);
    qemu_log("CXL Type2: Coherent pool retire: offset=0x%lx size=%lu"
             " retired_bytes=%lu\n",
             (unsigned long)offset, (unsigned long)size,
             (unsigned long)ct2d->coherent_pool.retired_bytes);
    qemu_log("cxl_coherent_device_retire status=0 offset=0x%" PRIx64
             " bytes=%" PRIu64 " retired_bytes=%" PRIu64 "\n",
             offset, size, ct2d->coherent_pool.retired_bytes);
    return 0;
  }
  ct2d->coherent_pool.used -= size;

  /* Insert back into sorted free list with coalescing */
  CXLCohFreeBlock *new_blk = g_new0(CXLCohFreeBlock, 1);
  new_blk->offset = offset;
  new_blk->size = size;
  new_blk->next = NULL;

  CXLCohFreeBlock **prev = &ct2d->coherent_pool.free_list;
  CXLCohFreeBlock *cur = ct2d->coherent_pool.free_list;

  /* Find insertion point (sorted by offset) */
  while (cur && cur->offset < offset) {
    prev = &cur->next;
    cur = cur->next;
  }

  new_blk->next = cur;
  *prev = new_blk;

  /* Coalesce with next block */
  if (new_blk->next &&
      new_blk->offset + new_blk->size == new_blk->next->offset) {
    CXLCohFreeBlock *merged = new_blk->next;
    new_blk->size += merged->size;
    new_blk->next = merged->next;
    g_free(merged);
  }

  /* Coalesce with previous block */
  if (prev != &ct2d->coherent_pool.free_list) {
    CXLCohFreeBlock *prev_blk = ct2d->coherent_pool.free_list;
    while (prev_blk && prev_blk->next != new_blk) {
      prev_blk = prev_blk->next;
    }
    if (prev_blk && prev_blk->offset + prev_blk->size == new_blk->offset) {
      prev_blk->size += new_blk->size;
      prev_blk->next = new_blk->next;
      g_free(new_blk);
    }
  }

  qemu_mutex_unlock(&ct2d->coherent_pool.lock);

  qemu_log("CXL Type2: Coherent pool free: offset=0x%lx size=%lu\n",
           (unsigned long)offset, (unsigned long)size);
  return 0;
}

static uint64_t cxl_coherent_htod_command_calls(CXLType2State *ct2d) {
  uint64_t *calls = ct2d->paired_case.case_command_scope.command_calls;

  return calls[CXL_GPU_CMD_MEM_COPY_HTOD] +
         calls[CXL_GPU_CMD_MEM_COPY_HTOD_ASYNC] +
         calls[CXL_GPU_CMD_BATCH_HTOD_ASYNC] + calls[CXL_GPU_CMD_BULK_HTOD] +
         calls[CXL_GPU_CMD_BULK_HTOD_ASYNC] +
         calls[CXL_GPU_CMD_BATCH_HTOD_DIRECT_ASYNC] +
         calls[CXL_GPU_CMD_SOURCE_REGISTER_BATCH_HTOD_DIRECT_ASYNC];
}

static int cxl_coherent_map_device(CXLType2State *ct2d, uint64_t offset,
                                   uint64_t mapped_bytes,
                                   uint64_t request_bytes,
                                   uint64_t *device_alias, int *can_map) {
  HetGPUState *hetgpu = &ct2d->gpu_info.hetgpu_state;
  CXLCohAllocation *allocation;
  uint8_t *host_ptr;
  int result;

  if (!device_alias || !can_map || !mapped_bytes || !request_bytes ||
      request_bytes > mapped_bytes) {
    return CXL_GPU_ERROR_INVALID_VALUE;
  }
  qemu_mutex_lock(&ct2d->coherent_pool.lock);
  allocation = g_hash_table_lookup(ct2d->coherent_pool.allocations, &offset);
  if (!allocation || allocation->host_registered ||
      mapped_bytes != allocation->size) {
    qemu_mutex_unlock(&ct2d->coherent_pool.lock);
    return CXL_GPU_ERROR_INVALID_VALUE;
  }
  qemu_mutex_unlock(&ct2d->coherent_pool.lock);

  host_ptr = cxl_type2_bar4_host_ptr(ct2d, offset, mapped_bytes);
  if (!host_ptr || !cxl_type2_bulk_memsim_access(ct2d, CXL_OP_READ, offset,
                                                 request_bytes, host_ptr)) {
    return CXL_GPU_ERROR_INVALID_VALUE;
  }
  result = hetgpu_cuda_device_get_attribute(
      hetgpu, CXL_CUDA_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, can_map);
  if (result != CXL_GPU_SUCCESS || !*can_map) {
    return result != CXL_GPU_SUCCESS ? result : CXL_GPU_ERROR_NOT_SUPPORTED;
  }
  result = hetgpu_cuda_mem_host_register(
      hetgpu, host_ptr, mapped_bytes,
      CXL_CUDA_MEMHOSTREGISTER_PORTABLE | CXL_CUDA_MEMHOSTREGISTER_DEVICEMAP);
  if (result != CXL_GPU_SUCCESS) {
    return result;
  }

  qemu_mutex_lock(&ct2d->coherent_pool.lock);
  allocation->host_registered = true;
  allocation->htod_calls_at_map = cxl_coherent_htod_command_calls(ct2d);
  qemu_mutex_unlock(&ct2d->coherent_pool.lock);
  result =
      hetgpu_cuda_mem_host_get_device_pointer(hetgpu, device_alias, host_ptr);
  if (result != CXL_GPU_SUCCESS || !*device_alias) {
    int unregister_result = hetgpu_cuda_mem_host_unregister(hetgpu, host_ptr);
    if (unregister_result == CXL_GPU_SUCCESS) {
      qemu_mutex_lock(&ct2d->coherent_pool.lock);
      allocation->host_registered = false;
      qemu_mutex_unlock(&ct2d->coherent_pool.lock);
    }
    return result != CXL_GPU_SUCCESS ? result : CXL_GPU_ERROR_INVALID_VALUE;
  }
  qemu_mutex_lock(&ct2d->coherent_pool.lock);
  allocation->ever_mapped = true;
  allocation->device_alias = *device_alias;
  qemu_mutex_unlock(&ct2d->coherent_pool.lock);
  return CXL_GPU_SUCCESS;
}

typedef struct CXLCoherentUnmapContext {
  CXLType2State *ct2d;
  HetGPUState *hetgpu;
  uint8_t *host_ptr;
} CXLCoherentUnmapContext;

static int cxl_coherent_unmap_synchronize(void *opaque) {
  CXLCoherentUnmapContext *context = opaque;

  return hetgpu_synchronize(context->hetgpu) == HETGPU_SUCCESS
             ? CXL_GPU_SUCCESS
             : CXL_GPU_ERROR_UNKNOWN;
}

static uint64_t cxl_coherent_unmap_htod_calls(void *opaque) {
  CXLCoherentUnmapContext *context = opaque;

  return cxl_coherent_htod_command_calls(context->ct2d);
}

static int cxl_coherent_unmap_unregister(void *opaque) {
  CXLCoherentUnmapContext *context = opaque;

  return hetgpu_cuda_mem_host_unregister(context->hetgpu, context->host_ptr);
}

static int cxl_coherent_unmap_device(CXLType2State *ct2d, uint64_t offset,
                                     uint64_t device_alias,
                                     uint64_t *htod_delta) {
  HetGPUState *hetgpu = &ct2d->gpu_info.hetgpu_state;
  const CXLType2CoherentUnmapOps ops = {
      .synchronize = cxl_coherent_unmap_synchronize,
      .htod_calls = cxl_coherent_unmap_htod_calls,
      .unregister_host = cxl_coherent_unmap_unregister,
  };
  CXLCohAllocation *allocation;
  CXLCoherentUnmapContext context;
  uint8_t *host_ptr;
  uint64_t calls_at_map;
  uint64_t stored_alias;
  bool host_registered;
  bool mapping_invalidated;
  int result;

  if (!htod_delta) {
    return CXL_GPU_ERROR_INVALID_VALUE;
  }
  qemu_mutex_lock(&ct2d->coherent_pool.lock);
  allocation = g_hash_table_lookup(ct2d->coherent_pool.allocations, &offset);
  if (!allocation) {
    qemu_mutex_unlock(&ct2d->coherent_pool.lock);
    return CXL_GPU_ERROR_INVALID_VALUE;
  }
  host_registered = allocation->host_registered;
  stored_alias = allocation->device_alias;
  calls_at_map = allocation->htod_calls_at_map;
  host_ptr = cxl_type2_bar4_host_ptr(ct2d, offset, allocation->size);
  qemu_mutex_unlock(&ct2d->coherent_pool.lock);
  if (!host_ptr) {
    return CXL_GPU_ERROR_INVALID_VALUE;
  }
  context = (CXLCoherentUnmapContext){
      .ct2d = ct2d,
      .hetgpu = hetgpu,
      .host_ptr = host_ptr,
  };
  result = cxl_type2_coherent_unmap_execute(
      host_registered, stored_alias, device_alias, calls_at_map, &ops, &context,
      htod_delta, &mapping_invalidated);
  if (mapping_invalidated) {
    qemu_mutex_lock(&ct2d->coherent_pool.lock);
    allocation->host_registered = false;
    allocation->device_alias = 0;
    qemu_mutex_unlock(&ct2d->coherent_pool.lock);
  }
  return result;
}

static void cxl_coherent_pool_cleanup_mappings(CXLType2State *ct2d) {
  GHashTableIter iter;
  gpointer key;
  gpointer value;

  if (!ct2d->coherent_pool.allocations) {
    return;
  }
  g_hash_table_iter_init(&iter, ct2d->coherent_pool.allocations);
  while (g_hash_table_iter_next(&iter, &key, &value)) {
    uint64_t offset = *(uint64_t *)key;
    CXLCohAllocation *allocation = value;
    uint8_t *host_ptr;

    if (!allocation->host_registered) {
      continue;
    }
    host_ptr = cxl_type2_bar4_host_ptr(ct2d, offset, allocation->size);
    if (!host_ptr ||
        hetgpu_cuda_mem_host_unregister(&ct2d->gpu_info.hetgpu_state,
                                        host_ptr) != CXL_GPU_SUCCESS) {
      error_report("CXL Type2 coherent mapping cleanup failed at"
                   " offset=0x%" PRIx64,
                   offset);
      continue;
    }
    allocation->host_registered = false;
    allocation->device_alias = 0;
  }
}

/* ========================================================================
 * GPU Command Interface
 * ======================================================================== */

static const char *cxl_type2_paired_case_name(uint32_t case_kind) {
  switch (case_kind) {
  case CXL_GPU_CASE_BASELINE:
    return "baseline";
  case CXL_GPU_CASE_CONCORDIA:
    return "concordia";
  default:
    return "invalid";
  }
}

static void cxl_type2_qdict_put_u64(QDict *dict, const char *key,
                                    uint64_t value) {
  qdict_put(dict, key, qnum_from_uint(value));
}

static QDict *cxl_type2_model_supply_event(CXLType2State *ct2d) {
  QDict *event = qdict_new();

  cxl_type2_qdict_put_u64(event, "run_binding", ct2d->paired_case.run_binding);
  qdict_put_str(event, "case",
                cxl_type2_paired_case_name(ct2d->paired_case.active_case));
  cxl_type2_qdict_put_u64(event, "case_epoch", ct2d->paired_case.active_epoch);
  qdict_put_str(event, "planned_route",
                cxl_type2_model_supply_route_name(ct2d->model_supply.route));
  return event;
}

static void cxl_type2_model_supply_emit(const char *marker, QDict *event) {
  g_autoptr(GString) json = qobject_to_json(QOBJECT(event));

  qemu_log("%s %s\n", marker, json->str);
  qobject_unref(event);
}

static void cxl_type2_model_supply_hash_event(GChecksum *checksum,
                                              QDict *event) {
  g_autoptr(GString) json = qobject_to_json(QOBJECT(event));

  g_checksum_update(checksum, (const guchar *)json->str, json->len);
  g_checksum_update(checksum, (const guchar *)"\n", 1);
}

static QList *
cxl_type2_model_supply_source_calls(const CXLType2CudaPageableAlias *alias) {
  QList *calls = qlist_new();

  if (alias) {
    for (size_t i = 0; i < alias->contributing_source_call_count; i++) {
      qlist_append(calls,
                   qnum_from_uint(alias->contributing_source_call_ids[i]));
    }
  }
  return calls;
}

static void cxl_type2_model_supply_fail(CXLType2State *ct2d, uint64_t call_id,
                                        const char *stage, const char *detail,
                                        uint64_t source_index,
                                        uint64_t allocation_base,
                                        uint64_t allocation_epoch,
                                        uint64_t generation) {
  QDict *event = cxl_type2_model_supply_event(ct2d);

  if (!ct2d->model_supply.first_failure_stage) {
    ct2d->model_supply.first_failure_stage = g_strdup(stage);
    ct2d->model_supply.first_failure_detail = g_strdup(detail);
  }
  qdict_put_str(event, "stage", stage);
  qdict_put_str(event, "detail", detail);
  if (call_id) {
    cxl_type2_qdict_put_u64(event, "call_id", call_id);
  } else {
    qdict_put_null(event, "call_id");
  }
  if (source_index != UINT64_MAX) {
    cxl_type2_qdict_put_u64(event, "source_index", source_index);
  } else {
    qdict_put_null(event, "source_index");
  }
  if (allocation_base) {
    cxl_type2_qdict_put_u64(event, "destination_allocation_id",
                            allocation_base);
    cxl_type2_qdict_put_u64(event, "allocation_epoch", allocation_epoch);
  } else {
    qdict_put_null(event, "destination_allocation_id");
    qdict_put_null(event, "allocation_epoch");
  }
  if (generation) {
    cxl_type2_qdict_put_u64(event, "content_generation", generation);
  } else {
    qdict_put_null(event, "content_generation");
  }
  cxl_type2_model_supply_emit("KIMI_MODEL_SUPPLY_FAILURE", event);
}

static const char *cxl_type2_model_supply_failure_stage(const char *stage) {
  if (!stage) {
    return "source-resolve";
  }
  if (!strcmp(stage, "batch-validate") ||
      !strcmp(stage, "command-prevalidate") ||
      !strcmp(stage, "wire-contract")) {
    return "batch-validate";
  }
  if (!strcmp(stage, "range-admission") ||
      !strcmp(stage, "destination-allocation") ||
      !strcmp(stage, "member-range-admission") ||
      !strcmp(stage, "source-coverage")) {
    return "range-admission";
  }
  if (!strcmp(stage, "fabric-request")) {
    return "fabric-request";
  }
  if (!strcmp(stage, "aperture-capacity")) {
    return "aperture-capacity";
  }
  if (!strcmp(stage, "alias-geometry") ||
      !strcmp(stage, "alias-stable-va-size-mismatch")) {
    return "alias-geometry";
  }
  if (!strcmp(stage, "alias-byte-oracle-failed")) {
    return "alias-oracle";
  }
  if (g_str_has_prefix(stage, "alias-") ||
      !strcmp(stage, "source-view-allocate")) {
    return "alias-map";
  }
  if (!strcmp(stage, "source-cleanup") || !strcmp(stage, "rollback-cleanup")) {
    return "cleanup";
  }
  return "source-resolve";
}

static QDict *cxl_type2_model_supply_alias_release_event(
    CXLType2State *ct2d, const CXLType2CudaAllocation *allocation,
    const CXLType2CudaPageableAlias *alias) {
  QDict *event = cxl_type2_model_supply_event(ct2d);

  qdict_put_str(event, "action", "release");
  qdict_put_str(event, "result", "pass");
  cxl_type2_qdict_put_u64(event, "destination_allocation_id", allocation->base);
  cxl_type2_qdict_put_u64(event, "allocation_epoch", allocation->epoch);
  cxl_type2_qdict_put_u64(event, "content_generation",
                          allocation->content_generation);
  qdict_put(event, "contributing_source_call_ids",
            cxl_type2_model_supply_source_calls(alias));
  cxl_type2_qdict_put_u64(event, "destination_offset",
                          alias->destination_offset);
  cxl_type2_qdict_put_u64(event, "logical_bytes", alias->logical_bytes);
  cxl_type2_qdict_put_u64(event, "alias_offset", alias->device_alias);
  cxl_type2_qdict_put_u64(event, "file_mapped_bytes", alias->file_mapped_bytes);
  cxl_type2_qdict_put_u64(event, "host_composition_copy_bytes",
                          alias->host_composition_copy_bytes);
  cxl_type2_qdict_put_u64(event, "derived_boundary_pages",
                          alias->derived_boundary_pages);
  cxl_type2_qdict_put_u64(event, "derived_boundary_copy_bytes",
                          alias->derived_boundary_copy_bytes);
  cxl_type2_qdict_put_u64(event, "guard_bytes", alias->guard_bytes);
  cxl_type2_qdict_put_u64(event, "duration_ns", 0);
  return event;
}

static int cxl_type2_model_alias_register(CXLType2State *ct2d,
                                          CXLType2CudaAllocation *allocation,
                                          CXLType2CudaPageableAlias *alias) {
  HetGPUDevicePtr mapping_device = 0;
  int result;

  if (!allocation || !alias || alias->host_registered || !alias->mapping_base ||
      !alias->mapping_size) {
    return CXL_GPU_ERROR_INVALID_VALUE;
  }
  result = hetgpu_cuda_mem_host_register(
      &ct2d->gpu_info.hetgpu_state, alias->mapping_base, alias->mapping_size,
      CXL_CUDA_MEMHOSTREGISTER_PORTABLE | CXL_CUDA_MEMHOSTREGISTER_DEVICEMAP);
  if (result != CXL_GPU_SUCCESS) {
    return result;
  }
  alias->host_registered = true;
  result = hetgpu_cuda_mem_host_get_device_pointer(
      &ct2d->gpu_info.hetgpu_state, &mapping_device, alias->mapping_base);
  if (result != CXL_GPU_SUCCESS || !mapping_device ||
      mapping_device > UINT64_MAX - alias->pointer_bias) {
    int unregister_result = hetgpu_cuda_mem_host_unregister(
        &ct2d->gpu_info.hetgpu_state, alias->mapping_base);

    if (unregister_result == CXL_GPU_SUCCESS) {
      alias->host_registered = false;
    }
    return result != CXL_GPU_SUCCESS ? result : CXL_GPU_ERROR_INVALID_VALUE;
  }
  alias->device_alias = mapping_device + alias->pointer_bias;
  allocation->device_alias = alias->device_alias;
  return CXL_GPU_SUCCESS;
}

static int
cxl_type2_model_aliases_unregister(CXLType2State *ct2d,
                                   CXLType2CudaAllocation *allocation) {
  int first_error = CXL_GPU_SUCCESS;

  for (CXLType2CudaPageableAlias *alias = allocation->pageable_alias; alias;
       alias = alias->next) {
    if (!alias->host_registered) {
      continue;
    }
    int result = hetgpu_cuda_mem_host_unregister(&ct2d->gpu_info.hetgpu_state,
                                                 alias->mapping_base);

    if (result == CXL_GPU_SUCCESS) {
      alias->host_registered = false;
    } else if (first_error == CXL_GPU_SUCCESS) {
      first_error = result;
    }
  }
  return first_error;
}

static int cxl_type2_model_aliases_replace_range(
    CXLType2State *ct2d, CXLType2CudaAllocation *allocation,
    uint64_t destination_offset, uint64_t logical_bytes, bool *remap) {
  uint64_t end;
  size_t removed_count = 0;

  if (!allocation || !logical_bytes || !remap ||
      destination_offset > UINT64_MAX - logical_bytes ||
      allocation->consumer_refs || allocation->normal_inflight_refs ||
      allocation->graph_binding_refs || allocation->graph_inflight_refs) {
    return CXL_GPU_ERROR_INVALID_VALUE;
  }
  end = destination_offset + logical_bytes;
  for (CXLType2CudaPageableAlias *alias = allocation->pageable_alias; alias;
       alias = alias->next) {
    uint64_t alias_end = alias->destination_offset + alias->logical_bytes;

    if (alias_end <= destination_offset || alias->destination_offset >= end ||
        !alias->host_registered) {
      continue;
    }
    int result = hetgpu_cuda_mem_host_unregister(&ct2d->gpu_info.hetgpu_state,
                                                 alias->mapping_base);

    if (result != CXL_GPU_SUCCESS) {
      return result;
    }
    alias->host_registered = false;
  }
  if (!cxl_type2_cuda_allocation_remove_pageable_aliases(
          &ct2d->cuda_allocations, allocation->base, allocation->epoch,
          destination_offset, logical_bytes, &removed_count)) {
    return CXL_GPU_ERROR_INVALID_VALUE;
  }
  *remap = removed_count != 0;
  return CXL_GPU_SUCCESS;
}

static int cxl_type2_release_model_aliases(CXLType2State *ct2d) {
  int first_error = CXL_GPU_SUCCESS;

  for (size_t i = 0; i < ct2d->cuda_allocations.count; i++) {
    CXLType2CudaAllocation *allocation = &ct2d->cuda_allocations.entries[i];
    GPtrArray *events;
    uint64_t generation;

    if (!allocation->pageable_alias) {
      continue;
    }
    generation = allocation->content_generation;
    events = g_ptr_array_new();
    for (const CXLType2CudaPageableAlias *alias = allocation->pageable_alias;
         alias; alias = alias->next) {
      g_ptr_array_add(events, cxl_type2_model_supply_alias_release_event(
                                  ct2d, allocation, alias));
    }
    int unregister_result =
        cxl_type2_model_aliases_unregister(ct2d, allocation);

    if (unregister_result != CXL_GPU_SUCCESS ||
        !cxl_type2_cuda_allocation_drop_pageable_alias(
            &ct2d->cuda_allocations, allocation->base, allocation->epoch,
            generation)) {
      for (size_t j = 0; j < events->len; j++) {
        QDict *event = g_ptr_array_index(events, j);

        qobject_unref(event);
      }
      g_ptr_array_unref(events);
      cxl_type2_model_supply_fail(
          ct2d, ct2d->gpu_cmd.call_id, "cleanup", "model-alias-release-failed",
          UINT64_MAX, allocation->base, allocation->epoch, generation);
      if (first_error == CXL_GPU_SUCCESS) {
        first_error = CXL_GPU_ERROR_UNKNOWN;
      }
      continue;
    }
    for (size_t j = 0; j < events->len; j++) {
      cxl_type2_model_supply_emit("KIMI_MODEL_SUPPLY_ALIAS",
                                  g_ptr_array_index(events, j));
      ct2d->model_supply.alias_releases++;
    }
    g_ptr_array_unref(events);
  }
  return first_error;
}

static void cxl_type2_notify_case_scope(uint64_t run_binding,
                                        uint32_t case_kind, uint64_t epoch,
                                        bool begin) {
  g_autofree char *identity = g_strdup_printf(
      "run_binding=%" PRIu64 " case=%s epoch=%" PRIu64, run_binding,
      cxl_type2_paired_case_name(case_kind), epoch);

  qemu_plugin_notify_scope("type2-case", identity, begin);
}

static CXLType2IntervalIdentity cxl_type2_interval_identity(
    uint64_t sequence, const char *owner, const char *category,
    const char *operation, uint32_t operation_code, bool operation_code_valid) {
  return (CXLType2IntervalIdentity){
      .valid = true,
      .operation_code_valid = operation_code_valid,
      .operation_code = operation_code,
      .sequence = sequence,
      .owner = owner,
      .category = category,
      .operation = operation,
  };
}

static void cxl_type2_interval_fail(CXLType2IntervalLedger *ledger,
                                    const char *error) {
  if (!ledger->first_error) {
    ledger->first_error = error;
  }
}

static bool cxl_type2_interval_add(uint64_t *target, uint64_t value) {
  if (UINT64_MAX - *target < value) {
    return false;
  }
  *target += value;
  return true;
}

static void
cxl_type2_interval_reset(CXLType2IntervalLedger *ledger, int64_t span_begin_ns,
                         CXLType2IntervalIdentity span_begin_identity) {
  *ledger = (CXLType2IntervalLedger){
      .span_begin_ns = span_begin_ns,
      .span_end_ns = span_begin_ns,
      .last_begin_ns = span_begin_ns,
      .union_end_ns = span_begin_ns,
      .span_begin_identity = span_begin_identity,
      .union_end_identity = span_begin_identity,
  };
}

static void
cxl_type2_command_scope_reset(CXLType2CommandScopeLedger *scope,
                              int64_t span_begin_ns,
                              CXLType2IntervalIdentity span_begin_identity) {
  *scope = (CXLType2CommandScopeLedger){
      .active = true,
  };
  cxl_type2_interval_reset(&scope->command_intervals, span_begin_ns,
                           span_begin_identity);
  cxl_type2_interval_reset(&scope->driver_intervals, span_begin_ns,
                           span_begin_identity);
}

static void cxl_type2_interval_consider_gap(CXLType2IntervalLedger *ledger,
                                            int64_t begin_ns, int64_t end_ns,
                                            CXLType2IntervalIdentity previous,
                                            CXLType2IntervalIdentity next) {
  uint64_t duration_ns;

  if (end_ns < begin_ns) {
    cxl_type2_interval_fail(ledger, "negative-gap");
    return;
  }
  duration_ns = end_ns - begin_ns;
  if (duration_ns > ledger->largest_gap_duration_ns) {
    ledger->largest_gap_duration_ns = duration_ns;
    ledger->largest_gap_begin_ns = begin_ns;
    ledger->largest_gap_end_ns = end_ns;
    ledger->largest_gap_previous = previous;
    ledger->largest_gap_next = next;
  }
}

static void cxl_type2_interval_record(CXLType2IntervalLedger *ledger,
                                      int64_t begin_ns, int64_t end_ns,
                                      CXLType2IntervalIdentity identity) {
  uint64_t duration_ns;
  uint64_t union_increment;

  if (end_ns < begin_ns) {
    cxl_type2_interval_fail(ledger, "clock-regressed-within-interval");
    return;
  }
  if (begin_ns < ledger->span_begin_ns) {
    cxl_type2_interval_fail(ledger, "interval-before-scope");
    return;
  }
  if (ledger->interval_count > 0 && begin_ns < ledger->last_begin_ns) {
    cxl_type2_interval_fail(ledger, "interval-order-regressed");
    return;
  }
  duration_ns = end_ns - begin_ns;
  if (!cxl_type2_interval_add(&ledger->total_duration_ns, duration_ns) ||
      ledger->interval_count == UINT64_MAX) {
    cxl_type2_interval_fail(ledger, "interval-counter-overflow");
    return;
  }

  if (ledger->interval_count == 0 || begin_ns > ledger->union_end_ns) {
    cxl_type2_interval_consider_gap(ledger, ledger->union_end_ns, begin_ns,
                                    ledger->union_end_identity, identity);
    union_increment = duration_ns;
    ledger->union_end_ns = end_ns;
    ledger->union_end_identity = identity;
  } else if (end_ns > ledger->union_end_ns) {
    union_increment = end_ns - ledger->union_end_ns;
    ledger->union_end_ns = end_ns;
    ledger->union_end_identity = identity;
  } else {
    union_increment = 0;
  }
  if (!cxl_type2_interval_add(&ledger->union_duration_ns, union_increment)) {
    cxl_type2_interval_fail(ledger, "interval-counter-overflow");
    return;
  }
  ledger->interval_count++;
  ledger->last_begin_ns = begin_ns;
}

static void
cxl_type2_interval_finish(CXLType2IntervalLedger *ledger, int64_t span_end_ns,
                          CXLType2IntervalIdentity span_end_identity) {
  uint64_t span_duration_ns;

  ledger->span_end_ns = span_end_ns;
  if (span_end_ns < ledger->span_begin_ns ||
      span_end_ns < ledger->union_end_ns) {
    cxl_type2_interval_fail(ledger, "interval-after-scope");
    return;
  }
  cxl_type2_interval_consider_gap(ledger, ledger->union_end_ns, span_end_ns,
                                  ledger->union_end_identity,
                                  span_end_identity);
  span_duration_ns = span_end_ns - ledger->span_begin_ns;
  if (ledger->union_duration_ns > span_duration_ns ||
      ledger->total_duration_ns < ledger->union_duration_ns) {
    cxl_type2_interval_fail(ledger, "interval-arithmetic-invalid");
  }
}

static void
cxl_type2_command_scope_finish(CXLType2CommandScopeLedger *scope,
                               int64_t span_end_ns,
                               CXLType2IntervalIdentity span_end_identity) {
  cxl_type2_interval_finish(&scope->command_intervals, span_end_ns,
                            span_end_identity);
  cxl_type2_interval_finish(&scope->driver_intervals, span_end_ns,
                            span_end_identity);
  scope->active = false;
}

static void
cxl_type2_interval_format_identity(const CXLType2IntervalIdentity *identity,
                                   char *sequence, size_t sequence_size,
                                   char *operation, size_t operation_size) {
  if (!identity->valid) {
    pstrcpy(sequence, sequence_size, "null");
    pstrcpy(operation, operation_size, "null");
    return;
  }
  snprintf(sequence, sequence_size, "%" PRIu64, identity->sequence);
  if (identity->operation_code_valid) {
    if (identity->operation) {
      snprintf(operation, operation_size, "%s-%u", identity->operation,
               identity->operation_code);
    } else {
      snprintf(operation, operation_size, "cmd-0x%02x",
               identity->operation_code);
    }
  } else {
    pstrcpy(operation, operation_size,
            identity->operation ? identity->operation : "null");
  }
}

static void cxl_type2_log_interval_summary(uint64_t run_binding,
                                           uint32_t case_kind, uint64_t epoch,
                                           const char *scope,
                                           const char *category,
                                           CXLType2IntervalLedger *ledger) {
  uint64_t span_duration_ns = ledger->span_end_ns >= ledger->span_begin_ns
                                  ? ledger->span_end_ns - ledger->span_begin_ns
                                  : 0;
  uint64_t overlap_duration_ns =
      ledger->total_duration_ns >= ledger->union_duration_ns
          ? ledger->total_duration_ns - ledger->union_duration_ns
          : 0;
  uint64_t gap_duration_ns = span_duration_ns >= ledger->union_duration_ns
                                 ? span_duration_ns - ledger->union_duration_ns
                                 : 0;
  char largest_begin[32] = "null";
  char largest_end[32] = "null";
  char previous_sequence[32];
  char previous_operation[80];
  char next_sequence[32];
  char next_operation[80];

  if (ledger->largest_gap_duration_ns > 0) {
    snprintf(largest_begin, sizeof(largest_begin), "%" PRId64,
             ledger->largest_gap_begin_ns);
    snprintf(largest_end, sizeof(largest_end), "%" PRId64,
             ledger->largest_gap_end_ns);
  }
  cxl_type2_interval_format_identity(
      &ledger->largest_gap_previous, previous_sequence,
      sizeof(previous_sequence), previous_operation,
      sizeof(previous_operation));
  cxl_type2_interval_format_identity(&ledger->largest_gap_next, next_sequence,
                                     sizeof(next_sequence), next_operation,
                                     sizeof(next_operation));

  qemu_log(
      "KIMI_INTERVAL_SUMMARY schema=interval-summary-v1 producer=qemu"
      " clock_domain=host-monotonic run_binding=%" PRIu64
      " case=%s case_epoch=%" PRIu64 " scope=%s category=%s owner=qemu"
      " status=%s span_begin_ns=%" PRId64 " span_end_ns=%" PRId64
      " interval_count=%" PRIu64 " total_duration_ns=%" PRIu64
      " union_duration_ns=%" PRIu64 " overlap_duration_ns=%" PRIu64
      " gap_duration_ns=%" PRIu64
      " largest_gap_begin_ns=%s largest_gap_end_ns=%s"
      " previous_sequence=%s previous_owner=%s previous_category=%s"
      " previous_operation=%s next_sequence=%s next_owner=%s"
      " next_category=%s next_operation=%s first_error=%s\n",
      run_binding, cxl_type2_paired_case_name(case_kind), epoch, scope,
      category, ledger->first_error ? "incomplete" : "complete",
      ledger->span_begin_ns, ledger->span_end_ns, ledger->interval_count,
      ledger->total_duration_ns, ledger->union_duration_ns, overlap_duration_ns,
      gap_duration_ns, largest_begin, largest_end, previous_sequence,
      ledger->largest_gap_previous.valid ? ledger->largest_gap_previous.owner
                                         : "null",
      ledger->largest_gap_previous.valid ? ledger->largest_gap_previous.category
                                         : "null",
      previous_operation, next_sequence,
      ledger->largest_gap_next.valid ? ledger->largest_gap_next.owner : "null",
      ledger->largest_gap_next.valid ? ledger->largest_gap_next.category
                                     : "null",
      next_operation, ledger->first_error ? ledger->first_error : "none");
}

static void cxl_type2_cuda_classifier_reject(
    CXLType2State *ct2d, CXLType2CudaClassifierStatus status,
    CXLType2CudaRejectionReason reason, uint32_t command, bool command_valid) {
  if (ct2d->paired_case.classifier_status ==
          CXL_TYPE2_CUDA_CLASSIFIER_AVAILABLE ||
      status == CXL_TYPE2_CUDA_CLASSIFIER_CONTRADICTED) {
    ct2d->paired_case.classifier_status = status;
  }
  if (ct2d->paired_case.first_rejection_reason ==
      CXL_TYPE2_CUDA_REJECTION_NONE) {
    ct2d->paired_case.first_rejection_reason = reason;
    ct2d->paired_case.first_rejection_command = command;
    ct2d->paired_case.first_rejection_command_valid = command_valid;
  }
}

static bool cxl_type2_cuda_counter_can_add(uint64_t value, uint64_t addend) {
  return value <= UINT64_MAX - addend;
}

static void cxl_type2_cuda_classify_graph(CXLType2State *ct2d,
                                          HetGPUGraph graph) {
  HetGPUState *hetgpu = &ct2d->gpu_info.hetgpu_state;
  g_autofree HetGPUGraphNode *nodes = NULL;
  size_t count = 0;

  ct2d->paired_case.graph_exec_observed++;
  if (hetgpu_cuda_graph_get_nodes(hetgpu, graph, NULL, &count) !=
          CXL_GPU_SUCCESS ||
      !count) {
    goto incomplete;
  }
  nodes = g_try_new(HetGPUGraphNode, count);
  if (!nodes ||
      hetgpu_cuda_graph_get_nodes(hetgpu, graph, nodes, &count) !=
          CXL_GPU_SUCCESS ||
      !count) {
    goto incomplete;
  }
  for (size_t i = 0; i < count; i++) {
    int node_type = -1;

    if (hetgpu_cuda_graph_node_get_type(hetgpu, nodes[i], &node_type) !=
        CXL_GPU_SUCCESS) {
      goto incomplete;
    }
    if (node_type != 0) {
      ct2d->paired_case.graph_child_or_non_kernel++;
      cxl_type2_cuda_classifier_reject(
          ct2d, CXL_TYPE2_CUDA_CLASSIFIER_UNAVAILABLE,
          CXL_TYPE2_CUDA_REJECTION_GRAPH_NON_KERNEL,
          CXL_GPU_CMD_GRAPH_INSTANTIATE, true);
      return;
    }
  }
  ct2d->paired_case.graph_all_kernel_flat++;
  return;

incomplete:
  ct2d->paired_case.graph_incomplete++;
  cxl_type2_cuda_classifier_reject(ct2d, CXL_TYPE2_CUDA_CLASSIFIER_UNAVAILABLE,
                                   CXL_TYPE2_CUDA_REJECTION_GRAPH_INCOMPLETE,
                                   CXL_GPU_CMD_GRAPH_INSTANTIATE, true);
}

typedef struct CXLType2GraphNodeGenerationConsumers {
  HetGPUGraphNode node;
  uint32_t function_id;
  uint64_t graph_node_token;
  GArray *identities;
  GArray *alias_bindings;
} CXLType2GraphNodeGenerationConsumers;

typedef struct CXLType2GraphExecGenerationConsumers {
  CXLType2State *ct2d;
  uint32_t graph_exec_id;
  GArray *nodes;
} CXLType2GraphExecGenerationConsumers;

typedef struct CXLType2AliasLaunchBinding {
  CXLType2CudaAllocationIdentity identity;
  uint64_t generation;
  uint64_t allocation_begin;
  uint64_t allocation_end;
  uint64_t alias_begin;
  bool owns_inflight_ref;
} CXLType2AliasLaunchBinding;

static const CXLType2CudaPageableAlias *
cxl_type2_model_alias_for_binding(const CXLType2CudaAllocation *allocation,
                                  const CXLType2AliasLaunchBinding *binding) {
  if (!allocation || !binding) {
    return NULL;
  }
  for (const CXLType2CudaPageableAlias *alias = allocation->pageable_alias;
       alias; alias = alias->next) {
    if (alias->destination_offset == binding->allocation_begin &&
        alias->destination_offset + alias->logical_bytes ==
            binding->allocation_end &&
        alias->device_alias == binding->alias_begin) {
      return alias;
    }
  }
  return NULL;
}

static void cxl_type2_generation_reuse_fail(CXLType2State *ct2d,
                                            const char *reason) {
  if (!ct2d->generation_reuse_error) {
    ct2d->generation_reuse_error = reason;
  }
}

static bool cxl_type2_cuda_param_allocation_identity(
    CXLType2State *ct2d, const void *value, size_t size,
    CXLType2CudaAllocationIdentity *identity) {
  uint64_t pointer;

  if (size != sizeof(pointer)) {
    return false;
  }
  ct2d->paired_case.graph_pointer_sized_params++;
  memcpy(&pointer, value, sizeof(pointer));
  if (!cxl_type2_cuda_allocation_identity_for_address(&ct2d->cuda_allocations,
                                                      pointer, identity)) {
    uint64_t generation;

    if (!cxl_type2_cuda_allocation_identity_for_alias_address(
            &ct2d->cuda_allocations, pointer, identity, &generation)) {
      return false;
    }
  }
  ct2d->paired_case.graph_allocation_pointer_params++;
  if (pointer == identity->base) {
    ct2d->paired_case.graph_allocation_base_params++;
  }
  return true;
}

static void cxl_type2_generation_identities_append(
    GArray *identities, CXLType2CudaAllocationIdentity identity) {
  for (size_t i = 0; i < identities->len; i++) {
    const CXLType2CudaAllocationIdentity *candidate =
        &g_array_index(identities, CXLType2CudaAllocationIdentity, i);

    if (candidate->base == identity.base &&
        candidate->epoch == identity.epoch) {
      return;
    }
  }
  g_array_append_val(identities, identity);
}

static GArray *cxl_type2_generation_identities_from_params(
    CXLType2State *ct2d, const HetGPUParamLayout *layout, void **params) {
  GArray *identities =
      g_array_new(false, false, sizeof(CXLType2CudaAllocationIdentity));

  for (uint32_t i = 0; i < layout->num_args; i++) {
    CXLType2CudaAllocationIdentity identity;

    if (params[i] && cxl_type2_cuda_param_allocation_identity(
                         ct2d, params[i], layout->sizes[i], &identity)) {
      cxl_type2_generation_identities_append(identities, identity);
    }
  }
  return identities;
}

typedef enum CXLType2ModelParamClass {
  CXL_TYPE2_MODEL_PARAM_ALIAS,
  CXL_TYPE2_MODEL_PARAM_GPU_DESTINATION,
  CXL_TYPE2_MODEL_PARAM_NON_MODEL_ALLOCATION,
  CXL_TYPE2_MODEL_PARAM_UNCLASSIFIED,
} CXLType2ModelParamClass;

static CXLType2ModelParamClass
cxl_type2_model_param_classify(CXLType2State *ct2d, uint64_t pointer) {
  CXLType2CudaAllocationIdentity identity;
  CXLType2CudaAllocation *allocation;
  uint64_t generation;

  if (cxl_type2_cuda_allocation_identity_for_alias_address(
          &ct2d->cuda_allocations, pointer, &identity, &generation)) {
    return CXL_TYPE2_MODEL_PARAM_ALIAS;
  }
  if (!cxl_type2_cuda_allocation_identity_for_address(&ct2d->cuda_allocations,
                                                      pointer, &identity)) {
    return CXL_TYPE2_MODEL_PARAM_UNCLASSIFIED;
  }
  allocation = cxl_type2_cuda_allocation_find(&ct2d->cuda_allocations,
                                              identity.base, identity.epoch);
  if (allocation && allocation->pageable_alias &&
      cxl_type2_cuda_pageable_alias_contains(allocation->pageable_alias,
                                             pointer - identity.base)) {
    return CXL_TYPE2_MODEL_PARAM_GPU_DESTINATION;
  }
  return CXL_TYPE2_MODEL_PARAM_NON_MODEL_ALLOCATION;
}

static bool cxl_type2_model_consumer_validate(
    CXLType2State *ct2d, const CXLType2FunctionEntry *function,
    const HetGPUParamLayout *layout, void **params, const char **reason) {
  const CXLType2ModelConsumerFunction *consumer;

  if (reason) {
    *reason = NULL;
  }
  if (ct2d->model_supply.route != CXL_TYPE2_MODEL_SUPPLY_CXL_DIRECT) {
    return true;
  }
  if (!reason || !function || !function->handle || !layout || !params ||
      !(consumer = function->consumer)) {
    if (reason) {
      *reason = "consumer-certificate-entry-missing";
    }
    return false;
  }
  if (consumer->parameter_count != layout->num_args) {
    *reason = "consumer-parameter-count-mismatch";
    return false;
  }
  for (uint32_t i = 0; i < layout->num_args; i++) {
    const CXLType2ModelConsumerRoot *parameter = &consumer->parameters[i];

    if (parameter->parameter_index != i ||
        parameter->parameter_offset != layout->offsets[i] ||
        parameter->parameter_size != layout->sizes[i]) {
      *reason = "consumer-parameter-layout-mismatch";
      return false;
    }
  }
  for (size_t i = 0; i < consumer->sink_count; i++) {
    const CXLType2ModelConsumerSink *sink = &consumer->sinks[i];
    bool global = !strcmp(sink->memory_space, "global") ||
                  !strcmp(sink->memory_space, "generic");
    bool has_model_source = false;
    bool has_other_root = false;

    if (!global) {
      continue;
    }
    for (size_t j = 0; j < sink->root_count; j++) {
      const CXLType2ModelConsumerRoot *root = &sink->roots[j];
      CXLType2ModelParamClass classification;
      uint64_t pointer;

      if (root->parameter_index >= layout->num_args ||
          root->parameter_offset != layout->offsets[root->parameter_index] ||
          root->parameter_size != layout->sizes[root->parameter_index] ||
          !params[root->parameter_index]) {
        *reason = "consumer-root-not-visible";
        return false;
      }
      /* Address roots also contain scalar indices and strides. */
      if (root->parameter_size != sizeof(pointer)) {
        continue;
      }
      memcpy(&pointer, params[root->parameter_index], sizeof(pointer));
      classification = cxl_type2_model_param_classify(ct2d, pointer);
      if (classification == CXL_TYPE2_MODEL_PARAM_UNCLASSIFIED) {
        continue;
      }
      if (classification == CXL_TYPE2_MODEL_PARAM_ALIAS ||
          classification == CXL_TYPE2_MODEL_PARAM_GPU_DESTINATION) {
        has_model_source = true;
      } else {
        has_other_root = true;
      }
    }
    if (has_model_source && strcmp(sink->access_mode, "read")) {
      *reason = "model-alias-access-not-read";
      return false;
    }
    if (has_model_source && has_other_root) {
      *reason = "model-alias-root-mixed";
      return false;
    }
  }
  return true;
}

static const char *
cxl_type2_model_param_class_name(CXLType2ModelParamClass classification) {
  switch (classification) {
  case CXL_TYPE2_MODEL_PARAM_ALIAS:
    return "model-alias";
  case CXL_TYPE2_MODEL_PARAM_GPU_DESTINATION:
    return "gpu-destination";
  case CXL_TYPE2_MODEL_PARAM_NON_MODEL_ALLOCATION:
    return "non-model-allocation";
  case CXL_TYPE2_MODEL_PARAM_UNCLASSIFIED:
    return "unclassified";
  }
  g_assert_not_reached();
}

static bool cxl_type2_model_consumer_parameter_is_read_root(
    const CXLType2ModelConsumerFunction *consumer, uint32_t parameter_index) {
  if (!consumer) {
    return false;
  }
  for (size_t i = 0; i < consumer->sink_count; i++) {
    const CXLType2ModelConsumerSink *sink = &consumer->sinks[i];
    bool global = !strcmp(sink->memory_space, "global") ||
                  !strcmp(sink->memory_space, "generic");

    if (!global || strcmp(sink->access_mode, "read")) {
      continue;
    }
    for (size_t j = 0; j < sink->root_count; j++) {
      if (sink->roots[j].parameter_index == parameter_index &&
          sink->roots[j].parameter_size == sizeof(uint64_t)) {
        return true;
      }
    }
  }
  return false;
}

static void cxl_type2_log_model_consumer_launch(
    CXLType2State *ct2d, const char *action, uint64_t launch_id,
    uint32_t function_id, const HetGPUParamLayout *layout, void **params,
    const GArray *bindings, CXLType2CudaAliasConsumer consumer,
    uint64_t graph_exec_id, const uint64_t *graph_node_tokens,
    size_t graph_node_count) {
  const CXLType2FunctionEntry *function = &ct2d->gpu_cmd.functions[function_id];
  const CXLType2ModuleEntry *module =
      &ct2d->gpu_cmd.modules[function->module_id];
  for (size_t i = 0; i < bindings->len; i++) {
    const CXLType2AliasLaunchBinding *binding =
        &g_array_index(bindings, CXLType2AliasLaunchBinding, i);
    CXLType2CudaAllocation *allocation = cxl_type2_cuda_allocation_find(
        &ct2d->cuda_allocations, binding->identity.base,
        binding->identity.epoch);

    QList *roots = qlist_new();
    QList *nodes = qlist_new();
    QDict *event;

    const CXLType2CudaPageableAlias *alias =
        cxl_type2_model_alias_for_binding(allocation, binding);

    if (!alias) {
      cxl_type2_model_supply_fail(ct2d, ct2d->gpu_cmd.call_id, "consumer-bind",
                                  "consumer-alias-missing", 0,
                                  binding->identity.base,
                                  binding->identity.epoch, binding->generation);
      continue;
    }
    for (uint32_t parameter_index = 0; parameter_index < layout->num_args;
         parameter_index++) {
      CXLType2CudaAllocationIdentity identity = {0};
      CXLType2ModelParamClass classification =
          CXL_TYPE2_MODEL_PARAM_UNCLASSIFIED;
      uint64_t pointer = 0;
      uint64_t root_generation = 0;
      QDict *root;

      if (layout->sizes[parameter_index] != sizeof(pointer) ||
          !params[parameter_index]) {
        continue;
      }
      memcpy(&pointer, params[parameter_index], sizeof(pointer));
      classification = cxl_type2_model_param_classify(ct2d, pointer);
      if (classification == CXL_TYPE2_MODEL_PARAM_ALIAS) {
        cxl_type2_cuda_allocation_identity_for_alias_address(
            &ct2d->cuda_allocations, pointer, &identity, &root_generation);
      } else if (classification != CXL_TYPE2_MODEL_PARAM_UNCLASSIFIED) {
        cxl_type2_cuda_allocation_identity_for_address(&ct2d->cuda_allocations,
                                                       pointer, &identity);
      }
      root = qdict_new();
      cxl_type2_qdict_put_u64(root, "parameter_index", parameter_index);
      cxl_type2_qdict_put_u64(root, "parameter_offset",
                              layout->offsets[parameter_index]);
      cxl_type2_qdict_put_u64(root, "parameter_size",
                              layout->sizes[parameter_index]);
      qdict_put_str(root, "runtime_classification",
                    cxl_type2_model_param_class_name(classification));
      if (identity.base) {
        QDict *allocation_identity = qdict_new();

        cxl_type2_qdict_put_u64(allocation_identity, "base", identity.base);
        cxl_type2_qdict_put_u64(allocation_identity, "epoch", identity.epoch);
        cxl_type2_qdict_put_u64(allocation_identity, "generation",
                                root_generation);
        qdict_put(root, "allocation_identity", allocation_identity);
      } else {
        qdict_put_null(root, "allocation_identity");
      }
      qdict_put_null(root, "source_view_identity");
      qlist_append(roots, root);
    }

    event = cxl_type2_model_supply_event(ct2d);
    qdict_put_str(event, "action", action);
    qdict_put_str(event, "result", "pass");
    cxl_type2_qdict_put_u64(event, "launch_id", launch_id);
    cxl_type2_qdict_put_u64(event, "destination_allocation_id",
                            binding->identity.base);
    cxl_type2_qdict_put_u64(event, "allocation_epoch", binding->identity.epoch);
    cxl_type2_qdict_put_u64(event, "content_generation", binding->generation);
    qdict_put(event, "contributing_source_call_ids",
              cxl_type2_model_supply_source_calls(alias));
    cxl_type2_qdict_put_u64(event, "destination_cuda_token",
                            binding->alias_begin);
    qdict_put_str(event, "binding_kind",
                  consumer == CXL_TYPE2_CUDA_ALIAS_NORMAL ? "normal-top-level"
                                                          : "graph-top-level");
    if (consumer == CXL_TYPE2_CUDA_ALIAS_NORMAL) {
      qdict_put_null(event, "graph_exec_id");
    } else {
      cxl_type2_qdict_put_u64(event, "graph_exec_id", graph_exec_id);
    }
    qdict_put_str(event, "node_kind",
                  consumer == CXL_TYPE2_CUDA_ALIAS_NORMAL ? "kernel"
                                                          : "graph-kernel");
    for (size_t node_index = 0; node_index < graph_node_count; node_index++) {
      qlist_append(nodes, qnum_from_uint(graph_node_tokens[node_index]));
    }
    qdict_put(event, "graph_node_tokens", nodes);
    qdict_put_str(event, "module_load_kind",
                  module->load_kind == CXL_TYPE2_MODULE_LOAD_CUBIN ? "cubin"
                                                                   : "ptx");
    qdict_put_str(event, "driver_input_cubin_sha256",
                  module->driver_input_cubin_sha256);
    qdict_put_str(event, "function_name", function->name);
    qdict_put_str(event, "static_consumer_certificate_sha256",
                  ct2d->model_supply.consumer_certificate.sha256);
    qdict_put(event, "visible_parameter_roots", roots);
    cxl_type2_model_supply_emit("KIMI_MODEL_SUPPLY_CONSUMER", event);
  }
  if (!strcmp(action, "launch")) {
    if (consumer == CXL_TYPE2_CUDA_ALIAS_NORMAL) {
      ct2d->model_supply.normal_launches++;
    } else {
      ct2d->model_supply.graph_launches++;
    }
  }
}

static void cxl_type2_log_model_graph_unbind(
    CXLType2State *ct2d, uint32_t graph_exec_id,
    const CXLType2GraphNodeGenerationConsumers *node,
    const CXLType2AliasLaunchBinding *binding) {
  const CXLType2CudaAllocation *allocation = cxl_type2_cuda_allocation_find(
      &ct2d->cuda_allocations, binding->identity.base, binding->identity.epoch);
  const CXLType2CudaPageableAlias *alias =
      cxl_type2_model_alias_for_binding(allocation, binding);
  const CXLType2FunctionEntry *function =
      &ct2d->gpu_cmd.functions[node->function_id];
  const CXLType2ModuleEntry *module =
      &ct2d->gpu_cmd.modules[function->module_id];
  QDict *event = cxl_type2_model_supply_event(ct2d);
  QList *nodes = qlist_new();

  qlist_append(nodes, qnum_from_uint(node->graph_node_token));
  qdict_put_str(event, "action", "unbind");
  qdict_put_str(event, "result", "pass");
  cxl_type2_qdict_put_u64(event, "launch_id", 0);
  cxl_type2_qdict_put_u64(event, "destination_allocation_id", allocation->base);
  cxl_type2_qdict_put_u64(event, "allocation_epoch", allocation->epoch);
  cxl_type2_qdict_put_u64(event, "content_generation", binding->generation);
  qdict_put(event, "contributing_source_call_ids",
            cxl_type2_model_supply_source_calls(alias));
  cxl_type2_qdict_put_u64(event, "destination_cuda_token",
                          binding->alias_begin);
  qdict_put_str(event, "node_kind", "graph-kernel");
  qdict_put_str(event, "binding_kind", "graph-top-level");
  cxl_type2_qdict_put_u64(event, "graph_exec_id", graph_exec_id);
  qdict_put(event, "graph_node_tokens", nodes);
  qdict_put_str(event, "module_load_kind",
                module->load_kind == CXL_TYPE2_MODULE_LOAD_CUBIN ? "cubin"
                                                                 : "ptx");
  qdict_put_str(event, "driver_input_cubin_sha256",
                module->driver_input_cubin_sha256);
  qdict_put_str(event, "function_name", function->name);
  qdict_put_str(event, "static_consumer_certificate_sha256",
                ct2d->model_supply.consumer_certificate.sha256);
  qdict_put(event, "visible_parameter_roots", qlist_new());
  cxl_type2_model_supply_emit("KIMI_MODEL_SUPPLY_CONSUMER", event);
}

static void
cxl_type2_alias_launch_bindings_release(CXLType2State *ct2d, GArray *bindings,
                                        CXLType2CudaAliasConsumer consumer) {
  if (!bindings) {
    return;
  }
  for (size_t i = 0; i < bindings->len; i++) {
    CXLType2AliasLaunchBinding *binding =
        &g_array_index(bindings, CXLType2AliasLaunchBinding, i);

    if (binding->owns_inflight_ref &&
        !cxl_type2_cuda_allocation_release_pageable_alias(
            &ct2d->cuda_allocations, binding->identity.base,
            binding->identity.epoch, binding->generation, consumer)) {
      cxl_type2_generation_reuse_fail(ct2d,
                                      "model-alias-consumer-release-failed");
    }
  }
  g_array_unref(bindings);
}

static GArray *cxl_type2_model_alias_rewrite_params(
    CXLType2State *ct2d, const CXLType2FunctionEntry *function,
    const HetGPUParamLayout *layout, void **params,
    CXLType2CudaAliasConsumer consumer, const char **reason) {
  GArray *bindings =
      g_array_new(false, false, sizeof(CXLType2AliasLaunchBinding));

  if (reason) {
    *reason = NULL;
  }
  if (ct2d->model_supply.route != CXL_TYPE2_MODEL_SUPPLY_CXL_DIRECT) {
    return bindings;
  }
  for (uint32_t i = 0; i < layout->num_args; i++) {
    CXLType2CudaAllocationIdentity identity;
    CXLType2AliasLaunchBinding *binding = NULL;
    uint64_t pointer;
    uint64_t generation;
    uint64_t alias_address;
    CXLType2ModelParamClass classification;

    if (!params[i] || layout->sizes[i] != sizeof(pointer)) {
      continue;
    }
    memcpy(&pointer, params[i], sizeof(pointer));
    classification = cxl_type2_model_param_classify(ct2d, pointer);
    for (size_t j = 0; j < bindings->len; j++) {
      CXLType2AliasLaunchBinding *candidate =
          &g_array_index(bindings, CXLType2AliasLaunchBinding, j);
      uint64_t offset = pointer - candidate->identity.base;

      if (pointer >= candidate->identity.base &&
          offset >= candidate->allocation_begin &&
          offset < candidate->allocation_end) {
        binding = candidate;
        break;
      }
    }
    if (binding) {
      if (!cxl_type2_model_consumer_parameter_is_read_root(function->consumer,
                                                           i)) {
        if (reason) {
          *reason = "model-destination-not-certified-read-root";
        }
        cxl_type2_alias_launch_bindings_release(ct2d, bindings, consumer);
        return NULL;
      }
      alias_address = binding->alias_begin + pointer - binding->identity.base -
                      binding->allocation_begin;
      memcpy(params[i], &alias_address, sizeof(alias_address));
      continue;
    }
    if (!cxl_type2_cuda_allocation_acquire_pageable_alias_for_address(
            &ct2d->cuda_allocations, pointer, consumer, &identity, &generation,
            &alias_address)) {
      if (classification == CXL_TYPE2_MODEL_PARAM_GPU_DESTINATION) {
        if (reason) {
          *reason = "model-destination-alias-unavailable";
        }
        cxl_type2_alias_launch_bindings_release(ct2d, bindings, consumer);
        return NULL;
      }
      continue;
    }
    if (!cxl_type2_model_consumer_parameter_is_read_root(function->consumer,
                                                         i)) {
      if (reason) {
        *reason = "model-destination-not-certified-read-root";
      }
      cxl_type2_cuda_allocation_release_pageable_alias(
          &ct2d->cuda_allocations, identity.base, identity.epoch, generation,
          consumer);
      cxl_type2_alias_launch_bindings_release(ct2d, bindings, consumer);
      return NULL;
    }
    CXLType2AliasLaunchBinding current = {
        .identity = identity,
        .generation = generation,
        .owns_inflight_ref = true,
    };
    CXLType2CudaAllocation *allocation = cxl_type2_cuda_allocation_find(
        &ct2d->cuda_allocations, identity.base, identity.epoch);
    const CXLType2CudaPageableAlias *alias =
        cxl_type2_cuda_allocation_pageable_alias_for_offset(
            allocation, pointer - identity.base);

    if (!alias) {
      cxl_type2_alias_launch_bindings_release(ct2d, bindings, consumer);
      return NULL;
    }
    current.allocation_begin = alias->destination_offset;
    current.allocation_end = alias->destination_offset + alias->logical_bytes;
    current.alias_begin = alias->device_alias;

    g_array_append_val(bindings, current);
    CXLType2CudaAllocationIdentity alias_identity;
    uint64_t alias_generation;

    if (!cxl_type2_cuda_allocation_identity_for_alias_address(
            &ct2d->cuda_allocations, alias_address, &alias_identity,
            &alias_generation) ||
        alias_identity.base != identity.base ||
        alias_identity.epoch != identity.epoch ||
        alias_generation != generation) {
      if (reason) {
        *reason = "model-alias-identity-mismatch";
      }
      cxl_type2_alias_launch_bindings_release(ct2d, bindings, consumer);
      return NULL;
    }
    memcpy(params[i], &alias_address, sizeof(alias_address));
  }
  return bindings;
}

static bool cxl_type2_alias_bindings_contain_identity(
    const GArray *bindings, CXLType2CudaAllocationIdentity identity) {
  if (!bindings) {
    return false;
  }
  for (size_t i = 0; i < bindings->len; i++) {
    const CXLType2AliasLaunchBinding *binding =
        &g_array_index(bindings, CXLType2AliasLaunchBinding, i);

    if (binding->identity.base == identity.base &&
        binding->identity.epoch == identity.epoch) {
      return true;
    }
  }
  return false;
}

static bool cxl_type2_alias_bindings_contain_alias(
    const GArray *bindings, const CXLType2AliasLaunchBinding *candidate) {
  if (!bindings || !candidate) {
    return false;
  }
  for (size_t i = 0; i < bindings->len; i++) {
    const CXLType2AliasLaunchBinding *binding =
        &g_array_index(bindings, CXLType2AliasLaunchBinding, i);

    if (binding->identity.base == candidate->identity.base &&
        binding->identity.epoch == candidate->identity.epoch &&
        binding->allocation_begin == candidate->allocation_begin &&
        binding->allocation_end == candidate->allocation_end &&
        binding->alias_begin == candidate->alias_begin) {
      return true;
    }
  }
  return false;
}

static void cxl_type2_graph_alias_bindings_unbind(
    CXLType2State *ct2d, uint32_t graph_exec_id,
    const CXLType2GraphNodeGenerationConsumers *node, GArray *bindings) {
  if (!bindings) {
    return;
  }
  for (size_t i = 0; i < bindings->len; i++) {
    const CXLType2AliasLaunchBinding *binding =
        &g_array_index(bindings, CXLType2AliasLaunchBinding, i);
    CXLType2CudaAllocationIdentity identity = binding->identity;
    CXLType2CudaAllocation *allocation = cxl_type2_cuda_allocation_find(
        &ct2d->cuda_allocations, identity.base, identity.epoch);
    bool duplicate = false;

    for (size_t j = 0; j < i; j++) {
      const CXLType2AliasLaunchBinding *previous =
          &g_array_index(bindings, CXLType2AliasLaunchBinding, j);

      duplicate = previous->identity.base == identity.base &&
                  previous->identity.epoch == identity.epoch;
      if (duplicate) {
        break;
      }
    }
    if (duplicate) {
      continue;
    }

    if (!allocation || !cxl_type2_cuda_allocation_unbind_graph_alias(
                           &ct2d->cuda_allocations, identity.base,
                           identity.epoch, binding->alias_begin)) {
      cxl_type2_generation_reuse_fail(ct2d, "graph-alias-unbind-failed");
      cxl_type2_model_supply_fail(
          ct2d, ct2d->gpu_cmd.call_id, "cleanup", "graph-alias-unbind-failed",
          UINT64_MAX, identity.base, identity.epoch,
          allocation ? allocation->content_generation : 0);
      continue;
    }
  }
  if (node) {
    for (size_t i = 0; i < bindings->len; i++) {
      const CXLType2AliasLaunchBinding *binding =
          &g_array_index(bindings, CXLType2AliasLaunchBinding, i);

      cxl_type2_log_model_graph_unbind(ct2d, graph_exec_id, node, binding);
    }
  }
  g_array_unref(bindings);
}

static bool
cxl_type2_model_alias_bind_graph_params(CXLType2State *ct2d,
                                        const CXLType2FunctionEntry *function,
                                        const HetGPUParamLayout *layout,
                                        void **params, GArray **bindings_out,
                                        const char **reason) {
  GArray *bindings =
      g_array_new(false, false, sizeof(CXLType2AliasLaunchBinding));

  if (reason) {
    *reason = NULL;
  }
  if (!bindings_out) {
    g_array_unref(bindings);
    return false;
  }
  if (ct2d->model_supply.route != CXL_TYPE2_MODEL_SUPPLY_CXL_DIRECT) {
    *bindings_out = bindings;
    return true;
  }
  for (uint32_t i = 0; i < layout->num_args; i++) {
    CXLType2CudaAllocationIdentity identity;
    CXLType2CudaAllocation *allocation;
    uint64_t pointer;
    uint64_t generation;
    uint64_t alias_address;
    CXLType2ModelParamClass classification;
    CXLType2AliasLaunchBinding binding = {0};
    const CXLType2CudaPageableAlias *alias = NULL;
    uint64_t allocation_offset = 0;

    if (!params[i] || layout->sizes[i] != sizeof(pointer)) {
      continue;
    }
    memcpy(&pointer, params[i], sizeof(pointer));
    classification = cxl_type2_model_param_classify(ct2d, pointer);
    if (cxl_type2_cuda_allocation_identity_for_address(&ct2d->cuda_allocations,
                                                       pointer, &identity)) {
      allocation = cxl_type2_cuda_allocation_find(
          &ct2d->cuda_allocations, identity.base, identity.epoch);
      allocation_offset = pointer - identity.base;
      alias = cxl_type2_cuda_allocation_pageable_alias_for_offset(
          allocation, allocation_offset);
    } else {
      if (!cxl_type2_cuda_allocation_identity_for_alias_address(
              &ct2d->cuda_allocations, pointer, &identity, &generation)) {
        if (classification == CXL_TYPE2_MODEL_PARAM_GPU_DESTINATION ||
            classification == CXL_TYPE2_MODEL_PARAM_ALIAS) {
          if (reason) {
            *reason = "model-destination-alias-unavailable";
          }
          cxl_type2_graph_alias_bindings_unbind(ct2d, 0, NULL, bindings);
          return false;
        }
        continue;
      }
      allocation = cxl_type2_cuda_allocation_find(
          &ct2d->cuda_allocations, identity.base, identity.epoch);
      for (alias = allocation ? allocation->pageable_alias : NULL; alias;
           alias = alias->next) {
        if (pointer >= alias->device_alias &&
            pointer - alias->device_alias < alias->logical_bytes) {
          allocation_offset =
              alias->destination_offset + pointer - alias->device_alias;
          break;
        }
      }
    }
    if (!allocation || !alias || !alias->device_alias) {
      if (classification == CXL_TYPE2_MODEL_PARAM_GPU_DESTINATION ||
          classification == CXL_TYPE2_MODEL_PARAM_ALIAS) {
        if (reason) {
          *reason = "model-destination-alias-unavailable";
        }
        cxl_type2_graph_alias_bindings_unbind(ct2d, 0, NULL, bindings);
        return false;
      }
      continue;
    }
    if (!cxl_type2_model_consumer_parameter_is_read_root(function->consumer,
                                                         i)) {
      if (reason) {
        *reason = "model-destination-not-certified-read-root";
      }
      cxl_type2_graph_alias_bindings_unbind(ct2d, 0, NULL, bindings);
      return false;
    }
    binding = (CXLType2AliasLaunchBinding){
        .identity = identity,
        .generation = allocation->content_generation,
        .allocation_begin = alias->destination_offset,
        .allocation_end = alias->destination_offset + alias->logical_bytes,
        .alias_begin = alias->device_alias,
    };
    alias_address =
        alias->device_alias + allocation_offset - alias->destination_offset;
    if (!cxl_type2_alias_bindings_contain_identity(bindings, identity) &&
        !cxl_type2_cuda_allocation_bind_graph_alias(
            &ct2d->cuda_allocations, identity.base, identity.epoch,
            alias->device_alias)) {
      cxl_type2_graph_alias_bindings_unbind(ct2d, 0, NULL, bindings);
      return false;
    }
    if (!cxl_type2_alias_bindings_contain_alias(bindings, &binding)) {
      g_array_append_val(bindings, binding);
    }
    memcpy(params[i], &alias_address, sizeof(alias_address));
  }
  *bindings_out = bindings;
  return true;
}

static void cxl_type2_graph_exec_generation_consumers_free(gpointer opaque) {
  CXLType2GraphExecGenerationConsumers *exec = opaque;

  if (!exec) {
    return;
  }
  for (size_t i = 0; i < exec->nodes->len; i++) {
    CXLType2GraphNodeGenerationConsumers *node =
        &g_array_index(exec->nodes, CXLType2GraphNodeGenerationConsumers, i);

    cxl_type2_graph_alias_bindings_unbind(exec->ct2d, exec->graph_exec_id, node,
                                          node->alias_bindings);
    g_array_unref(node->identities);
  }
  g_array_unref(exec->nodes);
  g_free(exec);
}

static CXLType2GraphExecGenerationConsumers *
cxl_type2_graph_generation_consumers_snapshot(CXLType2State *ct2d,
                                              HetGPUGraph graph,
                                              uint32_t graph_exec_id) {
  HetGPUState *hetgpu = &ct2d->gpu_info.hetgpu_state;
  CXLType2GraphExecGenerationConsumers *exec =
      g_new0(CXLType2GraphExecGenerationConsumers, 1);
  g_autofree HetGPUGraphNode *nodes = NULL;
  size_t count = 0;

  exec->ct2d = ct2d;
  exec->graph_exec_id = graph_exec_id;
  exec->nodes =
      g_array_new(false, false, sizeof(CXLType2GraphNodeGenerationConsumers));
  if (hetgpu_cuda_graph_get_nodes(hetgpu, graph, NULL, &count) !=
      CXL_GPU_SUCCESS) {
    goto failed;
  }
  nodes = g_try_new(HetGPUGraphNode, MAX(count, 1));
  if (!nodes || hetgpu_cuda_graph_get_nodes(hetgpu, graph, nodes, &count) !=
                    CXL_GPU_SUCCESS) {
    goto failed;
  }
  for (size_t i = 0; i < count; i++) {
    CudaKernelNodeParams params = {0};
    const HetGPUParamLayout *layout = NULL;
    CXLType2GraphNodeGenerationConsumers node;
    uint32_t function_id = UINT32_MAX;
    uint32_t graph_node_id;
    uint64_t graph_node_token;
    const char *consumer_reason = NULL;
    int node_type = -1;

    if (hetgpu_cuda_graph_node_get_type(hetgpu, nodes[i], &node_type) !=
        CXL_GPU_SUCCESS) {
      goto failed;
    }
    HetGPUGraphNodeRole node_role = hetgpu_cuda_graph_node_role(node_type);

    if (node_role == HETGPU_GRAPH_NODE_INERT_CONTROL) {
      continue;
    }
    if (node_role != HETGPU_GRAPH_NODE_KERNEL) {
      qemu_log("KIMI_MODEL_SUPPLY_GRAPH_REJECT graph_exec_id=%u "
               "node_index=%zu node_type=%d reason=memory-consumer-uncovered\n",
               graph_exec_id, i, node_type);
      cxl_type2_model_supply_fail(
          ct2d, ct2d->gpu_cmd.call_id, "driver-binding",
          "graph-memory-consumer-uncovered", i, 0, 0, 0);
      goto failed;
    }
    if (hetgpu_cuda_graph_kernel_node_get_params(hetgpu, nodes[i], &params) !=
            CXL_GPU_SUCCESS ||
        !params.func || params.extra ||
        hetgpu_get_param_layout(hetgpu, params.func, &layout) !=
            HETGPU_SUCCESS ||
        !params.kernel_params) {
      goto failed;
    }
    for (uint32_t j = 0; j < ct2d->gpu_cmd.num_functions; j++) {
      if (ct2d->gpu_cmd.functions[j].handle == params.func) {
        function_id = j;
        break;
      }
    }
    if (!cxl_type2_register_gpu_handle(
            &ct2d->gpu_cmd.graph_nodes, &ct2d->gpu_cmd.graph_nodes_capacity,
            &ct2d->gpu_cmd.num_graph_nodes, nodes[i], &graph_node_id)) {
      goto failed;
    }
    graph_node_token = graph_node_id;
    if (function_id == UINT32_MAX ||
        !cxl_type2_model_consumer_validate(
            ct2d, &ct2d->gpu_cmd.functions[function_id], layout,
            params.kernel_params, &consumer_reason)) {
      goto failed;
    }
    node = (CXLType2GraphNodeGenerationConsumers){
        .node = nodes[i],
        .function_id = function_id,
        .graph_node_token = graph_node_token,
        .identities = cxl_type2_generation_identities_from_params(
            ct2d, layout, params.kernel_params),
        .alias_bindings = NULL,
    };
    if (ct2d->model_supply.route == CXL_TYPE2_MODEL_SUPPLY_CXL_DIRECT) {
      if (!cxl_type2_model_alias_bind_graph_params(
              ct2d, &ct2d->gpu_cmd.functions[function_id], layout,
              params.kernel_params, &node.alias_bindings, &consumer_reason)) {
        cxl_type2_model_supply_fail(
            ct2d, ct2d->gpu_cmd.call_id, "driver-binding",
            consumer_reason ? consumer_reason : "graph-alias-bind-failed",
            UINT64_MAX, 0, 0, 0);
        g_array_unref(node.identities);
        goto failed;
      }
      if (node.alias_bindings->len) {
        cxl_type2_log_model_consumer_launch(
            ct2d, "bind", 0, function_id, layout, params.kernel_params,
            node.alias_bindings, CXL_TYPE2_CUDA_ALIAS_GRAPH, graph_exec_id,
            &graph_node_token, 1);
      }
    } else {
      node.alias_bindings =
          g_array_new(false, false, sizeof(CXLType2AliasLaunchBinding));
    }
    g_array_append_val(exec->nodes, node);
    if (ct2d->model_supply.route == CXL_TYPE2_MODEL_SUPPLY_CXL_DIRECT &&
        hetgpu_cuda_graph_kernel_node_set_params(hetgpu, nodes[i], &params) !=
            CXL_GPU_SUCCESS) {
      goto failed;
    }
  }
  return exec;

failed:
  cxl_type2_graph_exec_generation_consumers_free(exec);
  return NULL;
}

static void cxl_type2_graph_generation_consumers_set(
    CXLType2State *ct2d, uint32_t exec_id,
    CXLType2GraphExecGenerationConsumers *consumers) {
  if (!ct2d->graph_exec_generation_consumers) {
    ct2d->graph_exec_generation_consumers = g_ptr_array_new_with_free_func(
        cxl_type2_graph_exec_generation_consumers_free);
  }
  while (ct2d->graph_exec_generation_consumers->len <= exec_id) {
    g_ptr_array_add(ct2d->graph_exec_generation_consumers, NULL);
  }
  gpointer previous =
      g_ptr_array_index(ct2d->graph_exec_generation_consumers, exec_id);

  if (previous) {
    cxl_type2_graph_exec_generation_consumers_free(previous);
  }
  g_ptr_array_index(ct2d->graph_exec_generation_consumers, exec_id) = consumers;
}

static CXLType2GraphExecGenerationConsumers *
cxl_type2_graph_generation_consumers_get(CXLType2State *ct2d,
                                         uint32_t exec_id) {
  if (!ct2d->graph_exec_generation_consumers ||
      exec_id >= ct2d->graph_exec_generation_consumers->len) {
    return NULL;
  }
  return g_ptr_array_index(ct2d->graph_exec_generation_consumers, exec_id);
}

static void cxl_type2_log_model_graph_launch(CXLType2State *ct2d,
                                             uint64_t launch_id,
                                             uint32_t exec_id,
                                             const GArray *bindings) {
  CXLType2GraphExecGenerationConsumers *exec =
      cxl_type2_graph_generation_consumers_get(ct2d, exec_id);
  QList *nodes = qlist_new();

  if (!exec || !bindings->len) {
    return;
  }
  for (size_t i = 0; i < exec->nodes->len; i++) {
    CXLType2GraphNodeGenerationConsumers *node =
        &g_array_index(exec->nodes, CXLType2GraphNodeGenerationConsumers, i);
    uint32_t node_id = UINT32_MAX;

    for (uint32_t j = 0; j < ct2d->gpu_cmd.num_graph_nodes; j++) {
      if (ct2d->gpu_cmd.graph_nodes[j] == node->node) {
        node_id = j;
        break;
      }
    }
    qlist_append(nodes, qnum_from_uint(node_id));
  }
  for (size_t i = 0; i < bindings->len; i++) {
    const CXLType2AliasLaunchBinding *binding =
        &g_array_index(bindings, CXLType2AliasLaunchBinding, i);
    CXLType2CudaAllocation *allocation = cxl_type2_cuda_allocation_find(
        &ct2d->cuda_allocations, binding->identity.base,
        binding->identity.epoch);
    QDict *event = cxl_type2_model_supply_event(ct2d);

    qdict_put_str(event, "action", "launch");
    qdict_put_str(event, "result", "pass");
    cxl_type2_qdict_put_u64(event, "launch_id", launch_id);
    cxl_type2_qdict_put_u64(event, "destination_allocation_id",
                            binding->identity.base);
    cxl_type2_qdict_put_u64(event, "allocation_epoch", binding->identity.epoch);
    cxl_type2_qdict_put_u64(event, "content_generation", binding->generation);
    const CXLType2CudaPageableAlias *alias =
        cxl_type2_model_alias_for_binding(allocation, binding);

    qdict_put(event, "contributing_source_call_ids",
              cxl_type2_model_supply_source_calls(alias));
    cxl_type2_qdict_put_u64(event, "destination_cuda_token",
                            binding->alias_begin);
    qdict_put_str(event, "node_kind", "graph-kernel");
    qdict_put_str(event, "binding_kind", "graph-top-level");
    cxl_type2_qdict_put_u64(event, "graph_exec_id", exec_id);
    qdict_put(event, "graph_node_tokens", qobject_ref(nodes));
    qdict_put_null(event, "module_load_kind");
    qdict_put_null(event, "driver_input_cubin_sha256");
    qdict_put_null(event, "function_name");
    qdict_put_str(event, "static_consumer_certificate_sha256",
                  ct2d->model_supply.consumer_certificate.sha256);
    qdict_put(event, "visible_parameter_roots", qlist_new());
    cxl_type2_model_supply_emit("KIMI_MODEL_SUPPLY_CONSUMER", event);
  }
  qobject_unref(nodes);
  ct2d->model_supply.graph_launches++;
}

static bool cxl_type2_graph_generation_consumers_set_node(
    CXLType2State *ct2d, uint32_t exec_id, HetGPUGraphNode node_handle,
    uint32_t function_id, uint64_t graph_node_token, GArray *identities,
    GArray *alias_bindings) {
  CXLType2GraphExecGenerationConsumers *exec =
      cxl_type2_graph_generation_consumers_get(ct2d, exec_id);

  if (!exec) {
    return false;
  }
  for (size_t i = 0; i < exec->nodes->len; i++) {
    CXLType2GraphNodeGenerationConsumers *node =
        &g_array_index(exec->nodes, CXLType2GraphNodeGenerationConsumers, i);

    if (node->node == node_handle) {
      cxl_type2_graph_alias_bindings_unbind(ct2d, exec_id, node,
                                            node->alias_bindings);
      g_array_unref(node->identities);
      node->identities = identities;
      node->alias_bindings = alias_bindings;
      node->function_id = function_id;
      node->graph_node_token = graph_node_token;
      return true;
    }
  }
  CXLType2GraphNodeGenerationConsumers node = {
      .node = node_handle,
      .function_id = function_id,
      .graph_node_token = graph_node_token,
      .identities = identities,
      .alias_bindings = alias_bindings,
  };

  g_array_append_val(exec->nodes, node);
  return true;
}

static void cxl_type2_graph_generation_consumers_clear(CXLType2State *ct2d) {
  g_clear_pointer(&ct2d->graph_exec_generation_consumers, g_ptr_array_unref);
}

static void cxl_type2_generation_consume_graph(CXLType2State *ct2d,
                                               uint32_t exec_id,
                                               uint64_t call_id) {
  CXLType2GraphExecGenerationConsumers *exec =
      cxl_type2_graph_generation_consumers_get(ct2d, exec_id);

  if (!exec) {
    cxl_type2_generation_reuse_fail(ct2d,
                                    "graph-consumer-identity-unavailable");
    return;
  }
  for (size_t i = 0; i < exec->nodes->len; i++) {
    CXLType2GraphNodeGenerationConsumers *node =
        &g_array_index(exec->nodes, CXLType2GraphNodeGenerationConsumers, i);

    if (!cxl_type2_cuda_generation_consume(
            &ct2d->cuda_allocations,
            (const CXLType2CudaAllocationIdentity *)node->identities->data,
            node->identities->len, CXL_GPU_CMD_GRAPH_LAUNCH, call_id)) {
      cxl_type2_generation_reuse_fail(
          ct2d, ct2d->cuda_allocations.first_generation_error);
      return;
    }
  }
}

static GArray *cxl_type2_graph_alias_launch_acquire(CXLType2State *ct2d,
                                                    uint32_t exec_id) {
  CXLType2GraphExecGenerationConsumers *exec =
      cxl_type2_graph_generation_consumers_get(ct2d, exec_id);
  GArray *bindings =
      g_array_new(false, false, sizeof(CXLType2AliasLaunchBinding));

  if (!exec || ct2d->model_supply.route != CXL_TYPE2_MODEL_SUPPLY_CXL_DIRECT) {
    return bindings;
  }
  for (size_t i = 0; i < exec->nodes->len; i++) {
    CXLType2GraphNodeGenerationConsumers *node =
        &g_array_index(exec->nodes, CXLType2GraphNodeGenerationConsumers, i);

    for (size_t j = 0; j < node->alias_bindings->len; j++) {
      CXLType2AliasLaunchBinding binding =
          g_array_index(node->alias_bindings, CXLType2AliasLaunchBinding, j);
      bool acquired =
          cxl_type2_alias_bindings_contain_identity(bindings, binding.identity);

      binding.owns_inflight_ref = !acquired;
      if (!acquired) {
        CXLType2CudaAllocation *allocation = cxl_type2_cuda_allocation_find(
            &ct2d->cuda_allocations, binding.identity.base,
            binding.identity.epoch);
        uint64_t ignored_alias = 0;

        if (!allocation ||
            !cxl_type2_cuda_allocation_acquire_pageable_alias(
                &ct2d->cuda_allocations, binding.identity.base,
                binding.identity.epoch, allocation->content_generation,
                CXL_TYPE2_CUDA_ALIAS_GRAPH, &ignored_alias)) {
          cxl_type2_alias_launch_bindings_release(ct2d, bindings,
                                                  CXL_TYPE2_CUDA_ALIAS_GRAPH);
          return NULL;
        }
        binding.generation = allocation->content_generation;
      }
      if (!cxl_type2_alias_bindings_contain_alias(bindings, &binding)) {
        g_array_append_val(bindings, binding);
      } else if (binding.owns_inflight_ref) {
        if (!cxl_type2_cuda_allocation_release_pageable_alias(
                &ct2d->cuda_allocations, binding.identity.base,
                binding.identity.epoch, binding.generation,
                CXL_TYPE2_CUDA_ALIAS_GRAPH)) {
          cxl_type2_generation_reuse_fail(
              ct2d, "model-alias-consumer-release-failed");
        }
      }
    }
  }
  return bindings;
}

static CXLType2CudaCoverageKind cxl_type2_cuda_record_coverage(
    CXLType2State *ct2d, const uint64_t *destinations, const size_t *sizes,
    size_t count, uint64_t *classified_bytes) {
  CXLType2CudaCoverageResult result;
  uint64_t *batches;
  uint64_t *bytes;

  cxl_type2_cuda_destination_union_classify(
      &ct2d->cuda_allocations, destinations, sizes, count, &result);
  if (classified_bytes) {
    *classified_bytes = result.bytes;
  }
  switch (result.kind) {
  case CXL_TYPE2_CUDA_COVERAGE_WHOLE:
    batches = &ct2d->paired_case.allocation_whole_batches;
    bytes = &ct2d->paired_case.allocation_whole_bytes;
    break;
  case CXL_TYPE2_CUDA_COVERAGE_PARTIAL:
    batches = &ct2d->paired_case.allocation_partial_batches;
    bytes = &ct2d->paired_case.allocation_partial_bytes;
    break;
  case CXL_TYPE2_CUDA_COVERAGE_CROSS:
    batches = &ct2d->paired_case.allocation_cross_batches;
    bytes = &ct2d->paired_case.allocation_cross_bytes;
    break;
  case CXL_TYPE2_CUDA_COVERAGE_UNKNOWN:
    batches = &ct2d->paired_case.allocation_unknown_batches;
    bytes = &ct2d->paired_case.allocation_unknown_bytes;
    break;
  default:
    g_assert_not_reached();
  }
  if (!cxl_type2_cuda_counter_can_add(
          ct2d->paired_case.allocation_classified_batches, 1) ||
      !cxl_type2_cuda_counter_can_add(
          ct2d->paired_case.allocation_classified_bytes, result.bytes) ||
      !cxl_type2_cuda_counter_can_add(*batches, 1) ||
      !cxl_type2_cuda_counter_can_add(*bytes, result.bytes)) {
    cxl_type2_cuda_classifier_reject(
        ct2d, CXL_TYPE2_CUDA_CLASSIFIER_CONTRADICTED,
        CXL_TYPE2_CUDA_REJECTION_DESTINATION_OVERFLOW,
        CXL_GPU_CMD_SOURCE_REGISTER_BATCH_HTOD_DIRECT_ASYNC, true);
    return result.kind;
  }
  ct2d->paired_case.allocation_classified_batches++;
  ct2d->paired_case.allocation_classified_bytes += result.bytes;
  (*batches)++;
  *bytes += result.bytes;
  if (!result.available) {
    cxl_type2_cuda_classifier_reject(
        ct2d, CXL_TYPE2_CUDA_CLASSIFIER_UNAVAILABLE, result.reason,
        CXL_GPU_CMD_SOURCE_REGISTER_BATCH_HTOD_DIRECT_ASYNC, true);
  } else if (result.reason != CXL_TYPE2_CUDA_REJECTION_NONE) {
    cxl_type2_cuda_classifier_reject(
        ct2d, ct2d->paired_case.classifier_status, result.reason,
        CXL_GPU_CMD_SOURCE_REGISTER_BATCH_HTOD_DIRECT_ASYNC, true);
  }
  return result.kind;
}

static void cxl_type2_reset_case_summary(CXLType2State *ct2d) {
  int64_t span_begin_ns = cxl_type2_host_monotonic_ns();
  uint64_t retained_groups = 0;
  uint64_t retained_physicals = 0;
  CXLType2DirectRegistration *registration;
  CXLType2DirectPhysical *physical;
  CXLType2IntervalIdentity begin_identity =
      cxl_type2_interval_identity(ct2d->paired_case.active_first_sequence,
                                  "qemu", "case", "case-begin", 0, false);

  ct2d->generation_reuse_error = NULL;
  g_clear_pointer(&ct2d->model_supply.first_failure_stage, g_free);
  g_clear_pointer(&ct2d->model_supply.first_failure_detail, g_free);
  g_clear_pointer(&ct2d->model_supply.source_geometry, g_checksum_free);
  g_clear_pointer(&ct2d->model_supply.alias_geometry, g_checksum_free);
  ct2d->model_supply.source_geometry = g_checksum_new(G_CHECKSUM_SHA256);
  ct2d->model_supply.alias_geometry = g_checksum_new(G_CHECKSUM_SHA256);
  ct2d->model_supply.source_descriptor_count = 0;
  ct2d->model_supply.source_logical_bytes = 0;
  ct2d->model_supply.alias_builds = 0;
  ct2d->model_supply.alias_remaps = 0;
  ct2d->model_supply.alias_releases = 0;
  ct2d->model_supply.alias_logical_bytes = 0;
  ct2d->model_supply.logical_source_view_bytes = 0;
  ct2d->model_supply.page_collateral_bytes = 0;
  ct2d->model_supply.file_mapped_bytes = 0;
  ct2d->model_supply.host_composition_copy_bytes = 0;
  ct2d->model_supply.derived_boundary_pages = 0;
  ct2d->model_supply.derived_boundary_copy_bytes = 0;
  ct2d->model_supply.boundary_composition_wall_ns = 0;
  ct2d->model_supply.guard_bytes = 0;
  ct2d->model_supply.model_htod_calls = 0;
  ct2d->model_supply.model_htod_bytes = 0;
  ct2d->model_supply.logical_direct_bytes = 0;
  ct2d->model_supply.reserved_gpu_token_bytes = 0;
  ct2d->model_supply.normal_launches = 0;
  ct2d->model_supply.graph_launches = 0;
  ct2d->model_supply.normal_completions = 0;
  ct2d->model_supply.graph_completions = 0;
  ct2d->model_supply.completion_wait_calls = 0;
  ct2d->model_supply.completion_wait_wall_ns = 0;
  ct2d->model_supply.completion_wait_work_items = 0;
  ct2d->model_supply.completion_nonblocking_work_items = 0;
  ct2d->model_supply.consumer_wall_ns = 0;
  ct2d->model_supply.mapping_pin_acquires = 0;
  ct2d->model_supply.mapping_pin_releases = 0;
  ct2d->model_supply.poisoned = false;

  for (registration = ct2d->direct_registrations; registration;
       registration = registration->next) {
    retained_groups++;
  }
  for (physical = ct2d->direct_physicals; physical; physical = physical->next) {
    retained_physicals++;
  }

  cxl_type2_command_scope_reset(&ct2d->paired_case.case_command_scope,
                                span_begin_ns, begin_identity);
  ct2d->paired_case.decode_command_scope.active = false;
  ct2d->paired_case.active_command_sequence = 0;
  ct2d->paired_case.active_command_code = 0;
  ct2d->paired_case.active_cxl_request_count = 0;
  ct2d->paired_case.active_cxl_read_count = 0;
  ct2d->paired_case.active_cxl_write_count = 0;
  ct2d->paired_case.active_cxl_logical_bytes = 0;
  ct2d->paired_case.active_cxl_qemu_wall_ns = 0;
  ct2d->paired_case.active_cxl_response_count = 0;
  ct2d->paired_case.active_cxl_request_failures = 0;
  ct2d->paired_case.active_cxl_server_reported_latency_ns = 0;
  ct2d->paired_case.active_cxl_range_requests = 0;
  ct2d->paired_case.active_cxl_range_bytes = 0;
  ct2d->paired_case.active_cxl_wire_bytes = 0;
  ct2d->paired_case.active_direct_register_calls = 0;
  ct2d->paired_case.active_direct_unregister_calls = 0;
  ct2d->paired_case.active_direct_register_validate_ns = 0;
  ct2d->paired_case.active_direct_register_resolve_ns = 0;
  ct2d->paired_case.active_direct_register_acquire_ns = 0;
  ct2d->paired_case.active_direct_register_commit_ns = 0;
  ct2d->paired_case.active_direct_unregister_release_ns = 0;
  ct2d->paired_case.active_direct_physical_register_calls = 0;
  ct2d->paired_case.active_direct_physical_register_ns = 0;
  ct2d->paired_case.active_direct_registration_views = 0;
  ct2d->paired_case.active_direct_registration_bytes = 0;
  ct2d->paired_case.active_direct_registration_padding_bytes = 0;
  ct2d->paired_case.active_direct_registration_min_bytes = 0;
  ct2d->paired_case.active_direct_registration_max_bytes = 0;
  ct2d->paired_case.active_direct_registration_le_2m_calls = 0;
  ct2d->paired_case.active_direct_registration_2m_4m_calls = 0;
  ct2d->paired_case.active_direct_registration_4m_16m_calls = 0;
  ct2d->paired_case.active_direct_registration_16m_64m_calls = 0;
  ct2d->paired_case.active_direct_registration_gt_64m_calls = 0;
  ct2d->paired_case.active_direct_tile_extension_mappings = 0;
  ct2d->paired_case.active_direct_tile_extension_bytes = 0;
  ct2d->paired_case.active_direct_tile_unavailable_stops = 0;
  ct2d->paired_case.active_direct_tile_conflict_stops = 0;
  ct2d->paired_case.active_direct_tile_pin_failures = 0;
  ct2d->paired_case.active_direct_cross_mapping_groups = 0;
  ct2d->paired_case.active_direct_cross_mapping_members = 0;
  ct2d->paired_case.active_direct_registration_groups = 0;
  ct2d->paired_case.active_direct_group_members = 0;
  ct2d->paired_case.active_direct_max_group_members = 0;
  ct2d->paired_case.active_direct_retained_groups = retained_groups;
  ct2d->paired_case.active_direct_peak_retained_groups = retained_groups;
  ct2d->paired_case.active_direct_coalesced_views = 0;
  ct2d->paired_case.active_direct_max_registration_views = 0;
  ct2d->paired_case.active_direct_physical_boundaries = 0;
  ct2d->paired_case.active_direct_host_contiguous_boundaries = 0;
  ct2d->paired_case.active_direct_host_contiguous_following_bytes = 0;
  ct2d->paired_case.active_direct_cross_reg_boundaries = 0;
  ct2d->paired_case.active_direct_cross_reg_bytes = 0;
  ct2d->paired_case.active_direct_cross_reg_any_registered_boundaries = 0;
  ct2d->paired_case.active_direct_cross_reg_any_registered_bytes = 0;
  ct2d->paired_case.active_direct_cross_reg_both_registered_boundaries = 0;
  ct2d->paired_case.active_direct_cross_reg_both_registered_bytes = 0;
  ct2d->paired_case.active_direct_physical_unregister_calls = 0;
  ct2d->paired_case.active_direct_physical_unregister_ns = 0;
  ct2d->paired_case.active_direct_cache_hits = 0;
  ct2d->paired_case.active_direct_cross_case_hits = 0;
  ct2d->paired_case.active_direct_active_hits = 0;
  ct2d->paired_case.active_direct_cache_misses = 0;
  ct2d->paired_case.active_direct_revoke_releases = 0;
  ct2d->paired_case.active_direct_retained_physicals = retained_physicals;
  ct2d->paired_case.active_direct_peak_retained_physicals = retained_physicals;
  ct2d->paired_case.active_direct_logical_ranges = 0;
  ct2d->paired_case.active_direct_fragments = 0;
  ct2d->paired_case.active_direct_bytes = 0;
  ct2d->paired_case.active_direct_pending_copies = 0;
  ct2d->paired_case.active_direct_pending_bytes = 0;
  ct2d->paired_case.active_direct_peak_pending_copies = 0;
  ct2d->paired_case.active_direct_peak_pending_bytes = 0;
  ct2d->paired_case.active_payload_batches = 0;
  ct2d->paired_case.active_payload_source_bytes = 0;
  ct2d->paired_case.allocation_classified_batches = 0;
  ct2d->paired_case.allocation_classified_bytes = 0;
  ct2d->paired_case.allocation_whole_batches = 0;
  ct2d->paired_case.allocation_whole_bytes = 0;
  ct2d->paired_case.allocation_whole_single_span_batches = 0;
  ct2d->paired_case.allocation_whole_single_span_bytes = 0;
  ct2d->paired_case.allocation_partial_batches = 0;
  ct2d->paired_case.allocation_partial_bytes = 0;
  ct2d->paired_case.allocation_cross_batches = 0;
  ct2d->paired_case.allocation_cross_bytes = 0;
  ct2d->paired_case.allocation_unknown_batches = 0;
  ct2d->paired_case.allocation_unknown_bytes = 0;
  ct2d->paired_case.graph_exec_observed = 0;
  ct2d->paired_case.graph_all_kernel_flat = 0;
  ct2d->paired_case.graph_child_or_non_kernel = 0;
  ct2d->paired_case.graph_incomplete = 0;
  ct2d->paired_case.graph_pointer_sized_params = 0;
  ct2d->paired_case.graph_allocation_pointer_params = 0;
  ct2d->paired_case.graph_allocation_base_params = 0;
  ct2d->paired_case.classifier_status = CXL_TYPE2_CUDA_CLASSIFIER_AVAILABLE;
  ct2d->paired_case.first_rejection_reason = CXL_TYPE2_CUDA_REJECTION_NONE;
  ct2d->paired_case.first_rejection_command = 0;
  ct2d->paired_case.first_rejection_command_valid = false;
  ct2d->paired_case.active_htod_pool_hits = 0;
  ct2d->paired_case.active_htod_pool_misses = 0;
  ct2d->paired_case.active_htod_driver_allocations = 0;
  ct2d->paired_case.active_htod_driver_frees = 0;
  ct2d->paired_case.active_htod_pool_evictions = 0;
  ct2d->paired_case.active_htod_staging_pending_bytes = 0;
  ct2d->paired_case.active_htod_peak_staging_pending_bytes = 0;
  ct2d->paired_case.active_htod_peak_pooled_bytes = 0;
  ct2d->paired_case.active_stream_work_commands = 0;
  ct2d->paired_case.active_stream_sync_driver_calls = 0;
  ct2d->paired_case.active_elided_stream_syncs = 0;
  ct2d->paired_case.active_stream_sync_reason = CXL_GPU_STREAM_SYNC_PUBLIC_API;
  ct2d->paired_case.active_stream_sync_reason_valid = false;
  ct2d->paired_case.last_successful_stream_sync_wire = 0;
  ct2d->paired_case.last_command_was_successful_stream_sync = false;
}

static void cxl_type2_record_driver_scope(CXLType2State *ct2d,
                                          CXLType2CommandScopeLedger *scope,
                                          uint64_t call_id, uint32_t occurrence,
                                          const char *symbol,
                                          int64_t begin_host_ns,
                                          int64_t end_host_ns, int result) {
  uint64_t duration_ns;
  uint32_t symbol_index;

  if (!scope->active) {
    return;
  }
  if (ct2d->paired_case.active_command_sequence == 0) {
    cxl_type2_interval_fail(&scope->driver_intervals,
                            "driver-without-command-sequence");
    return;
  }
  if (call_id != ct2d->gpu_cmd.call_id) {
    cxl_type2_interval_fail(&scope->driver_intervals,
                            "driver-call-id-mismatch");
    return;
  }
  if (ct2d->paired_case.active_command_code >=
      G_N_ELEMENTS(scope->driver_calls_by_command)) {
    cxl_type2_interval_fail(&scope->driver_intervals,
                            "driver-command-out-of-range");
    return;
  }
  cxl_type2_interval_record(
      &scope->driver_intervals, begin_host_ns, end_host_ns,
      cxl_type2_interval_identity(ct2d->paired_case.active_command_sequence,
                                  "qemu", "driver", symbol, occurrence, true));
  duration_ns = end_host_ns >= begin_host_ns
                    ? (uint64_t)(end_host_ns - begin_host_ns)
                    : 0;
  scope->driver_calls_by_command[ct2d->paired_case.active_command_code]++;
  scope->driver_busy_ns_by_command[ct2d->paired_case.active_command_code] +=
      duration_ns;
  if (ct2d->paired_case.active_command_code == CXL_GPU_CMD_STREAM_SYNC) {
    if (!ct2d->paired_case.active_stream_sync_reason_valid) {
      scope->stream_sync_reason_error = "invalid-reason";
    } else {
      CXLGPUStreamSyncReason reason =
          ct2d->paired_case.active_stream_sync_reason;

      scope->stream_sync_driver_calls_by_reason[reason]++;
      scope->stream_sync_driver_busy_ns_by_reason[reason] += duration_ns;
    }
  }
  scope->driver_failures += result != 0;

  if (scope->driver_symbol_error) {
    return;
  }
  if (!symbol) {
    scope->driver_symbol_error = "missing-symbol";
    return;
  }
  for (symbol_index = 0; symbol_index < scope->driver_symbol_count;
       symbol_index++) {
    if (strcmp(scope->driver_symbols[symbol_index].symbol, symbol) == 0) {
      break;
    }
  }
  if (symbol_index == scope->driver_symbol_count) {
    if (symbol_index == G_N_ELEMENTS(scope->driver_symbols)) {
      scope->driver_symbol_error = "capacity-exceeded";
      return;
    }
    scope->driver_symbols[symbol_index].symbol = symbol;
    scope->driver_symbol_count++;
  }
  if (scope->driver_symbols[symbol_index].calls == UINT64_MAX ||
      scope->driver_symbols[symbol_index].busy_ns > UINT64_MAX - duration_ns ||
      (result != 0 &&
       scope->driver_symbols[symbol_index].failures == UINT64_MAX)) {
    scope->driver_symbol_error = "counter-overflow";
    return;
  }
  scope->driver_symbols[symbol_index].calls++;
  scope->driver_symbols[symbol_index].failures += result != 0;
  scope->driver_symbols[symbol_index].busy_ns += duration_ns;
}

static void cxl_type2_record_driver_interval(void *opaque, uint64_t call_id,
                                             uint32_t occurrence,
                                             const char *symbol,
                                             int64_t begin_host_ns,
                                             int64_t end_host_ns, int result) {
  CXLType2State *ct2d = opaque;

  if (ct2d->paired_case.active_case == CXL_GPU_CASE_NONE) {
    return;
  }
  cxl_type2_record_driver_scope(ct2d, &ct2d->paired_case.case_command_scope,
                                call_id, occurrence, symbol, begin_host_ns,
                                end_host_ns, result);
  cxl_type2_record_driver_scope(ct2d, &ct2d->paired_case.decode_command_scope,
                                call_id, occurrence, symbol, begin_host_ns,
                                end_host_ns, result);
}

static void cxl_type2_record_command_scope(CXLType2CommandScopeLedger *scope,
                                           uint32_t command, uint64_t sequence,
                                           int64_t begin_host_ns,
                                           int64_t end_host_ns,
                                           uint32_t result) {
  uint64_t duration_ns;

  if (!scope->active || command >= G_N_ELEMENTS(scope->command_calls)) {
    return;
  }

  duration_ns = end_host_ns >= begin_host_ns
                    ? (uint64_t)(end_host_ns - begin_host_ns)
                    : 0;
  scope->command_count++;
  scope->command_failures += result != CXL_GPU_SUCCESS;
  scope->command_busy_ns += duration_ns;
  if (scope->first_command_host_ns == 0) {
    scope->first_command_host_ns = begin_host_ns;
  }
  scope->last_command_host_ns = end_host_ns;
  scope->command_calls[command]++;
  scope->command_busy_ns_by_command[command] += duration_ns;
  cxl_type2_interval_record(
      &scope->command_intervals, begin_host_ns, end_host_ns,
      cxl_type2_interval_identity(sequence, "qemu", "command", NULL, command,
                                  true));
}

static void
cxl_type2_record_stream_sync_scope(CXLType2CommandScopeLedger *scope,
                                   bool reason_valid,
                                   CXLGPUStreamSyncReason reason, bool elided) {
  if (!scope->active) {
    return;
  }
  if (!reason_valid || reason >= CXL_GPU_STREAM_SYNC_REASON_COUNT) {
    scope->stream_sync_reason_error = "invalid-reason";
    return;
  }
  scope->stream_sync_calls_by_reason[reason]++;
  scope->stream_sync_elided_calls_by_reason[reason] += elided;
}

static void cxl_type2_record_case_command(
    CXLType2State *ct2d, uint32_t command, uint64_t sequence,
    int64_t begin_host_ns, int64_t end_host_ns, uint32_t result,
    bool stream_sync_reason_valid, CXLGPUStreamSyncReason stream_sync_reason,
    bool stream_sync_elided) {
  if (ct2d->paired_case.active_case == CXL_GPU_CASE_NONE) {
    return;
  }
  cxl_type2_record_command_scope(&ct2d->paired_case.case_command_scope, command,
                                 sequence, begin_host_ns, end_host_ns, result);
  cxl_type2_record_command_scope(&ct2d->paired_case.decode_command_scope,
                                 command, sequence, begin_host_ns, end_host_ns,
                                 result);
  if (command == CXL_GPU_CMD_STREAM_SYNC) {
    cxl_type2_record_stream_sync_scope(&ct2d->paired_case.case_command_scope,
                                       stream_sync_reason_valid,
                                       stream_sync_reason, stream_sync_elided);
    cxl_type2_record_stream_sync_scope(&ct2d->paired_case.decode_command_scope,
                                       stream_sync_reason_valid,
                                       stream_sync_reason, stream_sync_elided);
  }
}

static const char *
cxl_type2_stream_sync_reason_name(CXLGPUStreamSyncReason reason) {
  static const char *const names[] = {
      [CXL_GPU_STREAM_SYNC_PUBLIC_API] = "public-api",
      [CXL_GPU_STREAM_SYNC_DTOH_ASYNC_DRAIN] = "dtoh-async-drain",
      [CXL_GPU_STREAM_SYNC_MEMSET_D8_ASYNC_DRAIN] = "memset-d8-async-drain",
  };

  return reason < G_N_ELEMENTS(names) ? names[reason] : "invalid";
}

static void
cxl_type2_log_command_scope_summary(uint64_t run_binding, uint32_t case_kind,
                                    uint64_t epoch, const char *scope_name,
                                    CXLType2CommandScopeLedger *scope) {
  uint64_t command_driver_calls = 0;
  uint64_t command_driver_busy_ns = 0;
  uint64_t symbol_calls = 0;
  uint64_t symbol_failures = 0;
  uint64_t symbol_busy_ns = 0;
  uint64_t reason_calls = 0;
  uint64_t reason_elided_calls = 0;
  uint64_t reason_driver_calls = 0;
  uint64_t reason_driver_busy_ns = 0;
  const char *symbol_error = scope->driver_symbol_error;
  const char *reason_error = scope->stream_sync_reason_error;

  for (uint32_t command = 0; command < G_N_ELEMENTS(scope->command_calls);
       command++) {
    command_driver_calls += scope->driver_calls_by_command[command];
    command_driver_busy_ns += scope->driver_busy_ns_by_command[command];
    if (scope->command_calls[command] == 0) {
      continue;
    }
    qemu_log("KIMI_COMMAND_SUMMARY schema=qemu-command-summary-v2"
             " run_binding=%" PRIu64 " case=%s case_epoch=%" PRIu64
             " scope=%s command=0x%x calls=%" PRIu64 " busy_ns=%" PRIu64
             " driver_calls=%" PRIu64 " driver_busy_ns=%" PRIu64 "\n",
             run_binding, cxl_type2_paired_case_name(case_kind), epoch,
             scope_name, command, scope->command_calls[command],
             scope->command_busy_ns_by_command[command],
             scope->driver_calls_by_command[command],
             scope->driver_busy_ns_by_command[command]);
  }

  for (CXLGPUStreamSyncReason reason = CXL_GPU_STREAM_SYNC_PUBLIC_API;
       reason < CXL_GPU_STREAM_SYNC_REASON_COUNT; reason++) {
    reason_calls += scope->stream_sync_calls_by_reason[reason];
    reason_elided_calls += scope->stream_sync_elided_calls_by_reason[reason];
    reason_driver_calls += scope->stream_sync_driver_calls_by_reason[reason];
    reason_driver_busy_ns +=
        scope->stream_sync_driver_busy_ns_by_reason[reason];
    qemu_log("KIMI_STREAM_SYNC_REASON_SUMMARY"
             " schema=qemu-stream-sync-reason-summary-v1"
             " run_binding=%" PRIu64 " case=%s case_epoch=%" PRIu64
             " scope=%s reason=%s calls=%" PRIu64 " elided_calls=%" PRIu64
             " driver_calls=%" PRIu64 " driver_busy_ns=%" PRIu64 "\n",
             run_binding, cxl_type2_paired_case_name(case_kind), epoch,
             scope_name, cxl_type2_stream_sync_reason_name(reason),
             scope->stream_sync_calls_by_reason[reason],
             scope->stream_sync_elided_calls_by_reason[reason],
             scope->stream_sync_driver_calls_by_reason[reason],
             scope->stream_sync_driver_busy_ns_by_reason[reason]);
  }
  if (!reason_error &&
      (reason_calls != scope->command_calls[CXL_GPU_CMD_STREAM_SYNC] ||
       reason_driver_calls !=
           scope->driver_calls_by_command[CXL_GPU_CMD_STREAM_SYNC] ||
       reason_driver_busy_ns !=
           scope->driver_busy_ns_by_command[CXL_GPU_CMD_STREAM_SYNC] ||
       reason_elided_calls > reason_calls)) {
    reason_error = "totals-mismatch";
  }
  qemu_log("KIMI_STREAM_SYNC_REASON_TERMINAL"
           " schema=qemu-stream-sync-reason-terminal-v1"
           " run_binding=%" PRIu64 " case=%s case_epoch=%" PRIu64
           " scope=%s status=%s reason_count=%u calls=%" PRIu64
           " elided_calls=%" PRIu64 " driver_calls=%" PRIu64
           " driver_busy_ns=%" PRIu64 " error=%s\n",
           run_binding, cxl_type2_paired_case_name(case_kind), epoch,
           scope_name, reason_error ? "unavailable" : "available",
           CXL_GPU_STREAM_SYNC_REASON_COUNT, reason_calls, reason_elided_calls,
           reason_driver_calls, reason_driver_busy_ns,
           reason_error ? reason_error : "none");

  if (!symbol_error) {
    for (uint32_t index = 0; index < scope->driver_symbol_count; index++) {
      symbol_calls += scope->driver_symbols[index].calls;
      symbol_failures += scope->driver_symbols[index].failures;
      symbol_busy_ns += scope->driver_symbols[index].busy_ns;
    }
    if (symbol_calls != command_driver_calls ||
        symbol_failures != scope->driver_failures ||
        symbol_busy_ns != command_driver_busy_ns) {
      symbol_error = "totals-mismatch";
    }
  }
  if (!symbol_error) {
    for (uint32_t index = 0; index < scope->driver_symbol_count; index++) {
      qemu_log("KIMI_DRIVER_SYMBOL_SUMMARY"
               " schema=qemu-driver-symbol-summary-v1"
               " run_binding=%" PRIu64 " case=%s case_epoch=%" PRIu64
               " scope=%s symbol=%s calls=%" PRIu64 " failures=%" PRIu64
               " busy_ns=%" PRIu64 "\n",
               run_binding, cxl_type2_paired_case_name(case_kind), epoch,
               scope_name, scope->driver_symbols[index].symbol,
               scope->driver_symbols[index].calls,
               scope->driver_symbols[index].failures,
               scope->driver_symbols[index].busy_ns);
    }
  }
  qemu_log("KIMI_DRIVER_SYMBOL_TERMINAL"
           " schema=qemu-driver-symbol-terminal-v1"
           " run_binding=%" PRIu64 " case=%s case_epoch=%" PRIu64
           " scope=%s status=%s symbol_count=%u calls=%" PRIu64
           " failures=%" PRIu64 " busy_ns=%" PRIu64 " reason=%s\n",
           run_binding, cxl_type2_paired_case_name(case_kind), epoch,
           scope_name, symbol_error ? "unavailable" : "available",
           symbol_error ? 0 : scope->driver_symbol_count, command_driver_calls,
           scope->driver_failures, command_driver_busy_ns,
           symbol_error ? symbol_error : "none");

  cxl_type2_log_interval_summary(run_binding, case_kind, epoch, scope_name,
                                 "command", &scope->command_intervals);
  cxl_type2_log_interval_summary(run_binding, case_kind, epoch, scope_name,
                                 "driver", &scope->driver_intervals);
}

static void cxl_type2_finish_decode_command_scope(CXLType2State *ct2d,
                                                  int64_t end_host_ns,
                                                  uint64_t sequence,
                                                  bool terminal_observed) {
  CXLType2CommandScopeLedger *scope = &ct2d->paired_case.decode_command_scope;
  const char *operation = terminal_observed ? "decode-end" : "case-end";
  CXLType2IntervalIdentity end_identity = cxl_type2_interval_identity(
      sequence, "qemu", "decode", operation, 0, false);

  if (!terminal_observed) {
    cxl_type2_interval_fail(&scope->command_intervals, "decode-end-missing");
    cxl_type2_interval_fail(&scope->driver_intervals, "decode-end-missing");
  }
  cxl_type2_command_scope_finish(scope, end_host_ns, end_identity);
  cxl_type2_log_command_scope_summary(
      ct2d->paired_case.run_binding, ct2d->paired_case.active_case,
      ct2d->paired_case.active_epoch, "decode", scope);
}

static void cxl_type2_log_generation_reuse_summary(CXLType2State *ct2d,
                                                   uint64_t run_binding,
                                                   uint32_t case_kind,
                                                   uint64_t case_epoch) {
  CXLType2CudaAllocationTable *table = &ct2d->cuda_allocations;
  const char *terminal_reason = ct2d->generation_reuse_error
                                    ? ct2d->generation_reuse_error
                                    : table->first_generation_error;
  bool available = table->available && !terminal_reason;
  const char *reason = terminal_reason;
  const char *status = "unavailable";
  size_t entry_count = 0;

  if (!reason) {
    reason = table->available ? "none" : "allocation-ledger-unavailable";
  }
  if (available) {
    status = "available";
    entry_count = table->generation_count;
  }

  if (available) {
    for (size_t i = 0; i < table->generation_count; i++) {
      const CXLType2CudaGenerationRecord *record = &table->generations[i];
      char first_consumer[64] = "none";
      char last_consumer[64] = "none";
      const char *direct_demand_bytes =
          record->direct_consumer_count ? "unavailable" : "0";
      const char *direct_demand_status =
          record->direct_consumer_count ? "unavailable" : "available";
      const char *direct_demand_reason =
          record->direct_consumer_count ? "consumer-read-bytes-not-produced"
                                        : "none";
      const char *gpu_local_demand_bytes =
          record->gpu_local_consumer_count ? "unavailable" : "0";
      const char *gpu_local_demand_status =
          record->gpu_local_consumer_count ? "unavailable" : "available";
      const char *gpu_local_demand_reason = record->gpu_local_consumer_count
                                                ? "gpu-read-bytes-not-produced"
                                                : "none";
      char prefetch_bytes[32];
      char prefetch_enqueue_wall_ns[32];
      char prefetch_completion_wall_ns[32];
      const char *prefetch_cost_status =
          record->prefetch_completion_available ? "available" : "unavailable";
      const char *prefetch_cost_reason =
          record->prefetch_completion_available ? "none"
          : record->prefetch_count ? "prefetch-completion-boundary-not-produced"
                                   : "none";

      snprintf(prefetch_bytes, sizeof(prefetch_bytes), "%" PRIu64,
               record->prefetch_requested_bytes);
      snprintf(prefetch_enqueue_wall_ns, sizeof(prefetch_enqueue_wall_ns),
               "%" PRIu64, record->prefetch_enqueue_wall_ns);
      snprintf(prefetch_completion_wall_ns, sizeof(prefetch_completion_wall_ns),
               "%" PRIu64, record->promotion_wall_ns);
      if (record->consumer_observed) {
        snprintf(first_consumer, sizeof(first_consumer),
                 "opcode:0x%02x/call:%" PRIu64, record->first_consumer_opcode,
                 record->first_consumer_call_id);
        snprintf(last_consumer, sizeof(last_consumer),
                 "opcode:0x%02x/call:%" PRIu64, record->last_consumer_opcode,
                 record->last_consumer_call_id);
      }
      qemu_log(
          "KIMI_COHERENT_GENERATION_REUSE_ENTRY"
          " schema=coherent-generation-reuse-v1"
          " run_binding=%" PRIu64 " case=%s case_epoch=%" PRIu64
          " allocation_base=0x%" PRIx64 " epoch=%" PRIu64 " generation=%" PRIu64
          " unique_visible_population_bytes=%" PRIu64
          " direct_consumer_count=%" PRIu64 " direct_consumer_demand_bytes=%s"
          " direct_consumer_demand_status=%s"
          " direct_consumer_demand_reason=%s"
          " backing_kind=%s prefetch_count=%" PRIu64
          " prefetch_bytes=%s prefetch_enqueue_wall_ns=%s"
          " prefetch_completion_wall_ns=%s prefetch_cost_status=%s"
          " prefetch_cost_reason=%s promotion_count=%" PRIu64
          " promotion_bytes=%" PRIu64 " promotion_wall_ns=%" PRIu64
          " gpu_local_consumer_count=%" PRIu64
          " gpu_local_consumer_demand_bytes=%s"
          " gpu_local_consumer_demand_status=%s"
          " gpu_local_consumer_demand_reason=%s"
          " first_consumer=%s last_consumer=%s next_boundary=%s\n",
          run_binding, cxl_type2_paired_case_name(case_kind), case_epoch,
          record->allocation_base, record->epoch, record->generation,
          record->unique_visible_population_bytes,
          record->direct_consumer_count, direct_demand_bytes,
          direct_demand_status, direct_demand_reason,
          record->file_backed_system_uva ? "file-backed-system-uva" : "other",
          record->prefetch_count, prefetch_bytes, prefetch_enqueue_wall_ns,
          prefetch_completion_wall_ns, prefetch_cost_status,
          prefetch_cost_reason, record->promotion_count,
          record->promotion_bytes, record->promotion_wall_ns,
          record->gpu_local_consumer_count, gpu_local_demand_bytes,
          gpu_local_demand_status, gpu_local_demand_reason, first_consumer,
          last_consumer,
          cxl_type2_cuda_generation_boundary_name(record->next_boundary));
    }
  }
  qemu_log("KIMI_COHERENT_GENERATION_REUSE_TERMINAL"
           " schema=coherent-generation-reuse-v1"
           " run_binding=%" PRIu64 " case=%s case_epoch=%" PRIu64
           " status=%s entry_count=%zu reason=%s\n",
           run_binding, cxl_type2_paired_case_name(case_kind), case_epoch,
           status, entry_count, reason);
}

static void cxl_type2_log_model_supply_terminal(CXLType2State *ct2d,
                                                bool case_success,
                                                bool context_quiesced,
                                                int cleanup_status) {
  uint64_t live_graph_bindings = 0;
  uint64_t live_inflight_refs = 0;
  uint64_t live_aliases = 0;
  uint64_t live_source_views = 0;
  uint64_t live_source_fd_refs = 0;
  bool poisoned = ct2d->model_supply.poisoned || ct2d->direct_source_poisoned;
  const char *completion_wait_status;
  const char *completion_wait_reason;
  bool pass;
  QDict *event;

  for (size_t i = 0; i < ct2d->cuda_allocations.count; i++) {
    CXLType2CudaAllocation *allocation = &ct2d->cuda_allocations.entries[i];

    live_graph_bindings += allocation->graph_binding_refs;
    live_inflight_refs +=
        allocation->normal_inflight_refs + allocation->graph_inflight_refs;
    poisoned |= allocation->poisoned;
    if (allocation->pageable_alias) {
      live_aliases++;
      for (const CXLType2CudaPageableAlias *alias = allocation->pageable_alias;
           alias; alias = alias->next) {
        if (alias != allocation->pageable_alias) {
          live_aliases++;
        }
        live_source_views += alias->source_count;
        for (size_t j = 0; j < alias->source_count; j++) {
          live_source_fd_refs += alias->sources[j].fd >= 0;
        }
      }
    }
  }
  pass = case_success && !ct2d->model_supply.first_failure_stage &&
         context_quiesced && cleanup_status == 0 &&
         ct2d->model_supply.mapping_pin_acquires ==
             ct2d->model_supply.mapping_pin_releases &&
         !live_graph_bindings && !live_inflight_refs && !live_aliases &&
         !live_source_views && !live_source_fd_refs && !poisoned;

  if (!ct2d->model_supply.model_htod_calls &&
      !ct2d->model_supply.normal_launches &&
      !ct2d->model_supply.graph_launches) {
    completion_wait_status = "not-applicable";
    completion_wait_reason = "model-supply-work-not-submitted";
  } else if (ct2d->model_supply.completion_nonblocking_work_items) {
    completion_wait_status = "unavailable";
    completion_wait_reason = "model-supply-completed-by-nonblocking-query";
  } else if (!ct2d->model_supply.completion_wait_calls ||
             !ct2d->model_supply.completion_wait_work_items) {
    completion_wait_status = "unavailable";
    completion_wait_reason = "model-supply-completion-wait-not-observed";
  } else {
    completion_wait_status = "available";
    completion_wait_reason = NULL;
  }

  event = cxl_type2_model_supply_event(ct2d);
  qdict_put_str(event, "actual_route",
                cxl_type2_model_supply_route_name(ct2d->model_supply.route));
  qdict_put_str(event, "status", pass ? "pass" : "fail");
  if (ct2d->model_supply.first_failure_stage) {
    qdict_put_str(event, "first_failure_stage",
                  ct2d->model_supply.first_failure_stage);
  } else {
    qdict_put_null(event, "first_failure_stage");
  }
  cxl_type2_qdict_put_u64(event, "source_descriptor_count",
                          ct2d->model_supply.source_descriptor_count);
  cxl_type2_qdict_put_u64(event, "source_logical_bytes",
                          ct2d->model_supply.source_logical_bytes);
  qdict_put_str(event, "source_geometry_sha256",
                g_checksum_get_string(ct2d->model_supply.source_geometry));
  cxl_type2_qdict_put_u64(event, "logical_source_view_bytes",
                          ct2d->model_supply.logical_source_view_bytes);
  cxl_type2_qdict_put_u64(event, "page_collateral_bytes",
                          ct2d->model_supply.page_collateral_bytes);
  qdict_put_str(event, "alias_geometry_sha256",
                g_checksum_get_string(ct2d->model_supply.alias_geometry));
  cxl_type2_qdict_put_u64(event, "alias_builds",
                          ct2d->model_supply.alias_builds);
  cxl_type2_qdict_put_u64(event, "alias_remaps",
                          ct2d->model_supply.alias_remaps);
  cxl_type2_qdict_put_u64(event, "alias_releases",
                          ct2d->model_supply.alias_releases);
  cxl_type2_qdict_put_u64(event, "file_mapped_bytes",
                          ct2d->model_supply.file_mapped_bytes);
  cxl_type2_qdict_put_u64(event, "host_composition_copy_bytes",
                          ct2d->model_supply.host_composition_copy_bytes);
  cxl_type2_qdict_put_u64(event, "derived_boundary_pages",
                          ct2d->model_supply.derived_boundary_pages);
  cxl_type2_qdict_put_u64(event, "derived_boundary_copy_bytes",
                          ct2d->model_supply.derived_boundary_copy_bytes);
  cxl_type2_qdict_put_u64(event, "guard_bytes", ct2d->model_supply.guard_bytes);
  cxl_type2_qdict_put_u64(event, "model_htod_calls",
                          ct2d->model_supply.model_htod_calls);
  cxl_type2_qdict_put_u64(event, "model_htod_bytes",
                          ct2d->model_supply.model_htod_bytes);
  cxl_type2_qdict_put_u64(event, "logical_direct_bytes",
                          ct2d->model_supply.logical_direct_bytes);
  cxl_type2_qdict_put_u64(event, "reserved_gpu_token_bytes",
                          ct2d->model_supply.reserved_gpu_token_bytes);
  cxl_type2_qdict_put_u64(event, "normal_launches",
                          ct2d->model_supply.normal_launches);
  cxl_type2_qdict_put_u64(event, "normal_completions",
                          ct2d->model_supply.normal_completions);
  cxl_type2_qdict_put_u64(event, "graph_launches",
                          ct2d->model_supply.graph_launches);
  cxl_type2_qdict_put_u64(event, "graph_completions",
                          ct2d->model_supply.graph_completions);
  qdict_put_str(event, "completion_wait_status", completion_wait_status);
  if (completion_wait_reason) {
    qdict_put_str(event, "completion_wait_reason", completion_wait_reason);
  } else {
    qdict_put_null(event, "completion_wait_reason");
  }
  cxl_type2_qdict_put_u64(event, "completion_wait_calls",
                          ct2d->model_supply.completion_wait_calls);
  cxl_type2_qdict_put_u64(event, "completion_wait_wall_ns",
                          ct2d->model_supply.completion_wait_wall_ns);
  cxl_type2_qdict_put_u64(event, "completion_wait_work_items",
                          ct2d->model_supply.completion_wait_work_items);
  cxl_type2_qdict_put_u64(event, "completion_nonblocking_work_items",
                          ct2d->model_supply.completion_nonblocking_work_items);
  cxl_type2_qdict_put_u64(event, "consumer_wall_ns",
                          ct2d->model_supply.consumer_wall_ns);
  cxl_type2_qdict_put_u64(event, "live_graph_bindings", live_graph_bindings);
  cxl_type2_qdict_put_u64(event, "live_inflight_refs", live_inflight_refs);
  cxl_type2_qdict_put_u64(event, "live_aliases", live_aliases);
  cxl_type2_qdict_put_u64(event, "live_source_views", live_source_views);
  cxl_type2_qdict_put_u64(event, "live_source_fd_refs", live_source_fd_refs);
  cxl_type2_qdict_put_u64(event, "mapping_pin_acquires",
                          ct2d->model_supply.mapping_pin_acquires);
  cxl_type2_qdict_put_u64(event, "mapping_pin_releases",
                          ct2d->model_supply.mapping_pin_releases);
  qdict_put_bool(event, "context_quiesced", context_quiesced);
  qdict_put_int(event, "cleanup_status", cleanup_status);
  qdict_put_bool(event, "poisoned", poisoned);
  cxl_type2_model_supply_emit("KIMI_MODEL_SUPPLY_TERMINAL", event);
}

static void cxl_type2_log_case_summary(CXLType2State *ct2d,
                                       uint64_t run_binding, uint32_t case_kind,
                                       uint64_t epoch) {
  CXLType2CommandScopeLedger *command_scope =
      &ct2d->paired_case.case_command_scope;
  uint64_t *calls = command_scope->command_calls;
  uint64_t *busy = command_scope->command_busy_ns_by_command;
  uint64_t *driver_calls = command_scope->driver_calls_by_command;
  uint64_t *driver_busy = command_scope->driver_busy_ns_by_command;
  uint64_t host_window_ns =
      command_scope->last_command_host_ns >=
              command_scope->first_command_host_ns
          ? (uint64_t)(command_scope->last_command_host_ns -
                       command_scope->first_command_host_ns)
          : 0;

  cxl_type2_log_generation_reuse_summary(ct2d, run_binding, case_kind, epoch);

  qemu_log(
      "KIMI_CASE_SUMMARY run_binding=%" PRIu64 " case=%s epoch=%" PRIu64
      " commands=%" PRIu64 " failures=%" PRIu64 " busy_ns=%" PRIu64
      " host_window_ns=%" PRIu64 " launch_calls=%" PRIu64
      " launch_busy_ns=%" PRIu64 " htod_async_calls=%" PRIu64
      " htod_async_busy_ns=%" PRIu64 " dtoh_calls=%" PRIu64
      " dtoh_busy_ns=%" PRIu64 " stream_sync_calls=%" PRIu64
      " stream_sync_busy_ns=%" PRIu64 " htod_pending_copies=%" PRIu64
      " htod_pending_bytes=%" PRIu64 " htod_pool_hits=%" PRIu64
      " htod_pool_misses=%" PRIu64 " htod_driver_allocations=%" PRIu64
      " htod_driver_frees=%" PRIu64 " htod_pool_evictions=%" PRIu64
      " htod_staging_pending_bytes=%" PRIu64
      " htod_peak_staging_pending_bytes=%" PRIu64
      " htod_peak_pooled_bytes=%" PRIu64 " stream_work_commands=%" PRIu64
      " stream_sync_driver_calls=%" PRIu64 " elided_stream_syncs=%" PRIu64 "\n",
      run_binding, cxl_type2_paired_case_name(case_kind), epoch,
      command_scope->command_count, command_scope->command_failures,
      command_scope->command_busy_ns, host_window_ns,
      calls[CXL_GPU_CMD_LAUNCH_KERNEL], busy[CXL_GPU_CMD_LAUNCH_KERNEL],
      calls[CXL_GPU_CMD_MEM_COPY_HTOD_ASYNC],
      busy[CXL_GPU_CMD_MEM_COPY_HTOD_ASYNC], calls[CXL_GPU_CMD_MEM_COPY_DTOH],
      busy[CXL_GPU_CMD_MEM_COPY_DTOH], calls[CXL_GPU_CMD_STREAM_SYNC],
      busy[CXL_GPU_CMD_STREAM_SYNC], ct2d->htod_pending_copies,
      ct2d->htod_pending_bytes, ct2d->paired_case.active_htod_pool_hits,
      ct2d->paired_case.active_htod_pool_misses,
      ct2d->paired_case.active_htod_driver_allocations,
      ct2d->paired_case.active_htod_driver_frees,
      ct2d->paired_case.active_htod_pool_evictions,
      ct2d->paired_case.active_htod_staging_pending_bytes,
      ct2d->paired_case.active_htod_peak_staging_pending_bytes,
      ct2d->paired_case.active_htod_peak_pooled_bytes,
      ct2d->paired_case.active_stream_work_commands,
      ct2d->paired_case.active_stream_sync_driver_calls,
      ct2d->paired_case.active_elided_stream_syncs);

  qemu_log("KIMI_CXL_REQUEST_SUMMARY run_binding=%" PRIu64
           " case=%s epoch=%" PRIu64 " request_count=%" PRIu64
           " read_count=%" PRIu64 " write_count=%" PRIu64
           " logical_bytes=%" PRIu64 " qemu_wall_ns=%" PRIu64
           " response_count=%" PRIu64 " request_failures=%" PRIu64
           " server_reported_latency_ns=%" PRIu64 " range_requests=%" PRIu64
           " range_bytes=%" PRIu64 " wire_bytes=%" PRIu64 "\n",
           run_binding, cxl_type2_paired_case_name(case_kind), epoch,
           ct2d->paired_case.active_cxl_request_count,
           ct2d->paired_case.active_cxl_read_count,
           ct2d->paired_case.active_cxl_write_count,
           ct2d->paired_case.active_cxl_logical_bytes,
           ct2d->paired_case.active_cxl_qemu_wall_ns,
           ct2d->paired_case.active_cxl_response_count,
           ct2d->paired_case.active_cxl_request_failures,
           ct2d->paired_case.active_cxl_server_reported_latency_ns,
           ct2d->paired_case.active_cxl_range_requests,
           ct2d->paired_case.active_cxl_range_bytes,
           ct2d->paired_case.active_cxl_wire_bytes);

  {
    uint64_t live_sources = 0;
    uint64_t live_physicals = 0;
    uint64_t live_registration_groups = 0;
    uint64_t source_pending_refs = 0;
    CXLType2DirectSource *source;
    CXLType2DirectPhysical *physical;
    CXLType2DirectRegistration *registration;
    CXLType2CudaOpcodeSummary opcode_summary;
    char opcode_records[CXL_TYPE2_CUDA_OPCODE_RECORDS_CAPACITY];
    bool opcode_summary_valid;
    CXLType2CudaClassifierStatus classifier_status =
        ct2d->paired_case.classifier_status;
    CXLType2CudaRejectionReason rejection_reason =
        ct2d->paired_case.first_rejection_reason;
    uint32_t rejection_command = ct2d->paired_case.first_rejection_command;
    bool rejection_command_valid =
        ct2d->paired_case.first_rejection_command_valid;
    char rejection_command_name[16] = "none";

    opcode_summary_valid = cxl_type2_cuda_opcode_summary_build(
        command_scope->command_calls, opcode_records, sizeof(opcode_records),
        &opcode_summary);
    if (!opcode_summary_valid) {
      classifier_status = CXL_TYPE2_CUDA_CLASSIFIER_CONTRADICTED;
    } else if (!opcode_summary.estimated_materialize_complete &&
               classifier_status != CXL_TYPE2_CUDA_CLASSIFIER_CONTRADICTED) {
      classifier_status = CXL_TYPE2_CUDA_CLASSIFIER_UNAVAILABLE;
    }
    if (rejection_reason == CXL_TYPE2_CUDA_REJECTION_NONE &&
        !opcode_summary.estimated_materialize_complete) {
      rejection_reason = CXL_TYPE2_CUDA_REJECTION_OPCODE_UNKNOWN;
      rejection_command = opcode_summary.first_incomplete_command;
      rejection_command_valid = opcode_summary.first_incomplete_command_valid;
    }
    if (rejection_command_valid) {
      snprintf(rejection_command_name, sizeof(rejection_command_name), "0x%02X",
               rejection_command);
    }

    for (source = ct2d->direct_sources; source; source = source->next) {
      live_sources++;
      source_pending_refs += source->pending_refcount;
    }
    for (physical = ct2d->direct_physicals; physical;
         physical = physical->next) {
      live_physicals++;
    }
    for (registration = ct2d->direct_registrations; registration;
         registration = registration->next) {
      live_registration_groups++;
    }
    qemu_log(
        "KIMI_DIRECT_SOURCE_SUMMARY schema=direct-source-summary-v13"
        " run_binding=%" PRIu64 " case=%s case_epoch=%" PRIu64
        " policy_enabled=%u register_calls=%" PRIu64
        " register_busy_ns=%" PRIu64 " legacy_register_calls=%" PRIu64
        " legacy_register_busy_ns=%" PRIu64
        " fused_register_batch_calls=%" PRIu64
        " fused_register_batch_busy_ns=%" PRIu64 " unregister_calls=%" PRIu64
        " unregister_busy_ns=%" PRIu64 " direct_batch_calls=%" PRIu64
        " direct_batch_busy_ns=%" PRIu64 " driver_register_calls=%" PRIu64
        " driver_register_ns=%" PRIu64 " driver_unregister_calls=%" PRIu64
        " driver_unregister_ns=%" PRIu64 " register_validate_ns=%" PRIu64
        " register_resolve_ns=%" PRIu64 " register_acquire_ns=%" PRIu64
        " register_commit_ns=%" PRIu64 " unregister_release_ns=%" PRIu64
        " physical_register_calls=%" PRIu64 " physical_register_ns=%" PRIu64
        " registration_views=%" PRIu64 " registration_bytes=%" PRIu64
        " registration_padding_bytes=%" PRIu64
        " registration_min_bytes=%" PRIu64 " registration_max_bytes=%" PRIu64
        " registration_le_2m_calls=%" PRIu64
        " registration_2m_4m_calls=%" PRIu64
        " registration_4m_16m_calls=%" PRIu64
        " registration_16m_64m_calls=%" PRIu64
        " registration_gt_64m_calls=%" PRIu64
        " tile_extension_mappings=%" PRIu64 " tile_extension_bytes=%" PRIu64
        " tile_unavailable_stops=%" PRIu64 " tile_conflict_stops=%" PRIu64
        " tile_pin_failures=%" PRIu64 " cross_mapping_groups=%" PRIu64
        " cross_mapping_members=%" PRIu64 " registration_tile_size=%" PRIu64
        " registration_padding_limit=%" PRIu64
        " retained_registration_padding_bytes=%" PRIu64
        " registration_groups=%" PRIu64 " group_members=%" PRIu64
        " max_group_members=%" PRIu64 " peak_retained_groups=%" PRIu64
        " coalesced_views=%" PRIu64 " max_registration_views=%" PRIu64
        " physical_boundaries=%" PRIu64 " host_contiguous_boundaries=%" PRIu64
        " host_contiguous_following_bytes=%" PRIu64
        " cross_registration_boundaries=%" PRIu64
        " cross_registration_following_bytes=%" PRIu64
        " cross_registration_any_registered_boundaries=%" PRIu64
        " cross_registration_any_registered_following_bytes=%" PRIu64
        " cross_registration_both_registered_boundaries=%" PRIu64
        " cross_registration_both_registered_following_bytes=%" PRIu64
        " physical_unregister_calls=%" PRIu64 " physical_unregister_ns=%" PRIu64
        " cache_hits=%" PRIu64 " cross_case_hits=%" PRIu64
        " active_hits=%" PRIu64 " cache_misses=%" PRIu64
        " revoke_releases=%" PRIu64 " peak_retained_physicals=%" PRIu64
        " logical_ranges=%" PRIu64 " driver_fragments=%" PRIu64
        " direct_bytes=%" PRIu64 " direct_pending_copies=%" PRIu64
        " direct_pending_bytes=%" PRIu64 " direct_peak_pending_copies=%" PRIu64
        " direct_peak_pending_bytes=%" PRIu64 " payload_batches=%" PRIu64
        " payload_source_bytes=%" PRIu64 " live_sources=%" PRIu64
        " live_physicals=%" PRIu64 " live_registration_groups=%" PRIu64
        " pending_source_refs=%" PRIu64 " poisoned=%u"
        " allocation_live_count=%zu allocation_peak_live_count=%zu"
        " allocation_next_epoch=%" PRIu64
        " allocation_classified_batches=%" PRIu64
        " allocation_classified_bytes=%" PRIu64
        " allocation_whole_batches=%" PRIu64 " allocation_whole_bytes=%" PRIu64
        " allocation_whole_single_span_batches=%" PRIu64
        " allocation_whole_single_span_bytes=%" PRIu64
        " allocation_partial_batches=%" PRIu64
        " allocation_partial_bytes=%" PRIu64
        " allocation_cross_batches=%" PRIu64 " allocation_cross_bytes=%" PRIu64
        " allocation_unknown_batches=%" PRIu64
        " allocation_unknown_bytes=%" PRIu64 " graph_exec_observed=%" PRIu64
        " graph_all_kernel_flat=%" PRIu64 " graph_child_or_non_kernel=%" PRIu64
        " graph_incomplete=%" PRIu64 " graph_pointer_sized_params=%" PRIu64
        " graph_allocation_pointer_params=%" PRIu64
        " graph_allocation_base_params=%" PRIu64
        " ordinary_reader_commands=%" PRIu64
        " ordinary_writer_commands=%" PRIu64
        " ordinary_lifecycle_commands=%" PRIu64
        " ordinary_no_change_commands=%" PRIu64
        " ordinary_unknown_commands=%" PRIu64 " ordinary_opcode_records=%s"
        " estimated_materialize_bytes=%" PRIu64
        " estimated_materialize_complete=%u"
        " classifier_status=%s first_rejection_reason=%s"
        " first_rejection_command=%s\n",
        run_binding, cxl_type2_paired_case_name(case_kind), epoch,
        ct2d->cuda_direct_source,
        ct2d->paired_case.active_direct_register_calls,
        busy[CXL_GPU_CMD_SOURCE_REGISTER] +
            busy[CXL_GPU_CMD_SOURCE_REGISTER_BATCH_HTOD_DIRECT_ASYNC],
        calls[CXL_GPU_CMD_SOURCE_REGISTER], busy[CXL_GPU_CMD_SOURCE_REGISTER],
        calls[CXL_GPU_CMD_SOURCE_REGISTER_BATCH_HTOD_DIRECT_ASYNC],
        busy[CXL_GPU_CMD_SOURCE_REGISTER_BATCH_HTOD_DIRECT_ASYNC],
        ct2d->paired_case.active_direct_unregister_calls,
        busy[CXL_GPU_CMD_SOURCE_UNREGISTER],
        calls[CXL_GPU_CMD_BATCH_HTOD_DIRECT_ASYNC],
        busy[CXL_GPU_CMD_BATCH_HTOD_DIRECT_ASYNC],
        driver_calls[CXL_GPU_CMD_SOURCE_REGISTER],
        driver_busy[CXL_GPU_CMD_SOURCE_REGISTER],
        driver_calls[CXL_GPU_CMD_SOURCE_UNREGISTER],
        driver_busy[CXL_GPU_CMD_SOURCE_UNREGISTER],
        ct2d->paired_case.active_direct_register_validate_ns,
        ct2d->paired_case.active_direct_register_resolve_ns,
        ct2d->paired_case.active_direct_register_acquire_ns,
        ct2d->paired_case.active_direct_register_commit_ns,
        ct2d->paired_case.active_direct_unregister_release_ns,
        ct2d->paired_case.active_direct_physical_register_calls,
        ct2d->paired_case.active_direct_physical_register_ns,
        ct2d->paired_case.active_direct_registration_views,
        ct2d->paired_case.active_direct_registration_bytes,
        ct2d->paired_case.active_direct_registration_padding_bytes,
        ct2d->paired_case.active_direct_registration_min_bytes,
        ct2d->paired_case.active_direct_registration_max_bytes,
        ct2d->paired_case.active_direct_registration_le_2m_calls,
        ct2d->paired_case.active_direct_registration_2m_4m_calls,
        ct2d->paired_case.active_direct_registration_4m_16m_calls,
        ct2d->paired_case.active_direct_registration_16m_64m_calls,
        ct2d->paired_case.active_direct_registration_gt_64m_calls,
        ct2d->paired_case.active_direct_tile_extension_mappings,
        ct2d->paired_case.active_direct_tile_extension_bytes,
        ct2d->paired_case.active_direct_tile_unavailable_stops,
        ct2d->paired_case.active_direct_tile_conflict_stops,
        ct2d->paired_case.active_direct_tile_pin_failures,
        ct2d->paired_case.active_direct_cross_mapping_groups,
        ct2d->paired_case.active_direct_cross_mapping_members,
        ct2d->direct_registration_tile_size,
        ct2d->direct_registration_padding_limit,
        ct2d->direct_registration_padding_bytes,
        ct2d->paired_case.active_direct_registration_groups,
        ct2d->paired_case.active_direct_group_members,
        ct2d->paired_case.active_direct_max_group_members,
        ct2d->paired_case.active_direct_peak_retained_groups,
        ct2d->paired_case.active_direct_coalesced_views,
        ct2d->paired_case.active_direct_max_registration_views,
        ct2d->paired_case.active_direct_physical_boundaries,
        ct2d->paired_case.active_direct_host_contiguous_boundaries,
        ct2d->paired_case.active_direct_host_contiguous_following_bytes,
        ct2d->paired_case.active_direct_cross_reg_boundaries,
        ct2d->paired_case.active_direct_cross_reg_bytes,
        ct2d->paired_case.active_direct_cross_reg_any_registered_boundaries,
        ct2d->paired_case.active_direct_cross_reg_any_registered_bytes,
        ct2d->paired_case.active_direct_cross_reg_both_registered_boundaries,
        ct2d->paired_case.active_direct_cross_reg_both_registered_bytes,
        ct2d->paired_case.active_direct_physical_unregister_calls,
        ct2d->paired_case.active_direct_physical_unregister_ns,
        ct2d->paired_case.active_direct_cache_hits,
        ct2d->paired_case.active_direct_cross_case_hits,
        ct2d->paired_case.active_direct_active_hits,
        ct2d->paired_case.active_direct_cache_misses,
        ct2d->paired_case.active_direct_revoke_releases,
        ct2d->paired_case.active_direct_peak_retained_physicals,
        ct2d->paired_case.active_direct_logical_ranges,
        ct2d->paired_case.active_direct_fragments,
        ct2d->paired_case.active_direct_bytes,
        ct2d->paired_case.active_direct_pending_copies,
        ct2d->paired_case.active_direct_pending_bytes,
        ct2d->paired_case.active_direct_peak_pending_copies,
        ct2d->paired_case.active_direct_peak_pending_bytes,
        ct2d->paired_case.active_payload_batches,
        ct2d->paired_case.active_payload_source_bytes, live_sources,
        live_physicals, live_registration_groups, source_pending_refs,
        ct2d->direct_source_poisoned, ct2d->cuda_allocations.count,
        ct2d->cuda_allocations.peak_count, ct2d->cuda_allocations.next_epoch,
        ct2d->paired_case.allocation_classified_batches,
        ct2d->paired_case.allocation_classified_bytes,
        ct2d->paired_case.allocation_whole_batches,
        ct2d->paired_case.allocation_whole_bytes,
        ct2d->paired_case.allocation_whole_single_span_batches,
        ct2d->paired_case.allocation_whole_single_span_bytes,
        ct2d->paired_case.allocation_partial_batches,
        ct2d->paired_case.allocation_partial_bytes,
        ct2d->paired_case.allocation_cross_batches,
        ct2d->paired_case.allocation_cross_bytes,
        ct2d->paired_case.allocation_unknown_batches,
        ct2d->paired_case.allocation_unknown_bytes,
        ct2d->paired_case.graph_exec_observed,
        ct2d->paired_case.graph_all_kernel_flat,
        ct2d->paired_case.graph_child_or_non_kernel,
        ct2d->paired_case.graph_incomplete,
        ct2d->paired_case.graph_pointer_sized_params,
        ct2d->paired_case.graph_allocation_pointer_params,
        ct2d->paired_case.graph_allocation_base_params,
        opcode_summary.reader_commands, opcode_summary.writer_commands,
        opcode_summary.lifecycle_commands, opcode_summary.no_change_commands,
        opcode_summary.unknown_commands, opcode_records,
        opcode_summary.estimated_materialize_bytes,
        opcode_summary.estimated_materialize_complete,
        cxl_type2_cuda_classifier_status_name(classifier_status),
        cxl_type2_cuda_rejection_reason_name(rejection_reason),
        rejection_command_name);
  }

  cxl_type2_log_command_scope_summary(run_binding, case_kind, epoch, "case",
                                      command_scope);
}

static void cxl_type2_log_kimi_case_stage(uint64_t run_binding,
                                          uint32_t case_kind, uint64_t epoch,
                                          const char *operation,
                                          const char *state, int status,
                                          bool status_valid) {
  int64_t host_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);

  if (status_valid) {
    qemu_log("KIMI_CASE_STAGE run_binding=%" PRIu64 " case=%s epoch=%" PRIu64
             " operation=%s state=%s host_ns=%" PRId64 " status=%d\n",
             run_binding, cxl_type2_paired_case_name(case_kind), epoch,
             operation, state, host_ns, status);
  } else {
    qemu_log("KIMI_CASE_STAGE run_binding=%" PRIu64 " case=%s epoch=%" PRIu64
             " operation=%s state=%s host_ns=%" PRId64
             " status=not-applicable\n",
             run_binding, cxl_type2_paired_case_name(case_kind), epoch,
             operation, state, host_ns);
  }
}

static void cxl_type2_log_kimi_cleanup_progress(
    uint64_t run_binding, uint32_t case_kind, uint64_t epoch,
    const char *operation, const char *state, uint32_t completed,
    uint32_t total, int status, bool status_valid) {
  int64_t host_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);

  if (status_valid) {
    qemu_log("KIMI_CASE_STAGE run_binding=%" PRIu64 " case=%s epoch=%" PRIu64
             " operation=%s state=%s host_ns=%" PRId64
             " completed=%u total=%u status=%d\n",
             run_binding, cxl_type2_paired_case_name(case_kind), epoch,
             operation, state, host_ns, completed, total, status);
  } else {
    qemu_log("KIMI_CASE_STAGE run_binding=%" PRIu64 " case=%s epoch=%" PRIu64
             " operation=%s state=%s host_ns=%" PRId64
             " completed=%u total=%u status=not-applicable\n",
             run_binding, cxl_type2_paired_case_name(case_kind), epoch,
             operation, state, host_ns, completed, total);
  }
}

static bool cxl_type2_cleanup_progress_due(uint32_t completed, uint32_t total) {
  return completed == total || completed % 256 == 0;
}

static bool cxl_type2_command_requires_formal_case(uint32_t cmd) {
  switch (cmd) {
  case CXL_GPU_CMD_NOP:
  case CXL_GPU_CMD_INIT:
  case CXL_GPU_CMD_GET_DEVICE_COUNT:
  case CXL_GPU_CMD_GET_DEVICE:
  case CXL_GPU_CMD_GET_DEVICE_NAME:
  case CXL_GPU_CMD_GET_DEVICE_PROPS:
  case CXL_GPU_CMD_GET_TOTAL_MEM:
  case CXL_GPU_CMD_GET_DEVICE_ATTRIBUTE:
  case CXL_GPU_CMD_GET_ERROR_NAME:
  case CXL_GPU_CMD_GET_ERROR_STRING:
  case CXL_GPU_CMD_OBSERVATION_ANCHOR:
  case CXL_GPU_CMD_MODULE_GET_LOADING_MODE:
  case CXL_GPU_CMD_CASE_BEGIN:
  case CXL_GPU_CMD_CASE_END:
    return false;
  default:
    return true;
  }
}

typedef struct CXLType2CudaAttributeRequest {
  HetGPUState *hetgpu;
  int value;
} CXLType2CudaAttributeRequest;

typedef struct CXLType2CudaMemInfoRequest {
  HetGPUState *hetgpu;
  size_t free_bytes;
  size_t total_bytes;
} CXLType2CudaMemInfoRequest;

static int cxl_type2_cuda_query_attribute(void *opaque, int32_t attribute) {
  CXLType2CudaAttributeRequest *request = opaque;

  return hetgpu_cuda_device_get_attribute(request->hetgpu, attribute,
                                          &request->value);
}

static int cxl_type2_cuda_query_mem_info(void *opaque) {
  CXLType2CudaMemInfoRequest *request = opaque;

  return hetgpu_cuda_mem_get_info(request->hetgpu, &request->free_bytes,
                                  &request->total_bytes);
}

static bool cxl_type2_reserve_gpu_entries(void **entries, size_t *capacity,
                                          uint32_t count, size_t entry_size) {
  size_t next_capacity;
  void *grown;

  if (count == G_MAXUINT32) {
    return false;
  }
  if (count < *capacity) {
    return true;
  }

  next_capacity = *capacity ? *capacity : 64;
  while (next_capacity <= count) {
    if (next_capacity > G_MAXUINT32 / 2) {
      next_capacity = G_MAXUINT32;
    } else {
      next_capacity *= 2;
    }
  }
  grown = g_try_realloc_n(*entries, next_capacity, entry_size);
  if (!grown) {
    return false;
  }
  memset((uint8_t *)grown + *capacity * entry_size, 0,
         (next_capacity - *capacity) * entry_size);
  *entries = grown;
  *capacity = next_capacity;
  return true;
}

static bool cxl_type2_reserve_gpu_handle(void ***handles, size_t *capacity,
                                         uint32_t count) {
  return cxl_type2_reserve_gpu_entries((void **)handles, capacity, count,
                                       sizeof(**handles));
}

static bool cxl_type2_register_gpu_handle(void ***handles, size_t *capacity,
                                          uint32_t *count, void *handle,
                                          uint32_t *id) {
  if (!handle || !count || !id) {
    return false;
  }
  for (uint32_t i = 0; i < *count; i++) {
    if ((*handles)[i] == handle) {
      *id = i;
      return true;
    }
  }
  if (!cxl_type2_reserve_gpu_handle(handles, capacity, *count)) {
    return false;
  }
  (*handles)[*count] = handle;
  *id = *count;
  (*count)++;
  return true;
}

static bool cxl_type2_stream_from_wire(CXLType2State *ct2d, uint64_t wire,
                                       void **stream) {
  if (!stream) {
    return false;
  }
  if (wire == CXL_GPU_STREAM_WIRE_NULL || wire == CXL_GPU_STREAM_WIRE_LEGACY ||
      wire == CXL_GPU_STREAM_WIRE_PER_THREAD) {
    return cxl_type2_cuda_special_stream_from_wire(
        wire, ct2d->gpu_cmd.per_thread_stream, stream);
  }
  if (wire > UINT32_MAX || wire >= ct2d->gpu_cmd.num_streams ||
      !ct2d->gpu_cmd.streams[wire]) {
    return false;
  }
  *stream = ct2d->gpu_cmd.streams[wire];
  return true;
}

static int cxl_type2_htod_staging_free(CXLType2State *ct2d, void *data) {
  int result = hetgpu_cuda_mem_free_host(&ct2d->gpu_info.hetgpu_state, data);

  if (result == CXL_GPU_SUCCESS) {
    ct2d->htod_driver_frees++;
    if (ct2d->paired_case.active_case != CXL_GPU_CASE_NONE) {
      ct2d->paired_case.active_htod_driver_frees++;
    }
  }
  return result;
}

static int cxl_type2_htod_staging_acquire(CXLType2State *ct2d,
                                          size_t request_bytes, void **data,
                                          size_t *capacity, uint64_t *buffer_id,
                                          bool *pool_hit) {
  CXLType2HtoDStagingBuffer **best = NULL;
  CXLType2HtoDStagingBuffer **cursor = &ct2d->htod_staging_pool;
  CXLType2HtoDStagingBuffer *buffer;
  int result;

  while (*cursor) {
    if ((*cursor)->capacity >= request_bytes &&
        (!best || (*cursor)->capacity < (*best)->capacity)) {
      best = cursor;
    }
    cursor = &(*cursor)->next;
  }
  if (best) {
    buffer = *best;
    *best = buffer->next;
    ct2d->htod_pooled_buffers--;
    ct2d->htod_pooled_bytes -= buffer->capacity;
    ct2d->htod_pool_hits++;
    ct2d->paired_case.active_htod_pool_hits++;
    *data = buffer->data;
    *capacity = buffer->capacity;
    *buffer_id = buffer->id;
    *pool_hit = true;
    g_free(buffer);
    return CXL_GPU_SUCCESS;
  }

  ct2d->htod_pool_misses++;
  ct2d->paired_case.active_htod_pool_misses++;
  result = hetgpu_cuda_mem_host_alloc(&ct2d->gpu_info.hetgpu_state, data,
                                      request_bytes);
  if (result != CXL_GPU_SUCCESS) {
    return result;
  }
  ct2d->htod_driver_allocations++;
  ct2d->paired_case.active_htod_driver_allocations++;
  *capacity = request_bytes;
  *buffer_id = ++ct2d->next_htod_staging_id;
  *pool_hit = false;
  return CXL_GPU_SUCCESS;
}

static int cxl_type2_htod_staging_release(CXLType2State *ct2d, void *data,
                                          size_t capacity, uint64_t buffer_id,
                                          bool *pooled);

static gint cxl_type2_direct_physical_range_compare(gconstpointer a,
                                                    gconstpointer b,
                                                    gpointer opaque) {
  const CXLType2DirectPhysical *left = a;
  const CXLType2DirectPhysical *right = b;
  uintptr_t left_mapping = (uintptr_t)left->mapping;
  uintptr_t right_mapping = (uintptr_t)right->mapping;

  (void)opaque;
  if (left_mapping < right_mapping) {
    return -1;
  }
  if (left_mapping > right_mapping) {
    return 1;
  }
  if (left->mapping_offset < right->mapping_offset) {
    return left->length <= right->mapping_offset - left->mapping_offset ? -1
                                                                        : 0;
  }
  if (left->mapping_offset > right->mapping_offset) {
    return right->length <= left->mapping_offset - right->mapping_offset ? 1
                                                                         : 0;
  }
  return 0;
}

static void cxl_type2_direct_indexes_ensure(CXLType2State *ct2d) {
  if (!ct2d->direct_physical_ranges) {
    ct2d->direct_physical_ranges = g_tree_new_full(
        cxl_type2_direct_physical_range_compare, NULL, NULL, NULL);
  }
  if (!ct2d->direct_source_ids) {
    ct2d->direct_source_ids = g_hash_table_new(g_int64_hash, g_int64_equal);
  }
}

typedef struct CXLType2DirectPhysicalSearch {
  VirtioSharedMemoryMapping *mapping;
  hwaddr mapping_offset;
  CXLType2DirectPhysical *candidate;
} CXLType2DirectPhysicalSearch;

static gint cxl_type2_direct_physical_search(gconstpointer key,
                                             gconstpointer opaque) {
  const CXLType2DirectPhysical *physical = key;
  CXLType2DirectPhysicalSearch *search = (CXLType2DirectPhysicalSearch *)opaque;
  uintptr_t physical_mapping = (uintptr_t)physical->mapping;
  uintptr_t target_mapping = (uintptr_t)search->mapping;

  if (physical_mapping < target_mapping) {
    return 1;
  }
  if (physical_mapping > target_mapping) {
    search->candidate = (CXLType2DirectPhysical *)physical;
    return -1;
  }
  if (physical->mapping_offset > search->mapping_offset) {
    search->candidate = (CXLType2DirectPhysical *)physical;
    return -1;
  }
  if (physical->length <= search->mapping_offset - physical->mapping_offset) {
    return 1;
  }
  search->candidate = (CXLType2DirectPhysical *)physical;
  return 0;
}

static CXLType2DirectPhysical *
cxl_type2_direct_physical_find_at_or_after(CXLType2State *ct2d,
                                           VirtioSharedMemoryMapping *mapping,
                                           hwaddr mapping_offset) {
  CXLType2DirectPhysicalSearch search = {
      .mapping = mapping,
      .mapping_offset = mapping_offset,
  };
  CXLType2DirectPhysical *physical = g_tree_search(
      ct2d->direct_physical_ranges, cxl_type2_direct_physical_search, &search);

  physical = physical ? physical : search.candidate;
  return physical && physical->mapping == mapping ? physical : NULL;
}

static void cxl_type2_direct_physical_unlink(CXLType2State *ct2d,
                                             CXLType2DirectPhysical *physical) {
  CXLType2DirectPhysical **cursor;

  g_assert(g_tree_remove(ct2d->direct_physical_ranges, physical));
  virtio_shared_memory_unpin(physical->mapping);
  cursor = &ct2d->direct_physicals;
  while (*cursor != physical) {
    cursor = &(*cursor)->next;
  }
  *cursor = physical->next;
  g_assert(ct2d->direct_registration_padding_bytes >= physical->padding_bytes);
  ct2d->direct_registration_padding_bytes -= physical->padding_bytes;
  g_assert(ct2d->paired_case.active_direct_retained_physicals > 0);
  ct2d->paired_case.active_direct_retained_physicals--;
  g_free(physical);
}

static int
cxl_type2_direct_registration_release(CXLType2State *ct2d,
                                      CXLType2DirectRegistration *registration,
                                      bool revoke) {
  CXLType2DirectRegistration **cursor;
  CXLType2DirectPhysical *physical;
  HetGPUState *hetgpu = &ct2d->gpu_info.hetgpu_state;
  int64_t begin_ns;
  int result;

  g_assert(registration && registration->references == 0);
  if (registration->cuda_registered) {
    begin_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
    result =
        hetgpu_cuda_mem_host_unregister(hetgpu, registration->host_address);
    ct2d->paired_case.active_direct_physical_unregister_calls++;
    ct2d->paired_case.active_direct_physical_unregister_ns +=
        qemu_clock_get_ns(QEMU_CLOCK_HOST) - begin_ns;
    if (result != CXL_GPU_SUCCESS) {
      ct2d->direct_source_poisoned = true;
      return result;
    }
    registration->cuda_registered = false;
  }
  physical = registration->members;
  while (physical) {
    CXLType2DirectPhysical *next = physical->group_next;

    g_assert(physical->references == 0);
    cxl_type2_direct_physical_unlink(ct2d, physical);
    physical = next;
  }
  cursor = &ct2d->direct_registrations;
  while (*cursor != registration) {
    cursor = &(*cursor)->next;
  }
  *cursor = registration->next;
  g_assert(ct2d->paired_case.active_direct_retained_groups > 0);
  ct2d->paired_case.active_direct_retained_groups--;
  if (revoke) {
    ct2d->paired_case.active_direct_revoke_releases +=
        registration->member_count;
  }
  g_free(registration);
  return CXL_GPU_SUCCESS;
}

static int
cxl_type2_direct_registration_ensure(CXLType2State *ct2d,
                                     CXLType2DirectRegistration *registration) {
  int64_t begin_ns;
  int result;

  if (registration->revoke_pending || ct2d->direct_source_poisoned) {
    return CXL_GPU_ERROR_INVALID_VALUE;
  }
  if (registration->cuda_registered) {
    return CXL_GPU_SUCCESS;
  }
  begin_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
  result = hetgpu_cuda_mem_host_register(
      &ct2d->gpu_info.hetgpu_state, registration->host_address,
      registration->length, CXL_CUDA_MEMHOSTREGISTER_PORTABLE);
  ct2d->paired_case.active_direct_physical_register_calls++;
  ct2d->paired_case.active_direct_physical_register_ns +=
      qemu_clock_get_ns(QEMU_CLOCK_HOST) - begin_ns;
  if (result == CXL_GPU_SUCCESS) {
    registration->cuda_registered = true;
    ct2d->paired_case.active_direct_registration_views +=
        registration->view_count;
    ct2d->paired_case.active_direct_registration_bytes += registration->length;
    ct2d->paired_case.active_direct_registration_padding_bytes +=
        registration->padding_bytes;
    if (!ct2d->paired_case.active_direct_registration_min_bytes ||
        registration->length <
            ct2d->paired_case.active_direct_registration_min_bytes) {
      ct2d->paired_case.active_direct_registration_min_bytes =
          registration->length;
    }
    ct2d->paired_case.active_direct_registration_max_bytes =
        MAX(ct2d->paired_case.active_direct_registration_max_bytes,
            registration->length);
    if (registration->length <= 2 * MiB) {
      ct2d->paired_case.active_direct_registration_le_2m_calls++;
    } else if (registration->length <= 4 * MiB) {
      ct2d->paired_case.active_direct_registration_2m_4m_calls++;
    } else if (registration->length <= 16 * MiB) {
      ct2d->paired_case.active_direct_registration_4m_16m_calls++;
    } else if (registration->length <= 64 * MiB) {
      ct2d->paired_case.active_direct_registration_16m_64m_calls++;
    } else {
      ct2d->paired_case.active_direct_registration_gt_64m_calls++;
    }
    ct2d->paired_case.active_direct_registration_groups++;
    ct2d->paired_case.active_direct_group_members += registration->member_count;
    ct2d->paired_case.active_direct_max_group_members =
        MAX(ct2d->paired_case.active_direct_max_group_members,
            registration->member_count);
    if (registration->view_count) {
      ct2d->paired_case.active_direct_coalesced_views +=
          registration->view_count - 1;
    }
    ct2d->paired_case.active_direct_max_registration_views =
        MAX(ct2d->paired_case.active_direct_max_registration_views,
            registration->view_count);
  }
  return result;
}

static int cxl_type2_direct_physical_put(CXLType2State *ct2d,
                                         CXLType2DirectPhysical *physical) {
  g_assert(physical && physical->references > 0);
  g_assert(physical->registration && physical->registration->references > 0);
  physical->references--;
  physical->registration->references--;
  if (physical->registration->references == 0 &&
      physical->registration->revoke_pending) {
    int result = cxl_type2_direct_registration_release(
        ct2d, physical->registration, true);

    if (result != CXL_GPU_SUCCESS) {
      physical->references++;
      physical->registration->references++;
    }
    return result;
  }
  return CXL_GPU_SUCCESS;
}

static void
cxl_type2_direct_mapping_prepare_revoke(VirtioSharedMemoryMapping *mapping,
                                        void *opaque) {
  CXLType2State *ct2d = opaque;
  CXLType2DirectRegistration *registration;

  for (registration = ct2d->direct_registrations; registration;
       registration = registration->next) {
    CXLType2DirectPhysical *physical;

    for (physical = registration->members; physical;
         physical = physical->group_next) {
      if (physical->mapping == mapping) {
        registration->revoke_pending = true;
        break;
      }
    }
  }
  registration = ct2d->direct_registrations;
  while (registration) {
    CXLType2DirectRegistration *next = registration->next;

    if (registration->revoke_pending && registration->references == 0 &&
        cxl_type2_direct_registration_release(ct2d, registration, true) !=
            CXL_GPU_SUCCESS) {
      return;
    }
    registration = next;
  }
}

static int cxl_type2_direct_idle_cleanup(CXLType2State *ct2d) {
  CXLType2DirectRegistration *registration = ct2d->direct_registrations;

  while (registration) {
    CXLType2DirectRegistration *next = registration->next;

    if (registration->references == 0) {
      int result =
          cxl_type2_direct_registration_release(ct2d, registration, false);

      if (result != CXL_GPU_SUCCESS) {
        return result;
      }
    }
    registration = next;
  }
  return CXL_GPU_SUCCESS;
}

static int cxl_type2_direct_physicals_cleanup(CXLType2State *ct2d) {
  int result = cxl_type2_direct_idle_cleanup(ct2d);

  if (result != CXL_GPU_SUCCESS) {
    return result;
  }
  return ct2d->direct_physicals || ct2d->direct_registrations
             ? CXL_GPU_ERROR_INVALID_VALUE
             : CXL_GPU_SUCCESS;
}

static void cxl_type2_direct_indexes_destroy(CXLType2State *ct2d) {
  g_clear_pointer(&ct2d->direct_physical_ranges, g_tree_destroy);
  g_clear_pointer(&ct2d->direct_source_ids, g_hash_table_destroy);
}

static CXLType2DirectSource *cxl_type2_direct_source_find(CXLType2State *ct2d,
                                                          uint64_t source_id) {
  return ct2d->direct_source_ids
             ? g_hash_table_lookup(ct2d->direct_source_ids, &source_id)
             : NULL;
}

static int cxl_type2_direct_source_unregister(CXLType2State *ct2d,
                                              uint64_t source_id) {
  CXLType2DirectSource **cursor = &ct2d->direct_sources;
  CXLType2DirectSource *source;
  int first_error = CXL_GPU_SUCCESS;
  int64_t release_begin_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);

  while (*cursor && (*cursor)->source_id != source_id) {
    cursor = &(*cursor)->next;
  }
  if (!*cursor) {
    return CXL_GPU_ERROR_INVALID_VALUE;
  }
  source = *cursor;
  if ((ct2d->paired_case.active_epoch &&
       source->case_epoch != ct2d->paired_case.active_epoch) ||
      source->pending_refcount) {
    return CXL_GPU_ERROR_INVALID_VALUE;
  }
  for (uint32_t i = 0; i < source->run_count; i++) {
    int result;

    if (source->runs[i].source_fd >= 0) {
      close(source->runs[i].source_fd);
      source->runs[i].source_fd = -1;
    }
    if (!source->runs[i].physical) {
      continue;
    }
    result = cxl_type2_direct_physical_put(ct2d, source->runs[i].physical);
    if (result == CXL_GPU_SUCCESS) {
      source->runs[i].physical = NULL;
    } else if (first_error == CXL_GPU_SUCCESS) {
      first_error = result;
    }
  }
  if (first_error != CXL_GPU_SUCCESS) {
    return first_error;
  }
  bool removed =
      g_hash_table_remove(ct2d->direct_source_ids, &source->source_id);

  g_assert(removed);
  (void)removed;
  *cursor = source->next;
  g_free(source->ranges);
  g_free(source->runs);
  g_free(source);
  ct2d->paired_case.active_direct_unregister_release_ns +=
      qemu_clock_get_ns(QEMU_CLOCK_HOST) - release_begin_ns;
  return CXL_GPU_SUCCESS;
}

static int cxl_type2_direct_sources_release(CXLType2State *ct2d) {
  int result;
  CXLType2DirectRegistration *registration;
  CXLType2DirectPhysical *physical;

  while (ct2d->direct_sources) {
    uint64_t source_id = ct2d->direct_sources->source_id;

    result = cxl_type2_direct_source_unregister(ct2d, source_id);
    if (result != CXL_GPU_SUCCESS) {
      return result;
    }
  }
  for (registration = ct2d->direct_registrations; registration;
       registration = registration->next) {
    g_assert(registration->references == 0);
  }
  for (physical = ct2d->direct_physicals; physical; physical = physical->next) {
    g_assert(physical->references == 0);
  }
  return CXL_GPU_SUCCESS;
}

static int cxl_type2_direct_sources_cleanup(CXLType2State *ct2d) {
  int result = cxl_type2_direct_sources_release(ct2d);

  return result == CXL_GPU_SUCCESS ? cxl_type2_direct_physicals_cleanup(ct2d)
                                   : result;
}

typedef struct CXLType2DirectValidatedRun {
  hwaddr guest_phys_addr;
  uint64_t length;
  VirtioSharedMemoryMapping *mapping;
  const void *ordinary_host;
  uint64_t generation;
  hwaddr mapping_offset;
} CXLType2DirectValidatedRun;

static uint32_t
cxl_type2_direct_contiguous_run_end(const CXLType2DirectValidatedRun *validated,
                                    uint32_t run_count, uint32_t first) {
  uint32_t end = first + 1;

  while (end < run_count) {
    const CXLType2DirectValidatedRun *previous = &validated[end - 1];
    const CXLType2DirectValidatedRun *current = &validated[end];

    if (current->mapping != previous->mapping ||
        current->generation != previous->generation ||
        previous->mapping_offset > UINT64_MAX - previous->length ||
        current->mapping_offset !=
            previous->mapping_offset + previous->length ||
        previous->guest_phys_addr > UINT64_MAX - previous->length ||
        current->guest_phys_addr !=
            previous->guest_phys_addr + previous->length) {
      break;
    }
    end++;
  }
  return end;
}

static CXLType2DirectPhysical *
cxl_type2_direct_pending_find_at_or_after(GPtrArray *pending,
                                          VirtioSharedMemoryMapping *mapping,
                                          hwaddr mapping_offset) {
  CXLType2DirectPhysical *candidate = NULL;

  for (guint i = 0; i < pending->len; i++) {
    CXLType2DirectPhysical *physical = g_ptr_array_index(pending, i);

    if (!physical || physical->mapping != mapping ||
        (physical->mapping_offset <= mapping_offset &&
         physical->length <= mapping_offset - physical->mapping_offset)) {
      continue;
    }
    if (!candidate || physical->mapping_offset < candidate->mapping_offset) {
      candidate = physical;
    }
  }
  return candidate;
}

static gint cxl_type2_direct_physical_host_compare(gconstpointer a,
                                                   gconstpointer b) {
  const CXLType2DirectPhysical *left = *(CXLType2DirectPhysical *const *)a;
  const CXLType2DirectPhysical *right = *(CXLType2DirectPhysical *const *)b;

  return cxl_gpu_direct_host_address_order((uintptr_t)left->host_address,
                                           (uintptr_t)right->host_address);
}

static void cxl_type2_log_direct_source_register_failure(
    CXLType2State *ct2d, const char *failure_stage, uint64_t failure_index,
    int result, uint64_t payload_bytes) {
  qemu_log("KIMI_DIRECT_SOURCE_REGISTER_FAILURE"
           " schema=direct-source-register-failure-v1"
           " run_binding=%" PRIu64 " case=%s case_epoch=%" PRIu64
           " call_id=%" PRIu64 " stage=%s index_valid=%u"
           " index=%" PRIu64 " result=%d payload_bytes=%" PRIu64 "\n",
           ct2d->paired_case.run_binding,
           cxl_type2_paired_case_name(ct2d->paired_case.active_case),
           ct2d->paired_case.active_epoch, ct2d->gpu_cmd.call_id,
           failure_stage ? failure_stage : "unknown", failure_index != SIZE_MAX,
           failure_index == SIZE_MAX ? 0 : failure_index, result,
           payload_bytes);
}

static int cxl_type2_direct_source_register(
    CXLType2State *ct2d, uint64_t payload_bytes, uint64_t *source_id_out,
    bool allow_ordinary_source, bool *ordinary_source_out,
    const char **failure_stage_out, uint64_t *failure_index_out) {
  CXLGPUSourceRegisterV1 header;
  CXLType2DirectValidatedRun *validated = NULL;
  CXLType2DirectSource *source = NULL;
  VirtioSharedMemory *shmem;
  MemoryRegion *dax_mr;
  const uint8_t *range_base;
  GArray *resolved_runs = NULL;
  GArray *mapped_runs = NULL;
  CXLType2DirectRangeLayout *range_layouts = NULL;
  uint64_t fail_index;
  int result = CXL_GPU_ERROR_INVALID_VALUE;
  int64_t phase_begin_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
  uint64_t target_page_size = qemu_target_page_size();
  uint32_t file_run_count = 0;
  uint32_t ordinary_run_count = 0;

  *failure_stage_out = NULL;
  *failure_index_out = SIZE_MAX;
  if (ordinary_source_out) {
    *ordinary_source_out = false;
  }
  if (!ct2d->cuda_direct_source) {
    *failure_stage_out = "policy-disabled";
    return CXL_GPU_ERROR_INVALID_VALUE;
  }
  if (ct2d->direct_source_poisoned) {
    *failure_stage_out = "source-poisoned";
    return CXL_GPU_ERROR_INVALID_VALUE;
  }
  if (!ct2d->direct_source_fs) {
    *failure_stage_out = "source-fs-missing";
    return CXL_GPU_ERROR_INVALID_VALUE;
  }
  if (!source_id_out || !ct2d->paired_case.active_epoch) {
    *failure_stage_out = "case-inactive";
    return CXL_GPU_ERROR_INVALID_VALUE;
  }
  if (!current_cpu) {
    *failure_stage_out = "cpu-context-missing";
    return CXL_GPU_ERROR_INVALID_VALUE;
  }
  if (!vhost_user_fs_pci_get_dax(ct2d->direct_source_fs, &shmem, &dax_mr)) {
    *failure_stage_out = "dax-unavailable";
    return CXL_GPU_ERROR_INVALID_VALUE;
  }
  cpu_synchronize_state(current_cpu);
  cxl_type2_direct_indexes_ensure(ct2d);
  if (!cxl_gpu_source_register_validate(ct2d->gpu_cmd.batch_data,
                                        CXL_GPU_BATCH_DATA_SIZE, payload_bytes,
                                        &header, &fail_index)) {
    *failure_stage_out = "wire-contract";
    *failure_index_out = fail_index;
    return CXL_GPU_ERROR_INVALID_VALUE;
  }
  range_base = ct2d->gpu_cmd.batch_data + sizeof(header);
  resolved_runs = g_array_new(false, false, sizeof(CXLType2DirectValidatedRun));
  range_layouts = g_new0(CXLType2DirectRangeLayout, header.range_count);
  for (uint32_t i = 0; i < header.range_count; i++) {
    CXLGPUSourceVirtualRangeV1 wire_range;
    uint64_t cursor;
    uint64_t remaining;

    memcpy(&wire_range, range_base + (uint64_t)i * sizeof(wire_range),
           sizeof(wire_range));
    if (resolved_runs->len > UINT32_MAX) {
      *failure_stage_out = "run-capacity";
      *failure_index_out = i;
      goto out;
    }
    range_layouts[i].first_run = resolved_runs->len;
    range_layouts[i].length = wire_range.length;
    cursor = wire_range.guest_virtual_address;
    remaining = wire_range.length;
    while (remaining) {
      hwaddr page = cpu_get_phys_page_debug(current_cpu, cursor);
      uint64_t page_offset = cursor & (target_page_size - 1);
      uint64_t chunk = MIN(remaining, target_page_size - page_offset);
      CXLType2DirectValidatedRun run;

      if (page == (hwaddr)-1 || page > UINT64_MAX - page_offset) {
        *failure_stage_out = "gva-translate";
        *failure_index_out = i;
        goto out;
      }
      run = (CXLType2DirectValidatedRun){
          .guest_phys_addr = page + page_offset,
          .length = chunk,
      };
      if (resolved_runs->len > range_layouts[i].first_run) {
        CXLType2DirectValidatedRun *previous = &g_array_index(
            resolved_runs, CXLType2DirectValidatedRun, resolved_runs->len - 1);

        if (previous->guest_phys_addr <= UINT64_MAX - previous->length &&
            previous->guest_phys_addr + previous->length ==
                run.guest_phys_addr &&
            previous->length <= UINT64_MAX - run.length) {
          previous->length += run.length;
        } else {
          g_array_append_val(resolved_runs, run);
        }
      } else {
        g_array_append_val(resolved_runs, run);
      }
      cursor += chunk;
      remaining -= chunk;
    }
    if (resolved_runs->len - range_layouts[i].first_run > UINT32_MAX) {
      *failure_stage_out = "run-capacity";
      *failure_index_out = i;
      goto out;
    }
    range_layouts[i].run_count =
        resolved_runs->len - range_layouts[i].first_run;
  }
  if (resolved_runs->len > UINT32_MAX) {
    *failure_stage_out = "run-capacity";
    goto out;
  }
  ct2d->paired_case.active_direct_register_validate_ns +=
      qemu_clock_get_ns(QEMU_CLOCK_HOST) - phase_begin_ns;
  phase_begin_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);

  /* Split contiguous guest runs at actual RAM and DAX mapping boundaries. */
  mapped_runs = g_array_new(false, false, sizeof(CXLType2DirectValidatedRun));
  for (uint32_t range_index = 0; range_index < header.range_count;
       range_index++) {
    uint32_t old_first = range_layouts[range_index].first_run;
    uint32_t old_count = range_layouts[range_index].run_count;

    range_layouts[range_index].first_run = mapped_runs->len;
    for (uint32_t old_index = 0; old_index < old_count; old_index++) {
      CXLType2DirectValidatedRun original = g_array_index(
          resolved_runs, CXLType2DirectValidatedRun, old_first + old_index);
      hwaddr guest_phys_addr = original.guest_phys_addr;
      uint64_t remaining = original.length;

      while (remaining) {
        CXLType2DirectValidatedRun run = {
            .guest_phys_addr = guest_phys_addr,
        };
        hwaddr translated = 0;
        hwaddr translated_len = remaining;
        MemoryRegion *mr = address_space_translate(
            &address_space_memory, guest_phys_addr, &translated,
            &translated_len, false, MEMTXATTRS_UNSPECIFIED);

        if (!translated_len) {
          *failure_stage_out = "address-translate";
          *failure_index_out = old_first + old_index;
          goto out;
        }
        run.length = MIN(remaining, translated_len);
        if (mr != dax_mr) {
          if (!allow_ordinary_source || !ordinary_source_out ||
              !memory_region_is_ram(mr)) {
            *failure_stage_out = "address-translate";
            *failure_index_out = old_first + old_index;
            goto out;
          }
          run.ordinary_host =
              (const uint8_t *)memory_region_get_ram_ptr(mr) + translated;
          ordinary_run_count++;
        } else {
          VirtioSharedMemoryMapping *mapping =
              virtio_find_shmem_map(shmem, translated, 1);
          uint64_t mapping_offset;

          if (!mapping || translated < mapping->offset) {
            *failure_stage_out = "mapping-lookup";
            *failure_index_out = old_first + old_index;
            goto out;
          }
          if (mapping->revoke_pending) {
            *failure_stage_out = "mapping-revoking";
            *failure_index_out = old_first + old_index;
            goto out;
          }
          mapping_offset = translated - mapping->offset;
          run.length = MIN(run.length, mapping->len - mapping_offset);
          run.mapping = mapping;
          run.generation = mapping->generation;
          run.mapping_offset = translated;
          file_run_count++;
          for (uint32_t j = 0; j < mapped_runs->len; j++) {
            CXLType2DirectValidatedRun *previous =
                &g_array_index(mapped_runs, CXLType2DirectValidatedRun, j);

            if (previous->mapping == mapping &&
                ranges_overlap(previous->mapping_offset, previous->length,
                               translated, run.length) &&
                (previous->mapping_offset != translated ||
                 previous->length != run.length)) {
              *failure_stage_out = "request-run-overlap";
              *failure_index_out = old_first + old_index;
              goto out;
            }
          }
        }
        if (mapped_runs->len == UINT32_MAX) {
          *failure_stage_out = "run-capacity";
          *failure_index_out = range_index;
          goto out;
        }
        g_array_append_val(mapped_runs, run);
        guest_phys_addr += run.length;
        remaining -= run.length;
      }
    }
    range_layouts[range_index].run_count =
        mapped_runs->len - range_layouts[range_index].first_run;
  }
  g_array_free(resolved_runs, true);
  resolved_runs = g_steal_pointer(&mapped_runs);
  validated = (CXLType2DirectValidatedRun *)resolved_runs->data;
  if (ordinary_run_count &&
      (!file_run_count ||
       ct2d->model_supply.route != CXL_TYPE2_MODEL_SUPPLY_CXL_DIRECT)) {
    *ordinary_source_out = true;
    result = CXL_GPU_SUCCESS;
    goto out;
  }
  ct2d->paired_case.active_direct_register_resolve_ns +=
      qemu_clock_get_ns(QEMU_CLOCK_HOST) - phase_begin_ns;
  phase_begin_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);

  source = g_new0(CXLType2DirectSource, 1);
  source->case_epoch = ct2d->paired_case.active_epoch;
  source->logical_bytes = header.logical_bytes;
  source->pageable_alias =
      ct2d->model_supply.route == CXL_TYPE2_MODEL_SUPPLY_CXL_DIRECT;
  for (uint32_t i = 0; i < resolved_runs->len; i++) {
    bool duplicate = false;

    for (uint32_t j = 0; j < i; j++) {
      if (validated[i].mapping &&
          validated[j].mapping == validated[i].mapping &&
          validated[j].mapping_offset == validated[i].mapping_offset &&
          validated[j].length == validated[i].length) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate) {
      source->unique_dmap_bytes += validated[i].length;
    }
  }
  source->range_count = header.range_count;
  source->ranges = g_steal_pointer(&range_layouts);
  if (source->pageable_alias) {
    source->runs = g_try_new0(CXLType2DirectRun, resolved_runs->len);
    if (!source->runs) {
      result = CXL_GPU_ERROR_OUT_OF_MEMORY;
      *failure_stage_out = "source-view-allocate";
      goto rollback;
    }
    source->run_count = resolved_runs->len;
    for (uint32_t i = 0; i < source->run_count; i++) {
      source->runs[i].source_fd = -1;
    }
    for (uint32_t i = 0; i < source->run_count; i++) {
      CXLType2DirectRun *run = &source->runs[i];
      VirtioSharedMemoryMapping *pinned_mapping = NULL;
      uint64_t pinned_generation = 0;
      uint64_t mapping_file_offset = 0;
      uint64_t mapping_length = 0;
      void *mapping_host = NULL;
      struct stat source_stat;
      const CXLType2ModelMember *member;
      int source_result;

      run->source_fd = -1;
      run->length = validated[i].length;
      if (validated[i].ordinary_host) {
        run->ordinary_host = validated[i].ordinary_host;
        continue;
      }
      if (validated[i].mapping->allow_write) {
        result = CXL_GPU_ERROR_INVALID_VALUE;
        *failure_stage_out = "source-view-writable";
        *failure_index_out = i;
        goto rollback;
      }
      source_result = virtio_shared_memory_pin_range(
          shmem, validated[i].mapping_offset, validated[i].length,
          &pinned_mapping, &pinned_generation, &mapping_host, NULL, NULL);
      if (source_result != 0) {
        result = CXL_GPU_ERROR_INVALID_VALUE;
        *failure_stage_out = "source-view-pin";
        *failure_index_out = i;
        goto rollback;
      }
      ct2d->model_supply.mapping_pin_acquires++;
      if (pinned_mapping != validated[i].mapping ||
          pinned_generation != validated[i].generation) {
        virtio_shared_memory_unpin(pinned_mapping);
        ct2d->model_supply.mapping_pin_releases++;
        result = CXL_GPU_ERROR_INVALID_VALUE;
        *failure_stage_out = "source-view-generation";
        *failure_index_out = i;
        goto rollback;
      }
      source_result = virtio_shared_memory_mapping_dup_source(
          pinned_mapping, pinned_generation, &run->source_fd,
          &mapping_file_offset, &mapping_length);
      if (source_result != 0) {
        virtio_shared_memory_unpin(pinned_mapping);
        ct2d->model_supply.mapping_pin_releases++;
        result = CXL_GPU_ERROR_INVALID_VALUE;
        *failure_stage_out = "source-view-duplicate";
        *failure_index_out = i;
        goto rollback;
      }
      if (validated[i].mapping_offset < pinned_mapping->offset ||
          validated[i].mapping_offset - pinned_mapping->offset >
              UINT64_MAX - mapping_file_offset) {
        virtio_shared_memory_unpin(pinned_mapping);
        ct2d->model_supply.mapping_pin_releases++;
        result = CXL_GPU_ERROR_INVALID_VALUE;
        *failure_stage_out = "source-view-offset";
        *failure_index_out = i;
        goto rollback;
      }
      if (fstat(run->source_fd, &source_stat) != 0) {
        virtio_shared_memory_unpin(pinned_mapping);
        ct2d->model_supply.mapping_pin_releases++;
        result = CXL_GPU_ERROR_INVALID_VALUE;
        *failure_stage_out = "source-view-stat";
        *failure_index_out = i;
        goto rollback;
      }
      run->file_offset = mapping_file_offset + validated[i].mapping_offset -
                         pinned_mapping->offset;
      run->mapping_generation = pinned_generation;
      run->source_device = source_stat.st_dev;
      run->source_inode = source_stat.st_ino;
      run->source_size = source_stat.st_size;
      run->source_mode = source_stat.st_mode;
      member = cxl_type2_model_member_find(
          &ct2d->model_supply.member_manifest, run->source_device,
          run->source_inode, run->source_size, run->source_mode,
          run->file_offset, run->length);
      if (!member ||
          member->logical_cxl_offset >
              UINT64_MAX - ct2d->model_aperture.offset ||
          run->file_offset > UINT64_MAX - member->logical_cxl_offset -
                                 ct2d->model_aperture.offset) {
        virtio_shared_memory_unpin(pinned_mapping);
        ct2d->model_supply.mapping_pin_releases++;
        result = CXL_GPU_ERROR_INVALID_VALUE;
        *failure_stage_out = "member-range-admission";
        *failure_index_out = i;
        goto rollback;
      }
      run->logical_cxl_offset = ct2d->model_aperture.offset +
                                member->logical_cxl_offset + run->file_offset;
      run->model_member_index =
          member - ct2d->model_supply.member_manifest.members;
      if (!cxl_type2_range_contains(ct2d->model_aperture.offset,
                                    ct2d->model_aperture.size,
                                    run->logical_cxl_offset, run->length)) {
        virtio_shared_memory_unpin(pinned_mapping);
        ct2d->model_supply.mapping_pin_releases++;
        result = CXL_GPU_ERROR_INVALID_VALUE;
        *failure_stage_out = "aperture-capacity";
        *failure_index_out = i;
        goto rollback;
      }
      if (!cxl_type2_fabric_access_allowed(ct2d, run->logical_cxl_offset,
                                           run->length, false, false) ||
          !cxl_type2_memsim_request(ct2d, CXL_OP_RANGE_READ,
                                    run->logical_cxl_offset, run->length, NULL,
                                    NULL)) {
        virtio_shared_memory_unpin(pinned_mapping);
        ct2d->model_supply.mapping_pin_releases++;
        result = CXL_GPU_ERROR_INVALID_VALUE;
        *failure_stage_out = "fabric-request";
        *failure_index_out = i;
        goto rollback;
      }
      virtio_shared_memory_unpin(pinned_mapping);
      ct2d->model_supply.mapping_pin_releases++;
    }
    goto views_done;
  }
  {
    GArray *views = g_array_new(false, false, sizeof(CXLType2DirectRun));
    GPtrArray *pending = g_ptr_array_new();
    GPtrArray *new_groups = g_ptr_array_new();
    uint32_t *wire_first_view = g_new(uint32_t, resolved_runs->len);
    uint32_t *wire_view_count = g_new0(uint32_t, resolved_runs->len);
    uint64_t pending_padding_bytes = 0;

    /* Pin every missing mapping segment before choosing CUDA ranges. */
    for (uint32_t i = 0; i < resolved_runs->len;) {
      uint32_t group_end =
          cxl_type2_direct_contiguous_run_end(validated, resolved_runs->len, i);
      hwaddr cursor = validated[i].mapping_offset;
      uint64_t group_length = 0;

      for (uint32_t j = i; j < group_end; j++) {
        group_length += validated[j].length;
      }
      while (group_length) {
        CXLType2DirectPhysical *physical =
            cxl_type2_direct_physical_find_at_or_after(
                ct2d, validated[i].mapping, cursor);
        CXLType2DirectPhysical *pending_physical =
            cxl_type2_direct_pending_find_at_or_after(
                pending, validated[i].mapping, cursor);
        uint64_t segment_length;

        if (pending_physical && (!physical || pending_physical->mapping_offset <
                                                  physical->mapping_offset)) {
          physical = pending_physical;
        }

        if (physical && physical->mapping_offset <= cursor) {
          if (physical->generation != validated[i].generation ||
              physical->mapping->revoke_pending ||
              (physical->registration &&
               physical->registration->revoke_pending)) {
            result = CXL_GPU_ERROR_INVALID_VALUE;
            *failure_stage_out = "registration-revoking";
            *failure_index_out = i;
            goto view_rollback;
          }
          segment_length =
              MIN(group_length,
                  physical->length - (cursor - physical->mapping_offset));
          if (physical->registration) {
            if (cxl_gpu_direct_epoch_is_cross_case(
                    physical->last_case_epoch,
                    ct2d->paired_case.active_epoch)) {
              ct2d->paired_case.active_direct_cross_case_hits++;
              physical->last_case_epoch = ct2d->paired_case.active_epoch;
            }
            if (physical->references == 0) {
              ct2d->paired_case.active_direct_cache_hits++;
            } else {
              ct2d->paired_case.active_direct_active_hits++;
            }
          }
        } else {
          VirtioSharedMemoryMapping *pinned_mapping;
          uint64_t pinned_generation;
          uint64_t following_offset;
          uint64_t padding_budget = 0;
          uint64_t registration_offset;
          uint64_t registration_request_length;
          uint64_t host_page_size = qemu_real_host_page_size();
          uint64_t pin_length;
          void *mapping_host;

          segment_length =
              physical ? MIN(group_length, physical->mapping_offset - cursor)
                       : group_length;
          if (validated[i].mapping->offset >
              UINT64_MAX - validated[i].mapping->len) {
            result = CXL_GPU_ERROR_INVALID_VALUE;
            *failure_stage_out = "mapping-bounds";
            *failure_index_out = i;
            goto view_rollback;
          }
          following_offset = physical ? physical->mapping_offset
                                      : validated[i].mapping->offset +
                                            validated[i].mapping->len;
          if (ct2d->direct_registration_padding_bytes <
                  ct2d->direct_registration_padding_limit &&
              pending_padding_bytes <
                  ct2d->direct_registration_padding_limit -
                      ct2d->direct_registration_padding_bytes) {
            padding_budget = ct2d->direct_registration_padding_limit -
                             ct2d->direct_registration_padding_bytes -
                             pending_padding_bytes;
          }
          if (!cxl_gpu_direct_page_registration_span(
                  validated[i].mapping->offset, validated[i].mapping->len,
                  cursor, segment_length, following_offset, host_page_size,
                  &registration_offset, &registration_request_length)) {
            result = CXL_GPU_ERROR_INVALID_VALUE;
            *failure_stage_out = "registration-page-span";
            *failure_index_out = i;
            goto view_rollback;
          }
          pin_length = cxl_gpu_direct_registration_length(
              validated[i].mapping->offset, validated[i].mapping->len,
              registration_offset, registration_request_length,
              following_offset, ct2d->direct_registration_tile_size,
              padding_budget);
          if (!pin_length) {
            result = CXL_GPU_ERROR_INVALID_VALUE;
            *failure_stage_out = "registration-tile";
            *failure_index_out = i;
            goto view_rollback;
          }
          if (virtio_shared_memory_pin_range(
                  shmem, registration_offset, pin_length, &pinned_mapping,
                  &pinned_generation, &mapping_host,
                  cxl_type2_direct_mapping_prepare_revoke, ct2d) != 0) {
            *failure_stage_out = "mapping-pin";
            *failure_index_out = i;
            goto view_rollback;
          }
          if (pinned_mapping != validated[i].mapping ||
              pinned_generation != validated[i].generation) {
            *failure_stage_out = "mapping-generation";
            *failure_index_out = i;
            virtio_shared_memory_unpin(pinned_mapping);
            goto view_rollback;
          }
          if ((uintptr_t)mapping_host % qemu_real_host_page_size()) {
            *failure_stage_out = "host-map-alignment";
            *failure_index_out = i;
            virtio_shared_memory_unpin(pinned_mapping);
            goto view_rollback;
          }
          physical = g_new0(CXLType2DirectPhysical, 1);
          physical->mapping = pinned_mapping;
          physical->generation = pinned_generation;
          physical->last_case_epoch = ct2d->paired_case.active_epoch;
          physical->mapping_offset = registration_offset;
          physical->length = pin_length;
          physical->padding_bytes = pin_length - segment_length;
          physical->host_address = mapping_host;
          pending_padding_bytes += physical->padding_bytes;
          g_ptr_array_add(pending, physical);
        }
        cursor += segment_length;
        group_length -= segment_length;
      }
      i = group_end;
    }

    /* Logical range order does not define the host DAX layout. */
    g_ptr_array_sort(pending, cxl_type2_direct_physical_host_compare);

    /* Adjacent ranges in one pinned mapping share one registration. */
    for (guint first = 0; first < pending->len;) {
      CXLType2DirectPhysical *member = g_ptr_array_index(pending, first);
      CXLType2DirectRegistration *registration;
      uintptr_t base = (uintptr_t)member->host_address;
      uint64_t registration_length = member->length;
      uint64_t registration_padding = member->padding_bytes;
      guint end = first + 1;
      bool cross_mapping = false;

      while (end < pending->len) {
        CXLType2DirectPhysical *next = g_ptr_array_index(pending, end);

        /* Merge on host-VA contiguity alone. Members keep their own
         * mapping pins; prepare_revoke marks the whole registration
         * when any member mapping is revoked, so the stricter
         * same-mapping rule is not required for correctness and it
         * defeats coalescing when the guest maps the file through
         * small DAX windows. */
        if (!cxl_gpu_direct_host_range_follows(base, registration_length,
                                               (uintptr_t)next->host_address,
                                               next->length)) {
          break;
        }
        if (next->mapping != member->mapping) {
          cross_mapping = true;
        }
        registration_length += next->length;
        registration_padding += next->padding_bytes;
        end++;
      }
      registration = g_new0(CXLType2DirectRegistration, 1);
      registration->host_address = member->host_address;
      registration->length = registration_length;
      registration->padding_bytes = registration_padding;
      registration->member_count = end - first;
      registration->next = ct2d->direct_registrations;
      ct2d->direct_registrations = registration;
      g_ptr_array_add(new_groups, registration);
      if (cross_mapping) {
        ct2d->paired_case.active_direct_cross_mapping_groups++;
        ct2d->paired_case.active_direct_cross_mapping_members +=
            registration->member_count;
      }
      ct2d->paired_case.active_direct_retained_groups++;
      ct2d->paired_case.active_direct_peak_retained_groups =
          MAX(ct2d->paired_case.active_direct_peak_retained_groups,
              ct2d->paired_case.active_direct_retained_groups);
      for (guint j = first; j < end; j++) {
        CXLType2DirectPhysical *physical = g_ptr_array_index(pending, j);

        physical->registration = registration;
        physical->group_next = registration->members;
        registration->members = physical;
        g_assert(!g_tree_lookup(ct2d->direct_physical_ranges, physical));
        g_tree_insert(ct2d->direct_physical_ranges, physical, physical);
        physical->next = ct2d->direct_physicals;
        ct2d->direct_physicals = physical;
        ct2d->direct_registration_padding_bytes += physical->padding_bytes;
        ct2d->paired_case.active_direct_cache_misses++;
        ct2d->paired_case.active_direct_retained_physicals++;
        ct2d->paired_case.active_direct_peak_retained_physicals =
            MAX(ct2d->paired_case.active_direct_peak_retained_physicals,
                ct2d->paired_case.active_direct_retained_physicals);
        g_ptr_array_index(pending, j) = NULL;
      }
      first = end;
    }

    for (uint32_t wire = 0; wire < resolved_runs->len; wire++) {
      hwaddr cursor = validated[wire].mapping_offset;
      uint64_t remaining = validated[wire].length;

      wire_first_view[wire] = views->len;
      while (remaining) {
        CXLType2DirectPhysical *physical =
            cxl_type2_direct_physical_find_at_or_after(
                ct2d, validated[wire].mapping, cursor);
        CXLType2DirectRegistration *registration;
        uint64_t view_length;
        CXLType2DirectRun view;

        if (!physical || physical->mapping_offset > cursor ||
            physical->generation != validated[wire].generation ||
            physical->mapping->revoke_pending || !physical->registration ||
            physical->registration->revoke_pending) {
          result = CXL_GPU_ERROR_INVALID_VALUE;
          *failure_stage_out = "physical-coverage";
          *failure_index_out = wire;
          goto view_rollback;
        }
        registration = physical->registration;
        if (physical->references == UINT64_MAX ||
            registration->references == UINT64_MAX) {
          result = CXL_GPU_ERROR_OUT_OF_MEMORY;
          *failure_stage_out = "reference-overflow";
          *failure_index_out = wire;
          goto view_rollback;
        }
        view_length = MIN(remaining, physical->length -
                                         (cursor - physical->mapping_offset));
        view = (CXLType2DirectRun){
            .physical = physical,
            .physical_offset = cursor - physical->mapping_offset,
            .length = view_length,
        };
        physical->references++;
        registration->references++;
        g_array_append_val(views, view);
        wire_view_count[wire]++;
        cursor += view_length;
        remaining -= view_length;
      }
    }

    for (guint i = 0; i < new_groups->len; i++) {
      CXLType2DirectRegistration *registration =
          g_ptr_array_index(new_groups, i);
      uint64_t registration_views = 0;

      for (guint j = 0; j < views->len; j++) {
        CXLType2DirectRun *view = &g_array_index(views, CXLType2DirectRun, j);

        if (view->physical->registration == registration) {
          registration_views++;
        }
      }
      registration->view_count = registration_views;
    }
    g_ptr_array_free(pending, true);
    pending = NULL;
    g_ptr_array_free(new_groups, true);
    new_groups = NULL;
    for (uint32_t i = 0; i < header.range_count; i++) {
      uint32_t old_first = source->ranges[i].first_run;
      uint32_t old_count = source->ranges[i].run_count;
      uint64_t view_count = 0;

      for (uint32_t j = 0; j < old_count; j++) {
        view_count += wire_view_count[old_first + j];
      }
      if (view_count > UINT32_MAX) {
        result = CXL_GPU_ERROR_OUT_OF_MEMORY;
        *failure_stage_out = "view-count-overflow";
        *failure_index_out = i;
        goto view_rollback;
      }
      source->ranges[i].first_run = wire_first_view[old_first];
      source->ranges[i].run_count = view_count;
    }
    source->run_count = views->len;
    source->runs = (CXLType2DirectRun *)g_array_free(views, false);
    for (uint32_t i = 0; i < source->range_count; i++) {
      const CXLType2DirectRangeLayout *range = &source->ranges[i];

      for (uint32_t j = 1; j < range->run_count; j++) {
        CXLType2DirectRun *previous = &source->runs[range->first_run + j - 1];
        CXLType2DirectRun *current = &source->runs[range->first_run + j];
        uintptr_t previous_address =
            (uintptr_t)previous->physical->host_address +
            previous->physical_offset;
        uintptr_t current_address = (uintptr_t)current->physical->host_address +
                                    current->physical_offset;

        if (previous->physical == current->physical) {
          continue;
        }
        ct2d->paired_case.active_direct_physical_boundaries++;
        if (previous_address <= UINTPTR_MAX - previous->length &&
            previous_address + previous->length == current_address) {
          bool any_registered;
          bool both_registered;

          ct2d->paired_case.active_direct_host_contiguous_boundaries++;
          ct2d->paired_case.active_direct_host_contiguous_following_bytes +=
              current->length;
          if (previous->physical->registration ==
              current->physical->registration) {
            continue;
          }
          any_registered = previous->physical->registration->cuda_registered ||
                           current->physical->registration->cuda_registered;
          both_registered = previous->physical->registration->cuda_registered &&
                            current->physical->registration->cuda_registered;
          ct2d->paired_case.active_direct_cross_reg_boundaries++;
          ct2d->paired_case.active_direct_cross_reg_bytes += current->length;
          if (any_registered) {
            ct2d->paired_case
                .active_direct_cross_reg_any_registered_boundaries++;
            ct2d->paired_case.active_direct_cross_reg_any_registered_bytes +=
                current->length;
          }
          if (both_registered) {
            ct2d->paired_case
                .active_direct_cross_reg_both_registered_boundaries++;
            ct2d->paired_case.active_direct_cross_reg_both_registered_bytes +=
                current->length;
          }
        }
      }
    }
    g_free(wire_first_view);
    g_free(wire_view_count);
    goto views_done;

  view_rollback:
    source->run_count = views->len;
    source->runs = (CXLType2DirectRun *)g_array_free(views, false);
    if (pending) {
      for (guint i = 0; i < pending->len; i++) {
        CXLType2DirectPhysical *unregistered = g_ptr_array_index(pending, i);

        if (unregistered) {
          virtio_shared_memory_unpin(unregistered->mapping);
          g_free(unregistered);
        }
      }
      g_ptr_array_free(pending, true);
    }
    if (new_groups) {
      g_ptr_array_free(new_groups, true);
    }
    g_free(wire_first_view);
    g_free(wire_view_count);
    goto rollback;
  }
views_done:
  ct2d->paired_case.active_direct_register_acquire_ns +=
      qemu_clock_get_ns(QEMU_CLOCK_HOST) - phase_begin_ns;
  phase_begin_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
  if (ct2d->next_direct_source_id == UINT64_MAX) {
    result = CXL_GPU_ERROR_OUT_OF_MEMORY;
    *failure_stage_out = "source-id-overflow";
    goto rollback;
  }
  source->source_id = ++ct2d->next_direct_source_id;
  source->register_call_id = ct2d->gpu_cmd.call_id;
  source->next = ct2d->direct_sources;
  ct2d->direct_sources = source;
  bool inserted =
      g_hash_table_insert(ct2d->direct_source_ids, &source->source_id, source);

  g_assert(inserted);
  (void)inserted;
  *source_id_out = source->source_id;
  g_array_free(resolved_runs, true);
  ct2d->paired_case.active_direct_register_commit_ns +=
      qemu_clock_get_ns(QEMU_CLOCK_HOST) - phase_begin_ns;
  return CXL_GPU_SUCCESS;

rollback:
  for (uint32_t i = 0; i < source->run_count; i++) {
    if (source->runs[i].source_fd >= 0) {
      close(source->runs[i].source_fd);
      source->runs[i].source_fd = -1;
    }
    if (source->runs[i].physical) {
      int cleanup =
          cxl_type2_direct_physical_put(ct2d, source->runs[i].physical);
      source->runs[i].physical = NULL;
      if (cleanup != CXL_GPU_SUCCESS) {
        result = cleanup;
        *failure_stage_out = "rollback-cleanup";
        *failure_index_out = i;
      }
    }
  }
  int rollback_cleanup = cxl_type2_direct_idle_cleanup(ct2d);
  if (rollback_cleanup != CXL_GPU_SUCCESS) {
    result = rollback_cleanup;
    *failure_stage_out = "rollback-cleanup";
    *failure_index_out = source->run_count;
  }
out:
  if (source) {
    g_free(source->ranges);
    g_free(source->runs);
    g_free(source);
  }
  g_free(range_layouts);
  if (resolved_runs) {
    g_array_free(resolved_runs, true);
  }
  if (mapped_runs) {
    g_array_free(mapped_runs, true);
  }
  return result;
}

static void cxl_type2_record_htod_direct(CXLType2State *ct2d, uint64_t dev_ptr,
                                         const void *source_host, size_t size,
                                         void *stream,
                                         CXLType2DirectSource *source,
                                         int64_t enqueue_start_ns) {
  CXLType2PendingHtoD *pending;

  pending = g_new0(CXLType2PendingHtoD, 1);
  pending->stream = stream;
  pending->dev_ptr = dev_ptr;
  pending->sequence = ++ct2d->next_htod_sequence;
  pending->call_id = ct2d->gpu_cmd.call_id;
  pending->enqueue_host_ns = enqueue_start_ns;
  pending->size = size;
  pending->direct_source = true;
  pending->model_supply =
      ct2d->model_supply.route == CXL_TYPE2_MODEL_SUPPLY_SELECTED_HTOD &&
      ct2d->paired_case.active_command_code ==
          CXL_GPU_CMD_SOURCE_REGISTER_BATCH_HTOD_DIRECT_ASYNC;
  pending->direct_host = source_host;
  pending->source = source;
  pending->next = ct2d->pending_htod;
  ct2d->pending_htod = pending;
  source->pending_refcount++;
  ct2d->htod_pending_copies++;
  ct2d->htod_pending_bytes += size;
  ct2d->paired_case.active_direct_fragments++;
  ct2d->paired_case.active_direct_bytes += size;
  ct2d->paired_case.active_direct_pending_copies++;
  ct2d->paired_case.active_direct_pending_bytes += size;
  ct2d->paired_case.active_direct_peak_pending_copies =
      MAX(ct2d->paired_case.active_direct_peak_pending_copies,
          ct2d->paired_case.active_direct_pending_copies);
  ct2d->paired_case.active_direct_peak_pending_bytes =
      MAX(ct2d->paired_case.active_direct_peak_pending_bytes,
          ct2d->paired_case.active_direct_pending_bytes);
  ct2d->htod_peak_pending_copies =
      MAX(ct2d->htod_peak_pending_copies, ct2d->htod_pending_copies);
  ct2d->htod_peak_pending_bytes =
      MAX(ct2d->htod_peak_pending_bytes, ct2d->htod_pending_bytes);
}

static bool cxl_type2_direct_range_is_valid(
    CXLType2State *ct2d, const CXLGPUDirectRangeV1 *wire,
    CXLType2DirectSource *source_override, CXLType2DirectSource **source_out) {
  CXLType2DirectSource *source =
      source_override ? source_override
                      : cxl_type2_direct_source_find(ct2d, wire->source_id);
  CXLType2DirectRangeLayout *range;
  uint64_t skip;
  uint64_t remaining;

  if ((source_override && wire->source_id) || !source ||
      source->case_epoch != ct2d->paired_case.active_epoch ||
      wire->source_range >= source->range_count) {
    return false;
  }
  range = &source->ranges[wire->source_range];
  if (wire->source_offset > range->length ||
      wire->size > range->length - wire->source_offset ||
      wire->source_offset > UINT64_MAX - range->first_run_byte_offset) {
    return false;
  }
  skip = range->first_run_byte_offset + wire->source_offset;
  remaining = wire->size;
  for (uint32_t i = 0; i < range->run_count && remaining; i++) {
    CXLType2DirectRun *run = &source->runs[range->first_run + i];

    if (source->pageable_alias
            ? ((run->source_fd >= 0) == (run->ordinary_host != NULL))
            : !run->physical) {
      return false;
    }
    if (skip >= run->length) {
      skip -= run->length;
      continue;
    }
    uint64_t chunk = MIN(remaining, run->length - skip);
    remaining -= chunk;
    skip = 0;
  }
  if (remaining) {
    return false;
  }
  *source_out = source;
  return true;
}

typedef struct CXLType2PageableAliasPiece {
  CXLType2CudaAliasSource source;
  uint64_t destination;
  uint64_t source_index;
  uint32_t model_member_index;
  bool file_backed;
} CXLType2PageableAliasPiece;

static gint cxl_type2_pageable_alias_piece_compare(gconstpointer left,
                                                   gconstpointer right) {
  const CXLType2PageableAliasPiece *a = left;
  const CXLType2PageableAliasPiece *b = right;

  return a->destination < b->destination   ? -1
         : a->destination > b->destination ? 1
                                           : 0;
}

static int cxl_type2_direct_alias_submit(
    CXLType2State *ct2d, const uint8_t *payload, uint64_t payload_capacity,
    uint64_t range_count, uint64_t payload_bytes,
    CXLType2DirectSource *source_override, uint64_t *fail_index,
    const char **failure_stage, uint64_t *logical_enqueued,
    uint64_t *fragments_enqueued) {
  GArray *pieces = NULL;
  CXLType2CudaAllocationIdentity allocation_identity = {0};
  CXLType2CudaAllocation *allocation = NULL;
  CXLType2CudaAliasSource *alias_sources = NULL;
  uint8_t *alias_guard = NULL;
  uint64_t generation = source_override ? source_override->source_id : 0;
  uint64_t destination_offset = 0;
  uint64_t covered = 0;
  uint64_t alias = 0;
  uint64_t alias_begin_ns = 0;
  uint64_t guard_bytes = qemu_real_host_page_size();
  const char *alias_reason = NULL;
  bool remap = false;
  bool first_alias = false;
  int result = CXL_GPU_ERROR_INVALID_VALUE;

  *failure_stage = NULL;
  *logical_enqueued = 0;
  *fragments_enqueued = 0;
  if (!source_override || !source_override->pageable_alias || !generation ||
      !cxl_gpu_direct_batch_validate(payload, payload_capacity, range_count,
                                     payload_bytes, fail_index)) {
    *failure_stage = "batch-validate";
    return CXL_GPU_ERROR_INVALID_VALUE;
  }
  pieces = g_array_new(false, false, sizeof(CXLType2PageableAliasPiece));
  for (uint64_t i = 0; i < range_count; i++) {
    CXLGPUDirectRangeV1 wire;
    CXLType2DirectSource *source = NULL;
    CXLType2DirectRangeLayout *range;
    CXLType2CudaAllocationIdentity current_identity;
    uint64_t allocation_offset;
    uint64_t skip;
    uint64_t remaining;
    uint64_t copied = 0;

    memcpy(&wire, payload + i * sizeof(wire), sizeof(wire));
    if (!cxl_type2_direct_range_is_valid(ct2d, &wire, source_override,
                                         &source) ||
        !cxl_type2_cuda_allocation_identity_for_range(
            &ct2d->cuda_allocations, wire.destination, wire.size,
            &current_identity, &allocation_offset)) {
      *fail_index = i;
      *failure_stage = "range-admission";
      goto out;
    }
    if (i && (current_identity.base != allocation_identity.base ||
              current_identity.epoch != allocation_identity.epoch)) {
      *fail_index = i;
      *failure_stage = "destination-allocation";
      goto out;
    }
    allocation_identity = current_identity;
    range = &source->ranges[wire.source_range];
    skip = range->first_run_byte_offset + wire.source_offset;
    remaining = wire.size;
    for (uint32_t j = 0; j < range->run_count && remaining; j++) {
      CXLType2DirectRun *run = &source->runs[range->first_run + j];
      CXLType2PageableAliasPiece piece;
      uint64_t chunk;

      if (skip >= run->length) {
        skip -= run->length;
        continue;
      }
      chunk = MIN(remaining, run->length - skip);
      piece = (CXLType2PageableAliasPiece){
          .source =
              {
                  .fd = run->source_fd,
                  .host_copy_source =
                      run->ordinary_host
                          ? (const uint8_t *)run->ordinary_host + skip
                          : NULL,
                  .file_offset = run->file_offset + skip,
                  .destination_offset = allocation_offset + copied,
                  .length = chunk,
                  .mapping_generation = run->mapping_generation,
                  .logical_cxl_offset = run->logical_cxl_offset + skip,
                  .stat_device = run->source_device,
                  .stat_inode = run->source_inode,
                  .stat_size = run->source_size,
                  .stat_mode = run->source_mode,
                  .readonly = true,
              },
          .destination = wire.destination + copied,
          .source_index = i,
          .model_member_index = run->model_member_index,
          .file_backed = run->source_fd >= 0,
      };
      g_array_append_val(pieces, piece);
      copied += chunk;
      remaining -= chunk;
      skip = 0;
    }
    if (remaining) {
      *fail_index = i;
      *failure_stage = "source-coverage";
      goto out;
    }
  }
  g_array_sort(pieces, cxl_type2_pageable_alias_piece_compare);
  allocation = cxl_type2_cuda_allocation_find(&ct2d->cuda_allocations,
                                              allocation_identity.base,
                                              allocation_identity.epoch);
  if (!allocation || !pieces->len) {
    *failure_stage = "destination-allocation";
    goto out;
  }
  destination_offset = g_array_index(pieces, CXLType2PageableAliasPiece, 0)
                           .source.destination_offset;
  for (size_t i = 0; i < pieces->len; i++) {
    CXLType2PageableAliasPiece *piece =
        &g_array_index(pieces, CXLType2PageableAliasPiece, i);

    if (destination_offset > allocation->size ||
        covered > allocation->size - destination_offset ||
        piece->source.destination_offset != destination_offset + covered ||
        piece->source.length >
            allocation->size - destination_offset - covered) {
      *fail_index = i;
      *failure_stage = "alias-geometry";
      goto out;
    }
    if (piece->file_backed &&
        piece->model_member_index >=
            ct2d->model_supply.member_manifest.member_count) {
      *fail_index = i;
      *failure_stage = "member-range-admission";
      goto out;
    }
    piece->source.destination_offset -= destination_offset;
    covered += piece->source.length;
  }
  for (size_t i = 0; i < pieces->len; i++) {
    CXLType2PageableAliasPiece *piece =
        &g_array_index(pieces, CXLType2PageableAliasPiece, i);
    const CXLType2ModelMember *member =
        piece->file_backed ? &ct2d->model_supply.member_manifest
                                  .members[piece->model_member_index]
                           : NULL;

    if (!piece->file_backed) {
      continue;
    }

    QDict *event = cxl_type2_model_supply_event(ct2d);

    cxl_type2_qdict_put_u64(event, "call_id",
                            source_override->register_call_id);
    cxl_type2_qdict_put_u64(event, "source_index", piece->source_index);
    cxl_type2_qdict_put_u64(event, "model_member", piece->model_member_index);
    qdict_put_str(event, "artifact_sha256", member->artifact_sha256);
    cxl_type2_qdict_put_u64(event, "file_offset", piece->source.file_offset);
    cxl_type2_qdict_put_u64(event, "logical_bytes", piece->source.length);
    cxl_type2_qdict_put_u64(event, "mapping_generation",
                            piece->source.mapping_generation);
    cxl_type2_qdict_put_u64(event, "logical_cxl_offset",
                            piece->source.logical_cxl_offset);
    cxl_type2_qdict_put_u64(event, "destination_cuda_token",
                            piece->destination);
    cxl_type2_qdict_put_u64(event, "destination_bytes", piece->source.length);
    cxl_type2_qdict_put_u64(event, "destination_allocation_id",
                            allocation_identity.base);
    cxl_type2_qdict_put_u64(event, "allocation_epoch",
                            allocation_identity.epoch);
    cxl_type2_qdict_put_u64(event, "destination_offset",
                            destination_offset +
                                piece->source.destination_offset);
    cxl_type2_model_supply_hash_event(ct2d->model_supply.source_geometry,
                                      event);
    cxl_type2_model_supply_emit("KIMI_MODEL_SUPPLY_SOURCE", event);
    ct2d->model_supply.source_descriptor_count++;
    ct2d->model_supply.source_logical_bytes += piece->source.length;
  }
  alias_sources = g_try_new0(CXLType2CudaAliasSource, pieces->len + 1);
  alias_guard = g_try_malloc0(guard_bytes);
  if (!alias_sources || !alias_guard) {
    *failure_stage = "alias-source-allocate";
    result = CXL_GPU_ERROR_OUT_OF_MEMORY;
    goto out;
  }
  for (size_t i = 0; i < pieces->len; i++) {
    alias_sources[i] =
        g_array_index(pieces, CXLType2PageableAliasPiece, i).source;
  }
  alias_sources[pieces->len] = (CXLType2CudaAliasSource){
      .fd = -1,
      .host_copy_source = alias_guard,
      .destination_offset = covered,
      .length = guard_bytes,
      .readonly = true,
  };
  first_alias = allocation->pageable_alias == NULL;
  result = cxl_type2_model_aliases_replace_range(
      ct2d, allocation, destination_offset, covered, &remap);
  if (result != CXL_GPU_SUCCESS) {
    *failure_stage = "alias-generation-replace";
    goto out;
  }
  alias_begin_ns = cxl_type2_host_monotonic_ns();
  if (!covered || !cxl_type2_cuda_allocation_map_pageable_alias(
                      &ct2d->cuda_allocations, allocation_identity.base,
                      allocation_identity.epoch, generation, alias_sources,
                      pieces->len + 1, &source_override->register_call_id, 1,
                      destination_offset, covered, guard_bytes, &alias,
                      &alias_reason)) {
    *failure_stage = alias_reason ? alias_reason : "alias-map";
    goto out;
  }
  allocation = cxl_type2_cuda_allocation_find(&ct2d->cuda_allocations,
                                              allocation_identity.base,
                                              allocation_identity.epoch);
  {
    CXLType2CudaPageableAlias *pageable =
        cxl_type2_cuda_allocation_pageable_alias_for_offset_mutable(
            allocation, destination_offset);

    result = cxl_type2_model_alias_register(ct2d, allocation, pageable);
    if (result != CXL_GPU_SUCCESS) {
      *failure_stage = "alias-host-register";
      goto out;
    }
    alias = pageable->device_alias;
    uint64_t duration_ns = cxl_type2_host_monotonic_ns() - alias_begin_ns;
    QDict *event = cxl_type2_model_supply_event(ct2d);

    qdict_put_str(event, "action", remap ? "remap" : "build");
    qdict_put_str(event, "result", "pass");
    cxl_type2_qdict_put_u64(event, "destination_allocation_id",
                            allocation_identity.base);
    cxl_type2_qdict_put_u64(event, "allocation_epoch",
                            allocation_identity.epoch);
    cxl_type2_qdict_put_u64(event, "content_generation", generation);
    qdict_put(event, "contributing_source_call_ids",
              cxl_type2_model_supply_source_calls(pageable));
    cxl_type2_qdict_put_u64(event, "destination_offset", destination_offset);
    cxl_type2_qdict_put_u64(event, "logical_bytes", covered);
    cxl_type2_qdict_put_u64(event, "alias_offset", alias);
    cxl_type2_qdict_put_u64(event, "file_mapped_bytes",
                            pageable->file_mapped_bytes);
    cxl_type2_qdict_put_u64(event, "host_composition_copy_bytes",
                            pageable->host_composition_copy_bytes);
    cxl_type2_qdict_put_u64(event, "derived_boundary_pages",
                            pageable->derived_boundary_pages);
    cxl_type2_qdict_put_u64(event, "derived_boundary_copy_bytes",
                            pageable->derived_boundary_copy_bytes);
    cxl_type2_qdict_put_u64(event, "guard_bytes", pageable->guard_bytes);
    cxl_type2_qdict_put_u64(event, "duration_ns", duration_ns);
    cxl_type2_model_supply_hash_event(ct2d->model_supply.alias_geometry, event);
    cxl_type2_model_supply_emit("KIMI_MODEL_SUPPLY_ALIAS", event);

    if (remap) {
      ct2d->model_supply.alias_remaps++;
    } else {
      ct2d->model_supply.alias_builds++;
      if (first_alias) {
        ct2d->model_supply.reserved_gpu_token_bytes += allocation->size;
      }
    }
    ct2d->model_supply.alias_logical_bytes += covered;
    ct2d->model_supply.logical_source_view_bytes += covered;
    ct2d->model_supply.file_mapped_bytes += pageable->file_mapped_bytes;
    ct2d->model_supply.host_composition_copy_bytes +=
        pageable->host_composition_copy_bytes;
    ct2d->model_supply.derived_boundary_pages +=
        pageable->derived_boundary_pages;
    ct2d->model_supply.derived_boundary_copy_bytes +=
        pageable->derived_boundary_copy_bytes;
    ct2d->model_supply.boundary_composition_wall_ns += duration_ns;
    ct2d->model_supply.guard_bytes += pageable->guard_bytes;
    if (pageable->file_mapped_bytes + pageable->host_composition_copy_bytes +
            pageable->derived_boundary_copy_bytes >
        covered) {
      ct2d->model_supply.page_collateral_bytes +=
          pageable->file_mapped_bytes + pageable->host_composition_copy_bytes +
          pageable->derived_boundary_copy_bytes - covered;
    }
    uint64_t logical_host_copy =
        pageable->host_composition_copy_bytes > pageable->guard_bytes
            ? pageable->host_composition_copy_bytes - pageable->guard_bytes
            : 0;
    ct2d->model_supply.logical_direct_bytes += covered - logical_host_copy;
  }
  *logical_enqueued = range_count;
  *fragments_enqueued = pieces->len;
  *fail_index = SIZE_MAX;
  result = CXL_GPU_SUCCESS;

out:
  g_free(alias_guard);
  g_free(alias_sources);
  if (pieces) {
    g_array_free(pieces, true);
  }
  return result;
}

static int cxl_type2_enqueue_htod_from_host(CXLType2State *ct2d,
                                            uint64_t dev_ptr,
                                            const void *source, size_t size,
                                            void *stream,
                                            const char *transport);

static int cxl_type2_direct_batch_submit(
    CXLType2State *ct2d, const uint8_t *payload, uint64_t payload_capacity,
    uint64_t range_count, uint64_t payload_bytes, void *stream,
    CXLType2DirectSource *source_override, uint64_t *fail_index,
    const char **failure_stage, uint64_t *logical_enqueued,
    uint64_t *fragments_enqueued) {
  typedef struct CXLType2DirectResolvedRange {
    CXLGPUDirectRangeV1 wire;
    CXLType2DirectSource *source;
  } CXLType2DirectResolvedRange;
  typedef struct CXLType2DirectSpan {
    HetGPUDevicePtr destination;
    const void *host;
    size_t length;
    CXLType2DirectSource *source;
    CXLType2DirectRegistration *registration;
    uint64_t logical_index;
    int64_t enqueue_start_ns;
  } CXLType2DirectSpan;

  CXLType2DirectResolvedRange *resolved = NULL;
  GArray *spans = NULL;
  HetGPUDevicePtr *destinations = NULL;
  const void **hosts = NULL;
  size_t *sizes = NULL;
  uint64_t *logical_destinations = NULL;
  size_t *logical_sizes = NULL;
  size_t failed_span = SIZE_MAX;
  int result = CXL_GPU_SUCCESS;

  if (source_override && source_override->pageable_alias) {
    return cxl_type2_direct_alias_submit(
        ct2d, payload, payload_capacity, range_count, payload_bytes,
        source_override, fail_index, failure_stage, logical_enqueued,
        fragments_enqueued);
  }

  *failure_stage = NULL;
  *logical_enqueued = 0;
  *fragments_enqueued = 0;
  if (!ct2d->cuda_direct_source || ct2d->direct_source_poisoned ||
      !cxl_gpu_direct_batch_validate(payload, payload_capacity, range_count,
                                     payload_bytes, fail_index)) {
    *failure_stage = "batch-validate";
    return CXL_GPU_ERROR_INVALID_VALUE;
  }
  resolved = g_try_new(CXLType2DirectResolvedRange, range_count);
  if (!resolved) {
    *failure_stage = "batch-allocate";
    result = CXL_GPU_ERROR_OUT_OF_MEMORY;
    goto out;
  }
  spans = g_array_new(false, false, sizeof(CXLType2DirectSpan));
  if (source_override) {
    logical_destinations = g_try_new(uint64_t, range_count);
    logical_sizes = g_try_new(size_t, range_count);
    if (!logical_destinations || !logical_sizes) {
      g_free(logical_destinations);
      g_free(logical_sizes);
      logical_destinations = NULL;
      logical_sizes = NULL;
      cxl_type2_cuda_classifier_reject(
          ct2d, CXL_TYPE2_CUDA_CLASSIFIER_UNAVAILABLE,
          CXL_TYPE2_CUDA_REJECTION_ALLOCATION_MISSING,
          CXL_GPU_CMD_SOURCE_REGISTER_BATCH_HTOD_DIRECT_ASYNC, true);
    }
  }
  /* Resolve every source and range before the first Driver enqueue. */
  for (uint64_t i = 0; i < range_count; i++) {
    memcpy(&resolved[i].wire, payload + i * sizeof(resolved[i].wire),
           sizeof(resolved[i].wire));
    if (!cxl_type2_direct_range_is_valid(
            ct2d, &resolved[i].wire, source_override, &resolved[i].source)) {
      *fail_index = i;
      *failure_stage = "range-resolve";
      result = CXL_GPU_ERROR_INVALID_VALUE;
      goto out;
    }
    if (logical_destinations) {
      logical_destinations[i] = resolved[i].wire.destination;
      logical_sizes[i] = resolved[i].wire.size;
    }
  }

  for (uint64_t i = 0; i < range_count; i++) {
    CXLGPUDirectRangeV1 *wire = &resolved[i].wire;
    CXLType2DirectSource *source = resolved[i].source;
    CXLType2DirectRangeLayout *range;
    uint64_t skip;
    uint64_t remaining;
    uint64_t copied = 0;
    CXLType2DirectRegistration *span_registration = NULL;
    const void *span_host = NULL;
    uint64_t span_destination = 0;
    uint64_t span_length = 0;

    range = &source->ranges[wire->source_range];
    skip = range->first_run_byte_offset + wire->source_offset;
    remaining = wire->size;
    for (uint32_t j = 0; j < range->run_count && remaining; j++) {
      CXLType2DirectRun *run = &source->runs[range->first_run + j];
      CXLType2DirectPhysical *physical = run->physical;
      const void *host;
      uint64_t destination;
      uint64_t chunk;

      if (skip >= run->length) {
        skip -= run->length;
        continue;
      }
      chunk = MIN(remaining, run->length - skip);
      host =
          (const uint8_t *)physical->host_address + run->physical_offset + skip;
      destination = wire->destination + copied;
      if (span_length &&
          !cxl_gpu_direct_copy_span_follows(
              (uintptr_t)source, (uintptr_t)span_registration,
              (uintptr_t)span_host, span_destination, span_length,
              (uintptr_t)source, (uintptr_t)physical->registration,
              (uintptr_t)host, destination, chunk)) {
        CXLType2DirectSpan span = {
            .destination = span_destination,
            .host = span_host,
            .length = span_length,
            .source = source,
            .registration = span_registration,
            .logical_index = i,
        };
        g_array_append_val(spans, span);
        span_length = 0;
      }
      if (!span_length) {
        span_registration = physical->registration;
        span_host = host;
        span_destination = destination;
      }
      span_length += chunk;
      copied += chunk;
      remaining -= chunk;
      skip = 0;
    }
    if (span_length) {
      CXLType2DirectSpan span = {
          .destination = span_destination,
          .host = span_host,
          .length = span_length,
          .source = source,
          .registration = span_registration,
          .logical_index = i,
      };
      g_array_append_val(spans, span);
    }
  }

  destinations = g_try_new(HetGPUDevicePtr, spans->len);
  hosts = g_try_new(const void *, spans->len);
  sizes = g_try_new(size_t, spans->len);
  if (!destinations || !hosts || !sizes) {
    *failure_stage = "batch-allocate";
    result = CXL_GPU_ERROR_OUT_OF_MEMORY;
    goto out;
  }
  for (size_t i = 0; i < spans->len; i++) {
    CXLType2DirectSpan *span = &g_array_index(spans, CXLType2DirectSpan, i);

    if (span->registration->revoke_pending) {
      *fail_index = span->logical_index;
      *failure_stage = "registration-revoked";
      result = CXL_GPU_ERROR_INVALID_VALUE;
      goto out;
    }
    destinations[i] = span->destination;
    hosts[i] = span->host;
    sizes[i] = span->length;
  }
  if (!cxl_gpu_direct_destinations_are_independent(destinations, sizes,
                                                   spans->len, &failed_span)) {
    *fail_index = failed_span < spans->len
                      ? g_array_index(spans, CXLType2DirectSpan, failed_span)
                            .logical_index
                      : 0;
    *failure_stage = "destination-overlap";
    result = CXL_GPU_ERROR_INVALID_VALUE;
    goto out;
  }
  if (source_override && logical_destinations) {
    uint64_t classified_bytes = 0;
    CXLType2CudaCoverageKind coverage = cxl_type2_cuda_record_coverage(
        ct2d, logical_destinations, logical_sizes, range_count,
        &classified_bytes);

    if (coverage == CXL_TYPE2_CUDA_COVERAGE_WHOLE && spans->len == 1 &&
        cxl_type2_cuda_counter_can_add(
            ct2d->paired_case.allocation_whole_single_span_batches, 1) &&
        cxl_type2_cuda_counter_can_add(
            ct2d->paired_case.allocation_whole_single_span_bytes,
            classified_bytes)) {
      ct2d->paired_case.allocation_whole_single_span_batches++;
      ct2d->paired_case.allocation_whole_single_span_bytes += classified_bytes;
    }
  }
  if (source_override) {
    for (size_t i = 0; i < spans->len; i++) {
      CXLType2DirectSpan *span = &g_array_index(spans, CXLType2DirectSpan, i);

      result = cxl_type2_enqueue_htod_from_host(ct2d, span->destination,
                                                span->host, span->length,
                                                stream, "cxl-source-copy");
      if (result != CXL_GPU_SUCCESS) {
        *fail_index = span->logical_index;
        *failure_stage = "driver-copy-submit";
        goto out;
      }
      ct2d->model_supply.model_htod_calls++;
      ct2d->model_supply.model_htod_bytes += span->length;
      (*fragments_enqueued)++;
      *logical_enqueued =
          i + 1 == spans->len || g_array_index(spans, CXLType2DirectSpan, i + 1)
                                         .logical_index != span->logical_index
              ? span->logical_index + 1
              : span->logical_index;
    }
    g_assert(*logical_enqueued == range_count);
    *fail_index = SIZE_MAX;
    goto out;
  }
  /* Let each ready prefix run while the host registers the next range. */
  for (size_t slice_start = 0, boundary = 0; boundary <= spans->len;
       boundary++) {
    CXLType2DirectSpan *boundary_span =
        boundary < spans->len
            ? &g_array_index(spans, CXLType2DirectSpan, boundary)
            : NULL;

    if (boundary_span && boundary_span->registration->cuda_registered) {
      continue;
    }
    if (boundary > slice_start) {
      size_t slice_count = boundary - slice_start;
      size_t slice_submitted = 0;
      size_t slice_failed = SIZE_MAX;
      int64_t enqueue_start_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);

      for (size_t i = slice_start; i < boundary; i++) {
        CXLType2DirectSpan *span = &g_array_index(spans, CXLType2DirectSpan, i);

        if (!span->registration->cuda_registered ||
            span->registration->revoke_pending) {
          *fail_index = span->logical_index;
          *failure_stage = "registration-not-ready";
          result = CXL_GPU_ERROR_INVALID_VALUE;
          goto out;
        }
        span->enqueue_start_ns = enqueue_start_ns;
      }
      result = hetgpu_cuda_memcpy_htod_batch_async(
          &ct2d->gpu_info.hetgpu_state, destinations + slice_start,
          hosts + slice_start, sizes + slice_start, slice_count, stream,
          &slice_submitted, &slice_failed);
      for (size_t i = slice_start; i < slice_start + slice_submitted; i++) {
        CXLType2DirectSpan *span = &g_array_index(spans, CXLType2DirectSpan, i);

        cxl_type2_record_htod_direct(ct2d, span->destination, span->host,
                                     span->length, stream, span->source,
                                     span->enqueue_start_ns);
        (*fragments_enqueued)++;
      }
      if (slice_submitted) {
        size_t next = slice_start + slice_submitted;
        CXLType2DirectSpan *last =
            &g_array_index(spans, CXLType2DirectSpan, next - 1);

        *logical_enqueued =
            next == spans->len || g_array_index(spans, CXLType2DirectSpan, next)
                                          .logical_index != last->logical_index
                ? last->logical_index + 1
                : last->logical_index;
      }
      if (result != CXL_GPU_SUCCESS) {
        size_t failed = slice_failed < slice_count ? slice_start + slice_failed
                                                   : slice_start;

        *fail_index =
            g_array_index(spans, CXLType2DirectSpan, failed).logical_index;
        *failure_stage = "driver-copy-submit";
        goto out;
      }
    }
    if (!boundary_span) {
      break;
    }
    result =
        cxl_type2_direct_registration_ensure(ct2d, boundary_span->registration);
    if (result != CXL_GPU_SUCCESS) {
      *fail_index = boundary_span->logical_index;
      *failure_stage = "physical-register";
      goto out;
    }
    slice_start = boundary;
  }
  g_assert(*logical_enqueued == range_count);
  *fail_index = SIZE_MAX;
out:
  g_free(logical_sizes);
  g_free(logical_destinations);
  g_free(sizes);
  g_free(hosts);
  g_free(destinations);
  if (spans) {
    g_array_free(spans, true);
  }
  g_free(resolved);
  return result;
}

static int cxl_type2_enqueue_htod_from_host(CXLType2State *ct2d,
                                            uint64_t dev_ptr,
                                            const void *source, size_t size,
                                            void *stream,
                                            const char *transport) {
  HetGPUState *hetgpu = &ct2d->gpu_info.hetgpu_state;
  CXLType2PendingHtoD *pending;
  void *staging = NULL;
  size_t staging_capacity;
  uint64_t staging_id;
  bool pool_hit;
  int result;
  int64_t acquire_start_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
  int64_t acquire_duration_ns;
  int64_t staging_start_ns;
  int64_t staging_duration_ns;
  int64_t enqueue_start_ns;
  int64_t enqueue_duration_ns;

  result = cxl_type2_htod_staging_acquire(
      ct2d, size, &staging, &staging_capacity, &staging_id, &pool_hit);
  acquire_duration_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST) - acquire_start_ns;
  if (result != CXL_GPU_SUCCESS) {
    return result;
  }
  staging_start_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
  memcpy(staging, source, size);
  staging_duration_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST) - staging_start_ns;
  enqueue_start_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
  result =
      hetgpu_cuda_memcpy_htod_async(hetgpu, dev_ptr, staging, size, stream);
  enqueue_duration_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST) - enqueue_start_ns;
  if (result != CXL_GPU_SUCCESS) {
    bool pooled;
    (void)cxl_type2_htod_staging_release(ct2d, staging, staging_capacity,
                                         staging_id, &pooled);
    return result;
  }

  pending = g_new0(CXLType2PendingHtoD, 1);
  pending->stream = stream;
  pending->staging = staging;
  pending->staging_id = staging_id;
  pending->staging_capacity = staging_capacity;
  pending->dev_ptr = dev_ptr;
  pending->sequence = ++ct2d->next_htod_sequence;
  pending->call_id = ct2d->gpu_cmd.call_id;
  pending->enqueue_host_ns = enqueue_start_ns;
  pending->size = size;
  pending->model_supply =
      ct2d->model_supply.route == CXL_TYPE2_MODEL_SUPPLY_SELECTED_HTOD &&
      (ct2d->paired_case.active_command_code == CXL_GPU_CMD_BATCH_HTOD_ASYNC ||
       ct2d->paired_case.active_command_code ==
           CXL_GPU_CMD_SOURCE_REGISTER_BATCH_HTOD_DIRECT_ASYNC);
  pending->next = ct2d->pending_htod;
  ct2d->pending_htod = pending;
  ct2d->htod_pending_copies++;
  ct2d->htod_pending_bytes += staging_capacity;
  ct2d->paired_case.active_htod_staging_pending_bytes += staging_capacity;
  ct2d->paired_case.active_htod_peak_staging_pending_bytes =
      MAX(ct2d->paired_case.active_htod_peak_staging_pending_bytes,
          ct2d->paired_case.active_htod_staging_pending_bytes);
  ct2d->htod_peak_pending_copies =
      MAX(ct2d->htod_peak_pending_copies, ct2d->htod_pending_copies);
  ct2d->htod_peak_pending_bytes =
      MAX(ct2d->htod_peak_pending_bytes, ct2d->htod_pending_bytes);
  if (ct2d->paired_case.qemu_cuda_calls_enabled) {
    qemu_log("CXL TYPE2 TRACE copy_driver call_id=0x%016" PRIx64
             " sequence=%" PRIu64 " direction=htod bytes=%zu backend_result=%d "
             "implementation=async-enqueue stream_forwarded=1 "
             "transport=%s staging_id=%" PRIu64 " capacity_bytes=%zu "
             "staging_source=%s acquire_duration_ns=%" PRId64
             " host_alloc_duration_ns=%" PRId64
             " staging_memcpy_duration_ns=%" PRId64
             " driver_enqueue_duration_ns=%" PRId64 " pending_copies=%" PRIu64
             " pending_bytes=%" PRIu64 " peak_pending_copies=%" PRIu64
             " peak_pending_bytes=%" PRIu64 " pooled_buffers=%" PRIu64
             " pooled_bytes=%" PRIu64 " peak_pooled_bytes=%" PRIu64
             " pool_hits=%" PRIu64 " pool_misses=%" PRIu64
             " driver_allocations=%" PRIu64 " driver_frees=%" PRIu64
             " evictions=%" PRIu64 "\n",
             ct2d->gpu_cmd.call_id, pending->sequence, size, result, transport,
             staging_id, staging_capacity, pool_hit ? "pool" : "driver",
             acquire_duration_ns, pool_hit ? 0 : acquire_duration_ns,
             staging_duration_ns, enqueue_duration_ns,
             ct2d->htod_pending_copies, ct2d->htod_pending_bytes,
             ct2d->htod_peak_pending_copies, ct2d->htod_peak_pending_bytes,
             ct2d->htod_pooled_buffers, ct2d->htod_pooled_bytes,
             ct2d->htod_peak_pooled_bytes, ct2d->htod_pool_hits,
             ct2d->htod_pool_misses, ct2d->htod_driver_allocations,
             ct2d->htod_driver_frees, ct2d->htod_pool_evictions);
  }
  return CXL_GPU_SUCCESS;
}

typedef struct CXLType2BatchHtoDEnqueueContext {
  CXLType2State *ct2d;
  void *stream;
} CXLType2BatchHtoDEnqueueContext;

static int cxl_type2_batch_htod_enqueue_one(void *opaque, uint64_t destination,
                                            const void *source, size_t size) {
  CXLType2BatchHtoDEnqueueContext *context = opaque;

  int result = cxl_type2_enqueue_htod_from_host(
      context->ct2d, destination, source, size, context->stream, "bar2-batch");

  if (result == CXL_GPU_SUCCESS) {
    context->ct2d->paired_case.active_payload_source_bytes += size;
  }
  return result;
}

static int cxl_type2_htod_staging_release(CXLType2State *ct2d, void *data,
                                          size_t capacity, uint64_t buffer_id,
                                          bool *pooled) {
  CXLType2HtoDStagingBuffer *buffer;

  if (ct2d->htod_staging_pool_size >= capacity &&
      ct2d->htod_pooled_bytes <= ct2d->htod_staging_pool_size - capacity) {
    buffer = g_new0(CXLType2HtoDStagingBuffer, 1);
    buffer->data = data;
    buffer->capacity = capacity;
    buffer->id = buffer_id;
    buffer->next = ct2d->htod_staging_pool;
    ct2d->htod_staging_pool = buffer;
    ct2d->htod_pooled_buffers++;
    ct2d->htod_pooled_bytes += capacity;
    ct2d->htod_peak_pooled_bytes =
        MAX(ct2d->htod_peak_pooled_bytes, ct2d->htod_pooled_bytes);
    ct2d->paired_case.active_htod_peak_pooled_bytes =
        MAX(ct2d->paired_case.active_htod_peak_pooled_bytes,
            ct2d->htod_pooled_bytes);
    *pooled = true;
    return CXL_GPU_SUCCESS;
  }

  if (ct2d->htod_staging_pool_size) {
    ct2d->htod_pool_evictions++;
    ct2d->paired_case.active_htod_pool_evictions++;
  }
  *pooled = false;
  return cxl_type2_htod_staging_free(ct2d, data);
}

static int cxl_type2_clear_htod_staging_pool(CXLType2State *ct2d) {
  CXLType2HtoDStagingBuffer *buffer = ct2d->htod_staging_pool;
  int first_error = CXL_GPU_SUCCESS;

  ct2d->htod_staging_pool = NULL;
  while (buffer) {
    CXLType2HtoDStagingBuffer *next = buffer->next;
    int result = cxl_type2_htod_staging_free(ct2d, buffer->data);

    if (first_error == CXL_GPU_SUCCESS && result != CXL_GPU_SUCCESS) {
      first_error = result;
    }
    g_free(buffer);
    buffer = next;
  }
  ct2d->htod_pooled_buffers = 0;
  ct2d->htod_pooled_bytes = 0;
  return first_error;
}

static void cxl_type2_pending_alias_consumers_add(
    CXLType2State *ct2d, HetGPUStream stream, uint64_t launch_id,
    CXLType2CudaAliasConsumer consumer, GArray *bindings) {
  int64_t enqueue_host_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);

  for (size_t i = 0; i < bindings->len; i++) {
    const CXLType2AliasLaunchBinding *binding =
        &g_array_index(bindings, CXLType2AliasLaunchBinding, i);
    CXLType2PendingAliasConsumer *pending =
        g_new0(CXLType2PendingAliasConsumer, 1);

    pending->stream = stream;
    pending->identity = binding->identity;
    pending->generation = binding->generation;
    pending->sequence = ++ct2d->next_alias_consumer_sequence;
    pending->launch_id = launch_id;
    pending->allocation_begin = binding->allocation_begin;
    pending->allocation_end = binding->allocation_end;
    pending->alias_begin = binding->alias_begin;
    pending->enqueue_host_ns = enqueue_host_ns;
    pending->consumer = consumer;
    pending->owns_inflight_ref = binding->owns_inflight_ref;
    pending->next = ct2d->pending_alias_consumers;
    ct2d->pending_alias_consumers = pending;
  }
  g_array_unref(bindings);
}

static int cxl_type2_release_pending_alias_consumers(
    CXLType2State *ct2d, HetGPUStream stream, bool all_streams,
    uint64_t through_sequence, uint64_t *completed_launches) {
  CXLType2PendingAliasConsumer **cursor = &ct2d->pending_alias_consumers;

  if (completed_launches) {
    *completed_launches = 0;
  }

  while (*cursor) {
    CXLType2PendingAliasConsumer *pending = *cursor;

    if ((!all_streams && pending->stream != stream) ||
        pending->sequence > through_sequence) {
      cursor = &pending->next;
      continue;
    }
    *cursor = pending->next;
    CXLType2CudaAllocation *allocation = cxl_type2_cuda_allocation_find(
        &ct2d->cuda_allocations, pending->identity.base,
        pending->identity.epoch);
    CXLType2AliasLaunchBinding binding = {
        .identity = pending->identity,
        .generation = pending->generation,
        .allocation_begin = pending->allocation_begin,
        .allocation_end = pending->allocation_end,
        .alias_begin = pending->alias_begin,
    };
    const CXLType2CudaPageableAlias *alias =
        cxl_type2_model_alias_for_binding(allocation, &binding);
    QList *source_calls = cxl_type2_model_supply_source_calls(alias);
    if (pending->owns_inflight_ref &&
        !cxl_type2_cuda_allocation_release_pageable_alias(
            &ct2d->cuda_allocations, pending->identity.base,
            pending->identity.epoch, pending->generation, pending->consumer)) {
      qobject_unref(source_calls);
      cxl_type2_model_supply_fail(ct2d, ct2d->gpu_cmd.call_id, "completion",
                                  "consumer-release-failed", UINT64_MAX,
                                  pending->identity.base,
                                  pending->identity.epoch, pending->generation);
      g_free(pending);
      return CXL_GPU_ERROR_INVALID_VALUE;
    }
    QDict *event = cxl_type2_model_supply_event(ct2d);

    qdict_put_str(event, "action", "complete");
    qdict_put_str(event, "result", "pass");
    cxl_type2_qdict_put_u64(event, "launch_id", pending->launch_id);
    cxl_type2_qdict_put_u64(event, "destination_allocation_id",
                            pending->identity.base);
    cxl_type2_qdict_put_u64(event, "allocation_epoch", pending->identity.epoch);
    cxl_type2_qdict_put_u64(event, "content_generation", pending->generation);
    qdict_put(event, "contributing_source_call_ids", source_calls);
    cxl_type2_qdict_put_u64(event, "destination_cuda_token",
                            pending->alias_begin);
    qdict_put_null(event, "node_kind");
    qdict_put_str(event, "binding_kind",
                  pending->consumer == CXL_TYPE2_CUDA_ALIAS_NORMAL
                      ? "normal-top-level"
                      : "graph-top-level");
    qdict_put_null(event, "graph_exec_id");
    qdict_put(event, "graph_node_tokens", qlist_new());
    qdict_put_null(event, "module_load_kind");
    qdict_put_null(event, "driver_input_cubin_sha256");
    qdict_put_null(event, "function_name");
    qdict_put_null(event, "static_consumer_certificate_sha256");
    qdict_put(event, "visible_parameter_roots", qlist_new());
    cxl_type2_model_supply_emit("KIMI_MODEL_SUPPLY_CONSUMER", event);
    bool launch_still_pending = false;

    for (CXLType2PendingAliasConsumer *candidate =
             ct2d->pending_alias_consumers;
         candidate; candidate = candidate->next) {
      if (candidate->launch_id == pending->launch_id) {
        launch_still_pending = true;
        break;
      }
    }
    if (!launch_still_pending) {
      int64_t completion_host_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);

      if (completion_host_ns < pending->enqueue_host_ns ||
          UINT64_MAX - ct2d->model_supply.consumer_wall_ns <
              (uint64_t)(completion_host_ns - pending->enqueue_host_ns)) {
        ct2d->model_supply.poisoned = true;
        cxl_type2_model_supply_fail(ct2d, ct2d->gpu_cmd.call_id, "completion",
                                    "consumer-wall-counter-overflow",
                                    UINT64_MAX, pending->identity.base,
                                    pending->identity.epoch,
                                    pending->generation);
        g_free(pending);
        return CXL_GPU_ERROR_INVALID_VALUE;
      }
      ct2d->model_supply.consumer_wall_ns +=
          completion_host_ns - pending->enqueue_host_ns;
      if (pending->consumer == CXL_TYPE2_CUDA_ALIAS_NORMAL) {
        ct2d->model_supply.normal_completions++;
      } else {
        ct2d->model_supply.graph_completions++;
      }
      if (completed_launches) {
        (*completed_launches)++;
      }
    }
    g_free(pending);
  }
  return CXL_GPU_SUCCESS;
}

static void cxl_type2_record_model_supply_completion(
    CXLType2State *ct2d, int64_t wait_begin_ns, int64_t wait_end_ns,
    bool blocking_wait, uint64_t htod_work_items, uint64_t consumer_launches) {
  uint64_t work_items;
  uint64_t wait_ns;

  if (UINT64_MAX - htod_work_items < consumer_launches) {
    ct2d->model_supply.poisoned = true;
    cxl_type2_model_supply_fail(ct2d, ct2d->gpu_cmd.call_id, "completion",
                                "completion-work-item-counter-overflow",
                                UINT64_MAX, 0, 0, 0);
    return;
  }
  work_items = htod_work_items + consumer_launches;
  if (!work_items) {
    return;
  }
  if (!blocking_wait) {
    if (UINT64_MAX - ct2d->model_supply.completion_nonblocking_work_items <
        work_items) {
      ct2d->model_supply.poisoned = true;
      cxl_type2_model_supply_fail(ct2d, ct2d->gpu_cmd.call_id, "completion",
                                  "completion-nonblocking-counter-overflow",
                                  UINT64_MAX, 0, 0, 0);
      return;
    }
    ct2d->model_supply.completion_nonblocking_work_items += work_items;
    return;
  }
  if (wait_begin_ns < 0 || wait_end_ns < wait_begin_ns) {
    ct2d->model_supply.poisoned = true;
    cxl_type2_model_supply_fail(ct2d, ct2d->gpu_cmd.call_id, "completion",
                                "completion-wait-clock-invalid", UINT64_MAX, 0,
                                0, 0);
    return;
  }
  wait_ns = wait_end_ns - wait_begin_ns;
  if (ct2d->model_supply.completion_wait_calls == UINT64_MAX ||
      UINT64_MAX - ct2d->model_supply.completion_wait_wall_ns < wait_ns ||
      UINT64_MAX - ct2d->model_supply.completion_wait_work_items < work_items) {
    ct2d->model_supply.poisoned = true;
    cxl_type2_model_supply_fail(ct2d, ct2d->gpu_cmd.call_id, "completion",
                                "completion-wait-counter-overflow", UINT64_MAX,
                                0, 0, 0);
    return;
  }
  ct2d->model_supply.completion_wait_calls++;
  ct2d->model_supply.completion_wait_wall_ns += wait_ns;
  ct2d->model_supply.completion_wait_work_items += work_items;
}

static uint64_t cxl_type2_latest_pending_alias_sequence(CXLType2State *ct2d,
                                                        HetGPUStream stream) {
  uint64_t latest = 0;

  for (CXLType2PendingAliasConsumer *pending = ct2d->pending_alias_consumers;
       pending; pending = pending->next) {
    if (pending->stream == stream && pending->sequence > latest) {
      latest = pending->sequence;
    }
  }
  return latest;
}

static bool cxl_type2_stream_wire_is_capturing(CXLType2State *ct2d,
                                               uint64_t stream_wire) {
  return ct2d->gpu_cmd.capturing_stream_wires &&
         g_hash_table_contains(ct2d->gpu_cmd.capturing_stream_wires,
                               &stream_wire);
}

static void cxl_type2_stream_wire_capture_set(CXLType2State *ct2d,
                                              uint64_t stream_wire,
                                              bool capturing) {
  if (capturing) {
    if (!ct2d->gpu_cmd.capturing_stream_wires) {
      ct2d->gpu_cmd.capturing_stream_wires =
          g_hash_table_new_full(g_int64_hash, g_int64_equal, g_free, NULL);
    }
    uint64_t *key = g_new(uint64_t, 1);

    *key = stream_wire;
    g_hash_table_replace(ct2d->gpu_cmd.capturing_stream_wires, key, key);
  } else if (ct2d->gpu_cmd.capturing_stream_wires) {
    g_hash_table_remove(ct2d->gpu_cmd.capturing_stream_wires, &stream_wire);
  }
}

static int cxl_type2_release_pending_htod(CXLType2State *ct2d,
                                          HetGPUStream stream, bool all_streams,
                                          uint64_t through_sequence,
                                          const char *completion,
                                          uint64_t *model_work_items) {
  CXLType2PendingHtoD **cursor = &ct2d->pending_htod;
  int first_error = CXL_GPU_SUCCESS;

  if (model_work_items) {
    *model_work_items = 0;
  }

  while (*cursor) {
    CXLType2PendingHtoD *pending = *cursor;

    if ((!all_streams && pending->stream != stream) ||
        pending->sequence > through_sequence) {
      cursor = &pending->next;
      continue;
    }
    *cursor = pending->next;
    if (model_work_items && pending->model_supply) {
      (*model_work_items)++;
    }
    if (pending->dev_ptr + pending->size <= ct2d->device_mem_size &&
        cxl_type2_fabric_access_allowed(ct2d, pending->dev_ptr, pending->size,
                                        true, false)) {
      uint8_t *mem = memory_region_get_ram_ptr(&ct2d->device_mem);
      if (mem) {
        memcpy(mem + pending->dev_ptr,
               pending->direct_source ? pending->direct_host : pending->staging,
               pending->size);
        if (ct2d->bar_coherency.enabled) {
          cxl_bar_notify_gpu_access(&ct2d->bar_coherency, pending->dev_ptr,
                                    pending->size, true);
        }
      }
    }
    bool pooled = false;
    int64_t release_start_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
    int result;

    if (pending->direct_source) {
      CXLType2DirectSource *source = pending->source;

      g_assert(pending->source && pending->source->pending_refcount > 0);
      g_assert(ct2d->paired_case.active_direct_pending_copies > 0);
      g_assert(ct2d->paired_case.active_direct_pending_bytes >= pending->size);
      source->pending_refcount--;
      ct2d->paired_case.active_direct_pending_copies--;
      ct2d->paired_case.active_direct_pending_bytes -= pending->size;
      result = CXL_GPU_SUCCESS;
      if (!source->pending_refcount && source->auto_unregister) {
        result = cxl_type2_direct_source_unregister(ct2d, source->source_id);
      }
    } else {
      result = cxl_type2_htod_staging_release(ct2d, pending->staging,
                                              pending->staging_capacity,
                                              pending->staging_id, &pooled);
    }
    int64_t release_duration_ns =
        qemu_clock_get_ns(QEMU_CLOCK_HOST) - release_start_ns;

    CXLType2CudaAllocationIdentity population_identity;

    if (cxl_type2_cuda_allocation_identity_for_address(
            &ct2d->cuda_allocations, pending->dev_ptr, &population_identity) &&
        !cxl_type2_cuda_generation_population_complete(
            &ct2d->cuda_allocations, pending->dev_ptr, pending->size)) {
      cxl_type2_generation_reuse_fail(
          ct2d, ct2d->cuda_allocations.first_generation_error);
    }

    ct2d->htod_pending_copies--;
    ct2d->htod_pending_bytes -=
        pending->direct_source ? pending->size : pending->staging_capacity;
    if (!pending->direct_source) {
      g_assert(ct2d->paired_case.active_htod_staging_pending_bytes >=
               pending->staging_capacity);
      ct2d->paired_case.active_htod_staging_pending_bytes -=
          pending->staging_capacity;
    }
    if (ct2d->paired_case.qemu_cuda_calls_enabled) {
      qemu_log("CXL TYPE2 TRACE copy_completion original_call_id=0x%016" PRIx64
               " completion_call_id=0x%016" PRIx64
               " direction=htod sequence=%" PRIu64 " bytes=%zu"
               " staging_id=%" PRIu64 " capacity_bytes=%zu"
               " completion=%s lifetime_ns=%" PRId64
               " disposition=%s release_duration_ns=%" PRId64
               " free_result=%d pending_copies=%" PRIu64
               " pending_bytes=%" PRIu64 " peak_pending_copies=%" PRIu64
               " peak_pending_bytes=%" PRIu64 " pooled_buffers=%" PRIu64
               " pooled_bytes=%" PRIu64 " peak_pooled_bytes=%" PRIu64
               " pool_hits=%" PRIu64 " pool_misses=%" PRIu64
               " driver_allocations=%" PRIu64 " driver_frees=%" PRIu64
               " evictions=%" PRIu64 "\n",
               pending->call_id, ct2d->gpu_cmd.call_id, pending->sequence,
               pending->size, pending->staging_id, pending->staging_capacity,
               completion,
               qemu_clock_get_ns(QEMU_CLOCK_HOST) - pending->enqueue_host_ns,
               pending->direct_source ? "direct-source"
               : pooled               ? "pool"
                                      : "driver-free",
               release_duration_ns, result, ct2d->htod_pending_copies,
               ct2d->htod_pending_bytes, ct2d->htod_peak_pending_copies,
               ct2d->htod_peak_pending_bytes, ct2d->htod_pooled_buffers,
               ct2d->htod_pooled_bytes, ct2d->htod_peak_pooled_bytes,
               ct2d->htod_pool_hits, ct2d->htod_pool_misses,
               ct2d->htod_driver_allocations, ct2d->htod_driver_frees,
               ct2d->htod_pool_evictions);
    }
    if (first_error == CXL_GPU_SUCCESS && result != CXL_GPU_SUCCESS) {
      first_error = result;
    }
    g_free(pending);
  }
  return first_error;
}

static uint64_t cxl_type2_latest_pending_htod_sequence(CXLType2State *ct2d,
                                                       HetGPUStream stream) {
  uint64_t latest = 0;

  for (CXLType2PendingHtoD *pending = ct2d->pending_htod; pending;
       pending = pending->next) {
    if (pending->stream == stream && pending->sequence > latest) {
      latest = pending->sequence;
    }
  }
  return latest;
}

static CXLType2EventHtoDMark *
cxl_type2_event_htod_mark(CXLType2State *ct2d, void *event, bool create) {
  CXLType2EventHtoDMark *mark;

  for (mark = ct2d->event_htod_marks; mark; mark = mark->next) {
    if (mark->event == event) {
      return mark;
    }
  }
  if (!create) {
    return NULL;
  }
  mark = g_new0(CXLType2EventHtoDMark, 1);
  mark->event = event;
  mark->next = ct2d->event_htod_marks;
  ct2d->event_htod_marks = mark;
  return mark;
}

static void cxl_type2_remove_event_htod_mark(CXLType2State *ct2d, void *event) {
  CXLType2EventHtoDMark **cursor = &ct2d->event_htod_marks;

  while (*cursor) {
    CXLType2EventHtoDMark *mark = *cursor;
    if (mark->event == event) {
      *cursor = mark->next;
      g_free(mark);
      return;
    }
    cursor = &mark->next;
  }
}

static void cxl_type2_clear_event_htod_marks(CXLType2State *ct2d) {
  while (ct2d->event_htod_marks) {
    CXLType2EventHtoDMark *mark = ct2d->event_htod_marks;
    ct2d->event_htod_marks = mark->next;
    g_free(mark);
  }
}

static uint32_t cxl_type2_count_live_handles(void **handles, uint32_t count) {
  uint32_t live = 0;

  for (uint32_t i = 0; i < count; i++) {
    if (handles[i]) {
      live++;
    }
  }
  return live;
}

static uint32_t cxl_type2_count_live_modules(const CXLType2ModuleEntry *modules,
                                             uint32_t count) {
  uint32_t live = 0;

  for (uint32_t i = 0; i < count; i++) {
    if (modules[i].handle) {
      live++;
    }
  }
  return live;
}

static int cxl_type2_clear_gpu_handles(CXLType2State *ct2d,
                                       uint64_t run_binding, uint32_t case_kind,
                                       uint64_t epoch) {
  HetGPUState *hetgpu = &ct2d->gpu_info.hetgpu_state;
  const char *operations[] = {
      "event_cleanup",
      "stream_cleanup",
      "link_cleanup",
      "module_cleanup",
  };
  int first_error = 0;
  int category_error;
  uint32_t completed;
  uint32_t total;
  bool log_stages =
      run_binding != 0 && epoch != 0 && case_kind != CXL_GPU_CASE_NONE;

  if (ct2d->gpu_cmd.num_modules || ct2d->gpu_cmd.num_functions ||
      ct2d->gpu_cmd.num_graphs || ct2d->gpu_cmd.num_graph_execs ||
      ct2d->gpu_cmd.num_graph_nodes || ct2d->gpu_cmd.num_link_states ||
      ct2d->gpu_cmd.num_streams || ct2d->gpu_cmd.num_events) {
    qemu_log("CXL Type2: GPU handle tables reset modules live=%u high_water=%u "
             "capacity=%zu functions live=%u high_water=%u capacity=%zu graphs "
             "live=%u execs live=%u nodes live=%u links live=%u streams "
             "live=%u events live=%u\n",
             ct2d->gpu_cmd.num_modules, ct2d->gpu_cmd.modules_high_water,
             ct2d->gpu_cmd.modules_capacity, ct2d->gpu_cmd.num_functions,
             ct2d->gpu_cmd.functions_high_water,
             ct2d->gpu_cmd.functions_capacity, ct2d->gpu_cmd.num_graphs,
             ct2d->gpu_cmd.num_graph_execs, ct2d->gpu_cmd.num_graph_nodes,
             ct2d->gpu_cmd.num_link_states, ct2d->gpu_cmd.num_streams,
             ct2d->gpu_cmd.num_events);
  }
  if (hetgpu->initialized) {
    category_error = 0;
    completed = 0;
    total = cxl_type2_count_live_handles(ct2d->gpu_cmd.events,
                                         ct2d->gpu_cmd.num_events);
    if (log_stages) {
      cxl_type2_log_kimi_cleanup_progress(run_binding, case_kind, epoch,
                                          operations[0], "begin", completed,
                                          total, 0, false);
    }
    for (uint32_t i = 0; i < ct2d->gpu_cmd.num_events; i++) {
      if (ct2d->gpu_cmd.events[i]) {
        int error = hetgpu_cuda_event_destroy(hetgpu, ct2d->gpu_cmd.events[i]);
        if (category_error == 0 && error != 0) {
          category_error = error;
        }
        if (first_error == 0 && error != 0) {
          first_error = error;
        }
        completed++;
        if (log_stages && cxl_type2_cleanup_progress_due(completed, total)) {
          cxl_type2_log_kimi_cleanup_progress(
              run_binding, case_kind, epoch, operations[0], "progress",
              completed, total, category_error, true);
        }
      }
    }
    if (log_stages) {
      cxl_type2_log_kimi_cleanup_progress(run_binding, case_kind, epoch,
                                          operations[0], "end", completed,
                                          total, category_error, true);
    }

    category_error = 0;
    completed = 0;
    total = cxl_type2_count_live_handles(ct2d->gpu_cmd.streams,
                                         ct2d->gpu_cmd.num_streams) +
            (ct2d->gpu_cmd.per_thread_stream != NULL);
    if (log_stages) {
      cxl_type2_log_kimi_cleanup_progress(run_binding, case_kind, epoch,
                                          operations[1], "begin", completed,
                                          total, 0, false);
    }
    if (ct2d->gpu_cmd.per_thread_stream) {
      int error =
          hetgpu_cuda_stream_destroy(hetgpu, ct2d->gpu_cmd.per_thread_stream);

      if (category_error == 0 && error != 0) {
        category_error = error;
      }
      if (first_error == 0 && error != 0) {
        first_error = error;
      }
      ct2d->gpu_cmd.per_thread_stream = NULL;
      completed++;
    }
    for (uint32_t i = 0; i < ct2d->gpu_cmd.num_streams; i++) {
      if (ct2d->gpu_cmd.streams[i]) {
        int error =
            hetgpu_cuda_stream_destroy(hetgpu, ct2d->gpu_cmd.streams[i]);
        if (category_error == 0 && error != 0) {
          category_error = error;
        }
        if (first_error == 0 && error != 0) {
          first_error = error;
        }
        completed++;
        if (log_stages && cxl_type2_cleanup_progress_due(completed, total)) {
          cxl_type2_log_kimi_cleanup_progress(
              run_binding, case_kind, epoch, operations[1], "progress",
              completed, total, category_error, true);
        }
      }
    }
    if (log_stages) {
      cxl_type2_log_kimi_cleanup_progress(run_binding, case_kind, epoch,
                                          operations[1], "end", completed,
                                          total, category_error, true);
    }

    for (uint32_t i = 0; i < ct2d->gpu_cmd.num_graph_execs; i++) {
      if (ct2d->gpu_cmd.graph_execs[i]) {
        int error = hetgpu_cuda_graph_exec_destroy(
            hetgpu, ct2d->gpu_cmd.graph_execs[i]);

        if (first_error == 0 && error != 0) {
          first_error = error;
        }
        if (error == CXL_GPU_SUCCESS) {
          ct2d->gpu_cmd.graph_execs[i] = NULL;
        }
      }
    }
    cxl_type2_graph_generation_consumers_clear(ct2d);
    for (uint32_t i = 0; i < ct2d->gpu_cmd.num_graphs; i++) {
      if (ct2d->gpu_cmd.graphs[i]) {
        int error = hetgpu_cuda_graph_destroy(hetgpu, ct2d->gpu_cmd.graphs[i]);

        if (first_error == 0 && error != 0) {
          first_error = error;
        }
        if (error == CXL_GPU_SUCCESS) {
          ct2d->gpu_cmd.graphs[i] = NULL;
        }
      }
    }

    category_error = 0;
    completed = 0;
    total = cxl_type2_count_live_handles(ct2d->gpu_cmd.link_states,
                                         ct2d->gpu_cmd.num_link_states);
    if (log_stages) {
      cxl_type2_log_kimi_cleanup_progress(run_binding, case_kind, epoch,
                                          operations[2], "begin", completed,
                                          total, 0, false);
    }
    for (uint32_t i = 0; i < ct2d->gpu_cmd.num_link_states; i++) {
      if (ct2d->gpu_cmd.link_states[i]) {
        int error =
            hetgpu_cuda_link_destroy(hetgpu, ct2d->gpu_cmd.link_states[i]);
        if (category_error == 0 && error != 0) {
          category_error = error;
        }
        if (first_error == 0 && error != 0) {
          first_error = error;
        }
        completed++;
        if (log_stages && cxl_type2_cleanup_progress_due(completed, total)) {
          cxl_type2_log_kimi_cleanup_progress(
              run_binding, case_kind, epoch, operations[2], "progress",
              completed, total, category_error, true);
        }
      }
    }
    if (log_stages) {
      cxl_type2_log_kimi_cleanup_progress(run_binding, case_kind, epoch,
                                          operations[2], "end", completed,
                                          total, category_error, true);
    }

    category_error = 0;
    completed = 0;
    total = cxl_type2_count_live_modules(ct2d->gpu_cmd.modules,
                                         ct2d->gpu_cmd.num_modules);
    if (log_stages) {
      cxl_type2_log_kimi_cleanup_progress(run_binding, case_kind, epoch,
                                          operations[3], "begin", completed,
                                          total, 0, false);
    }
    for (uint32_t i = 0; i < ct2d->gpu_cmd.num_modules; i++) {
      if (ct2d->gpu_cmd.modules[i].handle) {
        int error =
            hetgpu_unload_module(hetgpu, ct2d->gpu_cmd.modules[i].handle);
        if (category_error == 0 && error != 0) {
          category_error = error;
        }
        if (first_error == 0 && error != 0) {
          first_error = error;
        }
        completed++;
        if (log_stages && cxl_type2_cleanup_progress_due(completed, total)) {
          cxl_type2_log_kimi_cleanup_progress(
              run_binding, case_kind, epoch, operations[3], "progress",
              completed, total, category_error, true);
        }
      }
    }
    if (log_stages) {
      cxl_type2_log_kimi_cleanup_progress(run_binding, case_kind, epoch,
                                          operations[3], "end", completed,
                                          total, category_error, true);
    }
  }
  if (log_stages) {
    cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch,
                                  "handle_table_release", "begin", 0, false);
  }
  for (uint32_t i = 0; i < ct2d->gpu_cmd.num_functions; i++) {
    g_free(ct2d->gpu_cmd.functions[i].name);
  }
  g_free(ct2d->gpu_cmd.modules);
  g_free(ct2d->gpu_cmd.functions);
  g_free(ct2d->gpu_cmd.graphs);
  g_free(ct2d->gpu_cmd.graph_execs);
  g_free(ct2d->gpu_cmd.graph_nodes);
  g_free(ct2d->gpu_cmd.link_states);
  g_free(ct2d->gpu_cmd.streams);
  g_free(ct2d->gpu_cmd.events);
  g_clear_pointer(&ct2d->gpu_cmd.capturing_stream_wires, g_hash_table_destroy);
  if (!hetgpu->initialized) {
    cxl_type2_graph_generation_consumers_clear(ct2d);
  }
  cxl_type2_clear_event_htod_marks(ct2d);
  ct2d->gpu_cmd.modules = NULL;
  ct2d->gpu_cmd.functions = NULL;
  ct2d->gpu_cmd.graphs = NULL;
  ct2d->gpu_cmd.graph_execs = NULL;
  ct2d->gpu_cmd.graph_nodes = NULL;
  ct2d->gpu_cmd.link_states = NULL;
  ct2d->gpu_cmd.streams = NULL;
  ct2d->gpu_cmd.events = NULL;
  ct2d->gpu_cmd.per_thread_stream = NULL;
  ct2d->gpu_cmd.modules_capacity = 0;
  ct2d->gpu_cmd.functions_capacity = 0;
  ct2d->gpu_cmd.graphs_capacity = 0;
  ct2d->gpu_cmd.graph_execs_capacity = 0;
  ct2d->gpu_cmd.graph_nodes_capacity = 0;
  ct2d->gpu_cmd.link_states_capacity = 0;
  ct2d->gpu_cmd.streams_capacity = 0;
  ct2d->gpu_cmd.events_capacity = 0;
  ct2d->gpu_cmd.num_modules = 0;
  ct2d->gpu_cmd.num_functions = 0;
  ct2d->gpu_cmd.num_graphs = 0;
  ct2d->gpu_cmd.num_graph_execs = 0;
  ct2d->gpu_cmd.num_graph_nodes = 0;
  ct2d->gpu_cmd.num_link_states = 0;
  ct2d->gpu_cmd.num_streams = 0;
  ct2d->gpu_cmd.num_events = 0;
  ct2d->gpu_cmd.modules_high_water = 0;
  ct2d->gpu_cmd.functions_high_water = 0;
  if (log_stages) {
    cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch,
                                  "handle_table_release", "end", 0, false);
  }
  return first_error;
}

static uint64_t cxl_type2_paired_config_binding(CXLType2State *ct2d,
                                                uint32_t case_kind,
                                                const char *aof_path,
                                                const char *restore_path,
                                                const char *manifest_path) {
  g_autofree char *identity = g_strdup_printf(
      "kimi-case-config-v2\ncase=%u\nenabled=%u\nrestore=0\n"
      "min-allocation=%" PRIu64 "\nmax-regions=%" PRIu64
      "\ncheckpoint-enabled=%u\ncheckpoint-every=%" PRIu64
      "\nqemu-cuda-calls=%u\nconcordia-runtime-details=%u\naof=%s\n"
      "restore-aof=%s\nmanifest=%s\n",
      case_kind, case_kind == CXL_GPU_CASE_CONCORDIA,
      ct2d->paired_case.min_allocation_bytes, ct2d->paired_case.max_regions,
      ct2d->paired_case.checkpoint_enabled,
      ct2d->paired_case.checkpoint_every_launches,
      ct2d->paired_case.qemu_cuda_calls_enabled,
      ct2d->paired_case.concordia_runtime_details_enabled, aof_path,
      restore_path, manifest_path);
  g_autofree char *digest =
      g_compute_checksum_for_string(G_CHECKSUM_SHA256, identity, -1);
  char prefix[17];
  uint64_t binding;

  memcpy(prefix, digest, 16);
  prefix[16] = '\0';
  if (qemu_strtou64(prefix, NULL, 16, &binding) < 0) {
    return 1;
  }
  return binding ? binding : 1;
}

static HetGPUError cxl_type2_reset_formal_backend(CXLType2State *ct2d,
                                                  uint64_t run_binding,
                                                  uint32_t case_kind,
                                                  uint64_t epoch) {
  HetGPUState *hetgpu = &ct2d->gpu_info.hetgpu_state;
  HetGPUError error;
  int cleanup_error;

  cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch,
                                "handle_cleanup_before_reset", "begin", 0,
                                false);
  cleanup_error =
      cxl_type2_clear_gpu_handles(ct2d, run_binding, case_kind, epoch);
  cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch,
                                "handle_cleanup_before_reset", "end",
                                cleanup_error, true);
  cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch,
                                "formal_backend_reset", "begin", 0, false);
  error = hetgpu_reset_formal(hetgpu, HETGPU_BACKEND_NVIDIA,
                              ct2d->gpu_info.hetgpu_device_index,
                              ct2d->gpu_info.hetgpu_lib_path);
  if (error != HETGPU_SUCCESS) {
    cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch,
                                  "formal_backend_reset", "end", error, true);
    return error;
  }
  if (!hetgpu->initialized || hetgpu->backend != HETGPU_BACKEND_NVIDIA ||
      !hetgpu->context || !hetgpu->formal_case_strict ||
      !hetgpu->kimi_case_begin_v2 || !hetgpu->kimi_case_end_v2) {
    qemu_log("CXL Type2: formal backend identity incomplete "
             "initialized=%d backend=%u context=%p begin=%p end=%p\n",
             hetgpu->initialized, hetgpu->backend, hetgpu->context,
             hetgpu->kimi_case_begin_v2, hetgpu->kimi_case_end_v2);
    error = HETGPU_ERROR_INVALID_CONTEXT;
    cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch,
                                  "formal_backend_reset", "end", error, true);
    return error;
  }

  hetgpu_set_coherency_callback(hetgpu, cxl_type2_hetgpu_coherency_callback,
                                ct2d);
  ct2d->gpu_info.passthrough_enabled = true;
  ct2d->gpu_info.gpu_mem_size = hetgpu->props.total_memory;
  cxl_type2_cuda_allocation_table_reset(&ct2d->cuda_allocations);
  cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch,
                                "formal_backend_reset", "end", HETGPU_SUCCESS,
                                true);
  return HETGPU_SUCCESS;
}

static void cxl_type2_paired_case_begin(CXLType2State *ct2d,
                                        uint64_t trace_sequence) {
  HetGPUState *hetgpu = &ct2d->gpu_info.hetgpu_state;
  uint32_t protocol = ct2d->gpu_cmd.params[0];
  uint32_t case_kind = ct2d->gpu_cmd.params[1];
  uint64_t run_binding = ct2d->gpu_cmd.params[2];
  uint64_t flags = ct2d->gpu_cmd.params[3];
  uint64_t epoch = ct2d->paired_case.next_epoch;
  const char *case_name = cxl_type2_paired_case_name(case_kind);
  g_autofree char *case_dir_name = NULL;
  g_autofree char *case_dir = NULL;
  g_autofree char *aof_path = NULL;
  g_autofree char *restore_path = NULL;
  g_autofree char *manifest_path = NULL;
  HetGPUKimiCaseBeginV2 input = {0};
  HetGPUKimiCaseResultV1 result = {0};
  HetGPUError error;
  uint64_t config_binding;

  memset(ct2d->gpu_cmd.results, 0, sizeof(ct2d->gpu_cmd.results));
  if (!ct2d->paired_case.required) {
    ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_SUPPORTED;
    return;
  }
  if (ct2d->paired_case.failed) {
    ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_DEINITIALIZED;
    return;
  }
  if (protocol != CXL_GPU_CASE_PROTOCOL_VERSION || flags != 0 ||
      (case_kind != CXL_GPU_CASE_BASELINE &&
       case_kind != CXL_GPU_CASE_CONCORDIA) ||
      run_binding != ct2d->paired_case.run_binding ||
      ct2d->paired_case.active_case != CXL_GPU_CASE_NONE) {
    ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
    return;
  }

  case_dir_name = g_strdup_printf("%" PRIu64 "-%s", epoch, case_name);
  case_dir = g_build_filename(ct2d->paired_case.run_root, "cases",
                              case_dir_name, NULL);
  if (g_mkdir_with_parents(case_dir, 0700) != 0) {
    qemu_log("CXL Type2: create paired case directory %s failed: %s\n",
             case_dir, strerror(errno));
    ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_UNKNOWN;
    return;
  }
  aof_path = g_build_filename(case_dir, "kimi.aof", NULL);
  restore_path = g_build_filename(case_dir, "restore.aof", NULL);
  manifest_path = g_build_filename(case_dir, "kimi.manifest.json", NULL);
  config_binding = cxl_type2_paired_config_binding(ct2d, case_kind, aof_path,
                                                   restore_path, manifest_path);

  cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch, "case_begin",
                                "begin", 0, false);
  error = cxl_type2_reset_formal_backend(ct2d, run_binding, case_kind, epoch);
  if (error != HETGPU_SUCCESS) {
    ct2d->paired_case.failed = true;
    ct2d->paired_case.failure_code = error;
    ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_CONTEXT;
    cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch, "case_begin",
                                  "end", error, true);
    return;
  }

  input.abi_version = HETGPU_KIMI_CASE_ABI_VERSION;
  input.struct_size = sizeof(input);
  input.case_kind = case_kind;
  input.epoch = epoch;
  input.run_binding = run_binding;
  input.config_binding = config_binding;
  input.min_allocation_bytes = ct2d->paired_case.min_allocation_bytes;
  input.max_regions = ct2d->paired_case.max_regions;
  input.checkpoint_every_launches = ct2d->paired_case.checkpoint_every_launches;
  input.flags = ct2d->paired_case.checkpoint_enabled
                    ? 0
                    : HETGPU_KIMI_CASE_FLAG_CHECKPOINT_DISABLED;
  input.enabled = case_kind == CXL_GPU_CASE_CONCORDIA;
  input.runtime_details_enabled =
      ct2d->paired_case.concordia_runtime_details_enabled;
  hetgpu->detailed_logs = ct2d->paired_case.qemu_cuda_calls_enabled;
  input.aof_path = (const uint8_t *)aof_path;
  input.aof_path_len = strlen(aof_path);
  input.restore_aof_path = (const uint8_t *)restore_path;
  input.restore_aof_path_len = strlen(restore_path);
  input.manifest_path = (const uint8_t *)manifest_path;
  input.manifest_path_len = strlen(manifest_path);

  cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch,
                                "concordia_case_begin", "begin", 0, false);
  error = hetgpu_kimi_case_begin(hetgpu, &input, &result);
  cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch,
                                "concordia_case_begin", "end", error, true);
  if (error != HETGPU_SUCCESS ||
      result.abi_version != HETGPU_KIMI_CASE_ABI_VERSION ||
      result.struct_size != sizeof(result) ||
      result.outcome != HETGPU_KIMI_CASE_SUCCEEDED ||
      result.case_kind != case_kind || result.epoch != epoch ||
      result.run_binding != run_binding ||
      result.config_binding != config_binding) {
    HetGPUError cleanup_error;
    int direct_cleanup_error;
    int handle_error;
    int cleanup_status;

    qemu_log("CXL Type2: Concordia case begin failed error=%u "
             "outcome=%u reason=%u detail=%.*s\n",
             error, result.outcome, result.reason,
             (int)MIN(result.error_len, HETGPU_KIMI_CASE_ERROR_BYTES),
             result.error);
    cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch,
                                  "failed_begin_cleanup", "begin", 0, false);
    direct_cleanup_error = cxl_type2_direct_sources_cleanup(ct2d);
    if (direct_cleanup_error != CXL_GPU_SUCCESS) {
      qemu_log("CXL Type2: direct source cleanup after failed case "
               "begin failed: %d\n",
               direct_cleanup_error);
    }
    cleanup_error = hetgpu_cleanup_formal(hetgpu);
    if (cleanup_error != HETGPU_SUCCESS) {
      qemu_log("CXL Type2: cleanup after failed case begin also "
               "failed: %u\n",
               cleanup_error);
    }
    handle_error =
        cxl_type2_clear_gpu_handles(ct2d, run_binding, case_kind, epoch);
    cleanup_status = direct_cleanup_error != CXL_GPU_SUCCESS
                         ? direct_cleanup_error
                     : cleanup_error != HETGPU_SUCCESS ? cleanup_error
                                                       : handle_error;
    cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch,
                                  "failed_begin_cleanup", "end", cleanup_status,
                                  true);
    ct2d->gpu_info.passthrough_enabled = false;
    ct2d->paired_case.failed = true;
    ct2d->paired_case.failure_code = error != HETGPU_SUCCESS ? error
                                     : result.reason         ? result.reason
                                                     : HETGPU_ERROR_UNKNOWN;
    ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_UNKNOWN;
    cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch, "case_begin",
                                  "end",
                                  error != HETGPU_SUCCESS ? error
                                  : result.reason         ? result.reason
                                                  : HETGPU_ERROR_UNKNOWN,
                                  true);
    return;
  }

  ct2d->paired_case.active_case = case_kind;
  ct2d->paired_case.active_epoch = epoch;
  ct2d->paired_case.active_first_sequence = trace_sequence;
  ct2d->paired_case.active_config_binding = config_binding;
  cxl_type2_reset_case_summary(ct2d);
  hetgpu_set_driver_interval_callback(hetgpu, cxl_type2_record_driver_interval,
                                      ct2d);
  cxl_type2_notify_case_scope(run_binding, case_kind, epoch, true);
  ct2d->paired_case.next_epoch++;
  ct2d->gpu_cmd.results[0] = epoch;
  ct2d->gpu_cmd.results[1] = case_kind;
  ct2d->gpu_cmd.results[2] = trace_sequence;
  ct2d->gpu_cmd.results[3] = config_binding;
  cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch, "case_begin",
                                "end", HETGPU_SUCCESS, true);
  qemu_log("KIMI_CASE_BEGIN run_binding=%" PRIu64 " case=%s epoch=%" PRIu64
           " first_sequence=%" PRIu64 " config_binding=%" PRIu64
           " gpu_backend=%s\n",
           run_binding, case_name, epoch, trace_sequence, config_binding,
           hetgpu_get_backend_name(hetgpu->backend));
}

static void cxl_type2_paired_case_end(CXLType2State *ct2d,
                                      uint64_t trace_sequence) {
  HetGPUState *hetgpu = &ct2d->gpu_info.hetgpu_state;
  uint32_t protocol = ct2d->gpu_cmd.params[0];
  uint64_t expected_epoch = ct2d->gpu_cmd.params[1];
  int32_t application_exit = (int32_t)ct2d->gpu_cmd.params[2];
  uint64_t run_binding = ct2d->gpu_cmd.params[3];
  uint64_t epoch = ct2d->paired_case.active_epoch;
  uint32_t case_kind = ct2d->paired_case.active_case;
  HetGPUKimiCaseEndV2 input = {0};
  HetGPUKimiCaseResultV1 result = {0};
  HetGPUError sync_error;
  HetGPUError concordia_error;
  HetGPUError reset_error;
  int pool_clear_result;
  int handle_clear_result;
  int alias_clear_result;
  int cleanup_status;
  int64_t sync_begin_ns;
  int64_t sync_end_ns;
  uint64_t model_htod_completed = 0;
  uint64_t consumer_launches_completed = 0;
  bool result_valid;
  bool case_success;

  memset(ct2d->gpu_cmd.results, 0, sizeof(ct2d->gpu_cmd.results));
  if (!ct2d->paired_case.required) {
    ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_SUPPORTED;
    return;
  }
  if (protocol != CXL_GPU_CASE_PROTOCOL_VERSION ||
      case_kind == CXL_GPU_CASE_NONE || expected_epoch != epoch ||
      run_binding != ct2d->paired_case.run_binding) {
    ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
    return;
  }

  cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch, "case_end",
                                "begin", 0, false);
  if (ct2d->paired_case.decode_command_scope.active) {
    cxl_type2_finish_decode_command_scope(ct2d, cxl_type2_host_monotonic_ns(),
                                          trace_sequence, false);
  }
  cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch,
                                "backend_synchronize", "begin", 0, false);
  sync_begin_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
  sync_error = hetgpu_synchronize(hetgpu);
  sync_end_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
  if (sync_error == HETGPU_SUCCESS &&
      cxl_type2_release_pending_htod(ct2d, NULL, true, UINT64_MAX, "case-end",
                                     &model_htod_completed) !=
          CXL_GPU_SUCCESS) {
    sync_error = HETGPU_ERROR_UNKNOWN;
  }
  if (sync_error == HETGPU_SUCCESS &&
      cxl_type2_release_pending_alias_consumers(ct2d, NULL, true, UINT64_MAX,
                                                &consumer_launches_completed) !=
          CXL_GPU_SUCCESS) {
    sync_error = HETGPU_ERROR_UNKNOWN;
  }
  cxl_type2_record_model_supply_completion(ct2d, sync_begin_ns, sync_end_ns,
                                           true, model_htod_completed,
                                           consumer_launches_completed);
  if (sync_error == HETGPU_SUCCESS &&
      cxl_type2_direct_sources_release(ct2d) != CXL_GPU_SUCCESS) {
    sync_error = HETGPU_ERROR_UNKNOWN;
  }
  pool_clear_result = cxl_type2_clear_htod_staging_pool(ct2d);
  if (sync_error == HETGPU_SUCCESS && pool_clear_result != CXL_GPU_SUCCESS) {
    sync_error = HETGPU_ERROR_UNKNOWN;
  }
  cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch,
                                "backend_synchronize", "end", sync_error, true);
  input.abi_version = HETGPU_KIMI_CASE_ABI_VERSION;
  input.struct_size = sizeof(input);
  input.epoch = epoch;
  input.run_binding = run_binding;
  input.config_binding = ct2d->paired_case.active_config_binding;
  input.application_exit = application_exit;
  cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch,
                                "concordia_case_end", "begin", 0, false);
  concordia_error = hetgpu_kimi_case_end(hetgpu, &input, &result);
  cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch,
                                "concordia_case_end", "end", concordia_error,
                                true);
  result_valid = result.abi_version == HETGPU_KIMI_CASE_ABI_VERSION &&
                 result.struct_size == sizeof(result) &&
                 result.outcome == HETGPU_KIMI_CASE_SUCCEEDED &&
                 result.case_kind == case_kind && result.epoch == epoch &&
                 result.run_binding == run_binding &&
                 result.config_binding == input.config_binding;
  case_success = application_exit == 0 && sync_error == HETGPU_SUCCESS &&
                 concordia_error == HETGPU_SUCCESS && result_valid;
  if (case_success) {
    /* Keep READY visible so the next guest can submit CASE_BEGIN. */
    cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch,
                                  "formal_backend_cleanup", "begin", 0, false);
    cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch,
                                  "formal_backend_cleanup", "end", 0, false);
    reset_error = HETGPU_SUCCESS;
  } else {
    int direct_cleanup_error;

    cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch,
                                  "formal_backend_cleanup", "begin", 0, false);
    direct_cleanup_error = cxl_type2_direct_sources_cleanup(ct2d);
    if (direct_cleanup_error != CXL_GPU_SUCCESS) {
      qemu_log("CXL Type2: direct source cleanup after failed case "
               "end failed: %d\n",
               direct_cleanup_error);
    }
    reset_error = hetgpu_cleanup_formal(hetgpu);
    if (reset_error == HETGPU_SUCCESS) {
      for (size_t i = 0; i < ct2d->cuda_allocations.count; i++) {
        if (!cxl_type2_cuda_generation_release(
                &ct2d->cuda_allocations,
                ct2d->cuda_allocations.entries[i].base)) {
          cxl_type2_generation_reuse_fail(
              ct2d, ct2d->cuda_allocations.first_generation_error);
          break;
        }
      }
    }
    if (direct_cleanup_error != CXL_GPU_SUCCESS &&
        reset_error == HETGPU_SUCCESS) {
      reset_error = HETGPU_ERROR_UNKNOWN;
    }
    cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch,
                                  "formal_backend_cleanup", "end", reset_error,
                                  true);
    ct2d->gpu_info.passthrough_enabled = false;
  }
  handle_clear_result =
      cxl_type2_clear_gpu_handles(ct2d, run_binding, case_kind, epoch);
  alias_clear_result = cxl_type2_release_model_aliases(ct2d);
  cleanup_status = reset_error != HETGPU_SUCCESS ? reset_error
                   : handle_clear_result != CXL_GPU_SUCCESS
                       ? handle_clear_result
                       : alias_clear_result;
  if (handle_clear_result != CXL_GPU_SUCCESS ||
      alias_clear_result != CXL_GPU_SUCCESS) {
    case_success = false;
  }

  ct2d->gpu_cmd.results[0] = epoch;
  ct2d->gpu_cmd.results[1] = trace_sequence;
  ct2d->gpu_cmd.results[2] = result.outcome;
  ct2d->gpu_cmd.results[3] = cleanup_status;

  {
    int64_t interval_end_ns = cxl_type2_host_monotonic_ns();
    CXLType2IntervalIdentity end_identity = cxl_type2_interval_identity(
        trace_sequence, "qemu", "case", "case-end", 0, false);

    if (hetgpu->driver_interval_callback != cxl_type2_record_driver_interval ||
        hetgpu->driver_interval_opaque != ct2d) {
      cxl_type2_interval_fail(
          &ct2d->paired_case.case_command_scope.driver_intervals,
          "driver-observer-detached");
    }
    hetgpu_set_driver_interval_callback(hetgpu, NULL, NULL);
    cxl_type2_command_scope_finish(&ct2d->paired_case.case_command_scope,
                                   interval_end_ns, end_identity);
  }
  cxl_type2_log_case_summary(ct2d, run_binding, case_kind, epoch);
  cxl_type2_log_model_supply_terminal(
      ct2d, case_success, sync_error == HETGPU_SUCCESS, cleanup_status);
  cxl_type2_notify_case_scope(run_binding, case_kind, epoch, false);

  cxl_type2_log_kimi_case_stage(
      run_binding, case_kind, epoch, "case_end", "end",
      application_exit != 0               ? HETGPU_ERROR_UNKNOWN
      : sync_error != HETGPU_SUCCESS      ? sync_error
      : concordia_error != HETGPU_SUCCESS ? concordia_error
      : !result_valid ? (result.reason ? result.reason : HETGPU_ERROR_UNKNOWN)
      : cleanup_status != CXL_GPU_SUCCESS ? cleanup_status
                                          : HETGPU_SUCCESS,
      true);

  qemu_log("KIMI_CASE_END run_binding=%" PRIu64 " case=%s epoch=%" PRIu64
           " last_sequence=%" PRIu64 " app_exit=%d sync_status=%u"
           " concordia_status=%u concordia_reason=%u stateful_launches=%" PRIu64
           " reset_status=%u detail=%.*s\n",
           run_binding, cxl_type2_paired_case_name(case_kind), epoch,
           trace_sequence, application_exit, sync_error, result.outcome,
           result.reason, result.stateful_launches, reset_error,
           (int)MIN(result.error_len, HETGPU_KIMI_CASE_ERROR_BYTES),
           result.error);

  ct2d->paired_case.active_case = CXL_GPU_CASE_NONE;
  ct2d->paired_case.active_epoch = 0;
  ct2d->paired_case.active_first_sequence = 0;
  ct2d->paired_case.active_config_binding = 0;

  if (application_exit != 0 || sync_error != HETGPU_SUCCESS ||
      concordia_error != HETGPU_SUCCESS || !result_valid ||
      reset_error != HETGPU_SUCCESS || handle_clear_result != CXL_GPU_SUCCESS ||
      alias_clear_result != CXL_GPU_SUCCESS) {
    ct2d->paired_case.failed = true;
    if (application_exit != 0) {
      ct2d->paired_case.failure_code = (uint32_t)application_exit;
    } else if (sync_error != HETGPU_SUCCESS) {
      ct2d->paired_case.failure_code = sync_error;
    } else if (concordia_error != HETGPU_SUCCESS) {
      ct2d->paired_case.failure_code = concordia_error;
    } else if (!result_valid) {
      ct2d->paired_case.failure_code =
          result.reason ? result.reason : HETGPU_ERROR_UNKNOWN;
    } else if (reset_error != HETGPU_SUCCESS) {
      ct2d->paired_case.failure_code = reset_error;
    } else {
      ct2d->paired_case.failure_code = handle_clear_result != CXL_GPU_SUCCESS
                                           ? handle_clear_result
                                           : alias_clear_result;
    }
    ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_UNKNOWN;
  }
}

static void cxl_type2_gpu_execute_cmd(CXLType2State *ct2d, uint32_t cmd) {
  HetGPUState *hetgpu = &ct2d->gpu_info.hetgpu_state;
  HetGPUError err;
  uint64_t dev_ptr;
  size_t size;
  uint64_t trace_sequence = ++ct2d->gpu_cmd.trace_sequence;
  int64_t trace_start_ns = cxl_type2_host_monotonic_ns();
  uint64_t stream_work_wire = 0;
  CXLGPUStreamSyncReason stream_sync_reason = CXL_GPU_STREAM_SYNC_PUBLIC_API;
  bool stream_sync_reason_valid = true;
  bool stream_sync_elided = false;
  bool stream_progress_command = cxl_type2_cuda_stream_progress_wire(
      cmd, ct2d->gpu_cmd.params, &stream_work_wire);
  if (cmd == CXL_GPU_CMD_STREAM_SYNC) {
    uint32_t protocol_version = ct2d->gpu_cmd.descriptor
                                    ? ct2d->gpu_cmd.descriptor->protocol_version
                                    : 1U;

    stream_sync_reason_valid = cxl_type2_cuda_decode_stream_sync_reason(
        protocol_version, ct2d->gpu_cmd.params[1], &stream_sync_reason);
  }
  bool elide_stream_sync =
      cmd == CXL_GPU_CMD_STREAM_SYNC && stream_sync_reason_valid &&
      cxl_type2_cuda_adjacent_stream_sync_can_elide(
          ct2d->paired_case.last_command_was_successful_stream_sync,
          ct2d->paired_case.last_successful_stream_sync_wire,
          ct2d->gpu_cmd.params[0]);

  ct2d->paired_case.last_command_was_successful_stream_sync = false;

  if (stream_progress_command) {
    ct2d->paired_case.active_stream_work_commands++;
  }

  if (ct2d->paired_case.qemu_cuda_calls_enabled)
    qemu_log("CXL TYPE2 TRACE cmd_begin seq=%" PRIu64 " call_id=0x%016" PRIx64
             " cmd=0x%x host_ns=%" PRId64 " p0=0x%" PRIx64 " p1=%" PRIu64 "\n",
             trace_sequence, ct2d->gpu_cmd.call_id, cmd, trace_start_ns,
             ct2d->gpu_cmd.params[0], ct2d->gpu_cmd.params[1]);

  if (ct2d->paired_case.qemu_cuda_calls_enabled)
    qemu_log_mask(LOG_GUEST_ERROR,
                  "CXL GPU: execute cmd 0x%x, hetgpu_init=%d, ctx=%p\n", cmd,
                  hetgpu->initialized, hetgpu->context);

  ct2d->gpu_cmd.cmd_status = CXL_GPU_CMD_STATUS_RUNNING;
  ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
  hetgpu_cuda_trace_set_detailed_logs(
      ct2d->paired_case.qemu_cuda_calls_enabled);
  hetgpu_cuda_trace_set_call_id(ct2d->gpu_cmd.call_id);
  if (ct2d->paired_case.active_case != CXL_GPU_CASE_NONE) {
    ct2d->paired_case.active_command_sequence = trace_sequence;
    ct2d->paired_case.active_command_code = cmd;
    ct2d->paired_case.active_stream_sync_reason = stream_sync_reason;
    ct2d->paired_case.active_stream_sync_reason_valid =
        cmd == CXL_GPU_CMD_STREAM_SYNC && stream_sync_reason_valid;
  }

  if (ct2d->paired_case.required && ct2d->paired_case.failed) {
    ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_DEINITIALIZED;
    goto complete;
  }

  if (ct2d->paired_case.required &&
      cxl_type2_command_requires_formal_case(cmd) &&
      ct2d->paired_case.active_case == CXL_GPU_CASE_NONE) {
    ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_READY;
    goto complete;
  }

  switch (cmd) {
  case CXL_GPU_CMD_NOP:
    break;

  case CXL_GPU_CMD_OBSERVATION_ANCHOR:
    if (ct2d->gpu_cmd.params[0] != CXL_GPU_OBSERVATION_ANCHOR_VERSION ||
        (ct2d->gpu_cmd.params[1] != CXL_GPU_OBSERVATION_ANCHOR_DECODE_BEGIN &&
         ct2d->gpu_cmd.params[1] != CXL_GPU_OBSERVATION_ANCHOR_DECODE_END) ||
        (ct2d->paired_case.required &&
         ct2d->paired_case.active_case == CXL_GPU_CASE_NONE) ||
        (ct2d->paired_case.active_case != CXL_GPU_CASE_NONE &&
         (ct2d->gpu_cmd.params[2] != ct2d->paired_case.active_epoch ||
          (ct2d->gpu_cmd.params[1] == CXL_GPU_OBSERVATION_ANCHOR_DECODE_BEGIN &&
           ct2d->paired_case.decode_command_scope.active) ||
          (ct2d->gpu_cmd.params[1] == CXL_GPU_OBSERVATION_ANCHOR_DECODE_END &&
           !ct2d->paired_case.decode_command_scope.active)))) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      break;
    }
    {
      struct timespec monotonic_before;
      struct timespec realtime;
      struct timespec monotonic_after;

      if (clock_gettime(CLOCK_MONOTONIC, &monotonic_before) != 0 ||
          clock_gettime(CLOCK_REALTIME, &realtime) != 0 ||
          clock_gettime(CLOCK_MONOTONIC, &monotonic_after) != 0) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_UNKNOWN;
        break;
      }
      int64_t host_before_ns =
          monotonic_before.tv_sec * NANOSECONDS_PER_SECOND +
          monotonic_before.tv_nsec;
      int64_t host_realtime_ns =
          realtime.tv_sec * NANOSECONDS_PER_SECOND + realtime.tv_nsec;
      int64_t host_after_ns = monotonic_after.tv_sec * NANOSECONDS_PER_SECOND +
                              monotonic_after.tv_nsec;

      if (host_before_ns <= 0 || host_realtime_ns <= 0 ||
          host_after_ns < host_before_ns) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_UNKNOWN;
        break;
      }
      ct2d->gpu_cmd.results[0] =
          host_before_ns + (host_after_ns - host_before_ns) / 2;
      ct2d->gpu_cmd.results[1] = host_realtime_ns;
      ct2d->gpu_cmd.results[2] = (host_after_ns - host_before_ns + 1) / 2;
      ct2d->gpu_cmd.results[3] = CXL_GPU_OBSERVATION_ANCHOR_VERSION;
      if (ct2d->paired_case.active_case != CXL_GPU_CASE_NONE) {
        int64_t host_mid_ns = ct2d->gpu_cmd.results[0];

        if (ct2d->gpu_cmd.params[1] ==
            CXL_GPU_OBSERVATION_ANCHOR_DECODE_BEGIN) {
          cxl_type2_command_scope_reset(
              &ct2d->paired_case.decode_command_scope, host_mid_ns,
              cxl_type2_interval_identity(trace_sequence, "qemu", "decode",
                                          "decode-begin", 0, false));
        } else {
          cxl_type2_finish_decode_command_scope(ct2d, host_mid_ns,
                                                trace_sequence, true);
        }
      }
    }
    break;

  case CXL_GPU_CMD_INIT:
    if (!hetgpu->initialized) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
    }
    break;

  case CXL_GPU_CMD_GET_ERROR_STRING: {
    const char *string = NULL;
    size_t length;
    int result = hetgpu_cuda_get_error_string(
        hetgpu, (int)ct2d->gpu_cmd.params[0], &string);

    ct2d->gpu_cmd.cmd_result = result;
    if (result != CXL_GPU_SUCCESS)
      break;
    if (!string) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      break;
    }
    length = strnlen(string, ct2d->gpu_cmd.data_size);
    if (length == ct2d->gpu_cmd.data_size) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      break;
    }
    memcpy(ct2d->gpu_cmd.data, string, length + 1);
    ct2d->gpu_cmd.results[0] = length;
  } break;

  case CXL_GPU_CMD_GET_DEVICE_COUNT:
    ct2d->gpu_cmd.results[0] = hetgpu->initialized ? 1 : 0;
    break;

  case CXL_GPU_CMD_GET_DEVICE:
    if (ct2d->gpu_cmd.params[0] == 0 && hetgpu->initialized) {
      ct2d->gpu_cmd.results[0] = 0; /* Device handle */
    } else {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_DEVICE;
    }
    break;

  case CXL_GPU_CMD_GET_DEVICE_PROPS:
    if (hetgpu->initialized) {
      HetGPUDeviceProps props;
      err = hetgpu_get_device_props(hetgpu, &props);
      if (err == HETGPU_SUCCESS) {
        /* Copy device name to data buffer */
        memcpy(ct2d->gpu_cmd.data, props.name, sizeof(props.name));
        ct2d->gpu_cmd.results[0] = props.total_memory;
        ct2d->gpu_cmd.results[1] = (props.compute_capability_major << 16) |
                                   props.compute_capability_minor;
        ct2d->gpu_cmd.results[2] = props.multiprocessor_count;
        ct2d->gpu_cmd.results[3] = props.max_threads_per_block;
      } else {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_DEVICE;
      }
    } else {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
    }
    break;

  case CXL_GPU_CMD_GET_TOTAL_MEM:
    memset(ct2d->gpu_cmd.results, 0, sizeof(ct2d->gpu_cmd.results));
    if (!hetgpu->initialized) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
      break;
    }
    {
      size_t total_bytes = 0;
      int cuda_result = hetgpu_cuda_device_total_memory(hetgpu, &total_bytes);
      ct2d->gpu_cmd.cmd_result = cuda_result;
      if (cuda_result == CXL_GPU_SUCCESS) {
        ct2d->gpu_cmd.results[0] = total_bytes;
      }
      qemu_log("CXL TYPE2 CUDA total_mem driver_result=%d total=%zu\n",
               cuda_result, total_bytes);
    }
    break;

  case CXL_GPU_CMD_GET_DEVICE_ATTRIBUTE:
    memset(ct2d->gpu_cmd.results, 0, sizeof(ct2d->gpu_cmd.results));
    {
      CXLType2CudaAttributeRequest request = {
          .hetgpu = hetgpu,
      };
      int cuda_result;

      if (!cxl_type2_cuda_attribute_wire_is_valid(ct2d->gpu_cmd.params[0])) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
        break;
      }
      if (!hetgpu->initialized) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
        break;
      }
      if (!cxl_type2_cuda_dispatch_attribute(ct2d->gpu_cmd.params[0],
                                             cxl_type2_cuda_query_attribute,
                                             &request, &cuda_result)) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
        break;
      }
      ct2d->gpu_cmd.cmd_result = cuda_result;
      if (cuda_result == CXL_GPU_SUCCESS) {
        ct2d->gpu_cmd.results[0] = (uint64_t)(int64_t)(int32_t)request.value;
      }
      qemu_log("CXL TYPE2 CUDA device_attribute attribute=%d "
               "driver_result=%d value=%d\n",
               (int)(int32_t)ct2d->gpu_cmd.params[0], cuda_result,
               request.value);
    }
    break;

  case CXL_GPU_CMD_GET_ERROR_NAME: {
    const char *name = NULL;
    size_t length;
    int cuda_result =
        hetgpu_cuda_get_error_name(hetgpu, (int)ct2d->gpu_cmd.params[0], &name);

    ct2d->gpu_cmd.cmd_result = cuda_result;
    if (cuda_result != CXL_GPU_SUCCESS) {
      break;
    }
    if (!name) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      break;
    }
    length = strnlen(name, ct2d->gpu_cmd.data_size);
    if (length == ct2d->gpu_cmd.data_size) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      break;
    }
    memcpy(ct2d->gpu_cmd.data, name, length + 1);
    ct2d->gpu_cmd.results[0] = length;
  } break;

  case CXL_GPU_CMD_CASE_BEGIN:
    cxl_type2_paired_case_begin(ct2d, trace_sequence);
    break;

  case CXL_GPU_CMD_CASE_END:
    cxl_type2_paired_case_end(ct2d, trace_sequence);
    break;

  case CXL_GPU_CMD_CTX_CREATE:
    if (hetgpu->initialized) {
      err = hetgpu_create_context(hetgpu);
      if (err == HETGPU_SUCCESS && !ct2d->gpu_cmd.per_thread_stream) {
        void *stream = NULL;
        int stream_result = hetgpu_cuda_stream_create(hetgpu, 1U, &stream);

        if (stream_result == CXL_GPU_SUCCESS) {
          ct2d->gpu_cmd.per_thread_stream = stream;
        } else {
          ct2d->gpu_cmd.cmd_result = stream_result;
          break;
        }
      }
      if (err == HETGPU_SUCCESS && ct2d->paired_case.active_epoch != 0) {
        ct2d->gpu_cmd.results[0] = ct2d->paired_case.active_epoch;
      } else {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_CONTEXT;
      }
    } else {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
    }
    break;

  case CXL_GPU_CMD_CTX_DESTROY:
    /* Context destroyed on device cleanup */
    break;

  case CXL_GPU_CMD_CTX_SYNC:
    if (hetgpu->initialized) {
      int64_t wait_begin_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
      uint64_t model_htod_completed = 0;
      uint64_t consumer_launches_completed = 0;

      err = hetgpu_synchronize(hetgpu);
      int64_t wait_end_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
      if (err == HETGPU_SUCCESS &&
          cxl_type2_release_pending_htod(
              ct2d, NULL, true, UINT64_MAX, "context-sync",
              &model_htod_completed) != CXL_GPU_SUCCESS) {
        err = HETGPU_ERROR_UNKNOWN;
      }
      if (err == HETGPU_SUCCESS &&
          cxl_type2_release_pending_alias_consumers(
              ct2d, NULL, true, UINT64_MAX, &consumer_launches_completed) !=
              CXL_GPU_SUCCESS) {
        err = HETGPU_ERROR_UNKNOWN;
      }
      cxl_type2_record_model_supply_completion(ct2d, wait_begin_ns, wait_end_ns,
                                               true, model_htod_completed,
                                               consumer_launches_completed);
      if (err != HETGPU_SUCCESS) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_UNKNOWN;
      }
    }
    break;

  case CXL_GPU_CMD_MEM_GET_INFO:
    memset(ct2d->gpu_cmd.results, 0, sizeof(ct2d->gpu_cmd.results));
    {
      CXLType2CudaMemInfoRequest request = {
          .hetgpu = hetgpu,
      };
      int cuda_result;

      if (!cxl_type2_cuda_dispatch_mem_info(
              ct2d->paired_case.active_case != CXL_GPU_CASE_NONE,
              hetgpu->context != NULL, ct2d->gpu_cmd.params[0],
              ct2d->paired_case.active_epoch, cxl_type2_cuda_query_mem_info,
              &request, &cuda_result)) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_CONTEXT;
        break;
      }
      ct2d->gpu_cmd.cmd_result = cuda_result;
      if (cuda_result == CXL_GPU_SUCCESS) {
        ct2d->gpu_cmd.results[0] = request.free_bytes;
        ct2d->gpu_cmd.results[1] = request.total_bytes;
      }
      qemu_log("CXL TYPE2 CUDA mem_info token=%" PRIu64 " active_epoch=%" PRIu64
               " active_case=%u "
               "live_context=%u driver_result=%d free=%zu total=%zu\n",
               ct2d->gpu_cmd.params[0], ct2d->paired_case.active_epoch,
               ct2d->paired_case.active_case, hetgpu->context != NULL,
               cuda_result, request.free_bytes, request.total_bytes);
    }
    break;

  case CXL_GPU_CMD_MEM_GET_POINTER_MEMORY_TYPE: {
    int memory_type = 0;
    int cuda_result = hetgpu_pointer_get_memory_type(
        hetgpu, ct2d->gpu_cmd.params[0], &memory_type);

    ct2d->gpu_cmd.cmd_result = cuda_result;
    if (cuda_result == CXL_GPU_SUCCESS) {
      ct2d->gpu_cmd.results[0] = (uint64_t)memory_type;
    }
  } break;

  case CXL_GPU_CMD_MEM_ALLOC:
    size = ct2d->gpu_cmd.params[0];
    if (hetgpu->initialized) {
      err = hetgpu_malloc(hetgpu, size, HETGPU_MEM_DEFAULT, &dev_ptr);
      if (err == HETGPU_SUCCESS) {
        ct2d->gpu_cmd.results[0] = dev_ptr;
        if (!cxl_type2_cuda_allocation_record(&ct2d->cuda_allocations, dev_ptr,
                                              size, NULL)) {
          cxl_type2_cuda_classifier_reject(
              ct2d, CXL_TYPE2_CUDA_CLASSIFIER_UNAVAILABLE,
              CXL_TYPE2_CUDA_REJECTION_ALLOCATION_MISSING,
              CXL_GPU_CMD_MEM_ALLOC, true);
        }
      } else {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_OUT_OF_MEMORY;
      }
    } else {
      /* Fallback: allocate from device memory region */
      /* Simple bump allocator - in real impl, use proper allocator */
      static uint64_t next_alloc = 0;
      if (next_alloc + size <= ct2d->device_mem_size) {
        ct2d->gpu_cmd.results[0] = next_alloc;
        next_alloc += (size + 0xFFF) & ~0xFFF; /* Page align */
      } else {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_OUT_OF_MEMORY;
      }
    }
    break;

  case CXL_GPU_CMD_MEM_FREE: {
    CXLType2CudaAllocation *allocation;
    CXLType2CudaAllocationIdentity identity;
    GPtrArray *release_events = NULL;

    dev_ptr = ct2d->gpu_cmd.params[0];
    allocation =
        cxl_type2_cuda_allocation_identity_for_address(&ct2d->cuda_allocations,
                                                       dev_ptr, &identity)
            ? cxl_type2_cuda_allocation_find(&ct2d->cuda_allocations,
                                             identity.base, identity.epoch)
            : NULL;
    if (allocation && allocation->pageable_alias) {
      release_events = g_ptr_array_new();
      for (const CXLType2CudaPageableAlias *alias = allocation->pageable_alias;
           alias; alias = alias->next) {
        g_ptr_array_add(release_events,
                        cxl_type2_model_supply_alias_release_event(
                            ct2d, allocation, alias));
      }
    }
    if (hetgpu->initialized) {
      if (allocation && cxl_type2_model_aliases_unregister(ct2d, allocation) !=
                            CXL_GPU_SUCCESS) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_UNKNOWN;
        err = HETGPU_ERROR_UNKNOWN;
      } else {
        err = hetgpu_free(hetgpu, dev_ptr);
      }
      if (err != HETGPU_SUCCESS) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      } else if (!cxl_type2_cuda_allocation_forget(&ct2d->cuda_allocations,
                                                   dev_ptr)) {
        cxl_type2_cuda_classifier_reject(
            ct2d, CXL_TYPE2_CUDA_CLASSIFIER_UNAVAILABLE,
            CXL_TYPE2_CUDA_REJECTION_ALLOCATION_MISSING, CXL_GPU_CMD_MEM_FREE,
            true);
      } else if (release_events) {
        for (size_t i = 0; i < release_events->len; i++) {
          cxl_type2_model_supply_emit("KIMI_MODEL_SUPPLY_ALIAS",
                                      g_ptr_array_index(release_events, i));
          ct2d->model_supply.alias_releases++;
        }
        g_ptr_array_unref(release_events);
        release_events = NULL;
      }
    }
    if (release_events) {
      for (size_t i = 0; i < release_events->len; i++) {
        QDict *event = g_ptr_array_index(release_events, i);

        qobject_unref(event);
      }
      g_ptr_array_unref(release_events);
    }
  }
  /* Fallback: no-op for simple allocator */
  break;

  case CXL_GPU_CMD_MEM_COPY_HTOD:
    dev_ptr = ct2d->gpu_cmd.params[0]; /* dst device ptr */
    size = ct2d->gpu_cmd.params[1];    /* size */
    /* Data is in ct2d->gpu_cmd.data buffer */
    if (size > ct2d->gpu_cmd.data_size) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      break;
    }
    if (hetgpu->initialized) {
      int64_t driver_start_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
      err = hetgpu_memcpy_htod(hetgpu, dev_ptr, ct2d->gpu_cmd.data, size);
      if (ct2d->paired_case.qemu_cuda_calls_enabled)
        qemu_log(
            "CXL TYPE2 TRACE copy_driver call_id=0x%016" PRIx64
            " direction=htod bytes=%zu driver_duration_ns=%" PRId64
            " backend_result=%d implementation=blocking stream_forwarded=0\n",
            ct2d->gpu_cmd.call_id, size,
            qemu_clock_get_ns(QEMU_CLOCK_HOST) - driver_start_ns, err);
      if (err != HETGPU_SUCCESS) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      } else {
        CXLType2CudaAllocationIdentity population_identity;

        if (cxl_type2_cuda_allocation_identity_for_address(
                &ct2d->cuda_allocations, dev_ptr, &population_identity) &&
            !cxl_type2_cuda_generation_population_complete(
                &ct2d->cuda_allocations, dev_ptr, size)) {
          cxl_type2_generation_reuse_fail(
              ct2d, ct2d->cuda_allocations.first_generation_error);
        }
      }
      /* Also update shadow copy in device_mem for coherency tracking */
      if (dev_ptr + size <= ct2d->device_mem_size &&
          cxl_type2_fabric_access_allowed(ct2d, dev_ptr, size, true, false)) {
        uint8_t *mem = memory_region_get_ram_ptr(&ct2d->device_mem);
        if (mem) {
          memcpy(mem + dev_ptr, ct2d->gpu_cmd.data, size);
          /* Notify BAR coherency layer of GPU write */
          if (ct2d->bar_coherency.enabled) {
            cxl_bar_notify_gpu_access(&ct2d->bar_coherency, dev_ptr, size,
                                      true);
          }
        }
      }
    } else {
      /* Fallback: copy to device memory region */
      if (dev_ptr + size <= ct2d->device_mem_size &&
          cxl_type2_fabric_access_allowed(ct2d, dev_ptr, size, true, false)) {
        uint8_t *mem = memory_region_get_ram_ptr(&ct2d->device_mem);
        if (mem) {
          memcpy(mem + dev_ptr, ct2d->gpu_cmd.data, size);
          /* Notify BAR coherency layer of GPU write */
          if (ct2d->bar_coherency.enabled) {
            cxl_bar_notify_gpu_access(&ct2d->bar_coherency, dev_ptr, size,
                                      true);
          }
        }
      } else {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      }
    }
    break;

  case CXL_GPU_CMD_MEM_COPY_HTOD_ASYNC: {
    void *stream = NULL;

    dev_ptr = ct2d->gpu_cmd.params[0];
    size = ct2d->gpu_cmd.params[1];
    if (!size || size > ct2d->gpu_cmd.data_size ||
        !cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[2], &stream)) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      break;
    }
    ct2d->gpu_cmd.cmd_result = cxl_type2_enqueue_htod_from_host(
        ct2d, dev_ptr, ct2d->gpu_cmd.data, size, stream, "bar2");
  } break;

  case CXL_GPU_CMD_BATCH_HTOD_ASYNC: {
    uint64_t range_count = ct2d->gpu_cmd.params[0];
    uint64_t payload_bytes = ct2d->gpu_cmd.params[1];
    uint64_t fail_idx = SIZE_MAX;
    uint64_t enqueued = 0;
    uint64_t source_bytes_before =
        ct2d->paired_case.active_payload_source_bytes;
    void *stream = NULL;
    CXLType2BatchHtoDEnqueueContext context;

    ct2d->gpu_cmd.results[0] = SIZE_MAX;
    ct2d->gpu_cmd.results[1] = 0;
    ct2d->paired_case.active_payload_batches++;
    if (!hetgpu->initialized) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
      break;
    }
    if (!ct2d->gpu_cmd.batch_data ||
        !cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[2], &stream)) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      break;
    }
    context = (CXLType2BatchHtoDEnqueueContext){
        .ct2d = ct2d,
        .stream = stream,
    };
    ct2d->gpu_cmd.cmd_result = cxl_gpu_batch_htod_submit(
        ct2d->gpu_cmd.batch_data, CXL_GPU_BATCH_DATA_SIZE, range_count,
        payload_bytes, cxl_type2_batch_htod_enqueue_one, &context, &fail_idx,
        &enqueued);
    ct2d->gpu_cmd.results[0] = fail_idx;
    ct2d->gpu_cmd.results[1] = enqueued;
    if (ct2d->gpu_cmd.cmd_result == CXL_GPU_SUCCESS &&
        ct2d->model_supply.route == CXL_TYPE2_MODEL_SUPPLY_SELECTED_HTOD) {
      ct2d->model_supply.model_htod_calls++;
      ct2d->model_supply.model_htod_bytes +=
          ct2d->paired_case.active_payload_source_bytes - source_bytes_before;
    }
  } break;

  case CXL_GPU_CMD_MEM_COPY_DTOH:
    dev_ptr = ct2d->gpu_cmd.params[0]; /* src device ptr */
    size = ct2d->gpu_cmd.params[1];    /* size */
    if (size > ct2d->gpu_cmd.data_size) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      break;
    }
    if (hetgpu->initialized) {
      /* Notify BAR coherency layer before GPU read */
      if (ct2d->bar_coherency.enabled) {
        cxl_bar_notify_gpu_access(&ct2d->bar_coherency, dev_ptr, size, false);
      }
      int64_t driver_start_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
      err = hetgpu_memcpy_dtoh(hetgpu, ct2d->gpu_cmd.data, dev_ptr, size);
      if (ct2d->paired_case.qemu_cuda_calls_enabled)
        qemu_log(
            "CXL TYPE2 TRACE copy_driver call_id=0x%016" PRIx64
            " direction=dtoh bytes=%zu driver_duration_ns=%" PRId64
            " backend_result=%d implementation=blocking stream_forwarded=0\n",
            ct2d->gpu_cmd.call_id, size,
            qemu_clock_get_ns(QEMU_CLOCK_HOST) - driver_start_ns, err);
      if (err != HETGPU_SUCCESS) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      }
      /* Update shadow copy from GPU for coherency */
      if (dev_ptr + size <= ct2d->device_mem_size &&
          cxl_type2_fabric_access_allowed(ct2d, dev_ptr, size, false, false)) {
        uint8_t *mem = memory_region_get_ram_ptr(&ct2d->device_mem);
        if (mem) {
          memcpy(mem + dev_ptr, ct2d->gpu_cmd.data, size);
        }
      }
    } else {
      /* Fallback: copy from device memory region */
      if (dev_ptr + size <= ct2d->device_mem_size &&
          cxl_type2_fabric_access_allowed(ct2d, dev_ptr, size, false, false)) {
        uint8_t *mem = memory_region_get_ram_ptr(&ct2d->device_mem);
        if (mem) {
          /* Notify BAR coherency layer before GPU read */
          if (ct2d->bar_coherency.enabled) {
            cxl_bar_notify_gpu_access(&ct2d->bar_coherency, dev_ptr, size,
                                      false);
          }
          memcpy(ct2d->gpu_cmd.data, mem + dev_ptr, size);
        }
      } else {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      }
    }
    break;

  case CXL_GPU_CMD_MEM_COPY_DTOD: {
    uint64_t dst_dev_ptr = ct2d->gpu_cmd.params[0];
    uint64_t src_dev_ptr = ct2d->gpu_cmd.params[1];
    size_t xfer_size = ct2d->gpu_cmd.params[2];

    if (hetgpu->initialized) {
      int64_t driver_start_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
      err = hetgpu_memcpy_dtod(hetgpu, dst_dev_ptr, src_dev_ptr, xfer_size);
      if (ct2d->paired_case.qemu_cuda_calls_enabled)
        qemu_log("CXL TYPE2 TRACE copy_driver call_id=0x%016" PRIx64
                 " direction=dtod bytes=%zu driver_duration_ns=%" PRId64
                 " backend_result=%d implementation=blocking-direct "
                 "stream_forwarded=0\n",
                 ct2d->gpu_cmd.call_id, xfer_size,
                 qemu_clock_get_ns(QEMU_CLOCK_HOST) - driver_start_ns, err);
      if (err != HETGPU_SUCCESS) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
        break;
      }
    } else {
      uint8_t *mem = memory_region_get_ram_ptr(&ct2d->device_mem);
      if (!mem || src_dev_ptr > ct2d->device_mem_size ||
          xfer_size > ct2d->device_mem_size - src_dev_ptr ||
          dst_dev_ptr > ct2d->device_mem_size ||
          xfer_size > ct2d->device_mem_size - dst_dev_ptr ||
          !cxl_type2_fabric_access_allowed(ct2d, src_dev_ptr, xfer_size, false,
                                           false) ||
          !cxl_type2_fabric_access_allowed(ct2d, dst_dev_ptr, xfer_size, true,
                                           false)) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
        break;
      }
      memmove(mem + dst_dev_ptr, mem + src_dev_ptr, xfer_size);
    }

    if (ct2d->bar_coherency.enabled) {
      cxl_bar_notify_gpu_access(&ct2d->bar_coherency, src_dev_ptr, xfer_size,
                                false);
      cxl_bar_notify_gpu_access(&ct2d->bar_coherency, dst_dev_ptr, xfer_size,
                                true);
    }
    if (hetgpu->initialized && src_dev_ptr <= ct2d->device_mem_size &&
        xfer_size <= ct2d->device_mem_size - src_dev_ptr &&
        dst_dev_ptr <= ct2d->device_mem_size &&
        xfer_size <= ct2d->device_mem_size - dst_dev_ptr) {
      uint8_t *mem = memory_region_get_ram_ptr(&ct2d->device_mem);
      if (mem) {
        memmove(mem + dst_dev_ptr, mem + src_dev_ptr, xfer_size);
      }
    }
  } break;

  case CXL_GPU_CMD_MEM_COPY_DTOD_ASYNC: {
    uint64_t dst_dev_ptr = ct2d->gpu_cmd.params[0];
    uint64_t src_dev_ptr = ct2d->gpu_cmd.params[1];
    size_t xfer_size = ct2d->gpu_cmd.params[2];
    void *stream = NULL;

    if (!cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[3], &stream)) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
      break;
    }
    ct2d->gpu_cmd.cmd_result = hetgpu_cuda_memcpy_dtod_async(
        hetgpu, dst_dev_ptr, src_dev_ptr, xfer_size, stream);
    if (ct2d->paired_case.qemu_cuda_calls_enabled) {
      qemu_log("CXL TYPE2 TRACE copy_driver call_id=0x%016" PRIx64
               " direction=dtod bytes=%zu backend_result=%u "
               "implementation=async-direct stream_forwarded=1\n",
               ct2d->gpu_cmd.call_id, xfer_size, ct2d->gpu_cmd.cmd_result);
    }
  } break;

  case CXL_GPU_CMD_MEM_COPY_2D_DTOD: {
    uint64_t dst_dev_ptr = ct2d->gpu_cmd.params[0];
    uint64_t src_dev_ptr = ct2d->gpu_cmd.params[1];
    size_t dst_pitch = ct2d->gpu_cmd.params[2];
    size_t src_pitch = ct2d->gpu_cmd.params[3];
    size_t width = ct2d->gpu_cmd.params[4];
    size_t height = ct2d->gpu_cmd.params[5];
    size_t total_bytes;

    if (!width || !height || width > src_pitch || width > dst_pitch ||
        height > SIZE_MAX / width || src_dev_ptr > UINT64_MAX - width ||
        dst_dev_ptr > UINT64_MAX - width ||
        (height - 1 &&
         (src_pitch > (UINT64_MAX - src_dev_ptr - width) / (height - 1) ||
          dst_pitch > (UINT64_MAX - dst_dev_ptr - width) / (height - 1)))) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      break;
    }
    total_bytes = width * height;

    int64_t driver_start_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
    void *copy_stream = NULL;
    bool copy_async = false;
    if (ct2d->gpu_cmd.descriptor &&
        ct2d->gpu_cmd.descriptor->protocol_version >= 2) {
      uint64_t stream_wire = ct2d->gpu_cmd.params[6];
      if (stream_wire != CXL_GPU_STREAM_WIRE_NULL &&
          stream_wire != CXL_GPU_STREAM_WIRE_LEGACY &&
          cxl_type2_stream_from_wire(ct2d, stream_wire, &copy_stream)) {
        copy_async = true;
      }
    }
    if (hetgpu->initialized && hetgpu->backend != HETGPU_BACKEND_SIMULATION) {
      if (copy_async) {
        err = hetgpu_memcpy2d_dtod_async(hetgpu, dst_dev_ptr, dst_pitch,
                                         src_dev_ptr, src_pitch, width, height,
                                         copy_stream);
      } else {
        err = hetgpu_memcpy2d_dtod(hetgpu, dst_dev_ptr, dst_pitch, src_dev_ptr,
                                   src_pitch, width, height);
      }
    } else if (hetgpu->backend == HETGPU_BACKEND_SIMULATION) {
      err = HETGPU_SUCCESS;
      for (size_t row = 0; row < height; row++) {
        err = hetgpu_memcpy_dtod(hetgpu, dst_dev_ptr + row * dst_pitch,
                                 src_dev_ptr + row * src_pitch, width);
        if (err != HETGPU_SUCCESS) {
          break;
        }
      }
    } else {
      err = HETGPU_ERROR_INVALID_CONTEXT;
    }
    if (ct2d->paired_case.qemu_cuda_calls_enabled)
      qemu_log("CXL TYPE2 TRACE copy_driver call_id=0x%016" PRIx64
               " direction=dtod bytes=%zu rows=%zu row_commands_eliminated=%zu "
               "driver_duration_ns=%" PRId64
               " backend_result=%d implementation=%s "
               "stream_forwarded=%d\n",
               ct2d->gpu_cmd.call_id, total_bytes, height, height - 1,
               qemu_clock_get_ns(QEMU_CLOCK_HOST) - driver_start_ns, err,
               copy_async ? "async-stream-2d" : "blocking-direct-2d",
               copy_async ? 1 : 0);
    if (err != HETGPU_SUCCESS) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      break;
    }

    for (size_t row = 0; row < height; row++) {
      uint64_t src = src_dev_ptr + row * src_pitch;
      uint64_t dst = dst_dev_ptr + row * dst_pitch;
      if (ct2d->bar_coherency.enabled) {
        cxl_bar_notify_gpu_access(&ct2d->bar_coherency, src, width, false);
        cxl_bar_notify_gpu_access(&ct2d->bar_coherency, dst, width, true);
      }
      if (src <= ct2d->device_mem_size &&
          width <= ct2d->device_mem_size - src &&
          dst <= ct2d->device_mem_size &&
          width <= ct2d->device_mem_size - dst) {
        uint8_t *mem = memory_region_get_ram_ptr(&ct2d->device_mem);
        if (mem) {
          memmove(mem + dst, mem + src, width);
        }
      }
    }
  } break;

  case CXL_GPU_CMD_MODULE_LOAD_PTX:
    if (hetgpu->initialized) {
      /* PTX source is in data buffer */
      void *module = NULL;
      if (!cxl_type2_reserve_gpu_entries(
              (void **)&ct2d->gpu_cmd.modules, &ct2d->gpu_cmd.modules_capacity,
              ct2d->gpu_cmd.num_modules, sizeof(*ct2d->gpu_cmd.modules))) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_OUT_OF_MEMORY;
        break;
      }
      err = hetgpu_load_ptx(hetgpu, (const char *)ct2d->gpu_cmd.data,
                            (HetGPUModule *)&module);
      if (err == HETGPU_SUCCESS) {
        CXLType2ModuleEntry *entry =
            &ct2d->gpu_cmd.modules[ct2d->gpu_cmd.num_modules];

        entry->handle = module;
        entry->load_kind = CXL_TYPE2_MODULE_LOAD_PTX;
        ct2d->gpu_cmd.results[0] = ct2d->gpu_cmd.num_modules;
        ct2d->gpu_cmd.num_modules++;
        ct2d->gpu_cmd.modules_high_water =
            MAX(ct2d->gpu_cmd.modules_high_water, ct2d->gpu_cmd.num_modules);
      } else {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_PTX;
      }
    } else {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
    }
    break;

  case CXL_GPU_CMD_MODULE_LOAD_CUBIN:
    if (hetgpu->initialized) {
      size = ct2d->gpu_cmd.params[0];
      uint32_t encoding = ct2d->gpu_cmd.params[1];
      size_t uncompressed_size = ct2d->gpu_cmd.params[2];
      if (!size || size > ct2d->gpu_cmd.data_size) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
        break;
      }

      const void *cubin_data = ct2d->gpu_cmd.data;
      size_t cubin_size = size;
      void *decoded = NULL;
      if (encoding != 0 && encoding != CXL_GPU_MODULE_DATA_ZSTD &&
          encoding != CXL_GPU_MODULE_DATA_LZ4) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
        break;
      }
      if (encoding != 0) {
        if (!uncompressed_size || uncompressed_size > 64 * MiB) {
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
          break;
        }
        decoded = g_try_malloc(uncompressed_size);
        if (!decoded) {
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_OUT_OF_MEMORY;
          break;
        }
        if (encoding == CXL_GPU_MODULE_DATA_ZSTD) {
          size_t result = ZSTD_decompress(decoded, uncompressed_size,
                                          ct2d->gpu_cmd.data, size);
          if (ZSTD_isError(result) || result != uncompressed_size) {
            qemu_log("CXL hetGPU: CUBIN Zstd decode failed: %s result=%zu "
                     "expected=%zu\n",
                     ZSTD_getErrorName(result), result, uncompressed_size);
            g_free(decoded);
            ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
            break;
          }
          cubin_size = result;
        } else {
          int result =
              LZ4_decompress_safe((const char *)ct2d->gpu_cmd.data, decoded,
                                  (int)size, (int)uncompressed_size);
          if (result < 0 || (size_t)result != uncompressed_size) {
            qemu_log(
                "CXL hetGPU: CUBIN LZ4 decode failed: result=%d expected=%zu\n",
                result, uncompressed_size);
            g_free(decoded);
            ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
            break;
          }
          cubin_size = (size_t)result;
        }
        cubin_data = decoded;
      }

      qemu_log("CXL TYPE2 TRACE module_payload call_id=0x%016" PRIx64
               " encoding=%s input_size=%zu decoded_size=%zu\n",
               ct2d->gpu_cmd.call_id,
               encoding == CXL_GPU_MODULE_DATA_ZSTD  ? "zstd"
               : encoding == CXL_GPU_MODULE_DATA_LZ4 ? "lz4"
                                                     : "raw",
               size, cubin_size);

      void *module = NULL;
      g_autoptr(GChecksum) cubin_checksum = g_checksum_new(G_CHECKSUM_SHA256);
      if (!cxl_type2_reserve_gpu_entries(
              (void **)&ct2d->gpu_cmd.modules, &ct2d->gpu_cmd.modules_capacity,
              ct2d->gpu_cmd.num_modules, sizeof(*ct2d->gpu_cmd.modules))) {
        g_free(decoded);
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_OUT_OF_MEMORY;
        break;
      }
      g_checksum_update(cubin_checksum, cubin_data, cubin_size);
      err = hetgpu_load_cubin(hetgpu, cubin_data, cubin_size,
                              (HetGPUModule *)&module);
      if (err == HETGPU_SUCCESS) {
        CXLType2ModuleEntry *entry =
            &ct2d->gpu_cmd.modules[ct2d->gpu_cmd.num_modules];

        entry->handle = module;
        entry->load_kind = CXL_TYPE2_MODULE_LOAD_CUBIN;
        pstrcpy(entry->driver_input_cubin_sha256,
                sizeof(entry->driver_input_cubin_sha256),
                g_checksum_get_string(cubin_checksum));
        ct2d->gpu_cmd.results[0] = ct2d->gpu_cmd.num_modules;
        ct2d->gpu_cmd.num_modules++;
        ct2d->gpu_cmd.modules_high_water =
            MAX(ct2d->gpu_cmd.modules_high_water, ct2d->gpu_cmd.num_modules);
      } else {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_PTX;
      }
      g_free(decoded);
    } else {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
    }
    break;

  case CXL_GPU_CMD_FUNC_GET:
    if (hetgpu->initialized) {
      uint32_t module_id = ct2d->gpu_cmd.params[0];
      /* Function name is in data buffer */
      if (module_id < ct2d->gpu_cmd.num_modules &&
          ct2d->gpu_cmd.modules[module_id].handle) {
        void *func = NULL;
        CXLType2FunctionEntry *entry;

        if (!memchr(ct2d->gpu_cmd.data, '\0', ct2d->gpu_cmd.data_size)) {
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
          break;
        }
        if (!cxl_type2_reserve_gpu_entries((void **)&ct2d->gpu_cmd.functions,
                                           &ct2d->gpu_cmd.functions_capacity,
                                           ct2d->gpu_cmd.num_functions,
                                           sizeof(*ct2d->gpu_cmd.functions))) {
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_OUT_OF_MEMORY;
          break;
        }
        err = hetgpu_get_function(
            hetgpu, ct2d->gpu_cmd.modules[module_id].handle,
            (const char *)ct2d->gpu_cmd.data, (HetGPUFunction *)&func);
        if (err == HETGPU_SUCCESS) {
          entry = &ct2d->gpu_cmd.functions[ct2d->gpu_cmd.num_functions];
          entry->handle = func;
          entry->module_id = module_id;
          entry->name = g_strdup((const char *)ct2d->gpu_cmd.data);
          if (ct2d->gpu_cmd.modules[module_id].load_kind ==
              CXL_TYPE2_MODULE_LOAD_CUBIN) {
            entry->consumer = cxl_type2_model_consumer_function_find(
                &ct2d->model_supply.consumer_certificate,
                ct2d->gpu_cmd.modules[module_id].driver_input_cubin_sha256,
                entry->name);
          }
          ct2d->gpu_cmd.results[0] = ct2d->gpu_cmd.num_functions;
          ct2d->gpu_cmd.num_functions++;
          ct2d->gpu_cmd.functions_high_water = MAX(
              ct2d->gpu_cmd.functions_high_water, ct2d->gpu_cmd.num_functions);
        } else {
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_FOUND;
        }
      } else {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
      }
    } else {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
    }
    break;

  case CXL_GPU_CMD_MODULE_UNLOAD:
    if (hetgpu->initialized) {
      uint32_t module_id = ct2d->gpu_cmd.params[0];
      if (module_id >= ct2d->gpu_cmd.num_modules ||
          !ct2d->gpu_cmd.modules[module_id].handle) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
        break;
      }
      err =
          hetgpu_unload_module(hetgpu, ct2d->gpu_cmd.modules[module_id].handle);
      if (err == HETGPU_SUCCESS) {
        memset(&ct2d->gpu_cmd.modules[module_id], 0,
               sizeof(ct2d->gpu_cmd.modules[module_id]));
      } else if (err == HETGPU_ERROR_INVALID_VALUE) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      } else if (err == HETGPU_ERROR_NOT_INITIALIZED) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
      } else if (err == HETGPU_ERROR_INVALID_CONTEXT) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_CONTEXT;
      } else if (err == HETGPU_ERROR_INVALID_HANDLE) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
      } else if (err == HETGPU_ERROR_NOT_SUPPORTED) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_SUPPORTED;
      } else {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_UNKNOWN;
      }
    } else {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
    }
    break;

  case CXL_GPU_CMD_MODULE_GET_GLOBAL:
    if (hetgpu->initialized) {
      uint32_t module_id = ct2d->gpu_cmd.params[0];
      if (module_id >= ct2d->gpu_cmd.num_modules ||
          !ct2d->gpu_cmd.modules[module_id].handle) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
        break;
      }
      if (!memchr(ct2d->gpu_cmd.data, '\0', ct2d->gpu_cmd.data_size)) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
        break;
      }

      dev_ptr = 0;
      size = 0;
      err =
          hetgpu_get_global(hetgpu, ct2d->gpu_cmd.modules[module_id].handle,
                            (const char *)ct2d->gpu_cmd.data, &dev_ptr, &size);
      if (err == HETGPU_SUCCESS) {
        ct2d->gpu_cmd.results[0] = dev_ptr;
        ct2d->gpu_cmd.results[1] = size;
      } else if (err == HETGPU_ERROR_NOT_FOUND) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_FOUND;
      } else if (err == HETGPU_ERROR_INVALID_HANDLE) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
      } else if (err == HETGPU_ERROR_NOT_SUPPORTED) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_SUPPORTED;
      } else if (err == HETGPU_ERROR_INVALID_CONTEXT) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_CONTEXT;
      } else if (err == HETGPU_ERROR_INVALID_VALUE) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      } else {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_UNKNOWN;
      }
    } else {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
    }
    break;

  case CXL_GPU_CMD_FUNC_GET_PARAM_LAYOUT:
    if (hetgpu->initialized) {
      uint32_t func_id = ct2d->gpu_cmd.params[0];
      CXLFunctionParamLayoutWire wire = {0};
      const HetGPUParamLayout *layout = NULL;
      uint64_t backend_queries_before = hetgpu->param_info_backend_queries;

      if (func_id >= ct2d->gpu_cmd.num_functions) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
        break;
      }
      if (!ct2d->gpu_cmd.functions[func_id].handle) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
        break;
      }
      if (ct2d->paired_case.qemu_cuda_calls_enabled) {
        qemu_log("CXL TYPE2 TRACE function_param_layout event=query_begin "
                 "layer=qemu function_id=%u backend_function=%p\n",
                 func_id, ct2d->gpu_cmd.functions[func_id].handle);
      }
      err = hetgpu_get_param_layout(
          hetgpu, ct2d->gpu_cmd.functions[func_id].handle, &layout);
      if (ct2d->paired_case.qemu_cuda_calls_enabled) {
        qemu_log("CXL TYPE2 TRACE function_param_layout event=query_end "
                 "layer=qemu function_id=%u backend_function=%p "
                 "backend_result=%d layout=%p backend_queries=%" PRIu64 "\n",
                 func_id, ct2d->gpu_cmd.functions[func_id].handle, err, layout,
                 hetgpu->param_info_backend_queries - backend_queries_before);
      }
      if (err != HETGPU_SUCCESS) {
        switch (err) {
        case HETGPU_ERROR_INVALID_HANDLE:
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
          break;
        case HETGPU_ERROR_NOT_SUPPORTED:
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_SUPPORTED;
          break;
        case HETGPU_ERROR_INVALID_CONTEXT:
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_CONTEXT;
          break;
        case HETGPU_ERROR_INVALID_VALUE:
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
          break;
        default:
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_UNKNOWN;
          break;
        }
        break;
      }
      if (!layout || layout->num_args > ARRAY_SIZE(wire.params) ||
          layout->extent > ct2d->gpu_cmd.data_size) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
        break;
      }
      wire.num_args = layout->num_args;
      wire.extent = layout->extent;
      for (uint32_t i = 0; i < wire.num_args; i++) {
        wire.params[i].offset = layout->offsets[i];
        wire.params[i].size = layout->sizes[i];
      }
      memcpy(ct2d->gpu_cmd.data, &wire, sizeof(wire));
      ct2d->gpu_cmd.results[0] =
          hetgpu->param_info_backend_queries - backend_queries_before;
    } else {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
    }
    break;

  case CXL_GPU_CMD_FUNC_SET_ATTRIBUTE:
    if (hetgpu->initialized) {
      uint32_t func_id = ct2d->gpu_cmd.params[0];

      if (func_id >= ct2d->gpu_cmd.num_functions) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
        break;
      }
      err = hetgpu_set_function_attribute(
          hetgpu, ct2d->gpu_cmd.functions[func_id].handle,
          (int)ct2d->gpu_cmd.params[1], (int)ct2d->gpu_cmd.params[2]);
      if (err == HETGPU_ERROR_INVALID_VALUE) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      } else if (err == HETGPU_ERROR_INVALID_CONTEXT) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_CONTEXT;
      } else if (err == HETGPU_ERROR_INVALID_HANDLE) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
      } else if (err == HETGPU_ERROR_NOT_SUPPORTED) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_SUPPORTED;
      } else if (err != HETGPU_SUCCESS) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_UNKNOWN;
      }
    } else {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
    }
    break;

  case CXL_GPU_CMD_FUNC_GET_ATTRIBUTE:
    ct2d->gpu_cmd.results[0] = 0;
    if (hetgpu->initialized) {
      uint32_t func_id = ct2d->gpu_cmd.params[0];
      int value = 0;

      if (func_id >= ct2d->gpu_cmd.num_functions ||
          !ct2d->gpu_cmd.functions[func_id].handle) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
        break;
      }
      err = hetgpu_get_function_attribute(
          hetgpu, ct2d->gpu_cmd.functions[func_id].handle,
          (int)ct2d->gpu_cmd.params[1], &value);
      if (err == HETGPU_SUCCESS) {
        ct2d->gpu_cmd.results[0] = (uint64_t)(int64_t)value;
      } else if (err == HETGPU_ERROR_INVALID_VALUE) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      } else if (err == HETGPU_ERROR_NOT_INITIALIZED) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
      } else if (err == HETGPU_ERROR_INVALID_CONTEXT) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_CONTEXT;
      } else if (err == HETGPU_ERROR_INVALID_HANDLE) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
      } else if (err == HETGPU_ERROR_NOT_SUPPORTED) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_SUPPORTED;
      } else {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_UNKNOWN;
      }
    } else {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
    }
    break;

  case CXL_GPU_CMD_FUNC_GET_OCCUPANCY:
    ct2d->gpu_cmd.results[0] = 0;
    if (hetgpu->initialized) {
      uint32_t func_id = ct2d->gpu_cmd.params[0];
      int num_blocks = 0;

      if (func_id >= ct2d->gpu_cmd.num_functions) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
        break;
      }
      err = hetgpu_get_max_active_blocks_per_multiprocessor(
          hetgpu, ct2d->gpu_cmd.functions[func_id].handle,
          (int)(int64_t)ct2d->gpu_cmd.params[1],
          (size_t)ct2d->gpu_cmd.params[2],
          (unsigned int)ct2d->gpu_cmd.params[3], &num_blocks);
      if (err == HETGPU_SUCCESS) {
        ct2d->gpu_cmd.results[0] = (uint64_t)(int64_t)num_blocks;
      } else if (err == HETGPU_ERROR_INVALID_VALUE) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      } else if (err == HETGPU_ERROR_NOT_INITIALIZED) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
      } else if (err == HETGPU_ERROR_INVALID_CONTEXT) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_CONTEXT;
      } else if (err == HETGPU_ERROR_INVALID_HANDLE) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
      } else if (err == HETGPU_ERROR_NOT_SUPPORTED) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_SUPPORTED;
      } else {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_UNKNOWN;
      }
    } else {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
    }
    break;

  case CXL_GPU_CMD_MODULE_GET_LOADING_MODE:
    memset(ct2d->gpu_cmd.results, 0, sizeof(ct2d->gpu_cmd.results));
    if (!hetgpu->initialized) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
      break;
    }
    {
      int mode = 0;
      int cuda_result = hetgpu_cuda_module_get_loading_mode(hetgpu, &mode);

      ct2d->gpu_cmd.cmd_result = cuda_result;
      if (cuda_result == CXL_GPU_SUCCESS) {
        ct2d->gpu_cmd.results[0] = (uint64_t)(uint32_t)mode;
      }
      qemu_log("CXL TYPE2 CUDA module_loading_mode "
               "driver_result=%d mode=%d\n",
               cuda_result, mode);
    }
    break;

  case CXL_GPU_CMD_GRAPH_EXEC_KERNEL_NODE_SET_PARAMS:
    if (!hetgpu->initialized) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
      break;
    }
    {
      uint64_t graph_exec_id_raw = ct2d->gpu_cmd.params[0];
      uint64_t graph_node_id_raw = ct2d->gpu_cmd.params[1];
      uint64_t func_id_raw = ct2d->gpu_cmd.params[2];
      HetGPULaunchConfig config = {
          .grid_dim =
              {
                  ct2d->gpu_cmd.params[3] & 0xFFFFFFFF,
                  (ct2d->gpu_cmd.params[3] >> 32) & 0xFFFFFFFF,
                  ct2d->gpu_cmd.params[4] & 0xFFFFFFFF,
              },
          .block_dim =
              {
                  (ct2d->gpu_cmd.params[4] >> 32) & 0xFFFFFFFF,
                  ct2d->gpu_cmd.params[5] & 0xFFFFFFFF,
                  (ct2d->gpu_cmd.params[5] >> 32) & 0xFFFFFFFF,
              },
          .shared_mem_bytes = ct2d->gpu_cmd.params[6] & 0xFFFFFFFF,
          .stream = NULL,
      };
      uint32_t num_args = ct2d->gpu_cmd.params[6] >> 32;
      size_t param_extent = ct2d->gpu_cmd.params[7];
      void *args[256];
      const HetGPUParamLayout *layout = NULL;
      GArray *identities = NULL;
      GArray *alias_bindings = NULL;
      const char *consumer_reason = NULL;
      int cuda_result;

      if (graph_exec_id_raw > UINT32_MAX || graph_node_id_raw > UINT32_MAX ||
          func_id_raw > UINT32_MAX ||
          graph_exec_id_raw >= ct2d->gpu_cmd.num_graph_execs ||
          graph_node_id_raw >= ct2d->gpu_cmd.num_graph_nodes ||
          func_id_raw >= ct2d->gpu_cmd.num_functions) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
        break;
      }
      uint32_t graph_exec_id = graph_exec_id_raw;
      uint32_t graph_node_id = graph_node_id_raw;
      uint32_t func_id = func_id_raw;
      if (!ct2d->gpu_cmd.graph_execs[graph_exec_id] ||
          !ct2d->gpu_cmd.graph_nodes[graph_node_id] ||
          !ct2d->gpu_cmd.functions[func_id].handle) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
        break;
      }
      err = hetgpu_get_param_layout(
          hetgpu, ct2d->gpu_cmd.functions[func_id].handle, &layout);
      if (err != HETGPU_SUCCESS || num_args != layout->num_args ||
          param_extent != layout->extent || num_args > ARRAY_SIZE(args) ||
          param_extent > ct2d->gpu_cmd.data_size) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
        break;
      }
      for (uint32_t i = 0; i < num_args; i++) {
        if (layout->offsets[i] > param_extent ||
            layout->sizes[i] > param_extent - layout->offsets[i]) {
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
          break;
        }
        args[i] = ct2d->gpu_cmd.data + layout->offsets[i];
      }
      if (ct2d->gpu_cmd.cmd_result != CXL_GPU_SUCCESS) {
        break;
      }
      if (!cxl_type2_model_consumer_validate(ct2d,
                                             &ct2d->gpu_cmd.functions[func_id],
                                             layout, args, &consumer_reason)) {
        cxl_type2_model_supply_fail(ct2d, ct2d->gpu_cmd.call_id,
                                    "driver-binding", consumer_reason,
                                    UINT64_MAX, 0, 0, 0);
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
        break;
      }
      identities =
          cxl_type2_generation_identities_from_params(ct2d, layout, args);
      if (!cxl_type2_model_alias_bind_graph_params(
              ct2d, &ct2d->gpu_cmd.functions[func_id], layout, args,
              &alias_bindings, &consumer_reason)) {
        cxl_type2_model_supply_fail(
            ct2d, ct2d->gpu_cmd.call_id, "driver-binding",
            consumer_reason ? consumer_reason : "graph-alias-bind-failed",
            UINT64_MAX, 0, 0, 0);
        g_array_unref(identities);
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
        break;
      }
      CXLType2GraphNodeGenerationConsumers event_node = {
          .node = ct2d->gpu_cmd.graph_nodes[graph_node_id],
          .function_id = func_id,
          .graph_node_token = graph_node_id,
      };

      if (alias_bindings->len) {
        cxl_type2_log_model_consumer_launch(
            ct2d, "bind", 0, func_id, layout, args, alias_bindings,
            CXL_TYPE2_CUDA_ALIAS_GRAPH, graph_exec_id, &graph_node_id_raw, 1);
      }
      cuda_result = hetgpu_cuda_graph_exec_kernel_node_set_params(
          hetgpu, ct2d->gpu_cmd.graph_execs[graph_exec_id],
          ct2d->gpu_cmd.graph_nodes[graph_node_id],
          ct2d->gpu_cmd.functions[func_id].handle, &config, args);
      ct2d->gpu_cmd.cmd_result = cuda_result;
      if (cuda_result == CXL_GPU_SUCCESS) {
        if (!cxl_type2_graph_generation_consumers_set_node(
                ct2d, graph_exec_id, ct2d->gpu_cmd.graph_nodes[graph_node_id],
                func_id, graph_node_id, identities, alias_bindings)) {
          g_array_unref(identities);
          cxl_type2_graph_alias_bindings_unbind(ct2d, graph_exec_id,
                                                &event_node, alias_bindings);
          cxl_type2_generation_reuse_fail(
              ct2d, "graph-node-consumer-identity-unavailable");
        }
      } else {
        g_array_unref(identities);
        cxl_type2_graph_alias_bindings_unbind(ct2d, graph_exec_id, &event_node,
                                              alias_bindings);
      }
    }
    break;

  case CXL_GPU_CMD_GRAPH_KERNEL_NODE_GET_PARAMS:
    if (!hetgpu->initialized) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
      break;
    }
    {
      uint64_t graph_node_id_raw = ct2d->gpu_cmd.params[0];
      CudaKernelNodeParams params = {0};
      CXLGraphKernelNodeParamsWire wire = {0};
      CXLGraphKernelNodeParamWire param_wires[64] = {0};
      const HetGPUParamLayout *layout = NULL;
      uint32_t function_id = UINT32_MAX;
      size_t param_extent = 0;
      int cuda_result;

      if (graph_node_id_raw > UINT32_MAX ||
          graph_node_id_raw >= ct2d->gpu_cmd.num_graph_nodes) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
        break;
      }
      uint32_t graph_node_id = graph_node_id_raw;
      if (!ct2d->gpu_cmd.graph_nodes[graph_node_id]) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
        break;
      }
      cuda_result = hetgpu_cuda_graph_kernel_node_get_params(
          hetgpu, ct2d->gpu_cmd.graph_nodes[graph_node_id], &params);
      if (cuda_result != CXL_GPU_SUCCESS) {
        ct2d->gpu_cmd.cmd_result = cuda_result;
        break;
      }
      if (!params.func || params.extra) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_SUPPORTED;
        break;
      }
      for (uint32_t i = 0; i < ct2d->gpu_cmd.num_functions; i++) {
        if (ct2d->gpu_cmd.functions[i].handle == params.func) {
          function_id = i;
          break;
        }
      }
      if (function_id == UINT32_MAX) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
        break;
      }
      err = hetgpu_get_param_layout(hetgpu, params.func, &layout);
      if (err != HETGPU_SUCCESS || layout->num_args > ARRAY_SIZE(param_wires)) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
        break;
      }
      wire.num_args = layout->num_args;
      param_extent = layout->extent;
      size_t param_wires_size = wire.num_args * sizeof(*param_wires);
      if (ct2d->gpu_cmd.cmd_result != CXL_GPU_SUCCESS ||
          sizeof(wire) > ct2d->gpu_cmd.data_size ||
          param_wires_size > ct2d->gpu_cmd.data_size - sizeof(wire) ||
          param_extent >
              ct2d->gpu_cmd.data_size - sizeof(wire) - param_wires_size) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
        break;
      }
      wire.function_id = function_id;
      wire.grid_dim_x = params.grid_dim_x;
      wire.grid_dim_y = params.grid_dim_y;
      wire.grid_dim_z = params.grid_dim_z;
      wire.block_dim_x = params.block_dim_x;
      wire.block_dim_y = params.block_dim_y;
      wire.block_dim_z = params.block_dim_z;
      wire.shared_mem_bytes = params.shared_mem_bytes;
      wire.param_extent = param_extent;
      memcpy(ct2d->gpu_cmd.data, &wire, sizeof(wire));
      for (uint32_t i = 0; i < wire.num_args; i++) {
        size_t param_offset = layout->offsets[i];
        size_t param_size = layout->sizes[i];

        if (!params.kernel_params || !params.kernel_params[i] ||
            param_offset > param_extent ||
            param_size > param_extent - param_offset) {
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
          break;
        }
        param_wires[i].offset = param_offset;
        param_wires[i].size = param_size;
        memcpy(ct2d->gpu_cmd.data + sizeof(wire) + param_wires_size +
                   param_offset,
               params.kernel_params[i], param_size);
      }
      if (ct2d->gpu_cmd.cmd_result == CXL_GPU_SUCCESS)
        memcpy(ct2d->gpu_cmd.data + sizeof(wire), param_wires,
               param_wires_size);
    }
    break;

  case CXL_GPU_CMD_GRAPH_EXEC_DESTROY:
    if (!hetgpu->initialized) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
      break;
    }
    {
      uint64_t graph_exec_id_raw = ct2d->gpu_cmd.params[0];

      if (graph_exec_id_raw > UINT32_MAX ||
          graph_exec_id_raw >= ct2d->gpu_cmd.num_graph_execs ||
          !ct2d->gpu_cmd.graph_execs[graph_exec_id_raw]) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
        break;
      }
      int cuda_result = hetgpu_cuda_graph_exec_destroy(
          hetgpu, ct2d->gpu_cmd.graph_execs[graph_exec_id_raw]);
      ct2d->gpu_cmd.cmd_result = cuda_result;
      if (cuda_result == CXL_GPU_SUCCESS) {
        ct2d->gpu_cmd.graph_execs[graph_exec_id_raw] = NULL;
        cxl_type2_graph_generation_consumers_set(ct2d, graph_exec_id_raw, NULL);
      }
    }
    break;

  case CXL_GPU_CMD_GRAPH_LAUNCH:
    if (!hetgpu->initialized) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
      break;
    }
    {
      uint64_t graph_exec_id_raw = ct2d->gpu_cmd.params[0];
      void *stream = NULL;
      GArray *alias_bindings = NULL;

      if (graph_exec_id_raw > UINT32_MAX ||
          graph_exec_id_raw >= ct2d->gpu_cmd.num_graph_execs ||
          !ct2d->gpu_cmd.graph_execs[graph_exec_id_raw] ||
          !cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[1], &stream)) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
        break;
      }
      alias_bindings =
          cxl_type2_graph_alias_launch_acquire(ct2d, graph_exec_id_raw);
      if (!alias_bindings) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
        break;
      }
      ct2d->gpu_cmd.cmd_result = hetgpu_cuda_graph_launch(
          hetgpu, ct2d->gpu_cmd.graph_execs[graph_exec_id_raw], stream);
      if (ct2d->gpu_cmd.cmd_result == CXL_GPU_SUCCESS) {
        uint64_t launch_id = ++ct2d->next_model_supply_launch_id;

        if (alias_bindings->len) {
          cxl_type2_log_model_graph_launch(ct2d, launch_id, graph_exec_id_raw,
                                           alias_bindings);
          cxl_type2_pending_alias_consumers_add(ct2d, stream, launch_id,
                                                CXL_TYPE2_CUDA_ALIAS_GRAPH,
                                                alias_bindings);
        } else {
          g_array_unref(alias_bindings);
        }
        cxl_type2_generation_consume_graph(ct2d, graph_exec_id_raw,
                                           ct2d->gpu_cmd.call_id);
      } else {
        if (alias_bindings->len) {
          const CXLType2AliasLaunchBinding *binding =
              &g_array_index(alias_bindings, CXLType2AliasLaunchBinding, 0);

          cxl_type2_model_supply_fail(
              ct2d, ct2d->gpu_cmd.call_id, "driver-launch",
              "graph-launch-failed", UINT64_MAX, binding->identity.base,
              binding->identity.epoch, binding->generation);
        }
        cxl_type2_alias_launch_bindings_release(ct2d, alias_bindings,
                                                CXL_TYPE2_CUDA_ALIAS_GRAPH);
      }
    }
    break;

  case CXL_GPU_CMD_GRAPH_DESTROY:
    if (!hetgpu->initialized) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
      break;
    }
    {
      uint64_t graph_id_raw = ct2d->gpu_cmd.params[0];

      if (graph_id_raw > UINT32_MAX ||
          graph_id_raw >= ct2d->gpu_cmd.num_graphs ||
          !ct2d->gpu_cmd.graphs[graph_id_raw]) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
        break;
      }
      int cuda_result =
          hetgpu_cuda_graph_destroy(hetgpu, ct2d->gpu_cmd.graphs[graph_id_raw]);
      ct2d->gpu_cmd.cmd_result = cuda_result;
      if (cuda_result == CXL_GPU_SUCCESS) {
        ct2d->gpu_cmd.graphs[graph_id_raw] = NULL;
      }
    }
    break;

  case CXL_GPU_CMD_GRAPH_INSTANTIATE:
    memset(ct2d->gpu_cmd.results, 0xff, sizeof(ct2d->gpu_cmd.results));
    if (!hetgpu->initialized) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
      break;
    }
    {
      uint64_t graph_id_raw = ct2d->gpu_cmd.params[0];
      size_t buffer_size = ct2d->gpu_cmd.params[1];
      bool want_error_node = ct2d->gpu_cmd.params[2];
      bool want_log = ct2d->gpu_cmd.params[3];
      g_autofree char *log_buffer = NULL;
      char log_dummy = 0;
      HetGPUGraphExec graph_exec = NULL;
      HetGPUGraphNode error_node = NULL;
      CXLType2GraphExecGenerationConsumers *consumers = NULL;
      uint32_t graph_exec_id = ct2d->gpu_cmd.num_graph_execs;

      if (graph_id_raw > UINT32_MAX ||
          graph_id_raw >= ct2d->gpu_cmd.num_graphs ||
          !ct2d->gpu_cmd.graphs[graph_id_raw]) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
        break;
      }
      if (ct2d->gpu_cmd.params[2] > 1 || ct2d->gpu_cmd.params[3] > 1 ||
          (!want_log && buffer_size) || buffer_size > ct2d->gpu_cmd.data_size) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
        break;
      }
      if (!cxl_type2_reserve_gpu_handle(&ct2d->gpu_cmd.graph_execs,
                                        &ct2d->gpu_cmd.graph_execs_capacity,
                                        ct2d->gpu_cmd.num_graph_execs)) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_OUT_OF_MEMORY;
        break;
      }
      if (want_log && buffer_size) {
        log_buffer = g_try_malloc0(buffer_size);
        if (!log_buffer) {
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_OUT_OF_MEMORY;
          break;
        }
      }
      if (ct2d->model_supply.route == CXL_TYPE2_MODEL_SUPPLY_CXL_DIRECT) {
        consumers = cxl_type2_graph_generation_consumers_snapshot(
            ct2d, ct2d->gpu_cmd.graphs[graph_id_raw], graph_exec_id);
        if (!consumers) {
          cxl_type2_model_supply_fail(
              ct2d, ct2d->gpu_cmd.call_id, "driver-binding",
              "graph-consumer-snapshot-unavailable", UINT64_MAX, 0, 0, 0);
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
          break;
        }
      }
      int cuda_result = hetgpu_cuda_graph_instantiate(
          hetgpu, ct2d->gpu_cmd.graphs[graph_id_raw], &graph_exec,
          want_error_node ? &error_node : NULL,
          want_log ? (buffer_size ? log_buffer : &log_dummy) : NULL,
          buffer_size);
      ct2d->gpu_cmd.cmd_result = cuda_result;
      if (want_log && buffer_size) {
        memcpy(ct2d->gpu_cmd.data, log_buffer, buffer_size);
      }
      if (error_node) {
        uint32_t error_node_id;

        if (cxl_type2_register_gpu_handle(
                &ct2d->gpu_cmd.graph_nodes, &ct2d->gpu_cmd.graph_nodes_capacity,
                &ct2d->gpu_cmd.num_graph_nodes, error_node, &error_node_id)) {
          ct2d->gpu_cmd.results[1] = error_node_id;
        } else {
          qemu_log(
              "CXL Type2: failed to register graph instantiate error node\n");
        }
      }
      if (cuda_result == CXL_GPU_SUCCESS) {
        if (!graph_exec) {
          cxl_type2_graph_exec_generation_consumers_free(consumers);
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
          break;
        }
        ct2d->gpu_cmd.graph_execs[graph_exec_id] = graph_exec;
        ct2d->gpu_cmd.results[0] = graph_exec_id;
        ct2d->gpu_cmd.num_graph_execs++;
        if (!consumers) {
          consumers = cxl_type2_graph_generation_consumers_snapshot(
              ct2d, ct2d->gpu_cmd.graphs[graph_id_raw], graph_exec_id);
        }

        if (consumers) {
          cxl_type2_graph_generation_consumers_set(ct2d, graph_exec_id,
                                                   consumers);
        } else {
          cxl_type2_generation_reuse_fail(
              ct2d, "graph-consumer-snapshot-unavailable");
        }
        if (ct2d->paired_case.active_case != CXL_GPU_CASE_NONE) {
          cxl_type2_cuda_classify_graph(ct2d,
                                        ct2d->gpu_cmd.graphs[graph_id_raw]);
        }
      } else {
        cxl_type2_graph_exec_generation_consumers_free(consumers);
      }
    }
    break;

  case CXL_GPU_CMD_GRAPH_GET_NODES:
    memset(ct2d->gpu_cmd.results, 0, sizeof(ct2d->gpu_cmd.results));
    if (!hetgpu->initialized) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
      break;
    }
    {
      uint64_t graph_id_raw = ct2d->gpu_cmd.params[0];
      size_t requested = ct2d->gpu_cmd.params[1];
      bool want_nodes = ct2d->gpu_cmd.params[2];
      g_autofree HetGPUGraphNode *nodes = NULL;
      g_autofree uint64_t *node_ids = NULL;

      if (graph_id_raw > UINT32_MAX ||
          graph_id_raw >= ct2d->gpu_cmd.num_graphs ||
          !ct2d->gpu_cmd.graphs[graph_id_raw]) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
        break;
      }
      if (ct2d->gpu_cmd.params[2] > 1 || (!want_nodes && requested) ||
          requested > ct2d->gpu_cmd.data_size / sizeof(uint64_t)) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
        break;
      }
      if (want_nodes) {
        nodes = g_try_new0(HetGPUGraphNode, MAX(requested, 1));
        if (!nodes) {
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_OUT_OF_MEMORY;
          break;
        }
      }
      size_t count = requested;
      int cuda_result = hetgpu_cuda_graph_get_nodes(
          hetgpu, ct2d->gpu_cmd.graphs[graph_id_raw], want_nodes ? nodes : NULL,
          &count);
      ct2d->gpu_cmd.cmd_result = cuda_result;
      if (cuda_result != CXL_GPU_SUCCESS) {
        break;
      }
      if (want_nodes && count > requested) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
        break;
      }
      ct2d->gpu_cmd.results[0] = count;
      if (!want_nodes || count == 0) {
        break;
      }
      node_ids = g_try_new(uint64_t, count);
      if (!node_ids) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_OUT_OF_MEMORY;
        break;
      }
      for (size_t i = 0; i < count; i++) {
        uint32_t node_id;

        if (!cxl_type2_register_gpu_handle(
                &ct2d->gpu_cmd.graph_nodes, &ct2d->gpu_cmd.graph_nodes_capacity,
                &ct2d->gpu_cmd.num_graph_nodes, nodes[i], &node_id)) {
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_OUT_OF_MEMORY;
          break;
        }
        node_ids[i] = node_id;
      }
      if (ct2d->gpu_cmd.cmd_result == CXL_GPU_SUCCESS) {
        memcpy(ct2d->gpu_cmd.data, node_ids, count * sizeof(*node_ids));
      }
    }
    break;

  case CXL_GPU_CMD_GRAPH_EXEC_UPDATE:
    memset(ct2d->gpu_cmd.results, 0xff, sizeof(ct2d->gpu_cmd.results));
    if (!hetgpu->initialized) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
      break;
    }
    {
      uint64_t graph_exec_id = ct2d->gpu_cmd.params[0];
      uint64_t graph_id = ct2d->gpu_cmd.params[1];
      HetGPUGraphExecUpdateResultInfo info = {0};
      CXLType2GraphExecGenerationConsumers *consumers = NULL;

      if (graph_exec_id > UINT32_MAX || graph_id > UINT32_MAX ||
          graph_exec_id >= ct2d->gpu_cmd.num_graph_execs ||
          graph_id >= ct2d->gpu_cmd.num_graphs ||
          !ct2d->gpu_cmd.graph_execs[graph_exec_id] ||
          !ct2d->gpu_cmd.graphs[graph_id]) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
        break;
      }
      if (ct2d->model_supply.route == CXL_TYPE2_MODEL_SUPPLY_CXL_DIRECT) {
        consumers = cxl_type2_graph_generation_consumers_snapshot(
            ct2d, ct2d->gpu_cmd.graphs[graph_id], graph_exec_id);
        if (!consumers) {
          cxl_type2_model_supply_fail(
              ct2d, ct2d->gpu_cmd.call_id, "driver-binding",
              "graph-update-consumer-snapshot-unavailable", UINT64_MAX, 0, 0,
              0);
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
          break;
        }
      }
      int cuda_result = hetgpu_cuda_graph_exec_update(
          hetgpu, ct2d->gpu_cmd.graph_execs[graph_exec_id],
          ct2d->gpu_cmd.graphs[graph_id], &info);
      ct2d->gpu_cmd.cmd_result = cuda_result;
      if (cuda_result == CXL_GPU_SUCCESS) {
        if (!consumers) {
          consumers = cxl_type2_graph_generation_consumers_snapshot(
              ct2d, ct2d->gpu_cmd.graphs[graph_id], graph_exec_id);
        }
        if (consumers) {
          cxl_type2_graph_generation_consumers_set(ct2d, graph_exec_id,
                                                   consumers);
        } else {
          cxl_type2_generation_reuse_fail(
              ct2d, "graph-update-consumer-snapshot-unavailable");
        }
      } else {
        cxl_type2_graph_exec_generation_consumers_free(consumers);
      }
      ct2d->gpu_cmd.results[0] = (uint64_t)(int64_t)info.result;
      HetGPUGraphNode nodes[] = {info.error_node, info.error_from_node};
      for (size_t i = 0; i < ARRAY_SIZE(nodes); i++) {
        uint32_t node_id;

        if (nodes[i] &&
            cxl_type2_register_gpu_handle(
                &ct2d->gpu_cmd.graph_nodes, &ct2d->gpu_cmd.graph_nodes_capacity,
                &ct2d->gpu_cmd.num_graph_nodes, nodes[i], &node_id)) {
          ct2d->gpu_cmd.results[i + 1] = node_id;
        } else if (nodes[i]) {
          qemu_log("CXL Type2: failed to register graph update error node\n");
        }
      }
    }
    break;

  case CXL_GPU_CMD_GRAPH_NODE_GET_TYPE:
    memset(ct2d->gpu_cmd.results, 0, sizeof(ct2d->gpu_cmd.results));
    if (!hetgpu->initialized) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
      break;
    }
    {
      uint64_t graph_node_id_raw = ct2d->gpu_cmd.params[0];
      int node_type = 0;

      if (graph_node_id_raw > UINT32_MAX ||
          graph_node_id_raw >= ct2d->gpu_cmd.num_graph_nodes ||
          !ct2d->gpu_cmd.graph_nodes[graph_node_id_raw]) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
        break;
      }
      int cuda_result = hetgpu_cuda_graph_node_get_type(
          hetgpu, ct2d->gpu_cmd.graph_nodes[graph_node_id_raw], &node_type);
      ct2d->gpu_cmd.cmd_result = cuda_result;
      if (cuda_result == CXL_GPU_SUCCESS) {
        ct2d->gpu_cmd.results[0] = (uint64_t)(int64_t)node_type;
      }
    }
    break;

  case CXL_GPU_CMD_LINK_CREATE:
    memset(ct2d->gpu_cmd.results, 0xff, sizeof(ct2d->gpu_cmd.results));
    {
      void *link_state = NULL;
      uint32_t id;
      int result = hetgpu_cuda_link_create(hetgpu, &link_state);

      ct2d->gpu_cmd.cmd_result = result;
      if (result == CXL_GPU_SUCCESS) {
        if (!cxl_type2_register_gpu_handle(
                &ct2d->gpu_cmd.link_states, &ct2d->gpu_cmd.link_states_capacity,
                &ct2d->gpu_cmd.num_link_states, link_state, &id)) {
          (void)hetgpu_cuda_link_destroy(hetgpu, link_state);
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_OUT_OF_MEMORY;
        } else {
          ct2d->gpu_cmd.results[0] = id;
        }
      }
    }
    break;

  case CXL_GPU_CMD_LINK_ADD_DATA: {
    uint64_t id = ct2d->gpu_cmd.params[0];
    size_t input_size = ct2d->gpu_cmd.params[2];
    size_t name_size = ct2d->gpu_cmd.params[3];

    if (id > UINT32_MAX || id >= ct2d->gpu_cmd.num_link_states ||
        !ct2d->gpu_cmd.link_states[id]) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
      break;
    }
    if (!input_size || input_size > ct2d->gpu_cmd.data_size ||
        name_size > ct2d->gpu_cmd.data_size - input_size ||
        (name_size && ct2d->gpu_cmd.data[input_size + name_size - 1] != 0)) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      break;
    }
    ct2d->gpu_cmd.cmd_result = hetgpu_cuda_link_add_data(
        hetgpu, ct2d->gpu_cmd.link_states[id], (int)ct2d->gpu_cmd.params[1],
        ct2d->gpu_cmd.data, input_size,
        name_size ? (char *)ct2d->gpu_cmd.data + input_size : NULL);
  } break;

  case CXL_GPU_CMD_LINK_COMPLETE:
    memset(ct2d->gpu_cmd.results, 0, sizeof(ct2d->gpu_cmd.results));
    {
      uint64_t id = ct2d->gpu_cmd.params[0];
      void *cubin = NULL;
      size_t cubin_size = 0;

      if (id > UINT32_MAX || id >= ct2d->gpu_cmd.num_link_states ||
          !ct2d->gpu_cmd.link_states[id]) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
        break;
      }
      int result = hetgpu_cuda_link_complete(
          hetgpu, ct2d->gpu_cmd.link_states[id], &cubin, &cubin_size);
      ct2d->gpu_cmd.cmd_result = result;
      if (result == CXL_GPU_SUCCESS) {
        if (!cubin || cubin_size > ct2d->gpu_cmd.data_size) {
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_SUPPORTED;
          break;
        }
        memcpy(ct2d->gpu_cmd.data, cubin, cubin_size);
        ct2d->gpu_cmd.results[0] = cubin_size;
      }
    }
    break;

  case CXL_GPU_CMD_LINK_DESTROY: {
    uint64_t id = ct2d->gpu_cmd.params[0];

    if (id > UINT32_MAX || id >= ct2d->gpu_cmd.num_link_states ||
        !ct2d->gpu_cmd.link_states[id]) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
      break;
    }
    int result =
        hetgpu_cuda_link_destroy(hetgpu, ct2d->gpu_cmd.link_states[id]);
    ct2d->gpu_cmd.cmd_result = result;
    if (result == CXL_GPU_SUCCESS)
      ct2d->gpu_cmd.link_states[id] = NULL;
  } break;

  case CXL_GPU_CMD_CTX_GET_LIMIT: {
    size_t value = 0;
    int result =
        hetgpu_cuda_ctx_get_limit(hetgpu, &value, (int)ct2d->gpu_cmd.params[0]);
    ct2d->gpu_cmd.cmd_result = result;
    if (result == CXL_GPU_SUCCESS)
      ct2d->gpu_cmd.results[0] = value;
  } break;

  case CXL_GPU_CMD_DEVICE_CAN_ACCESS_PEER: {
    int can_access = 0;
    int result = hetgpu_cuda_device_can_access_peer(
        hetgpu, &can_access, (int)ct2d->gpu_cmd.params[0],
        (int)ct2d->gpu_cmd.params[1]);
    ct2d->gpu_cmd.cmd_result = result;
    if (result == CXL_GPU_SUCCESS)
      ct2d->gpu_cmd.results[0] = can_access;
  } break;

  case CXL_GPU_CMD_CTX_ENABLE_PEER:
    ct2d->gpu_cmd.cmd_result = hetgpu_cuda_ctx_enable_peer_access(
        hetgpu, hetgpu->context, (unsigned int)ct2d->gpu_cmd.params[1]);
    break;

  case CXL_GPU_CMD_CTX_DISABLE_PEER:
    ct2d->gpu_cmd.cmd_result =
        hetgpu_cuda_ctx_disable_peer_access(hetgpu, hetgpu->context);
    break;

  case CXL_GPU_CMD_MEM_PREFETCH_ASYNC: {
    void *stream = NULL;
    int64_t enqueue_begin_ns;
    int64_t enqueue_end_ns;
    if (!cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[3], &stream)) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
      break;
    }
    enqueue_begin_ns = cxl_type2_host_monotonic_ns();
    ct2d->gpu_cmd.cmd_result = hetgpu_cuda_mem_prefetch_async(
        hetgpu, ct2d->gpu_cmd.params[0], ct2d->gpu_cmd.params[1],
        (int)ct2d->gpu_cmd.params[2], stream);
    enqueue_end_ns = cxl_type2_host_monotonic_ns();
    if (ct2d->gpu_cmd.cmd_result == CXL_GPU_SUCCESS) {
      CXLType2CudaAllocationIdentity identity;

      if (cxl_type2_cuda_allocation_identity_for_address(
              &ct2d->cuda_allocations, ct2d->gpu_cmd.params[0], &identity) &&
          !cxl_type2_cuda_generation_prefetch_enqueue(
              &ct2d->cuda_allocations, ct2d->gpu_cmd.params[0],
              ct2d->gpu_cmd.params[1],
              enqueue_end_ns >= enqueue_begin_ns
                  ? (uint64_t)(enqueue_end_ns - enqueue_begin_ns)
                  : 0)) {
        cxl_type2_generation_reuse_fail(
            ct2d, ct2d->cuda_allocations.first_generation_error);
      }
    }
  } break;

  case CXL_GPU_CMD_LAUNCH_KERNEL:
    if (hetgpu->initialized) {
      uint32_t func_id = ct2d->gpu_cmd.params[0];
      if (func_id < ct2d->gpu_cmd.num_functions) {
        HetGPULaunchConfig config;
        config.grid_dim[0] = ct2d->gpu_cmd.params[1] & 0xFFFFFFFF;
        config.grid_dim[1] = (ct2d->gpu_cmd.params[1] >> 32) & 0xFFFFFFFF;
        config.grid_dim[2] = ct2d->gpu_cmd.params[2] & 0xFFFFFFFF;
        config.block_dim[0] = (ct2d->gpu_cmd.params[2] >> 32) & 0xFFFFFFFF;
        config.block_dim[1] = ct2d->gpu_cmd.params[3] & 0xFFFFFFFF;
        config.block_dim[2] = (ct2d->gpu_cmd.params[3] >> 32) & 0xFFFFFFFF;
        config.shared_mem_bytes = ct2d->gpu_cmd.params[4] & 0xFFFFFFFF;
        if (!cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[6],
                                        &config.stream)) {
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
          break;
        }

        uint32_t num_args = (ct2d->gpu_cmd.params[4] >> 32) & 0xFF;
        size_t param_extent = ct2d->gpu_cmd.params[5];
        void *args[256];
        CXLType2CudaAllocationIdentity consumer_identities[256];
        size_t consumer_identity_count = 0;
        GArray *alias_bindings = NULL;
        const HetGPUParamLayout *layout = NULL;
        const char *consumer_reason = NULL;
        uint64_t cache_hits_before = hetgpu->param_info_cache_hits;
        uint64_t cache_misses_before = hetgpu->param_info_cache_misses;
        uint64_t backend_queries_before = hetgpu->param_info_backend_queries;
        bool capturing =
            cxl_type2_stream_wire_is_capturing(ct2d, ct2d->gpu_cmd.params[6]);

        err = hetgpu_get_param_layout(
            hetgpu, ct2d->gpu_cmd.functions[func_id].handle, &layout);
        if (err != HETGPU_SUCCESS || num_args != layout->num_args ||
            param_extent != layout->extent || num_args > ARRAY_SIZE(args) ||
            param_extent > ct2d->gpu_cmd.data_size) {
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
          break;
        }
        for (uint32_t i = 0; i < num_args; i++) {
          if (layout->offsets[i] > param_extent ||
              layout->sizes[i] > param_extent - layout->offsets[i]) {
            ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
            break;
          }
          args[i] = ct2d->gpu_cmd.data + layout->offsets[i];
          if (layout->sizes[i] == sizeof(uint64_t)) {
            uint64_t pointer;
            CXLType2CudaAllocationIdentity identity;

            memcpy(&pointer, args[i], sizeof(pointer));
            if (cxl_type2_cuda_allocation_identity_for_address(
                    &ct2d->cuda_allocations, pointer, &identity)) {
              consumer_identities[consumer_identity_count++] = identity;
            }
          }
        }
        if (ct2d->gpu_cmd.cmd_result != CXL_GPU_SUCCESS) {
          break;
        }
        if (!cxl_type2_model_consumer_validate(
                ct2d, &ct2d->gpu_cmd.functions[func_id], layout, args,
                &consumer_reason)) {
          cxl_type2_model_supply_fail(ct2d, ct2d->gpu_cmd.call_id,
                                      "driver-binding", consumer_reason,
                                      UINT64_MAX, 0, 0, 0);
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
          break;
        }
        alias_bindings = cxl_type2_model_alias_rewrite_params(
            ct2d, &ct2d->gpu_cmd.functions[func_id], layout, args,
            CXL_TYPE2_CUDA_ALIAS_NORMAL, &consumer_reason);
        if (!alias_bindings) {
          cxl_type2_model_supply_fail(
              ct2d, ct2d->gpu_cmd.call_id, "driver-binding",
              consumer_reason ? consumer_reason : "normal-alias-bind-failed",
              UINT64_MAX, 0, 0, 0);
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
          break;
        }

        if (ct2d->paired_case.qemu_cuda_calls_enabled)
          qemu_log("CXL TYPE2 TRACE function_param_layout event=launch "
                   "layer=qemu function_id=%u args=%u extent=%zu hits=%" PRIu64
                   " misses=%" PRIu64 " backend_queries=%" PRIu64 "\n",
                   func_id, num_args, param_extent,
                   hetgpu->param_info_cache_hits - cache_hits_before,
                   hetgpu->param_info_cache_misses - cache_misses_before,
                   hetgpu->param_info_backend_queries - backend_queries_before);

        err = hetgpu_launch_kernel(hetgpu,
                                   ct2d->gpu_cmd.functions[func_id].handle,
                                   &config, args, num_args);
        if (err != HETGPU_SUCCESS) {
          if (alias_bindings->len) {
            const CXLType2AliasLaunchBinding *binding =
                &g_array_index(alias_bindings, CXLType2AliasLaunchBinding, 0);

            cxl_type2_model_supply_fail(
                ct2d, ct2d->gpu_cmd.call_id, "driver-launch",
                "kernel-launch-failed", UINT64_MAX, binding->identity.base,
                binding->identity.epoch, binding->generation);
          }
          cxl_type2_alias_launch_bindings_release(ct2d, alias_bindings,
                                                  CXL_TYPE2_CUDA_ALIAS_NORMAL);
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_LAUNCH_FAILED;
        } else {
          uint64_t launch_id = ++ct2d->next_model_supply_launch_id;

          if (capturing) {
            cxl_type2_alias_launch_bindings_release(
                ct2d, alias_bindings, CXL_TYPE2_CUDA_ALIAS_NORMAL);
          } else if (alias_bindings->len) {
            cxl_type2_log_model_consumer_launch(
                ct2d, "launch", launch_id, func_id, layout, args,
                alias_bindings, CXL_TYPE2_CUDA_ALIAS_NORMAL, 0, NULL, 0);
            cxl_type2_pending_alias_consumers_add(
                ct2d, config.stream, launch_id, CXL_TYPE2_CUDA_ALIAS_NORMAL,
                alias_bindings);
          } else {
            g_array_unref(alias_bindings);
          }
          if (!capturing &&
              !cxl_type2_cuda_generation_consume(
                  &ct2d->cuda_allocations, consumer_identities,
                  consumer_identity_count, CXL_GPU_CMD_LAUNCH_KERNEL,
                  ct2d->gpu_cmd.call_id)) {
            cxl_type2_generation_reuse_fail(
                ct2d, ct2d->cuda_allocations.first_generation_error);
          }
        }
      } else {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
      }
    } else {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
    }
    break;

  case CXL_GPU_CMD_STREAM_CREATE:
    memset(ct2d->gpu_cmd.results, 0xff, sizeof(ct2d->gpu_cmd.results));
    {
      void *stream = NULL;
      uint32_t id;
      int result = hetgpu_cuda_stream_create(
          hetgpu, (unsigned int)ct2d->gpu_cmd.params[0], &stream);
      ct2d->gpu_cmd.cmd_result = result;
      if (result == CXL_GPU_SUCCESS) {
        if (!cxl_type2_register_gpu_handle(
                &ct2d->gpu_cmd.streams, &ct2d->gpu_cmd.streams_capacity,
                &ct2d->gpu_cmd.num_streams, stream, &id)) {
          (void)hetgpu_cuda_stream_destroy(hetgpu, stream);
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_OUT_OF_MEMORY;
        } else {
          ct2d->gpu_cmd.results[0] = id;
        }
      }
    }
    break;

  case CXL_GPU_CMD_STREAM_DESTROY: {
    uint64_t id = ct2d->gpu_cmd.params[0];
    if (id > UINT32_MAX || id >= ct2d->gpu_cmd.num_streams ||
        !ct2d->gpu_cmd.streams[id]) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
      break;
    }
    void *stream = ct2d->gpu_cmd.streams[id];
    int64_t wait_begin_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
    uint64_t model_htod_completed = 0;
    uint64_t consumer_launches_completed = 0;
    int result = hetgpu_cuda_stream_synchronize(hetgpu, stream);
    int64_t wait_end_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
    if (result == CXL_GPU_SUCCESS) {
      result = cxl_type2_release_pending_htod(ct2d, stream, false, UINT64_MAX,
                                              "stream-destroy",
                                              &model_htod_completed);
    }
    if (result == CXL_GPU_SUCCESS) {
      result = cxl_type2_release_pending_alias_consumers(
          ct2d, stream, false, UINT64_MAX, &consumer_launches_completed);
    }
    cxl_type2_record_model_supply_completion(ct2d, wait_begin_ns, wait_end_ns,
                                             true, model_htod_completed,
                                             consumer_launches_completed);
    if (result == CXL_GPU_SUCCESS) {
      result = hetgpu_cuda_stream_destroy(hetgpu, stream);
    }
    ct2d->gpu_cmd.cmd_result = result;
    if (result == CXL_GPU_SUCCESS) {
      ct2d->gpu_cmd.streams[id] = NULL;
    }
  } break;

  case CXL_GPU_CMD_STREAM_SYNC: {
    if (!stream_sync_reason_valid) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      break;
    }
    if (elide_stream_sync) {
      ct2d->paired_case.active_elided_stream_syncs++;
      stream_sync_elided = true;
      ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
      break;
    }
    void *stream = NULL;
    if (!cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[0], &stream)) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
      break;
    }
    ct2d->paired_case.active_stream_sync_driver_calls++;
    int64_t wait_begin_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
    uint64_t model_htod_completed = 0;
    uint64_t consumer_launches_completed = 0;
    ct2d->gpu_cmd.cmd_result = hetgpu_cuda_stream_synchronize(hetgpu, stream);
    int64_t wait_end_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
    if (ct2d->gpu_cmd.cmd_result == CXL_GPU_SUCCESS) {
      int release_result =
          cxl_type2_release_pending_htod(ct2d, stream, false, UINT64_MAX,
                                         "stream-sync", &model_htod_completed);
      if (release_result != CXL_GPU_SUCCESS) {
        ct2d->gpu_cmd.cmd_result = release_result;
      } else {
        release_result = cxl_type2_release_pending_alias_consumers(
            ct2d, stream, false, UINT64_MAX, &consumer_launches_completed);
        if (release_result != CXL_GPU_SUCCESS) {
          ct2d->gpu_cmd.cmd_result = release_result;
        }
      }
    }
    cxl_type2_record_model_supply_completion(ct2d, wait_begin_ns, wait_end_ns,
                                             true, model_htod_completed,
                                             consumer_launches_completed);
  } break;

  case CXL_GPU_CMD_STREAM_WAIT_EVENT: {
    void *stream = NULL;
    uint64_t event_id = ct2d->gpu_cmd.params[1];
    if (!cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[0], &stream) ||
        event_id > UINT32_MAX || event_id >= ct2d->gpu_cmd.num_events ||
        !ct2d->gpu_cmd.events[event_id]) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
      break;
    }
    ct2d->gpu_cmd.cmd_result = hetgpu_cuda_stream_wait_event(
        hetgpu, stream, ct2d->gpu_cmd.events[event_id],
        (unsigned int)ct2d->gpu_cmd.params[2]);
  } break;

  case CXL_GPU_CMD_STREAM_WAIT_VALUE32: {
    void *stream = NULL;
    if (!cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[0], &stream)) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
      break;
    }
    ct2d->gpu_cmd.cmd_result =
        hetgpu_cuda_stream_wait_value32(hetgpu, stream, ct2d->gpu_cmd.params[1],
                                        (uint32_t)ct2d->gpu_cmd.params[2],
                                        (unsigned int)ct2d->gpu_cmd.params[3]);
  } break;

  case CXL_GPU_CMD_STREAM_BATCH_MEM_OP: {
    void *stream = NULL;
    unsigned int count = ct2d->gpu_cmd.params[1];
    if (!cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[0], &stream)) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
      break;
    }
    if (count >= 256 || count > ct2d->gpu_cmd.data_size / 48) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      break;
    }
    ct2d->gpu_cmd.cmd_result = hetgpu_cuda_stream_batch_mem_op(
        hetgpu, stream, count, count ? ct2d->gpu_cmd.data : NULL,
        (unsigned int)ct2d->gpu_cmd.params[2]);
  } break;

  case CXL_GPU_CMD_STREAM_GET_CTX: {
    void *stream = NULL;
    void *context = NULL;
    if (!cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[0], &stream)) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
      break;
    }
    int result = hetgpu_cuda_stream_get_ctx(hetgpu, stream, &context);
    ct2d->gpu_cmd.cmd_result = result;
    if (result == CXL_GPU_SUCCESS) {
      if (!context || context != hetgpu->context)
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_CONTEXT;
      else
        ct2d->gpu_cmd.results[0] = 1;
    }
  } break;

  case CXL_GPU_CMD_STREAM_BEGIN_CAPTURE: {
    void *stream = NULL;
    if (!cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[0], &stream)) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
      break;
    }
    ct2d->gpu_cmd.cmd_result = hetgpu_cuda_stream_begin_capture(
        hetgpu, stream, (int)ct2d->gpu_cmd.params[1]);
    if (ct2d->gpu_cmd.cmd_result == CXL_GPU_SUCCESS) {
      cxl_type2_stream_wire_capture_set(ct2d, ct2d->gpu_cmd.params[0], true);
    }
  } break;

  case CXL_GPU_CMD_STREAM_END_CAPTURE:
    memset(ct2d->gpu_cmd.results, 0xff, sizeof(ct2d->gpu_cmd.results));
    {
      void *stream = NULL;
      void *graph = NULL;
      uint32_t graph_id;
      if (!cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[0], &stream)) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
        break;
      }
      int result = hetgpu_cuda_stream_end_capture(hetgpu, stream, &graph);
      ct2d->gpu_cmd.cmd_result = result;
      cxl_type2_stream_wire_capture_set(ct2d, ct2d->gpu_cmd.params[0], false);
      if (result == CXL_GPU_SUCCESS && graph) {
        if (!cxl_type2_register_gpu_handle(
                &ct2d->gpu_cmd.graphs, &ct2d->gpu_cmd.graphs_capacity,
                &ct2d->gpu_cmd.num_graphs, graph, &graph_id)) {
          (void)hetgpu_cuda_graph_destroy(hetgpu, graph);
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_OUT_OF_MEMORY;
        } else {
          ct2d->gpu_cmd.results[0] = graph_id;
        }
      }
    }
    break;

  case CXL_GPU_CMD_STREAM_IS_CAPTURING: {
    void *stream = NULL;
    int status = 0;
    if (!cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[0], &stream)) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
      break;
    }
    int result = hetgpu_cuda_stream_is_capturing(hetgpu, stream, &status);
    ct2d->gpu_cmd.cmd_result = result;
    if (result == CXL_GPU_SUCCESS)
      ct2d->gpu_cmd.results[0] = status;
  } break;

  case CXL_GPU_CMD_STREAM_GET_CAPTURE_INFO:
    memset(ct2d->gpu_cmd.results, 0xff, sizeof(ct2d->gpu_cmd.results));
    {
      void *stream = NULL;
      void *graph = NULL;
      void **dependencies = NULL;
      size_t count = 0;
      int status = 0;
      uint64_t capture_id = 0;
      if (!cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[0], &stream)) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
        break;
      }
      int result = hetgpu_cuda_stream_get_capture_info(
          hetgpu, stream, &status, &capture_id, &graph, &dependencies, &count);
      ct2d->gpu_cmd.cmd_result = result;
      if (result != CXL_GPU_SUCCESS)
        break;
      if (count > ct2d->gpu_cmd.data_size / sizeof(uint64_t)) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_SUPPORTED;
        break;
      }
      ct2d->gpu_cmd.results[0] = (uint64_t)(uint32_t)status;
      ct2d->gpu_cmd.results[1] = capture_id;
      ct2d->gpu_cmd.results[2] = count;
      if (graph) {
        uint32_t graph_id;
        if (!cxl_type2_register_gpu_handle(
                &ct2d->gpu_cmd.graphs, &ct2d->gpu_cmd.graphs_capacity,
                &ct2d->gpu_cmd.num_graphs, graph, &graph_id)) {
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_OUT_OF_MEMORY;
          break;
        }
        ct2d->gpu_cmd.results[3] = graph_id;
      }
      uint64_t *ids = (uint64_t *)ct2d->gpu_cmd.data;
      for (size_t i = 0; i < count; i++) {
        uint32_t node_id;
        if (!dependencies ||
            !cxl_type2_register_gpu_handle(
                &ct2d->gpu_cmd.graph_nodes, &ct2d->gpu_cmd.graph_nodes_capacity,
                &ct2d->gpu_cmd.num_graph_nodes, dependencies[i], &node_id)) {
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_OUT_OF_MEMORY;
          break;
        }
        ids[i] = node_id;
      }
    }
    break;

  case CXL_GPU_CMD_EVENT_CREATE:
    memset(ct2d->gpu_cmd.results, 0xff, sizeof(ct2d->gpu_cmd.results));
    {
      void *event = NULL;
      uint32_t id;
      int result = hetgpu_cuda_event_create(
          hetgpu, (unsigned int)ct2d->gpu_cmd.params[0], &event);
      ct2d->gpu_cmd.cmd_result = result;
      if (result == CXL_GPU_SUCCESS) {
        if (!cxl_type2_register_gpu_handle(
                &ct2d->gpu_cmd.events, &ct2d->gpu_cmd.events_capacity,
                &ct2d->gpu_cmd.num_events, event, &id)) {
          (void)hetgpu_cuda_event_destroy(hetgpu, event);
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_OUT_OF_MEMORY;
        } else {
          ct2d->gpu_cmd.results[0] = id;
        }
      }
    }
    break;

  case CXL_GPU_CMD_EVENT_DESTROY:
  case CXL_GPU_CMD_EVENT_QUERY:
  case CXL_GPU_CMD_EVENT_SYNC: {
    uint64_t id = ct2d->gpu_cmd.params[0];
    if (id > UINT32_MAX || id >= ct2d->gpu_cmd.num_events ||
        !ct2d->gpu_cmd.events[id]) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
      break;
    }
    int result;
    void *event = ct2d->gpu_cmd.events[id];
    int64_t completion_begin_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
    uint64_t model_htod_completed = 0;
    uint64_t consumer_launches_completed = 0;
    if (cmd == CXL_GPU_CMD_EVENT_DESTROY)
      result = hetgpu_cuda_event_destroy(hetgpu, event);
    else if (cmd == CXL_GPU_CMD_EVENT_QUERY)
      result = hetgpu_cuda_event_query(hetgpu, event);
    else
      result = hetgpu_cuda_event_synchronize(hetgpu, event);
    int64_t completion_end_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
    ct2d->gpu_cmd.cmd_result = result;
    if ((cmd == CXL_GPU_CMD_EVENT_QUERY || cmd == CXL_GPU_CMD_EVENT_SYNC) &&
        result == CXL_GPU_SUCCESS) {
      CXLType2EventHtoDMark *mark =
          cxl_type2_event_htod_mark(ct2d, event, false);
      if (mark) {
        int release_result = cxl_type2_release_pending_htod(
            ct2d, mark->stream, false, mark->sequence,
            cmd == CXL_GPU_CMD_EVENT_QUERY ? "event-query" : "event-sync",
            &model_htod_completed);
        if (release_result != CXL_GPU_SUCCESS) {
          ct2d->gpu_cmd.cmd_result = release_result;
        } else {
          release_result = cxl_type2_release_pending_alias_consumers(
              ct2d, mark->stream, false, mark->alias_sequence,
              &consumer_launches_completed);
          if (release_result != CXL_GPU_SUCCESS) {
            ct2d->gpu_cmd.cmd_result = release_result;
          }
        }
      }
      cxl_type2_record_model_supply_completion(
          ct2d, completion_begin_ns, completion_end_ns,
          cmd == CXL_GPU_CMD_EVENT_SYNC, model_htod_completed,
          consumer_launches_completed);
    }
    if (cmd == CXL_GPU_CMD_EVENT_DESTROY && result == CXL_GPU_SUCCESS) {
      cxl_type2_remove_event_htod_mark(ct2d, event);
      ct2d->gpu_cmd.events[id] = NULL;
    }
  } break;

  case CXL_GPU_CMD_EVENT_RECORD: {
    uint64_t event_id = ct2d->gpu_cmd.params[0];
    void *stream = NULL;
    if (event_id > UINT32_MAX || event_id >= ct2d->gpu_cmd.num_events ||
        !ct2d->gpu_cmd.events[event_id] ||
        !cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[1], &stream)) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
      break;
    }
    ct2d->gpu_cmd.cmd_result = hetgpu_cuda_event_record(
        hetgpu, ct2d->gpu_cmd.events[event_id], stream);
    if (ct2d->gpu_cmd.cmd_result == CXL_GPU_SUCCESS) {
      CXLType2EventHtoDMark *mark =
          cxl_type2_event_htod_mark(ct2d, ct2d->gpu_cmd.events[event_id], true);
      mark->stream = stream;
      mark->sequence = cxl_type2_latest_pending_htod_sequence(ct2d, stream);
      mark->alias_sequence =
          cxl_type2_latest_pending_alias_sequence(ct2d, stream);
    }
  } break;

  case CXL_GPU_CMD_EVENT_ELAPSED_TIME: {
    uint64_t start = ct2d->gpu_cmd.params[0];
    uint64_t end = ct2d->gpu_cmd.params[1];
    float milliseconds = 0.0f;
    uint32_t bits;
    if (start > UINT32_MAX || end > UINT32_MAX ||
        start >= ct2d->gpu_cmd.num_events || end >= ct2d->gpu_cmd.num_events ||
        !ct2d->gpu_cmd.events[start] || !ct2d->gpu_cmd.events[end]) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
      break;
    }
    int result = hetgpu_cuda_event_elapsed_time(hetgpu, &milliseconds,
                                                ct2d->gpu_cmd.events[start],
                                                ct2d->gpu_cmd.events[end]);
    ct2d->gpu_cmd.cmd_result = result;
    if (result == CXL_GPU_SUCCESS) {
      memcpy(&bits, &milliseconds, sizeof(bits));
      ct2d->gpu_cmd.results[0] = bits;
    }
  } break;

  /* Bulk transfer commands - optimized for large memory operations */
  case CXL_GPU_CMD_BULK_HTOD:
    /* Bulk host-to-device transfer using BAR4 region */
    {
      uint64_t bar4_offset = ct2d->gpu_cmd.params[0]; /* Offset in BAR4 */
      uint64_t dst_dev_ptr = ct2d->gpu_cmd.params[1]; /* Device destination */
      size_t xfer_size = ct2d->gpu_cmd.params[2];     /* Transfer size */

      if (xfer_size > CXL_GPU_BULK_TRANSFER_SIZE) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
        break;
      }

      if (hetgpu->initialized) {
        /* Get data from device memory region (BAR4/HDM) */
        uint8_t *mem = cxl_type2_bar4_host_ptr(ct2d, bar4_offset, xfer_size);
        if (mem && cxl_type2_fabric_access_allowed(ct2d, bar4_offset, xfer_size,
                                                   false, false)) {
          if (!cxl_type2_bulk_memsim_access(ct2d, CXL_OP_READ, bar4_offset,
                                            xfer_size, mem)) {
            ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
            break;
          }
          err = hetgpu_memcpy_htod(hetgpu, dst_dev_ptr, mem, xfer_size);
          if (err != HETGPU_SUCCESS) {
            ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
          }
        } else {
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
        }
      } else {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
      }
    }
    break;

  case CXL_GPU_CMD_BULK_HTOD_ASYNC: {
    uint64_t bar4_offset = ct2d->gpu_cmd.params[0];
    uint64_t dst_dev_ptr = ct2d->gpu_cmd.params[1];
    size_t xfer_size = ct2d->gpu_cmd.params[2];
    uint64_t stream_wire = ct2d->gpu_cmd.params[3];
    uint8_t *mem = memory_region_get_ram_ptr(&ct2d->device_mem);
    void *stream = NULL;
    uint64_t requests_before;
    uint64_t bytes_before;
    uint64_t latency_before;
    int64_t aggregate_begin_ns;
    int64_t aggregate_end_ns;
    bool access_ok;
    PCIDevice *pci_dev = PCI_DEVICE(ct2d);

    if (!xfer_size || xfer_size > CXL_GPU_BULK_TRANSFER_SIZE ||
        bar4_offset > ct2d->device_mem_size ||
        xfer_size > ct2d->device_mem_size - bar4_offset ||
        dst_dev_ptr > UINT64_MAX - xfer_size ||
        !cxl_type2_stream_from_wire(ct2d, stream_wire, &stream) ||
        !hetgpu->initialized || !mem ||
        !cxl_type2_fabric_access_allowed(ct2d, bar4_offset, xfer_size, false,
                                         false)) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      break;
    }
    requests_before = ct2d->paired_case.active_cxl_request_count;
    bytes_before = ct2d->paired_case.active_cxl_logical_bytes;
    latency_before = ct2d->paired_case.active_cxl_server_reported_latency_ns;
    aggregate_begin_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
    mem = cxl_type2_bar4_host_ptr(ct2d, bar4_offset, xfer_size);
    if (!mem) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      break;
    }
    access_ok = cxl_type2_bulk_memsim_access(ct2d, CXL_OP_READ, bar4_offset,
                                             xfer_size, mem);
    aggregate_end_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
    qemu_log(
        "CXL WEIGHT_SOURCE event=access-aggregate"
        " run_binding=%" PRIu64 " case=%s case_epoch=%" PRIu64
        " call_id=0x%016" PRIx64 " device_bdf=%04x:%02x:%02x.%x"
        " bar_index=4 bar_offset=0x%016" PRIx64
        " range_bytes=%zu destination_cuda_start=0x%016" PRIx64
        " destination_cuda_bytes=%zu op=read accesses=%" PRIu64
        " bytes=%" PRIu64 " cache_hits=unavailable"
        " cache_misses=unavailable latency_model=unavailable"
        " server_reported_latency_ns=%" PRIu64 " host_ns_begin=%" PRId64
        " host_ns_end=%" PRId64 " result=%s\n",
        ct2d->paired_case.run_binding,
        cxl_type2_paired_case_name(ct2d->paired_case.active_case),
        ct2d->paired_case.active_epoch, ct2d->gpu_cmd.call_id, 0,
        pci_bus_num(pci_get_bus(pci_dev)), PCI_SLOT(pci_dev->devfn),
        PCI_FUNC(pci_dev->devfn), bar4_offset, xfer_size, dst_dev_ptr,
        xfer_size, ct2d->paired_case.active_cxl_request_count - requests_before,
        ct2d->paired_case.active_cxl_logical_bytes - bytes_before,
        ct2d->paired_case.active_cxl_server_reported_latency_ns -
            latency_before,
        aggregate_begin_ns, aggregate_end_ns, access_ok ? "success" : "failed");
    if (!access_ok) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      break;
    }
    ct2d->gpu_cmd.cmd_result = cxl_type2_enqueue_htod_from_host(
        ct2d, dst_dev_ptr, mem, xfer_size, stream, "cxlmem-bar4");
  } break;

  case CXL_GPU_CMD_SOURCE_REGISTER: {
    uint64_t source_id = 0;
    uint64_t failure_index = SIZE_MAX;
    const char *failure_stage = NULL;

    ct2d->gpu_cmd.cmd_result = cxl_type2_direct_source_register(
        ct2d, ct2d->gpu_cmd.params[0], &source_id, false, NULL, &failure_stage,
        &failure_index);
    if (ct2d->gpu_cmd.cmd_result == CXL_GPU_SUCCESS) {
      ct2d->gpu_cmd.results[0] = source_id;
      ct2d->paired_case.active_direct_register_calls++;
    } else {
      cxl_type2_log_direct_source_register_failure(
          ct2d, failure_stage, failure_index, ct2d->gpu_cmd.cmd_result,
          ct2d->gpu_cmd.params[0]);
      if (ct2d->model_supply.route == CXL_TYPE2_MODEL_SUPPLY_CXL_DIRECT) {
        cxl_type2_model_supply_fail(
            ct2d, ct2d->gpu_cmd.call_id,
            cxl_type2_model_supply_failure_stage(failure_stage),
            failure_stage ? failure_stage : "source-register", failure_index, 0,
            0, 0);
      }
    }
  } break;

  case CXL_GPU_CMD_SOURCE_UNREGISTER:
    ct2d->gpu_cmd.cmd_result =
        cxl_type2_direct_source_unregister(ct2d, ct2d->gpu_cmd.params[0]);
    if (ct2d->gpu_cmd.cmd_result == CXL_GPU_SUCCESS) {
      ct2d->paired_case.active_direct_unregister_calls++;
    }
    break;

  case CXL_GPU_CMD_SOURCE_REGISTER_BATCH_HTOD_DIRECT_ASYNC: {
    uint64_t register_bytes = ct2d->gpu_cmd.params[0];
    uint64_t range_count = ct2d->gpu_cmd.params[1];
    uint64_t direct_bytes;
    uint64_t source_id = 0;
    bool ordinary_source = false;
    uint64_t fail_idx = SIZE_MAX;
    uint64_t logical_enqueued = 0;
    uint64_t fragments_enqueued = 0;
    const char *batch_failure_stage = NULL;
    const char *failure_stage = NULL;
    void *stream = NULL;

    ct2d->gpu_cmd.results[0] = SIZE_MAX;
    ct2d->gpu_cmd.results[1] = 0;
    ct2d->gpu_cmd.results[2] = 0;
    ct2d->gpu_cmd.results[3] = CXL_GPU_SOURCE_ROUTE_UNKNOWN;
    if (!hetgpu->initialized || !ct2d->gpu_cmd.batch_data ||
        !ct2d->gpu_cmd.descriptor ||
        ct2d->gpu_cmd.descriptor->protocol_version < 4U ||
        range_count > UINT64_MAX / sizeof(CXLGPUDirectRangeV1) ||
        register_bytes > CXL_GPU_BATCH_DATA_SIZE ||
        !cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[2], &stream)) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      cxl_type2_log_direct_source_register_failure(
          ct2d, "command-prevalidate", SIZE_MAX, ct2d->gpu_cmd.cmd_result,
          register_bytes);
      if (ct2d->model_supply.route == CXL_TYPE2_MODEL_SUPPLY_CXL_DIRECT) {
        cxl_type2_model_supply_fail(ct2d, ct2d->gpu_cmd.call_id,
                                    "batch-validate", "command-prevalidate",
                                    UINT64_MAX, 0, 0, 0);
      }
      break;
    }
    direct_bytes = range_count * sizeof(CXLGPUDirectRangeV1);
    if (direct_bytes > CXL_GPU_BATCH_DATA_SIZE - register_bytes) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      cxl_type2_log_direct_source_register_failure(
          ct2d, "command-prevalidate", SIZE_MAX, ct2d->gpu_cmd.cmd_result,
          register_bytes);
      if (ct2d->model_supply.route == CXL_TYPE2_MODEL_SUPPLY_CXL_DIRECT) {
        cxl_type2_model_supply_fail(ct2d, ct2d->gpu_cmd.call_id,
                                    "batch-validate", "command-prevalidate",
                                    UINT64_MAX, 0, 0, 0);
      }
      break;
    }
    ct2d->gpu_cmd.cmd_result = cxl_type2_direct_source_register(
        ct2d, register_bytes, &source_id, true, &ordinary_source,
        &failure_stage, &fail_idx);
    if (ct2d->gpu_cmd.cmd_result == CXL_GPU_SUCCESS && ordinary_source) {
      ct2d->gpu_cmd.results[3] = CXL_GPU_SOURCE_ROUTE_ORDINARY;
      break;
    }
    if (ct2d->gpu_cmd.cmd_result != CXL_GPU_SUCCESS) {
      ct2d->gpu_cmd.results[0] = fail_idx;
      cxl_type2_log_direct_source_register_failure(
          ct2d, failure_stage, fail_idx, ct2d->gpu_cmd.cmd_result,
          register_bytes);
      if (ct2d->model_supply.route == CXL_TYPE2_MODEL_SUPPLY_CXL_DIRECT) {
        cxl_type2_model_supply_fail(
            ct2d, ct2d->gpu_cmd.call_id,
            cxl_type2_model_supply_failure_stage(failure_stage),
            failure_stage ? failure_stage : "source-register", fail_idx, 0, 0,
            0);
      }
      break;
    }
    ct2d->gpu_cmd.results[3] = CXL_GPU_SOURCE_ROUTE_DIRECT;
    CXLType2DirectSource *source =
        cxl_type2_direct_source_find(ct2d, source_id);

    g_assert(source);
    source->auto_unregister = true;
    ct2d->paired_case.active_direct_register_calls++;
    ct2d->gpu_cmd.cmd_result = cxl_type2_direct_batch_submit(
        ct2d, ct2d->gpu_cmd.batch_data + register_bytes,
        CXL_GPU_BATCH_DATA_SIZE - register_bytes, range_count, direct_bytes,
        stream, source, &fail_idx, &batch_failure_stage, &logical_enqueued,
        &fragments_enqueued);
    ct2d->gpu_cmd.results[0] = fail_idx;
    ct2d->gpu_cmd.results[1] = logical_enqueued;
    ct2d->gpu_cmd.results[2] = fragments_enqueued;
    ct2d->paired_case.active_direct_logical_ranges += logical_enqueued;
    if (ct2d->gpu_cmd.cmd_result != CXL_GPU_SUCCESS) {
      g_assert(batch_failure_stage);
      cxl_type2_log_direct_source_register_failure(
          ct2d, batch_failure_stage, fail_idx, ct2d->gpu_cmd.cmd_result,
          register_bytes + direct_bytes);
      if (ct2d->model_supply.route == CXL_TYPE2_MODEL_SUPPLY_CXL_DIRECT) {
        cxl_type2_model_supply_fail(
            ct2d, ct2d->gpu_cmd.call_id,
            cxl_type2_model_supply_failure_stage(batch_failure_stage),
            batch_failure_stage, fail_idx, 0, 0, source_id);
      }
    }
    if (!source->pending_refcount) {
      int cleanup = cxl_type2_direct_source_unregister(ct2d, source_id);

      if (ct2d->gpu_cmd.cmd_result == CXL_GPU_SUCCESS &&
          cleanup != CXL_GPU_SUCCESS) {
        cxl_type2_log_direct_source_register_failure(
            ct2d, "source-cleanup", SIZE_MAX, cleanup,
            register_bytes + direct_bytes);
        if (ct2d->model_supply.route == CXL_TYPE2_MODEL_SUPPLY_CXL_DIRECT) {
          cxl_type2_model_supply_fail(ct2d, ct2d->gpu_cmd.call_id, "cleanup",
                                      "source-cleanup", UINT64_MAX, 0, 0,
                                      source_id);
        }
        ct2d->gpu_cmd.cmd_result = cleanup;
      }
    }
  } break;

  case CXL_GPU_CMD_BATCH_HTOD_DIRECT_ASYNC: {
    uint64_t fail_idx = SIZE_MAX;
    uint64_t logical_enqueued = 0;
    uint64_t fragments_enqueued = 0;
    const char *failure_stage = NULL;
    void *stream = NULL;

    ct2d->gpu_cmd.results[0] = SIZE_MAX;
    if (!hetgpu->initialized) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
      break;
    }
    if (!ct2d->gpu_cmd.batch_data ||
        !cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[2], &stream)) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      break;
    }
    ct2d->gpu_cmd.cmd_result = cxl_type2_direct_batch_submit(
        ct2d, ct2d->gpu_cmd.batch_data, CXL_GPU_BATCH_DATA_SIZE,
        ct2d->gpu_cmd.params[0], ct2d->gpu_cmd.params[1], stream, NULL,
        &fail_idx, &failure_stage, &logical_enqueued, &fragments_enqueued);
    ct2d->gpu_cmd.results[0] = fail_idx;
    ct2d->gpu_cmd.results[1] = logical_enqueued;
    ct2d->gpu_cmd.results[2] = fragments_enqueued;
    ct2d->paired_case.active_direct_logical_ranges += logical_enqueued;
  } break;

  case CXL_GPU_CMD_BULK_DTOH:
    /* Bulk device-to-host transfer using BAR4 region */
    {
      uint64_t src_dev_ptr = ct2d->gpu_cmd.params[0]; /* Device source */
      uint64_t bar4_offset = ct2d->gpu_cmd.params[1]; /* Offset in BAR4 */
      size_t xfer_size = ct2d->gpu_cmd.params[2];     /* Transfer size */

      if (xfer_size > CXL_GPU_BULK_TRANSFER_SIZE) {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
        break;
      }

      if (hetgpu->initialized) {
        /* Write data to device memory region (BAR4/HDM) */
        uint8_t *mem = cxl_type2_bar4_host_ptr(ct2d, bar4_offset, xfer_size);
        if (mem && cxl_type2_fabric_access_allowed(ct2d, bar4_offset, xfer_size,
                                                   true, false)) {
          err = hetgpu_memcpy_dtoh(hetgpu, mem, src_dev_ptr, xfer_size);
          if (err == HETGPU_SUCCESS) {
            if (!cxl_type2_bulk_memsim_access(ct2d, CXL_OP_WRITE, bar4_offset,
                                              xfer_size, mem)) {
              ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
            }
          }
          if (err != HETGPU_SUCCESS) {
            ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
          }
        } else {
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
        }
      } else {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
      }
    }
    break;

  case CXL_GPU_CMD_BULK_DTOD:
    /* Bulk device-to-device transfer */
    {
      uint64_t src_dev_ptr = ct2d->gpu_cmd.params[0]; /* Source device ptr */
      uint64_t dst_dev_ptr = ct2d->gpu_cmd.params[1]; /* Dest device ptr */
      size_t xfer_size = ct2d->gpu_cmd.params[2];     /* Transfer size */

      if (hetgpu->initialized) {
        err = hetgpu_memcpy_dtod(hetgpu, dst_dev_ptr, src_dev_ptr, xfer_size);
        if (err != HETGPU_SUCCESS) {
          ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
        }
      } else {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
      }
    }
    break;

  /* CXL.cache coherency commands */
  case CXL_GPU_CMD_CACHE_FLUSH:
    /* Flush cache lines to device - notify CXLMemSim */
    {
      uint64_t flush_addr = ct2d->gpu_cmd.params[0];
      size_t flush_size = ct2d->gpu_cmd.params[1];

      if (ct2d->coherency.coherency_enabled && ct2d->memsim.connected) {
        cxl_type2_memsim_request(ct2d, CXL_OP_FENCE, flush_addr, flush_size,
                                 NULL, NULL);
        /* Invalidate local cache entries */
        for (uint64_t addr = flush_addr; addr < flush_addr + flush_size;
             addr += 64) {
          cxl_type2_cache_invalidate(ct2d, addr);
        }
        ct2d->coherency.coherency_ops++;
      }
    }
    break;

  case CXL_GPU_CMD_CACHE_INVALIDATE:
    /* Invalidate cache lines */
    {
      uint64_t inv_addr = ct2d->gpu_cmd.params[0];
      size_t inv_size = ct2d->gpu_cmd.params[1];

      if (ct2d->coherency.coherency_enabled) {
        for (uint64_t addr = inv_addr; addr < inv_addr + inv_size; addr += 64) {
          cxl_type2_cache_invalidate(ct2d, addr);
        }
        ct2d->coherency.coherency_ops++;
      }
    }
    break;

  case CXL_GPU_CMD_CACHE_WRITEBACK:
    /* Writeback dirty cache lines */
    {
      uint64_t wb_addr = ct2d->gpu_cmd.params[0];
      size_t wb_size = ct2d->gpu_cmd.params[1];

      if (ct2d->coherency.coherency_enabled) {
        for (uint64_t addr = wb_addr; addr < wb_addr + wb_size; addr += 64) {
          cxl_type2_cache_writeback(ct2d, addr);
        }
        ct2d->coherency.coherency_ops++;
      }
    }
    break;

  case CXL_GPU_CMD_CACHE_PREFETCH:
    /* Prefetch cache lines into the Type2 coherent cache */
    {
      uint64_t pf_addr = ct2d->gpu_cmd.params[0];
      uint64_t pf_size = ct2d->gpu_cmd.params[1];
      bool write_intent = (ct2d->gpu_cmd.params[2] & 1) != 0;

      cxl_type2_cache_prefetch(ct2d, pf_addr, pf_size, write_intent);
      ct2d->gpu_cmd.results[0] = pf_size;
      ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
    }
    break;

  /* P2P DMA commands */
  case CXL_GPU_CMD_P2P_DISCOVER:
    /* Discover P2P peer devices */
    {
      int num_peers = cxl_p2p_discover_peers(&ct2d->p2p_engine);
      if (num_peers >= 0) {
        ct2d->gpu_cmd.results[0] = num_peers;
        ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
      } else {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
      }
    }
    break;

  case CXL_GPU_CMD_P2P_GET_PEER_INFO:
    /* Get peer device info: params[0] = peer_id */
    {
      uint32_t peer_id = ct2d->gpu_cmd.params[0];
      CXLP2PPeer *peer = cxl_p2p_get_peer(&ct2d->p2p_engine, peer_id);
      if (peer && peer->active) {
        ct2d->gpu_cmd.results[0] = peer->type;
        ct2d->gpu_cmd.results[1] = peer->mem_size;
        ct2d->gpu_cmd.results[2] = peer->coherent;
        ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
      } else {
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
      }
    }
    break;

  case CXL_GPU_CMD_P2P_GPU_TO_MEM:
    /* GPU -> Type3 transfer: params[0]=peer_id, params[1]=gpu_off,
     * params[2]=mem_off, params[3]=size */
    {
      uint32_t t3_peer_id = ct2d->gpu_cmd.params[0];
      uint64_t gpu_offset = ct2d->gpu_cmd.params[1];
      uint64_t mem_offset = ct2d->gpu_cmd.params[2];
      uint64_t xfer_size = ct2d->gpu_cmd.params[3];
      uint32_t flags = CXL_P2P_FLAG_COHERENT;

      int ret = cxl_p2p_gpu_to_mem(&ct2d->p2p_engine, t3_peer_id, gpu_offset,
                                   mem_offset, xfer_size, flags);
      ct2d->gpu_cmd.cmd_result =
          (ret == 0) ? CXL_GPU_SUCCESS : CXL_GPU_ERROR_OUT_OF_MEMORY;
    }
    break;

  case CXL_GPU_CMD_P2P_MEM_TO_GPU:
    /* Type3 -> GPU transfer: params[0]=peer_id, params[1]=mem_off,
     * params[2]=gpu_off, params[3]=size */
    {
      uint32_t t3_peer_id = ct2d->gpu_cmd.params[0];
      uint64_t mem_offset = ct2d->gpu_cmd.params[1];
      uint64_t gpu_offset = ct2d->gpu_cmd.params[2];
      uint64_t xfer_size = ct2d->gpu_cmd.params[3];
      uint32_t flags = CXL_P2P_FLAG_COHERENT;

      int ret = cxl_p2p_mem_to_gpu(&ct2d->p2p_engine, t3_peer_id, mem_offset,
                                   gpu_offset, xfer_size, flags);
      ct2d->gpu_cmd.cmd_result =
          (ret == 0) ? CXL_GPU_SUCCESS : CXL_GPU_ERROR_OUT_OF_MEMORY;
    }
    break;

  case CXL_GPU_CMD_P2P_MEM_TO_MEM:
    /* Type3 -> Type3 transfer: params[0]=src_peer, params[1]=dst_peer,
     * params[2]=src_off, params[3]=dst_off, params[4]=size */
    {
      uint32_t src_peer_id = ct2d->gpu_cmd.params[0];
      uint32_t dst_peer_id = ct2d->gpu_cmd.params[1];
      uint64_t src_offset = ct2d->gpu_cmd.params[2];
      uint64_t dst_offset = ct2d->gpu_cmd.params[3];
      uint64_t xfer_size = ct2d->gpu_cmd.params[4];
      uint32_t flags = CXL_P2P_FLAG_COHERENT;

      int ret = cxl_p2p_mem_to_mem(&ct2d->p2p_engine, src_peer_id, dst_peer_id,
                                   src_offset, dst_offset, xfer_size, flags);
      ct2d->gpu_cmd.cmd_result =
          (ret == 0) ? CXL_GPU_SUCCESS : CXL_GPU_ERROR_OUT_OF_MEMORY;
    }
    break;

  case CXL_GPU_CMD_P2P_SYNC:
    /* Wait for all pending P2P transfers */
    /* Currently all transfers are synchronous, so this is a no-op */
    ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
    break;

  case CXL_GPU_CMD_P2P_GET_STATUS:
    /* Get P2P engine status and stats */
    {
      ct2d->gpu_cmd.results[0] = ct2d->p2p_engine.num_peers;
      ct2d->gpu_cmd.results[1] = ct2d->p2p_engine.stats.transfers_completed;
      ct2d->gpu_cmd.results[2] = ct2d->p2p_engine.stats.bytes_transferred;
      ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
    }
    break;

  /* ---- Coherent shared memory pool commands ---- */
  case CXL_GPU_CMD_COHERENT_ALLOC: {
    uint64_t alloc_size = ct2d->gpu_cmd.params[0];
    int64_t offset = cxl_coherent_pool_alloc(ct2d, alloc_size);
    if (offset >= 0) {
      ct2d->gpu_cmd.results[0] = (uint64_t)offset;
      ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
    } else {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_OUT_OF_MEMORY;
    }
  } break;

  case CXL_GPU_CMD_COHERENT_FREE: {
    uint64_t free_offset = ct2d->gpu_cmd.params[0];
    if (cxl_coherent_pool_free(ct2d, free_offset) == 0) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
    } else {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
    }
  } break;

  case CXL_GPU_CMD_COHERENT_GET_INFO: {
    ct2d->gpu_cmd.results[0] = ct2d->coherent_pool.base_offset;
    ct2d->gpu_cmd.results[1] = ct2d->coherent_pool.size;
    ct2d->gpu_cmd.results[2] =
        ct2d->coherent_pool.size - ct2d->coherent_pool.used;
    ct2d->gpu_cmd.results[3] = ct2d->bar_coherency.snoop_filter_size;
    ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
  } break;

  case CXL_GPU_CMD_COHERENT_FENCE: {
    /* Memory fence - ensure all pending coherency ops complete */
    cxl_bar_memory_fence(&ct2d->bar_coherency, CXL_DOMAIN_CPU);
    cxl_bar_process_back_invalidations(ct2d);
    ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
  } break;

  case CXL_GPU_CMD_COHERENT_MAP_DEVICE: {
    uint64_t alias = 0;
    int can_map = 0;
    int result = cxl_coherent_map_device(
        ct2d, ct2d->gpu_cmd.params[0], ct2d->gpu_cmd.params[1],
        ct2d->gpu_cmd.params[2], &alias, &can_map);
    ct2d->gpu_cmd.cmd_result = result;
    ct2d->gpu_cmd.results[0] = alias;
    ct2d->gpu_cmd.results[1] = ct2d->gpu_cmd.params[1];
    ct2d->gpu_cmd.results[2] = can_map;
    qemu_log("cxl_coherent_device_map status=%d offset=0x%" PRIx64
             " mapped_bytes=%" PRIu64 " request_bytes=%" PRIu64
             " device_alias=0x%" PRIx64 " can_map_host_memory=%d\n",
             result, ct2d->gpu_cmd.params[0], ct2d->gpu_cmd.params[1],
             ct2d->gpu_cmd.params[2], alias, can_map);
  } break;

  case CXL_GPU_CMD_COHERENT_UNMAP_DEVICE: {
    uint64_t htod_delta = 0;
    int result = cxl_coherent_unmap_device(
        ct2d, ct2d->gpu_cmd.params[0], ct2d->gpu_cmd.params[1], &htod_delta);
    ct2d->gpu_cmd.cmd_result = result;
    ct2d->gpu_cmd.results[0] = htod_delta;
    qemu_log("cxl_coherent_device_unmap status=%d offset=0x%" PRIx64
             " device_alias=0x%" PRIx64 " htod_command_delta=%" PRIu64 "\n",
             result, ct2d->gpu_cmd.params[0], ct2d->gpu_cmd.params[1],
             htod_delta);
  } break;

  /* ---- Device-biased directory commands ---- */
  case CXL_GPU_CMD_SET_BIAS: {
    uint64_t bias_addr = ct2d->gpu_cmd.params[0];
    uint64_t bias_size = ct2d->gpu_cmd.params[1];
    uint8_t bias_mode = (uint8_t)ct2d->gpu_cmd.params[2];
    if (bias_mode > CXL_BIAS_DEVICE) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
    } else {
      cxl_bar_set_bias(&ct2d->bar_coherency, bias_addr, bias_size, bias_mode);
      ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
    }
  } break;

  case CXL_GPU_CMD_GET_BIAS: {
    uint64_t query_addr = ct2d->gpu_cmd.params[0];
    ct2d->gpu_cmd.results[0] =
        cxl_bar_get_bias(&ct2d->bar_coherency, query_addr);
    ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
  } break;

  case CXL_GPU_CMD_BIAS_FLIP: {
    uint64_t flip_addr = ct2d->gpu_cmd.params[0];
    uint64_t flip_size = ct2d->gpu_cmd.params[1];
    uint8_t new_bias = (uint8_t)ct2d->gpu_cmd.params[2];
    if (new_bias > CXL_BIAS_DEVICE) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
    } else {
      cxl_bar_bias_flip(ct2d, flip_addr, flip_size, new_bias);
      ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
    }
  } break;

  /* ---- Coherency statistics commands ---- */
  case CXL_GPU_CMD_COH_GET_STATS: {
    CXLBARCoherencyState *coh = &ct2d->bar_coherency;
    /* Pack stats into results and data buffer */
    ct2d->gpu_cmd.results[0] = coh->stats.snoop_hits;
    ct2d->gpu_cmd.results[1] = coh->stats.snoop_misses;
    ct2d->gpu_cmd.results[2] = coh->stats.coherency_requests;
    ct2d->gpu_cmd.results[3] = coh->stats.back_invalidations;
    /* Extended stats in data buffer */
    if (ct2d->gpu_cmd.data_size >= 64) {
      uint64_t *stats_buf = (uint64_t *)ct2d->gpu_cmd.data;
      stats_buf[0] = coh->stats.writebacks;
      stats_buf[1] = coh->stats.evictions;
      stats_buf[2] = coh->stats.bias_flips;
      stats_buf[3] = coh->stats.device_bias_hits;
      stats_buf[4] = coh->stats.host_bias_hits;
      stats_buf[5] = coh->stats.upgrades;
      stats_buf[6] = coh->stats.downgrades;
      stats_buf[7] = coh->snoop_filter_size;
    }
    ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
  } break;

  case CXL_GPU_CMD_COH_RESET_STATS: {
    memset(&ct2d->bar_coherency.stats, 0, sizeof(ct2d->bar_coherency.stats));
    ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
  } break;

  /* ---- DCD / GFAM / MH-SLD fabric-memory commands ---- */
  case CXL_GPU_CMD_DCD_ADD: {
    uint64_t base = ct2d->gpu_cmd.params[0];
    uint64_t add_size = ct2d->gpu_cmd.params[1];
    uint64_t tag = ct2d->gpu_cmd.params[2];
    uint64_t out_base = 0;
    uint64_t out_size = 0;
    uint64_t out_tag = 0;
    int ret;

    ret = cxl_type2_dcd_add(ct2d, base, add_size, tag, &out_base, &out_size,
                            &out_tag);
    if (ret == 0) {
      ct2d->gpu_cmd.results[0] = out_base;
      ct2d->gpu_cmd.results[1] = out_size;
      ct2d->gpu_cmd.results[2] = out_tag;
      ct2d->gpu_cmd.results[3] = ct2d->dcd.allocated;
      if (ct2d->gfam.enabled) {
        for (uint32_t host = 0; host < ct2d->gfam.num_hosts; host++) {
          cxl_type2_gfam_grant(ct2d, host, out_base, out_size,
                               ct2d->gfam.default_permissions);
        }
      }
      ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
    } else {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_OUT_OF_MEMORY;
    }
  } break;

  case CXL_GPU_CMD_DCD_RELEASE: {
    uint64_t base = ct2d->gpu_cmd.params[0];
    uint64_t release_size = ct2d->gpu_cmd.params[1];
    uint64_t tag = ct2d->gpu_cmd.params[2];
    int ret;

    ret = cxl_type2_dcd_release(ct2d, base, release_size, tag);
    if (ret == 0) {
      cxl_type2_gfam_revoke_range(ct2d, base, release_size);
      ct2d->gpu_cmd.results[0] = ct2d->dcd.allocated;
      ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
    } else {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
    }
  } break;

  case CXL_GPU_CMD_DCD_GET_INFO: {
    qemu_mutex_lock(&ct2d->dcd.lock);
    ct2d->gpu_cmd.results[0] = ct2d->device_mem_size;
    ct2d->gpu_cmd.results[1] =
        ct2d->dcd.enabled ? ct2d->dcd.allocated : ct2d->device_mem_size;
    ct2d->gpu_cmd.results[2] =
        ct2d->dcd.enabled ? ct2d->device_mem_size - ct2d->dcd.allocated : 0;
    ct2d->gpu_cmd.results[3] = cxl_type2_dcd_active_extents_locked(ct2d);
    if (ct2d->gpu_cmd.data_size >= sizeof(ct2d->dcd.extents)) {
      memcpy(ct2d->gpu_cmd.data, ct2d->dcd.extents, sizeof(ct2d->dcd.extents));
    }
    qemu_mutex_unlock(&ct2d->dcd.lock);
    ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
  } break;

  case CXL_GPU_CMD_GFAM_GRANT: {
    uint32_t host = ct2d->gpu_cmd.params[0];
    uint64_t base = ct2d->gpu_cmd.params[1];
    uint64_t grant_size = ct2d->gpu_cmd.params[2];
    uint32_t permissions = ct2d->gpu_cmd.params[3];
    int ret;

    ret = cxl_type2_gfam_grant(ct2d, host, base, grant_size, permissions);
    ct2d->gpu_cmd.cmd_result =
        ret == 0 ? CXL_GPU_SUCCESS : CXL_GPU_ERROR_INVALID_VALUE;
  } break;

  case CXL_GPU_CMD_GFAM_REVOKE: {
    uint32_t host = ct2d->gpu_cmd.params[0];
    uint64_t base = ct2d->gpu_cmd.params[1];
    uint64_t revoke_size = ct2d->gpu_cmd.params[2];
    int ret;

    ret = cxl_type2_gfam_revoke(ct2d, host, base, revoke_size);
    ct2d->gpu_cmd.cmd_result =
        ret == 0 ? CXL_GPU_SUCCESS : CXL_GPU_ERROR_INVALID_VALUE;
  } break;

  case CXL_GPU_CMD_GFAM_GET_INFO: {
    qemu_mutex_lock(&ct2d->gfam.lock);
    ct2d->gpu_cmd.results[0] = ct2d->gfam.num_hosts;
    ct2d->gpu_cmd.results[1] = cxl_type2_gfam_active_mappings_locked(ct2d);
    ct2d->gpu_cmd.results[2] = ct2d->gfam.allowed_accesses;
    ct2d->gpu_cmd.results[3] = ct2d->gfam.denied_accesses;
    if (ct2d->gpu_cmd.data_size >= sizeof(ct2d->gfam.mappings)) {
      memcpy(ct2d->gpu_cmd.data, ct2d->gfam.mappings,
             sizeof(ct2d->gfam.mappings));
    }
    qemu_mutex_unlock(&ct2d->gfam.lock);
    ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
  } break;

  case CXL_GPU_CMD_MHSLD_GET_INFO: {
    qemu_mutex_lock(&ct2d->mhsld.lock);
    ct2d->gpu_cmd.results[0] = ct2d->mhsld.num_heads;
    ct2d->gpu_cmd.results[1] = ct2d->mhsld.local_head_id;
    ct2d->gpu_cmd.results[2] = ct2d->mhsld.reads;
    ct2d->gpu_cmd.results[3] = ct2d->mhsld.writes;
    if (ct2d->gpu_cmd.data_size >= 4 * sizeof(uint64_t)) {
      uint64_t *stats = (uint64_t *)ct2d->gpu_cmd.data;

      stats[0] = ct2d->mhsld.atomics;
      stats[1] = ct2d->mhsld.conflicts;
      stats[2] = ct2d->mhsld.invalidations;
      stats[3] = ct2d->mhsld.coherency_latency_ns;
    }
    qemu_mutex_unlock(&ct2d->mhsld.lock);
    ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
  } break;

  case CXL_GPU_CMD_MHSLD_SET_HEAD: {
    uint32_t head = ct2d->gpu_cmd.params[0];

    if (head >= ct2d->mhsld.num_heads) {
      ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
    } else {
      ct2d->mhsld.local_head_id = head;
      ct2d->gpu_cmd.results[0] = head;
      ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
    }
  } break;

  default:
    ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
    break;
  }

complete:
  if (cmd == CXL_GPU_CMD_STREAM_SYNC &&
      ct2d->gpu_cmd.cmd_result == CXL_GPU_SUCCESS) {
    ct2d->paired_case.last_command_was_successful_stream_sync = true;
    ct2d->paired_case.last_successful_stream_sync_wire =
        ct2d->gpu_cmd.params[0];
  }
  hetgpu_cuda_trace_set_call_id(0);
  ct2d->gpu_cmd.cmd_status = CXL_GPU_CMD_STATUS_COMPLETE;
  int64_t trace_end_ns = cxl_type2_host_monotonic_ns();
  int64_t trace_duration_ns = trace_end_ns - trace_start_ns;
  if (cmd != CXL_GPU_CMD_CASE_BEGIN && cmd != CXL_GPU_CMD_CASE_END &&
      cmd != CXL_GPU_CMD_OBSERVATION_ANCHOR) {
    cxl_type2_record_case_command(ct2d, cmd, trace_sequence, trace_start_ns,
                                  trace_end_ns, ct2d->gpu_cmd.cmd_result,
                                  stream_sync_reason_valid, stream_sync_reason,
                                  stream_sync_elided);
  }
  ct2d->paired_case.active_command_sequence = 0;
  ct2d->paired_case.active_command_code = 0;
  ct2d->paired_case.active_stream_sync_reason_valid = false;
  if (ct2d->paired_case.qemu_cuda_calls_enabled)
    qemu_log("CXL TYPE2 TRACE cmd_end seq=%" PRIu64 " call_id=0x%016" PRIx64
             " cmd=0x%x host_ns=%" PRId64 " result=%u duration_ns=%" PRId64
             " context_generation=%" PRIu64 " context_binding_hits=%" PRIu64
             " context_binding_misses=%" PRIu64 "\n",
             trace_sequence, ct2d->gpu_cmd.call_id, cmd, trace_end_ns,
             ct2d->gpu_cmd.cmd_result, trace_duration_ns,
             hetgpu->context_generation, hetgpu->context_binding_hits,
             hetgpu->context_binding_misses);
  if (ct2d->paired_case.qemu_cuda_calls_enabled)
    qemu_log_mask(
        LOG_GUEST_ERROR, "CXL GPU: cmd 0x%x done, result=%u results[0]=0x%lx\n",
        cmd, ct2d->gpu_cmd.cmd_result, (unsigned long)ct2d->gpu_cmd.results[0]);
}

static uint64_t cxl_type2_gpu_cmd_read(void *opaque, hwaddr addr,
                                       unsigned size) {
  CXLType2State *ct2d = opaque;
  HetGPUState *hetgpu = &ct2d->gpu_info.hetgpu_state;
  uint64_t value = 0;

  switch (addr) {
  case CXL_GPU_REG_MAGIC:
    value = CXL_GPU_MAGIC;
    break;
  case CXL_GPU_REG_VERSION:
    value = CXL_GPU_VERSION;
    break;
  case CXL_GPU_REG_STATUS:
    value = ct2d->gpu_cmd.status;
    if (hetgpu->initialized) {
      value |= CXL_GPU_STATUS_READY;
    }
    if (hetgpu->context) {
      value |= CXL_GPU_STATUS_CTX_ACTIVE;
    }
    break;
  case CXL_GPU_REG_CAPS:
    value = ct2d->gpu_cmd.capabilities;
    break;
  case CXL_GPU_REG_TOTAL_MEM:
  case CXL_GPU_REG_FREE_MEM:
    value = 0;
    break;
  case CXL_GPU_REG_CC_MAJOR:
    value = hetgpu->initialized ? hetgpu->props.compute_capability_major : 8;
    break;
  case CXL_GPU_REG_CC_MINOR:
    value = hetgpu->initialized ? hetgpu->props.compute_capability_minor : 0;
    break;
  case CXL_GPU_REG_MP_COUNT:
    value = hetgpu->initialized ? hetgpu->props.multiprocessor_count : 80;
    break;
  case CXL_GPU_REG_MAX_THREADS:
    value = hetgpu->initialized ? hetgpu->props.max_threads_per_block : 1024;
    break;
  case CXL_GPU_REG_WARP_SIZE:
    value = hetgpu->initialized ? hetgpu->props.warp_size : 32;
    break;
  case CXL_GPU_REG_BACKEND:
    value = hetgpu->backend;
    break;
  case CXL_GPU_REG_DRIVER_VERSION:
    value = hetgpu->driver_version;
    break;
  /* Coherent pool registers */
  case CXL_GPU_REG_COH_POOL_BASE:
    value = ct2d->coherent_pool.base_offset;
    break;
  case CXL_GPU_REG_COH_POOL_SIZE:
    value = ct2d->coherent_pool.size;
    break;
  case CXL_GPU_REG_COH_POOL_FREE:
    value = ct2d->coherent_pool.size - ct2d->coherent_pool.used;
    break;
  case CXL_GPU_REG_COH_DIR_SIZE:
    value = ct2d->bar_coherency.snoop_filter_capacity;
    break;
  case CXL_GPU_REG_COH_DIR_USED:
    value = ct2d->bar_coherency.snoop_filter_size;
    break;
  case CXL_GPU_REG_DCD_TOTAL:
    value = ct2d->device_mem_size;
    break;
  case CXL_GPU_REG_DCD_ALLOCATED:
    value = ct2d->dcd.enabled ? ct2d->dcd.allocated : ct2d->device_mem_size;
    break;
  case CXL_GPU_REG_DCD_FREE:
    value = ct2d->dcd.enabled ? ct2d->device_mem_size - ct2d->dcd.allocated : 0;
    break;
  case CXL_GPU_REG_DCD_EXTENTS:
    qemu_mutex_lock(&ct2d->dcd.lock);
    value = cxl_type2_dcd_active_extents_locked(ct2d);
    qemu_mutex_unlock(&ct2d->dcd.lock);
    break;
  case CXL_GPU_REG_GFAM_HOSTS:
    value = ct2d->gfam.num_hosts;
    break;
  case CXL_GPU_REG_GFAM_MAPPINGS:
    qemu_mutex_lock(&ct2d->gfam.lock);
    value = cxl_type2_gfam_active_mappings_locked(ct2d);
    qemu_mutex_unlock(&ct2d->gfam.lock);
    break;
  case CXL_GPU_REG_GFAM_DENIED:
    value = ct2d->gfam.denied_accesses;
    break;
  case CXL_GPU_REG_MHSLD_HEADS:
    value = ct2d->mhsld.num_heads;
    break;
  case CXL_GPU_REG_MHSLD_HEAD_ID:
    value = ct2d->mhsld.local_head_id;
    break;
  case CXL_GPU_REG_MHSLD_CONFLICTS:
    value = ct2d->mhsld.conflicts;
    break;
  case CXL_GPU_REG_MHSLD_INV:
    value = ct2d->mhsld.invalidations;
    break;

  default:
    if (addr >= CXL_GPU_REG_DEV_NAME && addr < CXL_GPU_REG_DEV_NAME + 64) {
      /* Device name */
      size_t offset = addr - CXL_GPU_REG_DEV_NAME;
      if (hetgpu->initialized) {
        memcpy(&value, &hetgpu->props.name[offset], MIN(size, 8));
      }
    }
    break;
  }

  return value;
}

static void cxl_type2_descriptor_publish_error(CXLType2State *ct2d,
                                               uint64_t submission,
                                               uint64_t generation,
                                               const char *reason) {
  CXLGPURAMCommandDescriptor *descriptor = ct2d->gpu_cmd.descriptor;

  qemu_log_mask(LOG_GUEST_ERROR,
                "CXL GPU descriptor protocol error: %s submission=%" PRIu64
                " generation=%" PRIu64 "\n",
                reason, submission, generation);
  descriptor->completion_submission = submission;
  descriptor->completion_device_generation = generation;
  descriptor->result = CXL_GPU_ERROR_UNKNOWN;
  memset(descriptor->results, 0, sizeof(descriptor->results));
  descriptor->active_case_epoch = ct2d->paired_case.active_epoch;
  descriptor->device_generation = 0;
  ct2d->gpu_cmd.device_generation = 0;
  qatomic_store_release(&descriptor->completion_status,
                        CXL_GPU_DESCRIPTOR_COMPLETION_ERROR);
}

static void cxl_type2_gpu_descriptor_doorbell(CXLType2State *ct2d,
                                              uint64_t value, unsigned size) {
  CXLGPURAMCommandDescriptor request;
  CXLGPURAMCommandDescriptor *descriptor = ct2d->gpu_cmd.descriptor;
  CXLType2DescriptorRequestVerdict verdict;

  if (!descriptor) {
    return;
  }
  smp_rmb();
  memcpy(&request, descriptor, sizeof(request));

  verdict = cxl_type2_descriptor_validate_request(
      &request, value, size, ct2d->gpu_cmd.device_generation,
      ct2d->gpu_cmd.last_accepted_submission,
      ct2d->gpu_cmd.last_completed_submission, ct2d->paired_case.required,
      ct2d->paired_case.active_epoch);
  if (verdict == CXL_TYPE2_DESCRIPTOR_DUPLICATE) {
    return;
  }
  if (verdict != CXL_TYPE2_DESCRIPTOR_ACCEPT) {
    cxl_type2_descriptor_publish_error(
        ct2d, request.request_submission, request.request_device_generation,
        cxl_type2_descriptor_verdict_reason(verdict));
    return;
  }

  ct2d->gpu_cmd.last_accepted_submission = request.request_submission;
  ct2d->gpu_cmd.call_id = request.request_call_id;
  memcpy(ct2d->gpu_cmd.params, request.params, sizeof(request.params));
  memset(ct2d->gpu_cmd.results, 0, sizeof(ct2d->gpu_cmd.results));
  cxl_type2_gpu_execute_cmd(ct2d, request.request_command);

  descriptor->completion_submission = request.request_submission;
  descriptor->completion_device_generation = request.request_device_generation;
  descriptor->result = ct2d->gpu_cmd.cmd_result;
  memcpy(descriptor->results, ct2d->gpu_cmd.results,
         sizeof(descriptor->results));
  descriptor->active_case_epoch = ct2d->paired_case.active_epoch;
  descriptor->device_generation = ct2d->gpu_cmd.device_generation;
  ct2d->gpu_cmd.last_completed_submission = request.request_submission;
  qatomic_store_release(&descriptor->completion_status,
                        CXL_GPU_DESCRIPTOR_COMPLETION_COMPLETE);
}

static void cxl_type2_gpu_cmd_write(void *opaque, hwaddr addr, uint64_t value,
                                    unsigned size) {
  CXLType2State *ct2d = opaque;

  switch (addr) {
  case CXL_GPU_REG_CMD:
    cxl_type2_gpu_descriptor_doorbell(ct2d, value, size);
    break;
  default:
    break;
  }
}

/* GPU command ops handled directly in cache_read/cache_write */

/* ========================================================================
 * DVSEC Configuration
 * ======================================================================== */

static void cxl_type2_bi_control_write(CXLComponentState *cxl_cstate,
                                       uint32_t old_ctrl, uint32_t new_ctrl,
                                       void *opaque) {
  CXLType2State *ct2d = opaque;
  bool old_enabled = FIELD_EX32(old_ctrl, CXL_BI_DECODER_CTRL, BI_ENABLE);
  bool new_enabled = FIELD_EX32(new_ctrl, CXL_BI_DECODER_CTRL, BI_ENABLE);

  if (old_enabled == new_enabled) {
    return;
  }

  ct2d->bi_enabled = new_enabled;
  cxl_type2_memsim_request(ct2d,
                           new_enabled ? CXL_OP_BI_ENABLE : CXL_OP_BI_DISABLE,
                           0, ct2d->device_mem_size, NULL, NULL);
  qemu_log("CXL Type2: BI %s via decoder control\n",
           new_enabled ? "enabled" : "disabled");
}

static void build_dvsecs(CXLType2State *ct2d) {
  CXLComponentState *cxl_cstate = &ct2d->cxl_cstate;
  uint8_t *dvsec;

  /* Type 2 Device DVSEC - includes both cache and memory capabilities */
  dvsec = (uint8_t *)&(CXLDVSECDevice){
      .cap = 0x1f, /* Cache+, IO+, Mem+, Mem HWInit+, HDMCount=1 */
      .ctrl = 0x7, /* Cache+, IO+, Mem+ enabled */
      .status = 0,
      .ctrl2 = 0,
      .status2 = 0x2,
      .lock = 0,
      .cap2 = (ct2d->cache_size >> 20) & 0xFFFF, /* Cache size in MB */
      .range1_size_hi = ct2d->cache_size >> 32,
      .range1_size_lo =
          (ct2d->cache_size & 0xFFFFFFF0) | 0x3, /* Cache: Valid, Active */
      .range1_base_hi = 0,
      .range1_base_lo = 0,
      .range2_size_hi = ct2d->device_mem_size >> 32,
      .range2_size_lo =
          (ct2d->device_mem_size & 0xFFFFFFF0) | 0x1, /* Mem: Valid */
      .range2_base_hi = 0,
      .range2_base_lo = 0,
  };

  cxl_component_create_dvsec(
      cxl_cstate, CXL2_TYPE3_DEVICE, PCIE_CXL_DEVICE_DVSEC_LENGTH,
      PCIE_CXL_DEVICE_DVSEC, PCIE_CXL31_DEVICE_DVSEC_REVID, dvsec);

  /* Register Locator DVSEC
   * Type 2 devices only have component registers in BAR0
   * BAR2 is used for cache memory, not CXL device registers
   */
  dvsec = (uint8_t *)&(CXLDVSECRegisterLocator){
      .rsvd = 0,
      .reg0_base_lo = RBI_COMPONENT_REG | CXL_COMPONENT_REG_BAR_IDX,
      .reg0_base_hi = 0,
      .reg1_base_lo = RBI_EMPTY, /* No device registers - Type 2 uses cache
                                    memory at BAR2 */
      .reg1_base_hi = 0,
  };

  cxl_component_create_dvsec(cxl_cstate, CXL2_TYPE3_DEVICE,
                             REG_LOC_DVSEC_LENGTH, REG_LOC_DVSEC,
                             REG_LOC_DVSEC_REVID, dvsec);

  /* FlexBus Port DVSEC */
  dvsec = (uint8_t *)&(CXLDVSECPortFlexBus){
      .cap = 0x26,
      .ctrl = 0x02,
      .status = ct2d->flitmode ? 0x6 : 0x26,
      .rcvd_mod_ts_data_phase1 = 0xef,
  };

  cxl_component_create_dvsec(
      cxl_cstate, CXL2_TYPE3_DEVICE, PCIE_CXL3_FLEXBUS_PORT_DVSEC_LENGTH,
      PCIE_FLEXBUS_PORT_DVSEC, PCIE_CXL3_FLEXBUS_PORT_DVSEC_REVID, dvsec);
}

/* ========================================================================
 * Device Lifecycle
 * ======================================================================== */

static void cxl_type2_reset(DeviceState *dev) {
  CXLType2State *ct2d = CXL_TYPE2(dev);
  CXLComponentState *cxl_cstate = &ct2d->cxl_cstate;
  uint32_t *reg_state = cxl_cstate->crb.cache_mem_registers;
  uint32_t *write_msk = cxl_cstate->crb.cache_mem_regs_write_mask;

  pcie_cap_fill_link_ep_usp(PCI_DEVICE(dev), ct2d->width, ct2d->speed,
                            ct2d->flitmode);
  cxl_component_register_init_common(reg_state, write_msk, CXL2_TYPE3_DEVICE,
                                     ct2d->hdmdb);
  cxl_cstate->bi_control_write = cxl_type2_bi_control_write;
  cxl_cstate->bi_control_opaque = ct2d;
  ct2d->bi_enabled = false;
  cxl_type2_cuda_allocation_table_reset(&ct2d->cuda_allocations);

  if (ct2d->direct_sources || ct2d->direct_physicals ||
      ct2d->direct_registrations) {
    int result = HETGPU_ERROR_UNKNOWN;

    if (ct2d->gpu_info.hetgpu_state.initialized &&
        hetgpu_synchronize(&ct2d->gpu_info.hetgpu_state) == HETGPU_SUCCESS &&
        cxl_type2_release_pending_htod(ct2d, NULL, true, UINT64_MAX,
                                       "device-reset",
                                       NULL) == CXL_GPU_SUCCESS &&
        cxl_type2_release_pending_alias_consumers(ct2d, NULL, true, UINT64_MAX,
                                                  NULL) == CXL_GPU_SUCCESS) {
      result = cxl_type2_direct_sources_cleanup(ct2d);
    }
    if (result != CXL_GPU_SUCCESS) {
      ct2d->direct_source_poisoned = true;
      ct2d->paired_case.failed = true;
      ct2d->paired_case.failure_code = result;
    }
  }

  /* Reset statistics */
  memset(&ct2d->stats, 0, sizeof(ct2d->stats));

  if (ct2d->gpu_cmd.descriptor) {
    CXLGPURAMCommandDescriptor *descriptor = ct2d->gpu_cmd.descriptor;

    if (ct2d->paired_case.active_case != CXL_GPU_CASE_NONE ||
        ct2d->gpu_cmd.device_generation == UINT64_MAX) {
      ct2d->gpu_cmd.device_generation = 0;
      descriptor->device_generation = 0;
    } else {
      uint64_t generation = ct2d->gpu_cmd.device_generation + 1;
      memset(descriptor, 0, sizeof(*descriptor));
      ct2d->gpu_cmd.device_generation = generation;
      ct2d->gpu_cmd.last_accepted_submission = 0;
      ct2d->gpu_cmd.last_completed_submission = 0;
      descriptor->device_generation = generation;
    }
  }

  qemu_log("CXL Type2: Device reset\n");
}

static void cxl_type2_realize(PCIDevice *pci_dev, Error **errp) {
  CXLType2State *ct2d = CXL_TYPE2(pci_dev);
  CXLComponentState *cxl_cstate = &ct2d->cxl_cstate;
  Error *local_err = NULL;

  pci_config_set_prog_interface(pci_dev->config, 0x10);

  /* Set default values */
  if (!ct2d->memsim.server_addr) {
    ct2d->memsim.server_addr = g_strdup("127.0.0.1");
  }
  if (ct2d->memsim.server_port == 0) {
    ct2d->memsim.server_port = 9999;
  }
  if (ct2d->cache_size == 0) {
    ct2d->cache_size = CXL_TYPE2_DEFAULT_CACHE_SIZE;
  }
  if (ct2d->device_mem_size == 0) {
    ct2d->device_mem_size = CXL_TYPE2_DEFAULT_MEM_SIZE;
  }
  if (ct2d->cache_size < CXL_GPU_CMD_REG_SIZE) {
    error_setg(errp,
               "cache-size (%" PRIu64
               ") is smaller than the BAR2 GPU command region (%u)",
               ct2d->cache_size, CXL_GPU_CMD_REG_SIZE);
    return;
  }
  if (!cxl_type2_model_supply_route_parse(ct2d->model_supply.route_text,
                                          &ct2d->model_supply.route, errp)) {
    return;
  }
  if (ct2d->model_supply.route != CXL_TYPE2_MODEL_SUPPLY_CXL_DIRECT) {
    if (ct2d->model_supply.member_manifest_path ||
        ct2d->model_supply.member_manifest_sha256 ||
        ct2d->model_supply.consumer_certificate_path ||
        ct2d->model_supply.consumer_certificate_sha256) {
      error_setg(errp, "%s does not accept model supply manifests",
                 cxl_type2_model_supply_route_name(ct2d->model_supply.route));
      return;
    }
    if (ct2d->model_supply.route == CXL_TYPE2_MODEL_SUPPLY_GPU_RESIDENT &&
        (ct2d->cuda_direct_source || ct2d->model_aperture.size)) {
      error_setg(errp, "gpu-resident requires cuda-direct-source=off and "
                       "a zero model aperture");
      return;
    }
  } else {
    if (!ct2d->cuda_direct_source || !ct2d->direct_source_fs ||
        !ct2d->model_aperture.size ||
        !ct2d->model_supply.member_manifest_path ||
        !ct2d->model_supply.member_manifest_sha256 ||
        !ct2d->model_supply.consumer_certificate_path ||
        !ct2d->model_supply.consumer_certificate_sha256 ||
        ct2d->gpu_info.mode != CXL_TYPE2_GPU_MODE_HETGPU ||
        ct2d->gpu_info.hetgpu_backend != HETGPU_BACKEND_NVIDIA ||
        !ct2d->gpu_info.hetgpu_lib_path ||
        ct2d->gpu_info.hetgpu_lib_path[0] != '/') {
      error_setg(errp, "cxl-direct requires cuda-direct-source, a DAX "
                       "direct-source-fs, a nonzero model aperture, a sealed "
                       "model member and consumer manifests, and explicit "
                       "NVIDIA hetGPU mode");
      return;
    }
    if (!cxl_type2_model_member_manifest_load(
            &ct2d->model_supply.member_manifest,
            ct2d->model_supply.member_manifest_path,
            ct2d->model_supply.member_manifest_sha256,
            ct2d->model_aperture.size, errp)) {
      return;
    }
    if (!cxl_type2_model_consumer_certificate_load(
            &ct2d->model_supply.consumer_certificate,
            ct2d->model_supply.consumer_certificate_path,
            ct2d->model_supply.consumer_certificate_sha256, errp)) {
      return;
    }
  }

  {
    size_t host_page_size = qemu_real_host_page_size();
    uint64_t coherent_pool_size = 256 * MiB;
    bool gate_requested =
        ct2d->model_alias_gate.output || ct2d->model_alias_gate.fixture ||
        ct2d->model_alias_gate.mode || ct2d->model_alias_gate.length ||
        ct2d->model_alias_gate.file_offset ||
        ct2d->model_alias_gate.shmem_offset;

    if (coherent_pool_size > ct2d->device_mem_size / 2) {
      coherent_pool_size = ct2d->device_mem_size / 4;
    }
    if (ct2d->model_aperture.offset && !ct2d->model_aperture.size) {
      error_setg(errp, "model-aperture-offset requires a nonzero size");
      return;
    }
    if (ct2d->model_aperture.size &&
        ((ct2d->model_aperture.offset % host_page_size) ||
         (ct2d->model_aperture.size % host_page_size) ||
         !cxl_type2_range_contains(0, ct2d->device_mem_size,
                                   ct2d->model_aperture.offset,
                                   ct2d->model_aperture.size) ||
         ct2d->model_aperture.offset + ct2d->model_aperture.size >
             ct2d->device_mem_size - coherent_pool_size)) {
      error_setg(errp, "model aperture must be host-page aligned, "
                       "inside BAR4, and below the coherent pool");
      return;
    }
    if (gate_requested) {
      if (!ct2d->model_alias_gate.output ||
          ct2d->model_alias_gate.output[0] != '/' ||
          !ct2d->model_alias_gate.fixture ||
          ct2d->model_alias_gate.fixture[0] != '/' ||
          !ct2d->model_alias_gate.mode ||
          (strcmp(ct2d->model_alias_gate.mode, "read") &&
           strcmp(ct2d->model_alias_gate.mode, "write-negative")) ||
          !ct2d->model_alias_gate.length ||
          (ct2d->model_alias_gate.file_offset % host_page_size) ||
          (ct2d->model_alias_gate.shmem_offset % host_page_size) ||
          (ct2d->model_alias_gate.length % host_page_size) ||
          ct2d->model_alias_gate.length < 10 * host_page_size ||
          ct2d->model_alias_gate.length > ct2d->model_aperture.size ||
          !ct2d->direct_source_fs ||
          ct2d->gpu_info.mode != CXL_TYPE2_GPU_MODE_HETGPU ||
          ct2d->gpu_info.hetgpu_backend != HETGPU_BACKEND_NVIDIA ||
          !ct2d->gpu_info.hetgpu_lib_path ||
          ct2d->gpu_info.hetgpu_lib_path[0] != '/') {
        error_setg(errp, "model alias gate requires absolute fixture, "
                         "output and hetgpu-lib paths, an aligned ten-page "
                         "fixture range inside the model aperture, a DAX "
                         "direct-source-fs, and explicit NVIDIA hetGPU mode");
        return;
      }
    }
  }

  if (ct2d->paired_case.required) {
    if (!ct2d->paired_case.run_root || ct2d->paired_case.run_root[0] != '/' ||
        ct2d->paired_case.run_binding == 0) {
      error_setg(errp, "paired case control requires absolute "
                       "paired-run-root and nonzero paired-run-binding");
      return;
    }
    if (ct2d->gpu_info.mode != CXL_TYPE2_GPU_MODE_HETGPU ||
        ct2d->gpu_info.hetgpu_backend != HETGPU_BACKEND_NVIDIA ||
        !ct2d->gpu_info.hetgpu_lib_path ||
        ct2d->gpu_info.hetgpu_lib_path[0] != '/') {
      error_setg(errp, "paired case control requires exact hetGPU "
                       "mode, NVIDIA backend, and absolute hetgpu-lib");
      return;
    }
    if (ct2d->paired_case.min_allocation_bytes < 4096 ||
        ct2d->paired_case.max_regions == 0 ||
        (ct2d->paired_case.checkpoint_enabled &&
         ct2d->paired_case.checkpoint_every_launches == 0) ||
        (!ct2d->paired_case.checkpoint_enabled &&
         ct2d->paired_case.checkpoint_every_launches != 0)) {
      error_setg(errp,
                 "paired case Concordia limits require a positive "
                 "checkpoint interval when enabled and zero when disabled; "
                 "min allocation must be at least 4096 bytes");
      return;
    }
    ct2d->paired_case.next_epoch = 1;
    ct2d->paired_case.active_case = CXL_GPU_CASE_NONE;
  }

  if ((ct2d->direct_registration_tile_size ||
       ct2d->direct_registration_padding_limit) &&
      !ct2d->cuda_direct_source) {
    error_setg(errp, "direct registration tiling requires "
                     "cuda-direct-source=on");
    return;
  }
  if (ct2d->cuda_direct_source) {
    VirtioSharedMemory *source_shmem;
    MemoryRegion *source_mr;
    size_t host_page_size = qemu_real_host_page_size();

    if (!ct2d->direct_source_fs ||
        !vhost_user_fs_pci_get_dax(ct2d->direct_source_fs, &source_shmem,
                                   &source_mr)) {
      error_setg(errp, "cuda-direct-source requires direct-source-fs "
                       "to reference a realized vhost-user-fs-pci DAX device");
      return;
    }
    if (!!ct2d->direct_registration_tile_size !=
        !!ct2d->direct_registration_padding_limit) {
      error_setg(errp, "direct registration tile and padding limit "
                       "must both be zero or both be nonzero");
      return;
    }
    if ((ct2d->direct_registration_tile_size % host_page_size) ||
        (ct2d->direct_registration_padding_limit % host_page_size)) {
      error_setg(errp, "direct registration tile and padding limit "
                       "must be host-page aligned");
      return;
    }
  }

  if (ct2d->hdmdb && !ct2d->flitmode) {
    error_setg(errp, "hdm-db requires operating in 256B flit mode");
    return;
  }

  /* Initialize Type2 CXL.mem fabric feature models. */
  cxl_type2_fabric_features_init(ct2d);

  /* Initialize coherency protocol */
  cxl_type2_coherency_init(ct2d);

  /* Initialize enhanced BAR coherency tracking */
  cxl_bar_coherency_init(&ct2d->bar_coherency);

  /* Initialize P2P DMA engine */
  cxl_p2p_dma_init(&ct2d->p2p_engine, ct2d);

  /* Initialize CXLMemSim connection */
  qemu_mutex_init(&ct2d->memsim.lock);

  /* Setup PCIe capabilities */
  pcie_endpoint_cap_init(pci_dev, 0x80);
  if (ct2d->sn != 0) {
    pcie_dev_ser_num_init(pci_dev, 0x100, ct2d->sn);
    cxl_cstate->dvsec_offset = 0x100 + 0x0c;
  } else {
    cxl_cstate->dvsec_offset = 0x100;
  }

  ct2d->cxl_cstate.pdev = pci_dev;
  build_dvsecs(ct2d);

  cxl_component_register_block_init(OBJECT(pci_dev), cxl_cstate,
                                    TYPE_CXL_TYPE2);

  /* BAR0: standard CXL component register block */
  pci_register_bar(pci_dev, 0,
                   PCI_BASE_ADDRESS_SPACE_MEMORY | PCI_BASE_ADDRESS_MEM_TYPE_64,
                   &cxl_cstate->crb.component_registers);

  /* BAR2: Cache memory region (Type 1 feature) */
  memory_region_init_ram(&ct2d->cache_mem, OBJECT(ct2d), "cxl-type2-cache",
                         ct2d->cache_size, &local_err);
  if (local_err) {
    error_propagate(errp, local_err);
    return;
  }

  memory_region_init_io(&ct2d->cache_io, OBJECT(ct2d), &cxl_type2_cache_ops,
                        ct2d, "cxl-type2-cache-io", ct2d->cache_size);

  memory_region_add_subregion_overlap(&ct2d->cache_mem, 0, &ct2d->cache_io, 1);

  memory_region_init_ram(&ct2d->gpu_descriptor_mem, OBJECT(ct2d),
                         "cxl-type2-gpu-command-descriptor",
                         CXL_GPU_DESCRIPTOR_REGION_SIZE, &local_err);
  if (local_err) {
    error_propagate(errp, local_err);
    return;
  }
  memory_region_add_subregion_overlap(&ct2d->cache_mem,
                                      CXL_GPU_DESCRIPTOR_OFFSET,
                                      &ct2d->gpu_descriptor_mem, 2);

  /* The command data window is a payload mailbox, not a CXL cache access.
   * Expose it as RAM so guest bulk copies do not dispatch one device callback
   * per store through cache_io. */
  memory_region_init_ram(&ct2d->gpu_data_mem, OBJECT(ct2d),
                         "cxl-type2-gpu-data", CXL_GPU_DATA_SIZE, &local_err);
  if (local_err) {
    error_propagate(errp, local_err);
    return;
  }
  memory_region_add_subregion_overlap(&ct2d->cache_mem, CXL_GPU_DATA_OFFSET,
                                      &ct2d->gpu_data_mem, 2);

  memory_region_init_ram(&ct2d->gpu_batch_data_mem, OBJECT(ct2d),
                         "cxl-type2-gpu-batch-data", CXL_GPU_BATCH_DATA_SIZE,
                         &local_err);
  if (local_err) {
    error_propagate(errp, local_err);
    return;
  }
  memory_region_add_subregion_overlap(&ct2d->cache_mem,
                                      CXL_GPU_BATCH_DATA_OFFSET,
                                      &ct2d->gpu_batch_data_mem, 2);

  pci_register_bar(pci_dev, 2,
                   PCI_BASE_ADDRESS_SPACE_MEMORY |
                       PCI_BASE_ADDRESS_MEM_TYPE_64 |
                       PCI_BASE_ADDRESS_MEM_PREFETCH,
                   &ct2d->cache_mem);

  /* BAR4: Device-attached memory (Type 3 feature) */
  memory_region_init_ram(&ct2d->device_mem, OBJECT(ct2d),
                         "cxl-type2-device-mem", ct2d->device_mem_size,
                         &local_err);
  if (local_err) {
    error_propagate(errp, local_err);
    return;
  }

  memory_region_init_io(&ct2d->device_mem_io, OBJECT(ct2d),
                        &cxl_type2_device_mem_ops, ct2d,
                        "cxl-type2-device-mem-io", ct2d->device_mem_size);

  memory_region_add_subregion_overlap(&ct2d->device_mem, 0,
                                      &ct2d->device_mem_io, 1);

  pci_register_bar(pci_dev, 4,
                   PCI_BASE_ADDRESS_SPACE_MEMORY |
                       PCI_BASE_ADDRESS_MEM_TYPE_64 |
                       PCI_BASE_ADDRESS_MEM_PREFETCH,
                   &ct2d->device_mem);

  /* Register BAR regions for enhanced coherency tracking */
  cxl_bar_coherency_add_region(&ct2d->bar_coherency, 2, 0, ct2d->cache_size,
                               true,  /* GPU accessible */
                               true); /* CPU accessible */
  cxl_bar_coherency_add_region(&ct2d->bar_coherency, 4, 0,
                               ct2d->device_mem_size, true, /* GPU accessible */
                               true);                       /* CPU accessible */

  /* Initialize GPU command state (handled directly in cache_read/write) */
  memset(&ct2d->gpu_cmd, 0, sizeof(ct2d->gpu_cmd));
  /* NOTE: Do NOT set CXL_GPU_STATUS_READY here - it will be set dynamically
   * in the register read handler based on hetgpu->initialized.
   * Setting it here unconditionally would make cuInit succeed even when
   * the GPU backend failed to initialize. */
  ct2d->gpu_cmd.status = 0;
  ct2d->gpu_cmd.cmd_status = CXL_GPU_CMD_STATUS_IDLE;

  ct2d->gpu_cmd.data_size = CXL_GPU_DATA_SIZE;
  ct2d->gpu_cmd.data = memory_region_get_ram_ptr(&ct2d->gpu_data_mem);
  ct2d->gpu_cmd.batch_data =
      memory_region_get_ram_ptr(&ct2d->gpu_batch_data_mem);
  ct2d->gpu_cmd.descriptor =
      memory_region_get_ram_ptr(&ct2d->gpu_descriptor_mem);
  memset(ct2d->gpu_cmd.descriptor, 0, CXL_GPU_DESCRIPTOR_WIRE_SIZE);
  ct2d->gpu_cmd.device_generation = 1;
  ct2d->gpu_cmd.descriptor->device_generation = 1;

  /* Set capabilities */
  ct2d->gpu_cmd.capabilities =
      CXL_GPU_CAP_BULK_TRANSFER | CXL_GPU_CAP_CACHE_COHERENT |
      CXL_GPU_CAP_COHERENT_POOL | CXL_GPU_CAP_DEVICE_BIAS;
  if (ct2d->dcd.enabled) {
    ct2d->gpu_cmd.capabilities |= CXL_GPU_CAP_DCD;
  }
  if (ct2d->gfam.enabled) {
    ct2d->gpu_cmd.capabilities |= CXL_GPU_CAP_GFAM;
  }
  if (ct2d->mhsld.enabled) {
    ct2d->gpu_cmd.capabilities |= CXL_GPU_CAP_MHSLD;
  }

  /* Initialize coherent shared memory pool at top of BAR4 */
  {
    uint64_t coh_pool_size = 256 * MiB; /* Default 256MB */
    if (coh_pool_size > ct2d->device_mem_size / 2) {
      coh_pool_size = ct2d->device_mem_size / 4;
    }
    ct2d->coherent_pool.size = coh_pool_size;
    ct2d->coherent_pool.base_offset = ct2d->device_mem_size - coh_pool_size;
    ct2d->coherent_pool.used = 0;
    ct2d->coherent_pool.allocations =
        g_hash_table_new_full(g_int64_hash, g_int64_equal, g_free, g_free);
    /* Initialize free list with single block spanning the whole pool */
    CXLCohFreeBlock *initial = g_new0(CXLCohFreeBlock, 1);
    initial->offset = ct2d->coherent_pool.base_offset;
    initial->size = coh_pool_size;
    initial->next = NULL;
    ct2d->coherent_pool.free_list = initial;
    qemu_mutex_init(&ct2d->coherent_pool.lock);

    memory_region_init_ram(&ct2d->coherent_pool_mem, OBJECT(ct2d),
                           "cxl-type2-coherent-pool", coh_pool_size,
                           &local_err);
    if (local_err) {
      error_propagate(errp, local_err);
      return;
    }
    memory_region_add_subregion_overlap(&ct2d->device_mem,
                                        ct2d->coherent_pool.base_offset,
                                        &ct2d->coherent_pool_mem, 2);

    qemu_log("CXL Type2: Coherent pool initialized: base=0x%lx size=%lu MB\n",
             (unsigned long)ct2d->coherent_pool.base_offset,
             (unsigned long)(coh_pool_size / MiB));
  }

  qemu_log("CXL Type2: GPU command interface enabled at BAR2 offset 0\n");
  qemu_log(
      "CXL Type2: Data buffer size: %zu KB (optimized for large transfers)\n",
      ct2d->gpu_cmd.data_size / 1024);

  /* Initialize MSI-X */
  if (msix_init_exclusive_bar(pci_dev, 16, 6, NULL)) {
    error_setg(errp, "Failed to initialize MSI-X");
    return;
  }

  /* Initialize GPU passthrough */
  if (cxl_type2_gpu_init(ct2d, &local_err) < 0) {
    if (local_err) {
      error_propagate(errp, local_err);
      return;
    }
  }

  /* Connect to CXLMemSim. The TCP server protocol is request/response. */
  cxlmemsim_connect(ct2d);

  if (ct2d->model_alias_gate.output) {
    if (!cxl_type2_model_alias_gate_run(ct2d, errp)) {
      return;
    }
    qemu_system_shutdown_request(SHUTDOWN_CAUSE_GUEST_SHUTDOWN);
  }

  qemu_log("CXL Type2: Device realized - Cache: %zu MB, DevMem: %zu MB\n",
           ct2d->cache_size / MiB, ct2d->device_mem_size / MiB);
}

static void cxl_type2_exit(PCIDevice *pci_dev) {
  CXLType2State *ct2d = CXL_TYPE2(pci_dev);

  /* Disconnect from CXLMemSim */
  cxlmemsim_disconnect(ct2d);
  qemu_mutex_destroy(&ct2d->memsim.lock);

  if (ct2d->gpu_info.hetgpu_state.initialized) {
    (void)hetgpu_synchronize(&ct2d->gpu_info.hetgpu_state);
    (void)cxl_type2_release_pending_htod(ct2d, NULL, true, UINT64_MAX,
                                         "device-exit", NULL);
    (void)cxl_type2_release_pending_alias_consumers(ct2d, NULL, true,
                                                    UINT64_MAX, NULL);
    (void)cxl_type2_clear_htod_staging_pool(ct2d);
    if (cxl_type2_direct_sources_cleanup(ct2d) != CXL_GPU_SUCCESS) {
      error_report("CXL Type2 direct source cleanup failed at device exit");
    }
    cxl_coherent_pool_cleanup_mappings(ct2d);
  }
  cxl_type2_direct_indexes_destroy(ct2d);
  cxl_type2_cuda_allocation_table_destroy(&ct2d->cuda_allocations);
  (void)cxl_type2_clear_gpu_handles(ct2d, 0, CXL_GPU_CASE_NONE, 0);

  /* Cleanup GPU passthrough */
  cxl_type2_gpu_cleanup(ct2d);

  /* Cleanup coherency protocol */
  cxl_type2_coherency_cleanup(ct2d);

  /* Cleanup enhanced BAR coherency tracking */
  cxl_bar_coherency_cleanup(&ct2d->bar_coherency);

  /* Cleanup P2P DMA engine */
  cxl_p2p_dma_cleanup(&ct2d->p2p_engine);

  /* Cleanup coherent pool */
  if (ct2d->coherent_pool.allocations) {
    g_hash_table_destroy(ct2d->coherent_pool.allocations);
    ct2d->coherent_pool.allocations = NULL;
  }
  {
    CXLCohFreeBlock *blk = ct2d->coherent_pool.free_list;
    while (blk) {
      CXLCohFreeBlock *next = blk->next;
      g_free(blk);
      blk = next;
    }
    ct2d->coherent_pool.free_list = NULL;
  }
  qemu_mutex_destroy(&ct2d->coherent_pool.lock);

  ct2d->gpu_cmd.data = NULL;
  ct2d->gpu_cmd.batch_data = NULL;

  /* Free bulk transfer region if allocated */
  if (ct2d->bulk_transfer_ptr) {
    g_free(ct2d->bulk_transfer_ptr);
    ct2d->bulk_transfer_ptr = NULL;
  }

  cxl_type2_fabric_features_cleanup(ct2d);
  cxl_type2_model_member_manifest_clear(&ct2d->model_supply.member_manifest);
  cxl_type2_model_consumer_certificate_clear(
      &ct2d->model_supply.consumer_certificate);
  g_clear_pointer(&ct2d->model_supply.source_geometry, g_checksum_free);
  g_clear_pointer(&ct2d->model_supply.alias_geometry, g_checksum_free);
  g_clear_pointer(&ct2d->model_supply.first_failure_stage, g_free);
  g_clear_pointer(&ct2d->model_supply.first_failure_detail, g_free);

  qemu_log("CXL Type2: Device exit complete\n");
}

static const Property cxl_type2_props[] = {
    DEFINE_PROP_SIZE("cache-size", CXLType2State, cache_size,
                     CXL_TYPE2_DEFAULT_CACHE_SIZE),
    DEFINE_PROP_SIZE("mem-size", CXLType2State, device_mem_size,
                     CXL_TYPE2_DEFAULT_MEM_SIZE),
    DEFINE_PROP_SIZE("htod-staging-pool-size", CXLType2State,
                     htod_staging_pool_size, 0),
    DEFINE_PROP_BOOL("cuda-direct-source", CXLType2State, cuda_direct_source,
                     false),
    DEFINE_PROP_SIZE("cuda-direct-registration-tile-size", CXLType2State,
                     direct_registration_tile_size, 0),
    DEFINE_PROP_SIZE("cuda-direct-registration-padding-size", CXLType2State,
                     direct_registration_padding_limit, 0),
    DEFINE_PROP_LINK("direct-source-fs", CXLType2State, direct_source_fs,
                     TYPE_VHOST_USER_FS_PCI, Object *),
    DEFINE_PROP_STRING("model-supply-route", CXLType2State,
                       model_supply.route_text),
    DEFINE_PROP_STRING("model-member-manifest", CXLType2State,
                       model_supply.member_manifest_path),
    DEFINE_PROP_STRING("model-member-manifest-sha256", CXLType2State,
                       model_supply.member_manifest_sha256),
    DEFINE_PROP_STRING("model-consumer-certificate", CXLType2State,
                       model_supply.consumer_certificate_path),
    DEFINE_PROP_STRING("model-consumer-certificate-sha256", CXLType2State,
                       model_supply.consumer_certificate_sha256),
    DEFINE_PROP_SIZE("model-aperture-offset", CXLType2State,
                     model_aperture.offset, 0),
    DEFINE_PROP_SIZE("model-aperture-size", CXLType2State, model_aperture.size,
                     0),
    DEFINE_PROP_STRING("model-alias-gate-output", CXLType2State,
                       model_alias_gate.output),
    DEFINE_PROP_STRING("model-alias-gate-fixture", CXLType2State,
                       model_alias_gate.fixture),
    DEFINE_PROP_STRING("model-alias-gate-mode", CXLType2State,
                       model_alias_gate.mode),
    DEFINE_PROP_SIZE("model-alias-gate-file-offset", CXLType2State,
                     model_alias_gate.file_offset, 0),
    DEFINE_PROP_SIZE("model-alias-gate-shmem-offset", CXLType2State,
                     model_alias_gate.shmem_offset, 0),
    DEFINE_PROP_SIZE("model-alias-gate-length", CXLType2State,
                     model_alias_gate.length, 0),
    DEFINE_PROP_UINT64("sn", CXLType2State, sn, 0),
    DEFINE_PROP_PCIE_LINK_SPEED("x-speed", CXLType2State, speed,
                                PCIE_LINK_SPEED_64),
    DEFINE_PROP_PCIE_LINK_WIDTH("x-width", CXLType2State, width,
                                PCIE_LINK_WIDTH_32),
    DEFINE_PROP_BOOL("x-256b-flit", CXLType2State, flitmode, true),
    DEFINE_PROP_BOOL("hdm-db", CXLType2State, hdmdb, true),
    DEFINE_PROP_STRING("cxlmemsim-addr", CXLType2State, memsim.server_addr),
    DEFINE_PROP_UINT16("cxlmemsim-port", CXLType2State, memsim.server_port,
                       9999),
    DEFINE_PROP_STRING("gpu-device", CXLType2State, gpu_info.vfio_device),
    DEFINE_PROP_BOOL("coherency-enabled", CXLType2State,
                     coherency.coherency_enabled, true),
    /* hetGPU backend configuration */
    DEFINE_PROP_UINT32("gpu-mode", CXLType2State, gpu_info.mode,
                       CXL_TYPE2_GPU_MODE_AUTO),
    DEFINE_PROP_STRING("hetgpu-lib", CXLType2State, gpu_info.hetgpu_lib_path),
    DEFINE_PROP_INT32("hetgpu-device", CXLType2State,
                      gpu_info.hetgpu_device_index, 0),
    DEFINE_PROP_UINT32("hetgpu-backend", CXLType2State, gpu_info.hetgpu_backend,
                       HETGPU_BACKEND_AUTO),
    DEFINE_PROP_BOOL("paired-case-control", CXLType2State, paired_case.required,
                     false),
    DEFINE_PROP_STRING("paired-run-root", CXLType2State, paired_case.run_root),
    DEFINE_PROP_UINT64("paired-run-binding", CXLType2State,
                       paired_case.run_binding, 0),
    DEFINE_PROP_UINT64("paired-kimi-min-allocation-bytes", CXLType2State,
                       paired_case.min_allocation_bytes, 1ULL << 20),
    DEFINE_PROP_UINT64("paired-kimi-max-regions", CXLType2State,
                       paired_case.max_regions, 128),
    DEFINE_PROP_UINT64("paired-kimi-checkpoint-every", CXLType2State,
                       paired_case.checkpoint_every_launches, 0),
    DEFINE_PROP_BOOL("paired-kimi-checkpoint", CXLType2State,
                     paired_case.checkpoint_enabled, false),
    DEFINE_PROP_BOOL("paired-kimi-qemu-cuda-calls", CXLType2State,
                     paired_case.qemu_cuda_calls_enabled, false),
    DEFINE_PROP_BOOL("paired-kimi-concordia-runtime-details", CXLType2State,
                     paired_case.concordia_runtime_details_enabled, false),
    DEFINE_PROP_BOOL("dcd", CXLType2State, dcd.enabled, false),
    DEFINE_PROP_SIZE("dcd-granularity", CXLType2State, dcd.granularity,
                     CXL_TYPE2_DCD_DEFAULT_GRANULARITY),
    DEFINE_PROP_SIZE("dcd-initial-size", CXLType2State, dcd.initial_size,
                     CXL_TYPE2_DCD_INIT_AUTO),
    DEFINE_PROP_BOOL("gfam", CXLType2State, gfam.enabled, false),
    DEFINE_PROP_UINT32("gfam-hosts", CXLType2State, gfam.num_hosts, 1),
    DEFINE_PROP_UINT32("gfam-host-id", CXLType2State, gfam.local_host_id, 0),
    DEFINE_PROP_UINT32("gfam-default-perms", CXLType2State,
                       gfam.default_permissions, CXL_DCD_PERM_ALL),
    DEFINE_PROP_UINT32("gfam-latency-ns", CXLType2State, gfam.fabric_latency_ns,
                       150),
    DEFINE_PROP_UINT32("gfam-bandwidth-mbps", CXLType2State,
                       gfam.bandwidth_mbps, 32768),
    DEFINE_PROP_BOOL("mhsld", CXLType2State, mhsld.enabled, false),
    DEFINE_PROP_UINT32("mhsld-heads", CXLType2State, mhsld.num_heads, 1),
    DEFINE_PROP_UINT32("mhsld-head-id", CXLType2State, mhsld.local_head_id, 0),
    DEFINE_PROP_UINT32("mhsld-coh-latency-ns", CXLType2State,
                       mhsld.coherency_latency_ns, 200),
};

static const VMStateDescription cxl_type2_vmstate = {
    .name = TYPE_CXL_TYPE2,
    .unmigratable = 1,
};

static void cxl_type2_class_init(ObjectClass *oc, const void *data) {
  DeviceClass *dc = DEVICE_CLASS(oc);
  PCIDeviceClass *pc = PCI_DEVICE_CLASS(oc);

  pc->realize = cxl_type2_realize;
  pc->exit = cxl_type2_exit;
  pc->vendor_id = CXL_TYPE2_VENDOR_ID;
  pc->device_id = CXL_TYPE2_DEVICE_ID;
  pc->revision = 1;
  pc->class_id = PCI_CLASS_MEMORY_CXL;

  dc->desc =
      "CXL Type 2 Accelerator Device with Coherent Memory (GPU Passthrough)";
  dc->vmsd = &cxl_type2_vmstate;
  device_class_set_legacy_reset(dc, cxl_type2_reset);
  device_class_set_props(dc, cxl_type2_props);
}

static const TypeInfo cxl_type2_info = {
    .name = TYPE_CXL_TYPE2,
    .parent = TYPE_PCI_DEVICE,
    .instance_size = sizeof(CXLType2State),
    .class_init = cxl_type2_class_init,
    .interfaces =
        (InterfaceInfo[]){{INTERFACE_PCIE_DEVICE}, {INTERFACE_CXL_DEVICE}, {}},
};

static void cxl_type2_register_types(void) {
  type_register_static(&cxl_type2_info);
}

type_init(cxl_type2_register_types)
