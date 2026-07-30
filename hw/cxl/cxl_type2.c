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
#include "qemu/cutils.h"
#include "qemu/error-report.h"
#include "qemu/log.h"
#include "qemu/module.h"
#include "qemu/range.h"
#include "qemu/rcu.h"
#include "qemu/sockets.h"
#include "qemu/timer.h"
#include "qemu/units.h"
#include "qapi/error.h"
#include "hw/cxl/cxl.h"
#include "hw/cxl/cxl_device.h"
#include "hw/cxl/cxl_component.h"
#include "hw/cxl/cxl_cdat.h"
#include "hw/cxl/cxl_pci.h"
#include "hw/cxl/cxl_type2.h"
#include "hw/cxl/cxl_hetgpu.h"
#include "hw/cxl/cxl_type2_gpu_cmd.h"
#include "hw/cxl/cxl_type2_cuda_contract.h"
#include "hw/cxl/cxl_type2_coherency.h"
#include "hw/pci/pci.h"
#include "hw/pci/pcie.h"
#include "hw/pci/msix.h"
#include "hw/qdev-properties.h"
#include "hw/qdev-properties-system.h"
#include "system/memory.h"
#include "io/channel-socket.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/eventfd.h>
#include <linux/vfio.h>
#include <unistd.h>
#include <zstd.h>

extern int LZ4_decompress_safe(const char *src, char *dst,
                               int compressed_size, int dst_capacity);

#ifndef DEFAULT_HETGPU_LIB_PATH
#define DEFAULT_HETGPU_LIB_PATH NULL
#endif

#define CXL_OP_READ         0
#define CXL_OP_WRITE        1
#define CXL_OP_FENCE        5
#define CXL_OP_BI_ENABLE    14
#define CXL_OP_BI_DISABLE   15
#define CXL_OP_BI_INVALIDATE 16
#define CXL_OP_BI_WRITEBACK 17
#define CXL_OP_BI_QUERY     18

/* A CXL.mem response is one fixed-size message. A missing response must
 * release the QEMU device path instead of consuming the outer CNB timeout. */
#define CXL_MEMSIM_RESPONSE_TIMEOUT_NS (1000LL * 1000 * 1000)

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

static bool cxl_type2_bulk_memsim_access(CXLType2State *ct2d, uint8_t op_type,
                                          uint64_t addr, size_t size,
                                          uint8_t *data)
{
    /* The wire protocol carries one cacheline, so a bulk CUDA copy is split
     * into bounded CXL.mem requests while the GPU transfer remains batched. */
    if (ct2d->memsim.use_shm) {
        /* SHM transport has no TCP request/response to acknowledge here. */
        return true;
    }
    if (!ct2d->memsim.connected) {
        return false;
    }
    while (size > 0) {
        size_t chunk = MIN(size, sizeof(((CXLMemSimRequest *)0)->data));
        CXLMemSimResponse response;
        const uint8_t *request_data = op_type == CXL_OP_WRITE ? data : NULL;

        if (op_type == CXL_OP_READ && !data) {
            return false;
        }
        if (!cxl_type2_memsim_request(ct2d, op_type, addr, chunk,
                                      request_data, &response)) {
            return false;
        }
        if (op_type == CXL_OP_READ) {
            memcpy(data, response.data, chunk);
        }
        addr += chunk;
        size -= chunk;
        if (data)
            data += chunk;
    }
    return true;
}

/* ========================================================================
 * Coherency Protocol Implementation
 * ======================================================================== */

void cxl_type2_coherency_init(CXLType2State *ct2d)
{
    qemu_mutex_init(&ct2d->coherency.lock);
    ct2d->coherency.cache_lines = g_hash_table_new_full(
        g_int64_hash, g_int64_equal, g_free, g_free);
    ct2d->coherency.cache_hits = 0;
    ct2d->coherency.cache_misses = 0;
    ct2d->coherency.coherency_ops = 0;
    ct2d->coherency.snoops = 0;
    ct2d->coherency.coherency_enabled = true;

    qemu_log("CXL Type2: Coherency protocol initialized\n");
}

void cxl_type2_coherency_cleanup(CXLType2State *ct2d)
{
    if (ct2d->coherency.cache_lines) {
        g_hash_table_destroy(ct2d->coherency.cache_lines);
        ct2d->coherency.cache_lines = NULL;
    }
    qemu_mutex_destroy(&ct2d->coherency.lock);

    qemu_log("CXL Type2: Coherency stats - Hits: %lu, Misses: %lu, Ops: %lu, Snoops: %lu\n",
             ct2d->coherency.cache_hits, ct2d->coherency.cache_misses,
             ct2d->coherency.coherency_ops, ct2d->coherency.snoops);
}

CXLCacheLine *cxl_type2_cache_lookup(CXLType2State *ct2d, uint64_t addr)
{
    /* Align to cache line boundary */
    uint64_t cache_line_addr = addr & ~0x3F;
    CXLCacheLine *line;

    qemu_mutex_lock(&ct2d->coherency.lock);
    line = g_hash_table_lookup(ct2d->coherency.cache_lines,
                               &cache_line_addr);

    if (line) {
        ct2d->coherency.cache_hits++;
    } else {
        ct2d->coherency.cache_misses++;
    }
    qemu_mutex_unlock(&ct2d->coherency.lock);

    return line;
}

void cxl_type2_cache_insert(CXLType2State *ct2d, uint64_t addr,
                            const uint8_t *data, CXLCoherencyState state)
{
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

    qemu_log_mask(LOG_TRACE, "CXL Type2: Cache insert at 0x%lx, state=%d\n",
                 cache_line_addr, state);
}

void cxl_type2_cache_invalidate(CXLType2State *ct2d, uint64_t addr)
{
    uint64_t cache_line_addr = addr & ~0x3F;

    qemu_mutex_lock(&ct2d->coherency.lock);
    if (g_hash_table_remove(ct2d->coherency.cache_lines, &cache_line_addr)) {
        ct2d->coherency.coherency_ops++;
        qemu_log_mask(LOG_TRACE, "CXL Type2: Cache invalidate at 0x%lx\n",
                     cache_line_addr);
    }
    qemu_mutex_unlock(&ct2d->coherency.lock);

    if (ct2d->bi_enabled) {
        cxl_type2_memsim_request(ct2d, CXL_OP_BI_INVALIDATE,
                                 cache_line_addr, 64, NULL, NULL);
    }
}

void cxl_type2_cache_writeback(CXLType2State *ct2d, uint64_t addr)
{
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
            cxl_type2_memsim_request(ct2d, CXL_OP_BI_WRITEBACK,
                                     cache_line_addr, 64, line->data, NULL);
        } else {
            cxl_type2_memsim_request(ct2d, CXL_OP_WRITE, cache_line_addr,
                                     64, line->data, NULL);
        }

        line->dirty = false;
        ct2d->coherency.coherency_ops++;

        qemu_log_mask(LOG_TRACE, "CXL Type2: Cache writeback at 0x%lx\n",
                     cache_line_addr);
    }
    qemu_mutex_unlock(&ct2d->coherency.lock);
}

static bool cxl_type2_cache_fill_line(CXLType2State *ct2d, uint64_t cache_line_addr,
                                      uint8_t cache_data[64])
{
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
                                     uint64_t size, bool write_intent)
{
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
            CXLCoherencyState state = write_intent ?
                CXL_COHERENCY_EXCLUSIVE : CXL_COHERENCY_SHARED;

            if (cxl_type2_cache_fill_line(ct2d, line_addr, cache_data)) {
                cxl_type2_cache_insert(ct2d, line_addr, cache_data, state);
                cxl_type2_memsim_request(ct2d, CXL_OP_READ, line_addr, 64,
                                         NULL, NULL);
                ct2d->stats.read_ops++;
            }
        }

        if (line_addr == last || line_addr > UINT64_MAX - 64) {
            break;
        }
    }

    ct2d->coherency.coherency_ops++;

    qemu_log_mask(LOG_TRACE,
                  "CXL Type2: Cache prefetch addr=0x%" PRIx64
                  " size=0x%" PRIx64 " write_intent=%d\n",
                  addr, size, write_intent);
}

bool cxl_type2_snoop_request(CXLType2State *ct2d, uint64_t addr, bool invalidate)
{
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

    qemu_log_mask(LOG_TRACE, "CXL Type2: Snoop request at 0x%lx, hit=%d, invalidate=%d\n",
                 cache_line_addr, hit, invalidate);

    return hit;
}

/* ========================================================================
 * GPU Passthrough Implementation - VFIO Helpers
 * ======================================================================== */

/* Open and setup VFIO container */
static int cxl_type2_vfio_container_init(CXLType2State *ct2d, Error **errp)
{
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
                                      Error **errp)
{
    char group_path[256];
    char iommu_group_path[512];
    char *group_name;
    int group_fd, container_fd;
    ssize_t len;
    struct vfio_group_status group_status = { .argsz = sizeof(group_status) };

    /* Get IOMMU group for the device */
    snprintf(iommu_group_path, sizeof(iommu_group_path),
             "/sys/bus/pci/devices/%s/iommu_group", pci_addr);

    len = readlink(iommu_group_path, group_path, sizeof(group_path) - 1);
    if (len < 0) {
        error_setg(errp, "Failed to read IOMMU group for %s: %s",
                   pci_addr, strerror(errno));
        return -1;
    }
    group_path[len] = '\0';

    /* Extract group number from path */
    group_name = basename(group_path);

    /* Open VFIO group */
    snprintf(group_path, sizeof(group_path), "/dev/vfio/%s", group_name);
    group_fd = open(group_path, O_RDWR);
    if (group_fd < 0) {
        error_setg(errp, "Failed to open VFIO group %s: %s",
                   group_path, strerror(errno));
        return -1;
    }

    /* Check group is viable */
    if (ioctl(group_fd, VFIO_GROUP_GET_STATUS, &group_status) < 0) {
        error_setg(errp, "Failed to get VFIO group status: %s", strerror(errno));
        close(group_fd);
        return -1;
    }

    if (!(group_status.flags & VFIO_GROUP_FLAGS_VIABLE)) {
        error_setg(errp, "VFIO group is not viable (not all devices bound to VFIO)");
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
    qemu_log("CXL Type2: VFIO group %s initialized (fd=%d)\n", group_name, group_fd);

    return group_fd;
}

/* Open and setup VFIO device */
static int cxl_type2_vfio_device_init(CXLType2State *ct2d, const char *pci_addr,
                                       Error **errp)
{
    int group_fd, device_fd;
    struct vfio_device_info device_info = { .argsz = sizeof(device_info) };
    struct vfio_region_info region_info = { .argsz = sizeof(region_info) };

    group_fd = GPOINTER_TO_INT(ct2d->gpu_info.vfio_group);

    /* Get device FD */
    device_fd = ioctl(group_fd, VFIO_GROUP_GET_DEVICE_FD, pci_addr);
    if (device_fd < 0) {
        error_setg(errp, "Failed to get device FD for %s: %s",
                   pci_addr, strerror(errno));
        return -1;
    }

    /* Get device info */
    if (ioctl(device_fd, VFIO_DEVICE_GET_INFO, &device_info) < 0) {
        error_setg(errp, "Failed to get device info: %s", strerror(errno));
        close(device_fd);
        return -1;
    }

    qemu_log("CXL Type2: GPU device %s has %d regions, %d IRQs\n",
             pci_addr, device_info.num_regions, device_info.num_irqs);

    /* Get BAR0 region info (GPU memory typically in BAR0 or BAR1) */
    region_info.index = VFIO_PCI_BAR0_REGION_INDEX;
    if (ioctl(device_fd, VFIO_DEVICE_GET_REGION_INFO, &region_info) < 0) {
        error_setg(errp, "Failed to get BAR0 region info: %s", strerror(errno));
        close(device_fd);
        return -1;
    }

    /* Map GPU memory region */
    if (region_info.size > 0) {
        void *bar_mem = mmap(NULL, region_info.size,
                            PROT_READ | PROT_WRITE,
                            MAP_SHARED,
                            device_fd,
                            region_info.offset);

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

/* Setup DMA mapping for GPU coherent access */
static int cxl_type2_vfio_dma_map(CXLType2State *ct2d, Error **errp)
{
    int container_fd;
    struct vfio_iommu_type1_dma_map dma_map = {
        .argsz = sizeof(dma_map),
        .flags = VFIO_DMA_MAP_FLAG_READ | VFIO_DMA_MAP_FLAG_WRITE,
        .vaddr = 0,  /* Will be set to device memory */
        .iova = 0,   /* IO virtual address */
        .size = 0,   /* Will be set to device memory size */
    };

    container_fd = GPOINTER_TO_INT(ct2d->gpu_info.vfio_container);

    /* Get device memory pointer */
    uint8_t *mem_ptr = memory_region_get_ram_ptr(&ct2d->device_mem);
    if (!mem_ptr) {
        error_setg(errp, "Failed to get device memory pointer");
        return -1;
    }

    dma_map.vaddr = (uint64_t)mem_ptr;
    dma_map.iova = 0;  /* Map at IOVA 0 */
    dma_map.size = ct2d->device_mem_size;

    /* Map the device memory for DMA */
    if (ioctl(container_fd, VFIO_IOMMU_MAP_DMA, &dma_map) < 0) {
        error_setg(errp, "Failed to map DMA region: %s", strerror(errno));
        return -1;
    }

    qemu_log("CXL Type2: DMA mapping configured - IOVA: 0x%llx, Size: %llu MB\n",
             dma_map.iova, dma_map.size / (1024 * 1024));

    return 0;
}

/* Interrupt handling thread for GPU events */
static void *cxl_type2_irq_thread(void *opaque)
{
    CXLType2State *ct2d = opaque;
    PCIDevice *pci_dev = PCI_DEVICE(ct2d);
    int device_fd = ct2d->gpu_info.vfio_device_fd;
    struct vfio_irq_info irq_info = { .argsz = sizeof(irq_info) };
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
            qemu_log("CXL Type2: Failed to create event FD %d: %s\n", i, strerror(errno));
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

        struct timeval tv = { .tv_sec = 1, .tv_usec = 0 };
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
                    qemu_log_mask(LOG_TRACE, "CXL Type2: GPU interrupt %d triggered\n", i);

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
                                                 uint64_t size, bool invalidate)
{
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

int cxl_type2_hetgpu_init(CXLType2State *ct2d, Error **errp)
{
    HetGPUState *hetgpu = &ct2d->gpu_info.hetgpu_state;
    HetGPUError err;
    const char *lib_path;

    fprintf(stderr, "cxl_type2_hetgpu_init: ENTERED\n");
    fflush(stderr);

    /* Determine library path - try multiple locations */
    lib_path = ct2d->gpu_info.hetgpu_lib_path;
    fprintf(stderr, "cxl_type2_hetgpu_init: hetgpu_lib_path = '%s'\n", lib_path ? lib_path : "(null)");
    fflush(stderr);
    if (!lib_path || lib_path[0] == '\0') {
        lib_path = getenv("HETGPU_LIB_PATH");
        fprintf(stderr, "cxl_type2_hetgpu_init: env HETGPU_LIB_PATH = '%s'\n", lib_path ? lib_path : "(null)");
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
    err = ct2d->paired_case.required ?
          hetgpu_init_formal(hetgpu,
                             ct2d->gpu_info.hetgpu_backend,
                             ct2d->gpu_info.hetgpu_device_index,
                             lib_path) :
          hetgpu_init(hetgpu,
                      ct2d->gpu_info.hetgpu_backend,
                      ct2d->gpu_info.hetgpu_device_index,
                      lib_path);
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
    hetgpu_set_coherency_callback(hetgpu,
                                   cxl_type2_hetgpu_coherency_callback,
                                   ct2d);

    ct2d->gpu_info.passthrough_enabled = true;
    ct2d->gpu_info.gpu_mem_size = hetgpu->props.total_memory;

    qemu_log("CXL Type2: hetGPU initialized - Backend: %s, Device: %s\n",
             hetgpu_get_backend_name(hetgpu->backend),
             hetgpu->props.name);
    qemu_log("CXL Type2: GPU Memory: %lu MB, Compute: %d.%d\n",
             hetgpu->props.total_memory / (1024 * 1024),
             hetgpu->props.compute_capability_major,
             hetgpu->props.compute_capability_minor);

    return 0;
}

void cxl_type2_hetgpu_cleanup(CXLType2State *ct2d)
{
    HetGPUState *hetgpu = &ct2d->gpu_info.hetgpu_state;

    if (!hetgpu->initialized) {
        return;
    }

    qemu_log("CXL Type2: Cleaning up hetGPU backend\n");
    hetgpu_cleanup(hetgpu);
}

int cxl_type2_hetgpu_load_ptx(CXLType2State *ct2d, const char *ptx_source,
                               void **module)
{
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
                                    uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                                    uint32_t block_x, uint32_t block_y, uint32_t block_z,
                                    uint32_t shared_mem, void **args, size_t num_args)
{
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

int cxl_type2_hetgpu_malloc(CXLType2State *ct2d, size_t size, uint64_t *dev_ptr)
{
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

int cxl_type2_hetgpu_free(CXLType2State *ct2d, uint64_t dev_ptr)
{
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
                                  const void *src, size_t size)
{
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

int cxl_type2_hetgpu_memcpy_dtoh(CXLType2State *ct2d, void *dst,
                                  uint64_t src, size_t size)
{
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

int cxl_type2_hetgpu_sync(CXLType2State *ct2d)
{
    HetGPUState *hetgpu = &ct2d->gpu_info.hetgpu_state;
    HetGPUError err;

    if (!hetgpu->initialized) {
        return -1;
    }

    err = hetgpu_synchronize(hetgpu);
    if (err != HETGPU_SUCCESS) {
        qemu_log("CXL Type2: Sync failed: %s\n",
                 hetgpu_get_error_string(err));
        return -1;
    }

    return 0;
}

/* ========================================================================
 * GPU Passthrough Implementation - Main Functions
 * ======================================================================== */

int cxl_type2_gpu_init(CXLType2State *ct2d, Error **errp)
{
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
    fprintf(stderr, "CXL Type2: GPU mode = %d (NONE=0, VFIO=1, HETGPU=2, AUTO=3)\n", ct2d->gpu_info.mode);
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
        if (ct2d->paired_case.required) {
            qemu_log("CXL Type2: formal paired mode rejects hetGPU "
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
    ret = cxl_type2_vfio_device_init(ct2d, ct2d->gpu_info.vfio_device, &local_err);
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

void cxl_type2_gpu_cleanup(CXLType2State *ct2d)
{
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
        ct2d->gpu_info.passthrough_enabled = false;  /* Signal thread to exit */
        qemu_thread_join(&ct2d->gpu_info.irq_thread);
        qemu_log("CXL Type2: IRQ thread stopped\n");
    }

    /* Unmap DMA regions */
    if (ct2d->gpu_info.vfio_container) {
        int container_fd = GPOINTER_TO_INT(ct2d->gpu_info.vfio_container);
        struct vfio_iommu_type1_dma_unmap dma_unmap = {
            .argsz = sizeof(dma_unmap),
            .flags = 0,
            .iova = 0,
            .size = ct2d->device_mem_size,
        };

        if (ioctl(container_fd, VFIO_IOMMU_UNMAP_DMA, &dma_unmap) < 0) {
            qemu_log("CXL Type2: Warning - Failed to unmap DMA: %s\n", strerror(errno));
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

int cxl_type2_gpu_read(CXLType2State *ct2d, uint64_t offset, void *buf, size_t size)
{
    ssize_t ret;

    if (!ct2d->gpu_info.passthrough_enabled) {
        return -1;
    }

    /* If VFIO device is configured, read from actual GPU BAR */
    if (ct2d->gpu_info.vfio_device_fd >= 0) {
        uint64_t bar_offset = ct2d->gpu_info.gpu_mem_base + offset;

        ret = pread(ct2d->gpu_info.vfio_device_fd, buf, size, bar_offset);
        if (ret < 0) {
            qemu_log("CXL Type2: GPU read failed at offset 0x%lx: %s\n",
                     offset, strerror(errno));
            return -1;
        }

        if (ret != size) {
            qemu_log("CXL Type2: GPU read incomplete at offset 0x%lx: %zd/%zu bytes\n",
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

int cxl_type2_gpu_write(CXLType2State *ct2d, uint64_t offset, const void *buf, size_t size)
{
    ssize_t ret;

    if (!ct2d->gpu_info.passthrough_enabled) {
        return -1;
    }

    /* If VFIO device is configured, write to actual GPU BAR */
    if (ct2d->gpu_info.vfio_device_fd >= 0) {
        uint64_t bar_offset = ct2d->gpu_info.gpu_mem_base + offset;

        ret = pwrite(ct2d->gpu_info.vfio_device_fd, buf, size, bar_offset);
        if (ret < 0) {
            qemu_log("CXL Type2: GPU write failed at offset 0x%lx: %s\n",
                     offset, strerror(errno));
            return -1;
        }

        if (ret != size) {
            qemu_log("CXL Type2: GPU write incomplete at offset 0x%lx: %zd/%zu bytes\n",
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

static void cxlmemsim_drop_locked(CXLType2State *ct2d)
{
    ct2d->memsim.connected = false;

    if (ct2d->memsim.socket) {
        qio_channel_close(QIO_CHANNEL(ct2d->memsim.socket), NULL);
        object_unref(OBJECT(ct2d->memsim.socket));
        ct2d->memsim.socket = NULL;
    }
}

static void cxlmemsim_connect(CXLType2State *ct2d)
{
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

    qemu_log("CXL Type2: Transport mode = %s\n", transport_mode ? transport_mode : "(not set)");

    if (transport_mode && (strcmp(transport_mode, "shm") == 0 ||
                           strcmp(transport_mode, "pgas") == 0)) {
        ct2d->memsim.use_shm = true;
        qemu_log("CXL Type2: Using shared memory transport - skipping TCP connection\n");
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

static void cxlmemsim_disconnect(CXLType2State *ct2d)
{
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
                                           Error **errp)
{
    QIOChannelSocket *socket = ct2d->memsim.socket;
    uint8_t *cursor = (uint8_t *)resp;
    size_t remaining = sizeof(*resp);
    int64_t deadline = g_get_monotonic_time() * 1000 +
                       CXL_MEMSIM_RESPONSE_TIMEOUT_NS;

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
            error_setg_errno(errp, errno,
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
            error_setg_errno(errp, errno,
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

static bool cxl_type2_memsim_request_ext(CXLType2State *ct2d, uint8_t op_type,
                                         uint64_t addr, uint64_t size,
                                         const uint8_t *data, uint64_t value,
                                         uint64_t expected,
                                         CXLMemSimResponse *resp)
{
    CXLMemSimRequest req;
    CXLMemSimResponse local_resp;
    Error *err = NULL;
    bool ok = false;

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

    if (qio_channel_write_all(QIO_CHANNEL(ct2d->memsim.socket),
                              (const char *)&req, sizeof(req), &err) < 0) {
        error_report("CXL Type2: Failed to send request to CXLMemSim: %s",
                     error_get_pretty(err));
        error_free(err);
        cxlmemsim_drop_locked(ct2d);
        goto out;
    }

    if (!cxl_type2_memsim_read_response(ct2d, &req, &local_resp, &err)) {
        error_report("CXL Type2: Failed to receive response from CXLMemSim: %s",
                     error_get_pretty(err));
        error_free(err);
        cxlmemsim_drop_locked(ct2d);
        goto out;
    }

    if (resp) {
        *resp = local_resp;
    }
    ok = local_resp.status == 0;

out:
    qemu_mutex_unlock(&ct2d->memsim.lock);
    return ok;
}

static bool cxl_type2_memsim_request(CXLType2State *ct2d, uint8_t op_type,
                                     uint64_t addr, uint64_t size,
                                     const uint8_t *data,
                                     CXLMemSimResponse *resp)
{
    return cxl_type2_memsim_request_ext(ct2d, op_type, addr, size, data, 0, 0,
                                        resp);
}

/* ========================================================================
 * Memory Access Handlers with Coherency
 * ======================================================================== */

/* Forward declaration for GPU command handler */
static uint64_t cxl_type2_gpu_cmd_read(void *opaque, hwaddr addr, unsigned size);
static void cxl_type2_gpu_cmd_write(void *opaque, hwaddr addr, uint64_t value, unsigned size);

static uint64_t cxl_type2_cache_read(void *opaque, hwaddr addr, unsigned size)
{
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
        cxl_bar_coherency_request(&ct2d->bar_coherency,
                                  CXL_COH_REQ_RD_SHARED,
                                  addr, size,
                                  CXL_DOMAIN_CPU, NULL);
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

static void cxl_type2_cache_write(void *opaque, hwaddr addr, uint64_t value, unsigned size)
{
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
        cxl_bar_coherency_request(&ct2d->bar_coherency,
                                  CXL_COH_REQ_WR_INV,
                                  addr, size,
                                  CXL_DOMAIN_CPU, NULL);
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

    qemu_log_mask(LOG_TRACE, "CXL Type2: Cache write at 0x%lx = 0x%lx\n",
                 addr, value);
}

static bool cxl_type2_fabric_access_allowed(CXLType2State *ct2d, uint64_t addr,
                                            uint64_t size, bool is_write,
                                            bool is_atomic);

static uint64_t cxl_type2_device_mem_read(void *opaque, hwaddr addr, unsigned size)
{
    CXLType2State *ct2d = opaque;

    if (!cxl_type2_fabric_access_allowed(ct2d, addr, size, false, false)) {
        return 0;
    }

    /* Forward all device memory reads through the cache coherency layer */
    return cxl_type2_cache_read(opaque, addr, size);
}

static void cxl_type2_device_mem_write(void *opaque, hwaddr addr, uint64_t value, unsigned size)
{
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
    .valid = {
        .min_access_size = 1,
        .max_access_size = 8,
    },
    .impl = {
        .min_access_size = 1,
        .max_access_size = 8,
    },
};

static const MemoryRegionOps cxl_type2_device_mem_ops = {
    .read = cxl_type2_device_mem_read,
    .write = cxl_type2_device_mem_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = {
        .min_access_size = 1,
        .max_access_size = 8,
    },
    .impl = {
        .min_access_size = 1,
        .max_access_size = 8,
    },
};

/* ========================================================================
 * DCD / GFAM / MH-SLD Fabric Memory Models
 * ======================================================================== */

static uint64_t cxl_type2_dcd_align(CXLType2State *ct2d, uint64_t value)
{
    uint64_t granularity = ct2d->dcd.granularity;

    if (!granularity) {
        granularity = CXL_TYPE2_DCD_DEFAULT_GRANULARITY;
    }

    return (value + granularity - 1) & ~(granularity - 1);
}

static bool cxl_type2_range_valid(CXLType2State *ct2d, uint64_t base,
                                  uint64_t size)
{
    return size && base < ct2d->device_mem_size &&
           size <= ct2d->device_mem_size - base;
}

static bool cxl_type2_dcd_overlaps_locked(CXLType2State *ct2d,
                                          uint64_t base, uint64_t size)
{
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

static bool cxl_type2_dcd_allocated_locked(CXLType2State *ct2d,
                                           uint64_t base, uint64_t size)
{
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
                                    uint64_t size)
{
    bool allocated;

    qemu_mutex_lock(&ct2d->dcd.lock);
    allocated = cxl_type2_dcd_allocated_locked(ct2d, base, size);
    qemu_mutex_unlock(&ct2d->dcd.lock);

    return allocated;
}

static uint64_t cxl_type2_dcd_active_extents_locked(CXLType2State *ct2d)
{
    uint64_t count = 0;

    for (int i = 0; i < CXL_TYPE2_MAX_DCD_EXTENTS; i++) {
        if (ct2d->dcd.extents[i].active) {
            count++;
        }
    }

    return count;
}

static bool cxl_type2_dcd_find_free_locked(CXLType2State *ct2d, uint64_t size,
                                           uint64_t *base)
{
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
                candidate = cxl_type2_dcd_align(ct2d,
                                                extent->base + extent->size);
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
                             uint64_t *out_size, uint64_t *out_tag)
{
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

    ct2d->dcd.extents[slot] = (CXLType2DCDExtent) {
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
             " tag=%" PRIu64 "\n", base, aligned_size, tag);
    return 0;
}

static int cxl_type2_dcd_release(CXLType2State *ct2d, uint64_t base,
                                 uint64_t size, uint64_t tag)
{
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

        qemu_log("CXL Type2 DCD: release base=0x%" PRIx64
                 " size=0x%" PRIx64 " tag=%" PRIu64 "\n",
                 base, aligned_size, tag);
        return 0;
    }

    ct2d->dcd.failed_requests++;
    qemu_mutex_unlock(&ct2d->dcd.lock);
    return -ENOENT;
}

static void cxl_type2_gfam_revoke_range(CXLType2State *ct2d, uint64_t base,
                                        uint64_t size)
{
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
                                uint32_t permissions)
{
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

    ct2d->gfam.mappings[slot] = (CXLType2GFAMMapping) {
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
                                 uint64_t base, uint64_t size)
{
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

static uint64_t cxl_type2_gfam_active_mappings_locked(CXLType2State *ct2d)
{
    uint64_t count = 0;

    for (int i = 0; i < CXL_TYPE2_MAX_GFAM_MAPPINGS; i++) {
        if (ct2d->gfam.mappings[i].active) {
            count++;
        }
    }

    return count;
}

static bool cxl_type2_gfam_allowed(CXLType2State *ct2d, uint32_t host_id,
                                   uint64_t base, uint64_t size,
                                   bool is_write, bool is_atomic)
{
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

static uint32_t cxl_type2_popcount32(uint32_t value)
{
    uint32_t count = 0;

    while (value) {
        count += value & 1;
        value >>= 1;
    }

    return count;
}

static void cxl_type2_mhsld_record(CXLType2State *ct2d, uint64_t addr,
                                   bool is_write, bool is_atomic)
{
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
        *line = (CXLType2MHSLDLine) {
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
                                            bool is_atomic)
{
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

static void cxl_type2_fabric_features_init(CXLType2State *ct2d)
{
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

        if (cxl_type2_dcd_add(ct2d, 0, initial_size, 1, &base, &size,
                              &tag) == 0 && ct2d->gfam.enabled) {
            for (uint32_t host = 0; host < ct2d->gfam.num_hosts; host++) {
                cxl_type2_gfam_grant(ct2d, host, base, size,
                                     ct2d->gfam.default_permissions);
            }
        }
    }

    qemu_log("CXL Type2 fabric: DCD=%d GFAM=%d MHSLD=%d initial=%" PRIu64
             "MB hosts=%u heads=%u\n",
             ct2d->dcd.enabled, ct2d->gfam.enabled, ct2d->mhsld.enabled,
             initial_size / MiB, ct2d->gfam.num_hosts,
             ct2d->mhsld.num_heads);
}

static void cxl_type2_fabric_features_cleanup(CXLType2State *ct2d)
{
    qemu_mutex_destroy(&ct2d->mhsld.lock);
    qemu_mutex_destroy(&ct2d->gfam.lock);
    qemu_mutex_destroy(&ct2d->dcd.lock);
}

/* ========================================================================
 * Coherent Pool Allocator
 * ======================================================================== */

/* First-fit allocator from free list, page-aligned (4KB) */
static int64_t cxl_coherent_pool_alloc(CXLType2State *ct2d, uint64_t size)
{
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
            uint64_t *val = g_new(uint64_t, 1);
            *key = alloc_offset;
            *val = aligned_size;
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
             (unsigned long)size,
             (unsigned long)ct2d->coherent_pool.used,
             (unsigned long)ct2d->coherent_pool.size);
    return -1;
}

/* Free allocation and coalesce with adjacent free blocks */
static int cxl_coherent_pool_free(CXLType2State *ct2d, uint64_t offset)
{
    qemu_mutex_lock(&ct2d->coherent_pool.lock);

    uint64_t *alloc_size = g_hash_table_lookup(ct2d->coherent_pool.allocations,
                                                &offset);
    if (!alloc_size) {
        qemu_mutex_unlock(&ct2d->coherent_pool.lock);
        qemu_log("CXL Type2: Coherent pool free FAILED: offset=0x%lx not found\n",
                 (unsigned long)offset);
        return -1;
    }

    uint64_t size = *alloc_size;
    g_hash_table_remove(ct2d->coherent_pool.allocations, &offset);
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
    if (new_blk->next && new_blk->offset + new_blk->size == new_blk->next->offset) {
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

/* ========================================================================
 * GPU Command Interface
 * ======================================================================== */

static const char *cxl_type2_paired_case_name(uint32_t case_kind)
{
    switch (case_kind) {
    case CXL_GPU_CASE_BASELINE:
        return "baseline";
    case CXL_GPU_CASE_CONCORDIA:
        return "concordia";
    default:
        return "invalid";
    }
}

static void cxl_type2_log_kimi_case_stage(uint64_t run_binding,
                                          uint32_t case_kind, uint64_t epoch,
                                          const char *operation,
                                          const char *state, int status,
                                          bool status_valid)
{
    int64_t host_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);

    if (status_valid) {
        qemu_log("KIMI_CASE_STAGE run_binding=%" PRIu64
                 " case=%s epoch=%" PRIu64
                 " operation=%s state=%s host_ns=%" PRId64 " status=%d\n",
                 run_binding, cxl_type2_paired_case_name(case_kind), epoch,
                 operation, state, host_ns, status);
    } else {
        qemu_log("KIMI_CASE_STAGE run_binding=%" PRIu64
                 " case=%s epoch=%" PRIu64
                 " operation=%s state=%s host_ns=%" PRId64
                 " status=not-applicable\n",
                 run_binding, cxl_type2_paired_case_name(case_kind), epoch,
                 operation, state, host_ns);
    }
}

static void cxl_type2_log_kimi_cleanup_progress(
    uint64_t run_binding, uint32_t case_kind, uint64_t epoch,
    const char *operation, const char *state, uint32_t completed,
    uint32_t total, int status, bool status_valid)
{
    int64_t host_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);

    if (status_valid) {
        qemu_log("KIMI_CASE_STAGE run_binding=%" PRIu64
                 " case=%s epoch=%" PRIu64
                 " operation=%s state=%s host_ns=%" PRId64
                 " completed=%u total=%u status=%d\n",
                 run_binding, cxl_type2_paired_case_name(case_kind), epoch,
                 operation, state, host_ns, completed, total, status);
    } else {
        qemu_log("KIMI_CASE_STAGE run_binding=%" PRIu64
                 " case=%s epoch=%" PRIu64
                 " operation=%s state=%s host_ns=%" PRId64
                 " completed=%u total=%u status=not-applicable\n",
                 run_binding, cxl_type2_paired_case_name(case_kind), epoch,
                 operation, state, host_ns, completed, total);
    }
}

static bool cxl_type2_cleanup_progress_due(uint32_t completed, uint32_t total)
{
    return completed == total || completed % 256 == 0;
}



static bool cxl_type2_command_requires_formal_case(uint32_t cmd)
{
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

static int cxl_type2_cuda_query_attribute(void *opaque, int32_t attribute)
{
    CXLType2CudaAttributeRequest *request = opaque;

    return hetgpu_cuda_device_get_attribute(request->hetgpu, attribute,
                                            &request->value);
}

static int cxl_type2_cuda_query_mem_info(void *opaque)
{
    CXLType2CudaMemInfoRequest *request = opaque;

    return hetgpu_cuda_mem_get_info(request->hetgpu, &request->free_bytes,
                                    &request->total_bytes);
}

static bool cxl_type2_reserve_gpu_handle(void ***handles, size_t *capacity,
                                         uint32_t count)
{
    size_t next_capacity;
    void **grown;

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
    grown = g_try_realloc_n(*handles, next_capacity, sizeof(**handles));
    if (!grown) {
        return false;
    }
    *handles = grown;
    *capacity = next_capacity;
    return true;
}

static bool cxl_type2_register_gpu_handle(void ***handles, size_t *capacity,
                                          uint32_t *count, void *handle,
                                          uint32_t *id)
{
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
                                       void **stream)
{
    if (!stream) {
        return false;
    }
    if (wire == CXL_GPU_STREAM_WIRE_NULL) {
        *stream = NULL;
        return true;
    }
    if (wire == CXL_GPU_STREAM_WIRE_LEGACY) {
        *stream = (void *)(uintptr_t)1;
        return true;
    }
    if (wire == CXL_GPU_STREAM_WIRE_PER_THREAD) {
        *stream = (void *)(uintptr_t)2;
        return true;
    }
    if (wire > UINT32_MAX || wire >= ct2d->gpu_cmd.num_streams ||
        !ct2d->gpu_cmd.streams[wire]) {
        return false;
    }
    *stream = ct2d->gpu_cmd.streams[wire];
    return true;
}

static int cxl_type2_htod_staging_free(CXLType2State *ct2d, void *data)
{
    int result = hetgpu_cuda_mem_free_host(&ct2d->gpu_info.hetgpu_state,
                                           data);

    if (result == CXL_GPU_SUCCESS) {
        ct2d->htod_driver_frees++;
    }
    return result;
}

static int cxl_type2_htod_staging_acquire(CXLType2State *ct2d,
                                           size_t request_bytes,
                                           void **data,
                                           size_t *capacity,
                                           uint64_t *buffer_id,
                                           bool *pool_hit)
{
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
        *data = buffer->data;
        *capacity = buffer->capacity;
        *buffer_id = buffer->id;
        *pool_hit = true;
        g_free(buffer);
        return CXL_GPU_SUCCESS;
    }

    ct2d->htod_pool_misses++;
    result = hetgpu_cuda_mem_host_alloc(&ct2d->gpu_info.hetgpu_state,
                                        data, request_bytes);
    if (result != CXL_GPU_SUCCESS) {
        return result;
    }
    ct2d->htod_driver_allocations++;
    *capacity = request_bytes;
    *buffer_id = ++ct2d->next_htod_staging_id;
    *pool_hit = false;
    return CXL_GPU_SUCCESS;
}

static int cxl_type2_htod_staging_release(CXLType2State *ct2d,
                                           void *data,
                                           size_t capacity,
                                           uint64_t buffer_id,
                                           bool *pooled)
{
    CXLType2HtoDStagingBuffer *buffer;

    if (ct2d->htod_staging_pool_size >= capacity &&
        ct2d->htod_pooled_bytes <=
            ct2d->htod_staging_pool_size - capacity) {
        buffer = g_new0(CXLType2HtoDStagingBuffer, 1);
        buffer->data = data;
        buffer->capacity = capacity;
        buffer->id = buffer_id;
        buffer->next = ct2d->htod_staging_pool;
        ct2d->htod_staging_pool = buffer;
        ct2d->htod_pooled_buffers++;
        ct2d->htod_pooled_bytes += capacity;
        ct2d->htod_peak_pooled_bytes = MAX(ct2d->htod_peak_pooled_bytes,
                                           ct2d->htod_pooled_bytes);
        *pooled = true;
        return CXL_GPU_SUCCESS;
    }

    if (ct2d->htod_staging_pool_size) {
        ct2d->htod_pool_evictions++;
    }
    *pooled = false;
    return cxl_type2_htod_staging_free(ct2d, data);
}

static int cxl_type2_clear_htod_staging_pool(CXLType2State *ct2d)
{
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

static int cxl_type2_release_pending_htod(CXLType2State *ct2d,
                                          HetGPUStream stream,
                                          bool all_streams,
                                          uint64_t through_sequence,
                                          const char *completion)
{
    CXLType2PendingHtoD **cursor = &ct2d->pending_htod;
    int first_error = CXL_GPU_SUCCESS;

    while (*cursor) {
        CXLType2PendingHtoD *pending = *cursor;

        if ((!all_streams && pending->stream != stream) ||
            pending->sequence > through_sequence) {
            cursor = &pending->next;
            continue;
        }
        *cursor = pending->next;
        if (pending->dev_ptr + pending->size <= ct2d->device_mem_size &&
            cxl_type2_fabric_access_allowed(ct2d, pending->dev_ptr,
                                            pending->size, true, false)) {
            uint8_t *mem = memory_region_get_ram_ptr(&ct2d->device_mem);
            if (mem) {
                memcpy(mem + pending->dev_ptr, pending->staging,
                       pending->size);
                if (ct2d->bar_coherency.enabled) {
                    cxl_bar_notify_gpu_access(&ct2d->bar_coherency,
                                              pending->dev_ptr,
                                              pending->size, true);
                }
            }
        }
        bool pooled;
        int64_t release_start_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
        int result = cxl_type2_htod_staging_release(
            ct2d, pending->staging, pending->staging_capacity,
            pending->staging_id, &pooled);
        int64_t release_duration_ns =
            qemu_clock_get_ns(QEMU_CLOCK_HOST) - release_start_ns;

        ct2d->htod_pending_copies--;
        ct2d->htod_pending_bytes -= pending->staging_capacity;
        if (ct2d->paired_case.qemu_cuda_calls_enabled) {
            qemu_log("CXL TYPE2 TRACE copy_completion original_call_id=0x%016"
                     PRIx64 " completion_call_id=0x%016" PRIx64
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
                     pending->call_id, ct2d->gpu_cmd.call_id,
                     pending->sequence, pending->size, pending->staging_id,
                     pending->staging_capacity, completion,
                     qemu_clock_get_ns(QEMU_CLOCK_HOST) -
                         pending->enqueue_host_ns,
                     pooled ? "pool" : "driver-free", release_duration_ns,
                     result, ct2d->htod_pending_copies,
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

static uint64_t cxl_type2_latest_pending_htod_sequence(
    CXLType2State *ct2d, HetGPUStream stream)
{
    uint64_t latest = 0;

    for (CXLType2PendingHtoD *pending = ct2d->pending_htod; pending;
         pending = pending->next) {
        if (pending->stream == stream && pending->sequence > latest) {
            latest = pending->sequence;
        }
    }
    return latest;
}

static CXLType2EventHtoDMark *cxl_type2_event_htod_mark(
    CXLType2State *ct2d, void *event, bool create)
{
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

static void cxl_type2_remove_event_htod_mark(CXLType2State *ct2d,
                                              void *event)
{
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

static void cxl_type2_clear_event_htod_marks(CXLType2State *ct2d)
{
    while (ct2d->event_htod_marks) {
        CXLType2EventHtoDMark *mark = ct2d->event_htod_marks;
        ct2d->event_htod_marks = mark->next;
        g_free(mark);
    }
}

static uint32_t cxl_type2_count_live_handles(void **handles, uint32_t count)
{
    uint32_t live = 0;

    for (uint32_t i = 0; i < count; i++) {
        if (handles[i]) {
            live++;
        }
    }
    return live;
}

static int cxl_type2_clear_gpu_handles(CXLType2State *ct2d,
                                       uint64_t run_binding, uint32_t case_kind,
                                       uint64_t epoch)
{
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
        qemu_log(
            "CXL Type2: GPU handle tables reset modules live=%u high_water=%u "
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
                                                operations[0], "begin",
                                                completed, total, 0, false);
        }
        for (uint32_t i = 0; i < ct2d->gpu_cmd.num_events; i++) {
            if (ct2d->gpu_cmd.events[i]) {
                int error =
                    hetgpu_cuda_event_destroy(hetgpu, ct2d->gpu_cmd.events[i]);
                if (category_error == 0 && error != 0) {
                    category_error = error;
                }
                if (first_error == 0 && error != 0) {
                    first_error = error;
                }
                completed++;
                if (log_stages &&
                    cxl_type2_cleanup_progress_due(completed, total)) {
                    cxl_type2_log_kimi_cleanup_progress(
                        run_binding, case_kind, epoch, operations[0],
                        "progress", completed, total, category_error, true);
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
                                             ct2d->gpu_cmd.num_streams);
        if (log_stages) {
            cxl_type2_log_kimi_cleanup_progress(run_binding, case_kind, epoch,
                                                operations[1], "begin",
                                                completed, total, 0, false);
        }
        for (uint32_t i = 0; i < ct2d->gpu_cmd.num_streams; i++) {
            if (ct2d->gpu_cmd.streams[i]) {
                int error = hetgpu_cuda_stream_destroy(
                    hetgpu, ct2d->gpu_cmd.streams[i]);
                if (category_error == 0 && error != 0) {
                    category_error = error;
                }
                if (first_error == 0 && error != 0) {
                    first_error = error;
                }
                completed++;
                if (log_stages &&
                    cxl_type2_cleanup_progress_due(completed, total)) {
                    cxl_type2_log_kimi_cleanup_progress(
                        run_binding, case_kind, epoch, operations[1],
                        "progress", completed, total, category_error, true);
                }
            }
        }
        if (log_stages) {
            cxl_type2_log_kimi_cleanup_progress(run_binding, case_kind, epoch,
                                                operations[1], "end", completed,
                                                total, category_error, true);
        }

        category_error = 0;
        completed = 0;
        total = cxl_type2_count_live_handles(ct2d->gpu_cmd.link_states,
                                             ct2d->gpu_cmd.num_link_states);
        if (log_stages) {
            cxl_type2_log_kimi_cleanup_progress(run_binding, case_kind, epoch,
                                                operations[2], "begin",
                                                completed, total, 0, false);
        }
        for (uint32_t i = 0; i < ct2d->gpu_cmd.num_link_states; i++) {
            if (ct2d->gpu_cmd.link_states[i]) {
                int error = hetgpu_cuda_link_destroy(
                    hetgpu, ct2d->gpu_cmd.link_states[i]);
                if (category_error == 0 && error != 0) {
                    category_error = error;
                }
                if (first_error == 0 && error != 0) {
                    first_error = error;
                }
                completed++;
                if (log_stages &&
                    cxl_type2_cleanup_progress_due(completed, total)) {
                    cxl_type2_log_kimi_cleanup_progress(
                        run_binding, case_kind, epoch, operations[2],
                        "progress", completed, total, category_error, true);
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
        total = cxl_type2_count_live_handles(ct2d->gpu_cmd.modules,
                                             ct2d->gpu_cmd.num_modules);
        if (log_stages) {
            cxl_type2_log_kimi_cleanup_progress(run_binding, case_kind, epoch,
                                                operations[3], "begin",
                                                completed, total, 0, false);
        }
        for (uint32_t i = 0; i < ct2d->gpu_cmd.num_modules; i++) {
            if (ct2d->gpu_cmd.modules[i]) {
                int error =
                    hetgpu_unload_module(hetgpu, ct2d->gpu_cmd.modules[i]);
                if (category_error == 0 && error != 0) {
                    category_error = error;
                }
                if (first_error == 0 && error != 0) {
                    first_error = error;
                }
                completed++;
                if (log_stages &&
                    cxl_type2_cleanup_progress_due(completed, total)) {
                    cxl_type2_log_kimi_cleanup_progress(
                        run_binding, case_kind, epoch, operations[3],
                        "progress", completed, total, category_error, true);
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
                                      "handle_table_release", "begin", 0,
                                      false);
    }
    g_free(ct2d->gpu_cmd.modules);
    g_free(ct2d->gpu_cmd.functions);
    g_free(ct2d->gpu_cmd.graphs);
    g_free(ct2d->gpu_cmd.graph_execs);
    g_free(ct2d->gpu_cmd.graph_nodes);
    g_free(ct2d->gpu_cmd.link_states);
    g_free(ct2d->gpu_cmd.streams);
    g_free(ct2d->gpu_cmd.events);
    cxl_type2_clear_event_htod_marks(ct2d);
    ct2d->gpu_cmd.modules = NULL;
    ct2d->gpu_cmd.functions = NULL;
    ct2d->gpu_cmd.graphs = NULL;
    ct2d->gpu_cmd.graph_execs = NULL;
    ct2d->gpu_cmd.graph_nodes = NULL;
    ct2d->gpu_cmd.link_states = NULL;
    ct2d->gpu_cmd.streams = NULL;
    ct2d->gpu_cmd.events = NULL;
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

static uint64_t cxl_type2_paired_config_binding(
    CXLType2State *ct2d, uint32_t case_kind, const char *aof_path,
    const char *restore_path, const char *manifest_path)
{
    g_autofree char *identity = g_strdup_printf(
        "kimi-case-config-v2\ncase=%u\nenabled=%u\nrestore=0\n"
        "min-allocation=%" PRIu64 "\nmax-regions=%" PRIu64
        "\ncheckpoint-enabled=%u\ncheckpoint-every=%" PRIu64
        "\nqemu-cuda-calls=%u\nconcordia-runtime-details=%u\naof=%s\n"
        "restore-aof=%s\nmanifest=%s\n",
        case_kind, case_kind == CXL_GPU_CASE_CONCORDIA,
        ct2d->paired_case.min_allocation_bytes,
        ct2d->paired_case.max_regions,
        ct2d->paired_case.checkpoint_enabled,
        ct2d->paired_case.checkpoint_every_launches,
        ct2d->paired_case.qemu_cuda_calls_enabled,
        ct2d->paired_case.concordia_runtime_details_enabled, aof_path, restore_path,
        manifest_path);
    g_autofree char *digest = g_compute_checksum_for_string(
        G_CHECKSUM_SHA256, identity, -1);
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
                                                  uint64_t epoch)
{
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
                                      "formal_backend_reset", "end", error,
                                      true);
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
                                      "formal_backend_reset", "end", error,
                                      true);
        return error;
    }

    hetgpu_set_coherency_callback(hetgpu, cxl_type2_hetgpu_coherency_callback,
                                  ct2d);
    ct2d->gpu_info.passthrough_enabled = true;
    ct2d->gpu_info.gpu_mem_size = hetgpu->props.total_memory;
    cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch,
                                  "formal_backend_reset", "end", HETGPU_SUCCESS,
                                  true);
    return HETGPU_SUCCESS;
}

static void cxl_type2_paired_case_begin(CXLType2State *ct2d,
                                        uint64_t trace_sequence)
{
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
    HetGPUKimiCaseBeginV2 input = { 0 };
    HetGPUKimiCaseResultV1 result = { 0 };
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
    config_binding = cxl_type2_paired_config_binding(
        ct2d, case_kind, aof_path, restore_path, manifest_path);

    cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch, "case_begin",
                                  "begin", 0, false);
    error = cxl_type2_reset_formal_backend(ct2d, run_binding, case_kind, epoch);
    if (error != HETGPU_SUCCESS) {
        ct2d->paired_case.failed = true;
        ct2d->paired_case.failure_code = error;
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_CONTEXT;
        cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch,
                                      "case_begin", "end", error, true);
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
    input.checkpoint_every_launches =
        ct2d->paired_case.checkpoint_every_launches;
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
        int handle_error;
        int cleanup_status;

        qemu_log("CXL Type2: Concordia case begin failed error=%u "
                 "outcome=%u reason=%u detail=%.*s\n",
                 error, result.outcome, result.reason,
                 (int)MIN(result.error_len, HETGPU_KIMI_CASE_ERROR_BYTES),
                 result.error);
        cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch,
                                      "failed_begin_cleanup", "begin", 0,
                                      false);
        cleanup_error = hetgpu_cleanup_formal(hetgpu);
        if (cleanup_error != HETGPU_SUCCESS) {
            qemu_log("CXL Type2: cleanup after failed case begin also "
                     "failed: %u\n",
                     cleanup_error);
        }
        handle_error =
            cxl_type2_clear_gpu_handles(ct2d, run_binding, case_kind, epoch);
        cleanup_status =
            cleanup_error != HETGPU_SUCCESS ? cleanup_error : handle_error;
        cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch,
                                      "failed_begin_cleanup", "end",
                                      cleanup_status, true);
        ct2d->gpu_info.passthrough_enabled = false;
        ct2d->paired_case.failed = true;
        ct2d->paired_case.failure_code = error != HETGPU_SUCCESS ? error
                                         : result.reason         ? result.reason
                                                         : HETGPU_ERROR_UNKNOWN;
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_UNKNOWN;
        cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch,
                                      "case_begin", "end",
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
                                      uint64_t trace_sequence)
{
    HetGPUState *hetgpu = &ct2d->gpu_info.hetgpu_state;
    uint32_t protocol = ct2d->gpu_cmd.params[0];
    uint64_t expected_epoch = ct2d->gpu_cmd.params[1];
    int32_t application_exit = (int32_t)ct2d->gpu_cmd.params[2];
    uint64_t run_binding = ct2d->gpu_cmd.params[3];
    uint64_t epoch = ct2d->paired_case.active_epoch;
    uint32_t case_kind = ct2d->paired_case.active_case;
    HetGPUKimiCaseEndV2 input = { 0 };
    HetGPUKimiCaseResultV1 result = { 0 };
    HetGPUError sync_error;
    HetGPUError concordia_error;
    HetGPUError reset_error;
    int pool_clear_result;
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
    cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch,
                                  "backend_synchronize", "begin", 0, false);
    sync_error = hetgpu_synchronize(hetgpu);
    if (sync_error == HETGPU_SUCCESS &&
        cxl_type2_release_pending_htod(ct2d, NULL, true, UINT64_MAX,
                                       "case-end") !=
            CXL_GPU_SUCCESS) {
        sync_error = HETGPU_ERROR_UNKNOWN;
    }
    pool_clear_result = cxl_type2_clear_htod_staging_pool(ct2d);
    if (sync_error == HETGPU_SUCCESS &&
        pool_clear_result != CXL_GPU_SUCCESS) {
        sync_error = HETGPU_ERROR_UNKNOWN;
    }
    cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch,
                                  "backend_synchronize", "end", sync_error,
                                  true);
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
        cxl_type2_log_kimi_case_stage(
            run_binding, case_kind, epoch, "formal_backend_cleanup", "begin",
            0, false);
        cxl_type2_log_kimi_case_stage(
            run_binding, case_kind, epoch, "formal_backend_cleanup", "end", 0,
            false);
        reset_error = HETGPU_SUCCESS;
    } else {
        cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch,
                                      "formal_backend_cleanup", "begin", 0,
                                      false);
        reset_error = hetgpu_cleanup_formal(hetgpu);
        cxl_type2_log_kimi_case_stage(run_binding, case_kind, epoch,
                                      "formal_backend_cleanup", "end",
                                      reset_error, true);
        ct2d->gpu_info.passthrough_enabled = false;
    }
    (void)cxl_type2_clear_gpu_handles(ct2d, run_binding, case_kind, epoch);

    ct2d->gpu_cmd.results[0] = epoch;
    ct2d->gpu_cmd.results[1] = trace_sequence;
    ct2d->gpu_cmd.results[2] = result.outcome;
    ct2d->gpu_cmd.results[3] = reset_error;

    qemu_log("KIMI_CASE_END run_binding=%" PRIu64 " case=%s epoch=%" PRIu64
             " last_sequence=%" PRIu64 " app_exit=%d sync_status=%u"
             " concordia_status=%u concordia_reason=%u stateful_launches=%"
             PRIu64 " reset_status=%u detail=%.*s\n",
             run_binding, cxl_type2_paired_case_name(case_kind), epoch,
             trace_sequence, application_exit, sync_error, result.outcome,
             result.reason, result.stateful_launches, reset_error,
             (int)MIN(result.error_len, HETGPU_KIMI_CASE_ERROR_BYTES),
             result.error);

    cxl_type2_log_kimi_case_stage(
        run_binding, case_kind, epoch, "case_end", "end",
        application_exit != 0               ? HETGPU_ERROR_UNKNOWN
        : sync_error != HETGPU_SUCCESS      ? sync_error
        : concordia_error != HETGPU_SUCCESS ? concordia_error
        : !result_valid ? (result.reason ? result.reason : HETGPU_ERROR_UNKNOWN)
        : reset_error != HETGPU_SUCCESS ? reset_error : HETGPU_SUCCESS,
        true);

    ct2d->paired_case.active_case = CXL_GPU_CASE_NONE;
    ct2d->paired_case.active_epoch = 0;
    ct2d->paired_case.active_first_sequence = 0;
    ct2d->paired_case.active_config_binding = 0;

    if (application_exit != 0 || sync_error != HETGPU_SUCCESS ||
        concordia_error != HETGPU_SUCCESS || !result_valid ||
        reset_error != HETGPU_SUCCESS) {
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
        } else {
            ct2d->paired_case.failure_code = reset_error;
        }
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_UNKNOWN;
    }
}

static void cxl_type2_gpu_execute_cmd(CXLType2State *ct2d, uint32_t cmd)
{
    HetGPUState *hetgpu = &ct2d->gpu_info.hetgpu_state;
    HetGPUError err;
    uint64_t dev_ptr;
    size_t size;
    uint64_t trace_sequence = ++ct2d->gpu_cmd.trace_sequence;
    int64_t trace_start_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);

    if (ct2d->paired_case.qemu_cuda_calls_enabled) qemu_log("CXL TYPE2 TRACE cmd_begin seq=%" PRIu64
             " call_id=0x%016" PRIx64
             " cmd=0x%x host_ns=%" PRId64 " p0=0x%" PRIx64 " p1=%" PRIu64 "\n",
             trace_sequence, ct2d->gpu_cmd.call_id, cmd, trace_start_ns, ct2d->gpu_cmd.params[0],
             ct2d->gpu_cmd.params[1]);

    if (ct2d->paired_case.qemu_cuda_calls_enabled) qemu_log_mask(LOG_GUEST_ERROR,
                  "CXL GPU: execute cmd 0x%x, hetgpu_init=%d, ctx=%p\n",
                  cmd, hetgpu->initialized, hetgpu->context);

    ct2d->gpu_cmd.cmd_status = CXL_GPU_CMD_STATUS_RUNNING;
    ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
    hetgpu_cuda_trace_set_detailed_logs(ct2d->paired_case.qemu_cuda_calls_enabled);
    hetgpu_cuda_trace_set_call_id(ct2d->gpu_cmd.call_id);

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

    case CXL_GPU_CMD_INIT:
        if (!hetgpu->initialized) {
            ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
        }
        break;

    case CXL_GPU_CMD_GET_ERROR_STRING:
        {
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
        }
        break;

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
            int cuda_result = hetgpu_cuda_device_total_memory(hetgpu,
                                                                &total_bytes);
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

            if (!cxl_type2_cuda_attribute_wire_is_valid(
                    ct2d->gpu_cmd.params[0])) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
                break;
            }
            if (!hetgpu->initialized) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
                break;
            }
            if (!cxl_type2_cuda_dispatch_attribute(
                    ct2d->gpu_cmd.params[0], cxl_type2_cuda_query_attribute,
                    &request, &cuda_result)) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
                break;
            }
            ct2d->gpu_cmd.cmd_result = cuda_result;
            if (cuda_result == CXL_GPU_SUCCESS) {
                ct2d->gpu_cmd.results[0] =
                    (uint64_t)(int64_t)(int32_t)request.value;
            }
            qemu_log("CXL TYPE2 CUDA device_attribute attribute=%d "
                     "driver_result=%d value=%d\n",
                     (int)(int32_t)ct2d->gpu_cmd.params[0], cuda_result,
                     request.value);
        }
        break;

    case CXL_GPU_CMD_GET_ERROR_NAME:
        {
            const char *name = NULL;
            size_t length;
            int cuda_result = hetgpu_cuda_get_error_name(
                hetgpu, (int)ct2d->gpu_cmd.params[0], &name);

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
        }
        break;

    case CXL_GPU_CMD_CASE_BEGIN:
        cxl_type2_paired_case_begin(ct2d, trace_sequence);
        break;

    case CXL_GPU_CMD_CASE_END:
        cxl_type2_paired_case_end(ct2d, trace_sequence);
        break;

    case CXL_GPU_CMD_CTX_CREATE:
        if (hetgpu->initialized) {
            err = hetgpu_create_context(hetgpu);
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
            err = hetgpu_synchronize(hetgpu);
            if (err == HETGPU_SUCCESS &&
                cxl_type2_release_pending_htod(ct2d, NULL, true,
                                               UINT64_MAX, "context-sync") !=
                    CXL_GPU_SUCCESS) {
                err = HETGPU_ERROR_UNKNOWN;
            }
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
                    ct2d->paired_case.active_epoch,
                    cxl_type2_cuda_query_mem_info, &request, &cuda_result)) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_CONTEXT;
                break;
            }
            ct2d->gpu_cmd.cmd_result = cuda_result;
            if (cuda_result == CXL_GPU_SUCCESS) {
                ct2d->gpu_cmd.results[0] = request.free_bytes;
                ct2d->gpu_cmd.results[1] = request.total_bytes;
            }
            qemu_log("CXL TYPE2 CUDA mem_info token=%" PRIu64
                     " active_epoch=%" PRIu64 " active_case=%u "
                     "live_context=%u driver_result=%d free=%zu total=%zu\n",
                     ct2d->gpu_cmd.params[0], ct2d->paired_case.active_epoch,
                     ct2d->paired_case.active_case, hetgpu->context != NULL,
                     cuda_result, request.free_bytes, request.total_bytes);
        }
        break;

    case CXL_GPU_CMD_MEM_GET_POINTER_MEMORY_TYPE:
        {
            int memory_type = 0;
            int cuda_result = hetgpu_pointer_get_memory_type(
                hetgpu, ct2d->gpu_cmd.params[0], &memory_type);

            ct2d->gpu_cmd.cmd_result = cuda_result;
            if (cuda_result == CXL_GPU_SUCCESS) {
                ct2d->gpu_cmd.results[0] = (uint64_t)memory_type;
            }
        }
        break;

    case CXL_GPU_CMD_MEM_ALLOC:
        size = ct2d->gpu_cmd.params[0];
        if (hetgpu->initialized) {
            err = hetgpu_malloc(hetgpu, size, HETGPU_MEM_DEFAULT, &dev_ptr);
            if (err == HETGPU_SUCCESS) {
                ct2d->gpu_cmd.results[0] = dev_ptr;
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

    case CXL_GPU_CMD_MEM_FREE:
        dev_ptr = ct2d->gpu_cmd.params[0];
        if (hetgpu->initialized) {
            err = hetgpu_free(hetgpu, dev_ptr);
            if (err != HETGPU_SUCCESS) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
            }
        }
        /* Fallback: no-op for simple allocator */
        break;

    case CXL_GPU_CMD_MEM_COPY_HTOD:
        dev_ptr = ct2d->gpu_cmd.params[0];  /* dst device ptr */
        size = ct2d->gpu_cmd.params[1];     /* size */
        /* Data is in ct2d->gpu_cmd.data buffer */
        if (size > ct2d->gpu_cmd.data_size) {
            ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
            break;
        }
        if (hetgpu->initialized) {
            int64_t driver_start_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
            err = hetgpu_memcpy_htod(hetgpu, dev_ptr, ct2d->gpu_cmd.data, size);
            if (ct2d->paired_case.qemu_cuda_calls_enabled) qemu_log("CXL TYPE2 TRACE copy_driver call_id=0x%016" PRIx64
                     " direction=htod bytes=%zu driver_duration_ns=%" PRId64
                     " backend_result=%d implementation=blocking stream_forwarded=0\n",
                     ct2d->gpu_cmd.call_id, size,
                     qemu_clock_get_ns(QEMU_CLOCK_HOST) - driver_start_ns, err);
            if (err != HETGPU_SUCCESS) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
            }
            /* Also update shadow copy in device_mem for coherency tracking */
            if (dev_ptr + size <= ct2d->device_mem_size &&
                cxl_type2_fabric_access_allowed(ct2d, dev_ptr, size,
                                                true, false)) {
                uint8_t *mem = memory_region_get_ram_ptr(&ct2d->device_mem);
                if (mem) {
                    memcpy(mem + dev_ptr, ct2d->gpu_cmd.data, size);
                    /* Notify BAR coherency layer of GPU write */
                    if (ct2d->bar_coherency.enabled) {
                        cxl_bar_notify_gpu_access(&ct2d->bar_coherency,
                                                   dev_ptr, size, true);
                    }
                }
            }
        } else {
            /* Fallback: copy to device memory region */
            if (dev_ptr + size <= ct2d->device_mem_size &&
                cxl_type2_fabric_access_allowed(ct2d, dev_ptr, size,
                                                true, false)) {
                uint8_t *mem = memory_region_get_ram_ptr(&ct2d->device_mem);
                if (mem) {
                    memcpy(mem + dev_ptr, ct2d->gpu_cmd.data, size);
                    /* Notify BAR coherency layer of GPU write */
                    if (ct2d->bar_coherency.enabled) {
                        cxl_bar_notify_gpu_access(&ct2d->bar_coherency,
                                                   dev_ptr, size, true);
                    }
                }
            } else {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
            }
        }
        break;

    case CXL_GPU_CMD_MEM_COPY_HTOD_ASYNC:
        {
            void *stream = NULL;
            void *staging = NULL;
            CXLType2PendingHtoD *pending;
            int result;
            int64_t acquire_start_ns;
            int64_t acquire_duration_ns;
            int64_t staging_start_ns;
            int64_t staging_duration_ns;
            int64_t enqueue_start_ns;
            int64_t enqueue_duration_ns;
            size_t staging_capacity;
            uint64_t staging_id;
            bool pool_hit;

            dev_ptr = ct2d->gpu_cmd.params[0];
            size = ct2d->gpu_cmd.params[1];
            if (!size || size > ct2d->gpu_cmd.data_size ||
                !cxl_type2_stream_from_wire(ct2d,
                                            ct2d->gpu_cmd.params[2],
                                            &stream)) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
                break;
            }
            acquire_start_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
            result = cxl_type2_htod_staging_acquire(
                ct2d, size, &staging, &staging_capacity, &staging_id,
                &pool_hit);
            acquire_duration_ns =
                qemu_clock_get_ns(QEMU_CLOCK_HOST) - acquire_start_ns;
            if (result != CXL_GPU_SUCCESS) {
                ct2d->gpu_cmd.cmd_result = result;
                break;
            }
            staging_start_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
            memcpy(staging, ct2d->gpu_cmd.data, size);
            staging_duration_ns =
                qemu_clock_get_ns(QEMU_CLOCK_HOST) - staging_start_ns;
            enqueue_start_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
            result = hetgpu_cuda_memcpy_htod_async(
                hetgpu, dev_ptr, staging, size, stream);
            enqueue_duration_ns =
                qemu_clock_get_ns(QEMU_CLOCK_HOST) - enqueue_start_ns;
            if (result != CXL_GPU_SUCCESS) {
                bool pooled;
                (void)cxl_type2_htod_staging_release(
                    ct2d, staging, staging_capacity, staging_id, &pooled);
                ct2d->gpu_cmd.cmd_result = result;
                break;
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
            pending->next = ct2d->pending_htod;
            ct2d->pending_htod = pending;
            ct2d->htod_pending_copies++;
            ct2d->htod_pending_bytes += staging_capacity;
            ct2d->htod_peak_pending_copies =
                MAX(ct2d->htod_peak_pending_copies,
                    ct2d->htod_pending_copies);
            ct2d->htod_peak_pending_bytes =
                MAX(ct2d->htod_peak_pending_bytes,
                    ct2d->htod_pending_bytes);
            if (ct2d->paired_case.qemu_cuda_calls_enabled) qemu_log("CXL TYPE2 TRACE copy_driver call_id=0x%016" PRIx64
                     " direction=htod bytes=%zu backend_result=%d "
                     "implementation=async-enqueue stream_forwarded=1 "
                     "staging_id=%" PRIu64 " capacity_bytes=%zu"
                     " staging_source=%s acquire_duration_ns=%" PRId64
                     " host_alloc_duration_ns=%" PRId64
                     " staging_memcpy_duration_ns=%" PRId64
                     " driver_enqueue_duration_ns=%" PRId64
                     " pending_copies=%" PRIu64 " pending_bytes=%" PRIu64
                     " peak_pending_copies=%" PRIu64
                     " peak_pending_bytes=%" PRIu64
                     " pooled_buffers=%" PRIu64 " pooled_bytes=%" PRIu64
                     " peak_pooled_bytes=%" PRIu64
                     " pool_hits=%" PRIu64 " pool_misses=%" PRIu64
                     " driver_allocations=%" PRIu64
                     " driver_frees=%" PRIu64 " evictions=%" PRIu64 "\n",
                     ct2d->gpu_cmd.call_id, size, result, staging_id,
                     staging_capacity, pool_hit ? "pool" : "driver",
                     acquire_duration_ns,
                     pool_hit ? 0 : acquire_duration_ns,
                     staging_duration_ns, enqueue_duration_ns,
                     ct2d->htod_pending_copies,
                     ct2d->htod_pending_bytes,
                     ct2d->htod_peak_pending_copies,
                     ct2d->htod_peak_pending_bytes,
                     ct2d->htod_pooled_buffers, ct2d->htod_pooled_bytes,
                     ct2d->htod_peak_pooled_bytes, ct2d->htod_pool_hits,
                     ct2d->htod_pool_misses, ct2d->htod_driver_allocations,
                     ct2d->htod_driver_frees, ct2d->htod_pool_evictions);
        }
        break;

    case CXL_GPU_CMD_MEM_COPY_DTOH:
        dev_ptr = ct2d->gpu_cmd.params[0];  /* src device ptr */
        size = ct2d->gpu_cmd.params[1];     /* size */
        if (size > ct2d->gpu_cmd.data_size) {
            ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
            break;
        }
        if (hetgpu->initialized) {
            /* Notify BAR coherency layer before GPU read */
            if (ct2d->bar_coherency.enabled) {
                cxl_bar_notify_gpu_access(&ct2d->bar_coherency,
                                           dev_ptr, size, false);
            }
            int64_t driver_start_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
            err = hetgpu_memcpy_dtoh(hetgpu, ct2d->gpu_cmd.data, dev_ptr, size);
            if (ct2d->paired_case.qemu_cuda_calls_enabled) qemu_log("CXL TYPE2 TRACE copy_driver call_id=0x%016" PRIx64
                     " direction=dtoh bytes=%zu driver_duration_ns=%" PRId64
                     " backend_result=%d implementation=blocking stream_forwarded=0\n",
                     ct2d->gpu_cmd.call_id, size,
                     qemu_clock_get_ns(QEMU_CLOCK_HOST) - driver_start_ns, err);
            if (err != HETGPU_SUCCESS) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
            }
            /* Update shadow copy from GPU for coherency */
            if (dev_ptr + size <= ct2d->device_mem_size &&
                cxl_type2_fabric_access_allowed(ct2d, dev_ptr, size,
                                                false, false)) {
                uint8_t *mem = memory_region_get_ram_ptr(&ct2d->device_mem);
                if (mem) {
                    memcpy(mem + dev_ptr, ct2d->gpu_cmd.data, size);
                }
            }
        } else {
            /* Fallback: copy from device memory region */
            if (dev_ptr + size <= ct2d->device_mem_size &&
                cxl_type2_fabric_access_allowed(ct2d, dev_ptr, size,
                                                false, false)) {
                uint8_t *mem = memory_region_get_ram_ptr(&ct2d->device_mem);
                if (mem) {
                    /* Notify BAR coherency layer before GPU read */
                    if (ct2d->bar_coherency.enabled) {
                        cxl_bar_notify_gpu_access(&ct2d->bar_coherency,
                                                   dev_ptr, size, false);
                    }
                    memcpy(ct2d->gpu_cmd.data, mem + dev_ptr, size);
                }
            } else {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
            }
        }
        break;

    case CXL_GPU_CMD_MEM_COPY_DTOD:
        {
            uint64_t dst_dev_ptr = ct2d->gpu_cmd.params[0];
            uint64_t src_dev_ptr = ct2d->gpu_cmd.params[1];
            size_t xfer_size = ct2d->gpu_cmd.params[2];

            if (hetgpu->initialized) {
                int64_t driver_start_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
                err = hetgpu_memcpy_dtod(hetgpu, dst_dev_ptr, src_dev_ptr,
                                         xfer_size);
                if (ct2d->paired_case.qemu_cuda_calls_enabled) qemu_log("CXL TYPE2 TRACE copy_driver call_id=0x%016" PRIx64
                         " direction=dtod bytes=%zu driver_duration_ns=%" PRId64
                         " backend_result=%d implementation=blocking-direct "
                         "stream_forwarded=0\n",
                         ct2d->gpu_cmd.call_id, xfer_size,
                         qemu_clock_get_ns(QEMU_CLOCK_HOST) - driver_start_ns,
                         err);
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
                    !cxl_type2_fabric_access_allowed(ct2d, src_dev_ptr,
                                                     xfer_size, false, false) ||
                    !cxl_type2_fabric_access_allowed(ct2d, dst_dev_ptr,
                                                     xfer_size, true, false)) {
                    ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
                    break;
                }
                memmove(mem + dst_dev_ptr, mem + src_dev_ptr, xfer_size);
            }

            if (ct2d->bar_coherency.enabled) {
                cxl_bar_notify_gpu_access(&ct2d->bar_coherency,
                                          src_dev_ptr, xfer_size, false);
                cxl_bar_notify_gpu_access(&ct2d->bar_coherency,
                                          dst_dev_ptr, xfer_size, true);
            }
            if (hetgpu->initialized &&
                src_dev_ptr <= ct2d->device_mem_size &&
                xfer_size <= ct2d->device_mem_size - src_dev_ptr &&
                dst_dev_ptr <= ct2d->device_mem_size &&
                xfer_size <= ct2d->device_mem_size - dst_dev_ptr) {
                uint8_t *mem = memory_region_get_ram_ptr(&ct2d->device_mem);
                if (mem) {
                    memmove(mem + dst_dev_ptr, mem + src_dev_ptr, xfer_size);
                }
            }
        }
        break;

    case CXL_GPU_CMD_MEM_COPY_2D_DTOD:
        {
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
                (height - 1 && (src_pitch > (UINT64_MAX - src_dev_ptr - width) / (height - 1) ||
                                dst_pitch > (UINT64_MAX - dst_dev_ptr - width) / (height - 1)))) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
                break;
            }
            total_bytes = width * height;

            int64_t driver_start_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
            if (hetgpu->initialized && hetgpu->backend != HETGPU_BACKEND_SIMULATION) {
                err = hetgpu_memcpy2d_dtod(hetgpu, dst_dev_ptr, dst_pitch,
                                           src_dev_ptr, src_pitch, width, height);
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
            if (ct2d->paired_case.qemu_cuda_calls_enabled) qemu_log("CXL TYPE2 TRACE copy_driver call_id=0x%016" PRIx64
                     " direction=dtod bytes=%zu rows=%zu row_commands_eliminated=%zu "
                     "driver_duration_ns=%" PRId64
                     " backend_result=%d implementation=blocking-direct-2d "
                     "stream_forwarded=0\n",
                     ct2d->gpu_cmd.call_id, total_bytes, height, height - 1,
                     qemu_clock_get_ns(QEMU_CLOCK_HOST) - driver_start_ns, err);
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
                if (src <= ct2d->device_mem_size && width <= ct2d->device_mem_size - src &&
                    dst <= ct2d->device_mem_size && width <= ct2d->device_mem_size - dst) {
                    uint8_t *mem = memory_region_get_ram_ptr(&ct2d->device_mem);
                    if (mem) {
                        memmove(mem + dst, mem + src, width);
                    }
                }
            }
        }
        break;

    case CXL_GPU_CMD_MODULE_LOAD_PTX:
        if (hetgpu->initialized) {
            /* PTX source is in data buffer */
            void *module = NULL;
            if (!cxl_type2_reserve_gpu_handle(&ct2d->gpu_cmd.modules,
                                               &ct2d->gpu_cmd.modules_capacity,
                                               ct2d->gpu_cmd.num_modules)) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_OUT_OF_MEMORY;
                break;
            }
            err = hetgpu_load_ptx(hetgpu, (const char *)ct2d->gpu_cmd.data,
                                  (HetGPUModule *)&module);
            if (err == HETGPU_SUCCESS) {
                ct2d->gpu_cmd.modules[ct2d->gpu_cmd.num_modules] = module;
                ct2d->gpu_cmd.results[0] = ct2d->gpu_cmd.num_modules;
                ct2d->gpu_cmd.num_modules++;
                ct2d->gpu_cmd.modules_high_water =
                    MAX(ct2d->gpu_cmd.modules_high_water,
                        ct2d->gpu_cmd.num_modules);
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
            if (encoding != 0 &&
                encoding != CXL_GPU_MODULE_DATA_ZSTD &&
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
                        qemu_log("CXL hetGPU: CUBIN Zstd decode failed: %s result=%zu expected=%zu\n",
                                 ZSTD_getErrorName(result), result,
                                 uncompressed_size);
                        g_free(decoded);
                        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
                        break;
                    }
                    cubin_size = result;
                } else {
                    int result = LZ4_decompress_safe(
                        (const char *)ct2d->gpu_cmd.data, decoded,
                        (int)size, (int)uncompressed_size);
                    if (result < 0 || (size_t)result != uncompressed_size) {
                        qemu_log("CXL hetGPU: CUBIN LZ4 decode failed: result=%d expected=%zu\n",
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
                     encoding == CXL_GPU_MODULE_DATA_ZSTD ? "zstd" :
                     encoding == CXL_GPU_MODULE_DATA_LZ4 ? "lz4" : "raw",
                     size, cubin_size);

            void *module = NULL;
            if (!cxl_type2_reserve_gpu_handle(&ct2d->gpu_cmd.modules,
                                               &ct2d->gpu_cmd.modules_capacity,
                                               ct2d->gpu_cmd.num_modules)) {
                g_free(decoded);
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_OUT_OF_MEMORY;
                break;
            }
            err = hetgpu_load_cubin(hetgpu, cubin_data, cubin_size,
                                    (HetGPUModule *)&module);
            g_free(decoded);
            if (err == HETGPU_SUCCESS) {
                ct2d->gpu_cmd.modules[ct2d->gpu_cmd.num_modules] = module;
                ct2d->gpu_cmd.results[0] = ct2d->gpu_cmd.num_modules;
                ct2d->gpu_cmd.num_modules++;
                ct2d->gpu_cmd.modules_high_water =
                    MAX(ct2d->gpu_cmd.modules_high_water,
                        ct2d->gpu_cmd.num_modules);
            } else {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_PTX;
            }
        } else {
            ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_NOT_INITIALIZED;
        }
        break;

    case CXL_GPU_CMD_FUNC_GET:
        if (hetgpu->initialized) {
            uint32_t module_id = ct2d->gpu_cmd.params[0];
            /* Function name is in data buffer */
            if (module_id < ct2d->gpu_cmd.num_modules &&
                ct2d->gpu_cmd.modules[module_id]) {
                void *func = NULL;
                if (!cxl_type2_reserve_gpu_handle(
                        &ct2d->gpu_cmd.functions,
                        &ct2d->gpu_cmd.functions_capacity,
                        ct2d->gpu_cmd.num_functions)) {
                    ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_OUT_OF_MEMORY;
                    break;
                }
                err = hetgpu_get_function(hetgpu,
                                          ct2d->gpu_cmd.modules[module_id],
                                          (const char *)ct2d->gpu_cmd.data,
                                          (HetGPUFunction *)&func);
                if (err == HETGPU_SUCCESS) {
                    ct2d->gpu_cmd.functions[ct2d->gpu_cmd.num_functions] = func;
                    ct2d->gpu_cmd.results[0] = ct2d->gpu_cmd.num_functions;
                    ct2d->gpu_cmd.num_functions++;
                    ct2d->gpu_cmd.functions_high_water =
                        MAX(ct2d->gpu_cmd.functions_high_water,
                            ct2d->gpu_cmd.num_functions);
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
                !ct2d->gpu_cmd.modules[module_id]) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
                break;
            }
            err = hetgpu_unload_module(hetgpu,
                                       ct2d->gpu_cmd.modules[module_id]);
            if (err == HETGPU_SUCCESS) {
                ct2d->gpu_cmd.modules[module_id] = NULL;
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
                !ct2d->gpu_cmd.modules[module_id]) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
                break;
            }
            if (!memchr(ct2d->gpu_cmd.data, '\0', ct2d->gpu_cmd.data_size)) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
                break;
            }

            dev_ptr = 0;
            size = 0;
            err = hetgpu_get_global(hetgpu,
                                    ct2d->gpu_cmd.modules[module_id],
                                    (const char *)ct2d->gpu_cmd.data,
                                    &dev_ptr, &size);
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
            err = hetgpu_get_param_layout(hetgpu,
                                          ct2d->gpu_cmd.functions[func_id],
                                          &layout);
            if (err != HETGPU_SUCCESS ||
                layout->num_args > ARRAY_SIZE(wire.params) ||
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
                hetgpu, ct2d->gpu_cmd.functions[func_id],
                (int)ct2d->gpu_cmd.params[1],
                (int)ct2d->gpu_cmd.params[2]);
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
                !ct2d->gpu_cmd.functions[func_id]) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
                break;
            }
            err = hetgpu_get_function_attribute(
                hetgpu, ct2d->gpu_cmd.functions[func_id],
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
                hetgpu, ct2d->gpu_cmd.functions[func_id],
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
            int cuda_result = hetgpu_cuda_module_get_loading_mode(hetgpu,
                                                                   &mode);

            ct2d->gpu_cmd.cmd_result = cuda_result;
            if (cuda_result == CXL_GPU_SUCCESS) {
                ct2d->gpu_cmd.results[0] = (uint64_t)(uint32_t)mode;
            }
            qemu_log("CXL TYPE2 CUDA module_loading_mode "
                     "driver_result=%d mode=%d\n", cuda_result, mode);
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
                .grid_dim = {
                    ct2d->gpu_cmd.params[3] & 0xFFFFFFFF,
                    (ct2d->gpu_cmd.params[3] >> 32) & 0xFFFFFFFF,
                    ct2d->gpu_cmd.params[4] & 0xFFFFFFFF,
                },
                .block_dim = {
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
                !ct2d->gpu_cmd.functions[func_id]) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
                break;
            }
            err = hetgpu_get_param_layout(
                hetgpu, ct2d->gpu_cmd.functions[func_id], &layout);
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
            cuda_result = hetgpu_cuda_graph_exec_kernel_node_set_params(
                hetgpu, ct2d->gpu_cmd.graph_execs[graph_exec_id],
                ct2d->gpu_cmd.graph_nodes[graph_node_id],
                ct2d->gpu_cmd.functions[func_id], &config, args);
            ct2d->gpu_cmd.cmd_result = cuda_result;
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
                if (ct2d->gpu_cmd.functions[i] == params.func) {
                    function_id = i;
                    break;
                }
            }
            if (function_id == UINT32_MAX) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
                break;
            }
            err = hetgpu_get_param_layout(hetgpu, params.func, &layout);
            if (err != HETGPU_SUCCESS ||
                layout->num_args > ARRAY_SIZE(param_wires)) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
                break;
            }
            wire.num_args = layout->num_args;
            param_extent = layout->extent;
            size_t param_wires_size = wire.num_args * sizeof(*param_wires);
            if (ct2d->gpu_cmd.cmd_result != CXL_GPU_SUCCESS ||
                sizeof(wire) > ct2d->gpu_cmd.data_size ||
                param_wires_size > ct2d->gpu_cmd.data_size - sizeof(wire) ||
                param_extent > ct2d->gpu_cmd.data_size - sizeof(wire) - param_wires_size) {
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
                memcpy(ct2d->gpu_cmd.data + sizeof(wire) + param_wires_size + param_offset,
                       params.kernel_params[i], param_size);
            }
            if (ct2d->gpu_cmd.cmd_result == CXL_GPU_SUCCESS)
                memcpy(ct2d->gpu_cmd.data + sizeof(wire), param_wires, param_wires_size);
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

            if (graph_exec_id_raw > UINT32_MAX ||
                graph_exec_id_raw >= ct2d->gpu_cmd.num_graph_execs ||
                !ct2d->gpu_cmd.graph_execs[graph_exec_id_raw] ||
                !cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[1],
                                            &stream)) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
                break;
            }
            ct2d->gpu_cmd.cmd_result = hetgpu_cuda_graph_launch(
                hetgpu, ct2d->gpu_cmd.graph_execs[graph_exec_id_raw], stream);
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
            int cuda_result = hetgpu_cuda_graph_destroy(
                hetgpu, ct2d->gpu_cmd.graphs[graph_id_raw]);
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

            if (graph_id_raw > UINT32_MAX ||
                graph_id_raw >= ct2d->gpu_cmd.num_graphs ||
                !ct2d->gpu_cmd.graphs[graph_id_raw]) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
                break;
            }
            if (ct2d->gpu_cmd.params[2] > 1 ||
                ct2d->gpu_cmd.params[3] > 1 ||
                (!want_log && buffer_size) ||
                buffer_size > ct2d->gpu_cmd.data_size) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
                break;
            }
            if (!cxl_type2_reserve_gpu_handle(
                    &ct2d->gpu_cmd.graph_execs,
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
                        &ct2d->gpu_cmd.graph_nodes,
                        &ct2d->gpu_cmd.graph_nodes_capacity,
                        &ct2d->gpu_cmd.num_graph_nodes, error_node,
                        &error_node_id)) {
                    ct2d->gpu_cmd.results[1] = error_node_id;
                } else {
                    qemu_log("CXL Type2: failed to register graph instantiate error node\n");
                }
            }
            if (cuda_result == CXL_GPU_SUCCESS) {
                if (!graph_exec) {
                    ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
                    break;
                }
                uint32_t graph_exec_id = ct2d->gpu_cmd.num_graph_execs;
                ct2d->gpu_cmd.graph_execs[graph_exec_id] = graph_exec;
                ct2d->gpu_cmd.results[0] = graph_exec_id;
                ct2d->gpu_cmd.num_graph_execs++;
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
            if (ct2d->gpu_cmd.params[2] > 1 ||
                (!want_nodes && requested) ||
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
                hetgpu, ct2d->gpu_cmd.graphs[graph_id_raw],
                want_nodes ? nodes : NULL, &count);
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
                        &ct2d->gpu_cmd.graph_nodes,
                        &ct2d->gpu_cmd.graph_nodes_capacity,
                        &ct2d->gpu_cmd.num_graph_nodes, nodes[i], &node_id)) {
                    ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_OUT_OF_MEMORY;
                    break;
                }
                node_ids[i] = node_id;
            }
            if (ct2d->gpu_cmd.cmd_result == CXL_GPU_SUCCESS) {
                memcpy(ct2d->gpu_cmd.data, node_ids,
                       count * sizeof(*node_ids));
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
                hetgpu, ct2d->gpu_cmd.graph_nodes[graph_node_id_raw],
                &node_type);
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
                if (!cxl_type2_register_gpu_handle(&ct2d->gpu_cmd.link_states,
                        &ct2d->gpu_cmd.link_states_capacity,
                        &ct2d->gpu_cmd.num_link_states, link_state, &id)) {
                    (void)hetgpu_cuda_link_destroy(hetgpu, link_state);
                    ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_OUT_OF_MEMORY;
                } else {
                    ct2d->gpu_cmd.results[0] = id;
                }
            }
        }
        break;

    case CXL_GPU_CMD_LINK_ADD_DATA:
        {
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
                hetgpu, ct2d->gpu_cmd.link_states[id],
                (int)ct2d->gpu_cmd.params[1], ct2d->gpu_cmd.data, input_size,
                name_size ? (char *)ct2d->gpu_cmd.data + input_size : NULL);
        }
        break;

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

    case CXL_GPU_CMD_LINK_DESTROY:
        {
            uint64_t id = ct2d->gpu_cmd.params[0];

            if (id > UINT32_MAX || id >= ct2d->gpu_cmd.num_link_states ||
                !ct2d->gpu_cmd.link_states[id]) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
                break;
            }
            int result = hetgpu_cuda_link_destroy(
                hetgpu, ct2d->gpu_cmd.link_states[id]);
            ct2d->gpu_cmd.cmd_result = result;
            if (result == CXL_GPU_SUCCESS)
                ct2d->gpu_cmd.link_states[id] = NULL;
        }
        break;

    case CXL_GPU_CMD_CTX_GET_LIMIT:
        {
            size_t value = 0;
            int result = hetgpu_cuda_ctx_get_limit(
                hetgpu, &value, (int)ct2d->gpu_cmd.params[0]);
            ct2d->gpu_cmd.cmd_result = result;
            if (result == CXL_GPU_SUCCESS)
                ct2d->gpu_cmd.results[0] = value;
        }
        break;

    case CXL_GPU_CMD_DEVICE_CAN_ACCESS_PEER:
        {
            int can_access = 0;
            int result = hetgpu_cuda_device_can_access_peer(
                hetgpu, &can_access, (int)ct2d->gpu_cmd.params[0],
                (int)ct2d->gpu_cmd.params[1]);
            ct2d->gpu_cmd.cmd_result = result;
            if (result == CXL_GPU_SUCCESS)
                ct2d->gpu_cmd.results[0] = can_access;
        }
        break;

    case CXL_GPU_CMD_CTX_ENABLE_PEER:
        ct2d->gpu_cmd.cmd_result = hetgpu_cuda_ctx_enable_peer_access(
            hetgpu, hetgpu->context, (unsigned int)ct2d->gpu_cmd.params[1]);
        break;

    case CXL_GPU_CMD_CTX_DISABLE_PEER:
        ct2d->gpu_cmd.cmd_result = hetgpu_cuda_ctx_disable_peer_access(
            hetgpu, hetgpu->context);
        break;

    case CXL_GPU_CMD_MEM_PREFETCH_ASYNC:
        {
            void *stream = NULL;
            if (!cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[3],
                                            &stream)) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
                break;
            }
            ct2d->gpu_cmd.cmd_result = hetgpu_cuda_mem_prefetch_async(
                hetgpu, ct2d->gpu_cmd.params[0], ct2d->gpu_cmd.params[1],
                (int)ct2d->gpu_cmd.params[2], stream);
        }
        break;

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
                if (!cxl_type2_stream_from_wire(ct2d,
                                                ct2d->gpu_cmd.params[6],
                                                &config.stream)) {
                    ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
                    break;
                }

                uint32_t num_args = (ct2d->gpu_cmd.params[4] >> 32) & 0xFF;
                size_t param_extent = ct2d->gpu_cmd.params[5];
                void *args[256];
                const HetGPUParamLayout *layout = NULL;
                uint64_t cache_hits_before = hetgpu->param_info_cache_hits;
                uint64_t cache_misses_before = hetgpu->param_info_cache_misses;
                uint64_t backend_queries_before = hetgpu->param_info_backend_queries;

                err = hetgpu_get_param_layout(
                    hetgpu, ct2d->gpu_cmd.functions[func_id], &layout);
                if (err != HETGPU_SUCCESS || num_args != layout->num_args ||
                    param_extent != layout->extent ||
                    num_args > ARRAY_SIZE(args) ||
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

                if (ct2d->paired_case.qemu_cuda_calls_enabled) qemu_log(
                    "CXL TYPE2 TRACE function_param_layout event=launch "
                    "layer=qemu function_id=%u args=%u extent=%zu hits=%" PRIu64
                    " misses=%" PRIu64 " backend_queries=%" PRIu64 "\n",
                    func_id, num_args, param_extent,
                    hetgpu->param_info_cache_hits - cache_hits_before,
                    hetgpu->param_info_cache_misses - cache_misses_before,
                    hetgpu->param_info_backend_queries - backend_queries_before);

                err = hetgpu_launch_kernel(hetgpu,
                                           ct2d->gpu_cmd.functions[func_id],
                                           &config, args, num_args);
                if (err != HETGPU_SUCCESS) {
                    ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_LAUNCH_FAILED;
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
                if (!cxl_type2_register_gpu_handle(&ct2d->gpu_cmd.streams,
                        &ct2d->gpu_cmd.streams_capacity,
                        &ct2d->gpu_cmd.num_streams, stream, &id)) {
                    (void)hetgpu_cuda_stream_destroy(hetgpu, stream);
                    ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_OUT_OF_MEMORY;
                } else {
                    ct2d->gpu_cmd.results[0] = id;
                }
            }
        }
        break;

    case CXL_GPU_CMD_STREAM_DESTROY:
        {
            uint64_t id = ct2d->gpu_cmd.params[0];
            if (id > UINT32_MAX || id >= ct2d->gpu_cmd.num_streams ||
                !ct2d->gpu_cmd.streams[id]) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
                break;
            }
            void *stream = ct2d->gpu_cmd.streams[id];
            int result = hetgpu_cuda_stream_synchronize(hetgpu, stream);
            if (result == CXL_GPU_SUCCESS) {
                result = cxl_type2_release_pending_htod(
                    ct2d, stream, false, UINT64_MAX, "stream-destroy");
            }
            if (result == CXL_GPU_SUCCESS) {
                result = hetgpu_cuda_stream_destroy(hetgpu, stream);
            }
            ct2d->gpu_cmd.cmd_result = result;
            if (result == CXL_GPU_SUCCESS)
                ct2d->gpu_cmd.streams[id] = NULL;
        }
        break;

    case CXL_GPU_CMD_STREAM_SYNC:
        {
            void *stream = NULL;
            if (!cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[0],
                                            &stream)) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
                break;
            }
            ct2d->gpu_cmd.cmd_result = hetgpu_cuda_stream_synchronize(hetgpu,
                                                                      stream);
            if (ct2d->gpu_cmd.cmd_result == CXL_GPU_SUCCESS) {
                int release_result =
                    cxl_type2_release_pending_htod(ct2d, stream, false,
                                                   UINT64_MAX, "stream-sync");
                if (release_result != CXL_GPU_SUCCESS) {
                    ct2d->gpu_cmd.cmd_result = release_result;
                }
            }
        }
        break;

    case CXL_GPU_CMD_STREAM_WAIT_EVENT:
        {
            void *stream = NULL;
            uint64_t event_id = ct2d->gpu_cmd.params[1];
            if (!cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[0],
                                            &stream) ||
                event_id > UINT32_MAX || event_id >= ct2d->gpu_cmd.num_events ||
                !ct2d->gpu_cmd.events[event_id]) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
                break;
            }
            ct2d->gpu_cmd.cmd_result = hetgpu_cuda_stream_wait_event(
                hetgpu, stream, ct2d->gpu_cmd.events[event_id],
                (unsigned int)ct2d->gpu_cmd.params[2]);
        }
        break;

    case CXL_GPU_CMD_STREAM_WAIT_VALUE32:
        {
            void *stream = NULL;
            if (!cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[0],
                                            &stream)) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
                break;
            }
            ct2d->gpu_cmd.cmd_result = hetgpu_cuda_stream_wait_value32(
                hetgpu, stream, ct2d->gpu_cmd.params[1],
                (uint32_t)ct2d->gpu_cmd.params[2],
                (unsigned int)ct2d->gpu_cmd.params[3]);
        }
        break;

    case CXL_GPU_CMD_STREAM_BATCH_MEM_OP:
        {
            void *stream = NULL;
            unsigned int count = ct2d->gpu_cmd.params[1];
            if (!cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[0],
                                            &stream)) {
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
        }
        break;

    case CXL_GPU_CMD_STREAM_GET_CTX:
        {
            void *stream = NULL;
            void *context = NULL;
            if (!cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[0],
                                            &stream)) {
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
        }
        break;

    case CXL_GPU_CMD_STREAM_BEGIN_CAPTURE:
        {
            void *stream = NULL;
            if (!cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[0],
                                            &stream)) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
                break;
            }
            ct2d->gpu_cmd.cmd_result = hetgpu_cuda_stream_begin_capture(
                hetgpu, stream, (int)ct2d->gpu_cmd.params[1]);
        }
        break;

    case CXL_GPU_CMD_STREAM_END_CAPTURE:
        memset(ct2d->gpu_cmd.results, 0xff, sizeof(ct2d->gpu_cmd.results));
        {
            void *stream = NULL;
            void *graph = NULL;
            uint32_t graph_id;
            if (!cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[0],
                                            &stream)) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
                break;
            }
            int result = hetgpu_cuda_stream_end_capture(hetgpu, stream, &graph);
            ct2d->gpu_cmd.cmd_result = result;
            if (result == CXL_GPU_SUCCESS && graph) {
                if (!cxl_type2_register_gpu_handle(&ct2d->gpu_cmd.graphs,
                        &ct2d->gpu_cmd.graphs_capacity,
                        &ct2d->gpu_cmd.num_graphs, graph, &graph_id)) {
                    (void)hetgpu_cuda_graph_destroy(hetgpu, graph);
                    ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_OUT_OF_MEMORY;
                } else {
                    ct2d->gpu_cmd.results[0] = graph_id;
                }
            }
        }
        break;

    case CXL_GPU_CMD_STREAM_IS_CAPTURING:
        {
            void *stream = NULL;
            int status = 0;
            if (!cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[0],
                                            &stream)) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
                break;
            }
            int result = hetgpu_cuda_stream_is_capturing(hetgpu, stream, &status);
            ct2d->gpu_cmd.cmd_result = result;
            if (result == CXL_GPU_SUCCESS)
                ct2d->gpu_cmd.results[0] = status;
        }
        break;

    case CXL_GPU_CMD_STREAM_GET_CAPTURE_INFO:
        memset(ct2d->gpu_cmd.results, 0xff, sizeof(ct2d->gpu_cmd.results));
        {
            void *stream = NULL;
            void *graph = NULL;
            void **dependencies = NULL;
            size_t count = 0;
            int status = 0;
            uint64_t capture_id = 0;
            if (!cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[0],
                                            &stream)) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
                break;
            }
            int result = hetgpu_cuda_stream_get_capture_info(
                hetgpu, stream, &status, &capture_id, &graph, &dependencies,
                &count);
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
                if (!cxl_type2_register_gpu_handle(&ct2d->gpu_cmd.graphs,
                        &ct2d->gpu_cmd.graphs_capacity,
                        &ct2d->gpu_cmd.num_graphs, graph, &graph_id)) {
                    ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_OUT_OF_MEMORY;
                    break;
                }
                ct2d->gpu_cmd.results[3] = graph_id;
            }
            uint64_t *ids = (uint64_t *)ct2d->gpu_cmd.data;
            for (size_t i = 0; i < count; i++) {
                uint32_t node_id;
                if (!dependencies || !cxl_type2_register_gpu_handle(
                        &ct2d->gpu_cmd.graph_nodes,
                        &ct2d->gpu_cmd.graph_nodes_capacity,
                        &ct2d->gpu_cmd.num_graph_nodes, dependencies[i],
                        &node_id)) {
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
                if (!cxl_type2_register_gpu_handle(&ct2d->gpu_cmd.events,
                        &ct2d->gpu_cmd.events_capacity,
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
    case CXL_GPU_CMD_EVENT_SYNC:
        {
            uint64_t id = ct2d->gpu_cmd.params[0];
            if (id > UINT32_MAX || id >= ct2d->gpu_cmd.num_events ||
                !ct2d->gpu_cmd.events[id]) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
                break;
            }
            int result;
            void *event = ct2d->gpu_cmd.events[id];
            if (cmd == CXL_GPU_CMD_EVENT_DESTROY)
                result = hetgpu_cuda_event_destroy(hetgpu, event);
            else if (cmd == CXL_GPU_CMD_EVENT_QUERY)
                result = hetgpu_cuda_event_query(hetgpu, event);
            else
                result = hetgpu_cuda_event_synchronize(hetgpu, event);
            ct2d->gpu_cmd.cmd_result = result;
            if ((cmd == CXL_GPU_CMD_EVENT_QUERY ||
                 cmd == CXL_GPU_CMD_EVENT_SYNC) &&
                result == CXL_GPU_SUCCESS) {
                CXLType2EventHtoDMark *mark =
                    cxl_type2_event_htod_mark(ct2d, event, false);
                if (mark) {
                    int release_result = cxl_type2_release_pending_htod(
                        ct2d, mark->stream, false, mark->sequence,
                        cmd == CXL_GPU_CMD_EVENT_QUERY ? "event-query"
                                                       : "event-sync");
                    if (release_result != CXL_GPU_SUCCESS) {
                        ct2d->gpu_cmd.cmd_result = release_result;
                    }
                }
            }
            if (cmd == CXL_GPU_CMD_EVENT_DESTROY && result == CXL_GPU_SUCCESS) {
                cxl_type2_remove_event_htod_mark(ct2d, event);
                ct2d->gpu_cmd.events[id] = NULL;
            }
        }
        break;

    case CXL_GPU_CMD_EVENT_RECORD:
        {
            uint64_t event_id = ct2d->gpu_cmd.params[0];
            void *stream = NULL;
            if (event_id > UINT32_MAX || event_id >= ct2d->gpu_cmd.num_events ||
                !ct2d->gpu_cmd.events[event_id] ||
                !cxl_type2_stream_from_wire(ct2d, ct2d->gpu_cmd.params[1],
                                            &stream)) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
                break;
            }
            ct2d->gpu_cmd.cmd_result = hetgpu_cuda_event_record(
                hetgpu, ct2d->gpu_cmd.events[event_id], stream);
            if (ct2d->gpu_cmd.cmd_result == CXL_GPU_SUCCESS) {
                CXLType2EventHtoDMark *mark = cxl_type2_event_htod_mark(
                    ct2d, ct2d->gpu_cmd.events[event_id], true);
                mark->stream = stream;
                mark->sequence =
                    cxl_type2_latest_pending_htod_sequence(ct2d, stream);
            }
        }
        break;

    case CXL_GPU_CMD_EVENT_ELAPSED_TIME:
        {
            uint64_t start = ct2d->gpu_cmd.params[0];
            uint64_t end = ct2d->gpu_cmd.params[1];
            float milliseconds = 0.0f;
            uint32_t bits;
            if (start > UINT32_MAX || end > UINT32_MAX ||
                start >= ct2d->gpu_cmd.num_events ||
                end >= ct2d->gpu_cmd.num_events ||
                !ct2d->gpu_cmd.events[start] || !ct2d->gpu_cmd.events[end]) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_HANDLE;
                break;
            }
            int result = hetgpu_cuda_event_elapsed_time(
                hetgpu, &milliseconds, ct2d->gpu_cmd.events[start],
                ct2d->gpu_cmd.events[end]);
            ct2d->gpu_cmd.cmd_result = result;
            if (result == CXL_GPU_SUCCESS) {
                memcpy(&bits, &milliseconds, sizeof(bits));
                ct2d->gpu_cmd.results[0] = bits;
            }
        }
        break;

    /* Bulk transfer commands - optimized for large memory operations */
    case CXL_GPU_CMD_BULK_HTOD:
        /* Bulk host-to-device transfer using BAR4 region */
        {
            uint64_t bar4_offset = ct2d->gpu_cmd.params[0];  /* Offset in BAR4 */
            uint64_t dst_dev_ptr = ct2d->gpu_cmd.params[1];  /* Device destination */
            size_t xfer_size = ct2d->gpu_cmd.params[2];       /* Transfer size */

            if (xfer_size > CXL_GPU_BULK_TRANSFER_SIZE) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
                break;
            }

            if (hetgpu->initialized) {
                /* Get data from device memory region (BAR4/HDM) */
                uint8_t *mem = memory_region_get_ram_ptr(&ct2d->device_mem);
                if (mem && bar4_offset + xfer_size <= ct2d->device_mem_size &&
                    cxl_type2_fabric_access_allowed(ct2d, bar4_offset,
                                                    xfer_size, false, false)) {
                    if (!cxl_type2_bulk_memsim_access(ct2d, CXL_OP_READ,
                                                      bar4_offset, xfer_size,
                                                      mem + bar4_offset)) {
                        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
                        break;
                    }
                    err = hetgpu_memcpy_htod(hetgpu, dst_dev_ptr,
                                             mem + bar4_offset, xfer_size);
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

    case CXL_GPU_CMD_BULK_DTOH:
        /* Bulk device-to-host transfer using BAR4 region */
        {
            uint64_t src_dev_ptr = ct2d->gpu_cmd.params[0];   /* Device source */
            uint64_t bar4_offset = ct2d->gpu_cmd.params[1];   /* Offset in BAR4 */
            size_t xfer_size = ct2d->gpu_cmd.params[2];        /* Transfer size */

            if (xfer_size > CXL_GPU_BULK_TRANSFER_SIZE) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
                break;
            }

            if (hetgpu->initialized) {
                /* Write data to device memory region (BAR4/HDM) */
                uint8_t *mem = memory_region_get_ram_ptr(&ct2d->device_mem);
                if (mem && bar4_offset + xfer_size <= ct2d->device_mem_size &&
                    cxl_type2_fabric_access_allowed(ct2d, bar4_offset,
                                                    xfer_size, true, false)) {
                    err = hetgpu_memcpy_dtoh(hetgpu, mem + bar4_offset,
                                             src_dev_ptr, xfer_size);
                    if (err == HETGPU_SUCCESS) {
                        if (!cxl_type2_bulk_memsim_access(
                                ct2d, CXL_OP_WRITE, bar4_offset, xfer_size,
                                mem + bar4_offset)) {
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
            uint64_t src_dev_ptr = ct2d->gpu_cmd.params[0];   /* Source device ptr */
            uint64_t dst_dev_ptr = ct2d->gpu_cmd.params[1];   /* Dest device ptr */
            size_t xfer_size = ct2d->gpu_cmd.params[2];        /* Transfer size */

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
                cxl_type2_memsim_request(ct2d, CXL_OP_FENCE, flush_addr,
                                         flush_size, NULL, NULL);
                /* Invalidate local cache entries */
                for (uint64_t addr = flush_addr; addr < flush_addr + flush_size; addr += 64) {
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
        /* GPU -> Type3 transfer: params[0]=peer_id, params[1]=gpu_off, params[2]=mem_off, params[3]=size */
        {
            uint32_t t3_peer_id = ct2d->gpu_cmd.params[0];
            uint64_t gpu_offset = ct2d->gpu_cmd.params[1];
            uint64_t mem_offset = ct2d->gpu_cmd.params[2];
            uint64_t xfer_size = ct2d->gpu_cmd.params[3];
            uint32_t flags = CXL_P2P_FLAG_COHERENT;

            int ret = cxl_p2p_gpu_to_mem(&ct2d->p2p_engine, t3_peer_id,
                                          gpu_offset, mem_offset, xfer_size, flags);
            ct2d->gpu_cmd.cmd_result = (ret == 0) ? CXL_GPU_SUCCESS
                                                   : CXL_GPU_ERROR_OUT_OF_MEMORY;
        }
        break;

    case CXL_GPU_CMD_P2P_MEM_TO_GPU:
        /* Type3 -> GPU transfer: params[0]=peer_id, params[1]=mem_off, params[2]=gpu_off, params[3]=size */
        {
            uint32_t t3_peer_id = ct2d->gpu_cmd.params[0];
            uint64_t mem_offset = ct2d->gpu_cmd.params[1];
            uint64_t gpu_offset = ct2d->gpu_cmd.params[2];
            uint64_t xfer_size = ct2d->gpu_cmd.params[3];
            uint32_t flags = CXL_P2P_FLAG_COHERENT;

            int ret = cxl_p2p_mem_to_gpu(&ct2d->p2p_engine, t3_peer_id,
                                          mem_offset, gpu_offset, xfer_size, flags);
            ct2d->gpu_cmd.cmd_result = (ret == 0) ? CXL_GPU_SUCCESS
                                                   : CXL_GPU_ERROR_OUT_OF_MEMORY;
        }
        break;

    case CXL_GPU_CMD_P2P_MEM_TO_MEM:
        /* Type3 -> Type3 transfer: params[0]=src_peer, params[1]=dst_peer, params[2]=src_off, params[3]=dst_off, params[4]=size */
        {
            uint32_t src_peer_id = ct2d->gpu_cmd.params[0];
            uint32_t dst_peer_id = ct2d->gpu_cmd.params[1];
            uint64_t src_offset = ct2d->gpu_cmd.params[2];
            uint64_t dst_offset = ct2d->gpu_cmd.params[3];
            uint64_t xfer_size = ct2d->gpu_cmd.params[4];
            uint32_t flags = CXL_P2P_FLAG_COHERENT;

            int ret = cxl_p2p_mem_to_mem(&ct2d->p2p_engine, src_peer_id, dst_peer_id,
                                          src_offset, dst_offset, xfer_size, flags);
            ct2d->gpu_cmd.cmd_result = (ret == 0) ? CXL_GPU_SUCCESS
                                                   : CXL_GPU_ERROR_OUT_OF_MEMORY;
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
    case CXL_GPU_CMD_COHERENT_ALLOC:
        {
            uint64_t alloc_size = ct2d->gpu_cmd.params[0];
            int64_t offset = cxl_coherent_pool_alloc(ct2d, alloc_size);
            if (offset >= 0) {
                ct2d->gpu_cmd.results[0] = (uint64_t)offset;
                ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
            } else {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_OUT_OF_MEMORY;
            }
        }
        break;

    case CXL_GPU_CMD_COHERENT_FREE:
        {
            uint64_t free_offset = ct2d->gpu_cmd.params[0];
            if (cxl_coherent_pool_free(ct2d, free_offset) == 0) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
            } else {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
            }
        }
        break;

    case CXL_GPU_CMD_COHERENT_GET_INFO:
        {
            ct2d->gpu_cmd.results[0] = ct2d->coherent_pool.base_offset;
            ct2d->gpu_cmd.results[1] = ct2d->coherent_pool.size;
            ct2d->gpu_cmd.results[2] = ct2d->coherent_pool.size -
                                        ct2d->coherent_pool.used;
            ct2d->gpu_cmd.results[3] = ct2d->bar_coherency.snoop_filter_size;
            ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
        }
        break;

    case CXL_GPU_CMD_COHERENT_FENCE:
        {
            /* Memory fence - ensure all pending coherency ops complete */
            cxl_bar_memory_fence(&ct2d->bar_coherency, CXL_DOMAIN_CPU);
            cxl_bar_process_back_invalidations(ct2d);
            ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
        }
        break;

    /* ---- Device-biased directory commands ---- */
    case CXL_GPU_CMD_SET_BIAS:
        {
            uint64_t bias_addr = ct2d->gpu_cmd.params[0];
            uint64_t bias_size = ct2d->gpu_cmd.params[1];
            uint8_t bias_mode = (uint8_t)ct2d->gpu_cmd.params[2];
            if (bias_mode > CXL_BIAS_DEVICE) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
            } else {
                cxl_bar_set_bias(&ct2d->bar_coherency, bias_addr, bias_size,
                                 bias_mode);
                ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
            }
        }
        break;

    case CXL_GPU_CMD_GET_BIAS:
        {
            uint64_t query_addr = ct2d->gpu_cmd.params[0];
            ct2d->gpu_cmd.results[0] = cxl_bar_get_bias(&ct2d->bar_coherency,
                                                          query_addr);
            ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
        }
        break;

    case CXL_GPU_CMD_BIAS_FLIP:
        {
            uint64_t flip_addr = ct2d->gpu_cmd.params[0];
            uint64_t flip_size = ct2d->gpu_cmd.params[1];
            uint8_t new_bias = (uint8_t)ct2d->gpu_cmd.params[2];
            if (new_bias > CXL_BIAS_DEVICE) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
            } else {
                cxl_bar_bias_flip(ct2d, flip_addr, flip_size, new_bias);
                ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
            }
        }
        break;

    /* ---- Coherency statistics commands ---- */
    case CXL_GPU_CMD_COH_GET_STATS:
        {
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
        }
        break;

    case CXL_GPU_CMD_COH_RESET_STATS:
        {
            memset(&ct2d->bar_coherency.stats, 0,
                   sizeof(ct2d->bar_coherency.stats));
            ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
        }
        break;

    /* ---- DCD / GFAM / MH-SLD fabric-memory commands ---- */
    case CXL_GPU_CMD_DCD_ADD:
        {
            uint64_t base = ct2d->gpu_cmd.params[0];
            uint64_t add_size = ct2d->gpu_cmd.params[1];
            uint64_t tag = ct2d->gpu_cmd.params[2];
            uint64_t out_base = 0;
            uint64_t out_size = 0;
            uint64_t out_tag = 0;
            int ret;

            ret = cxl_type2_dcd_add(ct2d, base, add_size, tag, &out_base,
                                    &out_size, &out_tag);
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
        }
        break;

    case CXL_GPU_CMD_DCD_RELEASE:
        {
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
        }
        break;

    case CXL_GPU_CMD_DCD_GET_INFO:
        {
            qemu_mutex_lock(&ct2d->dcd.lock);
            ct2d->gpu_cmd.results[0] = ct2d->device_mem_size;
            ct2d->gpu_cmd.results[1] = ct2d->dcd.enabled ?
                                       ct2d->dcd.allocated :
                                       ct2d->device_mem_size;
            ct2d->gpu_cmd.results[2] = ct2d->dcd.enabled ?
                                       ct2d->device_mem_size -
                                       ct2d->dcd.allocated : 0;
            ct2d->gpu_cmd.results[3] =
                cxl_type2_dcd_active_extents_locked(ct2d);
            if (ct2d->gpu_cmd.data_size >=
                sizeof(ct2d->dcd.extents)) {
                memcpy(ct2d->gpu_cmd.data, ct2d->dcd.extents,
                       sizeof(ct2d->dcd.extents));
            }
            qemu_mutex_unlock(&ct2d->dcd.lock);
            ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
        }
        break;

    case CXL_GPU_CMD_GFAM_GRANT:
        {
            uint32_t host = ct2d->gpu_cmd.params[0];
            uint64_t base = ct2d->gpu_cmd.params[1];
            uint64_t grant_size = ct2d->gpu_cmd.params[2];
            uint32_t permissions = ct2d->gpu_cmd.params[3];
            int ret;

            ret = cxl_type2_gfam_grant(ct2d, host, base, grant_size,
                                       permissions);
            ct2d->gpu_cmd.cmd_result = ret == 0 ? CXL_GPU_SUCCESS :
                                                  CXL_GPU_ERROR_INVALID_VALUE;
        }
        break;

    case CXL_GPU_CMD_GFAM_REVOKE:
        {
            uint32_t host = ct2d->gpu_cmd.params[0];
            uint64_t base = ct2d->gpu_cmd.params[1];
            uint64_t revoke_size = ct2d->gpu_cmd.params[2];
            int ret;

            ret = cxl_type2_gfam_revoke(ct2d, host, base, revoke_size);
            ct2d->gpu_cmd.cmd_result = ret == 0 ? CXL_GPU_SUCCESS :
                                                  CXL_GPU_ERROR_INVALID_VALUE;
        }
        break;

    case CXL_GPU_CMD_GFAM_GET_INFO:
        {
            qemu_mutex_lock(&ct2d->gfam.lock);
            ct2d->gpu_cmd.results[0] = ct2d->gfam.num_hosts;
            ct2d->gpu_cmd.results[1] =
                cxl_type2_gfam_active_mappings_locked(ct2d);
            ct2d->gpu_cmd.results[2] = ct2d->gfam.allowed_accesses;
            ct2d->gpu_cmd.results[3] = ct2d->gfam.denied_accesses;
            if (ct2d->gpu_cmd.data_size >=
                sizeof(ct2d->gfam.mappings)) {
                memcpy(ct2d->gpu_cmd.data, ct2d->gfam.mappings,
                       sizeof(ct2d->gfam.mappings));
            }
            qemu_mutex_unlock(&ct2d->gfam.lock);
            ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
        }
        break;

    case CXL_GPU_CMD_MHSLD_GET_INFO:
        {
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
        }
        break;

    case CXL_GPU_CMD_MHSLD_SET_HEAD:
        {
            uint32_t head = ct2d->gpu_cmd.params[0];

            if (head >= ct2d->mhsld.num_heads) {
                ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
            } else {
                ct2d->mhsld.local_head_id = head;
                ct2d->gpu_cmd.results[0] = head;
                ct2d->gpu_cmd.cmd_result = CXL_GPU_SUCCESS;
            }
        }
        break;

    default:
        ct2d->gpu_cmd.cmd_result = CXL_GPU_ERROR_INVALID_VALUE;
        break;
    }

complete:
    hetgpu_cuda_trace_set_call_id(0);
    ct2d->gpu_cmd.cmd_status = CXL_GPU_CMD_STATUS_COMPLETE;
    int64_t trace_end_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);
    int64_t trace_duration_ns = trace_end_ns - trace_start_ns;
    if (ct2d->paired_case.qemu_cuda_calls_enabled) qemu_log("CXL TYPE2 TRACE cmd_end seq=%" PRIu64
             " call_id=0x%016" PRIx64
             " cmd=0x%x host_ns=%" PRId64 " result=%u duration_ns=%" PRId64
             " context_generation=%" PRIu64
             " context_binding_hits=%" PRIu64
             " context_binding_misses=%" PRIu64 "\n",
             trace_sequence, ct2d->gpu_cmd.call_id, cmd, trace_end_ns,
             ct2d->gpu_cmd.cmd_result, trace_duration_ns,
             hetgpu->context_generation, hetgpu->context_binding_hits,
             hetgpu->context_binding_misses);
    if (ct2d->paired_case.qemu_cuda_calls_enabled) qemu_log_mask(LOG_GUEST_ERROR,
                  "CXL GPU: cmd 0x%x done, result=%u results[0]=0x%lx\n",
                  cmd, ct2d->gpu_cmd.cmd_result,
                  (unsigned long)ct2d->gpu_cmd.results[0]);
}

static uint64_t cxl_type2_gpu_cmd_read(void *opaque, hwaddr addr, unsigned size)
{
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
    case CXL_GPU_REG_CMD_STATUS:
        value = ct2d->gpu_cmd.cmd_status;
        // fprintf(stderr, "CXL GPU: read CMD_STATUS = %lu\n", (unsigned long)value);
        break;
    case CXL_GPU_REG_CMD_RESULT:
        value = ct2d->gpu_cmd.cmd_result;
        break;
    case CXL_GPU_REG_CALL_ID:
        value = ct2d->gpu_cmd.call_id;
        break;
    case CXL_GPU_REG_RESULT0:
        value = ct2d->gpu_cmd.results[0];
        // fprintf(stderr, "CXL GPU: read RESULT0 = 0x%lx\n", (unsigned long)value);
        break;
    case CXL_GPU_REG_RESULT1:
        value = ct2d->gpu_cmd.results[1];
        break;
    case CXL_GPU_REG_RESULT2:
        value = ct2d->gpu_cmd.results[2];
        break;
    case CXL_GPU_REG_RESULT3:
        value = ct2d->gpu_cmd.results[3];
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
        value = ct2d->dcd.enabled ? ct2d->dcd.allocated :
                                    ct2d->device_mem_size;
        break;
    case CXL_GPU_REG_DCD_FREE:
        value = ct2d->dcd.enabled ? ct2d->device_mem_size -
                                    ct2d->dcd.allocated : 0;
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
        if (addr >= CXL_GPU_REG_DEV_NAME &&
            addr < CXL_GPU_REG_DEV_NAME + 64) {
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

static void cxl_type2_gpu_cmd_write(void *opaque, hwaddr addr,
                                     uint64_t value, unsigned size)
{
    CXLType2State *ct2d = opaque;

    switch (addr) {
    case CXL_GPU_REG_CMD:
        /* Execute command */
        cxl_type2_gpu_execute_cmd(ct2d, value);
        break;
    case CXL_GPU_REG_CALL_ID:
        ct2d->gpu_cmd.call_id = value;
        break;
    case CXL_GPU_REG_PARAM0:
        ct2d->gpu_cmd.params[0] = value;
        break;
    case CXL_GPU_REG_PARAM1:
        ct2d->gpu_cmd.params[1] = value;
        break;
    case CXL_GPU_REG_PARAM2:
        ct2d->gpu_cmd.params[2] = value;
        break;
    case CXL_GPU_REG_PARAM3:
        ct2d->gpu_cmd.params[3] = value;
        break;
    case CXL_GPU_REG_PARAM4:
        ct2d->gpu_cmd.params[4] = value;
        break;
    case CXL_GPU_REG_PARAM5:
        ct2d->gpu_cmd.params[5] = value;
        break;
    case CXL_GPU_REG_PARAM6:
        ct2d->gpu_cmd.params[6] = value;
        break;
    case CXL_GPU_REG_PARAM7:
        ct2d->gpu_cmd.params[7] = value;
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
                                       void *opaque)
{
    CXLType2State *ct2d = opaque;
    bool old_enabled = FIELD_EX32(old_ctrl, CXL_BI_DECODER_CTRL, BI_ENABLE);
    bool new_enabled = FIELD_EX32(new_ctrl, CXL_BI_DECODER_CTRL, BI_ENABLE);

    if (old_enabled == new_enabled) {
        return;
    }

    ct2d->bi_enabled = new_enabled;
    cxl_type2_memsim_request(ct2d,
                             new_enabled ? CXL_OP_BI_ENABLE :
                                           CXL_OP_BI_DISABLE,
                             0, ct2d->device_mem_size, NULL, NULL);
    qemu_log("CXL Type2: BI %s via decoder control\n",
             new_enabled ? "enabled" : "disabled");
}

static void build_dvsecs(CXLType2State *ct2d)
{
    CXLComponentState *cxl_cstate = &ct2d->cxl_cstate;
    uint8_t *dvsec;

    /* Type 2 Device DVSEC - includes both cache and memory capabilities */
    dvsec = (uint8_t *)&(CXLDVSECDevice){
        .cap = 0x1f,  /* Cache+, IO+, Mem+, Mem HWInit+, HDMCount=1 */
        .ctrl = 0x7,  /* Cache+, IO+, Mem+ enabled */
        .status = 0,
        .ctrl2 = 0,
        .status2 = 0x2,
        .lock = 0,
        .cap2 = (ct2d->cache_size >> 20) & 0xFFFF,  /* Cache size in MB */
        .range1_size_hi = ct2d->cache_size >> 32,
        .range1_size_lo = (ct2d->cache_size & 0xFFFFFFF0) | 0x3,  /* Cache: Valid, Active */
        .range1_base_hi = 0,
        .range1_base_lo = 0,
        .range2_size_hi = ct2d->device_mem_size >> 32,
        .range2_size_lo = (ct2d->device_mem_size & 0xFFFFFFF0) | 0x1,  /* Mem: Valid */
        .range2_base_hi = 0,
        .range2_base_lo = 0,
    };

    cxl_component_create_dvsec(cxl_cstate, CXL2_TYPE3_DEVICE,
                              PCIE_CXL_DEVICE_DVSEC_LENGTH,
                              PCIE_CXL_DEVICE_DVSEC,
                              PCIE_CXL31_DEVICE_DVSEC_REVID,
                              dvsec);

    /* Register Locator DVSEC
     * Type 2 devices only have component registers in BAR0
     * BAR2 is used for cache memory, not CXL device registers
     */
    dvsec = (uint8_t *)&(CXLDVSECRegisterLocator){
        .rsvd = 0,
        .reg0_base_lo = RBI_COMPONENT_REG | CXL_COMPONENT_REG_BAR_IDX,
        .reg0_base_hi = 0,
        .reg1_base_lo = RBI_EMPTY,  /* No device registers - Type 2 uses cache memory at BAR2 */
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

    cxl_component_create_dvsec(cxl_cstate, CXL2_TYPE3_DEVICE,
                              PCIE_CXL3_FLEXBUS_PORT_DVSEC_LENGTH,
                              PCIE_FLEXBUS_PORT_DVSEC,
                              PCIE_CXL3_FLEXBUS_PORT_DVSEC_REVID, dvsec);
}

/* ========================================================================
 * Device Lifecycle
 * ======================================================================== */

static void cxl_type2_reset(DeviceState *dev)
{
    CXLType2State *ct2d = CXL_TYPE2(dev);
    CXLComponentState *cxl_cstate = &ct2d->cxl_cstate;
    uint32_t *reg_state = cxl_cstate->crb.cache_mem_registers;
    uint32_t *write_msk = cxl_cstate->crb.cache_mem_regs_write_mask;

    pcie_cap_fill_link_ep_usp(PCI_DEVICE(dev), ct2d->width, ct2d->speed,
                              ct2d->flitmode);
    cxl_component_register_init_common(reg_state, write_msk,
                                       CXL2_TYPE3_DEVICE, ct2d->hdmdb);
    cxl_cstate->bi_control_write = cxl_type2_bi_control_write;
    cxl_cstate->bi_control_opaque = ct2d;
    ct2d->bi_enabled = false;

    /* Reset statistics */
    memset(&ct2d->stats, 0, sizeof(ct2d->stats));

    qemu_log("CXL Type2: Device reset\n");
}

static void cxl_type2_realize(PCIDevice *pci_dev, Error **errp)
{
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
                   "cache-size (%" PRIu64 ") is smaller than the BAR2 GPU command region (%u)",
                   ct2d->cache_size, CXL_GPU_CMD_REG_SIZE);
        return;
    }

    if (ct2d->paired_case.required) {
        if (!ct2d->paired_case.run_root ||
            ct2d->paired_case.run_root[0] != '/' ||
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
            error_setg(errp, "paired case Concordia limits require a positive "
                       "checkpoint interval when enabled and zero when disabled; "
                       "min allocation must be at least 4096 bytes");
            return;
        }
        ct2d->paired_case.next_epoch = 1;
        ct2d->paired_case.active_case = CXL_GPU_CASE_NONE;
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
                    PCI_BASE_ADDRESS_SPACE_MEMORY |
                    PCI_BASE_ADDRESS_MEM_TYPE_64,
                    &cxl_cstate->crb.component_registers);

    /* BAR2: Cache memory region (Type 1 feature) */
    memory_region_init_ram(&ct2d->cache_mem, OBJECT(ct2d),
                          "cxl-type2-cache", ct2d->cache_size, &local_err);
    if (local_err) {
        error_propagate(errp, local_err);
        return;
    }

    memory_region_init_io(&ct2d->cache_io, OBJECT(ct2d),
                         &cxl_type2_cache_ops, ct2d,
                         "cxl-type2-cache-io", ct2d->cache_size);

    memory_region_add_subregion_overlap(&ct2d->cache_mem, 0, &ct2d->cache_io, 1);

    /* The command data window is a payload mailbox, not a CXL cache access.
     * Expose it as RAM so guest bulk copies do not dispatch one device callback
     * per store through cache_io. */
    memory_region_init_ram(&ct2d->gpu_data_mem, OBJECT(ct2d),
                           "cxl-type2-gpu-data", CXL_GPU_DATA_SIZE,
                           &local_err);
    if (local_err) {
        error_propagate(errp, local_err);
        return;
    }
    memory_region_add_subregion_overlap(&ct2d->cache_mem,
                                        CXL_GPU_DATA_OFFSET,
                                        &ct2d->gpu_data_mem, 2);

    pci_register_bar(pci_dev, 2,
                    PCI_BASE_ADDRESS_SPACE_MEMORY |
                    PCI_BASE_ADDRESS_MEM_TYPE_64 |
                    PCI_BASE_ADDRESS_MEM_PREFETCH,
                    &ct2d->cache_mem);

    /* BAR4: Device-attached memory (Type 3 feature) */
    memory_region_init_ram(&ct2d->device_mem, OBJECT(ct2d),
                          "cxl-type2-device-mem", ct2d->device_mem_size, &local_err);
    if (local_err) {
        error_propagate(errp, local_err);
        return;
    }

    memory_region_init_io(&ct2d->device_mem_io, OBJECT(ct2d),
                         &cxl_type2_device_mem_ops, ct2d,
                         "cxl-type2-device-mem-io", ct2d->device_mem_size);

    memory_region_add_subregion_overlap(&ct2d->device_mem, 0, &ct2d->device_mem_io, 1);

    pci_register_bar(pci_dev, 4,
                    PCI_BASE_ADDRESS_SPACE_MEMORY |
                    PCI_BASE_ADDRESS_MEM_TYPE_64 |
                    PCI_BASE_ADDRESS_MEM_PREFETCH,
                    &ct2d->device_mem);

    /* Register BAR regions for enhanced coherency tracking */
    cxl_bar_coherency_add_region(&ct2d->bar_coherency, 2, 0, ct2d->cache_size,
                                  true,   /* GPU accessible */
                                  true);  /* CPU accessible */
    cxl_bar_coherency_add_region(&ct2d->bar_coherency, 4, 0, ct2d->device_mem_size,
                                  true,   /* GPU accessible */
                                  true);  /* CPU accessible */

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

    /* Set capabilities */
    ct2d->gpu_cmd.capabilities = CXL_GPU_CAP_BULK_TRANSFER |
                                 CXL_GPU_CAP_CACHE_COHERENT |
                                 CXL_GPU_CAP_COHERENT_POOL |
                                 CXL_GPU_CAP_DEVICE_BIAS;
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
        ct2d->coherent_pool.allocations = g_hash_table_new(g_int64_hash,
                                                            g_int64_equal);
        /* Initialize free list with single block spanning the whole pool */
        CXLCohFreeBlock *initial = g_new0(CXLCohFreeBlock, 1);
        initial->offset = ct2d->coherent_pool.base_offset;
        initial->size = coh_pool_size;
        initial->next = NULL;
        ct2d->coherent_pool.free_list = initial;
        qemu_mutex_init(&ct2d->coherent_pool.lock);

        qemu_log("CXL Type2: Coherent pool initialized: base=0x%lx size=%lu MB\n",
                 (unsigned long)ct2d->coherent_pool.base_offset,
                 (unsigned long)(coh_pool_size / MiB));
    }

    qemu_log("CXL Type2: GPU command interface enabled at BAR2 offset 0\n");
    qemu_log("CXL Type2: Data buffer size: %zu KB (optimized for large transfers)\n",
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

    qemu_log("CXL Type2: Device realized - Cache: %zu MB, DevMem: %zu MB\n",
             ct2d->cache_size / MiB, ct2d->device_mem_size / MiB);
}

static void cxl_type2_exit(PCIDevice *pci_dev)
{
    CXLType2State *ct2d = CXL_TYPE2(pci_dev);

    /* Disconnect from CXLMemSim */
    cxlmemsim_disconnect(ct2d);
    qemu_mutex_destroy(&ct2d->memsim.lock);

    if (ct2d->gpu_info.hetgpu_state.initialized) {
        (void)hetgpu_synchronize(&ct2d->gpu_info.hetgpu_state);
        (void)cxl_type2_release_pending_htod(ct2d, NULL, true, UINT64_MAX,
                                             "device-exit");
        (void)cxl_type2_clear_htod_staging_pool(ct2d);
    }

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

    /* Free bulk transfer region if allocated */
    if (ct2d->bulk_transfer_ptr) {
        g_free(ct2d->bulk_transfer_ptr);
        ct2d->bulk_transfer_ptr = NULL;
    }

    cxl_type2_fabric_features_cleanup(ct2d);

    qemu_log("CXL Type2: Device exit complete\n");
}

static const Property cxl_type2_props[] = {
    DEFINE_PROP_SIZE("cache-size", CXLType2State, cache_size,
                     CXL_TYPE2_DEFAULT_CACHE_SIZE),
    DEFINE_PROP_SIZE("mem-size", CXLType2State, device_mem_size,
                     CXL_TYPE2_DEFAULT_MEM_SIZE),
    DEFINE_PROP_SIZE("htod-staging-pool-size", CXLType2State,
                     htod_staging_pool_size, 0),
    DEFINE_PROP_UINT64("sn", CXLType2State, sn, 0),
    DEFINE_PROP_PCIE_LINK_SPEED("x-speed", CXLType2State,
                                speed, PCIE_LINK_SPEED_64),
    DEFINE_PROP_PCIE_LINK_WIDTH("x-width", CXLType2State,
                                width, PCIE_LINK_WIDTH_32),
    DEFINE_PROP_BOOL("x-256b-flit", CXLType2State, flitmode, true),
    DEFINE_PROP_BOOL("hdm-db", CXLType2State, hdmdb, true),
    DEFINE_PROP_STRING("cxlmemsim-addr", CXLType2State, memsim.server_addr),
    DEFINE_PROP_UINT16("cxlmemsim-port", CXLType2State, memsim.server_port, 9999),
    DEFINE_PROP_STRING("gpu-device", CXLType2State, gpu_info.vfio_device),
    DEFINE_PROP_BOOL("coherency-enabled", CXLType2State,
                     coherency.coherency_enabled, true),
    /* hetGPU backend configuration */
    DEFINE_PROP_UINT32("gpu-mode", CXLType2State, gpu_info.mode,
                       CXL_TYPE2_GPU_MODE_AUTO),
    DEFINE_PROP_STRING("hetgpu-lib", CXLType2State, gpu_info.hetgpu_lib_path),
    DEFINE_PROP_INT32("hetgpu-device", CXLType2State, gpu_info.hetgpu_device_index, 0),
    DEFINE_PROP_UINT32("hetgpu-backend", CXLType2State, gpu_info.hetgpu_backend,
                       HETGPU_BACKEND_AUTO),
    DEFINE_PROP_BOOL("paired-case-control", CXLType2State,
                     paired_case.required, false),
    DEFINE_PROP_STRING("paired-run-root", CXLType2State,
                       paired_case.run_root),
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
    DEFINE_PROP_UINT32("gfam-latency-ns", CXLType2State,
                       gfam.fabric_latency_ns, 150),
    DEFINE_PROP_UINT32("gfam-bandwidth-mbps", CXLType2State,
                       gfam.bandwidth_mbps, 32768),
    DEFINE_PROP_BOOL("mhsld", CXLType2State, mhsld.enabled, false),
    DEFINE_PROP_UINT32("mhsld-heads", CXLType2State, mhsld.num_heads, 1),
    DEFINE_PROP_UINT32("mhsld-head-id", CXLType2State, mhsld.local_head_id, 0),
    DEFINE_PROP_UINT32("mhsld-coh-latency-ns", CXLType2State,
                       mhsld.coherency_latency_ns, 200),
};

static void cxl_type2_class_init(ObjectClass *oc, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(oc);
    PCIDeviceClass *pc = PCI_DEVICE_CLASS(oc);

    pc->realize = cxl_type2_realize;
    pc->exit = cxl_type2_exit;
    pc->vendor_id = CXL_TYPE2_VENDOR_ID;
    pc->device_id = CXL_TYPE2_DEVICE_ID;
    pc->revision = 1;
    pc->class_id = PCI_CLASS_MEMORY_CXL;

    dc->desc = "CXL Type 2 Accelerator Device with Coherent Memory (GPU Passthrough)";
    device_class_set_legacy_reset(dc, cxl_type2_reset);
    device_class_set_props(dc, cxl_type2_props);
}

static const TypeInfo cxl_type2_info = {
    .name = TYPE_CXL_TYPE2,
    .parent = TYPE_PCI_DEVICE,
    .instance_size = sizeof(CXLType2State),
    .class_init = cxl_type2_class_init,
    .interfaces = (InterfaceInfo[]) {
        { INTERFACE_PCIE_DEVICE },
        { INTERFACE_CXL_DEVICE },
        { }
    },
};

static void cxl_type2_register_types(void)
{
    type_register_static(&cxl_type2_info);
}

type_init(cxl_type2_register_types)
