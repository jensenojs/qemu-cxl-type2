/*
 * CXL Type 2 Device (Accelerator with Coherent Memory) Header
 * Designed for GPU passthrough with CPU-GPU coherency
 *
 * Supports two GPU backends:
 * 1. VFIO passthrough - Direct hardware access
 * 2. hetGPU - Software CUDA translation layer for any GPU
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef CXL_TYPE2_H
#define CXL_TYPE2_H

#include "hw/pci/pci_device.h"
#include "hw/cxl/cxl_device.h"
#include "hw/cxl/cxl_component.h"
#include "hw/cxl/cxl_hetgpu.h"
#include "hw/cxl/cxl_type2_gpu_cmd.h"
#include "hw/cxl/cxl_type2_coherency.h"
#include "hw/cxl/cxl_p2p_dma.h"
#include "hw/virtio/virtio.h"
#include "qemu/thread.h"
#include "io/channel-socket.h"

#define TYPE_CXL_TYPE2 "cxl-type2"
#define CXL_TYPE2_VENDOR_ID 0x8086
#define CXL_TYPE2_DEVICE_ID 0x0d92

/* Type 2 combines Type 1 (accelerator/cache) + Type 3 (memory) */
#define CXL_TYPE2_DEFAULT_CACHE_SIZE (128 * MiB)
#define CXL_TYPE2_DEFAULT_MEM_SIZE (4 * GiB)
#define CXL_TYPE2_DCD_DEFAULT_GRANULARITY (1 * MiB)
#define CXL_TYPE2_DCD_INIT_AUTO UINT64_MAX
#define CXL_TYPE2_MAX_DCD_EXTENTS 64
#define CXL_TYPE2_MAX_GFAM_HOSTS 16
#define CXL_TYPE2_MAX_GFAM_MAPPINGS 128
#define CXL_TYPE2_MAX_MHSLD_LINES 4096

/* Coherency states for cache lines */
typedef enum {
    CXL_COHERENCY_INVALID = 0,
    CXL_COHERENCY_SHARED = 1,
    CXL_COHERENCY_EXCLUSIVE = 2,
    CXL_COHERENCY_MODIFIED = 3,
} CXLCoherencyState;

/* Cache line metadata for coherency tracking */
typedef struct CXLCacheLine {
    uint64_t tag;
    CXLCoherencyState state;
    bool dirty;
    uint8_t data[64];  /* Standard cache line size */
    uint64_t timestamp;
} CXLCacheLine;

/* GPU backend mode */
typedef enum {
    CXL_TYPE2_GPU_MODE_NONE = 0,    /* No GPU backend */
    CXL_TYPE2_GPU_MODE_VFIO = 1,    /* VFIO passthrough */
    CXL_TYPE2_GPU_MODE_HETGPU = 2,  /* hetGPU software translation */
    CXL_TYPE2_GPU_MODE_AUTO = 3,    /* Auto-detect best backend */
} CXLType2GPUMode;

/* GPU passthrough information */
typedef struct CXLType2GPUInfo {
    char *vfio_device;      /* VFIO device path (e.g., "0000:01:00.0") */
    bool passthrough_enabled;
    uint64_t gpu_mem_base;
    uint64_t gpu_mem_size;
    void *vfio_container;
    void *vfio_group;
    int vfio_device_fd;
    QemuThread irq_thread;  /* IRQ forwarding thread */
    bool irq_thread_running;

    /* hetGPU backend support */
    uint32_t mode;                  /* GPU backend mode (CXLType2GPUMode) */
    char *hetgpu_lib_path;          /* Path to hetGPU library */
    HetGPUState hetgpu_state;       /* hetGPU backend state */
    int32_t hetgpu_device_index;    /* hetGPU device index */
    uint32_t hetgpu_backend;        /* hetGPU backend type (HetGPUBackendType) */
} CXLType2GPUInfo;

/* Coherency protocol state */
typedef struct CXLType2CoherencyState {
    QemuMutex lock;
    GHashTable *cache_lines;  /* Maps address -> CXLCacheLine */
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t coherency_ops;
    uint64_t snoops;
    bool coherency_enabled;
} CXLType2CoherencyState;

/* CXLMemSim connection for coherency protocol */
typedef struct CXLType2MemSimConn {
    QIOChannelSocket *socket;
    char *server_addr;
    uint16_t server_port;
    QemuThread recv_thread;
    bool connected;
    QemuMutex lock;

    /* Shared memory mode support */
    bool use_shm;
    void *shm_base;
    size_t shm_size;
    char *shm_name;
} CXLType2MemSimConn;

/* Free block for coherent pool allocator */
typedef struct CXLCohFreeBlock {
    uint64_t offset;            /* Offset within BAR4 */
    uint64_t size;
    struct CXLCohFreeBlock *next;
} CXLCohFreeBlock;

/* Dynamic Capacity Device extent state for Type2 CXL.mem BAR4 */
typedef struct CXLType2DCDExtent {
    uint64_t base;
    uint64_t size;
    uint64_t tag;
    bool active;
} CXLType2DCDExtent;

typedef struct CXLType2DCDState {
    bool enabled;
    uint64_t granularity;
    uint64_t initial_size;
    uint64_t allocated;
    uint64_t next_tag;
    uint64_t add_requests;
    uint64_t release_requests;
    uint64_t failed_requests;
    CXLType2DCDExtent extents[CXL_TYPE2_MAX_DCD_EXTENTS];
    QemuMutex lock;
} CXLType2DCDState;

typedef struct CXLType2GFAMMapping {
    uint32_t host_id;
    uint32_t permissions;
    uint64_t base;
    uint64_t size;
    bool active;
} CXLType2GFAMMapping;

typedef struct CXLType2GFAMState {
    bool enabled;
    uint32_t local_host_id;
    uint32_t num_hosts;
    uint32_t default_permissions;
    uint32_t fabric_latency_ns;
    uint32_t bandwidth_mbps;
    uint64_t allowed_accesses;
    uint64_t denied_accesses;
    uint64_t total_latency_ns;
    CXLType2GFAMMapping mappings[CXL_TYPE2_MAX_GFAM_MAPPINGS];
    QemuMutex lock;
} CXLType2GFAMState;

typedef struct CXLType2MHSLDLine {
    uint64_t line_addr;
    uint32_t owner_head;
    uint32_t sharer_mask;
    bool valid;
    bool modified;
} CXLType2MHSLDLine;

typedef struct CXLType2MHSLDState {
    bool enabled;
    uint32_t num_heads;
    uint32_t local_head_id;
    uint32_t coherency_latency_ns;
    uint64_t reads;
    uint64_t writes;
    uint64_t atomics;
    uint64_t conflicts;
    uint64_t invalidations;
    CXLType2MHSLDLine lines[CXL_TYPE2_MAX_MHSLD_LINES];
    QemuMutex lock;
} CXLType2MHSLDState;

typedef struct CXLType2PendingHtoD {
    HetGPUStream stream;
    void *staging;
    uint64_t staging_id;
    size_t staging_capacity;
    uint64_t dev_ptr;
    uint64_t sequence;
    uint64_t call_id;
    int64_t enqueue_host_ns;
    size_t size;
    bool direct_source;
    const void *direct_host;
    struct CXLType2DirectSource *source;
    struct CXLType2PendingHtoD *next;
} CXLType2PendingHtoD;

typedef struct CXLType2DirectRegistration {
    void *host_address;
    uint64_t length;
    uint64_t references;
    uint64_t view_count;
    uint32_t member_count;
    bool cuda_registered;
    bool revoke_pending;
    struct CXLType2DirectPhysical *members;
    struct CXLType2DirectRegistration *next;
} CXLType2DirectRegistration;

typedef struct CXLType2DirectPhysical {
    VirtioSharedMemoryMapping *mapping;
    uint64_t generation;
    hwaddr mapping_offset;
    uint64_t length;
    void *host_address;
    uint64_t references;
    CXLType2DirectRegistration *registration;
    struct CXLType2DirectPhysical *group_next;
    struct CXLType2DirectPhysical *next;
} CXLType2DirectPhysical;

typedef struct CXLType2DirectRun {
    CXLType2DirectPhysical *physical;
    uint64_t physical_offset;
    uint64_t length;
} CXLType2DirectRun;

typedef struct CXLType2DirectSource {
    uint64_t source_id;
    uint64_t case_epoch;
    uint64_t lease_handle;
    uint64_t logical_bytes;
    uint64_t unique_dmap_bytes;
    CXLGPUSourceRangeV1 *ranges;
    CXLType2DirectRun *runs;
    uint32_t range_count;
    uint32_t run_count;
    uint64_t pending_refcount;
    struct CXLType2DirectSource *next;
} CXLType2DirectSource;

typedef struct CXLType2HtoDStagingBuffer {
    void *data;
    uint64_t id;
    size_t capacity;
    struct CXLType2HtoDStagingBuffer *next;
} CXLType2HtoDStagingBuffer;

typedef struct CXLType2EventHtoDMark {
    void *event;
    HetGPUStream stream;
    uint64_t sequence;
    struct CXLType2EventHtoDMark *next;
} CXLType2EventHtoDMark;

typedef struct CXLType2IntervalIdentity {
    bool valid;
    bool operation_code_valid;
    uint32_t operation_code;
    uint64_t sequence;
    const char *owner;
    const char *category;
    const char *operation;
} CXLType2IntervalIdentity;

typedef struct CXLType2IntervalLedger {
    int64_t span_begin_ns;
    int64_t span_end_ns;
    int64_t last_begin_ns;
    int64_t union_end_ns;
    uint64_t interval_count;
    uint64_t total_duration_ns;
    uint64_t union_duration_ns;
    uint64_t largest_gap_duration_ns;
    int64_t largest_gap_begin_ns;
    int64_t largest_gap_end_ns;
    CXLType2IntervalIdentity span_begin_identity;
    CXLType2IntervalIdentity union_end_identity;
    CXLType2IntervalIdentity largest_gap_previous;
    CXLType2IntervalIdentity largest_gap_next;
    const char *first_error;
} CXLType2IntervalLedger;

/* Main Type 2 device state */
typedef struct CXLType2State {
    PCIDevice parent_obj;

    /* CXL component and device states */
    CXLComponentState cxl_cstate;
    CXLDeviceState cxl_dstate;

    /* Memory regions */
    MemoryRegion bar0;                 /* Component registers */
    MemoryRegion cache_mem;            /* Type 1: Cache for coherent access */
    MemoryRegion cache_io;             /* Cache access interceptor */
    MemoryRegion gpu_descriptor_mem;   /* RAM-backed BAR2 command descriptor */
    MemoryRegion gpu_data_mem;         /* RAM-backed BAR2 command data window */
    MemoryRegion gpu_batch_data_mem;   /* RAM-backed BAR2 batch payload */
    MemoryRegion device_mem;           /* Type 3: Device-attached memory */
    MemoryRegion device_mem_io;        /* Device memory interceptor */
    MemoryRegion coherent_pool_mem;    /* Direct RAM staging inside BAR4 */

    /* GPU command state */
    struct {
        uint32_t status;
        uint32_t cmd_status;
        uint32_t cmd_result;
        uint64_t trace_sequence;
        uint64_t call_id;
        uint64_t params[8];
        uint64_t results[4];
        uint8_t  *data;                /* Host pointer into gpu_data_mem */
        size_t   data_size;            /* Size of data buffer */
        uint8_t  *batch_data;          /* Host pointer into gpu_batch_data_mem */
        CXLGPURAMCommandDescriptor *descriptor;
        uint64_t device_generation;
        uint64_t last_accepted_submission;
        uint64_t last_completed_submission;
        /* Guest-visible IDs are indexes into tables that grow with the active
         * CUDA workload. A case reset releases both tables and invalidates all
         * IDs from that case. */
        void    **modules;             /* Loaded PTX/CUBIN modules */
        void    **functions;           /* Kernel function handles */
        void    **graphs;              /* CUDA graph handles */
        void    **graph_execs;         /* CUDA executable graph handles */
        void    **graph_nodes;         /* CUDA graph node handles */
        void    **link_states;         /* CUDA JIT link state handles */
        void    **streams;             /* CUDA stream handles */
        void    **events;              /* CUDA event handles */
        size_t   modules_capacity;
        size_t   functions_capacity;
        size_t   graphs_capacity;
        size_t   graph_execs_capacity;
        size_t   graph_nodes_capacity;
        size_t   link_states_capacity;
        size_t   streams_capacity;
        size_t   events_capacity;
        uint32_t num_modules;
        uint32_t num_functions;
        uint32_t num_graphs;
        uint32_t num_graph_execs;
        uint32_t num_graph_nodes;
        uint32_t num_link_states;
        uint32_t num_streams;
        uint32_t num_events;
        uint32_t modules_high_water;
        uint32_t functions_high_water;
        uint32_t capabilities;         /* Device capabilities (bulk transfer, etc.) */
    } gpu_cmd;

    CXLType2PendingHtoD *pending_htod;
    CXLType2HtoDStagingBuffer *htod_staging_pool;
    CXLType2EventHtoDMark *event_htod_marks;
    uint64_t next_htod_sequence;
    uint64_t next_htod_staging_id;
    uint64_t htod_staging_pool_size;
    uint64_t htod_pending_copies;
    uint64_t htod_pending_bytes;
    uint64_t htod_peak_pending_copies;
    uint64_t htod_peak_pending_bytes;
    uint64_t htod_pooled_buffers;
    uint64_t htod_pooled_bytes;
    uint64_t htod_peak_pooled_bytes;
    uint64_t htod_pool_hits;
    uint64_t htod_pool_misses;
    uint64_t htod_driver_allocations;
    uint64_t htod_driver_frees;
    uint64_t htod_pool_evictions;
    bool cuda_direct_source;
    Object *direct_source_fs;
    CXLType2DirectPhysical *direct_physicals;
    CXLType2DirectRegistration *direct_registrations;
    GTree *direct_physical_ranges;
    CXLType2DirectSource *direct_sources;
    uint64_t next_direct_source_id;
    bool direct_source_poisoned;

    struct {
        bool required;
        char *run_root;
        uint64_t run_binding;
        uint64_t min_allocation_bytes;
        uint64_t max_regions;
        uint64_t checkpoint_every_launches;
        bool checkpoint_enabled;
        bool qemu_cuda_calls_enabled;
        bool concordia_runtime_details_enabled;
        uint64_t next_epoch;
        bool failed;
        uint32_t failure_code;
        uint32_t active_case;
        uint64_t active_epoch;
        uint64_t active_first_sequence;
        uint64_t active_config_binding;
        uint64_t active_command_count;
        uint64_t active_command_failures;
        uint64_t active_command_busy_ns;
        int64_t active_first_command_host_ns;
        int64_t active_last_command_host_ns;
        uint64_t active_command_calls[256];
        uint64_t active_command_busy_ns_by_command[256];
        uint64_t active_driver_calls_by_command[256];
        uint64_t active_driver_busy_ns_by_command[256];
        uint64_t active_command_sequence;
        uint32_t active_command_code;
        CXLType2IntervalLedger command_intervals;
        CXLType2IntervalLedger driver_intervals;
        uint64_t active_cxl_request_count;
        uint64_t active_cxl_read_count;
        uint64_t active_cxl_write_count;
        uint64_t active_cxl_logical_bytes;
        uint64_t active_cxl_qemu_wall_ns;
        uint64_t active_cxl_response_count;
        uint64_t active_cxl_request_failures;
        uint64_t active_cxl_server_reported_latency_ns;
        uint64_t active_cxl_range_requests;
        uint64_t active_cxl_range_bytes;
        uint64_t active_cxl_wire_bytes;
        uint64_t active_direct_register_calls;
        uint64_t active_direct_unregister_calls;
        uint64_t active_direct_register_validate_ns;
        uint64_t active_direct_register_resolve_ns;
        uint64_t active_direct_register_acquire_ns;
        uint64_t active_direct_register_commit_ns;
        uint64_t active_direct_unregister_release_ns;
        uint64_t active_direct_physical_register_calls;
        uint64_t active_direct_physical_register_ns;
        uint64_t active_direct_registration_views;
        uint64_t active_direct_registration_bytes;
        uint64_t active_direct_registration_groups;
        uint64_t active_direct_group_members;
        uint64_t active_direct_max_group_members;
        uint64_t active_direct_retained_groups;
        uint64_t active_direct_peak_retained_groups;
        uint64_t active_direct_coalesced_views;
        uint64_t active_direct_max_registration_views;
        uint64_t active_direct_physical_boundaries;
        uint64_t active_direct_host_contiguous_boundaries;
        uint64_t active_direct_host_contiguous_following_bytes;
        uint64_t active_direct_physical_unregister_calls;
        uint64_t active_direct_physical_unregister_ns;
        uint64_t active_direct_cache_hits;
        uint64_t active_direct_active_hits;
        uint64_t active_direct_cache_misses;
        uint64_t active_direct_revoke_releases;
        uint64_t active_direct_retained_physicals;
        uint64_t active_direct_peak_retained_physicals;
        uint64_t active_direct_logical_ranges;
        uint64_t active_direct_fragments;
        uint64_t active_direct_bytes;
        uint64_t active_payload_batches;
        uint64_t active_payload_source_bytes;
        uint64_t active_htod_pool_hits;
        uint64_t active_htod_pool_misses;
        uint64_t active_htod_driver_allocations;
        uint64_t active_htod_driver_frees;
        uint64_t active_htod_pool_evictions;
        uint64_t active_htod_staging_pending_bytes;
        uint64_t active_htod_peak_staging_pending_bytes;
        uint64_t active_htod_peak_pooled_bytes;
        uint64_t active_elided_stream_syncs;
        uint64_t last_successful_stream_sync_wire;
        bool last_command_was_successful_stream_sync;
    } paired_case;

    /* Bulk transfer region for large memory operations */
    MemoryRegion bulk_transfer_region;
    void *bulk_transfer_ptr;           /* Mapped pointer for bulk transfers */
    size_t bulk_transfer_size;

    /* Device configuration */
    uint64_t cache_size;
    uint64_t device_mem_size;
    uint64_t sn;                       /* Serial number */

    PCIExpLinkSpeed speed;
    PCIExpLinkWidth width;
    bool flitmode;
    bool hdmdb;
    bool bi_enabled;

    /* Fabric memory feature models */
    CXLType2DCDState dcd;
    CXLType2GFAMState gfam;
    CXLType2MHSLDState mhsld;

    /* GPU passthrough */
    CXLType2GPUInfo gpu_info;

    /* Coherency protocol */
    CXLType2CoherencyState coherency;

    /* Enhanced BAR coherency tracking */
    CXLBARCoherencyState bar_coherency;

    /* P2P DMA engine for Type2 <-> Type3 transfers */
    CXLP2PDMAEngine p2p_engine;

    /* CXLMemSim connection */
    CXLType2MemSimConn memsim;

    /* Memory backend for device memory */
    HostMemoryBackend *hostmem;

    /* Coherent shared memory pool (carved from top of BAR4) */
    struct {
        uint64_t base_offset;       /* device_mem_size - pool_size */
        uint64_t size;              /* Configurable, default 256MB */
        uint64_t used;
        GHashTable *allocations;    /* bar4_offset -> alloc_size */
        struct CXLCohFreeBlock *free_list;
        QemuMutex lock;
    } coherent_pool;

    /* Statistics and monitoring */
    struct {
        uint64_t read_ops;
        uint64_t write_ops;
        uint64_t gpu_accesses;
        uint64_t cpu_accesses;
        uint64_t coherency_violations;
    } stats;

    /* Latency simulation */
    bool latency_enabled;
    uint32_t read_latency_ns;
    uint32_t write_latency_ns;
    uint32_t coherency_latency_ns;

} CXLType2State;

#define CXL_TYPE2(obj) OBJECT_CHECK(CXLType2State, (obj), TYPE_CXL_TYPE2)

/* Message types for CXLMemSim communication */
enum CXLType2MsgType {
    CXL_T2_MSG_READ = 1,
    CXL_T2_MSG_WRITE = 2,
    CXL_T2_MSG_CACHE_FLUSH = 3,
    CXL_T2_MSG_COHERENCY_REQ = 4,
    CXL_T2_MSG_SNOOP_REQ = 5,
    CXL_T2_MSG_SNOOP_RESP = 6,
    CXL_T2_MSG_INVALIDATE = 7,
    CXL_T2_MSG_WRITEBACK = 8,
    CXL_T2_MSG_GPU_ACCESS = 9,
    CXL_T2_MSG_RESPONSE = 10,
};

/* Message structure for CXLMemSim protocol */
typedef struct CXLType2Message {
    uint32_t type;
    uint32_t size;
    uint64_t addr;
    uint64_t timestamp;
    uint8_t coherency_state;
    uint8_t source_id;
    uint8_t data[64];
} CXLType2Message;

/* Coherency protocol functions */
void cxl_type2_coherency_init(CXLType2State *ct2d);
void cxl_type2_coherency_cleanup(CXLType2State *ct2d);
CXLCacheLine *cxl_type2_cache_lookup(CXLType2State *ct2d, uint64_t addr);
void cxl_type2_cache_insert(CXLType2State *ct2d, uint64_t addr,
                            const uint8_t *data, CXLCoherencyState state);
void cxl_type2_cache_invalidate(CXLType2State *ct2d, uint64_t addr);
void cxl_type2_cache_writeback(CXLType2State *ct2d, uint64_t addr);
bool cxl_type2_snoop_request(CXLType2State *ct2d, uint64_t addr, bool invalidate);

/* GPU passthrough functions */
int cxl_type2_gpu_init(CXLType2State *ct2d, Error **errp);
void cxl_type2_gpu_cleanup(CXLType2State *ct2d);
int cxl_type2_gpu_read(CXLType2State *ct2d, uint64_t offset, void *buf, size_t size);
int cxl_type2_gpu_write(CXLType2State *ct2d, uint64_t offset, const void *buf, size_t size);

/* hetGPU backend functions */
int cxl_type2_hetgpu_init(CXLType2State *ct2d, Error **errp);
void cxl_type2_hetgpu_cleanup(CXLType2State *ct2d);
int cxl_type2_hetgpu_load_ptx(CXLType2State *ct2d, const char *ptx_source,
                               void **module);
int cxl_type2_hetgpu_launch_kernel(CXLType2State *ct2d, void *function,
                                    uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                                    uint32_t block_x, uint32_t block_y, uint32_t block_z,
                                    uint32_t shared_mem, void **args, size_t num_args);
int cxl_type2_hetgpu_malloc(CXLType2State *ct2d, size_t size, uint64_t *dev_ptr);
int cxl_type2_hetgpu_free(CXLType2State *ct2d, uint64_t dev_ptr);
int cxl_type2_hetgpu_memcpy_htod(CXLType2State *ct2d, uint64_t dst, const void *src, size_t size);
int cxl_type2_hetgpu_memcpy_dtoh(CXLType2State *ct2d, void *dst, uint64_t src, size_t size);
int cxl_type2_hetgpu_sync(CXLType2State *ct2d);

#endif /* CXL_TYPE2_H */
