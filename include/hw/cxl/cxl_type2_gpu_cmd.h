/*
 * CXL Type 2 GPU Command Interface
 * Defines the command protocol between guest libcuda and host hetGPU
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef CXL_TYPE2_GPU_CMD_H
#define CXL_TYPE2_GPU_CMD_H

#include <stdint.h>
#include <stddef.h>

/* GPU Command Register Offsets (from BAR2 base) */
#define CXL_GPU_REG_MAGIC           0x0000  /* Magic number: 0x43584C32 "CXL2" */
#define CXL_GPU_REG_VERSION         0x0004  /* Interface version */
#define CXL_GPU_REG_STATUS          0x0008  /* Device status */
#define CXL_GPU_REG_CAPS            0x000C  /* Device capabilities */

#define CXL_GPU_REG_CMD             0x0010  /* Command register */
#define CXL_GPU_REG_CMD_STATUS      0x0014  /* Command status */
#define CXL_GPU_REG_CMD_RESULT      0x0018  /* Command result/error code */
#define CXL_GPU_REG_CMD_DATA_LO     0x001C  /* Command data low 32 bits */
#define CXL_GPU_REG_CMD_DATA_HI     0x0020  /* Command data high 32 bits */
#define CXL_GPU_REG_CALL_ID         0x0028  /* Guest process/call sequence identity */

#define CXL_GPU_REG_PARAM0          0x0040  /* Parameter 0 */
#define CXL_GPU_REG_PARAM1          0x0048  /* Parameter 1 */
#define CXL_GPU_REG_PARAM2          0x0050  /* Parameter 2 */
#define CXL_GPU_REG_PARAM3          0x0058  /* Parameter 3 */
#define CXL_GPU_REG_PARAM4          0x0060  /* Parameter 4 */
#define CXL_GPU_REG_PARAM5          0x0068  /* Parameter 5 */
#define CXL_GPU_REG_PARAM6          0x0070  /* Parameter 6 */
#define CXL_GPU_REG_PARAM7          0x0078  /* Parameter 7 */

#define CXL_GPU_REG_RESULT0         0x0080  /* Result 0 */
#define CXL_GPU_REG_RESULT1         0x0088  /* Result 1 */
#define CXL_GPU_REG_RESULT2         0x0090  /* Result 2 */
#define CXL_GPU_REG_RESULT3         0x0098  /* Result 3 */

#define CXL_GPU_DESCRIPTOR_OFFSET 0x801000
#define CXL_GPU_DESCRIPTOR_REGION_SIZE 0x1000
#define CXL_GPU_DESCRIPTOR_WIRE_SIZE 0x0100
#define CXL_GPU_DESCRIPTOR_PROTOCOL_VERSION 3U
#define CXL_GPU_DESCRIPTOR_DOORBELL_OFFSET CXL_GPU_REG_CMD
#define CXL_GPU_DESCRIPTOR_DOORBELL_VALUE 1U

#define CXL_GPU_DESCRIPTOR_COMPLETION_EMPTY 0U
#define CXL_GPU_DESCRIPTOR_COMPLETION_COMPLETE 1U
#define CXL_GPU_DESCRIPTOR_COMPLETION_ERROR 2U

typedef struct CXLGPURAMCommandDescriptor {
    uint32_t protocol_version;
    uint32_t descriptor_size;
    uint64_t request_submission;
    uint64_t request_device_generation;
    uint64_t request_command;
    uint64_t request_call_id;
    uint64_t request_case_epoch;
    uint64_t params[8];
    uint64_t completion_submission;
    uint64_t completion_device_generation;
    uint64_t completion_status;
    int64_t result;
    uint64_t results[4];
    uint64_t active_case_epoch;
    uint64_t device_generation;
    uint8_t reserved[0x40];
} CXLGPURAMCommandDescriptor;

_Static_assert(sizeof(CXLGPURAMCommandDescriptor) == CXL_GPU_DESCRIPTOR_WIRE_SIZE,
               "CXL GPU descriptor size mismatch");
_Static_assert(offsetof(CXLGPURAMCommandDescriptor, request_submission) == 0x08,
               "CXL GPU request submission offset mismatch");
_Static_assert(offsetof(CXLGPURAMCommandDescriptor, params) == 0x30,
               "CXL GPU params offset mismatch");
_Static_assert(offsetof(CXLGPURAMCommandDescriptor, completion_submission) == 0x70,
               "CXL GPU completion submission offset mismatch");
_Static_assert(offsetof(CXLGPURAMCommandDescriptor, completion_status) == 0x80,
               "CXL GPU completion status offset mismatch");
_Static_assert(offsetof(CXLGPURAMCommandDescriptor, results) == 0x90,
               "CXL GPU results offset mismatch");
_Static_assert(offsetof(CXLGPURAMCommandDescriptor, active_case_epoch) == 0xb0,
               "CXL GPU case epoch offset mismatch");
_Static_assert(offsetof(CXLGPURAMCommandDescriptor, device_generation) == 0xb8,
               "CXL GPU generation offset mismatch");

/* Device info registers */
#define CXL_GPU_REG_DEV_NAME        0x0100  /* Device name (64 bytes) */
#define CXL_GPU_REG_TOTAL_MEM       0x0140  /* Total memory */
#define CXL_GPU_REG_FREE_MEM        0x0148  /* Free memory */
#define CXL_GPU_REG_CC_MAJOR        0x0150  /* Compute capability major */
#define CXL_GPU_REG_CC_MINOR        0x0154  /* Compute capability minor */
#define CXL_GPU_REG_MP_COUNT        0x0158  /* Multiprocessor count */
#define CXL_GPU_REG_MAX_THREADS     0x015C  /* Max threads per block */
#define CXL_GPU_REG_WARP_SIZE       0x0160  /* Warp size */
#define CXL_GPU_REG_BACKEND         0x0164  /* Backend type */
#define CXL_GPU_REG_DRIVER_VERSION  0x0168  /* Real host CUDA Driver API version */

/* BAR2 command payload.  Compressed module encodings keep the bytes crossing
 * this window independent from the decoded CUBIN size. */
#define CXL_GPU_DATA_OFFSET         0x1000    /* Data buffer offset */
#define CXL_GPU_DATA_SIZE           0x800000  /* Data buffer size (8 MiB) */
#define CXL_GPU_BATCH_DATA_OFFSET   0x802000
#define CXL_GPU_BATCH_DATA_SIZE     0x2000000 /* 32 MiB batch payload */
#define CXL_GPU_CMD_REG_SIZE        (CXL_GPU_BATCH_DATA_OFFSET + CXL_GPU_BATCH_DATA_SIZE)

typedef struct CXLGPUBatchHtoDHeader {
    uint32_t header_size;
    uint32_t range_count;
    uint32_t range_size;
    uint32_t reserved0;
    uint64_t payload_bytes;
    uint64_t reserved1;
} CXLGPUBatchHtoDHeader;

typedef struct CXLGPUBatchHtoDRange {
    uint64_t source_offset;
    uint64_t destination;
    uint64_t size;
} CXLGPUBatchHtoDRange;

typedef struct CXLGPUSourceRegisterV1 {
    uint32_t flags;
    uint32_t range_count;
    uint32_t run_count;
    uint32_t reserved0;
    uint64_t lease_handle;
    uint64_t logical_bytes;
    uint64_t unique_dmap_bytes;
    uint64_t reserved1;
} CXLGPUSourceRegisterV1;

typedef struct CXLGPUSourceRangeV1 {
    uint32_t first_run;
    uint32_t run_count;
    uint64_t first_run_byte_offset;
    uint64_t length;
} CXLGPUSourceRangeV1;

typedef struct CXLGPUSourceRunV1 {
    uint64_t guest_phys_addr;
    uint64_t length;
} CXLGPUSourceRunV1;

typedef struct CXLGPUDirectRangeV1 {
    uint64_t destination;
    uint64_t size;
    uint64_t source_id;
    uint32_t source_range;
    uint32_t reserved0;
    uint64_t source_offset;
} CXLGPUDirectRangeV1;

_Static_assert(sizeof(CXLGPUBatchHtoDHeader) == 32,
               "CXL GPU batch header size mismatch");
_Static_assert(sizeof(CXLGPUBatchHtoDRange) == 24,
               "CXL GPU batch range size mismatch");
_Static_assert(sizeof(CXLGPUSourceRegisterV1) == 48,
               "CXL GPU source register size mismatch");
_Static_assert(sizeof(CXLGPUSourceRangeV1) == 24,
               "CXL GPU source range size mismatch");
_Static_assert(sizeof(CXLGPUSourceRunV1) == 16,
               "CXL GPU source run size mismatch");
_Static_assert(sizeof(CXLGPUDirectRangeV1) == 40,
               "CXL GPU direct range size mismatch");
_Static_assert(offsetof(CXLGPUSourceRegisterV1, lease_handle) == 16,
               "CXL GPU source lease offset mismatch");
_Static_assert(offsetof(CXLGPUSourceRangeV1, first_run_byte_offset) == 8,
               "CXL GPU source range offset mismatch");
_Static_assert(offsetof(CXLGPUDirectRangeV1, source_offset) == 32,
               "CXL GPU direct source offset mismatch");
_Static_assert(offsetof(CXLGPUBatchHtoDHeader, payload_bytes) == 16,
               "CXL GPU batch payload size offset mismatch");

_Static_assert(CXL_GPU_DESCRIPTOR_OFFSET % CXL_GPU_DESCRIPTOR_REGION_SIZE == 0,
               "CXL GPU descriptor region must be page aligned");
_Static_assert(CXL_GPU_DESCRIPTOR_WIRE_SIZE <= CXL_GPU_DESCRIPTOR_REGION_SIZE,
               "CXL GPU descriptor wire does not fit in its RAM region");
_Static_assert(CXL_GPU_DESCRIPTOR_OFFSET >= CXL_GPU_DATA_OFFSET + CXL_GPU_DATA_SIZE,
               "CXL GPU descriptor region overlaps the payload window");
_Static_assert(CXL_GPU_BATCH_DATA_OFFSET >=
                   CXL_GPU_DESCRIPTOR_OFFSET + CXL_GPU_DESCRIPTOR_REGION_SIZE,
               "CXL GPU batch region overlaps the descriptor");

/* Module payload encoding in CXL_GPU_REG_PARAM1. */
#define CXL_GPU_MODULE_DATA_ZSTD    (1U << 0)
#define CXL_GPU_MODULE_DATA_LZ4     (1U << 1)

/* Bulk transfer region (BAR4 direct access) */
#define CXL_GPU_BULK_TRANSFER_SIZE  0x4000000 /* 64MB bulk transfer region */

/* Capability bits */
#define CXL_GPU_CAP_BULK_TRANSFER   (1 << 0)  /* Supports bulk transfer mode */
#define CXL_GPU_CAP_CACHE_COHERENT  (1 << 1)  /* CXL.cache coherent memory */
#define CXL_GPU_CAP_DMA_ENGINE      (1 << 2)  /* Hardware DMA engine available */
#define CXL_GPU_CAP_COHERENT_POOL   (1 << 3)  /* Coherent shared memory pool */
#define CXL_GPU_CAP_DEVICE_BIAS     (1 << 4)  /* Device-biased directory mode */
#define CXL_GPU_CAP_DCD             (1 << 5)  /* Dynamic Capacity Device model */
#define CXL_GPU_CAP_GFAM            (1 << 6)  /* Global Fabric Attached Memory */
#define CXL_GPU_CAP_MHSLD           (1 << 7)  /* Multi-headed SLD coherency */

/* Magic number */
#define CXL_GPU_MAGIC               0x43584C32  /* "CXL2" */
#define CXL_GPU_VERSION             0x00011300  /* v1.19.0: fused direct-source batch */

#define CXL_GPU_STREAM_WIRE_NULL       0xffffffffffffffffULL
#define CXL_GPU_STREAM_WIRE_LEGACY     0xfffffffffffffffeULL
#define CXL_GPU_STREAM_WIRE_PER_THREAD 0xfffffffffffffffdULL

typedef enum {
    CXL_GPU_STREAM_SYNC_PUBLIC_API = 0,
    CXL_GPU_STREAM_SYNC_DTOH_ASYNC_DRAIN = 1,
    CXL_GPU_STREAM_SYNC_MEMSET_D8_ASYNC_DRAIN = 2,
    CXL_GPU_STREAM_SYNC_REASON_COUNT = 3,
} CXLGPUStreamSyncReason;

#define CXL_GPU_CASE_PROTOCOL_VERSION 1U
#define CXL_GPU_OBSERVATION_ANCHOR_VERSION 1U

typedef enum {
    CXL_GPU_OBSERVATION_ANCHOR_DECODE_BEGIN = 1,
    CXL_GPU_OBSERVATION_ANCHOR_DECODE_END = 2,
} CXLGPUObservationAnchorPhase;

typedef enum {
    CXL_GPU_CASE_NONE = 0,
    CXL_GPU_CASE_BASELINE = 1,
    CXL_GPU_CASE_CONCORDIA = 2,
} CXLGPUPairedCase;

/* Device status bits */
#define CXL_GPU_STATUS_READY        (1 << 0)
#define CXL_GPU_STATUS_BUSY         (1 << 1)
#define CXL_GPU_STATUS_ERROR        (1 << 2)
#define CXL_GPU_STATUS_CTX_ACTIVE   (1 << 3)

/* Command status */
#define CXL_GPU_CMD_STATUS_IDLE     0
#define CXL_GPU_CMD_STATUS_PENDING  1
#define CXL_GPU_CMD_STATUS_RUNNING  2
#define CXL_GPU_CMD_STATUS_COMPLETE 3
#define CXL_GPU_CMD_STATUS_ERROR    4

/* GPU Commands */
typedef enum {
    CXL_GPU_CMD_NOP             = 0x00,
    CXL_GPU_CMD_INIT            = 0x01,
    CXL_GPU_CMD_GET_DEVICE_COUNT= 0x02,
    CXL_GPU_CMD_GET_DEVICE      = 0x03,
    CXL_GPU_CMD_GET_DEVICE_NAME = 0x04,
    CXL_GPU_CMD_GET_DEVICE_PROPS= 0x05,
    CXL_GPU_CMD_GET_TOTAL_MEM   = 0x06,
    CXL_GPU_CMD_GET_DEVICE_ATTRIBUTE = 0x07,
    CXL_GPU_CMD_CASE_BEGIN      = 0x08,
    CXL_GPU_CMD_CASE_END        = 0x09,
    CXL_GPU_CMD_GET_ERROR_NAME  = 0x0A,
    CXL_GPU_CMD_OBSERVATION_ANCHOR = 0x0B,

    CXL_GPU_CMD_CTX_CREATE      = 0x10,
    CXL_GPU_CMD_CTX_DESTROY     = 0x11,
    CXL_GPU_CMD_CTX_SYNC        = 0x12,
    CXL_GPU_CMD_CTX_GET_LIMIT   = 0x13,
    CXL_GPU_CMD_CTX_ENABLE_PEER = 0x14,
    CXL_GPU_CMD_CTX_DISABLE_PEER = 0x15,
    CXL_GPU_CMD_DEVICE_CAN_ACCESS_PEER = 0x16,

    CXL_GPU_CMD_MEM_ALLOC       = 0x20,
    CXL_GPU_CMD_MEM_FREE        = 0x21,
    CXL_GPU_CMD_MEM_COPY_HTOD   = 0x22,
    CXL_GPU_CMD_MEM_COPY_DTOH   = 0x23,
    CXL_GPU_CMD_MEM_COPY_DTOD   = 0x24,
    CXL_GPU_CMD_MEM_SET         = 0x25,
    CXL_GPU_CMD_MEM_GET_INFO    = 0x26,
    CXL_GPU_CMD_MEM_GET_POINTER_MEMORY_TYPE = 0x27,
    CXL_GPU_CMD_MEM_PREFETCH_ASYNC = 0x28,
    CXL_GPU_CMD_MEM_COPY_HTOD_ASYNC = 0x29,
    CXL_GPU_CMD_MEM_COPY_2D_DTOD = 0x2A,
    CXL_GPU_CMD_BATCH_HTOD_ASYNC = 0x2B,
    CXL_GPU_CMD_SOURCE_REGISTER = 0x2C,
    CXL_GPU_CMD_SOURCE_UNREGISTER = 0x2D,
    CXL_GPU_CMD_BATCH_HTOD_DIRECT_ASYNC = 0x2E,
    CXL_GPU_CMD_MEM_COPY_DTOD_ASYNC = 0x2F,

    CXL_GPU_CMD_MODULE_LOAD_PTX = 0x30,
    CXL_GPU_CMD_MODULE_UNLOAD   = 0x31,
    CXL_GPU_CMD_FUNC_GET        = 0x32,
    CXL_GPU_CMD_MODULE_LOAD_CUBIN = 0x33,
    CXL_GPU_CMD_MODULE_GET_GLOBAL = 0x34,
    CXL_GPU_CMD_SOURCE_REGISTER_BATCH_HTOD_DIRECT_ASYNC = 0x35,
    CXL_GPU_CMD_FUNC_SET_ATTRIBUTE = 0x36,
    CXL_GPU_CMD_FUNC_GET_OCCUPANCY = 0x37,
    CXL_GPU_CMD_MODULE_GET_LOADING_MODE = 0x38,
    CXL_GPU_CMD_GRAPH_EXEC_KERNEL_NODE_SET_PARAMS = 0x39,
    CXL_GPU_CMD_GRAPH_KERNEL_NODE_GET_PARAMS = 0x3A,
    CXL_GPU_CMD_GRAPH_EXEC_DESTROY = 0x3B,
    CXL_GPU_CMD_GRAPH_LAUNCH = 0x3C,
    CXL_GPU_CMD_GRAPH_DESTROY = 0x3D,
    CXL_GPU_CMD_GRAPH_INSTANTIATE = 0x3E,
    CXL_GPU_CMD_GRAPH_GET_NODES = 0x3F,

    CXL_GPU_CMD_LAUNCH_KERNEL   = 0x40,
    CXL_GPU_CMD_GRAPH_NODE_GET_TYPE = 0x41,
    CXL_GPU_CMD_LINK_CREATE = 0x42,
    CXL_GPU_CMD_LINK_ADD_DATA = 0x43,
    CXL_GPU_CMD_LINK_COMPLETE = 0x44,
    CXL_GPU_CMD_LINK_DESTROY = 0x45,
    CXL_GPU_CMD_FUNC_GET_ATTRIBUTE = 0x46,
    CXL_GPU_CMD_FUNC_GET_PARAM_LAYOUT = 0x47,
    CXL_GPU_CMD_GRAPH_EXEC_UPDATE = 0x48,

    CXL_GPU_CMD_STREAM_CREATE   = 0x50,
    CXL_GPU_CMD_STREAM_DESTROY  = 0x51,
    CXL_GPU_CMD_STREAM_SYNC     = 0x52,
    CXL_GPU_CMD_STREAM_WAIT_EVENT = 0x53,
    CXL_GPU_CMD_STREAM_WAIT_VALUE32 = 0x54,
    CXL_GPU_CMD_STREAM_BATCH_MEM_OP = 0x55,
    CXL_GPU_CMD_STREAM_GET_CAPTURE_INFO = 0x56,
    CXL_GPU_CMD_STREAM_GET_CTX = 0x57,
    CXL_GPU_CMD_STREAM_BEGIN_CAPTURE = 0x58,
    CXL_GPU_CMD_STREAM_END_CAPTURE = 0x59,
    CXL_GPU_CMD_STREAM_IS_CAPTURING = 0x5A,

    CXL_GPU_CMD_EVENT_CREATE    = 0x60,
    CXL_GPU_CMD_EVENT_DESTROY   = 0x61,
    CXL_GPU_CMD_EVENT_RECORD    = 0x62,
    CXL_GPU_CMD_EVENT_SYNC      = 0x63,
    CXL_GPU_CMD_EVENT_QUERY = 0x64,
    CXL_GPU_CMD_EVENT_ELAPSED_TIME = 0x65,
    CXL_GPU_CMD_GET_ERROR_STRING = 0x66,

    /* Bulk transfer commands (optimized for large transfers) */
    CXL_GPU_CMD_BULK_HTOD       = 0x70,  /* Bulk host-to-device via BAR4 */
    CXL_GPU_CMD_BULK_DTOH       = 0x71,  /* Bulk device-to-host via BAR4 */
    CXL_GPU_CMD_BULK_DTOD       = 0x72,  /* Bulk device-to-device */
    CXL_GPU_CMD_BULK_HTOD_ASYNC = 0x73,  /* BAR4 source, stream-aware HtoD */

    /* CXL.cache coherency commands */
    CXL_GPU_CMD_CACHE_FLUSH     = 0x80,  /* Flush cache lines to device */
    CXL_GPU_CMD_CACHE_INVALIDATE= 0x81,  /* Invalidate cache lines */
    CXL_GPU_CMD_CACHE_WRITEBACK = 0x82,  /* Writeback dirty cache lines */
    CXL_GPU_CMD_CACHE_PREFETCH  = 0x83,  /* Prefetch cache lines into Type2 cache */

    /* P2P DMA commands: defined as macros in cxl_p2p_dma.h (0x90-0x96) */

    /* Coherent shared memory pool commands */
    CXL_GPU_CMD_COHERENT_ALLOC      = 0xA0,  /* Allocate from coherent pool */
    CXL_GPU_CMD_COHERENT_FREE       = 0xA1,  /* Free coherent pool allocation */
    CXL_GPU_CMD_COHERENT_GET_INFO   = 0xA2,  /* Get coherent pool info */
    CXL_GPU_CMD_COHERENT_FENCE      = 0xA3,  /* Coherent memory fence */

    /* Device-biased directory commands */
    CXL_GPU_CMD_SET_BIAS            = 0xA4,  /* Set bias mode for region */
    CXL_GPU_CMD_GET_BIAS            = 0xA5,  /* Get bias mode for address */
    CXL_GPU_CMD_BIAS_FLIP           = 0xA6,  /* Flip bias with cache flush */

    /* Coherency statistics commands */
    CXL_GPU_CMD_COH_GET_STATS       = 0xB0,  /* Get coherency statistics */
    CXL_GPU_CMD_COH_RESET_STATS     = 0xB1,  /* Reset coherency statistics */

    /* DCD/GFAM/MH-SLD fabric-memory commands */
    CXL_GPU_CMD_DCD_ADD             = 0xC0,  /* params: base, size, tag */
    CXL_GPU_CMD_DCD_RELEASE         = 0xC1,  /* params: base, size, tag */
    CXL_GPU_CMD_DCD_GET_INFO        = 0xC2,  /* results: total, alloc, free */
    CXL_GPU_CMD_GFAM_GRANT          = 0xC8,  /* params: host, base, size, perms */
    CXL_GPU_CMD_GFAM_REVOKE         = 0xC9,  /* params: host, base, size */
    CXL_GPU_CMD_GFAM_GET_INFO       = 0xCA,  /* results: hosts, mappings, deny */
    CXL_GPU_CMD_MHSLD_GET_INFO      = 0xD0,  /* results: heads, current, stats */
    CXL_GPU_CMD_MHSLD_SET_HEAD      = 0xD1,  /* params: head_id */
} CXLGPUCommand;

/* Guest-visible representation of a CUDA_KERNEL_NODE_PARAMS response. BAR2 carries
 * only ids, scalars and copied argument bytes; host addresses never cross it. */
typedef struct CXLGraphKernelNodeParamsWire {
    uint32_t function_id;
    uint32_t grid_dim_x;
    uint32_t grid_dim_y;
    uint32_t grid_dim_z;
    uint32_t block_dim_x;
    uint32_t block_dim_y;
    uint32_t block_dim_z;
    uint32_t shared_mem_bytes;
    uint32_t num_args;
    uint32_t reserved;
    uint64_t param_extent;
} CXLGraphKernelNodeParamsWire;

typedef struct CXLGraphKernelNodeParamWire {
    uint64_t offset;
    uint64_t size;
} CXLGraphKernelNodeParamWire;

typedef struct CXLFunctionParamLayoutWire {
    uint32_t num_args;
    uint32_t reserved;
    uint64_t extent;
    CXLGraphKernelNodeParamWire params[64];
} CXLFunctionParamLayoutWire;

/* P2P register offsets and peer types: defined in cxl_p2p_dma.h */

/* Coherent pool register offsets (in GPU command region) */
#define CXL_GPU_REG_COH_POOL_BASE   0x0300  /* Coherent pool base offset */
#define CXL_GPU_REG_COH_POOL_SIZE   0x0308  /* Coherent pool total size */
#define CXL_GPU_REG_COH_POOL_FREE   0x0310  /* Coherent pool free space */
#define CXL_GPU_REG_COH_DIR_SIZE    0x0318  /* Directory size (entries) */
#define CXL_GPU_REG_COH_DIR_USED    0x0320  /* Directory used entries */

/* DCD/GFAM/MH-SLD status registers */
#define CXL_GPU_REG_DCD_TOTAL       0x0330  /* DCD total capacity */
#define CXL_GPU_REG_DCD_ALLOCATED   0x0338  /* DCD allocated capacity */
#define CXL_GPU_REG_DCD_FREE        0x0340  /* DCD free capacity */
#define CXL_GPU_REG_DCD_EXTENTS     0x0348  /* Active DCD extent count */
#define CXL_GPU_REG_GFAM_HOSTS      0x0350  /* Configured GFAM hosts */
#define CXL_GPU_REG_GFAM_MAPPINGS   0x0358  /* Active GFAM mappings */
#define CXL_GPU_REG_GFAM_DENIED     0x0360  /* Denied GFAM accesses */
#define CXL_GPU_REG_MHSLD_HEADS     0x0370  /* MH-SLD head count */
#define CXL_GPU_REG_MHSLD_HEAD_ID   0x0378  /* Local MH-SLD head id */
#define CXL_GPU_REG_MHSLD_CONFLICTS 0x0380  /* MH-SLD coherency conflicts */
#define CXL_GPU_REG_MHSLD_INV       0x0388  /* MH-SLD invalidations */

/* Bias mode constants */
#define CXL_BIAS_HOST               0       /* Host-biased: CPU is coherence home */
#define CXL_BIAS_DEVICE             1       /* Device-biased: GPU snoop filter is home */

/* DCD/GFAM permission bits */
#define CXL_DCD_PERM_READ           (1 << 0)
#define CXL_DCD_PERM_WRITE          (1 << 1)
#define CXL_DCD_PERM_ATOMIC         (1 << 2)
#define CXL_DCD_PERM_SHARED         (1 << 3)
#define CXL_DCD_PERM_ALL            (CXL_DCD_PERM_READ | \
                                     CXL_DCD_PERM_WRITE | \
                                     CXL_DCD_PERM_ATOMIC | \
                                     CXL_DCD_PERM_SHARED)

/* Error codes (matching CUDA error codes) */
typedef enum {
    CXL_GPU_SUCCESS                     = 0,
    CXL_GPU_ERROR_INVALID_VALUE         = 1,
    CXL_GPU_ERROR_OUT_OF_MEMORY         = 2,
    CXL_GPU_ERROR_NOT_INITIALIZED       = 3,
    CXL_GPU_ERROR_DEINITIALIZED         = 4,
    CXL_GPU_ERROR_NO_DEVICE             = 100,
    CXL_GPU_ERROR_INVALID_DEVICE        = 101,
    CXL_GPU_ERROR_INVALID_CONTEXT       = 201,
    CXL_GPU_ERROR_INVALID_HANDLE        = 400,
    CXL_GPU_ERROR_NOT_FOUND             = 500,
    CXL_GPU_ERROR_NOT_READY             = 600,
    CXL_GPU_ERROR_LAUNCH_FAILED         = 700,
    CXL_GPU_ERROR_INVALID_PTX           = 800,
    CXL_GPU_ERROR_NOT_SUPPORTED         = 801,
    CXL_GPU_ERROR_UNKNOWN               = 999,
} CXLGPUError;

/* Memory allocation info */
typedef struct {
    uint64_t device_ptr;
    uint64_t size;
    uint32_t flags;
    uint32_t reserved;
} CXLGPUMemInfo;

/* Kernel launch configuration */
typedef struct {
    uint32_t grid_dim_x;
    uint32_t grid_dim_y;
    uint32_t grid_dim_z;
    uint32_t block_dim_x;
    uint32_t block_dim_y;
    uint32_t block_dim_z;
    uint32_t shared_mem_bytes;
    uint32_t stream;
    uint64_t function_handle;
    uint64_t args_ptr;      /* Pointer to kernel arguments in data region */
    uint32_t num_args;
    uint32_t reserved;
} CXLGPUKernelLaunch;

/* Bulk transfer descriptor (for large memory operations) */
typedef struct {
    uint64_t host_addr;      /* Host virtual address (for BAR4 offset) */
    uint64_t device_ptr;     /* GPU device pointer */
    uint64_t size;           /* Transfer size in bytes */
    uint32_t flags;          /* Transfer flags */
    uint32_t stream;         /* Stream for async transfers */
} CXLGPUBulkTransfer;

/* Bulk transfer flags */
#define CXL_GPU_BULK_FLAG_ASYNC     (1 << 0)  /* Asynchronous transfer */
#define CXL_GPU_BULK_FLAG_COHERENT  (1 << 1)  /* Use CXL.cache coherency */
#define CXL_GPU_BULK_FLAG_NOCACHE   (1 << 2)  /* Bypass cache (write-combining) */

/* Cache operation descriptor */
typedef struct {
    uint64_t addr;           /* Start address */
    uint64_t size;           /* Size of region */
    uint32_t operation;      /* Flush/Invalidate/Writeback */
    uint32_t flags;          /* Operation flags */
} CXLGPUCacheOp;

#endif /* CXL_TYPE2_GPU_CMD_H */
