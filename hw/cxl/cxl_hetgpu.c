/*
 * CXL Type 2 Device - hetGPU Backend Integration
 * Provides CUDA compatibility layer for CXL Type 2 GPU accelerators
 *
 * This file implements stub functions for the hetGPU backend.
 * When the actual hetGPU library is available, it will be loaded dynamically.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "qemu/thread.h"
#include "qemu/timer.h"
#include "hw/cxl/cxl_hetgpu.h"
#include <dlfcn.h>

#ifndef DEFAULT_HETGPU_LIB_PATH
#define DEFAULT_HETGPU_LIB_PATH NULL
#endif

/* Default device properties for simulation mode */
static const HetGPUDeviceProps default_props = {
    .name = "Virtual GPU (TMatmul)",
    .total_memory = 4ULL * 1024 * 1024 * 1024,  /* 4GB */
    .compute_capability_major = 8,
    .compute_capability_minor = 0,
    .max_threads_per_block = 1024,
    .max_block_dim = {1024, 1024, 64},
    .max_grid_dim = {65535, 65535, 65535},
    .warp_size = 32,
    .multiprocessor_count = 80,
    .clock_rate_khz = 1500000,
    .memory_clock_rate_khz = 5000000,
    .memory_bus_width = 256,
    .l2_cache_size = 6 * 1024 * 1024,  /* 6MB */
    .supports_managed_memory = true,
    .supports_coherent_memory = true,
    .backend_type = HETGPU_BACKEND_SIMULATION,
};

/* Function pointer types for dynamic loading */
typedef int (*cuInit_fn)(unsigned int);
typedef int (*cuDriverGetVersion_fn)(int *);
typedef int (*cuDeviceGetCount_fn)(int *);
typedef int (*cuDeviceGet_fn)(int *, int);
typedef int (*cuDeviceGetName_fn)(char *, int, int);
typedef int (*cuDeviceTotalMem_fn)(size_t *, int);
typedef int (*cuDeviceGetAttribute_fn)(int *, int, int);
typedef int (*cuMemGetInfo_fn)(size_t *, size_t *);
typedef int (*cuCtxCreate_fn)(void **, unsigned int, int);
typedef int (*cuCtxDestroy_fn)(void *);
typedef int (*cuCtxSynchronize_fn)(void);
typedef int (*cuMemAlloc_fn)(uint64_t *, size_t);
typedef int (*cuMemFree_fn)(uint64_t);
typedef int (*cuMemcpyHtoD_fn)(uint64_t, const void *, size_t);
typedef int (*cuMemcpyDtoH_fn)(void *, uint64_t, size_t);
typedef int (*cuMemcpyHtoDAsync_fn)(uint64_t, const void *, size_t, void *);
typedef int (*cuMemHostAlloc_fn)(void **, size_t, unsigned int);
typedef int (*cuMemFreeHost_fn)(void *);
typedef int (*cuMemcpyDtoD_fn)(uint64_t, uint64_t, size_t);
typedef int (*cuPointerGetAttribute_fn)(void *, int, uint64_t);
typedef int (*cuModuleLoadData_fn)(void **, const void *);
typedef int (*cuModuleUnload_fn)(void *);
typedef int (*cuModuleGetLoadingMode_fn)(int *);
typedef int (*cuModuleGetFunction_fn)(void **, void *, const char *);
typedef int (*cuModuleGetGlobal_fn)(uint64_t *, size_t *, void *, const char *);
typedef int (*cuFuncGetParamInfo_fn)(void *, size_t, size_t *, size_t *);
typedef int (*cuFuncGetAttribute_fn)(int *, int, void *);
typedef int (*cuFuncSetAttribute_fn)(void *, int, int);
typedef int (*cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_fn)(int *, void *, int, size_t,
                                                                         unsigned int);
typedef int (*cuLaunchKernel_fn)(void *, unsigned int, unsigned int, unsigned int,
                                  unsigned int, unsigned int, unsigned int,
                                  unsigned int, void *, void **, void **);
typedef int (*cuGraphExecKernelNodeSetParams_fn)(void *, void *,
                                                  const CudaKernelNodeParams *);
typedef int (*cuGraphKernelNodeGetParams_fn)(void *, CudaKernelNodeParams *);
typedef int (*cuGraphExecDestroy_fn)(void *);
typedef int (*cuGraphLaunch_fn)(void *, void *);
typedef int (*cuGraphDestroy_fn)(void *);
typedef int (*cuGraphInstantiate_fn)(void **, void *, void **, char *, size_t);
typedef int (*cuGraphGetNodes_fn)(void *, void **, size_t *);
typedef int (*cuGraphNodeGetType_fn)(void *, int *);
typedef int (*cuLinkCreate_fn)(unsigned int, int *, void **, void **);
typedef int (*cuLinkAddData_fn)(void *, int, void *, size_t, const char *, unsigned int, int *, void **);
typedef int (*cuLinkComplete_fn)(void *, void **, size_t *);
typedef int (*cuLinkDestroy_fn)(void *);
typedef int (*cuCtxGetLimit_fn)(size_t *, int);
typedef int (*cuStreamCreate_fn)(void **, unsigned int);
typedef int (*cuStreamDestroy_fn)(void *);
typedef int (*cuStreamSynchronize_fn)(void *);
typedef int (*cuStreamWaitEvent_fn)(void *, void *, unsigned int);
typedef int (*cuStreamWaitValue32_fn)(void *, uint64_t, uint32_t, unsigned int);
typedef int (*cuStreamBatchMemOp_fn)(void *, unsigned int, void *, unsigned int);
typedef int (*cuStreamGetCaptureInfo_v2_fn)(void *, int *, uint64_t *, void **, void ***, size_t *);
typedef int (*cuStreamGetCtx_fn)(void *, void **);
typedef int (*cuStreamBeginCapture_fn)(void *, int);
typedef int (*cuStreamEndCapture_fn)(void *, void **);
typedef int (*cuStreamIsCapturing_fn)(void *, int *);
typedef int (*cuEventCreate_fn)(void **, unsigned int);
typedef int (*cuEventDestroy_fn)(void *);
typedef int (*cuEventRecord_fn)(void *, void *);
typedef int (*cuEventQuery_fn)(void *);
typedef int (*cuEventSynchronize_fn)(void *);
typedef int (*cuEventElapsedTime_fn)(float *, void *, void *);
typedef int (*cuMemPrefetchAsync_fn)(uint64_t, size_t, int, void *);
typedef int (*cuCtxEnablePeerAccess_fn)(void *, unsigned int);
typedef int (*cuCtxDisablePeerAccess_fn)(void *);
typedef int (*cuDeviceCanAccessPeer_fn)(int *, int, int);
typedef int (*cuCtxPushCurrent_fn)(void *);
typedef int (*cuCtxPopCurrent_fn)(void **);
typedef int (*cuCtxSetCurrent_fn)(void *);
typedef int (*cuCtxGetCurrent_fn)(void **);
typedef int (*cuGetErrorString_fn)(int, const char **);
typedef int (*cuGetErrorName_fn)(int, const char **);

/* CUDA device attribute constants */
#define CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK 1
#define CU_DEVICE_ATTRIBUTE_WARP_SIZE 10
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 16
#define CU_DEVICE_ATTRIBUTE_CLOCK_RATE 13
#define CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE 36
#define CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH 37
#define CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE 38
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76

#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_VALUE 1
#define CUDA_ERROR_NOT_INITIALIZED 3
#define CUDA_ERROR_INVALID_CONTEXT 201
#define CUDA_ERROR_INVALID_HANDLE 400
#define CUDA_ERROR_NOT_SUPPORTED 801

static __thread uint64_t g_cuda_trace_call_id;
static __thread uint32_t g_cuda_trace_occurrence;

void hetgpu_cuda_trace_set_call_id(uint64_t call_id) {
  g_cuda_trace_call_id = call_id;
  g_cuda_trace_occurrence = 0;
}

#define HETGPU_CUDA_CALL(field, ...)                                           \
  ({                                                                           \
    uint32_t _driver_occurrence = ++g_cuda_trace_occurrence;                   \
    int64_t _driver_start_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);             \
    qemu_log("CXL TYPE2 TRACE driver_begin call_id=0x%016" PRIx64              \
             " occurrence=%u symbol=%s host_ns=%" PRId64 "\n",                 \
             g_cuda_trace_call_id, _driver_occurrence, #field,                 \
             _driver_start_ns);                                                \
    int _driver_result = g_cuda_funcs.field(__VA_ARGS__);                      \
    int64_t _driver_end_ns = qemu_clock_get_ns(QEMU_CLOCK_HOST);               \
    qemu_log("CXL TYPE2 TRACE driver_end call_id=0x%016" PRIx64                \
             " occurrence=%u symbol=%s host_ns=%" PRId64                       \
             " duration_ns=%" PRId64 " result=%d\n",                           \
             g_cuda_trace_call_id, _driver_occurrence, #field, _driver_end_ns, \
             _driver_end_ns - _driver_start_ns, _driver_result);               \
    _driver_result;                                                            \
  })

/* Loaded function pointers */
static struct {
    cuInit_fn cuInit;
    cuDriverGetVersion_fn cuDriverGetVersion;
    cuDeviceGetCount_fn cuDeviceGetCount;
    cuDeviceGet_fn cuDeviceGet;
    cuDeviceGetName_fn cuDeviceGetName;
    cuDeviceTotalMem_fn cuDeviceTotalMem;
    cuDeviceGetAttribute_fn cuDeviceGetAttribute;
    cuMemGetInfo_fn cuMemGetInfo;
    cuCtxCreate_fn cuCtxCreate;
    cuCtxDestroy_fn cuCtxDestroy;
    cuCtxSynchronize_fn cuCtxSynchronize;
    cuMemAlloc_fn cuMemAlloc;
    cuMemFree_fn cuMemFree;
    cuMemcpyHtoD_fn cuMemcpyHtoD;
    cuMemcpyDtoH_fn cuMemcpyDtoH;
    cuMemcpyHtoDAsync_fn cuMemcpyHtoDAsync;
    cuMemHostAlloc_fn cuMemHostAlloc;
    cuMemFreeHost_fn cuMemFreeHost;
    cuMemcpyDtoD_fn cuMemcpyDtoD;
    cuPointerGetAttribute_fn cuPointerGetAttribute;
    cuModuleLoadData_fn cuModuleLoadData;
    cuModuleUnload_fn cuModuleUnload;
    cuModuleGetLoadingMode_fn cuModuleGetLoadingMode;
    cuModuleGetFunction_fn cuModuleGetFunction;
    cuModuleGetGlobal_fn cuModuleGetGlobal;
    cuFuncGetParamInfo_fn cuFuncGetParamInfo;
    cuFuncGetAttribute_fn cuFuncGetAttribute;
    cuFuncSetAttribute_fn cuFuncSetAttribute;
    cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_fn cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags;
    cuLaunchKernel_fn cuLaunchKernel;
    cuGraphExecKernelNodeSetParams_fn cuGraphExecKernelNodeSetParams;
    cuGraphKernelNodeGetParams_fn cuGraphKernelNodeGetParams;
    cuGraphExecDestroy_fn cuGraphExecDestroy;
    cuGraphLaunch_fn cuGraphLaunch;
    cuGraphDestroy_fn cuGraphDestroy;
    cuGraphInstantiate_fn cuGraphInstantiate;
    cuGraphGetNodes_fn cuGraphGetNodes;
    cuGraphNodeGetType_fn cuGraphNodeGetType;
    cuLinkCreate_fn cuLinkCreate;
    cuLinkAddData_fn cuLinkAddData;
    cuLinkComplete_fn cuLinkComplete;
    cuLinkDestroy_fn cuLinkDestroy;
    cuCtxGetLimit_fn cuCtxGetLimit;
    cuStreamCreate_fn cuStreamCreate;
    cuStreamDestroy_fn cuStreamDestroy;
    cuStreamSynchronize_fn cuStreamSynchronize;
    cuStreamWaitEvent_fn cuStreamWaitEvent;
    cuStreamWaitValue32_fn cuStreamWaitValue32;
    cuStreamBatchMemOp_fn cuStreamBatchMemOp;
    cuStreamGetCaptureInfo_v2_fn cuStreamGetCaptureInfo_v2;
    cuStreamGetCtx_fn cuStreamGetCtx;
    cuStreamBeginCapture_fn cuStreamBeginCapture;
    cuStreamEndCapture_fn cuStreamEndCapture;
    cuStreamIsCapturing_fn cuStreamIsCapturing;
    cuEventCreate_fn cuEventCreate;
    cuEventDestroy_fn cuEventDestroy;
    cuEventRecord_fn cuEventRecord;
    cuEventQuery_fn cuEventQuery;
    cuEventSynchronize_fn cuEventSynchronize;
    cuEventElapsedTime_fn cuEventElapsedTime;
    cuMemPrefetchAsync_fn cuMemPrefetchAsync;
    cuCtxEnablePeerAccess_fn cuCtxEnablePeerAccess;
    cuCtxDisablePeerAccess_fn cuCtxDisablePeerAccess;
    cuDeviceCanAccessPeer_fn cuDeviceCanAccessPeer;
    cuCtxPushCurrent_fn cuCtxPushCurrent;
    cuCtxPopCurrent_fn cuCtxPopCurrent;
    cuCtxSetCurrent_fn cuCtxSetCurrent;
    cuCtxGetCurrent_fn cuCtxGetCurrent;
    cuGetErrorString_fn cuGetErrorString;
    cuGetErrorName_fn cuGetErrorName;
} g_cuda_funcs = {0};

/* Global mutex for multi-device context switching safety */
static QemuMutex g_cuda_mutex;
static bool g_cuda_mutex_initialized = false;
static bool g_cuda_lib_initialized = false;
static void *g_cuda_lib_handle = NULL;
static char *g_cuda_lib_path;
static int g_cuda_driver_version;

static void ensure_cuda_mutex_init(void)
{
    if (!g_cuda_mutex_initialized) {
        qemu_mutex_init(&g_cuda_mutex);
        g_cuda_mutex_initialized = true;
    }
}

static HetGPUError hetgpu_init_internal(HetGPUState *state,
                                        HetGPUBackendType backend,
                                        int device_index,
                                        const char *hetgpu_lib_path,
                                        bool formal_case_strict)
{
    if (!state) {
        return HETGPU_ERROR_INVALID_VALUE;
    }

    ensure_cuda_mutex_init();

    memset(state, 0, sizeof(*state));
    state->backend = backend;
    state->device_index = device_index;
    state->formal_case_strict = formal_case_strict;

    if (formal_case_strict &&
        (backend != HETGPU_BACKEND_NVIDIA || !hetgpu_lib_path ||
         hetgpu_lib_path[0] == '\0')) {
        qemu_log("CXL hetGPU: formal init requires explicit NVIDIA "
                 "backend and library\n");
        return HETGPU_ERROR_INVALID_VALUE;
    }

    /* Try to load hetGPU library (only once for all devices) */
    const char *lib_path = hetgpu_lib_path;
    if (!formal_case_strict && (!lib_path || lib_path[0] == '\0')) {
        lib_path = DEFAULT_HETGPU_LIB_PATH;
    }
    if (!formal_case_strict && (!lib_path || lib_path[0] == '\0')) {
        lib_path = "/usr/lib/x86_64-linux-gnu/libcuda.so";
    }

    fprintf(stderr, "CXL hetGPU: ========================================\n");
    fprintf(stderr, "CXL hetGPU: Initializing GPU backend for device %d\n", device_index);
    fprintf(stderr, "CXL hetGPU: Library path: %s\n", lib_path);
    fprintf(stderr, "CXL hetGPU: Library already loaded: %s\n",
            g_cuda_lib_initialized ? "yes" : "no");
    fprintf(stderr, "CXL hetGPU: ========================================\n");
    fflush(stderr);

    qemu_mutex_lock(&g_cuda_mutex);

    /* Only load library and call cuInit once across all devices */
    if (formal_case_strict && g_cuda_lib_initialized) {
        g_autofree char *requested = g_canonicalize_filename(lib_path, NULL);

        if (!g_cuda_lib_path || strcmp(requested, g_cuda_lib_path) != 0) {
            qemu_log("CXL hetGPU: formal library mismatch requested=%s "
                     "loaded=%s\n", requested,
                     g_cuda_lib_path ? g_cuda_lib_path : "(unknown)");
            qemu_mutex_unlock(&g_cuda_mutex);
            return HETGPU_ERROR_INVALID_VALUE;
        }
    }

    if (!g_cuda_lib_initialized) {
        g_cuda_lib_handle = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL);
        if (!formal_case_strict && !g_cuda_lib_handle) {
            fprintf(stderr, "CXL hetGPU: FAILED to load library: %s\n", dlerror());
            fprintf(stderr, "CXL hetGPU: Trying alternate path /usr/lib64/libcuda.so\n");
            fflush(stderr);
            lib_path = "/usr/lib64/libcuda.so";
            g_cuda_lib_handle = dlopen("/usr/lib64/libcuda.so", RTLD_NOW | RTLD_GLOBAL);
        }
        if (!formal_case_strict && !g_cuda_lib_handle) {
            fprintf(stderr, "CXL hetGPU: FAILED to load library: %s\n", dlerror());
            fprintf(stderr, "CXL hetGPU: Trying alternate path libcuda.so.1\n");
            fflush(stderr);
            lib_path = "libcuda.so.1";
            g_cuda_lib_handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        }

        if (g_cuda_lib_handle) {
            fprintf(stderr, "CXL hetGPU: Successfully loaded CUDA library\n");
            fflush(stderr);

            /* Load function pointers */
            g_cuda_funcs.cuInit = dlsym(g_cuda_lib_handle, "cuInit");
            g_cuda_funcs.cuDriverGetVersion = dlsym(g_cuda_lib_handle, "cuDriverGetVersion");
            g_cuda_funcs.cuDeviceGetCount = dlsym(g_cuda_lib_handle, "cuDeviceGetCount");
            g_cuda_funcs.cuDeviceGet = dlsym(g_cuda_lib_handle, "cuDeviceGet");
            g_cuda_funcs.cuDeviceGetName = dlsym(g_cuda_lib_handle, "cuDeviceGetName");
            g_cuda_funcs.cuDeviceTotalMem = dlsym(g_cuda_lib_handle, "cuDeviceTotalMem_v2");
            g_cuda_funcs.cuDeviceGetAttribute = dlsym(g_cuda_lib_handle, "cuDeviceGetAttribute");
            g_cuda_funcs.cuMemGetInfo = dlsym(g_cuda_lib_handle, "cuMemGetInfo_v2");
            g_cuda_funcs.cuCtxCreate = dlsym(g_cuda_lib_handle, "cuCtxCreate_v2");
            g_cuda_funcs.cuCtxDestroy = dlsym(g_cuda_lib_handle, "cuCtxDestroy_v2");
            g_cuda_funcs.cuCtxSynchronize = dlsym(g_cuda_lib_handle, "cuCtxSynchronize");
            g_cuda_funcs.cuMemAlloc = dlsym(g_cuda_lib_handle, "cuMemAlloc_v2");
            g_cuda_funcs.cuMemFree = dlsym(g_cuda_lib_handle, "cuMemFree_v2");
            g_cuda_funcs.cuMemcpyHtoD = dlsym(g_cuda_lib_handle, "cuMemcpyHtoD_v2");
            g_cuda_funcs.cuMemcpyDtoH = dlsym(g_cuda_lib_handle, "cuMemcpyDtoH_v2");
            g_cuda_funcs.cuMemcpyHtoDAsync =
                dlsym(g_cuda_lib_handle, "cuMemcpyHtoDAsync_v2");
            g_cuda_funcs.cuMemHostAlloc =
                dlsym(g_cuda_lib_handle, "cuMemHostAlloc");
            g_cuda_funcs.cuMemFreeHost =
                dlsym(g_cuda_lib_handle, "cuMemFreeHost");
            g_cuda_funcs.cuMemcpyDtoD = dlsym(g_cuda_lib_handle, "cuMemcpyDtoD_v2");
            g_cuda_funcs.cuPointerGetAttribute = dlsym(g_cuda_lib_handle, "cuPointerGetAttribute");
            g_cuda_funcs.cuModuleLoadData = dlsym(g_cuda_lib_handle, "cuModuleLoadData");
            g_cuda_funcs.cuModuleUnload = dlsym(g_cuda_lib_handle, "cuModuleUnload");
            g_cuda_funcs.cuModuleGetLoadingMode = dlsym(g_cuda_lib_handle, "cuModuleGetLoadingMode");
            g_cuda_funcs.cuModuleGetFunction = dlsym(g_cuda_lib_handle, "cuModuleGetFunction");
            g_cuda_funcs.cuModuleGetGlobal = dlsym(g_cuda_lib_handle, "cuModuleGetGlobal_v2");
            g_cuda_funcs.cuFuncGetParamInfo = dlsym(g_cuda_lib_handle, "cuFuncGetParamInfo");
            g_cuda_funcs.cuFuncGetAttribute = dlsym(g_cuda_lib_handle, "cuFuncGetAttribute");
            g_cuda_funcs.cuFuncSetAttribute = dlsym(g_cuda_lib_handle, "cuFuncSetAttribute");
            g_cuda_funcs.cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags =
                dlsym(g_cuda_lib_handle, "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
            g_cuda_funcs.cuLaunchKernel = dlsym(g_cuda_lib_handle, "cuLaunchKernel");
            g_cuda_funcs.cuGraphExecKernelNodeSetParams =
                dlsym(g_cuda_lib_handle, "cuGraphExecKernelNodeSetParams_v2");
            g_cuda_funcs.cuGraphKernelNodeGetParams =
                dlsym(g_cuda_lib_handle, "cuGraphKernelNodeGetParams_v2");
            g_cuda_funcs.cuGraphExecDestroy =
                dlsym(g_cuda_lib_handle, "cuGraphExecDestroy");
            g_cuda_funcs.cuGraphLaunch =
                dlsym(g_cuda_lib_handle, "cuGraphLaunch");
            g_cuda_funcs.cuGraphDestroy =
                dlsym(g_cuda_lib_handle, "cuGraphDestroy");
            g_cuda_funcs.cuGraphInstantiate =
                dlsym(g_cuda_lib_handle, "cuGraphInstantiate_v2");
            g_cuda_funcs.cuGraphGetNodes =
                dlsym(g_cuda_lib_handle, "cuGraphGetNodes");
            g_cuda_funcs.cuGraphNodeGetType =
                dlsym(g_cuda_lib_handle, "cuGraphNodeGetType");
            g_cuda_funcs.cuLinkCreate = dlsym(g_cuda_lib_handle, "cuLinkCreate_v2");
            if (!g_cuda_funcs.cuLinkCreate)
                g_cuda_funcs.cuLinkCreate = dlsym(g_cuda_lib_handle, "cuLinkCreate");
            g_cuda_funcs.cuLinkAddData = dlsym(g_cuda_lib_handle, "cuLinkAddData_v2");
            if (!g_cuda_funcs.cuLinkAddData)
                g_cuda_funcs.cuLinkAddData = dlsym(g_cuda_lib_handle, "cuLinkAddData");
            g_cuda_funcs.cuLinkComplete = dlsym(g_cuda_lib_handle, "cuLinkComplete");
            g_cuda_funcs.cuLinkDestroy = dlsym(g_cuda_lib_handle, "cuLinkDestroy");
            g_cuda_funcs.cuCtxGetLimit = dlsym(g_cuda_lib_handle, "cuCtxGetLimit");
            g_cuda_funcs.cuStreamCreate = dlsym(g_cuda_lib_handle, "cuStreamCreate");
            g_cuda_funcs.cuStreamDestroy = dlsym(g_cuda_lib_handle, "cuStreamDestroy_v2");
            g_cuda_funcs.cuStreamSynchronize = dlsym(g_cuda_lib_handle, "cuStreamSynchronize");
            g_cuda_funcs.cuStreamWaitEvent = dlsym(g_cuda_lib_handle, "cuStreamWaitEvent");
            g_cuda_funcs.cuStreamWaitValue32 = dlsym(g_cuda_lib_handle, "cuStreamWaitValue32");
            g_cuda_funcs.cuStreamBatchMemOp = dlsym(g_cuda_lib_handle, "cuStreamBatchMemOp");
            g_cuda_funcs.cuStreamGetCaptureInfo_v2 = dlsym(g_cuda_lib_handle, "cuStreamGetCaptureInfo_v2");
            g_cuda_funcs.cuStreamGetCtx = dlsym(g_cuda_lib_handle, "cuStreamGetCtx");
            g_cuda_funcs.cuStreamBeginCapture = dlsym(g_cuda_lib_handle, "cuStreamBeginCapture_v2");
            if (!g_cuda_funcs.cuStreamBeginCapture)
                g_cuda_funcs.cuStreamBeginCapture = dlsym(g_cuda_lib_handle, "cuStreamBeginCapture");
            g_cuda_funcs.cuStreamEndCapture = dlsym(g_cuda_lib_handle, "cuStreamEndCapture");
            g_cuda_funcs.cuStreamIsCapturing = dlsym(g_cuda_lib_handle, "cuStreamIsCapturing");
            g_cuda_funcs.cuEventCreate = dlsym(g_cuda_lib_handle, "cuEventCreate");
            g_cuda_funcs.cuEventDestroy = dlsym(g_cuda_lib_handle, "cuEventDestroy_v2");
            g_cuda_funcs.cuEventRecord = dlsym(g_cuda_lib_handle, "cuEventRecord");
            g_cuda_funcs.cuEventQuery = dlsym(g_cuda_lib_handle, "cuEventQuery");
            g_cuda_funcs.cuEventSynchronize = dlsym(g_cuda_lib_handle, "cuEventSynchronize");
            g_cuda_funcs.cuEventElapsedTime = dlsym(g_cuda_lib_handle, "cuEventElapsedTime_v2");
            if (!g_cuda_funcs.cuEventElapsedTime)
                g_cuda_funcs.cuEventElapsedTime = dlsym(g_cuda_lib_handle, "cuEventElapsedTime");
            g_cuda_funcs.cuMemPrefetchAsync = dlsym(g_cuda_lib_handle, "cuMemPrefetchAsync");
            g_cuda_funcs.cuCtxEnablePeerAccess = dlsym(g_cuda_lib_handle, "cuCtxEnablePeerAccess");
            g_cuda_funcs.cuCtxDisablePeerAccess = dlsym(g_cuda_lib_handle, "cuCtxDisablePeerAccess");
            g_cuda_funcs.cuDeviceCanAccessPeer = dlsym(g_cuda_lib_handle, "cuDeviceCanAccessPeer");
            g_cuda_funcs.cuCtxPushCurrent = dlsym(g_cuda_lib_handle, "cuCtxPushCurrent_v2");
            g_cuda_funcs.cuCtxPopCurrent = dlsym(g_cuda_lib_handle, "cuCtxPopCurrent_v2");
            g_cuda_funcs.cuCtxSetCurrent = dlsym(g_cuda_lib_handle, "cuCtxSetCurrent");
            g_cuda_funcs.cuCtxGetCurrent = dlsym(g_cuda_lib_handle, "cuCtxGetCurrent");
            g_cuda_funcs.cuGetErrorString = dlsym(g_cuda_lib_handle, "cuGetErrorString");
            g_cuda_funcs.cuGetErrorName = dlsym(g_cuda_lib_handle, "cuGetErrorName");

            g_free(g_cuda_lib_path);
            g_cuda_lib_path = g_canonicalize_filename(lib_path, NULL);

            fprintf(stderr,
                    "CXL hetGPU: Loaded CUDA functions - cuInit=%p, cuDriverGetVersion=%p, cuCtxSetCurrent=%p\n",
                    g_cuda_funcs.cuInit, g_cuda_funcs.cuDriverGetVersion,
                    g_cuda_funcs.cuCtxSetCurrent);
            fflush(stderr);

            if (!g_cuda_funcs.cuDriverGetVersion) {
                qemu_log("CXL hetGPU: cuDriverGetVersion symbol not found\n");
                qemu_mutex_unlock(&g_cuda_mutex);
                goto simulation_fallback;
            }
            int version_err = HETGPU_CUDA_CALL(cuDriverGetVersion,
                                                &g_cuda_driver_version);
            if (version_err != CUDA_SUCCESS || g_cuda_driver_version <= 0) {
                qemu_log("CXL hetGPU: cuDriverGetVersion failed result=%d version=%d\n",
                         version_err, g_cuda_driver_version);
                qemu_mutex_unlock(&g_cuda_mutex);
                goto simulation_fallback;
            }
            qemu_log("CXL hetGPU: CUDA Driver API version=%d\n",
                     g_cuda_driver_version);

            if (g_cuda_funcs.cuInit) {
                int err = HETGPU_CUDA_CALL(cuInit, 0);
                fprintf(stderr, "CXL hetGPU: cuInit returned err=%d\n", err);
                fflush(stderr);
                if (err != 0) {
                    qemu_log("CXL hetGPU: cuInit failed with error %d\n", err);
                    qemu_mutex_unlock(&g_cuda_mutex);
                    goto simulation_fallback;
                }
            }
            g_cuda_lib_initialized = true;
        }
    }

    /* Store library handle reference (shared, don't dlclose individually) */
    state->hetgpu_lib = g_cuda_lib_handle;
    state->driver_version = g_cuda_driver_version;
    if (state->hetgpu_lib) {
        state->kimi_case_begin_v1 = dlsym(state->hetgpu_lib,
                                          "hetgpu_kimi_case_begin_v1");
        state->kimi_case_end_v1 = dlsym(state->hetgpu_lib,
                                        "hetgpu_kimi_case_end_v1");
    }

    qemu_mutex_unlock(&g_cuda_mutex);

    if (formal_case_strict &&
        (!state->kimi_case_begin_v1 || !state->kimi_case_end_v1)) {
        qemu_log("CXL hetGPU: formal Kimi case symbols are missing "
                 "begin=%p end=%p\n",
                 state->kimi_case_begin_v1, state->kimi_case_end_v1);
        goto simulation_fallback;
    }

    if (state->hetgpu_lib && g_cuda_lib_initialized) {
        /* Library loaded and cuInit succeeded — create per-device context */
        int err = 0;
        int cuda_dev = 0;
        size_t total_mem = 0;
        int attr_val = 0;
        void *ctx = NULL;

        /* Get device handle — with MIG, different device_index = different MIG instance */
        if (g_cuda_funcs.cuDeviceGet) {
            err = HETGPU_CUDA_CALL(cuDeviceGet, &cuda_dev, device_index);
            if (err != 0) {
                qemu_log("CXL hetGPU: cuDeviceGet(%d) failed: %d\n", device_index, err);
                qemu_log("CXL hetGPU: If using MIG, ensure MIG instances are configured\n");
                goto simulation_fallback;
            }
            qemu_log("CXL hetGPU: Got CUDA device %d for device_index %d\n",
                     cuda_dev, device_index);
        }

        /* Create per-device context */
        if (g_cuda_funcs.cuCtxCreate) {
            qemu_log("CXL hetGPU: Calling cuCtxCreate_v2 for device %d\n", cuda_dev);
            err = HETGPU_CUDA_CALL(cuCtxCreate, &ctx, 0, cuda_dev);

            if (err != 0) {
                const char *err_name = "UNKNOWN";
                const char *err_str = "Unknown error";
                if (g_cuda_funcs.cuGetErrorName) {
                    HETGPU_CUDA_CALL(cuGetErrorName, err, &err_name);
                }
                if (g_cuda_funcs.cuGetErrorString) {
                    HETGPU_CUDA_CALL(cuGetErrorString, err, &err_str);
                }
                qemu_log("CXL hetGPU: cuCtxCreate FAILED: %s (%d) - %s\n",
                         err_name, err, err_str);
                goto simulation_fallback;
            } else if (ctx == NULL && formal_case_strict) {
                qemu_log("CXL hetGPU: formal init rejected NULL CUDA "
                         "context\n");
                goto simulation_fallback;
            } else if (ctx == NULL) {
                /* hetGPU returns NULL context - use hetGPU managed mode */
                qemu_log("CXL hetGPU: cuCtxCreate returned NULL context\n");
                qemu_log("CXL hetGPU: Using hetGPU managed mode (backend=%d)\n", backend);

                state->initialized = true;
                state->backend = (backend == HETGPU_BACKEND_AUTO) ?
                                 HETGPU_BACKEND_NVIDIA : backend;
                state->context = NULL;
                state->cuda_device = cuda_dev;

                state->props = default_props;
                if (g_cuda_funcs.cuDeviceGetName) {
                    HETGPU_CUDA_CALL(cuDeviceGetName, state->props.name,
                                                 sizeof(state->props.name), cuda_dev);
                }
                if (g_cuda_funcs.cuDeviceTotalMem) {
                    size_t mem = 0;
                    if (HETGPU_CUDA_CALL(cuDeviceTotalMem, &mem, cuda_dev) == 0) {
                        state->props.total_memory = mem;
                    }
                }
                state->props.backend_type = state->backend;

                qemu_log("CXL hetGPU: hetGPU mode initialized: %s, %lu MB, backend=%d\n",
                         state->props.name,
                         (unsigned long)(state->props.total_memory / (1024*1024)),
                         state->backend);
                return HETGPU_SUCCESS;
            }

            qemu_log("CXL hetGPU: Successfully created CUDA context %p for device %d\n",
                     ctx, device_index);

            /* Pop the context cuCtxCreate auto-pushed, we'll use cuCtxSetCurrent */
            if (g_cuda_funcs.cuCtxPopCurrent) {
                void *popped = NULL;
                HETGPU_CUDA_CALL(cuCtxPopCurrent, &popped);
            }
        } else {
            qemu_log("CXL hetGPU: cuCtxCreate_v2 symbol not found\n");
            goto simulation_fallback;
        }

        state->initialized = true;
        state->backend = HETGPU_BACKEND_NVIDIA;
        state->cuda_device = cuda_dev;
        state->context = ctx;

        /* Query real GPU properties */
        state->props = default_props;

        if (g_cuda_funcs.cuDeviceGetName) {
            err = HETGPU_CUDA_CALL(cuDeviceGetName, state->props.name,
                                         sizeof(state->props.name), cuda_dev);
        }
        if (g_cuda_funcs.cuDeviceTotalMem) {
            if (HETGPU_CUDA_CALL(cuDeviceTotalMem, &total_mem, cuda_dev) == 0) {
                state->props.total_memory = total_mem;
                qemu_log("CXL hetGPU: Device %d total memory: %lu MB\n",
                         device_index, (unsigned long)(total_mem / (1024*1024)));
            }
        }
        if (g_cuda_funcs.cuDeviceGetAttribute) {
            if (HETGPU_CUDA_CALL(cuDeviceGetAttribute, &attr_val,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuda_dev) == 0) {
                state->props.compute_capability_major = attr_val;
            }
            if (HETGPU_CUDA_CALL(cuDeviceGetAttribute, &attr_val,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuda_dev) == 0) {
                state->props.compute_capability_minor = attr_val;
            }
            if (HETGPU_CUDA_CALL(cuDeviceGetAttribute, &attr_val,
                    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuda_dev) == 0) {
                state->props.multiprocessor_count = attr_val;
            }
            if (HETGPU_CUDA_CALL(cuDeviceGetAttribute, &attr_val,
                    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, cuda_dev) == 0) {
                state->props.max_threads_per_block = attr_val;
            }
            if (HETGPU_CUDA_CALL(cuDeviceGetAttribute, &attr_val,
                    CU_DEVICE_ATTRIBUTE_WARP_SIZE, cuda_dev) == 0) {
                state->props.warp_size = attr_val;
            }
        }

        state->props.backend_type = HETGPU_BACKEND_NVIDIA;

        qemu_log("CXL hetGPU: Device %d initialized: %s, %lu MB, CC %d.%d\n",
                 device_index, state->props.name,
                 (unsigned long)(state->props.total_memory / (1024*1024)),
                 state->props.compute_capability_major,
                 state->props.compute_capability_minor);
        return HETGPU_SUCCESS;
    }

    /* Fall through to error if library not loaded or cuInit not found */

simulation_fallback:
    /* Do NOT fall back to simulation - require real GPU */
    qemu_log("CXL hetGPU: ERROR - Real GPU initialization failed!\n");
    qemu_log("CXL hetGPU: To use the CXL Type 2 GPU device, ensure:\n");
    qemu_log("CXL hetGPU:   1. NVIDIA GPU is present and working (run nvidia-smi)\n");
    qemu_log("CXL hetGPU:   2. QEMU has GPU access (run as root or add user to video group)\n");
    qemu_log("CXL hetGPU:   3. libcuda.so is accessible at /usr/lib/x86_64-linux-gnu/libcuda.so\n");
    qemu_log("CXL hetGPU:   4. NVIDIA driver is loaded (lsmod | grep nvidia)\n");

    state->initialized = false;
    state->backend = formal_case_strict ? backend : HETGPU_BACKEND_SIMULATION;
    state->context = NULL;

    return HETGPU_ERROR_NO_DEVICE;
}

HetGPUError hetgpu_init(HetGPUState *state, HetGPUBackendType backend,
                        int device_index, const char *hetgpu_lib_path)
{
    return hetgpu_init_internal(state, backend, device_index, hetgpu_lib_path,
                                false);
}

HetGPUError hetgpu_init_formal(HetGPUState *state, HetGPUBackendType backend,
                               int device_index, const char *hetgpu_lib_path)
{
    return hetgpu_init_internal(state, backend, device_index, hetgpu_lib_path,
                                true);
}

HetGPUError hetgpu_reset_formal(HetGPUState *state, HetGPUBackendType backend,
                                int device_index, const char *hetgpu_lib_path)
{
    HetGPUError cleanup_error;

    if (!state) {
        return HETGPU_ERROR_INVALID_VALUE;
    }
    if (state->initialized) {
        if (!state->formal_case_strict) {
            qemu_log("CXL hetGPU: formal reset rejected non-formal "
                     "initialized state\n");
            return HETGPU_ERROR_INVALID_VALUE;
        }
        cleanup_error = hetgpu_cleanup_formal(state);
        if (cleanup_error != HETGPU_SUCCESS) {
            return cleanup_error;
        }
    }
    return hetgpu_init_formal(state, backend, device_index, hetgpu_lib_path);
}

HetGPUError hetgpu_kimi_case_begin(HetGPUState *state,
                                   const HetGPUKimiCaseBeginV1 *input,
                                   HetGPUKimiCaseResultV1 *result)
{
    if (!state || !state->initialized || !state->formal_case_strict ||
        !state->kimi_case_begin_v1 || !input || !result) {
        return HETGPU_ERROR_NOT_INITIALIZED;
    }
    return state->kimi_case_begin_v1(input, result) == 0 ?
           HETGPU_SUCCESS : HETGPU_ERROR_UNKNOWN;
}

HetGPUError hetgpu_kimi_case_end(HetGPUState *state,
                                 const HetGPUKimiCaseEndV1 *input,
                                 HetGPUKimiCaseResultV1 *result)
{
    if (!state || !state->initialized || !state->formal_case_strict ||
        !state->kimi_case_end_v1 || !input || !result) {
        return HETGPU_ERROR_NOT_INITIALIZED;
    }
    return state->kimi_case_end_v1(input, result) == 0 ?
           HETGPU_SUCCESS : HETGPU_ERROR_UNKNOWN;
}

/* Helper to find simulation allocation by device pointer */
static HetGPUSimAlloc *find_sim_alloc(HetGPUState *state, HetGPUDevicePtr dev_ptr)
{
    HetGPUSimAlloc *alloc = state->sim_allocs;
    while (alloc) {
        if (dev_ptr >= alloc->dev_ptr && dev_ptr < alloc->dev_ptr + alloc->size) {
            return alloc;
        }
        alloc = alloc->next;
    }
    return NULL;
}

static HetGPUError hetgpu_cleanup_internal(HetGPUState *state)
{
    HetGPUError result = HETGPU_SUCCESS;

    if (!state) {
        return HETGPU_ERROR_INVALID_VALUE;
    }

    /* Free simulation allocations */
    HetGPUSimAlloc *alloc = state->sim_allocs;
    while (alloc) {
        HetGPUSimAlloc *next = alloc->next;
        if (alloc->host_buffer) {
            g_free(alloc->host_buffer);
        }
        g_free(alloc);
        alloc = next;
    }
    state->sim_allocs = NULL;

    if (state->context && state->backend != HETGPU_BACKEND_SIMULATION &&
        g_cuda_funcs.cuCtxDestroy) {
        int err;

        qemu_mutex_lock(&g_cuda_mutex);
        err = HETGPU_CUDA_CALL(cuCtxDestroy, state->context);
        qemu_mutex_unlock(&g_cuda_mutex);
        if (err != 0) {
            qemu_log("CXL hetGPU: cuCtxDestroy failed context=%p error=%d\n",
                     state->context, err);
            result = HETGPU_ERROR_INVALID_CONTEXT;
        }
    }

    /* Don't dlclose — library handle is shared across all devices */

    qemu_log("CXL hetGPU: Stats - Kernel launches: %lu, Memory ops: %lu, Coherency ops: %lu\n",
             state->kernel_launches, state->memory_ops, state->coherency_ops);

    memset(state, 0, sizeof(*state));
    return result;
}

void hetgpu_cleanup(HetGPUState *state)
{
    (void)hetgpu_cleanup_internal(state);
}

HetGPUError hetgpu_cleanup_formal(HetGPUState *state)
{
    if (!state || !state->formal_case_strict) {
        return HETGPU_ERROR_INVALID_VALUE;
    }
    return hetgpu_cleanup_internal(state);
}

HetGPUError hetgpu_get_device_count(int *count)
{
    fprintf(stderr, "CXL hetGPU: hetgpu_get_device_count called\n");
    fflush(stderr);

    if (!count) {
        fprintf(stderr, "CXL hetGPU: hetgpu_get_device_count - NULL count pointer!\n");
        fflush(stderr);
        return HETGPU_ERROR_INVALID_VALUE;
    }

    fprintf(stderr, "CXL hetGPU: cuDeviceGetCount func ptr = %p\n", (void*)g_cuda_funcs.cuDeviceGetCount);
    fflush(stderr);

    if (g_cuda_funcs.cuDeviceGetCount) {
        *count = -999;  /* Sentinel value to detect if function updates it */
        int err = HETGPU_CUDA_CALL(cuDeviceGetCount, count);
        fprintf(stderr, "CXL hetGPU: cuDeviceGetCount returned err=%d, count=%d\n", err, *count);
        fflush(stderr);
        if (err != 0) {
            fprintf(stderr, "CXL hetGPU: cuDeviceGetCount FAILED with error %d\n", err);
            fflush(stderr);
            return HETGPU_ERROR_NO_DEVICE;
        }
        if (*count <= 0) {
            fprintf(stderr, "CXL hetGPU: WARNING - cuDeviceGetCount returned count=%d\n", *count);
            fflush(stderr);
        }
        return HETGPU_SUCCESS;
    }

    fprintf(stderr, "CXL hetGPU: No cuDeviceGetCount function, using simulation mode\n");
    fflush(stderr);
    *count = 1;  /* Simulation mode: 1 virtual device */
    return HETGPU_SUCCESS;
}

HetGPUError hetgpu_get_device_props(HetGPUState *state, HetGPUDeviceProps *props)
{
    if (!state || !props) {
        return HETGPU_ERROR_INVALID_VALUE;
    }

    *props = state->props;
    return HETGPU_SUCCESS;
}

int hetgpu_cuda_device_get_attribute(HetGPUState *state, int attribute,
                                     int *value)
{
    int result;

    if (!value) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!state || !state->initialized ||
        state->backend == HETGPU_BACKEND_SIMULATION ||
        !g_cuda_mutex_initialized || !g_cuda_funcs.cuDeviceGetAttribute) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    qemu_mutex_lock(&g_cuda_mutex);
    result = HETGPU_CUDA_CALL(cuDeviceGetAttribute, value, attribute,
                                               state->cuda_device);
    qemu_mutex_unlock(&g_cuda_mutex);
    return result;
}

int hetgpu_cuda_device_total_memory(HetGPUState *state, size_t *bytes)
{
    int result;

    if (!bytes) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!state || !state->initialized ||
        state->backend == HETGPU_BACKEND_SIMULATION ||
        !g_cuda_mutex_initialized || !g_cuda_funcs.cuDeviceTotalMem) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    qemu_mutex_lock(&g_cuda_mutex);
    result = HETGPU_CUDA_CALL(cuDeviceTotalMem, bytes, state->cuda_device);
    qemu_mutex_unlock(&g_cuda_mutex);
    return result;
}

int hetgpu_cuda_get_error_name(HetGPUState *state, int error,
                               const char **name)
{
    int result;

    if (!name) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *name = NULL;
    if (!state || !state->initialized ||
        state->backend == HETGPU_BACKEND_SIMULATION ||
        !g_cuda_mutex_initialized || !g_cuda_funcs.cuGetErrorName) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    qemu_mutex_lock(&g_cuda_mutex);
    result = HETGPU_CUDA_CALL(cuGetErrorName, error, name);
    qemu_mutex_unlock(&g_cuda_mutex);
    return result;
}

int hetgpu_cuda_get_error_string(HetGPUState *state, int error,
                                 const char **string)
{
    int result;

    if (!string) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *string = NULL;
    if (!state || !state->initialized ||
        state->backend == HETGPU_BACKEND_SIMULATION ||
        !g_cuda_mutex_initialized || !g_cuda_funcs.cuGetErrorString) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    qemu_mutex_lock(&g_cuda_mutex);
    result = HETGPU_CUDA_CALL(cuGetErrorString, error, string);
    qemu_mutex_unlock(&g_cuda_mutex);
    return result;
}

int hetgpu_cuda_module_get_loading_mode(HetGPUState *state, int *mode)
{
    int result;

    if (!mode) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!state || !state->initialized ||
        state->backend == HETGPU_BACKEND_SIMULATION ||
        !g_cuda_mutex_initialized) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    if (!g_cuda_funcs.cuModuleGetLoadingMode) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    qemu_mutex_lock(&g_cuda_mutex);
    result = HETGPU_CUDA_CALL(cuModuleGetLoadingMode, mode);
    qemu_mutex_unlock(&g_cuda_mutex);
    return result;
}

int hetgpu_cuda_mem_get_info(HetGPUState *state, size_t *free_bytes,
                             size_t *total_bytes)
{
    int result;

    if (!free_bytes || !total_bytes) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!state || !state->initialized || !state->context ||
        state->backend == HETGPU_BACKEND_SIMULATION ||
        !g_cuda_mutex_initialized || !g_cuda_funcs.cuCtxSetCurrent ||
        !g_cuda_funcs.cuMemGetInfo) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }

    qemu_mutex_lock(&g_cuda_mutex);
    result = HETGPU_CUDA_CALL(cuCtxSetCurrent, state->context);
    if (result == CUDA_SUCCESS) {
        int driver_result = HETGPU_CUDA_CALL(cuMemGetInfo, free_bytes, total_bytes);

        qemu_log("CXL TYPE2 CUDA mem_info_driver context_activation_result=%d "
                 "driver_called=1 driver_result=%d free=%zu total=%zu\n",
                 result, driver_result, *free_bytes, *total_bytes);
        result = driver_result;
    } else {
        qemu_log("CXL TYPE2 CUDA mem_info_driver context_activation_result=%d "
                 "driver_called=0\n", result);
    }
    qemu_mutex_unlock(&g_cuda_mutex);
    return result;
}

HetGPUError hetgpu_create_context(HetGPUState *state)
{
    if (!state || !state->initialized) {
        return HETGPU_ERROR_NOT_INITIALIZED;
    }

    /* Context was already created during initialization */
    if (state->context != NULL) {
        qemu_log("CXL hetGPU: Using existing context %p\n", state->context);
        return HETGPU_SUCCESS;
    }

    if (state->backend == HETGPU_BACKEND_SIMULATION) {
        state->context = (void *)0xDEADBEEF;  /* Dummy context */
        return HETGPU_SUCCESS;
    }

    /* hetGPU managed mode: NVIDIA/AMD/Intel backend with NULL context
     * means hetGPU library manages the context internally */
    if (state->backend == HETGPU_BACKEND_NVIDIA ||
        state->backend == HETGPU_BACKEND_AMD ||
        state->backend == HETGPU_BACKEND_INTEL ||
        state->backend == HETGPU_BACKEND_TENSTORRENT) {
        qemu_log("CXL hetGPU: Using hetGPU managed context (backend=%d)\n", state->backend);
        /* Context remains NULL - hetGPU manages it internally */
        return HETGPU_SUCCESS;
    }

    if (g_cuda_funcs.cuCtxCreate) {
        void *ctx = NULL;
        int err = HETGPU_CUDA_CALL(cuCtxCreate, &ctx, 0, state->cuda_device);
        if (err == 0 && ctx != NULL) {
            state->context = ctx;
            qemu_log("CXL hetGPU: Created new CUDA context %p\n", ctx);
            return HETGPU_SUCCESS;
        }
        qemu_log("CXL hetGPU: cuCtxCreate failed with error %d\n", err);
        return HETGPU_ERROR_INVALID_CONTEXT;
    }

    return HETGPU_ERROR_NOT_INITIALIZED;
}

void hetgpu_destroy_context(HetGPUState *state)
{
    if (!state || !state->context) {
        return;
    }

    if (state->backend != HETGPU_BACKEND_SIMULATION && g_cuda_funcs.cuCtxDestroy) {
        HETGPU_CUDA_CALL(cuCtxDestroy, state->context);
    }

    state->context = NULL;
}

/*
 * Multi-device context management:
 * Uses cuCtxSetCurrent (replaces current) instead of cuCtxPushCurrent (grows stack).
 * Each operation must be wrapped in cuda_lock/cuda_unlock to prevent races
 * when two CXL Type 2 devices share the same physical GPU (or MIG instances).
 */
static bool cuda_lock(HetGPUState *state)
{
    if (state->backend == HETGPU_BACKEND_SIMULATION) {
        return true;
    }
    qemu_mutex_lock(&g_cuda_mutex);

    if (state->context) {
        /* Use cuCtxSetCurrent — does NOT grow the context stack */
        if (g_cuda_funcs.cuCtxSetCurrent) {
            int err = HETGPU_CUDA_CALL(cuCtxSetCurrent, state->context);
            if (err != 0) {
                if (state->formal_case_strict) {
                    qemu_log("CXL hetGPU: formal context activation failed "
                             "context=%p error=%d\n",
                             state->context, err);
                    qemu_mutex_unlock(&g_cuda_mutex);
                    return false;
                }
                qemu_log("CXL hetGPU: cuCtxSetCurrent(%p) failed (%d), "
                         "attempting context re-creation\n", state->context, err);
                if (g_cuda_funcs.cuCtxCreate) {
                    void *ctx = NULL;
                    err = HETGPU_CUDA_CALL(cuCtxCreate, &ctx, 0, state->cuda_device);
                    if (err == 0 && ctx != NULL) {
                        /* Pop the auto-pushed context from cuCtxCreate */
                        if (g_cuda_funcs.cuCtxPopCurrent) {
                            void *popped = NULL;
                            HETGPU_CUDA_CALL(cuCtxPopCurrent, &popped);
                        }
                        state->context = ctx;
                        HETGPU_CUDA_CALL(cuCtxSetCurrent, ctx);
                        qemu_log("CXL hetGPU: Re-created CUDA context %p\n", ctx);
                    }
                }
            }
        } else if (!state->formal_case_strict &&
                   g_cuda_funcs.cuCtxPushCurrent) {
            /* Fallback to push if SetCurrent not available */
            HETGPU_CUDA_CALL(cuCtxPushCurrent, state->context);
        } else {
            qemu_log("CXL hetGPU: formal context requires cuCtxSetCurrent\n");
            qemu_mutex_unlock(&g_cuda_mutex);
            return false;
        }
    } else if (state->formal_case_strict) {
        qemu_log("CXL hetGPU: formal operation has no active context\n");
        qemu_mutex_unlock(&g_cuda_mutex);
        return false;
    } else if (g_cuda_funcs.cuCtxCreate) {
        /* No context stored (hetGPU managed mode) - create one */
        void *ctx = NULL;
        int err = HETGPU_CUDA_CALL(cuCtxCreate, &ctx, 0, state->cuda_device);
        if (err == 0 && ctx != NULL) {
            /* Pop the auto-pushed context, use SetCurrent instead */
            if (g_cuda_funcs.cuCtxPopCurrent) {
                void *popped = NULL;
                HETGPU_CUDA_CALL(cuCtxPopCurrent, &popped);
            }
            state->context = ctx;
            if (g_cuda_funcs.cuCtxSetCurrent) {
                HETGPU_CUDA_CALL(cuCtxSetCurrent, ctx);
            }
            qemu_log("CXL hetGPU: Created CUDA context on demand: %p\n", ctx);
        }
    }
    return true;
}

static void cuda_unlock(HetGPUState *state)
{
    if (state->backend == HETGPU_BACKEND_SIMULATION) {
        return;
    }
    qemu_mutex_unlock(&g_cuda_mutex);
}


HetGPUError hetgpu_synchronize(HetGPUState *state)
{
    if (!state || !state->initialized) {
        return HETGPU_ERROR_NOT_INITIALIZED;
    }

    if (state->backend == HETGPU_BACKEND_SIMULATION) {
        return HETGPU_SUCCESS;
    }

    /* For hetGPU managed mode (NULL context), still call synchronize */
    if (!cuda_lock(state)) {
        return HETGPU_ERROR_INVALID_CONTEXT;
    }

    if (g_cuda_funcs.cuCtxSynchronize) {
        int err = HETGPU_CUDA_CALL(cuCtxSynchronize);
        cuda_unlock(state);
        if (err != 0) {
            const char *err_name = "UNKNOWN";
            if (g_cuda_funcs.cuGetErrorName) {
                HETGPU_CUDA_CALL(cuGetErrorName, err, &err_name);
            }
            qemu_log("CXL hetGPU: cuCtxSynchronize failed: %s (%d)\n", err_name, err);
            return HETGPU_ERROR_UNKNOWN;
        }
        return HETGPU_SUCCESS;
    }

    cuda_unlock(state);
    return HETGPU_SUCCESS;
}

HetGPUError hetgpu_malloc(HetGPUState *state, size_t size,
                          HetGPUMemFlags flags, HetGPUDevicePtr *dev_ptr)
{
    (void)flags;

    if (!state || !state->initialized || !dev_ptr) {
        return HETGPU_ERROR_INVALID_VALUE;
    }

    state->memory_ops++;

    /* For hetGPU managed mode (NULL context), still use real GPU through hetGPU library */
    if (g_cuda_funcs.cuMemAlloc && state->backend != HETGPU_BACKEND_SIMULATION) {
        if (!cuda_lock(state)) {
            return HETGPU_ERROR_INVALID_CONTEXT;
        }

        uint64_t ptr = 0;
        int err = HETGPU_CUDA_CALL(cuMemAlloc, &ptr, size);
        cuda_unlock(state);

        if (err == 0) {
            *dev_ptr = ptr;
            state->allocated_memory += size;
            qemu_log("CXL hetGPU: [dev%d] Allocated %zu bytes at device ptr 0x%lx\n",
                     state->device_index, size, (unsigned long)ptr);
            return HETGPU_SUCCESS;
        }
        const char *err_name = "UNKNOWN";
        if (g_cuda_funcs.cuGetErrorName) {
            HETGPU_CUDA_CALL(cuGetErrorName, err, &err_name);
        }
        qemu_log("CXL hetGPU: [dev%d] cuMemAlloc(%zu) failed: %s (%d)\n",
                 state->device_index, size, err_name, err);
        return HETGPU_ERROR_OUT_OF_MEMORY;
    }

    if (state->backend == HETGPU_BACKEND_SIMULATION) {
        /* Allocate real memory for simulation */
        HetGPUSimAlloc *alloc = g_new0(HetGPUSimAlloc, 1);
        if (!alloc) {
            return HETGPU_ERROR_OUT_OF_MEMORY;
        }
        alloc->host_buffer = g_malloc0(size);
        if (!alloc->host_buffer) {
            g_free(alloc);
            return HETGPU_ERROR_OUT_OF_MEMORY;
        }
        alloc->dev_ptr = state->sim_next_ptr;
        alloc->size = size;
        alloc->next = state->sim_allocs;
        state->sim_allocs = alloc;

        *dev_ptr = state->sim_next_ptr;
        state->sim_next_ptr += (size + 0xFFF) & ~0xFFF;  /* Page align */
        state->allocated_memory += size;

        qemu_log("CXL hetGPU: SIM allocated %zu bytes at 0x%lx -> host %p\n",
                 size, (unsigned long)*dev_ptr, alloc->host_buffer);
        return HETGPU_SUCCESS;
    }

    return HETGPU_ERROR_NOT_INITIALIZED;
}

HetGPUError hetgpu_free(HetGPUState *state, HetGPUDevicePtr dev_ptr)
{
    if (!state || !state->initialized) {
        return HETGPU_ERROR_INVALID_VALUE;
    }

    state->memory_ops++;

    /* For hetGPU managed mode, use real GPU through hetGPU library */
    if (g_cuda_funcs.cuMemFree && state->backend != HETGPU_BACKEND_SIMULATION) {
        if (!cuda_lock(state)) {
            return HETGPU_ERROR_INVALID_CONTEXT;
        }
        int err = HETGPU_CUDA_CALL(cuMemFree, dev_ptr);
        cuda_unlock(state);

        if (err != 0) {
            const char *err_name = "UNKNOWN";
            if (g_cuda_funcs.cuGetErrorName) {
                HETGPU_CUDA_CALL(cuGetErrorName, err, &err_name);
            }
            qemu_log("CXL hetGPU: [dev%d] cuMemFree failed: %s (%d)\n",
                     state->device_index, err_name, err);
            return HETGPU_ERROR_INVALID_VALUE;
        }
        return HETGPU_SUCCESS;
    }

    if (state->backend == HETGPU_BACKEND_SIMULATION) {
        /* Find and free the simulation allocation */
        HetGPUSimAlloc *prev = NULL;
        HetGPUSimAlloc *alloc = state->sim_allocs;
        while (alloc) {
            if (alloc->dev_ptr == dev_ptr) {
                if (prev) {
                    prev->next = alloc->next;
                } else {
                    state->sim_allocs = alloc->next;
                }
                qemu_log("CXL hetGPU: SIM freed 0x%lx\n", (unsigned long)dev_ptr);
                g_free(alloc->host_buffer);
                g_free(alloc);
                return HETGPU_SUCCESS;
            }
            prev = alloc;
            alloc = alloc->next;
        }
        return HETGPU_SUCCESS;  /* Not found, ignore */
    }

    return HETGPU_SUCCESS;
}

int hetgpu_pointer_get_memory_type(HetGPUState *state, HetGPUDevicePtr ptr,
                                   int *memory_type)
{
    if (!state || !state->initialized || !memory_type) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    if (g_cuda_funcs.cuPointerGetAttribute &&
        state->backend != HETGPU_BACKEND_SIMULATION) {
        if (!cuda_lock(state)) {
            return CUDA_ERROR_INVALID_CONTEXT;
        }
        int result = HETGPU_CUDA_CALL(cuPointerGetAttribute, memory_type,
                                     2, ptr);
        cuda_unlock(state);
        return result;
    }

    if (state->backend == HETGPU_BACKEND_SIMULATION) {
        if (find_sim_alloc(state, ptr)) {
            *memory_type = 2;
            return CUDA_SUCCESS;
        }
        return CUDA_ERROR_INVALID_VALUE;
    }

    return CUDA_ERROR_NOT_SUPPORTED;
}

HetGPUError hetgpu_memcpy_htod(HetGPUState *state, HetGPUDevicePtr dst,
                               const void *src, size_t size)
{
    if (!state || !state->initialized || !src) {
        return HETGPU_ERROR_INVALID_VALUE;
    }

    state->memory_ops++;

    /* For hetGPU managed mode, use real GPU through hetGPU library */
    if (g_cuda_funcs.cuMemcpyHtoD && state->backend != HETGPU_BACKEND_SIMULATION) {
        if (!cuda_lock(state)) {
            return HETGPU_ERROR_INVALID_CONTEXT;
        }
        int err = HETGPU_CUDA_CALL(cuMemcpyHtoD, dst, src, size);
        cuda_unlock(state);

        if (err != 0) {
            const char *err_name = "UNKNOWN";
            if (g_cuda_funcs.cuGetErrorName) {
                HETGPU_CUDA_CALL(cuGetErrorName, err, &err_name);
            }
            qemu_log("CXL hetGPU: [dev%d] cuMemcpyHtoD failed: %s (%d)\n",
                     state->device_index, err_name, err);
            return HETGPU_ERROR_INVALID_VALUE;
        }
        return HETGPU_SUCCESS;
    }

    if (state->backend == HETGPU_BACKEND_SIMULATION) {
        /* Find the allocation and copy data */
        HetGPUSimAlloc *alloc = find_sim_alloc(state, dst);
        if (alloc) {
            size_t offset = dst - alloc->dev_ptr;
            if (offset + size <= alloc->size) {
                memcpy((uint8_t *)alloc->host_buffer + offset, src, size);
                qemu_log("CXL hetGPU: SIM memcpy HtoD 0x%lx <- %zu bytes\n",
                         (unsigned long)dst, size);
                return HETGPU_SUCCESS;
            }
        }
        qemu_log("CXL hetGPU: SIM memcpy HtoD failed - allocation not found for 0x%lx\n",
                 (unsigned long)dst);
        return HETGPU_ERROR_INVALID_VALUE;
    }

    return HETGPU_SUCCESS;
}

int hetgpu_cuda_mem_host_alloc(HetGPUState *state, void **ptr, size_t size)
{
    if (!state || !state->initialized || !ptr || !size ||
        !g_cuda_funcs.cuMemHostAlloc) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!cuda_lock(state)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    int result = HETGPU_CUDA_CALL(cuMemHostAlloc, ptr, size, 0);
    cuda_unlock(state);
    return result;
}

int hetgpu_cuda_mem_free_host(HetGPUState *state, void *ptr)
{
    if (!ptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!cuda_lock(state)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    int result = HETGPU_CUDA_CALL(cuMemFreeHost, ptr);
    cuda_unlock(state);
    return result;
}

int hetgpu_cuda_memcpy_htod_async(HetGPUState *state, HetGPUDevicePtr dst,
                                  const void *src, size_t size,
                                  HetGPUStream stream)
{
    if (!state || !state->initialized || !src ||
        !g_cuda_funcs.cuMemcpyHtoDAsync) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!cuda_lock(state)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    int result = HETGPU_CUDA_CALL(cuMemcpyHtoDAsync, dst, src, size, stream);
    cuda_unlock(state);
    return result;
}

HetGPUError hetgpu_memcpy_dtoh(HetGPUState *state, void *dst,
                               HetGPUDevicePtr src, size_t size)
{
    if (!state || !state->initialized || !dst) {
        return HETGPU_ERROR_INVALID_VALUE;
    }

    state->memory_ops++;

    /* For hetGPU managed mode, use real GPU through hetGPU library */
    if (g_cuda_funcs.cuMemcpyDtoH && state->backend != HETGPU_BACKEND_SIMULATION) {
        if (!cuda_lock(state)) {
            return HETGPU_ERROR_INVALID_CONTEXT;
        }
        int err = HETGPU_CUDA_CALL(cuMemcpyDtoH, dst, src, size);
        cuda_unlock(state);

        if (err != 0) {
            const char *err_name = "UNKNOWN";
            if (g_cuda_funcs.cuGetErrorName) {
                HETGPU_CUDA_CALL(cuGetErrorName, err, &err_name);
            }
            qemu_log("CXL hetGPU: [dev%d] cuMemcpyDtoH failed: %s (%d)\n",
                     state->device_index, err_name, err);
            return HETGPU_ERROR_INVALID_VALUE;
        }
        return HETGPU_SUCCESS;
    }

    if (state->backend == HETGPU_BACKEND_SIMULATION) {
        /* Find the allocation and copy data */
        HetGPUSimAlloc *alloc = find_sim_alloc(state, src);
        if (alloc) {
            size_t offset = src - alloc->dev_ptr;
            if (offset + size <= alloc->size) {
                memcpy(dst, (uint8_t *)alloc->host_buffer + offset, size);
                qemu_log("CXL hetGPU: SIM memcpy DtoH 0x%lx -> %zu bytes\n",
                         (unsigned long)src, size);
                return HETGPU_SUCCESS;
            }
        }
        qemu_log("CXL hetGPU: SIM memcpy DtoH failed - allocation not found for 0x%lx\n",
                 (unsigned long)src);
        memset(dst, 0, size);  /* Return zeros on error */
        return HETGPU_ERROR_INVALID_VALUE;
    }

    return HETGPU_SUCCESS;
}

HetGPUError hetgpu_memcpy_dtod(HetGPUState *state, HetGPUDevicePtr dst,
                               HetGPUDevicePtr src, size_t size)
{
    if (!state || !state->initialized) {
        return HETGPU_ERROR_INVALID_VALUE;
    }

    state->memory_ops++;

    /* For hetGPU managed mode, use real GPU through hetGPU library */
    if (g_cuda_funcs.cuMemcpyDtoD && state->backend != HETGPU_BACKEND_SIMULATION) {
        if (!cuda_lock(state)) {
            return HETGPU_ERROR_INVALID_CONTEXT;
        }
        int err = HETGPU_CUDA_CALL(cuMemcpyDtoD, dst, src, size);
        cuda_unlock(state);

        if (err != 0) {
            const char *err_name = "UNKNOWN";
            if (g_cuda_funcs.cuGetErrorName) {
                HETGPU_CUDA_CALL(cuGetErrorName, err, &err_name);
            }
            qemu_log("CXL hetGPU: [dev%d] cuMemcpyDtoD failed: %s (%d)\n",
                     state->device_index, err_name, err);
            return HETGPU_ERROR_INVALID_VALUE;
        }
        return HETGPU_SUCCESS;
    }

    if (state->backend == HETGPU_BACKEND_SIMULATION) {
        /* Find both allocations and copy between them */
        HetGPUSimAlloc *src_alloc = find_sim_alloc(state, src);
        HetGPUSimAlloc *dst_alloc = find_sim_alloc(state, dst);
        if (src_alloc && dst_alloc) {
            size_t src_offset = src - src_alloc->dev_ptr;
            size_t dst_offset = dst - dst_alloc->dev_ptr;
            if (src_offset + size <= src_alloc->size &&
                dst_offset + size <= dst_alloc->size) {
                memcpy((uint8_t *)dst_alloc->host_buffer + dst_offset,
                       (uint8_t *)src_alloc->host_buffer + src_offset, size);
                qemu_log("CXL hetGPU: SIM memcpy DtoD 0x%lx -> 0x%lx (%zu bytes)\n",
                         (unsigned long)src, (unsigned long)dst, size);
                return HETGPU_SUCCESS;
            }
        }
        qemu_log("CXL hetGPU: SIM memcpy DtoD failed - allocation not found\n");
        return HETGPU_ERROR_INVALID_VALUE;
    }

    return HETGPU_SUCCESS;
}

HetGPUError hetgpu_memset(HetGPUState *state, HetGPUDevicePtr dev_ptr,
                          int value, size_t size)
{
    if (!state || !state->initialized) {
        return HETGPU_ERROR_INVALID_VALUE;
    }

    state->memory_ops++;

    if (state->backend == HETGPU_BACKEND_SIMULATION) {
        /* Find the allocation and set memory */
        HetGPUSimAlloc *alloc = find_sim_alloc(state, dev_ptr);
        if (alloc) {
            size_t offset = dev_ptr - alloc->dev_ptr;
            if (offset + size <= alloc->size) {
                memset((uint8_t *)alloc->host_buffer + offset, value, size);
                return HETGPU_SUCCESS;
            }
        }
        return HETGPU_ERROR_INVALID_VALUE;
    }

    return HETGPU_SUCCESS;
}

HetGPUError hetgpu_create_coherent_region(HetGPUState *state, size_t size,
                                          HetGPUCoherentRegion *region)
{
    if (!state || !state->initialized || !region) {
        return HETGPU_ERROR_INVALID_VALUE;
    }

    state->coherency_ops++;

    /* Allocate host-mapped coherent memory */
    region->host_ptr = g_malloc0(size);
    if (!region->host_ptr) {
        return HETGPU_ERROR_OUT_OF_MEMORY;
    }

    region->size = size;
    region->flags = HETGPU_MEM_HOST_MAPPED;
    region->is_coherent = true;

    /* Get device pointer */
    HetGPUError err = hetgpu_malloc(state, size, HETGPU_MEM_HOST_MAPPED,
                                    &region->device_ptr);
    if (err != HETGPU_SUCCESS) {
        g_free(region->host_ptr);
        return err;
    }

    return HETGPU_SUCCESS;
}

void hetgpu_destroy_coherent_region(HetGPUState *state,
                                    HetGPUCoherentRegion *region)
{
    if (!state || !region) {
        return;
    }

    if (region->device_ptr) {
        hetgpu_free(state, region->device_ptr);
    }
    if (region->host_ptr) {
        g_free(region->host_ptr);
    }

    memset(region, 0, sizeof(*region));
}

void hetgpu_set_coherency_callback(HetGPUState *state,
                                   HetGPUCoherencyCallback callback,
                                   void *opaque)
{
    if (!state) {
        return;
    }

    state->coherency_callback = callback;
    state->cxl_opaque = opaque;
}

HetGPUError hetgpu_flush_cache(HetGPUState *state, HetGPUDevicePtr dev_ptr,
                               size_t size)
{
    (void)dev_ptr;
    (void)size;

    if (!state || !state->initialized) {
        return HETGPU_ERROR_INVALID_VALUE;
    }

    state->coherency_ops++;
    return HETGPU_SUCCESS;
}

HetGPUError hetgpu_invalidate_cache(HetGPUState *state, HetGPUDevicePtr dev_ptr,
                                    size_t size)
{
    (void)dev_ptr;
    (void)size;

    if (!state || !state->initialized) {
        return HETGPU_ERROR_INVALID_VALUE;
    }

    state->coherency_ops++;
    return HETGPU_SUCCESS;
}

HetGPUError hetgpu_load_ptx(HetGPUState *state, const char *ptx_source,
                            HetGPUModule *module)
{
    if (!state || !state->initialized || !ptx_source || !module) {
        return HETGPU_ERROR_INVALID_VALUE;
    }

    /* For hetGPU managed mode, use real GPU through hetGPU library */
    if (g_cuda_funcs.cuModuleLoadData && state->backend != HETGPU_BACKEND_SIMULATION) {
        if (!cuda_lock(state)) {
            return HETGPU_ERROR_INVALID_CONTEXT;
        }
        void *mod = NULL;
        int err = HETGPU_CUDA_CALL(cuModuleLoadData, &mod, ptx_source);
        cuda_unlock(state);

        if (err == 0) {
            *module = mod;
            qemu_log("CXL hetGPU: [dev%d] Loaded PTX module %p\n",
                     state->device_index, mod);
            return HETGPU_SUCCESS;
        }
        const char *err_name = "UNKNOWN";
        if (g_cuda_funcs.cuGetErrorName) {
            HETGPU_CUDA_CALL(cuGetErrorName, err, &err_name);
        }
        qemu_log("CXL hetGPU: [dev%d] cuModuleLoadData failed: %s (%d)\n",
                 state->device_index, err_name, err);
        return HETGPU_ERROR_INVALID_PTX;
    }

    if (state->backend == HETGPU_BACKEND_SIMULATION) {
        *module = (void *)0x12345678;  /* Dummy module handle */
        return HETGPU_SUCCESS;
    }

    return HETGPU_ERROR_NOT_INITIALIZED;
}

HetGPUError hetgpu_load_cubin(HetGPUState *state, const void *cubin_data,
                              size_t cubin_size, HetGPUModule *module)
{
    (void)cubin_size;

    if (!state || !state->initialized || !cubin_data || !module) {
        return HETGPU_ERROR_INVALID_VALUE;
    }

    /* For hetGPU managed mode, use real GPU through hetGPU library */
    if (g_cuda_funcs.cuModuleLoadData && state->backend != HETGPU_BACKEND_SIMULATION) {
        if (!cuda_lock(state)) {
            return HETGPU_ERROR_INVALID_CONTEXT;
        }
        void *mod = NULL;
        int err = HETGPU_CUDA_CALL(cuModuleLoadData, &mod, cubin_data);
        cuda_unlock(state);

        if (err == 0) {
            *module = mod;
            qemu_log("CXL hetGPU: [dev%d] Loaded CUBIN module %p\n",
                     state->device_index, mod);
            return HETGPU_SUCCESS;
        }
        const char *err_name = "UNKNOWN";
        if (g_cuda_funcs.cuGetErrorName) {
            HETGPU_CUDA_CALL(cuGetErrorName, err, &err_name);
        }
        qemu_log("CXL hetGPU: [dev%d] cuModuleLoadData (cubin) failed: %s (%d)\n",
                 state->device_index, err_name, err);
        return HETGPU_ERROR_INVALID_PTX;
    }

    if (state->backend == HETGPU_BACKEND_SIMULATION) {
        *module = (void *)0x12345678;
        return HETGPU_SUCCESS;
    }

    return HETGPU_ERROR_NOT_INITIALIZED;
}

HetGPUError hetgpu_unload_module(HetGPUState *state, HetGPUModule module)
{
    if (!state || !state->initialized || !module) {
        return HETGPU_ERROR_INVALID_VALUE;
    }

    if (g_cuda_funcs.cuModuleUnload &&
        state->backend != HETGPU_BACKEND_SIMULATION) {
        if (!cuda_lock(state)) {
            return HETGPU_ERROR_INVALID_CONTEXT;
        }
        int err = HETGPU_CUDA_CALL(cuModuleUnload, module);
        cuda_unlock(state);

        switch (err) {
        case CUDA_SUCCESS:
            return HETGPU_SUCCESS;
        case CUDA_ERROR_INVALID_VALUE:
            return HETGPU_ERROR_INVALID_VALUE;
        case CUDA_ERROR_NOT_INITIALIZED:
            return HETGPU_ERROR_NOT_INITIALIZED;
        case CUDA_ERROR_INVALID_CONTEXT:
            return HETGPU_ERROR_INVALID_CONTEXT;
        case CUDA_ERROR_INVALID_HANDLE:
            return HETGPU_ERROR_INVALID_HANDLE;
        case CUDA_ERROR_NOT_SUPPORTED:
            return HETGPU_ERROR_NOT_SUPPORTED;
        default:
            return HETGPU_ERROR_UNKNOWN;
        }
    }

    if (state->backend == HETGPU_BACKEND_SIMULATION) {
        return HETGPU_SUCCESS;
    }
    return HETGPU_ERROR_NOT_SUPPORTED;
}

HetGPUError hetgpu_get_function(HetGPUState *state, HetGPUModule module,
                                const char *name, HetGPUFunction *function)
{
    if (!state || !state->initialized || !module || !name || !function) {
        return HETGPU_ERROR_INVALID_VALUE;
    }

    /* For hetGPU managed mode, use real GPU through hetGPU library */
    if (g_cuda_funcs.cuModuleGetFunction && state->backend != HETGPU_BACKEND_SIMULATION) {
        if (!cuda_lock(state)) {
            return HETGPU_ERROR_INVALID_CONTEXT;
        }
        void *func = NULL;
        int err = HETGPU_CUDA_CALL(cuModuleGetFunction, &func, module, name);
        cuda_unlock(state);

        if (err == 0) {
            *function = func;
            qemu_log("CXL hetGPU: [dev%d] Got function '%s' -> %p\n",
                     state->device_index, name, func);
            return HETGPU_SUCCESS;
        }
        const char *err_name = "UNKNOWN";
        if (g_cuda_funcs.cuGetErrorName) {
            HETGPU_CUDA_CALL(cuGetErrorName, err, &err_name);
        }
        qemu_log("CXL hetGPU: [dev%d] cuModuleGetFunction('%s') failed: %s (%d)\n",
                 state->device_index, name, err_name, err);
        return HETGPU_ERROR_UNKNOWN;
    }

    if (state->backend == HETGPU_BACKEND_SIMULATION) {
        *function = (void *)0x87654321;  /* Dummy function handle */
        return HETGPU_SUCCESS;
    }

    return HETGPU_ERROR_NOT_INITIALIZED;
}

HetGPUError hetgpu_get_global(HetGPUState *state, HetGPUModule module,
                              const char *name, HetGPUDevicePtr *dev_ptr,
                              size_t *size)
{
    if (!state || !state->initialized || !module || !name || !dev_ptr || !size) {
        return HETGPU_ERROR_INVALID_VALUE;
    }

    if (g_cuda_funcs.cuModuleGetGlobal && state->backend != HETGPU_BACKEND_SIMULATION) {
        if (!cuda_lock(state)) {
            return HETGPU_ERROR_INVALID_CONTEXT;
        }
        uint64_t ptr = 0;
        size_t bytes = 0;
        int err = HETGPU_CUDA_CALL(cuModuleGetGlobal, &ptr, &bytes, module, name);
        cuda_unlock(state);

        if (err == 0) {
            *dev_ptr = ptr;
            *size = bytes;
            qemu_log("CXL hetGPU: [dev%d] Got global '%s' -> 0x%" PRIx64 " size=%zu\n",
                     state->device_index, name, ptr, bytes);
            return HETGPU_SUCCESS;
        }

        const char *err_name = "UNKNOWN";
        if (g_cuda_funcs.cuGetErrorName) {
            HETGPU_CUDA_CALL(cuGetErrorName, err, &err_name);
        }
        qemu_log("CXL hetGPU: [dev%d] cuModuleGetGlobal('%s') failed: %s (%d)\n",
                 state->device_index, name, err_name, err);
        switch (err) {
        case 1:
            return HETGPU_ERROR_INVALID_VALUE;
        case 201:
            return HETGPU_ERROR_INVALID_CONTEXT;
        case 400:
            return HETGPU_ERROR_INVALID_HANDLE;
        case 500:
            return HETGPU_ERROR_NOT_FOUND;
        case 801:
            return HETGPU_ERROR_NOT_SUPPORTED;
        default:
            return HETGPU_ERROR_UNKNOWN;
        }
    }

    return HETGPU_ERROR_NOT_SUPPORTED;
}

HetGPUError hetgpu_get_param_info(HetGPUState *state, HetGPUFunction function,
                                  size_t param_index, size_t *param_offset,
                                  size_t *param_size)
{
    if (!state || !state->initialized || !function || !param_offset ||
        !param_size) {
        return HETGPU_ERROR_INVALID_VALUE;
    }

    if (g_cuda_funcs.cuFuncGetParamInfo &&
        state->backend != HETGPU_BACKEND_SIMULATION) {
        if (!cuda_lock(state)) {
            return HETGPU_ERROR_INVALID_CONTEXT;
        }
        int err = HETGPU_CUDA_CALL(cuFuncGetParamInfo, function, param_index,
                                                  param_offset, param_size);
        cuda_unlock(state);

        switch (err) {
        case 0:
            return HETGPU_SUCCESS;
        case 1:
            return HETGPU_ERROR_INVALID_VALUE;
        case 400:
            return HETGPU_ERROR_INVALID_HANDLE;
        case 801:
            return HETGPU_ERROR_NOT_SUPPORTED;
        default:
            return HETGPU_ERROR_UNKNOWN;
        }
    }

    return HETGPU_ERROR_NOT_SUPPORTED;
}

HetGPUError hetgpu_set_function_attribute(HetGPUState *state,
                                          HetGPUFunction function,
                                          int attribute, int value)
{
    if (!state || !state->initialized || !function) {
        return HETGPU_ERROR_INVALID_VALUE;
    }

    if (g_cuda_funcs.cuFuncSetAttribute &&
        state->backend != HETGPU_BACKEND_SIMULATION) {
        if (!cuda_lock(state)) {
            return HETGPU_ERROR_INVALID_CONTEXT;
        }
        int err = HETGPU_CUDA_CALL(cuFuncSetAttribute, function, attribute, value);
        cuda_unlock(state);

        switch (err) {
        case 0:
            return HETGPU_SUCCESS;
        case 1:
            return HETGPU_ERROR_INVALID_VALUE;
        case 201:
            return HETGPU_ERROR_INVALID_CONTEXT;
        case 400:
            return HETGPU_ERROR_INVALID_HANDLE;
        case 801:
            return HETGPU_ERROR_NOT_SUPPORTED;
        default:
            return HETGPU_ERROR_UNKNOWN;
        }
    }

    return HETGPU_ERROR_NOT_SUPPORTED;
}

HetGPUError hetgpu_get_function_attribute(HetGPUState *state,
                                          HetGPUFunction function,
                                          int attribute, int *value)
{
    if (!state || !state->initialized || !function || !value) {
        return HETGPU_ERROR_INVALID_VALUE;
    }

    if (g_cuda_funcs.cuFuncGetAttribute &&
        state->backend != HETGPU_BACKEND_SIMULATION) {
        if (!cuda_lock(state)) {
            return HETGPU_ERROR_INVALID_CONTEXT;
        }
        int err = HETGPU_CUDA_CALL(cuFuncGetAttribute, value, attribute,
                                   function);
        cuda_unlock(state);

        switch (err) {
        case CUDA_SUCCESS:
            return HETGPU_SUCCESS;
        case CUDA_ERROR_INVALID_VALUE:
            return HETGPU_ERROR_INVALID_VALUE;
        case CUDA_ERROR_NOT_INITIALIZED:
            return HETGPU_ERROR_NOT_INITIALIZED;
        case CUDA_ERROR_INVALID_CONTEXT:
            return HETGPU_ERROR_INVALID_CONTEXT;
        case CUDA_ERROR_INVALID_HANDLE:
            return HETGPU_ERROR_INVALID_HANDLE;
        case CUDA_ERROR_NOT_SUPPORTED:
            return HETGPU_ERROR_NOT_SUPPORTED;
        default:
            return HETGPU_ERROR_UNKNOWN;
        }
    }

    return HETGPU_ERROR_NOT_SUPPORTED;
}

HetGPUError hetgpu_get_max_active_blocks_per_multiprocessor(HetGPUState *state,
                                                             HetGPUFunction function,
                                                             int block_size,
                                                             size_t dynamic_smem_size,
                                                             unsigned int flags,
                                                             int *num_blocks)
{
    if (!state || !state->initialized || !function || !num_blocks) {
        return HETGPU_ERROR_INVALID_VALUE;
    }
    *num_blocks = 0;

    if (g_cuda_funcs.cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags &&
        state->backend != HETGPU_BACKEND_SIMULATION) {
        if (!cuda_lock(state)) {
            return HETGPU_ERROR_INVALID_CONTEXT;
        }
        int err = HETGPU_CUDA_CALL(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags,
            num_blocks, function, block_size, dynamic_smem_size, flags);
        cuda_unlock(state);

        qemu_log("CXL hetGPU: occupancy function=%p block_size=%d dynamic_smem_size=%zu "
                 "flags=%u driver_result=%d num_blocks=%d\n",
                 function, block_size, dynamic_smem_size, flags, err, *num_blocks);

        switch (err) {
        case 0:
            return HETGPU_SUCCESS;
        case 1:
            return HETGPU_ERROR_INVALID_VALUE;
        case 3:
            return HETGPU_ERROR_NOT_INITIALIZED;
        case 201:
            return HETGPU_ERROR_INVALID_CONTEXT;
        case 400:
            return HETGPU_ERROR_INVALID_HANDLE;
        case 801:
            return HETGPU_ERROR_NOT_SUPPORTED;
        default:
            return HETGPU_ERROR_UNKNOWN;
        }
    }

    return HETGPU_ERROR_NOT_SUPPORTED;
}

HetGPUError hetgpu_launch_kernel(HetGPUState *state, HetGPUFunction function,
                                 const HetGPULaunchConfig *config,
                                 void **args, size_t num_args)
{
    (void)num_args;

    if (!state || !state->initialized || !function || !config) {
        return HETGPU_ERROR_INVALID_VALUE;
    }

    state->kernel_launches++;

    /* For hetGPU managed mode, use real GPU through hetGPU library */
    if (g_cuda_funcs.cuLaunchKernel && state->backend != HETGPU_BACKEND_SIMULATION) {
        if (!cuda_lock(state)) {
            return HETGPU_ERROR_INVALID_CONTEXT;
        }

        qemu_log("CXL hetGPU: [dev%d] Launching kernel grid=(%u,%u,%u) block=(%u,%u,%u) shared=%u\n",
                 state->device_index,
                 config->grid_dim[0], config->grid_dim[1], config->grid_dim[2],
                 config->block_dim[0], config->block_dim[1], config->block_dim[2],
                 config->shared_mem_bytes);

        int err = HETGPU_CUDA_CALL(cuLaunchKernel,
            function,
            config->grid_dim[0], config->grid_dim[1], config->grid_dim[2],
            config->block_dim[0], config->block_dim[1], config->block_dim[2],
            config->shared_mem_bytes, config->stream,
            args, NULL);
        cuda_unlock(state);

        if (err == 0) {
            return HETGPU_SUCCESS;
        }
        const char *err_name = "UNKNOWN";
        if (g_cuda_funcs.cuGetErrorName) {
            HETGPU_CUDA_CALL(cuGetErrorName, err, &err_name);
        }
        qemu_log("CXL hetGPU: [dev%d] cuLaunchKernel failed: %s (%d)\n",
                 state->device_index, err_name, err);
        return HETGPU_ERROR_LAUNCH_FAILED;
    }

    if (state->backend == HETGPU_BACKEND_SIMULATION) {
        qemu_log("CXL hetGPU: Simulated kernel launch grid=(%u,%u,%u) block=(%u,%u,%u)\n",
                 config->grid_dim[0], config->grid_dim[1], config->grid_dim[2],
                 config->block_dim[0], config->block_dim[1], config->block_dim[2]);
        return HETGPU_SUCCESS;
    }

    return HETGPU_SUCCESS;
}

int hetgpu_cuda_graph_exec_kernel_node_set_params(
    HetGPUState *state, HetGPUGraphExec graph_exec, HetGPUGraphNode graph_node,
    HetGPUFunction function, const HetGPULaunchConfig *config, void **args)
{
    int result;

    if (!state || !state->initialized) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    if (!graph_exec || !graph_node || !function || !config) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (state->backend == HETGPU_BACKEND_SIMULATION ||
        !g_cuda_funcs.cuGraphExecKernelNodeSetParams) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    CudaKernelNodeParams params = {
        .func = function,
        .grid_dim_x = config->grid_dim[0],
        .grid_dim_y = config->grid_dim[1],
        .grid_dim_z = config->grid_dim[2],
        .block_dim_x = config->block_dim[0],
        .block_dim_y = config->block_dim[1],
        .block_dim_z = config->block_dim[2],
        .shared_mem_bytes = config->shared_mem_bytes,
        .kernel_params = args,
        .extra = NULL,
        .kernel = NULL,
        .context = NULL,
    };
    if (!cuda_lock(state)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    result = HETGPU_CUDA_CALL(cuGraphExecKernelNodeSetParams, graph_exec, graph_node, &params);
    cuda_unlock(state);
    return result;
}

int hetgpu_cuda_graph_kernel_node_get_params(HetGPUState *state, HetGPUGraphNode graph_node,
                                             CudaKernelNodeParams *params)
{
    int result;

    if (!state || !state->initialized) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    if (!graph_node || !params) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (state->backend == HETGPU_BACKEND_SIMULATION ||
        !g_cuda_funcs.cuGraphKernelNodeGetParams) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (!cuda_lock(state)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    result = HETGPU_CUDA_CALL(cuGraphKernelNodeGetParams, graph_node, params);
    cuda_unlock(state);
    return result;
}

int hetgpu_cuda_graph_exec_destroy(HetGPUState *state,
                                   HetGPUGraphExec graph_exec)
{
    int result;

    if (!state || !state->initialized) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    if (!graph_exec) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (state->backend == HETGPU_BACKEND_SIMULATION ||
        !g_cuda_funcs.cuGraphExecDestroy) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (!cuda_lock(state)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    result = HETGPU_CUDA_CALL(cuGraphExecDestroy, graph_exec);
    cuda_unlock(state);
    return result;
}

int hetgpu_cuda_graph_launch(HetGPUState *state, HetGPUGraphExec graph_exec,
                             HetGPUStream stream)
{
    int result;

    if (!state || !state->initialized) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    if (!graph_exec) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (state->backend == HETGPU_BACKEND_SIMULATION ||
        !g_cuda_funcs.cuGraphLaunch) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (!cuda_lock(state)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    result = HETGPU_CUDA_CALL(cuGraphLaunch, graph_exec, stream);
    cuda_unlock(state);
    return result;
}

int hetgpu_cuda_graph_destroy(HetGPUState *state, HetGPUGraph graph)
{
    int result;

    if (!state || !state->initialized) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    if (!graph) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (state->backend == HETGPU_BACKEND_SIMULATION ||
        !g_cuda_funcs.cuGraphDestroy) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (!cuda_lock(state)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    result = HETGPU_CUDA_CALL(cuGraphDestroy, graph);
    cuda_unlock(state);
    return result;
}

int hetgpu_cuda_graph_instantiate(HetGPUState *state, HetGPUGraph graph,
                                  HetGPUGraphExec *graph_exec,
                                  HetGPUGraphNode *error_node,
                                  char *log_buffer, size_t buffer_size)
{
    int result;

    if (!state || !state->initialized) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    if (!graph || !graph_exec) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (state->backend == HETGPU_BACKEND_SIMULATION ||
        !g_cuda_funcs.cuGraphInstantiate) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (!cuda_lock(state)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    result = HETGPU_CUDA_CALL(cuGraphInstantiate, graph_exec, graph,
                              error_node, log_buffer, buffer_size);
    cuda_unlock(state);
    return result;
}

int hetgpu_cuda_graph_get_nodes(HetGPUState *state, HetGPUGraph graph,
                                HetGPUGraphNode *nodes, size_t *num_nodes)
{
    int result;

    if (!state || !state->initialized) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    if (!graph || !num_nodes) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (state->backend == HETGPU_BACKEND_SIMULATION ||
        !g_cuda_funcs.cuGraphGetNodes) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (!cuda_lock(state)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    result = HETGPU_CUDA_CALL(cuGraphGetNodes, graph, nodes, num_nodes);
    cuda_unlock(state);
    return result;
}

int hetgpu_cuda_graph_node_get_type(HetGPUState *state,
                                    HetGPUGraphNode graph_node,
                                    int *node_type)
{
    int result;

    if (!state || !state->initialized) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    if (!graph_node || !node_type) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (state->backend == HETGPU_BACKEND_SIMULATION ||
        !g_cuda_funcs.cuGraphNodeGetType) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (!cuda_lock(state)) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    result = HETGPU_CUDA_CALL(cuGraphNodeGetType, graph_node, node_type);
    cuda_unlock(state);
    return result;
}

static bool hetgpu_cuda_ready(HetGPUState *state)
{
    return state && state->initialized && state->backend != HETGPU_BACKEND_SIMULATION;
}

#define HETGPU_FORWARD_BODY(field, ...)                                        \
    do {                                                                        \
        if (!hetgpu_cuda_ready(state))                                          \
            return CUDA_ERROR_INVALID_CONTEXT;                                  \
        if (!g_cuda_funcs.field)                                                \
            return CUDA_ERROR_NOT_SUPPORTED;                                    \
        if (!cuda_lock(state))                                                  \
            return CUDA_ERROR_INVALID_CONTEXT;                                  \
        int result = HETGPU_CUDA_CALL(field, __VA_ARGS__);                       \
        cuda_unlock(state);                                                     \
        return result;                                                          \
    } while (0)

int hetgpu_cuda_link_create(HetGPUState *state, void **link_state)
{
    if (!link_state)
        return CUDA_ERROR_INVALID_VALUE;
    HETGPU_FORWARD_BODY(cuLinkCreate, 0, NULL, NULL, link_state);
}

int hetgpu_cuda_link_add_data(HetGPUState *state, void *link_state, int input_type,
                              void *data, size_t size, const char *name)
{
    if (!link_state || !data || !size)
        return CUDA_ERROR_INVALID_VALUE;
    HETGPU_FORWARD_BODY(cuLinkAddData, link_state, input_type, data, size, name,
                        0, NULL, NULL);
}

int hetgpu_cuda_link_complete(HetGPUState *state, void *link_state,
                              void **cubin, size_t *size)
{
    if (!link_state || !cubin || !size)
        return CUDA_ERROR_INVALID_VALUE;
    HETGPU_FORWARD_BODY(cuLinkComplete, link_state, cubin, size);
}

int hetgpu_cuda_link_destroy(HetGPUState *state, void *link_state)
{
    if (!link_state)
        return CUDA_ERROR_INVALID_VALUE;
    HETGPU_FORWARD_BODY(cuLinkDestroy, link_state);
}

int hetgpu_cuda_ctx_get_limit(HetGPUState *state, size_t *value, int limit)
{
    if (!value)
        return CUDA_ERROR_INVALID_VALUE;
    HETGPU_FORWARD_BODY(cuCtxGetLimit, value, limit);
}

int hetgpu_cuda_stream_create(HetGPUState *state, unsigned int flags,
                              HetGPUStream *stream)
{
    if (!stream)
        return CUDA_ERROR_INVALID_VALUE;
    HETGPU_FORWARD_BODY(cuStreamCreate, stream, flags);
}

int hetgpu_cuda_stream_destroy(HetGPUState *state, HetGPUStream stream)
{
    if (!stream)
        return CUDA_ERROR_INVALID_VALUE;
    HETGPU_FORWARD_BODY(cuStreamDestroy, stream);
}

int hetgpu_cuda_stream_synchronize(HetGPUState *state, HetGPUStream stream)
{
    HETGPU_FORWARD_BODY(cuStreamSynchronize, stream);
}

int hetgpu_cuda_stream_wait_event(HetGPUState *state, HetGPUStream stream,
                                  void *event, unsigned int flags)
{
    if (!event)
        return CUDA_ERROR_INVALID_VALUE;
    HETGPU_FORWARD_BODY(cuStreamWaitEvent, stream, event, flags);
}

int hetgpu_cuda_stream_wait_value32(HetGPUState *state, HetGPUStream stream,
                                    uint64_t address, uint32_t value,
                                    unsigned int flags)
{
    HETGPU_FORWARD_BODY(cuStreamWaitValue32, stream, address, value, flags);
}

int hetgpu_cuda_stream_batch_mem_op(HetGPUState *state, HetGPUStream stream,
                                    unsigned int count, void *params,
                                    unsigned int flags)
{
    if (count && !params)
        return CUDA_ERROR_INVALID_VALUE;
    HETGPU_FORWARD_BODY(cuStreamBatchMemOp, stream, count, params, flags);
}

int hetgpu_cuda_stream_get_capture_info(HetGPUState *state, HetGPUStream stream,
                                        int *status, uint64_t *id, void **graph,
                                        void ***dependencies, size_t *count)
{
    if (!status)
        return CUDA_ERROR_INVALID_VALUE;
    HETGPU_FORWARD_BODY(cuStreamGetCaptureInfo_v2, stream, status, id, graph,
                        dependencies, count);
}

int hetgpu_cuda_stream_get_ctx(HetGPUState *state, HetGPUStream stream,
                               void **context)
{
    if (!context)
        return CUDA_ERROR_INVALID_VALUE;
    HETGPU_FORWARD_BODY(cuStreamGetCtx, stream, context);
}

int hetgpu_cuda_stream_begin_capture(HetGPUState *state, HetGPUStream stream,
                                     int mode)
{
    HETGPU_FORWARD_BODY(cuStreamBeginCapture, stream, mode);
}

int hetgpu_cuda_stream_end_capture(HetGPUState *state, HetGPUStream stream,
                                   void **graph)
{
    if (!graph)
        return CUDA_ERROR_INVALID_VALUE;
    HETGPU_FORWARD_BODY(cuStreamEndCapture, stream, graph);
}

int hetgpu_cuda_stream_is_capturing(HetGPUState *state, HetGPUStream stream,
                                    int *status)
{
    if (!status)
        return CUDA_ERROR_INVALID_VALUE;
    HETGPU_FORWARD_BODY(cuStreamIsCapturing, stream, status);
}

int hetgpu_cuda_event_create(HetGPUState *state, unsigned int flags, void **event)
{
    if (!event)
        return CUDA_ERROR_INVALID_VALUE;
    HETGPU_FORWARD_BODY(cuEventCreate, event, flags);
}

int hetgpu_cuda_event_destroy(HetGPUState *state, void *event)
{
    if (!event)
        return CUDA_ERROR_INVALID_VALUE;
    HETGPU_FORWARD_BODY(cuEventDestroy, event);
}

int hetgpu_cuda_event_record(HetGPUState *state, void *event, HetGPUStream stream)
{
    if (!event)
        return CUDA_ERROR_INVALID_VALUE;
    HETGPU_FORWARD_BODY(cuEventRecord, event, stream);
}

int hetgpu_cuda_event_query(HetGPUState *state, void *event)
{
    if (!event)
        return CUDA_ERROR_INVALID_VALUE;
    HETGPU_FORWARD_BODY(cuEventQuery, event);
}

int hetgpu_cuda_event_synchronize(HetGPUState *state, void *event)
{
    if (!event)
        return CUDA_ERROR_INVALID_VALUE;
    HETGPU_FORWARD_BODY(cuEventSynchronize, event);
}

int hetgpu_cuda_event_elapsed_time(HetGPUState *state, float *milliseconds,
                                   void *start, void *end)
{
    if (!milliseconds || !start || !end)
        return CUDA_ERROR_INVALID_VALUE;
    HETGPU_FORWARD_BODY(cuEventElapsedTime, milliseconds, start, end);
}

int hetgpu_cuda_mem_prefetch_async(HetGPUState *state, uint64_t ptr, size_t count,
                                   int device, HetGPUStream stream)
{
    HETGPU_FORWARD_BODY(cuMemPrefetchAsync, ptr, count, device, stream);
}

int hetgpu_cuda_ctx_enable_peer_access(HetGPUState *state, void *peer_context,
                                       unsigned int flags)
{
    if (!peer_context)
        return CUDA_ERROR_INVALID_VALUE;
    HETGPU_FORWARD_BODY(cuCtxEnablePeerAccess, peer_context, flags);
}

int hetgpu_cuda_ctx_disable_peer_access(HetGPUState *state, void *peer_context)
{
    if (!peer_context)
        return CUDA_ERROR_INVALID_VALUE;
    HETGPU_FORWARD_BODY(cuCtxDisablePeerAccess, peer_context);
}

int hetgpu_cuda_device_can_access_peer(HetGPUState *state, int *can_access,
                                       int device, int peer_device)
{
    if (!can_access)
        return CUDA_ERROR_INVALID_VALUE;
    HETGPU_FORWARD_BODY(cuDeviceCanAccessPeer, can_access, device, peer_device);
}

#undef HETGPU_FORWARD_BODY

HetGPUError hetgpu_create_stream(HetGPUState *state, HetGPUStream *stream)
{
    return (HetGPUError)hetgpu_cuda_stream_create(state, 0, stream);
}

void hetgpu_destroy_stream(HetGPUState *state, HetGPUStream stream)
{
    (void)hetgpu_cuda_stream_destroy(state, stream);
}

HetGPUError hetgpu_stream_synchronize(HetGPUState *state, HetGPUStream stream)
{
    return (HetGPUError)hetgpu_cuda_stream_synchronize(state, stream);
}

const char *hetgpu_get_error_string(HetGPUError error)
{
    switch (error) {
    case HETGPU_SUCCESS:
        return "Success";
    case HETGPU_ERROR_NOT_INITIALIZED:
        return "Not initialized";
    case HETGPU_ERROR_NO_DEVICE:
        return "No device";
    case HETGPU_ERROR_INVALID_DEVICE:
        return "Invalid device";
    case HETGPU_ERROR_INVALID_CONTEXT:
        return "Invalid context";
    case HETGPU_ERROR_OUT_OF_MEMORY:
        return "Out of memory";
    case HETGPU_ERROR_INVALID_PTX:
        return "Invalid PTX";
    case HETGPU_ERROR_LAUNCH_FAILED:
        return "Kernel launch failed";
    case HETGPU_ERROR_INVALID_VALUE:
        return "Invalid value";
    case HETGPU_ERROR_NOT_SUPPORTED:
        return "Not supported";
    default:
        return "Unknown error";
    }
}

const char *hetgpu_get_backend_name(HetGPUBackendType backend)
{
    switch (backend) {
    case HETGPU_BACKEND_AUTO:
        return "Auto";
    case HETGPU_BACKEND_INTEL:
        return "Intel Level Zero";
    case HETGPU_BACKEND_AMD:
        return "AMD HIP/ROCm";
    case HETGPU_BACKEND_NVIDIA:
        return "NVIDIA CUDA";
    case HETGPU_BACKEND_TENSTORRENT:
        return "Tenstorrent";
    case HETGPU_BACKEND_SIMULATION:
        return "Simulation";
    default:
        return "Unknown";
    }
}
