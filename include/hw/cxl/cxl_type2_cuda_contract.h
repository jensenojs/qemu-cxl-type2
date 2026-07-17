#ifndef HW_CXL_TYPE2_CUDA_CONTRACT_H
#define HW_CXL_TYPE2_CUDA_CONTRACT_H

#include <stdbool.h>
#include <stdint.h>

bool cxl_type2_cuda_decode_attribute(uint64_t wire_value, int32_t *attribute);
bool cxl_type2_cuda_attribute_wire_is_valid(uint64_t wire_value);
bool cxl_type2_cuda_mem_info_is_allowed(bool active_case, bool live_context,
                                        uint64_t token, uint64_t active_epoch);

typedef int (*CXLType2CudaAttributeQuery)(void *opaque, int32_t attribute);
typedef int (*CXLType2CudaMemInfoQuery)(void *opaque);

/*
 * The command handler must enter the real CUDA helper only through these
 * dispatch points.  Keeping the gate and call adjacent makes the reject-path
 * side-effect contract unit-testable without constructing a QEMU device.
 */
bool cxl_type2_cuda_dispatch_attribute(uint64_t wire_value,
                                       CXLType2CudaAttributeQuery query,
                                       void *opaque, int *query_result);
bool cxl_type2_cuda_dispatch_mem_info(bool active_case, bool live_context,
                                      uint64_t token, uint64_t active_epoch,
                                      CXLType2CudaMemInfoQuery query,
                                      void *opaque, int *query_result);

#endif
