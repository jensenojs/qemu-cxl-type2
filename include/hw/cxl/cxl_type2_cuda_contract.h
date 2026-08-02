#ifndef HW_CXL_TYPE2_CUDA_CONTRACT_H
#define HW_CXL_TYPE2_CUDA_CONTRACT_H

#include <stdbool.h>
#include <stdint.h>

#include "hw/cxl/cxl_type2_gpu_cmd.h"

typedef enum CXLType2DescriptorRequestVerdict {
    CXL_TYPE2_DESCRIPTOR_ACCEPT,
    CXL_TYPE2_DESCRIPTOR_DUPLICATE,
    CXL_TYPE2_DESCRIPTOR_INVALID_DOORBELL,
    CXL_TYPE2_DESCRIPTOR_INVALID_HEADER,
    CXL_TYPE2_DESCRIPTOR_INVALID_IDENTITY,
    CXL_TYPE2_DESCRIPTOR_STALE_SUBMISSION,
    CXL_TYPE2_DESCRIPTOR_INVALID_CASE_EPOCH,
} CXLType2DescriptorRequestVerdict;

CXLType2DescriptorRequestVerdict cxl_type2_descriptor_validate_request(
    const CXLGPURAMCommandDescriptor *request, uint64_t doorbell_value,
    unsigned doorbell_size, uint64_t device_generation,
    uint64_t last_accepted_submission, uint64_t last_completed_submission,
    bool paired_case_required, uint64_t active_case_epoch);
const char *cxl_type2_descriptor_verdict_reason(
    CXLType2DescriptorRequestVerdict verdict);

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
