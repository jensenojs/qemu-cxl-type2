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
bool cxl_type2_cuda_stream_progress_wire(uint32_t command,
                                         const uint64_t params[8],
                                         uint64_t *stream_wire);
bool cxl_type2_cuda_adjacent_stream_sync_can_elide(
    bool previous_command_was_successful_sync,
    uint64_t previous_stream_wire,
    uint64_t current_stream_wire);
bool cxl_type2_cuda_special_stream_from_wire(uint64_t wire,
                                              void *per_thread_stream,
                                              void **stream);
bool cxl_type2_cuda_mem_info_is_allowed(bool active_case, bool live_context,
                                        uint64_t token, uint64_t active_epoch);

typedef int (*CXLType2CudaAttributeQuery)(void *opaque, int32_t attribute);
typedef int (*CXLType2CudaMemInfoQuery)(void *opaque);
typedef int (*CXLGPUBatchHtoDEnqueue)(void *opaque, uint64_t destination,
                                      const void *source, size_t size);

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
bool cxl_gpu_batch_htod_validate(const uint8_t *payload,
                                 uint64_t payload_capacity,
                                 uint64_t expected_range_count,
                                 uint64_t expected_payload_bytes,
                                 uint64_t *fail_idx);
int cxl_gpu_batch_htod_enqueue(const uint8_t *payload, uint64_t range_count,
                               CXLGPUBatchHtoDEnqueue enqueue, void *opaque,
                               uint64_t *fail_idx,
                               uint64_t *successfully_enqueued);
int cxl_gpu_batch_htod_submit(const uint8_t *payload,
                              uint64_t payload_capacity,
                              uint64_t range_count, uint64_t payload_bytes,
                              CXLGPUBatchHtoDEnqueue enqueue, void *opaque,
                              uint64_t *fail_idx,
                              uint64_t *successfully_enqueued);
bool cxl_gpu_source_register_validate(const uint8_t *payload,
                                      uint64_t payload_capacity,
                                      uint64_t payload_bytes,
                                      CXLGPUSourceRegisterV1 *header_out,
                                      uint64_t *fail_index);
bool cxl_gpu_direct_batch_validate(const uint8_t *payload,
                                   uint64_t payload_capacity,
                                   uint64_t range_count,
                                   uint64_t payload_bytes,
                                   uint64_t *fail_index);
int cxl_gpu_direct_host_address_order(uintptr_t left, uintptr_t right);
bool cxl_gpu_direct_host_range_follows(uintptr_t base, uint64_t length,
                                       uintptr_t next, uint64_t next_length);
bool cxl_gpu_direct_registration_group_follows(
    uintptr_t mapping, uintptr_t base, uint64_t length,
    uintptr_t next_mapping, uintptr_t next, uint64_t next_length);
bool cxl_gpu_direct_copy_span_follows(
    uintptr_t source, uintptr_t registration, uintptr_t host,
    uint64_t destination, uint64_t length, uintptr_t next_source,
    uintptr_t next_registration, uintptr_t next_host,
    uint64_t next_destination, uint64_t next_length);
uint64_t cxl_gpu_direct_registration_length(
    uint64_t mapping_offset, uint64_t mapping_length,
    uint64_t request_offset, uint64_t request_length,
    uint64_t following_offset, uint64_t tile_size,
    uint64_t padding_budget);

#endif
