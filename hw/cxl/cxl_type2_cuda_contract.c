#include "qemu/osdep.h"
#include "hw/cxl/cxl_type2_cuda_contract.h"

static bool cxl_type2_descriptor_reserved_is_zero(
    const CXLGPURAMCommandDescriptor *descriptor)
{
    for (size_t i = 0; i < sizeof(descriptor->reserved); i++) {
        if (descriptor->reserved[i] != 0) {
            return false;
        }
    }
    return true;
}

CXLType2DescriptorRequestVerdict cxl_type2_descriptor_validate_request(
    const CXLGPURAMCommandDescriptor *request, uint64_t doorbell_value,
    unsigned doorbell_size, uint64_t device_generation,
    uint64_t last_accepted_submission, uint64_t last_completed_submission,
    bool paired_case_required, uint64_t active_case_epoch)
{
    if (!request || doorbell_size != sizeof(uint32_t) ||
        doorbell_value != CXL_GPU_DESCRIPTOR_DOORBELL_VALUE) {
        return CXL_TYPE2_DESCRIPTOR_INVALID_DOORBELL;
    }
    if (device_generation == 0 ||
        request->protocol_version != CXL_GPU_DESCRIPTOR_PROTOCOL_VERSION ||
        request->descriptor_size != CXL_GPU_DESCRIPTOR_WIRE_SIZE ||
        !cxl_type2_descriptor_reserved_is_zero(request)) {
        return CXL_TYPE2_DESCRIPTOR_INVALID_HEADER;
    }
    if (request->request_device_generation != device_generation ||
        request->request_submission == 0 ||
        request->request_submission == UINT64_MAX) {
        return CXL_TYPE2_DESCRIPTOR_INVALID_IDENTITY;
    }
    if (request->request_submission == last_completed_submission) {
        return CXL_TYPE2_DESCRIPTOR_DUPLICATE;
    }
    if (request->request_submission <= last_accepted_submission) {
        return CXL_TYPE2_DESCRIPTOR_STALE_SUBMISSION;
    }
    if (request->request_command == CXL_GPU_CMD_CASE_BEGIN) {
        if (request->request_case_epoch != 0) {
            return CXL_TYPE2_DESCRIPTOR_INVALID_CASE_EPOCH;
        }
    } else if (paired_case_required) {
        if (request->request_case_epoch != active_case_epoch) {
            return CXL_TYPE2_DESCRIPTOR_INVALID_CASE_EPOCH;
        }
    } else if (request->request_case_epoch != 0) {
        return CXL_TYPE2_DESCRIPTOR_INVALID_CASE_EPOCH;
    }
    return CXL_TYPE2_DESCRIPTOR_ACCEPT;
}

const char *cxl_type2_descriptor_verdict_reason(
    CXLType2DescriptorRequestVerdict verdict)
{
    switch (verdict) {
    case CXL_TYPE2_DESCRIPTOR_ACCEPT:
        return "accepted";
    case CXL_TYPE2_DESCRIPTOR_DUPLICATE:
        return "duplicate completed submission";
    case CXL_TYPE2_DESCRIPTOR_INVALID_DOORBELL:
        return "invalid doorbell";
    case CXL_TYPE2_DESCRIPTOR_INVALID_HEADER:
        return "invalid descriptor header";
    case CXL_TYPE2_DESCRIPTOR_INVALID_IDENTITY:
        return "invalid request identity";
    case CXL_TYPE2_DESCRIPTOR_STALE_SUBMISSION:
        return "stale submission";
    case CXL_TYPE2_DESCRIPTOR_INVALID_CASE_EPOCH:
        return "case epoch mismatch";
    default:
        g_assert_not_reached();
    }
}

bool cxl_type2_cuda_decode_attribute(uint64_t wire_value, int32_t *attribute)
{
    int32_t decoded;

    if (!attribute) {
        return false;
    }
    decoded = (int32_t)wire_value;
    if ((uint64_t)(int64_t)decoded != wire_value) {
        return false;
    }
    *attribute = decoded;
    return true;
}

bool cxl_type2_cuda_attribute_wire_is_valid(uint64_t wire_value)
{
    int32_t attribute;

    return cxl_type2_cuda_decode_attribute(wire_value, &attribute);
}

bool cxl_type2_cuda_mem_info_is_allowed(bool active_case, bool live_context,
                                        uint64_t token, uint64_t active_epoch)
{
    return active_case && live_context && active_epoch != 0 &&
           token == active_epoch;
}

bool cxl_type2_cuda_dispatch_attribute(uint64_t wire_value,
                                       CXLType2CudaAttributeQuery query,
                                       void *opaque, int *query_result)
{
    int32_t attribute;

    if (!query_result || !query ||
        !cxl_type2_cuda_decode_attribute(wire_value, &attribute)) {
        return false;
    }
    *query_result = query(opaque, attribute);
    return true;
}

bool cxl_type2_cuda_dispatch_mem_info(bool active_case, bool live_context,
                                      uint64_t token, uint64_t active_epoch,
                                      CXLType2CudaMemInfoQuery query,
                                      void *opaque, int *query_result)
{
    if (!query_result || !query ||
        !cxl_type2_cuda_mem_info_is_allowed(active_case, live_context, token,
                                            active_epoch)) {
        return false;
    }
    *query_result = query(opaque);
    return true;
}

bool cxl_gpu_batch_htod_validate(const uint8_t *payload,
                                 uint64_t payload_capacity,
                                 uint64_t expected_range_count,
                                 uint64_t expected_payload_bytes,
                                 uint64_t *fail_idx)
{
    CXLGPUBatchHtoDHeader header;
    uint64_t source_offset;

    if (!fail_idx) {
        return false;
    }
    *fail_idx = SIZE_MAX;
    if (!payload || expected_range_count == 0 ||
        expected_range_count > UINT32_MAX ||
        expected_payload_bytes < sizeof(header) ||
        expected_payload_bytes > payload_capacity) {
        return false;
    }
    memcpy(&header, payload, sizeof(header));
    if (header.header_size != sizeof(header) ||
        header.range_count != expected_range_count ||
        header.range_size != sizeof(CXLGPUBatchHtoDRange) ||
        header.reserved0 != 0 || header.reserved1 != 0 ||
        header.payload_bytes != expected_payload_bytes ||
        expected_range_count >
            (expected_payload_bytes - sizeof(header)) /
                sizeof(CXLGPUBatchHtoDRange)) {
        return false;
    }

    source_offset = sizeof(header) +
                    expected_range_count * sizeof(CXLGPUBatchHtoDRange);
    if (source_offset > UINT64_MAX - 63) {
        return false;
    }
    source_offset = (source_offset + 63) & ~UINT64_C(63);
    if (source_offset > expected_payload_bytes) {
        return false;
    }

    for (uint64_t i = 0; i < expected_range_count; i++) {
        CXLGPUBatchHtoDRange range;

        memcpy(&range, payload + sizeof(header) + i * sizeof(range),
               sizeof(range));
        if (!range.size || range.source_offset != source_offset ||
            range.size > expected_payload_bytes - source_offset ||
            range.destination > UINT64_MAX - range.size) {
            *fail_idx = i;
            return false;
        }
        source_offset += range.size;
    }
    if (source_offset != expected_payload_bytes) {
        *fail_idx = expected_range_count - 1;
        return false;
    }
    return true;
}

int cxl_gpu_batch_htod_enqueue(const uint8_t *payload, uint64_t range_count,
                               CXLGPUBatchHtoDEnqueue enqueue, void *opaque,
                               uint64_t *fail_idx,
                               uint64_t *successfully_enqueued)
{
    if (!payload || !enqueue || !fail_idx || !successfully_enqueued) {
        return CXL_GPU_ERROR_INVALID_VALUE;
    }
    *fail_idx = SIZE_MAX;
    *successfully_enqueued = 0;
    for (uint64_t i = 0; i < range_count; i++) {
        CXLGPUBatchHtoDRange range;
        int result;

        memcpy(&range, payload + sizeof(CXLGPUBatchHtoDHeader) +
                           i * sizeof(range), sizeof(range));
        result = enqueue(opaque, range.destination,
                         payload + range.source_offset, range.size);
        if (result != CXL_GPU_SUCCESS) {
            *fail_idx = i;
            return result;
        }
        (*successfully_enqueued)++;
    }
    return CXL_GPU_SUCCESS;
}

int cxl_gpu_batch_htod_submit(const uint8_t *payload,
                              uint64_t payload_capacity,
                              uint64_t range_count, uint64_t payload_bytes,
                              CXLGPUBatchHtoDEnqueue enqueue, void *opaque,
                              uint64_t *fail_idx,
                              uint64_t *successfully_enqueued)
{
    if (!successfully_enqueued || !fail_idx) {
        return CXL_GPU_ERROR_INVALID_VALUE;
    }
    *successfully_enqueued = 0;
    if (!cxl_gpu_batch_htod_validate(payload, payload_capacity, range_count,
                                     payload_bytes, fail_idx)) {
        return CXL_GPU_ERROR_INVALID_VALUE;
    }
    return cxl_gpu_batch_htod_enqueue(payload, range_count, enqueue, opaque,
                                      fail_idx, successfully_enqueued);
}
