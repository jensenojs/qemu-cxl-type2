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
