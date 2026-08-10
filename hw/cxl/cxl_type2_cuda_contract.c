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
        (request->protocol_version < 1U ||
         request->protocol_version > CXL_GPU_DESCRIPTOR_PROTOCOL_VERSION) ||
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

bool cxl_type2_cuda_stream_progress_wire(uint32_t command,
                                         const uint64_t params[8],
                                         uint64_t *stream_wire)
{
    unsigned int parameter;

    if (!params || !stream_wire) {
        return false;
    }
    switch (command) {
    case CXL_GPU_CMD_MEM_COPY_HTOD_ASYNC:
    case CXL_GPU_CMD_BATCH_HTOD_ASYNC:
    case CXL_GPU_CMD_BATCH_HTOD_DIRECT_ASYNC:
    case CXL_GPU_CMD_SOURCE_REGISTER_BATCH_HTOD_DIRECT_ASYNC:
        parameter = 2;
        break;
    case CXL_GPU_CMD_MEM_COPY_DTOD_ASYNC:
    case CXL_GPU_CMD_MEM_PREFETCH_ASYNC:
    case CXL_GPU_CMD_BULK_HTOD_ASYNC:
        parameter = 3;
        break;
    case CXL_GPU_CMD_GRAPH_LAUNCH:
    case CXL_GPU_CMD_EVENT_RECORD:
        parameter = 1;
        break;
    case CXL_GPU_CMD_LAUNCH_KERNEL:
        parameter = 6;
        break;
    case CXL_GPU_CMD_STREAM_WAIT_EVENT:
    case CXL_GPU_CMD_STREAM_WAIT_VALUE32:
    case CXL_GPU_CMD_STREAM_BATCH_MEM_OP:
    case CXL_GPU_CMD_STREAM_BEGIN_CAPTURE:
    case CXL_GPU_CMD_STREAM_END_CAPTURE:
        parameter = 0;
        break;
    default:
        return false;
    }
    *stream_wire = params[parameter];
    return true;
}

bool cxl_type2_cuda_adjacent_stream_sync_can_elide(
    bool previous_command_was_successful_sync,
    uint64_t previous_stream_wire,
    uint64_t current_stream_wire)
{
    return previous_command_was_successful_sync &&
           previous_stream_wire == current_stream_wire;
}

bool cxl_type2_cuda_decode_stream_sync_reason(
    uint32_t descriptor_protocol_version, uint64_t wire_reason,
    CXLGPUStreamSyncReason *reason)
{
    if (!reason) {
        return false;
    }
    if (descriptor_protocol_version < 3U) {
        *reason = CXL_GPU_STREAM_SYNC_PUBLIC_API;
        return true;
    }
    if (wire_reason >= CXL_GPU_STREAM_SYNC_REASON_COUNT) {
        return false;
    }
    *reason = wire_reason;
    return true;
}

bool cxl_type2_cuda_special_stream_from_wire(uint64_t wire,
                                              void *per_thread_stream,
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
    if (wire == CXL_GPU_STREAM_WIRE_PER_THREAD && per_thread_stream) {
        *stream = per_thread_stream;
        return true;
    }
    return false;
}

int cxl_gpu_direct_host_address_order(uintptr_t left, uintptr_t right)
{
    return left < right ? -1 : left > right;
}

bool cxl_gpu_direct_host_range_follows(uintptr_t base, uint64_t length,
                                       uintptr_t next, uint64_t next_length)
{
    return length <= UINTPTR_MAX - base && next == base + length &&
           next_length <= UINT64_MAX - length;
}

bool cxl_gpu_direct_epoch_is_cross_case(uint64_t last_case_epoch,
                                        uint64_t active_case_epoch)
{
    return last_case_epoch && active_case_epoch &&
           last_case_epoch != active_case_epoch;
}

bool cxl_gpu_direct_registration_group_follows(
    uintptr_t mapping, uintptr_t base, uint64_t length,
    uintptr_t next_mapping, uintptr_t next, uint64_t next_length)
{
    return mapping == next_mapping &&
           cxl_gpu_direct_host_range_follows(
               base, length, next, next_length);
}

bool cxl_gpu_direct_copy_span_follows(
    uintptr_t source, uintptr_t registration, uintptr_t host,
    uint64_t destination, uint64_t length, uintptr_t next_source,
    uintptr_t next_registration, uintptr_t next_host,
    uint64_t next_destination, uint64_t next_length)
{
    return source == next_source && registration == next_registration &&
           cxl_gpu_direct_host_range_follows(
               host, length, next_host, next_length) &&
           length <= UINT64_MAX - destination &&
           next_destination == destination + length;
}

uint64_t cxl_gpu_direct_registration_length(
    uint64_t mapping_offset, uint64_t mapping_length,
    uint64_t request_offset, uint64_t request_length,
    uint64_t following_offset, uint64_t tile_size,
    uint64_t padding_budget)
{
    uint64_t mapping_end;
    uint64_t request_end;
    uint64_t limit;
    uint64_t relative_end;
    uint64_t rounded_end;
    uint64_t tile_end;

    if (!mapping_length || !request_length ||
        mapping_offset > UINT64_MAX - mapping_length ||
        request_offset < mapping_offset ||
        request_offset > UINT64_MAX - request_length) {
        return 0;
    }
    mapping_end = mapping_offset + mapping_length;
    request_end = request_offset + request_length;
    if (request_end > mapping_end || following_offset < request_end ||
        following_offset > mapping_end) {
        return 0;
    }
    if (!tile_size || !padding_budget || request_end == following_offset) {
        return request_length;
    }

    relative_end = request_end - mapping_offset;
    if (relative_end > UINT64_MAX - (tile_size - 1)) {
        rounded_end = mapping_length;
    } else {
        rounded_end =
            ((relative_end + tile_size - 1) / tile_size) * tile_size;
        rounded_end = MIN(rounded_end, mapping_length);
    }
    tile_end = mapping_offset + rounded_end;
    tile_end = MIN(tile_end, following_offset);
    limit = padding_budget > UINT64_MAX - request_end
                ? UINT64_MAX
                : request_end + padding_budget;
    tile_end = MIN(tile_end, limit);
    return tile_end - request_offset;
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

bool cxl_gpu_source_register_validate(const uint8_t *payload,
                                      uint64_t payload_capacity,
                                      uint64_t payload_bytes,
                                      CXLGPUSourceRegisterV1 *header_out,
                                      uint64_t *fail_index)
{
    CXLGPUSourceRegisterV1 header;
    const uint8_t *range_base;
    const uint8_t *run_base;
    uint64_t ranges_bytes;
    uint64_t runs_bytes;
    uint64_t logical_bytes = 0;
    uint64_t unique_bytes = 0;

    if (!payload || !header_out || !fail_index ||
        payload_bytes < sizeof(header) || payload_bytes > payload_capacity) {
        return false;
    }
    *fail_index = SIZE_MAX;
    memcpy(&header, payload, sizeof(header));
    if (header.flags || header.reserved0 || header.reserved1 ||
        !header.lease_handle || !header.range_count || !header.run_count ||
        header.range_count > UINT32_MAX || header.run_count > UINT32_MAX ||
        header.range_count >
            (payload_bytes - sizeof(header)) / sizeof(CXLGPUSourceRangeV1)) {
        return false;
    }
    ranges_bytes = (uint64_t)header.range_count * sizeof(CXLGPUSourceRangeV1);
    if (header.run_count >
        (payload_bytes - sizeof(header) - ranges_bytes) /
            sizeof(CXLGPUSourceRunV1)) {
        return false;
    }
    runs_bytes = (uint64_t)header.run_count * sizeof(CXLGPUSourceRunV1);
    if (sizeof(header) + ranges_bytes + runs_bytes != payload_bytes) {
        return false;
    }
    range_base = payload + sizeof(header);
    run_base = range_base + ranges_bytes;

    for (uint64_t i = 0; i < header.run_count; i++) {
        CXLGPUSourceRunV1 run;
        bool duplicate = false;

        memcpy(&run, run_base + i * sizeof(run), sizeof(run));
        if (!run.length || run.guest_phys_addr > UINT64_MAX - run.length) {
            *fail_index = i;
            return false;
        }
        for (uint64_t j = 0; j < i; j++) {
            CXLGPUSourceRunV1 previous;

            memcpy(&previous, run_base + j * sizeof(previous),
                   sizeof(previous));
            if (run.guest_phys_addr == previous.guest_phys_addr &&
                run.length == previous.length) {
                duplicate = true;
                break;
            }
            if (run.guest_phys_addr <
                    previous.guest_phys_addr + previous.length &&
                previous.guest_phys_addr < run.guest_phys_addr + run.length) {
                *fail_index = i;
                return false;
            }
        }
        if (!duplicate) {
            if (unique_bytes > UINT64_MAX - run.length) {
                *fail_index = i;
                return false;
            }
            unique_bytes += run.length;
        }
    }
    if (unique_bytes != header.unique_dmap_bytes) {
        return false;
    }

    for (uint64_t i = 0; i < header.range_count; i++) {
        CXLGPUSourceRangeV1 range;
        uint64_t available = 0;

        memcpy(&range, range_base + i * sizeof(range), sizeof(range));
        if (!range.run_count || !range.length ||
            range.first_run >= header.run_count ||
            range.run_count > header.run_count - range.first_run) {
            *fail_index = i;
            return false;
        }
        for (uint64_t j = 0; j < range.run_count; j++) {
            CXLGPUSourceRunV1 run;
            uint64_t offset = j ? 0 : range.first_run_byte_offset;

            memcpy(&run, run_base +
                   (range.first_run + j) * sizeof(run), sizeof(run));
            if (offset >= run.length || available > UINT64_MAX -
                                              (run.length - offset)) {
                *fail_index = i;
                return false;
            }
            available += run.length - offset;
        }
        if (range.length > available || logical_bytes > UINT64_MAX - range.length) {
            *fail_index = i;
            return false;
        }
        logical_bytes += range.length;
    }
    if (logical_bytes != header.logical_bytes) {
        return false;
    }
    *header_out = header;
    return true;
}

bool cxl_gpu_direct_batch_validate(const uint8_t *payload,
                                   uint64_t payload_capacity,
                                   uint64_t range_count,
                                   uint64_t payload_bytes,
                                   uint64_t *fail_index)
{
    if (!payload || !fail_index || !range_count ||
        range_count > payload_capacity / sizeof(CXLGPUDirectRangeV1) ||
        payload_bytes != range_count * sizeof(CXLGPUDirectRangeV1)) {
        return false;
    }
    *fail_index = SIZE_MAX;
    for (uint64_t i = 0; i < range_count; i++) {
        CXLGPUDirectRangeV1 range;

        memcpy(&range, payload + i * sizeof(range), sizeof(range));
        if (!range.size || range.reserved0 ||
            range.destination > UINT64_MAX - range.size ||
            range.source_offset > UINT64_MAX - range.size) {
            *fail_index = i;
            return false;
        }
    }
    return true;
}
