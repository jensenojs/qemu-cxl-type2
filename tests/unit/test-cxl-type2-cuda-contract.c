#include "qemu/osdep.h"
#include "hw/cxl/cxl_type2_cuda_contract.h"

typedef struct QueryCounter {
    unsigned int calls;
    int32_t attribute;
} QueryCounter;

typedef struct BatchEnqueueCounter {
    uint64_t calls;
    uint64_t fail_at;
    uint64_t bytes;
} BatchEnqueueCounter;

static uint8_t *valid_batch_payload(uint64_t *payload_bytes)
{
    const uint64_t count = 3;
    const uint64_t source_begin = 128;
    const uint64_t sizes[] = { 3, 5, 7 };
    CXLGPUBatchHtoDHeader header = {
        .header_size = sizeof(CXLGPUBatchHtoDHeader),
        .range_count = count,
        .range_size = sizeof(CXLGPUBatchHtoDRange),
        .payload_bytes = source_begin + 15,
    };
    uint8_t *payload = g_malloc0(header.payload_bytes);
    uint64_t source_offset = source_begin;

    memcpy(payload, &header, sizeof(header));
    for (uint64_t i = 0; i < count; i++) {
        CXLGPUBatchHtoDRange range = {
            .source_offset = source_offset,
            .destination = UINT64_C(0x1000) + i * 0x100,
            .size = sizes[i],
        };

        memcpy(payload + sizeof(header) + i * sizeof(range),
               &range, sizeof(range));
        memset(payload + source_offset, (int)(i + 1), sizes[i]);
        source_offset += sizes[i];
    }
    *payload_bytes = header.payload_bytes;
    return payload;
}

static int count_batch_enqueue(void *opaque, uint64_t destination,
                               const void *source, size_t size)
{
    BatchEnqueueCounter *counter = opaque;

    g_assert_cmphex(destination, ==,
                    UINT64_C(0x1000) + counter->calls * 0x100);
    g_assert_cmpuint(*(const uint8_t *)source, ==, counter->calls + 1);
    if (counter->calls == counter->fail_at) {
        return CXL_GPU_ERROR_OUT_OF_MEMORY;
    }
    counter->calls++;
    counter->bytes += size;
    return CXL_GPU_SUCCESS;
}

static int count_attribute_query(void *opaque, int32_t attribute)
{
    QueryCounter *counter = opaque;

    counter->calls++;
    counter->attribute = attribute;
    return 37;
}

static int count_mem_info_query(void *opaque)
{
    QueryCounter *counter = opaque;

    counter->calls++;
    return 41;
}

static CXLGPURAMCommandDescriptor valid_descriptor_request(void)
{
    CXLGPURAMCommandDescriptor request = {
        .protocol_version = CXL_GPU_DESCRIPTOR_PROTOCOL_VERSION,
        .descriptor_size = CXL_GPU_DESCRIPTOR_WIRE_SIZE,
        .request_submission = 7,
        .request_device_generation = 3,
        .request_command = CXL_GPU_CMD_MEM_ALLOC,
    };

    return request;
}

static CXLType2DescriptorRequestVerdict validate_descriptor(
    const CXLGPURAMCommandDescriptor *request, uint64_t doorbell_value,
    unsigned doorbell_size, uint64_t generation, uint64_t last_accepted,
    uint64_t last_completed, bool paired_required, uint64_t active_epoch)
{
    return cxl_type2_descriptor_validate_request(
        request, doorbell_value, doorbell_size, generation, last_accepted,
        last_completed, paired_required, active_epoch);
}

static void test_descriptor_accept_and_duplicate(void)
{
    CXLGPURAMCommandDescriptor request = valid_descriptor_request();

    g_assert_cmpint(validate_descriptor(
                        &request, CXL_GPU_DESCRIPTOR_DOORBELL_VALUE,
                        sizeof(uint32_t), 3, 6, 6, false, 0),
                    ==, CXL_TYPE2_DESCRIPTOR_ACCEPT);
    g_assert_cmpint(validate_descriptor(
                        &request, CXL_GPU_DESCRIPTOR_DOORBELL_VALUE,
                        sizeof(uint32_t), 3, 7, 7, false, 0),
                    ==, CXL_TYPE2_DESCRIPTOR_DUPLICATE);
}

static void test_descriptor_rejects_wire_errors(void)
{
    CXLGPURAMCommandDescriptor request = valid_descriptor_request();

    g_assert_cmpint(validate_descriptor(&request, 2, sizeof(uint32_t), 3, 6,
                                        6, false, 0),
                    ==, CXL_TYPE2_DESCRIPTOR_INVALID_DOORBELL);
    g_assert_cmpint(validate_descriptor(
                        &request, CXL_GPU_DESCRIPTOR_DOORBELL_VALUE,
                        sizeof(uint64_t), 3, 6, 6, false, 0),
                    ==, CXL_TYPE2_DESCRIPTOR_INVALID_DOORBELL);

    request.protocol_version++;
    g_assert_cmpint(validate_descriptor(
                        &request, CXL_GPU_DESCRIPTOR_DOORBELL_VALUE,
                        sizeof(uint32_t), 3, 6, 6, false, 0),
                    ==, CXL_TYPE2_DESCRIPTOR_INVALID_HEADER);
    request = valid_descriptor_request();
    request.descriptor_size--;
    g_assert_cmpint(validate_descriptor(
                        &request, CXL_GPU_DESCRIPTOR_DOORBELL_VALUE,
                        sizeof(uint32_t), 3, 6, 6, false, 0),
                    ==, CXL_TYPE2_DESCRIPTOR_INVALID_HEADER);
    request = valid_descriptor_request();
    request.reserved[0] = 1;
    g_assert_cmpint(validate_descriptor(
                        &request, CXL_GPU_DESCRIPTOR_DOORBELL_VALUE,
                        sizeof(uint32_t), 3, 6, 6, false, 0),
                    ==, CXL_TYPE2_DESCRIPTOR_INVALID_HEADER);
}

static void test_descriptor_rejects_identity_errors(void)
{
    CXLGPURAMCommandDescriptor request = valid_descriptor_request();

    g_assert_cmpint(validate_descriptor(
                        &request, CXL_GPU_DESCRIPTOR_DOORBELL_VALUE,
                        sizeof(uint32_t), 0, 6, 6, false, 0),
                    ==, CXL_TYPE2_DESCRIPTOR_INVALID_HEADER);
    request.request_device_generation = 2;
    g_assert_cmpint(validate_descriptor(
                        &request, CXL_GPU_DESCRIPTOR_DOORBELL_VALUE,
                        sizeof(uint32_t), 3, 6, 6, false, 0),
                    ==, CXL_TYPE2_DESCRIPTOR_INVALID_IDENTITY);
    request = valid_descriptor_request();
    request.request_submission = 0;
    g_assert_cmpint(validate_descriptor(
                        &request, CXL_GPU_DESCRIPTOR_DOORBELL_VALUE,
                        sizeof(uint32_t), 3, 6, 6, false, 0),
                    ==, CXL_TYPE2_DESCRIPTOR_INVALID_IDENTITY);
    request = valid_descriptor_request();
    request.request_submission = UINT64_MAX;
    g_assert_cmpint(validate_descriptor(
                        &request, CXL_GPU_DESCRIPTOR_DOORBELL_VALUE,
                        sizeof(uint32_t), 3, 6, 6, false, 0),
                    ==, CXL_TYPE2_DESCRIPTOR_INVALID_IDENTITY);
    request = valid_descriptor_request();
    request.request_submission = 6;
    g_assert_cmpint(validate_descriptor(
                        &request, CXL_GPU_DESCRIPTOR_DOORBELL_VALUE,
                        sizeof(uint32_t), 3, 6, 5, false, 0),
                    ==, CXL_TYPE2_DESCRIPTOR_STALE_SUBMISSION);
}

static void test_descriptor_case_epoch_gate(void)
{
    CXLGPURAMCommandDescriptor request = valid_descriptor_request();

    request.request_command = CXL_GPU_CMD_CASE_BEGIN;
    request.request_case_epoch = 9;
    g_assert_cmpint(validate_descriptor(
                        &request, CXL_GPU_DESCRIPTOR_DOORBELL_VALUE,
                        sizeof(uint32_t), 3, 6, 6, true, 9),
                    ==, CXL_TYPE2_DESCRIPTOR_INVALID_CASE_EPOCH);
    request.request_case_epoch = 0;
    g_assert_cmpint(validate_descriptor(
                        &request, CXL_GPU_DESCRIPTOR_DOORBELL_VALUE,
                        sizeof(uint32_t), 3, 6, 6, true, 9),
                    ==, CXL_TYPE2_DESCRIPTOR_ACCEPT);

    request = valid_descriptor_request();
    request.request_case_epoch = 9;
    g_assert_cmpint(validate_descriptor(
                        &request, CXL_GPU_DESCRIPTOR_DOORBELL_VALUE,
                        sizeof(uint32_t), 3, 6, 6, true, 9),
                    ==, CXL_TYPE2_DESCRIPTOR_ACCEPT);
    request.request_case_epoch = 8;
    g_assert_cmpint(validate_descriptor(
                        &request, CXL_GPU_DESCRIPTOR_DOORBELL_VALUE,
                        sizeof(uint32_t), 3, 6, 6, true, 9),
                    ==, CXL_TYPE2_DESCRIPTOR_INVALID_CASE_EPOCH);
    request.request_case_epoch = 9;
    g_assert_cmpint(validate_descriptor(
                        &request, CXL_GPU_DESCRIPTOR_DOORBELL_VALUE,
                        sizeof(uint32_t), 3, 6, 6, false, 0),
                    ==, CXL_TYPE2_DESCRIPTOR_INVALID_CASE_EPOCH);
}

static void test_decode_attribute(void)
{
    int32_t attribute = 0;

    g_assert_true(cxl_type2_cuda_decode_attribute(
        (uint64_t)(int64_t)(int32_t)-1, &attribute));
    g_assert_cmpint(attribute, ==, -1);
    g_assert_true(cxl_type2_cuda_decode_attribute(97, &attribute));
    g_assert_cmpint(attribute, ==, 97);
    g_assert_false(cxl_type2_cuda_decode_attribute(UINT64_C(0x0000000100000008),
                                                    &attribute));
    g_assert_false(cxl_type2_cuda_decode_attribute(8, NULL));
}

static void test_mem_info_gate(void)
{
    g_assert_true(cxl_type2_cuda_mem_info_is_allowed(true, true, 19, 19));
    g_assert_false(cxl_type2_cuda_mem_info_is_allowed(false, true, 19, 19));
    g_assert_false(cxl_type2_cuda_mem_info_is_allowed(true, false, 19, 19));
    g_assert_false(cxl_type2_cuda_mem_info_is_allowed(true, true, 18, 19));
    g_assert_false(cxl_type2_cuda_mem_info_is_allowed(true, true, 0, 0));
}

static void test_attribute_dispatch_rejects_before_query(void)
{
    QueryCounter counter = { 0 };
    int result = 0;

    g_assert_false(cxl_type2_cuda_dispatch_attribute(
        UINT64_C(0x0000000100000008), count_attribute_query, &counter,
        &result));
    g_assert_cmpuint(counter.calls, ==, 0);
    g_assert_true(cxl_type2_cuda_dispatch_attribute(
        (uint64_t)(int64_t)(int32_t)-1, count_attribute_query, &counter,
        &result));
    g_assert_cmpuint(counter.calls, ==, 1);
    g_assert_cmpint(counter.attribute, ==, -1);
    g_assert_cmpint(result, ==, 37);
}

static void test_mem_info_dispatch_rejects_before_query(void)
{
    QueryCounter counter = { 0 };
    int result = 0;

    g_assert_false(cxl_type2_cuda_dispatch_mem_info(false, true, 19, 19,
                                                     count_mem_info_query,
                                                     &counter, &result));
    g_assert_false(cxl_type2_cuda_dispatch_mem_info(true, false, 19, 19,
                                                     count_mem_info_query,
                                                     &counter, &result));
    g_assert_false(cxl_type2_cuda_dispatch_mem_info(true, true, 18, 19,
                                                     count_mem_info_query,
                                                     &counter, &result));
    g_assert_cmpuint(counter.calls, ==, 0);
    g_assert_true(cxl_type2_cuda_dispatch_mem_info(true, true, 19, 19,
                                                    count_mem_info_query,
                                                    &counter, &result));
    g_assert_cmpuint(counter.calls, ==, 1);
    g_assert_cmpint(result, ==, 41);
}

static void test_batch_wire_layout_and_validation(void)
{
    uint64_t payload_bytes;
    uint64_t fail_idx;
    uint8_t *payload = valid_batch_payload(&payload_bytes);
    CXLGPUBatchHtoDRange range;

    g_assert_cmphex(CXL_GPU_VERSION, ==, UINT64_C(0x00011300));
    g_assert_cmphex(CXL_GPU_CMD_SOURCE_REGISTER_BATCH_HTOD_DIRECT_ASYNC,
                    ==, 0x35);
    g_assert_cmphex(CXL_GPU_CMD_MEM_COPY_DTOD_ASYNC, ==, 0x2f);
    g_assert_cmphex(CXL_GPU_CMD_COHERENT_MAP_DEVICE, ==, 0xa7);
    g_assert_cmphex(CXL_GPU_CMD_COHERENT_UNMAP_DEVICE, ==, 0xa8);
    g_assert_cmphex(CXL_GPU_CMD_COHERENT_STALE_ALIAS_PROBE, ==, 0xa9);
    g_assert_cmpuint(CXL_GPU_DESCRIPTOR_PROTOCOL_VERSION, ==, 3);
    g_assert_cmpuint(CXL_GPU_CASE_PROTOCOL_VERSION, ==, 1);
    g_assert_cmphex(CXL_GPU_BATCH_DATA_OFFSET, ==, UINT64_C(0x802000));
    g_assert_cmphex(CXL_GPU_CMD_REG_SIZE, ==, UINT64_C(0x2802000));
    g_assert_true(cxl_gpu_batch_htod_validate(
        payload, CXL_GPU_BATCH_DATA_SIZE, 3, payload_bytes, &fail_idx));
    g_assert_cmpuint(fail_idx, ==, SIZE_MAX);

    memcpy(&range, payload + sizeof(CXLGPUBatchHtoDHeader) + sizeof(range),
           sizeof(range));
    range.source_offset = 64;
    memcpy(payload + sizeof(CXLGPUBatchHtoDHeader) + sizeof(range),
           &range, sizeof(range));
    g_assert_false(cxl_gpu_batch_htod_validate(
        payload, CXL_GPU_BATCH_DATA_SIZE, 3, payload_bytes, &fail_idx));
    g_assert_cmpuint(fail_idx, ==, 1);
    g_free(payload);
}

static void test_batch_rejects_header_without_enqueue(void)
{
    uint64_t payload_bytes;
    uint64_t fail_idx;
    uint64_t enqueued;
    uint8_t *payload = valid_batch_payload(&payload_bytes);
    CXLGPUBatchHtoDHeader header;
    BatchEnqueueCounter counter = { .fail_at = UINT64_MAX };

    memcpy(&header, payload, sizeof(header));
    header.reserved0 = 1;
    memcpy(payload, &header, sizeof(header));
    g_assert_cmpint(cxl_gpu_batch_htod_submit(
                        payload, CXL_GPU_BATCH_DATA_SIZE, 3, payload_bytes,
                        count_batch_enqueue, &counter, &fail_idx,
                        &enqueued),
                    ==, CXL_GPU_ERROR_INVALID_VALUE);
    g_assert_cmpuint(fail_idx, ==, SIZE_MAX);
    g_assert_cmpuint(enqueued, ==, 0);
    g_assert_cmpuint(counter.calls, ==, 0);
    g_free(payload);
}

static void test_batch_partial_enqueue_result(void)
{
    uint64_t payload_bytes;
    uint64_t fail_idx;
    uint64_t enqueued;
    uint8_t *payload = valid_batch_payload(&payload_bytes);
    BatchEnqueueCounter counter = { .fail_at = 2 };
    int result;

    g_assert_true(cxl_gpu_batch_htod_validate(
        payload, CXL_GPU_BATCH_DATA_SIZE, 3, payload_bytes, &fail_idx));
    result = cxl_gpu_batch_htod_enqueue(
        payload, 3, count_batch_enqueue, &counter, &fail_idx, &enqueued);
    g_assert_cmpint(result, ==, CXL_GPU_ERROR_OUT_OF_MEMORY);
    g_assert_cmpuint(fail_idx, ==, 2);
    g_assert_cmpuint(enqueued, ==, 2);
    g_assert_cmpuint(counter.calls, ==, 2);
    g_assert_cmpuint(counter.bytes, ==, 8);
    g_free(payload);
}

static void test_batch_success_result(void)
{
    uint64_t payload_bytes;
    uint64_t fail_idx;
    uint64_t enqueued;
    uint8_t *payload = valid_batch_payload(&payload_bytes);
    BatchEnqueueCounter counter = { .fail_at = UINT64_MAX };
    int result;

    g_assert_true(cxl_gpu_batch_htod_validate(
        payload, CXL_GPU_BATCH_DATA_SIZE, 3, payload_bytes, &fail_idx));
    result = cxl_gpu_batch_htod_enqueue(
        payload, 3, count_batch_enqueue, &counter, &fail_idx, &enqueued);
    g_assert_cmpint(result, ==, CXL_GPU_SUCCESS);
    g_assert_cmpuint(fail_idx, ==, SIZE_MAX);
    g_assert_cmpuint(enqueued, ==, 3);
    g_assert_cmpuint(counter.calls, ==, 3);
    g_assert_cmpuint(counter.bytes, ==, 15);
    g_free(payload);
}

static void test_direct_source_wire_validation(void)
{
    CXLGPUSourceRegisterV1 header = {
        .range_count = 2,
        .run_count = 2,
        .lease_handle = 7,
        .logical_bytes = 228,
        .unique_dmap_bytes = 8192,
    };
    CXLGPUSourceRangeV1 ranges[] = {
        { .first_run = 0, .run_count = 1,
          .first_run_byte_offset = 16, .length = 100 },
        { .first_run = 1, .run_count = 1,
          .first_run_byte_offset = 0, .length = 128 },
    };
    CXLGPUSourceRunV1 runs[] = {
        { .guest_phys_addr = 0x10000, .length = 4096 },
        { .guest_phys_addr = 0x20000, .length = 4096 },
    };
    uint8_t payload[sizeof(header) + sizeof(ranges) + sizeof(runs)] = { 0 };
    CXLGPUSourceRegisterV1 parsed;
    uint64_t fail_index;

    memcpy(payload, &header, sizeof(header));
    memcpy(payload + sizeof(header), ranges, sizeof(ranges));
    memcpy(payload + sizeof(header) + sizeof(ranges), runs, sizeof(runs));
    g_assert_true(cxl_gpu_source_register_validate(
        payload, sizeof(payload), sizeof(payload), &parsed, &fail_index));
    g_assert_cmpuint(parsed.logical_bytes, ==, 228);
    g_assert_cmpuint(fail_index, ==, SIZE_MAX);

    runs[0].guest_phys_addr = 0x20000;
    runs[1].guest_phys_addr = 0x10000;
    memcpy(payload + sizeof(header) + sizeof(ranges), runs, sizeof(runs));
    g_assert_true(cxl_gpu_source_register_validate(
        payload, sizeof(payload), sizeof(payload), &parsed, &fail_index));

    runs[0].guest_phys_addr = 0x10000;
    runs[1].guest_phys_addr = 0x10000;
    header.unique_dmap_bytes = 4096;
    memcpy(payload, &header, sizeof(header));
    memcpy(payload + sizeof(header) + sizeof(ranges), runs, sizeof(runs));
    g_assert_true(cxl_gpu_source_register_validate(
        payload, sizeof(payload), sizeof(payload), &parsed, &fail_index));

    runs[1].guest_phys_addr = 0x10800;
    header.unique_dmap_bytes = 8192;
    memcpy(payload, &header, sizeof(header));
    memcpy(payload + sizeof(header) + sizeof(ranges), runs, sizeof(runs));
    g_assert_false(cxl_gpu_source_register_validate(
        payload, sizeof(payload), sizeof(payload), &parsed, &fail_index));
    g_assert_cmpuint(fail_index, ==, 1);
}

static void test_direct_batch_wire_validation(void)
{
    CXLGPUDirectRangeV1 ranges[] = {
        { .destination = 0x1000, .size = 64, .source_id = 3,
          .source_range = 0, .source_offset = 16 },
        { .destination = 0x2000, .size = 128, .source_id = 4,
          .source_range = 1, .source_offset = 0 },
    };
    uint64_t fail_index;

    g_assert_true(cxl_gpu_direct_batch_validate(
        (const uint8_t *)ranges, sizeof(ranges), G_N_ELEMENTS(ranges),
        sizeof(ranges), &fail_index));
    g_assert_cmpuint(fail_index, ==, SIZE_MAX);
    ranges[0].source_id = 0;
    g_assert_true(cxl_gpu_direct_batch_validate(
        (const uint8_t *)ranges, sizeof(ranges), G_N_ELEMENTS(ranges),
        sizeof(ranges), &fail_index));
    g_assert_cmpuint(fail_index, ==, SIZE_MAX);
    ranges[0].source_id = 3;
    ranges[1].reserved0 = 1;
    g_assert_false(cxl_gpu_direct_batch_validate(
        (const uint8_t *)ranges, sizeof(ranges), G_N_ELEMENTS(ranges),
        sizeof(ranges), &fail_index));
    g_assert_cmpuint(fail_index, ==, 1);
}

static void test_direct_host_range_order_and_adjacency(void)
{
    g_assert_cmpint(cxl_gpu_direct_host_address_order(0x2000, 0x1000), >, 0);
    g_assert_cmpint(cxl_gpu_direct_host_address_order(0x1000, 0x2000), <, 0);
    g_assert_cmpint(cxl_gpu_direct_host_address_order(0x1000, 0x1000), ==, 0);

    g_assert_true(cxl_gpu_direct_host_range_follows(
        0x1000, 0x1000, 0x2000, 0x800));
    g_assert_false(cxl_gpu_direct_host_range_follows(
        0x1000, 0x1000, 0x3000, 0x800));
    g_assert_false(cxl_gpu_direct_host_range_follows(
        UINTPTR_MAX - 0x7ff, 0x1000, 0x800, 0x800));
    g_assert_false(cxl_gpu_direct_host_range_follows(
        0x1000, 0x1000, 0x2000, UINT64_MAX));
}

static void test_direct_registration_mapping_owner(void)
{
    g_assert_true(cxl_gpu_direct_registration_group_follows(
        1, 0x1000, 0x1000, 1, 0x2000, 0x1000));

    g_assert_false(cxl_gpu_direct_registration_group_follows(
        1, 0x1000, 0x1000, 2, 0x1000, 0x1000));
    g_assert_false(cxl_gpu_direct_registration_group_follows(
        1, 0x1000, 0x1000, 2, 0x1800, 0x1000));
    g_assert_false(cxl_gpu_direct_registration_group_follows(
        1, 0x1000, 0x1000, 2, 0x2000, 0x1000));
}

static void test_direct_copy_span_adjacency(void)
{
    g_assert_true(cxl_gpu_direct_copy_span_follows(
        1, 2, 0x1000, 0x4000, 0x1000,
        1, 2, 0x2000, 0x5000, 0x800));
    g_assert_false(cxl_gpu_direct_copy_span_follows(
        1, 2, 0x1000, 0x4000, 0x1000,
        1, 2, 0x2000, 0x6000, 0x800));
    g_assert_false(cxl_gpu_direct_copy_span_follows(
        1, 2, 0x1000, 0x4000, 0x1000,
        1, 3, 0x2000, 0x5000, 0x800));
    g_assert_false(cxl_gpu_direct_copy_span_follows(
        1, 2, 0x1000, 0x4000, 0x1000,
        4, 2, 0x2000, 0x5000, 0x800));
    g_assert_false(cxl_gpu_direct_copy_span_follows(
        1, 2, UINTPTR_MAX - 0x7ff, 0x4000, 0x1000,
        1, 2, 0x800, 0x5000, 0x800));
    g_assert_false(cxl_gpu_direct_copy_span_follows(
        1, 2, 0x1000, UINT64_MAX - 0x7ff, 0x1000,
        1, 2, 0x2000, 0x800, 0x800));
}

static void test_direct_registration_tile_bounds(void)
{
    const uint64_t mib = 1024 * 1024;

    g_assert_cmpuint(cxl_gpu_direct_registration_length(
                         64 * mib, 128 * mib, 80 * mib, 4 * mib,
                         192 * mib, 64 * mib, 32 * mib),
                     ==, 36 * mib);
    g_assert_cmpuint(cxl_gpu_direct_registration_length(
                         64 * mib, 128 * mib, 80 * mib, 4 * mib,
                         96 * mib, 64 * mib, 32 * mib),
                     ==, 16 * mib);
    g_assert_cmpuint(cxl_gpu_direct_registration_length(
                         64 * mib, 128 * mib, 80 * mib, 4 * mib,
                         192 * mib, 64 * mib, 8 * mib),
                     ==, 12 * mib);
    g_assert_cmpuint(cxl_gpu_direct_registration_length(
                         64 * mib, 128 * mib, 80 * mib, 4 * mib,
                         192 * mib, 0, 32 * mib),
                     ==, 4 * mib);
    g_assert_cmpuint(cxl_gpu_direct_registration_length(
                         64 * mib, 128 * mib, 60 * mib, 4 * mib,
                         192 * mib, 64 * mib, 32 * mib),
                     ==, 0);
}

static void test_direct_cross_case_epoch(void)
{
    g_assert_false(cxl_gpu_direct_epoch_is_cross_case(0, 2));
    g_assert_false(cxl_gpu_direct_epoch_is_cross_case(1, 0));
    g_assert_false(cxl_gpu_direct_epoch_is_cross_case(2, 2));
    g_assert_true(cxl_gpu_direct_epoch_is_cross_case(1, 2));
}

static void test_stream_progress_classification(void)
{
    uint64_t params[8] = { 11, 12, 13, 14, 15, 16, 17, 18 };
    uint64_t stream_wire = 0;

    g_assert_true(cxl_type2_cuda_stream_progress_wire(
        CXL_GPU_CMD_MEM_COPY_HTOD_ASYNC, params, &stream_wire));
    g_assert_cmpuint(stream_wire, ==, 13);
    g_assert_true(cxl_type2_cuda_stream_progress_wire(
        CXL_GPU_CMD_MEM_COPY_DTOD_ASYNC, params, &stream_wire));
    g_assert_cmpuint(stream_wire, ==, 14);
    g_assert_true(cxl_type2_cuda_stream_progress_wire(
        CXL_GPU_CMD_GRAPH_LAUNCH, params, &stream_wire));
    g_assert_cmpuint(stream_wire, ==, 12);
    g_assert_true(cxl_type2_cuda_stream_progress_wire(
        CXL_GPU_CMD_LAUNCH_KERNEL, params, &stream_wire));
    g_assert_cmpuint(stream_wire, ==, 17);
    g_assert_true(cxl_type2_cuda_stream_progress_wire(
        CXL_GPU_CMD_STREAM_WAIT_EVENT, params, &stream_wire));
    g_assert_cmpuint(stream_wire, ==, 11);
    g_assert_false(cxl_type2_cuda_stream_progress_wire(
        CXL_GPU_CMD_MEM_ALLOC, params, &stream_wire));
    g_assert_false(cxl_type2_cuda_stream_progress_wire(
        CXL_GPU_CMD_MEM_COPY_HTOD_ASYNC, NULL, &stream_wire));
}

static void test_stream_sync_requires_unchanged_work_generation(void)
{
    g_assert_false(cxl_type2_cuda_adjacent_stream_sync_can_elide(
        false, 7, 7));
    g_assert_true(cxl_type2_cuda_adjacent_stream_sync_can_elide(
        true, 7, 7));
    g_assert_false(cxl_type2_cuda_adjacent_stream_sync_can_elide(
        true, 8, 7));
}

static void test_stream_sync_reason_protocol_boundary(void)
{
    CXLGPUStreamSyncReason reason = CXL_GPU_STREAM_SYNC_REASON_COUNT;

    g_assert_true(cxl_type2_cuda_decode_stream_sync_reason(
        2, UINT64_MAX, &reason));
    g_assert_cmpint(reason, ==, CXL_GPU_STREAM_SYNC_PUBLIC_API);
    for (uint64_t wire = 0; wire < CXL_GPU_STREAM_SYNC_REASON_COUNT; wire++) {
        g_assert_true(cxl_type2_cuda_decode_stream_sync_reason(3, wire,
                                                               &reason));
        g_assert_cmpint(reason, ==, wire);
    }
    g_assert_false(cxl_type2_cuda_decode_stream_sync_reason(
        3, CXL_GPU_STREAM_SYNC_REASON_COUNT, &reason));
    g_assert_false(cxl_type2_cuda_decode_stream_sync_reason(3, 0, NULL));
}

static void test_per_thread_stream_uses_stable_qemu_handle(void)
{
    void *const qemu_stream = (void *)(uintptr_t)0x1234;
    void *stream = NULL;

    g_assert_true(cxl_type2_cuda_special_stream_from_wire(
        CXL_GPU_STREAM_WIRE_PER_THREAD, qemu_stream, &stream));
    g_assert_true(stream == qemu_stream);
    g_assert_true(stream != (void *)(uintptr_t)2);

    stream = NULL;
    g_assert_true(cxl_type2_cuda_special_stream_from_wire(
        CXL_GPU_STREAM_WIRE_PER_THREAD, qemu_stream, &stream));
    g_assert_true(stream == qemu_stream);
    g_assert_false(cxl_type2_cuda_special_stream_from_wire(
        CXL_GPU_STREAM_WIRE_PER_THREAD, NULL, &stream));
}

int main(int argc, char **argv)
{
    g_test_init(&argc, &argv, NULL);
    g_test_add_func("/cxl/type2/cuda-contract/decode-attribute",
                    test_decode_attribute);
    g_test_add_func("/cxl/type2/cuda-contract/mem-info-gate",
                    test_mem_info_gate);
    g_test_add_func("/cxl/type2/cuda-contract/attribute-dispatch",
                    test_attribute_dispatch_rejects_before_query);
    g_test_add_func("/cxl/type2/cuda-contract/mem-info-dispatch",
                    test_mem_info_dispatch_rejects_before_query);
    g_test_add_func("/cxl/type2/descriptor/accept-and-duplicate",
                    test_descriptor_accept_and_duplicate);
    g_test_add_func("/cxl/type2/descriptor/wire-errors",
                    test_descriptor_rejects_wire_errors);
    g_test_add_func("/cxl/type2/descriptor/identity-errors",
                    test_descriptor_rejects_identity_errors);
    g_test_add_func("/cxl/type2/descriptor/case-epoch",
                    test_descriptor_case_epoch_gate);
    g_test_add_func("/cxl/type2/batch/wire-validation",
                    test_batch_wire_layout_and_validation);
    g_test_add_func("/cxl/type2/batch/header-zero-enqueue",
                    test_batch_rejects_header_without_enqueue);
    g_test_add_func("/cxl/type2/batch/partial-enqueue",
                    test_batch_partial_enqueue_result);
    g_test_add_func("/cxl/type2/batch/success",
                    test_batch_success_result);
    g_test_add_func("/cxl/type2/direct/source-wire",
                    test_direct_source_wire_validation);
    g_test_add_func("/cxl/type2/direct/batch-wire",
                    test_direct_batch_wire_validation);
    g_test_add_func("/cxl/type2/direct/host-range-layout",
                    test_direct_host_range_order_and_adjacency);
    g_test_add_func("/cxl/type2/direct/registration-mapping-owner",
                    test_direct_registration_mapping_owner);
    g_test_add_func("/cxl/type2/direct/copy-span-layout",
                    test_direct_copy_span_adjacency);
    g_test_add_func("/cxl/type2/direct/registration-tile-bounds",
                    test_direct_registration_tile_bounds);
    g_test_add_func("/cxl/type2/direct/cross-case-epoch",
                    test_direct_cross_case_epoch);
    g_test_add_func("/cxl/type2/stream/progress-classification",
                    test_stream_progress_classification);
    g_test_add_func("/cxl/type2/stream/sync-generation",
                    test_stream_sync_requires_unchanged_work_generation);
    g_test_add_func("/cxl/type2/stream/sync-reason-protocol",
                    test_stream_sync_reason_protocol_boundary);
    g_test_add_func("/cxl/type2/stream/per-thread-stable-handle",
                    test_per_thread_stream_uses_stable_qemu_handle);
    return g_test_run();
}
