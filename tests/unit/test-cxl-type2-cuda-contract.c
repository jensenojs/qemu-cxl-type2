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

typedef struct CoherentUnmapCounter {
    unsigned int synchronize_calls;
    unsigned int htod_calls_queries;
    unsigned int unregister_calls;
    int synchronize_result;
    uint64_t htod_calls;
    int unregister_result;
} CoherentUnmapCounter;

static int coherent_unmap_synchronize(void *opaque)
{
    CoherentUnmapCounter *counter = opaque;

    counter->synchronize_calls++;
    return counter->synchronize_result;
}

static uint64_t coherent_unmap_htod_calls(void *opaque)
{
    CoherentUnmapCounter *counter = opaque;

    counter->htod_calls_queries++;
    return counter->htod_calls;
}

static int coherent_unmap_unregister(void *opaque)
{
    CoherentUnmapCounter *counter = opaque;

    counter->unregister_calls++;
    return counter->unregister_result;
}

static const CXLType2CoherentUnmapOps coherent_unmap_ops = {
    .synchronize = coherent_unmap_synchronize,
    .htod_calls = coherent_unmap_htod_calls,
    .unregister_host = coherent_unmap_unregister,
};

static int run_coherent_unmap(CoherentUnmapCounter *counter,
                              bool host_registered, uint64_t stored_alias,
                              uint64_t requested_alias, uint64_t calls_at_map,
                              uint64_t *htod_delta, bool *mapping_invalidated)
{
    return cxl_type2_coherent_unmap_execute(
        host_registered, stored_alias, requested_alias, calls_at_map,
        &coherent_unmap_ops, counter, htod_delta, mapping_invalidated);
}

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

    g_assert_cmphex(CXL_GPU_VERSION, ==, UINT64_C(0x00011400));
    g_assert_cmphex(CXL_GPU_CMD_SOURCE_REGISTER_BATCH_HTOD_DIRECT_ASYNC,
                    ==, 0x35);
    g_assert_cmphex(CXL_GPU_CMD_MEM_COPY_DTOD_ASYNC, ==, 0x2f);
    g_assert_cmphex(CXL_GPU_CMD_COHERENT_MAP_DEVICE, ==, 0xa7);
    g_assert_cmphex(CXL_GPU_CMD_COHERENT_UNMAP_DEVICE, ==, 0xa8);
    g_assert_cmpuint(CXL_GPU_DESCRIPTOR_PROTOCOL_VERSION, ==, 4);
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
        .logical_bytes = 228,
    };
    CXLGPUSourceVirtualRangeV1 ranges[] = {
        { .guest_virtual_address = 0x10010, .length = 100 },
        { .guest_virtual_address = 0x20000, .length = 128 },
    };
    uint8_t payload[sizeof(header) + sizeof(ranges)] = { 0 };
    CXLGPUSourceRegisterV1 parsed;
    uint64_t fail_index;

    memcpy(payload, &header, sizeof(header));
    memcpy(payload + sizeof(header), ranges, sizeof(ranges));
    g_assert_true(cxl_gpu_source_register_validate(
        payload, sizeof(payload), sizeof(payload), &parsed, &fail_index));
    g_assert_cmpuint(parsed.logical_bytes, ==, 228);
    g_assert_cmpuint(fail_index, ==, SIZE_MAX);

    ranges[1].guest_virtual_address = UINT64_MAX - 63;
    memcpy(payload + sizeof(header), ranges, sizeof(ranges));
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

static void test_direct_batch_destinations_are_independent(void)
{
    const uint64_t destinations[] = { 0x1000, 0x3000, 0x2000 };
    const size_t sizes[] = { 0x1000, 0x800, 0x1000 };
    const uint64_t overlapping[] = { 0x1000, 0x1800 };
    const size_t overlapping_sizes[] = { 0x1000, 0x1000 };
    const uint64_t overflowing[] = { UINT64_MAX - 7 };
    const size_t overflowing_sizes[] = { 8 };
    size_t conflict = 0;

    g_assert_true(cxl_gpu_direct_destinations_are_independent(
        destinations, sizes, G_N_ELEMENTS(destinations), &conflict));
    g_assert_cmpuint(conflict, ==, SIZE_MAX);
    g_assert_false(cxl_gpu_direct_destinations_are_independent(
        overlapping, overlapping_sizes, G_N_ELEMENTS(overlapping),
        &conflict));
    g_assert_cmpuint(conflict, ==, 1);
    g_assert_false(cxl_gpu_direct_destinations_are_independent(
        overflowing, overflowing_sizes, G_N_ELEMENTS(overflowing),
        &conflict));
    g_assert_cmpuint(conflict, ==, 0);
}

static void assert_cuda_coverage(
    const CXLType2CudaAllocationTable *table,
    const uint64_t *destinations, const size_t *sizes, size_t count,
    CXLType2CudaCoverageKind kind, CXLType2CudaRejectionReason reason,
    uint64_t bytes)
{
    CXLType2CudaCoverageResult result;

    cxl_type2_cuda_destination_union_classify(
        table, destinations, sizes, count, &result);
    g_assert_true(result.available);
    g_assert_cmpint(result.kind, ==, kind);
    g_assert_cmpint(result.reason, ==, reason);
    g_assert_cmpuint(result.bytes, ==, bytes);
}

static void test_cuda_allocation_lifecycle_and_epoch(void)
{
    CXLType2CudaAllocationTable table;
    uint64_t first_epoch = 0;
    uint64_t second_epoch = 0;
    uint64_t reused_epoch = 0;

    cxl_type2_cuda_allocation_table_init(&table);
    g_assert_true(cxl_type2_cuda_allocation_record(
        &table, 0x3000, 0x1000, &first_epoch));
    g_assert_true(cxl_type2_cuda_allocation_record(
        &table, 0x1000, 0x1000, &second_epoch));
    g_assert_cmpuint(first_epoch, ==, 1);
    g_assert_cmpuint(second_epoch, ==, 2);
    g_assert_cmpuint(table.count, ==, 2);
    g_assert_cmpuint(table.peak_count, ==, 2);
    g_assert_cmphex(table.entries[0].base, ==, 0x1000);
    g_assert_cmphex(table.entries[1].base, ==, 0x3000);

    g_assert_true(cxl_type2_cuda_allocation_forget(&table, 0x1000));
    g_assert_true(cxl_type2_cuda_allocation_record(
        &table, 0x1000, 0x1000, &reused_epoch));
    g_assert_cmpuint(reused_epoch, >, second_epoch);
    g_assert_cmpuint(table.count, ==, 2);
    g_assert_cmpuint(table.peak_count, ==, 2);
    g_assert_cmpuint(table.next_epoch, ==, reused_epoch + 1);
    cxl_type2_cuda_allocation_table_destroy(&table);
}

static void test_cuda_allocation_rejects_invalid_state(void)
{
    CXLType2CudaAllocationTable table;

    cxl_type2_cuda_allocation_table_init(&table);
    g_assert_true(cxl_type2_cuda_allocation_record(
        &table, 0x1000, 0x1000, NULL));
    g_assert_false(cxl_type2_cuda_allocation_record(
        &table, 0x1800, 0x1000, NULL));
    g_assert_false(table.available);

    cxl_type2_cuda_allocation_table_reset(&table);
    g_assert_true(cxl_type2_cuda_allocation_record(
        &table, 0x1000, 0x1000, NULL));
    g_assert_false(cxl_type2_cuda_allocation_forget(&table, 0x1800));
    g_assert_false(table.available);

    cxl_type2_cuda_allocation_table_reset(&table);
    g_assert_false(cxl_type2_cuda_allocation_record(
        &table, UINT64_MAX - 7, 8, NULL));
    g_assert_false(table.available);
    cxl_type2_cuda_allocation_table_destroy(&table);
}

static void test_cuda_allocation_alias_lifecycle(void)
{
    CXLType2CudaAllocationTable table;
    uint64_t epoch = 0;
    uint64_t alias = 0;

    cxl_type2_cuda_allocation_table_init(&table);
    g_assert_true(cxl_type2_cuda_allocation_record(
        &table, 0x1000, 0x1000, &epoch));
    g_assert_true(cxl_type2_cuda_allocation_publish_alias(
        &table, 0x1000, epoch, 7, 0x9000));
    g_assert_false(cxl_type2_cuda_allocation_publish_alias(
        &table, 0x1000, epoch, 8, 0xa000));
    g_assert_false(cxl_type2_cuda_allocation_acquire_alias(
        &table, 0x1000, epoch, 6, &alias));
    g_assert_true(cxl_type2_cuda_allocation_acquire_alias(
        &table, 0x1000, epoch, 7, &alias));
    g_assert_cmphex(alias, ==, 0x9000);
    g_assert_false(cxl_type2_cuda_allocation_forget(&table, 0x1000));
    g_assert_false(cxl_type2_cuda_allocation_materialize(
        &table, 0x1000, epoch, 7));
    g_assert_true(cxl_type2_cuda_allocation_release_alias(
        &table, 0x1000, epoch, 7));
    g_assert_true(cxl_type2_cuda_allocation_materialize(
        &table, 0x1000, epoch, 7));
    g_assert_false(cxl_type2_cuda_allocation_acquire_alias(
        &table, 0x1000, epoch, 7, &alias));
    g_assert_true(cxl_type2_cuda_allocation_forget(&table, 0x1000));
    g_assert_false(cxl_type2_cuda_allocation_publish_alias(
        &table, 0x1000, epoch, 9, 0xb000));
    cxl_type2_cuda_allocation_table_destroy(&table);
}

static CXLType2CudaAliasSource alias_source(int fd,
                                            const struct stat *source_stat,
                                            uint64_t file_offset,
                                            uint64_t destination_offset,
                                            uint64_t length,
                                            uint64_t generation)
{
    return (CXLType2CudaAliasSource) {
        .fd = fd,
        .file_offset = file_offset,
        .destination_offset = destination_offset,
        .length = length,
        .mapping_generation = generation,
        .logical_cxl_offset = UINT64_C(0x10000000) + destination_offset,
        .stat_device = source_stat->st_dev,
        .stat_inode = source_stat->st_ino,
        .stat_size = source_stat->st_size,
        .stat_mode = source_stat->st_mode,
        .readonly = true,
    };
}

static void test_cuda_pageable_alias_composite_remap(void)
{
    CXLType2CudaAllocationTable table;
    CXLType2CudaAliasSource sources[4];
    g_autofree char *path = NULL;
    g_autofree uint8_t *file_bytes = NULL;
    uint8_t host_bytes[128];
    GError *error = NULL;
    struct stat source_stat;
    size_t page_size = qemu_real_host_page_size();
    uint64_t logical_bytes = 3 * page_size - 32;
    uint64_t epoch = 0;
    uint64_t alias_one = 0;
    uint64_t alias_two = 0;
    const char *reason = NULL;
    int fd;
    pid_t child;
    int status;
    uint64_t source_call_one[] = { 1, 3 };
    uint64_t source_call_two[] = { 2 };
    uint64_t invalid_source_calls[] = { 4, 2 };

    fd = g_file_open_tmp("cxl-pageable-alias-XXXXXX", &path, &error);
    g_assert_no_error(error);
    g_assert_cmpint(fd, >=, 0);
    g_assert_cmpint(ftruncate(fd, 5 * page_size), ==, 0);
    file_bytes = g_malloc(5 * page_size);
    for (size_t i = 0; i < 5 * page_size; i++) {
        file_bytes[i] = (i * 17 + 3) & 0xff;
    }
    for (size_t i = 0; i < sizeof(host_bytes); i++) {
        host_bytes[i] = (i * 29 + 7) & 0xff;
    }
    g_assert_cmpint(pwrite(fd, file_bytes, 5 * page_size, 0), ==,
                    5 * page_size);
    g_assert_cmpint(fstat(fd, &source_stat), ==, 0);

    sources[0] = alias_source(fd, &source_stat, 32, 0,
                              page_size - 64, 1);
    sources[1] = alias_source(fd, &source_stat, 2 * page_size + 17,
                              page_size - 64, 128, 1);
    sources[1].fd = -1;
    sources[1].host_copy_source = host_bytes;
    sources[2] = alias_source(fd, &source_stat, 3 * page_size + 64,
                              page_size + 64, page_size - 64, 1);
    sources[3] = alias_source(fd, &source_stat, 4 * page_size,
                              2 * page_size, page_size, 1);

    cxl_type2_cuda_allocation_table_init(&table);
    g_assert_true(cxl_type2_cuda_allocation_record(
        &table, UINT64_C(0x100000), 3 * page_size, &epoch));
    g_assert_true(cxl_type2_cuda_allocation_map_pageable_alias(
        &table, UINT64_C(0x100000), epoch, 1, sources,
        G_N_ELEMENTS(sources), source_call_one, G_N_ELEMENTS(source_call_one),
        0, logical_bytes, 32, &alias_one, &reason));
    g_assert_null(reason);
    g_assert_cmpuint(
        table.entries[0].pageable_alias->contributing_source_call_count,
        ==, 2);
    g_assert_cmpuint(
        table.entries[0].pageable_alias->contributing_source_call_ids[0],
        ==, 1);
    g_assert_cmpuint(
        table.entries[0].pageable_alias->contributing_source_call_ids[1],
        ==, 3);
    g_assert_false(cxl_type2_cuda_allocation_map_pageable_alias(
        &table, UINT64_C(0x100000), epoch, 1, sources,
        G_N_ELEMENTS(sources), invalid_source_calls,
        G_N_ELEMENTS(invalid_source_calls), 0, logical_bytes, 32,
        &alias_two, &reason));
    g_assert_cmpstr(reason, ==, "alias-source-call-order-invalid");
    reason = NULL;
    g_assert_cmpuint(table.entries[0].pageable_alias->file_mapped_bytes,
                     ==, page_size);
    g_assert_cmpuint(
        table.entries[0].pageable_alias->derived_boundary_pages, ==, 2);
    g_assert_cmpuint(
        table.entries[0].pageable_alias->derived_boundary_copy_bytes,
        ==, 2 * page_size - sizeof(host_bytes));
    g_assert_cmpuint(
        table.entries[0].pageable_alias->host_composition_copy_bytes,
        ==, sizeof(host_bytes));
    g_assert_cmpmem((const uint8_t *)(uintptr_t)alias_one + page_size - 64,
                    sizeof(host_bytes), host_bytes, sizeof(host_bytes));
    g_assert_nonnull(table.entries[0].pageable_alias->owned_reservation);
    g_assert_cmpuint(
        table.entries[0].pageable_alias->owned_reservation_size,
        ==, 5 * page_size);
    g_assert_cmpuint(
        table.entries[0].pageable_alias->boundary_composition_wall_ns,
        >, 0);

    child = fork();
    g_assert_cmpint(child, >=, 0);
    if (child == 0) {
        /* Volatile forces the readonly mapping to receive a real store. */
        *(volatile uint8_t *)(uintptr_t)alias_one ^= 1;
        _exit(0);
    }
    g_assert_cmpint(waitpid(child, &status, 0), ==, child);
    g_assert_true(WIFSIGNALED(status));
    g_assert_true(WTERMSIG(status) == SIGSEGV || WTERMSIG(status) == SIGBUS);

    child = fork();
    g_assert_cmpint(child, >=, 0);
    if (child == 0) {
        uint8_t *sentinel =
            table.entries[0].pageable_alias->owned_reservation;

        *sentinel = 1;
        _exit(0);
    }
    g_assert_cmpint(waitpid(child, &status, 0), ==, child);
    g_assert_true(WIFSIGNALED(status));
    g_assert_true(WTERMSIG(status) == SIGSEGV || WTERMSIG(status) == SIGBUS);

    g_assert_true(cxl_type2_cuda_allocation_acquire_pageable_alias(
        &table, UINT64_C(0x100000), epoch, 1,
        CXL_TYPE2_CUDA_ALIAS_NORMAL, &alias_two));
    g_assert_cmphex(alias_two, ==, alias_one);
    g_assert_false(cxl_type2_cuda_allocation_map_pageable_alias(
        &table, UINT64_C(0x100000), epoch, 2, sources,
        G_N_ELEMENTS(sources), source_call_two, G_N_ELEMENTS(source_call_two),
        0, logical_bytes, 32, &alias_two, &reason));
    g_assert_cmpstr(reason, ==, "alias-consumer-in-flight");
    g_assert_true(cxl_type2_cuda_allocation_release_pageable_alias(
        &table, UINT64_C(0x100000), epoch, 1,
        CXL_TYPE2_CUDA_ALIAS_NORMAL));

    g_assert_true(cxl_type2_cuda_allocation_bind_graph_alias(
        &table, UINT64_C(0x100000), epoch, alias_one));
    g_assert_true(cxl_type2_cuda_allocation_acquire_pageable_alias(
        &table, UINT64_C(0x100000), epoch, 1,
        CXL_TYPE2_CUDA_ALIAS_GRAPH, &alias_two));
    g_assert_true(cxl_type2_cuda_allocation_release_pageable_alias(
        &table, UINT64_C(0x100000), epoch, 1,
        CXL_TYPE2_CUDA_ALIAS_GRAPH));

    for (size_t i = 0; i < 5 * page_size; i++) {
        file_bytes[i] ^= 0xa5;
    }
    g_assert_cmpint(pwrite(fd, file_bytes, 5 * page_size, 0), ==,
                    5 * page_size);
    for (size_t i = 0; i < G_N_ELEMENTS(sources); i++) {
        sources[i].mapping_generation = 2;
    }
    reason = NULL;
    g_assert_true(cxl_type2_cuda_allocation_map_pageable_alias(
        &table, UINT64_C(0x100000), epoch, 2, sources,
        G_N_ELEMENTS(sources), source_call_two, G_N_ELEMENTS(source_call_two),
        0, logical_bytes, 32, &alias_two, &reason));
    g_assert_null(reason);
    g_assert_cmphex(alias_two, ==, alias_one);
    g_assert_false(cxl_type2_cuda_allocation_drop_pageable_alias(
        &table, UINT64_C(0x100000), epoch, 2));
    g_assert_true(cxl_type2_cuda_allocation_unbind_graph_alias(
        &table, UINT64_C(0x100000), epoch, alias_two));
    g_assert_true(cxl_type2_cuda_allocation_drop_pageable_alias(
        &table, UINT64_C(0x100000), epoch, 2));
    g_assert_true(cxl_type2_cuda_allocation_forget(
        &table, UINT64_C(0x100000)));
    cxl_type2_cuda_allocation_table_destroy(&table);
    close(fd);
    g_assert_cmpint(unlink(path), ==, 0);
}

static void test_cuda_pageable_alias_appends_allocation_subranges(void)
{
    CXLType2CudaAllocationTable table;
    CXLType2CudaAliasSource source;
    CXLType2CudaAllocationIdentity identity;
    g_autofree char *path = NULL;
    g_autofree uint8_t *file_bytes = NULL;
    GError *error = NULL;
    struct stat source_stat;
    size_t page_size = qemu_real_host_page_size();
    uint64_t base = UINT64_C(0x200000);
    uint64_t epoch = 0;
    uint64_t alias = 0;
    uint64_t stable_alias = 0;
    uint64_t generation = 0;
    uint64_t source_call = 1;
    const char *reason = NULL;
    int fd;

    fd = g_file_open_tmp("cxl-pageable-subranges-XXXXXX", &path, &error);
    g_assert_no_error(error);
    g_assert_cmpint(fd, >=, 0);
    g_assert_cmpint(ftruncate(fd, 4 * page_size), ==, 0);
    file_bytes = g_malloc(4 * page_size);
    for (size_t i = 0; i < 4 * page_size; i++) {
        file_bytes[i] = (i * 11 + 5) & 0xff;
    }
    g_assert_cmpint(pwrite(fd, file_bytes, 4 * page_size, 0), ==,
                    4 * page_size);
    g_assert_cmpint(fstat(fd, &source_stat), ==, 0);

    cxl_type2_cuda_allocation_table_init(&table);
    g_assert_true(cxl_type2_cuda_allocation_record(
        &table, base, 4 * page_size, &epoch));

    source = alias_source(fd, &source_stat, 0, 0, page_size, 1);
    g_assert_true(cxl_type2_cuda_allocation_map_pageable_alias(
        &table, base, epoch, 1, &source, 1, &source_call, 1,
        0, page_size, 0, &stable_alias, &reason));
    g_assert_null(reason);

    source = alias_source(fd, &source_stat, page_size, 0,
                          2 * page_size, 2);
    source_call = 2;
    g_assert_true(cxl_type2_cuda_allocation_map_pageable_alias(
        &table, base, epoch, 2, &source, 1, &source_call, 1,
        page_size, 2 * page_size, 0, &alias, &reason));
    g_assert_null(reason);
    g_assert_cmphex(alias, ==, stable_alias);
    g_assert_cmpmem((const uint8_t *)(uintptr_t)alias, 3 * page_size,
                    file_bytes, 3 * page_size);
    g_assert_true(cxl_type2_cuda_pageable_alias_contains(
        table.entries[0].pageable_alias, 0));
    g_assert_true(cxl_type2_cuda_pageable_alias_contains(
        table.entries[0].pageable_alias, 3 * page_size - 1));
    g_assert_false(cxl_type2_cuda_pageable_alias_contains(
        table.entries[0].pageable_alias, 3 * page_size));
    source = alias_source(fd, &source_stat, 0, 0, page_size, 3);
    source_call = 3;
    g_assert_false(cxl_type2_cuda_allocation_map_pageable_alias(
        &table, base, epoch, 3, &source, 1, &source_call, 1,
        page_size / 2, page_size, 0, &alias, &reason));
    g_assert_cmpstr(reason, ==, "alias-subrange-overlap");
    g_assert_cmpmem((const uint8_t *)(uintptr_t)stable_alias, 3 * page_size,
                    file_bytes, 3 * page_size);
    g_assert_true(cxl_type2_cuda_allocation_acquire_pageable_alias_for_address(
        &table, base + page_size, CXL_TYPE2_CUDA_ALIAS_NORMAL,
        &identity, &generation, &alias));
    g_assert_cmphex(alias, ==, stable_alias + page_size);
    g_assert_cmpuint(generation, ==, 2);
    g_assert_true(cxl_type2_cuda_allocation_release_pageable_alias(
        &table, base, epoch, generation, CXL_TYPE2_CUDA_ALIAS_NORMAL));
    g_assert_true(cxl_type2_cuda_allocation_drop_pageable_alias(
        &table, base, epoch, generation));
    g_assert_true(cxl_type2_cuda_allocation_forget(&table, base));
    cxl_type2_cuda_allocation_table_destroy(&table);
    close(fd);
    g_assert_cmpint(unlink(path), ==, 0);
}

static void test_cuda_generation_reuse_lifecycle(void)
{
    CXLType2CudaAllocationTable table;
    CXLType2CudaAllocationIdentity duplicate_refs[2];
    uint64_t epoch = 0;

    cxl_type2_cuda_allocation_table_init(&table);
    g_assert_true(cxl_type2_cuda_allocation_record(
        &table, 0x1000, 0x1000, &epoch));
    duplicate_refs[0] = (CXLType2CudaAllocationIdentity) {
        .base = 0x1000,
        .epoch = epoch,
    };
    duplicate_refs[1] = duplicate_refs[0];

    g_assert_true(cxl_type2_cuda_generation_population_complete(
        &table, 0x1000, 0x800));
    g_assert_true(cxl_type2_cuda_generation_population_complete(
        &table, 0x1400, 0x800));
    g_assert_cmpuint(table.generation_count, ==, 1);
    g_assert_cmpuint(table.generations[0].unique_visible_population_bytes,
                     ==, 0xc00);
    g_assert_true(cxl_type2_cuda_generation_consume(
        &table, duplicate_refs, G_N_ELEMENTS(duplicate_refs),
        CXL_GPU_CMD_LAUNCH_KERNEL, 17));
    g_assert_cmpuint(table.generations[0].gpu_local_consumer_count, ==, 1);
    g_assert_cmpuint(table.generations[0].first_consumer_call_id, ==, 17);

    g_assert_true(cxl_type2_cuda_generation_population_complete(
        &table, 0x1800, 0x400));
    g_assert_cmpuint(table.generation_count, ==, 2);
    g_assert_cmpint(table.generations[0].next_boundary, ==,
                    CXL_TYPE2_CUDA_GENERATION_POPULATION);
    g_assert_cmpuint(table.generations[1].generation, ==, 2);
    g_assert_cmpuint(table.generations[1].unique_visible_population_bytes,
                     ==, 0x400);
    g_assert_true(cxl_type2_cuda_generation_prefetch_enqueue(
        &table, 0x1800, 0x200, 31));
    g_assert_cmpuint(table.generations[1].prefetch_count, ==, 1);
    g_assert_false(table.generations[1].prefetch_completion_available);

    g_assert_true(cxl_type2_cuda_allocation_forget(&table, 0x1000));
    g_assert_cmpint(table.generations[1].next_boundary, ==,
                    CXL_TYPE2_CUDA_GENERATION_RELEASE);
    g_assert_cmpstr(cxl_type2_cuda_generation_boundary_name(
                        table.generations[1].next_boundary),
                    ==, "release");
    g_assert_false(cxl_type2_cuda_generation_population_complete(
        &table, 0x1800, 0x1000));
    g_assert_false(table.available);
    g_assert_cmpstr(table.first_generation_error, ==,
                    "population-allocation-missing");
    cxl_type2_cuda_allocation_table_destroy(&table);
}

static void test_cuda_destination_union_coverage(void)
{
    CXLType2CudaAllocationTable table;
    const uint64_t whole_destinations[] = { 0x1800, 0x1000 };
    const size_t whole_sizes[] = { 0x800, 0x800 };
    const uint64_t partial_destinations[] = { 0x1000 };
    const size_t partial_sizes[] = { 0x800 };
    const uint64_t gap_destinations[] = { 0x1000, 0x1800 };
    const size_t gap_sizes[] = { 0x400, 0x800 };
    const uint64_t overlap_destinations[] = { 0x1000, 0x1800 };
    const size_t overlap_sizes[] = { 0x1000, 0x800 };
    const uint64_t cross_destinations[] = { 0x3800 };
    const size_t cross_sizes[] = { 0x1000 };
    const uint64_t separate_allocations[] = { 0x1000, 0x3000 };
    const size_t separate_allocation_sizes[] = { 0x1000, 0x1000 };
    const uint64_t overflow_destinations[] = { UINT64_MAX - 7 };
    const size_t overflow_sizes[] = { 8 };
    const uint64_t missing_destinations[] = { 0x6000 };
    const size_t missing_sizes[] = { 0x100 };

    cxl_type2_cuda_allocation_table_init(&table);
    g_assert_true(cxl_type2_cuda_allocation_record(
        &table, 0x1000, 0x1000, NULL));
    g_assert_true(cxl_type2_cuda_allocation_record(
        &table, 0x3000, 0x1000, NULL));
    g_assert_true(cxl_type2_cuda_allocation_record(
        &table, 0x4000, 0x1000, NULL));

    assert_cuda_coverage(
        &table, whole_destinations, whole_sizes,
        G_N_ELEMENTS(whole_destinations), CXL_TYPE2_CUDA_COVERAGE_WHOLE,
        CXL_TYPE2_CUDA_REJECTION_NONE, 0x1000);
    assert_cuda_coverage(
        &table, partial_destinations, partial_sizes,
        G_N_ELEMENTS(partial_destinations), CXL_TYPE2_CUDA_COVERAGE_PARTIAL,
        CXL_TYPE2_CUDA_REJECTION_PARTIAL_COVERAGE, 0x800);
    assert_cuda_coverage(
        &table, gap_destinations, gap_sizes,
        G_N_ELEMENTS(gap_destinations), CXL_TYPE2_CUDA_COVERAGE_PARTIAL,
        CXL_TYPE2_CUDA_REJECTION_PARTIAL_COVERAGE, 0xc00);
    assert_cuda_coverage(
        &table, overlap_destinations, overlap_sizes,
        G_N_ELEMENTS(overlap_destinations), CXL_TYPE2_CUDA_COVERAGE_UNKNOWN,
        CXL_TYPE2_CUDA_REJECTION_DESTINATION_OVERLAP, 0x1800);
    assert_cuda_coverage(
        &table, cross_destinations, cross_sizes,
        G_N_ELEMENTS(cross_destinations), CXL_TYPE2_CUDA_COVERAGE_CROSS,
        CXL_TYPE2_CUDA_REJECTION_CROSS_ALLOCATION, 0x1000);
    assert_cuda_coverage(
        &table, separate_allocations, separate_allocation_sizes,
        G_N_ELEMENTS(separate_allocations), CXL_TYPE2_CUDA_COVERAGE_CROSS,
        CXL_TYPE2_CUDA_REJECTION_CROSS_ALLOCATION, 0x2000);
    assert_cuda_coverage(
        &table, overflow_destinations, overflow_sizes,
        G_N_ELEMENTS(overflow_destinations), CXL_TYPE2_CUDA_COVERAGE_UNKNOWN,
        CXL_TYPE2_CUDA_REJECTION_DESTINATION_OVERFLOW, 0);
    assert_cuda_coverage(
        &table, missing_destinations, missing_sizes,
        G_N_ELEMENTS(missing_destinations), CXL_TYPE2_CUDA_COVERAGE_UNKNOWN,
        CXL_TYPE2_CUDA_REJECTION_ALLOCATION_MISSING, 0x100);
    cxl_type2_cuda_allocation_table_destroy(&table);
}

static void test_cuda_command_roles_use_generated_authority(void)
{
    CXLType2CudaCommandRole role = cxl_type2_cuda_command_role(
        CXL_GPU_CMD_MEM_COPY_2D_DTOD);

    g_assert_true(role & CXL_TYPE2_CUDA_COMMAND_READER);
    g_assert_true(role & CXL_TYPE2_CUDA_COMMAND_WRITER);
    g_assert_cmpstr(cxl_type2_cuda_command_role_name(role), ==,
                    "reader+writer");
    g_assert_cmpint(cxl_type2_cuda_command_role(CXL_GPU_CMD_MEM_ALLOC), ==,
                    CXL_TYPE2_CUDA_COMMAND_LIFECYCLE);
    g_assert_cmpstr(cxl_type2_cuda_command_role_name(
                        cxl_type2_cuda_command_role(0xff)),
                    ==, "unknown");
}

static void test_cuda_opcode_summary_wire_and_conservation(void)
{
    uint64_t command_counts[256] = { 0 };
    char records[CXL_TYPE2_CUDA_OPCODE_RECORDS_CAPACITY];
    CXLType2CudaOpcodeSummary summary;

    command_counts[CXL_GPU_CMD_MEM_ALLOC] = 2;
    command_counts[CXL_GPU_CMD_MEM_COPY_DTOH] = 4;
    command_counts[CXL_GPU_CMD_MEM_COPY_DTOD] = 3;
    command_counts[CXL_GPU_CMD_LAUNCH_KERNEL] = 1;
    command_counts[CXL_GPU_CMD_STREAM_SYNC] = 5;
    g_assert_true(cxl_type2_cuda_opcode_summary_build(
        command_counts, records, sizeof(records), &summary));
    g_assert_cmpstr(
        records, ==,
        "0x20:lifecycle:2:0:incomplete;"
        "0x23:reader:4:0:incomplete;"
        "0x24:reader+writer:3:0:incomplete;"
        "0x40:unknown:1:0:incomplete;"
        "0x52:lifecycle:5:0:incomplete");
    g_assert_cmpuint(summary.reader_commands, ==, 7);
    g_assert_cmpuint(summary.writer_commands, ==, 3);
    g_assert_cmpuint(summary.lifecycle_commands, ==, 7);
    g_assert_cmpuint(summary.no_change_commands, ==, 0);
    g_assert_cmpuint(summary.unknown_commands, ==, 1);
    g_assert_cmpuint(summary.estimated_materialize_bytes, ==, 0);
    g_assert_false(summary.estimated_materialize_complete);
    g_assert_cmphex(summary.first_incomplete_command, ==,
                    CXL_GPU_CMD_MEM_ALLOC);
}

static void test_cuda_opcode_summary_empty_and_bounded(void)
{
    uint64_t command_counts[256] = { 0 };
    char records[CXL_TYPE2_CUDA_OPCODE_RECORDS_CAPACITY];
    CXLType2CudaOpcodeSummary summary;

    g_assert_true(cxl_type2_cuda_opcode_summary_build(
        command_counts, records, sizeof(records), &summary));
    g_assert_cmpstr(records, ==, "none");
    g_assert_true(summary.estimated_materialize_complete);
    command_counts[CXL_GPU_CMD_MEM_ALLOC] = 1;
    g_assert_false(cxl_type2_cuda_opcode_summary_build(
        command_counts, records, sizeof("none"), &summary));
    g_assert_cmpstr(records, ==, "none");

    memset(command_counts, 0, sizeof(command_counts));
    command_counts[CXL_GPU_CMD_MEM_COPY_DTOH] = UINT64_MAX;
    command_counts[CXL_GPU_CMD_MEM_COPY_DTOD] = 1;
    g_assert_false(cxl_type2_cuda_opcode_summary_build(
        command_counts, records, sizeof(records), &summary));
    g_assert_cmpuint(summary.reader_commands, ==, UINT64_MAX);
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

static void test_direct_registration_page_span(void)
{
    uint64_t offset = 0;
    uint64_t length = 0;

    g_assert_true(cxl_gpu_direct_page_registration_span(
        0x1000, 0x4000, 0x1800, 0x1001, 0x5000, 0x1000, &offset,
        &length));
    g_assert_cmphex(offset, ==, 0x1000);
    g_assert_cmphex(length, ==, 0x2000);

    g_assert_false(cxl_gpu_direct_page_registration_span(
        0x1000, 0x4000, 0x1800, 0x1001, 0x2801, 0x1000, &offset,
        &length));
    g_assert_false(cxl_gpu_direct_page_registration_span(
        0x1800, 0x3800, 0x1800, 0x100, 0x5000, 0x1000, &offset,
        &length));
    g_assert_false(cxl_gpu_direct_page_registration_span(
        0, UINT64_MAX, UINT64_MAX - 1, 1, UINT64_MAX, 0x1000, &offset,
        &length));
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

static void test_coherent_unmap_unregisters_exact_mapping(void)
{
    const uint64_t alias = UINT64_C(0x12340000);
    CoherentUnmapCounter counter = {
        .synchronize_result = CXL_GPU_SUCCESS,
        .htod_calls = 17,
        .unregister_result = CXL_GPU_SUCCESS,
    };
    uint64_t htod_delta = UINT64_MAX;
    bool invalidated = false;

    g_assert_cmpint(run_coherent_unmap(&counter, true, alias, alias, 17,
                                       &htod_delta, &invalidated),
                    ==, CXL_GPU_SUCCESS);
    g_assert_cmpuint(htod_delta, ==, 0);
    g_assert_true(invalidated);
    g_assert_cmpuint(counter.synchronize_calls, ==, 1);
    g_assert_cmpuint(counter.htod_calls_queries, ==, 1);
    g_assert_cmpuint(counter.unregister_calls, ==, 1);
}

static void test_coherent_unmap_rejects_identity_before_driver(void)
{
    const uint64_t alias = UINT64_C(0x12340000);
    CoherentUnmapCounter counter = { 0 };
    uint64_t htod_delta = UINT64_MAX;
    bool invalidated = true;

    g_assert_cmpint(run_coherent_unmap(&counter, true, alias, alias + 1, 0,
                                       &htod_delta, &invalidated),
                    ==, CXL_GPU_ERROR_INVALID_VALUE);
    g_assert_false(invalidated);
    g_assert_cmpuint(counter.synchronize_calls, ==, 0);
    g_assert_cmpuint(counter.unregister_calls, ==, 0);

    g_assert_cmpint(run_coherent_unmap(&counter, false, 0, alias, 0,
                                       &htod_delta, &invalidated),
                    ==, CXL_GPU_ERROR_INVALID_VALUE);
    g_assert_false(invalidated);
    g_assert_cmpuint(counter.synchronize_calls, ==, 0);
    g_assert_cmpuint(counter.unregister_calls, ==, 0);
}

static void test_coherent_unmap_preserves_mapping_before_unregister(void)
{
    const uint64_t alias = UINT64_C(0x12340000);
    CoherentUnmapCounter counter = {
        .synchronize_result = CXL_GPU_ERROR_UNKNOWN,
    };
    uint64_t htod_delta;
    bool invalidated;

    g_assert_cmpint(run_coherent_unmap(&counter, true, alias, alias, 17,
                                       &htod_delta, &invalidated),
                    ==, CXL_GPU_ERROR_UNKNOWN);
    g_assert_false(invalidated);
    g_assert_cmpuint(counter.unregister_calls, ==, 0);

    counter = (CoherentUnmapCounter) {
        .synchronize_result = CXL_GPU_SUCCESS,
        .htod_calls = 18,
    };
    g_assert_cmpint(run_coherent_unmap(&counter, true, alias, alias, 17,
                                       &htod_delta, &invalidated),
                    ==, CXL_GPU_ERROR_INVALID_VALUE);
    g_assert_false(invalidated);
    g_assert_cmpuint(counter.unregister_calls, ==, 0);

    counter = (CoherentUnmapCounter) {
        .synchronize_result = CXL_GPU_SUCCESS,
        .htod_calls = 17,
        .unregister_result = CXL_GPU_ERROR_INVALID_VALUE,
    };
    g_assert_cmpint(run_coherent_unmap(&counter, true, alias, alias, 17,
                                       &htod_delta, &invalidated),
                    ==, CXL_GPU_ERROR_INVALID_VALUE);
    g_assert_false(invalidated);
    g_assert_cmpuint(counter.unregister_calls, ==, 1);
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
    g_test_add_func("/cxl/type2/direct/batch-destination-independence",
                    test_direct_batch_destinations_are_independent);
    g_test_add_func("/cxl/type2/direct/allocation-lifecycle-epoch",
                    test_cuda_allocation_lifecycle_and_epoch);
    g_test_add_func("/cxl/type2/direct/allocation-invalid-state",
                    test_cuda_allocation_rejects_invalid_state);
    g_test_add_func("/cxl/type2/direct/allocation-alias-lifecycle",
                    test_cuda_allocation_alias_lifecycle);
    g_test_add_func("/cxl/type2/direct/pageable-alias-composite-remap",
                    test_cuda_pageable_alias_composite_remap);
    g_test_add_func("/cxl/type2/direct/pageable-alias-allocation-subranges",
                    test_cuda_pageable_alias_appends_allocation_subranges);
    g_test_add_func("/cxl/type2/direct/generation-reuse-lifecycle",
                    test_cuda_generation_reuse_lifecycle);
    g_test_add_func("/cxl/type2/direct/destination-union-coverage",
                    test_cuda_destination_union_coverage);
    g_test_add_func("/cxl/type2/direct/generated-command-roles",
                    test_cuda_command_roles_use_generated_authority);
    g_test_add_func("/cxl/type2/direct/opcode-summary-wire",
                    test_cuda_opcode_summary_wire_and_conservation);
    g_test_add_func("/cxl/type2/direct/opcode-summary-bounds",
                    test_cuda_opcode_summary_empty_and_bounded);
    g_test_add_func("/cxl/type2/direct/registration-tile-bounds",
                    test_direct_registration_tile_bounds);
    g_test_add_func("/cxl/type2/direct/registration-page-span",
                    test_direct_registration_page_span);
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
    g_test_add_func("/cxl/type2/coherent-unmap/exact-mapping",
                    test_coherent_unmap_unregisters_exact_mapping);
    g_test_add_func("/cxl/type2/coherent-unmap/identity",
                    test_coherent_unmap_rejects_identity_before_driver);
    g_test_add_func("/cxl/type2/coherent-unmap/pre-unregister-failures",
                    test_coherent_unmap_preserves_mapping_before_unregister);
    return g_test_run();
}
