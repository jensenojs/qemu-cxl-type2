#include "qemu/osdep.h"
#include "hw/cxl/cxl_type2_cuda_contract.h"

typedef struct QueryCounter {
    unsigned int calls;
    int32_t attribute;
} QueryCounter;

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
    return g_test_run();
}
