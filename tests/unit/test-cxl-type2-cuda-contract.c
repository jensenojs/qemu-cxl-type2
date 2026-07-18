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
    return g_test_run();
}
