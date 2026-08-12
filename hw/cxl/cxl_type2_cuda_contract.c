#include "qemu/osdep.h"
#include "hw/cxl/cxl_type2_cuda_contract.h"
#include "hw/cxl/cxl_p2p_dma.h"

static const uint8_t cxl_type2_cuda_command_roles[256] = {
#define CXL_TYPE2_CUDA_COMMAND_ROLE(command, role) [command] = role,
#include "hw/cxl/cxl_type2_cuda_command_roles.inc"
#undef CXL_TYPE2_CUDA_COMMAND_ROLE
};

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

bool cxl_gpu_direct_destinations_are_independent(
    const uint64_t *destinations, const size_t *sizes, size_t count,
    size_t *conflict_index)
{
    if (!destinations || !sizes || !count || !conflict_index) {
        return false;
    }
    *conflict_index = SIZE_MAX;
    for (size_t i = 0; i < count; i++) {
        if (!sizes[i] || destinations[i] > UINT64_MAX - sizes[i]) {
            *conflict_index = i;
            return false;
        }
        for (size_t j = 0; j < i; j++) {
            uint64_t end = destinations[i] + sizes[i];
            uint64_t previous_end = destinations[j] + sizes[j];

            if (destinations[i] < previous_end && destinations[j] < end) {
                *conflict_index = i;
                return false;
            }
        }
    }
    return true;
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

static size_t cxl_type2_cuda_allocation_lower_bound(
    const CXLType2CudaAllocationTable *table, uint64_t base)
{
    size_t low = 0;
    size_t high = table->count;

    while (low < high) {
        size_t middle = low + (high - low) / 2;

        if (table->entries[middle].base < base) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low;
}

void cxl_type2_cuda_allocation_table_init(
    CXLType2CudaAllocationTable *table)
{
    g_assert(table);
    memset(table, 0, sizeof(*table));
    table->next_epoch = 1;
    table->available = true;
}

void cxl_type2_cuda_allocation_table_reset(
    CXLType2CudaAllocationTable *table)
{
    g_assert(table);
    table->count = 0;
    table->peak_count = 0;
    table->next_epoch = 1;
    table->available = true;
}

void cxl_type2_cuda_allocation_table_destroy(
    CXLType2CudaAllocationTable *table)
{
    if (!table) {
        return;
    }
    g_free(table->entries);
    memset(table, 0, sizeof(*table));
}

bool cxl_type2_cuda_allocation_record(CXLType2CudaAllocationTable *table,
                                      uint64_t base, uint64_t size,
                                      uint64_t *epoch)
{
    CXLType2CudaAllocation *resized;
    size_t position;
    size_t capacity;

    if (!table || !table->available || !size ||
        base > UINT64_MAX - size || table->next_epoch == UINT64_MAX) {
        if (table) {
            table->available = false;
        }
        return false;
    }
    position = cxl_type2_cuda_allocation_lower_bound(table, base);
    if ((position &&
         table->entries[position - 1].base +
                 table->entries[position - 1].size > base) ||
        (position < table->count &&
         base + size > table->entries[position].base)) {
        table->available = false;
        return false;
    }
    if (table->count == table->capacity) {
        capacity = table->capacity ? table->capacity * 2 : 8;
        if (capacity < table->capacity ||
            capacity > SIZE_MAX / sizeof(*table->entries)) {
            table->available = false;
            return false;
        }
        resized = g_try_realloc_n(table->entries, capacity,
                                  sizeof(*table->entries));
        if (!resized) {
            table->available = false;
            return false;
        }
        table->entries = resized;
        table->capacity = capacity;
    }
    memmove(&table->entries[position + 1], &table->entries[position],
            (table->count - position) * sizeof(*table->entries));
    table->entries[position] = (CXLType2CudaAllocation) {
        .base = base,
        .size = size,
        .epoch = table->next_epoch++,
    };
    table->count++;
    table->peak_count = MAX(table->peak_count, table->count);
    if (epoch) {
        *epoch = table->entries[position].epoch;
    }
    return true;
}

bool cxl_type2_cuda_allocation_forget(CXLType2CudaAllocationTable *table,
                                      uint64_t base)
{
    size_t position;

    if (!table || !table->available) {
        return false;
    }
    position = cxl_type2_cuda_allocation_lower_bound(table, base);
    if (position == table->count || table->entries[position].base != base) {
        table->available = false;
        return false;
    }
    memmove(&table->entries[position], &table->entries[position + 1],
            (table->count - position - 1) * sizeof(*table->entries));
    table->count--;
    return true;
}

typedef struct CXLType2CudaDestinationRange {
    uint64_t base;
    uint64_t size;
} CXLType2CudaDestinationRange;

static int cxl_type2_cuda_destination_range_compare(const void *left,
                                                    const void *right)
{
    const CXLType2CudaDestinationRange *a = left;
    const CXLType2CudaDestinationRange *b = right;

    return a->base < b->base ? -1 : a->base > b->base;
}

static void cxl_type2_cuda_coverage_unknown(
    CXLType2CudaCoverageResult *result,
    CXLType2CudaRejectionReason reason, bool available)
{
    result->kind = CXL_TYPE2_CUDA_COVERAGE_UNKNOWN;
    result->reason = reason;
    result->available = available;
}

void cxl_type2_cuda_destination_union_classify(
    const CXLType2CudaAllocationTable *table,
    const uint64_t *destinations, const size_t *sizes, size_t count,
    CXLType2CudaCoverageResult *result)
{
    g_autofree CXLType2CudaDestinationRange *ranges = NULL;
    size_t allocation_index = 0;
    size_t active_allocation = SIZE_MAX;
    uint64_t covered_end = 0;
    bool whole = true;

    g_assert(result);
    *result = (CXLType2CudaCoverageResult) {
        .kind = CXL_TYPE2_CUDA_COVERAGE_UNKNOWN,
        .reason = CXL_TYPE2_CUDA_REJECTION_ALLOCATION_MISSING,
    };
    if (!table || !table->available || !destinations || !sizes || !count ||
        !table->count) {
        return;
    }
    ranges = g_try_new(CXLType2CudaDestinationRange, count);
    if (!ranges) {
        return;
    }
    for (size_t i = 0; i < count; i++) {
        if (!sizes[i] || destinations[i] > UINT64_MAX - sizes[i] ||
            result->bytes > UINT64_MAX - sizes[i]) {
            cxl_type2_cuda_coverage_unknown(
                result, CXL_TYPE2_CUDA_REJECTION_DESTINATION_OVERFLOW, true);
            return;
        }
        ranges[i] = (CXLType2CudaDestinationRange) {
            .base = destinations[i],
            .size = sizes[i],
        };
        result->bytes += sizes[i];
    }
    qsort(ranges, count, sizeof(*ranges),
          cxl_type2_cuda_destination_range_compare);
    for (size_t i = 1; i < count; i++) {
        if (ranges[i].base < ranges[i - 1].base + ranges[i - 1].size) {
            cxl_type2_cuda_coverage_unknown(
                result, CXL_TYPE2_CUDA_REJECTION_DESTINATION_OVERLAP, true);
            return;
        }
    }

    for (size_t i = 0; i < count; i++) {
        uint64_t range_end = ranges[i].base + ranges[i].size;
        size_t first_overlap;
        size_t overlap_count = 0;

        while (allocation_index < table->count &&
               table->entries[allocation_index].base +
                       table->entries[allocation_index].size <= ranges[i].base) {
            allocation_index++;
        }
        first_overlap = allocation_index;
        while (first_overlap + overlap_count < table->count &&
               table->entries[first_overlap + overlap_count].base < range_end) {
            const CXLType2CudaAllocation *allocation =
                &table->entries[first_overlap + overlap_count];

            if (allocation->base + allocation->size > ranges[i].base) {
                overlap_count++;
            } else {
                first_overlap++;
            }
        }
        if (overlap_count > 1) {
            result->kind = CXL_TYPE2_CUDA_COVERAGE_CROSS;
            result->reason = CXL_TYPE2_CUDA_REJECTION_CROSS_ALLOCATION;
            result->available = true;
            return;
        }
        if (overlap_count != 1) {
            cxl_type2_cuda_coverage_unknown(
                result, CXL_TYPE2_CUDA_REJECTION_ALLOCATION_MISSING, true);
            return;
        }

        const CXLType2CudaAllocation *allocation =
            &table->entries[first_overlap];
        uint64_t allocation_end = allocation->base + allocation->size;

        if (ranges[i].base < allocation->base || range_end > allocation_end) {
            cxl_type2_cuda_coverage_unknown(
                result, CXL_TYPE2_CUDA_REJECTION_ALLOCATION_MISSING, true);
            return;
        }
        if (active_allocation != SIZE_MAX &&
            active_allocation != first_overlap) {
            result->kind = CXL_TYPE2_CUDA_COVERAGE_CROSS;
            result->reason = CXL_TYPE2_CUDA_REJECTION_CROSS_ALLOCATION;
            result->available = true;
            return;
        }
        if (active_allocation == SIZE_MAX) {
            active_allocation = first_overlap;
            covered_end = allocation->base;
        }
        if (ranges[i].base != covered_end) {
            whole = false;
        }
        covered_end = range_end;
    }
    if (active_allocation == SIZE_MAX ||
        covered_end != table->entries[active_allocation].base +
                           table->entries[active_allocation].size) {
        whole = false;
    }
    result->kind = whole ? CXL_TYPE2_CUDA_COVERAGE_WHOLE
                         : CXL_TYPE2_CUDA_COVERAGE_PARTIAL;
    result->reason = whole ? CXL_TYPE2_CUDA_REJECTION_NONE
                           : CXL_TYPE2_CUDA_REJECTION_PARTIAL_COVERAGE;
    result->available = true;
}

CXLType2CudaCommandRole cxl_type2_cuda_command_role(uint32_t command)
{
    if (command >= G_N_ELEMENTS(cxl_type2_cuda_command_roles) ||
        !cxl_type2_cuda_command_roles[command]) {
        return CXL_TYPE2_CUDA_COMMAND_UNKNOWN;
    }
    return cxl_type2_cuda_command_roles[command];
}

const char *cxl_type2_cuda_command_role_name(CXLType2CudaCommandRole role)
{
    switch (role) {
    case CXL_TYPE2_CUDA_COMMAND_READER:
        return "reader";
    case CXL_TYPE2_CUDA_COMMAND_WRITER:
        return "writer";
    case CXL_TYPE2_CUDA_COMMAND_READER_WRITER:
        return "reader+writer";
    case CXL_TYPE2_CUDA_COMMAND_LIFECYCLE:
        return "lifecycle";
    case CXL_TYPE2_CUDA_COMMAND_NO_CHANGE:
        return "no-change";
    case CXL_TYPE2_CUDA_COMMAND_UNKNOWN:
        return "unknown";
    default:
        return "unknown";
    }
}

static bool cxl_type2_cuda_opcode_count_add(uint64_t *value,
                                            uint64_t addend)
{
    if (*value > UINT64_MAX - addend) {
        return false;
    }
    *value += addend;
    return true;
}

bool cxl_type2_cuda_opcode_summary_build(
    const uint64_t command_counts[256], char *records,
    size_t records_capacity, CXLType2CudaOpcodeSummary *summary)
{
    size_t used = 0;
    bool first_record = true;
    bool valid = true;

    if (!command_counts || !records || records_capacity < sizeof("none") ||
        !summary) {
        return false;
    }
    memset(summary, 0, sizeof(*summary));
    records[0] = '\0';
    summary->estimated_materialize_complete = true;
    for (uint32_t command = 0; command < 256; command++) {
        CXLType2CudaCommandRole role;
        uint64_t count = command_counts[command];
        int written;

        if (!count) {
            continue;
        }
        role = cxl_type2_cuda_command_role(command);
        if ((role & CXL_TYPE2_CUDA_COMMAND_READER) &&
            !cxl_type2_cuda_opcode_count_add(
                &summary->reader_commands, count)) {
            valid = false;
        }
        if ((role & CXL_TYPE2_CUDA_COMMAND_WRITER) &&
            !cxl_type2_cuda_opcode_count_add(
                &summary->writer_commands, count)) {
            valid = false;
        }
        if (role == CXL_TYPE2_CUDA_COMMAND_LIFECYCLE &&
            !cxl_type2_cuda_opcode_count_add(
                &summary->lifecycle_commands, count)) {
            valid = false;
        }
        if (role == CXL_TYPE2_CUDA_COMMAND_NO_CHANGE &&
            !cxl_type2_cuda_opcode_count_add(
                &summary->no_change_commands, count)) {
            valid = false;
        }
        if (role == CXL_TYPE2_CUDA_COMMAND_UNKNOWN &&
            !cxl_type2_cuda_opcode_count_add(
                &summary->unknown_commands, count)) {
            valid = false;
        }
        if (!summary->first_incomplete_command_valid) {
            summary->first_incomplete_command = command;
            summary->first_incomplete_command_valid = true;
        }
        summary->estimated_materialize_complete = false;
        written = snprintf(
            records + used, records_capacity - used,
            "%s0x%02X:%s:%" PRIu64 ":0:incomplete",
            first_record ? "" : ";", command,
            cxl_type2_cuda_command_role_name(role), count);
        if (written < 0 || (size_t)written >= records_capacity - used) {
            memcpy(records, "none", sizeof("none"));
            return false;
        }
        used += written;
        first_record = false;
    }
    if (first_record) {
        memcpy(records, "none", sizeof("none"));
    }
    return valid;
}

const char *cxl_type2_cuda_classifier_status_name(
    CXLType2CudaClassifierStatus status)
{
    switch (status) {
    case CXL_TYPE2_CUDA_CLASSIFIER_AVAILABLE:
        return "available";
    case CXL_TYPE2_CUDA_CLASSIFIER_CONTRADICTED:
        return "contradicted";
    case CXL_TYPE2_CUDA_CLASSIFIER_UNAVAILABLE:
        return "unavailable";
    default:
        g_assert_not_reached();
    }
}

const char *cxl_type2_cuda_rejection_reason_name(
    CXLType2CudaRejectionReason reason)
{
    switch (reason) {
    case CXL_TYPE2_CUDA_REJECTION_NONE:
        return "none";
    case CXL_TYPE2_CUDA_REJECTION_ALLOCATION_MISSING:
        return "allocation-missing";
    case CXL_TYPE2_CUDA_REJECTION_PARTIAL_COVERAGE:
        return "partial-coverage";
    case CXL_TYPE2_CUDA_REJECTION_CROSS_ALLOCATION:
        return "cross-allocation";
    case CXL_TYPE2_CUDA_REJECTION_DESTINATION_OVERLAP:
        return "destination-overlap";
    case CXL_TYPE2_CUDA_REJECTION_DESTINATION_OVERFLOW:
        return "destination-overflow";
    case CXL_TYPE2_CUDA_REJECTION_GRAPH_NON_KERNEL:
        return "graph-non-kernel";
    case CXL_TYPE2_CUDA_REJECTION_GRAPH_INCOMPLETE:
        return "graph-incomplete";
    case CXL_TYPE2_CUDA_REJECTION_OPCODE_UNKNOWN:
        return "opcode-unknown";
    default:
        g_assert_not_reached();
    }
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

int cxl_type2_coherent_unmap_execute(
    bool host_registered, uint64_t stored_alias, uint64_t requested_alias,
    uint64_t htod_calls_at_map, const CXLType2CoherentUnmapOps *ops,
    void *opaque, uint64_t *htod_delta, int *stale_query_status,
    bool *mapping_invalidated)
{
    int result;

    if (!ops || !ops->synchronize || !ops->htod_calls ||
        !ops->unregister_host || !ops->query_memory_type || !htod_delta ||
        !stale_query_status || !mapping_invalidated) {
        return CXL_GPU_ERROR_INVALID_VALUE;
    }
    *htod_delta = 0;
    *stale_query_status = CXL_GPU_ERROR_UNKNOWN;
    *mapping_invalidated = false;
    if (!host_registered || !stored_alias || stored_alias != requested_alias) {
        return CXL_GPU_ERROR_INVALID_VALUE;
    }
    result = ops->synchronize(opaque);
    if (result != CXL_GPU_SUCCESS) {
        return result;
    }
    *htod_delta = ops->htod_calls(opaque) - htod_calls_at_map;
    if (*htod_delta != 0) {
        return CXL_GPU_ERROR_INVALID_VALUE;
    }
    result = ops->unregister_host(opaque);
    if (result != CXL_GPU_SUCCESS) {
        return result;
    }
    *mapping_invalidated = true;
    *stale_query_status = ops->query_memory_type(opaque, requested_alias);
    if (*stale_query_status == CXL_GPU_ERROR_INVALID_VALUE) {
        return CXL_GPU_SUCCESS;
    }
    return *stale_query_status == CXL_GPU_SUCCESS
               ? CXL_GPU_ERROR_INVALID_VALUE
               : *stale_query_status;
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
    uint64_t ranges_bytes;
    uint64_t logical_bytes = 0;

    if (!payload || !header_out || !fail_index ||
        payload_bytes < sizeof(header) || payload_bytes > payload_capacity) {
        return false;
    }
    *fail_index = SIZE_MAX;
    memcpy(&header, payload, sizeof(header));
    if (header.flags || header.reserved0 || header.reserved1 ||
        header.reserved2[0] || header.reserved2[1] || header.reserved2[2] ||
        !header.range_count ||
        header.range_count >
            (payload_bytes - sizeof(header)) /
                sizeof(CXLGPUSourceVirtualRangeV1)) {
        return false;
    }
    ranges_bytes = (uint64_t)header.range_count *
                   sizeof(CXLGPUSourceVirtualRangeV1);
    if (sizeof(header) + ranges_bytes != payload_bytes) {
        return false;
    }
    range_base = payload + sizeof(header);

    for (uint64_t i = 0; i < header.range_count; i++) {
        CXLGPUSourceVirtualRangeV1 range;

        memcpy(&range, range_base + i * sizeof(range), sizeof(range));
        if (!range.guest_virtual_address || !range.length ||
            range.guest_virtual_address > UINT64_MAX - range.length) {
            *fail_index = i;
            return false;
        }
        if (logical_bytes > UINT64_MAX - range.length) {
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
