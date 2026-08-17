#include "hw/cxl/cxl_type2_cuda_contract.h"
#include "hw/cxl/cxl_p2p_dma.h"
#include "qemu/osdep.h"
#include "qemu/timer.h"

static const uint8_t cxl_type2_cuda_command_roles[256] = {
#define CXL_TYPE2_CUDA_COMMAND_ROLE(command, role) [command] = role,
#include "hw/cxl/cxl_type2_cuda_command_roles.inc"
#undef CXL_TYPE2_CUDA_COMMAND_ROLE
};

static void
cxl_type2_cuda_pageable_alias_free(CXLType2CudaPageableAlias *alias) {
  if (!alias) {
    return;
  }
  g_assert(!alias->host_registered);
  if (alias->owned_reservation) {
    munmap(alias->owned_reservation, alias->owned_reservation_size);
  }
  for (size_t i = 0; i < alias->source_count; i++) {
    close(alias->sources[i].fd);
    g_free(alias->sources[i].owned_host_copy);
  }
  g_free(alias->contributing_source_call_ids);
  g_free(alias->sources);
  g_free(alias);
}

static void
cxl_type2_cuda_allocation_alias_clear(CXLType2CudaAllocation *allocation) {
  CXLType2CudaPageableAlias *alias = allocation->pageable_alias;

  while (alias) {
    CXLType2CudaPageableAlias *next = alias->next;

    cxl_type2_cuda_pageable_alias_free(alias);
    alias = next;
  }
  allocation->pageable_alias = NULL;
  allocation->device_alias = 0;
  allocation->content_generation = 0;
  allocation->dax_backed = false;
}

static bool cxl_type2_descriptor_reserved_is_zero(
    const CXLGPURAMCommandDescriptor *descriptor) {
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
    bool paired_case_required, uint64_t active_case_epoch) {
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

const char *
cxl_type2_descriptor_verdict_reason(CXLType2DescriptorRequestVerdict verdict) {
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

bool cxl_type2_cuda_decode_attribute(uint64_t wire_value, int32_t *attribute) {
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

bool cxl_type2_cuda_attribute_wire_is_valid(uint64_t wire_value) {
  int32_t attribute;

  return cxl_type2_cuda_decode_attribute(wire_value, &attribute);
}

bool cxl_type2_cuda_stream_progress_wire(uint32_t command,
                                         const uint64_t params[8],
                                         uint64_t *stream_wire) {
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
    bool previous_command_was_successful_sync, uint64_t previous_stream_wire,
    uint64_t current_stream_wire) {
  return previous_command_was_successful_sync &&
         previous_stream_wire == current_stream_wire;
}

bool cxl_type2_cuda_decode_stream_sync_reason(
    uint32_t descriptor_protocol_version, uint64_t wire_reason,
    CXLGPUStreamSyncReason *reason) {
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
                                             void **stream) {
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

int cxl_gpu_direct_host_address_order(uintptr_t left, uintptr_t right) {
  return left < right ? -1 : left > right;
}

bool cxl_gpu_direct_host_range_follows(uintptr_t base, uint64_t length,
                                       uintptr_t next, uint64_t next_length) {
  return length <= UINTPTR_MAX - base && next == base + length &&
         next_length <= UINT64_MAX - length;
}

bool cxl_gpu_direct_epoch_is_cross_case(uint64_t last_case_epoch,
                                        uint64_t active_case_epoch) {
  return last_case_epoch && active_case_epoch &&
         last_case_epoch != active_case_epoch;
}

bool cxl_gpu_direct_registration_group_follows(uintptr_t mapping,
                                               uintptr_t base, uint64_t length,
                                               uintptr_t next_mapping,
                                               uintptr_t next,
                                               uint64_t next_length) {
  return mapping == next_mapping &&
         cxl_gpu_direct_host_range_follows(base, length, next, next_length);
}

bool cxl_gpu_direct_copy_span_follows(uintptr_t source, uintptr_t registration,
                                      uintptr_t host, uint64_t destination,
                                      uint64_t length, uintptr_t next_source,
                                      uintptr_t next_registration,
                                      uintptr_t next_host,
                                      uint64_t next_destination,
                                      uint64_t next_length) {
  return source == next_source && registration == next_registration &&
         cxl_gpu_direct_host_range_follows(host, length, next_host,
                                           next_length) &&
         length <= UINT64_MAX - destination &&
         next_destination == destination + length;
}

bool cxl_gpu_direct_destinations_are_independent(const uint64_t *destinations,
                                                 const size_t *sizes,
                                                 size_t count,
                                                 size_t *conflict_index) {
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
    uint64_t mapping_offset, uint64_t mapping_length, uint64_t request_offset,
    uint64_t request_length, uint64_t following_offset, uint64_t tile_size,
    uint64_t padding_budget) {
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
    rounded_end = ((relative_end + tile_size - 1) / tile_size) * tile_size;
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

bool cxl_gpu_direct_page_registration_span(
    uint64_t mapping_offset, uint64_t mapping_length, uint64_t request_offset,
    uint64_t request_length, uint64_t following_offset, uint64_t page_size,
    uint64_t *registration_offset, uint64_t *registration_length) {
  uint64_t mapping_end;
  uint64_t request_end;
  uint64_t aligned_end;

  if (!mapping_length || !request_length || !page_size ||
      (page_size & (page_size - 1)) || !registration_offset ||
      !registration_length || mapping_offset > UINT64_MAX - mapping_length ||
      request_offset < mapping_offset ||
      request_offset > UINT64_MAX - request_length) {
    return false;
  }
  mapping_end = mapping_offset + mapping_length;
  request_end = request_offset + request_length;
  if (request_end > mapping_end || following_offset < request_end ||
      following_offset > mapping_end ||
      request_end > UINT64_MAX - (page_size - 1)) {
    return false;
  }
  aligned_end = (request_end + page_size - 1) & ~(page_size - 1);
  *registration_offset = request_offset & ~(page_size - 1);
  if (*registration_offset < mapping_offset || aligned_end > following_offset ||
      aligned_end > mapping_end) {
    return false;
  }
  *registration_length = aligned_end - *registration_offset;
  return *registration_length != 0;
}

static size_t
cxl_type2_cuda_allocation_lower_bound(const CXLType2CudaAllocationTable *table,
                                      uint64_t base) {
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

void cxl_type2_cuda_allocation_table_init(CXLType2CudaAllocationTable *table) {
  g_assert(table);
  memset(table, 0, sizeof(*table));
  table->next_epoch = 1;
  table->available = true;
}

void cxl_type2_cuda_allocation_table_reset(CXLType2CudaAllocationTable *table) {
  g_assert(table);
  for (size_t i = 0; i < table->count; i++) {
    cxl_type2_cuda_allocation_alias_clear(&table->entries[i]);
  }
  table->count = 0;
  table->peak_count = 0;
  table->next_epoch = 1;
  table->generation_count = 0;
  table->population_interval_count = 0;
  table->available = true;
  table->first_generation_error = NULL;
}

void cxl_type2_cuda_allocation_table_destroy(
    CXLType2CudaAllocationTable *table) {
  if (!table) {
    return;
  }
  for (size_t i = 0; i < table->count; i++) {
    cxl_type2_cuda_allocation_alias_clear(&table->entries[i]);
  }
  g_free(table->entries);
  g_free(table->generations);
  g_free(table->population_intervals);
  memset(table, 0, sizeof(*table));
}

static CXLType2CudaAllocation *
cxl_type2_cuda_allocation_containing(CXLType2CudaAllocationTable *table,
                                     uint64_t address, uint64_t size) {
  size_t position;

  if (!table || !table->available || !size || address > UINT64_MAX - size) {
    return NULL;
  }
  position = cxl_type2_cuda_allocation_lower_bound(table, address);
  if (position < table->count && table->entries[position].base == address) {
    CXLType2CudaAllocation *allocation = &table->entries[position];

    return size <= allocation->size ? allocation : NULL;
  }
  if (!position) {
    return NULL;
  }
  CXLType2CudaAllocation *allocation = &table->entries[position - 1];

  return address >= allocation->base &&
                 address - allocation->base <= allocation->size &&
                 size <= allocation->size - (address - allocation->base)
             ? allocation
             : NULL;
}

static void cxl_type2_cuda_generation_fail(CXLType2CudaAllocationTable *table,
                                           const char *reason) {
  table->available = false;
  if (!table->first_generation_error) {
    table->first_generation_error = reason;
  }
}

static CXLType2CudaGenerationRecord *
cxl_type2_cuda_generation_publish(CXLType2CudaAllocationTable *table,
                                  CXLType2CudaAllocation *allocation,
                                  uint64_t generation) {
  CXLType2CudaGenerationRecord *resized;
  size_t capacity;

  if (!table || !allocation || !generation) {
    return NULL;
  }
  for (size_t i = table->generation_count; i > 0; i--) {
    CXLType2CudaGenerationRecord *record = &table->generations[i - 1];

    if (record->allocation_base != allocation->base ||
        record->epoch != allocation->epoch) {
      continue;
    }
    if (record->generation == generation) {
      record->file_backed_system_uva = allocation->dax_backed;
      return record;
    }
    if (record->generation > generation) {
      cxl_type2_cuda_generation_fail(table, "generation-order-invalid");
      return NULL;
    }
    break;
  }
  if (table->generation_count == table->generation_capacity) {
    capacity = table->generation_capacity ? table->generation_capacity * 2 : 16;
    if (capacity < table->generation_capacity ||
        capacity > SIZE_MAX / sizeof(*table->generations)) {
      cxl_type2_cuda_generation_fail(table, "generation-capacity-overflow");
      return NULL;
    }
    resized = g_try_realloc_n(table->generations, capacity,
                              sizeof(*table->generations));
    if (!resized) {
      cxl_type2_cuda_generation_fail(table, "generation-allocation-failed");
      return NULL;
    }
    table->generations = resized;
    table->generation_capacity = capacity;
  }
  CXLType2CudaGenerationRecord *record =
      &table->generations[table->generation_count++];

  *record = (CXLType2CudaGenerationRecord){
      .allocation_base = allocation->base,
      .epoch = allocation->epoch,
      .generation = generation,
      .file_backed_system_uva = allocation->dax_backed,
      .prefetch_completion_available = true,
      .next_boundary = CXL_TYPE2_CUDA_GENERATION_OPEN,
  };
  return record;
}

static CXLType2CudaGenerationRecord *
cxl_type2_cuda_generation_current(CXLType2CudaAllocationTable *table,
                                  CXLType2CudaAllocation *allocation,
                                  bool population) {
  CXLType2CudaGenerationRecord *record = NULL;
  uint64_t next_generation = 1;

  for (size_t i = table->generation_count; i > 0; i--) {
    CXLType2CudaGenerationRecord *candidate = &table->generations[i - 1];

    if (candidate->allocation_base != allocation->base ||
        candidate->epoch != allocation->epoch) {
      continue;
    }
    next_generation = candidate->generation + 1;
    if (candidate->next_boundary == CXL_TYPE2_CUDA_GENERATION_OPEN) {
      record = candidate;
    }
    break;
  }
  if (record && (!population ||
                 (!record->consumer_observed && record->prefetch_count == 0))) {
    record->file_backed_system_uva = allocation->dax_backed;
    return record;
  }
  if (record) {
    record->next_boundary = CXL_TYPE2_CUDA_GENERATION_POPULATION;
  }
  if (next_generation == 0) {
    cxl_type2_cuda_generation_fail(table, "generation-overflow");
    return NULL;
  }
  return cxl_type2_cuda_generation_publish(table, allocation, next_generation);
}

static bool
cxl_type2_cuda_population_interval_add(CXLType2CudaAllocationTable *table,
                                       size_t generation_index, uint64_t begin,
                                       uint64_t end) {
  CXLType2CudaGenerationRecord *record = &table->generations[generation_index];
  uint64_t merged_begin = begin;
  uint64_t merged_end = end;
  size_t index = 0;

  while (index < table->population_interval_count) {
    CXLType2CudaPopulationInterval *interval =
        &table->population_intervals[index];

    if (interval->generation_index != generation_index ||
        interval->end < merged_begin || merged_end < interval->begin) {
      index++;
      continue;
    }
    merged_begin = MIN(merged_begin, interval->begin);
    merged_end = MAX(merged_end, interval->end);
    record->unique_visible_population_bytes -= interval->end - interval->begin;
    memmove(interval, interval + 1,
            (table->population_interval_count - index - 1) * sizeof(*interval));
    table->population_interval_count--;
  }
  if (table->population_interval_count == table->population_interval_capacity) {
    size_t capacity = table->population_interval_capacity
                          ? table->population_interval_capacity * 2
                          : 16;
    CXLType2CudaPopulationInterval *resized;

    if (capacity < table->population_interval_capacity ||
        capacity > SIZE_MAX / sizeof(*table->population_intervals)) {
      cxl_type2_cuda_generation_fail(table, "population-capacity-overflow");
      return false;
    }
    resized = g_try_realloc_n(table->population_intervals, capacity,
                              sizeof(*table->population_intervals));
    if (!resized) {
      cxl_type2_cuda_generation_fail(table, "population-allocation-failed");
      return false;
    }
    table->population_intervals = resized;
    table->population_interval_capacity = capacity;
  }
  table->population_intervals[table->population_interval_count++] =
      (CXLType2CudaPopulationInterval){
          .generation_index = generation_index,
          .begin = merged_begin,
          .end = merged_end,
      };
  record->unique_visible_population_bytes += merged_end - merged_begin;
  return true;
}

bool cxl_type2_cuda_allocation_record(CXLType2CudaAllocationTable *table,
                                      uint64_t base, uint64_t size,
                                      uint64_t *epoch) {
  CXLType2CudaAllocation *resized;
  size_t position;
  size_t capacity;

  if (!table || !table->available || !size || base > UINT64_MAX - size ||
      table->next_epoch == UINT64_MAX) {
    if (table) {
      table->available = false;
    }
    return false;
  }
  position = cxl_type2_cuda_allocation_lower_bound(table, base);
  if ((position &&
       table->entries[position - 1].base + table->entries[position - 1].size >
           base) ||
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
    resized =
        g_try_realloc_n(table->entries, capacity, sizeof(*table->entries));
    if (!resized) {
      table->available = false;
      return false;
    }
    table->entries = resized;
    table->capacity = capacity;
  }
  memmove(&table->entries[position + 1], &table->entries[position],
          (table->count - position) * sizeof(*table->entries));
  table->entries[position] = (CXLType2CudaAllocation){
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
                                      uint64_t base) {
  size_t position;

  if (!table || !table->available) {
    return false;
  }
  position = cxl_type2_cuda_allocation_lower_bound(table, base);
  if (position == table->count || table->entries[position].base != base) {
    table->available = false;
    return false;
  }
  if (table->entries[position].consumer_refs ||
      table->entries[position].normal_inflight_refs ||
      table->entries[position].graph_binding_refs ||
      table->entries[position].graph_inflight_refs) {
    return false;
  }
  if (!cxl_type2_cuda_generation_release(table, base)) {
    return false;
  }
  cxl_type2_cuda_allocation_alias_clear(&table->entries[position]);
  memmove(&table->entries[position], &table->entries[position + 1],
          (table->count - position - 1) * sizeof(*table->entries));
  table->count--;
  return true;
}

CXLType2CudaAllocation *
cxl_type2_cuda_allocation_find(CXLType2CudaAllocationTable *table,
                               uint64_t base, uint64_t epoch) {
  size_t position;

  if (!table || !table->available || !epoch) {
    return NULL;
  }
  position = cxl_type2_cuda_allocation_lower_bound(table, base);
  if (position == table->count || table->entries[position].base != base ||
      table->entries[position].epoch != epoch) {
    return NULL;
  }
  return &table->entries[position];
}

bool cxl_type2_cuda_allocation_publish_alias(CXLType2CudaAllocationTable *table,
                                             uint64_t base, uint64_t epoch,
                                             uint64_t generation,
                                             uint64_t device_alias) {
  CXLType2CudaAllocation *allocation =
      cxl_type2_cuda_allocation_find(table, base, epoch);

  if (!allocation || !generation || !device_alias || allocation->poisoned ||
      allocation->dax_backed || allocation->consumer_refs ||
      allocation->pageable_alias) {
    return false;
  }
  allocation->content_generation = generation;
  allocation->device_alias = device_alias;
  allocation->dax_backed = true;
  for (size_t i = table->generation_count; i > 0; i--) {
    CXLType2CudaGenerationRecord *record = &table->generations[i - 1];

    if (record->allocation_base == base && record->epoch == epoch &&
        record->next_boundary == CXL_TYPE2_CUDA_GENERATION_OPEN) {
      record->file_backed_system_uva = true;
      break;
    }
  }
  return true;
}

bool cxl_type2_cuda_allocation_acquire_alias(CXLType2CudaAllocationTable *table,
                                             uint64_t base, uint64_t epoch,
                                             uint64_t generation,
                                             uint64_t *device_alias) {
  CXLType2CudaAllocation *allocation =
      cxl_type2_cuda_allocation_find(table, base, epoch);

  if (!allocation || !device_alias || allocation->poisoned ||
      !allocation->dax_backed || allocation->content_generation != generation ||
      allocation->consumer_refs == UINT64_MAX) {
    return false;
  }
  allocation->consumer_refs++;
  *device_alias = allocation->device_alias;
  return true;
}

bool cxl_type2_cuda_allocation_release_alias(CXLType2CudaAllocationTable *table,
                                             uint64_t base, uint64_t epoch,
                                             uint64_t generation) {
  CXLType2CudaAllocation *allocation =
      cxl_type2_cuda_allocation_find(table, base, epoch);

  if (!allocation || allocation->poisoned || !allocation->dax_backed ||
      allocation->content_generation != generation ||
      !allocation->consumer_refs) {
    return false;
  }
  allocation->consumer_refs--;
  return true;
}

bool cxl_type2_cuda_allocation_materialize(CXLType2CudaAllocationTable *table,
                                           uint64_t base, uint64_t epoch,
                                           uint64_t generation) {
  CXLType2CudaAllocation *allocation =
      cxl_type2_cuda_allocation_find(table, base, epoch);

  if (!allocation || allocation->poisoned || !allocation->dax_backed ||
      allocation->content_generation != generation ||
      allocation->consumer_refs || allocation->normal_inflight_refs ||
      allocation->graph_binding_refs || allocation->graph_inflight_refs) {
    return false;
  }
  cxl_type2_cuda_allocation_alias_clear(allocation);
  return true;
}

static bool
cxl_type2_cuda_alias_source_copy(CXLType2CudaAliasSource *destination,
                                 const CXLType2CudaAliasSource *source,
                                 const char **reason) {
  struct stat source_stat;
  int duplicate;
  bool host_copy = source->host_copy_source != NULL;

  if (!source->length || !source->readonly ||
      (host_copy == (source->fd >= 0)) ||
      (!host_copy && (!source->mapping_generation ||
                      source->file_offset > UINT64_MAX - source->length))) {
    *reason = "source-invalid";
    return false;
  }
  if (host_copy) {
    *destination = *source;
    destination->fd = -1;
    destination->owned_host_copy =
        g_memdup2(source->host_copy_source, source->length);
    destination->host_copy_source = destination->owned_host_copy;
    if (!destination->owned_host_copy) {
      *reason = "source-copy-allocation-failed";
      return false;
    }
    return true;
  }
  if (fstat(source->fd, &source_stat) < 0) {
    *reason = "source-stat-failed";
    return false;
  }
  if (source_stat.st_size < 0 ||
      (uint64_t)source_stat.st_dev != source->stat_device ||
      (uint64_t)source_stat.st_ino != source->stat_inode ||
      (uint64_t)source_stat.st_size != source->stat_size ||
      (uint32_t)source_stat.st_mode != source->stat_mode ||
      source->file_offset + source->length > source->stat_size) {
    *reason = "source-stat-mismatch";
    return false;
  }
  duplicate = qemu_dup(source->fd);
  if (duplicate < 0) {
    *reason = "source-dup-failed";
    return false;
  }
  *destination = *source;
  destination->fd = duplicate;
  return true;
}

static bool cxl_type2_cuda_pread_all(int fd, uint8_t *destination,
                                     uint64_t count, uint64_t offset) {
  while (count) {
    size_t request = MIN(count, (uint64_t)SSIZE_MAX);
    ssize_t done = pread(fd, destination, request, offset);

    if (done < 0 && errno == EINTR) {
      continue;
    }
    if (done <= 0) {
      return false;
    }
    destination += done;
    offset += done;
    count -= done;
  }
  return true;
}

static int cxl_type2_cuda_alias_source_compare(const void *left,
                                               const void *right) {
  const CXLType2CudaAliasSource *a = left;
  const CXLType2CudaAliasSource *b = right;

  return a->destination_offset < b->destination_offset   ? -1
         : a->destination_offset > b->destination_offset ? 1
                                                         : 0;
}

static int cxl_type2_cuda_u64_compare(const void *left, const void *right) {
  uint64_t a = *(const uint64_t *)left;
  uint64_t b = *(const uint64_t *)right;

  return a < b ? -1 : a > b ? 1 : 0;
}

bool cxl_type2_cuda_pageable_alias_contains(
    const CXLType2CudaPageableAlias *alias, uint64_t allocation_offset) {
  for (; alias; alias = alias->next) {
    if (allocation_offset >= alias->destination_offset &&
        allocation_offset - alias->destination_offset < alias->logical_bytes) {
      return true;
    }
  }
  return false;
}

const CXLType2CudaPageableAlias *
cxl_type2_cuda_allocation_pageable_alias_for_offset(
    const CXLType2CudaAllocation *allocation, uint64_t allocation_offset) {
  const CXLType2CudaPageableAlias *alias =
      allocation ? allocation->pageable_alias : NULL;

  for (; alias; alias = alias->next) {
    if (allocation_offset >= alias->destination_offset &&
        allocation_offset - alias->destination_offset < alias->logical_bytes) {
      return alias;
    }
  }
  return NULL;
}

CXLType2CudaPageableAlias *
cxl_type2_cuda_allocation_pageable_alias_for_offset_mutable(
    CXLType2CudaAllocation *allocation, uint64_t allocation_offset) {
  return (CXLType2CudaPageableAlias *)
      cxl_type2_cuda_allocation_pageable_alias_for_offset(allocation,
                                                          allocation_offset);
}

static bool
cxl_type2_cuda_alias_verify_range(const CXLType2CudaPageableAlias *alias,
                                  uint64_t begin, uint64_t end) {
  uint8_t expected[64 * 1024];

  for (size_t i = 0; i < alias->source_count; i++) {
    const CXLType2CudaAliasSource *source = &alias->sources[i];
    uint64_t source_end = source->destination_offset + source->length;
    uint64_t compare_begin = MAX(begin, source->destination_offset);
    uint64_t compare_end = MIN(end, source_end);
    uint64_t compared;

    if (compare_begin >= compare_end) {
      continue;
    }
    compared = compare_begin - source->destination_offset;

    while (source->destination_offset + compared < compare_end) {
      size_t count = MIN((uint64_t)sizeof(expected),
                         compare_end - source->destination_offset - compared);

      bool read_ok =
          source->host_copy_source
              ? (memcpy(expected,
                        (const uint8_t *)source->host_copy_source + compared,
                        count),
                 true)
              : cxl_type2_cuda_pread_all(source->fd, expected, count,
                                         source->file_offset + compared);

      if (!read_ok ||
          memcmp(expected,
                 alias->reservation + source->destination_offset + compared,
                 count) != 0) {
        return false;
      }
      compared += count;
    }
  }
  return true;
}

static bool cxl_type2_cuda_alias_map_pages(CXLType2CudaPageableAlias *alias,
                                           uint64_t begin, uint64_t end,
                                           const char **reason) {
  size_t page_size = qemu_real_host_page_size();
  size_t source_index = 0;
  uint64_t page_first = QEMU_ALIGN_DOWN(alias->pointer_bias + begin, page_size);
  uint64_t page_limit = QEMU_ALIGN_UP(alias->pointer_bias + end, page_size);

  for (uint64_t mapping_begin = page_first; mapping_begin < page_limit;
       mapping_begin += page_size) {
    uint64_t page_begin = mapping_begin > alias->pointer_bias
                              ? mapping_begin - alias->pointer_bias
                              : 0;
    uint64_t page_end =
        MIN(end, mapping_begin + page_size - alias->pointer_bias);
    uint64_t covered_end = page_end;
    const CXLType2CudaAliasSource *source;
    uint64_t source_end;
    uint64_t file_page_offset;
    void *page_address = alias->mapping_base + mapping_begin;
    void *mapped;

    while (source_index < alias->source_count &&
           alias->sources[source_index].destination_offset +
                   alias->sources[source_index].length <=
               page_begin) {
      source_index++;
    }
    source = source_index < alias->source_count ? &alias->sources[source_index]
                                                : NULL;
    source_end = source ? source->destination_offset + source->length : 0;
    file_page_offset =
        source ? source->file_offset + page_begin - source->destination_offset
               : 0;
    if (mapping_begin >= alias->pointer_bias &&
        page_end - page_begin == page_size && source && source->fd >= 0 &&
        source->destination_offset <= page_begin && source_end >= page_end &&
        file_page_offset % page_size == 0) {
      mapped = mmap(page_address, page_size, PROT_READ, MAP_SHARED | MAP_FIXED,
                    source->fd, file_page_offset);
      if (mapped != page_address) {
        *reason = "file-page-map-failed";
        return false;
      }
      alias->file_mapped_bytes += page_size;
      continue;
    }

    mapped = mmap(page_address, page_size, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    if (mapped != page_address) {
      *reason = "boundary-page-map-failed";
      return false;
    }
    uint64_t composition_begin_ns = qemu_clock_get_ns(QEMU_CLOCK_REALTIME);

    for (size_t i = source_index; i < alias->source_count; i++) {
      source = &alias->sources[i];
      source_end = source->destination_offset + source->length;
      uint64_t copy_begin = MAX(page_begin, source->destination_offset);
      uint64_t copy_end = MIN(covered_end, source_end);

      if (source->destination_offset >= covered_end) {
        break;
      }
      if (copy_begin >= copy_end) {
        continue;
      }
      if (source->host_copy_source) {
        memcpy(alias->reservation + copy_begin,
               (const uint8_t *)source->host_copy_source + copy_begin -
                   source->destination_offset,
               copy_end - copy_begin);
        alias->host_composition_copy_bytes += copy_end - copy_begin;
      } else {
        if (!cxl_type2_cuda_pread_all(source->fd,
                                      alias->reservation + copy_begin,
                                      copy_end - copy_begin,
                                      source->file_offset + copy_begin -
                                          source->destination_offset)) {
          *reason = "boundary-page-read-failed";
          return false;
        }
        alias->derived_boundary_copy_bytes += copy_end - copy_begin;
      }
    }
    if (mprotect(page_address, page_size, PROT_READ) < 0) {
      *reason = "boundary-page-protect-failed";
      return false;
    }
    alias->boundary_composition_wall_ns +=
        qemu_clock_get_ns(QEMU_CLOCK_REALTIME) - composition_begin_ns;
    alias->derived_boundary_pages++;
  }
  if (!cxl_type2_cuda_alias_verify_range(alias, begin, end)) {
    *reason = "alias-byte-oracle-failed";
    return false;
  }
  return true;
}

bool cxl_type2_cuda_allocation_map_pageable_alias(
    CXLType2CudaAllocationTable *table, uint64_t base, uint64_t epoch,
    uint64_t generation, const CXLType2CudaAliasSource *sources,
    size_t source_count, const uint64_t *contributing_source_call_ids,
    size_t contributing_source_call_count, uint64_t destination_offset,
    uint64_t logical_bytes, uint64_t guard_bytes, uint64_t *device_alias,
    const char **reason) {
  CXLType2CudaAllocation *allocation =
      cxl_type2_cuda_allocation_find(table, base, epoch);
  CXLType2CudaPageableAlias *alias = NULL;
  uint64_t total_bytes;
  uint64_t range_end;
  size_t page_size = qemu_real_host_page_size();
  uint64_t covered = 0;
  bool bias_set = false;

  if (reason) {
    *reason = NULL;
  }
  if (!reason || !allocation || allocation->poisoned || !generation ||
      !sources || !source_count || !contributing_source_call_ids ||
      !contributing_source_call_count || !logical_bytes || !device_alias ||
      guard_bytes > UINT64_MAX - logical_bytes) {
    if (reason) {
      *reason = "alias-request-invalid";
    }
    return false;
  }
  total_bytes = logical_bytes + guard_bytes;
  if (destination_offset > allocation->size ||
      total_bytes > allocation->size - destination_offset ||
      total_bytes > UINT64_MAX - (page_size - 1) ||
      allocation->size > UINT64_MAX - (page_size - 1)) {
    *reason = "alias-allocation-bounds";
    return false;
  }
  range_end = destination_offset + total_bytes;
  if (allocation->consumer_refs || allocation->normal_inflight_refs ||
      allocation->graph_binding_refs || allocation->graph_inflight_refs) {
    *reason = "alias-consumer-in-flight";
    return false;
  }

  for (CXLType2CudaPageableAlias *existing = allocation->pageable_alias;
       existing; existing = existing->next) {
    uint64_t existing_end =
        existing->destination_offset + existing->logical_bytes;

    if (existing_end > destination_offset &&
        existing->destination_offset < range_end) {
      *reason = "alias-subrange-overlap";
      return false;
    }
  }

  alias = g_try_new0(CXLType2CudaPageableAlias, 1);
  if (!alias) {
    *reason = "alias-allocation-failed";
    return false;
  }
  alias->sources = g_try_new0(CXLType2CudaAliasSource, source_count);
  if (!alias->sources) {
    *reason = "alias-source-allocation-failed";
    goto fail;
  }
  alias->source_count = source_count;
  for (size_t i = 0; i < alias->source_count; i++) {
    alias->sources[i].fd = -1;
  }
  alias->contributing_source_call_ids =
      g_try_new(uint64_t, contributing_source_call_count);
  if (!alias->contributing_source_call_ids) {
    *reason = "alias-source-call-allocation-failed";
    goto fail;
  }
  for (size_t i = 0; i < contributing_source_call_count; i++) {
    if (!contributing_source_call_ids[i] ||
        (i && contributing_source_call_ids[i] <=
                  contributing_source_call_ids[i - 1])) {
      *reason = "alias-source-call-order-invalid";
      goto fail;
    }
  }
  alias->content_generation = generation;
  alias->destination_offset = destination_offset;
  alias->logical_bytes = total_bytes;
  alias->guard_bytes = guard_bytes;

  for (size_t i = 0; i < source_count; i++) {
    uint64_t residue;

    if (sources[i].fd < 0) {
      continue;
    }
    residue = (sources[i].file_offset % page_size + page_size -
               sources[i].destination_offset % page_size) %
              page_size;
    if (!bias_set) {
      alias->pointer_bias = residue;
      bias_set = true;
    } else if (alias->pointer_bias != residue) {
      *reason = "alias-source-page-offset-mismatch";
      goto fail;
    }
  }
  alias->reservation_size = total_bytes;
  alias->mapping_size =
      QEMU_ALIGN_UP(alias->pointer_bias + total_bytes, page_size);
  if (alias->mapping_size > SIZE_MAX - 2 * page_size) {
    *reason = "alias-reservation-size-overflow";
    goto fail;
  }
  alias->owned_reservation_size = alias->mapping_size + 2 * page_size;

  for (size_t i = 0; i < source_count; i++) {
    if (sources[i].destination_offset != covered ||
        sources[i].length > total_bytes - covered) {
      *reason = "alias-source-coverage";
      goto fail;
    }
    if (!cxl_type2_cuda_alias_source_copy(&alias->sources[i], &sources[i],
                                          reason)) {
      goto fail;
    }
    if (contributing_source_call_count == 1) {
      alias->sources[i].source_call_id = contributing_source_call_ids[0];
    }
    covered += sources[i].length;
  }
  if (covered != total_bytes) {
    *reason = "alias-source-coverage";
    goto fail;
  }
  for (size_t i = 0; i < contributing_source_call_count; i++) {
    alias->contributing_source_call_ids
        [alias->contributing_source_call_count++] =
        contributing_source_call_ids[i];
  }
  qsort(alias->contributing_source_call_ids,
        alias->contributing_source_call_count, sizeof(uint64_t),
        cxl_type2_cuda_u64_compare);
  size_t unique_call_count = 0;
  for (size_t i = 0; i < alias->contributing_source_call_count; i++) {
    if (!i || alias->contributing_source_call_ids[i] !=
                  alias->contributing_source_call_ids[i - 1]) {
      alias->contributing_source_call_ids[unique_call_count++] =
          alias->contributing_source_call_ids[i];
    }
  }
  alias->contributing_source_call_count = unique_call_count;
  qsort(alias->sources, alias->source_count, sizeof(CXLType2CudaAliasSource),
        cxl_type2_cuda_alias_source_compare);
  covered = 0;
  for (size_t i = 0; i < alias->source_count; i++) {
    CXLType2CudaAliasSource *source = &alias->sources[i];
    uint64_t source_end = source->destination_offset + source->length;

    if (source->destination_offset != covered ||
        source_end < source->destination_offset || source_end > total_bytes) {
      *reason = "alias-source-overlap";
      goto fail;
    }
    covered = source_end;
  }
  alias->owned_reservation =
      mmap(NULL, alias->owned_reservation_size, PROT_NONE,
           MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (alias->owned_reservation == MAP_FAILED) {
    alias->owned_reservation = NULL;
    *reason = "alias-reservation-failed";
    goto fail;
  }
  alias->mapping_base = alias->owned_reservation + page_size;
  alias->reservation = alias->mapping_base + alias->pointer_bias;

  if (!cxl_type2_cuda_alias_map_pages(alias, 0, total_bytes, reason)) {
    goto poison;
  }
  allocation->dax_backed = true;
  if (!cxl_type2_cuda_generation_publish(table, allocation, generation)) {
    *reason = table->first_generation_error ? table->first_generation_error
                                            : "alias-generation-publish-failed";
    goto poison;
  }
  for (CXLType2CudaPageableAlias *existing = allocation->pageable_alias;
       existing; existing = existing->next) {
    existing->content_generation = generation;
  }
  alias->next = allocation->pageable_alias;
  allocation->pageable_alias = alias;
  allocation->content_generation = generation;
  alias->device_alias = (uintptr_t)alias->reservation;
  allocation->device_alias = alias->device_alias;
  *device_alias = alias->device_alias;
  return true;

poison:
  allocation->poisoned = true;
fail:
  cxl_type2_cuda_pageable_alias_free(alias);
  return false;
}

bool cxl_type2_cuda_allocation_remove_pageable_aliases(
    CXLType2CudaAllocationTable *table, uint64_t base, uint64_t epoch,
    uint64_t destination_offset, uint64_t logical_bytes,
    size_t *removed_count) {
  CXLType2CudaAllocation *allocation =
      cxl_type2_cuda_allocation_find(table, base, epoch);
  CXLType2CudaPageableAlias **link;
  uint64_t end;

  if (removed_count) {
    *removed_count = 0;
  }
  if (!allocation || !removed_count || !logical_bytes ||
      destination_offset > UINT64_MAX - logical_bytes ||
      allocation->consumer_refs || allocation->normal_inflight_refs ||
      allocation->graph_binding_refs || allocation->graph_inflight_refs) {
    return false;
  }
  end = destination_offset + logical_bytes;
  link = &allocation->pageable_alias;
  while (*link) {
    CXLType2CudaPageableAlias *alias = *link;
    uint64_t alias_end = alias->destination_offset + alias->logical_bytes;

    if (alias_end <= destination_offset || alias->destination_offset >= end) {
      link = &alias->next;
      continue;
    }
    if (alias->host_registered) {
      return false;
    }
    *link = alias->next;
    alias->next = NULL;
    cxl_type2_cuda_pageable_alias_free(alias);
    (*removed_count)++;
  }
  allocation->device_alias =
      allocation->pageable_alias ? allocation->pageable_alias->device_alias : 0;
  return true;
}

bool cxl_type2_cuda_allocation_drop_pageable_alias(
    CXLType2CudaAllocationTable *table, uint64_t base, uint64_t epoch,
    uint64_t generation) {
  CXLType2CudaAllocation *allocation =
      cxl_type2_cuda_allocation_find(table, base, epoch);

  if (!allocation || allocation->poisoned || !allocation->pageable_alias ||
      allocation->content_generation != generation ||
      allocation->consumer_refs || allocation->normal_inflight_refs ||
      allocation->graph_binding_refs || allocation->graph_inflight_refs) {
    return false;
  }
  cxl_type2_cuda_allocation_alias_clear(allocation);
  return true;
}

bool cxl_type2_cuda_allocation_bind_graph_alias(
    CXLType2CudaAllocationTable *table, uint64_t base, uint64_t epoch,
    uint64_t device_alias) {
  CXLType2CudaAllocation *allocation =
      cxl_type2_cuda_allocation_find(table, base, epoch);

  if (!allocation || allocation->poisoned || !allocation->pageable_alias ||
      !device_alias || allocation->graph_binding_refs == UINT64_MAX) {
    return false;
  }
  allocation->graph_binding_refs++;
  return true;
}

bool cxl_type2_cuda_allocation_unbind_graph_alias(
    CXLType2CudaAllocationTable *table, uint64_t base, uint64_t epoch,
    uint64_t device_alias) {
  CXLType2CudaAllocation *allocation =
      cxl_type2_cuda_allocation_find(table, base, epoch);

  if (!allocation || !device_alias || !allocation->graph_binding_refs) {
    return false;
  }
  allocation->graph_binding_refs--;
  return true;
}

bool cxl_type2_cuda_allocation_bind_pageable_alias_for_address(
    CXLType2CudaAllocationTable *table, uint64_t address,
    CXLType2CudaAllocationIdentity *identity, uint64_t *generation,
    uint64_t *alias_address) {
  CXLType2CudaAllocation *allocation = NULL;
  CXLType2CudaPageableAlias *alias = NULL;
  uint64_t offset = 0;

  if (!table || !identity || !generation || !alias_address) {
    return false;
  }
  allocation = cxl_type2_cuda_allocation_containing(table, address, 1);
  if (allocation) {
    offset = address - allocation->base;
    alias = cxl_type2_cuda_allocation_pageable_alias_for_offset_mutable(
        allocation, offset);
  } else {
    for (size_t i = 0; i < table->count; i++) {
      CXLType2CudaAllocation *candidate = &table->entries[i];

      for (CXLType2CudaPageableAlias *current = candidate->pageable_alias;
           current; current = current->next) {
        if (current->device_alias && address >= current->device_alias &&
            address - current->device_alias < current->logical_bytes) {
          allocation = candidate;
          alias = current;
          offset =
              current->destination_offset + address - current->device_alias;
          break;
        }
      }
      if (allocation) {
        break;
      }
    }
  }
  if (!allocation || !alias || !alias->device_alias ||
      !cxl_type2_cuda_allocation_bind_graph_alias(
          table, allocation->base, allocation->epoch, alias->device_alias)) {
    return false;
  }
  *identity = (CXLType2CudaAllocationIdentity){
      .base = allocation->base,
      .epoch = allocation->epoch,
  };
  *generation = allocation->content_generation;
  *alias_address = alias->device_alias + offset - alias->destination_offset;
  return true;
}

bool cxl_type2_cuda_allocation_acquire_pageable_alias(
    CXLType2CudaAllocationTable *table, uint64_t base, uint64_t epoch,
    uint64_t generation, CXLType2CudaAliasConsumer consumer,
    uint64_t *device_alias) {
  CXLType2CudaAllocation *allocation =
      cxl_type2_cuda_allocation_find(table, base, epoch);
  uint64_t *refs;

  if (!allocation || !device_alias || allocation->poisoned ||
      !allocation->pageable_alias ||
      allocation->content_generation != generation) {
    return false;
  }
  if (consumer == CXL_TYPE2_CUDA_ALIAS_NORMAL) {
    refs = &allocation->normal_inflight_refs;
  } else if (consumer == CXL_TYPE2_CUDA_ALIAS_GRAPH &&
             allocation->graph_binding_refs) {
    refs = &allocation->graph_inflight_refs;
  } else {
    return false;
  }
  if (*refs == UINT64_MAX) {
    return false;
  }
  (*refs)++;
  *device_alias = allocation->pageable_alias->device_alias;
  return true;
}

bool cxl_type2_cuda_allocation_acquire_pageable_alias_for_address(
    CXLType2CudaAllocationTable *table, uint64_t address,
    CXLType2CudaAliasConsumer consumer,
    CXLType2CudaAllocationIdentity *identity, uint64_t *generation,
    uint64_t *alias_address) {
  CXLType2CudaAllocation *allocation;
  CXLType2CudaPageableAlias *alias;
  uint64_t offset;

  if (!table || !identity || !generation || !alias_address) {
    return false;
  }
  allocation = cxl_type2_cuda_allocation_containing(table, address, 1);
  if (!allocation || !allocation->pageable_alias ||
      address < allocation->base) {
    return false;
  }
  offset = address - allocation->base;
  alias = cxl_type2_cuda_allocation_pageable_alias_for_offset_mutable(
      allocation, offset);
  if (!alias || !alias->device_alias ||
      !cxl_type2_cuda_allocation_acquire_pageable_alias(
          table, allocation->base, allocation->epoch,
          allocation->content_generation, consumer, alias_address)) {
    return false;
  }
  *identity = (CXLType2CudaAllocationIdentity){
      .base = allocation->base,
      .epoch = allocation->epoch,
  };
  *generation = allocation->content_generation;
  *alias_address = alias->device_alias + offset - alias->destination_offset;
  return true;
}

bool cxl_type2_cuda_allocation_release_pageable_alias(
    CXLType2CudaAllocationTable *table, uint64_t base, uint64_t epoch,
    uint64_t generation, CXLType2CudaAliasConsumer consumer) {
  CXLType2CudaAllocation *allocation =
      cxl_type2_cuda_allocation_find(table, base, epoch);
  uint64_t *refs;

  if (!allocation || allocation->content_generation != generation) {
    return false;
  }
  if (consumer == CXL_TYPE2_CUDA_ALIAS_NORMAL) {
    refs = &allocation->normal_inflight_refs;
  } else if (consumer == CXL_TYPE2_CUDA_ALIAS_GRAPH) {
    refs = &allocation->graph_inflight_refs;
  } else {
    return false;
  }
  if (!*refs) {
    return false;
  }
  (*refs)--;
  return true;
}

bool cxl_type2_cuda_generation_population_complete(
    CXLType2CudaAllocationTable *table, uint64_t destination, uint64_t size) {
  CXLType2CudaAllocation *allocation =
      cxl_type2_cuda_allocation_containing(table, destination, size);
  CXLType2CudaGenerationRecord *record;
  size_t generation_index;

  if (!allocation) {
    if (table) {
      cxl_type2_cuda_generation_fail(table, "population-allocation-missing");
    }
    return false;
  }
  record = cxl_type2_cuda_generation_current(table, allocation, true);
  if (!record) {
    return false;
  }
  generation_index = record - table->generations;
  return cxl_type2_cuda_population_interval_add(
      table, generation_index, destination - allocation->base,
      destination - allocation->base + size);
}

bool cxl_type2_cuda_allocation_identity_for_address(
    const CXLType2CudaAllocationTable *table, uint64_t address,
    CXLType2CudaAllocationIdentity *identity) {
  CXLType2CudaAllocation *allocation;

  if (!identity) {
    return false;
  }
  allocation = cxl_type2_cuda_allocation_containing(
      (CXLType2CudaAllocationTable *)table, address, 1);
  if (!allocation) {
    return false;
  }
  *identity = (CXLType2CudaAllocationIdentity){
      .base = allocation->base,
      .epoch = allocation->epoch,
  };
  return true;
}

bool cxl_type2_cuda_allocation_identity_for_alias_address(
    const CXLType2CudaAllocationTable *table, uint64_t address,
    CXLType2CudaAllocationIdentity *identity, uint64_t *generation) {
  if (!table || !table->available || !identity || !generation) {
    return false;
  }
  for (size_t i = 0; i < table->count; i++) {
    const CXLType2CudaAllocation *allocation = &table->entries[i];

    for (const CXLType2CudaPageableAlias *alias = allocation->pageable_alias;
         alias; alias = alias->next) {
      if (!alias->device_alias || address < alias->device_alias ||
          address - alias->device_alias >= alias->logical_bytes) {
        continue;
      }
      *identity = (CXLType2CudaAllocationIdentity){
          .base = allocation->base,
          .epoch = allocation->epoch,
      };
      *generation = allocation->content_generation;
      return true;
    }
  }
  return false;
}

bool cxl_type2_cuda_allocation_alias_address_for_address(
    const CXLType2CudaAllocationTable *table, uint64_t address,
    uint64_t *alias_address) {
  CXLType2CudaAllocation *allocation;
  const CXLType2CudaPageableAlias *alias;
  uint64_t offset;

  if (!table || !alias_address) {
    return false;
  }
  allocation = cxl_type2_cuda_allocation_containing(
      (CXLType2CudaAllocationTable *)table, address, 1);
  if (!allocation) {
    for (size_t i = 0; i < table->count; i++) {
      const CXLType2CudaAllocation *candidate = &table->entries[i];

      for (alias = candidate->pageable_alias; alias; alias = alias->next) {
        if (alias->device_alias && address >= alias->device_alias &&
            address - alias->device_alias < alias->logical_bytes) {
          *alias_address = address;
          return true;
        }
      }
    }
    return false;
  }
  offset = address - allocation->base;
  alias =
      cxl_type2_cuda_allocation_pageable_alias_for_offset(allocation, offset);
  if (!alias || !alias->device_alias) {
    return false;
  }
  *alias_address = alias->device_alias + offset - alias->destination_offset;
  return true;
}

bool cxl_type2_cuda_allocation_identity_for_range(
    const CXLType2CudaAllocationTable *table, uint64_t address, uint64_t size,
    CXLType2CudaAllocationIdentity *identity, uint64_t *allocation_offset) {
  CXLType2CudaAllocation *allocation;

  if (!identity || !allocation_offset || !size) {
    return false;
  }
  allocation = cxl_type2_cuda_allocation_containing(
      (CXLType2CudaAllocationTable *)table, address, size);
  if (!allocation) {
    return false;
  }
  *identity = (CXLType2CudaAllocationIdentity){
      .base = allocation->base,
      .epoch = allocation->epoch,
  };
  *allocation_offset = address - allocation->base;
  return true;
}

bool cxl_type2_cuda_generation_consume(
    CXLType2CudaAllocationTable *table,
    const CXLType2CudaAllocationIdentity *identities, size_t count,
    uint32_t opcode, uint64_t call_id) {
  for (size_t i = 0; i < count; i++) {
    CXLType2CudaAllocation *allocation;
    CXLType2CudaGenerationRecord *record;
    bool duplicate = false;

    for (size_t j = 0; j < i; j++) {
      if (identities[j].base == identities[i].base &&
          identities[j].epoch == identities[i].epoch) {
        duplicate = true;
        break;
      }
    }
    if (duplicate) {
      continue;
    }
    allocation = cxl_type2_cuda_allocation_find(table, identities[i].base,
                                                identities[i].epoch);
    if (!allocation) {
      cxl_type2_cuda_generation_fail(table, "consumer-allocation-missing");
      return false;
    }
    record = cxl_type2_cuda_generation_current(table, allocation, false);
    if (!record || (allocation->dax_backed
                        ? record->direct_consumer_count == UINT64_MAX
                        : record->gpu_local_consumer_count == UINT64_MAX)) {
      cxl_type2_cuda_generation_fail(table, "consumer-count-overflow");
      return false;
    }
    if (allocation->dax_backed) {
      record->direct_consumer_count++;
    } else {
      record->gpu_local_consumer_count++;
    }
    if (!record->consumer_observed) {
      record->first_consumer_opcode = opcode;
      record->first_consumer_call_id = call_id;
      record->consumer_observed = true;
    }
    record->last_consumer_opcode = opcode;
    record->last_consumer_call_id = call_id;
  }
  return true;
}

bool cxl_type2_cuda_generation_prefetch_enqueue(
    CXLType2CudaAllocationTable *table, uint64_t address, uint64_t size,
    uint64_t enqueue_wall_ns) {
  CXLType2CudaAllocation *allocation =
      cxl_type2_cuda_allocation_containing(table, address, size);
  CXLType2CudaGenerationRecord *record;

  if (!allocation) {
    if (table) {
      cxl_type2_cuda_generation_fail(table, "prefetch-allocation-missing");
    }
    return false;
  }
  record = cxl_type2_cuda_generation_current(table, allocation, false);
  if (!record || record->prefetch_count == UINT64_MAX ||
      record->prefetch_requested_bytes > UINT64_MAX - size ||
      record->prefetch_enqueue_wall_ns > UINT64_MAX - enqueue_wall_ns) {
    cxl_type2_cuda_generation_fail(table, "prefetch-cost-overflow");
    return false;
  }
  record->prefetch_count++;
  record->prefetch_requested_bytes += size;
  record->prefetch_enqueue_wall_ns += enqueue_wall_ns;
  record->prefetch_completion_available = false;
  return true;
}

static CXLType2CudaGenerationRecord *
cxl_type2_cuda_generation_find(CXLType2CudaAllocationTable *table,
                               CXLType2CudaAllocationIdentity identity,
                               uint64_t generation) {
  if (!table || !table->available || !generation) {
    return NULL;
  }
  for (size_t i = table->generation_count; i > 0; i--) {
    CXLType2CudaGenerationRecord *record = &table->generations[i - 1];

    if (record->allocation_base == identity.base &&
        record->epoch == identity.epoch && record->generation == generation) {
      return record;
    }
  }
  return NULL;
}

bool cxl_type2_cuda_generation_prefetch_required(
    CXLType2CudaAllocationTable *table, CXLType2CudaAllocationIdentity identity,
    uint64_t generation, bool *required) {
  CXLType2CudaAllocation *allocation =
      cxl_type2_cuda_allocation_find(table, identity.base, identity.epoch);
  CXLType2CudaGenerationRecord *record =
      cxl_type2_cuda_generation_find(table, identity, generation);

  if (!required || !allocation || !allocation->pageable_alias || !record ||
      allocation->content_generation != generation ||
      allocation->pageable_alias->content_generation != generation) {
    if (table) {
      cxl_type2_cuda_generation_fail(table, "prefetch-generation-missing");
    }
    return false;
  }
  *required = record->prefetch_count == 0;
  return true;
}

bool cxl_type2_cuda_generation_prefetch_enqueue_for_identity(
    CXLType2CudaAllocationTable *table, CXLType2CudaAllocationIdentity identity,
    uint64_t generation, uint64_t size, uint64_t enqueue_wall_ns) {
  CXLType2CudaAllocation *allocation =
      cxl_type2_cuda_allocation_find(table, identity.base, identity.epoch);
  CXLType2CudaGenerationRecord *record =
      cxl_type2_cuda_generation_find(table, identity, generation);

  if (!allocation || !record || !size ||
      allocation->content_generation != generation) {
    if (table) {
      cxl_type2_cuda_generation_fail(table, "prefetch-generation-missing");
    }
    return false;
  }
  if (record->prefetch_count == UINT64_MAX ||
      record->prefetch_requested_bytes > UINT64_MAX - size ||
      record->prefetch_enqueue_wall_ns > UINT64_MAX - enqueue_wall_ns) {
    cxl_type2_cuda_generation_fail(table, "prefetch-cost-overflow");
    return false;
  }
  record->prefetch_count++;
  record->prefetch_requested_bytes += size;
  record->prefetch_enqueue_wall_ns += enqueue_wall_ns;
  record->prefetch_completion_available = false;
  return true;
}

bool cxl_type2_cuda_generation_prefetch_complete_for_identity(
    CXLType2CudaAllocationTable *table, CXLType2CudaAllocationIdentity identity,
    uint64_t generation, uint64_t size, uint64_t completion_wall_ns) {
  CXLType2CudaAllocation *allocation =
      cxl_type2_cuda_allocation_find(table, identity.base, identity.epoch);
  CXLType2CudaGenerationRecord *record =
      cxl_type2_cuda_generation_find(table, identity, generation);

  if (!allocation || !record || !record->prefetch_count || !size ||
      allocation->content_generation != generation ||
      record->promotion_count == UINT64_MAX ||
      record->promotion_bytes > UINT64_MAX - size ||
      record->promotion_wall_ns > UINT64_MAX - completion_wall_ns) {
    if (table) {
      cxl_type2_cuda_generation_fail(table, "prefetch-completion-invalid");
    }
    return false;
  }
  record->promotion_count++;
  record->promotion_bytes += size;
  record->promotion_wall_ns += completion_wall_ns;
  record->prefetch_completion_available = true;
  return true;
}

bool cxl_type2_cuda_generation_release(CXLType2CudaAllocationTable *table,
                                       uint64_t base) {
  size_t position;
  CXLType2CudaAllocation *allocation;

  if (!table || !table->available) {
    return false;
  }
  position = cxl_type2_cuda_allocation_lower_bound(table, base);
  if (position == table->count || table->entries[position].base != base) {
    cxl_type2_cuda_generation_fail(table, "release-allocation-missing");
    return false;
  }
  allocation = &table->entries[position];
  for (size_t i = table->generation_count; i > 0; i--) {
    CXLType2CudaGenerationRecord *record = &table->generations[i - 1];

    if (record->allocation_base == allocation->base &&
        record->epoch == allocation->epoch &&
        record->next_boundary == CXL_TYPE2_CUDA_GENERATION_OPEN) {
      record->next_boundary = CXL_TYPE2_CUDA_GENERATION_RELEASE;
      break;
    }
  }
  return true;
}

const char *cxl_type2_cuda_generation_boundary_name(
    CXLType2CudaGenerationBoundary boundary) {
  switch (boundary) {
  case CXL_TYPE2_CUDA_GENERATION_OPEN:
    return "open";
  case CXL_TYPE2_CUDA_GENERATION_POPULATION:
    return "population";
  case CXL_TYPE2_CUDA_GENERATION_RELEASE:
    return "release";
  default:
    return "invalid";
  }
}

typedef struct CXLType2CudaDestinationRange {
  uint64_t base;
  uint64_t size;
} CXLType2CudaDestinationRange;

static int cxl_type2_cuda_destination_range_compare(const void *left,
                                                    const void *right) {
  const CXLType2CudaDestinationRange *a = left;
  const CXLType2CudaDestinationRange *b = right;

  return a->base < b->base ? -1 : a->base > b->base;
}

static void cxl_type2_cuda_coverage_unknown(CXLType2CudaCoverageResult *result,
                                            CXLType2CudaRejectionReason reason,
                                            bool available) {
  result->kind = CXL_TYPE2_CUDA_COVERAGE_UNKNOWN;
  result->reason = reason;
  result->available = available;
}

void cxl_type2_cuda_destination_union_classify(
    const CXLType2CudaAllocationTable *table, const uint64_t *destinations,
    const size_t *sizes, size_t count, CXLType2CudaCoverageResult *result) {
  g_autofree CXLType2CudaDestinationRange *ranges = NULL;
  size_t allocation_index = 0;
  size_t active_allocation = SIZE_MAX;
  uint64_t covered_end = 0;
  bool whole = true;

  g_assert(result);
  *result = (CXLType2CudaCoverageResult){
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
    ranges[i] = (CXLType2CudaDestinationRange){
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
                   table->entries[allocation_index].size <=
               ranges[i].base) {
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

    const CXLType2CudaAllocation *allocation = &table->entries[first_overlap];
    uint64_t allocation_end = allocation->base + allocation->size;

    if (ranges[i].base < allocation->base || range_end > allocation_end) {
      cxl_type2_cuda_coverage_unknown(
          result, CXL_TYPE2_CUDA_REJECTION_ALLOCATION_MISSING, true);
      return;
    }
    if (active_allocation != SIZE_MAX && active_allocation != first_overlap) {
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
  result->kind =
      whole ? CXL_TYPE2_CUDA_COVERAGE_WHOLE : CXL_TYPE2_CUDA_COVERAGE_PARTIAL;
  result->reason = whole ? CXL_TYPE2_CUDA_REJECTION_NONE
                         : CXL_TYPE2_CUDA_REJECTION_PARTIAL_COVERAGE;
  result->available = true;
}

CXLType2CudaCommandRole cxl_type2_cuda_command_role(uint32_t command) {
  if (command >= G_N_ELEMENTS(cxl_type2_cuda_command_roles) ||
      !cxl_type2_cuda_command_roles[command]) {
    return CXL_TYPE2_CUDA_COMMAND_UNKNOWN;
  }
  return cxl_type2_cuda_command_roles[command];
}

const char *cxl_type2_cuda_command_role_name(CXLType2CudaCommandRole role) {
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

static bool cxl_type2_cuda_opcode_count_add(uint64_t *value, uint64_t addend) {
  if (*value > UINT64_MAX - addend) {
    return false;
  }
  *value += addend;
  return true;
}

bool cxl_type2_cuda_opcode_summary_build(const uint64_t command_counts[256],
                                         char *records, size_t records_capacity,
                                         CXLType2CudaOpcodeSummary *summary) {
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
        !cxl_type2_cuda_opcode_count_add(&summary->reader_commands, count)) {
      valid = false;
    }
    if ((role & CXL_TYPE2_CUDA_COMMAND_WRITER) &&
        !cxl_type2_cuda_opcode_count_add(&summary->writer_commands, count)) {
      valid = false;
    }
    if (role == CXL_TYPE2_CUDA_COMMAND_LIFECYCLE &&
        !cxl_type2_cuda_opcode_count_add(&summary->lifecycle_commands, count)) {
      valid = false;
    }
    if (role == CXL_TYPE2_CUDA_COMMAND_NO_CHANGE &&
        !cxl_type2_cuda_opcode_count_add(&summary->no_change_commands, count)) {
      valid = false;
    }
    if (role == CXL_TYPE2_CUDA_COMMAND_UNKNOWN &&
        !cxl_type2_cuda_opcode_count_add(&summary->unknown_commands, count)) {
      valid = false;
    }
    if (!summary->first_incomplete_command_valid) {
      summary->first_incomplete_command = command;
      summary->first_incomplete_command_valid = true;
    }
    summary->estimated_materialize_complete = false;
    written = snprintf(records + used, records_capacity - used,
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

const char *
cxl_type2_cuda_classifier_status_name(CXLType2CudaClassifierStatus status) {
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

const char *
cxl_type2_cuda_rejection_reason_name(CXLType2CudaRejectionReason reason) {
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
                                        uint64_t token, uint64_t active_epoch) {
  return active_case && live_context && active_epoch != 0 &&
         token == active_epoch;
}

bool cxl_type2_cuda_dispatch_attribute(uint64_t wire_value,
                                       CXLType2CudaAttributeQuery query,
                                       void *opaque, int *query_result) {
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
                                      void *opaque, int *query_result) {
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
    void *opaque, uint64_t *htod_delta, bool *mapping_invalidated) {
  int result;

  if (!ops || !ops->synchronize || !ops->htod_calls || !ops->unregister_host ||
      !htod_delta || !mapping_invalidated) {
    return CXL_GPU_ERROR_INVALID_VALUE;
  }
  *htod_delta = 0;
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
  return CXL_GPU_SUCCESS;
}

bool cxl_gpu_batch_htod_validate(const uint8_t *payload,
                                 uint64_t payload_capacity,
                                 uint64_t expected_range_count,
                                 uint64_t expected_payload_bytes,
                                 uint64_t *fail_idx) {
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
      expected_range_count > (expected_payload_bytes - sizeof(header)) /
                                 sizeof(CXLGPUBatchHtoDRange)) {
    return false;
  }

  source_offset =
      sizeof(header) + expected_range_count * sizeof(CXLGPUBatchHtoDRange);
  if (source_offset > UINT64_MAX - 63) {
    return false;
  }
  source_offset = (source_offset + 63) & ~UINT64_C(63);
  if (source_offset > expected_payload_bytes) {
    return false;
  }

  for (uint64_t i = 0; i < expected_range_count; i++) {
    CXLGPUBatchHtoDRange range;

    memcpy(&range, payload + sizeof(header) + i * sizeof(range), sizeof(range));
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
                               uint64_t *successfully_enqueued) {
  if (!payload || !enqueue || !fail_idx || !successfully_enqueued) {
    return CXL_GPU_ERROR_INVALID_VALUE;
  }
  *fail_idx = SIZE_MAX;
  *successfully_enqueued = 0;
  for (uint64_t i = 0; i < range_count; i++) {
    CXLGPUBatchHtoDRange range;
    int result;

    memcpy(&range, payload + sizeof(CXLGPUBatchHtoDHeader) + i * sizeof(range),
           sizeof(range));
    result = enqueue(opaque, range.destination, payload + range.source_offset,
                     range.size);
    if (result != CXL_GPU_SUCCESS) {
      *fail_idx = i;
      return result;
    }
    (*successfully_enqueued)++;
  }
  return CXL_GPU_SUCCESS;
}

int cxl_gpu_batch_htod_submit(const uint8_t *payload, uint64_t payload_capacity,
                              uint64_t range_count, uint64_t payload_bytes,
                              CXLGPUBatchHtoDEnqueue enqueue, void *opaque,
                              uint64_t *fail_idx,
                              uint64_t *successfully_enqueued) {
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
                                      uint64_t *fail_index) {
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
      header.range_count > (payload_bytes - sizeof(header)) /
                               sizeof(CXLGPUSourceVirtualRangeV1)) {
    return false;
  }
  ranges_bytes =
      (uint64_t)header.range_count * sizeof(CXLGPUSourceVirtualRangeV1);
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
                                   uint64_t range_count, uint64_t payload_bytes,
                                   uint64_t *fail_index) {
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
