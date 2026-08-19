#ifndef HW_CXL_TYPE2_CUDA_CONTRACT_H
#define HW_CXL_TYPE2_CUDA_CONTRACT_H

#include <stdbool.h>
#include <stddef.h>
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
const char *
cxl_type2_descriptor_verdict_reason(CXLType2DescriptorRequestVerdict verdict);

bool cxl_type2_cuda_decode_attribute(uint64_t wire_value, int32_t *attribute);
bool cxl_type2_cuda_attribute_wire_is_valid(uint64_t wire_value);
bool cxl_type2_cuda_stream_progress_wire(uint32_t command,
                                         const uint64_t params[8],
                                         uint64_t *stream_wire);
bool cxl_type2_cuda_adjacent_stream_sync_can_elide(
    bool previous_command_was_successful_sync, uint64_t previous_stream_wire,
    uint64_t current_stream_wire);
bool cxl_type2_cuda_decode_stream_sync_reason(
    uint32_t descriptor_protocol_version, uint64_t wire_reason,
    CXLGPUStreamSyncReason *reason);
bool cxl_type2_cuda_special_stream_from_wire(uint64_t wire,
                                             void *per_thread_stream,
                                             void **stream);
bool cxl_type2_cuda_mem_info_is_allowed(bool active_case, bool live_context,
                                        uint64_t token, uint64_t active_epoch);

typedef int (*CXLType2CudaAttributeQuery)(void *opaque, int32_t attribute);
typedef int (*CXLType2CudaMemInfoQuery)(void *opaque);
typedef int (*CXLGPUBatchHtoDEnqueue)(void *opaque, uint64_t destination,
                                      const void *source, size_t size);
typedef struct CXLType2CoherentUnmapOps {
  int (*synchronize)(void *opaque);
  uint64_t (*htod_calls)(void *opaque);
  int (*unregister_host)(void *opaque);
} CXLType2CoherentUnmapOps;

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
int cxl_type2_coherent_unmap_execute(
    bool host_registered, uint64_t stored_alias, uint64_t requested_alias,
    uint64_t htod_calls_at_map, const CXLType2CoherentUnmapOps *ops,
    void *opaque, uint64_t *htod_delta, bool *mapping_invalidated);
bool cxl_gpu_batch_htod_validate(const uint8_t *payload,
                                 uint64_t payload_capacity,
                                 uint64_t expected_range_count,
                                 uint64_t expected_payload_bytes,
                                 uint64_t *fail_idx);
int cxl_gpu_batch_htod_enqueue(const uint8_t *payload, uint64_t range_count,
                               CXLGPUBatchHtoDEnqueue enqueue, void *opaque,
                               uint64_t *fail_idx,
                               uint64_t *successfully_enqueued);
int cxl_gpu_batch_htod_submit(const uint8_t *payload, uint64_t payload_capacity,
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
                                   uint64_t range_count, uint64_t payload_bytes,
                                   uint64_t *fail_index);
int cxl_gpu_direct_host_address_order(uintptr_t left, uintptr_t right);
bool cxl_gpu_direct_host_range_follows(uintptr_t base, uint64_t length,
                                       uintptr_t next, uint64_t next_length);
bool cxl_gpu_direct_epoch_is_cross_case(uint64_t last_case_epoch,
                                        uint64_t active_case_epoch);
bool cxl_gpu_direct_registration_group_follows(uintptr_t mapping,
                                               uintptr_t base, uint64_t length,
                                               uintptr_t next_mapping,
                                               uintptr_t next,
                                               uint64_t next_length);
bool cxl_gpu_direct_copy_span_follows(uintptr_t source, uintptr_t registration,
                                      uintptr_t host, uint64_t destination,
                                      uint64_t length, uintptr_t next_source,
                                      uintptr_t next_registration,
                                      uintptr_t next_host,
                                      uint64_t next_destination,
                                      uint64_t next_length);
bool cxl_gpu_direct_destinations_are_independent(const uint64_t *destinations,
                                                 const size_t *sizes,
                                                 size_t count,
                                                 size_t *conflict_index);
uint64_t cxl_gpu_direct_registration_length(
    uint64_t mapping_offset, uint64_t mapping_length, uint64_t request_offset,
    uint64_t request_length, uint64_t following_offset, uint64_t tile_size,
    uint64_t padding_budget);
typedef struct CXLType2CudaAllocation {
  uint64_t base;
  uint64_t size;
  uint64_t epoch;
  uint64_t content_generation;
  uint64_t device_alias;
  uint64_t consumer_refs;
  uint64_t normal_inflight_refs;
  uint64_t graph_binding_refs;
  uint64_t graph_inflight_refs;
  struct CXLType2CudaPageableAlias *pageable_alias;
  bool poisoned;
  bool dax_backed;
} CXLType2CudaAllocation;

typedef struct CXLType2CudaAliasSource {
  int fd;
  const void *host_copy_source;
  void *owned_host_copy;
  uint64_t file_offset;
  uint64_t destination_offset;
  uint64_t length;
  uint64_t mapping_generation;
  uint64_t source_call_id;
  uint64_t logical_cxl_offset;
  uint64_t stat_device;
  uint64_t stat_inode;
  uint64_t stat_size;
  uint32_t stat_mode;
  bool readonly;
} CXLType2CudaAliasSource;

typedef struct CXLType2CudaPageableAlias {
  struct CXLType2CudaPageableAlias *next;
  void *owned_reservation;
  uint64_t owned_reservation_size;
  void *mapping_base;
  uint64_t mapping_size;
  void *reservation;
  uint64_t reservation_size;
  uint64_t pointer_bias;
  uint64_t device_alias;
  uint64_t destination_offset;
  uint64_t logical_bytes;
  uint64_t guard_bytes;
  uint64_t content_generation;
  uint64_t *contributing_source_call_ids;
  size_t contributing_source_call_count;
  uint64_t file_mapped_bytes;
  uint64_t host_composition_copy_bytes;
  uint64_t derived_boundary_copy_bytes;
  uint64_t boundary_composition_wall_ns;
  size_t derived_boundary_pages;
  CXLType2CudaAliasSource *sources;
  size_t source_count;
  bool host_registered;
} CXLType2CudaPageableAlias;

typedef enum CXLType2CudaAliasConsumer {
  CXL_TYPE2_CUDA_ALIAS_NORMAL,
  CXL_TYPE2_CUDA_ALIAS_GRAPH,
} CXLType2CudaAliasConsumer;

typedef enum CXLType2CudaGenerationBoundary {
  CXL_TYPE2_CUDA_GENERATION_OPEN,
  CXL_TYPE2_CUDA_GENERATION_POPULATION,
  CXL_TYPE2_CUDA_GENERATION_RELEASE,
} CXLType2CudaGenerationBoundary;

typedef struct CXLType2CudaGenerationRecord {
  uint64_t allocation_base;
  uint64_t epoch;
  uint64_t generation;
  uint64_t unique_visible_population_bytes;
  uint64_t direct_consumer_count;
  uint64_t gpu_local_consumer_count;
  uint64_t prefetch_count;
  uint64_t prefetch_requested_bytes;
  uint64_t prefetch_enqueue_wall_ns;
  uint64_t promotion_count;
  uint64_t promotion_bytes;
  uint64_t promotion_wall_ns;
  uint64_t first_consumer_call_id;
  uint64_t last_consumer_call_id;
  uint32_t first_consumer_opcode;
  uint32_t last_consumer_opcode;
  bool file_backed_system_uva;
  bool consumer_observed;
  bool prefetch_completion_available;
  CXLType2CudaGenerationBoundary next_boundary;
} CXLType2CudaGenerationRecord;

typedef struct CXLType2CudaPopulationInterval {
  size_t generation_index;
  uint64_t begin;
  uint64_t end;
} CXLType2CudaPopulationInterval;

typedef struct CXLType2CudaAllocationIdentity {
  uint64_t base;
  uint64_t epoch;
} CXLType2CudaAllocationIdentity;

typedef struct CXLType2CudaAllocationTable {
  CXLType2CudaAllocation *entries;
  size_t count;
  size_t capacity;
  size_t peak_count;
  uint64_t next_epoch;
  CXLType2CudaGenerationRecord *generations;
  size_t generation_count;
  size_t generation_capacity;
  CXLType2CudaPopulationInterval *population_intervals;
  size_t population_interval_count;
  size_t population_interval_capacity;
  bool available;
  const char *first_generation_error;
} CXLType2CudaAllocationTable;

typedef enum CXLType2CudaCoverageKind {
  CXL_TYPE2_CUDA_COVERAGE_WHOLE,
  CXL_TYPE2_CUDA_COVERAGE_PARTIAL,
  CXL_TYPE2_CUDA_COVERAGE_CROSS,
  CXL_TYPE2_CUDA_COVERAGE_UNKNOWN,
} CXLType2CudaCoverageKind;

typedef enum CXLType2CudaClassifierStatus {
  CXL_TYPE2_CUDA_CLASSIFIER_AVAILABLE,
  CXL_TYPE2_CUDA_CLASSIFIER_CONTRADICTED,
  CXL_TYPE2_CUDA_CLASSIFIER_UNAVAILABLE,
} CXLType2CudaClassifierStatus;

typedef enum CXLType2CudaRejectionReason {
  CXL_TYPE2_CUDA_REJECTION_NONE,
  CXL_TYPE2_CUDA_REJECTION_ALLOCATION_MISSING,
  CXL_TYPE2_CUDA_REJECTION_PARTIAL_COVERAGE,
  CXL_TYPE2_CUDA_REJECTION_CROSS_ALLOCATION,
  CXL_TYPE2_CUDA_REJECTION_DESTINATION_OVERLAP,
  CXL_TYPE2_CUDA_REJECTION_DESTINATION_OVERFLOW,
  CXL_TYPE2_CUDA_REJECTION_GRAPH_NON_KERNEL,
  CXL_TYPE2_CUDA_REJECTION_GRAPH_INCOMPLETE,
  CXL_TYPE2_CUDA_REJECTION_OPCODE_UNKNOWN,
} CXLType2CudaRejectionReason;

typedef enum CXLType2CudaCommandRole {
  CXL_TYPE2_CUDA_COMMAND_READER = 1U << 0,
  CXL_TYPE2_CUDA_COMMAND_WRITER = 1U << 1,
  CXL_TYPE2_CUDA_COMMAND_READER_WRITER =
      CXL_TYPE2_CUDA_COMMAND_READER | CXL_TYPE2_CUDA_COMMAND_WRITER,
  CXL_TYPE2_CUDA_COMMAND_LIFECYCLE = 1U << 2,
  CXL_TYPE2_CUDA_COMMAND_NO_CHANGE = 1U << 3,
  CXL_TYPE2_CUDA_COMMAND_UNKNOWN = 1U << 4,
} CXLType2CudaCommandRole;

typedef struct CXLType2CudaCoverageResult {
  CXLType2CudaCoverageKind kind;
  CXLType2CudaRejectionReason reason;
  uint64_t bytes;
  bool available;
} CXLType2CudaCoverageResult;

#define CXL_TYPE2_CUDA_OPCODE_RECORDS_CAPACITY (256 * 64)

typedef struct CXLType2CudaOpcodeSummary {
  uint64_t reader_commands;
  uint64_t writer_commands;
  uint64_t lifecycle_commands;
  uint64_t no_change_commands;
  uint64_t unknown_commands;
  uint64_t estimated_materialize_bytes;
  bool estimated_materialize_complete;
  uint32_t first_incomplete_command;
  bool first_incomplete_command_valid;
} CXLType2CudaOpcodeSummary;

void cxl_type2_cuda_allocation_table_init(CXLType2CudaAllocationTable *table);
void cxl_type2_cuda_allocation_table_reset(CXLType2CudaAllocationTable *table);
void cxl_type2_cuda_allocation_table_destroy(
    CXLType2CudaAllocationTable *table);
bool cxl_type2_cuda_allocation_record(CXLType2CudaAllocationTable *table,
                                      uint64_t base, uint64_t size,
                                      uint64_t *epoch);
bool cxl_type2_cuda_allocation_forget(CXLType2CudaAllocationTable *table,
                                      uint64_t base);
CXLType2CudaAllocation *
cxl_type2_cuda_allocation_find(CXLType2CudaAllocationTable *table,
                               uint64_t base, uint64_t epoch);
bool cxl_type2_cuda_allocation_publish_alias(CXLType2CudaAllocationTable *table,
                                             uint64_t base, uint64_t epoch,
                                             uint64_t generation,
                                             uint64_t device_alias);
bool cxl_type2_cuda_allocation_acquire_alias(CXLType2CudaAllocationTable *table,
                                             uint64_t base, uint64_t epoch,
                                             uint64_t generation,
                                             uint64_t *device_alias);
bool cxl_type2_cuda_allocation_release_alias(CXLType2CudaAllocationTable *table,
                                             uint64_t base, uint64_t epoch,
                                             uint64_t generation);
bool cxl_type2_cuda_allocation_materialize(CXLType2CudaAllocationTable *table,
                                           uint64_t base, uint64_t epoch,
                                           uint64_t generation);
bool cxl_type2_cuda_allocation_map_pageable_alias(
    CXLType2CudaAllocationTable *table, uint64_t base, uint64_t epoch,
    uint64_t generation, const CXLType2CudaAliasSource *sources,
    size_t source_count, const uint64_t *contributing_source_call_ids,
    size_t contributing_source_call_count, uint64_t destination_offset,
    uint64_t logical_bytes, uint64_t guard_bytes, uint64_t *device_alias,
    const char **reason);
bool cxl_type2_cuda_allocation_remove_pageable_aliases(
    CXLType2CudaAllocationTable *table, uint64_t base, uint64_t epoch,
    uint64_t destination_offset, uint64_t logical_bytes, size_t *removed_count);
bool cxl_type2_cuda_pageable_alias_contains(
    const CXLType2CudaPageableAlias *alias, uint64_t allocation_offset);
const CXLType2CudaPageableAlias *
cxl_type2_cuda_allocation_pageable_alias_for_offset(
    const CXLType2CudaAllocation *allocation, uint64_t allocation_offset);
CXLType2CudaPageableAlias *
cxl_type2_cuda_allocation_pageable_alias_for_offset_mutable(
    CXLType2CudaAllocation *allocation, uint64_t allocation_offset);
bool cxl_type2_cuda_allocation_alias_address_for_address(
    const CXLType2CudaAllocationTable *table, uint64_t address,
    uint64_t *alias_address);
bool cxl_type2_cuda_allocation_drop_pageable_alias(
    CXLType2CudaAllocationTable *table, uint64_t base, uint64_t epoch,
    uint64_t generation);
bool cxl_type2_cuda_allocation_bind_graph_alias(
    CXLType2CudaAllocationTable *table, uint64_t base, uint64_t epoch,
    uint64_t device_alias);
bool cxl_type2_cuda_allocation_unbind_graph_alias(
    CXLType2CudaAllocationTable *table, uint64_t base, uint64_t epoch,
    uint64_t device_alias);
bool cxl_type2_cuda_allocation_bind_pageable_alias_for_address(
    CXLType2CudaAllocationTable *table, uint64_t address,
    CXLType2CudaAllocationIdentity *identity, uint64_t *generation,
    uint64_t *alias_address);
bool cxl_type2_cuda_allocation_acquire_pageable_alias(
    CXLType2CudaAllocationTable *table, uint64_t base, uint64_t epoch,
    uint64_t generation, CXLType2CudaAliasConsumer consumer,
    uint64_t *device_alias);
bool cxl_type2_cuda_allocation_acquire_pageable_alias_for_address(
    CXLType2CudaAllocationTable *table, uint64_t address,
    CXLType2CudaAliasConsumer consumer,
    CXLType2CudaAllocationIdentity *identity, uint64_t *generation,
    uint64_t *alias_address);
bool cxl_type2_cuda_allocation_release_pageable_alias(
    CXLType2CudaAllocationTable *table, uint64_t base, uint64_t epoch,
    uint64_t generation, CXLType2CudaAliasConsumer consumer);
bool cxl_type2_cuda_generation_population_complete(
    CXLType2CudaAllocationTable *table, uint64_t destination, uint64_t size);
bool cxl_type2_cuda_generation_consume(
    CXLType2CudaAllocationTable *table,
    const CXLType2CudaAllocationIdentity *identities, size_t count,
    uint32_t opcode, uint64_t call_id);
bool cxl_type2_cuda_generation_prefetch_enqueue(
    CXLType2CudaAllocationTable *table, uint64_t address, uint64_t size,
    uint64_t enqueue_wall_ns);
bool cxl_type2_cuda_generation_prefetch_required(
    CXLType2CudaAllocationTable *table, CXLType2CudaAllocationIdentity identity,
    uint64_t generation, bool *required);
bool cxl_type2_cuda_generation_prefetch_enqueue_for_identity(
    CXLType2CudaAllocationTable *table, CXLType2CudaAllocationIdentity identity,
    uint64_t generation, uint64_t size, uint64_t enqueue_wall_ns);
bool cxl_type2_cuda_generation_prefetch_complete_for_identity(
    CXLType2CudaAllocationTable *table, CXLType2CudaAllocationIdentity identity,
    uint64_t generation, uint64_t size, uint64_t completion_wall_ns);
bool cxl_type2_cuda_generation_release(CXLType2CudaAllocationTable *table,
                                       uint64_t base);
bool cxl_type2_cuda_allocation_identity_for_address(
    const CXLType2CudaAllocationTable *table, uint64_t address,
    CXLType2CudaAllocationIdentity *identity);
bool cxl_type2_cuda_allocation_identity_for_alias_address(
    const CXLType2CudaAllocationTable *table, uint64_t address,
    CXLType2CudaAllocationIdentity *identity, uint64_t *generation);
bool cxl_type2_cuda_allocation_identity_for_range(
    const CXLType2CudaAllocationTable *table, uint64_t address, uint64_t size,
    CXLType2CudaAllocationIdentity *identity, uint64_t *allocation_offset);
const char *cxl_type2_cuda_generation_boundary_name(
    CXLType2CudaGenerationBoundary boundary);
void cxl_type2_cuda_destination_union_classify(
    const CXLType2CudaAllocationTable *table, const uint64_t *destinations,
    const size_t *sizes, size_t count, CXLType2CudaCoverageResult *result);
CXLType2CudaCommandRole cxl_type2_cuda_command_role(uint32_t command);
const char *cxl_type2_cuda_command_role_name(CXLType2CudaCommandRole role);
bool cxl_type2_cuda_opcode_summary_build(const uint64_t command_counts[256],
                                         char *records, size_t records_capacity,
                                         CXLType2CudaOpcodeSummary *summary);
const char *
cxl_type2_cuda_classifier_status_name(CXLType2CudaClassifierStatus status);
const char *
cxl_type2_cuda_rejection_reason_name(CXLType2CudaRejectionReason reason);

#endif
