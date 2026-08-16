/*
 * Bounded production-path gate for the CXL Type-2 pageable model alias.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/cutils.h"
#include "qemu/timer.h"
#include "qemu/units.h"
#include "qapi/error.h"
#include "qobject/json-writer.h"
#include "hw/cxl/cxl_type2.h"
#include "hw/cxl/cxl_type2_model_alias_gate.h"
#include "hw/virtio/vhost-user-fs.h"
#include "standard-headers/linux/virtio_fs.h"

typedef enum GateStatus {
    GATE_PASS,
    GATE_FAIL,
    GATE_UNAVAILABLE,
    GATE_NOT_RUN,
} GateStatus;

typedef struct GatePredicate {
    const char *name;
    bool required;
    GateStatus status;
    int result;
    const char *reason;
    const char *blocking;
} GatePredicate;

typedef struct GateState {
    GatePredicate predicates[32];
    size_t predicate_count;
    const char *first_failure;
    int first_result;
    uint64_t mapping_generation;
    uint64_t source_device;
    uint64_t source_inode;
    uint64_t source_size;
    uint32_t source_mode;
    uint64_t allocation_epoch;
    uint64_t alias_address_generation_one;
    uint64_t alias_address_generation_two;
    uint64_t alias_build_wall_ns_generation_one;
    uint64_t alias_build_wall_ns_generation_two;
    uint64_t boundary_composition_wall_ns_generation_one;
    uint64_t boundary_composition_wall_ns_generation_two;
    uint64_t logical_bytes;
    uint64_t guard_bytes;
    uint64_t file_mapped_bytes;
    uint64_t derived_boundary_pages;
    uint64_t derived_boundary_copy_bytes;
    uint64_t cxl_latency_ns;
    CXLType2MemoryRange vfio_dma_ranges[2];
    size_t vfio_dma_range_count;
    uint64_t normal_generation_one;
    uint64_t normal_generation_two;
    uint64_t graph_generation_one;
    uint64_t graph_generation_two;
    char source_sha256_before[65];
    char source_sha256_after[65];
} GateState;

static const char gate_ptx[] =
    ".version 6.0\n"
    ".target sm_52\n"
    ".address_size 64\n"
    ".visible .entry cxl_model_alias_read(\n"
    " .param .u64 root, .param .u64 result,\n"
    " .param .u64 page_size, .param .u64 total_bytes) {\n"
    " .reg .b64 %rd<20>;\n"
    " ld.param.u64 %rd1,[root];\n"
    " ld.param.u64 %rd2,[result];\n"
    " ld.param.u64 %rd3,[page_size];\n"
    " ld.param.u64 %rd4,[total_bytes];\n"
    " ld.global.u64 %rd5,[%rd1];\n"
    " sub.u64 %rd6,%rd3,8; add.u64 %rd7,%rd1,%rd6;\n"
    " ld.global.u64 %rd8,[%rd7];\n"
    " add.u64 %rd9,%rd1,%rd3; ld.global.u64 %rd10,[%rd9];\n"
    " add.u64 %rd11,%rd9,%rd6; ld.global.u64 %rd12,[%rd11];\n"
    " add.u64 %rd13,%rd9,%rd3; ld.global.u64 %rd14,[%rd13];\n"
    " sub.u64 %rd15,%rd4,8; add.u64 %rd16,%rd1,%rd15;\n"
    " ld.global.u64 %rd17,[%rd16];\n"
    " xor.b64 %rd18,%rd5,%rd8; xor.b64 %rd18,%rd18,%rd10;\n"
    " xor.b64 %rd18,%rd18,%rd12; xor.b64 %rd18,%rd18,%rd14;\n"
    " xor.b64 %rd18,%rd18,%rd17; st.global.u64 [%rd2],%rd18;\n"
    " ret;\n"
    "}\n"
    ".visible .entry cxl_model_alias_write(.param .u64 root) {\n"
    " .reg .b64 %rd1; ld.param.u64 %rd1,[root];\n"
    " st.global.u32 [%rd1],0; ret;\n"
    "}\n";

static const char *gate_status_name(GateStatus status)
{
    static const char *const names[] = {
        [GATE_PASS] = "pass",
        [GATE_FAIL] = "fail",
        [GATE_UNAVAILABLE] = "unavailable",
        [GATE_NOT_RUN] = "not-run",
    };

    return names[status];
}

static void gate_record(GateState *state, const char *name, bool required,
                        GateStatus status, int result, const char *reason,
                        const char *blocking)
{
    g_assert(state->predicate_count < ARRAY_SIZE(state->predicates));
    state->predicates[state->predicate_count++] = (GatePredicate) {
        .name = name,
        .required = required,
        .status = status,
        .result = result,
        .reason = reason,
        .blocking = blocking,
    };
    if (required && status != GATE_PASS && !state->first_failure) {
        state->first_failure = name;
        state->first_result = result ? result : ENOTSUP;
    }
}

static bool gate_hash_file_range(int fd, uint64_t offset, uint64_t length,
                                 char output[65])
{
    g_autoptr(GChecksum) checksum = g_checksum_new(G_CHECKSUM_SHA256);
    uint8_t buffer[64 * KiB];

    while (length) {
        size_t chunk = MIN(length, sizeof(buffer));
        ssize_t received = pread(fd, buffer, chunk, offset);

        if (received != chunk) {
            return false;
        }
        g_checksum_update(checksum, buffer, chunk);
        offset += chunk;
        length -= chunk;
    }
    pstrcpy(output, 65, g_checksum_get_string(checksum));
    return true;
}

static CXLType2CudaAliasSource gate_source(
    int fd, const struct stat *source_stat, uint64_t file_offset,
    uint64_t destination_offset, uint64_t length,
    uint64_t mapping_generation, uint64_t logical_cxl_offset)
{
    return (CXLType2CudaAliasSource) {
        .fd = fd,
        .file_offset = file_offset,
        .destination_offset = destination_offset,
        .length = length,
        .mapping_generation = mapping_generation,
        .logical_cxl_offset = logical_cxl_offset,
        .stat_device = source_stat->st_dev,
        .stat_inode = source_stat->st_ino,
        .stat_size = source_stat->st_size,
        .stat_mode = source_stat->st_mode,
        .readonly = true,
    };
}

static void gate_sources(CXLType2CudaAliasSource sources[4], int fd,
                         const struct stat *source_stat, uint64_t file_base,
                         uint64_t logical_base, uint64_t mapping_generation,
                         size_t page_size)
{
    sources[0] = gate_source(fd, source_stat, file_base + 32, 0,
                             page_size - 64, mapping_generation,
                             logical_base + 32);
    sources[1] = gate_source(fd, source_stat,
                             file_base + 2 * page_size + 17,
                             page_size - 64, 128, mapping_generation,
                             logical_base + 2 * page_size + 17);
    sources[2] = gate_source(fd, source_stat,
                             file_base + 3 * page_size + 64,
                             page_size + 64, page_size - 64,
                             mapping_generation,
                             logical_base + 3 * page_size + 64);
    sources[3] = gate_source(fd, source_stat, file_base + 4 * page_size,
                             2 * page_size, page_size, mapping_generation,
                             logical_base + 4 * page_size);
}

static uint64_t gate_expected(const uint8_t *alias, uint64_t page_size,
                              uint64_t total_bytes)
{
    uint64_t values[6];

    memcpy(&values[0], alias, sizeof(values[0]));
    memcpy(&values[1], alias + page_size - 8, sizeof(values[1]));
    memcpy(&values[2], alias + page_size, sizeof(values[2]));
    memcpy(&values[3], alias + 2 * page_size - 8, sizeof(values[3]));
    memcpy(&values[4], alias + 2 * page_size, sizeof(values[4]));
    memcpy(&values[5], alias + total_bytes - 8, sizeof(values[5]));
    return values[0] ^ values[1] ^ values[2] ^ values[3] ^ values[4] ^
           values[5];
}

static int gate_launch_read(HetGPUState *hetgpu, HetGPUFunction function,
                            HetGPUStream stream, uint64_t alias,
                            uint64_t result, uint64_t page_size,
                            uint64_t total_bytes)
{
    HetGPULaunchConfig config = {
        .grid_dim = { 1, 1, 1 },
        .block_dim = { 1, 1, 1 },
        .stream = stream,
    };
    void *args[] = { &alias, &result, &page_size, &total_bytes };

    return hetgpu_launch_kernel(hetgpu, function, &config, args,
                                G_N_ELEMENTS(args));
}

static int gate_launch_write(HetGPUState *hetgpu, HetGPUFunction function,
                             HetGPUStream stream, uint64_t alias)
{
    HetGPULaunchConfig config = {
        .grid_dim = { 1, 1, 1 },
        .block_dim = { 1, 1, 1 },
        .stream = stream,
    };
    void *args[] = { &alias };

    return hetgpu_launch_kernel(hetgpu, function, &config, args,
                                G_N_ELEMENTS(args));
}

static bool gate_run_normal(CXLType2State *ct2d, uint64_t base,
                            uint64_t epoch, uint64_t generation,
                            HetGPUFunction function, HetGPUStream stream,
                            HetGPUDevicePtr result_ptr, uint64_t page_size,
                            uint64_t total_bytes, uint64_t *observed)
{
    uint64_t alias;
    int result;

    if (!cxl_type2_cuda_allocation_acquire_pageable_alias(
            &ct2d->cuda_allocations, base, epoch, generation,
            CXL_TYPE2_CUDA_ALIAS_NORMAL, &alias)) {
        return false;
    }
    result = gate_launch_read(&ct2d->gpu_info.hetgpu_state, function, stream,
                              alias, result_ptr, page_size, total_bytes);
    if (!result) {
        result = hetgpu_cuda_stream_synchronize(
            &ct2d->gpu_info.hetgpu_state, stream);
    }
    if (!cxl_type2_cuda_allocation_release_pageable_alias(
            &ct2d->cuda_allocations, base, epoch, generation,
            CXL_TYPE2_CUDA_ALIAS_NORMAL)) {
        return false;
    }
    if (!result) {
        result = hetgpu_memcpy_dtoh(&ct2d->gpu_info.hetgpu_state, observed,
                                    result_ptr, sizeof(*observed));
    }
    return result == HETGPU_SUCCESS;
}

static bool gate_run_graph(CXLType2State *ct2d, uint64_t base,
                           uint64_t epoch, uint64_t generation,
                           HetGPUGraphExec graph_exec, HetGPUStream stream,
                           HetGPUDevicePtr result_ptr, uint64_t *observed)
{
    uint64_t alias;
    int result;

    if (!cxl_type2_cuda_allocation_acquire_pageable_alias(
            &ct2d->cuda_allocations, base, epoch, generation,
            CXL_TYPE2_CUDA_ALIAS_GRAPH, &alias)) {
        return false;
    }
    result = hetgpu_cuda_graph_launch(&ct2d->gpu_info.hetgpu_state,
                                      graph_exec, stream);
    if (!result) {
        result = hetgpu_cuda_stream_synchronize(
            &ct2d->gpu_info.hetgpu_state, stream);
    }
    if (!cxl_type2_cuda_allocation_release_pageable_alias(
            &ct2d->cuda_allocations, base, epoch, generation,
            CXL_TYPE2_CUDA_ALIAS_GRAPH)) {
        return false;
    }
    if (!result) {
        result = hetgpu_memcpy_dtoh(&ct2d->gpu_info.hetgpu_state, observed,
                                    result_ptr, sizeof(*observed));
    }
    return result == HETGPU_SUCCESS;
}

static bool gate_publish(const GateState *state, const CXLType2State *ct2d,
                         Error **errp)
{
    g_autoptr(JSONWriter) writer = json_writer_new(true);
    g_autoptr(GString) text = NULL;
    g_autofree char *temporary =
        g_strdup_printf("%s.tmp", ct2d->model_alias_gate.output);
    g_autoptr(GError) error = NULL;

    json_writer_start_object(writer, NULL);
    json_writer_str(writer, "kind", "cxl-type2-model-alias-gate");
    json_writer_str(writer, "mode", ct2d->model_alias_gate.mode);
    json_writer_str(writer, "status", state->first_failure ? "fail" : "pass");
    if (state->first_failure) {
        json_writer_start_object(writer, "first_failure");
        json_writer_str(writer, "stage", state->first_failure);
        json_writer_int64(writer, "result", state->first_result);
        json_writer_end_object(writer);
    } else {
        json_writer_null(writer, "first_failure");
    }
    json_writer_start_array(writer, "predicates");
    for (size_t i = 0; i < state->predicate_count; i++) {
        const GatePredicate *predicate = &state->predicates[i];

        json_writer_start_object(writer, NULL);
        json_writer_str(writer, "name", predicate->name);
        json_writer_bool(writer, "required", predicate->required);
        json_writer_str(writer, "status",
                        gate_status_name(predicate->status));
        json_writer_int64(writer, "result", predicate->result);
        predicate->reason ?
            json_writer_str(writer, "reason", predicate->reason) :
            json_writer_null(writer, "reason");
        predicate->blocking ?
            json_writer_str(writer, "blocking_identity", predicate->blocking) :
            json_writer_null(writer, "blocking_identity");
        json_writer_end_object(writer);
    }
    json_writer_end_array(writer);
    json_writer_start_object(writer, "facts");
    json_writer_uint64(writer, "model_aperture_offset",
                       ct2d->model_aperture.offset);
    json_writer_uint64(writer, "model_aperture_size",
                       ct2d->model_aperture.size);
    json_writer_uint64(writer, "fixture_file_offset",
                       ct2d->model_alias_gate.file_offset);
    json_writer_uint64(writer, "fixture_length",
                       ct2d->model_alias_gate.length);
    json_writer_uint64(writer, "mapping_generation",
                       state->mapping_generation);
    json_writer_uint64(writer, "source_device", state->source_device);
    json_writer_uint64(writer, "source_inode", state->source_inode);
    json_writer_uint64(writer, "source_size", state->source_size);
    json_writer_uint64(writer, "source_mode", state->source_mode);
    json_writer_str(writer, "source_sha256_before",
                    state->source_sha256_before);
    json_writer_str(writer, "source_sha256_after",
                    state->source_sha256_after);
    json_writer_uint64(writer, "allocation_epoch", state->allocation_epoch);
    json_writer_uint64(writer, "alias_address_generation_one",
                       state->alias_address_generation_one);
    json_writer_uint64(writer, "alias_address_generation_two",
                       state->alias_address_generation_two);
    json_writer_uint64(writer, "alias_build_wall_ns_generation_one",
                       state->alias_build_wall_ns_generation_one);
    json_writer_uint64(writer, "alias_build_wall_ns_generation_two",
                       state->alias_build_wall_ns_generation_two);
    json_writer_uint64(writer,
                       "boundary_composition_wall_ns_generation_one",
                       state->boundary_composition_wall_ns_generation_one);
    json_writer_uint64(writer,
                       "boundary_composition_wall_ns_generation_two",
                       state->boundary_composition_wall_ns_generation_two);
    json_writer_uint64(writer, "logical_bytes", state->logical_bytes);
    json_writer_uint64(writer, "guard_bytes", state->guard_bytes);
    json_writer_uint64(writer, "file_mapped_bytes",
                       state->file_mapped_bytes);
    json_writer_uint64(writer, "derived_boundary_pages",
                       state->derived_boundary_pages);
    json_writer_uint64(writer, "derived_boundary_copy_bytes",
                       state->derived_boundary_copy_bytes);
    json_writer_uint64(writer, "cxl_latency_ns", state->cxl_latency_ns);
    json_writer_start_array(writer, "vfio_dma_ranges");
    for (size_t i = 0; i < state->vfio_dma_range_count; i++) {
        json_writer_start_object(writer, NULL);
        json_writer_uint64(writer, "offset",
                           state->vfio_dma_ranges[i].offset);
        json_writer_uint64(writer, "size", state->vfio_dma_ranges[i].size);
        json_writer_end_object(writer);
    }
    json_writer_end_array(writer);
    json_writer_uint64(writer, "normal_generation_one",
                       state->normal_generation_one);
    json_writer_uint64(writer, "normal_generation_two",
                       state->normal_generation_two);
    json_writer_uint64(writer, "graph_generation_one",
                       state->graph_generation_one);
    json_writer_uint64(writer, "graph_generation_two",
                       state->graph_generation_two);
    json_writer_uint64(writer, "model_htod_calls", 0);
    json_writer_uint64(writer, "model_htod_bytes", 0);
    json_writer_str(writer, "physical_service", "unavailable");
    json_writer_str(writer, "physical_service_reason",
                    "um-migration-producer-not-recorded");
    json_writer_str(writer, "nested_consumer_coverage", "unsupported");
    json_writer_end_object(writer);
    json_writer_str(writer, "proves",
                    "bounded production mapping, CXL admission, pageable alias, and normal/graph CUDA consumers on this exact host and GPU");
    json_writer_str(writer, "does_not_prove",
                    "guest source handoff, complete Kimi consumer coverage, physical remote service, overlap, or TPS");
    json_writer_end_object(writer);
    text = json_writer_get_and_free(g_steal_pointer(&writer));
    if (!g_file_set_contents(temporary, text->str, text->len, &error)) {
        error_setg(errp, "cannot write model alias gate output %s: %s",
                   temporary, error->message);
        return false;
    }
    if (rename(temporary, ct2d->model_alias_gate.output) < 0) {
        error_setg_errno(errp, errno,
                         "cannot publish model alias gate output %s",
                         ct2d->model_alias_gate.output);
        return false;
    }
    return !state->first_failure;
}

bool cxl_type2_model_alias_gate_run(CXLType2State *ct2d, Error **errp)
{
    GateState state = { 0 };
    VirtioSharedMemory *shmem = NULL;
    MemoryRegion *dax_mr = NULL;
    VirtioSharedMemoryMapping *mapping = NULL;
    VirtioSharedMemoryMapping *pinned_mapping = NULL;
    CXLType2CudaAliasSource sources[4];
    HetGPUState *hetgpu = &ct2d->gpu_info.hetgpu_state;
    HetGPUModule module = NULL;
    HetGPUFunction read_function = NULL;
    HetGPUFunction write_function = NULL;
    HetGPUStream stream = NULL;
    HetGPUGraph graph = NULL;
    HetGPUGraphExec graph_exec = NULL;
    HetGPUDevicePtr result_ptr = 0;
    struct stat source_stat;
    size_t page_size = qemu_real_host_page_size();
    uint64_t total_bytes = 3 * page_size;
    uint64_t base = ct2d->model_aperture.offset;
    CXLType2MemoryRange vfio_ranges[2];
    size_t vfio_range_count;
    uint64_t epoch = 0;
    uint64_t alias = 0;
    uint64_t pinned_generation = 0;
    uint64_t pinned_file_offset = 0;
    uint64_t pinned_length = 0;
    void *host_address = NULL;
    const char *reason = NULL;
    int fixture_fd = -1;
    int source_fd = -1;
    int result = 0;
    bool mapping_installed = false;
    bool allocation_recorded = false;
    bool alias_mapped = false;
    bool graph_bound = false;
    bool cuda_ready = false;
    bool read_mode = !strcmp(ct2d->model_alias_gate.mode, "read");

    state.logical_bytes = total_bytes - 512;
    state.guard_bytes = 512;

    fixture_fd = open(ct2d->model_alias_gate.fixture,
                      O_RDONLY | O_CLOEXEC);
    if (fixture_fd < 0 || fstat(fixture_fd, &source_stat) < 0 ||
        !S_ISREG(source_stat.st_mode) || source_stat.st_size < 0 ||
        ct2d->model_alias_gate.file_offset > (uint64_t)source_stat.st_size ||
        ct2d->model_alias_gate.length >
            (uint64_t)source_stat.st_size -
                ct2d->model_alias_gate.file_offset ||
        !gate_hash_file_range(fixture_fd,
                              ct2d->model_alias_gate.file_offset,
                              ct2d->model_alias_gate.length,
                              state.source_sha256_before)) {
        gate_record(&state, "source-file-identity", true, GATE_FAIL,
                    errno ? errno : EINVAL,
                    "fixture stat, bounds, or hash failed", NULL);
    } else {
        state.source_device = source_stat.st_dev;
        state.source_inode = source_stat.st_ino;
        state.source_size = source_stat.st_size;
        state.source_mode = source_stat.st_mode;
        gate_record(&state, "source-file-identity", true, GATE_PASS, 0,
                    NULL, NULL);
    }

    if (fixture_fd >= 0 &&
        vhost_user_fs_pci_get_dax(ct2d->direct_source_fs, &shmem, &dax_mr)) {
        mapping = virtio_shared_memory_mapping_new(
            VIRTIO_FS_SHMCAP_ID_CACHE, fixture_fd,
            ct2d->model_alias_gate.file_offset,
            ct2d->model_alias_gate.shmem_offset,
            ct2d->model_alias_gate.length, false);
        if (mapping && virtio_add_shmem_map(shmem, mapping) == 0) {
            mapping_installed = true;
            gate_record(&state, "production-mapping-installed", true,
                        GATE_PASS, 0, NULL, NULL);
        } else {
            if (mapping) {
                object_unref(OBJECT(mapping));
                mapping = NULL;
            }
            gate_record(&state, "production-mapping-installed", true,
                        GATE_FAIL, EINVAL,
                        "production DAX mapping installation failed",
                        "source-file-identity");
        }
    } else {
        gate_record(&state, "production-mapping-installed", true,
                    GATE_FAIL, ENODEV,
                    "realized vhost-user-fs DAX region unavailable",
                    "source-file-identity");
    }

    if (mapping_installed &&
        virtio_shared_memory_pin_range(
            shmem, ct2d->model_alias_gate.shmem_offset,
            ct2d->model_alias_gate.length, &pinned_mapping,
            &pinned_generation, &host_address, NULL, NULL) == 0 &&
        virtio_shared_memory_mapping_dup_source(
            pinned_mapping, pinned_generation, &source_fd,
            &pinned_file_offset, &pinned_length) == 0 &&
        pinned_file_offset == ct2d->model_alias_gate.file_offset &&
        pinned_length == ct2d->model_alias_gate.length &&
        fstat(source_fd, &source_stat) == 0) {
        state.mapping_generation = pinned_generation;
        gate_record(&state, "mapping-source-handoff", true, GATE_PASS, 0,
                    NULL, NULL);
    } else {
        gate_record(&state, "mapping-source-handoff", true, GATE_FAIL,
                    errno ? errno : ESTALE,
                    "short mapping pin could not produce an exact source fd",
                    "production-mapping-installed");
    }
    if (pinned_mapping) {
        virtio_shared_memory_unpin(pinned_mapping);
        pinned_mapping = NULL;
    }

    if (source_fd >= 0 &&
        cxl_type2_cuda_allocation_record(&ct2d->cuda_allocations, base,
                                         total_bytes, &epoch)) {
        uint64_t alias_begin_ns;
        uint64_t source_call_id = 1;

        allocation_recorded = true;
        state.allocation_epoch = epoch;
        gate_sources(sources, source_fd, &source_stat, pinned_file_offset,
                     ct2d->model_aperture.offset, pinned_generation,
                     page_size);
        alias_begin_ns = qemu_clock_get_ns(QEMU_CLOCK_REALTIME);
        if (cxl_type2_cuda_allocation_map_pageable_alias(
                &ct2d->cuda_allocations, base, epoch, 1, sources,
                G_N_ELEMENTS(sources), &source_call_id, 1, 0,
                state.logical_bytes,
                state.guard_bytes, &alias, &reason)) {
            CXLType2CudaAllocation *allocation =
                cxl_type2_cuda_allocation_find(&ct2d->cuda_allocations,
                                                base, epoch);

            alias_mapped = true;
            state.alias_address_generation_one = alias;
            state.alias_build_wall_ns_generation_one =
                qemu_clock_get_ns(QEMU_CLOCK_REALTIME) - alias_begin_ns;
            state.file_mapped_bytes =
                allocation->pageable_alias->file_mapped_bytes;
            state.derived_boundary_pages =
                allocation->pageable_alias->derived_boundary_pages;
            state.derived_boundary_copy_bytes =
                allocation->pageable_alias->derived_boundary_copy_bytes;
            state.boundary_composition_wall_ns_generation_one =
                allocation->pageable_alias->boundary_composition_wall_ns;
            gate_record(&state, "alias-generation-one", true, GATE_PASS, 0,
                        NULL, NULL);
        } else {
            gate_record(&state, "alias-generation-one", true, GATE_FAIL,
                        EINVAL, reason, "mapping-source-handoff");
        }
    } else {
        gate_record(&state, "alias-generation-one", true, GATE_FAIL,
                    EINVAL, "allocation identity or source fd unavailable",
                    "mapping-source-handoff");
    }

    if (mapping_installed &&
        virtio_del_shmem_map(shmem, ct2d->model_alias_gate.shmem_offset,
                             ct2d->model_alias_gate.length) == 0) {
        mapping_installed = false;
        gate_record(&state, "derived-fd-survives-revoke", true,
                    alias_mapped ? GATE_PASS : GATE_FAIL,
                    alias_mapped ? 0 : EINVAL,
                    alias_mapped ? NULL : "alias was not published",
                    "alias-generation-one");
    } else {
        gate_record(&state, "derived-fd-survives-revoke", true, GATE_FAIL,
                    EBUSY, "installed mapping could not be revoked",
                    "mapping-source-handoff");
    }

    if (cxl_type2_model_aperture_access(
            ct2d, ct2d->model_aperture.offset, 8, false,
            &state.cxl_latency_ns, &reason)) {
        gate_record(&state, "model-aperture-cxl-read", true, GATE_PASS, 0,
                    NULL, NULL);
    } else {
        gate_record(&state, "model-aperture-cxl-read", true, GATE_FAIL,
                    EIO, reason, NULL);
    }
    result = cxl_type2_model_aperture_access(
        ct2d, ct2d->model_aperture.offset + ct2d->model_aperture.size,
        8, false, &state.cxl_latency_ns, &reason);
    gate_record(&state, "ordinary-gap-rejected", true,
                !result ? GATE_PASS : GATE_FAIL,
                !result ? 0 : EACCES,
                !result ? NULL : "range outside aperture was admitted", NULL);
    result = cxl_type2_model_aperture_access(
        ct2d, ct2d->model_aperture.offset, 8, true,
        &state.cxl_latency_ns, &reason);
    gate_record(&state, "model-aperture-write-rejected", true,
                !result ? GATE_PASS : GATE_FAIL,
                !result ? 0 : EACCES,
                !result ? NULL : "readonly aperture admitted a write", NULL);
    result = cxl_type2_model_aperture_access(
        ct2d, ct2d->coherent_pool.base_offset, 8, false,
        &state.cxl_latency_ns, &reason);
    gate_record(&state, "coherent-pool-overlap-rejected", true,
                !result ? GATE_PASS : GATE_FAIL,
                !result ? 0 : EACCES,
                !result ? NULL : "coherent pool entered model aperture", NULL);

    vfio_range_count = cxl_type2_vfio_dma_ranges(ct2d, vfio_ranges);
    state.vfio_dma_range_count = vfio_range_count;
    memcpy(state.vfio_dma_ranges, vfio_ranges,
           vfio_range_count * sizeof(*vfio_ranges));
    result = 0;
    uint64_t vfio_bytes = 0;
    for (size_t i = 0; i < vfio_range_count; i++) {
        uint64_t range_end = vfio_ranges[i].offset + vfio_ranges[i].size;
        uint64_t aperture_end = ct2d->model_aperture.offset +
                                ct2d->model_aperture.size;

        vfio_bytes += vfio_ranges[i].size;
        if (range_end < vfio_ranges[i].offset ||
            (vfio_ranges[i].offset < aperture_end &&
             range_end > ct2d->model_aperture.offset)) {
            result = EINVAL;
        }
    }
    if (vfio_bytes != ct2d->device_mem_size - ct2d->model_aperture.size) {
        result = EINVAL;
    }
    gate_record(&state, "vfio-model-aperture-excluded", true,
                !result ? GATE_PASS : GATE_FAIL, result,
                !result ? NULL :
                    "production VFIO DMA ranges overlap or omit BAR4 bytes",
                NULL);

    if (alias_mapped && hetgpu->initialized) {
        result = hetgpu_load_ptx(hetgpu, gate_ptx, &module);
        if (!result && read_mode) {
            result = hetgpu_get_function(hetgpu, module,
                                         "cxl_model_alias_read",
                                         &read_function);
        }
        if (!result && !read_mode) {
            result = hetgpu_get_function(hetgpu, module,
                                         "cxl_model_alias_write",
                                         &write_function);
        }
        if (!result) {
            result = hetgpu_cuda_stream_create(hetgpu, 0, &stream);
        }
        if (!result && read_mode) {
            result = hetgpu_malloc(hetgpu, sizeof(uint64_t),
                                   HETGPU_MEM_DEVICE_ONLY, &result_ptr);
        }
        cuda_ready = result == HETGPU_SUCCESS;
        gate_record(&state, "cuda-consumer-setup", true,
                    cuda_ready ? GATE_PASS : GATE_FAIL, result,
                    cuda_ready ? NULL :
                        "PTX, stream, or result allocation failed",
                    "alias-generation-one");
    } else {
        gate_record(&state, "cuda-consumer-setup", true, GATE_NOT_RUN, 0,
                    "alias and strict CUDA context required",
                    "alias-generation-one");
    }

    if (cuda_ready && read_mode) {
        uint64_t expected = gate_expected((uint8_t *)(uintptr_t)alias,
                                          page_size, total_bytes);
        bool ok = gate_run_normal(ct2d, base, epoch, 1, read_function,
                                  stream, result_ptr, page_size, total_bytes,
                                  &state.normal_generation_one);

        gate_record(&state, "normal-generation-one", true,
                    ok && state.normal_generation_one == expected ?
                        GATE_PASS : GATE_FAIL,
                    ok ? 0 : EIO,
                    ok && state.normal_generation_one == expected ? NULL :
                        "normal generation-one oracle failed",
                    "cuda-consumer-setup");

        result = hetgpu_cuda_stream_begin_capture(hetgpu, stream, 0);
        if (!result) {
            result = gate_launch_read(hetgpu, read_function, stream, alias,
                                      result_ptr, page_size, total_bytes);
        }
        if (!result) {
            result = hetgpu_cuda_stream_end_capture(hetgpu, stream,
                                                    (void **)&graph);
        }
        if (!result) {
            result = hetgpu_cuda_graph_instantiate(hetgpu, graph,
                                                   &graph_exec, NULL,
                                                   NULL, 0);
        }
        if (!result &&
            !cxl_type2_cuda_allocation_bind_graph_alias(
                &ct2d->cuda_allocations, base, epoch, alias)) {
            result = EINVAL;
        } else if (!result) {
            graph_bound = true;
        }
        gate_record(&state, "graph-stable-binding", true,
                    !result ? GATE_PASS : GATE_FAIL, result,
                    !result ? NULL : "graph capture or stable binding failed",
                    "cuda-consumer-setup");

        if (!result) {
            ok = gate_run_graph(ct2d, base, epoch, 1, graph_exec, stream,
                                result_ptr, &state.graph_generation_one);
            gate_record(&state, "graph-generation-one", true,
                        ok && state.graph_generation_one == expected ?
                            GATE_PASS : GATE_FAIL,
                        ok ? 0 : EIO,
                        ok && state.graph_generation_one == expected ? NULL :
                            "graph generation-one oracle failed",
                        "graph-stable-binding");
        } else {
            gate_record(&state, "graph-generation-one", true, GATE_NOT_RUN,
                        0, "graph binding unavailable",
                        "graph-stable-binding");
        }

        gate_sources(sources, source_fd, &source_stat,
                     pinned_file_offset + 5 * page_size,
                     ct2d->model_aperture.offset + 5 * page_size,
                     pinned_generation, page_size);
        reason = NULL;
        uint64_t alias_begin_ns = qemu_clock_get_ns(QEMU_CLOCK_REALTIME);
        uint64_t source_call_id = 2;

        if (graph_bound && cxl_type2_cuda_allocation_map_pageable_alias(
                &ct2d->cuda_allocations, base, epoch, 2, sources,
                G_N_ELEMENTS(sources), &source_call_id, 1, 0,
                state.logical_bytes,
                state.guard_bytes, &state.alias_address_generation_two,
                &reason) &&
            state.alias_address_generation_two ==
                state.alias_address_generation_one) {
            CXLType2CudaAllocation *allocation =
                cxl_type2_cuda_allocation_find(&ct2d->cuda_allocations,
                                                base, epoch);

            alias = state.alias_address_generation_two;
            state.alias_build_wall_ns_generation_two =
                qemu_clock_get_ns(QEMU_CLOCK_REALTIME) - alias_begin_ns;
            state.boundary_composition_wall_ns_generation_two =
                allocation->pageable_alias->boundary_composition_wall_ns;
            gate_record(&state, "same-va-generation-two", true, GATE_PASS,
                        0, NULL, NULL);
        } else {
            gate_record(&state, "same-va-generation-two", true, GATE_FAIL,
                        EINVAL, reason ? reason : "stable VA changed",
                        "graph-generation-one");
        }

        if (state.alias_address_generation_two ==
            state.alias_address_generation_one) {
            expected = gate_expected((uint8_t *)(uintptr_t)alias,
                                     page_size, total_bytes);
            ok = gate_run_normal(ct2d, base, epoch, 2, read_function,
                                 stream, result_ptr, page_size, total_bytes,
                                 &state.normal_generation_two);
            gate_record(&state, "normal-generation-two", true,
                        ok && state.normal_generation_two == expected ?
                            GATE_PASS : GATE_FAIL,
                        ok ? 0 : EIO,
                        ok && state.normal_generation_two == expected ? NULL :
                            "normal generation-two oracle failed",
                        "same-va-generation-two");
            ok = gate_run_graph(ct2d, base, epoch, 2, graph_exec, stream,
                                result_ptr, &state.graph_generation_two);
            gate_record(&state, "graph-generation-two", true,
                        ok && state.graph_generation_two == expected ?
                            GATE_PASS : GATE_FAIL,
                        ok ? 0 : EIO,
                        ok && state.graph_generation_two == expected ? NULL :
                            "same graph exec generation-two oracle failed",
                        "same-va-generation-two");
        } else {
            gate_record(&state, "normal-generation-two", true, GATE_NOT_RUN,
                        0, "generation-two alias unavailable",
                        "same-va-generation-two");
            gate_record(&state, "graph-generation-two", true, GATE_NOT_RUN,
                        0, "generation-two alias unavailable",
                        "same-va-generation-two");
        }
    } else if (read_mode) {
        gate_record(&state, "normal-generation-one", true, GATE_NOT_RUN, 0,
                    "CUDA setup failed", "cuda-consumer-setup");
        gate_record(&state, "graph-stable-binding", true, GATE_NOT_RUN, 0,
                    "CUDA setup failed", "cuda-consumer-setup");
        gate_record(&state, "graph-generation-one", true, GATE_NOT_RUN, 0,
                    "CUDA setup failed", "cuda-consumer-setup");
        gate_record(&state, "same-va-generation-two", true, GATE_NOT_RUN, 0,
                    "CUDA setup failed", "cuda-consumer-setup");
        gate_record(&state, "normal-generation-two", true, GATE_NOT_RUN, 0,
                    "CUDA setup failed", "cuda-consumer-setup");
        gate_record(&state, "graph-generation-two", true, GATE_NOT_RUN, 0,
                    "CUDA setup failed", "cuda-consumer-setup");
    }

    if (cuda_ready && !read_mode) {
        result = gate_launch_write(hetgpu, write_function, stream, alias);
        if (!result) {
            result = hetgpu_cuda_stream_synchronize(hetgpu, stream);
        }
        gate_record(&state, "gpu-write-rejected", true,
                    result ? GATE_PASS : GATE_FAIL, result,
                    result ? NULL : "GPU write to readonly alias succeeded",
                    "alias-generation-one");
    } else if (!read_mode) {
        gate_record(&state, "gpu-write-rejected", true, GATE_NOT_RUN, 0,
                    "generation-one CUDA alias unavailable",
                    "alias-generation-one");
    }

    if (fixture_fd >= 0 && gate_hash_file_range(
            fixture_fd, ct2d->model_alias_gate.file_offset,
            ct2d->model_alias_gate.length, state.source_sha256_after) &&
        !strcmp(state.source_sha256_before, state.source_sha256_after)) {
        gate_record(&state, "file-hash-unchanged", true, GATE_PASS, 0,
                    NULL, NULL);
    } else {
        gate_record(&state, "file-hash-unchanged", true, GATE_FAIL, EIO,
                    "fixture bytes changed", NULL);
    }

    if (read_mode) {
        int cleanup_result = HETGPU_SUCCESS;

        if (graph_exec) {
            cleanup_result = hetgpu_cuda_graph_exec_destroy(hetgpu,
                                                            graph_exec);
            graph_exec = NULL;
        }
        if (!cleanup_result && graph) {
            cleanup_result = hetgpu_cuda_graph_destroy(hetgpu, graph);
            graph = NULL;
        }
        if (!cleanup_result && result_ptr) {
            cleanup_result = hetgpu_free(hetgpu, result_ptr);
            result_ptr = 0;
        }
        if (!cleanup_result && stream) {
            cleanup_result = hetgpu_cuda_stream_destroy(hetgpu, stream);
            stream = NULL;
        }
        if (!cleanup_result && module) {
            cleanup_result = hetgpu_unload_module(hetgpu, module);
            module = NULL;
        }
        gate_record(&state, "cuda-resource-cleanup", true,
                    !cleanup_result ? GATE_PASS : GATE_FAIL,
                    cleanup_result,
                    !cleanup_result ? NULL : "CUDA resource cleanup failed",
                    "cuda-consumer-setup");
        if (graph_bound) {
            graph_bound = !cxl_type2_cuda_allocation_unbind_graph_alias(
                &ct2d->cuda_allocations, base, epoch, alias);
        }
    }
    if (graph_bound) {
        gate_record(&state, "allocation-cleanup", true, GATE_FAIL, EBUSY,
                    "graph binding remained live", NULL);
    } else if (alias_mapped &&
               !cxl_type2_cuda_allocation_drop_pageable_alias(
                   &ct2d->cuda_allocations, base, epoch,
                   state.alias_address_generation_two ? 2 : 1)) {
        gate_record(&state, "allocation-cleanup", true, GATE_FAIL, EBUSY,
                    "pageable alias release failed", NULL);
    } else if (allocation_recorded &&
               !cxl_type2_cuda_allocation_forget(&ct2d->cuda_allocations,
                                                 base)) {
        gate_record(&state, "allocation-cleanup", true, GATE_FAIL, EBUSY,
                    "allocation identity release failed", NULL);
    } else {
        gate_record(&state, "allocation-cleanup", true, GATE_PASS, 0,
                    NULL, NULL);
    }
    if (mapping_installed) {
        (void)virtio_del_shmem_map(shmem,
                                   ct2d->model_alias_gate.shmem_offset,
                                   ct2d->model_alias_gate.length);
    }
    if (source_fd >= 0) {
        close(source_fd);
    }
    if (fixture_fd >= 0) {
        close(fixture_fd);
    }

    gate_record(&state, "nested-consumer-coverage", false,
                GATE_UNAVAILABLE, 0,
                "bounded gate covers top-level pointer parameters only",
                NULL);
    gate_record(&state, "physical-service", false, GATE_UNAVAILABLE, 0,
                "UM migration and remote-map producer is absent", NULL);
    if (!gate_publish(&state, ct2d, errp)) {
        if (!*errp) {
            error_setg(errp, "model alias gate failed at %s",
                       state.first_failure);
        }
        return false;
    }
    return true;
}
