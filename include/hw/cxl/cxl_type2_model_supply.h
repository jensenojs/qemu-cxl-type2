/*
 * Sealed model-member admission for CXL Type-2 pageable aliases.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef CXL_TYPE2_MODEL_SUPPLY_H
#define CXL_TYPE2_MODEL_SUPPLY_H

#include "qapi/error.h"

typedef enum CXLType2ModelSupplyRoute {
    CXL_TYPE2_MODEL_SUPPLY_GPU_RESIDENT,
    CXL_TYPE2_MODEL_SUPPLY_SELECTED_HTOD,
    CXL_TYPE2_MODEL_SUPPLY_CXL_DIRECT,
} CXLType2ModelSupplyRoute;

typedef struct CXLType2ModelMember {
    char *name;
    char artifact_sha256[65];
    uint64_t device;
    uint64_t inode;
    uint64_t size;
    uint32_t mode;
    uint64_t logical_cxl_offset;
} CXLType2ModelMember;

typedef struct CXLType2ModelMemberManifest {
    char model_view_identity[65];
    char sha256[65];
    CXLType2ModelMember *members;
    size_t member_count;
} CXLType2ModelMemberManifest;

typedef struct CXLType2ModelConsumerRoot {
    uint32_t parameter_index;
    uint64_t parameter_offset;
    uint64_t parameter_size;
} CXLType2ModelConsumerRoot;

typedef struct CXLType2ModelConsumerSink {
    char *memory_space;
    char *access_mode;
    CXLType2ModelConsumerRoot *roots;
    size_t root_count;
} CXLType2ModelConsumerSink;

typedef struct CXLType2ModelConsumerFunction {
    char driver_input_cubin_sha256[65];
    char *function_name;
    char function_code_sha256[65];
    CXLType2ModelConsumerRoot *parameters;
    size_t parameter_count;
    CXLType2ModelConsumerSink *sinks;
    size_t sink_count;
} CXLType2ModelConsumerFunction;

typedef struct CXLType2ModelConsumerCertificate {
    char sha256[65];
    CXLType2ModelConsumerFunction *functions;
    size_t function_count;
} CXLType2ModelConsumerCertificate;

typedef enum CXLType2ModuleLoadKind {
    CXL_TYPE2_MODULE_LOAD_NONE,
    CXL_TYPE2_MODULE_LOAD_PTX,
    CXL_TYPE2_MODULE_LOAD_CUBIN,
} CXLType2ModuleLoadKind;

typedef struct CXLType2ModuleEntry {
    void *handle;
    CXLType2ModuleLoadKind load_kind;
    char driver_input_cubin_sha256[65];
} CXLType2ModuleEntry;

typedef struct CXLType2FunctionEntry {
    void *handle;
    uint32_t module_id;
    char *name;
    const CXLType2ModelConsumerFunction *consumer;
} CXLType2FunctionEntry;

bool cxl_type2_model_supply_route_parse(
    const char *text, CXLType2ModelSupplyRoute *route, Error **errp);
const char *cxl_type2_model_supply_route_name(
    CXLType2ModelSupplyRoute route);
bool cxl_type2_model_member_manifest_load(
    CXLType2ModelMemberManifest *manifest, const char *path,
    const char *expected_sha256, uint64_t aperture_size, Error **errp);
void cxl_type2_model_member_manifest_clear(
    CXLType2ModelMemberManifest *manifest);
const CXLType2ModelMember *cxl_type2_model_member_find(
    const CXLType2ModelMemberManifest *manifest, uint64_t device,
    uint64_t inode, uint64_t size, uint32_t mode, uint64_t file_offset,
    uint64_t length);
bool cxl_type2_model_consumer_certificate_load(
    CXLType2ModelConsumerCertificate *certificate, const char *path,
    const char *expected_sha256, Error **errp);
void cxl_type2_model_consumer_certificate_clear(
    CXLType2ModelConsumerCertificate *certificate);
const CXLType2ModelConsumerFunction *cxl_type2_model_consumer_function_find(
    const CXLType2ModelConsumerCertificate *certificate,
    const char *driver_input_cubin_sha256, const char *function_name);

#endif
