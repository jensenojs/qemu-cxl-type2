/*
 * Sealed model-member admission for CXL Type-2 pageable aliases.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/cutils.h"
#include "qapi/error.h"
#include "qobject/qdict.h"
#include "qobject/qjson.h"
#include "qobject/qlist.h"
#include "qobject/qnum.h"
#include "qobject/qstring.h"
#include "hw/cxl/cxl_type2_model_supply.h"

static bool model_supply_sha256_valid(const char *value)
{
    if (!value || strlen(value) != 64) {
        return false;
    }
    for (size_t i = 0; i < 64; i++) {
        if (!g_ascii_isxdigit(value[i]) || g_ascii_isupper(value[i])) {
            return false;
        }
    }
    return true;
}

static bool model_supply_dict_has_exact_keys(const QDict *dict,
                                             const char *const *keys,
                                             size_t key_count)
{
    if (!dict || qdict_size(dict) != key_count) {
        return false;
    }
    for (size_t i = 0; i < key_count; i++) {
        if (!qdict_haskey(dict, keys[i])) {
            return false;
        }
    }
    return true;
}

static bool model_supply_dict_uint(const QDict *dict, const char *key,
                                   uint64_t *value)
{
    QNum *number = qobject_to(QNum, qdict_get(dict, key));

    return number && qnum_get_try_uint(number, value);
}

static const char *model_supply_dict_string(const QDict *dict,
                                            const char *key)
{
    QString *string = qobject_to(QString, qdict_get(dict, key));

    return string ? qstring_get_str(string) : NULL;
}

bool cxl_type2_model_supply_route_parse(
    const char *text, CXLType2ModelSupplyRoute *route, Error **errp)
{
    if (!route) {
        error_setg(errp, "model supply route output is missing");
        return false;
    }
    if (!text || !text[0] || !strcmp(text, "selected-htod")) {
        *route = CXL_TYPE2_MODEL_SUPPLY_SELECTED_HTOD;
        return true;
    }
    if (!strcmp(text, "gpu-resident")) {
        *route = CXL_TYPE2_MODEL_SUPPLY_GPU_RESIDENT;
        return true;
    }
    if (!strcmp(text, "cxl-direct")) {
        *route = CXL_TYPE2_MODEL_SUPPLY_CXL_DIRECT;
        return true;
    }
    error_setg(errp,
               "model-supply-route must be gpu-resident, selected-htod or "
               "cxl-direct");
    return false;
}

const char *cxl_type2_model_supply_route_name(
    CXLType2ModelSupplyRoute route)
{
    switch (route) {
    case CXL_TYPE2_MODEL_SUPPLY_GPU_RESIDENT:
        return "gpu-resident";
    case CXL_TYPE2_MODEL_SUPPLY_SELECTED_HTOD:
        return "selected-htod";
    case CXL_TYPE2_MODEL_SUPPLY_CXL_DIRECT:
        return "cxl-direct";
    }
    g_assert_not_reached();
}

void cxl_type2_model_member_manifest_clear(
    CXLType2ModelMemberManifest *manifest)
{
    if (!manifest) {
        return;
    }
    for (size_t i = 0; i < manifest->member_count; i++) {
        g_free(manifest->members[i].name);
    }
    g_free(manifest->members);
    memset(manifest, 0, sizeof(*manifest));
}

bool cxl_type2_model_member_manifest_load(
    CXLType2ModelMemberManifest *manifest, const char *path,
    const char *expected_sha256, uint64_t aperture_size, Error **errp)
{
    static const char *const top_keys[] = {
        "model_view_identity", "members",
    };
    static const char *const member_keys[] = {
        "name", "artifact_sha256", "device", "inode", "size", "mode",
        "logical_cxl_offset",
    };
    g_autofree char *contents = NULL;
    QObject *object = NULL;
    g_autoptr(GChecksum) checksum = NULL;
    gsize length = 0;
    QDict *root;
    QList *members;
    const QListEntry *entry;
    const char *model_view_identity;
    size_t index = 0;

    if (!manifest || !path || path[0] != '/' ||
        !model_supply_sha256_valid(expected_sha256) || !aperture_size) {
        error_setg(errp, "cxl-direct requires an absolute model member "
                   "manifest, its SHA256, and a nonzero aperture");
        return false;
    }
    if (!g_file_get_contents(path, &contents, &length, NULL)) {
        error_setg(errp, "cannot read model member manifest '%s'", path);
        return false;
    }
    checksum = g_checksum_new(G_CHECKSUM_SHA256);
    g_checksum_update(checksum, (const guchar *)contents, length);
    if (strcmp(g_checksum_get_string(checksum), expected_sha256)) {
        error_setg(errp,
                   "model member manifest SHA256 differs from the sealed value");
        return false;
    }
    object = qobject_from_json(contents, errp);
    root = qobject_to(QDict, object);
    if (!root || !model_supply_dict_has_exact_keys(
                     root, top_keys, G_N_ELEMENTS(top_keys))) {
        error_setg(errp, "model member manifest has invalid top-level fields");
        goto fail;
    }
    model_view_identity = model_supply_dict_string(root,
                                                   "model_view_identity");
    members = qobject_to(QList, qdict_get(root, "members"));
    if (!model_supply_sha256_valid(model_view_identity) || !members ||
        qlist_empty(members)) {
        error_setg(
            errp,
            "model member manifest identity or members are invalid");
        goto fail;
    }

    cxl_type2_model_member_manifest_clear(manifest);
    manifest->members = g_try_new0(CXLType2ModelMember,
                                   qlist_size(members));
    if (!manifest->members) {
        error_setg(errp, "cannot allocate model member manifest");
        goto fail;
    }
    pstrcpy(manifest->model_view_identity,
            sizeof(manifest->model_view_identity), model_view_identity);
    pstrcpy(manifest->sha256, sizeof(manifest->sha256), expected_sha256);

    QLIST_FOREACH_ENTRY(members, entry) {
        QDict *member_dict = qobject_to(QDict, qlist_entry_obj(entry));
        CXLType2ModelMember *member = &manifest->members[index];
        const char *name;
        const char *artifact_sha256;
        uint64_t mode;

        if (!member_dict || !model_supply_dict_has_exact_keys(
                                member_dict, member_keys,
                                G_N_ELEMENTS(member_keys))) {
            error_setg(errp, "model member %zu has invalid fields", index);
            goto fail;
        }
        name = model_supply_dict_string(member_dict, "name");
        artifact_sha256 = model_supply_dict_string(member_dict,
                                                    "artifact_sha256");
        if (!name || !name[0] || strchr(name, '\n') ||
            !model_supply_sha256_valid(artifact_sha256) ||
            !model_supply_dict_uint(member_dict, "device", &member->device) ||
            !model_supply_dict_uint(member_dict, "inode", &member->inode) ||
            !model_supply_dict_uint(member_dict, "size", &member->size) ||
            !model_supply_dict_uint(member_dict, "mode", &mode) ||
            mode > UINT32_MAX ||
            !model_supply_dict_uint(member_dict, "logical_cxl_offset",
                                    &member->logical_cxl_offset) ||
            !member->size || member->logical_cxl_offset > aperture_size ||
            member->size > aperture_size - member->logical_cxl_offset) {
            error_setg(errp,
                       "model member %zu has invalid identity or range",
                       index);
            goto fail;
        }
        for (size_t previous = 0; previous < index; previous++) {
            CXLType2ModelMember *other = &manifest->members[previous];

            if ((member->device == other->device &&
                 member->inode == other->inode) ||
                (member->logical_cxl_offset <
                     other->logical_cxl_offset + other->size &&
                 other->logical_cxl_offset <
                     member->logical_cxl_offset + member->size)) {
                error_setg(errp, "model member %zu duplicates an identity "
                           "or overlaps the aperture", index);
                goto fail;
            }
        }
        member->name = g_strdup(name);
        pstrcpy(member->artifact_sha256, sizeof(member->artifact_sha256),
                artifact_sha256);
        member->mode = mode;
        manifest->member_count = ++index;
    }
    qobject_unref(object);
    return true;

fail:
    cxl_type2_model_member_manifest_clear(manifest);
    if (object) {
        qobject_unref(object);
    }
    return false;
}

const CXLType2ModelMember *cxl_type2_model_member_find(
    const CXLType2ModelMemberManifest *manifest, uint64_t device,
    uint64_t inode, uint64_t size, uint32_t mode, uint64_t file_offset,
    uint64_t length)
{
    if (!manifest || !length || file_offset > UINT64_MAX - length) {
        return NULL;
    }
    for (size_t i = 0; i < manifest->member_count; i++) {
        const CXLType2ModelMember *member = &manifest->members[i];

        if (member->device == device && member->inode == inode &&
            member->size == size && member->mode == mode &&
            file_offset <= member->size &&
            length <= member->size - file_offset) {
            return member;
        }
    }
    return NULL;
}

void cxl_type2_model_consumer_certificate_clear(
    CXLType2ModelConsumerCertificate *certificate)
{
    if (!certificate) {
        return;
    }
    for (size_t i = 0; i < certificate->function_count; i++) {
        CXLType2ModelConsumerFunction *function = &certificate->functions[i];

        for (size_t j = 0; j < function->sink_count; j++) {
            g_free(function->sinks[j].memory_space);
            g_free(function->sinks[j].access_mode);
            g_free(function->sinks[j].roots);
        }
        g_free(function->sinks);
        g_free(function->parameters);
        g_free(function->function_name);
    }
    g_free(certificate->functions);
    memset(certificate, 0, sizeof(*certificate));
}

static bool model_supply_parse_root(QDict *dict,
                                    CXLType2ModelConsumerRoot *root)
{
    static const char *const keys[] = {
        "parameter_index", "parameter_offset", "parameter_size",
    };
    uint64_t index;

    return model_supply_dict_has_exact_keys(dict, keys, G_N_ELEMENTS(keys)) &&
           model_supply_dict_uint(dict, "parameter_index", &index) &&
           index <= UINT32_MAX &&
           model_supply_dict_uint(dict, "parameter_offset",
                                  &root->parameter_offset) &&
           model_supply_dict_uint(dict, "parameter_size",
                                  &root->parameter_size) &&
           root->parameter_size &&
           (root->parameter_index = index, true);
}

static bool model_supply_parse_roots(
    QList *list, CXLType2ModelConsumerRoot **roots_out, size_t *count_out)
{
    CXLType2ModelConsumerRoot *roots;
    const QListEntry *entry;
    size_t index = 0;

    if (!list || !roots_out || !count_out) {
        return false;
    }
    roots = g_try_new0(CXLType2ModelConsumerRoot, qlist_size(list));
    if (qlist_size(list) && !roots) {
        return false;
    }
    QLIST_FOREACH_ENTRY(list, entry) {
        QDict *dict = qobject_to(QDict, qlist_entry_obj(entry));

        if (!dict || !model_supply_parse_root(dict, &roots[index]) ||
            (index && roots[index - 1].parameter_index >=
                          roots[index].parameter_index)) {
            g_free(roots);
            return false;
        }
        index++;
    }
    *roots_out = roots;
    *count_out = index;
    return true;
}

static bool model_supply_parse_sink(QDict *dict,
                                    CXLType2ModelConsumerSink *sink)
{
    static const char *const keys[] = {
        "instruction_pc", "address_operand_index", "memory_space",
        "access_mode", "parameter_roots", "unresolved_roots",
    };
    const char *space;
    const char *access;
    QList *roots;
    QList *unresolved;
    uint64_t ignored;

    if (!model_supply_dict_has_exact_keys(dict, keys, G_N_ELEMENTS(keys)) ||
        !model_supply_dict_uint(dict, "instruction_pc", &ignored) ||
        !model_supply_dict_uint(dict, "address_operand_index", &ignored)) {
        return false;
    }
    space = model_supply_dict_string(dict, "memory_space");
    access = model_supply_dict_string(dict, "access_mode");
    roots = qobject_to(QList, qdict_get(dict, "parameter_roots"));
    unresolved = qobject_to(QList, qdict_get(dict, "unresolved_roots"));
    if (!space || (strcmp(space, "global") && strcmp(space, "generic") &&
                   strcmp(space, "shared") && strcmp(space, "local")) ||
        !access || (strcmp(access, "read") && strcmp(access, "write") &&
                    strcmp(access, "read-write")) ||
        !roots || !unresolved || !qlist_empty(unresolved) ||
        !model_supply_parse_roots(roots, &sink->roots, &sink->root_count) ||
        ((!strcmp(space, "global") || !strcmp(space, "generic")) &&
         !sink->root_count) ||
        ((!strcmp(space, "shared") || !strcmp(space, "local")) &&
         sink->root_count)) {
        return false;
    }
    sink->memory_space = g_strdup(space);
    sink->access_mode = g_strdup(access);
    return true;
}

static bool model_supply_parse_function(
    QDict *dict, CXLType2ModelConsumerFunction *function)
{
    static const char *const keys[] = {
        "driver_input_cubin_sha256", "function_name",
        "function_code_sha256", "parameter_abi_inventory",
        "address_sink_coverage", "memory_address_sinks",
    };
    const char *cubin_sha256;
    const char *function_name;
    const char *function_code_sha256;
    const char *coverage;
    QList *parameters;
    QList *sinks;
    const QListEntry *entry;
    size_t index = 0;

    if (!model_supply_dict_has_exact_keys(dict, keys, G_N_ELEMENTS(keys))) {
        return false;
    }
    cubin_sha256 = model_supply_dict_string(
        dict, "driver_input_cubin_sha256");
    function_name = model_supply_dict_string(dict, "function_name");
    function_code_sha256 = model_supply_dict_string(
        dict, "function_code_sha256");
    coverage = model_supply_dict_string(dict, "address_sink_coverage");
    parameters = qobject_to(QList, qdict_get(dict, "parameter_abi_inventory"));
    sinks = qobject_to(QList, qdict_get(dict, "memory_address_sinks"));
    if (!model_supply_sha256_valid(cubin_sha256) || !function_name ||
        !function_name[0] || strchr(function_name, '\n') ||
        !model_supply_sha256_valid(function_code_sha256) ||
        !coverage || strcmp(coverage, "complete") || !parameters || !sinks ||
        !model_supply_parse_roots(parameters, &function->parameters,
                                  &function->parameter_count)) {
        return false;
    }
    function->sinks = g_try_new0(CXLType2ModelConsumerSink,
                                  qlist_size(sinks));
    if (qlist_size(sinks) && !function->sinks) {
        return false;
    }
    function->sink_count = qlist_size(sinks);
    QLIST_FOREACH_ENTRY(sinks, entry) {
        QDict *sink = qobject_to(QDict, qlist_entry_obj(entry));

        if (!sink || !model_supply_parse_sink(sink, &function->sinks[index])) {
            return false;
        }
        index++;
    }
    pstrcpy(function->driver_input_cubin_sha256,
            sizeof(function->driver_input_cubin_sha256), cubin_sha256);
    pstrcpy(function->function_code_sha256,
            sizeof(function->function_code_sha256), function_code_sha256);
    function->function_name = g_strdup(function_name);
    return true;
}

bool cxl_type2_model_consumer_certificate_load(
    CXLType2ModelConsumerCertificate *certificate, const char *path,
    const char *expected_sha256, Error **errp)
{
    static const char *const keys[] = {
        "source_result_identity", "producer_identity", "functions",
    };
    g_autofree char *contents = NULL;
    g_autoptr(GChecksum) checksum = NULL;
    QObject *object = NULL;
    QDict *root;
    QList *functions;
    const QListEntry *entry;
    gsize length = 0;
    size_t index = 0;

    if (!certificate || !path || path[0] != '/' ||
        !model_supply_sha256_valid(expected_sha256)) {
        error_setg(errp, "cxl-direct requires an absolute model consumer "
                   "certificate and its SHA256");
        return false;
    }
    if (!g_file_get_contents(path, &contents, &length, NULL)) {
        error_setg(errp, "cannot read model consumer certificate '%s'", path);
        return false;
    }
    checksum = g_checksum_new(G_CHECKSUM_SHA256);
    g_checksum_update(checksum, (const guchar *)contents, length);
    if (strcmp(g_checksum_get_string(checksum), expected_sha256)) {
        error_setg(errp, "model consumer certificate SHA256 differs from "
                   "the sealed value");
        return false;
    }
    object = qobject_from_json(contents, errp);
    if (!object) {
        return false;
    }
    root = qobject_to(QDict, object);
    functions = root ? qobject_to(QList, qdict_get(root, "functions")) : NULL;
    if (!root || !model_supply_dict_has_exact_keys(
                     root, keys, G_N_ELEMENTS(keys)) ||
        !qobject_to(QDict, qdict_get(root, "source_result_identity")) ||
        !qobject_to(QDict, qdict_get(root, "producer_identity")) ||
        !functions || qlist_empty(functions)) {
        error_setg(
            errp,
            "model consumer certificate has invalid top-level fields");
        goto fail;
    }
    cxl_type2_model_consumer_certificate_clear(certificate);
    certificate->functions = g_try_new0(
        CXLType2ModelConsumerFunction, qlist_size(functions));
    if (!certificate->functions) {
        error_setg(errp, "cannot allocate model consumer certificate");
        goto fail;
    }
    QLIST_FOREACH_ENTRY(functions, entry) {
        QDict *function = qobject_to(QDict, qlist_entry_obj(entry));
        CXLType2ModelConsumerFunction *parsed =
            &certificate->functions[index];

        certificate->function_count = index + 1;
        if (!function || !model_supply_parse_function(function, parsed) ||
            (index &&
             (strcmp(certificate->functions[index - 1]
                         .driver_input_cubin_sha256,
                     parsed->driver_input_cubin_sha256) > 0 ||
              (!strcmp(certificate->functions[index - 1]
                           .driver_input_cubin_sha256,
                       parsed->driver_input_cubin_sha256) &&
               strcmp(certificate->functions[index - 1].function_name,
                      parsed->function_name) >= 0)))) {
            error_setg(errp, "model consumer certificate function %zu is "
                       "invalid or unsorted", index);
            goto fail;
        }
        index++;
    }
    pstrcpy(certificate->sha256, sizeof(certificate->sha256),
            expected_sha256);
    qobject_unref(object);
    return true;

fail:
    cxl_type2_model_consumer_certificate_clear(certificate);
    qobject_unref(object);
    return false;
}

const CXLType2ModelConsumerFunction *cxl_type2_model_consumer_function_find(
    const CXLType2ModelConsumerCertificate *certificate,
    const char *driver_input_cubin_sha256, const char *function_name)
{
    if (!certificate || !driver_input_cubin_sha256 || !function_name) {
        return NULL;
    }
    for (size_t i = 0; i < certificate->function_count; i++) {
        const CXLType2ModelConsumerFunction *function =
            &certificate->functions[i];

        if (!strcmp(function->driver_input_cubin_sha256,
                    driver_input_cubin_sha256) &&
            !strcmp(function->function_name, function_name)) {
            return function;
        }
    }
    return NULL;
}
