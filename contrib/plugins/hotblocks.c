/*
 * Copyright (C) 2019, Alex Bennée <alex.bennee@linaro.org>
 *
 * License: GNU GPL, version 2 or later.
 *   See the COPYING file in the top-level directory.
 */
#include <inttypes.h>
#include <assert.h>
#include <stdlib.h>
#include <inttypes.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <glib.h>

#include <qemu-plugin.h>

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

static bool do_inline;

/* Plugins need to take care of their own locking */
static GMutex lock;
static GHashTable *hotblocks;
static guint64 limit = 20;
static char *scope_name;
static char *active_identity;
static bool scope_open;

/*
 * Counting Structure
 *
 * The internals of the TCG are not exposed to plugins so we can only
 * get the starting PC for each block. We cheat this slightly by
 * checking the number of instructions as well to help
 * differentiate.
 */
typedef struct {
    uint64_t start_addr;
    struct qemu_plugin_scoreboard *exec_count;
    int trans_count;
    unsigned long insns;
    uint64_t scope_start_count;
    char *disassembly;
    bool disassembly_truncated;
    bool disassembly_ambiguous;
} ExecCount;

typedef struct {
    ExecCount *count;
    uint64_t executions;
} ScopedCount;

static gint cmp_exec_count(gconstpointer a, gconstpointer b, gpointer d)
{
    ExecCount *ea = (ExecCount *) a;
    ExecCount *eb = (ExecCount *) b;
    uint64_t count_a =
        qemu_plugin_u64_sum(qemu_plugin_scoreboard_u64(ea->exec_count));
    uint64_t count_b =
        qemu_plugin_u64_sum(qemu_plugin_scoreboard_u64(eb->exec_count));
    return count_a > count_b ? -1 : count_a < count_b ? 1 : 0;
}

static gint cmp_scoped_count(gconstpointer a, gconstpointer b)
{
    const ScopedCount *sa = *(ScopedCount * const *)a;
    const ScopedCount *sb = *(ScopedCount * const *)b;

    return sa->executions > sb->executions ? -1
           : sa->executions < sb->executions ? 1 : 0;
}

static guint exec_count_hash(gconstpointer v)
{
    const ExecCount *e = v;
    return e->start_addr ^ e->insns;
}

static gboolean exec_count_equal(gconstpointer v1, gconstpointer v2)
{
    const ExecCount *ea = v1;
    const ExecCount *eb = v2;
    return (ea->start_addr == eb->start_addr) &&
           (ea->insns == eb->insns);
}

static void exec_count_free(gpointer key, gpointer value, gpointer user_data)
{
    ExecCount *cnt = value;
    qemu_plugin_scoreboard_free(cnt->exec_count);
    g_free(cnt->disassembly);
}

static char *tb_disassembly(const struct qemu_plugin_tb *tb, size_t insns,
                            bool *truncated)
{
    g_autoptr(GString) text = g_string_new(NULL);

    *truncated = false;
    for (size_t i = 0; i < insns; i++) {
        struct qemu_plugin_insn *insn = qemu_plugin_tb_get_insn(tb, i);
        g_autofree char *disas = qemu_plugin_insn_disas(insn);
        g_autofree char *entry = g_strdup_printf(
            "%s0x%016" PRIx64 ":%s", i ? " | " : "",
            qemu_plugin_insn_vaddr(insn), disas);

        if (text->len + strlen(entry) > 512) {
            *truncated = true;
            break;
        }
        g_string_append(text, entry);
    }
    return g_string_free(g_steal_pointer(&text), false);
}

static void plugin_exit(qemu_plugin_id_t id, void *p)
{
    g_autoptr(GString) report = g_string_new("collected ");
    GList *counts, *it;
    int i;

    if (scope_open) {
        g_string_append_printf(report,
                               "TCG_HOTBLOCKS_ERROR reason=open-scope-at-exit identity=%s\n",
                               active_identity);
    }
    g_string_append_printf(report, "%d entries in the hash table\n",
                           g_hash_table_size(hotblocks));
    counts = g_hash_table_get_values(hotblocks);
    it = g_list_sort_with_data(counts, cmp_exec_count, NULL);

    if (it) {
        g_string_append_printf(report, "pc, tcount, icount, ecount\n");

        for (i = 0; i < limit && it->next; i++, it = it->next) {
            ExecCount *rec = (ExecCount *) it->data;
            g_string_append_printf(
                report, "0x%016"PRIx64", %d, %ld, %"PRId64"\n",
                rec->start_addr, rec->trans_count,
                rec->insns,
                qemu_plugin_u64_sum(
                    qemu_plugin_scoreboard_u64(rec->exec_count)));
        }

        g_list_free(it);
    }

    qemu_plugin_outs(report->str);

    g_hash_table_foreach(hotblocks, exec_count_free, NULL);
    g_hash_table_destroy(hotblocks);
    g_free(scope_name);
    g_free(active_identity);
}

static void scope_begin_snapshot(gpointer key, gpointer value,
                                 gpointer user_data)
{
    ExecCount *cnt = value;

    cnt->scope_start_count =
        qemu_plugin_u64_sum(qemu_plugin_scoreboard_u64(cnt->exec_count));
}

static void scope_collect_delta(gpointer key, gpointer value,
                                gpointer user_data)
{
    ExecCount *cnt = value;
    GPtrArray *counts = user_data;
    uint64_t current =
        qemu_plugin_u64_sum(qemu_plugin_scoreboard_u64(cnt->exec_count));

    if (current > cnt->scope_start_count) {
        ScopedCount *scoped = g_new(ScopedCount, 1);

        scoped->count = cnt;
        scoped->executions = current - cnt->scope_start_count;
        g_ptr_array_add(counts, scoped);
    }
}

static void scope_event(qemu_plugin_id_t id, const char *scope,
                        const char *identity, bool begin, void *userdata)
{
    g_autoptr(GPtrArray) counts = NULL;
    g_autoptr(GString) report = NULL;
    uint64_t total_executions = 0;
    uint64_t total_guest_instructions = 0;
    uint64_t top_executions = 0;
    guint rows;

    if (!scope_name || g_strcmp0(scope, scope_name) != 0) {
        return;
    }

    g_mutex_lock(&lock);
    if (begin) {
        if (scope_open) {
            qemu_plugin_outs("TCG_HOTBLOCKS_ERROR reason=overlapping-scope\n");
            g_mutex_unlock(&lock);
            return;
        }
        g_hash_table_foreach(hotblocks, scope_begin_snapshot, NULL);
        active_identity = g_strdup(identity);
        scope_open = true;
        g_mutex_unlock(&lock);
        return;
    }

    if (!scope_open || g_strcmp0(identity, active_identity) != 0) {
        qemu_plugin_outs("TCG_HOTBLOCKS_ERROR reason=scope-end-mismatch\n");
        g_mutex_unlock(&lock);
        return;
    }

    counts = g_ptr_array_new_with_free_func(g_free);
    g_hash_table_foreach(hotblocks, scope_collect_delta, counts);
    g_ptr_array_sort(counts, cmp_scoped_count);
    for (guint i = 0; i < counts->len; i++) {
        ScopedCount *scoped = g_ptr_array_index(counts, i);

        total_executions += scoped->executions;
        total_guest_instructions += scoped->executions * scoped->count->insns;
    }
    rows = MIN(limit, counts->len);
    for (guint i = 0; i < rows; i++) {
        ScopedCount *scoped = g_ptr_array_index(counts, i);

        top_executions += scoped->executions;
    }

    report = g_string_new(NULL);
    g_string_append_printf(
        report,
        "TCG_HOTBLOCKS_SUMMARY scope=%s %s total_executions=%" PRIu64
        " total_guest_instructions=%" PRIu64 " unique_blocks=%u"
        " top_rows=%u top_executions=%" PRIu64 "\n",
        scope, identity, total_executions, total_guest_instructions,
        counts->len, rows, top_executions);
    for (guint i = 0; i < rows; i++) {
        ScopedCount *scoped = g_ptr_array_index(counts, i);
        ExecCount *cnt = scoped->count;
        g_autofree char *disassembly_b64 = g_base64_encode(
            (const guchar *)cnt->disassembly, strlen(cnt->disassembly));

        g_string_append_printf(
            report,
            "TCG_HOTBLOCK scope=%s %s rank=%u pc=0x%016" PRIx64
            " instructions=%lu translations=%d executions=%" PRIu64
            " disassembly_b64=%s disassembly_truncated=%u"
            " disassembly_ambiguous=%u\n",
            scope, identity, i + 1, cnt->start_addr, cnt->insns,
            cnt->trans_count, scoped->executions, disassembly_b64,
            cnt->disassembly_truncated, cnt->disassembly_ambiguous);
    }
    qemu_plugin_outs(report->str);
    g_clear_pointer(&active_identity, g_free);
    scope_open = false;
    g_mutex_unlock(&lock);
}

static void plugin_init(void)
{
    hotblocks = g_hash_table_new(exec_count_hash, exec_count_equal);
}

static void vcpu_tb_exec(unsigned int cpu_index, void *udata)
{
    ExecCount *cnt = (ExecCount *)udata;
    qemu_plugin_u64_add(qemu_plugin_scoreboard_u64(cnt->exec_count),
                        cpu_index, 1);
}

/*
 * When do_inline we ask the plugin to increment the counter for us.
 * Otherwise a helper is inserted which calls the vcpu_tb_exec
 * callback.
 */
static void vcpu_tb_trans(qemu_plugin_id_t id, struct qemu_plugin_tb *tb)
{
    ExecCount *cnt;
    uint64_t pc = qemu_plugin_tb_vaddr(tb);
    size_t insns = qemu_plugin_tb_n_insns(tb);
    bool disassembly_truncated;
    g_autofree char *disassembly = tb_disassembly(
        tb, insns, &disassembly_truncated);

    g_mutex_lock(&lock);
    {
        ExecCount e;
        e.start_addr = pc;
        e.insns = insns;
        cnt = (ExecCount *) g_hash_table_lookup(hotblocks, &e);
    }

    if (cnt) {
        cnt->trans_count++;
        if (g_strcmp0(cnt->disassembly, disassembly) != 0 ||
            cnt->disassembly_truncated != disassembly_truncated) {
            cnt->disassembly_ambiguous = true;
        }
    } else {
        cnt = g_new0(ExecCount, 1);
        cnt->start_addr = pc;
        cnt->trans_count = 1;
        cnt->insns = insns;
        cnt->disassembly = g_steal_pointer(&disassembly);
        cnt->disassembly_truncated = disassembly_truncated;
        cnt->exec_count = qemu_plugin_scoreboard_new(sizeof(uint64_t));
        g_hash_table_insert(hotblocks, cnt, cnt);
    }

    g_mutex_unlock(&lock);

    if (do_inline) {
        qemu_plugin_register_vcpu_tb_exec_inline_per_vcpu(
            tb, QEMU_PLUGIN_INLINE_ADD_U64,
            qemu_plugin_scoreboard_u64(cnt->exec_count), 1);
    } else {
        qemu_plugin_register_vcpu_tb_exec_cb(tb, vcpu_tb_exec,
                                             QEMU_PLUGIN_CB_NO_REGS,
                                             (void *)cnt);
    }
}

QEMU_PLUGIN_EXPORT
int qemu_plugin_install(qemu_plugin_id_t id, const qemu_info_t *info,
                        int argc, char **argv)
{
    for (int i = 0; i < argc; i++) {
        char *opt = argv[i];
        g_auto(GStrv) tokens = g_strsplit(opt, "=", 2);
        if (g_strcmp0(tokens[0], "inline") == 0) {
            if (!qemu_plugin_bool_parse(tokens[0], tokens[1], &do_inline)) {
                fprintf(stderr, "boolean argument parsing failed: %s\n", opt);
                return -1;
            }
        } else if (g_strcmp0(tokens[0], "limit") == 0) {
            char *end = NULL;
            guint64 parsed;

            if (!tokens[1] || !tokens[1][0]) {
                fprintf(stderr, "invalid limit: %s\n", opt);
                return -1;
            }
            parsed = g_ascii_strtoull(tokens[1], &end, 10);
            if (!end || *end ||
                parsed == 0 || parsed > 1024) {
                fprintf(stderr, "invalid limit: %s\n", opt);
                return -1;
            }
            limit = parsed;
        } else if (g_strcmp0(tokens[0], "scope") == 0) {
            if (!tokens[1] || !tokens[1][0]) {
                fprintf(stderr, "invalid scope: %s\n", opt);
                return -1;
            }
            g_free(scope_name);
            scope_name = g_strdup(tokens[1]);
        } else {
            fprintf(stderr, "option parsing failed: %s\n", opt);
            return -1;
        }
    }

    plugin_init();

    qemu_plugin_register_vcpu_tb_trans_cb(id, vcpu_tb_trans);
    qemu_plugin_register_scope_cb(id, scope_event, NULL);
    qemu_plugin_register_atexit_cb(id, plugin_exit, NULL);
    return 0;
}
