#include <stdio.h>
#include <stdlib.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <pthread.h>
#endif
#include <string.h>
#include "acuda/compiler.h"
#include "ptx.h"
#include "parse/parse_internal.h"
#include "spirv_internal.h"

#if defined(_MSC_VER)
#if defined(_WIN32)
#include <windows.h>
#endif
static __inline uint64_t acuda_atomic_inc_u64(uint64_t *p) {
#if defined(_M_IX86)
    volatile LONG64 *vp = (volatile LONG64 *)p;
    LONG64 oldv, newv;
    do {
        oldv = *vp;
        newv = oldv + 1;
    } while (InterlockedCompareExchange64(vp, newv, oldv) != oldv);
    return (uint64_t)newv;
#else
    return (uint64_t)InterlockedIncrement64((volatile LONG64 *)p);
#endif
}
static __inline int acuda_atomic_inc_i32(int *p) { return (int)InterlockedIncrement((volatile LONG *)p); }
#else
static __inline uint64_t acuda_atomic_inc_u64(uint64_t *p) { return __atomic_add_fetch(p, 1, __ATOMIC_SEQ_CST); }
static __inline int acuda_atomic_inc_i32(int *p) { return __atomic_add_fetch(p, 1, __ATOMIC_SEQ_CST); }
#endif

struct acuda_kernel {
    uint32_t *spv;
    size_t nwords;
    acuda_kernel_info_t info;
    struct acuda_kernel *next; // host-call segment chain
    uint32_t nsegments;        // valid on the head (0 = not a chain)
    uint32_t full_num_ptr_params; // includes internal host buffers
    uint32_t full_num_params;
    uint32_t spv_va;           // 1 if spv allocated via VirtualAlloc/mmap-load
    char *spv_path;            // backing AKB file (load-on-demand); owned
    long spv_file_off;         // offset of this segment's SPIR-V words in file
};

static uint64_t kernel_serial_counter; // 0 = invalid; atomic

static void dump_kernel(const acuda_kernel_t *k) {
    const char *dir = getenv("ACUDA_DUMP_SPV_DIR");
    if (!dir || !dir[0] || !k)
        return;
    uint32_t si = 0;
    for (const acuda_kernel_t *kq = k; kq; kq = kq->next, si++) {
        char safe[208];
        size_t n = 0;
        for (const char *s = kq->info.entry;
             *s && n < sizeof(safe) - 1; s++) {
            char c = *s;
            safe[n++] = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                        (c >= '0' && c <= '9') || c == '_' || c == '-' ||
                        c == '.'
                    ? c
                    : '_';
        }
        safe[n] = '\0';
        char path[1024];
        if (kq == k)
            snprintf(path, sizeof(path), "%s/%s.spv", dir, safe);
        else
            snprintf(path, sizeof(path), "%s/%s_seg%u.spv", dir, safe, si);
        FILE *f = fopen(path, "wb");
        if (f) {
            fwrite(kq->spv, 4, kq->nwords, f);
            fclose(f);
        }
    }
}

// ---- Conservative per-parameter write analysis -----------------------
// The runtime skips a buffer barrier when the buffer has not been written
// since the last acquire. That needs to know which pointer params a
// kernel may write. This is a small taint pass over the PTX IR:
//   * a pointer param taints the register it is loaded into;
//   * taint propagates through address arithmetic (and, conservatively,
//     through every non-memory-producing instruction);
//   * a store/atomic/reduction marks the tainted params in its address,
//     and any ambiguous case marks every param (safe over-approximation).
// Data loads do not propagate taint, so loading a value through param A
// and storing it through param B does not mark A as written.
#define AW_MAXREGS 4096

typedef struct {
    const char *name;
    uint64_t taint;
} aw_reg_t;

typedef struct {
    aw_reg_t regs[AW_MAXREGS];
    int n;
    int overflow;
} aw_map_t;

static int aw_index(aw_map_t *m, const char *name) {
    if (!name || !name[0])
        return -1;
    for (int i = 0; i < m->n; i++)
        if (strcmp(m->regs[i].name, name) == 0)
            return i;
    if (m->n >= AW_MAXREGS) {
        m->overflow = 1;
        return -1;
    }
    m->regs[m->n].name = name;
    m->regs[m->n].taint = 0;
    return m->n++;
}

static uint64_t aw_get(aw_map_t *m, const char *name) {
    int i = aw_index(m, name);
    return i >= 0 ? m->regs[i].taint : 0;
}

static void aw_add(aw_map_t *m, const char *name, uint64_t t,
                   int *changed) {
    int i = aw_index(m, name);
    if (i < 0)
        return;
    uint64_t n = m->regs[i].taint | t;
    if (n != m->regs[i].taint) {
        m->regs[i].taint = n;
        *changed = 1;
    }
}

static uint64_t aw_operand_taint(aw_map_t *m, const ptx_opd_t *o) {
    if ((o->kind == OPD_REG || o->kind == OPD_REGOFF) && o->reg[0])
        return aw_get(m, o->reg);
    if (o->kind == OPD_ADDR && o->addr_reg[0])
        return aw_get(m, o->addr_reg);
    return 0;
}

static int aw_param_bit(const ptx_func_t *fn, const int *kinds,
                        const int *pbit, const char *name) {
    if (!name)
        return -1;
    for (size_t i = 0; i < fn->nparams; i++) {
        if (fn->params[i].is_retval || kinds[i] != 1)
            continue;
        if (strcmp(fn->params[i].name, name) == 0)
            return pbit[i];
    }
    return -1;
}

// True for instructions whose destination is data read from memory (or a
// shared-memory copy); propagating pointer taint through them would mark
// unrelated params as written.
static int aw_blocks_taint(ptx_op_t op) {
    switch (op) {
    case POP_LD:
    case POP_ATOM:
    case POP_ATOMCAS:
    case POP_TEX:
    case POP_TLD4:
    case POP_TXQ:
    case POP_SULD:
    case POP_SUQ:
    case POP_ISSPACEP:
    case POP_CPASYNC:
    case POP_CPASYNC_BULK:
    case POP_LDMATRIX:
    case POP_WMMA_LOAD:
    case POP_MBARRIER:
    case POP_HOSTCALL:
    case POP_HOST_REQUEST:
    case POP_HOST_REPLY:
    case POP_HOST_SPILL:
    case POP_HOST_RESTORE:
    case POP_HOST_RESULT:
        return 1;
    default:
        return 0;
    }
}

static void analyze_param_writes(const ptx_func_t *fn, const int *kinds,
                                 uint32_t written[ACUDA_MAX_PARAMS]) {
    memset(written, 0, sizeof(uint32_t) * ACUDA_MAX_PARAMS);
    int pbit[ACUDA_MAX_PARAMS];
    for (int i = 0; i < ACUDA_MAX_PARAMS; i++)
        pbit[i] = -1;
    int nptr = 0;
    for (size_t i = 0; i < fn->nparams; i++) {
        if (fn->params[i].is_retval)
            continue;
        if (kinds[i] == 1)
            pbit[i] = nptr++;
    }
    if (nptr == 0)
        return;

    aw_map_t m;
    memset(&m, 0, sizeof(m));
    int mark_all = fn->host_calls != 0;

    // Seed pointer values loaded from kernel params.
    for (size_t bi = 0; bi < fn->nblocks; bi++) {
        const ptx_block_t *blk = &fn->blocks[bi];
        for (size_t ii = 0; ii < blk->count; ii++) {
            const ptx_instr_t *in = &blk->instrs[ii];
            if (in->op != POP_LD || in->space != PSP_PARAM)
                continue;
            if (in->src[0].kind != OPD_ADDR || !in->dst[0])
                continue;
            if (in->src[0].addr_reg[0]) {
                // Symbolic parameter address: rare; stay conservative.
                mark_all = 1;
                continue;
            }
            int bit = aw_param_bit(fn, kinds, pbit, in->src[0].param);
            if (bit < 0)
                continue;
            int changed = 0;
            aw_add(&m, in->dst, 1ull << bit, &changed);
        }
    }

    // Propagate taint through address-producing operations. Data loads
    // are skipped above, so stored values do not taint addresses.
    for (int pass = 0; pass < 128; pass++) {
        int changed = 0;
        for (size_t bi = 0; bi < fn->nblocks; bi++) {
            const ptx_block_t *blk = &fn->blocks[bi];
            for (size_t ii = 0; ii < blk->count; ii++) {
                const ptx_instr_t *in = &blk->instrs[ii];
                if (!in->dst[0] || aw_blocks_taint(in->op))
                    continue;
                uint64_t t = 0;
                for (int k = 0; k < in->nsrc && k < 8; k++)
                    t |= aw_operand_taint(&m, &in->src[k]);
                if (!t)
                    continue;
                aw_add(&m, in->dst, t, &changed);
                if (in->dst2[0])
                    aw_add(&m, in->dst2, t, &changed);
            }
        }
        if (!changed)
            break;
    }

    // Mark writes. Unknown addresses mark every param: over-approximating
    // is safe (a redundant barrier), under-approximating is not.
    uint64_t written_mask = 0;
    for (size_t bi = 0; bi < fn->nblocks; bi++) {
        const ptx_block_t *blk = &fn->blocks[bi];
        for (size_t ii = 0; ii < blk->count; ii++) {
            const ptx_instr_t *in = &blk->instrs[ii];
            if (in->op == POP_SUST || in->op == POP_SURED ||
                in->op == POP_CPASYNC_BULK || in->op == POP_CALL ||
                in->op == POP_BRX) {
                // Surface writes, bulk copies, and calls/jump tables may
                // target memory this analysis does not model.
                mark_all = 1;
                continue;
            }
            int wr = in->op == POP_ST || in->op == POP_ATOM ||
                     in->op == POP_ATOMCAS || in->op == POP_RED_ASYNC;
            if (!wr)
                continue;
            if (in->space == PSP_SHARED || in->space == PSP_LOCAL) {
                // A pointer spilled to shared/local may be loaded and
                // dereferenced later; be conservative for value spills.
                if (in->op == POP_ST && in->nsrc > 1 &&
                    aw_operand_taint(&m, &in->src[1]))
                    mark_all = 1;
                continue;
            }
            const ptx_opd_t *a = &in->src[0];
            uint64_t t = 0;
            if (a->kind == OPD_ADDR) {
                if (a->addr_reg[0]) {
                    t = aw_get(&m, a->addr_reg);
                } else if (a->param[0]) {
                    int bit = aw_param_bit(fn, kinds, pbit, a->param);
                    if (bit >= 0)
                        written_mask |= 1ull << bit;
                    else
                        mark_all = 1;
                    continue;
                }
            } else if ((a->kind == OPD_REG || a->kind == OPD_REGOFF) &&
                       a->reg[0]) {
                t = aw_get(&m, a->reg);
            }
            if (!t) {
                mark_all = 1;
                continue;
            }
            written_mask |= t;
            // Storing an address-derived value through a pointer may
            // create a new alias; the target memory must be treated as
            // written.
            if (in->op == POP_ST && in->nsrc > 1 &&
                aw_operand_taint(&m, &in->src[1]))
                mark_all = 1;
        }
    }
    if (m.overflow)
        mark_all = 1;
    if (mark_all)
        written_mask = ~0ull;
    for (int b = 0; b < ACUDA_MAX_PARAMS; b++)
        written[b] = (b < nptr && ((written_mask >> b) & 1ull)) ? 1u : 0u;
}


static void *acuda_spv_alloc(size_t bytes, uint32_t *out_va) {
    if (out_va) *out_va = 0;
#if defined(_WIN32)
    /* 60MB+ SPIR-V blobs can fail in 32-bit CRT heap; use VirtualAlloc.
     * TOP_DOWN helps 32-bit VA fragmentation for very large allocations. */
    DWORD fl = MEM_COMMIT | MEM_RESERVE;
    if (sizeof(void *) <= 4 && bytes >= (8u * 1024u * 1024u)) fl |= MEM_TOP_DOWN;
    void *p = VirtualAlloc(NULL, bytes, fl, PAGE_READWRITE);
    if (p) { if (out_va) *out_va = 1; return p; }
#endif
    return malloc(bytes);
}
static void acuda_spv_free(void *p, uint32_t va) {
    if (!p) return;
#if defined(_WIN32)
    if (va) { VirtualFree(p, 0, MEM_RELEASE); return; }
#endif
    (void)va;
    free(p);
}

typedef struct { const ptx_unit_t *unit; const ptx_func_t *fn; spv_blob_t *blob;
    int *kinds; size_t *shared_bytes; ptx_emit_meta_t *meta; char *err;
    size_t errlen; const acuda_compiler_caps_t *caps; int result; } emit_job_t;
#if defined(_WIN32)
static DWORD WINAPI emit_thread_proc(LPVOID arg) {
    emit_job_t *j = (emit_job_t *)arg;
    __try {
        j->result = ptx_emit_spv(j->unit, j->fn, j->blob, j->kinds,
                                 j->shared_bytes, j->meta, j->err, j->errlen,
                                 j->caps);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        j->result = -1;
        if (j->err && j->errlen > 0)
            snprintf(j->err, j->errlen, "SEH in ptx_emit_spv");
    }
    return 0;
}
#else
static void *emit_thread_proc(void *arg) {
    emit_job_t *j = (emit_job_t *)arg;
    j->result = ptx_emit_spv(j->unit, j->fn, j->blob, j->kinds,
                             j->shared_bytes, j->meta, j->err, j->errlen,
                             j->caps);
    return NULL;
}
#endif

static acuda_status_t compile_func(const ptx_unit_t *unit,
                                   const ptx_func_t *fn,
                                   acuda_kernel_t **out_kernel,
                                   char *errbuf, size_t errlen,
                                   const acuda_compiler_caps_t *caps) {
    *out_kernel = NULL;
    char err[1024];
    err[0] = '\0';
    spv_blob_t blob;
    memset(&blob, 0, sizeof(blob));
    int kinds[ACUDA_MAX_PARAMS];
    for (int i = 0; i < ACUDA_MAX_PARAMS; i++)
        kinds[i] = 0;
    size_t shared_bytes = 0;
    ptx_emit_meta_t meta;
    memset(&meta, 0, sizeof(meta));
    emit_job_t job;
    memset(&job, 0, sizeof(job));
    job.unit = unit;
    job.fn = fn;
    job.blob = &blob;
    job.kinds = kinds;
    job.shared_bytes = &shared_bytes;
    job.meta = &meta;
    job.err = err;
    job.errlen = sizeof(err);
    job.caps = caps;
    job.result = -1;
#if defined(_WIN32)
    HANDLE th = CreateThread(NULL, 8 * 1024 * 1024, emit_thread_proc, &job, 0,
                             NULL);
    if (th) {
        WaitForSingleObject(th, INFINITE);
        CloseHandle(th);
    } else {
        job.result = ptx_emit_spv(unit, fn, &blob, kinds, &shared_bytes, &meta,
                                  err, sizeof(err), caps);
    }
#else
    pthread_t th;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 8 * 1024 * 1024);
    if (pthread_create(&th, &attr, emit_thread_proc, &job) == 0) {
        pthread_join(th, NULL);
    } else {
        job.result = ptx_emit_spv(unit, fn, &blob, kinds, &shared_bytes, &meta,
                                  err, sizeof(err), caps);
    }
    pthread_attr_destroy(&attr);
#endif
    if (job.result != 0) {
        if (errbuf && errlen > 0)
            snprintf(errbuf, errlen, "PTX lower error: %s", err);
        return ACUDA_ERROR_NOT_SUPPORTED;
    }

    uint32_t param_written[ACUDA_MAX_PARAMS];
    analyze_param_writes(fn, kinds, param_written);

    acuda_kernel_t *k = calloc(1, sizeof(*k));
    if (!k) {
        spv_blob_free(&blob);
        return ACUDA_ERROR_OUT_OF_MEMORY;
    }
    k->spv = blob.words;
    k->nwords = blob.count;
    if (strlen(fn->name) >= sizeof(k->info.entry)) {
        if (errbuf && errlen > 0)
            snprintf(errbuf, errlen, "entry name too long");
        acuda_spv_free(k->spv, k->spv_va);
        free(k);
        return ACUDA_ERROR_NOT_SUPPORTED;
    }
    strcpy(k->info.entry, fn->name);
    k->info.serial = acuda_atomic_inc_u64(&kernel_serial_counter);
    if (k->info.serial == 0)
        k->info.serial = acuda_atomic_inc_u64(&kernel_serial_counter);
    k->info.block_x = 256;
    size_t dense = 0;
    for (size_t i = 0; i < fn->nparams; i++) {
        if (fn->params[i].is_retval)
            continue;
        if (dense >= ACUDA_MAX_PARAMS) {
            if (errbuf && errlen > 0)
                snprintf(errbuf, errlen, "too many params (dense=%zu max=%d nparams=%zu fn=%s)",
                         dense, ACUDA_MAX_PARAMS, fn->nparams, fn->name);
            acuda_spv_free(k->spv, k->spv_va);
            free(k);
            return ACUDA_ERROR_NOT_SUPPORTED;
        }
        k->info.is_tex[dense] = (uint32_t)fn->params[i].is_tex;
        k->info.hidden_texref[dense] =
            (uint32_t)fn->params[i].is_hidden_texref;
        if (fn->params[i].is_hidden_texref)
            strncpy(k->info.hidden_texref_name[dense], fn->params[i].name,
                    ACUDA_ENTRY_NAME_MAX - 1);
        if (fn->params[i].wide_words && kinds[i] == 5) {
            // Pointer words: the first uses the host argument, the
            // rest are continuations at +8, +16, ...
            int words = fn->params[i].wide_words;
            if (words < 1)
                words = 1;
            if (dense + (size_t)words > ACUDA_MAX_PARAMS) {
                if (errbuf && errlen > 0)
                    snprintf(errbuf, errlen, "too many params (wide dense=%zu+%d max=%d fn=%s)",
                             dense, words, ACUDA_MAX_PARAMS, fn->name);
                acuda_spv_free(k->spv, k->spv_va);
                free(k);
                return ACUDA_ERROR_NOT_SUPPORTED;
            }
            k->info.is_tex[dense] = 0;
            k->info.wide_words[dense] = 0;
            k->info.is_ptr[dense] = 1;
            k->info.param_stride[dense] = (uint32_t)words * 8u;
            dense++;
            for (int part = 1; part < words; part++) {
                k->info.is_tex[dense] = 0;
                k->info.wide_words[dense] = 0;
                k->info.is_ptr[dense] = 5;
                k->info.param_stride[dense] = 0;
                dense++;
            }
            continue;
        }
        k->info.wide_words[dense] =
            (uint32_t)fn->params[i].wide_words;
        k->info.param_ssbo[dense] =
            kinds[i] == 4 &&
                    (uint32_t)fn->params[i].wide_words >
                        ACUDA_BYVAL_SSBO_WORDS
                ? 1u : 0u;
        k->info.is_ptr[dense] = (uint32_t)kinds[i];
        {
            // Declared packed stride (what cudart lays out): wide
            // blobs take their exact declared bytes (rounded to the
            // client's packing granularity); 64-bit types 8; everything
            // else 4; continuations, hidden texrefs and unused params
            // take 0.  A 32-bit client packs tightly at 4 bytes (even
            // 8-byte .b8 blobs land at 4-aligned offsets, as observed
            // in 32-bit PhysX packed buffers); 64-bit keeps 8-byte
            // units for wide blobs.
            uint32_t wb = 0;
            if (fn->params[i].wide_words) {
                wb = fn->params[i].wide_bytes > 0
                         ? (uint32_t)fn->params[i].wide_bytes
                         : (uint32_t)fn->params[i].wide_words * 8u;
#if defined(_M_IX86) || defined(__i386__)
                wb = (wb + 3u) & ~3u;
#endif
            }
            uint32_t st = 0;
            if (fn->params[i].is_hidden_texref) {
                st = 0;
            } else if (kinds[i] == 5) {
                st = 0; // continuation inside the head's bytes
            } else if (kinds[i] == 3) {
                // Declared-but-unused: cudart still packs it, so the
                // walk must advance past it (no binding, no push).
                // A wide (by-value blob) param occupies its declared
                // client bytes even when unused (e.g. an unread .b8
                // array would otherwise desync every later argument).
                if (fn->params[i].wide_words)
                    st = wb;
                else
                    st = (fn->params[i].ty == PTY_U64 ||
                          fn->params[i].ty == PTY_F64) ? 8u : 4u;
            } else if (fn->params[i].wide_words) {
                st = wb;
            } else if (kinds[i] == 6) {
                st = 8;
            } else if (kinds[i] == 2) {
                st = 8;
            } else if (kinds[i] == 1 &&
                       (fn->params[i].ty == PTY_U64 ||
                        fn->params[i].ty == PTY_F64)) {
                st = 8;
            } else if (kinds[i] == 1 || kinds[i] == 0) {
                st = 4;
            }
            k->info.param_stride[dense] = st;
        }
        dense++;
    }
    k->info.num_params = (uint32_t)dense;
    uint32_t nptr = 0, npush = 0, ntex_meta = 0;
    for (size_t i = 0; i < dense; i++) {
        if (k->info.is_ptr[i] == 1 || k->info.is_ptr[i] == 5) {
            nptr++;
            if (k->info.is_tex[i] == 1 || k->info.is_tex[i] == 2 ||
                k->info.is_tex[i] == 3)
                ntex_meta++;
        } else if (k->info.is_ptr[i] == 6) {
            // Mixed 32-bit blob (word0 device pointer, word1 scalar):
            // bind word0's target like a plain pointer and pass word1
            // as one push word.
            nptr++;
            npush += 1;
        } else if (k->info.is_ptr[i] == 2) {
            if (npush % 2)
                npush++; // even-align 64-bit scalars (matches emitter)
            npush += 2;
        } else if (k->info.is_ptr[i] == 4) {
            if (k->info.param_ssbo[i]) {
                nptr++;
                continue;
            }
            uint32_t words = k->info.wide_words[i];
            if (words < 2)
                words = 2;
            if (npush % 2)
                npush++;
            npush += 2 * words;
        } else if (k->info.is_ptr[i] == 0) {
            npush += 1;
        }
    }
    k->info.num_ptr_params = nptr;
    {
        const char *pd = getenv("ACUDA_DUMP_PARAMS");
        if (pd && *pd) {
            fprintf(stderr, "ACuda params %s: dense=%zu nptr=%u npush=%u\n",
                    fn->name, dense, nptr, npush);
            for (size_t i = 0; i < dense; i++)
                fprintf(stderr, "  [%zu] kind=%u tex=%u hidden=%u wide=%u ssbo=%u\n",
                        i, k->info.is_ptr[i], k->info.is_tex[i],
                        k->info.hidden_texref[i] ? 1u : 0u,
                        k->info.wide_words[i], k->info.param_ssbo[i]);
        }
        uint32_t nb = 0;
        for (size_t i = 0; i < dense; i++) {
            if (k->info.is_ptr[i] == 4 && k->info.param_ssbo[i]) {
                k->info.binding_kind[nb++] = 0;
                continue;
            }
            if (k->info.is_ptr[i] == 6) {
                // Mixed blob: plain storage-buffer binding for word0's
                // target; word1 travels as a push word.
                k->info.binding_kind[nb++] = 0;
                continue;
            }
            if (k->info.is_ptr[i] != 1 && k->info.is_ptr[i] != 5)
                continue;
            k->info.binding_kind[nb++] =
                k->info.is_tex[i] == 2 ? 3u :
                (k->info.is_tex[i] == 1 ? 1u :
                 (k->info.is_tex[i] == 3 ? 2u : 0u));
        }
    }
    k->info.tex_meta_bytes = ntex_meta * 7u * 4u;
    k->full_num_ptr_params = nptr;
    k->full_num_params = (uint32_t)dense;
    k->info.push_bytes = npush * 4;
    k->info.num_regs = fn->nregs > 0xffffffffu ? 0xffffffffu
                                                  : (uint32_t)fn->nregs;
    k->info.ptx_version =
        (uint32_t)(unit->version_major * 10 + unit->version_minor);
    k->info.sm_version = unit->sm > 0 ? (uint32_t)unit->sm : 0;
    k->info.shared_bytes =
        shared_bytes > 0xffffffffu ? 0xffffffffu : (uint32_t)shared_bytes;
    k->info.surf_table = (uint32_t)meta.surf_table;
    k->info.surf_table_count = (uint32_t)meta.surf_table_count;
    for (int st = 0; st < ACUDA_SURF_TABLE_MAX_TYPES; st++) {
        k->info.surf_table_binding[st] =
            (uint32_t)meta.surf_binding[st];
        k->info.surf_table_dim[st] = (uint32_t)meta.surf_dim[st];
        k->info.surf_table_arrayed[st] =
            (uint32_t)meta.surf_arrayed[st];
        k->info.surf_table_int[st] = (uint32_t)meta.surf_int[st];
        k->info.surf_table_cas[st] =
            (uint32_t)meta.surf_table_cas[st];
        k->info.surf_table_lock[st] =
            (uint32_t)meta.surf_table_lock[st];
    }
    for (int bp = 0; bp < ACUDA_MAX_PARAMS; bp++) {
        k->info.surf_f32_cas[bp] = (uint32_t)meta.surf_f32_cas[bp];
        k->info.surf_f32_lock[bp] = (uint32_t)meta.surf_f32_lock[bp];
    }
    k->info.surf_lock = (uint32_t)meta.surf_lock;
    k->info.has_extern = 0;
    k->info.uses_subgroup = (uint32_t)meta.uses_subgroup;
    k->info.uses_barrier = (uint32_t)meta.uses_barrier;
    k->info.requires_float64 = (uint32_t)meta.requires_float64;
    k->info.bda = (uint32_t)meta.bda;
    k->info.uses_bda_access = (uint32_t)meta.uses_bda_access;
    k->info.addr_push_bytes = (uint32_t)meta.n_bda * 8u;
    for (int bp = 0; bp < ACUDA_MAX_PARAMS; bp++)
        k->info.bda_param[bp] =
            bp < meta.n_bda ? (uint32_t)meta.bda_param[bp] : 0u;
    for (int bp = 0; bp < ACUDA_MAX_PARAMS; bp++)
        k->info.param_written[bp] = param_written[bp];
    for (size_t gi = 0; gi < unit->nshvars; gi++) {
        if (unit->shvars[gi].is_extern) {
            k->info.has_extern = 1;
            break;
        }
    }
    *out_kernel = k;
    return ACUDA_SUCCESS;
}

/* 32-bit large-PTX: extract one entry + referenced module-scope decls. */
static int acuda_is_modscope_keyword(const char *p) {
    return strncmp(p, ".global", 7) == 0 || strncmp(p, ".texref", 7) == 0 ||
           strncmp(p, ".texture", 8) == 0 || strncmp(p, ".samplerref", 11) == 0 ||
           strncmp(p, ".surfref", 8) == 0 || strncmp(p, ".const", 6) == 0 ||
           strncmp(p, ".shared", 7) == 0 || strncmp(p, ".local", 6) == 0 ||
           strncmp(p, ".extern", 7) == 0;
}
static int acuda_is_modscope_decl(const char *s) {
    while (*s == ' ' || *s == '\t') s++;
    if (acuda_is_modscope_keyword(s)) return 1;
    /* prefixed: .visible/.weak/.extern before data decl (not .func/.entry defs) */
    if (strncmp(s, ".visible", 8) == 0 || strncmp(s, ".weak", 5) == 0 || strncmp(s, ".extern", 7) == 0) {
        const char *p = s;
        while (*p && *p != '\n') {
            while (*p == ' ' || *p == '\t') p++;
            if (strncmp(p, ".visible", 8) == 0) { p += 8; continue; }
            if (strncmp(p, ".weak", 5) == 0) { p += 5; continue; }
            if (strncmp(p, ".extern", 7) == 0) { p += 7; continue; }
            if (strncmp(p, ".func", 5) == 0 || strncmp(p, ".entry", 6) == 0) return 0;
            if (acuda_is_modscope_keyword(p)) return 1;
            /* skip tokens (types/attrs like .align, .b32, .param, numbers, names) */
            if (*p == '.' || *p == '_' || *p == '$' || *p == '%' || (*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z')) {
                while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != ';' && *p != '(' && *p != '[') p++;
            } else if (*p) p++;
            else break;
        }
    }
    return 0;
}
/* extract declared name from a modscope decl line into buf (best-effort) */
static void acuda_decl_name(const char *line, size_t ln, char *buf, size_t bufn) {
    buf[0] = '\0';
    /* scan tokens and keep the last identifier-like token before ; [ < or EOL */
    const char *p = line, *end = line + ln, *last = NULL; size_t lastn = 0;
    while (p < end) {
        while (p < end && (*p==' '||*p=='\t'||*p=='\r'||*p==',')) p++;
        if (p >= end || *p==';' || *p=='[' || *p=='<') break;
        const char *t = p;
        while (p < end && *p!=' '&&*p!='\t'&&*p!='\r'&&*p!=';'&&*p!='['&&*p!=','&&*p!='<'&&*p!='(') p++;
        size_t tn = (size_t)(p - t);
        if (tn > 0 && tn < 512 && ( (*t>='A'&&*t<='Z')||(*t>='a'&&*t<='z')||*t=='_'||*t=='$'||*t=='%') && *t!='.') {
            last = t; lastn = tn;
        }
        if (p < end && *p!=' '&&*p!='\t'&&*p!='\r') p++; /* consume separator-ish */
    }
    if (last && lastn < bufn) { memcpy(buf, last, lastn); buf[lastn] = '\0'; }
}
/* true if token `name` appears in [text,text+tn) (word-boundary-ish) */
static int acuda_text_has(const char *text, size_t tn, const char *name, size_t nlen) {
    if (!name || !nlen || tn < nlen) return 0;
    for (size_t i = 0; i + nlen <= tn; i++) {
        if (memcmp(text + i, name, nlen) != 0) continue;
        char b = (i==0)?' ':text[i-1];
        char a = (i+nlen>=tn)?' ':text[i+nlen];
        int bb = (b>='A'&&b<='Z')||(b>='a'&&b<='z')||(b>='0'&&b<='9')||b=='_'||b=='$';
        int ba = (a>='A'&&a<='Z')||(a>='a'&&a<='z')||(a>='0'&&a<='9')||a=='_'||a=='$';
        if (!bb && !ba) return 1;
    }
    return 0;
}
static char *acuda_ptx_extract_entry(const char *ptx, const char *entry_name) {
    if (!ptx || !entry_name || !*entry_name) return NULL;
    size_t nlen = strlen(entry_name);
    const char *p = ptx;
    /* preamble */
    char pre[4096]; size_t pre_n = 0;
    while (*p && pre_n + 512 < sizeof(pre)) {
        const char *eol = strchr(p, '\n'); size_t ln = eol ? (size_t)(eol - p) : strlen(p);
        const char *s = p; while (*s == ' ' || *s == '\t' || *s == '\r') s++;
        if (strncmp(s, ".version", 8) == 0 || strncmp(s, ".target", 7) == 0 || strncmp(s, ".address_size", 13) == 0) {
            memcpy(pre + pre_n, p, ln); pre_n += ln; pre[pre_n++] = '\n';
        }
        if (!eol)
            break;
        p = eol + 1;
        if (pre_n > 0 && (strncmp(s, ".entry", 6) == 0 || strncmp(s, ".visible", 8) == 0 || strncmp(s, ".weak", 5) == 0 || strncmp(s, ".func", 5) == 0)) break;
    }
    /* find target entry first */
    p = ptx; int in_func = 0, func_brace = 0;
    const char *ent_start = NULL, *ent_end = NULL;
    while (*p) {
        const char *eol = strchr(p, '\n'); size_t ln = eol ? (size_t)(eol - p) : strlen(p);
        const char *s = p; while (*s == ' ' || *s == '\t') s++;
        int is_entry = (strncmp(s, ".entry", 6) == 0 || strncmp(s, ".visible", 8) == 0 || strncmp(s, ".weak", 5) == 0);
        if (!in_func && is_entry && ln >= nlen) {
            for (size_t i = 0; i + nlen <= ln; i++) {
                if (memcmp(p + i, entry_name, nlen) == 0) {
                    char before = (i==0)?' ':p[i-1];
                    char after = (i+nlen>=ln)?'(':p[i+nlen];
                    if ((before==' '||before=='\t') && (after=='('||after==' '||after=='\t'||after=='\r')) {
                        ent_start = p; in_func = 1; func_brace = 0; break;
                    }
                }
            }
        }
        if (in_func) {
            for (size_t i = 0; i < ln; i++) { if (p[i]=='{') func_brace++; if (p[i]=='}') func_brace--; }
            if (ent_start && func_brace <= 0 && (memchr(p,'}',ln)!=NULL)) { ent_end = eol ? eol+1 : p+ln; break; }
        }
        if (!eol)
            break;
        p = eol + 1;
    }
    if (!ent_start || !ent_end) return NULL;
    size_t ent_n = (size_t)(ent_end - ent_start);
    /* collect module-scope decls referenced by this entry (incl. multi-line inits) */
    size_t decl_cap = 256 * 1024, decl_n = 0; char *decls = malloc(decl_cap);
    if (!decls) return NULL;
    p = ptx; in_func = 0; func_brace = 0;
    while (*p) {
        const char *eol = strchr(p, '\n'); size_t ln = eol ? (size_t)(eol - p) : strlen(p);
        const char *s = p; while (*s == ' ' || *s == '\t') s++;
        if (!in_func) {
            if (acuda_is_modscope_decl(s)) {
                /* capture full declaration statement (may span lines with { ... } init) */
                const char *ds = p; int dbr = 0, dsemi = 0; const char *q = p;
                while (*q) {
                    const char *qe = strchr(q, '\n'); size_t qn = qe ? (size_t)(qe - q) : strlen(q);
                    for (size_t i = 0; i < qn; i++) { if (q[i]=='{') dbr++; if (q[i]=='}') dbr--; if (q[i]==';' && dbr<=0) dsemi=1; }
                    q = qe ? qe + 1 : q + qn;
                    if (dsemi) break;
                    if (!qe) break;
                }
                size_t dln = (size_t)(q - ds);
                char dn[512]; acuda_decl_name(ds, ln, dn, sizeof(dn));
                if (!dn[0] || acuda_text_has(ent_start, ent_n, dn, strlen(dn))) {
                    if (decl_n + dln + 4 > decl_cap) { size_t nc = decl_cap * 2; while (nc < decl_n + dln + 4) nc *= 2; char *nd = realloc(decls, nc); if (!nd) { free(decls); return NULL; } decls = nd; decl_cap = nc; }
                    memcpy(decls + decl_n, ds, dln); decl_n += dln;
                    if (decl_n == 0 || decls[decl_n-1] != '\n') decls[decl_n++] = '\n';
                }
                p = q; continue;
            } else if (strncmp(s, ".entry", 6) == 0 || strncmp(s, ".func", 5) == 0 || strncmp(s, ".visible", 8) == 0 || strncmp(s, ".weak", 5) == 0) {
                in_func = 1; func_brace = 0;
                for (size_t i = 0; i < ln; i++) { if (p[i]=='{') func_brace++; if (p[i]=='}') func_brace--; }
            }
        } else {
            for (size_t i = 0; i < ln; i++) { if (p[i]=='{') func_brace++; if (p[i]=='}') func_brace--; }
            if (func_brace <= 0) in_func = 0;
        }
        if (!eol)
            break;
        p = eol + 1;
    }
    char *out = malloc(pre_n + decl_n + ent_n + 16);
    if (!out) { free(decls); return NULL; }
    size_t on = 0;
    memcpy(out + on, pre, pre_n); on += pre_n;
    memcpy(out + on, decls, decl_n); on += decl_n;
    memcpy(out + on, ent_start, ent_n); on += ent_n;
    out[on] = '\0';
    free(decls);
    {
        const char *dd = getenv("ACUDA_DUMP_PTX_DIR");
        if (dd && *dd) {
            static int exn;
            char safe[96]; size_t si = 0;
            for (const char *r = entry_name; *r && si < sizeof(safe)-1; r++) {
                char c = (*r>='A'&&*r<='Z')||(*r>='a'&&*r<='z')||(*r>='0'&&*r<='9')||*r=='_' ? *r : '_';
                safe[si++] = c;
            }
            safe[si] = '\0';
            char pth[1024];
            int exn_id = acuda_atomic_inc_i32(&exn) - 1;
            snprintf(pth, sizeof(pth), "%s/extracted_%d_%.60s.ptx", dd, exn_id, safe);
            FILE *fp = fopen(pth, "wb"); if (fp) { fwrite(out,1,on,fp); fclose(fp); }
        }
    }
    return out;
}

acuda_status_t acuda_compile_ptx(const char *ptx_text, const char *entry_name,
                                 acuda_kernel_t **out_kernel,
                                 char *errbuf, size_t errlen,
                                 const acuda_compiler_caps_t *caps) {
    if (!ptx_text || !out_kernel)
        return ACUDA_ERROR_INVALID_VALUE;
    *out_kernel = NULL;
    if (errbuf && errlen > 0)
        errbuf[0] = '\0';
    acuda_compiler_caps_t def_caps;
    if (!caps) {
        acuda_compiler_caps_init(&def_caps);
        caps = &def_caps;
    }

    const char *input_dump = getenv("ACUDA_DUMP_PTX_INPUT");
    if (input_dump && *input_dump) {
        static int input_serial;
        char path[1024];
        int iser = acuda_atomic_inc_i32(&input_serial) - 1;
        snprintf(path, sizeof(path), "%s/ptx_input_%d.ptx",
                 input_dump, iser);
        FILE *fp = fopen(path, "wb");
        if (fp) {
            fwrite(ptx_text, 1, strlen(ptx_text), fp);
            fclose(fp);
        }
    }
    char err[1024];
    err[0] = '\0';
    const char *ptx_src = ptx_text;
    char *ptx_extracted = NULL;
    /* 32-bit: full parse of very large PTX modules OOMs/crashes. Try to
     * extract just the target entry + module-scope decls; else fall back to
     * rtxdisam via the driver. */
    if (sizeof(void *) <= 4) {
        size_t ptxsz = strlen(ptx_text);
        if (ptxsz > (size_t)1536 * 1024) {
            if (entry_name && *entry_name)
                ptx_extracted = acuda_ptx_extract_entry(ptx_text, entry_name);
            if (ptx_extracted) {
                ptx_src = ptx_extracted;
            } else {
                if (errbuf && errlen > 0)
                    snprintf(errbuf, errlen,
                             "PTX text too large for 32-bit full parse (%zu bytes); "
                             "entry extract failed, rtxdisam fallback required", ptxsz);
                return ACUDA_ERROR_NOT_SUPPORTED;
            }
        }
    }
    ptx_unit_t unit;
    memset(&unit, 0, sizeof(unit));
    if (ptx_parse(ptx_src, &unit, err, sizeof(err)) != 0) {
        free(ptx_extracted);
        const char *dump_dir = getenv("ACUDA_DUMP_PTX_ON_ERROR");
        if (dump_dir && *dump_dir) {
            static int dump_serial;
            char path[1024];
            int dser = acuda_atomic_inc_i32(&dump_serial) - 1;
            snprintf(path, sizeof(path), "%s/ptx_error_%d.ptx",
                     dump_dir, dser);
            FILE *fp = fopen(path, "wb");
            if (fp) {
                fwrite(ptx_text, 1, strlen(ptx_text), fp);
                fclose(fp);
            }
        }
        if (errbuf && errlen > 0)
            snprintf(errbuf, errlen, "PTX parse error: %s", err);
        return ACUDA_ERROR_NOT_SUPPORTED;
    }
    ptx_func_t *fn = NULL;
    // ISA/SM ceilings (pinned by docs/ptx-baseline.md).  The default
    // .version ceiling is 9.4; ACUDA_PTX="9.4" (or "ISA 9.4") overrides
    // it.  ACUDA_SM="sm_89" (or "89") rejects a newer .target.
    // Unparseable values fall back to the default ceiling.
    {
        long ceil_maj = 9, ceil_min = 4;
        int from_env = 0;
        const char *pv = getenv("ACUDA_PTX");
        if (pv && pv[0]) {
            while (*pv == ' ' || *pv == '\t')
                pv++;
            if ((pv[0] == 'I' || pv[0] == 'i') && (pv[1] == 'S' || pv[1] == 's') &&
                (pv[2] == 'A' || pv[2] == 'a'))
                pv += 3;
            while (*pv == ' ' || *pv == '\t')
                pv++;
            char *end = NULL;
            long maj = strtol(pv, &end, 10);
            long min = 0;
            if (end != pv && maj >= 0 && maj < 100) {
                if (*end == '.') {
                    char *end2 = NULL;
                    min = strtol(end + 1, &end2, 10);
                    if (end2 == end + 1 || min < 0 || min >= 100)
                        min = -1; // unparseable minor: keep default
                }
                if (min >= 0) {
                    ceil_maj = maj;
                    ceil_min = min;
                    from_env = 1;
                }
            }
        }
        if (unit.version_major * 100 + unit.version_minor >
            (int)(ceil_maj * 100 + ceil_min)) {
            if (errbuf && errlen > 0) {
                if (from_env)
                    snprintf(errbuf, errlen,
                             "PTX ISA %d.%d newer than ACUDA_PTX=%ld.%ld",
                             unit.version_major, unit.version_minor,
                             ceil_maj, ceil_min);
                else
                    snprintf(errbuf, errlen,
                             "PTX ISA %d.%d newer than supported %ld.%ld",
                             unit.version_major, unit.version_minor,
                             ceil_maj, ceil_min);
            }
            ptx_unit_free(&unit);
            free(ptx_extracted);
            return ACUDA_ERROR_NOT_SUPPORTED;
        }
        const char *sm = getenv("ACUDA_SM");
        if (sm && sm[0]) {
            while (*sm == ' ' || *sm == '\t')
                sm++;
            if ((sm[0] == 's' || sm[0] == 'S') && (sm[1] == 'm' || sm[1] == 'M'))
                sm += 2;
            if (sm[0] == '_')
                sm++;
            char *end = NULL;
            long cap = strtol(sm, &end, 10);
            if (end != sm && cap >= 30 && cap <= 300 &&
                unit.sm > (int)cap) {
                if (errbuf && errlen > 0)
                    snprintf(errbuf, errlen, ".target sm_%d above ACUDA_SM=sm_%ld",
                             unit.sm, cap);
                ptx_unit_free(&unit);
                free(ptx_extracted);
                return ACUDA_ERROR_NOT_SUPPORTED;
            }
        }
    }
    if (entry_name) {
        for (size_t i = 0; i < unit.nfuncs; i++) {
            if (strcmp(unit.funcs[i].name, entry_name) == 0) {
                fn = &unit.funcs[i];
                break;
            }
        }
        if (!fn) {
            if (errbuf && errlen > 0)
                snprintf(errbuf, errlen, "entry '%s' not found", entry_name);
            ptx_unit_free(&unit);
            free(ptx_extracted);
            return ACUDA_ERROR_NOT_SUPPORTED;
        }
    } else {
        fn = &unit.funcs[0];
    }

    int host = ptx_hostcall_prepare(&unit, fn, err, sizeof(err));
    if (host < 0) {
        if (errbuf && errlen > 0)
            snprintf(errbuf, errlen, "PTX lower error: %s", err);
        ptx_unit_free(&unit);
        free(ptx_extracted);
        return ACUDA_ERROR_NOT_SUPPORTED;
    }

    acuda_kernel_t *k = NULL;
    acuda_status_t st = compile_func(&unit, fn, &k, errbuf, errlen, caps);
    if (st != ACUDA_SUCCESS) {
        ptx_unit_free(&unit);
        free(ptx_extracted);
        return st;
    }
    if (host > 0) {
        k->info.host_segment_count = 1;
        k->info.host_state_machine = 1;
        k->info.host_state_stride = fn->host_state_stride;
        k->info.host_shared_stride = fn->host_shared_stride;
        k->info.host_request_stride = ACUDA_HOST_REQ_STRIDE;
        uint32_t nc = (uint32_t)(unit.nextfuncs < ACUDA_MAX_HOST_CALLS
                                     ? unit.nextfuncs
                                     : ACUDA_MAX_HOST_CALLS);
        k->info.host_call_count = nc;
        for (uint32_t ci = 0; ci < nc; ci++) {
            snprintf(k->info.host_call_names[ci],
                     sizeof(k->info.host_call_names[ci]), "%.63s",
                     unit.extfuncs[ci].name);
            k->info.host_arg_bytes[ci] =
                (uint32_t)unit.extfuncs[ci].arg_bytes;
            k->info.host_ret_bytes[ci] =
                (uint32_t)unit.extfuncs[ci].ret_bytes;
            k->info.host_nargs[ci] =
                (uint32_t)unit.extfuncs[ci].nargs;
            k->info.host_ret_offsets[ci] =
                (uint32_t)fn->host_ret_off[ci];
            for (uint32_t ai = 0; ai < ACUDA_MAX_HOST_CALLS; ai++)
                k->info.host_arg_offsets[ci][ai] =
                    ai < (uint32_t)unit.extfuncs[ci].nargs
                        ? (uint32_t)unit.extfuncs[ci].arg_off[ai]
                        : 0;
        }
        // Host callbacks receive user pointers and may write them.
        for (int bp = 0; bp < ACUDA_MAX_PARAMS; bp++)
            k->info.param_written[bp] = 1;
        // Hide the two internal state/request pointer params.
        if (k->info.num_params >= 2 && k->info.num_ptr_params >= 2) {
            k->info.num_params -= 2;
            k->info.num_ptr_params -= 2;
        }
    }

    ptx_unit_free(&unit);
    free(ptx_extracted);
    *out_kernel = k;
    dump_kernel(k);
    return ACUDA_SUCCESS;
}


void acuda_kernel_free(acuda_kernel_t *kernel) {
    while (kernel) {
        acuda_kernel_t *next = kernel->next;
        acuda_spv_free(kernel->spv, kernel->spv_va);
        if (kernel->spv_path) {
            /* backing AKB temp file owned by this kernel segment */
            remove(kernel->spv_path);
            free(kernel->spv_path);
        }
        free(kernel);
        kernel = next;
    }
}

uint32_t acuda_kernel_segment_count(const acuda_kernel_t *kernel) {
    if (!kernel)
        return 0;
    if (kernel->nsegments > 0)
        return kernel->nsegments;
    uint32_t n = 0;
    for (const acuda_kernel_t *k = kernel; k; k = k->next)
        n++;
    return n;
}

acuda_status_t acuda_kernel_segment(const acuda_kernel_t *kernel,
                                    uint32_t index,
                                    acuda_kernel_t **out_segment) {
    if (!kernel || !out_segment)
        return ACUDA_ERROR_INVALID_VALUE;
    *out_segment = NULL;
    const acuda_kernel_t *k = kernel;
    for (uint32_t i = 0; i < index && k; i++)
        k = k->next;
    if (!k)
        return ACUDA_ERROR_INVALID_VALUE;
    *out_segment = (acuda_kernel_t *)k;
    return ACUDA_SUCCESS;
}

uint32_t acuda_kernel_launch_ptr_params(const acuda_kernel_t *kernel) {
    if (!kernel)
        return 0;
    return kernel->full_num_ptr_params;
}

acuda_status_t acuda_ptx_validate(const char *ptx_text,
                                  char *errbuf, size_t errlen) {
    if (!ptx_text)
        return ACUDA_ERROR_INVALID_VALUE;
    char err[1024];
    ptx_unit_t unit;
    memset(&unit, 0, sizeof(unit));
    if (ptx_parse(ptx_text, &unit, err, sizeof(err)) != 0) {
        if (errbuf && errlen > 0)
            snprintf(errbuf, errlen, "PTX parse error: %s", err);
        ptx_unit_free(&unit);
        return ACUDA_ERROR_NOT_SUPPORTED;
    }
    ptx_unit_free(&unit);
    return ACUDA_SUCCESS;
}

acuda_status_t acuda_kernel_info(const acuda_kernel_t *kernel,
                                 acuda_kernel_info_t *out_info) {
    if (!kernel || !out_info)
        return ACUDA_ERROR_INVALID_VALUE;
    *out_info = kernel->info;
    return ACUDA_SUCCESS;
}

const acuda_kernel_info_t *acuda_kernel_info_ptr(const acuda_kernel_t *kernel) {
    if (!kernel)
        return NULL;
    return &kernel->info;
}

acuda_status_t acuda_kernel_release_spirv(acuda_kernel_t *kernel) {
    if (!kernel) return ACUDA_ERROR_INVALID_VALUE;
    if (kernel->spv_path && kernel->spv) {
        acuda_spv_free(kernel->spv, kernel->spv_va);
        kernel->spv = NULL;
        kernel->spv_va = 0;
    }
    return ACUDA_SUCCESS;
}

acuda_status_t acuda_kernel_spirv(const acuda_kernel_t *kernel,
                                  const uint32_t **out_words,
                                  size_t *out_word_count) {
    if (!kernel || !out_words || !out_word_count)
        return ACUDA_ERROR_INVALID_VALUE;
    acuda_kernel_t *mk = (acuda_kernel_t *)kernel; /* mutable cache */
    if (!mk->spv && mk->spv_path && mk->nwords) {
        FILE *f = fopen(mk->spv_path, "rb");
        if (!f) return ACUDA_ERROR_INVALID_VALUE;
        if (fseek(f, mk->spv_file_off, SEEK_SET) != 0) { fclose(f); return ACUDA_ERROR_INVALID_VALUE; }
        mk->spv = (uint32_t *)acuda_spv_alloc((size_t)mk->nwords * 4, &mk->spv_va);
        if (!mk->spv) { fclose(f); return ACUDA_ERROR_OUT_OF_MEMORY; }
        if (fread(mk->spv, 4, mk->nwords, f) != mk->nwords) {
            acuda_spv_free(mk->spv, mk->spv_va); mk->spv = NULL; mk->spv_va = 0;
            fclose(f); return ACUDA_ERROR_INVALID_VALUE;
        }
        fclose(f);
    }
    *out_words = mk->spv;
    *out_word_count = mk->nwords;
    return ACUDA_SUCCESS;
}

#define ACUDA_KBLOB_MAGIC 0x31424b41u /* AKB1 */

acuda_status_t acuda_kernel_serialize(const acuda_kernel_t *kernel,
                                      uint8_t **out_data, size_t *out_size) {
    if (!kernel || !out_data || !out_size) return ACUDA_ERROR_INVALID_VALUE;
    *out_data = NULL; *out_size = 0;
    uint32_t nseg = acuda_kernel_segment_count(kernel);
    if (nseg == 0) nseg = 1;
    size_t sz = 8;
    for (uint32_t i = 0; i < nseg; i++) {
        acuda_kernel_t *seg = NULL;
        if (acuda_kernel_segment(kernel, i, &seg) != ACUDA_SUCCESS || !seg)
            return ACUDA_ERROR_INVALID_VALUE;
        sz += 20 + sizeof(acuda_kernel_info_t) + (size_t)seg->nwords * 4;
    }
    uint8_t *buf = (uint8_t *)malloc(sz);
    if (!buf) return ACUDA_ERROR_OUT_OF_MEMORY;
    uint8_t *p = buf;
    uint32_t hdr[2] = { ACUDA_KBLOB_MAGIC, nseg };
    memcpy(p, hdr, 8); p += 8;
    for (uint32_t i = 0; i < nseg; i++) {
        acuda_kernel_t *seg = NULL;
        acuda_kernel_segment(kernel, i, &seg);
        uint32_t meta[5];
        meta[0] = (uint32_t)seg->nwords;
        meta[1] = seg->full_num_ptr_params;
        meta[2] = seg->full_num_params;
        meta[3] = (i == 0) ? seg->nsegments : 0;
        meta[4] = (uint32_t)sizeof(acuda_kernel_info_t);
        memcpy(p, meta, 20); p += 20;
        memcpy(p, &seg->info, sizeof(seg->info)); p += sizeof(seg->info);
        if (seg->nwords) {
            if (seg->spv) {
                memcpy(p, seg->spv, (size_t)seg->nwords * 4); p += (size_t)seg->nwords * 4;
            } else if (seg->spv_path) {
                FILE *f = fopen(seg->spv_path, "rb");
                if (!f || fseek(f, seg->spv_file_off, SEEK_SET) != 0 ||
                    fread(p, 4, seg->nwords, f) != seg->nwords) { if (f) fclose(f); free(buf); return ACUDA_ERROR_INVALID_VALUE; }
                fclose(f); p += (size_t)seg->nwords * 4;
            } else { free(buf); return ACUDA_ERROR_INVALID_VALUE; }
        }
    }
    *out_data = buf; *out_size = (size_t)(p - buf);
    return ACUDA_SUCCESS;
}

acuda_status_t acuda_kernel_deserialize(const uint8_t *data, size_t size,
                                        acuda_kernel_t **out_kernel) {
    if (!data || size < 8 || !out_kernel) return ACUDA_ERROR_INVALID_VALUE;
    *out_kernel = NULL;
    const uint8_t *p = data;
    uint32_t hdr[2]; memcpy(hdr, p, 8); p += 8;
    if (hdr[0] != ACUDA_KBLOB_MAGIC) return ACUDA_ERROR_INVALID_VALUE;
    uint32_t nseg = hdr[1];
    if (nseg == 0 || nseg > 256) return ACUDA_ERROR_INVALID_VALUE;
    acuda_kernel_t *head = NULL, *prev = NULL;
    acuda_status_t bst = ACUDA_ERROR_INVALID_VALUE;
    for (uint32_t i = 0; i < nseg; i++) {
        if ((size_t)(p - data) + 20 > size) goto bad;
        uint32_t meta[5]; memcpy(meta, p, 20); p += 20;
        uint32_t nwords = meta[0], infosz = meta[4];
        if (infosz == 0 || infosz > 1024*1024) goto bad;
        if ((size_t)(p - data) + infosz + (size_t)nwords * 4 > size) goto bad;
        acuda_kernel_t *k = (acuda_kernel_t *)calloc(1, sizeof(*k));
        if (!k) { bst = ACUDA_ERROR_OUT_OF_MEMORY; goto bad; }
        k->nwords = nwords;
        k->full_num_ptr_params = meta[1];
        k->full_num_params = meta[2];
        if (i == 0) k->nsegments = meta[3] ? meta[3] : nseg;
        memset(&k->info, 0, sizeof(k->info));
        memcpy(&k->info, p, infosz < sizeof(k->info) ? infosz : sizeof(k->info));
        p += infosz;
        if (nwords) {
            k->spv = (uint32_t *)acuda_spv_alloc((size_t)nwords * 4, &k->spv_va);
            if (!k->spv) { free(k); bst = ACUDA_ERROR_OUT_OF_MEMORY; goto bad; }
            memcpy(k->spv, p, (size_t)nwords * 4); p += (size_t)nwords * 4;
        }
        if (!head) head = k; else prev->next = k;
        prev = k;
    }
    *out_kernel = head;
    return ACUDA_SUCCESS;
bad:
    if (head) acuda_kernel_free(head);
    return bst;
}

acuda_status_t acuda_kernel_deserialize_file(const char *path,
                                             acuda_kernel_t **out_kernel) {
    if (!path || !out_kernel) return ACUDA_ERROR_INVALID_VALUE;
    *out_kernel = NULL;
    FILE *f = fopen(path, "rb");
    if (!f) return ACUDA_ERROR_INVALID_VALUE;
    uint32_t hdr[2];
    if (fread(hdr, 4, 2, f) != 2 || hdr[0] != ACUDA_KBLOB_MAGIC) { fclose(f); return ACUDA_ERROR_INVALID_VALUE; }
    uint32_t nseg = hdr[1];
    if (nseg == 0 || nseg > 256) { fclose(f); return ACUDA_ERROR_INVALID_VALUE; }
    acuda_kernel_t *head = NULL, *prev = NULL;
    acuda_status_t bst = ACUDA_ERROR_INVALID_VALUE;
    for (uint32_t i = 0; i < nseg; i++) {
        uint32_t meta[5];
        if (fread(meta, 4, 5, f) != 5) goto bad;
        uint32_t nwords = meta[0], infosz = meta[4];
        if (infosz == 0 || infosz > 1024*1024) goto bad;
        acuda_kernel_t *k = (acuda_kernel_t *)calloc(1, sizeof(*k));
        if (!k) { bst = ACUDA_ERROR_OUT_OF_MEMORY; goto bad; }
        k->nwords = nwords;
        k->full_num_ptr_params = meta[1];
        k->full_num_params = meta[2];
        if (i == 0) k->nsegments = meta[3] ? meta[3] : nseg;
        memset(&k->info, 0, sizeof(k->info));
        {
            uint32_t toread = infosz < (uint32_t)sizeof(k->info) ? infosz : (uint32_t)sizeof(k->info);
            if (fread(&k->info, 1, toread, f) != toread) { free(k); goto bad; }
            if (infosz > toread) fseek(f, (long)(infosz - toread), SEEK_CUR);
        }
        if (nwords) {
            size_t spv_bytes = (size_t)nwords * 4;
            int use_backing = (sizeof(void *) <= 4 && spv_bytes >= (8u * 1024u * 1024u));
            if (use_backing) {
                long off = ftell(f);
                if (off < 0) { free(k); goto bad; }
                { size_t pl = strlen(path) + 1; k->spv_path = (char*)malloc(pl); if (k->spv_path) memcpy(k->spv_path, path, pl); }
                k->spv_file_off = off;
                if (!k->spv_path) { free(k); bst = ACUDA_ERROR_OUT_OF_MEMORY; goto bad; }
                if (fseek(f, (long)spv_bytes, SEEK_CUR) != 0) { free(k->spv_path); free(k); goto bad; }
            } else {
                k->spv = (uint32_t *)acuda_spv_alloc(spv_bytes, &k->spv_va);
                if (!k->spv) { free(k); bst = ACUDA_ERROR_OUT_OF_MEMORY; goto bad; }
                if (fread(k->spv, 4, nwords, f) != nwords) { acuda_spv_free(k->spv, k->spv_va); free(k); goto bad; }
            }
        }
        if (!head) head = k; else prev->next = k;
        prev = k;
    }
    fclose(f);
    /* if no segment uses the file as SPIR-V backing, remove temp AKB */
    {
        int need = 0;
        for (acuda_kernel_t *t = head; t; t = t->next) if (t->spv_path) { need = 1; break; }
        if (!need) remove(path);
    }
    *out_kernel = head;
    return ACUDA_SUCCESS;
bad:
    fclose(f);
    if (head) acuda_kernel_free(head);
    return bst;
}
