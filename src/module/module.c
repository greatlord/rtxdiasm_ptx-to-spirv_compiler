// Module-level emission: capabilities/types/constants, entry point and
// execution modes, per-function prelude, and the top-level ptx_emit_spv.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#include <windows.h>
#endif
#if defined(_MSC_VER)
#define strcasecmp _stricmp
#else
#include <strings.h>
#endif
#include "spirv_internal.h"


static int cm_find_reg(ctx_t *c, const char *name) {
    for (int i = 0; i < c->nregs; i++)
        if (strcmp(c->regs[i].name, name) == 0)
            return i;
    return -1;
}

static int cm_ready(ctx_t *c, int comp) {
    (void)c;
    if (comp == 0)
        return c->caps.coopmat_f16;
    if (comp == 1)
        return c->caps.coopmat_bf16;
    return c->caps.coopmat_f16 || c->caps.coopmat_bf16;
}

static void cm_mark(ctx_t *c, const char *name, int comp, int use) {
    int ri = cm_find_reg(c, name);
    if (ri < 0)
        return;
    reg_t *r = &c->regs[ri];
    r->is_coopmat = 1;
    r->coopmat_comp = comp;
    r->coopmat_use = use;
    r->coopmat_ty = acuda_coopmat_type(c, comp, use);
}

// Decide once per function whether every WMMA shape/type can use a real
// cooperative matrix. If not, none of the fragments are marked and the
// existing scalar lowering is used for the whole function.
static void prepare_coopmat(ctx_t *c, const ptx_func_t *fn) {
    if (!c->caps.coopmat_f16 && !c->caps.coopmat_bf16)
        return;
    int ok = 1;
    for (size_t bi = 0; bi < fn->nblocks && ok; bi++) {
        for (size_t ii = 0; ii < fn->blocks[bi].count && ok; ii++) {
            const ptx_instr_t *in = &fn->blocks[bi].instrs[ii];
            if (in->op != POP_WMMA_LOAD && in->op != POP_WMMA_STORE &&
                in->op != POP_WMMA_MMA)
                continue;
            if (in->mma_elem == 2 || in->wmma_shape != 0 ||
                in->mma_f16acc) {
                ok = 0;
                break;
            }
            if (in->op == POP_WMMA_MMA) {
                int comp = in->mma_elem == 1 ? 1 : 0;
                int compc = in->mma_f16acc ? 0 : 2;
                if (!cm_ready(c, comp) || !cm_ready(c, compc))
                    ok = 0;
            } else if (in->op == POP_WMMA_LOAD) {
                int comp;
                if (in->wmma_kind <= 1)
                    comp = in->mma_elem ? 1 : 0;
                else
                    comp = in->mma_f16acc ? 0 : 2;
                if (!cm_ready(c, comp))
                    ok = 0;
            } else {
                int comp = in->mma_f16acc ? 0 : 2;
                if (!cm_ready(c, comp))
                    ok = 0;
            }
        }
    }
    if (!ok)
        return;
    for (size_t bi = 0; bi < fn->nblocks; bi++) {
        for (size_t ii = 0; ii < fn->blocks[bi].count; ii++) {
            const ptx_instr_t *in = &fn->blocks[bi].instrs[ii];
            if (in->op == POP_WMMA_MMA) {
                int comp = in->mma_elem == 1 ? 1 : 0;
                int compc = in->mma_f16acc ? 0 : 2;
                cm_mark(c, in->wregs[1][0], comp, 0);
                cm_mark(c, in->wregs[2][0], comp, 1);
                cm_mark(c, in->wregs[3][0], compc, 2);
                cm_mark(c, in->wregs[0][0], compc, 2);
            } else if (in->op == POP_WMMA_LOAD) {
                int comp, use;
                if (in->wmma_kind <= 1) {
                    comp = in->mma_elem ? 1 : 0;
                    use = in->wmma_kind;
                } else {
                    comp = in->mma_f16acc ? 0 : 2;
                    use = 2;
                }
                cm_mark(c, in->wregs[0][0], comp, use);
            } else if (in->op == POP_WMMA_STORE) {
                int comp = in->mma_f16acc ? 0 : 2;
                cm_mark(c, in->wregs[0][0], comp, 2);
            }
        }
    }
}


static void detect_shared_coopmat(ctx_t *c, const ptx_func_t *fn) {
    if (!c->caps.coopmat_f16 && !c->caps.coopmat_bf16)
        return;
    int comp = -1;
    for (size_t bi = 0; bi < fn->nblocks; bi++) {
        for (size_t ii = 0; ii < fn->blocks[bi].count; ii++) {
            const ptx_instr_t *in = &fn->blocks[bi].instrs[ii];
            if (in->op == POP_LD || in->op == POP_ST ||
                in->op == POP_ATOM || in->op == POP_ATOMCAS) {
                // A generic access may alias the shared allocation.
                if (in->space == PSP_SHARED || in->space == PSP_GENERIC)
                    return;
            }
            if (in->op == POP_BARRED || in->op == POP_LDMATRIX)
                return;
            if (in->op == POP_BAR &&
                (in->bar_cluster || in->bar_arrive || in->nsrc > 0))
                return;
            if (in->op == POP_WMMA_STORE && in->space == PSP_SHARED)
                return;
            if (in->op == POP_CPASYNC &&
                (in->src_size != in->cp_size || in->nsrc >= 4))
                return;
            if (in->op == POP_WMMA_LOAD && in->space == PSP_SHARED &&
                in->wmma_shape != 0)
                return;
            if (in->op == POP_WMMA_LOAD && in->space == PSP_SHARED &&
                in->wmma_shape == 0) {
                // A shared f32 C fragment cannot share the 16-bit view.
                if (in->wmma_kind == 2 && !in->mma_f16acc)
                    return;
                int cc = in->mma_elem == 1 ? 1 : 0;
                if (comp < 0)
                    comp = cc;
                else if (comp != cc)
                    return;
            }
        }
    }
    if (comp < 0)
        return;
    for (int i = 0; i < c->nshvars; i++) {
        if (c->shvars[i].is_extern || c->shvars[i].elem != 1 ||
            (c->shvars[i].bytes % 2) != 0)
            return;
    }
    c->sh_coop_comp = comp;
    c->sh_coop_typed = 1;
}

int ptx_emit_spv(const ptx_unit_t *unit, const ptx_func_t *fn,
                 spv_blob_t *blob, int *param_kinds, size_t *shared_bytes,
                 ptx_emit_meta_t *meta, char *err, size_t errlen,
                 const acuda_compiler_caps_t *caps) {
    ctx_t *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        snprintf(err, errlen, "out of memory");
        return -1;
    }
    // `done:` always calls cfg_free, including early failures before
    // control-flow analysis; keep the struct initialized from the start
    // so those paths free null pointers rather than stack garbage.
    cfg_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    for (int mi = 0; mi < MAX_PARAMS; mi++) {
        ctx->tex_meta_word[mi] = -1;
        ctx->tex_image_ty[mi] = 0;
    }
    if (caps)
        ctx->caps = *caps;
    else
        acuda_compiler_caps_init(&ctx->caps);
    ctx->err = err;
     ctx->addr_size = unit->addr_size ? unit->addr_size : 64;
    ctx->errlen = errlen;
    ctx->next_id = 1;
    ctx->sp_pack_gen = (uint32_t)-1;
    err[0] = '\0';
    uint32_t *flow_tramp = NULL, *flow_exit = NULL;
    uint32_t flow_cur = 0;

    // f64 emulation request: demote doubles to floats (approximate;
    // explicit opt-in only, native path untouched when unset).
    {
        const char *e = getenv("ACUDA_EMU_F64");
        ctx->emu_f64 = (e && (strcmp(e, "1") == 0 || strcasecmp(e, "true") == 0 ||
                             strcasecmp(e, "yes") == 0))
                          ? 1
                          : 0;
    }
    // Float64 capability/type are emitted only when f64 is actually used
    // (declaring them unconditionally would needlessly require the
    // device feature for every kernel, including f32-only ones).
    {
        int seen = 0;
        for (size_t i = 0; i < fn->nregs && !seen; i++)
            seen = fn->regs[i].ty == PTY_F64;
        for (size_t bi = 0; bi < fn->nblocks && !seen; bi++)
            for (size_t ii = 0; ii < fn->blocks[bi].count && !seen; ii++)
                seen = fn->blocks[bi].instrs[ii].ty == PTY_F64;
        ctx->uses_f64 = seen;
    }
    // GroupNonUniformShuffle/Ballot capabilities only with their
    // instructions (match and mma ride the Shuffle capability: both
    // lower to OpGroupNonUniformShuffle with no other requirements).
    {
        int seen = 0;
        for (size_t bi = 0; bi < fn->nblocks && !seen; bi++)
            for (size_t ii = 0; ii < fn->blocks[bi].count && !seen; ii++)
                seen = fn->blocks[bi].instrs[ii].op == POP_SHFL ||
                       fn->blocks[bi].instrs[ii].op == POP_MATCH ||
                       fn->blocks[bi].instrs[ii].op == POP_MMA ||
                       fn->blocks[bi].instrs[ii].op == POP_LDMATRIX ||
                       fn->blocks[bi].instrs[ii].op == POP_WMMA_LOAD ||
                       fn->blocks[bi].instrs[ii].op == POP_WMMA_STORE ||
                       fn->blocks[bi].instrs[ii].op == POP_WMMA_MMA;
        ctx->uses_shfl = seen;
    }
    {
        int seen = 0;
        for (size_t bi = 0; bi < fn->nblocks && !seen; bi++)
            for (size_t ii = 0; ii < fn->blocks[bi].count && !seen; ii++)
                seen = fn->blocks[bi].instrs[ii].op == POP_VOTE ||
                       fn->blocks[bi].instrs[ii].op == POP_BARRED;
        ctx->uses_ballot = seen;
    }
    {
        int seen = 0;
        for (size_t bi = 0; bi < fn->nblocks && !seen; bi++)
            for (size_t ii = 0; ii < fn->blocks[bi].count && !seen; ii++)
                seen = fn->blocks[bi].instrs[ii].op == POP_REDUX;
        ctx->uses_redux = seen;
    }
    {
        int seen = 0;
        for (size_t bi = 0; bi < fn->nblocks && !seen; bi++)
            for (size_t ii = 0; ii < fn->blocks[bi].count && !seen; ii++) {
                const ptx_instr_t *in = &fn->blocks[bi].instrs[ii];
                // POP_FMA covers fma/mad (including the startup probe);
                // directed add/mul keep the flag set for older paths.
                seen = in->op == POP_FMA ||
                       ((in->op == POP_ADD || in->op == POP_MULLO) &&
                        in->frnd != 0);
            }
        ctx->uses_frnd = seen;
    }
    ctx->use_fma = ctx->uses_frnd && ctx->caps.shader_fma;
    ctx->use_fma_f64 = ctx->uses_frnd && ctx->caps.shader_fma_f64;
    {
        int seen = 0;
        for (size_t bi = 0; bi < fn->nblocks && !seen; bi++)
            for (size_t ii = 0; ii < fn->blocks[bi].count && !seen; ii++) {
                const ptx_instr_t *in = &fn->blocks[bi].instrs[ii];
                for (int si = 0; si < in->nsrc; si++) {
                    if (in->src[si].kind == OPD_SPECIAL &&
                        (in->src[si].special == SPC_CLOCK ||
                         in->src[si].special == SPC_CLOCK64)) {
                        seen = 1;
                        break;
                    }
                }
            }
        ctx->uses_shader_clock = seen;
    }
    if (ctx->uses_shader_clock &&
        !ctx->caps.shader_clock_device && !ctx->caps.shader_clock_subgroup) {
        fail(ctx, "%clock/%clock64 requires VK_KHR_shader_clock "
                  "(shaderDeviceClock or shaderSubgroupClock)");
        goto done;
    }
    {
        int seen = 0;
        for (size_t bi = 0; bi < fn->nblocks && !seen; bi++)
            for (size_t ii = 0; ii < fn->blocks[bi].count && !seen; ii++)
                seen = fn->blocks[bi].instrs[ii].op == POP_DP4A ||
                       (fn->blocks[bi].instrs[ii].op == POP_MMA &&
                        (fn->blocks[bi].instrs[ii].mma_elem == 2 ||
                         fn->blocks[bi].instrs[ii].mma_elem == 7));
        ctx->use_dot_product = seen && ctx->caps.dot_product &&
                              (ctx->caps.dot_signed || ctx->caps.dot_unsigned ||
                               ctx->caps.dot_mixed);
    }
    if (ctx->caps.mixed_fdot) {
        for (size_t bi = 0; bi < fn->nblocks && !ctx->uses_fdot2; bi++)
            for (size_t ii = 0;
                 ii < fn->blocks[bi].count && !ctx->uses_fdot2; ii++) {
                const ptx_instr_t *ti = &fn->blocks[bi].instrs[ii];
                if ((ti->op == POP_MMA && ti->mma_elem == 0 &&
                     !ti->mma_f16acc) || ti->op == POP_DOT2)
                    ctx->uses_fdot2 = 1;
            }
    }
    ctx->zero_init_wg = ctx->caps.zero_init_wg;
    {
        int seen = 0;
        for (size_t bi = 0; bi < fn->nblocks && !seen; bi++)
            for (size_t ii = 0; ii < fn->blocks[bi].count && !seen; ii++)
                seen = fn->blocks[bi].instrs[ii].op == POP_BRKPT;
        ctx->uses_brkpt = seen;
        if (seen)
            ctx->v_file = nid(ctx);
    }
    ctx->use_fc32 = ctx->caps.fc32;
    ctx->use_fc64 = ctx->caps.fc64 && ctx->uses_f64 && !ctx->emu_f64;
    ctx->use_vmm = ctx->caps.vmm;
    {
        // Native byte storage-buffer access only applies to .b8/.u8
        // global/generic ld/st; shared and Private stay word-packed.
        int seen = 0;
        for (size_t bi = 0; bi < fn->nblocks && !seen; bi++)
            for (size_t ii = 0; ii < fn->blocks[bi].count && !seen; ii++) {
                const ptx_instr_t *in = &fn->blocks[bi].instrs[ii];
                if ((in->op == POP_LD || in->op == POP_ST) &&
                    in->ty == PTY_U8 &&
                    (in->space == PSP_GLOBAL || in->space == PSP_GENERIC))
                    seen = 1;
            }
        ctx->use_ssbo8 = seen && ctx->caps.ssbo8;
    }
    {
        int seen = 0;
        for (size_t bi = 0; bi < fn->nblocks && !seen; bi++)
            for (size_t ii = 0; ii < fn->blocks[bi].count && !seen; ii++) {
                const ptx_instr_t *in = &fn->blocks[bi].instrs[ii];
                if ((in->op == POP_LD || in->op == POP_ST) &&
                    in->ty == PTY_U16 &&
                    (in->space == PSP_GLOBAL || in->space == PSP_GENERIC))
                    seen = 1;
            }
        ctx->use_ssbo16 = seen && ctx->caps.ssbo16;
    }
    {
        int seen = 0;
        for (size_t bi = 0; bi < fn->nblocks && !seen; bi++)
            for (size_t ii = 0; ii < fn->blocks[bi].count && !seen; ii++) {
                const ptx_instr_t *in = &fn->blocks[bi].instrs[ii];
                seen = (in->op == POP_ATOM || in->op == POP_ATOMCAS) &&
                       (in->ty == PTY_U64 || in->ty == PTY_F64);
            }
        ctx->uses_i64_atomics = seen;
    }
    {
        int any_sw = 0;
        for (size_t bi = 0; bi < fn->nblocks; bi++)
            for (size_t ii = 0; ii < fn->blocks[bi].count; ii++) {
                const ptx_instr_t *in = &fn->blocks[bi].instrs[ii];
                if (in->op != POP_BAR)
                    continue;
                if (in->bar_cluster || in->bar_arrive || in->nsrc > 0) {
                    any_sw = 1;
                    if (in->arg >= 0 && in->arg < 17)
                        ctx->bar_sw_id[in->arg] = 1;
                }
            }
        ctx->uses_swbar = any_sw;
    }
    ctx->uses_subid = ctx->uses_shfl || ctx->uses_ballot || ctx->uses_redux;
    for (size_t bi = 0; bi < fn->nblocks; bi++)
        for (size_t ii = 0; ii < fn->blocks[bi].count; ii++) {
            const ptx_instr_t *ti = &fn->blocks[bi].instrs[ii];
            if (ti->op == POP_TXQ)
                ctx->uses_image_query = 1;
            if (ti->op == POP_TEX || ti->op == POP_TLD4) {
                // 1D textures are lowered as 2D tiled images.
                if (ti->tex_geom == 7)
                    ctx->uses_cube_array = 1;
                if (ti->tex_noff > 0)
                    ctx->uses_gather_ext = 1;
            }
            if ((ti->op == POP_SULD || ti->op == POP_SUST ||
                 ti->op == POP_SURED) &&
                (ti->surf_geom == 1 || ti->surf_geom == 4)) {
                ctx->uses_sampled1d = 1;
                ctx->uses_image1d = 1;
            }
        }
    if (ctx->emu_f64 && ctx->uses_f64) {
        // Warn once per process: demotion is approximate by design.
        static int warned = 0;
        if (!warned) {
            warned = 1;
            fprintf(stderr, "ACUDA_EMU_F64=1: f64 demoted to f32 "
                            "(approximate, not bit-exact)\n");
        }
    }

    // Static .shared variables come from the unit (module scope).
    ctx->shvars = unit->nshvars > 0 ? unit->shvars : NULL;
    ctx->nshvars = (int)unit->nshvars;
    size_t shbytes = 0;
    for (size_t i = 0; i < unit->nshvars; i++) {
        size_t sb = unit->shvars[i].bytes;
        if (!unit->shvars[i].is_extern) {
            // Match the 16-byte rounding used for the Workgroup arrays.
            sb = sb < 16 ? 16 : (sb + 15) & ~(size_t)15;
        }
        shbytes += sb;
    }
    if (shared_bytes)
        *shared_bytes = shbytes;
    // Function-scope .local variables come from the entry (Private
    // per-invocation storage; excluded from the shared-bytes tally).
    ctx->locals = fn->nlocals > 0 ? fn->locals : NULL;
    ctx->nlocals = (int)fn->nlocals;
    ctx->uses_commons = 0;
    for (int li = 0; li < ctx->nlocals; li++)
        if (ctx->locals[li].is_common)
            ctx->uses_commons = 1;
    // Initialized module-scope .const arrays (loads materialize).
    ctx->consts = unit->nconsts > 0 ? unit->consts : NULL;
    ctx->nconsts = (int)unit->nconsts;
    ctx->const_bank_bytes = (uint32_t)unit->const_bank_bytes;
    ctx->const_bank_size = (uint32_t)unit->const_bank_bytes;
    ctx->has_const_bank = fn->uses_dynamic_const;
    // Initialized module-scope .global arrays (loads served from a
    // read-only Private image; stores stay clean errors).
    ctx->globals = unit->nglobals > 0 ? unit->globals : NULL;
    ctx->nglobals = (int)unit->nglobals;
    ctx->fnparams = fn->nparams > 0 ? fn->params : NULL;
    ctx->nfnparams = (int)fn->nparams;
    ctx->fns = unit->funcs;
    ctx->cur_fn = fn;
    ctx->host_machine = fn->host_calls;
    ctx->host_state_stride = fn->host_state_stride;
    ctx->host_call_count = unit->nextfuncs > 32 ? 32 : (int)unit->nextfuncs;
    for (int hi = 0; hi < 32; hi++)
        ctx->host_ret_off[hi] = (uint32_t)fn->host_ret_off[hi];
    ctx->host_local_base = fn->host_local_base;
    ctx->host_nlocals = (int)fn->nlocals;
    for (int li = 0; li < 32; li++)
        ctx->host_local_off[li] = fn->host_local_off[li];
    ctx->host_shared_stride = fn->host_shared_stride;
    for (int si = 0; si < 32; si++)
        ctx->host_shared_off[si] = fn->host_shared_off[si];
    ctx->host_global_base = fn->host_global_base;
    ctx->host_nglobals = (int)unit->nglobals;
    for (int gi = 0; gi < ACUDA_MAX_GLOBALS; gi++)
        ctx->host_global_off[gi] = fn->host_global_off[gi];
    ctx->nfns = (int)unit->nfuncs;
    ctx->extfuncs = unit->extfuncs;
    ctx->nextfuncs = (int)unit->nextfuncs;

    // Parameter analysis: pointers -> descriptor bindings, scalars ->
    // push words (.u64 scalars take two). Retval slots take no arg slot.
    // NOTE: analyze failure must mark the context failed (its message is
    // already in err); otherwise we would return a bogus empty success.
    // `.reg`-form functions are call-only (no launch ABI for register
    // parameters): selecting one as the entry is an honest error.
    if (fn->nrparams > 0) {
        snprintf(err, errlen,
                 "cannot launch .func with .reg params (only via call)");
        ctx->failed = 1;
        goto done;
    }
    if (fn->param_kinds_override) {
        for (size_t pi = 0; pi < fn->nparams; pi++)
            param_kinds[pi] = fn->param_kinds_override[pi];
    } else if (analyze_params(unit, fn, param_kinds, err, errlen) != 0) {
        ctx->failed = 1;
        goto done;
    }
    if (getenv("ACUDA_DEBUG_CLASSIFY")) {
        for (size_t di = 0; di < fn->nparams; di++)
            fprintf(stderr, "classify %s -> %d%s\n", fn->params[di].name,
                    param_kinds[di],
                    fn->params[di].is_retval ? " (retval)" : "");
    }
    if (getenv("ACUDA_DEBUG_SYMBOLS")) {
        for (size_t di = 0; di < fn->nparams; di++)
            fprintf(stderr, "ACuda: fnparam[%zu] %s ret=%d hidden=%d kind=%d\n",
                    di, fn->params[di].name, fn->params[di].is_retval,
                    fn->params[di].is_hidden_symbol,
                    (int)param_kinds[di]);
    }
    for (size_t i = 0; i < fn->nparams; i++) {
        if (fn->params[i].is_retval)
            continue;
        if (param_kinds[i] == 3)
            continue; // unused: no descriptor, no push words, no arg read
        if (param_kinds[i] == 5) {
            // By-value struct whose words are device pointers: one
            // binding per word, all packed from a single host argument
            // by the launcher.
            int words = fn->params[i].wide_words;
            if (words < 1)
                words = 1;
            if (ctx->ncparams + words > MAX_PARAMS) {
                char _em[128]; snprintf(_em,sizeof(_em),"too many params (nc=%d+%d>%d)",ctx->ncparams,words,MAX_PARAMS);
                fail(ctx, _em);
                goto done;
            }
            for (int part = 0; part < words; part++) {
                cparam_t *hp = &ctx->cparams[ctx->ncparams++];
                int n = part
                    ? snprintf(hp->name, sizeof(hp->name), "%s#%d",
                               fn->params[i].name, part)
                    : snprintf(hp->name, sizeof(hp->name), "%s",
                               fn->params[i].name);
                if (n < 0 || (size_t)n >= sizeof(hp->name)) {
                    fail(ctx, "param name too long");
                    goto done;
                }
                hp->ty = fn->params[i].ty;
                hp->kind = 1;
                hp->is_tex = 0;
                hp->binding = ctx->nbindings++;
                hp->push_word = -1;
                hp->push_words = 0;
            }
            continue;
        }
        if (ctx->ncparams >= MAX_PARAMS) {
            char _em[128]; snprintf(_em,sizeof(_em),"too many params (nc=%d>=%d)",ctx->ncparams,MAX_PARAMS);
            fail(ctx, _em);
            goto done;
        }
        cparam_t *cp = &ctx->cparams[ctx->ncparams++];
        if (strlen(fn->params[i].name) >= sizeof(cp->name)) {
            fail(ctx, "param name too long");
            goto done;
        }
        memset(cp, 0, sizeof(*cp));
        strcpy(cp->name, fn->params[i].name);
        cp->ty = fn->params[i].ty;
        cp->kind = param_kinds[i];
        cp->is_tex = fn->params[i].is_tex;
        cp->is_hidden_symbol = fn->params[i].is_hidden_symbol;
        if (getenv("ACUDA_DEBUG_SYMBOLS") && cp->is_hidden_symbol)
            fprintf(stderr, "ACuda: cparam build hidden %s kind=%d\n", cp->name, cp->kind);
        if (cp->kind == 1) {
            cp->binding = ctx->nbindings++;
            cp->push_word = -1;
            cp->push_words = 0;
        } else if (cp->kind == 2) {
            cp->binding = -1;
            if (ctx->npush % 2)
                ctx->npush++; // even-align 64-bit scalars
            cp->push_word = ctx->npush;
            cp->push_words = 2;
            ctx->npush += 2;
        } else if (cp->kind == 4) {
            // By-value struct of scalar words. Small structs use two
            // push words per 8-byte word; large structs are bound as a
            // storage buffer and loaded by byte offset instead.
            int words = fn->params[i].wide_words;
            if (words < 2)
                words = 2;
            cp->push_words = 2 * words;
            if ((uint32_t)words > ACUDA_BYVAL_SSBO_WORDS) {
                cp->binding = ctx->nbindings++;
                cp->push_word = -1;
            } else {
                cp->binding = -1;
                if (ctx->npush % 2)
                    ctx->npush++;
                cp->push_word = ctx->npush;
                ctx->npush += 2 * words;
            }
        } else if (cp->kind == 6) {
            // Mixed 32-bit blob: word0's target gets a plain storage-buffer
            // binding (like a kind-1 pointer); word1 travels as one push
            // word.
            cp->binding = ctx->nbindings++;
            cp->push_word = ctx->npush++;
            cp->push_words = 1;
        } else {
            cp->binding = -1;
            cp->push_word = ctx->npush++;
            cp->push_words = 1;
        }
    }
    // Texture/sampler metadata rides in the push-constant block after
    // all scalar params: 6 words per texture binding, in binding order.
    for (int b = 0; b < ctx->nbindings; b++) {
        cparam_t *cp = NULL;
        for (int q = 0; q < ctx->ncparams; q++)
            if (ctx->cparams[q].binding == b) {
                cp = &ctx->cparams[q];
                break;
            }
        if (cp && (cp->is_tex == 1 || cp->is_tex == 2 ||
                   cp->is_tex == 3)) {
            ctx->tex_meta_word[b] = ctx->npush;
            ctx->npush += 7;
        }
    }

    // PhysicalStorageBuffer mode: every plain pointer parameter gets a
    // 64-bit device address appended to the push-constant block (after
    // the scalar and texture-metadata region, 8-byte aligned). Texture/
    // sampler/surface parameters keep their descriptor bindings, so a
    // kernel can mix both models. Host-call internal state buffers are
    // skipped below and stay bindings; user pointers in those kernels
    // (and matrix-addressing kernels) use BDA too.
    ctx->use_bda = ctx->caps.bda != 0;
    {
        int n_plain = 0;
        for (int b = 0; b < ctx->nbindings; b++) {
            cparam_t *cp = NULL;
            for (int q = 0; q < ctx->ncparams; q++)
                if (ctx->cparams[q].binding == b) {
                    cp = &ctx->cparams[q];
                    break;
                }
            if (cp && cp->kind == 1 && cp->is_tex == 0 &&
                strncmp(cp->name, "__acuda_host_", 13) != 0)
                n_plain++;
        }
        // A kernel can have no direct pointer parameter yet still load a
        // device address from a by-value metadata struct (torch's
        // elementwise offset-calculator functor does exactly this).  Those
        // values are only usable as physical addresses, so keep the BDA
        // fallback available; the address push region below is still only
        // allocated when a plain pointer parameter needs it.
        int has_byval = 0;
        for (int q = 0; q < ctx->ncparams; q++)
            if (ctx->cparams[q].kind == 4)
                has_byval = 1;
        // 64-bit scalar parameters can hold a device address in lifted
        // metadata (the kernel then derives a real address from the
        // value), so they also keep physical-address mode available.
        int has_scalar64 = 0;
        for (size_t pi = 0; pi < fn->nparams; pi++)
            if (param_kinds[pi] == 2)
                has_scalar64 = 1;
        // Keep physical-address mode whenever the kernel has any bindings.
        // Legacy SASS-lifted kernels can derive addresses from a scalar
        // parameter or a by-value struct even when no direct pointer
        // parameter exists; those addresses are still real device
        // addresses, not descriptor tokens.
        if (getenv("ACUDA_DEBUG_SYMBOLS"))
            fprintf(stderr, "ACuda: bda pre fn=%s use_bda=%d n_plain=%d nbind=%d has_byval=%d has_scalar64=%d caps_bda=%d\n",
                    fn->name ? fn->name : "?", ctx->use_bda, n_plain, ctx->nbindings, has_byval, has_scalar64, ctx->caps.bda);
        if (n_plain == 0 && !has_byval && !has_scalar64 &&
            ctx->nbindings == 0)
            ctx->use_bda = 0;
        if (ctx->use_bda) {
            int base = (int)ctx->npush;
            int nb = 0;
            if (n_plain > 0) {
                if (ctx->npush % 2)
                    ctx->npush++; // 8-byte align the address region
                base = (int)ctx->npush;
                for (int b = 0; b < ctx->nbindings; b++) {
                    cparam_t *cp = NULL;
                    for (int q = 0; q < ctx->ncparams; q++)
                        if (ctx->cparams[q].binding == b) {
                            cp = &ctx->cparams[q];
                            break;
                        }
                    if (!cp || cp->kind != 1 || cp->is_tex != 0 ||
                        strncmp(cp->name, "__acuda_host_", 13) == 0)
                        continue;
                    ctx->bda_word[b] = ctx->npush;
                    if (getenv("ACUDA_DEBUG_SYMBOLS"))
                        fprintf(stderr, "ACuda: bda assign bind=%d name=%s word=%u\n",
                                b, cp->name, ctx->bda_word[b]);
                    ctx->npush += 2;
                    if (meta && nb < ACUDA_MAX_PARAMS)
                        meta->bda_param[nb++] = b;
                }
            }
            if (meta)
                meta->uses_bda_access = ctx->use_bda;
            if (meta) {
                meta->bda = nb > 0;
                meta->addr_base_word = base;
                meta->n_bda = nb;
            }
        }
    }

    // Abort early: pointer params need an Int64-capable device at RUNTIME,
    // but u64 ARITHMETIC beyond base+offset is already restricted by the
    // emitter. Scalars must be .u32/.f32 (checked at parse).
    // Relaxed: allow kernels with all params unused (kind 3) or zero params.
    if (ctx->nbindings == 0 && ctx->npush == 0 && fn->nparams > 0) {
        // Check if all params are unused
        int all_unused = 1;
        for (size_t i = 0; i < fn->nparams; i++) {
            if (fn->params[i].is_retval)
                continue;
            if (param_kinds[i] != 3) {
                all_unused = 0;
                break;
            }
        }
        if (!all_unused) {
            fail(ctx, "kernel needs at least one param in this subset");
            goto done;
        }
        // All params unused - allow
    } else if (ctx->nbindings == 0 && ctx->npush == 0 && fn->nparams == 0) {
        // Zero params - allow
    } else if (ctx->nbindings == 0 && ctx->npush == 0) {
        fail(ctx, "kernel needs at least one param in this subset");
        goto done;
    }

    // Register declarations.
    for (size_t i = 0; i < fn->nregs; i++) {
        if (ctx->nregs >= MAX_REGS) {
            fail(ctx, "too many registers");
            goto done;
        }
        reg_t *r = &ctx->regs[ctx->nregs++];
        if (strlen(fn->regs[i].name) >= sizeof(r->name)) {
            fail(ctx, "reg name too long");
            goto done;
        }
        strcpy(r->name, fn->regs[i].name);
        r->decl_ty = fn->regs[i].ty;
        r->cur_ty = r->decl_ty;
        if (ctx->emu_f64 && r->decl_ty == PTY_F64)
            r->decl_ty = PTY_F32; // demoted: floats from here on
        r->binding = -1;
        r->lovar = -1;
        r->shvar = -1;
        r->glvar = -1;
        r->paddr = -1;
    }

    ctx->block_ids = malloc(sizeof(*ctx->block_ids) * (fn->nblocks ? fn->nblocks : 1));
    if (!ctx->block_ids) {
        fail(ctx, "out of memory");
        goto done;
    }
    ctx->nblocks = (int)fn->nblocks;
    for (size_t i = 0; i < fn->nblocks; i++)
        ctx->block_ids[i] = 0;

    if (ctx->host_machine &&
        build_host_homes(ctx, fn, err, errlen) != 0) {
        ctx->failed = 1;
        goto done;
    }

    if (!ctx->host_machine) {
#if defined(_MSC_VER)
        // MSVC SEH catches an access violation from a malformed PTX
        // instead of taking down the host process.  Keep the failing call
        // inside __try and leave the block with a plain return code: a
        // goto out of __try is not valid SEH, and a fault can leave cfg
        // only partially linked (discard it rather than freeing it).
        int cfg_rc = -1;
        __try {
            cfg_rc = analyze_cfg(ctx, fn, &cfg);
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            fprintf(stderr, "SEH in analyze_cfg 0x%08lx\n", GetExceptionCode());
            memset(&cfg, 0, sizeof(cfg));
            ctx->failed = 1;
            cfg_rc = -1;
        }
        if (cfg_rc != 0)
            goto done;
#else
        if (analyze_cfg(ctx, fn, &cfg) != 0)
            goto done;
#endif
    }
    if (!ctx->host_machine && cfg.kind == 8) {
        ctx->cfg_machine = 1;
        // Subgroup operations and barriers inside a state machine need
        // reconvergence: a lane that reaches a shuffle/barrier while its
        // peers are still in an earlier block would read stale peer
        // values, and a barrier executed by a divergent subset does not
        // synchronize the lanes that are still catching up.  Dispatch
        // only the lanes at the subgroup-minimum program counter so
        // divergent paths rejoin before the next block runs.
        //
        // A shared-memory store has the same requirement even without a
        // barrier: warp-synchronous code (notably cub's WarpMergeSort)
        // writes a transposed tile and immediately reads words produced
        // by peer lanes, relying on the hardware issuing a converged
        // warp's STS before the following LDS.  Independent per-lane
        // scheduling lets a fast lane read the tile before its peers
        // store to it, producing a garbled sort.
        int has_barrier = 0;
        int has_shared_store = 0;
        for (size_t bi = 0; bi < fn->nblocks; bi++)
            for (size_t ii = 0; ii < fn->blocks[bi].count; ii++) {
                const ptx_instr_t *ti = &fn->blocks[bi].instrs[ii];
                if (ti->op == POP_BAR || ti->op == POP_BARRED ||
                    ti->op == POP_BARWARP)
                    has_barrier = 1;
                else if (ti->op == POP_ST && ti->space == PSP_SHARED)
                    has_shared_store = 1;
            }
        if (ctx->uses_subid || has_shared_store) {
            // Subgroup ops and warp-synchronous shared-store patterns
            // (notably cuDNN implicit-GEMM tiles and cub's WarpMergeSort)
            // rely on converged progress: peer lanes must execute their
            // stores before a lane's dependent load.  Keep lockstep even
            // when the kernel also has block barriers.
            ctx->uses_state_lockstep = 1;
        } else if (has_barrier) {
            // Pure barrier kernels in divergent CFG state machines can
            // deadlock under subgroup-min PC lockstep when lanes must make
            // independent progress to reach the same barrier instance.
            // Dispatch lanes by their own PC and rely on the barrier for
            // synchronization/visibility.
            ctx->uses_state_lockstep = 0;
        }

        if (build_host_homes(ctx, fn, err, errlen) != 0) {
            ctx->failed = 1;
            goto done;
        }
    }

    surf_scan_dynamic(ctx);
    if (ctx->caps.f16_native) {
        for (size_t bi = 0; bi < fn->nblocks && !ctx->uses_f16_native; bi++)
            for (size_t ii = 0;
                 ii < fn->blocks[bi].count && !ctx->uses_f16_native; ii++) {
                const ptx_instr_t *ti = &fn->blocks[bi].instrs[ii];
                if ((ti->op == POP_FMA && ti->ty == PTY_F16X2 &&
                     !ti->f32x2 && !ti->ftz && !ti->sat && !ti->relu) ||
                    ti->ty == PTY_F16 || ti->ty == PTY_F16X2 ||
                    (ti->op == POP_MMA &&
                     (ti->mma_elem == 0 || ti->mma_elem == 5 ||
                      ti->mma_elem == 6)) ||
                    ti->op == POP_DOT2 ||
                    (ti->op == POP_WMMA_MMA && ti->mma_elem == 0))
                    ctx->uses_f16_native = 1;
            }
    }
    {
        // Native shared f32/f64 atomics need the shared variable typed
        // as that float, which is only possible when the whole module's
        // shared memory is scalar f32 (or f64). Buffer (BDA) and image
        // adds use typed physical/image pointers and need no retyping.
        int shared_bad = 0, f32_layout = 1, f64_layout = 1;
        int saw_f32_shared = 0, saw_f64_shared = 0;
        int saw_f32_global = 0, saw_f64_global = 0;
        int saw_f32_image = 0, saw_f32_minmax = 0;
        int s64_seen[MAX_PARAMS] = {0}, s32_seen[MAX_PARAMS] = {0};
        int s64_atomic[MAX_PARAMS] = {0};
        unsigned mask = ctx->caps.atomic_float;
        int native_f32_minmax =
            (mask & ACUDA_ATOMIC_IMAGE_F32_MINMAX) != 0;
        for (int i = 0; i < ctx->nshvars; i++)
            if (ctx->shvars[i].is_extern)
                shared_bad = 1;
        for (size_t bi = 0; bi < fn->nblocks; bi++)
            for (size_t ii = 0; ii < fn->blocks[bi].count; ii++) {
                const ptx_instr_t *ti = &fn->blocks[bi].instrs[ii];
                switch (ti->op) {
                case POP_LD:
                case POP_ST:
                    if (ti->space == PSP_SHARED) {
                        if (ti->ty == PTY_F32)
                            f64_layout = 0;
                        else if (ti->ty == PTY_F64)
                            f32_layout = 0;
                        else
                            f32_layout = f64_layout = 0;
                    } else if (ti->space == PSP_GENERIC) {
                        // A generic address may alias shared.
                        shared_bad = 1;
                    }
                    break;
                case POP_ATOM:
                    if (ti->space == PSP_SHARED) {
                        if (ti->ty == PTY_F32 && ti->atom_op == 0) {
                            saw_f32_shared = 1;
                            f64_layout = 0;
                        } else if (ti->ty == PTY_F64 && ti->atom_op == 0) {
                            saw_f64_shared = 1;
                            f32_layout = 0;
                        } else {
                            f32_layout = f64_layout = 0;
                        }
                    } else if (ti->space == PSP_GENERIC) {
                        shared_bad = 1;
                    } else if (ti->space == PSP_GLOBAL &&
                               ti->atom_op == 0) {
                        if (ti->ty == PTY_F32)
                            saw_f32_global = 1;
                        else if (ti->ty == PTY_F64)
                            saw_f64_global = 1;
                    }
                    break;
                case POP_SULD:
                case POP_SUST:
                    if (ctx->caps.int64_image && ti->surf_ref[0])
                        for (int q = 0; q < ctx->ncparams; q++)
                            if (ctx->cparams[q].binding >= 0 &&
                                strcmp(ctx->cparams[q].name,
                                       ti->surf_ref) == 0) {
                                int b = ctx->cparams[q].binding;
                                if (b < MAX_PARAMS &&
                                    ti->surf_ty == PTY_U64)
                                    s64_seen[b] = 1;
                                else if (b < MAX_PARAMS)
                                    s32_seen[b] = 1;
                            }
                    break;
                case POP_SURED:
                    if (ti->surf_ty == PTY_F32) {
                        if (ti->surf_redop == 0)
                            saw_f32_image = 1;
                        else if (ti->surf_redop == 1 ||
                                 ti->surf_redop == 2)
                            saw_f32_minmax = 1;
                        else
                            break;
                        for (int q = 0; q < ctx->ncparams; q++)
                            if (ctx->cparams[q].binding >= 0 &&
                                strcmp(ctx->cparams[q].name,
                                       ti->surf_ref) == 0) {
                                int b = ctx->cparams[q].binding;
                                int native_add =
                                    (mask & ACUDA_ATOMIC_IMAGE_F32_ADD) != 0;
                                if ((ti->surf_redop == 0 && native_add) ||
                                    (ti->surf_redop != 0 &&
                                     native_f32_minmax)) {
                                    ctx->surf_f32_atomic[b] = 1;
                                } else if (ti->surf_redop == 0) {
                                    if (ctx->caps.surf_alias_ok)
                                        ctx->surf_f32_cas[b] = 1;
                                    else {
                                        ctx->surf_f32_lock[b] = 1;
                                        ctx->surf_f32_atomic[b] = 1;
                                        ctx->surf_lock = 1;
                                    }
                                } else if (ctx->caps.surf_alias_ok) {
                                    ctx->surf_f32_cas[b] = 1;
                                } else {
                                    ctx->surf_f32_lock[b] = 1;
                                    ctx->surf_f32_atomic[b] = 1;
                                    ctx->surf_lock = 1;
                                }
                                if (b < MAX_PARAMS)
                                    s32_seen[b] = 1;
                            }
                    } else if (ti->surf_ty == PTY_U64 &&
                               ctx->caps.int64_image) {
                        for (int q = 0; q < ctx->ncparams; q++)
                            if (ctx->cparams[q].binding >= 0 &&
                                strcmp(ctx->cparams[q].name,
                                       ti->surf_ref) == 0) {
                                int b = ctx->cparams[q].binding;
                                if (b < MAX_PARAMS) {
                                    s64_seen[b] = 1;
                                    s64_atomic[b] = 1;
                                }
                            }
                    } else if (ctx->caps.int64_image && ti->surf_ref[0]) {
                        for (int q = 0; q < ctx->ncparams; q++)
                            if (ctx->cparams[q].binding >= 0 &&
                                strcmp(ctx->cparams[q].name,
                                       ti->surf_ref) == 0) {
                                int b = ctx->cparams[q].binding;
                                if (b < MAX_PARAMS)
                                    s32_seen[b] = 1;
                            }
                    }
                    break;
                case POP_CPASYNC:
                case POP_CPASYNC_BULK:
                case POP_MBARRIER:
                case POP_RED_ASYNC:
                case POP_LDMATRIX:
                case POP_WMMA_LOAD:
                case POP_WMMA_STORE:
                    shared_bad = 1;
                    break;
                default:
                    break;
                }
            }
        if (ctx->host_machine || ctx->cfg_machine)
            shared_bad = 1;
        ctx->shared_f32_view =
            !shared_bad && f32_layout && saw_f32_shared &&
            (mask & ACUDA_ATOMIC_F32_SHARED) != 0;
        ctx->shared_f64_view =
            !shared_bad && f64_layout && saw_f64_shared &&
            (mask & ACUDA_ATOMIC_F64_SHARED) != 0;
        ctx->uses_atomic_f32_add =
            ctx->shared_f32_view ||
            ((mask & ACUDA_ATOMIC_F32_BUFFER) && saw_f32_global) ||
            ((mask & ACUDA_ATOMIC_IMAGE_F32_ADD) && saw_f32_image);
        ctx->uses_atomic_f64_add =
            ctx->shared_f64_view ||
            ((mask & ACUDA_ATOMIC_F64_BUFFER) && saw_f64_global);
        ctx->uses_atomic_f32_minmax =
            ((mask & ACUDA_ATOMIC_IMAGE_F32_MINMAX) && saw_f32_minmax);
        if (ctx->caps.int64_image)
            for (int b = 0; b < MAX_PARAMS; b++)
                if (s64_seen[b] && !s32_seen[b]) {
                    ctx->surf_u64[b] = 1;
                    ctx->uses_int64_image = 1;
                    if (s64_atomic[b])
                        ctx->uses_int64_image_atomic = 1;
                }
    }
    detect_shared_coopmat(ctx, fn);
    emit_head(ctx);
    prepare_coopmat(ctx, fn);
    if (ctx->npush > 0)
        ctx->push_len = const_u32(ctx, (uint32_t)ctx->npush);
    // Reserve builtin var ids and the entry function id. (v_ntid is a
    // spec-constant composite, assigned inside prelude.)
    ctx->fn_id = nid(ctx);
    ctx->v_tid = nid(ctx);
    ctx->v_ctaid = nid(ctx);
    ctx->v_nctaid = nid(ctx);
    ctx->v_subid = nid(ctx);
    ctx->sp_x = nid(ctx);
    ctx->sp_y = nid(ctx);
    ctx->sp_z = nid(ctx);
    ctx->sp_dyn = nid(ctx);
    emit_prelude(ctx);
    if (ctx->failed)
        goto done;

    // Reserve a SPIR-V label id per PTX block.
    for (size_t i = 0; i < fn->nblocks; i++)
        ctx->block_ids[i] = nid(ctx);

    // Kind-4 flow extras: one merge trampoline per self-loop, one
    // epilogue-copy block per exit to the final block.
    if (getenv("ACUDA_TRACE")) fprintf(stderr, "acuda-cfg kind=%d blocks=%d name=%s\n", cfg.kind, (int)fn->nblocks, fn->name);
    if (cfg.kind == 4) {
        flow_tramp = calloc(fn->nblocks, sizeof(*flow_tramp));
        flow_exit = calloc(fn->nblocks, sizeof(*flow_exit));
        if (!flow_tramp || !flow_exit) {
            free(flow_tramp);
            free(flow_exit);
            fail(ctx, "out of memory");
            goto done;
        }
        for (size_t i = 0; i < fn->nblocks; i++) {
            if (cfg.selfloop[i])
                flow_tramp[i] = nid(ctx);
            if (cfg.fwdtgt[i] == (int)fn->nblocks - 1)
                flow_exit[i] = nid(ctx);
        }
    }

    // Function header.
    put(ctx, &ctx->fn, (5u << 16) | S_Function);
    put(ctx, &ctx->fn, ctx->t_void);
    put(ctx, &ctx->fn, ctx->fn_id);
    put(ctx, &ctx->fn, 0); // FunctionControl None
    put(ctx, &ctx->fn, ctx->t_funty);

    if (ctx->host_machine) {
        emit_host_machine(ctx, fn);
        put(ctx, &ctx->fn, (1u << 16) | S_FunctionEnd);
        // The interface list needs every global created by the prelude
        // and the lazy binding helpers, so the entry point is last.
        emit_entry_ann(ctx, fn);
        goto done;
    }
    if (cfg.kind == 8) {
        emit_cfg_machine(ctx, fn);
        put(ctx, &ctx->fn, (1u << 16) | S_FunctionEnd);
        // The interface list needs every global created by the prelude
        // and the lazy binding helpers, so the entry point is last.
        emit_entry_ann(ctx, fn);
        goto done;
    }

    // Software barriers need initialized per-workgroup counters. Emit a
    // synthetic entry block (all invocations pass through it) before the
    // PTX entry block; the extra barrier there is uniform and neutral.
    if (ctx->uses_swbar || ctx->uses_commons) {
        uint32_t entry = nid(ctx);
        put(ctx, &ctx->fn, (2u << 16) | S_Label);
        put(ctx, &ctx->fn, entry);
        if (ctx->uses_swbar)
            emit_swbar_init(ctx, &ctx->fn);
        if (ctx->uses_commons)
            emit_common_init(ctx, &ctx->fn);
        if (ctx->failed)
            goto done;
        emit_branch(ctx, &ctx->fn, ctx->block_ids[0]);
    }

    reg_t *d7h = NULL, *d7e = NULL, *d7t = NULL;
    if (cfg.kind == 7 && ctx->nregs > 0) {
        size_t rb = sizeof(reg_t) * (size_t)ctx->nregs;
        d7h = malloc(rb);
        d7e = malloc(rb);
        d7t = malloc(rb);
        if (!d7h || !d7e || !d7t) {
            free(d7h);
            free(d7e);
            free(d7t);
            fail(ctx, "out of memory");
            goto done;
        }
    }
    for (size_t bi = 0; bi < fn->nblocks && !ctx->failed; bi++) {
        if (cfg.kind == 4) {
            emit_flow_block(ctx, fn, &cfg, (int)bi, flow_tramp,
                            flow_exit, &flow_cur);
            continue;
        }
        const ptx_block_t *blk = &fn->blocks[bi];
        // Diamond else-jump block is consumed by the branch block.
        if (cfg.kind == 2 && (int)bi == cfg.j + 1)
            continue;
        // Loop bodies are consumed by the loop header emission.
        if (cfg.kind == 3 && !cfg.loop_self && (int)bi > cfg.loop_h &&
            (int)bi <= cfg.loop_M)
            continue;
        if (cfg.kind == 3 && (int)bi == cfg.loop_h) {
            emit_loop(ctx, fn, &cfg);
            continue;
        }
        if (cfg.kind == 9 && (int)bi > cfg.loop_h &&
            (int)bi <= cfg.loop_M)
            continue;
        if (cfg.kind == 9 && (int)bi == cfg.loop_h) {
            emit_nested_loop(ctx, fn, &cfg);
            continue;
        }
        put(ctx, &ctx->fn, (2u << 16) | S_Label);
        put(ctx, &ctx->fn, ctx->block_ids[bi]);
        if (cfg.kind == 7 && (int)bi == cfg.J && d7e && d7t && d7h) {
            for (int ri = 0; ri < (int)fn->nregs; ri++) {
                reg_t *re = &d7e[ri], *rt = &d7t[ri];
                reg_t *rh = &d7h[ri], *rr = &ctx->regs[ri];
                int e_def = re->defined, t_def = rt->defined;
                if (!e_def && !t_def)
                    continue;
                if (e_def != t_def) {
                    fail(ctx, "diamond merges an undefined value");
                    break;
                }
                int e_ptr = re->is_ptr, t_ptr = rt->is_ptr;
                if (e_ptr != t_ptr ||
                    (e_ptr && (re->binding != rt->binding ||
                               re->shvar != rt->shvar ||
                               re->glvar != rt->glvar))) {
                    fail(ctx, "diamond merges different pointer homes");
                    break;
                }
                if (re->cur == rt->cur && e_ptr == 0)
                    continue;
                if (e_ptr && re->off == rt->off)
                    continue;
                uint32_t phi = nid(ctx);
                put(ctx, &ctx->fn, (7u << 16) | S_Phi);
                put(ctx, &ctx->fn, e_ptr ? ctx->t_u64
                                         : ty_of(ctx, rr->decl_ty));
                put(ctx, &ctx->fn, phi);
                put(ctx, &ctx->fn, e_ptr ? re->off : re->cur);
                put(ctx, &ctx->fn, ctx->block_ids[cfg.e]);
                put(ctx, &ctx->fn, e_ptr ? rt->off : rt->cur);
                put(ctx, &ctx->fn, ctx->block_ids[cfg.t]);
                if (e_ptr) {
                    rr->off = phi;
                    rr->cur = 0;
                } else {
                    rr->cur = phi;
                }
                (void)rh;
            }
            if (ctx->failed)
                break;
        }
        int term = block_terminator(blk);
        if ((cfg.kind == 1 && (int)bi == cfg.j) ||
            (cfg.kind == 2 && (int)bi == cfg.j) ||
            (cfg.kind == 7 && (int)bi == cfg.j)) {
            // Selection header. Guard skips its 1 terminator, diamond 2.
            emit_block_body(ctx, &ctx->fn, blk, cfg.kind == 2 ? 2 : 1);
            reg_t *pr = find_reg(ctx, cfg.pred);
            uint32_t cond = pr ? pr->cur : 0;
            if (!cond) {
                fail(ctx, "guard predicate has no value");
                break;
            }
            if (cfg.pred_neg) {
                uint32_t n = nid(ctx);
                put(ctx, &ctx->fn, (4u << 16) | S_LogicalNot);
                put(ctx, &ctx->fn, ctx->t_bool);
                put(ctx, &ctx->fn, n);
                put(ctx, &ctx->fn, cond);
                cond = n;
            }
            uint32_t merge, thenb, elseb, copy = 0;
            if (cfg.kind == 1) {
                // Guard: pred true -> target, else fallthrough body.
                // The copy path is only needed to keep SSA dominance
                // when the body defines a value the target consumes.
                // Duplicating a target that contains a barrier would
                // execute that barrier on two divergent paths, so only
                // copy when the body actually defines something.
                int body_defs = 0;
                for (int q = (int)bi + 1; q < cfg.t && !body_defs; q++) {
                    const ptx_block_t *qb = &fn->blocks[q];
                    for (size_t qi = 0; qi < qb->count; qi++)
                        if (qb->instrs[qi].dst[0]) {
                            body_defs = 1;
                            break;
                        }
                }
                if ((int)bi + 1 < cfg.t &&
                    cfg.t == (int)fn->nblocks - 1 && body_defs) {
                    // The target is the final block and the body may
                    // define values it uses. Merging directly at the
                    // final block would let the true edge bypass those
                    // definitions (SPIR-V dominance). Duplicate the
                    // epilogue on the true edge instead, merging at
                    // the body start.
                    merge = ctx->block_ids[bi + 1];
                    elseb = merge;
                    copy = nid(ctx);
                    thenb = copy;
                } else {
                    merge = ctx->block_ids[cfg.t];
                    thenb = merge;
                    elseb = ((int)bi + 1 < cfg.t)
                                ? ctx->block_ids[bi + 1]
                                : merge;
                }
            } else {
                // Diamond: pred true -> then, else else-block; join merges.
                merge = ctx->block_ids[cfg.J];
                thenb = ctx->block_ids[cfg.t];
                elseb = cfg.e == cfg.J ? merge : ctx->block_ids[cfg.e];
            }
            if (cfg.kind == 7 && d7h) {
                // Default both arm snapshots to the branch-header state;
                // an arm that does not redefine a register keeps it.
                memcpy(d7h, ctx->regs, sizeof(reg_t) * (size_t)ctx->nregs);
                memcpy(d7e, ctx->regs, sizeof(reg_t) * (size_t)ctx->nregs);
                memcpy(d7t, ctx->regs, sizeof(reg_t) * (size_t)ctx->nregs);
            }
            if (copy) {
                put(ctx, &ctx->fn, (3u << 16) | S_SelectionMerge);
                put(ctx, &ctx->fn, merge);
                put(ctx, &ctx->fn, 0);
                put(ctx, &ctx->fn, (4u << 16) | S_BranchCond);
                put(ctx, &ctx->fn, cond);
                put(ctx, &ctx->fn, thenb);
                put(ctx, &ctx->fn, elseb);
                put(ctx, &ctx->fn, (2u << 16) | S_Label);
                put(ctx, &ctx->fn, copy);
                size_t rbytes = sizeof(*ctx->regs) * (size_t)ctx->nregs;
                reg_t *rsave = malloc(rbytes ? rbytes : 1);
                if (!rsave) {
                    fail(ctx, "out of memory");
                    break;
                }
                memcpy(rsave, ctx->regs, rbytes);
                emit_block_body(ctx, &ctx->fn, &fn->blocks[cfg.t], 1);
                memcpy(ctx->regs, rsave, rbytes);
                free(rsave);
                if (ctx->failed)
                    break;
                put(ctx, &ctx->fn, (1u << 16) | S_Return);
            } else if (cfg.kind == 1 && elseb == merge) {
                // Degenerate guard: both arms identical -> plain branch.
                emit_branch(ctx, &ctx->fn, merge);
            } else {
                put(ctx, &ctx->fn, (3u << 16) | S_SelectionMerge);
                put(ctx, &ctx->fn, merge);
                put(ctx, &ctx->fn, 0);
                put(ctx, &ctx->fn, (4u << 16) | S_BranchCond);
                put(ctx, &ctx->fn, cond);
                put(ctx, &ctx->fn, thenb);
                put(ctx, &ctx->fn, elseb);
            }
            continue;
        }
        if (term == POP_RET || term == POP_EXIT) {
            emit_block_body(ctx, &ctx->fn, blk, 1);
            put(ctx, &ctx->fn, (1u << 16) | S_Return);
            continue;
        }
        if (term == POP_BRA) {
            // Only branch-to-next survives analysis, except the diamond
            // then-block which jumps to the join.
            emit_block_body(ctx, &ctx->fn, blk, 1);
            if (cfg.kind == 2 && (int)bi == cfg.t) {
                emit_branch(ctx, &ctx->fn, ctx->block_ids[cfg.J]);
                continue;
            }
            if (cfg.kind == 7 && (int)bi == cfg.e) {
                if (d7e)
                    memcpy(d7e, ctx->regs,
                           sizeof(reg_t) * (size_t)ctx->nregs);
                emit_branch(ctx, &ctx->fn, ctx->block_ids[cfg.J]);
                continue;
            }
            if (cfg.kind == 7 && (int)bi == cfg.t) {
                if (d7t)
                    memcpy(d7t, ctx->regs,
                           sizeof(reg_t) * (size_t)ctx->nregs);
                emit_branch(ctx, &ctx->fn, ctx->block_ids[cfg.J]);
                continue;
            }
            if (bi + 1 < fn->nblocks)
                emit_branch(ctx, &ctx->fn, ctx->block_ids[bi + 1]);
            else {
                fail(ctx, "dangling branch at end");
                break;
            }
            continue;
        }
        if (term == POP_CBRA) {
            fail(ctx, "unexpected conditional branch (analysis bug)");
            break;
        }
        emit_block_body(ctx, &ctx->fn, blk, 0);
        if (cfg.kind == 7 && (int)bi == cfg.t && d7t)
            memcpy(d7t, ctx->regs, sizeof(reg_t) * (size_t)ctx->nregs);
        if (bi + 1 < fn->nblocks)
            emit_branch(ctx, &ctx->fn, ctx->block_ids[bi + 1]);
        else {
            fail(ctx, "kernel falls off the end (missing ret)");
            break;
        }
    }
    put(ctx, &ctx->fn, (1u << 16) | S_FunctionEnd);
    // The interface list needs every global created by the prelude and
    // the lazy binding helpers, so the entry point is emitted last.
    emit_entry_ann(ctx, fn);

    if (getenv("ACUDA_DEBUG_EMIT")) {
        fprintf(stderr, "emit sections head=%zu entry=%zu ann=%zu types=%zu "
                        "fn=%zu next=%u\n",
                ctx->head.count, ctx->entry.count, ctx->ann.count,
                ctx->types.count, ctx->fn.count, ctx->next_id);
        (void)0;
    }

done:;
    if (meta) {
        meta->surf_table = ctx->surf_table;
        meta->surf_table_count = ctx->surf_table_count;
        for (int st = 0; st < ACUDA_SURF_TABLE_MAX_TYPES; st++) {
            meta->surf_dim[st] = ctx->surf_tables[st].dim;
            meta->surf_arrayed[st] = ctx->surf_tables[st].arrayed;
            meta->surf_int[st] = ctx->surf_tables[st].isint;
            meta->surf_binding[st] = ctx->surf_tables[st].binding;
            meta->surf_table_cas[st] = ctx->surf_tables[st].cas;
            meta->surf_table_lock[st] = ctx->surf_tables[st].lock;
        }
        for (int b = 0; b < ACUDA_MAX_PARAMS; b++) {
            meta->surf_f32_cas[b] = ctx->surf_f32_cas[b];
            meta->surf_f32_lock[b] = ctx->surf_f32_lock[b];
        }
        meta->surf_lock = ctx->surf_lock;
        {
            // bar.warp.sync needs the 32-lane guarantee but does not
            // read SubgroupLocalInvocationId, so it is not part of
            // uses_subid (which also declares that built-in).
            int warpbar = 0;
            for (size_t bi = 0; bi < fn->nblocks && !warpbar; bi++)
                for (size_t ii = 0;
                     ii < fn->blocks[bi].count && !warpbar; ii++)
                    warpbar = fn->blocks[bi].instrs[ii].op == POP_BARWARP;
            meta->uses_barrier = ctx->uses_barrier;
            // The generic CFG state machine dispatches with
            // OpGroupNonUniformUMin (subgroup scope), so it needs exact
            // warp semantics too.
            meta->uses_subgroup = ctx->uses_subid || ctx->uses_ballot ||
                                  ctx->uses_shfl || ctx->uses_redux ||
                                  warpbar || ctx->cfg_machine;
            meta->requires_float64 = (ctx->uses_f64 && !ctx->emu_f64) ? 2 : 1;
        }
    }
    int ok = !ctx->failed;
    if (ok) {
        // Assemble final module: header + head + entry + ann + types + fn.
        size_t total = 5 + ctx->head.count + ctx->entry.count +
                       ctx->ann.count + ctx->types.count + ctx->fn.count;
        blob->words = malloc(sizeof(uint32_t) * total);
        if (!blob->words) {
            snprintf(err, errlen, "out of memory");
            ok = 0;
        } else {
            blob->words[0] = 0x07230203u;
            blob->words[1] = 0x00010600u; // SPIR-V 1.6 (Vulkan 1.3)
            blob->words[2] = ('A' << 24) | ('C' << 16) | ('U' << 8) | 'A';
            blob->words[3] = ctx->next_id; // bound
            blob->words[4] = 0;
            size_t off = 5;
            memcpy(blob->words + off, ctx->head.words,
                   sizeof(uint32_t) * ctx->head.count);
            off += ctx->head.count;
            memcpy(blob->words + off, ctx->entry.words,
                   sizeof(uint32_t) * ctx->entry.count);
            off += ctx->entry.count;
            memcpy(blob->words + off, ctx->ann.words,
                   sizeof(uint32_t) * ctx->ann.count);
            off += ctx->ann.count;
            memcpy(blob->words + off, ctx->types.words,
                   sizeof(uint32_t) * ctx->types.count);
            off += ctx->types.count;
            memcpy(blob->words + off, ctx->fn.words,
                   sizeof(uint32_t) * ctx->fn.count);
            blob->count = total;
            blob->cap = blob->count;
        }
    }
    free(ctx->head.words);
    free(ctx->entry.words);
    free(ctx->ann.words);
    free(ctx->types.words);
    free(ctx->fn.words);
    free(ctx->block_ids);
    free(flow_tramp);
    free(flow_exit);
    cfg_free(&cfg);
    if (!ok) {
        spv_blob_free(blob);
        free(ctx);
        return -1;
    }
    // Structural self-check: walk the module by instruction word counts.
    // Catches truncation/overrun bugs at compile time instead of in driver.
    {
        size_t pos = 5;
        while (pos < blob->count) {
            uint32_t wc = blob->words[pos] >> 16;
            if (wc == 0 || pos + wc > blob->count) {
                snprintf(err, errlen, "emitter bug: bad word count at %zu", pos);
                if (getenv("ACUDA_DUMP_SPV")) {
                    FILE *f = fopen("/tmp/acuda_debug.spv", "wb");
                    if (f) {
                        fwrite(blob->words, 4, blob->count, f);
                        fclose(f);
                    }
                }
                spv_blob_free(blob);
                free(ctx);
                return -1;
            }
            pos += wc;
        }
        if (pos != blob->count) {
            snprintf(err, errlen, "emitter bug: trailing words");
            spv_blob_free(blob);
            free(ctx);
            return -1;
        }
    }
    free(ctx);
    return 0;
}

