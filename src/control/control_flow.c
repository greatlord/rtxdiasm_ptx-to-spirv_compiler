#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "spirv_internal.h"

// Emits one single-block self-loop for a kind-4 flow: block H ends with
// a cbra to itself. Same phi discipline as emit_loop's form (b), but
// the merge is a caller-provided trampoline id (a direct merge at
// H+1 would be shared with an enclosing selection's merge whenever a
// forward branch skips over the loop, and shared merges break the
// header-dominates-merge rule). The caller emits the header Label
// first and the trampoline (Label + Branch to H+1) after.
static void emit_self_loop(ctx_t *c, const ptx_func_t *fn, int h,
                            uint32_t pre_label, uint32_t merge_id) {
    int nregs = c->nregs;
    size_t nalloc = (size_t)(nregs ? nregs : 1);
    uint32_t *pre_cur = malloc(sizeof(*pre_cur) * nalloc);
    uint32_t *pre_off = malloc(sizeof(*pre_off) * nalloc);
    char *pre_def = malloc(nalloc);
    char *pre_ptr = malloc(nalloc);
    int *pre_binding = malloc(sizeof(*pre_binding) * nalloc);
    int *pre_shvar = malloc(sizeof(*pre_shvar) * nalloc);
    int *pre_glvar = malloc(sizeof(*pre_glvar) * nalloc);
    int *pre_paddr = malloc(sizeof(*pre_paddr) * nalloc);
    int *pre_lovar = malloc(sizeof(*pre_lovar) * nalloc);
    char *def_in = calloc(nalloc, 1);
    size_t *patch_at = malloc(sizeof(*patch_at) * nalloc);
    int *is_carried = malloc(sizeof(*is_carried) * nalloc);
    if (!pre_cur || !pre_off || !pre_def || !pre_ptr || !pre_binding ||
        !pre_shvar || !pre_glvar || !pre_paddr || !pre_lovar || !def_in || !patch_at ||
        !is_carried) {
        free(pre_cur); free(pre_off); free(pre_def); free(pre_ptr);
        free(pre_binding); free(pre_shvar); free(pre_glvar); free(pre_paddr); free(pre_lovar);
        free(def_in); free(patch_at); free(is_carried);
        fail(c, "out of memory");
        return;
    }
    for (int i = 0; i < nregs; i++) {
        pre_def[i] = c->regs[i].defined ? 1 : 0;
        pre_cur[i] = c->regs[i].cur;
        pre_off[i] = c->regs[i].off;
        pre_ptr[i] = c->regs[i].is_ptr ? 1 : 0;
        pre_binding[i] = c->regs[i].binding;
        pre_shvar[i] = c->regs[i].shvar;
        pre_glvar[i] = c->regs[i].glvar;
        pre_paddr[i] = c->regs[i].paddr;
        pre_lovar[i] = c->regs[i].lovar;
    }
    const ptx_block_t *hblk = &fn->blocks[h];
    size_t body_end = hblk->count > 0 ? hblk->count - 1 : 0;
    for (size_t k = 0; k < body_end; k++) {
        const ptx_instr_t *in = &hblk->instrs[k];
        if (!in->dst[0])
            continue;
        reg_t *r = find_reg(c, in->dst);
        if (!r)
            continue; // checked at emission; keep scan lenient
        int idx = (int)(r - c->regs);
        if (idx < 0 || idx >= nregs)
            continue;
        def_in[idx] = 1;
        if (pre_ptr[idx] &&
            ((in->op == POP_MOV || in->op == POP_ADD) && in->ty == PTY_U64)) {
            failf(c, in->line, "loop-carried pointer unsupported "
                               "(recompute addresses from carried indices)");
            goto out;
        }
    }
    uint32_t cont = nid(c);
    // Header Label was emitted by the caller; phis come first.
    for (int i = 0; i < nregs; i++) {
        is_carried[i] = 0;
        patch_at[i] = 0;
        if (!def_in[i] || !pre_def[i])
            continue;
        is_carried[i] = 1;
        int isptr = pre_ptr[i] ? 1 : 0;
        uint32_t phi = nid(c);
        put(c, &c->fn, (7u << 16) | S_Phi);
        put(c, &c->fn, isptr ? c->t_u64 : ty_of(c, c->regs[i].decl_ty));
        put(c, &c->fn, phi);
        put(c, &c->fn, isptr ? pre_off[i] : pre_cur[i]);
        put(c, &c->fn, pre_label);
        patch_at[i] = c->fn.count;
        put(c, &c->fn, 0); // patched to latch value below
        put(c, &c->fn, cont);
        if (isptr) {
            c->regs[i].defined = 1;
            c->regs[i].is_ptr = 1;
            c->regs[i].binding = pre_binding[i];
            c->regs[i].shvar = pre_shvar[i];
            c->regs[i].glvar = pre_glvar[i];
            c->regs[i].paddr = pre_paddr[i];
            c->regs[i].lovar = pre_lovar[i];
            c->regs[i].off = phi;
            c->regs[i].cur = 0;
            c->regs[i].cur_ty = PTY_U64;
        } else {
            c->regs[i].cur = phi;
        }
        if (c->failed)
            goto out;
    }
    emit_range(c, &c->fn, hblk, 0, body_end);
    if (c->failed)
        goto out;
    {
        // The back-edge predicate is defined by the just-emitted body.
        const ptx_instr_t *cb = &hblk->instrs[hblk->count - 1];
        reg_t *pr = find_reg(c, cb->pred);
        uint32_t cond = pr ? pr->cur : 0;
        if (!cond) {
            fail(c, "branch predicate has no value");
            goto out;
        }
        if (cb->pred_neg) {
            uint32_t n = nid(c);
            put(c, &c->fn, (4u << 16) | S_LogicalNot);
            put(c, &c->fn, c->t_bool);
            put(c, &c->fn, n);
            put(c, &c->fn, cond);
            cond = n;
        }
        put(c, &c->fn, (4u << 16) | S_LoopMerge);
        put(c, &c->fn, merge_id);
        put(c, &c->fn, cont);
        put(c, &c->fn, 0);
        put(c, &c->fn, (4u << 16) | S_BranchCond);
        put(c, &c->fn, cond);
        put(c, &c->fn, cont);
        put(c, &c->fn, merge_id);
        put(c, &c->fn, (2u << 16) | S_Label);
        put(c, &c->fn, cont);
        emit_branch(c, &c->fn, c->block_ids[h]);
    }
    for (int i = 0; i < nregs; i++) {
        if (!is_carried[i])
            continue;
        reg_t *r = &c->regs[i];
        if (!r->defined) {
            fail(c, "loop-carried register lost its value");
            goto out;
        }
        c->fn.words[patch_at[i]] = pre_ptr[i] ? r->off : r->cur;
    }
out:
    free(pre_cur); free(pre_off); free(pre_def); free(pre_ptr);
    free(pre_binding); free(pre_shvar); free(pre_glvar); free(pre_paddr); free(pre_lovar);
    free(def_in); free(patch_at); free(is_carried);
}

// Records every register's state at the end of flow block bi: the
// value SSA id, or the byte-offset SSA id for pointers.
static void flow_snapshot(ctx_t *c, const cfg_t *cfg, int bi) {
    int lim = c->nregs < cfg->flow_nregs ? c->nregs : cfg->flow_nregs;
    for (int i = 0; i < lim; i++) {
        cfg->flow_snap[bi * cfg->flow_nregs + i].id = c->regs[i].cur;
        cfg->flow_snap[bi * cfg->flow_nregs + i].off = c->regs[i].off;
        cfg->flow_snap[bi * cfg->flow_nregs + i].cur_ty =
            c->regs[i].cur_ty;
        cfg->flow_snap[bi * cfg->flow_nregs + i].defined =
            c->regs[i].defined ? 1 : 0;
        cfg->flow_snap[bi * cfg->flow_nregs + i].is_ptr =
            c->regs[i].is_ptr ? 1 : 0;
        cfg->flow_snap[bi * cfg->flow_nregs + i].binding =
            c->regs[i].binding;
        cfg->flow_snap[bi * cfg->flow_nregs + i].shvar =
            c->regs[i].shvar;
        cfg->flow_snap[bi * cfg->flow_nregs + i].glvar =
            c->regs[i].glvar;
        cfg->flow_snap[bi * cfg->flow_nregs + i].paddr =
            c->regs[i].paddr;
        cfg->flow_snap[bi * cfg->flow_nregs + i].lovar =
            c->regs[i].lovar;
        if (getenv("ACUDA_TRACE") && c->regs[i].binding == -2 && c->regs[i].lovar < 0 && c->regs[i].is_ptr)
            fprintf(stderr, "acuda-local-warn snap b%d r%d(%s) lovar=%d\n", bi, i, c->regs[i].name, c->regs[i].lovar);
    }
}

static uint32_t flow_undef(ctx_t *c, uint32_t ty) {
    // OpUndef is a constant-section instruction. Emitting it into the
    // function blob is only safe at instruction boundaries; join phis
    // need it as an operand, so keep every OpUndef in types/constants
    // (which precede the function and may hold Undef anywhere).
    uint32_t id = nid(c);
    put(c, &c->types, (3u << 16) | 1); // OpUndef
    put(c, &c->types, ty);
    put(c, &c->types, id);
    return id;
}

// Static predecessors of flow block bi: the fallthrough block (unless
// bi-1 is a self-loop, whose edge arrives via its trampoline) plus
// every forward exit targeting bi. All preds have smaller indices,
// so their snapshots already exist at emission time.
static int flow_preds(const ptx_func_t *fn, const cfg_t *cfg, int bi,
                      int *list, int *via_tramp) {
    int n = 0;
    *via_tramp = 0;
    if (bi > 0) {
        if (cfg->selfloop[bi - 1]) {
            *via_tramp = 1;
        } else {
            int pt = block_terminator(&fn->blocks[bi - 1]);
            int fall = (pt < 0 || pt == POP_CBRA);
            if (pt == POP_BRA) {
                // A surviving unconditional branch may target the final
                // early-return block rather than fall through here.
                const ptx_block_t *pb = &fn->blocks[bi - 1];
                const ptx_instr_t *pl =
                    &pb->instrs[pb->count - 1];
                fall = block_index_of(NULL, fn, pl->target) == bi;
            }
            if (fall)
                list[n++] = bi - 1;
        }
    }
    for (int k = 0; k < bi; k++) {
        // Final-target exits use the early-return pattern (their true
        // edge reaches a private epilogue copy, not bi itself).
        if (cfg->fwdtgt[k] == bi && bi != (int)fn->nblocks - 1) {
            if (n >= 256) {
                return -1;
            }
            list[n++] = k;
        }
    }
    return n;
}

// Merges path-dependent values at a join block (2+ static preds).
// Registers live at the join with disagreeing incomings get OpPhi
// nodes (OpUndef for paths that leave them undefined); dead ones are
// dropped so later accidental uses fail honestly; path-dependent
// pointers across different bindings (or pointer/value mixes) are a
// clean error.
static void emit_flow_join(ctx_t *c, const ptx_func_t *fn, const cfg_t *cfg,
                            int j, const uint32_t *tramp,
                            uint32_t fallthrough_pred) {
    int plist[256], via_tramp = 0;
    int npl = flow_preds(fn, cfg, j, plist, &via_tramp);
    if (npl < 0) {
        fail(c, "too many predecessors");
        return;
    }
    int npred = npl + (via_tramp ? 1 : 0);
    if (npred < 1)
        return;
    if (npred == 1 && !via_tramp) {
        // Single forward predecessor: the block physically before this
        // one may not fall through here, so take the value state from
        // the actual predecessor snapshot (dominance).
        int pb = plist[0];
        for (int i = 0; i < c->nregs; i++) {
            const acuda_flow_snap_t *s =
                &cfg->flow_snap[pb * cfg->flow_nregs + i];
            reg_t *r = &c->regs[i];
            r->defined = s->defined;
            r->is_ptr = s->is_ptr;
            r->binding = s->binding;
            r->shvar = s->shvar;
            r->glvar = s->glvar;
            r->paddr = s->paddr;
            r->lovar = s->lovar;
            if (getenv("ACUDA_TRACE") && r->binding == -2 && r->lovar < 0 && r->is_ptr)
                fprintf(stderr, "acuda-local-warn join-restore r%d(%s) lovar=%d from b%d\n", i, r->name, r->lovar, pb);
            r->cur = s->id;
            r->cur_ty = s->cur_ty;
            if (s->is_ptr)
                r->off = s->off;
        }
        return;
    }
    if (npred < 2)
        return;
    int nregs = c->nregs;
    for (int i = 0; i < nregs; i++) {
        int agree = 1;
        int d0 = 0, p0 = 0, b0 = 0, h0 = -1, g0 = -1, a0 = -1, l0 = -1;
        uint32_t id0 = 0, off0 = 0;
        ptx_ty_t ct0 = PTY_U32;
        int first = 1;
        int ninc = 0;
        // Compare all incomings against the first predecessor's.
        for (int q = -1; q < npl; q++) {
            int pb = (q < 0) ? (j - 1) : plist[q]; // q<0 only if via_tramp
            if (q < 0 && !via_tramp)
                continue;
            const acuda_flow_snap_t *s =
                &cfg->flow_snap[pb * cfg->flow_nregs + i];
            if (first) {
                d0 = s->defined;
                p0 = s->is_ptr;
                b0 = s->binding;
                h0 = s->shvar;
                g0 = s->glvar;
                a0 = s->paddr;
                l0 = s->lovar;
                id0 = s->id;
                off0 = s->off;
                ct0 = s->cur_ty;
                first = 0;
            } else if (d0 != s->defined || p0 != s->is_ptr ||
                       b0 != s->binding || h0 != s->shvar ||
                       g0 != s->glvar || a0 != s->paddr || l0 != s->lovar ||
                       id0 != s->id || ct0 != s->cur_ty ||
                       (p0 && off0 != s->off)) {
                agree = 0;
            }
            ninc++;
        }
        if (agree) {
            reg_t *r = &c->regs[i];
            r->defined = d0;
            r->is_ptr = p0;
            r->binding = b0;
            r->shvar = h0;
            r->glvar = g0;
            r->paddr = a0;
            r->lovar = l0;
            r->cur = id0;
            r->cur_ty = ct0;
            if (p0)
                r->off = off0;
            if (getenv("ACUDA_TRACE") && r->binding == -2 && r->lovar < 0 && r->is_ptr)
                fprintf(stderr, "acuda-local-warn agree r%d(%s) lovar=%d b%d\n", i, r->name, r->lovar, j);
            // All-undefined but live: give it an honest undef value.
            if (!d0 && cfg->flow_live[j * cfg->flow_nregs + i]) {
                r->defined = 1;
                r->is_ptr = 0;
                r->binding = -1;
                r->shvar = -1;
                r->glvar = -1;
                r->paddr = -1;
                r->lovar = -1;
                r->cur = flow_undef(c, ty_of(c, r->decl_ty));
                r->cur_ty = r->decl_ty;
            }
            continue;
        }
        if (!cfg->flow_live[j * cfg->flow_nregs + i]) {
            c->regs[i].defined = 0;
            continue;
        }
        // Need a phi: all DEFINED sides must be values, or pointers
        // with one shared (binding, Workgroup var, .global image, .local) home.
        int any_val = 0, any_ptr = 0, pbind = -1, pshvar = -2, pglvar = -2;
        int ppaddr = -2, plovar = -2;
        for (int q = -1; q < npl; q++) {
            if (q < 0 && !via_tramp)
                continue;
            int pb = (q < 0) ? (j - 1) : plist[q];
            const acuda_flow_snap_t *s =
                &cfg->flow_snap[pb * cfg->flow_nregs + i];
            if (!s->defined)
                continue;
            if (s->is_ptr) {
                any_ptr = 1;
                if (pbind < 0 && pshvar == -2 && pglvar == -2 &&
                    ppaddr == -2 && plovar == -2) {
                    pbind = s->binding;
                    pshvar = s->shvar;
                    pglvar = s->glvar;
                    ppaddr = s->paddr;
                    plovar = s->lovar;
                } else if (pbind != s->binding || pshvar != s->shvar ||
                           pglvar != s->glvar || ppaddr != s->paddr ||
                           plovar != s->lovar) {
                    pbind = -2;
                }
            } else {
                any_val = 1;
            }
        }
        if ((any_val && any_ptr) || pbind == -2) {
            fail(c, "path-dependent pointer at join (recompute addresses from carried indices)");
            return;
        }
        uint32_t ty =
            any_ptr ? c->t_u64 : ty_of(c, c->regs[i].decl_ty);
        uint32_t phi = nid(c);
        put(c, &c->fn,
            (uint32_t)((3 + 2 * (npl + (via_tramp ? 1 : 0))) << 16) | S_Phi);
        put(c, &c->fn, ty);
        put(c, &c->fn, phi);
        for (int q = -1; q < npl; q++) {
            int pb;
            uint32_t edge;
            if (q < 0) {
                if (!via_tramp)
                    continue;
                pb = j - 1;
                edge = tramp[pb];
            } else {
                pb = plist[q];
                edge = c->block_ids[pb];
                // The fallthrough predecessor (j-1) branches from its
                // flow-emission edge label (block entry or selection
                // fallthrough), not necessarily the last emitted label
                // (e.g. an early-return epilogue copy ends at a private
                // exit block while the false arm comes from the entry).
                if (pb == j - 1 && fallthrough_pred != 0)
                    edge = fallthrough_pred;
            }
            const acuda_flow_snap_t *s =
                &cfg->flow_snap[pb * cfg->flow_nregs + i];
            uint32_t vid;
            if (s->defined) {
                /* Pointer-carried registers phi their byte offset (off),
                 * not cur/id (which is 0 for def_ptr/32-bit cvta copies). */
                vid = any_ptr ? s->off : s->id;
            } else {
                vid = flow_undef(c, any_ptr ? c->t_u64
                                            : ty_of(c, c->regs[i].decl_ty));
                if (c->failed)
                    return;
            }
            put(c, &c->fn, vid);
            put(c, &c->fn, edge);
        }
        reg_t *r = &c->regs[i];
        r->defined = 1;
        if (any_ptr) {
            r->is_ptr = 1;
            r->binding = pbind;
            r->shvar = pshvar;
            r->glvar = pglvar;
            r->paddr = ppaddr == -2 ? -1 : ppaddr;
            r->lovar = plovar == -2 ? -1 : plovar;
            r->cur = 0;
            r->cur_ty = PTY_U64;
            r->off = phi;
            if (getenv("ACUDA_TRACE") && r->binding == -2 && r->lovar < 0)
                fprintf(stderr, "acuda-local-warn phi-ptr r%d(%s) lovar=%d b%d\n", i, r->name, r->lovar, j);
        } else {
            r->is_ptr = 0;
            r->binding = -1;
            r->shvar = -1;
            r->glvar = -1;
            r->paddr = -1;
            r->lovar = -1;
            r->cur = phi;
            r->cur_ty = r->decl_ty;
        }
        if (c->failed)
            return;
        (void)ninc;
    }
}
// A branch to the next block may land on the merge of an enclosing
// selection.  Exiting inner selections without going through their own
// merges is invalid structured SPIR-V, so duplicate the target block on
// this edge instead (it must end in ret or a final-block branch).
// Returns 1 when handled (control is fully emitted), 0 when the target
// is not an enclosing merge, -1 on a hard error.
static int flow_inline_merge_target(ctx_t *c, const ptx_func_t *fn,
                                    const cfg_t *cfg, int bi, int btgt,
                                    uint32_t *flow_cur) {
    int nblocks = (int)fn->nblocks;
    int enclosing = 0;
    for (int k = 0; k < bi; k++)
        if (cfg->fwdtgt[k] == btgt) {
            enclosing = 1;
            break;
        }
    if (!enclosing)
        return 0;
    const ptx_block_t *tb = &fn->blocks[btgt];
    int tterm = block_terminator(tb);
    flow_snapshot(c, cfg, bi);
    emit_block_body(c, &c->fn, tb, 1);
    if (c->failed)
        return -1;
    if (tterm < 0) {
        // The target's branch-to-next was dropped as a no-op; continue
        // on this edge to the block it falls through to.
        if (btgt + 1 != nblocks - 1) {
            fail(c, "unconditional branch to an enclosing merge target "
                    "has unsupported fallthrough");
            return -1;
        }
        size_t rbytes = sizeof(*c->regs) * (size_t)c->nregs;
        reg_t *rsave = malloc(rbytes ? rbytes : 1);
        if (!rsave) {
            fail(c, "out of memory");
            return -1;
        }
        memcpy(rsave, c->regs, rbytes);
        flow_snapshot(c, cfg, btgt);
        emit_block_body(c, &c->fn, &fn->blocks[nblocks - 1], 1);
        memcpy(c->regs, rsave, rbytes);
        free(rsave);
        if (c->failed)
            return -1;
        put(c, &c->fn, (1u << 16) | S_Return);
    } else if (tterm == POP_RET || tterm == POP_EXIT) {
        put(c, &c->fn, (1u << 16) | S_Return);
    } else if (tterm == POP_BRA) {
        const ptx_instr_t *tl = &tb->instrs[tb->count - 1];
        int tt = block_index_of(c, fn, tl->target);
        if (tt != nblocks - 1) {
            fail(c, "unconditional branch to an enclosing merge target "
                    "has unsupported flow");
            return -1;
        }
        size_t rbytes = sizeof(*c->regs) * (size_t)c->nregs;
        reg_t *rsave = malloc(rbytes ? rbytes : 1);
        if (!rsave) {
            fail(c, "out of memory");
            return -1;
        }
        memcpy(rsave, c->regs, rbytes);
        flow_snapshot(c, cfg, btgt);
        emit_block_body(c, &c->fn, &fn->blocks[nblocks - 1], 1);
        memcpy(c->regs, rsave, rbytes);
        free(rsave);
        if (c->failed)
            return -1;
        put(c, &c->fn, (1u << 16) | S_Return);
    } else if (tterm == POP_CBRA) {
        // The target is a merge block that itself starts a selection.
        // Falling through to its label is the normal merge edge.
        return 0;
    } else {
        fail(c, "unconditional branch to an enclosing merge target has "
                "unsupported terminator");
        return -1;
    }
    *flow_cur = c->block_ids[bi];
    return 1;
}

// Emits one block of a kind-4 flow in linear order. flow_cur tracks
// the SPIR-V label control falls through from (a block id, or a loop
// trampoline id): it is the phi preheader when the next block is a
// self-loop. Forward exits to a middle block become nested selections
// merging at the target; exits to the final ret block duplicate the
// epilogue (early-return pattern) so no two selections share a merge.
void emit_flow_block(ctx_t *c, const ptx_func_t *fn, const cfg_t *cfg,
                             int bi, const uint32_t *tramp,
                             const uint32_t *exitcopy, uint32_t *flow_cur) {
    int nblocks = (int)fn->nblocks;
    const ptx_block_t *blk = &fn->blocks[bi];
    uint32_t fallthrough_pred = *flow_cur;
    put(c, &c->fn, (2u << 16) | S_Label);
    put(c, &c->fn, c->block_ids[bi]);
    emit_flow_join(c, fn, cfg, bi, tramp, fallthrough_pred);
    if (c->failed)
        return;
    if (getenv("ACUDA_TRACE")) {
        for (int _di = 0; _di < c->nregs; _di++) {
            reg_t *_dr = &c->regs[_di];
            if (_dr->binding == -2 && _dr->lovar < 0 && _dr->is_ptr)
                fprintf(stderr, "acuda-local-warn post-join b%d r%d(%s) lovar=%d\n", bi, _di, _dr->name, _dr->lovar);
        }
    }
    if (cfg->selfloop[bi]) {
        if (fallthrough_pred == 0) {
            fail(c, "loop has no preheader block");
            return;
        }
        // The preheader is the block that actually branches here from
        // the preceding flow block (or its trampoline for a prior loop).
        emit_self_loop(c, fn, bi, fallthrough_pred, tramp[bi]);
        if (c->failed)
            return;
        put(c, &c->fn, (2u << 16) | S_Label);
        put(c, &c->fn, tramp[bi]);
        emit_branch(c, &c->fn, c->block_ids[bi + 1]);
        *flow_cur = tramp[bi];
        flow_snapshot(c, cfg, bi);
        return;
    }
    int term = block_terminator(blk);
    if (term == POP_RET || term == POP_EXIT) {
        emit_block_body(c, &c->fn, blk, 1);
        put(c, &c->fn, (1u << 16) | S_Return);
        return;
    }
    if (term == POP_CBRA) {
        int t = cfg->fwdtgt[bi];
        emit_block_body(c, &c->fn, blk, 1);
        if (c->failed)
            return;
        const ptx_instr_t *cb = &blk->instrs[blk->count - 1];
        reg_t *pr = find_reg(c, cb->pred);
        uint32_t cond = pr ? pr->cur : 0;
        if (!cond) {
            fail(c, "branch predicate has no value");
            return;
        }
        if (cb->pred_neg) {
            uint32_t n = nid(c);
            put(c, &c->fn, (4u << 16) | S_LogicalNot);
            put(c, &c->fn, c->t_bool);
            put(c, &c->fn, n);
            put(c, &c->fn, cond);
            cond = n;
        }
        // Snapshot the skip-source state BEFORE emitting the copy:
        // the copy re-emits the epilogue with these values and must
        // not pollute the fallthrough state (saved/restored below).
        flow_snapshot(c, cfg, bi);
        if (t == nblocks - 1) {
            // Early return: duplicate the final block's body so this
            // selection merges at the fallthrough block.
            put(c, &c->fn, (3u << 16) | S_SelectionMerge);
            put(c, &c->fn, c->block_ids[bi + 1]);
            put(c, &c->fn, 0);
            put(c, &c->fn, (4u << 16) | S_BranchCond);
            put(c, &c->fn, cond);
            put(c, &c->fn, exitcopy[bi]);
            put(c, &c->fn, c->block_ids[bi + 1]);
            put(c, &c->fn, (2u << 16) | S_Label);
            put(c, &c->fn, exitcopy[bi]);
            size_t rbytes = sizeof(*c->regs) * (size_t)c->nregs;
            reg_t *rsave = malloc(rbytes ? rbytes : 1);
            if (!rsave) {
                fail(c, "out of memory");
                return;
            }
            memcpy(rsave, c->regs, rbytes);
            emit_block_body(c, &c->fn, &fn->blocks[nblocks - 1], 1);
            memcpy(c->regs, rsave, rbytes);
            free(rsave);
            if (c->failed)
                return;
            put(c, &c->fn, (1u << 16) | S_Return);
        } else if (t == bi + 1) {
            emit_branch(c, &c->fn, c->block_ids[bi + 1]);
        } else {
            put(c, &c->fn, (3u << 16) | S_SelectionMerge);
            put(c, &c->fn, c->block_ids[t]);
            put(c, &c->fn, 0);
            put(c, &c->fn, (4u << 16) | S_BranchCond);
            put(c, &c->fn, cond);
            put(c, &c->fn, c->block_ids[t]);
            put(c, &c->fn, c->block_ids[bi + 1]);
        }
        *flow_cur = c->block_ids[bi];
        return;
    }
    if (term == POP_BRA) {
        // Only branch-to-next and early returns survive flow analysis.
        emit_block_body(c, &c->fn, blk, 1);
        if (c->failed)
            return;
        const ptx_instr_t *blast = &blk->instrs[blk->count - 1];
        int btgt = block_index_of(c, fn, blast->target);
        if (btgt == nblocks - 1) {
            // Unconditional early return: emit the final epilogue inline.
            // No selection/merge block is involved, so this cannot
            // collide with another selection's merge block.
            flow_snapshot(c, cfg, bi);
            size_t rbytes = sizeof(*c->regs) * (size_t)c->nregs;
            reg_t *rsave = malloc(rbytes ? rbytes : 1);
            if (!rsave) {
                fail(c, "out of memory");
                return;
            }
            memcpy(rsave, c->regs, rbytes);
            emit_block_body(c, &c->fn, &fn->blocks[nblocks - 1], 1);
            memcpy(c->regs, rsave, rbytes);
            free(rsave);
            if (c->failed)
                return;
            put(c, &c->fn, (1u << 16) | S_Return);
            *flow_cur = c->block_ids[bi];
            return;
        }
        if (btgt != bi + 1) {
            fail(c, "unconditional forward skip unsupported (dead code?)");
            return;
        }
        if (flow_inline_merge_target(c, fn, cfg, bi, btgt, flow_cur) != 0)
            return;
        if (bi + 1 < nblocks)
            emit_branch(c, &c->fn, c->block_ids[bi + 1]);
        else {
            fail(c, "dangling branch at end");
            return;
        }
        *flow_cur = c->block_ids[bi];
        flow_snapshot(c, cfg, bi);
        return;
    }
    emit_block_body(c, &c->fn, blk, 0);
    if (c->failed)
        return;
    if (bi + 1 < nblocks) {
        if (flow_inline_merge_target(c, fn, cfg, bi, bi + 1, flow_cur) != 0)
            return;
        emit_branch(c, &c->fn, c->block_ids[bi + 1]);
    } else {
        fail(c, "kernel falls off the end (missing ret)");
        return;
    }
    *flow_cur = c->block_ids[bi];
    flow_snapshot(c, cfg, bi);
}


