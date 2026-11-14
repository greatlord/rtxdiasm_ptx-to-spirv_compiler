// Control-flow analysis and emission: guards, diamonds, single-header
// loops, and the multi-region flow/phi composer.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "spirv_internal.h"

// ---- Control flow: straight line, single forward guard, a single
// ---- forward if/else diamond (numba shape), single-header loops, or
// ---- a sequential multi-region flow (kind 4: forward exits plus any
// ---- number of single-block self-loops, the real-numba-loop shape). ----




// Multi-region flow validation (kind 4): straight-line blocks,
// forward conditional exits (to a middle block = nested selection, to
// the final ret block = early-return copy) and any number of
// single-block self-loops in sequence. Forward edges may skip over
// loops but never target a loop header (a skipped loop's phis would
// lack an incoming value); backward edges other than self-cbras stay
// on the kind-3 path (or a clean error there).
static int flow_analyze(ctx_t *c, const ptx_func_t *fn, cfg_t *cfg) {
    int nblocks = (int)fn->nblocks;
    cfg->selfloop = calloc((size_t)nblocks, sizeof(int));
    cfg->fwdtgt = malloc(sizeof(int) * (size_t)nblocks);
    if (!cfg->selfloop || !cfg->fwdtgt) {
        cfg_free(cfg);
        fail(c, "out of memory");
        return -1;
    }
    for (int i = 0; i < nblocks; i++)
        cfg->fwdtgt[i] = -1;
    for (int i = 0; i < nblocks; i++) {
        int term = block_terminator(&fn->blocks[i]);
        if (term < 0)
            continue;
        const ptx_instr_t *last =
            &fn->blocks[i].instrs[fn->blocks[i].count - 1];
        if (term == POP_RET || term == POP_EXIT) {
            if (i != nblocks - 1) {
                failf(c, last->line,
                      "return must end the final block in this subset");
                goto fail;
            }
            continue;
        }
        int tgt = block_index_of(c, fn, last->target);
        if (tgt < 0) {
            failf(c, last->line, "branch to unknown label");
            goto fail;
        }
        if (term == POP_CBRA) {
            if (tgt == i) {
                cfg->selfloop[i] = 1;
            } else if (tgt > i) {
                cfg->fwdtgt[i] = tgt;
            } else {
                fail(c, "backward conditional branch must target its own block");
                goto fail;
            }
        } else { // POP_BRA
            if (tgt == i + 1)
                continue; // branch-to-next: harmless fallthrough
            if (tgt < i) {
                fail(c, "backward unconditional branch outside a single-header loop (multi-block loops unsupported)");
                goto fail;
            }
            // Allow unconditional forward jumps in multi-region flow
            if (tgt > i + 1) {
                // Store as forward jump (will be handled specially)
                // For now, treat as needing special handling
            }
        }
    }
    // The structured flow emitter can only inline an enclosing merge
    // target whose body ends in a return, a final-block branch or a
    // straight-line fallthrough. Other shapes (a branching merge
    // target, or a fallthrough into one) are handled by the generic
    // state machine instead of failing mid-emission.
    for (int i = 0; i < nblocks; i++) {
        int enclosing_next = 0;
        if (i + 1 < nblocks)
            for (int k = 0; k < i; k++)
                if (cfg->fwdtgt[k] == i + 1) {
                    enclosing_next = 1;
                    break;
                }
        if (enclosing_next) {
            int tt = block_terminator(&fn->blocks[i + 1]);
            if (tt != POP_RET && tt != POP_EXIT && tt != POP_BRA) {
                fail(c, "fallthrough into a branching merge target");
                goto fail;
            }
        }
        if (block_terminator(&fn->blocks[i]) != POP_BRA)
            continue;
        const ptx_instr_t *bl =
            &fn->blocks[i].instrs[fn->blocks[i].count - 1];
        int tgt = block_index_of(c, fn, bl->target);
        if (tgt < 0 || tgt == i + 1 || tgt == nblocks - 1)
            continue;
        int tt = block_terminator(&fn->blocks[tgt]);
        int ok = 0;
        if (tt == POP_RET || tt == POP_EXIT) {
            ok = 1;
        } else if (tt == POP_BRA) {
            const ptx_instr_t *tl =
                &fn->blocks[tgt].instrs[fn->blocks[tgt].count - 1];
            ok = block_index_of(c, fn, tl->target) == nblocks - 1;
        } else if (tt < 0) {
            ok = tgt + 1 == nblocks - 1;
        }
        if (!ok) {
            failf(c, bl->line, "forward branch target shape unsupported");
            goto fail;
        }
    }
    if (block_terminator(&fn->blocks[nblocks - 1]) != POP_RET &&
        block_terminator(&fn->blocks[nblocks - 1]) != POP_EXIT) {
        fail(c, "kernel must end with ret/exit");
        goto fail;
    }
    for (int h = 0; h < nblocks; h++) {
        if (!cfg->selfloop[h])
            continue;
        if (h + 1 >= nblocks) {
            fail(c, "loop at end of kernel needs a following exit block");
            goto fail;
        }
        // Exactly one static predecessor: the fallthrough block. It
        // must deliver control unconditionally-ish: unterminated, a
        // branch-to-next no-op, or a conditional whose false arm falls
        // through (a guard around/into the loop still enters only via
        // h-1, so the phi preheader stays single).
        int has_pre = 0;
        if (h > 0) {
            int pt = block_terminator(&fn->blocks[h - 1]);
            if (pt < 0 || pt == POP_BRA || pt == POP_CBRA)
                has_pre = 1;
        }
        if (!has_pre) {
            fail(c, "loop header must follow straight-line code");
            goto fail;
        }
        for (int k = 0; k < nblocks; k++) {
            if (k == h - 1 || k == h)
                continue; // fallthrough pred / the loop's own back-edge
            int kt = block_terminator(&fn->blocks[k]);
            if (kt != POP_BRA && kt != POP_CBRA)
                continue;
            const ptx_instr_t *kl =
                &fn->blocks[k].instrs[fn->blocks[k].count - 1];
            // (Unconditional non-next bras are rejected above, so any
            // surviving POP_BRA targets k+1 != h here.)
            if (kt == POP_CBRA && block_index_of(c, fn, kl->target) == h) {
                fail(c, "branch into loop header unsupported (multi-block loops)");
                goto fail;
            }
        }
        // The loop predicate must be a declared .pred register
        // (defined-ness is verified at emission: the setp precedes the
        // back-edge inside the header).
        {
            const ptx_instr_t *cb =
                &fn->blocks[h].instrs[fn->blocks[h].count - 1];
            reg_t *r = find_reg(c, cb->pred);
            if (!r || r->decl_ty != PTY_PRED) {
                fail(c, "loop predicate must be a .pred register");
                goto fail;
            }
        }
    }
    // Liveness (live-in per block) for join phis: a register live at a
    // merge needs a phi when incoming paths disagree. The final
    // block's uses are added to every block (early-return copies may
    // re-emit the epilogue at any exit).
    {
        int nregs = c->nregs;
        cfg->flow_nregs = nregs;
        size_t cells = (size_t)nblocks * (size_t)(nregs ? nregs : 1);
        char *use = calloc(cells, 1);
        char *def = calloc(cells, 1);
        cfg->flow_live = calloc(cells, 1);
        cfg->flow_snap = calloc(cells, sizeof(*cfg->flow_snap));
        if (!use || !def || !cfg->flow_live || !cfg->flow_snap) {
            free(use);
            free(def);
            fail(c, "out of memory");
            goto fail;
        }
        // Final-block uses are live everywhere (epilogue copies).
        for (int b = 0; b < nblocks; b++) {
            const ptx_block_t *blk = &fn->blocks[b];
            for (size_t k = 0; k < blk->count; k++) {
                const ptx_instr_t *in = &blk->instrs[k];
                if (in->pred[0]) {
                    reg_t *r = find_reg(c, in->pred);
                    if (r) {
                        int x = (int)(r - c->regs);
                        if (!def[b * nregs + x])
                            use[b * nregs + x] = 1;
                    }
                }
                for (int s = 0; s < in->nsrc; s++) {
                    if (in->src[s].kind != OPD_REG)
                        continue;
                    reg_t *r = find_reg(c, in->src[s].reg);
                    if (!r)
                        continue; // use-before-declare: emission errors
                    int x = (int)(r - c->regs);
                    if (x < 0 || x >= nregs)
                        continue;
                    if (!def[b * nregs + x])
                        use[b * nregs + x] = 1;
                }
                if (in->dst[0]) {
                    reg_t *r = find_reg(c, in->dst);
                    if (r) {
                        int x = (int)(r - c->regs);
                        if (x >= 0 && x < nregs)
                            def[b * nregs + x] = 1;
                    }
                }
            }
        }
        // Fold the final block's uses into every block (copies).
        for (int b = 0; b < nblocks - 1; b++) {
            for (int i = 0; i < nregs; i++) {
                if (use[(nblocks - 1) * nregs + i] &&
                    !def[b * nregs + i])
                    use[b * nregs + i] = 1;
            }
        }
        // Successors in the flow graph (exit copies add no edges;
        // self-loop exits run through their trampoline to h+1).
        for (int iter = 0; iter < nblocks + 1; iter++) {
            for (int b = nblocks - 1; b >= 0; b--) {
                int su[3], nsu = 0;
                int bt = block_terminator(&fn->blocks[b]);
                if (bt == POP_CBRA) {
                    if (cfg->selfloop[b]) {
                        su[nsu++] = b + 1; // loop exit (via trampoline)
                    } else {
                        if (cfg->fwdtgt[b] >= 0) {
                            su[nsu++] = cfg->fwdtgt[b];
                        }
                        su[nsu++] = b + 1;
                    }
                } else if (bt == POP_BRA) {
                    // Check if it's a forward jump to non-next block
                    const ptx_instr_t *blast = &fn->blocks[b].instrs[fn->blocks[b].count - 1];
                    int btgt = block_index_of(c, fn, blast->target);
                    if (btgt > b + 1) {
                        su[nsu++] = btgt;
                    } else {
                        su[nsu++] = b + 1;
                    }
                } else if (bt == POP_RET || bt == POP_EXIT) {
                    // none
                } else {
                    if (b + 1 < nblocks)
                        su[nsu++] = b + 1;
                }
                for (int i = 0; i < nregs; i++) {
                    if (use[b * nregs + i]) {
                        cfg->flow_live[b * nregs + i] = 1;
                        continue;
                    }
                    if (def[b * nregs + i])
                        continue;
                    for (int q = 0; q < nsu; q++) {
                        if (su[q] >= 0 && su[q] < nblocks &&
                            cfg->flow_live[su[q] * nregs + i]) {
                            cfg->flow_live[b * nregs + i] = 1;
                            break;
                        }
                    }
                }
            }
        }
        free(use);
        free(def);
    }
    cfg->kind = 4;
    return 0;
fail:
    cfg_free(cfg);
    return -1;
}

static int analyze_cfg_structured(ctx_t *c, const ptx_func_t *fn,
                                  cfg_t *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->loop_h = cfg->loop_l = cfg->loop_M = -1;
    int ncbra = 0, j = -1, t = -1;
    int pred_neg = 0;
    char pred[64] = "";
    // All non-noop unconditional forward bras (block, target).
    int ub_blk[8], ub_tgt[8], nub = 0;
    // Backward/self edges (loop candidates, validated below).
    int nback = 0, back_blk[8], back_tgt[8], back_is_cbra[8];
    int nblocks = (int)fn->nblocks;
    for (int i = 0; i < nblocks; i++) {
        int term = block_terminator(&fn->blocks[i]);
        const ptx_instr_t *last =
            term >= 0 ? &fn->blocks[i].instrs[fn->blocks[i].count - 1] : NULL;
        if (term == POP_CBRA) {
            int tgt = block_index_of(c, fn, last->target);
            if (tgt < 0) {
                failf(c, last->line, "branch to unknown label");
                return -1;
            }
            if (tgt <= i) {
                // Backward/self conditional: loop candidate (or error).
                if (nback >= 8) {
                    fail(c, "too many branches");
                    return -1;
                }
                back_blk[nback] = i;
                back_tgt[nback] = tgt;
                back_is_cbra[nback] = 1;
                nback++;
                continue;
            }
            ncbra++;
            j = i;
            t = tgt;
            pred_neg = last->pred_neg;
            if (strlen(last->pred) >= sizeof(pred)) {
                failf(c, last->line, "predicate name too long");
                return -1;
            }
            strcpy(pred, last->pred);
        } else if (term == POP_BRA) {
            int tgt = block_index_of(c, fn, last->target);
            if (tgt < 0) {
                failf(c, last->line, "branch to unknown label");
                return -1;
            }
            if (tgt == i + 1)
                continue; // branch-to-next: harmless fallthrough
            if (tgt <= i) {
                // Backward/self unconditional: loop latch candidate.
                if (nback >= 8) {
                    fail(c, "too many branches");
                    return -1;
                }
                back_blk[nback] = i;
                back_tgt[nback] = tgt;
                back_is_cbra[nback] = 0;
                nback++;
                continue;
            }
            if (nub >= 8) {
                fail(c, "too many branches");
                return -1;
            }
            ub_blk[nub] = i;
            ub_tgt[nub] = tgt;
            nub++;
        } else if (term == POP_RET || term == POP_EXIT) {
            if (i != nblocks - 1) {
                failf(c, last->line,
                      "return must end the final block in this subset");
                return -1;
            }
        }
    }
    if (nback > 0) {
        // Multi-region flow (kind 4): every back-edge is a self-cbra
        // (single-block while loop) and there is more than just that
        // one loop (otherwise the kind-3 path below handles it).
        {
            int allself = 1;
            for (int k = 0; k < nback; k++) {
                if (!back_is_cbra[k] || back_tgt[k] != back_blk[k])
                    allself = 0;
            }
            if (allself && !(nback == 1 && ncbra == 0 && nub == 0))
                return flow_analyze(c, fn, cfg);
        }
        // Extended form (a): one header exit test plus body blocks that
        // may `break` (branch to M) or `continue` (branch to the latch L).
        // The body is still acyclic and the latch is the single back edge.
        if (nback == 1 && !back_is_cbra[0]) {
            int L = back_blk[0], H = back_tgt[0];
            int M = -1, ok = (L > H && H > 0 && L + 1 < nblocks);
            if (ok) {
                const ptx_block_t *hb = &fn->blocks[H];
                ok = hb->count > 0 && block_terminator(hb) == POP_CBRA;
                if (ok) {
                    const ptx_instr_t *hl = &hb->instrs[hb->count - 1];
                    M = block_index_of(c, fn, hl->target);
                }
                ok = ok && M == nblocks - 1 &&
                     (block_terminator(&fn->blocks[M]) == POP_RET ||
                      block_terminator(&fn->blocks[M]) == POP_EXIT);
                if (ok) {
                    const ptx_block_t *pb = &fn->blocks[H - 1];
                    int pt = block_terminator(pb);
                    if (pt >= 0) {
                        // Preheader may jump straight to the header.
                        ok = pt == POP_BRA && pb->count > 0 &&
                             block_index_of(
                                 c, fn, pb->instrs[pb->count - 1].target) == H;
                    }
                }
            }
            for (int i = 0; ok && i < H; i++) {
                const ptx_block_t *b = &fn->blocks[i];
                int tt = block_terminator(b);
                if (tt == POP_CBRA || tt == POP_RET || tt == POP_EXIT)
                    ok = 0;
                else if (tt == POP_BRA && i < H - 1 &&
                         block_index_of(c, fn,
                                        b->instrs[b->count - 1].target) !=
                             i + 1)
                    ok = 0;
            }
            for (int i = H; ok && i <= L; i++) {
                const ptx_block_t *b = &fn->blocks[i];
                int tt = block_terminator(b);
                if (i == H)
                    continue;
                if (i == L) {
                    if (tt != POP_BRA || b->count == 0 ||
                        block_index_of(c, fn, b->instrs[b->count - 1].target) != H)
                        ok = 0;
                    continue;
                }
                if (tt == POP_CBRA) {
                    int tg = block_index_of(
                        c, fn, b->instrs[b->count - 1].target);
                    if (tg != M && tg != L)
                        ok = 0;
                } else if (tt == POP_BRA) {
                    int tg = block_index_of(
                        c, fn, b->instrs[b->count - 1].target);
                    if (tg != L && tg != i + 1)
                        ok = 0;
                } else if (tt == POP_RET || tt == POP_EXIT) {
                    ok = 0;
                }
            }
            if (ok) {
                cfg->kind = 3;
                cfg->loop_h = H;
                cfg->loop_l = L;
                cfg->loop_M = M;
                cfg->loop_self = 0;
                const ptx_instr_t *cb =
                    &fn->blocks[H].instrs[fn->blocks[H].count - 1];
                cfg->pred_neg = cb->pred_neg;
                if (strlen(cb->pred) >= sizeof(cfg->pred)) {
                    fail(c, "predicate name too long");
                    return -1;
                }
                strcpy(cfg->pred, cb->pred);
                reg_t *r = find_reg(c, cfg->pred);
                if (!r || r->decl_ty != PTY_PRED) {
                    fail(c, "loop predicate must be a .pred register");
                    return -1;
                }
                return 0;
            }
        }
        // Loop shapes only: exactly one back-edge event and no other
        // branches (one construct per kernel). Anything else is a clean
        // error naming the limitation.
        if (nback != 1 || ncbra > 1 || nub > 0) {
            // Try multi-region flow if all branches are valid for kind 4
            // (forward cbras/unconditional jumps; no backward conditional
            // branches except maybe none). flow_analyze will reject if invalid.
            if (flow_analyze(c, fn, cfg) == 0)
                return 0;
            fail(c, "loops must be a single header with one exit test and no other branches (multi-block loops unsupported)");
            return -1;
        }
        if (back_is_cbra[0]) {
            // Form (b): header H ends with cbra to itself.
            int H = back_blk[0];
            if (back_tgt[0] != H) {
                fail(c, "backward conditional branch must target its own block");
                return -1;
            }
            if (ncbra != 0) {
                fail(c, "loop header must not mix forward and backward branches");
                return -1;
            }
            if (H == 0 || H + 1 >= nblocks) {
                fail(c, "loop needs a preheader block and a following exit block");
                return -1;
            }
            int pt = block_terminator(&fn->blocks[H - 1]);
            if (pt != -1) {
                fail(c, "loop header must follow straight-line code");
                return -1;
            }
            for (int i = 0; i < nblocks; i++) {
                if (i == H)
                    continue;
                int tt = block_terminator(&fn->blocks[i]);
                if (i == nblocks - 1) {
                    if (tt != POP_RET && tt != POP_EXIT) {
                        fail(c, "kernel must end with ret/exit");
                        return -1;
                    }
                } else if (tt >= 0) {
                    fail(c, "only straight-line blocks allowed around a loop");
                    return -1;
                }
            }
            cfg->kind = 3;
            cfg->loop_h = H;
            cfg->loop_l = H;
            cfg->loop_M = H + 1;
            cfg->loop_self = 1;
            // Predicate from the self-cbra (terminal by construction).
            const ptx_instr_t *cb =
                &fn->blocks[H].instrs[fn->blocks[H].count - 1];
            cfg->pred_neg = cb->pred_neg;
            if (strlen(cb->pred) >= sizeof(cfg->pred)) {
                fail(c, "predicate name too long");
                return -1;
            }
            strcpy(cfg->pred, cb->pred);
            {
                reg_t *r = find_reg(c, cfg->pred);
                if (!r || r->decl_ty != PTY_PRED) {
                    fail(c, "loop predicate must be a .pred register");
                    return -1;
                }
            }
            return 0;
        }
        // Form (a): terminal backward bra.uni is the latch; the header H
        // holds the forward exit cbra (terminal in its own block, since
        // the parser splits at every branch).
        {
            int L = back_blk[0], H = back_tgt[0];
            if (L <= H) {
                fail(c, "loop latch must follow its header");
                return -1;
            }
            if (ncbra != 1 || j != H) {
                fail(c, "loop header needs exactly one forward exit test");
                return -1;
            }
            int M = t;
            if (M != nblocks - 1 ||
                (block_terminator(&fn->blocks[M]) != POP_RET &&
                 block_terminator(&fn->blocks[M]) != POP_EXIT)) {
                fail(c, "loop exit must target the final block ending in ret/exit");
                return -1;
            }
            if (H == 0) {
                fail(c, "loop needs a preheader block");
                return -1;
            }
            if (block_terminator(&fn->blocks[H - 1]) >= 0) {
                fail(c, "loop header must follow straight-line code");
                return -1;
            }
            for (int i = 0; i < nblocks; i++) {
                if (i == H || i == M)
                    continue;
                int tt = (i == L) ? POP_BRA
                                  : block_terminator(&fn->blocks[i]);
                if (i != L && tt >= 0) {
                    fail(c, "only straight-line blocks allowed inside a loop");
                    return -1;
                }
            }
            if (L + 1 != M) {
                fail(c, "no dead blocks allowed between loop latch and exit");
                return -1;
            }
            cfg->kind = 3;
            cfg->loop_h = H;
            cfg->loop_l = L;
            cfg->loop_M = M;
            cfg->loop_self = 0;
            cfg->pred_neg = pred_neg;
            strcpy(cfg->pred, pred);
            {
                reg_t *r = find_reg(c, cfg->pred);
                if (!r || r->decl_ty != PTY_PRED) {
                    fail(c, "loop predicate must be a .pred register");
                    return -1;
                }
            }
            return 0;
        }
    }
    if (ncbra > 1) {
        // Try multi-region flow for multiple forward branches
        if (flow_analyze(c, fn, cfg) == 0)
            return 0;
        fail(c, "at most one conditional forward branch per kernel");
        return -1;
    }
    if (ncbra == 0) {
        if (nub > 0) {
            // Check all unconditional jumps
            int ok_all = 1;
            for (int k = 0; k < nub; k++) {
                if (ub_tgt[k] != nblocks - 1) {
                    ok_all = 0;
                    break;
                }
            }
            if (!ok_all) {
                fail(c, "unconditional forward skip unsupported (dead code?)");
                return -1;
            }
        }
        // Straight line: final block must return.
        if (block_terminator(&fn->blocks[nblocks - 1]) < 0) {
            fail(c, "kernel must end with ret/exit");
            return -1;
        }
        cfg->kind = 0;
        return 0;
    }
    // Predicate must be a declared .pred register (defined-ness is
    // verified at emission: the setp precedes the branch).
    {
        reg_t *r = find_reg(c, pred);
        if (!r || r->decl_ty != PTY_PRED) {
            fail(c, "guard predicate must be a .pred register");
            return -1;
        }
    }
    cfg->j = j;
    cfg->t = t;
    cfg->pred_neg = pred_neg;
    strcpy(cfg->pred, pred);
    // Compact diamond: else body at j+1, then body at j+2, both ending
    // in a branch to the same final join. (The classic layout with a
    // one-instruction else marker stays on the kind-2 path below.)
    // Compact diamond: else body at j+1, then body at j+2, both ending
    // at the same final join (the then block may fall through when the
    // join immediately follows it).
    if (t == j + 2) {
        int e = j + 1;
        const ptx_block_t *eb = &fn->blocks[e];
        const ptx_block_t *tb = &fn->blocks[t];
        if (block_terminator(eb) == POP_BRA && eb->count > 0) {
            int J = block_index_of(c, fn, eb->instrs[eb->count - 1].target);
            int tto = -1;
            if (block_terminator(tb) == POP_BRA && tb->count > 0)
                tto = block_index_of(c, fn,
                                     tb->instrs[tb->count - 1].target);
            else if (t + 1 < nblocks)
                tto = t + 1;
            if (J > t && J == nblocks - 1 && tto == J &&
                block_terminator(&fn->blocks[J]) >= 0) {
                int straight = 1;
                for (int i = t + 1; i < J; i++)
                    if (block_terminator(&fn->blocks[i]) >= 0)
                        straight = 0;
                if (straight) {
                    cfg->kind = 7;
                    cfg->e = e;
                    cfg->J = J;
                    return 0;
                }
            }
        }
    }
    if (nub == 0) {
        // Guard: body straight-line, target final + return.
        cfg->kind = 1;
        for (int i = j + 1; i < t; i++) {
            if (block_terminator(&fn->blocks[i]) >= 0) {
                failf(c, fn->blocks[i].instrs[fn->blocks[i].count - 1].line,
                      "unsupported control flow in guard body");
                return -1;
            }
        }
        if (t != nblocks - 1 || block_terminator(&fn->blocks[t]) < 0) {
            fail(c, "guard target must be the final block ending in ret/exit");
            return -1;
        }
        return 0;
    }
    // Diamond: exactly two unconditional bras - the else marker at B(j+1)
    // and the then-block jump to the join.
    if (nub != 2) {
        fail(c, "if/else shape unsupported (want guard + 2 forward jumps)");
        return -1;
    }
    cfg->kind = 2;
    int ei = -1, ti = -1;
    for (int k = 0; k < 2; k++) {
        if (ub_blk[k] == j + 1 && fn->blocks[ub_blk[k]].count == 1)
            ei = k;
        else if (ub_blk[k] == t && fn->blocks[ub_blk[k]].count >= 1)
            ti = k;
    }
    if (ei < 0 || ti < 0) {
        fail(c, "if/else shape unsupported (else jump must follow guard)");
        return -1;
    }
    int e = ub_tgt[ei], J = ub_tgt[ti];
    cfg->e = e;
    if (e == t) {
        fail(c, "branch arms must differ");
        return -1;
    }
    if (t + 1 != e) {
        fail(c, "then-block must be single and immediately followed by else");
        return -1;
    }
    if (J <= t) {
        fail(c, "join must be forward");
        return -1;
    }
    for (int i = e; i < J; i++) {
        if (block_terminator(&fn->blocks[i]) >= 0) {
            fail(c, "else region must be straight-line");
            return -1;
        }
    }
    if (J != nblocks - 1 || block_terminator(&fn->blocks[J]) < 0) {
        fail(c, "join must be the final block ending in ret/exit");
        return -1;
    }
    cfg->J = J;
    return 0;
}


void emit_range(ctx_t *c, spv_blob_t *b, const ptx_block_t *blk,
                 size_t from, size_t to_excl);

// Nested-loop shape (kind 9): an outer form-(a) loop whose body contains
// exactly one inner form-(a) loop, with the inner merge block doubling as
// the outer latch. Block layout (Ho = outer header):
//   ...straight...        blocks 0..Ho-1 (last has no terminator)
//   Ho:     cbra Mo       outer exit test, fallthrough Ho+1
//   Ho+1:   straight      tile setup / accumulator reset
//   Ho+2:   cbra Mi       inner exit test, fallthrough Ho+3 (Hi)
//   Ho+3:   bra Hi        inner latch (Li)
//   Ho+4:   bra Ho        inner merge Mi == outer latch Lo
//   Ho+5:   ret           outer merge Mo (final)
static int analyze_nested(ctx_t *c, const ptx_func_t *fn, cfg_t *cfg) {
    int nb = (int)fn->nblocks;
    if (nb < 7)
        return -1;
    for (int Ho = 1; Ho + 5 < nb; Ho++) {
        const ptx_block_t *bh = &fn->blocks[Ho];
        if (block_terminator(bh) != POP_CBRA || bh->count == 0)
            continue;
        int Mo = block_index_of(c, fn, bh->instrs[bh->count - 1].target);
        if (Mo != nb - 1)
            continue;
        int mt = block_terminator(&fn->blocks[Mo]);
        if (mt != POP_RET && mt != POP_EXIT)
            continue;
        int ok = 1;
        for (int i = 0; i < Ho && ok; i++)
            if (block_terminator(&fn->blocks[i]) >= 0)
                ok = 0;
        if (!ok)
            continue;
        int Hi = Ho + 2;
        if (Hi >= nb || block_terminator(&fn->blocks[Ho + 1]) >= 0)
            continue;
        const ptx_block_t *bi = &fn->blocks[Hi];
        if (block_terminator(bi) != POP_CBRA || bi->count == 0)
            continue;
        int Mi = block_index_of(c, fn, bi->instrs[bi->count - 1].target);
        if (Mi <= Hi + 1 || Mi >= Mo)
            continue;
        int Li = Mi - 1;
        const ptx_block_t *bl = &fn->blocks[Li];
        if (block_terminator(bl) != POP_BRA || bl->count == 0)
            continue;
        if (block_index_of(c, fn, bl->instrs[bl->count - 1].target) != Hi)
            continue;
        // Inner body blocks Hi+1..Li-1 must be straight-line.
        for (int b = Hi + 1; b < Li && ok; b++)
            if (block_terminator(&fn->blocks[b]) >= 0)
                ok = 0;
        // Blocks Mi..Lo-1 straight (optionally bra to next); Lo ends with
        // bra Ho (the outer latch).
        int Lo = -1;
        for (int b = Mi; b < Mo; b++) {
            int t = block_terminator(&fn->blocks[b]);
            if (t == POP_BRA) {
                int tg = block_index_of(c, fn,
                                        fn->blocks[b].instrs[
                                            fn->blocks[b].count - 1].target);
                if (tg == Ho) {
                    Lo = b;
                    break;
                }
                if (tg != b + 1) {
                    ok = 0;
                    break;
                }
            } else if (t >= 0) {
                ok = 0;
                break;
            }
        }
        if (!ok || Lo < Mi || Lo + 1 != Mo)
            continue;
        cfg->kind = 9;
        cfg->loop_h = Ho;
        cfg->loop_l = Lo;
        cfg->loop_M = Mo;
        cfg->loop_h2 = Hi;
        cfg->loop_l2 = Li;
        cfg->loop_M2 = Mi;
        cfg->pred_neg = bh->instrs[bh->count - 1].pred_neg;
        cfg->pred_neg2 = bi->instrs[bi->count - 1].pred_neg;
        if (strlen(bh->instrs[bh->count - 1].pred) >= sizeof(cfg->pred) ||
            strlen(bi->instrs[bi->count - 1].pred) >= sizeof(cfg->pred2))
            return -1;
        strcpy(cfg->pred, bh->instrs[bh->count - 1].pred);
        strcpy(cfg->pred2, bi->instrs[bi->count - 1].pred);
        reg_t *r = find_reg(c, cfg->pred);
        reg_t *r2 = find_reg(c, cfg->pred2);
        if (!r || r->decl_ty != PTY_PRED || !r2 || r2->decl_ty != PTY_PRED)
            return -1;
        return 0;
    }
    return -1;
}

// Emit the kind-9 nested loop. No breaks are supported: both loops exit
// only through their header tests, and the inner merge is the outer latch.
void emit_nested_loop(ctx_t *c, const ptx_func_t *fn, const cfg_t *cfg) {
    int Ho = cfg->loop_h, Lo = cfg->loop_l, Mo = cfg->loop_M;
    int Hi = cfg->loop_h2, Li = cfg->loop_l2, Mi = cfg->loop_M2;
    int nregs = c->nregs;
    size_t n = nregs ? (size_t)nregs : 1;
    uint32_t *pre = malloc(sizeof(*pre) * n);
    char *pre_def = malloc(n), *pre_ptr = malloc(n);
    char *out_def = calloc(n, 1), *in_def = calloc(n, 1);
    uint32_t *ipre = malloc(sizeof(*ipre) * n);
    uint32_t *ihdr = malloc(sizeof(*ihdr) * n);
    uint32_t *ohdr = malloc(sizeof(*ohdr) * n);
    char *ipre_def = malloc(n);
    size_t *out_patch = calloc(n, sizeof(*out_patch));
    size_t *in_patch = calloc(n, sizeof(*in_patch));
    if (!pre || !pre_def || !pre_ptr || !out_def || !in_def || !ipre ||
        !ihdr || !ohdr || !ipre_def || !out_patch || !in_patch) {
        fail(c, "out of memory");
        goto out;
    }
    for (int i = 0; i < nregs; i++) {
        pre[i] = c->regs[i].cur;
        pre_def[i] = c->regs[i].defined ? 1 : 0;
        pre_ptr[i] = c->regs[i].is_ptr ? 1 : 0;
    }
    // Region def scans (skip terminators).
    for (int bi = Ho; bi <= Lo; bi++) {
        const ptx_block_t *blk = &fn->blocks[bi];
        size_t to = blk->count;
        if (block_terminator(blk) >= 0 && to > 0)
            to--;
        for (size_t k = 0; k < to; k++) {
            const ptx_instr_t *in = &blk->instrs[k];
            if (!in->dst[0])
                continue;
            reg_t *r = find_reg(c, in->dst);
            if (!r)
                continue;
            int idx = (int)(r - c->regs);
            if (idx < 0 || idx >= nregs)
                continue;
            out_def[idx] = 1;
            if (bi >= Hi && bi <= Li)
                in_def[idx] = 1;
            if (pre_ptr[idx] &&
                ((in->op == POP_MOV || in->op == POP_ADD) &&
                 in->ty == PTY_U64)) {
                failf(c, in->line, "loop-carried pointer unsupported "
                                   "(recompute addresses from carried indices)");
                goto out;
            }
        }
    }
    uint32_t cont_o = nid(c), cont_i = nid(c);
    // ---- outer header ----
    put(c, &c->fn, (2u << 16) | S_Label);
    put(c, &c->fn, c->block_ids[Ho]);
    for (int i = 0; i < nregs; i++) {
        if (!out_def[i] || !pre_def[i])
            continue;
        uint32_t phi = nid(c);
        put(c, &c->fn, (7u << 16) | S_Phi);
        put(c, &c->fn, ty_of(c, c->regs[i].decl_ty));
        put(c, &c->fn, phi);
        put(c, &c->fn, pre[i]);
        put(c, &c->fn, c->block_ids[Ho - 1]);
        out_patch[i] = c->fn.count;
        put(c, &c->fn, 0);
        put(c, &c->fn, cont_o);
        c->regs[i].cur = phi;
    }
    {
        const ptx_block_t *bh = &fn->blocks[Ho];
        emit_range(c, &c->fn, bh, 0, bh->count - 1);
    }
    for (int i = 0; i < nregs; i++)
        ohdr[i] = c->regs[i].cur;
    {
        reg_t *pr = find_reg(c, cfg->pred);
        uint32_t cond = pr ? pr->cur : 0;
        if (!cond) { fail(c, "outer loop predicate has no value"); goto out; }
        if (cfg->pred_neg) {
            uint32_t nn = nid(c);
            put(c, &c->fn, (4u << 16) | S_LogicalNot);
            put(c, &c->fn, c->t_bool);
            put(c, &c->fn, nn);
            put(c, &c->fn, cond);
            cond = nn;
        }
        put(c, &c->fn, (4u << 16) | S_LoopMerge);
        put(c, &c->fn, c->block_ids[Mo]);
        put(c, &c->fn, cont_o);
        put(c, &c->fn, 0);
        put(c, &c->fn, (4u << 16) | S_BranchCond);
        put(c, &c->fn, cond);
        put(c, &c->fn, c->block_ids[Mo]);
        put(c, &c->fn, c->block_ids[Ho + 1]);
    }
    // ---- prep block ----
    put(c, &c->fn, (2u << 16) | S_Label);
    put(c, &c->fn, c->block_ids[Ho + 1]);
    emit_range(c, &c->fn, &fn->blocks[Ho + 1], 0,
               fn->blocks[Ho + 1].count);
    emit_branch(c, &c->fn, c->block_ids[Hi]);
    // snapshot inner preheader
    for (int i = 0; i < nregs; i++) {
        ipre[i] = c->regs[i].cur;
        ipre_def[i] = c->regs[i].defined ? 1 : 0;
    }
    // ---- inner header ----
    put(c, &c->fn, (2u << 16) | S_Label);
    put(c, &c->fn, c->block_ids[Hi]);
    for (int i = 0; i < nregs; i++) {
        if (!in_def[i] || !ipre_def[i])
            continue;
        uint32_t phi = nid(c);
        put(c, &c->fn, (7u << 16) | S_Phi);
        put(c, &c->fn, ty_of(c, c->regs[i].decl_ty));
        put(c, &c->fn, phi);
        put(c, &c->fn, ipre[i]);
        put(c, &c->fn, c->block_ids[Ho + 1]);
        in_patch[i] = c->fn.count;
        put(c, &c->fn, 0);
        put(c, &c->fn, cont_i);
        c->regs[i].cur = phi;
    }
    {
        const ptx_block_t *bi = &fn->blocks[Hi];
        emit_range(c, &c->fn, bi, 0, bi->count - 1);
    }
    for (int i = 0; i < nregs; i++)
        ihdr[i] = c->regs[i].cur;
    {
        reg_t *pr = find_reg(c, cfg->pred2);
        uint32_t cond = pr ? pr->cur : 0;
        if (!cond) { fail(c, "inner loop predicate has no value"); goto out; }
        if (cfg->pred_neg2) {
            uint32_t nn = nid(c);
            put(c, &c->fn, (4u << 16) | S_LogicalNot);
            put(c, &c->fn, c->t_bool);
            put(c, &c->fn, nn);
            put(c, &c->fn, cond);
            cond = nn;
        }
        put(c, &c->fn, (4u << 16) | S_LoopMerge);
        put(c, &c->fn, c->block_ids[Mi]);
        put(c, &c->fn, cont_i);
        put(c, &c->fn, 0);
        put(c, &c->fn, (4u << 16) | S_BranchCond);
        put(c, &c->fn, cond);
        put(c, &c->fn, c->block_ids[Mi]);
        put(c, &c->fn, c->block_ids[Hi + 1]);
    }
    // ---- inner body + latch ----
    put(c, &c->fn, (2u << 16) | S_Label);
    put(c, &c->fn, c->block_ids[Li]);
    emit_range(c, &c->fn, &fn->blocks[Li], 0, fn->blocks[Li].count - 1);
    emit_branch(c, &c->fn, cont_i);
    put(c, &c->fn, (2u << 16) | S_Label);
    put(c, &c->fn, cont_i);
    emit_branch(c, &c->fn, c->block_ids[Hi]);
    for (int i = 0; i < nregs; i++) {
        if (!in_def[i] || !ipre_def[i])
            continue;
        if (c->regs[i].cur == 0) {
            char msg[160];
            snprintf(msg, sizeof(msg),
                     "inner loop-carried register %s lost its value",
                     c->regs[i].name);
            failf(c, 0, msg);
            goto out;
        }
        c->fn.words[in_patch[i]] = c->regs[i].cur;
    }
    // The inner merge is reached from the header test, so its values are
    // the header-edge values, not the latch values (which do not dominate
    // the merge). Only carried registers may be used past the merge.
    for (int i = 0; i < nregs; i++)
        if (in_def[i] && ipre_def[i])
            c->regs[i].cur = ihdr[i];
    // ---- inner merge, then the remaining outer body up to the latch ----
    for (int b = Mi; b <= Lo; b++) {
        const ptx_block_t *bb = &fn->blocks[b];
        size_t cnt = bb->count;
        if (block_terminator(bb) >= 0 && cnt > 0)
            cnt--;
        put(c, &c->fn, (2u << 16) | S_Label);
        put(c, &c->fn, c->block_ids[b]);
        emit_range(c, &c->fn, bb, 0, cnt);
        if (b == Lo)
            emit_branch(c, &c->fn, cont_o);
        else
            emit_branch(c, &c->fn, c->block_ids[b + 1]);
    }
    put(c, &c->fn, (2u << 16) | S_Label);
    put(c, &c->fn, cont_o);
    emit_branch(c, &c->fn, c->block_ids[Ho]);
    for (int i = 0; i < nregs; i++) {
        if (!out_def[i] || !pre_def[i])
            continue;
        if (c->regs[i].cur == 0) {
            char msg[160];
            snprintf(msg, sizeof(msg),
                     "outer loop-carried register %s lost its value",
                     c->regs[i].name);
            failf(c, 0, msg);
            goto out;
        }
        c->fn.words[out_patch[i]] = c->regs[i].cur;
    }
    for (int i = 0; i < nregs; i++)
        if (out_def[i] && pre_def[i])
            c->regs[i].cur = ohdr[i];
    // ---- outer merge ----
    put(c, &c->fn, (2u << 16) | S_Label);
    put(c, &c->fn, c->block_ids[Mo]);
    emit_range(c, &c->fn, &fn->blocks[Mo], 0, fn->blocks[Mo].count - 1);
    put(c, &c->fn, (1u << 16) | S_Return);
out:
    free(pre); free(pre_def); free(pre_ptr); free(out_def); free(in_def);
    free(ipre); free(ihdr); free(ohdr); free(ipre_def);
    free(out_patch); free(in_patch);
}

// Public entry: try the specialised structured emitters first.  If
// the function does not fit any of their shapes, fall back to the
// generic state-machine lowering (kind 8), which can represent any
// control-flow graph.  The structured attempt leaves no emission
// behind, so its diagnostic is discarded on fallback.
int analyze_cfg(ctx_t *c, const ptx_func_t *fn, cfg_t *cfg) {
    if (analyze_cfg_structured(c, fn, cfg) == 0)
        return 0;
    cfg_free(cfg);
    memset(cfg, 0, sizeof(*cfg));
    cfg->loop_h = cfg->loop_l = cfg->loop_M = -1;
    cfg->loop_h2 = cfg->loop_l2 = cfg->loop_M2 = -1;
    c->failed = 0;
    if (c->err && c->errlen)
        c->err[0] = '\0';
    if (analyze_nested(c, fn, cfg) == 0)
        return 0;
    cfg_free(cfg);
    memset(cfg, 0, sizeof(*cfg));
    cfg->loop_h = cfg->loop_l = cfg->loop_M = -1;
    cfg->loop_h2 = cfg->loop_l2 = cfg->loop_M2 = -1;
    c->failed = 0;
    if (c->err && c->errlen)
        c->err[0] = '\0';
    cfg->kind = 8;
    return 0;
}

void emit_branch(ctx_t *c, spv_blob_t *b, uint32_t target) {    put(c, b, (2u << 16) | S_Branch);
    put(c, b, target);
}

void emit_block_body(ctx_t *c, spv_blob_t *b, const ptx_block_t *blk,
                            int skip_last) {
    size_t n = blk->count;
    if (skip_last && n > 0)
        n--;
    for (size_t i = 0; i < n; i++)
        emit_instr(c, b, &blk->instrs[i]);
}

// Emits instructions [from, to_excl) of one block.
void emit_range(ctx_t *c, spv_blob_t *b, const ptx_block_t *blk,
                 size_t from, size_t to_excl) {
    for (size_t i = from; i < to_excl && i < blk->count; i++)
        emit_instr(c, b, &blk->instrs[i]);
}

// Resolves a branch predicate to a condition id (handles @!).
static uint32_t loop_cond(ctx_t *c, spv_blob_t *b, const cfg_t *cfg) {
    reg_t *pr = find_reg(c, cfg->pred);
    uint32_t cond = pr ? pr->cur : 0;
    if (!cond) {
        fail(c, "branch predicate has no value");
        return 0;
    }
    if (cfg->pred_neg) {
        uint32_t n = nid(c);
        put(c, b, (4u << 16) | S_LogicalNot);
        put(c, b, c->t_bool);
        put(c, b, n);
        put(c, b, cond);
        cond = n;
    }
    return cond;
}

// Emits a single-header loop (cfg kind 3) at block H. Register-carried
// values get OpPhi nodes (preheader + continue-block incomings); latch
// operands are patched after the body is emitted. Loop-carried
// POINTERS are rejected: recompute addresses from carried indices
// instead (pointer bump idiom unsupported).
uint32_t last_emitted_label(const spv_blob_t *blob) {
    uint32_t last = 0;
    size_t i = 0;
    while (i + 1 < blob->count) {
        uint32_t w = blob->words[i];
        uint16_t op = (uint16_t)(w & 0xffffu);
        uint16_t wc = (uint16_t)(w >> 16);
        if (wc == 0)
            break;
        if (op == S_Label && i + 1 < blob->count)
            last = blob->words[i + 1];
        i += wc;
    }
    return last;
}

void emit_loop(ctx_t *c, const ptx_func_t *fn, const cfg_t *cfg) {
    int H = cfg->loop_h, L = cfg->loop_l, M = cfg->loop_M;
    int nregs = c->nregs;
    size_t nalloc = (size_t)(nregs ? nregs : 1);
    uint32_t *pre_cur = malloc(sizeof(*pre_cur) * nalloc);
    char *pre_def = malloc(nalloc);
    char *pre_ptr = malloc(nalloc);
    char *def_in = calloc(nalloc, 1);
    size_t *patch_at = malloc(sizeof(*patch_at) * nalloc);
    int *is_carried = malloc(sizeof(*is_carried) * nalloc);
    uint32_t *hdr_cur = malloc(sizeof(*hdr_cur) * nalloc);
    uint32_t *brk_lab = malloc(sizeof(*brk_lab) * (size_t)fn->nblocks);
    uint32_t *brk_val = malloc(sizeof(*brk_val) * (size_t)fn->nblocks *
                              nalloc);
    uint32_t *lpred_lab = malloc(sizeof(*lpred_lab) * (size_t)fn->nblocks);
    uint32_t *lpred_val = malloc(sizeof(*lpred_val) * (size_t)fn->nblocks *
                                 nalloc);
    int nbrk = 0, nlpred = 0;
    if (!pre_cur || !pre_def || !pre_ptr || !def_in || !patch_at ||
        !is_carried || !hdr_cur || !brk_lab || !brk_val || !lpred_lab ||
        !lpred_val) {
        free(hdr_cur);
        free(brk_lab);
        free(brk_val);
        free(lpred_lab);
        free(lpred_val);
        free(pre_cur);
        free(pre_def);
        free(pre_ptr);
        free(def_in);
        free(patch_at);
        free(is_carried);
        fail(c, "out of memory");
        return;
    }
    for (int i = 0; i < nregs; i++) {
        pre_def[i] = c->regs[i].defined ? 1 : 0;
        pre_cur[i] = c->regs[i].cur;
        pre_ptr[i] = c->regs[i].is_ptr ? 1 : 0;
    }
    // Region def scan. Form (a): H minus terminal cbra, bodies H+1..L
    // minus latch bra. Form (b): H minus terminal self-cbra.
    for (int bi = H; bi <= L; bi++) {
        const ptx_block_t *blk = &fn->blocks[bi];
        size_t from = 0, to = blk->count;
        if (bi == H)
            to = blk->count > 0 ? blk->count - 1 : 0;
        if (bi == L && !cfg->loop_self)
            to = blk->count > 0 ? blk->count - 1 : 0;
        for (size_t k = from; k < to; k++) {
            const ptx_instr_t *in = &blk->instrs[k];
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
                ((in->op == POP_MOV || in->op == POP_ADD) &&
                 in->ty == PTY_U64)) {
                failf(c, in->line, "loop-carried pointer unsupported "
                                   "(recompute addresses from carried indices)");
                goto out;
            }
        }
    }
    // The preheader is the block whose code was emitted immediately
    // before the loop header; using H-1 breaks when an if/else or a
    // prior loop was emitted inside that PTX block.
    uint32_t pre_label = last_emitted_label(&c->fn);
    if (pre_label == 0)
        pre_label = c->block_ids[H - 1];
    uint32_t cont = nid(c);
    // Header label + phis (latch operands patched below), then the
    // LoopMerge immediately before the terminating branch.
    put(c, &c->fn, (2u << 16) | S_Label);
    put(c, &c->fn, c->block_ids[H]);
    // Phi per carried reg, in reg-index order for determinism.
    for (int i = 0; i < nregs; i++) {
        is_carried[i] = 0;
        patch_at[i] = 0;
        if (!def_in[i] || !pre_def[i])
            continue;
        is_carried[i] = 1;
        uint32_t phi = nid(c);
        put(c, &c->fn, (7u << 16) | S_Phi);
        put(c, &c->fn, ty_of(c, c->regs[i].decl_ty));
        put(c, &c->fn, phi);
        put(c, &c->fn, pre_cur[i]);
        put(c, &c->fn, pre_label);
        patch_at[i] = c->fn.count;
        put(c, &c->fn, 0); // patched to latch value below
        put(c, &c->fn, cont);
        c->regs[i].cur = phi;
        if (c->failed)
            goto out;
    }
    {
        if (!cfg->loop_self) {
            // Form (a): head-part, exit to M when true, else body.
            const ptx_block_t *hblk = &fn->blocks[H];
            emit_range(c, &c->fn, hblk, 0,
                       hblk->count > 0 ? hblk->count - 1 : 0);
            if (c->failed)
                goto out;
        } else {
            // Form (b): whole header is the body; continue on true.
            const ptx_block_t *hblk = &fn->blocks[H];
            emit_range(c, &c->fn, hblk, 0,
                       hblk->count > 0 ? hblk->count - 1 : 0);
            if (c->failed)
                goto out;
        }
        for (int i = 0; i < nregs; i++)
            hdr_cur[i] = c->regs[i].cur;
        // Predicate is defined by the just-emitted code; resolve now.
        uint32_t cond = loop_cond(c, &c->fn, cfg);
        if (c->failed)
            goto out;
        put(c, &c->fn, (4u << 16) | S_LoopMerge);
        put(c, &c->fn, c->block_ids[M]);
        put(c, &c->fn, cont);
        put(c, &c->fn, 0);
        if (!cfg->loop_self) {
            put(c, &c->fn, (4u << 16) | S_BranchCond);
            put(c, &c->fn, cond);
            put(c, &c->fn, c->block_ids[M]);
            put(c, &c->fn, c->block_ids[H + 1]);
            for (int bi = H + 1; bi <= L; bi++) {
                const ptx_block_t *blk = &fn->blocks[bi];
                int tt = block_terminator(blk);
                size_t to = blk->count;
                if (tt >= 0 && to > 0)
                    to--;
                put(c, &c->fn, (2u << 16) | S_Label);
                put(c, &c->fn, c->block_ids[bi]);
                if (bi == L && nlpred > 1) {
                    for (int ri = 0; ri < nregs; ri++) {
                        if (!is_carried[ri])
                            continue;
                        uint32_t phi = nid(c);
                        put(c, &c->fn,
                            (uint32_t)((3 + 2 * nlpred) << 16) | S_Phi);
                        put(c, &c->fn,
                            ty_of(c, c->regs[ri].decl_ty));
                        put(c, &c->fn, phi);
                        for (int k = 0; k < nlpred; k++) {
                            put(c, &c->fn,
                                lpred_val[(size_t)k * nalloc + ri]);
                            put(c, &c->fn, lpred_lab[k]);
                        }
                        c->regs[ri].cur = phi;
                    }
                }
                emit_range(c, &c->fn, blk, 0, to);
                if (c->failed)
                    goto out;
                uint32_t nxt = (bi < L) ? c->block_ids[bi + 1] : cont;
                // Record values on every edge that enters the latch.
                if (bi < L) {
                    int tt2 = block_terminator(blk);
                    int enters_L = 0;
                    if (tt2 == POP_CBRA) {
                        const ptx_instr_t *br2 =
                            &blk->instrs[blk->count - 1];
                        int tg2 = block_index_of(c, fn, br2->target);
                        enters_L = (tg2 == L);
                    } else if (tt2 == POP_BRA) {
                        const ptx_instr_t *br2 =
                            &blk->instrs[blk->count - 1];
                        enters_L =
                            block_index_of(c, fn, br2->target) == L;
                    } else {
                        enters_L = (bi + 1 == L);
                    }
                    if (enters_L) {
                        lpred_lab[nlpred] = c->block_ids[bi];
                        for (int ri = 0; ri < nregs; ri++)
                            lpred_val[(size_t)nlpred * nalloc + ri] =
                                c->regs[ri].cur;
                        nlpred++;
                    }
                }
                if (bi == L) {
                    emit_branch(c, &c->fn, cont);
                    continue;
                }
                if (tt == POP_CBRA) {
                    const ptx_instr_t *br = &blk->instrs[blk->count - 1];
                    reg_t *pr = find_reg(c, br->pred);
                    if (!pr || !pr->defined) {
                        fail(c, "branch predicate has no value");
                        goto out;
                    }
                    uint32_t bc = pr->cur;
                    if (br->pred_neg) {
                        uint32_t n = nid(c);
                        put(c, &c->fn, (4u << 16) | S_LogicalNot);
                        put(c, &c->fn, c->t_bool);
                        put(c, &c->fn, n);
                        put(c, &c->fn, bc);
                        bc = n;
                    }
                    int tg = block_index_of(c, fn, br->target);
                    if (tg == M) {
                        brk_lab[nbrk] = c->block_ids[bi];
                        for (int ri = 0; ri < nregs; ri++)
                            brk_val[(size_t)nbrk * nalloc + ri] =
                                c->regs[ri].cur;
                        nbrk++;
                    }
                    if (tg == L && nxt == c->block_ids[L]) {
                        emit_branch(c, &c->fn, c->block_ids[L]);
                    } else {
                        // The selection merge is the break target (loop
                        // exit) or continue target (latch), not the next
                        // block: both arms reconverge there.
                        uint32_t merge = (tg == L)
                                             ? c->block_ids[tg]
                                             : nxt;
                        if (merge == c->block_ids[tg] &&
                            merge == nxt) {
                            emit_branch(c, &c->fn, merge);
                        } else {
                            put(c, &c->fn, (3u << 16) | S_SelectionMerge);
                            put(c, &c->fn, merge);
                            put(c, &c->fn, 0);
                            put(c, &c->fn, (4u << 16) | S_BranchCond);
                            put(c, &c->fn, bc);
                            put(c, &c->fn, c->block_ids[tg]);
                            put(c, &c->fn, nxt);
                        }
                    }
                } else if (tt == POP_BRA) {
                    const ptx_instr_t *br = &blk->instrs[blk->count - 1];
                    int tg = block_index_of(c, fn, br->target);
                    if (tg != L && tg != bi + 1) {
                        fail(c, "unsupported branch inside loop body");
                        goto out;
                    }
                    emit_branch(c, &c->fn, c->block_ids[tg]);
                } else {
                    emit_branch(c, &c->fn, nxt);
                }
            }
        } else {
            put(c, &c->fn, (4u << 16) | S_BranchCond);
            put(c, &c->fn, cond);
            put(c, &c->fn, cont);
            put(c, &c->fn, c->block_ids[M]);
        }
        put(c, &c->fn, (2u << 16) | S_Label);
        put(c, &c->fn, cont);
        emit_branch(c, &c->fn, c->block_ids[H]);
    }
    // Patch phi latch operands now that body values exist.
    for (int i = 0; i < nregs; i++) {
        if (!is_carried[i])
            continue;
        reg_t *r = &c->regs[i];
        if (!r->defined) {
            fail(c, "loop-carried register lost its value");
            goto out;
        }
        c->fn.words[patch_at[i]] = r->cur;
    }
    // Exit block: merge the header-exit value with every break path.
    if (!cfg->loop_self) {
        put(c, &c->fn, (2u << 16) | S_Label);
        put(c, &c->fn, c->block_ids[M]);
        for (int i = 0; i < nregs; i++) {
            if (!is_carried[i])
                continue;
            if (nbrk > 0) {
                uint32_t phi = nid(c);
                put(c, &c->fn,
                    (uint32_t)((3 + 2 * (1 + nbrk)) << 16) | S_Phi);
                put(c, &c->fn, ty_of(c, c->regs[i].decl_ty));
                put(c, &c->fn, phi);
                put(c, &c->fn, hdr_cur[i]);
                put(c, &c->fn, c->block_ids[H]);
                for (int k = 0; k < nbrk; k++) {
                    put(c, &c->fn, brk_val[(size_t)k * nalloc + i]);
                    put(c, &c->fn, brk_lab[k]);
                }
                c->regs[i].cur = phi;
            } else {
                c->regs[i].cur = hdr_cur[i];
            }
        }
        const ptx_block_t *mb = &fn->blocks[M];
        size_t mto = mb->count > 0 ? mb->count - 1 : 0;
        emit_range(c, &c->fn, mb, 0, mto);
        if (c->failed)
            goto out;
        put(c, &c->fn, (1u << 16) | S_Return);
    }
out:
    free(pre_cur);
    free(pre_def);
    free(pre_ptr);
    free(def_in);
    free(patch_at);
    free(is_carried);
    free(hdr_cur);
    free(brk_lab);
    free(brk_val);
    free(lpred_lab);
    free(lpred_val);
}

