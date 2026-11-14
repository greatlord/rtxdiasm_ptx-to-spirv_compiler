// Entry/function signature and body parsing.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ptx.h"
#include "parse/parse_internal.h"

// Do-while latch guard: a backward `cbra` on a `setp.ne` latch over a
// monotonically +1-stepped counter spins ~2^32 iterations when the entry
// range is degenerate (start >= end) or the counter overshoots the limit
// (FluidMark density hangs the device this way on stale/empty cells).
// Rewrite NE to a directional compare (LTU/LT or GTU/GT matching operand
// order and signedness) when the counter is provably incremented exactly
// once in the loop by `add c, c, 1` and the limit is loop-invariant.
// For valid ranges (limit - start > 0, step +1) the directional compare
// trips trip-for-trip identically to NE; degenerate/overshoot input now
// exits instead of wrapping.
static void latch_guard(ptx_func_t *fn) {
    for (size_t bi = 0; bi < fn->nblocks; bi++) {
        ptx_block_t *blk = &fn->blocks[bi];
        // Raw IR blocks split at labels, not at branches: the latch
        // `cbra` may sit mid-block with the fallthrough below it.
        for (size_t bpos = 0; bpos < blk->count; bpos++) {
        ptx_instr_t *br = &blk->instrs[bpos];
        if (br->op != POP_CBRA || !br->pred[0] || br->pred_neg)
            continue;
        // Resolve the branch target; require a unique backward label so
        // duplicate-label forward edges can never be misread as loops.
        size_t tgt = fn->nblocks, nmatch = 0;
        for (size_t j = 0; j < fn->nblocks; j++) {
            if (fn->blocks[j].label[0] &&
                strcmp(fn->blocks[j].label, br->target) == 0) {
                if (nmatch == 0)
                    tgt = j;
                nmatch++;
            }
        }
        if (nmatch != 1 || tgt > bi)
            continue;
        // The latch predicate must come from exactly one plain
        // `setp.ne.<u32|u64|s32|s64> q, x, y` before the branch in this
        // block.
        ptx_instr_t *sp = NULL;
        int ambiguous = 0;
        for (size_t k = 0; k < bpos; k++) {
            ptx_instr_t *in = &blk->instrs[k];
            if (in->op == POP_SETP && strcmp(in->dst, br->pred) == 0) {
                if (sp)
                    ambiguous = 1;
                sp = in;
            }
        }
        if (ambiguous)
            continue;
        if (!sp || sp->cmp != PCM_NE || sp->bop != 0 || sp->dst2[0] ||
            sp->pred[0] || sp->nsrc != 2)
            continue;
        if (sp->ty != PTY_U32 && sp->ty != PTY_U64)
            continue;
        if (sp->src[0].kind != OPD_REG || sp->src[1].kind != OPD_REG)
            continue;
        // The latch value must be used nowhere else: the rewritten value
        // would otherwise leak to other readers. (The latch branch itself
        // is skipped in this scan, so zero other uses are allowed.)
        int uses = 0;
        for (size_t ob = 0; ob < fn->nblocks && uses <= 1; ob++) {
            ptx_block_t *o = &fn->blocks[ob];
            for (size_t k = 0; k < o->count; k++) {
                ptx_instr_t *in = &o->instrs[k];
                if (in == br)
                    continue;
                if (in->pred[0] && strcmp(in->pred, br->pred) == 0)
                    uses++;
                for (int s = 0; s < in->nsrc && uses <= 1; s++)
                    if ((in->src[s].kind == OPD_REG ||
                         in->src[s].kind == OPD_REGOFF) &&
                        strcmp(in->src[s].reg, br->pred) == 0)
                        uses++;
                if (uses > 1)
                    break;
            }
        }
        if (uses != 0)
            continue;
        // Try each operand order: the counter is the side incremented
        // exactly once by `add c, c, 1` in [tgt..bi], the limit side
        // must be loop-invariant.
        for (int order = 0; order < 2; order++) {
            const char *cnm = sp->src[order].reg;
            const char *enm = sp->src[order ^ 1].reg;
            int adds = 0, edefs = 0, ok = 1;
            for (size_t lb = tgt; lb <= bi && ok; lb++) {
                ptx_block_t *l = &fn->blocks[lb];
                for (size_t k = 0; k < l->count; k++) {
                    ptx_instr_t *in = &l->instrs[k];
                    if (in == sp || in == br)
                        continue;
                    int wrc = (in->dst[0] && strcmp(in->dst, cnm) == 0) ||
                              (in->dst2[0] && strcmp(in->dst2, cnm) == 0);
                    int wre = (in->dst[0] && strcmp(in->dst, enm) == 0) ||
                              (in->dst2[0] && strcmp(in->dst2, enm) == 0);
                    for (int r = 0; r < in->nrets; r++)
                        if (strcmp(in->rets[r], cnm) == 0)
                            wrc = 1;
                    for (int r = 0; r < in->nrets; r++)
                        if (strcmp(in->rets[r], enm) == 0)
                            wre = 1;
                    if (wre)
                        edefs++;
                    if (wrc) {
                        // Must be `add.<same-ty> c, c, 1` (either order).
                        if (in->op != POP_ADD || in->ty != sp->ty ||
                            in->nsrc != 2) {
                            ok = 0;
                            break;
                        }
                        int self = -1, imm = -1;
                        for (int s = 0; s < 2; s++) {
                            if (in->src[s].kind == OPD_REG &&
                                strcmp(in->src[s].reg, cnm) == 0)
                                self = s;
                            else if (in->src[s].kind == OPD_IMM &&
                                     !in->src[s].imm_is_float &&
                                     in->src[s].imm == 1)
                                imm = s;
                        }
                        if (self < 0 || imm < 0 || self == imm) {
                            ok = 0;
                            break;
                        }
                        adds++;
                    }
                }
            }
            if (!ok || adds != 1 || edefs != 0)
                continue;
            // Integer LT/GT lower to S_ or U_ opcodes from `sign`
            // (LTU/GTU codes are unordered-float only).
            sp->cmp = (order == 0 ? PCM_LT : PCM_GT);
            if (getenv("ACUDA_TRACE"))
                fprintf(stderr, "acuda-latch-guard: ne->%c%c loop %s..%s line %d\n",
                        order == 0 ? 'l' : 'g', sp->sign ? 's' : 'u',
                        fn->blocks[tgt].label[0] ? fn->blocks[tgt].label
                                                 : "(entry)",
                        blk->label[0] ? blk->label : "(fallthrough)",
                        sp->line);
            break;
        }
        }
    }
}

// Forward-latch variant: the `setp.ne` latch sits in the loop header with
// a *forward* branch into the body (FluidMark density: header tests
// counter vs limit, body follows, `bra.uni` closes the loop). Counter and
// limit are (re)loaded on the latch-false first-entry path, so only a
// loop-aware analysis can accept them. The rewrite NE->LT/GT preserves
// behavior of every execution that terminates today: the sole divergence
// is a latch evaluated with counter > limit, from which the +1-stepped
// counter over an unchanging limit never returns (hang today).
// Fires only when, over the back-edge's natural loop: the latch target
// is inside; the single `add c,c,1` (modulo mov-copy aliases) sits on
// the cycle; every other counter/limit definition sits in first-entry
// blocks reachable solely via the latch-false edge; the latch predicate
// is used nowhere else.
static size_t lg_resolve(ptx_func_t *fn, const char *label, int *unique) {
    size_t tgt = fn->nblocks, n = 0;
    for (size_t j = 0; j < fn->nblocks; j++) {
        if (fn->blocks[j].label[0] && strcmp(fn->blocks[j].label, label) == 0) {
            if (n == 0)
                tgt = j;
            n++;
        }
    }
    *unique = (n == 1);
    return tgt;
}

static int lg_is_terminal(const ptx_instr_t *in) {
    if (in->op == POP_RET || in->op == POP_EXIT)
        return 1;
    if (in->op == POP_BRA && !in->pred[0])
        return 1;
    return 0;
}

static void lg_succ(ptx_func_t *fn, size_t bi, size_t *out, size_t *nout) {
    *nout = 0;
    ptx_block_t *blk = &fn->blocks[bi];
    for (size_t k = 0; k < blk->count; k++) {
        ptx_instr_t *in = &blk->instrs[k];
        if (in->op == POP_BRA || in->op == POP_CBRA) {
            int u = 0;
            size_t t = lg_resolve(fn, in->target, &u);
            if (u) {
                size_t d = 0;
                while (d < *nout && out[d] != t)
                    d++;
                if (d == *nout && *nout < 8)
                    out[(*nout)++] = t;
            }
        }
    }
    if (blk->count > 0 && !lg_is_terminal(&blk->instrs[blk->count - 1])) {
        if (bi + 1 < fn->nblocks) {
            size_t t = bi + 1, d = 0;
            while (d < *nout && out[d] != t)
                d++;
            if (d == *nout && *nout < 8)
                out[(*nout)++] = t;
        }
    }
}

static void lg_walk(ptx_func_t *fn, size_t start, char *in_loop, char *seen,
                    int fwd, size_t avoid) {
    // fwd: successors; else predecessors. Stays within in_loop, avoids
    // the avoid block (SIZE_MAX = none).
    size_t n = fn->nblocks;
    size_t *stack = malloc(sizeof(*stack) * (n ? n : 1));
    if (!stack)
        return;
    size_t top = 0;
    stack[top++] = start;
    seen[start] = 1;
    while (top > 0) {
        size_t cur = stack[--top];
        if (fwd) {
            size_t s[8], ns = 0;
            lg_succ(fn, cur, s, &ns);
            for (size_t i = 0; i < ns; i++) {
                if (s[i] == avoid || !in_loop[s[i]] || seen[s[i]])
                    continue;
                seen[s[i]] = 1;
                if (top < n)
                    stack[top++] = s[i];
            }
        } else {
            for (size_t p = 0; p < n; p++) {
                if (p == avoid || !in_loop[p] || seen[p])
                    continue;
                size_t s[8], ns = 0;
                lg_succ(fn, p, s, &ns);
                for (size_t i = 0; i < ns; i++) {
                    if (s[i] == cur) {
                        seen[p] = 1;
                        if (top < n)
                            stack[top++] = p;
                        break;
                    }
                }
            }
        }
    }
    free(stack);
}

typedef struct {
    char name[64];
    size_t parent;
} lg_alias_t;

static size_t lg_find(lg_alias_t *t, size_t n, const char *name) {
    size_t id = n;
    for (size_t i = 0; i < n; i++) {
        if (strcmp(t[i].name, name) == 0) {
            id = i;
            break;
        }
    }
    if (id == n)
        return (size_t)-1;
    while (t[id].parent != id)
        id = t[id].parent = t[t[id].parent].parent;
    return id;
}

static size_t lg_intern(lg_alias_t *t, size_t *n, size_t cap, const char *name) {
    size_t id = lg_find(t, *n, name);
    if (id != (size_t)-1)
        return id;
    if (*n >= cap)
        return (size_t)-1;
    strncpy(t[*n].name, name, sizeof(t[*n].name) - 1);
    t[*n].name[sizeof(t[*n].name) - 1] = '\0';
    t[*n].parent = *n;
    return (*n)++;
}

static void latch_guard_fwd(ptx_func_t *fn) {
    size_t n = fn->nblocks;
    if (n == 0 || n > 4096)
        return;
    for (size_t i = 0; i < n; i++)
        for (size_t k = 0; k < fn->blocks[i].count; k++)
            if (fn->blocks[i].instrs[k].op == POP_BRX)
                return; // indirect control flow: stay out
    for (size_t hb = 0; hb < n; hb++) {
        ptx_block_t *hblk = &fn->blocks[hb];
        for (size_t bpos = 0; bpos < hblk->count; bpos++) {
            ptx_instr_t *br = &hblk->instrs[bpos];
            if (br->op != POP_CBRA || !br->pred[0] || br->pred_neg)
                continue;
            int tu = 0;
            size_t tgt = lg_resolve(fn, br->target, &tu);
            if (!tu || tgt <= hb || tgt >= n)
                continue; // v1 owns backward/self latches
            // Plain integer NE latch in this block before the branch.
            ptx_instr_t *sp = NULL;
            int ambiguous = 0;
            for (size_t k = 0; k < bpos; k++) {
                ptx_instr_t *in = &hblk->instrs[k];
                if (in->op == POP_SETP && strcmp(in->dst, br->pred) == 0) {
                    if (sp)
                        ambiguous = 1;
                    sp = in;
                }
            }
            if (ambiguous || !sp || sp->cmp != PCM_NE || sp->bop != 0 ||
                sp->dst2[0] || sp->pred[0] || sp->nsrc != 2)
                continue;
            if (sp->ty != PTY_U32 && sp->ty != PTY_U64)
                continue;
            if (sp->src[0].kind != OPD_REG || sp->src[1].kind != OPD_REG)
                continue;
            // Latch predicate used nowhere else in the function.
            int uses = 0;
            for (size_t ob = 0; ob < n && uses == 0; ob++) {
                ptx_block_t *o = &fn->blocks[ob];
                for (size_t k = 0; k < o->count; k++) {
                    ptx_instr_t *in = &o->instrs[k];
                    if (in == br)
                        continue;
                    if (in->pred[0] && strcmp(in->pred, br->pred) == 0)
                        uses++;
                    for (int s = 0; s < in->nsrc && uses == 0; s++)
                        if ((in->src[s].kind == OPD_REG ||
                             in->src[s].kind == OPD_REGOFF) &&
                            strcmp(in->src[s].reg, br->pred) == 0)
                            uses++;
                    if (uses == 0 && in->call_args)
                        for (int s = 0; s < in->nsrc; s++)
                            if (in->call_args[s].kind == OPD_REG &&
                                strcmp(in->call_args[s].reg, br->pred) == 0)
                                uses++;
                    if (uses)
                        break;
                }
            }
            if (uses != 0)
                continue;
            for (int order = 0; order < 2; order++) {
                const char *cnm = sp->src[order].reg;
                const char *enm = sp->src[order ^ 1].reg;
                // Back-edges into the latch block.
                for (size_t lb = hb; lb < n; lb++) {
                    size_t s[8], ns = 0;
                    lg_succ(fn, lb, s, &ns);
                    int is_back = 0;
                    for (size_t i = 0; i < ns; i++)
                        if (s[i] == hb)
                            is_back = 1;
                    if (!is_back)
                        continue;
                    // Natural loop of lb -> hb, restricted to blocks the
                    // header can actually reach (excludes the preheader
                    // chain and outer code that merely fall into the loop).
                    char *in_loop = calloc(n, 1);
                    char *seen = calloc(n, 1);
                    if (!in_loop || !seen) {
                        free(in_loop);
                        free(seen);
                        continue;
                    }
                    for (size_t i = 0; i < n; i++)
                        in_loop[i] = 1;
                    memset(seen, 0, n);
                    lg_walk(fn, lb, in_loop, seen, 0, hb);
                    char *fwdh = calloc(n, 1);
                    if (!fwdh) {
                        free(in_loop);
                        free(seen);
                        continue;
                    }
                    memset(seen, 0, n);
                    // avoid=n: no block index equals n, so nothing avoided.
                    lg_walk(fn, hb, in_loop, fwdh, 1, n);
                    char *loop = calloc(n, 1);
                    if (!loop) {
                        free(in_loop);
                        free(seen);
                        free(fwdh);
                        continue;
                    }
                    for (size_t i = 0; i < n; i++)
                        loop[i] = (fwdh[i] && (seen[i] || i == hb));
                    free(fwdh);
                    if (!loop[tgt]) {
                        free(in_loop);
                        free(seen);
                        free(loop);
                        continue;
                    }
                    // Reachability inside the loop.
                    memset(seen, 0, n);
                    lg_walk(fn, tgt, loop, seen, 1, hb);
                    char *on_cycle = calloc(n, 1);
                    if (!on_cycle) {
                        free(in_loop);
                        free(seen);
                        free(loop);
                        continue;
                    }
                    for (size_t i = 0; i < n; i++)
                        on_cycle[i] = seen[i];
                    // F-start: header successors other than the latch target.
                    size_t hs[8], hns = 0;
                    lg_succ(fn, hb, hs, &hns);
                    memset(seen, 0, n);
                    for (size_t i = 0; i < hns; i++) {
                        if (hs[i] == tgt)
                            continue;
                        if (!loop[hs[i]] || hs[i] == hb || seen[hs[i]])
                            continue;
                        lg_walk(fn, hs[i], loop, seen, 1, hb);
                    }
                    char *first_only = calloc(n, 1);
                    if (!first_only) {
                        free(in_loop);
                        free(seen);
                        free(loop);
                        free(on_cycle);
                        continue;
                    }
                    for (size_t i = 0; i < n; i++)
                        first_only[i] = (seen[i] && !on_cycle[i]);
                    // Mov-copy aliases over the loop.
                    lg_alias_t *at = malloc(sizeof(*at) * 2048);
                    size_t an = 0;
                    int oom = (at == NULL);
                    for (size_t b2 = 0; b2 < n && !oom; b2++) {
                        if (!loop[b2])
                            continue;
                        ptx_block_t *bl = &fn->blocks[b2];
                        for (size_t k = 0; k < bl->count; k++) {
                            ptx_instr_t *in = &bl->instrs[k];
                            if (in->op == POP_MOV && !in->pred[0] &&
                                in->nsrc >= 1 &&
                                in->src[0].kind == OPD_REG && in->dst[0]) {
                                size_t a = lg_intern(at, &an, 2048, in->dst);
                                size_t b3 = lg_intern(at, &an, 2048,
                                                      in->src[0].reg);
                                if (a == (size_t)-1 || b3 == (size_t)-1)
                                    oom = 1;
                                else {
                                    size_t ra = lg_find(at, an,
                                                        at[a].name);
                                    size_t rb = lg_find(at, an,
                                                        at[b3].name);
                                    if (ra != rb)
                                        at[ra].parent = rb;
                                }
                            }
                        }
                    }
                    // Classify counter/limit definitions.
                    size_t cc = lg_intern(at, &an, 2048, cnm);
                    size_t ee = lg_intern(at, &an, 2048, enm);
                    int ok = !oom && cc != (size_t)-1 && ee != (size_t)-1;
                    int adds = 0;
                    if (ok) {
                        size_t rc = lg_find(at, an, at[cc].name);
                        size_t re = lg_find(at, an, at[ee].name);
                        for (size_t b2 = 0; b2 < n && ok; b2++) {
                            if (!loop[b2])
                                continue;
                            ptx_block_t *bl = &fn->blocks[b2];
                            for (size_t k = 0; k < bl->count; k++) {
                                ptx_instr_t *in = &bl->instrs[k];
                                if (in == sp || in == br)
                                    continue;
                                // Defined names on this instruction.
                                const char *defs[16];
                                int nd = 0;
                                if (in->dst[0])
                                    defs[nd++] = in->dst;
                                if (in->dst2[0])
                                    defs[nd++] = in->dst2;
                                for (int r = 0; r < in->nrets && nd < 16; r++)
                                    defs[nd++] = in->rets[r];
                                if ((in->op == POP_TEX || in->op == POP_TLD4 ||
                                     in->op == POP_TXQ) &&
                                    in->tex_dst[0][0]) {
                                    for (int t2 = 0;
                                         t2 < 4 && nd < 16; t2++)
                                        if (in->tex_dst[t2][0])
                                            defs[nd++] = in->tex_dst[t2];
                                }
                                if (in->op == POP_TEX && in->tex_dstp[0] &&
                                    nd < 16)
                                    defs[nd++] = in->tex_dstp;
                                for (int t2 = 0; t2 < 4 && nd < 16; t2++)
                                    if (in->surf_dst[t2][0])
                                        defs[nd++] = in->surf_dst[t2];
                                for (int d = 0; d < nd && ok; d++) {
                                    size_t rd = lg_find(at, an, defs[d]);
                                    int is_c = (rd != (size_t)-1 &&
                                                rd == rc);
                                    int is_e = (rd != (size_t)-1 &&
                                                rd == re);
                                    // Untracked name equal to c/e textually
                                    // but never aliased: compare directly.
                                    if (rd == (size_t)-1) {
                                        if (strcmp(defs[d], cnm) == 0)
                                            is_c = 1;
                                        if (strcmp(defs[d], enm) == 0)
                                            is_e = 1;
                                    }
                                    if (!is_c && !is_e)
                                        continue;
                                    if (first_only[b2])
                                        continue; // entry path: free
                                    if (is_e) {
                                        ok = 0; // limit must not move
                                        break;  // on the cycle
                                    }
                                    // Counter def on the cycle: copies, or
                                    // the single +1 step.
                                    if (in->op == POP_MOV && !in->pred[0] &&
                                        in->nsrc >= 1 &&
                                        in->src[0].kind == OPD_REG) {
                                        size_t rs = lg_find(
                                            at, an, in->src[0].reg);
                                        if (rs == rc)
                                            continue;
                                    }
                                    if (in->op == POP_ADD &&
                                        !in->pred[0] && in->ty == sp->ty &&
                                        in->nsrc == 2) {
                                        int self = -1, imm = -1;
                                        for (int qq = 0; qq < 2; qq++) {
                                            if (in->src[qq].kind ==
                                                    OPD_REG &&
                                                lg_find(at, an,
                                                        in->src[qq]
                                                            .reg) == rc)
                                                self = qq;
                                            else if (in->src[qq].kind ==
                                                         OPD_IMM &&
                                                     !in->src[qq]
                                                          .imm_is_float &&
                                                     in->src[qq].imm == 1)
                                                imm = qq;
                                        }
                                        size_t rd2 = lg_find(at, an,
                                                             in->dst);
                                        if (self >= 0 && imm >= 0 &&
                                            self != imm && rd2 == rc) {
                                            adds++;
                                            continue;
                                        }
                                    }
                                    ok = 0;
                                    break;
                                }
                            }
                        }
                    }
                    free(at);
                    free(in_loop);
                    free(seen);
                    free(loop);
                    free(on_cycle);
                    free(first_only);
                    if (!ok || adds != 1)
                        continue;
                    sp->cmp = (order == 0 ? PCM_LT : PCM_GT);
                    if (getenv("ACUDA_TRACE"))
                        fprintf(stderr,
                                "acuda-latch-guard: ne->%c%c fwd loop "
                                "%s..%s line %d\n",
                                order == 0 ? 'l' : 'g',
                                sp->sign ? 's' : 'u',
                                fn->blocks[hb].label[0]
                                    ? fn->blocks[hb].label
                                    : "(entry)",
                                hblk->label[0] ? hblk->label
                                               : "(fallthrough)",
                                sp->line);
                    order = 2; // done with this latch
                    lb = n;    // and its back-edges
                }
            }
        }
    }
}

// Record a `.reg`-form signature name: visible to the call-inline
// pass (rparams) and to emission (plain regdecl). A same-name
// declaration with a different type is an honest error.
int sig_add_reg(parser_t *p, ptx_func_t *fn, const char *name,
                       ptx_ty_t ty, int is_retval, int line) {
    for (size_t i = 0; i < fn->nregs; i++) {
        if (strcmp(fn->regs[i].name, name) == 0 &&
            fn->regs[i].ty != ty) {
            perr(p, line, "signature register type mismatch");
            return -1;
        }
    }
    {
        int known = 0;
        for (size_t i = 0; i < fn->nregs; i++)
            if (strcmp(fn->regs[i].name, name) == 0)
                known = 1;
        if (!known && func_add_reg(fn, name, ty) != 0) {
            perr(p, line, "out of memory");
            return -1;
        }
    }
    for (size_t i = 0; i < fn->nrparams; i++) {
        if (strcmp(fn->rparams[i].name, name) == 0) {
            perr(p, line, "duplicate signature name");
            return -1;
        }
    }
    if (fn->nrparams + 1 > fn->rparams_cap) {        size_t ncap = fn->rparams_cap ? fn->rparams_cap * 2 : 4;
        ptx_param_t *np = malloc(sizeof(*np) * ncap);
        if (!np) {
            perr(p, line, "out of memory");
            return -1;
        }
        memcpy(np, fn->rparams, sizeof(*np) * fn->nrparams);
        free(fn->rparams);
        fn->rparams = np;
        fn->rparams_cap = ncap;
    }
    if (copy_name(fn->rparams[fn->nrparams].name,
                  sizeof(fn->rparams[fn->nrparams].name), name, p,
                  line) != 0)
        return -1;
    fn->rparams[fn->nrparams].ty = ty;
    fn->rparams[fn->nrparams].is_retval = is_retval;
    fn->rparams[fn->nrparams].is_internal = 0;
    fn->nrparams++;
    return 0;
}

// Register a `.reg`-form vector signature item: the vector name (for
// body `v.x` refs) plus one scalar rparam per component (scalarized
// boundary — the inline core only ever sees component names).
int sig_add_reg_vec(parser_t *p, ptx_func_t *fn, const char *base,
                           int n, ptx_ty_t elem, int is_retval, int line) {
    if (vec_find(fn, base)) {
        perr(p, line, "duplicate signature name");
        return -1;
    }
    if (func_add_vec(fn, base, n, elem) != 0) {
        perr(p, line, "out of memory");
        return -1;
    }
    size_t lb = strlen(base);
    if (lb + 2 >= 64) {
        perr(p, line, "name too long");
        return -1;
    }
    for (int i = 0; i < n; i++) {
        char nm[64];
        memcpy(nm, base, lb);
        nm[lb] = '_';
        nm[lb + 1] = "xyzw"[i];
        nm[lb + 2] = '\0';
        if (sig_add_reg(p, fn, nm, elem, is_retval, line) != 0)
            return -1;
    }
    return 0;
}

static int ext_declared(const ptx_unit_t *unit, const char *name) {
    for (size_t i = 0; i < unit->nextfuncs; i++)
        if (strcmp(unit->extfuncs[i].name, name) == 0)
            return 1;
    return 0;
}

int parse_entry(parser_t *p, ptx_unit_t *unit, int is_entry) {
    // ".entry"/".func" already consumed.
    ptx_tok_t t;
    ptx_func_t fn;
    memset(&fn, 0, sizeof(fn));
    fn.is_entry = is_entry;
    p->fn = &fn; // visible to operand parsing for .reg lookup
    if (next(p, &t) != 0)
        return -1;
    if (t.kind == T_PUNCT && t.punct == '(') {
        // .func return slot(s): "(.param .b32 func_retval0)" (numba),
        // "(.param.u64 output)" (ZLUDA combined), or ".reg" results
        // "(.reg .u32 a, .reg .pred b)" (multi-return form). Only
        // `.reg` takes lists; .param singles keep the retval-param
        // record so st.param writes drop.
        int rkind = 0; // 1 = .param, 2 = .reg
        for (;;) {
            if (next(p, &t) != 0)
                return -1;
            int is_p = t.kind == T_IDENT &&
                       (strcmp(t.text, ".param") == 0 ||
                        strncmp(t.text, ".param.", 7) == 0);
            int is_r = !is_p && t.kind == T_IDENT &&
                       (strcmp(t.text, ".reg") == 0 ||
                        strncmp(t.text, ".reg.", 5) == 0);
            if (!is_p && !is_r) {
                perr(p, t.line, "expected .param/.reg in return slot");
                return -1;
            }
            if (rkind && rkind != (is_p ? 1 : 2)) {
                perr(p, t.line, "mixed .param/.reg return slots");
                return -1;
            }
            rkind = is_p ? 1 : 2;
            ptx_ty_t rty = PTY_U32;
            int rwide = 0;
            if (is_p && strcmp(t.text, ".param") != 0) {
                // Combined ".param.<type>".
                if (map_ty(t.text + 7, &rty) != 0 || rty == PTY_PRED) {
                    perr(p, t.line, "bad return slot type");
                    return -1;
                }
                if (next(p, &t) != 0)
                    return -1;
            } else if (is_p) {
                if (next(p, &t) != 0)
                    return -1;
                // Optional .align
                if (t.kind == T_IDENT && strcmp(t.text, ".align") == 0) {
                    ptx_tok_t av;
                    if (next(p, &av) != 0 || av.kind != T_INT) {
                        perr(p, t.line, "expected alignment");
                        return -1;
                    }
                    if (next(p, &t) != 0)
                        return -1;
                }
                // .b8 name[M] aggregate return
                if (t.kind == T_IDENT && strcmp(t.text, ".b8") == 0) {
                    ptx_tok_t nm, lb, sz, rb;
                    if (next(p, &nm) != 0 || nm.kind != T_IDENT) {
                        perr(p, t.line, "expected return slot name");
                        return -1;
                    }
                    if (next(p, &lb) != 0 || lb.kind != T_PUNCT || lb.punct != '[') {
                        perr(p, lb.line, "expected [M] after .b8 return");
                        return -1;
                    }
                    if (next(p, &sz) != 0 || sz.kind != T_INT || sz.imm <= 0 ||
                        (sz.imm + 7) / 8 > 4096) {
                        perr(p, sz.line, "bad .b8 return size");
                        return -1;
                    }
                    if (next(p, &rb) != 0 || rb.kind != T_PUNCT || rb.punct != ']') {
                        perr(p, rb.line, "expected ]");
                        return -1;
                    }
                    rty = PTY_U32;
                    rwide = (int)((sz.imm + 7) / 8);
                    t = nm; /* slot name */
                    goto have_ret_slot;
                }
                if (t.kind != T_IDENT) {
                    perr(p, t.line, "expected return slot type");
                    return -1;
                }
                if (map_ty(t.text, &rty) != 0 || rty == PTY_PRED) {
                    perr(p, t.line, "bad return slot type");
                    return -1;
                }
                if (next(p, &t) != 0)
                    return -1;
            } else if (strcmp(t.text, ".reg") != 0) {
                // Combined ".reg.<type>" (vector `.reg.vN.T` expands
                // to component rparams below).
                int vn = 0;
                ptx_ty_t velem = PTY_U32;
                if (vec_type(t.text + 4, &vn, &velem) == 0) {
                    if (next(p, &t) != 0)
                        return -1;
                    if (t.kind != T_IDENT) {
                        perr(p, t.line, "expected return slot name");
                        return -1;
                    }
                    if (sig_add_reg_vec(p, &fn, t.text, vn, velem, 1,
                                        t.line) != 0)
                        return -1;
                    if (at_punct(p, ',')) {
                        continue;
                    }
                    if (expect_punct(p, ')', "after return slot") != 0)
                        return -1;
                    break;
                }
                if (map_ty(t.text + 5, &rty) != 0) {
                    perr(p, t.line, "bad return register type");
                    return -1;
                }
                if (next(p, &t) != 0)
                    return -1;
            } else {
                if (next(p, &t) != 0)
                    return -1;
                int vn = 0;
                ptx_ty_t velem = PTY_U32;
                int vr = (t.kind == T_IDENT) ?
                         vec_type(t.text, &vn, &velem) : -1;
                if (vr == 1) {
                    // Spaced vector `.reg .vN .T name`.
                    if (next(p, &t) != 0)
                        return -1;
                    if (t.kind != T_IDENT ||
                        vec_elem_ty(t.text, &velem) != 0) {
                        perr(p, t.line, "bad return register type");
                        return -1;
                    }
                    if (next(p, &t) != 0)
                        return -1;
                    if (t.kind != T_IDENT) {
                        perr(p, t.line, "expected return slot name");
                        return -1;
                    }
                    if (sig_add_reg_vec(p, &fn, t.text, vn, velem, 1,
                                        t.line) != 0)
                        return -1;
                    if (at_punct(p, ',')) {
                        continue;
                    }
                    if (expect_punct(p, ')', "after return slot") != 0)
                        return -1;
                    break;
                }
                if (t.kind != T_IDENT || map_ty(t.text, &rty) != 0) {
                    perr(p, t.line, "bad return register type");
                    return -1;
                }
                if (next(p, &t) != 0)
                    return -1;
            }
        have_ret_slot:
            if (t.kind != T_IDENT) {
                perr(p, t.line, "expected return slot name");
                return -1;
            }
            if (rkind == 1) {
                // Stash as retval param (no arg slot; see compile.c).
                fn.params = malloc(sizeof(*fn.params));
                if (!fn.params) {
                    perr(p, t.line, "out of memory");
                    return -1;
                }
                memset(fn.params, 0, sizeof(*fn.params));
                fn.params_cap = fn.nparams = 1;
                if (copy_name(fn.params[0].name,
                              sizeof(fn.params[0].name), t.text, p,
                              t.line) != 0)
                    return -1;
                fn.params[0].ty = rty;
                fn.params[0].is_retval = 1;
                fn.params[0].is_internal = 0;
                fn.params[0].wide_words = rwide;
            } else {
                if (sig_add_reg(p, &fn, t.text, rty, 1, t.line) != 0)
                    return -1;
            }
            if (at_punct(p, ',')) {
                if (rkind == 1) {
                    perr(p, t.line, "multiple .param return slots unsupported");
                    return -1;
                }
                continue;
            }
            if (expect_punct(p, ')', "after return slot") != 0)
                return -1;
            break;
        }
        if (next(p, &t) != 0)
            return -1;
    }
    if (t.kind != T_IDENT) {
        perr(p, t.line, "expected entry name");
        return -1;
    }
    if (copy_name(fn.name, sizeof(fn.name), t.text, p, t.line) != 0)
        return -1;
    if (expect_punct(p, '(', "after entry name") != 0)
        return -1;
    // Parameter list (".param .u64 x" numba spacing or ".param.u64
    // x" ZLUDA combined token).
    if (!at_punct(p, ')')) {
        for (;;) {
            if (next(p, &t) != 0)
                return -1;
            if (t.kind == T_IDENT &&
                (strcmp(t.text, ".reg") == 0 ||
                 strncmp(t.text, ".reg.", 5) == 0)) {
                // `.reg`-form parameter (call-only ABI): recorded
                // for the inline pass, never a launch argument
                // (emitting such a function directly is an error).
                // `.reg` return slots may follow `.param` arguments for the
                // mixed multi-return ABI; all call arguments stay `.param`.
                ptx_ty_t rty;
                if (strcmp(t.text, ".reg") != 0) {
                    int vn = 0;
                    ptx_ty_t velem = PTY_U32;
                    if (vec_type(t.text + 4, &vn, &velem) == 0) {
                        // Combined vector `.reg.vN.T name`.
                        if (next(p, &t) != 0)
                            return -1;
                        if (t.kind != T_IDENT) {
                            perr(p, t.line, "expected param name");
                            return -1;
                        }
                        if (sig_add_reg_vec(p, &fn, t.text, vn, velem,
                                            0, t.line) != 0)
                            return -1;
                        if (at_punct(p, ',')) {
                            continue;
                        }
                        if (at_punct(p, ')')) {
                            break;
                        }
                        perr(p, t.line, "expected ',' or ')' in param list");
                        return -1;
                    }
                    if (map_ty(t.text + 5, &rty) != 0) {
                        perr(p, t.line, "bad signature register type");
                        return -1;
                    }
                    if (next(p, &t) != 0)
                        return -1;
                } else {
                    if (next(p, &t) != 0)
                        return -1;
                    int vn = 0;
                    ptx_ty_t velem = PTY_U32;
                    int vr = (t.kind == T_IDENT) ?
                             vec_type(t.text, &vn, &velem) : -1;
                    if (vr == 1) {
                        // Spaced vector `.reg .vN .T name`.
                        if (next(p, &t) != 0)
                            return -1;
                        if (t.kind != T_IDENT ||
                            vec_elem_ty(t.text, &velem) != 0) {
                            perr(p, t.line, "bad signature register type");
                            return -1;
                        }
                        if (next(p, &t) != 0)
                            return -1;
                        if (t.kind != T_IDENT) {
                            perr(p, t.line, "expected param name");
                            return -1;
                        }
                        if (sig_add_reg_vec(p, &fn, t.text, vn, velem,
                                            0, t.line) != 0)
                            return -1;
                        if (at_punct(p, ',')) {
                            continue;
                        }
                        if (at_punct(p, ')')) {
                            break;
                        }
                        perr(p, t.line, "expected ',' or ')' in param list");
                        return -1;
                    }
                    if (t.kind != T_IDENT || map_ty(t.text, &rty) != 0) {
                        perr(p, t.line, "bad signature register type");
                        return -1;
                    }
                    if (next(p, &t) != 0)
                        return -1;
                }
                if (t.kind != T_IDENT) {
                    perr(p, t.line, "expected param name");
                    return -1;
                }
                if (sig_add_reg(p, &fn, t.text, rty, 0, t.line) != 0)
                    return -1;
                if (at_punct(p, ',')) {
                    continue;
                }
                if (at_punct(p, ')')) {
                    break;
                }
                perr(p, t.line, "expected ',' or ')' in param list");
                return -1;
            }
            // A `.param` argument list after `.reg` return slots is the
            // mixed multi-return ABI.
            ptx_ty_t ty;
            int is_tex = 0;
            int wide_words = 0;
            int wide_bytes = 0;
            int b8_words = 0;
            int b8_bytes = 0;
            int name_ready = 0;
            if (t.kind == T_IDENT && strncmp(t.text, ".param.", 7) == 0) {
                const char *pt = t.text + 7;
                if (strcmp(pt, "texref") == 0 ||
                    strcmp(pt, "samplerref") == 0 ||
                    strcmp(pt, "surfref") == 0) {
                    ty = PTY_U64;
                    is_tex = strcmp(pt, "texref") == 0 ? 1
                             : (strcmp(pt, "samplerref") == 0 ? 2 : 3);
                } else if (strncmp(pt, "u64x", 4) == 0) {
                    int n = atoi(pt + 4);
                    if (n < 2 || n > 4096) {
                        perr(p, t.line, "bad .u64xN parameter type");
                        return -1;
                    }
                    ty = PTY_U64;
                    wide_words = n;
                    wide_bytes = n * 8;
                } else if (map_ty(pt, &ty) != 0) {
                    perr(p, t.line, "unsupported entry param type (want .u32/.s32/.u64/.s64/.f32)");
                    return -1;
                }
            } else {
                if (t.kind != T_IDENT || strcmp(t.text, ".param") != 0) {
                    perr(p, t.line, "expected .param in entry signature");
                    return -1;
                }
                if (next(p, &t) != 0)
                    return -1;
                // Optional alignment (old-toolkit byte arrays spell
                // `.param .align 4 .b8 name[8]`).
                if (t.kind == T_IDENT && strcmp(t.text, ".align") == 0) {
                    if (next(p, &t) != 0)
                        return -1;
                    if (t.kind != T_INT) {
                        perr(p, t.line, "expected alignment value");
                        return -1;
                    }
                    if (next(p, &t) != 0)
                        return -1;
                }
                // Raw byte blob (`.param .b8 name[M]`, CUDA 4.x style):
                // model as wide_words 8-byte words; word-granular
                // ld.param.u32/f32 loads are already supported.
                if (t.kind == T_IDENT && strcmp(t.text, ".b8") == 0) {
                    ptx_tok_t nm, lb, sz, rb;
                    if (next(p, &nm) != 0)
                        return -1;
                    if (nm.kind != T_IDENT) {
                        perr(p, nm.line, "expected param name");
                        return -1;
                    }
                    if (next(p, &lb) != 0)
                        return -1;
                    if (lb.kind != T_PUNCT || lb.punct != '[') {
                        perr(p, lb.line, "expected [M] after .b8 param");
                        return -1;
                    }
                    if (next(p, &sz) != 0)
                        return -1;
                    if (sz.kind != T_INT || sz.imm <= 0 ||
                        (sz.imm + 7) / 8 > 4096) {
                        perr(p, sz.line, "bad .b8 array size");
                        return -1;
                    }
                    if (next(p, &rb) != 0)
                        return -1;
                    if (rb.kind != T_PUNCT || rb.punct != ']') {
                        perr(p, rb.line, "expected ] after .b8 size");
                        return -1;
                    }
                    ty = PTY_U32;
                    b8_words = (int)((sz.imm + 7) / 8);
                    b8_bytes = (int)sz.imm;
                    t = nm;
                    name_ready = 1;
                }
                if (b8_words) {
                    wide_words = b8_words;
                    wide_bytes = b8_bytes;
                } else if (t.kind == T_IDENT &&
                    (strcmp(t.text, ".texref") == 0 ||
                     strcmp(t.text, ".samplerref") == 0 ||
                     strcmp(t.text, ".surfref") == 0)) {
                    ty = PTY_U64;
                    is_tex = strcmp(t.text, ".texref") == 0 ? 1
                             : (strcmp(t.text, ".samplerref") == 0 ? 2 : 3);
                } else if (t.kind == T_IDENT &&
                           strncmp(t.text, ".u64x", 5) == 0) {
                    int n = atoi(t.text + 5);
                    if (n < 2 || n > 4096) {
                        perr(p, t.line, "bad .u64xN parameter type");
                        return -1;
                    }
                    ty = PTY_U64;
                    wide_words = n;
                    wide_bytes = n * 8;
                } else if (t.kind != T_IDENT || map_ty(t.text, &ty) != 0) {
                    perr(p, t.line, "unsupported entry param type (want .u32/.s32/.u64/.s64/.f32)");
                    return -1;
                }
            }
            if (ty == PTY_PRED || ty == PTY_F16 || ty == PTY_BF16 ||
                ty == PTY_F16X2 || ty == PTY_BF16X2) {
                perr(p, t.line, "unsupported entry param type");
                return -1;
            }
            // Spaced form leaves the type token in t; combined form
            // keeps it as the ".param.<type>" token itself.
            const char *typetok = t.text;
            if (t.kind == T_IDENT && strncmp(t.text, ".param.", 7) == 0)
                typetok = t.text + 7;
            if (is_entry && typetok[0] == '.' &&
                (typetok[1] == 'b' || typetok[1] == 's') &&
                strcmp(typetok, ".b64") != 0 &&
                strcmp(typetok, ".b32") != 0 &&
                strcmp(typetok, ".b16") != 0 &&
                strcmp(typetok, ".b8") != 0 &&
                strcmp(typetok, ".s32") != 0 &&
                strcmp(typetok, ".s64") != 0 &&
                strcmp(typetok, ".s16") != 0 &&
                strcmp(typetok, ".s8") != 0 &&
                strcmp(typetok, ".samplerref") != 0 &&
                strcmp(typetok, ".surfref") != 0) {
                // allow .f64
                if (strcmp(typetok, ".f64") != 0) {
                    perr(p, t.line, "entry params need scalar type (no unsupported .b aliases)");
                    return -1;
                }
            }
            if (!name_ready && next(p, &t) != 0)
                return -1;
            if (!name_ready && t.kind != T_IDENT) {
                perr(p, t.line, "expected param name");
                return -1;
            }
            if (fn.nparams + 1 > fn.params_cap) {
                size_t ncap = fn.params_cap ? fn.params_cap * 2 : 4;
                ptx_param_t *np = malloc(sizeof(*np) * ncap);
                if (!np) {
                    perr(p, t.line, "out of memory");
                    return -1;
                }
                memcpy(np, fn.params, sizeof(*np) * fn.nparams);
                free(fn.params);
                fn.params = np;
                fn.params_cap = ncap;
            }
            if (copy_name(fn.params[fn.nparams].name, sizeof(fn.params[fn.nparams].name), t.text, p, t.line) != 0)
                return -1;
            fn.params[fn.nparams].ty = ty;
            fn.params[fn.nparams].is_retval =
                strncmp(fn.params[fn.nparams].name, "func_retval",
                        strlen("func_retval")) == 0;
            fn.params[fn.nparams].is_internal = 0;
            fn.params[fn.nparams].is_tex = is_tex;
            fn.params[fn.nparams].is_hidden_texref = 0;
            fn.params[fn.nparams].wide_words = wide_words;
            fn.params[fn.nparams].wide_bytes = wide_bytes;
            fn.nparams++;
            // Array-typed parameter (`.param .b32 p[]`, ZLUDA's
            // param_func_array_0): the parameter holds the address of
            // the caller's array. The unsized suffix only marks
            // addressability; the slot itself is pointer-sized.
            if (at_punct(p, '[')) {
                if (expect_punct(p, ']', "in array param") != 0)
                    return -1;
                if (ty == PTY_U32)
                    fn.params[fn.nparams - 1].ty = PTY_U64;
            }
            if (at_punct(p, ',')) {
                // Optional alignment: ".align 8" — parse and skip.
                ptx_tok_t u;
                if (next(p, &u) != 0)
                    return -1;
                if (u.kind == T_IDENT && strcmp(u.text, ".align") == 0) {
                    if (next(p, &u) != 0)
                        return -1;
                    if (u.kind != T_INT) {
                        perr(p, u.line, "expected alignment value");
                        return -1;
                    }
                    if (at_punct(p, ',')) {
                        continue;
                    }
                    if (at_punct(p, ')'))
                        break;
                    perr(p, u.line, "expected ',' or ')' in param list");
                    return -1;
                }
                ptx_unget(&p->lx, &u);
                continue;
            }
            if (at_punct(p, ')'))
                break;
            perr(p, t.line, "expected ',' or ')' in param list");
            return -1;
        }
    }
    // Launch-tuning attributes (performance-only; the host launch
    // config wins — `.maxntid/.minnctapersm/.reqntid`, newline- or
    // semicolon-terminated, up to the body-open '{').
    for (;;) {
        ptx_tok_t u;
        if (next(p, &u) != 0)
            return -1;
        if (u.kind == T_IDENT && (strcmp(u.text, ".maxntid") == 0 ||
                                  strcmp(u.text, ".minnctapersm") == 0 ||
                                  strcmp(u.text, ".reqntid") == 0)) {
            for (;;) {
                if (next(p, &u) != 0)
                    return -1;
                if (u.kind == T_PUNCT && u.punct == ';')
                    break;
                if (u.kind == T_PUNCT && u.punct == '{') {
                    ptx_unget(&p->lx, &u);
                    goto attrs_done;
                }
                if (u.kind == T_EOF) {
                    perr(p, u.line, "unterminated attribute");
                    return -1;
                }
            }
            continue;
        }
        ptx_unget(&p->lx, &u);
        break;
    }
attrs_done:;
    {
        ptx_tok_t u;
        if (next(p, &u) != 0)
            return -1;
        if (u.kind == T_PUNCT && u.punct == ';') {
            // .func prototype (ZLUDA corpus): declaration only, not
            // added to the unit. Calls resolve against the later
            // definition (or fail honestly as undefined).
            free(fn.params);
            return 0;
        }
        ptx_unget(&p->lx, &u);
    }
    // Optional ".noreturn" attribute after the entry signature
    // (PTX 6.4 `.entry name( ) .noreturn`): a hint that the entry
    // body ends with control flow (trap/exit) rather than ret. Treated
    // as informational; semantics ride on the terminator.
    {
        ptx_tok_t v;
        int noreturn = 0;
        if (next(p, &v) == 0) {
            if (v.kind == T_IDENT && strcmp(v.text, ".noreturn") == 0)
                noreturn = 1;
            else
                ptx_unget(&p->lx, &v);
        }
        (void)noreturn;
        if (expect_punct(p, '{', "to open entry body") != 0)
            return -1;
    }
    size_t cur = new_block(&fn, p, NULL);
    if (cur == (size_t)-1)
        return -1;
    p->last_label[0] = '\0';
    // A predicated return cannot be a mid-block terminator, so such
    // functions are rewritten after parsing: every return branches to
    // one synthetic epilogue block. Functions without a predicated
    // return are left untouched (their natural final ret is what the
    // flow/CFG composers expect).
    int has_predicated_ret = 0;
    const char *const synthetic_ret_ph = "@acuda_ret";
    // PTX allows anonymous `{ ... }` blocks inside a body. They carry
    // no runtime semantics but do scope register declarations: an
    // inner re-declaration shadows the outer register (declare_scalar_reg).
    for (;;) {
        if (next(p, &t) != 0)
            return -1;
        if (t.kind == T_PUNCT && t.punct == '{') {
            p->depth++;
            continue;
        }
        if (t.kind == T_PUNCT && t.punct == '}') {
            if (p->depth > 0) {
                scope_pop(p);
                p->depth--;
                continue;
            }
            // Empty trailing block with no label and no instrs: drop it.
            if (fn.blocks[cur].count == 0 && fn.blocks[cur].label[0] == '\0' &&
                fn.nblocks > 1)
                fn.nblocks--;
            break;
        }
        if (t.kind != T_IDENT &&
            !(t.kind == T_PUNCT && t.punct == '@')) {
            perr(p, t.line, "expected directive, label or instruction");
            return -1;
        }
        if (t.kind == T_PUNCT) {
            // Predicated instruction "@[%]p bra ...".
            char pred[64] = "";
            int pred_neg = 0;
            char opbuf[512];
            if (next(p, &t) != 0)
                return -1;
            if (t.kind == T_PUNCT && t.punct == '!') {
                pred_neg = 1;
                if (next(p, &t) != 0)
                    return -1;
            }
            if (t.kind != T_IDENT || t.text[0] == '$') {
                perr(p, t.line, "expected predicate register after @");
                return -1;
            }
            if (copy_name(pred, sizeof(pred), t.text, p, t.line) != 0)
                return -1;
            if (next(p, &t) != 0)
                return -1;
            if (t.kind != T_IDENT) {
                perr(p, t.line, "expected opcode after predicate");
                return -1;
            }
            snprintf(opbuf, sizeof(opbuf), "%s", t.text);
            if (parse_instr(p, &fn, cur, pred, pred_neg, opbuf, t.line) != 0)
                return -1;
            ptx_block_t *pb = &fn.blocks[cur];
            if (pb->count == 0)
                continue; // nothing emitted (dropped decl): no terminator
            ptx_instr_t *plast = &pb->instrs[pb->count - 1];
            int pterm = plast->op;
            if (pterm == POP_RET || pterm == POP_EXIT) {
                // @p ret -> @p bra <epilogue>; the not-taken path
                // continues in the next block. The epilogue target is
                // resolved after the body is parsed.
                plast->op = POP_CBRA;
                snprintf(plast->target, sizeof(plast->target), "%s",
                         synthetic_ret_ph);
                has_predicated_ret = 1;
                cur = new_block(&fn, p, NULL);
                if (cur == (size_t)-1)
                    return -1;
            } else if (pterm == POP_BRA || pterm == POP_CBRA ||
                       pterm == POP_BRX) {
                cur = new_block(&fn, p, NULL);
                if (cur == (size_t)-1)
                    return -1;
            }
            continue;
        }
        if (strcmp(t.text, ".reg") == 0 ||
            strncmp(t.text, ".reg.", 5) == 0) {
            if (parse_reg_decl(p, &fn, t.text, t.line) != 0)
                return -1;
            continue;
        }
        if (strcmp(t.text, ".shared") == 0) {
            // Function-scope static `.shared` (ZLUDA corpus): same
            // per-CTA storage as module scope, registered unit-wide.
            if (parse_shared_decl(p, unit) != 0)
                return -1;
            continue;
        }
        if (strcmp(t.text, ".const") == 0) {
            // Function-scope `.const`: constants are module-wide in the
            // emitter (loads materialize as immediates).
            if (parse_const_decl(p, unit) != 0)
                return -1;
            continue;
        }
        if (strcmp(t.text, ".local") == 0) {
            // Function-scope `.local` (Private per-invocation storage).
            if (parse_local_decl(p, &fn) != 0)
                return -1;
            continue;
        }
        if (strcmp(t.text, ".global") == 0) {
            // Function-scope `.global` (used in some PTX corpora).
            if (parse_global_decl(p, unit) != 0)
                return -1;
            continue;
        }
        if (strcmp(t.text, ".shared") == 0) {
            // Function-scope `.shared`.
            if (parse_shared_decl(p, unit) != 0)
                return -1;
            continue;
        }
        if (strcmp(t.text, ".param") == 0 ||
            strncmp(t.text, ".param.", 7) == 0) {
            // In-function `.param` call slot (both spellings).
            if (parse_param_slot(p, &fn, t.text, t.line) != 0)
                return -1;
            continue;
        }
        if (strcmp(t.text, ".pragma") == 0) {
            // Optimization hints (numba emits .pragma "nounroll" on
            // remainder loops): no semantic content, skip to ';'.
            // String literals lex as T_STR so the quoted text skips
            // cleanly here; anywhere else it is a hard error.
            for (;;) {
                if (next(p, &t) != 0)
                    return -1;
                if (t.kind == T_PUNCT && t.punct == ';')
                    break;
                if (t.kind == T_EOF) {
                    perr(p, t.line, "unterminated .pragma");
                    return -1;
                }
            }
            continue;
        }
        if (strcmp(t.text, ".branchtargets") == 0) {
            int bi2 = fn.nbranchtargets++;
            if (bi2 >= 8) {
                perr(p, t.line, "too many .branchtargets lists");
                return -1;
            }
            if (p->last_label[0] == '\0') {
                perr(p, t.line, ".branchtargets needs a label");
                return -1;
            }
            memset(&fn.branchtargets[bi2], 0,
                   sizeof(fn.branchtargets[bi2]));
            if (copy_name(fn.branchtargets[bi2].label,
                          sizeof(fn.branchtargets[bi2].label),
                          p->last_label, p, t.line) != 0)
                return -1;
            for (;;) {
                if (next(p, &t) != 0)
                    return -1;
                if (t.kind == T_PUNCT && t.punct == ';')
                    break;
                if (t.kind == T_EOF) {
                    perr(p, t.line, "unterminated .branchtargets");
                    return -1;
                }
                if (t.kind == T_IDENT) {
                    if (fn.branchtargets[bi2].n >= 64) {
                        perr(p, t.line, "too many branch targets");
                        return -1;
                    }
                    if (copy_name(
                            fn.branchtargets[bi2].targets[
                                fn.branchtargets[bi2].n++],
                            64, t.text, p, t.line) != 0)
                        return -1;
                }
            }
            continue;
        }
        if (strcmp(t.text, ".calltargets") == 0) {
            int ci = fn.ncalltargets++;
            if (ci >= 8) {
                perr(p, t.line, "too many .calltargets lists");
                return -1;
            }
            if (p->last_label[0] == '\0') {
                perr(p, t.line, ".calltargets needs a label");
                return -1;
            }
            memset(&fn.calltargets[ci], 0, sizeof(fn.calltargets[ci]));
            if (copy_name(fn.calltargets[ci].label,
                          sizeof(fn.calltargets[ci].label),
                          p->last_label, p, t.line) != 0)
                return -1;
            for (;;) {
                if (next(p, &t) != 0)
                    return -1;
                if (t.kind == T_PUNCT && t.punct == ';')
                    break;
                if (t.kind == T_EOF) {
                    perr(p, t.line, "unterminated .calltargets");
                    return -1;
                }
                if (t.kind == T_IDENT && strcmp(t.text, ",") != 0) {
                    if (fn.calltargets[ci].n >= 8) {
                        perr(p, t.line, "too many call targets");
                        return -1;
                    }
                    if (copy_name(
                            fn.calltargets[ci].funcs[
                                fn.calltargets[ci].n++],
                            128, t.text, p, t.line) != 0)
                        return -1;
                }
            }
            continue;
        }
        if (strcmp(t.text, ".callprototype") == 0) {
            for (;;) {
                if (next(p, &t) != 0)
                    return -1;
                if (t.kind == T_PUNCT && t.punct == ';')
                    break;
                if (t.kind == T_EOF) {
                    perr(p, t.line, "unterminated .callprototype");
                    return -1;
                }
            }
            continue;
        }
        if (t.text[0] == '.') {
            char m[640];
            snprintf(m, sizeof(m), "unsupported directive '%s'", t.text);
            perr(p, t.line, m);
            return -1;
        }
        // Label? (ident followed by ':')
        ptx_tok_t u;
        if (next(p, &u) != 0)
            return -1;
        if (u.kind == T_PUNCT && u.punct == ':') {
            if (copy_name(p->last_label, sizeof(p->last_label), t.text,
                          p, t.line) != 0)
                return -1;
            // A label with pending instructions starts a new block; an
            // empty current block just takes the label.
            if (fn.blocks[cur].count > 0 || fn.blocks[cur].label[0] != '\0') {
                cur = new_block(&fn, p, t.text);
                if (cur == (size_t)-1)
                    return -1;
            } else {
                if (copy_name(fn.blocks[cur].label, sizeof(fn.blocks[cur].label), t.text, p, t.line) != 0)
                    return -1;
            }
            continue;
        }
        ptx_unget(&p->lx, &u);
        // Instruction, maybe predicated.
        char pred[64] = "";
        int pred_neg = 0;
        const char *opname = t.text;
        char opbuf[512];
        if (strcmp(t.text, "@") == 0) {
            if (next(p, &t) != 0)
                return -1;
            if (t.kind == T_PUNCT && t.punct == '!') {
                pred_neg = 1;
                if (next(p, &t) != 0)
                    return -1;
            }
            if (t.kind != T_IDENT || t.text[0] == '$') {
                perr(p, t.line, "expected predicate register after @");
                return -1;
            }
            if (copy_name(pred, sizeof(pred), t.text, p, t.line) != 0)
                return -1;
            if (next(p, &t) != 0)
                return -1;
            if (t.kind != T_IDENT) {
                perr(p, t.line, "expected opcode after predicate");
                return -1;
            }
            snprintf(opbuf, sizeof(opbuf), "%s", t.text);
            opname = opbuf;
        }
        if (parse_instr(p, &fn, cur, pred, pred_neg, opname, t.line) != 0)
            return -1;
        // An unconditional terminator starts a new (possibly unreachable)
        // block for following instructions.
        ptx_block_t *b = &fn.blocks[cur];
        if (b->count == 0)
            continue; // nothing emitted (dropped decl): no terminator
        ptx_instr_t *last = &b->instrs[b->count - 1];
        if (last->op == POP_BRA || last->op == POP_CBRA ||
            last->op == POP_BRX ||
            last->op == POP_RET || last->op == POP_EXIT) {
            cur = new_block(&fn, p, NULL);
            if (cur == (size_t)-1)
                return -1;
        }
    }
    if (has_predicated_ret) {
        // Every return becomes a branch to one shared epilogue so the
        // flow composer sees a single final ret block. The label is
        // chosen after all user labels are known.
        char ret_label[80] = "__acuda_ret";
        int clash = 1;
        while (clash) {
            clash = 0;
            for (size_t bi = 0; bi < fn.nblocks; bi++)
                if (fn.blocks[bi].label[0] &&
                    strcmp(fn.blocks[bi].label, ret_label) == 0) {
                    clash = 1;
                    break;
                }
            if (clash)
                strcat(ret_label, "_");
        }
        for (size_t bi = 0; bi < fn.nblocks; bi++) {
            ptx_block_t *blk = &fn.blocks[bi];
            if (blk->count == 0)
                continue;
            ptx_instr_t *last = &blk->instrs[blk->count - 1];
            if (last->op == POP_RET || last->op == POP_EXIT) {
                last->op = POP_BRA;
                snprintf(last->target, sizeof(last->target), "%s",
                         ret_label);
            } else if ((last->op == POP_BRA || last->op == POP_CBRA) &&
                       strcmp(last->target, synthetic_ret_ph) == 0) {
                snprintf(last->target, sizeof(last->target), "%s",
                         ret_label);
            }
        }
        size_t rb = new_block(&fn, p, ret_label);
        if (rb == (size_t)-1)
            return -1;
        ptx_instr_t r;
        memset(&r, 0, sizeof(r));
        r.op = POP_RET;
        if (block_add_instr(&fn, rb, &r, p, 0) != 0)
            return -1;
    }
    /* Keep the old array on OOM: assigning realloc's NULL result directly
     * would leave nfuncs>0 with funcs==NULL, and ptx_unit_free then walks
     * a NULL array (crash) instead of reporting the allocation failure. */
    {
        ptx_func_t *nf = realloc(unit->funcs,
                                 sizeof(*unit->funcs) * (unit->nfuncs + 1));
        if (!nf)
            return -1;
        unit->funcs = nf;
    }
    // Predicated execution beyond bra/mov: if-convert predicated
    // st/ld/alu/... into guard diamonds (head -cbra-> arm
    // -fallthrough-> join) for the existing flow machinery. bra
    // carries its condition natively, mov selects, ret/exit/bar were
    // rejected at parse. Synthetic labels are unit-unique; the arm
    // holds the single plain op, so joins need no new PHI kinds.
    // (Callees with predicated ops become multi-block and stay clean
    // errors at inline, like any multi-block callee.)
    {
        size_t bi = 0;
        while (bi < fn.nblocks) {
            ptx_block_t *blk = &fn.blocks[bi];
            size_t k = 0;
            while (k < blk->count &&
                   (blk->instrs[k].op == POP_MOV ||
                    blk->instrs[k].op == POP_BRA ||
                    blk->instrs[k].op == POP_CBRA ||
                    blk->instrs[k].op == POP_BRX ||
                    blk->instrs[k].op == POP_RET ||
                    blk->instrs[k].op == POP_EXIT ||
                    blk->instrs[k].op == POP_MBARRIER ||
                    blk->instrs[k].op == POP_RED_ASYNC ||
                    blk->instrs[k].op == POP_ELECT ||
                    blk->instrs[k].op == POP_BRKPT ||
                    !blk->instrs[k].pred[0] ||
                    (blk->instrs[k].op == POP_CALL &&
                     ext_declared(unit, blk->instrs[k].target))))
                k++;
            if (k >= blk->count) {
                bi++;
                continue;
            }
            ptx_instr_t guarded = blk->instrs[k];
            size_t oldcount = blk->count;
            // Predicated stores and mad.wide (which can produce a
            // pointer) cannot use the value-select form: lay them out as
            // a guard diamond (head -> arm / join), placing the arm and
            // join immediately after the head so the taken edge falls
            // through to the arm and both edges reconverge at the join.
            // Keeping the join as a real merge block is what lets a
            // barrier in the remainder stay converged (duplicating the
            // remainder onto one edge would execute the barrier on two
            // divergent paths).
            if (guarded.op == POP_MADWIDE || guarded.op == POP_ST ||
                guarded.op == POP_LD) {
                char lab[64], lab2[64];
                snprintf(lab, sizeof(lab), "__pred_%d", p->predst_n);
                snprintf(lab2, sizeof(lab2), "__predj_%d", p->predst_n);
                p->predst_n++;
                size_t ab = new_block(&fn, p, lab);
                if (ab == (size_t)-1)
                    return -1;
                ptx_instr_t plain = guarded;
                plain.pred[0] = '\0';
                plain.pred_neg = 0;
                if (block_add_instr(&fn, ab, &plain, p, guarded.line) != 0)
                    return -1;
                size_t jb = new_block(&fn, p, lab2);
                if (jb == (size_t)-1)
                    return -1;
                blk = &fn.blocks[bi];
                for (size_t j = k + 1; j < oldcount; j++)
                    if (block_add_instr(&fn, jb, &blk->instrs[j], p,
                                        guarded.line) != 0)
                        return -1;
                blk = &fn.blocks[bi];
                blk->count = k;
                ptx_instr_t cb;
                memset(&cb, 0, sizeof(cb));
                cb.op = POP_CBRA;
                cb.line = guarded.line;
                if (copy_name(cb.pred, sizeof(cb.pred), guarded.pred, p,
                              guarded.line) != 0)
                    return -1;
                cb.pred_neg = !guarded.pred_neg;
                if (copy_name(cb.target, sizeof(cb.target), lab2, p,
                              guarded.line) != 0)
                    return -1;
                if (block_add_instr(&fn, bi, &cb, p, guarded.line) != 0)
                    return -1;
                ptx_block_t arm = fn.blocks[ab];
                ptx_block_t join = fn.blocks[jb];
                size_t shift = fn.nblocks - (bi + 1) - 2;
                memmove(&fn.blocks[bi + 3], &fn.blocks[bi + 1],
                        sizeof(ptx_block_t) * shift);
                fn.blocks[bi + 1] = arm;
                fn.blocks[bi + 2] = join;
                bi++;
                continue;
            }
            // Select form for pure value ops. Excluded: loads/stores
            // (side effects), atomics/calls (convergence), u64 (may be
            // pointers), f16/bf16/packed/u8 destinations (no select
            // form); pred destinations only for vote/setp (bool select
            // over t_bool, same shape).
            if (guarded.op == POP_ATOM ||
                guarded.op == POP_ATOMCAS || guarded.op == POP_CALL ||
                guarded.op == POP_CVTPACK || guarded.op == POP_SET) {
                perr(p, guarded.line, "predicated execution unsupported for this instruction (needs convergent guard)");
                return -1;
            }
            // Wide multiply-add/mul-wide produce a 64-bit value
            // even though their source type is .u32. They are plain
            // value ops, so they can use the u64 select form.
            ptx_ty_t gty = guarded.ty;
            if (guarded.op == POP_SETP || guarded.op == POP_VOTE)
                gty = PTY_PRED;
            else if ((guarded.op == POP_MADWIDE ||
                      guarded.op == POP_MULWIDE) &&
                     guarded.ty != PTY_U16)
                gty = PTY_U64;
            else if (guarded.ty == PTY_F16 || guarded.ty == PTY_BF16)
                gty = PTY_U16; // half values live in 16-bit holders
            else if (guarded.ty == PTY_F16X2 || guarded.ty == PTY_BF16X2)
                gty = PTY_U32; // packed pairs live in 32-bit holders
            if ((gty == PTY_U64 && guarded.op != POP_MADWIDE &&
                 guarded.op != POP_MULWIDE) ||
                guarded.ty == PTY_U8 ||
                (guarded.ty == PTY_PRED &&
                 guarded.op != POP_VOTE && guarded.op != POP_SETP)) {
                perr(p, guarded.line, "predicated execution unsupported for this type (no select form)");
                return -1;
            }
            // `OP __ps_N, args` + `selp T d, new/old, p` (arms swapped
            // under @!p so the predicate itself stays unnegated; the
            // old value is Undef when never defined before).
            char tmpn[64];
            snprintf(tmpn, sizeof(tmpn), "__ps_%d", p->predst_n);
            p->predst_n++;
            if (func_add_reg(&fn, tmpn, gty) != 0) {
                perr(p, guarded.line, "out of memory");
                return -1;
            }
            ptx_instr_t ev, sv;
            memset(&ev, 0, sizeof(ev));
            memset(&sv, 0, sizeof(sv));
            ev = guarded;
            ev.pred[0] = '\0';
            ev.pred_neg = 0;
            if (copy_name(ev.dst, sizeof(ev.dst), tmpn, p,
                          guarded.line) != 0)
                return -1;
            sv.op = POP_SELP;
            sv.ty = gty;
            sv.line = guarded.line;
            if (copy_name(sv.dst, sizeof(sv.dst), guarded.dst, p,
                          guarded.line) != 0)
                return -1;
            int old_known = 0;
            for (size_t b2 = 0; b2 < fn.nblocks && !old_known; b2++) {
                size_t lim = (b2 == bi) ? k : fn.blocks[b2].count;
                for (size_t j = 0; j < lim; j++) {
                    const ptx_instr_t *o = &fn.blocks[b2].instrs[j];
                    if ((o->dst[0] && strcmp(o->dst, guarded.dst) == 0) ||
                        (o->dst2[0] && strcmp(o->dst2, guarded.dst) == 0))
                        old_known = 1;
                }
            }
            memset(&sv.src[0], 0, sizeof(sv.src[0]));
            memset(&sv.src[1], 0, sizeof(sv.src[1]));
            memset(&sv.src[2], 0, sizeof(sv.src[2]));
            // Predicates are physical registers: a loop-carried value is
            // defined in a later block of the same loop, so the earlier
            // "seen before" scan misses it.  Sourcing the destination
            // resolves to its current SSA value (or Undef on first use),
            // which is the correct skip-arm value.
            int old_source = old_known || gty == PTY_PRED;
            if (guarded.pred_neg) {
                // @!p: d = p ? old : new.
                if (old_source) {
                    sv.src[0].kind = OPD_REG;
                    if (copy_name(sv.src[0].reg, sizeof(sv.src[0].reg),
                                  guarded.dst, p, guarded.line) != 0)
                        return -1;
                } else {
                    sv.src[0].kind = OPD_UNDEF;
                }
                sv.src[1].kind = OPD_REG;
                if (copy_name(sv.src[1].reg, sizeof(sv.src[1].reg), tmpn,
                              p, guarded.line) != 0)
                    return -1;
            } else {
                // @p: d = p ? new : old.
                sv.src[0].kind = OPD_REG;
                if (copy_name(sv.src[0].reg, sizeof(sv.src[0].reg), tmpn,
                              p, guarded.line) != 0)
                    return -1;
                if (old_source) {
                    sv.src[1].kind = OPD_REG;
                    if (copy_name(sv.src[1].reg, sizeof(sv.src[1].reg),
                                  guarded.dst, p, guarded.line) != 0)
                        return -1;
                } else {
                    sv.src[1].kind = OPD_UNDEF;
                }
            }
            sv.src[2].kind = OPD_REG;
            if (copy_name(sv.src[2].reg, sizeof(sv.src[2].reg),
                          guarded.pred, p, guarded.line) != 0)
                return -1;
            sv.nsrc = 3;
            // Replace in place: eval then select, both unpredicated.
            blk->instrs[k] = ev;
            if (blk->count + 1 > blk->cap) {
                size_t ncap = blk->cap ? blk->cap * 2 : 8;
                ptx_instr_t *ni = malloc(sizeof(*ni) * ncap);
                if (!ni) {
                    perr(p, guarded.line, "out of memory");
                    return -1;
                }
                memcpy(ni, blk->instrs, sizeof(*ni) * blk->count);
                free(blk->instrs);
                blk->instrs = ni;
                blk->cap = ncap;
            }
            memmove(&blk->instrs[k + 2], &blk->instrs[k + 1],
                    sizeof(*blk->instrs) * (blk->count - (k + 1)));
            blk->instrs[k + 1] = sv;
            blk->count++;
            // Rescan this block: it may hold more predicated ops
            // (each desugar consumes exactly one).
            continue;
        }
    }
    // Unreachable-block elimination (bra/malformed_label corpus):    // drop blocks unreachable from the entry via bra/cbra/fallthrough
    // edges (standard DCE; dead code is unvalidated), then drop
    // branch-to-next terminators (semantic no-ops). Branches with
    // unknown targets stay for honest errors downstream.
    {
        size_t n = fn.nblocks;
        char *reach = calloc(n ? n : 1, 1);
        if (!reach) {
            perr(p, 0, "out of memory");
            return -1;
        }
        reach[0] = 1;
        int changed = 1;
        while (changed) {
            changed = 0;
            for (size_t i = 0; i < n; i++) {
                if (!reach[i])
                    continue;
                ptx_block_t *blk = &fn.blocks[i];
                if (blk->count == 0) {
                    if (i + 1 < n && !reach[i + 1]) {
                        reach[i + 1] = 1;
                        changed = 1;
                    }
                    continue;
                }
                ptx_instr_t *last = &blk->instrs[blk->count - 1];
                if (last->op == POP_BRX) {
                    for (int li = 0; li < fn.nbranchtargets; li++) {
                        if (strcmp(fn.branchtargets[li].label,
                                   last->target_list) != 0)
                            continue;
                        for (int q = 0; q < fn.branchtargets[li].n; q++)
                            for (size_t j = 0; j < n; j++)
                                if (fn.blocks[j].label[0] &&
                                    strcmp(fn.blocks[j].label,
                                           fn.branchtargets[li].targets[q]) ==
                                        0 &&
                                    !reach[j]) {
                                    reach[j] = 1;
                                    changed = 1;
                                }
                        break;
                    }
                }
                if (last->op == POP_BRA || last->op == POP_CBRA) {
                    size_t tgt = n;
                    for (size_t j = 0; j < n; j++) {
                        if (fn.blocks[j].label[0] &&
                            strcmp(fn.blocks[j].label, last->target) == 0) {
                            tgt = j;
                            break;
                        }
                    }
                    if (tgt < n && !reach[tgt]) {
                        reach[tgt] = 1;
                        changed = 1;
                    }
                }
                if (last->op != POP_BRA && last->op != POP_RET &&
                    last->op != POP_EXIT) {
                    if (i + 1 < n && !reach[i + 1]) {
                        reach[i + 1] = 1;
                        changed = 1;
                    }
                }
            }
        }
        size_t w = 0;
        for (size_t i = 0; i < n; i++) {
            if (!reach[i]) {
                free(fn.blocks[i].instrs);
                continue;
            }
            if (w != i)
                fn.blocks[w] = fn.blocks[i];
            w++;
        }
        fn.nblocks = w;
        free(reach);
        for (size_t i = 0; i + 1 < w; i++) {
            ptx_block_t *blk = &fn.blocks[i];
            if (blk->count == 0)
                continue;
            ptx_instr_t *last = &blk->instrs[blk->count - 1];
            if (last->op != POP_BRA)
                continue;
            // Drop only when the FIRST matching label is the next
            // block (duplicate labels keep their back-edge meaning).
            size_t tgt = w;
            for (size_t j = 0; j < w; j++) {
                if (fn.blocks[j].label[0] &&
                    strcmp(fn.blocks[j].label, last->target) == 0) {
                    tgt = j;
                    break;
                }
            }
            if (tgt == i + 1)
                blk->count--;
        }
    }
    // Module-scope .texref declarations referenced by texture
    // instructions become hidden kernel parameters so the existing
    // image descriptor path can bind them at launch.
    for (size_t bi = 0; bi < fn.nblocks; bi++) {
        ptx_block_t *blk = &fn.blocks[bi];
        for (size_t ii = 0; ii < blk->count; ii++) {
            ptx_instr_t *in = &blk->instrs[ii];
            if (in->op != POP_TEX && in->op != POP_TLD4 &&
                in->op != POP_TXQ)
                continue;
            if (in->tex_ref[0] == '\0')
                continue;
            int already = 0;
            for (size_t pi = 0; pi < fn.nparams; pi++)
                if (strcmp(fn.params[pi].name, in->tex_ref) == 0)
                    already = 1;
            if (already)
                continue;
            int is_module_texref = 0;
            for (size_t ti = 0; ti < unit->ntexrefs; ti++)
                if (strcmp(unit->texrefs[ti].name, in->tex_ref) == 0)
                    is_module_texref = 1;
            if (!is_module_texref)
                continue;
            if (fn.nparams + 1 > fn.params_cap) {
                size_t ncap = fn.params_cap ? fn.params_cap * 2 : 4;
                void *np = realloc(fn.params, sizeof(*fn.params) * ncap);
                if (!np) {
                    perr(p, 0, "out of memory");
                    return -1;
                }
                fn.params = np;
                fn.params_cap = ncap;
            }
            ptx_param_t *hp = &fn.params[fn.nparams++];
            memset(hp, 0, sizeof(*hp));
            strncpy(hp->name, in->tex_ref, sizeof(hp->name) - 1);
            hp->ty = PTY_U64;
            hp->is_tex = 1;
            hp->is_hidden_texref = 1;
        }
    }
    latch_guard(&fn);
    latch_guard_fwd(&fn);
    unit->funcs[unit->nfuncs++] = fn;
    return 0;
}

