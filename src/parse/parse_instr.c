// Instruction parsing: one PTX opcode line to ptx_instr_t, including
// operand/vector/memory-order forms.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "ptx.h"
#include "parse/parse_internal.h"

static int call_arg_push(parser_t *p, ptx_instr_t *in, const char *name,
                        int line) {
    if (in->nsrc >= 32) {
        perr(p, line, "too many call arguments");
        return -1;
    }
    ptx_opd_t *na = realloc(in->call_args,
                            sizeof(*na) * (size_t)(in->nsrc + 1));
    if (!na) {
        perr(p, line, "out of memory");
        return -1;
    }
    in->call_args = na;
    ptx_opd_t *op = &na[in->nsrc];
    memset(op, 0, sizeof(*op));
    op->kind = OPD_PARAM;
    if (copy_name(op->param, sizeof(op->param), name, p, line) != 0)
        return -1;
    in->nsrc++;
    return 0;
}

static int is_vec_type_token(const char *t) {
    return strcmp(t, "b128") == 0 || strcmp(t, "v2") == 0 ||
           strcmp(t, "v4") == 0 || strcmp(t, "v8") == 0;
}

// Scalar/vector ld/st cache operators are performance hints on this
// coherent backend; drop them so the regular space/type parsers apply.
static int is_ldst_cache_token(const char *t) {
    return strcmp(t, "ca") == 0 || strcmp(t, "cg") == 0 ||
           strcmp(t, "cs") == 0 || strcmp(t, "lu") == 0 ||
           strcmp(t, "cv") == 0 || strcmp(t, "wb") == 0 ||
           strcmp(t, "wt") == 0 || strcmp(t, "nc") == 0;
}

// Ordering and cache-hint qualifiers are accepted as hints on the
// coherent single-device model.
static int is_ldst_hint_token(const char *t) {
    if (strcmp(t, "volatile") == 0 || strcmp(t, "weak") == 0 ||
        strcmp(t, "relaxed") == 0 || strcmp(t, "acquire") == 0 ||
        strcmp(t, "release") == 0 || strcmp(t, "acq_rel") == 0 ||
        strcmp(t, "cta") == 0 || strcmp(t, "cluster") == 0 ||
        strcmp(t, "gpu") == 0 || strcmp(t, "sys") == 0 ||
        strcmp(t, "mmio") == 0 ||
        strcmp(t, "unified") == 0)
        return 1;
    return strncmp(t, "L1::", 4) == 0 || strncmp(t, "L2::", 4) == 0;
}

// Optional `.unified` suffix on an ld address operand.
static int skip_unified_suffix(parser_t *p) {
    ptx_tok_t t;
    if (next(p, &t) != 0)
        return -1;
    if (!(t.kind == T_IDENT && strcmp(t.text, ".unified") == 0))
        ptx_unget(&p->lx, &t);
    return 0;
}

int parse_instr(parser_t *p, ptx_func_t *fn, size_t bi,
                       const char *pred, int pred_neg, const char *opname, int opline) {    // Atomics take up to 6 dot-parts (sem+scope+space+op+type), past
    // the shared 5-part splitter, so they parse the opcode here.
    // Covered: add/inc/cas over .u32/.b32/.s32/.f32 (f32-add only,
    // integer CAS only); all return the OLD value like hardware.
    // Scope lowers by SPACE (shared->Workgroup, global/generic->Device;
    // explicit .cta/.gpu/.sys spellings accepted, .cluster rejected);
    // semantics always lower at AcquireRelease strength (sound for
    // .relaxed/.acquire/.release/.acq_rel). Other ops/widths
    // (and/or/xor/exch/min/max/dec, .u64/.f64, .b128, acquire/release
    // ld/st vectors) stay clean errors.
    if (strncmp(opname, "red.async.", 10) == 0 ||
        strncmp(opname, "multimem.red.async.", 19) == 0)
        return parse_red_async(p, fn, bi, pred, pred_neg, opname, opline);
    if (strncmp(opname, "multimem.", 9) == 0)
        return parse_multimem(p, fn, bi, pred, pred_neg, opname, opline);
    if ((strncmp(opname, "mapa", 4) == 0 &&
         (opname[4] == '\0' || opname[4] == '.')) ||
        (strncmp(opname, "getctarank", 10) == 0 &&
         (opname[10] == '\0' || opname[10] == '.')))
        return parse_cluster_instr(p, fn, bi, pred, pred_neg, opname,
                                   opline);
    if ((strncmp(opname, "atom", 4) == 0 &&
         (opname[4] == '\0' || opname[4] == '.')) ||
        (strncmp(opname, "red", 3) == 0 &&
         (opname[3] == '\0' || opname[3] == '.')))
        return parse_atom_instr(p, fn, bi, opname, opline);
    if ((strncmp(opname, "tex", 3) == 0 &&
         (opname[3] == '\0' || opname[3] == '.')) ||
        (strncmp(opname, "txq", 3) == 0 &&
         (opname[3] == '\0' || opname[3] == '.')) ||
        (strncmp(opname, "istypep", 7) == 0 &&
         (opname[7] == '\0' || opname[7] == '.')) ||
        (strncmp(opname, "tld4", 4) == 0 &&
         (opname[4] == '\0' || opname[4] == '.')))
        return parse_texture_instr(p, fn, bi, opname, opline);
    if (strncmp(opname, "wgmma", 5) == 0 && opname[5] == '.') {
        perr(p, opline, "wgmma requires sm_90a (Hopper) and is not implemented");
        return -1;
    }
    if (strncmp(opname, "tcgen05", 7) == 0 && opname[7] == '.') {
        perr(p, opline, "tcgen05 requires sm_100 (Blackwell) and is not implemented");
        return -1;
    }
    if (strstr(opname, "block_scale") != NULL) {
        perr(p, opline, "mma.sync block scaling is not implemented");
        return -1;
    }
    if ((strncmp(opname, "add", 3) == 0 ||
         strncmp(opname, "sub", 3) == 0 ||
         strncmp(opname, "fma", 3) == 0) &&
        (strstr(opname, ".f32.f16") != NULL ||
         strstr(opname, ".f32.bf16") != NULL)) {
        perr(p, opline, "mixed-precision add/sub/fma requires sm_100 and is not implemented");
        return -1;
    }
    if ((strncmp(opname, "st.async", 8) == 0 && opname[8] == '.') ||
        (strncmp(opname, "st.bulk", 7) == 0 && opname[7] == '.')) {
        perr(p, opline, "this instruction requires sm_90 (cluster/bulk/multimem) and is not implemented");
        return -1;
    }
    if (strncmp(opname, "tensormap.", 10) == 0) {
        perr(p, opline, "tensormap instructions require sm_90a and are not implemented");
        return -1;
    }
    if (strncmp(opname, "isspacep.", 9) == 0) {
        ptx_instr_t in;
        memset(&in, 0, sizeof(in));
        in.line = opline;
        const char *sp = opname + 9;
        if (strcmp(sp, "shared") == 0 || strcmp(sp, "shared::cta") == 0 ||
            strcmp(sp, "shared::cluster") == 0)
            in.isp_space = 0;
        else if (strcmp(sp, "global") == 0)
            in.isp_space = 1;
        else if (strcmp(sp, "local") == 0)
            in.isp_space = 2;
        else if (strcmp(sp, "const") == 0)
            in.isp_space = 3;
        else if (strcmp(sp, "param") == 0 ||
                 strcmp(sp, "param::entry") == 0)
            in.isp_space = 4;
        else {
            perr(p, opline, "unsupported isspacep state space");
            return -1;
        }
        in.op = POP_ISSPACEP;
        if (parse_dst(p, in.dst, sizeof(in.dst), 1) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (in.src[0].kind != OPD_REG) {
            perr(p, opline, "isspacep source must be a register");
            return -1;
        }
        in.nsrc = 1;
        if (expect_punct(p, ';', "after isspacep") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if ((strncmp(opname, "ldmatrix", 8) == 0 && opname[8] == '.') ||
        (strncmp(opname, "wmma", 4) == 0 && opname[4] == '.') ||
        (strncmp(opname, "mma", 3) == 0 && opname[3] == '.'))
        return parse_matrix_instr(p, fn, bi, pred, pred_neg, opname, opline);
    if (strncmp(opname, "mbarrier.", 9) == 0 ||
        (strncmp(opname, "cp.async.mbarrier.arrive", 24) == 0 &&
         (opname[24] == '\0' || opname[24] == '.')) ||
        strncmp(opname, "elect.sync", 10) == 0)
        return parse_mbarrier_instr(p, fn, bi, pred, pred_neg, opname,
                                    opline);
    if (strncmp(opname, "cvt.", 4) == 0 &&
        (strstr(opname, "e2m1") || strstr(opname, "e2m3") ||
         strstr(opname, "e3m2") || strstr(opname, "ue8m0") ||
         strstr(opname, "s2f6") || strstr(opname, "x4") ||
         strstr(opname, "scaled::") || strstr(opname, "e4m3x2") ||
         strstr(opname, "e5m2x2")))
        return parse_cvt_micro(p, fn, bi, opname, opline);
    if ((strncmp(opname, "suld", 4) == 0 &&
         (opname[4] == '\0' || opname[4] == '.')) ||
        (strncmp(opname, "sust", 4) == 0 &&
         (opname[4] == '\0' || opname[4] == '.')) ||
        (strncmp(opname, "sured", 5) == 0 &&
         (opname[5] == '\0' || opname[5] == '.')) ||
        (strncmp(opname, "suq", 3) == 0 &&
         (opname[3] == '\0' || opname[3] == '.')))
        return parse_surface_instr(p, fn, bi, opname, opline);
    char parts[6][32];
    int np = 0;
    memset(parts, 0, sizeof(parts));
    if (split_op(opname, parts, &np) != 0) {
        char m[640];
        snprintf(m, sizeof(m), "bad opcode '%s'", opname);
        perr(p, opline, m);
        return -1;
    }
    int has_cache_hint = 0;
    int ldst_mem_sem = 0;
    int ldst_mem_scope = 0;
    if ((strcmp(parts[0], "ld") == 0 || strcmp(parts[0], "st") == 0) &&
        np > 1) {
        int w = 1;
        for (int i = 1; i < np; i++) {
            if (is_ldst_cache_token(parts[i]))
                continue;
            if (is_ldst_hint_token(parts[i])) {
                if (strcmp(parts[i], "L2::cache_hint") == 0)
                    has_cache_hint = 1;
                else if (strcmp(parts[i], "relaxed") == 0 || strcmp(parts[i], "weak") == 0)
                    ldst_mem_sem = ldst_mem_sem < 1 ? 1 : ldst_mem_sem;
                else if (strcmp(parts[i], "acquire") == 0)
                    ldst_mem_sem = ldst_mem_sem < 2 ? 2 : ldst_mem_sem;
                else if (strcmp(parts[i], "release") == 0)
                    ldst_mem_sem = ldst_mem_sem < 3 ? 3 : ldst_mem_sem;
                else if (strcmp(parts[i], "acq_rel") == 0 || strcmp(parts[i], "volatile") == 0)
                    ldst_mem_sem = 4;
                else if (strcmp(parts[i], "mmio") == 0)
                    ldst_mem_sem = ldst_mem_sem < 4 ? 4 : ldst_mem_sem;
                else if (strcmp(parts[i], "cta") == 0)
                    ldst_mem_scope = 1;
                else if (strcmp(parts[i], "cluster") == 0)
                    ldst_mem_scope = 2;
                else if (strcmp(parts[i], "gpu") == 0)
                    ldst_mem_scope = 3;
                else if (strcmp(parts[i], "sys") == 0)
                    ldst_mem_scope = 4;
                continue;
            }
            if (w != i)
                memcpy(parts[w], parts[i], sizeof(parts[0]));
            w++;
        }
        np = w;
    }
    ptx_instr_t in;
    memset(&in, 0, sizeof(in));
    in.line = opline;
    if (copy_name(in.pred, sizeof(in.pred), pred[0] ? pred : "", p, opline) !=
        0)
        return -1;
    in.pred_neg = pred_neg;

    if (strcmp(opname, "acuda.dot2.f32.f16") == 0) {
        // ACuda extension: d = c + lo(a)*lo(b) + hi(a)*hi(b) for two
        // f16x2 words and an f32 accumulator. Lowers to the VALVE mixed
        // dot instruction when available, otherwise to widen + FMA.
        in.op = POP_DOT2;
        in.ty = PTY_F32;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        for (int i = 0; i < 3; i++) {
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[i]) != 0)
                return -1;
        }
        in.nsrc = 3;
        if (expect_punct(p, ';', "after acuda.dot2") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }

    if (strcmp(parts[0], "ret") == 0 && np == 1) {
        in.op = POP_RET;
        if (expect_punct(p, ';', "after ret") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "exit") == 0 && np == 1) {
        in.op = POP_EXIT;
        if (expect_punct(p, ';', "after exit") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "brkpt") == 0 && np == 1) {
        // Debug breakpoint. SPIR-V has no portable compute breakpoint;
        // emit an OpLine source marker (the closest debug aid) and
        // otherwise remain a no-op.
        in.op = POP_BRKPT;
        if (expect_punct(p, ';', "after brkpt") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "trap") == 0 && np == 1) {
        // Trap halts the kernel: closest supported semantics is a
        // control-flow end of function (the GPU traps the work item
        // like a divergence exit).
        in.op = POP_EXIT;
        if (expect_punct(p, ';', "after trap") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "bra") == 0 && (np == 1 || (np == 2 && strcmp(parts[1], "uni") == 0))) {
        in.op = pred[0] ? POP_CBRA : POP_BRA;
        // Predicated bra carries its condition in pred/pred_neg.
        ptx_tok_t t;
        if (next(p, &t) != 0)
            return -1;
        if (t.kind != T_IDENT) {
            perr(p, t.line, "expected branch target label");
            return -1;
        }
        if (copy_name(in.target, sizeof(in.target), t.text, p, t.line) != 0)
            return -1;
        if (expect_punct(p, ';', "after bra") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "brx") == 0 && np >= 2 &&
        strcmp(parts[1], "idx") == 0 &&
        (np == 2 || (np == 3 && strcmp(parts[2], "uni") == 0))) {
        // Indirect branch over a .branchtargets label list. Lowered
        // post-parse into an index comparison chain.
        in.op = POP_BRX;
        in.ty = PTY_U32;
        if (parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (in.src[0].kind != OPD_REG && in.src[0].kind != OPD_IMM) {
            perr(p, opline, "brx.idx index must be a .u32 register or immediate");
            return -1;
        }
        in.nsrc = 1;
        if (expect_comma(p) != 0)
            return -1;
        ptx_tok_t t;
        if (next(p, &t) != 0)
            return -1;
        if (t.kind != T_IDENT) {
            perr(p, t.line, "expected .branchtargets label");
            return -1;
        }
        if (copy_name(in.target_list, sizeof(in.target_list), t.text, p,
                      t.line) != 0)
            return -1;
        if (expect_punct(p, ';', "after brx.idx") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    // Predicated control/barrier stays explicit: ret/exit need
    // control-flow design, barriers must stay converged. Everything
    // else predicated if-converts in a post-pass (bra/mov lower
    // natively: bra carries its condition, mov selects).
    if (in.pred[0]) {
        if (strcmp(parts[0], "ret") == 0 || strcmp(parts[0], "exit") == 0) {
            perr(p, opline, "predicated ret/exit unsupported");
            return -1;
        }
        if (strcmp(parts[0], "bar") == 0 ||
            strcmp(parts[0], "barrier") == 0) {
            perr(p, opline, "predicated barrier unsupported");
            return -1;
        }
    }
    if (strcmp(parts[0], "shfl") == 0 &&
        ((np == 4 && strcmp(parts[1], "sync") == 0) ||
         (np == 3 && (strcmp(parts[1], "up") == 0 ||
                      strcmp(parts[1], "down") == 0 ||
                      strcmp(parts[1], "bfly") == 0 ||
                      strcmp(parts[1], "idx") == 0)))) {
        // Warp shuffle (see emit_shfl for the model): .up/.down/.bfly/
        // .idx over .b32 (or aliases) or .f32, with an optional `d|p`
        // validity predicate. The current spelling takes four operands
        // (value, lane/delta, clamp, membermask); the deprecated
        // unsuffixed spelling (PTX < 6.0, still accepted by ptxas with
        // warnings) takes three and an implicit full-warp membermask.
        int legacy = np == 3;
        int mo = legacy ? 1 : 2;
        int mode = -1;
        if (strcmp(parts[mo], "up") == 0)
            mode = 0;
        else if (strcmp(parts[mo], "down") == 0)
            mode = 1;
        else if (strcmp(parts[mo], "bfly") == 0)
            mode = 2;
        else if (strcmp(parts[mo], "idx") == 0)
            mode = 3;
        else {
            perr(p, opline, "unsupported shfl mode (only .up/.down/.bfly/.idx)");
            return -1;
        }
        if (map_ty(parts[mo + 1], &in.ty) != 0 ||
            (in.ty != PTY_U32 && in.ty != PTY_F32)) {
            perr(p, opline, "shfl supports .b32/.f32 only");
            return -1;
        }
        in.op = POP_SHFL;
        in.shfl_mode = mode;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (at_punct(p, '|')) {
            ptx_tok_t q;
            if (next(p, &q) != 0)
                return -1;
            if (q.kind != T_IDENT) {
                perr(p, q.line, "expected predicate after '|'");
                return -1;
            }
            if (copy_name(in.dst2, sizeof(in.dst2), q.text, p, q.line) !=
                0)
                return -1;
        }
        if (expect_comma(p) != 0)
            return -1;
        int nops = legacy ? 3 : 4;
        for (int s = 0; s < nops; s++) {
            if (parse_operand(p, &in.src[s]) != 0)
                return -1;
            if (s + 1 < nops && expect_comma(p) != 0)
                return -1;
        }
        if (legacy) {
            // Deprecated spelling: implicit full-warp membermask.
            in.src[3].kind = OPD_IMM;
            in.src[3].imm = 0xffffffffu;
            in.src[3].imm_is_float = 0;
        }
        in.nsrc = 4;
        if (expect_punct(p, ';', "after shfl") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "activemask") == 0 && np == 2) {
        // Active-lane mask of the warp (see emit_vote): single b32
        // destination, no operands. Converged execution required.
        if (map_ty(parts[1], &in.ty) != 0 || in.ty != PTY_U32) {
            perr(p, opline, "activemask supports .b32 only");
            return -1;
        }
        in.op = POP_VOTE;
        in.vote_mode = 3;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        in.nsrc = 0;
        if (expect_punct(p, ';', "after activemask") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "vote") == 0 &&
        ((np == 4 && strcmp(parts[1], "sync") == 0) ||
         (np == 3 && (strcmp(parts[1], "all") == 0 ||
                      strcmp(parts[1], "any") == 0 ||
                      strcmp(parts[1], "uni") == 0 ||
                      strcmp(parts[1], "ballot") == 0)))) {
        // Warp votes. Current spelling: vote.sync.{all,any,uni}.pred
        // and vote.sync.ballot.b32, both with a membermask. The
        // deprecated unsuffixed spelling (vote.{all,any,uni}.pred,
        // vote.ballot.b32) uses an implicit full-warp membermask.
        // .uni is true when all member lanes agree (all true or all
        // false); lowering compares the masked ballot against both the
        // active mask and zero.
        int legacy = np == 3;
        int mo = legacy ? 1 : 2;
        int mode = -1;
        if (strcmp(parts[mo], "all") == 0)
            mode = 0;
        else if (strcmp(parts[mo], "any") == 0)
            mode = 1;
        else if (strcmp(parts[mo], "ballot") == 0)
            mode = 2;
        else if (strcmp(parts[mo], "uni") == 0)
            mode = 4;
        else {
            perr(p, opline, "unsupported vote mode (only .all/.any/.uni/.ballot)");
            return -1;
        }
        const char *ts = parts[mo + 1];
        ptx_ty_t ty;
        if (map_ty(ts, &ty) != 0) {
            perr(p, opline, "bad vote type");
            return -1;
        }
        if ((mode == 2 && ty != PTY_U32) ||
            (mode != 2 && ty != PTY_PRED)) {
            perr(p, opline, "vote.all/any/uni need .pred, vote.ballot needs .b32");
            return -1;
        }
        in.op = POP_VOTE;
        in.vote_mode = mode;
        in.ty = ty;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0)
            return -1;
        if (at_punct(p, '!'))
            in.bneg = 1;
        if (parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (in.src[0].kind != OPD_REG &&
            (in.src[0].kind != OPD_IMM || in.src[0].imm_is_float ||
             (in.src[0].imm != 0 && in.src[0].imm != 1))) {
            perr(p, opline, "vote value needs a predicate register or 0/1");
            return -1;
        }
        in.nsrc = 1;
        if (!legacy) {
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
                return -1;
            in.nsrc = 2;
        }
        if (expect_punct(p, ';', "after vote") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "match") == 0 && np == 4 &&
        strcmp(parts[1], "any") == 0 && strcmp(parts[2], "sync") == 0) {
        // Warp match (see emit_match): bitmask of warp lanes holding
        // the same value, via an unrolled shuffle-compare loop over
        // the 32-lane warp. .b32 (or aliases) or .f32 (bitwise);
        // .all stays out (no corpus coverage).
        if (map_ty(parts[3], &in.ty) != 0 ||
            (in.ty != PTY_U32 && in.ty != PTY_F32)) {
            perr(p, opline, "match supports .b32/.f32 only");
            return -1;
        }
        in.op = POP_MATCH;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (expect_punct(p, ';', "after match") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "redux") == 0 && np == 4 &&
        strcmp(parts[1], "sync") == 0) {
        // Warp reduction (see emit_redux): .add/.min/.max over
        // .u32/.s32 (aliases share bits; signedness picks the opcode),
        // value + membermask operands. Mask-excluded lanes contribute
        // the op identity, so partial masks reduce correctly.
        // Predicated redux desugars via selects like other pure ops.
        int op = -1;
        if (strcmp(parts[2], "add") == 0)
            op = 0;
        else if (strcmp(parts[2], "min") == 0)
            op = 1;
        else if (strcmp(parts[2], "max") == 0)
            op = 2;
        else if (strcmp(parts[2], "and") == 0)
            op = 3;
        else if (strcmp(parts[2], "or") == 0)
            op = 4;
        else if (strcmp(parts[2], "xor") == 0)
            op = 5;
        else {
            perr(p, opline, "unsupported redux op");
            return -1;
        }
        if (strcmp(parts[3], "s32") == 0) {
            in.ty = PTY_U32;
            in.sign = 1;
        } else if (strcmp(parts[3], "u32") == 0 ||
                   strcmp(parts[3], "b32") == 0) {
            in.ty = PTY_U32;
            in.sign = 0;
        } else {
            perr(p, opline, "redux supports .u32/.s32/.b32 only");
            return -1;
        }
        in.op = POP_REDUX;
        in.redux_op = op;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (expect_punct(p, ';', "after redux") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "call") == 0 &&
        (np == 1 || (np == 2 && strcmp(parts[1], "uni") == 0))) {
        // Direct call (ZLUDA corpus): "call (rets...), fname,
        // (args...)", with the rets and/or args omitted when empty
        // ("call (out), add;" and bare "call f;"). ".uni" is a
        // uniformity assertion; inlining is exact either way.
        // Results/arguments name slots or registers (resolved by the
        // post-parse inline pass); indirect calls stay clean errors
        // (callee must be a bare name).
        in.op = POP_CALL;
        if (at_punct(p, '(')) {
            if (!at_punct(p, ')')) {
                for (;;) {
                    ptx_tok_t t;
                    if (next(p, &t) != 0)
                        return -1;
                    if (t.kind != T_IDENT) {
                        perr(p, t.line, "expected call result name");
                        return -1;
                    }
                    // Vector-reg results expand to components (the
                    // callee signature expands identically, so counts
                    // and types still agree in the inline pass).
                    ptx_vecdecl_t *v = vec_find(fn, t.text);
                    if (v) {
                        if (in.nrets + v->n > 4) {
                            perr(p, t.line, "too many call results");
                            return -1;
                        }
                        size_t lb = strlen(t.text);
                        for (int i = 0; i < v->n; i++) {
                            char nm[64];
                            memcpy(nm, t.text, lb);
                            nm[lb] = '_';
                            nm[lb + 1] = "xyzw"[i];
                            nm[lb + 2] = '\0';
                            if (copy_name(in.rets[in.nrets],
                                          sizeof(in.rets[in.nrets]),
                                          nm, p, t.line) != 0)
                                return -1;
                            in.nrets++;
                        }
                    } else {
                        if (in.nrets + 1 > 4) {
                            perr(p, t.line, "too many call results");
                            return -1;
                        }
                        if (copy_name(in.rets[in.nrets],
                                      sizeof(in.rets[in.nrets]), t.text, p,
                                      t.line) != 0)
                            return -1;
                        in.nrets++;
                    }
                    if (at_punct(p, ','))
                        continue;
                    break;
                }
                if (expect_punct(p, ')', "after call results") != 0)
                    return -1;
            }
            if (expect_comma(p) != 0)
                return -1;
        }
        {
            ptx_tok_t t;
            if (next(p, &t) != 0)
                return -1;
            if (t.kind != T_IDENT || t.text[0] == '$') {
                perr(p, t.line, "indirect calls unsupported (need callee name)");
                return -1;
            }
            if (copy_name(in.target, sizeof(in.target), t.text, p,
                          t.line) != 0)
                return -1;
        }
        if (at_punct(p, ';')) {
            return block_add_instr(fn, bi, &in, p, opline);
        }
        if (expect_comma(p) != 0)
            return -1;
        if (at_punct(p, '(')) {
            // Argument list follows ('(' already consumed).
        } else {
            // Indirect call with a target list/prototype and no args:
            // "call f, flist;".
            ptx_tok_t t;
            if (next(p, &t) != 0)
                return -1;
            if (t.kind != T_IDENT) {
                perr(p, t.line, "expected indirect call target list");
                return -1;
            }
            if (copy_name(in.target_list, sizeof(in.target_list), t.text,
                          p, t.line) != 0)
                return -1;
            if (expect_punct(p, ';', "after indirect call target list") !=
                0)
                return -1;
            if (in.nsrc > 0 && in.nsrc <= 8)
                memcpy(in.src, in.call_args,
                       sizeof(in.src[0]) * (size_t)in.nsrc);
            return block_add_instr(fn, bi, &in, p, opline);
        }
        if (!at_punct(p, ')')) {
            for (;;) {
                ptx_tok_t t;
                if (next(p, &t) != 0)
                    return -1;
                if (t.kind != T_IDENT) {
                    perr(p, t.line, "expected call argument name");
                    return -1;
                }
                // Vector-reg arguments expand to components (like
                // results above).
                {
                    ptx_vecdecl_t *v = vec_find(fn, t.text);
                    if (v) {
                        if (in.nsrc + v->n >
                            (int)(sizeof(in.src) / sizeof(in.src[0]))) {
                            perr(p, t.line, "too many call arguments");
                            return -1;
                        }
                        size_t lb = strlen(t.text);
                        for (int i = 0; i < v->n; i++) {
                            char nm[256];
                            memcpy(nm, t.text, lb);
                            nm[lb] = '_';
                            nm[lb + 1] = "xyzw"[i];
                            nm[lb + 2] = '\0';
                            if (call_arg_push(p, &in, nm, t.line) != 0)
                                return -1;
                        }
                    } else {
                        if (call_arg_push(p, &in, t.text, t.line) != 0)
                            return -1;
                    }
                }
                if (at_punct(p, ','))
                    continue;
                break;
            }
            if (expect_punct(p, ')', "after call arguments") != 0)
                return -1;
        }
        if (at_punct(p, ',')) {
            // Indirect-call flist/fproto operand. A .calltargets label
            // supplies the candidate set for runtime dispatch; a
            // .callprototype label carries no target list.
            ptx_tok_t t;
            if (next(p, &t) != 0)
                return -1;
            if (t.kind != T_IDENT) {
                perr(p, t.line, "expected indirect call target list");
                return -1;
            }
            if (copy_name(in.target_list, sizeof(in.target_list), t.text,
                          p, t.line) != 0)
                return -1;
            if (expect_punct(p, ';', "after indirect call target list") != 0)
                return -1;
        } else if (expect_punct(p, ';', "after call") != 0)
            return -1;
        // The device-call inliner still reads the fixed src[] array.
        if (in.nsrc > 0 && in.nsrc <= 8)
            memcpy(in.src, in.call_args,
                   sizeof(in.src[0]) * (size_t)in.nsrc);
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "mov") == 0 && np >= 2 && np <= 3 &&
        (np == 3 ? (strcmp(parts[1], "v2") == 0 ||
                    strcmp(parts[1], "v4") == 0 ||
                    strcmp(parts[1], "v8") == 0)
                 : strcmp(parts[1], "b128") == 0)) {
        // Scalarized vector move: reg/brace operands expand to N
        // scalar movs of the element type (see docs/supported-isa.md).
        // .v8 takes eight brace-listed registers (ZLUDA's
        // vector8_extract); .v8 vector-reg declarations are not a
        // thing, so that spelling stays a clean error below.
        // Predicated vector mov stays a clean error (no corpus need).
        int b128 = strcmp(parts[1], "b128") == 0;
        int n = b128 ? 2 : parts[1][1] - '0';
        ptx_ty_t elem = PTY_U64;
        if (!b128 && vec_elem_ty(parts[2], &elem) != 0) {
            perr(p, opline, "vector mov needs u/s/b 8/16/32 element types");
            return -1;
        }
        if (pred[0] || pred_neg) {
            perr(p, opline, "predicated vector mov unsupported");
            return -1;
        }
        char dd[8][64], ss[8][64];
        memset(dd, 0, sizeof(dd));
        memset(ss, 0, sizeof(ss));
        if (parse_vec_operand(p, dd, n, elem, opline) != 0)
            return -1;
        if (expect_comma(p) != 0)
            return -1;
        if (parse_vec_operand(p, ss, n, elem, opline) != 0)
            return -1;
        if (expect_punct(p, ';', "after vector mov") != 0)
            return -1;
        // Overlapping dst/src names need simultaneous-assignment
        // semantics (sequential movs would clobber later sources):
        // route through fresh temps when they intersect.
        int overlap = 0;
        for (int i = 0; i < n && !overlap; i++) {
            for (int j = 0; j < n && !overlap; j++) {
                if (strcmp(dd[i], ss[j]) == 0)
                    overlap = 1;
            }
        }
        if (overlap) {
            char tt[8][64];
            memset(tt, 0, sizeof(tt));
            for (int i = 0; i < n; i++) {
                snprintf(tt[i], sizeof(tt[i]), "__vmv_%u",
                         (unsigned)fn->nregs);
                if (func_add_reg(fn, tt[i], elem) != 0) {
                    perr(p, opline, "out of memory");
                    return -1;
                }
            }
            for (int i = 0; i < n; i++) {
                ptx_instr_t mi;
                memset(&mi, 0, sizeof(mi));
                mi.op = POP_MOV;
                mi.ty = elem;
                mi.line = opline;
                if (copy_name(mi.dst, sizeof(mi.dst), tt[i], p,
                              opline) != 0)
                    return -1;
                mi.src[0].kind = OPD_REG;
                if (copy_name(mi.src[0].reg, sizeof(mi.src[0].reg),
                              ss[i], p, opline) != 0)
                    return -1;
                mi.nsrc = 1;
                if (block_add_instr(fn, bi, &mi, p, opline) != 0)
                    return -1;
            }
            for (int i = 0; i < n; i++) {
                if (copy_name(ss[i], sizeof(ss[i]), tt[i], p,
                              opline) != 0)
                    return -1;
            }
        }
        for (int i = 0; i < n; i++) {
            ptx_instr_t mi;
            memset(&mi, 0, sizeof(mi));
            mi.op = POP_MOV;
            mi.ty = elem;
            mi.line = opline;
            if (copy_name(mi.dst, sizeof(mi.dst), dd[i], p,
                          opline) != 0)
                return -1;
            mi.src[0].kind = OPD_REG;
            if (copy_name(mi.src[0].reg, sizeof(mi.src[0].reg), ss[i],
                          p, opline) != 0)
                return -1;
            mi.nsrc = 1;
            if (block_add_instr(fn, bi, &mi, p, opline) != 0)
                return -1;
        }
        return 0;
    }
    if (strcmp(parts[0], "cp") == 0 && np >= 3 &&
        strcmp(parts[1], "async") == 0) {
        // Asynchronous shared<->global copies.  The backend performs
        // them synchronously (there is no Vulkan compute async-copy
        // primitive), so completion is immediate and group/wait ops
        // are no-ops.
        if (strcmp(parts[2], "bulk") == 0) {
            if (np >= 4 && strcmp(parts[3], "commit_group") == 0) {
                if (expect_punct(p, ';', "after cp.async.bulk.commit_group") != 0)
                    return -1;
                return 0;
            }
            if (np >= 4 && strcmp(parts[3], "wait_group") == 0) {
                int wi = 4;
                if (wi < np && strcmp(parts[wi], "read") == 0)
                    wi++;
                ptx_opd_t depth;
                if (parse_operand(p, &depth) != 0)
                    return -1;
                if (depth.kind != OPD_IMM) {
                    perr(p, opline,
                         "cp.async.bulk.wait_group needs an immediate depth");
                    return -1;
                }
                if (expect_punct(p, ';', "after cp.async.bulk.wait_group") != 0)
                    return -1;
                return 0;
            }
            if (np >= 4 && strcmp(parts[3], "prefetch") == 0) {
                if (np >= 5 && strcmp(parts[4], "tensor") == 0) {
                    perr(p, opline,
                         "cp.async.bulk.prefetch.tensor is not implemented");
                    return -1;
                }
                // Prefetch is a cache hint; consume its operands.
                if (parse_operand(p, &in.src[0]) != 0)
                    return -1;
                if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
                    return -1;
                if (at_punct(p, ',') && parse_operand(p, &in.src[2]) != 0)
                    return -1;
                if (expect_punct(p, ';', "after cp.async.bulk.prefetch") != 0)
                    return -1;
                return 0;
            }
            if (np >= 4 && (strcmp(parts[3], "tensor") == 0 ||
                            strcmp(parts[3], "reduce") == 0)) {
                perr(p, opline,
                     "cp.async.bulk tensor/reduce forms are not implemented");
                return -1;
            }
            int shared_pos = -1, global_pos = -1;
            int ignore_oob = 0, multicast = 0;
            for (int i = 3; i < np; i++) {
                if (strcmp(parts[i], "shared::cta") == 0 ||
                    strcmp(parts[i], "shared::cluster") == 0 ||
                    strcmp(parts[i], "shared") == 0) {
                    if (shared_pos < 0)
                        shared_pos = i;
                } else if (strcmp(parts[i], "global") == 0) {
                    if (global_pos < 0)
                        global_pos = i;
                } else if (strcmp(parts[i], "ignore_oob") == 0) {
                    ignore_oob = 1;
                } else if (strcmp(parts[i], "multicast::cluster") == 0) {
                    multicast = 1;
                }
            }
            if (shared_pos < 0 || global_pos < 0 || multicast ||
                ignore_oob) {
                perr(p, opline,
                     "cp.async.bulk supports shared::cta<->global only");
                return -1;
            }
            int dir = shared_pos < global_pos ? 1 : 2;
            in.op = POP_CPASYNC_BULK;
            in.cp_bulk = dir;
            if (parse_operand(p, &in.src[0]) != 0 ||
                expect_comma(p) != 0 ||
                parse_operand(p, &in.src[1]) != 0 ||
                expect_comma(p) != 0 ||
                parse_operand(p, &in.src[2]) != 0)
                return -1;
            if (in.src[0].kind != OPD_ADDR || in.src[1].kind != OPD_ADDR) {
                perr(p, opline, "cp.async.bulk takes [dst], [src] addresses");
                return -1;
            }
            if (in.src[2].kind == OPD_IMM &&
                (in.src[2].imm == 0 || (in.src[2].imm & 15u) != 0)) {
                perr(p, opline,
                     "cp.async.bulk size must be a nonzero multiple of 16");
                return -1;
            }
            in.nsrc = 3;
            if (dir == 1) {
                // [mbar] completion operand; the synchronous copy
                // makes it unnecessary, but consume it for syntax.
                if (expect_comma(p) != 0 ||
                    parse_operand(p, &in.src[3]) != 0)
                    return -1;
                if (in.src[3].kind != OPD_ADDR) {
                    perr(p, opline, "cp.async.bulk needs an mbarrier address");
                    return -1;
                }
            } else if (at_punct(p, ',')) {
                if (parse_operand(p, &in.src[3]) != 0)
                    return -1;
            }
            if (expect_punct(p, ';', "after cp.async.bulk") != 0)
                return -1;
            return block_add_instr(fn, bi, &in, p, opline);
        }
        if (np == 3 && (strcmp(parts[2], "commit_group") == 0 ||
                        strcmp(parts[2], "wait_all") == 0)) {
            if (expect_punct(p, ';', "after cp.async group op") != 0)
                return -1;
            return 0;
        }
        if (np == 3 && strcmp(parts[2], "wait_group") == 0) {
            ptx_opd_t depth;
            if (parse_operand(p, &depth) != 0)
                return -1;
            if (depth.kind != OPD_IMM) {
                perr(p, opline, "cp.async.wait_group needs an immediate depth");
                return -1;
            }
            if (expect_punct(p, ';', "after cp.async.wait_group") != 0)
                return -1;
            return 0;
        }
        int pi = 2;
        if (strcmp(parts[pi], "ca") == 0 || strcmp(parts[pi], "cg") == 0)
            pi++;
        while (pi < np &&
               (strncmp(parts[pi], "L2::", 4) == 0 ||
                strncmp(parts[pi], "level::", 7) == 0))
            pi++;
        if (pi + 1 >= np ||
            !(strcmp(parts[pi], "shared") == 0 ||
              strcmp(parts[pi], "shared::cta") == 0) ||
            strcmp(parts[pi + 1], "global") != 0) {
            perr(p, opline,
                 "only cp.async.{ca,cg}.shared.global is supported");
            return -1;
        }
        in.op = POP_CPASYNC;
        in.ty = PTY_U32;
        if (parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        if (in.src[0].kind != OPD_ADDR || in.src[1].kind != OPD_ADDR) {
            perr(p, opline, "cp.async takes [dst], [src] addresses");
            return -1;
        }
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[2]) != 0)
            return -1;
        if (in.src[2].kind != OPD_IMM ||
            (in.src[2].imm != 4 && in.src[2].imm != 8 &&
             in.src[2].imm != 16)) {
            perr(p, opline, "cp.async copy size must be 4, 8 or 16");
            return -1;
        }
        in.cp_size = (int)in.src[2].imm;
        in.src_size = in.cp_size;
        in.nsrc = 3;
        if (at_punct(p, ',')) {
            if (parse_operand(p, &in.src[3]) != 0)
                return -1;
            in.nsrc = 4;
            // A predicate operand is ignore-src; otherwise it is the
            // runtime source size.
            if (in.src[3].kind == OPD_IMM) {
                if (in.src[3].imm > (uint64_t)in.cp_size) {
                    perr(p, opline,
                         "cp.async source size must be <= copy size");
                    return -1;
                }
                in.src_size = (int)in.src[3].imm;
            } else {
                in.src_size = -1;
            }
            if (at_punct(p, ',')) {
                if (parse_operand(p, &in.src[4]) != 0)
                    return -1;
                in.nsrc = 5;
            }
        }
        if (expect_punct(p, ';', "after cp.async") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "createpolicy") == 0 && np == 4 &&
        strcmp(parts[1], "fractional") == 0 && strcmp(parts[3], "b64") == 0) {
        // createpolicy.fractional.L2::evict_last.b64 d, fraction
        // A 64-bit L2 cache-eviction policy descriptor. Cache hints
        // have no architectural effect on results, and we do not model
        // them, so the descriptor is the default-policy value 0; any
        // ld/st that consume it already ignore their ::cache_hint.
        in.op = POP_MOV;
        in.ty = PTY_U64;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        in.src[0].kind = OPD_IMM;
        in.src[0].imm = 0;
        in.src[0].imm_is_float = 0;
        in.nsrc = 1;
        if (expect_punct(p, ';', "after createpolicy") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "mov") == 0 && np == 2) {
        in.op = POP_MOV;
        // Brace dest for unpacking: mov.b64 {lo, hi}, src (and sinks)
        {
            ptx_tok_t bt;
            if (next(p, &bt) == 0 && bt.kind == T_PUNCT && bt.punct == '{') {
                ptx_unget(&p->lx, &bt);
                if (strcmp(parts[1], "b64") == 0 || strcmp(parts[1], "b32") == 0) {
                    char d0[64], d1[64];
                    // parse brace list
                    if (expect_punct(p, '{', "to open brace list") != 0) return -1;
                    for (int i = 0; i < 2; i++) {
                        ptx_tok_t tt;
                        if (next(p, &tt) != 0) return -1;
                        if (tt.kind == T_PUNCT && tt.punct == '_') {
                            snprintf(i==0?d0:d1, 64, "__sink_%d_%d", opline, i);
                        } else if (tt.kind == T_IDENT) {
                            if (strcmp(tt.text, "_") == 0) {
                                snprintf(i==0?d0:d1, 64, "__sink_%d_%d", opline, i);
                            } else {
                                if (copy_name(i==0?d0:d1, 64, tt.text, p, tt.line) != 0) return -1;
                                // handle vector component
                                char vc[64];
                                int vr = vec_component_reg(p, i==0?d0:d1, vc, tt.line);
                                if (vr > 0) { if (copy_name(i==0?d0:d1, 64, vc, p, tt.line)!=0) return -1; }
                            }
                        } else {
                            perr(p, tt.line, "expected destination register");
                            return -1;
                        }
                        if (i == 0 && expect_comma(p) != 0) return -1;
                    }
                    if (expect_punct(p, '}', "to close brace list") != 0) return -1;
                    if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0) return -1;
                    if (expect_punct(p, ';', "after mov") != 0) return -1;
                    // expand
                    if (strcmp(parts[1], "b64") == 0) {
                        // mov.b64 {lo,hi}, src -> split into two u32
                        // words.  A predicate must guard both halves:
                        // expanding the pair unconditionally clobbers
                        // the destination when the predicate is false
                        // (predicated DMUL/DFMA pair writes hit this).
                        int guarded = pred[0] || pred_neg;
                        char t0[64], lo[64], hi[64];
                        // RZ as a destination discards the word; give it
                        // a throwaway register so the split has a legal
                        // destination.
                        if (strcmp(d0, "RZ") == 0) {
                            snprintf(d0, 64, "__mvs_%u", (unsigned)fn->nregs);
                            if (func_add_reg(fn, d0, PTY_U32) != 0) return -1;
                        }
                        if (strcmp(d1, "RZ") == 0) {
                            snprintf(d1, 64, "__mvs_%u", (unsigned)fn->nregs);
                            if (func_add_reg(fn, d1, PTY_U32) != 0) return -1;
                        }
                        snprintf(t0,64,"__mvu_%u",(unsigned)fn->nregs); if(func_add_reg(fn,t0,PTY_U64)!=0)return -1;
                        if (guarded) {
                            snprintf(lo,64,"__mvu_%u",(unsigned)fn->nregs); if(func_add_reg(fn,lo,PTY_U32)!=0)return -1;
                            snprintf(hi,64,"__mvu_%u",(unsigned)fn->nregs); if(func_add_reg(fn,hi,PTY_U32)!=0)return -1;
                        } else {
                            strcpy(lo,d0); strcpy(hi,d1);
                        }
                        // lo
                        { ptx_instr_t mi; memset(&mi,0,sizeof(mi)); mi.op=POP_CVT; mi.ty=PTY_U32; mi.ty2=PTY_U64; mi.line=opline; strcpy(mi.dst,lo); mi.src[0]=in.src[0]; mi.nsrc=1; if(block_add_instr(fn,bi,&mi,p,opline)!=0)return -1; }
                        // hi = shr.u64 src, 32 then cvt
                        { ptx_instr_t mi; memset(&mi,0,sizeof(mi)); mi.op=POP_SHR; mi.ty=PTY_U64; mi.line=opline; strcpy(mi.dst,t0); mi.src[0]=in.src[0]; mi.src[1].kind=OPD_IMM; mi.src[1].imm=32; mi.nsrc=2; if(block_add_instr(fn,bi,&mi,p,opline)!=0)return -1; }
                        { ptx_instr_t mi; memset(&mi,0,sizeof(mi)); mi.op=POP_CVT; mi.ty=PTY_U32; mi.ty2=PTY_U64; mi.line=opline; strcpy(mi.dst,hi); mi.src[0].kind=OPD_REG; strcpy(mi.src[0].reg,t0); mi.nsrc=1; if(block_add_instr(fn,bi,&mi,p,opline)!=0)return -1; }
                        if (guarded) {
                            { ptx_instr_t mi; memset(&mi,0,sizeof(mi)); mi.op=POP_MOV; mi.ty=PTY_U32; mi.line=opline; strcpy(mi.dst,d0); mi.src[0].kind=OPD_REG; strcpy(mi.src[0].reg,lo); mi.nsrc=1; snprintf(mi.pred,sizeof(mi.pred),"%s",pred); mi.pred_neg=pred_neg; if(block_add_instr(fn,bi,&mi,p,opline)!=0)return -1; }
                            { ptx_instr_t mi; memset(&mi,0,sizeof(mi)); mi.op=POP_MOV; mi.ty=PTY_U32; mi.line=opline; strcpy(mi.dst,d1); mi.src[0].kind=OPD_REG; strcpy(mi.src[0].reg,hi); mi.nsrc=1; snprintf(mi.pred,sizeof(mi.pred),"%s",pred); mi.pred_neg=pred_neg; if(block_add_instr(fn,bi,&mi,p,opline)!=0)return -1; }
                        }
                        return 0;
                    } else { // b32 -> u16 pair
                        int guarded = pred[0] || pred_neg;
                        char t0[64], lo[64], hi[64];
                        if (strcmp(d0, "RZ") == 0) {
                            snprintf(d0, 64, "__mvs_%u", (unsigned)fn->nregs);
                            if (func_add_reg(fn, d0, PTY_U16) != 0) return -1;
                        }
                        if (strcmp(d1, "RZ") == 0) {
                            snprintf(d1, 64, "__mvs_%u", (unsigned)fn->nregs);
                            if (func_add_reg(fn, d1, PTY_U16) != 0) return -1;
                        }
                        snprintf(t0,64,"__mvu_%u",(unsigned)fn->nregs); if(func_add_reg(fn,t0,PTY_U32)!=0)return -1;
                        if (guarded) {
                            snprintf(lo,64,"__mvu_%u",(unsigned)fn->nregs); if(func_add_reg(fn,lo,PTY_U16)!=0)return -1;
                            snprintf(hi,64,"__mvu_%u",(unsigned)fn->nregs); if(func_add_reg(fn,hi,PTY_U16)!=0)return -1;
                        } else {
                            strcpy(lo,d0); strcpy(hi,d1);
                        }
                        { ptx_instr_t mi; memset(&mi,0,sizeof(mi)); mi.op=POP_MOV; mi.ty=PTY_U32; mi.line=opline; strcpy(mi.dst,t0); mi.src[0]=in.src[0]; mi.nsrc=1; if(block_add_instr(fn,bi,&mi,p,opline)!=0)return -1; }
                        { ptx_instr_t mi; memset(&mi,0,sizeof(mi)); mi.op=POP_CVT; mi.ty=PTY_U16; mi.ty2=PTY_U32; mi.line=opline; strcpy(mi.dst,lo); mi.src[0].kind=OPD_REG; strcpy(mi.src[0].reg,t0); mi.nsrc=1; if(block_add_instr(fn,bi,&mi,p,opline)!=0)return -1; }
                        { ptx_instr_t mi; memset(&mi,0,sizeof(mi)); mi.op=POP_SHR; mi.ty=PTY_U32; mi.line=opline; strcpy(mi.dst,t0); mi.src[0]=in.src[0]; mi.src[1].kind=OPD_IMM; mi.src[1].imm=16; mi.nsrc=2; if(block_add_instr(fn,bi,&mi,p,opline)!=0)return -1; }
                        { ptx_instr_t mi; memset(&mi,0,sizeof(mi)); mi.op=POP_CVT; mi.ty=PTY_U16; mi.ty2=PTY_U32; mi.line=opline; strcpy(mi.dst,hi); mi.src[0].kind=OPD_REG; strcpy(mi.src[0].reg,t0); mi.nsrc=1; if(block_add_instr(fn,bi,&mi,p,opline)!=0)return -1; }
                        if (guarded) {
                            { ptx_instr_t mi; memset(&mi,0,sizeof(mi)); mi.op=POP_MOV; mi.ty=PTY_U16; mi.line=opline; strcpy(mi.dst,d0); mi.src[0].kind=OPD_REG; strcpy(mi.src[0].reg,lo); mi.nsrc=1; snprintf(mi.pred,sizeof(mi.pred),"%s",pred); mi.pred_neg=pred_neg; if(block_add_instr(fn,bi,&mi,p,opline)!=0)return -1; }
                            { ptx_instr_t mi; memset(&mi,0,sizeof(mi)); mi.op=POP_MOV; mi.ty=PTY_U16; mi.line=opline; strcpy(mi.dst,d1); mi.src[0].kind=OPD_REG; strcpy(mi.src[0].reg,hi); mi.nsrc=1; snprintf(mi.pred,sizeof(mi.pred),"%s",pred); mi.pred_neg=pred_neg; if(block_add_instr(fn,bi,&mi,p,opline)!=0)return -1; }
                        }
                        return 0;
                    }
                }
            }
            ptx_unget(&p->lx, &bt);
        }
        // Whole-vector 32-bit move INTO a .vN register base
        // (`mov.b32 v, s` where v is a .v2 of 8/16-bit elements, then
        // `v.x` / `v.y` reads): distribute the 32 bits across the
        // components little-endian (.x = low half). Packed widths that
        // do not fit one 32-bit word stay clean errors.
        {
            ptx_tok_t t0;
            if (next(p, &t0) != 0)
                return -1;
            ptx_vecdecl_t *vd = t0.kind == T_IDENT
                                    ? vec_find(fn, t0.text)
                                    : NULL;
            if (vd && vd->n == 2 && vd->elem == PTY_U16 &&
                strcmp(parts[1], "b32") == 0) {
                char base[256];
                if (copy_name(base, sizeof(base), t0.text, p, t0.line) != 0)
                    return -1;
                if (expect_comma(p) != 0 ||
                    parse_operand(p, &in.src[0]) != 0)
                    return -1;
                if (in.src[0].kind != OPD_REG &&
                    in.src[0].kind != OPD_IMM) {
                    perr(p, opline, "whole-vector mov needs a register or immediate source");
                    return -1;
                }
                if (expect_punct(p, ';', "after vector mov") != 0)
                    return -1;
                size_t lb = strlen(base);
                if (lb + 2 >= sizeof(base)) {
                    perr(p, opline, "name too long");
                    return -1;
                }
                for (int i = 0; i < 2; i++) {
                    char comp[256];
                    memcpy(comp, base, lb);
                    comp[lb] = '_';
                    comp[lb + 1] = "xyzw"[i];
                    comp[lb + 2] = '\0';
                    char tmp[64] = { 0 };
                    if (i == 1) {
                        // No 16-bit shift in PTX: shift in u32 first.
                        snprintf(tmp, sizeof(tmp), "__vmv_%u",
                                 (unsigned)fn->nregs);
                        if (func_add_reg(fn, tmp, PTY_U32) != 0) {
                            perr(p, opline, "out of memory");
                            return -1;
                        }
                        ptx_instr_t si;
                        memset(&si, 0, sizeof(si));
                        si.op = POP_SHR;
                        si.ty = PTY_U32;
                        si.line = opline;
                        if (copy_name(si.dst, sizeof(si.dst), tmp, p,
                                      opline) != 0)
                            return -1;
                        si.src[0] = in.src[0];
                        si.src[1].kind = OPD_IMM;
                        si.src[1].imm = 16;
                        si.nsrc = 2;
                        if (block_add_instr(fn, bi, &si, p, opline) != 0)
                            return -1;
                    }
                    // Narrowing cvt supplies the 16-bit store: the
                    // components are .b16 regs, so a 32-bit source must
                    // be truncated rather than moved.
                    ptx_instr_t mi;
                    memset(&mi, 0, sizeof(mi));
                    mi.op = POP_CVT;
                    mi.ty = PTY_U16;
                    mi.ty2 = PTY_U32;
                    mi.line = opline;
                    if (copy_name(mi.dst, sizeof(mi.dst), comp, p, opline) !=
                        0)
                        return -1;
                    if (i == 1) {
                        if (copy_name(mi.src[0].reg, sizeof(mi.src[0].reg),
                                      tmp, p, opline) != 0)
                            return -1;
                    } else {
                        mi.src[0] = in.src[0];
                    }
                    mi.nsrc = 1;
                    if (block_add_instr(fn, bi, &mi, p, opline) != 0)
                        return -1;
                }
                return 0;
            }
            ptx_unget(&p->lx, &t0);
        }
        // Predicate moves (vote masks): register, !register, or 0/1.
        // The negation (if any) is carried in bneg like setp sources.
        if (strcmp(parts[1], "pred") == 0) {
            in.ty = PTY_PRED;
            if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
                return -1;
            if (expect_comma(p) != 0)
                return -1;
            if (at_punct(p, '!'))
                in.bneg = 1;
            if (parse_operand(p, &in.src[0]) != 0)
                return -1;
            // Integer immediates are booleanized like ptxas: 0 is
            // false, any nonzero value (-1 included) is true.
            if (in.src[0].kind != OPD_REG &&
                (in.src[0].kind != OPD_IMM || in.src[0].imm_is_float)) {
                perr(p, opline, "mov.pred needs a predicate register or integer");
                return -1;
            }
            in.nsrc = 1;
            if (expect_punct(p, ';', "after mov") != 0)
                return -1;
            return block_add_instr(fn, bi, &in, p, opline);
        }
        if (map_ty(parts[1], &in.ty) != 0 || in.ty == PTY_PRED) {
            perr(p, opline, "unsupported mov type");
            return -1;
        }
        // PTX mov has no 8-bit or .f16 forms (narrow values come
        // from ld/cvt); u16 and wider copy as usual. Packed and bf16
        // have no mov form either (halves move as .b32/.b16 bits).
        if (in.ty == PTY_U8 || in.ty == PTY_F16 || in.ty == PTY_BF16 ||
            in.ty == PTY_F16X2 || in.ty == PTY_BF16X2) {
            perr(p, opline, "mov has no .u8/.f16/.bf16/packed form (use ld/cvt)");
            return -1;
        }
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0)
            return -1;
        // Whole-vector bit move (`mov.b64 d, v` for .v2.u32, ptxas
        // parity): packs hi<<32|lo via converts+shift+or. Other
        // vector-to-scalar shapes stay clean errors.
        if (strcmp(parts[1], "b64") == 0) {
            ptx_tok_t st;
            if (next(p, &st) != 0)
                return -1;
            ptx_vecdecl_t *sv = st.kind == T_IDENT ?
                                vec_find(fn, st.text) : NULL;
            if (sv && sv->n == 2 && sv->elem == PTY_U32) {
                char dst[64], hi[64], lo[64];
                size_t lb = strlen(st.text);
                if (lb + 2 >= sizeof(hi)) {
                    perr(p, opline, "name too long");
                    return -1;
                }
                if (copy_name(dst, sizeof(dst), in.dst, p,
                              opline) != 0)
                    return -1;
                memcpy(hi, st.text, lb);
                hi[lb] = '_';
                hi[lb + 1] = 'y';
                hi[lb + 2] = '\0';
                memcpy(lo, st.text, lb);
                lo[lb] = '_';
                lo[lb + 1] = 'x';
                lo[lb + 2] = '\0';
                if (expect_punct(p, ';', "after vector mov") != 0)
                    return -1;
                char t0[64], t1[64], t2[64];
                snprintf(t0, sizeof(t0), "__vmv_%u",
                         (unsigned)fn->nregs);
                if (func_add_reg(fn, t0, PTY_U64) != 0) {
                    perr(p, opline, "out of memory");
                    return -1;
                }
                snprintf(t1, sizeof(t1), "__vmv_%u",
                         (unsigned)fn->nregs);
                if (func_add_reg(fn, t1, PTY_U64) != 0) {
                    perr(p, opline, "out of memory");
                    return -1;
                }
                snprintf(t2, sizeof(t2), "__vmv_%u",
                         (unsigned)fn->nregs);
                if (func_add_reg(fn, t2, PTY_U64) != 0) {
                    perr(p, opline, "out of memory");
                    return -1;
                }
                // cvt.u64.u32 hi/lo, shl.b64 hi,32, or.b64 (all
                // in-subset spellings).
                {
                    ptx_instr_t mi;
                    memset(&mi, 0, sizeof(mi));
                    mi.op = POP_CVT;
                    mi.ty = PTY_U64;
                    mi.ty2 = PTY_U32;
                    mi.line = opline;
                    if (copy_name(mi.dst, sizeof(mi.dst), t0, p,
                                  opline) != 0)
                        return -1;
                    mi.src[0].kind = OPD_REG;
                    if (copy_name(mi.src[0].reg, sizeof(mi.src[0].reg),
                                  hi, p, opline) != 0)
                        return -1;
                    mi.nsrc = 1;
                    if (block_add_instr(fn, bi, &mi, p, opline) != 0)
                        return -1;
                }
                {
                    ptx_instr_t mi;
                    memset(&mi, 0, sizeof(mi));
                    mi.op = POP_CVT;
                    mi.ty = PTY_U64;
                    mi.ty2 = PTY_U32;
                    mi.line = opline;
                    if (copy_name(mi.dst, sizeof(mi.dst), t1, p,
                                  opline) != 0)
                        return -1;
                    mi.src[0].kind = OPD_REG;
                    if (copy_name(mi.src[0].reg, sizeof(mi.src[0].reg),
                                  lo, p, opline) != 0)
                        return -1;
                    mi.nsrc = 1;
                    if (block_add_instr(fn, bi, &mi, p, opline) != 0)
                        return -1;
                }
                {
                    ptx_instr_t mi;
                    memset(&mi, 0, sizeof(mi));
                    mi.op = POP_SHL;
                    mi.ty = PTY_U64;
                    mi.line = opline;
                    if (copy_name(mi.dst, sizeof(mi.dst), t2, p,
                                  opline) != 0)
                        return -1;
                    mi.src[0].kind = OPD_REG;
                    if (copy_name(mi.src[0].reg, sizeof(mi.src[0].reg),
                                  t0, p, opline) != 0)
                        return -1;
                    mi.src[1].kind = OPD_IMM;
                    mi.src[1].imm = 32;
                    mi.nsrc = 2;
                    if (block_add_instr(fn, bi, &mi, p, opline) != 0)
                        return -1;
                }
                {
                    ptx_instr_t mi;
                    memset(&mi, 0, sizeof(mi));
                    mi.op = POP_OR;
                    mi.ty = PTY_U64;
                    mi.line = opline;
                    if (copy_name(mi.dst, sizeof(mi.dst), dst, p,
                                  opline) != 0)
                        return -1;
                    mi.src[0].kind = OPD_REG;
                    if (copy_name(mi.src[0].reg, sizeof(mi.src[0].reg),
                                  t2, p, opline) != 0)
                        return -1;
                    mi.src[1].kind = OPD_REG;
                    if (copy_name(mi.src[1].reg, sizeof(mi.src[1].reg),
                                  t1, p, opline) != 0)
                        return -1;
                    mi.nsrc = 2;
                    if (block_add_instr(fn, bi, &mi, p, opline) != 0)
                        return -1;
                }
                return 0;
            }
            ptx_unget(&p->lx, &st);
        }
        if (at_punct(p, '{')) {
            // Brace-vector pack (ZLUDA corpus): mov.b64 d, {lo, hi}
            // and mov.b32 d, {lo16, hi16} (first operand is LOW).
            if (strcmp(parts[1], "b64") != 0 &&
                strcmp(parts[1], "b32") != 0) {
                perr(p, opline, "vector mov supports .b64/.b32 only");
                return -1;
            }
            if (parse_operand(p, &in.src[0]) != 0)
                return -1;
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
                return -1;
            if (expect_punct(p, '}', "in vector mov") != 0)
                return -1;
            in.nsrc = 2;
            if (expect_punct(p, ';', "after mov") != 0)
                return -1;
            return block_add_instr(fn, bi, &in, p, opline);
        }
        if (parse_operand(p, &in.src[0]) != 0)
            return -1;
        in.nsrc = 1;
        if (expect_punct(p, ';', "after mov") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if ((strcmp(parts[0], "add") == 0 || strcmp(parts[0], "sub") == 0) &&
        np == 3 && strcmp(parts[1], "sat") == 0 &&
        strcmp(parts[2], "f32") != 0 && strcmp(parts[2], "f64") != 0 &&
        strcmp(parts[2], "f16") != 0 && strcmp(parts[2], "f16x2") != 0 &&
        strcmp(parts[2], "bf16") != 0 && strcmp(parts[2], "bf16x2") != 0) {
        // Saturating add/sub (wrap-free): signed clamps to the type
        // range, unsigned clamps to [0, MAX] (add overflow / sub
        // underflow). Packed x2/x4 forms clamp each lane independently.
        // The u32/packed sat forms are family-specific in newer PTX;
        // this target-agnostic subset accepts them everywhere.
        in.sat_sub = parts[0][0] == 's';
        in.pack = 0;
        // PTX ISA 9.3 saturating type sets: add.sat covers u32/s32 and
        // all four packed types; sub.sat covers s32 and the byte-packed
        // .u8x4/.s8x4 only.
        if (strcmp(parts[2], "s32") == 0) {
            in.ty = PTY_U32;
            in.sign = 1;
        } else if (!in.sat_sub && strcmp(parts[2], "u32") == 0) {
            in.ty = PTY_U32;
            in.sign = 0;
        } else if (!in.sat_sub &&
                   (strcmp(parts[2], "u16x2") == 0 ||
                    strcmp(parts[2], "s16x2") == 0)) {
            in.ty = PTY_U32;
            in.pack = 2;
            in.sign = parts[2][0] == 's';
        } else if (strcmp(parts[2], "u8x4") == 0 ||
                   strcmp(parts[2], "s8x4") == 0) {
            in.ty = PTY_U32;
            in.pack = 4;
            in.sign = parts[2][0] == 's';
        } else {
            perr(p, opline, in.sat_sub
                              ? "sub.sat supports .s32/.u8x4/.s8x4 only"
                              : "add.sat supports .u32/.s32/.u16x2/.s16x2/.u8x4/.s8x4 only");
            return -1;
        }
        in.op = POP_ADDSAT;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (expect_punct(p, ';', "after saturating add/sub") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "testp") == 0 && np == 3) {
        // testp.{finite,infinite,number,notanumber,normal,subnormal}.f32/f64
        int op;
        if (strcmp(parts[1], "finite") == 0)
            op = 0;
        else if (strcmp(parts[1], "infinite") == 0)
            op = 1;
        else if (strcmp(parts[1], "number") == 0)
            op = 2;
        else if (strcmp(parts[1], "notanumber") == 0)
            op = 3;
        else if (strcmp(parts[1], "normal") == 0)
            op = 4;
        else if (strcmp(parts[1], "subnormal") == 0)
            op = 5;
        else {
            perr(p, opline, "testp supports .finite/.infinite/.number/.notanumber/.normal/.subnormal");
            return -1;
        }
        if (strcmp(parts[2], "f32") == 0) {
            in.ty = PTY_F32;
        } else if (strcmp(parts[2], "f64") == 0) {
            in.ty = PTY_F64;
        } else {
            perr(p, opline, "testp supports .f32/.f64 only");
            return -1;
        }
        in.op = POP_TESTP;
        in.arg = op;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        in.nsrc = 1;
        if (expect_punct(p, ';', "after testp") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "copysign") == 0 && np == 2) {
        // d = sign(a) | magnitude(b) (operand order pinned by ZLUDA's
        // vector: the FIRST source supplies the sign bit).
        if (strcmp(parts[1], "f32") == 0) {
            in.ty = PTY_F32;
        } else if (strcmp(parts[1], "f64") == 0) {
            in.ty = PTY_F64;
        } else {
            perr(p, opline, "copysign supports .f32/.f64 only");
            return -1;
        }
        in.op = POP_COPYSIGN;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (expect_punct(p, ';', "after copysign") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if ((strcmp(parts[0], "min") == 0 || strcmp(parts[0], "max") == 0) &&
        np == 3 && strcmp(parts[1], "NaN") == 0) {
        // NaN-propagating min/max: any NaN input produces a canonical
        // quiet NaN; otherwise the ordered result.
        if (strcmp(parts[2], "f16") == 0)
            in.ty = PTY_F16;
        else if (strcmp(parts[2], "f16x2") == 0)
            in.ty = PTY_F16X2;
        else if (strcmp(parts[2], "bf16") == 0)
            in.ty = PTY_BF16;
        else if (strcmp(parts[2], "bf16x2") == 0)
            in.ty = PTY_BF16X2;
        else if (strcmp(parts[2], "f32") == 0)
            in.ty = PTY_F32;
        else {
            perr(p, opline, "min.NaN/max.NaN supports .f16/.f16x2/.bf16/.bf16x2/.f32 (no .f64 NaN modifier)");
            return -1;
        }
        in.op = POP_MINNAN;
        in.is_max = parts[0][1] == 'a';
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (expect_punct(p, ';', "after min.NaN/max.NaN") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (((strcmp(parts[0], "add") == 0 || strcmp(parts[0], "sub") == 0) &&
         np == 3 && strcmp(parts[1], "cc") == 0)) {
        // Extended arithmetic, carry-out only (ZLUDA corpus): full
        // 64-bit sum/difference, low 32 to dst, carry/borrow to CF.
        // SUB stores NOT-borrow (reverse carry): subc reads it back
        // inverted, matching ZLUDA's lowering and its vectors.
        int is_add = parts[0][0] == 'a';
        if (strcmp(parts[2], "u32") == 0 || strcmp(parts[2], "s32") == 0 ||
            strcmp(parts[2], "b32") == 0) {
            in.ty = PTY_U32;
        } else if (strcmp(parts[2], "u64") == 0 ||
                   strcmp(parts[2], "s64") == 0 ||
                   strcmp(parts[2], "b64") == 0) {
            in.ty = PTY_U64;
        } else {
            perr(p, opline, "extended add/sub supports .u32/.s32/.u64/.s64 (and .b aliases)");
            return -1;
        }
        in.sign = parts[2][0] == 's';
        in.op = is_add ? POP_ADD : POP_SUB;
        in.cc_out = 1;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (expect_punct(p, ';', "after extended add/sub") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if ((strcmp(parts[0], "addc") == 0 || strcmp(parts[0], "subc") == 0) &&
        (np == 2 || (np == 3 && strcmp(parts[1], "cc") == 0))) {
        // Carry-in add/sub (optionally carrying out with .cc).
        int is_add = parts[0][0] == 'a';
        const char *ts = np == 3 ? parts[2] : parts[1];
        if (strcmp(ts, "u32") == 0 || strcmp(ts, "s32") == 0 ||
            strcmp(ts, "b32") == 0) {
            in.ty = PTY_U32;
        } else if (strcmp(ts, "u64") == 0 || strcmp(ts, "s64") == 0 ||
                   strcmp(ts, "b64") == 0) {
            in.ty = PTY_U64;
        } else {
            perr(p, opline, "extended addc/subc supports .u32/.s32/.u64/.s64 (and .b aliases)");
            return -1;
        }
        in.sign = ts[0] == 's';
        in.op = is_add ? POP_ADD : POP_SUB;
        in.cc_in = 1;
        in.cc_out = np == 3;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (expect_punct(p, ';', "after extended addc/subc") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (((strcmp(parts[0], "mad") == 0 && np == 4 &&
          strcmp(parts[2], "cc") == 0) ||
         (strcmp(parts[0], "madc") == 0 && np >= 3)) &&
        (strcmp(parts[1], "lo") == 0 ||
         strcmp(parts[1], "hi") == 0)) {
        // mad.{hi,lo}.cc.type (carry-out) and
        // madc.{hi,lo}{.cc}.type (carry-in, optional carry-out).
        // Types: u32/s32/u64/s64.
        int is_madc = strcmp(parts[0], "madc") == 0;
        int hi = -1, cc_out = 0;
        const char *ts = NULL;
        if (np >= 3 && (strcmp(parts[1], "lo") == 0 ||
                        strcmp(parts[1], "hi") == 0)) {
            hi = parts[1][0] == 'h';
            if (np == 3 && is_madc) {
                ts = parts[2];
            } else if (np == 4 && strcmp(parts[2], "cc") == 0) {
                cc_out = 1;
                ts = parts[3];
            }
        }
        if (!ts) {
            perr(p, opline,
                 is_madc ? "madc syntax is madc.{hi,lo}{.cc}.type"
                         : "mad.cc syntax is mad.{hi,lo}.cc.type");
            return -1;
        }
        if (strcmp(ts, "u32") == 0 || strcmp(ts, "s32") == 0) {
            in.ty = PTY_U32;
        } else if (strcmp(ts, "u64") == 0 || strcmp(ts, "s64") == 0) {
            in.ty = PTY_U64;
        } else {
            perr(p, opline, "extended mad supports .u32/.s32/.u64/.s64 only");
            return -1;
        }
        in.sign = ts[0] == 's';
        in.op = POP_MADLO;
        in.is_hi = hi;
        in.cc_in = is_madc;
        in.cc_out = cc_out;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        for (int i = 0; i < 3; i++) {
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[i]) != 0)
                return -1;
            if (in.src[i].kind == OPD_SPECIAL) {
                perr(p, opline, "special register argument not allowed for extended mad");
                return -1;
            }
        }
        in.nsrc = 3;
        if (expect_punct(p, ';', "after extended mad") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if ((strcmp(parts[0], "add") == 0 || strcmp(parts[0], "sub") == 0 ||
         strcmp(parts[0], "mul") == 0) &&
        np >= 3 &&
        (strcmp(parts[np - 1], "f32") == 0 ||
         strcmp(parts[np - 1], "f32x2") == 0 ||
         strcmp(parts[np - 1], "f64") == 0 ||
         strcmp(parts[np - 1], "f16") == 0 ||
         strcmp(parts[np - 1], "f16x2") == 0 ||
         strcmp(parts[np - 1], "bf16") == 0 ||
         strcmp(parts[np - 1], "bf16x2") == 0)) {
        // add/sub/mul{.rnd}{.ftz}{.sat}.f32, ...f32x2, and ...{.rnd}.f64.
        // Directed f32 uses the exact FMA/integer machinery; directed
        // f64 is not implemented. .ftz/.sat are f32-only.
        int is_add = strcmp(parts[0], "add") == 0;
        int is_sub = strcmp(parts[0], "sub") == 0;
        int rnd = 0, ftz = 0, sat = 0;
        for (int pi = 1; pi < np - 1; pi++) {
            if (strcmp(parts[pi], "rn") == 0)
                rnd = 0;
            else if (strcmp(parts[pi], "rz") == 0)
                rnd = 1;
            else if (strcmp(parts[pi], "rm") == 0)
                rnd = 2;
            else if (strcmp(parts[pi], "rp") == 0)
                rnd = 3;
            else if (strcmp(parts[pi], "ftz") == 0)
                ftz = 1;
            else if (strcmp(parts[pi], "sat") == 0)
                sat = 1;
            else {
                perr(p, opline, "unsupported float add/sub/mul modifier");
                return -1;
            }
        }
        const char *ts = parts[np - 1];
        int is_x2 = strcmp(ts, "f32x2") == 0;
        if (strcmp(ts, "f64") == 0)
            in.ty = PTY_F64;
        else if (strcmp(ts, "f32") == 0)
            in.ty = PTY_F32;
        else if (is_x2)
            in.ty = PTY_U64;
        else if (strcmp(ts, "f16") == 0)
            in.ty = PTY_F16;
        else if (strcmp(ts, "f16x2") == 0)
            in.ty = PTY_F16X2;
        else if (strcmp(ts, "bf16") == 0)
            in.ty = PTY_BF16;
        else
            in.ty = PTY_BF16X2;
        if (in.ty == PTY_F64 && (ftz || sat)) {
            perr(p, opline, "f64 add/sub/mul does not take .ftz/.sat");
            return -1;
        }
        if (is_x2 && sat) {
            perr(p, opline, "f32x2 add/sub/mul does not take .sat");
            return -1;
        }
        if ((in.ty == PTY_BF16 || in.ty == PTY_BF16X2) && (ftz || sat)) {
            perr(p, opline, "bf16 add/sub/mul does not take .ftz/.sat");
            return -1;
        }
        if ((in.ty == PTY_F16 || in.ty == PTY_F16X2 ||
             in.ty == PTY_BF16 || in.ty == PTY_BF16X2) && rnd != 0) {
            perr(p, opline, "half add/sub/mul supports only .rn");
            return -1;
        }
        in.op = is_add ? POP_ADD : is_sub ? POP_SUB : POP_MULLO;
        in.frnd = rnd;
        in.ftz = ftz;
        in.sat = sat;
        in.f32x2 = is_x2;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (expect_punct(p, ';', "after float arithmetic") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if ((strcmp(parts[0], "add") == 0 || strcmp(parts[0], "sub") == 0) && np == 2) {
        int is_add = parts[0][0] == 'a';
        in.op = is_add ? POP_ADD : POP_SUB;
        in.pack = 0;
        // Packed integer types operate on b32 holders with 2 half-word
        // or 4 byte lanes (lane arithmetic wraps). PTX ISA 9.3/ptxas:
        // add and sub both accept .u16x2/.s16x2/.u8x4/.s8x4; the packed
        // byte forms require sm_120f.
        int p16 = strcmp(parts[1], "u16x2") == 0 ||
                  strcmp(parts[1], "s16x2") == 0;
        int p8 = strcmp(parts[1], "u8x4") == 0 ||
                 strcmp(parts[1], "s8x4") == 0;
        if (p16) {
            in.ty = PTY_U32;
            in.pack = 2;
            in.sign = parts[1][0] == 's';
        } else if (p8) {
            in.ty = PTY_U32;
            in.pack = 4;
            in.sign = parts[1][0] == 's';
        } else {
            if (map_ty(parts[1], &in.ty) != 0 || in.ty == PTY_PRED) {
                perr(p, opline, "unsupported arithmetic type");
                return -1;
            }
            if (in.ty != PTY_U32 && in.ty != PTY_F32 && in.ty != PTY_F64 &&
                in.ty != PTY_U64 && in.ty != PTY_F16 && in.ty != PTY_U16 &&
                in.ty != PTY_U8 && in.ty != PTY_F16X2 && in.ty != PTY_BF16 &&
                in.ty != PTY_BF16X2) {
                perr(p, opline, "unsupported arithmetic type (want .u8/.u16/.u32/.u64/.f16/.f16x2/.bf16/.bf16x2/.f32/.f64 or packed int)");
                return -1;
            }
        }
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (expect_punct(p, ';', "after add/sub") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "mul") == 0 && np == 2) {
        // Plain mul is only defined for floats (integer multiply needs .lo).
        if (strcmp(parts[1], "f32") != 0 && strcmp(parts[1], "f64") != 0 &&
            strcmp(parts[1], "f16") != 0 && strcmp(parts[1], "f16x2") != 0 &&
            strcmp(parts[1], "bf16") != 0 && strcmp(parts[1], "bf16x2") != 0) {
            perr(p, opline, "integer mul needs .lo (mul.lo.u32)");
            return -1;
        }
        in.op = POP_MULLO;
        in.ty = strcmp(parts[1], "f64") == 0   ? PTY_F64
                : strcmp(parts[1], "f32") == 0 ? PTY_F32
                : strcmp(parts[1], "f16") == 0 ? PTY_F16
                : strcmp(parts[1], "f16x2") == 0 ? PTY_F16X2
                : strcmp(parts[1], "bf16") == 0 ? PTY_BF16
                                                : PTY_BF16X2;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (expect_punct(p, ';', "after mul.f32") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "mul") == 0 && np == 3 && strcmp(parts[1], "lo") == 0) {
        in.op = POP_MULLO;
        // .u32 and .s32 wrap identically; 64-bit .lo is the low 64
        // bits of the full product (same for .u64/.s64); .u16/.s16 is
        // the low 16 bits (zero-extended canonical).
        if (strcmp(parts[2], "u32") != 0 && strcmp(parts[2], "s32") != 0 &&
            strcmp(parts[2], "s64") != 0 && strcmp(parts[2], "u64") != 0 &&
            strcmp(parts[2], "u16") != 0 && strcmp(parts[2], "s16") != 0 &&
            strcmp(parts[2], "u8") != 0 && strcmp(parts[2], "s8") != 0 &&
            strcmp(parts[2], "b8") != 0) {
            perr(p, opline, "mul.lo supports .u8/.s8/.u16/.s16/.u32/.s32/.u64/.s64 only");
            return -1;
        }
        in.ty = (strcmp(parts[2], "s64") == 0 ||
                 strcmp(parts[2], "u64") == 0)
                ? PTY_U64
                : (strcmp(parts[2], "u16") == 0 ||
                   strcmp(parts[2], "s16") == 0)
                  ? PTY_U16
                  : (strcmp(parts[2], "u8") == 0 ||
                     strcmp(parts[2], "s8") == 0 ||
                     strcmp(parts[2], "b8") == 0)
                    ? PTY_U8
                    : PTY_U32;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (expect_punct(p, ';', "after mul") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "mul") == 0 && np == 3 && strcmp(parts[1], "wide") == 0) {
        in.op = POP_MULWIDE;
        in.sign = 0;
        if (strcmp(parts[2], "u16") == 0 || strcmp(parts[2], "s16") == 0) {
            in.ty = PTY_U16;
            in.sign = parts[2][0] == 's';
        } else if (strcmp(parts[2], "u32") == 0 ||
                   strcmp(parts[2], "s32") == 0) {
            in.ty = PTY_U32;
            in.sign = parts[2][0] == 's';
        } else {
            perr(p, opline, "mul.wide supports .u16/.s16/.u32/.s32 only");
            return -1;
        }
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (expect_punct(p, ';', "after mul.wide") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "mad") == 0 && np == 3 && strcmp(parts[1], "wide") == 0) {
        // Wide multiply-add (exact 64-bit): d = (64)a * (64)b + c.
        if (strcmp(parts[2], "u16") == 0 || strcmp(parts[2], "s16") == 0) {
            in.ty = PTY_U16;
            in.sign = parts[2][0] == 's';
        } else if (strcmp(parts[2], "u32") == 0 ||
                   strcmp(parts[2], "s32") == 0) {
            in.ty = PTY_U32;
            in.sign = parts[2][0] == 's';
        } else {
            perr(p, opline, "mad.wide supports .u16/.s16/.u32/.s32 only");
            return -1;
        }
        in.op = POP_MADWIDE;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        for (int i = 0; i < 3; i++) {
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[i]) != 0)
                return -1;
        }
        in.nsrc = 3;
        if (expect_punct(p, ';', "after mad.wide") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "mul") == 0 && np == 3 && strcmp(parts[1], "hi") == 0) {
        // High half of the full product: 64x64 needs the 32-bit
        // schoolbook (exact); 32-bit forms widen first.
        if (strcmp(parts[2], "u64") == 0 ||
            strcmp(parts[2], "s64") == 0) {
            in.ty = PTY_U64;
            in.sign = parts[2][0] == 's';
        } else if (strcmp(parts[2], "u32") == 0 ||
                   strcmp(parts[2], "s32") == 0) {
            in.ty = PTY_U32;
            in.sign = parts[2][0] == 's';
        } else if (strcmp(parts[2], "u16") == 0 ||
                   strcmp(parts[2], "s16") == 0) {
            in.ty = PTY_U16;
            in.sign = parts[2][0] == 's';
        } else {
            perr(p, opline, "mul.hi supports .u16/.s16/.u32/.s32/.u64/.s64 only");
            return -1;
        }
        in.op = POP_MULHI;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (expect_punct(p, ';', "after mul.hi") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "acuda") == 0 && np == 5 &&
        strcmp(parts[1], "dp4a") == 0 && strcmp(parts[2], "sat") == 0 &&
        (strcmp(parts[3], "u32") == 0 || strcmp(parts[3], "s32") == 0) &&
        (strcmp(parts[4], "u32") == 0 || strcmp(parts[4], "s32") == 0)) {
        // ACuda extension: saturating four-way byte dot (OpSDotAccSat
        // family). ptxas has no .sat dp4a, so this is not PTX.
        in.op = POP_DP4A;
        in.ty = PTY_U32;
        in.sat = 1;
        in.sign = parts[3][0] == 's';
        in.arg = parts[4][0] == 's';
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        for (int i = 0; i < 3; i++) {
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[i]) != 0)
                return -1;
        }
        in.nsrc = 3;
        if (expect_punct(p, ';', "after acuda.dp4a.sat") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "dp4a") == 0 && np == 3 &&
        (strcmp(parts[1], "u32") == 0 || strcmp(parts[1], "s32") == 0) &&
        (strcmp(parts[2], "u32") == 0 || strcmp(parts[2], "s32") == 0)) {
        // Four-way byte dot plus accumulator; each operand independently
        // signed or unsigned.
        in.op = POP_DP4A;
        in.ty = PTY_U32;
        in.sign = parts[1][0] == 's';
        in.arg = parts[2][0] == 's';
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        for (int i = 0; i < 3; i++) {
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[i]) != 0)
                return -1;
        }
        in.nsrc = 3;
        if (expect_punct(p, ';', "after dp4a") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "dp2a") == 0 && np == 4 &&
        (strcmp(parts[1], "lo") == 0 || strcmp(parts[1], "hi") == 0) &&
        (strcmp(parts[2], "u32") == 0 || strcmp(parts[2], "s32") == 0) &&
        (strcmp(parts[3], "u32") == 0 || strcmp(parts[3], "s32") == 0)) {
        // Two-way dot (16-bit lanes of `a` with 8-bit lanes of `b`;
        // `.lo` takes b lanes 0-1, `.hi` lanes 2-3).
        in.op = POP_DP2A;
        in.ty = PTY_U32;
        in.is_hi = parts[1][0] == 'h';
        in.sign = parts[2][0] == 's';
        in.arg = parts[3][0] == 's';
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        for (int i = 0; i < 3; i++) {
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[i]) != 0)
                return -1;
        }
        in.nsrc = 3;
        if (expect_punct(p, ';', "after dp2a") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "sad") == 0 && np == 2) {        // Scalar sum of absolute differences: d = |a-b| + c (exact in
        // u64; magnitude never overflows unsigned).
        if (strcmp(parts[1], "u16") == 0 ||
            strcmp(parts[1], "s16") == 0) {
            in.ty = PTY_U16;
            in.sign = parts[1][0] == 's';
        } else if (strcmp(parts[1], "u32") == 0 ||
                   strcmp(parts[1], "s32") == 0) {
            in.ty = PTY_U32;
            in.sign = parts[1][0] == 's';
        } else if (strcmp(parts[1], "u64") == 0 ||
                   strcmp(parts[1], "s64") == 0) {
            in.ty = PTY_U64;
            in.sign = parts[1][0] == 's';
        } else {
            perr(p, opline, "sad supports .u16/.s16/.u32/.s32/.u64/.s64 only");
            return -1;
        }
        in.op = POP_SAD;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        for (int i = 0; i < 3; i++) {
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[i]) != 0)
                return -1;
        }
        in.nsrc = 3;
        if (expect_punct(p, ';', "after sad") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "cnot") == 0 && np == 2) {
        // C-style logical negation: d = (a == 0) ? 1 : 0.
        if (strcmp(parts[1], "b16") == 0)
            in.ty = PTY_U16;
        else if (strcmp(parts[1], "b32") == 0)
            in.ty = PTY_U32;
        else if (strcmp(parts[1], "b64") == 0)
            in.ty = PTY_U64;
        else {
            perr(p, opline, "cnot supports .b16/.b32/.b64 only");
            return -1;
        }
        in.op = POP_CNOT;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        in.nsrc = 1;
        if (expect_punct(p, ';', "after cnot") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "lop3") == 0 &&
        (np == 2 || np == 3) && strcmp(parts[np - 1], "b32") == 0) {
        // 3-input bitwise lookup table (immLut 0..255).  The BoolOp
        // forms add a predicate output and combine (d != 0) with q.
        int bop = 0;
        if (np == 3) {
            if (strcmp(parts[1], "or") == 0)
                bop = 1;
            else if (strcmp(parts[1], "and") == 0)
                bop = 2;
            else {
                perr(p, opline, "lop3.BoolOp supports .or/.and");
                return -1;
            }
        }
        in.op = POP_LOP3;
        in.ty = PTY_U32;
        in.lop3_boolop = bop;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (bop && at_punct(p, '|')) {
            ptx_tok_t qt;
            if (next(p, &qt) != 0)
                return -1;
            if (qt.kind != T_IDENT) {
                perr(p, qt.line, "expected predicate after '|'");
                return -1;
            }
            if (copy_name(in.dst2, sizeof(in.dst2), qt.text, p,
                          qt.line) != 0)
                return -1;
        }
        for (int i = 0; i < 3; i++) {
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[i]) != 0)
                return -1;
        }
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[3]) != 0)
            return -1;
        if (in.src[3].kind != OPD_IMM || in.src[3].imm_is_float ||
            in.src[3].imm > 255) {
            perr(p, opline, "lop3 immLut must be an integer constant 0..255");
            return -1;
        }
        in.arg = (int)in.src[3].imm;
        in.nsrc = 3;
        if (bop) {
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[4]) != 0)
                return -1;
            in.nsrc = 5;
        }
        if (expect_punct(p, ';', "after lop3") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "slct") == 0 &&
        (np == 3 || (np == 4 && strcmp(parts[1], "ftz") == 0))) {
        // slct.dtype.s32 / slct{.ftz}.dtype.f32: d = (c >= 0) ? a : b.
        int ftz = np == 4;
        const char *dts = parts[np - 2];
        const char *cts = parts[np - 1];
        if (map_ty(dts, &in.ty) != 0) {
            perr(p, opline, "bad slct destination type");
            return -1;
        }
        if (strcmp(cts, "s32") == 0) {
            in.ty2 = PTY_U32;
            in.sign = 1;
            if (ftz) {
                perr(p, opline, "slct.ftz requires the .f32 compare form");
                return -1;
            }
        } else if (strcmp(cts, "f32") == 0) {
            in.ty2 = PTY_F32;
            in.sign = 0;
        } else {
            perr(p, opline, "slct compare operand must be .s32 or .f32");
            return -1;
        }
        in.op = POP_SLCT;
        in.ftz = ftz;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        for (int i = 0; i < 3; i++) {
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[i]) != 0)
                return -1;
        }
        in.nsrc = 3;
        if (expect_punct(p, ';', "after slct") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "prmt") == 0 && np == 2) {
        // Byte permute (NVIDIA semantics, ZLUDA-verified): each output
        // byte i takes concat byte [nibble i] (a low), or the sign bit
        // of concat[nibble & 7] replicated when the nibble is >= 8.
        if (strcmp(parts[1], "b32") != 0 &&
            strcmp(parts[1], "u32") != 0 &&
            strcmp(parts[1], "s32") != 0) {
            perr(p, opline, "only prmt.b32 (and .u32/.s32 aliases) supported");
            return -1;
        }
        in.op = POP_PRMT;
        in.ty = PTY_U32;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        for (int i = 0; i < 3; i++) {
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[i]) != 0)
                return -1;
        }
        in.nsrc = 3;
        if (expect_punct(p, ';', "after prmt") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "mul24") == 0 && np == 3 &&
        (strcmp(parts[1], "lo") == 0 || strcmp(parts[1], "hi") == 0)) {
        // 24-bit multiply: operands are the low 24 bits (.u32) or
        // sign-extended 24-bit values (.s32); upper source bits are
        // ignored. .hi returns bits 47..16 (NVIDIA semantics, unlike
        // AMD's mulhi which returns 63..32).
        if (strcmp(parts[2], "u32") == 0) {
            in.ty = PTY_U32;
            in.sign = 0;
        } else if (strcmp(parts[2], "s32") == 0) {
            in.ty = PTY_U32;
            in.sign = 1;
        } else {
            perr(p, opline, "mul24 supports .u32/.s32 only");
            return -1;
        }
        in.op = POP_MUL24;
        in.is_hi = parts[1][0] == 'h';
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (expect_punct(p, ';', "after mul24") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "mad24") == 0) {
        // mad24.{lo,hi}.{u32,s32} and mad24.hi.sat.s32:
        // d = (a*b)<47..16 or <31..0> + c, 24-bit inputs.
        int hi = 0, sat = 0;
        const char *ts = NULL;
        if (np == 3 && (strcmp(parts[1], "lo") == 0 ||
                        strcmp(parts[1], "hi") == 0)) {
            hi = parts[1][0] == 'h';
            ts = parts[2];
        } else if (np == 4 && strcmp(parts[1], "hi") == 0 &&
                   strcmp(parts[2], "sat") == 0) {
            hi = 1;
            sat = 1;
            ts = parts[3];
        } else {
            perr(p, opline, "mad24 supports .lo/.hi.{u32,s32} and .hi.sat.s32");
            return -1;
        }
        if (strcmp(ts, "u32") == 0) {
            in.ty = PTY_U32;
            in.sign = 0;
        } else if (strcmp(ts, "s32") == 0) {
            in.ty = PTY_U32;
            in.sign = 1;
        } else {
            perr(p, opline, "mad24 supports .u32/.s32 only");
            return -1;
        }
        if (sat && (!hi || !in.sign)) {
            perr(p, opline, "mad24 .sat is only valid as mad24.hi.sat.s32");
            return -1;
        }
        in.op = POP_MAD24;
        in.is_hi = hi;
        in.sat = sat;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        for (int i = 0; i < 3; i++) {
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[i]) != 0)
                return -1;
        }
        in.nsrc = 3;
        if (expect_punct(p, ';', "after mad24") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "clmad") == 0 && np == 3 &&
        (strcmp(parts[1], "lo") == 0 || strcmp(parts[1], "hi") == 0) &&
        strcmp(parts[2], "u64") == 0) {
        // Carryless 64x64 multiply-add: XOR-accumulated shifts.
        in.op = POP_CLMAD;
        in.is_hi = parts[1][0] == 'h';
        in.ty = PTY_U64;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        for (int i = 0; i < 3; i++) {
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[i]) != 0)
                return -1;
        }
        in.nsrc = 3;
        if (expect_punct(p, ';', "after clmad") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "fns") == 0 && np == 2 && strcmp(parts[1], "b32") == 0) {
        // fns.b32 d, mask, base, offset: position of the n-th set bit
        // scanning from `base` in the signed `offset` direction.
        in.op = POP_FNS;
        in.ty = PTY_U32;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        for (int i = 0; i < 3; i++) {
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[i]) != 0)
                return -1;
        }
        in.nsrc = 3;
        if (expect_punct(p, ';', "after fns") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "szext") == 0 && np == 3 &&
        (strcmp(parts[1], "clamp") == 0 || strcmp(parts[1], "wrap") == 0) &&
        (strcmp(parts[2], "u32") == 0 || strcmp(parts[2], "s32") == 0)) {
        // szext.{clamp,wrap}.{u32,s32} d, a, b: sign/zero extend the
        // low N bits of a, where N = b (with the documented clamp/wrap
        // treatment of N >= 32).
        in.op = POP_SZEXT;
        in.ty = PTY_U32;
        in.sign = parts[2][0] == 's';
        in.sz_clamp = parts[1][0] == 'c';
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (expect_punct(p, ';', "after szext") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "abs") == 0 &&
        (np == 2 || (np == 3 && strcmp(parts[1], "ftz") == 0))) {
        // Absolute value for signed integers and floats (f32 optionally ftz).
        const char *ts = np == 3 ? parts[2] : parts[1];
        in.ftz = np == 3;
        if (strcmp(ts, "s32") == 0) {
            in.ty = PTY_U32;
        } else if (strcmp(ts, "s16") == 0) {
            in.ty = PTY_U16;
        } else if (strcmp(ts, "s64") == 0) {
            in.ty = PTY_U64;
        } else if (strcmp(ts, "f32") == 0) {
            in.ty = PTY_F32;
        } else if (strcmp(ts, "f64") == 0) {
            in.ty = PTY_F64;
            if (in.ftz) {
                perr(p, opline, "abs.ftz is not .f64");
                return -1;
            }
        } else if (strcmp(ts, "f16") == 0) {
            in.ty = PTY_F16;
        } else if (strcmp(ts, "f16x2") == 0) {
            in.ty = PTY_F16X2;
        } else if (strcmp(ts, "bf16") == 0) {
            in.ty = PTY_BF16;
            if (in.ftz) {
                perr(p, opline, "bf16 abs has no .ftz");
                return -1;
            }
        } else if (strcmp(ts, "bf16x2") == 0) {
            in.ty = PTY_BF16X2;
            if (in.ftz) {
                perr(p, opline, "bf16x2 abs has no .ftz");
                return -1;
            }
        } else {
            perr(p, opline, "abs supports .s16/.s32/.s64/.f16/.f16x2/.bf16/.bf16x2/.f32/.f64 and .ftz f16/f16x2/f32");
            return -1;
        }
        in.op = POP_ABS;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        in.nsrc = 1;
        if (expect_punct(p, ';', "after abs") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "popc") == 0 && np == 2) {
        // Population count (parallel sums; b64 splits into halves).
        if (strcmp(parts[1], "b32") == 0 ||
            strcmp(parts[1], "u32") == 0 ||
            strcmp(parts[1], "s32") == 0) {
            in.ty = PTY_U32;
        } else if (strcmp(parts[1], "b64") == 0 ||
                   strcmp(parts[1], "u64") == 0 ||
                   strcmp(parts[1], "s64") == 0) {
            in.ty = PTY_U64;
        } else {
            perr(p, opline, "popc supports .b32/.b64 (and .u/.s aliases) only");
            return -1;
        }
        in.op = POP_POPC;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        in.nsrc = 1;
        if (expect_punct(p, ';', "after popc") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "clz") == 0 && np == 2) {
        // Count leading zeros (binary search + explicit zero arm:
        // clz(0) is 32, unlike bfind which yields 0xFFFFFFFF).
        if (strcmp(parts[1], "b32") == 0 ||
            strcmp(parts[1], "u32") == 0 ||
            strcmp(parts[1], "s32") == 0) {
            in.ty = PTY_U32;
        } else if (strcmp(parts[1], "b64") == 0 ||
                   strcmp(parts[1], "u64") == 0 ||
                   strcmp(parts[1], "s64") == 0) {
            in.ty = PTY_U64;
        } else {
            perr(p, opline, "clz supports .b32/.b64 (and .u/.s aliases) only");
            return -1;
        }
        in.op = POP_CLZ;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        in.nsrc = 1;
        if (expect_punct(p, ';', "after clz") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "brev") == 0 && np == 2) {        // Bit reverse (5-stage swap).
        if (strcmp(parts[1], "b32") == 0 ||
            strcmp(parts[1], "u32") == 0 ||
            strcmp(parts[1], "s32") == 0) {
            in.ty = PTY_U32;
        } else if (strcmp(parts[1], "b64") == 0 ||
                   strcmp(parts[1], "u64") == 0 ||
                   strcmp(parts[1], "s64") == 0) {
            in.ty = PTY_U64;
        } else {
            perr(p, opline, "brev supports .b32/.b64 (and .u/.s aliases) only");
            return -1;
        }
        in.op = POP_BREV;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        in.nsrc = 1;
        if (expect_punct(p, ';', "after brev") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "bfe") == 0 && np == 2) {
        // Bitfield extract (ockl-equivalent semantics, verified against
        // hardware by ZLUDA): pos/len masked to 8 bits; pos >= 32 yields
        // 0; len >= 32 yields base >> pos.
        if (strcmp(parts[1], "u32") == 0 ||
            strcmp(parts[1], "s32") == 0) {
            in.ty = PTY_U32;
        } else if (strcmp(parts[1], "u64") == 0 ||
                   strcmp(parts[1], "s64") == 0) {
            in.ty = PTY_U64;
        } else {
            perr(p, opline, "bfe supports .u32/.s32/.u64/.s64 only");
            return -1;
        }
        in.sign = parts[1][0] == 's';
        in.op = POP_BFE;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        for (int i = 0; i < 3; i++) {
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[i]) != 0)
                return -1;
        }
        in.nsrc = 3;
        if (expect_punct(p, ';', "after bfe") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "bfi") == 0 && np == 2) {
        // Bitfield insert (ockl bfi_b32 semantics): d = (~m & base) |
        // (m & (insert << pos)) with m = ((1<<len)-1)<<pos; the FIRST
        // source is inserted into the SECOND (pinned by vector).
        if (strcmp(parts[1], "b32") == 0) {
            in.ty = PTY_U32;
        } else if (strcmp(parts[1], "b64") == 0) {
            in.ty = PTY_U64;
        } else {
            perr(p, opline, "bfi supports .b32/.b64 only");
            return -1;
        }
        in.op = POP_BFI;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        for (int i = 0; i < 4; i++) {
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[i]) != 0)
                return -1;
        }
        in.nsrc = 4;
        if (expect_punct(p, ';', "after bfi") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "bmsk") == 0 && np == 3 &&
        (strcmp(parts[1], "clamp") == 0 || strcmp(parts[1], "wrap") == 0)) {
        // Bit mask: `width` bits at `pos`, with clamp/wrap edge rules.
        if (strcmp(parts[2], "b32") != 0) {
            perr(p, opline, "bmsk supports .clamp.b32/.wrap.b32 only");
            return -1;
        }
        in.op = POP_BMSK;
        in.ty = PTY_U32;
        in.sz_clamp = parts[1][0] == 'c';
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (expect_punct(p, ';', "after bmsk") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "bfind") == 0 && np == 2 &&
        (strcmp(parts[1], "u32") == 0 || strcmp(parts[1], "s32") == 0 ||
         strcmp(parts[1], "u64") == 0 || strcmp(parts[1], "s64") == 0)) {
        // Most-significant non-sign bit position, 0xFFFFFFFF when none
        // (RTX 3070 truth: bfind.u32(41) = 5).
        in.op = POP_BFIND;
        in.ty = (parts[1][1] == '6') ? PTY_U64 : PTY_U32;
        in.sign = parts[1][0] == 's';
        in.arg = 1;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        in.nsrc = 1;
        if (expect_punct(p, ';', "after bfind") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "bfind") == 0 && np == 3 && strcmp(parts[1], "shiftamt") == 0) {
        // Shift amount to normalize (leading-zero count), 0xFFFFFFFF
        // when no bit is set (pinned by ZLUDA's vector).
        if (strcmp(parts[2], "u32") == 0 ||
            strcmp(parts[2], "s32") == 0) {
            in.ty = PTY_U32;
        } else if (strcmp(parts[2], "u64") == 0 ||
                   strcmp(parts[2], "s64") == 0) {
            in.ty = PTY_U64;
        } else {
            perr(p, opline, "bfind.shiftamt supports .u32/.s32/.u64/.s64 only");
            return -1;
        }
        in.sign = parts[2][0] == 's';
        in.op = POP_BFIND;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        in.nsrc = 1;
        if (expect_punct(p, ';', "after bfind") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "vshr") == 0 &&
        (np == 4 || np == 5 || np == 6)) {
        // vshr.type.atype.btype{.clamp}{.add} d, a, b{, c}
        // (ZLUDA's vshr_clamp): d = shift_right(a, b) [+ c]. With
        // .clamp a shift amount >= the width saturates (0 for
        // unsigned, sign fill for signed) instead of masking.
        int clamp = 0, add = 0;
        if (np >= 5) {
            if (strcmp(parts[4], "clamp") == 0)
                clamp = 1;
            else if (strcmp(parts[4], "add") == 0)
                add = 1;
            else {
                perr(p, opline, "only vshr .clamp/.add qualifiers supported");
                return -1;
            }
        }
        if (np == 6) {
            if (strcmp(parts[5], "add") == 0)
                add = 1;
            else if (strcmp(parts[5], "clamp") != 0) {
                perr(p, opline, "only vshr .clamp/.add qualifiers supported");
                return -1;
            }
        }
        if ((strcmp(parts[1], "u32") != 0 &&
             strcmp(parts[1], "s32") != 0) ||
            strcmp(parts[2], parts[1]) != 0 ||
            strcmp(parts[3], parts[1]) != 0) {
            perr(p, opline,
                 "vshr supports .u32.u32.u32/.s32.s32.s32 only");
            return -1;
        }
        in.op = POP_VSHR;
        in.ty = PTY_U32;
        in.sign = parts[1][0] == 's';
        in.vs_clamp = clamp;
        in.vs_add = add;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (add) {
            if (expect_comma(p) != 0 ||
                parse_operand(p, &in.src[2]) != 0)
                return -1;
            in.nsrc = 3;
        }
        if (expect_punct(p, ';', "after vshr") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "shf") == 0 && np == 4) {        // Funnel shift (ZLUDA corpus): 64-bit concat (b:a) shifted,
        // high-32 (.l) or low-32 (.r) taken. .wrap masks the amount
        // mod 32; .clamp returns a (.l) or b (.r) for amounts >= 32.
        // Unlike shl/shr, ptxas takes a register amount here.
        int left, clamp;
        if (strcmp(parts[1], "l") == 0)
            left = 1;
        else if (strcmp(parts[1], "r") == 0)
            left = 0;
        else {
            perr(p, opline, "shf needs .l or .r direction");
            return -1;
        }
        if (strcmp(parts[2], "clamp") == 0)
            clamp = 1;
        else if (strcmp(parts[2], "wrap") == 0)
            clamp = 0;
        else {
            perr(p, opline, "shf needs .clamp or .wrap mode");
            return -1;
        }
        if (strcmp(parts[3], "b32") != 0) {
            perr(p, opline, "only shf.b32 supported");
            return -1;
        }
        in.op = POP_SHF;
        in.ty = PTY_U32;
        in.shf_dir = left;
        in.shf_clamp = clamp;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        for (int i = 0; i < 3; i++) {
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[i]) != 0)
                return -1;
        }
        in.nsrc = 3;
        if (expect_punct(p, ';', "after shf") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "rcp") == 0) {
        // rcp.approx{.ftz}.f32, rcp.rn{.ftz}.f32, and rcp.rn/rz/rm/rp.f64.
        // Directed f32 works through the exact integer division path too.
        // The special truncated rcp.approx.ftz.f64 stays a clean error.
        int mode = -1, ftz = 0, idx = 1;
        if (idx < np && (strcmp(parts[idx], "approx") == 0 ||
                         strcmp(parts[idx], "rn") == 0 ||
                         strcmp(parts[idx], "rz") == 0 ||
                         strcmp(parts[idx], "rm") == 0 ||
                         strcmp(parts[idx], "rp") == 0)) {
            mode = strcmp(parts[idx], "approx") == 0 ? 0
                   : strcmp(parts[idx], "rn") == 0   ? 1
                   : strcmp(parts[idx], "rz") == 0   ? 2
                   : strcmp(parts[idx], "rm") == 0   ? 3
                                                     : 4;
            idx++;
        }
        if (idx < np && strcmp(parts[idx], "ftz") == 0) {
            ftz = 1;
            idx++;
        }
        if (mode < 0 || idx != np - 1) {
            perr(p, opline, "malformed rcp");
            return -1;
        }
        const char *ts = parts[idx];
        ptx_ty_t ty;
        if (strcmp(ts, "f32") == 0)
            ty = PTY_F32;
        else if (strcmp(ts, "f64") == 0)
            ty = PTY_F64;
        else {
            perr(p, opline, "rcp supports .f32/.f64 only");
            return -1;
        }
        if (ty == PTY_F64 && (mode == 0 || ftz)) {
            perr(p, opline, "rcp.f64 supports .rn/.rz/.rm/.rp (rcp.approx.ftz.f64 is a special truncated form)");
            return -1;
        }
        in.op = POP_RCP;
        in.ty = ty;
        in.rnd = mode; // 0 approx, 1 rn, 2 rz, 3 rm, 4 rp
        in.ftz = ftz;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        in.nsrc = 1;
        if (expect_punct(p, ';', "after rcp") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "sin") == 0 || strcmp(parts[0], "cos") == 0 ||
        strcmp(parts[0], "sqrt") == 0 || strcmp(parts[0], "rsqrt") == 0 ||
        strcmp(parts[0], "lg2") == 0 || strcmp(parts[0], "ex2") == 0 ||
        strcmp(parts[0], "tanh") == 0) {
        // op.mode{.ftz}.type. Only .approx is approximate-legal for
        // sin/cos/lg2/ex2/tanh; sqrt/rsqrt also support their exact
        // documented forms. Directed rounding stays a clean error.
        const char *op = parts[0];
        int mode = -1; // 0 approx, 1 rn, 2 rz, 3 rm, 4 rp
        int ftz = 0;
        int idx = 1;
        if (idx < np && (strcmp(parts[idx], "approx") == 0 ||
                         strcmp(parts[idx], "rn") == 0 ||
                         strcmp(parts[idx], "rz") == 0 ||
                         strcmp(parts[idx], "rm") == 0 ||
                         strcmp(parts[idx], "rp") == 0)) {
            mode = strcmp(parts[idx], "approx") == 0 ? 0
                   : strcmp(parts[idx], "rn") == 0   ? 1
                   : strcmp(parts[idx], "rz") == 0   ? 2
                   : strcmp(parts[idx], "rm") == 0   ? 3
                                                     : 4;
            idx++;
        }
        if (idx < np && strcmp(parts[idx], "ftz") == 0) {
            ftz = 1;
            idx++;
        }
        if (mode < 0 || idx != np - 1) {
            perr(p, opline, "malformed floating-point instruction");
            return -1;
        }
        const char *ts = parts[idx];
        ptx_ty_t ty;
        if (strcmp(ts, "f32") == 0)
            ty = PTY_F32;
        else if (strcmp(ts, "f64") == 0)
            ty = PTY_F64;
        else if (strcmp(ts, "f16") == 0)
            ty = PTY_F16;
        else if (strcmp(ts, "f16x2") == 0)
            ty = PTY_F16X2;
        else if (strcmp(ts, "bf16") == 0)
            ty = PTY_BF16;
        else if (strcmp(ts, "bf16x2") == 0)
            ty = PTY_BF16X2;
        else {
            perr(p, opline, "unsupported transcendental type");
            return -1;
        }
        if (mode >= 2) {
            perr(p, opline, "directed rounding is not supported for this instruction");
            return -1;
        }
        int is_half = ty == PTY_F16 || ty == PTY_F16X2 ||
                      ty == PTY_BF16 || ty == PTY_BF16X2;
        if (strcmp(op, "tanh") == 0) {
            if (mode != 0 || ftz || (ty != PTY_F32 && !is_half)) {
                perr(p, opline, "tanh supports .approx.f32/.f16/.f16x2/.bf16/.bf16x2");
                return -1;
            }
        } else if (strcmp(op, "ex2") == 0) {
            if (mode != 0 || (!is_half && ty != PTY_F32)) {
                perr(p, opline, "ex2 supports .approx.f32/.f16/.f16x2 and .approx.ftz.bf16/bf16x2");
                return -1;
            }
            int is_bf = ty == PTY_BF16 || ty == PTY_BF16X2;
            int is_f16 = ty == PTY_F16 || ty == PTY_F16X2;
            if ((is_bf && !ftz) || (is_f16 && ftz)) {
                perr(p, opline, "ex2 bf16 forms require .ftz; f16 forms do not");
                return -1;
            }
        } else if (strcmp(op, "sin") == 0 || strcmp(op, "cos") == 0 ||
                   strcmp(op, "lg2") == 0) {
            if (mode != 0 || ty != PTY_F32) {
                perr(p, opline, "sin/cos/lg2 support only .approx.f32");
                return -1;
            }
        } else if (is_half) {
            perr(p, opline, "half precision is supported only for tanh/ex2");
            return -1;
        }
        if (strcmp(op, "rsqrt") == 0) {
            if (mode != 0 || (ftz && ty == PTY_F64)) {
                perr(p, opline, "rsqrt supports .approx.f32/.approx.ftz.f32/.approx.f64");
                return -1;
            }
        }
        if (strcmp(op, "sqrt") == 0) {
            if (ty == PTY_F64 && (mode != 1 || ftz)) {
                perr(p, opline, "sqrt.f64 supports .rn only");
                return -1;
            }
            if (ty == PTY_F32 && mode != 0 && mode != 1) {
                perr(p, opline, "sqrt.f32 supports .approx/.rn");
                return -1;
            }
        }
        int glop = strcmp(op, "sin") == 0      ? 13
                   : strcmp(op, "cos") == 0    ? 14
                   : strcmp(op, "sqrt") == 0   ? 31
                   : strcmp(op, "rsqrt") == 0  ? 32
                   : strcmp(op, "lg2") == 0    ? 30
                   : strcmp(op, "ex2") == 0    ? 29
                                               : 21;
        in.op = POP_GLSL;
        in.ty = ty;
        in.glop = glop;
        in.ftz = ftz;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        in.nsrc = 1;
        if (expect_punct(p, ';', "after transcendental") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "div") == 0 &&
        strcmp(parts[np - 1], "f16") != 0 &&
        strcmp(parts[np - 1], "f16x2") != 0 &&
        (strcmp(parts[1], "approx") == 0 || strcmp(parts[1], "full") == 0 ||
         strcmp(parts[1], "rn") == 0 || strcmp(parts[1], "rz") == 0 ||
         strcmp(parts[1], "rm") == 0 || strcmp(parts[1], "rp") == 0)) {
        // div.approx/full{.ftz}.f32, div.rn{.ftz}.f32 and div.rn.f64.
        // Directed rounding stays a clean error; approx/full are f32-only.
        int mode = strcmp(parts[1], "approx") == 0 ? 0
                   : strcmp(parts[1], "full") == 0 ? 1
                   : strcmp(parts[1], "rn") == 0   ? 2
                   : strcmp(parts[1], "rz") == 0   ? 3
                   : strcmp(parts[1], "rm") == 0   ? 4
                                                   : 5;
        int ftz = 0, idx = 2;
        if (idx < np && strcmp(parts[idx], "ftz") == 0) {
            ftz = 1;
            idx++;
        }
        if (idx != np - 1) {
            perr(p, opline, "malformed div");
            return -1;
        }
        const char *ts = parts[idx];
        ptx_ty_t ty;
        if (strcmp(ts, "f32") == 0)
            ty = PTY_F32;
        else if (strcmp(ts, "f64") == 0)
            ty = PTY_F64;
        else {
            perr(p, opline, "float div supports .f32/.f64");
            return -1;
        }
        if (ty == PTY_F64 && ftz) {
            perr(p, opline, "div.f64 does not take .ftz");
            return -1;
        }
        in.op = POP_DIV;
        in.ty = ty;
        in.ftz = ftz;
        in.frnd = mode >= 3 ? mode - 2 : 0;
        in.approx_div = (mode == 0 || mode == 1);
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (expect_punct(p, ';', "after div") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "div") == 0 && np == 4 && strcmp(parts[1], "ftz") == 0 &&
        strcmp(parts[2], "rn") == 0 && strcmp(parts[3], "f32") == 0) {
        // Flush-to-zero division (f32 only; see ftz lowering).
        in.op = POP_DIV;
        in.ty = PTY_F32;
        in.ftz = 1;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (expect_punct(p, ';', "after div.ftz") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "div") == 0 && (np == 2 || np == 3)) {
        // Integer div truncates toward zero (C/PTX semantics); float
        // div takes .rn (exact request) or .approx (anything goes —
        // our hardware fdiv satisfies both). Directed roundings
        // (.rz/.rm/.rp) stay clean errors: OpFDiv cannot promise them.
        in.op = POP_DIV;
        in.sign = 0;
        if (np == 2) {
            if (strcmp(parts[1], "u32") == 0 ||
                strcmp(parts[1], "b32") == 0) {
                in.ty = PTY_U32;
            } else if (strcmp(parts[1], "s32") == 0) {
                in.ty = PTY_U32;
                in.sign = 1;
            } else if (strcmp(parts[1], "u64") == 0 ||
                       strcmp(parts[1], "b64") == 0) {
                in.ty = PTY_U64;
            } else if (strcmp(parts[1], "s64") == 0) {
                in.ty = PTY_U64;
                in.sign = 1;
            } else if (strcmp(parts[1], "u16") == 0 ||
                       strcmp(parts[1], "b16") == 0) {
                in.ty = PTY_U16;
            } else if (strcmp(parts[1], "s16") == 0) {
                in.ty = PTY_U16;
                in.sign = 1;
            } else {
                perr(p, opline, "div supports .u16/.s16/.u32/.s32/.u64/.s64 (floats need .rn/.approx)");
                return -1;
            }
        } else {
            if (strcmp(parts[1], "rn") != 0 &&
                strcmp(parts[1], "approx") != 0) {
                perr(p, opline, "float div needs .rn or .approx (directed roundings unsupported)");
                return -1;
            }
            if (strcmp(parts[2], "f32") == 0) {
                in.ty = PTY_F32;
            } else if (strcmp(parts[2], "f64") == 0) {
                in.ty = PTY_F64;
            } else if (strcmp(parts[2], "f16") == 0) {
                in.ty = PTY_F16;
            } else if (strcmp(parts[2], "f16x2") == 0) {
                in.ty = PTY_F16X2;
            } else {
                perr(p, opline, "float div supports .f32/.f64/.f16/.f16x2 only");
                return -1;
            }
        }
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (expect_punct(p, ';', "after div") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "rem") == 0 && np == 2) {
        // C-like remainder (sign follows the dividend). Float rem
        // does not exist in PTX (use fmod-style idioms instead).
        in.op = POP_REM;
        in.sign = 0;
        if (strcmp(parts[1], "u32") == 0 ||
            strcmp(parts[1], "b32") == 0) {
            in.ty = PTY_U32;
        } else if (strcmp(parts[1], "s32") == 0) {
            in.ty = PTY_U32;
            in.sign = 1;
        } else if (strcmp(parts[1], "u64") == 0 ||
                   strcmp(parts[1], "b64") == 0) {
            in.ty = PTY_U64;
        } else if (strcmp(parts[1], "s64") == 0) {
            in.ty = PTY_U64;
            in.sign = 1;
        } else if (strcmp(parts[1], "u16") == 0 ||
                   strcmp(parts[1], "b16") == 0) {
            in.ty = PTY_U16;
        } else if (strcmp(parts[1], "s16") == 0) {
            in.ty = PTY_U16;
            in.sign = 1;
        } else {
            perr(p, opline, "rem supports .u16/.s16/.u32/.s32/.u64/.s64 only (no float rem in PTX)");
            return -1;
        }
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (expect_punct(p, ';', "after rem") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if ((strcmp(parts[0], "fma") == 0 || strcmp(parts[0], "mad") == 0) &&
        np >= 2 &&
        (strcmp(parts[np - 1], "f32") == 0 ||
         (strcmp(parts[0], "fma") == 0 &&
          strcmp(parts[np - 1], "f32x2") == 0) ||
         strcmp(parts[np - 1], "f64") == 0 ||
         (strcmp(parts[0], "fma") == 0 &&
          (strcmp(parts[np - 1], "f16") == 0 ||
           strcmp(parts[np - 1], "f16x2") == 0 ||
           strcmp(parts[np - 1], "bf16") == 0 ||
           strcmp(parts[np - 1], "bf16x2") == 0)))) {
        // Float fma/mad. fma requires an explicit rounding modifier;
        // mad defaults to .rn. .rn and directed modes are lowered via
        // the integer software FMA when the device fused path is not
        // proven correct. .ftz/.sat are f32-only.
        int is_mad = strcmp(parts[0], "mad") == 0;
        const char *ts = parts[np - 1];
        ptx_ty_t ty;
        int is_x2 = 0;
        if (strcmp(ts, "f32") == 0)
            ty = PTY_F32;
        else if (!is_mad && strcmp(ts, "f32x2") == 0) {
            ty = PTY_U64;
            is_x2 = 1;
        } else if (strcmp(ts, "f64") == 0)
            ty = PTY_F64;
        else if (!is_mad && strcmp(ts, "f16") == 0)
            ty = PTY_F16;
        else if (!is_mad && strcmp(ts, "f16x2") == 0)
            ty = PTY_F16X2;
        else if (!is_mad && strcmp(ts, "bf16") == 0)
            ty = PTY_BF16;
        else if (!is_mad && strcmp(ts, "bf16x2") == 0)
            ty = PTY_BF16X2;
        else {
            perr(p, opline, "fma/mad supports .f32/.f64 (and fma .f16/.f16x2/.bf16x2)");
            return -1;
        }
        int have_rnd = 0, ftz = 0, sat = 0, relu = 0, oob = 0, frnd = 0;
        for (int pi = 1; pi < np - 1; pi++) {
            if (strcmp(parts[pi], "rn") == 0 ||
                strcmp(parts[pi], "rz") == 0 ||
                strcmp(parts[pi], "rm") == 0 ||
                strcmp(parts[pi], "rp") == 0) {
                frnd = strcmp(parts[pi], "rn") == 0 ? 0
                       : strcmp(parts[pi], "rz") == 0 ? 1
                       : strcmp(parts[pi], "rm") == 0 ? 2 : 3;
                have_rnd = 1;
            } else if (strcmp(parts[pi], "ftz") == 0) {
                ftz = 1;
            } else if (strcmp(parts[pi], "sat") == 0) {
                sat = 1;
            } else if (strcmp(parts[pi], "relu") == 0) {
                relu = 1;
            } else if (strcmp(parts[pi], "oob") == 0) {
                oob = 1;
            } else {
                perr(p, opline, "unsupported fma/mad modifier");
                return -1;
            }
        }
        if (!is_mad && !have_rnd) {
            perr(p, opline, "fma requires a rounding modifier (.rn)");
            return -1;
        }
        if (frnd && ty != PTY_F32 && ty != PTY_F64) {
            perr(p, opline,
                 "directed fma/mad supports .f32/.f64 only");
            return -1;
        }
        if (is_x2 && sat) {
            perr(p, opline, "fma.f32x2 does not take .sat");
            return -1;
        }
        if (relu || oob) {
            if (ty != PTY_F16 && ty != PTY_F16X2 && ty != PTY_BF16 &&
                ty != PTY_BF16X2) {
                perr(p, opline, "fma .relu/.oob are half-precision only");
                return -1;
            }
            if (relu && sat) {
                perr(p, opline, "fma cannot combine .relu and .sat");
                return -1;
            }
            // .oob is accepted.  The tensor OOB-NaN payload is not
            // documented, so the special clamp is not modelled; the
            // arithmetic is an ordinary fused multiply-add.  This is
            // exact for every non-OOB input.
            (void)oob;
        }
        if ((ftz || sat) && ty != PTY_F32 && ty != PTY_F16 &&
            ty != PTY_F16X2 && !is_x2) {
            perr(p, opline, "fma/mad .ftz/.sat are supported for .f32/.f16/.f16x2");
            return -1;
        }
        in.op = POP_FMA;
        in.ty = ty;
        in.ftz = ftz;
        in.sat = sat;
        in.f32x2 = is_x2;
        in.relu = relu;
        in.frnd = frnd;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        for (int i = 0; i < 3; i++) {
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[i]) != 0)
                return -1;
        }
        in.nsrc = 3;
        if (expect_punct(p, ';', "after fma/mad") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strncmp(opname, "cvta.", 5) == 0) {
        // Convert address between state spaces. Common forms:
        //   cvta.to.{global,local,shared,const,param,generic}.{u32,u64}
        //   cvta.{global,local,shared,const,param,generic}.{u32,u64}
        // Lower as MOV while tracking destination address space for pointer lineage.
        const char *rest = opname + 5;
        if (strncmp(rest, "to.", 3) == 0)
            rest += 3;
        ptx_space_t cvta_space = PSP_GENERIC;
        int cvta_ok = 1;
        ptx_ty_t cvta_ty = PTY_U64;
        if (strncmp(rest, "global.", 7) == 0) { cvta_space = PSP_GLOBAL; rest += 7; }
        else if (strncmp(rest, "local.", 6) == 0) { cvta_space = PSP_LOCAL; rest += 6; }
        else if (strncmp(rest, "shared.", 7) == 0) { cvta_space = PSP_SHARED; rest += 7; }
        else if (strncmp(rest, "const.", 6) == 0) { cvta_space = PSP_CONST; rest += 6; }
        else if (strncmp(rest, "param.", 6) == 0) { cvta_space = PSP_PARAM; rest += 6; }
        else if (strncmp(rest, "generic.", 8) == 0) { cvta_space = PSP_GENERIC; rest += 8; }
        else { cvta_ok = 0; }
        if (cvta_ok) {
            if (strcmp(rest, "u64") == 0) cvta_ty = PTY_U64;
            else if (strcmp(rest, "u32") == 0) cvta_ty = PTY_U32;
            else if (strcmp(rest, "s64") == 0) cvta_ty = PTY_U64;
            else if (strcmp(rest, "s32") == 0) cvta_ty = PTY_U32;
            else cvta_ok = 0;
        }
        if (cvta_ok) {
            in.op = POP_MOV;
            in.ty = cvta_ty;
            in.space = cvta_space;
            if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
                return -1;
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
                return -1;
            in.nsrc = 1;
            if (expect_punct(p, ';', "after cvta") != 0)
                return -1;
            return block_add_instr(fn, bi, &in, p, opline);
        }
        /* fall through to unsupported if form is unrecognized */
    }
    if (strcmp(parts[0], "bar") == 0 && np == 3 &&
        strcmp(parts[1], "warp") == 0 && strcmp(parts[2], "sync") == 0) {
        // bar.warp.sync membermask. Full-warp masks lower to a subgroup
        // barrier; partial/dynamic masks are accepted and conservatively
        // lowered as a subgroup barrier (best-effort for active-mask patterns).
        if (parse_operand(p, &in.src[0]) != 0)
            return -1;
        in.op = POP_BARWARP;
        in.nsrc = 1;
        if (expect_punct(p, ';', "after bar.warp.sync") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    // CTA barriers: bar/barrier{.cta}.{sync,arrive,red}.
    // Normalize the prefix once so every accepted spelling shares the
    // same lowering.  .cta is a no-op qualifier and .aligned is a
    // convergence promise we can honour by construction.
    if (strcmp(parts[0], "barrier") == 0 && np >= 3 &&
        strcmp(parts[1], "cluster") == 0) {
        // barrier.cluster.arrive{.sem}{.aligned}
        // barrier.cluster.wait{.acquire}{.aligned}
        // Vulkan has no cluster execution scope.  Model the implicit
        // cluster barrier as a CTA-wide software barrier (correct for a
        // single-CTA cluster; see the docs).
        int arrive = strcmp(parts[2], "arrive") == 0;
        int wait = strcmp(parts[2], "wait") == 0;
        if (!arrive && !wait) {
            perr(p, opline,
                 "barrier.cluster supports .arrive/.wait");
            return -1;
        }
        int seen_align = 0, seen_sem = 0;
        for (int i = 3; i < np; i++) {
            if (strcmp(parts[i], "aligned") == 0 && !seen_align) {
                seen_align = 1;
                continue;
            }
            if (arrive && !seen_sem &&
                (strcmp(parts[i], "relaxed") == 0 ||
                 strcmp(parts[i], "release") == 0)) {
                seen_sem = 1;
                continue;
            }
            if (wait && !seen_sem && strcmp(parts[i], "acquire") == 0) {
                seen_sem = 1;
                continue;
            }
            perr(p, opline, "bad barrier.cluster qualifier");
            return -1;
        }
        in.op = POP_BAR;
        in.bar_cluster = 1;
        in.bar_arrive = arrive;
        in.arg = 16; // dedicated software counter slot
        in.nsrc = 0;
        if (expect_punct(p, ';', "after barrier.cluster") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    {
        int bbase = -1;
        if (strcmp(parts[0], "bar") == 0 ||
            strcmp(parts[0], "barrier") == 0) {
            bbase = 1;
            if (np > 1 && strcmp(parts[1], "cta") == 0)
                bbase = 2;
        }
        if (bbase >= 0 && bbase < np &&
            (strcmp(parts[bbase], "sync") == 0 ||
             strcmp(parts[bbase], "arrive") == 0)) {
            // barrier{.cta}.sync{.aligned} a{, b}
            int pi = bbase + 1;
            if (pi < np && strcmp(parts[pi], "aligned") == 0)
                pi++;
            if (pi != np) {
                perr(p, opline, "bad bar/barrier sync qualifier");
                return -1;
            }
            int arrive = strcmp(parts[bbase], "arrive") == 0;
            in.op = POP_BAR;
            in.bar_arrive = arrive;
            ptx_tok_t bt;
            if (next(p, &bt) != 0)
                return -1;
            if (bt.kind == T_INT && bt.imm <= 15) {
                in.arg = (int)bt.imm;
            } else if (bt.kind == T_IDENT) {
                in.arg = -1;
            } else {
                perr(p, bt.line,
                     "barrier needs a barrier id 0..15");
                return -1;
            }
            if (at_punct(p, ',')) {
                if (parse_operand(p, &in.src[0]) != 0)
                    return -1;
                if (in.src[0].kind == OPD_IMM &&
                    !in.src[0].imm_is_float &&
                    (in.src[0].imm == 0 ||
                     (in.src[0].imm % 32) != 0)) {
                    perr(p, opline,
                         "barrier thread count must be a nonzero "
                         "multiple of the warp size");
                    return -1;
                }
                in.nsrc = 1;
            } else if (arrive) {
                perr(p, opline,
                     "barrier arrive requires a non-zero thread count");
                return -1;
            }
            if (in.arg < 0 && in.nsrc > 0) {
                perr(p, opline,
                     "counted bar/barrier needs an immediate id 0..15");
                return -1;
            }
            if (expect_punct(p, ';', "after bar/barrier") != 0)
                return -1;
            return block_add_instr(fn, bi, &in, p, opline);
        }
        if (bbase >= 0 && bbase + 1 < np &&
            strcmp(parts[bbase], "red") == 0) {
            // barrier{.cta}.red.{popc|and|or}{.aligned}.{u32|pred}
            int pi = bbase + 1;
            const char *rop = parts[pi++];
            if (pi < np && strcmp(parts[pi], "aligned") == 0)
                pi++;
            if (pi >= np || pi + 1 != np) {
                perr(p, opline, "bad bar/barrier.red qualifiers");
                return -1;
            }
            const char *rty = parts[pi];
            in.op = POP_BARRED;
            if (strcmp(rop, "popc") == 0 && strcmp(rty, "u32") == 0) {
                in.vote_mode = 2; // count true predicates
                in.ty = PTY_U32;
            } else if ((strcmp(rop, "and") == 0 ||
                        strcmp(rop, "or") == 0) &&
                       strcmp(rty, "pred") == 0) {
                in.vote_mode = strcmp(rop, "or") == 0 ? 1 : 0;
                in.ty = PTY_PRED;
            } else {
                perr(p, opline,
                     "barrier.red supports .popc.u32 and "
                     ".{and,or}.pred");
                return -1;
            }
            if (parse_dst(p, in.dst, sizeof(in.dst),
                          in.ty == PTY_PRED ? 1 : 0) != 0)
                return -1;
            if (expect_comma(p) != 0)
                return -1;
            ptx_tok_t bt;
            if (next(p, &bt) != 0)
                return -1;
            if (!(bt.kind == T_INT && bt.imm <= 15) &&
                bt.kind != T_IDENT) {
                perr(p, bt.line, "barrier.red needs a barrier id 0..15");
                return -1;
            }
            if (expect_comma(p) != 0)
                return -1;
            // Operands are either `{!}c` or `count, {!}c`.
            if (at_punct(p, '!')) {
                in.bneg = 1;
                if (parse_operand(p, &in.src[0]) != 0)
                    return -1;
            } else {
                ptx_opd_t first;
                if (parse_operand(p, &first) != 0)
                    return -1;
                if (at_punct(p, ',')) {
                    if (first.kind == OPD_IMM && !first.imm_is_float &&
                        (first.imm == 0 || (first.imm % 32) != 0)) {
                        perr(p, opline,
                             "barrier.red thread count must be a "
                             "nonzero multiple of the warp size");
                        return -1;
                    }
                    // at_punct consumed the comma before the
                    // predicate/count; do not expect another one.
                    if (at_punct(p, '!'))
                        in.bneg = 1;
                    if (parse_operand(p, &in.src[0]) != 0)
                        return -1;
                } else {
                    in.src[0] = first;
                }
            }
            in.nsrc = 1;
            if (expect_punct(p, ';', "after bar/barrier.red") != 0)
                return -1;
            return block_add_instr(fn, bi, &in, p, opline);
        }
    }
    if (strcmp(parts[0], "fence") == 0 ||
        (strcmp(parts[0], "membar") == 0 && np >= 3 &&
         strcmp(parts[1], "proxy") == 0)) {
        // fence{.sem}.scope / fence.proxy.* / membar.proxy.*
        // Vulkan has no cluster scope and no proxy distinction; thread
        // fences map to OpMemoryBarrier and proxy fences use a full
        // device-scope barrier (a safe superset of the requested order).
        int proxy = (strcmp(parts[0], "membar") == 0 ||
                     (np > 1 && strcmp(parts[1], "proxy") == 0));
        int sem = -1, scope = -1;
        int start = 1;
        if (strcmp(parts[0], "membar") == 0)
            start = 2;
        else if (proxy)
            start = 2;
        if (proxy && np <= 2) {
            perr(p, opline, "fence.proxy needs a proxy kind");
            return -1;
        }
        for (int i = start; i < np; i++) {
            if (strcmp(parts[i], "sc") == 0 ||
                strcmp(parts[i], "acq_rel") == 0)
                sem = 0;
            else if (strcmp(parts[i], "acquire") == 0)
                sem = 1;
            else if (strcmp(parts[i], "release") == 0)
                sem = 2;
            else if (strcmp(parts[i], "cta") == 0)
                scope = 0;
            else if (strcmp(parts[i], "cluster") == 0)
                scope = 1;
            else if (strcmp(parts[i], "gpu") == 0)
                scope = 2;
            else if (strcmp(parts[i], "sys") == 0)
                scope = 3;
            else if (proxy)
                continue; // async/alias/fabric proxy kinds
            else if (strcmp(parts[i], "mbarrier_init") == 0 ||
                     strcmp(parts[i], "op_restrict") == 0)
                continue;
            else if (strncmp(parts[i], "sync_restrict::", 14) == 0)
                continue;
            else {
                perr(p, opline, "bad fence qualifier");
                return -1;
            }
        }
        if (sem < 0)
            sem = 0;
        if (scope < 0) {
            if (proxy)
                scope = 3; // proxy fences are device-wide
            else {
                perr(p, opline, "fence needs a scope");
                return -1;
            }
        }
        in.op = POP_MEMBAR;
        in.fence_sem = sem;
        in.fence_scope = scope;
        in.arg = 0;
        // Uni-directional proxy acquire fences take [addr], size.
        if (proxy) {
            ptx_tok_t tk;
            if (next(p, &tk) != 0)
                return -1;
            if (tk.kind == T_PUNCT && tk.punct == '[') {
                ptx_unget(&p->lx, &tk);
                if (parse_operand(p, &in.src[0]) != 0)
                    return -1;
                if (expect_comma(p) != 0)
                    return -1;
                if (parse_operand(p, &in.src[1]) != 0)
                    return -1;
                in.nsrc = 2;
            } else {
                ptx_unget(&p->lx, &tk);
            }
        }
        if (expect_punct(p, ';', "after fence") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "membar") == 0 && np == 2) {
        // Memory fence: membar.cta/gl/sys -> OpMemoryBarrier at the
        // matching scope with AcquireRelease semantics (Vulkan does not
        // allow SequentiallyConsistent on OpMemoryBarrier).
        in.fence_sem = 0;
        if (strcmp(parts[1], "cta") == 0) {
            in.arg = 0;
            in.fence_scope = 0;
        } else if (strcmp(parts[1], "gl") == 0) {
            in.arg = 1;
            in.fence_scope = 2;
        } else if (strcmp(parts[1], "sys") == 0) {
            in.arg = 2;
            in.fence_scope = 3;
        } else {
            perr(p, opline, "membar supports .cta/.gl/.sys only");
            return -1;
        }
        in.op = POP_MEMBAR;
        if (expect_punct(p, ';', "after membar") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "nanosleep") == 0 && np == 2 &&
        strcmp(parts[1], "u32") == 0) {
        // Sleep hint, elided at emit (Vulkan has no portable sleep and
        // dropping a hint changes no observable result). The operand is
        // still type-checked there so ptxas-illegal forms stay errors.
        in.op = POP_NANOSLEEP;
        in.ty = PTY_U32;
        if (parse_operand(p, &in.src[0]) != 0)
            return -1;
        in.nsrc = 1;
        if (expect_punct(p, ';', "after nanosleep") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if ((strcmp(parts[0], "shr") == 0 || strcmp(parts[0], "shl") == 0) &&
        np == 2) {
        int is_shr = parts[0][1] == 'h' && parts[0][2] == 'r';
        // shl takes .b types (numba shape); shr takes .u (logical),
        // .s (arithmetic) plus .b (logical, accepted by ptxas).
        // 32- and 64-bit alike.
        in.sign = 0;
        if (is_shr) {
            if (strcmp(parts[1], "u32") == 0 ||
                strcmp(parts[1], "b32") == 0) {
                in.ty = PTY_U32;
            } else if (strcmp(parts[1], "s32") == 0) {
                in.ty = PTY_U32;
                in.sign = 1;
            } else if (strcmp(parts[1], "u16") == 0 ||
                       strcmp(parts[1], "b16") == 0) {
                in.ty = PTY_U16;
            } else if (strcmp(parts[1], "s16") == 0) {
                in.ty = PTY_U16;
                in.sign = 1;
            } else if (strcmp(parts[1], "u64") == 0 ||
                       strcmp(parts[1], "b64") == 0) {
                in.ty = PTY_U64;
            } else if (strcmp(parts[1], "s64") == 0) {
                in.ty = PTY_U64;
                in.sign = 1;
            } else if (strcmp(parts[1], "u8") == 0 ||
                       strcmp(parts[1], "b8") == 0) {
                in.ty = PTY_U8;
            } else if (strcmp(parts[1], "s8") == 0) {
                in.ty = PTY_U8;
                in.sign = 1;
            } else {
                perr(p, opline, "shr supports .u8/.s8/.u16/.s16/.u32/.s32/.u64/.s64 (and .b aliases) only");
                return -1;
            }
        } else {
            if (strcmp(parts[1], "b16") == 0) {
                in.ty = PTY_U16;
            } else if (strcmp(parts[1], "b32") == 0) {
                in.ty = PTY_U32;
            } else if (strcmp(parts[1], "b64") == 0) {
                in.ty = PTY_U64;
            } else if (strcmp(parts[1], "b8") == 0 ||
                       strcmp(parts[1], "u8") == 0 ||
                       strcmp(parts[1], "s8") == 0) {
                in.ty = PTY_U8;
            } else {
                perr(p, opline, "shl supports .b8/.b16/.b32/.b64 only");
                return -1;
            }
        }
        in.op = is_shr ? POP_SHR : POP_SHL;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        // PTX allows a register shift amount (warp-lane masks compute
        // `1ull << lane`); immediates are just the common case.
        in.nsrc = 2;
        if (expect_punct(p, ';', "after shr") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if ((strcmp(parts[0], "and") == 0 || strcmp(parts[0], "or") == 0 ||
         strcmp(parts[0], "xor") == 0) &&
        np == 2) {
        // Bitwise logic takes .b types only (ptxas rejects .u32
        // spellings); .b16/.b32/.b64 lower identically apart from width.
        if (strcmp(parts[1], "b16") == 0) {
            in.ty = PTY_U16;
        } else if (strcmp(parts[1], "b32") == 0) {
            in.ty = PTY_U32;
        } else if (strcmp(parts[1], "b64") == 0) {
            in.ty = PTY_U64;
        } else if (strcmp(parts[1], "pred") == 0) {
            in.ty = PTY_PRED;
        } else {
            perr(p, opline,
                 "logic ops support .b16/.b32/.b64/.pred only");
            return -1;
        }
        in.op = parts[0][0] == 'a' ? POP_AND
                : parts[0][0] == 'o' ? POP_OR
                                     : POP_XOR;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (expect_punct(p, ';', "after and/or/xor") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "selp") == 0 && np == 2) {
        // Conditional select (numba loop prologues/epilogues):
        // selp.type dst, a, b, pred. Lowered as OpSelect. PTX selp
        // starts at 16 bits (no .u8/.f16 forms).
        in.op = POP_SELP;
        if (strcmp(parts[1], "u16") == 0 ||
            strcmp(parts[1], "s16") == 0 ||
            strcmp(parts[1], "b16") == 0) {
            in.ty = PTY_U16;
        } else if (strcmp(parts[1], "u32") == 0 ||
                   strcmp(parts[1], "s32") == 0 ||
                   strcmp(parts[1], "b32") == 0) {
            in.ty = PTY_U32;
        } else if (strcmp(parts[1], "u64") == 0 ||
                   strcmp(parts[1], "s64") == 0 ||
                   strcmp(parts[1], "b64") == 0) {
            in.ty = PTY_U64;
        } else if (strcmp(parts[1], "f32") == 0) {
            in.ty = PTY_F32;
        } else if (strcmp(parts[1], "f64") == 0) {
            in.ty = PTY_F64;
        } else {
            perr(p, opline, "selp supports .u16/.u32/.u64/.f32/.f64 (and .s/.b aliases) only");
            return -1;
        }
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[2]) != 0)
            return -1;
        if (in.src[2].kind == OPD_IMM) {
            // ptxas accepts constant predicates (ZLUDA corpus uses 0/1
            // for unconditional selects): fold to a mov of the picked
            // source at parse time.
            if (in.src[2].imm != 0 && in.src[2].imm != 1) {
                perr(p, opline, "selp immediate predicate must be 0 or 1");
                return -1;
            }
            in.op = POP_MOV;
            in.src[0] = in.src[2].imm == 0 ? in.src[1] : in.src[0];
            in.nsrc = 1;
            if (expect_punct(p, ';', "after selp") != 0)
                return -1;
            return block_add_instr(fn, bi, &in, p, opline);
        }
        if (in.src[2].kind != OPD_REG) {
            perr(p, opline, "selp predicate must be a .pred register");
            return -1;
        }
        in.nsrc = 3;
        if (expect_punct(p, ';', "after selp") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "not") == 0 && np == 2) {
        // Bitwise complement (numba loop-bound math): not.b64.
        // .u32/.b32 and .u64/.b64/.s64 spellings accepted (wrap).
        // not.pred is the logical complement (parity with ptxas).
        if (map_ty(parts[1], &in.ty) != 0 ||
            (in.ty != PTY_U32 && in.ty != PTY_U64 &&
             in.ty != PTY_PRED)) {
            perr(p, opline, "not supports integer and .pred types only");
            return -1;
        }
        in.op = POP_NOT;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        in.nsrc = 1;
        if (expect_punct(p, ';', "after not") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "neg") == 0 &&
        (np == 2 || (np == 3 && strcmp(parts[1], "ftz") == 0))) {
        // Arithmetic negation. Like ptxas, only signed and float
        // forms (neg applies to signed numeric types).
        const char *ts = np == 3 ? parts[2] : parts[1];
        in.ftz = np == 3;
        if (strcmp(ts, "s32") == 0) {
            in.ty = PTY_U32;
            in.sign = 1;
        } else if (strcmp(ts, "s16") == 0) {
            in.ty = PTY_U16;
            in.sign = 1;
        } else if (strcmp(ts, "s64") == 0) {
            in.ty = PTY_U64;
            in.sign = 1;
        } else if (strcmp(ts, "s8x4") == 0) {
            in.ty = PTY_U32;
            in.sign = 1;
            in.pack = 4;
        } else if (strcmp(ts, "f32") == 0) {
            in.ty = PTY_F32;
        } else if (strcmp(ts, "f64") == 0) {
            in.ty = PTY_F64;
        } else if (strcmp(ts, "f16") == 0) {
            in.ty = PTY_F16;
        } else if (strcmp(ts, "f16x2") == 0) {
            in.ty = PTY_F16X2;
        } else if (strcmp(ts, "bf16") == 0) {
            in.ty = PTY_BF16;
            if (in.ftz) {
                perr(p, opline, "bf16 neg has no .ftz");
                return -1;
            }
        } else if (strcmp(ts, "bf16x2") == 0) {
            in.ty = PTY_BF16X2;
            if (in.ftz) {
                perr(p, opline, "bf16x2 neg has no .ftz");
                return -1;
            }
        } else {
            perr(p, opline, "neg supports .s8x4/.s16/.s32/.s64/.f16/.f16x2/.bf16/.bf16x2/.f32/.f64 and .ftz f16/f16x2/f32");
            return -1;
        }
        in.op = POP_NEG;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        in.nsrc = 1;
        if (expect_punct(p, ';', "after neg") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if ((strcmp(parts[0], "min") == 0 || strcmp(parts[0], "max") == 0) &&
        (np == 2 || (np == 3 && (strcmp(parts[1], "relu") == 0 ||
                                 strcmp(parts[1], "ftz") == 0)))) {
        // Integer/float min/max. PTX 9.3 adds packed lanes and .relu
        // (clamp a negative signed result to 0); .ftz is f32-only.
        int relu = np == 3 && strcmp(parts[1], "relu") == 0;
        int ftz = np == 3 && strcmp(parts[1], "ftz") == 0;
        const char *ts = np == 3 ? parts[2] : parts[1];
        in.op = POP_MINMAX;
        in.is_max = parts[0][1] == 'a';
        in.sign = 0;
        in.pack = 0;
        in.relu = relu;
        in.ftz = ftz;
        if (strcmp(ts, "u32") == 0 || strcmp(ts, "b32") == 0) {
            in.ty = PTY_U32;
        } else if (strcmp(ts, "s32") == 0) {
            in.ty = PTY_U32;
            in.sign = 1;
        } else if (strcmp(ts, "u64") == 0 || strcmp(ts, "b64") == 0) {
            in.ty = PTY_U64;
        } else if (strcmp(ts, "s64") == 0) {
            in.ty = PTY_U64;
            in.sign = 1;
        } else if (strcmp(ts, "f32") == 0) {
            in.ty = PTY_F32;
        } else if (strcmp(ts, "f64") == 0) {
            in.ty = PTY_F64;
        } else if (strcmp(ts, "f16") == 0) {
            in.ty = PTY_F16;
        } else if (strcmp(ts, "f16x2") == 0) {
            in.ty = PTY_F16X2;
        } else if (strcmp(ts, "bf16") == 0) {
            in.ty = PTY_BF16;
        } else if (strcmp(ts, "bf16x2") == 0) {
            in.ty = PTY_BF16X2;
        } else if (strcmp(ts, "u16") == 0 || strcmp(ts, "b16") == 0) {
            in.ty = PTY_U16;
        } else if (strcmp(ts, "s16") == 0) {
            in.ty = PTY_U16;
            in.sign = 1;
        } else if (strcmp(ts, "u8") == 0 || strcmp(ts, "b8") == 0) {
            in.ty = PTY_U8;
        } else if (strcmp(ts, "s8") == 0) {
            in.ty = PTY_U8;
            in.sign = 1;
        } else if (strcmp(ts, "u16x2") == 0 || strcmp(ts, "s16x2") == 0) {
            in.ty = PTY_U32;
            in.pack = 2;
            in.sign = ts[0] == 's';
        } else if (strcmp(ts, "u8x4") == 0 || strcmp(ts, "s8x4") == 0) {
            in.ty = PTY_U32;
            in.pack = 4;
            in.sign = ts[0] == 's';
        } else {
            perr(p, opline, "min/max supports scalar int/float types, .u16x2/.s16x2/.u8x4/.s8x4, and .relu.s16x2/.relu.s8x4/.relu.s32");
            return -1;
        }
        if (ftz && in.ty != PTY_F32 && in.ty != PTY_F16 &&
            in.ty != PTY_F16X2) {
            perr(p, opline, "min.ftz/max.ftz supports .f16/.f16x2/.f32 only");
            return -1;
        }
        if (relu && (!in.sign ||
                     (in.pack == 0 && in.ty != PTY_U32))) {
            perr(p, opline, ".relu applies only to .s32/.s16x2/.s8x4");
            return -1;
        }
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (expect_punct(p, ';', "after min/max") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "mad") == 0 && np == 3 && strcmp(parts[1], "lo") == 0) {
        in.op = POP_MADLO;
        // .u32 and .s32 wrap identically; .u64/.s64 wrap at 64 bits.
        if (strcmp(parts[2], "u16") == 0 || strcmp(parts[2], "s16") == 0) {
            in.ty = PTY_U16;
            in.sign = parts[2][0] == 's';
        } else if (strcmp(parts[2], "u32") == 0 ||
                   strcmp(parts[2], "s32") == 0) {
            in.ty = PTY_U32;
            in.sign = parts[2][0] == 's';
        } else if (strcmp(parts[2], "u64") == 0 ||
                   strcmp(parts[2], "s64") == 0) {
            in.ty = PTY_U64;
            in.sign = parts[2][0] == 's';
        } else {
            perr(p, opline, "mad.lo supports .u16/.s16/.u32/.s32/.u64/.s64 only");
            return -1;
        }
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        for (int i = 0; i < 3; i++) {
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[i]) != 0)
                return -1;
            // Matches ptxas: special registers are not allowed as mad
            // sources (move to a plain register first).
            if (in.src[i].kind == OPD_SPECIAL) {
                perr(p, opline, "special register argument not allowed for instruction 'mad.lo'");
                return -1;
            }
        }
        in.nsrc = 3;
        if (expect_punct(p, ';', "after mad") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "mad") == 0 && np >= 3 &&
        strcmp(parts[1], "hi") == 0) {
        // mad.hi.{u16,s16,u32,s32,u64,s64} and mad.hi.sat.s32.
        int sat = 0;
        const char *ts;
        if (np == 3) {
            ts = parts[2];
        } else if (np == 4 && strcmp(parts[2], "sat") == 0) {
            sat = 1;
            ts = parts[3];
        } else {
            perr(p, opline, "mad.hi supports .hi.type and .hi.sat.s32");
            return -1;
        }
        if (strcmp(ts, "u16") == 0 || strcmp(ts, "s16") == 0) {
            in.ty = PTY_U16;
            in.sign = ts[0] == 's';
        } else if (strcmp(ts, "u32") == 0 || strcmp(ts, "s32") == 0) {
            in.ty = PTY_U32;
            in.sign = ts[0] == 's';
        } else if (strcmp(ts, "u64") == 0 || strcmp(ts, "s64") == 0) {
            in.ty = PTY_U64;
            in.sign = ts[0] == 's';
        } else {
            perr(p, opline, "mad.hi supports .u16/.s16/.u32/.s32/.u64/.s64 only");
            return -1;
        }
        if (sat && !(in.ty == PTY_U32 && in.sign)) {
            perr(p, opline, "mad.hi.sat is only valid for .s32");
            return -1;
        }
        in.op = POP_MADLO;
        in.is_hi = 1;
        in.sat = sat;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        for (int i = 0; i < 3; i++) {
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[i]) != 0)
                return -1;
            if (in.src[i].kind == OPD_SPECIAL) {
                perr(p, opline, "special register argument not allowed for instruction 'mad.hi'");
                return -1;
            }
        }
        in.nsrc = 3;
        if (expect_punct(p, ';', "after mad.hi") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "cvt") == 0 && np == 3) {
        in.op = POP_CVT;
        in.sign = 0;
        // Integer conversions (any width combo, exact spellings): the
        // sign follows the source (signed extends, unsigned
        // zero-extends, narrowing truncates); same-width pairs copy.
        ptx_ty_t dt, st;
        int ssign = 0, dsign = 0;
        if (cvt_int_ty(parts[1], &dt, &dsign) == 0 &&
            cvt_int_ty(parts[2], &st, &ssign) == 0) {
            in.ty = dt;
            in.ty2 = st;
            in.sign = ssign;
            in.dsign = dsign;
        } else if (strcmp(parts[1], "f64") == 0 &&
                   strcmp(parts[2], "f32") == 0) {
            // Exact widening; no rounding modifier in practice (numba).
            in.ty = PTY_F64;
            in.ty2 = PTY_F32;
        } else if (strcmp(parts[1], "f64") == 0 &&
                   strcmp(parts[2], "f16") == 0) {
            // Exact widening (numba f16 idiom: ld.u16 bits + widen).
            in.ty = PTY_F64;
            in.ty2 = PTY_F16;
        } else if (strcmp(parts[1], "f32") == 0 &&
                   strcmp(parts[2], "f16") == 0) {
            // Exact widening (f16 values are exact in f32).
            in.ty = PTY_F32;
            in.ty2 = PTY_F16;
        } else {
            perr(p, opline, "only integer cvts, cvt.f64.f32 and cvt.f32.f16 supported (other float cvts need .rn; no int<->float cvt)");
            return -1;
        }
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        in.nsrc = 1;
        if (expect_punct(p, ';', "after cvt") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "cvt") == 0 && strcmp(parts[1], "pack") == 0 &&
        strcmp(parts[2], "sat") == 0 &&
        ((np == 5 && strcmp(parts[4], "s32") == 0) ||
         (np == 6 && strcmp(parts[4], "s32") == 0 &&
          strcmp(parts[5], "b32") == 0))) {
        // cvt.pack.sat.{u16|s16}.s32 d, a, b          (no c)
        // cvt.pack.sat.{u8|s8|u4|s4|u2|s2}.s32.b32 d, a, b, c
        int pbits = 0;
        if (strcmp(parts[3], "u16") == 0) pbits = 16;
        else if (strcmp(parts[3], "s16") == 0) pbits = -16;
        else if (strcmp(parts[3], "u8") == 0) pbits = 8;
        else if (strcmp(parts[3], "s8") == 0) pbits = -8;
        else if (strcmp(parts[3], "u4") == 0) pbits = 4;
        else if (strcmp(parts[3], "s4") == 0) pbits = -4;
        else if (strcmp(parts[3], "u2") == 0) pbits = 2;
        else if (strcmp(parts[3], "s2") == 0) pbits = -2;
        if (pbits == 0 || (np == 5 && (pbits == 8 || pbits == 4 ||
                                       pbits == 2)) ||
            (np == 6 && (pbits == 16 || pbits == -16))) {
            char m[640];
            snprintf(m, sizeof(m), "bad opcode '%s'", opname);
            perr(p, opline, m);
            return -1;
        }
        in.op = POP_CVTPACK8;
        in.ty = PTY_U32;
        in.sign = pbits < 0 ? 1 : 0;
        in.packbits = pbits < 0 ? -pbits : pbits;
        in.sat = 1;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        for (int i = 0; i < np - 3; i++) {
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[i]) != 0)
                return -1;
        }
        in.nsrc = np - 3;
        if (expect_punct(p, ';', "after cvt.pack.sat") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "cvt") == 0 && np == 4 && strcmp(parts[1], "ftz") == 0 &&
        ((strcmp(parts[2], "f64") == 0 && strcmp(parts[3], "f32") == 0) ||
         (strcmp(parts[2], "f32") == 0 && strcmp(parts[3], "f32") == 0))) {
        // Flush-to-zero widening (denormal f32 inputs flush; the exact
        // widening itself never creates f64 denormals from normals) and
        // flush-to-zero f32 identity (PhysX fluid hashing emits it).
        in.op = POP_CVT;
        in.ty = strcmp(parts[2], "f64") == 0 ? PTY_F64 : PTY_F32;
        in.ty2 = PTY_F32;
        in.ftz = 1;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        in.nsrc = 1;
        if (expect_punct(p, ';', "after cvt.ftz") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "cvt") == 0 && np == 4 &&
        (strcmp(parts[1], "rn") == 0 || strcmp(parts[1], "rna") == 0) &&
        strcmp(parts[2], "tf32") == 0 && strcmp(parts[3], "f32") == 0) {
        if (p->sm && p->sm < 90) {
            perr(p, opline, "cvt.tf32 requires sm_90 or higher");
            return -1;
        }
        in.op = POP_CVT;
        in.ty = PTY_U32;
        in.ty2 = PTY_F32;
        in.tf32 = 1;
        in.rnd = strcmp(parts[1], "rna") == 0 ? 1 : 0;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        in.nsrc = 1;
        if (expect_punct(p, ';', "after cvt.tf32") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "cvt") == 0 && (np == 4 || np == 5) &&
        (strcmp(parts[1], "rn") == 0 || strcmp(parts[1], "rz") == 0 ||
         strcmp(parts[1], "rm") == 0 || strcmp(parts[1], "rp") == 0) &&
        ((np == 4 &&
          ((strcmp(parts[2], "f32") == 0 && strcmp(parts[3], "f32") == 0) ||
           (strcmp(parts[2], "f64") == 0 && strcmp(parts[3], "f64") == 0))) ||
         (np == 5 && strcmp(parts[2], "ftz") == 0 &&
          ((strcmp(parts[3], "f32") == 0 && strcmp(parts[4], "f32") == 0) ||
           (strcmp(parts[3], "f64") == 0 && strcmp(parts[4], "f64") == 0))))) {
        // Legacy float identity rounding (pre-PTX-3.x `rm` spelling):
        // cvt.rm[.ftz].f32.f32 rounds to integer-valued float. nvcc of
        // the CUDA 6.5 era emits these for sm_1x targets (PhysX CalcHash
        // uses cvt.rm.ftz.f32.f32 for grid hashing). Lower to POP_RND
        // exactly like the rmi/rzi/rpi forms below.
        int rnd = strcmp(parts[1], "rn") == 0 ? 1
                : strcmp(parts[1], "rz") == 0 ? 2
                : strcmp(parts[1], "rm") == 0 ? 3 : 4;
        int is64 = (np == 4 && strcmp(parts[2], "f64") == 0) ||
                   (np == 5 && strcmp(parts[3], "f64") == 0);
        in.op = POP_RND;
        in.ty = is64 ? PTY_F64 : PTY_F32;
        in.rnd = rnd;
        in.ftz = (np == 5) ? 1 : 0;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        in.nsrc = 1;
        if (expect_punct(p, ';', "after cvt.<rnd>") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "cvt") == 0 && np == 4 &&
        (strcmp(parts[1], "rn") == 0 || strcmp(parts[1], "rz") == 0 ||
         strcmp(parts[1], "rm") == 0 || strcmp(parts[1], "rp") == 0) &&
        strcmp(parts[2], "f16x2") != 0 && strcmp(parts[2], "bf16x2") != 0) {
        // Float narrowing is .rn-only; int->float supports all four
        // rounding directions.  Packed f16x2/bf16x2 forms are parsed
        // by the arm below.
        int rmode = strcmp(parts[1], "rn") == 0 ? 0
                    : strcmp(parts[1], "rz") == 0 ? 1
                    : strcmp(parts[1], "rm") == 0 ? 2 : 3;
        in.op = POP_CVT;
        in.sign = 0;
        if (strcmp(parts[2], "f32") == 0 && strcmp(parts[3], "f64") == 0) {
            if (rmode != 0) {
                perr(p, opline, "directed float-to-float cvt is not supported");
                return -1;
            }
            in.ty = PTY_F32;
            in.ty2 = PTY_F64;
        } else if (strcmp(parts[2], "f64") == 0 &&
                   strcmp(parts[3], "f32") == 0) {
            if (rmode != 0) {
                perr(p, opline, "directed float-to-float cvt is not supported");
                return -1;
            }
            in.ty = PTY_F64;
            in.ty2 = PTY_F32;
        } else if (strcmp(parts[2], "f32") == 0 &&
                   strcmp(parts[3], "f16") == 0) {
            if (rmode != 0) {
                perr(p, opline, "directed float-to-float cvt is not supported");
                return -1;
            }
            in.ty = PTY_F32;
            in.ty2 = PTY_F16;
        } else if (strcmp(parts[2], "f64") == 0 &&
                   strcmp(parts[3], "f16") == 0) {
            if (rmode != 0) {
                perr(p, opline, "directed float-to-float cvt is not supported");
                return -1;
            }
            in.ty = PTY_F64;
            in.ty2 = PTY_F16;
        } else if (strcmp(parts[2], "f16") == 0 &&
                   strcmp(parts[3], "f32") == 0) {
            if (rmode != 0) {
                perr(p, opline, "directed float-to-float cvt is not supported");
                return -1;
            }
            // Round-to-nearest narrowing to half (exact algorithm,
            // no capability needed; denormals handled, see emitter).
            in.ty = PTY_F16;
            in.ty2 = PTY_F32;
        } else if (strcmp(parts[2], "f16") == 0 &&
                   strcmp(parts[3], "f64") == 0) {
            if (rmode != 0) {
                perr(p, opline, "directed float-to-float cvt is not supported");
                return -1;
            }
            // Round-to-nearest narrowing (numba f16 idiom: narrow the
            // f64 accumulator straight to half bits).
            in.ty = PTY_F16;
            in.ty2 = PTY_F64;
        } else if ((strcmp(parts[2], "f32") == 0 ||
                    strcmp(parts[2], "f64") == 0) &&
                   (strcmp(parts[3], "u8") == 0 ||
                    strcmp(parts[3], "s8") == 0 ||
                    strcmp(parts[3], "u16") == 0 ||
                    strcmp(parts[3], "s16") == 0 ||
                    strcmp(parts[3], "u32") == 0 ||
                    strcmp(parts[3], "s32") == 0 ||
                    strcmp(parts[3], "u64") == 0 ||
                    strcmp(parts[3], "s64") == 0)) {
            // Integer to float (round-to-nearest).
            ptx_ty_t st;
            int ss = 0;
            if (cvt_int_ty(parts[3], &st, &ss) != 0) {
                perr(p, opline, "bad cvt.rn integer source");
                return -1;
            }
            in.ty = strcmp(parts[2], "f64") == 0 ? PTY_F64 : PTY_F32;
            in.ty2 = st;
            in.sign = ss;
            in.rnd = rmode;
        } else if (strcmp(parts[2], "f16") == 0 &&
                   (strcmp(parts[3], "u8") == 0 ||
                    strcmp(parts[3], "s8") == 0 ||
                    strcmp(parts[3], "u16") == 0 ||
                    strcmp(parts[3], "s16") == 0 ||
                    strcmp(parts[3], "u32") == 0 ||
                    strcmp(parts[3], "s32") == 0)) {
            ptx_ty_t st;
            int ss = 0;
            if (cvt_int_ty(parts[3], &st, &ss) != 0) {
                perr(p, opline, "bad cvt.rn.f16 integer source");
                return -1;
            }
            in.ty = PTY_F16;
            in.ty2 = st;
            in.sign = ss;
            in.rnd = rmode;
        } else if (strcmp(parts[2], "bf16") == 0 &&
                   strcmp(parts[3], "f32") == 0) {
            if (rmode != 0) {
                perr(p, opline, "directed float-to-float cvt is not supported");
                return -1;
            }
            // Round-to-nearest narrowing to brain-float bits (exact
            // integer algorithm, no capability needed; denormals and
            // NaN handled, see emitter).
            in.ty = PTY_BF16;
            in.ty2 = PTY_F32;
        } else {
            char cvtmsg[128];
            snprintf(cvtmsg, sizeof(cvtmsg),
                     "unsupported cvt.rn.%s.%s", parts[2], parts[3]);
            perr(p, opline, cvtmsg);
            return -1;
        }
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        in.nsrc = 1;
        if (expect_punct(p, ';', "after cvt.rn") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "cvt") == 0 && np == 4 &&
        strcmp(parts[1], "rn") == 0 && strcmp(parts[2], "f16x2") == 0 &&
        (strcmp(parts[3], "e4m3x2") == 0 || strcmp(parts[3], "e5m2x2") == 0)) {
        // FP8 x2 -> f16x2 widening: dst is a .b32 pattern register, src
        // a .b16 pattern register (two FP8 bytes; low byte -> low half).
        in.op = POP_FP8_TO_F16X2;
        in.fp8 = (strcmp(parts[3], "e4m3x2") == 0) ? 1 : 2;
        in.ty = PTY_U32;
        in.ty2 = PTY_U16;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        in.nsrc = 1;
        if (expect_punct(p, ';', "after cvt.rn.f16x2.fp8") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "cvt") == 0 && (np == 5 || np == 6) &&
        strcmp(parts[1], "rn") == 0 &&
        ((np == 5 && strcmp(parts[2], "satfinite") == 0 &&
          (strcmp(parts[3], "f16") == 0 || strcmp(parts[3], "f16x2") == 0) &&
          strcmp(parts[4], "f32") == 0) ||
         (np == 6 && strcmp(parts[2], "relu") == 0 &&
          strcmp(parts[3], "satfinite") == 0 &&
          (strcmp(parts[4], "f16") == 0 || strcmp(parts[4], "f16x2") == 0) &&
          strcmp(parts[5], "f32") == 0))) {
        // .satfinite narrowing to half: Infinity and every value above
        // MAX_NORM clamp to +-0x7bff; NaN stays NaN. Two sources for
        // the packed f16x2 form (first source -> high half).
        int ti = (np == 5) ? 3 : 4;
        int packed = strcmp(parts[ti], "f16x2") == 0;
        in.op = packed ? POP_CVTPACK : POP_CVT;
        in.ty = packed ? PTY_F16X2 : PTY_F16;
        in.ty2 = PTY_F32;
        in.sat = 1;
        in.relu = (np == 6);
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (packed &&
            (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0))
            return -1;
        in.nsrc = packed ? 2 : 1;
        if (expect_punct(p, ';', "after cvt.rn.satfinite") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "cvt") == 0 && np == 5 &&
        strcmp(parts[1], "rn") == 0 && strcmp(parts[2], "satfinite") == 0 &&
        (strcmp(parts[3], "e4m3x2") == 0 || strcmp(parts[3], "e5m2x2") == 0) &&
        strcmp(parts[4], "f32") == 0) {
        // Packed FP8 narrowing: two f32 -> .b16 pattern register. The
        // first source lands in the HIGH byte, second in the low byte
        // (hardware order, pinned by ZLUDA's vectors).
        in.op = POP_F32_TO_FP8X2;
        in.fp8 = (strcmp(parts[3], "e4m3x2") == 0) ? 1 : 2;
        in.ty = PTY_U16;
        in.ty2 = PTY_F32;
        in.sat = 1;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (expect_punct(p, ';', "after cvt.rn.satfinite.fp8") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "cvt") == 0 && strcmp(parts[1], "rn") == 0 &&
        ((np == 4 && (strcmp(parts[2], "f16x2") == 0 ||
                      strcmp(parts[2], "bf16x2") == 0) &&
          strcmp(parts[3], "f32") == 0) ||
         (np == 5 && strcmp(parts[2], "relu") == 0 &&
          strcmp(parts[3], "f16x2") == 0 && strcmp(parts[4], "f32") == 0))) {
        // Packed float conversion: narrow two f32 to halves and pack
        // (HIGH half = first source, matching ZLUDA's InsertElement
        // order and the corpus vectors). Optional .relu takes
        // max(+0) per lane first (NaN maps to +0 here; the corpus
        // deliberately skips NaN: AMD hardware disagrees anyway).
        // e4m3/e5m2 (FP8) have their own arms above (bi-directional).
        int is_bf16 = (np == 4 && strcmp(parts[2], "bf16x2") == 0);
        in.op = POP_CVTPACK;
        in.ty = is_bf16 ? PTY_BF16X2 : PTY_F16X2;
        in.relu = (np == 5) ? 1 : 0;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (expect_punct(p, ';', "after cvt.rn.packed") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "cvt") == 0 && (np == 4 || np == 5) &&
        (strcmp(parts[1], "rni") == 0 || strcmp(parts[1], "rzi") == 0 ||
         strcmp(parts[1], "rmi") == 0 || strcmp(parts[1], "rpi") == 0)) {
        // Integer-directed rounding. rni = nearest-even, rzi = toward
        // zero, rmi = toward -inf, rpi = toward +inf (exact integer bit
        // manipulation, no GLSL import). Optional .ftz flushes denormal
        // inputs first — observable for rmi/rpi, where a flushed
        // denormal rounds to 0 instead of +-1.
        int rnd = strcmp(parts[1], "rni") == 0 ? 1
                : strcmp(parts[1], "rzi") == 0 ? 2
                : strcmp(parts[1], "rmi") == 0 ? 3
                                              : 4;
        int ti = 2, has_ftz = 0;
        if (np == 5) {
            if (strcmp(parts[2], "ftz") != 0) {
                perr(p, opline, "only .ftz may follow the rounding mode");
                return -1;
            }
            has_ftz = 1;
            ti = 3;
        }
        if ((strcmp(parts[ti], "f32") == 0 ||
             strcmp(parts[ti], "f64") == 0) &&
            strcmp(parts[ti], parts[ti + 1]) == 0) {
            // Round to integer-valued float (no conversion).
            in.op = POP_RND;
            in.ty = strcmp(parts[ti], "f64") == 0 ? PTY_F64 : PTY_F32;
            in.rnd = rnd;
            in.ftz = has_ftz;
        } else if (strcmp(parts[ti + 1], "f32") == 0 ||
                   strcmp(parts[ti + 1], "f64") == 0) {
            // Float-to-int (PTX saturates, like llvm.fptoui/fptosi.sat;
            // NaN converts to 0). 16/32-bit dests only.
            ptx_ty_t dt;
            int dsign = 0;
            if (cvt_int_ty(parts[ti], &dt, &dsign) != 0 ||
                (dt != PTY_U8 && dt != PTY_U16 && dt != PTY_U32 &&
                 dt != PTY_U64)) {
                perr(p, opline, "cvt.rni/rzi/rmi/rpi to int supports 8/16/32/64-bit integer destinations");
                return -1;
            }
            int srcf64 = strcmp(parts[ti + 1], "f64") == 0;
            if (!srcf64 && strcmp(parts[ti + 1], "f32") != 0) {
                perr(p, opline, "cvt.<rnd> to int needs an .f32 or .f64 source");
                return -1;
            }
            in.op = POP_CVT;
            in.ty = dt;
            in.ty2 = srcf64 ? PTY_F64 : PTY_F32;
            in.sign = dsign;
            in.rnd = rnd;
            in.ftz = has_ftz;
        } else {
            perr(p, opline, "only cvt.<rnd>[.ftz].f32.f32 and cvt.<rnd>[.ftz].u16/s16/u32/s32/u64/s64.{f32,f64} supported");
            return -1;
        }
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        in.nsrc = 1;
        if (expect_punct(p, ';', "after cvt.<rnd>") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "cvt") == 0 && np == 4 && strcmp(parts[1], "sat") == 0) {
        // Saturating conversion: integer clamp/narrow, or float clamp to [0,1].
        ptx_ty_t dt = PTY_U32, st = PTY_U32;
        int dsign = 0, ssign = 0;
        int is_int = (cvt_int_ty(parts[2], &dt, &dsign) == 0 &&
                      cvt_int_ty(parts[3], &st, &ssign) == 0);
        int is_f32 = (strcmp(parts[2], "f32") == 0 && strcmp(parts[3], "f32") == 0);
        int is_f64 = (strcmp(parts[2], "f64") == 0 && strcmp(parts[3], "f64") == 0);
        if (!is_int && !is_f32 && !is_f64) {
            perr(p, opline, "cvt.sat needs integer, f32.f32, or f64.f64 types");
            return -1;
        }
        in.op = POP_CVT;
        if (is_int) {
            in.ty = dt;
            in.ty2 = st;
            in.sign = ssign;
            in.dsign = dsign;
        } else {
            in.ty = is_f64 ? PTY_F64 : PTY_F32;
            in.ty2 = in.ty;
        }
        in.sat = 1;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        in.nsrc = 1;
        if (expect_punct(p, ';', "after cvt.sat") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "set") == 0 && np >= 4 && np <= 6) {
        // Half-precision comparison matrix (f16/f16x2). bf16 needs
        // sm_90 and is target-gated. Destination types supported here:
        // integer masks (.u16/.s16/.u32/.s32/.b16/.b32), .f16 for a
        // scalar source, and .f16x2 for a packed source.
        int idx = 1;
        if (map_cmp(parts[idx], &in.cmp) != 0) {
            if (strcmp(parts[idx], "nan") == 0)
                in.cmp = PCM_NAN;
            else if (strcmp(parts[idx], "num") == 0)
                in.cmp = PCM_NUM;
            else {
                perr(p, opline, "unsupported set comparison");
                return -1;
            }
        }
        idx++;
        if (idx < np - 2 &&
            (strcmp(parts[idx], "and") == 0 ||
             strcmp(parts[idx], "or") == 0 ||
             strcmp(parts[idx], "xor") == 0)) {
            in.bop = strcmp(parts[idx], "and") == 0 ? 1
                     : strcmp(parts[idx], "or") == 0 ? 2 : 3;
            idx++;
        }
        if (idx < np - 2 && strcmp(parts[idx], "ftz") == 0) {
            in.ftz = 1;
            idx++;
        }
        if (idx != np - 2) {
            perr(p, opline, "bad set qualifier order");
            return -1;
        }
        const char *dts = parts[idx];
        const char *sts = parts[idx + 1];
        if (strcmp(sts, "f16") == 0) {
            in.ty2 = PTY_F16;
        } else if (strcmp(sts, "f16x2") == 0) {
            in.ty2 = PTY_F16X2;
        } else if (strcmp(sts, "bf16") == 0 ||
                   strcmp(sts, "bf16x2") == 0) {
            perr(p, opline, "bf16 comparisons require sm_90 and are not implemented");
            return -1;
        } else {
            perr(p, opline, "set half sources must be .f16/.f16x2");
            return -1;
        }
        if (strcmp(dts, "u32") == 0 || strcmp(dts, "s32") == 0 ||
            strcmp(dts, "b32") == 0)
            in.ty = PTY_U32;
        else if (strcmp(dts, "u16") == 0 || strcmp(dts, "s16") == 0 ||
                 strcmp(dts, "b16") == 0)
            in.ty = PTY_U16;
        else if (strcmp(dts, "f16") == 0 && in.ty2 == PTY_F16)
            in.ty = PTY_F16;
        else if (strcmp(dts, "f16x2") == 0 && in.ty2 == PTY_F16X2)
            in.ty = PTY_F16X2;
        else {
            perr(p, opline, "set destination must be integer mask, .f16, or .f16x2");
            return -1;
        }
        in.op = POP_SET;
        in.sign = 0;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (in.bop) {
            if (expect_comma(p) != 0)
                return -1;
            if (at_punct(p, '!'))
                in.bneg = 1;
            if (parse_operand(p, &in.src[2]) != 0)
                return -1;
            if (in.src[2].kind != OPD_REG) {
                perr(p, opline, "set boolean source must be a predicate register");
                return -1;
            }
            in.nsrc = 3;
        }
        if (expect_punct(p, ';', "after set") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "setp") == 0 && np == 4 && strcmp(parts[2], "ftz") == 0) {
        // Flush-to-zero comparison (denormal inputs flush before the
        // compare; f32 only). Any comparison, ordered or unordered.
        if (strcmp(parts[1], "nan") == 0) {
            in.cmp = PCM_NAN;
        } else if (strcmp(parts[1], "num") == 0) {
            in.cmp = PCM_NUM;
        } else if (map_cmp(parts[1], &in.cmp) != 0) {
            perr(p, opline, "unsupported setp comparison");
            return -1;
        }
        if (strcmp(parts[3], "f32") == 0) {
            in.ty = PTY_F32;
        } else if (strcmp(parts[3], "f16") == 0) {
            in.ty = PTY_F16;
        } else {
            perr(p, opline, "setp.ftz supports .f16/.f32 only");
            return -1;
        }
        in.op = POP_SETP;
        in.sign = 0;
        in.ftz = 1;
        if (parse_dst(p, in.dst, sizeof(in.dst), 1) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (expect_punct(p, ';', "after setp.ftz") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "setp") == 0 && np == 3) {
        in.op = POP_SETP;
        if (strcmp(parts[1], "nan") == 0 || strcmp(parts[1], "num") == 0) {
            // Unordered NaN tests (ZLUDA corpus): nan = either operand
            // NaN, num = both numeric. Ordered comparisons stay exact;
            // boolean-op forms are parsed in the np==4 branch below.
            in.cmp = parts[1][1] == 'a' ? PCM_NAN : PCM_NUM;
            if (strcmp(parts[2], "f32") == 0) {
                in.ty = PTY_F32;
            } else if (strcmp(parts[2], "f64") == 0) {
                in.ty = PTY_F64;
            } else if (strcmp(parts[2], "f16") == 0) {
                in.ty = PTY_F16;
            } else if (strcmp(parts[2], "f16x2") == 0) {
                in.ty = PTY_F16X2;
            } else {
                perr(p, opline, "setp.nan/num supports .f16/.f16x2/.f32/.f64 only");
                return -1;
            }
            in.sign = 0;
        } else {
            if (map_cmp(parts[1], &in.cmp) != 0) {
                perr(p, opline, "unsupported setp comparison");
                return -1;
            }
            if (setp_ty(p, parts[2], &in.ty, &in.sign, opline) != 0)
                return -1;
        } // end ordered-comparison ladder (nan/num skip it)
        if (parse_dst(p, in.dst, sizeof(in.dst), 1) != 0)
            return -1;
        // Dual predicate destination (`p|q`: q takes the complement).
        if (at_punct(p, '|')) {
            ptx_tok_t q;
            if (next(p, &q) != 0)
                return -1;
            if (q.kind != T_IDENT) {
                perr(p, q.line, "expected second predicate after '|'");
                return -1;
            }
            if (copy_name(in.dst2, sizeof(in.dst2), q.text, p, q.line) != 0)
                return -1;
        }
        if (in.ty == PTY_F16X2 && !in.dst2[0]) {
            perr(p, opline, "setp.f16x2 needs a dual p|q destination");
            return -1;
        }
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (expect_punct(p, ';', "after setp") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "setp") == 0 && np == 4 &&
        (strcmp(parts[2], "and") == 0 || strcmp(parts[2], "or") == 0 ||
         strcmp(parts[2], "xor") == 0)) {
        // setp with a boolean operator over a third predicate source:
        // `setp.lt.and.u32 p, a, b, [!]q` (any comparison, ordered or
        // unordered; dual `p|q` destinations allowed).
        if (strcmp(parts[1], "nan") == 0) {
            in.cmp = PCM_NAN;
        } else if (strcmp(parts[1], "num") == 0) {
            in.cmp = PCM_NUM;
        } else if (map_cmp(parts[1], &in.cmp) != 0) {
            perr(p, opline, "unsupported setp comparison");
            return -1;
        }
        if (setp_ty(p, parts[3], &in.ty, &in.sign, opline) != 0)
            return -1;
        in.op = POP_SETP;
        in.bop = strcmp(parts[2], "and") == 0 ? 1
                 : strcmp(parts[2], "or") == 0 ? 2
                                              : 3;
        if (parse_dst(p, in.dst, sizeof(in.dst), 1) != 0)
            return -1;
        if (at_punct(p, '|')) {
            ptx_tok_t q;
            if (next(p, &q) != 0)
                return -1;
            if (q.kind != T_IDENT) {
                perr(p, q.line, "expected second predicate after '|'");
                return -1;
            }
            if (copy_name(in.dst2, sizeof(in.dst2), q.text, p, q.line) != 0)
                return -1;
        }
        if (in.ty == PTY_F16X2 && !in.dst2[0]) {
            perr(p, opline, "setp.f16x2 needs a dual p|q destination");
            return -1;
        }
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        if (expect_comma(p) != 0)
            return -1;
        if (at_punct(p, '!'))
            in.bneg = 1;
        if (parse_operand(p, &in.src[2]) != 0)
            return -1;
        if (in.src[2].kind != OPD_REG) {
            perr(p, opline, "setp boolean source must be a predicate register");
            return -1;
        }
        in.nsrc = 3;
        if (expect_punct(p, ';', "after setp boolean op") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    // Scalarized vector loads/stores: `ld.v4.u32 v, [a]` (or brace
    // `{r0..}` dests) expands to N scalar loads at ascending byte
    // offsets; stores mirror. Spaces mirror the scalar arms (generic
    // plus global/shared/local, const loads only). Predicated vector
    // memory ops stay clean errors (no corpus need).
    if (strcmp(parts[0], "ldu") == 0 && (np == 2 || np == 3) &&
        !is_vec_type_token(parts[(np == 3 &&
                                  strcmp(parts[1], "global") == 0)
                                     ? 2
                                     : 1])) {
        // Uniform global load. The uniform-address guarantee is a
        // performance hint on this backend; scalar loads use the normal
        // global/generic path. Vector/.b128 forms fall through to the
        // generic vector ld parser below.
        int pi = 1;
        ptx_space_t sp = PSP_GENERIC;
        if (strcmp(parts[pi], "global") == 0) {
            sp = PSP_GLOBAL;
            pi++;
        }
        if (pi != np - 1) {
            perr(p, opline, "ldu form is ldu{.global}.type d, [a]");
            return -1;
        }
        ptx_ty_t ty;
        if (map_ty(parts[pi], &ty) != 0 ||
            (ty != PTY_U8 && ty != PTY_U16 && ty != PTY_U32 &&
             ty != PTY_U64 && ty != PTY_F32 && ty != PTY_F64)) {
            perr(p, opline, "ldu supports scalar .b8/.b16/.b32/.b64/.u*/.s*/.f32/.f64");
            return -1;
        }
        in.op = POP_LD;
        in.space = sp;
        in.ty = ty;
        in.mem_sem = ldst_mem_sem;
        in.mem_scope = ldst_mem_scope;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        in.nsrc = 1;
        if (expect_punct(p, ';', "after ldu") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "prefetch") == 0 ||
        strcmp(parts[0], "prefetchu") == 0) {
        // Cache prefetch hints; accept the documented scalar forms and
        // parse the address for validation, then drop the hint.
        if (strcmp(parts[0], "prefetchu") == 0) {
            if (np != 2 || strcmp(parts[1], "L1") != 0) {
                perr(p, opline, "prefetchu supports prefetchu.L1 [a] only");
                return -1;
            }
        } else if (np < 2 || np > 3) {
            perr(p, opline, "prefetch form is prefetch{.space}.level [a]");
            return -1;
        }
        in.op = POP_PREFETCH;
        if (parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (in.src[0].kind != OPD_ADDR) {
            perr(p, opline, "prefetch needs [address]");
            return -1;
        }
        in.nsrc = 1;
        if (expect_punct(p, ';', "after prefetch") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "applypriority") == 0 ||
        strcmp(parts[0], "discard") == 0) {
        // Cache priority/discard hints. They have no required effect on
        // results in this backend; validate the address and size.
        in.op = POP_HINT;
        if (parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (in.src[0].kind != OPD_ADDR ||
            expect_comma(p) != 0 ||
            parse_operand(p, &in.src[1]) != 0)
            return -1;
        if (in.src[1].kind != OPD_IMM || in.src[1].imm_is_float ||
            in.src[1].imm != 128) {
            perr(p, opline, "applypriority/discard size must be 128");
            return -1;
        }
        in.nsrc = 2;
        if (expect_punct(p, ';', "after cache hint") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "griddepcontrol") == 0 && np == 2 &&
        (strcmp(parts[1], "launch_dependents") == 0 ||
         strcmp(parts[1], "wait") == 0)) {
        // Dependent-grid scheduling hint; no observable effect in this
        // single-grid execution model.
        in.op = POP_HINT;
        in.nsrc = 0;
        if (expect_punct(p, ';', "after griddepcontrol") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "ld") == 0 || strcmp(parts[0], "st") == 0 ||
        strcmp(parts[0], "ldu") == 0) {
        // Skip memory order/scope qualifiers for vector ld/st
        int is_st = parts[0][1] == 't';
        int vpart_idx = -1;
        int epart_idx = -1;
        int b128 = 0;
        ptx_space_t space = PSP_GENERIC;
        int n_vec = 0;
        int n = 0;
        ptx_ty_t elem = PTY_U32;
        for (int i = 1; i < np; i++) {
            if (strcmp(parts[i], "b128") == 0) {
                vpart_idx = i;
                epart_idx = i;
                b128 = 1;
                break;
            }
            if (strcmp(parts[i], "v2") == 0 || strcmp(parts[i], "v4") == 0 ||
                strcmp(parts[i], "v8") == 0) {
                vpart_idx = i;
                if (i + 1 < np)
                    epart_idx = i + 1;
                break;
            }
        }
        if (vpart_idx >= 0 && epart_idx >= 0) {
            const char *vpart = parts[vpart_idx];
            const char *epart = parts[epart_idx];
            if (b128) {
                n_vec = 2;
                n = 2;
                elem = PTY_U64;
            } else {
                n_vec = vpart[1] - '0';
                n = n_vec;
                if (vec_elem_ty(epart, &elem) != 0) {
                    perr(p, opline, "vector ld/st needs u/s/b 8/16/32/64 element types");
                    return -1;
                }
            }
            // Check for space qualifier before vpart
            for (int i = 1; i < vpart_idx; i++) {
                if (strcmp(parts[i], "global") == 0)
                    space = PSP_GLOBAL;
                else if (strcmp(parts[i], "shared") == 0 ||
                         strcmp(parts[i], "shared::cta") == 0 ||
                         strcmp(parts[i], "shared::cluster") == 0)
                    space = PSP_SHARED;
                else if (strcmp(parts[i], "local") == 0)
                    space = PSP_LOCAL;
                else if (!is_st && strcmp(parts[i], "const") == 0)
                    space = PSP_CONST;
                else if (strcmp(parts[i], "acquire") == 0 || strcmp(parts[i], "release") == 0 ||
                         strcmp(parts[i], "relaxed") == 0 || strcmp(parts[i], "acq_rel") == 0 ||
                         strcmp(parts[i], "gpu") == 0 || strcmp(parts[i], "cta") == 0 ||
                         strcmp(parts[i], "cluster") == 0 || strcmp(parts[i], "sys") == 0 ||
                         strcmp(parts[i], "volatile") == 0 || strcmp(parts[i], "weak") == 0 ||
                         strcmp(parts[i], "mmio") == 0 || strcmp(parts[i], "unified") == 0 ||
                         strcmp(parts[i], "nc") == 0 || strcmp(parts[i], "ca") == 0 ||
                         strcmp(parts[i], "cg") == 0 || strcmp(parts[i], "cs") == 0 ||
                         strcmp(parts[i], "lu") == 0 || strcmp(parts[i], "cv") == 0) {
                    // skip qualifiers
                } else if (vpart_idx == 2 && i == 1) { // might be space in old form
                    if (strcmp(parts[i], "global") == 0) space = PSP_GLOBAL;
                    else if (strcmp(parts[i], "shared") == 0) space = PSP_SHARED;
                    else if (strcmp(parts[i], "local") == 0) space = PSP_LOCAL;
                    else if (!is_st && strcmp(parts[i], "const") == 0) space = PSP_CONST;
                    else {
                        // Could be other qualifier, continue
                    }
                }
            }
            // If space wasn't set by position-based logic and we have old style (np==4 with space at pos1)
            if (space == PSP_GENERIC && np == 4 && vpart_idx == 2) {
                if (strcmp(parts[1], "global") == 0)
                    space = PSP_GLOBAL;
                else if (strcmp(parts[1], "shared") == 0)
                    space = PSP_SHARED;
                else if (strcmp(parts[1], "local") == 0)
                    space = PSP_LOCAL;
                else if (!is_st && strcmp(parts[1], "const") == 0)
                    space = PSP_CONST;
            }
        }
        if (vpart_idx < 0 || epart_idx < 0)
            goto try_scalar_ldst;
        int w = vec_elem_bytes(elem);
        if (!is_st) {
            char dd[8][64];
            memset(dd, 0, sizeof(dd));
            ptx_opd_t addr;
            memset(&addr, 0, sizeof(addr));
            if (parse_vec_operand(p, dd, n, elem, opline) != 0)
                return -1;
            if (expect_comma(p) != 0 ||
                parse_operand(p, &addr) != 0)
                return -1;
            if (addr.kind != OPD_ADDR) {
                perr(p, opline, "ld address must use []");
                return -1;
            }
            if (space == PSP_CONST && addr.addr_reg[0])
                fn->uses_dynamic_const = 1;
            if (skip_unified_suffix(p) != 0)
                return -1;
            if (has_cache_hint) {
                ptx_opd_t pol;
                memset(&pol, 0, sizeof(pol));
                if (expect_comma(p) != 0 || parse_operand(p, &pol) != 0)
                    return -1;
            }
            if (expect_punct(p, ';', "after vector ld") != 0)
                return -1;
            for (int i = 0; i < n; i++) {
                ptx_instr_t li;
                memset(&li, 0, sizeof(li));
                li.op = POP_LD;
                li.ty = elem;
                li.space = space;
                li.mem_sem = ldst_mem_sem;
                li.mem_scope = ldst_mem_scope;
                li.line = opline;
                if (copy_name(li.dst, sizeof(li.dst), dd[i], p,
                              opline) != 0)
                    return -1;
                li.src[0] = addr;
                li.src[0].addr_imm += (int64_t)(i * w);
                li.nsrc = 1;
                snprintf(li.pred, sizeof(li.pred), "%s", pred);
                li.pred_neg = pred_neg;
                if (block_add_instr(fn, bi, &li, p, opline) != 0)
                    return -1;
            }
            return 0;
        }
        {
            char ss[8][64];
            ptx_opd_t addr;
            memset(ss, 0, sizeof(ss));
            memset(&addr, 0, sizeof(addr));
            if (parse_operand(p, &addr) != 0)
                return -1;
            if (addr.kind != OPD_ADDR) {
                perr(p, opline, "st address must use []");
                return -1;
            }
            if (expect_comma(p) != 0)
                return -1;
            if (parse_vec_operand(p, ss, n, elem, opline) != 0)
                return -1;
            if (has_cache_hint) {
                ptx_opd_t pol;
                memset(&pol, 0, sizeof(pol));
                if (expect_comma(p) != 0 || parse_operand(p, &pol) != 0)
                    return -1;
            }
            if (expect_punct(p, ';', "after vector st") != 0)
                return -1;
            for (int i = 0; i < n; i++) {
                ptx_instr_t si;
                memset(&si, 0, sizeof(si));
                si.op = POP_ST;
                si.ty = elem;
                si.space = space;
                si.mem_sem = ldst_mem_sem;
                si.mem_scope = ldst_mem_scope;
                si.line = opline;
                si.src[0] = addr;
                si.src[0].addr_imm += (int64_t)(i * w);
                si.src[1].kind = OPD_REG;
                if (copy_name(si.src[1].reg, sizeof(si.src[1].reg),
                              ss[i], p, opline) != 0)
                    return -1;
                si.nsrc = 2;
                snprintf(si.pred, sizeof(si.pred), "%s", pred);
                si.pred_neg = pred_neg;
                if (block_add_instr(fn, bi, &si, p, opline) != 0)
                    return -1;
            }
            return 0;
        }
    }
try_scalar_ldst:
    // Generic (stateless) loads/stores: treated as global when the address
    // register tracks to a kernel pointer (see emitter). numba emits these.
    if ((strcmp(parts[0], "ld") == 0 || strcmp(parts[0], "st") == 0) &&
        np == 2) {        int is_st = parts[0][1] == 't';
        in.op = is_st ? POP_ST : POP_LD;
        in.space = PSP_GENERIC;
        in.mem_sem = ldst_mem_sem;
        in.mem_scope = ldst_mem_scope;
        if (map_ty(parts[1], &in.ty) != 0 || in.ty == PTY_PRED ||
            in.ty == PTY_F16 || in.ty == PTY_BF16 ||
            in.ty == PTY_F16X2 || in.ty == PTY_BF16X2) {
            // No .f16 memory access in PTX (halves move as .u16 bits).
            perr(p, opline, "unsupported generic ld/st type");
            return -1;
        }
        if (is_st) {
            if (parse_operand(p, &in.src[0]) != 0)
                return -1;
            if (in.src[0].kind != OPD_ADDR) {
                perr(p, opline, "st address must use []");
                return -1;
            }
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
                return -1;
            in.nsrc = 2;
            if (is_st)
                st_f32_bits_fixup(fn, &in);
            if (has_cache_hint) {
                if (expect_comma(p) != 0 ||
                    parse_operand(p, &in.src[2]) != 0)
                    return -1;
                in.nsrc = 3;
            }
            if (expect_punct(p, ';', "after st") != 0)
                return -1;
        } else {
            if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
                return -1;
            if (in.op == POP_LD)
                ld_f32_bits_fixup(fn, &in);
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
                return -1;
            if (in.src[0].kind != OPD_ADDR) {
                perr(p, opline, "ld address must use []");
                return -1;
            }
            if (skip_unified_suffix(p) != 0)
                return -1;
            in.nsrc = 1;
            if (has_cache_hint) {
                if (expect_comma(p) != 0 ||
                    parse_operand(p, &in.src[1]) != 0)
                    return -1;
                in.nsrc = 2;
            }
            if (expect_punct(p, ';', "after ld") != 0)
                return -1;
        }
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "ld") == 0 && np == 3) {
        in.op = POP_LD;
        in.mem_sem = ldst_mem_sem;
        in.mem_scope = ldst_mem_scope;
        if (strcmp(parts[1], "param") == 0 ||
            strcmp(parts[1], "param::entry") == 0 ||
            strcmp(parts[1], "param::func") == 0)
            in.space = PSP_PARAM;
        else if (strcmp(parts[1], "global") == 0)
            in.space = PSP_GLOBAL;
        else if (strcmp(parts[1], "shared") == 0 ||
                 strcmp(parts[1], "shared::cta") == 0 ||
                 strcmp(parts[1], "shared::cluster") == 0)
            in.space = PSP_SHARED;
        else if (strcmp(parts[1], "local") == 0)
            in.space = PSP_LOCAL;
        else if (strcmp(parts[1], "const") == 0)
            in.space = PSP_CONST;
        else {
            perr(p, opline, "ld only supports .param/.global/.shared/.local/.const spaces");
            return -1;
        }
        if (map_ty(parts[2], &in.ty) != 0 || in.ty == PTY_PRED ||
            in.ty == PTY_F16 || in.ty == PTY_BF16 ||
            in.ty == PTY_F16X2 || in.ty == PTY_BF16X2) {
            // No .f16 memory access in PTX (halves move as .u16 bits
            // plus cvt); anything else unmapped is out of subset.
            perr(p, opline, "unsupported ld type");
            return -1;
        }
        if (strcmp(parts[2], "s8") == 0 || strcmp(parts[2], "s16") == 0 ||
            strcmp(parts[2], "s32") == 0 || strcmp(parts[2], "s64") == 0)
            in.sign = 1;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        ld_f32_bits_fixup(fn, &in);
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (in.src[0].kind != OPD_ADDR) {
            perr(p, opline, "ld address must use []");
            return -1;
        }
        if (in.space == PSP_CONST && in.src[0].addr_reg[0])
            fn->uses_dynamic_const = 1;
        if (skip_unified_suffix(p) != 0)
            return -1;
        // Direct shared-variable use ([var]/[var+N] with no address
        // register) resolves at emission; unknown names fail there.
        in.nsrc = 1;
        if (has_cache_hint) {
            if (expect_comma(p) != 0 ||
                parse_operand(p, &in.src[1]) != 0)
                return -1;
            in.nsrc = 2;
        }
        if (expect_punct(p, ';', "after ld") != 0)
            return -1;
        if (getenv("ACUDA_DEBUG_SYMBOLS") && in.src[0].kind == OPD_ADDR &&
            in.src[0].addr_reg[0] && strcmp(in.src[0].addr_reg, "%rd3") == 0 &&
            in.src[0].addr_imm == 12)
            fprintf(stderr, "ACuda: parse ld dst=%s sp=%d ty=%d reg=%s imm=%lld line=%d fn=%s\n",
                    in.dst, (int)in.space, (int)in.ty, in.src[0].addr_reg,
                    (long long)in.src[0].addr_imm, opline, fn->name);
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strcmp(parts[0], "st") == 0 && np == 3) {
        in.op = POP_ST;
        in.mem_sem = ldst_mem_sem;
        in.mem_scope = ldst_mem_scope;
        if (strcmp(parts[1], "param") == 0) {
            // Return-slot store (numba .funcABI) or call-slot store:
            // recorded for the post-parse pass, which drops writes to
            // the function's own return slot (kernel return slots are
            // not modeled) and wires callee returns/call slots to movs.
            in.space = PSP_PARAM;
            if (map_ty(parts[2], &in.ty) != 0) {
                perr(p, opline, "unsupported st.param type");
                return -1;
            }
            if (parse_operand(p, &in.src[0]) != 0)
                return -1;
            if (in.src[0].kind != OPD_ADDR || in.src[0].addr_reg[0]) {
                perr(p, opline, "st.param needs [param]");
                return -1;
            }
            int is_ret = 0;
            for (size_t pi = 0; pi < fn->nparams; pi++) {
                if (strcmp(fn->params[pi].name, in.src[0].param) == 0 &&
                    fn->params[pi].is_retval)
                    is_ret = 1;
            }
            if (!is_ret) {
                int is_slot = 0;
                for (size_t si = 0; si < fn->nslots; si++) {
                    if (strcmp(fn->slots[si].name, in.src[0].param) == 0)
                        is_slot = 1;
                }
                if (!is_slot) {
                    perr(p, opline, "st.param only supported for func_retval and call slots");
                    return -1;
                }
                if (in.src[0].addr_imm != 0) {
                    perr(p, opline, "st.param to call slot needs [slot] (no offset)");
                    return -1;
                }
                if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
                    return -1;
                in.nsrc = 2;
                if (expect_punct(p, ';', "after st.param") != 0)
                    return -1;
                return block_add_instr(fn, bi, &in, p, opline);
            }
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
                return -1;
            in.nsrc = 2;
            if (expect_punct(p, ';', "after st.param") != 0)
                return -1;
            return block_add_instr(fn, bi, &in, p, opline);
        }
        if (strcmp(parts[1], "global") != 0 &&
            strcmp(parts[1], "shared") != 0 &&
            strcmp(parts[1], "shared::cta") != 0 &&
            strcmp(parts[1], "shared::cluster") != 0 &&
            strcmp(parts[1], "local") != 0) {
            perr(p, opline, "st only supports .global/.shared/.local spaces");
            return -1;
        }
        in.space = (strcmp(parts[1], "shared") == 0 ||
                    strcmp(parts[1], "shared::cta") == 0 ||
                    strcmp(parts[1], "shared::cluster") == 0)
                       ? PSP_SHARED
                   : strcmp(parts[1], "local") == 0 ? PSP_LOCAL
                                                    : PSP_GLOBAL;
        if (map_ty(parts[2], &in.ty) != 0 || in.ty == PTY_PRED ||
            in.ty == PTY_F16 || in.ty == PTY_BF16 ||
            in.ty == PTY_F16X2 || in.ty == PTY_BF16X2) {
            // No .f16 memory access in PTX (halves move as .u16 bits).
            perr(p, opline, "unsupported st type");
            return -1;
        }
        if (parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (in.src[0].kind != OPD_ADDR) {
            perr(p, opline, "st address must use []");
            return -1;
        }
        // Direct shared-variable use resolves at emission (see ld).
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        st_f32_bits_fixup(fn, &in);
        if (has_cache_hint) {
            if (expect_comma(p) != 0 ||
                parse_operand(p, &in.src[2]) != 0)
                return -1;
            in.nsrc = 3;
        }
        if (expect_punct(p, ';', "after st") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    {
        char m[640];
        snprintf(m, sizeof(m), "unsupported instruction '%s'", opname);
        perr(p, opline, m);
        return -1;
    }
}

size_t new_block(ptx_func_t *fn, parser_t *p, const char *label) {
    if (fn->nblocks + 1 > fn->blocks_cap) {
        size_t ncap = fn->blocks_cap ? fn->blocks_cap * 2 : 4;
        ptx_block_t *nb = malloc(sizeof(*nb) * ncap);
        if (!nb) {
            perr(p, 0, "out of memory");
            return (size_t)-1;
        }
        memcpy(nb, fn->blocks, sizeof(*nb) * fn->nblocks);
        free(fn->blocks);
        fn->blocks = nb;
        fn->blocks_cap = ncap;
    }
    ptx_block_t *b = &fn->blocks[fn->nblocks];
    memset(b, 0, sizeof(*b));
    if (label)
        if (copy_name(b->label, sizeof(b->label), label ? label : "", p, 0) !=
            0)
        return (size_t)-1;
    return fn->nblocks++;
}

// Parses a module-scope ".shared [.align N] .type name[dim];"
// declaration (".shared" already consumed). Dynamic (extern/unsized)
// shared stays a clean error.
