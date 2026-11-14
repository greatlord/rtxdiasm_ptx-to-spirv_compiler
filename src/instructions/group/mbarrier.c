// elect.sync and mbarrier lowering.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "spirv_internal.h"

// mbarrier state packing (64 bits):
//   pending  [0:20)   expected [20:40)   tx [40:60)
//   parity   [60]     layout   [61]      valid [62]
#define MB_PEND_SHIFT 0
#define MB_EXP_SHIFT 20
#define MB_TX_SHIFT 40
#define MB_PAR_SHIFT 60
#define MB_LAY_SHIFT 61
#define MB_VAL_SHIFT 62
#define MB_MASK 0xfffffull

static uint32_t mb_ptr(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in,
                       int line) {
    int binding = -1, sh = -1, gl = -1;
    uint32_t off = 0;
    if (resolve_addr(c, b, &in->src[0], &binding, &off, &sh, &gl, line) != 0)
        return 0;
    if (sh < 0) {
        failf(c, line, "mbarrier address must be shared memory");
        return 0;
    }
    return sh_addr_ptr64(c, b, sh, off);
}

static uint32_t mb_field(ctx_t *c, spv_blob_t *b, uint32_t old, int shift) {
    uint32_t v = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, v, old,
               const_u64(c, (uint64_t)shift));
    uint32_t m = nid(c);
    emit_arith(c, b, S_And, c->t_u64, m, v,
               const_u64(c, MB_MASK));
    return m;
}

// Single-bit field (phase parity, layout). The bits above the 20-bit
// count fields (layout, valid) must not leak into the result.
static uint32_t mb_bit(ctx_t *c, spv_blob_t *b, uint32_t old, int shift) {
    uint32_t v = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, v, old,
               const_u64(c, (uint64_t)shift));
    uint32_t m = nid(c);
    emit_arith(c, b, S_And, c->t_u64, m, v, const_u64(c, 1));
    return m;
}

static uint32_t mb_sub_sat(ctx_t *c, spv_blob_t *b, uint32_t a, uint32_t bb) {
    uint32_t under = do_cmp(c, b, PCM_LT, PTY_U64, 0, a, bb, 0);
    uint32_t d = nid(c);
    emit_arith(c, b, S_ISub, c->t_u64, d, a, bb);
    return sel_u64(c, b, under, const_u64(c, 0), d);
}

static uint32_t mb_u64(ctx_t *c, spv_blob_t *b, uint32_t u32) {
    uint32_t v = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u64);
    put(c, b, v);
    put(c, b, u32);
    return v;
}

// Compute the next state. `count`/`tx` are u32 ids; `state_out` receives
// the arrival state for arrive operations.
static uint32_t mb_next(ctx_t *c, spv_blob_t *b, uint32_t old, int op,
                        uint32_t count, uint32_t tx, int nocomplete,
                        int expect, int drop, uint32_t *state_out) {
    uint32_t pend = mb_field(c, b, old, MB_PEND_SHIFT);
    uint32_t exp = mb_field(c, b, old, MB_EXP_SHIFT);
    uint32_t txc = mb_field(c, b, old, MB_TX_SHIFT);
    uint32_t par = mb_bit(c, b, old, MB_PAR_SHIFT);
    uint32_t cnt = mb_u64(c, b, count);
    uint32_t ntx = tx ? mb_u64(c, b, tx) : const_u64(c, 0);
    uint32_t new_pend = pend, new_exp = exp, new_tx = txc;
    uint32_t complete = 0;
    if (op == MBAR_ARRIVE || op == MBAR_ARRIVE_DROP ||
        op == MBAR_CP_ARRIVE) {
        if (state_out) {
            uint32_t s1 = nid(c);
            emit_arith(c, b, S_Shl, c->t_u64, s1, pend,
                       const_u64(c, 1));
            *state_out = nid(c);
            emit_arith(c, b, S_Or, c->t_u64, *state_out, s1, par);
        }
        new_pend = mb_sub_sat(c, b, pend, cnt);
        if (drop)
            new_exp = mb_sub_sat(c, b, exp, cnt);
        if (expect) {
            new_tx = nid(c);
            emit_arith(c, b, S_IAdd, c->t_u64, new_tx, txc, ntx);
        }
        if (!nocomplete) {
            uint32_t pz = do_cmp(c, b, PCM_EQ, PTY_U64, 0, new_pend,
                                 const_u64(c, 0), 0);
            uint32_t tz = do_cmp(c, b, PCM_EQ, PTY_U64, 0, new_tx,
                                 const_u64(c, 0), 0);
            complete = nid(c);
            put(c, b, (5u << 16) | S_LogicalAnd);
            put(c, b, c->t_bool);
            put(c, b, complete);
            put(c, b, pz);
            put(c, b, tz);
        }
    } else if (op == MBAR_EXPECT_TX) {
        new_tx = nid(c);
        emit_arith(c, b, S_IAdd, c->t_u64, new_tx, txc, cnt);
    } else if (op == MBAR_COMPLETE_TX) {
        new_tx = mb_sub_sat(c, b, txc, cnt);
        uint32_t pz = do_cmp(c, b, PCM_EQ, PTY_U64, 0, pend,
                             const_u64(c, 0), 0);
        uint32_t tz = do_cmp(c, b, PCM_EQ, PTY_U64, 0, new_tx,
                             const_u64(c, 0), 0);
        complete = nid(c);
        put(c, b, (5u << 16) | S_LogicalAnd);
        put(c, b, c->t_bool);
        put(c, b, complete);
        put(c, b, pz);
        put(c, b, tz);
    }
    uint32_t parf = par;
    if (complete) {
        uint32_t ox = nid(c);
        emit_arith(c, b, S_Xor, c->t_u64, ox, par, const_u64(c, 1));
        parf = sel_u64(c, b, complete, ox, par);
        new_pend = sel_u64(c, b, complete, new_exp, new_pend);
    }
    uint32_t keep = nid(c);
    emit_arith(c, b, S_And, c->t_u64, keep, old,
               const_u64(c, (3ull << MB_LAY_SHIFT)));
    uint32_t p = nid(c);
    emit_arith(c, b, S_Shl, c->t_u64, p, new_pend,
               const_u64(c, MB_PEND_SHIFT));
    uint32_t e = nid(c);
    emit_arith(c, b, S_Shl, c->t_u64, e, new_exp,
               const_u64(c, MB_EXP_SHIFT));
    uint32_t t = nid(c);
    emit_arith(c, b, S_Shl, c->t_u64, t, new_tx,
               const_u64(c, MB_TX_SHIFT));
    uint32_t pf = nid(c);
    emit_arith(c, b, S_Shl, c->t_u64, pf, parf,
               const_u64(c, MB_PAR_SHIFT));
    uint32_t res = nid(c);
    emit_arith(c, b, S_Or, c->t_u64, res, keep, p);
    uint32_t r2 = nid(c);
    emit_arith(c, b, S_Or, c->t_u64, r2, res, e);
    uint32_t r3 = nid(c);
    emit_arith(c, b, S_Or, c->t_u64, r3, r2, t);
    uint32_t r4 = nid(c);
    emit_arith(c, b, S_Or, c->t_u64, r4, r3, pf);
    return r4;
}

static uint32_t mb_update(ctx_t *c, spv_blob_t *b, uint32_t ptr, int op,
                          uint32_t count, uint32_t tx, int nocomplete,
                          int expect, int drop, uint32_t *state_out) {
    uint32_t hdr = nid(c), merge = nid(c), cont = nid(c);
    uint32_t scope = const_u32(c, 2);
    uint32_t sem_eq = const_u32(c, 0x108u);
    uint32_t sem_ne = const_u32(c, 0x102u);
    emit_branch(c, b, hdr);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, hdr);
    uint32_t old = nid(c);
    emit_load(c, b, c->t_u64, old, ptr);
    uint32_t nw = mb_next(c, b, old, op, count, tx, nocomplete, expect,
                          drop, state_out);
    uint32_t prev = nid(c);
    put(c, b, (9u << 16) | S_AtomicCAS);
    put(c, b, c->t_u64);
    put(c, b, prev);
    put(c, b, ptr);
    put(c, b, scope);
    put(c, b, sem_eq);
    put(c, b, sem_ne);
    put(c, b, nw);
    put(c, b, old);
    uint32_t ok = nid(c);
    put(c, b, (5u << 16) | S_IEqual);
    put(c, b, c->t_bool);
    put(c, b, ok);
    put(c, b, prev);
    put(c, b, old);
    put(c, b, (4u << 16) | S_LoopMerge);
    put(c, b, merge);
    put(c, b, cont);
    put(c, b, 0);
    put(c, b, (4u << 16) | S_BranchCond);
    put(c, b, ok);
    put(c, b, merge);
    put(c, b, cont);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, cont);
    emit_branch(c, b, hdr);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, merge);
    return state_out ? *state_out : 0;
}

// Label id of the last block started in the function body.
static uint32_t last_block_label(const spv_blob_t *b) {
    uint32_t last = 0;
    for (size_t i = 0; i < b->count;) {
        uint32_t wc = b->words[i] >> 16;
        if (wc == 0)
            break;
        if ((b->words[i] & 0xffffu) == S_Label && i + 1 < b->count)
            last = b->words[i + 1];
        i += wc;
    }
    return last;
}

// Emit `fn` only when the instruction's guard predicate is true.
// The guard is a real SPIR-V selection so side effects are skipped;
// result registers are merged at the join with an OpPhi (an undef
// incoming value when this is their first definition).
static int pred_dst_phi(ctx_t *c, spv_blob_t *b, const char *name,
                        uint32_t *old_out, uint32_t *ty_out, int line) {
    if (!name[0] || strncmp(name, "__sink_", 7) == 0) {
        *old_out = 0;
        *ty_out = 0;
        return 0;
    }
    reg_t *r = find_reg(c, name);
    if (!r) {
        failf(c, line, "assignment to undeclared register");
        return -1;
    }
    *ty_out = ty_of(c, r->decl_ty);
    if (r->defined) {
        *old_out = r->cur;
    } else {
        uint32_t u = nid(c);
        put(c, b, (3u << 16) | 1); // OpUndef
        put(c, b, *ty_out);
        put(c, b, u);
        *old_out = u;
    }
    return 0;
}

static void pred_dst_merge(ctx_t *c, spv_blob_t *b, const char *name,
                           uint32_t old_id, uint32_t ty, uint32_t thenl,
                           uint32_t elsel) {
    if (!name[0] || strncmp(name, "__sink_", 7) == 0)
        return;
    reg_t *r = find_reg(c, name);
    if (!r)
        return;
    uint32_t nv = r->defined ? r->cur : old_id;
    uint32_t phi = nid(c);
    put(c, b, (7u << 16) | S_Phi);
    put(c, b, ty);
    put(c, b, phi);
    put(c, b, nv);
    put(c, b, thenl);
    put(c, b, old_id);
    put(c, b, elsel);
    r->defined = 1;
    r->cur = phi;
    r->is_ptr = 0;
}

int emit_predicated_instr(ctx_t *c, spv_blob_t *b,
                          const ptx_instr_t *in,
                          int (*fn)(ctx_t *, spv_blob_t *,
                                    const ptx_instr_t *)) {
    reg_t *pr = find_reg(c, in->pred);
    if (!pr || pr->decl_ty != PTY_PRED || !pr->defined) {
        failf(c, in->line,
              "predicated instruction needs a defined .pred register");
        return -1;
    }
    uint32_t old0 = 0, ty0 = 0, old1 = 0, ty1 = 0;
    if (pred_dst_phi(c, b, in->dst, &old0, &ty0, in->line) != 0 ||
        pred_dst_phi(c, b, in->dst2, &old1, &ty1, in->line) != 0)
        return -1;
    uint32_t cond = pr->cur;
    if (in->pred_neg) {
        uint32_t n = nid(c);
        put(c, b, (4u << 16) | S_LogicalNot);
        put(c, b, c->t_bool);
        put(c, b, n);
        put(c, b, pr->cur);
        cond = n;
    }
    uint32_t thenl = nid(c), elsel = nid(c), mergel = nid(c);
    put(c, b, (3u << 16) | S_SelectionMerge);
    put(c, b, mergel);
    put(c, b, 0);
    put(c, b, (4u << 16) | S_BranchCond);
    put(c, b, cond);
    put(c, b, thenl);
    put(c, b, elsel);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, thenl);
    if (fn(c, b, in) != 0)
        return -1;
    // The operation may itself leave several blocks behind (mbarrier's
    // CAS loop); the phi edge comes from whichever block it ended in.
    uint32_t thenj = last_block_label(b);
    emit_branch(c, b, mergel);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, elsel);
    emit_branch(c, b, mergel);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, mergel);
    pred_dst_merge(c, b, in->dst, old0, ty0, thenj, elsel);
    pred_dst_merge(c, b, in->dst2, old1, ty1, thenj, elsel);
    if (c->failed)
        return -1;
    return 0;
}

static int emit_elect_op_body(ctx_t *c, spv_blob_t *b,
                              const ptx_instr_t *in) {
    int line = in->line;
    val_t mask = resolve(c, b, &in->src[0], PTY_U32, line);
    if (c->failed)
        return -1;
    uint32_t lane = lane_id(c, b, line);
    uint32_t sh = nid(c), bit = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, sh, mask.id, lane);
    emit_arith(c, b, S_And, c->t_u32, bit, sh, const_u32(c, 1));
    uint32_t mine = do_cmp(c, b, PCM_NE, PTY_U32, 0, bit,
                           const_u32(c, 0), line);
    uint32_t scope = const_u32(c, 3);
    uint32_t bal = nid(c);
    put(c, b, (5u << 16) | S_GroupBallot);
    put(c, b, c->t_uvec4);
    put(c, b, bal);
    put(c, b, scope);
    put(c, b, mine);
    uint32_t w = nid(c);
    put(c, b, (5u << 16) | S_CompositeExtract);
    put(c, b, c->t_u32);
    put(c, b, w);
    put(c, b, bal);
    put(c, b, 0u);
    uint32_t neg = nid(c);
    emit_arith(c, b, S_ISub, c->t_u32, neg, const_u32(c, 0), w);
    uint32_t low = nid(c);
    emit_arith(c, b, S_And, c->t_u32, low, w, neg);
    uint32_t lz = clz32(c, b, low, line);
    uint32_t leader = nid(c);
    emit_arith(c, b, S_ISub, c->t_u32, leader, const_u32(c, 31), lz);
    uint32_t islead = do_cmp(c, b, PCM_EQ, PTY_U32, 0, lane, leader, line);
    uint32_t pred = nid(c);
    put(c, b, (5u << 16) | S_LogicalAnd);
    put(c, b, c->t_bool);
    put(c, b, pred);
    put(c, b, mine);
    put(c, b, islead);
    if (in->dst[0] && strncmp(in->dst, "__sink_", 7) != 0)
        def_val(c, in->dst, PTY_U32, leader);
    if (in->dst2[0] && strncmp(in->dst2, "__sink_", 7) != 0)
        def_val(c, in->dst2, PTY_PRED, pred);
    return c->failed ? -1 : 0;
}

static int emit_mbarrier_op_body(ctx_t *c, spv_blob_t *b,
                                 const ptx_instr_t *in) {
    int line = in->line;
    uint32_t ptr = 0;
    if (in->mb_op != MBAR_PENDING) {
        ptr = mb_ptr(c, b, in, line);
        if (!ptr && c->failed)
            return -1;
    }
    if (in->mb_op == MBAR_INIT) {
        val_t cnt = resolve(c, b, &in->src[1], PTY_U32, line);
        if (c->failed)
            return -1;
        uint32_t c64 = mb_u64(c, b, cnt.id);
        uint32_t pen = nid(c);
        emit_arith(c, b, S_Shl, c->t_u64, pen, c64,
                   const_u64(c, MB_PEND_SHIFT));
        uint32_t exp = nid(c);
        emit_arith(c, b, S_Shl, c->t_u64, exp, c64,
                   const_u64(c, MB_EXP_SHIFT));
        uint32_t bits = nid(c);
        emit_arith(c, b, S_Or, c->t_u64, bits, pen, exp);
        uint32_t b2 = nid(c);
        emit_arith(c, b, S_Or, c->t_u64, b2, bits,
                   const_u64(c, 1ull << MB_VAL_SHIFT));
        bits = b2;
        if (in->mb_layout) {
            uint32_t b3 = nid(c);
            emit_arith(c, b, S_Or, c->t_u64, b3, bits,
                       const_u64(c, 1ull << MB_LAY_SHIFT));
            bits = b3;
        }
        emit_store(c, b, ptr, bits);
        return 0;
    }
    if (in->mb_op == MBAR_INVAL) {
        emit_store(c, b, ptr, const_u64(c, 0));
        return 0;
    }
    if (in->mb_op == MBAR_CP_ARRIVE) {
        if (!in->mb_nocomplete)
            return 0; // cp.async already completed synchronously
        mb_update(c, b, ptr, MBAR_ARRIVE, const_u32(c, 1), 0, 0, 0, 0,
                  NULL);
        return c->failed ? -1 : 0;
    }
    if (in->mb_op == MBAR_TEST_WAIT || in->mb_op == MBAR_TRY_WAIT) {
        uint32_t old = nid(c);
        emit_load(c, b, c->t_u64, old, ptr);
        uint32_t par = mb_bit(c, b, old, MB_PAR_SHIFT);
        uint32_t arg;
        if (in->mb_parity) {
            val_t pv = resolve(c, b, &in->src[1], PTY_U32, line);
            arg = mb_u64(c, b, pv.id);
        } else {
            val_t sv = resolve(c, b, &in->src[1], PTY_U64, line);
            arg = sv.id;
        }
        uint32_t a1 = nid(c);
        emit_arith(c, b, S_And, c->t_u64, a1, arg, const_u64(c, 1));
        uint32_t wait = do_cmp(c, b, PCM_NE, PTY_U64, 0, par, a1, line);
        if (in->dst[0] && strncmp(in->dst, "__sink_", 7) != 0)
            def_val(c, in->dst, PTY_PRED, wait);
        return c->failed ? -1 : 0;
    }
    if (in->mb_op == MBAR_PENDING) {
        val_t sv = resolve(c, b, &in->src[0], PTY_U64, line);
        uint32_t sh = nid(c);
        emit_arith(c, b, S_ShrL, c->t_u64, sh, sv.id, const_u64(c, 1));
        uint32_t m = nid(c);
        emit_arith(c, b, S_And, c->t_u64, m, sh, const_u64(c, MB_MASK));
        uint32_t u = nid(c);
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, c->t_u32);
        put(c, b, u);
        put(c, b, m);
        def_val(c, in->dst, PTY_U32, u);
        return c->failed ? -1 : 0;
    }
    if (in->mb_op == MBAR_CHECK_LAYOUT) {
        uint32_t old = nid(c);
        emit_load(c, b, c->t_u64, old, ptr);
        uint32_t lay = mb_bit(c, b, old, MB_LAY_SHIFT);
        uint32_t want = const_u32(c, in->mb_layout ? 1u : 0u);
        uint32_t w64 = mb_u64(c, b, want);
        uint32_t eq = do_cmp(c, b, PCM_EQ, PTY_U64, 0, lay, w64, line);
        if (in->dst[0] && strncmp(in->dst, "__sink_", 7) != 0)
            def_val(c, in->dst, PTY_PRED, eq);
        return c->failed ? -1 : 0;
    }
    // Arrive / expect_tx / complete_tx.
    uint32_t count = const_u32(c, 1);
    uint32_t tx = 0;
    if (in->mb_op == MBAR_ARRIVE || in->mb_op == MBAR_ARRIVE_DROP) {
        if (in->nsrc > 1) {
            val_t v = resolve(c, b, &in->src[1], PTY_U32, line);
            if (c->failed)
                return -1;
            if (in->mb_expect)
                tx = v.id;
            else
                count = v.id;
        }
    } else if (in->mb_op == MBAR_EXPECT_TX ||
               in->mb_op == MBAR_COMPLETE_TX) {
        val_t v = resolve(c, b, &in->src[1], PTY_U32, line);
        if (c->failed)
            return -1;
        count = v.id;
    }
    uint32_t st = 0;
    mb_update(c, b, ptr, in->mb_op, count, tx, in->mb_nocomplete,
              in->mb_expect, in->mb_op == MBAR_ARRIVE_DROP, &st);
    if ((in->mb_op == MBAR_ARRIVE || in->mb_op == MBAR_ARRIVE_DROP) &&
        in->dst[0] && strncmp(in->dst, "__sink_", 7) != 0)
        def_val(c, in->dst, PTY_U64, st);
    return c->failed ? -1 : 0;
}

// Complete `bytes` of transaction count on the mbarrier at `ptr`.
// Used by red.async's mbarrier::complete_tx::bytes completion, which
// is immediate here because the reduction itself is synchronous.
void emit_mbarrier_complete_tx(ctx_t *c, spv_blob_t *b, uint32_t ptr,
                               uint32_t bytes) {
    mb_update(c, b, ptr, MBAR_COMPLETE_TX, const_u32(c, bytes), 0, 0, 0, 0,
              NULL);
}

int emit_elect_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    if (!in->pred[0])
        return emit_elect_op_body(c, b, in);
    return emit_predicated_instr(c, b, in, emit_elect_op_body);
}

int emit_mbarrier_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    c->uses_barrier = 1;
    if (!in->pred[0])
        return emit_mbarrier_op_body(c, b, in);
    return emit_predicated_instr(c, b, in, emit_mbarrier_op_body);
}
