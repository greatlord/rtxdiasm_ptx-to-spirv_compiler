#include "instructions/logic_shift/logic_shift.h"

void emit_logic_bitwise(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2;
    uint32_t id;
    uint32_t lty;
    uint32_t lop;
    if (in->ty == PTY_PRED) {
        lty = c->t_bool;
        lop = in->op == POP_AND ? S_LogicalAnd
            : in->op == POP_OR ? S_LogicalOr
                               : S_LogicalNotEqual;
    } else {
        lty = in->ty == PTY_U64 ? c->t_u64 : c->t_u32;
        lop = in->op == POP_AND ? S_And : in->op == POP_OR ? S_Or
                                                           : S_Xor;
    }
    if (check_dst(c, in->dst, in->ty, line) != 0)
        return;
    a = resolve(c, b, &in->src[0], in->ty, line);
    c2 = resolve(c, b, &in->src[1], in->ty, line);
    if (c->failed)
        return;
    id = nid(c);
    emit_arith(c, b, lop, lty, id, a.id, c2.id);
    def_val(c, in->dst, in->ty, id);
}

void emit_not_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a;
    uint32_t id;
    if (in->ty == PTY_PRED) {
        if (check_dst(c, in->dst, PTY_PRED, line) != 0)
            return;
        a = resolve(c, b, &in->src[0], PTY_PRED, line);
        if (c->failed)
            return;
        id = nid(c);
        put(c, b, (4u << 16) | S_LogicalNot);
        put(c, b, c->t_bool);
        put(c, b, id);
        put(c, b, a.id);
        def_val(c, in->dst, PTY_PRED, id);
        return;
    }
    ptx_ty_t nt = eff_ty(c, in->ty);
    uint32_t nty = nt == PTY_U64 ? c->t_u64 : c->t_u32;
    if (check_dst(c, in->dst, in->ty, line) != 0)
        return;
    a = resolve(c, b, &in->src[0], in->ty, line);
    if (c->failed)
        return;
    id = nid(c);
    put(c, b, (4u << 16) | S_Not);
    put(c, b, nty);
    put(c, b, id);
    put(c, b, a.id);
    def_val(c, in->dst, in->ty, id);
}

void emit_shf_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2, d;
    uint32_t id;
    if (check_dst(c, in->dst, PTY_U32, line) != 0)
        return;
    a = resolve(c, b, &in->src[0], PTY_U32, line);
    c2 = resolve(c, b, &in->src[1], PTY_U32, line);
    d = resolve(c, b, &in->src[2], PTY_U32, line);
    if (c->failed)
        return;
    uint32_t s = nid(c);
    emit_arith(c, b, S_And, c->t_u32, s, d.id, const_u32(c, 31));
    uint32_t ns = nid(c);
    emit_arith(c, b, S_ISub, c->t_u32, ns, const_u32(c, 32), s);
    uint32_t ns2 = nid(c);
    emit_arith(c, b, S_And, c->t_u32, ns2, ns, const_u32(c, 31));
    uint32_t is0 = do_cmp(c, b, PCM_EQ, PTY_U32, 0, s, const_u32(c, 0), line);
    if (c->failed)
        return;
    uint32_t funnel;
    if (in->shf_dir) {
        uint32_t hi = nid(c);
        emit_arith(c, b, S_Shl, c->t_u32, hi, c2.id, s);
        uint32_t lo0 = nid(c);
        emit_arith(c, b, S_ShrL, c->t_u32, lo0, a.id, ns2);
        uint32_t lo = sel_u32(c, b, is0, const_u32(c, 0), lo0);
        funnel = nid(c);
        emit_arith(c, b, S_Or, c->t_u32, funnel, hi, lo);
    } else {
        uint32_t lo = nid(c);
        emit_arith(c, b, S_ShrL, c->t_u32, lo, a.id, s);
        uint32_t hi0 = nid(c);
        emit_arith(c, b, S_Shl, c->t_u32, hi0, c2.id, ns2);
        uint32_t hi = sel_u32(c, b, is0, const_u32(c, 0), hi0);
        funnel = nid(c);
        emit_arith(c, b, S_Or, c->t_u32, funnel, lo, hi);
    }
    if (!in->shf_clamp) {
        def_val(c, in->dst, PTY_U32, funnel);
        return;
    }
    uint32_t over = do_cmp(c, b, PCM_GE, PTY_U32, 0, d.id, const_u32(c, 32), line);
    if (c->failed)
        return;
    id = sel_u32(c, b, over, in->shf_dir ? a.id : c2.id, funnel);
    def_val(c, in->dst, PTY_U32, id);
}

void emit_cnot_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a;
    uint32_t id;
    if (check_dst(c, in->dst, in->ty, line) != 0)
        return;
    a = resolve(c, b, &in->src[0], in->ty, line);
    if (c->failed)
        return;
    uint32_t z = in->ty == PTY_U64 ? const_u64(c, 0)
                 : in->ty == PTY_U16 ? const_u32(c, 0)
                                     : const_u32(c, 0);
    uint32_t eq = do_cmp(c, b, PCM_EQ, in->ty, 0, a.id, z, line);
    if (c->failed)
        return;
    uint32_t ty = in->ty == PTY_U64 ? c->t_u64
                  : in->ty == PTY_U16 ? c->t_u32
                                      : c->t_u32;
    uint32_t one = in->ty == PTY_U64 ? const_u64(c, 1)
                                     : const_u32(c, 1);
    id = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, ty);
    put(c, b, id);
    put(c, b, eq);
    put(c, b, one);
    put(c, b, z);
    def_val(c, in->dst, in->ty, id);
}

void emit_lop3_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2, d;
    if (check_dst(c, in->dst, PTY_U32, line) != 0)
        return;
    a = resolve(c, b, &in->src[0], PTY_U32, line);
    c2 = resolve(c, b, &in->src[1], PTY_U32, line);
    d = resolve(c, b, &in->src[2], PTY_U32, line);
    if (c->failed)
        return;
    uint32_t na = nid(c), nb = nid(c), nc = nid(c);
    put(c, b, (4u << 16) | S_Not);
    put(c, b, c->t_u32);
    put(c, b, na);
    put(c, b, a.id);
    put(c, b, (4u << 16) | S_Not);
    put(c, b, c->t_u32);
    put(c, b, nb);
    put(c, b, c2.id);
    put(c, b, (4u << 16) | S_Not);
    put(c, b, c->t_u32);
    put(c, b, nc);
    put(c, b, d.id);
    uint32_t out = const_u32(c, 0);
    for (int row = 0; row < 8; row++) {
        if (!(in->arg & (1 << row)))
            continue;
        uint32_t ta = (row & 4) ? a.id : na;
        uint32_t tb = (row & 2) ? c2.id : nb;
        uint32_t tc = (row & 1) ? d.id : nc;
        uint32_t t1 = nid(c), t2 = nid(c), t3 = nid(c);
        emit_arith(c, b, S_And, c->t_u32, t1, ta, tb);
        emit_arith(c, b, S_And, c->t_u32, t2, t1, tc);
        emit_arith(c, b, S_Or, c->t_u32, t3, out, t2);
        out = t3;
    }
    if (in->lop3_boolop) {
        uint32_t nz = do_cmp(c, b, PCM_NE, PTY_U32, 0, out,
                             const_u32(c, 0), line);
        val_t qv = resolve(c, b, &in->src[4], PTY_PRED, line);
        if (c->failed)
            return;
        uint32_t pr = nid(c);
        put(c, b, (5u << 16) |
                      (in->lop3_boolop == 1 ? S_LogicalOr
                                            : S_LogicalAnd));
        put(c, b, c->t_bool);
        put(c, b, pr);
        put(c, b, nz);
        put(c, b, qv.id);
        if (in->dst2[0])
            def_val(c, in->dst2, PTY_PRED, pr);
    }
    def_val(c, in->dst, PTY_U32, out);
}

void emit_shift_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2;
    uint32_t id;
    int is_shr = in->op == POP_SHR;
    uint32_t st = in->ty == PTY_U64 ? c->t_u64 : c->t_u32;
    uint32_t shop = !is_shr ? S_Shl : in->sign ? S_ShrA : S_ShrL;
    if (check_dst(c, in->dst, in->ty, line) != 0)
        return;
    if (in->ty == PTY_U64 && in->src[0].kind == OPD_REG) {
        reg_t *sr = find_reg(c, in->src[0].reg);
        if (sr && sr->defined && sr->is_ptr) {
            a.id = ptr_token_u64(c, b, sr, line);
            a.ty = PTY_U64;
        } else {
            a = resolve(c, b, &in->src[0], in->ty, line);
        }
    } else {
        a = resolve(c, b, &in->src[0], in->ty, line);
    }
    c2 = resolve(c, b, &in->src[1], PTY_U32, line);
    if (c->failed)
        return;
    if (in->ty == PTY_U8 || in->ty == PTY_U16) {
        uint32_t amt = nid(c);
        emit_arith(c, b, S_And, c->t_u32, amt, c2.id, const_u32(c, 31));
        uint32_t base = a.id;
        if (is_shr && in->sign) {
            uint32_t shift = (uint32_t)(in->ty == PTY_U8 ? 24 : 16);
            uint32_t l = nid(c);
            emit_arith(c, b, S_Shl, c->t_u32, l, a.id, const_u32(c, shift));
            base = nid(c);
            emit_arith(c, b, S_ShrA, c->t_u32, base, l, const_u32(c, shift));
        }
        id = nid(c);
        emit_arith(c, b, shop, c->t_u32, id, base, amt);
        uint32_t mask = const_u32(c, in->ty == PTY_U8 ? 0xffu : 0xffffu);
        uint32_t m = nid(c);
        emit_arith(c, b, S_And, c->t_u32, m, id, mask);
        def_val(c, in->dst, in->ty, m);
        return;
    }
    uint32_t width = in->ty == PTY_U64 ? 64 : 32;
    uint32_t inrange = nid(c);
    put(c, b, (5u << 16) | S_ULess);
    put(c, b, c->t_bool);
    put(c, b, inrange);
    put(c, b, c2.id);
    put(c, b, const_u32(c, width));
    uint32_t amt = nid(c);
    emit_arith(c, b, S_And, c->t_u32, amt, c2.id, const_u32(c, width - 1));
    uint32_t fill = 0;
    if (is_shr && in->sign) {
        fill = nid(c);
        emit_arith(c, b, S_ShrA, st, fill, a.id,
                   in->ty == PTY_U64 ? const_u64(c, 63) : const_u32(c, 31));
    } else {
        fill = in->ty == PTY_U64 ? const_u64(c, 0) : const_u32(c, 0);
    }
    if (in->ty == PTY_U64) {
        uint32_t amt64 = nid(c);
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, c->t_u64);
        put(c, b, amt64);
        put(c, b, amt);
        uint32_t shifted = nid(c);
        emit_arith(c, b, shop, st, shifted, a.id, amt64);
        id = sel_u64(c, b, inrange, shifted, fill);
    } else {
        uint32_t shifted = nid(c);
        emit_arith(c, b, shop, st, shifted, a.id, amt);
        id = sel_u32(c, b, inrange, shifted, fill);
    }
    def_val(c, in->dst, in->ty, id);
}

