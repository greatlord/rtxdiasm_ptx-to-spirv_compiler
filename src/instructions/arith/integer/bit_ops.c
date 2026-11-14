#include "instructions/arith/integer/integer.h"

void emit_fns_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2, d;
    if (check_dst(c, in->dst, PTY_U32, line) != 0)
        return;
    a = resolve(c, b, &in->src[0], PTY_U32, line); // mask
    c2 = resolve(c, b, &in->src[1], PTY_U32, line); // base
    d = resolve(c, b, &in->src[2], PTY_U32, line); // offset
    if (c->failed)
        return;
    uint32_t off = d.id, base = c2.id, mask = a.id;
    uint32_t sgn = nid(c);
    emit_arith(c, b, S_ShrA, c->t_u32, sgn, off, const_u32(c, 31));
    uint32_t ab = nid(c);
    emit_arith(c, b, S_Xor, c->t_u32, ab, off, sgn);
    uint32_t absv = nid(c);
    emit_arith(c, b, S_ISub, c->t_u32, absv, ab, sgn);
    uint32_t count = nid(c);
    emit_arith(c, b, S_ISub, c->t_u32, count, absv, const_u32(c, 1));
    uint32_t ineg = nid(c), inc = nid(c);
    put(c, b, (4u << 16) | S_SNegate);
    put(c, b, c->t_u32);
    put(c, b, ineg);
    put(c, b, const_u32(c, 1));
    uint32_t is_pos = do_cmp(c, b, PCM_GT, PTY_U32, 1, off,
                             const_u32(c, 0), line);
    if (c->failed)
        return;
    inc = sel_u32(c, b, is_pos, const_u32(c, 1), ineg);
    uint32_t bvalid = do_cmp(c, b, PCM_LT, PTY_U32, 0, base,
                             const_u32(c, 32), line);
    uint32_t bsh = nid(c), bbit = nid(c);
    emit_arith(c, b, S_And, c->t_u32, bsh, base, const_u32(c, 31));
    uint32_t bm = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, bm, mask, bsh);
    emit_arith(c, b, S_And, c->t_u32, bbit, bm, const_u32(c, 1));
    uint32_t bhit = do_cmp(c, b, PCM_NE, PTY_U32, 0, bbit,
                           const_u32(c, 0), line);
    uint32_t zero_off = do_cmp(c, b, PCM_EQ, PTY_U32, 0, off,
                               const_u32(c, 0), line);
    if (c->failed)
        return;
    uint32_t take0 = nid(c);
    put(c, b, (5u << 16) | S_LogicalAnd);
    put(c, b, c->t_bool);
    put(c, b, take0);
    put(c, b, bvalid);
    put(c, b, bhit);
    uint32_t d0 = sel_u32(c, b, take0, base, const_u32(c, 0xffffffffu));
    uint32_t found = zero_off;
    uint32_t res = sel_u32(c, b, zero_off, d0, const_u32(c, 0xffffffffu));
    for (int n = 0; n < 32; n++) {
        uint32_t cand = nid(c);
        if (n == 0) {
            cand = base;
        } else {
            uint32_t step = nid(c);
            emit_arith(c, b, S_IMul, c->t_u32, step, inc,
                       const_u32(c, (uint32_t)n));
            emit_arith(c, b, S_IAdd, c->t_u32, cand, base, step);
        }
        uint32_t valid = do_cmp(c, b, PCM_LT, PTY_U32, 0, cand,
                                const_u32(c, 32), line);
        uint32_t psh = nid(c), mv = nid(c), bit = nid(c);
        emit_arith(c, b, S_And, c->t_u32, psh, cand, const_u32(c, 31));
        emit_arith(c, b, S_ShrL, c->t_u32, mv, mask, psh);
        emit_arith(c, b, S_And, c->t_u32, bit, mv, const_u32(c, 1));
        uint32_t hit = do_cmp(c, b, PCM_NE, PTY_U32, 0, bit,
                              const_u32(c, 0), line);
        uint32_t nf = nid(c);
        put(c, b, (4u << 16) | S_LogicalNot);
        put(c, b, c->t_bool);
        put(c, b, nf);
        put(c, b, found);
        uint32_t count0 = do_cmp(c, b, PCM_EQ, PTY_U32, 0, count,
                                 const_u32(c, 0), line);
        if (c->failed)
            return;
        uint32_t h1 = nid(c), h2 = nid(c), take = nid(c);
        put(c, b, (5u << 16) | S_LogicalAnd);
        put(c, b, c->t_bool);
        put(c, b, h1);
        put(c, b, valid);
        put(c, b, hit);
        put(c, b, (5u << 16) | S_LogicalAnd);
        put(c, b, c->t_bool);
        put(c, b, h2);
        put(c, b, h1);
        put(c, b, nf);
        put(c, b, (5u << 16) | S_LogicalAnd);
        put(c, b, c->t_bool);
        put(c, b, take);
        put(c, b, h2);
        put(c, b, count0);
        res = sel_u32(c, b, take, cand, res);
        uint32_t nf2 = nid(c);
        put(c, b, (4u << 16) | S_LogicalNot);
        put(c, b, c->t_bool);
        put(c, b, nf2);
        put(c, b, take);
        uint32_t found2 = nid(c);
        put(c, b, (5u << 16) | S_LogicalOr);
        put(c, b, c->t_bool);
        put(c, b, found2);
        put(c, b, found);
        put(c, b, take);
        uint32_t count_pos = do_cmp(c, b, PCM_GT, PTY_U32, 0, count,
                                    const_u32(c, 0), line);
        uint32_t cons = nid(c), dec = nid(c), cdec = nid(c);
        emit_arith(c, b, S_ISub, c->t_u32, cdec, count,
                   const_u32(c, 1));
        put(c, b, (5u << 16) | S_LogicalAnd);
        put(c, b, c->t_bool);
        put(c, b, cons);
        put(c, b, h2);
        put(c, b, count_pos);
        put(c, b, (5u << 16) | S_LogicalAnd);
        put(c, b, c->t_bool);
        put(c, b, dec);
        put(c, b, cons);
        put(c, b, nf);
        count = sel_u32(c, b, dec, cdec, count);
        found = found2;
    }
    def_val(c, in->dst, PTY_U32, res);
}

void emit_szext_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2;
    uint32_t id;
    if (check_dst(c, in->dst, PTY_U32, line) != 0)
        return;
    a = resolve(c, b, &in->src[0], PTY_U32, line);
    c2 = resolve(c, b, &in->src[1], PTY_U32, line);
    if (c->failed)
        return;
    uint32_t b1 = nid(c), sh = nid(c), mask = nid(c);
    emit_arith(c, b, S_And, c->t_u32, b1, c2.id, const_u32(c, 31));
    emit_arith(c, b, S_Shl, c->t_u32, sh, const_u32(c, 0xffffffffu),
               b1);
    mask = sh;
    uint32_t too = const_u32(c, 0);
    if (in->sz_clamp) {
        too = do_cmp(c, b, PCM_GE, PTY_U32, 0, c2.id,
                     const_u32(c, 32), line);
        if (c->failed)
            return;
        mask = sel_u32(c, b, too, const_u32(c, 0), sh);
    }
    uint32_t nz = do_cmp(c, b, PCM_NE, PTY_U32, 0, b1,
                         const_u32(c, 0), line);
    uint32_t signbit = const_u32(c, 0);
    if (in->sign) {
        uint32_t sc = nid(c), sp = nid(c), v = nid(c), sb = nid(c);
        emit_arith(c, b, S_ISub, c->t_u32, sc, b1, const_u32(c, 1));
        emit_arith(c, b, S_And, c->t_u32, sp, sc, const_u32(c, 31));
        emit_arith(c, b, S_ShrL, c->t_u32, v, a.id, sp);
        emit_arith(c, b, S_And, c->t_u32, sb, v, const_u32(c, 1));
        uint32_t ntoo = nid(c);
        put(c, b, (4u << 16) | S_LogicalNot);
        put(c, b, c->t_bool);
        put(c, b, ntoo);
        put(c, b, too);
        uint32_t ok = nid(c);
        put(c, b, (5u << 16) | S_LogicalAnd);
        put(c, b, c->t_bool);
        put(c, b, ok);
        put(c, b, nz);
        put(c, b, ntoo);
        signbit = sel_u32(c, b, ok, sb, const_u32(c, 0));
    }
    uint32_t nm = nid(c), low = nid(c), sb2 = nid(c), repl = nid(c);
    put(c, b, (4u << 16) | S_Not);
    put(c, b, c->t_u32);
    put(c, b, nm);
    put(c, b, mask);
    emit_arith(c, b, S_And, c->t_u32, low, a.id, nm);
    sb2 = do_cmp(c, b, PCM_NE, PTY_U32, 0, signbit,
                 const_u32(c, 0), line);
    repl = sel_u32(c, b, sb2, mask, const_u32(c, 0));
    id = nid(c);
    emit_arith(c, b, S_Or, c->t_u32, id, low, repl);
    def_val(c, in->dst, PTY_U32, id);
}

void emit_popc_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a;
    uint32_t id;
    if (check_dst(c, in->dst, PTY_U32, line) != 0)
        return;
    a = resolve(c, b, &in->src[0], in->ty, line);
    if (c->failed)
        return;
    if (in->ty == PTY_U64) {
        uint32_t lo = nid(c), hi = nid(c);
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, c->t_u32);
        put(c, b, lo);
        put(c, b, a.id);
        emit_arith(c, b, S_ShrL, c->t_u64, hi, a.id, const_u64(c, 32));
        uint32_t hi32 = nid(c);
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, c->t_u32);
        put(c, b, hi32);
        put(c, b, hi);
        uint32_t cl = popc32(c, b, lo), ch = popc32(c, b, hi32);
        id = nid(c);
        emit_arith(c, b, S_IAdd, c->t_u32, id, cl, ch);
    } else {
        id = popc32(c, b, a.id);
    }
    def_val(c, in->dst, PTY_U32, id);
}

void emit_clz_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a;
    uint32_t id;
    if (check_dst(c, in->dst, PTY_U32, line) != 0)
        return;
    a = resolve(c, b, &in->src[0], in->ty, line);
    if (c->failed)
        return;
    if (in->ty == PTY_U64)
        id = clz64_val(c, b, a.id, line);
    else
        id = clz32(c, b, a.id, line);
    if (c->failed)
        return;
    def_val(c, in->dst, PTY_U32, id);
}

void emit_brev_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a;
    uint32_t id;
    if (check_dst(c, in->dst, in->ty, line) != 0)
        return;
    a = resolve(c, b, &in->src[0], in->ty, line);
    if (c->failed)
        return;
    if (in->ty == PTY_U64) {
        uint32_t hi = nid(c);
        emit_arith(c, b, S_ShrL, c->t_u64, hi, a.id, const_u64(c, 32));
        uint32_t hi32 = nid(c), lo32 = nid(c);
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, c->t_u32);
        put(c, b, hi32);
        put(c, b, hi);
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, c->t_u32);
        put(c, b, lo32);
        put(c, b, a.id);
        uint32_t rh = brev32_val(c, b, hi32);
        uint32_t rl = brev32_val(c, b, lo32);
        uint32_t rl64 = nid(c), rh64 = nid(c), sh = nid(c);
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, c->t_u64);
        put(c, b, rl64);
        put(c, b, rl);
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, c->t_u64);
        put(c, b, rh64);
        put(c, b, rh);
        emit_arith(c, b, S_Shl, c->t_u64, sh, rl64, const_u64(c, 32));
        id = nid(c);
        emit_arith(c, b, S_Or, c->t_u64, id, sh, rh64);
        def_val(c, in->dst, PTY_U64, id);
        return;
    }
    id = brev32_val(c, b, a.id);
    def_val(c, in->dst, PTY_U32, id);
}

void emit_bfe_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2, d;
    if (check_dst(c, in->dst, in->ty, line) != 0)
        return;
    a = resolve(c, b, &in->src[0], in->ty, line);
    c2 = resolve(c, b, &in->src[1], PTY_U32, line);
    d = resolve(c, b, &in->src[2], PTY_U32, line);
    if (c->failed)
        return;
    uint32_t a64;
    if (in->ty == PTY_U64) {
        a64 = a.id;
    } else {
        a64 = nid(c);
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, c->t_u64);
        put(c, b, a64);
        put(c, b, a.id);
    }
    uint32_t res = bfe_value(c, b, a64, c2.id, d.id,
                              in->ty == PTY_U64, in->sign, line);
    if (c->failed)
        return;
    if (in->ty == PTY_U32) {
        uint32_t r32 = nid(c);
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, c->t_u32);
        put(c, b, r32);
        put(c, b, res);
        res = r32;
    }
    def_val(c, in->dst, in->ty, res);
}

void emit_bfi_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2, d;
    if (check_dst(c, in->dst, in->ty, line) != 0)
        return;
    a = resolve(c, b, &in->src[0], in->ty, line);
    c2 = resolve(c, b, &in->src[1], in->ty, line);
    d = resolve(c, b, &in->src[2], PTY_U32, line);
    val_t e = resolve(c, b, &in->src[3], PTY_U32, line);
    if (c->failed)
        return;
    int w64 = in->ty == PTY_U64;
    uint32_t a64 = a.id, b64 = c2.id;
    if (!w64) {
        a64 = nid(c);
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, c->t_u64);
        put(c, b, a64);
        put(c, b, a.id);
        b64 = nid(c);
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, c->t_u64);
        put(c, b, b64);
        put(c, b, c2.id);
    }
    uint32_t res = bfi_value(c, b, a64, b64, d.id, e.id, w64, line);
    if (c->failed)
        return;
    if (!w64) {
        uint32_t r32 = nid(c);
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, c->t_u32);
        put(c, b, r32);
        put(c, b, res);
        res = r32;
    }
    def_val(c, in->dst, in->ty, res);
}

void emit_bmsk_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2;
    uint32_t id;
    if (check_dst(c, in->dst, PTY_U32, line) != 0)
        return;
    a = resolve(c, b, &in->src[0], PTY_U32, line);
    c2 = resolve(c, b, &in->src[1], PTY_U32, line);
    if (c->failed)
        return;
    uint32_t a1 = nid(c), b1 = nid(c);
    emit_arith(c, b, S_And, c->t_u32, a1, a.id, const_u32(c, 31));
    emit_arith(c, b, S_And, c->t_u32, b1, c2.id, const_u32(c, 31));
    uint32_t m0 = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, m0, const_u32(c, 0xffffffffu),
               a1);
    if (in->sz_clamp) {
        uint32_t po = do_cmp(c, b, PCM_GE, PTY_U32, 0, a.id,
                             const_u32(c, 32), line);
        if (c->failed)
            return;
        m0 = sel_u32(c, b, po, const_u32(c, 0), m0);
    }
    uint32_t sum = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u32, sum, a1, b1);
    uint32_t over = do_cmp(c, b, PCM_GE, PTY_U32, 0, sum,
                           const_u32(c, 32), line);
    uint32_t sm = nid(c), m1s = nid(c);
    emit_arith(c, b, S_And, c->t_u32, sm, sum, const_u32(c, 31));
    emit_arith(c, b, S_Shl, c->t_u32, m1s, const_u32(c, 0xffffffffu),
               sm);
    uint32_t bz = do_cmp(c, b, PCM_EQ, PTY_U32, 0, b1,
                         const_u32(c, 0), line);
    if (c->failed)
        return;
    uint32_t notzero = nid(c), t1 = nid(c);
    put(c, b, (4u << 16) | S_LogicalNot);
    put(c, b, c->t_bool);
    put(c, b, notzero);
    put(c, b, bz);
    t1 = sel_u32(c, b, notzero, m1s, const_u32(c, 0xffffffffu));
    uint32_t bad = over;
    if (in->sz_clamp) {
        uint32_t wo = do_cmp(c, b, PCM_GE, PTY_U32, 0, c2.id,
                             const_u32(c, 32), line);
        if (c->failed)
            return;
        uint32_t t = nid(c);
        put(c, b, (5u << 16) | S_LogicalOr);
        put(c, b, c->t_bool);
        put(c, b, t);
        put(c, b, over);
        put(c, b, wo);
        bad = t;
    }
    uint32_t m1 = sel_u32(c, b, bad, const_u32(c, 0), t1);
    uint32_t nm1 = nid(c);
    put(c, b, (4u << 16) | S_Not);
    put(c, b, c->t_u32);
    put(c, b, nm1);
    put(c, b, m1);
    id = nid(c);
    emit_arith(c, b, S_And, c->t_u32, id, m0, nm1);
    def_val(c, in->dst, PTY_U32, id);
}

void emit_bfind_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a;
    uint32_t id;
    if (check_dst(c, in->dst, PTY_U32, line) != 0)
        return;
    a = resolve(c, b, &in->src[0], in->ty, line);
    if (c->failed)
        return;
    uint32_t v = a.id;
    if (in->sign) {
        uint32_t top, all;
        if (in->ty == PTY_U64) {
            top = nid(c);
            emit_arith(c, b, S_ShrL, c->t_u64, top, v, const_u64(c, 63));
            all = const_u64(c, 0xffffffffffffffffull);
        } else {
            top = nid(c);
            emit_arith(c, b, S_ShrL, c->t_u32, top, v, const_u32(c, 31));
            all = const_u32(c, 0xffffffffu);
        }
        uint32_t neg = do_cmp(c, b, PCM_NE, in->ty == PTY_U64 ? PTY_U64 : PTY_U32,
                              0, top, in->ty == PTY_U64 ? const_u64(c, 0)
                                                        : const_u32(c, 0), line);
        if (c->failed)
            return;
        uint32_t m = nid(c);
        if (in->ty == PTY_U64) {
            uint32_t nv = nid(c);
            put(c, b, (4u << 16) | S_SNegate);
            put(c, b, c->t_u64);
            put(c, b, nv);
            put(c, b, top);
            m = sel_u64(c, b, neg, all, const_u64(c, 0));
        } else {
            uint32_t nv = nid(c);
            put(c, b, (4u << 16) | S_SNegate);
            put(c, b, c->t_u32);
            put(c, b, nv);
            put(c, b, top);
            m = sel_u32(c, b, neg, all, const_u32(c, 0));
        }
        v = nid(c);
        if (in->ty == PTY_U64)
            emit_arith(c, b, S_Xor, c->t_u64, v, a.id, m);
        else
            emit_arith(c, b, S_Xor, c->t_u32, v, a.id, m);
    }
    uint32_t cz;
    if (in->ty == PTY_U64)
        cz = clz64_val(c, b, v, line);
    else
        cz = clz32(c, b, v, line);
    if (c->failed)
        return;
    uint32_t z = do_cmp(c, b, PCM_EQ, in->ty == PTY_U64 ? PTY_U64 : PTY_U32,
                        0, v, in->ty == PTY_U64 ? const_u64(c, 0)
                                                : const_u32(c, 0), line);
    if (c->failed)
        return;
    uint32_t res;
    if (in->arg) {
        uint32_t maxb = const_u32(c, in->ty == PTY_U64 ? 63u : 31u);
        res = nid(c);
        emit_arith(c, b, S_ISub, c->t_u32, res, maxb, cz);
    } else {
        res = cz;
    }
    id = sel_u32(c, b, z, const_u32(c, 0xffffffffu), res);
    def_val(c, in->dst, PTY_U32, id);
}
