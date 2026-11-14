#include "instructions/arith/arith.h"

void emit_madlo_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2, d;
    uint32_t id;
        if (in->cc_in || in->cc_out) {
            // Extended mad/madc: select the high or low product half,
            // add c (+ carry-in for madc), and optionally write the
            // addition carry-out. Supports 32/64-bit types.
            if (in->ty != PTY_U32 && in->ty != PTY_U64) {
                failf(c, line, "extended mad supports .u32/.s32/.u64/.s64");
                return;
            }
            if (check_dst(c, in->dst, in->ty, line) != 0)
                return;
            a = resolve(c, b, &in->src[0], in->ty, line);
            c2 = resolve(c, b, &in->src[1], in->ty, line);
            d = resolve(c, b, &in->src[2], in->ty, line);
            if (c->failed)
                return;
            int w64 = in->ty == PTY_U64;
            uint32_t part;
            if (in->is_hi) {
                part = mul_hi_value(c, b, in->ty, in->sign, a.id,
                                    c2.id, line);
                if (c->failed)
                    return;
            } else if (w64) {
                part = nid(c);
                emit_arith(c, b, S_IMul, c->t_u64, part, a.id, c2.id);
            } else {
                part = nid(c);
                emit_arith(c, b, S_IMul, c->t_u32, part, a.id, c2.id);
            }
            uint32_t p64, c64v;
            if (w64) {
                p64 = part;
                c64v = d.id;
            } else {
                p64 = nid(c);
                c64v = nid(c);
                put(c, b, (4u << 16) | S_UConvert);
                put(c, b, c->t_u64);
                put(c, b, p64);
                put(c, b, part);
                put(c, b, (4u << 16) | S_UConvert);
                put(c, b, c->t_u64);
                put(c, b, c64v);
                put(c, b, d.id);
            }
            uint32_t cin = const_u64(c, 0);
            if (in->cc_in) {
                if (!c->cc_defined) {
                    failf(c, line, "carry flag read before any .cc write");
                    return;
                }
                cin = sel_u64(c, b, c->cc_cf, const_u64(c, 1),
                              const_u64(c, 0));
            }
            uint32_t sum = 0;
            uint32_t carry = add_carry_u64(c, b, w64, p64, c64v, cin,
                                           &sum, line);
            if (c->failed)
                return;
            if (w64) {
                id = sum;
            } else {
                id = nid(c);
                put(c, b, (4u << 16) | S_UConvert);
                put(c, b, c->t_u32);
                put(c, b, id);
                put(c, b, sum);
            }
            if (in->cc_out) {
                c->cc_cf = carry;
                c->cc_defined = 1;
            }
            def_val(c, in->dst, in->ty, id);
            return;
        }
        if (in->is_hi) {
            // mad.hi: high half of a*b plus c (optionally saturated).
            if (check_dst(c, in->dst, in->ty, line) != 0)
                return;
            a = resolve(c, b, &in->src[0], in->ty, line);
            c2 = resolve(c, b, &in->src[1], in->ty, line);
            d = resolve(c, b, &in->src[2], in->ty, line);
            if (c->failed)
                return;
            uint32_t hi = mul_hi_value(c, b, in->ty, in->sign, a.id,
                                       c2.id, line);
            if (c->failed)
                return;
            if (in->sat) {
                uint32_t x64 = nid(c), c64v = nid(c);
                put(c, b, (4u << 16) | S_SConvert);
                put(c, b, c->t_u64);
                put(c, b, x64);
                put(c, b, hi);
                put(c, b, (4u << 16) | S_SConvert);
                put(c, b, c->t_u64);
                put(c, b, c64v);
                put(c, b, d.id);
                uint32_t sum = nid(c);
                emit_arith(c, b, S_IAdd, c->t_u64, sum, x64, c64v);
                uint32_t lo_ok = do_cmp(c, b, PCM_GE, PTY_U64, 1, sum,
                                        const_u64(c, 0xffffffff80000000ull), line);
                uint32_t hi_ok = do_cmp(c, b, PCM_LT, PTY_U64, 1, sum,
                                        const_u64(c, 0x80000000ull), line);
                if (c->failed)
                    return;
                uint32_t lo = nid(c);
                put(c, b, (4u << 16) | S_UConvert);
                put(c, b, c->t_u32);
                put(c, b, lo);
                put(c, b, sum);
                uint32_t mid = sel_u32(c, b, hi_ok, lo,
                                       const_u32(c, 0x7fffffffu));
                id = sel_u32(c, b, lo_ok, mid, const_u32(c, 0x80000000u));
            } else if (in->ty == PTY_U64) {
                id = nid(c);
                emit_arith(c, b, S_IAdd, c->t_u64, id, hi, d.id);
            } else {
                id = nid(c);
                emit_arith(c, b, S_IAdd, c->t_u32, id, hi, d.id);
            }
            if (in->ty == PTY_U16) {
                uint32_t m = nid(c);
                emit_arith(c, b, S_And, c->t_u32, m, id,
                           const_u32(c, 0xffffu));
                id = m;
            }
            def_val(c, in->dst, in->ty, id);
            return;
        }
        if (check_dst(c, in->dst, in->ty, line) != 0)
            return;
        // Pointer form (address arithmetic): d = base + a*b, exactly one
        // pointer among the three sources, same propagation as add/sub.
        {
            int pi = -1, nptr = 0;
            for (int s = 0; s < 3; s++) {
                if (in->src[s].kind == OPD_REG) {
                    reg_t *qr = find_reg(c, in->src[s].reg);
                    if (qr && qr->defined && qr->is_ptr) {
                        nptr++;
                        pi = s;
                    }
                }
            }
            if (nptr == 1) {
                reg_t *pr = find_reg(c, in->src[pi].reg);
                val_t mv1, mv2;
                int s1 = pi == 0 ? 1 : 0;
                int s2 = pi == 2 ? 1 : 2;
                mv1 = resolve(c, b, &in->src[s1], in->ty, line);
                if (c->failed)
                    return;
                mv2 = resolve(c, b, &in->src[s2], in->ty, line);
                if (c->failed)
                    return;
                uint32_t prod = nid(c), sum = nid(c);
                // Offsets are tracked as u64; a 32-bit mad must compute
                // in 64 bits and truncate (the register only holds 32).
                if (in->ty == PTY_U64) {
                    emit_arith(c, b, S_IMul, c->t_u64, prod, mv1.id,
                               mv2.id);
                    emit_arith(c, b, S_IAdd, c->t_u64, sum, prod,
                               pr->off);
                } else {
                    uint32_t a64 = nid(c), b64 = nid(c), p64 = nid(c),
                             s64 = nid(c), s32 = nid(c);
                    put(c, b, (4u << 16) | S_UConvert);
                    put(c, b, c->t_u64);
                    put(c, b, a64);
                    put(c, b, mv1.id);
                    put(c, b, (4u << 16) | S_UConvert);
                    put(c, b, c->t_u64);
                    put(c, b, b64);
                    put(c, b, mv2.id);
                    emit_arith(c, b, S_IMul, c->t_u64, p64, a64, b64);
                    emit_arith(c, b, S_IAdd, c->t_u64, s64, p64, pr->off);
                    put(c, b, (4u << 16) | S_UConvert);
                    put(c, b, c->t_u32);
                    put(c, b, s32);
                    put(c, b, s64);
                    // Tracked offsets are u64 ids everywhere else (and
                    // the .b32 register only ever holds 32 bits), so
                    // hand back the zero-extended 32-bit result.
                    uint32_t z64 = nid(c);
                    put(c, b, (4u << 16) | S_UConvert);
                    put(c, b, c->t_u64);
                    put(c, b, z64);
                    put(c, b, s32);
                    sum = z64;
                }
                // A 32-bit register also carries the numeric word: the
                // lifted addressing idiom reuses the same register as
                // an address (offset) and as one half of a 64-bit
                // pointer.  Keep the word alongside the pointer home so
                // value uses and state saves see it.
                uint32_t vword = 0;
                int wok = 0;
                if (decl_int_width(in->ty) == 32 && pr->cur &&
                    pr->cur_ty == PTY_U32) {
                    uint32_t x, y, z;
                    if (pi == 2) {
                        x = mv1.id; y = mv2.id; z = pr->cur;
                    } else if (pi == 0) {
                        x = pr->cur; y = mv1.id; z = mv2.id;
                    } else {
                        x = mv1.id; y = pr->cur; z = mv2.id;
                    }
                    uint32_t prod = nid(c);
                    emit_arith(c, b, S_IMul, c->t_u32, prod, x, y);
                    vword = nid(c);
                    emit_arith(c, b, S_IAdd, c->t_u32, vword, prod, z);
                    wok = 1;
                }
                if (pr->shvar >= 0)
                    def_shared(c, in->dst, pr->shvar, sum);
                else if (pr->glvar >= 0)
                    def_global(c, in->dst, pr->glvar, sum);
                else {
                    def_ptr(c, in->dst, pr->binding, sum);
                    if (pr->binding == -2) {
                        reg_t *dr = find_reg(c, in->dst);
                        if (dr) dr->lovar = pr->lovar;
                    }
                }
                if (wok) {
                    reg_t *dr = find_reg(c, in->dst);
                    if (dr) {
                        dr->cur = vword;
                        dr->cur_ty = PTY_U32;
                    }
                }
                return;
            }
            if (nptr > 1) {
                failf(c, line, "mad.lo with more than one pointer");
                return;
            }
        }
        a = resolve(c, b, &in->src[0], in->ty, line);
        c2 = resolve(c, b, &in->src[1], in->ty, line);
        d = resolve(c, b, &in->src[2], in->ty, line);
        if (c->failed)
            return;
        id = nid(c);
        // 64-bit: OpIMul wraps mod 2^64, which is the low 64 bits of the
        // full product (signed and unsigned agree bit-for-bit).
        emit_arith(c, b, S_IMul, in->ty == PTY_U64 ? c->t_u64 : c->t_u32,
                   id, a.id, c2.id);
        {
            uint32_t r = nid(c);
            emit_arith(c, b, S_IAdd, in->ty == PTY_U64 ? c->t_u64 : c->t_u32,
                       r, id, d.id);
            if (in->ty == PTY_U16) {
                uint32_t m = nid(c);
                emit_arith(c, b, S_And, c->t_u32, m, r,
                           const_u32(c, 0xffffu));
                r = m;
            }
            def_val(c, in->dst, in->ty, r);
        }
        return;
}
