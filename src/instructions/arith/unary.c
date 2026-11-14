#include "instructions/arith/arith.h"

void emit_abs_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a;
    uint32_t id;
        // Wrap absolute value (INT_MIN stays INT_MIN, like hardware) or
        // float absolute value (sign bit cleared).
        if (check_dst(c, in->dst, in->ty, line) != 0)
            return;
        a = resolve(c, b, &in->src[0], in->ty, line);
        if (c->failed)
            return;
        if (in->ty == PTY_F16 || in->ty == PTY_BF16) {
            int is_bf16 = in->ty == PTY_BF16;
            uint32_t v = a.id;
            if (in->ftz) {
                v = half_ftz_bits(c, b, is_bf16, v, line);
                if (c->failed)
                    return;
            }
            id = nid(c);
            emit_arith(c, b, S_And, c->t_u32, id, v,
                       const_u32(c, 0x7fffu));
            def_val(c, in->dst, in->ty, id);
            return;
        }
        if (in->ty == PTY_F16X2 || in->ty == PTY_BF16X2) {
            int is_bf16 = in->ty == PTY_BF16X2;
            uint32_t hi, lo;
            split_pack(c, b, a.id, &hi, &lo);
            if (in->ftz) {
                hi = half_ftz_bits(c, b, is_bf16, hi, line);
                lo = half_ftz_bits(c, b, is_bf16, lo, line);
                if (c->failed)
                    return;
            }
            uint32_t h = nid(c), l = nid(c);
            emit_arith(c, b, S_And, c->t_u32, h, hi,
                       const_u32(c, 0x7fffu));
            emit_arith(c, b, S_And, c->t_u32, l, lo,
                       const_u32(c, 0x7fffu));
            def_val(c, in->dst, in->ty, pack2(c, b, h, l));
            return;
        }
        if (in->ty == PTY_F32 || in->ty == PTY_F64) {
            uint32_t v = a.id;
            if (in->ty == PTY_F32 && in->ftz) {
                v = flush_f32_denorm(c, b, v, line);
                if (c->failed)
                    return;
            }
            if (in->ty == PTY_F64) {
                uint32_t bits = nid(c), m = nid(c);
                put(c, b, (4u << 16) | S_Bitcast);
                put(c, b, c->t_u64);
                put(c, b, bits);
                put(c, b, v);
                emit_arith(c, b, S_And, c->t_u64, m, bits,
                           const_u64(c, 0x7fffffffffffffffull));
                id = nid(c);
                put(c, b, (4u << 16) | S_Bitcast);
                put(c, b, c->t_f64);
                put(c, b, id);
                put(c, b, m);
            } else {
                uint32_t bits = nid(c), m = nid(c);
                put(c, b, (4u << 16) | S_Bitcast);
                put(c, b, c->t_u32);
                put(c, b, bits);
                put(c, b, v);
                emit_arith(c, b, S_And, c->t_u32, m, bits,
                           const_u32(c, 0x7fffffffu));
                id = nid(c);
                put(c, b, (4u << 16) | S_Bitcast);
                put(c, b, c->t_f32);
                put(c, b, id);
                put(c, b, m);
                if (in->ftz) {
                    id = flush_f32_denorm(c, b, id, line);
                    if (c->failed)
                        return;
                }
            }
            def_val(c, in->dst, in->ty, id);
            return;
        }
        if (in->pack) {
            int width = in->pack == 2 ? 16 : 8;
            uint32_t mask = (1u << width) - 1u;
            uint32_t out = 0;
            for (int i = 0; i < in->pack; i++) {
                uint32_t v = pack_lane_get(c, b, a.id, i, width, 1);
                uint32_t neg = nid(c);
                put(c, b, (4u << 16) | S_SNegate);
                put(c, b, c->t_u32);
                put(c, b, neg);
                put(c, b, v);
                uint32_t lt = do_cmp(c, b, PCM_LT, PTY_U32, 1, v,
                                     const_u32(c, 0), line);
                if (c->failed)
                    return;
                uint32_t lane = sel_u32(c, b, lt, neg, v);
                out = pack_lane_put(c, b, out, lane, i, width, mask);
            }
            def_val(c, in->dst, PTY_U32, out);
            return;
        }
        if (in->ty == PTY_U64) {
            uint32_t neg = nid(c);
            put(c, b, (4u << 16) | S_SNegate);
            put(c, b, c->t_u64);
            put(c, b, neg);
            put(c, b, a.id);
            uint32_t lt = do_cmp(c, b, PCM_LT, PTY_U64, 1, a.id,
                                 const_u64(c, 0), line);
            if (c->failed)
                return;
            id = sel_u64(c, b, lt, neg, a.id);
            def_val(c, in->dst, PTY_U64, id);
            return;
        }
        uint32_t v = a.id;
        if (in->ty == PTY_U16)
            v = pack_lane_get(c, b, a.id, 0, 16, 1);
        uint32_t neg = nid(c);
        put(c, b, (4u << 16) | S_SNegate);
        put(c, b, c->t_u32);
        put(c, b, neg);
        put(c, b, v);
        uint32_t lt = do_cmp(c, b, PCM_LT, PTY_U32, 1, v,
                             const_u32(c, 0), line);
        if (c->failed)
            return;
        id = sel_u32(c, b, lt, neg, v);
        if (in->ty == PTY_U16) {
            uint32_t m = nid(c);
            emit_arith(c, b, S_And, c->t_u32, m, id,
                       const_u32(c, 0xffffu));
            id = m;
        }
        def_val(c, in->dst, in->ty, id);
        return;
    }

void emit_neg_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a;
    uint32_t id;
        // Integer neg is wrap (OpSNegate); float neg preserves NaN
        // payload conventions via OpFNegate; f16 neg flips bit 15
        // (exact for all values, NaN payloads included). s8x4 negates
        // each byte lane independently.
        ptx_ty_t nt = eff_ty(c, in->ty);
        if (check_dst(c, in->dst, in->ty, line) != 0)
            return;
        a = resolve(c, b, &in->src[0], in->ty, line);
        if (c->failed)
            return;
        if (in->ty == PTY_F16 || in->ty == PTY_BF16) {
            int is_bf16 = in->ty == PTY_BF16;
            uint32_t v = a.id;
            if (in->ftz) {
                v = half_ftz_bits(c, b, is_bf16, v, line);
                if (c->failed)
                    return;
            }
            id = nid(c);
            emit_arith(c, b, S_Xor, c->t_u32, id, v,
                       const_u32(c, 0x8000u));
            def_val(c, in->dst, in->ty, id);
            return;
        }
        if (in->ty == PTY_F16X2 || in->ty == PTY_BF16X2) {
            int is_bf16 = in->ty == PTY_BF16X2;
            uint32_t hi, lo;
            split_pack(c, b, a.id, &hi, &lo);
            if (in->ftz) {
                hi = half_ftz_bits(c, b, is_bf16, hi, line);
                lo = half_ftz_bits(c, b, is_bf16, lo, line);
                if (c->failed)
                    return;
            }
            uint32_t h = nid(c), l = nid(c);
            emit_arith(c, b, S_Xor, c->t_u32, h, hi,
                       const_u32(c, 0x8000u));
            emit_arith(c, b, S_Xor, c->t_u32, l, lo,
                       const_u32(c, 0x8000u));
            def_val(c, in->dst, in->ty, pack2(c, b, h, l));
            return;
        }
        if (in->pack) {
            uint32_t mask = 0xffu, out = 0;
            for (int i = 0; i < 4; i++) {
                uint32_t v = pack_lane_get(c, b, a.id, i, 8, 0);
                uint32_t neg = nid(c);
                emit_arith(c, b, S_ISub, c->t_u32, neg,
                           const_u32(c, 0), v);
                out = pack_lane_put(c, b, out, neg, i, 8, mask);
            }
            def_val(c, in->dst, PTY_U32, out);
            return;
        }
        id = nid(c);
        if (nt == PTY_F16) {
            emit_arith(c, b, S_Xor, c->t_u32, id, a.id,
                       const_u32(c, 0x8000u));
            def_val(c, in->dst, PTY_F16, id);
            return;
        }
        if (nt == PTY_F32 || nt == PTY_F64) {
            uint32_t v = a.id;
            if (nt == PTY_F32 && in->ftz) {
                v = flush_f32_denorm(c, b, v, line);
                if (c->failed)
                    return;
            }
            put(c, b, (4u << 16) | S_FNegate);
            put(c, b, nt == PTY_F64 ? c->t_f64 : c->t_f32);
            id = nid(c);
            put(c, b, id);
            put(c, b, v);
            if (nt == PTY_F32 && in->ftz) {
                id = flush_f32_denorm(c, b, id, line);
                if (c->failed)
                    return;
            }
            def_val(c, in->dst, in->ty, id);
            return;
        } else if (nt == PTY_U64) {
            put(c, b, (4u << 16) | S_SNegate);
            put(c, b, c->t_u64);
        } else {
            put(c, b, (4u << 16) | S_SNegate);
            put(c, b, c->t_u32);
        }
        put(c, b, id);
        put(c, b, a.id);
        if (nt == PTY_U16) {
            uint32_t m = nid(c);
            emit_arith(c, b, S_And, c->t_u32, m, id,
                       const_u32(c, 0xffffu));
            id = m;
        }
        def_val(c, in->dst, in->ty, id);
        return;
    }
