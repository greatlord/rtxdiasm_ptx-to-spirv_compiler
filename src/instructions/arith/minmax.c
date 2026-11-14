#include "instructions/arith/arith.h"

void emit_minmax_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2;
    uint32_t id;
        // Lowered as compare+select (no GLSL.std.450 import needed).
        if (check_dst(c, in->dst, in->ty, line) != 0)
            return;
        a = resolve(c, b, &in->src[0], in->ty, line);
        c2 = resolve(c, b, &in->src[1], in->ty, line);
        if (c->failed)
            return;
        if (in->ty == PTY_F32 && in->ftz) {
            a.id = flush_f32_denorm(c, b, a.id, line);
            c2.id = flush_f32_denorm(c, b, c2.id, line);
            if (c->failed)
                return;
        }
        if (in->pack) {
            // Packed lane-wise min/max; .relu clamps negative signed
            // lanes to zero before repacking.
            int width = in->pack == 2 ? 16 : 8;
            uint32_t mask = (1u << width) - 1u;
            uint32_t out = 0;
            for (int i = 0; i < in->pack; i++) {
                uint32_t la = pack_lane_get(c, b, a.id, i, width, in->sign);
                uint32_t lb = pack_lane_get(c, b, c2.id, i, width, in->sign);
                uint32_t cond = do_cmp(c, b,
                                       in->is_max ? PCM_GT : PCM_LT,
                                       PTY_U32, in->sign, la, lb, line);
                if (c->failed)
                    return;
                uint32_t lane = nid(c);
                put(c, b, (6u << 16) | S_Select);
                put(c, b, c->t_u32);
                put(c, b, lane);
                put(c, b, cond);
                put(c, b, la);
                put(c, b, lb);
                if (in->relu) {
                    uint32_t neg = do_cmp(c, b, PCM_LT, PTY_U32, 1, lane,
                                          const_u32(c, 0), line);
                    if (c->failed)
                        return;
                    lane = sel_u32(c, b, neg, const_u32(c, 0), lane);
                }
                out = pack_lane_put(c, b, out, lane, i, width, mask);
                if (c->failed)
                    return;
            }
            def_val(c, in->dst, PTY_U32, out);
            return;
        }
        if (in->ty == PTY_F16 || in->ty == PTY_BF16) {
            // Halves compare as exact f32 values; select over the
            // original patterns. min_num semantics: a lone NaN yields
            // the other input.
            int is_bf16 = in->ty == PTY_BF16;
            uint32_t av = a.id, bv = c2.id;
            if (in->ftz) {
                av = half_ftz_bits(c, b, is_bf16, av, line);
                bv = half_ftz_bits(c, b, is_bf16, bv, line);
                if (c->failed)
                    return;
            }
            uint32_t fa = is_bf16 ? bf16_to_f32val(c, b, av, line)
                                  : f16_to_f32val(c, b, av, line);
            uint32_t fb = is_bf16 ? bf16_to_f32val(c, b, bv, line)
                                  : f16_to_f32val(c, b, bv, line);
            if (c->failed)
                return;
            uint32_t cond = do_cmp(c, b, in->is_max ? PCM_GT : PCM_LT,
                                   PTY_F32, 0, fa, fb, line);
            uint32_t na = nid(c), nb = nid(c);
            put(c, b, (5u << 16) | S_FUnordNotEqual);
            put(c, b, c->t_bool);
            put(c, b, na);
            put(c, b, fa);
            put(c, b, fa);
            put(c, b, (5u << 16) | S_FUnordNotEqual);
            put(c, b, c->t_bool);
            put(c, b, nb);
            put(c, b, fb);
            put(c, b, fb);
            if (c->failed)
                return;
            id = nid(c);
            put(c, b, (6u << 16) | S_Select);
            put(c, b, c->t_u32);
            put(c, b, id);
            put(c, b, cond);
            put(c, b, av);
            put(c, b, bv);
            uint32_t r1 = sel_u32(c, b, nb, av, id);
            uint32_t r2 = sel_u32(c, b, na, bv, r1);
            def_val(c, in->dst, in->ty, r2);
            return;
        }
        if (in->ty == PTY_F16X2 || in->ty == PTY_BF16X2) {
            int is_bf16 = in->ty == PTY_BF16X2;
            uint32_t ahi, alo, bhi, blo;
            split_pack(c, b, a.id, &ahi, &alo);
            split_pack(c, b, c2.id, &bhi, &blo);
            uint32_t lanes[4] = { ahi, alo, bhi, blo };
            if (in->ftz) {
                for (int i = 0; i < 4; i++)
                    lanes[i] = half_ftz_bits(c, b, is_bf16, lanes[i], line);
                ahi = lanes[0]; alo = lanes[1];
                bhi = lanes[2]; blo = lanes[3];
                if (c->failed)
                    return;
            }
            uint32_t rh = nid(c), rl = nid(c);
            for (int lane = 0; lane < 2; lane++) {
                uint32_t av = lane ? alo : ahi;
                uint32_t bv = lane ? blo : bhi;
                uint32_t fa = is_bf16 ? bf16_to_f32val(c, b, av, line)
                                      : f16_to_f32val(c, b, av, line);
                uint32_t fb = is_bf16 ? bf16_to_f32val(c, b, bv, line)
                                      : f16_to_f32val(c, b, bv, line);
                if (c->failed)
                    return;
                uint32_t cond = do_cmp(c, b, in->is_max ? PCM_GT : PCM_LT,
                                       PTY_F32, 0, fa, fb, line);
                uint32_t na = nid(c), nb = nid(c);
                put(c, b, (5u << 16) | S_FUnordNotEqual);
                put(c, b, c->t_bool);
                put(c, b, na);
                put(c, b, fa);
                put(c, b, fa);
                put(c, b, (5u << 16) | S_FUnordNotEqual);
                put(c, b, c->t_bool);
                put(c, b, nb);
                put(c, b, fb);
                put(c, b, fb);
                if (c->failed)
                    return;
                uint32_t sel = nid(c);
                put(c, b, (6u << 16) | S_Select);
                put(c, b, c->t_u32);
                put(c, b, sel);
                put(c, b, cond);
                put(c, b, av);
                put(c, b, bv);
                uint32_t r1 = sel_u32(c, b, nb, av, sel);
                uint32_t r2 = sel_u32(c, b, na, bv, r1);
                if (lane == 0)
                    rh = r2;
                else
                    rl = r2;
            }
            def_val(c, in->dst, in->ty, pack2(c, b, rh, rl));
            return;
        }
        uint32_t ca = a.id, cb = c2.id;
        if ((in->ty == PTY_U8 || in->ty == PTY_U16) && in->sign) {
            // Signed narrow values canonicalize zero-extended; widen to
            // a full two's-complement 32-bit value for the compare.
            uint32_t shift = (uint32_t)(in->ty == PTY_U8 ? 24 : 16);
            uint32_t la = nid(c), lb = nid(c);
            emit_arith(c, b, S_Shl, c->t_u32, la, a.id,
                       const_u32(c, shift));
            emit_arith(c, b, S_Shl, c->t_u32, lb, c2.id,
                       const_u32(c, shift));
            ca = nid(c);
            emit_arith(c, b, S_ShrA, c->t_u32, ca, la,
                       const_u32(c, shift));
            cb = nid(c);
            emit_arith(c, b, S_ShrA, c->t_u32, cb, lb,
                       const_u32(c, shift));
        }
        uint32_t cond = do_cmp(c, b, in->is_max ? PCM_GT : PCM_LT,
                               in->ty, in->sign, ca, cb, line);
        if (c->failed)
            return;
        id = nid(c);
        put(c, b, (6u << 16) | S_Select);
        put(c, b, ty_of(c, in->ty));
        put(c, b, id);
        put(c, b, cond);
        put(c, b, a.id);
        put(c, b, c2.id);
        if (in->ty == PTY_F32 && in->ftz) {
            id = flush_f32_denorm(c, b, id, line);
            if (c->failed)
                return;
        }
        if (in->relu) {
            uint32_t neg = do_cmp(c, b, PCM_LT, PTY_U32, 1, id,
                                  const_u32(c, 0), line);
            if (c->failed)
                return;
            id = sel_u32(c, b, neg, const_u32(c, 0), id);
        }
        def_val(c, in->dst, in->ty, id);
        return;
    }
