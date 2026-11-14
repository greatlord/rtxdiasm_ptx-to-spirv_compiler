#include "instructions/arith/arith.h"

void emit_fma_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2, d;
    uint32_t id;
        if (!in->f32x2 && in->ty != PTY_F32 && in->ty != PTY_F16 &&
            in->ty != PTY_BF16 && in->ty != PTY_F64 &&
            in->ty != PTY_F16X2 && in->ty != PTY_BF16X2) {
            failf(c, line, "fma needs .f32/.f16/.bf16/.f64/.f16x2/.bf16x2");
            return;
        }
        if (check_dst(c, in->dst, in->ty, line) != 0)
            return;
        a = resolve(c, b, &in->src[0], in->ty, line);
        c2 = resolve(c, b, &in->src[1], in->ty, line);
        d = resolve(c, b, &in->src[2], in->ty, line);
        if (c->failed)
            return;
        if (in->f32x2) {
            uint32_t alo, ahi, blo, bhi, clo, chi;
            f32x2_split(c, b, a.id, &alo, &ahi);
            f32x2_split(c, b, c2.id, &blo, &bhi);
            f32x2_split(c, b, d.id, &clo, &chi);
            if (c->failed)
                return;
            uint32_t rlo = f32_fma_lane(c, b, alo, blo, clo, in->ftz,
                                        in->sat, line);
            uint32_t rhi = f32_fma_lane(c, b, ahi, bhi, chi, in->ftz,
                                        in->sat, line);
            if (c->failed)
                return;
            def_val(c, in->dst, PTY_U64,
                    f32x2_pack(c, b, rlo, rhi));
            return;
        }
        if (in->ty == PTY_F16X2 && c->caps.f16_native && !c->caps.soft_fma &&
            !in->ftz && !in->sat && !in->relu) {
            ensure_f16_types(c);
            uint32_t va = nid(c), vb = nid(c), vc = nid(c), vf = nid(c);
            put(c, b, (4u << 16) | S_Bitcast);
            put(c, b, c->t_v2f16);
            put(c, b, va);
            put(c, b, a.id);
            put(c, b, (4u << 16) | S_Bitcast);
            put(c, b, c->t_v2f16);
            put(c, b, vb);
            put(c, b, c2.id);
            put(c, b, (4u << 16) | S_Bitcast);
            put(c, b, c->t_v2f16);
            put(c, b, vc);
            put(c, b, d.id);
            put(c, b, (8u << 16) | S_ExtInst);
            put(c, b, c->t_v2f16);
            put(c, b, vf);
            put(c, b, c->glsl450);
            put(c, b, 50); // GLSL.std.450 Fma
            put(c, b, va);
            put(c, b, vb);
            put(c, b, vc);
            id = nid(c);
            put(c, b, (4u << 16) | S_Bitcast);
            put(c, b, c->t_u32);
            put(c, b, id);
            put(c, b, vf);
            def_val(c, in->dst, PTY_F16X2, id);
            return;
        }
        if (in->ty == PTY_F16X2 || in->ty == PTY_BF16X2) {
            // Packed lanes reuse the scalar mul+add-then-narrow
            // sequence per lane (bit-identical to two scalar fmas;
            // same last-ulp caveat as scalar f16).
            int is_bf16 = in->ty == PTY_BF16X2;
            uint32_t ahi, alo, bhi, blo, chi, clo;
            split_pack(c, b, a.id, &ahi, &alo);
            split_pack(c, b, c2.id, &bhi, &blo);
            split_pack(c, b, d.id, &chi, &clo);
            uint32_t rhi = pack_lane_fma(c, b, is_bf16, ahi, bhi, chi,
                                         in->ftz, in->sat, in->relu, line);
            if (c->failed)
                return;
            uint32_t rlo = pack_lane_fma(c, b, is_bf16, alo, blo, clo,
                                         in->ftz, in->sat, in->relu, line);
            if (c->failed)
                return;
            def_val(c, in->dst, in->ty, pack2(c, b, rhi, rlo));
            return;
        }
        if (in->ty == PTY_F16 || in->ty == PTY_BF16) {
            int is_bf = in->ty == PTY_BF16;
            uint32_t fa = is_bf ? bf16_to_f32val(c, b, a.id, line)
                                : f16_to_f32val(c, b, a.id, line);
            if (c->failed)
                return;
            uint32_t fb = is_bf ? bf16_to_f32val(c, b, c2.id, line)
                                : f16_to_f32val(c, b, c2.id, line);
            if (c->failed)
                return;
            uint32_t fc = is_bf ? bf16_to_f32val(c, b, d.id, line)
                                : f16_to_f32val(c, b, d.id, line);
            if (c->failed)
                return;
            if (in->ftz) {
                fa = flush_f32_denorm(c, b, fa, line);
                fb = flush_f32_denorm(c, b, fb, line);
                fc = flush_f32_denorm(c, b, fc, line);
                if (c->failed)
                    return;
            }
            uint32_t fr = nid(c);
            emit_fma_into(c, b, c->t_f32, fr, fa, fb, fc);
            if (in->ftz) {
                fr = flush_f32_denorm(c, b, fr, line);
                if (c->failed)
                    return;
            }
            if (in->sat) {
                fr = sat_f32(c, b, fr, line);
                if (c->failed)
                    return;
            }
            if (in->relu) {
                fr = relu_f32(c, b, fr, line);
                if (c->failed)
                    return;
            }
            id = is_bf ? f32_to_bf16bits(c, b, fr, line)
                       : f32_to_f16bits(c, b, fr, line);
            if (c->failed)
                return;
            def_val(c, in->dst, in->ty, id);
            return;
        }
        uint32_t fa = a.id, fb = c2.id, fc = d.id;
        if (in->ty == PTY_F32 && in->ftz) {
            fa = flush_f32_denorm(c, b, fa, line);
            fb = flush_f32_denorm(c, b, fb, line);
            fc = flush_f32_denorm(c, b, fc, line);
            if (c->failed)
                return;
        }
        id = nid(c);
        uint32_t fty = (in->ty == PTY_F64 && !c->emu_f64) ? c->t_f64
                                                          : c->t_f32;
        if (in->ty == PTY_F64 && !c->emu_f64) {
            // Always use the integer software FMA for f64.  RADV/ACO
            // reassociates plain floating-point add/sub (it collapses
            // (a+b)-b even through memory), which breaks the magic
            // 1.5*2^52 range reduction used by libdevice's exp/log/
            // normal code: (x*c + magic) - magic became x*c instead of
            // the rounded integer.  The software FMA builds the result
            // from integer significand operations, so the following
            // subtraction cannot be folded away.
            uint32_t r = f64_soft_fma(c, b, fa, fb, fc, in->frnd, line,
                                    NULL, NULL);
            if (c->failed)
                return;
            def_val(c, in->dst, in->ty, r);
            return;
        }
        if (in->ty == PTY_F32 && (in->frnd || c->caps.soft_fma)) {
            uint32_t r = f32_soft_fma(c, b, fa, fb, fc, in->frnd, line,
                                    NULL, NULL);
            if (c->failed)
                return;
            if (in->ftz) {
                r = flush_f32_denorm(c, b, r, line);
                if (c->failed)
                    return;
            }
            if (in->sat) {
                r = sat_f32(c, b, r, line);
                if (c->failed)
                    return;
            }
            def_val(c, in->dst, in->ty, r);
            return;
        }
        if (in->ty == PTY_F32) {
            uint32_t r = c->use_fma ? f32_fma3(c, b, fa, fb, fc)
                                    : f32_glsl_fma3(c, b, fa, fb, fc);
            if (c->failed)
                return;
            if (in->ftz) {
                r = flush_f32_denorm(c, b, r, line);
                if (c->failed)
                    return;
            }
            if (in->sat) {
                r = sat_f32(c, b, r, line);
                if (c->failed)
                    return;
            }
            def_val(c, in->dst, in->ty, r);
            return;
        }
        emit_arith(c, b, S_FMul, fty, id, fa, fb);
        {
            uint32_t r = nid(c);
            emit_arith(c, b, S_FAdd, fty, r, id, fc);
            if (in->ty == PTY_F32 && in->ftz) {
                r = flush_f32_denorm(c, b, r, line);
                if (c->failed)
                    return;
            }
            if (in->ty == PTY_F32 && in->sat) {
                r = sat_f32(c, b, r, line);
                if (c->failed)
                    return;
            }
            def_val(c, in->dst, in->ty, r);
        }
        return;
}
