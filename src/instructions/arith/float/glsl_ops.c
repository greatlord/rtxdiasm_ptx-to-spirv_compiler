#include "instructions/arith/float/float.h"

void emit_glsl_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a;
    uint32_t id;
        // Transcendentals via GLSL.std.450. Half forms widen each lane
        // to f32, evaluate, and narrow back; f64 uses the f64 op.
        if (in->ty != PTY_F32 && in->ty != PTY_F64 && in->ty != PTY_F16 &&
            in->ty != PTY_F16X2 && in->ty != PTY_BF16 &&
            in->ty != PTY_BF16X2) {
            failf(c, line, "transcendentals need a float type");
            return;
        }
        if (check_dst(c, in->dst, in->ty, line) != 0)
            return;
        a = resolve(c, b, &in->src[0], in->ty, line);
        if (c->failed)
            return;
        int is_bf = in->ty == PTY_BF16 || in->ty == PTY_BF16X2;
        int is_x2 = in->ty == PTY_F16X2 || in->ty == PTY_BF16X2;
        uint32_t f32op = nid(c);
        if (in->ty == PTY_F16 || in->ty == PTY_BF16) {
            uint32_t f = is_bf ? bf16_to_f32val(c, b, a.id, line)
                               : f16_to_f32val(c, b, a.id, line);
            if (c->failed)
                return;
            if (in->ftz) {
                f = flush_f32_denorm(c, b, f, line);
                if (c->failed)
                    return;
            }
            put(c, b, (6u << 16) | S_ExtInst);
            put(c, b, c->t_f32);
            put(c, b, f32op);
            put(c, b, c->glsl450);
            put(c, b, (uint32_t)in->glop);
            put(c, b, f);
            if (in->ftz) {
                f32op = flush_f32_denorm(c, b, f32op, line);
                if (c->failed)
                    return;
            }
            id = is_bf ? f32_to_bf16bits(c, b, f32op, line)
                       : f32_to_f16bits(c, b, f32op, line);
            if (c->failed)
                return;
            def_val(c, in->dst, in->ty, id);
            return;
        }
        if (is_x2) {
            uint32_t hi, lo;
            split_pack(c, b, a.id, &hi, &lo);
            uint32_t rh = nid(c), rl = nid(c);
            for (int lane = 0; lane < 2; lane++) {
                uint32_t bits = lane ? lo : hi;
                uint32_t f = is_bf ? bf16_to_f32val(c, b, bits, line)
                                   : f16_to_f32val(c, b, bits, line);
                if (c->failed)
                    return;
                if (in->ftz) {
                    f = flush_f32_denorm(c, b, f, line);
                    if (c->failed)
                        return;
                }
                uint32_t r = nid(c);
                put(c, b, (6u << 16) | S_ExtInst);
                put(c, b, c->t_f32);
                put(c, b, r);
                put(c, b, c->glsl450);
                put(c, b, (uint32_t)in->glop);
                put(c, b, f);
                if (in->ftz) {
                    r = flush_f32_denorm(c, b, r, line);
                    if (c->failed)
                        return;
                }
                uint32_t h = is_bf ? f32_to_bf16bits(c, b, r, line)
                                   : f32_to_f16bits(c, b, r, line);
                if (c->failed)
                    return;
                if (lane == 0)
                    rh = h;
                else
                    rl = h;
            }
            def_val(c, in->dst, in->ty, pack2(c, b, rh, rl));
            return;
        }
        ptx_ty_t gt = eff_ty(c, in->ty);
        uint32_t fa = a.id;
        if (in->ftz) {
            if (gt != PTY_F32) {
                failf(c, line, "ftz transcendentals need .f32");
                return;
            }
            fa = flush_f32_denorm(c, b, fa, line);
            if (c->failed)
                return;
        }
        id = nid(c);
        put(c, b, (6u << 16) | S_ExtInst);
        put(c, b, gt == PTY_F64 ? c->t_f64 : c->t_f32);
        put(c, b, id);
        put(c, b, c->glsl450);
        put(c, b, (uint32_t)in->glop);
        put(c, b, fa);
        if (in->ftz) {
            id = flush_f32_denorm(c, b, id, line);
            if (c->failed)
                return;
        }
        def_val(c, in->dst, in->ty, id);
        return;
    }
