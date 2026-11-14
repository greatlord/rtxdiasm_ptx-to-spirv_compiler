#include "instructions/arith/float/float.h"

void emit_minnan_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2;
    uint32_t id;
        // min/max.NaN: any NaN input produces a canonical quiet NaN;
        // otherwise the ordered result over the original patterns.
        if (check_dst(c, in->dst, in->ty, line) != 0)
            return;
        a = resolve(c, b, &in->src[0], in->ty, line);
        c2 = resolve(c, b, &in->src[1], in->ty, line);
        if (c->failed)
            return;
        if (in->ty == PTY_F16 || in->ty == PTY_BF16) {
            id = half_minnan_lane(c, b, in->ty == PTY_BF16, in->is_max,
                                  a.id, c2.id, line);
            if (c->failed)
                return;
            def_val(c, in->dst, in->ty, id);
            return;
        }
        if (in->ty == PTY_F16X2 || in->ty == PTY_BF16X2) {
            int is_bf16 = in->ty == PTY_BF16X2;
            uint32_t ahi, alo, bhi, blo;
            split_pack(c, b, a.id, &ahi, &alo);
            split_pack(c, b, c2.id, &bhi, &blo);
            uint32_t rh = half_minnan_lane(c, b, is_bf16, in->is_max,
                                           ahi, bhi, line);
            uint32_t rl = half_minnan_lane(c, b, is_bf16, in->is_max,
                                           alo, blo, line);
            if (c->failed)
                return;
            def_val(c, in->dst, in->ty, pack2(c, b, rh, rl));
            return;
        }
        uint32_t fa = a.id, fb = c2.id, pty =
            in->ty == PTY_F64 ? c->t_f64 : c->t_f32;
        ptx_ty_t ct = in->ty == PTY_F64 ? PTY_F64 : PTY_F32;
        uint32_t na = nid(c), nb = nid(c), nan = nid(c);
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
        put(c, b, (5u << 16) | S_LogicalOr);
        put(c, b, c->t_bool);
        put(c, b, nan);
        put(c, b, na);
        put(c, b, nb);
        uint32_t cond = do_cmp(c, b, in->is_max ? PCM_GT : PCM_LT,
                               ct, 0, fa, fb, line);
        if (c->failed)
            return;
        uint32_t mm = nid(c);
        put(c, b, (6u << 16) | S_Select);
        put(c, b, pty);
        put(c, b, mm);
        put(c, b, cond);
        put(c, b, fa);
        put(c, b, fb);
        uint32_t cn;
        if (in->ty == PTY_F32) {
            cn = const_f32(c, 0x7fc00000u);
        } else {
            uint32_t bits = const_u64(c, 0x7ff8000000000000ull);
            cn = nid(c);
            put(c, b, (4u << 16) | S_Bitcast);
            put(c, b, c->t_f64);
            put(c, b, cn);
            put(c, b, bits);
        }
        id = nid(c);
        put(c, b, (6u << 16) | S_Select);
        put(c, b, pty);
        put(c, b, id);
        put(c, b, nan);
        put(c, b, cn);
        put(c, b, mm);
        def_val(c, in->dst, in->ty, id);
        return;
    }
