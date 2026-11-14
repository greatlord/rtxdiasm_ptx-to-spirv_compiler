#include "instructions/compare/compare.h"

void emit_set_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
        // Half-precision comparison matrix. f16x2 compares lane-wise;
        // integer destinations get 0xffff/0xffffffff masks, .f16/.f16x2
        // destinations get 1.0/0.0 in the corresponding precision.
        if (check_dst(c, in->dst, in->ty, line) != 0)
            return;
        if (in->ty2 == PTY_F16) {
            val_t x = resolve(c, b, &in->src[0], PTY_F16, line);
            if (c->failed)
                return;
            val_t y = resolve(c, b, &in->src[1], PTY_F16, line);
            if (c->failed)
                return;
            uint32_t xb = nid(c), yb = nid(c);
            emit_arith(c, b, S_And, c->t_u32, xb, x.id,
                       const_u32(c, 0xffffu));
            emit_arith(c, b, S_And, c->t_u32, yb, y.id,
                       const_u32(c, 0xffffu));
            if (in->ftz) {
                xb = flush_f16_bits(c, b, xb, line);
                yb = flush_f16_bits(c, b, yb, line);
            }
            uint32_t fa = f16_to_f32val(c, b, xb, line);
            uint32_t fb = f16_to_f32val(c, b, yb, line);
            if (c->failed)
                return;
            uint32_t t = do_cmp(c, b, in->cmp, PTY_F32, 0, fa, fb, line);
            if (c->failed)
                return;
            t = set_bop_join(c, b, in, t, line);
            if (c->failed)
                return;
            uint32_t tv = in->ty == PTY_U32 ? 0xffffffffu
                          : in->ty == PTY_U16 ? 0xffffu : 0x3c00u;
            uint32_t v = sel_u32(c, b, t, const_u32(c, tv),
                                 const_u32(c, 0));
            def_val(c, in->dst, in->ty, v);
            return;
        }
        // Packed f16x2 source.
        val_t px = resolve(c, b, &in->src[0], PTY_U32, line);
        if (c->failed)
            return;
        val_t py = resolve(c, b, &in->src[1], PTY_U32, line);
        if (c->failed)
            return;
        uint32_t xhi, xlo, yhi, ylo;
        split_pack(c, b, px.id, &xhi, &xlo);
        split_pack(c, b, py.id, &yhi, &ylo);
        if (in->ftz) {
            xhi = flush_f16_bits(c, b, xhi, line);
            xlo = flush_f16_bits(c, b, xlo, line);
            yhi = flush_f16_bits(c, b, yhi, line);
            ylo = flush_f16_bits(c, b, ylo, line);
        }
        uint32_t truev = in->ty == PTY_F16X2 ? 0x3c00u : 0xffffu;
        uint32_t mhi, mlo;
        {
            uint32_t fa = f16_to_f32val(c, b, xhi, line);
            uint32_t fb = f16_to_f32val(c, b, yhi, line);
            if (c->failed)
                return;
            uint32_t t = do_cmp(c, b, in->cmp, PTY_F32, 0, fa, fb, line);
            if (c->failed)
                return;
            t = set_bop_join(c, b, in, t, line);
            if (c->failed)
                return;
            mhi = sel_u32(c, b, t, const_u32(c, truev),
                          const_u32(c, 0));
        }
        {
            uint32_t fa = f16_to_f32val(c, b, xlo, line);
            uint32_t fb = f16_to_f32val(c, b, ylo, line);
            if (c->failed)
                return;
            uint32_t t = do_cmp(c, b, in->cmp, PTY_F32, 0, fa, fb, line);
            if (c->failed)
                return;
            t = set_bop_join(c, b, in, t, line);
            if (c->failed)
                return;
            mlo = sel_u32(c, b, t, const_u32(c, truev),
                          const_u32(c, 0));
        }
        def_val(c, in->dst, in->ty, pack2(c, b, mhi, mlo));
        return;
    }
