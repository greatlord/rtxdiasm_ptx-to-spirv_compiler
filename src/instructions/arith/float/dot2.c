#include "instructions/arith/float/float.h"

void emit_dot2_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
        // ACuda extension: d = c + lo(a)*lo(b) + hi(a)*hi(b), where a and
        // b are f16x2 words and c/d are f32. Uses the VALVE mixed dot
        // instruction when available, otherwise widens and issues FMAs.
        if (check_dst(c, in->dst, PTY_F32, line) != 0)
            return;
        val_t av = resolve(c, b, &in->src[0], PTY_U32, line);
        val_t bv = resolve(c, b, &in->src[1], PTY_U32, line);
        val_t cv = resolve(c, b, &in->src[2], PTY_F32, line);
        if (c->failed)
            return;
        if (c->uses_fdot2) {
            uint32_t r = emit_fdot2_f32(c, b, av.id, bv.id, cv.id);
            if (c->failed)
                return;
            def_val(c, in->dst, PTY_F32, r);
            return;
        }
        uint32_t alo = nid(c), ahi = nid(c);
        uint32_t blo = nid(c), bhi = nid(c);
        emit_arith(c, b, S_And, c->t_u32, alo, av.id, const_u32(c, 0xffffu));
        emit_arith(c, b, S_ShrL, c->t_u32, ahi, av.id, const_u32(c, 16));
        emit_arith(c, b, S_And, c->t_u32, blo, bv.id, const_u32(c, 0xffffu));
        emit_arith(c, b, S_ShrL, c->t_u32, bhi, bv.id, const_u32(c, 16));
        uint32_t fa0 = f16_to_f32val(c, b, alo, line);
        uint32_t fa1 = f16_to_f32val(c, b, ahi, line);
        uint32_t fb0 = f16_to_f32val(c, b, blo, line);
        uint32_t fb1 = f16_to_f32val(c, b, bhi, line);
        if (c->failed)
            return;
        uint32_t t = nid(c);
        emit_fma_into(c, b, c->t_f32, t, fa0, fb0, cv.id);
        uint32_t r = nid(c);
        emit_fma_into(c, b, c->t_f32, r, fa1, fb1, t);
        def_val(c, in->dst, PTY_F32, r);
        return;
    }
