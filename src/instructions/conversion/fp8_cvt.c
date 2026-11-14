#include "instructions/conversion/conversion.h"

void emit_fp8_to_f16x2_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a;
        // cvt.rn.f16x2.e4m3x2/e5m2x2: two FP8 bytes (low byte -> low
        // half) widen exactly to f16. Dest is a .b32 pattern register
        // (not .f32: the generic same-width bitcast allowance does not
        // make the float spelling legal for this opcode).
        reg_t *fdr = find_reg(c, in->dst);
        if (!fdr || (fdr->decl_ty != PTY_U32 &&
                     fdr->decl_ty != PTY_F16X2)) {
            failf(c, line,
                  "fp8 widen destination must be .b32/.u32/.s32");
            return;
        }
        a = resolve(c, b, &in->src[0], PTY_U16, line);
        if (c->failed)
            return;
        uint32_t b0 = nid(c), b1 = nid(c), b10 = nid(c);
        emit_arith(c, b, S_And, c->t_u32, b0, a.id, const_u32(c, 0xffu));
        emit_arith(c, b, S_ShrL, c->t_u32, b10, a.id, const_u32(c, 8));
        emit_arith(c, b, S_And, c->t_u32, b1, b10, const_u32(c, 0xffu));
        int e4m3 = (in->fp8 == 1);
        uint32_t h0 = f16_from_fp8byte(c, b, b0, e4m3, line);
        uint32_t h1 = f16_from_fp8byte(c, b, b1, e4m3, line);
        if (c->failed)
            return;
        def_val(c, in->dst, PTY_U32, pack2(c, b, h1, h0));
        return;
    }

void emit_f32_to_fp8x2_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a;
        // cvt.rn.satfinite.{e4m3x2,e5m2x2}.f32: first source -> high
        // byte, second -> low byte (hardware/ZLUDA order, verified by
        // ZLUDA's pinned vectors). Dest is a .b16 pattern register.
        if (check_dst(c, in->dst, PTY_U16, line) != 0)
            return;
        a = resolve(c, b, &in->src[0], PTY_F32, line);
        if (c->failed)
            return;
        val_t s2 = resolve(c, b, &in->src[1], PTY_F32, line);
        if (c->failed)
            return;
        int e4m3 = (in->fp8 == 1);
        uint32_t ha = fp8byte_from_f32(c, b, a.id, e4m3, line);
        uint32_t hb = fp8byte_from_f32(c, b, s2.id, e4m3, line);
        if (c->failed)
            return;
        uint32_t hi = nid(c), res = nid(c);
        emit_arith(c, b, S_Shl, c->t_u32, hi, ha, const_u32(c, 8));
        emit_arith(c, b, S_Or, c->t_u32, res, hi, hb);
        def_val(c, in->dst, PTY_U16, res);
        return;
    }
