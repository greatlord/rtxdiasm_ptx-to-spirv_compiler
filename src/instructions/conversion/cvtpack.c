#include "instructions/conversion/conversion.h"

void emit_cvtpack_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
        // Packed narrowing: two f32 sources to half pairs (HIGH half
        // = first source). Optional .relu takes max(+0) per lane in
        // the f32 domain first (NaN maps to +0; corpus skips NaN).
        // Sources may be .b32 bit-holders (ld.f32 into .b32 is
        // ptxas-clean in the corpus): reinterpret those locally
        // instead of weakening the global f32/u32 register check.
        if (check_dst(c, in->dst, in->ty, line) != 0)
            return;
        uint32_t s1id, s2id;
        {
            val_t s1;
            const ptx_opd_t *o = &in->src[0];
            reg_t *r = o->kind == OPD_REG ? find_reg(c, o->reg) : NULL;
            if (r && r->decl_ty == PTY_U32) {
                val_t w = resolve(c, b, o, PTY_U32, line);
                if (c->failed)
                    return;
                uint32_t f = nid(c);
                put(c, b, (4u << 16) | S_Bitcast);
                put(c, b, c->t_f32);
                put(c, b, f);
                put(c, b, w.id);
                s1id = f;
            } else {
                s1 = resolve(c, b, o, PTY_F32, line);
                if (c->failed)
                    return;
                s1id = s1.id;
            }
        }
        {
            val_t s2;
            const ptx_opd_t *o = &in->src[1];
            reg_t *r = o->kind == OPD_REG ? find_reg(c, o->reg) : NULL;
            if (r && r->decl_ty == PTY_U32) {
                val_t w = resolve(c, b, o, PTY_U32, line);
                if (c->failed)
                    return;
                uint32_t f = nid(c);
                put(c, b, (4u << 16) | S_Bitcast);
                put(c, b, c->t_f32);
                put(c, b, f);
                put(c, b, w.id);
                s2id = f;
            } else {
                s2 = resolve(c, b, o, PTY_F32, line);
                if (c->failed)
                    return;
                s2id = s2.id;
            }
        }
        int is_bf16 = in->ty == PTY_BF16X2;
        uint32_t v1 = s1id, v2 = s2id;
        if (in->relu) {
            // max(+0) per lane as ordered-gt selects (NaN takes +0).
            uint32_t c1 = do_cmp(c, b, PCM_GT, PTY_F32, 0, v1,
                                 const_f32(c, 0x00000000u), line);
            if (c->failed)
                return;
            uint32_t r1 = nid(c);
            put(c, b, (6u << 16) | S_Select);
            put(c, b, c->t_f32);
            put(c, b, r1);
            put(c, b, c1);
            put(c, b, v1);
            put(c, b, const_f32(c, 0x00000000u));
            v1 = r1;
            uint32_t c2 = do_cmp(c, b, PCM_GT, PTY_F32, 0, v2,
                                 const_f32(c, 0x00000000u), line);
            if (c->failed)
                return;
            uint32_t r2 = nid(c);
            put(c, b, (6u << 16) | S_Select);
            put(c, b, c->t_f32);
            put(c, b, r2);
            put(c, b, c2);
            put(c, b, v2);
            put(c, b, const_f32(c, 0x00000000u));
            v2 = r2;
        }
        uint32_t h1 = is_bf16 ? f32_to_bf16bits(c, b, v1, line)
                              : f32_to_f16bits(c, b, v1, line);
        if (c->failed)
            return;
        uint32_t h2 = is_bf16 ? f32_to_bf16bits(c, b, v2, line)
                              : f32_to_f16bits(c, b, v2, line);
        if (c->failed)
            return;
        if (in->sat && !is_bf16) {
            h1 = f16_satfinite_bits(c, b, h1, line);
            h2 = f16_satfinite_bits(c, b, h2, line);
            if (c->failed)
                return;
        }
        def_val(c, in->dst, in->ty, pack2(c, b, h1, h2));
        return;
    }
