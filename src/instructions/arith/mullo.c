#include "instructions/arith/arith.h"

void emit_mullo_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2;
    uint32_t id;
        if (check_dst(c, in->dst, in->ty, line) != 0)
            return;
        a = resolve(c, b, &in->src[0], in->ty, line);
        c2 = resolve(c, b, &in->src[1], in->ty, line);
        if (c->failed)
            return;
        id = nid(c);
        if (in->f32x2) {
            uint32_t alo, ahi, blo, bhi;
            f32x2_split(c, b, a.id, &alo, &ahi);
            f32x2_split(c, b, c2.id, &blo, &bhi);
            if (c->failed)
                return;
            uint32_t rlo = f32_mul_lane(c, b, alo, blo, in->frnd, in->ftz,
                                        in->sat, line);
            uint32_t rhi = f32_mul_lane(c, b, ahi, bhi, in->frnd, in->ftz,
                                        in->sat, line);
            if (c->failed)
                return;
            def_val(c, in->dst, PTY_U64,
                    f32x2_pack(c, b, rlo, rhi));
            return;
        }
        ptx_ty_t met = eff_ty(c, in->ty);
        if (met == PTY_U8 || met == PTY_U16) {
            // Low 8/16 bits of the product (inputs are zero-extended
            // canonical values, so IMul+mask is exact).
            uint32_t pr = nid(c);
            emit_arith(c, b, S_IMul, c->t_u32, pr, a.id, c2.id);
            id = nid(c);
            emit_arith(c, b, S_And, c->t_u32, id, pr,
                       const_u32(c, met == PTY_U8 ? 0xffu : 0xffffu));
            def_val(c, in->dst, in->ty, id);
            return;
        }
        if (met == PTY_F16 || met == PTY_BF16) {
            int is_bf16 = met == PTY_BF16;
            id = half_arith_lane(c, b, is_bf16, S_FMul, in->ftz, in->sat,
                                 a.id, c2.id, line);
            if (c->failed)
                return;
            def_val(c, in->dst, in->ty, id);
            return;
        }
        if (met == PTY_F16X2 || met == PTY_BF16X2) {
            int is_bf16 = met == PTY_BF16X2;
            uint32_t ahi, alo, bhi, blo;
            split_pack(c, b, a.id, &ahi, &alo);
            split_pack(c, b, c2.id, &bhi, &blo);
            uint32_t rhi = half_arith_lane(c, b, is_bf16, S_FMul, in->ftz,
                                           in->sat, ahi, bhi, line);
            uint32_t rlo = half_arith_lane(c, b, is_bf16, S_FMul, in->ftz,
                                           in->sat, alo, blo, line);
            if (c->failed)
                return;
            def_val(c, in->dst, in->ty, pack2(c, b, rhi, rlo));
            return;
        }
        int mfloat = met == PTY_F32 || met == PTY_F64;
        uint32_t ma = a.id, mb = c2.id;
        if (in->ftz && met == PTY_F32) {
            // Flush-to-zero (parse guarantees f32 here).
            ma = flush_f32_denorm(c, b, ma, line);
            if (c->failed)
                return;
            mb = flush_f32_denorm(c, b, mb, line);
            if (c->failed)
                return;
        }
        if (met == PTY_F32 && in->frnd) {
            id = f32_directed_mul_soft(c, b, ma, mb, in->frnd, line);
            if (c->failed)
                return;
        } else if (met == PTY_F64 && in->frnd) {
            id = f64_directed_mul_soft(c, b, ma, mb, in->frnd, line);
            if (c->failed)
                return;
        } else {
            emit_arith(c, b, mfloat ? S_FMul : S_IMul,
                       met == PTY_F32 ? c->t_f32
                       : met == PTY_F64 ? c->t_f64
                       : in->ty == PTY_U64 ? c->t_u64
                       : c->t_u32,
                       id, ma, mb);
        }
        if (in->ftz && met == PTY_F32) {
            id = flush_f32_denorm(c, b, id, line);
            if (c->failed)
                return;
        }
        if (in->sat && met == PTY_F32) {
            id = sat_f32(c, b, id, line);
            if (c->failed)
                return;
        }
        def_val(c, in->dst, in->ty, id);
        return;
}
