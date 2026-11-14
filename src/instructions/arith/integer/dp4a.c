#include "instructions/arith/integer/integer.h"

void emit_dp4a_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2, d;
        if (check_dst(c, in->dst, PTY_U32, line) != 0)
            return;
        a = resolve(c, b, &in->src[0], PTY_U32, line);
        c2 = resolve(c, b, &in->src[1], PTY_U32, line);
        d = resolve(c, b, &in->src[2], PTY_U32, line);
        if (c->failed)
            return;
        int dot_form_ok =
            in->sign && in->arg ? c->caps.dot_signed
            : !in->sign && !in->arg ? c->caps.dot_unsigned
                                    : c->caps.dot_mixed;
        if (c->use_dot_product && dot_form_ok) {
            // Native packed 4x8-bit dot product: OpSDot/OpUDot/OpSUDot
            // interprets each 32-bit operand as four 8-bit lanes (the
            // PackedVectorFormat operand). SUDot is signed-first, so a
            // u8 x s8 mix swaps the operands. The ACuda `acuda.dp4a.sat`
            // form uses the AccSat variants, which saturate the final
            // accumulation.
            uint32_t opc, acc_opc, x = a.id, y = c2.id;
            if (in->sign && in->arg) {
                opc = S_SDot;
                acc_opc = S_SDotAccSat;
            } else if (!in->sign && !in->arg) {
                opc = S_UDot;
                acc_opc = S_UDotAccSat;
            } else if (in->sign) {
                opc = S_SUDot; // a signed, b unsigned
                acc_opc = S_SUDotAccSat;
            } else {
                opc = S_SUDot; // a unsigned, b signed: swap
                acc_opc = S_SUDotAccSat;
                x = c2.id;
                y = a.id;
            }
            if (in->sat) {
                uint32_t r = nid(c);
                put(c, b, (7u << 16) | acc_opc);
                put(c, b, c->t_u32);
                put(c, b, r);
                put(c, b, x);
                put(c, b, y);
                put(c, b, d.id);
                put(c, b, 0u); // PackedVectorFormat4x8Bit
                def_val(c, in->dst, PTY_U32, r);
                return;
            }
            uint32_t dot = nid(c);
            put(c, b, (6u << 16) | opc);
            put(c, b, c->t_u32);
            put(c, b, dot);
            put(c, b, x);
            put(c, b, y);
            put(c, b, 0u); // PackedVectorFormat4x8Bit
            uint32_t sum = nid(c);
            emit_arith(c, b, S_IAdd, c->t_u32, sum, d.id, dot);
            def_val(c, in->dst, PTY_U32, sum);
            return;
        }
        if (!in->sat) {
            uint32_t acc = d.id;
            for (int i = 0; i < 4; i++) {
                uint32_t av = pack_lane_get(c, b, a.id, i, 8, in->sign);
                uint32_t bv = pack_lane_get(c, b, c2.id, i, 8, in->arg);
                uint32_t pr = nid(c), nx = nid(c);
                emit_arith(c, b, S_IMul, c->t_u32, pr, av, bv);
                emit_arith(c, b, S_IAdd, c->t_u32, nx, acc, pr);
                acc = nx;
            }
            def_val(c, in->dst, PTY_U32, acc);
            return;
        }
        // Saturating fallback: exact 64-bit accumulation, then clamp to
        // the result width (the products themselves cannot overflow 64
        // bits). Mixed-signedness uses the signed range, matching
        // OpSUDotAccSat.
        int uform = !in->sign && !in->arg;
        uint32_t acc = nid(c);
        put(c, b, (4u << 16) | (uform ? S_UConvert : S_SConvert));
        put(c, b, c->t_u64);
        put(c, b, acc);
        put(c, b, d.id);
        for (int i = 0; i < 4; i++) {
            uint32_t av = pack_lane_get(c, b, a.id, i, 8, in->sign);
            uint32_t bv = pack_lane_get(c, b, c2.id, i, 8, in->arg);
            uint32_t av64 = nid(c), bv64 = nid(c);
            put(c, b, (4u << 16) | (in->sign ? S_SConvert : S_UConvert));
            put(c, b, c->t_u64);
            put(c, b, av64);
            put(c, b, av);
            put(c, b, (4u << 16) | (in->arg ? S_SConvert : S_UConvert));
            put(c, b, c->t_u64);
            put(c, b, bv64);
            put(c, b, bv);
            uint32_t pr = nid(c), nx = nid(c);
            emit_arith(c, b, S_IMul, c->t_u64, pr, av64, bv64);
            emit_arith(c, b, S_IAdd, c->t_u64, nx, acc, pr);
            acc = nx;
        }
        uint32_t clamped;
        if (uform) {
            uint32_t gt = do_cmp(c, b, PCM_GT, PTY_U64, 1, acc,
                                 const_u64(c, 0xffffffffull), line);
            clamped = nid(c);
            put(c, b, (6u << 16) | S_Select);
            put(c, b, c->t_u64);
            put(c, b, clamped);
            put(c, b, gt);
            put(c, b, const_u64(c, 0xffffffffull));
            put(c, b, acc);
        } else {
            uint32_t gt = do_cmp(c, b, PCM_GT, PTY_U64, 1, acc,
                                 const_u64(c, 0x7fffffffull), line);
            uint32_t hi = nid(c);
            put(c, b, (6u << 16) | S_Select);
            put(c, b, c->t_u64);
            put(c, b, hi);
            put(c, b, gt);
            put(c, b, const_u64(c, 0x7fffffffull));
            put(c, b, acc);
            uint32_t lt = do_cmp(c, b, PCM_LT, PTY_U64, 1, hi,
                                 const_u64(c, 0xffffffff80000000ull), line);
            clamped = nid(c);
            put(c, b, (6u << 16) | S_Select);
            put(c, b, c->t_u64);
            put(c, b, clamped);
            put(c, b, lt);
            put(c, b, const_u64(c, 0xffffffff80000000ull));
            put(c, b, hi);
        }
        uint32_t res = nid(c);
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, c->t_u32);
        put(c, b, res);
        put(c, b, clamped);
        def_val(c, in->dst, PTY_U32, res);
        return;
    }
