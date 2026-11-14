#include "instructions/arith/arith.h"

void emit_div_rem_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2;
    uint32_t id;
        // Unsigned and float division are single instructions. Signed
        // div/rem lower through the unsigned core (abs, divide, fix
        // signs): Vulkan leaves negative-operand SRem/SMod undefined,
        // while this sequence is exact two's-complement C semantics
        // (truncation toward zero, remainder follows the dividend) on
        // every driver. Divide-by-zero stays undefined, as in PTX.
        int is_rem = in->op == POP_REM;
        if (check_dst(c, in->dst, in->ty, line) != 0)
            return;
        a = resolve(c, b, &in->src[0], in->ty, line);
        c2 = resolve(c, b, &in->src[1], in->ty, line);
        if (c->failed)
            return;
        ptx_ty_t et = eff_ty(c, in->ty);
        uint32_t wt = et == PTY_U64 ? c->t_u64
                      : et == PTY_F64 ? c->t_f64
                      : et == PTY_F32 ? c->t_f32
                                      : c->t_u32;
        if (et == PTY_U16 && in->sign) {
            a.id = pack_lane_get(c, b, a.id, 0, 16, 1);
            c2.id = pack_lane_get(c, b, c2.id, 0, 16, 1);
        }
        if (et == PTY_F16) {
            // f16 division (ptxas has no such form; accepted as a
            // widening extension): divide the exact f32 values and
            // round once to half. f16x2 is the same per packed lane.
            if (is_rem) {
                failf(c, line, "float rem does not exist in PTX");
                return;
            }
            uint32_t fa = f16_to_f32val(c, b, a.id, line);
            uint32_t fb = f16_to_f32val(c, b, c2.id, line);
            if (c->failed)
                return;
            uint32_t q = nid(c);
            emit_arith(c, b, S_FDiv, c->t_f32, q, fa, fb);
            id = f32_to_f16bits(c, b, q, line);
            if (c->failed)
                return;
            def_val(c, in->dst, PTY_F16, id);
            return;
        }
        if (et == PTY_F16X2) {
            if (is_rem) {
                failf(c, line, "float rem does not exist in PTX");
                return;
            }
            uint32_t ahi = nid(c), alo = nid(c), bhi = nid(c), blo = nid(c);
            emit_arith(c, b, S_ShrL, c->t_u32, ahi, a.id,
                       const_u32(c, 16));
            emit_arith(c, b, S_And, c->t_u32, alo, a.id,
                       const_u32(c, 0xffffu));
            emit_arith(c, b, S_ShrL, c->t_u32, bhi, c2.id,
                       const_u32(c, 16));
            emit_arith(c, b, S_And, c->t_u32, blo, c2.id,
                       const_u32(c, 0xffffu));
            uint32_t fah = f16_to_f32val(c, b, ahi, line);
            uint32_t fal = f16_to_f32val(c, b, alo, line);
            uint32_t fbh = f16_to_f32val(c, b, bhi, line);
            uint32_t fbl = f16_to_f32val(c, b, blo, line);
            if (c->failed)
                return;
            uint32_t qh = nid(c), ql = nid(c);
            emit_arith(c, b, S_FDiv, c->t_f32, qh, fah, fbh);
            emit_arith(c, b, S_FDiv, c->t_f32, ql, fal, fbl);
            uint32_t rh = f32_to_f16bits(c, b, qh, line);
            uint32_t rl = f32_to_f16bits(c, b, ql, line);
            if (c->failed)
                return;
            id = pack2(c, b, rh, rl);
            def_val(c, in->dst, PTY_F16X2, id);
            return;
        }
        if (et == PTY_F32 || et == PTY_F64) {
            if (is_rem) {
                failf(c, line, "float rem does not exist in PTX");
                return;
            }
            uint32_t da = a.id, db = c2.id;
            if (in->ftz && et == PTY_F32) {
                // Flush-to-zero (parse guarantees f32 here).
                da = flush_f32_denorm(c, b, da, line);
                if (c->failed)
                    return;
                db = flush_f32_denorm(c, b, db, line);
                if (c->failed)
                    return;
            }
            if (!in->approx_div && !(et == PTY_F64 && c->emu_f64)) {
                if (et == PTY_F64) {
                    id = f64_directed_div_exact(c, b, da, db, in->frnd,
                                                line);
                } else {
                    id = f32_directed_div_exact(c, b, da, db, in->frnd,
                                                line);
                }
                if (c->failed)
                    return;
            } else {
                id = nid(c);
                emit_arith(c, b, S_FDiv, wt, id, da, db);
            }
            if (in->ftz && et == PTY_F32) {
                id = flush_f32_denorm(c, b, id, line);
                if (c->failed)
                    return;
            }
            def_val(c, in->dst, in->ty, id);
            return;
        }
        if (!in->sign) {
            id = nid(c);
            emit_arith(c, b, is_rem ? S_UMod : S_UDiv, wt, id, a.id,
                       c2.id);
            if (et == PTY_U16) {
                uint32_t m = nid(c);
                emit_arith(c, b, S_And, c->t_u32, m, id,
                           const_u32(c, 0xffffu));
                id = m;
            }
            def_val(c, in->dst, in->ty, id);
            return;
        }
        uint32_t zero = et == PTY_U64 ? const_u64(c, 0) : const_u32(c, 0);
        uint32_t sx = nid(c), sy = nid(c);
        put(c, b, (5u << 16) | S_SLess);
        put(c, b, c->t_bool);
        put(c, b, sx);
        put(c, b, a.id);
        put(c, b, zero);
        put(c, b, (5u << 16) | S_SLess);
        put(c, b, c->t_bool);
        put(c, b, sy);
        put(c, b, c2.id);
        put(c, b, zero);
        uint32_t na = nid(c), nbd = nid(c);
        put(c, b, (4u << 16) | S_SNegate);
        put(c, b, wt);
        put(c, b, na);
        put(c, b, a.id);
        put(c, b, (4u << 16) | S_SNegate);
        put(c, b, wt);
        put(c, b, nbd);
        put(c, b, c2.id);
        uint32_t ax = nid(c), bx = nid(c);
        put(c, b, (6u << 16) | S_Select);
        put(c, b, wt);
        put(c, b, ax);
        put(c, b, sx);
        put(c, b, na);
        put(c, b, a.id);
        put(c, b, (6u << 16) | S_Select);
        put(c, b, wt);
        put(c, b, bx);
        put(c, b, sy);
        put(c, b, nbd);
        put(c, b, c2.id);
        uint32_t m = nid(c);
        emit_arith(c, b, is_rem ? S_UMod : S_UDiv, wt, m, ax, bx);
        uint32_t nm = nid(c);
        put(c, b, (4u << 16) | S_SNegate);
        put(c, b, wt);
        put(c, b, nm);
        put(c, b, m);
        id = nid(c);
        if (is_rem) {
            put(c, b, (6u << 16) | S_Select);
            put(c, b, wt);
            put(c, b, id);
            put(c, b, sx);
            put(c, b, nm);
            put(c, b, m);
        } else {
            // Quotient sign = sx XOR sy, via select (no bool-xor op).
            uint32_t nsy = nid(c);
            put(c, b, (4u << 16) | S_LogicalNot);
            put(c, b, c->t_bool);
            put(c, b, nsy);
            put(c, b, sy);
            uint32_t s = nid(c);
            put(c, b, (6u << 16) | S_Select);
            put(c, b, c->t_bool);
            put(c, b, s);
            put(c, b, sx);
            put(c, b, nsy);
            put(c, b, sy);
            put(c, b, (6u << 16) | S_Select);
            put(c, b, wt);
            put(c, b, id);
            put(c, b, s);
            put(c, b, nm);
            put(c, b, m);
        }
        if (et == PTY_U16) {
            uint32_t m = nid(c);
            emit_arith(c, b, S_And, c->t_u32, m, id,
                       const_u32(c, 0xffffu));
            id = m;
        }
        def_val(c, in->dst, in->ty, id);
        return;
    }
