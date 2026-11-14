#include "instructions/arith/integer/integer.h"

void emit_sad_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2, d;
    uint32_t id;
        // Scalar sum of absolute differences: d = |a-b| + c. Operands
        // are widened exactly to u64 (sign- or zero-extended), so the
        // magnitude and final sum never overflow before truncation.
        if (check_dst(c, in->dst, in->ty, line) != 0)
            return;
        a = resolve(c, b, &in->src[0], in->ty, line);
        c2 = resolve(c, b, &in->src[1], in->ty, line);
        d = resolve(c, b, &in->src[2], in->ty, line);
        if (c->failed)
            return;
        uint32_t wa, wb, wc;
        if (in->ty == PTY_U64) {
            wa = a.id;
            wb = c2.id;
            wc = d.id;
        } else {
            uint32_t av = a.id, bv = c2.id;
            if (in->ty == PTY_U16 && in->sign) {
                av = pack_lane_get(c, b, a.id, 0, 16, 1);
                bv = pack_lane_get(c, b, c2.id, 0, 16, 1);
            }
            uint32_t conv = in->sign ? S_SConvert : S_UConvert;
            wa = nid(c);
            wb = nid(c);
            wc = nid(c);
            put(c, b, (4u << 16) | conv);
            put(c, b, c->t_u64);
            put(c, b, wa);
            put(c, b, av);
            put(c, b, (4u << 16) | conv);
            put(c, b, c->t_u64);
            put(c, b, wb);
            put(c, b, bv);
            put(c, b, (4u << 16) | S_UConvert);
            put(c, b, c->t_u64);
            put(c, b, wc);
            put(c, b, d.id);
        }
        uint32_t lt = do_cmp(c, b, PCM_LT, PTY_U64, in->sign, wa, wb,
                             line);
        if (c->failed)
            return;
        uint32_t d1 = nid(c), d2 = nid(c);
        emit_arith(c, b, S_ISub, c->t_u64, d1, wa, wb);
        emit_arith(c, b, S_ISub, c->t_u64, d2, wb, wa);
        uint32_t m = sel_u64(c, b, lt, d2, d1);
        id = nid(c);
        emit_arith(c, b, S_IAdd, c->t_u64, id, m, wc);
        if (in->ty != PTY_U64) {
            uint32_t tr = nid(c);
            put(c, b, (4u << 16) | S_UConvert);
            put(c, b, c->t_u32);
            put(c, b, tr);
            put(c, b, id);
            id = tr;
            if (in->ty == PTY_U16) {
                uint32_t mm = nid(c);
                emit_arith(c, b, S_And, c->t_u32, mm, id,
                           const_u32(c, 0xffffu));
                id = mm;
            }
        }
        def_val(c, in->dst, in->ty, id);
        return;
    }
