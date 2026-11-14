#include "instructions/arith/integer/integer.h"

void emit_mulwide_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2;
    uint32_t id;
        if (in->ty == PTY_U16) {
            if (check_dst(c, in->dst, PTY_U32, line) != 0)
                return;
            a = resolve(c, b, &in->src[0], PTY_U16, line);
            c2 = resolve(c, b, &in->src[1], PTY_U16, line);
            if (c->failed)
                return;
            uint32_t av = a.id, bv = c2.id;
            if (in->sign) {
                av = pack_lane_get(c, b, a.id, 0, 16, 1);
                bv = pack_lane_get(c, b, c2.id, 0, 16, 1);
            }
            id = nid(c);
            emit_arith(c, b, S_IMul, c->t_u32, id, av, bv);
            def_val(c, in->dst, PTY_U32, id);
            return;
        }
        if (check_dst(c, in->dst, PTY_U64, line) != 0)
            return;
        a = resolve(c, b, &in->src[0], PTY_U32, line);
        c2 = resolve(c, b, &in->src[1], PTY_U32, line);
        if (c->failed)
            return;
        {
            uint32_t w0 = nid(c), w1 = nid(c);
            uint32_t conv = in->sign ? S_SConvert : S_UConvert;
            put(c, b, (4u << 16) | conv);
            put(c, b, c->t_u64);
            put(c, b, w0);
            put(c, b, a.id);
            put(c, b, (4u << 16) | conv);
            put(c, b, c->t_u64);
            put(c, b, w1);
            put(c, b, c2.id);
            id = nid(c);
            emit_arith(c, b, S_IMul, c->t_u64, id, w0, w1);
            def_val(c, in->dst, PTY_U64, id);
        }
        return;
}
