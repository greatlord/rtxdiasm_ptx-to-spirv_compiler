#include "instructions/video/video_ops.h"

void emit_vshr_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2;
    uint32_t id;
    if (in->ty != PTY_U32) {
        failf(c, line, "vshr supports .u32/.s32 only");
        return;
    }
    if (check_dst(c, in->dst, PTY_U32, line) != 0)
        return;
    a = resolve(c, b, &in->src[0], PTY_U32, line);
    c2 = resolve(c, b, &in->src[1], PTY_U32, line);
    if (c->failed)
        return;
    uint32_t amt;
    uint32_t inrange = 0;
    if (in->vs_clamp) {
        inrange = nid(c);
        put(c, b, (5u << 16) | S_ULess);
        put(c, b, c->t_bool);
        put(c, b, inrange);
        put(c, b, c2.id);
        put(c, b, const_u32(c, 32));
        amt = sel_u32(c, b, inrange, c2.id, const_u32(c, 31));
    } else {
        amt = nid(c);
        emit_arith(c, b, S_And, c->t_u32, amt, c2.id, const_u32(c, 31));
    }
    id = nid(c);
    emit_arith(c, b, in->sign ? S_ShrA : S_ShrL, c->t_u32, id, a.id, amt);
    if (in->vs_clamp && !in->sign)
        id = sel_u32(c, b, inrange, id, const_u32(c, 0));
    if (in->vs_add) {
        val_t c3 = resolve(c, b, &in->src[2], PTY_U32, line);
        if (c->failed)
            return;
        uint32_t sum = nid(c);
        emit_arith(c, b, S_IAdd, c->t_u32, sum, id, c3.id);
        id = sum;
    }
    def_val(c, in->dst, PTY_U32, id);
}
