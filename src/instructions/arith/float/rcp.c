#include "instructions/arith/float/float.h"

void emit_rcp_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a;
    uint32_t id;
    // Reciprocal. f32 .approx uses OpFDiv; .rn and all f64 modes use
    // the exact integer division (the hardware fdiv is not correctly
    // rounded on some drivers).
    if (in->ty != PTY_F32 && in->ty != PTY_F64) {
        failf(c, line, "rcp supports .f32/.f64 only");
        return;
    }
    if (check_dst(c, in->dst, in->ty, line) != 0)
        return;
    a = resolve(c, b, &in->src[0], in->ty, line);
    if (c->failed)
        return;
    int f64 = in->ty == PTY_F64;
    uint32_t v = a.id;
    if (!f64 && in->ftz) {
        v = flush_f32_denorm(c, b, v, line);
        if (c->failed)
            return;
    }
    uint32_t one = f64 ? 0 : const_f32(c, 0x3f800000u);
    if (f64) {
        uint32_t ob = const_u64(c, 0x3ff0000000000000ull);
        one = nid(c);
        put(c, b, (4u << 16) | S_Bitcast);
        put(c, b, c->t_f64);
        put(c, b, one);
        put(c, b, ob);
    }
    int dmode = in->rnd >= 1 ? in->rnd - 1 : 0; // rn/rz/rm/rp
    if (f64) {
        id = f64_directed_div_exact(c, b, one, v, dmode, line);
        if (c->failed)
            return;
    } else if (in->rnd >= 1) {
        id = f32_directed_div_exact(c, b, one, v, dmode, line);
        if (c->failed)
            return;
    } else {
        id = nid(c);
        emit_arith(c, b, S_FDiv, c->t_f32, id, one, v);
    }
    if (!f64 && in->ftz) {
        id = flush_f32_denorm(c, b, id, line);
        if (c->failed)
            return;
    }
    def_val(c, in->dst, in->ty, id);
}
