#include "instructions/conversion/conversion.h"

void emit_rnd_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a;
    uint32_t id;
        // Round to integer-valued float (rni/rzi/rmi/rpi). f32 uses
        // exact integer bit manipulation; f64 uses GLSL RoundEven/
        // Trunc/Floor/Ceil.
        if (in->ty == PTY_F64) {
            if (check_dst(c, in->dst, PTY_F64, line) != 0)
                return;
            a = resolve(c, b, &in->src[0], PTY_F64, line);
            if (c->failed)
                return;
            id = round_f64_directed(c, b, a.id, in->rnd, line);
            if (c->failed)
                return;
            def_val(c, in->dst, PTY_F64, id);
            return;
        }
        if (check_dst(c, in->dst, PTY_F32, line) != 0)
            return;
        a = resolve_f32_bits(c, b, &in->src[0], line);
        if (c->failed)
            return;
        uint32_t xf = a.id;
        if (in->ftz) {
            xf = flush_f32_denorm(c, b, xf, line);
            if (c->failed)
                return;
        }
        id = round_f32_directed(c, b, xf, in->rnd, line);
        if (c->failed)
            return;
        def_val(c, in->dst, PTY_F32, id);
        return;
    }
