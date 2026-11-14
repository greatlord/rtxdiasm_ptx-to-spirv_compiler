#include "instructions/arith/integer/integer.h"

void emit_mulhi_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2;
    uint32_t id;
        // High half of the full product (16/32/64-bit, signed or
        // unsigned); see mul_hi_value.
        if (check_dst(c, in->dst, in->ty, line) != 0)
            return;
        a = resolve(c, b, &in->src[0], in->ty, line);
        c2 = resolve(c, b, &in->src[1], in->ty, line);
        if (c->failed)
            return;
        id = mul_hi_value(c, b, in->ty, in->sign, a.id, c2.id, line);
        if (c->failed)
            return;
        def_val(c, in->dst, in->ty, id);
        return;
    }
