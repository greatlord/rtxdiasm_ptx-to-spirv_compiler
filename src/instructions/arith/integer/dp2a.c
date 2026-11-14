#include "instructions/arith/integer/integer.h"

void emit_dp2a_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2, d;
        // Two-way dot: 16-bit lanes from a, selected 8-bit lane pair
        // from b; each operand independently signed/unsigned.
        if (check_dst(c, in->dst, PTY_U32, line) != 0)
            return;
        a = resolve(c, b, &in->src[0], PTY_U32, line);
        c2 = resolve(c, b, &in->src[1], PTY_U32, line);
        d = resolve(c, b, &in->src[2], PTY_U32, line);
        if (c->failed)
            return;
        // A DotProductInputAll <2 x i16> experiment (bitcast a, extend
        // and pack b's selected bytes, OpSDot + IAdd) was measured on
        // RADV: it cut static ISA instructions from 320 to 181 but was
        // runtime-neutral (~1.0x) and required shaderInt16/Int16, so
        // the two-multiply expansion stays.
        int boff = in->is_hi ? 2 : 0;
        uint32_t acc = d.id;
        for (int i = 0; i < 2; i++) {
            uint32_t av = pack_lane_get(c, b, a.id, i, 16, in->sign);
            uint32_t bv = pack_lane_get(c, b, c2.id, boff + i, 8,
                                        in->arg);
            uint32_t pr = nid(c), nx = nid(c);
            emit_arith(c, b, S_IMul, c->t_u32, pr, av, bv);
            emit_arith(c, b, S_IAdd, c->t_u32, nx, acc, pr);
            acc = nx;
        }
        def_val(c, in->dst, PTY_U32, acc);
        return;
    }
