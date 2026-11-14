#include "instructions/conversion/conversion.h"

void emit_cvtpack8_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2, d;
    uint32_t id;
        // cvt.pack.sat.<ty>.s32[.b32] d, a, b[, c]
        // Lane width is 16 with no c (d = td) or 2/4/8 with c, where
        // the fat part of d comes from c: d = (c << 2*size) |
        // (sat(a) << size) | sat(b).  NVIDIA order: first source in
        // the high packed lane, second in the low one.
        int bits = in->packbits ? in->packbits : 8;
        if (check_dst(c, in->dst, PTY_U32, line) != 0)
            return;
        a = resolve(c, b, &in->src[0], PTY_U32, line);
        c2 = resolve(c, b, &in->src[1], PTY_U32, line);
        if (c->failed)
            return;
        uint32_t ta = cvt_pack_lane(c, b, a.id, in->sign, bits, line);
        uint32_t tb = cvt_pack_lane(c, b, c2.id, in->sign, bits, line);
        if (c->failed)
            return;
        uint32_t ash = nid(c);
        emit_arith(c, b, S_Shl, c->t_u32, ash, ta, const_u32(c, bits));
        uint32_t low = nid(c);
        emit_arith(c, b, S_Or, c->t_u32, low, ash, tb);
        if (bits == 16) {
            def_val(c, in->dst, PTY_U32, low);
            return;
        }
        d = resolve(c, b, &in->src[2], PTY_U32, line);
        if (c->failed)
            return;
        uint32_t high = nid(c);
        emit_arith(c, b, S_Shl, c->t_u32, high, d.id,
                   const_u32(c, 2 * bits));
        id = nid(c);
        emit_arith(c, b, S_Or, c->t_u32, id, high, low);
        def_val(c, in->dst, PTY_U32, id);
        return;
    }
