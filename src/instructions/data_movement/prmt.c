#include "instructions/data_movement/prmt.h"

void emit_prmt_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2, d;
        // Byte permute: output byte i takes concat byte [nibble i]
        // (a low), or the sign bit of concat[nibble & 7] replicated
        // when the nibble is >= 8. All shifts structurally < 32.
        if (check_dst(c, in->dst, PTY_U32, line) != 0)
            return;
        a = resolve(c, b, &in->src[0], PTY_U32, line);
        c2 = resolve(c, b, &in->src[1], PTY_U32, line);
        d = resolve(c, b, &in->src[2], PTY_U32, line);
        if (c->failed)
            return;
        uint32_t out = const_u32(c, 0);
        for (int i = 0; i < 4; i++) {
            uint32_t sh = nid(c), s = nid(c), addr = nid(c);
            emit_arith(c, b, S_ShrL, c->t_u32, sh, d.id,
                       const_u32(c, (uint32_t)(4 * i)));
            emit_arith(c, b, S_And, c->t_u32, s, sh, const_u32(c, 0xfu));
            emit_arith(c, b, S_And, c->t_u32, addr, s, const_u32(c, 7u));
            uint32_t ka = nid(c), kb = nid(c), sa = nid(c), sb = nid(c);
            emit_arith(c, b, S_And, c->t_u32, ka, addr, const_u32(c, 3u));
            emit_arith(c, b, S_ISub, c->t_u32, sb, addr, const_u32(c, 4u));
            emit_arith(c, b, S_And, c->t_u32, kb, sb, const_u32(c, 3u));
            emit_arith(c, b, S_Shl, c->t_u32, sa, ka, const_u32(c, 3u));
            uint32_t shb = nid(c);
            emit_arith(c, b, S_Shl, c->t_u32, shb, kb, const_u32(c, 3u));
            uint32_t ba = nid(c), ta = nid(c), bb = nid(c), tb = nid(c);
            emit_arith(c, b, S_ShrL, c->t_u32, ta, a.id, sa);
            emit_arith(c, b, S_And, c->t_u32, ba, ta, const_u32(c, 0xffu));
            emit_arith(c, b, S_ShrL, c->t_u32, tb, c2.id, shb);
            emit_arith(c, b, S_And, c->t_u32, bb, tb, const_u32(c, 0xffu));
            uint32_t selA = do_cmp(c, b, PCM_LT, PTY_U32, 0, addr, const_u32(c, 4u), line);
            if (c->failed)
                return;
            uint32_t byte = sel_u32(c, b, selA, ba, bb);
            uint32_t sbit = nid(c);
            emit_arith(c, b, S_And, c->t_u32, sbit, byte, const_u32(c, 0x80u));
            uint32_t isneg = do_cmp(c, b, PCM_NE, PTY_U32, 0, sbit, const_u32(c, 0), line);
            if (c->failed)
                return;
            uint32_t rep = sel_u32(c, b, isneg, const_u32(c, 0xffu), const_u32(c, 0));
            uint32_t ob = sel_u32(c, b, do_cmp(c, b, PCM_GE, PTY_U32, 0, s, const_u32(c, 8u), line), rep, byte);
            if (c->failed)
                return;
            uint32_t shob = nid(c), nx = nid(c);
            emit_arith(c, b, S_Shl, c->t_u32, shob, ob, const_u32(c, (uint32_t)(8 * i)));
            emit_arith(c, b, S_Or, c->t_u32, nx, out, shob);
            out = nx;
        }
        def_val(c, in->dst, PTY_U32, out);
        return;
    }
