#include "instructions/arith/integer/integer.h"

void emit_clmad_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2, d;
    uint32_t id;
        // Carryless multiply-add: 128-bit polynomial product built from
        // XOR-accumulated shifts of b, then XOR c.
        if (check_dst(c, in->dst, PTY_U64, line) != 0)
            return;
        a = resolve(c, b, &in->src[0], PTY_U64, line);
        c2 = resolve(c, b, &in->src[1], PTY_U64, line);
        d = resolve(c, b, &in->src[2], PTY_U64, line);
        if (c->failed)
            return;
        uint32_t lo = const_u64(c, 0), hi = const_u64(c, 0);
        for (int i = 0; i < 64; i++) {
            uint32_t bit = nid(c), m = nid(c);
            if (i == 0) {
                emit_arith(c, b, S_And, c->t_u64, bit, a.id,
                           const_u64(c, 1));
            } else {
                uint32_t sh = nid(c);
                emit_arith(c, b, S_ShrL, c->t_u64, sh, a.id,
                           const_u64(c, (uint64_t)i));
                emit_arith(c, b, S_And, c->t_u64, bit, sh,
                           const_u64(c, 1));
            }
            emit_arith(c, b, S_ISub, c->t_u64, m, const_u64(c, 0), bit);
            uint32_t bl = nid(c);
            emit_arith(c, b, S_Shl, c->t_u64, bl, c2.id,
                       const_u64(c, (uint64_t)i));
            uint32_t lm = nid(c), nlo = nid(c);
            emit_arith(c, b, S_And, c->t_u64, lm, bl, m);
            emit_arith(c, b, S_Xor, c->t_u64, nlo, lo, lm);
            lo = nlo;
            if (i > 0) {
                uint32_t bh = nid(c);
                emit_arith(c, b, S_ShrL, c->t_u64, bh, c2.id,
                           const_u64(c, (uint64_t)(64 - i)));
                uint32_t hm = nid(c), nhi = nid(c);
                emit_arith(c, b, S_And, c->t_u64, hm, bh, m);
                emit_arith(c, b, S_Xor, c->t_u64, nhi, hi, hm);
                hi = nhi;
            }
        }
        uint32_t part = in->is_hi ? hi : lo;
        id = nid(c);
        emit_arith(c, b, S_Xor, c->t_u64, id, part, d.id);
        def_val(c, in->dst, PTY_U64, id);
        return;
    }
