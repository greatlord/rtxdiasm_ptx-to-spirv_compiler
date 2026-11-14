#include "instructions/arith/float/float.h"

void emit_copysign_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2;
    uint32_t id;
        // d = sign(a) | magnitude(b) (source order pinned by ZLUDA's
        // vector: the FIRST source supplies the sign bit).
        if (check_dst(c, in->dst, in->ty, line) != 0)
            return;
        a = resolve(c, b, &in->src[0], in->ty, line);
        c2 = resolve(c, b, &in->src[1], in->ty, line);
        if (c->failed)
            return;
        if (in->ty == PTY_F64) {
            uint32_t ab = nid(c), bb = nid(c);
            put(c, b, (4u << 16) | S_Bitcast);
            put(c, b, c->t_u64);
            put(c, b, ab);
            put(c, b, a.id);
            put(c, b, (4u << 16) | S_Bitcast);
            put(c, b, c->t_u64);
            put(c, b, bb);
            put(c, b, c2.id);
            uint32_t s = nid(c), m = nid(c), r = nid(c);
            emit_arith(c, b, S_And, c->t_u64, s, ab,
                       const_u64(c, 0x8000000000000000ull));
            emit_arith(c, b, S_And, c->t_u64, m, bb,
                       const_u64(c, 0x7fffffffffffffffull));
            emit_arith(c, b, S_Or, c->t_u64, r, s, m);
            id = nid(c);
            put(c, b, (4u << 16) | S_Bitcast);
            put(c, b, c->t_f64);
            put(c, b, id);
            put(c, b, r);
            def_val(c, in->dst, PTY_F64, id);
            return;
        }
        uint32_t ab = nid(c), bb = nid(c);
        put(c, b, (4u << 16) | S_Bitcast);
        put(c, b, c->t_u32);
        put(c, b, ab);
        put(c, b, a.id);
        put(c, b, (4u << 16) | S_Bitcast);
        put(c, b, c->t_u32);
        put(c, b, bb);
        put(c, b, c2.id);
        uint32_t s = nid(c), m = nid(c), r = nid(c);
        emit_arith(c, b, S_And, c->t_u32, s, ab, const_u32(c, 0x80000000u));
        emit_arith(c, b, S_And, c->t_u32, m, bb, const_u32(c, 0x7fffffffu));
        emit_arith(c, b, S_Or, c->t_u32, r, s, m);
        id = nid(c);
        put(c, b, (4u << 16) | S_Bitcast);
        put(c, b, c->t_f32);
        put(c, b, id);
        put(c, b, r);
        def_val(c, in->dst, PTY_F32, id);
        return;
    }
