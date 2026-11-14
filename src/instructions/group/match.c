#include "instructions/group/group.h"

void emit_match_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    uint32_t id;
        // Match-any over the 32-lane warp: for each lane i, shuffle
        // lane i's value to me, compare with mine, set bit i.
        // Unrolled straight line (32 shuffles); converged mask
        // members required, as in CUDA. Result is narrowed to the
        // membermask (bits outside mask cleared), matching
        // __match_any_sync semantics; full-mask callers see the raw
        // equality mask.
        if (check_dst(c, in->dst, in->ty, line) != 0)
            return;
        val_t av = resolve(c, b, &in->src[0], in->ty, line);
        if (c->failed)
            return;
        val_t mv = resolve(c, b, &in->src[1], PTY_U32, line);
        if (c->failed)
            return;
        uint32_t mine = av.id;
        if (in->ty == PTY_F32) {
            uint32_t w = nid(c);
            put(c, b, (4u << 16) | S_Bitcast);
            put(c, b, c->t_u32);
            put(c, b, w);
            put(c, b, av.id);
            mine = w;
        }
        uint32_t sub = nid(c);
        emit_load(c, b, c->t_u32, sub, c->v_subid);
        uint32_t W = nid(c);
        emit_arith(c, b, S_ShrL, c->t_u32, W, sub, const_u32(c, 5));
        uint32_t scope = const_u32(c, 3); // Subgroup
        uint32_t acc = const_u32(c, 0);
        for (uint32_t i = 0; i < 32; i++) {
            // src_sub = (W << 5) | i: lane i's value to me.
            uint32_t wb = nid(c), src = nid(c);
            emit_arith(c, b, S_Shl, c->t_u32, wb, W, const_u32(c, 5));
            emit_arith(c, b, S_Or, c->t_u32, src, wb, const_u32(c, i));
            uint32_t got = nid(c);
            put(c, b, (6u << 16) | S_GroupShuffle);
            put(c, b, c->t_u32);
            put(c, b, got);
            put(c, b, scope);
            put(c, b, mine);
            put(c, b, src);
            uint32_t eq = nid(c);
            put(c, b, (5u << 16) | S_IEqual);
            put(c, b, c->t_bool);
            put(c, b, eq);
            put(c, b, got);
            put(c, b, mine);
            uint32_t bit = nid(c);
            put(c, b, (6u << 16) | S_Select);
            put(c, b, c->t_u32);
            put(c, b, bit);
            put(c, b, eq);
            put(c, b, const_u32(c, 1u << i));
            put(c, b, const_u32(c, 0));
            uint32_t na = nid(c);
            emit_arith(c, b, S_Or, c->t_u32, na, acc, bit);
            acc = na;
        }
        {
            uint32_t narrowed = nid(c);
            emit_arith(c, b, S_And, c->t_u32, narrowed, acc, mv.id);
            acc = narrowed;
        }
        if (in->ty == PTY_F32) {
            // Float-typed id required downstream (bits preserved).
            id = nid(c);
            put(c, b, (4u << 16) | S_Bitcast);
            put(c, b, c->t_f32);
            put(c, b, id);
            put(c, b, acc);
            def_val(c, in->dst, PTY_F32, id);
        } else {
            def_val(c, in->dst, PTY_U32, acc);
        }
        return;
    }
