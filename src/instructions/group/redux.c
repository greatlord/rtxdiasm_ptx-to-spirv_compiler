#include "instructions/group/group.h"

void emit_redux_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
        // Warp-clustered reduction over 32 lanes (ClusteredReduce +
        // ClusterSize 32 = warp segments on any subgroup size).
        // Mask-excluded lanes contribute the op identity (add 0, min
        // +inf, max -inf), so partial masks reduce correctly;
        // predicated redux desugars via selects (old value kept on
        // the false arm). Converged mask members required, as in CUDA.
        if (check_dst(c, in->dst, PTY_U32, line) != 0)
            return;
        val_t av = resolve(c, b, &in->src[0], PTY_U32, line);
        if (c->failed)
            return;
        val_t mv = resolve(c, b, &in->src[1], PTY_U32, line);
        if (c->failed)
            return;
        uint32_t sub = nid(c);
        emit_load(c, b, c->t_u32, sub, c->v_subid);
        uint32_t L = nid(c);
        emit_arith(c, b, S_And, c->t_u32, L, sub, const_u32(c, 31));
        uint32_t sh = nid(c), bt = nid(c), inb = nid(c);
        emit_arith(c, b, S_ShrL, c->t_u32, sh, mv.id, L);
        emit_arith(c, b, S_And, c->t_u32, bt, sh, const_u32(c, 1));
        put(c, b, (5u << 16) | S_INotEqual);
        put(c, b, c->t_bool);
        put(c, b, inb);
        put(c, b, bt);
        put(c, b, const_u32(c, 0));
        uint32_t ident;
        uint32_t op;
        if (in->redux_op == 0) {
            op = S_GroupIAdd;
            ident = const_u32(c, 0);
        } else if (in->redux_op == 1) {
            op = in->sign ? S_GroupSMin : S_GroupUMin;
            ident = const_u32(c, in->sign ? 0x7fffffffu : 0xffffffffu);
        } else if (in->redux_op == 2) {
            op = in->sign ? S_GroupSMax : S_GroupUMax;
            ident = const_u32(c, in->sign ? 0x80000000u : 0u);
        } else if (in->redux_op == 3) {
            op = S_GroupBitwiseAnd;
            ident = const_u32(c, 0xffffffffu);
        } else {
            op = in->redux_op == 4 ? S_GroupBitwiseOr
                                   : S_GroupBitwiseXor;
            ident = const_u32(c, 0);
        }
        uint32_t val = nid(c);
        put(c, b, (6u << 16) | S_Select);
        put(c, b, c->t_u32);
        put(c, b, val);
        put(c, b, inb);
        put(c, b, av.id);
        put(c, b, ident);
        uint32_t scope = const_u32(c, 3); // Subgroup
        uint32_t res = nid(c);
        put(c, b, (7u << 16) | op);
        put(c, b, c->t_u32);
        put(c, b, res);
        put(c, b, scope);
        put(c, b, 3u); // ClusteredReduce (literal, grammar ValueEnum)
        put(c, b, val);
        put(c, b, const_u32(c, 32)); // ClusterSize id
        def_val(c, in->dst, PTY_U32, res);
        return;
    }
