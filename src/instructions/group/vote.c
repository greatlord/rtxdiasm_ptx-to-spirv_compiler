#include "instructions/group/group.h"

void emit_vote_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
        // Warp votes over 32-lane warps. bal = Ballot(pred) as uvec4;
        // word = the ballot word of my warp (all four words extracted
        // by constant index, selected by warp index); active = same
        // for constant-true. all = ((word & mask) == (active & mask))
        // so inactive lanes are excluded like the execution-mask
        // comparison; any = ((word & mask) != 0); ballot = raw word.
        // Converged mask members required, as in CUDA (divergence is
        // UB either way). Predicated votes desugar to selects in the
        // front end under the same rule (skipped lanes keep old).
        uint32_t scope = const_u32(c, 3); // Subgroup
        uint32_t sub = nid(c);
        emit_load(c, b, c->t_u32, sub, c->v_subid);
        uint32_t W = nid(c);
        emit_arith(c, b, S_ShrL, c->t_u32, W, sub, const_u32(c, 5));
        uint32_t pv = 0, mask = 0;
        if (in->vote_mode != 3) {
            pv = pred_val(c, b, &in->src[0], in->bneg, line);
            if (c->failed)
                return;
            if (in->nsrc > 1) {
                val_t mv = resolve(c, b, &in->src[1], PTY_U32, line);
                if (c->failed)
                    return;
                mask = mv.id;
            } else {
                mask = const_u32(c, 0xffffffffu);
            }
        } else {
            if (check_dst(c, in->dst, PTY_U32, line) != 0)
                return;
            pv = nid(c);
            put(c, &c->types, (3u << 16) | S_ConstantTrue);
            put(c, &c->types, c->t_bool);
            put(c, &c->types, pv);
        }
        uint32_t bal = nid(c);
        put(c, b, (5u << 16) | S_GroupBallot);
        put(c, b, c->t_uvec4);
        put(c, b, bal);
        put(c, b, scope);
        put(c, b, pv);
        // Warp word out of the ballot (constant extracts + select).
        uint32_t word = ballot_warp_word(c, b, bal, W);
        if (in->vote_mode == 2 || in->vote_mode == 3) {
            if (in->vote_mode == 2 &&
                check_dst(c, in->dst, PTY_U32, line) != 0)
                return;
            def_val(c, in->dst, PTY_U32, word);
            return;
        }
        if (check_dst(c, in->dst, PTY_PRED, line) != 0)
            return;
        if (in->vote_mode == 1) {
            uint32_t m = nid(c), nz = nid(c);
            emit_arith(c, b, S_And, c->t_u32, m, word, mask);
            put(c, b, (5u << 16) | S_INotEqual);
            put(c, b, c->t_bool);
            put(c, b, nz);
            put(c, b, m);
            put(c, b, const_u32(c, 0));
            def_val(c, in->dst, PTY_PRED, nz);
            return;
        }
        uint32_t abal = nid(c);
        {
            uint32_t t = nid(c);
            put(c, &c->types, (3u << 16) | S_ConstantTrue);
            put(c, &c->types, c->t_bool);
            put(c, &c->types, t);
            put(c, b, (5u << 16) | S_GroupBallot);
            put(c, b, c->t_uvec4);
            put(c, b, abal);
            put(c, b, scope);
            put(c, b, t);
        }
        uint32_t aword = ballot_warp_word(c, b, abal, W);
        uint32_t lm = nid(c), rm = nid(c), eq = nid(c);
        emit_arith(c, b, S_And, c->t_u32, lm, word, mask);
        emit_arith(c, b, S_And, c->t_u32, rm, aword, mask);
        put(c, b, (5u << 16) | S_IEqual);
        put(c, b, c->t_bool);
        put(c, b, eq);
        put(c, b, lm);
        put(c, b, rm);
        if (in->vote_mode == 4) {
            // .uni: all member lanes agree. True when the masked ballot
            // is either empty (all false) or equal to the masked active
            // set (all true).
            uint32_t zer = nid(c);
            put(c, b, (5u << 16) | S_IEqual);
            put(c, b, c->t_bool);
            put(c, b, zer);
            put(c, b, lm);
            put(c, b, const_u32(c, 0));
            uint32_t uni = nid(c);
            put(c, b, (5u << 16) | S_LogicalOr);
            put(c, b, c->t_bool);
            put(c, b, uni);
            put(c, b, zer);
            put(c, b, eq);
            def_val(c, in->dst, PTY_PRED, uni);
            return;
        }
        def_val(c, in->dst, PTY_PRED, eq);
        return;
    }
