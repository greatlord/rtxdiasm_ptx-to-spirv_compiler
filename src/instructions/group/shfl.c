#include "instructions/group/group.h"

void emit_shfl_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    uint32_t id;
        // Warp shuffle over 32-lane warps (ZLUDA ground truth in
        // ptx/lib/zluda_ptx_impl.cpp: section_mask = (c>>8)&31,
        // warp_end = c&31 (0 for up, 31 else), delta &= 31).
        // Lanes map into the hardware subgroup (SubgroupLocalInvocationId
        // S: warp W = S>>5, lane L = S&31), so any subgroup size that is
        // a multiple of 32 works (32/64/128); one Shuffle + selects.
        // Out-of-range/out-of-mask lanes take their own value with a
        // false predicate. Converged execution required, as in CUDA.
        if (check_dst(c, in->dst, in->ty, line) != 0)
            return;
        if (in->dst2[0]) {
            reg_t *r2 = find_reg(c, in->dst2);
            if (!r2 || r2->decl_ty != PTY_PRED) {
                failf(c, line, "shfl second destination must be a .pred register");
                return;
            }
        }
        val_t av = resolve(c, b, &in->src[0], in->ty, line);
        if (c->failed)
            return;
        val_t bv = resolve(c, b, &in->src[1], PTY_U32, line);
        if (c->failed)
            return;
        val_t cv = resolve(c, b, &in->src[2], PTY_U32, line);
        if (c->failed)
            return;
        val_t mv = resolve(c, b, &in->src[3], PTY_U32, line);
        if (c->failed)
            return;
        uint32_t own = av.id;
        if (in->ty == PTY_F32) {
            uint32_t w = nid(c);
            put(c, b, (4u << 16) | S_Bitcast);
            put(c, b, c->t_u32);
            put(c, b, w);
            put(c, b, av.id);
            own = w;
        }
        uint32_t sub = nid(c);
        emit_load(c, b, c->t_u32, sub, c->v_subid);
        uint32_t W = nid(c), L = nid(c);
        emit_arith(c, b, S_ShrL, c->t_u32, W, sub, const_u32(c, 5));
        emit_arith(c, b, S_And, c->t_u32, L, sub, const_u32(c, 31));
        uint32_t segmask = nid(c), segbase = nid(c);
        {
            uint32_t t = nid(c);
            emit_arith(c, b, S_ShrL, c->t_u32, t, cv.id, const_u32(c, 8));
            emit_arith(c, b, S_And, c->t_u32, segmask, t, const_u32(c, 31));
        }
        emit_arith(c, b, S_And, c->t_u32, segbase, segmask, L);
        uint32_t segend = nid(c);
        {
            uint32_t nm = nid(c), em = nid(c);
            put(c, b, (4u << 16) | S_Not);
            put(c, b, c->t_u32);
            put(c, b, nm);
            put(c, b, segmask);
            uint32_t endc = const_u32(c, in->shfl_mode == 0 ? 0u : 31u);
            emit_arith(c, b, S_And, c->t_u32, em, nm, endc);
            emit_arith(c, b, S_Or, c->t_u32, segend, segbase, em);
        }
        uint32_t delta = nid(c);
        emit_arith(c, b, S_And, c->t_u32, delta, bv.id, const_u32(c, 31));
        uint32_t src = nid(c), ok0 = nid(c);
        if (in->shfl_mode == 0) {
            emit_arith(c, b, S_ISub, c->t_u32, src, L, delta);
            put(c, b, (5u << 16) | S_SGreaterEqual);
            put(c, b, c->t_bool);
            put(c, b, ok0);
            put(c, b, src);
            put(c, b, segend);
        } else if (in->shfl_mode == 1) {
            emit_arith(c, b, S_IAdd, c->t_u32, src, L, delta);
            put(c, b, (5u << 16) | S_ULessEqual);
            put(c, b, c->t_bool);
            put(c, b, ok0);
            put(c, b, src);
            put(c, b, segend);
        } else if (in->shfl_mode == 2) {
            emit_arith(c, b, S_Xor, c->t_u32, src, L, delta);
            put(c, b, (5u << 16) | S_ULessEqual);
            put(c, b, c->t_bool);
            put(c, b, ok0);
            put(c, b, src);
            put(c, b, segend);
        } else {
            uint32_t t = nid(c);
            put(c, b, (4u << 16) | S_Not);
            put(c, b, c->t_u32);
            put(c, b, t);
            put(c, b, segmask);
            uint32_t m = nid(c);
            emit_arith(c, b, S_And, c->t_u32, m, delta, t);
            emit_arith(c, b, S_Or, c->t_u32, src, m, segbase);
            put(c, b, (5u << 16) | S_ULessEqual);
            put(c, b, c->t_bool);
            put(c, b, ok0);
            put(c, b, src);
            put(c, b, segend);
        }
        uint32_t ok = nid(c);
        {
            uint32_t sh = nid(c), bt = nid(c), nz = nid(c);
            emit_arith(c, b, S_ShrL, c->t_u32, sh, mv.id, L);
            emit_arith(c, b, S_And, c->t_u32, bt, sh, const_u32(c, 1));
            put(c, b, (5u << 16) | S_INotEqual);
            put(c, b, c->t_bool);
            put(c, b, nz);
            put(c, b, bt);
            put(c, b, const_u32(c, 0));
            put(c, b, (5u << 16) | S_LogicalAnd);
            put(c, b, c->t_bool);
            put(c, b, ok);
            put(c, b, ok0);
            put(c, b, nz);
        }
        uint32_t src_sub = nid(c);
        {
            uint32_t wb = nid(c);
            emit_arith(c, b, S_Shl, c->t_u32, wb, W, const_u32(c, 5));
            emit_arith(c, b, S_Or, c->t_u32, src_sub, wb, src);
        }
        uint32_t scope = const_u32(c, 3); // Subgroup
        uint32_t shuffled = nid(c);
        put(c, b, (6u << 16) | S_GroupShuffle);
        put(c, b, c->t_u32);
        put(c, b, shuffled);
        put(c, b, scope);
        put(c, b, own);
        put(c, b, src_sub);
        uint32_t dbits = nid(c);
        put(c, b, (6u << 16) | S_Select);
        put(c, b, c->t_u32);
        put(c, b, dbits);
        put(c, b, ok);
        put(c, b, shuffled);
        put(c, b, own);
        if (in->ty == PTY_F32) {
            id = nid(c);
            put(c, b, (4u << 16) | S_Bitcast);
            put(c, b, c->t_f32);
            put(c, b, id);
            put(c, b, dbits);
            def_val(c, in->dst, PTY_F32, id);
        } else {
            def_val(c, in->dst, PTY_U32, dbits);
        }
        if (in->dst2[0])
            def_val(c, in->dst2, PTY_PRED, ok);
        return;
    }
