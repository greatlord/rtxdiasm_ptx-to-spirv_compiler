#include "instructions/group/group.h"

void emit_ldmatrix_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
        // ldmatrix.sync.aligned.m8n8.x{1,2,4}[.trans].b16 lowered as one
        // shared load per destination register, with the row address
        // fetched from the owning lane (PTX-ISA "addresses required for
        // each matrix are provided by eight threads": matrix m's rows
        // live in lanes 8m+0..8m+7).
        //
        // non-trans:
        //   row  r = myLane >> 2   (address provider lane 8*m + r)
        //   frag f = myLane & 3    (4 bytes = 2 b16 elements)
        //   value = b32 at p[8*m + r] + 4*f
        // .trans: lane t holds row r of the logical (transposed) tile in
        // fragment pair f, i.e. elements (r, 2f) and (r, 2f+1), which
        // live in the chunks provided by lanes 8m+2f and 8m+2f+1 at
        // 2*r bytes into each chunk (two u16 loads, same 16-bit half).
        //
        // Verified bit-exact against NVIDIA vectors via an exact integer
        // oracle before lowering (see docs/supported-isa.md).
        int binding = -1;
        uint32_t off = 0;
        int shvar = -1, glvar = -1;
        if (resolve_addr(c, b, &in->src[0], &binding, &off, &shvar, &glvar,
                         line) != 0)
            return;
        if (shvar < 0) {
            failf(c, line, "ldmatrix needs a .shared-derived address");
            return;
        }
        for (int m = 0; m < in->ldm_num; m++) {
            if (check_dst(c, in->mregs[m], PTY_U32, line) != 0)
                return;
        }
        // ldmatrix.sync is itself a warp sync point ("causes the
        // executing thread to wait until all threads in the warp execute
        // the same ldmatrix"), and the tiles it reads are normally
        // written by that same warp's earlier st.shared. Without a
        // barrier here the driver is free to sink the loads above the
        // stores, which reads back zeros.
        {
            uint32_t exec = const_u32(c, 2); // Workgroup
            uint32_t mem = const_u32(c, 2);
            uint32_t sem = const_u32(c, 0x108u);
            put(c, b, (4u << 16) | S_ControlBarrier);
            put(c, b, exec);
            put(c, b, mem);
            put(c, b, sem);
        }
        uint32_t sub = nid(c);
        emit_load(c, b, c->t_u32, sub, c->v_subid);
        uint32_t W = nid(c), L = nid(c);
        emit_arith(c, b, S_ShrL, c->t_u32, W, sub, const_u32(c, 5));
        emit_arith(c, b, S_And, c->t_u32, L, sub, const_u32(c, 31));
        uint32_t r = nid(c), f = nid(c);
        emit_arith(c, b, S_ShrL, c->t_u32, r, L, const_u32(c, 2));
        emit_arith(c, b, S_And, c->t_u32, f, L, const_u32(c, 3));
        // Row word index = byte offset >> 2 (shared offsets are small;
        // the u32 truncation is the same word address either way).
        uint32_t myword = nid(c), myword32 = nid(c);
        emit_arith(c, b, S_ShrL, c->t_u64, myword, off, const_u64(c, 2));
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, c->t_u32);
        put(c, b, myword32);
        put(c, b, myword);
        uint32_t w5 = nid(c);
        emit_arith(c, b, S_Shl, c->t_u32, w5, W, const_u32(c, 5));
        uint32_t scope = const_u32(c, 3); // Subgroup
        // .trans reads two u16s per destination register from two
        // adjacent 16-byte chunks that are provided by adjacent lanes:
        // lane t's fragment for matrix m is
        //   x = u16(A[8m + 2f] + 2r) | u16(A[8m + 2f + 1] + 2r) << 16
        // (the b32 halves are not adjacent in shared memory, so this is
        // two word loads + shifts, not one). Matches ZLUDA's pinned
        // NVIDIA ldmatrix_trans vectors 256/256.
        uint32_t chunk = nid(c);
        emit_arith(c, b, S_Shl, c->t_u32, chunk, f, const_u32(c, 1));
        uint32_t rowoff = nid(c), sh4 = nid(c), sh16 = nid(c);
        emit_arith(c, b, S_ShrL, c->t_u32, rowoff, r, const_u32(c, 1));
        emit_arith(c, b, S_And, c->t_u32, sh4, L, const_u32(c, 4));
        emit_arith(c, b, S_Shl, c->t_u32, sh16, sh4, const_u32(c, 2));
        for (int m = 0; m < in->ldm_num; m++) {
            if (in->ldm_trans) {
                uint32_t p0 = nid(c), p1 = nid(c), s0 = nid(c), s1 = nid(c);
                emit_arith(c, b, S_IAdd, c->t_u32, p0,
                           const_u32(c, (uint32_t)(8 * m)), chunk);
                emit_arith(c, b, S_IAdd, c->t_u32, p1, p0, const_u32(c, 1));
                emit_arith(c, b, S_Or, c->t_u32, s0, w5, p0);
                emit_arith(c, b, S_Or, c->t_u32, s1, w5, p1);
                uint32_t b0 = nid(c), b1 = nid(c);
                put(c, b, (6u << 16) | S_GroupShuffle);
                put(c, b, c->t_u32);
                put(c, b, b0);
                put(c, b, scope);
                put(c, b, myword32);
                put(c, b, s0);
                put(c, b, (6u << 16) | S_GroupShuffle);
                put(c, b, c->t_u32);
                put(c, b, b1);
                put(c, b, scope);
                put(c, b, myword32);
                put(c, b, s1);
                uint32_t v0 = nid(c), v1 = nid(c);
                for (int half = 0; half < 2; half++) {
                    uint32_t w = nid(c);
                    emit_arith(c, b, S_IAdd, c->t_u32, w,
                               half == 0 ? b0 : b1, rowoff);
                    uint32_t ptr = sh_word_ptr(c, b, shvar, w);
                    emit_load(c, b, c->t_u32, half == 0 ? v0 : v1, ptr);
                }
                uint32_t r0 = nid(c), r1 = nid(c), l0 = nid(c), l1 = nid(c);
                uint32_t x = nid(c), xt = nid(c);
                emit_arith(c, b, S_ShrL, c->t_u32, r0, v0, sh16);
                emit_arith(c, b, S_And, c->t_u32, l0, r0,
                           const_u32(c, 0xFFFFu));
                emit_arith(c, b, S_ShrL, c->t_u32, r1, v1, sh16);
                emit_arith(c, b, S_And, c->t_u32, l1, r1,
                           const_u32(c, 0xFFFFu));
                emit_arith(c, b, S_Shl, c->t_u32, xt, l1, const_u32(c, 16));
                emit_arith(c, b, S_Or, c->t_u32, x, l0, xt);
                def_val(c, in->mregs[m], PTY_U32, x);
                continue;
            }
            // Provider lane and within-row word offset.
            uint32_t srcLane = nid(c), srcSub = nid(c);
            emit_arith(c, b, S_IAdd, c->t_u32, srcLane,
                       const_u32(c, (uint32_t)(8 * m)), r);
            emit_arith(c, b, S_Or, c->t_u32, srcSub, w5, srcLane);
            uint32_t base = nid(c);
            put(c, b, (6u << 16) | S_GroupShuffle);
            put(c, b, c->t_u32);
            put(c, b, base);
            put(c, b, scope);
            put(c, b, myword32);
            put(c, b, srcSub);
            uint32_t w = nid(c);
            emit_arith(c, b, S_IAdd, c->t_u32, w, base, f);
            uint32_t ptr = sh_word_ptr(c, b, shvar, w);
            uint32_t v = nid(c);
            emit_load(c, b, c->t_u32, v, ptr);
            def_val(c, in->mregs[m], PTY_U32, v);
        }
        return;
    }
