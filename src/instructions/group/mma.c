#include "instructions/group/group.h"

#include <string.h>

void emit_mma_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
        if (in->mma_sparse && in->mma_elem == 7) {
            // Sparse 4:8 u4/s4 m16n8k64/k128: four nonzeros per
            // 8-wide chunk, selected as two 2-wide subchunks.
            int kd = in->mma_k;
            int sn = kd == 128 ? 4 : 2;
            for (int m = 0; m < 4; m++) {
                if (check_dst(c, in->mregs[m], PTY_U32, line) != 0)
                    return;
            }
            uint32_t Areg[4], Breg[4], Creg[4], Ereg;
            for (int j = 0; j < sn; j++) {
                ptx_opd_t o;
                memset(&o, 0, sizeof(o));
                o.kind = OPD_REG;
                strcpy(o.reg, in->mregs[4 + j]);
                val_t v = resolve(c, b, &o, PTY_U32, line);
                if (c->failed)
                    return;
                Areg[j] = v.id;
                memset(&o, 0, sizeof(o));
                o.kind = OPD_REG;
                strcpy(o.reg, in->mregs[8 + j]);
                v = resolve(c, b, &o, PTY_U32, line);
                if (c->failed)
                    return;
                Breg[j] = v.id;
            }
            for (int j = 0; j < 4; j++) {
                ptx_opd_t o;
                memset(&o, 0, sizeof(o));
                o.kind = OPD_REG;
                strcpy(o.reg, in->mregs[8 + sn + j]);
                val_t v = resolve(c, b, &o, PTY_U32, line);
                if (c->failed)
                    return;
                Creg[j] = v.id;
            }
            {
                ptx_opd_t o;
                memset(&o, 0, sizeof(o));
                o.kind = OPD_REG;
                strcpy(o.reg, in->smreg);
                val_t v = resolve(c, b, &o, PTY_U32, line);
                if (c->failed)
                    return;
                Ereg = v.id;
            }
            uint32_t sub = nid(c), L = nid(c), W = nid(c);
            uint32_t gg = nid(c), tt = nid(c), wbase = nid(c);
            emit_load(c, b, c->t_u32, sub, c->v_subid);
            emit_arith(c, b, S_And, c->t_u32, L, sub, const_u32(c, 31));
            emit_arith(c, b, S_ShrL, c->t_u32, W, sub, const_u32(c, 5));
            emit_arith(c, b, S_ShrL, c->t_u32, gg, L, const_u32(c, 2));
            emit_arith(c, b, S_And, c->t_u32, tt, L, const_u32(c, 3));
            emit_arith(c, b, S_Shl, c->t_u32, wbase, W, const_u32(c, 5));
            int is_signed = in->mma_asign || in->mma_bsign;
            for (int m = 0; m < 4; m++) {
                uint32_t row = nid(c), col = nid(c), acc = Creg[m];
                uint32_t sh = const_u32(c, m >= 2 ? 8u : 0u);
                emit_arith(c, b, S_IAdd, c->t_u32, row, gg, sh);
                uint32_t c0 = nid(c), cc = nid(c);
                emit_arith(c, b, S_Shl, c->t_u32, c0, tt,
                           const_u32(c, 1));
                emit_arith(c, b, S_IAdd, c->t_u32, cc, c0,
                           const_u32(c, (uint32_t)(m & 1)));
                col = cc;
                uint32_t rr = nid(c), rowhi = nid(c), rbase = nid(c);
                emit_arith(c, b, S_And, c->t_u32, rr, row,
                           const_u32(c, 7));
                emit_arith(c, b, S_ShrL, c->t_u32, rowhi, row,
                           const_u32(c, 3));
                emit_arith(c, b, S_Shl, c->t_u32, rbase, rr,
                           const_u32(c, 2));
                int chunks = kd / 8;
                for (int cch = 0; cch < chunks; cch++) {
                    uint32_t owner = nid(c), msh = nid(c), nib = nid(c);
                    int half = cch >= 8;
                    int meta_off, nib_idx, areg, a_off;
                    if (kd == 128) {
                        meta_off = (half ? 2 : 0) + (m >= 2 ? 1 : 0);
                        nib_idx = half ? cch - 8 : cch;
                        areg = (half ? 2 : 0) + (m >= 2 ? 1 : 0);
                        a_off = half ? cch - 8 : cch;
                    } else {
                        meta_off = in->mma_sel * 2 + (m >= 2 ? 1 : 0);
                        nib_idx = cch;
                        areg = (m >= 2 ? 1 : 0);
                        a_off = cch;
                    }
                    emit_arith(c, b, S_IAdd, c->t_u32, owner, rbase,
                               const_u32(c, (uint32_t)meta_off));
                    uint32_t meta =
                        wmma_shfl(c, b, Ereg, owner, wbase);
                    emit_arith(c, b, S_ShrL, c->t_u32, msh, meta,
                               const_u32(c, (uint32_t)(4 * nib_idx)));
                    emit_arith(c, b, S_And, c->t_u32, nib, msh,
                               const_u32(c, 0xfu));
                    uint32_t aown = nid(c), abase = nid(c);
                    emit_arith(c, b, S_Shl, c->t_u32, abase, rr,
                               const_u32(c, 2));
                    emit_arith(c, b, S_IAdd, c->t_u32, aown, abase,
                               const_u32(c, (uint32_t)(a_off >> 1)));
                    uint32_t aw = wmma_shfl(
                        c, b, Areg[areg], aown, wbase);
                    for (int which = 0; which < 2; which++) {
                        uint32_t subc = nid(c);
                        if (which) {
                            emit_arith(c, b, S_ShrL, c->t_u32, subc, nib,
                                       const_u32(c, 2));
                        } else {
                            emit_arith(c, b, S_And, c->t_u32, subc, nib,
                                       const_u32(c, 3));
                        }
                        for (int e = 0; e < 2; e++) {
                            int anib = (a_off & 1) * 4 + which * 2 + e;
                            uint32_t ash = nid(c), an = nid(c);
                            emit_arith(c, b, S_ShrL, c->t_u32, ash, aw,
                                       const_u32(c, (uint32_t)(4 * anib)));
                            emit_arith(c, b, S_And, c->t_u32, an, ash,
                                       const_u32(c, 0xfu));
                            if (is_signed) {
                                uint32_t sl = nid(c), sa = nid(c);
                                emit_arith(c, b, S_Shl, c->t_u32, sl, an,
                                           const_u32(c, 28));
                                emit_arith(c, b, S_ShrA, c->t_u32, sa, sl,
                                           const_u32(c, 28));
                                an = sa;
                            }
                            uint32_t bown = nid(c), bown2 = nid(c);
                            uint32_t bsh = nid(c), bn = nid(c);
                            emit_arith(c, b, S_Shl, c->t_u32, bown, col,
                                       const_u32(c, 2));
                            emit_arith(c, b, S_IAdd, c->t_u32, bown2, bown,
                                       const_u32(c, (uint32_t)(cch & 3)));
                            uint32_t bw = wmma_shfl(
                                c, b, Breg[cch >> 2], bown2, wbase);
                            uint32_t ks1 = nid(c), ks2 = nid(c);
                            uint32_t kshift = nid(c);
                            emit_arith(c, b, S_Shl, c->t_u32, ks1, subc,
                                       const_u32(c, 1));
                            emit_arith(c, b, S_IAdd, c->t_u32, ks2, ks1,
                                       const_u32(c, (uint32_t)e));
                            emit_arith(c, b, S_Shl, c->t_u32, kshift, ks2,
                                       const_u32(c, 2));
                            emit_arith(c, b, S_ShrL, c->t_u32, bsh, bw,
                                       kshift);
                            emit_arith(c, b, S_And, c->t_u32, bn, bsh,
                                       const_u32(c, 0xfu));
                            if (is_signed) {
                                uint32_t sl = nid(c), sa = nid(c);
                                emit_arith(c, b, S_Shl, c->t_u32, sl, bn,
                                           const_u32(c, 28));
                                emit_arith(c, b, S_ShrA, c->t_u32, sa, sl,
                                           const_u32(c, 28));
                                bn = sa;
                            }
                            uint32_t pr = nid(c), nx = nid(c);
                            emit_arith(c, b, S_IMul, c->t_u32, pr, an, bn);
                            emit_arith(c, b, S_IAdd, c->t_u32, nx, acc, pr);
                            acc = nx;
                        }
                    }
                }
                def_val(c, in->mregs[m], PTY_U32, acc);
            }
            return;
        }
        if (in->mma_sparse && in->mma_elem == 3) {
            // Sparse TF32: 1:2 structured A, 4-bit metadata.
            int sk = in->mma_k;
            int sn = sk == 16 ? 4 : 2;
            for (int m = 0; m < 4; m++) {
                if (check_dst(c, in->mregs[m], PTY_F32, line) != 0)
                    return;
            }
            uint32_t Areg[4], Breg[4], Creg[4], Ereg;
            for (int j = 0; j < sn; j++) {
                ptx_opd_t o;
                memset(&o, 0, sizeof(o));
                o.kind = OPD_REG;
                strcpy(o.reg, in->mregs[4 + j]);
                val_t v = resolve(c, b, &o, PTY_U32, line);
                if (c->failed)
                    return;
                Areg[j] = v.id;
                memset(&o, 0, sizeof(o));
                o.kind = OPD_REG;
                strcpy(o.reg, in->mregs[8 + j]);
                v = resolve(c, b, &o, PTY_U32, line);
                if (c->failed)
                    return;
                Breg[j] = v.id;
            }
            for (int j = 0; j < 4; j++) {
                ptx_opd_t o;
                memset(&o, 0, sizeof(o));
                o.kind = OPD_REG;
                strcpy(o.reg, in->mregs[8 + sn + j]);
                val_t v = resolve(c, b, &o, PTY_F32, line);
                if (c->failed)
                    return;
                Creg[j] = v.id;
            }
            {
                ptx_opd_t o;
                memset(&o, 0, sizeof(o));
                o.kind = OPD_REG;
                strcpy(o.reg, in->smreg);
                val_t v = resolve(c, b, &o, PTY_U32, line);
                if (c->failed)
                    return;
                Ereg = v.id;
            }
            uint32_t sub = nid(c), L = nid(c), W = nid(c);
            uint32_t gg = nid(c), tt = nid(c), wbase = nid(c);
            emit_load(c, b, c->t_u32, sub, c->v_subid);
            emit_arith(c, b, S_And, c->t_u32, L, sub, const_u32(c, 31));
            emit_arith(c, b, S_ShrL, c->t_u32, W, sub, const_u32(c, 5));
            emit_arith(c, b, S_ShrL, c->t_u32, gg, L, const_u32(c, 2));
            emit_arith(c, b, S_And, c->t_u32, tt, L, const_u32(c, 3));
            emit_arith(c, b, S_Shl, c->t_u32, wbase, W, const_u32(c, 5));
            int chunks = sk / 2;
            for (int m = 0; m < 4; m++) {
                uint32_t row = nid(c), col = nid(c), acc = Creg[m];
                uint32_t sh = const_u32(c, m >= 2 ? 8u : 0u);
                emit_arith(c, b, S_IAdd, c->t_u32, row, gg, sh);
                uint32_t c0 = nid(c), cc = nid(c);
                emit_arith(c, b, S_Shl, c->t_u32, c0, tt,
                           const_u32(c, 1));
                emit_arith(c, b, S_IAdd, c->t_u32, cc, c0,
                           const_u32(c, (uint32_t)(m & 1)));
                col = cc;
                uint32_t rr = nid(c), rowhi = nid(c), rbase = nid(c);
                emit_arith(c, b, S_And, c->t_u32, rr, row,
                           const_u32(c, 7));
                emit_arith(c, b, S_ShrL, c->t_u32, rowhi, row,
                           const_u32(c, 3));
                emit_arith(c, b, S_Shl, c->t_u32, rbase, rr,
                           const_u32(c, 2));
                for (int cch = 0; cch < chunks; cch++) {
                    uint32_t owner = nid(c);
                    if (sk == 16) {
                        emit_arith(c, b, S_IAdd, c->t_u32, owner, rbase,
                                   const_u32(c, (uint32_t)(in->mma_sel * 2 +
                                                           (cch >= 4 ? 1 : 0))));
                    } else {
                        emit_arith(c, b, S_IAdd, c->t_u32, owner, rbase,
                                   const_u32(c, (uint32_t)in->mma_sel));
                    }
                    uint32_t meta =
                        wmma_shfl(c, b, Ereg, owner, wbase);
                    int shbits = (cch & 3) + (m >= 2 ? 4 : 0);
                    uint32_t msh = nid(c), mnib = nid(c);
                    emit_arith(c, b, S_ShrL, c->t_u32, msh, meta,
                               const_u32(c, (uint32_t)(4 * shbits)));
                    emit_arith(c, b, S_And, c->t_u32, mnib, msh,
                               const_u32(c, 0xfu));
                    uint32_t cond = do_cmp(c, b, PCM_EQ, PTY_U32, 0,
                                           mnib, const_u32(c, 0xeu), line);
                    uint32_t pos = sel_u32(c, b, cond, const_u32(c, 1),
                                           const_u32(c, 0));
                    int areg;
                    if (sk == 16)
                        areg = (cch >= 4 ? 2 : 0) + (m >= 2);
                    else
                        areg = (m >= 2);
                    uint32_t aown = nid(c);
                    emit_arith(c, b, S_IAdd, c->t_u32, aown, rbase,
                               const_u32(c, (uint32_t)(cch & 3)));
                    uint32_t aval =
                        wmma_shfl(c, b, Areg[areg], aown, wbase);
                    uint32_t bown = nid(c), bown2 = nid(c);
                    uint32_t koff = nid(c), kon = nid(c);
                    emit_arith(c, b, S_IAdd, c->t_u32, koff, pos,
                               const_u32(c, (uint32_t)(2 * cch)));
                    emit_arith(c, b, S_And, c->t_u32, kon, koff,
                               const_u32(c, 3));
                    emit_arith(c, b, S_Shl, c->t_u32, bown, col,
                               const_u32(c, 2));
                    emit_arith(c, b, S_IAdd, c->t_u32, bown2, bown, kon);
                    uint32_t bval = wmma_shfl(
                        c, b, Breg[cch >> 1], bown2, wbase);
                    uint32_t ab = nid(c), bb = nid(c);
                    emit_arith(c, b, S_And, c->t_u32, ab, aval,
                               const_u32(c, 0xffffe000u));
                    emit_arith(c, b, S_And, c->t_u32, bb, bval,
                               const_u32(c, 0xffffe000u));
                    uint32_t af = nid(c), bf = nid(c);
                    put(c, b, (4u << 16) | S_Bitcast);
                    put(c, b, c->t_f32);
                    put(c, b, af);
                    put(c, b, ab);
                    put(c, b, (4u << 16) | S_Bitcast);
                    put(c, b, c->t_f32);
                    put(c, b, bf);
                    put(c, b, bb);
                    uint32_t pr = nid(c);
                    emit_fma_into(c, b, c->t_f32, pr, af, bf, acc);
                    acc = pr;
                }
                def_val(c, in->mregs[m], PTY_F32, acc);
            }
            return;
        }
        if (in->mma_sparse && in->mma_k == 32) {
            // Sparse m16n8k32 f16/bf16: 8 chunks, metadata split
            // between the selected thread pair (lanes 4g+i and 4g+i+1).
            int is_bf16 = in->mma_elem == 1;
            for (int m = 0; m < 4; m++) {
                if (check_dst(c, in->mregs[m], PTY_F32, line) != 0)
                    return;
            }
            uint32_t Areg[4], Breg[4], Creg[4], Ereg;
            for (int j = 0; j < 4; j++) {
                ptx_opd_t o;
                memset(&o, 0, sizeof(o));
                o.kind = OPD_REG;
                strcpy(o.reg, in->mregs[4 + j]);
                val_t v = resolve(c, b, &o, PTY_U32, line);
                if (c->failed)
                    return;
                Areg[j] = v.id;
                memset(&o, 0, sizeof(o));
                o.kind = OPD_REG;
                strcpy(o.reg, in->mregs[8 + j]);
                v = resolve(c, b, &o, PTY_U32, line);
                if (c->failed)
                    return;
                Breg[j] = v.id;
                memset(&o, 0, sizeof(o));
                o.kind = OPD_REG;
                strcpy(o.reg, in->mregs[12 + j]);
                v = resolve(c, b, &o, PTY_F32, line);
                if (c->failed)
                    return;
                Creg[j] = v.id;
            }
            {
                ptx_opd_t o;
                memset(&o, 0, sizeof(o));
                o.kind = OPD_REG;
                strcpy(o.reg, in->smreg);
                val_t v = resolve(c, b, &o, PTY_U32, line);
                if (c->failed)
                    return;
                Ereg = v.id;
            }
            uint32_t sub = nid(c);
            emit_load(c, b, c->t_u32, sub, c->v_subid);
            uint32_t W = nid(c), g = nid(c), t = nid(c);
            emit_arith(c, b, S_ShrL, c->t_u32, W, sub, const_u32(c, 5));
            uint32_t L = nid(c);
            emit_arith(c, b, S_And, c->t_u32, L, sub, const_u32(c, 31));
            emit_arith(c, b, S_ShrL, c->t_u32, g, L, const_u32(c, 2));
            emit_arith(c, b, S_And, c->t_u32, t, L, const_u32(c, 3));
            uint32_t scope = const_u32(c, 3);
            uint32_t g4 = nid(c), t2 = nid(c), w5 = nid(c);
            emit_arith(c, b, S_Shl, c->t_u32, g4, g, const_u32(c, 2));
            emit_arith(c, b, S_Shl, c->t_u32, t2, t, const_u32(c, 1));
            emit_arith(c, b, S_Shl, c->t_u32, w5, W, const_u32(c, 5));
            uint32_t Aexp[2][32];
            uint32_t Apack[2][8][2];
            int use_dot2 = c->caps.mixed_fdot && !is_bf16;
            int dot2_hit = 0;
            if (use_dot2) {
                dot2_hit = c->sp_pack_gen == c->cse_gen &&
                           c->sp_pack_key[0] == Areg[0] &&
                           c->sp_pack_key[1] == Areg[1] &&
                           c->sp_pack_key[2] == Areg[2] &&
                           c->sp_pack_key[3] == Areg[3] &&
                           c->sp_pack_key[4] == Ereg &&
                           c->sp_pack_key[5] == (uint32_t)in->mma_sel;
                if (dot2_hit)
                    memcpy(Apack, c->sp_pack_a, sizeof(Apack));
            }
            uint32_t m0 = 0, m1 = 0;
            if (!dot2_hit) {
                m0 = nid(c);
                m1 = nid(c);
                uint32_t mown0 = nid(c), mown1 = nid(c);
                emit_arith(c, b, S_Or, c->t_u32, mown0, g4,
                           const_u32(c, (uint32_t)in->mma_sel));
                emit_arith(c, b, S_Or, c->t_u32, mown1, g4,
                           const_u32(c, (uint32_t)(in->mma_sel + 1)));
                uint32_t msrc0 = nid(c), msrc1 = nid(c);
                emit_arith(c, b, S_Or, c->t_u32, msrc0, w5, mown0);
                emit_arith(c, b, S_Or, c->t_u32, msrc1, w5, mown1);
                put(c, b, (6u << 16) | S_GroupShuffle);
                put(c, b, c->t_u32);
                put(c, b, m0);
                put(c, b, scope);
                put(c, b, Ereg);
                put(c, b, msrc0);
                put(c, b, (6u << 16) | S_GroupShuffle);
                put(c, b, c->t_u32);
                put(c, b, m1);
                put(c, b, scope);
                put(c, b, Ereg);
                put(c, b, msrc1);
                for (int hr = 0; hr < 2; hr++) {
                    for (int cch = 0; cch < 8; cch++) {
                        uint32_t alane = nid(c);
                        emit_arith(c, b, S_Or, c->t_u32, alane, g4,
                                   const_u32(c, (uint32_t)(cch & 3)));
                        uint32_t asrc = nid(c), agot = nid(c);
                        emit_arith(c, b, S_Or, c->t_u32, asrc, w5, alane);
                        put(c, b, (6u << 16) | S_GroupShuffle);
                        put(c, b, c->t_u32);
                        put(c, b, agot);
                        put(c, b, scope);
                        put(c, b, Areg[(cch >= 4 ? 2 : 0) + hr]);
                        put(c, b, asrc);
                        uint32_t v0b = nid(c), v1b = nid(c);
                        emit_arith(c, b, S_And, c->t_u32, v0b, agot,
                                   const_u32(c, 0xffffu));
                        emit_arith(c, b, S_ShrL, c->t_u32, v1b, agot,
                                   const_u32(c, 16));
                        uint32_t v0 = 0, v1 = 0;
                        if (!use_dot2) {
                            v0 = is_bf16
                                     ? bf16_to_f32val(c, b, v0b, line)
                                     : f16_to_f32val(c, b, v0b, line);
                            v1 = is_bf16
                                     ? bf16_to_f32val(c, b, v1b, line)
                                     : f16_to_f32val(c, b, v1b, line);
                            if (c->failed)
                                return;
                        }
                        uint32_t shift = (uint32_t)(4 * (cch & 3) +
                                                    (hr ? 16 : 0));
                        uint32_t sh0 = nid(c), sh1 = nid(c);
                        emit_arith(c, b, S_ShrL, c->t_u32, sh0,
                                   cch < 4 ? m0 : m1,
                                   const_u32(c, shift));
                        emit_arith(c, b, S_ShrL, c->t_u32, sh1,
                                   cch < 4 ? m0 : m1,
                                   const_u32(c, shift + 2));
                        uint32_t i0 = nid(c), i1 = nid(c);
                        emit_arith(c, b, S_And, c->t_u32, i0, sh0,
                                   const_u32(c, 3));
                        emit_arith(c, b, S_And, c->t_u32, i1, sh1,
                                   const_u32(c, 3));
                        if (use_dot2) {
                            // Sparse pair: the two nonzeros in metadata
                            // order packed into one f16x2 word, plus the
                            // indices used to select the matching B values.
                            uint32_t vsh = nid(c), vp = nid(c);
                            emit_arith(c, b, S_Shl, c->t_u32, vsh, v1b,
                                       const_u32(c, 16));
                            emit_arith(c, b, S_Or, c->t_u32, vp, v0b, vsh);
                            uint32_t msh = nid(c), mm = nid(c);
                            emit_arith(c, b, S_Shl, c->t_u32, msh, i1,
                                       const_u32(c, 2));
                            emit_arith(c, b, S_Or, c->t_u32, mm, i0, msh);
                            Apack[hr][cch][0] = vp;
                            Apack[hr][cch][1] = mm;
                        } else {
                            for (int p = 0; p < 4; p++) {
                                uint32_t e0 = do_cmp(
                                    c, b, PCM_EQ, PTY_U32, 0,
                                    const_u32(c, (uint32_t)p), i0, line);
                                uint32_t e1 = do_cmp(
                                    c, b, PCM_EQ, PTY_U32, 0,
                                    const_u32(c, (uint32_t)p), i1, line);
                                if (c->failed)
                                    return;
                                uint32_t s1 = nid(c), sv = nid(c);
                                put(c, b, (6u << 16) | S_Select);
                                put(c, b, c->t_f32);
                                put(c, b, s1);
                                put(c, b, e1);
                                put(c, b, v1);
                                put(c, b, const_f32(c, 0));
                                put(c, b, (6u << 16) | S_Select);
                                put(c, b, c->t_f32);
                                put(c, b, sv);
                                put(c, b, e0);
                                put(c, b, v0);
                                put(c, b, s1);
                                Aexp[hr][cch * 4 + p] = sv;
                            }
                        }
                    }
                }
                if (use_dot2) {
                    memcpy(c->sp_pack_a, Apack, sizeof(Apack));
                    c->sp_pack_key[0] = Areg[0];
                    c->sp_pack_key[1] = Areg[1];
                    c->sp_pack_key[2] = Areg[2];
                    c->sp_pack_key[3] = Areg[3];
                    c->sp_pack_key[4] = Ereg;
                    c->sp_pack_key[5] = (uint32_t)in->mma_sel;
                    c->sp_pack_gen = c->cse_gen;
                }
            }
            uint32_t accs[4];
            for (int m = 0; m < 4; m++)
                accs[m] = Creg[m];
            if (use_dot2) {
                for (int cch = 0; cch < 8; cch++) {
                    int breg = cch >> 1;
                    uint32_t Braw[2][2];
                    for (int pair = 0; pair < 2; pair++) {
                        int laneoff = (2 * cch + pair) & 3;
                        for (int cc = 0; cc < 2; cc++) {
                            uint32_t ccol = nid(c), c4 = nid(c);
                            emit_arith(c, b, S_Or, c->t_u32, ccol, t2,
                                       const_u32(c, (uint32_t)cc));
                            emit_arith(c, b, S_Shl, c->t_u32, c4, ccol,
                                       const_u32(c, 2));
                            uint32_t bLane = nid(c), bGot = nid(c);
                            emit_arith(c, b, S_Or, c->t_u32, bLane, c4,
                                       const_u32(c, (uint32_t)laneoff));
                            put(c, b, (6u << 16) | S_GroupShuffle);
                            put(c, b, c->t_u32);
                            put(c, b, bGot);
                            put(c, b, scope);
                            put(c, b, Breg[breg]);
                            put(c, b, bLane);
                            Braw[pair][cc] = bGot;
                        }
                    }
                    uint32_t Bpack[2][2];
                    for (int ar = 0; ar < 2; ar++) {
                        uint32_t meta = Apack[ar][cch][1];
                        uint32_t i0 = nid(c), i1 = nid(c), i1m = nid(c);
                        emit_arith(c, b, S_And, c->t_u32, i0, meta,
                                   const_u32(c, 3));
                        emit_arith(c, b, S_ShrL, c->t_u32, i1, meta,
                                   const_u32(c, 2));
                        emit_arith(c, b, S_And, c->t_u32, i1m, i1,
                                   const_u32(c, 3));
                        for (int cc = 0; cc < 2; cc++) {
                            uint32_t b0 = sparse_pick_b(c, b, Braw, cc,
                                                        i0, line);
                            uint32_t b1 = sparse_pick_b(c, b, Braw, cc,
                                                        i1m, line);
                            uint32_t bsh = nid(c), bp = nid(c);
                            emit_arith(c, b, S_Shl, c->t_u32, bsh, b1,
                                       const_u32(c, 16));
                            emit_arith(c, b, S_Or, c->t_u32, bp, b0, bsh);
                            Bpack[ar][cc] = bp;
                        }
                    }
                    for (int m = 0; m < 4; m++) {
                        int ar = (m >= 2) ? 1 : 0;
                        int bc = (m & 1) ? 1 : 0;
                        accs[m] = emit_fdot2_f32(c, b, Apack[ar][cch][0],
                                                 Bpack[ar][bc], accs[m]);
                    }
                }
                for (int m = 0; m < 4; m++)
                    def_val(c, in->mregs[m], PTY_F32, accs[m]);
                return;
            }
            for (int kp = 0; kp < 16; kp++) {
                int laneoff = kp & 3;
                int breg = kp >> 2;
                uint32_t Bf[4];
                for (int cc = 0; cc < 2; cc++) {
                    uint32_t ccol = nid(c), c4 = nid(c);
                    emit_arith(c, b, S_Or, c->t_u32, ccol, t2,
                               const_u32(c, (uint32_t)cc));
                    emit_arith(c, b, S_Shl, c->t_u32, c4, ccol,
                               const_u32(c, 2));
                    uint32_t bLane = nid(c), bGot = nid(c);
                    emit_arith(c, b, S_Or, c->t_u32, bLane, c4,
                               const_u32(c, (uint32_t)laneoff));
                    put(c, b, (6u << 16) | S_GroupShuffle);
                    put(c, b, c->t_u32);
                    put(c, b, bGot);
                    put(c, b, scope);
                    put(c, b, Breg[breg]);
                    put(c, b, bLane);
                    uint32_t lo = nid(c), hi = nid(c);
                    emit_arith(c, b, S_And, c->t_u32, lo, bGot,
                               const_u32(c, 0xffffu));
                    emit_arith(c, b, S_ShrL, c->t_u32, hi, bGot,
                               const_u32(c, 16));
                    Bf[2 * cc] = is_bf16
                                     ? bf16_to_f32val(c, b, lo, line)
                                     : f16_to_f32val(c, b, lo, line);
                    Bf[2 * cc + 1] =
                        is_bf16 ? bf16_to_f32val(c, b, hi, line)
                                : f16_to_f32val(c, b, hi, line);
                    if (c->failed)
                        return;
                }
                for (int m = 0; m < 4; m++) {
                    int high = m >= 2;
                    int bc = (m & 1) ? 2 : 0;
                    uint32_t n0 = nid(c), n1 = nid(c);
                    emit_fma_into(c, b, c->t_f32, n0,
                                  Aexp[high][2 * kp], Bf[bc], accs[m]);
                    emit_fma_into(c, b, c->t_f32, n1,
                                  Aexp[high][2 * kp + 1],
                                  Bf[bc + 1], n0);
                    accs[m] = n1;
                }
            }
            for (int m = 0; m < 4; m++)
                def_val(c, in->mregs[m], PTY_F32, accs[m]);

            return;
        }
        if (in->mma_sparse) {
            // Sparse m16n8k16 f16/bf16: 2:4 structured A. Each lane's
            // compressed A regs hold the two nonzeros of its 4-wide
            // chunk; the selected lane of each group carries a b32 of
            // 16 2-bit indices (row g in bits [15:0], row g+8 in
            // [31:16], four chunks low-to-high). Expand A to dense
            // fragments, then reuse the dense B/C maps.
            int is_bf16 = in->mma_elem == 1;
            for (int m = 0; m < 4; m++) {
                if (check_dst(c, in->mregs[m], PTY_F32, line) != 0)
                    return;
            }
            uint32_t Areg[2], Breg[2], Creg[4], Ereg;
            for (int j = 0; j < 2; j++) {
                ptx_opd_t o;
                memset(&o, 0, sizeof(o));
                o.kind = OPD_REG;
                strcpy(o.reg, in->mregs[4 + j]);
                val_t v = resolve(c, b, &o, PTY_U32, line);
                if (c->failed)
                    return;
                Areg[j] = v.id;
            }
            for (int j = 0; j < 2; j++) {
                ptx_opd_t o;
                memset(&o, 0, sizeof(o));
                o.kind = OPD_REG;
                strcpy(o.reg, in->mregs[8 + j]);
                val_t v = resolve(c, b, &o, PTY_U32, line);
                if (c->failed)
                    return;
                Breg[j] = v.id;
            }
            for (int m = 0; m < 4; m++) {
                ptx_opd_t o;
                memset(&o, 0, sizeof(o));
                o.kind = OPD_REG;
                strcpy(o.reg, in->mregs[10 + m]);
                val_t v = resolve(c, b, &o, PTY_F32, line);
                if (c->failed)
                    return;
                Creg[m] = v.id;
            }
            {
                ptx_opd_t o;
                memset(&o, 0, sizeof(o));
                o.kind = OPD_REG;
                strcpy(o.reg, in->smreg);
                val_t v = resolve(c, b, &o, PTY_U32, line);
                if (c->failed)
                    return;
                Ereg = v.id;
            }
            uint32_t sub = nid(c);
            emit_load(c, b, c->t_u32, sub, c->v_subid);
            uint32_t W = nid(c), myG = nid(c), myT = nid(c);
            emit_arith(c, b, S_ShrL, c->t_u32, W, sub, const_u32(c, 5));
            uint32_t L = nid(c);
            emit_arith(c, b, S_And, c->t_u32, L, sub, const_u32(c, 31));
            emit_arith(c, b, S_ShrL, c->t_u32, myG, L, const_u32(c, 2));
            emit_arith(c, b, S_And, c->t_u32, myT, L, const_u32(c, 3));
            uint32_t scope = const_u32(c, 3);
            uint32_t g4 = nid(c), t2 = nid(c), w5 = nid(c);
            emit_arith(c, b, S_Shl, c->t_u32, g4, myG, const_u32(c, 2));
            emit_arith(c, b, S_Shl, c->t_u32, t2, myT, const_u32(c, 1));
            emit_arith(c, b, S_Shl, c->t_u32, w5, W, const_u32(c, 5));
            // Broadcast the group's metadata from lane 4*g + selector.
            uint32_t mown = nid(c), msrc = nid(c), meta = nid(c);
            emit_arith(c, b, S_Or, c->t_u32, mown, g4,
                       const_u32(c, (uint32_t)in->mma_sel));
            emit_arith(c, b, S_Or, c->t_u32, msrc, w5, mown);
            put(c, b, (6u << 16) | S_GroupShuffle);
            put(c, b, c->t_u32);
            put(c, b, meta);
            put(c, b, scope);
            put(c, b, Ereg);
            put(c, b, msrc);
            // Dense expansion of A rows g and g+8 (16 columns each).
            uint32_t Aexp[2][16];
            for (int hr = 0; hr < 2; hr++) {
                for (int cch = 0; cch < 4; cch++) {
                    uint32_t alane = nid(c), asrc = nid(c);
                    emit_arith(c, b, S_Or, c->t_u32, alane, g4,
                               const_u32(c, (uint32_t)cch));
                    emit_arith(c, b, S_Or, c->t_u32, asrc, w5, alane);
                    uint32_t agot = nid(c);
                    put(c, b, (6u << 16) | S_GroupShuffle);
                    put(c, b, c->t_u32);
                    put(c, b, agot);
                    put(c, b, scope);
                    put(c, b, Areg[hr]);
                    put(c, b, asrc);
                    uint32_t v0b = nid(c), v1b = nid(c);
                    emit_arith(c, b, S_And, c->t_u32, v0b, agot,
                               const_u32(c, 0xffffu));
                    emit_arith(c, b, S_ShrL, c->t_u32, v1b, agot,
                               const_u32(c, 16));
                    uint32_t v0 = is_bf16
                                      ? bf16_to_f32val(c, b, v0b, line)
                                      : f16_to_f32val(c, b, v0b, line);
                    uint32_t v1 = is_bf16
                                      ? bf16_to_f32val(c, b, v1b, line)
                                      : f16_to_f32val(c, b, v1b, line);
                    if (c->failed)
                        return;
                    uint32_t shift = (uint32_t)(hr * 16 + cch * 4);
                    uint32_t sh0 = nid(c), sh1 = nid(c);
                    emit_arith(c, b, S_ShrL, c->t_u32, sh0, meta,
                               const_u32(c, shift));
                    emit_arith(c, b, S_ShrL, c->t_u32, sh1, meta,
                               const_u32(c, shift + 2));
                    uint32_t i0 = nid(c), i1 = nid(c);
                    emit_arith(c, b, S_And, c->t_u32, i0, sh0,
                               const_u32(c, 3));
                    emit_arith(c, b, S_And, c->t_u32, i1, sh1,
                               const_u32(c, 3));
                    for (int p = 0; p < 4; p++) {
                        uint32_t e0 = do_cmp(c, b, PCM_EQ, PTY_U32, 0,
                                             const_u32(c, (uint32_t)p), i0,
                                             line);
                        uint32_t e1 = do_cmp(c, b, PCM_EQ, PTY_U32, 0,
                                             const_u32(c, (uint32_t)p), i1,
                                             line);
                        if (c->failed)
                            return;
                        uint32_t s1 = nid(c), sv = nid(c);
                        put(c, b, (6u << 16) | S_Select);
                        put(c, b, c->t_f32);
                        put(c, b, s1);
                        put(c, b, e1);
                        put(c, b, v1);
                        put(c, b, const_f32(c, 0));
                        put(c, b, (6u << 16) | S_Select);
                        put(c, b, c->t_f32);
                        put(c, b, sv);
                        put(c, b, e0);
                        put(c, b, v0);
                        put(c, b, s1);
                        Aexp[hr][cch * 4 + p] = sv;
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                uint32_t acc = Creg[m];
                uint32_t ccol = nid(c), c4 = nid(c);
                emit_arith(c, b, S_Or, c->t_u32, ccol, t2,
                           const_u32(c, (uint32_t)(m & 1)));
                emit_arith(c, b, S_Shl, c->t_u32, c4, ccol,
                           const_u32(c, 2));
                int hr = m >= 2;
                for (int k = 0; k < 16; k++) {
                    uint32_t Af = Aexp[hr][k];
                    int bElem = (k & 1) + (k >= 8 ? 2 : 0);
                    uint32_t bLane = nid(c), bSrc = nid(c);
                    emit_arith(c, b, S_Or, c->t_u32, bLane, c4,
                               const_u32(c, (uint32_t)((k & 7) >> 1)));
                    emit_arith(c, b, S_Or, c->t_u32, bSrc, w5, bLane);
                    uint32_t bGot = nid(c);
                    put(c, b, (6u << 16) | S_GroupShuffle);
                    put(c, b, c->t_u32);
                    put(c, b, bGot);
                    put(c, b, scope);
                    put(c, b, Breg[bElem >> 1]);
                    put(c, b, bSrc);
                    uint32_t bh = nid(c);
                    if (bElem & 1)
                        emit_arith(c, b, S_ShrL, c->t_u32, bh, bGot,
                                   const_u32(c, 16));
                    else
                        emit_arith(c, b, S_And, c->t_u32, bh, bGot,
                                   const_u32(c, 0xffffu));
                    uint32_t Bf = is_bf16
                                      ? bf16_to_f32val(c, b, bh, line)
                                      : f16_to_f32val(c, b, bh, line);
                    if (c->failed)
                        return;
                    uint32_t pr = nid(c);
                    emit_fma_into(c, b, c->t_f32, pr, Af, Bf, acc);
                    acc = pr;
                }
                def_val(c, in->mregs[m], PTY_F32, acc);
            }
            return;
        }
        if (in->mma_m8f16) {
            // m8n8k4 f16: four independent 8x8x4 MMAs, one per lane
            // group. A/B fragments are two packed regs each; C/D are
            // eight f32 regs per lane.
            for (int m = 0; m < 8; m++) {
                if (check_dst(c, in->mregs[m], PTY_F32, line) != 0)
                    return;
            }
            uint32_t Areg[2], Breg[2], Creg[8];
            for (int j = 0; j < 2; j++) {
                ptx_opd_t o;
                memset(&o, 0, sizeof(o));
                o.kind = OPD_REG;
                strcpy(o.reg, in->mregs[8 + j]);
                val_t v = resolve(c, b, &o, PTY_U32, line);
                if (c->failed)
                    return;
                Areg[j] = v.id;
                memset(&o, 0, sizeof(o));
                o.kind = OPD_REG;
                strcpy(o.reg, in->mregs[10 + j]);
                v = resolve(c, b, &o, PTY_U32, line);
                if (c->failed)
                    return;
                Breg[j] = v.id;
            }
            for (int i = 0; i < 8; i++) {
                ptx_opd_t o;
                memset(&o, 0, sizeof(o));
                o.kind = OPD_REG;
                strcpy(o.reg, in->mregs[12 + i]);
                val_t v = resolve(c, b, &o, PTY_F32, line);
                if (c->failed)
                    return;
                Creg[i] = v.id;
            }
            uint32_t sub = nid(c);
            emit_load(c, b, c->t_u32, sub, c->v_subid);
            uint32_t W = nid(c), lane = nid(c);
            emit_arith(c, b, S_ShrL, c->t_u32, W, sub, const_u32(c, 5));
            emit_arith(c, b, S_And, c->t_u32, lane, sub, const_u32(c, 31));
            uint32_t p = nid(c), q = nid(c);
            emit_arith(c, b, S_And, c->t_u32, p, lane, const_u32(c, 1));
            emit_arith(c, b, S_And, c->t_u32, q, lane, const_u32(c, 2));
            uint32_t shc = nid(c), comp = nid(c), comp4 = nid(c);
            uint32_t w5 = nid(c);
            emit_arith(c, b, S_ShrL, c->t_u32, shc, lane,
                       const_u32(c, 2));
            emit_arith(c, b, S_And, c->t_u32, comp, shc,
                       const_u32(c, 3));
            emit_arith(c, b, S_Shl, c->t_u32, comp4, comp, const_u32(c, 2));
            emit_arith(c, b, S_Shl, c->t_u32, w5, W, const_u32(c, 5));
            uint32_t hic = do_cmp(c, b, PCM_GE, PTY_U32, 0, lane,
                                  const_u32(c, 16), line);
            uint32_t hi4 = sel_u32(c, b, hic, const_u32(c, 4),
                                   const_u32(c, 0));
            uint32_t scope = const_u32(c, 3);
            for (int i = 0; i < 8; i++) {
                uint32_t X = nid(c), row = nid(c), col = nid(c);
                emit_arith(c, b, S_IAdd, c->t_u32, X, p,
                           const_u32(c, (uint32_t)(i & 2)));
                emit_arith(c, b, S_IAdd, c->t_u32, row, X, hi4);
                emit_arith(c, b, S_IAdd, c->t_u32, col, q,
                           const_u32(c, (uint32_t)((i & 4) + (i & 1))));
                uint32_t acc = Creg[i];
                for (int k = 0; k < 4; k++) {
                    uint32_t rmod = nid(c), rhic = nid(c);
                    uint32_t al1 = nid(c), alane = nid(c);
                    emit_arith(c, b, S_And, c->t_u32, rmod, row,
                               const_u32(c, 3));
                    rhic = do_cmp(c, b, PCM_GE, PTY_U32, 0, row,
                                  const_u32(c, 4), line);
                    uint32_t rhi = sel_u32(c, b, rhic, const_u32(c, 16),
                                           const_u32(c, 0));
                    emit_arith(c, b, S_IAdd, c->t_u32, al1, comp4, rmod);
                    emit_arith(c, b, S_IAdd, c->t_u32, alane, al1, rhi);
                    uint32_t aSrc = nid(c), aGot = nid(c);
                    emit_arith(c, b, S_Or, c->t_u32, aSrc, w5, alane);
                    put(c, b, (6u << 16) | S_GroupShuffle);
                    put(c, b, c->t_u32);
                    put(c, b, aGot);
                    put(c, b, scope);
                    put(c, b, Areg[k >> 1]);
                    put(c, b, aSrc);
                    uint32_t ah = nid(c);
                    if (k & 1)
                        emit_arith(c, b, S_ShrL, c->t_u32, ah, aGot,
                                   const_u32(c, 16));
                    else
                        emit_arith(c, b, S_And, c->t_u32, ah, aGot,
                                   const_u32(c, 0xffffu));
                    uint32_t Af = f16_to_f32val(c, b, ah, line);
                    uint32_t cmod = nid(c), chic = nid(c);
                    uint32_t bl1 = nid(c), blane = nid(c);
                    emit_arith(c, b, S_And, c->t_u32, cmod, col,
                               const_u32(c, 3));
                    chic = do_cmp(c, b, PCM_GE, PTY_U32, 0, col,
                                  const_u32(c, 4), line);
                    uint32_t chi = sel_u32(c, b, chic, const_u32(c, 16),
                                           const_u32(c, 0));
                    emit_arith(c, b, S_IAdd, c->t_u32, bl1, comp4, cmod);
                    emit_arith(c, b, S_IAdd, c->t_u32, blane, bl1, chi);
                    uint32_t bSrc = nid(c), bGot = nid(c);
                    emit_arith(c, b, S_Or, c->t_u32, bSrc, w5, blane);
                    put(c, b, (6u << 16) | S_GroupShuffle);
                    put(c, b, c->t_u32);
                    put(c, b, bGot);
                    put(c, b, scope);
                    put(c, b, Breg[k >> 1]);
                    put(c, b, bSrc);
                    uint32_t bh = nid(c);
                    if (k & 1)
                        emit_arith(c, b, S_ShrL, c->t_u32, bh, bGot,
                                   const_u32(c, 16));
                    else
                        emit_arith(c, b, S_And, c->t_u32, bh, bGot,
                                   const_u32(c, 0xffffu));
                    uint32_t Bf = f16_to_f32val(c, b, bh, line);
                    if (c->failed)
                        return;
                    uint32_t pr = nid(c);
                    emit_fma_into(c, b, c->t_f32, pr, Af, Bf, acc);
                    acc = pr;
                }
                def_val(c, in->mregs[i], PTY_F32, acc);
            }
            return;
        }
        if (in->mma_elem == 4 && in->mma_f64_m16) {
            // f64 m16n8k4/k8/k16: 4 C/D f64 regs; A lane 4g+(k&3) with
            // reg 2*(k>>2)+row-high; B lane 4c+(k&3) with reg k>>2.
            int kd = in->mma_k;
            int na = kd == 4 ? 2 : kd == 8 ? 4 : 8;
            int nb = kd == 4 ? 1 : kd == 8 ? 2 : 4;
            for (int m = 0; m < 4; m++) {
                if (check_dst(c, in->mregs[m], PTY_F64, line) != 0)
                    return;
            }
            uint32_t Areg[8], Breg[4], Creg[4];
            for (int j = 0; j < na; j++) {
                ptx_opd_t o;
                memset(&o, 0, sizeof(o));
                o.kind = OPD_REG;
                strcpy(o.reg, in->mregs[4 + j]);
                val_t v = resolve(c, b, &o, PTY_F64, line);
                if (c->failed)
                    return;
                Areg[j] = v.id;
            }
            for (int j = 0; j < nb; j++) {
                ptx_opd_t o;
                memset(&o, 0, sizeof(o));
                o.kind = OPD_REG;
                strcpy(o.reg, in->mregs[4 + na + j]);
                val_t v = resolve(c, b, &o, PTY_F64, line);
                if (c->failed)
                    return;
                Breg[j] = v.id;
            }
            for (int j = 0; j < 4; j++) {
                ptx_opd_t o;
                memset(&o, 0, sizeof(o));
                o.kind = OPD_REG;
                strcpy(o.reg, in->mregs[4 + na + nb + j]);
                val_t v = resolve(c, b, &o, PTY_F64, line);
                if (c->failed)
                    return;
                Creg[j] = v.id;
            }
            uint32_t sub = nid(c);
            emit_load(c, b, c->t_u32, sub, c->v_subid);
            uint32_t W = nid(c), g = nid(c), t = nid(c), L = nid(c);
            emit_arith(c, b, S_ShrL, c->t_u32, W, sub, const_u32(c, 5));
            emit_arith(c, b, S_And, c->t_u32, L, sub, const_u32(c, 31));
            emit_arith(c, b, S_ShrL, c->t_u32, g, L, const_u32(c, 2));
            emit_arith(c, b, S_And, c->t_u32, t, L, const_u32(c, 3));
            uint32_t scope = const_u32(c, 3);
            uint32_t g4 = nid(c), t2 = nid(c), w5 = nid(c);
            emit_arith(c, b, S_Shl, c->t_u32, g4, g, const_u32(c, 2));
            emit_arith(c, b, S_Shl, c->t_u32, t2, t, const_u32(c, 1));
            emit_arith(c, b, S_Shl, c->t_u32, w5, W, const_u32(c, 5));
            for (int m = 0; m < 4; m++) {
                int high = m >= 2;
                uint32_t acc = Creg[m];
                uint32_t ccol = nid(c), c4 = nid(c);
                emit_arith(c, b, S_Or, c->t_u32, ccol, t2,
                           const_u32(c, (uint32_t)(m & 1)));
                emit_arith(c, b, S_Shl, c->t_u32, c4, ccol,
                           const_u32(c, 2));
                for (int k = 0; k < kd; k++) {
                    int areg = 2 * (k >> 2) + high;
                    int breg = k >> 2;
                    uint32_t aLane = nid(c), aSrc = nid(c), aGot = nid(c);
                    emit_arith(c, b, S_Or, c->t_u32, aLane, g4,
                               const_u32(c, (uint32_t)(k & 3)));
                    emit_arith(c, b, S_Or, c->t_u32, aSrc, w5, aLane);
                    put(c, b, (6u << 16) | S_GroupShuffle);
                    put(c, b, c->t_f64);
                    put(c, b, aGot);
                    put(c, b, scope);
                    put(c, b, Areg[areg]);
                    put(c, b, aSrc);
                    uint32_t bLane = nid(c), bSrc = nid(c), bGot = nid(c);
                    emit_arith(c, b, S_Or, c->t_u32, bLane, c4,
                               const_u32(c, (uint32_t)(k & 3)));
                    emit_arith(c, b, S_Or, c->t_u32, bSrc, w5, bLane);
                    put(c, b, (6u << 16) | S_GroupShuffle);
                    put(c, b, c->t_f64);
                    put(c, b, bGot);
                    put(c, b, scope);
                    put(c, b, Breg[breg]);
                    put(c, b, bSrc);
                    uint32_t pr;
                    if (in->frnd) {
                        pr = emit_f64_fma_rnd(c, b, aGot, bGot, acc,
                                              in->frnd, line);
                    } else {
                        pr = nid(c);
                        emit_fma_into(c, b, c->t_f64, pr, aGot, bGot, acc);
                    }
                    if (c->failed)
                        return;
                    acc = pr;
                }
                def_val(c, in->mregs[m], PTY_F64, acc);
            }
            return;
        }
        if (in->mma_elem == 4 && !in->mma_f64_m16) {
            // m8n8k4 f64: A lane g4+k, B lane c4+k, one 8x8x4 MMA.
            for (int m = 0; m < 2; m++) {
                if (check_dst(c, in->mregs[m], PTY_F64, line) != 0)
                    return;
            }
            uint32_t Areg, Breg, Creg[2];
            {
                ptx_opd_t o;
                memset(&o, 0, sizeof(o));
                o.kind = OPD_REG;
                strcpy(o.reg, in->mregs[4]);
                val_t v = resolve(c, b, &o, PTY_F64, line);
                if (c->failed)
                    return;
                Areg = v.id;
                memset(&o, 0, sizeof(o));
                o.kind = OPD_REG;
                strcpy(o.reg, in->mregs[8]);
                v = resolve(c, b, &o, PTY_F64, line);
                if (c->failed)
                    return;
                Breg = v.id;
                for (int j = 0; j < 2; j++) {
                    memset(&o, 0, sizeof(o));
                    o.kind = OPD_REG;
                    strcpy(o.reg, in->mregs[10 + j]);
                    v = resolve(c, b, &o, PTY_F64, line);
                    if (c->failed)
                        return;
                    Creg[j] = v.id;
                }
            }
            uint32_t sub = nid(c);
            emit_load(c, b, c->t_u32, sub, c->v_subid);
            uint32_t W = nid(c), g = nid(c), t = nid(c), L = nid(c);
            emit_arith(c, b, S_ShrL, c->t_u32, W, sub, const_u32(c, 5));
            emit_arith(c, b, S_And, c->t_u32, L, sub, const_u32(c, 31));
            emit_arith(c, b, S_ShrL, c->t_u32, g, L, const_u32(c, 2));
            emit_arith(c, b, S_And, c->t_u32, t, L, const_u32(c, 3));
            uint32_t scope = const_u32(c, 3);
            uint32_t g4 = nid(c), t2 = nid(c), w5 = nid(c);
            emit_arith(c, b, S_Shl, c->t_u32, g4, g, const_u32(c, 2));
            emit_arith(c, b, S_Shl, c->t_u32, t2, t, const_u32(c, 1));
            emit_arith(c, b, S_Shl, c->t_u32, w5, W, const_u32(c, 5));
            for (int j = 0; j < 2; j++) {
                uint32_t acc = Creg[j];
                uint32_t ccol = nid(c), c4 = nid(c);
                emit_arith(c, b, S_Or, c->t_u32, ccol, t2,
                           const_u32(c, (uint32_t)j));
                emit_arith(c, b, S_Shl, c->t_u32, c4, ccol,
                           const_u32(c, 2));
                for (int k = 0; k < 4; k++) {
                    uint32_t aLane = nid(c), aSrc = nid(c), aGot = nid(c);
                    emit_arith(c, b, S_Or, c->t_u32, aLane, g4,
                               const_u32(c, (uint32_t)k));
                    emit_arith(c, b, S_Or, c->t_u32, aSrc, w5, aLane);
                    put(c, b, (6u << 16) | S_GroupShuffle);
                    put(c, b, c->t_f64);
                    put(c, b, aGot);
                    put(c, b, scope);
                    put(c, b, Areg);
                    put(c, b, aSrc);
                    uint32_t bLane = nid(c), bSrc = nid(c), bGot = nid(c);
                    emit_arith(c, b, S_Or, c->t_u32, bLane, c4,
                               const_u32(c, (uint32_t)k));
                    emit_arith(c, b, S_Or, c->t_u32, bSrc, w5, bLane);
                    put(c, b, (6u << 16) | S_GroupShuffle);
                    put(c, b, c->t_f64);
                    put(c, b, bGot);
                    put(c, b, scope);
                    put(c, b, Breg);
                    put(c, b, bSrc);
                    uint32_t pr;
                    if (in->frnd) {
                        pr = emit_f64_fma_rnd(c, b, aGot, bGot, acc,
                                              in->frnd, line);
                    } else {
                        pr = nid(c);
                        emit_fma_into(c, b, c->t_f64, pr, aGot, bGot, acc);
                    }
                    if (c->failed)
                        return;
                    acc = pr;
                }
                def_val(c, in->mregs[j], PTY_F64, acc);
            }
            return;
        }
        // m16n8k16 row.col f32.f16/bf16 and m16n8k32 row.col s32.s8
        // warp matmul by shuffle-gather. Fragment maps are the
        // PTX-ISA closed forms (groupID = lane>>2,
        // threadID_in_group = lane&3), verified bit-exact against
        // NVIDIA-generated ZLUDA vectors by exact integer oracles
        // before lowering (see docs/supported-isa.md). Converged warp
        // execution required, as in CUDA.
        // mregs: 4 dst + 4 A + 2 B + 4 C (f32 regs or b32 bits).
        int is_bf16 = in->mma_elem == 1;
        int is_s8 = in->mma_elem == 2;
        int is_int4 = in->mma_elem == 7;
        int is_b1 = in->mma_elem == 8;
        int is_fp8 = in->mma_elem == 5 || in->mma_elem == 6;
        int fp8_e4m3 = in->mma_elem == 5;
        int is_tf32 = in->mma_elem == 3;
        int kdim = in->mma_k ? in->mma_k : 16;
        int f16acc = in->mma_f16acc;
        int na = in->mma_m8s8 ? 1
                 : is_b1 ? (kdim == 256 ? 4 : 2)
                 : is_int4 ? (kdim == 64 ? 4 : 2)
                 : is_tf32 ? (kdim == 4 ? 2 : 4)
                 : is_s8 ? (kdim == 16 ? 2 : 4)
                 : (kdim == 8 || kdim == 4) ? 2 : 4;
        int nb = in->mma_m8s8 ? 1
                 : is_b1 ? (kdim == 256 ? 2 : 1)
                 : is_int4 ? (kdim == 64 ? 2 : 1)
                 : is_tf32 ? (kdim == 4 ? 1 : 2)
                 : is_s8 ? (kdim == 16 ? 1 : 2)
                 : (kdim == 8 || kdim == 4) ? 1 : 2;
        ptx_ty_t accty = (is_s8 || is_int4 || is_b1) ? PTY_U32
                        : f16acc ? PTY_F16X2
                                 : PTY_F32;
        int ndst = in->mma_m8s8 ? 2
                 : (is_int4 || is_b1) ? 4
                 : f16acc ? 2 : 4;
        for (int m = 0; m < ndst; m++) {
            if (check_dst(c, in->mregs[m], accty, line) != 0)
                return;
        }
        uint32_t Areg[4], Breg[2], Creg[4];
        for (int j = 0; j < na; j++) {
            ptx_opd_t o;
            memset(&o, 0, sizeof(o));
            o.kind = OPD_REG;
            strcpy(o.reg, in->mregs[4 + j]);
            val_t v = resolve(c, b, &o, PTY_U32, line);
            if (c->failed)
                return;
            Areg[j] = v.id;
        }
        for (int j = 0; j < nb; j++) {
            ptx_opd_t o;
            memset(&o, 0, sizeof(o));
            o.kind = OPD_REG;
            strcpy(o.reg, in->mregs[8 + j]);
            val_t v = resolve(c, b, &o, PTY_U32, line);
            if (c->failed)
                return;
            Breg[j] = v.id;
        }
        if (f16acc) {
            // C/D are two .f16x2 regs: reg0 = row g cols 2t/2t+1,
            // reg1 = row g+8. Expand to four f32 accumulators.
            uint32_t pk[2];
            for (int j = 0; j < 2; j++) {
                reg_t *cr = find_reg(c, in->mregs[10 + j]);
                if (!cr || !cr->defined) {
                    failf(c, line, "mma f16 accumulator has no value");
                    return;
                }
                pk[j] = cr->cur;
            }
            for (int j = 0; j < 2; j++) {
                uint32_t lo = nid(c), hi = nid(c);
                emit_arith(c, b, S_And, c->t_u32, lo, pk[j],
                           const_u32(c, 0xffffu));
                emit_arith(c, b, S_ShrL, c->t_u32, hi, pk[j],
                           const_u32(c, 16));
                Creg[2 * j] = f16_to_f32val(c, b, lo, line);
                Creg[2 * j + 1] = f16_to_f32val(c, b, hi, line);
                if (c->failed)
                    return;
            }
        } else {
            for (int m = 0; m < ndst; m++) {
                ptx_opd_t o;
                memset(&o, 0, sizeof(o));
                o.kind = OPD_REG;
                strcpy(o.reg, in->mregs[10 + m]);
                val_t v = resolve(c, b, &o, accty, line);
                if (c->failed)
                    return;
                Creg[m] = v.id;
            }
        }
        if (is_b1) {
            uint32_t sub = nid(c), L = nid(c), W = nid(c);
            uint32_t gg = nid(c), tt = nid(c), wbase = nid(c);
            emit_load(c, b, c->t_u32, sub, c->v_subid);
            emit_arith(c, b, S_And, c->t_u32, L, sub, const_u32(c, 31));
            emit_arith(c, b, S_ShrL, c->t_u32, W, sub, const_u32(c, 5));
            emit_arith(c, b, S_ShrL, c->t_u32, gg, L, const_u32(c, 2));
            emit_arith(c, b, S_And, c->t_u32, tt, L, const_u32(c, 3));
            emit_arith(c, b, S_Shl, c->t_u32, wbase, W, const_u32(c, 5));
            int m8 = in->mma_m8s8;
            int nkk = kdim / 32;
            for (int m = 0; m < ndst; m++) {
                uint32_t row = nid(c), col = nid(c), acc = Creg[m];
                if (m8) {
                    row = gg;
                    emit_arith(c, b, S_Shl, c->t_u32, col, tt,
                               const_u32(c, 1));
                    uint32_t cc = nid(c);
                    emit_arith(c, b, S_IAdd, c->t_u32, cc, col,
                               const_u32(c, (uint32_t)m));
                    col = cc;
                } else {
                    uint32_t sh = const_u32(c, m >= 2 ? 8u : 0u);
                    emit_arith(c, b, S_IAdd, c->t_u32, row, gg, sh);
                    uint32_t c0 = nid(c), cc = nid(c);
                    emit_arith(c, b, S_Shl, c->t_u32, c0, tt,
                               const_u32(c, 1));
                    emit_arith(c, b, S_IAdd, c->t_u32, cc, c0,
                               const_u32(c, (uint32_t)(m & 1)));
                    col = cc;
                }
                for (int kk = 0; kk < nkk; kk++) {
                    uint32_t owner = nid(c), base = nid(c);
                    int base_reg = m8 ? 0 : (kk >> 2) * 2;
                    int breg = (m8 || kdim == 128) ? 0 : (kk >> 2);
                    uint32_t rr = nid(c), rowhi = nid(c);
                    emit_arith(c, b, S_And, c->t_u32, rr, row,
                               const_u32(c, 7));
                    emit_arith(c, b, S_ShrL, c->t_u32, rowhi, row,
                               const_u32(c, 3));
                    uint32_t aregval = Areg[base_reg];
                    if (!m8) {
                        uint32_t cond = do_cmp(
                            c, b, PCM_NE, PTY_U32, 0, rowhi,
                            const_u32(c, 0), line);
                        aregval = sel_u32(c, b, cond,
                                          Areg[base_reg + 1],
                                          Areg[base_reg]);
                    }
                    emit_arith(c, b, S_Shl, c->t_u32, base, rr,
                               const_u32(c, 2));
                    emit_arith(c, b, S_IAdd, c->t_u32, owner, base,
                               const_u32(c, (uint32_t)(kk & 3)));
                    uint32_t aword =
                        wmma_shfl(c, b, aregval, owner, wbase);
                    uint32_t cbase = nid(c), bowner = nid(c);
                    emit_arith(c, b, S_Shl, c->t_u32, cbase, col,
                               const_u32(c, 2));
                    emit_arith(c, b, S_IAdd, c->t_u32, bowner, cbase,
                               const_u32(c, (uint32_t)(kk & 3)));
                    uint32_t bword =
                        wmma_shfl(c, b, Breg[breg], bowner, wbase);
                    uint32_t bits = nid(c);
                    emit_arith(c, b,
                               in->mma_b1_xor ? S_Xor : S_And,
                               c->t_u32, bits, aword, bword);
                    uint32_t pc = popc32(c, b, bits);
                    uint32_t sum = nid(c);
                    emit_arith(c, b, S_IAdd, c->t_u32, sum, acc, pc);
                    acc = sum;
                }
                def_val(c, in->mregs[m], PTY_U32, acc);
            }
            return;
        }
        if (is_int4) {
            uint32_t sub = nid(c), L = nid(c), W = nid(c);
            uint32_t gg = nid(c), tt = nid(c), wbase = nid(c);
            emit_load(c, b, c->t_u32, sub, c->v_subid);
            emit_arith(c, b, S_And, c->t_u32, L, sub, const_u32(c, 31));
            emit_arith(c, b, S_ShrL, c->t_u32, W, sub, const_u32(c, 5));
            emit_arith(c, b, S_ShrL, c->t_u32, gg, L, const_u32(c, 2));
            emit_arith(c, b, S_And, c->t_u32, tt, L, const_u32(c, 3));
            emit_arith(c, b, S_Shl, c->t_u32, wbase, W, const_u32(c, 5));
            int m8 = in->mma_m8s8;
            int nkk = kdim / 8;
            for (int m = 0; m < ndst; m++) {
                uint32_t row = nid(c), col = nid(c), acc = Creg[m];
                if (m8) {
                    row = gg;
                    emit_arith(c, b, S_Shl, c->t_u32, col, tt,
                               const_u32(c, 1));
                    uint32_t cc = nid(c);
                    emit_arith(c, b, S_IAdd, c->t_u32, cc, col,
                               const_u32(c, (uint32_t)m));
                    col = cc;
                } else {
                    uint32_t sh = const_u32(c, m >= 2 ? 8u : 0u);
                    emit_arith(c, b, S_IAdd, c->t_u32, row, gg, sh);
                    uint32_t c0 = nid(c), cc = nid(c);
                    emit_arith(c, b, S_Shl, c->t_u32, c0, tt,
                               const_u32(c, 1));
                    emit_arith(c, b, S_IAdd, c->t_u32, cc, c0,
                               const_u32(c, (uint32_t)(m & 1)));
                    col = cc;
                }
                for (int kk = 0; kk < nkk; kk++) {
                    uint32_t owner = nid(c), base = nid(c);
                    int base_reg = m8 ? 0 : (kk >> 2) * 2;
                    int breg = (m8 || kdim == 32) ? 0 : (kk >> 2);
                    uint32_t rr = nid(c), rowhi = nid(c);
                    emit_arith(c, b, S_And, c->t_u32, rr, row,
                               const_u32(c, 7));
                    emit_arith(c, b, S_ShrL, c->t_u32, rowhi, row,
                               const_u32(c, 3));
                    uint32_t aregval = Areg[base_reg];
                    if (!m8) {
                        uint32_t cond = do_cmp(
                            c, b, PCM_NE, PTY_U32, 0, rowhi,
                            const_u32(c, 0), line);
                        aregval = sel_u32(c, b, cond,
                                          Areg[base_reg + 1],
                                          Areg[base_reg]);
                    }
                    emit_arith(c, b, S_Shl, c->t_u32, base, rr,
                               const_u32(c, 2));
                    emit_arith(c, b, S_IAdd, c->t_u32, owner, base,
                               const_u32(c, (uint32_t)(kk & 3)));
                    uint32_t aword =
                        wmma_shfl(c, b, aregval, owner, wbase);
                    uint32_t cbase = nid(c), bowner = nid(c);
                    emit_arith(c, b, S_Shl, c->t_u32, cbase, col,
                               const_u32(c, 2));
                    emit_arith(c, b, S_IAdd, c->t_u32, bowner, cbase,
                               const_u32(c, (uint32_t)(kk & 3)));
                    uint32_t bword =
                        wmma_shfl(c, b, Breg[breg], bowner, wbase);
                    uint32_t ab[2], bb[2];
                    unpack4_to_bytes(c, b, aword, in->mma_asign, ab);
                    unpack4_to_bytes(c, b, bword, in->mma_bsign, bb);
                    for (int h = 0; h < 2; h++) {
                        uint32_t dot = dot4_packed(
                            c, b, in->mma_asign, in->mma_bsign, ab[h],
                            bb[h]);
                        uint32_t sum = nid(c);
                        emit_arith(c, b, S_IAdd, c->t_u32, sum, acc, dot);
                        acc = sum;
                    }
                }
                def_val(c, in->mregs[m], PTY_U32, acc);
            }
            return;
        }
        uint32_t sub = nid(c);
        emit_load(c, b, c->t_u32, sub, c->v_subid);
        uint32_t W = nid(c), myG = nid(c), myT = nid(c);
        emit_arith(c, b, S_ShrL, c->t_u32, W, sub, const_u32(c, 5));
        {
            uint32_t L = nid(c);
            emit_arith(c, b, S_And, c->t_u32, L, sub, const_u32(c, 31));
            emit_arith(c, b, S_ShrL, c->t_u32, myG, L, const_u32(c, 2));
            emit_arith(c, b, S_And, c->t_u32, myT, L, const_u32(c, 3));
        }
        uint32_t scope = const_u32(c, 3); // Subgroup
        // 4*myG and 2*myT shared by all lane computations below;
        // w5 = W<<5 so sources pack as (W<<5)|lane (32-lane warps
        // mapped into the hardware subgroup, like shfl/match).
        uint32_t g4 = nid(c), t2 = nid(c), w5 = nid(c);
        emit_arith(c, b, S_Shl, c->t_u32, g4, myG, const_u32(c, 2));
        emit_arith(c, b, S_Shl, c->t_u32, t2, myT, const_u32(c, 1));
        emit_arith(c, b, S_Shl, c->t_u32, w5, W, const_u32(c, 5));
        if (in->mma_m8s8) {
            // m8n8k16 s8/u8: A row g, col 4*t+i; B row 4*t+i, col g;
            // C/D row g, col 2*t+m (two accumulators).
            uint32_t Dreg[2];
            for (int m = 0; m < 2; m++) {
                uint32_t acc = Creg[m];
                uint32_t ccol = nid(c), c4 = nid(c);
                emit_arith(c, b, S_Or, c->t_u32, ccol, t2,
                           const_u32(c, (uint32_t)m));
                emit_arith(c, b, S_Shl, c->t_u32, c4, ccol,
                           const_u32(c, 2));
                if (c->use_dot_product) {
                    for (int kg = 0; kg < 16; kg += 4) {
                        int j = kg >> 2;
                        uint32_t aLane = nid(c), aSrc = nid(c);
                        emit_arith(c, b, S_Or, c->t_u32, aLane, g4,
                                   const_u32(c, (uint32_t)j));
                        emit_arith(c, b, S_Or, c->t_u32, aSrc, w5, aLane);
                        uint32_t aGot = nid(c);
                        put(c, b, (6u << 16) | S_GroupShuffle);
                        put(c, b, c->t_u32);
                        put(c, b, aGot);
                        put(c, b, scope);
                        put(c, b, Areg[0]);
                        put(c, b, aSrc);
                        uint32_t bLane = nid(c), bSrc = nid(c);
                        emit_arith(c, b, S_Or, c->t_u32, bLane, c4,
                                   const_u32(c, (uint32_t)j));
                        emit_arith(c, b, S_Or, c->t_u32, bSrc, w5, bLane);
                        uint32_t bGot = nid(c);
                        put(c, b, (6u << 16) | S_GroupShuffle);
                        put(c, b, c->t_u32);
                        put(c, b, bGot);
                        put(c, b, scope);
                        put(c, b, Breg[0]);
                        put(c, b, bSrc);
                        uint32_t dot = dot4_packed(c, b, in->mma_asign,
                                                   in->mma_bsign, aGot,
                                                   bGot);
                        uint32_t nx = nid(c);
                        emit_arith(c, b, S_IAdd, c->t_u32, nx, acc, dot);
                        acc = nx;
                    }
                } else {
for (int k = 0; k < 16; k++) {
                    uint32_t aLane = nid(c), aSrc = nid(c), aGot = nid(c);
                    emit_arith(c, b, S_Or, c->t_u32, aLane, g4,
                               const_u32(c, (uint32_t)(k >> 2)));
                    emit_arith(c, b, S_Or, c->t_u32, aSrc, w5, aLane);
                    put(c, b, (6u << 16) | S_GroupShuffle);
                    put(c, b, c->t_u32);
                    put(c, b, aGot);
                    put(c, b, scope);
                    put(c, b, Areg[0]);
                    put(c, b, aSrc);
                    uint32_t aBy = nid(c), aSx = nid(c);
                    emit_arith(c, b, S_ShrL, c->t_u32, aBy, aGot,
                               const_u32(c, (uint32_t)(8 * (k & 3))));
                    emit_arith(c, b, S_And, c->t_u32, aSx, aBy,
                               const_u32(c, 0xffu));
                    uint32_t As = aSx;
                    if (in->mma_asign) {
                        uint32_t ai = nid(c);
                        emit_arith(c, b, S_Shl, c->t_u32, ai, aSx,
                                   const_u32(c, 24));
                        As = nid(c);
                        emit_arith(c, b, S_ShrA, c->t_u32, As, ai,
                                   const_u32(c, 24));
                    }
                    uint32_t bLane = nid(c), bSrc = nid(c), bGot = nid(c);
                    emit_arith(c, b, S_Or, c->t_u32, bLane, c4,
                               const_u32(c, (uint32_t)(k >> 2)));
                    emit_arith(c, b, S_Or, c->t_u32, bSrc, w5, bLane);
                    put(c, b, (6u << 16) | S_GroupShuffle);
                    put(c, b, c->t_u32);
                    put(c, b, bGot);
                    put(c, b, scope);
                    put(c, b, Breg[0]);
                    put(c, b, bSrc);
                    uint32_t bBy = nid(c), bSx = nid(c);
                    emit_arith(c, b, S_ShrL, c->t_u32, bBy, bGot,
                               const_u32(c, (uint32_t)(8 * (k & 3))));
                    emit_arith(c, b, S_And, c->t_u32, bSx, bBy,
                               const_u32(c, 0xffu));
                    uint32_t Bs = bSx;
                    if (in->mma_bsign) {
                        uint32_t bi = nid(c);
                        emit_arith(c, b, S_Shl, c->t_u32, bi, bSx,
                                   const_u32(c, 24));
                        Bs = nid(c);
                        emit_arith(c, b, S_ShrA, c->t_u32, Bs, bi,
                                   const_u32(c, 24));
                    }
                    uint32_t pr = nid(c), nx = nid(c);
                    emit_arith(c, b, S_IMul, c->t_u32, pr, As, Bs);
                    emit_arith(c, b, S_IAdd, c->t_u32, nx, acc, pr);
                    acc = nx;
                }
                }
                Dreg[m] = acc;
            }
            def_val(c, in->mregs[0], PTY_U32, Dreg[0]);
            def_val(c, in->mregs[1], PTY_U32, Dreg[1]);
            return;
        }
        uint32_t accs[4];
        if (!is_s8 && !is_fp8 && !is_tf32 && kdim == 16) {
            // f16/bf16 m16n8k16: process a pair of K values per 32-bit
            // fragment register. One shuffle per A row-register and one
            // per B column-register feeds all four output accumulators,
            // instead of re-shuffling/re-converting per output.
            for (int m = 0; m < 4; m++)
                accs[m] = Creg[m];
            int use_dot2 = c->caps.mixed_fdot && !is_bf16 && !f16acc;
            for (int kp = 0; kp < 8; kp++) {
                int laneoff = kp & 3;
                int areg = (kp >= 4 ? 2 : 0);
                int breg = (kp >= 4 ? 1 : 0);
                uint32_t Af[4], Bf[4], Araw[2] = {0, 0}, Braw[2] = {0, 0};
                for (int r = 0; r < 2; r++) {
                    uint32_t aLane = nid(c), aSrc = nid(c);
                    emit_arith(c, b, S_Or, c->t_u32, aLane, g4,
                               const_u32(c, (uint32_t)laneoff));
                    emit_arith(c, b, S_Or, c->t_u32, aSrc, w5, aLane);
                    uint32_t aGot = nid(c);
                    put(c, b, (6u << 16) | S_GroupShuffle);
                    put(c, b, c->t_u32);
                    put(c, b, aGot);
                    put(c, b, scope);
                    put(c, b, Areg[areg + r]);
                    put(c, b, aSrc);
                    if (use_dot2) {
                        Araw[r] = aGot;
                        continue;
                    }
                    uint32_t lo = nid(c), hi = nid(c);
                    emit_arith(c, b, S_And, c->t_u32, lo, aGot,
                               const_u32(c, 0xffffu));
                    emit_arith(c, b, S_ShrL, c->t_u32, hi, aGot,
                               const_u32(c, 16));
                    Af[2 * r] = is_bf16
                                    ? bf16_to_f32val(c, b, lo, line)
                                    : f16_to_f32val(c, b, lo, line);
                    Af[2 * r + 1] = is_bf16
                                        ? bf16_to_f32val(c, b, hi, line)
                                        : f16_to_f32val(c, b, hi, line);
                    if (c->failed)
                        return;
                }
                for (int cc = 0; cc < 2; cc++) {
                    uint32_t ccol = nid(c), c4 = nid(c);
                    emit_arith(c, b, S_Or, c->t_u32, ccol, t2,
                               const_u32(c, (uint32_t)cc));
                    emit_arith(c, b, S_Shl, c->t_u32, c4, ccol,
                               const_u32(c, 2));
                    uint32_t bLane = nid(c), bSrc = nid(c);
                    emit_arith(c, b, S_Or, c->t_u32, bLane, c4,
                               const_u32(c, (uint32_t)laneoff));
                    emit_arith(c, b, S_Or, c->t_u32, bSrc, w5, bLane);
                    uint32_t bGot = nid(c);
                    put(c, b, (6u << 16) | S_GroupShuffle);
                    put(c, b, c->t_u32);
                    put(c, b, bGot);
                    put(c, b, scope);
                    put(c, b, Breg[breg]);
                    put(c, b, bSrc);
                    if (use_dot2) {
                        Braw[cc] = bGot;
                        continue;
                    }
                    uint32_t lo = nid(c), hi = nid(c);
                    emit_arith(c, b, S_And, c->t_u32, lo, bGot,
                               const_u32(c, 0xffffu));
                    emit_arith(c, b, S_ShrL, c->t_u32, hi, bGot,
                               const_u32(c, 16));
                    Bf[2 * cc] = is_bf16
                                     ? bf16_to_f32val(c, b, lo, line)
                                     : f16_to_f32val(c, b, lo, line);
                    Bf[2 * cc + 1] =
                        is_bf16 ? bf16_to_f32val(c, b, hi, line)
                                : f16_to_f32val(c, b, hi, line);
                    if (c->failed)
                        return;
                }
                for (int m = 0; m < 4; m++) {
                    int ar = (m >= 2) ? 1 : 0;
                    int bc = (m & 1) ? 1 : 0;
                    if (use_dot2) {
                        accs[m] = emit_fdot2_f32(c, b, Araw[ar], Braw[bc],
                                            accs[m]);
                        continue;
                    }
                    uint32_t n0 = nid(c), n1 = nid(c);
                    emit_fma_into(c, b, c->t_f32, n0, Af[2 * ar],
                                  Bf[2 * bc], accs[m]);
                    emit_fma_into(c, b, c->t_f32, n1, Af[2 * ar + 1],
                                  Bf[2 * bc + 1], n0);
                    accs[m] = n1;
                }
            }
        } else {
        for (int m = 0; m < 4; m++) {
            // Column of my m-th accumulator: c = 2*myT + (m&1);
            // A rows come from my group (r&7 = myG), B column is c.
            uint32_t acc = Creg[m];
            uint32_t ccol = nid(c), c4 = nid(c);
            emit_arith(c, b, S_Or, c->t_u32, ccol, t2,
                       const_u32(c, (uint32_t)(m & 1)));
            emit_arith(c, b, S_Shl, c->t_u32, c4, ccol, const_u32(c, 2));
            if (is_tf32) {
            for (int k = 0; k < kdim; k++) {
                // tf32 K=4/K=8: A lane g4+(k&3), register selected by
                // the K half and the accumulator row (g or g+8); B lane
                // c4+(k&3), register selected by the K half.  The low
                // 13 mantissa bits are not part of tf32.
                int high = m >= 2;
                uint32_t aLane = nid(c), aSrc = nid(c), aGot = nid(c);
                emit_arith(c, b, S_Or, c->t_u32, aLane, g4,
                           const_u32(c, (uint32_t)(k & 3)));
                emit_arith(c, b, S_Or, c->t_u32, aSrc, w5, aLane);
                put(c, b, (6u << 16) | S_GroupShuffle);
                put(c, b, c->t_u32);
                put(c, b, aGot);
                put(c, b, scope);
                put(c, b, Areg[(k >> 2) * 2 + high]);
                put(c, b, aSrc);
                uint32_t ab = nid(c), Af = nid(c);
                emit_arith(c, b, S_And, c->t_u32, ab, aGot,
                           const_u32(c, 0xffffe000u));
                put(c, b, (4u << 16) | S_Bitcast);
                put(c, b, c->t_f32);
                put(c, b, Af);
                put(c, b, ab);
                uint32_t bLane = nid(c), bSrc = nid(c), bGot = nid(c);
                emit_arith(c, b, S_Or, c->t_u32, bLane, c4,
                           const_u32(c, (uint32_t)(k & 3)));
                emit_arith(c, b, S_Or, c->t_u32, bSrc, w5, bLane);
                put(c, b, (6u << 16) | S_GroupShuffle);
                put(c, b, c->t_u32);
                put(c, b, bGot);
                put(c, b, scope);
                put(c, b, Breg[k >> 2]);
                put(c, b, bSrc);
                uint32_t bb = nid(c), Bf = nid(c);
                emit_arith(c, b, S_And, c->t_u32, bb, bGot,
                           const_u32(c, 0xffffe000u));
                put(c, b, (4u << 16) | S_Bitcast);
                put(c, b, c->t_f32);
                put(c, b, Bf);
                put(c, b, bb);
                uint32_t pr = nid(c);
                emit_fma_into(c, b, c->t_f32, pr, Af, Bf, acc);
                acc = pr;
            }
            } else if (kdim == 8) {
            for (int k = 0; k < 8; k++) {
                // K=8 f16/bf16: A lane g4+(k>>1), reg a0/a1 selected
                // by the accumulator row, half k&1; B lane c4+(k>>1),
                // single b0 register, half k&1.
                int high = m >= 2;
                uint32_t aLane = nid(c), aSrc = nid(c), aGot = nid(c);
                emit_arith(c, b, S_Or, c->t_u32, aLane, g4,
                           const_u32(c, (uint32_t)(k >> 1)));
                emit_arith(c, b, S_Or, c->t_u32, aSrc, w5, aLane);
                put(c, b, (6u << 16) | S_GroupShuffle);
                put(c, b, c->t_u32);
                put(c, b, aGot);
                put(c, b, scope);
                put(c, b, Areg[high]);
                put(c, b, aSrc);
                uint32_t aHalf = nid(c);
                if (k & 1)
                    emit_arith(c, b, S_ShrL, c->t_u32, aHalf, aGot,
                               const_u32(c, 16));
                else
                    emit_arith(c, b, S_And, c->t_u32, aHalf, aGot,
                               const_u32(c, 0xffffu));
                uint32_t Af = is_bf16 ? bf16_to_f32val(c, b, aHalf, line)
                                      : f16_to_f32val(c, b, aHalf, line);
                if (c->failed)
                    return;
                uint32_t bLane = nid(c), bSrc = nid(c), bGot = nid(c);
                emit_arith(c, b, S_Or, c->t_u32, bLane, c4,
                           const_u32(c, (uint32_t)(k >> 1)));
                emit_arith(c, b, S_Or, c->t_u32, bSrc, w5, bLane);
                put(c, b, (6u << 16) | S_GroupShuffle);
                put(c, b, c->t_u32);
                put(c, b, bGot);
                put(c, b, scope);
                put(c, b, Breg[0]);
                put(c, b, bSrc);
                uint32_t bHalf = nid(c);
                if (k & 1)
                    emit_arith(c, b, S_ShrL, c->t_u32, bHalf, bGot,
                               const_u32(c, 16));
                else
                    emit_arith(c, b, S_And, c->t_u32, bHalf, bGot,
                               const_u32(c, 0xffffu));
                uint32_t Bf = is_bf16 ? bf16_to_f32val(c, b, bHalf, line)
                                      : f16_to_f32val(c, b, bHalf, line);
                if (c->failed)
                    return;
                uint32_t pr = nid(c);
                emit_fma_into(c, b, c->t_f32, pr, Af, Bf, acc);
                acc = pr;
            }
            } else if (is_fp8) {
            // FP8 e4m3/e5m2 m16n8k32: same 8-bit fragment layout as
            // the s8 form.  Each byte widens to f16 and accumulates in
            // f32.
            for (int k = 0; k < kdim; k++) {
                int aElem = (k & 3) + (k >= 16 ? 8 : 0) +
                            (m >= 2 ? 4 : 0);
                uint32_t aLane = nid(c), aSrc = nid(c);
                emit_arith(c, b, S_Or, c->t_u32, aLane, g4,
                           const_u32(c, (uint32_t)((k & 15) >> 2)));
                emit_arith(c, b, S_Or, c->t_u32, aSrc, w5, aLane);
                uint32_t aGot = nid(c);
                put(c, b, (6u << 16) | S_GroupShuffle);
                put(c, b, c->t_u32);
                put(c, b, aGot);
                put(c, b, scope);
                put(c, b, Areg[aElem >> 2]);
                put(c, b, aSrc);
                uint32_t aSh = nid(c), aByte = nid(c);
                emit_arith(c, b, S_ShrL, c->t_u32, aSh, aGot,
                           const_u32(c, (uint32_t)(8 * (aElem & 3))));
                emit_arith(c, b, S_And, c->t_u32, aByte, aSh,
                           const_u32(c, 0xffu));
                uint32_t aHalf = f16_from_fp8byte(c, b, aByte,
                                                  fp8_e4m3, line);
                if (c->failed)
                    return;
                uint32_t Af = f16_to_f32val(c, b, aHalf, line);
                if (c->failed)
                    return;
                int bElem = (k & 3) + (k >= 16 ? 4 : 0);
                uint32_t bLane = nid(c), bSrc = nid(c);
                emit_arith(c, b, S_Or, c->t_u32, bLane, c4,
                           const_u32(c, (uint32_t)((k & 15) >> 2)));
                emit_arith(c, b, S_Or, c->t_u32, bSrc, w5, bLane);
                uint32_t bGot = nid(c);
                put(c, b, (6u << 16) | S_GroupShuffle);
                put(c, b, c->t_u32);
                put(c, b, bGot);
                put(c, b, scope);
                put(c, b, Breg[bElem >> 2]);
                put(c, b, bSrc);
                uint32_t bSh = nid(c), bByte = nid(c);
                emit_arith(c, b, S_ShrL, c->t_u32, bSh, bGot,
                           const_u32(c, (uint32_t)(8 * (bElem & 3))));
                emit_arith(c, b, S_And, c->t_u32, bByte, bSh,
                           const_u32(c, 0xffu));
                uint32_t bHalf = f16_from_fp8byte(c, b, bByte,
                                                  fp8_e4m3, line);
                if (c->failed)
                    return;
                uint32_t Bf = f16_to_f32val(c, b, bHalf, line);
                if (c->failed)
                    return;
                uint32_t pr = nid(c);
                emit_fma_into(c, b, c->t_f32, pr, Af, Bf, acc);
                acc = pr;
            }
            } else if (!is_s8) {
            for (int k = 0; k < 16; k++) {
                // A[r][k]: the m-th accumulator is row g + 8*(m>=2) and
                // column 2*myT + (m&1) (PTX-ISA C/D map: c0,c1 -> row
                // groupID; c2,c3 -> row groupID+8). A element index
                // (low to high) i has row groupID + 8*((i>>1)&1) and
                // col 2*myT + (i&1) + 8*(i>=4), so i = 2*(m>=2) +
                // 4*(k>=8) + (k&1) and the owning lane is
                // 4*myG + ((k>>1)&3).
                int aElem = (k & 1) + (k >= 8 ? 4 : 0) +
                            (m >= 2 ? 2 : 0);
                uint32_t aLane = nid(c), aSrc = nid(c);
                emit_arith(c, b, S_Or, c->t_u32, aLane, g4,
                           const_u32(c, (uint32_t)((k & 7) >> 1)));
                emit_arith(c, b, S_Or, c->t_u32, aSrc, w5, aLane);
                uint32_t aGot = nid(c);
                put(c, b, (6u << 16) | S_GroupShuffle);
                put(c, b, c->t_u32);
                put(c, b, aGot);
                put(c, b, scope);
                put(c, b, Areg[aElem >> 1]);
                put(c, b, aSrc);
                uint32_t aHalf = nid(c);
                if (aElem & 1) {
                    emit_arith(c, b, S_ShrL, c->t_u32, aHalf, aGot,
                               const_u32(c, 16));
                } else {
                    emit_arith(c, b, S_And, c->t_u32, aHalf, aGot,
                               const_u32(c, 0xffffu));
                }
                uint32_t Af = is_bf16 ? bf16_to_f32val(c, b, aHalf, line)
                                      : f16_to_f32val(c, b, aHalf, line);
                if (c->failed)
                    return;
                // B[k][c]: lane 4*c + ((k&7)>>1), element
                // (k&1) + (k>=8 ? 2 : 0).
                int bElem = (k & 1) + (k >= 8 ? 2 : 0);
                uint32_t bLane = nid(c), bSrc = nid(c);
                emit_arith(c, b, S_Or, c->t_u32, bLane, c4,
                           const_u32(c, (uint32_t)((k & 7) >> 1)));
                emit_arith(c, b, S_Or, c->t_u32, bSrc, w5, bLane);
                uint32_t bGot = nid(c);
                put(c, b, (6u << 16) | S_GroupShuffle);
                put(c, b, c->t_u32);
                put(c, b, bGot);
                put(c, b, scope);
                put(c, b, Breg[bElem >> 1]);
                put(c, b, bSrc);
                uint32_t bHalf = nid(c);
                if (bElem & 1) {
                    emit_arith(c, b, S_ShrL, c->t_u32, bHalf, bGot,
                               const_u32(c, 16));
                } else {
                    emit_arith(c, b, S_And, c->t_u32, bHalf, bGot,
                               const_u32(c, 0xffffu));
                }
                uint32_t Bf = is_bf16 ? bf16_to_f32val(c, b, bHalf, line)
                                      : f16_to_f32val(c, b, bHalf, line);
                if (c->failed)
                    return;
                uint32_t pr = nid(c);
                emit_fma_into(c, b, c->t_f32, pr, Af, Bf, acc);
                acc = pr;
            }
            } else if (c->use_dot_product) {
            // s8 K=16/K=32: one packed dot per four K values. A and B
            // bytes for a K group live in one register in the owning
            // lanes, so four MACs and four shuffles become one OpSDot
            // plus one shuffle per operand.
            for (int kg = 0; kg < kdim; kg += 4) {
                int j = kg >> 2;
                int areg = (j >= 4 ? 2 : 0) + (m >= 2 ? 1 : 0);
                int breg = (j >= 4 ? 1 : 0);
                uint32_t aLane = nid(c), aSrc = nid(c);
                emit_arith(c, b, S_Or, c->t_u32, aLane, g4,
                           const_u32(c, (uint32_t)(j & 3)));
                emit_arith(c, b, S_Or, c->t_u32, aSrc, w5, aLane);
                uint32_t aGot = nid(c);
                put(c, b, (6u << 16) | S_GroupShuffle);
                put(c, b, c->t_u32);
                put(c, b, aGot);
                put(c, b, scope);
                put(c, b, Areg[areg]);
                put(c, b, aSrc);
                uint32_t bLane = nid(c), bSrc = nid(c);
                emit_arith(c, b, S_Or, c->t_u32, bLane, c4,
                           const_u32(c, (uint32_t)(j & 3)));
                emit_arith(c, b, S_Or, c->t_u32, bSrc, w5, bLane);
                uint32_t bGot = nid(c);
                put(c, b, (6u << 16) | S_GroupShuffle);
                put(c, b, c->t_u32);
                put(c, b, bGot);
                put(c, b, scope);
                put(c, b, Breg[breg]);
                put(c, b, bSrc);
                uint32_t dot = dot4_packed(c, b, in->mma_asign,
                                           in->mma_bsign, aGot, bGot);
                uint32_t nx = nid(c);
                emit_arith(c, b, S_IAdd, c->t_u32, nx, acc, dot);
                acc = nx;
            }
            } else {
            // s8 K=16/K=32 loop. A[r][k]: the m-th accumulator is row
            // g + 8*(m>=2), column 2*myT + (m&1). A element index i
            // (low to high) has row groupID + 8*((i>>2)&1) and col
            // 4*myT + (i&3) + 16*(i>=8), so i = 4*((m>=2) +
            // 2*(k>=16)) + (k&3) and the owning lane is
            // 4*myG + ((k>>2)&3). B[k][c]: register (k>=16), byte
            // k&3, lane 4*c + ((k>>2)&3) (groupID is the N column).
            // Bytes sign-extend; exact s32 IMul/IAdd chains (same
            // forms the oracle verified).
            for (int k = 0; k < kdim; k++) {
                int aElem = (k & 3) + (k >= 16 ? 8 : 0) +
                            (m >= 2 ? 4 : 0);
                uint32_t aLane = nid(c), aSrc = nid(c);
                emit_arith(c, b, S_Or, c->t_u32, aLane, g4,
                           const_u32(c, (uint32_t)((k & 15) >> 2)));
                emit_arith(c, b, S_Or, c->t_u32, aSrc, w5, aLane);
                uint32_t aGot = nid(c);
                put(c, b, (6u << 16) | S_GroupShuffle);
                put(c, b, c->t_u32);
                put(c, b, aGot);
                put(c, b, scope);
                put(c, b, Areg[aElem >> 2]);
                put(c, b, aSrc);
                uint32_t aBy = nid(c), aSx = nid(c);
                emit_arith(c, b, S_ShrL, c->t_u32, aBy, aGot,
                           const_u32(c, (uint32_t)(8 * (aElem & 3))));
                emit_arith(c, b, S_And, c->t_u32, aSx, aBy,
                           const_u32(c, 0xffu));
                uint32_t As = aSx;
                if (in->mma_asign) {
                    uint32_t Ai = nid(c);
                    emit_arith(c, b, S_Shl, c->t_u32, Ai, aSx,
                               const_u32(c, 24));
                    As = nid(c);
                    emit_arith(c, b, S_ShrA, c->t_u32, As, Ai,
                               const_u32(c, 24));
                }
                int bElem = (k & 3) + (k >= 16 ? 4 : 0);
                uint32_t bLane = nid(c), bSrc = nid(c);
                emit_arith(c, b, S_Or, c->t_u32, bLane, c4,
                           const_u32(c, (uint32_t)((k & 15) >> 2)));
                emit_arith(c, b, S_Or, c->t_u32, bSrc, w5, bLane);
                uint32_t bGot = nid(c);
                put(c, b, (6u << 16) | S_GroupShuffle);
                put(c, b, c->t_u32);
                put(c, b, bGot);
                put(c, b, scope);
                put(c, b, Breg[bElem >> 2]);
                put(c, b, bSrc);
                uint32_t bBy = nid(c), bSx = nid(c);
                emit_arith(c, b, S_ShrL, c->t_u32, bBy, bGot,
                           const_u32(c, (uint32_t)(8 * (bElem & 3))));
                emit_arith(c, b, S_And, c->t_u32, bSx, bBy,
                           const_u32(c, 0xffu));
                uint32_t Bs = bSx;
                if (in->mma_bsign) {
                    uint32_t Bi = nid(c);
                    emit_arith(c, b, S_Shl, c->t_u32, Bi, bSx,
                               const_u32(c, 24));
                    Bs = nid(c);
                    emit_arith(c, b, S_ShrA, c->t_u32, Bs, Bi,
                               const_u32(c, 24));
                }
                uint32_t pr = nid(c), na = nid(c);
                emit_arith(c, b, S_IMul, c->t_u32, pr, As, Bs);
                emit_arith(c, b, S_IAdd, c->t_u32, na, acc, pr);
                acc = na;
            }
            }
            accs[m] = acc;
        }
        }
        if (f16acc) {
            for (int j = 0; j < 2; j++) {
                uint32_t lo = f32_to_f16bits(c, b, accs[2 * j], line);
                uint32_t hi = f32_to_f16bits(c, b, accs[2 * j + 1], line);
                if (c->failed)
                    return;
                uint32_t sh = nid(c), pk = nid(c);
                emit_arith(c, b, S_Shl, c->t_u32, sh, hi,
                           const_u32(c, 16));
                emit_arith(c, b, S_Or, c->t_u32, pk, lo, sh);
                def_val(c, in->mregs[j], PTY_F16X2, pk);
            }
        } else {
            for (int m = 0; m < 4; m++)
                def_val(c, in->mregs[m], PTY_F32, accs[m]);
        }
        return;
    }
