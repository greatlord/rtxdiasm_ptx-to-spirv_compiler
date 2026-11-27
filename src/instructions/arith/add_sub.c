#include "instructions/arith/arith.h"

void emit_add_sub_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2;
    uint32_t id;
        int is_add = in->op == POP_ADD;
        if (in->cc_in || in->cc_out) {
            // Extended arithmetic over 32/64-bit integers. CF is stored
            // with PTX semantics: carry-out for add/mad, borrow-out for
            // sub. addc/madc add CF, subc subtracts CF.
            if (in->ty != PTY_U32 && in->ty != PTY_U64) {
                failf(c, line, "extended arithmetic supports .u32/.s32/.u64/.s64");
                return;
            }
            if (check_dst(c, in->dst, in->ty, line) != 0)
                return;
            a = resolve(c, b, &in->src[0], in->ty, line);
            c2 = resolve(c, b, &in->src[1], in->ty, line);
            if (c->failed)
                return;
            int w64 = in->ty == PTY_U64;
            uint32_t wa, wb;
            if (w64) {
                wa = a.id;
                wb = c2.id;
            } else {
                wa = nid(c);
                wb = nid(c);
                put(c, b, (4u << 16) | S_UConvert);
                put(c, b, c->t_u64);
                put(c, b, wa);
                put(c, b, a.id);
                put(c, b, (4u << 16) | S_UConvert);
                put(c, b, c->t_u64);
                put(c, b, wb);
                put(c, b, c2.id);
            }
            uint32_t cin = const_u64(c, 0);
            if (in->cc_in) {
                if (!c->cc_defined) {
                    failf(c, line, "carry flag read before any .cc write");
                    return;
                }
                cin = sel_u64(c, b, c->cc_cf, const_u64(c, 1),
                              const_u64(c, 0));
            }
            uint32_t wide = 0, flag = 0;
            if (is_add) {
                flag = add_carry_u64(c, b, w64, wa, wb, cin, &wide, line);
                if (c->failed)
                    return;
            } else {
                flag = sub_borrow_u64(c, b, w64, wa, wb, cin, &wide, line);
                if (c->failed)
                    return;
            }
            if (w64) {
                id = wide;
            } else {
                id = nid(c);
                put(c, b, (4u << 16) | S_UConvert);
                put(c, b, c->t_u32);
                put(c, b, id);
                put(c, b, wide);
            }
            if (in->cc_out) {
                c->cc_cf = flag;
                c->cc_defined = 1;
            }
            def_val(c, in->dst, in->ty, id);
            return;
        }
        if (in->f32x2) {
            if (check_dst(c, in->dst, PTY_U64, line) != 0)
                return;
            a = resolve(c, b, &in->src[0], PTY_U64, line);
            c2 = resolve(c, b, &in->src[1], PTY_U64, line);
            if (c->failed)
                return;
            uint32_t alo, ahi, blo, bhi;
            f32x2_split(c, b, a.id, &alo, &ahi);
            f32x2_split(c, b, c2.id, &blo, &bhi);
            if (c->failed)
                return;
            uint32_t rlo = f32_addsub_lane(c, b, is_add, alo, blo, in->frnd,
                                           in->ftz, in->sat, line);
            uint32_t rhi = f32_addsub_lane(c, b, is_add, ahi, bhi, in->frnd,
                                           in->ftz, in->sat, line);
            if (c->failed)
                return;
            def_val(c, in->dst, PTY_U64,
                    f32x2_pack(c, b, rlo, rhi));
            return;
        }
        // A 32-bit shared/global byte offset is pointer arithmetic too:
        // hand-written PTX does `mov.u32 %r, smem; add.u32 %r, %r, idx`
        // and uses %r as a .shared address.  Treat the destination as
        // the pointer home (its 32-bit scalar value is unused).  For
        // .address_size 32 kernels, pointer arithmetic uses s32/u32/b32
        // (all map to PTY_U32 in this compiler's type system).
        int ptr_u32 = 0;
        if (in->ty == PTY_U32) {
            for (int s = 0; s < 2; s++) {
                if (in->src[s].kind != OPD_REG)
                    continue;
                reg_t *qr = find_reg(c, in->src[s].reg);
                 if (qr && qr->defined && qr->is_ptr)
                     ptr_u32 = 1;
        }
    }
        if (in->ty == PTY_U64 || ptr_u32) {
            // Pointer arithmetic: exactly one operand may be a pointer
            // (ptr +/- scalar offset, home preserved). Pure-value
            // sub.u64/s64 is wrap arithmetic; ptr-ptr and scalar-ptr
            // sub stay errors.
            if (in->ty == PTY_U64 &&
                check_dst(c, in->dst, PTY_U64, line) != 0)
                return;
            int pi = -1, nptr = 0;
            for (int s = 0; s < 2; s++) {
                if (in->src[s].kind == OPD_REG) {
                    reg_t *qr = find_reg(c, in->src[s].reg);
                    if (getenv("ACUDA_DEBUG_SYMBOLS") && in->dst[0] && strcmp(in->dst, "%rd3") == 0) {
                        fprintf(stderr, "ACuda: add dbg src%d=%s def=%d ptr=%d bind=%d paddr=%d sh=%d gl=%d off=%u\n",
                                s, in->src[s].reg, qr?qr->defined:-1, qr?qr->is_ptr:-1,
                                qr?qr->binding:-99, qr?qr->paddr:-99, qr?qr->shvar:-99, qr?qr->glvar:-99,
                                qr?qr->off:0u);
                    }
                    if (qr && qr->defined && qr->is_ptr) {
                        if (qr->paddr >= 0) {
                            failf(c, line,
                                  "arithmetic on a parameter address unsupported");
                            return;
                        }
                        nptr++;
                        pi = s;
                    }
                }
            }
            if (getenv("ACUDA_DEBUG_SYMBOLS") && in->dst[0] && strcmp(in->dst, "%rd3") == 0) {
                fprintf(stderr, "ACuda: add ptr classify dst=%s nptr=%d pi=%d\n", in->dst, nptr, pi);
            }
            if (nptr > 1) {
                // Two addresses: PTX u64 add/sub over addresses is plain
                // 64-bit integer arithmetic and the result is a value,
                // not an address (stateful_neg_offset folds two kernel
                // pointers into a dead offset). Only kernel-parameter
                // addresses are raw 64-bit values; shared/global-image
                // homes carry offsets, so those pairs stay errors.
                for (int s = 0; s < 2; s++) {
                    if (in->src[s].kind != OPD_REG)
                        continue;
                    reg_t *qr = find_reg(c, in->src[s].reg);
                    if (qr && qr->defined && qr->is_ptr &&
                        ((qr->binding < 0 && qr->binding != -3) ||
                         qr->shvar >= 0 || qr->glvar >= 0)) {
                        failf(c, line, is_add
                                          ? "pointer+pointer unsupported"
                                          : "pointer-pointer sub unsupported");
                        return;
                    }
                }
                uint32_t va[2];
                for (int s = 0; s < 2; s++) {
                    reg_t *qr = in->src[s].kind == OPD_REG
                                    ? find_reg(c, in->src[s].reg)
                                    : NULL;
                    if (qr && qr->defined && qr->is_ptr) {
                        va[s] = qr->off;
                    } else {
                        val_t v = resolve(c, b, &in->src[s], PTY_U64, line);
                        if (c->failed)
                            return;
                        va[s] = v.id;
                    }
                }
                id = nid(c);
                emit_arith(c, b, is_add ? S_IAdd : S_ISub, c->t_u64, id,
                           va[0], va[1]);
                def_val(c, in->dst, PTY_U64, id);
                return;
            }
            if (nptr == 0) {
                a = resolve(c, b, &in->src[0], PTY_U64, line);
                c2 = resolve(c, b, &in->src[1], PTY_U64, line);
                if (c->failed)
                    return;
                id = nid(c);
                emit_arith(c, b, is_add ? S_IAdd : S_ISub, c->t_u64, id,
                           a.id, c2.id);
                def_val(c, in->dst, PTY_U64, id);
                return;
            }
            if (!is_add && pi != 0) {
                failf(c, line, "scalar-pointer sub unsupported");
                return;
            }
            int vi = 1 - pi;
            reg_t *pr = find_reg(c, in->src[pi].reg);
            // A 32-bit pointer-word register also carries its numeric
            // word; keep it alongside the address offset (see the
            // matching mad.lo arm) so value uses and state saves do not
            // lose the low half of a 64-bit pointer.
            uint32_t idx32 = 0;
            if (in->ty == PTY_U32) {
                c2 = resolve(c, b, &in->src[vi], PTY_U32, line);
                if (c->failed)
                    return;
                idx32 = c2.id;
                uint32_t wide = nid(c);
                /* 32-bit PTX pointer arithmetic wraps modulo 2^32, so a
                 * displacement whose high bit is set is a NEGATIVE offset
                 * (e.g. 0xFFFFFFFC == -4).  Sign-extend it before adding
                 * to the 64-bit base (BDA / descriptor offset); zero-
                 * extension would turn -4 into +4 GiB and fault the GPU. */
                put(c, b, (4u << 16) | S_SConvert);
                put(c, b, c->t_u64);
                put(c, b, wide);
                put(c, b, c2.id);
                c2.id = wide;
            } else {
                c2 = resolve(c, b, &in->src[vi], PTY_U64, line);
                if (c->failed)
                    return;
            }
            uint32_t vword = 0;
            int wok = 0;
            if (in->ty == PTY_U32 && pr->cur && pr->cur_ty == PTY_U32) {
                vword = nid(c);
                emit_arith(c, b, is_add ? S_IAdd : S_ISub, c->t_u32, vword,
                           pr->cur, idx32);
                wok = 1;
            }
            id = nid(c);
            emit_arith(c, b, is_add ? S_IAdd : S_ISub, c->t_u64, id,
                       pr->off, c2.id);
            if (pr->shvar >= 0)
                def_shared(c, in->dst, pr->shvar, id);
            else if (pr->glvar >= 0)
                def_global(c, in->dst, pr->glvar, id);
            else {
                def_ptr(c, in->dst, pr->binding, id);
                if (pr->binding == -2) {
                    reg_t *dr = find_reg(c, in->dst);
                    if (dr) dr->lovar = pr->lovar;
                    if (pr->lovar < 0)
                        ACUDA_DEBUG_PRINT("acuda-local-warn add %s from %s: base lovar=%d bind=%d\n",
                                          in->dst, in->src[pi].reg, pr->lovar, pr->binding);
                }
            }
            if (wok) {
                reg_t *dr = find_reg(c, in->dst);
                if (dr) {
                    dr->cur = vword;
                    dr->cur_ty = PTY_U32;
                }
            }
            return;
        }
        if (check_dst(c, in->dst, in->ty, line) != 0)
            return;
        a = resolve(c, b, &in->src[0], in->ty, line);
        c2 = resolve(c, b, &in->src[1], in->ty, line);
        if (c->failed)
            return;
        id = nid(c);
        ptx_ty_t et = eff_ty(c, in->ty);
        if (in->pack) {
            // Packed integer add/sub: lane-wise 32-bit op truncated back
            // to the lane width (wrap semantics).
            int width = in->pack == 2 ? 16 : 8;
            uint32_t mask = (1u << width) - 1u;
            uint32_t out = 0;
            for (int i = 0; i < in->pack; i++) {
                uint32_t la = pack_lane_get(c, b, a.id, i, width, 0);
                uint32_t lb = pack_lane_get(c, b, c2.id, i, width, 0);
                uint32_t r = nid(c);
                emit_arith(c, b, is_add ? S_IAdd : S_ISub, c->t_u32, r,
                           la, lb);
                out = pack_lane_put(c, b, out, r, i, width, mask);
                if (c->failed)
                    return;
            }
            def_val(c, in->dst, PTY_U32, out);
            return;
        }
        if (et == PTY_U8 || et == PTY_U16) {
            // Narrow wraparound on zero-extended canonical values.
            uint32_t r = nid(c);
            emit_arith(c, b, is_add ? S_IAdd : S_ISub, c->t_u32, r,
                       a.id, c2.id);
            id = nid(c);
            emit_arith(c, b, S_And, c->t_u32, id, r,
                       const_u32(c, et == PTY_U8 ? 0xffu : 0xffffu));
            def_val(c, in->dst, in->ty, id);
            return;
        }
        // F16 values are 16-bit patterns: widen to f32, compute, and
        // narrow back (see f32_to_f16bits for exactness). The f32
        // compute + single RNE matches .rn request semantics.
        if (et == PTY_F16 || et == PTY_BF16) {
            int is_bf16 = et == PTY_BF16;
            id = half_arith_lane(c, b, is_bf16,
                                 is_add ? S_FAdd : S_FSub, in->ftz,
                                 in->sat, a.id, c2.id, line);
            if (c->failed)
                return;
            def_val(c, in->dst, in->ty, id);
            return;
        }
        if (et == PTY_F16X2 || et == PTY_BF16X2) {
            int is_bf16 = et == PTY_BF16X2;
            uint32_t ahi, alo, bhi, blo;
            split_pack(c, b, a.id, &ahi, &alo);
            split_pack(c, b, c2.id, &bhi, &blo);
            uint32_t rhi = half_arith_lane(c, b, is_bf16,
                                           is_add ? S_FAdd : S_FSub, in->ftz,
                                           in->sat, ahi, bhi, line);
            uint32_t rlo = half_arith_lane(c, b, is_bf16,
                                           is_add ? S_FAdd : S_FSub, in->ftz,
                                           in->sat, alo, blo, line);
            if (c->failed)
                return;
            def_val(c, in->dst, in->ty, pack2(c, b, rhi, rlo));
            return;
        }
        if (et == PTY_F32) {
            uint32_t fa = a.id, fb = c2.id;
            if (in->ftz) {
                // Flush-to-zero: denormal inputs and results flush.
                fa = flush_f32_denorm(c, b, fa, line);
                if (c->failed)
                    return;
                fb = flush_f32_denorm(c, b, fb, line);
                if (c->failed)
                    return;
            }
            if (in->frnd) {
                id = f32_soft_addsub(c, b, fa, fb, !is_add, in->frnd,
                                     line);
                if (c->failed)
                    return;
            } else {
                emit_arith(c, b, is_add ? S_FAdd : S_FSub, c->t_f32, id,
                           fa, fb);
            }
            if (in->ftz) {
                id = flush_f32_denorm(c, b, id, line);
                if (c->failed)
                    return;
            }
            if (in->sat) {
                id = sat_f32(c, b, id, line);
                if (c->failed)
                    return;
            }
        } else if (et == PTY_F64) {
            if (in->frnd) {
                id = f64_soft_addsub(c, b, a.id, c2.id, !is_add,
                                     in->frnd, line);
                if (c->failed)
                    return;
            } else {
                emit_arith(c, b, is_add ? S_FAdd : S_FSub, c->t_f64, id,
                           a.id, c2.id);
            }
        } else
            emit_arith(c, b, is_add ? S_IAdd : S_ISub, c->t_u32, id, a.id, c2.id);
        def_val(c, in->dst, in->ty, id);
        return;
    }
