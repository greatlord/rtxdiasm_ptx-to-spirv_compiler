#include "instructions/conversion/conversion.h"

#include <string.h>

void emit_cvt_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a;
    uint32_t id;
        // Integer conversions into a wider integer register fill the
        // whole register (cvt.s16.s8 into .b32 sign-extends, unsigned
        // zero-extends — hardware behavior pinned by cvt_s16_s8).
        reg_t *dr = find_reg(c, in->dst);
        int wide_fill = dr && dr->decl_ty == PTY_U32 &&
                        (in->ty == PTY_U8 || in->ty == PTY_U16) &&
                        (in->ty2 == PTY_U8 || in->ty2 == PTY_U16 ||
                         in->ty2 == PTY_U32 || in->ty2 == PTY_U64);
        if (!wide_fill && check_dst(c, in->dst, in->ty, line) != 0)
            return;
        // Pointer word extraction: mov.b64 {lo,hi}, ptr expands to a
        // cvt/shr/cvt sequence.  Materialize the pointer token so the
        // halves can be stored and later reassembled (dynamic pointer
        // tables); see ptr_token_u64 for the tag layout.
        if (in->nsrc == 1 && in->src[0].kind == OPD_REG &&
            in->ty2 == PTY_U64 &&
            (in->ty == PTY_U64 || in->ty == PTY_U32 ||
             in->ty == PTY_U16 || in->ty == PTY_U8)) {
            reg_t *sr = find_reg(c, in->src[0].reg);
            if (sr && sr->defined && sr->is_ptr) {
                uint32_t tok = ptr_token_u64(c, b, sr, line);
                if (in->ty != PTY_U64) {
                    uint32_t t = nid(c);
                    put(c, b, (4u << 16) | S_UConvert);
                    put(c, b, c->t_u32);
                    put(c, b, t);
                    put(c, b, tok);
                    tok = t;
                    if (in->ty != PTY_U32) {
                        uint32_t m = nid(c);
                        emit_arith(c, b, S_And, c->t_u32, m, tok,
                                   const_u32(c, in->ty == PTY_U16
                                                    ? 0xffffu : 0xffu));
                        tok = m;
                    }
                }
                if (c->failed)
                    return;
                def_val(c, in->dst, in->ty, tok);
                return;
            }
        }
        if (in->tf32) {
            // cvt.{rn,rna}.tf32.f32: round the f32 significand to 10
            // explicit bits and clear the low 13 bits.  rna rounds
            // ties away, rn rounds ties to even; NaN is canonical.
            if (in->ty2 != PTY_F32)
                return;
            a = resolve(c, b, &in->src[0], PTY_F32, line);
            if (c->failed)
                return;
            uint32_t bits = f32_to_bits(c, b, a.id);
            uint32_t sign = u32_bin(c, b, S_And, bits,
                                    const_u32(c, 0x80000000u));
            uint32_t exp = u32_bin(c, b, S_And, bits,
                                   const_u32(c, 0x7f800000u));
            uint32_t mant = u32_bin(c, b, S_And, bits,
                                    const_u32(c, 0x7fffffu));
            uint32_t nan = bool_bin(c, b, S_LogicalAnd,
                                    cmp_u32(c, b, PCM_EQ, 0, exp,
                                            const_u32(c, 0x7f800000u),
                                            line),
                                    cmp_u32(c, b, PCM_NE, 0, mant,
                                            const_u32(c, 0), line));
            uint32_t inc;
            if (in->rnd) {
                inc = const_u32(c, 0x1000u);
            } else {
                uint32_t odd = u32_bin(c, b, S_And,
                                       u32_bin(c, b, S_ShrL, bits,
                                               const_u32(c, 13)),
                                       const_u32(c, 1));
                inc = u32_bin(c, b, S_IAdd, odd, const_u32(c, 0x0fffu));
            }
            uint32_t sum = u32_bin(c, b, S_IAdd, bits, inc);
            uint32_t rounded = u32_bin(c, b, S_And, sum,
                                       const_u32(c, 0xffffe000u));
            uint32_t canon = u32_bin(c, b, S_Or, sign,
                                     const_u32(c, 0x7fc00000u));
            uint32_t out = sel_u32(c, b, nan, canon, rounded);
            def_val(c, in->dst, PTY_U32, out);
            return;
        }
        if (in->ty == PTY_F32 && in->ty2 == PTY_F32) {
            // float identity/saturation (cvt[.ftz][.sat].f32.f32).
            a = resolve(c, b, &in->src[0], PTY_F32, line);
            if (c->failed)
                return;
            uint32_t fa = a.id;
            if (in->ftz) {
                fa = flush_f32_denorm(c, b, fa, line);
                if (c->failed)
                    return;
            }
            if (in->sat) {
                fa = sat_f32(c, b, fa, line);
                if (c->failed)
                    return;
            }
            def_val(c, in->dst, PTY_F32, fa);
            return;
        }
        if (in->ty == PTY_F64 && in->ty2 == PTY_F32) {
            // Exact widening. Under emulation this is a plain copy.
            a = resolve(c, b, &in->src[0], PTY_F32, line);
            if (c->failed)
                return;
            uint32_t fa = a.id;
            if (in->ftz) {
                // Flush denormal f32 inputs (the widening itself never
                // creates f64 denormals from flushed inputs).
                fa = flush_f32_denorm(c, b, fa, line);
                if (c->failed)
                    return;
            }
            if (c->emu_f64) {
                def_val(c, in->dst, PTY_F32, fa);
                return;
            }
            uint32_t u = f32bits_to_f64bits(c, b, fa);
            id = nid(c);
            put(c, b, (4u << 16) | S_Bitcast);
            put(c, b, c->t_f64);
            put(c, b, id);
            put(c, b, u);
            def_val(c, in->dst, PTY_F64, id);
            return;
        }
        if (in->ty == PTY_F32 && in->ty2 == PTY_F64) {
            // Round-to-nearest narrowing. Under emulation a plain copy.
            a = resolve(c, b, &in->src[0], PTY_F64, line);
            if (c->failed)
                return;
            if (c->emu_f64) {
                def_val(c, in->dst, PTY_F32, a.id);
                return;
            }
            uint32_t u = nid(c);
            put(c, b, (4u << 16) | S_Bitcast);
            put(c, b, c->t_u64);
            put(c, b, u);
            put(c, b, a.id);
            id = f64bits_to_f32bits(c, b, u);
            def_val(c, in->dst, PTY_F32, id);
            return;
        }
        // Half widenings: the source is 16-bit patterns (in .u16
        // or .f16 registers, per the numba b16 idiom); the result is
        // the exact wider value. check_dst on the caller side already
        // enforced a .f32/.f64 destination.
        if ((in->ty == PTY_F32 || in->ty == PTY_F64) &&
            in->ty2 == PTY_F16) {
            a = resolve(c, b, &in->src[0], PTY_F16, line);
            if (c->failed)
                return;
            id = f16_to_f32val(c, b, a.id, line);
            if (c->failed)
                return;
            if (in->ty == PTY_F64) {
                // Exact: f32 holds every f16 value; widen to f64.
                uint32_t f = id;
                id = nid(c);
                put(c, b, (4u << 16) | S_FConvert);
                put(c, b, c->t_f64);
                put(c, b, id);
                put(c, b, f);
            }
            def_val(c, in->dst, in->ty, id);
            return;
        }
        // Round-to-nearest narrowing to half bits. Destinations hold
        // patterns (.u16 or .f16 declarations).
        if (in->ty == PTY_F16 && in->ty2 == PTY_F32) {
            if (check_dst(c, in->dst, PTY_F16, line) != 0)
                return;
            a = resolve(c, b, &in->src[0], PTY_F32, line);
            if (c->failed)
                return;
            uint32_t fa = a.id;
            if (in->relu) {
                // .relu clamps negative results to +0 (NaN -> +0,
                // matching the packed narrowing path).
                fa = f32_relu_nonneg(c, b, fa, line);
                if (c->failed)
                    return;
            }
            id = f32_to_f16bits(c, b, fa, line);
            if (c->failed)
                return;
            if (in->sat) {
                id = f16_satfinite_bits(c, b, id, line);
                if (c->failed)
                    return;
            }
            def_val(c, in->dst, PTY_F16, id);
            return;
        }
        if (in->ty == PTY_F16 && in->ty2 == PTY_F64) {
            if (check_dst(c, in->dst, PTY_F16, line) != 0)
                return;
            a = resolve(c, b, &in->src[0], PTY_F64, line);
            if (c->failed)
                return;
            id = f64_to_f16bits(c, b, a.id, line);
            if (c->failed)
                return;
            def_val(c, in->dst, PTY_F16, id);
            return;
        }
        // Round-to-nearest narrowing to brain-float bits (destinations
        // hold patterns in .bf16/.u16 declarations, like f16).
        if (in->ty == PTY_BF16 && in->ty2 == PTY_F32) {
            if (check_dst(c, in->dst, PTY_BF16, line) != 0)
                return;
            a = resolve(c, b, &in->src[0], PTY_F32, line);
            if (c->failed)
                return;
            id = f32_to_bf16bits(c, b, a.id, line);
            if (c->failed)
                return;
            def_val(c, in->dst, PTY_BF16, id);
            return;
        }
        // Directed integer -> float: exact magnitude normalized to the
        // target precision and rounded with the requested direction.
        if ((in->ty == PTY_F32 || in->ty == PTY_F64 || in->ty == PTY_F16) &&
            (in->ty2 == PTY_U8 || in->ty2 == PTY_U16 ||
             in->ty2 == PTY_U32 || in->ty2 == PTY_U64) && in->rnd) {
            int wsrc = decl_int_width(in->ty2);
            a = resolve(c, b, &in->src[0], in->ty2, line);
            if (c->failed)
                return;
            id = int_to_float_dir(c, b, a.id, in->sign, wsrc, in->ty,
                                  in->rnd, line);
            if (c->failed)
                return;
            def_val(c, in->dst, in->ty, id);
            return;
        }
        // Integer to float (round-to-nearest). Signed sources use
        // ConvertSToF, unsigned use ConvertUToF; f16 narrows from f32.
        if ((in->ty == PTY_F32 || in->ty == PTY_F64 || in->ty == PTY_F16) &&
            (in->ty2 == PTY_U8 || in->ty2 == PTY_U16)) {
            int wsrc = decl_int_width(in->ty2);
            a = resolve(c, b, &in->src[0], in->ty2, line);
            if (c->failed)
                return;
            uint32_t v = a.id;
            if (wsrc == 8 || wsrc == 16) {
                uint32_t m = nid(c);
                emit_arith(c, b, S_And, c->t_u32, m, v,
                           const_u32(c, wsrc == 8 ? 0xffu : 0xffffu));
                v = m;
                if (in->sign) {
                    uint32_t t1 = nid(c), t2 = nid(c);
                    emit_arith(c, b, S_Shl, c->t_u32, t1, v,
                               const_u32(c, (uint32_t)(32 - wsrc)));
                    emit_arith(c, b, S_ShrA, c->t_u32, t2, t1,
                               const_u32(c, (uint32_t)(32 - wsrc)));
                    v = t2;
                }
            }
            uint32_t fty = in->ty == PTY_F64 ? c->t_f64 : c->t_f32;
            uint32_t conv = nid(c);
            put(c, b, (4u << 16) | (in->sign ? S_ConvertSToF
                                             : S_ConvertUToF));
            put(c, b, fty);
            put(c, b, conv);
            put(c, b, v);
            if (in->ty == PTY_F16) {
                id = f32_to_f16bits(c, b, conv, line);
                if (c->failed)
                    return;
                def_val(c, in->dst, PTY_F16, id);
            } else {
                def_val(c, in->dst, in->ty, conv);
            }
            return;
        }
        if ((in->ty == PTY_F32 || in->ty == PTY_F64 || in->ty == PTY_F16) &&
            (in->ty2 == PTY_U32 || in->ty2 == PTY_U64)) {
            if (check_dst(c, in->dst, in->ty, line) != 0)
                return;
            a = resolve(c, b, &in->src[0], in->ty2, line);
            if (c->failed)
                return;
            uint32_t fty = in->ty == PTY_F64 ? c->t_f64 : c->t_f32;
            uint32_t conv = nid(c);
            put(c, b, (4u << 16) | (in->sign ? S_ConvertSToF
                                             : S_ConvertUToF));
            put(c, b, fty);
            put(c, b, conv);
            put(c, b, a.id);
            if (in->ty == PTY_F16) {
                id = f32_to_f16bits(c, b, conv, line);
                if (c->failed)
                    return;
                def_val(c, in->dst, PTY_F16, id);
            } else {
                def_val(c, in->dst, in->ty, conv);
            }
            return;
        }
        // Float-to-integer with PTX saturation. f32 sources use the
        // existing directed-rounding helper; f64 sources support .rzi
        // only (OpConvertFToS/U truncates). Out-of-range values clamp
        // to the destination bounds; NaN converts to 0.
        if ((in->ty2 == PTY_F32 || in->ty2 == PTY_F64) &&
            (in->ty == PTY_U8 || in->ty == PTY_U16 || in->ty == PTY_U32 ||
             in->ty == PTY_U64)) {
            int src64 = in->ty2 == PTY_F64;
            uint32_t rv;
            if (src64) {
                a = resolve(c, b, &in->src[0], PTY_F64, line);
                if (c->failed)
                    return;
                rv = round_f64_directed(c, b, a.id,
                                        in->rnd ? in->rnd : 2, line);
                if (c->failed)
                    return;
            } else {
                a = resolve_f32_bits(c, b, &in->src[0], line);
                if (c->failed)
                    return;
                rv = a.id;
                if (in->ftz) {
                    rv = flush_f32_denorm(c, b, rv, line);
                    if (c->failed)
                        return;
                }
                rv = round_f32_directed(c, b, rv,
                                        in->rnd ? in->rnd : 1, line);
                if (c->failed)
                    return;
            }
            int w = in->ty == PTY_U64 ? 64
                    : in->ty == PTY_U32 ? 32
                    : in->ty == PTY_U16 ? 16 : 8;
            uint32_t lo_b, hi_b;
            if (!in->sign) {
                lo_b = const_f32(c, 0u);
                hi_b = const_f32(c, w == 64 ? 0x5f800000u
                                    : w == 32 ? 0x4f800000u
                                    : w == 16 ? 0x47800000u
                                              : 0x43800000u);
            } else {
                lo_b = const_f32(c, w == 64 ? 0xdf000000u
                                    : w == 32 ? 0xcf000000u
                                    : w == 16 ? 0xc7000000u
                                              : 0xc3000000u);
                hi_b = const_f32(c, w == 64 ? 0x5f000000u
                                    : w == 32 ? 0x4f000000u
                                    : w == 16 ? 0x47000000u
                                              : 0x43000000u);
            }
            if (src64) {
                uint32_t ub = f32bits_to_f64bits(c, b, lo_b);
                uint32_t lb = nid(c);
                put(c, b, (4u << 16) | S_Bitcast);
                put(c, b, c->t_f64);
                put(c, b, lb);
                put(c, b, ub);
                lo_b = lb;
                ub = f32bits_to_f64bits(c, b, hi_b);
                lb = nid(c);
                put(c, b, (4u << 16) | S_Bitcast);
                put(c, b, c->t_f64);
                put(c, b, lb);
                put(c, b, ub);
                hi_b = lb;
            }
            uint32_t iseq = do_cmp(c, b, PCM_EQ, src64 ? PTY_F64 : PTY_F32,
                                   0, rv, rv, line);
            uint32_t nan = nid(c);
            put(c, b, (4u << 16) | S_LogicalNot);
            put(c, b, c->t_bool);
            put(c, b, nan);
            put(c, b, iseq);
            uint32_t lo_ok = do_cmp(c, b, PCM_GE,
                                    src64 ? PTY_F64 : PTY_F32, 0, rv, lo_b,
                                    line);
            uint32_t hi_ok = do_cmp(c, b, PCM_LT,
                                    src64 ? PTY_F64 : PTY_F32, 0, rv, hi_b,
                                    line);
            uint32_t conv = nid(c);
            put(c, b, (4u << 16) | (in->sign ? S_ConvertFToS
                                             : S_ConvertFToU));
            put(c, b, w == 64 ? c->t_u64 : c->t_u32);
            put(c, b, conv);
            put(c, b, rv);
            uint32_t maxv, lowv;
            if (w == 64) {
                maxv = in->sign ? const_u64(c, 0x7fffffffffffffffull)
                                : const_u64(c, 0xffffffffffffffffull);
                lowv = in->sign ? const_u64(c, 0x8000000000000000ull)
                                : const_u64(c, 0);
                uint32_t mid = sel_u64(c, b, hi_ok, conv, maxv);
                uint32_t rng = sel_u64(c, b, lo_ok, mid, lowv);
                id = sel_u64(c, b, nan, const_u64(c, 0), rng);
            } else {
                maxv = in->sign ? const_u32(c, w == 32 ? 0x7fffffffu
                                                       : w == 16 ? 0x7fffu
                                                                 : 0x7fu)
                                : const_u32(c, w == 32 ? 0xffffffffu
                                                       : w == 16 ? 0xffffu
                                                                 : 0xffu);
                lowv = in->sign ? const_u32(c, w == 32 ? 0x80000000u
                                                       : w == 16 ? 0xffff8000u
                                                                 : 0xffffff80u)
                                : const_u32(c, 0);
                uint32_t mid = sel_u32(c, b, hi_ok, conv, maxv);
                uint32_t rng = sel_u32(c, b, lo_ok, mid, lowv);
                id = sel_u32(c, b, nan, const_u32(c, 0), rng);
            }
            if (c->failed)
                return;
            def_val(c, in->dst, in->ty, id);
            return;
        }
        if (in->sat) {
            // Saturating integer conversion: lift the source to u64
            // (sign-extending when signed), clamp to the destination
            // bounds, then narrow.
            int wsrc = decl_int_width(in->ty2);
            int wdst = decl_int_width(in->ty);
            if (wsrc < 0 || wdst < 0) {
                failf(c, line, "bad saturating conversion");
                return;
            }
            a = resolve(c, b, &in->src[0], in->ty2, line);
            if (c->failed)
                return;
            uint32_t v = a.id;
            if (wsrc < 32) {
                v = nid(c);
                emit_arith(c, b, S_And, c->t_u32, v, a.id,
                           const_u32(c, wsrc == 8 ? 0xffu : 0xffffu));
                if (in->sign) {
                    uint32_t t1 = nid(c), t2 = nid(c);
                    emit_arith(c, b, S_Shl, c->t_u32, t1, v,
                               const_u32(c, (uint32_t)(32 - wsrc)));
                    emit_arith(c, b, S_ShrA, c->t_u32, t2, t1,
                               const_u32(c, (uint32_t)(32 - wsrc)));
                    v = t2;
                }
            }
            uint32_t v64;
            if (wsrc == 64) {
                v64 = v;
            } else {
                v64 = nid(c);
                put(c, b, (4u << 16) |
                              (in->sign ? S_SConvert : S_UConvert));
                put(c, b, c->t_u64);
                put(c, b, v64);
                put(c, b, v);
            }
            uint64_t dmin = 0, dmax;
            if (in->dsign) {
                dmin = (uint64_t)(int64_t)(-((int64_t)1 << (wdst - 1)));
                dmax = ((uint64_t)1 << (wdst - 1)) - 1;
            } else {
                dmax = wdst == 64 ? ~(uint64_t)0
                                  : (((uint64_t)1 << wdst) - 1);
            }
            if (!in->sign)
                dmin = 0;
            uint32_t lt = do_cmp(c, b, PCM_LT, PTY_U64, in->sign, v64,
                                 const_u64(c, dmin), line);
            uint32_t gt = do_cmp(c, b, PCM_GT, PTY_U64, in->sign, v64,
                                 const_u64(c, dmax), line);
            if (c->failed)
                return;
            v64 = sel_u64(c, b, lt, const_u64(c, dmin), v64);
            v64 = sel_u64(c, b, gt, const_u64(c, dmax), v64);
            if (wdst == 64) {
                def_val(c, in->dst, PTY_U64, v64);
                return;
            }
            id = nid(c);
            put(c, b, (4u << 16) | S_UConvert);
            put(c, b, c->t_u32);
            put(c, b, id);
            put(c, b, v64);
            if (wdst == 8 || wdst == 16) {
                uint32_t m = nid(c);
                emit_arith(c, b, S_And, c->t_u32, m, id,
                           const_u32(c, wdst == 8 ? 0xffu : 0xffffu));
                id = m;
            }
            if (wide_fill && in->dsign) {
                uint32_t t1 = nid(c);
                emit_arith(c, b, S_Shl, c->t_u32, t1, id,
                           const_u32(c, (uint32_t)(32 - wdst)));
                id = nid(c);
                emit_arith(c, b, S_ShrA, c->t_u32, id, t1,
                           const_u32(c, (uint32_t)(32 - wdst)));
            }
            def_val(c, in->dst, in->ty, id);
            return;
        }
        // Integer conversions across any widths. Narrow sources are
        // canonical (zero-extended), so signed sources narrow to
        // their width first; narrow destinations mask; 64<->32 uses
        // the matching convert (truncate down, extend up).
        {
            int wsrc = decl_int_width(in->ty2);
            int wdst = decl_int_width(in->ty);
            if (wsrc < 0 || wdst < 0) {
                failf(c, line, "bad integer conversion");
                return;
            }
            // Narrow sources may sit in wider integer registers
            // (cvt.s16.s8 from a .b32 holder reads the low bits); the
            // canonicalize mask below applies either way.
            val_t av;
            memset(&av, 0, sizeof(av));
            if (wdst == 64 && wsrc == 32 && !in->sat && !in->pred[0] &&
                in->src[0].kind == OPD_REG) {
                // Zero-extension of a 32-bit pointer home (lifted
                // 32-bit addressing: the low word carries the home).
                // The widened value addresses the same home; the
                // offset is already a u64 SSA (home arithmetic
                // widens) or zero.  Handle before resolve(), which
                // rejects pointer registers as values.
                reg_t *srw = find_reg(c, in->src[0].reg);
                if (srw && srw->defined && srw->is_ptr &&
                    srw->binding >= 0 && srw->shvar < 0 &&
                    srw->glvar < 0 && srw->paddr < 0) {
                    if (check_dst(c, in->dst, PTY_U64, line) != 0)
                        return;
                    def_ptr(c, in->dst, srw->binding, srw->off);
                    reg_t *drw = find_reg(c, in->dst);
                    if (drw)
                        drw->is_tex = srw->is_tex;
                    return;
                }
                // else fall through (generic conversion below)
            }
            if (in->src[0].kind == OPD_REG && (wsrc == 8 || wsrc == 16)) {
                reg_t *sr = find_reg(c, in->src[0].reg);
                if (sr && sr->defined && !sr->is_ptr &&
                    decl_int_width(sr->decl_ty) == 32) {
                    av.id = sr->cur;
                    av.ty = in->ty2;
                } else {
                    av = resolve(c, b, &in->src[0], in->ty2, line);
                }
            } else {
                av = resolve(c, b, &in->src[0], in->ty2, line);
            }
            if (c->failed)
                return;
            a = av;
            uint32_t v = a.id;
            if (wsrc == 8 || wsrc == 16) {
                // Canonicalize: some producers (bitwise-not) leave
                // garbage above the width; consumers read low bits.
                uint32_t m = nid(c);
                emit_arith(c, b, S_And, c->t_u32, m, v,
                           const_u32(c, wsrc == 8 ? 0xffu : 0xffffu));
                v = m;
            }
            if (in->sign && wsrc < 32) {
                uint32_t t1 = nid(c), t2 = nid(c);
                emit_arith(c, b, S_Shl, c->t_u32, t1, v,
                           const_u32(c, (uint32_t)(32 - wsrc)));
                emit_arith(c, b, S_ShrA, c->t_u32, t2, t1,
                           const_u32(c, (uint32_t)(32 - wsrc)));
                v = t2;
            }
            if (wdst == 64 && wsrc == 32) {
                id = nid(c);
                put(c, b, (4u << 16) | (in->sign ? S_SConvert : S_UConvert));
                put(c, b, c->t_u64);
                put(c, b, id);
                put(c, b, v);
                v = id;
            } else if (wdst == 32 && wsrc == 64) {
                id = nid(c);
                put(c, b, (4u << 16) | S_UConvert);
                put(c, b, c->t_u32);
                put(c, b, id);
                put(c, b, v);
                v = id;
            }
            if (wdst == 8 || wdst == 16) {
                id = nid(c);
                emit_arith(c, b, S_And, c->t_u32, id, v,
                           const_u32(c, wdst == 8 ? 0xffu : 0xffffu));
                v = id;
            }
            if (in->sat) {
                // sat.u32.s32 (only supported sat form): clamp
                // negatives to 0.
                uint32_t ok = do_cmp(c, b, PCM_GE, PTY_U32, 1, v,
                                     const_u32(c, 0), line);
                if (c->failed)
                    return;
                v = sel_u32(c, b, ok, v, const_u32(c, 0));
            }
            if (wide_fill && in->dsign) {
                // Signed narrow result fills the 32-bit register.
                uint32_t t1 = nid(c);
                emit_arith(c, b, S_Shl, c->t_u32, t1, v,
                           const_u32(c, (uint32_t)(32 - wdst)));
                v = nid(c);
                emit_arith(c, b, S_ShrA, c->t_u32, v, t1,
                           const_u32(c, (uint32_t)(32 - wdst)));
            }
            def_val(c, in->dst, in->ty, v);
        }
        return;
    }
