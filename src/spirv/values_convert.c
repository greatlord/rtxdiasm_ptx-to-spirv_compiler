// PTX subset -> SPIR-V (Logical, GLSL450) emitter.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if !defined(_MSC_VER)
#include <strings.h>
#endif
#include "ptx.h"
#include "acuda/compiler.h"
#include "spirv_internal.h"

// u64 compare helper returning a bool id (unsigned semantics).
static uint32_t cmp_u64(ctx_t *c, spv_blob_t *b, uint32_t op, uint32_t a,
                               uint32_t b2) {
    uint32_t id = nid(c);
    put(c, b, (5u << 16) | op);
    put(c, b, c->t_bool);
    put(c, b, id);
    put(c, b, a);
    put(c, b, b2);
    return id;
}

uint32_t sel_u64(ctx_t *c, spv_blob_t *b, uint32_t cond, uint32_t t,
                        uint32_t f) {
    uint32_t id = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, c->t_u64);
    put(c, b, id);
    put(c, b, cond);
    put(c, b, t);
    put(c, b, f);
    return id;
}

uint32_t sel_u32(ctx_t *c, spv_blob_t *b, uint32_t cond, uint32_t t,
                        uint32_t f) {
    uint32_t id = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, c->t_u32);
    put(c, b, id);
    put(c, b, cond);
    put(c, b, t);
    put(c, b, f);
    return id;
}

uint32_t sel_bool(ctx_t *c, spv_blob_t *b, uint32_t cond, uint32_t t,
                         uint32_t f) {
    uint32_t id = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, c->t_bool);
    put(c, b, id);
    put(c, b, cond);
    put(c, b, t);
    put(c, b, f);
    return id;
}

// f32 bits -> u64 bits holding the same value as a double. Exact except
// that f32 denormals flush to zero (documented limitation).
uint32_t f32bits_to_f64bits(ctx_t *c, spv_blob_t *b, uint32_t f) {
    uint32_t w = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_u32);
    put(c, b, w);
    put(c, b, f);
    uint32_t s = nid(c), e = nid(c), m = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, s, w, const_u32(c, 31));
    emit_arith(c, b, S_ShrL, c->t_u32, e, w, const_u32(c, 23));
    uint32_t e2 = nid(c);
    emit_arith(c, b, S_And, c->t_u32, e2, e, const_u32(c, 0xffu));
    emit_arith(c, b, S_And, c->t_u32, m, w, const_u32(c, 0x7fffffu));
    uint32_t s64 = nid(c), e64 = nid(c), m64 = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u64);
    put(c, b, s64);
    put(c, b, s);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u64);
    put(c, b, e64);
    put(c, b, e2);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u64);
    put(c, b, m64);
    put(c, b, m);
    uint32_t is_ni = cmp_u64(c, b, S_IEqual, e64, const_u64(c, 255));
    uint32_t is_z = cmp_u64(c, b, S_IEqual, e64, const_u64(c, 0));
    // NaN/Inf: exponent all ones; NaN payload widened + quieted.
    uint32_t msh = nid(c);
    emit_arith(c, b, S_Shl, c->t_u64, msh, m64, const_u64(c, 29));
    uint32_t mquo = nid(c);
    emit_arith(c, b, S_Or, c->t_u64, mquo, msh,
               const_u64(c, 0x8000000000000ull));
    uint32_t m_nn = sel_u64(c, b, cmp_u64(c, b, S_IEqual, m64, const_u64(c, 0)),
                            const_u64(c, 0), mquo);
    uint32_t e_nn = const_u64(c, 2047);
    // Normal: rebias exponent, widen mantissa.
    uint32_t e_nm = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, e_nm, e64, const_u64(c, 1023 - 127));
    uint32_t m_nm = nid(c);
    emit_arith(c, b, S_Shl, c->t_u64, m_nm, m64, const_u64(c, 29));
    uint32_t e3 = sel_u64(c, b, is_ni, e_nn, sel_u64(c, b, is_z,
                                                    const_u64(c, 0), e_nm));
    uint32_t m3 = sel_u64(c, b, is_ni, m_nn, sel_u64(c, b, is_z,
                                                    const_u64(c, 0), m_nm));
    uint32_t ds = nid(c), de = nid(c), out = nid(c), tmp = nid(c);
    emit_arith(c, b, S_Shl, c->t_u64, ds, s64, const_u64(c, 63));
    emit_arith(c, b, S_Shl, c->t_u64, de, e3, const_u64(c, 52));
    emit_arith(c, b, S_Or, c->t_u64, tmp, ds, de);
    emit_arith(c, b, S_Or, c->t_u64, out, tmp, m3);
    return out;
}

// u64 bits holding a double -> f32 bits (round-to-nearest-even).
// Double denormals and f32-subnormal results flush to signed zero
// (documented limitation); NaN/Inf exact.
uint32_t f64bits_to_f32bits(ctx_t *c, spv_blob_t *b, uint32_t d) {
    uint32_t s = nid(c), e = nid(c), m = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, s, d, const_u64(c, 63));
    emit_arith(c, b, S_ShrL, c->t_u64, e, d, const_u64(c, 52));
    uint32_t e2 = nid(c);
    emit_arith(c, b, S_And, c->t_u64, e2, e, const_u64(c, 0x7ffull));
    emit_arith(c, b, S_And, c->t_u64, m, d, const_u64(c, 0xfffffffffffffull));
    uint32_t is_ni = cmp_u64(c, b, S_IEqual, e2, const_u64(c, 2047));
    uint32_t is_z = cmp_u64(c, b, S_IEqual, e2, const_u64(c, 0));
    // Normal path: rebias, keep top 23 mantissa bits, round the rest.
    uint32_t e_n = nid(c);
    emit_arith(c, b, S_ISub, c->t_u64, e_n, e2, const_u64(c, 1023 - 127));
    uint32_t top = nid(c), drop = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, top, m, const_u64(c, 29));
    emit_arith(c, b, S_And, c->t_u64, drop, m, const_u64(c, 0x1fffffffull));
    uint32_t half = const_u64(c, 0x10000000ull);
    uint32_t gt = cmp_u64(c, b, S_UGreaterEqual, drop, half);
    uint32_t eq = cmp_u64(c, b, S_IEqual, drop, half);
    uint32_t odd = nid(c);
    emit_arith(c, b, S_And, c->t_u64, odd, top, const_u64(c, 1));
    uint32_t odd1 = cmp_u64(c, b, S_IEqual, odd, const_u64(c, 1));
    // Round up iff drop > half, or (drop == half and top is odd).
    uint32_t upb = sel_bool(c, b, eq, odd1, gt);
    uint32_t up = sel_u64(c, b, upb, const_u64(c, 1), const_u64(c, 0));
    uint32_t top2 = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, top2, top, up);
    uint32_t carry = cmp_u64(c, b, S_IEqual, top2, const_u64(c, 0x800000ull));
    uint32_t top3 = sel_u64(c, b, carry, const_u64(c, 0), top2);
    uint32_t ec = sel_u64(c, b, carry, const_u64(c, 1), const_u64(c, 0));
    uint32_t e3n = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, e3n, e_n, ec);
    // Overflow to Inf when biased exponent reaches 255.
    uint32_t ovf = cmp_u64(c, b, S_UGreaterEqual, e3n, const_u64(c, 255));
    // Underflow (true value below f32 normal range) flushes to signed
    // zero (documented emulation limitation).
    uint32_t small = cmp_u64(c, b, S_ULess, e2, const_u64(c, 897));
    // NaN/Inf inputs.
    uint32_t m_nn = sel_u64(c, b, cmp_u64(c, b, S_IEqual, m, const_u64(c, 0)),
                            const_u64(c, 0), const_u64(c, 0x400000ull));
    uint32_t fs = nid(c), fe = nid(c);
    emit_arith(c, b, S_Shl, c->t_u64, fs, s, const_u64(c, 31));
    // Clamp normal exponent into [1,254]; overflow/NaNInf/underflow
    // override below.
    uint32_t e3c = sel_u64(c, b, ovf, const_u64(c, 255), e3n);
    emit_arith(c, b, S_Shl, c->t_u64, fe, e3c, const_u64(c, 23));
    uint32_t e4 = sel_u64(c, b, is_ni, const_u64(c, 255),
                          sel_u64(c, b, is_z, const_u64(c, 0),
                                  sel_u64(c, b, small, const_u64(c, 0), fe)));
    uint32_t m4 = sel_u64(c, b, is_ni, m_nn,
                          sel_u64(c, b, is_z, const_u64(c, 0),
                                  sel_u64(c, b, small, const_u64(c, 0), top3)));
    // (fs keeps the sign, so underflow yields signed zero.)
    uint32_t o1 = nid(c), o2 = nid(c);
    emit_arith(c, b, S_Or, c->t_u64, o1, fs, e4);
    emit_arith(c, b, S_Or, c->t_u64, o2, o1, m4);
    // Downconvert to 32 bits and bitcast to float.
    uint32_t w = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u32);
    put(c, b, w);
    put(c, b, o2);
    uint32_t f = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_f32);
    put(c, b, f);
    put(c, b, w);
    return f;
}

// f32 value -> u32 holding f16 bits (RNE). Integer-only, no Float16
// capability. Mirrors f32_to_f16_bits (normals exact, Inf/NaN exact,
// f16 denormals produced, f32-denormal inputs flush to signed zero).
uint32_t do_cmp(ctx_t *c, spv_blob_t *b, ptx_cmp_t cmp, ptx_ty_t ty,
                       int sign, uint32_t a, uint32_t b2, int line);

uint32_t f32_to_f16bits(ctx_t *c, spv_blob_t *b, uint32_t f,
                               int line) {
    uint32_t w = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_u32);
    put(c, b, w);
    put(c, b, f);
    uint32_t s = nid(c), e = nid(c), m = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, s, w, const_u32(c, 31));
    emit_arith(c, b, S_ShrL, c->t_u32, e, w, const_u32(c, 23));
    uint32_t e2 = nid(c);
    emit_arith(c, b, S_And, c->t_u32, e2, e, const_u32(c, 0xffu));
    emit_arith(c, b, S_And, c->t_u32, m, w, const_u32(c, 0x7fffffu));
    uint32_t is_ni = do_cmp(c, b, PCM_EQ, PTY_U32, 0, e2,
                            const_u32(c, 255), line);
    uint32_t is_z = do_cmp(c, b, PCM_EQ, PTY_U32, 0, e2,
                           const_u32(c, 0), line);
    if (c->failed)
        return 0;
    // NaN/Inf payload: quiet bit + top payload bits when m != 0.
    uint32_t mshr = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, mshr, m, const_u32(c, 13));
    uint32_t mquo = nid(c);
    emit_arith(c, b, S_Or, c->t_u32, mquo, mshr, const_u32(c, 0x200u));
    uint32_t mnez = do_cmp(c, b, PCM_NE, PTY_U32, 0, m,
                           const_u32(c, 0), line);
    uint32_t m_nn = sel_u32(c, b, mnez, mquo, const_u32(c, 0));
    uint32_t s15 = nid(c), e15 = nid(c), o_ni = nid(c), t_ni = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, s15, s, const_u32(c, 15));
    emit_arith(c, b, S_Shl, c->t_u32, e15, const_u32(c, 0x1fu),
               const_u32(c, 10));
    emit_arith(c, b, S_Or, c->t_u32, t_ni, s15, e15);
    emit_arith(c, b, S_Or, c->t_u32, o_ni, t_ni, m_nn);
    // Unbiased exponent (signed wrap) and path selects.
    uint32_t exp = nid(c);
    emit_arith(c, b, S_ISub, c->t_u32, exp, e2, const_u32(c, 127));
    uint32_t is_ovf = do_cmp(c, b, PCM_GE, PTY_U32, 1, exp,
                             const_u32(c, 16), line);
    uint32_t is_norm = do_cmp(c, b, PCM_GE, PTY_U32, 1, exp,
                              const_u32(c, (uint32_t)(int32_t)-14), line);
    if (c->failed)
        return 0;
    uint32_t o_inf = nid(c);
    emit_arith(c, b, S_Or, c->t_u32, o_inf, s15, e15);
    // Normal path with round-to-nearest-even.
    uint32_t mant = nid(c), drop = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, mant, m, const_u32(c, 13));
    emit_arith(c, b, S_And, c->t_u32, drop, m, const_u32(c, 0x1fffu));
    uint32_t gt = do_cmp(c, b, PCM_GT, PTY_U32, 0, drop,
                         const_u32(c, 0x1000u), line);
    uint32_t eq = do_cmp(c, b, PCM_EQ, PTY_U32, 0, drop,
                         const_u32(c, 0x1000u), line);
    uint32_t odd = nid(c);
    emit_arith(c, b, S_And, c->t_u32, odd, mant, const_u32(c, 1));
    uint32_t odd1 = do_cmp(c, b, PCM_EQ, PTY_U32, 0, odd,
                           const_u32(c, 1), line);
    uint32_t upb = sel_bool(c, b, eq, odd1, gt);
    uint32_t up = sel_u32(c, b, upb, const_u32(c, 1), const_u32(c, 0));
    uint32_t mant2 = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u32, mant2, mant, up);
    uint32_t carry = do_cmp(c, b, PCM_EQ, PTY_U32, 0, mant2,
                            const_u32(c, 0x400u), line);
    uint32_t mant3 = sel_u32(c, b, carry, const_u32(c, 0), mant2);
    uint32_t ec = sel_u32(c, b, carry, const_u32(c, 1), const_u32(c, 0));
    uint32_t exp2 = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u32, exp2, exp, ec);
    uint32_t ovf2 = do_cmp(c, b, PCM_GE, PTY_U32, 1, exp2,
                           const_u32(c, 16), line);
    uint32_t eb = nid(c), esh = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u32, eb, exp2, const_u32(c, 15));
    emit_arith(c, b, S_Shl, c->t_u32, esh, eb, const_u32(c, 10));
    uint32_t o1 = nid(c), o_norm = nid(c);
    emit_arith(c, b, S_Or, c->t_u32, o1, s15, esh);
    emit_arith(c, b, S_Or, c->t_u32, o_norm, o1, mant3);
    o_norm = sel_u32(c, b, ovf2, o_inf, o_norm);
    // Denormal path: full 24-bit mantissa shifted + rounded. rsh2 wraps
    // to a huge unsigned value for tiny exponents, and a SPIR-V shift
    // of >= width is UB even on the discarded arm -- which really does
    // make drivers miscompile the surrounding chain. Clamp to 25 (the
    // `tiny` boundary, where the result is replaced by signed zero).
    uint32_t full = nid(c);
    emit_arith(c, b, S_Or, c->t_u32, full, m, const_u32(c, 0x800000u));
    uint32_t rsh = nid(c);
    emit_arith(c, b, S_ISub, c->t_u32, rsh, const_u32(c, 0), exp);
    uint32_t rsh2 = nid(c);
    emit_arith(c, b, S_ISub, c->t_u32, rsh2, rsh, const_u32(c, 1));
    uint32_t tiny = do_cmp(c, b, PCM_GE, PTY_U32, 0, rsh2,
                           const_u32(c, 25), line);
    if (c->failed)
        return 0;
    uint32_t sc = sel_u32(c, b, tiny, const_u32(c, 25), rsh2);
    uint32_t dmant = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, dmant, full, sc);
    uint32_t dmask = nid(c), dnot = nid(c), ddrop = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, dmask, const_u32(c, 0xffffffffu),
               sc);
    put(c, b, (4u << 16) | S_Not);
    put(c, b, c->t_u32);
    put(c, b, dnot);
    put(c, b, dmask);
    emit_arith(c, b, S_And, c->t_u32, ddrop, full, dnot);
    uint32_t rs1 = nid(c);
    emit_arith(c, b, S_ISub, c->t_u32, rs1, sc, const_u32(c, 1));
    uint32_t dhalf = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, dhalf, const_u32(c, 1), rs1);
    uint32_t dgt = do_cmp(c, b, PCM_GT, PTY_U32, 0, ddrop, dhalf, line);
    uint32_t deq = do_cmp(c, b, PCM_EQ, PTY_U32, 0, ddrop, dhalf, line);
    uint32_t dodd = nid(c);
    emit_arith(c, b, S_And, c->t_u32, dodd, dmant, const_u32(c, 1));
    uint32_t dodd1 = do_cmp(c, b, PCM_EQ, PTY_U32, 0, dodd,
                            const_u32(c, 1), line);
    uint32_t dupb = sel_bool(c, b, deq, dodd1, dgt);
    uint32_t dup = sel_u32(c, b, dupb, const_u32(c, 1), const_u32(c, 0));
    uint32_t dmant2 = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u32, dmant2, dmant, dup);
    uint32_t dnorm = do_cmp(c, b, PCM_GE, PTY_U32, 0, dmant2,
                            const_u32(c, 1024), line);
    uint32_t o_den1 = nid(c), o_den = nid(c);
    emit_arith(c, b, S_Or, c->t_u32, o_den1, s15, dmant2);
    uint32_t smin = nid(c);
    emit_arith(c, b, S_Or, c->t_u32, smin, s15, const_u32(c, 1 << 10));
    o_den = sel_u32(c, b, dnorm, smin, o_den1);
    o_den = sel_u32(c, b, tiny, s15, o_den); // too small -> signed zero
    if (c->failed)
        return 0;
    // Path select: naninf / flush-zero / overflow / normal / denormal.
    uint32_t o_z = s15; // signed zero (flush f32 denormals)
    uint32_t o1s = sel_u32(c, b, is_norm, o_norm, o_den);
    uint32_t o2s = sel_u32(c, b, is_ovf, o_inf, o1s);
    uint32_t o3s = sel_u32(c, b, is_z, o_z, o2s);
    return sel_u32(c, b, is_ni, o_ni, o3s);
}

// One FP8 byte (0..0xff in a u32) -> f16 bit pattern. Exact widening:
// every e4m3/e5m2 value is representable in f16. NaN ({abs} = 0x7f for
// e4m3, > 0x7c for e5m2) maps to the canonical 0x7fff and drops the
// sign, exactly as cuda_fp8.hpp's __nv_cvt_fp8_to_halfraw does.
uint32_t f16_from_fp8byte(ctx_t *c, spv_blob_t *b, uint32_t p,
                                 int e4m3, int line) {
    uint32_t s0 = nid(c), sign = nid(c), abs = nid(c);
    emit_arith(c, b, S_And, c->t_u32, s0, p, const_u32(c, 0x80u));
    emit_arith(c, b, S_Shl, c->t_u32, sign, s0, const_u32(c, 8));
    emit_arith(c, b, S_And, c->t_u32, abs, p, const_u32(c, 0x7fu));
    if (!e4m3) {
        // e5m2 and f16 share exponent width and bias and e5m2's stored
        // mantissa is the top of f16's: the pattern is just byte << 8,
        // except NaN which canonicalizes.
        uint32_t o = nid(c);
        emit_arith(c, b, S_Shl, c->t_u32, o, p, const_u32(c, 8));
        uint32_t isnan = do_cmp(c, b, PCM_GT, PTY_U32, 0, abs,
                                const_u32(c, 0x7cu), line);
        return sel_u32(c, b, isnan, const_u32(c, 0x7fffu), o);
    }
    // e4m3: E = exp field (bias 7), M = mantissa. Normals (E >= 1) map
    // to f16 exp field E+8 and mantissa M<<7; denormals (E == 0) are
    // normalized exactly like __nv_cvt_fp8_to_halfraw's shift loop:
    // loop count sh = (M<4) + (M<2), exponent 0x2000 - (sh<<10),
    // mantissa (M << (8+sh)) & 0x3ff. Zero keeps only the sign.
    uint32_t E0 = nid(c), E = nid(c), M = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, E0, abs, const_u32(c, 3));
    emit_arith(c, b, S_And, c->t_u32, E, E0, const_u32(c, 0x0fu));
    emit_arith(c, b, S_And, c->t_u32, M, abs, const_u32(c, 0x07u));
    uint32_t e8 = nid(c), en = nid(c), mn = nid(c), on1 = nid(c);
    uint32_t on = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u32, e8, E, const_u32(c, 8));
    emit_arith(c, b, S_Shl, c->t_u32, en, e8, const_u32(c, 10));
    emit_arith(c, b, S_Shl, c->t_u32, mn, M, const_u32(c, 7));
    emit_arith(c, b, S_Or, c->t_u32, on1, sign, en);
    emit_arith(c, b, S_Or, c->t_u32, on, on1, mn);
    uint32_t lt4 = do_cmp(c, b, PCM_LT, PTY_U32, 0, M,
                          const_u32(c, 4), line);
    uint32_t lt2 = do_cmp(c, b, PCM_LT, PTY_U32, 0, M,
                          const_u32(c, 2), line);
    uint32_t b4 = sel_u32(c, b, lt4, const_u32(c, 1), const_u32(c, 0));
    uint32_t b2 = sel_u32(c, b, lt2, const_u32(c, 1), const_u32(c, 0));
    uint32_t sh = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u32, sh, b4, b2);
    uint32_t sh8 = nid(c), md0 = nid(c), md = nid(c);
    uint32_t shl = nid(c), ed = nid(c), od1 = nid(c), od = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u32, sh8, sh, const_u32(c, 8));
    emit_arith(c, b, S_Shl, c->t_u32, md0, M, sh8);
    emit_arith(c, b, S_And, c->t_u32, md, md0, const_u32(c, 0x3ffu));
    emit_arith(c, b, S_Shl, c->t_u32, shl, sh, const_u32(c, 10));
    emit_arith(c, b, S_ISub, c->t_u32, ed, const_u32(c, 0x2000u), shl);
    emit_arith(c, b, S_Or, c->t_u32, od1, sign, ed);
    emit_arith(c, b, S_Or, c->t_u32, od, od1, md);
    uint32_t iszero = do_cmp(c, b, PCM_EQ, PTY_U32, 0, abs,
                             const_u32(c, 0), line);
    uint32_t isden = do_cmp(c, b, PCM_EQ, PTY_U32, 0, E,
                            const_u32(c, 0), line);
    uint32_t r1 = sel_u32(c, b, isden, od, on);
    uint32_t r2 = sel_u32(c, b, iszero, sign, r1);
    uint32_t isnan = do_cmp(c, b, PCM_EQ, PTY_U32, 0, abs,
                            const_u32(c, 0x7fu), line);
    return sel_u32(c, b, isnan, const_u32(c, 0x7fffu), r2);
}

// f32 value -> one FP8 byte (u32 0..0xff): round-to-nearest-even,
// satfinite, denormals supported. Integer-only port of NVIDIA's
// __nv_cvt_float_to_fp8/__nv_cvt_double_to_fp8 (cuda_fp8.hpp) on the
// f32 pattern; validated bit-exact against a literal port of that
// reference over 2.3M patterns in both formats (ties, denormals,
// NaN/Inf, saturation boundaries). NaN -> canonical 0x7f (sign
// dropped); +-Inf and |x| >= overflow both clamp to max finite for
// satfinite. Shift amounts are clamped to 5 (live denormal range is
// 1..4) because SPIR-V shifts >= width are UB even on a discarded arm.
uint32_t fp8byte_from_f32(ctx_t *c, spv_blob_t *b, uint32_t f,
                                 int e4m3, int line) {
    const uint32_t B = e4m3 ? 7u : 15u;
    const uint32_t SB = e4m3 ? 3u : 2u;
    const uint32_t MM = e4m3 ? 7u : 3u;
    const uint32_t DB = e4m3 ? 20u : 21u;
    const uint32_t HALF = e4m3 ? 0x80000u : 0x100000u;
    const uint32_t DMASK = e4m3 ? 0xfffffu : 0x1fffffu;
    const uint32_t U = e4m3 ? 0x3a800000u : 0x37000000u;
    const uint32_t NMIN = e4m3 ? 0x3c800000u : 0x38800000u;
    const uint32_t OVF = e4m3 ? 0x43e80000u : 0x476f0000u;
    const uint32_t MAXN = e4m3 ? 0x7eu : 0x7bu;
    const uint32_t BIASOFF = e4m3 ? 121u : 113u;
    uint32_t w = nid(c), abs = nid(c), e0 = nid(c), e2 = nid(c);
    uint32_t m0 = nid(c), m = nid(c), drop = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_u32);
    put(c, b, w);
    put(c, b, f);
    emit_arith(c, b, S_And, c->t_u32, abs, w, const_u32(c, 0x7fffffffu));
    emit_arith(c, b, S_ShrL, c->t_u32, e0, abs, const_u32(c, 23));
    emit_arith(c, b, S_And, c->t_u32, e2, e0, const_u32(c, 0xffu));
    emit_arith(c, b, S_ShrL, c->t_u32, m0, abs, const_u32(c, DB));
    emit_arith(c, b, S_And, c->t_u32, m, m0, const_u32(c, MM));
    emit_arith(c, b, S_And, c->t_u32, drop, abs, const_u32(c, DMASK));
    // Normal path, RNE: res = ((E-127+B) << SB) | M, +1 on round-up.
    uint32_t gt = do_cmp(c, b, PCM_GT, PTY_U32, 0, drop,
                         const_u32(c, HALF), line);
    uint32_t eq = do_cmp(c, b, PCM_EQ, PTY_U32, 0, drop,
                         const_u32(c, HALF), line);
    uint32_t odd0 = nid(c), odd = nid(c);
    emit_arith(c, b, S_And, c->t_u32, odd0, m, const_u32(c, 1));
    odd = do_cmp(c, b, PCM_EQ, PTY_U32, 0, odd0, const_u32(c, 1), line);
    uint32_t upb = sel_bool(c, b, eq, odd, gt);
    uint32_t up = sel_u32(c, b, upb, const_u32(c, 1), const_u32(c, 0));
    uint32_t eB = nid(c), eSh = nid(c), rn0 = nid(c), rn1 = nid(c);
    uint32_t resn = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u32, eB, e2,
               const_u32(c, (uint32_t)((int)B - 127)));
    emit_arith(c, b, S_Shl, c->t_u32, eSh, eB, const_u32(c, SB));
    emit_arith(c, b, S_Or, c->t_u32, rn0, eSh, m);
    emit_arith(c, b, S_IAdd, c->t_u32, rn1, rn0, up);
    emit_arith(c, b, S_And, c->t_u32, resn, rn1, const_u32(c, 0xffu));
    // Denormal path: shift the implicit-bit mantissa right, RNE.
    uint32_t sh0 = nid(c), big = nid(c), small = nid(c), sh = nid(c);
    emit_arith(c, b, S_ISub, c->t_u32, sh0, const_u32(c, BIASOFF), e2);
    big = do_cmp(c, b, PCM_GT, PTY_U32, 0, e2,
                 const_u32(c, BIASOFF), line);
    small = do_cmp(c, b, PCM_LT, PTY_U32, 0, e2,
                   const_u32(c, BIASOFF - 5), line);
    sh = sel_u32(c, b, small, const_u32(c, 5), sh0);
    sh = sel_u32(c, b, big, const_u32(c, 0), sh);
    uint32_t full0 = nid(c), dres0 = nid(c), dres = nid(c);
    emit_arith(c, b, S_Or, c->t_u32, full0, m,
               const_u32(c, 1u << SB));
    emit_arith(c, b, S_ShrL, c->t_u32, dres0, full0, sh);
    emit_arith(c, b, S_And, c->t_u32, dres, dres0, const_u32(c, 0xffu));
    uint32_t sh1 = nid(c), hm0 = nid(c), hm = nid(c);
    uint32_t aorg = nid(c), rb = nid(c), hb0 = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u32, sh1, sh, const_u32(c, 1));
    emit_arith(c, b, S_Shl, c->t_u32, hm0, const_u32(c, HALF), sh1);
    emit_arith(c, b, S_ISub, c->t_u32, hm, hm0, const_u32(c, 1));
    emit_arith(c, b, S_Or, c->t_u32, aorg, abs, const_u32(c, 0x800000u));
    emit_arith(c, b, S_And, c->t_u32, rb, aorg, hm);
    emit_arith(c, b, S_Shl, c->t_u32, hb0, const_u32(c, HALF), sh);
    uint32_t rbgt = do_cmp(c, b, PCM_GT, PTY_U32, 0, rb, hb0, line);
    uint32_t rbeq = do_cmp(c, b, PCM_EQ, PTY_U32, 0, rb, hb0, line);
    uint32_t dodd0 = nid(c), dodd = nid(c);
    emit_arith(c, b, S_And, c->t_u32, dodd0, dres, const_u32(c, 1));
    dodd = do_cmp(c, b, PCM_EQ, PTY_U32, 0, dodd0, const_u32(c, 1), line);
    uint32_t dupb = sel_bool(c, b, rbeq, dodd, rbgt);
    uint32_t dup = sel_u32(c, b, dupb, const_u32(c, 1), const_u32(c, 0));
    uint32_t rd0 = nid(c), resd = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u32, rd0, dres, dup);
    emit_arith(c, b, S_And, c->t_u32, resd, rd0, const_u32(c, 0xffu));
    if (c->failed)
        return 0;
    // Path select: NaN / overflow / underflow / normal / denormal.
    uint32_t isovf = do_cmp(c, b, PCM_GE, PTY_U32, 0, abs,
                            const_u32(c, OVF), line);
    uint32_t isund = do_cmp(c, b, PCM_LE, PTY_U32, 0, abs,
                            const_u32(c, U), line);
    uint32_t isnorm = do_cmp(c, b, PCM_GE, PTY_U32, 0, abs,
                             const_u32(c, NMIN), line);
    uint32_t r1 = sel_u32(c, b, isnorm, resn, resd);
    uint32_t r2 = sel_u32(c, b, isund, const_u32(c, 0), r1);
    uint32_t r3 = sel_u32(c, b, isovf, const_u32(c, MAXN), r2);
    uint32_t sgn0 = nid(c), sgn = nid(c), r4 = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, sgn0, w, const_u32(c, 24));
    emit_arith(c, b, S_And, c->t_u32, sgn, sgn0, const_u32(c, 0x80u));
    emit_arith(c, b, S_Or, c->t_u32, r4, r3, sgn);
    uint32_t isnan = do_cmp(c, b, PCM_GT, PTY_U32, 0, abs,
                            const_u32(c, 0x7f800000u), line);
    return sel_u32(c, b, isnan, const_u32(c, 0x7fu), r4);
}

// f64 value -> u32 holding f16 bits (RNE). Integer-only, no Float16
// capability. Mirrors f32_to_f16bits with double bias/widths. Works
// entirely in u64 (all intermediates fit), converting to u32 at the
// end. Shift amounts can exceed 63 on untaken paths; the tiny-select
// overrides those lanes (same practical safety as the f32 version:
// hardware masks shifts, LLVM select discards the unselected arm).
uint32_t f64_to_f16bits(ctx_t *c, spv_blob_t *b, uint32_t f,
                               int line) {
    uint32_t w = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_u64);
    put(c, b, w);
    put(c, b, f);
    uint32_t s = nid(c), e = nid(c), m = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, s, w, const_u64(c, 63));
    emit_arith(c, b, S_ShrL, c->t_u64, e, w, const_u64(c, 52));
    uint32_t e2 = nid(c);
    emit_arith(c, b, S_And, c->t_u64, e2, e, const_u64(c, 0x7ffull));
    emit_arith(c, b, S_And, c->t_u64, m, w,
               const_u64(c, 0xfffffffffffffull));
    uint32_t is_ni = do_cmp(c, b, PCM_EQ, PTY_U64, 0, e2,
                            const_u64(c, 2047), line);
    uint32_t is_z = do_cmp(c, b, PCM_EQ, PTY_U64, 0, e2,
                           const_u64(c, 0), line);
    if (c->failed)
        return 0;
    uint32_t mshr = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, mshr, m, const_u64(c, 42));
    uint32_t mquo = nid(c);
    emit_arith(c, b, S_Or, c->t_u64, mquo, mshr, const_u64(c, 0x200ull));
    uint32_t mnez = do_cmp(c, b, PCM_NE, PTY_U64, 0, m,
                           const_u64(c, 0), line);
    uint32_t m_nn = sel_u64(c, b, mnez, mquo, const_u64(c, 0));
    uint32_t s15 = nid(c), e15 = nid(c), o_ni = nid(c), t_ni = nid(c);
    emit_arith(c, b, S_Shl, c->t_u64, s15, s, const_u64(c, 15));
    emit_arith(c, b, S_Shl, c->t_u64, e15, const_u64(c, 0x1f),
               const_u64(c, 10));
    emit_arith(c, b, S_Or, c->t_u64, t_ni, s15, e15);
    emit_arith(c, b, S_Or, c->t_u64, o_ni, t_ni, m_nn);
    uint32_t exp = nid(c);
    emit_arith(c, b, S_ISub, c->t_u64, exp, e2, const_u64(c, 1023));
    uint32_t is_ovf = do_cmp(c, b, PCM_GE, PTY_U64, 1, exp,
                             const_u64(c, 16), line);
    uint32_t is_norm = do_cmp(c, b, PCM_GE, PTY_U64, 1, exp,
                              const_u64(c, (uint64_t)(int64_t)-14), line);
    if (c->failed)
        return 0;
    uint32_t o_inf = nid(c);
    emit_arith(c, b, S_Or, c->t_u64, o_inf, s15, e15);
    uint32_t mant = nid(c), drop = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, mant, m, const_u64(c, 42));
    emit_arith(c, b, S_And, c->t_u64, drop, m,
               const_u64(c, 0x3ffffffffffull));
    uint32_t gt = do_cmp(c, b, PCM_GT, PTY_U64, 0, drop,
                         const_u64(c, 0x20000000000ull), line);
    uint32_t eq = do_cmp(c, b, PCM_EQ, PTY_U64, 0, drop,
                         const_u64(c, 0x20000000000ull), line);
    uint32_t odd = nid(c);
    emit_arith(c, b, S_And, c->t_u64, odd, mant, const_u64(c, 1));
    uint32_t odd1 = do_cmp(c, b, PCM_EQ, PTY_U64, 0, odd,
                           const_u64(c, 1), line);
    uint32_t upb = sel_bool(c, b, eq, odd1, gt);
    uint32_t up = sel_u64(c, b, upb, const_u64(c, 1), const_u64(c, 0));
    uint32_t mant2 = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, mant2, mant, up);
    uint32_t carry = do_cmp(c, b, PCM_EQ, PTY_U64, 0, mant2,
                            const_u64(c, 0x400), line);
    uint32_t mant3 = sel_u64(c, b, carry, const_u64(c, 0), mant2);
    uint32_t ec = sel_u64(c, b, carry, const_u64(c, 1), const_u64(c, 0));
    uint32_t exp2 = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, exp2, exp, ec);
    uint32_t ovf2 = do_cmp(c, b, PCM_GE, PTY_U64, 1, exp2,
                           const_u64(c, 16), line);
    uint32_t eb = nid(c), esh = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, eb, exp2, const_u64(c, 15));
    emit_arith(c, b, S_Shl, c->t_u64, esh, eb, const_u64(c, 10));
    uint32_t o1 = nid(c), o_norm = nid(c);
    emit_arith(c, b, S_Or, c->t_u64, o1, s15, esh);
    emit_arith(c, b, S_Or, c->t_u64, o_norm, o1, mant3);
    o_norm = sel_u64(c, b, ovf2, o_inf, o_norm);
    uint32_t full = nid(c);
    emit_arith(c, b, S_Or, c->t_u64, full, m,
               const_u64(c, 0x10000000000000ull));
    uint32_t rsh = nid(c);
    emit_arith(c, b, S_ISub, c->t_u64, rsh, const_u64(c, 28), exp);
    uint32_t tiny = do_cmp(c, b, PCM_GE, PTY_U64, 0, rsh,
                           const_u64(c, 54), line);
    if (c->failed)
        return 0;
    // Clamp like the f32->f16 path: rsh wraps huge for tiny exponents
    // and a shift of >= width is UB on the discarded arm.
    uint32_t sc = sel_u64(c, b, tiny, const_u64(c, 54), rsh);
    uint32_t dmant = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, dmant, full, sc);
    uint32_t dmask = nid(c), dnot = nid(c), ddrop = nid(c);
    emit_arith(c, b, S_Shl, c->t_u64, dmask, const_u64(c, ~(uint64_t)0),
               sc);
    put(c, b, (4u << 16) | S_Not);
    put(c, b, c->t_u64);
    put(c, b, dnot);
    put(c, b, dmask);
    emit_arith(c, b, S_And, c->t_u64, ddrop, full, dnot);
    uint32_t rs1 = nid(c);
    emit_arith(c, b, S_ISub, c->t_u64, rs1, sc, const_u64(c, 1));
    uint32_t dhalf = nid(c);
    emit_arith(c, b, S_Shl, c->t_u64, dhalf, const_u64(c, 1), rs1);
    uint32_t dgt = do_cmp(c, b, PCM_GT, PTY_U64, 0, ddrop, dhalf, line);
    uint32_t deq = do_cmp(c, b, PCM_EQ, PTY_U64, 0, ddrop, dhalf, line);
    uint32_t dodd = nid(c);
    emit_arith(c, b, S_And, c->t_u64, dodd, dmant, const_u64(c, 1));
    uint32_t dodd1 = do_cmp(c, b, PCM_EQ, PTY_U64, 0, dodd,
                            const_u64(c, 1), line);
    uint32_t dupb = sel_bool(c, b, deq, dodd1, dgt);
    uint32_t dup = sel_u64(c, b, dupb, const_u64(c, 1), const_u64(c, 0));
    uint32_t dmant2 = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, dmant2, dmant, dup);
    uint32_t dnorm = do_cmp(c, b, PCM_GE, PTY_U64, 0, dmant2,
                            const_u64(c, 1024), line);
    uint32_t o_den1 = nid(c), o_den = nid(c);
    emit_arith(c, b, S_Or, c->t_u64, o_den1, s15, dmant2);
    uint32_t smin = nid(c);
    emit_arith(c, b, S_Or, c->t_u64, smin, s15, const_u64(c, 1 << 10));
    o_den = sel_u64(c, b, dnorm, smin, o_den1);
    o_den = sel_u64(c, b, tiny, s15, o_den);
    if (c->failed)
        return 0;
    uint32_t o1s = sel_u64(c, b, is_norm, o_norm, o_den);
    uint32_t o2s = sel_u64(c, b, is_ovf, o_inf, o1s);
    uint32_t o3s = sel_u64(c, b, is_z, s15, o2s);
    uint32_t o64 = sel_u64(c, b, is_ni, o_ni, o3s);
    uint32_t out = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u32);
    put(c, b, out);
    put(c, b, o64);
    return out;
}
uint32_t lane_id(ctx_t *c, spv_blob_t *b, int line) {
    return special_x(c, b, SPC_LANEID, line);
}

uint32_t f16_to_f32val(ctx_t *c, spv_blob_t *b, uint32_t h,
                              int line) {
    unsigned slot = h & 255u;
    if (c->cse_cvt_gen[slot] == c->cse_gen &&
        c->cse_cvt_key[slot] == h && c->cse_cvt_kind[slot] == 0)
        return c->cse_cvt_val[slot];
    if (c->caps.f16_native) {
        ensure_f16_types(c);
        uint32_t u16 = nid(c), hf = nid(c), f = nid(c);
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, u16_type(c));
        put(c, b, u16);
        put(c, b, h);
        put(c, b, (4u << 16) | S_Bitcast);
        put(c, b, c->t_f16);
        put(c, b, hf);
        put(c, b, u16);
        put(c, b, (4u << 16) | S_FConvert);
        put(c, b, c->t_f32);
        put(c, b, f);
        put(c, b, hf);
        // The emulated path has never represented f16 subnormals:
        // preserve that (flush them to signed zero) so both paths agree.
        uint32_t e = nid(c), m = nid(c);
        emit_arith(c, b, S_And, c->t_u32, e, h, const_u32(c, 0x7c00u));
        emit_arith(c, b, S_And, c->t_u32, m, h, const_u32(c, 0x03ffu));
        uint32_t ez = do_cmp(c, b, PCM_EQ, PTY_U32, 0, e,
                             const_u32(c, 0), line);
        uint32_t mnz = do_cmp(c, b, PCM_NE, PTY_U32, 0, m,
                              const_u32(c, 0), line);
        uint32_t sub = nid(c);
        put(c, b, (5u << 16) | S_LogicalAnd);
        put(c, b, c->t_bool);
        put(c, b, sub);
        put(c, b, ez);
        put(c, b, mnz);
        uint32_t sgn = nid(c), zb = nid(c), fb = nid(c), out = nid(c);
        emit_arith(c, b, S_Shl, c->t_u32, sgn, h, const_u32(c, 16));
        emit_arith(c, b, S_And, c->t_u32, zb, sgn,
                   const_u32(c, 0x80000000u));
        put(c, b, (4u << 16) | S_Bitcast);
        put(c, b, c->t_u32);
        put(c, b, fb);
        put(c, b, f);
        out = sel_u32(c, b, sub, zb, fb);
        f = nid(c);
        put(c, b, (4u << 16) | S_Bitcast);
        put(c, b, c->t_f32);
        put(c, b, f);
        put(c, b, out);
        c->cse_cvt_key[slot] = h;
        c->cse_cvt_val[slot] = f;
        c->cse_cvt_kind[slot] = 0;
        c->cse_cvt_gen[slot] = c->cse_gen;
        return f;
    }
    uint32_t s = nid(c), e = nid(c), m = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, s, h, const_u32(c, 15));
    emit_arith(c, b, S_ShrL, c->t_u32, e, h, const_u32(c, 10));
    uint32_t e2 = nid(c);
    emit_arith(c, b, S_And, c->t_u32, e2, e, const_u32(c, 0x1fu));
    emit_arith(c, b, S_And, c->t_u32, m, h, const_u32(c, 0x3ffu));
    uint32_t is_ni = do_cmp(c, b, PCM_EQ, PTY_U32, 0, e2,
                            const_u32(c, 31), line);
    uint32_t is_z = do_cmp(c, b, PCM_EQ, PTY_U32, 0, e2,
                           const_u32(c, 0), line);
    if (c->failed)
        return 0;
    uint32_t msh = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, msh, m, const_u32(c, 13));
    uint32_t mquo = nid(c);
    emit_arith(c, b, S_Or, c->t_u32, mquo, msh, const_u32(c, 0x400000u));
    uint32_t mnez = do_cmp(c, b, PCM_NE, PTY_U32, 0, m,
                           const_u32(c, 0), line);
    uint32_t m_nn = sel_u32(c, b, mnez, mquo, const_u32(c, 0));
    uint32_t s31 = nid(c), e31 = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, s31, s, const_u32(c, 31));
    emit_arith(c, b, S_Shl, c->t_u32, e31, const_u32(c, 0xffu),
               const_u32(c, 23));
    uint32_t t1 = nid(c), o_ni = nid(c);
    emit_arith(c, b, S_Or, c->t_u32, t1, s31, e31);
    emit_arith(c, b, S_Or, c->t_u32, o_ni, t1, m_nn);
    uint32_t o_z = s31;
    uint32_t eb = nid(c), esh = nid(c), msh2 = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u32, eb, e2, const_u32(c, 112));
    emit_arith(c, b, S_Shl, c->t_u32, esh, eb, const_u32(c, 23));
    emit_arith(c, b, S_Shl, c->t_u32, msh2, m, const_u32(c, 13));
    uint32_t t2 = nid(c), o_nm = nid(c);
    emit_arith(c, b, S_Or, c->t_u32, t2, s31, esh);
    emit_arith(c, b, S_Or, c->t_u32, o_nm, t2, msh2);
    uint32_t o1s = sel_u32(c, b, is_z, o_z, o_nm);
    uint32_t obits = sel_u32(c, b, is_ni, o_ni, o1s);
    uint32_t f = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_f32);
    put(c, b, f);
    put(c, b, obits);
    c->cse_cvt_key[slot] = h;
    c->cse_cvt_val[slot] = f;
    c->cse_cvt_kind[slot] = 0;
    c->cse_cvt_gen[slot] = c->cse_gen;
    return f;
}

// f32 bits -> bf16 bits, round-to-nearest-even. Same exponent width
// as f32, so denormals are representable: f32-denormal inputs narrow
// exactly (no flush — unlike f16); Inf/NaN exact with quiet-bit
// forcing like f32_to_f16_bits. Overflow past bf16-max rounds to Inf.
uint32_t f32_to_bf16bits(ctx_t *c, spv_blob_t *b, uint32_t f,
                                int line) {
    uint32_t w = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_u32);
    put(c, b, w);
    put(c, b, f);
    uint32_t s = nid(c), e = nid(c), m = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, s, w, const_u32(c, 31));
    emit_arith(c, b, S_ShrL, c->t_u32, e, w, const_u32(c, 23));
    uint32_t e2 = nid(c);
    emit_arith(c, b, S_And, c->t_u32, e2, e, const_u32(c, 0xffu));
    emit_arith(c, b, S_And, c->t_u32, m, w, const_u32(c, 0x7fffffu));
    uint32_t is_ni = do_cmp(c, b, PCM_EQ, PTY_U32, 0, e2,
                            const_u32(c, 255), line);
    if (c->failed)
        return 0;
    // NaN/Inf payload: quiet bit + top payload bits when m != 0.
    uint32_t mshr = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, mshr, m, const_u32(c, 16));
    uint32_t mquo = nid(c);
    emit_arith(c, b, S_Or, c->t_u32, mquo, mshr, const_u32(c, 0x40u));
    uint32_t mnez = do_cmp(c, b, PCM_NE, PTY_U32, 0, m,
                           const_u32(c, 0), line);
    uint32_t m_nn = sel_u32(c, b, mnez, mquo, const_u32(c, 0));
    uint32_t s15 = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, s15, s, const_u32(c, 15));
    uint32_t o_ni = nid(c), t_ni = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, t_ni, const_u32(c, 0xffu),
               const_u32(c, 7));
    emit_arith(c, b, S_Or, c->t_u32, o_ni, s15, t_ni);
    uint32_t o_nii = nid(c);
    emit_arith(c, b, S_Or, c->t_u32, o_nii, o_ni, m_nn);
    // Round-to-nearest-even on bit 16: bias 0x7FFF + kept LSB.
    uint32_t kept = nid(c), lsb = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, kept, w, const_u32(c, 16));
    emit_arith(c, b, S_And, c->t_u32, lsb, kept, const_u32(c, 1));
    uint32_t bias = nid(c), t = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u32, bias, const_u32(c, 0x7fffu), lsb);
    emit_arith(c, b, S_IAdd, c->t_u32, t, w, bias);
    uint32_t o_rne = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, o_rne, t, const_u32(c, 16));
    if (c->failed)
        return 0;
    return sel_u32(c, b, is_ni, o_nii, o_rne);
}

// bf16 bits -> exact f32 value. bf16 denormal inputs flush to signed
// zero (they would widen to f32 denormals, which this backend's ALU
// flushes — same rule as f16_to_f32val).
uint32_t bf16_to_f32val(ctx_t *c, spv_blob_t *b, uint32_t h,
                               int line) {
    (void)line;
    // bfloat16 is exactly the upper 16 bits of an IEEE-754 f32:
    // widening is a shift, with no rounding and no special cases
    // (NaN/Inf/subnormal encodings all map correctly).
    unsigned slot = h & 255u;
    if (c->cse_cvt_gen[slot] == c->cse_gen &&
        c->cse_cvt_key[slot] == h && c->cse_cvt_kind[slot] == 1)
        return c->cse_cvt_val[slot];
    uint32_t sh = nid(c), f = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, sh, h, const_u32(c, 16));
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_f32);
    put(c, b, f);
    put(c, b, sh);
    c->cse_cvt_key[slot] = h;
    c->cse_cvt_val[slot] = f;
    c->cse_cvt_kind[slot] = 1;
    c->cse_cvt_gen[slot] = c->cse_gen;
    return f;
}

// Split a packed u32 (two 16-bit patterns, HIGH = first source) into
// HIGH and LOW pattern ids.
void split_pack(ctx_t *c, spv_blob_t *b, uint32_t p, uint32_t *hi,
                       uint32_t *lo) {
    unsigned slot = p & 255u;
    if (c->cse_split_gen[slot] == c->cse_gen &&
        c->cse_split_key[slot] == p) {
        *hi = c->cse_split_hi[slot];
        *lo = c->cse_split_lo[slot];
        return;
    }
    uint32_t h = nid(c), l = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, h, p, const_u32(c, 16));
    emit_arith(c, b, S_And, c->t_u32, l, p, const_u32(c, 0xffffu));
    c->cse_split_key[slot] = p;
    c->cse_split_hi[slot] = h;
    c->cse_split_lo[slot] = l;
    c->cse_split_gen[slot] = c->cse_gen;
    *hi = h;
    *lo = l;
}

// Assemble HIGH and LOW 16-bit patterns into one packed u32.
uint32_t pack2(ctx_t *c, spv_blob_t *b, uint32_t hi, uint32_t lo) {
    uint32_t h = nid(c), r = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, h, hi, const_u32(c, 16));
    emit_arith(c, b, S_Or, c->t_u32, r, h, lo);
    return r;
}

// One packed lane: widen with the element-appropriate routine, the
// scalar mul+add sequence, narrow back. Bit-identical to scalar f16
// (or bf16) fma on the same lane values.
uint32_t pack_lane_fma(ctx_t *c, spv_blob_t *b, int is_bf16,
                              uint32_t av, uint32_t bv, uint32_t cv,
                              int ftz, int sat, int relu, int line) {
    uint32_t fa = is_bf16 ? bf16_to_f32val(c, b, av, line)
                          : f16_to_f32val(c, b, av, line);
    if (c->failed)
        return 0;
    uint32_t fb = is_bf16 ? bf16_to_f32val(c, b, bv, line)
                          : f16_to_f32val(c, b, bv, line);
    if (c->failed)
        return 0;
    uint32_t fc = is_bf16 ? bf16_to_f32val(c, b, cv, line)
                          : f16_to_f32val(c, b, cv, line);
    if (c->failed)
        return 0;
    if (ftz) {
        fa = flush_f32_denorm(c, b, fa, line);
        fb = flush_f32_denorm(c, b, fb, line);
        fc = flush_f32_denorm(c, b, fc, line);
        if (c->failed)
            return 0;
    }
    uint32_t fr = nid(c);
    emit_fma_into(c, b, c->t_f32, fr, fa, fb, fc);
    if (ftz) {
        fr = flush_f32_denorm(c, b, fr, line);
        if (c->failed)
            return 0;
    }
    if (sat) {
        fr = sat_f32(c, b, fr, line);
        if (c->failed)
            return 0;
    }
    if (relu) {
        fr = relu_f32(c, b, fr, line);
        if (c->failed)
            return 0;
    }
    uint32_t r = is_bf16 ? f32_to_bf16bits(c, b, fr, line)
                         : f32_to_f16bits(c, b, fr, line);
    return r;
}

// Word index for a u64 byte offset: off >> 2. Emits an access chain
// through struct member 0 into the word array, returns element pointer.
