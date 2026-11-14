// Micro-float cvt lowering: e2m1 (f4), e2m3/e3m2 (f6), e4m3/e5m2
// (fp8) x2/x4, ue8m0 scale factors, s2f6 fixed point, and the
// .scaled::n2::ue8m0 block-scaled forms.
//
// Values are converted on the GPU with integer/float SPIR-V ops.  The
// narrow formats have no infinity; all bit patterns are finite except
// ue8m0's 0xff NaN.  Out-of-range and NaN inputs saturate per the PTX
// satfinite rules.
#include <stdint.h>
#include "spirv_internal.h"

static uint32_t cvt_bitcast_u32(ctx_t *c, spv_blob_t *b, uint32_t f) {
    uint32_t id = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_u32);
    put(c, b, id);
    put(c, b, f);
    return id;
}

static uint32_t cvt_bitcast_f32(ctx_t *c, spv_blob_t *b, uint32_t bits) {
    uint32_t id = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_f32);
    put(c, b, id);
    put(c, b, bits);
    return id;
}

static uint32_t cvt_shl(ctx_t *c, spv_blob_t *b, uint32_t a, uint32_t s) {
    uint32_t id = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, id, a, s);
    return id;
}
static uint32_t cvt_shr(ctx_t *c, spv_blob_t *b, uint32_t a, uint32_t s) {
    uint32_t id = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, id, a, s);
    return id;
}
static uint32_t cvt_and(ctx_t *c, spv_blob_t *b, uint32_t a, uint32_t m) {
    uint32_t id = nid(c);
    emit_arith(c, b, S_And, c->t_u32, id, a, m);
    return id;
}
static uint32_t cvt_or(ctx_t *c, spv_blob_t *b, uint32_t a, uint32_t m) {
    uint32_t id = nid(c);
    emit_arith(c, b, S_Or, c->t_u32, id, a, m);
    return id;
}
static uint32_t cvt_add(ctx_t *c, spv_blob_t *b, uint32_t a, uint32_t m) {
    uint32_t id = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u32, id, a, m);
    return id;
}
static uint32_t cvt_sub(ctx_t *c, spv_blob_t *b, uint32_t a, uint32_t m) {
    uint32_t id = nid(c);
    emit_arith(c, b, S_ISub, c->t_u32, id, a, m);
    return id;
}

static uint32_t cvt_bool_const(ctx_t *c, int v) {
    uint32_t id = nid(c);
    put(c, &c->types, (3u << 16) | (v ? S_ConstantTrue : S_ConstantFalse));
    put(c, &c->types, c->t_bool);
    put(c, &c->types, id);
    return id;
}

static uint32_t cvt_sel_bool(ctx_t *c, spv_blob_t *b, uint32_t cond,
                             uint32_t a, uint32_t bb) {
    uint32_t id = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, c->t_bool);
    put(c, b, id);
    put(c, b, cond);
    put(c, b, a);
    put(c, b, bb);
    return id;
}

static uint32_t cvt_sel_f32(ctx_t *c, spv_blob_t *b, uint32_t cond,
                            uint32_t a, uint32_t bb) {
    uint32_t id = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, c->t_f32);
    put(c, b, id);
    put(c, b, cond);
    put(c, b, a);
    put(c, b, bb);
    return id;
}

// .relu: negative results become +0 (positive NaN is left alone).
static uint32_t cvt_relu_f32(ctx_t *c, spv_blob_t *b, uint32_t f,
                             int line) {
    uint32_t neg = do_cmp(c, b, PCM_LT, PTY_F32, 0, f,
                          const_f32(c, 0), line);
    return cvt_sel_f32(c, b, neg, const_f32(c, 0), f);
}

// 2^k as f32 bits, for k in the signed 32-bit range.
static uint32_t cvt_pow2_f32(ctx_t *c, spv_blob_t *b, uint32_t k) {
    uint32_t kp = cvt_add(c, b, k, const_u32(c, 127));
    uint32_t normal = cvt_shl(c, b, kp, const_u32(c, 23));
    uint32_t ge_norm = do_cmp(c, b, PCM_GE, PTY_U32, 1, k,
                              const_u32(c, (uint32_t)-126), 0);
    uint32_t ge_sub = do_cmp(c, b, PCM_GE, PTY_U32, 1, k,
                             const_u32(c, (uint32_t)-149), 0);
    // Keep the unused subnormal arm's shift below 32: SPIR-V shifts by
    // the full width are undefined even when the result is discarded.
    uint32_t ks = sel_u32(c, b, ge_norm, const_u32(c, 31),
                          cvt_add(c, b, k, const_u32(c, 149)));
    ks = sel_u32(c, b, ge_sub, ks, const_u32(c, 0));
    uint32_t sub = cvt_shl(c, b, const_u32(c, 1), ks);
    uint32_t subv = sel_u32(c, b, ge_sub, sub, const_u32(c, 0));
    return sel_u32(c, b, ge_norm, normal, subv);
}

// 2^(byte - 127) for a ue8m0 scale-factor byte, or NaN for 0xff.
// `divide` gives the reciprocal 2^(127 - byte) used by down-conversions.
static uint32_t cvt_scale_pow2(ctx_t *c, spv_blob_t *b, uint32_t byte,
                               int divide) {
    uint32_t is_nan = do_cmp(c, b, PCM_EQ, PTY_U32, 0, byte,
                             const_u32(c, 255), 0);
    uint32_t k = divide
                     ? cvt_sub(c, b, const_u32(c, 127), byte)
                     : cvt_sub(c, b, byte, const_u32(c, 127));
    uint32_t bits = cvt_pow2_f32(c, b, k);
    return sel_u32(c, b, is_nan, const_u32(c, 0x7fc00000u), bits);
}

// f16 pattern -> f32 value.  Unlike the shared half-precision ALU
// helper this keeps subnormals: e5m2 subnormals are f16 subnormals.
static uint32_t mf_f16_to_f32(ctx_t *c, spv_blob_t *b, uint32_t h,
                              int line) {
    uint32_t s = cvt_shr(c, b, h, const_u32(c, 15));
    uint32_t e = cvt_and(c, b, cvt_shr(c, b, h, const_u32(c, 10)),
                         const_u32(c, 0x1fu));
    uint32_t m = cvt_and(c, b, h, const_u32(c, 0x3ffu));
    uint32_t s31 = cvt_shl(c, b, s, const_u32(c, 31));
    uint32_t normal = cvt_or(
        c, b, s31,
        cvt_or(c, b, cvt_shl(c, b, cvt_add(c, b, e, const_u32(c, 112)),
                             const_u32(c, 23)),
               cvt_shl(c, b, m, const_u32(c, 13))));
    uint32_t mq = cvt_or(c, b, cvt_shl(c, b, m, const_u32(c, 13)),
                         const_u32(c, 0x400000u));
    uint32_t mnez = do_cmp(c, b, PCM_NE, PTY_U32, 0, m,
                           const_u32(c, 0), 0);
    uint32_t o_ni = cvt_or(
        c, b, cvt_or(c, b, s31, const_u32(c, 0x7f800000u)),
        sel_u32(c, b, mnez, mq, const_u32(c, 0)));
    uint32_t k = cvt_sub(c, b, const_u32(c, 31), clz32(c, b, m, line));
    uint32_t expn = cvt_add(c, b, k, const_u32(c, 103));
    uint32_t shbits = cvt_sub(c, b, const_u32(c, 23), k);
    uint32_t emant = cvt_and(c, b, cvt_shl(c, b, m, shbits),
                             const_u32(c, 0x7fffffu));
    uint32_t sub = cvt_or(
        c, b, cvt_or(c, b, s31, cvt_shl(c, b, expn, const_u32(c, 23))),
        emant);
    uint32_t small = sel_u32(c, b, mnez, sub, s31);
    uint32_t bits = sel_u32(
        c, b,
        do_cmp(c, b, PCM_EQ, PTY_U32, 0, e, const_u32(c, 0), 0), small,
        normal);
    bits = sel_u32(
        c, b,
        do_cmp(c, b, PCM_EQ, PTY_U32, 0, e, const_u32(c, 31), 0), o_ni,
        bits);
    return cvt_bitcast_f32(c, b, bits);
}

// bf16 pattern -> f32 value, keeping bf16 subnormals (they are f32
// subnormals with mantissa m << 16).
static uint32_t mf_bf16_to_f32(ctx_t *c, spv_blob_t *b, uint32_t h,
                               int line) {
    (void)line;
    uint32_t s = cvt_shr(c, b, h, const_u32(c, 15));
    uint32_t e = cvt_and(c, b, cvt_shr(c, b, h, const_u32(c, 7)),
                         const_u32(c, 0xffu));
    uint32_t m = cvt_and(c, b, h, const_u32(c, 0x7fu));
    uint32_t s31 = cvt_shl(c, b, s, const_u32(c, 31));
    uint32_t msh = cvt_shl(c, b, m, const_u32(c, 16));
    uint32_t normal = cvt_or(
        c, b, cvt_or(c, b, s31,
                     cvt_shl(c, b, e, const_u32(c, 23))),
        msh);
    uint32_t mnez = do_cmp(c, b, PCM_NE, PTY_U32, 0, m,
                           const_u32(c, 0), 0);
    uint32_t sub = cvt_or(c, b, s31, msh);
    uint32_t small = sel_u32(c, b, mnez, sub, s31);
    uint32_t mq = cvt_or(c, b, msh, const_u32(c, 0x400000u));
    uint32_t o_ni = cvt_or(
        c, b, cvt_or(c, b, s31, const_u32(c, 0x7f800000u)),
        sel_u32(c, b, mnez, mq, const_u32(c, 0)));
    uint32_t bits = sel_u32(
        c, b,
        do_cmp(c, b, PCM_EQ, PTY_U32, 0, e, const_u32(c, 0), 0), small,
        normal);
    bits = sel_u32(
        c, b,
        do_cmp(c, b, PCM_EQ, PTY_U32, 0, e, const_u32(c, 255), 0), o_ni,
        bits);
    return cvt_bitcast_f32(c, b, bits);
}

// Round `sig >> sh` to nearest-even, for sh >= 1.  sh >= 25 yields 0
// (sig < 2^24, so the value is below half a unit at sh = 25).
static uint32_t cvt_rne_shift(ctx_t *c, spv_blob_t *b, uint32_t sig,
                              uint32_t sh) {
    uint32_t over = do_cmp(c, b, PCM_GE, PTY_U32, 0, sh,
                           const_u32(c, 25), 0);
    uint32_t amt = sel_u32(c, b, over, const_u32(c, 25),
                           cvt_and(c, b, sh, const_u32(c, 31)));
    uint32_t q = cvt_shr(c, b, sig, amt);
    uint32_t mask = cvt_sub(c, b,
                            cvt_shl(c, b, const_u32(c, 1), amt),
                            const_u32(c, 1));
    uint32_t rem = cvt_and(c, b, sig, mask);
    uint32_t half = cvt_shl(c, b, const_u32(c, 1),
                            cvt_sub(c, b, amt, const_u32(c, 1)));
    uint32_t gt = do_cmp(c, b, PCM_GT, PTY_U32, 0, rem, half, 0);
    uint32_t eq = do_cmp(c, b, PCM_EQ, PTY_U32, 0, rem, half, 0);
    uint32_t odd = do_cmp(c, b, PCM_NE, PTY_U32, 0,
                          cvt_and(c, b, q, const_u32(c, 1)),
                          const_u32(c, 0), 0);
    uint32_t oddu = sel_u32(c, b, odd, const_u32(c, 1),
                            const_u32(c, 0));
    uint32_t tie = sel_u32(c, b, eq, oddu, const_u32(c, 0));
    uint32_t up = sel_u32(c, b, gt, const_u32(c, 1), tie);
    uint32_t q2 = cvt_add(c, b, q, up);
    return sel_u32(c, b, over, const_u32(c, 0), q2);
}

static int cvt_mf_params(int fmt, int *E, int *M, int *bias) {
    switch (fmt) {
    case CVTMF_E2M1X2: *E = 2; *M = 1; *bias = 1; return 0;
    case CVTMF_E2M3X2: *E = 2; *M = 3; *bias = 1; return 0;
    case CVTMF_E3M2X2: *E = 3; *M = 2; *bias = 3; return 0;
    }
    return -1;
}

// f32 (bit pattern) -> unsigned micro-float magnitude code, RNE with
// satfinite saturation.  relu clamps negative results to zero.
static uint32_t cvt_mf_from_f32(ctx_t *c, spv_blob_t *b, uint32_t f,
                                int E, int M, int bias, int relu) {
    int maxf = (1 << E) - 1;
    int maxmant = (1 << M) - 1;
    uint32_t maxcode = const_u32(c,
                                 ((uint32_t)maxf << M) | (uint32_t)maxmant);
    uint32_t w = cvt_bitcast_u32(c, b, f);
    uint32_t s = cvt_shr(c, b, w, const_u32(c, 31));
    uint32_t e = cvt_and(c, b, cvt_shr(c, b, w, const_u32(c, 23)),
                         const_u32(c, 0xff));
    uint32_t m = cvt_and(c, b, w, const_u32(c, 0x7fffff));
    uint32_t sig = cvt_or(c, b, m, const_u32(c, 0x800000));
    uint32_t te = cvt_add(c, b, e, const_u32(c, (uint32_t)(bias - 127)));

    // Normal path.
    int sh1 = 23 - M;
    uint32_t qn = cvt_shr(c, b, sig, const_u32(c, (uint32_t)sh1));
    uint32_t maskn = const_u32(c, ((uint32_t)1 << sh1) - 1);
    uint32_t remn = cvt_and(c, b, sig, maskn);
    uint32_t halfn = const_u32(c, (uint32_t)1 << (sh1 - 1));
    uint32_t gtn = do_cmp(c, b, PCM_GT, PTY_U32, 0, remn, halfn, 0);
    uint32_t eqn = do_cmp(c, b, PCM_EQ, PTY_U32, 0, remn, halfn, 0);
    uint32_t oddn = do_cmp(c, b, PCM_NE, PTY_U32, 0,
                           cvt_and(c, b, qn, const_u32(c, 1)),
                           const_u32(c, 0), 0);
    uint32_t oddnu = sel_u32(c, b, oddn, const_u32(c, 1),
                             const_u32(c, 0));
    uint32_t tienn = sel_u32(c, b, eqn, oddnu, const_u32(c, 0));
    uint32_t upn = sel_u32(c, b, gtn, const_u32(c, 1), tienn);
    uint32_t qn2 = cvt_add(c, b, qn, upn);
    uint32_t carry = do_cmp(c, b, PCM_EQ, PTY_U32, 0, qn2,
                            const_u32(c, (uint32_t)1 << (M + 1)), 0);
    uint32_t qn3 = sel_u32(c, b, carry,
                           cvt_shr(c, b, qn2, const_u32(c, 1)), qn2);
    uint32_t te2 = sel_u32(c, b, carry, cvt_add(c, b, te, const_u32(c, 1)),
                           te);
    uint32_t code_n = cvt_or(c, b,
                             cvt_shl(c, b, te2, const_u32(c, (uint32_t)M)),
                             cvt_and(c, b, qn3,
                                     const_u32(c, (uint32_t)maxmant)));

    // Subnormal path: shift = 151 - (e + bias + M).
    uint32_t sh = cvt_sub(c, b, const_u32(c, 151),
                          cvt_add(c, b, e,
                                  const_u32(c, (uint32_t)(bias + M))));
    uint32_t qs = cvt_rne_shift(c, b, sig, sh);
    // Rounding the largest subnormal up promotes it to the smallest
    // normal (exponent field 1, zero mantissa).
    uint32_t qs_c = sel_u32(
        c, b,
        do_cmp(c, b, PCM_GT, PTY_U32, 0, qs,
               const_u32(c, (uint32_t)maxmant), 0),
        const_u32(c, (uint32_t)(1 << M)), qs);

    uint32_t zero_e = do_cmp(c, b, PCM_EQ, PTY_U32, 0, e,
                             const_u32(c, 0), 0);
    uint32_t spec_e = do_cmp(c, b, PCM_EQ, PTY_U32, 0, e,
                             const_u32(c, 255), 0);
    uint32_t mnez = do_cmp(c, b, PCM_NE, PTY_U32, 0, m,
                           const_u32(c, 0), 0);
    uint32_t is_nan = cvt_sel_bool(c, b, spec_e, mnez,
                                   cvt_bool_const(c, 0));
    // Signed compares: te wraps for inputs below the smallest normal.
    // Every exponent field is finite in these formats, so only an
    // exponent field past the top one overflows.
    uint32_t over = do_cmp(c, b, PCM_GT, PTY_U32, 1, te2,
                           const_u32(c, (uint32_t)maxf), 0);
    uint32_t normal = do_cmp(c, b, PCM_GE, PTY_U32, 1, te,
                             const_u32(c, 1), 0);
    uint32_t val = sel_u32(c, b, spec_e, maxcode,
                           sel_u32(c, b, zero_e, const_u32(c, 0),
                                   sel_u32(c, b, over, maxcode,
                                           sel_u32(c, b, normal, code_n,
                                                   qs_c))));
    uint32_t sign = sel_u32(c, b, is_nan, const_u32(c, 0), s);
    if (relu) {
        uint32_t neg = do_cmp(c, b, PCM_NE, PTY_U32, 0, sign,
                              const_u32(c, 0), 0);
        val = sel_u32(c, b, neg, const_u32(c, 0), val);
        sign = sel_u32(c, b, neg, const_u32(c, 0), sign);
    }
    return cvt_or(c, b, cvt_shl(c, b, sign,
                                const_u32(c, (uint32_t)(E + M))),
                  val);
}

// Unsigned micro-float code -> f32.
static uint32_t cvt_mf_to_f32(ctx_t *c, spv_blob_t *b, uint32_t code,
                              int E, int M, int bias) {
    int maxf = (1 << E) - 1;
    int maxmant = (1 << M) - 1;
    uint32_t s = cvt_shr(c, b, code, const_u32(c, (uint32_t)(E + M)));
    uint32_t ef = cvt_and(c, b,
                          cvt_shr(c, b, code, const_u32(c, (uint32_t)M)),
                          const_u32(c, (uint32_t)maxf));
    uint32_t mf = cvt_and(c, b, code, const_u32(c, (uint32_t)maxmant));
    uint32_t sig = sel_u32(c, b,
                           do_cmp(c, b, PCM_EQ, PTY_U32, 0, ef,
                                  const_u32(c, 0), 0),
                           mf,
                           cvt_or(c, b, mf,
                                  const_u32(c, (uint32_t)(1 << M))));
    uint32_t f_sig = nid(c);
    put(c, b, (4u << 16) | S_ConvertUToF);
    put(c, b, c->t_f32);
    put(c, b, f_sig);
    put(c, b, sig);
    uint32_t k = sel_u32(c, b,
                         do_cmp(c, b, PCM_EQ, PTY_U32, 0, ef,
                                const_u32(c, 0), 0),
                         const_u32(c, (uint32_t)(1 - bias - M)),
                         cvt_sub(c, b, ef,
                                 const_u32(c, (uint32_t)(bias + M))));
    uint32_t p = cvt_bitcast_f32(c, b, cvt_pow2_f32(c, b, k));
    uint32_t val = nid(c);
    emit_arith(c, b, S_FMul, c->t_f32, val, f_sig, p);
    uint32_t neg = do_cmp(c, b, PCM_NE, PTY_U32, 0, s,
                          const_u32(c, 0), 0);
    uint32_t nv = nid(c);
    put(c, b, (4u << 16) | S_FNegate);
    put(c, b, c->t_f32);
    put(c, b, nv);
    put(c, b, val);
    return cvt_sel_f32(c, b, neg, nv, val);
}

// f32 -> ue8m0 code.  round_up selects ceil(log2) (rp) else floor (rz).
static uint32_t cvt_ue8m0_from_f32(ctx_t *c, spv_blob_t *b, uint32_t f,
                                   int round_up) {
    uint32_t w = cvt_bitcast_u32(c, b, f);
    uint32_t e = cvt_and(c, b, cvt_shr(c, b, w, const_u32(c, 23)),
                         const_u32(c, 0xff));
    uint32_t m = cvt_and(c, b, w, const_u32(c, 0x7fffff));
    uint32_t mnez = do_cmp(c, b, PCM_NE, PTY_U32, 0, m,
                           const_u32(c, 0), 0);
    uint32_t e_rup = cvt_add(c, b, e,
                             sel_u32(c, b, mnez, const_u32(c, 1),
                                     const_u32(c, 0)));
    uint32_t vnorm = round_up ? e_rup : e;
    // Inputs below the smallest normal f32 are m * 2^-149.  Their
    // biased exponent is p - 22 (p = highest set bit of m); rp rounds
    // non-powers of two up to p - 21.  Anything below code 0 clamps to
    // 0 (ue8m0 has no smaller exponent).
    uint32_t p = cvt_sub(c, b, const_u32(c, 31), clz32(c, b, m, 0));
    uint32_t ispow = do_cmp(
        c, b, PCM_EQ, PTY_U32, 0,
        cvt_and(c, b, m, cvt_sub(c, b, m, const_u32(c, 1))),
        const_u32(c, 0), 0);
    uint32_t bias = round_up
                        ? sel_u32(c, b, ispow, const_u32(c, 22),
                                  const_u32(c, 21))
                        : const_u32(c, 22);
    uint32_t code_sub = cvt_sub(c, b, p, bias);
    code_sub = sel_u32(
        c, b, do_cmp(c, b, PCM_LT, PTY_U32, 1, p, bias, 0),
        const_u32(c, 0), code_sub);
    uint32_t v = sel_u32(c, b, mnez, code_sub, const_u32(c, 0));
    v = sel_u32(c, b, do_cmp(c, b, PCM_EQ, PTY_U32, 0, e,
                             const_u32(c, 0), 0),
                v, vnorm);
    // Infinity -> MAX_NORM (0xfe); NaN -> 0xff.  The destination has
    // no sign, so finite inputs use their magnitude.
    uint32_t is_spec = do_cmp(c, b, PCM_EQ, PTY_U32, 0, e,
                              const_u32(c, 255), 0);
    uint32_t is_inf = do_cmp(c, b, PCM_EQ, PTY_U32, 0, m,
                             const_u32(c, 0), 0);
    uint32_t spec = sel_u32(c, b, is_inf, const_u32(c, 0xfe),
                            const_u32(c, 0xff));
    // Clamp finite values before applying the NaN/infinity overrides
    // (0xff is the ue8m0 NaN, not an out-of-range exponent).
    v = sel_u32(c, b,
                do_cmp(c, b, PCM_GT, PTY_U32, 0, v, const_u32(c, 254),
                       0),
                const_u32(c, 254), v);
    return sel_u32(c, b, is_spec, spec, v);
}

static uint32_t cvt_ue8m0_to_f32(ctx_t *c, spv_blob_t *b, uint32_t code) {
    uint32_t e = cvt_and(c, b, code, const_u32(c, 0xff));
    uint32_t k = cvt_sub(c, b, e, const_u32(c, 127));
    uint32_t bits = cvt_pow2_f32(c, b, k);
    return sel_u32(c, b,
                   do_cmp(c, b, PCM_EQ, PTY_U32, 0, e, const_u32(c, 255),
                          0),
                   const_u32(c, 0x7fc00000u), bits);
}
// f32 -> s2f6 byte (s8 fixed point, value = s8 * 2^-6).
static uint32_t cvt_s2f6_from_f32(ctx_t *c, spv_blob_t *b, uint32_t f,
                                  int relu) {
    if (relu) {
        uint32_t neg = do_cmp(c, b, PCM_LT, PTY_F32, 0, f,
                              const_f32(c, 0.0f), 0);
        f = cvt_sel_f32(c, b, neg, const_f32(c, 0.0f), f);
    }
    // NaN and infinity saturate to MAX_NORM with the sign preserved;
    // finite values are clamped before the float->int conversion (which
    // is otherwise undefined outside the destination range).
    uint32_t w = cvt_bitcast_u32(c, b, f);
    uint32_t s = cvt_shr(c, b, w, const_u32(c, 31));
    uint32_t e = cvt_and(c, b, cvt_shr(c, b, w, const_u32(c, 23)),
                         const_u32(c, 0xff));
    uint32_t m = cvt_and(c, b, w, const_u32(c, 0x7fffff));
    uint32_t spec = do_cmp(c, b, PCM_EQ, PTY_U32, 0, e,
                           const_u32(c, 255), 0);
    uint32_t nan = do_cmp(c, b, PCM_NE, PTY_U32, 0, m,
                          const_u32(c, 0), 0);
    uint32_t specv = sel_u32(
        c, b, nan, const_u32(c, 127),
        sel_u32(c, b,
                do_cmp(c, b, PCM_NE, PTY_U32, 0, s, const_u32(c, 0), 0),
                const_u32(c, (uint32_t)-127), const_u32(c, 127)));
    uint32_t scaled = nid(c);
    emit_arith(c, b, S_FMul, c->t_f32, scaled, f, const_f32(c, 0x42800000u));
    scaled = cvt_sel_f32(c, b,
                         do_cmp(c, b, PCM_GT, PTY_F32, 0, scaled,
                                const_f32(c, 0x42FE0000u), 0),
                         const_f32(c, 0x42FE0000u), scaled);
    scaled = cvt_sel_f32(c, b,
                         do_cmp(c, b, PCM_LT, PTY_F32, 0, scaled,
                                const_f32(c, 0xc2fe0000u), 0),
                         const_f32(c, 0xc2fe0000u), scaled);
    uint32_t r = nid(c);
    put(c, b, (6u << 16) | S_ExtInst);
    put(c, b, c->t_f32);
    put(c, b, r);
    put(c, b, c->glsl450);
    put(c, b, 2); // RoundEven
    put(c, b, scaled);
    uint32_t si = nid(c);
    put(c, b, (4u << 16) | S_ConvertFToS);
    put(c, b, c->t_u32);
    put(c, b, si);
    put(c, b, r);
    si = sel_u32(c, b, spec, specv, si);
    return cvt_and(c, b, si, const_u32(c, 0xff));
}

static uint32_t cvt_s2f6_to_f32(ctx_t *c, spv_blob_t *b, uint32_t code) {
    uint32_t sh = cvt_shl(c, b, code, const_u32(c, 24));
    uint32_t sx = nid(c);
    emit_arith(c, b, S_ShrA, c->t_u32, sx, sh, const_u32(c, 24));
    uint32_t f = nid(c);
    put(c, b, (4u << 16) | S_ConvertSToF);
    put(c, b, c->t_f32);
    put(c, b, f);
    put(c, b, sx);
    uint32_t out = nid(c);
    emit_arith(c, b, S_FMul, c->t_f32, out, f, const_f32(c, 0x3c800000u));
    return out;
}

// Two f32 from an f16x2/bf16x2 pattern (high half first).
static void cvt_unpack_16(ctx_t *c, spv_blob_t *b, uint32_t packed,
                          int bf16, uint32_t *hi, uint32_t *lo) {
    uint32_t h = cvt_and(c, b, cvt_shr(c, b, packed, const_u32(c, 16)),
                         const_u32(c, 0xffff));
    uint32_t l = cvt_and(c, b, packed, const_u32(c, 0xffff));
    *hi = bf16 ? mf_bf16_to_f32(c, b, h, 0)
               : mf_f16_to_f32(c, b, h, 0);
    *lo = bf16 ? mf_bf16_to_f32(c, b, l, 0)
               : mf_f16_to_f32(c, b, l, 0);
}
static int cvt_is_x4(int d) {
    return d == CVTMF_E4M3X4 || d == CVTMF_E5M2X4 || d == CVTMF_E2M1X4 ||
           d == CVTMF_E2M3X4 || d == CVTMF_E3M2X4;
}

// Pack one f32 into the destination format code (fp8 uses the existing
// byte converter; f4/f6 use the integer routine).
static uint32_t cvt_one_code(ctx_t *c, spv_blob_t *b, int d, uint32_t f,
                             int relu) {
    if (d == CVTMF_E4M3X2 || d == CVTMF_E4M3X4 ||
        d == CVTMF_E5M2X2 || d == CVTMF_E5M2X4) {
        if (relu)
            f = cvt_relu_f32(c, b, f, 0);
        return fp8byte_from_f32(c, b, f, d == CVTMF_E4M3X2 ||
                                            d == CVTMF_E4M3X4, 0);
    }
    int E, M, bias;
    if (cvt_mf_params(d, &E, &M, &bias) != 0)
        return 0;
    return cvt_mf_from_f32(c, b, f, E, M, bias, relu);
}

int emit_cvt_micro(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    int d = in->cvt_dst;

    if (d == CVTMF_F16X2 || d == CVTMF_BF16X2) {
        // Unpack a packed micro pair into f16x2/bf16x2.  The result is
        // a 32-bit pattern, so reject destinations of the wrong width
        // (notably .f32) instead of leaving a mis-typed SSA value.
        reg_t *fdr = find_reg(c, in->dst);
        ptx_ty_t exp_ty = (d == CVTMF_F16X2) ? PTY_F16X2 : PTY_BF16X2;
        if (!fdr || (fdr->decl_ty != PTY_U32 && fdr->decl_ty != exp_ty)) {
            failf(c, line, d == CVTMF_F16X2
                              ? "micro-float unpack destination must be "
                                ".b32/.u32/.s32 or .f16x2"
                              : "micro-float unpack destination must be "
                                ".b32/.u32/.s32 or .bf16x2");
            return -1;
        }
        // e2m1x2 is a .b8 value in PTX; accept a .b16 holder too.
        ptx_ty_t sw = PTY_U16;
        if (in->src[0].kind == OPD_REG) {
            reg_t *sr = find_reg(c, in->src[0].reg);
            if (sr && sr->decl_ty == PTY_U8)
                sw = PTY_U8;
        }
        val_t a = resolve(c, b, &in->src[0], sw, line);
        if (c->failed)
            return -1;
        uint32_t hi = 0, lo = 0;
        int sfmt = in->cvt_src;
        if (sfmt == CVTMF_E2M1X2) {
            hi = cvt_and(c, b, cvt_shr(c, b, a.id, const_u32(c, 4)),
                         const_u32(c, 0xf));
            lo = cvt_and(c, b, a.id, const_u32(c, 0xf));
        } else {
            hi = cvt_and(c, b, cvt_shr(c, b, a.id, const_u32(c, 8)),
                         const_u32(c, 0xff));
            lo = cvt_and(c, b, a.id, const_u32(c, 0xff));
        }
        uint32_t fhi, flo;
        if (sfmt == CVTMF_E4M3X2 || sfmt == CVTMF_E5M2X2) {
            int e4 = sfmt == CVTMF_E4M3X2;
            uint32_t hh = f16_from_fp8byte(c, b, hi, e4, line);
            uint32_t ll = f16_from_fp8byte(c, b, lo, e4, line);
            fhi = mf_f16_to_f32(c, b, hh, line);
            flo = mf_f16_to_f32(c, b, ll, line);
        } else if (sfmt == CVTMF_UE8M0X2) {
            fhi = cvt_ue8m0_to_f32(c, b, hi);
            flo = cvt_ue8m0_to_f32(c, b, lo);
        } else if (sfmt == CVTMF_S2F6X2) {
            fhi = cvt_s2f6_to_f32(c, b, hi);
            flo = cvt_s2f6_to_f32(c, b, lo);
        } else {
            int E, M, bias;
            if (cvt_mf_params(sfmt, &E, &M, &bias) != 0) {
                failf(c, line, "unsupported micro-float source");
                return -1;
            }
            fhi = cvt_mf_to_f32(c, b, hi, E, M, bias);
            flo = cvt_mf_to_f32(c, b, lo, E, M, bias);
        }
        if (in->cvt_scaled && in->nsrc > 1) {
            val_t sf = resolve(c, b, &in->src[1], PTY_U16, line);
            uint32_t sfh = cvt_and(c, b,
                                   cvt_shr(c, b, sf.id, const_u32(c, 8)),
                                   const_u32(c, 0xff));
            uint32_t sfl = cvt_and(c, b, sf.id, const_u32(c, 0xff));
            uint32_t shi = cvt_bitcast_f32(c, b,
                                           cvt_scale_pow2(c, b, sfh, 0));
            uint32_t slo = cvt_bitcast_f32(c, b,
                                           cvt_scale_pow2(c, b, sfl, 0));
            uint32_t t = nid(c);
            emit_arith(c, b, S_FMul, c->t_f32, t, fhi, shi);
            fhi = t;
            t = nid(c);
            emit_arith(c, b, S_FMul, c->t_f32, t, flo, slo);
            flo = t;
        }
        if (in->cvt_relu) {
            fhi = cvt_relu_f32(c, b, fhi, line);
            flo = cvt_relu_f32(c, b, flo, line);
        }
        uint32_t hbits, lbits;
        if (d == CVTMF_F16X2) {
            hbits = f32_to_f16bits(c, b, fhi, line);
            lbits = f32_to_f16bits(c, b, flo, line);
        } else {
            hbits = f32_to_bf16bits(c, b, fhi, line);
            lbits = f32_to_bf16bits(c, b, flo, line);
        }
        def_val(c, in->dst, PTY_U32,
                cvt_or(c, b, cvt_shl(c, b, hbits, const_u32(c, 16)),
                       cvt_and(c, b, lbits, const_u32(c, 0xffff))));
        return c->failed ? -1 : 0;
    }

    if (d == CVTMF_UE8M0X2) {
        uint32_t ch, cl;
        if (in->cvt_src == CVTMF_SRC_F32) {
            val_t a = resolve(c, b, &in->src[0], PTY_F32, line);
            val_t bb = resolve(c, b, &in->src[1], PTY_F32, line);
            ch = cvt_ue8m0_from_f32(c, b, a.id, in->cvt_frnd == 2);
            cl = cvt_ue8m0_from_f32(c, b, bb.id, in->cvt_frnd == 2);
        } else if (in->cvt_src == CVTMF_SRC_BF16X2) {
            val_t a = resolve(c, b, &in->src[0], PTY_U32, line);
            uint32_t fh, fl;
            cvt_unpack_16(c, b, a.id, 1, &fh, &fl);
            ch = cvt_ue8m0_from_f32(c, b, fh, in->cvt_frnd == 2);
            cl = cvt_ue8m0_from_f32(c, b, fl, in->cvt_frnd == 2);
        } else {
            failf(c, line, "cvt to ue8m0x2 needs .f32 or .bf16x2");
            return -1;
        }
        def_val(c, in->dst, PTY_U16,
                cvt_or(c, b, cvt_shl(c, b, cvt_and(c, b, ch,
                                                   const_u32(c, 0xff)),
                                     const_u32(c, 8)),
                       cvt_and(c, b, cl, const_u32(c, 0xff))));
        return c->failed ? -1 : 0;
    }

    if (d == CVTMF_S2F6X2) {
        uint32_t fh, fl;
        if (in->cvt_src == CVTMF_SRC_F32) {
            val_t a = resolve(c, b, &in->src[0], PTY_F32, line);
            val_t bb = resolve(c, b, &in->src[1], PTY_F32, line);
            fh = a.id;
            fl = bb.id;
        } else if (in->cvt_src == CVTMF_SRC_BF16X2) {
            val_t a = resolve(c, b, &in->src[0], PTY_U32, line);
            cvt_unpack_16(c, b, a.id, 1, &fh, &fl);
        } else {
            failf(c, line, "cvt to s2f6x2 needs .f32 or .bf16x2");
            return -1;
        }
        if (in->cvt_scaled && in->nsrc > (in->cvt_src == CVTMF_SRC_F32 ? 2
                                                                       : 1)) {
            val_t sf = resolve(c, b,
                               &in->src[in->cvt_src == CVTMF_SRC_F32 ? 2
                                                                     : 1],
                               PTY_U16, line);
            uint32_t sfh = cvt_and(c, b,
                                   cvt_shr(c, b, sf.id, const_u32(c, 8)),
                                   const_u32(c, 0xff));
            uint32_t sfl = cvt_and(c, b, sf.id, const_u32(c, 0xff));
            // Divide by 2^(E-127) == multiply by 2^(127-E).
            uint32_t ph = cvt_bitcast_f32(c, b,
                                          cvt_scale_pow2(c, b, sfh, 1));
            uint32_t pl = cvt_bitcast_f32(c, b,
                                          cvt_scale_pow2(c, b, sfl, 1));
            uint32_t t = nid(c);
            emit_arith(c, b, S_FMul, c->t_f32, t, fh, ph);
            fh = t;
            t = nid(c);
            emit_arith(c, b, S_FMul, c->t_f32, t, fl, pl);
            fl = t;
        }
        uint32_t ch = cvt_s2f6_from_f32(c, b, fh, in->cvt_relu);
        uint32_t cl = cvt_s2f6_from_f32(c, b, fl, in->cvt_relu);
        def_val(c, in->dst, PTY_U16,
                cvt_or(c, b, cvt_shl(c, b, cvt_and(c, b, ch,
                                                   const_u32(c, 0xff)),
                                     const_u32(c, 8)),
                       cvt_and(c, b, cl, const_u32(c, 0xff))));
        return c->failed ? -1 : 0;
    }

    if (cvt_is_x4(d)) {
        failf(c, line, "cvt x4 micro-float forms are not implemented "
                       "(.rs stochastic rounding)");
        return -1;
    }

    // Pack two inputs into a micro x2 format.
    uint32_t fh, fl;
    if (in->cvt_src == CVTMF_SRC_F32) {
        val_t a = resolve(c, b, &in->src[0], PTY_F32, line);
        val_t bb = resolve(c, b, &in->src[1], PTY_F32, line);
        fh = a.id;
        fl = bb.id;
    } else if (in->cvt_src == CVTMF_SRC_F16X2 ||
               in->cvt_src == CVTMF_SRC_BF16X2) {
        val_t a = resolve(c, b, &in->src[0], PTY_U32, line);
        cvt_unpack_16(c, b, a.id, in->cvt_src == CVTMF_SRC_BF16X2, &fh,
                      &fl);
    } else {
        failf(c, line, "unsupported cvt source for micro destination");
        return -1;
    }
    uint32_t ch = cvt_one_code(c, b, d, fh, in->cvt_relu);
    uint32_t cl = cvt_one_code(c, b, d, fl, in->cvt_relu);
    if (c->failed)
        return -1;
    if (d == CVTMF_E2M1X2) {
        def_val(c, in->dst, PTY_U8,
                cvt_or(c, b,
                       cvt_shl(c, b, cvt_and(c, b, ch, const_u32(c, 0xf)),
                               const_u32(c, 4)),
                       cvt_and(c, b, cl, const_u32(c, 0xf))));
    } else {
        // e2m3/e3m2: one byte per element, a in the high byte.
        def_val(c, in->dst, PTY_U16,
                cvt_or(c, b,
                       cvt_shl(c, b,
                               cvt_and(c, b, ch, const_u32(c, 0xff)),
                               const_u32(c, 8)),
                       cvt_and(c, b, cl, const_u32(c, 0xff))));
    }
    return c->failed ? -1 : 0;
}
