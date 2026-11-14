// Arithmetic, conversion, bit-manipulation, comparison and select
// opcodes (POP_ADD .. POP_SET), plus the directed-rounding helpers.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "spirv_internal.h"
#include "instructions/logic_shift/logic_shift.h"
#include "instructions/conversion/conversion.h"
#include "instructions/memory/memory.h"
#include "instructions/arith/arith.h"
#include "instructions/arith/integer/integer.h"
#include "instructions/data_movement/prmt.h"
#include "instructions/video/video_ops.h"
#include "instructions/compare/compare.h"
#include "instructions/arith/float/float.h"
#include <stdio.h>
#include <stdlib.h>

// f32 saturate to [0.0, 1.0]; NaN becomes +0.0 (PTX .sat semantics).
uint32_t relu_f32(ctx_t *c, spv_blob_t *b, uint32_t x, int line) {
    uint32_t nan = nid(c);
    put(c, b, (5u << 16) | S_FUnordNotEqual);
    put(c, b, c->t_bool);
    put(c, b, nan);
    put(c, b, x);
    put(c, b, x);
    uint32_t neg = do_cmp(c, b, PCM_LT, PTY_F32, 0, x, const_f32(c, 0),
                          line);
    if (c->failed)
        return 0;
    uint32_t r = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, c->t_f32);
    put(c, b, r);
    put(c, b, neg);
    put(c, b, const_f32(c, 0));
    put(c, b, x);
    uint32_t out = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, c->t_f32);
    put(c, b, out);
    put(c, b, nan);
    put(c, b, const_f32(c, 0x7fc00000u));
    put(c, b, r);
    return out;
}

uint32_t sat_f32(ctx_t *c, spv_blob_t *b, uint32_t x, int line) {
    uint32_t nan = nid(c);
    put(c, b, (5u << 16) | S_FUnordNotEqual);
    put(c, b, c->t_bool);
    put(c, b, nan);
    put(c, b, x);
    put(c, b, x);
    uint32_t gt = do_cmp(c, b, PCM_GT, PTY_F32, 0, x,
                         const_f32(c, 0x3f800000u), line);
    uint32_t lt = do_cmp(c, b, PCM_LT, PTY_F32, 0, x,
                         const_f32(c, 0), line);
    if (c->failed)
        return 0;
    uint32_t r = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, c->t_f32);
    put(c, b, r);
    put(c, b, gt);
    put(c, b, const_f32(c, 0x3f800000u));
    put(c, b, x);
    uint32_t r2 = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, c->t_f32);
    put(c, b, r2);
    put(c, b, lt);
    put(c, b, const_f32(c, 0));
    put(c, b, r);
    uint32_t r3 = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, c->t_f32);
    put(c, b, r3);
    put(c, b, nan);
    put(c, b, const_f32(c, 0));
    put(c, b, r2);
    return r3;
}

// min/max.NaN for one f16/bf16 lane: canonical quiet NaN if either
// input is NaN, otherwise the ordered result over the original patterns.
uint32_t half_minnan_lane(ctx_t *c, spv_blob_t *b, int is_bf16,
                                 int is_max, uint32_t av, uint32_t bv,
                                 int line) {
    uint32_t fa = is_bf16 ? bf16_to_f32val(c, b, av, line)
                          : f16_to_f32val(c, b, av, line);
    uint32_t fb = is_bf16 ? bf16_to_f32val(c, b, bv, line)
                          : f16_to_f32val(c, b, bv, line);
    if (c->failed)
        return 0;
    uint32_t na = nid(c), nb = nid(c), nan = nid(c);
    put(c, b, (5u << 16) | S_FUnordNotEqual);
    put(c, b, c->t_bool);
    put(c, b, na);
    put(c, b, fa);
    put(c, b, fa);
    put(c, b, (5u << 16) | S_FUnordNotEqual);
    put(c, b, c->t_bool);
    put(c, b, nb);
    put(c, b, fb);
    put(c, b, fb);
    put(c, b, (5u << 16) | S_LogicalOr);
    put(c, b, c->t_bool);
    put(c, b, nan);
    put(c, b, na);
    put(c, b, nb);
    uint32_t cond = do_cmp(c, b, is_max ? PCM_GT : PCM_LT, PTY_F32, 0,
                           fa, fb, line);
    if (c->failed)
        return 0;
    uint32_t mm = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, c->t_u32);
    put(c, b, mm);
    put(c, b, cond);
    put(c, b, av);
    put(c, b, bv);
    uint32_t cn = is_bf16 ? const_u32(c, 0x7fc0u) : const_u32(c, 0x7e00u);
    return sel_u32(c, b, nan, cn, mm);
}

// Flush an f16/bf16 bit pattern through f32 so subnormals become signed
// zero (used by .ftz half-precision forms).
uint32_t half_ftz_bits(ctx_t *c, spv_blob_t *b, int is_bf16,
                              uint32_t bits, int line) {
    uint32_t f = is_bf16 ? bf16_to_f32val(c, b, bits, line)
                         : f16_to_f32val(c, b, bits, line);
    if (c->failed)
        return 0;
    f = flush_f32_denorm(c, b, f, line);
    if (c->failed)
        return 0;
    return is_bf16 ? f32_to_bf16bits(c, b, f, line)
                   : f32_to_f16bits(c, b, f, line);
}

// One f16/bf16 arithmetic lane, computed in f32 and narrowed back.
uint32_t half_arith_lane(ctx_t *c, spv_blob_t *b, int is_bf16,
                                uint32_t fop, int ftz, int sat,
                                uint32_t av, uint32_t bv, int line) {
    uint32_t fa = is_bf16 ? bf16_to_f32val(c, b, av, line)
                          : f16_to_f32val(c, b, av, line);
    uint32_t fb = is_bf16 ? bf16_to_f32val(c, b, bv, line)
                          : f16_to_f32val(c, b, bv, line);
    if (c->failed)
        return 0;
    if (ftz) {
        fa = flush_f32_denorm(c, b, fa, line);
        fb = flush_f32_denorm(c, b, fb, line);
        if (c->failed)
            return 0;
    }
    uint32_t r = nid(c);
    emit_arith(c, b, fop, c->t_f32, r, fa, fb);
    if (ftz) {
        r = flush_f32_denorm(c, b, r, line);
        if (c->failed)
            return 0;
    }
    if (sat) {
        r = sat_f32(c, b, r, line);
        if (c->failed)
            return 0;
    }
    return is_bf16 ? f32_to_bf16bits(c, b, r, line)
                   : f32_to_f16bits(c, b, r, line);
}

// 64-bit leading-zero count from the 32-bit primitive.
uint32_t clz64_val(ctx_t *c, spv_blob_t *b, uint32_t v, int line) {
    uint32_t hi = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, hi, v, const_u64(c, 32));
    uint32_t hi32 = nid(c), lo32 = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u32);
    put(c, b, hi32);
    put(c, b, hi);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u32);
    put(c, b, lo32);
    put(c, b, v);
    uint32_t ch = clz32(c, b, hi32, line);
    uint32_t cl = clz32(c, b, lo32, line);
    if (c->failed)
        return 0;
    uint32_t hz = do_cmp(c, b, PCM_EQ, PTY_U32, 0, hi32,
                         const_u32(c, 0), line);
    if (c->failed)
        return 0;
    uint32_t c32 = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u32, c32, cl, const_u32(c, 32));
    return sel_u32(c, b, hz, c32, ch);
}

// 32-bit bit reversal (5 swap stages).
uint32_t brev32_val(ctx_t *c, spv_blob_t *b, uint32_t v) {
    const uint32_t masks[5] = { 0x55555555u, 0x33333333u, 0x0f0f0f0fu,
                                0x00ff00ffu, 0x0000ffffu };
    const uint32_t shifts[5] = { 1, 2, 4, 8, 16 };
    for (int i = 0; i < 5; i++) {
        uint32_t lo = nid(c), hi = nid(c), mv = nid(c), w = nid(c);
        emit_arith(c, b, S_ShrL, c->t_u32, lo, v, const_u32(c, shifts[i]));
        emit_arith(c, b, S_And, c->t_u32, mv, lo, const_u32(c, masks[i]));
        emit_arith(c, b, S_And, c->t_u32, w, v, const_u32(c, masks[i]));
        emit_arith(c, b, S_Shl, c->t_u32, hi, w, const_u32(c, shifts[i]));
        v = nid(c);
        emit_arith(c, b, S_Or, c->t_u32, v, mv, hi);
    }
    return v;
}

// Packed-integer lane helpers: extract lane (optionally sign-extended
// to s32) and merge a lane back into a b32 holder.
uint32_t pack_lane_get(ctx_t *c, spv_blob_t *b, uint32_t v,
                       int lane, int width, int sign) {
    uint32_t sh = (uint32_t)(lane * width);
    uint32_t x = v;
    if (sh) {
        x = nid(c);
        emit_arith(c, b, S_ShrL, c->t_u32, x, v, const_u32(c, sh));
    }
    uint32_t mask = (1u << width) - 1u;
    uint32_t y = nid(c);
    emit_arith(c, b, S_And, c->t_u32, y, x, const_u32(c, mask));
    if (sign) {
        // The compiler holds 32-bit values in unsigned SPIR-V types;
        // several drivers zero the high bit for OpShiftRightArithmetic
        // on those. OpBitFieldSExtract performs the expansion explicitly.
        uint32_t ext = nid(c);
        put(c, b, (6u << 16) | 202u); // OpBitFieldSExtract
        put(c, b, c->t_u32);
        put(c, b, ext);
        put(c, b, y);
        put(c, b, const_u32(c, 0));
        put(c, b, const_u32(c, (uint32_t)width));
        y = ext;
    }
    return y;
}

uint32_t pack_lane_put(ctx_t *c, spv_blob_t *b, uint32_t out,
                              uint32_t lane, int idx, int width,
                              uint32_t mask) {
    uint32_t v = nid(c);
    emit_arith(c, b, S_And, c->t_u32, v, lane, const_u32(c, mask));
    uint32_t sh = (uint32_t)(idx * width);
    if (idx == 0)
        return v;
    if (sh) {
        uint32_t l = nid(c);
        emit_arith(c, b, S_Shl, c->t_u32, l, v, const_u32(c, sh));
        v = l;
    }
    uint32_t r = nid(c);
    emit_arith(c, b, S_Or, c->t_u32, r, out, v);
    return r;
}

// Step the round-to-nearest bit pattern one ulp in the requested
// direction when errpos/errneg says the exact value lies above/below.
// ---- Portable f32 fused multiply-add ---------------------------------
// Exact integer implementation of IEEE-754 fma for rn/rz/rm/rp.  The
// significand product and the addend are aligned by normalized leading
// bit rather than by raw exponent, so cancellation keeps every bit that
// can influence rounding.  Only integer SPIR-V is emitted; the host
// never computes the result.
uint32_t u32_bin(ctx_t *c, spv_blob_t *b, uint32_t op, uint32_t x,
                        uint32_t y) {
    uint32_t id = nid(c);
    emit_arith(c, b, op, c->t_u32, id, x, y);
    return id;
}

uint32_t u64_bin(ctx_t *c, spv_blob_t *b, uint32_t op, uint32_t x,
                        uint32_t y) {
    uint32_t id = nid(c);
    emit_arith(c, b, op, c->t_u64, id, x, y);
    return id;
}

uint32_t bool_bin(ctx_t *c, spv_blob_t *b, uint32_t op, uint32_t x,
                         uint32_t y) {
    uint32_t id = nid(c);
    put(c, b, (5u << 16) | op);
    put(c, b, c->t_bool);
    put(c, b, id);
    put(c, b, x);
    put(c, b, y);
    return id;
}

uint32_t bool_not(ctx_t *c, spv_blob_t *b, uint32_t x) {
    uint32_t id = nid(c);
    put(c, b, (4u << 16) | S_LogicalNot);
    put(c, b, c->t_bool);
    put(c, b, id);
    put(c, b, x);
    return id;
}

uint32_t bool_const(ctx_t *c, int v) {
    uint32_t id = nid(c);
    put(c, &c->types, (3u << 16) | (v ? S_ConstantTrue : S_ConstantFalse));
    put(c, &c->types, c->t_bool);
    put(c, &c->types, id);
    return id;
}

uint32_t zext_u32_u64(ctx_t *c, spv_blob_t *b, uint32_t v) {
    uint32_t id = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u64);
    put(c, b, id);
    put(c, b, v);
    return id;
}

uint32_t trunc_u64_u32(ctx_t *c, spv_blob_t *b, uint32_t v) {
    uint32_t id = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u32);
    put(c, b, id);
    put(c, b, v);
    return id;
}

uint32_t f32_to_bits(ctx_t *c, spv_blob_t *b, uint32_t f) {
    uint32_t id = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_u32);
    put(c, b, id);
    put(c, b, f);
    return id;
}

uint32_t bits_to_f32(ctx_t *c, spv_blob_t *b, uint32_t v) {
    uint32_t id = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_f32);
    put(c, b, id);
    put(c, b, v);
    return id;
}

uint32_t cmp_u32(ctx_t *c, spv_blob_t *b, ptx_cmp_t cmp, int sign,
                        uint32_t x, uint32_t y, int line) {
    return do_cmp(c, b, cmp, PTY_U32, sign, x, y, line);
}

uint32_t arith_cmp_u64(ctx_t *c, spv_blob_t *b, ptx_cmp_t cmp,
                        uint32_t x, uint32_t y, int line) {
    return do_cmp(c, b, cmp, PTY_U64, 0, x, y, line);
}

uint32_t f32_soft_fma(ctx_t *c, spv_blob_t *b, uint32_t fa,
                             uint32_t fb, uint32_t fc, int mode, int line,
                             uint32_t *exact_sign, uint32_t *exact_nz) {
    uint32_t ba = f32_to_bits(c, b, fa);
    uint32_t bb = f32_to_bits(c, b, fb);
    uint32_t bc = f32_to_bits(c, b, fc);
    uint32_t ea = u32_bin(c, b, S_And,
                          u32_bin(c, b, S_ShrL, ba, const_u32(c, 23)),
                          const_u32(c, 0xffu));
    uint32_t eb = u32_bin(c, b, S_And,
                          u32_bin(c, b, S_ShrL, bb, const_u32(c, 23)),
                          const_u32(c, 0xffu));
    uint32_t ec = u32_bin(c, b, S_And,
                          u32_bin(c, b, S_ShrL, bc, const_u32(c, 23)),
                          const_u32(c, 0xffu));
    uint32_t ma = u32_bin(c, b, S_And, ba, const_u32(c, 0x7fffffu));
    uint32_t mb = u32_bin(c, b, S_And, bb, const_u32(c, 0x7fffffu));
    uint32_t mc = u32_bin(c, b, S_And, bc, const_u32(c, 0x7fffffu));
    uint32_t sa = u32_bin(c, b, S_ShrL, ba, const_u32(c, 31));
    uint32_t sb = u32_bin(c, b, S_ShrL, bb, const_u32(c, 31));
    uint32_t sc = u32_bin(c, b, S_ShrL, bc, const_u32(c, 31));

    uint32_t ma_nz = cmp_u32(c, b, PCM_NE, 0, ma, const_u32(c, 0), line);
    uint32_t mb_nz = cmp_u32(c, b, PCM_NE, 0, mb, const_u32(c, 0), line);
    uint32_t mc_nz = cmp_u32(c, b, PCM_NE, 0, mc, const_u32(c, 0), line);
    uint32_t ea0 = cmp_u32(c, b, PCM_EQ, 0, ea, const_u32(c, 0), line);
    uint32_t eb0 = cmp_u32(c, b, PCM_EQ, 0, eb, const_u32(c, 0), line);
    uint32_t ec0 = cmp_u32(c, b, PCM_EQ, 0, ec, const_u32(c, 0), line);
    uint32_t ea255 = cmp_u32(c, b, PCM_EQ, 0, ea, const_u32(c, 255), line);
    uint32_t eb255 = cmp_u32(c, b, PCM_EQ, 0, eb, const_u32(c, 255), line);
    uint32_t ec255 = cmp_u32(c, b, PCM_EQ, 0, ec, const_u32(c, 255), line);
    uint32_t nan_a = bool_bin(c, b, S_LogicalAnd, ea255, ma_nz);
    uint32_t nan_b = bool_bin(c, b, S_LogicalAnd, eb255, mb_nz);
    uint32_t nan_c = bool_bin(c, b, S_LogicalAnd, ec255, mc_nz);
    uint32_t inf_a = bool_bin(c, b, S_LogicalAnd, ea255, bool_not(c, b, ma_nz));
    uint32_t inf_b = bool_bin(c, b, S_LogicalAnd, eb255, bool_not(c, b, mb_nz));
    uint32_t inf_c = bool_bin(c, b, S_LogicalAnd, ec255, bool_not(c, b, mc_nz));
    uint32_t zero_a = bool_bin(c, b, S_LogicalAnd, ea0, bool_not(c, b, ma_nz));
    uint32_t zero_b = bool_bin(c, b, S_LogicalAnd, eb0, bool_not(c, b, mb_nz));
    uint32_t zero_c = bool_bin(c, b, S_LogicalAnd, ec0, bool_not(c, b, mc_nz));

    // Significands with the implicit bit, and the raw LSB exponents.
    uint32_t aimp = u32_bin(c, b, S_Or, ma, const_u32(c, 0x800000u));
    uint32_t bimp = u32_bin(c, b, S_Or, mb, const_u32(c, 0x800000u));
    uint32_t cimp = u32_bin(c, b, S_Or, mc, const_u32(c, 0x800000u));
    uint32_t A = sel_u32(c, b, ea0, ma, aimp);
    uint32_t B = sel_u32(c, b, eb0, mb, bimp);
    uint32_t C = sel_u32(c, b, ec0, mc, cimp);
    uint32_t exA = sel_u32(c, b, ea0, const_u32(c, (uint32_t)-149),
                           u32_bin(c, b, S_ISub, ea, const_u32(c, 150)));
    uint32_t exB = sel_u32(c, b, eb0, const_u32(c, (uint32_t)-149),
                           u32_bin(c, b, S_ISub, eb, const_u32(c, 150)));
    uint32_t exC = sel_u32(c, b, ec0, const_u32(c, (uint32_t)-149),
                           u32_bin(c, b, S_ISub, ec, const_u32(c, 150)));

    uint32_t A64 = zext_u32_u64(c, b, A);
    uint32_t B64 = zext_u32_u64(c, b, B);
    uint32_t C64 = zext_u32_u64(c, b, C);
    uint32_t P = u64_bin(c, b, S_IMul, A64, B64);
    uint32_t exP = u32_bin(c, b, S_IAdd, exA, exB);
    uint32_t sP = u32_bin(c, b, S_Xor, sa, sb);

    uint32_t clzP = clz64_val(c, b, P, line);
    uint32_t bitsP = u32_bin(c, b, S_ISub, const_u32(c, 64), clzP);
    uint32_t clzC = clz32(c, b, C, line);
    uint32_t bitsC = u32_bin(c, b, S_ISub, const_u32(c, 32), clzC);
    uint32_t mP = u32_bin(c, b, S_ISub,
                          u32_bin(c, b, S_IAdd, exP, bitsP),
                          const_u32(c, 1));
    uint32_t mC = u32_bin(c, b, S_ISub,
                          u32_bin(c, b, S_IAdd, exC, bitsC),
                          const_u32(c, 1));
    uint32_t pnz = arith_cmp_u64(c, b, PCM_NE, P, const_u64(c, 0), line);
    uint32_t cnz = cmp_u32(c, b, PCM_NE, 0, C, const_u32(c, 0), line);
    mP = sel_u32(c, b, pnz, mP, const_u32(c, 0x80000000u));
    mC = sel_u32(c, b, cnz, mC, const_u32(c, 0x80000000u));

    uint32_t useP = cmp_u32(c, b, PCM_GE, 1, mP, mC, line);
    uint32_t bitsHi = sel_u32(c, b, useP, bitsP, bitsC);
    uint32_t hiExp = sel_u32(c, b, useP, exP, exC);
    uint32_t loExp = sel_u32(c, b, useP, exC, exP);
    uint32_t hiSign = sel_u32(c, b, useP, sP, sc);
    uint32_t loSign = sel_u32(c, b, useP, sc, sP);
    uint32_t hiSig = sel_u64(c, b, useP, P, C64);
    uint32_t loSig = sel_u64(c, b, useP, C64, P);

    // The leading bit of hi lands at bit 62; lo is aligned against it.
    uint32_t shiftHi = u32_bin(c, b, S_IAdd,
                               u32_bin(c, b, S_ISub, const_u32(c, 62), bitsHi),
                               const_u32(c, 1));
    uint32_t hi = u64_bin(c, b, S_Shl, hiSig, zext_u32_u64(c, b, shiftHi));
    uint32_t shiftLo = u32_bin(c, b, S_IAdd,
                               u32_bin(c, b, S_ISub, loExp, hiExp), shiftHi);
    uint32_t loNeg = cmp_u32(c, b, PCM_LT, 1, shiftLo, const_u32(c, 0), line);
    uint32_t shlAmt = u32_bin(c, b, S_And, shiftLo, const_u32(c, 63));
    uint32_t loLeft = u64_bin(c, b, S_Shl, loSig,
                              zext_u32_u64(c, b, shlAmt));
    uint32_t sh = u32_bin(c, b, S_ISub, const_u32(c, 0), shiftLo);
    uint32_t shc = u32_bin(c, b, S_And, sh, const_u32(c, 63));
    uint32_t loRight = u64_bin(c, b, S_ShrL, loSig,
                               zext_u32_u64(c, b, shc));
    uint32_t lmask = u64_bin(c, b, S_ISub,
                             u64_bin(c, b, S_Shl, const_u64(c, 1),
                                     zext_u32_u64(c, b, shc)),
                             const_u64(c, 1));
    uint32_t stLow = arith_cmp_u64(c, b, PCM_NE,
                             u64_bin(c, b, S_And, loSig, lmask),
                             const_u64(c, 0), line);
    uint32_t shBig = cmp_u32(c, b, PCM_GE, 0, sh, const_u32(c, 64), line);
    uint32_t loNZ = arith_cmp_u64(c, b, PCM_NE, loSig, const_u64(c, 0), line);
    uint32_t loShifted = sel_u64(c, b, shBig, const_u64(c, 0), loRight);
    uint32_t lo = sel_u64(c, b, loNeg, loShifted, loLeft);
    uint32_t sticky = sel_bool(c, b, loNeg,
                               sel_bool(c, b, shBig, loNZ, stLow),
                               bool_const(c, 0));
    lo = sel_u64(c, b, loNZ, lo, const_u64(c, 0));
    sticky = sel_bool(c, b, loNZ, sticky, bool_const(c, 0));

    uint32_t sameSign = cmp_u32(c, b, PCM_EQ, 0, hiSign, loSign, line);
    uint32_t hiGT = arith_cmp_u64(c, b, PCM_GT, hi, lo, line);
    uint32_t hiEQ = arith_cmp_u64(c, b, PCM_EQ, hi, lo, line);
    uint32_t sum = u64_bin(c, b, S_IAdd, hi, lo);
    uint32_t dgt = u64_bin(c, b, S_ISub, hi, lo);
    uint32_t dgt1 = u64_bin(c, b, S_ISub, dgt, const_u64(c, 1));
    uint32_t dlt = u64_bin(c, b, S_ISub, lo, hi);
    uint32_t dgtS = sel_u64(c, b, sticky, dgt1, dgt);
    uint32_t opp = sel_u64(c, b, hiGT, dgtS,
                           sel_u64(c, b, hiEQ, const_u64(c, 0), dlt));
    uint32_t full = sel_u64(c, b, sameSign, sum, opp);

    uint32_t notHiSign = u32_bin(c, b, S_Xor, hiSign, const_u32(c, 1));
    uint32_t eqRes = sel_u32(c, b, sticky, notHiSign, const_u32(c, 0));
    uint32_t oppRes = sel_u32(c, b, hiGT, hiSign,
                              sel_u32(c, b, hiEQ, eqRes, loSign));
    uint32_t sRes = sel_u32(c, b, sameSign, hiSign, oppRes);
    uint32_t sticky32 = sel_u32(c, b, sticky, const_u32(c, 1),
                                const_u32(c, 0));
    uint32_t gtFrac = sel_u32(c, b, sticky, const_u32(c, 1),
                              const_u32(c, 0));
    uint32_t frac = sel_u32(c, b, sameSign, sticky32,
                            sel_u32(c, b, hiGT, gtFrac, sticky32));

    uint32_t full_nz = arith_cmp_u64(c, b, PCM_NE, full, const_u64(c, 0), line);
    if (exact_sign)
        *exact_sign = sRes;
    if (exact_nz)
        *exact_nz = bool_bin(c, b, S_LogicalOr, full_nz,
                             cmp_u32(c, b, PCM_NE, 0, frac,
                                     const_u32(c, 0), line));
    uint32_t clzF = clz64_val(c, b, full, line);
    uint32_t kRaw = u32_bin(c, b, S_ISub, const_u32(c, 63), clzF);
    uint32_t k = sel_u32(c, b, full_nz, kRaw, const_u32(c, 0));
    uint32_t E = u32_bin(c, b, S_ISub, hiExp, shiftHi);
    uint32_t eVal = u32_bin(c, b, S_IAdd, E, k);

    // Normalize to a 27-bit integer total: value = total * 2^(eVal-26).
    uint32_t big = cmp_u32(c, b, PCM_GT, 0, k, const_u32(c, 26), line);
    uint32_t shR = u32_bin(c, b, S_ISub, k, const_u32(c, 26));
    uint32_t shRc = u32_bin(c, b, S_And, shR, const_u32(c, 63));
    uint32_t dmask = u64_bin(c, b, S_ISub,
                             u64_bin(c, b, S_Shl, const_u64(c, 1),
                                     zext_u32_u64(c, b, shRc)),
                             const_u64(c, 1));
    uint32_t dropped = arith_cmp_u64(c, b, PCM_NE,
                               u64_bin(c, b, S_And, full, dmask),
                               const_u64(c, 0), line);
    uint32_t totalR = u64_bin(c, b, S_ShrL, full,
                              zext_u32_u64(c, b, shRc));
    uint32_t shL = u32_bin(c, b, S_And,
                           u32_bin(c, b, S_ISub, const_u32(c, 26), k),
                           const_u32(c, 63));
    uint32_t totalL = u64_bin(c, b, S_Shl, full,
                              zext_u32_u64(c, b, shL));
    uint32_t total = sel_u64(c, b, big, totalR, totalL);
    uint32_t drop32 = sel_u32(c, b, dropped, const_u32(c, 1),
                              const_u32(c, 0));
    uint32_t big32 = sel_u32(c, b, big, const_u32(c, 1), const_u32(c, 0));
    uint32_t extra = u32_bin(c, b, S_Or, frac,
                             u32_bin(c, b, S_And, big32, drop32));

    uint32_t extraNZ = cmp_u32(c, b, PCM_NE, 0, extra,
                               const_u32(c, 0), line);
    uint32_t sResNeg = cmp_u32(c, b, PCM_NE, 0, sRes,
                               const_u32(c, 0), line);
    uint32_t sResPos = bool_not(c, b, sResNeg);
    uint32_t signbit = u32_bin(c, b, S_Shl, sRes, const_u32(c, 31));

    // ---- normal-result path ----
    uint32_t qq = u64_bin(c, b, S_ShrL, total, const_u64(c, 3));
    uint32_t rem = u64_bin(c, b, S_And, total, const_u64(c, 7));
    uint32_t remNZ = arith_cmp_u64(c, b, PCM_NE, rem, const_u64(c, 0), line);
    uint32_t anyN = bool_bin(c, b, S_LogicalOr, remNZ, extraNZ);
    uint32_t gtHalf = arith_cmp_u64(c, b, PCM_GT, rem, const_u64(c, 4), line);
    uint32_t eqHalf = arith_cmp_u64(c, b, PCM_EQ, rem, const_u64(c, 4), line);
    uint32_t gtN = bool_bin(c, b, S_LogicalOr, gtHalf,
                            bool_bin(c, b, S_LogicalAnd, eqHalf, extraNZ));
    uint32_t eqN = bool_bin(c, b, S_LogicalAnd, eqHalf,
                            bool_not(c, b, extraNZ));
    uint32_t rupN;
    if (mode == 0) {
        uint32_t odd = arith_cmp_u64(c, b, PCM_NE,
                               u64_bin(c, b, S_And, qq, const_u64(c, 1)),
                               const_u64(c, 0), line);
        rupN = bool_bin(c, b, S_LogicalOr, gtN,
                        bool_bin(c, b, S_LogicalAnd, eqN, odd));
    } else if (mode == 3) {
        rupN = bool_bin(c, b, S_LogicalAnd, sResPos, anyN);
    } else if (mode == 2) {
        rupN = bool_bin(c, b, S_LogicalAnd, sResNeg, anyN);
    } else {
        rupN = bool_const(c, 0);
    }
    uint32_t qq2 = u64_bin(c, b, S_IAdd, qq,
                           sel_u64(c, b, rupN, const_u64(c, 1),
                                   const_u64(c, 0)));
    uint32_t carryN = arith_cmp_u64(c, b, PCM_EQ, qq2,
                              const_u64(c, (uint64_t)1 << 24), line);
    uint32_t qq3 = sel_u64(c, b, carryN,
                           u64_bin(c, b, S_ShrL, qq2, const_u64(c, 1)), qq2);
    uint32_t eVal2 = u32_bin(c, b, S_IAdd, eVal,
                             sel_u32(c, b, carryN, const_u32(c, 1),
                                     const_u32(c, 0)));
    uint32_t be = u32_bin(c, b, S_IAdd, eVal2, const_u32(c, 127));
    uint32_t ovf = cmp_u32(c, b, PCM_GE, 0, be, const_u32(c, 255), line);
    uint32_t mbits = u32_bin(c, b, S_And, trunc_u64_u32(c, b, qq3),
                             const_u32(c, 0x7fffffu));
    uint32_t bitsNorm = u32_bin(c, b, S_Or, signbit,
                                u32_bin(c, b, S_Or,
                                        u32_bin(c, b, S_Shl, be,
                                                const_u32(c, 23)),
                                        mbits));
    uint32_t infBits = u32_bin(c, b, S_Or, signbit,
                               const_u32(c, 0x7f800000u));
    uint32_t maxBits = u32_bin(c, b, S_Or, signbit,
                               const_u32(c, 0x7f7fffffu));
    uint32_t ovfBits;
    if (mode == 0) {
        ovfBits = infBits;
    } else if (mode == 1) {
        ovfBits = maxBits;
    } else if (mode == 3) {
        ovfBits = sel_u32(c, b, sResNeg, maxBits,
                          const_u32(c, 0x7f800000u));
    } else {
        ovfBits = sel_u32(c, b, sResNeg, const_u32(c, 0xff800000u),
                          const_u32(c, 0x7f7fffffu));
    }
    uint32_t bitsNormal = sel_u32(c, b, ovf, ovfBits, bitsNorm);

    // ---- subnormal-result path ----
    uint32_t sU = u32_bin(c, b, S_ISub, const_u32(c, (uint32_t)-123), eVal);
    uint32_t sBig = cmp_u32(c, b, PCM_GE, 0, sU, const_u32(c, 64), line);
    uint32_t subSh = u32_bin(c, b, S_And, sU, const_u32(c, 63));
    uint32_t qsRaw = u64_bin(c, b, S_ShrL, total,
                             zext_u32_u64(c, b, subSh));
    uint32_t smask = u64_bin(c, b, S_ISub,
                             u64_bin(c, b, S_Shl, const_u64(c, 1),
                                     zext_u32_u64(c, b, subSh)),
                             const_u64(c, 1));
    uint32_t srem = u64_bin(c, b, S_And, total, smask);
    uint32_t shalf = u64_bin(c, b, S_Shl, const_u64(c, 1),
                             zext_u32_u64(c, b,
                                          u32_bin(c, b, S_And,
                                                  u32_bin(c, b, S_ISub,
                                                          subSh,
                                                          const_u32(c, 1)),
                                                  const_u32(c, 63))));
    uint32_t sremNZ = arith_cmp_u64(c, b, PCM_NE, srem, const_u64(c, 0), line);
    uint32_t sAny0 = bool_bin(c, b, S_LogicalOr, sremNZ, extraNZ);
    uint32_t sgt0 = bool_bin(c, b, S_LogicalOr,
                             arith_cmp_u64(c, b, PCM_GT, srem, shalf, line),
                             bool_bin(c, b, S_LogicalAnd,
                                      arith_cmp_u64(c, b, PCM_EQ, srem, shalf,
                                              line),
                                      extraNZ));
    uint32_t seq0 = bool_bin(c, b, S_LogicalAnd,
                             arith_cmp_u64(c, b, PCM_EQ, srem, shalf, line),
                             bool_not(c, b, extraNZ));
    uint32_t sAny = sel_bool(c, b, sBig, bool_const(c, 1), sAny0);
    uint32_t sgt = sel_bool(c, b, sBig, bool_const(c, 0), sgt0);
    uint32_t seq = sel_bool(c, b, sBig, bool_const(c, 0), seq0);
    uint32_t rupS;
    if (mode == 0) {
        uint32_t oddS = arith_cmp_u64(c, b, PCM_NE,
                                u64_bin(c, b, S_And, qsRaw,
                                        const_u64(c, 1)),
                                const_u64(c, 0), line);
        rupS = bool_bin(c, b, S_LogicalOr, sgt,
                        bool_bin(c, b, S_LogicalAnd, seq, oddS));
    } else if (mode == 3) {
        rupS = bool_bin(c, b, S_LogicalAnd, sResPos, sAny);
    } else if (mode == 2) {
        rupS = bool_bin(c, b, S_LogicalAnd, sResNeg, sAny);
    } else {
        rupS = bool_const(c, 0);
    }
    uint32_t qs = sel_u64(c, b, sBig, const_u64(c, 0), qsRaw);
    uint32_t qs2 = u64_bin(c, b, S_IAdd, qs,
                           sel_u64(c, b, rupS, const_u64(c, 1),
                                   const_u64(c, 0)));
    uint32_t minnorm = arith_cmp_u64(c, b, PCM_GE, qs2,
                               const_u64(c, (uint64_t)1 << 23), line);
    uint32_t subBits = u32_bin(c, b, S_Or, signbit,
                               u32_bin(c, b, S_And,
                                       trunc_u64_u32(c, b, qs2),
                                       const_u32(c, 0x7fffffu)));
    uint32_t bitsSub = sel_u32(c, b, minnorm,
                               u32_bin(c, b, S_Or, signbit,
                                       const_u32(c, 1u << 23)),
                               subBits);

    uint32_t isNorm = cmp_u32(c, b, PCM_GE, 1, eVal,
                              const_u32(c, (uint32_t)-126), line);
    uint32_t bitsFinite = sel_u32(c, b, isNorm, bitsNormal, bitsSub);
    uint32_t zeroSign = const_u32(c, mode == 2 ? 1u : 0u);
    uint32_t bitsZero = u32_bin(c, b, S_Shl, zeroSign, const_u32(c, 31));
    uint32_t bits = sel_u32(c, b, full_nz, bitsFinite, bitsZero);

    // ---- special cases, NaN taking priority ----
    uint32_t sp = sP;
    uint32_t samesz = cmp_u32(c, b, PCM_EQ, 0, sp, sc, line);
    uint32_t zsign = sel_u32(c, b, samesz, sp,
                             const_u32(c, mode == 2 ? 1u : 0u));
    uint32_t zeroRule = u32_bin(c, b, S_Shl, zsign, const_u32(c, 31));
    uint32_t prodZero = bool_bin(c, b, S_LogicalOr, zero_a, zero_b);
    uint32_t zeroCase = bool_bin(c, b, S_LogicalAnd, prodZero, zero_c);
    uint32_t infP = u32_bin(c, b, S_Or,
                            u32_bin(c, b, S_Shl, sp, const_u32(c, 31)),
                            const_u32(c, 0x7f800000u));
    uint32_t infCb = u32_bin(c, b, S_Or,
                             u32_bin(c, b, S_Shl, sc, const_u32(c, 31)),
                             const_u32(c, 0x7f800000u));
    uint32_t prodInf = bool_bin(c, b, S_LogicalOr, inf_a, inf_b);
    uint32_t bothInf = bool_bin(c, b, S_LogicalAnd, inf_c,
                                cmp_u32(c, b, PCM_NE, 0, sc, sp, line));
    uint32_t pInfBits = sel_u32(c, b, bothInf,
                                const_u32(c, 0x7fc00000u), infP);
    uint32_t infTimesZero =
        bool_bin(c, b, S_LogicalOr,
                 bool_bin(c, b, S_LogicalAnd, inf_a, zero_b),
                 bool_bin(c, b, S_LogicalAnd, zero_a, inf_b));
    uint32_t anyNan = bool_bin(c, b, S_LogicalOr,
                               bool_bin(c, b, S_LogicalOr, nan_a, nan_b),
                               nan_c);
    bits = sel_u32(c, b, zeroCase, zeroRule, bits);
    bits = sel_u32(c, b, inf_c, infCb, bits);
    bits = sel_u32(c, b, prodInf, pInfBits, bits);
    bits = sel_u32(c, b, infTimesZero, const_u32(c, 0x7fc00000u), bits);
    bits = sel_u32(c, b, anyNan, const_u32(c, 0x7fc00000u), bits);
    return bits_to_f32(c, b, bits);
}

