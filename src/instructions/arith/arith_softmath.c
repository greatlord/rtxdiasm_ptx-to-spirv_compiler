// Arithmetic, conversion, bit-manipulation, comparison and select
// helpers and opcode emission (split from arith.c).
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
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

// ---- 128-bit unsigned helpers (two u64 words) -------------------------
static uint32_t u64x2_type(ctx_t *c) {
    if (!c->t_u64x2) {
        c->t_u64x2 = nid(c);
        put(c, &c->types, (4u << 16) | S_TypeStruct);
        put(c, &c->types, c->t_u64x2);
        put(c, &c->types, c->t_u64);
        put(c, &c->types, c->t_u64);
    }
    return c->t_u64x2;
}

static void u128_mul(ctx_t *c, spv_blob_t *b, uint32_t x, uint32_t y,
                     uint32_t *ohi, uint32_t *olo) {
    uint32_t ty = u64x2_type(c);
    uint32_t r = nid(c);
    put(c, b, (5u << 16) | S_UMulExtended);
    put(c, b, ty);
    put(c, b, r);
    put(c, b, x);
    put(c, b, y);
    uint32_t lo = nid(c);
    put(c, b, (5u << 16) | S_CompositeExtract);
    put(c, b, c->t_u64);
    put(c, b, lo);
    put(c, b, r);
    put(c, b, 0);
    uint32_t hi = nid(c);
    put(c, b, (5u << 16) | S_CompositeExtract);
    put(c, b, c->t_u64);
    put(c, b, hi);
    put(c, b, r);
    put(c, b, 1);
    *ohi = hi;
    *olo = lo;
}

static void u128_shl(ctx_t *c, spv_blob_t *b, uint32_t hi, uint32_t lo,
                     uint32_t s, uint32_t *ohi, uint32_t *olo) {
    uint32_t w = u32_bin(c, b, S_ShrL, s, const_u32(c, 6));
    uint32_t sh = u32_bin(c, b, S_And, s, const_u32(c, 63));
    uint32_t loShl = u64_bin(c, b, S_Shl, lo, zext_u32_u64(c, b, sh));
    uint32_t hiShl = u64_bin(c, b, S_Shl, hi, zext_u32_u64(c, b, sh));
    uint32_t sh0 = cmp_u32(c, b, PCM_EQ, 0, sh, const_u32(c, 0), 0);
    uint32_t nb = u32_bin(c, b, S_And,
                          u32_bin(c, b, S_ISub, const_u32(c, 64), sh),
                          const_u32(c, 63));
    uint32_t cross = u64_bin(c, b, S_ShrL, lo, zext_u32_u64(c, b, nb));
    cross = sel_u64(c, b, sh0, const_u64(c, 0), cross);
    uint32_t hiMid = u64_bin(c, b, S_Or, hiShl, cross);
    uint32_t wNZ = cmp_u32(c, b, PCM_NE, 0, w, const_u32(c, 0), 0);
    *ohi = sel_u64(c, b, wNZ, loShl, hiMid);
    *olo = sel_u64(c, b, wNZ, const_u64(c, 0), loShl);
}

static void u128_shr(ctx_t *c, spv_blob_t *b, uint32_t hi, uint32_t lo,
                     uint32_t s, uint32_t *ohi, uint32_t *olo,
                     uint32_t *sticky) {
    uint32_t w = u32_bin(c, b, S_ShrL, s, const_u32(c, 6));
    uint32_t sh = u32_bin(c, b, S_And, s, const_u32(c, 63));
    uint32_t loShr = u64_bin(c, b, S_ShrL, lo, zext_u32_u64(c, b, sh));
    uint32_t hiShr = u64_bin(c, b, S_ShrL, hi, zext_u32_u64(c, b, sh));
    uint32_t sh0 = cmp_u32(c, b, PCM_EQ, 0, sh, const_u32(c, 0), 0);
    uint32_t nb = u32_bin(c, b, S_And,
                          u32_bin(c, b, S_ISub, const_u32(c, 64), sh),
                          const_u32(c, 63));
    uint32_t cross = u64_bin(c, b, S_Shl, hi, zext_u32_u64(c, b, nb));
    cross = sel_u64(c, b, sh0, const_u64(c, 0), cross);
    uint32_t loMid = u64_bin(c, b, S_Or, loShr, cross);
    uint32_t mask = u64_bin(c, b, S_ISub,
                            u64_bin(c, b, S_Shl, const_u64(c, 1),
                                    zext_u32_u64(c, b, sh)),
                            const_u64(c, 1));
    uint32_t st0 = arith_cmp_u64(c, b, PCM_NE,
                           u64_bin(c, b, S_And, lo, mask),
                           const_u64(c, 0), 0);
    uint32_t st1 = bool_bin(c, b, S_LogicalOr,
                            arith_cmp_u64(c, b, PCM_NE, lo, const_u64(c, 0), 0),
                            arith_cmp_u64(c, b, PCM_NE,
                                    u64_bin(c, b, S_And, hi, mask),
                                    const_u64(c, 0), 0));
    uint32_t wNZ = cmp_u32(c, b, PCM_NE, 0, w, const_u32(c, 0), 0);
    *ohi = sel_u64(c, b, wNZ, const_u64(c, 0), hiShr);
    *olo = sel_u64(c, b, wNZ, hiShr, loMid);
    *sticky = sel_bool(c, b, wNZ, st1, st0);
}

static void u128_add(ctx_t *c, spv_blob_t *b, uint32_t ah, uint32_t al,
                     uint32_t bh, uint32_t bl, uint32_t *ohi,
                     uint32_t *olo) {
    uint32_t sum = u64_bin(c, b, S_IAdd, al, bl);
    uint32_t carry = arith_cmp_u64(c, b, PCM_LT, sum, al, 0);
    uint32_t hiSum = u64_bin(c, b, S_IAdd, ah, bh);
    *ohi = u64_bin(c, b, S_IAdd, hiSum,
                   zext_u32_u64(c, b,
                                sel_u32(c, b, carry, const_u32(c, 1),
                                        const_u32(c, 0))));
    *olo = sum;
}

static void u128_sub(ctx_t *c, spv_blob_t *b, uint32_t ah, uint32_t al,
                     uint32_t bh, uint32_t bl, uint32_t *ohi,
                     uint32_t *olo) {
    uint32_t dif = u64_bin(c, b, S_ISub, al, bl);
    uint32_t borrow = arith_cmp_u64(c, b, PCM_LT, al, bl, 0);
    uint32_t hiDif = u64_bin(c, b, S_ISub, ah, bh);
    *ohi = u64_bin(c, b, S_ISub, hiDif,
                   zext_u32_u64(c, b,
                                sel_u32(c, b, borrow, const_u32(c, 1),
                                        const_u32(c, 0))));
    *olo = dif;
}

static uint32_t u128_gt(ctx_t *c, spv_blob_t *b, uint32_t ah, uint32_t al,
                        uint32_t bh, uint32_t bl) {
    uint32_t hgt = arith_cmp_u64(c, b, PCM_GT, ah, bh, 0);
    uint32_t heq = arith_cmp_u64(c, b, PCM_EQ, ah, bh, 0);
    uint32_t lgt = arith_cmp_u64(c, b, PCM_GT, al, bl, 0);
    return bool_bin(c, b, S_LogicalOr, hgt,
                    bool_bin(c, b, S_LogicalAnd, heq, lgt));
}

static uint32_t u128_eq(ctx_t *c, spv_blob_t *b, uint32_t ah, uint32_t al,
                        uint32_t bh, uint32_t bl) {
    return bool_bin(c, b, S_LogicalAnd,
                    arith_cmp_u64(c, b, PCM_EQ, ah, bh, 0),
                    arith_cmp_u64(c, b, PCM_EQ, al, bl, 0));
}

static uint32_t u128_nz(ctx_t *c, spv_blob_t *b, uint32_t hi,
                        uint32_t lo) {
    return bool_bin(c, b, S_LogicalOr,
                    arith_cmp_u64(c, b, PCM_NE, hi, const_u64(c, 0), 0),
                    arith_cmp_u64(c, b, PCM_NE, lo, const_u64(c, 0), 0));
}

static uint32_t u128_bits(ctx_t *c, spv_blob_t *b, uint32_t hi,
                          uint32_t lo) {
    uint32_t clzHi = clz64_val(c, b, hi, 0);
    uint32_t clzLo = clz64_val(c, b, lo, 0);
    uint32_t hiNZ = arith_cmp_u64(c, b, PCM_NE, hi, const_u64(c, 0), 0);
    uint32_t clz = sel_u32(c, b, hiNZ, clzHi,
                           u32_bin(c, b, S_IAdd, clzLo,
                                   const_u32(c, 64)));
    return u32_bin(c, b, S_ISub, const_u32(c, 128), clz);
}

uint32_t f64_to_bits(ctx_t *c, spv_blob_t *b, uint32_t f) {
    uint32_t id = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_u64);
    put(c, b, id);
    put(c, b, f);
    return id;
}

uint32_t bits_to_f64(ctx_t *c, spv_blob_t *b, uint32_t v) {
    uint32_t id = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_f64);
    put(c, b, id);
    put(c, b, v);
    return id;
}

// ---- Portable f64 fused multiply-add ---------------------------------
// Same normalized-window algorithm as the f32 version, with 128-bit
// integer significands (two u64 words) and a 56-bit normalized total.
uint32_t f64_soft_fma(ctx_t *c, spv_blob_t *b, uint32_t fa,
                             uint32_t fb, uint32_t fc, int mode, int line,
                             uint32_t *exact_sign, uint32_t *exact_nz) {
    uint32_t ba = f64_to_bits(c, b, fa);
    uint32_t bb = f64_to_bits(c, b, fb);
    uint32_t bc = f64_to_bits(c, b, fc);
    uint32_t ea = u32_bin(c, b, S_And,
                          trunc_u64_u32(c, b,
                                        u64_bin(c, b, S_ShrL, ba,
                                                const_u64(c, 52))),
                          const_u32(c, 0x7ffu));
    uint32_t eb = u32_bin(c, b, S_And,
                          trunc_u64_u32(c, b,
                                        u64_bin(c, b, S_ShrL, bb,
                                                const_u64(c, 52))),
                          const_u32(c, 0x7ffu));
    uint32_t ec = u32_bin(c, b, S_And,
                          trunc_u64_u32(c, b,
                                        u64_bin(c, b, S_ShrL, bc,
                                                const_u64(c, 52))),
                          const_u32(c, 0x7ffu));
    uint32_t ma = u64_bin(c, b, S_And, ba,
                          const_u64(c, 0xfffffffffffffull));
    uint32_t mb = u64_bin(c, b, S_And, bb,
                          const_u64(c, 0xfffffffffffffull));
    uint32_t mc = u64_bin(c, b, S_And, bc,
                          const_u64(c, 0xfffffffffffffull));
    uint32_t sa = trunc_u64_u32(c, b,
                                u64_bin(c, b, S_ShrL, ba,
                                        const_u64(c, 63)));
    uint32_t sb = trunc_u64_u32(c, b,
                                u64_bin(c, b, S_ShrL, bb,
                                        const_u64(c, 63)));
    uint32_t sc = trunc_u64_u32(c, b,
                                u64_bin(c, b, S_ShrL, bc,
                                        const_u64(c, 63)));

    uint32_t ma_nz = arith_cmp_u64(c, b, PCM_NE, ma, const_u64(c, 0), line);
    uint32_t mb_nz = arith_cmp_u64(c, b, PCM_NE, mb, const_u64(c, 0), line);
    uint32_t mc_nz = arith_cmp_u64(c, b, PCM_NE, mc, const_u64(c, 0), line);
    uint32_t ea0 = cmp_u32(c, b, PCM_EQ, 0, ea, const_u32(c, 0), line);
    uint32_t eb0 = cmp_u32(c, b, PCM_EQ, 0, eb, const_u32(c, 0), line);
    uint32_t ec0 = cmp_u32(c, b, PCM_EQ, 0, ec, const_u32(c, 0), line);
    uint32_t ea255 = cmp_u32(c, b, PCM_EQ, 0, ea, const_u32(c, 0x7ffu),
                             line);
    uint32_t eb255 = cmp_u32(c, b, PCM_EQ, 0, eb, const_u32(c, 0x7ffu),
                             line);
    uint32_t ec255 = cmp_u32(c, b, PCM_EQ, 0, ec, const_u32(c, 0x7ffu),
                             line);
    uint32_t nan_a = bool_bin(c, b, S_LogicalAnd, ea255, ma_nz);
    uint32_t nan_b = bool_bin(c, b, S_LogicalAnd, eb255, mb_nz);
    uint32_t nan_c = bool_bin(c, b, S_LogicalAnd, ec255, mc_nz);
    uint32_t inf_a = bool_bin(c, b, S_LogicalAnd, ea255, bool_not(c, b, ma_nz));
    uint32_t inf_b = bool_bin(c, b, S_LogicalAnd, eb255, bool_not(c, b, mb_nz));
    uint32_t inf_c = bool_bin(c, b, S_LogicalAnd, ec255, bool_not(c, b, mc_nz));
    uint32_t zero_a = bool_bin(c, b, S_LogicalAnd, ea0, bool_not(c, b, ma_nz));
    uint32_t zero_b = bool_bin(c, b, S_LogicalAnd, eb0, bool_not(c, b, mb_nz));
    uint32_t zero_c = bool_bin(c, b, S_LogicalAnd, ec0, bool_not(c, b, mc_nz));

    uint32_t aimp = u64_bin(c, b, S_Or, ma, const_u64(c, 1ull << 52));
    uint32_t bimp = u64_bin(c, b, S_Or, mb, const_u64(c, 1ull << 52));
    uint32_t cimp = u64_bin(c, b, S_Or, mc, const_u64(c, 1ull << 52));
    uint32_t A = sel_u64(c, b, ea0, ma, aimp);
    uint32_t B = sel_u64(c, b, eb0, mb, bimp);
    uint32_t C = sel_u64(c, b, ec0, mc, cimp);
    uint32_t exA = sel_u32(c, b, ea0, const_u32(c, (uint32_t)-1074),
                           u32_bin(c, b, S_ISub, ea, const_u32(c, 1075)));
    uint32_t exB = sel_u32(c, b, eb0, const_u32(c, (uint32_t)-1074),
                           u32_bin(c, b, S_ISub, eb, const_u32(c, 1075)));
    uint32_t exC = sel_u32(c, b, ec0, const_u32(c, (uint32_t)-1074),
                           u32_bin(c, b, S_ISub, ec, const_u32(c, 1075)));

    uint32_t Phi, Plo;
    u128_mul(c, b, A, B, &Phi, &Plo);
    uint32_t exP = u32_bin(c, b, S_IAdd, exA, exB);
    uint32_t sP = u32_bin(c, b, S_Xor, sa, sb);

    uint32_t bitsP = u128_bits(c, b, Phi, Plo);
    uint32_t bitsC = u32_bin(c, b, S_ISub, const_u32(c, 64),
                             clz64_val(c, b, C, line));
    uint32_t mP = u32_bin(c, b, S_ISub,
                          u32_bin(c, b, S_IAdd, exP, bitsP),
                          const_u32(c, 1));
    uint32_t mC = u32_bin(c, b, S_ISub,
                          u32_bin(c, b, S_IAdd, exC, bitsC),
                          const_u32(c, 1));
    uint32_t pnz = u128_nz(c, b, Phi, Plo);
    uint32_t cnz = arith_cmp_u64(c, b, PCM_NE, C, const_u64(c, 0), line);
    mP = sel_u32(c, b, pnz, mP, const_u32(c, 0x80000000u));
    mC = sel_u32(c, b, cnz, mC, const_u32(c, 0x80000000u));

    uint32_t useP = cmp_u32(c, b, PCM_GE, 1, mP, mC, line);
    uint32_t bitsHi = sel_u32(c, b, useP, bitsP, bitsC);
    uint32_t hiExp = sel_u32(c, b, useP, exP, exC);
    uint32_t loExp = sel_u32(c, b, useP, exC, exP);
    uint32_t hiSign = sel_u32(c, b, useP, sP, sc);
    uint32_t loSign = sel_u32(c, b, useP, sc, sP);
    uint32_t hiSigH = sel_u64(c, b, useP, Phi, const_u64(c, 0));
    uint32_t hiSigL = sel_u64(c, b, useP, Plo, C);
    uint32_t loSigH = sel_u64(c, b, useP, const_u64(c, 0), Phi);
    uint32_t loSigL = sel_u64(c, b, useP, C, Plo);

    const uint32_t W = 126;
    uint32_t shiftHi = u32_bin(c, b, S_ISub, const_u32(c, W + 1), bitsHi);
    uint32_t hiH, hiL;
    u128_shl(c, b, hiSigH, hiSigL, shiftHi, &hiH, &hiL);
    uint32_t shiftLo = u32_bin(c, b, S_IAdd,
                               u32_bin(c, b, S_ISub, loExp, hiExp), shiftHi);
    uint32_t loNeg = cmp_u32(c, b, PCM_LT, 1, shiftLo, const_u32(c, 0),
                             line);
    uint32_t shlAmt = u32_bin(c, b, S_And, shiftLo, const_u32(c, 127));
    uint32_t lLeftH, lLeftL;
    u128_shl(c, b, loSigH, loSigL, shlAmt, &lLeftH, &lLeftL);
    uint32_t sh = u32_bin(c, b, S_ISub, const_u32(c, 0), shiftLo);
    uint32_t shc = u32_bin(c, b, S_And, sh, const_u32(c, 127));
    uint32_t lRightH, lRightL, stLow;
    u128_shr(c, b, loSigH, loSigL, shc, &lRightH, &lRightL, &stLow);
    uint32_t shBig = cmp_u32(c, b, PCM_GE, 0, sh, const_u32(c, 128), line);
    uint32_t loNZ = u128_nz(c, b, loSigH, loSigL);
    uint32_t loH = sel_u64(c, b, loNeg,
                           sel_u64(c, b, shBig, const_u64(c, 0), lRightH),
                           lLeftH);
    uint32_t loL = sel_u64(c, b, loNeg,
                           sel_u64(c, b, shBig, const_u64(c, 0), lRightL),
                           lLeftL);
    uint32_t sticky = sel_bool(c, b, loNeg,
                               sel_bool(c, b, shBig, loNZ, stLow),
                               bool_const(c, 0));
    loH = sel_u64(c, b, loNZ, loH, const_u64(c, 0));
    loL = sel_u64(c, b, loNZ, loL, const_u64(c, 0));
    sticky = sel_bool(c, b, loNZ, sticky, bool_const(c, 0));

    uint32_t sameSign = cmp_u32(c, b, PCM_EQ, 0, hiSign, loSign, line);
    uint32_t hiGT = u128_gt(c, b, hiH, hiL, loH, loL);
    uint32_t hiEQ = u128_eq(c, b, hiH, hiL, loH, loL);
    uint32_t sumH, sumL;
    u128_add(c, b, hiH, hiL, loH, loL, &sumH, &sumL);
    uint32_t dgtH, dgtL, dgt1H, dgt1L, dltH, dltL;
    u128_sub(c, b, hiH, hiL, loH, loL, &dgtH, &dgtL);
    u128_sub(c, b, dgtH, dgtL, const_u64(c, 0), const_u64(c, 1),
             &dgt1H, &dgt1L);
    u128_sub(c, b, loH, loL, hiH, hiL, &dltH, &dltL);
    uint32_t dgtSH = sel_u64(c, b, sticky, dgt1H, dgtH);
    uint32_t dgtSL = sel_u64(c, b, sticky, dgt1L, dgtL);
    uint32_t oppH = sel_u64(c, b, hiGT, dgtSH,
                            sel_u64(c, b, hiEQ, const_u64(c, 0), dltH));
    uint32_t oppL = sel_u64(c, b, hiGT, dgtSL,
                            sel_u64(c, b, hiEQ, const_u64(c, 0), dltL));
    uint32_t fullH = sel_u64(c, b, sameSign, sumH, oppH);
    uint32_t fullL = sel_u64(c, b, sameSign, sumL, oppL);

    uint32_t notHiSign = u32_bin(c, b, S_Xor, hiSign, const_u32(c, 1));
    uint32_t eqRes = sel_u32(c, b, sticky, notHiSign, const_u32(c, 0));
    uint32_t oppRes = sel_u32(c, b, hiGT, hiSign,
                              sel_u32(c, b, hiEQ, eqRes, loSign));
    uint32_t sRes = sel_u32(c, b, sameSign, hiSign, oppRes);
    uint32_t sticky32 = sel_u32(c, b, sticky, const_u32(c, 1),
                                const_u32(c, 0));
    uint32_t frac = sel_u32(c, b, sameSign, sticky32,
                            sel_u32(c, b, hiGT, sticky32, sticky32));

    uint32_t full_nz = u128_nz(c, b, fullH, fullL);
    if (exact_sign)
        *exact_sign = sRes;
    if (exact_nz)
        *exact_nz = bool_bin(c, b, S_LogicalOr, full_nz,
                             cmp_u32(c, b, PCM_NE, 0, frac,
                                     const_u32(c, 0), line));
    uint32_t kRaw = u32_bin(c, b, S_ISub,
                            u128_bits(c, b, fullH, fullL), const_u32(c, 1));
    uint32_t k = sel_u32(c, b, full_nz, kRaw, const_u32(c, 0));
    uint32_t E = u32_bin(c, b, S_ISub, hiExp, shiftHi);
    uint32_t eVal = u32_bin(c, b, S_IAdd, E, k);

    // Normalize to a 56-bit integer total: value = total * 2^(eVal-55).
    uint32_t big = cmp_u32(c, b, PCM_GT, 0, k, const_u32(c, 55), line);
    uint32_t shR = u32_bin(c, b, S_ISub, k, const_u32(c, 55));
    uint32_t shRc = u32_bin(c, b, S_And, shR, const_u32(c, 127));
    uint32_t tRH, tRL, dropped;
    u128_shr(c, b, fullH, fullL, shRc, &tRH, &tRL, &dropped);
    uint32_t shL = u32_bin(c, b, S_And,
                           u32_bin(c, b, S_ISub, const_u32(c, 55), k),
                           const_u32(c, 127));
    uint32_t tLH, tLL;
    u128_shl(c, b, fullH, fullL, shL, &tLH, &tLL);
    uint32_t totalH = sel_u64(c, b, big, tRH, tLH);
    uint32_t total = sel_u64(c, b, big, tRL, tLL);
    uint32_t drop32 = sel_u32(c, b, dropped, const_u32(c, 1),
                              const_u32(c, 0));
    uint32_t big32 = sel_u32(c, b, big, const_u32(c, 1), const_u32(c, 0));
    uint32_t extra = u32_bin(c, b, S_Or, frac,
                             u32_bin(c, b, S_And, big32, drop32));
    (void)totalH;

    uint32_t extraNZ = cmp_u32(c, b, PCM_NE, 0, extra,
                               const_u32(c, 0), line);
    uint32_t sResNeg = cmp_u32(c, b, PCM_NE, 0, sRes,
                               const_u32(c, 0), line);
    uint32_t sResPos = bool_not(c, b, sResNeg);
    uint32_t signbit = u64_bin(c, b, S_Shl, zext_u32_u64(c, b, sRes),
                               const_u64(c, 63));

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
                              const_u64(c, (uint64_t)1 << 53), line);
    uint32_t qq3 = sel_u64(c, b, carryN,
                           u64_bin(c, b, S_ShrL, qq2, const_u64(c, 1)), qq2);
    uint32_t eVal2 = u32_bin(c, b, S_IAdd, eVal,
                             sel_u32(c, b, carryN, const_u32(c, 1),
                                     const_u32(c, 0)));
    uint32_t be = u32_bin(c, b, S_IAdd, eVal2, const_u32(c, 1023));
    uint32_t ovf = cmp_u32(c, b, PCM_GE, 0, be, const_u32(c, 2047), line);
    uint32_t mbits = u64_bin(c, b, S_And, qq3,
                             const_u64(c, 0xfffffffffffffull));
    uint32_t bitsNorm = u64_bin(c, b, S_Or, signbit,
                                u64_bin(c, b, S_Or,
                                        u64_bin(c, b, S_Shl,
                                                zext_u32_u64(c, b, be),
                                                const_u64(c, 52)),
                                        mbits));
    uint32_t infBits = u64_bin(c, b, S_Or, signbit,
                               const_u64(c, 0x7ff0000000000000ull));
    uint32_t maxBits = u64_bin(c, b, S_Or, signbit,
                               const_u64(c, 0x7fefffffffffffffull));
    uint32_t ovfBits;
    if (mode == 0) {
        ovfBits = infBits;
    } else if (mode == 1) {
        ovfBits = maxBits;
    } else if (mode == 3) {
        ovfBits = sel_u64(c, b, sResNeg, maxBits,
                          const_u64(c, 0x7ff0000000000000ull));
    } else {
        ovfBits = sel_u64(c, b, sResNeg,
                          const_u64(c, 0xfff0000000000000ull),
                          const_u64(c, 0x7fefffffffffffffull));
    }
    uint32_t bitsNormal = sel_u64(c, b, ovf, ovfBits, bitsNorm);

    // ---- subnormal-result path ----
    uint32_t sU = u32_bin(c, b, S_ISub, const_u32(c, (uint32_t)-1019),
                          eVal);
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
                               const_u64(c, (uint64_t)1 << 52), line);
    uint32_t subBits = u64_bin(c, b, S_Or, signbit,
                               u64_bin(c, b, S_And, qs2,
                                       const_u64(c, 0xfffffffffffffull)));
    uint32_t bitsSub = sel_u64(c, b, minnorm,
                               u64_bin(c, b, S_Or, signbit,
                                       const_u64(c, 1ull << 52)),
                               subBits);

    uint32_t isNorm = cmp_u32(c, b, PCM_GE, 1, eVal,
                              const_u32(c, (uint32_t)-1022), line);
    uint32_t bitsFinite = sel_u64(c, b, isNorm, bitsNormal, bitsSub);
    uint32_t zeroSign = const_u32(c, mode == 2 ? 1u : 0u);
    uint32_t bitsZero = u64_bin(c, b, S_Shl, zext_u32_u64(c, b, zeroSign),
                                const_u64(c, 63));
    uint32_t bits = sel_u64(c, b, full_nz, bitsFinite, bitsZero);

    // ---- special cases, NaN taking priority ----
    uint32_t sp = sP;
    uint32_t samesz = cmp_u32(c, b, PCM_EQ, 0, sp, sc, line);
    uint32_t zsign = sel_u32(c, b, samesz, sp,
                             const_u32(c, mode == 2 ? 1u : 0u));
    uint32_t zeroRule = u64_bin(c, b, S_Shl,
                                zext_u32_u64(c, b, zsign),
                                const_u64(c, 63));
    uint32_t prodZero = bool_bin(c, b, S_LogicalOr, zero_a, zero_b);
    uint32_t zeroCase = bool_bin(c, b, S_LogicalAnd, prodZero, zero_c);
    uint32_t infP = u64_bin(c, b, S_Or,
                            u64_bin(c, b, S_Shl,
                                    zext_u32_u64(c, b, sp),
                                    const_u64(c, 63)),
                            const_u64(c, 0x7ff0000000000000ull));
    uint32_t infCb = u64_bin(c, b, S_Or,
                             u64_bin(c, b, S_Shl,
                                     zext_u32_u64(c, b, sc),
                                     const_u64(c, 63)),
                             const_u64(c, 0x7ff0000000000000ull));
    uint32_t prodInf = bool_bin(c, b, S_LogicalOr, inf_a, inf_b);
    uint32_t bothInf = bool_bin(c, b, S_LogicalAnd, inf_c,
                                cmp_u32(c, b, PCM_NE, 0, sc, sp, line));
    uint32_t pInfBits = sel_u64(c, b, bothInf,
                                const_u64(c, 0x7ff8000000000000ull), infP);
    uint32_t infTimesZero =
        bool_bin(c, b, S_LogicalOr,
                 bool_bin(c, b, S_LogicalAnd, inf_a, zero_b),
                 bool_bin(c, b, S_LogicalAnd, zero_a, inf_b));
    uint32_t anyNan = bool_bin(c, b, S_LogicalOr,
                               bool_bin(c, b, S_LogicalOr, nan_a, nan_b),
                               nan_c);
    bits = sel_u64(c, b, zeroCase, zeroRule, bits);
    bits = sel_u64(c, b, inf_c, infCb, bits);
    bits = sel_u64(c, b, prodInf, pInfBits, bits);
    bits = sel_u64(c, b, infTimesZero,
                   const_u64(c, 0x7ff8000000000000ull), bits);
    bits = sel_u64(c, b, anyNan, const_u64(c, 0x7ff8000000000000ull), bits);
    return bits_to_f64(c, b, bits);
}

uint32_t f32_neg(ctx_t *c, spv_blob_t *b, uint32_t f);
uint32_t f64_neg(ctx_t *c, spv_blob_t *b, uint32_t f);

uint32_t emit_f64_fma_rnd(ctx_t *c, spv_blob_t *b, uint32_t x, uint32_t y,
                          uint32_t z, int mode, int line) {
    return f64_soft_fma(c, b, x, y, z, mode, line, NULL, NULL);
}

// Directed add/sub through the integer software FMA with a unit
// multiplier, which is exact for subnormal results too (hardware FAdd
// may flush those, and drivers may reassociate the Fast2Sum residual).
uint32_t f32_soft_addsub(ctx_t *c, spv_blob_t *b, uint32_t fa,
                                uint32_t fb, int is_sub, int mode,
                                int line) {
    uint32_t fc = is_sub ? f32_neg(c, b, fb) : fb;
    return f32_soft_fma(c, b, const_f32(c, 0x3f800000u), fa, fc, mode,
                        line, NULL, NULL);
}

uint32_t f64_soft_addsub(ctx_t *c, spv_blob_t *b, uint32_t fa,
                                uint32_t fb, int is_sub, int mode,
                                int line) {
    uint32_t fc = is_sub ? f64_neg(c, b, fb) : fb;
    return f64_soft_fma(c, b, const_f64(c, 0x3ff0000000000000ull), fa, fc,
                        mode, line, NULL, NULL);
}

// Directed multiply: fma(a,b,+0) has the same rounded value for every
// nonzero product; only the sign of an exactly-zero product differs
// from multiplication, so that case is selected back in.
uint32_t f32_directed_mul_soft(ctx_t *c, spv_blob_t *b, uint32_t fa,
                                      uint32_t fb, int mode, int line) {
    uint32_t r = f32_soft_fma(c, b, fa, fb, const_f32(c, 0), mode, line,
                              NULL, NULL);
    if (c->failed)
        return 0;
    uint32_t ba = f32_to_bits(c, b, fa);
    uint32_t bb = f32_to_bits(c, b, fb);
    uint32_t ea = u32_bin(c, b, S_And,
                          u32_bin(c, b, S_ShrL, ba, const_u32(c, 23)),
                          const_u32(c, 0xffu));
    uint32_t eb = u32_bin(c, b, S_And,
                          u32_bin(c, b, S_ShrL, bb, const_u32(c, 23)),
                          const_u32(c, 0xffu));
    uint32_t ma = u32_bin(c, b, S_And, ba, const_u32(c, 0x7fffffu));
    uint32_t mb = u32_bin(c, b, S_And, bb, const_u32(c, 0x7fffffu));
    uint32_t ea0 = cmp_u32(c, b, PCM_EQ, 0, ea, const_u32(c, 0), line);
    uint32_t eb0 = cmp_u32(c, b, PCM_EQ, 0, eb, const_u32(c, 0), line);
    uint32_t ea255 = cmp_u32(c, b, PCM_EQ, 0, ea, const_u32(c, 255), line);
    uint32_t eb255 = cmp_u32(c, b, PCM_EQ, 0, eb, const_u32(c, 255), line);
    uint32_t ma_nz = cmp_u32(c, b, PCM_NE, 0, ma, const_u32(c, 0), line);
    uint32_t mb_nz = cmp_u32(c, b, PCM_NE, 0, mb, const_u32(c, 0), line);
    uint32_t zero_a = bool_bin(c, b, S_LogicalAnd, ea0,
                               bool_not(c, b, ma_nz));
    uint32_t zero_b = bool_bin(c, b, S_LogicalAnd, eb0,
                               bool_not(c, b, mb_nz));
    uint32_t special = bool_bin(c, b, S_LogicalOr, ea255, eb255);
    uint32_t prodzero = bool_bin(c, b, S_LogicalOr, zero_a, zero_b);
    uint32_t usezero = bool_bin(c, b, S_LogicalAnd, prodzero,
                                bool_not(c, b, special));
    uint32_t sp = u32_bin(c, b, S_Xor,
                          u32_bin(c, b, S_ShrL, ba, const_u32(c, 31)),
                          u32_bin(c, b, S_ShrL, bb, const_u32(c, 31)));
    uint32_t zbits = u32_bin(c, b, S_Shl, sp, const_u32(c, 31));
    uint32_t bits = sel_u32(c, b, usezero, zbits, f32_to_bits(c, b, r));
    return bits_to_f32(c, b, bits);
}

uint32_t f64_directed_mul_soft(ctx_t *c, spv_blob_t *b, uint32_t fa,
                                      uint32_t fb, int mode, int line) {
    uint32_t r = f64_soft_fma(c, b, fa, fb, const_f64(c, 0), mode, line,
                              NULL, NULL);
    if (c->failed)
        return 0;
    uint32_t ba = f64_to_bits(c, b, fa);
    uint32_t bb = f64_to_bits(c, b, fb);
    uint32_t ea = u32_bin(c, b, S_And,
                          trunc_u64_u32(c, b,
                                        u64_bin(c, b, S_ShrL, ba,
                                                const_u64(c, 52))),
                          const_u32(c, 0x7ffu));
    uint32_t eb = u32_bin(c, b, S_And,
                          trunc_u64_u32(c, b,
                                        u64_bin(c, b, S_ShrL, bb,
                                                const_u64(c, 52))),
                          const_u32(c, 0x7ffu));
    uint32_t ma = u64_bin(c, b, S_And, ba,
                          const_u64(c, 0xfffffffffffffull));
    uint32_t mb = u64_bin(c, b, S_And, bb,
                          const_u64(c, 0xfffffffffffffull));
    uint32_t ea0 = cmp_u32(c, b, PCM_EQ, 0, ea, const_u32(c, 0), line);
    uint32_t eb0 = cmp_u32(c, b, PCM_EQ, 0, eb, const_u32(c, 0), line);
    uint32_t ea255 = cmp_u32(c, b, PCM_EQ, 0, ea, const_u32(c, 0x7ffu),
                             line);
    uint32_t eb255 = cmp_u32(c, b, PCM_EQ, 0, eb, const_u32(c, 0x7ffu),
                             line);
    uint32_t ma_nz = arith_cmp_u64(c, b, PCM_NE, ma, const_u64(c, 0), line);
    uint32_t mb_nz = arith_cmp_u64(c, b, PCM_NE, mb, const_u64(c, 0), line);
    uint32_t zero_a = bool_bin(c, b, S_LogicalAnd, ea0,
                               bool_not(c, b, ma_nz));
    uint32_t zero_b = bool_bin(c, b, S_LogicalAnd, eb0,
                               bool_not(c, b, mb_nz));
    uint32_t special = bool_bin(c, b, S_LogicalOr, ea255, eb255);
    uint32_t prodzero = bool_bin(c, b, S_LogicalOr, zero_a, zero_b);
    uint32_t usezero = bool_bin(c, b, S_LogicalAnd, prodzero,
                                bool_not(c, b, special));
    uint32_t sp = u32_bin(c, b, S_Xor,
                          trunc_u64_u32(c, b,
                                        u64_bin(c, b, S_ShrL, ba,
                                                const_u64(c, 63))),
                          trunc_u64_u32(c, b,
                                        u64_bin(c, b, S_ShrL, bb,
                                                const_u64(c, 63))));
    uint32_t zbits = u64_bin(c, b, S_Shl, zext_u32_u64(c, b, sp),
                             const_u64(c, 63));
    uint32_t bits = sel_u64(c, b, usezero, zbits, f64_to_bits(c, b, r));
    return bits_to_f64(c, b, bits);
}

// Round and pack a normalized significand+exponent.  total has 27
// significant bits (leading bit 26), extra is a sticky bit, and eVal is
// the signed unbiased exponent.
uint32_t f32_round_pack(ctx_t *c, spv_blob_t *b, uint32_t total,
                               uint32_t extra, uint32_t eVal, uint32_t sRes,
                               int mode, int line) {
    uint32_t extraNZ = cmp_u32(c, b, PCM_NE, 0, extra,
                               const_u32(c, 0), line);
    uint32_t sResNeg = cmp_u32(c, b, PCM_NE, 0, sRes,
                               const_u32(c, 0), line);
    uint32_t sResPos = bool_not(c, b, sResNeg);
    uint32_t signbit = u32_bin(c, b, S_Shl, sRes, const_u32(c, 31));

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
    return sel_u32(c, b, isNorm, bitsNormal, bitsSub);
}

uint32_t f64_round_pack(ctx_t *c, spv_blob_t *b, uint32_t total,
                               uint32_t extra, uint32_t eVal, uint32_t sRes,
                               int mode, int line) {
    uint32_t extraNZ = cmp_u32(c, b, PCM_NE, 0, extra,
                               const_u32(c, 0), line);
    uint32_t sResNeg = cmp_u32(c, b, PCM_NE, 0, sRes,
                               const_u32(c, 0), line);
    uint32_t sResPos = bool_not(c, b, sResNeg);
    uint32_t signbit = u64_bin(c, b, S_Shl, zext_u32_u64(c, b, sRes),
                               const_u64(c, 63));

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
                              const_u64(c, (uint64_t)1 << 53), line);
    uint32_t qq3 = sel_u64(c, b, carryN,
                           u64_bin(c, b, S_ShrL, qq2, const_u64(c, 1)), qq2);
    uint32_t eVal2 = u32_bin(c, b, S_IAdd, eVal,
                             sel_u32(c, b, carryN, const_u32(c, 1),
                                     const_u32(c, 0)));
    uint32_t be = u32_bin(c, b, S_IAdd, eVal2, const_u32(c, 1023));
    uint32_t ovf = cmp_u32(c, b, PCM_GE, 0, be, const_u32(c, 2047), line);
    uint32_t mbits = u64_bin(c, b, S_And, qq3,
                             const_u64(c, 0xfffffffffffffull));
    uint32_t bitsNorm = u64_bin(c, b, S_Or, signbit,
                                u64_bin(c, b, S_Or,
                                        u64_bin(c, b, S_Shl,
                                                zext_u32_u64(c, b, be),
                                                const_u64(c, 52)),
                                        mbits));
    uint32_t infBits = u64_bin(c, b, S_Or, signbit,
                               const_u64(c, 0x7ff0000000000000ull));
    uint32_t maxBits = u64_bin(c, b, S_Or, signbit,
                               const_u64(c, 0x7fefffffffffffffull));
    uint32_t ovfBits;
    if (mode == 0) {
        ovfBits = infBits;
    } else if (mode == 1) {
        ovfBits = maxBits;
    } else if (mode == 3) {
        ovfBits = sel_u64(c, b, sResNeg, maxBits,
                          const_u64(c, 0x7ff0000000000000ull));
    } else {
        ovfBits = sel_u64(c, b, sResNeg,
                          const_u64(c, 0xfff0000000000000ull),
                          const_u64(c, 0x7fefffffffffffffull));
    }
    uint32_t bitsNormal = sel_u64(c, b, ovf, ovfBits, bitsNorm);

    uint32_t sU = u32_bin(c, b, S_ISub, const_u32(c, (uint32_t)-1019),
                          eVal);
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
                               const_u64(c, (uint64_t)1 << 52), line);
    uint32_t subBits = u64_bin(c, b, S_Or, signbit,
                               u64_bin(c, b, S_And, qs2,
                                       const_u64(c, 0xfffffffffffffull)));
    uint32_t bitsSub = sel_u64(c, b, minnorm,
                               u64_bin(c, b, S_Or, signbit,
                                       const_u64(c, 1ull << 52)),
                               subBits);
    uint32_t isNorm = cmp_u32(c, b, PCM_GE, 1, eVal,
                              const_u32(c, (uint32_t)-1022), line);
    return sel_u64(c, b, isNorm, bitsNormal, bitsSub);
}

// Round and pack an f16 pattern from a 14-bit normalized total
// (leading bit 13); mirrors f32_round_pack with the half-precision
// exponent range.
