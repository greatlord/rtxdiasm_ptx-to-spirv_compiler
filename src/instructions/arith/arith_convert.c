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

uint32_t f16_round_pack(ctx_t *c, spv_blob_t *b, uint32_t total,
                               uint32_t extra, uint32_t eVal, uint32_t sRes,
                               int mode, int line) {
    uint32_t extraNZ = cmp_u32(c, b, PCM_NE, 0, extra,
                               const_u32(c, 0), line);
    uint32_t sResNeg = cmp_u32(c, b, PCM_NE, 0, sRes,
                               const_u32(c, 0), line);
    uint32_t sResPos = bool_not(c, b, sResNeg);
    uint32_t signbit = u32_bin(c, b, S_Shl, sRes, const_u32(c, 15));

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
                              const_u64(c, (uint64_t)1 << 11), line);
    uint32_t qq3 = sel_u64(c, b, carryN,
                           u64_bin(c, b, S_ShrL, qq2, const_u64(c, 1)), qq2);
    uint32_t eVal2 = u32_bin(c, b, S_IAdd, eVal,
                             sel_u32(c, b, carryN, const_u32(c, 1),
                                     const_u32(c, 0)));
    uint32_t be = u32_bin(c, b, S_IAdd, eVal2, const_u32(c, 15));
    uint32_t ovf = cmp_u32(c, b, PCM_GE, 0, be, const_u32(c, 31), line);
    uint32_t mbits = u32_bin(c, b, S_And, trunc_u64_u32(c, b, qq3),
                             const_u32(c, 0x3ffu));
    uint32_t bitsNorm = u32_bin(c, b, S_Or, signbit,
                                u32_bin(c, b, S_Or,
                                        u32_bin(c, b, S_Shl, be,
                                                const_u32(c, 10)),
                                        mbits));
    uint32_t infBits = u32_bin(c, b, S_Or, signbit,
                               const_u32(c, 0x7c00u));
    uint32_t maxBits = u32_bin(c, b, S_Or, signbit,
                               const_u32(c, 0x7bffu));
    uint32_t ovfBits;
    if (mode == 0) {
        ovfBits = infBits;
    } else if (mode == 1) {
        ovfBits = maxBits;
    } else if (mode == 3) {
        ovfBits = sel_u32(c, b, sResNeg, maxBits,
                          const_u32(c, 0x7c00u));
    } else {
        ovfBits = sel_u32(c, b, sResNeg, const_u32(c, 0xfc00u),
                          const_u32(c, 0x7bffu));
    }
    uint32_t bitsNormal = sel_u32(c, b, ovf, ovfBits, bitsNorm);

    uint32_t sU = u32_bin(c, b, S_ISub, const_u32(c, (uint32_t)-11), eVal);
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
                               const_u64(c, (uint64_t)1 << 10), line);
    uint32_t subBits = u32_bin(c, b, S_Or, signbit,
                               u32_bin(c, b, S_And,
                                       trunc_u64_u32(c, b, qs2),
                                       const_u32(c, 0x3ffu)));
    uint32_t bitsSub = sel_u32(c, b, minnorm,
                               u32_bin(c, b, S_Or, signbit,
                                       const_u32(c, 1u << 10)),
                               subBits);
    uint32_t isNorm = cmp_u32(c, b, PCM_GE, 1, eVal,
                              const_u32(c, (uint32_t)-14), line);
    return sel_u32(c, b, isNorm, bitsNormal, bitsSub);
}

// Exact directed integer -> f32/f64/f16.  The magnitude is normalized
// to the target precision with a sticky bit and handed to the matching
// round_pack helper.
uint32_t int_to_float_dir(ctx_t *c, spv_blob_t *b, uint32_t v,
                                 int sign, int wsrc, ptx_ty_t ty,
                                 int mode, int line) {
    uint32_t v64;
    if (wsrc == 64) {
        v64 = v;
    } else {
        uint32_t u = v;
        if (wsrc == 8 || wsrc == 16) {
            uint32_t m = u32_bin(c, b, S_And, u,
                                 const_u32(c, wsrc == 8 ? 0xffu : 0xffffu));
            u = m;
            if (sign) {
                uint32_t t1 = u32_bin(c, b, S_Shl, u,
                                      const_u32(c, (uint32_t)(32 - wsrc)));
                u = u32_bin(c, b, S_ShrA, t1,
                            const_u32(c, (uint32_t)(32 - wsrc)));
            }
        }
        v64 = nid(c);
        put(c, b, (4u << 16) | (sign ? S_SConvert : S_UConvert));
        put(c, b, c->t_u64);
        put(c, b, v64);
        put(c, b, u);
    }
    uint32_t neg = const_u32(c, 0);
    uint32_t absv = v64;
    if (sign) {
        uint32_t hi = u32_bin(c, b, S_And,
                              trunc_u64_u32(c, b,
                                            u64_bin(c, b, S_ShrL, v64,
                                                    const_u64(c, 63))),
                              const_u32(c, 1));
        neg = hi;
        uint32_t zero = const_u64(c, 0);
        uint32_t nv = u64_bin(c, b, S_ISub, zero, v64);
        absv = sel_u64(c, b, cmp_u32(c, b, PCM_NE, 0, hi,
                                     const_u32(c, 0), line), nv, v64);
    }
    if (ty == PTY_F16) {
        uint32_t zero = const_u32(c, 0);
        uint32_t nz = arith_cmp_u64(c, b, PCM_NE, absv, const_u64(c, 0), line);
        uint32_t k = u32_bin(c, b, S_ISub, const_u32(c, 63),
                             clz64_val(c, b, absv, line));
        uint32_t big = cmp_u32(c, b, PCM_GT, 0, k, const_u32(c, 13), line);
        uint32_t shR = u32_bin(c, b, S_And,
                               u32_bin(c, b, S_ISub, k, const_u32(c, 13)),
                               const_u32(c, 63));
        uint32_t totalR = u64_bin(c, b, S_ShrL, absv,
                                  zext_u32_u64(c, b, shR));
        uint32_t dmask = u64_bin(c, b, S_ISub,
                                 u64_bin(c, b, S_Shl, const_u64(c, 1),
                                         zext_u32_u64(c, b, shR)),
                                 const_u64(c, 1));
        uint32_t drop = arith_cmp_u64(c, b, PCM_NE,
                                u64_bin(c, b, S_And, absv, dmask),
                                const_u64(c, 0), line);
        uint32_t shL = u32_bin(c, b, S_And,
                               u32_bin(c, b, S_ISub, const_u32(c, 13), k),
                               const_u32(c, 63));
        uint32_t totalL = u64_bin(c, b, S_Shl, absv,
                                  zext_u32_u64(c, b, shL));
        uint32_t total = sel_u64(c, b, big, totalR, totalL);
        uint32_t extra = sel_u32(c, b,
                                 bool_bin(c, b, S_LogicalAnd, big, drop),
                                 const_u32(c, 1), const_u32(c, 0));
        uint32_t r = f16_round_pack(c, b, total, extra, k, neg, mode, line);
        return sel_u32(c, b, nz, r, zero);
    }
    if (ty == PTY_F32) {
        uint32_t nz = arith_cmp_u64(c, b, PCM_NE, absv, const_u64(c, 0), line);
        uint32_t k = u32_bin(c, b, S_ISub, const_u32(c, 63),
                             clz64_val(c, b, absv, line));
        uint32_t big = cmp_u32(c, b, PCM_GT, 0, k, const_u32(c, 26), line);
        uint32_t shR = u32_bin(c, b, S_And,
                               u32_bin(c, b, S_ISub, k, const_u32(c, 26)),
                               const_u32(c, 63));
        uint32_t totalR = u64_bin(c, b, S_ShrL, absv,
                                  zext_u32_u64(c, b, shR));
        uint32_t dmask = u64_bin(c, b, S_ISub,
                                 u64_bin(c, b, S_Shl, const_u64(c, 1),
                                         zext_u32_u64(c, b, shR)),
                                 const_u64(c, 1));
        uint32_t drop = arith_cmp_u64(c, b, PCM_NE,
                                u64_bin(c, b, S_And, absv, dmask),
                                const_u64(c, 0), line);
        uint32_t shL = u32_bin(c, b, S_And,
                               u32_bin(c, b, S_ISub, const_u32(c, 26), k),
                               const_u32(c, 63));
        uint32_t totalL = u64_bin(c, b, S_Shl, absv,
                                  zext_u32_u64(c, b, shL));
        uint32_t total = sel_u64(c, b, big, totalR, totalL);
        uint32_t extra = sel_u32(c, b,
                                 bool_bin(c, b, S_LogicalAnd, big, drop),
                                 const_u32(c, 1), const_u32(c, 0));
        uint32_t bits = f32_round_pack(c, b, total, extra, k, neg, mode,
                                       line);
        uint32_t rb = sel_u32(c, b, nz, bits, const_u32(c, 0));
        return bits_to_f32(c, b, rb);
    }
    uint32_t nz = arith_cmp_u64(c, b, PCM_NE, absv, const_u64(c, 0), line);
    uint32_t k = u32_bin(c, b, S_ISub, const_u32(c, 63),
                         clz64_val(c, b, absv, line));
    uint32_t big = cmp_u32(c, b, PCM_GT, 0, k, const_u32(c, 55), line);
    uint32_t shR = u32_bin(c, b, S_And,
                           u32_bin(c, b, S_ISub, k, const_u32(c, 55)),
                           const_u32(c, 63));
    uint32_t totalR = u64_bin(c, b, S_ShrL, absv,
                              zext_u32_u64(c, b, shR));
    uint32_t dmask = u64_bin(c, b, S_ISub,
                             u64_bin(c, b, S_Shl, const_u64(c, 1),
                                     zext_u32_u64(c, b, shR)),
                             const_u64(c, 1));
    uint32_t drop = arith_cmp_u64(c, b, PCM_NE,
                            u64_bin(c, b, S_And, absv, dmask),
                            const_u64(c, 0), line);
    uint32_t shL = u32_bin(c, b, S_And,
                           u32_bin(c, b, S_ISub, const_u32(c, 55), k),
                           const_u32(c, 63));
    uint32_t totalL = u64_bin(c, b, S_Shl, absv,
                              zext_u32_u64(c, b, shL));
    uint32_t total = sel_u64(c, b, big, totalR, totalL);
    uint32_t extra = sel_u32(c, b,
                             bool_bin(c, b, S_LogicalAnd, big, drop),
                             const_u32(c, 1), const_u32(c, 0));
    uint32_t bits = f64_round_pack(c, b, total, extra, k, neg, mode, line);
    uint32_t rb = sel_u64(c, b, nz, bits, const_u64(c, 0));
    return bits_to_f64(c, b, rb);
}

// Directed division by exact integer long division.  Both significands
// are normalized to a fixed leading-bit position, giving a fixed number
// of quotient bits independent of the exponent (and of the driver's
// division accuracy).
uint32_t f32_directed_div_exact(ctx_t *c, spv_blob_t *b,
                                       uint32_t fa, uint32_t fb, int mode,
                                       int line) {
    uint32_t ba = f32_to_bits(c, b, fa);
    uint32_t bb = f32_to_bits(c, b, fb);
    uint32_t sa = u32_bin(c, b, S_ShrL, ba, const_u32(c, 31));
    uint32_t sb = u32_bin(c, b, S_ShrL, bb, const_u32(c, 31));
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
    uint32_t nan_a = bool_bin(c, b, S_LogicalAnd, ea255, ma_nz);
    uint32_t nan_b = bool_bin(c, b, S_LogicalAnd, eb255, mb_nz);
    uint32_t inf_a = bool_bin(c, b, S_LogicalAnd, ea255,
                              bool_not(c, b, ma_nz));
    uint32_t inf_b = bool_bin(c, b, S_LogicalAnd, eb255,
                              bool_not(c, b, mb_nz));
    uint32_t zero_a = bool_bin(c, b, S_LogicalAnd, ea0,
                               bool_not(c, b, ma_nz));
    uint32_t zero_b = bool_bin(c, b, S_LogicalAnd, eb0,
                               bool_not(c, b, mb_nz));

    uint32_t aimp = u32_bin(c, b, S_Or, ma, const_u32(c, 0x800000u));
    uint32_t bimp = u32_bin(c, b, S_Or, mb, const_u32(c, 0x800000u));
    uint32_t A = sel_u32(c, b, ea0, ma, aimp);
    uint32_t B = sel_u32(c, b, eb0, mb, bimp);
    uint32_t exA = sel_u32(c, b, ea0, const_u32(c, (uint32_t)-149),
                           u32_bin(c, b, S_ISub, ea, const_u32(c, 150)));
    uint32_t exB = sel_u32(c, b, eb0, const_u32(c, (uint32_t)-149),
                           u32_bin(c, b, S_ISub, eb, const_u32(c, 150)));
    uint32_t sP = u32_bin(c, b, S_Xor, sa, sb);

    uint32_t kA = u32_bin(c, b, S_ISub, const_u32(c, 31),
                          clz32(c, b, A, line));
    uint32_t kB = u32_bin(c, b, S_ISub, const_u32(c, 31),
                          clz32(c, b, B, line));
    uint32_t shA = u32_bin(c, b, S_And,
                           u32_bin(c, b, S_ISub, const_u32(c, 23), kA),
                           const_u32(c, 31));
    uint32_t shB = u32_bin(c, b, S_And,
                           u32_bin(c, b, S_ISub, const_u32(c, 23), kB),
                           const_u32(c, 31));
    uint32_t An = u32_bin(c, b, S_Shl, A, shA);
    uint32_t Bn = u32_bin(c, b, S_Shl, B, shB);
    uint32_t bump = cmp_u32(c, b, PCM_GE, 0, An, Bn, line);
    Bn = sel_u32(c, b, bump, u32_bin(c, b, S_Shl, Bn, const_u32(c, 1)),
                 Bn);

    uint32_t rem = An, Q = const_u64(c, 0);
    for (int i = 0; i < 28; i++) {
        rem = u32_bin(c, b, S_Shl, rem, const_u32(c, 1));
        uint32_t ge = cmp_u32(c, b, PCM_GE, 0, rem, Bn, line);
        rem = sel_u32(c, b, ge, u32_bin(c, b, S_ISub, rem, Bn), rem);
        Q = u64_bin(c, b, S_Or,
                    u64_bin(c, b, S_Shl, Q, const_u64(c, 1)),
                    zext_u32_u64(c, b,
                                 sel_u32(c, b, ge, const_u32(c, 1),
                                         const_u32(c, 0))));
    }

    uint32_t remNZ = cmp_u32(c, b, PCM_NE, 0, rem, const_u32(c, 0), line);
    uint32_t qNZ = arith_cmp_u64(c, b, PCM_NE, Q, const_u64(c, 0), line);
    uint32_t k = sel_u32(c, b, qNZ,
                         u32_bin(c, b, S_ISub, const_u32(c, 63),
                                 clz64_val(c, b, Q, line)),
                         const_u32(c, 0));
    uint32_t eBase = u32_bin(c, b, S_IAdd,
                             u32_bin(c, b, S_ISub,
                                     u32_bin(c, b, S_IAdd,
                                             u32_bin(c, b, S_ISub, exA, exB),
                                             u32_bin(c, b, S_ISub, kA, kB)),
                                     const_u32(c, 28)),
                             sel_u32(c, b, bump, const_u32(c, 1),
                                     const_u32(c, 0)));
    uint32_t eVal = u32_bin(c, b, S_IAdd, eBase, k);
    uint32_t big = cmp_u32(c, b, PCM_GT, 0, k, const_u32(c, 26), line);
    uint32_t shR = u32_bin(c, b, S_And,
                           u32_bin(c, b, S_ISub, k, const_u32(c, 26)),
                           const_u32(c, 63));
    uint32_t totalR = u64_bin(c, b, S_ShrL, Q, zext_u32_u64(c, b, shR));
    uint32_t dmask = u64_bin(c, b, S_ISub,
                             u64_bin(c, b, S_Shl, const_u64(c, 1),
                                     zext_u32_u64(c, b, shR)),
                             const_u64(c, 1));
    uint32_t dropped = arith_cmp_u64(c, b, PCM_NE,
                               u64_bin(c, b, S_And, Q, dmask),
                               const_u64(c, 0), line);
    uint32_t shL = u32_bin(c, b, S_And,
                           u32_bin(c, b, S_ISub, const_u32(c, 26), k),
                           const_u32(c, 63));
    uint32_t totalL = u64_bin(c, b, S_Shl, Q, zext_u32_u64(c, b, shL));
    uint32_t total = sel_u64(c, b, big, totalR, totalL);
    uint32_t extra = u32_bin(c, b, S_Or,
                             sel_u32(c, b, remNZ, const_u32(c, 1),
                                     const_u32(c, 0)),
                             sel_u32(c, b,
                                     bool_bin(c, b, S_LogicalAnd, big,
                                              dropped),
                                     const_u32(c, 1), const_u32(c, 0)));

    uint32_t bits = f32_round_pack(c, b, total, extra, eVal, sP, mode, line);
    uint32_t anyNan = bool_bin(c, b, S_LogicalOr, nan_a, nan_b);
    uint32_t invalid = bool_bin(c, b, S_LogicalOr,
                                bool_bin(c, b, S_LogicalAnd, zero_a, zero_b),
                                bool_bin(c, b, S_LogicalAnd, inf_a, inf_b));
    uint32_t zeroRes = bool_bin(c, b, S_LogicalOr, zero_a, inf_b);
    uint32_t infRes = bool_bin(c, b, S_LogicalOr, inf_a, zero_b);
    uint32_t sbit = u32_bin(c, b, S_Shl, sP, const_u32(c, 31));
    bits = sel_u32(c, b, zeroRes, sbit, bits);
    bits = sel_u32(c, b, infRes,
                   u32_bin(c, b, S_Or, sbit,
                           const_u32(c, 0x7f800000u)), bits);
    bits = sel_u32(c, b, invalid, const_u32(c, 0x7fc00000u), bits);
    bits = sel_u32(c, b, anyNan, const_u32(c, 0x7fc00000u), bits);
    return bits_to_f32(c, b, bits);
}

uint32_t f64_directed_div_exact(ctx_t *c, spv_blob_t *b,
                                       uint32_t fa, uint32_t fb, int mode,
                                       int line) {
    uint32_t ba = f64_to_bits(c, b, fa);
    uint32_t bb = f64_to_bits(c, b, fb);
    uint32_t sa = trunc_u64_u32(c, b, u64_bin(c, b, S_ShrL, ba,
                                              const_u64(c, 63)));
    uint32_t sb = trunc_u64_u32(c, b, u64_bin(c, b, S_ShrL, bb,
                                              const_u64(c, 63)));
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
    uint32_t nan_a = bool_bin(c, b, S_LogicalAnd, ea255, ma_nz);
    uint32_t nan_b = bool_bin(c, b, S_LogicalAnd, eb255, mb_nz);
    uint32_t inf_a = bool_bin(c, b, S_LogicalAnd, ea255,
                              bool_not(c, b, ma_nz));
    uint32_t inf_b = bool_bin(c, b, S_LogicalAnd, eb255,
                              bool_not(c, b, mb_nz));
    uint32_t zero_a = bool_bin(c, b, S_LogicalAnd, ea0,
                               bool_not(c, b, ma_nz));
    uint32_t zero_b = bool_bin(c, b, S_LogicalAnd, eb0,
                               bool_not(c, b, mb_nz));

    uint32_t aimp = u64_bin(c, b, S_Or, ma, const_u64(c, 1ull << 52));
    uint32_t bimp = u64_bin(c, b, S_Or, mb, const_u64(c, 1ull << 52));
    uint32_t A = sel_u64(c, b, ea0, ma, aimp);
    uint32_t B = sel_u64(c, b, eb0, mb, bimp);
    uint32_t exA = sel_u32(c, b, ea0, const_u32(c, (uint32_t)-1074),
                           u32_bin(c, b, S_ISub, ea, const_u32(c, 1075)));
    uint32_t exB = sel_u32(c, b, eb0, const_u32(c, (uint32_t)-1074),
                           u32_bin(c, b, S_ISub, eb, const_u32(c, 1075)));
    uint32_t sP = u32_bin(c, b, S_Xor, sa, sb);

    uint32_t kA = u32_bin(c, b, S_ISub, const_u32(c, 63),
                          clz64_val(c, b, A, line));
    uint32_t kB = u32_bin(c, b, S_ISub, const_u32(c, 63),
                          clz64_val(c, b, B, line));
    uint32_t shA = u32_bin(c, b, S_And,
                           u32_bin(c, b, S_ISub, const_u32(c, 52), kA),
                           const_u32(c, 63));
    uint32_t shB = u32_bin(c, b, S_And,
                           u32_bin(c, b, S_ISub, const_u32(c, 52), kB),
                           const_u32(c, 63));
    uint32_t An = u64_bin(c, b, S_Shl, A, zext_u32_u64(c, b, shA));
    uint32_t Bn = u64_bin(c, b, S_Shl, B, zext_u32_u64(c, b, shB));
    uint32_t bump = arith_cmp_u64(c, b, PCM_GE, An, Bn, line);
    Bn = sel_u64(c, b, bump, u64_bin(c, b, S_Shl, Bn, const_u64(c, 1)),
                 Bn);

    uint32_t rem = An, Q = const_u64(c, 0);
    for (int i = 0; i < 57; i++) {
        rem = u64_bin(c, b, S_Shl, rem, const_u64(c, 1));
        uint32_t ge = arith_cmp_u64(c, b, PCM_GE, rem, Bn, line);
        rem = sel_u64(c, b, ge, u64_bin(c, b, S_ISub, rem, Bn), rem);
        Q = u64_bin(c, b, S_Or,
                    u64_bin(c, b, S_Shl, Q, const_u64(c, 1)),
                    zext_u32_u64(c, b,
                                 sel_u32(c, b, ge, const_u32(c, 1),
                                         const_u32(c, 0))));
    }

    uint32_t remNZ = arith_cmp_u64(c, b, PCM_NE, rem, const_u64(c, 0), line);
    uint32_t qNZ = arith_cmp_u64(c, b, PCM_NE, Q, const_u64(c, 0), line);
    uint32_t k = sel_u32(c, b, qNZ,
                         u32_bin(c, b, S_ISub, const_u32(c, 63),
                                 clz64_val(c, b, Q, line)),
                         const_u32(c, 0));
    uint32_t eBase = u32_bin(c, b, S_IAdd,
                             u32_bin(c, b, S_ISub,
                                     u32_bin(c, b, S_IAdd,
                                             u32_bin(c, b, S_ISub, exA, exB),
                                             u32_bin(c, b, S_ISub, kA, kB)),
                                     const_u32(c, 57)),
                             sel_u32(c, b, bump, const_u32(c, 1),
                                     const_u32(c, 0)));
    uint32_t eVal = u32_bin(c, b, S_IAdd, eBase, k);
    uint32_t big = cmp_u32(c, b, PCM_GT, 0, k, const_u32(c, 55), line);
    uint32_t shR = u32_bin(c, b, S_And,
                           u32_bin(c, b, S_ISub, k, const_u32(c, 55)),
                           const_u32(c, 63));
    uint32_t totalR = u64_bin(c, b, S_ShrL, Q, zext_u32_u64(c, b, shR));
    uint32_t dmask = u64_bin(c, b, S_ISub,
                             u64_bin(c, b, S_Shl, const_u64(c, 1),
                                     zext_u32_u64(c, b, shR)),
                             const_u64(c, 1));
    uint32_t dropped = arith_cmp_u64(c, b, PCM_NE,
                               u64_bin(c, b, S_And, Q, dmask),
                               const_u64(c, 0), line);
    uint32_t shL = u32_bin(c, b, S_And,
                           u32_bin(c, b, S_ISub, const_u32(c, 55), k),
                           const_u32(c, 63));
    uint32_t totalL = u64_bin(c, b, S_Shl, Q, zext_u32_u64(c, b, shL));
    uint32_t total = sel_u64(c, b, big, totalR, totalL);
    uint32_t extra = u32_bin(c, b, S_Or,
                             sel_u32(c, b, remNZ, const_u32(c, 1),
                                     const_u32(c, 0)),
                             sel_u32(c, b,
                                     bool_bin(c, b, S_LogicalAnd, big,
                                              dropped),
                                     const_u32(c, 1), const_u32(c, 0)));

    uint32_t bits = f64_round_pack(c, b, total, extra, eVal, sP, mode, line);
    uint32_t anyNan = bool_bin(c, b, S_LogicalOr, nan_a, nan_b);
    uint32_t invalid = bool_bin(c, b, S_LogicalOr,
                                bool_bin(c, b, S_LogicalAnd, zero_a, zero_b),
                                bool_bin(c, b, S_LogicalAnd, inf_a, inf_b));
    uint32_t zeroRes = bool_bin(c, b, S_LogicalOr, zero_a, inf_b);
    uint32_t infRes = bool_bin(c, b, S_LogicalOr, inf_a, zero_b);
    uint32_t sbit = u64_bin(c, b, S_Shl, zext_u32_u64(c, b, sP),
                            const_u64(c, 63));
    bits = sel_u64(c, b, zeroRes, sbit, bits);
    bits = sel_u64(c, b, infRes,
                   u64_bin(c, b, S_Or, sbit,
                           const_u64(c, 0x7ff0000000000000ull)), bits);
    bits = sel_u64(c, b, invalid,
                   const_u64(c, 0x7ff8000000000000ull), bits);
    bits = sel_u64(c, b, anyNan,
                   const_u64(c, 0x7ff8000000000000ull), bits);
    return bits_to_f64(c, b, bits);
}

// Guaranteed-fused f32 FMA (SPV_KHR_fma). Used for the exact residual
// when the device advertises VK_KHR_shader_fma.
uint32_t f32_fma3(ctx_t *c, spv_blob_t *b, uint32_t x, uint32_t y,
                         uint32_t z) {
    uint32_t id = nid(c);
    put(c, b, (6u << 16) | S_FmaKHR);
    put(c, b, c->t_f32);
    put(c, b, id);
    put(c, b, x);
    put(c, b, y);
    put(c, b, z);
    return id;
}

// GLSL.std.450 Fma is specified as a fused multiply-add and is reliable
// on drivers where OpFmaKHR (the VK_KHR_shader_fma path) is not.
uint32_t f32_glsl_fma3(ctx_t *c, spv_blob_t *b, uint32_t x,
                              uint32_t y, uint32_t z) {
    uint32_t id = nid(c);
    put(c, b, (8u << 16) | S_ExtInst);
    put(c, b, c->t_f32);
    put(c, b, id);
    put(c, b, c->glsl450);
    put(c, b, 50); // GLSL.std.450 Fma
    put(c, b, x);
    put(c, b, y);
    put(c, b, z);
    return id;
}

uint32_t f32_neg(ctx_t *c, spv_blob_t *b, uint32_t f) {
    uint32_t fb = nid(c), nfb = nid(c), out = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_u32);
    put(c, b, fb);
    put(c, b, f);
    emit_arith(c, b, S_Xor, c->t_u32, nfb, fb,
               const_u32(c, 0x80000000u));
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_f32);
    put(c, b, out);
    put(c, b, nfb);
    return out;
}

void ensure_f16_types(ctx_t *c) {
    if (c->t_f16)
        return;
    c->t_f16 = nid(c);
    put(c, &c->types, (3u << 16) | S_TypeFloat);
    put(c, &c->types, c->t_f16);
    put(c, &c->types, 16);
    c->t_v2f16 = nid(c);
    put(c, &c->types, (4u << 16) | S_TypeVector);
    put(c, &c->types, c->t_v2f16);
    put(c, &c->types, c->t_f16);
    put(c, &c->types, 2);
}

// Emit one fused multiply-add into an existing result id. Used by the
// MMA/WMMA accumulation chains so each product+add is one instruction.
void emit_fma_into(ctx_t *c, spv_blob_t *b, uint32_t ty, uint32_t dst,
                   uint32_t x, uint32_t y, uint32_t z) {
    if (c->caps.soft_fma) {
        uint32_t pr = nid(c);
        emit_arith(c, b, S_FMul, ty, pr, x, y);
        emit_arith(c, b, S_FAdd, ty, dst, pr, z);
        return;
    }
    if (ty == c->t_f32 && c->use_fma) {
        put(c, b, (6u << 16) | S_FmaKHR);
        put(c, b, c->t_f32);
        put(c, b, dst);
        put(c, b, x);
        put(c, b, y);
        put(c, b, z);
        return;
    }
    if (ty == c->t_f32 || ty == c->t_f64) {
        put(c, b, (8u << 16) | S_ExtInst);
        put(c, b, ty);
        put(c, b, dst);
        put(c, b, c->glsl450);
        put(c, b, 50); // GLSL.std.450 Fma
        put(c, b, x);
        put(c, b, y);
        put(c, b, z);
        return;
    }
    uint32_t pr = nid(c);
    emit_arith(c, b, S_FMul, ty, pr, x, y);
    emit_arith(c, b, S_FAdd, ty, dst, pr, z);
}

// Directed f32 add/sub via Fast2Sum with the guaranteed-fused opcode:
// s = RN(a+b); t = s-a (exact); err = b-t (exact); step s one ulp.

uint32_t f64_neg(ctx_t *c, spv_blob_t *b, uint32_t f) {
    uint32_t fb = nid(c), nfb = nid(c), out = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_u64);
    put(c, b, fb);
    put(c, b, f);
    emit_arith(c, b, S_Xor, c->t_u64, nfb, fb,
               const_u64(c, 0x8000000000000000ull));
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_f64);
    put(c, b, out);
    put(c, b, nfb);
    return out;
}

void f32x2_split(ctx_t *c, spv_blob_t *b, uint32_t u, uint32_t *lo,
                        uint32_t *hi) {
    uint32_t l32 = nid(c), h32 = nid(c), sh = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u32);
    put(c, b, l32);
    put(c, b, u);
    emit_arith(c, b, S_ShrL, c->t_u64, sh, u, const_u64(c, 32));
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u32);
    put(c, b, h32);
    put(c, b, sh);
    *lo = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_f32);
    put(c, b, *lo);
    put(c, b, l32);
    *hi = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_f32);
    put(c, b, *hi);
    put(c, b, h32);
}

uint32_t f32x2_pack(ctx_t *c, spv_blob_t *b, uint32_t lo, uint32_t hi) {
    uint32_t l32 = nid(c), h32 = nid(c), l64 = nid(c), h64 = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_u32);
    put(c, b, l32);
    put(c, b, lo);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_u32);
    put(c, b, h32);
    put(c, b, hi);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u64);
    put(c, b, l64);
    put(c, b, l32);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u64);
    put(c, b, h64);
    put(c, b, h32);
    uint32_t sh = nid(c), out = nid(c);
    emit_arith(c, b, S_Shl, c->t_u64, sh, h64, const_u64(c, 32));
    emit_arith(c, b, S_Or, c->t_u64, out, sh, l64);
    return out;
}

uint32_t f32_addsub_lane(ctx_t *c, spv_blob_t *b, int is_add,
                                uint32_t fa, uint32_t fb, int frnd,
                                int ftz, int sat, int line) {
    uint32_t id;
    if (ftz) {
        fa = flush_f32_denorm(c, b, fa, line);
        if (c->failed)
            return 0;
        fb = flush_f32_denorm(c, b, fb, line);
        if (c->failed)
            return 0;
    }
    if (frnd) {
        id = f32_soft_addsub(c, b, fa, fb, !is_add, frnd, line);
        if (c->failed)
            return 0;
    } else {
        id = nid(c);
        emit_arith(c, b, is_add ? S_FAdd : S_FSub, c->t_f32, id, fa, fb);
    }
    if (ftz) {
        id = flush_f32_denorm(c, b, id, line);
        if (c->failed)
            return 0;
    }
    if (sat) {
        id = sat_f32(c, b, id, line);
        if (c->failed)
            return 0;
    }
    return id;
}

uint32_t f32_mul_lane(ctx_t *c, spv_blob_t *b, uint32_t fa,
                             uint32_t fb, int frnd, int ftz, int sat,
                             int line) {
    uint32_t id;
    if (ftz) {
        fa = flush_f32_denorm(c, b, fa, line);
        if (c->failed)
            return 0;
        fb = flush_f32_denorm(c, b, fb, line);
        if (c->failed)
            return 0;
    }
    if (frnd) {
        id = f32_directed_mul_soft(c, b, fa, fb, frnd, line);
        if (c->failed)
            return 0;
    } else {
        id = nid(c);
        emit_arith(c, b, S_FMul, c->t_f32, id, fa, fb);
    }
    if (ftz) {
        id = flush_f32_denorm(c, b, id, line);
        if (c->failed)
            return 0;
    }
    if (sat) {
        id = sat_f32(c, b, id, line);
        if (c->failed)
            return 0;
    }
    return id;
}

uint32_t f32_fma_lane(ctx_t *c, spv_blob_t *b, uint32_t fa,
                             uint32_t fb, uint32_t fc, int ftz, int sat,
                             int line) {
    if (ftz) {
        fa = flush_f32_denorm(c, b, fa, line);
        fb = flush_f32_denorm(c, b, fb, line);
        fc = flush_f32_denorm(c, b, fc, line);
        if (c->failed)
            return 0;
    }
    uint32_t fr;
    if (c->caps.soft_fma) {
        fr = f32_soft_fma(c, b, fa, fb, fc, 0, line, NULL, NULL);
        if (c->failed)
            return 0;
    } else if (c->use_fma)
        fr = f32_fma3(c, b, fa, fb, fc);
    else
        fr = f32_glsl_fma3(c, b, fa, fb, fc);
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
    return fr;
}

// Directed-rounding f32 add/sub using an *integer* exact comparison:
// the driver may reassociate any floating-point error-free transform
// (RADV collapses (a+b)-a to b even through memory), so the sign of
// (exact(a+b) - RN(a+b)) is computed on aligned 24-bit significands.
// Exponent gaps > 26 collapse to the larger operand; the small operand
// then only decides the residual sign.
// Bit-field extract/insert on a u64 view (width 32 or 64); the caller
// converts the result back for 32-bit forms.
uint32_t bfe_value(ctx_t *c, spv_blob_t *b, uint32_t a64,
                          uint32_t bpos, uint32_t clen, int w64,
                          int sign, int line) {
    uint32_t W = w64 ? 64u : 32u;
    uint64_t all = w64 ? 0xffffffffffffffffull : 0xffffffffull;
    uint32_t pm = nid(c), lm = nid(c);
    emit_arith(c, b, S_And, c->t_u32, pm, bpos, const_u32(c, 0xffu));
    emit_arith(c, b, S_And, c->t_u32, lm, clen, const_u32(c, 0xffu));
    uint32_t pm64 = nid(c), lm64 = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u64);
    put(c, b, pm64);
    put(c, b, pm);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u64);
    put(c, b, lm64);
    put(c, b, lm);
    uint32_t Wc = const_u64(c, W);
    uint32_t pm_ge = do_cmp(c, b, PCM_GE, PTY_U64, 0, pm64, Wc, line);
    uint32_t lm_zero = do_cmp(c, b, PCM_EQ, PTY_U64, 0, lm64,
                              const_u64(c, 0), line);
    // sbit = a[min(pos+len-1, msb)]
    uint32_t idx = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, idx, pm64, lm64);
    uint32_t idx1 = nid(c);
    emit_arith(c, b, S_ISub, c->t_u64, idx1, idx, const_u64(c, 1));
    uint32_t idx_ge = do_cmp(c, b, PCM_GE, PTY_U64, 0, idx1, Wc, line);
    uint32_t idxc = sel_u64(c, b, idx_ge, const_u64(c, W - 1), idx1);
    uint32_t sh = nid(c), bit = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, sh, a64, idxc);
    emit_arith(c, b, S_And, c->t_u64, bit, sh, const_u64(c, 1));
    uint32_t sbit = bit;
    if (!sign) {
        sbit = const_u64(c, 0);
    } else {
        uint32_t nz = nid(c);
        put(c, b, (4u << 16) | S_LogicalNot);
        put(c, b, c->t_bool);
        put(c, b, nz);
        put(c, b, lm_zero);
        sbit = sel_u64(c, b, nz, bit, const_u64(c, 0));
    }
    // eff = min(len, max(0, W - pos))
    uint32_t avail = nid(c);
    emit_arith(c, b, S_ISub, c->t_u64, avail, Wc, pm64);
    avail = sel_u64(c, b, pm_ge, const_u64(c, 0), avail);
    uint32_t eff = nid(c);
    uint32_t lm_lt = do_cmp(c, b, PCM_LT, PTY_U64, 0, lm64, avail, line);
    eff = sel_u64(c, b, lm_lt, lm64, avail);
    uint32_t eff_ge = do_cmp(c, b, PCM_GE, PTY_U64, 0, eff, Wc, line);
    uint32_t effc = nid(c);
    emit_arith(c, b, S_And, c->t_u64, effc, eff,
               const_u64(c, W - 1));
    uint32_t one = nid(c);
    emit_arith(c, b, S_Shl, c->t_u64, one, const_u64(c, 1), effc);
    uint32_t mask0 = nid(c);
    emit_arith(c, b, S_ISub, c->t_u64, mask0, one, const_u64(c, 1));
    uint32_t mask = sel_u64(c, b, eff_ge, const_u64(c, all), mask0);
    uint32_t pmc = nid(c);
    emit_arith(c, b, S_And, c->t_u64, pmc, pm64,
               const_u64(c, W - 1));
    uint32_t ash = nid(c), field = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, ash, a64, pmc);
    emit_arith(c, b, S_And, c->t_u64, field, ash, mask);
    // The extracted field lands at bit 0; sign fill starts at bit eff.
    uint32_t hs = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, hs, eff, const_u64(c, 0));
    uint32_t hs_ge = do_cmp(c, b, PCM_GE, PTY_U64, 0, hs, Wc, line);
    uint32_t hsc = nid(c), high0 = nid(c);
    emit_arith(c, b, S_And, c->t_u64, hsc, hs, const_u64(c, W - 1));
    emit_arith(c, b, S_Shl, c->t_u64, high0, const_u64(c, all), hsc);
    uint32_t high = sel_u64(c, b, hs_ge, const_u64(c, 0), high0);
    uint32_t sb = do_cmp(c, b, PCM_NE, PTY_U64, 0, sbit,
                         const_u64(c, 0), line);
    uint32_t ext = sel_u64(c, b, sb, high, const_u64(c, 0));
    uint32_t norm = nid(c);
    emit_arith(c, b, S_Or, c->t_u64, norm, field, ext);
    uint32_t posres = sel_u64(c, b, sb, const_u64(c, all),
                              const_u64(c, 0));
    return sel_u64(c, b, pm_ge, posres, norm);
}

uint32_t bfi_value(ctx_t *c, spv_blob_t *b, uint32_t a64,
                          uint32_t b64, uint32_t cpos, uint32_t dlen,
                          int w64, int line) {
    uint32_t W = w64 ? 64u : 32u;
    uint64_t all = w64 ? 0xffffffffffffffffull : 0xffffffffull;
    uint32_t pm = nid(c), lm = nid(c);
    emit_arith(c, b, S_And, c->t_u32, pm, cpos, const_u32(c, 0xffu));
    emit_arith(c, b, S_And, c->t_u32, lm, dlen, const_u32(c, 0xffu));
    uint32_t pm64 = nid(c), lm64 = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u64);
    put(c, b, pm64);
    put(c, b, pm);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u64);
    put(c, b, lm64);
    put(c, b, lm);
    uint32_t Wc = const_u64(c, W);
    uint32_t pm_ge = do_cmp(c, b, PCM_GE, PTY_U64, 0, pm64, Wc, line);
    uint32_t lm_zero = do_cmp(c, b, PCM_EQ, PTY_U64, 0, lm64,
                              const_u64(c, 0), line);
    uint32_t avail = nid(c);
    emit_arith(c, b, S_ISub, c->t_u64, avail, Wc, pm64);
    avail = sel_u64(c, b, pm_ge, const_u64(c, 0), avail);
    uint32_t lm_lt = do_cmp(c, b, PCM_LT, PTY_U64, 0, lm64, avail, line);
    uint32_t eff = sel_u64(c, b, lm_lt, lm64, avail);
    uint32_t eff_ge = do_cmp(c, b, PCM_GE, PTY_U64, 0, eff, Wc, line);
    uint32_t effc = nid(c);
    emit_arith(c, b, S_And, c->t_u64, effc, eff,
               const_u64(c, W - 1));
    uint32_t one = nid(c);
    emit_arith(c, b, S_Shl, c->t_u64, one, const_u64(c, 1), effc);
    uint32_t mask0 = nid(c);
    emit_arith(c, b, S_ISub, c->t_u64, mask0, one, const_u64(c, 1));
    uint32_t mask = sel_u64(c, b, eff_ge, const_u64(c, all), mask0);
    uint32_t pmc = nid(c);
    emit_arith(c, b, S_And, c->t_u64, pmc, pm64,
               const_u64(c, W - 1));
    uint32_t shm = nid(c);
    emit_arith(c, b, S_Shl, c->t_u64, shm, mask, pmc);
    mask = shm;
    uint32_t nm = nid(c), ins = nid(c);
    put(c, b, (4u << 16) | S_Not);
    put(c, b, c->t_u64);
    put(c, b, nm);
    put(c, b, mask);
    emit_arith(c, b, S_Shl, c->t_u64, ins, a64, pmc);
    uint32_t t1 = nid(c), t2 = nid(c), res = nid(c);
    emit_arith(c, b, S_And, c->t_u64, t1, b64, nm);
    emit_arith(c, b, S_And, c->t_u64, t2, ins, mask);
    emit_arith(c, b, S_Or, c->t_u64, res, t1, t2);
    uint32_t bad = nid(c);
    put(c, b, (5u << 16) | S_LogicalOr);
    put(c, b, c->t_bool);
    put(c, b, bad);
    put(c, b, pm_ge);
    put(c, b, lm_zero);
    return sel_u64(c, b, bad, b64, res);
}

// Wide add with carry-in; returns the carry-out bool and writes the
// truncated sum. `a`/`b`/`cin` are u64 values; for 32-bit operations
// the caller passes zero-extended operands and the carry threshold is
// 2^32 (the u64 sum cannot itself overflow).
uint32_t add_carry_u64(ctx_t *c, spv_blob_t *b, int w64,
                              uint32_t a, uint32_t b64, uint32_t cin,
                              uint32_t *sum_out, int line) {
    uint32_t s1 = nid(c), sum = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, s1, a, b64);
    emit_arith(c, b, S_IAdd, c->t_u64, sum, s1, cin);
    *sum_out = sum;
    if (!w64) {
        uint32_t carry = do_cmp(c, b, PCM_GE, PTY_U64, 0, sum,
                                const_u64(c, 0x100000000ull), line);
        return carry;
    }
    uint32_t c1 = do_cmp(c, b, PCM_LT, PTY_U64, 0, s1, a, line);
    uint32_t c2 = do_cmp(c, b, PCM_LT, PTY_U64, 0, sum, s1, line);
    uint32_t out = nid(c);
    put(c, b, (5u << 16) | S_LogicalOr);
    put(c, b, c->t_bool);
    put(c, b, out);
    put(c, b, c1);
    put(c, b, c2);
    return out;
}

// Wide subtract with borrow-in; returns the borrow-out bool and writes
// the truncated difference.
uint32_t sub_borrow_u64(ctx_t *c, spv_blob_t *b, int w64,
                               uint32_t a, uint32_t b64, uint32_t cin,
                               uint32_t *diff_out, int line) {
    uint32_t t = nid(c), diff = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, t, b64, cin);
    emit_arith(c, b, S_ISub, c->t_u64, diff, a, t);
    *diff_out = diff;
    uint32_t lt = do_cmp(c, b, PCM_LT, PTY_U64, 0, a, t, line);
    if (!w64)
        return lt;
    uint32_t ov = do_cmp(c, b, PCM_LT, PTY_U64, 0, t, b64, line);
    uint32_t out = nid(c);
    put(c, b, (5u << 16) | S_LogicalOr);
    put(c, b, c->t_bool);
    put(c, b, out);
    put(c, b, lt);
    put(c, b, ov);
    return out;
}

// High half of a full-width integer product (mul.hi / mad.hi).
uint32_t mul_hi_value(ctx_t *c, spv_blob_t *b, ptx_ty_t ty,
                             int sign, uint32_t a_id, uint32_t b_id,
                             int line) {
    if (ty == PTY_U16) {
        uint32_t av = a_id, bv = b_id;
        if (sign) {
            av = pack_lane_get(c, b, a_id, 0, 16, 1);
            bv = pack_lane_get(c, b, b_id, 0, 16, 1);
        }
        uint32_t pr = nid(c);
        emit_arith(c, b, S_IMul, c->t_u32, pr, av, bv);
        uint32_t hi = nid(c);
        emit_arith(c, b, sign ? S_ShrA : S_ShrL, c->t_u32, hi, pr,
                   const_u32(c, 16));
        uint32_t m = nid(c);
        emit_arith(c, b, S_And, c->t_u32, m, hi, const_u32(c, 0xffffu));
        return m;
    }
    if (ty == PTY_U32) {
        uint32_t wa = nid(c), wb = nid(c);
        uint32_t conv = sign ? S_SConvert : S_UConvert;
        put(c, b, (4u << 16) | conv);
        put(c, b, c->t_u64);
        put(c, b, wa);
        put(c, b, a_id);
        put(c, b, (4u << 16) | conv);
        put(c, b, c->t_u64);
        put(c, b, wb);
        put(c, b, b_id);
        uint32_t pr = nid(c);
        emit_arith(c, b, S_IMul, c->t_u64, pr, wa, wb);
        uint32_t hi = nid(c);
        emit_arith(c, b, sign ? S_ShrA : S_ShrL, c->t_u64, hi, pr,
                   const_u64(c, 32));
        uint32_t out = nid(c);
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, c->t_u32);
        put(c, b, out);
        put(c, b, hi);
        return out;
    }
    uint32_t a0 = nid(c), a1 = nid(c), b0 = nid(c), b1 = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u32);
    put(c, b, a0);
    put(c, b, a_id);
    emit_arith(c, b, S_ShrL, c->t_u64, a1, a_id, const_u64(c, 32));
    uint32_t a1t = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u32);
    put(c, b, a1t);
    put(c, b, a1);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u32);
    put(c, b, b0);
    put(c, b, b_id);
    emit_arith(c, b, S_ShrL, c->t_u64, b1, b_id, const_u64(c, 32));
    uint32_t b1t = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u32);
    put(c, b, b1t);
    put(c, b, b1);
    uint32_t a0w = nid(c), a1w = nid(c), b0w = nid(c), b1w = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u64);
    put(c, b, a0w);
    put(c, b, a0);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u64);
    put(c, b, a1w);
    put(c, b, a1t);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u64);
    put(c, b, b0w);
    put(c, b, b0);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u64);
    put(c, b, b1w);
    put(c, b, b1t);
    uint32_t p0 = nid(c), p1 = nid(c), p2 = nid(c), p3 = nid(c);
    emit_arith(c, b, S_IMul, c->t_u64, p0, a0w, b0w);
    emit_arith(c, b, S_IMul, c->t_u64, p1, a0w, b1w);
    emit_arith(c, b, S_IMul, c->t_u64, p2, a1w, b0w);
    emit_arith(c, b, S_IMul, c->t_u64, p3, a1w, b1w);
    uint32_t m1 = nid(c), m2 = nid(c), l1 = nid(c), l2 = nid(c);
    emit_arith(c, b, S_And, c->t_u64, m1, p1, const_u64(c, 0xffffffffull));
    emit_arith(c, b, S_And, c->t_u64, m2, p2, const_u64(c, 0xffffffffull));
    emit_arith(c, b, S_Shl, c->t_u64, l1, m1, const_u64(c, 32));
    emit_arith(c, b, S_Shl, c->t_u64, l2, m2, const_u64(c, 32));
    uint32_t t = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, t, p0, l1);
    uint32_t c1a = do_cmp(c, b, PCM_LT, PTY_U64, 0, t, p0, line);
    if (c->failed)
        return 0;
    uint32_t t2 = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, t2, t, l2);
    uint32_t c1b = do_cmp(c, b, PCM_LT, PTY_U64, 0, t2, t, line);
    if (c->failed)
        return 0;
    uint32_t k32a = sel_u32(c, b, c1a, const_u32(c, 1), const_u32(c, 0));
    uint32_t k32b = sel_u32(c, b, c1b, const_u32(c, 1), const_u32(c, 0));
    uint32_t kk = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u32, kk, k32a, k32b);
    uint32_t h1 = nid(c), h2 = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, h1, p1, const_u64(c, 32));
    emit_arith(c, b, S_ShrL, c->t_u64, h2, p2, const_u64(c, 32));
    uint32_t s1 = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, s1, h1, h2);
    uint32_t k64 = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u64);
    put(c, b, k64);
    put(c, b, kk);
    uint32_t small = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, small, s1, k64);
    uint32_t out = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, out, p3, small);
    if (sign) {
        // high_signed = high_unsigned - (a<0 ? b : 0) - (b<0 ? a : 0)
        uint32_t sa = nid(c), sb = nid(c);
        emit_arith(c, b, S_ShrL, c->t_u64, sa, a_id, const_u64(c, 63));
        emit_arith(c, b, S_ShrL, c->t_u64, sb, b_id, const_u64(c, 63));
        uint32_t an = do_cmp(c, b, PCM_NE, PTY_U64, 0, sa,
                             const_u64(c, 0), line);
        uint32_t bn = do_cmp(c, b, PCM_NE, PTY_U64, 0, sb,
                             const_u64(c, 0), line);
        if (c->failed)
            return 0;
        uint32_t ca = sel_u64(c, b, an, b_id, const_u64(c, 0));
        uint32_t cb = sel_u64(c, b, bn, a_id, const_u64(c, 0));
        uint32_t d1 = nid(c), d2 = nid(c);
        emit_arith(c, b, S_ISub, c->t_u64, d1, out, ca);
        emit_arith(c, b, S_ISub, c->t_u64, d2, d1, cb);
        out = d2;
    }
    return out;
}

uint32_t flush_f16_bits(ctx_t *c, spv_blob_t *b, uint32_t bits,
                               int line) {
    uint32_t e = nid(c), m = nid(c);
    emit_arith(c, b, S_And, c->t_u32, e, bits, const_u32(c, 0x7c00u));
    emit_arith(c, b, S_And, c->t_u32, m, bits, const_u32(c, 0x3ffu));
    uint32_t ez = do_cmp(c, b, PCM_EQ, PTY_U32, 0, e, const_u32(c, 0),
                         line);
    uint32_t nz = do_cmp(c, b, PCM_NE, PTY_U32, 0, m, const_u32(c, 0),
                         line);
    uint32_t sub = nid(c);
    put(c, b, (5u << 16) | S_LogicalAnd);
    put(c, b, c->t_bool);
    put(c, b, sub);
    put(c, b, ez);
    put(c, b, nz);
    uint32_t sgn = nid(c);
    emit_arith(c, b, S_And, c->t_u32, sgn, bits, const_u32(c, 0x8000u));
    return sel_u32(c, b, sub, sgn, bits);
}

uint32_t set_bop_join(ctx_t *c, spv_blob_t *b,
                             const ptx_instr_t *in, uint32_t t, int line) {
    if (!in->bop)
        return t;
    reg_t *qr = find_reg(c, in->src[2].reg);
    if (!qr || qr->decl_ty != PTY_PRED || !qr->defined) {
        failf(c, line, "set boolean source must be a defined .pred register");
        return t;
    }
    uint32_t q = qr->cur;
    if (in->bneg) {
        uint32_t nq = nid(c);
        put(c, b, (4u << 16) | S_LogicalNot);
        put(c, b, c->t_bool);
        put(c, b, nq);
        put(c, b, q);
        q = nq;
    }
    uint32_t r = nid(c);
    put(c, b, (5u << 16) | (in->bop == 1 ? S_LogicalAnd
                            : in->bop == 2 ? S_LogicalOr
                                           : S_LogicalNotEqual));
    put(c, b, c->t_bool);
    put(c, b, r);
    put(c, b, t);
    put(c, b, q);
    return r;
}


// .relu on a narrowing source: ordered max(v, +0), where NaN maps to
// +0 (the corpus convention used by the packed narrowing path).
uint32_t f32_relu_nonneg(ctx_t *c, spv_blob_t *b, uint32_t v,
                                int line) {
    uint32_t gt = do_cmp(c, b, PCM_GT, PTY_F32, 0, v,
                         const_f32(c, 0x00000000u), line);
    if (c->failed)
        return 0;
    uint32_t r = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, c->t_f32);
    put(c, b, r);
    put(c, b, gt);
    put(c, b, v);
    put(c, b, const_f32(c, 0x00000000u));
    return r;
}

// .satfinite narrowing to f16: clamp a rounded half pattern that came
// out infinite to the largest finite magnitude, preserving the sign.
// NaN (mantissa non-zero) is left untouched, as required.
uint32_t f16_satfinite_bits(ctx_t *c, spv_blob_t *b, uint32_t bits,
                                   int line) {
    uint32_t mag = u32_bin(c, b, S_And, bits, const_u32(c, 0x7fffu));
    uint32_t isinf = cmp_u32(c, b, PCM_EQ, 0, mag, const_u32(c, 0x7c00u),
                             line);
    uint32_t sign = u32_bin(c, b, S_And, bits, const_u32(c, 0x8000u));
    uint32_t maxb = u32_bin(c, b, S_Or, sign, const_u32(c, 0x7bffu));
    return sel_u32(c, b, isinf, maxb, bits);
}

