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

uint32_t do_cmp(ctx_t *c, spv_blob_t *b, ptx_cmp_t cmp, ptx_ty_t ty,
                       int sign, uint32_t a, uint32_t b2, int line) {
    ty = eff_ty(c, ty);
    uint32_t op = 0;
    if (ty == PTY_U32 && !sign) {
        switch (cmp) {
        case PCM_GE: op = S_UGreaterEqual; break;
        case PCM_LT: op = S_ULess; break;
        case PCM_LE: op = S_ULessEqual; break;
        case PCM_GT: op = S_UGreater; break;
        case PCM_EQ: op = S_IEqual; break;
        case PCM_NE: op = S_INotEqual; break;
        case PCM_NAN:
        case PCM_NUM:
        case PCM_EQU:
        case PCM_NEU:
        case PCM_LTU:
        case PCM_LEU:
        case PCM_GTU:
        case PCM_GEU:
            failf(c, line, "unordered comparison needs .f32/.f64");
            return 0;
        }
    } else if (ty == PTY_U64 && sign) {
        switch (cmp) {
        case PCM_GE: op = S_SGreaterEqual; break;
        case PCM_LT: op = S_SLess; break;
        case PCM_LE: op = S_SLessEqual; break;
        case PCM_GT: op = S_SGreater; break;
        case PCM_EQ: op = S_IEqual; break;
        case PCM_NE: op = S_INotEqual; break;
        case PCM_NAN:
        case PCM_NUM:
        case PCM_EQU:
        case PCM_NEU:
        case PCM_LTU:
        case PCM_LEU:
        case PCM_GTU:
        case PCM_GEU:
            failf(c, line, "unordered comparison needs .f32/.f64");
            return 0;
        }
    } else if (ty == PTY_U32 && sign) {
        // Signed 32-bit (setp.s32, min/max.s32); S-prefixed ops
        // interpret the 32-bit words as signed.
        switch (cmp) {
        case PCM_GE: op = S_SGreaterEqual; break;
        case PCM_LT: op = S_SLess; break;
        case PCM_LE: op = S_SLessEqual; break;
        case PCM_GT: op = S_SGreater; break;
        case PCM_EQ: op = S_IEqual; break;
        case PCM_NE: op = S_INotEqual; break;
        case PCM_NAN:
        case PCM_NUM:
        case PCM_EQU:
        case PCM_NEU:
        case PCM_LTU:
        case PCM_LEU:
        case PCM_GTU:
        case PCM_GEU:
            failf(c, line, "unordered comparison needs .f32/.f64");
            return 0;
        }
    } else if (ty == PTY_U8 || ty == PTY_U16) {
        // Narrow canonical values are zero-extended; signed forms
        // narrow to full 32-bit two's complement before calling.
        switch (cmp) {
        case PCM_GE: op = sign ? S_SGreaterEqual : S_UGreaterEqual; break;
        case PCM_LT: op = sign ? S_SLess : S_ULess; break;
        case PCM_LE: op = sign ? S_SLessEqual : S_ULessEqual; break;
        case PCM_GT: op = sign ? S_SGreater : S_UGreater; break;
        case PCM_EQ: op = S_IEqual; break;
        case PCM_NE: op = S_INotEqual; break;
        case PCM_NAN:
        case PCM_NUM:
        case PCM_EQU:
        case PCM_NEU:
        case PCM_LTU:
        case PCM_LEU:
        case PCM_GTU:
        case PCM_GEU:
            failf(c, line, "unordered comparison needs .f32/.f64");
            return 0;
        }
    } else if (ty == PTY_U64 && !sign) {
        switch (cmp) {
        case PCM_GE: op = S_UGreaterEqual; break;
        case PCM_LT: op = S_ULess; break;
        case PCM_LE: op = S_ULessEqual; break;
        case PCM_GT: op = S_UGreater; break;
        case PCM_EQ: op = S_IEqual; break;
        case PCM_NE: op = S_INotEqual; break;
        case PCM_NAN:
        case PCM_NUM:
        case PCM_EQU:
        case PCM_NEU:
        case PCM_LTU:
        case PCM_LEU:
        case PCM_GTU:
        case PCM_GEU:
            failf(c, line, "unordered comparison needs .f32/.f64");
            return 0;
        }
    } else if ((ty == PTY_F32 || ty == PTY_F64) &&
               (cmp == PCM_NAN || cmp == PCM_NUM)) {
        // Unordered NaN tests (setp.nan/num): nan = either operand NaN
        // (unordered-not-equal self-compares ORed); num = both numeric.
        uint32_t ua = nid(c), ub = nid(c);
        put(c, b, (5u << 16) | S_FUnordNotEqual);
        put(c, b, c->t_bool);
        put(c, b, ua);
        put(c, b, a);
        put(c, b, a);
        put(c, b, (5u << 16) | S_FUnordNotEqual);
        put(c, b, c->t_bool);
        put(c, b, ub);
        put(c, b, b2);
        put(c, b, b2);
        uint32_t is_nan = nid(c);
        put(c, b, (5u << 16) | S_LogicalOr);
        put(c, b, c->t_bool);
        put(c, b, is_nan);
        put(c, b, ua);
        put(c, b, ub);
        if (cmp == PCM_NAN)
            return is_nan;
        uint32_t is_num = nid(c);
        put(c, b, (4u << 16) | S_LogicalNot);
        put(c, b, c->t_bool);
        put(c, b, is_num);
        put(c, b, is_nan);
        return is_num;
    } else if ((ty == PTY_F32 || ty == PTY_F64) &&
               (cmp == PCM_EQU || cmp == PCM_NEU || cmp == PCM_LTU ||
                cmp == PCM_LEU || cmp == PCM_GTU || cmp == PCM_GEU)) {
        // Unordered comparisons (true when either operand is NaN):
        // single unordered opcodes.
        uint32_t op = cmp == PCM_EQU   ? S_FUnordEqual
                      : cmp == PCM_NEU ? S_FUnordNotEqual
                      : cmp == PCM_LTU ? S_FUnordLessThan
                      : cmp == PCM_LEU ? S_FUnordLessThanEqual
                      : cmp == PCM_GTU ? S_FUnordGreaterThan
                                       : S_FUnordGreaterThanEqual;
        uint32_t id = nid(c);
        emit_arith(c, b, op, c->t_bool, id, a, b2);
        return id;
    } else if (ty == PTY_F32 || ty == PTY_F64) {
        switch (cmp) {
        case PCM_GE: op = S_FOrdGreaterEqual; break;
        case PCM_LT: op = S_FOrdLess; break;
        case PCM_LE: op = S_FOrdLessEqual; break;
        case PCM_GT: op = S_FOrdGreater; break;
        case PCM_EQ: op = S_FOrdEqual; break;
        case PCM_NE: op = S_FOrdNotEqual; break;
        case PCM_NAN:
        case PCM_NUM:
        case PCM_EQU:
        case PCM_NEU:
        case PCM_LTU:
        case PCM_LEU:
        case PCM_GTU:
        case PCM_GEU:
            failf(c, line, "unordered comparison needs .f32/.f64");
            return 0;
        }
    } else {
        failf(c, line, "setp needs integer, .f16/.f32/.f64, or signed integer comparisons");
        return 0;
    }
    uint32_t id = nid(c);
    emit_arith(c, b, op, c->t_bool, id, a, b2);
    return id;
}

// Clamp a 32-bit value to a signed/unsigned lane of `bits` bits and
// return it masked to exactly that width (cvt.pack.sat.<ty>.s32[.b32]).
uint32_t cvt_pack_lane(ctx_t *c, spv_blob_t *b, uint32_t v,
                       int sign, int bits, int line) {
    uint32_t mask = bits >= 32 ? 0xffffffffu : ((1u << bits) - 1u);
    uint32_t x;
    if (!sign) {
        uint32_t ge0 = do_cmp(c, b, PCM_GE, PTY_U32, 1, v,
                              const_u32(c, 0), line);
        x = sel_u32(c, b, ge0, v, const_u32(c, 0));
        if (c->failed)
            return 0;
        uint32_t le = do_cmp(c, b, PCM_LE, PTY_U32, 1, x,
                             const_u32(c, mask), line);
        x = sel_u32(c, b, le, x, const_u32(c, mask));
    } else {
        uint32_t maxv = (1u << (bits - 1)) - 1u;
        uint32_t minv = (uint32_t)0 - (1u << (bits - 1));
        uint32_t ge = do_cmp(c, b, PCM_GE, PTY_U32, 1, v,
                             const_u32(c, minv), line);
        x = sel_u32(c, b, ge, v, const_u32(c, minv));
        if (c->failed)
            return 0;
        uint32_t le = do_cmp(c, b, PCM_LE, PTY_U32, 1, x,
                             const_u32(c, maxv), line);
        x = sel_u32(c, b, le, x, const_u32(c, maxv));
    }
    if (c->failed)
        return 0;
    uint32_t id = nid(c);
    emit_arith(c, b, S_And, c->t_u32, id, x, const_u32(c, mask));
    return id;
}

// Round f32 bits to an integer-valued f32: rni = nearest-even, rzi =
// toward-zero. Exact integer bit manipulation in the f32_to_f16bits
// idiom (no GLSL import): Inf/NaN and already-integral values pass
// through; denormals and |x| < 0.5 round to signed zero. The shift
// amount is clamped to [1,24] so every emitted shift is valid even on
// discarded select arms.
uint32_t round_f32_intvalued(ctx_t *c, spv_blob_t *b, uint32_t f,
                                   int rni, int line) {
    // f32 SSA values are float-typed; the bit tricks need u32 bits.
    uint32_t bits = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_u32);
    put(c, b, bits);
    put(c, b, f);
    uint32_t s = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, s, bits, const_u32(c, 31));
    uint32_t e = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, e, bits, const_u32(c, 23));
    uint32_t e2 = nid(c);
    emit_arith(c, b, S_And, c->t_u32, e2, e, const_u32(c, 0xffu));
    uint32_t m = nid(c);
    emit_arith(c, b, S_And, c->t_u32, m, bits, const_u32(c, 0x7fffffu));
    uint32_t full = nid(c);
    emit_arith(c, b, S_Or, c->t_u32, full, m, const_u32(c, 0x800000u));
    // k = 150 - e2: fractional significand bits (signed two's
    // complement). Uses the MASKED exponent: the raw shift keeps the
    // sign bit, which would send every negative input down the
    // already-integral path.
    uint32_t k = nid(c);
    emit_arith(c, b, S_ISub, c->t_u32, k, const_u32(c, 150), e2);
    // Clamped shift 1..24: valid on every arm, garbage discarded.
    uint32_t lt24 = do_cmp(c, b, PCM_LT, PTY_U32, 0, k, const_u32(c, 24), line);
    uint32_t tmin = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, c->t_u32);
    put(c, b, tmin);
    put(c, b, lt24);
    put(c, b, k);
    put(c, b, const_u32(c, 24));
    uint32_t lt1 = do_cmp(c, b, PCM_LT, PTY_U32, 0, tmin, const_u32(c, 1), line);
    uint32_t kk = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, c->t_u32);
    put(c, b, kk);
    put(c, b, lt1);
    put(c, b, const_u32(c, 1));
    put(c, b, tmin);
    uint32_t keep = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, keep, full, kk);
    uint32_t one = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, one, const_u32(c, 1), kk);
    uint32_t mask = nid(c);
    emit_arith(c, b, S_ISub, c->t_u32, mask, one, const_u32(c, 1));
    uint32_t drop = nid(c);
    emit_arith(c, b, S_And, c->t_u32, drop, full, mask);
    uint32_t km1 = nid(c);
    emit_arith(c, b, S_ISub, c->t_u32, km1, kk, const_u32(c, 1));
    uint32_t half = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, half, const_u32(c, 1), km1);
    uint32_t up = const_u32(c, 0);
    if (rni) {
        // Ties-to-even: round up past half, or on a tie with odd keep.
        uint32_t gt = do_cmp(c, b, PCM_GT, PTY_U32, 0, drop, half, line);
        uint32_t eq = do_cmp(c, b, PCM_EQ, PTY_U32, 0, drop, half, line);
        uint32_t odd = nid(c);
        emit_arith(c, b, S_And, c->t_u32, odd, keep, const_u32(c, 1));
        uint32_t gt32 = sel_u32(c, b, gt, const_u32(c, 1), const_u32(c, 0));
        uint32_t tie = sel_u32(c, b, eq, odd, const_u32(c, 0));
        uint32_t up32 = nid(c);
        emit_arith(c, b, S_Or, c->t_u32, up32, gt32, tie);
        up = up32;
    }
    uint32_t nf = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u32, nf, keep, up);
    uint32_t shifted = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, shifted, nf, kk);
    uint32_t carry = do_cmp(c, b, PCM_EQ, PTY_U32, 0, shifted,
                            const_u32(c, 0x1000000u), line);
    // Carry: rounded mantissa hit 2^24 (bit 24 set after the shift).
    uint32_t e1 = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u32, e1, e2, const_u32(c, 1));
    uint32_t eprime = sel_u32(c, b, carry, e1, e2);
    uint32_t mant0 = nid(c);
    emit_arith(c, b, S_And, c->t_u32, mant0, shifted, const_u32(c, 0x7fffffu));
    uint32_t mant = sel_u32(c, b, carry, const_u32(c, 0), mant0);
    uint32_t sh_s = nid(c), sh_e = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, sh_s, s, const_u32(c, 31));
    emit_arith(c, b, S_Shl, c->t_u32, sh_e, eprime, const_u32(c, 23));
    uint32_t r = nid(c);
    emit_arith(c, b, S_Or, c->t_u32, r, sh_s, sh_e);
    uint32_t r2 = nid(c);
    emit_arith(c, b, S_Or, c->t_u32, r2, r, mant);
    // newfull == 0 (whole significand fractional, no roundup: k == 24
    // with up == 0) means signed zero, not the unchanged exponent.
    uint32_t k_is_24 = do_cmp(c, b, PCM_EQ, PTY_U32, 0, k, const_u32(c, 24), line);
    uint32_t up_is_0 = do_cmp(c, b, PCM_EQ, PTY_U32, 0, up, const_u32(c, 0), line);
    uint32_t up0flag = sel_u32(c, b, up_is_0, const_u32(c, 1), const_u32(c, 0));
    uint32_t zflag = sel_u32(c, b, k_is_24, up0flag, const_u32(c, 0));
    uint32_t is_zero = do_cmp(c, b, PCM_NE, PTY_U32, 0, zflag, const_u32(c, 0), line);
    // Case selects over u32 bit patterns (overlaps agree: special/k<=0
    // keep the input bits, denorm/big-k yield signed zero); the result
    // is bitcast back to float type for f32 consumers.
    uint32_t signzero = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, signzero, s, const_u32(c, 31));
    r2 = sel_u32(c, b, is_zero, signzero, r2);
    uint32_t is_special = do_cmp(c, b, PCM_EQ, PTY_U32, 0, e2, const_u32(c, 0xffu), line);
    uint32_t v1 = sel_u32(c, b, is_special, bits, r2);
    uint32_t k_le_0 = do_cmp(c, b, PCM_LE, PTY_U32, 1, k, const_u32(c, 0), line);
    uint32_t v2 = sel_u32(c, b, k_le_0, bits, v1);
    uint32_t is_denorm = do_cmp(c, b, PCM_EQ, PTY_U32, 0, e2, const_u32(c, 0), line);
    uint32_t v3 = sel_u32(c, b, is_denorm, signzero, v2);
    if (c->failed)
        return 0;
    uint32_t k_ge_25 = do_cmp(c, b, PCM_GE, PTY_U32, 1, k, const_u32(c, 25), line);
    uint32_t v4 = sel_u32(c, b, k_ge_25, signzero, v3);
    uint32_t out = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_f32);
    put(c, b, out);
    put(c, b, v4);
    return out;
}

// Flush f32 denormals to signed zero (ftz mode: inputs and results).
// Normals, zeros, Infs and NaNs pass through unchanged.
// Resolve a float source that may sit in a `.b32`-declared holder: the
// corpus loads f32 with `ld.f32` into .b32 temporaries (ptxas treats
// .b32 as untyped storage), which the front end normalizes to a .b32
// load. Reinterpret those bits locally instead of weakening the global
// f32/u32 register check (same approach as POP_CVTPACK).
val_t resolve_f32_bits(ctx_t *c, spv_blob_t *b, const ptx_opd_t *op,
                              int line) {
    reg_t *r = op->kind == OPD_REG ? find_reg(c, op->reg) : NULL;
    if (r && r->defined && !r->is_ptr && r->decl_ty == PTY_U32) {
        val_t v;
        memset(&v, 0, sizeof(v));
        v.ty = PTY_F32;
        val_t w = resolve(c, b, op, PTY_U32, line);
        if (c->failed)
            return v;
        v.id = nid(c);
        put(c, b, (4u << 16) | S_Bitcast);
        put(c, b, c->t_f32);
        put(c, b, v.id);
        put(c, b, w.id);
        return v;
    }
    return resolve(c, b, op, PTY_F32, line);
}

// Directed rounding of an f32 to an integer-valued f32: rnd 1 = rni
// (nearest-even), 2 = rzi (toward zero), 3 = rmi (toward -inf),
// 4 = rpi (toward +inf). Directed modes start from the nearest (or
// truncated) integer and step by 1.0 when that overshoots the source;
// the value is already integral and |step| is exact, so the only
// inexact case is |x| >= 2^23, where the floor/ceil equals x and no
// step is taken. NaN/Inf compare false and pass through unchanged.
uint32_t round_f64_directed(ctx_t *c, spv_blob_t *b, uint32_t x,
                                   int rnd, int line) {
    // GLSL.std.450: RoundEven=2, Trunc=3, Floor=8, Ceil=9.
    uint32_t op = rnd == 1 ? 2 : rnd == 2 ? 3 : rnd == 3 ? 8 : 9;
    uint32_t id = nid(c);
    put(c, b, (6u << 16) | S_ExtInst);
    put(c, b, c->t_f64);
    put(c, b, id);
    put(c, b, c->glsl450);
    put(c, b, op);
    put(c, b, x);
    (void)line;
    return id;
}

uint32_t round_f32_directed(ctx_t *c, spv_blob_t *b, uint32_t x,
                                   int rnd, int line) {
    uint32_t r = round_f32_intvalued(c, b, x, rnd != 2, line);
    if (c->failed)
        return 0;
    if (rnd != 3 && rnd != 4)
        return r;
    uint32_t over = do_cmp(c, b, rnd == 3 ? PCM_GT : PCM_LT, PTY_F32, 0, r,
                           x, line);
    if (c->failed)
        return 0;
    uint32_t stepped = nid(c);
    emit_arith(c, b, rnd == 3 ? S_FSub : S_FAdd, c->t_f32, stepped, r,
               const_f32(c, 0x3f800000u));
    uint32_t id = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, c->t_f32);
    put(c, b, id);
    put(c, b, over);
    put(c, b, stepped);
    put(c, b, r);
    // Directed rounding through zero keeps the source sign (IEEE:
    // ceil(-0.3) = -0, floor(0.3) = +0). The 1.0 step can drop it
    // (-1.0 + 1.0 = +0), so OR the source sign back when the result is
    // a zero.
    uint32_t vb = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_u32);
    put(c, b, vb);
    put(c, b, id);
    uint32_t xb = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_u32);
    put(c, b, xb);
    put(c, b, x);
    uint32_t absv = nid(c), sgn = nid(c);
    emit_arith(c, b, S_And, c->t_u32, absv, vb, const_u32(c, 0x7fffffffu));
    emit_arith(c, b, S_And, c->t_u32, sgn, xb, const_u32(c, 0x80000000u));
    uint32_t isz = do_cmp(c, b, PCM_EQ, PTY_U32, 0, absv, const_u32(c, 0), line);
    if (c->failed)
        return 0;
    uint32_t fixed = nid(c);
    emit_arith(c, b, S_Or, c->t_u32, fixed, vb, sgn);
    uint32_t sel = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, c->t_u32);
    put(c, b, sel);
    put(c, b, isz);
    put(c, b, fixed);
    put(c, b, vb);
    uint32_t out = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_f32);
    put(c, b, out);
    put(c, b, sel);
    return out;
}

uint32_t flush_f32_denorm(ctx_t *c, spv_blob_t *b, uint32_t f, int line) {
    uint32_t bits = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_u32);
    put(c, b, bits);
    put(c, b, f);
    uint32_t e = nid(c), m = nid(c), s = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, e, bits, const_u32(c, 23));
    uint32_t e2 = nid(c);
    emit_arith(c, b, S_And, c->t_u32, e2, e, const_u32(c, 0xffu));
    emit_arith(c, b, S_And, c->t_u32, m, bits, const_u32(c, 0x7fffffu));
    emit_arith(c, b, S_ShrL, c->t_u32, s, bits, const_u32(c, 31));
    uint32_t e0 = do_cmp(c, b, PCM_EQ, PTY_U32, 0, e2, const_u32(c, 0), line);
    if (c->failed)
        return 0;
    uint32_t mnz = do_cmp(c, b, PCM_NE, PTY_U32, 0, m, const_u32(c, 0), line);
    if (c->failed)
        return 0;
    // No LogicalAnd opcode allocated: denormal = e==0 AND m!=0 via a
    // select chain (e0 ? (mne0 ? true : false) : false).
    uint32_t tsel = sel_u32(c, b, mnz, const_u32(c, 1), const_u32(c, 0));
    uint32_t is_den = sel_u32(c, b, e0, tsel, const_u32(c, 0));
    uint32_t is_den_b = do_cmp(c, b, PCM_NE, PTY_U32, 0, is_den, const_u32(c, 0), line);
    if (c->failed)
        return 0;
    uint32_t szb = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, szb, s, const_u32(c, 31));
    uint32_t sz = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_f32);
    put(c, b, sz);
    put(c, b, szb);
    uint32_t out = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, c->t_f32);
    put(c, b, out);
    put(c, b, is_den_b);
    put(c, b, sz);
    put(c, b, f);
    return out;
}

// 32-bit population count via parallel sums (all shifts < 32).
uint32_t popc32(ctx_t *c, spv_blob_t *b, uint32_t x) {
    uint32_t t1 = nid(c), t2 = nid(c), s1 = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, t1, x, const_u32(c, 1));
    emit_arith(c, b, S_And, c->t_u32, t2, t1, const_u32(c, 0x55555555u));
    emit_arith(c, b, S_ISub, c->t_u32, s1, x, t2);
    uint32_t t3 = nid(c), t4 = nid(c), t5 = nid(c), s2 = nid(c);
    emit_arith(c, b, S_And, c->t_u32, t3, s1, const_u32(c, 0x33333333u));
    emit_arith(c, b, S_ShrL, c->t_u32, t4, s1, const_u32(c, 2));
    emit_arith(c, b, S_And, c->t_u32, t5, t4, const_u32(c, 0x33333333u));
    emit_arith(c, b, S_IAdd, c->t_u32, s2, t3, t5);
    uint32_t t6 = nid(c), s3 = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, t6, s2, const_u32(c, 4));
    emit_arith(c, b, S_IAdd, c->t_u32, s3, s2, t6);
    uint32_t s4 = nid(c);
    emit_arith(c, b, S_And, c->t_u32, s4, s3, const_u32(c, 0x0f0f0f0fu));
    uint32_t t7 = nid(c), s5 = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, t7, s4, const_u32(c, 8));
    emit_arith(c, b, S_IAdd, c->t_u32, s5, s4, t7);
    uint32_t t8 = nid(c), s6 = nid(c), out = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, t8, s5, const_u32(c, 16));
    emit_arith(c, b, S_IAdd, c->t_u32, s6, s5, t8);
    emit_arith(c, b, S_And, c->t_u32, out, s6, const_u32(c, 0x3fu));
    return out;
}

// 32-bit count-leading-zeros via binary search from the top. The
// search alone stops at 31, so zero takes an explicit arm (32).
uint32_t clz32(ctx_t *c, spv_blob_t *b, uint32_t x, int line) {
    uint32_t n = const_u32(c, 0), v = x;
    const uint32_t masks[5] = { 0xffff0000u, 0xff000000u, 0xf0000000u,
                                0xc0000000u, 0x80000000u };
    const uint32_t adds[5] = { 16, 8, 4, 2, 1 };
    const uint32_t shifts[5] = { 16, 8, 4, 2, 1 };
    for (int i = 0; i < 5; i++) {
        uint32_t m = nid(c);
        emit_arith(c, b, S_And, c->t_u32, m, v, const_u32(c, masks[i]));
        uint32_t z = do_cmp(c, b, PCM_EQ, PTY_U32, 0, m, const_u32(c, 0), line);
        if (c->failed)
            return 0;
        uint32_t sh = nid(c), vs = nid(c);
        emit_arith(c, b, S_Shl, c->t_u32, sh, v, const_u32(c, shifts[i]));
        vs = sel_u32(c, b, z, sh, v);
        uint32_t na = nid(c);
        emit_arith(c, b, S_IAdd, c->t_u32, na, n, const_u32(c, adds[i]));
        n = sel_u32(c, b, z, na, n);
        v = vs;
    }
    uint32_t is_zero = do_cmp(c, b, PCM_EQ, PTY_U32, 0, x, const_u32(c, 0), line);
    if (c->failed)
        return 0;
    return sel_u32(c, b, is_zero, const_u32(c, 32), n);
}

void emit_push_load(ctx_t *c, spv_blob_t *b, uint32_t *id, ptx_ty_t ty,
                           int word, int line) {
    uint32_t zero = const_u32(c, 0);
    uint32_t widx = const_u32(c, (uint32_t)word);
    uint32_t ptr = nid(c);
    put(c, b, (6u << 16) | S_AccessChain);
    put(c, b, c->t_ptr_u_push);
    put(c, b, ptr);
    put(c, b, c->push_var);
    put(c, b, zero);
    put(c, b, widx);
    uint32_t w = nid(c);
    emit_load(c, b, c->t_u32, w, ptr);
    if (ty == PTY_F32) {
        uint32_t f = nid(c);
        put(c, b, (4u << 16) | S_Bitcast);
        put(c, b, c->t_f32);
        put(c, b, f);
        put(c, b, w);
        *id = f;
    } else {
        *id = w;
    }
    (void)line;
}

// Materialize a predicate operand (register, !register, or 0/1
// constant) as a bool SSA id. Shared by vote values and mov.pred.
// Constants land in the types section (graph-section constants are
// illegal); the LogicalNot for !register stays in the body.
uint32_t pred_val(ctx_t *c, spv_blob_t *b, const ptx_opd_t *op,
                         int neg, int line) {
    if (op->kind == OPD_IMM) {
        if (op->imm_is_float) {
            failf(c, line, "predicate value must be a register or integer");
            return 0;
        }
        uint32_t id = nid(c);
        // Integer immediates booleanize: 0 false, any nonzero true.
        int v = (op->imm != 0) ^ (neg != 0);
        put(c, &c->types, (3u << 16) | (v ? S_ConstantTrue : S_ConstantFalse));
        put(c, &c->types, c->t_bool);
        put(c, &c->types, id);
        return id;
    }
    if (op->kind != OPD_REG) {
        failf(c, line, "predicate value must be a register or 0/1");
        return 0;
    }
    reg_t *r = find_reg(c, op->reg);
    if (!r || !r->defined || r->decl_ty != PTY_PRED || r->is_ptr) {
        failf(c, line, "predicate operand must be a defined .pred register");
        return 0;
    }
    if (!neg)
        return r->cur;
    uint32_t id = nid(c);
    put(c, b, (4u << 16) | S_LogicalNot);
    put(c, b, c->t_bool);
    put(c, b, id);
    put(c, b, r->cur);
    return id;
}

// One 32-bit word of a ballot uvec4: the word of my warp,
// selected by warp index from constant extracts.
uint32_t ballot_warp_word(ctx_t *c, spv_blob_t *b, uint32_t bal,
                                 uint32_t W) {
    uint32_t wv[4];
    for (int wi = 0; wi < 4; wi++) {
        wv[wi] = nid(c);
        put(c, b, (5u << 16) | S_CompositeExtract);
        put(c, b, c->t_u32);
        put(c, b, wv[wi]);
        put(c, b, bal);
        put(c, b, (uint32_t)wi);
    }
    uint32_t word = wv[0];
    for (int wi = 1; wi < 4; wi++) {
        uint32_t cnd = nid(c), nv = nid(c);
        put(c, b, (5u << 16) | S_IEqual);
        put(c, b, c->t_bool);
        put(c, b, cnd);
        put(c, b, W);
        put(c, b, const_u32(c, (uint32_t)wi));
        put(c, b, (6u << 16) | S_Select);
        put(c, b, c->t_u32);
        put(c, b, nv);
        put(c, b, cnd);
        put(c, b, wv[wi]);
        put(c, b, word);
        word = nv;
    }
    return word;
}

