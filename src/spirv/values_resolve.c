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

uint32_t f32_to_f16_bits(uint32_t w);

void emit_store(ctx_t *c, spv_blob_t *b, uint32_t ptr, uint32_t val) {
    uint32_t al = phys_align_of(c, ptr);
    if (al) {
        put(c, b, (5u << 16) | S_Store);
        put(c, b, ptr);
        put(c, b, val);
        put(c, b, 0x2u); // Memory Operand: Aligned
        put(c, b, al);
        return;
    }
    put(c, b, (3u << 16) | S_Store);
    put(c, b, ptr);
    put(c, b, val);
}

void emit_arith(ctx_t *c, spv_blob_t *b, uint32_t op, uint32_t ty,
                       uint32_t id, uint32_t a, uint32_t c2) {
    put(c, b, (5u << 16) | op);
    put(c, b, ty);
    put(c, b, id);
    put(c, b, a);
    put(c, b, c2);
}

void emit_extract(ctx_t *c, spv_blob_t *b, uint32_t id, uint32_t vec,
                         uint32_t index) {
    put(c, b, (5u << 16) | S_CompositeExtract);
    put(c, b, c->t_u32);
    put(c, b, id);
    put(c, b, vec);
    put(c, b, index);
}

uint32_t special_comp(ctx_t *c, spv_blob_t *b, ptx_special_t sp,
                             int comp, int line) {
    if (sp == SPC_LANEID || sp == SPC_LANEMASKLT || sp == SPC_LANEMASKLE ||
        sp == SPC_LANEMASKGE || sp == SPC_LANEMASKGT || sp == SPC_LANEMASKEQ) {
        // Lane id within the 32-thread warp + lanes mask. Pure functions
        // of tid.x (exact, no divergence involved).
        uint32_t tid = special_comp(c, b, SPC_TID, 0, line);
        uint32_t lane = nid(c);
        emit_arith(c, b, S_And, c->t_u32, lane, tid, const_u32(c, 31));
        if (sp == SPC_LANEID)
            return lane;
        uint32_t sh = nid(c), m1 = nid(c);
        if (sp == SPC_LANEMASKGE) {
            // lanes >= lane: 0xffffffff << lane (shift 0..31).
            emit_arith(c, b, S_Shl, c->t_u32, m1, const_u32(c, 0xffffffffu),
                       lane);
            return m1;
        }
        if (sp == SPC_LANEMASKGT) {
            // lanes > lane: complement of the lanes-<= mask.
            uint32_t inv = nid(c);
            emit_arith(c, b, S_ISub, c->t_u32, inv, const_u32(c, 31), lane);
            emit_arith(c, b, S_ShrL, c->t_u32, sh, const_u32(c, 0xffffffffu),
                       inv);
            put(c, b, (4u << 16) | S_Not); // unary: OpNot
            put(c, b, c->t_u32);
            put(c, b, m1);
            put(c, b, sh);
            return m1;
        }
        if (sp == SPC_LANEMASKEQ) {
            emit_arith(c, b, S_Shl, c->t_u32, m1, const_u32(c, 1), lane);
            return m1;
        }
        if (sp == SPC_LANEMASKLE) {
            // lanes 0..lane: 0xffffffff >> (31 - lane). Written this way
            // so lane 31 never shifts by 32 (undefined in SPIR-V).
            uint32_t inv = nid(c);
            emit_arith(c, b, S_ISub, c->t_u32, inv, const_u32(c, 31), lane);
            emit_arith(c, b, S_ShrL, c->t_u32, m1, const_u32(c, 0xffffffffu),
                       inv);
            return m1;
        }
        emit_arith(c, b, S_Shl, c->t_u32, sh, const_u32(c, 1), lane);
        emit_arith(c, b, S_ISub, c->t_u32, m1, sh, const_u32(c, 1));
        return m1;
    }
    uint32_t base;
    if (sp == SPC_NTID) {
        base = c->v_ntid; // constant composite: extract directly, no load
    } else {
        uint32_t var = (sp == SPC_TID) ? c->v_tid
                     : (sp == SPC_NCTAID) ? c->v_nctaid
                     : c->v_ctaid;
        base = nid(c);
        emit_load(c, b, c->t_v3u32, base, var);
    }
    uint32_t id = nid(c);
    emit_extract(c, b, id, base, (uint32_t)comp);
    return id;
}

uint32_t special_x(ctx_t *c, spv_blob_t *b, ptx_special_t sp, int line) {
    return special_comp(c, b, sp, 0, line);
}

// Resolves an operand to an SSA id of the expected scalar type.
// Immediates must match: int literal -> .u32/.u64, float literal -> .f32.
// Under emulation f64 wants arrive as f32 (see eff_ty).
val_t resolve(ctx_t *c, spv_blob_t *b, const ptx_opd_t *op,
                     ptx_ty_t want, int line) {
    val_t v;
    memset(&v, 0, sizeof(v));
    want = eff_ty(c, want);
    v.ty = want;
    if (c->failed)
        return v;
    if (op->kind == OPD_UNDEF) {
        // Synthesized unknown value (predicated-desugar false arm
        // with no prior definition): Undef of the wanted type, by
        // the usual convention. Nothing else produces this kind.
        v.id = nid(c);
        put(c, b, (3u << 16) | 1); // OpUndef (grammar: 1)
        put(c, b, ty_of(c, want));
        put(c, b, v.id);
        return v;
    }
    if (op->kind == OPD_REGOFF) {
        // reg + uint value form (ZLUDA corpus: st operands like
        // `temp + 1`): materialize the addition. Integer types only;
        // the holder width must match (narrow holders add full-width,
        // consumers mask per the narrow discipline).
        if (want != PTY_U32 && want != PTY_U64 && want != PTY_U16 &&
            want != PTY_U8) {
            failf(c, line, "register-offset operand needs integer type");
            return v;
        }
        reg_t *r = find_reg(c, op->reg);
        if (!r || !r->defined || r->is_ptr) {
            failf(c, line, !r || !r->defined ? "use of undeclared register"
                                             : "pointer register used as scalar value");
            return v;
        }
        int dw = decl_int_width(r->decl_ty);
        int ww = want == PTY_U64 ? 64
                 : want == PTY_U32 ? 32
                 : want == PTY_U16 ? 16
                                   : 8;
        if (dw != ww) {
            failf(c, line, "register-offset width mismatch");
            return v;
        }
        v.id = nid(c);
        if (want == PTY_U64)
            emit_arith(c, b, S_IAdd, c->t_u64, v.id, r->cur,
                       const_u64(c, op->regoff_imm));
        else
            emit_arith(c, b, S_IAdd, c->t_u32, v.id, r->cur,
                       const_u32(c, (uint32_t)op->regoff_imm));
        return v;
    }
    if (op->kind == OPD_REG) {
        reg_t *r = find_reg(c, op->reg);
        if (!r) {
            failf(c, line, "use of undeclared register");
            return v;
        }
        if (!r->defined) {
            // PTX registers are uninitialized at declaration; reading
            // one yields an undefined value (ptxas accepts it). Model
            // that as OpUndef instead of rejecting the program.
            v.id = nid(c);
            put(c, b, (3u << 16) | 1); // OpUndef
            // f16 registers hold 16-bit patterns in u32 SSA (they are
            // widened explicitly by each consumer), unlike ty_of's
            // float alias.
            put(c, b, want == PTY_F16 ? c->t_u32 : ty_of(c, want));
            put(c, b, v.id);
            return v;
        }
        if (r->is_ptr) {
            if (r->binding == -3 && want == PTY_U64) {
                // PhysicalStorageBuffer pointers have a real 64-bit
                // address value (PTX pointer-as-integer idioms).
                v.id = r->off;
                return v;
            }
            if (want == PTY_U32 && r->cur_ty == PTY_U32 && r->cur) {
                // Materialized pointer word (lifted 32-bit
                // addressing, e.g. the low half of a .u64 pointer
                // parameter): the register carries a numeric word
                // alongside its pointer home.  Value uses read the
                // word; address uses still see the home through
                // resolve_addr_ex.
                v.id = r->cur;
                v.ty = PTY_U32;
                return v;
            }
            if (getenv("ACUDA_DEBUG_RESOLVE"))
                fprintf(stderr, "ACuda resolve fail: reg=%s bind=%d want=%d decl=%d line=%d cfg=%d host=%d dyn=%d\n",
                        op->reg, r->binding, (int)want, (int)r->decl_ty, line,
                        c->cfg_machine, c->host_machine, r->dyn_home);
            failf(c, line, "pointer register used as scalar value");
            return v;
        }
        // Half-bit patterns live interchangeably in .u16 and .f16
        // registers (the numba b16 idiom); packed pairs live
        // interchangeably in .b32 and .f16x2/.bf16x2 registers;
        // bf16 patterns in .b16/.bf16. Every other width needs an
        // exact declaration match.
        int ok = r->decl_ty == want;
        if (!ok && want == PTY_U16 &&
            (r->decl_ty == PTY_F16 || r->decl_ty == PTY_BF16))
            ok = 1;
        if (!ok && want == PTY_F16 &&
            (r->decl_ty == PTY_U16))
            ok = 1;
        if (!ok && want == PTY_BF16 &&
            (r->decl_ty == PTY_U16))
            ok = 1;
        if (!ok && want == PTY_F16X2 &&
            (r->decl_ty == PTY_U32))
            ok = 1;
        if (!ok && want == PTY_BF16X2 &&
            (r->decl_ty == PTY_U32))
            ok = 1;
        if (!ok && want == PTY_U32 &&
            (r->decl_ty == PTY_F16X2 || r->decl_ty == PTY_BF16X2))
            ok = 1;
        if (!ok && want == PTY_F32 && r->decl_ty == PTY_U64) {
            // f32 bits in a 64-bit holder (ld.f32 into .b64 reads the
            // low word): truncate + bitcast to float type so all float
            // consumers keep working.
            uint32_t t = nid(c), f = nid(c);
            put(c, b, (4u << 16) | S_UConvert);
            put(c, b, c->t_u32);
            put(c, b, t);
            put(c, b, r->cur);
            put(c, b, (4u << 16) | S_Bitcast);
            put(c, b, c->t_f32);
            put(c, b, f);
            put(c, b, t);
            v.id = f;
            return v;
        }
        // Bitwise moves between same-width integer/float types (e.g. mov.b64
        // between .b64/.u64 and .f64). Bitcast in SPIR-V.
        if (!ok) {
            int samew = 0;
            uint32_t dst_t = 0;
            if ((want == PTY_U64 && r->decl_ty == PTY_F64) ||
                (want == PTY_F64 && r->decl_ty == PTY_U64)) {
                samew = 1;
                dst_t = (want == PTY_U64) ? c->t_u64 : c->t_f64;
            } else if ((want == PTY_U32 && r->decl_ty == PTY_F32) ||
                       (want == PTY_F32 && r->decl_ty == PTY_U32)) {
                samew = 1;
                dst_t = (want == PTY_U32) ? c->t_u32 : c->t_f32;
            }
            if (samew) {
                uint32_t out = nid(c);
                put(c, b, (4u << 16) | S_Bitcast);
                put(c, b, dst_t);
                put(c, b, out);
                put(c, b, r->cur);
                v.id = out;
                return v;
            }
        }
        if (!ok) {
            failf(c, line, "register type mismatch");
            return v;
        }
        // The register's declaration may be a bitwise alias of the
        // type its current SSA value was produced with (e.g. fma.f32
        // into a .b32 register, then st.u32). Bitcast to the wanted
        // view instead of handing a float-typed id to an integer use.
        if (r->cur_ty != want) {
            uint32_t bt = 0;
            if ((want == PTY_U32 && r->cur_ty == PTY_F32) ||
                (want == PTY_F32 && r->cur_ty == PTY_U32))
                bt = want == PTY_U32 ? c->t_u32 : c->t_f32;
            else if ((want == PTY_U64 && r->cur_ty == PTY_F64) ||
                     (want == PTY_F64 && r->cur_ty == PTY_U64))
                bt = want == PTY_U64 ? c->t_u64 : c->t_f64;
            if (bt) {
                uint32_t out = nid(c);
                put(c, b, (4u << 16) | S_Bitcast);
                put(c, b, bt);
                put(c, b, out);
                put(c, b, r->cur);
                v.id = out;
                return v;
            }
        }
        v.id = r->cur;
        return v;
    }
    if (op->kind == OPD_SPECIAL) {
        if (want != PTY_U32) {
            failf(c, line, "special registers are .u32");
            return v;
        }
        v.id = special_comp(c, b, op->special, op->comp, line);
        return v;
    }
    if (op->kind == OPD_IMM) {
        if ((want == PTY_F32 || want == PTY_F64) && !op->imm_is_float) {
            failf(c, line, "integer literal for float operand (use explicit constant)");
            return v;
        }
        if (want != PTY_F32 && want != PTY_F64 && op->imm_is_float) {
            failf(c, line, "float literal for integer operand");
            return v;
        }
        if (want == PTY_F64) {
            uint64_t bits = 0;
            if (op->imm_text[0] == '0' &&
                (op->imm_text[1] == 'D' || op->imm_text[1] == 'd')) {
                bits = op->imm; // lexer already validated 16 hex digits
            } else if (op->imm_text[0] == '0' &&
                       (op->imm_text[1] == 'F' || op->imm_text[1] == 'f')) {
                float f;
                uint32_t fb = (uint32_t)op->imm;
                memcpy(&f, &fb, 4);
                double d = (double)f; // exact: float -> double widens
                memcpy(&bits, &d, 8);
            } else {
                char *end = NULL;
                double d = strtod(op->imm_text, &end);
                if (!end || *end) {
                    failf(c, line, "bad double literal");
                    return v;
                }
                memcpy(&bits, &d, 8);
            }
            v.id = const_f64(c, bits);
            return v;
        }
        if (want == PTY_U32) {
            // ptxas silently truncates out-of-range integer literals
            // (verified: mov.u32 with 5000000000 assembles); match it.
            v.id = const_u32(c, (uint32_t)op->imm);
            return v;
        }
        if (want == PTY_U64) {
            v.id = const_u64(c, op->imm);
            return v;
        }
        if (want == PTY_F16X2 || want == PTY_BF16X2) {
            // Packed immediates are raw 32-bit patterns (ptxas
            // truncation parity, like .u32).
            v.id = const_u32(c, (uint32_t)op->imm);
            return v;
        }
        if (want == PTY_U8 || want == PTY_U16) {
            // Narrow literals truncate like ptxas (verified: mov.u16
            // with 70000 assembles); canonicalized to low bits.
            uint64_t lim = want == PTY_U8 ? 0xffu : 0xffffu;
            v.id = const_u32(c, (uint32_t)(op->imm & lim));
            return v;
        }
        if (want == PTY_F16) {
            // f16 values are 16-bit patterns: round the literal to
            // half at compile time (RNE). Unreachable from valid PTX
            // (no f16 op takes immediates), kept for uniformity.
            uint32_t hb = 0;
            if (!op->imm_is_float) {
                failf(c, line, "integer literal for float operand (use explicit constant)");
                return v;
            }
            if (op->imm_text[0] == '0' &&
                (op->imm_text[1] == 'H' || op->imm_text[1] == 'h')) {
                hb = (uint32_t)op->imm;
            } else if (op->imm_text[0] == '0' &&
                       (op->imm_text[1] == 'D' || op->imm_text[1] == 'd')) {
                double d;
                memcpy(&d, &op->imm, 8);
                float f = (float)d;
                uint32_t fb;
                memcpy(&fb, &f, 4);
                hb = f32_to_f16_bits(fb);
            } else {
                uint32_t fb = 0;
                if (op->imm_text[0] == '0' &&
                    (op->imm_text[1] == 'F' || op->imm_text[1] == 'f')) {
                    fb = (uint32_t)op->imm;
                } else {
                    char *end = NULL;
                    float f = strtof(op->imm_text, &end);
                    if (!end || *end) {
                        failf(c, line, "bad float literal");
                        return v;
                    }
                    memcpy(&fb, &f, 4);
                }
                hb = f32_to_f16_bits(fb);
            }
            v.id = const_u32(c, hb);
            return v;
        }
        if (op->imm_text[0] == '0' &&
            (op->imm_text[1] == 'D' || op->imm_text[1] == 'd')) {
            // 0D hex-double narrowed to float (emulation or f32 context).
            double d;
            memcpy(&d, &op->imm, 8);
            float f = (float)d;
            uint32_t bits;
            memcpy(&bits, &f, 4);
            v.id = const_f32(c, bits);
            return v;
        }
        v.id = const_f32(c, (uint32_t)op->imm);
        return v;
    }
    failf(c, line, "bad operand kind for scalar position");
    return v;
}

// Runtime token for a pointer used as an integer value.  The high 32
// bits carry the home tag (descriptor binding, or the shared/global-
// image/parameter sentinels) and the low 32 bits the byte offset.
// This lets pointer tables live in ordinary memory (e.g. shared) and
// be dereferenced dynamically; it matches the home tags used by the
// host-call dynamic-pointer dispatcher.
uint32_t ptr_token_u64(ctx_t *c, spv_blob_t *b, const reg_t *r, int line) {
    (void)line;
    if (r->binding == -3)
        return r->off ? r->off : const_u64(c, 0);
    uint32_t tag;
    if (r->binding >= 0)
        tag = const_u32(c, (uint32_t)r->binding);
    else if (r->shvar >= 0)
        tag = const_u32(c, 0xfffffffeu);
    else if (r->glvar >= 0)
        tag = const_u32(c, 0xfffffffbu);
    else if (r->paddr >= 0)
        tag = const_u32(c, 0xffff0000u | (uint32_t)r->paddr);
    else
        tag = const_u32(c, 0xffffffffu);
    uint32_t tag64 = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u64);
    put(c, b, tag64);
    put(c, b, tag);
    uint32_t hi = nid(c);
    emit_arith(c, b, S_Shl, c->t_u64, hi, tag64, const_u64(c, 32));
    uint32_t off = r->off ? r->off : const_u64(c, 0);
    uint32_t low = nid(c);
    emit_arith(c, b, S_And, c->t_u64, low, off,
               const_u64(c, 0xffffffffull));
    uint32_t tok = nid(c);
    emit_arith(c, b, S_Or, c->t_u64, tok, hi, low);
    return tok;
}

// Resolves an address operand to (SSBO binding, u64 byte-offset SSA),
// (shared variable, byte-offset SSA, *shvar >= 0), or (.global image
// variable, byte-offset SSA, *glvar >= 0).
uint32_t addr_ptr(ctx_t *c, spv_blob_t *b, int binding, uint32_t off);
uint32_t gl_addr_ptr(ctx_t *c, spv_blob_t *b, int var, uint32_t off);
uint32_t sh_addr_ptr(ctx_t *c, spv_blob_t *b, int var, uint32_t off);
uint32_t lo_addr_ptr(ctx_t *c, spv_blob_t *b, int var, uint32_t off);

int resolve_addr_ex(ctx_t *c, spv_blob_t *b, const ptx_opd_t *op,
                     int *binding, uint32_t *off, int *shvar, int *glvar,
                     int64_t *sh_imm, int line) {
    if (sh_imm)
        *sh_imm = 0;
    c->last_local = -1;
    if (op->kind != OPD_ADDR) {
        failf(c, line, "address must use []");
        return -1;
    }
    if (op->addr_reg[0]) {
        reg_t *r = find_reg(c, op->addr_reg);
        if (!r || !r->defined) {
            char _em[160];
            snprintf(_em, sizeof(_em), "address must be a pointer register (%s %s)",
                     op->addr_reg, !r ? "undeclared" : "undefined");
            failf(c, line, _em);
            return -1;
        }
        if (c->use_bda && !r->is_ptr && r->decl_ty == PTY_U64) {
            // PhysicalStorageBuffer mode: a plain u64 used as a global
            // address is a real device address (not a binding token).
            *binding = -3;
            *off = r->cur;
            *shvar = -1;
            *glvar = -1;
            if (op->addr_imm != 0) {
                uint32_t id = nid(c);
                emit_arith(c, b, S_IAdd, c->t_u64, id, *off,
                           const_u64(c, (uint64_t)op->addr_imm));
                *off = id;
            }
            return 0;
        }
        if (!r->is_ptr && c->addr_size == 32 &&
            (r->decl_ty == PTY_U32 || r->decl_ty == PTY_U64)) {
            // 32-bit PTX: pointer values are plain integers (stored/loaded
            // through memory, passed as u32 fields). Treat as a raw global
            // device address (BDA-style, binding=-3) so generic/global ld/st
            // can proceed. (Scoping to 32-bit avoids breaking 64-bit cases.)
            uint32_t base = r->cur ? r->cur : const_u64(c, 0);
            if (r->decl_ty == PTY_U32 && r->cur) {
                uint32_t wide = nid(c);
                put(c, b, (4u << 16) | S_UConvert);
                put(c, b, c->t_u64);
                put(c, b, wide);
                put(c, b, r->cur);
                base = wide;
            }
            *binding = -3;
            *off = base;
            *shvar = -1;
            *glvar = -1;
            if (op->addr_imm != 0) {
                uint32_t id = nid(c);
                emit_arith(c, b, S_IAdd, c->t_u64, id, *off,
                           const_u64(c, (uint64_t)op->addr_imm));
                *off = id;
            }
            return 0;
        }
        if (!r->is_ptr) {
            char _em[256];
            snprintf(_em, sizeof(_em), "address must be a pointer register (%s def=%d ptr=%d ty=%d bind=%d sh=%d gl=%d as=%d off=%llx)",
                     op->addr_reg, r->defined, r->is_ptr, (int)r->decl_ty,
                     r->binding, r->shvar, r->glvar, c->addr_size,
                     (unsigned long long)r->off);
            failf(c, line, _em);
            return -1;
        }
        *binding = r->binding;
        *off = r->off;
        *shvar = r->shvar;
        *glvar = r->glvar;
        if (r->binding == -2) {
            c->last_local = r->lovar;
            if (getenv("ACUDA_TRACE"))
                fprintf(stderr, "acuda-local-trace resolve %s: binding=-2 lovar=%d is_ptr=%d\n",
                        op->addr_reg, r->lovar, r->is_ptr);
        }
        if (op->addr_imm != 0) {
            if (sh_imm && *shvar >= 0) {
                // Keep the byte offset out of the 64-bit pointer chain:
                // the shared access helpers add it to a 32-bit word
                // index, where it can fold into the LDS offset.
                *sh_imm = op->addr_imm;
            } else {
                uint32_t id = nid(c);
                emit_arith(c, b, S_IAdd, c->t_u64, id, *off,
                           const_u64(c, (uint64_t)op->addr_imm));
                *off = id;
            }
        }
        return 0;
    }
    if (op->param[0]) {
        // Direct module-variable use ([var]/[var+N], shared or
        // .global): the offset folds the constant part.
        int sv = find_shvar(c, op->param);
        if (sv >= 0) {
            *binding = -1;
            *off = const_u64(c, (uint64_t)op->addr_imm);
            *shvar = sv;
            *glvar = -1;
            return 0;
        }
        int gv = find_global(c, op->param);
        if (gv >= 0) {
            *binding = -1;
            *off = const_u64(c, (uint64_t)op->addr_imm);
            *shvar = -1;
            *glvar = gv;
            return 0;
        }
        int lv = find_local(c, op->param);
        if (lv >= 0) {
            *binding = -2;
            *off = const_u64(c, (uint64_t)op->addr_imm);
            *shvar = -1;
            *glvar = -1;
            c->last_local = lv;
            return 0;
        }
    }
    failf(c, line, "address must be a register (param base unsupported)");
    return -1;
}

int check_dst(ctx_t *c, const char *dst, ptx_ty_t ty, int line) {
    ty = eff_ty(c, ty);
    reg_t *r = find_reg(c, dst);
    if (!r) {
        failf(c, line, "assignment to undeclared register");
        return -1;
    }
    // Halves live as 16-bit patterns in .f16 or .u16 declarations;
    // bf16 the same in .bf16/.u16; packed pairs in .f16x2/.bf16x2 or
    // plain .b32 declarations.
    if (ty == PTY_F16) {
        if (r->decl_ty != PTY_F16 && r->decl_ty != PTY_U16) {
            failf(c, line, "type mismatch on assignment (want .f16/.u16)");
            return -1;
        }
        return 0;
    }
    if (ty == PTY_BF16) {
        if (r->decl_ty != PTY_BF16 && r->decl_ty != PTY_U16) {
            failf(c, line, "type mismatch on assignment (want .bf16/.u16)");
            return -1;
        }
        return 0;
    }
    if (ty == PTY_F16X2) {
        if (r->decl_ty != PTY_F16X2 && r->decl_ty != PTY_U32) {
            failf(c, line, "type mismatch on assignment (want .f16x2/.b32)");
            return -1;
        }
        return 0;
    }
    if (ty == PTY_BF16X2) {
        if (r->decl_ty != PTY_BF16X2 && r->decl_ty != PTY_U32) {
            failf(c, line, "type mismatch on assignment (want .bf16x2/.b32)");
            return -1;
        }
        return 0;
    }
    if (r->decl_ty != ty) {
        // Allow bitwise moves between same-width integer/float types
        if ((ty == PTY_U64 && r->decl_ty == PTY_F64) ||
            (ty == PTY_F64 && r->decl_ty == PTY_U64) ||
            (ty == PTY_U32 && r->decl_ty == PTY_F32) ||
            (ty == PTY_F32 && r->decl_ty == PTY_U32) ||
            (ty == PTY_U16 && r->decl_ty == PTY_F16) ||
            (ty == PTY_U16 && r->decl_ty == PTY_BF16) ||
            (ty == PTY_U32 && r->decl_ty == PTY_F16X2) ||
            (ty == PTY_U32 && r->decl_ty == PTY_BF16X2)) {
            return 0;
        }
        failf(c, line, "type mismatch on assignment");
        return -1;
    }
    return 0;
}

void def_val(ctx_t *c, const char *dst, ptx_ty_t ty, uint32_t id) {
    reg_t *r = find_reg(c, dst);
    r->defined = 1;
    r->cur = id;
    r->cur_ty = ty;
    r->cur_is_coopmat = 0;
    r->is_ptr = 0;
    r->binding = -1;
    r->shvar = -1;
    r->glvar = -1;
    r->paddr = -1;
    r->dyn_home = 0;
    r->is_tex = 0;
    (void)ty;
}

void def_ptr(ctx_t *c, const char *dst, int binding, uint32_t off) {
    if (getenv("ACUDA_DEBUG_SYMBOLS") && dst && strcmp(dst, "%rd3") == 0)
        fprintf(stderr, "ACuda: def_ptr %s bind=%d off=%u\n", dst, binding, off);
    reg_t *r = find_reg(c, dst);
    r->defined = 1;
    r->cur = 0;
    r->cur_ty = PTY_U64;
    r->is_ptr = 1;
    r->binding = binding;
    r->off = off;
    r->shvar = -1;
    r->glvar = -1;
    r->paddr = -1;
    r->dyn_home = 0;
    r->is_tex = 0;
}

// Defines dst as the symbolic address of kernel parameter pidx.
void def_paddr(ctx_t *c, const char *dst, int pidx) {
    if (getenv("ACUDA_DEBUG_SYMBOLS") && dst && strcmp(dst, "%rd3") == 0)
        fprintf(stderr, "ACuda: def_paddr %s pidx=%d\n", dst, pidx);
    reg_t *r = find_reg(c, dst);
    r->defined = 1;
    r->cur = 0;
    r->is_ptr = 1;
    r->binding = -1;
    r->off = 0;
    r->shvar = -1;
    r->glvar = -1;
    r->paddr = pidx;
    r->dyn_home = 0;
    r->is_tex = 0;
}

// Defines dst as (Workgroup shared variable var, byte offset off).
void def_shared(ctx_t *c, const char *dst, int var, uint32_t off) {
    reg_t *r = find_reg(c, dst);
    r->defined = 1;
    r->cur = 0;
    r->cur_ty = PTY_U64;
    r->is_ptr = 1;
    r->binding = -1;
    r->off = off;
    r->shvar = var;
    r->glvar = -1;
    r->paddr = -1;
    r->dyn_home = 0;
    r->is_tex = 0;
}

// Defines dst as (.global image variable var, byte offset off).
// Reads serve the init bytes; the image has no device backing.
void def_global(ctx_t *c, const char *dst, int var, uint32_t off) {
    reg_t *r = find_reg(c, dst);
    r->defined = 1;
    r->cur = 0;
    r->cur_ty = PTY_U64;
    r->is_ptr = 1;
    r->binding = -1;
    r->off = off;
    r->shvar = -1;
    r->glvar = var;
    r->paddr = -1;
    r->dyn_home = 0;
}


// PhysicalStorageBuffer address (u64) -> typed physical pointer.
uint32_t bda_convert(ctx_t *c, spv_blob_t *b, uint32_t pty,
                            uint32_t addr) {
    if (pty == 0) {
        // Missing physical-pointer type (emitted only when the module
        // needs BDA).  Fail loudly instead of writing an instruction
        // with a null type id, which crashes the vendor compiler on
        // pipeline creation.
        fail(c, "bda_convert with null pointer type");
        return 0;
    }
    uint32_t ptr = nid(c);
    put(c, b, (4u << 16) | S_ConvertUToPtr);
    put(c, b, pty);
    put(c, b, ptr);
    put(c, b, addr);
    if (c->nphys < (int)(sizeof(c->phys_ptrs) / sizeof(c->phys_ptrs[0]))) {
        c->phys_ptrs[c->nphys] = ptr;
        c->phys_align[c->nphys] =
            pty == c->t_ptr_phys_u64 ? 8u
            : pty == c->t_ptr_phys_u16 ? 2u
                                       : 4u;
        c->nphys++;
    }
    return ptr;
}

uint32_t u16_type(ctx_t *c) {
    if (!c->t_u16) {
        c->t_u16 = nid(c);
        put(c, &c->types, (4u << 16) | S_TypeInt);
        put(c, &c->types, c->t_u16);
        put(c, &c->types, 16);
        put(c, &c->types, 0);
    }
    return c->t_u16;
}

uint32_t phys_u16_ty(ctx_t *c) {
    if (!c->t_u16) {
        c->t_u16 = nid(c);
        put(c, &c->types, (4u << 16) | S_TypeInt);
        put(c, &c->types, c->t_u16);
        put(c, &c->types, 16);
        put(c, &c->types, 0);
    }
    if (!c->t_ptr_phys_u16) {
        c->t_ptr_phys_u16 = nid(c);
        put(c, &c->types, (4u << 16) | S_TypePointer);
        put(c, &c->types, c->t_ptr_phys_u16);
        put(c, &c->types, SC_PhysicalStorageBuffer);
        put(c, &c->types, c->t_u16);
    }
    return c->t_ptr_phys_u16;
}

uint32_t phys_align_of(ctx_t *c, uint32_t ptr) {
    for (int i = c->nphys - 1; i >= 0; i--)
        if (c->phys_ptrs[i] == ptr)
            return c->phys_align[i];
    return 0;
}

// Aligned u32 physical pointer + lane shift for byte/half accesses.
uint32_t bda_word_ptr(ctx_t *c, spv_blob_t *b, uint32_t addr,
                             uint32_t *sh) {
    uint32_t base = nid(c), lane = nid(c), lane32 = nid(c);
    emit_arith(c, b, S_And, c->t_u64, base, addr,
               const_u64(c, ~(uint64_t)3));
    emit_arith(c, b, S_And, c->t_u64, lane, addr, const_u64(c, 3));
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u32);
    put(c, b, lane32);
    put(c, b, lane);
    *sh = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, *sh, lane32, const_u32(c, 3));
    return bda_convert(c, b, c->t_ptr_phys_u32, base);
}

uint32_t load_u64(ctx_t *c, spv_blob_t *b, int binding, uint32_t off) {
    if (binding == -3) {
        uint32_t ptr = bda_convert(c, b, c->t_ptr_phys_u64, off);
        uint32_t id = nid(c);
        emit_load(c, b, c->t_u64, id, ptr);
        return id;
    }
    // Little-endian pair: lo = words[off>>2], hi = words[(off>>2)+1].
    uint32_t word = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, word, off, const_u64(c, 2));
    uint32_t zero = const_u32(c, 0);
    uint32_t p0 = nid(c), p1 = nid(c);
    put(c, b, (6u << 16) | S_AccessChain);
    put(c, b, c->t_ptr_u_ssbo);
    put(c, b, p0);
    put(c, b, c->ssbo_vars[binding]);
    put(c, b, zero);
    put(c, b, word);
    // Second index needs 64-bit word+1: recompute via separate chain.
    // (AccessChain indices must match sequentially; use an IAdd u64.)
    // NOTE: the IAdd must come BEFORE the chain header (it is its own
    // 5-word instruction, not part of the chain's operands).
    uint32_t word1 = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, word1, word, const_u64(c, 1));
    put(c, b, (6u << 16) | S_AccessChain);
    put(c, b, c->t_ptr_u_ssbo);
    put(c, b, p1);
    put(c, b, c->ssbo_vars[binding]);
    put(c, b, zero);
    put(c, b, word1);
    uint32_t lo = nid(c), hi = nid(c);
    emit_load(c, b, c->t_u32, lo, p0);
    emit_load(c, b, c->t_u32, hi, p1);
    uint32_t lo64 = nid(c), hi64 = nid(c), sh = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u64);
    put(c, b, lo64);
    put(c, b, lo);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u64);
    put(c, b, hi64);
    put(c, b, hi);
    put(c, b, (5u << 16) | S_Shl);
    put(c, b, c->t_u64);
    put(c, b, sh);
    put(c, b, hi64);
    put(c, b, const_u64(c, 32));
    uint32_t id = nid(c);
    emit_arith(c, b, S_Or, c->t_u64, id, sh, lo64);
    return id;
}

void store_u64(ctx_t *c, spv_blob_t *b, int binding, uint32_t off,
                      uint32_t val) {
    if (binding == -3) {
        uint32_t ptr = bda_convert(c, b, c->t_ptr_phys_u64, off);
        emit_store(c, b, ptr, val);
        return;
    }
    uint32_t word = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, word, off, const_u64(c, 2));
    uint32_t zero = const_u32(c, 0);
    uint32_t p0 = nid(c), p1 = nid(c);
    put(c, b, (6u << 16) | S_AccessChain);
    put(c, b, c->t_ptr_u_ssbo);
    put(c, b, p0);
    put(c, b, c->ssbo_vars[binding]);
    put(c, b, zero);
    put(c, b, word);
    uint32_t word1 = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, word1, word, const_u64(c, 1));
    put(c, b, (6u << 16) | S_AccessChain);
    put(c, b, c->t_ptr_u_ssbo);
    put(c, b, p1);
    put(c, b, c->ssbo_vars[binding]);
    put(c, b, zero);
    put(c, b, word1);
    uint32_t lo = nid(c), hi = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u32);
    put(c, b, lo);
    put(c, b, val);
    emit_arith(c, b, S_ShrL, c->t_u64, hi, val, const_u64(c, 32));
    uint32_t hi32 = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u32);
    put(c, b, hi32);
    put(c, b, hi);
    emit_store(c, b, p0, lo);
    emit_store(c, b, p1, hi32);
}

// ---- f64 emulation converts (integer-only, no Float64 required) ----

