// Atomic read-modify-write opcodes (POP_ATOM / POP_ATOMCAS).
#include <stdint.h>
#include "spirv_internal.h"

// Half-word lane shift (0 or 16) from a byte offset.
static uint32_t atom_lane_shift(ctx_t *c, spv_blob_t *b, uint32_t off,
                                int line) {
    (void)line;
    uint32_t bit = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, bit, off, const_u64(c, 1));
    uint32_t bit32 = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u32);
    put(c, b, bit32);
    put(c, b, bit);
    uint32_t one = nid(c);
    emit_arith(c, b, S_And, c->t_u32, one, bit32, const_u32(c, 1));
    uint32_t sh = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, sh, one, const_u32(c, 4));
    return sh;
}

// 16-bit CAS on the containing u32 word, preserving the other half.
static uint32_t atom_cas16_loop(ctx_t *c, spv_blob_t *b, uint32_t ptr,
                                int shared, uint32_t sh, uint32_t cmp,
                                uint32_t newv) {
    uint32_t scope = const_u32(c, shared ? 2 : 1);
    uint32_t sem_eq = const_u32(c, shared ? 0x108u : 0x48u);
    uint32_t sem_ne = const_u32(c, shared ? 0x102u : 0x42u);
    uint32_t header = nid(c), merge = nid(c), cont = nid(c);
    emit_branch(c, b, header);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, header);
    uint32_t old = nid(c);
    emit_load(c, b, c->t_u32, old, ptr);
    uint32_t half = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, half, old, sh);
    uint32_t h = nid(c);
    emit_arith(c, b, S_And, c->t_u32, h, half, const_u32(c, 0xffffu));
    uint32_t eq = nid(c);
    put(c, b, (5u << 16) | S_IEqual);
    put(c, b, c->t_bool);
    put(c, b, eq);
    put(c, b, h);
    put(c, b, cmp);
    uint32_t nh = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, c->t_u32);
    put(c, b, nh);
    put(c, b, eq);
    put(c, b, newv);
    put(c, b, h);
    uint32_t mask0 = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, mask0, const_u32(c, 0xffffu), sh);
    uint32_t inv = nid(c);
    put(c, b, (4u << 16) | S_Not);
    put(c, b, c->t_u32);
    put(c, b, inv);
    put(c, b, mask0);
    uint32_t keep = nid(c);
    emit_arith(c, b, S_And, c->t_u32, keep, old, inv);
    uint32_t ins = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, ins, nh, sh);
    uint32_t merged = nid(c);
    emit_arith(c, b, S_Or, c->t_u32, merged, keep, ins);
    uint32_t prev = nid(c);
    put(c, b, (9u << 16) | S_AtomicCAS);
    put(c, b, c->t_u32);
    put(c, b, prev);
    put(c, b, ptr);
    put(c, b, scope);
    put(c, b, sem_eq);
    put(c, b, sem_ne);
    put(c, b, merged);
    put(c, b, old);
    uint32_t ok = nid(c);
    put(c, b, (5u << 16) | S_IEqual);
    put(c, b, c->t_bool);
    put(c, b, ok);
    put(c, b, prev);
    put(c, b, old);
    put(c, b, (4u << 16) | S_LoopMerge);
    put(c, b, merge);
    put(c, b, cont);
    put(c, b, 0);
    put(c, b, (4u << 16) | S_BranchCond);
    put(c, b, ok);
    put(c, b, merge);
    put(c, b, cont);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, cont);
    emit_branch(c, b, header);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, merge);
    return prev;
}

// Scalar f16/bf16 add on the containing u32 word.
static uint32_t atom_half_add_loop(ctx_t *c, spv_blob_t *b, uint32_t ptr,
                                   int shared, uint32_t sh, int is_bf,
                                   uint32_t add, int line) {
    uint32_t scope = const_u32(c, shared ? 2 : 1);
    uint32_t sem_eq = const_u32(c, shared ? 0x108u : 0x48u);
    uint32_t sem_ne = const_u32(c, shared ? 0x102u : 0x42u);
    uint32_t header = nid(c), merge = nid(c), cont = nid(c);
    emit_branch(c, b, header);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, header);
    uint32_t old = nid(c);
    emit_load(c, b, c->t_u32, old, ptr);
    uint32_t half = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, half, old, sh);
    uint32_t hp = nid(c);
    emit_arith(c, b, S_And, c->t_u32, hp, half, const_u32(c, 0xffffu));
    uint32_t f = is_bf ? bf16_to_f32val(c, b, hp, line)
                       : f16_to_f32val(c, b, hp, line);
    uint32_t nf = nid(c);
    emit_arith(c, b, S_FAdd, c->t_f32, nf, f, add);
    uint32_t nb = is_bf ? f32_to_bf16bits(c, b, nf, line)
                        : f32_to_f16bits(c, b, nf, line);
    uint32_t mask0 = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, mask0, const_u32(c, 0xffffu), sh);
    uint32_t inv = nid(c);
    put(c, b, (4u << 16) | S_Not);
    put(c, b, c->t_u32);
    put(c, b, inv);
    put(c, b, mask0);
    uint32_t keep = nid(c);
    emit_arith(c, b, S_And, c->t_u32, keep, old, inv);
    uint32_t ins = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, ins, nb, sh);
    uint32_t merged = nid(c);
    emit_arith(c, b, S_Or, c->t_u32, merged, keep, ins);
    uint32_t prev = nid(c);
    put(c, b, (9u << 16) | S_AtomicCAS);
    put(c, b, c->t_u32);
    put(c, b, prev);
    put(c, b, ptr);
    put(c, b, scope);
    put(c, b, sem_eq);
    put(c, b, sem_ne);
    put(c, b, merged);
    put(c, b, old);
    uint32_t ok = nid(c);
    put(c, b, (5u << 16) | S_IEqual);
    put(c, b, c->t_bool);
    put(c, b, ok);
    put(c, b, prev);
    put(c, b, old);
    put(c, b, (4u << 16) | S_LoopMerge);
    put(c, b, merge);
    put(c, b, cont);
    put(c, b, 0);
    put(c, b, (4u << 16) | S_BranchCond);
    put(c, b, ok);
    put(c, b, merge);
    put(c, b, cont);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, cont);
    emit_branch(c, b, header);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, merge);
    return prev;
}

// Packed f16x2/bf16x2 add on the whole u32 word.
static uint32_t atom_packed_add_loop(ctx_t *c, spv_blob_t *b, uint32_t ptr,
                                     int shared, int is_bf, uint32_t add,
                                     int line) {
    uint32_t scope = const_u32(c, shared ? 2 : 1);
    uint32_t sem_eq = const_u32(c, shared ? 0x108u : 0x48u);
    uint32_t sem_ne = const_u32(c, shared ? 0x102u : 0x42u);
    uint32_t header = nid(c), merge = nid(c), cont = nid(c);
    emit_branch(c, b, header);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, header);
    uint32_t old = nid(c);
    emit_load(c, b, c->t_u32, old, ptr);
    uint32_t lo = nid(c), hi = nid(c);
    emit_arith(c, b, S_And, c->t_u32, lo, old, const_u32(c, 0xffffu));
    emit_arith(c, b, S_ShrL, c->t_u32, hi, old, const_u32(c, 16));
    uint32_t alo = nid(c), ahi = nid(c);
    emit_arith(c, b, S_And, c->t_u32, alo, add, const_u32(c, 0xffffu));
    emit_arith(c, b, S_ShrL, c->t_u32, ahi, add, const_u32(c, 16));
    uint32_t flo = is_bf ? bf16_to_f32val(c, b, lo, line)
                         : f16_to_f32val(c, b, lo, line);
    uint32_t fhi = is_bf ? bf16_to_f32val(c, b, hi, line)
                         : f16_to_f32val(c, b, hi, line);
    uint32_t aflo = is_bf ? bf16_to_f32val(c, b, alo, line)
                          : f16_to_f32val(c, b, alo, line);
    uint32_t afhi = is_bf ? bf16_to_f32val(c, b, ahi, line)
                          : f16_to_f32val(c, b, ahi, line);
    uint32_t nflo = nid(c), nfhi = nid(c);
    emit_arith(c, b, S_FAdd, c->t_f32, nflo, flo, aflo);
    emit_arith(c, b, S_FAdd, c->t_f32, nfhi, fhi, afhi);
    uint32_t nlo = is_bf ? f32_to_bf16bits(c, b, nflo, line)
                         : f32_to_f16bits(c, b, nflo, line);
    uint32_t nhi = is_bf ? f32_to_bf16bits(c, b, nfhi, line)
                         : f32_to_f16bits(c, b, nfhi, line);
    uint32_t hish = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, hish, nhi, const_u32(c, 16));
    uint32_t merged = nid(c);
    emit_arith(c, b, S_Or, c->t_u32, merged, hish, nlo);
    uint32_t prev = nid(c);
    put(c, b, (9u << 16) | S_AtomicCAS);
    put(c, b, c->t_u32);
    put(c, b, prev);
    put(c, b, ptr);
    put(c, b, scope);
    put(c, b, sem_eq);
    put(c, b, sem_ne);
    put(c, b, merged);
    put(c, b, old);
    uint32_t ok = nid(c);
    put(c, b, (5u << 16) | S_IEqual);
    put(c, b, c->t_bool);
    put(c, b, ok);
    put(c, b, prev);
    put(c, b, old);
    put(c, b, (4u << 16) | S_LoopMerge);
    put(c, b, merge);
    put(c, b, cont);
    put(c, b, 0);
    put(c, b, (4u << 16) | S_BranchCond);
    put(c, b, ok);
    put(c, b, merge);
    put(c, b, cont);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, cont);
    emit_branch(c, b, header);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, merge);
    return prev;
}

// 64-bit shared atomics on a .u64-typed shared variable: Workgroup
// storage is a native u64 array, so the hardware atomic ops (including
// 64-bit CAS) work directly.
static void emit_atom_shared64(ctx_t *c, spv_blob_t *b,
                               const ptx_instr_t *in, int shvar,
                               uint32_t off, int line) {
    if (c->shvars[shvar].elem != 8) {
        failf(c, line,
              "64-bit shared atomics require a .u64 shared variable");
        return;
    }
    if (in->dst[0] && check_dst(c, in->dst, in->ty, line) != 0)
        return;
    uint32_t ptr = sh_addr_ptr64(c, b, shvar, off);
    if (c->failed)
        return;
    uint32_t scope = const_u32(c, 2);     // Workgroup
    uint32_t sem = const_u32(c, 0x108u);  // AcquireRelease|Workgroup
    uint32_t sem_ne = const_u32(c, 0x102u); // Acquire|Workgroup
    if (in->op == POP_ATOMCAS) {
        val_t cmp = resolve(c, b, &in->src[1], PTY_U64, line);
        if (c->failed)
            return;
        val_t swv = resolve(c, b, &in->src[2], PTY_U64, line);
        if (c->failed)
            return;
        uint32_t prev = nid(c);
        put(c, b, (9u << 16) | S_AtomicCAS);
        put(c, b, c->t_u64);
        put(c, b, prev);
        put(c, b, ptr);
        put(c, b, scope);
        put(c, b, sem);
        put(c, b, sem_ne);
        put(c, b, swv.id);
        put(c, b, cmp.id);
        if (in->dst[0]) def_val(c, in->dst, PTY_U64, prev);
        return;
    }
    if (in->ty == PTY_F64) {
        if (in->atom_op != 0) {
            failf(c, line, "atom.f64 supports .add only");
            return;
        }
        val_t add = resolve(c, b, &in->src[1], PTY_F64, line);
        if (c->failed)
            return;
        if (c->shared_f64_view &&
            (c->caps.atomic_float & ACUDA_ATOMIC_F64_SHARED)) {
            uint32_t res = nid(c);
            put(c, b, (7u << 16) | S_AtomicFAddEXT);
            put(c, b, c->t_f64);
            put(c, b, res);
            put(c, b, ptr);
            put(c, b, scope);
            put(c, b, sem);
            put(c, b, add.id);
            if (in->dst[0])
                def_val(c, in->dst, PTY_F64, res);
            return;
        }
        uint32_t bits = atom_fadd64_loop(c, b, ptr, add.id, 1);
        if (c->failed)
            return;
        uint32_t f = nid(c);
        put(c, b, (4u << 16) | S_Bitcast);
        put(c, b, c->t_f64);
        put(c, b, f);
        put(c, b, bits);
        if (in->dst[0]) def_val(c, in->dst, PTY_F64, f);
        return;
    }
    val_t v = resolve(c, b, &in->src[1], PTY_U64, line);
    if (c->failed)
        return;
    uint32_t opc;
    switch (in->atom_op) {
    case 0: opc = S_AtomicIAdd; break;
    case 3: opc = S_AtomicAnd; break;
    case 4: opc = S_AtomicOr; break;
    case 5: opc = S_AtomicXor; break;
    case 6: opc = S_AtomicExchange; break;
    case 7: opc = in->sign ? S_AtomicSMin : S_AtomicUMin; break;
    case 8: opc = in->sign ? S_AtomicSMax : S_AtomicUMax; break;
    default:
        failf(c, line, "unsupported shared 64-bit atom");
        return;
    }
    uint32_t res = nid(c);
    put(c, b, (7u << 16) | opc);
    put(c, b, c->t_u64);
    put(c, b, res);
    put(c, b, ptr);
    put(c, b, scope);
    put(c, b, sem);
    put(c, b, v.id);
    if (in->dst[0]) def_val(c, in->dst, PTY_U64, res);
}

void emit_atom(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = line;
    uint32_t id;
        // Atomic read-modify-write; every form returns the OLD value.
        // Scope/semantics follow the space, mirroring the sub-word
        // CAS loop: shared = Workgroup scope + AcquireRelease|
        // WorkgroupMemory (0x108/0x102), global/generic = Device scope
        // + AcquireRelease|UniformMemory (0x48/0x42). Uniformly strong
        // semantics are sound for .relaxed/.acquire/.release/.acq_rel
        // on this single-device backend.
        int binding = -1;
        uint32_t off = 0;
        int shvar = -1;
        int glvar = -1;
        if (resolve_addr(c, b, &in->src[0], &binding, &off, &shvar, &glvar,
                         line) != 0)
            return;
        int use_shared = shvar >= 0;
        if (in->space == PSP_SHARED && !use_shared) {
            failf(c, line, "shared atom needs a .shared-derived address");
            return;
        }
        if (glvar >= 0) {
            failf(c, line, "atomics on .global variables unsupported (no device backing)");
            return;
        }
        if (binding == -2) {
            failf(c, line, "atomics on .local variables unsupported");
            return;
        }
        if (in->space == PSP_GLOBAL && use_shared) {
            failf(c, line, "global atom of a shared address (space mismatch)");
            return;
        }
        uint32_t scope = const_u32(c, use_shared ? 2 : 1);
        uint32_t sem = const_u32(c, use_shared ? 0x108u : 0x48u);
        uint32_t sem_ne = const_u32(c, use_shared ? 0x102u : 0x42u);
        int is64 = in->ty == PTY_U64 || in->ty == PTY_F64;
        uint32_t ptr;
        if (is64 && use_shared) {
            emit_atom_shared64(c, b, in, shvar, off, line);
            return;
        }
        if (is64) {
            ptr = ssbo64_ptr(c, b, binding, off);
            if (c->failed)
                return;
        } else {
            ptr = use_shared ? sh_addr_ptr(c, b, shvar, off)
                             : addr_ptr(c, b, binding, off);
        }
        uint32_t aty = is64 ? c->t_u64 : c->t_u32;
        if (in->op == POP_ATOMCAS && in->ty == PTY_U16) {
            // 16-bit CAS: update the containing u32 word, preserving
            // the other half.
            if (in->dst[0] && check_dst(c, in->dst, PTY_U16, line) != 0)
                return;
            val_t cmp = resolve(c, b, &in->src[1], PTY_U16, line);
            if (c->failed)
                return;
            val_t newv = resolve(c, b, &in->src[2], PTY_U16, line);
            if (c->failed)
                return;
            uint32_t sh = atom_lane_shift(c, b, off, line);
            uint32_t prev = atom_cas16_loop(c, b, ptr, use_shared, sh,
                                            cmp.id, newv.id);
            if (c->failed)
                return;
            uint32_t ph = nid(c);
            emit_arith(c, b, S_ShrL, c->t_u32, ph, prev, sh);
            uint32_t oldh = nid(c);
            emit_arith(c, b, S_And, c->t_u32, oldh, ph,
                       const_u32(c, 0xffffu));
            if (in->dst[0]) def_val(c, in->dst, PTY_U16, oldh);
            return;
        }
        if (in->op == POP_ATOMCAS) {
            if (in->dst[0] && check_dst(c, in->dst, in->ty, line) != 0)
                return;
            val_t cmp = resolve(c, b, &in->src[1], in->ty, line);
            if (c->failed)
                return;
            val_t newv = resolve(c, b, &in->src[2], in->ty, line);
            if (c->failed)
                return;
            uint32_t prev = nid(c);
            put(c, b, (9u << 16) | S_AtomicCAS);
            put(c, b, aty);
            put(c, b, prev);
            put(c, b, ptr);
            put(c, b, scope);
            put(c, b, sem);
            put(c, b, sem_ne);
            put(c, b, newv.id);
            put(c, b, cmp.id);
            if (in->dst[0]) def_val(c, in->dst, in->ty, prev);
            return;
        }
        if (in->atom_op == 1 || in->atom_op == 2) {
            // Bounded increment/decrement via a CAS loop (PTX wrap
            // semantics; SPIR-V's atomic inc/dec has no bound).
            if (in->dst[0] && check_dst(c, in->dst, PTY_U32, line) != 0)
                return;
            val_t bound = resolve(c, b, &in->src[1], PTY_U32, line);
            if (c->failed)
                return;
            id = in->atom_op == 1
                     ? atom_wrap_loop(c, b, ptr, use_shared, bound.id)
                     : atom_dec_loop(c, b, ptr, use_shared, bound.id);
            if (in->dst[0]) def_val(c, in->dst, PTY_U32, id);
            return;
        }
        if (in->ty == PTY_F16 || in->ty == PTY_BF16) {
            // 16-bit float add: CAS the containing u32 word.
            if (in->atom_op != 0) {
                failf(c, line, "atom f16/bf16 supports .add only");
                return;
            }
            if (in->dst[0] && check_dst(c, in->dst, in->ty, line) != 0)
                return;
            int is_bf = in->ty == PTY_BF16;
            val_t add = resolve(c, b, &in->src[1], in->ty, line);
            if (c->failed)
                return;
            // A .b16/.u16 holder carries the raw half pattern; a .f16
            // register carries the exact f32 value in this backend.
            reg_t *ar = in->src[1].kind == OPD_REG
                            ? find_reg(c, in->src[1].reg) : NULL;
            if (ar && ar->decl_ty == PTY_U16) {
                add.id = is_bf ? bf16_to_f32val(c, b, add.id, line)
                               : f16_to_f32val(c, b, add.id, line);
                if (c->failed)
                    return;
            }
            uint32_t sh = atom_lane_shift(c, b, off, line);
            uint32_t prev = atom_half_add_loop(c, b, ptr, use_shared, sh,
                                               is_bf, add.id, line);
            if (c->failed)
                return;
            uint32_t ph = nid(c);
            emit_arith(c, b, S_ShrL, c->t_u32, ph, prev, sh);
            uint32_t oldh = nid(c);
            emit_arith(c, b, S_And, c->t_u32, oldh, ph,
                       const_u32(c, 0xffffu));
            uint32_t oldf = is_bf
                                ? bf16_to_f32val(c, b, oldh, line)
                                : f16_to_f32val(c, b, oldh, line);
            reg_t *dr = in->dst[0] ? find_reg(c, in->dst) : NULL;
            if (in->dst[0]) {
                if (dr && dr->decl_ty == PTY_U16)
                    def_val(c, in->dst, PTY_U16, oldh);
                else
                    def_val(c, in->dst, in->ty, oldf);
            }
            return;
        }
        if (in->ty == PTY_F16X2 || in->ty == PTY_BF16X2) {
            // Packed half add: CAS the whole u32 word.
            if (in->atom_op != 0) {
                failf(c, line, "atom f16x2/bf16x2 supports .add only");
                return;
            }
            if (in->dst[0] && check_dst(c, in->dst, in->ty, line) != 0)
                return;
            int is_bf = in->ty == PTY_BF16X2;
            val_t add = resolve(c, b, &in->src[1], in->ty, line);
            if (c->failed)
                return;
            uint32_t prev = atom_packed_add_loop(c, b, ptr, use_shared,
                                                 is_bf, add.id, line);
            if (c->failed)
                return;
            if (in->dst[0]) def_val(c, in->dst, in->ty, prev);
            return;
        }
        if (in->ty == PTY_F64) {
            // atom.add.f64: global/generic only (shared 64-bit atomics
            // are outside the current shared-memory representation).
            if (in->atom_op != 0) {
                failf(c, line, "atom.f64 supports .add only");
                return;
            }
            if (use_shared) {
                failf(c, line,
                      "64-bit atomics on shared memory unsupported");
                return;
            }
            if (in->dst[0] && check_dst(c, in->dst, PTY_F64, line) != 0)
                return;
            val_t add = resolve(c, b, &in->src[1], PTY_F64, line);
            if (c->failed)
                return;
            if (binding == -3 &&
                (c->caps.atomic_float & ACUDA_ATOMIC_F64_BUFFER)) {
                // BDA: typed physical f64 pointer, no pointer bitcast.
                uint32_t fp = bda_convert(c, b, c->t_ptr_phys_f64, off);
                if (c->failed)
                    return;
                uint32_t res = nid(c);
                put(c, b, (7u << 16) | S_AtomicFAddEXT);
                put(c, b, c->t_f64);
                put(c, b, res);
                put(c, b, fp);
                put(c, b, scope);
                put(c, b, sem);
                put(c, b, add.id);
                if (in->dst[0])
                    def_val(c, in->dst, PTY_F64, res);
                return;
            }
            id = atom_fadd64_loop(c, b, ptr, add.id, 0);
            uint32_t f = nid(c);
            put(c, b, (4u << 16) | S_Bitcast);
            put(c, b, c->t_f64);
            put(c, b, f);
            put(c, b, id);
            if (in->dst[0]) def_val(c, in->dst, PTY_F64, f);
            return;
        }
        if (in->ty == PTY_F32) {
            // Float add: use the native OpAtomicFAddEXT where the
            // device enables the matching VK_EXT_shader_atomic_float
            // feature, otherwise the u32-word CAS loop.
            if (in->dst[0] && check_dst(c, in->dst, PTY_F32, line) != 0)
                return;
            val_t add = resolve(c, b, &in->src[1], PTY_F32, line);
            if (c->failed)
                return;
            int native_add = 0;
            uint32_t aptr = ptr;
            if (in->atom_op == 0) {
                if (use_shared && !c->host_machine && c->shared_f32_view &&
                    (c->caps.atomic_float & ACUDA_ATOMIC_F32_SHARED)) {
                    native_add = 1; // ptr is Workgroup float
                } else if (!use_shared && binding == -3 &&
                           (c->caps.atomic_float & ACUDA_ATOMIC_F32_BUFFER)) {
                    // BDA: ConvertUToPtr produces a typed physical
                    // pointer (no pointer bitcast in Logical).
                    aptr = bda_convert(c, b, c->t_ptr_phys_f32, off);
                    if (c->failed)
                        return;
                    native_add = 1;
                }
            }
            if (native_add) {
                uint32_t res = nid(c);
                put(c, b, (7u << 16) | S_AtomicFAddEXT);
                put(c, b, c->t_f32);
                put(c, b, res);
                put(c, b, aptr);
                put(c, b, scope);
                put(c, b, sem);
                put(c, b, add.id);
                if (in->dst[0])
                    def_val(c, in->dst, PTY_F32, res);
                return;
            }
            id = atom_fadd_loop(c, b, ptr, use_shared, add.id);
            uint32_t f = nid(c);
            put(c, b, (4u << 16) | S_Bitcast);
            put(c, b, c->t_f32);
            put(c, b, f);
            put(c, b, id);
            if (in->dst[0]) def_val(c, in->dst, PTY_F32, f);
            return;
        }
        if (!is64 && in->ty != PTY_U32) {
            failf(c, line, "unsupported atom type");
            return;
        }
        if (in->dst[0] && check_dst(c, in->dst, in->ty, line) != 0)
            return;
        val_t v = resolve(c, b, &in->src[1], in->ty, line);
        if (c->failed)
            return;
        uint32_t opc;
        switch (in->atom_op) {
        case 0: opc = S_AtomicIAdd; break;
        case 3: opc = S_AtomicAnd; break;
        case 4: opc = S_AtomicOr; break;
        case 5: opc = S_AtomicXor; break;
        case 6: opc = S_AtomicExchange; break;
        case 7:
            opc = in->sign ? S_AtomicSMin : S_AtomicUMin;
            break;
        case 8:
            opc = in->sign ? S_AtomicSMax : S_AtomicUMax;
            break;
        default:
            failf(c, line, "unsupported atom form");
            return;
        }
        uint32_t res = nid(c);
        put(c, b, (7u << 16) | opc);
        put(c, b, aty);
        put(c, b, res);
        put(c, b, ptr);
        put(c, b, scope);
        put(c, b, sem);
        put(c, b, v.id);
        if (in->dst[0]) def_val(c, in->dst, in->ty, res);
        return;
}
