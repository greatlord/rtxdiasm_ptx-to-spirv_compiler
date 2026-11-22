// Per-invocation PC state machine for functions containing host calls.
//
// Every register lives in a Function-storage variable. The entry block
// loads the saved state from the state SSBO, the loop header reloads the
// variables into SSA values and dispatches on the PC, and each case
// stores its results back before jumping to the continue block. A host
// call stores the register file plus the continuation PC, writes the
// request, and returns; the next dispatch resumes each invocation at its
// own PC, so branches, loops, and divergence are ordinary control flow.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "spirv_internal.h"
#include "acuda/hostcall.h"

#define HM_SHARED (1ull << 32)
#define HM_GLVAR  (1ull << 33)
#define HM_PADDR  (1ull << 34)
#define HM_BDA    (1ull << 35)
#define HM_LOCAL  (1ull << 36)
#define HM_SCALAR (1ull << 63)

enum {
    HOME_UNINIT = -2,
    HOME_CONFLICT = -1,
    HOME_SCALAR = 0,
    HOME_BINDING = 1,
    HOME_SHARED = 2,
    HOME_GLOBAL = 3,
    HOME_PADDR = 4,
    HOME_BDA = 5,
    HOME_LOCAL = 6,
};

static uint32_t sm_load_u32(ctx_t *c, spv_blob_t *b, int binding,
                            uint32_t off);
static int mask_popcount(uint64_t m);
static uint32_t sm_ptr_tag(ctx_t *c, spv_blob_t *b, reg_t *r);
static uint32_t sm_ptr_off(ctx_t *c, spv_blob_t *b, reg_t *r);
static uint32_t sm_to_u64(ctx_t *c, spv_blob_t *b, uint32_t id);

static int home_is_ptr(int h) {
    return h == HOME_BINDING || h == HOME_SHARED || h == HOME_GLOBAL ||
           h == HOME_PADDR || h == HOME_BDA || h == HOME_LOCAL;
}

static uint32_t sm_index(ctx_t *c, spv_blob_t *b, int line) {
    ptx_opd_t op;
    memset(&op, 0, sizeof(op));
    op.kind = OPD_SPECIAL;
    op.special = SPC_TID;
    val_t tid = resolve(c, b, &op, PTY_U32, line);
    op.special = SPC_NTID;
    val_t ntid = resolve(c, b, &op, PTY_U32, line);
    op.special = SPC_CTAID;
    val_t cta = resolve(c, b, &op, PTY_U32, line);
    if (c->failed)
        return 0;
    uint32_t mul = nid(c);
    emit_arith(c, b, S_IMul, c->t_u32, mul, cta.id, ntid.id);
    uint32_t add = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u32, add, mul, tid.id);
    uint32_t id = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u64);
    put(c, b, id);
    put(c, b, add);
    return id;
}

static uint32_t sm_state_base(ctx_t *c, spv_blob_t *b) {
    cparam_t *cp = find_cparam(c, "__acuda_host_state");
    if (!cp || cp->binding < 0)
        return 0;
    uint32_t b32 = sm_load_u32(c, b, cp->binding, const_u64(c, 0));
    return sm_to_u64(c, b, b32);
}

static uint32_t sm_off(ctx_t *c, spv_blob_t *b, uint32_t index,
                       uint32_t field) {
    uint32_t base = sm_state_base(c, b);
    uint32_t stride = const_u64(c, c->host_state_stride);
    uint32_t mul = nid(c);
    emit_arith(c, b, S_IMul, c->t_u64, mul, index, stride);
    uint32_t rel = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, rel, mul, const_u64(c, field));
    uint32_t off = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, off, base, rel);
    return off;
}

static void sm_store_u32(ctx_t *c, spv_blob_t *b, int binding,
                         uint32_t off, uint32_t val) {
    uint32_t ptr = addr_ptr(c, b, binding, off);
    emit_store(c, b, ptr, val);
}

static uint32_t sm_load_u32(ctx_t *c, spv_blob_t *b, int binding,
                            uint32_t off) {
    uint32_t ptr = addr_ptr(c, b, binding, off);
    uint32_t id = nid(c);
    emit_load(c, b, c->t_u32, id, ptr);
    return id;
}

static uint32_t sm_bitcast(ctx_t *c, spv_blob_t *b, uint32_t ty,
                           uint32_t id) {
    uint32_t out = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, ty);
    put(c, b, out);
    put(c, b, id);
    return out;
}

static uint32_t sm_to_u64(ctx_t *c, spv_blob_t *b, uint32_t id) {
    uint32_t out = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u64);
    put(c, b, out);
    put(c, b, id);
    return out;
}

static uint32_t sm_to_u32(ctx_t *c, spv_blob_t *b, uint32_t id) {
    uint32_t out = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u32);
    put(c, b, out);
    put(c, b, id);
    return out;
}

static cparam_t *sm_param(ctx_t *c, const char *name, int line) {
    cparam_t *cp = find_cparam(c, name);
    if (!cp || cp->kind != 1 || cp->binding < 0) {
        failf(c, line, "host-call internal buffer parameter missing");
        return NULL;
    }
    return cp;
}

static uint32_t sm_var_ty(ctx_t *c, int home, ptx_ty_t decl) {
    if (home_is_ptr(home))
        return c->t_ptr_fn_u64;
    if (home == HOME_CONFLICT)
        // Dynamic homes round-trip u64 offsets and widened scalar
        // values through one slot; conversion happens on save/restore.
        return c->t_ptr_fn_u64;
    if (decl == PTY_PRED)
        return c->t_ptr_fn_bool;
    if (decl == PTY_U64)
        return c->t_ptr_fn_u64;
    if (decl == PTY_F64)
        return c->t_ptr_fn_f64;
    if (decl == PTY_F32)
        return c->t_ptr_fn_f32;
    // .f16/.bf16 registers carry raw 16-bit patterns in a u32 SSA value
    // (consumers widen them), so their state-machine home is u32 too.
    return c->t_ptr_fn_u32;
}

// A declaration whose SSA value is a 32-bit word (integer, float, or a
// packed pair) can also be reinterpreted as a raw word.
static int sm_decl_word32(ptx_ty_t ty) {
    return ty == PTY_U32 || ty == PTY_F32 ||
           ty == PTY_F16X2 || ty == PTY_BF16X2;
}

// Reinterpret the register's current SSA value as a raw 32-bit word.
static uint32_t sm_reg_word(ctx_t *c, spv_blob_t *b, const reg_t *r) {
    if (!r->defined || !r->cur)
        return const_u32(c, 0);
    if (r->cur_ty == PTY_U32)
        return r->cur;
    if (sm_decl_word32(r->cur_ty)) {
        uint32_t out = nid(c);
        put(c, b, (4u << 16) | S_Bitcast);
        put(c, b, c->t_u32);
        put(c, b, out);
        put(c, b, r->cur);
        return out;
    }
    return sm_to_u32(c, b, r->cur);
}

static uint32_t sm_val_ty(ctx_t *c, int home, ptx_ty_t decl) {
    if (home_is_ptr(home))
        return c->t_u64;
    if (decl == PTY_PRED)
        return c->t_bool;
    if (decl == PTY_F16 || decl == PTY_BF16)
        return c->t_u32;
    return ty_of(c, decl);
}

static uint32_t sm_reg_ptr_ty(ctx_t *c, int i) {
    reg_t *r = &c->regs[i];
    if (r->is_coopmat) {
        uint32_t id = nid(c);
        put(c, &c->types, (4u << 16) | S_TypePointer);
        put(c, &c->types, id);
        put(c, &c->types, 7); // Function
        put(c, &c->types, r->coopmat_ty);
        return id;
    }
    return sm_var_ty(c, c->host_home[i], r->decl_ty);
}

static uint32_t sm_zero_coopmat(ctx_t *c, uint32_t ty) {
    uint32_t id = nid(c);
    put(c, &c->types, (3u << 16) | 46); // OpConstantNull
    put(c, &c->types, ty);
    put(c, &c->types, id);
    return id;
}

static uint32_t sm_var(ctx_t *c, spv_blob_t *b, uint32_t ptr_ty) {
    uint32_t id = nid(c);
    put(c, b, (4u << 16) | S_Variable);
    put(c, b, ptr_ty);
    put(c, b, id);
    put(c, b, 7); // StorageClass Function
    return id;
}

static uint32_t sm_load_var(ctx_t *c, spv_blob_t *b, uint32_t ty,
                            uint32_t var) {
    uint32_t id = nid(c);
    put(c, b, (4u << 16) | S_Load);
    put(c, b, ty);
    put(c, b, id);
    put(c, b, var);
    return id;
}

static void sm_store_var(ctx_t *c, spv_blob_t *b, uint32_t var,
                         uint32_t val) {
    put(c, b, (3u << 16) | S_Store);
    put(c, b, var);
    put(c, b, val);
}

// Convert a current SSA register value to its state-buffer u64 form.
static uint32_t sm_reg_to_u64(ctx_t *c, spv_blob_t *b, const reg_t *r) {
    if (r->is_ptr)
        return r->paddr >= 0 ? const_u64(c, 0) : r->off;
    if (r->decl_ty == PTY_PRED) {
        uint32_t sel = nid(c);
        put(c, b, (6u << 16) | S_Select);
        put(c, b, c->t_u32);
        put(c, b, sel);
        put(c, b, r->cur);
        put(c, b, const_u32(c, 1));
        put(c, b, const_u32(c, 0));
        return sm_to_u64(c, b, sel);
    }
    if (r->decl_ty == PTY_U64)
        return r->cur;
    if (r->decl_ty == PTY_F64)
        return sm_bitcast(c, b, c->t_u64, r->cur);
    uint32_t u = r->cur;
    if (r->decl_ty == PTY_F32)
        u = sm_bitcast(c, b, c->t_u32, u);
    return sm_to_u64(c, b, u);
}

// Convert a state-buffer u64 back into the register's SSA type.
static uint32_t sm_u64_to_reg(ctx_t *c, spv_blob_t *b, ptx_ty_t decl,
                              uint32_t v) {
    if (decl == PTY_U64)
        return v;
    if (decl == PTY_F64)
        return sm_bitcast(c, b, c->t_f64, v);
    uint32_t u = sm_to_u32(c, b, v);
    if (decl == PTY_F32)
        return sm_bitcast(c, b, c->t_f32, u);
    return u;
}

// Store one register slot (value at +8) into the state buffer.
static void sm_store_reg(ctx_t *c, spv_blob_t *b, uint32_t index,
                         int i, const reg_t *r) {
    uint32_t off = sm_off(c, b, index,
                          ACUDA_HOST_STATE_HEADER +
                              (uint32_t)i * ACUDA_HOST_STATE_REG_STRIDE + 8);
    cparam_t *cp = sm_param(c, "__acuda_host_state", 0);
    if (!cp)
        return;
    store_u64(c, b, cp->binding, off, sm_reg_to_u64(c, b, r));
    if (c->host_dynamic[i]) {
        uint32_t tag = sm_load_var(c, b, c->t_u32, r->dyn_tag);
        uint32_t toff = sm_off(c, b, index,
                               ACUDA_HOST_STATE_HEADER +
                                   (uint32_t)i *
                                       ACUDA_HOST_STATE_REG_STRIDE + 4);
        sm_store_u32(c, b, cp->binding, toff, tag);
    }
}

static void sm_store_all_regs(ctx_t *c, spv_blob_t *b, uint32_t index) {
    for (int i = 0; i < c->nregs; i++)
        sm_store_reg(c, b, index, i, &c->regs[i]);
}

static int sm_target_index(ctx_t *c, const ptx_func_t *fn,
                           const char *label, int line) {
    int idx = block_index_of(c, fn, label);
    if (idx < 0)
        failf(c, line, "branch target not found");
    return idx;
}

// Define all registers from their Function variables at the loop header.
static void sm_define_regs(ctx_t *c, spv_blob_t *b) {
    for (int i = 0; i < c->nregs; i++) {
        reg_t *r = &c->regs[i];
        int home = c->host_home[i];
        if (!c->host_need[i]) {
            // Not live across a block edge: its value is produced and
            // consumed inside a single case, so it needs no slot.
            r->defined = 0;
            continue;
        }
        if (r->is_coopmat) {
            uint32_t v = sm_load_var(c, b, r->coopmat_ty,
                                     c->host_reg_var[i]);
            if (c->failed)
                return;
            r->defined = 1;
            r->cur = v;
            r->cur_ty = PTY_U32;
            r->cur_is_coopmat = 1;
            continue;
        }
        uint32_t ty = sm_val_ty(c, home, r->decl_ty);
        uint32_t v;
        if (c->host_dynamic[i]) {
            // Dynamic home: the slot is u64 (see sm_var_ty). Pointer
            // uses re-establish is_ptr per case via the tag switch;
            // scalar phases read the converted value below.
            v = sm_load_var(c, b, c->t_u64, c->host_reg_var[i]);
        } else {
            v = sm_load_var(c, b, ty, c->host_reg_var[i]);
        }
        if (c->failed)
            return;
        if (c->host_dynamic[i]) {
            r->defined = 1;
            r->cur = sm_u64_to_reg(c, b, r->decl_ty, v);
            r->cur_ty = r->decl_ty;
            if (c->host_word_var[i] && decl_int_width(r->decl_ty) == 32) {
                // Restore the materialized 32-bit word alongside the
                // address offset: a dynamic pointer-word register holds
                // both, and value uses (packing the low half back into
                // a 64-bit pointer) need the word, not the offset.
                uint32_t w = sm_load_var(c, b, c->t_u32,
                                         c->host_word_var[i]);
                if (c->failed)
                    return;
                r->cur = w;
                r->cur_ty = PTY_U32;
            }
            r->is_ptr = 0;
            r->binding = -2;
            r->off = v;
            r->shvar = -1;
            r->glvar = -1;
            r->paddr = -1;
            r->dyn_home = 1;
            r->dyn_tag = c->host_tag_var[i];
        } else if (home == HOME_BINDING) {
            def_ptr(c, r->name, c->host_home_aux[i], v);
        } else if (home == HOME_SHARED) {
            def_shared(c, r->name, c->host_home_aux[i], v);
        } else if (home == HOME_GLOBAL) {
            def_global(c, r->name, c->host_home_aux[i], v);
        } else if (home == HOME_BDA) {
            def_ptr(c, r->name, -3, v);
        } else if (home == HOME_LOCAL) {
            r->defined = 1;
            r->is_ptr = 1;
            r->binding = -2;
            r->lovar = c->host_home_aux[i];
            r->off = v ? v : const_u64(c, 0);
            r->cur = lo_addr_ptr(c, b, c->host_home_aux[i], r->off);
            r->cur_ty = PTY_U64;
            r->shvar = -1;
            r->glvar = -1;
            r->paddr = -1;
            if (getenv("ACUDA_TRACE") && r->lovar < 0)
                fprintf(stderr, "acuda-local-warn smdef HOME_LOCAL r%d(%s) lovar=%d aux=%d\n", i, r->name, r->lovar, c->host_home_aux[i]);
        } else if (home == HOME_PADDR) {
            r->defined = 1;
            r->cur = 0;
            r->is_ptr = 1;
            r->binding = -1;
            r->off = 0;
            r->shvar = -1;
            r->glvar = -1;
            r->paddr = c->host_home_aux[i];
        } else {
            def_val(c, r->name, r->decl_ty, v);
        }
    }
}

// Save current SSA registers into their Function variables.
static void sm_save_regs(ctx_t *c, spv_blob_t *b) {
    for (int i = 0; i < c->nregs; i++) {
        reg_t *r = &c->regs[i];
        int home = c->host_home[i];
        uint32_t v;
        if (!c->host_need[i])
            continue;
        if (r->is_coopmat) {
            v = (r->defined && r->cur_is_coopmat)
                    ? r->cur
                    : sm_zero_coopmat(c, r->coopmat_ty);
            sm_store_var(c, b, c->host_reg_var[i], v);
            continue;
        }
        if (c->host_dynamic[i]) {
            if (c->host_word_var[i] && decl_int_width(r->decl_ty) == 32) {
                // Save the materialized word too: pointer-word
                // registers are used both as an offset (address
                // phases) and as a raw word (value phases such as
                // repacking a 64-bit pointer from its halves).
                sm_store_var(c, b, c->host_word_var[i],
                             sm_reg_word(c, b, r));
                if (c->failed)
                    return;
            }
            if (!r->is_ptr) {
                // Scalar phase of a dynamic register (tagged scalar
                // at its definition): save the value itself so the
                // next state restores it as a value.
                v = sm_reg_to_u64(c, b, r);
            } else {
                v = r->off;
            }
        } else if (home_is_ptr(home)) {
            if (r->paddr >= 0)
                v = const_u64(c, 0);
            else if (r->is_ptr)
                v = r->off;
            else if (home == HOME_BDA || home == HOME_BINDING) {
                // A 64-bit value assembled from pointer words (or a raw
                // BDA address) is a value in this phase; round-trip it
                // through the slot and let the restore define the
                // pointer value again.
                v = sm_reg_to_u64(c, b, r);
            } else {
                if (getenv("ACUDA_DEBUG_STATE"))
                    fprintf(stderr, "acuda-trace ?? sm_save_regs lost home reg=%s bind=%d home=%d mask=0x%llx\n",
                            r->name, r->binding, home,
                            (unsigned long long)c->host_home_mask[i]);
                failf(c, 0, "host-call register lost its pointer home");
                return;
            }
        } else if (r->decl_ty == PTY_PRED) {
            v = r->cur;
        } else if (r->is_ptr) {
            if (getenv("ACUDA_DEBUG_STATE"))
                fprintf(stderr, "acuda-trace ?? sm_save_regs fail reg=%s bind=%d sh=%d gl=%d paddr=%d home=%d dyn=%d mask=0x%llx\n",
                        r->name, r->binding, r->shvar, r->glvar, r->paddr,
                        home_is_ptr(home) ? home : -999,
                        c->host_dynamic[i],
                        (unsigned long long)c->host_home_mask[i]);
            failf(c, 0, "host-call pointer register has no stable home");
            return;
        } else {
            // The home variable follows the register's declaration;
            // a value produced as an alias view (f32 in a .b32 reg)
            // must be bitcast before the store.
            v = r->cur;
            if (r->cur_ty != r->decl_ty) {
                uint32_t bt = 0;
                if ((r->decl_ty == PTY_U32 && r->cur_ty == PTY_F32) ||
                    (r->decl_ty == PTY_F32 && r->cur_ty == PTY_U32))
                    bt = ty_of(c, r->decl_ty);
                else if ((r->decl_ty == PTY_U64 &&
                          r->cur_ty == PTY_F64) ||
                         (r->decl_ty == PTY_F64 &&
                          r->cur_ty == PTY_U64))
                    bt = ty_of(c, r->decl_ty);
                if (bt) {
                    uint32_t out = nid(c);
                    put(c, b, (4u << 16) | S_Bitcast);
                    put(c, b, bt);
                    put(c, b, out);
                    put(c, b, v);
                    v = out;
                }
            }
        }
        sm_store_var(c, b, c->host_reg_var[i], v);
    }
}

static void sm_finish_block(ctx_t *c, spv_blob_t *b, uint32_t pc_id,
                            uint32_t cont_label, uint32_t done_label) {
    sm_save_regs(c, b);
    sm_store_var(c, b, c->host_pc_var, pc_id);
    put(c, b, (2u << 16) | S_Branch);
    put(c, b, done_label);
    (void)cont_label;
}

// Emit the actual host-call terminator (state + request + return).
static void sm_emit_host_call(ctx_t *c, spv_blob_t *b,
                              const ptx_instr_t *call, uint32_t index) {
    // State: PC first, then the register file.
    uint32_t pc_off = sm_off(c, b, index, 0);
    sm_store_u32(c, b, sm_param(c, "__acuda_host_state", call->line)->binding,
                 pc_off, const_u32(c, (uint32_t)call->host_cont));
    sm_store_all_regs(c, b, index);
    emit_host_request(c, b, call);
    put(c, b, (1u << 16) | S_Return);
}

static uint32_t sm_cond(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    reg_t *pr = find_reg(c, in->pred);
    if (!pr || !pr->defined) {
        failf(c, in->line, "host-call predicate has no value");
        return 0;
    }
    uint32_t cond = pr->cur;
    if (in->pred_neg) {
        uint32_t n = nid(c);
        put(c, b, (4u << 16) | S_LogicalNot);
        put(c, b, c->t_bool);
        put(c, b, n);
        put(c, b, cond);
        cond = n;
    }
    return cond;
}

static int sm_reg_index(ctx_t *c, const char *name) {
    for (int i = 0; i < c->nregs; i++)
        if (strcmp(c->regs[i].name, name) == 0)
            return i;
    return -1;
}

// Keep the runtime home tag in sync after a definition of a dynamic
// pointer register.
static void sm_update_tag(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    if (!in->dst[0])
        return;
    int ri = sm_reg_index(c, in->dst);
    if (ri < 0 || !c->host_dynamic[ri])
        return;
    reg_t *r = &c->regs[ri];
    if (!r->is_ptr && r->defined) {
        // Scalar phase of a dynamic register (the emitter produced a
        // plain value, e.g. integer arithmetic over a former home):
        // tag it scalar so save/restore round-trips the value and
        // address uses take the safe default instead of a stale tag.
        // (Checked first: the emitter's per-def state is authoritative.
        // A dynamic source below must not re-mark a value result.)
        r->dyn_home = 1;
        r->dyn_tag = c->host_tag_var[ri];
        sm_store_var(c, b, c->host_tag_var[ri],
                     const_u32(c, 0xfffffff9u));
        return;
    }
    for (int s = 0; s < in->nsrc; s++) {
        if (in->src[s].kind != OPD_REG)
            continue;
        int sr = sm_reg_index(c, in->src[s].reg);
        if (sr >= 0 && c->host_dynamic[sr]) {
            uint32_t tag = sm_load_var(c, b, c->t_u32,
                                       c->host_tag_var[sr]);
            sm_store_var(c, b, c->host_tag_var[ri], tag);
            r->is_ptr = 1;
            r->binding = -2;
            r->dyn_home = 1;
            r->dyn_tag = c->host_tag_var[ri];
            r->shvar = -1;
            r->glvar = -1;
            r->paddr = -1;
            return;
        }
    }
    if (r->is_ptr && r->binding == -3) {
        sm_store_var(c, b, c->host_tag_var[ri],
                     const_u32(c, 0xfffffffdu));
        r->dyn_home = 1;
        r->dyn_tag = c->host_tag_var[ri];
        return;
    }
    if (r->is_ptr && r->binding >= 0) {
        sm_store_var(c, b, c->host_tag_var[ri],
                     const_u32(c, (uint32_t)r->binding));
        r->dyn_home = 1;
        r->dyn_tag = c->host_tag_var[ri];
        return;
    }
    if (r->is_ptr && r->paddr >= 0) {
        r->off = const_u64(c, 0);
        r->binding = -2;
        r->shvar = -1;
        r->glvar = -1;
        r->dyn_home = 1;
        r->dyn_tag = c->host_tag_var[ri];
        sm_store_var(c, b, c->host_tag_var[ri],
                     const_u32(c, 0xffff0000u | (uint32_t)r->paddr));
        r->paddr = -1;
        return;
    }
    if (r->is_ptr && r->glvar >= 0) {
        uint32_t base = host_global_off_bytes(c, b, r->glvar, 0);
        uint32_t abs = nid(c);
        emit_arith(c, b, S_IAdd, c->t_u64, abs, base, r->off);
        r->off = abs;
        r->binding = -2;
        r->shvar = -1;
        r->glvar = -1;
        r->paddr = -1;
        r->dyn_home = 1;
        r->dyn_tag = c->host_tag_var[ri];
        sm_store_var(c, b, c->host_tag_var[ri],
                     const_u32(c, 0xfffffffbu));
        return;
    }
    if (r->is_ptr && r->shvar >= 0) {
        uint32_t base = host_shared_off_bytes(c, b, r->shvar, 0);
        uint32_t abs = nid(c);
        emit_arith(c, b, S_IAdd, c->t_u64, abs, base, r->off);
        r->off = abs;
        r->binding = -2;
        r->shvar = -1;
        r->glvar = -1;
        r->paddr = -1;
        r->dyn_home = 1;
        r->dyn_tag = c->host_tag_var[ri];
        sm_store_var(c, b, c->host_tag_var[ri],
                     const_u32(c, 0xfffffffeu));
        return;
    }
    if (!r->is_ptr && r->defined) {
        // Scalar phase of a dynamic register (register reuse across
        // value and pointer roles): tag it scalar so save/restore
        // round-trips the value and address uses take the safe
        // default instead of a stale binding.
        r->dyn_home = 1;
        r->dyn_tag = c->host_tag_var[ri];
        sm_store_var(c, b, c->host_tag_var[ri],
                     const_u32(c, 0xfffffff9u));
        return;
    }
    failf(c, in->line,
          "dynamic pointer home must stay a global or shared pointer");
}

static void sm_emit_dyn_mem(ctx_t *c, spv_blob_t *b,
                            const ptx_instr_t *in) {
    int ri = sm_reg_index(c, in->src[0].addr_reg);
    if (ri < 0 || !c->host_dynamic[ri]) {
        emit_instr(c, b, in);
        return;
    }
    reg_t *ar = &c->regs[ri];
    uint64_t mask = c->host_home_mask[ri] &
                    (0xffffffffull | HM_SHARED | HM_GLVAR | HM_BDA | HM_LOCAL);
    uint32_t pmask = c->host_paddr_mask[ri];
    if (getenv("ACUDA_DEBUG_SYMBOLS") && in->src[0].addr_reg[0] &&
        strcmp(in->src[0].addr_reg, "%rd3") == 0 && in->src[0].addr_imm == 12) {
        fprintf(stderr, "ACuda: dyn mem %s op=%d sp=%d line=%d mask=0x%llx pmask=0x%x dyn=%d aux=%d\n",
                in->src[0].addr_reg, (int)in->op, (int)in->space, in->line,
                (unsigned long long)c->host_home_mask[ri], pmask, (int)c->host_dynamic[ri], c->host_home_aux[ri]);
    }
    uint32_t tags[40];
    int ncases = 0;
    for (int bit = 0; bit < 32; bit++)
        if (mask & (1ull << bit))
            tags[ncases++] = (uint32_t)bit;
    if (mask & HM_SHARED)
        tags[ncases++] = 0xfffffffeu;
    if (mask & HM_GLVAR)
        tags[ncases++] = 0xfffffffbu;
    if (mask & HM_BDA)
        tags[ncases++] = 0xfffffffdu;
    if (mask & HM_LOCAL)
        tags[ncases++] = 0xfffffffcu;
    for (int p = 0; p < 32; p++)
        if (pmask & (1u << p))
            tags[ncases++] = 0xffff0000u | (uint32_t)p;
    if (ncases == 0) {
        failf(c, in->line, "dynamic pointer has no resolvable home");
        return;
    }
    uint32_t tag = sm_load_var(c, b, c->t_u32, ar->dyn_tag);
    uint32_t off = ar->off ? ar->off : const_u64(c, 0);
    uint32_t merge = nid(c);
    uint32_t deflabel = nid(c);
    uint32_t labels[40];
    for (int k = 0; k < ncases; k++)
        labels[k] = nid(c);

    put(c, b, (3u << 16) | S_SelectionMerge);
    put(c, b, merge);
    put(c, b, 0);
    put(c, b, (uint32_t)((3u + 2u * (uint32_t)ncases) << 16) | S_Switch);
    put(c, b, tag);
    put(c, b, deflabel);
    for (int k = 0; k < ncases; k++) {
        put(c, b, tags[k]);
        put(c, b, labels[k]);
    }
    uint32_t vals[40];
    int arm_ptr[40];
    int arm_bind[40], arm_sh[40], arm_gl[40];
    uint32_t arm_off[40], arm_tag[40];
    int nv = 0;
    int any_ptr = 0;
    for (int k = 0; k < ncases; k++) {
        put(c, b, (2u << 16) | S_Label);
        put(c, b, labels[k]);
        ptx_instr_t tmp = *in;
        ar->is_ptr = 1;
        ar->dyn_home = 0;
        ar->off = off;
        ar->cur = 0;
        if (tags[k] < 32) {
            ar->binding = (int)tags[k];
            ar->shvar = -1;
            ar->glvar = -1;
            ar->paddr = -1;
            tmp.space = PSP_GLOBAL;
        } else if (tags[k] == 0xfffffffdu) {
            ar->binding = -3;
            ar->shvar = -1;
            ar->glvar = -1;
            ar->paddr = -1;
            tmp.space = PSP_GLOBAL;
        } else if (tags[k] >= 0xffff0000u &&
                   tags[k] != 0xfffffffeu && tags[k] != 0xfffffffbu) {
            ar->binding = -1;
            ar->shvar = -1;
            ar->glvar = -1;
            ar->paddr = (int)(tags[k] & 0xffffu);
            tmp.space = PSP_PARAM;
        } else if (!c->host_machine && c->host_home_aux[ri] >= 0 &&
                   (tags[k] == 0xfffffffeu || tags[k] == 0xfffffffbu)) {
            // CFG machines without host calls keep shared and global
            // image state in their real storage, not the host-state
            // buffer.  The canonical offset in ar->off is relative to
            // the single variable captured in home_aux.
            ar->binding = -1;
            ar->shvar = tags[k] == 0xfffffffeu ? c->host_home_aux[ri] : -1;
            ar->glvar = tags[k] == 0xfffffffbu ? c->host_home_aux[ri] : -1;
            ar->paddr = -1;
            tmp.space = tags[k] == 0xfffffffeu ? PSP_SHARED : PSP_GLOBAL;
        } else {
            cparam_t *cp = find_cparam(c, "__acuda_host_state");
            if (!cp || cp->binding < 0) {
                failf(c, in->line, "host-call state buffer missing");
                return;
            }
            ar->binding = cp->binding;
            ar->shvar = -1;
            ar->glvar = -1;
            ar->paddr = -1;
            tmp.space = PSP_GLOBAL;
        }
        if (in->op == POP_ATOM || in->op == POP_ATOMCAS)
            emit_instr(c, b, &tmp);
        else
            emit_mem_op(c, b, &tmp);
        if (c->failed)
            return;
        if (in->op == POP_LD || in->op == POP_ATOM ||
            in->op == POP_ATOMCAS) {
            reg_t *dr = find_reg(c, in->dst);
            if (!dr || !dr->defined) {
                failf(c, in->line, "dynamic load result undefined");
                return;
            }
            arm_ptr[nv] = dr->is_ptr;
            if (dr->is_ptr) {
                any_ptr = 1;
                arm_bind[nv] = dr->binding;
                arm_sh[nv] = dr->shvar;
                arm_gl[nv] = dr->glvar;
                arm_off[nv] = dr->off;
                arm_tag[nv] = sm_ptr_tag(c, b, dr);
                vals[nv] = 0;
            } else {
                vals[nv] = dr->cur;
            }
            nv++;
        }
        put(c, b, (2u << 16) | S_Branch);
        put(c, b, merge);
    }
    put(c, b, (2u << 16) | S_Label);
    put(c, b, deflabel);
    if (in->op == POP_LD || in->op == POP_ATOM ||
        in->op == POP_ATOMCAS) {
        uint32_t undef = nid(c);
        if (any_ptr) {
            put(c, b, (3u << 16) | 1); // OpUndef
            put(c, b, c->t_u64);
            put(c, b, undef);
            arm_ptr[nv] = 1;
            arm_bind[nv] = -2;
            arm_sh[nv] = -1;
            arm_gl[nv] = -1;
            arm_off[nv] = undef;
            arm_tag[nv] = const_u32(c, 0xffffffffu);
            vals[nv] = 0;
        } else {
            put(c, b, (3u << 16) | 1); // OpUndef
            put(c, b, ty_of(c, in->ty));
            put(c, b, undef);
            vals[nv] = undef;
            arm_ptr[nv] = 0;
        }
        labels[nv] = deflabel;
        nv++;
    }
    put(c, b, (2u << 16) | S_Branch);
    put(c, b, merge);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, merge);
    ar->is_ptr = 1;
    ar->binding = -2;
    ar->dyn_home = 1;
    ar->dyn_tag = c->host_tag_var[ri];
    ar->off = off;
    ar->cur = 0;
    ar->paddr = -1;
    if (in->op == POP_LD || in->op == POP_ATOM ||
        in->op == POP_ATOMCAS) {
        int di = sm_reg_index(c, in->dst);
        if (any_ptr) {
            for (int k = 0; k < nv; k++)
                if (!arm_ptr[k]) {
                    failf(c, in->line,
                          "dynamic parameter load mixes pointer and scalar");
                    return;
                }
            uint32_t phi = nid(c);
            put(c, b, (uint32_t)((3u + 2u * (uint32_t)nv) << 16) | S_Phi);
            put(c, b, c->t_u64);
            put(c, b, phi);
            for (int k = 0; k < nv; k++) {
                put(c, b, arm_off[k]);
                put(c, b, labels[k]);
            }
            reg_t *dr = &c->regs[di];
            if (c->host_dynamic[di]) {
                uint32_t tphi = nid(c);
                put(c, b, (uint32_t)((3u + 2u * (uint32_t)nv) << 16) |
                           S_Phi);
                put(c, b, c->t_u32);
                put(c, b, tphi);
                for (int k = 0; k < nv; k++) {
                    put(c, b, arm_tag[k]);
                    put(c, b, labels[k]);
                }
                sm_store_var(c, b, c->host_tag_var[di], tphi);
                dr->defined = 1;
                dr->cur = 0;
                dr->is_ptr = 1;
                dr->binding = -2;
                dr->off = phi;
                dr->shvar = -1;
                dr->glvar = -1;
                dr->paddr = -1;
                dr->dyn_home = 1;
                dr->dyn_tag = c->host_tag_var[di];
            } else if (arm_bind[0] == -3 && arm_bind[0] == arm_bind[1])
                def_ptr(c, in->dst, -3, phi);
            else if (arm_bind[0] >= 0 && arm_bind[0] == arm_bind[1])
                def_ptr(c, in->dst, arm_bind[0], phi);
            else if (arm_sh[0] >= 0 && arm_sh[0] == arm_sh[1])
                def_shared(c, in->dst, arm_sh[0], phi);
            else if (arm_gl[0] >= 0 && arm_gl[0] == arm_gl[1])
                def_global(c, in->dst, arm_gl[0], phi);
            else
                failf(c, in->line, "dynamic pointer load homes differ");
            return;
        }
        uint32_t phi = nid(c);
        put(c, b, (uint32_t)((3u + 2u * (uint32_t)nv) << 16) | S_Phi);
        put(c, b, ty_of(c, in->ty));
        put(c, b, phi);
        for (int k = 0; k < nv; k++) {
            put(c, b, vals[k]);
            put(c, b, labels[k]);
        }
        def_val(c, in->dst, in->ty, phi);
    }
}

static uint32_t sm_ptr_tag(ctx_t *c, spv_blob_t *b, reg_t *r) {
    if (r->dyn_home)
        return sm_load_var(c, b, c->t_u32, r->dyn_tag);
    if (r->binding == -3)
        return const_u32(c, 0xfffffffdu);
    if (r->binding >= 0)
        return const_u32(c, (uint32_t)r->binding);
    if (r->shvar >= 0)
        return const_u32(c, 0xfffffffeu);
    if (r->glvar >= 0)
        return const_u32(c, 0xfffffffbu);
    if (r->paddr >= 0)
        return const_u32(c, 0xffff0000u | (uint32_t)r->paddr);
    return 0;
}

static uint32_t sm_ptr_off(ctx_t *c, spv_blob_t *b, reg_t *r) {
    if (r->shvar >= 0) {
        uint32_t base = host_shared_off_bytes(c, b, r->shvar, 0);
        uint32_t abs = nid(c);
        emit_arith(c, b, S_IAdd, c->t_u64, abs, base, r->off);
        return abs;
    }
    if (r->glvar >= 0) {
        uint32_t base = host_global_off_bytes(c, b, r->glvar, 0);
        uint32_t abs = nid(c);
        emit_arith(c, b, S_IAdd, c->t_u64, abs, base, r->off);
        return abs;
    }
    if (r->paddr >= 0)
        return const_u64(c, 0);
    return r->off;
}

static void sm_emit_madwide_ptr(ctx_t *c, spv_blob_t *b,
                                const ptx_instr_t *in) {
    reg_t *base = in->src[2].kind == OPD_REG
                      ? find_reg(c, in->src[2].reg)
                      : NULL;
    int di = sm_reg_index(c, in->dst);
    if (!base || !base->is_ptr || di < 0) {
        failf(c, in->line, "mad.wide pointer base must be a pointer");
        return;
    }
    val_t a = resolve(c, b, &in->src[0], PTY_U32, in->line);
    val_t bv = resolve(c, b, &in->src[1], PTY_U32, in->line);
    if (c->failed)
        return;
    uint32_t a64 = nid(c), b64 = nid(c);
    uint32_t conv = in->sign ? S_SConvert : S_UConvert;
    put(c, b, (4u << 16) | conv);
    put(c, b, c->t_u64);
    put(c, b, a64);
    put(c, b, a.id);
    put(c, b, (4u << 16) | conv);
    put(c, b, c->t_u64);
    put(c, b, b64);
    put(c, b, bv.id);
    uint32_t mul = nid(c);
    emit_arith(c, b, S_IMul, c->t_u64, mul, a64, b64);
    uint32_t boff = c->host_dynamic[di] ? sm_ptr_off(c, b, base)
                                          : base->off;
    uint32_t off = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, off, mul, boff);
    reg_t *dst = &c->regs[di];
    if (!c->host_dynamic[di]) {
        if (base->binding == -3)
            def_ptr(c, in->dst, -3, off);
        else if (base->binding >= 0)
            def_ptr(c, in->dst, base->binding, off);
        else if (base->shvar >= 0)
            def_shared(c, in->dst, base->shvar, off);
        else if (base->glvar >= 0)
            def_global(c, in->dst, base->glvar, off);
        else
            failf(c, in->line, "mad.wide pointer home unsupported");
        return;
    }
    uint32_t tag = sm_ptr_tag(c, b, base);
    sm_store_var(c, b, c->host_tag_var[di], tag);
    dst->defined = 1;
    dst->cur = 0;
    dst->is_ptr = 1;
    dst->binding = -2;
    dst->off = off;
    dst->shvar = -1;
    dst->glvar = -1;
    dst->paddr = -1;
    dst->dyn_home = 1;
    dst->dyn_tag = c->host_tag_var[di];
}

static void sm_emit_selp_ptr(ctx_t *c, spv_blob_t *b,
                             const ptx_instr_t *in) {
    reg_t *r0 = in->src[0].kind == OPD_REG
                    ? find_reg(c, in->src[0].reg)
                    : NULL;
    reg_t *r1 = in->src[1].kind == OPD_REG
                    ? find_reg(c, in->src[1].reg)
                    : NULL;
    reg_t *pr = in->src[2].kind == OPD_REG
                    ? find_reg(c, in->src[2].reg)
                    : NULL;
    if (!r0 || !r1 || !r0->is_ptr || !r1->is_ptr || !pr) {
        failf(c, in->line, "selp pointer operands must both be pointers");
        return;
    }
    int di = sm_reg_index(c, in->dst);
    if (di < 0)
        return;
    uint32_t o0 = c->host_dynamic[di] ? sm_ptr_off(c, b, r0) : r0->off;
    uint32_t o1 = c->host_dynamic[di] ? sm_ptr_off(c, b, r1) : r1->off;
    uint32_t off = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, c->t_u64);
    put(c, b, off);
    put(c, b, pr->cur);
    put(c, b, o0);
    put(c, b, o1);
    reg_t *dst = &c->regs[di];
    if (!c->host_dynamic[di]) {
        if (r0->binding == -3 && r1->binding == -3)
            def_ptr(c, in->dst, -3, off);
        else if (r0->binding >= 0 && r1->binding == r0->binding)
            def_ptr(c, in->dst, r0->binding, off);
        else if (r0->shvar >= 0 && r1->shvar == r0->shvar)
            def_shared(c, in->dst, r0->shvar, off);
        else if (r0->glvar >= 0 && r1->glvar == r0->glvar)
            def_global(c, in->dst, r0->glvar, off);
        else
            failf(c, in->line, "selp pointer homes differ");
        return;
    }
    uint32_t t0 = sm_ptr_tag(c, b, r0);
    uint32_t t1 = sm_ptr_tag(c, b, r1);
    uint32_t tag = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, c->t_u32);
    put(c, b, tag);
    put(c, b, pr->cur);
    put(c, b, t0);
    put(c, b, t1);
    sm_store_var(c, b, c->host_tag_var[di], tag);
    dst->defined = 1;
    dst->cur = 0;
    dst->is_ptr = 1;
    dst->binding = -2;
    dst->off = off;
    dst->shvar = -1;
    dst->glvar = -1;
    dst->paddr = -1;
    dst->dyn_home = 1;
    dst->dyn_tag = c->host_tag_var[di];
}

static void sm_emit_one(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    if (in->op == POP_MADWIDE && in->ty != PTY_U16 &&
        in->src[2].kind == OPD_REG) {
        reg_t *base = find_reg(c, in->src[2].reg);
        if (base && base->is_ptr) {
            sm_emit_madwide_ptr(c, b, in);
            return;
        }
    }
    if (in->op == POP_SELP && in->ty == PTY_U64 &&
        in->src[0].kind == OPD_REG && in->src[1].kind == OPD_REG) {
        reg_t *r0 = find_reg(c, in->src[0].reg);
        reg_t *r1 = find_reg(c, in->src[1].reg);
        if (r0 && r1 && (r0->is_ptr || r1->is_ptr)) {
            sm_emit_selp_ptr(c, b, in);
            return;
        }
    }
    if ((in->op == POP_LD || in->op == POP_ST) &&
        in->src[0].kind == OPD_ADDR && in->src[0].addr_reg[0]) {
        int ri = sm_reg_index(c, in->src[0].addr_reg);
        if (ri >= 0 && c->host_dynamic[ri]) {
            sm_emit_dyn_mem(c, b, in);
            return;
        }
    }
    if ((in->op == POP_ATOM || in->op == POP_ATOMCAS) &&
        in->src[0].kind == OPD_ADDR && in->src[0].addr_reg[0]) {
        int ri = sm_reg_index(c, in->src[0].addr_reg);
        if (ri >= 0 && c->host_dynamic[ri]) {
            sm_emit_dyn_mem(c, b, in);
            return;
        }
    }
    emit_instr(c, b, in);
    sm_update_tag(c, b, in);
}

static void sm_emit_case(ctx_t *c, spv_blob_t *b, const ptx_func_t *fn,
                         int bi, uint32_t dispatch_done, uint32_t merge) {
    const ptx_block_t *blk = &fn->blocks[bi];
    put(c, b, (2u << 16) | S_Label);
    put(c, b, c->block_ids[bi]);

    int host_last = blk->count > 0 &&
                    blk->instrs[blk->count - 1].op == POP_HOSTCALL;
    if (host_last) {
        for (size_t i = 0; i + 1 < blk->count; i++) {
            ptx_op_t op = blk->instrs[i].op;
            if (op == POP_BRA || op == POP_CBRA || op == POP_RET ||
                op == POP_EXIT || op == POP_HOSTCALL)
                continue;
            sm_emit_one(c, b, &blk->instrs[i]);
            if (c->failed)
                return;
        }
        const ptx_instr_t *call = &blk->instrs[blk->count - 1];
        uint32_t index = sm_index(c, b, call->line);
        if (call->pred[0]) {
            uint32_t cond = sm_cond(c, b, call);
            uint32_t arm = nid(c), skip = nid(c), selm = nid(c);
            put(c, b, (3u << 16) | S_SelectionMerge);
            put(c, b, selm);
            put(c, b, 0);
            put(c, b, (4u << 16) | S_BranchCond);
            put(c, b, cond);
            put(c, b, arm);
            put(c, b, skip);
            put(c, b, (2u << 16) | S_Label);
            put(c, b, arm);
            sm_emit_host_call(c, b, call, index);
            put(c, b, (2u << 16) | S_Label);
            put(c, b, skip);
            put(c, b, (2u << 16) | S_Branch);
            put(c, b, selm);
            put(c, b, (2u << 16) | S_Label);
            put(c, b, selm);
            sm_finish_block(c, b, const_u32(c, (uint32_t)call->host_cont),
                            0, dispatch_done);
            return;
        }
        sm_emit_host_call(c, b, call, index);
        return;
    }

    // Ordinary straight-line body (control flow is terminal).
    size_t n = blk->count;
    ptx_op_t term = POP_RET;
    int has_term = 0;
    if (n > 0) {
        ptx_op_t last = blk->instrs[n - 1].op;
        if (last == POP_BRA || last == POP_CBRA || last == POP_RET ||
            last == POP_EXIT) {
            term = last;
            has_term = 1;
            n--;
        }
    }
    for (size_t i = 0; i < n; i++) {
        ptx_op_t op = blk->instrs[i].op;
        if (op == POP_BRA || op == POP_CBRA || op == POP_RET ||
            op == POP_EXIT || op == POP_HOSTCALL)
            continue;
        sm_emit_one(c, b, &blk->instrs[i]);
        if (c->failed)
            return;
    }
    if (has_term && (term == POP_RET || term == POP_EXIT)) {
        // Mark this invocation done so a later segment does not
        // re-enter it at the last PC.
        uint32_t index = sm_index(c, b, blk->instrs[blk->count - 1].line);
        cparam_t *st = sm_param(c, "__acuda_host_state",
                                blk->instrs[blk->count - 1].line);
        if (!st)
            return;
        uint32_t poff = sm_off(c, b, index, 0);
        sm_store_u32(c, b, st->binding, poff, const_u32(c, 0xffffffffu));
        put(c, b, (2u << 16) | S_Branch);
        put(c, b, merge);
        return;
    }
    if (has_term && term == POP_BRA) {
        int t = sm_target_index(c, fn, blk->instrs[blk->count - 1].target,
                                blk->instrs[blk->count - 1].line);
        if (c->failed)
            return;
        sm_finish_block(c, b, const_u32(c, (uint32_t)t), 0,
                        dispatch_done);
        return;
    }
    if (has_term && term == POP_CBRA) {
        const ptx_instr_t *br = &blk->instrs[blk->count - 1];
        uint32_t cond = sm_cond(c, b, br);
        int t = sm_target_index(c, fn, br->target, br->line);
        if (c->failed)
            return;
        uint32_t pc = nid(c);
        put(c, b, (6u << 16) | S_Select);
        put(c, b, c->t_u32);
        put(c, b, pc);
        put(c, b, cond);
        put(c, b, const_u32(c, (uint32_t)t));
        put(c, b, const_u32(c, (uint32_t)(bi + 1)));
        sm_finish_block(c, b, pc, 0, dispatch_done);
        return;
    }
    // Fallthrough.
    sm_finish_block(c, b, const_u32(c, (uint32_t)(bi + 1)), 0,
                    dispatch_done);
}

void emit_host_machine(ctx_t *c, const ptx_func_t *fn) {
    uint32_t entry = nid(c), loop = nid(c), dispatch = nid(c);
    uint32_t cont = nid(c), merge = nid(c), done = nid(c), load = nid(c);

    // Function variables must live in the first block.
    put(c, &c->fn, (2u << 16) | S_Label);
    put(c, &c->fn, entry);
    c->host_pc_var = sm_var(c, &c->fn, c->t_ptr_fn_u32);
    for (int i = 0; i < c->nregs; i++) {
        c->host_need[i] = 1;
        c->host_reg_var[i] = sm_var(c, &c->fn, sm_reg_ptr_ty(c, i));
        c->host_word_var[i] = 0;
        if (c->host_dynamic[i])
            c->host_tag_var[i] = sm_var(c, &c->fn, c->t_ptr_fn_u32);
    }

    if (c->uses_swbar)
        emit_swbar_init(c, &c->fn);

    // Entry: load PC and register file from the state SSBO.
    uint32_t index = sm_index(c, &c->fn, 0);
    cparam_t *st = sm_param(c, "__acuda_host_state", 0);
    if (!st)
        return;
    // Initialize the per-invocation .global image area from the
    // module's declared bytes (the device image is read-only).
    for (int gi = 0; gi < c->nglobals; gi++) {
        const ptx_const_t *g = &c->globals[gi];
        size_t words = (g->nbytes + 3u) / 4u;
        for (size_t w = 0; w < words; w++) {
            uint32_t v = 0;
            for (int k = 0; k < 4; k++) {
                size_t pos = w * 4 + (size_t)k;
                if (pos < g->nbytes)
                    v |= (uint32_t)g->bytes[pos] << (8 * k);
            }
            uint32_t goff = sm_off(c, &c->fn, index,
                                   c->host_global_off[gi] +
                                       (uint32_t)w * 4u);
            sm_store_u32(c, &c->fn, st->binding, goff, const_u32(c, v));
        }
    }
    uint32_t pc_off = sm_off(c, &c->fn, index, 0);
    uint32_t pcv = sm_load_u32(c, &c->fn, st->binding, pc_off);
    sm_store_var(c, &c->fn, c->host_pc_var, pcv);
    for (int i = 0; i < c->nregs; i++) {
        uint32_t off = sm_off(c, &c->fn, index,
                              ACUDA_HOST_STATE_HEADER +
                                  (uint32_t)i * ACUDA_HOST_STATE_REG_STRIDE +
                                  8);
        uint32_t v64 = load_u64(c, &c->fn, st->binding, off);
        int home = c->host_home[i];
        ptx_ty_t decl = c->regs[i].decl_ty;
        uint32_t v;
        if (home_is_ptr(home))
            v = v64;
        else if (decl == PTY_PRED) {
            uint32_t bv = nid(c);
            put(c, &c->fn, (5u << 16) | S_INotEqual);
            put(c, &c->fn, c->t_bool);
            put(c, &c->fn, bv);
            put(c, &c->fn, v64);
            put(c, &c->fn, const_u64(c, 0));
            v = bv;
        } else {
            v = sm_u64_to_reg(c, &c->fn, decl, v64);
        }
        sm_store_var(c, &c->fn, c->host_reg_var[i], v);
        if (c->host_dynamic[i]) {
            uint32_t toff = sm_off(c, &c->fn, index,
                                   ACUDA_HOST_STATE_HEADER +
                                       (uint32_t)i *
                                           ACUDA_HOST_STATE_REG_STRIDE + 4);
            uint32_t tag = sm_load_u32(c, &c->fn, st->binding, toff);
            sm_store_var(c, &c->fn, c->host_tag_var[i], tag);
        }
    }
    put(c, &c->fn, (2u << 16) | S_Branch);
    put(c, &c->fn, loop);

    // Loop header: reload state and dispatch on the PC.
    put(c, &c->fn, (2u << 16) | S_Label);
    put(c, &c->fn, loop);
    put(c, &c->fn, (4u << 16) | S_LoopMerge);
    put(c, &c->fn, merge);
    put(c, &c->fn, cont);
    put(c, &c->fn, 0);
    put(c, &c->fn, (2u << 16) | S_Branch);
    put(c, &c->fn, load);

    put(c, &c->fn, (2u << 16) | S_Label);
    put(c, &c->fn, load);
    sm_define_regs(c, &c->fn);
    reg_t *hdr_regs = malloc(sizeof(*hdr_regs) *
                             (c->nregs ? (size_t)c->nregs : 1));
    if (!hdr_regs) {
        fail(c, "out of memory");
        return;
    }
    memcpy(hdr_regs, c->regs, sizeof(*hdr_regs) * (size_t)c->nregs);
    uint32_t pc = sm_load_var(c, &c->fn, c->t_u32, c->host_pc_var);
    put(c, &c->fn, (2u << 16) | S_Branch);
    put(c, &c->fn, dispatch);

    // Switch dispatch.
    put(c, &c->fn, (2u << 16) | S_Label);
    put(c, &c->fn, dispatch);
    put(c, &c->fn, (3u << 16) | S_SelectionMerge);
    put(c, &c->fn, done);
    put(c, &c->fn, 0);
    put(c, &c->fn, (uint32_t)((3u + 2u * (uint32_t)fn->nblocks) << 16) |
                       S_Switch);
    put(c, &c->fn, pc);
    put(c, &c->fn, done);
    for (size_t i = 0; i < fn->nblocks; i++) {
        put(c, &c->fn, (uint32_t)i);
        put(c, &c->fn, c->block_ids[i]);
    }

    for (size_t i = 0; i < fn->nblocks && !c->failed; i++) {
        memcpy(c->regs, hdr_regs,
               sizeof(*hdr_regs) * (size_t)c->nregs);
        // Built-in values cached in an earlier case do not dominate
        // this one; redefine them locally.
        c->c_tid = 0;
        c->c_ctaid = 0;
        c->c_ntid = 0;
        sm_emit_case(c, &c->fn, fn, (int)i, done, merge);
    }
    free(hdr_regs);

    put(c, &c->fn, (2u << 16) | S_Label);
    put(c, &c->fn, done);
    uint32_t finished = nid(c);
    put(c, &c->fn, (5u << 16) | S_IEqual);
    put(c, &c->fn, c->t_bool);
    put(c, &c->fn, finished);
    put(c, &c->fn, pc);
    put(c, &c->fn, const_u32(c, 0xffffffffu));
    put(c, &c->fn, (4u << 16) | S_BranchCond);
    put(c, &c->fn, finished);
    put(c, &c->fn, merge);
    put(c, &c->fn, cont);

    put(c, &c->fn, (2u << 16) | S_Label);
    put(c, &c->fn, cont);
    put(c, &c->fn, (2u << 16) | S_Branch);
    put(c, &c->fn, loop);

    put(c, &c->fn, (2u << 16) | S_Label);
    put(c, &c->fn, merge);
    put(c, &c->fn, (1u << 16) | S_Return);
}

// Store every live SSA register into its Function variable and jump
// to the state-machine dispatcher.
static void cfg_finish_block(ctx_t *c, spv_blob_t *b, uint32_t pc,
                             uint32_t done) {
    sm_save_regs(c, b);
    if (c->failed)
        return;
    sm_store_var(c, b, c->host_pc_var, pc);
    put(c, b, (2u << 16) | S_Branch);
    put(c, b, done);
}

// One PTX block as a switch case of the state machine.
static void cfg_emit_case(ctx_t *c, spv_blob_t *b, const ptx_func_t *fn,
                          int bi, uint32_t done) {
    const ptx_block_t *blk = &fn->blocks[bi];
    put(c, b, (2u << 16) | S_Label);
    put(c, b, c->block_ids[bi]);
    size_t n = blk->count;
    ptx_op_t term = POP_RET;
    int has_term = 0;
    if (n > 0) {
        ptx_op_t last = blk->instrs[n - 1].op;
        if (last == POP_BRA || last == POP_CBRA || last == POP_RET ||
            last == POP_EXIT) {
            term = last;
            has_term = 1;
            n--;
        }
    }
    for (size_t i = 0; i < n; i++) {
        ptx_op_t op = blk->instrs[i].op;
        if (op == POP_BRA || op == POP_CBRA || op == POP_RET ||
            op == POP_EXIT || op == POP_HOSTCALL)
            continue;
        sm_emit_one(c, b, &blk->instrs[i]);
        if (c->failed)
            return;
    }
    if (!has_term) {
        uint32_t next = bi + 1 < (int)fn->nblocks
                            ? (uint32_t)(bi + 1) : 0xffffffffu;
        cfg_finish_block(c, b, const_u32(c, next), done);
        return;
    }
    if (term == POP_RET || term == POP_EXIT) {
        cfg_finish_block(c, b, const_u32(c, 0xffffffffu), done);
        return;
    }
    if (term == POP_BRA) {
        const ptx_instr_t *br = &blk->instrs[blk->count - 1];
        int t = sm_target_index(c, fn, br->target, br->line);
        if (c->failed)
            return;
        cfg_finish_block(c, b, const_u32(c, (uint32_t)t), done);
        return;
    }
    if (term == POP_CBRA) {
        const ptx_instr_t *br = &blk->instrs[blk->count - 1];
        uint32_t cond = sm_cond(c, b, br);
        int t = sm_target_index(c, fn, br->target, br->line);
        if (c->failed)
            return;
        uint32_t pc = nid(c);
        put(c, b, (6u << 16) | S_Select);
        put(c, b, c->t_u32);
        put(c, b, pc);
        put(c, b, cond);
        put(c, b, const_u32(c, (uint32_t)t));
        put(c, b, const_u32(c, (uint32_t)(bi + 1)));
        cfg_finish_block(c, b, pc, done);
        return;
    }
    failf(c, blk->instrs[blk->count - 1].line,
          "unsupported block terminator in state machine");
}

// ---- State-machine register liveness ---------------------------------
// The generic CFG machine checkpoints registers at every block
// boundary.  Lifted SASS declares thousands of single-use temporaries
// (%rt/%rdt/%ps/...); checkpointing all of them bloats the shader
// enormously.  Only a register that is live across a control-flow edge
// needs a state slot.

static void sm_bitset(uint64_t *set, int i) {
    set[i >> 6] |= 1ull << (i & 63);
}

static int sm_bitget(const uint64_t *set, int i) {
    return (int)((set[i >> 6] >> (i & 63)) & 1ull);
}

// Successors of a block under the state machine's terminator rules.
static int sm_block_succs(ctx_t *c, const ptx_func_t *fn, int bi,
                          int out[2]) {
    int n = 0;
    const ptx_block_t *b = &fn->blocks[bi];
    if (b->count > 0) {
        const ptx_instr_t *last = &b->instrs[b->count - 1];
        if (last->op == POP_BRA) {
            int t = block_index_of(c, fn, last->target);
            if (t >= 0)
                out[n++] = t;
            return n;
        }
        if (last->op == POP_CBRA) {
            int t = block_index_of(c, fn, last->target);
            if (t >= 0)
                out[n++] = t;
            if (bi + 1 < (int)fn->nblocks)
                out[n++] = bi + 1;
            return n;
        }
        if (last->op == POP_RET || last->op == POP_EXIT)
            return 0;
    }
    if (bi + 1 < (int)fn->nblocks)
        out[n++] = bi + 1;
    return n;
}

// Mark every operand that names a register as a use/def in one block.
static void sm_mark_instr_refs(ctx_t *c, const ptx_instr_t *in,
                               uint64_t *uses, uint64_t *defs) {
    {
        if (in->pred[0]) {
            int ri = sm_reg_index(c, in->pred);
            if (ri >= 0)
                sm_bitset(uses, ri);
        }
        for (int s = 0; s < in->nsrc; s++) {
            const ptx_opd_t *op = &in->src[s];
            const char *nm = NULL;
            if (op->kind == OPD_REG)
                nm = op->reg;
            else if (op->kind == OPD_ADDR)
                nm = op->addr_reg;
            else if (op->kind == OPD_REGOFF)
                nm = op->reg;
            if (!nm || !nm[0])
                continue;
            int ri = sm_reg_index(c, nm);
            if (ri >= 0)
                sm_bitset(uses, ri);
        }
        if (in->smreg[0]) {
            int ri = sm_reg_index(c, in->smreg);
            if (ri >= 0)
                sm_bitset(uses, ri);
        }
    }
    // Standard scalar destinations are written by this instruction.
    int ri = sm_reg_index(c, in->dst);
    if (ri >= 0)
        sm_bitset(defs, ri);
    ri = sm_reg_index(c, in->dst2);
    if (ri >= 0)
        sm_bitset(defs, ri);
    // Fragment/vector operand fields mix destinations and sources.  Treat
    // them as reads: that only over-approximates liveness (a state slot
    // that is not strictly needed), never drops a live value.
    for (int m = 0; m < 20; m++) {
        if (!in->mregs[m][0])
            continue;
        ri = sm_reg_index(c, in->mregs[m]);
        if (ri >= 0)
            sm_bitset(uses, ri);
    }
    for (int k = 0; k < 4; k++)
        for (int m = 0; m < 8; m++) {
            if (!in->wregs[k][m][0])
                continue;
            ri = sm_reg_index(c, in->wregs[k][m]);
            if (ri >= 0)
                sm_bitset(uses, ri);
        }
    for (int k = 0; k < 4; k++) {
        if (in->tex_dst[k][0]) {
            ri = sm_reg_index(c, in->tex_dst[k]);
            if (ri >= 0)
                sm_bitset(uses, ri);
        }
        if (in->surf_dst[k][0]) {
            ri = sm_reg_index(c, in->surf_dst[k]);
            if (ri >= 0)
                sm_bitset(uses, ri);
        }
    }
}

// needs[i] = 1 when register i must be checkpointed across a block edge.
static int sm_compute_state_regs(ctx_t *c, const ptx_func_t *fn,
                                 uint8_t *needs) {
    int nr = c->nregs;
    int nb = (int)fn->nblocks;
    size_t W = (size_t)((nr + 63) / 64);
    if (W == 0)
        W = 1;
    uint64_t *use = calloc((size_t)nb * W, sizeof(uint64_t));
    uint64_t *def = calloc((size_t)nb * W, sizeof(uint64_t));
    uint64_t *lin = calloc((size_t)nb * W, sizeof(uint64_t));
    uint64_t *lout = calloc((size_t)nb * W, sizeof(uint64_t));
    if (!use || !def || !lin || !lout) {
        free(use); free(def); free(lin); free(lout);
        return -1;
    }
    uint64_t *tu = calloc(W, sizeof(uint64_t));
    uint64_t *td = calloc(W, sizeof(uint64_t));
    uint64_t *seen = calloc(W, sizeof(uint64_t));
    if (!tu || !td || !seen) {
        free(use); free(def); free(lin); free(lout);
        free(tu); free(td); free(seen);
        return -1;
    }
    for (int b = 0; b < nb; b++) {
        uint64_t *ub = use + (size_t)b * W;
        uint64_t *db = def + (size_t)b * W;
        const ptx_block_t *blk = &fn->blocks[b];
        memset(seen, 0, W * sizeof(uint64_t));
        for (size_t ii = 0; ii < blk->count; ii++) {
            const ptx_instr_t *in = &blk->instrs[ii];
            memset(tu, 0, W * sizeof(uint64_t));
            memset(td, 0, W * sizeof(uint64_t));
            sm_mark_instr_refs(c, in, tu, td);
            for (size_t w = 0; w < W; w++) {
                // A read before the first write in this block is
                // live-in; earlier definitions kill the use.
                ub[w] |= tu[w] & ~seen[w];
                seen[w] |= td[w];
                db[w] |= td[w];
            }
        }
    }
    free(tu);
    free(td);
    free(seen);
    // Fixpoint: live_out = union(live_in[succ]);
    //           live_in  = use | (live_out & ~def).
    int changed = 1;
    while (changed) {
        changed = 0;
        for (int b = 0; b < nb; b++) {
            int succ[2];
            int ns = sm_block_succs(c, fn, b, succ);
            uint64_t *ob = lout + (size_t)b * W;
            uint64_t *ib = lin + (size_t)b * W;
            uint64_t *ub = use + (size_t)b * W;
            uint64_t *db = def + (size_t)b * W;
            for (size_t w = 0; w < W; w++) {
                uint64_t no = 0;
                for (int s = 0; s < ns; s++)
                    no |= lin[(size_t)succ[s] * W + w];
                if (no != ob[w]) {
                    ob[w] = no;
                    changed = 1;
                }
                uint64_t ni = ub[w] | (ob[w] & ~db[w]);
                if (ni != ib[w]) {
                    ib[w] = ni;
                    changed = 1;
                }
            }
        }
    }
    for (int b = 0; b < nb; b++) {
        uint64_t *ib = lin + (size_t)b * W;
        for (int i = 0; i < nr; i++)
            if (sm_bitget(ib, i))
                needs[i] = 1;
    }
    free(use);
    free(def);
    free(lin);
    free(lout);
    return 0;
}

// Generic CFG lowering: every PTX block becomes a case in one dispatch
// loop, and register state lives in Function variables.  This models
// arbitrary (including irreducible) control flow without relying on
// SPIR-V structured constructs per branch.
void emit_cfg_machine(ctx_t *c, const ptx_func_t *fn) {
    uint32_t entry = nid(c), loop = nid(c), load = nid(c);
    uint32_t dispatch = nid(c), cont = nid(c), merge = nid(c);
    uint32_t done = nid(c);
    // Lockstep mode: only the lanes holding the minimum program counter
    // in the selected scope execute their block this iteration.  Lanes
    // behind catch up before any lane advances past the join point, which
    // makes subgroup shuffles/votes and workgroup barriers see coherent
    // progress.  Workgroup-wide barriers require workgroup scope.
    int lockstep = c->uses_state_lockstep;

    // Function variables must be declared in the first block.
    put(c, &c->fn, (2u << 16) | S_Label);
    put(c, &c->fn, entry);
    c->host_pc_var = sm_var(c, &c->fn, c->t_ptr_fn_u32);
    memset(c->host_need, 0, sizeof(c->host_need));
    if (sm_compute_state_regs(c, fn, c->host_need) != 0) {
        fail(c, "out of memory");
        return;
    }
    for (int i = 0; i < c->nregs; i++) {
        // Dynamic-home registers carry a runtime tag, so their slot and
        // tag variable always exist.
        if (c->host_dynamic[i])
            c->host_need[i] = 1;
        c->host_word_var[i] = 0;
        if (!c->host_need[i]) {
            c->host_reg_var[i] = 0;
            c->host_tag_var[i] = 0;
            continue;
        }
        c->host_reg_var[i] = sm_var(c, &c->fn, sm_reg_ptr_ty(c, i));
        if (c->host_dynamic[i]) {
            c->host_tag_var[i] = sm_var(c, &c->fn, c->t_ptr_fn_u32);
            // Dynamic 32-bit registers can hold a pointer word and an
            // address offset at the same time; keep the word too.
            if (decl_int_width(c->regs[i].decl_ty) == 32)
                c->host_word_var[i] =
                    sm_var(c, &c->fn, c->t_ptr_fn_u32);
        }
    }
    if (c->uses_swbar)
        emit_swbar_init(c, &c->fn);
    if (c->failed)
        return;
    sm_store_var(c, &c->fn, c->host_pc_var, const_u32(c, 0));
    put(c, &c->fn, (2u << 16) | S_Branch);
    put(c, &c->fn, loop);

    // Loop header and state reload.
    put(c, &c->fn, (2u << 16) | S_Label);
    put(c, &c->fn, loop);
    put(c, &c->fn, (4u << 16) | S_LoopMerge);
    put(c, &c->fn, merge);
    put(c, &c->fn, cont);
    put(c, &c->fn, 0);
    put(c, &c->fn, (2u << 16) | S_Branch);
    put(c, &c->fn, load);

    // A single switch with one case per PTX block gives every lifted
    // register a live range spanning the whole dispatch; on large
    // functions (compute_mode has 932 blocks) ACO's register allocator
    // then runs off the rails.  Split the dispatch into fixed-size
    // chunks: the per-block register reload only happens inside the
    // chunk that is actually executing, so live ranges span at most one
    // chunk.  Semantics are unchanged (same reload before the case and
    // same save at its end).
    enum { CFG_CHUNK_BLOCKS = 32 };
    int chunked = fn->nblocks > (size_t)(2 * CFG_CHUNK_BLOCKS);
    put(c, &c->fn, (2u << 16) | S_Label);
    put(c, &c->fn, load);
    if (!chunked)
        sm_define_regs(c, &c->fn);
    if (c->failed)
        return;
    reg_t *hdr_regs = malloc(sizeof(*hdr_regs) *
                             (c->nregs ? (size_t)c->nregs : 1));
    if (!hdr_regs) {
        fail(c, "out of memory");
        return;
    }
    memcpy(hdr_regs, c->regs, sizeof(*hdr_regs) * (size_t)c->nregs);
    uint32_t pc = sm_load_var(c, &c->fn, c->t_u32, c->host_pc_var);
    put(c, &c->fn, (2u << 16) | S_Branch);
    put(c, &c->fn, dispatch);

    // Dispatch on the block index.  In lockstep mode a lane only runs its
    // block when it holds the minimum PC in the selected scope; the
    // others are sent to the switch default (the merge block), which
    // loops them back.
    put(c, &c->fn, (2u << 16) | S_Label);
    put(c, &c->fn, dispatch);
    uint32_t dpc = pc;
    if (lockstep) {
        uint32_t minpc = nid(c);
        if (c->uses_state_lockstep_wg && c->lockstep_min_var) {
            // Workgroup-wide min-PC reduction: OpGroupNonUniformUMin cannot
            // take Workgroup scope, so reduce via shared atomicMin with
            // barriers (all invocations participate every iteration).
            uint32_t wg = const_u32(c, 2u);     // Workgroup
            uint32_t sem_acqrel_wg = const_u32(c, 0x108u); // AcqRel|WorkgroupMemory
            uint32_t sem_relaxed = const_u32(c, 0u);
            // Reset min for this iteration: lane 0 stores UINT_MAX.
            uint32_t tid0 = special_comp(c, &c->fn, SPC_TID, 0, 0);
            uint32_t tid1 = special_comp(c, &c->fn, SPC_TID, 1, 0);
            uint32_t tid2 = special_comp(c, &c->fn, SPC_TID, 2, 0);
            uint32_t isx0 = nid(c), isy0 = nid(c), isz0 = nid(c), isxy0 = nid(c), is0 = nid(c);
            put(c, &c->fn, (5u << 16) | S_IEqual);
            put(c, &c->fn, c->t_bool); put(c, &c->fn, isx0); put(c, &c->fn, tid0); put(c, &c->fn, const_u32(c, 0));
            put(c, &c->fn, (5u << 16) | S_IEqual);
            put(c, &c->fn, c->t_bool); put(c, &c->fn, isy0); put(c, &c->fn, tid1); put(c, &c->fn, const_u32(c, 0));
            put(c, &c->fn, (5u << 16) | S_IEqual);
            put(c, &c->fn, c->t_bool); put(c, &c->fn, isz0); put(c, &c->fn, tid2); put(c, &c->fn, const_u32(c, 0));
            put(c, &c->fn, (5u << 16) | S_LogicalAnd);
            put(c, &c->fn, c->t_bool); put(c, &c->fn, isxy0); put(c, &c->fn, isx0); put(c, &c->fn, isy0);
            put(c, &c->fn, (5u << 16) | S_LogicalAnd);
            put(c, &c->fn, c->t_bool); put(c, &c->fn, is0); put(c, &c->fn, isxy0); put(c, &c->fn, isz0);
            uint32_t rst_l = nid(c), rst_m = nid(c);
            put(c, &c->fn, (3u << 16) | S_SelectionMerge);
            put(c, &c->fn, rst_m); put(c, &c->fn, 0);
            put(c, &c->fn, (4u << 16) | S_BranchCond);
            put(c, &c->fn, is0); put(c, &c->fn, rst_l); put(c, &c->fn, rst_m);
            put(c, &c->fn, (2u << 16) | S_Label); put(c, &c->fn, rst_l);
            put(c, &c->fn, (3u << 16) | S_Store);
            put(c, &c->fn, c->lockstep_min_var); put(c, &c->fn, const_u32(c, 0xffffffffu));
            put(c, &c->fn, (2u << 16) | S_Branch); put(c, &c->fn, rst_m);
            put(c, &c->fn, (2u << 16) | S_Label); put(c, &c->fn, rst_m);
            put(c, &c->fn, (4u << 16) | S_ControlBarrier);
            put(c, &c->fn, wg); put(c, &c->fn, wg); put(c, &c->fn, sem_acqrel_wg);
            // atomicMin(&lockstep_min_var, pc)
            uint32_t oldv = nid(c);
            put(c, &c->fn, (7u << 16) | S_AtomicUMin);
            put(c, &c->fn, c->t_u32); put(c, &c->fn, oldv);
            put(c, &c->fn, c->lockstep_min_var);
            put(c, &c->fn, wg); put(c, &c->fn, sem_acqrel_wg); put(c, &c->fn, pc);
            put(c, &c->fn, (4u << 16) | S_ControlBarrier);
            put(c, &c->fn, wg); put(c, &c->fn, wg); put(c, &c->fn, sem_acqrel_wg);
            minpc = sm_load_var(c, &c->fn, c->t_u32, c->lockstep_min_var);
        } else {
            put(c, &c->fn, (6u << 16) | S_GroupUMin);
            put(c, &c->fn, c->t_u32);
            put(c, &c->fn, minpc);
            // Execution scope: Subgroup (3).
            put(c, &c->fn, const_u32(c, 3u));
            put(c, &c->fn, 0u);              // GroupOperation: Reduce
            put(c, &c->fn, pc);
        }
        uint32_t ismin = nid(c);
        put(c, &c->fn, (5u << 16) | S_IEqual);
        put(c, &c->fn, c->t_bool);
        put(c, &c->fn, ismin);
        put(c, &c->fn, pc);
        put(c, &c->fn, minpc);
        uint32_t finished = nid(c);
        put(c, &c->fn, (5u << 16) | S_IEqual);
        put(c, &c->fn, c->t_bool);
        put(c, &c->fn, finished);
        put(c, &c->fn, pc);
        put(c, &c->fn, const_u32(c, 0xffffffffu));
        uint32_t notfin = nid(c);
        put(c, &c->fn, (4u << 16) | S_LogicalNot);
        put(c, &c->fn, c->t_bool);
        put(c, &c->fn, notfin);
        put(c, &c->fn, finished);
        uint32_t allow = nid(c);
        put(c, &c->fn, (5u << 16) | S_LogicalAnd);
        put(c, &c->fn, c->t_bool);
        put(c, &c->fn, allow);
        put(c, &c->fn, ismin);
        put(c, &c->fn, notfin);
        dpc = nid(c);
        put(c, &c->fn, (6u << 16) | S_Select);
        put(c, &c->fn, c->t_u32);
        put(c, &c->fn, dpc);
        put(c, &c->fn, allow);
        put(c, &c->fn, pc);
        put(c, &c->fn, const_u32(c, 0xffffffffu));
    }
    if (!chunked) {
        put(c, &c->fn, (3u << 16) | S_SelectionMerge);
        put(c, &c->fn, done);
        put(c, &c->fn, 0);
        put(c, &c->fn, (uint32_t)((3u + 2u * (uint32_t)fn->nblocks) << 16) |
                           S_Switch);
        put(c, &c->fn, dpc);
        put(c, &c->fn, done);
        for (size_t i = 0; i < fn->nblocks; i++) {
            put(c, &c->fn, (uint32_t)i);
            put(c, &c->fn, c->block_ids[i]);
        }
        for (size_t i = 0; i < fn->nblocks && !c->failed; i++) {
            memcpy(c->regs, hdr_regs,
                   sizeof(*hdr_regs) * (size_t)c->nregs);
            c->c_tid = 0;
            c->c_ctaid = 0;
            c->c_ntid = 0;
            cfg_emit_case(c, &c->fn, fn, (int)i, done);
        }
    } else {
        int nchunks = ((int)fn->nblocks + CFG_CHUNK_BLOCKS - 1) /
                      CFG_CHUNK_BLOCKS;
        uint32_t *chunk_lbl = calloc((size_t)nchunks, sizeof(uint32_t));
        uint32_t *chunk_done = calloc((size_t)nchunks, sizeof(uint32_t));
        if (!chunk_lbl || !chunk_done) {
            free(chunk_lbl);
            free(chunk_done);
            free(hdr_regs);
            fail(c, "out of memory");
            return;
        }
        for (int q = 0; q < nchunks; q++) {
            chunk_lbl[q] = nid(c);
            chunk_done[q] = nid(c);
        }
        unsigned shift = 5; // log2(CFG_CHUNK_BLOCKS)
        uint32_t cidx = nid(c);
        emit_arith(c, &c->fn, S_ShrL, c->t_u32, cidx, dpc,
                   const_u32(c, shift));
        put(c, &c->fn, (3u << 16) | S_SelectionMerge);
        put(c, &c->fn, done);
        put(c, &c->fn, 0);
        put(c, &c->fn, (uint32_t)((3u + 2u * (uint32_t)nchunks) << 16) |
                           S_Switch);
        put(c, &c->fn, cidx);
        put(c, &c->fn, done);
        for (int q = 0; q < nchunks; q++) {
            put(c, &c->fn, (uint32_t)q);
            put(c, &c->fn, chunk_lbl[q]);
        }
        for (int q = 0; q < nchunks && !c->failed; q++) {
            int lo = q * CFG_CHUNK_BLOCKS;
            int hi = lo + CFG_CHUNK_BLOCKS;
            if ((size_t)hi > fn->nblocks)
                hi = (int)fn->nblocks;
            put(c, &c->fn, (2u << 16) | S_Label);
            put(c, &c->fn, chunk_lbl[q]);
            sm_define_regs(c, &c->fn);
            if (c->failed)
                break;
            // Snapshot the canonical post-load state: each case starts
            // from it, like the header reload in the flat dispatch.
            memcpy(hdr_regs, c->regs,
                   sizeof(*hdr_regs) * (size_t)c->nregs);
            uint32_t inner = nid(c);
            emit_arith(c, &c->fn, S_And, c->t_u32, inner, dpc,
                       const_u32(c, (uint32_t)(CFG_CHUNK_BLOCKS - 1)));
            put(c, &c->fn, (3u << 16) | S_SelectionMerge);
            put(c, &c->fn, chunk_done[q]);
            put(c, &c->fn, 0);
            put(c, &c->fn, (uint32_t)((3u + 2u * (uint32_t)(hi - lo)) << 16) |
                               S_Switch);
            put(c, &c->fn, inner);
            put(c, &c->fn, chunk_done[q]);
            for (int i = lo; i < hi; i++) {
                put(c, &c->fn, (uint32_t)(i - lo));
                put(c, &c->fn, c->block_ids[i]);
            }
            for (int i = lo; i < hi && !c->failed; i++) {
                memcpy(c->regs, hdr_regs,
                       sizeof(*hdr_regs) * (size_t)c->nregs);
                c->c_tid = 0;
                c->c_ctaid = 0;
                c->c_ntid = 0;
                cfg_emit_case(c, &c->fn, fn, i, chunk_done[q]);
            }
            put(c, &c->fn, (2u << 16) | S_Label);
            put(c, &c->fn, chunk_done[q]);
            put(c, &c->fn, (2u << 16) | S_Branch);
            put(c, &c->fn, done);
        }
        free(chunk_lbl);
        free(chunk_done);
    }
    free(hdr_regs);
    if (c->failed)
        return;

    // The dispatcher merge decides between returning and looping.  The
    // PC variable is reloaded because cases store the next PC.
    put(c, &c->fn, (2u << 16) | S_Label);
    put(c, &c->fn, done);
    uint32_t pc2 = sm_load_var(c, &c->fn, c->t_u32, c->host_pc_var);
    uint32_t finished = nid(c);
    if (c->uses_state_lockstep_wg && c->lockstep_min_var) {
        // All invocations must keep participating in the dispatch loop
        // (and its workgroup barriers) until every invocation is done.
        // Exit only when the workgroup-wide minimum PC is 0xffffffff.
        uint32_t wg = const_u32(c, 2u);
        uint32_t sem_acqrel_wg = const_u32(c, 0x108u);
        uint32_t sem_relaxed = const_u32(c, 0u);
        uint32_t tid0 = special_comp(c, &c->fn, SPC_TID, 0, 0);
        uint32_t tid1 = special_comp(c, &c->fn, SPC_TID, 1, 0);
        uint32_t tid2 = special_comp(c, &c->fn, SPC_TID, 2, 0);
        uint32_t ex0 = nid(c), ey0 = nid(c), ez0 = nid(c), exy0 = nid(c), eis0 = nid(c);
        put(c, &c->fn, (5u << 16) | S_IEqual);
        put(c, &c->fn, c->t_bool); put(c, &c->fn, ex0); put(c, &c->fn, tid0); put(c, &c->fn, const_u32(c, 0));
        put(c, &c->fn, (5u << 16) | S_IEqual);
        put(c, &c->fn, c->t_bool); put(c, &c->fn, ey0); put(c, &c->fn, tid1); put(c, &c->fn, const_u32(c, 0));
        put(c, &c->fn, (5u << 16) | S_IEqual);
        put(c, &c->fn, c->t_bool); put(c, &c->fn, ez0); put(c, &c->fn, tid2); put(c, &c->fn, const_u32(c, 0));
        put(c, &c->fn, (5u << 16) | S_LogicalAnd);
        put(c, &c->fn, c->t_bool); put(c, &c->fn, exy0); put(c, &c->fn, ex0); put(c, &c->fn, ey0);
        put(c, &c->fn, (5u << 16) | S_LogicalAnd);
        put(c, &c->fn, c->t_bool); put(c, &c->fn, eis0); put(c, &c->fn, exy0); put(c, &c->fn, ez0);
        uint32_t erst_l = nid(c), erst_m = nid(c);
        put(c, &c->fn, (3u << 16) | S_SelectionMerge);
        put(c, &c->fn, erst_m); put(c, &c->fn, 0);
        put(c, &c->fn, (4u << 16) | S_BranchCond);
        put(c, &c->fn, eis0); put(c, &c->fn, erst_l); put(c, &c->fn, erst_m);
        put(c, &c->fn, (2u << 16) | S_Label); put(c, &c->fn, erst_l);
        put(c, &c->fn, (3u << 16) | S_Store);
        put(c, &c->fn, c->lockstep_min_var); put(c, &c->fn, const_u32(c, 0xffffffffu));
        put(c, &c->fn, (2u << 16) | S_Branch); put(c, &c->fn, erst_m);
        put(c, &c->fn, (2u << 16) | S_Label); put(c, &c->fn, erst_m);
        put(c, &c->fn, (4u << 16) | S_ControlBarrier);
        put(c, &c->fn, wg); put(c, &c->fn, wg); put(c, &c->fn, sem_acqrel_wg);
        uint32_t eold = nid(c);
        put(c, &c->fn, (7u << 16) | S_AtomicUMin);
        put(c, &c->fn, c->t_u32); put(c, &c->fn, eold);
        put(c, &c->fn, c->lockstep_min_var);
        put(c, &c->fn, wg); put(c, &c->fn, sem_acqrel_wg); put(c, &c->fn, pc2);
        put(c, &c->fn, (4u << 16) | S_ControlBarrier);
        put(c, &c->fn, wg); put(c, &c->fn, wg); put(c, &c->fn, sem_acqrel_wg);
        uint32_t minpc2 = sm_load_var(c, &c->fn, c->t_u32, c->lockstep_min_var);
        put(c, &c->fn, (5u << 16) | S_IEqual);
        put(c, &c->fn, c->t_bool);
        put(c, &c->fn, finished);
        put(c, &c->fn, minpc2);
        put(c, &c->fn, const_u32(c, 0xffffffffu));
    } else {
        put(c, &c->fn, (5u << 16) | S_IEqual);
        put(c, &c->fn, c->t_bool);
        put(c, &c->fn, finished);
        put(c, &c->fn, pc2);
        put(c, &c->fn, const_u32(c, 0xffffffffu));
    }
    put(c, &c->fn, (4u << 16) | S_BranchCond);
    put(c, &c->fn, finished);
    put(c, &c->fn, merge);
    put(c, &c->fn, cont);

    put(c, &c->fn, (2u << 16) | S_Label);
    put(c, &c->fn, cont);
    put(c, &c->fn, (2u << 16) | S_Branch);
    put(c, &c->fn, loop);

    put(c, &c->fn, (2u << 16) | S_Label);
    put(c, &c->fn, merge);
    put(c, &c->fn, (1u << 16) | S_Return);
}

// Pointer-home analysis. A register gets a bitmask of possible homes.
// One bit is a static home; several global-buffer bits are a dynamic
// home resolved by a runtime tag. Mixed scalar/pointer or special-space
// merges remain clean errors when used as an address.

// Resolve a load of a packed by-value pointer parameter to the
// parameter word it actually reads.  Word 0 keeps the parameter name;
// the remaining words are bound as "name#N".  *rel receives the byte
// offset within the 8-byte word (0 = low word, 4 = high word).
static cparam_t *param_word(ctx_t *c, const char *name, int64_t off,
                            int *rel) {
    int words = 0;
    for (int q = 0; q < c->nfnparams; q++)
        if (c->fnparams[q].wide_words &&
            strcmp(c->fnparams[q].name, name) == 0) {
            words = c->fnparams[q].wide_words;
            break;
        }
    *rel = (int)(off & 7);
    if (words <= 0)
        return find_cparam(c, name);
    int part = (int)(off / 8);
    if (part < 0 || part >= words)
        return NULL;
    cparam_t *base = find_cparam(c, name);
    if (base && base->kind == 6) {
        // Kind 6 is a single cparam covering the whole mixed blob; rel is
        // the full byte offset within the blob.
        *rel = (int)off;
        return base;
    }
    if (part == 0)
        return base;
    char pn[300];
    if (snprintf(pn, sizeof(pn), "%s#%d", name, part) >= (int)sizeof(pn))
        return NULL;
    return find_cparam(c, pn);
}

static int mask_bit_index(uint64_t m) {
    int i = 0;
    while (i < 63 && !(m & 1ull)) {
        m >>= 1;
        i++;
    }
    return i;
}

static int mask_popcount(uint64_t m) {
    int n = 0;
    for (; m; m &= m - 1)
        n++;
    return n;
}

int build_host_homes(ctx_t *c, const ptx_func_t *fn, char *err,
                     size_t errlen) {
    for (int i = 0; i < c->nregs; i++) {
        c->host_home[i] = HOME_UNINIT;
        c->host_home_aux[i] = -1;
        c->host_home_mask[i] = 0;
        c->host_paddr_mask[i] = 0;
        c->host_dynamic[i] = 0;
    }
    for (int pass = 0; pass < 64; pass++) {
        int changed = 0;
        for (size_t bi = 0; bi < fn->nblocks; bi++) {
            for (size_t ii = 0; ii < fn->blocks[bi].count; ii++) {
                const ptx_instr_t *in = &fn->blocks[bi].instrs[ii];
                if (!in->dst[0])
                    continue;
                int ri = -1;
                for (int r = 0; r < c->nregs; r++)
                    if (strcmp(c->regs[r].name, in->dst) == 0) {
                        ri = r;
                        break;
                    }
                if (ri < 0)
                    continue;
                uint64_t nm = 0;
                int na = -1;
                if (in->op == POP_LD && in->space == PSP_PARAM &&
                    in->src[0].kind == OPD_ADDR && !in->src[0].addr_reg[0]) {
                    int rel = 0;
                    cparam_t *cp = param_word(c, in->src[0].param,
                                              in->src[0].addr_imm, &rel);
                    // A full-width load of a pointer param carries the
                    // pointer home; sub-word loads read an address word
                    // and are plain scalars (except the low word of a
                    // pointer word, see below). Kind 6 is the mixed
                    // 32-bit blob {ptr32, scalar32}: word 0 is a device
                    // pointer resolved through its binding like kind 1.
                    if (cp && (cp->kind == 1 || cp->kind == 6) &&
                        in->ty == PTY_U64)
                        nm = c->bda_word[cp->binding]
                                 ? HM_BDA
                                 : (1ull << (unsigned)cp->binding);
                    else if (cp && (cp->kind == 1 || cp->kind == 6) &&
                             in->ty == PTY_U32 &&
                             rel == 0 && cp->binding >= 0) {
                        // Low word of a pointer parameter (lifted 32-bit
                        // addressing): carries the pointer home so later
                        // zero-extension + global access stays on the
                        // right buffer. Value uses are unaffected (the
                        // home only guides address operands/state saves).
                        if (getenv("ACUDA_DEBUG_HALF"))
                            fprintf(stderr, "ACuda host-half: param %s word0 -> binding %d\n",
                                    in->src[0].param, cp->binding);
                        nm = c->bda_word[cp->binding]
                                 ? HM_BDA
                                 : (1ull << (unsigned)cp->binding);
                    } else {
                        // In 32-bit BDA kernels, values loaded from params
                        // (including by-value struct fields) may be device
                        // pointers used via physical/global access. Keep a
                        // scalar phase but allow a runtime BDA pointer home.
                        if (c->addr_size == 32 && c->use_bda &&
                            (in->ty == PTY_U32 || in->ty == PTY_U64))
                            nm = HM_SCALAR | HM_BDA;
                        else
                            nm = HM_SCALAR; // scalar/wide param words
                    }
                } else if (in->op == POP_LD && in->space == PSP_PARAM &&
                           in->src[0].kind == OPD_ADDR &&
                           in->src[0].addr_reg[0]) {
                    int sr = -1;
                    for (int r = 0; r < c->nregs; r++)
                        if (strcmp(c->regs[r].name,
                                   in->src[0].addr_reg) == 0) {
                            sr = r;
                            break;
                        }
                    if (sr >= 0 && c->host_paddr_mask[sr]) {
                        // Loading through a parameter address yields the
                        // parameter value: pointer params contribute
                        // their binding, scalar params are scalars.
                        uint32_t pm = c->host_paddr_mask[sr];
                        for (int p = 0; p < 32; p++) {
                            if (!(pm & (1u << p)))
                                continue;
                            cparam_t *cp = NULL;
                            for (int cp_i = 0; cp_i < c->nfnparams; cp_i++)
                                if (strcmp(c->fnparams[cp_i].name,
                                           c->fnparams[p].name) == 0) {
                                    cp = find_cparam(c,
                                        c->fnparams[cp_i].name);
                                    break;
                                }
                            if (cp && cp->kind == 1)
                                nm |= c->bda_word[cp->binding]
                                          ? HM_BDA
                                          : (1ull << (unsigned)cp->binding);
                            else
                                nm |= HM_SCALAR;
                        }
                    } else if (sr >= 0) {
                        nm = c->host_home_mask[sr];
                    }
                } else if (in->op == POP_MOV && in->nsrc == 1 &&
                           in->src[0].kind == OPD_PARAM) {
                    int found = 0;
                    for (int v = 0; v < c->nshvars; v++)
                        if (strcmp(c->shvars[v].name,
                                   in->src[0].param) == 0) {
                            nm = HM_SHARED;
                            na = v;
                            found = 1;
                            break;
                        }
                    if (!found)
                        for (int v = 0; v < c->nglobals; v++)
                            if (strcmp(c->globals[v].name,
                                       in->src[0].param) == 0) {
                                nm = HM_GLVAR;
                                na = v;
                                found = 1;
                                break;
                            }
                    if (!found)
                        for (int v = 0; v < c->nfnparams; v++)
                            if (strcmp(c->fnparams[v].name,
                                       in->src[0].param) == 0) {
                                if (c->fnparams[v].is_hidden_symbol) {
                                    nm = HM_BDA;
                                    na = -1;
                                } else {
                                    nm = HM_PADDR;
                                    na = v;
                                    c->host_paddr_mask[ri] |= 1u << (unsigned)v;
                                }
                                found = 1;
                                break;
                            }
                    if (!found)
                        for (int v = 0; v < c->nlocals; v++)
                            if (strcmp(c->locals[v].name,
                                       in->src[0].param) == 0) {
                                nm = HM_LOCAL;
                                na = v;
                                found = 1;
                                break;
                            }
                    if (!found)
                        nm = HM_SCALAR;
                } else if ((in->op == POP_MOV || in->op == POP_ADD ||
                            in->op == POP_SUB || in->op == POP_MADLO ||
                            in->op == POP_MADWIDE || in->op == POP_SELP) &&
                           in->dst[0]) {
                    // cvta.to.global from a plain value produces a raw/BDA
                    // global pointer (32-bit PTX loads pointer fields as
                    // integer values). When the source already carries a
                    // pointer home (e.g. a 32-bit descriptor binding the
                    // emitter preserves), propagate it instead so the
                    // state machine keeps the real home.
                    int force_bda = 0;
                    if (in->op == POP_MOV && in->space == PSP_GLOBAL) {
                        force_bda = 1;
                        for (int s = 0; s < in->nsrc && force_bda; s++) {
                            if (in->src[s].kind != OPD_REG)
                                continue;
                            for (int r = 0; r < c->nregs; r++) {
                                if (strcmp(c->regs[r].name,
                                           in->src[s].reg) != 0)
                                    continue;
                                if ((c->host_home_mask[r] & ~HM_SCALAR) ||
                                    c->host_paddr_mask[r])
                                    force_bda = 0;
                                break;
                            }
                        }
                    }
                    if (force_bda) {
                        nm = HM_BDA;
                    } else {
                    uint64_t nm_shared = 0;
                    int saw_shared = 0, saw_binding = 0;
                    for (int s = 0; s < in->nsrc; s++) {
                        if (in->src[s].kind != OPD_REG)
                            continue;
                        int sr = -1;
                        for (int r = 0; r < c->nregs; r++)
                            if (strcmp(c->regs[r].name,
                                       in->src[s].reg) == 0) {
                                sr = r;
                                break;
                            }
                        if (sr >= 0) {
                            uint64_t sm = c->host_home_mask[sr] & ~HM_SCALAR;
                            nm |= sm;
                            if (sm & (HM_SHARED | HM_GLVAR | HM_LOCAL)) {
                                saw_shared = 1;
                                nm_shared |= sm;
                            } else if (sm && (sm & ~0xffffffffull) == 0) {
                                saw_binding = 1;
                            }
                            c->host_paddr_mask[ri] |=
                                c->host_paddr_mask[sr];
                            // A derived pointer keeps the source's
                            // shared/global/local variable home (aux = var index).
                            if (((c->host_home_mask[sr] & HM_SHARED) ||
                                 (c->host_home_mask[sr] & HM_GLVAR) ||
                                 (c->host_home_mask[sr] & HM_LOCAL)) &&
                                c->host_home_aux[sr] >= 0) {
                                if (c->host_home_aux[ri] < 0)
                                    c->host_home_aux[ri] =
                                        c->host_home_aux[sr];
                                else if (c->host_home_aux[ri] !=
                                         c->host_home_aux[sr])
                                    c->host_home_aux[ri] = -1;
                            }
                        }
                    }
                    // Lifted 32-bit addressing stores a scalar offset
                    // in a pointer-typed word and adds it to a shared
                    // base.  The binding is an offset here, not an
                    // alternative home; keep the shared home so the
                    // CFG machine does not turn the base into a dynamic
                    // shared/host-state pointer.
                    if (in->op == POP_ADD && saw_shared && saw_binding)
                        nm = nm_shared;
                    if (nm == 0)
                        nm = HM_SCALAR;
                    }
                } else if (in->op == POP_CVT && in->ty == PTY_U64 &&
                           in->ty2 == PTY_U32 && in->nsrc == 1 &&
                           in->src[0].kind == OPD_REG && in->dst[0]) {
                    // Zero-extension preserves a single pointer home
                    // (lifted 32-bit addressing: low word + offset is
                    // widened for the global access). Propagate the
                    // source's non-scalar homes like mov/add do.
                    int sr = -1;
                    for (int r = 0; r < c->nregs; r++)
                        if (strcmp(c->regs[r].name,
                                   in->src[0].reg) == 0) {
                            sr = r;
                            break;
                        }
                    if (sr >= 0) {
                        // Preserve the scalar phase too: a widened
                        // pointer word that was reused as a numeric
                        // offset is a dynamic scalar/pointer register,
                        // not a pure binding token.
                        nm |= c->host_home_mask[sr];
                        c->host_paddr_mask[ri] |=
                            c->host_paddr_mask[sr];
                    }
                    if (nm == 0)
                        nm = HM_SCALAR;
                } else {
                    nm = HM_SCALAR;
                }
                uint64_t merged = c->host_home_mask[ri] | nm;
                if (merged != c->host_home_mask[ri]) {
                    c->host_home_mask[ri] = merged;
                    changed = 1;
                }
                if (mask_popcount(nm) == 1 &&
                    (nm & ~(HM_SCALAR | HM_SHARED | HM_GLVAR | HM_PADDR | HM_LOCAL))) {
                    if (c->host_home_aux[ri] < 0)
                        c->host_home_aux[ri] = mask_bit_index(nm);
                } else if (nm == HM_SHARED || nm == HM_GLVAR ||
                           nm == HM_PADDR || nm == HM_LOCAL) {
                    if (na >= 0) {
                        if (c->host_home_aux[ri] < 0)
                            c->host_home_aux[ri] = na;
                        else if (c->host_home_aux[ri] != na)
                            c->host_home_aux[ri] = -1;
                    }
                }
            }
        }
        if (!changed)
            break;
    }
    for (int i = 0; i < c->nregs; i++) {
        uint64_t m = c->host_home_mask[i];
        if (m == 0) {
            c->host_home[i] = HOME_UNINIT;
            c->host_home_aux[i] = -1;
            continue;
        }
        int n = mask_popcount(m);
        if (getenv("ACUDA_DEBUG_HOMES") && (m & ~(uint64_t)HM_SCALAR)) {
            fprintf(stderr, "ACuda host-home: reg=%s mask=0x%llx\n",
                    c->regs[i].name, (unsigned long long)m);
        }
        // A 32-bit shared-offset register is used as a scalar while its
        // byte offset is computed and as a pointer at the access.  It
        // has a single pointer home; the scalar bit is a temporary, not
        // a conflicting home.
        {
            uint64_t ptr_only = m & ~(uint64_t)HM_SCALAR;
            // Only offset-based homes (shared/global/local/parameter)
            // have a meaningful scalar phase.  A plain binding word
            // reused as a value is a genuine dynamic scalar/pointer
            // register, and misclassifying it leaves the state machine
            // with a pointer home but a scalar value at save time.
            if (ptr_only && (ptr_only & (ptr_only - 1)) == 0 &&
                (ptr_only & (HM_SHARED | HM_GLVAR | HM_LOCAL))) {
                m = ptr_only;
                n = 1;
            }
        }
        if (m == HM_SCALAR) {
            c->host_home[i] = HOME_SCALAR;
            c->host_home_aux[i] = -1;
        } else if (n == 1) {
            if (m & HM_SHARED) {
                c->host_home[i] = HOME_SHARED;
            } else if (m & HM_GLVAR) {
                c->host_home[i] = HOME_GLOBAL;
            } else if (m & HM_PADDR) {
                c->host_home[i] = HOME_PADDR;
            } else if (m & HM_BDA) {
                c->host_home[i] = HOME_BDA;
            } else if (m & HM_LOCAL) {
                c->host_home[i] = HOME_LOCAL;
            } else {
                c->host_home[i] = HOME_BINDING;
                c->host_home_aux[i] = mask_bit_index(m);
            }
        } else {
            // Several homes. Global bindings and shared storage can be
            // resolved at runtime; scalar/global-image/parameter mixes
            // cannot.
            uint64_t special = m & HM_SCALAR;
            uint64_t pointer = m & ~HM_SCALAR;
            uint64_t plain_bindings = pointer &
                ~(HM_SHARED | HM_GLVAR | HM_PADDR | HM_BDA | HM_LOCAL);
            if ((!special && pointer != 0) ||
                (special && (plain_bindings != 0 ||
                             (pointer & HM_BDA) != 0)) ||
                (plain_bindings != 0 &&
                 (pointer & (HM_SHARED | HM_GLVAR | HM_PADDR |
                             HM_BDA | HM_LOCAL)) == 0)) {
                // Plain binding homes (optionally mixed with scalar
                // value phases from register reuse, e.g. lifted 32-bit
                // addressing): resolve at runtime with a tag; scalar
                // phases carry a scalar tag (see sm_update_tag).
                c->host_home[i] = HOME_CONFLICT;
                c->host_dynamic[i] = 1;
            } else {
                c->host_home[i] = HOME_CONFLICT;
                c->host_dynamic[i] = 0;
            }
        }
        if (c->host_home[i] == HOME_SHARED &&
            c->host_home_aux[i] < 0) {
            // The same register points at different shared variables on
            // different paths; offsets are canonicalized to the shared
            // area, so a runtime shared tag is enough.
            c->host_home[i] = HOME_CONFLICT;
            c->host_dynamic[i] = 1;
        }
        if (c->host_home[i] == HOME_GLOBAL &&
            c->host_home_aux[i] < 0) {
            // Different .global image variables: canonical absolute
            // offsets plus a runtime global-image tag.
            c->host_home[i] = HOME_CONFLICT;
            c->host_dynamic[i] = 1;
        }
        if (c->host_home[i] == HOME_PADDR &&
            (c->host_home_aux[i] < 0 ||
             mask_popcount(c->host_paddr_mask[i]) > 1)) {
            c->host_home[i] = HOME_CONFLICT;
            c->host_dynamic[i] = 1;
        }
    }
    // Reject a non-dynamic conflict at an address use.
    for (size_t bi = 0; bi < fn->nblocks; bi++) {
        for (size_t ii = 0; ii < fn->blocks[bi].count; ii++) {
            const ptx_instr_t *in = &fn->blocks[bi].instrs[ii];
            if ((in->op == POP_LD || in->op == POP_ST ||
                 in->op == POP_ATOM || in->op == POP_ATOMCAS) &&
                in->src[0].kind == OPD_ADDR &&
                in->src[0].addr_reg[0]) {
                for (int r = 0; r < c->nregs; r++)
                    if (strcmp(c->regs[r].name,
                               in->src[0].addr_reg) == 0 &&
                        c->host_home[r] == HOME_CONFLICT &&
                        !c->host_dynamic[r]) {
                        snprintf(err, errlen,
                                 "line %d: pointer register has conflicting homes across control flow",
                                 in->line);
                        return -1;
                    }
            }
        }
    }
    return 0;
}
