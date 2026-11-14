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

uint32_t addr_ptr(ctx_t *c, spv_blob_t *b, int binding, uint32_t off) {
    if (binding == -3)
        return bda_convert(c, b, c->t_ptr_phys_u32, off);
    uint32_t word = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, word, off, const_u64(c, 2));
    uint32_t zero = const_u32(c, 0);
    uint32_t ptr = nid(c);
    put(c, b, (6u << 16) | S_AccessChain);
    put(c, b, c->t_ptr_u_ssbo);
    put(c, b, ptr);
    put(c, b, c->ssbo_vars[binding]);
    put(c, b, zero);
    put(c, b, word);
    return ptr;
}

// .global-image twins of the shared accessors: each .global variable
// is a per-thread Private u32-word image array initialized from the
// declaration bytes (reads serve the init data; dynamic offsets index
// it at runtime like any array).
uint32_t gl_addr_ptr(ctx_t *c, spv_blob_t *b, int var, uint32_t off) {
    if (c->host_machine) {
        cparam_t *cp = find_cparam(c, "__acuda_host_state");
        if (!cp || cp->binding < 0)
            return 0;
        // Base is a constant state offset; `off` is a runtime byte
        // offset into the global image (already computed before this).
        uint32_t base = host_global_off_bytes(c, b, var, 0);
        uint32_t abs = nid(c);
        emit_arith(c, b, S_IAdd, c->t_u64, abs, base, off);
        return addr_ptr(c, b, cp->binding, abs);
    }
    uint32_t word = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, word, off, const_u64(c, 2));
    uint32_t ptr = nid(c);
    put(c, b, (5u << 16) | S_AccessChain);
    put(c, b, c->t_ptr_priv_u32);
    put(c, b, ptr);
    put(c, b, c->gl_vars[var]);
    put(c, b, word);
    return ptr;
}

uint32_t gl_load_u64(ctx_t *c, spv_blob_t *b, int var, uint32_t off) {
    if (c->host_machine) {
        cparam_t *cp = find_cparam(c, "__acuda_host_state");
        if (!cp || cp->binding < 0)
            return 0;
        uint32_t base = host_global_off_bytes(c, b, var, 0);
        uint32_t abs = nid(c);
        emit_arith(c, b, S_IAdd, c->t_u64, abs, base, off);
        return load_u64(c, b, cp->binding, abs);
    }
    uint32_t word = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, word, off, const_u64(c, 2));
    uint32_t p0 = nid(c), p1 = nid(c);
    put(c, b, (5u << 16) | S_AccessChain);
    put(c, b, c->t_ptr_priv_u32);
    put(c, b, p0);
    put(c, b, c->gl_vars[var]);
    put(c, b, word);
    uint32_t word1 = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, word1, word, const_u64(c, 1));
    put(c, b, (5u << 16) | S_AccessChain);
    put(c, b, c->t_ptr_priv_u32);
    put(c, b, p1);
    put(c, b, c->gl_vars[var]);
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

uint32_t const_bank_ptr(ctx_t *c, spv_blob_t *b, uint32_t off) {
    uint32_t word = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, word, off, const_u64(c, 2));
    /* Clamp the word index into the bank. Real CUDA tolerates an
       out-of-range dynamic ld.const (undefined result, no fault),
       but an OOB Private access faults and kills the dispatch
       (e.g. FluidMark density reads const params with a data-driven
       index that can exceed the declared bank). Saturate instead. */
    size_t bank_bytes = c->const_bank_size ? c->const_bank_size : 65536;
    if (bank_bytes < 65536)
        bank_bytes = 65536;
    uint64_t maxw = bank_bytes / 4 - 1;
    uint32_t over = nid(c);
    put(c, b, (5u << 16) | S_UGreater);
    put(c, b, c->t_bool);
    put(c, b, over);
    put(c, b, word);
    put(c, b, const_u64(c, maxw));
    uint32_t clamped = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, c->t_u64);
    put(c, b, clamped);
    put(c, b, over);
    put(c, b, const_u64(c, maxw));
    put(c, b, word);
    uint32_t ptr = nid(c);
    put(c, b, (5u << 16) | S_AccessChain);
    put(c, b, c->t_ptr_priv_u32);
    put(c, b, ptr);
    put(c, b, c->const_bank_var);
    put(c, b, clamped);
    return ptr;
}

uint32_t const_bank_load_u32(ctx_t *c, spv_blob_t *b, uint32_t off) {
    uint32_t ptr = const_bank_ptr(c, b, off);
    uint32_t id = nid(c);
    emit_load(c, b, c->t_u32, id, ptr);
    return id;
}

uint32_t const_bank_load_u64(ctx_t *c, spv_blob_t *b, uint32_t off) {
    uint32_t lo = const_bank_load_u32(c, b, off);
    uint32_t plus4 = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, plus4, off, const_u64(c, 4));
    uint32_t hi = const_bank_load_u32(c, b, plus4);
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

uint32_t const_bank_load_narrow(ctx_t *c, spv_blob_t *b, uint32_t off,
                                int width) {
    uint32_t word_off = nid(c);
    emit_arith(c, b, S_And, c->t_u64, word_off, off, const_u64(c, ~(uint64_t)3));
    uint32_t word = const_bank_load_u32(c, b, word_off);
    uint32_t lane = nid(c);
    emit_arith(c, b, S_And, c->t_u64, lane, off, const_u64(c, 3));
    uint32_t sh = nid(c);
    emit_arith(c, b, S_Shl, c->t_u64, sh, lane, const_u64(c, 3));
    uint32_t sh32 = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u32);
    put(c, b, sh32);
    put(c, b, sh);
    uint32_t value = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, value, word, sh32);
    if (width < 32) {
        uint32_t mask = (width == 8) ? 0xffu : 0xffffu;
        uint32_t masked = nid(c);
        emit_arith(c, b, S_And, c->t_u32, masked, value, const_u32(c, mask));
        return masked;
    }
    return value;
}

// Workgroup-shared twins of the SSBO accessors: each .shared variable
// is its own u32-word array in Workgroup storage (no struct wrapper,
// so the chain takes just the word index).
uint32_t sh_host_base(ctx_t *c, spv_blob_t *b, int var,
                             uint32_t off) {
    cparam_t *cp = find_cparam(c, "__acuda_host_state");
    if (!cp || cp->binding < 0)
        return 0;
    uint32_t base = host_shared_off_bytes(c, b, var, 0);
    uint32_t sum = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, sum, base, off);
    return sum;
}

int resolve_addr(ctx_t *c, spv_blob_t *b, const ptx_opd_t *op,
                int *binding, uint32_t *off, int *shvar, int *glvar,
                int line) {
    return resolve_addr_ex(c, b, op, binding, off, shvar, glvar, NULL, line);
}

// Shared address with a byte offset split out of the 64-bit pointer.
// The offset is added to a 32-bit word index; ACO folds the constant
// into the LDS instruction offset.
uint32_t sh_addr_ptr_imm(ctx_t *c, spv_blob_t *b, int var, uint32_t off,
                         int64_t imm) {
    if (!c->host_machine && c->shvars[var].elem == 8) {
        fail(c, "32-bit access to a .u64 shared variable unsupported");
        return 0;
    }
    if (c->host_machine) {
        cparam_t *cp = find_cparam(c, "__acuda_host_state");
        if (!cp || cp->binding < 0)
            return 0;
        uint32_t byteoff = sh_host_base(c, b, var, off);
        if (imm) {
            uint32_t t = nid(c);
            emit_arith(c, b, S_IAdd, c->t_u64, t, byteoff,
                       const_u64(c, (uint64_t)imm));
            byteoff = t;
        }
        return addr_ptr(c, b, cp->binding, byteoff);
    }
    if (c->sh_coop_typed) {
        if ((imm & 1) != 0) {
            fail(c, "unaligned shared 16-bit access");
            return 0;
        }
        uint32_t by = off;
        if (imm != 0) {
            by = nid(c);
            emit_arith(c, b, S_IAdd, c->t_u64, by, off,
                       const_u64(c, (uint64_t)imm));
        }
        uint32_t idx = nid(c);
        emit_arith(c, b, S_ShrL, c->t_u64, idx, by, const_u64(c, 1));
        uint32_t ptr = nid(c);
        put(c, b, (5u << 16) | S_AccessChain);
        put(c, b, c->t_ptr_w_coop);
        put(c, b, ptr);
        put(c, b, c->sh_vars[var]);
        put(c, b, idx);
        return ptr;
    }
    uint32_t w32 = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u32);
    put(c, b, w32);
    put(c, b, off);
    uint32_t word = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, word, w32, const_u32(c, 2));
    if (imm != 0) {
        if ((imm & 3) != 0) {
            fail(c, "unaligned shared <32-bit access");
            return 0;
        }
        uint32_t idx = nid(c);
        emit_arith(c, b, S_IAdd, c->t_u32, idx, word,
                   const_u32(c, (uint32_t)(imm >> 2)));
        word = idx;
    }
    uint32_t ptr = nid(c);
    put(c, b, (5u << 16) | S_AccessChain);
    put(c, b, c->shared_f32_view ? c->t_ptr_w_f32 : c->t_ptr_w_u32);
    put(c, b, ptr);
    put(c, b, c->sh_vars[var]);
    put(c, b, word);
    return ptr;
}

uint32_t sh_addr_ptr(ctx_t *c, spv_blob_t *b, int var, uint32_t off) {
    if (!c->host_machine && c->shvars[var].elem == 8) {
        fail(c, "32-bit access to a .u64 shared variable unsupported");
        return 0;
    }
    if (c->host_machine) {
        cparam_t *cp = find_cparam(c, "__acuda_host_state");
        if (!cp || cp->binding < 0)
            return 0;
        uint32_t byteoff = sh_host_base(c, b, var, off);
        return addr_ptr(c, b, cp->binding, byteoff);
    }
    if (c->sh_coop_typed) {
        uint32_t idx = nid(c);
        emit_arith(c, b, S_ShrL, c->t_u64, idx, off, const_u64(c, 1));
        uint32_t ptr = nid(c);
        put(c, b, (5u << 16) | S_AccessChain);
        put(c, b, c->t_ptr_w_coop);
        put(c, b, ptr);
        put(c, b, c->sh_vars[var]);
        put(c, b, idx);
        return ptr;
    }
    uint32_t word = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, word, off, const_u64(c, 2));
    uint32_t ptr = nid(c);
    put(c, b, (5u << 16) | S_AccessChain);
    put(c, b, c->shared_f32_view ? c->t_ptr_w_f32 : c->t_ptr_w_u32);
    put(c, b, ptr);
    put(c, b, c->sh_vars[var]);
    put(c, b, word);
    return ptr;
}

uint32_t sh_addr_ptr64(ctx_t *c, spv_blob_t *b, int var, uint32_t off) {
    if (c->host_machine) {
        cparam_t *cp = find_cparam(c, "__acuda_host_state");
        if (!cp || cp->binding < 0)
            return 0;
        uint32_t byteoff = sh_host_base(c, b, var, off);
        return ssbo64_ptr(c, b, cp->binding, byteoff);
    }
    uint32_t word = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, word, off, const_u64(c, 3));
    uint32_t ptr = nid(c);
    put(c, b, (5u << 16) | S_AccessChain);
    put(c, b, c->shared_f64_view ? c->t_ptr_w_f64 : c->t_ptr_w_u64);
    put(c, b, ptr);
    put(c, b, c->sh_vars[var]);
    put(c, b, word);
    return ptr;
}

uint32_t sh_word_ptr(ctx_t *c, spv_blob_t *b, int var, uint32_t word) {
    if (!c->host_machine && c->shvars[var].elem == 8) {
        fail(c, "32-bit access to a .u64 shared variable unsupported");
        return 0;
    }
    if (c->host_machine) {
        cparam_t *cp = find_cparam(c, "__acuda_host_state");
        if (!cp || cp->binding < 0)
            return 0;
        uint32_t w64 = nid(c);
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, c->t_u64);
        put(c, b, w64);
        put(c, b, word);
        uint32_t byteoff = nid(c);
        emit_arith(c, b, S_Shl, c->t_u64, byteoff, w64, const_u64(c, 2));
        uint32_t off = sh_host_base(c, b, var, byteoff);
        return addr_ptr(c, b, cp->binding, off);
    }
    uint32_t ptr = nid(c);
    put(c, b, (5u << 16) | S_AccessChain);
    put(c, b, c->t_ptr_w_u32);
    put(c, b, ptr);
    put(c, b, c->sh_vars[var]);
    put(c, b, word);
    return ptr;
}

// Pointer to a software-barrier counter in the internal Workgroup
// array (see emit_swbar_init and the POP_BAR lowering).
uint32_t swbar_ptr(ctx_t *c, spv_blob_t *b, uint32_t idx) {
    uint32_t i = const_u32(c, idx);
    uint32_t ptr = nid(c);
    put(c, b, (5u << 16) | S_AccessChain);
    put(c, b, c->t_ptr_w_u32);
    put(c, b, ptr);
    put(c, b, c->bar_var);
    put(c, b, i);
    return ptr;
}

// Entry prologue for counted/arrive barriers: invocation 0 zeroes the
// counter array, then a full-CTA barrier publishes it. Every invocation
// executes this, so the extra rendezvous is uniform and semantically
// neutral (PTX kernels start synchronized).
void emit_swbar_init(ctx_t *c, spv_blob_t *b) {
    // Always publish the counter array explicitly. Relying on
    // shaderZeroInitializeWorkgroupMemory caused counted barriers to
    // deadlock on some Windows drivers that advertise the feature.
    if (!c->uses_swbar)
        return;
    uint32_t tid = special_x(c, b, SPC_TID, 0);
    uint32_t is0 = nid(c);
    put(c, b, (5u << 16) | S_IEqual);
    put(c, b, c->t_bool);
    put(c, b, is0);
    put(c, b, tid);
    put(c, b, const_u32(c, 0));
    uint32_t init = nid(c), merge = nid(c);
    put(c, b, (3u << 16) | S_SelectionMerge);
    put(c, b, merge);
    put(c, b, 0);
    put(c, b, (4u << 16) | S_BranchCond);
    put(c, b, is0);
    put(c, b, init);
    put(c, b, merge);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, init);
    for (uint32_t i = 0; i < 17; i++) {
        uint32_t ptr = swbar_ptr(c, b, i);
        uint32_t zero = const_u32(c, 0);
        put(c, b, (3u << 16) | S_Store);
        put(c, b, ptr);
        put(c, b, zero);
    }
    emit_branch(c, b, merge);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, merge);
    uint32_t scope = const_u32(c, 2); // Workgroup
    uint32_t sem = const_u32(c, 0x108u);
    put(c, b, (4u << 16) | S_ControlBarrier);
    put(c, b, scope);
    put(c, b, scope);
    put(c, b, sem);
}

// .common prologue: zero each module-scope common variable (Private
// storage is undefined otherwise). Callers branch here from the entry.
void emit_common_init(ctx_t *c, spv_blob_t *b) {
    if (!c->uses_commons)
        return;
    for (int i = 0; i < c->nlocals; i++) {
        if (!c->locals[i].is_common)
            continue;
        size_t words = (c->locals[i].bytes + 3) / 4;
        if (words > 4096)
            words = 4096;
        for (size_t w = 0; w < words; w++) {
            uint32_t ptr = nid(c);
            put(c, b, (5u << 16) | S_AccessChain);
            put(c, b, c->t_ptr_priv_u32);
            put(c, b, ptr);
            put(c, b, c->lo_vars[i]);
            put(c, b, const_u32(c, (uint32_t)w));
            emit_store(c, b, ptr, const_u32(c, 0));
        }
    }
}

uint32_t sh_load_u64(ctx_t *c, spv_blob_t *b, int var, uint32_t off) {
    if (!c->host_machine && c->shvars[var].elem == 8) {
        uint32_t ptr = sh_addr_ptr64(c, b, var, off);
        if (c->shared_f64_view) {
            uint32_t f = nid(c);
            emit_load(c, b, c->t_f64, f, ptr);
            uint32_t v = nid(c);
            put(c, b, (4u << 16) | S_Bitcast);
            put(c, b, c->t_u64);
            put(c, b, v);
            put(c, b, f);
            return v;
        }
        uint32_t v = nid(c);
        emit_load(c, b, c->t_u64, v, ptr);
        return v;
    }
    if (c->host_machine) {
        cparam_t *cp = find_cparam(c, "__acuda_host_state");
        if (!cp || cp->binding < 0)
            return 0;
        uint32_t byteoff = sh_host_base(c, b, var, off);
        return load_u64(c, b, cp->binding, byteoff);
    }
    uint32_t word = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, word, off, const_u64(c, 2));
    uint32_t p0 = nid(c), p1 = nid(c);
    put(c, b, (5u << 16) | S_AccessChain);
    put(c, b, c->t_ptr_w_u32);
    put(c, b, p0);
    put(c, b, c->sh_vars[var]);
    put(c, b, word);
    uint32_t word1 = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, word1, word, const_u64(c, 1));
    put(c, b, (5u << 16) | S_AccessChain);
    put(c, b, c->t_ptr_w_u32);
    put(c, b, p1);
    put(c, b, c->sh_vars[var]);
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

void sh_store_u64(ctx_t *c, spv_blob_t *b, int var, uint32_t off,
                         uint32_t val) {
    if (!c->host_machine && c->shvars[var].elem == 8) {
        uint32_t ptr = sh_addr_ptr64(c, b, var, off);
        if (c->shared_f64_view) {
            uint32_t f = nid(c);
            put(c, b, (4u << 16) | S_Bitcast);
            put(c, b, c->t_f64);
            put(c, b, f);
            put(c, b, val);
            put(c, b, (3u << 16) | S_Store);
            put(c, b, ptr);
            put(c, b, f);
            return;
        }
        put(c, b, (3u << 16) | S_Store);
        put(c, b, ptr);
        put(c, b, val);
        return;
    }
    if (c->host_machine) {
        cparam_t *cp = find_cparam(c, "__acuda_host_state");
        if (!cp || cp->binding < 0)
            return;
        uint32_t byteoff = sh_host_base(c, b, var, off);
        store_u64(c, b, cp->binding, byteoff, val);
        return;
    }
    uint32_t word = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, word, off, const_u64(c, 2));
    uint32_t p0 = nid(c), p1 = nid(c);
    put(c, b, (5u << 16) | S_AccessChain);
    put(c, b, c->t_ptr_w_u32);
    put(c, b, p0);
    put(c, b, c->sh_vars[var]);
    put(c, b, word);
    uint32_t word1 = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, word1, word, const_u64(c, 1));
    put(c, b, (5u << 16) | S_AccessChain);
    put(c, b, c->t_ptr_w_u32);
    put(c, b, p1);
    put(c, b, c->sh_vars[var]);
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

// Private (.local) twins of the shared accessors: per-invocation u32
// arrays, same word/bit layout. Sub-word .local access stays a clean
// error (no corpus need); Private memory is per-thread, so stores
// need no atomic RMW.
uint32_t lo_addr_ptr(ctx_t *c, spv_blob_t *b, int var, uint32_t off) {
    uint32_t word = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, word, off, const_u64(c, 2));
    uint32_t ptr = nid(c);
    put(c, b, (5u << 16) | S_AccessChain);
    put(c, b, c->t_ptr_priv_u32);
    put(c, b, ptr);
    put(c, b, c->lo_vars[var]);
    put(c, b, word);
    return ptr;
}

uint32_t lo_load_u64(ctx_t *c, spv_blob_t *b, int var, uint32_t off) {
    uint32_t word = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, word, off, const_u64(c, 2));
    uint32_t p0 = nid(c), p1 = nid(c);
    put(c, b, (5u << 16) | S_AccessChain);
    put(c, b, c->t_ptr_priv_u32);
    put(c, b, p0);
    put(c, b, c->lo_vars[var]);
    put(c, b, word);
    uint32_t word1 = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, word1, word, const_u64(c, 1));
    put(c, b, (5u << 16) | S_AccessChain);
    put(c, b, c->t_ptr_priv_u32);
    put(c, b, p1);
    put(c, b, c->lo_vars[var]);
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

void lo_store_u64(ctx_t *c, spv_blob_t *b, int var, uint32_t off,
                         uint32_t val) {
    uint32_t word = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, word, off, const_u64(c, 2));
    uint32_t p0 = nid(c), p1 = nid(c);
    put(c, b, (5u << 16) | S_AccessChain);
    put(c, b, c->t_ptr_priv_u32);
    put(c, b, p0);
    put(c, b, c->lo_vars[var]);
    put(c, b, word);
    uint32_t word1 = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, word1, word, const_u64(c, 1));
    put(c, b, (5u << 16) | S_AccessChain);
    put(c, b, c->t_ptr_priv_u32);
    put(c, b, p1);
    put(c, b, c->lo_vars[var]);
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
// Sub-word memory access. Narrow values are zero-extended canonical
// (low bits valid); signedness is applied at use (setp/cvt narrow
// explicitly). Lane shift is (off & 3) * 8 for bytes, (off & 2) * 8
// for halfwords — i.e. ((off & (width == 8 ? 3 : 2)) << 3) uniformly.
uint32_t narrow_lane_shift(ctx_t *c, spv_blob_t *b, uint32_t off,
                                  int width) {
    uint32_t lane = nid(c);
    emit_arith(c, b, S_And, c->t_u64, lane, off,
               const_u64(c, width == 8 ? 3u : 2u));
    uint32_t lane32 = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u32);
    put(c, b, lane32);
    put(c, b, lane);
    uint32_t sh = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, sh, lane32, const_u32(c, 3));
    return sh;
}

uint32_t u8_ssbo_ptr(ctx_t *c, spv_blob_t *b, int binding,
                            uint32_t off);
uint32_t u16_ssbo_ptr(ctx_t *c, spv_blob_t *b, int binding,
                             uint32_t off);

uint32_t narrow_ld_global(ctx_t *c, spv_blob_t *b, int binding,
                                 uint32_t off, int width) {
    if (width == 32) {
        uint32_t ptr = binding == -3
                           ? bda_convert(c, b, c->t_ptr_phys_u32, off)
                           : addr_ptr(c, b, binding, off);
        uint32_t v = nid(c);
        emit_load(c, b, c->t_u32, v, ptr);
        return v;
    }
    if (binding == -3) {
        uint32_t sh;
        uint32_t ptr = bda_word_ptr(c, b, off, &sh);
        uint32_t w = nid(c);
        emit_load(c, b, c->t_u32, w, ptr);
        uint32_t sr = nid(c);
        emit_arith(c, b, S_ShrL, c->t_u32, sr, w, sh);
        uint32_t out = nid(c);
        emit_arith(c, b, S_And, c->t_u32, out, sr,
                   const_u32(c, width == 8 ? 0xffu : 0xffffu));
        return out;
    }
    if (width == 16 && c->use_ssbo16) {
        uint32_t ptr = binding == -3
                           ? bda_convert(c, b, phys_u16_ty(c), off)
                           : u16_ssbo_ptr(c, b, binding, off);
        uint32_t v = nid(c);
        emit_load(c, b, c->t_u16, v, ptr);
        uint32_t out = nid(c);
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, c->t_u32);
        put(c, b, out);
        put(c, b, v);
        return out;
    }
    if (width == 8 && c->use_ssbo8) {
        uint32_t ptr = u8_ssbo_ptr(c, b, binding, off);
        uint32_t v = nid(c);
        emit_load(c, b, c->t_u8, v, ptr);
        uint32_t out = nid(c);
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, c->t_u32);
        put(c, b, out);
        put(c, b, v);
        return out;
    }
    uint32_t word = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, word, off, const_u64(c, 2));
    uint32_t zero = const_u32(c, 0);
    uint32_t ptr = nid(c);
    put(c, b, (6u << 16) | S_AccessChain);
    put(c, b, c->t_ptr_u_ssbo);
    put(c, b, ptr);
    put(c, b, c->ssbo_vars[binding]);
    put(c, b, zero);
    put(c, b, word);
    uint32_t w = nid(c);
    emit_load(c, b, c->t_u32, w, ptr);
    uint32_t sh = narrow_lane_shift(c, b, off, width);
    uint32_t sr = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, sr, w, sh);
    uint32_t out = nid(c);
    emit_arith(c, b, S_And, c->t_u32, out, sr,
               const_u32(c, width == 8 ? 0xffu : 0xffffu));
    return out;
}

uint32_t narrow_ld_shared(ctx_t *c, spv_blob_t *b, int var,
                                 uint32_t off, int width) {
    if (c->sh_coop_typed && width == 16) {
        uint32_t ptr = sh_addr_ptr(c, b, var, off);
        uint32_t v = nid(c);
        emit_load(c, b, c->t_coop_elem, v, ptr);
        uint32_t u = nid(c);
        put(c, b, (4u << 16) | S_Bitcast);
        put(c, b, c->t_u16);
        put(c, b, u);
        put(c, b, v);
        uint32_t out = nid(c);
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, c->t_u32);
        put(c, b, out);
        put(c, b, u);
        return out;
    }
    uint32_t word = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, word, off, const_u64(c, 2));
    uint32_t ptr = nid(c);
    put(c, b, (5u << 16) | S_AccessChain);
    put(c, b, c->t_ptr_w_u32);
    put(c, b, ptr);
    put(c, b, c->sh_vars[var]);
    put(c, b, word);
    uint32_t w = nid(c);
    emit_load(c, b, c->t_u32, w, ptr);
    uint32_t sh = narrow_lane_shift(c, b, off, width);
    uint32_t sr = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, sr, w, sh);
    uint32_t out = nid(c);
    emit_arith(c, b, S_And, c->t_u32, out, sr,
               const_u32(c, width == 8 ? 0xffu : 0xffffu));
    return out;
}

uint32_t narrow_ld_glvar(ctx_t *c, spv_blob_t *b, int var,
                                uint32_t off, int width) {
    uint32_t word = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, word, off, const_u64(c, 2));
    uint32_t ptr = nid(c);
    put(c, b, (5u << 16) | S_AccessChain);
    put(c, b, c->t_ptr_priv_u32);
    put(c, b, ptr);
    put(c, b, c->gl_vars[var]);
    put(c, b, word);
    uint32_t w = nid(c);
    emit_load(c, b, c->t_u32, w, ptr);
    uint32_t sh = narrow_lane_shift(c, b, off, width);
    uint32_t sr = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, sr, w, sh);
    uint32_t out = nid(c);
    emit_arith(c, b, S_And, c->t_u32, out, sr,
               const_u32(c, width == 8 ? 0xffu : 0xffffu));
    return out;
}

// Sub-word stores via an atomic compare-and-swap loop (plain
// read-modify-write would lose concurrent same-word lanes; real
// kernels store adjacent bytes from neighboring threads). All values
// loop-local or invariant, so no phis are needed and the loop nests
// legally anywhere straight-line code can appear.

void cas_loop(ctx_t *c, spv_blob_t *b, uint32_t ptr, int is_shared,
                     uint32_t ins, uint32_t nmask) {
    uint32_t scope = const_u32(c, is_shared ? 2 : 1); // Workgroup/Device
    uint32_t sem_eq = const_u32(c, is_shared ? 0x108u : 0x48u);
    uint32_t sem_ne = const_u32(c, is_shared ? 0x102u : 0x42u);
    uint32_t header = nid(c), merge = nid(c), cont = nid(c);
    emit_branch(c, b, header);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, header);
    uint32_t old = nid(c);
    emit_load(c, b, c->t_u32, old, ptr);
    uint32_t keep = nid(c), nw = nid(c);
    emit_arith(c, b, S_And, c->t_u32, keep, old, nmask);
    emit_arith(c, b, S_Or, c->t_u32, nw, keep, ins);
    uint32_t prev = nid(c);
    put(c, b, (9u << 16) | S_AtomicCAS);
    put(c, b, c->t_u32);
    put(c, b, prev);
    put(c, b, ptr);
    put(c, b, scope);
    put(c, b, sem_eq);
    put(c, b, sem_ne);
    put(c, b, nw);
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
}

// Atomic inc-wrap loop: mem = old >= bound ? 0 : old + 1 (PTX wraps
// at the bound; SPIR-V's bare increment has no bound, hence the
// loop). Returns the winning old word; the header dominates the
// merge, so no phi is needed.
uint32_t atom_wrap_loop(ctx_t *c, spv_blob_t *b, uint32_t ptr,
                               int is_shared, uint32_t bound) {
    uint32_t scope = const_u32(c, is_shared ? 2 : 1); // Workgroup/Device
    uint32_t sem_eq = const_u32(c, is_shared ? 0x108u : 0x48u);
    uint32_t sem_ne = const_u32(c, is_shared ? 0x102u : 0x42u);
    uint32_t zero = const_u32(c, 0);
    uint32_t one = const_u32(c, 1);
    uint32_t header = nid(c), merge = nid(c), cont = nid(c);
    emit_branch(c, b, header);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, header);
    uint32_t old = nid(c);
    emit_load(c, b, c->t_u32, old, ptr);
    uint32_t is_ge = nid(c);
    put(c, b, (5u << 16) | S_UGreaterEqual);
    put(c, b, c->t_bool);
    put(c, b, is_ge);
    put(c, b, old);
    put(c, b, bound);
    uint32_t incr = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u32, incr, old, one);
    uint32_t nw = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, c->t_u32);
    put(c, b, nw);
    put(c, b, is_ge);
    put(c, b, zero);
    put(c, b, incr);
    uint32_t prev = nid(c);
    put(c, b, (9u << 16) | S_AtomicCAS);
    put(c, b, c->t_u32);
    put(c, b, prev);
    put(c, b, ptr);
    put(c, b, scope);
    put(c, b, sem_eq);
    put(c, b, sem_ne);
    put(c, b, nw);
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

// Atomic decrement (PTX atom.dec): old = mem; mem = (old == 0 ||
// old > bound) ? bound : old - 1.
uint32_t atom_dec_loop(ctx_t *c, spv_blob_t *b, uint32_t ptr,
                       int is_shared, uint32_t bound) {
    uint32_t scope = const_u32(c, is_shared ? 2 : 1);
    uint32_t sem_eq = const_u32(c, is_shared ? 0x108u : 0x48u);
    uint32_t sem_ne = const_u32(c, is_shared ? 0x102u : 0x42u);
    uint32_t zero = const_u32(c, 0);
    uint32_t one = const_u32(c, 1);
    uint32_t header = nid(c), merge = nid(c), cont = nid(c);
    emit_branch(c, b, header);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, header);
    uint32_t old = nid(c);
    emit_load(c, b, c->t_u32, old, ptr);
    uint32_t is_zero = nid(c), is_gt = nid(c), reset = nid(c);
    put(c, b, (5u << 16) | S_IEqual);
    put(c, b, c->t_bool);
    put(c, b, is_zero);
    put(c, b, old);
    put(c, b, zero);
    put(c, b, (5u << 16) | S_UGreater);
    put(c, b, c->t_bool);
    put(c, b, is_gt);
    put(c, b, old);
    put(c, b, bound);
    put(c, b, (5u << 16) | S_LogicalOr);
    put(c, b, c->t_bool);
    put(c, b, reset);
    put(c, b, is_zero);
    put(c, b, is_gt);
    uint32_t dec = nid(c);
    emit_arith(c, b, S_ISub, c->t_u32, dec, old, one);
    uint32_t nw = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, c->t_u32);
    put(c, b, nw);
    put(c, b, reset);
    put(c, b, bound);
    put(c, b, dec);
    uint32_t prev = nid(c);
    put(c, b, (9u << 16) | S_AtomicCAS);
    put(c, b, c->t_u32);
    put(c, b, prev);
    put(c, b, ptr);
    put(c, b, scope);
    put(c, b, sem_eq);
    put(c, b, sem_ne);
    put(c, b, nw);
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

// Atomic f32-add loop: the profile has no float-atomics capability,
// so the add runs inside a CAS loop over bitcast u32 words (same
// shape as the sub-word stores). Returns the winning old word BITS;
// the caller bitcasts to f32 for the destination.
uint32_t atom_fadd_loop(ctx_t *c, spv_blob_t *b, uint32_t ptr,
                               int is_shared, uint32_t add) {
    uint32_t scope = const_u32(c, is_shared ? 2 : 1); // Workgroup/Device
    uint32_t sem_eq = const_u32(c, is_shared ? 0x108u : 0x48u);
    uint32_t sem_ne = const_u32(c, is_shared ? 0x102u : 0x42u);
    uint32_t header = nid(c), merge = nid(c), cont = nid(c);
    emit_branch(c, b, header);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, header);
    uint32_t old_bits = nid(c);
    emit_load(c, b, c->t_u32, old_bits, ptr);
    uint32_t old_f = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_f32);
    put(c, b, old_f);
    put(c, b, old_bits);
    uint32_t new_f = nid(c);
    emit_arith(c, b, S_FAdd, c->t_f32, new_f, old_f, add);
    uint32_t new_bits = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_u32);
    put(c, b, new_bits);
    put(c, b, new_f);
    uint32_t prev = nid(c);
    put(c, b, (9u << 16) | S_AtomicCAS);
    put(c, b, c->t_u32);
    put(c, b, prev);
    put(c, b, ptr);
    put(c, b, scope);
    put(c, b, sem_eq);
    put(c, b, sem_ne);
    put(c, b, new_bits);
    put(c, b, old_bits);
    uint32_t ok = nid(c);
    put(c, b, (5u << 16) | S_IEqual);
    put(c, b, c->t_bool);
    put(c, b, ok);
    put(c, b, prev);
    put(c, b, old_bits);
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

// f64 atomic add via a u64 CAS loop (SPIR-V has no atomic fadd in the
// base profile, and the 64-bit alias view gives a CASable u64).
uint32_t atom_fadd64_loop(ctx_t *c, spv_blob_t *b, uint32_t ptr,
                                 uint32_t add, int is_shared) {
    uint32_t scope = const_u32(c, is_shared ? 2 : 1);
    uint32_t sem_eq = const_u32(c, is_shared ? 0x108u : 0x48u);
    uint32_t sem_ne = const_u32(c, is_shared ? 0x102u : 0x42u);
    uint32_t header = nid(c), merge = nid(c), cont = nid(c);
    emit_branch(c, b, header);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, header);
    uint32_t old_bits = nid(c);
    emit_load(c, b, c->t_u64, old_bits, ptr);
    uint32_t old_f = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_f64);
    put(c, b, old_f);
    put(c, b, old_bits);
    uint32_t new_f = nid(c);
    emit_arith(c, b, S_FAdd, c->t_f64, new_f, old_f, add);
    uint32_t new_bits = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_u64);
    put(c, b, new_bits);
    put(c, b, new_f);
    uint32_t prev = nid(c);
    put(c, b, (9u << 16) | S_AtomicCAS);
    put(c, b, c->t_u64);
    put(c, b, prev);
    put(c, b, ptr);
    put(c, b, scope);
    put(c, b, sem_eq);
    put(c, b, sem_ne);
    put(c, b, new_bits);
    put(c, b, old_bits);
    uint32_t ok = nid(c);
    put(c, b, (5u << 16) | S_IEqual);
    put(c, b, c->t_bool);
    put(c, b, ok);
    put(c, b, prev);
    put(c, b, old_bits);
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

// Pointer to a u64 element in an alias view of the same storage
// binding. The u32 word view used by the rest of the emitter is a
// RuntimeArray<uint>, which cannot be atomically accessed as u64; a
// second variable with the same DescriptorSet/Binding and a
// RuntimeArray<ulong> view gives the atomic op the right pointee type.
// Byte view of a storage binding (RuntimeArray<uchar>), used for true
// 8-bit loads/stores when storageBuffer8BitAccess is available. The
// same descriptor backs both views; the u32 word view remains for the
// rest of the emitter.
uint32_t u16_ssbo_ptr(ctx_t *c, spv_blob_t *b, int binding,
                             uint32_t off) {
    if (!c->t_u16) {
        c->t_u16 = nid(c);
        put(c, &c->types, (4u << 16) | S_TypeInt);
        put(c, &c->types, c->t_u16);
        put(c, &c->types, 16);
        put(c, &c->types, 0);
    }
    if (!c->ssbo16_vars[binding]) {
        uint32_t arr = nid(c);
        put(c, &c->types, (3u << 16) | S_TypeRuntimeArray);
        put(c, &c->types, arr);
        put(c, &c->types, c->t_u16);
        emit_decorate1(c, arr, D_ArrayStride, 2);
        uint32_t st = nid(c);
        put(c, &c->types, (3u << 16) | S_TypeStruct);
        put(c, &c->types, st);
        put(c, &c->types, arr);
        emit_decorate(c, st, D_Block);
        emit_member_decorate(c, st, 0, D_Offset, 0);
        uint32_t tp = nid(c);
        put(c, &c->types, (4u << 16) | S_TypePointer);
        put(c, &c->types, tp);
        put(c, &c->types, SC_StorageBuffer);
        put(c, &c->types, st);
        uint32_t var = nid(c);
        put(c, &c->types, (4u << 16) | S_Variable);
        put(c, &c->types, tp);
        put(c, &c->types, var);
        put(c, &c->types, SC_StorageBuffer);
        emit_decorate1(c, var, D_DescriptorSet, 0);
        emit_decorate1(c, var, D_Binding, (uint32_t)binding);
        uint32_t elem = nid(c);
        put(c, &c->types, (4u << 16) | S_TypePointer);
        put(c, &c->types, elem);
        put(c, &c->types, SC_StorageBuffer);
        put(c, &c->types, c->t_u16);
        c->ssbo16_vars[binding] = var;
        c->ssbo16_ptr_ty[binding] = elem;
    }
    // off is a byte offset; the array element is 16 bits.
    uint32_t idx = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, idx, off, const_u64(c, 1));
    uint32_t zero = const_u32(c, 0);
    uint32_t ptr = nid(c);
    put(c, b, (6u << 16) | S_AccessChain);
    put(c, b, c->ssbo16_ptr_ty[binding]);
    put(c, b, ptr);
    put(c, b, c->ssbo16_vars[binding]);
    put(c, b, zero);
    put(c, b, idx);
    return ptr;
}

uint32_t u8_ssbo_ptr(ctx_t *c, spv_blob_t *b, int binding,
                            uint32_t off) {
    if (!c->t_u8) {
        c->t_u8 = nid(c);
        put(c, &c->types, (4u << 16) | S_TypeInt);
        put(c, &c->types, c->t_u8);
        put(c, &c->types, 8);
        put(c, &c->types, 0);
    }
    if (!c->ssbo8_vars[binding]) {
        uint32_t arr = nid(c);
        put(c, &c->types, (3u << 16) | S_TypeRuntimeArray);
        put(c, &c->types, arr);
        put(c, &c->types, c->t_u8);
        emit_decorate1(c, arr, D_ArrayStride, 1);
        uint32_t st = nid(c);
        put(c, &c->types, (3u << 16) | S_TypeStruct);
        put(c, &c->types, st);
        put(c, &c->types, arr);
        emit_decorate(c, st, D_Block);
        emit_member_decorate(c, st, 0, D_Offset, 0);
        uint32_t tp = nid(c);
        put(c, &c->types, (4u << 16) | S_TypePointer);
        put(c, &c->types, tp);
        put(c, &c->types, SC_StorageBuffer);
        put(c, &c->types, st);
        uint32_t var = nid(c);
        put(c, &c->types, (4u << 16) | S_Variable);
        put(c, &c->types, tp);
        put(c, &c->types, var);
        put(c, &c->types, SC_StorageBuffer);
        emit_decorate1(c, var, D_DescriptorSet, 0);
        emit_decorate1(c, var, D_Binding, (uint32_t)binding);
        uint32_t elem = nid(c);
        put(c, &c->types, (4u << 16) | S_TypePointer);
        put(c, &c->types, elem);
        put(c, &c->types, SC_StorageBuffer);
        put(c, &c->types, c->t_u8);
        c->ssbo8_vars[binding] = var;
        c->ssbo8_ptr_ty[binding] = elem;
    }
    uint32_t zero = const_u32(c, 0);
    uint32_t ptr = nid(c);
    put(c, b, (6u << 16) | S_AccessChain);
    put(c, b, c->ssbo8_ptr_ty[binding]);
    put(c, b, ptr);
    put(c, b, c->ssbo8_vars[binding]);
    put(c, b, zero);
    put(c, b, off);
    return ptr;
}

uint32_t ssbo64_ptr(ctx_t *c, spv_blob_t *b, int binding, uint32_t off) {
    if (binding == -3)
        return bda_convert(c, b, c->t_ptr_phys_u64, off);
    if (binding < 0 || binding >= MAX_PARAMS) {
        fail(c, "64-bit atomic needs a storage-buffer binding");
        return 0;
    }
    if (!c->ssbo64_vars[binding]) {
        uint32_t arr = nid(c);
        put(c, &c->types, (3u << 16) | S_TypeRuntimeArray);
        put(c, &c->types, arr);
        put(c, &c->types, c->t_u64);
        emit_decorate1(c, arr, D_ArrayStride, 8);
        uint32_t st = nid(c);
        put(c, &c->types, (3u << 16) | S_TypeStruct);
        put(c, &c->types, st);
        put(c, &c->types, arr);
        emit_decorate(c, st, D_Block);
        emit_member_decorate(c, st, 0, D_Offset, 0);
        uint32_t tp = nid(c);
        put(c, &c->types, (4u << 16) | S_TypePointer);
        put(c, &c->types, tp);
        put(c, &c->types, SC_StorageBuffer);
        put(c, &c->types, st);
        uint32_t var = nid(c);
        put(c, &c->types, (4u << 16) | S_Variable);
        put(c, &c->types, tp);
        put(c, &c->types, var);
        put(c, &c->types, SC_StorageBuffer);
        emit_decorate1(c, var, D_DescriptorSet, 0);
        emit_decorate1(c, var, D_Binding, (uint32_t)binding);
        uint32_t tpu = nid(c);
        put(c, &c->types, (4u << 16) | S_TypePointer);
        put(c, &c->types, tpu);
        put(c, &c->types, SC_StorageBuffer);
        put(c, &c->types, c->t_u64);
        c->ssbo64_vars[binding] = var;
        c->ssbo64_ptr_ty[binding] = tpu;
    }
    uint32_t zero = const_u32(c, 0);
    uint32_t word = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, word, off, const_u64(c, 3));
    uint32_t ptr = nid(c);
    put(c, b, (6u << 16) | S_AccessChain);
    put(c, b, c->ssbo64_ptr_ty[binding]);
    put(c, b, ptr);
    put(c, b, c->ssbo64_vars[binding]);
    put(c, b, zero);
    put(c, b, word);
    return ptr;
}

void narrow_st_global(ctx_t *c, spv_blob_t *b, int binding,
                             uint32_t off, uint32_t val, int width) {
    if (binding == -3) {
        uint32_t sh;
        uint32_t ptr = bda_word_ptr(c, b, off, &sh);
        uint32_t maskv = const_u32(c, width == 8 ? 0xffu : 0xffffu);
        uint32_t bits = nid(c), ins = nid(c), mask = nid(c), nmask = nid(c);
        emit_arith(c, b, S_And, c->t_u32, bits, val, maskv);
        emit_arith(c, b, S_Shl, c->t_u32, ins, bits, sh);
        emit_arith(c, b, S_Shl, c->t_u32, mask, maskv, sh);
        put(c, b, (4u << 16) | S_Not);
        put(c, b, c->t_u32);
        put(c, b, nmask);
        put(c, b, mask);
        cas_loop(c, b, ptr, 0, ins, nmask);
        return;
    }
    if (width == 16 && c->use_ssbo16) {
        uint32_t ptr = binding == -3
                           ? bda_convert(c, b, phys_u16_ty(c), off)
                           : u16_ssbo_ptr(c, b, binding, off);
        uint32_t v = nid(c);
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, c->t_u16);
        put(c, b, v);
        put(c, b, val);
        emit_store(c, b, ptr, v);
        return;
    }
    if (width == 8 && c->use_ssbo8) {
        uint32_t ptr = u8_ssbo_ptr(c, b, binding, off);
        uint32_t v = nid(c);
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, c->t_u8);
        put(c, b, v);
        put(c, b, val);
        emit_store(c, b, ptr, v);
        return;
    }
    uint32_t word = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, word, off, const_u64(c, 2));
    uint32_t zero = const_u32(c, 0);
    uint32_t ptr = nid(c);
    put(c, b, (6u << 16) | S_AccessChain);
    put(c, b, c->t_ptr_u_ssbo);
    put(c, b, ptr);
    put(c, b, c->ssbo_vars[binding]);
    put(c, b, zero);
    put(c, b, word);
    uint32_t sh = narrow_lane_shift(c, b, off, width);
    uint32_t bits = nid(c);
    emit_arith(c, b, S_And, c->t_u32, bits, val,
               const_u32(c, width == 8 ? 0xffu : 0xffffu));
    uint32_t ins = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, ins, bits, sh);
    uint32_t mask = nid(c), nmask = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, mask,
               const_u32(c, width == 8 ? 0xffu : 0xffffu), sh);
    put(c, b, (4u << 16) | S_Not);
    put(c, b, c->t_u32);
    put(c, b, nmask);
    put(c, b, mask);
    cas_loop(c, b, ptr, 0, ins, nmask);
}

void narrow_st_shared(ctx_t *c, spv_blob_t *b, int var,
                             uint32_t off, uint32_t val, int width) {
    if (c->sh_coop_typed && width == 16) {
        uint32_t ptr = sh_addr_ptr(c, b, var, off);
        uint32_t u = nid(c);
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, c->t_u16);
        put(c, b, u);
        put(c, b, val);
        uint32_t v = nid(c);
        put(c, b, (4u << 16) | S_Bitcast);
        put(c, b, c->t_coop_elem);
        put(c, b, v);
        put(c, b, u);
        emit_store(c, b, ptr, v);
        return;
    }
    if (width == 32) {
        if (c->sh_coop_typed) {
            uint32_t lo = nid(c), hi = nid(c), off2 = nid(c);
            emit_arith(c, b, S_And, c->t_u32, lo, val,
                       const_u32(c, 0xffffu));
            emit_arith(c, b, S_ShrL, c->t_u32, hi, val, const_u32(c, 16));
            emit_arith(c, b, S_IAdd, c->t_u64, off2, off,
                       const_u64(c, 2));
            narrow_st_shared(c, b, var, off, lo, 16);
            narrow_st_shared(c, b, var, off2, hi, 16);
            return;
        }
        uint32_t ptr = sh_addr_ptr(c, b, var, off);
        emit_store(c, b, ptr, val);
        return;
    }
    uint32_t word = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, word, off, const_u64(c, 2));
    uint32_t ptr = nid(c);
    put(c, b, (5u << 16) | S_AccessChain);
    put(c, b, c->t_ptr_w_u32);
    put(c, b, ptr);
    put(c, b, c->sh_vars[var]);
    put(c, b, word);
    uint32_t sh = narrow_lane_shift(c, b, off, width);
    uint32_t bits = nid(c);
    emit_arith(c, b, S_And, c->t_u32, bits, val,
               const_u32(c, width == 8 ? 0xffu : 0xffffu));
    uint32_t ins = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, ins, bits, sh);
    uint32_t mask = nid(c), nmask = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, mask,
               const_u32(c, width == 8 ? 0xffu : 0xffffu), sh);
    put(c, b, (4u << 16) | S_Not);
    put(c, b, c->t_u32);
    put(c, b, nmask);
    put(c, b, mask);
    cas_loop(c, b, ptr, 1, ins, nmask);
}

