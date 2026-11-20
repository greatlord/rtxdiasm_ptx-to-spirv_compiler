#include "instructions/memory/memory.h"

#include <string.h>

static void emit_ldst_mem_barrier(ctx_t *c, spv_blob_t *b, int mem_sem, int mem_scope, int is_release_before_store) {
    int sem = 0;
    if (mem_sem == 2) sem = 0x2;       /* acquire */
    else if (mem_sem == 3) sem = 0x4;  /* release */
    else if (mem_sem == 4) sem = 0x8;  /* acq_rel/sc/volatile/mmio */
    else return;
    /* For stores, emit release semantics before the store. For loads, emit acquire semantics after. */
    if (is_release_before_store) {
        if (!(sem & 0x4)) return;
        sem = 0x4;
    } else {
        if (!(sem & 0x2) && mem_sem != 4) return;
        if (mem_sem == 4) sem = 0x2; /* acquire side after load */
        else sem = 0x2;
    }
    uint32_t scope = const_u32(c, (mem_scope == 1 || mem_scope == 2) ? 2u : 1u);
    uint32_t s = const_u32(c, (uint32_t)sem | 0x40u | 0x100u);
    put(c, b, (3u << 16) | S_MemoryBarrier);
    put(c, b, scope);
    put(c, b, s);
}

static void emit_ld_op_impl(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);

void emit_ld_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    emit_ld_op_impl(c, b, in);
    /* Acquire barrier after acquire/acq_rel loads is intentionally omitted:
     * it can create driver-hostile SPIR-V patterns in tight loops; the
     * compiler also lowers fences and atomics with strong semantics. */
    (void)in;
}

static void emit_ld_op_impl(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    uint32_t id;
        if (getenv("ACUDA_DEBUG_SYMBOLS") && in->src[0].kind == OPD_ADDR &&
            in->src[0].addr_reg[0] && strcmp(in->src[0].addr_reg, "%rd3") == 0 &&
            in->src[0].addr_imm == 12) {
            reg_t *_ar = find_reg(c, in->src[0].addr_reg);
            fprintf(stderr, "ACuda: ld dbg dst=%s op=%d sp=%d reg=%s imm=%lld ty=%d ar_def=%d ar_ptr=%d ar_paddr=%d ar_bind=%d ar_sh=%d ar_gl=%d dyn=%d\n",
                    in->dst, (int)in->op, (int)in->space, in->src[0].addr_reg,
                    (long long)in->src[0].addr_imm, (int)in->ty,
                    _ar ? _ar->defined : -1, _ar ? _ar->is_ptr : -1,
                    _ar ? _ar->paddr : -99, _ar ? _ar->binding : -99,
                    _ar ? _ar->shvar : -99, _ar ? _ar->glvar : -99,
                    _ar ? _ar->dyn_home : -99);
        }
        if (in->space == PSP_PARAM) {
            if (in->src[0].kind != OPD_ADDR) {
                failf(c, line, "ld.param needs [param_name]");
                return;
            }
            const char *pname = in->src[0].param;
            if (in->src[0].addr_reg[0]) {
                // ld.param [%rd] where %rd holds the symbolic address
                // of a parameter (mov.u64 %rd, param). Reads the same
                // value the direct [param] load would.
                reg_t *pr = find_reg(c, in->src[0].addr_reg);
                if (!pr || !pr->defined || !pr->is_ptr ||
                    pr->paddr < 0) {
                    failf(c, line,
                          "ld.param address must be a parameter address");
                    return;
                }
                if (in->src[0].addr_imm != 0) {
                    char _dbg[320];
                    snprintf(_dbg, sizeof(_dbg),
                             "offset into a parameter address unsupported (op=%d sp=%d dst=%s reg=%s paddr=%d imm=%lld pname=%s ty=%d nfn=%d)",
                             (int)in->op, (int)in->space, in->dst, in->src[0].addr_reg, pr->paddr,
                             (long long)in->src[0].addr_imm,
                             (pr->paddr >= 0 && pr->paddr < c->nfnparams) ? c->fnparams[pr->paddr].name : "?",
                             (int)in->ty, c->nfnparams);
                    failf(c, line, _dbg);
                    return;
                }
                pname = c->fnparams[pr->paddr].name;
            }
            int64_t poff = in->src[0].addr_imm;
            char partname[300];
            if (!in->src[0].addr_reg[0]) {
                int words = 0;
                for (int q = 0; q < c->nfnparams; q++) {
                    if (c->fnparams[q].wide_words &&
                        strcmp(c->fnparams[q].name, pname) == 0) {
                        words = c->fnparams[q].wide_words;
                        break;
                    }
                }
                cparam_t *base = find_cparam(c, pname);
                if (words > 0 && base && base->kind == 1) {
                    // Pointer words are split into one binding each:
                    // p (word 0), p#1, p#2, ...
                    int part = (int)(poff / 8);
                    int rel = (int)(poff % 8);
                    if (part < 0 || part >= words || (rel != 0 && rel != 4)) {
                        failf(c, line,
                              "offset into a by-value pointer parameter is out of range");
                        return;
                    }
                    if (part > 0) {
                        if (snprintf(partname, sizeof(partname),
                                     "%s#%d", pname, part) >=
                            (int)sizeof(partname)) {
                            failf(c, line, "param name too long");
                            return;
                        }
                        pname = partname;
                    }
                    poff = rel;
                }
            }
            cparam_t *cp = find_cparam(c, pname);
            if (!cp) {
                // Declared-but-unused parameter (kind 3, no slot): the
                // loaded value is dead by construction (any address or
                // value use would have marked it), so define undef of
                // the load type. Undeclared names stay hard errors.
                int known = 0;
                for (int i = 0; i < c->nfnparams; i++) {
                    if (strcmp(c->fnparams[i].name, pname) == 0) {
                        known = 1;
                        break;
                    }
                }
                if (!known) {
                    failf(c, line, "unknown .param in ld.param");
                    return;
                }
                if (check_dst(c, in->dst, in->ty, line) != 0)
                    return;
                uint32_t u = nid(c);
                put(c, b, (3u << 16) | 1); // OpUndef (grammar: 1)
                put(c, b, ty_of(c, in->ty));
                put(c, b, u);
                def_val(c, in->dst, in->ty, u);
                return;
            }
            if (cp->kind == 1) {
                if (in->ty == PTY_U32 && poff == 0 && cp->binding >= 0 &&
                    (c->addr_size != 32 || c->use_bda)) {
                    // Low word of a pointer parameter: track it as a
                    // pointer home for later zero-extension +
                    // global access (lifted 32-bit addressing), while
                    // materializing the real low word when BDA pushes
                    // carry it (otherwise zero, like the scalar path).
                    // Value uses, null probes and address arithmetic
                    // all behave correctly.
                    if (check_dst(c, in->dst, in->ty, line) != 0)
                        return;
                    // 32-bit PTX with BDA: the device address is 64-bit.
                    // Rebuild the full base (low|high<<32) and track it
                    // as a raw BDA pointer (binding -3) so later add +
                    // global access uses the full address, not just the
                    // low word. The low word is still materialized for
                    // value uses (null probes). Matches the HM_BDA home
                    // the state machine assigns (see build_host_homes).
                    if (c->addr_size == 32 && c->use_bda &&
                        c->bda_word[cp->binding]) {
                        uint32_t lo = nid(c), hi = nid(c);
                        emit_push_load(c, b, &lo, PTY_U32,
                                       (uint32_t)c->bda_word[cp->binding],
                                       line);
                        emit_push_load(c, b, &hi, PTY_U32,
                                       (uint32_t)c->bda_word[cp->binding] + 1,
                                       line);
                        if (c->failed)
                            return;
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
                        uint32_t base = nid(c);
                        emit_arith(c, b, S_Or, c->t_u64, base, sh, lo64);
                        def_ptr(c, in->dst, -3, base);
                        reg_t *dr0 = find_reg(c, in->dst);
                        if (dr0) {
                            dr0->is_tex = 0;
                            dr0->cur = lo;
                            dr0->cur_ty = PTY_U32;
                        }
                        return;
                    }
                    uint32_t w = const_u32(c, 0);
                    if (c->use_bda && c->bda_word[cp->binding]) {
                        uint32_t word = nid(c);
                        emit_push_load(c, b, &word, PTY_U32,
                                       (uint32_t)c->bda_word[cp->binding],
                                       line);
                        if (c->failed)
                            return;
                        w = word;
                    }
                    def_ptr(c, in->dst, cp->binding, const_u64(c, 0));
                    reg_t *dr0 = find_reg(c, in->dst);
                    if (dr0) {
                        dr0->is_tex = cp->is_tex;
                        dr0->cur = w;
                        dr0->cur_ty = PTY_U32;
                    }
                    return;
                }
                if (c->use_bda && cp->binding >= 0 &&
                    c->bda_word[cp->binding]) {
                    // PhysicalStorageBuffer mode: the param slot holds
                    // the buffer's 64-bit device address.  Sub-word
                    // loads read the low/high address word.
                    if (in->ty != PTY_U64) {
                        if (poff != 0 && poff != 4) {
                            failf(c, line,
                                  "offset into a pointer parameter must be 0 or 4");
                            return;
                        }
                        if (check_dst(c, in->dst, in->ty, line) != 0)
                            return;
                        uint32_t word = nid(c);
                        emit_push_load(c, b, &word, PTY_U32,
                                       (uint32_t)c->bda_word[cp->binding] +
                                           (poff == 4 ? 1u : 0u),
                                       line);
                        if (in->ty == PTY_F32) {
                            uint32_t bits = word;
                            word = nid(c);
                            put(c, b, (4u << 16) | S_Bitcast);
                            put(c, b, c->t_f32);
                            put(c, b, word);
                            put(c, b, bits);
                        } else if (in->ty == PTY_U8 || in->ty == PTY_U16) {
                            uint32_t raw = word;
                            word = nid(c);
                            emit_arith(c, b, S_And, c->t_u32, word, raw,
                                       const_u32(c, in->ty == PTY_U8 ? 0xffu : 0xffffu));
                        }
                        def_val(c, in->dst, in->ty, word);
                        return;
                    }
                    if (check_dst(c, in->dst, PTY_U64, line) != 0)
                        return;
                    uint32_t lo = nid(c), hi = nid(c);
                    emit_push_load(c, b, &lo, PTY_U32,
                                   (uint32_t)c->bda_word[cp->binding], line);
                    emit_push_load(c, b, &hi, PTY_U32,
                                   (uint32_t)c->bda_word[cp->binding] + 1,
                                   line);
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
                    uint32_t addr = nid(c);
                    emit_arith(c, b, S_Or, c->t_u64, addr, sh, lo64);
                    def_ptr(c, in->dst, -3, addr);
                    return;
                }
                if (c->addr_size == 32 && in->ty == PTY_U32 && poff == 0) {
                    // 32-bit PTX: the low word of a pointer parameter is
                    // itself the pointer (not a numeric word). Track it as
                    // a pointer home so cvta/add/ld.global preserve it.
                    if (check_dst(c, in->dst, in->ty, line) != 0)
                        return;
                    def_ptr(c, in->dst, cp->binding, const_u64(c, 0));
                    reg_t *dr = find_reg(c, in->dst);
                    if (dr)
                        dr->is_tex = cp->is_tex;
                    return;
                }
                if (c->addr_size == 32 && in->ty == PTY_U32 && poff == 4 &&
                    cp->binding >= 0) {
                    // 32-bit PTX: the high word of a .b8[8] pointer
                    // parameter is a scalar (e.g. PxgGpuArray.count),
                    // stored in the binding buffer at byte offset 4.
                    if (check_dst(c, in->dst, in->ty, line) != 0)
                        return;
                    uint32_t ptr = addr_ptr(c, b, cp->binding,
                                            const_u64(c, 4));
                    uint32_t w = nid(c);
                    emit_load(c, b, c->t_u32, w, ptr);
                    def_val(c, in->dst, in->ty, w);
                    return;
                }
                if (in->ty == PTY_U32 || in->ty == PTY_U16 ||
                    in->ty == PTY_U8) {
                    // Reading the words of a pointer parameter. Logical
                    // SPIR-V cannot expose the numeric pointer value, so
                    // give the low word a defined zero and the high
                    // word a defined nonzero value (NVIDIA device
                    // pointers are addressable, and the common
                    // "[input+4] != 0" address-space probe relies on
                    // this). Only the two words of the 64-bit slot are
                    // meaningful.
                    int64_t off = poff;
                    if (off != 0 && off != 4) {
                        failf(c, line,
                              "offset into a pointer parameter must be 0 or 4");
                        return;
                    }
                    if (check_dst(c, in->dst, in->ty, line) != 0)
                        return;
                    uint32_t w = off == 4 ? 1u : 0u;
                    if (in->ty == PTY_U8)
                        w &= 0xffu;
                    else if (in->ty == PTY_U16)
                        w &= 0xffffu;
                    def_val(c, in->dst, in->ty, const_u32(c, w));
                    return;
                }
                if (in->ty == PTY_U64) {
                    if (check_dst(c, in->dst, PTY_U64, line) != 0)
                        return;
                    def_ptr(c, in->dst, cp->binding, const_u64(c, 0));
                    reg_t *dr = find_reg(c, in->dst);
                    if (dr)
                        dr->is_tex = cp->is_tex;
                    return;
                } else if (in->ty == PTY_F64) {
                    if (check_dst(c, in->dst, PTY_F64, line) != 0)
                        return;
                    reg_t *dr = find_reg(c, in->dst);
                    if (dr) {
                        dr->defined = 1;
                        dr->decl_ty = PTY_F64;
                        dr->is_ptr = 1;
                        dr->binding = cp->binding;
                        dr->off = const_u64(c, 0);
                        dr->shvar = -1;
                        dr->glvar = -1;
                        dr->cur = dr->off;
                    }
                    return;
                } else {
                    failf(c, line, "pointer param must load as .u64");
                    return;
                }
            }
            if (cp->kind == 6) {
                // Mixed 32-bit blob {ptr32, scalar32}: word0 is a device
                // pointer resolved through the binding (like a kind-1
                // pointer); word1 is a scalar push word.
                int64_t koff = in->src[0].addr_imm;
                if (koff == 0) {
                    if (in->ty != PTY_U32) {
                        failf(c, line,
                              "kind-6 word0 must load as .u32");
                        return;
                    }
                    if (check_dst(c, in->dst, in->ty, line) != 0)
                        return;
                    def_ptr(c, in->dst, cp->binding, const_u64(c, 0));
                    reg_t *dr6 = find_reg(c, in->dst);
                    if (dr6)
                        dr6->is_tex = cp->is_tex;
                    return;
                } else if (koff == 4) {
                    if (in->ty != PTY_U32) {
                        failf(c, line,
                              "kind-6 word1 must load as .u32");
                        return;
                    }
                    if (check_dst(c, in->dst, in->ty, line) != 0)
                        return;
                    uint32_t id6 = nid(c);
                    emit_push_load(c, b, &id6, PTY_U32, cp->push_word,
                                   line);
                    def_val(c, in->dst, in->ty, id6);
                    return;
                }
                failf(c, line,
                      "offset into a kind-6 blob must be 0 or 4");
                return;
            }

            if (cp->kind == 4) {
                // 16-byte by-value struct: four push words packed
                // from one host argument.  Loads address the struct
                // at byte offsets 0/4/8/12.
                int64_t woff = in->src[0].addr_imm;
                int words = cp->push_words / 2;
                int64_t limit = (int64_t)(in->ty == PTY_U64 ? 8 : 4);
                if (words < 2)
                    words = 2;
                if (woff < 0 || woff + limit > (int64_t)words * 8 ||
                    (woff & 3)) {
                    failf(c, line,
                          "offset into a by-value parameter is out of range");
                    return;
                }
                if (cp->binding >= 0) {
                    // Large by-value struct: the host uploaded the raw
                    // argument bytes into a descriptor-bound storage
                    // buffer. Address it by byte offset.
                    uint32_t byteoff = const_u64(c, (uint64_t)woff);
                    if (in->ty == PTY_U64 || in->ty == PTY_F64) {
                        if (check_dst(c, in->dst, in->ty, line) != 0)
                            return;
                        uint32_t u = load_u64(c, b, cp->binding, byteoff);
                        if (in->ty == PTY_F64) {
                            uint32_t f = nid(c);
                            put(c, b, (4u << 16) | S_Bitcast);
                            put(c, b, c->t_f64);
                            put(c, b, f);
                            put(c, b, u);
                            def_val(c, in->dst, PTY_F64, f);
                        } else {
                            def_val(c, in->dst, PTY_U64, u);
                        }
                        return;
                    }
                    if (check_dst(c, in->dst, in->ty, line) != 0)
                        return;
                    uint32_t ptr = addr_ptr(c, b, cp->binding, byteoff);
                    uint32_t word = nid(c);
                    emit_load(c, b, c->t_u32, word, ptr);
                    if (in->ty == PTY_F32) {
                        uint32_t bits = word;
                        word = nid(c);
                        put(c, b, (4u << 16) | S_Bitcast);
                        put(c, b, c->t_f32);
                        put(c, b, word);
                        put(c, b, bits);
                    } else if (in->ty == PTY_U8 || in->ty == PTY_U16) {
                        uint32_t raw = word;
                        word = nid(c);
                        emit_arith(c, b, S_And, c->t_u32, word, raw,
                                   const_u32(c, in->ty == PTY_U8 ? 0xffu : 0xffffu));
                    }
                    def_val(c, in->dst, in->ty, word);
                    return;
                }
                uint32_t base = (uint32_t)cp->push_word +
                                (uint32_t)(woff / 4);
                if (in->ty == PTY_U64) {
                    if (check_dst(c, in->dst, PTY_U64, line) != 0)
                        return;
                    uint32_t lo = nid(c), hi = nid(c);
                    emit_push_load(c, b, &lo, PTY_U32, base, line);
                    emit_push_load(c, b, &hi, PTY_U32, base + 1, line);
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
                    def_val(c, in->dst, PTY_U64, id);
                    return;
                }
                if (check_dst(c, in->dst, in->ty, line) != 0)
                    return;
                uint32_t word = nid(c);
                emit_push_load(c, b, &word, PTY_U32, base, line);
                if (in->ty == PTY_F32) {
                    uint32_t bits = word;
                    word = nid(c);
                    put(c, b, (4u << 16) | S_Bitcast);
                    put(c, b, c->t_f32);
                    put(c, b, word);
                    put(c, b, bits);
                } else if (in->ty == PTY_U8 || in->ty == PTY_U16) {
                    uint32_t raw = word;
                    word = nid(c);
                    emit_arith(c, b, S_And, c->t_u32, word, raw,
                               const_u32(c, in->ty == PTY_U8 ? 0xffu
                                                             : 0xffffu));
                }
                def_val(c, in->dst, in->ty, word);
                return;
            }
            if (cp->kind == 2) {
                // 64-bit scalar: two push words.  A .u64/.f64 load combines
                // them little-endian; a sub-word load reads one half
                // (offset 0 = low, 4 = high).  ptxas uses the latter
                // for a by-value struct whose fields are scalars.
                int64_t soff = in->src[0].addr_imm;
                if (soff != 0 && soff != 4) {
                    failf(c, line,
                          "offset into a 64-bit scalar parameter must be 0 or 4");
                    return;
                }
                if (in->ty != PTY_U64 && in->ty != PTY_F64) {
                    if (check_dst(c, in->dst, in->ty, line) != 0)
                        return;
                    uint32_t word = nid(c);
                    emit_push_load(c, b, &word, PTY_U32,
                                   (uint32_t)cp->push_word +
                                       (soff == 4 ? 1u : 0u),
                                   line);
                    if (in->ty == PTY_F32) {
                        uint32_t bits = word;
                        word = nid(c);
                        put(c, b, (4u << 16) | S_Bitcast);
                        put(c, b, c->t_f32);
                        put(c, b, word);
                        put(c, b, bits);
                    } else if (in->ty == PTY_U8 || in->ty == PTY_U16) {
                        uint32_t raw = word;
                        word = nid(c);
                        emit_arith(c, b, S_And, c->t_u32, word, raw,
                                   const_u32(c, in->ty == PTY_U8 ? 0xffu : 0xffffu));
                    }
                    def_val(c, in->dst, in->ty, word);
                    return;
                }
                if (soff != 0) {
                    failf(c, line,
                          "offset into a 64-bit scalar parameter must be 0");
                    return;
                }
                if (check_dst(c, in->dst, in->ty, line) != 0)
                    return;
                uint32_t lo = nid(c), hi = nid(c);
                emit_push_load(c, b, &lo, PTY_U32, cp->push_word, line);
                emit_push_load(c, b, &hi, PTY_U32, cp->push_word + 1, line);
                {
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
                    id = nid(c);
                    emit_arith(c, b, S_Or, c->t_u64, id, sh, lo64);
                    if (in->ty == PTY_F64) {
                        // Reinterpret the assembled 64-bit word as a double.
                        uint32_t f = nid(c);
                        put(c, b, (4u << 16) | S_Bitcast);
                        put(c, b, c->t_f64);
                        put(c, b, f);
                        put(c, b, id);
                        def_val(c, in->dst, PTY_F64, f);
                    } else {
                        def_val(c, in->dst, PTY_U64, id);
                    }
                }
                return;
            }
            reg_t *dr = find_reg(c, in->dst);
            if ((in->ty == PTY_U8 || in->ty == PTY_U16) && dr) {
                /* narrow integer param load: allow wider integer dest, sign/zero extend */
                int lw = in->ty == PTY_U8 ? 8 : 16;
                int dw = decl_int_width(dr->decl_ty);
                if (dw < lw) {
                    failf(c, line, "narrow ld.param destination too small");
                    return;
                }
                emit_push_load(c, b, &id, PTY_U32, cp->push_word, line);
                if (lw == 8) {
                    uint32_t raw = id; id = nid(c);
                    emit_arith(c, b, S_And, c->t_u32, id, raw, const_u32(c, 0xffu));
                } else {
                    uint32_t raw = id; id = nid(c);
                    emit_arith(c, b, S_And, c->t_u32, id, raw, const_u32(c, 0xffffu));
                }
                if (in->sign) {
                    uint32_t t1 = nid(c), t2 = nid(c);
                    emit_arith(c, b, S_Shl, c->t_u32, t1, id, const_u32(c, (uint32_t)(32 - lw)));
                    emit_arith(c, b, S_ShrA, c->t_u32, t2, t1, const_u32(c, (uint32_t)(32 - lw)));
                    id = t2;
                }
                if (dw == 64) {
                    uint32_t ext = nid(c);
                    put(c, b, (4u << 16) | (in->sign ? S_SConvert : S_UConvert));
                    put(c, b, c->t_u64); put(c, b, ext); put(c, b, id);
                    def_val(c, in->dst, dr->decl_ty, ext);
                } else {
                    def_val(c, in->dst, dr->decl_ty, id);
                }
                return;
            }
            if (check_dst(c, in->dst, in->ty, line) != 0)
                return;
            if (in->ty != cp->ty) {
                /* allow s/u/b width-compatible mismatch (same PTY from map_ty) */
                if (decl_int_width(in->ty) < 0 || decl_int_width(in->ty) != decl_int_width(cp->ty)) {
                    failf(c, line, "scalar param type mismatch on ld.param");
                    return;
                }
            }
            // Scalar param type must match: find declared type via push word.
            // (cparams stores only kind; ty check via load type vs word count.)
            emit_push_load(c, b, &id, in->ty, cp->push_word, line);
            def_val(c, in->dst, in->ty, id);
            return;
        }
        // Global / generic / shared / local load. Generic (stateless)
        // routes by lineage: shared-tracked addresses use Workgroup
        // access, kernel-pointer addresses use SSBO access.
        if (in->space == PSP_CONST) {
            // Register-indexed constant loads may target either the flattened
            // private const bank (offset tokens/scalars) or real device-backed
            // pointers (e.g. module symbol buffers supplied at runtime). For
            // pointer/BDA addresses, fall through to generic memory load paths.
            if (in->src[0].kind == OPD_ADDR && in->src[0].addr_reg[0]) {
                reg_t *ar = find_reg(c, in->src[0].addr_reg);
                int use_const_bank = 0;
                if (ar && ar->defined && !ar->is_ptr &&
                    (ar->decl_ty == PTY_U32 || ar->decl_ty == PTY_U64)) {
                    use_const_bank = 1;
                }
                if (use_const_bank) {
                    if (!c->has_const_bank) {
                        failf(c, line, "dynamic const bank was not allocated");
                        return;
                    }
                    uint32_t off = ar->cur;
                    if (ar->decl_ty == PTY_U32) {
                        uint32_t wide = nid(c);
                        put(c, b, (4u << 16) | S_UConvert);
                        put(c, b, c->t_u64);
                        put(c, b, wide);
                        put(c, b, off);
                        off = wide;
                    }
                    uint32_t raw = nid(c);
                    emit_arith(c, b, S_And, c->t_u64, raw, off,
                               const_u64(c, ~PTX_CONST_ADDR_BASE));
                    if (in->src[0].addr_imm != 0) {
                        uint32_t adjusted = nid(c);
                        emit_arith(c, b, S_IAdd, c->t_u64, adjusted, raw,
                                   const_u64(c,
                                       (uint64_t)in->src[0].addr_imm));
                        raw = adjusted;
                    }
                    if (in->ty == PTY_U8 || in->ty == PTY_U16 || in->ty == PTY_F16) {
                        reg_t *dr = find_reg(c, in->dst);
                        int lw = in->ty == PTY_U8 ? 8 : 16;
                        int dw = dr ? decl_int_width(dr->decl_ty) : -1;
                        if (dr && dr->decl_ty == PTY_F16 && lw == 16)
                            dw = 16;
                        if (!dr || dw < lw) {
                            failf(c, line, "narrow const destination must be an "
                                           "8/16/32/64-bit integer register");
                            return;
                        }
                        uint32_t v = const_bank_load_narrow(c, b, raw, lw);
                        def_val(c, in->dst, in->ty, v);
                    } else if (in->ty == PTY_U32 || in->ty == PTY_F32) {
                        if (check_dst(c, in->dst, in->ty, line) != 0)
                            return;
                        uint32_t v = const_bank_load_u32(c, b, raw);
                        if (in->ty == PTY_F32) {
                            uint32_t cv = nid(c);
                            put(c, b, (4u << 16) | S_Bitcast);
                            put(c, b, c->t_f32);
                            put(c, b, cv);
                            put(c, b, v);
                            v = cv;
                        }
                        def_val(c, in->dst, in->ty, v);
                    } else if (in->ty == PTY_U64 || in->ty == PTY_F64) {
                        if (check_dst(c, in->dst, in->ty, line) != 0)
                            return;
                        uint32_t v = const_bank_load_u64(c, b, raw);
                        if (in->ty == PTY_F64) {
                            uint32_t cv = nid(c);
                            put(c, b, (4u << 16) | S_Bitcast);
                            put(c, b, c->t_f64);
                            put(c, b, cv);
                            put(c, b, v);
                            v = cv;
                        }
                        def_val(c, in->dst, in->ty, v);
                    } else {
                        failf(c, line,
                              "dynamic const loads support .b8/.b16/.b32/.b64/.f16/.f32/.f64 only");
                    }
                    return;
                }
                /* Real pointer/BDA address in const space: treat as generic/global load. */
                goto ld_generic;
            }
            // Initialized module-scope constants materialize as
            // immediates (direct [name]/[name+imm] only; the offset
            // must land on a lane).
            if (in->src[0].kind != OPD_ADDR || in->src[0].addr_reg[0]) {
                failf(c, line, "const address must be [name] or [name+imm]");
                return;
            }
            int ci = find_const(c, in->src[0].param);
            if (ci < 0) {
                failf(c, line, "unknown .const variable");
                return;
            }
            size_t want;
            switch (in->ty) {
            case PTY_U8: want = 1; break;
            case PTY_U16: case PTY_F16: want = 2; break;
            case PTY_U32: case PTY_F32: want = 4; break;
            case PTY_U64: case PTY_F64: want = 8; break;
            default:
                failf(c, line, "const loads support .b8/.b16/.b32/.b64/"
                               ".f16/.f32/.f64 only");
                return;
            }
            size_t elem = c->consts[ci].elem;
            size_t readsz = want;
            /* .b8 (byte) const arrays may be accessed with any aligned
             * width (e.g. ld.const.f32 from a .b8[]), so validate/read
             * by access width, not declaration element width. */
            if (elem != 1 && want != elem) {
                failf(c, line, "const access width must match declaration");
                return;
            }
            int64_t off = in->src[0].addr_imm;
            if (off < 0 || (uint64_t)off % readsz != 0 ||
                (uint64_t)off + readsz > c->consts[ci].nbytes) {
                failf(c, line, "const offset out of range");
                return;
            }
            // Byte/half constants may target a same-or-wider integer
            // register, matching the generic narrow-load rule.
            if (in->ty == PTY_U8 || in->ty == PTY_U16) {
                reg_t *dr = find_reg(c, in->dst);
                int lw = in->ty == PTY_U8 ? 8 : 16;
                int dw = dr ? decl_int_width(dr->decl_ty) : -1;
                if (dr && dr->decl_ty == PTY_F16 && lw == 16)
                    dw = 16;
                if (!dr || dw < lw) {
                    failf(c, line, "narrow const destination must be an "
                                   "8/16/32/64-bit integer register");
                    return;
                }
            } else if (check_dst(c, in->dst, in->ty, line) != 0) {
                return;
            }
            uint64_t val = 0;
            for (size_t k = 0; k < readsz; k++)
                val |= (uint64_t)c->consts[ci].bytes[(size_t)off + k] << (8 * k);
            uint32_t res;
            if (in->ty == PTY_F32)
                res = const_f32(c, (uint32_t)val);
            else if (in->ty == PTY_F64)
                res = const_f64(c, val);
            else if (in->ty == PTY_U64)
                res = const_u64(c, val);
            else
                res = const_u32(c, (uint32_t)val);
            def_val(c, in->dst, in->ty, res);
            return;
        }
    ld_generic:
        if (in->ty != PTY_U32 && in->ty != PTY_F32 && in->ty != PTY_F64 &&
            in->ty != PTY_U64 && in->ty != PTY_U8 && in->ty != PTY_U16 &&
            in->ty != PTY_F16) {
            failf(c, line, "load only supports .u8/.u16/.u32/.f16/.f32/.f64/.u64");
            return;
        }
        // Narrow loads accept a same-or-wider integer destination
        // (ld.u8 into .u8/.u16/.u32/.u64 regs, zero-extended either
        // way); f16 loads need an exact .f16 destination. f32 into a
        // 64-bit holder skips the exact-type check (dedicated
        // zero-extending branch below).
        int wide_f32 = 0;
        if (in->ty == PTY_U8 || in->ty == PTY_U16) {
            reg_t *dr = find_reg(c, in->dst);
            int lw = in->ty == PTY_U8 ? 8 : 16;
            int dw = dr ? decl_int_width(dr->decl_ty) : -1;
            if (dr && dr->decl_ty == PTY_F16 && lw == 16)
                dw = 16; // halves are 16-bit patterns (ld.b16 into .f16)
            if (!dr || dw < lw) {
                failf(c, line, "narrow load destination must be an 8/16/32/64-bit integer register");
                return;
            }
        } else {
            reg_t *wdr = find_reg(c, in->dst);
            wide_f32 = in->ty == PTY_F32 && wdr && wdr->decl_ty == PTY_U64;
            if (!wide_f32 && check_dst(c, in->dst, in->ty, line) != 0)
                return;
        }
        // Direct .local variable use ([var]/[var+N], generic or .local
        // space): Private per-invocation storage. Other names fall
        // through (shared variables or an honest address error).
        if (!in->src[0].addr_reg[0] && in->src[0].param[0] &&
            (in->space == PSP_GENERIC || in->space == PSP_LOCAL)) {
            int lv = find_local(c, in->src[0].param);
            if (lv < 0 && in->space == PSP_LOCAL) {
                failf(c, line, "unknown .local variable");
                return;
            }
            if (lv >= 0) {
                if (in->ty == PTY_U8 || in->ty == PTY_U16) {
                    failf(c, line, "sub-word .local access unsupported");
                    return;
                }
                if (c->host_machine) {
                    cparam_t *scp = find_cparam(c, "__acuda_host_state");
                    if (!scp || scp->binding < 0) {
                        failf(c, line, "host-call state buffer missing");
                        return;
                    }
                    uint32_t soff = host_mem_off(
                        c, b, line,
                        c->host_local_off[lv] +
                            (uint32_t)in->src[0].addr_imm);
                    if (in->ty == PTY_U64) {
                        id = load_u64(c, b, scp->binding, soff);
                        def_val(c, in->dst, PTY_U64, id);
                        return;
                    }
                    if (in->ty == PTY_F64) {
                        uint32_t u = load_u64(c, b, scp->binding, soff);
                        if (c->emu_f64) {
                            id = f64bits_to_f32bits(c, b, u);
                        } else {
                            id = nid(c);
                            put(c, b, (4u << 16) | S_Bitcast);
                            put(c, b, c->t_f64);
                            put(c, b, id);
                            put(c, b, u);
                        }
                        def_val(c, in->dst, eff_ty(c, in->ty), id);
                        return;
                    }
                    uint32_t ptr = addr_ptr(c, b, scp->binding, soff);
                    uint32_t w = nid(c);
                    emit_load(c, b, c->t_u32, w, ptr);
                    if (in->ty == PTY_F32) {
                        id = nid(c);
                        put(c, b, (4u << 16) | S_Bitcast);
                        put(c, b, c->t_f32);
                        put(c, b, id);
                        put(c, b, w);
                    } else {
                        id = w;
                    }
                    def_val(c, in->dst, in->ty, id);
                    return;
                }
                uint32_t off = const_u64(c, (uint64_t)in->src[0].addr_imm);
                if (in->ty == PTY_U64) {
                    id = lo_load_u64(c, b, lv, off);
                    def_val(c, in->dst, PTY_U64, id);
                    return;
                }
                if (in->ty == PTY_F64) {
                    uint32_t u = lo_load_u64(c, b, lv, off);
                    if (c->emu_f64) {
                        id = f64bits_to_f32bits(c, b, u);
                    } else {
                        id = nid(c);
                        put(c, b, (4u << 16) | S_Bitcast);
                        put(c, b, c->t_f64);
                        put(c, b, id);
                        put(c, b, u);
                    }
                    def_val(c, in->dst, eff_ty(c, in->ty), id);
                    return;
                }
                {
                    uint32_t ptr = lo_addr_ptr(c, b, lv, off);
                    uint32_t w = nid(c);
                    emit_load(c, b, c->t_u32, w, ptr);
                    if (in->ty == PTY_F32) {
                        id = nid(c);
                        put(c, b, (4u << 16) | S_Bitcast);
                        put(c, b, c->t_f32);
                        put(c, b, id);
                        put(c, b, w);
                    } else {
                        id = w;
                    }
                    def_val(c, in->dst, in->ty, id);
                }
                return;
            }
        }
        int binding = -1;
        uint32_t off = 0;
        int shvar = -1;
        int glvar = -1;
        int64_t sh_imm = 0;
        if (resolve_addr_ex(c, b, &in->src[0], &binding, &off, &shvar,
                            &glvar, &sh_imm, line) != 0)
            return;
        int use_shared = shvar >= 0;
        int use_global = glvar >= 0;
        int use_local = binding == -2;
        if (in->space == PSP_SHARED && !use_shared) {
            failf(c, line, "shared load needs a .shared-derived address");
            return;
        }
        if (in->space == PSP_GLOBAL && use_shared) {
            failf(c, line, "global load of a shared address (space mismatch)");
            return;
        }
        if (in->space == PSP_SHARED && use_global) {
            failf(c, line, "shared load of a .global address (space mismatch)");
            return;
        }
        if (wide_f32) {
            // f32 into a 64-bit holder (ld_st_implicit): the word is
            // zero-extended; only the low word is observable via f32
            // (deterministic zero beats hardware-undefined garbage).
            uint32_t ptr = use_shared ? sh_addr_ptr_imm(c, b, shvar, off,
                                                        sh_imm)
                                      : use_global ? gl_addr_ptr(c, b, glvar, off)
                                      : use_local ? off
                                      : addr_ptr(c, b, binding, off);
            uint32_t w = nid(c);
            emit_load(c, b, c->t_u32, w, ptr);
            id = nid(c);
            put(c, b, (4u << 16) | S_UConvert);
            put(c, b, c->t_u64);
            put(c, b, id);
            put(c, b, w);
            def_val(c, in->dst, PTY_U64, id);
            return;
        }
        if (in->ty == PTY_U8 || in->ty == PTY_U16) {
            int width = in->ty == PTY_U8 ? 8 : 16;
            if (use_local) {
                failf(c, line, "sub-word .local access unsupported");
                return;
            }
            id = use_shared ? narrow_ld_shared(c, b, shvar,
                                               fold_sh_imm(c, b, off, sh_imm),
                                               width)
                            : use_global ? narrow_ld_glvar(c, b, glvar, off, width)
                            : narrow_ld_global(c, b, binding, off, width);
            if (c->failed)
                return;
            reg_t *dr = find_reg(c, in->dst);
            dr->defined = 1;
            dr->cur = id;
            dr->is_ptr = 0;
            dr->binding = -1;
            dr->shvar = -1;
            dr->glvar = -1;
            return;
        }
        if (use_shared) {
            if (shvar < 0) {
                failf(c, line, "shared load needs a .shared-derived address");
                return;
            }
            if (in->ty == PTY_U64) {
                id = sh_load_u64(c, b, shvar,
                                 fold_sh_imm(c, b, off, sh_imm));
                def_val(c, in->dst, PTY_U64, id);
                return;
            }
            if (in->ty == PTY_F64) {
                uint32_t u = sh_load_u64(c, b, shvar,
                                         fold_sh_imm(c, b, off, sh_imm));
                if (c->emu_f64) {
                    id = f64bits_to_f32bits(c, b, u);
                } else {
                    id = nid(c);
                    put(c, b, (4u << 16) | S_Bitcast);
                    put(c, b, c->t_f64);
                    put(c, b, id);
                    put(c, b, u);
                }
                def_val(c, in->dst, eff_ty(c, in->ty), id);
                return;
            }
            {
                uint32_t ptr = sh_addr_ptr_imm(c, b, shvar, off, sh_imm);
                uint32_t w = nid(c);
                if (c->shared_f32_view) {
                    emit_load(c, b, c->t_f32, w, ptr);
                    if (in->ty == PTY_F32) {
                        id = w;
                    } else {
                        id = nid(c);
                        put(c, b, (4u << 16) | S_Bitcast);
                        put(c, b, c->t_u32);
                        put(c, b, id);
                        put(c, b, w);
                    }
                } else {
                    emit_load(c, b, c->t_u32, w, ptr);
                    if (in->ty == PTY_F32) {
                        id = nid(c);
                        put(c, b, (4u << 16) | S_Bitcast);
                        put(c, b, c->t_f32);
                        put(c, b, id);
                        put(c, b, w);
                    } else {
                        id = w;
                    }
                }
                def_val(c, in->dst, in->ty, id);
            }
            return;
        }
        if (use_global) {
            // .global image read (same widths as shared; F16 halves
            // move as .u16 bits plus cvt, like everywhere else).
            if (in->ty == PTY_F16) {
                failf(c, line, "global f16 loads unsupported (use .u16 bits)");
                return;
            }
            if (in->ty == PTY_U64) {
                id = gl_load_u64(c, b, glvar, off);
                def_val(c, in->dst, PTY_U64, id);
                return;
            }
            if (in->ty == PTY_F64) {
                uint32_t u = gl_load_u64(c, b, glvar, off);
                if (c->emu_f64) {
                    id = f64bits_to_f32bits(c, b, u);
                } else {
                    id = nid(c);
                    put(c, b, (4u << 16) | S_Bitcast);
                    put(c, b, c->t_f64);
                    put(c, b, id);
                    put(c, b, u);
                }
                def_val(c, in->dst, eff_ty(c, in->ty), id);
                return;
            }
            {
                uint32_t ptr = gl_addr_ptr(c, b, glvar, off);
                uint32_t w = nid(c);
                emit_load(c, b, c->t_u32, w, ptr);
                if (in->ty == PTY_F32) {
                    id = nid(c);
                    put(c, b, (4u << 16) | S_Bitcast);
                    put(c, b, c->t_f32);
                    put(c, b, id);
                    put(c, b, w);
                } else {
                    id = w;
                }
                def_val(c, in->dst, in->ty, id);
            }
            return;
        }
        if (shvar >= 0) {
            failf(c, line, "global load of a shared address (space mismatch)");
            return;
        }
        if (use_local) {
            int lv = c->last_local;
            if (lv < 0) {
                if (getenv("ACUDA_TRACE") && in->src[0].kind == OPD_ADDR) {
                    reg_t *_rr = find_reg(c, in->src[0].addr_reg);
                    fprintf(stderr, "acuda-local-warn st-nolocal reg=%s rr=%p bind=%d lovar=%d ptr=%d def=%d\n",
                            in->src[0].addr_reg, (void*)_rr, _rr?_rr->binding:-9, _rr?_rr->lovar:-9, _rr?_rr->is_ptr:-9, _rr?_rr->defined:-9);
                }
                failf(c, line, "local address has no backing variable");
                return;
            }
            if (in->ty == PTY_U64) {
                id = lo_load_u64(c, b, lv, off);
                def_val(c, in->dst, PTY_U64, id);
                return;
            }
            if (in->ty == PTY_F64) {
                uint32_t u = lo_load_u64(c, b, lv, off);
                if (c->emu_f64) {
                    id = f64bits_to_f32bits(c, b, u);
                } else {
                    id = nid(c);
                    put(c, b, (4u << 16) | S_Bitcast);
                    put(c, b, c->t_f64);
                    put(c, b, id);
                    put(c, b, u);
                }
                def_val(c, in->dst, eff_ty(c, in->ty), id);
                return;
            }
            uint32_t ptr = lo_addr_ptr(c, b, lv, off);
            uint32_t w = nid(c);
            emit_load(c, b, c->t_u32, w, ptr);
            if (in->ty == PTY_F32) {
                id = nid(c);
                put(c, b, (4u << 16) | S_Bitcast);
                put(c, b, c->t_f32);
                put(c, b, id);
                put(c, b, w);
            } else {
                id = w;
            }
            def_val(c, in->dst, in->ty, id);
            return;
        }
        if (in->ty == PTY_U64) {
            // 4-byte alignment required (documented precondition).
            id = load_u64(c, b, binding, off);
            def_val(c, in->dst, PTY_U64, id);
            return;
        }
        if (in->ty == PTY_F64) {
            // 8-byte alignment required (documented precondition).
            uint32_t u = load_u64(c, b, binding, off);
            if (c->emu_f64) {
                // Demote at the boundary (integer-only, no Float64):
                // memory still holds doubles.
                id = f64bits_to_f32bits(c, b, u);
            } else {
                id = nid(c);
                put(c, b, (4u << 16) | S_Bitcast);
                put(c, b, c->t_f64);
                put(c, b, id);
                put(c, b, u);
            }
            def_val(c, in->dst, eff_ty(c, in->ty), id);
            return;
        }
        {
            uint32_t ptr = addr_ptr(c, b, binding, off);
            uint32_t w = nid(c);
            emit_load(c, b, c->t_u32, w, ptr);
            if (in->ty == PTY_F32) {
                id = nid(c);
                put(c, b, (4u << 16) | S_Bitcast);
                put(c, b, c->t_f32);
                put(c, b, id);
                put(c, b, w);
            } else {
                id = w;
            }
            def_val(c, in->dst, in->ty, id);
        }
        return;
    }
