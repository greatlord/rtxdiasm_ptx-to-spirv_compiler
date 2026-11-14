#include "instructions/memory/memory.h"

void emit_st_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t d;
        if (in->ty != PTY_U32 && in->ty != PTY_F32 && in->ty != PTY_F64 &&
            in->ty != PTY_U64 && in->ty != PTY_U8 && in->ty != PTY_U16 &&
            in->ty != PTY_F16) {
            failf(c, line, "store only supports .u8/.u16/.u32/.f16/.f32/.f64/.u64");
            return;
        }
        // Direct .local variable use ([var]/[var+N], generic or .local
        // space): Private per-invocation storage (mirrors the ld side).
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
                d = resolve(c, b, &in->src[1], in->ty, line);
                if (c->failed)
                    return;
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
                        store_u64(c, b, scp->binding, soff, d.id);
                        return;
                    }
                    if (in->ty == PTY_F64) {
                        uint32_t u;
                        if (c->emu_f64) {
                            u = f32bits_to_f64bits(c, b, d.id);
                        } else {
                            u = nid(c);
                            put(c, b, (4u << 16) | S_Bitcast);
                            put(c, b, c->t_u64);
                            put(c, b, u);
                            put(c, b, d.id);
                        }
                        store_u64(c, b, scp->binding, soff, u);
                        return;
                    }
                    uint32_t ptr = addr_ptr(c, b, scp->binding, soff);
                    uint32_t w = d.id;
                    if (in->ty == PTY_F32) {
                        w = nid(c);
                        put(c, b, (4u << 16) | S_Bitcast);
                        put(c, b, c->t_u32);
                        put(c, b, w);
                        put(c, b, d.id);
                    }
                    emit_store(c, b, ptr, w);
                    return;
                }
                uint32_t off = const_u64(c, (uint64_t)in->src[0].addr_imm);
                if (in->ty == PTY_U64) {
                    lo_store_u64(c, b, lv, off, d.id);
                    return;
                }
                if (in->ty == PTY_F64) {
                    uint32_t u;
                    if (c->emu_f64) {
                        u = f32bits_to_f64bits(c, b, d.id);
                    } else {
                        u = nid(c);
                        put(c, b, (4u << 16) | S_Bitcast);
                        put(c, b, c->t_u64);
                        put(c, b, u);
                        put(c, b, d.id);
                    }
                    lo_store_u64(c, b, lv, off, u);
                    return;
                }
                {
                    uint32_t ptr = lo_addr_ptr(c, b, lv, off);
                    uint32_t w = d.id;
                    if (in->ty == PTY_F32) {
                        w = nid(c);
                        put(c, b, (4u << 16) | S_Bitcast);
                        put(c, b, c->t_u32);
                        put(c, b, w);
                        put(c, b, d.id);
                    }
                    emit_store(c, b, ptr, w);
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
        if (in->space == PSP_SHARED && !use_shared) {
            failf(c, line, "shared store needs a .shared-derived address");
            return;
        }
        if (in->space == PSP_GLOBAL && use_shared) {
            failf(c, line, "global store of a shared address (space mismatch)");
            return;
        }
        if (glvar >= 0) {
            failf(c, line, "stores to .global variables unsupported (no device backing)");
            return;
        }
        if (in->ty == PTY_U8 || in->ty == PTY_U16) {
            if (binding == -2) {
                failf(c, line, "sub-word .local access unsupported");
                return;
            }
            // Lenient value: any integer register (low bits stored).
            int width = in->ty == PTY_U8 ? 8 : 16;
            uint32_t vid = 0;
            if (in->src[1].kind == OPD_REG) {
                reg_t *vr = find_reg(c, in->src[1].reg);
                if (!vr || !vr->defined) {
                    failf(c, line, "use of undeclared register");
                    return;
                }
                if (decl_int_width(vr->decl_ty) < 0 &&
                    !(width == 16 && vr->decl_ty == PTY_F16)) {
                    failf(c, line, "narrow store value must be an integer register");
                    return;
                }
                vid = vr->cur;
            } else {
                val_t vv = resolve(c, b, &in->src[1], PTY_U32, line);
                if (c->failed)
                    return;
                vid = vv.id;
            }
            if (use_shared)
                narrow_st_shared(c, b, shvar,
                                 fold_sh_imm(c, b, off, sh_imm), vid,
                                 width);
            else
                narrow_st_global(c, b, binding, off, vid, width);
            return;
        }
        d = resolve(c, b, &in->src[1], in->ty, line);
        if (c->failed)
            return;
        if (use_shared) {
            if (in->ty == PTY_U64) {
                sh_store_u64(c, b, shvar,
                             fold_sh_imm(c, b, off, sh_imm), d.id);
                return;
            }
            if (in->ty == PTY_F64) {
                uint32_t u;
                if (c->emu_f64) {
                    u = f32bits_to_f64bits(c, b, d.id);
                } else {
                    u = nid(c);
                    put(c, b, (4u << 16) | S_Bitcast);
                    put(c, b, c->t_u64);
                    put(c, b, u);
                    put(c, b, d.id);
                }
                sh_store_u64(c, b, shvar,
                             fold_sh_imm(c, b, off, sh_imm), u);
                return;
            }
            {
                uint32_t ptr = sh_addr_ptr_imm(c, b, shvar, off, sh_imm);
                uint32_t w = d.id;
                if (c->shared_f32_view) {
                    if (in->ty != PTY_F32) {
                        w = nid(c);
                        put(c, b, (4u << 16) | S_Bitcast);
                        put(c, b, c->t_f32);
                        put(c, b, w);
                        put(c, b, d.id);
                    }
                } else if (in->ty == PTY_F32) {
                    w = nid(c);
                    put(c, b, (4u << 16) | S_Bitcast);
                    put(c, b, c->t_u32);
                    put(c, b, w);
                    put(c, b, d.id);
                }
                emit_store(c, b, ptr, w);
            }
            return;
        }
        if (binding == -2) {
            int lv = c->last_local;
            if (lv < 0) {
                {
                    const char *_an = NULL;
                    if (in->src[0].kind == OPD_ADDR) _an = in->src[0].addr_reg;
                    else if (in->src[1].kind == OPD_ADDR) _an = in->src[1].addr_reg;
                    if (_an) {
                        reg_t *_rr = find_reg(c, _an);
                        ACUDA_DEBUG_PRINT("acuda-local-warn st-nolocal reg=%s bind=%d lovar=%d ptr=%d def=%d\n",
                                _an, _rr?_rr->binding:-9, _rr?_rr->lovar:-9, _rr?_rr->is_ptr:-9, _rr?_rr->defined:-9);
                    } else {
                        ACUDA_DEBUG_PRINT("acuda-local-warn st-nolocal (no addr opd) s0k=%d s1k=%d\n", in->src[0].kind, in->src[1].kind);
                    }
                }
                failf(c, line, "local address has no backing variable");
                return;
            }
            if (in->ty == PTY_U64) {
                lo_store_u64(c, b, lv, off, d.id);
                return;
            }
            if (in->ty == PTY_F64) {
                uint32_t u;
                if (c->emu_f64) {
                    u = f32bits_to_f64bits(c, b, d.id);
                } else {
                    u = nid(c);
                    put(c, b, (4u << 16) | S_Bitcast);
                    put(c, b, c->t_u64);
                    put(c, b, u);
                    put(c, b, d.id);
                }
                lo_store_u64(c, b, lv, off, u);
                return;
            }
            uint32_t ptr = lo_addr_ptr(c, b, lv, off);
            uint32_t w = d.id;
            if (in->ty == PTY_F32) {
                w = nid(c);
                put(c, b, (4u << 16) | S_Bitcast);
                put(c, b, c->t_u32);
                put(c, b, w);
                put(c, b, d.id);
            }
            emit_store(c, b, ptr, w);
            return;
        }
        if (in->ty == PTY_U64) {
            store_u64(c, b, binding, off, d.id);
            return;
        }
        if (in->ty == PTY_F64) {
            uint32_t u;
            if (c->emu_f64) {
                // Demote at the boundary (integer-only, no Float64):
                // memory still holds doubles.
                u = f32bits_to_f64bits(c, b, d.id);
            } else {
                u = nid(c);
                put(c, b, (4u << 16) | S_Bitcast);
                put(c, b, c->t_u64);
                put(c, b, u);
                put(c, b, d.id);
            }
            store_u64(c, b, binding, off, u);
            return;
        }
        {
            uint32_t ptr = addr_ptr(c, b, binding, off);
            uint32_t w = d.id;
            if (in->ty == PTY_F32) {
                w = nid(c);
                put(c, b, (4u << 16) | S_Bitcast);
                put(c, b, c->t_u32);
                put(c, b, w);
                put(c, b, d.id);
            }
            emit_store(c, b, ptr, w);
        }
        return;
    }
