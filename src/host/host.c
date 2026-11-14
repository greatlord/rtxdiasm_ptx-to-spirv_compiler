// Split-dispatch host-call lowering: request marshalling, scalar state
// spill/restore, and return-value reload. Pointer registers live across
// a call are replayed from their entry-parameter loads by the splitter;
// this file only serializes scalar/predicate state.
#include <stdint.h>
#include <string.h>
#include "spirv_internal.h"
#include "acuda/hostcall.h"

static uint32_t host_index(ctx_t *c, spv_blob_t *b, int line) {
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

static uint32_t host_off(ctx_t *c, spv_blob_t *b, uint32_t index,
                         uint32_t stride, uint32_t field) {
    uint32_t sid = const_u64(c, stride);
    uint32_t mul = nid(c);
    emit_arith(c, b, S_IMul, c->t_u64, mul, index, sid);
    uint32_t off = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, off, mul,
               const_u64(c, field));
    return off;
}

static void host_store_u32(ctx_t *c, spv_blob_t *b, int binding,
                           uint32_t off, uint32_t val) {
    uint32_t ptr = addr_ptr(c, b, binding, off);
    emit_store(c, b, ptr, val);
}

static uint32_t host_load_u32(ctx_t *c, spv_blob_t *b, int binding,
                              uint32_t off) {
    uint32_t ptr = addr_ptr(c, b, binding, off);
    uint32_t id = nid(c);
    emit_load(c, b, c->t_u32, id, ptr);
    return id;
}

static uint32_t host_bitcast(ctx_t *c, spv_blob_t *b, uint32_t ty,
                             uint32_t id) {
    uint32_t out = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, ty);
    put(c, b, out);
    put(c, b, id);
    return out;
}

static uint32_t host_to_u64(ctx_t *c, spv_blob_t *b, uint32_t id) {
    uint32_t out = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u64);
    put(c, b, out);
    put(c, b, id);
    return out;
}

static uint32_t host_to_u32(ctx_t *c, spv_blob_t *b, uint32_t id) {
    uint32_t out = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u32);
    put(c, b, out);
    put(c, b, id);
    return out;
}

static cparam_t *host_param(ctx_t *c, const char *name, int line) {
    cparam_t *cp = find_cparam(c, name);
    if (!cp || cp->kind != 1 || cp->binding < 0) {
        failf(c, line, "host-call internal buffer parameter missing");
        return NULL;
    }
    return cp;
}

uint32_t host_mem_off(ctx_t *c, spv_blob_t *b, int line, uint32_t field) {
    cparam_t *cp = find_cparam(c, "__acuda_host_state");
    if (!cp || cp->binding < 0) {
        failf(c, line, "host-call state buffer missing");
        return 0;
    }
    uint32_t base32 = host_load_u32(c, b, cp->binding, const_u64(c, 0));
    uint32_t base64 = host_to_u64(c, b, base32);
    uint32_t idx = host_index(c, b, line);
    if (c->failed)
        return 0;
    uint32_t off = host_off(c, b, idx, c->host_state_stride, field);
    uint32_t sum = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, sum, base64, off);
    return sum;
}

uint32_t host_global_off_bytes(ctx_t *c, spv_blob_t *b, int var,
                               uint32_t byteoff) {
    return host_mem_off(c, b, 0,
                        c->host_global_off[var] + byteoff);
}

uint32_t host_shared_off_bytes(ctx_t *c, spv_blob_t *b, int var,
                               uint32_t byteoff) {
    cparam_t *cp = find_cparam(c, "__acuda_host_state");
    if (!cp || cp->binding < 0)
        return const_u64(c, 0); // no host state: shared offset is canonical
    uint32_t stride32 = host_load_u32(c, b, cp->binding, const_u64(c, 4));
    uint32_t stride = host_to_u64(c, b, stride32);
    ptx_opd_t op;
    memset(&op, 0, sizeof(op));
    op.kind = OPD_SPECIAL;
    op.special = SPC_CTAID;
    val_t cta = resolve(c, b, &op, PTY_U32, 0);
    uint32_t cta64 = host_to_u64(c, b, cta.id);
    uint32_t mul = nid(c);
    emit_arith(c, b, S_IMul, c->t_u64, mul, cta64, stride);
    uint32_t base = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, base, mul, const_u64(c, 16));
    uint32_t off = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, off, base,
               const_u64(c, (uint32_t)c->host_shared_off[var] + byteoff));
    return off;
}

void emit_host_request(ctx_t *c, spv_blob_t *b,
                       const ptx_instr_t *in) {
    int line = in->line;
    if (in->host_call_id < 0 || in->host_call_id >= c->nextfuncs) {
        failf(c, line, "host call id out of range");
        return;
    }
    const ptx_extfunc_t *ef = &c->extfuncs[in->host_call_id];
    cparam_t *cp = host_param(c, "__acuda_host_req", line);
    if (!cp)
        return;
    uint32_t idx = host_index(c, b, line);
    if (c->failed)
        return;
    // Header: call id in the low word, pending flag in the high word.
    uint64_t hdr = (uint64_t)(uint32_t)in->host_call_id | (1ull << 32);
    uint32_t off0 = host_off(c, b, idx, ACUDA_HOST_REQ_STRIDE, 0);
    store_u64(c, b, cp->binding, off0, const_u64(c, hdr));

    if (in->nsrc != ef->nargs) {
        failf(c, line, "host call argument count mismatch");
        return;
    }
    for (int a = 0; a < in->nsrc; a++) {
        ptx_ty_t ty = ef->arg_ty[a];
        reg_t *pr = in->call_args[a].kind == OPD_REG
                        ? find_reg(c, in->call_args[a].reg)
                        : NULL;
        uint32_t bind_id = const_u32(c, 0xffffffffu);
        uint32_t value;
        int is_ptr = pr && pr->defined && pr->is_ptr;
        if (is_ptr) {
            if (pr->binding >= 0) {
                bind_id = const_u32(c, (uint32_t)pr->binding);
                value = pr->off;
            } else if (pr->dyn_home) {
                uint32_t tag = nid(c);
                put(c, b, (4u << 16) | S_Load);
                put(c, b, c->t_u32);
                put(c, b, tag);
                put(c, b, pr->dyn_tag);
                bind_id = tag;
                value = pr->off;
            } else if (pr->shvar >= 0) {
                // Shared pointers live in the workgroup's state-backed
                // shared area; pass the absolute state offset.
                bind_id = const_u32(c, 0xfffffffeu);
                uint32_t base = host_shared_off_bytes(c, b, pr->shvar, 0);
                value = nid(c);
                emit_arith(c, b, S_IAdd, c->t_u64, value, base, pr->off);
            } else {
                failf(c, line,
                      "host call pointer argument must be a global or shared pointer");
                return;
            }
        } else {
        ptx_ty_t want = (ty == PTY_U8 || ty == PTY_U16) ? PTY_U32 : ty;
        val_t v = resolve(c, b, &in->call_args[a], want, line);
        if (c->failed)
            return;
        if (v.is_ptr) {
            if (v.binding < 0) {
                failf(c, line,
                      "host call pointer argument must be a global buffer pointer");
                return;
            }
            bind_id = const_u32(c, (uint32_t)v.binding);
            value = v.off;
        } else if (ty == PTY_F64) {
            value = host_bitcast(c, b, c->t_u64, v.id);
        } else if (ty == PTY_U64) {
            value = v.id;
        } else if (ty == PTY_F32 || ty == PTY_F16) {
            uint32_t u = host_bitcast(c, b, c->t_u32, v.id);
            value = host_to_u64(c, b, u);
        } else {
            value = host_to_u64(c, b, v.id);
        }
        }
        uint32_t aoff = host_off(c, b, idx, ACUDA_HOST_REQ_STRIDE,
                                 ACUDA_HOST_REQ_ARGS_OFF +
                                     (uint32_t)ef->arg_off[a]);
        store_u64(c, b, cp->binding, aoff, value);
        uint32_t boff = host_off(c, b, idx, ACUDA_HOST_REQ_STRIDE,
                                 ACUDA_HOST_REQ_PTR_BIND_OFF +
                                     (uint32_t)a * 4u);
        host_store_u32(c, b, cp->binding, boff, bind_id);
    }
}

void emit_host_result(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    if (in->host_call_id < 0 || in->host_call_id >= c->host_call_count) {
        failf(c, line, "host result call id out of range");
        return;
    }
    cparam_t *cp = host_param(c, "__acuda_host_state", line);
    if (!cp)
        return;
    uint32_t base = c->host_ret_off[in->host_call_id] +
                    (uint32_t)(in->host_arg_off < 0 ? 0
                                                    : in->host_arg_off);
    uint32_t off = host_mem_off(c, b, line, base);
    size_t width = in->ty == PTY_U64 || in->ty == PTY_F64 ? 8
                   : in->ty == PTY_U16 || in->ty == PTY_F16 ||
                             in->ty == PTY_BF16 ? 2
                   : in->ty == PTY_U8 ? 1
                                      : 4;
    uint32_t val;
    if (width == 8 && (base & 3u) == 0) {
        val = load_u64(c, b, cp->binding, off);
        if (in->ty == PTY_F64)
            val = host_bitcast(c, b, c->t_f64, val);
    } else {
        uint32_t word = host_load_u32(c, b, cp->binding, off);
        uint32_t shift = (base & 3u) * 8u;
        if (shift) {
            uint32_t sh = const_u32(c, shift);
            uint32_t sv = nid(c);
            emit_arith(c, b, S_ShrL, c->t_u32, sv, word, sh);
            word = sv;
        }
        if (width < 4) {
            uint32_t mask = const_u32(c, width == 1 ? 0xffu : 0xffffu);
            uint32_t av = nid(c);
            emit_arith(c, b, S_And, c->t_u32, av, word, mask);
            word = av;
        }
        if (in->ty == PTY_F32 || in->ty == PTY_F16)
            word = host_bitcast(c, b, c->t_f32, word);
        val = word;
    }
    def_val(c, in->dst, in->ty, val);
}

void emit_host_reply(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    cparam_t *cp = host_param(c, "__acuda_host_req", line);
    if (!cp)
        return;
    uint32_t idx = host_index(c, b, line);
    if (c->failed)
        return;
    uint32_t off = host_off(c, b, idx, ACUDA_HOST_REQ_STRIDE,
                            ACUDA_HOST_REQ_RET_OFF);
    if (in->ty == PTY_U64) {
        uint32_t id = load_u64(c, b, cp->binding, off);
        def_val(c, in->dst, PTY_U64, id);
    } else {
        uint32_t id = host_load_u32(c, b, cp->binding, off);
        def_val(c, in->dst, in->ty, id);
    }
}

void emit_host_spill(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    cparam_t *cp = host_param(c, "__acuda_host_state", line);
    if (!cp)
        return;
    uint32_t idx = host_index(c, b, line);
    if (c->failed)
        return;
    uint32_t stride = (uint32_t)c->nregs * ACUDA_HOST_STATE_REG_STRIDE;
    for (int i = 0; i < c->nregs; i++) {
        reg_t *r = &c->regs[i];
        if (c->cur_fn && c->cur_fn->host_skip && c->cur_fn->host_skip[i])
            continue;
        uint32_t base = (uint32_t)i * ACUDA_HOST_STATE_REG_STRIDE;
        uint32_t tag = 0, aux = 0, val = const_u64(c, 0);
        if (r->defined) {
            if (r->is_ptr) {
                tag = r->shvar >= 0 ? 2 : (r->glvar >= 0 ? 3 : 1);
                aux = (uint32_t)(r->shvar >= 0 ? r->shvar
                                  : r->glvar >= 0 ? r->glvar
                                                  : r->binding);
                val = r->off;
            } else if (r->decl_ty == PTY_PRED) {
                tag = 4;
                uint32_t one = const_u32(c, 1);
                uint32_t zero = const_u32(c, 0);
                uint32_t sel = nid(c);
                put(c, b, (6u << 16) | S_Select);
                put(c, b, c->t_u32);
                put(c, b, sel);
                put(c, b, one);
                put(c, b, zero);
                put(c, b, r->cur);
                val = host_to_u64(c, b, sel);
            } else if (r->decl_ty == PTY_U64) {
                val = r->cur;
            } else if (r->decl_ty == PTY_F64) {
                val = host_bitcast(c, b, c->t_u64, r->cur);
            } else {
                uint32_t u = r->cur;
                if (r->decl_ty == PTY_F32 || r->decl_ty == PTY_F16)
                    u = host_bitcast(c, b, c->t_u32, u);
                val = host_to_u64(c, b, u);
            }
        }
        uint32_t off = host_off(c, b, idx, stride, base);
        host_store_u32(c, b, cp->binding, off, const_u32(c, tag));
        uint32_t offa = host_off(c, b, idx, stride, base + 4);
        host_store_u32(c, b, cp->binding, offa, const_u32(c, aux));
        uint32_t offv = host_off(c, b, idx, stride, base + 8);
        store_u64(c, b, cp->binding, offv, val);
    }
}

void emit_host_restore(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    cparam_t *cp = host_param(c, "__acuda_host_state", line);
    if (!cp)
        return;
    uint32_t idx = host_index(c, b, line);
    if (c->failed)
        return;
    uint32_t stride = (uint32_t)c->nregs * ACUDA_HOST_STATE_REG_STRIDE;
    for (int i = 0; i < c->nregs; i++) {
        reg_t *r = &c->regs[i];
        if (c->cur_fn && c->cur_fn->host_skip &&
            c->cur_fn->host_skip[c->nregs + i])
            continue;
        uint32_t off = host_off(c, b, idx, stride,
                                (uint32_t)i * ACUDA_HOST_STATE_REG_STRIDE +
                                    8);
        uint32_t v = load_u64(c, b, cp->binding, off);
        switch (r->decl_ty) {
        case PTY_U64:
            def_val(c, r->name, PTY_U64, v);
            break;
        case PTY_F64: {
            uint32_t f = host_bitcast(c, b, c->t_f64, v);
            def_val(c, r->name, PTY_F64, f);
            break;
        }
        case PTY_PRED: {
            uint32_t bv = nid(c);
            put(c, b, (5u << 16) | S_INotEqual);
            put(c, b, c->t_bool);
            put(c, b, bv);
            put(c, b, v);
            put(c, b, const_u64(c, 0));
            def_val(c, r->name, PTY_PRED, bv);
            break;
        }
        case PTY_F32:
        case PTY_F16: {
            uint32_t u = host_to_u32(c, b, v);
            uint32_t f = host_bitcast(c, b, c->t_f32, u);
            def_val(c, r->name, r->decl_ty, f);
            break;
        }
        default: {
            uint32_t u = host_to_u32(c, b, v);
            def_val(c, r->name, r->decl_ty, u);
            break;
        }
        }
    }
}
