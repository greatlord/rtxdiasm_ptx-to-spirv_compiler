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

void emit_instr(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    if (c->failed)
        return;
    int line = in->line;
    val_t a, c2;
    uint32_t id;
    switch (in->op) {
    case POP_MOV:
        if (in->ty == PTY_PRED) {
            // Predicate copy (vote masks): register, !register, 0/1.
            // Predicated form selects over the current value like
            // u32 movs (OpUndef on first definition).
            if (check_dst(c, in->dst, PTY_PRED, line) != 0)
                return;
            uint32_t v = pred_val(c, b, &in->src[0], in->bneg, line);
            if (c->failed)
                return;
            if (!in->pred[0]) {
                def_val(c, in->dst, PTY_PRED, v);
                return;
            }
            reg_t *pr = find_reg(c, in->pred);
            if (!pr || pr->decl_ty != PTY_PRED) {
                failf(c, line, "mov predicate must be a .pred register");
                return;
            }
            if (!pr->defined) {
                failf(c, line, "mov predicate has no value");
                return;
            }
            uint32_t cond = pr->cur;
            if (in->pred_neg) {
                uint32_t ncond = nid(c);
                put(c, b, (4u << 16) | S_LogicalNot);
                put(c, b, c->t_bool);
                put(c, b, ncond);
                put(c, b, pr->cur);
                cond = ncond;
            }
            reg_t *dr = find_reg(c, in->dst);
            uint32_t oldv;
            if (dr && dr->defined) {
                oldv = dr->cur;
            } else {
                oldv = nid(c);
                put(c, b, (3u << 16) | 1); // OpUndef (grammar: 1)
                put(c, b, c->t_bool);
                put(c, b, oldv);
            }
            id = nid(c);
            put(c, b, (6u << 16) | S_Select);
            put(c, b, c->t_bool);
            put(c, b, id);
            put(c, b, cond);
            put(c, b, v);
            put(c, b, oldv);
            def_val(c, in->dst, PTY_PRED, id);
            return;
        }
        if (check_dst(c, in->dst, in->ty, line) != 0)
            return;
        if (in->nsrc == 2) {
            // Brace-vector pack (first operand is LOW): mov.b64 d,{lo,hi}
            // concatenates two u32 words; mov.b32 d,{lo16,hi16} packs two
            // 16-bit halves (operands masked, like PTX 16-bit sources).
            if (in->pred[0]) {
                failf(c, line, "predicated vector mov unsupported");
                return;
            }
            // Brace sources resolve as u32, except 16-bit holders in a
            // .b32 pack (mov.b32 d, {lo16, hi16} takes 16-bit values).
            val_t av, bv;
            memset(&av, 0, sizeof(av));
            memset(&bv, 0, sizeof(bv));
            for (int sni = 0; sni < 2; sni++) {
                ptx_ty_t want = PTY_U32;
                if (in->ty == PTY_U32 && in->src[sni].kind == OPD_REG) {
                    reg_t *sr = find_reg(c, in->src[sni].reg);
                    if (sr && sr->decl_ty == PTY_U16)
                        want = PTY_U16;
                }
                val_t rv = resolve(c, b, &in->src[sni], want, line);
                if (c->failed)
                    return;
                if (sni == 0)
                    av = rv;
                else
                    bv = rv;
            }
            a = av;
            c2 = bv;
            if (in->ty == PTY_U64) {
                uint32_t wa = nid(c), wb = nid(c);
                put(c, b, (4u << 16) | S_UConvert);
                put(c, b, c->t_u64);
                put(c, b, wa);
                put(c, b, a.id);
                put(c, b, (4u << 16) | S_UConvert);
                put(c, b, c->t_u64);
                put(c, b, wb);
                put(c, b, c2.id);
                uint32_t sh = nid(c);
                emit_arith(c, b, S_Shl, c->t_u64, sh, wb, const_u64(c, 32));
                id = nid(c);
                emit_arith(c, b, S_Or, c->t_u64, id, sh, wa);
                def_val(c, in->dst, PTY_U64, id);
                return;
            }
            if (in->ty == PTY_U32) {
                uint32_t lo = nid(c), hi = nid(c);
                emit_arith(c, b, S_And, c->t_u32, lo, a.id, const_u32(c, 0xffffu));
                emit_arith(c, b, S_And, c->t_u32, hi, c2.id, const_u32(c, 0xffffu));
                uint32_t sh = nid(c);
                emit_arith(c, b, S_Shl, c->t_u32, sh, hi, const_u32(c, 16));
                id = nid(c);
                emit_arith(c, b, S_Or, c->t_u32, id, sh, lo);
                def_val(c, in->dst, PTY_U32, id);
                return;
            }
            failf(c, line, "vector mov supports .b64/.b32 only");
            return;
        }
        if (in->pred[0]) {
            // Predicated mov lowers as a select over the current value
            // (OpUndef when first-defined). Pointer copies are
            // compile-time bindings, inexpressible as a value select:
            // predicated pointer moves and overwrites stay clean errors.
            reg_t *pr = find_reg(c, in->pred);
            if (!pr || pr->decl_ty != PTY_PRED) {
                failf(c, line, "mov predicate must be a .pred register");
                return;
            }
            if (!pr->defined) {
                failf(c, line, "mov predicate has no value");
                return;
            }
            if (in->src[0].kind == OPD_REG) {
                reg_t *sr = find_reg(c, in->src[0].reg);
                if (sr && sr->defined && sr->is_ptr) {
                    failf(c, line, "predicated mov of pointer unsupported");
                    return;
                }
            }
            if (in->src[0].kind == OPD_PARAM && in->ty == PTY_U64) {
                failf(c, line, "predicated mov of variable address unsupported");
                return;
            }
            reg_t *dr = find_reg(c, in->dst);
            if (dr && dr->defined && dr->is_ptr) {
                failf(c, line, "predicated overwrite of pointer register unsupported");
                return;
            }
            a = resolve(c, b, &in->src[0], in->ty, line);
            if (c->failed)
                return;
            uint32_t cond = pr->cur;
            if (in->pred_neg) {
                uint32_t ncond = nid(c);
                put(c, b, (4u << 16) | S_LogicalNot);
                put(c, b, c->t_bool);
                put(c, b, ncond);
                put(c, b, pr->cur);
                cond = ncond;
            }
            uint32_t oldv;
            if (dr && dr->defined) {
                oldv = dr->cur;
            } else {
                oldv = nid(c);
                put(c, b, (3u << 16) | 1); // OpUndef (grammar: 1)
                put(c, b, ty_of(c, in->ty));
                put(c, b, oldv);
            }
            // The register may currently hold a value of a different
            // bit-compatible type (a .b32 move over a register last
            // written as f32, for example).  OpSelect requires all
            // operands to share the result type, so bitcast the old
            // value when needed.
            uint32_t want_ty = ty_of(c, in->ty);
            if (dr && dr->defined && dr->cur_ty != in->ty) {
                uint32_t have_ty = ty_of(c, dr->cur_ty);
                if (have_ty != want_ty &&
                    ((have_ty == c->t_u32 && want_ty == c->t_f32) ||
                     (have_ty == c->t_f32 && want_ty == c->t_u32) ||
                     (have_ty == c->t_u64 && want_ty == c->t_f64) ||
                     (have_ty == c->t_f64 && want_ty == c->t_u64))) {
                    uint32_t converted = nid(c);
                    put(c, b, (4u << 16) | S_Bitcast);
                    put(c, b, want_ty);
                    put(c, b, converted);
                    put(c, b, oldv);
                    oldv = converted;
                }
            }
            id = nid(c);
            put(c, b, (6u << 16) | S_Select);
            put(c, b, want_ty);
            put(c, b, id);
            put(c, b, cond);
            put(c, b, a.id);
            put(c, b, oldv);
            def_val(c, in->dst, in->ty, id);
            return;
        }
        if (in->src[0].kind == OPD_REG) {
            reg_t *r = find_reg(c, in->src[0].reg);
            if (!r || !r->defined) {
                failf(c, line, "use of undeclared register");
                return;
            }
            if (r->is_ptr) {
                if (in->ty == PTY_U64 && r->paddr >= 0) {
                    if (check_dst(c, in->dst, PTY_U64, line) != 0)
                        return;
                    def_paddr(c, in->dst, r->paddr);
                    return;
                }
                if (in->ty == PTY_U64) {
                    if (r->decl_ty == PTY_U64) {
                        if (r->shvar >= 0)
                            def_shared(c, in->dst, r->shvar, r->off);
                        else if (r->glvar >= 0)
                            def_global(c, in->dst, r->glvar, r->off);
                        else
                            def_ptr(c, in->dst, r->binding, r->off);
                        reg_t *dr = find_reg(c, in->dst);
                        if (dr && r->binding >= 0)
                            dr->is_tex = r->is_tex;
                        return;
                    } else if (r->decl_ty == PTY_F64) {
                        if (r->shvar >= 0)
                            def_shared(c, in->dst, r->shvar, r->off);
                        else if (r->glvar >= 0)
                            def_global(c, in->dst, r->glvar, r->off);
                        else
                            def_ptr(c, in->dst, r->binding, r->off);
                        reg_t *dr = find_reg(c, in->dst);
                        if (dr && r->binding >= 0)
                            dr->is_tex = r->is_tex;
                        return;
                    } else {
                        failf(c, line, "pointer copy type mismatch");
                        return;
                    }
                } else if (in->ty == PTY_F64) {
                    if (check_dst(c, in->dst, PTY_F64, line) != 0)
                        return;
                    reg_t *dr = find_reg(c, in->dst);
                    if (dr) {
                        dr->defined = 1;
                        dr->decl_ty = PTY_F64;
                        dr->is_ptr = 1;
                        dr->binding = r->binding;
                        dr->off = r->off;
                        dr->shvar = r->shvar;
                        dr->glvar = r->glvar;
                        if (r->binding >= 0)
                            dr->is_tex = r->is_tex;
                    }
                    return;
                } else if (in->ty == PTY_U32) {
                    // A materialized pointer word from a 64-bit pointer
                    // (lifted 64-bit addressing) is a value, not a
                    // 32-bit pointer copy.
                    if (r->decl_ty != PTY_U32 && r->cur &&
                        r->cur_ty == PTY_U32) {
                        if (check_dst(c, in->dst, PTY_U32, line) != 0)
                            return;
                        def_val(c, in->dst, PTY_U32, r->cur);
                        return;
                    }
                    // 32-bit PTX pointer copy (cvta.to.global.u32 and
                    // friends, or a plain mov.u32 of a 32-bit pointer):
                    // preserve the pointer home and the low word, so
                    // later address arithmetic keeps the binding (and
                    // BDA lowering keeps the full 64-bit base) while
                    // value uses still see the low word.
                    ptx_ty_t dty = PTY_U32;
                    if (check_dst(c, in->dst, dty, line) != 0)
                        return;
                    reg_t *dr = find_reg(c, in->dst);
                    if (dr) {
                        dr->defined = 1;
                        dr->decl_ty = dty;
                        dr->cur = r->cur;
                        dr->cur_ty = r->cur ? PTY_U32 : dty;
                        dr->is_ptr = 1;
                        dr->binding = r->binding;
                        dr->off = r->off;
                        dr->shvar = r->shvar;
                        dr->glvar = r->glvar;
                        dr->paddr = r->paddr;
                        if (r->binding >= 0)
                            dr->is_tex = r->is_tex;
                    }
                    return;
                }
            } else if (in->space == PSP_GLOBAL &&
                       (in->ty == PTY_U32 || in->ty == PTY_U64)) {
                // cvta.to.global from a plain integer value (32-bit PTX
                // loads pointer fields as u32 values from structs/params).
                // Treat the value as a raw global device address (BDA/raw
                // pointer form, binding=-3) so it can be used for ld/st.
                uint32_t off = r->cur ? r->cur : const_u64(c, 0);
                if (in->ty == PTY_U32 && r->cur) {
                    uint32_t wide = nid(c);
                    put(c, b, (4u << 16) | S_UConvert);
                    put(c, b, c->t_u64);
                    put(c, b, wide);
                    put(c, b, r->cur);
                    off = wide;
                }
                def_ptr(c, in->dst, -3, off);
                return;
            }
        }
        if (in->src[0].kind == OPD_PARAM &&
            (in->ty == PTY_U64 || in->ty == PTY_U32)) {
            // Symbolic shared-variable address (mov.u64/u32 %rd, svar).
            int sv = find_shvar(c, in->src[0].param);
            if (sv >= 0) {
                ptx_ty_t dty = in->ty == PTY_U64 ? PTY_U64 : PTY_U32;
                if (check_dst(c, in->dst, dty, line) != 0)
                    return;
                reg_t *r = find_reg(c, in->dst);
                if (r) {
                    r->defined = 1;
                    r->cur = 0;
                    r->cur_ty = dty;
                    r->decl_ty = dty;
                    r->is_ptr = 1;
                    r->binding = -1;
                    r->off = const_u64(c, 0);
                    r->shvar = sv;
                    r->glvar = -1;
                    r->paddr = -1;
                    r->dyn_home = 0;
                    r->is_tex = 0;
                }
                return;
            }
            // Hidden module-scope .global/.const symbol: load device-backed
            // buffer address (BDA pointer param supplied by driver/runtime).
            {
                cparam_t *symcp = find_cparam(c, in->src[0].param);
                if (symcp && symcp->is_hidden_symbol) {
                    ptx_ty_t dty = (in->ty == PTY_U64) ? PTY_U64 : PTY_U32;
                    if (check_dst(c, in->dst, dty, line) != 0)
                        return;
                    if (getenv("ACUDA_DEBUG_SYMBOLS"))
                        fprintf(stderr, "ACuda: hidden symbol mov %s -> bind=%d bda=%d bda_word=%d\n",
                                in->src[0].param, symcp->binding, c->use_bda,
                                (symcp->binding >= 0) ? c->bda_word[symcp->binding] : -1);
                    if (c->use_bda && symcp->binding >= 0 && symcp->kind == 1 &&
                        !symcp->is_tex && strncmp(symcp->name, "__acuda_host_", 13) != 0) {
                        if (getenv("ACUDA_DEBUG_SYMBOLS"))
                            fprintf(stderr, "ACuda: hidden symbol mov BDA path %s -> word=%u\n",
                                    in->src[0].param, (uint32_t)c->bda_word[symcp->binding]);
                        uint32_t lo = nid(c), hi = nid(c);
                        emit_push_load(c, b, &lo, PTY_U32,
                                       (uint32_t)c->bda_word[symcp->binding], line);
                        emit_push_load(c, b, &hi, PTY_U32,
                                       (uint32_t)c->bda_word[symcp->binding] + 1, line);
                        uint32_t lo64 = nid(c), hi64 = nid(c), sh = nid(c);
                        put(c, b, (4u << 16) | S_UConvert);
                        put(c, b, c->t_u64); put(c, b, lo64); put(c, b, lo);
                        put(c, b, (4u << 16) | S_UConvert);
                        put(c, b, c->t_u64); put(c, b, hi64); put(c, b, hi);
                        put(c, b, (5u << 16) | S_Shl);
                        put(c, b, c->t_u64); put(c, b, sh); put(c, b, hi64);
                        put(c, b, const_u64(c, 32));
                        uint32_t addr = nid(c);
                        emit_arith(c, b, S_Or, c->t_u64, addr, sh, lo64);
                        if (dty == PTY_U32) {
                            def_ptr(c, in->dst, -3, addr);
                            reg_t *dr = find_reg(c, in->dst);
                            if (dr) { dr->cur = lo; dr->cur_ty = PTY_U32; }
                        } else {
                            def_ptr(c, in->dst, -3, addr);
                        }
                        return;
                    }
                    if (getenv("ACUDA_DEBUG_SYMBOLS"))
                        fprintf(stderr, "ACuda: hidden symbol mov DESC fallback %s -> bind=%d\n",
                                in->src[0].param, symcp->binding);
                    def_ptr(c, in->dst, symcp->binding, const_u64(c, 0));
                    return;
                }
                if (getenv("ACUDA_DEBUG_SYMBOLS") && symcp) {
                    fprintf(stderr, "ACuda: cparam %s found but hidden=%d kind=%d bind=%d\n",
                            in->src[0].param, symcp->is_hidden_symbol, symcp->kind, symcp->binding);
                }
            }
            // Symbolic .global image address (mov.u64 %rd, gvar).
            int gv = find_global(c, in->src[0].param);
            if (gv >= 0) {
                if (check_dst(c, in->dst, PTY_U64, line) != 0)
                    return;
                def_global(c, in->dst, gv, const_u64(c, 0));
                return;
            }
            // Symbolic .local variable address (mov.u64/u32 %rd, lvar).
            int lv = find_local(c, in->src[0].param);
            if (lv >= 0) {
                if (in->pred[0]) {
                    failf(c, line, "predicated mov of variable address unsupported");
                    return;
                }
                ptx_ty_t dty = (in->ty == PTY_U64) ? PTY_U64 : PTY_U32;
                if (check_dst(c, in->dst, dty, line) != 0)
                    return;
                // NOTE: no SPIR-V is emitted here. The address is the
                // symbolic pair (lovar, byte offset in off); the offset
                // starts at zero. Storing the AccessChain pointer in cur
                // would poison later u32 address arithmetic with a
                // pointer-typed operand (invalid OpIAdd).
                reg_t *dr = find_reg(c, in->dst);
                if (dr) {
                    dr->defined = 1;
                    dr->cur = 0;
                    dr->cur_ty = dty;
                    dr->decl_ty = dty;
                    dr->is_ptr = 1;
                    dr->binding = -2;
                    dr->lovar = lv;
                    dr->shvar = -1;
                    dr->glvar = -1;
                    dr->off = const_u64(c, 0);
                }
                return;
            }
            // Opaque texture parameter value (mov.u64 %rd, texref):
            // a pointer to the buffer-backed descriptor.
            {
                cparam_t *tcp = find_cparam(c, in->src[0].param);
                if (tcp && tcp->is_tex) {
                    if (check_dst(c, in->dst, PTY_U64, line) != 0)
                        return;
                    def_ptr(c, in->dst, tcp->binding, const_u64(c, 0));
                    reg_t *tr = find_reg(c, in->dst);
                    if (tr)
                        tr->is_tex = tcp->is_tex;
                    return;
                }
            }
            // Address of a kernel parameter (mov.u64 %rd, param). The
            // parameter value stays in push constants; the address is a
            // symbolic token readable with ld.param [%rd] (parameter
            // storage is otherwise immutable, so no cell is needed).
            {
                int pidx = -1;
                for (int i = 0; i < c->nfnparams; i++) {
                    if (!c->fnparams[i].is_retval &&
                        !c->fnparams[i].is_hidden_symbol &&
                        strcmp(c->fnparams[i].name, in->src[0].param) == 0) {
                        pidx = i;
                        break;
                    }
                }
                if (pidx >= 0) {
                    if (in->pred[0]) {
                        failf(c, line,
                              "predicated mov of a parameter address unsupported");
                        return;
                    }
                    if (check_dst(c, in->dst, PTY_U64, line) != 0)
                        return;
                    if (getenv("ACUDA_DEBUG_SYMBOLS"))
                        fprintf(stderr, "ACuda: paddr mov %s -> pidx=%d hidden=%d\n",
                                in->src[0].param, pidx, c->fnparams[pidx].is_hidden_symbol);
                    def_paddr(c, in->dst, pidx);
                    return;
                }
            }
            // Symbolic .const address token (mov.u64 %rd, csym). Consts
            // have no storage (loads materialize immediates), so this is
            // an identity token, not a dereferenceable address.
            int ci = find_const(c, in->src[0].param);
            if (ci >= 0) {
                // 32-bit PTX (CUDA 4.x) takes const addresses in .u32
                // registers.  The bit-63 symbolic tag does not fit and is
                // not needed: ld.const masks it off, leaving the raw byte
                // offset into the flattened const bank.
                reg_t *dr = find_reg(c, in->dst);
                if (dr && dr->decl_ty == PTY_U32) {
                    def_val(c, in->dst, PTY_U32,
                            const_u32(c, (uint32_t)c->consts[ci].bank_off));
                    return;
                }
                if (check_dst(c, in->dst, PTY_U64, line) != 0)
                    return;
                def_val(c, in->dst, PTY_U64,
                        const_u64(c, PTX_CONST_ADDR_BASE |
                            (uint64_t)c->consts[ci].bank_off));
                return;
            }
            // Else: not a module variable; resolve() rejects OPD_PARAM
            // honestly below (params are only readable via ld.param).
        }
        if (in->src[0].kind == OPD_PARAM && in->ty == PTY_U32) {
            // 32-bit shared-variable base (ZLUDA corpus): tracks the
            // variable with a zero offset. Arithmetic on it stays an
            // honest error (pointer used as scalar value); predicated
            // takes match the u64 rule above (compile-time bindings).
            if (in->pred[0]) {
                failf(c, line, "predicated mov of variable address unsupported");
                return;
            }
            int sv = find_shvar(c, in->src[0].param);
            if (sv >= 0) {
                def_shared(c, in->dst, sv, const_u64(c, 0));
                return;
            }
            int gv = find_global(c, in->src[0].param);
            if (gv >= 0) {
                def_global(c, in->dst, gv, const_u64(c, 0));
                return;
            }
            int lv = find_local(c, in->src[0].param);
            if (lv >= 0) {
                // Symbolic pair (lovar, zero byte offset); see the u64
                // arm above (a stored AccessChain pointer here would
                // poison u32 address arithmetic with an invalid OpIAdd).
                reg_t *dr = find_reg(c, in->dst);
                if (dr) {
                    dr->defined = 1;
                    dr->cur = 0;
                    dr->cur_ty = PTY_U32;
                    dr->is_ptr = 1;
                    dr->binding = -2;
                    dr->lovar = lv;
                    dr->shvar = -1;
                    dr->glvar = -1;
                    dr->off = const_u64(c, 0);
                }
                return;
            }
            // Else: falls through to resolve(), which rejects OPD_PARAM.
        }
        if (in->src[0].kind == OPD_PARAM) {
            // Function address (mov.<ty> %rd, <funcname>): compile-time
            // callee home value. Only .u64/.f64-named moves are meaningful
            // here (pointer-sized); anything else stays an honest error.
            if (in->ty == PTY_U64 || in->ty == PTY_F64) {
                int ffound = -1;
                for (int i = 0; i < c->nfns; i++) {
                    if (strcmp(c->fns[i].name, in->src[0].param) == 0) {
                        ffound = i;
                        break;
                    }
                }
                if (ffound >= 0) {
                    uint32_t fp = const_u64(c, 0x1000u + (uint32_t)ffound);
                    def_val(c, in->dst, in->ty, fp);
                    return;
                }
            }
        }
        a = resolve(c, b, &in->src[0], in->ty, line);
        if (c->failed)
            return;
        {
            // mov.b32/.b64 between integer and float spellings is a
            // bit reinterpretation. resolve() returns the source's SSA
            // value, so the destination must be bitcast to its declared
            // type or later float consumers see an integer id.
            reg_t *mdr = find_reg(c, in->dst);
            ptx_ty_t dty = mdr ? mdr->decl_ty : in->ty;
            int samew = (in->ty == PTY_U32 && dty == PTY_F32) ||
                        (in->ty == PTY_F32 && dty == PTY_U32) ||
                        (in->ty == PTY_U64 && dty == PTY_F64) ||
                        (in->ty == PTY_F64 && dty == PTY_U64);
            if (samew) {
                id = nid(c);
                put(c, b, (4u << 16) | S_Bitcast);
                put(c, b, ty_of(c, dty));
                put(c, b, id);
                put(c, b, a.id);
                def_val(c, in->dst, dty, id);
                return;
            }
        }
        def_val(c, in->dst, in->ty, a.id);
        return;
    // Pointer arithmetic (including u32 shared/global offsets) is handled
    // centrally by emit_arith_op, which preserves the pointer home and
    // widens a 32-bit index before adding it to the 64-bit byte offset.
    case POP_ADD:
    case POP_SUB:
    case POP_MULLO:
    case POP_MADLO:
    case POP_FMA:
    case POP_MULWIDE:
    case POP_MADWIDE:
    case POP_MULHI:
    case POP_DP4A:
    case POP_DP2A:
    case POP_DOT2:
    case POP_SAD:
    case POP_PRMT:
    case POP_GLSL:
    case POP_ADDSAT:
    case POP_COPYSIGN:
    case POP_MINNAN:
    case POP_MUL24:
    case POP_SHF:
    case POP_ABS:
    case POP_POPC:
    case POP_CLZ:
    case POP_BREV:
    case POP_BFE:
    case POP_BFI:
    case POP_BMSK:
    case POP_BFIND:
    case POP_CVTPACK8:
    case POP_CVT:
    case POP_CVTPACK:
    case POP_FP8_TO_F16X2:
    case POP_F32_TO_FP8X2:
    case POP_RND:
    case POP_SHR:
    case POP_SHL:
    case POP_VSHR:
    case POP_AND:
    case POP_OR:
    case POP_XOR:
    case POP_SELP:
    case POP_NOT:
    case POP_NEG:
    case POP_MINMAX:
    case POP_DIV:
    case POP_REM:
    case POP_SETP:
    case POP_SET:
    case POP_FNS:
    case POP_SZEXT:
    case POP_CLMAD:
    case POP_MAD24:
    case POP_TESTP:
    case POP_RCP:
    case POP_CNOT:
    case POP_LOP3:
    case POP_SLCT:
    case POP_PREFETCH:
    case POP_HINT:
        emit_arith_op(c, b, in);
        return;
    case POP_LD:
    case POP_CPASYNC:
    case POP_CPASYNC_BULK:
    case POP_ST:
        emit_mem_op(c, b, in);
        return;
    case POP_ATOM:
    case POP_ATOMCAS:
        emit_atom(c, b, in);
        return;
    case POP_SHFL:
    case POP_VOTE:
    case POP_REDUX:
    case POP_MATCH:
    case POP_MMA:
    case POP_LDMATRIX:
    case POP_BAR:
    case POP_BARRED:
    case POP_MEMBAR:
    case POP_NANOSLEEP:
    case POP_BARWARP:
    case POP_ISSPACEP:
    case POP_WMMA_LOAD:
    case POP_WMMA_STORE:
    case POP_WMMA_MMA:
        emit_group_op(c, b, in);
        return;
    case POP_TEX:
    case POP_TXQ:
    case POP_ISTYPEP:
    case POP_TLD4:
        emit_texture_op(c, b, in);
        return;
    case POP_SULD:
    case POP_SUST:
    case POP_SURED:
    case POP_SUQ:
        emit_surface_op(c, b, in);
        return;
    case POP_CVT_MICRO:
        emit_cvt_micro(c, b, in);
        return;
    case POP_BRKPT:
        if (c->uses_brkpt) {
            put(c, b, (4u << 16) | S_Line);
            put(c, b, c->v_file);
            put(c, b, (uint32_t)in->line);
            put(c, b, 1u);
        }
        return;
    case POP_ELECT:
        emit_elect_op(c, b, in);
        return;
    case POP_MBARRIER:
        emit_mbarrier_op(c, b, in);
        return;
    case POP_RED_ASYNC:
        emit_red_async(c, b, in);
        return;
    case POP_BRA:
    case POP_CBRA:
    case POP_RET:
    case POP_EXIT:
        failf(c, line, "control flow handled at block level");
        return;
    case POP_HOST_REQUEST:
        emit_host_request(c, b, in);
        return;
    case POP_HOST_REPLY:
        emit_host_reply(c, b, in);
        return;
    case POP_HOST_SPILL:
        emit_host_spill(c, b, in);
        return;
    case POP_HOST_RESTORE:
        emit_host_restore(c, b, in);
        return;
    case POP_HOST_RESULT:
        emit_host_result(c, b, in);
        return;
    case POP_HOSTCALL:
        failf(c, line, "host call survived splitting (front-end bug)");
        return;
    case POP_CALL:
        failf(c, line, "call survived inlining (front-end bug)");
        return;
    case POP_BRX:
        failf(c, line, "brx.idx survived lowering (front-end bug)");
        return;
    }
    failf(c, line, "unhandled instruction");
}

