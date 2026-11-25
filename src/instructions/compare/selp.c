#include "instructions/compare/compare.h"

void emit_selp_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2;
    uint32_t id;
        // A select that merges a pointer home with a plain word is the
        // parser's desugaring of a predicated address computation
        // (`@p add %r, %off, %ptr` -> `add __ps_N` + `selp %r, __ps_N, %r, p`).
        // Keep the pointer arm's home so a later zero-extension still
        // addresses the same buffer; the other arm only supplies the
        // numeric word when the pointer arm carries one.
        if (in->nsrc >= 3 && in->ty == PTY_U32) {
            reg_t *r0 = in->src[0].kind == OPD_REG
                            ? find_reg(c, in->src[0].reg) : NULL;
            reg_t *r1 = in->src[1].kind == OPD_REG
                            ? find_reg(c, in->src[1].reg) : NULL;
            int p0 = r0 && r0->defined && r0->is_ptr;
            int p1 = r1 && r1->defined && r1->is_ptr;
            if (p0 != p1) {
                reg_t *ptr = p0 ? r0 : r1;
                const ptx_opd_t *other = p0 ? &in->src[1] : &in->src[0];
                reg_t *pr = in->src[2].kind == OPD_REG
                                ? find_reg(c, in->src[2].reg) : NULL;
                if (pr && pr->defined && pr->decl_ty == PTY_PRED) {
                    uint32_t word = 0;
                    if (ptr->cur && ptr->cur_ty == PTY_U32) {
                        val_t ov = resolve(c, b, other, PTY_U32, line);
                        if (c->failed)
                            return;
                        word = nid(c);
                        put(c, b, (6u << 16) | S_Select);
                        put(c, b, c->t_u32);
                        put(c, b, word);
                        put(c, b, pr->cur);
                        put(c, b, p0 ? ptr->cur : ov.id);
                        put(c, b, p0 ? ov.id : ptr->cur);
                    }
                    def_ptr(c, in->dst, ptr->binding, ptr->off);
                    reg_t *dr = find_reg(c, in->dst);
                    if (dr) {
                        dr->is_tex = ptr->binding >= 0 ? ptr->is_tex : 0;
                        if (word) {
                            dr->cur = word;
                            dr->cur_ty = PTY_U32;
                        }
                    }
                    return;
                }
            }
        }
        // selp.type dst, a, b, pred -> OpSelect (true picks a).
        if (check_dst(c, in->dst, in->ty, line) != 0)
            return;
        a = resolve(c, b, &in->src[0], in->ty, line);
        c2 = resolve(c, b, &in->src[1], in->ty, line);
        if (c->failed)
            return;
        reg_t *pr = find_reg(c, in->src[2].reg);
        if (!pr || pr->decl_ty != PTY_PRED) {
            failf(c, line, "selp predicate must be a .pred register");
            return;
        }
        if (!pr->defined) {
            failf(c, line, "selp predicate has no value");
            return;
        }
        id = nid(c);
        put(c, b, (6u << 16) | S_Select);
        put(c, b, ty_of(c, in->ty));
        put(c, b, id);
        put(c, b, pr->cur);
        put(c, b, a.id);
        put(c, b, c2.id);
        def_val(c, in->dst, in->ty, id);
        return;
    }
