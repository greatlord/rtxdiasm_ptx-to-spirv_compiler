#include "instructions/compare/compare.h"

void emit_selp_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2;
    uint32_t id;
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
