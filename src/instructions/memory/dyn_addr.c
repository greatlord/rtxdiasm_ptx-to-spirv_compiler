#include "instructions/memory/memory.h"

#include <string.h>

int dyn_addr_reg(ctx_t *c, const ptx_instr_t *in, reg_t **out) {
    if (in->op != POP_LD && in->op != POP_ST)
        return 0;
    if (in->space != PSP_GLOBAL && in->space != PSP_GENERIC)
        return 0;
    if (c->use_bda)
        return 0; // plain u64 addresses are physical, resolved directly
    if (in->src[0].kind != OPD_ADDR || !in->src[0].addr_reg[0])
        return 0;
    reg_t *r = find_reg(c, in->src[0].addr_reg);
    if (!r || !r->defined || r->is_ptr || r->decl_ty != PTY_U64)
        return 0;
    *out = r;
    return 1;
}

void emit_dyn_addr_mem(ctx_t *c, spv_blob_t *b,
                              const ptx_instr_t *in, reg_t *ar) {
    int line = in->line;
    int is_load = in->op == POP_LD;
    if (c->nbindings <= 0) {
        if (getenv("ACUDA_DYNAMIC_UNDEF")) {
            ACUDA_DEBUG_PRINT("acuda: dynamic undef path\n");
            // Diagnostic-only: legacy SASS lifting can produce an address
            // that is not tied to a pointer parameter.  Let the kernel
            // compile with an undefined result so the caller can see
            // whether module loading was the only blocker.
            if (is_load) {
                uint32_t undef = nid(c);
                put(c, b, (3u << 16) | 1); // OpUndef
                put(c, b, ty_of(c, in->ty));
                put(c, b, undef);
                def_val(c, in->dst, in->ty, undef);
            }
            return;
        }
        failf(c, line, "dynamic address has no pointer homes");
        return;
    }
    uint32_t tok = ar->cur;
    if (in->src[0].addr_imm) {
        uint32_t t = nid(c);
        emit_arith(c, b, S_IAdd, c->t_u64, t, tok,
                   const_u64(c, (uint64_t)in->src[0].addr_imm));
        tok = t;
    }
    uint32_t tag64 = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, tag64, tok, const_u64(c, 32));
    uint32_t tag = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u32);
    put(c, b, tag);
    put(c, b, tag64);
    uint32_t off = nid(c);
    emit_arith(c, b, S_And, c->t_u64, off, tok,
               const_u64(c, 0xffffffffull));
    if (c->failed)
        return;

    int nb = c->nbindings;
    uint32_t merge = nid(c), deflabel = nid(c);
    uint32_t *labels = malloc(sizeof(uint32_t) * (size_t)nb);
    uint32_t *vals = malloc(sizeof(uint32_t) * (size_t)nb);
    size_t rb = sizeof(*c->regs) * (size_t)c->nregs;
    reg_t *rsave = malloc(rb ? rb : 1);
    if (!labels || !vals || !rsave) {
        free(labels);
        free(vals);
        free(rsave);
        fail(c, "out of memory");
        return;
    }
    memcpy(rsave, c->regs, rb ? rb : 1);
    put(c, b, (3u << 16) | S_SelectionMerge);
    put(c, b, merge);
    put(c, b, 0);
    put(c, b, (uint32_t)((3u + 2u * (uint32_t)nb) << 16) | S_Switch);
    put(c, b, tag);
    put(c, b, deflabel);
    for (int k = 0; k < nb; k++) {
        labels[k] = nid(c);
        put(c, b, (uint32_t)k);
        put(c, b, labels[k]);
    }
    for (int k = 0; k < nb; k++) {
        put(c, b, (2u << 16) | S_Label);
        put(c, b, labels[k]);
        memcpy(c->regs, rsave, rb ? rb : 1);
        reg_t *a2 = find_reg(c, in->src[0].addr_reg);
        a2->is_ptr = 1;
        a2->binding = k;
        a2->off = off;
        a2->shvar = -1;
        a2->glvar = -1;
        a2->paddr = -1;
        a2->dyn_home = 0;
        a2->cur = 0;
        emit_mem_op(c, b, in);
        if (c->failed)
            goto out;
        if (is_load) {
            reg_t *dr = find_reg(c, in->dst);
            if (!dr || !dr->defined || dr->is_ptr) {
                failf(c, line, "dynamic load produced no scalar value");
                goto out;
            }
            vals[k] = dr->cur;
        }
        put(c, b, (2u << 16) | S_Branch);
        put(c, b, merge);
    }
    put(c, b, (2u << 16) | S_Label);
    put(c, b, deflabel);
    uint32_t undef = 0;
    if (is_load) {
        undef = nid(c);
        put(c, b, (3u << 16) | 1); // OpUndef
        put(c, b, ty_of(c, in->ty));
        put(c, b, undef);
    }
    put(c, b, (2u << 16) | S_Branch);
    put(c, b, merge);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, merge);
    memcpy(c->regs, rsave, rb ? rb : 1);
    if (is_load) {
        uint32_t phi = nid(c);
        put(c, b, (uint32_t)((3u + 2u * (uint32_t)(nb + 1)) << 16) | S_Phi);
        put(c, b, ty_of(c, in->ty));
        put(c, b, phi);
        for (int k = 0; k < nb; k++) {
            put(c, b, vals[k]);
            put(c, b, labels[k]);
        }
        put(c, b, undef);
        put(c, b, deflabel);
        def_val(c, in->dst, in->ty, phi);
    }
out:
    free(labels);
    free(vals);
    free(rsave);
}
