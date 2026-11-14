#include "instructions/memory/memory.h"

uint32_t fold_sh_imm(ctx_t *c, spv_blob_t *b, uint32_t off,
                            int64_t imm) {
    if (imm == 0)
        return off;
    uint32_t t = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, t, off, const_u64(c, (uint64_t)imm));
    return t;
}
