#ifndef ACUDA_COMPILER_MEMORY_H
#define ACUDA_COMPILER_MEMORY_H

#include "spirv_internal.h"

void emit_cp_async_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
int dyn_addr_reg(ctx_t *c, const ptx_instr_t *in, reg_t **out);
void emit_dyn_addr_mem(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in, reg_t *ar);
uint32_t fold_sh_imm(ctx_t *c, spv_blob_t *b, uint32_t off, int64_t imm);
void emit_ld_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_prefetch_hint_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_st_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);

#endif
