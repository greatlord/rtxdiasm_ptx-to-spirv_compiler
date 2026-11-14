#ifndef ACUDA_COMPILER_ARITH_H
#define ACUDA_COMPILER_ARITH_H

#include "spirv_internal.h"

void emit_add_sub_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_div_rem_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_fma_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_madlo_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_minmax_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_mullo_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_abs_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_neg_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);

#endif
