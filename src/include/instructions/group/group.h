#ifndef ACUDA_COMPILER_GROUP_H
#define ACUDA_COMPILER_GROUP_H

#include "spirv_internal.h"

void emit_barrier_group_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_isspacep_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_ldmatrix_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_match_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_mma_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_redux_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_shfl_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_vote_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_wmma_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);

#endif
