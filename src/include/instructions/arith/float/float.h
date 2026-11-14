#ifndef ACUDA_COMPILER_ARITH_FLOAT_H
#define ACUDA_COMPILER_ARITH_FLOAT_H

#include "spirv_internal.h"

void emit_copysign_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_dot2_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_glsl_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_minnan_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_rcp_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);

#endif
