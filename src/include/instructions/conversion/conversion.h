#ifndef ACUDA_COMPILER_CONVERSION_H
#define ACUDA_COMPILER_CONVERSION_H

#include "spirv_internal.h"

void emit_cvt_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_cvtpack_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_cvtpack8_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_fp8_to_f16x2_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_f32_to_fp8x2_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_rnd_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);

#endif
