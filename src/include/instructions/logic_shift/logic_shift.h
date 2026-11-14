#ifndef ACUDA_COMPILER_LOGIC_SHIFT_H
#define ACUDA_COMPILER_LOGIC_SHIFT_H

#include "spirv_internal.h"

void emit_logic_bitwise(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_not_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_shf_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_cnot_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_lop3_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_shift_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);

#endif
