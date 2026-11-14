#ifndef ACUDA_COMPILER_COMPARE_H
#define ACUDA_COMPILER_COMPARE_H

#include "spirv_internal.h"

void emit_selp_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_set_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_setp_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_slct_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_testp_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);

#endif
