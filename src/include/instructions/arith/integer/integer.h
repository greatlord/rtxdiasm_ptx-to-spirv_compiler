#ifndef ACUDA_COMPILER_ARITH_INTEGER_H
#define ACUDA_COMPILER_ARITH_INTEGER_H

#include "spirv_internal.h"

void emit_addsat_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_fns_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_szext_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_popc_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_clz_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_brev_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_bfe_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_bfi_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_bmsk_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_bfind_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_clmad_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_dp2a_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_dp4a_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_mad24_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_madwide_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_mul24_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_mulhi_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_mulwide_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_sad_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);

#endif
