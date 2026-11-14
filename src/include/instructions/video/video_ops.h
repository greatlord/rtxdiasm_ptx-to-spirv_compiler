#ifndef ACUDA_COMPILER_VIDEO_OPS_H
#define ACUDA_COMPILER_VIDEO_OPS_H

#include "spirv_internal.h"

void emit_vshr_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);

#endif
