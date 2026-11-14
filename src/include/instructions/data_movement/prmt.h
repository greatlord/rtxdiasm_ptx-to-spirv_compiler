#ifndef ACUDA_COMPILER_DATA_MOVEMENT_PRMT_H
#define ACUDA_COMPILER_DATA_MOVEMENT_PRMT_H

#include "spirv_internal.h"

void emit_prmt_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);

#endif
