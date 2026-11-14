// Memory operations: ld/st (parameter, shared, global image, generic)
// and cp.async.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "spirv_internal.h"
#include "instructions/memory/memory.h"
#include "instructions/memory/memory.h"
#include "instructions/memory/memory.h"
#include "instructions/memory/memory.h"

// A plain 64-bit scalar used as a global/generic address is treated
// as a runtime pointer token (see ptr_token_u64): high 32 bits select
// the descriptor binding, low 32 bits are the byte offset.  Emit a
// switch over the possible bindings and dispatch the load/store.

// Fold a shared byte offset back into the 64-bit pointer for access
// paths that cannot express it in the access index.

void emit_mem_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    reg_t *dynr = NULL;
    if (dyn_addr_reg(c, in, &dynr)) {
        emit_dyn_addr_mem(c, b, in, dynr);
        return;
    }
    switch (in->op) {
    case POP_LD:
        emit_ld_op(c, b, in);
        return;
    case POP_CPASYNC:
    case POP_CPASYNC_BULK:
        emit_cp_async_op(c, b, in);
        return;
    case POP_ST:
        emit_st_op(c, b, in);
        return;
    default:
        failf(c, line, "unhandled memory op");
        return;
    }
}
