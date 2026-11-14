#include "instructions/memory/memory.h"

void emit_prefetch_hint_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    (void)c;
    (void)b;
    (void)in;
    /* Pure cache/scheduling hints: validated at parse time, no
     * observable result. */
}
