// CFG utility helpers.
#include <stdint.h>
#include <stdlib.h>
#include "spirv_internal.h"

void cfg_free(cfg_t *cfg) {
    free(cfg->selfloop);
    free(cfg->fwdtgt);
    free(cfg->flow_snap);
    free(cfg->flow_live);
    cfg->selfloop = NULL;
    cfg->fwdtgt = NULL;
    cfg->flow_snap = NULL;
    cfg->flow_live = NULL;
}

int block_terminator(const ptx_block_t *b) {
    if (b->count == 0)
        return -1;
    ptx_op_t op = b->instrs[b->count - 1].op;
    if (op == POP_BRA || op == POP_CBRA || op == POP_RET || op == POP_EXIT)
        return (int)op;
    return -1;
}
