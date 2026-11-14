// Host-call preprocessing: split blocks at POP_HOSTCALL and add the
// internal state/request parameters used by the state-machine emitter.
//
// The actual control-flow flattening happens in spirv_hoststate.c: the
// function is emitted as a per-invocation PC loop, so calls may sit in
// branches, loops, and divergent paths.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ptx.h"
#include "acuda/compiler.h"
#include "parse/parse_internal.h"

static int host_fail(char *err, size_t errlen, int line, const char *msg) {
    if (err && errlen) {
        if (line > 0)
            snprintf(err, errlen, "line %d: %s", line, msg);
        else
            snprintf(err, errlen, "%s", msg);
    }
    return -1;
}

static int host_add_internal_param(ptx_func_t *fn, const char *name) {
    if (fn->nparams == fn->params_cap) {
        size_t ncap = fn->params_cap ? fn->params_cap * 2 : 8;
        ptx_param_t *np = realloc(fn->params, sizeof(*np) * ncap);
        if (!np)
            return -1;
        fn->params = np;
        fn->params_cap = ncap;
    }
    ptx_param_t *p = &fn->params[fn->nparams++];
    memset(p, 0, sizeof(*p));
    snprintf(p->name, sizeof(p->name), "%s", name);
    p->ty = PTY_U64;
    p->is_internal = 1;
    return 0;
}


// Split every block at each host call. The call stays at the end of the
// first half; the remainder is inserted immediately after it as an
// unlabeled continuation block, preserving fallthrough order.
static int host_split_blocks(ptx_func_t *fn, char *err, size_t errlen) {
    size_t bi = 0;
    while (bi < fn->nblocks) {
        ptx_block_t *blk = &fn->blocks[bi];
        size_t ci = (size_t)-1;
        for (size_t i = 0; i < blk->count; i++) {
            if (blk->instrs[i].op == POP_HOSTCALL) {
                ci = i;
                break;
            }
        }
        if (ci == (size_t)-1) {
            bi++;
            continue;
        }
        ptx_instr_t *call = &blk->instrs[ci];
        if (ci + 1 < blk->count) {
            size_t tail = blk->count - (ci + 1);
            ptx_block_t nb;
            memset(&nb, 0, sizeof(nb));
            nb.cap = tail;
            nb.instrs = malloc(sizeof(*nb.instrs) * tail);
            if (!nb.instrs)
                return host_fail(err, errlen, call->line, "out of memory");
            memcpy(nb.instrs, blk->instrs + ci + 1,
                   sizeof(*nb.instrs) * tail);
            nb.count = tail;
            if (fn->nblocks == fn->blocks_cap) {
                size_t ncap = fn->blocks_cap ? fn->blocks_cap * 2 : 8;
                ptx_block_t *nbs = realloc(fn->blocks,
                                           sizeof(*nbs) * ncap);
                if (!nbs) {
                    free(nb.instrs);
                    return host_fail(err, errlen, call->line,
                                     "out of memory");
                }
                fn->blocks = nbs;
                fn->blocks_cap = ncap;
            }
            blk = &fn->blocks[bi];
            call = &blk->instrs[ci];
            memmove(&fn->blocks[bi + 2], &fn->blocks[bi + 1],
                    sizeof(*fn->blocks) * (fn->nblocks - (bi + 1)));
            fn->blocks[bi + 1] = nb;
            fn->nblocks++;
            blk = &fn->blocks[bi];
            call = &blk->instrs[ci];
            blk->count = ci + 1;
            call->host_cont = (int)(bi + 1);
            // Scan the continuation next; it may hold another call.
            bi++;
            continue;
        }
        // Call ends the block: its continuation is the following block
        // (which may carry a label), or a fresh empty block at EOF.
        size_t cont = bi + 1;
        if (cont >= fn->nblocks) {
            if (fn->nblocks == fn->blocks_cap) {
                size_t ncap = fn->blocks_cap ? fn->blocks_cap * 2 : 8;
                ptx_block_t *nbs = realloc(fn->blocks,
                                           sizeof(*nbs) * ncap);
                if (!nbs)
                    return host_fail(err, errlen, call->line,
                                     "out of memory");
                fn->blocks = nbs;
                fn->blocks_cap = ncap;
            }
            memset(&fn->blocks[fn->nblocks], 0, sizeof(*fn->blocks));
            fn->blocks[fn->nblocks].cap = 1;
            fn->blocks[fn->nblocks].instrs =
                calloc(1, sizeof(*fn->blocks[fn->nblocks].instrs));
            if (!fn->blocks[fn->nblocks].instrs)
                return host_fail(err, errlen, call->line, "out of memory");
            cont = fn->nblocks;
            fn->nblocks++;
        }
        fn->blocks[bi].instrs[ci].host_cont = (int)cont;
        bi++;
    }
    return 0;
}


// Returns 1 when the function contains host calls (and has been
// prepared for the state-machine emitter), 0 when it does not, -1 on a
// clean unsupported error.
int ptx_hostcall_prepare(ptx_unit_t *u, ptx_func_t *fn, char *err,
                         size_t errlen) {
    size_t ncalls = 0;
    int first_line = 0;
    for (size_t b = 0; b < fn->nblocks; b++)
        for (size_t i = 0; i < fn->blocks[b].count; i++)
            if (fn->blocks[b].instrs[i].op == POP_HOSTCALL) {
                if (!first_line)
                    first_line = fn->blocks[b].instrs[i].line;
                int cid = fn->blocks[b].instrs[i].host_call_id;
                if (cid < 0 || cid >= ACUDA_MAX_HOST_CALLS)
                    return host_fail(err, errlen,
                                     fn->blocks[b].instrs[i].line,
                                     "too many distinct host-call ids (max 8)");
                ncalls++;
            }
    if (ncalls == 0)
        return 0;
    if (fn->nrparams > 0)
        return host_fail(err, errlen, first_line,
                         "host call in a .reg-parameter function unsupported");
    if (ncalls > ACUDA_MAX_HOST_CALLS)
        return host_fail(err, errlen, first_line,
                         "too many host calls in one function");

    if (host_add_internal_param(fn, "__acuda_host_state") != 0 ||
        host_add_internal_param(fn, "__acuda_host_req") != 0)
        return host_fail(err, errlen, first_line, "out of memory");

    if (host_split_blocks(fn, err, errlen) != 0)
        return -1;

    // State layout: header (pc), one 16-byte slot per register, then
    // one byte-addressed return area per call id.
    size_t stride = ACUDA_HOST_STATE_HEADER +
                    (size_t)fn->nregs * ACUDA_HOST_STATE_REG_STRIDE;
    for (size_t i = 0; i < u->nextfuncs && i < ACUDA_MAX_HOST_CALLS; i++) {
        fn->host_ret_off[i] = stride;
        size_t rb = (u->extfuncs[i].ret_bytes + 7u) & ~(size_t)7u;
        stride += rb;
    }
    fn->host_ret_total = (uint32_t)(stride - (ACUDA_HOST_STATE_HEADER +
                              (size_t)fn->nregs * ACUDA_HOST_STATE_REG_STRIDE));
    fn->host_local_base = (uint32_t)stride;
    for (size_t li = 0; li < fn->nlocals && li < 32; li++) {
        fn->host_local_off[li] = (uint32_t)stride;
        stride += (fn->locals[li].bytes + 3u) & ~(size_t)3u;
    }
    fn->host_local_total = (uint32_t)(stride - fn->host_local_base);
    size_t shstride = 0;
    for (size_t si = 0; si < u->nshvars && si < 32; si++) {
        if (u->shvars[si].is_extern)
            continue;
        fn->host_shared_off[si] = (uint32_t)shstride;
        shstride += (u->shvars[si].bytes + 3u) & ~(size_t)3u;
    }
    // Dynamic .shared aliases one runtime-sized array after the static
    // shared area; its size is supplied at launch.
    for (size_t si = 0; si < u->nshvars && si < 32; si++)
        if (u->shvars[si].is_extern)
            fn->host_shared_off[si] = (uint32_t)shstride;
    fn->host_shared_stride = (uint32_t)shstride;
    fn->host_global_base = (uint32_t)stride;
    for (size_t gi = 0; gi < u->nglobals && gi < ACUDA_MAX_GLOBALS; gi++) {
        fn->host_global_off[gi] = (uint32_t)stride;
        stride += (u->globals[gi].nbytes + 3u) & ~(size_t)3u;
    }
    fn->host_global_total = (uint32_t)(stride - fn->host_global_base);
    fn->host_state_stride = (uint32_t)stride;
    fn->host_calls = 1;
    return 1;
}
