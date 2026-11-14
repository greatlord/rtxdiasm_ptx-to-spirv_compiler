// Warp/CTA group operations: shfl, vote, redux, match, mma, ldmatrix,
// barriers and nanosleep.
#include <stdint.h>
#include <string.h>
#include "spirv_internal.h"
#include "instructions/group/group.h"
#include "instructions/group/group.h"
#include "instructions/group/group.h"
#include "instructions/group/group.h"
#include "instructions/group/group.h"
#include "instructions/group/group.h"
#include "instructions/group/group.h"
#include "instructions/group/group.h"
#include "instructions/group/group.h"


// Atomic add on a software-barrier counter (release so writes before a
// barrier arrival are visible to the threads that observe the count).
uint32_t swbar_add(ctx_t *c, spv_blob_t *b, uint32_t ptr,
                          uint32_t val) {
    uint32_t scope = const_u32(c, 2); // Workgroup
    uint32_t sem = const_u32(c, 0x108u); // AcquireRelease|WorkgroupMemory
    uint32_t id = nid(c);
    put(c, b, (7u << 16) | S_AtomicIAdd);
    put(c, b, c->t_u32);
    put(c, b, id);
    put(c, b, ptr);
    put(c, b, scope);
    put(c, b, sem);
    put(c, b, val);
    return id;
}

// Counted barrier wait. The counter is monotonic: each arrival adds one,
// and a waiter blocks until at least `count` arrivals have happened
// since its own round began. Using (cur - my) >= (count - my % count)
// keeps the comparison correct across u32 wrap and needs no counter
// reset, so a bar.arrive that immediately arrives again cannot lose an
// update. Participating threads must execute this in warp-uniform
// groups (PTX's own requirement); mismatched counts are UB as in CUDA.
void swbar_wait(ctx_t *c, spv_blob_t *b, uint32_t ptr,
                       uint32_t count) {
    uint32_t my = swbar_add(c, b, ptr, const_u32(c, 1));
    uint32_t mod = nid(c);
    emit_arith(c, b, S_UMod, c->t_u32, mod, my, count);
    uint32_t rem = nid(c);
    emit_arith(c, b, S_ISub, c->t_u32, rem, count, mod);
    uint32_t header = nid(c), merge = nid(c), cont = nid(c);
    emit_branch(c, b, header);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, header);
    uint32_t scope = const_u32(c, 2);
    uint32_t sem = const_u32(c, 0x102u); // Acquire|WorkgroupMemory
    uint32_t cur = nid(c);
    put(c, b, (6u << 16) | S_AtomicLoad);
    put(c, b, c->t_u32);
    put(c, b, cur);
    put(c, b, ptr);
    put(c, b, scope);
    put(c, b, sem);
    uint32_t diff = nid(c);
    emit_arith(c, b, S_ISub, c->t_u32, diff, cur, my);
    uint32_t ok = nid(c);
    put(c, b, (5u << 16) | S_UGreaterEqual);
    put(c, b, c->t_bool);
    put(c, b, ok);
    put(c, b, diff);
    put(c, b, rem);
    put(c, b, (4u << 16) | S_LoopMerge);
    put(c, b, merge);
    put(c, b, cont);
    put(c, b, 0);
    put(c, b, (4u << 16) | S_BranchCond);
    put(c, b, ok);
    put(c, b, merge);
    put(c, b, cont);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, cont);
    emit_branch(c, b, header);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, merge);
    // Acquire fence after the successful poll.
    put(c, b, (3u << 16) | S_MemoryBarrier);
    put(c, b, scope);
    put(c, b, sem);
}



void emit_group_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    (void)line;
    if (in->op == POP_BAR || in->op == POP_BARRED ||
        in->op == POP_BARWARP)
        c->uses_barrier = 1;
    switch (in->op) {
    case POP_ISSPACEP:
        emit_isspacep_op(c, b, in);
        return;
    case POP_WMMA_LOAD:
    case POP_WMMA_STORE:
    case POP_WMMA_MMA:
        emit_wmma_op(c, b, in);
        return;
    case POP_SHFL:
        emit_shfl_op(c, b, in);
        return;
    case POP_VOTE:
        emit_vote_op(c, b, in);
        return;
    case POP_REDUX:
        emit_redux_op(c, b, in);
        return;
    case POP_MATCH:
        emit_match_op(c, b, in);
        return;
    case POP_MMA:
        emit_mma_op(c, b, in);
        return;
    case POP_LDMATRIX:
        emit_ldmatrix_op(c, b, in);
        return;
    case POP_BAR:
    case POP_BARRED:
    case POP_MEMBAR:
    case POP_BARWARP:
    case POP_NANOSLEEP:
        emit_barrier_group_op(c, b, in);
        return;
    default:
        failf(c, line, "unhandled group op");
        return;
    }
}
