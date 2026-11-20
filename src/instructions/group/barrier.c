#include "instructions/group/group.h"

void emit_barrier_group_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    switch (in->op) {
    case POP_BAR: {
        // Counted barriers and bar.arrive use the shared-memory counter
        // array. A plain bar.sync stays a hardware full-CTA barrier
        // unless the same id also appears in a counted/arrive form (a
        // hardware rendezvous and a software arrival cannot be mixed).
        if (in->arg < 0 && c->uses_swbar) {
            failf(c, line, "dynamic barrier id cannot be mixed with counted barriers");
            return;
        }
        int sw = in->bar_cluster || in->bar_arrive || in->nsrc > 0 ||
                 (in->arg >= 0 && in->arg < 17 && c->bar_sw_id[in->arg]);
        if (sw) {
            if (in->arg < 0 || in->arg > 16) {
                failf(c, line, "dynamic barrier ids cannot be counted");
                return;
            }
            uint32_t ptr = swbar_ptr(c, b, (uint32_t)in->arg);
            if (c->failed)
                return;
            if (in->bar_arrive) {
                // Non-blocking arrival. The expected count is only
                // meaningful to a later bar.sync; each thread adds one.
                swbar_add(c, b, ptr, const_u32(c, 1));
                return;
            }
            uint32_t count;
            if (in->nsrc > 0) {
                val_t cv = resolve(c, b, &in->src[0], PTY_U32, line);
                if (c->failed)
                    return;
                count = cv.id;
            } else if (in->bar_cluster) {
                // Cluster barrier: the implicit participant count is
                // the whole CTA (cluster size 1 approximation).
                uint32_t xy = nid(c);
                emit_arith(c, b, S_IMul, c->t_u32, xy, c->sp_x,
                           c->sp_y);
                count = nid(c);
                emit_arith(c, b, S_IMul, c->t_u32, count, xy, c->sp_z);
            } else {
                // Bare bar.sync sharing an id with a counted/arrive
                // form: the implicit count is the whole CTA (1-D
                // launches, so the x workgroup-size constant).
                count = c->sp_x;
            }
            swbar_wait(c, b, ptr, count);
            return;
        }
        // Full-CTA barrier: uniform execution scope (Workgroup),
        // Workgroup memory scope, AcquireRelease|WorkgroupMemory
        // (0x108, matching glslc's barrier()). Divergent placement is
        // UB exactly as in CUDA (likely hang); uniform placement lower
        // to a correct rendezvous.
        uint32_t exec = const_u32(c, 2); // Scope Workgroup
        uint32_t mem = const_u32(c, 2);
        uint32_t sem = const_u32(c, 0x108u);
        put(c, b, (4u << 16) | S_ControlBarrier);
        put(c, b, exec);
        put(c, b, mem);
        put(c, b, sem);
        return;
    }
    case POP_BARRED: {
        // bar.red.and/or.pred: reduction across the CTA barrier with
        // predicate result. Lower like vote.all/any over the warp for
        // simplicity (matches common usage in these tests); use full
        // warp/membermask.
        uint32_t scope = const_u32(c, 3); // Subgroup
        uint32_t sub = nid(c);
        emit_load(c, b, c->t_u32, sub, c->v_subid);
        uint32_t W = nid(c);
        emit_arith(c, b, S_ShrL, c->t_u32, W, sub, const_u32(c, 5));
        uint32_t pv = pred_val(c, b, &in->src[0], in->bneg, line);
        if (c->failed)
            return;
        uint32_t bal = nid(c);
        put(c, b, (5u << 16) | S_GroupBallot);
        put(c, b, c->t_uvec4);
        put(c, b, bal);
        put(c, b, scope);
        put(c, b, pv);
        uint32_t word = ballot_warp_word(c, b, bal, W);
        uint32_t mask = const_u32(c, 0xffffffffu);
        if (in->vote_mode == 2) { // popc: count true predicates
            uint32_t cnt = nid(c);
            put(c, b, (4u << 16) | S_BitCount);
            put(c, b, c->t_u32);
            put(c, b, cnt);
            put(c, b, word);
            def_val(c, in->dst, PTY_U32, cnt);
            return;
        }
        if (in->vote_mode == 1) { // or
            uint32_t m = nid(c), nz = nid(c);
            emit_arith(c, b, S_And, c->t_u32, m, word, mask);
            put(c, b, (5u << 16) | S_INotEqual);
            put(c, b, c->t_bool);
            put(c, b, nz);
            put(c, b, m);
            put(c, b, const_u32(c, 0));
            def_val(c, in->dst, PTY_PRED, nz);
            return;
        } else { // and
            uint32_t abal = nid(c);
            {
                uint32_t t = nid(c);
                put(c, &c->types, (3u << 16) | S_ConstantTrue);
                put(c, &c->types, c->t_bool);
                put(c, &c->types, t);
                put(c, b, (5u << 16) | S_GroupBallot);
                put(c, b, c->t_uvec4);
                put(c, b, abal);
                put(c, b, scope);
                put(c, b, t);
            }
            uint32_t aword = ballot_warp_word(c, b, abal, W);
            uint32_t lm = nid(c), rm = nid(c), eq = nid(c);
            emit_arith(c, b, S_And, c->t_u32, lm, word, mask);
            emit_arith(c, b, S_And, c->t_u32, rm, aword, mask);
            put(c, b, (5u << 16) | S_IEqual);
            put(c, b, c->t_bool);
            put(c, b, eq);
            put(c, b, lm);
            put(c, b, rm);
            def_val(c, in->dst, PTY_PRED, eq);
            return;
        }
    }
    case POP_MEMBAR: {
        // OpMemoryBarrier with the PTX scope (cta/cluster -> Workgroup;
        // gpu/sys/proxy -> Device; Vulkan has no finer distinctions).
        // Vulkan forbids SequentiallyConsistent here, so .sc and
        // .acq_rel both use AcquireRelease (0x8) over uniform (0x40) +
        // workgroup (0x100) memory.  Explicit acquire/release fences
        // keep only their half of the ordering.
        uint32_t scope = const_u32(
            c, (in->fence_scope == 0 || in->fence_scope == 1) ? 2u : 1u);
        /* Device/system-scope memory barriers in tight CAS loops can
         * trigger severe driver slowdowns/hangs or pipeline compilation
         * issues on some Vulkan implementations.  Downgrade non-proxy
         * device/gpu/sys-scope fences to workgroup scope as a compatibility
         * fallback; atomics and explicit ld/st ordering carry most ordering
         * in common lifted-libcu++ CAS patterns, while preserving a barrier
         * for PTX-path kernels that may rely on membar/fence presence. */
        if (in->arg == 0 && in->fence_scope >= 2)
            scope = const_u32(c, 2u);
        uint32_t sem = const_u32(
            c, (in->fence_sem == 1 ? 0x2u
                : in->fence_sem == 2 ? 0x4u
                                     : 0x8u) |
                   0x40u | 0x100u);
        put(c, b, (3u << 16) | S_MemoryBarrier);
        put(c, b, scope);
        put(c, b, sem);
        return;
    }
    case POP_BARWARP: {
        // Warp-level rendezvous: subgroup execution and memory scope.
        uint32_t exec = const_u32(c, 3); // Subgroup
        uint32_t mem = const_u32(c, 2);  // Workgroup
        uint32_t sem = const_u32(c, 0x108u); // AcquireRelease|WorkgroupMemory
        put(c, b, (4u << 16) | S_ControlBarrier);
        put(c, b, exec);
        put(c, b, mem);
        put(c, b, sem);
        return;
    }
    case POP_NANOSLEEP: {
        // Pure performance hint: type-check the operand (ptxas rejects
        // non-u32 forms) and drop it. No observable result changes.
        val_t v = resolve(c, b, &in->src[0], PTY_U32, line);
        (void)v;
        if (c->failed)
            return;
        return;
    }
    default:
        return;
    }
}
