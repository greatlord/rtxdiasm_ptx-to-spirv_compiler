// red.async and multimem.red.async lowering.
//
// The relaxed form targets distributed shared memory in a cluster;
// this backend has no cluster support, so the reduction is performed
// on the local shared address and the completion (which is immediate
// because the operation is synchronous here) is reported to the
// mbarrier via a complete-tx of the reduced byte count.
//
// The release form targets global memory and maps directly to a
// release-scoped atomic add.  multimem.red.async uses the same
// lowering: with a single device the multimem address refers to that
// device's own memory.
#include <stdint.h>
#include <stdio.h>
#include "spirv_internal.h"

static int emit_red_async_body(ctx_t *c, spv_blob_t *b,
                               const ptx_instr_t *in) {
    int line = in->line;
    int binding = -1, shvar = -1, glvar = -1;
    uint32_t off = 0;
    if (resolve_addr(c, b, &in->src[0], &binding, &off, &shvar, &glvar,
                     line) != 0)
        return -1;
    int is64 = in->ty == PTY_U64;
    int relaxed = !in->rda_release;
    uint32_t ptr, scope, sem;
    if (relaxed) {
        if (shvar < 0) {
            failf(c, line,
                  "red.async shared form needs a .shared-derived address");
            return -1;
        }
        if (is64) {
            if (c->shvars[shvar].elem != 8) {
                failf(c, line,
                      "64-bit red.async requires a .u64 shared variable");
                return -1;
            }
            ptr = sh_addr_ptr64(c, b, shvar, off);
        } else {
            ptr = sh_addr_ptr(c, b, shvar, off);
        }
        scope = const_u32(c, 2);     // Workgroup
        sem = const_u32(c, 0x108u);  // AcquireRelease|WorkgroupMemory
    } else {
        if (shvar >= 0) {
            failf(c, line,
                  "red.async global form needs a global address");
            return -1;
        }
        if (binding == -2 || glvar >= 0) {
            failf(c, line, "red.async destination is not a device pointer");
            return -1;
        }
        if (is64) {
            ptr = ssbo64_ptr(c, b, binding, off);
        } else {
            ptr = addr_ptr(c, b, binding, off);
        }
        // .gpu and .sys both map to Device: Vulkan has no CrossDevice
        // scope, and the strong Device ordering is the same safe
        // superset the atom/ld/st paths use.
        scope = const_u32(c, 1u);
        sem = const_u32(c, 0x44u);   // Release|UniformMemory
    }
    if (c->failed)
        return -1;

    if (in->atom_op == 1 || in->atom_op == 2) {
        val_t bound = resolve(c, b, &in->src[1], PTY_U32, line);
        if (c->failed)
            return -1;
        if (in->atom_op == 1)
            atom_wrap_loop(c, b, ptr, 1, bound.id);
        else
            atom_dec_loop(c, b, ptr, 1, bound.id);
    } else {
        val_t val = resolve(c, b, &in->src[1],
                            is64 ? PTY_U64 : PTY_U32, line);
        if (c->failed)
            return -1;
        uint32_t opc;
        switch (in->atom_op) {
        case 0: opc = S_AtomicIAdd; break;
        case 3: opc = S_AtomicAnd; break;
        case 4: opc = S_AtomicOr; break;
        case 5: opc = S_AtomicXor; break;
        case 7: opc = in->sign ? S_AtomicSMin : S_AtomicUMin; break;
        case 8: opc = in->sign ? S_AtomicSMax : S_AtomicUMax; break;
        default:
            failf(c, line, "unsupported red.async reduction");
            return -1;
        }
        uint32_t res = nid(c);
        put(c, b, (7u << 16) | opc);
        put(c, b, is64 ? c->t_u64 : c->t_u32);
        put(c, b, res);
        put(c, b, ptr);
        put(c, b, scope);
        put(c, b, sem);
        put(c, b, val.id);
    }
    if (c->failed)
        return -1;

    if (relaxed && in->nsrc > 2) {
        int mbind = -1, msh = -1, mgl = -1;
        uint32_t moff = 0;
        if (resolve_addr(c, b, &in->src[2], &mbind, &moff, &msh, &mgl,
                         line) != 0)
            return -1;
        if (msh < 0) {
            failf(c, line, "red.async mbarrier must be shared memory");
            return -1;
        }
        uint32_t mptr = sh_addr_ptr64(c, b, msh, moff);
        emit_mbarrier_complete_tx(c, b, mptr, is64 ? 8u : 4u);
    }
    return c->failed ? -1 : 0;
}

void emit_red_async(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    if (!in->pred[0]) {
        emit_red_async_body(c, b, in);
        return;
    }
    emit_predicated_instr(c, b, in, emit_red_async_body);
}
