#include "instructions/memory/memory.h"

void emit_cp_async_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t d;
    uint32_t id;
    (void)d; (void)id;
    switch (in->op) {
    case POP_CPASYNC: {
        // cp.async.ca.shared.global [dst], [src], cp-size, src-size
        // Lowered synchronously byte-by-byte: the group/wait ops are
        // no-ops, so the copy result is already visible. Bytes past
        // src-size (but within cp-size) are zero-filled, matching the
        // hardware's short-copy semantics. Sub-word accesses use the
        // same lane-shifting/CAS helpers as ordinary ld.u8/st.u8.
        int db = -1, sb = -1, dsh = -1, dgl = -1, ssh = -1, sgl = -1;
        uint32_t doff = 0, soff = 0;
        if (resolve_addr(c, b, &in->src[0], &db, &doff, &dsh, &dgl,
                         line) != 0)
            return;
        if (dsh < 0) {
            failf(c, line, "cp.async destination must be shared memory");
            return;
        }
        if (resolve_addr(c, b, &in->src[1], &sb, &soff, &ssh, &sgl,
                         line) != 0)
            return;
        if (ssh >= 0) {
            failf(c, line, "cp.async source must be global memory");
            return;
        }
        uint32_t dyn_size = 0, ignore = 0;
        int have_dyn = 0, have_ignore = 0, src_sz = in->src_size;
        if (in->nsrc >= 4) {
            if (in->src[3].kind == OPD_REG) {
                reg_t *r = find_reg(c, in->src[3].reg);
                if (r && r->decl_ty == PTY_PRED) {
                    ptx_opd_t po = in->src[3];
                    val_t pv = resolve(c, b, &po, PTY_PRED, line);
                    if (c->failed)
                        return;
                    ignore = pv.id;
                    have_ignore = 1;
                    src_sz = in->cp_size;
                } else if (r && r->decl_ty == PTY_U64) {
                    // Cache policy operand (the trailing implicit
                    // operand of .L2::cache_hint); ignore it.
                    src_sz = in->cp_size;
                } else {
                    ptx_opd_t so = in->src[3];
                    val_t sv = resolve(c, b, &so, PTY_U32, line);
                    if (c->failed)
                        return;
                    dyn_size = sv.id;
                    have_dyn = 1;
                    src_sz = -1;
                }
            } else if (in->src[3].kind == OPD_IMM) {
                have_dyn = 0;
            }
        }
        // Fast path for full-width copies from a descriptor-backed
        // global buffer: copy u32 words directly instead of doing one
        // CAS-protected 8-bit shared store per byte. cp.async sizes are
        // naturally aligned, so 4-byte word copies are valid.
        if ((sb >= 0 || sb == -3 || sgl >= 0) &&
            src_sz == in->cp_size && !have_dyn && !have_ignore &&
            (in->cp_size % 4) == 0) {
            for (int i = 0; i < in->cp_size; i += 4) {
                uint32_t dop = doff, sop = soff;
                if (i != 0) {
                    dop = nid(c);
                    emit_arith(c, b, S_IAdd, c->t_u64, dop, doff,
                               const_u64(c, (uint64_t)i));
                    sop = nid(c);
                    emit_arith(c, b, S_IAdd, c->t_u64, sop, soff,
                               const_u64(c, (uint64_t)i));
                }
                uint32_t w;
                if (sgl >= 0) {
                    uint32_t gptr = gl_addr_ptr(c, b, sgl, sop);
                    w = nid(c);
                    emit_load(c, b, c->t_u32, w, gptr);
                } else {
                    w = narrow_ld_global(c, b, sb, sop, 32);
                }
                if (c->failed)
                    return;
                narrow_st_shared(c, b, dsh, dop, w, 32);
                if (c->failed)
                    return;
            }
            return;
        }
        for (int i = 0; i < in->cp_size; i++) {
            uint32_t dop = doff, sop = soff;
            if (i != 0) {
                dop = nid(c);
                emit_arith(c, b, S_IAdd, c->t_u64, dop, doff,
                           const_u64(c, (uint64_t)i));
                sop = nid(c);
                emit_arith(c, b, S_IAdd, c->t_u64, sop, soff,
                           const_u64(c, (uint64_t)i));
            }
            uint32_t byte;
            if (i < src_sz || have_dyn) {
                byte = sgl >= 0
                           ? narrow_ld_glvar(c, b, sgl, sop, 8)
                           : narrow_ld_global(c, b, sb, sop, 8);
                if (c->failed)
                    return;
            } else {
                byte = const_u32(c, 0);
            }
            if (have_dyn || have_ignore) {
                uint32_t keep = 0;
                if (have_dyn) {
                    keep = do_cmp(c, b, PCM_LT, PTY_U32, 0,
                                  const_u32(c, (uint32_t)i), dyn_size,
                                  line);
                }
                if (have_ignore) {
                    uint32_t nignore = nid(c);
                    put(c, b, (4u << 16) | S_LogicalNot);
                    put(c, b, c->t_bool);
                    put(c, b, nignore);
                    put(c, b, ignore);
                    if (keep) {
                        uint32_t both = nid(c);
                        put(c, b, (5u << 16) | S_LogicalAnd);
                        put(c, b, c->t_bool);
                        put(c, b, both);
                        put(c, b, nignore);
                        put(c, b, keep);
                        keep = both;
                    } else {
                        keep = nignore;
                    }
                }
                byte = sel_u32(c, b, keep, byte, const_u32(c, 0));
            }
            narrow_st_shared(c, b, dsh, dop, byte, 8);
        }
        return;
    }
    case POP_CPASYNC_BULK: {
        int db = -1, sb = -1, dsh = -1, dgl = -1, ssh = -1, sgl = -1;
        uint32_t doff = 0, soff = 0;
        if (resolve_addr(c, b, &in->src[0], &db, &doff, &dsh, &dgl,
                         line) != 0)
            return;
        if (resolve_addr(c, b, &in->src[1], &sb, &soff, &ssh, &sgl,
                         line) != 0)
            return;
        val_t sz = resolve(c, b, &in->src[2], PTY_U32, line);
        if (c->failed)
            return;
        if (in->cp_bulk == 1) {
            if (dsh < 0 || (sb < 0 && sgl < 0)) {
                failf(c, line,
                      "cp.async.bulk global->shared needs shared::cta dst");
                return;
            }
            bulk_copy_loop(c, b, dsh, -1, -1, doff, -1, sgl, sb, soff,
                           sz.id, line);
        } else {
            if (ssh < 0 || (db < 0 && dgl < 0)) {
                failf(c, line,
                      "cp.async.bulk shared->global needs shared::cta src");
                return;
            }
            bulk_copy_loop(c, b, -1, dgl, db, doff, ssh, -1, -1, soff,
                           sz.id, line);
        }
        return;
    }
    default:
        return;
    }
}

static uint32_t bulk_gptr(ctx_t *c, spv_blob_t *b, int binding,
                           int gvar, uint32_t off) {
    if (gvar >= 0)
        return gl_addr_ptr(c, b, gvar, off);
    uint32_t word = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u64, word, off, const_u64(c, 2));
    uint32_t zero = const_u32(c, 0);
    uint32_t ptr = nid(c);
    put(c, b, (6u << 16) | S_AccessChain);
    put(c, b, c->t_ptr_u_ssbo);
    put(c, b, ptr);
    put(c, b, c->ssbo_vars[binding]);
    put(c, b, zero);
    put(c, b, word);
    return ptr;
}

// Synchronous bulk copy (16 bytes per iteration).  Async completion is
// immediate in this backend, so the mbarrier/bulk-group operands are
// not needed at runtime.
void bulk_copy_loop(ctx_t *c, spv_blob_t *b, int dst_sh,
                           int dst_gl, int dst_binding, uint32_t dst_off,
                           int src_sh, int src_gl, int src_binding,
                           uint32_t src_off, uint32_t size, int line) {
    uint32_t size64 = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u64);
    put(c, b, size64);
    put(c, b, size);
    uint32_t zero = const_u64(c, 0);
    uint32_t step = const_u64(c, 16);
    uint32_t hdr = nid(c), body = nid(c), cont = nid(c), merge = nid(c);
    emit_store(c, b, c->bulk_i_var, zero);
    emit_branch(c, b, hdr);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, hdr);
    uint32_t iv = nid(c);
    emit_load(c, b, c->t_u64, iv, c->bulk_i_var);
    uint32_t cond = do_cmp(c, b, PCM_LT, PTY_U64, 0, iv, size64, line);
    put(c, b, (4u << 16) | S_LoopMerge);
    put(c, b, merge);
    put(c, b, cont);
    put(c, b, 0);
    put(c, b, (4u << 16) | S_BranchCond);
    put(c, b, cond);
    put(c, b, body);
    put(c, b, merge);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, body);
    for (int j = 0; j < 4; j++) {
        uint32_t soffj = iv;
        if (src_off) {
            uint32_t t = nid(c);
            emit_arith(c, b, S_IAdd, c->t_u64, t, soffj, src_off);
            soffj = t;
        }
        uint32_t doffj = iv;
        if (dst_off) {
            uint32_t t = nid(c);
            emit_arith(c, b, S_IAdd, c->t_u64, t, doffj, dst_off);
            doffj = t;
        }
        if (j) {
            uint32_t t = nid(c);
            emit_arith(c, b, S_IAdd, c->t_u64, t, soffj,
                       const_u64(c, (uint64_t)j * 4u));
            soffj = t;
            t = nid(c);
            emit_arith(c, b, S_IAdd, c->t_u64, t, doffj,
                       const_u64(c, (uint64_t)j * 4u));
            doffj = t;
        }
        uint32_t sp = src_sh >= 0
                          ? sh_addr_ptr(c, b, src_sh, soffj)
                          : bulk_gptr(c, b, src_binding, src_gl, soffj);
        uint32_t w = nid(c);
        emit_load(c, b, c->t_u32, w, sp);
        uint32_t dp = dst_sh >= 0
                          ? sh_addr_ptr(c, b, dst_sh, doffj)
                          : bulk_gptr(c, b, dst_binding, dst_gl, doffj);
        emit_store(c, b, dp, w);
    }
    uint32_t next = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, next, iv, step);
    emit_store(c, b, c->bulk_i_var, next);
    emit_branch(c, b, cont);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, cont);
    emit_branch(c, b, hdr);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, merge);
}
