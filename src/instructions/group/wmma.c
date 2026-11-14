#include "instructions/group/group.h"

uint32_t wmma_subid(ctx_t *c, spv_blob_t *b) {
    uint32_t s = nid(c);
    emit_load(c, b, c->t_u32, s, c->v_subid);
    return s;
}

// Shuffle owner is warp-relative; add the warp base so the same code
// works for subgroups larger than 32 lanes (e.g. 64-lane waves).
uint32_t wmma_shfl(ctx_t *c, spv_blob_t *b, uint32_t val,
                          uint32_t lane, uint32_t base) {
    uint32_t idx = nid(c), r = nid(c);
    emit_arith(c, b, S_Or, c->t_u32, idx, lane, base);
    put(c, b, (6u << 16) | S_GroupShuffle);
    put(c, b, c->t_u32);
    put(c, b, r);
    put(c, b, const_u32(c, 3)); // Subgroup
    put(c, b, val);
    put(c, b, idx);
    return r;
}


uint32_t acuda_coopmat_type(ctx_t *c, int comp, int use) {
    uint32_t *cache = (comp == 0) ? c->t_cm_f16
                      : (comp == 1) ? c->t_cm_bf16
                                    : c->t_cm_f32;
    if (cache[use])
        return cache[use];
    uint32_t comp_ty;
    if (comp == 0) {
        if (!c->t_f16) {
            c->t_f16 = nid(c);
            put(c, &c->types, (3u << 16) | S_TypeFloat);
            put(c, &c->types, c->t_f16);
            put(c, &c->types, 16);
        }
        comp_ty = c->t_f16;
    } else if (comp == 1) {
        if (!c->t_bf16) {
            c->t_bf16 = nid(c);
            put(c, &c->types, (4u << 16) | S_TypeFloat);
            put(c, &c->types, c->t_bf16);
            put(c, &c->types, 16);
            put(c, &c->types, 0); // BFloat16KHR
        }
        comp_ty = c->t_bf16;
    } else {
        comp_ty = c->t_f32;
    }
    uint32_t scope = const_u32(c, 3);
    uint32_t rows = const_u32(c, 16);
    uint32_t cols = const_u32(c, 16);
    uint32_t use_id = const_u32(c, (uint32_t)use);
    uint32_t id = nid(c);
    put(c, &c->types, (7u << 16) | 4456); // OpTypeCooperativeMatrixKHR
    put(c, &c->types, id);
    put(c, &c->types, comp_ty);
    put(c, &c->types, scope);
    put(c, &c->types, rows);
    put(c, &c->types, cols);
    put(c, &c->types, use_id);
    cache[use] = id;
    c->uses_coopmat = 1;
    return id;
}

// f16 pair dot with f32 accumulation (SPV_VALVE_mixed_float_dot_product).
uint32_t emit_fdot2_f32(ctx_t *c, spv_blob_t *b, uint32_t a_bits,
                        uint32_t b_bits, uint32_t acc) {
    ensure_f16_types(c);
    uint32_t va = nid(c), vb = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_v2f16);
    put(c, b, va);
    put(c, b, a_bits);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_v2f16);
    put(c, b, vb);
    put(c, b, b_bits);
    uint32_t r = nid(c);
    put(c, b, (6u << 16) | S_FDot2MixAcc32VALVE);
    put(c, b, c->t_f32);
    put(c, b, r);
    put(c, b, va);
    put(c, b, vb);
    put(c, b, acc);
    return r;
}

// Packed 4x8-bit dot product for the int8 MMA inner loop. Each group
// of four K values is exactly one 32-bit A register and one 32-bit B
// register in the owning lanes, so one op replaces four byte MACs and
// four shuffles.
uint32_t dot4_packed(ctx_t *c, spv_blob_t *b, int asign, int bsign,
                            uint32_t a, uint32_t bv) {
    int form_ok = asign && bsign ? c->caps.dot_signed
                  : !asign && !bsign ? c->caps.dot_unsigned
                                     : c->caps.dot_mixed;
    // Only use the packed-dot op when the module pre-scan decided that
    // the device can run it; otherwise fall back to byte multiplies.
    if (!c->use_dot_product || !form_ok) {
        uint32_t acc = const_u32(c, 0);
        for (int i = 0; i < 4; i++) {
            uint32_t av = pack_lane_get(c, b, a, i, 8, asign);
            uint32_t bvv = pack_lane_get(c, b, bv, i, 8, bsign);
            uint32_t pr = nid(c), nx = nid(c);
            emit_arith(c, b, S_IMul, c->t_u32, pr, av, bvv);
            emit_arith(c, b, S_IAdd, c->t_u32, nx, acc, pr);
            acc = nx;
        }
        return acc;
    }
    uint32_t opc, x = a, y = bv;
    if (asign && bsign) {
        opc = S_SDot;
    } else if (!asign && !bsign) {
        opc = S_UDot;
    } else if (asign) {
        opc = S_SUDot;
    } else {
        opc = S_SUDot; // unsigned A, signed B: swap
        x = bv;
        y = a;
    }
    c->use_dot_product = 1;
    uint32_t d = nid(c);
    put(c, b, (6u << 16) | opc);
    put(c, b, c->t_u32);
    put(c, b, d);
    put(c, b, x);
    put(c, b, y);
    put(c, b, 0u); // PackedVectorFormat4x8Bit
    return d;
}

// Expand eight packed 4-bit values into two packed 4-byte words.  The
// two words are laid out so dot4_packed can consume them directly.
void unpack4_to_bytes(ctx_t *c, spv_blob_t *b, uint32_t w,
                             int is_signed, uint32_t out[2]) {
    for (int h = 0; h < 2; h++) {
        uint32_t acc = const_u32(c, 0);
        for (int i = 0; i < 4; i++) {
            int sh = (h * 4 + i) * 4;
            uint32_t v = nid(c), n = nid(c);
            emit_arith(c, b, S_ShrL, c->t_u32, v, w,
                       const_u32(c, (uint32_t)sh));
            emit_arith(c, b, S_And, c->t_u32, n, v,
                       const_u32(c, 0xfu));
            if (is_signed) {
                uint32_t sl = nid(c), sa = nid(c);
                emit_arith(c, b, S_Shl, c->t_u32, sl, n,
                           const_u32(c, 28));
                emit_arith(c, b, S_ShrA, c->t_u32, sa, sl,
                           const_u32(c, 28));
                n = sa;
            }
            uint32_t byte = nid(c), shb = nid(c);
            emit_arith(c, b, S_And, c->t_u32, byte, n,
                       const_u32(c, 0xffu));
            emit_arith(c, b, S_Shl, c->t_u32, shb, byte,
                       const_u32(c, (uint32_t)(8 * i)));
            uint32_t o = nid(c);
            emit_arith(c, b, S_Or, c->t_u32, o, acc, shb);
            acc = o;
        }
        out[h] = acc;
    }
}

// Pick one f16 lane (idx 0..3) from the two packed B words for column
// cc, returning it in the low half (zero-extended). Used by the sparse
// dot2 path to pair the two nonzeros with the selected B values.
uint32_t sparse_pick_b(ctx_t *c, spv_blob_t *b,
                              const uint32_t raw[2][2], int cc,
                              uint32_t idx, int line) {
    uint32_t cond = do_cmp(c, b, PCM_GE, PTY_U32, 0, idx,
                           const_u32(c, 2), line);
    if (c->failed)
        return 0;
    uint32_t w = sel_u32(c, b, cond, raw[1][cc], raw[0][cc]);
    uint32_t half = nid(c), sh = nid(c);
    emit_arith(c, b, S_And, c->t_u32, half, idx, const_u32(c, 1));
    emit_arith(c, b, S_Shl, c->t_u32, sh, half, const_u32(c, 4));
    uint32_t v = nid(c), m = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, v, w, sh);
    emit_arith(c, b, S_And, c->t_u32, m, v, const_u32(c, 0xffffu));
    return m;
}

uint32_t wmma_index(ctx_t *c, spv_blob_t *b, uint32_t row,
                           uint32_t col, uint32_t stride, int rowmajor) {
    uint32_t a = rowmajor ? row : col;
    uint32_t bb = rowmajor ? col : row;
    uint32_t t = nid(c), r = nid(c);
    emit_arith(c, b, S_IMul, c->t_u32, t, a, stride);
    emit_arith(c, b, S_IAdd, c->t_u32, r, t, bb);
    return r;
}

uint32_t wmma_ld_half(ctx_t *c, spv_blob_t *b, int shvar,
                             int glvar, int binding, uint32_t off) {
    if (shvar >= 0)
        return narrow_ld_shared(c, b, shvar, off, 16);
    if (glvar >= 0)
        return narrow_ld_glvar(c, b, glvar, off, 16);
    return narrow_ld_global(c, b, binding, off, 16);
}

uint32_t wmma_ld_bytes4(ctx_t *c, spv_blob_t *b, int shvar,
                               int glvar, int binding, uint32_t off) {
    if (shvar >= 0)
        return narrow_ld_shared(c, b, shvar, off, 32);
    if (glvar >= 0)
        return narrow_ld_glvar(c, b, glvar, off, 32);
    return narrow_ld_global(c, b, binding, off, 32);
}

uint32_t wmma_ld_f32(ctx_t *c, spv_blob_t *b, int shvar, int glvar,
                            int binding, uint32_t off) {
    uint32_t ptr = shvar >= 0 ? sh_addr_ptr(c, b, shvar, off)
                  : glvar >= 0 ? gl_addr_ptr(c, b, glvar, off)
                               : addr_ptr(c, b, binding, off);
    uint32_t v = nid(c);
    emit_load(c, b, c->t_u32, v, ptr);
    return v;
}

void wmma_st_f32(ctx_t *c, spv_blob_t *b, int shvar, int glvar,
                        int binding, uint32_t off, uint32_t v) {
    uint32_t ptr = shvar >= 0 ? sh_addr_ptr(c, b, shvar, off)
                  : glvar >= 0 ? gl_addr_ptr(c, b, glvar, off)
                               : addr_ptr(c, b, binding, off);
    emit_store(c, b, ptr, v);
}

uint32_t wmma_byte_off(ctx_t *c, spv_blob_t *b, uint32_t base_off,
                              uint32_t idx, int eshift) {
    uint32_t i64 = nid(c), sh = nid(c), add = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u64);
    put(c, b, i64);
    put(c, b, idx);
    emit_arith(c, b, S_Shl, c->t_u64, sh, i64, const_u64(c, (uint64_t)eshift));
    uint32_t base_ssa = base_off ? base_off : const_u64(c, 0);
    emit_arith(c, b, S_IAdd, c->t_u64, add, base_ssa, sh);
    return add;
}


void emit_wmma_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    switch (in->op) {
    case POP_WMMA_LOAD:
    case POP_WMMA_STORE: {
        int binding = -1, shvar = -1, glvar = -1;
        uint32_t base = 0;
        if (resolve_addr(c, b, &in->src[0], &binding, &base, &shvar,
                         &glvar, line) != 0)
            return;
        if (binding < 0 && binding != -3 && shvar < 0 && glvar < 0) {
            failf(c, line, "wmma load/store needs a global or shared address");
            return;
        }
        uint32_t stride = const_u32(c, 16);
        if (in->nsrc > 1) {
            val_t sv = resolve(c, b, &in->src[1], PTY_U32, line);
            if (c->failed)
                return;
            stride = sv.id;
        }
        // Single-bit WMMA m8n8k128: one 32-bit word per lane for A/B.
        if (in->mma_elem == 2 && in->wmma_shape == 5) {
            uint32_t l = wmma_subid(c, b);
            uint32_t row = nid(c), g = nid(c);
            emit_arith(c, b, S_ShrL, c->t_u32, row, l, const_u32(c, 2));
            emit_arith(c, b, S_And, c->t_u32, g, l, const_u32(c, 3));
            if (in->op == POP_WMMA_LOAD) {
                if (in->wmma_kind <= 1) {
                    uint32_t k0 = nid(c), rr = nid(c), cc = nid(c);
                    emit_arith(c, b, S_Shl, c->t_u32, k0, g,
                               const_u32(c, 2));
                    if (in->wmma_kind == 0) {
                        rr = row;
                        cc = k0;
                    } else {
                        rr = k0;
                        cc = row;
                    }
                    uint32_t idx = wmma_index(
                        c, b, rr, cc, stride, in->wmma_row);
                    uint32_t off = wmma_byte_off(c, b, base, idx, 0);
                    uint32_t v = wmma_ld_bytes4(c, b, shvar, glvar,
                                                binding, off);
                    if (c->failed)
                        return;
                    def_val(c, in->wregs[0][0], PTY_U32, v);
                } else {
                    for (int i = 0; i < 2; i++) {
                        uint32_t col = nid(c), col2 = nid(c);
                        emit_arith(c, b, S_Shl, c->t_u32, col, g,
                                   const_u32(c, 1));
                        emit_arith(c, b, S_IAdd, c->t_u32, col2, col,
                                   const_u32(c, (uint32_t)i));
                        uint32_t idx = wmma_index(
                            c, b, row, col2, stride, in->wmma_row);
                        uint32_t off = wmma_byte_off(c, b, base, idx, 2);
                        uint32_t v = wmma_ld_f32(c, b, shvar, glvar,
                                                 binding, off);
                        if (c->failed)
                            return;
                        def_val(c, in->wregs[0][i], PTY_U32, v);
                    }
                }
            } else {
                for (int i = 0; i < 2; i++) {
                    uint32_t col = nid(c), col2 = nid(c);
                    emit_arith(c, b, S_Shl, c->t_u32, col, g,
                               const_u32(c, 1));
                    emit_arith(c, b, S_IAdd, c->t_u32, col2, col,
                               const_u32(c, (uint32_t)i));
                    uint32_t idx = wmma_index(
                        c, b, row, col2, stride, in->wmma_row);
                    uint32_t off = wmma_byte_off(c, b, base, idx, 2);
                    reg_t *dr = find_reg(c, in->wregs[0][i]);
                    if (!dr || !dr->defined) {
                        failf(c, line, "wmma.store register has no value");
                        return;
                    }
                    uint32_t v = nid(c);
                    put(c, b, (4u << 16) | S_Bitcast);
                    put(c, b, c->t_u32);
                    put(c, b, v);
                    put(c, b, dr->cur);
                    wmma_st_f32(c, b, shvar, glvar, binding, off, v);
                    if (c->failed)
                        return;
                }
            }
            return;
        }
        // 4-bit WMMA m8n8k32: one packed word per lane for A and B.
        if (in->mma_elem == 2 && in->wmma_shape == 4) {
            uint32_t l = wmma_subid(c, b);
            uint32_t row = nid(c), g = nid(c);
            emit_arith(c, b, S_ShrL, c->t_u32, row, l, const_u32(c, 2));
            emit_arith(c, b, S_And, c->t_u32, g, l, const_u32(c, 3));
            if (in->op == POP_WMMA_LOAD) {
                if (in->wmma_kind <= 1) {
                    uint32_t k0 = nid(c), rr = nid(c), cc = nid(c);
                    emit_arith(c, b, S_Shl, c->t_u32, k0, g,
                               const_u32(c, 2));
                    if (in->wmma_kind == 0) {
                        rr = row;
                        cc = k0;
                    } else {
                        rr = k0;
                        cc = row;
                    }
                    uint32_t idx = wmma_index(
                        c, b, rr, cc, stride, in->wmma_row);
                    uint32_t off = wmma_byte_off(c, b, base, idx, 0);
                    uint32_t v = wmma_ld_bytes4(c, b, shvar, glvar,
                                                binding, off);
                    if (c->failed)
                        return;
                    def_val(c, in->wregs[0][0], PTY_U32, v);
                } else {
                    for (int i = 0; i < 2; i++) {
                        uint32_t col = nid(c), col2 = nid(c);
                        emit_arith(c, b, S_Shl, c->t_u32, col, g,
                                   const_u32(c, 1));
                        emit_arith(c, b, S_IAdd, c->t_u32, col2, col,
                                   const_u32(c, (uint32_t)i));
                        uint32_t idx = wmma_index(
                            c, b, row, col2, stride, in->wmma_row);
                        uint32_t off = wmma_byte_off(c, b, base, idx, 2);
                        uint32_t v = wmma_ld_f32(c, b, shvar, glvar,
                                                 binding, off);
                        if (c->failed)
                            return;
                        def_val(c, in->wregs[0][i], PTY_U32, v);
                    }
                }
            } else {
                for (int i = 0; i < 2; i++) {
                    uint32_t col = nid(c), col2 = nid(c);
                    emit_arith(c, b, S_Shl, c->t_u32, col, g,
                               const_u32(c, 1));
                    emit_arith(c, b, S_IAdd, c->t_u32, col2, col,
                               const_u32(c, (uint32_t)i));
                    uint32_t idx = wmma_index(
                        c, b, row, col2, stride, in->wmma_row);
                    uint32_t off = wmma_byte_off(c, b, base, idx, 2);
                    reg_t *dr = find_reg(c, in->wregs[0][i]);
                    if (!dr || !dr->defined) {
                        failf(c, line, "wmma.store register has no value");
                        return;
                    }
                    uint32_t v = nid(c);
                    put(c, b, (4u << 16) | S_Bitcast);
                    put(c, b, c->t_u32);
                    put(c, b, v);
                    put(c, b, dr->cur);
                    wmma_st_f32(c, b, shvar, glvar, binding, off, v);
                    if (c->failed)
                        return;
                }
            }
            return;
        }
        // Integer WMMA fragments are packed four bytes per register.
        if (in->mma_elem == 2 && in->wmma_shape == 0) {
            uint32_t l = wmma_subid(c, b);
            if (in->op == POP_WMMA_LOAD) {
                if (in->wmma_kind <= 1) {
                    uint32_t rr = nid(c), hh = nid(c);
                    emit_arith(c, b, S_ShrL, c->t_u32, rr, l,
                               const_u32(c, 1));
                    emit_arith(c, b, S_And, c->t_u32, hh, l,
                               const_u32(c, 1));
                    for (int j = 0; j < 2; j++) {
                        uint32_t kg = nid(c), kg2 = nid(c);
                        uint32_t row = nid(c), col = nid(c);
                        emit_arith(c, b, S_Shl, c->t_u32, kg, hh,
                                   const_u32(c, 1));
                        emit_arith(c, b, S_IAdd, c->t_u32, kg2, kg,
                                   const_u32(c, (uint32_t)j));
                        emit_arith(c, b, S_Shl, c->t_u32, col, kg2,
                                   const_u32(c, 2));
                        if (in->wmma_kind == 0) {
                            row = rr;
                        } else {
                            row = col;
                            col = rr;
                        }
                        uint32_t idx = wmma_index(
                            c, b, row, col, stride, in->wmma_row);
                        uint32_t off = wmma_byte_off(c, b, base, idx, 0);
                        uint32_t v = wmma_ld_bytes4(c, b, shvar, glvar,
                                                    binding, off);
                        if (c->failed)
                            return;
                        def_val(c, in->wregs[0][j], PTY_U32, v);
                    }
                } else {
                    for (int i = 0; i < 8; i++) {
                        uint32_t row = nid(c), col = nid(c);
                        uint32_t col2 = nid(c), col3 = nid(c);
                        emit_arith(c, b, S_ShrL, c->t_u32, row, l,
                                   const_u32(c, 1));
                        emit_arith(c, b, S_And, c->t_u32, col, l,
                                   const_u32(c, 1));
                        emit_arith(c, b, S_Shl, c->t_u32, col2, col,
                                   const_u32(c, 3));
                        emit_arith(c, b, S_IAdd, c->t_u32, col3, col2,
                                   const_u32(c, (uint32_t)i));
                        uint32_t idx = wmma_index(
                            c, b, row, col3, stride, in->wmma_row);
                        uint32_t off = wmma_byte_off(c, b, base, idx, 2);
                        uint32_t v = wmma_ld_f32(c, b, shvar, glvar,
                                                 binding, off);
                        if (c->failed)
                            return;
                        def_val(c, in->wregs[0][i], PTY_U32, v);
                    }
                }
            } else {
                for (int i = 0; i < 8; i++) {
                    uint32_t row = nid(c), col = nid(c);
                    uint32_t col2 = nid(c), col3 = nid(c);
                    emit_arith(c, b, S_ShrL, c->t_u32, row, l,
                               const_u32(c, 1));
                    emit_arith(c, b, S_And, c->t_u32, col, l,
                               const_u32(c, 1));
                    emit_arith(c, b, S_Shl, c->t_u32, col2, col,
                               const_u32(c, 3));
                    emit_arith(c, b, S_IAdd, c->t_u32, col3, col2,
                               const_u32(c, (uint32_t)i));
                    uint32_t idx = wmma_index(
                        c, b, row, col3, stride, in->wmma_row);
                    uint32_t off = wmma_byte_off(c, b, base, idx, 2);
                    reg_t *dr = find_reg(c, in->wregs[0][i]);
                    if (!dr || !dr->defined) {
                        failf(c, line, "wmma.store register has no value");
                        return;
                    }
                    uint32_t v = nid(c);
                    put(c, b, (4u << 16) | S_Bitcast);
                    put(c, b, c->t_u32);
                    put(c, b, v);
                    put(c, b, dr->cur);
                    wmma_st_f32(c, b, shvar, glvar, binding, off, v);
                    if (c->failed)
                        return;
                }
            }
            return;
        }
        // Cooperative-matrix lowering for m16n16k16 fragments.
        if (in->wmma_shape == 0) {
            if (in->op == POP_WMMA_LOAD) {
                reg_t *dr = find_reg(c, in->wregs[0][0]);
                if (dr && dr->is_coopmat) {
                    uint32_t ptr;
                    if (shvar >= 0) ptr = sh_addr_ptr(c, b, shvar, base);
                    else if (glvar >= 0) ptr = gl_addr_ptr(c, b, glvar, base);
                    else ptr = addr_ptr(c, b, binding, base);
                    if (c->failed) return;
                    int typed_shared = (shvar >= 0 && c->sh_coop_typed);
                    uint32_t stride_use = stride;
                    if (!typed_shared && dr->coopmat_comp != 2) {
                        stride_use = nid(c);
                        emit_arith(c, b, S_ShrL, c->t_u32, stride_use, stride,
                                   const_u32(c, 1));
                    }
                    uint32_t layout = const_u32(c, in->wmma_row ? 0 : 1);
                    uint32_t id = nid(c);
                    put(c, b, (8u << 16) | 4457); // OpCooperativeMatrixLoadKHR
                    put(c, b, dr->coopmat_ty);
                    put(c, b, id);
                    put(c, b, ptr);
                    put(c, b, layout);
                    put(c, b, stride_use);
                    put(c, b, 2); // Aligned
                    put(c, b, typed_shared ? 2
                                           : (dr->coopmat_comp == 2 ? 4 : 2));
                    dr->defined = 1;
                    dr->cur = id;
                    dr->cur_ty = PTY_U32;
                    dr->cur_is_coopmat = 1;
                    c->uses_coopmat = 1;
                    return;
                }
            } else if (in->op == POP_WMMA_STORE) {
                reg_t *sr = find_reg(c, in->wregs[0][0]);
                if (sr && sr->is_coopmat && sr->defined) {
                    uint32_t ptr;
                    if (shvar >= 0) ptr = sh_addr_ptr(c, b, shvar, base);
                    else if (glvar >= 0) ptr = gl_addr_ptr(c, b, glvar, base);
                    else ptr = addr_ptr(c, b, binding, base);
                    if (c->failed) return;
                    uint32_t layout = const_u32(c, in->wmma_row ? 0 : 1);
                    put(c, b, (7u << 16) | 4458); // OpCooperativeMatrixStoreKHR
                    put(c, b, ptr);
                    put(c, b, sr->cur);
                    put(c, b, layout);
                    put(c, b, stride);
                    put(c, b, 2); // Aligned
                    put(c, b, sr->coopmat_comp == 2 ? 4 : 2);
                    c->uses_coopmat = 1;
                    return;
                }
            }
        }
        uint32_t sub = wmma_subid(c, b);
        uint32_t lane = nid(c), wbase = nid(c);
        emit_arith(c, b, S_And, c->t_u32, lane, sub, const_u32(c, 31));
        emit_arith(c, b, S_And, c->t_u32, wbase, sub,
                   const_u32(c, 0xffffffe0u));
        uint32_t r = nid(c), half = nid(c), c0 = nid(c);
        emit_arith(c, b, S_ShrL, c->t_u32, r, lane, const_u32(c, 1));
        emit_arith(c, b, S_And, c->t_u32, half, lane, const_u32(c, 1));
        emit_arith(c, b, S_Shl, c->t_u32, c0, half, const_u32(c, 3));
        int is_f32 = in->wmma_kind >= 2 && !in->mma_f16acc;
        int eshift = is_f32 ? 2 : 1;
        if (in->wmma_shape == 3) {
            // m16n16k8 tf32: lane r=l>>1, block=l&1. A 4 tf32 regs
            // (row r, cols 4g..4g+3); B 4 tf32 regs (col r, rows
            // 4g..4g+3); C/D 8 f32 row r cols 8g..8g+7.
            uint32_t rr = nid(c), bb = nid(c);
            emit_arith(c, b, S_ShrL, c->t_u32, rr, lane, const_u32(c, 1));
            emit_arith(c, b, S_And, c->t_u32, bb, lane, const_u32(c, 1));
            if (in->op == POP_WMMA_LOAD) {
                if (in->wmma_kind == 0) {
                    for (int j = 0; j < 4; j++) {
                        uint32_t sh = nid(c), col = nid(c);
                        emit_arith(c, b, S_Shl, c->t_u32, sh, bb,
                                   const_u32(c, 2));
                        emit_arith(c, b, S_IAdd, c->t_u32, col, sh,
                                   const_u32(c, (uint32_t)j));
                        uint32_t idx = wmma_index(c, b, rr, col, stride,
                                                   in->wmma_row);
                        uint32_t off = wmma_byte_off(c, b, base, idx, 2);
                        uint32_t v = wmma_ld_f32(c, b, shvar, glvar,
                                                 binding, off);
                        if (c->failed)
                            return;
                        def_val(c, in->wregs[0][j], PTY_U32, v);
                    }
                } else if (in->wmma_kind == 1) {
                    for (int j = 0; j < 4; j++) {
                        uint32_t sh = nid(c), row = nid(c);
                        emit_arith(c, b, S_Shl, c->t_u32, sh, bb,
                                   const_u32(c, 2));
                        emit_arith(c, b, S_IAdd, c->t_u32, row, sh,
                                   const_u32(c, (uint32_t)j));
                        uint32_t idx = wmma_index(c, b, row, rr, stride,
                                                   in->wmma_row);
                        uint32_t off = wmma_byte_off(c, b, base, idx, 2);
                        uint32_t v = wmma_ld_f32(c, b, shvar, glvar,
                                                 binding, off);
                        if (c->failed)
                            return;
                        def_val(c, in->wregs[0][j], PTY_U32, v);
                    }
                } else {
                    for (int j = 0; j < 8; j++) {
                        uint32_t sh = nid(c), col = nid(c);
                        emit_arith(c, b, S_Shl, c->t_u32, sh, bb,
                                   const_u32(c, 3));
                        emit_arith(c, b, S_IAdd, c->t_u32, col, sh,
                                   const_u32(c, (uint32_t)j));
                        uint32_t idx = wmma_index(c, b, rr, col, stride,
                                                   in->wmma_row);
                        uint32_t off = wmma_byte_off(c, b, base, idx, 2);
                        uint32_t v = wmma_ld_f32(c, b, shvar, glvar,
                                                 binding, off);
                        if (c->failed)
                            return;
                        uint32_t f = nid(c);
                        put(c, b, (4u << 16) | S_Bitcast);
                        put(c, b, c->t_f32);
                        put(c, b, f);
                        put(c, b, v);
                        def_val(c, in->wregs[0][j], PTY_F32, f);
                    }
                }
            } else {
                for (int j = 0; j < 8; j++) {
                    uint32_t sh = nid(c), col = nid(c);
                    emit_arith(c, b, S_Shl, c->t_u32, sh, bb,
                               const_u32(c, 3));
                    emit_arith(c, b, S_IAdd, c->t_u32, col, sh,
                               const_u32(c, (uint32_t)j));
                    uint32_t idx = wmma_index(c, b, rr, col, stride,
                                               in->wmma_row);
                    uint32_t off = wmma_byte_off(c, b, base, idx, 2);
                    reg_t *dr = find_reg(c, in->wregs[0][j]);
                    if (!dr || !dr->defined) {
                        failf(c, line, "wmma.store register has no value");
                        return;
                    }
                    uint32_t v = nid(c);
                    put(c, b, (4u << 16) | S_Bitcast);
                    put(c, b, c->t_u32);
                    put(c, b, v);
                    put(c, b, dr->cur);
                    wmma_st_f32(c, b, shvar, glvar, binding, off, v);
                    if (c->failed)
                        return;
                }
            }
            return;
        }
        if (in->wmma_shape == 1) {
            // m8n32k16: lane r=l&7, col block g=l>>3.
            // A[r][4g+j] in regs 0..3 (low); B column l with 16 rows
            // packed in 8 regs; C/D row r, cols 8g..8g+7.
            uint32_t rr = nid(c), gg = nid(c);
            emit_arith(c, b, S_And, c->t_u32, rr, lane, const_u32(c, 7));
            emit_arith(c, b, S_ShrL, c->t_u32, gg, lane, const_u32(c, 3));
            if (in->mma_elem == 2) {
                if (in->op == POP_WMMA_LOAD) {
                    if (in->wmma_kind == 0) {
                        uint32_t col = nid(c);
                        emit_arith(c, b, S_Shl, c->t_u32, col, gg,
                                   const_u32(c, 2));
                        uint32_t idx = wmma_index(
                            c, b, rr, col, stride, in->wmma_row);
                        uint32_t off = wmma_byte_off(c, b, base, idx, 0);
                        uint32_t v = wmma_ld_bytes4(c, b, shvar, glvar,
                                                    binding, off);
                        if (c->failed)
                            return;
                        def_val(c, in->wregs[0][0], PTY_U32, v);
                    } else if (in->wmma_kind == 1) {
                        for (int j = 0; j < 4; j++) {
                            uint32_t row = nid(c);
                            emit_arith(c, b, S_Shl, c->t_u32, row,
                                       const_u32(c, (uint32_t)j),
                                       const_u32(c, 2));
                            uint32_t idx = wmma_index(
                                c, b, row, lane, stride, in->wmma_row);
                            uint32_t off =
                                wmma_byte_off(c, b, base, idx, 0);
                            uint32_t v = wmma_ld_bytes4(
                                c, b, shvar, glvar, binding, off);
                            if (c->failed)
                                return;
                            def_val(c, in->wregs[0][j], PTY_U32, v);
                        }
                    } else {
                        for (int i = 0; i < 8; i++) {
                            uint32_t col = nid(c), col2 = nid(c);
                            emit_arith(c, b, S_Shl, c->t_u32, col, gg,
                                       const_u32(c, 3));
                            emit_arith(c, b, S_IAdd, c->t_u32, col2, col,
                                       const_u32(c, (uint32_t)i));
                            uint32_t idx = wmma_index(
                                c, b, rr, col2, stride, in->wmma_row);
                            uint32_t off =
                                wmma_byte_off(c, b, base, idx, 2);
                            uint32_t v = wmma_ld_f32(
                                c, b, shvar, glvar, binding, off);
                            if (c->failed)
                                return;
                            def_val(c, in->wregs[0][i], PTY_U32, v);
                        }
                    }
                } else {
                    for (int i = 0; i < 8; i++) {
                        uint32_t col = nid(c), col2 = nid(c);
                        emit_arith(c, b, S_Shl, c->t_u32, col, gg,
                                   const_u32(c, 3));
                        emit_arith(c, b, S_IAdd, c->t_u32, col2, col,
                                   const_u32(c, (uint32_t)i));
                        uint32_t idx = wmma_index(
                            c, b, rr, col2, stride, in->wmma_row);
                        uint32_t off = wmma_byte_off(c, b, base, idx, 2);
                        reg_t *dr = find_reg(c, in->wregs[0][i]);
                        if (!dr || !dr->defined) {
                            failf(c, line, "wmma.store register has no value");
                            return;
                        }
                        uint32_t v = nid(c);
                        put(c, b, (4u << 16) | S_Bitcast);
                        put(c, b, c->t_u32);
                        put(c, b, v);
                        put(c, b, dr->cur);
                        wmma_st_f32(c, b, shvar, glvar, binding, off, v);
                        if (c->failed)
                            return;
                    }
                }
                return;
            }
            if (in->op == POP_WMMA_LOAD) {
                if (in->wmma_kind == 0) {
                    // Pack the four A columns owned by this lane into
                    // two registers, two K values per register.
                    for (int j = 0; j < 2; j++) {
                        uint32_t sh = nid(c), col = nid(c);
                        emit_arith(c, b, S_Shl, c->t_u32, sh, gg,
                                   const_u32(c, 2));
                        emit_arith(c, b, S_IAdd, c->t_u32, col, sh,
                                   const_u32(c, (uint32_t)(2 * j)));
                        uint32_t idx0 = wmma_index(
                            c, b, rr, col, stride, in->wmma_row);
                        uint32_t off0 =
                            wmma_byte_off(c, b, base, idx0, 1);
                        uint32_t v0 = wmma_ld_half(c, b, shvar, glvar,
                                                   binding, off0);
                        uint32_t col1 = nid(c);
                        emit_arith(c, b, S_IAdd, c->t_u32, col1, col,
                                   const_u32(c, 1));
                        uint32_t idx1 = wmma_index(
                            c, b, rr, col1, stride, in->wmma_row);
                        uint32_t off1 =
                            wmma_byte_off(c, b, base, idx1, 1);
                        uint32_t v1 = wmma_ld_half(c, b, shvar, glvar,
                                                   binding, off1);
                        if (c->failed)
                            return;
                        uint32_t shh = nid(c), pk = nid(c);
                        emit_arith(c, b, S_Shl, c->t_u32, shh, v1,
                                   const_u32(c, 16));
                        emit_arith(c, b, S_Or, c->t_u32, pk, v0, shh);
                        def_val(c, in->wregs[0][j], PTY_F16X2, pk);
                    }
                } else if (in->wmma_kind == 1) {
                    for (int q = 0; q < 8; q++) {
                        uint32_t i0 = wmma_index(
                            c, b, const_u32(c, (uint32_t)(2 * q)), lane,
                            stride, in->wmma_row);
                        uint32_t o0 = wmma_byte_off(c, b, base, i0, 1);
                        uint32_t v0 = wmma_ld_half(c, b, shvar, glvar,
                                                   binding, o0);
                        uint32_t i1 = wmma_index(
                            c, b, const_u32(c, (uint32_t)(2 * q + 1)), lane,
                            stride, in->wmma_row);
                        uint32_t o1 = wmma_byte_off(c, b, base, i1, 1);
                        uint32_t v1 = wmma_ld_half(c, b, shvar, glvar,
                                                   binding, o1);
                        if (c->failed)
                            return;
                        uint32_t sh = nid(c), pk = nid(c);
                        emit_arith(c, b, S_Shl, c->t_u32, sh, v1,
                                   const_u32(c, 16));
                        emit_arith(c, b, S_Or, c->t_u32, pk, v0, sh);
                        def_val(c, in->wregs[0][q], PTY_F16X2, pk);
                    }
                } else {
                    for (int j = 0; j < 8; j++) {
                        uint32_t sh = nid(c), col = nid(c);
                        emit_arith(c, b, S_Shl, c->t_u32, sh, gg,
                                   const_u32(c, 3));
                        emit_arith(c, b, S_IAdd, c->t_u32, col, sh,
                                   const_u32(c, (uint32_t)j));
                        uint32_t idx = wmma_index(c, b, rr, col, stride,
                                                   in->wmma_row);
                        uint32_t off = wmma_byte_off(c, b, base, idx, 2);
                        uint32_t v = wmma_ld_f32(c, b, shvar, glvar,
                                                 binding, off);
                        if (c->failed)
                            return;
                        uint32_t f = nid(c);
                        put(c, b, (4u << 16) | S_Bitcast);
                        put(c, b, c->t_f32);
                        put(c, b, f);
                        put(c, b, v);
                        def_val(c, in->wregs[0][j], PTY_F32, f);
                    }
                }
            } else {
                for (int j = 0; j < 8; j++) {
                    uint32_t sh = nid(c), col = nid(c);
                    emit_arith(c, b, S_Shl, c->t_u32, sh, gg,
                               const_u32(c, 3));
                    emit_arith(c, b, S_IAdd, c->t_u32, col, sh,
                               const_u32(c, (uint32_t)j));
                    uint32_t idx = wmma_index(c, b, rr, col, stride,
                                               in->wmma_row);
                    uint32_t off = wmma_byte_off(c, b, base, idx, 2);
                    reg_t *dr = find_reg(c, in->wregs[0][j]);
                    if (!dr || !dr->defined) {
                        failf(c, line, "wmma.store register has no value");
                        return;
                    }
                    uint32_t v = nid(c);
                    put(c, b, (4u << 16) | S_Bitcast);
                    put(c, b, c->t_u32);
                    put(c, b, v);
                    put(c, b, dr->cur);
                    wmma_st_f32(c, b, shvar, glvar, binding, off, v);
                    if (c->failed)
                        return;
                }
            }
            return;
        }
        if (in->wmma_shape == 2) {
            // m32n8k16: lane r=l; A row r packed pairs, B column l&7
            // with rows 4*(l>>3)..+3 in regs 0..3, C/D row r cols 0..7.
            uint32_t gg = nid(c), cc = nid(c);
            emit_arith(c, b, S_ShrL, c->t_u32, gg, lane, const_u32(c, 3));
            emit_arith(c, b, S_And, c->t_u32, cc, lane, const_u32(c, 7));
            if (in->mma_elem == 2) {
                if (in->op == POP_WMMA_LOAD) {
                    if (in->wmma_kind == 0) {
                        for (int j = 0; j < 4; j++) {
                            uint32_t col = nid(c);
                            emit_arith(c, b, S_Shl, c->t_u32, col,
                                       const_u32(c, (uint32_t)j),
                                       const_u32(c, 2));
                            uint32_t idx = wmma_index(
                                c, b, lane, col, stride, in->wmma_row);
                            uint32_t off =
                                wmma_byte_off(c, b, base, idx, 0);
                            uint32_t v = wmma_ld_bytes4(
                                c, b, shvar, glvar, binding, off);
                            if (c->failed)
                                return;
                            def_val(c, in->wregs[0][j], PTY_U32, v);
                        }
                    } else if (in->wmma_kind == 1) {
                        uint32_t row = nid(c);
                        emit_arith(c, b, S_Shl, c->t_u32, row, gg,
                                   const_u32(c, 2));
                        uint32_t idx = wmma_index(
                            c, b, row, cc, stride, in->wmma_row);
                        uint32_t off = wmma_byte_off(c, b, base, idx, 0);
                        uint32_t v = wmma_ld_bytes4(c, b, shvar, glvar,
                                                    binding, off);
                        if (c->failed)
                            return;
                        def_val(c, in->wregs[0][0], PTY_U32, v);
                    } else {
                        for (int i = 0; i < 8; i++) {
                            uint32_t idx = wmma_index(
                                c, b, lane, const_u32(c, (uint32_t)i),
                                stride, in->wmma_row);
                            uint32_t off =
                                wmma_byte_off(c, b, base, idx, 2);
                            uint32_t v = wmma_ld_f32(
                                c, b, shvar, glvar, binding, off);
                            if (c->failed)
                                return;
                            def_val(c, in->wregs[0][i], PTY_U32, v);
                        }
                    }
                } else {
                    for (int i = 0; i < 8; i++) {
                        uint32_t idx = wmma_index(
                            c, b, lane, const_u32(c, (uint32_t)i),
                            stride, in->wmma_row);
                        uint32_t off = wmma_byte_off(c, b, base, idx, 2);
                        reg_t *dr = find_reg(c, in->wregs[0][i]);
                        if (!dr || !dr->defined) {
                            failf(c, line, "wmma.store register has no value");
                            return;
                        }
                        uint32_t v = nid(c);
                        put(c, b, (4u << 16) | S_Bitcast);
                        put(c, b, c->t_u32);
                        put(c, b, v);
                        put(c, b, dr->cur);
                        wmma_st_f32(c, b, shvar, glvar, binding, off, v);
                        if (c->failed)
                            return;
                    }
                }
                return;
            }
            if (in->op == POP_WMMA_LOAD) {
                if (in->wmma_kind == 0) {
                    for (int q = 0; q < 8; q++) {
                        uint32_t i0 = wmma_index(
                            c, b, lane, const_u32(c, (uint32_t)(2 * q)),
                            stride, in->wmma_row);
                        uint32_t o0 = wmma_byte_off(c, b, base, i0, 1);
                        uint32_t v0 = wmma_ld_half(c, b, shvar, glvar,
                                                   binding, o0);
                        uint32_t i1 = wmma_index(
                            c, b, lane, const_u32(c, (uint32_t)(2 * q + 1)),
                            stride, in->wmma_row);
                        uint32_t o1 = wmma_byte_off(c, b, base, i1, 1);
                        uint32_t v1 = wmma_ld_half(c, b, shvar, glvar,
                                                   binding, o1);
                        if (c->failed)
                            return;
                        uint32_t sh = nid(c), pk = nid(c);
                        emit_arith(c, b, S_Shl, c->t_u32, sh, v1,
                                   const_u32(c, 16));
                        emit_arith(c, b, S_Or, c->t_u32, pk, v0, sh);
                        def_val(c, in->wregs[0][q], PTY_F16X2, pk);
                    }
                } else if (in->wmma_kind == 1) {
                    // Pack two consecutive K rows per register so the
                    // fallback needs one B shuffle per K pair.
                    for (int j = 0; j < 2; j++) {
                        uint32_t sh = nid(c), row = nid(c);
                        emit_arith(c, b, S_Shl, c->t_u32, sh, gg,
                                   const_u32(c, 2));
                        emit_arith(c, b, S_IAdd, c->t_u32, row, sh,
                                   const_u32(c, (uint32_t)(2 * j)));
                        uint32_t idx0 = wmma_index(
                            c, b, row, cc, stride, in->wmma_row);
                        uint32_t off0 =
                            wmma_byte_off(c, b, base, idx0, 1);
                        uint32_t v0 = wmma_ld_half(c, b, shvar, glvar,
                                                   binding, off0);
                        uint32_t row1 = nid(c);
                        emit_arith(c, b, S_IAdd, c->t_u32, row1, row,
                                   const_u32(c, 1));
                        uint32_t idx1 = wmma_index(
                            c, b, row1, cc, stride, in->wmma_row);
                        uint32_t off1 =
                            wmma_byte_off(c, b, base, idx1, 1);
                        uint32_t v1 = wmma_ld_half(c, b, shvar, glvar,
                                                   binding, off1);
                        if (c->failed)
                            return;
                        uint32_t shh = nid(c), pk = nid(c);
                        emit_arith(c, b, S_Shl, c->t_u32, shh, v1,
                                   const_u32(c, 16));
                        emit_arith(c, b, S_Or, c->t_u32, pk, v0, shh);
                        def_val(c, in->wregs[0][j], PTY_F16X2, pk);
                    }
                } else {
                    for (int j = 0; j < 8; j++) {
                        uint32_t idx = wmma_index(
                            c, b, lane, const_u32(c, (uint32_t)j), stride,
                            in->wmma_row);
                        uint32_t off = wmma_byte_off(c, b, base, idx, 2);
                        uint32_t v = wmma_ld_f32(c, b, shvar, glvar,
                                                 binding, off);
                        if (c->failed)
                            return;
                        uint32_t f = nid(c);
                        put(c, b, (4u << 16) | S_Bitcast);
                        put(c, b, c->t_f32);
                        put(c, b, f);
                        put(c, b, v);
                        def_val(c, in->wregs[0][j], PTY_F32, f);
                    }
                }
            } else {
                for (int j = 0; j < 8; j++) {
                    uint32_t idx = wmma_index(
                        c, b, lane, const_u32(c, (uint32_t)j), stride,
                        in->wmma_row);
                    uint32_t off = wmma_byte_off(c, b, base, idx, 2);
                    reg_t *dr = find_reg(c, in->wregs[0][j]);
                    if (!dr || !dr->defined) {
                        failf(c, line, "wmma.store register has no value");
                        return;
                    }
                    uint32_t v = nid(c);
                    put(c, b, (4u << 16) | S_Bitcast);
                    put(c, b, c->t_u32);
                    put(c, b, v);
                    put(c, b, dr->cur);
                    wmma_st_f32(c, b, shvar, glvar, binding, off, v);
                    if (c->failed)
                        return;
                }
            }
            return;
        }
        // 16-bit A/B fragments and f16-accumulator C/D fragments are
        // four b32 regs holding two elements each.  Packing f16 the same
        // way as bf16 lets the fallback extract two K values per shuffle.
        int pair_load = in->op == POP_WMMA_LOAD && !is_f32 &&
                        (in->wmma_kind <= 1 ||
                         (in->wmma_kind == 2 && in->mma_f16acc));
        int nfrag = pair_load ? 4
                    : (in->op == POP_WMMA_STORE && in->mma_f16acc) ? 4
                                                                   : 8;
        for (int i = 0; i < nfrag; i++) {
            if (pair_load) {
                uint32_t vv[2];
                for (int q = 0; q < 2; q++) {
                    int e = 2 * i + q;
                    uint32_t prow, pcol;
                    if (in->wmma_kind == 1) {
                        prow = nid(c);
                        emit_arith(c, b, S_Shl, c->t_u32, prow, half,
                                   const_u32(c, 3));
                        uint32_t ri = const_u32(c, (uint32_t)e);
                        uint32_t rr = nid(c);
                        emit_arith(c, b, S_IAdd, c->t_u32, rr, prow, ri);
                        prow = rr;
                        pcol = r;
                    } else {
                        prow = r;
                        pcol = nid(c);
                        emit_arith(c, b, S_IAdd, c->t_u32, pcol, c0,
                                   const_u32(c, (uint32_t)e));
                    }
                    uint32_t idx = wmma_index(c, b, prow, pcol, stride,
                                               in->wmma_row);
                    uint32_t off = wmma_byte_off(c, b, base, idx, eshift);
                    if (c->failed)
                        return;
                    vv[q] = shvar >= 0
                                ? narrow_ld_shared(c, b, shvar, off, 16)
                                : glvar >= 0
                                      ? narrow_ld_glvar(c, b, glvar, off, 16)
                                      : narrow_ld_global(c, b, binding, off, 16);
                    if (c->failed)
                        return;
                }
                uint32_t sh = nid(c), pk = nid(c);
                emit_arith(c, b, S_Shl, c->t_u32, sh, vv[1],
                           const_u32(c, 16));
                emit_arith(c, b, S_Or, c->t_u32, pk, vv[0], sh);
                def_val(c, in->wregs[0][i], PTY_F16X2, pk);
                continue;
            }
            if (in->op == POP_WMMA_STORE && in->mma_f16acc) {
                reg_t *pr = find_reg(c, in->wregs[0][i]);
                if (!pr || !pr->defined) {
                    failf(c, line, "wmma.store register has no value");
                    return;
                }
                for (int q = 0; q < 2; q++) {
                    int e = 2 * i + q;
                    uint32_t half = nid(c);
                    if (q)
                        emit_arith(c, b, S_ShrL, c->t_u32, half, pr->cur,
                                   const_u32(c, 16));
                    else
                        emit_arith(c, b, S_And, c->t_u32, half, pr->cur,
                                   const_u32(c, 0xffffu));
                    uint32_t col = nid(c);
                    emit_arith(c, b, S_IAdd, c->t_u32, col, c0,
                               const_u32(c, (uint32_t)e));
                    uint32_t idx =
                        wmma_index(c, b, r, col, stride, in->wmma_row);
                    uint32_t off = wmma_byte_off(c, b, base, idx, 1);
                    if (shvar >= 0)
                        narrow_st_shared(c, b, shvar, off, half, 16);
                    else
                        narrow_st_global(c, b, binding, off, half, 16);
                    if (c->failed)
                        return;
                }
                continue;
            }
            uint32_t row, col;
            if (in->wmma_kind == 1) {
                row = nid(c);
                emit_arith(c, b, S_Shl, c->t_u32, row, half,
                           const_u32(c, 3));
                uint32_t ri = const_u32(c, (uint32_t)i), rr = nid(c);
                emit_arith(c, b, S_IAdd, c->t_u32, rr, row, ri);
                row = rr;
                col = r;
            } else {
                row = r;
                col = nid(c);
                emit_arith(c, b, S_IAdd, c->t_u32, col, c0,
                           const_u32(c, (uint32_t)i));
            }
            uint32_t idx = wmma_index(c, b, row, col, stride,
                                       in->wmma_row);
            uint32_t off = wmma_byte_off(c, b, base, idx, eshift);
            if (c->failed)
                return;
            if (in->op == POP_WMMA_LOAD) {
                if (is_f32) {
                    uint32_t ptr = shvar >= 0
                                       ? sh_addr_ptr(c, b, shvar, off)
                                       : glvar >= 0
                                             ? gl_addr_ptr(c, b, glvar, off)
                                             : addr_ptr(c, b, binding, off);
                    uint32_t v = nid(c);
                    emit_load(c, b, c->t_u32, v, ptr);
                    uint32_t f = nid(c);
                    put(c, b, (4u << 16) | S_Bitcast);
                    put(c, b, c->t_f32);
                    put(c, b, f);
                    put(c, b, v);
                    def_val(c, in->wregs[0][i], PTY_F32, f);
                } else {
                    uint32_t v = shvar >= 0
                                     ? narrow_ld_shared(c, b, shvar, off, 16)
                                     : glvar >= 0
                                           ? narrow_ld_glvar(c, b, glvar, off, 16)
                                           : narrow_ld_global(c, b, binding, off, 16);
                    if (c->failed)
                        return;
                    def_val(c, in->wregs[0][i], PTY_F16X2, v);
                }
            } else {
                reg_t *rr = find_reg(c, in->wregs[0][i]);
                if (!rr || !rr->defined) {
                    failf(c, line, "wmma.store register has no value");
                    return;
                }
                uint32_t v = nid(c);
                put(c, b, (4u << 16) | S_Bitcast);
                put(c, b, c->t_u32);
                put(c, b, v);
                put(c, b, rr->cur);
                if (is_f32) {
                    uint32_t ptr = shvar >= 0
                                       ? sh_addr_ptr(c, b, shvar, off)
                                       : glvar >= 0
                                             ? gl_addr_ptr(c, b, glvar, off)
                                             : addr_ptr(c, b, binding, off);
                    emit_store(c, b, ptr, v);
                } else if (shvar >= 0) {
                    narrow_st_shared(c, b, shvar, off, v, 16);
                } else {
                    narrow_st_global(c, b, binding, off, v, 16);
                }
                if (c->failed)
                    return;
            }
        }
        return;
    }
    case POP_WMMA_MMA: {
        int is_bf16 = in->mma_elem == 1;
        int f16acc = in->mma_f16acc;
        if (in->wmma_shape == 0) {
            reg_t *ra = find_reg(c, in->wregs[1][0]);
            reg_t *rb = find_reg(c, in->wregs[2][0]);
            reg_t *rc = find_reg(c, in->wregs[3][0]);
            reg_t *rd = find_reg(c, in->wregs[0][0]);
            if (ra && rb && rc && rd && ra->is_coopmat && rb->is_coopmat &&
                rc->is_coopmat && rd->is_coopmat) {
                uint32_t tyC = rd->coopmat_ty;
                uint32_t c_id;
                if (rc->defined && rc->cur_is_coopmat) {
                    c_id = rc->cur;
                } else {
                    c_id = nid(c);
                    put(c, &c->types, (3u << 16) | 46); // OpConstantNull
                    put(c, &c->types, tyC);
                    put(c, &c->types, c_id);
                }
                uint32_t id = nid(c);
                put(c, b, (6u << 16) | 4459); // OpCooperativeMatrixMulAddKHR
                put(c, b, tyC);
                put(c, b, id);
                put(c, b, ra->cur);
                put(c, b, rb->cur);
                put(c, b, c_id);
                rd->defined = 1;
                rd->cur = id;
                rd->cur_ty = PTY_U32;
                rd->cur_is_coopmat = 1;
                c->uses_coopmat = 1;
                return;
            }
        }
        uint32_t sub = wmma_subid(c, b);
        uint32_t lane = nid(c), wbase = nid(c);
        emit_arith(c, b, S_And, c->t_u32, lane, sub, const_u32(c, 31));
        emit_arith(c, b, S_And, c->t_u32, wbase, sub,
                   const_u32(c, 0xffffffe0u));
        if (in->mma_elem == 2 && in->wmma_shape == 5) {
            uint32_t row = nid(c), gg = nid(c), c0 = nid(c);
            emit_arith(c, b, S_ShrL, c->t_u32, row, lane,
                       const_u32(c, 2));
            emit_arith(c, b, S_And, c->t_u32, gg, lane,
                       const_u32(c, 3));
            emit_arith(c, b, S_Shl, c->t_u32, c0, gg,
                       const_u32(c, 1));
            uint32_t Areg, Breg, Creg[2], Aword[4];
            reg_t *ar = find_reg(c, in->wregs[1][0]);
            reg_t *br = find_reg(c, in->wregs[2][0]);
            if (!ar || !ar->defined || !br || !br->defined) {
                failf(c, line, "wmma.mma 1-bit fragment has no value");
                return;
            }
            Areg = ar->cur;
            Breg = br->cur;
            for (int i = 0; i < 2; i++) {
                reg_t *cr = find_reg(c, in->wregs[3][i]);
                if (!cr || !cr->defined) {
                    failf(c, line, "wmma.mma C fragment has no value");
                    return;
                }
                Creg[i] = cr->cur;
            }
            for (int g2 = 0; g2 < 4; g2++) {
                uint32_t owner = nid(c), base = nid(c);
                emit_arith(c, b, S_Shl, c->t_u32, base, row,
                           const_u32(c, 2));
                emit_arith(c, b, S_IAdd, c->t_u32, owner, base,
                           const_u32(c, (uint32_t)g2));
                Aword[g2] = wmma_shfl(c, b, Areg, owner, wbase);
            }
            for (int i = 0; i < 2; i++) {
                uint32_t col = nid(c), acc = Creg[i];
                emit_arith(c, b, S_IAdd, c->t_u32, col, c0,
                           const_u32(c, (uint32_t)i));
                for (int g2 = 0; g2 < 4; g2++) {
                    uint32_t owner = nid(c), base = nid(c), bits;
                    emit_arith(c, b, S_Shl, c->t_u32, base, col,
                               const_u32(c, 2));
                    emit_arith(c, b, S_IAdd, c->t_u32, owner, base,
                               const_u32(c, (uint32_t)g2));
                    uint32_t bword = wmma_shfl(c, b, Breg, owner, wbase);
                    bits = nid(c);
                    emit_arith(c, b,
                               in->mma_b1_xor ? S_Xor : S_And,
                               c->t_u32, bits, Aword[g2], bword);
                    uint32_t pc = popc32(c, b, bits);
                    uint32_t sum = nid(c);
                    emit_arith(c, b, S_IAdd, c->t_u32, sum, acc, pc);
                    acc = sum;
                }
                def_val(c, in->wregs[0][i], PTY_U32, acc);
            }
            return;
        }
        if (in->mma_elem == 2 && in->wmma_shape == 4) {
            uint32_t row = nid(c), gg = nid(c), c0 = nid(c);
            emit_arith(c, b, S_ShrL, c->t_u32, row, lane,
                       const_u32(c, 2));
            emit_arith(c, b, S_And, c->t_u32, gg, lane,
                       const_u32(c, 3));
            emit_arith(c, b, S_Shl, c->t_u32, c0, gg,
                       const_u32(c, 1));
            uint32_t Areg, Breg, Creg[2], Awords[4];
            reg_t *ar = find_reg(c, in->wregs[1][0]);
            reg_t *br = find_reg(c, in->wregs[2][0]);
            if (!ar || !ar->defined || !br || !br->defined) {
                failf(c, line, "wmma.mma 4-bit fragment has no value");
                return;
            }
            Areg = ar->cur;
            Breg = br->cur;
            for (int i = 0; i < 2; i++) {
                reg_t *cr = find_reg(c, in->wregs[3][i]);
                if (!cr || !cr->defined) {
                    failf(c, line, "wmma.mma C fragment has no value");
                    return;
                }
                Creg[i] = cr->cur;
            }
            for (int g2 = 0; g2 < 4; g2++) {
                uint32_t owner = nid(c), base = nid(c);
                emit_arith(c, b, S_Shl, c->t_u32, base, row,
                           const_u32(c, 2));
                emit_arith(c, b, S_IAdd, c->t_u32, owner, base,
                           const_u32(c, (uint32_t)g2));
                Awords[g2] = wmma_shfl(c, b, Areg, owner, wbase);
            }
            for (int i = 0; i < 2; i++) {
                uint32_t col = nid(c), acc = Creg[i];
                emit_arith(c, b, S_IAdd, c->t_u32, col, c0,
                           const_u32(c, (uint32_t)i));
                for (int g2 = 0; g2 < 4; g2++) {
                    uint32_t owner = nid(c), base = nid(c), bword;
                    emit_arith(c, b, S_Shl, c->t_u32, base, col,
                               const_u32(c, 2));
                    emit_arith(c, b, S_IAdd, c->t_u32, owner, base,
                               const_u32(c, (uint32_t)g2));
                    bword = wmma_shfl(c, b, Breg, owner, wbase);
                    uint32_t ab[2], bb[2];
                    unpack4_to_bytes(c, b, Awords[g2], in->mma_asign, ab);
                    unpack4_to_bytes(c, b, bword, in->mma_bsign, bb);
                    for (int h = 0; h < 2; h++) {
                        uint32_t dot = dot4_packed(
                            c, b, in->mma_asign, in->mma_bsign, ab[h],
                            bb[h]);
                        uint32_t sum = nid(c);
                        emit_arith(c, b, S_IAdd, c->t_u32, sum, acc, dot);
                        acc = sum;
                    }
                }
                def_val(c, in->wregs[0][i], PTY_U32, acc);
            }
            return;
        }
        if (in->mma_elem == 2 && in->wmma_shape != 3) {
            int na = in->wmma_shape == 0 ? 2
                     : in->wmma_shape == 1 ? 1
                                           : 4;
            int nb = in->wmma_shape == 0 ? 2
                     : in->wmma_shape == 1 ? 4
                                           : 1;
            uint32_t Areg[4], Breg[4], Creg[8], Aword[4];
            for (int i = 0; i < na; i++) {
                reg_t *ar = find_reg(c, in->wregs[1][i]);
                if (!ar || !ar->defined) {
                    failf(c, line, "wmma.mma A fragment has no value");
                    return;
                }
                Areg[i] = ar->cur;
            }
            for (int i = 0; i < nb; i++) {
                reg_t *br = find_reg(c, in->wregs[2][i]);
                if (!br || !br->defined) {
                    failf(c, line, "wmma.mma B fragment has no value");
                    return;
                }
                Breg[i] = br->cur;
            }
            for (int i = 0; i < 8; i++) {
                reg_t *cr = find_reg(c, in->wregs[3][i]);
                if (!cr || !cr->defined) {
                    failf(c, line, "wmma.mma C fragment has no value");
                    return;
                }
                Creg[i] = cr->cur;
            }
            uint32_t row = nid(c), c0 = nid(c);
            if (in->wmma_shape == 0) {
                uint32_t h = nid(c);
                emit_arith(c, b, S_ShrL, c->t_u32, row, lane,
                           const_u32(c, 1));
                emit_arith(c, b, S_And, c->t_u32, h, lane,
                           const_u32(c, 1));
                emit_arith(c, b, S_Shl, c->t_u32, c0, h,
                           const_u32(c, 3));
            } else if (in->wmma_shape == 1) {
                uint32_t g = nid(c);
                emit_arith(c, b, S_And, c->t_u32, row, lane,
                           const_u32(c, 7));
                emit_arith(c, b, S_ShrL, c->t_u32, g, lane,
                           const_u32(c, 3));
                emit_arith(c, b, S_Shl, c->t_u32, c0, g,
                           const_u32(c, 3));
            } else {
                row = lane;
                c0 = const_u32(c, 0);
            }
            for (int g = 0; g < 4; g++) {
                uint32_t owner = nid(c);
                if (in->wmma_shape == 0) {
                    emit_arith(c, b, S_Shl, c->t_u32, owner, row,
                               const_u32(c, 1));
                    if (g >= 2) {
                        uint32_t o2 = nid(c);
                        emit_arith(c, b, S_IAdd, c->t_u32, o2, owner,
                                   const_u32(c, 1));
                        owner = o2;
                    }
                    Aword[g] = wmma_shfl(
                        c, b, Areg[g & 1], owner, wbase);
                } else if (in->wmma_shape == 1) {
                    uint32_t base = nid(c);
                    emit_arith(c, b, S_Shl, c->t_u32, base,
                               const_u32(c, (uint32_t)g),
                               const_u32(c, 3));
                    emit_arith(c, b, S_IAdd, c->t_u32, owner, base, row);
                    Aword[g] = wmma_shfl(c, b, Areg[0], owner, wbase);
                } else {
                    Aword[g] = Areg[g];
                }
            }
            for (int i = 0; i < 8; i++) {
                uint32_t col = nid(c), acc = Creg[i];
                emit_arith(c, b, S_IAdd, c->t_u32, col, c0,
                           const_u32(c, (uint32_t)i));
                for (int g = 0; g < 4; g++) {
                    uint32_t owner = nid(c), bw;
                    if (in->wmma_shape == 0) {
                        emit_arith(c, b, S_Shl, c->t_u32, owner, col,
                                   const_u32(c, 1));
                        if (g >= 2) {
                            uint32_t o2 = nid(c);
                            emit_arith(c, b, S_IAdd, c->t_u32, o2, owner,
                                       const_u32(c, 1));
                            owner = o2;
                        }
                        bw = wmma_shfl(c, b, Breg[g & 1], owner, wbase);
                    } else if (in->wmma_shape == 1) {
                        bw = wmma_shfl(c, b, Breg[g], col, wbase);
                    } else {
                        uint32_t base = nid(c);
                        emit_arith(c, b, S_Shl, c->t_u32, base,
                                   const_u32(c, (uint32_t)g),
                                   const_u32(c, 3));
                        emit_arith(c, b, S_IAdd, c->t_u32, owner, base,
                                   col);
                        bw = wmma_shfl(c, b, Breg[0], owner, wbase);
                    }
                    uint32_t dot = dot4_packed(
                        c, b, in->mma_asign, in->mma_bsign, Aword[g], bw);
                    uint32_t sum = nid(c);
                    emit_arith(c, b, S_IAdd, c->t_u32, sum, acc, dot);
                    acc = sum;
                    if (c->failed)
                        return;
                }
                def_val(c, in->wregs[0][i], PTY_U32, acc);
            }
            return;
        }
        uint32_t r = nid(c), half = nid(c), c0 = nid(c);
        emit_arith(c, b, S_ShrL, c->t_u32, r, lane, const_u32(c, 1));
        emit_arith(c, b, S_And, c->t_u32, half, lane, const_u32(c, 1));
        emit_arith(c, b, S_Shl, c->t_u32, c0, half, const_u32(c, 3));
        if (in->wmma_shape == 3) {
            // m16n16k8 tf32: lane r=l>>1, col block g=l&1.
            for (int m = 0; m < 8; m++) {
                if (check_dst(c, in->wregs[0][m], PTY_F32, line) != 0)
                    return;
            }
            uint32_t aReg[4], bReg[4], cReg[8];
            for (int j = 0; j < 4; j++) {
                reg_t *r = find_reg(c, in->wregs[1][j]);
                if (!r || !r->defined) {
                    failf(c, line, "wmma.mma A fragment has no value");
                    return;
                }
                aReg[j] = r->cur;
                reg_t *br = find_reg(c, in->wregs[2][j]);
                if (!br || !br->defined) {
                    failf(c, line, "wmma.mma B fragment has no value");
                    return;
                }
                bReg[j] = br->cur;
            }
            for (int j = 0; j < 8; j++) {
                reg_t *cr = find_reg(c, in->wregs[3][j]);
                if (!cr || !cr->defined) {
                    failf(c, line, "wmma.mma C fragment has no value");
                    return;
                }
                if (cr->cur_ty != PTY_F32) {
                    uint32_t _f = nid(c);
                    put(c, b, (4u << 16) | S_Bitcast);
                    put(c, b, c->t_f32);
                    put(c, b, _f);
                    put(c, b, cr->cur);
                    cReg[j] = _f;
                } else {
                    cReg[j] = cr->cur;
                }
            }
            uint32_t rr = nid(c), bb = nid(c), c0 = nid(c);
            emit_arith(c, b, S_ShrL, c->t_u32, rr, lane, const_u32(c, 1));
            emit_arith(c, b, S_And, c->t_u32, bb, lane, const_u32(c, 1));
            emit_arith(c, b, S_Shl, c->t_u32, c0, bb, const_u32(c, 3));
            for (int i = 0; i < 8; i++) {
                uint32_t col = nid(c);
                emit_arith(c, b, S_IAdd, c->t_u32, col, c0,
                           const_u32(c, (uint32_t)i));
                uint32_t acc = cReg[i];
                for (int k = 0; k < 8; k++) {
                    uint32_t shl = nid(c), alane = nid(c);
                    emit_arith(c, b, S_Shl, c->t_u32, shl, rr,
                               const_u32(c, 1));
                    emit_arith(c, b, S_IAdd, c->t_u32, alane, shl,
                               const_u32(c, (uint32_t)(k >= 4 ? 1 : 0)));
                    uint32_t aGot = wmma_shfl(c, b, aReg[k & 3], alane,
                                              wbase);
                    uint32_t ab = nid(c), Af = nid(c);
                    emit_arith(c, b, S_And, c->t_u32, ab, aGot,
                               const_u32(c, 0xffffe000u));
                    put(c, b, (4u << 16) | S_Bitcast);
                    put(c, b, c->t_f32);
                    put(c, b, Af);
                    put(c, b, ab);
                    uint32_t shb = nid(c), blane = nid(c);
                    emit_arith(c, b, S_Shl, c->t_u32, shb, col,
                               const_u32(c, 1));
                    emit_arith(c, b, S_IAdd, c->t_u32, blane, shb,
                               const_u32(c, (uint32_t)(k >= 4 ? 1 : 0)));
                    uint32_t bGot = wmma_shfl(c, b, bReg[k & 3], blane,
                                              wbase);
                    uint32_t bbb = nid(c), Bf = nid(c);
                    emit_arith(c, b, S_And, c->t_u32, bbb, bGot,
                               const_u32(c, 0xffffe000u));
                    put(c, b, (4u << 16) | S_Bitcast);
                    put(c, b, c->t_f32);
                    put(c, b, Bf);
                    put(c, b, bbb);
                    if (c->failed)
                        return;
                    uint32_t pr = nid(c);
                    emit_fma_into(c, b, c->t_f32, pr, Af, Bf, acc);
                    acc = pr;
                }
                def_val(c, in->wregs[0][i], PTY_F32, acc);
            }
            return;
        }
        if (in->wmma_shape == 1) {
            // m8n32k16: lane r=l&7, col block g=l>>3. A[r][4g+j] in
            // regs 0..3, B column l packed in 8 regs, D row r 8 cols.
            for (int m = 0; m < 8; m++) {
                if (check_dst(c, in->wregs[0][m], PTY_F32, line) != 0)
                    return;
            }
            uint32_t aReg[2], bReg[8], cReg[8];
            for (int j = 0; j < 2; j++) {
                reg_t *r = find_reg(c, in->wregs[1][j]);
                if (!r || !r->defined) {
                    failf(c, line, "wmma.mma A fragment has no value");
                    return;
                }
                aReg[j] = r->cur;
            }
            for (int j = 0; j < 8; j++) {
                reg_t *r = find_reg(c, in->wregs[2][j]);
                if (!r || !r->defined) {
                    failf(c, line, "wmma.mma B fragment has no value");
                    return;
                }
                bReg[j] = r->cur;
                reg_t *cr = find_reg(c, in->wregs[3][j]);
                if (!cr || !cr->defined) {
                    failf(c, line, "wmma.mma C fragment has no value");
                    return;
                }
                if (cr->cur_ty != PTY_F32) {
                    uint32_t _f = nid(c);
                    put(c, b, (4u << 16) | S_Bitcast);
                    put(c, b, c->t_f32);
                    put(c, b, _f);
                    put(c, b, cr->cur);
                    cReg[j] = _f;
                } else {
                    cReg[j] = cr->cur;
                }
            }
            uint32_t rr = nid(c), gg = nid(c);
            emit_arith(c, b, S_And, c->t_u32, rr, lane, const_u32(c, 7));
            emit_arith(c, b, S_ShrL, c->t_u32, gg, lane, const_u32(c, 3));
            for (int j = 0; j < 8; j++) {
                uint32_t shc = nid(c), ccol = nid(c);
                emit_arith(c, b, S_Shl, c->t_u32, shc, gg,
                           const_u32(c, 3));
                emit_arith(c, b, S_IAdd, c->t_u32, ccol, shc,
                           const_u32(c, (uint32_t)j));
                uint32_t acc = cReg[j];
                for (int p = 0; p < 8; p++) {
                    uint32_t sha = nid(c), alane = nid(c);
                    emit_arith(c, b, S_Shl, c->t_u32, sha,
                               const_u32(c, (uint32_t)(p >> 1)),
                               const_u32(c, 3));
                    emit_arith(c, b, S_IAdd, c->t_u32, alane, sha, rr);
                    uint32_t aGot =
                        wmma_shfl(c, b, aReg[p & 1], alane, wbase);
                    uint32_t ah0 = nid(c), ah1 = nid(c);
                    emit_arith(c, b, S_And, c->t_u32, ah0, aGot,
                               const_u32(c, 0xffffu));
                    emit_arith(c, b, S_ShrL, c->t_u32, ah1, aGot,
                               const_u32(c, 16));
                    uint32_t Af0 = is_bf16
                                       ? bf16_to_f32val(c, b, ah0, line)
                                       : f16_to_f32val(c, b, ah0, line);
                    uint32_t Af1 = is_bf16
                                       ? bf16_to_f32val(c, b, ah1, line)
                                       : f16_to_f32val(c, b, ah1, line);
                    uint32_t bGot =
                        wmma_shfl(c, b, bReg[p], ccol, wbase);
                    uint32_t bh0 = nid(c), bh1 = nid(c);
                    emit_arith(c, b, S_And, c->t_u32, bh0, bGot,
                               const_u32(c, 0xffffu));
                    emit_arith(c, b, S_ShrL, c->t_u32, bh1, bGot,
                               const_u32(c, 16));
                    uint32_t Bf0 = is_bf16
                                       ? bf16_to_f32val(c, b, bh0, line)
                                       : f16_to_f32val(c, b, bh0, line);
                    uint32_t Bf1 = is_bf16
                                       ? bf16_to_f32val(c, b, bh1, line)
                                       : f16_to_f32val(c, b, bh1, line);
                    if (c->failed)
                        return;
                    uint32_t pr = nid(c);
                    emit_fma_into(c, b, c->t_f32, pr, Af0, Bf0, acc);
                    acc = pr;
                    pr = nid(c);
                    emit_fma_into(c, b, c->t_f32, pr, Af1, Bf1, acc);
                    acc = pr;
                }
                def_val(c, in->wregs[0][j], PTY_F32, acc);
            }
            return;
        }
        if (in->wmma_shape == 2) {
            // m32n8k16: lane r=l. A row r packed pairs, B column l&7
            // rows 4*(l>>3)..+3 in regs 0..3, D row r cols 0..7.
            for (int m = 0; m < 8; m++) {
                if (check_dst(c, in->wregs[0][m], PTY_F32, line) != 0)
                    return;
            }
            uint32_t aReg[8], bReg[4], cReg[8];
            for (int j = 0; j < 8; j++) {
                reg_t *r = find_reg(c, in->wregs[1][j]);
                if (!r || !r->defined) {
                    failf(c, line, "wmma.mma A fragment has no value");
                    return;
                }
                aReg[j] = r->cur;
                reg_t *cr = find_reg(c, in->wregs[3][j]);
                if (!cr || !cr->defined) {
                    failf(c, line, "wmma.mma C fragment has no value");
                    return;
                }
                if (cr->cur_ty != PTY_F32) {
                    uint32_t _f = nid(c);
                    put(c, b, (4u << 16) | S_Bitcast);
                    put(c, b, c->t_f32);
                    put(c, b, _f);
                    put(c, b, cr->cur);
                    cReg[j] = _f;
                } else {
                    cReg[j] = cr->cur;
                }
            }
            for (int j = 0; j < 2; j++) {
                reg_t *r = find_reg(c, in->wregs[2][j]);
                if (!r || !r->defined) {
                    failf(c, line, "wmma.mma B fragment has no value");
                    return;
                }
                bReg[j] = r->cur;
            }
            for (int j = 0; j < 8; j++) {
                uint32_t acc = cReg[j];
                for (int p = 0; p < 8; p++) {
                    uint32_t ah0 = nid(c), ah1 = nid(c);
                    emit_arith(c, b, S_And, c->t_u32, ah0, aReg[p],
                               const_u32(c, 0xffffu));
                    emit_arith(c, b, S_ShrL, c->t_u32, ah1, aReg[p],
                               const_u32(c, 16));
                    uint32_t Af0 = is_bf16 ? bf16_to_f32val(c, b, ah0, line)
                                           : f16_to_f32val(c, b, ah0, line);
                    uint32_t Af1 = is_bf16 ? bf16_to_f32val(c, b, ah1, line)
                                           : f16_to_f32val(c, b, ah1, line);
                    uint32_t shb = nid(c), blane = nid(c);
                    emit_arith(c, b, S_Shl, c->t_u32, shb,
                               const_u32(c, (uint32_t)(p >> 1)),
                               const_u32(c, 3));
                    emit_arith(c, b, S_IAdd, c->t_u32, blane, shb,
                               const_u32(c, (uint32_t)j));
                    uint32_t bGot = wmma_shfl(c, b, bReg[p & 1], blane,
                                              wbase);
                    uint32_t bh0 = nid(c), bh1 = nid(c);
                    emit_arith(c, b, S_And, c->t_u32, bh0, bGot,
                               const_u32(c, 0xffffu));
                    emit_arith(c, b, S_ShrL, c->t_u32, bh1, bGot,
                               const_u32(c, 16));
                    uint32_t Bf0 = is_bf16 ? bf16_to_f32val(c, b, bh0, line)
                                           : f16_to_f32val(c, b, bh0, line);
                    uint32_t Bf1 = is_bf16 ? bf16_to_f32val(c, b, bh1, line)
                                           : f16_to_f32val(c, b, bh1, line);
                    if (c->failed)
                        return;
                    uint32_t pr = nid(c);
                    emit_fma_into(c, b, c->t_f32, pr, Af0, Bf0, acc);
                    acc = pr;
                    pr = nid(c);
                    emit_fma_into(c, b, c->t_f32, pr, Af1, Bf1, acc);
                    acc = pr;
                }
                def_val(c, in->wregs[0][j], PTY_F32, acc);
            }
            return;
        }
        // Gather A[r][0..15] from the two owner lanes.  16-bit fragments
        // pack two K values per 32-bit word, so one shuffle feeds two
        // FMAs instead of two shuffles.
        uint32_t a[16];
        for (int p = 0; p < 8; p++) {
            int j = p & 3;
            reg_t *ar = find_reg(c, in->wregs[1][j]);
            if (!ar || !ar->defined) {
                failf(c, line, "wmma.mma A fragment has no value");
                return;
            }
            uint32_t owner = nid(c);
            emit_arith(c, b, S_Shl, c->t_u32, owner, r,
                       const_u32(c, 1));
            if (p >= 4) {
                uint32_t o2 = nid(c);
                emit_arith(c, b, S_IAdd, c->t_u32, o2, owner,
                           const_u32(c, 1));
                owner = o2;
            }
            uint32_t word = wmma_shfl(c, b, ar->cur, owner, wbase);
            uint32_t lo = nid(c), hi = nid(c);
            emit_arith(c, b, S_And, c->t_u32, lo, word,
                       const_u32(c, 0xffffu));
            emit_arith(c, b, S_ShrL, c->t_u32, hi, word,
                       const_u32(c, 16));
            a[2 * p] = is_bf16 ? bf16_to_f32val(c, b, lo, line)
                               : f16_to_f32val(c, b, lo, line);
            a[2 * p + 1] = is_bf16 ? bf16_to_f32val(c, b, hi, line)
                                   : f16_to_f32val(c, b, hi, line);
            if (c->failed)
                return;
        }
        // Snapshot B and C before writing any D register, so d may
        // alias b or c (reads all happen before writes).
        uint32_t bbits[8], cacc[8];
        int nbb = 4;
        for (int j = 0; j < nbb; j++) {
            reg_t *br = find_reg(c, in->wregs[2][j]);
            if (!br || !br->defined) {
                failf(c, line, "wmma.mma B fragment has no value");
                return;
            }
            bbits[j] = br->cur;
        }
        if (f16acc) {
            for (int j = 0; j < 4; j++) {
                reg_t *cr = find_reg(c, in->wregs[3][j]);
                if (!cr || !cr->defined) {
                    failf(c, line, "wmma.mma C fragment has no value");
                    return;
                }
                uint32_t lo = nid(c), hi = nid(c);
                emit_arith(c, b, S_And, c->t_u32, lo, cr->cur,
                           const_u32(c, 0xffffu));
                emit_arith(c, b, S_ShrL, c->t_u32, hi, cr->cur,
                           const_u32(c, 16));
                cacc[2 * j] = f16_to_f32val(c, b, lo, line);
                cacc[2 * j + 1] = f16_to_f32val(c, b, hi, line);
                if (c->failed)
                    return;
            }
        } else {
            for (int i = 0; i < 8; i++) {
                reg_t *cr = find_reg(c, in->wregs[3][i]);
                if (!cr || !cr->defined) {
                    failf(c, line, "wmma.mma C fragment has no value");
                    return;
                }
                // The C fragment is declared .b32 but carries f32 values;
                // after the CFG state machine reloads it, cur is a uint
                // load.  Bitcast so the FMA operands stay f32.
                if (cr->cur_ty != PTY_F32) {
                    uint32_t f = nid(c);
                    put(c, b, (4u << 16) | S_Bitcast);
                    put(c, b, c->t_f32);
                    put(c, b, f);
                    put(c, b, cr->cur);
                    cacc[i] = f;
                } else {
                    cacc[i] = cr->cur;
                }
            }
        }
        uint32_t accs[8];
        for (int i = 0; i < 8; i++) {
            uint32_t col = nid(c);
            emit_arith(c, b, S_IAdd, c->t_u32, col, c0,
                       const_u32(c, (uint32_t)i));
            uint32_t acc = cacc[i];
            // Each B fragment word holds two consecutive K values from
            // the same owner lane.
            for (int p = 0; p < 8; p++) {
                uint32_t owner = nid(c);
                emit_arith(c, b, S_Shl, c->t_u32, owner, col,
                           const_u32(c, 1));
                if (p >= 4) {
                    uint32_t o2 = nid(c);
                    emit_arith(c, b, S_IAdd, c->t_u32, o2, owner,
                               const_u32(c, 1));
                    owner = o2;
                }
                uint32_t word =
                    wmma_shfl(c, b, bbits[p & 3], owner, wbase);
                uint32_t lo = nid(c), hi = nid(c);
                emit_arith(c, b, S_And, c->t_u32, lo, word,
                           const_u32(c, 0xffffu));
                emit_arith(c, b, S_ShrL, c->t_u32, hi, word,
                           const_u32(c, 16));
                uint32_t bf0 = is_bf16 ? bf16_to_f32val(c, b, lo, line)
                                       : f16_to_f32val(c, b, lo, line);
                uint32_t bf1 = is_bf16 ? bf16_to_f32val(c, b, hi, line)
                                       : f16_to_f32val(c, b, hi, line);
                if (c->failed)
                    return;
                uint32_t pr = nid(c);
                emit_fma_into(c, b, c->t_f32, pr, a[2 * p], bf0, acc);
                acc = pr;
                pr = nid(c);
                emit_fma_into(c, b, c->t_f32, pr, a[2 * p + 1], bf1, acc);
                acc = pr;
            }
            accs[i] = acc;
        }
        if (f16acc) {
            for (int j = 0; j < 4; j++) {
                uint32_t lo = f32_to_f16bits(c, b, accs[2 * j], line);
                uint32_t hi = f32_to_f16bits(c, b, accs[2 * j + 1],
                                             line);
                if (c->failed)
                    return;
                uint32_t sh = nid(c), pk = nid(c);
                emit_arith(c, b, S_Shl, c->t_u32, sh, hi,
                           const_u32(c, 16));
                emit_arith(c, b, S_Or, c->t_u32, pk, lo, sh);
                def_val(c, in->wregs[0][j], PTY_F16X2, pk);
            }
        } else {
            for (int i = 0; i < 8; i++)
                def_val(c, in->wregs[0][i], PTY_F32, accs[i]);
        }
        return;
    }
    default:
        return;
    }
}
