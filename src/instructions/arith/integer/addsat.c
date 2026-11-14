#include "instructions/arith/integer/integer.h"

void emit_addsat_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2;
    uint32_t id;
        // Saturating add/sub, scalar or packed. Scalars use an exact
        // 64-bit core then clamp; packed lanes sign-extend (signed) or
        // zero-extend and clamp each lane before repacking.
        if (check_dst(c, in->dst, PTY_U32, line) != 0)
            return;
        a = resolve(c, b, &in->src[0], PTY_U32, line);
        c2 = resolve(c, b, &in->src[1], PTY_U32, line);
        if (c->failed)
            return;
        int sub = in->sat_sub;
        if (in->pack) {
            int width = in->pack == 2 ? 16 : 8;
            uint32_t mask = (1u << width) - 1u;
            // Signed clamp bounds as sign-extended 32-bit values so the
            // s32 compares and selects see the true lane range.
            uint32_t minv = (uint32_t)(-(1 << (width - 1)));
            uint32_t maxv = (1u << (width - 1)) - 1u;
            uint32_t out = 0;
            for (int i = 0; i < in->pack; i++) {
                uint32_t la = pack_lane_get(c, b, a.id, i, width, in->sign);
                uint32_t lb = pack_lane_get(c, b, c2.id, i, width, in->sign);
                uint32_t r = nid(c);
                emit_arith(c, b, sub ? S_ISub : S_IAdd, c->t_u32, r,
                           la, lb);
                uint32_t cl;
                if (in->sign) {
                    uint32_t lt = do_cmp(c, b, PCM_LT, PTY_U32, 1, r,
                                         const_u32(c, minv), line);
                    uint32_t gt = do_cmp(c, b, PCM_GT, PTY_U32, 1, r,
                                         const_u32(c, maxv), line);
                    if (c->failed)
                        return;
                    uint32_t mid = sel_u32(c, b, lt, const_u32(c, minv), r);
                    cl = sel_u32(c, b, gt, const_u32(c, maxv), mid);
                } else if (sub) {
                    uint32_t lt = do_cmp(c, b, PCM_LT, PTY_U32, 0, la,
                                         lb, line);
                    if (c->failed)
                        return;
                    uint32_t d = nid(c);
                    emit_arith(c, b, S_ISub, c->t_u32, d, la, lb);
                    cl = sel_u32(c, b, lt, const_u32(c, 0), d);
                } else {
                    uint32_t over = do_cmp(c, b, PCM_GT, PTY_U32, 0, r,
                                           const_u32(c, mask), line);
                    if (c->failed)
                        return;
                    cl = sel_u32(c, b, over, const_u32(c, mask), r);
                }
                out = pack_lane_put(c, b, out, cl, i, width, mask);
                if (c->failed)
                    return;
            }
            def_val(c, in->dst, PTY_U32, out);
            return;
        }
        uint32_t wa = nid(c), wb = nid(c);
        uint32_t conv = in->sign ? S_SConvert : S_UConvert;
        put(c, b, (4u << 16) | conv);
        put(c, b, c->t_u64);
        put(c, b, wa);
        put(c, b, a.id);
        put(c, b, (4u << 16) | conv);
        put(c, b, c->t_u64);
        put(c, b, wb);
        put(c, b, c2.id);
        uint32_t sum = nid(c);
        emit_arith(c, b, sub ? S_ISub : S_IAdd, c->t_u64, sum, wa, wb);
        if (!in->sign) {
            uint32_t lo = nid(c);
            put(c, b, (4u << 16) | S_UConvert);
            put(c, b, c->t_u32);
            put(c, b, lo);
            put(c, b, sum);
            if (sub) {
                // Unsigned subtraction underflows below zero.
                uint32_t under = do_cmp(c, b, PCM_LT, PTY_U64, 0, wa,
                                        wb, line);
                if (c->failed)
                    return;
                id = sel_u32(c, b, under, const_u32(c, 0), lo);
            } else {
                uint32_t over = do_cmp(c, b, PCM_GT, PTY_U64, 0, sum,
                                       const_u64(c, 0xffffffffull), line);
                if (c->failed)
                    return;
                id = sel_u32(c, b, over, const_u32(c, 0xffffffffu), lo);
            }
        } else {
            uint32_t lo_ok = do_cmp(c, b, PCM_GE, PTY_U64, 1, sum,
                                    const_u64(c, 0xffffffff80000000ull), line);
            if (c->failed)
                return;
            uint32_t hi_ok = do_cmp(c, b, PCM_LT, PTY_U64, 1, sum,
                                    const_u64(c, 0x80000000ull), line);
            if (c->failed)
                return;
            uint32_t lo = nid(c);
            put(c, b, (4u << 16) | S_UConvert);
            put(c, b, c->t_u32);
            put(c, b, lo);
            put(c, b, sum);
            uint32_t mid = sel_u32(c, b, hi_ok, lo, const_u32(c, 0x7fffffffu));
            id = sel_u32(c, b, lo_ok, mid, const_u32(c, 0x80000000u));
        }
        def_val(c, in->dst, PTY_U32, id);
        return;
    }
