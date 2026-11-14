#include "instructions/arith/integer/integer.h"

void emit_mad24_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2, d;
    uint32_t id;
        // 24-bit multiply-add: 48-bit exact product, then .lo uses
        // bits 31..0 and .hi bits 47..16, adding c. .hi.sat.s32 clamps
        // the signed 32-bit addition to [INT_MIN, INT_MAX].
        if (check_dst(c, in->dst, PTY_U32, line) != 0)
            return;
        a = resolve(c, b, &in->src[0], PTY_U32, line);
        c2 = resolve(c, b, &in->src[1], PTY_U32, line);
        d = resolve(c, b, &in->src[2], PTY_U32, line);
        if (c->failed)
            return;
        uint32_t wa = nid(c), wb = nid(c);
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, c->t_u64);
        put(c, b, wa);
        put(c, b, a.id);
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, c->t_u64);
        put(c, b, wb);
        put(c, b, c2.id);
        uint32_t ea = wa, eb = wb;
        if (in->sign) {
            uint32_t s1 = nid(c), s2 = nid(c);
            emit_arith(c, b, S_Shl, c->t_u64, s1, wa, const_u64(c, 40));
            ea = nid(c);
            emit_arith(c, b, S_ShrA, c->t_u64, ea, s1, const_u64(c, 40));
            emit_arith(c, b, S_Shl, c->t_u64, s2, wb, const_u64(c, 40));
            eb = nid(c);
            emit_arith(c, b, S_ShrA, c->t_u64, eb, s2, const_u64(c, 40));
        } else {
            ea = nid(c);
            emit_arith(c, b, S_And, c->t_u64, ea, wa,
                       const_u64(c, 0xffffffu));
            eb = nid(c);
            emit_arith(c, b, S_And, c->t_u64, eb, wb,
                       const_u64(c, 0xffffffu));
        }
        uint32_t prod = nid(c);
        emit_arith(c, b, S_IMul, c->t_u64, prod, ea, eb);
        uint32_t sel = prod;
        if (in->is_hi) {
            sel = nid(c);
            emit_arith(c, b, in->sign ? S_ShrA : S_ShrL, c->t_u64,
                       sel, prod, const_u64(c, 16));
        }
        uint32_t part = nid(c);
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, c->t_u32);
        put(c, b, part);
        put(c, b, sel);
        if (in->sat) {
            // Signed 32-bit saturated add of the selected product half
            // and c, via an exact s64 intermediate.
            uint32_t x64 = nid(c), c64v = nid(c);
            put(c, b, (4u << 16) | S_SConvert);
            put(c, b, c->t_u64);
            put(c, b, x64);
            put(c, b, part);
            put(c, b, (4u << 16) | S_SConvert);
            put(c, b, c->t_u64);
            put(c, b, c64v);
            put(c, b, d.id);
            uint32_t sum = nid(c);
            emit_arith(c, b, S_IAdd, c->t_u64, sum, x64, c64v);
            uint32_t lo_ok = do_cmp(c, b, PCM_GE, PTY_U64, 1, sum,
                                    const_u64(c, 0xffffffff80000000ull), line);
            uint32_t hi_ok = do_cmp(c, b, PCM_LT, PTY_U64, 1, sum,
                                    const_u64(c, 0x80000000ull), line);
            if (c->failed)
                return;
            uint32_t lo = nid(c);
            put(c, b, (4u << 16) | S_UConvert);
            put(c, b, c->t_u32);
            put(c, b, lo);
            put(c, b, sum);
            uint32_t mid = sel_u32(c, b, hi_ok, lo,
                                   const_u32(c, 0x7fffffffu));
            id = sel_u32(c, b, lo_ok, mid, const_u32(c, 0x80000000u));
        } else {
            id = nid(c);
            emit_arith(c, b, S_IAdd, c->t_u32, id, part, d.id);
        }
        def_val(c, in->dst, PTY_U32, id);
        return;
    }
