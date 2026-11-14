#include "instructions/arith/integer/integer.h"

void emit_mul24_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2;
    uint32_t id;
        // 24-bit multiply through the exact 64-bit core: a 24x24
        // product always fits in 48 bits, so u64 IMul is exact.
        // .u32 masks to the low 24 bits; .s32 sign-extends bit 23
        // (shl 40, arithmetic shr 40). .lo truncates the product;
        // .hi takes bits 47..16 (NVIDIA semantics, pinned by ZLUDA's
        // expected vectors in the corpus tests).
        if (check_dst(c, in->dst, PTY_U32, line) != 0)
            return;
        a = resolve(c, b, &in->src[0], PTY_U32, line);
        c2 = resolve(c, b, &in->src[1], PTY_U32, line);
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
            emit_arith(c, b, S_Shl, c->t_u64, s1, wa,
                       const_u64(c, 40));
            ea = nid(c);
            emit_arith(c, b, S_ShrA, c->t_u64, ea, s1,
                       const_u64(c, 40));
            emit_arith(c, b, S_Shl, c->t_u64, s2, wb,
                       const_u64(c, 40));
            eb = nid(c);
            emit_arith(c, b, S_ShrA, c->t_u64, eb, s2,
                       const_u64(c, 40));
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
            // Bits 47..16: logical shift for .u32, arithmetic for
            // .s32 (low 32 of the shifted value agree with the
            // 48-bit two's-complement reading either way).
            sel = nid(c);
            emit_arith(c, b, in->sign ? S_ShrA : S_ShrL, c->t_u64,
                       sel, prod, const_u64(c, 16));
        }
        id = nid(c);
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, c->t_u32);
        put(c, b, id);
        put(c, b, sel);
        def_val(c, in->dst, PTY_U32, id);
        return;
    }
