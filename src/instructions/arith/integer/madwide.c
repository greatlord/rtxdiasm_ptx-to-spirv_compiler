#include "instructions/arith/integer/integer.h"

void emit_madwide_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2, d;
    uint32_t id;
        // Wide multiply-add. 16-bit sources produce a 32-bit result;
        // 32-bit sources produce a 64-bit result.  When the 64-bit
        // addend is a pointer (the usual address idiom), the result
        // is that pointer plus a scaled index.
        if (in->ty != PTY_U16) {
            int pi = -1, nptr = 0;
            for (int s = 0; s < 3; s++) {
                if (in->src[s].kind == OPD_REG) {
                    reg_t *qr = find_reg(c, in->src[s].reg);
                    if (qr && qr->defined && qr->is_ptr) {
                        nptr++;
                        pi = s;
                    }
                }
            }
            if (nptr > 0) {
                if (nptr > 1 || pi != 2) {
                    failf(c, line,
                          "mad.wide needs its pointer in the addend");
                    return;
                }
                reg_t *pr = find_reg(c, in->src[2].reg);
                val_t mv1 = resolve(c, b, &in->src[0], PTY_U32, line);
                val_t mv2 = resolve(c, b, &in->src[1], PTY_U32, line);
                if (c->failed)
                    return;
                uint32_t w0 = nid(c), w1 = nid(c);
                uint32_t conv = in->sign ? S_SConvert : S_UConvert;
                put(c, b, (4u << 16) | conv);
                put(c, b, c->t_u64);
                put(c, b, w0);
                put(c, b, mv1.id);
                put(c, b, (4u << 16) | conv);
                put(c, b, c->t_u64);
                put(c, b, w1);
                put(c, b, mv2.id);
                uint32_t prod = nid(c), sum = nid(c);
                emit_arith(c, b, S_IMul, c->t_u64, prod, w0, w1);
                emit_arith(c, b, S_IAdd, c->t_u64, sum, prod, pr->off);
                if (pr->shvar >= 0)
                    def_shared(c, in->dst, pr->shvar, sum);
                else if (pr->glvar >= 0)
                    def_global(c, in->dst, pr->glvar, sum);
                else {
                    def_ptr(c, in->dst, pr->binding, sum);
                    if (pr->binding == -2) {
                        reg_t *dr = find_reg(c, in->dst);
                        if (dr) dr->lovar = pr->lovar;
                    }
                }
                return;
            }
        }
        if (in->ty == PTY_U16) {
            if (check_dst(c, in->dst, PTY_U32, line) != 0)
                return;
            a = resolve(c, b, &in->src[0], PTY_U16, line);
            c2 = resolve(c, b, &in->src[1], PTY_U16, line);
            d = resolve(c, b, &in->src[2], PTY_U32, line);
            if (c->failed)
                return;
            uint32_t av = a.id, bv = c2.id;
            if (in->sign) {
                av = pack_lane_get(c, b, a.id, 0, 16, 1);
                bv = pack_lane_get(c, b, c2.id, 0, 16, 1);
            }
            uint32_t pr = nid(c);
            emit_arith(c, b, S_IMul, c->t_u32, pr, av, bv);
            id = nid(c);
            emit_arith(c, b, S_IAdd, c->t_u32, id, pr, d.id);
            def_val(c, in->dst, PTY_U32, id);
            return;
        }
        if (check_dst(c, in->dst, PTY_U64, line) != 0)
            return;
        a = resolve(c, b, &in->src[0], PTY_U32, line);
        c2 = resolve(c, b, &in->src[1], PTY_U32, line);
        d = resolve(c, b, &in->src[2], PTY_U64, line);
        if (c->failed)
            return;
        {
            uint32_t w0 = nid(c), w1 = nid(c);
            uint32_t conv = in->sign ? S_SConvert : S_UConvert;
            put(c, b, (4u << 16) | conv);
            put(c, b, c->t_u64);
            put(c, b, w0);
            put(c, b, a.id);
            put(c, b, (4u << 16) | conv);
            put(c, b, c->t_u64);
            put(c, b, w1);
            put(c, b, c2.id);
            uint32_t pr = nid(c);
            emit_arith(c, b, S_IMul, c->t_u64, pr, w0, w1);
            id = nid(c);
            emit_arith(c, b, S_IAdd, c->t_u64, id, pr, d.id);
            def_val(c, in->dst, PTY_U64, id);
        }
        return;
    }
