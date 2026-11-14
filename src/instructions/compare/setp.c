#include "instructions/compare/compare.h"

void emit_setp_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2;
    uint32_t id;
        reg_t *r = find_reg(c, in->dst);
        if (!r || r->decl_ty != PTY_PRED) {
            failf(c, line, "setp destination must be a .pred register");
            return;
        }
        if (in->ty == PTY_F16X2) {
            val_t px = resolve(c, b, &in->src[0], PTY_U32, line);
            if (c->failed)
                return;
            val_t py = resolve(c, b, &in->src[1], PTY_U32, line);
            if (c->failed)
                return;
            uint32_t xhi, xlo, yhi, ylo;
            split_pack(c, b, px.id, &xhi, &xlo);
            split_pack(c, b, py.id, &yhi, &ylo);
            if (in->ftz) {
                xhi = flush_f16_bits(c, b, xhi, line);
                xlo = flush_f16_bits(c, b, xlo, line);
                yhi = flush_f16_bits(c, b, yhi, line);
                ylo = flush_f16_bits(c, b, ylo, line);
            }
            uint32_t tl = do_cmp(c, b, in->cmp, PTY_F32, 0,
                                 f16_to_f32val(c, b, xlo, line),
                                 f16_to_f32val(c, b, ylo, line), line);
            if (c->failed)
                return;
            tl = set_bop_join(c, b, in, tl, line);
            uint32_t th = do_cmp(c, b, in->cmp, PTY_F32, 0,
                                 f16_to_f32val(c, b, xhi, line),
                                 f16_to_f32val(c, b, yhi, line), line);
            if (c->failed)
                return;
            th = set_bop_join(c, b, in, th, line);
            if (c->failed)
                return;
            if (!in->dst2[0]) {
                failf(c, line, "setp.f16x2 needs a dual destination");
                return;
            }
            reg_t *q = find_reg(c, in->dst2);
            if (!q || q->decl_ty != PTY_PRED) {
                failf(c, line, "setp second destination must be .pred");
                return;
            }
            def_val(c, in->dst, PTY_PRED, tl);
            def_val(c, in->dst2, PTY_PRED, th);
            return;
        }
        a = resolve(c, b, &in->src[0], in->ty, line);
        c2 = resolve(c, b, &in->src[1], in->ty, line);
        if (c->failed)
            return;
        if (in->ftz) {
            // Flush-to-zero compare (parse guarantees f32 operands).
            a.id = flush_f32_denorm(c, b, a.id, line);
            if (c->failed)
                return;
            c2.id = flush_f32_denorm(c, b, c2.id, line);
            if (c->failed)
                return;
        }
        if ((in->ty == PTY_U8 || in->ty == PTY_U16) && in->sign) {
            // Signed narrow compare: narrow both sides to the width
            // first (canonical values are zero-extended).
            uint32_t sh = (uint32_t)(in->ty == PTY_U8 ? 24 : 16);
            uint32_t a1 = nid(c), a2 = nid(c);
            emit_arith(c, b, S_Shl, c->t_u32, a1, a.id, const_u32(c, sh));
            emit_arith(c, b, S_ShrA, c->t_u32, a2, a1, const_u32(c, sh));
            uint32_t b1 = nid(c), b2 = nid(c);
            emit_arith(c, b, S_Shl, c->t_u32, b1, c2.id, const_u32(c, sh));
            emit_arith(c, b, S_ShrA, c->t_u32, b2, b1, const_u32(c, sh));
            a.id = a2;
            c2.id = b2;
        } else if (in->ty == PTY_U8 || in->ty == PTY_U16) {
            // Unsigned narrow compare: mask to the width first (some
            // producers leave garbage above it).
            uint32_t lim = in->ty == PTY_U8 ? 0xffu : 0xffffu;
            uint32_t a1 = nid(c), b1 = nid(c);
            emit_arith(c, b, S_And, c->t_u32, a1, a.id, const_u32(c, lim));
            emit_arith(c, b, S_And, c->t_u32, b1, c2.id, const_u32(c, lim));
            a.id = a1;
            c2.id = b1;
        }
        if (in->ty == PTY_F16) {
            // Halves compare as their exact f32 values.
            uint32_t xb = a.id, yb = c2.id;
            if (in->ftz) {
                xb = flush_f16_bits(c, b, xb, line);
                yb = flush_f16_bits(c, b, yb, line);
            }
            uint32_t fa = f16_to_f32val(c, b, xb, line);
            if (c->failed)
                return;
            uint32_t fb = f16_to_f32val(c, b, yb, line);
            if (c->failed)
                return;
            id = do_cmp(c, b, in->cmp, PTY_F32, 0, fa, fb, line);
        } else {
            id = do_cmp(c, b, in->cmp, in->ty, in->sign, a.id, c2.id, line);
        }
        if (c->failed)
            return;
        if (in->bop) {
            // Boolean operator over a third predicate source (the
            // parser guaranteed a predicate register operand).
            reg_t *qr = find_reg(c, in->src[2].reg);
            if (!qr || qr->decl_ty != PTY_PRED) {
                failf(c, line, "setp boolean source must be a predicate register");
                return;
            }
            if (!qr->defined) {
                failf(c, line, "setp boolean source has no value");
                return;
            }
            uint32_t qb = qr->cur;
            if (in->bneg) {
                uint32_t nqb = nid(c);
                put(c, b, (4u << 16) | S_LogicalNot);
                put(c, b, c->t_bool);
                put(c, b, nqb);
                put(c, b, qb);
                qb = nqb;
            }
            uint32_t bp = nid(c);
            if (in->bop == 1) {
                put(c, b, (5u << 16) | S_LogicalAnd);
                put(c, b, c->t_bool);
                put(c, b, bp);
                put(c, b, id);
                put(c, b, qb);
            } else if (in->bop == 2) {
                put(c, b, (5u << 16) | S_LogicalOr);
                put(c, b, c->t_bool);
                put(c, b, bp);
                put(c, b, id);
                put(c, b, qb);
            } else {
                // No bool-xor opcode (INotEqual needs int operands):
                // (a & ~b) | (~a & b).
                uint32_t nb = nid(c), na = nid(c);
                put(c, b, (4u << 16) | S_LogicalNot);
                put(c, b, c->t_bool);
                put(c, b, nb);
                put(c, b, qb);
                put(c, b, (4u << 16) | S_LogicalNot);
                put(c, b, c->t_bool);
                put(c, b, na);
                put(c, b, id);
                uint32_t t1 = nid(c), t2 = nid(c);
                put(c, b, (5u << 16) | S_LogicalAnd);
                put(c, b, c->t_bool);
                put(c, b, t1);
                put(c, b, id);
                put(c, b, nb);
                put(c, b, (5u << 16) | S_LogicalAnd);
                put(c, b, c->t_bool);
                put(c, b, t2);
                put(c, b, na);
                put(c, b, qb);
                put(c, b, (5u << 16) | S_LogicalOr);
                put(c, b, c->t_bool);
                put(c, b, bp);
                put(c, b, t1);
                put(c, b, t2);
            }
            id = bp;
        }
        r->defined = 1;
        r->cur = id;
        r->is_ptr = 0;
        r->shvar = -1;
        r->glvar = -1;
        if (in->dst2[0]) {
            // Dual destination: the second predicate takes the complement.
            reg_t *r2 = find_reg(c, in->dst2);
            if (!r2 || r2->decl_ty != PTY_PRED) {
                failf(c, line, "setp second destination must be a .pred register");
                return;
            }
            uint32_t n = nid(c);
            put(c, b, (4u << 16) | S_LogicalNot);
            put(c, b, c->t_bool);
            put(c, b, n);
            put(c, b, id);
            r2->defined = 1;
            r2->cur = n;
            r2->is_ptr = 0;
            r2->shvar = -1;
            r2->glvar = -1;
        }
        return;
    }
