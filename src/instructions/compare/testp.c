#include "instructions/compare/compare.h"

void emit_testp_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a;
        // Floating-point property predicates, computed from the IEEE
        // exponent/mantissa fields (Â±0 counts as normal).
        if (check_dst(c, in->dst, PTY_PRED, line) != 0)
            return;
        a = resolve(c, b, &in->src[0], in->ty, line);
        if (c->failed)
            return;
        int f64 = in->ty == PTY_F64;
        uint32_t bits = nid(c), exp = nid(c), mant = nid(c);
        uint32_t maxexp = f64 ? 0x7ffu : 0xffu;
        uint64_t mantmask = f64 ? 0xfffffffffffffull : 0x7fffffu;
        if (f64) {
            put(c, b, (4u << 16) | S_Bitcast);
            put(c, b, c->t_u64);
            put(c, b, bits);
            put(c, b, a.id);
            emit_arith(c, b, S_ShrL, c->t_u64, exp, bits,
                       const_u64(c, 52));
            uint32_t e2 = nid(c);
            emit_arith(c, b, S_And, c->t_u64, e2, exp,
                       const_u64(c, maxexp));
            exp = e2;
            emit_arith(c, b, S_And, c->t_u64, mant, bits,
                       const_u64(c, mantmask));
        } else {
            put(c, b, (4u << 16) | S_Bitcast);
            put(c, b, c->t_u32);
            put(c, b, bits);
            put(c, b, a.id);
            emit_arith(c, b, S_ShrL, c->t_u32, exp, bits,
                       const_u32(c, 23));
            uint32_t e2 = nid(c);
            emit_arith(c, b, S_And, c->t_u32, e2, exp,
                       const_u32(c, maxexp));
            exp = e2;
            emit_arith(c, b, S_And, c->t_u32, mant, bits,
                       const_u32(c, mantmask));
        }
        ptx_ty_t ct = f64 ? PTY_U64 : PTY_U32;
        uint32_t zero = f64 ? const_u64(c, 0) : const_u32(c, 0);
        uint32_t maxc = f64 ? const_u64(c, maxexp)
                            : const_u32(c, maxexp);
        uint32_t is_max = do_cmp(c, b, PCM_EQ, ct, 0, exp, maxc, line);
        uint32_t mant_nz = do_cmp(c, b, PCM_NE, ct, 0, mant, zero, line);
        uint32_t exp_z = do_cmp(c, b, PCM_EQ, ct, 0, exp, zero, line);
        if (c->failed)
            return;
        uint32_t is_nan = nid(c), is_inf = nid(c);
        put(c, b, (5u << 16) | S_LogicalAnd);
        put(c, b, c->t_bool);
        put(c, b, is_nan);
        put(c, b, is_max);
        put(c, b, mant_nz);
        uint32_t mn = nid(c);
        put(c, b, (4u << 16) | S_LogicalNot);
        put(c, b, c->t_bool);
        put(c, b, mn);
        put(c, b, mant_nz);
        put(c, b, (5u << 16) | S_LogicalAnd);
        put(c, b, c->t_bool);
        put(c, b, is_inf);
        put(c, b, is_max);
        put(c, b, mn);
        uint32_t finite = nid(c), number = nid(c);
        put(c, b, (4u << 16) | S_LogicalNot);
        put(c, b, c->t_bool);
        put(c, b, finite);
        put(c, b, is_max);
        put(c, b, (4u << 16) | S_LogicalNot);
        put(c, b, c->t_bool);
        put(c, b, number);
        put(c, b, is_nan);
        uint32_t notzero_exp = nid(c), normal0 = nid(c), normal = nid(c);
        put(c, b, (4u << 16) | S_LogicalNot);
        put(c, b, c->t_bool);
        put(c, b, notzero_exp);
        put(c, b, exp_z);
        uint32_t mant_zero = nid(c);
        put(c, b, (4u << 16) | S_LogicalNot);
        put(c, b, c->t_bool);
        put(c, b, mant_zero);
        put(c, b, mant_nz);
        put(c, b, (5u << 16) | S_LogicalOr);
        put(c, b, c->t_bool);
        put(c, b, normal0);
        put(c, b, notzero_exp);
        put(c, b, mant_zero);
        put(c, b, (5u << 16) | S_LogicalAnd);
        put(c, b, c->t_bool);
        put(c, b, normal);
        put(c, b, finite);
        put(c, b, normal0);
        uint32_t subnormal = nid(c);
        put(c, b, (5u << 16) | S_LogicalAnd);
        put(c, b, c->t_bool);
        put(c, b, subnormal);
        put(c, b, exp_z);
        put(c, b, mant_nz);
        uint32_t res;
        switch (in->arg) {
        case 0: res = finite; break;
        case 1: res = is_inf; break;
        case 2: res = number; break;
        case 3: res = is_nan; break;
        case 4: res = normal; break;
        default: res = subnormal; break;
        }
        def_val(c, in->dst, PTY_PRED, res);
        return;
    }
