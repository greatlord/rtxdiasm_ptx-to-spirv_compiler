#include "instructions/group/group.h"

void emit_isspacep_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
        if (check_dst(c, in->dst, PTY_PRED, line) != 0)
            return;
        reg_t *r = find_reg(c, in->src[0].reg);
        if (!r || !r->defined) {
            failf(c, line, "isspacep source has no value");
            return;
        }
        int v = 0;
        if (!r->is_ptr) {
            // .const symbols materialize as an identity token
            // (PTX_CONST_ADDR_BASE | index), not a storage pointer.
            int is_const = 0;
            for (int i = 0; i < c->nu64; i++) {
                if (c->cu64[i].id != r->cur)
                    continue;
                uint64_t t = c->cu64[i].bits;
                if ((t & PTX_CONST_ADDR_BASE) &&
                    (t & ~PTX_CONST_ADDR_BASE) <
                        (uint64_t)c->const_bank_bytes)
                    is_const = 1;
                break;
            }
            if (!is_const) {
                if (in->isp_space == 3) {
                    // The token may have travelled through memory (the
                    // generic CFG state machine reloads registers), so
                    // test the const-token shape at runtime:
                    // (bit63 set) && (index < nconsts).
                    uint32_t ge = do_cmp(c, b, PCM_GE, PTY_U64, 0,
                                         r->cur,
                                         const_u64(c, PTX_CONST_ADDR_BASE),
                                         line);
                    uint32_t idx = nid(c);
                    emit_arith(c, b, S_And, c->t_u64, idx, r->cur,
                               const_u64(c,
                                   ~(uint64_t)PTX_CONST_ADDR_BASE));
                    uint32_t lt = do_cmp(c, b, PCM_LT, PTY_U64, 0, idx,
                                         const_u64(c,
                                             (uint64_t)c->const_bank_bytes),
                                         line);
                    uint32_t isc = nid(c);
                    emit_arith(c, b, S_LogicalAnd, c->t_bool, isc, ge, lt);
                    if (c->failed)
                        return;
                    def_val(c, in->dst, PTY_PRED, isc);
                    return;
                }
                // A plain value/token is not in shared/global/local/
                // param space.
                def_val(c, in->dst, PTY_PRED, bool_const(c, 0));
                return;
            }
            v = (in->isp_space == 3);
        } else {
            switch (in->isp_space) {
            case 0: v = r->shvar >= 0; break;
            case 1: v = r->binding >= 0 || r->binding == -3 ||
                       r->glvar >= 0 || r->paddr >= 0;
                    break;
            case 2: v = r->binding == -2; break;
            case 3: v = 0; break; // a tracked pointer is never const-space
            case 4: v = r->paddr >= 0; break;
            default:
                failf(c, line, "bad isspacep state space");
                return;
            }
        }
        def_val(c, in->dst, PTY_PRED, bool_const(c, v));
        return;
    }
