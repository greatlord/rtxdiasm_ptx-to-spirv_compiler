#include "instructions/compare/compare.h"

void emit_slct_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    val_t a, c2, d;
    uint32_t id;
        // Select a if c >= 0, else b. Integer c is .s32; float c is
        // ordered .f32 (NaN selects b), optionally flushed to zero.
        if (check_dst(c, in->dst, in->ty, line) != 0)
            return;
        a = resolve(c, b, &in->src[0], in->ty, line);
        c2 = resolve(c, b, &in->src[1], in->ty, line);
        d = resolve(c, b, &in->src[2], in->ty2, line);
        if (c->failed)
            return;
        uint32_t cond;
        if (in->ty2 == PTY_F32) {
            uint32_t cv = d.id;
            if (in->ftz) {
                cv = flush_f32_denorm(c, b, cv, line);
                if (c->failed)
                    return;
            }
            cond = do_cmp(c, b, PCM_GE, PTY_F32, 0, cv,
                          const_f32(c, 0), line);
        } else {
            cond = do_cmp(c, b, PCM_GE, PTY_U32, 1, d.id,
                          const_u32(c, 0), line);
        }
        if (c->failed)
            return;
        id = nid(c);
        put(c, b, (6u << 16) | S_Select);
        put(c, b, ty_of(c, in->ty));
        put(c, b, id);
        put(c, b, cond);
        put(c, b, a.id);
        put(c, b, c2.id);
        def_val(c, in->dst, in->ty, id);
        return;
    }
