// Matrix instruction parsing: ldmatrix and mma.sync fragments.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ptx.h"
#include "parse/parse_internal.h"

static int wmma_regs(parser_t *p, char regs[8][64], int nregs,
                     const char *what) {
    if (expect_punct(p, '{', what) != 0)
        return -1;
    for (int i = 0; i < nregs; i++) {
        ptx_tok_t t;
        if (next(p, &t) != 0)
            return -1;
        if (t.kind != T_IDENT) {
            perr(p, t.line, "expected fragment register");
            return -1;
        }
        if (copy_name(regs[i], 64, t.text, p, t.line) != 0)
            return -1;
        if (i + 1 < nregs && expect_comma(p) != 0)
            return -1;
    }
    if (expect_punct(p, '}', what) != 0)
        return -1;
    return 0;
}

static int wmma_tokens(const char *opname, char mp[16][32]) {
    int n = 0;
    const char *s = opname;
    memset(mp, 0, 16 * 32);
    while (*s && n < 16) {
        size_t l = 0;
        while (s[l] && s[l] != '.' && l + 1 < 32)
            l++;
        memcpy(mp[n], s, l);
        mp[n][l] = '\0';
        n++;
        s += l;
        if (*s == '.')
            s++;
        else if (*s)
            break;
    }
    return *s ? -1 : n;
}

int parse_matrix_instr(parser_t *p, ptx_func_t *fn, size_t bi,
                       const char *pred, int pred_neg, const char *opname,
                       int opline) {
    if (strncmp(opname, "wmma.", 5) == 0) {
        // Single-bit WMMA has an extra .op.popc pair before .sync.
        if (strncmp(opname, "wmma.mma.", 9) == 0 &&
            strstr(opname, ".popc.sync.") != NULL) {
            char mp[16][32];
            int mnp = wmma_tokens(opname, mp);
            if (mnp != 13 || strcmp(mp[1], "mma") != 0 ||
                (strcmp(mp[2], "and") != 0 && strcmp(mp[2], "xor") != 0) ||
                strcmp(mp[3], "popc") != 0 || strcmp(mp[4], "sync") != 0 ||
                strcmp(mp[5], "aligned") != 0 ||
                strcmp(mp[6], "row") != 0 || strcmp(mp[7], "col") != 0 ||
                strcmp(mp[8], "m8n8k128") != 0 ||
                strcmp(mp[9], "s32") != 0 || strcmp(mp[10], "b1") != 0 ||
                strcmp(mp[11], "b1") != 0 || strcmp(mp[12], "s32") != 0) {
                perr(p, opline, "unsupported single-bit wmma.mma form");
                return -1;
            }
            if (pred[0] || pred_neg) {
                perr(p, opline, "predicated wmma.mma unsupported");
                return -1;
            }
            ptx_instr_t in;
            memset(&in, 0, sizeof(in));
            in.line = opline;
            in.op = POP_WMMA_MMA;
            in.wmma_shape = 5;
            in.wmma_row = 1;
            in.mma_elem = 2;
            in.mma_b1_xor = strcmp(mp[2], "xor") == 0;
            in.ty = PTY_U32;
            in.ty2 = PTY_U32;
            if (wmma_regs(p, in.wregs[0], 2, "wmma D") != 0 ||
                expect_comma(p) != 0 ||
                wmma_regs(p, in.wregs[1], 1, "wmma A") != 0 ||
                expect_comma(p) != 0 ||
                wmma_regs(p, in.wregs[2], 1, "wmma B") != 0 ||
                expect_comma(p) != 0 ||
                wmma_regs(p, in.wregs[3], 2, "wmma C") != 0)
                return -1;
            if (expect_punct(p, ';', "after wmma.mma") != 0)
                return -1;
            return block_add_instr(fn, bi, &in, p, opline);
        }
        // Minimal wmma m16n16k16 f16->f32 subset with an ACuda-private
        // (but internally consistent) fragment layout. PTX explicitly
        // leaves the fragment identity target-dependent.
        char mp[16][32];
        int mnp = wmma_tokens(opname, mp);
        if (mnp < 0) {
            perr(p, opline, "wmma: too many qualifiers");
            return -1;
        }
        int is_load = mnp > 1 && strcmp(mp[1], "load") == 0;
        int is_store = mnp > 1 && strcmp(mp[1], "store") == 0;
        int is_mma = mnp > 1 && strcmp(mp[1], "mma") == 0;
        if ((!is_load && !is_store && !is_mma) || pred[0] || pred_neg) {
            perr(p, opline, "unsupported/predicated wmma form");
            return -1;
        }
        int kind = -1; // 0 A, 1 B, 2 C, 3 D
        int in_mma_bf16 = 0;
        int in_mma_f16acc = 0;
        int in_mma_tf32 = 0;
        int in_mma_int = 0;
        int in_mma_int4 = 0;
        int in_mma_int8_signed = 0;
        int in_space = PSP_GENERIC;
        int row = -1;
        int shape = -1, dtype_ok = 0;
        if (is_load) {
            if (mnp < 7 || strcmp(mp[3], "sync") != 0 ||
                strcmp(mp[4], "aligned") != 0) {
                perr(p, opline, "wmma.load must be wmma.load.{a,b,c}.sync.aligned.layout.m16n16k16.type");
                return -1;
            }
            if (strcmp(mp[2], "a") == 0) kind = 0;
            else if (strcmp(mp[2], "b") == 0) kind = 1;
            else if (strcmp(mp[2], "c") == 0) kind = 2;
            else kind = -1;
            for (int i = 5; i < mnp; i++) {
                if (strcmp(mp[i], "row") == 0) row = 1;
                else if (strcmp(mp[i], "col") == 0) row = 0;
                else if (strcmp(mp[i], "m16n16k16") == 0) shape = 0;
                else if (strcmp(mp[i], "m8n32k16") == 0) shape = 1;
                else if (strcmp(mp[i], "m32n8k16") == 0) shape = 2;
                else if (strcmp(mp[i], "m16n16k8") == 0) shape = 3;
                else if (strcmp(mp[i], "m8n8k32") == 0) shape = 4;
                else if (strcmp(mp[i], "m8n8k128") == 0) shape = 5;
                else if (strcmp(mp[i], "f16") == 0) dtype_ok = 1;
                else if (strcmp(mp[i], "f32") == 0) dtype_ok = 2;
                else if (strcmp(mp[i], "bf16") == 0) dtype_ok = 3;
                else if (strcmp(mp[i], "tf32") == 0) dtype_ok = 4;
                else if (strcmp(mp[i], "u8") == 0) dtype_ok = 5;
                else if (strcmp(mp[i], "s8") == 0) dtype_ok = 6;
                else if (strcmp(mp[i], "s32") == 0) dtype_ok = 7;
                else if (strcmp(mp[i], "u4") == 0) dtype_ok = 8;
                else if (strcmp(mp[i], "s4") == 0) dtype_ok = 9;
                else if (strcmp(mp[i], "b1") == 0) dtype_ok = 10;
                else if (strcmp(mp[i], "global") == 0)
                    in_space = PSP_GLOBAL;
                else if (strcmp(mp[i], "shared") == 0)
                    in_space = PSP_SHARED;
                else { shape = -1; }
            }
            int ab_ok =
                kind <= 1 &&
                (shape == 3 ? dtype_ok == 4
                 : shape == 4 ? (dtype_ok == 8 || dtype_ok == 9)
                 : shape == 5 ? dtype_ok == 10
                              : (dtype_ok == 1 || dtype_ok == 3 ||
                                 dtype_ok == 5 || dtype_ok == 6));
            int c_ok =
                kind == 2 &&
                (shape == 3 ? dtype_ok == 2
                 : (shape == 4 || shape == 5) ? dtype_ok == 7
                              : (dtype_ok == 1 || dtype_ok == 2 ||
                                 dtype_ok == 7)) &&
                !(dtype_ok == 1 && shape != 0);
            if (kind < 0 || row < 0 || shape < 0 || (!ab_ok && !c_ok)) {
                perr(p, opline, "only wmma.load.{a,b}.{f16,bf16,u8,s8} and wmma.load.c.{f16,f32,s32} supported for the selected shape");
                return -1;
            }
            if (in_mma_int && ((kind == 1 && row != 0) ||
                               (kind != 1 && row != 1))) {
                perr(p, opline, "integer WMMA requires row A/C/D and col B layouts");
                return -1;
            }
            in_mma_tf32 = (dtype_ok == 4);
            in_mma_bf16 = (dtype_ok == 3);
            in_mma_int = (dtype_ok == 5 || dtype_ok == 6 ||
                          dtype_ok == 7 || dtype_ok == 8 ||
                          dtype_ok == 9 || dtype_ok == 10);
            in_mma_int4 = (dtype_ok == 8 || dtype_ok == 9);
            in_mma_int8_signed = (dtype_ok == 6 || dtype_ok == 9);
            in_mma_f16acc = (kind == 2 && dtype_ok == 1);
        } else if (is_store) {
            if (mnp < 7 || strcmp(mp[2], "d") != 0 ||
                strcmp(mp[3], "sync") != 0 || strcmp(mp[4], "aligned") != 0) {
                perr(p, opline, "wmma.store must be wmma.store.d.sync.aligned.layout.m16n16k16.f32");
                return -1;
            }
            kind = 3;
            for (int i = 5; i < mnp; i++) {
                if (strcmp(mp[i], "row") == 0) row = 1;
                else if (strcmp(mp[i], "col") == 0) row = 0;
                else if (strcmp(mp[i], "m16n16k16") == 0) shape = 0;
                else if (strcmp(mp[i], "m8n32k16") == 0) shape = 1;
                else if (strcmp(mp[i], "m32n8k16") == 0) shape = 2;
                else if (strcmp(mp[i], "m16n16k8") == 0) shape = 3;
                else if (strcmp(mp[i], "m8n8k32") == 0) shape = 4;
                else if (strcmp(mp[i], "m8n8k128") == 0) shape = 5;
                else if (strcmp(mp[i], "f32") == 0) dtype_ok = 2;
                else if (strcmp(mp[i], "f16") == 0) dtype_ok = 1;
                else if (strcmp(mp[i], "s32") == 0) dtype_ok = 7;
                else if (strcmp(mp[i], "u4") == 0) dtype_ok = 8;
                else if (strcmp(mp[i], "s4") == 0) dtype_ok = 9;
                else if (strcmp(mp[i], "b1") == 0) dtype_ok = 10;
                else if (strcmp(mp[i], "global") == 0)
                    in_space = PSP_GLOBAL;
                else if (strcmp(mp[i], "shared") == 0)
                    in_space = PSP_SHARED;
                else { shape = -1; }
            }
            if (row < 0 || shape < 0 ||
                (dtype_ok != 1 && dtype_ok != 2 && dtype_ok != 7) ||
                (dtype_ok == 7 && shape == 3)) {
                perr(p, opline, "only wmma.store.d.{f16,f32,s32} supported");
                return -1;
            }
            if (dtype_ok == 7 && row != 1) {
                perr(p, opline, "integer WMMA requires row D layout");
                return -1;
            }
            if (shape != 0 && dtype_ok == 1) {
                perr(p, opline, "only f32 accumulators are supported for wmma m8n32k16/m32n8k16");
                return -1;
            }
            in_mma_f16acc = (dtype_ok == 1);
            in_mma_int = (dtype_ok == 7);
            in_mma_int4 = (shape == 4 || shape == 5);
        } else {
            // wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 (f16
            // multiplicands implied) or ...f32.bf16.bf16.f32.
            if (mnp < 5 || strcmp(mp[2], "sync") != 0 ||
                strcmp(mp[3], "aligned") != 0) {
                perr(p, opline, "only wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 or .f32.bf16.bf16.f32 supported");
                return -1;
            }
            int nlayout = 0, ntype = 0, lay[2] = {-1, -1};
            char ty[4][16] = {{0}};
            for (int i = 4; i < mnp; i++) {
                if (strcmp(mp[i], "row") == 0 || strcmp(mp[i], "col") == 0) {
                    if (nlayout < 2)
                        lay[nlayout] =
                            strcmp(mp[i], "row") == 0 ? 1 : 0;
                    nlayout++;
                } else if (strcmp(mp[i], "m16n16k16") == 0) {
                    shape = 0;
                } else if (strcmp(mp[i], "m8n32k16") == 0) {
                    shape = 1;
                } else if (strcmp(mp[i], "m32n8k16") == 0) {
                    shape = 2;
                } else if (strcmp(mp[i], "m16n16k8") == 0) {
                    shape = 3;
                } else if (strcmp(mp[i], "m8n8k32") == 0) {
                    shape = 4;
                } else if (strcmp(mp[i], "f32") == 0 ||
                           strcmp(mp[i], "f16") == 0 ||
                           strcmp(mp[i], "bf16") == 0 ||
                           strcmp(mp[i], "tf32") == 0 ||
                           strcmp(mp[i], "u8") == 0 ||
                           strcmp(mp[i], "s8") == 0 ||
                           strcmp(mp[i], "u4") == 0 ||
                           strcmp(mp[i], "s4") == 0 ||
                           strcmp(mp[i], "s32") == 0) {
                    if (ntype < 4)
                        snprintf(ty[ntype], sizeof(ty[ntype]), "%s", mp[i]);
                    ntype++;
                } else {
                    perr(p, opline, "only f16 or bf16 wmma multiplicands are supported");
                    return -1;
                }
            }
            int bf16 = ntype == 4 && strcmp(ty[0], "f32") == 0 &&
                       strcmp(ty[1], "bf16") == 0 &&
                       strcmp(ty[2], "bf16") == 0 &&
                       strcmp(ty[3], "f32") == 0;
            int f16acc = ntype == 2 && strcmp(ty[0], "f16") == 0 &&
                         strcmp(ty[1], "f16") == 0;
            int tf32 = ntype == 4 && strcmp(ty[0], "f32") == 0 &&
                       strcmp(ty[1], "tf32") == 0 &&
                       strcmp(ty[2], "tf32") == 0 &&
                       strcmp(ty[3], "f32") == 0;
            int f16 = ntype == 2 && strcmp(ty[0], "f32") == 0 &&
                      strcmp(ty[1], "f32") == 0;
            int int8 = ntype == 4 && strcmp(ty[0], "s32") == 0 &&
                       strcmp(ty[3], "s32") == 0 &&
                       (strcmp(ty[1], "u8") == 0 ||
                        strcmp(ty[1], "s8") == 0) &&
                       strcmp(ty[2], ty[1]) == 0;
            int int4 = ntype == 4 && strcmp(ty[0], "s32") == 0 &&
                       strcmp(ty[3], "s32") == 0 &&
                       (strcmp(ty[1], "u4") == 0 ||
                        strcmp(ty[1], "s4") == 0) &&
                       strcmp(ty[2], ty[1]) == 0;
            if (shape < 0 || nlayout != 2 || lay[0] != 1 ||
                lay[1] != 0 ||
                (!f16 && !bf16 && !f16acc && !tf32 && !int8 && !int4) ||
                (tf32 && shape != 3) || (shape == 3 && !tf32) ||
                (int8 && shape == 3) || (int4 && shape != 4) ||
                (shape == 4 && !int4)) {
                perr(p, opline, "only wmma.mma.sync.aligned.row.col.m16n16k16.{f32.f32,f16.f16,f32.bf16.bf16.f32} supported");
                return -1;
            }
            if (shape != 0 && f16acc) {
                perr(p, opline, "only f32 accumulate is supported for wmma m8n32k16/m32n8k16");
                return -1;
            }
            row = lay[0];
            in_mma_bf16 = bf16;
            in_mma_f16acc = f16acc;
            in_mma_tf32 = tf32;
            in_mma_int = int8 || int4;
            in_mma_int4 = int4;
            in_mma_int8_signed =
                (int8 && strcmp(ty[1], "s8") == 0) ||
                (int4 && strcmp(ty[1], "s4") == 0);
        }
        ptx_instr_t in;
        memset(&in, 0, sizeof(in));
        in.line = opline;
        in.space = in_space;
        in.wmma_kind = kind;
        in.wmma_row = row;
        in.wmma_shape = shape;
        in.mma_elem = in_mma_int ? 2 : in_mma_tf32 ? 2
                                 : in_mma_bf16 ? 1 : 0;
        in.mma_f16acc = in_mma_f16acc;
        if (in_mma_int) {
            in.mma_asign = in_mma_int8_signed;
            in.mma_bsign = in_mma_int8_signed;
            in.ty = PTY_U32;
            in.ty2 = PTY_U32;
        }
        if (is_load) {
            in.op = POP_WMMA_LOAD;
            int nregs;
            if (in_mma_int) {
                if (shape == 4 || shape == 5)
                    nregs = kind == 2 ? 2 : 1;
                else if (kind == 2)
                    nregs = 8;
                else if (shape == 0)
                    nregs = 2;
                else if (shape == 1)
                    nregs = kind == 0 ? 1 : 4;
                else
                    nregs = kind == 0 ? 4 : 1;
            } else {
                nregs = (kind <= 1 && dtype_ok == 3 && shape == 0) ? 4
                        : (kind <= 1 && dtype_ok == 4)           ? 4
                        : (kind == 2 && in_mma_f16acc)            ? 4
                                                                  : 8;
            }
            if (wmma_regs(p, in.wregs[0], nregs, "wmma load") != 0)
                return -1;
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
                return -1;
            if (in.src[0].kind != OPD_ADDR) {
                perr(p, opline, "wmma.load address must use []");
                return -1;
            }
            in.nsrc = 1;
            if (at_punct(p, ',')) {
                if (parse_operand(p, &in.src[1]) != 0)
                    return -1;
                in.nsrc = 2;
            }
            if (expect_punct(p, ';', "after wmma.load") != 0)
                return -1;
            return block_add_instr(fn, bi, &in, p, opline);
        }
        if (is_store) {
            in.op = POP_WMMA_STORE;
            if (parse_operand(p, &in.src[0]) != 0)
                return -1;
            if (in.src[0].kind != OPD_ADDR) {
                perr(p, opline, "wmma.store address must use []");
                return -1;
            }
            if (expect_comma(p) != 0 ||
                wmma_regs(p, in.wregs[0],
                          in_mma_f16acc ? 4 : (in_mma_int4 ? 2 : 8),
                          "wmma store") != 0)
                return -1;
            in.nsrc = 1;
            if (at_punct(p, ',')) {
                if (parse_operand(p, &in.src[1]) != 0)
                    return -1;
                in.nsrc = 2;
            }
            if (expect_punct(p, ';', "after wmma.store") != 0)
                return -1;
            return block_add_instr(fn, bi, &in, p, opline);
        }
        in.op = POP_WMMA_MMA;
        int na, nb;
        int ncd = in_mma_f16acc ? 4 : (in_mma_int4 ? 2 : 8);
        if (in_mma_int) {
            if (shape == 4) {
                na = 1;
                nb = 1;
            } else if (shape == 0) {
                na = 2;
                nb = 2;
            } else if (shape == 1) {
                na = 1;
                nb = 4;
            } else {
                na = 4;
                nb = 1;
            }
        } else {
            na = nb =
                ((in_mma_bf16 && shape == 0) || in_mma_tf32) ? 4 : 8;
        }
        if (wmma_regs(p, in.wregs[0], ncd, "wmma D") != 0 ||
            expect_comma(p) != 0 ||
            wmma_regs(p, in.wregs[1], na, "wmma A") != 0 ||
            expect_comma(p) != 0 ||
            wmma_regs(p, in.wregs[2], nb, "wmma B") != 0 ||
            expect_comma(p) != 0 ||
            wmma_regs(p, in.wregs[3], ncd, "wmma C") != 0)
            return -1;
        in.nsrc = 0;
        if (expect_punct(p, ';', "after wmma.mma") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strncmp(opname, "ldmatrix", 8) == 0 && opname[8] == '.') {
        // Warp-level shared-memory matrix load (see emit_ldmatrix):
        ptx_instr_t in;
        memset(&in, 0, sizeof(in));
        in.line = opline;
        // only ldmatrix.sync.aligned.m8n8.x{1,2,4}[.trans].b16 (the
        // 8x8 shape is 16-bit only; the .m16n16/.m8n16 shapes and the
        // .b8/.b8x16 compressed formats stay clean errors).
        char mp[8][32];
        int mnp = 0;
        {
            const char *s = opname;
            memset(mp, 0, sizeof(mp));
            while (*s && mnp < 8) {
                size_t n = 0;
                while (s[n] && s[n] != '.' && n + 1 < 32)
                    n++;
                memcpy(mp[mnp], s, n);
                mnp++;
                s += n;
                if (*s == '.')
                    s++;
            }
            if (*s) {
                perr(p, opline, "ldmatrix: too many qualifiers");
                return -1;
            }
        }
        int num = 0;
        int seen_shape = 0, seen_num = 0, seen_type = 0, seen_trans = 0;
        if (mnp < 5 || strcmp(mp[0], "ldmatrix") != 0 ||
            strcmp(mp[1], "sync") != 0 || strcmp(mp[2], "aligned") != 0) {
            perr(p, opline, "only ldmatrix.sync.aligned.m8n8.x{1,2,4}[.trans].b16 supported");
            return -1;
        }
        // The shape / num / trans / space qualifiers appear in either
        // order across real code (`...m8n8.x2.b16` vs
        // `...x4.trans.m8n8.shared.b16`), so collect them by name.
        for (int i = 3; i < mnp; i++) {
            if (strcmp(mp[i], "m8n8") == 0 && !seen_shape)
                seen_shape = 1;
            else if (strcmp(mp[i], "x1") == 0 && !seen_num) {
                seen_num = 1;
                num = 1;
            } else if (strcmp(mp[i], "x2") == 0 && !seen_num) {
                seen_num = 1;
                num = 2;
            } else if (strcmp(mp[i], "x4") == 0 && !seen_num) {
                seen_num = 1;
                num = 4;
            } else if (strcmp(mp[i], "trans") == 0 && !seen_trans) {
                seen_trans = 1;
            } else if (strcmp(mp[i], "b16") == 0 && !seen_type) {
                seen_type = 1;
            } else if (strcmp(mp[i], "shared") == 0) {
                // implied state space (the operand must be shared anyway)
            } else {
                perr(p, opline, "only ldmatrix.sync.aligned.m8n8.x{1,2,4}[.trans].b16 supported");
                return -1;
            }
        }
        if (!seen_shape || !seen_num || !seen_type) {
            perr(p, opline, "only ldmatrix.sync.aligned.m8n8.x{1,2,4}[.trans].b16 supported");
            return -1;
        }
        if (pred[0] || pred_neg) {
            perr(p, opline, "predicated ldmatrix unsupported (needs convergent guard)");
            return -1;
        }
        in.op = POP_LDMATRIX;
        in.ldm_num = num;
        in.ldm_trans = seen_trans;
        in.ty = PTY_U32;
        // Destination: brace list of `num` .b32 registers.
        if (expect_punct(p, '{', "ldmatrix destination") != 0)
            return -1;
        for (int i = 0; i < num; i++) {
            ptx_tok_t t;
            if (next(p, &t) != 0)
                return -1;
            if (t.kind != T_IDENT) {
                perr(p, t.line, "expected ldmatrix destination register");
                return -1;
            }
            if (copy_name(in.mregs[i], sizeof(in.mregs[i]), t.text, p,
                          t.line) != 0)
                return -1;
            if (i + 1 < num && expect_comma(p) != 0)
                return -1;
        }
        if (expect_punct(p, '}', "after ldmatrix destinations") != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (in.src[0].kind != OPD_ADDR) {
            perr(p, opline, "ldmatrix address must use []");
            return -1;
        }
        in.nsrc = 1;
        if (expect_punct(p, ';', "after ldmatrix") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strncmp(opname, "mma.sp", 6) == 0 &&
        (opname[6] == '.' ||
         strncmp(opname + 6, "::ordered_metadata", 18) == 0)) {
        // Sparse mma m16n8k16 f16/bf16 -> f32 (2:4 sparsity). Fragments:
        // 4 dst + 2 compressed A + 2 B + 4 C, plus a metadata .b32 and
        // a sparsity-selector immediate. Other sparse shapes/types stay
        // clean errors.
        char mp[16][32];
        int mnp = 0;
        {
            const char *s = opname;
            memset(mp, 0, sizeof(mp));
            while (*s && mnp < 12) {
                size_t n = 0;
                while (s[n] && s[n] != '.' && n + 1 < 32)
                    n++;
                if (s[n] == '.' && n + 1 >= 32)
                    break;
                memcpy(mp[mnp], s, n);
                mp[mnp][n] = '\0';
                mnp++;
                s += n;
                if (*s == '.')
                    s++;
                else if (*s)
                    break;
            }
            if (*s) {
                perr(p, opline, "bad opcode 'mma.sp' (too long)");
                return -1;
            }
        }
        int shape_ok = strcmp(mp[4], "m16n8k8") == 0 ||
                       strcmp(mp[4], "m16n8k16") == 0 ||
                       strcmp(mp[4], "m16n8k32") == 0 ||
                       strcmp(mp[4], "m16n8k64") == 0 ||
                       strcmp(mp[4], "m16n8k128") == 0;
        int float_ok =
            strcmp(mp[7], "f32") == 0 &&
            (strcmp(mp[8], "f16") == 0 || strcmp(mp[8], "bf16") == 0 ||
             (strcmp(mp[8], "tf32") == 0 &&
              strcmp(mp[4], "m16n8k32") != 0)) &&
            strcmp(mp[9], mp[8]) == 0 && strcmp(mp[10], "f32") == 0;
        int int4_ok =
            strcmp(mp[7], "s32") == 0 &&
            (strcmp(mp[4], "m16n8k64") == 0 ||
             strcmp(mp[4], "m16n8k128") == 0) &&
            (strcmp(mp[8], "u4") == 0 || strcmp(mp[8], "s4") == 0) &&
            strcmp(mp[9], mp[8]) == 0 && strcmp(mp[10], "s32") == 0;
        if (mnp != 11 ||
            (strcmp(mp[1], "sp") != 0 &&
             strcmp(mp[1], "sp::ordered_metadata") != 0) ||
            strcmp(mp[2], "sync") != 0 || strcmp(mp[3], "aligned") != 0 ||
            !shape_ok || strcmp(mp[5], "row") != 0 ||
            strcmp(mp[6], "col") != 0 || !(float_ok || int4_ok)) {
            perr(p, opline, "only mma.sp.sync.aligned.m16n8k8/m16n8k16/m16n8k32.row.col.f32.f16/bf16/tf32 and m16n8k64.s32.u4/s4 supported");
            return -1;
        }
        int is_int4 = strcmp(mp[8], "u4") == 0 || strcmp(mp[8], "s4") == 0;
        int is_tf32 = strcmp(mp[8], "tf32") == 0;
        int sk = strcmp(mp[4], "m16n8k128") == 0 ? 128
                 : strcmp(mp[4], "m16n8k64") == 0 ? 64
                 : strcmp(mp[4], "m16n8k32") == 0 ? 32
                 : strcmp(mp[4], "m16n8k16") == 0 ? 16 : 8;
        if (pred[0] || pred_neg) {
            perr(p, opline, "predicated mma.sp unsupported (needs convergent guard)");
            return -1;
        }
        ptx_instr_t in;
        memset(&in, 0, sizeof(in));
        in.line = opline;
        in.op = POP_MMA;
        in.mma_sparse = 1;
        in.ty = (is_int4) ? PTY_U32 : PTY_F32;
        in.ty2 = is_int4 ? PTY_U32
                : is_tf32 ? PTY_F32
                : strcmp(mp[8], "bf16") == 0 ? PTY_BF16 : PTY_F16;
        in.mma_elem = is_int4 ? 7
                    : is_tf32 ? 3
                    : strcmp(mp[8], "bf16") == 0 ? 1 : 0;
        if (is_int4) {
            in.mma_asign = strcmp(mp[8], "s4") == 0;
            in.mma_bsign = strcmp(mp[9], "s4") == 0;
        }
        in.mma_k = sk;
        int sn = is_int4 ? (sk == 128 ? 4 : 2)
                 : is_tf32 ? (sk == 16 ? 4 : 2)
                 : (sk == 32 ? 4 : 2);
        int scoff = 8 + sn;
        if (parse_brace_regs(p, &in.mregs[0], 4, opline) != 0 ||
            expect_comma(p) != 0 ||
            parse_brace_regs(p, &in.mregs[4], sn, opline) != 0 ||
            expect_comma(p) != 0 ||
            parse_brace_regs(p, &in.mregs[8], sn, opline) != 0 ||
            expect_comma(p) != 0 ||
            parse_brace_regs(p, &in.mregs[scoff], 4, opline) != 0)
            return -1;
        if (expect_comma(p) != 0 ||
            parse_dst(p, in.smreg, sizeof(in.smreg), 0) != 0)
            return -1;
        if (expect_comma(p) != 0)
            return -1;
        ptx_tok_t st;
        if (next(p, &st) != 0)
            return -1;
        int maxsel = (is_int4 && sk == 128) ? 0
                     : (sk == 64 || sk == 32 || (is_tf32 && sk == 16)) ? 1
                                                                        : 3;
        if (st.kind != T_INT || st.is_float || st.imm > (unsigned)maxsel) {
            perr(p, opline, maxsel == 1
                              ? "mma.sp selector must be 0..1 for this shape"
                              : "mma.sp sparsity selector must be an integer 0..3");
            return -1;
        }
        in.mma_sel = (int)st.imm;
        if (expect_punct(p, ';', "after mma.sp") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    if (strncmp(opname, "mma", 3) == 0 &&
        (opname[3] == '\0' || opname[3] == '.')) {
        // Warp tensor matmul (see emit_mma + docs/supported-isa.md):
        // only .sync.aligned.m16n8k16.row.col.f32.f16/bf16.f32 (other
        // shapes/layouts/types stay clean errors). Fragments are
        // brace register lists: 4 dst + 4 A + 2 B + 4 C.
        char mp[16][32];
        int mnp = 0;
        {
            const char *s = opname;
            memset(mp, 0, sizeof(mp));
            while (*s && mnp < 16) {
                size_t n = 0;
                while (s[n] && s[n] != '.' && n + 1 < 32)
                    n++;
                if (s[n] == '.' && n + 1 >= 32)
                    break;
                memcpy(mp[mnp], s, n);
                mp[mnp][n] = '\0';
                mnp++;
                s += n;
                if (*s == '.')
                    s++;
                else if (*s)
                    break;
            }
            if (*s) {
                perr(p, opline, "bad opcode 'mma' (too long)");
                return -1;
            }
        }
        ptx_instr_t in;
        memset(&in, 0, sizeof(in));
        in.line = opline;
        if (pred[0] || pred_neg) {
            perr(p, opline, "predicated mma.sync unsupported (needs convergent guard)");
            return -1;
        }
        // Supported dense shapes (all row.col, A fragments low to high):
        //   .m16n8k16.f32.f16/bf16.f32   4 A + 2 B + 4 C
        //   .m16n8k8.f32.f16/bf16.f32    2 A + 1 B + 4 C
        //   .m16n8k4.f32.tf32.tf32.f32   2 A + 1 B + 4 C
        //   .m16n8k8.f32.tf32.tf32.f32   4 A + 2 B + 4 C
        //   .m16n8k32.s32.s8.s8.s32      4 A + 2 B + 4 C
        int is_s8 = 0, is_tf32 = 0, is_f64 = 0, is_m8f16 = 0;
        int is_f64_m16 = 0;
        int is_int4 = 0;
        int int4_asign = 0, int4_bsign = 0;
        int is_b1 = 0, b1_xor = 0;
        int fp8_type = 0; // 1 = e4m3, 2 = e5m2 (8-bit, m16n8k32)
        int kdim = 16, f16acc = 0;
        int f64rnd = -1;
        if (mnp == 11) {
            if (strcmp(mp[10], "rn") == 0)
                f64rnd = 0;
            else if (strcmp(mp[10], "rz") == 0)
                f64rnd = 1;
            else if (strcmp(mp[10], "rm") == 0)
                f64rnd = 2;
            else if (strcmp(mp[10], "rp") == 0)
                f64rnd = 3;
        }
        int common = mnp == 10 && strcmp(mp[1], "sync") == 0 &&
                     strcmp(mp[2], "aligned") == 0 &&
                     strcmp(mp[4], "row") == 0 &&
                     strcmp(mp[5], "col") == 0;
        int f64common = (mnp == 10 || f64rnd >= 0) &&
                        strcmp(mp[1], "sync") == 0 &&
                        strcmp(mp[2], "aligned") == 0 &&
                        strcmp(mp[4], "row") == 0 &&
                        strcmp(mp[5], "col") == 0;
        if (mnp == 12 && strcmp(mp[1], "sync") == 0 &&
            strcmp(mp[2], "aligned") == 0 &&
            (strcmp(mp[3], "m8n8k128") == 0 ||
             strcmp(mp[3], "m16n8k128") == 0 ||
             strcmp(mp[3], "m16n8k256") == 0) &&
            strcmp(mp[4], "row") == 0 && strcmp(mp[5], "col") == 0 &&
            strcmp(mp[6], "s32") == 0 && strcmp(mp[7], "b1") == 0 &&
            strcmp(mp[8], "b1") == 0 && strcmp(mp[9], "s32") == 0 &&
            (strcmp(mp[10], "and") == 0 ||
             strcmp(mp[10], "xor") == 0) &&
            strcmp(mp[11], "popc") == 0) {
            is_b1 = 1;
            b1_xor = strcmp(mp[10], "xor") == 0;
            kdim = strcmp(mp[3], "m16n8k256") == 0 ? 256 : 128;
        } else if (common &&
            (strcmp(mp[3], "m8n8k32") == 0 ||
             strcmp(mp[3], "m16n8k32") == 0 ||
             strcmp(mp[3], "m16n8k64") == 0) &&
            strcmp(mp[6], "s32") == 0 &&
            (strcmp(mp[7], "s4") == 0 || strcmp(mp[7], "u4") == 0) &&
            (strcmp(mp[8], "s4") == 0 || strcmp(mp[8], "u4") == 0) &&
            strcmp(mp[9], "s32") == 0) {
            is_int4 = 1;
            int4_asign = strcmp(mp[7], "s4") == 0;
            int4_bsign = strcmp(mp[8], "s4") == 0;
            kdim = strcmp(mp[3], "m16n8k64") == 0 ? 64 : 32;
        } else if (common &&
            (strcmp(mp[3], "m16n8k32") == 0 ||
             strcmp(mp[3], "m16n8k16") == 0) &&
            strcmp(mp[6], "s32") == 0 &&
            (strcmp(mp[7], "s8") == 0 || strcmp(mp[7], "u8") == 0) &&
            (strcmp(mp[8], "s8") == 0 || strcmp(mp[8], "u8") == 0) &&
            strcmp(mp[9], "s32") == 0) {
            is_s8 = 1;
            kdim = strcmp(mp[3], "m16n8k32") == 0 ? 32 : 16;
            in.mma_asign = strcmp(mp[7], "s8") == 0;
            in.mma_bsign = strcmp(mp[8], "s8") == 0;
        } else if (common && strcmp(mp[3], "m8n8k16") == 0 &&
                   strcmp(mp[6], "s32") == 0 &&
                   (strcmp(mp[7], "s8") == 0 || strcmp(mp[7], "u8") == 0) &&
                   (strcmp(mp[8], "s8") == 0 || strcmp(mp[8], "u8") == 0) &&
                   strcmp(mp[9], "s32") == 0) {
            is_s8 = 1;
            in.mma_m8s8 = 1;
            kdim = 16;
            in.mma_asign = strcmp(mp[7], "s8") == 0;
            in.mma_bsign = strcmp(mp[8], "s8") == 0;
        } else if (mnp == 11 &&
                   strcmp(mp[1], "sync") == 0 &&
                   strcmp(mp[2], "aligned") == 0 &&
                   strcmp(mp[3], "m16n8k32") == 0 &&
                   strcmp(mp[4], "row") == 0 &&
                   strcmp(mp[5], "col") == 0 &&
                   strcmp(mp[6], "f32") == 0 &&
                   strcmp(mp[7], "kind::f8f6f4") == 0 &&
                   (strcmp(mp[8], "e4m3") == 0 ||
                    strcmp(mp[8], "e5m2") == 0) &&
                   strcmp(mp[9], mp[8]) == 0 &&
                   strcmp(mp[10], "f32") == 0) {
            // FP8 e4m3/e5m2 m16n8k32, same 8-bit fragment layout as
            // the s8 form: 4 A + 2 B + 4 f32 C/D registers.
            kdim = 32;
            fp8_type = strcmp(mp[8], "e4m3") == 0 ? 1 : 2;
        } else if (common && strcmp(mp[3], "m16n8k32") == 0 &&
                   strcmp(mp[6], "f32") == 0 &&
                   (strcmp(mp[7], "e4m3") == 0 ||
                    strcmp(mp[7], "e5m2") == 0) &&
                   strcmp(mp[8], mp[7]) == 0 &&
                   strcmp(mp[9], "f32") == 0) {
            // Plain FP8 form (sm_89): no kind:: qualifier.
            kdim = 32;
            fp8_type = strcmp(mp[7], "e4m3") == 0 ? 1 : 2;
        } else if (common && strcmp(mp[3], "m16n8k16") == 0 &&
                   (strcmp(mp[6], "f32") == 0 ||
                    strcmp(mp[6], "f16") == 0) &&
                   (strcmp(mp[7], "f16") == 0 ||
                    strcmp(mp[7], "bf16") == 0) &&
                   strcmp(mp[8], mp[7]) == 0 &&
                   strcmp(mp[9], mp[6]) == 0 &&
                   !(strcmp(mp[7], "bf16") == 0 &&
                     strcmp(mp[6], "f16") == 0)) {
            kdim = 16;
            f16acc = strcmp(mp[6], "f16") == 0;
        } else if (common && strcmp(mp[3], "m16n8k8") == 0 &&
                   (strcmp(mp[6], "f32") == 0 ||
                    strcmp(mp[6], "f16") == 0) &&
                   (strcmp(mp[7], "f16") == 0 ||
                    strcmp(mp[7], "bf16") == 0) &&
                   strcmp(mp[8], mp[7]) == 0 &&
                   strcmp(mp[9], mp[6]) == 0 &&
                   !(strcmp(mp[7], "bf16") == 0 &&
                     strcmp(mp[6], "f16") == 0)) {
            kdim = 8;
            f16acc = strcmp(mp[6], "f16") == 0;
        } else if (common && strcmp(mp[3], "m8n8k4") == 0 &&
                   strcmp(mp[6], "f32") == 0 &&
                   strcmp(mp[7], "f16") == 0 &&
                   strcmp(mp[8], "f16") == 0 &&
                   strcmp(mp[9], "f32") == 0) {
            is_m8f16 = 1;
            kdim = 4;
        } else if (f64common &&
                   (strcmp(mp[3], "m16n8k4") == 0 ||
                    strcmp(mp[3], "m16n8k8") == 0 ||
                    strcmp(mp[3], "m16n8k16") == 0) &&
                   strcmp(mp[6], "f64") == 0 &&
                   strcmp(mp[7], "f64") == 0 &&
                   strcmp(mp[8], "f64") == 0 &&
                   strcmp(mp[9], "f64") == 0) {
            is_f64 = 1;
            is_f64_m16 = 1;
            kdim = strcmp(mp[3], "m16n8k4") == 0 ? 4
                   : strcmp(mp[3], "m16n8k8") == 0 ? 8 : 16;
        } else if (f64common && strcmp(mp[3], "m8n8k4") == 0 &&
                   strcmp(mp[6], "f64") == 0 &&
                   strcmp(mp[7], "f64") == 0 &&
                   strcmp(mp[8], "f64") == 0 &&
                   strcmp(mp[9], "f64") == 0) {
            is_f64 = 1;
            kdim = 4;
        } else if (common &&
                   (strcmp(mp[3], "m16n8k4") == 0 ||
                    strcmp(mp[3], "m16n8k8") == 0) &&
                   strcmp(mp[6], "f32") == 0 &&
                   strcmp(mp[7], "tf32") == 0 &&
                   strcmp(mp[8], "tf32") == 0 &&
                   strcmp(mp[9], "f32") == 0) {
            is_tf32 = 1;
            kdim = strcmp(mp[3], "m16n8k4") == 0 ? 4 : 8;
        } else {
            perr(p, opline, "only mma.sync.aligned.m16n8k16.{f32,f16}.f16/bf16.{f32,f16}, m16n8k8.{f32,f16}.f16/bf16.{f32,f16}, m16n8k4/m16n8k8.f32.tf32.tf32.f32, m8n8k4.f32.f16.f16.f32 / .f64.f64.f64.f64, and m8n8k16/m16n8k16/m16n8k32.s32.{s8,u8}.{s8,u8}.s32 supported");
            return -1;
        }
        in.op = POP_MMA;
        in.ty = (is_s8 || is_int4 || is_b1) ? PTY_U32
                : is_f64 ? PTY_F64
                : f16acc ? PTY_F16X2 : PTY_F32;
        in.ty2 = (is_s8 || is_int4 || is_b1) ? PTY_U32
                : is_tf32 || is_f64 ? PTY_F32
                : strcmp(mp[7], "bf16") == 0 ? PTY_BF16 : PTY_F16;
        in.mma_elem = is_b1 ? 8
                    : is_int4 ? 7
                    : is_s8 ? 2
                    : is_tf32 ? 3
                    : is_f64 ? 4
                    : fp8_type == 1 ? 5
                    : fp8_type == 2 ? 6
                    : strcmp(mp[7], "bf16") == 0 ? 1 : 0;
        if (is_int4) {
            in.mma_asign = int4_asign;
            in.mma_bsign = int4_bsign;
            in.mma_m8s8 = strcmp(mp[3], "m8n8k32") == 0;
        }
        if (is_b1) {
            in.mma_b1_xor = b1_xor;
            in.mma_m8s8 = strcmp(mp[3], "m8n8k128") == 0;
        }
        if (fp8_type)
            in.ty2 = PTY_U32;
        if (is_f64)
            in.ty2 = PTY_F64;
        in.mma_k = kdim;
        in.mma_f16acc = f16acc;
        in.mma_m8f16 = is_m8f16;
        in.mma_f64_m16 = is_f64_m16;
        in.frnd = f64rnd >= 0 ? f64rnd : 0;
        int na, nb, nc;
        if (in.mma_m8s8) {
            na = 1;
            nb = 1;
            nc = 2;
        } else if (is_b1) {
            na = kdim == 256 ? 4 : 2;
            nb = kdim == 256 ? 2 : 1;
            nc = 4;
        } else if (is_int4) {
            na = kdim == 64 ? 4 : 2;
            nb = kdim == 64 ? 2 : 1;
            nc = 4;
        } else if (is_s8) {
            na = kdim == 16 ? 2 : 4;
            nb = kdim == 16 ? 1 : 2;
            nc = 4;
        } else if (is_m8f16) {
            na = 2;
            nb = 2;
            nc = 8;
        } else if (is_f64_m16) {
            na = kdim == 4 ? 2 : kdim == 8 ? 4 : 8;
            nb = kdim == 4 ? 1 : kdim == 8 ? 2 : 4;
            nc = 4;
        } else if (is_f64) {
            na = 1;
            nb = 1;
            nc = 2;
        } else if (is_tf32) {
            na = kdim == 4 ? 2 : 4;
            nb = kdim == 4 ? 1 : 2;
            nc = 4;
        } else {
            na = (kdim == 8 || kdim == 4) ? 2 : 4;
            nb = (kdim == 8 || kdim == 4) ? 1 : 2;
            nc = f16acc ? 2 : 4;
        }
        int aoff = is_m8f16 ? 8 : 4;
        int boff = is_m8f16 ? 10 : is_f64_m16 ? 4 + na : 8;
        int coff = is_m8f16 ? 12 : is_f64_m16 ? 4 + na + nb : 10;
        if (parse_brace_regs(p, &in.mregs[0], nc, opline) != 0)
            return -1;
        if (expect_comma(p) != 0)
            return -1;
        if (parse_brace_regs(p, &in.mregs[aoff], na, opline) != 0)
            return -1;
        if (expect_comma(p) != 0)
            return -1;
        if (parse_brace_regs(p, &in.mregs[boff], nb, opline) != 0)
            return -1;
        if (expect_comma(p) != 0)
            return -1;
        if (parse_brace_regs(p, &in.mregs[coff], nc, opline) != 0)
            return -1;
        if (expect_punct(p, ';', "after mma") != 0)
            return -1;
        in.nsrc = 0;
        return block_add_instr(fn, bi, &in, p, opline);
    }
    perr(p, opline, "unhandled matrix instruction");
    return -1;
}
