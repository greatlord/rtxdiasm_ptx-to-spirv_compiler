// Atomic instruction parsing (atom[.sem][.scope][.space].op.type).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ptx.h"
#include "parse/parse_internal.h"

int parse_atom_instr(parser_t *p, ptx_func_t *fn, size_t bi,
                     const char *opname, int opline) {
        int is_red = opname[0] == 'r';
        char ap[7][32];
        int anp = 0;
        {
            const char *s = opname;
            memset(ap, 0, sizeof(ap));
            while (*s && anp < 7) {
                size_t n = 0;
                while (s[n] && s[n] != '.' && n + 1 < 32)
                    n++;
                if (s[n] == '.' && n + 1 >= 32) {
                    perr(p, opline, "atom qualifier too long");
                    return -1;
                }
                memcpy(ap[anp], s, n);
                ap[anp][n] = '\0';
                anp++;
                s += n;
                if (*s == '.')
                    s++;
                else if (*s)
                    break;
            }
            if (*s) {
                perr(p, opline, "unsupported atom form");
                return -1;
            }
        }
        ptx_instr_t in;
        memset(&in, 0, sizeof(in));
        in.line = opline;
        in.space = PSP_GENERIC;
        int has_cache_hint = 0;
        const char *aop = NULL;
        int ty_idx = -1;
        /* PTX qualifier order varies across ISA versions/forms (e.g.
         * atom{.sem}{.scope}{.space}.op.type vs atom.op{.sem}{.scope}.type).
         * Classify tokens flexibly instead of assuming a fixed sequence. */
        for (int i = 1; i < anp; i++) {
            const char *t = ap[i];
            if (strcmp(t, "relaxed") == 0 || strcmp(t, "acquire") == 0 ||
                strcmp(t, "release") == 0 || strcmp(t, "acq_rel") == 0) {
                continue;
            }
            if (strcmp(t, "cta") == 0 || strcmp(t, "cluster") == 0 ||
                strcmp(t, "gpu") == 0 || strcmp(t, "sys") == 0) {
                continue;
            }
            if (strcmp(t, "global") == 0) {
                in.space = PSP_GLOBAL;
                continue;
            }
            if (strcmp(t, "shared") == 0 ||
                strcmp(t, "shared::cta") == 0 ||
                strcmp(t, "shared::cluster") == 0) {
                in.space = PSP_SHARED;
                continue;
            }
            if (strcmp(t, "L2::cache_hint") == 0) {
                has_cache_hint = 1;
                continue;
            }
            if (strcmp(t, "noftz") == 0) {
                continue;
            }
            if (!aop && (strcmp(t, "add") == 0 || strcmp(t, "inc") == 0 ||
                         strcmp(t, "dec") == 0 || strcmp(t, "and") == 0 ||
                         strcmp(t, "or") == 0 || strcmp(t, "xor") == 0 ||
                         strcmp(t, "exch") == 0 || strcmp(t, "min") == 0 ||
                         strcmp(t, "max") == 0 || strcmp(t, "cas") == 0)) {
                aop = t;
                continue;
            }
            {
                /* Distinguish type tokens from unknown ops/qualifiers for better diagnostics. */
                ptx_ty_t ty_tmp;
                if (map_ty(t, &ty_tmp) == 0) {
                    if (ty_idx < 0) {
                        ty_idx = i;
                        continue;
                    }
                } else if (!aop) {
                    char m[640];
                    snprintf(m, sizeof(m), "unsupported atom op '%s'", t);
                    perr(p, opline, m);
                    return -1;
                }
            }
            perr(p, opline, "unsupported atom form");
            return -1;
        }
        if (!aop) {
            perr(p, opline, "atom needs an op (.add/.inc/.cas)");
            return -1;
        }
        int is_add = strcmp(aop, "add") == 0;
        int is_inc = strcmp(aop, "inc") == 0;
        int is_dec = strcmp(aop, "dec") == 0;
        int is_and = strcmp(aop, "and") == 0;
        int is_or = strcmp(aop, "or") == 0;
        int is_xor = strcmp(aop, "xor") == 0;
        int is_exch = strcmp(aop, "exch") == 0;
        int is_min = strcmp(aop, "min") == 0;
        int is_max = strcmp(aop, "max") == 0;
        int is_cas = strcmp(aop, "cas") == 0;
        if (ty_idx < 0) {
            perr(p, opline, "atom needs a type (.b32/.b64/.u32/.s32/.u64/.s64/.f32/.f64)");
            return -1;
        }
        if (map_ty(ap[ty_idx], &in.ty) != 0) {
            char m[640];
            snprintf(m, sizeof(m), "unsupported atom type '%s'", ap[ty_idx]);
            perr(p, opline, m);
            return -1;
        }
        {
            const char *ts = ap[ty_idx][0] == '.' ? ap[ty_idx] + 1 : ap[ty_idx];
            in.sign = ts[0] == 's';
        }
        if (is_red && (is_cas || is_exch)) {
            perr(p, opline, "red does not support .cas/.exch");
            return -1;
        }
        if (is_cas) {
            if (in.ty != PTY_U16 && in.ty != PTY_U32 &&
                in.ty != PTY_U64) {
                perr(p, opline, "atom.cas supports .b16/.b32/.b64 only");
                return -1;
            }
            in.op = POP_ATOMCAS;
        } else if (is_inc || is_dec) {
            if (in.ty != PTY_U32) {
                perr(p, opline, "atom.inc/dec supports .u32/.b32 only");
                return -1;
            }
            in.op = POP_ATOM;
            in.atom_op = is_inc ? 1 : 2;
        } else {
            if (in.ty == PTY_F32) {
                if (!is_add) {
                    perr(p, opline, "atom float forms support .add only");
                    return -1;
                }
                in.op = POP_ATOM;
                in.atom_op = 0;
            } else if (in.ty == PTY_F64) {
                if (!is_add) {
                    perr(p, opline, "atom.f64 supports .add only");
                    return -1;
                }
                in.op = POP_ATOM;
                in.atom_op = 0;
            } else if (in.ty == PTY_F16 || in.ty == PTY_BF16 ||
                       in.ty == PTY_F16X2 || in.ty == PTY_BF16X2) {
                if (!is_add) {
                    perr(p, opline,
                         "atom f16/bf16 supports .add only");
                    return -1;
                }
                in.op = POP_ATOM;
                in.atom_op = 0;
            } else if (in.ty == PTY_U32 || in.ty == PTY_U64) {
                in.op = POP_ATOM;
                in.atom_op = is_add ? 0
                            : is_and ? 3
                            : is_or ? 4
                            : is_xor ? 5
                            : is_exch ? 6
                            : is_min ? 7
                                     : 8;
            } else {
                perr(p, opline, "unsupported atom type");
                return -1;
            }
        }
        if (!is_red) {
            if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
                return -1;
            if (expect_comma(p) != 0)
                return -1;
        }
        if (parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (in.src[0].kind != OPD_ADDR) {
            perr(p, opline, is_red ? "red address must use []"
                                   : "atom address must use []");
            return -1;
        }
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (is_cas) {
            if (has_cache_hint) {
                perr(p, opline, "atom.cas takes no cache hint");
                return -1;
            }
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[2]) != 0)
                return -1;
            in.nsrc = 3;
        } else if (has_cache_hint) {
            // Optional cache-policy operand; performance hint only.
            if (expect_comma(p) != 0 || parse_operand(p, &in.src[2]) != 0)
                return -1;
            in.nsrc = 3;
        }
        if (expect_punct(p, ';', is_red ? "after red" : "after atom") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }
