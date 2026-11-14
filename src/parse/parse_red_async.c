// red.async / multimem.red.async parsing.
#include <stdio.h>
#include <string.h>
#include "ptx.h"
#include "parse/parse_internal.h"

#define RDA_MAX_PARTS 12

static int rda_split(const char *op, char parts[RDA_MAX_PARTS][32],
                     int *np) {
    *np = 0;
    const char *s = op;
    while (*s) {
        if (*np >= RDA_MAX_PARTS)
            return -1;
        size_t n = 0;
        while (s[n] && s[n] != '.' && n + 1 < 32)
            n++;
        if (s[n] == '.' && n + 1 >= 32)
            return -1;
        memcpy(parts[*np], s, n);
        parts[*np][n] = '\0';
        (*np)++;
        s += n;
        if (*s == '.')
            s++;
        else if (*s)
            return -1;
    }
    return 0;
}

int parse_red_async(parser_t *p, ptx_func_t *fn, size_t bi,
                    const char *pred, int pred_neg,
                    const char *opname, int opline) {
    char parts[RDA_MAX_PARTS][32];
    int np = 0;
    memset(parts, 0, sizeof(parts));
    if (rda_split(opname, parts, &np) != 0) {
        perr(p, opline, "bad red.async opcode");
        return -1;
    }
    int mm = strcmp(parts[0], "multimem") == 0;
    int pi = mm ? 3 : 2; // past red/multimem + red + async

    ptx_instr_t in;
    memset(&in, 0, sizeof(in));
    in.line = opline;
    in.op = POP_RED_ASYNC;
    if (copy_name(in.pred, sizeof(in.pred), pred[0] ? pred : "", p,
                  opline) != 0)
        return -1;
    in.pred_neg = pred_neg;

    int mmio = 0;
    if (pi < np && strcmp(parts[pi], "mmio") == 0) {
        mmio = 1;
        pi++;
    }
    if (pi < np && strcmp(parts[pi], "relaxed") == 0) {
        in.rda_release = 0;
        pi++;
    } else if (pi < np && strcmp(parts[pi], "release") == 0) {
        in.rda_release = 1;
        pi++;
    } else {
        perr(p, opline, "red.async needs .relaxed or .release");
        return -1;
    }
    in.rda_scope = -1;
    if (pi < np && strcmp(parts[pi], "cluster") == 0) {
        in.rda_scope = 0;
        pi++;
    } else if (pi < np && strcmp(parts[pi], "gpu") == 0) {
        in.rda_scope = 1;
        pi++;
    } else if (pi < np && strcmp(parts[pi], "sys") == 0) {
        in.rda_scope = 2;
        pi++;
    } else {
        perr(p, opline, "red.async needs .cluster, .gpu or .sys scope");
        return -1;
    }
    in.space = PSP_GENERIC;
    if (pi < np && strcmp(parts[pi], "shared::cluster") == 0) {
        in.space = PSP_SHARED;
        pi++;
    } else if (pi < np && strcmp(parts[pi], "global") == 0) {
        in.space = PSP_GLOBAL;
        pi++;
    }
    int comp = 0;
    if (pi < np && strcmp(parts[pi], "mbarrier::complete_tx::bytes") == 0) {
        comp = 1;
        pi++;
    }
    if (pi >= np) {
        perr(p, opline, "red.async needs a reduction op");
        return -1;
    }
    const char *aop = parts[pi++];
    int op = -1;
    if (strcmp(aop, "add") == 0) op = 0;
    else if (strcmp(aop, "inc") == 0) op = 1;
    else if (strcmp(aop, "dec") == 0) op = 2;
    else if (strcmp(aop, "and") == 0) op = 3;
    else if (strcmp(aop, "or") == 0) op = 4;
    else if (strcmp(aop, "xor") == 0) op = 5;
    else if (strcmp(aop, "min") == 0) op = 7;
    else if (strcmp(aop, "max") == 0) op = 8;
    if (op < 0) {
        char m[128];
        snprintf(m, sizeof(m), "unsupported red.async op '%s'", aop);
        perr(p, opline, m);
        return -1;
    }
    in.atom_op = op;
    if (pi >= np) {
        perr(p, opline, "red.async needs a type");
        return -1;
    }
    const char *ts = parts[pi][0] == '.' ? parts[pi] + 1 : parts[pi];
    if (map_ty(parts[pi], &in.ty) != 0) {
        char m[128];
        snprintf(m, sizeof(m), "unsupported red.async type '%s'", parts[pi]);
        perr(p, opline, m);
        return -1;
    }
    in.sign = ts[0] == 's';
    pi++;
    if (pi != np) {
        perr(p, opline, "unsupported red.async form");
        return -1;
    }

    int is_u32 = strcmp(ts, "u32") == 0 || strcmp(ts, "b32") == 0;
    int is_s32 = strcmp(ts, "s32") == 0;
    int is_u64 = strcmp(ts, "u64") == 0;
    int is_s64 = strcmp(ts, "s64") == 0;
    if (mm && p->sm && p->sm < 100) {
        perr(p, opline, "multimem.red.async requires sm_100 or higher");
        return -1;
    }
    if (in.rda_release && p->sm && p->sm < 100) {
        perr(p, opline,
             "red.async release/global forms require sm_100 or higher");
        return -1;
    }
    if (!in.rda_release && p->sm && p->sm < 90) {
        perr(p, opline, "red.async requires sm_90 or higher");
        return -1;
    }
    if (mm) {
        if (!in.rda_release) {
            perr(p, opline, "multimem.red.async needs .release");
            return -1;
        }
        if (in.rda_scope != 1 && in.rda_scope != 2) {
            perr(p, opline, "multimem.red.async needs .gpu or .sys");
            return -1;
        }
        if (in.space == PSP_SHARED) {
            perr(p, opline, "multimem.red.async destination must be global");
            return -1;
        }
        if (in.atom_op != 0) {
            perr(p, opline, "multimem.red.async supports .add only");
            return -1;
        }
        if (!is_u32 && !is_s32 && !is_u64) {
            perr(p, opline,
                 "multimem.red.async supports .u32/.s32/.u64 only");
            return -1;
        }
    } else if (in.rda_release) {
        if (in.rda_scope != 1 && in.rda_scope != 2) {
            perr(p, opline, "red.async.release needs .gpu or .sys");
            return -1;
        }
        if (in.space == PSP_SHARED) {
            perr(p, opline, "red.async.release destination must be global");
            return -1;
        }
        if (comp) {
            perr(p, opline,
                 "red.async.release does not take a completion mechanism");
            return -1;
        }
        if (in.atom_op != 0) {
            perr(p, opline, "red.async.release supports .add only");
            return -1;
        }
        if (!is_u32 && !is_s32 && !is_u64 && !is_s64) {
            perr(p, opline,
                 "red.async.release supports .u32/.s32/.u64/.s64 only");
            return -1;
        }
        if (mmio && in.rda_scope != 2) {
            perr(p, opline, "red.async.mmio requires .sys scope");
            return -1;
        }
    } else {
        if (in.rda_scope != 0) {
            perr(p, opline, "red.async.relaxed needs .cluster scope");
            return -1;
        }
        if (in.space != PSP_SHARED) {
            perr(p, opline,
                 "red.async.relaxed destination must be .shared::cluster");
            return -1;
        }
        if (mmio) {
            perr(p, opline, "red.async.mmio requires .release");
            return -1;
        }
        if (in.atom_op == 1 || in.atom_op == 2) {
            if (!is_u32) {
                perr(p, opline, "red.async.inc/dec supports .u32 only");
                return -1;
            }
        } else if (in.atom_op == 7 || in.atom_op == 8) {
            if (!is_u32 && !is_s32) {
                perr(p, opline, "red.async.min/max supports .u32/.s32 only");
                return -1;
            }
        } else if (in.atom_op == 3 || in.atom_op == 4 || in.atom_op == 5) {
            if (!is_u32) {
                perr(p, opline,
                     "red.async.and/or/xor supports .b32 only");
                return -1;
            }
        } else if (!is_u32 && !is_s32 && !is_u64) {
            perr(p, opline, "red.async.add supports .u32/.s32/.u64 only");
            return -1;
        }
    }

    if (parse_operand(p, &in.src[0]) != 0)
        return -1;
    if (in.src[0].kind != OPD_ADDR) {
        perr(p, opline, "red.async destination must be [address]");
        return -1;
    }
    if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
        return -1;
    in.nsrc = 2;
    if (!in.rda_release) {
        // Relaxed shared-cluster forms take an mbarrier; completion
        // defaults to mbarrier::complete_tx::bytes when not spelled out.
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[2]) != 0)
            return -1;
        if (in.src[2].kind != OPD_ADDR) {
            perr(p, opline, "red.async mbarrier must be [address]");
            return -1;
        }
        in.nsrc = 3;
    }
    if (expect_punct(p, ';', "after red.async") != 0)
        return -1;
    return block_add_instr(fn, bi, &in, p, opline);
}

// multimem.ld_reduce / multimem.st / multimem.red. With a single GPU a
// multimem address names that GPU's own memory, so loading from it is
// just a load, storing is just a store, and a reduction is the same
// atomic reduction (the inter-GPU fan-out is what disappears).
int parse_multimem(parser_t *p, ptx_func_t *fn, size_t bi,
                   const char *pred, int pred_neg, const char *opname,
                   int opline) {
    char parts[RDA_MAX_PARTS][32];
    int np = 0;
    memset(parts, 0, sizeof(parts));
    if (rda_split(opname, parts, &np) != 0 || np < 3) {
        perr(p, opline, "bad multimem opcode");
        return -1;
    }
    int is_ld = strcmp(parts[1], "ld_reduce") == 0;
    int is_st = strcmp(parts[1], "st") == 0;
    int is_red = strcmp(parts[1], "red") == 0;
    if (!is_ld && !is_st && !is_red) {
        perr(p, opline,
             "unsupported multimem form (ld_reduce/st/red only)");
        return -1;
    }
    if (p->sm && p->sm < 90) {
        perr(p, opline, "multimem requires sm_90 or higher");
        return -1;
    }
    int async_st = is_st && np > 2 && strcmp(parts[2], "async") == 0;
    if (is_st && !async_st && np > 2 && strcmp(parts[2], "bulk") == 0) {
        perr(p, opline, "multimem bulk copies are not implemented");
        return -1;
    }
    if (np < (is_st ? 3 : 4)) {
        perr(p, opline, "incomplete multimem form");
        return -1;
    }
    const char *op = is_st ? NULL : parts[np - 2];
    const char *tyname = parts[np - 1];
    int start = async_st ? 3 : 2;
    int end = is_st ? np - 1 : np - 2;
    int global = 0, vec = 0;
    for (int i = start; i < end; i++) {
        const char *t = parts[i];
        if (strcmp(t, "relaxed") == 0 || strcmp(t, "acquire") == 0 ||
            strcmp(t, "release") == 0 || strcmp(t, "weak") == 0 ||
            strcmp(t, "cta") == 0 || strcmp(t, "cluster") == 0 ||
            strcmp(t, "gpu") == 0 || strcmp(t, "sys") == 0 ||
            strcmp(t, "mmio") == 0 || strncmp(t, "acc::", 5) == 0)
            continue;
        if (strcmp(t, "global") == 0) {
            global = 1;
            continue;
        }
        if (strcmp(t, "v2") == 0 || strcmp(t, "v4") == 0 ||
            strcmp(t, "v8") == 0) {
            vec = 1;
            continue;
        }
        perr(p, opline, "unsupported multimem qualifier");
        return -1;
    }
    if (vec) {
        perr(p, opline, "multimem vector forms are not supported");
        return -1;
    }
    ptx_ty_t ty;
    if (map_ty(tyname, &ty) != 0 ||
        (ty != PTY_U32 && ty != PTY_U64 && ty != PTY_F32 &&
         ty != PTY_F64)) {
        perr(p, opline,
             "multimem supports scalar .b32/.b64/.u32/.u64/.s32/.s64/"
             ".f32/.f64 only");
        return -1;
    }
    if (!is_st) {
        int okop = strcmp(op, "add") == 0 || strcmp(op, "min") == 0 ||
                   strcmp(op, "max") == 0 || strcmp(op, "and") == 0 ||
                   strcmp(op, "or") == 0 || strcmp(op, "xor") == 0;
        if (!okop) {
            perr(p, opline, "unsupported multimem reduction op");
            return -1;
        }
        if ((strcmp(op, "and") == 0 || strcmp(op, "or") == 0 ||
             strcmp(op, "xor") == 0) &&
            ty != PTY_U32 && ty != PTY_U64) {
            perr(p, opline,
                 "bitwise multimem reductions need .b32/.b64 integers");
            return -1;
        }
    }
    if (is_red) {
        // The single-device fallback is exactly the matching atomic
        // reduction; reuse the atom parser for operation/type checking.
        char aop[128];
        snprintf(aop, sizeof(aop), "red.%s.%s", op, tyname);
        return parse_atom_instr(p, fn, bi, aop, opline);
    }

    ptx_instr_t in;
    memset(&in, 0, sizeof(in));
    in.line = opline;
    in.op = is_st ? POP_ST : POP_LD;
    in.ty = ty;
    in.space = global ? PSP_GLOBAL : PSP_GENERIC;
    if (copy_name(in.pred, sizeof(in.pred), pred[0] ? pred : "", p,
                  opline) != 0)
        return -1;
    in.pred_neg = pred_neg;
    if (is_st) {
        if (parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (in.src[0].kind != OPD_ADDR) {
            perr(p, opline, "multimem.st destination must be [address]");
            return -1;
        }
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        st_f32_bits_fixup(fn, &in);
        if (expect_punct(p, ';', "after multimem.st") != 0)
            return -1;
    } else {
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        ld_f32_bits_fixup(fn, &in);
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        if (in.src[0].kind != OPD_ADDR) {
            perr(p, opline, "multimem.ld_reduce source must be [address]");
            return -1;
        }
        in.nsrc = 1;
        if (expect_punct(p, ';', "after multimem.ld_reduce") != 0)
            return -1;
    }
    return block_add_instr(fn, bi, &in, p, opline);
}

// mapa / getctarank. Vulkan has no cluster execution scope, so the
// runtime models a one-CTA cluster (the same approximation used by
// barrier.cluster and relaxed red.async): the local rank is 0 and a
// shared address maps to itself.
int parse_cluster_instr(parser_t *p, ptx_func_t *fn, size_t bi,
                        const char *pred, int pred_neg,
                        const char *opname, int opline) {
    char parts[RDA_MAX_PARTS][32];
    int np = 0;
    memset(parts, 0, sizeof(parts));
    if (rda_split(opname, parts, &np) != 0 || np < 2 || np > 3) {
        perr(p, opline, "bad cluster opcode");
        return -1;
    }
    if (p->sm && p->sm < 90) {
        perr(p, opline, "cluster instructions require sm_90 or higher");
        return -1;
    }
    if (np == 3 && strcmp(parts[1], "shared::cluster") != 0) {
        perr(p, opline, "unsupported cluster state space");
        return -1;
    }
    const char *tyname = parts[np - 1];
    ptx_ty_t ty;
    if (map_ty(tyname, &ty) != 0 ||
        (ty != PTY_U32 && ty != PTY_U64)) {
        perr(p, opline, "cluster instruction type must be .u32 or .u64");
        return -1;
    }

    ptx_instr_t in;
    memset(&in, 0, sizeof(in));
    in.line = opline;
    in.op = POP_MOV;
    in.ty = ty;
    if (copy_name(in.pred, sizeof(in.pred), pred[0] ? pred : "", p,
                  opline) != 0)
        return -1;
    in.pred_neg = pred_neg;

    if (strcmp(parts[0], "getctarank") == 0) {
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
            return -1;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        // A one-CTA cluster has exactly one member, rank 0.
        in.src[0].kind = OPD_IMM;
        in.src[0].imm = 0;
        in.src[0].imm_is_float = 0;
        in.nsrc = 1;
        if (expect_punct(p, ';', "after getctarank") != 0)
            return -1;
        return block_add_instr(fn, bi, &in, p, opline);
    }

    // mapa: the destination is the local shared address of the source,
    // regardless of the target rank.
    if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
        return -1;
    if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
        return -1;
    if (in.src[0].kind != OPD_REG) {
        perr(p, opline,
             "mapa source must be a register (single-CTA cluster)");
        return -1;
    }
    if (expect_comma(p) != 0 || parse_operand(p, &in.src[1]) != 0)
        return -1;
    (void)in.src[1]; // target rank is always this CTA
    in.nsrc = 1;
    if (expect_punct(p, ';', "after mapa") != 0)
        return -1;
    return block_add_instr(fn, bi, &in, p, opline);
}
