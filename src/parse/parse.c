#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "ptx.h"
#include "parse/parse_internal.h"

// Upper bound for .reg array declarations (%r<N>).  Unrolled elementwise
// kernels routinely need more than the old 1024-register cap once the
// SASS-to-PTX lifter has materialised temporaries.
#define PTX_MAX_REG_RANGE 65536

void perr(parser_t *p, int line, const char *msg) {
    if (!p->failed) {
        snprintf(p->err, p->errlen, "line %d: %s", line, msg);
        p->failed = 1;
    }
}

// Strict bounded copy: overlong names are a hard error (silent truncation
// could merge distinct registers/labels into one).
int copy_name(char *dst, size_t cap, const char *src, parser_t *p,
                     int line) {
    if (strlen(src) >= cap) {
        perr(p, line, "name too long");
        return -1;
    }
    strcpy(dst, src);
    return 0;
}


int next(parser_t *p, ptx_tok_t *t) {
    if (ptx_next(&p->lx, t, p->err, p->errlen) != 0) {
        p->failed = 1;
        return -1;
    }
    return 0;
}

int expect_punct(parser_t *p, char c, const char *what) {
    ptx_tok_t t;
    if (next(p, &t) != 0)
        return -1;
    if (t.kind != T_PUNCT || t.punct != c) {
        char m[640];
        snprintf(m, sizeof(m), "expected '%c' %s", c, what);
        perr(p, t.line, m);
        return -1;
    }
    return 0;
}

int at_punct(parser_t *p, char c) {
    ptx_tok_t t;
    if (next(p, &t) != 0)
        return 0;
    if (t.kind == T_PUNCT && t.punct == c)
        return 1;
    ptx_unget(&p->lx, &t);
    return 0;
}

void ptx_unit_free(ptx_unit_t *unit) {
    if (!unit)
        return;
    for (size_t f = 0; f < unit->nfuncs; f++) {
        ptx_func_t *fn = &unit->funcs[f];
        free(fn->params);
        free(fn->regs);
        free(fn->locals);
        free(fn->slots);
        free(fn->rparams);
        free(fn->vecs);
        free(fn->host_skip);
        free(fn->param_kinds_override);
        for (size_t b = 0; b < fn->nblocks; b++) {
            for (size_t i = 0; i < fn->blocks[b].count; i++)
                free(fn->blocks[b].instrs[i].call_args);
            free(fn->blocks[b].instrs);
        }
        free(fn->blocks);
    }
    free(unit->funcs);
    free(unit->shvars);
    free(unit->module_locals);
    for (size_t i = 0; i < unit->nconsts; i++)
        free(unit->consts[i].bytes);
    free(unit->consts);
    for (size_t i = 0; i < unit->nglobals; i++)
        free(unit->globals[i].bytes);
    free(unit->globals);
    free(unit->texrefs);
    free(unit->extfuncs);
    memset(unit, 0, sizeof(*unit));
}

int map_ty(const char *s, ptx_ty_t *out) {
    // Accepts ".u32" (directive tokens) and "u32" (split opcodes).
    if (s[0] == '.')
        s++;
    if (strcmp(s, "u32") == 0) { *out = PTY_U32; return 0; }
    if (strcmp(s, "u64") == 0) { *out = PTY_U64; return 0; }
    if (strcmp(s, "f32") == 0) { *out = PTY_F32; return 0; }
    if (strcmp(s, "f64") == 0) { *out = PTY_F64; return 0; }
    if (strcmp(s, "pred") == 0) { *out = PTY_PRED; return 0; }
    // Narrow storage/compute types (Phase 4 addendum): 8/16-bit ints
    // hold zero-extended canonical values in u32 SSA; f16 holds the
    // exact f32 value (rounded at every def, no capability needed).
    if (strcmp(s, "u8") == 0) { *out = PTY_U8; return 0; }
    if (strcmp(s, "s8") == 0) { *out = PTY_U8; return 0; }
    if (strcmp(s, "b8") == 0) { *out = PTY_U8; return 0; }
    if (strcmp(s, "u16") == 0) { *out = PTY_U16; return 0; }
    if (strcmp(s, "s16") == 0) { *out = PTY_U16; return 0; }
    if (strcmp(s, "b16") == 0) { *out = PTY_U16; return 0; }
    if (strcmp(s, "f16") == 0) { *out = PTY_F16; return 0; }
    if (strcmp(s, "bf16") == 0) { *out = PTY_BF16; return 0; }
    if (strcmp(s, "f16x2") == 0) { *out = PTY_F16X2; return 0; }
    if (strcmp(s, "bf16x2") == 0) { *out = PTY_BF16X2; return 0; }
    // Bit/storage aliases: exact for storage and wrap-arithmetic; see
    // docs/supported-isa.md for where each is accepted.
    if (strcmp(s, "b32") == 0) { *out = PTY_U32; return 0; }
    if (strcmp(s, "b64") == 0) { *out = PTY_U64; return 0; }
    if (strcmp(s, "s32") == 0) { *out = PTY_U32; return 0; }
    if (strcmp(s, "s64") == 0) { *out = PTY_U64; return 0; }
    return -1;
}

// True when name is declared as a register in the current function
// (exact match: `%` is part of the name, never mixed).
static int is_reg_declared(parser_t *p, const char *name) {
    if (!p->fn)
        return 0;
    for (size_t i = 0; i < p->fn->nregs; i++) {
        if (strcmp(p->fn->regs[i].name, name) == 0)
            return 1;
    }
    return 0;
}

// Resolve a register name through active anonymous-brace scopes
// (innermost binding wins). Copies name unchanged when not shadowed.
static const char *scope_resolve(parser_t *p, const char *name) {
    for (size_t i = p->nsbind; i-- > 0;) {
        if (strcmp(p->sbind[i].name, name) == 0)
            return p->sbind[i].internal;
    }
    return name;
}

// Bind a shadowed register name at the current brace depth.
static int scope_push(parser_t *p, const char *name, const char *internal,
                      int line) {
    if (p->nsbind + 1 > p->sbind_cap) {
        size_t ncap = p->sbind_cap ? p->sbind_cap * 2 : 4;
        scope_bind_t *nb = realloc(p->sbind, sizeof(*nb) * ncap);
        if (!nb) {
            perr(p, line, "out of memory");
            return -1;
        }
        p->sbind = nb;
        p->sbind_cap = ncap;
    }
    scope_bind_t *b = &p->sbind[p->nsbind++];
    snprintf(b->name, sizeof(b->name), "%s", name);
    snprintf(b->internal, sizeof(b->internal), "%s", internal);
    b->depth = p->depth;
    return 0;
}

// Drop every binding introduced by the scope being closed.
void scope_pop(parser_t *p) {
    while (p->nsbind > 0 && p->sbind[p->nsbind - 1].depth >= p->depth)
        p->nsbind--;
}

// Vector registers (scalarized at parse: `v` holds v_x/_y/... as
// scalar regs of the element type; see docs/supported-isa.md).
int vec_elem_ty(const char *s, ptx_ty_t *out) {
    if (s[0] == '.')
        s++;
    // 64-bit elements: legal for vector ld/st widths (a .v2.b64 access
    // is two 64-bit scalar accesses) and for .reg.vN.u64 declarations.
    if (strcmp(s, "u64") == 0 || strcmp(s, "s64") == 0 ||
        strcmp(s, "b64") == 0) { *out = PTY_U64; return 0; }
    if (strcmp(s, "u32") == 0 || strcmp(s, "s32") == 0 ||
        strcmp(s, "b32") == 0) { *out = PTY_U32; return 0; }
    // Floating vectors (ld/st.global.v4.f32) scalarize to their element
    // type; keep the lane type so the split loads stay float.
    if (strcmp(s, "f32") == 0) { *out = PTY_F32; return 0; }
    if (strcmp(s, "f64") == 0) { *out = PTY_F64; return 0; }
    if (strcmp(s, "u16") == 0 || strcmp(s, "s16") == 0 ||
        strcmp(s, "b16") == 0) { *out = PTY_U16; return 0; }
    if (strcmp(s, "u8") == 0 || strcmp(s, "s8") == 0 ||
        strcmp(s, "b8") == 0) { *out = PTY_U8; return 0; }
    return -1;
}

// Parse a `.vN` / `.vN.T` / `vN` / `vN.T` type text: 0 = full form
// (n + elem set), 1 = spaced form (n set, elem token still to read),
// -1 = not vector. Leading dot optional (combined `.reg` decls).
int vec_type(const char *s, int *n, ptx_ty_t *elem) {
    if (s[0] == '.')
        s++;
    if (s[0] != 'v')
        return -1;
    if (s[1] != '2' && s[1] != '4')
        return -1;
    *n = s[1] - '0';
    if (s[2] == '\0')
        return 1;
    if (s[2] != '.' || vec_elem_ty(s + 3, elem) != 0)
        return -1;
    return 0;
}

ptx_vecdecl_t *vec_find(ptx_func_t *fn, const char *name) {
    for (size_t i = 0; i < fn->nvecs; i++) {
        if (strcmp(fn->vecs[i].name, name) == 0)
            return &fn->vecs[i];
    }
    return NULL;
}

// Component letter to index (x/y for .v2, x/y/z/w for .v4), -1 if bad.
static int vec_comp(const ptx_vecdecl_t *v, char c) {
    const char *letters = "xyzw";
    for (int i = 0; i < v->n; i++) {
        if (letters[i] == c)
            return i;
    }
    return -1;
}

// If `text` is `vec.comp` for a declared vector reg, write the
// scalarized component name to out and return 1. Returns 0 when not a
// vector component (caller falls through to existing handling), -1 on
// a bad component letter (error set).
int vec_component_reg(parser_t *p, const char *text, char out[64],
                             int line) {
    const char *dot = strrchr(text, '.');
    if (!dot || dot == text || !dot[1] || dot[2] || !p->fn)
        return 0;
    size_t bl = (size_t)(dot - text);
    if (bl + 2 >= 64)
        return 0;
    char base[64];
    memcpy(base, text, bl);
    base[bl] = '\0';
    ptx_vecdecl_t *v = vec_find(p->fn, base);
    if (!v)
        return 0;
    if (vec_comp(v, dot[1]) < 0) {
        perr(p, line, "bad vector component (x/y for .v2, x/y/z/w for .v4)");
        return -1;
    }
    memcpy(out, base, bl);
    out[bl] = '_';
    out[bl + 1] = dot[1];
    out[bl + 2] = '\0';
    return 1;
}

int parse_operand(parser_t *p, ptx_opd_t *op) {
    ptx_tok_t t;
    if (next(p, &t) != 0)
        return -1;
    memset(op, 0, sizeof(*op));
    if (t.kind == T_STR) {
        perr(p, t.line, "string literals unsupported");
        return -1;
    }
    if (t.kind == T_PUNCT && (t.punct == '-' || t.punct == '+')) {
        // Unary sign on a numeric literal (numba emits -1/-2 as selp
        // and address-math operands). Only '-' is meaningful; '+'
        // is tolerated for symmetry.
        int neg = t.punct == '-';
        ptx_tok_t u;
        if (next(p, &u) != 0)
            return -1;
        if (u.kind != T_INT && u.kind != T_FLOAT) {
            perr(p, u.line, "sign must precede a numeric literal");
            return -1;
        }
        if (u.kind == T_INT) {
            op->kind = OPD_IMM;
            op->imm = neg ? (uint64_t)(-(int64_t)u.imm) : u.imm;
            return 0;
        }
        op->kind = OPD_IMM;
        op->imm_is_float = 1;
        // Exact negation: flip the sign bit of the already-lexed bits.
        // Decimal text (used by the f64 strtod path) gets '-' prepended.
        int is_d = u.text[0] == '0' && (u.text[1] == 'D' || u.text[1] == 'd');
        int is_h = u.text[0] == '0' && (u.text[1] == 'H' || u.text[1] == 'h');
        if (neg) {
            if (is_d)
                op->imm = u.imm ^ 0x8000000000000000ull;
            else if (is_h)
                op->imm = u.imm ^ 0x8000ull;
            else
                op->imm = u.imm ^ 0x80000000ull;
        } else {
            op->imm = u.imm;
        }
        if (!is_d && !is_h && !(u.text[0] == '0' &&
                       (u.text[1] == 'F' || u.text[1] == 'f'))) {
            if ((neg ? 1 : 0) + strlen(u.text) >= sizeof(op->imm_text)) {
                perr(p, u.line, "float literal too long");
                return -1;
            }
            if (neg) {
                op->imm_text[0] = '-';
                strcpy(op->imm_text + 1, u.text);
            } else {
                strcpy(op->imm_text, u.text);
            }
        } else if (strlen(u.text) >= sizeof(op->imm_text)) {
            perr(p, u.line, "float literal too long");
            return -1;
        } else {
            strcpy(op->imm_text, u.text);
        }
        return 0;
    }
    if (t.kind == T_PUNCT && t.punct == '[') {
        op->kind = OPD_ADDR;
        op->addr_imm = 0;
        if (next(p, &t) != 0)
            return -1;
        if (t.kind != T_IDENT) {
            perr(p, t.line, "expected register or parameter inside []");
            return -1;
        }
        if (t.text[0] == '%') {
            if (copy_name(op->addr_reg, sizeof(op->addr_reg), t.text, p, t.line) != 0)
                return -1;
        } else if (is_reg_declared(p, t.text)) {
            // Bare register acting as an address base (ptxas accepts
            // bare regs; `%` is part of the name, matched exactly).
            if (copy_name(op->addr_reg, sizeof(op->addr_reg), t.text, p, t.line) != 0)
                return -1;
        } else {
            if (copy_name(op->param, sizeof(op->param), t.text, p, t.line) != 0)
                return -1;
        }
        {
            // Signed byte offset ([reg+N], [reg-N], [reg+-N]): folds
            // into the address like ptxas (wraps like any immediate).
            int neg = 0, has_off = 0;
            if (at_punct(p, '+')) {
                has_off = 1;
                if (at_punct(p, '-'))
                    neg = 1;
            } else if (at_punct(p, '-')) {
                has_off = 1;
                neg = 1;
            }
            if (has_off) {
                if (next(p, &t) != 0)
                    return -1;
                if (t.kind != T_INT) {
                    perr(p, t.line, "address offset must be an integer");
                    return -1;
                }
                op->addr_imm = neg ? -(int64_t)t.imm : (int64_t)t.imm;
            }
        }
        // [reg+reg] is out of subset.
        ptx_tok_t u;
        if (next(p, &u) != 0)
            return -1;
        if (u.kind == T_PUNCT && u.punct == '+') {
            perr(p, u.line, "register+register addresses unsupported");
            return -1;
        }
        if (u.kind != T_PUNCT || u.punct != ']') {
            perr(p, u.line, "expected ']'");
            return -1;
        }
        return 0;
    }
    if (t.kind == T_INT) {
        op->kind = OPD_IMM;
        op->imm = t.imm;
        return 0;
    }
    if (t.kind == T_FLOAT) {
        op->kind = OPD_IMM;
        op->imm = t.imm;
        op->imm_is_float = 1;
        if (strlen(t.text) >= sizeof(op->imm_text)) {
            perr(p, t.line, "float literal too long");
            return -1;
        }
        strcpy(op->imm_text, t.text);
        return 0;
    }
    if (t.kind != T_IDENT) {
        if (t.kind == T_PUNCT && t.punct == '_') {
            op->kind = OPD_REG;
            snprintf(op->reg, sizeof(op->reg), "__sink_%d", (int)t.line);

            return 0;
        }
        perr(p, t.line, "expected operand");
        return -1;
    }
    if (strcmp(t.text, "_") == 0) {
        op->kind = OPD_REG;
        snprintf(op->reg, sizeof(op->reg), "__sink_%d", t.line);

        return 0;
    }
    // PTX built-in warp-size constant (32 lanes per warp).
    if (strcmp(t.text, "WARP_SZ") == 0) {
        op->kind = OPD_IMM;
        op->imm = 32;
        return 0;
    }
    if (strcmp(t.text, "%laneid") == 0) {
        op->kind = OPD_SPECIAL;
        op->special = SPC_LANEID;
        op->comp = 0;
        return 0;
    }
    if (strcmp(t.text, "%lanemask_lt") == 0) {
        op->kind = OPD_SPECIAL;
        op->special = SPC_LANEMASKLT;
        op->comp = 0;
        return 0;
    }
    if (strcmp(t.text, "%lanemask_le") == 0) {
        op->kind = OPD_SPECIAL;
        op->special = SPC_LANEMASKLE;
        op->comp = 0;
        return 0;
    }
    if (strcmp(t.text, "%lanemask_ge") == 0) {
        op->kind = OPD_SPECIAL;
        op->special = SPC_LANEMASKGE;
        op->comp = 0;
        return 0;
    }
    if (strcmp(t.text, "%lanemask_gt") == 0) {
        op->kind = OPD_SPECIAL;
        op->special = SPC_LANEMASKGT;
        op->comp = 0;
        return 0;
    }
    if (strcmp(t.text, "%lanemask_eq") == 0) {
        op->kind = OPD_SPECIAL;
        op->special = SPC_LANEMASKEQ;
        op->comp = 0;
        return 0;
    }
    if (strcmp(t.text, "%clock") == 0) {
        op->kind = OPD_SPECIAL;
        op->special = SPC_CLOCK;
        op->comp = 0;
        return 0;
    }
    if (strcmp(t.text, "%clock64") == 0) {
        op->kind = OPD_SPECIAL;
        op->special = SPC_CLOCK64;
        op->comp = 0;
        return 0;
    }
    if (strcmp(t.text, "%tid.x") == 0 || strcmp(t.text, "%tid.y") == 0 ||
        strcmp(t.text, "%tid.z") == 0) {
        op->kind = OPD_SPECIAL;
        op->special = SPC_TID;
        op->comp = (int)(t.text[5] - 'x');
        return 0;
    }
    if (strcmp(t.text, "%ntid.x") == 0 || strcmp(t.text, "%ntid.y") == 0 ||
        strcmp(t.text, "%ntid.z") == 0) {
        op->kind = OPD_SPECIAL;
        op->special = SPC_NTID;
        op->comp = (int)(t.text[6] - 'x');
        return 0;
    }
    if (strcmp(t.text, "%ctaid.x") == 0 || strcmp(t.text, "%ctaid.y") == 0 ||
        strcmp(t.text, "%ctaid.z") == 0) {
        op->kind = OPD_SPECIAL;
        op->special = SPC_CTAID;
        op->comp = (int)(t.text[7] - 'x');
        return 0;
    }
    if (strcmp(t.text, "%nctaid.x") == 0 || strcmp(t.text, "%nctaid.y") == 0 ||
        strcmp(t.text, "%nctaid.z") == 0) {
        op->kind = OPD_SPECIAL;
        op->special = SPC_NCTAID;
        op->comp = (int)(t.text[8] - 'x');
        return 0;
    }
    if (t.text[0] == '%') {
        op->kind = OPD_REG;
        const char *rn = scope_resolve(p, t.text);
        // Vector component (`%v.x`): same scalarization as bare `v.x`
        // (exact dotted names win: a declared `%a.b` is never a
        // component even when `%a` is a vector).
        if (!is_reg_declared(p, rn)) {
            char vc[64];
            int vr = vec_component_reg(p, t.text, vc, t.line);
            if (vr < 0)
                return -1;
            if (vr > 0) {
                if (copy_name(op->reg, sizeof(op->reg), vc, p,
                              t.line) != 0)
                    return -1;
                if (at_punct(p, '+')) {
                    if (next(p, &t) != 0)
                        return -1;
                    if (t.kind != T_INT) {
                        perr(p, t.line, "register offset must be an integer");
                        return -1;
                    }
                    op->kind = OPD_REGOFF;
                    op->regoff_imm = t.imm;
                }
                return 0;
            }
        }
        if (copy_name(op->reg, sizeof(op->reg), rn, p, t.line) != 0)
            return -1;
        // Register-plus-offset value form (ZLUDA corpus: st operands
        // like `temp + 1` materialize the addition).
        if (at_punct(p, '+')) {
            if (next(p, &t) != 0)
                return -1;
            if (t.kind != T_INT) {
                perr(p, t.line, "register offset must be an integer");
                return -1;
            }
            op->kind = OPD_REGOFF;
            op->regoff_imm = t.imm;
        }
        return 0;
    }
    // Bare identifier: a register when declared as one (exact match,
    // `%` never mixed), otherwise a parameter name. Use-before-declare
    // falls through to OPD_PARAM and fails honestly downstream.
    const char *rn = scope_resolve(p, t.text);
    if (is_reg_declared(p, rn)) {        op->kind = OPD_REG;
        if (copy_name(op->reg, sizeof(op->reg), rn, p, t.line) != 0)
            return -1;
        if (at_punct(p, '+')) {
            if (next(p, &t) != 0)
                return -1;
            if (t.kind != T_INT) {
                perr(p, t.line, "register offset must be an integer");
                return -1;
            }
            op->kind = OPD_REGOFF;
            op->regoff_imm = t.imm;
        }
        return 0;
    }
    // Vector component: `v.x` reads the scalarized component register
    // (ptxas accepts components as any scalar operand). Exact names
    // win above; anything else here is either a component or an
    // honest downstream error (use-before-declare looks identical).
    {
        char vc[64];
        int vr = vec_component_reg(p, t.text, vc, t.line);
        if (vr < 0)
            return -1;
        if (vr > 0) {
            op->kind = OPD_REG;
            if (copy_name(op->reg, sizeof(op->reg), vc, p,
                          t.line) != 0)
                return -1;
            return 0;
        }
    }
    op->kind = OPD_PARAM;
    if (copy_name(op->param, sizeof(op->param), t.text, p, t.line) != 0)
        return -1;
    return 0;
}

int func_add_reg(ptx_func_t *fn, const char *name, ptx_ty_t ty) {
    if (fn->nregs + 1 > fn->regs_cap) {
        size_t ncap = fn->regs_cap ? fn->regs_cap * 2 : 8;
        ptx_regdecl_t *nr = malloc(sizeof(*nr) * ncap);
        if (!nr)
            return -1;
        memcpy(nr, fn->regs, sizeof(*nr) * fn->nregs);
        free(fn->regs);
        fn->regs = nr;
        fn->regs_cap = ncap;
    }
    snprintf(fn->regs[fn->nregs].name, sizeof(fn->regs[fn->nregs].name), "%.*s",
             (int)sizeof(fn->regs[fn->nregs].name) - 1, name);
    fn->regs[fn->nregs].ty = ty;
    fn->nregs++;
    return 0;
}

int func_add_vec(ptx_func_t *fn, const char *name, int n,
                        ptx_ty_t elem) {
    if (fn->nvecs + 1 > fn->vecs_cap) {
        size_t ncap = fn->vecs_cap ? fn->vecs_cap * 2 : 4;
        ptx_vecdecl_t *nv = malloc(sizeof(*nv) * ncap);
        if (!nv)
            return -1;
        memcpy(nv, fn->vecs, sizeof(*nv) * fn->nvecs);
        free(fn->vecs);
        fn->vecs = nv;
        fn->vecs_cap = ncap;
    }
    snprintf(fn->vecs[fn->nvecs].name, sizeof(fn->vecs[fn->nvecs].name),
             "%.*s", (int)sizeof(fn->vecs[fn->nvecs].name) - 1, name);
    fn->vecs[fn->nvecs].n = n;
    fn->vecs[fn->nvecs].elem = elem;
    fn->nvecs++;
    return 0;
}

// Add one scalar register declaration. A name re-declared inside an
// anonymous `{ ... }` block shadows the outer register: it gets a
// fresh internal name and a scope binding, so the inner references do
// not clobber the outer value (block.ptx). A top-level re-declaration
// is idempotent (the first declaration wins).
static int declare_scalar_reg(parser_t *p, ptx_func_t *fn, const char *name,
                              ptx_ty_t ty, int line) {
    if (is_reg_declared(p, name)) {
        if (p->depth <= 0)
            return 0;
        char internal[64];
        for (;;) {
            snprintf(internal, sizeof(internal), "__sc%u_%.48s",
                     ++p->scope_seq, name);
            if (!is_reg_declared(p, internal) && !vec_find(fn, internal))
                break;
        }
        if (func_add_reg(fn, internal, ty) != 0) {
            perr(p, line, "out of memory");
            return -1;
        }
        return scope_push(p, name, internal, line);
    }
    if (func_add_reg(fn, name, ty) != 0) {
        perr(p, line, "out of memory");
        return -1;
    }
    return 0;
}

int parse_reg_decl(parser_t *p, ptx_func_t *fn, const char *decltok,
                          int dtline) {
    // decltok is ".reg" (type token follows) or ".reg.<type>"
    // (combined). Next is the name list in both cases.
    ptx_tok_t t;
    char tyspec[64];
    if (strcmp(decltok, ".reg") == 0) {
        if (next(p, &t) != 0)
            return -1;
        if (t.kind != T_IDENT) {
            perr(p, t.line, "expected register type after .reg");
            return -1;
        }
        if (copy_name(tyspec, sizeof(tyspec), t.text, p, t.line) != 0)
            return -1;
    } else {
        if (copy_name(tyspec, sizeof(tyspec), decltok + 5, p, dtline) != 0)
            return -1;
    }
    const char *typetext = tyspec;
    // Vector registers (`.reg .v2 .u32 v` spaced or `.reg .v4.u16 v`
    // dotted): scalarized to v_x/_y/... of the element type at parse;
    // no vector SSA exists downstream. Ranges with vectors stay out.
    int vn = 0;
    ptx_ty_t velem = PTY_U32;
    {
        int is_vec = 0;
        // typetext is ".vN[.T]", "vN[.T]" (combined decl), or scalar.
        if ((typetext[0] == '.' && typetext[1] == 'v') ||
            (typetext[0] == 'v' &&
             (typetext[1] == '2' || typetext[1] == '4')) ||
            strcmp(typetext, ".b128") == 0 ||
            strcmp(typetext, "b128") == 0)
            is_vec = 1;
        if (is_vec) {
        int r;
        if (strcmp(typetext, ".b128") == 0 ||
            strcmp(typetext, "b128") == 0) {
            vn = 2;
            velem = PTY_U64;
            r = 0;
        } else {
            r = vec_type(typetext, &vn, &velem);
        }
        if (r < 0) {
            char m[640];
            snprintf(m, sizeof(m), "unsupported vector register type '%s' (only .v2/.v4 of u/s/b 8/16/32)", typetext);
            perr(p, dtline, m);
            return -1;
        }
        if (r == 1) {
            if (next(p, &t) != 0)
                return -1;
            if (t.kind != T_IDENT || vec_elem_ty(t.text, &velem) != 0) {
                perr(p, t.line, "expected vector element type (.u32/.u16/.u8 and aliases)");
                return -1;
            }
        }
        for (;;) {
            if (next(p, &t) != 0)
                return -1;
            if (t.kind != T_IDENT || t.text[0] == '$') {
                perr(p, t.line, "expected register name in .reg");
                return -1;
            }
            char base[256];
            if (copy_name(base, sizeof(base), t.text, p, t.line) != 0)
                return -1;
            if (strlen(base) > 240) {
                perr(p, t.line, "register name too long");
                return -1;
            }
            if (at_punct(p, '<')) {
                // Vector ranges (ldmatrix_trans corpus): `.v2.b16 x<4>`
                // declares x0..x3 as vectors with components each.
                ptx_tok_t ntok;
                if (next(p, &ntok) != 0)
                    return -1;
                if (ntok.kind != T_INT || ntok.imm == 0 ||
                    ntok.imm > PTX_MAX_REG_RANGE) {
                    perr(p, ntok.line, "bad register range");
                    return -1;
                }
                if (expect_punct(p, '>', "in register range") != 0)
                    return -1;
                for (int i = 0; i < (int)ntok.imm; i++) {
                    char vb[256], nm[256];
                    snprintf(vb, sizeof(vb), "%.*s%d", 240, base, i);
                    if (is_reg_declared(p, vb) || vec_find(fn, vb)) {
                        perr(p, t.line, "duplicate register declaration");
                        return -1;
                    }
                    if (func_add_vec(fn, vb, vn, velem) != 0) {
                        perr(p, t.line, "out of memory");
                        return -1;
                    }
                    for (int k = 0; k < vn; k++) {
                        size_t lc = strlen(vb);
                        memcpy(nm, vb, lc);
                        nm[lc] = '_';
                        nm[lc + 1] = "xyzw"[k];
                        nm[lc + 2] = '\0';
                        if (is_reg_declared(p, nm)) {
                            perr(p, t.line, "vector component collides with a declared register");
                            return -1;
                        }
                        if (func_add_reg(fn, nm, velem) != 0) {
                            perr(p, t.line, "out of memory");
                            return -1;
                        }
                    }
                }
                if (at_punct(p, ',')) {
                    continue;
                }
                break;
            }
            if (is_reg_declared(p, base) || vec_find(fn, base)) {
                perr(p, t.line, "duplicate register declaration");
                return -1;
            }
            if (func_add_vec(fn, base, vn, velem) != 0) {
                perr(p, t.line, "out of memory");
                return -1;
            }
            for (int i = 0; i < vn; i++) {
                char nm[256];
                size_t lb = strlen(base);
                memcpy(nm, base, lb);
                nm[lb] = '_';
                nm[lb + 1] = "xyzw"[i];
                nm[lb + 2] = '\0';
                if (is_reg_declared(p, nm)) {
                    perr(p, t.line, "vector component collides with a declared register");
                    return -1;
                }
                if (func_add_reg(fn, nm, velem) != 0) {
                    perr(p, t.line, "out of memory");
                    return -1;
                }
            }
            if (at_punct(p, ',')) {
                continue;
            }
            break;
        }
        return expect_punct(p, ';', "after .reg declaration");
    }
    }
    ptx_ty_t ty;
    if (map_ty(typetext, &ty) != 0) {
        char m[640];
        snprintf(m, sizeof(m), "unsupported register type '%s'", typetext);
        perr(p, dtline, m);
        return -1;
    }
    for (;;) {
        if (next(p, &t) != 0)
            return -1;
        // Registers are `%`-prefixed or bare identifiers (ptxas
        // accepts both, but never mixed for one register: the `%`
        // is part of the name and matching is exact).
        if (t.kind != T_IDENT || t.text[0] == '$') {
            perr(p, t.line, "expected register name in .reg");
            return -1;
        }
        // %r<4> range form arrives as separate tokens ('<' is not an
        // ident char): IDENT "%r", '<', INT, '>'. Single names arrive whole.
        char base[256];
        int lo = -1, hi = -1;
        if (copy_name(base, sizeof(base), t.text, p, t.line) != 0)
            return -1;
        if (strlen(base) > 240) {
            perr(p, t.line, "register name too long");
            return -1;
        }
        if (at_punct(p, '<')) {
            ptx_tok_t n;
            if (next(p, &n) != 0)
                return -1;
            if (n.kind != T_INT || n.imm == 0 ||
                n.imm > PTX_MAX_REG_RANGE) {
                perr(p, n.line, "bad register range");
                return -1;
            }
            if (expect_punct(p, '>', "in register range") != 0)
                return -1;
            lo = 0;
            hi = (int)n.imm;
        } else {
            lo = hi = -2; // single
        }
        if (lo == -2) {
            if (declare_scalar_reg(p, fn, base, ty, t.line) != 0)
                return -1;
        } else {
            for (int i = lo; i < hi; i++) {
                char nm[256];
                snprintf(nm, sizeof(nm), "%.*s%d", 240, base, i);
                if (declare_scalar_reg(p, fn, nm, ty, t.line) != 0)
                    return -1;
            }
        }
        if (at_punct(p, ',')) {
            // Disallow range form mixed with lists (done above per item).
            continue;
        }
        break;
    }
    return expect_punct(p, ';', "after .reg declaration");
}


int expect_comma(parser_t *p) { return expect_punct(p, ',', "between operands"); }

int parse_dst(parser_t *p, char *name, size_t namelen, int want_pred) {
    ptx_tok_t t;
    if (next(p, &t) != 0)
        return -1;
    // Destinations are registers (`%`-prefixed or bare, matching the
    // .reg declaration exactly); use-before-declare fails honestly at
    // emission ("assignment to undeclared register"). Vector
    // components (`v.x`) rewrite to the scalarized name. `_` discards
    // the result (ZLUDA sink corpus).
    if (t.kind == T_PUNCT && t.punct == '_') {
        snprintf(name, namelen, "__sink_%d", (int)t.line);
        if (p->fn && !is_reg_declared(p, name)) func_add_reg(p->fn, name, PTY_U32);
        return 0;
    }
    if (t.kind != T_IDENT || t.text[0] == '$') {
        perr(p, t.line, "expected destination register");
        return -1;
    }
    if (strcmp(t.text, "_") == 0) {
        snprintf(name, namelen, "__sink_%d", t.line);
        if (p->fn && !is_reg_declared(p, name)) func_add_reg(p->fn, name, PTY_U32);
        return 0;
    }
    (void)want_pred;
    {
        char vc[64];
        int vr = vec_component_reg(p, t.text, vc, t.line);
        if (vr < 0)
            return -1;
        if (vr > 0) {
            if (copy_name(name, namelen, vc, p, t.line) != 0)
                return -1;
            return 0;
        }
    }
    if (copy_name(name, namelen, scope_resolve(p, t.text), p, t.line) != 0)
        return -1;
    return 0;
}

int block_add_instr(ptx_func_t *fn, size_t bi, const ptx_instr_t *in,
                           parser_t *p, int line) {    ptx_block_t *b = &fn->blocks[bi];
    if (b->count + 1 > b->cap) {
        size_t ncap = b->cap ? b->cap * 2 : 8;
        ptx_instr_t *ni = malloc(sizeof(*ni) * ncap);
        if (!ni) {
            perr(p, line, "out of memory");
            return -1;
        }
        memcpy(ni, b->instrs, sizeof(*ni) * b->count);
        free(b->instrs);
        b->instrs = ni;
        b->cap = ncap;
    }
    b->instrs[b->count++] = *in;
    return 0;
}

size_t new_block(ptx_func_t *fn, parser_t *p, const char *label);

int split_op(const char *op, char parts[6][32], int *nparts) {
    *nparts = 0;
    const char *s = op;
    while (*s && *nparts < 6) {
        size_t n = 0;
        while (s[n] && s[n] != '.' && n + 1 < 32)
            n++;
        if (s[n] == '.' && n + 1 >= 32)
            return -1;
        memcpy(parts[*nparts], s, n);
        parts[*nparts][n] = '\0';
        (*nparts)++;
        s += n;
        if (*s == '.')
            s++;
        else if (*s)
            return -1;
    }
    return *s ? -1 : 0;
}

int map_cmp(const char *s, ptx_cmp_t *out) {    if (strcmp(s, "ge") == 0) { *out = PCM_GE; return 0; }
    if (strcmp(s, "lt") == 0) { *out = PCM_LT; return 0; }
    if (strcmp(s, "le") == 0) { *out = PCM_LE; return 0; }
    if (strcmp(s, "gt") == 0) { *out = PCM_GT; return 0; }
    if (strcmp(s, "eq") == 0) { *out = PCM_EQ; return 0; }
    if (strcmp(s, "ne") == 0) { *out = PCM_NE; return 0; }
    // Unordered float comparisons (true when either operand is NaN):
    // equ/neu/ltu/leu/gtu/geu. Ordered forms stay exact.
    if (strcmp(s, "equ") == 0) { *out = PCM_EQU; return 0; }
    if (strcmp(s, "neu") == 0) { *out = PCM_NEU; return 0; }
    if (strcmp(s, "ltu") == 0) { *out = PCM_LTU; return 0; }
    if (strcmp(s, "leu") == 0) { *out = PCM_LEU; return 0; }
    if (strcmp(s, "gtu") == 0) { *out = PCM_GTU; return 0; }
    if (strcmp(s, "geu") == 0) { *out = PCM_GEU; return 0; }
    return -1;
}

// Integer cvt spelling -> (width type, source-signed flag). The sign
// follows the source (signed sources sign-extend, unsigned zero-
// extend); same-width different-sign spellings are bit-copies.
int cvt_int_ty(const char *s, ptx_ty_t *ty, int *sign) {
    *sign = 0;
    if (strcmp(s, "u8") == 0 || strcmp(s, "b8") == 0) { *ty = PTY_U8; return 0; }
    if (strcmp(s, "s8") == 0) { *ty = PTY_U8; *sign = 1; return 0; }
    if (strcmp(s, "u16") == 0 || strcmp(s, "b16") == 0) { *ty = PTY_U16; return 0; }
    if (strcmp(s, "s16") == 0) { *ty = PTY_U16; *sign = 1; return 0; }
    if (strcmp(s, "u32") == 0 || strcmp(s, "b32") == 0) { *ty = PTY_U32; return 0; }
    if (strcmp(s, "s32") == 0) { *ty = PTY_U32; *sign = 1; return 0; }
    if (strcmp(s, "u64") == 0 || strcmp(s, "b64") == 0) { *ty = PTY_U64; return 0; }
    if (strcmp(s, "s64") == 0) { *ty = PTY_U64; *sign = 1; return 0; }
    return -1;
}

// setp comparison-type ladder (shared by the plain and
// boolean-operator forms): exact spellings; narrow integers compare
// their low bits (canonical values are zero-extended; signed forms
// narrow explicitly). PTX setp starts at 16 bits (no .u8/.s8 forms).
int setp_ty(parser_t *p, const char *s, ptx_ty_t *ty, int *sign,
                   int opline) {
    if (strcmp(s, "u32") == 0) {
        *ty = PTY_U32;
        *sign = 0;
    } else if (strcmp(s, "s32") == 0) {
        *ty = PTY_U32;
        *sign = 1;
    } else if (strcmp(s, "u16") == 0 || strcmp(s, "b16") == 0) {
        *ty = PTY_U16;
        *sign = 0;
    } else if (strcmp(s, "s16") == 0) {
        *ty = PTY_U16;
        *sign = 1;
    } else if (strcmp(s, "u64") == 0) {
        *ty = PTY_U64;
        *sign = 0;
    } else if (strcmp(s, "f32") == 0) {
        *ty = PTY_F32;
        *sign = 0;
    } else if (strcmp(s, "f64") == 0) {
        *ty = PTY_F64;
        *sign = 0;
    } else if (strcmp(s, "f16") == 0) {
        *ty = PTY_F16;
        *sign = 0;
    } else if (strcmp(s, "f16x2") == 0) {
        *ty = PTY_F16X2;
        *sign = 0;
    } else if (strcmp(s, "b32") == 0) {
        // Equality on raw 32-bit words (integer equality, like .u32).
        *ty = PTY_U32;
        *sign = 0;
    } else if (strcmp(s, "s64") == 0) {
        *ty = PTY_U64;
        *sign = 1;
    } else {
        perr(p, opline, "setp supports .u16/.s16/.u32/.b32/.s32/.u64/.f16/.f16x2/.f32/.f64/.s64 comparisons");
        return -1;
    }
    return 0;
}

// ld.f32 into a .b32-declared register moves bits (ptxas-clean in the
// ZLUDA tensor corpus): normalize to a .b32 load so the checker sees
// the declared type. Float consumers reinterpret the bits (see
// POP_CVTPACK); the wire behavior is identical either way.
void ld_f32_bits_fixup(ptx_func_t *fn, ptx_instr_t *in) {
    if (in->ty != PTY_F32 || !in->dst[0])
        return;
    for (size_t i = 0; i < fn->nregs; i++) {
        if (strcmp(fn->regs[i].name, in->dst) == 0 &&
            fn->regs[i].ty == PTY_U32) {
            in->ty = PTY_U32;
            return;
        }
    }
}

// st.f32 of a .b32-declared register moves bits (ptxas-clean: the s8
// mma corpus stores s32 accumulators from .b32 regs): normalize to a
// .b32 store.
void st_f32_bits_fixup(ptx_func_t *fn, ptx_instr_t *in) {
    if (in->ty != PTY_F32 || in->nsrc < 2 ||
        in->src[1].kind != OPD_REG)
        return;
    for (size_t i = 0; i < fn->nregs; i++) {
        if (strcmp(fn->regs[i].name, in->src[1].reg) == 0 &&
            fn->regs[i].ty == PTY_U32) {
            in->ty = PTY_U32;
            return;
        }
    }
}

// Parse `{r0, r1, ...}` register fragment lists (mma.sync): exactly
// n plain registers or `_` sinks, '}'-terminated.
int parse_brace_regs(parser_t *p, char names[][64], int n,
                            int opline) {
    if (expect_punct(p, '{', "to open mma fragment list") != 0)
        return -1;
    for (int i = 0; i < n; i++) {
        ptx_tok_t t;
        if (next(p, &t) != 0)
            return -1;
        if (t.kind == T_PUNCT && t.punct == '_') {
            snprintf(names[i], 64, "__sink_%d_%d", opline, i);
        } else {
            ptx_unget(&p->lx, &t);
            ptx_opd_t o;
            memset(&o, 0, sizeof(o));
            if (parse_operand(p, &o) != 0)
                return -1;
            if (o.kind != OPD_REG) {
                if (o.kind == OPD_PARAM && strcmp(o.param, "_") == 0) {
                    snprintf(names[i], 64, "__sink_%d_%d", opline, i);
                } else {
                    perr(p, opline, "mma fragments need plain registers");
                    return -1;
                }
            } else {
                if (copy_name(names[i], 64, o.reg, p, opline) != 0)
                    return -1;
            }
        }
        if (i + 1 < n && expect_comma(p) != 0)
            return -1;
    }
    if (expect_punct(p, '}', "to close mma fragment list") != 0)
        return -1;
    return 0;
}

// Parse a vector-list operand for scalarized vector ops: either a
// declared .vN register (exactly n components of the mnemonic's elem
// type) or a brace list of exactly n plain registers or `_` sinks.
// Names land in out[][64] (component names for regs, listed names for
// braces; sinks get fresh dummy names).
int parse_vec_operand(parser_t *p, char out[][64], int n,
                             ptx_ty_t elem, int opline) {
    ptx_tok_t t;
    if (next(p, &t) != 0)
        return -1;
    if (t.kind == T_PUNCT && t.punct == '{') {
        for (int i = 0; i < n; i++) {
            ptx_tok_t tt;
            if (next(p, &tt) != 0)
                return -1;
            if (tt.kind == T_PUNCT && tt.punct == '_') {
                snprintf(out[i], 64, "__sink_%d_%d", opline, i);
            } else {
                ptx_unget(&p->lx, &tt);
                ptx_opd_t o;
                memset(&o, 0, sizeof(o));
                if (parse_operand(p, &o) != 0)
                    return -1;
                if (o.kind == OPD_REG) {
                    if (copy_name(out[i], 64, o.reg, p, opline) != 0)
                        return -1;
                } else if (o.kind == OPD_PARAM && strcmp(o.param, "_") == 0) {
                    snprintf(out[i], 64, "__sink_%d_%d", opline, i);
                } else {
                    perr(p, opline, "vector list needs plain registers");
                    return -1;
                }
            }
            if (i + 1 < n && expect_comma(p) != 0)
                return -1;
        }
        if (expect_punct(p, '}', "in vector list") != 0)
            return -1;
        return 0;
    }
    if (t.kind != T_IDENT) {
        perr(p, t.line, "expected a vector register or '{'");
        return -1;
    }
    ptx_vecdecl_t *v = p->fn ? vec_find(p->fn, t.text) : NULL;
    if (!v || v->n != n) {
        perr(p, t.line, "vector operand needs a matching .vN register or {N regs}");
        return -1;
    }
    if (v->elem != elem) {
        perr(p, t.line, "vector element type mismatch");
        return -1;
    }
    if (strlen(t.text) + 2 >= 64) {
        perr(p, t.line, "name too long");
        return -1;
    }
    // Vector-reg IDENTs only exist for .v2/.v4 (decl gate), so xyzw
    // covers every reachable index here.
    {
        size_t lb = strlen(t.text);
        for (int i = 0; i < n; i++) {
            memcpy(out[i], t.text, lb);
            out[i][lb] = '_';
            out[i][lb + 1] = "xyzw"[i];
            out[i][lb + 2] = '\0';
        }
    }
    return 0;
}

int vec_elem_bytes(ptx_ty_t e) {
    if (e == PTY_U8)
        return 1;
    if (e == PTY_U16)
        return 2;
    if (e == PTY_U64 || e == PTY_F64)
        return 8;
    return 4;
}

// Collapse blocks whose only instruction is an unconditional `bra` into
// their target. Lifted jump-table chains often contain such trampolines;
// removing them lets the structured SPIR-V builder see a single merge
// instead of illegal exits through several nested selections. Blocks
// that are back-edge targets (loop headers/latches) are left alone.
static void simplify_branch_blocks(ptx_func_t *fn) {
again:
    for (size_t b = 1; b < fn->nblocks; b++) {
        ptx_block_t *blk = &fn->blocks[b];
        if (blk->count != 1 || !blk->label[0])
            continue;
        ptx_instr_t *br = &blk->instrs[0];
        if (br->op != POP_BRA || br->pred[0] || !br->target[0])
            continue;
        int back = 0;
        for (size_t q = 0; q < fn->nblocks && !back; q++) {
            if (q >= b)
                for (size_t ii = 0; ii < fn->blocks[q].count; ii++) {
                    ptx_instr_t *in = &fn->blocks[q].instrs[ii];
                    if ((in->op == POP_BRA || in->op == POP_CBRA) &&
                        strcmp(in->target, blk->label) == 0) {
                        back = 1;
                        break;
                    }
                }
        }
        if (back)
            continue;
        // Removing this block would slide the following block into its
        // place.  That is only safe when nothing falls through into the
        // trampoline: an empty predecessor (or one ending in a
        // conditional branch) continues implicitly at b+1, and after
        // the memmove that is the block *after* the trampoline, not its
        // branch target.  Lifted kernels rely on this edge (a merge
        // label after a loop), so keep the trampoline in that case.
        if (b > 0) {
            ptx_block_t *prev = &fn->blocks[b - 1];
            int prev_falls = 1;
            if (prev->count > 0) {
                ptx_op_t lop = prev->instrs[prev->count - 1].op;
                prev_falls = lop != POP_BRA && lop != POP_RET &&
                             lop != POP_EXIT;
            }
            if (prev_falls)
                continue;
        }
        const char *dest = br->target;
        for (size_t q = 0; q < fn->nblocks; q++)
            for (size_t ii = 0; ii < fn->blocks[q].count; ii++) {
                ptx_instr_t *in = &fn->blocks[q].instrs[ii];
                if ((in->op == POP_BRA || in->op == POP_CBRA) &&
                    strcmp(in->target, blk->label) == 0)
                    snprintf(in->target, sizeof(in->target), "%s", dest);
            }
        memmove(&fn->blocks[b], &fn->blocks[b + 1],
                (fn->nblocks - b - 1) * sizeof(fn->blocks[0]));
        fn->nblocks--;
        goto again;
    }
}

// Released PTX ISA versions (the .version directive may only name one of
// these).  Newer releases also raise the compile-time ceiling; see
// compile.c.
typedef struct {
    int maj, min;
} ptx_ver_t;

static const ptx_ver_t ptx_versions[] = {
    {1,0},{1,1},{1,2},{1,3},{1,4},{1,5},
    {2,0},{2,1},{2,2},{2,3},
    {3,0},{3,1},{3,2},
    {4,0},{4,1},{4,2},{4,3},
    {5,0},
    {6,0},{6,1},{6,2},{6,3},{6,4},{6,5},
    {7,0},{7,1},{7,2},{7,3},{7,4},{7,5},{7,6},{7,7},{7,8},
    {8,0},{8,1},{8,2},{8,3},{8,4},{8,5},{8,6},{8,7},{8,8},
    {9,0},{9,1},{9,2},{9,3},{9,4},
};

static int ptx_version_known(int maj, int min) {
    size_t n = sizeof(ptx_versions) / sizeof(ptx_versions[0]);
    for (size_t i = 0; i < n; i++)
        if (ptx_versions[i].maj == maj && ptx_versions[i].min == min)
            return 1;
    return 0;
}

// First PTX release that accepts each .target SM name.
typedef struct {
    const char *name;
    int maj, min;
} ptx_sm_ver_t;

static const ptx_sm_ver_t ptx_sm_versions[] = {
    {"sm_10", 1, 0}, {"sm_11", 1, 0}, {"sm_12", 1, 2}, {"sm_13", 1, 2},
    {"sm_20", 2, 0}, {"sm_30", 3, 0}, {"sm_35", 3, 1},
    {"sm_32", 4, 0}, {"sm_37", 4, 1}, {"sm_50", 4, 0},
    {"sm_52", 4, 1}, {"sm_53", 4, 2},
    {"sm_60", 5, 0}, {"sm_61", 5, 0}, {"sm_62", 5, 0},
    {"sm_70", 6, 0}, {"sm_72", 6, 1}, {"sm_75", 6, 3},
    {"sm_80", 7, 0}, {"sm_86", 7, 1}, {"sm_87", 7, 4},
    {"sm_89", 7, 8}, {"sm_90", 7, 8}, {"sm_90a", 8, 0},
    {"sm_100", 8, 6}, {"sm_100a", 8, 6}, {"sm_101", 8, 6},
    {"sm_101a", 8, 6},
    {"sm_120", 8, 7}, {"sm_120a", 8, 7},
    {"sm_100f", 8, 8}, {"sm_101f", 8, 8},
    {"sm_103", 8, 8}, {"sm_103f", 8, 8}, {"sm_103a", 8, 8},
    {"sm_120f", 8, 8}, {"sm_121", 8, 8}, {"sm_121f", 8, 8},
    {"sm_121a", 8, 8},
    {"sm_88", 9, 0}, {"sm_110", 9, 0}, {"sm_110f", 9, 0},
    {"sm_110a", 9, 0},
    {"sm_107", 9, 4}, {"sm_107f", 9, 4}, {"sm_107a", 9, 4},
};

static const ptx_sm_ver_t *ptx_sm_lookup(const char *name) {
    size_t n = sizeof(ptx_sm_versions) / sizeof(ptx_sm_versions[0]);
    for (size_t i = 0; i < n; i++)
        if (strcmp(ptx_sm_versions[i].name, name) == 0)
            return &ptx_sm_versions[i];
    return NULL;
}

// Parse "sm_<digits>[a|f]" into the numeric SM (suffix ignored for
// feature gates) and return 0 on success.
static int ptx_sm_parse(const char *text, int *sm) {
    if (strncmp(text, "sm_", 3) != 0)
        return -1;
    const char *d = text + 3;
    if (*d < '0' || *d > '9')
        return -1;
    long v = 0;
    while (*d >= '0' && *d <= '9') {
        v = v * 10 + (*d - '0');
        d++;
    }
    if (v < 10 || v > 200)
        return -1;
    if (*d) {
        if ((*d != 'a' && *d != 'f') || d[1])
            return -1;
    }
    *sm = (int)v;
    return 0;
}

int ptx_parse(const char *src, ptx_unit_t *unit, char *err, size_t errlen) {
    memset(unit, 0, sizeof(*unit));
    parser_t p;
    memset(&p, 0, sizeof(p));
    ptx_lex_init(&p.lx, src);
    p.err = err;
    p.errlen = errlen;
    err[0] = '\0';
    ptx_tok_t t;
    for (;;) {
        if (next(&p, &t) != 0)
            goto fail;
        if (t.kind == T_EOF)
            break;
        if (t.kind != T_IDENT) {
            perr(&p, t.line, "expected directive or entry");
            goto fail;
        }
        if (strcmp(t.text, ".version") == 0) {
            if (next(&p, &t) != 0)
                goto fail;
            // Lexer merges "8.8" into one T_FLOAT? "8.8" lexes as float.
            // Accept float bits or int pairs; simplest: parse from text.
            if (t.kind == T_FLOAT) {
                // Can't recover text version; require rescan via text field.
                float f;
                uint32_t bits = (uint32_t)t.imm;
                memcpy(&f, &bits, 4);
                unit->version_major = (int)f;
                unit->version_minor = (int)((f - (int)f) * 10 + 0.5f);
            } else if (t.kind == T_INT) {
                unit->version_major = (int)t.imm;
                if (!at_punct(&p, '.')) {
                    perr(&p, t.line, "expected .version X.Y");
                    goto fail;
                }
                if (next(&p, &t) != 0)
                    goto fail;
                if (t.kind != T_INT) {
                    perr(&p, t.line, "expected .version X.Y");
                    goto fail;
                }
                unit->version_minor = (int)t.imm;
            } else {
                perr(&p, t.line, "expected .version X.Y");
                goto fail;
            }
            if (!ptx_version_known(unit->version_major,
                                   unit->version_minor)) {
                perr(&p, t.line,
                     "invalid .version (expected a released PTX ISA "
                     "version up to 9.4)");
                goto fail;
            }
            continue;
        }
        if (strcmp(t.text, ".target") == 0) {
            if (next(&p, &t) != 0)
                goto fail;
            if (t.kind != T_IDENT ||
                ptx_sm_parse(t.text, &unit->sm) != 0) {
                perr(&p, t.line, "expected .target sm_<NN>[a|f]");
                goto fail;
            }
            size_t tl = strlen(t.text);
            if (tl >= sizeof(unit->sm_name)) {
                perr(&p, t.line, "target name too long");
                goto fail;
            }
            memcpy(unit->sm_name, t.text, tl + 1);
            p.sm = unit->sm;
            const ptx_sm_ver_t *sv = ptx_sm_lookup(unit->sm_name);
            if (!sv) {
                perr(&p, t.line, "unknown .target SM name");
                goto fail;
            }
            if (unit->version_major != 0 &&
                unit->version_major * 100 + unit->version_minor <
                    sv->maj * 100 + sv->min) {
                char msg[128];
                snprintf(msg, sizeof(msg),
                         ".target %s requires .version %d.%d or newer",
                         unit->sm_name, sv->maj, sv->min);
                perr(&p, t.line, msg);
                goto fail;
            }
            // Consume any trailing ", option" qualifiers.
            while (at_punct(&p, ',')) {
                if (next(&p, &t) != 0)
                    goto fail;
                if (t.kind != T_IDENT || t.text[0] == '.') {
                    perr(&p, t.line, "expected target option");
                    goto fail;
                }
            }
            continue;
        }
        if (strcmp(t.text, ".address_size") == 0) {
            if (next(&p, &t) != 0)
                goto fail;
            if (t.kind != T_INT || (t.imm != 64 && t.imm != 32)) {
                perr(&p, t.line, "only .address_size 32/64 supported");
                goto fail;
            }
            unit->addr_size = (int)t.imm;
            continue;
        }
        if (strcmp(t.text, ".loc") == 0 || strcmp(t.text, ".file") == 0) {
            // Debug line table at module scope (`.file index "name"` is
            // emitted after the entries by nvcc 13; `.loc` can precede
            // the first directive of a body). Newline-terminated and
            // semantically inert, so drop the line.
            int dline = t.line;
            for (;;) {
                ptx_tok_t u;
                if (next(&p, &u) != 0)
                    goto fail;
                if (u.kind == T_EOF)
                    break;
                if (u.kind == T_PUNCT && u.punct == ';')
                    break;
                if (u.line != dline) {
                    ptx_unget(&p.lx, &u);
                    break;
                }
            }
            continue;
        }
        // `.weak` is the old-toolchain spelling of weak linkage (CUDA 4.x
        // fatbins are full of `.weak .func` stubs).  Treat it like
        // `.visible`: only the entry/func body matters for validation.
        if (strcmp(t.text, ".visible") == 0 || strcmp(t.text, ".weak") == 0) {
            if (next(&p, &t) != 0)
                goto fail;
            if (t.kind == T_IDENT && strcmp(t.text, ".shared") == 0) {
                // Static shared variable (numba `shared.array`).
                if (parse_shared_decl(&p, unit) != 0)
                    goto fail;
                continue;
            }
            if (t.kind == T_IDENT && strcmp(t.text, ".texref") == 0) {
                if (parse_texref_decl(&p, unit) != 0)
                    goto fail;
                continue;
            }
            if (t.kind == T_IDENT &&
                (strcmp(t.text, ".global") == 0 ||
                 strcmp(t.text, ".const") == 0)) {
                // Module-scope variable declaration (numba scaffolding):
                // skip to ';'. Referenced symbols fail later, honestly.
                // (Initialized `.const` with literal lanes is parsed for
                // real below under bare `.const`; the .visible form is
                // still scaffolding-skipped.)
                for (;;) {
                    if (next(&p, &t) != 0)
                        goto fail;
                    if (t.kind == T_PUNCT && t.punct == ';')
                        break;
                    if (t.kind == T_EOF) {
                        perr(&p, t.line, "unterminated .visible decl");
                        goto fail;
                    }
                }
                continue;
            }
            if (t.kind != T_IDENT ||
                (strcmp(t.text, ".entry") != 0 &&
                 strcmp(t.text, ".func") != 0)) {
                perr(&p, t.line, "expected .entry/.func after .visible");
                goto fail;
            }
            if (parse_entry(&p, unit, strcmp(t.text, ".entry") == 0) != 0)
                goto fail;
            continue;
        }
        if (strcmp(t.text, ".entry") == 0) {
            if (parse_entry(&p, unit, 1) != 0)
                goto fail;
            continue;
        }
        if (strcmp(t.text, ".func") == 0) {
            // Callable spelled as .func (numba, NVRTC-less toolchains):
            // compiled as an entry. The .param .b32 func_retval slot
            // becomes a skipped write (see st.param).
            if (parse_entry(&p, unit, 0) != 0)
                goto fail;
            continue;
        }
        if (strcmp(t.text, ".shared") == 0) {
            // Bare static shared variable (no .visible prefix).
            if (parse_shared_decl(&p, unit) != 0)
                goto fail;
            continue;
        }
        if (strcmp(t.text, ".const") == 0) {
            // Initialized constant array (loads materialize as
            // immediates; see ld.const).
            if (parse_const_decl(&p, unit) != 0)
                goto fail;
            continue;
        }
        if (strcmp(t.text, ".local") == 0) {
            // Module-scope .local (per-invocation Private storage).
            if (parse_module_local_decl(&p, unit) != 0)
                goto fail;
            continue;
        }
        if (strcmp(t.text, ".global") == 0) {
            // Initialized module-scope global array (loads served
            // from a read-only image; stores stay clean errors).
            // CUDA 4.x also spells texture references
            // `.global .texref name;`.
            ptx_tok_t u;
            if (next(&p, &u) != 0)
                goto fail;
            if (u.kind == T_IDENT && strcmp(u.text, ".texref") == 0) {
                if (parse_texref_decl(&p, unit) != 0)
                    goto fail;
                continue;
            }
            ptx_unget(&p.lx, &u);
            if (parse_global_decl(&p, unit) != 0)
                goto fail;
            continue;
        }
        if (strcmp(t.text, ".texref") == 0) {
            if (parse_texref_decl(&p, unit) != 0)
                goto fail;
            continue;
        }
        if (strcmp(t.text, ".extern") == 0) {
            // Dynamic `.shared` (sized at launch; aliases one dynamic
            // base with any static variables alongside it).
            ptx_tok_t u;
            if (next(&p, &u) == 0 && u.kind == T_IDENT &&
                strcmp(u.text, ".shared") == 0) {
                if (parse_extern_shared(&p, unit) != 0)
                    goto fail;
                continue;
            }
            if (u.kind == T_IDENT && strcmp(u.text, ".func") == 0) {
                if (parse_extern_func(&p, unit) != 0)
                    goto fail;
                continue;
            }
            // Be permissive: skip until semicolon for all other .extern
            // forms (some have complex syntax).
            if (u.kind == T_PUNCT && u.punct == ';') {
                continue;
            }
            // u might be .func or other - just keep scanning
            for (;;) {
                if (u.kind == T_PUNCT && u.punct == ';')
                    break;
                if (u.kind == T_EOF) {
                    perr(&p, t.line, ".extern unsupported");
                    goto fail;
                }
                if (next(&p, &u) != 0)
                    goto fail;
            }
            continue;
        }
        if (strcmp(t.text, ".common") == 0) {
            // Module-scope common global (numba's NumbaEnv): zero
            // initialized, writable Private storage.
            if (parse_common_decl(&p, unit) != 0)
                goto fail;
            continue;
        }
        if (strcmp(t.text, ".section") == 0) {
            // Debug/info sections are not needed for compilation; skip them.
            ptx_tok_t u;
            int depth = 0;
            int saw_brace = 0;
            for (;;) {
                if (next(&p, &u) != 0)
                    goto fail;
                if (u.kind == T_EOF) {
                    perr(&p, t.line, "unterminated .section");
                    goto fail;
                }
                if (u.kind == T_PUNCT && u.punct == '{') {
                    saw_brace = 1;
                    depth++;
                    continue;
                }
                if (u.kind == T_PUNCT && u.punct == '}') {
                    if (depth > 0)
                        depth--;
                    if (saw_brace && depth == 0)
                        break;
                    continue;
                }
                if (!saw_brace && depth == 0 && u.kind == T_PUNCT && u.punct == ';')
                    break;
            }
            continue;
        }
        {
            char m[640];
            snprintf(m, sizeof(m), "unsupported top-level directive '%s'", t.text);
            perr(&p, t.line, m);
            goto fail;
        }
    }
    if (unit->nfuncs == 0) {
        snprintf(err, errlen, "no .entry found");
        goto fail;
    }
    // Default entry selection (NULL name) takes funcs[0]: order real
    // entries before .func definitions (stable), so a callee listed
    // first never shadows the launchable entry. Explicit driver
    // selection by name is unaffected.
    {
        size_t w = 0;
        for (size_t i = 0; i < unit->nfuncs; i++) {
            if (unit->funcs[i].is_entry) {
                if (w != i) {
                    ptx_func_t tmp = unit->funcs[w];
                    unit->funcs[w] = unit->funcs[i];
                    unit->funcs[i] = tmp;
                }
                w++;
            }
        }
    }
    // .target may precede .version; validate the pair once both are
    // known (the common order is checked at the directive itself).
    if (unit->sm_name[0] && unit->version_major != 0) {
        const ptx_sm_ver_t *sv = ptx_sm_lookup(unit->sm_name);
        if (!sv) {
            snprintf(err, errlen, "unknown .target %s", unit->sm_name);
            goto fail;
        }
        if (unit->version_major * 100 + unit->version_minor <
            sv->maj * 100 + sv->min) {
            snprintf(err, errlen,
                     ".target %s requires .version %d.%d or newer",
                     unit->sm_name, sv->maj, sv->min);
            goto fail;
        }
    }
    // Post-parse: collapse conditional-branch trampolines, then desugar
    // call slots + inline direct calls.
    for (size_t f = 0; f < unit->nfuncs; f++)
        simplify_branch_blocks(&unit->funcs[f]);
    if (inl_run(unit, err, errlen) != 0)
        goto fail;
    // Module-scope .local variables are per-invocation Private storage;
    // append them to every surviving function so ld/st.local resolves
    // them. A function-local with the same name shadows the module one.
    for (size_t f = 0; f < unit->nfuncs; f++) {
        ptx_func_t *fn = &unit->funcs[f];
        for (size_t m = 0; m < unit->nmodule_locals; m++) {
            ptx_shvar_t *ml = &unit->module_locals[m];
            int found = 0;
            for (size_t i = 0; i < fn->nlocals; i++)
                if (strcmp(fn->locals[i].name, ml->name) == 0) {
                    found = 1;
                    break;
                }
            if (found)
                continue;
            if (fn->nlocals + 1 > fn->locals_cap) {
                size_t ncap = fn->locals_cap ? fn->locals_cap * 2 : 4;
                ptx_shvar_t *nl =
                    realloc(fn->locals, sizeof(*nl) * ncap);
                if (!nl) {
                    snprintf(err, errlen, "out of memory");
                    goto fail;
                }
                fn->locals = nl;
                fn->locals_cap = ncap;
            }
            fn->locals[fn->nlocals++] = *ml;
        }
    }
    free(p.sbind);
    return 0;
fail:
    free(p.sbind);
    if (err && errlen > 0 && err[0] == '\0')
        snprintf(err, errlen, "PTX parse error (no detail)");
    ptx_unit_free(unit);
    return -1;
}
