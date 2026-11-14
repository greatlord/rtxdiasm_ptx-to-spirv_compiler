// Surface instruction front end: suld, sust, sured and suq.
//
// A .surfref maps to a Vulkan storage image.  Byte-addressed .b forms
// use 32-bit words; .p forms are formatted sample stores.
#include <stdio.h>
#include <string.h>
#include "ptx.h"
#include "parse/parse_internal.h"

#define SURF_MAX_PARTS 10

static int surf_split(const char *op, char parts[SURF_MAX_PARTS][32],
                      int *np) {
    *np = 0;
    const char *s = op;
    while (*s) {
        if (*np >= SURF_MAX_PARTS)
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

static int surf_geom_code(const char *s) {
    if (strcmp(s, "1d") == 0) return 1;
    if (strcmp(s, "2d") == 0) return 2;
    if (strcmp(s, "3d") == 0) return 3;
    if (strcmp(s, "a1d") == 0) return 4;
    if (strcmp(s, "a2d") == 0) return 5;
    return 0;
}

static int surf_list(parser_t *p, char out[4][64], int *nout, int cap,
                     ptx_ty_t elem, int line) {
    ptx_tok_t t;
    if (next(p, &t) != 0)
        return -1;
    if (t.kind == T_PUNCT && t.punct == '{') {
        int n = 0;
        for (;;) {
            ptx_tok_t tt;
            if (next(p, &tt) != 0)
                return -1;
            if (tt.kind == T_PUNCT && tt.punct == '}')
                break;
            if (n >= cap) {
                perr(p, line, "surface operand list too long");
                return -1;
            }
            if (tt.kind == T_PUNCT && tt.punct == '_') {
                snprintf(out[n], 64, "__sink_%d_%d", line, n);
            } else {
                ptx_unget(&p->lx, &tt);
                ptx_opd_t o;
                memset(&o, 0, sizeof(o));
                if (parse_operand(p, &o) != 0)
                    return -1;
                if (o.kind == OPD_REG) {
                    if (copy_name(out[n], 64, o.reg, p, line) != 0)
                        return -1;
                } else if (o.kind == OPD_PARAM &&
                           strcmp(o.param, "_") == 0) {
                    snprintf(out[n], 64, "__sink_%d_%d", line, n);
                } else {
                    perr(p, line, "surface list needs registers");
                    return -1;
                }
            }
            n++;
            if (at_punct(p, ','))
                continue;
            if (at_punct(p, '}'))
                break;
            perr(p, line, "expected ',' or '}' in surface list");
            return -1;
        }
        if (n == 0) {
            perr(p, line, "empty surface list");
            return -1;
        }
        *nout = n;
        return 0;
    }
    if (t.kind != T_IDENT) {
        perr(p, line, "expected '{' or a register");
        return -1;
    }
    ptx_vecdecl_t *v = p->fn ? vec_find(p->fn, t.text) : NULL;
    if (v && v->elem == elem && v->n <= cap && v->n > 0) {
        size_t lb = strlen(t.text);
        if (lb + 2 >= 64) {
            perr(p, line, "name too long");
            return -1;
        }
        for (int i = 0; i < v->n; i++) {
            memcpy(out[i], t.text, lb);
            out[i][lb] = '_';
            out[i][lb + 1] = "xyzw"[i];
            out[i][lb + 2] = '\0';
        }
        *nout = v->n;
        return 0;
    }
    if (copy_name(out[0], 64, t.text, p, line) != 0)
        return -1;
    *nout = 1;
    return 0;
}

static int surf_ref(parser_t *p, char *out, size_t cap, int line) {
    ptx_tok_t t;
    if (next(p, &t) != 0)
        return -1;
    if (t.kind != T_IDENT) {
        perr(p, line, "expected surface register or .surfref name");
        return -1;
    }
    return copy_name(out, cap, t.text, p, line);
}

static int surf_alias_param(const ptx_func_t *fn, size_t bi,
                            const char *ref, char *out, size_t cap,
                            int depth) {
    for (size_t i = 0; i < fn->nparams; i++)
        if (strcmp(fn->params[i].name, ref) == 0) {
            snprintf(out, cap, "%s", ref);
            return 1;
        }
    if (depth > 8)
        return 0;
    for (size_t b = 0; b <= bi && b < fn->nblocks; b++)
        for (size_t ii = 0; ii < fn->blocks[b].count; ii++) {
            const ptx_instr_t *m = &fn->blocks[b].instrs[ii];
            if (strcmp(m->dst, ref) != 0)
                continue;
            if (m->op == POP_LD && m->space == PSP_PARAM &&
                m->src[0].kind == OPD_ADDR && !m->src[0].addr_reg[0]) {
                snprintf(out, cap, "%s", m->src[0].param);
                return 1;
            }
            if (m->op == POP_MOV && m->nsrc > 0) {
                if (m->src[0].kind == OPD_PARAM) {
                    snprintf(out, cap, "%s", m->src[0].param);
                    return 1;
                }
                if (m->src[0].kind == OPD_REG &&
                    surf_alias_param(fn, bi, m->src[0].reg, out, cap,
                                     depth + 1))
                    return 1;
            }
        }
    return 0;
}

// A `.u64` kernel parameter used as a surface operand is a modern
// cudaSurfaceObject_t handle.  Mark it as a surface binding so the
// emitter creates a storage image and the driver resolves the handle.
static void surf_note_ref(ptx_func_t *fn, size_t bi, const char *ref) {
    char pname[256];
    if (!surf_alias_param(fn, bi, ref, pname, sizeof(pname), 0))
        return;
    for (size_t i = 0; i < fn->nparams; i++) {
        if (strcmp(fn->params[i].name, pname) != 0)
            continue;
        if (fn->params[i].is_tex == 0 && fn->params[i].ty == PTY_U64)
            fn->params[i].is_tex = 3;
        return;
    }
}

// Parse the surface operand and coordinate list: [a, {x,y,...}].
static int surf_coords(parser_t *p, ptx_func_t *fn, size_t bi,
                       ptx_instr_t *in, int line) {
    if (expect_punct(p, '[', "after surface dest") != 0)
        return -1;
    if (surf_ref(p, in->surf_ref, sizeof(in->surf_ref), line) != 0)
        return -1;
    surf_note_ref(fn, bi, in->surf_ref);
    if (expect_comma(p) != 0)
        return -1;
    int n = 0;
    if (surf_list(p, in->surf_coord, &n, 4, PTY_U32, line) != 0)
        return -1;
    if (expect_punct(p, ']', "after surface coordinates") != 0)
        return -1;
    in->surf_ncoord = n;
    return 0;
}

static int surf_dtype(const char *s, ptx_ty_t *ty) {
    if (strcmp(s, "b8") == 0) { *ty = PTY_U8; return 0; }
    if (strcmp(s, "b16") == 0) { *ty = PTY_U16; return 0; }
    if (strcmp(s, "b32") == 0) { *ty = PTY_U32; return 0; }
    if (strcmp(s, "b64") == 0) { *ty = PTY_U64; return 0; }
    return -1;
}

static int parse_suld(parser_t *p, ptx_func_t *fn, size_t bi,
                      char parts[SURF_MAX_PARTS][32], int np, int opline) {
    int geom = 0, vec = 1;
    ptx_ty_t ty = PTY_U32;
    int have_dtype = 0;
    for (int i = 2; i < np; i++) {
        const char *s = parts[i];
        int g = surf_geom_code(s);
        if (g) { geom = g; continue; }
        if (strcmp(s, "v2") == 0) { vec = 2; continue; }
        if (strcmp(s, "v4") == 0) { vec = 4; continue; }
        if (strcmp(s, "ca") == 0 || strcmp(s, "cg") == 0 ||
            strcmp(s, "cs") == 0 || strcmp(s, "cv") == 0)
            continue; // cache operations have no effect here
        if (strcmp(s, "trap") == 0 || strcmp(s, "clamp") == 0 ||
            strcmp(s, "zero") == 0)
            continue;
        if (surf_dtype(s, &ty) == 0) { have_dtype = 1; continue; }
        perr(p, opline, "bad suld qualifier");
        return -1;
    }
    if (!geom || !have_dtype) {
        perr(p, opline, "suld needs .b.{geom}.{vec}.{b8,b16,b32,b64}");
        return -1;
    }
    ptx_instr_t in;
    memset(&in, 0, sizeof(in));
    in.line = opline;
    in.op = POP_SULD;
    in.surf_geom = geom;
    in.surf_vec = vec;
    in.surf_ty = ty;
    int n = 0;
    if (surf_list(p, in.surf_dst, &n, 4, ty == PTY_U64 ? PTY_U64 : PTY_U32,
                  opline) != 0)
        return -1;
    if (n != vec) {
        perr(p, opline, "suld destination vector size mismatch");
        return -1;
    }
    if (expect_comma(p) != 0 || surf_coords(p, fn, bi, &in, opline) != 0)
        return -1;
    if (expect_punct(p, ';', "after suld") != 0)
        return -1;
    return block_add_instr(fn, bi, &in, p, opline);
}

static int parse_sust(parser_t *p, ptx_func_t *fn, size_t bi,
                      char parts[SURF_MAX_PARTS][32], int np, int opline) {
    int geom = 0, vec = 1, formatted = 0;
    ptx_ty_t ty = PTY_U32;
    int have_dtype = 0;
    for (int i = 1; i < np; i++) {
        const char *s = parts[i];
        if (strcmp(s, "b") == 0) { formatted = 0; continue; }
        if (strcmp(s, "p") == 0) { formatted = 1; continue; }
        int g = surf_geom_code(s);
        if (g) { geom = g; continue; }
        if (strcmp(s, "v2") == 0) { vec = 2; continue; }
        if (strcmp(s, "v4") == 0) { vec = 4; continue; }
        if (strcmp(s, "wb") == 0 || strcmp(s, "cg") == 0 ||
            strcmp(s, "cs") == 0 || strcmp(s, "wt") == 0)
            continue;
        if (strcmp(s, "trap") == 0 || strcmp(s, "clamp") == 0 ||
            strcmp(s, "zero") == 0)
            continue;
        if (surf_dtype(s, &ty) == 0) { have_dtype = 1; continue; }
        perr(p, opline, "bad sust qualifier");
        return -1;
    }
    if (!geom || !have_dtype) {
        perr(p, opline, "sust needs .{b,p}.{geom}.{vec}.type");
        return -1;
    }
    if (formatted && ty != PTY_U32) {
        perr(p, opline, "sust.p requires .b32 source elements");
        return -1;
    }
    if (formatted && vec != 1 && vec != 4) {
        perr(p, opline, "sust.p supports .v4.b32 or .b32");
        return -1;
    }
    ptx_instr_t in;
    memset(&in, 0, sizeof(in));
    in.line = opline;
    in.op = POP_SUST;
    in.surf_geom = geom;
    in.surf_vec = vec;
    in.surf_ty = formatted ? PTY_F32 : ty;
    in.surf_formatted = formatted;
    if (expect_punct(p, '[', "after sust") != 0)
        return -1;
    if (surf_ref(p, in.surf_ref, sizeof(in.surf_ref), opline) != 0)
        return -1;
    surf_note_ref(fn, bi, in.surf_ref);
    if (expect_comma(p) != 0)
        return -1;
    int n = 0;
    if (surf_list(p, in.surf_coord, &n, 4, PTY_U32, opline) != 0)
        return -1;
    if (expect_punct(p, ']', "after sust coordinates") != 0)
        return -1;
    in.surf_ncoord = n;
    if (expect_comma(p) != 0)
        return -1;
    int ns = 0;
    ptx_ty_t sty = formatted ? PTY_F32
                             : (ty == PTY_U64 ? PTY_U64 : PTY_U32);
    if (surf_list(p, in.surf_src, &ns, 4, sty, opline) != 0)
        return -1;
    if (ns != vec) {
        perr(p, opline, "sust source vector size mismatch");
        return -1;
    }
    if (expect_punct(p, ';', "after sust") != 0)
        return -1;
    return block_add_instr(fn, bi, &in, p, opline);
}

static int parse_sured(parser_t *p, ptx_func_t *fn, size_t bi,
                       char parts[SURF_MAX_PARTS][32], int np, int opline) {
    int geom = 0, op = -1, formatted = 0;
    ptx_ty_t ty = PTY_U32;
    int sign = 0, have_dtype = 0;
    for (int i = 1; i < np; i++) {
        const char *s = parts[i];
        if (strcmp(s, "b") == 0) { formatted = 0; continue; }
        if (strcmp(s, "p") == 0) { formatted = 1; continue; }
        int g = surf_geom_code(s);
        if (g) { geom = g; continue; }
        if (strcmp(s, "add") == 0) { op = 0; continue; }
        if (strcmp(s, "min") == 0) { op = 1; continue; }
        if (strcmp(s, "max") == 0) { op = 2; continue; }
        if (strcmp(s, "and") == 0) { op = 3; continue; }
        if (strcmp(s, "or") == 0) { op = 4; continue; }
        if (strcmp(s, "trap") == 0 || strcmp(s, "clamp") == 0 ||
            strcmp(s, "zero") == 0)
            continue;
        if (strcmp(s, "u32") == 0) { ty = PTY_U32; have_dtype = 1; continue; }
        if (strcmp(s, "f32") == 0) { ty = PTY_F32; have_dtype = 1; continue; }
        if (strcmp(s, "s32") == 0) { ty = PTY_U32; sign = 1;
                                     have_dtype = 1; continue; }
        if (strcmp(s, "b32") == 0) { ty = PTY_U32; have_dtype = 1; continue; }
        if (strcmp(s, "u64") == 0 || strcmp(s, "b64") == 0) {
            ty = PTY_U64;
            have_dtype = 1;
            continue;
        }
        if (strcmp(s, "s64") == 0) { ty = PTY_U64; sign = 1;
                                     have_dtype = 1; continue; }
        perr(p, opline, "bad sured qualifier");
        return -1;
    }
    if (!geom || op < 0 || !have_dtype) {
        perr(p, opline, "sured needs .{b,p}.{add,min,max,and,or}.geom.type");
        return -1;
    }
    ptx_instr_t in;
    memset(&in, 0, sizeof(in));
    in.line = opline;
    in.op = POP_SURED;
    in.surf_geom = geom;
    in.surf_redop = op;
    in.surf_ty = ty;
    in.surf_sign = sign;
    in.surf_formatted = formatted;
    if (expect_punct(p, '[', "after sured") != 0)
        return -1;
    if (surf_ref(p, in.surf_ref, sizeof(in.surf_ref), opline) != 0)
        return -1;
    surf_note_ref(fn, bi, in.surf_ref);
    if (expect_comma(p) != 0)
        return -1;
    int n = 0;
    if (surf_list(p, in.surf_coord, &n, 4, PTY_U32, opline) != 0)
        return -1;
    if (expect_punct(p, ']', "after sured coordinates") != 0)
        return -1;
    in.surf_ncoord = n;
    if (expect_comma(p) != 0 ||
        surf_list(p, in.surf_src, &n, 1, ty, opline) != 0)
        return -1;
    if (expect_punct(p, ';', "after sured") != 0)
        return -1;
    return block_add_instr(fn, bi, &in, p, opline);
}

static int parse_suq(parser_t *p, ptx_func_t *fn, size_t bi,
                     char parts[SURF_MAX_PARTS][32], int np, int opline) {
    int q = -1;
    for (int i = 1; i < np; i++) {
        if (strcmp(parts[i], "width") == 0) q = 0;
        else if (strcmp(parts[i], "height") == 0) q = 1;
        else if (strcmp(parts[i], "depth") == 0) q = 2;
        else if (strcmp(parts[i], "channel_data_type") == 0) q = 3;
        else if (strcmp(parts[i], "channel_order") == 0) q = 4;
        else if (strcmp(parts[i], "array_size") == 0) q = 5;
        else if (strcmp(parts[i], "memory_layout") == 0) q = 6;
        else if (strcmp(parts[i], "b32") != 0) {
            perr(p, opline, "bad suq qualifier");
            return -1;
        }
    }
    if (q < 0) {
        perr(p, opline, "suq needs a query name");
        return -1;
    }
    ptx_instr_t in;
    memset(&in, 0, sizeof(in));
    in.line = opline;
    in.op = POP_SUQ;
    in.surf_query = q;
    if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
        return -1;
    if (expect_comma(p) != 0 || expect_punct(p, '[', "after suq dest") != 0)
        return -1;
    if (surf_ref(p, in.surf_ref, sizeof(in.surf_ref), opline) != 0)
        return -1;
    surf_note_ref(fn, bi, in.surf_ref);
    if (expect_punct(p, ']', "after suq surface") != 0)
        return -1;
    if (expect_punct(p, ';', "after suq") != 0)
        return -1;
    return block_add_instr(fn, bi, &in, p, opline);
}

int parse_surface_instr(parser_t *p, ptx_func_t *fn, size_t bi,
                        const char *opname, int opline) {
    char parts[SURF_MAX_PARTS][32];
    int np = 0;
    memset(parts, 0, sizeof(parts));
    if (surf_split(opname, parts, &np) != 0) {
        perr(p, opline, "bad surface opcode");
        return -1;
    }
    if (strcmp(parts[0], "suld") == 0)
        return parse_suld(p, fn, bi, parts, np, opline);
    if (strcmp(parts[0], "sust") == 0)
        return parse_sust(p, fn, bi, parts, np, opline);
    if (strcmp(parts[0], "sured") == 0)
        return parse_sured(p, fn, bi, parts, np, opline);
    if (strcmp(parts[0], "suq") == 0)
        return parse_suq(p, fn, bi, parts, np, opline);
    perr(p, opline, "unsupported surface instruction");
    return -1;
}
