// Texture instruction front end: tex, tld4, txq and istypep.
//
// A .texref parameter maps to a Vulkan image (combined with a sampler in
// unified mode, or paired with a .samplerref in independent mode).  The
// parsed instruction records the geometry, mipmap mode and the optional
// offset/depth-compare operands; the SPIR-V emitter turns them into
// OpImageSample*/OpImageGather/OpImageFetch.
#include <stdio.h>
#include <string.h>
#include "ptx.h"
#include "parse/parse_internal.h"

// Non-consuming punctuation peek.
static int tex_peek(parser_t *p, char c) {
    ptx_tok_t t;
    if (next(p, &t) != 0)
        return 0;
    ptx_unget(&p->lx, &t);
    return t.kind == T_PUNCT && t.punct == c;
}

static int tex_opd_name(parser_t *p, char *out, size_t cap, int line) {
    ptx_tok_t t;
    if (next(p, &t) != 0)
        return -1;
    if (t.kind != T_IDENT) {
        perr(p, line, "expected texture register or .texref name");
        return -1;
    }
    return copy_name(out, cap, t.text, p, line);
}

// Optional destination predicate: "{d,...}|p".
static int parse_tex_dstp(parser_t *p, char *out, size_t cap, int line) {
    if (!at_punct(p, '|'))
        return 0;
    ptx_tok_t t;
    if (next(p, &t) != 0)
        return -1;
    if (t.kind != T_IDENT) {
        perr(p, line, "expected predicate after '|'");
        return -1;
    }
    return copy_name(out, cap, t.text, p, line);
}

// Geometry code: 1 1d, 2 2d, 3 3d, 4 a1d, 5 a2d, 6 cube, 7 acube,
// 8 2dms, 9 a2dms.  Returns 0 for an unknown token.
static int tex_geom_code(const char *s) {
    if (strcmp(s, "1d") == 0) return 1;
    if (strcmp(s, "2d") == 0) return 2;
    if (strcmp(s, "3d") == 0) return 3;
    if (strcmp(s, "a1d") == 0) return 4;
    if (strcmp(s, "a2d") == 0) return 5;
    if (strcmp(s, "cube") == 0) return 6;
    if (strcmp(s, "acube") == 0) return 7;
    if (strcmp(s, "2dms") == 0) return 8;
    if (strcmp(s, "a2dms") == 0) return 9;
    return 0;
}

// Parse a register list of 1..cap entries: either "{a, b, ...}" or a
// plain register (single entry or scalarized .vN register).
static int parse_tex_list(parser_t *p, char out[4][64], int *nout, int cap,
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
                perr(p, line, "texture coordinate list too long");
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
                } else if (o.kind == OPD_IMM) {
                    snprintf(out[n], 64, "%lld", (long long)o.imm);
                } else if (o.kind == OPD_PARAM &&
                           strcmp(o.param, "_") == 0) {
                    snprintf(out[n], 64, "__sink_%d_%d", line, n);
                } else {
                    perr(p, line, "texture list needs plain registers");
                    return -1;
                }
            }
            n++;
            if (at_punct(p, ','))
                continue;
            if (at_punct(p, '}'))
                break;
            perr(p, line, "expected ',' or '}' in texture list");
            return -1;
        }
        if (n == 0) {
            perr(p, line, "empty texture list");
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

// Number of PTX coordinate elements for a geometry (before padding).
static int geom_coord_count(int geom) {
    switch (geom) {
    case 1: return 1; // 1d
    case 4: return 2; // a1d: index + s
    case 2: return 2; // 2d
    case 5: return 3; // a2d: index + s,t
    case 3: return 3; // 3d
    case 6: return 3; // cube: s,t,r
    case 7: return 4; // acube: index + s,t,r
    case 8: return 3; // 2dms: sample + x,y
    case 9: return 4; // a2dms: sample + index + x,y
    }
    return 2;
}

// tex[.mode].geom.v4.dtype.ctype d, [a{, b}, c] {, e} {, f}
static int parse_tex(parser_t *p, ptx_func_t *fn, size_t bi,
                     char parts[6][32], int np, int opline) {
    int geom = 0, mode = 0, vec = 0;
    const char *dtype = NULL, *ctype = NULL;
    for (int i = 1; i < np; i++) {
        int g = tex_geom_code(parts[i]);
        if (g) {
            if (geom && geom != g) {
                perr(p, opline, "conflicting tex geometries");
                return -1;
            }
            geom = g;
        } else if (strcmp(parts[i], "base") == 0) {
            mode = 0;
        } else if (strcmp(parts[i], "level") == 0) {
            mode = 1;
        } else if (strcmp(parts[i], "grad") == 0) {
            mode = 2;
        } else if (strncmp(parts[i], "v", 1) == 0 && parts[i][1] >= '2' &&
                   parts[i][1] <= '4') {
            vec = parts[i][1] - '0';
        } else if (strcmp(parts[i], "f32") == 0 ||
                   strcmp(parts[i], "s32") == 0 ||
                   strcmp(parts[i], "u32") == 0 ||
                   strcmp(parts[i], "f16") == 0 ||
                   strcmp(parts[i], "f16x2") == 0) {
            if (!dtype)
                dtype = parts[i];
            else
                ctype = parts[i];
        } else {
            perr(p, opline, "bad tex qualifier");
            return -1;
        }
    }
    int f16 = 0;
    if (dtype && strcmp(dtype, "f16") == 0)
        f16 = 1;
    else if (dtype && strcmp(dtype, "f16x2") == 0)
        f16 = 2;
    if (!geom || !dtype || !ctype ||
        (f16 == 1 && vec != 4) || (f16 == 2 && vec != 2) ||
        (f16 == 0 && vec != 4)) {
        perr(p, opline, "tex needs .geom.v4.{f32,s32,u32}.{f32,s32} or "
                        ".geom.v{4,2}.{f16,f16x2}.ctype");
        return -1;
    }
    int nchan = f16 == 2 ? 2 : 4;
    if ((geom == 6 || geom == 7) && strcmp(ctype, "f32") != 0) {
        perr(p, opline, "tex.cube/.acube require .f32 coordinates");
        return -1;
    }
    if ((geom == 8 || geom == 9)) {
        if (strcmp(ctype, "s32") != 0) {
            perr(p, opline, "tex.2dms/.a2dms require .s32 coordinates");
            return -1;
        }
        if (mode != 0) {
            perr(p, opline, "tex.2dms/.a2dms do not support mipmap modes");
            return -1;
        }
        if (f16) {
            perr(p, opline, "tex.2dms/.a2dms require .u32/.s32/.f32 "
                            "destinations");
            return -1;
        }
    }
    ptx_instr_t in;
    memset(&in, 0, sizeof(in));
    in.line = opline;
    in.op = POP_TEX;
    in.tex_geom = geom;
    in.tex_mode = mode;
    in.ty = (f16 || strcmp(dtype, "f32") == 0) ? PTY_F32 : PTY_U32;
    in.sign = strcmp(dtype, "s32") == 0;
    in.ty2 = strcmp(ctype, "f32") == 0 ? PTY_F32 : PTY_U32;
    in.tex_csign = strcmp(ctype, "s32") == 0;
    in.tex_f16 = f16;
    ptx_ty_t dt = f16 == 1 ? PTY_F16 : (f16 == 2 ? PTY_F16X2 : in.ty);
    char dst[4][64];
    if (parse_vec_operand(p, dst, nchan, dt, opline) != 0)
        return -1;
    for (int i = 0; i < nchan; i++)
        snprintf(in.tex_dst[i], sizeof(in.tex_dst[i]), "%.63s", dst[i]);
    if (parse_tex_dstp(p, in.tex_dstp, sizeof(in.tex_dstp), opline) != 0)
        return -1;
    if (expect_comma(p) != 0 || expect_punct(p, '[', "after tex dest") != 0)
        return -1;
    if (tex_opd_name(p, in.tex_ref, sizeof(in.tex_ref), opline) != 0)
        return -1;
    if (expect_comma(p) != 0)
        return -1;
    int ncoord = 0, closed = 0;
    ptx_tok_t t;
    if (next(p, &t) != 0)
        return -1;
    if (t.kind == T_PUNCT && t.punct == '{') {
        ptx_unget(&p->lx, &t);
        if (parse_tex_list(p, in.tex_coord, &ncoord, 4, in.ty2, opline) != 0)
            return -1;
    } else if (t.kind == T_IDENT) {
        ptx_tok_t u;
        if (next(p, &u) != 0)
            return -1;
        if (u.kind == T_PUNCT && u.punct == ',') {
            // Explicit sampler form: [tex, sampler, coords].
            if (copy_name(in.tex_sampler, sizeof(in.tex_sampler), t.text,
                          p, opline) != 0)
                return -1;
            if (parse_tex_list(p, in.tex_coord, &ncoord, 4, in.ty2,
                               opline) != 0)
                return -1;
        } else if (u.kind == T_PUNCT && u.punct == ']') {
            // Single bare coordinate, e.g. tex.1d ..., [tex, f1].
            if (copy_name(in.tex_coord[0], sizeof(in.tex_coord[0]),
                          t.text, p, opline) != 0)
                return -1;
            ncoord = 1;
            closed = 1;
        } else {
            perr(p, opline, "expected ',' or ']' after tex texture");
            return -1;
        }
    } else {
        perr(p, opline, "expected texture coordinates");
        return -1;
    }
    int want = geom_coord_count(geom);
    if (ncoord < want) {
        perr(p, opline, "too few texture coordinates for geometry");
        return -1;
    }
    if (ncoord > 4) {
        perr(p, opline, "too many texture coordinates");
        return -1;
    }
    if (!closed && expect_punct(p, ']', "after tex coordinates") != 0)
        return -1;
    in.tex_ncoord = ncoord;
    in.tex_nchan = nchan;

    if (mode == 1) {
        if (expect_comma(p) != 0)
            return -1;
        char lod[4][64];
        int n = 0;
        if (parse_tex_list(p, lod, &n, 1, in.ty2, opline) != 0)
            return -1;
        if (n != 1) {
            perr(p, opline, "tex.level needs a scalar LOD");
            return -1;
        }
        snprintf(in.tex_lod, sizeof(in.tex_lod), "%.63s", lod[0]);
    } else if (mode == 2) {
        int n = 0;
        if (expect_comma(p) != 0 ||
            parse_tex_list(p, in.tex_dpdx, &n, 4, PTY_F32, opline) != 0)
            return -1;
        if (expect_comma(p) != 0 ||
            parse_tex_list(p, in.tex_dpdy, &n, 4, PTY_F32, opline) != 0)
            return -1;
    }
    // Optional offset vector: {off...}.
    if (at_punct(p, ',')) {
        if (tex_peek(p, '{')) {
            int n = 0;
            if (parse_tex_list(p, in.tex_off, &n, 4, PTY_U32,
                               opline) != 0)
                return -1;
            in.tex_noff = n;
            if (at_punct(p, ',')) {
                if (tex_opd_name(p, in.tex_cmp, sizeof(in.tex_cmp),
                                 opline) != 0)
                    return -1;
                in.tex_compare = 1;
            }
        } else {
            // Depth-compare value (scalar .f32 register).
            if (tex_opd_name(p, in.tex_cmp, sizeof(in.tex_cmp), opline) != 0)
                return -1;
            in.tex_compare = 1;
        }
    }
    if (expect_punct(p, ';', "after tex") != 0)
        return -1;
    return block_add_instr(fn, bi, &in, p, opline);
}

// tld4.comp.geom.v4.dtype.f32 d, [a{, b}, c] {, e} {, f}
static int parse_tld4(parser_t *p, ptx_func_t *fn, size_t bi,
                      char parts[6][32], int np, int opline) {
    int comp = -1, geom = 0, vec = 0;
    const char *dtype = NULL;
    for (int i = 1; i < np; i++) {
        if (strcmp(parts[i], "r") == 0) comp = 0;
        else if (strcmp(parts[i], "g") == 0) comp = 1;
        else if (strcmp(parts[i], "b") == 0) comp = 2;
        else if (strcmp(parts[i], "a") == 0) comp = 3;
        else {
            int g = tex_geom_code(parts[i]);
            if (g)
                geom = g;
            else if (strncmp(parts[i], "v", 1) == 0 &&
                     parts[i][1] >= '2' && parts[i][1] <= '4')
                vec = parts[i][1] - '0';
            else if (strcmp(parts[i], "f32") == 0 ||
                     strcmp(parts[i], "s32") == 0 ||
                     strcmp(parts[i], "u32") == 0) {
                if (!dtype) dtype = parts[i];
            } else {
                perr(p, opline, "bad tld4 qualifier");
                return -1;
            }
        }
    }
    if (comp < 0 || vec != 4 || !dtype ||
        (geom != 2 && geom != 5 && geom != 6 && geom != 7)) {
        perr(p, opline, "tld4 supports tld4.{r,g,b,a}.{2d,a2d,cube,acube}"
                        ".v4.{f32,s32,u32}.f32");
        return -1;
    }
    ptx_instr_t in;
    memset(&in, 0, sizeof(in));
    in.line = opline;
    in.op = POP_TLD4;
    in.tex_geom = geom;
    in.tex_comp = comp;
    in.ty = strcmp(dtype, "f32") == 0 ? PTY_F32 : PTY_U32;
    in.ty2 = PTY_F32;
    char dst[4][64];
    if (parse_vec_operand(p, dst, 4, in.ty, opline) != 0)
        return -1;
    for (int i = 0; i < 4; i++)
        snprintf(in.tex_dst[i], sizeof(in.tex_dst[i]), "%.63s", dst[i]);
    if (parse_tex_dstp(p, in.tex_dstp, sizeof(in.tex_dstp), opline) != 0)
        return -1;
    if (expect_comma(p) != 0 || expect_punct(p, '[', "after tld4 dest") != 0)
        return -1;
    if (tex_opd_name(p, in.tex_ref, sizeof(in.tex_ref), opline) != 0)
        return -1;
    if (expect_comma(p) != 0)
        return -1;
    int ncoord = 0;
    ptx_tok_t t;
    if (next(p, &t) != 0)
        return -1;
    if (t.kind == T_PUNCT && t.punct == '{') {
        ptx_unget(&p->lx, &t);
        if (parse_tex_list(p, in.tex_coord, &ncoord, 4, PTY_F32, opline) != 0)
            return -1;
    } else if (t.kind == T_IDENT) {
        ptx_tok_t u;
        if (next(p, &u) != 0)
            return -1;
        if (u.kind == T_PUNCT && u.punct == ',') {
            if (copy_name(in.tex_sampler, sizeof(in.tex_sampler), t.text,
                          p, opline) != 0)
                return -1;
            if (parse_tex_list(p, in.tex_coord, &ncoord, 4, PTY_F32,
                               opline) != 0)
                return -1;
        } else if (u.kind == T_PUNCT && u.punct == ']') {
            if (copy_name(in.tex_coord[0], sizeof(in.tex_coord[0]),
                          t.text, p, opline) != 0)
                return -1;
            ncoord = 1;
        } else {
            perr(p, opline, "expected ',' or ']' after tld4 texture");
            return -1;
        }
    } else {
        perr(p, opline, "expected tld4 coordinates");
        return -1;
    }
    if (ncoord < geom_coord_count(geom)) {
        perr(p, opline, "too few tld4 coordinates for geometry");
        return -1;
    }
    if (expect_punct(p, ']', "after tld4 coordinates") != 0)
        return -1;
    if (at_punct(p, ',')) {
        if (tex_peek(p, '{')) {
            int n = 0;
            if (parse_tex_list(p, in.tex_off, &n, 4, PTY_U32,
                               opline) != 0)
                return -1;
            in.tex_noff = n;
            if (at_punct(p, ',')) {
                if (tex_opd_name(p, in.tex_cmp, sizeof(in.tex_cmp),
                                 opline) != 0)
                    return -1;
                in.tex_compare = 1;
            }
        } else {
            if (tex_opd_name(p, in.tex_cmp, sizeof(in.tex_cmp), opline) != 0)
                return -1;
            in.tex_compare = 1;
        }
    }
    if (expect_punct(p, ';', "after tld4") != 0)
        return -1;
    in.tex_ncoord = ncoord;
    in.tex_nchan = 4;
    return block_add_instr(fn, bi, &in, p, opline);
}

// txq.query{.level}.b32 d, [tex]{, lod}
static int parse_txq(parser_t *p, ptx_func_t *fn, size_t bi,
                     char parts[6][32], int np, int opline) {
    int q = -1, level = 0;
    for (int i = 1; i < np; i++) {
        if (strcmp(parts[i], "level") == 0) {
            level = 1;
        } else if (strcmp(parts[i], "width") == 0)
            q = 0;
        else if (strcmp(parts[i], "height") == 0)
            q = 1;
        else if (strcmp(parts[i], "depth") == 0)
            q = 2;
        else if (strcmp(parts[i], "channel_data_type") == 0)
            q = 3;
        else if (strcmp(parts[i], "channel_order") == 0)
            q = 4;
        else if (strcmp(parts[i], "normalized_coords") == 0)
            q = 5;
        else if (strcmp(parts[i], "array_size") == 0)
            q = 6;
        else if (strcmp(parts[i], "num_mipmap_levels") == 0)
            q = 7;
        else if (strcmp(parts[i], "num_samples") == 0)
            q = 8;
        else if (strcmp(parts[i], "filter_mode") == 0)
            q = 9;
        else if (strcmp(parts[i], "addr_mode_0") == 0)
            q = 10;
        else if (strcmp(parts[i], "addr_mode_1") == 0)
            q = 11;
        else if (strcmp(parts[i], "addr_mode_2") == 0)
            q = 12;
        else if (strcmp(parts[i], "force_unnormalized_coords") == 0)
            q = 13;
        else if (strcmp(parts[i], "b32") != 0) {
            perr(p, opline, "bad txq qualifier");
            return -1;
        }
    }
    if (q < 0) {
        perr(p, opline, "txq needs a query name");
        return -1;
    }
    ptx_instr_t in;
    memset(&in, 0, sizeof(in));
    in.line = opline;
    in.op = POP_TXQ;
    in.ty = PTY_U32;
    in.tex_query = q;
    if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
        return -1;
    if (expect_comma(p) != 0 || expect_punct(p, '[', "after txq dest") != 0)
        return -1;
    if (tex_opd_name(p, in.tex_ref, sizeof(in.tex_ref), opline) != 0)
        return -1;
    if (expect_punct(p, ']', "after txq texture") != 0)
        return -1;
    if (level) {
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
            return -1;
        in.nsrc = 1;
    }
    if (expect_punct(p, ';', "after txq") != 0)
        return -1;
    return block_add_instr(fn, bi, &in, p, opline);
}

// istypep.{texref,samplerref,surfref} p, a;
static int parse_istypep(parser_t *p, ptx_func_t *fn, size_t bi,
                         char parts[6][32], int np, int opline) {
    if (np != 2) {
        perr(p, opline, "bad istypep form");
        return -1;
    }
    int kind;
    if (strcmp(parts[1], "texref") == 0)
        kind = 0;
    else if (strcmp(parts[1], "samplerref") == 0)
        kind = 1;
    else if (strcmp(parts[1], "surfref") == 0)
        kind = 2;
    else {
        perr(p, opline, "istypep supports .texref/.samplerref/.surfref");
        return -1;
    }
    ptx_instr_t in;
    memset(&in, 0, sizeof(in));
    in.line = opline;
    in.op = POP_ISTYPEP;
    in.istypep_kind = kind;
    if (parse_dst(p, in.dst, sizeof(in.dst), 1) != 0)
        return -1;
    if (expect_comma(p) != 0)
        return -1;
    if (tex_opd_name(p, in.tex_ref, sizeof(in.tex_ref), opline) != 0)
        return -1;
    if (expect_punct(p, ';', "after istypep") != 0)
        return -1;
    return block_add_instr(fn, bi, &in, p, opline);
}

int parse_texture_instr(parser_t *p, ptx_func_t *fn, size_t bi,
                        const char *opname, int opline) {
    char parts[6][32];
    int np = 0;
    if (split_op(opname, parts, &np) != 0) {
        perr(p, opline, "bad texture opcode");
        return -1;
    }
    if (strcmp(parts[0], "tex") == 0)
        return parse_tex(p, fn, bi, parts, np, opline);
    if (strcmp(parts[0], "txq") == 0)
        return parse_txq(p, fn, bi, parts, np, opline);
    if (strcmp(parts[0], "istypep") == 0)
        return parse_istypep(p, fn, bi, parts, np, opline);
    if (strcmp(parts[0], "tld4") == 0)
        return parse_tld4(p, fn, bi, parts, np, opline);
    perr(p, opline, "unsupported texture instruction");
    return -1;
}
