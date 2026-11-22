// Module/function-scope declarations: .shared/.extern shared, call
// param slots, .local, .const and .global images.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ptx.h"
#include "parse/parse_internal.h"

static size_t ext_ty_bytes(ptx_ty_t ty);

// Round-to-nearest-even f32 bits -> f16 bits (decimal .f16 constants).
static uint16_t const_f32_to_f16(uint32_t w) {
    uint32_t s = (w >> 16) & 0x8000u;
    int e = (int)((w >> 23) & 0xffu) - 127;
    uint32_t m = w & 0x7fffffu;
    if (((w >> 23) & 0xffu) == 0xffu)
        return (uint16_t)(s | 0x7c00u | (m ? 0x200u | (m >> 13) : 0));
    if (e >= 16)
        return (uint16_t)(s | 0x7c00u);
    if (e >= -14) {
        uint32_t mant = m >> 13, drop = m & 0x1fffu;
        uint32_t half = 0x1000u;
        if (drop > half || (drop == half && (mant & 1u))) {
            mant++;
            if (mant == 0x400u) {
                mant = 0;
                e++;
                if (e >= 16)
                    return (uint16_t)(s | 0x7c00u);
            }
        }
        return (uint16_t)(s | ((uint32_t)(e + 15) << 10) | mant);
    }
    uint32_t full = m | 0x800000u;
    int rsh = -e - 1;
    if (rsh >= 25)
        return (uint16_t)s;
    uint32_t mant = full >> (unsigned)rsh;
    uint32_t drop = full & ~(0xffffffffu << (unsigned)rsh);
    uint32_t half = 1u << (unsigned)(rsh - 1);
    if (drop > half || (drop == half && (mant & 1u)))
        mant++;
    return (uint16_t)(s | mant);
}

int parse_shared_decl(parser_t *p, ptx_unit_t *unit) {
    ptx_tok_t t;
    int align = 0;
    if (next(p, &t) != 0)
        return -1;
    if (t.kind == T_IDENT && strcmp(t.text, ".align") == 0) {
        if (next(p, &t) != 0)
            return -1;
        if (t.kind != T_INT || t.imm == 0 || t.imm > 64) {
            perr(p, t.line, "expected sane .align value");
            return -1;
        }
        align = (int)t.imm;
        if (next(p, &t) != 0)
            return -1;
    }
    if (t.kind != T_IDENT) {
        perr(p, t.line, "expected shared variable type");
        return -1;
    }
    size_t elem = 0;
    const char *ts = t.text[0] == '.' ? t.text + 1 : t.text;
    if (strcmp(ts, "b8") == 0 || strcmp(ts, "u8") == 0 ||
        strcmp(ts, "s8") == 0) {
        elem = 1;
    } else if (strcmp(ts, "b16") == 0 || strcmp(ts, "u16") == 0 ||
               strcmp(ts, "s16") == 0 || strcmp(ts, "f16") == 0) {
        elem = 2;
    } else if (strcmp(ts, "b32") == 0 || strcmp(ts, "u32") == 0 ||
               strcmp(ts, "s32") == 0 || strcmp(ts, "f32") == 0) {
        elem = 4;
    } else if (strcmp(ts, "b64") == 0 || strcmp(ts, "u64") == 0 ||
               strcmp(ts, "s64") == 0 || strcmp(ts, "f64") == 0) {
        elem = 8;
    } else {
        char m[640];
        snprintf(m, sizeof(m), "unsupported shared type '%s'", t.text);
        perr(p, t.line, m);
        return -1;
    }
    if (next(p, &t) != 0)
        return -1;
    if (t.kind != T_IDENT || t.text[0] == '%' || t.text[0] == '.') {
        perr(p, t.line, "expected shared variable name");
        return -1;
    }
    char name[512];
    if (copy_name(name, sizeof(name), t.text, p, t.line) != 0)
        return -1;
    // Scalar form (no brackets) declares a single element.
    uint64_t dim = 1;
    if (at_punct(p, '[')) {
        if (next(p, &t) != 0)
            return -1;
        if (t.kind != T_INT || t.imm == 0) {
            perr(p, t.line, "shared dimension must be a nonzero size (dynamic .shared unsupported)");
            return -1;
        }
        dim = t.imm;
        if (expect_punct(p, ']', "in shared dimension") != 0)
            return -1;
    }
    if (expect_punct(p, ';', "after shared declaration") != 0)
        return -1;
    if (dim > ((uint64_t)1 << 20) ||
        dim * (uint64_t)elem > ((uint64_t)1 << 20)) {
        perr(p, t.line, "shared size insane (max 1MB sanity cap)");
        return -1;
    }
    for (size_t i = 0; i < unit->nshvars; i++) {
        if (strcmp(unit->shvars[i].name, name) == 0) {
            // Identical re-declaration (separate PTX scopes, including
            // in-function `.shared` across entries, legitimately repeat
            // the same variable): reuse. Conflicts stay errors.
            if (unit->shvars[i].bytes == (size_t)dim * elem &&
                unit->shvars[i].elem == elem &&
                unit->shvars[i].align == align)
                return 0;
            perr(p, t.line, "duplicate shared variable");
            return -1;
        }
    }
    ptx_shvar_t *nsv = realloc(unit->shvars,
                               sizeof(*nsv) * (unit->nshvars + 1));
    if (!nsv) {
        perr(p, t.line, "out of memory");
        return -1;
    }
    unit->shvars = nsv;
    strcpy(unit->shvars[unit->nshvars].name, name);
    unit->shvars[unit->nshvars].elem = elem;
    unit->shvars[unit->nshvars].bytes = (size_t)dim * elem;
    unit->shvars[unit->nshvars].align = align;
    unit->shvars[unit->nshvars].is_extern = 0;
    unit->nshvars++;
    return 0;
}

// Dynamic `.shared` declaration (`.extern` already consumed):
// ".extern .shared [.align N] .type name[]". Unsized only (sized
// externals are not valid PTX); the size arrives via launch args and
// all extern arrays alias the same dynamic base.
int parse_extern_shared(parser_t *p, ptx_unit_t *unit) {
    ptx_tok_t t;
    if (next(p, &t) != 0)
        return -1;
    if (t.kind == T_IDENT && strcmp(t.text, ".align") == 0) {
        if (next(p, &t) != 0)
            return -1;
        if (t.kind != T_INT || t.imm == 0 || t.imm > 64) {
            perr(p, t.line, "expected sane .align value");
            return -1;
        }
        if (next(p, &t) != 0)
            return -1;
    }
    if (t.kind != T_IDENT) {
        perr(p, t.line, "expected shared variable type");
        return -1;
    }
    {
        const char *ts = t.text[0] == '.' ? t.text + 1 : t.text;
        if (!(strcmp(ts, "b8") == 0 || strcmp(ts, "u8") == 0 ||
              strcmp(ts, "s8") == 0 || strcmp(ts, "b16") == 0 ||
              strcmp(ts, "u16") == 0 || strcmp(ts, "s16") == 0 ||
              strcmp(ts, "f16") == 0 || strcmp(ts, "b32") == 0 ||
              strcmp(ts, "u32") == 0 || strcmp(ts, "s32") == 0 ||
              strcmp(ts, "f32") == 0 || strcmp(ts, "b64") == 0 ||
              strcmp(ts, "u64") == 0 || strcmp(ts, "s64") == 0 ||
              strcmp(ts, "f64") == 0)) {
            char m[640];
            snprintf(m, sizeof(m), "unsupported shared type '%s'", t.text);
            perr(p, t.line, m);
            return -1;
        }
    }
    if (next(p, &t) != 0)
        return -1;
    if (t.kind != T_IDENT || t.text[0] == '%' || t.text[0] == '.') {
        perr(p, t.line, "expected shared variable name");
        return -1;
    }
    char name[512];
    if (copy_name(name, sizeof(name), t.text, p, t.line) != 0)
        return -1;
    if (expect_punct(p, '[', "after shared name") != 0)
        return -1;
    if (!at_punct(p, ']')) {
        perr(p, t.line, "extern .shared must be unsized ([])");
        return -1;
    }
    if (expect_punct(p, ';', "after shared declaration") != 0)
        return -1;
    for (size_t i = 0; i < unit->nshvars; i++) {
        if (strcmp(unit->shvars[i].name, name) == 0) {
            // All extern arrays alias the same dynamic base, so a
            // repeated extern declaration is naturally shared; any
            // other re-declaration is a conflict.
            if (unit->shvars[i].is_extern)
                return 0;
            perr(p, t.line, "shared variable redeclared as extern");
            return -1;
        }
    }
    ptx_shvar_t *nsv = realloc(unit->shvars,
                               sizeof(*nsv) * (unit->nshvars + 1));
    if (!nsv) {
        perr(p, t.line, "out of memory");
        return -1;
    }
    unit->shvars = nsv;
    strcpy(unit->shvars[unit->nshvars].name, name);
    unit->shvars[unit->nshvars].elem = 0;
    unit->shvars[unit->nshvars].bytes = 0;
    unit->shvars[unit->nshvars].align = 0;
    unit->shvars[unit->nshvars].is_extern = 1;
    unit->nshvars++;
    return 0;
}

// Function-scope ".local" declaration (".local" already consumed).
// Same shape as .shared; stored per-function (Private storage is
// per-invocation, so identical names across entries stay separate).
// In-function `.param` call slot (ZLUDA call corpus). `head` is the
// already-read first token: ".param" (spaced form, type follows) or
// ".param.<type>" (combined, name follows). Records the cell for the
// post-parse call-inline pass; arrays stay clean errors.
int parse_param_slot(parser_t *p, ptx_func_t *fn, const char *head,
                            int headline) {
    ptx_ty_t sty = PTY_U32;
    int align = 0;
    if (strncmp(head, ".param.", 7) == 0) {
        if (map_ty(head + 7, &sty) != 0 || sty == PTY_PRED) {
            perr(p, headline, "bad param slot type");
            return -1;
        }
    } else {
        ptx_tok_t t;
        if (next(p, &t) != 0)
            return -1;
        if (t.kind == T_IDENT && strcmp(t.text, ".align") == 0) {
            if (next(p, &t) != 0)
                return -1;
            if (t.kind != T_INT || t.imm == 0 || t.imm > 64) {
                perr(p, t.line, "expected sane alignment value");
                return -1;
            }
            align = (int)t.imm;
            if (next(p, &t) != 0)
                return -1;
        }
        if (t.kind != T_IDENT || map_ty(t.text, &sty) != 0 ||
            sty == PTY_PRED) {
            perr(p, t.line, "bad param slot type");
            return -1;
        }
    }
    size_t elem = ext_ty_bytes(sty);
    for (;;) {
        ptx_tok_t t;
        if (next(p, &t) != 0)
            return -1;
        if (t.kind != T_IDENT) {
            perr(p, t.line, "expected param slot name");
            return -1;
        }
        char slotname[256];
        if (copy_name(slotname, sizeof(slotname), t.text, p, t.line) != 0)
            return -1;
        for (size_t si = 0; si < fn->nslots; si++) {
            if (strcmp(fn->slots[si].name, slotname) == 0) {
                perr(p, t.line, "duplicate param slot");
                return -1;
            }
        }
        if (fn->nslots >= 256) {
            perr(p, t.line, "too many param slots");
            return -1;
        }
        size_t bytes = elem;
        int salign = align > 0 ? align : (int)(elem > 8 ? 8 : elem);
        if (at_punct(p, '[')) {
            if (next(p, &t) != 0)
                return -1;
            if (t.kind != T_INT || t.imm == 0 || t.imm > 65536) {
                perr(p, t.line, "bad param slot array size");
                return -1;
            }
            bytes = elem * (size_t)t.imm;
            if (expect_punct(p, ']', "in param slot array") != 0)
                return -1;
        }
        if (fn->nslots + 1 > fn->slots_cap) {
            size_t ncap = fn->slots_cap ? fn->slots_cap * 2 : 4;
            ptx_paramslot_t *ns = malloc(sizeof(*ns) * ncap);
            if (!ns) {
                perr(p, t.line, "out of memory");
                return -1;
            }
            memcpy(ns, fn->slots, sizeof(*ns) * fn->nslots);
            free(fn->slots);
            fn->slots = ns;
            fn->slots_cap = ncap;
        }
        snprintf(fn->slots[fn->nslots].name,
                 sizeof(fn->slots[fn->nslots].name), "%s", slotname);
        fn->slots[fn->nslots].ty = sty;
        fn->slots[fn->nslots].bytes = bytes;
        fn->slots[fn->nslots].align = salign;
        fn->slots[fn->nslots].host_call_id = -1;
        fn->nslots++;
        if (at_punct(p, ',')) {
            continue;
        }
        if (expect_punct(p, ';', "after param slot") != 0)
            return -1;
        break;
    }
    return 0; // declaration only
}

static int parse_local_into(parser_t *p, ptx_shvar_t **listp, size_t *np,
                            size_t *capp) {
    ptx_tok_t t;
    int align = 0;
    if (next(p, &t) != 0)
        return -1;
    if (t.kind == T_IDENT && strcmp(t.text, ".align") == 0) {
        if (next(p, &t) != 0)
            return -1;
        if (t.kind != T_INT || t.imm == 0 || t.imm > 64) {
            perr(p, t.line, "expected sane .align value");
            return -1;
        }
        align = (int)t.imm;
        if (next(p, &t) != 0)
            return -1;
    }
    if (t.kind != T_IDENT) {
        perr(p, t.line, "expected local variable type");
        return -1;
    }
    size_t elem = 0;
    const char *ts = t.text[0] == '.' ? t.text + 1 : t.text;
    if (strcmp(ts, "b8") == 0 || strcmp(ts, "u8") == 0 ||
        strcmp(ts, "s8") == 0) {
        elem = 1;
    } else if (strcmp(ts, "b16") == 0 || strcmp(ts, "u16") == 0 ||
               strcmp(ts, "s16") == 0 || strcmp(ts, "f16") == 0) {
        elem = 2;
    } else if (strcmp(ts, "b32") == 0 || strcmp(ts, "u32") == 0 ||
               strcmp(ts, "s32") == 0 || strcmp(ts, "f32") == 0) {
        elem = 4;
    } else if (strcmp(ts, "b64") == 0 || strcmp(ts, "u64") == 0 ||
               strcmp(ts, "s64") == 0 || strcmp(ts, "f64") == 0) {
        elem = 8;
    } else {
        char m[640];
        snprintf(m, sizeof(m), "unsupported local type '%s'", t.text);
        perr(p, t.line, m);
        return -1;
    }
    if (next(p, &t) != 0)
        return -1;
    if (t.kind != T_IDENT || t.text[0] == '%' || t.text[0] == '.') {
        perr(p, t.line, "expected local variable name");
        return -1;
    }
    char name[512];
    if (copy_name(name, sizeof(name), t.text, p, t.line) != 0)
        return -1;
    // Scalar form (no brackets) declares a single element.
    uint64_t dim = 1;
    if (at_punct(p, '[')) {
        if (next(p, &t) != 0)
            return -1;
        if (t.kind != T_INT || t.imm == 0) {
            perr(p, t.line, "local dimension must be a nonzero size");
            return -1;
        }
        dim = t.imm;
        if (expect_punct(p, ']', "in local dimension") != 0)
            return -1;
    }
    if (expect_punct(p, ';', "after local declaration") != 0)
        return -1;
    if (dim > ((uint64_t)1 << 20) ||
        dim * (uint64_t)elem > ((uint64_t)1 << 20)) {
        perr(p, t.line, "local size insane (max 1MB sanity cap)");
        return -1;
    }
    for (size_t i = 0; i < *np; i++) {
        if (strcmp((*listp)[i].name, name) == 0) {
            if ((*listp)[i].bytes == (size_t)dim * elem &&
                (*listp)[i].align == align)
                return 0;
            perr(p, t.line, "duplicate local variable");
            return -1;
        }
    }
    if (*np + 1 > *capp) {
        size_t ncap = *capp ? *capp * 2 : 4;
        ptx_shvar_t *nl = realloc(*listp, sizeof(*nl) * ncap);
        if (!nl) {
            perr(p, t.line, "out of memory");
            return -1;
        }
        *listp = nl;
        *capp = ncap;
    }
    strcpy((*listp)[*np].name, name);
    (*listp)[*np].bytes = (size_t)dim * elem;
    (*listp)[*np].align = align;
    (*listp)[*np].is_extern = 0;
    (*np)++;
    return 0;
}

int parse_local_decl(parser_t *p, ptx_func_t *fn) {
    return parse_local_into(p, &fn->locals, &fn->nlocals,
                            &fn->locals_cap);
}

int parse_module_local_decl(parser_t *p, ptx_unit_t *unit) {
    return parse_local_into(p, &unit->module_locals,
                            &unit->nmodule_locals,
                            &unit->module_locals_cap);
}

// Module-scope ".common" global:
//   .common [.global] [.align N] .type name [ '[' dim ']' ] ;
// The storage is zero-initialized and writable; it is lowered as
// module-scope Private storage (see parse_module_local_decl).
int parse_common_decl(parser_t *p, ptx_unit_t *unit) {
    ptx_tok_t t;
    int align = 0;
    if (next(p, &t) != 0)
        return -1;
    if (t.kind == T_IDENT && strcmp(t.text, ".global") == 0) {
        if (next(p, &t) != 0)
            return -1;
    }
    if (t.kind == T_IDENT && strcmp(t.text, ".align") == 0) {
        if (next(p, &t) != 0)
            return -1;
        if (t.kind != T_INT || t.imm == 0 || t.imm > 64) {
            perr(p, t.line, "expected sane .align value");
            return -1;
        }
        align = (int)t.imm;
        if (next(p, &t) != 0)
            return -1;
    }
    if (t.kind != T_IDENT) {
        perr(p, t.line, "expected .common variable type");
        return -1;
    }
    size_t elem = 0;
    const char *ts = t.text[0] == '.' ? t.text + 1 : t.text;
    if (strcmp(ts, "b8") == 0 || strcmp(ts, "u8") == 0 ||
        strcmp(ts, "s8") == 0) {
        elem = 1;
    } else if (strcmp(ts, "b16") == 0 || strcmp(ts, "u16") == 0 ||
               strcmp(ts, "s16") == 0 || strcmp(ts, "f16") == 0) {
        elem = 2;
    } else if (strcmp(ts, "b32") == 0 || strcmp(ts, "u32") == 0 ||
               strcmp(ts, "s32") == 0 || strcmp(ts, "f32") == 0) {
        elem = 4;
    } else if (strcmp(ts, "b64") == 0 || strcmp(ts, "u64") == 0 ||
               strcmp(ts, "s64") == 0 || strcmp(ts, "f64") == 0) {
        elem = 8;
    } else {
        char m[640];
        snprintf(m, sizeof(m), "unsupported .common type '%s'", t.text);
        perr(p, t.line, m);
        return -1;
    }
    if (next(p, &t) != 0)
        return -1;
    if (t.kind != T_IDENT || t.text[0] == '%' || t.text[0] == '.') {
        perr(p, t.line, "expected .common variable name");
        return -1;
    }
    char name[512];
    if (copy_name(name, sizeof(name), t.text, p, t.line) != 0)
        return -1;
    uint64_t dim = 1;
    if (at_punct(p, '[')) {
        if (next(p, &t) != 0)
            return -1;
        if (t.kind != T_INT || t.imm == 0) {
            perr(p, t.line, ".common dimension must be a nonzero size");
            return -1;
        }
        dim = t.imm;
        if (expect_punct(p, ']', "in .common dimension") != 0)
            return -1;
    }
    if (expect_punct(p, ';', "after .common declaration") != 0)
        return -1;
    if (dim > ((uint64_t)1 << 20) ||
        dim * (uint64_t)elem > ((uint64_t)1 << 20)) {
        perr(p, t.line, ".common size insane (max 1MB sanity cap)");
        return -1;
    }
    for (size_t i = 0; i < unit->nmodule_locals; i++) {
        if (strcmp(unit->module_locals[i].name, name) == 0) {
            if (unit->module_locals[i].bytes == (size_t)dim * elem &&
                unit->module_locals[i].align == align) {
                unit->module_locals[i].is_common = 1;
                return 0;
            }
            perr(p, t.line, "duplicate .common variable");
            return -1;
        }
    }
    if (unit->nmodule_locals + 1 > unit->module_locals_cap) {
        size_t ncap = unit->module_locals_cap
                          ? unit->module_locals_cap * 2
                          : 4;
        ptx_shvar_t *nl = realloc(unit->module_locals,
                                  sizeof(*nl) * ncap);
        if (!nl) {
            perr(p, t.line, "out of memory");
            return -1;
        }
        unit->module_locals = nl;
        unit->module_locals_cap = ncap;
    }
    ptx_shvar_t *nv = &unit->module_locals[unit->nmodule_locals++];
    memset(nv, 0, sizeof(*nv));
    strcpy(nv->name, name);
    nv->elem = elem;
    nv->bytes = (size_t)dim * elem;
    nv->align = align;
    nv->is_common = 1;
    return 0;
}

// Module-scope ".const" declaration with a literal initializer list
// (".const" already consumed). Loads materialize as immediates, so
// only literal integer lanes are accepted (symbolic references like
// const_ident's `{ x, y }` stay clean errors, as do unsized/u64 forms).
int parse_const_decl(parser_t *p, ptx_unit_t *unit) {
    ptx_tok_t t;
    if (next(p, &t) != 0)
        return -1;
    if (t.kind == T_IDENT && strcmp(t.text, ".align") == 0) {
        if (next(p, &t) != 0)
            return -1;
        if (t.kind != T_INT || t.imm == 0 || t.imm > 64) {
            perr(p, t.line, "expected sane .align value");
            return -1;
        }
        if (next(p, &t) != 0)
            return -1;
    }
    if (t.kind != T_IDENT) {
        perr(p, t.line, "expected const variable type");
        return -1;
    }
    size_t elem = 0;
    const char *ts = t.text[0] == '.' ? t.text + 1 : t.text;
    if (strcmp(ts, "b8") == 0 || strcmp(ts, "u8") == 0 ||
        strcmp(ts, "s8") == 0) {
        elem = 1;
    } else if (strcmp(ts, "b16") == 0 || strcmp(ts, "u16") == 0 ||
               strcmp(ts, "s16") == 0 || strcmp(ts, "f16") == 0 ||
               strcmp(ts, "bf16") == 0) {
        elem = 2;
    } else if (strcmp(ts, "b32") == 0 || strcmp(ts, "u32") == 0 ||
               strcmp(ts, "s32") == 0 || strcmp(ts, "f32") == 0) {
        elem = 4;
    } else if (strcmp(ts, "b64") == 0 || strcmp(ts, "u64") == 0 ||
               strcmp(ts, "s64") == 0 || strcmp(ts, "f64") == 0) {
        elem = 8;
    } else {
        char m[640];
        snprintf(m, sizeof(m), "unsupported const type '%s'", t.text);
        perr(p, t.line, m);
        return -1;
    }
    int isfloat = ts[0] == 'f' || strcmp(ts, "bf16") == 0;
    if (next(p, &t) != 0)
        return -1;
    if (t.kind != T_IDENT || t.text[0] == '%' || t.text[0] == '.') {
        perr(p, t.line, "expected const variable name");
        return -1;
    }
    char name[512];
    if (copy_name(name, sizeof(name), t.text, p, t.line) != 0)
        return -1;
    uint64_t dim = 1;
    int braced = 0;
    if (at_punct(p, '[')) {
        if (next(p, &t) != 0)
            return -1;
        if (t.kind != T_INT || t.imm == 0 || t.imm > ((uint64_t)1 << 20)) {
            perr(p, t.line, "const dimension must be a nonzero size");
            return -1;
        }
        dim = t.imm;
        if (dim * elem > ((uint64_t)1 << 20)) {
            perr(p, t.line, "const size insane (max 1MB sanity cap)");
            return -1;
        }
        if (expect_punct(p, ']', "in const dimension") != 0)
            return -1;
    }
    // A bare declaration (no `=`) is zero-initialized per the PTX spec
    // (const/global variables default to zero).  CUDA 4.x emits
    // `.const .align 4 .b8 name[N];` for parameter buffers.
    int no_init = 0;
    if (at_punct(p, ';')) {
        // at_punct already consumed the ';'.
        no_init = 1;
    } else {
        if (expect_punct(p, '=', "after const dimension") != 0)
            return -1;
        if (at_punct(p, '{'))
            braced = 1;
    }
    /* Module compiles with runtime-managed symbols must treat uninitialized
     * module-scope .const declarations as device-backed symbols (e.g.
     * __constant__ accessed via cudaMemcpyToSymbol/cuModuleGetGlobal), not
     * as compile-time immediates.  Hidden-symbol injection in parse_entry
     * handles the runtime binding when the symbol is not in unit->consts. */
    if (no_init && getenv("ACUDA_MODULE_SYMBOLS"))
        return 0;
    // Zero-fill matches ptxas: fewer initializer lanes than the
    // declared dimension leave the tail zeroed (ZLUDA const_ident).
    uint8_t *bytes = calloc(dim, elem);
    if (!bytes) {
        perr(p, t.line, "out of memory");
        return -1;
    }
    ptx_sym_t syms[16];
    int nsyms = 0;
    for (uint64_t i = 0; !no_init && i < dim; i++) {
        if (braced && at_punct(p, '}')) {
            braced = 0; // fewer lanes than dim: tail stays zero
            break;
        }
        int neg = 0;
        if (next(p, &t) != 0) {
            free(bytes);
            return -1;
        }
        if (t.kind == T_PUNCT && t.punct == '-') {
            neg = 1;
            if (next(p, &t) != 0) {
                free(bytes);
                return -1;
            }
        }
        uint64_t v = 0;
        if (t.kind == T_FLOAT) {
            if (isfloat && elem == 8) {
                if (t.text[0] == '0' &&
                    (t.text[1] == 'd' || t.text[1] == 'D')) {
                    v = t.imm;
                } else {
                    double d = strtod(t.text, NULL);
                    memcpy(&v, &d, 8);
                }
                if (neg)
                    v ^= 0x8000000000000000ull;
            } else if (isfloat && elem == 4) {
                v = t.imm & 0xffffffffu;
                if (neg)
                    v ^= 0x80000000u;
            } else if (isfloat && elem == 2) {
                if (t.text[0] == '0' &&
                    (t.text[1] == 'h' || t.text[1] == 'H'))
                    v = t.imm & 0xffffu;
                else
                    v = const_f32_to_f16((uint32_t)t.imm);
                if (neg)
                    v ^= 0x8000u;
            } else {
                perr(p, t.line, "float initializer needs a float const type");
                free(bytes);
                return -1;
            }
            neg = 0;
            for (size_t k = 0; k < elem; k++)
                bytes[i * elem + k] = (uint8_t)(v >> (8 * k));
            if (braced && at_punct(p, ','))
                continue;
            break;
        }
        if (t.kind != T_INT) {
            // Symbolic initializers (ZLUDA's const_ident: `{ x, y }`).
            // Consts are inlined at use sites, so a const symbol has no
            // real address; store the same symbolic address token that
            // `mov.u64 %rd, symbol` materializes (see POP_MOV). That
            // keeps address-identity comparisons exact.
            if (t.kind == T_IDENT) {
                int found = -1;
                for (size_t ci = 0; ci < unit->nconsts; ci++) {
                    if (strcmp(unit->consts[ci].name, t.text) == 0) {
                        found = (int)ci;
                        break;
                    }
                }
                if (found >= 0) {
                    t.imm = PTX_CONST_ADDR_BASE |
                            (uint64_t)unit->consts[found].bank_off;
                } else if (elem == 8) {
                    // Possibly a function address (resolved later).
                    if (nsyms < 16) {
                        syms[nsyms].off = (uint32_t)(i * elem);
                        if (copy_name(syms[nsyms].name,
                                      sizeof(syms[nsyms].name), t.text, p,
                                      t.line) != 0) {
                            free(bytes);
                            return -1;
                        }
                        nsyms++;
                    }
                    t.imm = 0;
                } else {
                    t.imm = 0;
                }
                neg = 0;
            } else {
                perr(p, t.line, "const initializer must be an integer literal");
                free(bytes);
                return -1;
            }
        }
        v = t.imm;
        if (neg)
            v = (uint64_t)(-(int64_t)t.imm);
        // Truncate like ptxas (immediates are bit patterns).
        if (elem < 8)
            v &= (elem == 1) ? 0xffull : elem == 2 ? 0xffffull : 0xffffffffull;
        for (size_t k = 0; k < elem; k++)
            bytes[i * elem + k] = (uint8_t)(v >> (8 * k));
        if (braced && at_punct(p, ','))
            continue;
        break;
    }
    if (braced && expect_punct(p, '}', "in const initializer") != 0) {
        free(bytes);
        return -1;
    }
    if (!no_init && expect_punct(p, ';', "after const declaration") != 0) {
        free(bytes);
        return -1;
    }
    for (size_t i = 0; i < unit->nconsts; i++) {
        if (strcmp(unit->consts[i].name, name) == 0) {
            perr(p, t.line, "duplicate const variable");
            free(bytes);
            return -1;
        }
    }
    ptx_const_t *nc =
        realloc(unit->consts, sizeof(*nc) * (unit->nconsts + 1));
    if (!nc) {
        perr(p, t.line, "out of memory");
        free(bytes);
        return -1;
    }
    unit->consts = nc;
    while (unit->const_bank_bytes % elem != 0)
        unit->const_bank_bytes++;
    size_t bank_off = unit->const_bank_bytes;
    unit->const_bank_bytes += dim * elem;
    strcpy(unit->consts[unit->nconsts].name, name);
    unit->consts[unit->nconsts].elem = elem;
    unit->consts[unit->nconsts].nbytes = dim * elem;
    unit->consts[unit->nconsts].bank_off = bank_off;
    unit->consts[unit->nconsts].bytes = bytes;
    unit->consts[unit->nconsts].nsyms = nsyms;
    for (int si = 0; si < nsyms; si++)
        unit->consts[unit->nconsts].syms[si] = syms[si];
    unit->nconsts++;
    return 0;
}

// Module-scope initialized `.global` array (ZLUDA global_array
// corpus): `.global [.align N] .type name[dim] [= {lanes}]`. Served
// as a read-only image (loads materialize from the init bytes;
// stores stay clean errors — no device backing yet). Fewer lanes
// than dim zero-fills the tail (ZLUDA vector-proven); float lanes
// take integer or hex-float/decimal-float literals.
int parse_global_decl(parser_t *p, ptx_unit_t *unit) {
    ptx_tok_t t;
    if (next(p, &t) != 0)
        return -1;
    if (t.kind == T_IDENT && strcmp(t.text, ".align") == 0) {
        if (next(p, &t) != 0)
            return -1;
        if (t.kind != T_INT || t.imm == 0 || t.imm > 64) {
            perr(p, t.line, "expected sane .align value");
            return -1;
        }
        if (next(p, &t) != 0)
            return -1;
    }
    if (t.kind != T_IDENT) {
        perr(p, t.line, "expected global variable type");
        return -1;
    }
    size_t elem = 0;
    const char *ts = t.text[0] == '.' ? t.text + 1 : t.text;
    if (strcmp(ts, "b8") == 0 || strcmp(ts, "u8") == 0 ||
        strcmp(ts, "s8") == 0) {
        elem = 1;
    } else if (strcmp(ts, "b16") == 0 || strcmp(ts, "u16") == 0 ||
               strcmp(ts, "s16") == 0) {
        elem = 2;
    } else if (strcmp(ts, "b32") == 0 || strcmp(ts, "u32") == 0 ||
               strcmp(ts, "s32") == 0 || strcmp(ts, "f32") == 0) {
        elem = 4;
    } else if (strcmp(ts, "b64") == 0 || strcmp(ts, "u64") == 0 ||
               strcmp(ts, "s64") == 0) {
        elem = 8;
    } else {
        char m[640];
        snprintf(m, sizeof(m), "unsupported global type '%s'", t.text);
        perr(p, t.line, m);
        return -1;
    }
    if (next(p, &t) != 0)
        return -1;
    if (t.kind != T_IDENT || t.text[0] == '%' || t.text[0] == '.') {
        perr(p, t.line, "expected global variable name");
        return -1;
    }
    char name[512];
    if (copy_name(name, sizeof(name), t.text, p, t.line) != 0)
        return -1;
    uint64_t dim = 1;
    if (at_punct(p, '[')) {
        if (next(p, &t) != 0)
            return -1;
        if (t.kind != T_INT || t.imm == 0 || t.imm > ((uint64_t)1 << 20)) {
            perr(p, t.line, "global dimension must be a nonzero size");
            return -1;
        }
        dim = t.imm;
        if (dim * elem > ((uint64_t)1 << 20)) {
            perr(p, t.line, "global size insane (max 1MB sanity cap)");
            return -1;
        }
        if (expect_punct(p, ']', "in global dimension") != 0)
            return -1;
    }
    uint8_t *bytes = calloc(dim * elem, 1);
    if (!bytes) {
        perr(p, t.line, "out of memory");
        return -1;
    }
    ptx_sym_t syms[16];
    int nsyms = 0;
    if (at_punct(p, '=')) {
        if (expect_punct(p, '{', "in global initializer") != 0) {
            free(bytes);
            return -1;
        }
        uint64_t i = 0;
        if (!at_punct(p, '}')) {
            for (;;) {
                if (i >= dim) {
                    perr(p, t.line, "too many global initializer lanes");
                    free(bytes);
                    return -1;
                }
                int neg = 0;
                if (at_punct(p, '-'))
                    neg = 1;
                if (next(p, &t) != 0) {
                    free(bytes);
                    return -1;
                }
                uint64_t v = 0;
                if (t.kind == T_INT) {
                    v = neg ? (uint64_t)(-(int64_t)t.imm) : t.imm;
                } else if (t.kind == T_FLOAT && elem == 4) {
                    float f;
                    memcpy(&f, &t.imm, 4);
                    if (neg)
                        f = -f;
                    memcpy(&v, &f, 4);
                } else if (t.kind == T_IDENT && elem == 8 && !neg) {
                    // Function address lane: resolved after the whole
                    // module is parsed (functions may follow).
                    nsyms++;
                    if (nsyms > 16) {
                        perr(p, t.line, "too many symbolic global lanes");
                        free(bytes);
                        return -1;
                    }
                    syms[nsyms - 1].off = (uint32_t)(i * elem);
                    if (copy_name(syms[nsyms - 1].name,
                                  sizeof(syms[nsyms - 1].name), t.text, p,
                                  t.line) != 0) {
                        free(bytes);
                        return -1;
                    }
                } else {
                    perr(p, t.line, "global initializer must be an integer, float, or function symbol");
                    free(bytes);
                    return -1;
                }
                // Truncate like ptxas (immediates are bit patterns).
                if (elem < 8)
                    v &= (elem == 1) ? 0xffull : elem == 2 ? 0xffffull : 0xffffffffull;
                for (size_t k = 0; k < elem; k++)
                    bytes[i * elem + k] = (uint8_t)(v >> (8 * k));
                i++;
                if (at_punct(p, ',')) {
                    continue;
                }
                break;
            }
            if (expect_punct(p, '}', "in global initializer") != 0) {
                free(bytes);
                return -1;
            }
        }
        if (expect_punct(p, ';', "after global declaration") != 0) {
            free(bytes);
            return -1;
        }
    } else {
        // Uninitialized globals read as zero (PTX zero-initializes).
        if (expect_punct(p, ';', "after global declaration") != 0) {
            free(bytes);
            return -1;
        }
    }
    for (size_t i = 0; i < unit->nglobals; i++) {
        if (strcmp(unit->globals[i].name, name) == 0) {
            perr(p, t.line, "duplicate global variable");
            free(bytes);
            return -1;
        }
    }
    for (size_t i = 0; i < unit->nconsts; i++) {
        if (strcmp(unit->consts[i].name, name) == 0) {
            perr(p, t.line, "global variable shadows .const variable");
            free(bytes);
            return -1;
        }
    }
    ptx_const_t *ng =
        realloc(unit->globals, sizeof(*ng) * (unit->nglobals + 1));
    if (!ng) {
        perr(p, t.line, "out of memory");
        free(bytes);
        return -1;
    }
    unit->globals = ng;
    strcpy(unit->globals[unit->nglobals].name, name);
    unit->globals[unit->nglobals].elem = elem;
    unit->globals[unit->nglobals].nbytes = dim * elem;
    unit->globals[unit->nglobals].bytes = bytes;
    unit->globals[unit->nglobals].nsyms = nsyms;
    for (int si = 0; si < nsyms; si++)
        unit->globals[unit->nglobals].syms[si] = syms[si];
    unit->nglobals++;
    return 0;
}


// Module-scope `.texref [.type] name;` declaration.  The legacy CUDA 4.x
// PhysX path resolves these through cuModuleGetTexRef and binds them to a
// texture object at launch time.  Parsing is permissive; only the name is
// retained here.
int parse_texref_decl(parser_t *p, ptx_unit_t *unit) {
    ptx_tok_t t;
    char name[256] = { 0 };
    for (;;) {
        if (next(p, &t) != 0)
            return -1;
        if (t.kind == T_PUNCT && t.punct == ';')
            break;
        if (t.kind == T_EOF) {
            perr(p, t.line, "unterminated .texref declaration");
            return -1;
        }
        if (t.kind == T_IDENT && t.text[0] != '.') {
            if (copy_name(name, sizeof(name), t.text, p, t.line) != 0)
                return -1;
        }
    }
    if (name[0] == '\0') {
        perr(p, t.line, "expected .texref name");
        return -1;
    }
    for (size_t i = 0; i < unit->ntexrefs; i++) {
        if (strcmp(unit->texrefs[i].name, name) == 0) {
            perr(p, t.line, "duplicate .texref");
            return -1;
        }
    }
    ptx_texref_t *nt = realloc(unit->texrefs,
                               sizeof(*nt) * (unit->ntexrefs + 1));
    if (!nt) {
        perr(p, t.line, "out of memory");
        return -1;
    }
    unit->texrefs = nt;
    strcpy(unit->texrefs[unit->ntexrefs].name, name);
    unit->ntexrefs++;
    return 0;
}

// ---- Top-level `.extern .func` signature ----
//
// Only the declaration shape used by CUDA's device runtime is needed:
//
//   .extern .func (.param .align 16 .b8 ret[16]) name
//   ( .param .b64 a, .param .b32 b );
//
// The body lives in another module (or is serviced as a host callback).
// Anything stranger stays a clean parse error rather than a silent guess.

static size_t ext_ty_bytes(ptx_ty_t ty) {
    switch (ty) {
    case PTY_U8: return 1;
    case PTY_U16: case PTY_F16: case PTY_BF16: return 2;
    case PTY_U64: case PTY_F64: return 8;
    default: return 4;
    }
}

static int parse_ext_param(parser_t *p, ptx_extfunc_t *ef, int ret,
                           char *name, size_t namesz, ptx_ty_t *ty,
                           size_t *bytes, int *align) {
    ptx_tok_t t;
    if (next(p, &t) != 0)
        return -1;
    ptx_ty_t sty = PTY_U32;
    int al = 0;
    if (strncmp(t.text, ".param.", 7) == 0) {
        if (map_ty(t.text + 7, &sty) != 0 || sty == PTY_PRED) {
            perr(p, t.line, "bad extern parameter type");
            return -1;
        }
    } else if (strcmp(t.text, ".param") == 0) {
        if (next(p, &t) != 0)
            return -1;
        if (t.kind == T_IDENT && strcmp(t.text, ".align") == 0) {
            if (next(p, &t) != 0)
                return -1;
            if (t.kind != T_INT || t.imm == 0 || t.imm > 64) {
                perr(p, t.line, "expected sane extern .align value");
                return -1;
            }
            al = (int)t.imm;
            if (next(p, &t) != 0)
                return -1;
        }
        if (t.kind != T_IDENT || map_ty(t.text, &sty) != 0 ||
            sty == PTY_PRED) {
            perr(p, t.line, "bad extern parameter type");
            return -1;
        }
    } else if (strcmp(t.text, ".reg") == 0 ||
               strncmp(t.text, ".reg.", 5) == 0) {
        perr(p, t.line, "extern .func with .reg parameters unsupported");
        return -1;
    } else {
        perr(p, t.line, "expected .param in extern signature");
        return -1;
    }
    if (next(p, &t) != 0)
        return -1;
    if (t.kind != T_IDENT) {
        perr(p, t.line, "expected extern parameter name");
        return -1;
    }
    if (copy_name(name, namesz, t.text, p, t.line) != 0)
        return -1;
    size_t nb = ext_ty_bytes(sty);
    if (at_punct(p, '[')) {
        if (next(p, &t) != 0)
            return -1;
        if (t.kind != T_INT || t.imm == 0 || t.imm > 4096) {
            perr(p, t.line, "bad extern parameter array size");
            return -1;
        }
        nb *= (size_t)t.imm;
        if (expect_punct(p, ']', "in extern parameter array") != 0)
            return -1;
    }
    *ty = sty;
    *bytes = nb;
    *align = al > 0 ? al : (int)nb;
    if (ret && ef->ret_bytes < nb)
        ef->ret_bytes = nb;
    return 0;
}

int parse_extern_func(parser_t *p, ptx_unit_t *unit) {
    // `.extern` and `.func` already consumed.
    ptx_extfunc_t ef;
    memset(&ef, 0, sizeof(ef));
    ptx_tok_t t;
    if (at_punct(p, '(')) {
        if (!at_punct(p, ')')) {
            for (;;) {
                char nm[512];
                ptx_ty_t ty;
                size_t bytes;
                int align;
                if (parse_ext_param(p, &ef, 1, nm, sizeof(nm), &ty, &bytes,
                                    &align) != 0)
                    return -1;
                if (at_punct(p, ','))
                    continue;
                break;
            }
        }
        if (expect_punct(p, ')', "after extern return parameters") != 0)
            return -1;
    }
    if (next(p, &t) != 0)
        return -1;
    if (t.kind != T_IDENT) {
        perr(p, t.line, "expected extern function name");
        return -1;
    }
    if (copy_name(ef.name, sizeof(ef.name), t.text, p, t.line) != 0)
        return -1;
    if (at_punct(p, '(')) {
        if (!at_punct(p, ')')) {
            for (;;) {
                char nm[512];
                ptx_ty_t ty;
                size_t bytes;
                int align;
                if (parse_ext_param(p, &ef, 0, nm, sizeof(nm), &ty, &bytes,
                                    &align) != 0)
                    return -1;
                if (ef.nargs >= 32) {
                    perr(p, t.line, "too many extern parameters");
                    return -1;
                }
                // Align this argument inside the request argument area.
                size_t a = (size_t)(align > 8 ? align : 8);
                size_t off = (ef.arg_bytes + a - 1) & ~(a - 1);
                ef.arg_ty[ef.nargs] = ty;
                ef.arg_off[ef.nargs] = off;
                ef.nargs++;
                ef.arg_bytes = off + bytes;
                if (at_punct(p, ','))
                    continue;
                break;
            }
        }
        if (expect_punct(p, ')', "after extern parameters") != 0)
            return -1;
    }
    if (expect_punct(p, ';', "after extern function") != 0)
        return -1;
    // Duplicate declarations (CUDA emits them) are harmless.
    for (size_t i = 0; i < unit->nextfuncs; i++) {
        if (strcmp(unit->extfuncs[i].name, ef.name) == 0)
            return 0;
    }
    if (unit->nextfuncs == unit->extfuncs_cap) {
        size_t ncap = unit->extfuncs_cap ? unit->extfuncs_cap * 2 : 4;
        ptx_extfunc_t *ne = realloc(unit->extfuncs, sizeof(*ne) * ncap);
        if (!ne) {
            perr(p, t.line, "out of memory");
            return -1;
        }
        unit->extfuncs = ne;
        unit->extfuncs_cap = ncap;
    }
    unit->extfuncs[unit->nextfuncs++] = ef;
    return 0;
}
