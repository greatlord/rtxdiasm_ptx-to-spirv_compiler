// PTX subset -> SPIR-V (Logical, GLSL450) emitter.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if !defined(_MSC_VER)
#include <strings.h>
#endif
#include "ptx.h"
#include "acuda/compiler.h"
#include "spirv_internal.h"

void spv_blob_free(spv_blob_t *blob) {
    free(blob->words);
    memset(blob, 0, sizeof(*blob));
}









void fail(ctx_t *c, const char *msg) {
    if (!c->failed) {
        snprintf(c->err, c->errlen, "%s", msg);
        c->failed = 1;
    }
}

void failf(ctx_t *c, int line, const char *msg) {
    if (!c->failed) {
        snprintf(c->err, c->errlen, "line %d: %s", line, msg);
        c->failed = 1;
    }
}

void put(ctx_t *c, spv_blob_t *b, uint32_t w) {
    if (c->failed)
        return;
    if (b->count + 1 > b->cap) {
        size_t ncap = b->cap ? b->cap * 2 : 256;
        uint32_t *nw = malloc(sizeof(*nw) * ncap);
        if (!nw) {
            fail(c, "out of memory");
            return;
        }
        memcpy(nw, b->words, sizeof(*nw) * b->count);
        free(b->words);
        b->words = nw;
        b->cap = ncap;
    }
    b->words[b->count++] = w;
    // A new basic block invalidates the within-block conversion CSE.
    // (A data word can theoretically match this pattern; that only
    // clears the cache early, which is harmless.)
    if (b == &c->fn && (w & 0xffffu) == S_Label)
        c->cse_gen++;
}

uint32_t nid(ctx_t *c) { return c->next_id++; }

// Whether the module needs the PhysicalStorageBuffer address space:
// explicit BDA mode, or 32-bit code where plain integers routinely
// serve as device addresses (scalar-BDA loads lower them through
// OpConvertUToPtr).  The types, capability, and addressing model must
// all agree before any conversion is emitted.
int need_phys_addr(ctx_t *c) {
    return c->use_bda || (c->caps.bda && c->addr_size == 32);
}

void emit1(ctx_t *c, spv_blob_t *b, uint32_t op, uint32_t w0) {
    put(c, b, (2u << 16) | op);
    put(c, b, w0);
}

void emit_str(ctx_t *c, spv_blob_t *b, const char *s) {
    size_t n = strlen(s) + 1;
    size_t words = (n + 3) / 4;
    for (size_t i = 0; i < words; i++) {
        uint32_t w = 0;
        for (int k = 0; k < 4; k++) {
            size_t idx = i * 4 + (size_t)k;
            uint32_t ch = idx < n ? (uint32_t)(unsigned char)s[idx] : 0;
            w |= ch << (8 * k);
        }
        put(c, b, w);
    }
}

// Effective type under emulation: f64 means f32 when ACUDA_EMU_F64
// demotion is on (documented approximation, explicit opt-in only).
ptx_ty_t eff_ty(ctx_t *c, ptx_ty_t ty) {
    if (c->emu_f64 && ty == PTY_F64)
        return PTY_F32;
    return ty;
}

uint32_t ty_of(ctx_t *c, ptx_ty_t ty) {
    switch (eff_ty(c, ty)) {
    case PTY_U32:
    case PTY_U8:
    case PTY_U16:
    case PTY_BF16:
    case PTY_F16X2:
    case PTY_BF16X2:
        return c->t_u32;
    case PTY_U64: return c->t_u64;
    case PTY_F32:
    case PTY_F16:
        return c->t_f32;
    case PTY_F64: return c->t_f64;
    case PTY_PRED: return c->t_bool;
    }
    return 0;
}

// f32 bits -> f16 bits, round-to-nearest-even. Inf/NaN exact;
// f16-subnormal results produced exactly; f32-denormal inputs flush
// to signed zero (documented limitation, same as f64 emulation).
uint32_t f32_to_f16_bits(uint32_t w) {
    uint32_t s = w >> 31, e = (w >> 23) & 0xffu, m = w & 0x7fffffu;
    if (e == 0xffu) {
        uint32_t mo = m ? ((m >> 13) | 0x200u) : 0;
        return (s << 15) | (0x1fu << 10) | mo;
    }
    if (e == 0)
        return s << 15; // f32-denormal flush (documented)
    int32_t exp = (int32_t)e - 127;
    if (exp >= 16)
        return (s << 15) | (0x1fu << 10); // overflow -> Inf
    if (exp >= -14) {
        uint32_t mant = m >> 13, drop = m & 0x1fffu;
        if (drop > 0x1000u || (drop == 0x1000u && (mant & 1u))) {
            mant++;
            if (mant == 0x400u) {
                mant = 0;
                exp++;
                if (exp >= 16)
                    return (s << 15) | (0x1fu << 10);
            }
        }
        return (s << 15) | (((uint32_t)(exp + 15)) << 10) | mant;
    }
    // f16-subnormal range: full 24-bit mantissa, shifted + rounded.
    uint32_t full = m | 0x800000u;
    int32_t rsh = -exp - 1;
    if (rsh >= 25)
        return s << 15; // too small even for a denormal
    uint32_t mant = full >> rsh;
    uint32_t drop = full & ~(0xffffffffu << rsh);
    uint32_t half = 1u << (rsh - 1);
    if (drop > half || (drop == half && (mant & 1u)))
        mant++;
    if (mant >= 1024u) // rounded up to the smallest normal
        return (s << 15) | (1u << 10);
    return (s << 15) | mant;
}

uint32_t const_u32(ctx_t *c, uint32_t v) {
    for (int i = 0; i < c->nu32; i++)
        if (c->cu32[i].bits == v)
            return c->cu32[i].id;
    if (c->nu32 >= (int)(sizeof(c->cu32) / sizeof(c->cu32[0]))) {
        fail(c, "too many u32 constants");
        return 0;
    }
    uint32_t id = nid(c);
    put(c, &c->types, (4u << 16) | S_Constant);
    put(c, &c->types, c->t_u32);
    put(c, &c->types, id);
    put(c, &c->types, v);
    c->cu32[c->nu32].bits = v;
    c->cu32[c->nu32].id = id;
    c->nu32++;
    return id;
}

// Intern a non-aggregate type declaration (OpTypeImage,
// OpTypeSampledImage, OpTypePointer).  Identical spellings must share
// one id: the validator rejects duplicates, and texture bindings with
// the same geometry/format would otherwise each mint their own.
uint32_t type_intern(ctx_t *c, uint32_t op, const uint32_t *ops,
                            uint32_t nops) {
    if (nops > 7)
        return 0;
    uint32_t w[7] = {0, 0, 0, 0, 0, 0, 0};
    for (uint32_t i = 0; i < nops; i++)
        w[i] = ops[i];
    for (int i = 0; i < c->ntypekeys; i++) {
        typent_t *t = &c->typekeys[i];
        if (t->op == op && t->w0 == w[0] && t->w1 == w[1] &&
            t->w2 == w[2] && t->w3 == w[3] && t->w4 == w[4] &&
            t->w5 == w[5] && t->w6 == w[6])
            return t->id;
    }
    uint32_t id = nid(c);
    put(c, &c->types, ((nops + 2u) << 16) | op);
    put(c, &c->types, id);
    for (uint32_t i = 0; i < nops; i++)
        put(c, &c->types, ops[i]);
    if (c->ntypekeys < ACUDA_MAX_TYPEKEYS) {
        typent_t *t = &c->typekeys[c->ntypekeys++];
        t->op = op;
        t->w0 = w[0];
        t->w1 = w[1];
        t->w2 = w[2];
        t->w3 = w[3];
        t->w4 = w[4];
        t->w5 = w[5];
        t->w6 = w[6];
        t->id = id;
    }
    return id;
}

uint32_t const_u64(ctx_t *c, uint64_t v) {
    for (int i = 0; i < c->nu64; i++)
        if (c->cu64[i].bits == v)
            return c->cu64[i].id;
    if (c->nu64 >= (int)(sizeof(c->cu64) / sizeof(c->cu64[0]))) {
        fail(c, "too many u64 constants");
        return 0;
    }
    uint32_t id = nid(c);
    put(c, &c->types, (5u << 16) | S_Constant);
    put(c, &c->types, c->t_u64);
    put(c, &c->types, id);
    put(c, &c->types, (uint32_t)(v & 0xffffffffu));
    put(c, &c->types, (uint32_t)(v >> 32));
    c->cu64[c->nu64].bits = v;
    c->cu64[c->nu64].id = id;
    c->nu64++;
    return id;
}

uint32_t const_f32(ctx_t *c, uint32_t bits) {
    for (int i = 0; i < c->nf32; i++)
        if (c->cf32[i].bits == bits)
            return c->cf32[i].id;
    if (c->nf32 >= (int)(sizeof(c->cf32) / sizeof(c->cf32[0]))) {
        fail(c, "too many f32 constants");
        return 0;
    }
    uint32_t id = nid(c);
    put(c, &c->types, (4u << 16) | S_Constant);
    put(c, &c->types, c->t_f32);
    put(c, &c->types, id);
    put(c, &c->types, bits);
    c->cf32[c->nf32].bits = bits;
    c->cf32[c->nf32].id = id;
    c->nf32++;
    return id;
}

uint32_t const_f64(ctx_t *c, uint64_t bits) {
    for (int i = 0; i < c->nf64; i++)
        if (c->cf64[i].bits == bits)
            return c->cf64[i].id;
    if (c->nf64 >= (int)(sizeof(c->cf64) / sizeof(c->cf64[0]))) {
        fail(c, "too many f64 constants");
        return 0;
    }
    uint32_t id = nid(c);
    put(c, &c->types, (5u << 16) | S_Constant);
    put(c, &c->types, c->t_f64);
    put(c, &c->types, id);
    put(c, &c->types, (uint32_t)(bits & 0xffffffffu));
    put(c, &c->types, (uint32_t)(bits >> 32));
    c->cf64[c->nf64].bits = bits;
    c->cf64[c->nf64].id = id;
    c->nf64++;
    return id;
}

reg_t *find_reg(ctx_t *c, const char *name) {
    for (int i = 0; i < c->nregs; i++)
        if (strcmp(c->regs[i].name, name) == 0)
            return &c->regs[i];
    if (name && name[0] == '_' && strstr(name, "sink")) {
        if (c->nregs < MAX_REGS) {
            reg_t *r = &c->regs[c->nregs++];
            memset(r, 0, sizeof(*r));
            snprintf(r->name, sizeof(r->name), "%s", name);
            r->decl_ty = PTY_U32;
            r->cur_ty = PTY_U32;
            r->defined = 1;
            r->cur = 0;
            r->is_ptr = 0;
            r->binding = -1;
            r->shvar = -1;
            r->glvar = -1;
            r->paddr = -1;
            return r;
        }
    }
    return NULL;
}

// Integer register width in bits (8/16/32/64), or -1 for non-integer
// declarations. Used for narrow-load destination leniency.
int decl_int_width(ptx_ty_t ty) {
    if (ty == PTY_U8)
        return 8;
    if (ty == PTY_U16)
        return 16;
    if (ty == PTY_U32)
        return 32;
    if (ty == PTY_U64)
        return 64;
    return -1;
}

cparam_t *find_cparam(ctx_t *c, const char *name) {
    for (int i = 0; i < c->ncparams; i++)
        if (strcmp(c->cparams[i].name, name) == 0)
            return &c->cparams[i];
    return NULL;
}

int find_shvar(ctx_t *c, const char *name) {
    for (int i = 0; i < c->nshvars; i++)
        if (strcmp(c->shvars[i].name, name) == 0)
            return i;
    return -1;
}

int find_local(ctx_t *c, const char *name) {
    for (int i = 0; i < c->nlocals; i++)
        if (strcmp(c->locals[i].name, name) == 0)
            return i;
    return -1;
}

int find_const(ctx_t *c, const char *name) {
    for (int i = 0; i < c->nconsts; i++)
        if (strcmp(c->consts[i].name, name) == 0)
            return i;
    return -1;
}

int find_global(ctx_t *c, const char *name) {
    for (int i = 0; i < c->nglobals; i++)
        if (strcmp(c->globals[i].name, name) == 0)
            return i;
    return -1;
}

int block_index_of(ctx_t *c, const ptx_func_t *fn, const char *label) {
    (void)c;
    for (size_t i = 0; i < fn->nblocks; i++)
        if (fn->blocks[i].label[0] && strcmp(fn->blocks[i].label, label) == 0)
            return (int)i;
    return -1;
}

void emit_decorate(ctx_t *c, uint32_t target, uint32_t deco) {
    spv_blob_t *b = &c->ann;
    put(c, b, (3u << 16) | S_Decorate);
    put(c, b, target);
    put(c, b, deco);
}

void emit_decorate1(ctx_t *c, uint32_t target, uint32_t deco, uint32_t lit) {
    spv_blob_t *b = &c->ann;
    put(c, b, (4u << 16) | S_Decorate);
    put(c, b, target);
    put(c, b, deco);
    put(c, b, lit);
}

void emit_member_decorate(ctx_t *c, uint32_t target, uint32_t member,
                                 uint32_t deco, uint32_t lit) {
    spv_blob_t *b = &c->ann;
    put(c, b, (5u << 16) | S_MemberDecorate);
    put(c, b, target);
    put(c, b, member);
    put(c, b, deco);
    put(c, b, lit);
}

