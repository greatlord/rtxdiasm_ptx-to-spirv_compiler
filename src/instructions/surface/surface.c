// Surface instructions: suld, sust, sured and suq.
//
// A .surfref parameter binds a Vulkan storage image (OpTypeImage with
// Sampled=2).  Byte-addressed .b forms treat the surface as 32-bit
// words (the runtime's surface formats are 4 bytes per texel); .p forms
// write formatted samples.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "spirv_internal.h"

static int surf_dim(int g) {
    switch (g) {
    case 1: case 4: return 0;
    case 2: case 5: return 1;
    case 3: return 2;
    }
    return 1;
}

static int surf_arrayed(int g) {
    return g == 4 || g == 5;
}

static uint32_t surf_vec_ty(ctx_t *c, int n) {
    if (n <= 1) return c->t_u32;
    if (n == 2) return c->t_v2u32;
    if (n == 3) return c->t_v3u32;
    return c->t_uvec4;
}

static uint32_t surf_scalar_as(ctx_t *c, spv_blob_t *b, const char *name,
                               ptx_ty_t ty, int line) {
    ptx_opd_t o;
    memset(&o, 0, sizeof(o));
    o.kind = OPD_REG;
    snprintf(o.reg, sizeof(o.reg), "%s", name);
    val_t v = resolve(c, b, &o, ty, line);
    return v.id;
}

static uint32_t surf_scalar(ctx_t *c, spv_blob_t *b, const char *name,
                            int line) {
    return surf_scalar_as(c, b, name, PTY_U32, line);
}

static uint32_t surf_make(ctx_t *c, spv_blob_t *b, uint32_t ty,
                          const uint32_t *comp, int n) {
    if (n <= 1)
        return comp[0];
    uint32_t id = nid(c);
    put(c, b, ((uint32_t)(n + 3) << 16) | S_CompositeConstruct);
    put(c, b, ty);
    put(c, b, id);
    for (int i = 0; i < n; i++)
        put(c, b, comp[i]);
    return id;
}

typedef struct {
    int dynamic;
    int binding;
    uint32_t img_ptr;
    uint32_t img;
    uint32_t slot;
    int u64; // R64ui image type (SPV_EXT_shader_image_int64)
    int cas;  // R32ui alias + integer CAS fallback
    int lock; // R32f image protected by the global surface lock
} surf_ref_t;

static int surf_slot_index(ctx_t *c, spv_blob_t *b, const char *name,
                           int line) {
    reg_t *r = find_reg(c, name);
    if (r && r->defined && !r->is_ptr) {
        ptx_opd_t o;
        memset(&o, 0, sizeof(o));
        o.kind = OPD_REG;
        snprintf(o.reg, sizeof(o.reg), "%s", name);
        val_t v = resolve(c, b, &o, PTY_U64, line);
        if (c->failed)
            return 0;
        uint32_t id = nid(c);
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, c->t_u32);
        put(c, b, id);
        put(c, b, v.id);
        return id;
    }
    cparam_t *cp = find_cparam(c, name);
    if (cp && cp->is_tex == 3 && cp->kind == 2 && cp->push_word >= 0) {
        uint32_t id = 0;
        emit_push_load(c, b, &id, PTY_U32, cp->push_word, line);
        return id;
    }
    failf(c, line, "dynamic surface handle must be a .u64 register");
    return 0;
}

static int surf_ref_resolve(ctx_t *c, spv_blob_t *b,
                            const ptx_instr_t *in, const char *name,
                            int line, surf_ref_t *sr) {
    memset(sr, 0, sizeof(*sr));
    sr->binding = -1;
    if (surf_ref_is_static(c, name, 0)) {
        int is_tex = 0;
        uint32_t off = 0;
        if (tex_ref_binding(c, name, &sr->binding, &off, &is_tex, line) != 0)
            return -1;
        (void)off;
        if (sr->binding < 0 || !c->tex_vars[sr->binding] || is_tex != 3) {
            failf(c, line, "surface operand must be a .surfref parameter");
            return -1;
        }
        sr->img_ptr = c->tex_vars[sr->binding];
        sr->u64 = c->surf_u64[sr->binding];
        sr->cas = c->surf_f32_cas[sr->binding];
        sr->lock = c->surf_f32_lock[sr->binding];
        sr->img = nid(c);
        put(c, b, (4u << 16) | S_Load);
        put(c, b, c->tex_image_ty[sr->binding]);
        put(c, b, sr->img);
        put(c, b, sr->img_ptr);
        return 0;
    }
    if (!c->surf_table) {
        failf(c, line, "dynamic surface table is unavailable");
        return -1;
    }
    sr->dynamic = 1;
    sr->slot = surf_slot_index(c, b, name, line);
    if (c->failed)
        return -1;
    put(c, &c->ann, (3u << 16) | S_Decorate);
    put(c, &c->ann, sr->slot);
    put(c, &c->ann, D_NonUniform);
    if (in->op == POP_SUQ)
        return 0; // metadata only
    int dim = surf_dim(in->surf_geom);
    int arrayed = surf_arrayed(in->surf_geom);
    int isint = in->surf_formatted ? 0 : 1;
    int want_cas = 0, want_lock = 0;
    if (in->op == POP_SURED && in->surf_ty == PTY_F32) {
        // Dynamic surface handles index the image at runtime, so use
        // the load/CAS (or lock) table even when the device has
        // hardware image atomics.
        int f32_fallback =
            (in->surf_redop == 1 || in->surf_redop == 2) &&
            (sr->dynamic ||
             !(c->caps.atomic_float & ACUDA_ATOMIC_IMAGE_F32_MINMAX));
        int f32_add_fallback =
            in->surf_redop == 0 &&
            (sr->dynamic ||
             !(c->caps.atomic_float & ACUDA_ATOMIC_IMAGE_F32_ADD));
        if (f32_fallback || f32_add_fallback) {
            if (c->caps.surf_alias_ok)
                want_cas = 1;
            else
                want_lock = 1;
        }
    }
    surf_table_t *t = NULL;
    for (int pass = 0; pass < 2 && !t; pass++) {
        for (int i = 0; i < c->surf_table_count; i++) {
            surf_table_t *q = &c->surf_tables[i];
            if (q->dim != dim || q->arrayed != arrayed ||
                q->isint != isint)
                continue;
            if (pass == 0 &&
                (q->cas != want_cas || q->lock != want_lock))
                continue;
            t = q;
            break;
        }
    }
    if (!t) {
        failf(c, line, "no dynamic surface table for this geometry/format");
        return -1;
    }
    sr->cas = t->cas;
    sr->lock = t->lock;
    sr->img_ptr = nid(c);
    put(c, b, (5u << 16) | S_AccessChain);
    put(c, b, t->elem_ptr_ty);
    put(c, b, sr->img_ptr);
    put(c, b, t->var);
    put(c, b, sr->slot);
    sr->img = nid(c);
    put(c, b, (4u << 16) | S_Load);
    put(c, b, t->img_ty);
    put(c, b, sr->img);
    put(c, b, sr->img_ptr);
    return 0;
}

// Integer coordinate vector.  For byte-addressed .b forms the lowest
// coordinate is a byte offset and is divided by four; `word_off` steps
// to the next 32-bit word.
static uint32_t surf_coord(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in,
                           int dim, int arrayed, int bytes, int word_off,
                           int line) {
    int first = arrayed ? 1 : 0;
    uint32_t comp[4];
    int n = 0;
    uint32_t x = surf_scalar(c, b, in->surf_coord[first], line);
    if (bytes) {
        uint32_t t = nid(c);
        emit_arith(c, b, S_ShrL, c->t_u32, t, x, const_u32(c, 2));
        if (word_off) {
            x = nid(c);
            emit_arith(c, b, S_IAdd, c->t_u32, x, t,
                       const_u32(c, (uint32_t)word_off));
        } else {
            x = t;
        }
    }
    comp[n++] = x;
    if (dim != 0)
        comp[n++] = surf_scalar(c, b, in->surf_coord[first + 1], line);
    if (dim == 2)
        comp[n++] = surf_scalar(c, b, in->surf_coord[first + 2], line);
    if (arrayed)
        comp[n++] = surf_scalar(c, b, in->surf_coord[0], line);
    return surf_make(c, b, surf_vec_ty(c, n), comp, n);
}

// 64-bit element coordinate: PTX .b coordinates are byte offsets, so
// shift by 3 (8-byte texels) and add the vector element index.
static uint32_t surf_coord64(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in,
                             int dim, int arrayed, int elem_off, int line) {
    int first = arrayed ? 1 : 0;
    uint32_t comp[4];
    int n = 0;
    uint32_t x = surf_scalar(c, b, in->surf_coord[first], line);
    uint32_t t = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, t, x, const_u32(c, 3));
    if (elem_off) {
        x = nid(c);
        emit_arith(c, b, S_IAdd, c->t_u32, x, t,
                   const_u32(c, (uint32_t)elem_off));
    } else {
        x = t;
    }
    comp[n++] = x;
    if (dim != 0)
        comp[n++] = surf_scalar(c, b, in->surf_coord[first + 1], line);
    if (dim == 2)
        comp[n++] = surf_scalar(c, b, in->surf_coord[first + 2], line);
    if (arrayed)
        comp[n++] = surf_scalar(c, b, in->surf_coord[0], line);
    (void)line;
    return surf_make(c, b, surf_vec_ty(c, n), comp, n);
}

static uint32_t surf_v4u64(ctx_t *c) {
    if (!c->t_v4u64) {
        c->t_v4u64 = nid(c);
        put(c, &c->types, (4u << 16) | S_TypeVector);
        put(c, &c->types, c->t_v4u64);
        put(c, &c->types, c->t_u64);
        put(c, &c->types, 4);
    }
    return c->t_v4u64;
}

static uint32_t surf_read_u64(ctx_t *c, spv_blob_t *b, uint32_t img,
                              uint32_t coord) {
    uint32_t vec = nid(c);
    put(c, b, (5u << 16) | S_ImageRead);
    put(c, b, surf_v4u64(c));
    put(c, b, vec);
    put(c, b, img);
    put(c, b, coord);
    uint32_t r = nid(c);
    put(c, b, (5u << 16) | S_CompositeExtract);
    put(c, b, c->t_u64);
    put(c, b, r);
    put(c, b, vec);
    put(c, b, 0u);
    return r;
}

static void surf_write_u64(ctx_t *c, spv_blob_t *b, uint32_t img,
                           uint32_t coord, uint32_t val) {
    uint32_t vec = nid(c);
    put(c, b, (7u << 16) | S_CompositeConstruct);
    put(c, b, surf_v4u64(c));
    put(c, b, vec);
    put(c, b, val);
    for (int i = 0; i < 3; i++)
        put(c, b, const_u64(c, 0));
    put(c, b, (4u << 16) | S_ImageWrite);
    put(c, b, img);
    put(c, b, coord);
    put(c, b, vec);
}

// One 32-bit word read from the storage image at `coord`.
static uint32_t surf_read_word(ctx_t *c, spv_blob_t *b, uint32_t img,
                               uint32_t coord, int line) {
    uint32_t res = nid(c);
    put(c, b, (5u << 16) | S_ImageRead);
    put(c, b, c->t_uvec4);
    put(c, b, res);
    put(c, b, img);
    put(c, b, coord);
    uint32_t w = nid(c);
    put(c, b, (5u << 16) | S_CompositeExtract);
    put(c, b, c->t_u32);
    put(c, b, w);
    put(c, b, res);
    put(c, b, 0u);
    (void)line;
    return w;
}

static void surf_write_word(ctx_t *c, spv_blob_t *b, uint32_t img,
                            uint32_t coord, uint32_t value) {
    uint32_t vec = nid(c);
    put(c, b, (7u << 16) | S_CompositeConstruct);
    put(c, b, c->t_uvec4);
    put(c, b, vec);
    put(c, b, value);
    put(c, b, const_u32(c, 0));
    put(c, b, const_u32(c, 0));
    put(c, b, const_u32(c, 0));
    put(c, b, (4u << 16) | S_ImageWrite);
    put(c, b, img);
    put(c, b, coord);
    put(c, b, vec);
}

// Map IEEE-754 bits to an unsigned key whose integer ordering matches
// float ordering, and back.  Negative values need the bits inverted;
// positive values need the sign bit set.
static uint32_t surf_float_key(ctx_t *c, spv_blob_t *b, uint32_t bits) {
    uint32_t sign = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, sign, bits, const_u32(c, 31));
    uint32_t neg = nid(c);
    put(c, b, (5u << 16) | S_IEqual);
    put(c, b, c->t_bool);
    put(c, b, neg);
    put(c, b, sign);
    put(c, b, const_u32(c, 1));
    uint32_t mask = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, c->t_u32);
    put(c, b, mask);
    put(c, b, neg);
    put(c, b, const_u32(c, 0xffffffffu));
    put(c, b, const_u32(c, 0x80000000u));
    uint32_t key = nid(c);
    emit_arith(c, b, S_Xor, c->t_u32, key, bits, mask);
    return key;
}

// One f32 texel read/write pair used by the lock fallback.
static uint32_t surf_read_f32(ctx_t *c, spv_blob_t *b, uint32_t img,
                              uint32_t coord) {
    uint32_t vec = nid(c);
    put(c, b, (5u << 16) | S_ImageRead);
    put(c, b, c->t_v4f32);
    put(c, b, vec);
    put(c, b, img);
    put(c, b, coord);
    uint32_t v = nid(c);
    put(c, b, (5u << 16) | S_CompositeExtract);
    put(c, b, c->t_f32);
    put(c, b, v);
    put(c, b, vec);
    put(c, b, 0u);
    return v;
}

static void surf_write_f32(ctx_t *c, spv_blob_t *b, uint32_t img,
                           uint32_t coord, uint32_t value) {
    uint32_t vec = nid(c);
    put(c, b, (7u << 16) | S_CompositeConstruct);
    put(c, b, c->t_v4f32);
    put(c, b, vec);
    put(c, b, value);
    put(c, b, const_f32(c, 0));
    put(c, b, const_f32(c, 0));
    put(c, b, const_f32(c, 0));
    put(c, b, (4u << 16) | S_ImageWrite);
    put(c, b, img);
    put(c, b, coord);
    put(c, b, vec);
}

static uint32_t surf_lock_ptr(ctx_t *c, spv_blob_t *b, uint32_t coord,
                              int ncoord, uint32_t key) {
    uint32_t h = const_u32(c, key * 16777619u + 0x9e3779b9u);
    for (int i = 0; i < ncoord; i++) {
        uint32_t ci = nid(c);
        put(c, b, (5u << 16) | S_CompositeExtract);
        put(c, b, c->t_u32);
        put(c, b, ci);
        put(c, b, coord);
        put(c, b, (uint32_t)i);
        uint32_t t = nid(c);
        emit_arith(c, b, S_IMul, c->t_u32, t, h,
                   const_u32(c, 16777619u));
        uint32_t x = nid(c);
        emit_arith(c, b, S_Xor, c->t_u32, x, t, ci);
        h = x;
    }
    uint32_t idx = nid(c);
    emit_arith(c, b, S_And, c->t_u32, idx, h,
               const_u32(c, ACUDA_SURF_LOCK_SLOTS - 1u));
    uint32_t zero = const_u32(c, 0);
    uint32_t ptr = nid(c);
    put(c, b, (6u << 16) | S_AccessChain);
    put(c, b, c->t_ptr_u_ssbo);
    put(c, b, ptr);
    put(c, b, c->lock_var);
    put(c, b, zero); // struct member 0
    put(c, b, idx);  // hashed per-texel lock slot
    return ptr;
}

static void surf_lock_acquire(ctx_t *c, spv_blob_t *b, uint32_t ptr) {
    uint32_t scope = const_u32(c, 1);
    uint32_t sem_acqrel = const_u32(c, 0x848u);
    uint32_t sem_acq = const_u32(c, 0x842u);
    uint32_t header = nid(c), merge = nid(c), cont = nid(c);
    emit_branch(c, b, header);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, header);
    uint32_t prev = nid(c);
    put(c, b, (9u << 16) | S_AtomicCAS);
    put(c, b, c->t_u32);
    put(c, b, prev);
    put(c, b, ptr);
    put(c, b, scope);
    put(c, b, sem_acqrel);
    put(c, b, sem_acq);
    put(c, b, const_u32(c, 1)); // acquire
    put(c, b, const_u32(c, 0)); // expected unlocked
    uint32_t ok = nid(c);
    put(c, b, (5u << 16) | S_IEqual);
    put(c, b, c->t_bool);
    put(c, b, ok);
    put(c, b, prev);
    put(c, b, const_u32(c, 0));
    put(c, b, (4u << 16) | S_LoopMerge);
    put(c, b, merge);
    put(c, b, cont);
    put(c, b, 0);
    put(c, b, (4u << 16) | S_BranchCond);
    put(c, b, ok);
    put(c, b, merge);
    put(c, b, cont);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, cont);
    emit_branch(c, b, header);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, merge);
}

static void surf_lock_release(ctx_t *c, spv_blob_t *b, uint32_t ptr) {
    uint32_t old = nid(c);
    put(c, b, (7u << 16) | S_AtomicExchange);
    put(c, b, c->t_u32);
    put(c, b, old);
    put(c, b, ptr);
    put(c, b, const_u32(c, 1));
    put(c, b, const_u32(c, 0x844u)); // Release | Uniform | Image
    put(c, b, const_u32(c, 0));
    (void)old;
}

static void surf_sured_f32_add_cas(ctx_t *c, spv_blob_t *b,
                                   const surf_ref_t *sr, uint32_t coord,
                                   uint32_t val, int line) {
    uint32_t ptr = nid(c);
    put(c, b, (6u << 16) | S_ImageTexelPointer);
    put(c, b, c->t_ptr_img_u32);
    put(c, b, ptr);
    put(c, b, sr->img_ptr);
    put(c, b, coord);
    put(c, b, const_u32(c, 0));
    uint32_t scope = const_u32(c, 1);
    uint32_t sem_eq = const_u32(c, 0x48u);
    uint32_t sem_ne = const_u32(c, 0x42u);
    uint32_t header = nid(c), merge = nid(c), cont = nid(c);
    emit_branch(c, b, header);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, header);
    uint32_t old = nid(c);
    put(c, b, (6u << 16) | S_AtomicLoad);
    put(c, b, c->t_u32);
    put(c, b, old);
    put(c, b, ptr);
    put(c, b, scope);
    put(c, b, sem_ne);
    uint32_t oldf = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_f32);
    put(c, b, oldf);
    put(c, b, old);
    uint32_t nf = nid(c);
    emit_arith(c, b, S_FAdd, c->t_f32, nf, oldf, val);
    uint32_t nb = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_u32);
    put(c, b, nb);
    put(c, b, nf);
    uint32_t prev = nid(c);
    put(c, b, (9u << 16) | S_AtomicCAS);
    put(c, b, c->t_u32);
    put(c, b, prev);
    put(c, b, ptr);
    put(c, b, scope);
    put(c, b, sem_eq);
    put(c, b, sem_ne);
    put(c, b, nb);
    put(c, b, old);
    uint32_t ok = nid(c);
    put(c, b, (5u << 16) | S_IEqual);
    put(c, b, c->t_bool);
    put(c, b, ok);
    put(c, b, prev);
    put(c, b, old);
    put(c, b, (4u << 16) | S_LoopMerge);
    put(c, b, merge);
    put(c, b, cont);
    put(c, b, 0);
    put(c, b, (4u << 16) | S_BranchCond);
    put(c, b, ok);
    put(c, b, merge);
    put(c, b, cont);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, cont);
    emit_branch(c, b, header);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, merge);
    (void)line;
}

static void surf_sured_f32_add_lock(ctx_t *c, spv_blob_t *b,
                                    const ptx_instr_t *in,
                                    const surf_ref_t *sr, uint32_t coord,
                                    uint32_t val, int line) {
    int ncoord = surf_dim(in->surf_geom) + 1 +
                 (surf_arrayed(in->surf_geom) ? 1 : 0);
    uint32_t key = sr->binding >= 0 ? (uint32_t)sr->binding + 1u
                                    : sr->slot + 2u;
    uint32_t lock = surf_lock_ptr(c, b, coord, ncoord, key);
    surf_lock_acquire(c, b, lock);
    uint32_t old = surf_read_f32(c, b, sr->img, coord);
    uint32_t nv = nid(c);
    emit_arith(c, b, S_FAdd, c->t_f32, nv, old, val);
    surf_write_f32(c, b, sr->img, coord, nv);
    surf_lock_release(c, b, lock);
    (void)line;
}

static void surf_sured_f32_lock(ctx_t *c, spv_blob_t *b,
                                const ptx_instr_t *in,
                                const surf_ref_t *sr, uint32_t coord,
                                uint32_t val, int line) {
    int ncoord = surf_dim(in->surf_geom) + 1 +
                 (surf_arrayed(in->surf_geom) ? 1 : 0);
    uint32_t key = sr->binding >= 0 ? (uint32_t)sr->binding + 1u
                                    : sr->slot + 2u;
    uint32_t lock = surf_lock_ptr(c, b, coord, ncoord, key);
    surf_lock_acquire(c, b, lock);
    uint32_t old = surf_read_f32(c, b, sr->img, coord);
    uint32_t cmp = nid(c);
    put(c, b, (5u << 16) |
                  (in->surf_redop == 1 ? S_FOrdLess : S_FOrdGreater));
    put(c, b, c->t_bool);
    put(c, b, cmp);
    put(c, b, old);
    put(c, b, val);
    uint32_t nv = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, c->t_f32);
    put(c, b, nv);
    put(c, b, cmp);
    put(c, b, old);
    put(c, b, val);
    surf_write_f32(c, b, sr->img, coord, nv);
    surf_lock_release(c, b, lock);
    (void)line;
}

static uint32_t surf_key_float(ctx_t *c, spv_blob_t *b, uint32_t key) {
    uint32_t sign = nid(c);
    emit_arith(c, b, S_ShrL, c->t_u32, sign, key, const_u32(c, 31));
    uint32_t pos = nid(c);
    put(c, b, (5u << 16) | S_IEqual);
    put(c, b, c->t_bool);
    put(c, b, pos);
    put(c, b, sign);
    put(c, b, const_u32(c, 1));
    uint32_t mask = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, c->t_u32);
    put(c, b, mask);
    put(c, b, pos);
    put(c, b, const_u32(c, 0x80000000u));
    put(c, b, const_u32(c, 0xffffffffu));
    uint32_t bits = nid(c);
    emit_arith(c, b, S_Xor, c->t_u32, bits, key, mask);
    return bits;
}

// Portable f32 min/max: CAS the R32_UINT alias view using ordered keys.
// The image remains a normal R32_SFLOAT surface to every other shader and
// to the host; only this atomic reduction uses the integer view.
static void surf_sured_f32_cas(ctx_t *c, spv_blob_t *b,
                               const ptx_instr_t *in,
                               const surf_ref_t *sr, uint32_t coord,
                               uint32_t val, int line) {
    uint32_t val_bits = nid(c);
    put(c, b, (4u << 16) | S_Bitcast);
    put(c, b, c->t_u32);
    put(c, b, val_bits);
    put(c, b, val);
    uint32_t val_key = surf_float_key(c, b, val_bits);
    uint32_t ptr = nid(c);
    put(c, b, (6u << 16) | S_ImageTexelPointer);
    put(c, b, c->t_ptr_img_u32);
    put(c, b, ptr);
    put(c, b, sr->img_ptr);
    put(c, b, coord);
    put(c, b, const_u32(c, 0)); // sample
    uint32_t scope = const_u32(c, 1);     // Device
    uint32_t sem_eq = const_u32(c, 0x48u); // AcquireRelease|Uniform
    uint32_t sem_ne = const_u32(c, 0x42u); // Acquire|Uniform
    uint32_t header = nid(c), merge = nid(c), cont = nid(c);
    emit_branch(c, b, header);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, header);
    uint32_t old = nid(c);
    put(c, b, (6u << 16) | S_AtomicLoad);
    put(c, b, c->t_u32);
    put(c, b, old);
    put(c, b, ptr);
    put(c, b, scope);
    put(c, b, sem_ne);
    uint32_t old_key = surf_float_key(c, b, old);
    uint32_t better = nid(c);
    put(c, b, (5u << 16) |
                  (in->surf_redop == 1 ? S_ULess : S_UGreater));
    put(c, b, c->t_bool);
    put(c, b, better);
    put(c, b, old_key);
    put(c, b, val_key);
    uint32_t new_key = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, c->t_u32);
    put(c, b, new_key);
    put(c, b, better);
    put(c, b, old_key);
    put(c, b, val_key);
    uint32_t new_bits = surf_key_float(c, b, new_key);
    uint32_t prev = nid(c);
    put(c, b, (9u << 16) | S_AtomicCAS);
    put(c, b, c->t_u32);
    put(c, b, prev);
    put(c, b, ptr);
    put(c, b, scope);
    put(c, b, sem_eq);
    put(c, b, sem_ne);
    put(c, b, new_bits);
    put(c, b, old);
    uint32_t ok = nid(c);
    put(c, b, (5u << 16) | S_IEqual);
    put(c, b, c->t_bool);
    put(c, b, ok);
    put(c, b, prev);
    put(c, b, old);
    put(c, b, (4u << 16) | S_LoopMerge);
    put(c, b, merge);
    put(c, b, cont);
    put(c, b, 0);
    put(c, b, (4u << 16) | S_BranchCond);
    put(c, b, ok);
    put(c, b, merge);
    put(c, b, cont);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, cont);
    emit_branch(c, b, header);
    put(c, b, (2u << 16) | S_Label);
    put(c, b, merge);
    (void)line;
}

// Byte placement of a sub-word access: (byteoff & 3) * 8.
static uint32_t surf_shift(ctx_t *c, spv_blob_t *b,
                           const ptx_instr_t *in, int line) {
    int first = surf_arrayed(in->surf_geom) ? 1 : 0;
    uint32_t x = surf_scalar(c, b, in->surf_coord[first], line);
    uint32_t m = nid(c);
    emit_arith(c, b, S_And, c->t_u32, m, x, const_u32(c, 3));
    uint32_t sh = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, sh, m, const_u32(c, 3));
    return sh;
}

static uint32_t surf_u64_or(ctx_t *c, spv_blob_t *b, uint32_t lo,
                            uint32_t hi) {
    uint32_t lo64 = nid(c), hi64 = nid(c);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u64);
    put(c, b, lo64);
    put(c, b, lo);
    put(c, b, (4u << 16) | S_UConvert);
    put(c, b, c->t_u64);
    put(c, b, hi64);
    put(c, b, hi);
    uint32_t sh = nid(c);
    emit_arith(c, b, S_Shl, c->t_u64, sh, hi64, const_u64(c, 32));
    uint32_t r = nid(c);
    emit_arith(c, b, S_Or, c->t_u64, r, lo64, sh);
    return r;
}

static int emit_suld(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in,
                     const surf_ref_t *sr, int line) {
    int dim = surf_dim(in->surf_geom);
    int arrayed = surf_arrayed(in->surf_geom);
    uint32_t img = sr->img;
    int n = in->surf_vec;
    for (int i = 0; i < n; i++) {
        uint32_t coord = surf_coord(c, b, in, dim, arrayed, 1, i, line);
        if (in->surf_ty == PTY_U64 && sr->u64) {
            uint32_t c64 = surf_coord64(c, b, in, dim, arrayed, i, line);
            def_val(c, in->surf_dst[i], PTY_U64,
                    surf_read_u64(c, b, img, c64));
        } else if (in->surf_ty == PTY_U64) {
            // A .b64 element spans two consecutive 32-bit words,
            // emitted once per destination register.
            uint32_t c0 = surf_coord(c, b, in, dim, arrayed, 1, i * 2, line);
            uint32_t c1 = surf_coord(c, b, in, dim, arrayed, 1, i * 2 + 1,
                                     line);
            uint32_t lo = surf_read_word(c, b, img, c0, line);
            uint32_t hi = surf_read_word(c, b, img, c1, line);
            def_val(c, in->surf_dst[i], PTY_U64,
                    surf_u64_or(c, b, lo, hi));
        } else if (in->surf_ty == PTY_F32) {
            uint32_t f;
            if (sr->cas) {
                uint32_t w = surf_read_word(c, b, img, coord, line);
                f = nid(c);
                put(c, b, (4u << 16) | S_Bitcast);
                put(c, b, c->t_f32);
                put(c, b, f);
                put(c, b, w);
            } else {
                f = surf_read_f32(c, b, img, coord);
            }
            def_val(c, in->surf_dst[i], PTY_F32, f);
        } else if (in->surf_ty == PTY_U8 || in->surf_ty == PTY_U16) {
            uint32_t w = surf_read_word(c, b, img, coord, line);
            uint32_t sh = surf_shift(c, b, in, line);
            uint32_t t = nid(c);
            emit_arith(c, b, S_ShrL, c->t_u32, t, w, sh);
            uint32_t mask = in->surf_ty == PTY_U8 ? 0xffu : 0xffffu;
            uint32_t v = nid(c);
            emit_arith(c, b, S_And, c->t_u32, v, t,
                       const_u32(c, mask));
            def_val(c, in->surf_dst[i], in->surf_ty, v);
        } else {
            uint32_t w = surf_read_word(c, b, img, coord, line);
            def_val(c, in->surf_dst[i], PTY_U32, w);
        }
    }
    return c->failed ? -1 : 0;
}

static int emit_sust(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in,
                     const surf_ref_t *sr, int line) {
    int dim = surf_dim(in->surf_geom);
    int arrayed = surf_arrayed(in->surf_geom);
    uint32_t img = sr->img;
    int n = in->surf_vec;
    if (in->surf_formatted) {
        // Formatted sample store: source elements are R,G,B,A.
        uint32_t coord = surf_coord(c, b, in, dim, arrayed, 0, 0, line);
        uint32_t comp[4];
        for (int i = 0; i < 4; i++) {
            if (i < n) {
                ptx_opd_t o;
                memset(&o, 0, sizeof(o));
                o.kind = OPD_REG;
                snprintf(o.reg, sizeof(o.reg), "%s", in->surf_src[i]);
                val_t v = resolve(c, b, &o, PTY_F32, line);
                comp[i] = v.id;
            } else {
                comp[i] = const_f32(c, 0.0f);
            }
        }
        uint32_t compbits[4];
        for (int i = 0; i < 4; i++) {
            if (sr->cas) {
                compbits[i] = nid(c);
                put(c, b, (4u << 16) | S_Bitcast);
                put(c, b, c->t_u32);
                put(c, b, compbits[i]);
                put(c, b, comp[i]);
            } else {
                compbits[i] = comp[i];
            }
        }
        uint32_t vec = nid(c);
        put(c, b, (7u << 16) | S_CompositeConstruct);
        put(c, b, sr->cas ? c->t_uvec4 : c->t_v4f32);
        put(c, b, vec);
        for (int i = 0; i < 4; i++)
            put(c, b, compbits[i]);
        put(c, b, (4u << 16) | S_ImageWrite);
        put(c, b, img);
        put(c, b, coord);
        put(c, b, vec);
        return c->failed ? -1 : 0;
    }
    for (int i = 0; i < n; i++) {
        uint32_t coord = surf_coord(c, b, in, dim, arrayed, 1, i, line);
        if (in->surf_ty == PTY_U64 && sr->u64) {
            uint32_t v64 = surf_scalar_as(c, b, in->surf_src[i],
                                          PTY_U64, line);
            uint32_t c64 = surf_coord64(c, b, in, dim, arrayed, i, line);
            surf_write_u64(c, b, img, c64, v64);
            continue;
        }
        if (in->surf_ty == PTY_F32) {
            uint32_t f = surf_scalar_as(c, b, in->surf_src[i], PTY_F32,
                                        line);
            if (sr->cas) {
                uint32_t bits = nid(c);
                put(c, b, (4u << 16) | S_Bitcast);
                put(c, b, c->t_u32);
                put(c, b, bits);
                put(c, b, f);
                surf_write_word(c, b, img, coord, bits);
            } else {
                surf_write_f32(c, b, img, coord, f);
            }
            continue;
        }
        uint32_t val = surf_scalar(c, b, in->surf_src[i], line);
        if (in->surf_ty == PTY_U64) {
            uint32_t hi64 = nid(c);
            emit_arith(c, b, S_ShrL, c->t_u64, hi64, val,
                       const_u64(c, 32));
            uint32_t lo = nid(c), hi = nid(c);
            put(c, b, (4u << 16) | S_UConvert);
            put(c, b, c->t_u32);
            put(c, b, lo);
            put(c, b, val);
            put(c, b, (4u << 16) | S_UConvert);
            put(c, b, c->t_u32);
            put(c, b, hi);
            put(c, b, hi64);
            uint32_t c0 = surf_coord(c, b, in, dim, arrayed, 1, i * 2, line);
            uint32_t c1 = surf_coord(c, b, in, dim, arrayed, 1, i * 2 + 1,
                                     line);
            surf_write_word(c, b, img, c0, lo);
            surf_write_word(c, b, img, c1, hi);
        } else if (in->surf_ty == PTY_U8 || in->surf_ty == PTY_U16) {
            // Read-modify-write the containing word.
            uint32_t w = surf_read_word(c, b, img, coord, line);
            uint32_t sh = surf_shift(c, b, in, line);
            uint32_t mask = in->surf_ty == PTY_U8 ? 0xffu : 0xffffu;
            uint32_t sm = nid(c);
            emit_arith(c, b, S_Shl, c->t_u32, sm, const_u32(c, mask), sh);
            uint32_t nm = nid(c);
            put(c, b, (4u << 16) | S_Not);
            put(c, b, c->t_u32);
            put(c, b, nm);
            put(c, b, sm);
            uint32_t cl = nid(c);
            emit_arith(c, b, S_And, c->t_u32, cl, w, nm);
            uint32_t vv = nid(c);
            emit_arith(c, b, S_And, c->t_u32, vv, val,
                       const_u32(c, mask));
            uint32_t sv = nid(c);
            emit_arith(c, b, S_Shl, c->t_u32, sv, vv, sh);
            uint32_t nw = nid(c);
            emit_arith(c, b, S_Or, c->t_u32, nw, cl, sv);
            surf_write_word(c, b, img, coord, nw);
        } else {
            surf_write_word(c, b, img, coord, val);
        }
    }
    return c->failed ? -1 : 0;
}

static int emit_sured(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in,
                      const surf_ref_t *sr, int line) {
    if (in->surf_ty == PTY_U64) {
        // 64-bit storage-image atomics: R64ui + Int64ImageEXT.
        if (!c->caps.int64_image || !sr->u64) {
            failf(c, line, "64-bit sured requires an R64ui surface "
                           "(SPV_EXT_shader_image_int64)");
            return -1;
        }
        if (in->surf_formatted) {
            failf(c, line, "sured u64 supports .b only");
            return -1;
        }
        if (sr->dynamic) {
            failf(c, line,
                  "sured u64 on a dynamic surface is unsupported");
            return -1;
        }
        int dim = surf_dim(in->surf_geom);
        int arrayed = surf_arrayed(in->surf_geom);
        uint32_t coord = surf_coord64(c, b, in, dim, arrayed, 0, line);
        uint32_t val = surf_scalar_as(c, b, in->surf_src[0], PTY_U64,
                                      line);
        uint32_t ptr = nid(c);
        put(c, b, (6u << 16) | S_ImageTexelPointer);
        put(c, b, c->t_ptr_img_u64);
        put(c, b, ptr);
        put(c, b, sr->img_ptr);
        put(c, b, coord);
        put(c, b, const_u32(c, 0)); // sample
        uint32_t opc;
        switch (in->surf_redop) {
        case 0: opc = S_AtomicIAdd; break;
        case 1: opc = in->surf_sign ? S_AtomicSMin : S_AtomicUMin; break;
        case 2: opc = in->surf_sign ? S_AtomicSMax : S_AtomicUMax; break;
        case 3: opc = S_AtomicAnd; break;
        default: opc = S_AtomicOr; break;
        }
        uint32_t old = nid(c);
        put(c, b, (7u << 16) | opc);
        put(c, b, c->t_u64);
        put(c, b, old);
        put(c, b, ptr);
        put(c, b, const_u32(c, 1)); // Device scope
        put(c, b, const_u32(c, 0)); // Relaxed
        put(c, b, val);
        (void)old;
        return c->failed ? -1 : 0;
    }
    if (in->surf_ty == PTY_F32) {
        // Storage-image f32 atomics on an OpImageTexelPointer.
        unsigned need;
        uint32_t opcode;
        const char *opname;
        if (in->surf_redop == 0) {
            need = ACUDA_ATOMIC_IMAGE_F32_ADD;
            opcode = S_AtomicFAddEXT;
            opname = "sured.add.f32";
        } else if (in->surf_redop == 1) {
            need = ACUDA_ATOMIC_IMAGE_F32_MINMAX;
            opcode = S_AtomicFMinEXT;
            opname = "sured.min.f32";
        } else if (in->surf_redop == 2) {
            need = ACUDA_ATOMIC_IMAGE_F32_MINMAX;
            opcode = S_AtomicFMaxEXT;
            opname = "sured.max.f32";
        } else {
            failf(c, line, "sured f32 supports .add/.min/.max only");
            return -1;
        }
        int fallback = 0;
        int add_fallback = 0;
        if (!(c->caps.atomic_float & need)) {
            if (in->surf_redop == 1 || in->surf_redop == 2) {
                // Use the alias view + CAS path when available; otherwise
                // serialize the compare/update through a per-texel lock.
                fallback = 1;
            } else if (sr->cas || sr->lock) {
                add_fallback = 1;
            } else {
                char msg[128];
                snprintf(msg, sizeof(msg), "%s requires %s", opname,
                         "shaderImageFloat32AtomicAdd");
                failf(c, line, msg);
                return -1;
            }
        }
        if (sr->dynamic) {
            if (in->surf_redop == 1 || in->surf_redop == 2)
                fallback = 1;
            else if (in->surf_redop == 0 && (sr->cas || sr->lock))
                add_fallback = 1;
        }
        if (sr->dynamic && !fallback && !add_fallback) {
            char msg[128];
            snprintf(msg, sizeof(msg),
                     "%s on a dynamic surface requires a fallback table",
                     opname);
            failf(c, line, msg);
            return -1;
        }
        int dim = surf_dim(in->surf_geom);
        int arrayed = surf_arrayed(in->surf_geom);
        int bytes = in->surf_formatted ? 0 : 1;
        uint32_t coord = surf_coord(c, b, in, dim, arrayed, bytes, 0, line);
        uint32_t val = surf_scalar_as(c, b, in->surf_src[0], PTY_F32, line);
        if (add_fallback) {
            if (sr->cas)
                surf_sured_f32_add_cas(c, b, sr, coord, val, line);
            else
                surf_sured_f32_add_lock(c, b, in, sr, coord, val, line);
            return c->failed ? -1 : 0;
        }
        if (fallback) {
            if (sr->cas) {
                surf_sured_f32_cas(c, b, in, sr, coord, val, line);
            } else if (sr->lock) {
                surf_sured_f32_lock(c, b, in, sr, coord, val, line);
            } else {
                failf(c, line, "f32 surface fallback has no path");
                return -1;
            }
            return c->failed ? -1 : 0;
        }
        uint32_t ptr = nid(c);
        put(c, b, (6u << 16) | S_ImageTexelPointer);
        put(c, b, c->t_ptr_img_f32);
        put(c, b, ptr);
        put(c, b, sr->img_ptr);
        put(c, b, coord);
        put(c, b, const_u32(c, 0)); // sample
        uint32_t old = nid(c);
        put(c, b, (7u << 16) | opcode);
        put(c, b, c->t_f32);
        put(c, b, old);
        put(c, b, ptr);
        put(c, b, const_u32(c, 1)); // Device scope
        put(c, b, const_u32(c, 0)); // Relaxed
        put(c, b, val);
        (void)old;
        return c->failed ? -1 : 0;
    }
    if (in->surf_ty != PTY_U32 || in->surf_sign) {
        failf(c, line, "sured supports 32-bit unsigned/b32 reductions only");
        return -1;
    }
    int dim = surf_dim(in->surf_geom);
    int arrayed = surf_arrayed(in->surf_geom);
    int bytes = in->surf_formatted ? 0 : 1;
    uint32_t coord = surf_coord(c, b, in, dim, arrayed, bytes, 0, line);
    uint32_t val = surf_scalar(c, b, in->surf_src[0], line);
    uint32_t ptr = nid(c);
    put(c, b, (6u << 16) | S_ImageTexelPointer);
    put(c, b, c->t_ptr_img_u32);
    put(c, b, ptr);
    put(c, b, sr->img_ptr); // pointer to the image variable
    put(c, b, coord);
    put(c, b, const_u32(c, 0)); // sample
    uint32_t op;
    switch (in->surf_redop) {
    case 0: op = S_AtomicIAdd; break;
    case 1: op = S_AtomicUMin; break;
    case 2: op = S_AtomicUMax; break;
    case 3: op = S_AtomicAnd; break;
    default: op = S_AtomicOr; break;
    }
    uint32_t old = nid(c);
    put(c, b, (7u << 16) | op);
    put(c, b, c->t_u32);
    put(c, b, old);
    put(c, b, ptr);
    put(c, b, const_u32(c, 1)); // Device scope
    put(c, b, const_u32(c, 0)); // Relaxed
    put(c, b, val);
    (void)old;
    return c->failed ? -1 : 0;
}

static int emit_suq(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in,
                    const surf_ref_t *sr, int line) {
    if (check_dst(c, in->dst, PTY_U32, line) != 0)
        return -1;
    uint32_t w[6];
    if (sr->dynamic) {
        uint32_t zero = const_u32(c, 0);
        uint32_t base = nid(c);
        emit_arith(c, b, S_IMul, c->t_u32, base, sr->slot,
                   const_u32(c, 6));
        for (int i = 0; i < 6; i++) {
            uint32_t word = nid(c);
            emit_arith(c, b, S_IAdd, c->t_u32, word, base,
                       const_u32(c, (uint32_t)i));
            uint32_t ptr = nid(c);
            put(c, b, (6u << 16) | S_AccessChain);
            put(c, b, c->surf_meta_ptr_u32);
            put(c, b, ptr);
            put(c, b, c->surf_meta_var);
            put(c, b, zero);
            put(c, b, word);
            w[i] = nid(c);
            put(c, b, (4u << 16) | S_Load);
            put(c, b, c->t_u32);
            put(c, b, w[i]);
            put(c, b, ptr);
        }
    } else {
        int base = c->tex_meta_word[sr->binding];
        if (base < 0) {
            failf(c, line, "surface has no metadata block");
            return -1;
        }
        for (int i = 0; i < 6; i++)
            emit_push_load(c, b, &w[i], PTY_U32, base + i, line);
    }
    if (c->failed)
        return -1;
    uint32_t v = 0;
    switch (in->surf_query) {
    case 0: case 1: case 2: case 5:
        v = w[in->surf_query == 5 ? 3 : in->surf_query];
        break;
    case 3:
        v = nid(c);
        emit_arith(c, b, S_And, c->t_u32, v, w[4], const_u32(c, 0xffu));
        break;
    case 4:
        v = nid(c);
        emit_arith(c, b, S_ShrL, c->t_u32, v, w[4], const_u32(c, 8));
        v = nid(c);
        emit_arith(c, b, S_And, c->t_u32, v, v, const_u32(c, 0xffu));
        break;
    default: // memory_layout: always tiled
        v = const_u32(c, 0);
        break;
    }
    def_val(c, in->dst, PTY_U32, v);
    return c->failed ? -1 : 0;
}

int emit_surface_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    surf_ref_t sr;
    if (surf_ref_resolve(c, b, in, in->surf_ref, line, &sr) != 0)
        return -1;
    if (in->op == POP_SUQ)
        return emit_suq(c, b, in, &sr, line);
    if (in->op == POP_SULD)
        return emit_suld(c, b, in, &sr, line);
    if (in->op == POP_SUST)
        return emit_sust(c, b, in, &sr, line);
    if (in->op == POP_SURED)
        return emit_sured(c, b, in, &sr, line);
    return -1;
}
