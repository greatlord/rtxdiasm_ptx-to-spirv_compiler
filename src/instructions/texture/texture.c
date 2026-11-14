// Texture instructions: tex, tld4, txq and istypep.
//
// A .texref parameter maps to a Vulkan image.  In unified mode the
// descriptor is a combined image sampler; when the kernel also declares
// a .samplerref (independent mode), the texture is bound as a plain
// image and OpSampledImage pairs it with the separately bound sampler.
// Geometry (1D/2D/3D/array/cube/multisample), mipmap modes (.base,
// .level, .grad), offsets and depth compare are all lowered to the
// matching SPIR-V image instruction.
//
// The legacy buffer-backed descriptor path (a 64-byte header followed by
// texels) is kept for textures that do not resolve to a descriptor
// binding.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "spirv_internal.h"

// The texture descriptor is a 64-byte header followed by texels:
//   0 magic, 1 width, 2 height, 3 depth, 4 flags, 5 addr0, 6 addr1,
//   7 addr2, 8 data_off, 9 pitch, 10 slice_pitch, 11 channels,
//   12 format (0 f32, 1 u32, 2 s32), 13 levels, 14 samples.
#define TEX_HDR_BYTES 64u
#define TEXF_NORMALIZED 1u
#define TEXF_FILTER_SHIFT 1
#define TEXF_CHAN_SHIFT 3
#define TEXF_FMT_SHIFT 5

static uint32_t tex_bool_const(ctx_t *c, int v) {
    uint32_t id = nid(c);
    put(c, &c->types, (3u << 16) | (v ? S_ConstantTrue : S_ConstantFalse));
    put(c, &c->types, c->t_bool);
    put(c, &c->types, id);
    return id;
}

int tex_ref_binding(ctx_t *c, const char *name, int *binding,
                           uint32_t *off, int *is_tex, int line) {
    reg_t *r = find_reg(c, name);
    if (r && r->defined && r->is_ptr) {
        *binding = r->binding;
        *off = r->off;
        *is_tex = r->is_tex;
        return 0;
    }
    cparam_t *cp = find_cparam(c, name);
    if (cp && cp->kind == 1) {
        *binding = cp->binding;
        *off = const_u64(c, 0);
        *is_tex = cp->is_tex;
        return 0;
    }
    failf(c, line, "texture operand must be a .texref parameter or a "
                   "register holding one");
    return -1;
}

// ---- image geometry helpers -------------------------------------------

static uint32_t tex_vec_ty(ctx_t *c, uint32_t scalar, int n) {
    if (n <= 1)
        return scalar;
    if (scalar == c->t_f32) {
        if (n == 2) return c->t_v2f32;
        if (n == 3) return c->t_v3f32;
        return c->t_v4f32;
    }
    if (n == 2) return c->t_v2u32;
    if (n == 3) return c->t_v3u32;
    return c->t_uvec4;
}

static uint32_t tex_scalar(ctx_t *c, spv_blob_t *b, const char *name,
                           ptx_ty_t from, int sign, int to_float,
                           int line) {
    ptx_opd_t o;
    memset(&o, 0, sizeof(o));
    o.kind = OPD_REG;
    snprintf(o.reg, sizeof(o.reg), "%s", name);
    val_t v = resolve(c, b, &o, from, line);
    if (c->failed)
        return 0;
    if (to_float) {
        if (v.ty == PTY_F32)
            return v.id;
        uint32_t id = nid(c);
        put(c, b, (4u << 16) | (sign ? S_ConvertSToF : S_ConvertUToF));
        put(c, b, c->t_f32);
        put(c, b, id);
        put(c, b, v.id);
        return id;
    }
    if (v.ty == PTY_F32) {
        uint32_t id = nid(c);
        put(c, b, (4u << 16) | S_ConvertFToS);
        put(c, b, c->t_u32);
        put(c, b, id);
        put(c, b, v.id);
        return id;
    }
    return v.id;
}

static uint32_t tex_floor(ctx_t *c, spv_blob_t *b, uint32_t f,
                            int line);

static uint32_t tex_make_vec(ctx_t *c, spv_blob_t *b, uint32_t ty,
                             const uint32_t *comp, int n, int line) {
    if (n <= 1)
        return comp[0];
    uint32_t id = nid(c);
    put(c, b, ((uint32_t)(n + 3) << 16) | S_CompositeConstruct);
    put(c, b, ty);
    put(c, b, id);
    for (int i = 0; i < n; i++)
        put(c, b, comp[i]);
    (void)line;
    return id;
}

// Resolve the coordinate registers of `in` into a float (or integer)
// vector shaped for the image.
static uint32_t tex_coord_vector(ctx_t *c, spv_blob_t *b,
                                 const ptx_instr_t *in, int dim, int arrayed,
                                 int ms, int to_float, int line) {
    int ctype = in->ty2;
    int sign = ctype != PTY_F32 && in->tex_csign;
    uint32_t comp[4];
    int n = 0;
    // SPIR-V orders arrayed coordinates as the texel coordinates
    // followed by the array layer; PTX puts the layer first.
    if (ms) {
        // 2dms: {sample, x, y}; a2dms: {sample, index, x, y}.
        int ci = arrayed ? 2 : 1;
        comp[n++] = tex_scalar(c, b, in->tex_coord[ci], PTY_U32, 0,
                               to_float, line);
        comp[n++] = tex_scalar(c, b, in->tex_coord[ci + 1], PTY_U32, 0,
                               to_float, line);
        if (arrayed)
            comp[n++] = tex_scalar(c, b, in->tex_coord[1], PTY_U32, 0,
                                   to_float, line);
    } else {
        int first = arrayed ? 1 : 0;
        comp[n++] = tex_scalar(c, b, in->tex_coord[first], (ptx_ty_t)ctype,
                               sign, to_float, line);
        if (dim != 0) {
            comp[n++] = tex_scalar(c, b, in->tex_coord[first + 1],
                                   (ptx_ty_t)ctype, sign, to_float, line);
        }
        if (dim == 2 || dim == 3) {
            comp[n++] = tex_scalar(c, b, in->tex_coord[first + 2],
                                   (ptx_ty_t)ctype, sign, to_float, line);
        }
        if (arrayed)
            comp[n++] = tex_scalar(c, b, in->tex_coord[0], PTY_U32, 0,
                                   to_float, line);
    }
    uint32_t ty = tex_vec_ty(c, to_float ? c->t_f32 : c->t_u32, n);
    return tex_make_vec(c, b, ty, comp, n, line);
}

// 1D textures are represented as 2D tiled images. Convert a 1D coordinate
// into 2D using tex metadata word 6 (tile stride) and word 0 (logical
// width). Integer coordinates (tex1Dfetch) tile directly with
// u=x%stride, v=x/stride. Normalized float coordinates are first scaled
// by the logical width (f=s*W), tiled in the float domain
// (u=f-v*stride, v=floor(f/stride)), then normalized for the 2D image
// (((u+0.5)/tile_w, (v+0.5)/tile_h)). Arrayed 1D keeps the PTX layer
// (coord[0]) as the final component.
static uint32_t tex_coord_vector_1d_tiled(ctx_t *c, spv_blob_t *b,
                                          const ptx_instr_t *in, int binding,
                                          int to_float, int line) {
    int arrayed = c->tex_arrayed[binding];
    int xidx = arrayed ? 1 : 0;
    int ctype = in->ty2;
    int sign = ctype != PTY_F32 && in->tex_csign;
    uint32_t stride = 0;
    int base = c->tex_meta_word[binding];
    if (base >= 0)
        emit_push_load(c, b, &stride, PTY_U32, base + 6, line);
    else
        stride = const_u32(c, 1);
    if (c->failed)
        return 0;
    // Normalized float sampling: scale by logical width, tile in float,
    // then normalize for the 2D image.
    if (to_float && ctype == PTY_F32) {
        uint32_t s = tex_scalar(c, b, in->tex_coord[xidx], PTY_F32, 0,
                                1, line);
        uint32_t w = 0;
        if (base >= 0)
            emit_push_load(c, b, &w, PTY_U32, base + 0, line);
        else
            w = stride;
        if (c->failed)
            return 0;
        uint32_t wf = nid(c), stride_f = nid(c);
        put(c, b, (4u << 16) | S_ConvertUToF);
        put(c, b, c->t_f32);
        put(c, b, wf);
        put(c, b, w);
        put(c, b, (4u << 16) | S_ConvertUToF);
        put(c, b, c->t_f32);
        put(c, b, stride_f);
        put(c, b, stride);
        // tile_h = ceil(W / stride) for the v normalization.
        uint32_t stride_m1 = nid(c), w_m1 = nid(c);
        emit_arith(c, b, S_IAdd, c->t_u32, stride_m1, stride,
                   const_u32(c, 0xffffffffu));
        emit_arith(c, b, S_IAdd, c->t_u32, w_m1, w, stride_m1);
        uint32_t tile_h = nid(c);
        emit_arith(c, b, S_UDiv, c->t_u32, tile_h, w_m1, stride);
        uint32_t tile_h_f = nid(c);
        put(c, b, (4u << 16) | S_ConvertUToF);
        put(c, b, c->t_f32);
        put(c, b, tile_h_f);
        put(c, b, tile_h);
        uint32_t f = nid(c), q = nid(c), v = nid(c);
        emit_arith(c, b, S_FMul, c->t_f32, f, s, wf);
        emit_arith(c, b, S_FDiv, c->t_f32, q, f, stride_f);
        v = tex_floor(c, b, q, line);
        uint32_t vs = nid(c), u = nid(c);
        emit_arith(c, b, S_FMul, c->t_f32, vs, v, stride_f);
        emit_arith(c, b, S_FSub, c->t_f32, u, f, vs);
        uint32_t half = const_f32(c, 0x3F000000u);
        uint32_t uh = nid(c), vh = nid(c), un = nid(c), vn = nid(c);
        emit_arith(c, b, S_FAdd, c->t_f32, uh, u, half);
        emit_arith(c, b, S_FAdd, c->t_f32, vh, v, half);
        emit_arith(c, b, S_FDiv, c->t_f32, un, uh, stride_f);
        emit_arith(c, b, S_FDiv, c->t_f32, vn, vh, tile_h_f);
        if (!arrayed) {
            uint32_t comp[2];
            comp[0] = un;
            comp[1] = vn;
            return tex_make_vec(c, b, c->t_v2f32, comp, 2, line);
        }
        uint32_t layer = tex_scalar(c, b, in->tex_coord[0], PTY_U32, 0,
                                    1, line);
        uint32_t comp[3];
        comp[0] = un;
        comp[1] = vn;
        comp[2] = layer;
        return tex_make_vec(c, b, tex_vec_ty(c, c->t_f32, 3), comp, 3,
                            line);
    }
    // Load x as integer for tiling arithmetic.
    uint32_t x = tex_scalar(c, b, in->tex_coord[xidx], (ptx_ty_t)ctype,
                            sign, 0, line);
    if (c->failed)
        return 0;
    uint32_t u = nid(c);
    uint32_t v = nid(c);
    emit_arith(c, b, S_UMod, c->t_u32, u, x, stride);
    emit_arith(c, b, S_UDiv, c->t_u32, v, x, stride);
    if (!arrayed) {
        uint32_t comp[2];
        if (to_float) {
            uint32_t uf = nid(c), vf = nid(c);
            put(c, b, (4u << 16) | S_ConvertUToF);
            put(c, b, c->t_f32);
            put(c, b, uf);
            put(c, b, u);
            put(c, b, (4u << 16) | S_ConvertUToF);
            put(c, b, c->t_f32);
            put(c, b, vf);
            put(c, b, v);
            comp[0] = uf;
            comp[1] = vf;
            return tex_make_vec(c, b, c->t_v2f32, comp, 2, line);
        }
        comp[0] = u;
        comp[1] = v;
        return tex_make_vec(c, b, c->t_v2u32, comp, 2, line);
    }
    uint32_t layer = tex_scalar(c, b, in->tex_coord[0], PTY_U32, 0,
                                to_float, line);
    if (to_float) {
        uint32_t uf = nid(c), vf = nid(c);
        put(c, b, (4u << 16) | S_ConvertUToF);
        put(c, b, c->t_f32);
        put(c, b, uf);
        put(c, b, u);
        put(c, b, (4u << 16) | S_ConvertUToF);
        put(c, b, c->t_f32);
        put(c, b, vf);
        put(c, b, v);
        uint32_t comp[3];
        comp[0] = uf;
        comp[1] = vf;
        comp[2] = layer;
        return tex_make_vec(c, b, tex_vec_ty(c, c->t_f32, 3), comp, 3,
                            line);
    }
    {
        uint32_t comp[3];
        comp[0] = u;
        comp[1] = v;
        comp[2] = layer;
        return tex_make_vec(c, b, tex_vec_ty(c, c->t_u32, 3), comp, 3,
                            line);
    }
}

// Resolve an offset operand to a compile-time integer.  Vulkan only
// permits runtime offsets on gather operations (and only with
// ImageGatherExtended), so sampling offsets must be constant.
static int tex_const_i32(ctx_t *c, const char *name, int32_t *out) {
    char *end = NULL;
    long v = strtol(name, &end, 0);
    if (end && *end == '\0' && end != name) {
        *out = (int32_t)v;
        return 0;
    }
    for (size_t bi = 0; bi < c->cur_fn->nblocks; bi++)
        for (size_t ii = 0; ii < c->cur_fn->blocks[bi].count; ii++) {
            const ptx_instr_t *m = &c->cur_fn->blocks[bi].instrs[ii];
            if (m->op == POP_MOV && strcmp(m->dst, name) == 0 &&
                m->nsrc == 1 && m->src[0].kind == OPD_IMM) {
                *out = (int32_t)(uint32_t)m->src[0].imm;
                return 0;
            }
        }
    return -1;
}

static uint32_t tex_const_vec_u32(ctx_t *c, const int32_t *vals, int n) {
    if (n == 1)
        return const_u32(c, (uint32_t)vals[0]);
    uint32_t ty = tex_vec_ty(c, c->t_u32, n);
    uint32_t id = nid(c);
    put(c, &c->types, ((uint32_t)(n + 3) << 16) | S_ConstantComposite);
    put(c, &c->types, ty);
    put(c, &c->types, id);
    for (int i = 0; i < n; i++)
        put(c, &c->types, const_u32(c, (uint32_t)vals[i]));
    return id;
}

// Offset vector (integer) of `n` components.  Returns 1 when every
// component is a compile-time constant.
static int tex_offset_vector(ctx_t *c, spv_blob_t *b,
                             const ptx_instr_t *in, int n, uint32_t *out,
                             int line) {
    uint32_t comp[4];
    int32_t cval[4];
    int all_const = 1;
    for (int i = 0; i < n; i++) {
        if (tex_const_i32(c, in->tex_off[i], &cval[i]) != 0)
            all_const = 0;
    }
    if (all_const) {
        *out = tex_const_vec_u32(c, cval, n);
        return 1;
    }
    for (int i = 0; i < n; i++)
        comp[i] = tex_scalar(c, b, in->tex_off[i], PTY_U32, 1, 0, line);
    *out = tex_make_vec(c, b, tex_vec_ty(c, c->t_u32, n), comp, n, line);
    return 0;
}

// Load the sampled image (and the underlying image) for a binding,
// pairing it with the explicit sampler when one is named.
static int tex_load_image(ctx_t *c, spv_blob_t *b, int binding,
                          const char *sampname, int line,
                          uint32_t *simg_out, uint32_t *img_out) {
    uint32_t var = c->tex_vars[binding];
    uint32_t imty = c->tex_image_ty[binding];
    uint32_t smty = c->tex_sampled_ty[binding];
    if (!var || !imty || !smty) {
        failf(c, line, "texture binding has no image descriptor");
        return -1;
    }
    uint32_t img, simg;
    if (c->tex_split) {
        img = nid(c);
        put(c, b, (4u << 16) | S_Load);
        put(c, b, imty);
        put(c, b, img);
        put(c, b, var);
        simg = 0;
    } else {
        simg = nid(c);
        put(c, b, (4u << 16) | S_Load);
        put(c, b, smty);
        put(c, b, simg);
        put(c, b, var);
        img = 0;
    }
    if (sampname && sampname[0]) {
        int sb = -1, sis = 0;
        uint32_t soff = 0;
        if (tex_ref_binding(c, sampname, &sb, &soff, &sis, line) != 0)
            return -1;
        if (sb < 0 || !c->tex_sampler_var[sb]) {
            char msg[128];
            snprintf(msg, sizeof(msg),
                     "%s is not a .samplerref parameter", sampname);
            failf(c, line, msg);
            return -1;
        }
        uint32_t samp = nid(c);
        put(c, b, (4u << 16) | S_Load);
        put(c, b, c->t_sampler_ty);
        put(c, b, samp);
        put(c, b, c->tex_sampler_var[sb]);
        if (!img) {
            img = nid(c);
            put(c, b, (4u << 16) | S_Image);
            put(c, b, imty);
            put(c, b, img);
            put(c, b, simg);
        }
        simg = nid(c);
        put(c, b, (5u << 16) | S_SampledImage);
        put(c, b, smty);
        put(c, b, simg);
        put(c, b, img);
        put(c, b, samp);
    } else if (c->tex_split) {
        failf(c, line, "kernel declares .samplerref; tex needs an explicit "
                       "sampler operand");
        return -1;
    }
    if (!simg) {
        // split mode without an explicit sampler is rejected above.
        failf(c, line, "texture has no sampler");
        return -1;
    }
    if (!img) {
        img = nid(c);
        put(c, b, (4u << 16) | S_Image);
        put(c, b, imty);
        put(c, b, img);
        put(c, b, simg);
    }
    *simg_out = simg;
    *img_out = img;
    return 0;
}

// txq answers come from the per-binding metadata words the runtime
// appends to the push constants.  This keeps the queries independent of
// the shader's image shape, so txq-only kernels work for every texture
// geometry and report the real sampler state.
static int emit_image_txq(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in,
                          int binding, int line) {
    if (check_dst(c, in->dst, PTY_U32, line) != 0)
        return -1;
    int base = c->tex_meta_word[binding];
    if (base < 0) {
        failf(c, line, "texture has no metadata block");
        return -1;
    }
    uint32_t w[6];
    for (int i = 0; i < 6; i++)
        emit_push_load(c, b, &w[i], PTY_U32, base + i, line);
    if (c->failed)
        return -1;
    uint32_t v = 0;
    switch (in->tex_query) {
    case 0: // width
    case 1: // height
    case 2: // depth
        v = w[in->tex_query];
        if (in->nsrc > 0) {
            // txq.level: max(1, extent >> lod).
            ptx_opd_t o = in->src[0];
            val_t lv = resolve(c, b, &o, PTY_U32, line);
            if (c->failed)
                return -1;
            uint32_t over = do_cmp(c, b, PCM_GE, PTY_U32, 1, lv.id,
                                   const_u32(c, 32), line);
            uint32_t amt = nid(c);
            emit_arith(c, b, S_And, c->t_u32, amt, lv.id,
                       const_u32(c, 31));
            uint32_t shifted = nid(c);
            emit_arith(c, b, S_ShrL, c->t_u32, shifted, v, amt);
            uint32_t zero = do_cmp(c, b, PCM_EQ, PTY_U32, 0, shifted,
                                   const_u32(c, 0), line);
            uint32_t clamped = sel_u32(c, b, zero, const_u32(c, 1),
                                       shifted);
            v = sel_u32(c, b, over, const_u32(c, 1), clamped);
        }
        break;
    case 3: // channel_data_type: FLOAT=2, UNSIGNED=0
        v = const_u32(c, c->tex_int[binding] ? 0u : 2u);
        break;
    case 4: // channel_order: RGBA
        v = const_u32(c, 3);
        break;
    case 5: { // normalized_coords
        uint32_t t = nid(c);
        emit_arith(c, b, S_ShrL, c->t_u32, t, w[5], const_u32(c, 2));
        v = nid(c);
        emit_arith(c, b, S_And, c->t_u32, v, t, const_u32(c, 1));
        break;
    }
    case 6: // array_size
        v = w[3];
        break;
    case 7: // num_mipmap_levels
        v = nid(c);
        emit_arith(c, b, S_And, c->t_u32, v, w[4],
                   const_u32(c, 0xffffu));
        break;
    case 8: // num_samples
        v = nid(c);
        emit_arith(c, b, S_ShrL, c->t_u32, v, w[4], const_u32(c, 16));
        break;
    case 9: // filter_mode
        v = nid(c);
        emit_arith(c, b, S_And, c->t_u32, v, w[5], const_u32(c, 3));
        break;
    case 10: case 11: case 12: {
        uint32_t sh = (uint32_t)(in->tex_query - 10) * 4u + 4u;
        uint32_t t = nid(c);
        emit_arith(c, b, S_ShrL, c->t_u32, t, w[5], const_u32(c, sh));
        v = nid(c);
        emit_arith(c, b, S_And, c->t_u32, v, t, const_u32(c, 0xfu));
        break;
    }
    case 13: // force_unnormalized_coords
        v = const_u32(c, 0);
        break;
    default:
        v = const_u32(c, 0);
        break;
    }
    if (c->failed)
        return -1;
    def_val(c, in->dst, PTY_U32, v);
    return 0;
}

typedef struct {
    uint32_t mask;
    uint32_t words[8];
    int nw;
} texops_t;

// Build the ImageOperands mask and values.  Any instruction emitted for
// gradient/offset vectors lands before the image instruction, which the
// caller emits afterwards.
static int tex_build_operands(ctx_t *c, spv_blob_t *b,
                              const ptx_instr_t *in, int dim,
                              uint32_t lod, int base_comps, int need_lod,
                              int allow_dynamic, texops_t *ops, int line) {
    memset(ops, 0, sizeof(*ops));
    if (need_lod && in->tex_mode != 2) {
        // OpImageSampleExplicitLod requires Lod or Grad.
        ops->mask |= 2u; // Lod
        ops->words[ops->nw++] = lod;
    } else if (in->tex_mode == 2) {
        ops->mask |= 4u; // Grad
        uint32_t comps[4];
        for (int i = 0; i < base_comps; i++)
            comps[i] = tex_scalar(c, b, in->tex_dpdx[i], PTY_F32, 0, 1, line);
        ops->words[ops->nw++] =
            tex_make_vec(c, b, tex_vec_ty(c, c->t_f32, base_comps), comps,
                         base_comps, line);
        for (int i = 0; i < base_comps; i++)
            comps[i] = tex_scalar(c, b, in->tex_dpdy[i], PTY_F32, 0, 1, line);
        ops->words[ops->nw++] =
            tex_make_vec(c, b, tex_vec_ty(c, c->t_f32, base_comps), comps,
                         base_comps, line);
    }
    if (in->tex_noff > 0) {
        if (dim == 3) {
            failf(c, line, "texture offsets are not supported for cubemaps");
            return -1;
        }
        if (in->tex_noff != base_comps) {
            failf(c, line, "texture offset has the wrong component count");
            return -1;
        }
        uint32_t offvec = 0;
        int all_const = tex_offset_vector(c, b, in, base_comps, &offvec,
                                          line);
        if (all_const) {
            ops->mask |= 0x8u; // ConstOffset
        } else if (allow_dynamic) {
            ops->mask |= 0x10u; // Offset (gather only)
        } else {
            failf(c, line, "tex offset must be a compile-time constant");
            return -1;
        }
        ops->words[ops->nw++] = offvec;
    }
    if (c->failed)
        return -1;
    return 0;
}

static void tex_emit_operands(ctx_t *c, spv_blob_t *b,
                              const texops_t *ops) {
    if (!ops->mask)
        return;
    put(c, b, ops->mask);
    for (int i = 0; i < ops->nw; i++)
        put(c, b, ops->words[i]);
}

// Bind the four sampled components to the PTX destination registers,
// narrowing to .f16 / .f16x2 when requested.
static void tex_def_result(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in,
                           const uint32_t comp[4], int line) {
    if (in->tex_f16 == 1) {
        for (int i = 0; i < 4; i++) {
            uint32_t h = f32_to_f16bits(c, b, comp[i], line);
            def_val(c, in->tex_dst[i], PTY_F16, h);
        }
    } else if (in->tex_f16 == 2) {
        // Lane order: the even component occupies the low half, the
        // odd component the high half (little-endian pair packing).
        for (int j = 0; j < 2; j++) {
            uint32_t lo = f32_to_f16bits(c, b, comp[2 * j], line);
            uint32_t hi = f32_to_f16bits(c, b, comp[2 * j + 1], line);
            uint32_t packed = pack2(c, b, hi, lo);
            def_val(c, in->tex_dst[j], PTY_F16X2, packed);
        }
    } else {
        for (int i = 0; i < 4; i++)
            def_val(c, in->tex_dst[i], in->ty, comp[i]);
    }
}

static int emit_image_tex(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in,
                          int binding, int line) {
    // Texture data is always resident in this model.
    if (in->tex_dstp[0])
        def_val(c, in->tex_dstp, PTY_PRED, tex_bool_const(c, 1));
    int dim = c->tex_dim[binding];
    int arrayed = c->tex_arrayed[binding];
    int ms = c->tex_ms[binding];
    int geom1d = (in->tex_geom == 1 || in->tex_geom == 4);
    uint32_t simg = 0, img = 0;
    if (tex_load_image(c, b, binding, in->tex_sampler, line, &simg,
                       &img) != 0)
        return -1;
    (void)img;
    int base_comps = dim == 2 ? 3 : (dim == 0 ? 1 : 3);
    if (dim == 1)
        base_comps = 2;
    // 1D textures are backed by 2D tiled images (2 comps, plus the
    // array layer when arrayed).
    if (geom1d)
        base_comps = arrayed ? 3 : 2;

    uint32_t lod = const_f32(c, 0.0f);
    if (in->tex_mode == 1)
        lod = tex_scalar(c, b, in->tex_lod, in->ty2, in->tex_csign, 1, line);

    if (ms) {
        uint32_t cvec = tex_coord_vector(c, b, in, dim, arrayed, 1, 0, line);
        uint32_t sample = tex_scalar(c, b, in->tex_coord[0], PTY_U32, 0, 0,
                                     line);
        uint32_t rty = in->ty == PTY_F32 ? c->t_v4f32 : c->t_uvec4;
        uint32_t res = nid(c);
        put(c, b, (7u << 16) | S_ImageFetch);
        put(c, b, rty);
        put(c, b, res);
        put(c, b, img);
        put(c, b, cvec);
        put(c, b, 0x40u); // ImageOperands Sample
        put(c, b, sample);
        if (c->failed)
            return -1;
        for (int i = 0; i < 4; i++) {
            uint32_t ex = nid(c);
            put(c, b, (5u << 16) | S_CompositeExtract);
            put(c, b, in->ty == PTY_F32 ? c->t_f32 : c->t_u32);
            put(c, b, ex);
            put(c, b, res);
            put(c, b, (uint32_t)i);
            def_val(c, in->tex_dst[i], in->ty, ex);
        }
        return c->failed ? -1 : 0;
    }

    if (geom1d && in->ty2 != PTY_F32) {
        // Integer-addressed 1D fetch (tex1Dfetch): unfiltered.
        // Use OpImageFetch on the tiled 2D image.
        uint32_t cvec = tex_coord_vector_1d_tiled(c, b, in, binding, 0, line);
        uint32_t rty = in->ty == PTY_F32 ? c->t_v4f32 : c->t_uvec4;
        uint32_t res = nid(c);
        if (in->tex_mode == 1) {
            uint32_t ilod = tex_scalar(c, b, in->tex_lod, in->ty2, in->tex_csign, 0, line);
            put(c, b, (7u << 16) | S_ImageFetch);
            put(c, b, rty);
            put(c, b, res);
            put(c, b, img ? img : simg);
            put(c, b, cvec);
            put(c, b, 2u); // Lod
            put(c, b, ilod);
        } else {
            put(c, b, (5u << 16) | S_ImageFetch);
            put(c, b, rty);
            put(c, b, res);
            put(c, b, img ? img : simg);
            put(c, b, cvec);
        }
        if (c->failed)
            return -1;
        uint32_t comp[4];
        for (int i = 0; i < 4; i++) {
            uint32_t ex = nid(c);
            put(c, b, (5u << 16) | S_CompositeExtract);
            put(c, b, in->ty == PTY_F32 ? c->t_f32 : c->t_u32);
            put(c, b, ex);
            put(c, b, res);
            put(c, b, (uint32_t)i);
            comp[i] = ex;
        }
        tex_def_result(c, b, in, comp, line);
        return c->failed ? -1 : 0;
    }

    uint32_t cvec = geom1d
        ? tex_coord_vector_1d_tiled(c, b, in, binding, 1, line)
        : tex_coord_vector(c, b, in, dim, arrayed, 0, 1, line);
    texops_t ops;
    if (tex_build_operands(c, b, in, dim, lod, base_comps, 1, 0, &ops,
                           line) != 0)
        return -1;
    if (in->tex_compare) {
        uint32_t dref = tex_scalar(c, b, in->tex_cmp, PTY_F32, 0, 1, line);
        uint32_t res = nid(c);
        uint32_t wcx = 6u + (ops.mask ? 1u + (uint32_t)ops.nw : 0u);
        put(c, b, (wcx << 16) | S_ImageSampleDrefExplicitLod);
        put(c, b, c->t_f32);
        put(c, b, res);
        put(c, b, simg);
        put(c, b, cvec);
        put(c, b, dref);
        tex_emit_operands(c, b, &ops);
        if (c->failed)
            return -1;
        uint32_t comp[4] = { res, res, res, res };
        tex_def_result(c, b, in, comp, line);
        return 0;
    }
    uint32_t rty = in->ty == PTY_F32 ? c->t_v4f32 : c->t_uvec4;
    uint32_t res = nid(c);
    uint32_t wcx = 5u + (ops.mask ? 1u + (uint32_t)ops.nw : 0u);
    put(c, b, (wcx << 16) | S_ImageSampleExplicitLod);
    put(c, b, rty);
    put(c, b, res);
    put(c, b, simg);
    put(c, b, cvec);
    tex_emit_operands(c, b, &ops);
    if (c->failed)
        return -1;
    uint32_t comp[4];
    for (int i = 0; i < 4; i++) {
        uint32_t ex = nid(c);
        put(c, b, (5u << 16) | S_CompositeExtract);
        put(c, b, in->ty == PTY_F32 ? c->t_f32 : c->t_u32);
        put(c, b, ex);
        put(c, b, res);
        put(c, b, (uint32_t)i);
        comp[i] = ex;
    }
    tex_def_result(c, b, in, comp, line);
    return c->failed ? -1 : 0;
}

static int emit_image_tld4(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in,
                           int binding, int line) {
    if (in->tex_dstp[0])
        def_val(c, in->tex_dstp, PTY_PRED, tex_bool_const(c, 1));
    int dim = c->tex_dim[binding];
    int arrayed = c->tex_arrayed[binding];
    uint32_t simg = 0, img = 0;
    if (tex_load_image(c, b, binding, in->tex_sampler, line, &simg,
                       &img) != 0)
        return -1;
    (void)img;
    int geom1d = (in->tex_geom == 1 || in->tex_geom == 4);
    uint32_t cvec = geom1d
        ? tex_coord_vector_1d_tiled(c, b, in, binding, 1, line)
        : tex_coord_vector(c, b, in, dim, arrayed, 0, 1, line);
    int base_comps = dim == 2 ? 3 : (dim == 0 ? 1 : 3);
    if (dim == 1)
        base_comps = 2;
    if (geom1d)
        base_comps = 2;
    texops_t ops;
    if (tex_build_operands(c, b, in, dim, const_f32(c, 0.0f), base_comps, 0,
                           1, &ops, line) != 0)
        return -1;
    uint32_t rty = in->ty == PTY_F32 ? c->t_v4f32 : c->t_uvec4;
    uint32_t res = nid(c);
    uint32_t wcx = 6u + (ops.mask ? 1u + (uint32_t)ops.nw : 0u);
    if (in->tex_compare) {
        uint32_t dref = tex_scalar(c, b, in->tex_cmp, PTY_F32, 0, 1, line);
        put(c, b, (wcx << 16) | S_ImageDrefGather);
        put(c, b, c->t_v4f32);
        put(c, b, res);
        put(c, b, simg);
        put(c, b, cvec);
        put(c, b, dref);
    } else {
        put(c, b, (wcx << 16) | S_ImageGather);
        put(c, b, rty);
        put(c, b, res);
        put(c, b, simg);
        put(c, b, cvec);
        put(c, b, const_u32(c, (uint32_t)in->tex_comp));
    }
    tex_emit_operands(c, b, &ops);
    if (c->failed)
        return -1;
    for (int i = 0; i < 4; i++) {
        uint32_t ex = nid(c);
        put(c, b, (5u << 16) | S_CompositeExtract);
        put(c, b, (in->ty == PTY_F32 || in->tex_compare) ? c->t_f32
                                                         : c->t_u32);
        put(c, b, ex);
        put(c, b, res);
        put(c, b, (uint32_t)i);
        def_val(c, in->tex_dst[i],
                in->tex_compare ? PTY_F32 : in->ty, ex);
    }
    return c->failed ? -1 : 0;
}

// ---- legacy buffer-backed descriptors ---------------------------------

static uint32_t tex_hdr(ctx_t *c, spv_blob_t *b, int binding,
                        uint32_t base, uint32_t byteoff, int line) {
    (void)line;
    uint32_t o = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u64, o, base,
               const_u64(c, (uint64_t)byteoff));
    uint32_t ptr = addr_ptr(c, b, binding, o);
    if (c->failed)
        return 0;
    uint32_t v = nid(c);
    emit_load(c, b, c->t_u32, v, ptr);
    return v;
}

static uint32_t tex_floor(ctx_t *c, spv_blob_t *b, uint32_t f, int line) {
    (void)line;
    uint32_t id = nid(c);
    put(c, b, (6u << 16) | S_ExtInst);
    put(c, b, c->t_f32);
    put(c, b, id);
    put(c, b, c->glsl450);
    put(c, b, 8); // Floor
    put(c, b, f);
    return id;
}

static uint32_t tex_sel_f32(ctx_t *c, spv_blob_t *b, uint32_t cond,
                            uint32_t a, uint32_t b2, int line) {
    uint32_t id = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, c->t_f32);
    put(c, b, id);
    put(c, b, cond);
    put(c, b, a);
    put(c, b, b2);
    (void)line;
    return id;
}

static uint32_t tex_sel_u32(ctx_t *c, spv_blob_t *b, uint32_t cond,
                            uint32_t a, uint32_t b2, int line) {
    uint32_t id = nid(c);
    put(c, b, (6u << 16) | S_Select);
    put(c, b, c->t_u32);
    put(c, b, id);
    put(c, b, cond);
    put(c, b, a);
    put(c, b, b2);
    (void)line;
    return id;
}

// Clamp/wrap/mirror an integer coordinate to [0, extent).
static uint32_t tex_wrap_coord(ctx_t *c, spv_blob_t *b, uint32_t ix,
                               uint32_t extent, uint32_t mode, int line) {
    // clamp_to_edge (2) and border (3) clamp here; border is treated
    // as edge (the descriptor model has no border color).
    uint32_t clampm = do_cmp(c, b, PCM_GE, PTY_U32, 1, mode,
                             const_u32(c, 2), line);
    uint32_t zero = const_u32(c, 0);
    uint32_t one = const_u32(c, 1);
    uint32_t extm1 = nid(c);
    emit_arith(c, b, S_ISub, c->t_u32, extm1, extent, one);
    uint32_t ge0 = do_cmp(c, b, PCM_GE, PTY_U32, 1, ix, zero, line);
    uint32_t cl = tex_sel_u32(c, b, ge0, ix, zero, line);
    uint32_t le = do_cmp(c, b, PCM_LE, PTY_U32, 1, cl, extm1, line);
    uint32_t clamped = tex_sel_u32(c, b, le, cl, extm1, line);

    // wrap/mirror for nonnegative inputs; add one period first so
    // small negative coordinates behave.
    uint32_t per = extent;
    uint32_t pos = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u32, pos, ix, per);
    uint32_t wrapped = nid(c);
    emit_arith(c, b, S_UMod, c->t_u32, wrapped, pos, per);
    uint32_t per2 = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, per2, per, const_u32(c, 1));
    uint32_t pos2 = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u32, pos2, ix, per2);
    uint32_t m = nid(c);
    emit_arith(c, b, S_UMod, c->t_u32, m, pos2, per2);
    uint32_t over = do_cmp(c, b, PCM_GE, PTY_U32, 1, m, per, line);
    // rev = 2*extent-1 - m
    uint32_t per2m1 = nid(c);
    emit_arith(c, b, S_ISub, c->t_u32, per2m1, per2, one);
    uint32_t rev = nid(c);
    emit_arith(c, b, S_ISub, c->t_u32, rev, per2m1, m);
    uint32_t mirrored = tex_sel_u32(c, b, over, rev, m, line);
    uint32_t mirror = do_cmp(c, b, PCM_EQ, PTY_U32, 0, mode,
                             const_u32(c, 1), line);
    uint32_t soft = tex_sel_u32(c, b, mirror, mirrored, wrapped, line);
    return tex_sel_u32(c, b, clampm, clamped, soft, line);
}

static void tex_fetch(ctx_t *c, spv_blob_t *b, int binding, uint32_t off,
                      uint32_t ix, uint32_t iy, uint32_t pitch,
                      uint32_t data_off, ptx_ty_t ty, int line) {
    uint32_t row = nid(c);
    emit_arith(c, b, S_IMul, c->t_u32, row, iy, pitch);
    uint32_t col = nid(c);
    emit_arith(c, b, S_Shl, c->t_u32, col, ix, const_u32(c, 3));
    uint32_t ar = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u32, ar, data_off, row);
    uint32_t addr = nid(c);
    emit_arith(c, b, S_IAdd, c->t_u32, addr, ar, col);
    for (int j = 0; j < 4; j++) {
        uint32_t add = addr;
        if (j) {
            add = nid(c);
            emit_arith(c, b, S_IAdd, c->t_u32, add, addr,
                       const_u32(c, (uint32_t)j * 4u));
        }
        uint32_t o64 = nid(c);
        put(c, b, (4u << 16) | S_UConvert);
        put(c, b, c->t_u64);
        put(c, b, o64);
        put(c, b, add);
        uint32_t base = nid(c);
        emit_arith(c, b, S_IAdd, c->t_u64, base, off, o64);
        uint32_t ptr = addr_ptr(c, b, binding, base);
        uint32_t v = nid(c);
        emit_load(c, b, c->t_u32, v, ptr);
        uint32_t outv = v;
        if (ty == PTY_F32) {
            outv = nid(c);
            put(c, b, (4u << 16) | S_Bitcast);
            put(c, b, c->t_f32);
            put(c, b, outv);
            put(c, b, v);
        }
        def_val(c, c->tex_dst_tmp[j], ty, outv);
    }
    (void)line;
}

int emit_texture_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    if (in->op == POP_ISTYPEP) {
        if (check_dst(c, in->dst, PTY_PRED, line) != 0)
            return -1;
        int is_tex = 0;
        reg_t *r = find_reg(c, in->tex_ref);
        if (r && r->defined)
            is_tex = r->is_tex;
        else {
            cparam_t *cp = find_cparam(c, in->tex_ref);
            if (cp)
                is_tex = cp->is_tex;
        }
        int want = in->istypep_kind + 1; // texref=1, samplerref=2
        uint32_t v = tex_bool_const(c, in->istypep_kind == 2 ? 0 :
                                      (is_tex == want));
        def_val(c, in->dst, PTY_PRED, v);
        return 0;
    }
    int binding = -1, is_tex = 0;
    uint32_t off = 0;
    if (tex_ref_binding(c, in->tex_ref, &binding, &off, &is_tex, line) != 0)
        return -1;
    if (binding >= 0 && binding < MAX_PARAMS &&
        (c->tex_vars[binding] || c->tex_sampler_var[binding])) {
        if (in->op == POP_TXQ)
            return emit_image_txq(c, b, in, binding, line);
        if (c->tex_vars[binding]) {
            if (in->op == POP_TLD4)
                return emit_image_tld4(c, b, in, binding, line);
            if (in->op == POP_TEX)
                return emit_image_tex(c, b, in, binding, line);
        }
        failf(c, line, "texture instruction on a non-texture binding");
        return -1;
    }
    if (in->op == POP_TXQ) {
        uint32_t v;
        switch (in->tex_query) {
        case 0: v = tex_hdr(c, b, binding, off, 4, line); break;
        case 1: v = tex_hdr(c, b, binding, off, 8, line); break;
        case 2: v = tex_hdr(c, b, binding, off, 12, line); break;
        case 3: v = tex_hdr(c, b, binding, off, 48, line); break;
        case 4: v = tex_hdr(c, b, binding, off, 44, line); break;
        case 5: {
            uint32_t f = tex_hdr(c, b, binding, off, 16, line);
            v = nid(c);
            emit_arith(c, b, S_And, c->t_u32, v, f, const_u32(c, 1));
            break;
        }
        case 6: v = const_u32(c, 0); break;
        case 7: v = tex_hdr(c, b, binding, off, 52, line); break;
        case 8: v = tex_hdr(c, b, binding, off, 56, line); break;
        case 9: {
            uint32_t f = tex_hdr(c, b, binding, off, 16, line);
            uint32_t t = nid(c);
            emit_arith(c, b, S_ShrL, c->t_u32, t, f,
                       const_u32(c, TEXF_FILTER_SHIFT));
            v = nid(c);
            emit_arith(c, b, S_And, c->t_u32, v, t, const_u32(c, 1));
            break;
        }
        case 10: v = tex_hdr(c, b, binding, off, 20, line); break;
        case 11: v = tex_hdr(c, b, binding, off, 24, line); break;
        case 12: v = tex_hdr(c, b, binding, off, 28, line); break;
        default: v = const_u32(c, 0); break;
        }
        if (c->failed)
            return -1;
        if (check_dst(c, in->dst, PTY_U32, line) != 0)
            return -1;
        def_val(c, in->dst, PTY_U32, v);
        return 0;
    }
    if (in->op != POP_TEX)
        return -1;
    if (in->tex_geom != 2 || in->tex_ncoord != 2 || in->tex_nchan != 4) {
        failf(c, line, "buffer-backed textures support only tex.2d.v4");
        return -1;
    }
    for (int i = 0; i < 4; i++)
        snprintf(c->tex_dst_tmp[i], sizeof(c->tex_dst_tmp[i]), "%s",
                 in->tex_dst[i]);

    uint32_t flags = tex_hdr(c, b, binding, off, 16, line);
    uint32_t width = tex_hdr(c, b, binding, off, 4, line);
    uint32_t height = tex_hdr(c, b, binding, off, 8, line);
    uint32_t addr0 = tex_hdr(c, b, binding, off, 20, line);
    uint32_t addr1 = tex_hdr(c, b, binding, off, 24, line);
    uint32_t data_off = tex_hdr(c, b, binding, off, 32, line);
    uint32_t pitch = tex_hdr(c, b, binding, off, 36, line);
    if (c->failed)
        return -1;
    uint32_t norm = nid(c);
    emit_arith(c, b, S_And, c->t_u32, norm, flags, const_u32(c, 1));

    ptx_opd_t co[2];
    for (int i = 0; i < 2; i++) {
        memset(&co[i], 0, sizeof(co[i]));
        co[i].kind = OPD_REG;
        snprintf(co[i].reg, sizeof(co[i].reg), "%s", in->tex_coord[i]);
    }
    val_t cs = resolve(c, b, &co[0], in->ty2, line);
    val_t ct = resolve(c, b, &co[1], in->ty2, line);
    if (c->failed)
        return -1;
    uint32_t ix, iy;
    if (in->ty2 == PTY_F32) {
        uint32_t wf = nid(c);
        put(c, b, (4u << 16) | S_ConvertSToF);
        put(c, b, c->t_f32);
        put(c, b, wf);
        put(c, b, width);
        uint32_t hf = nid(c);
        put(c, b, (4u << 16) | S_ConvertSToF);
        put(c, b, c->t_f32);
        put(c, b, hf);
        put(c, b, height);
        uint32_t sx, sy;
        uint32_t nzb = do_cmp(c, b, PCM_NE, PTY_U32, 0, norm,
                              const_u32(c, 0), line);
        uint32_t xm = nid(c), ym = nid(c);
        emit_arith(c, b, S_FMul, c->t_f32, xm, cs.id, wf);
        emit_arith(c, b, S_FMul, c->t_f32, ym, ct.id, hf);
        sx = tex_sel_f32(c, b, nzb, xm, cs.id, line);
        sy = tex_sel_f32(c, b, nzb, ym, ct.id, line);
        uint32_t fx = tex_floor(c, b, sx, line);
        uint32_t fy = tex_floor(c, b, sy, line);
        ix = nid(c);
        put(c, b, (4u << 16) | S_ConvertFToS);
        put(c, b, c->t_u32);
        put(c, b, ix);
        put(c, b, fx);
        iy = nid(c);
        put(c, b, (4u << 16) | S_ConvertFToS);
        put(c, b, c->t_u32);
        put(c, b, iy);
        put(c, b, fy);
    } else {
        ix = cs.id;
        iy = ct.id;
    }
    uint32_t ax = tex_wrap_coord(c, b, ix, width, addr0, line);
    uint32_t ay = tex_wrap_coord(c, b, iy, height, addr1, line);
    tex_fetch(c, b, binding, off, ax, ay, pitch, data_off, in->ty, line);
    return c->failed ? -1 : 0;
}
