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

void emit_head(ctx_t *c) {
    // Capabilities, memory model, scalar/vector types. Constants may be
    // created after this (they must never be appended mid-instruction).
    emit1(c, &c->head, S_Capability, C_Shader);
    emit1(c, &c->head, S_Capability, C_Int64);
    // Demoted modules must not declare Float64 (they run without the
    // feature, and the launch gate keys off this capability). Neither
    // should f32-only modules: the capability would needlessly require
    // shaderFloat64 at pipeline creation.
    if (c->uses_f64 && !c->emu_f64)
        emit1(c, &c->head, S_Capability, C_Float64);
    if (c->uses_f16_native) {
        emit1(c, &c->head, S_Capability, C_Float16);
        emit1(c, &c->head, S_Capability, C_Int16);
    }
    if (c->caps.coopmat_f16 || c->caps.coopmat_bf16) {
        emit1(c, &c->head, S_Capability, 6022); // CooperativeMatrixKHR
        if (c->caps.coopmat_bf16) {
            emit1(c, &c->head, S_Capability, 5116); // BFloat16TypeKHR
            emit1(c, &c->head, S_Capability, 5118); // BFloat16CooperativeMatrixKHR
        }
    }
    if (c->uses_fdot2)
        emit1(c, &c->head, S_Capability,
              C_DotProductFloat16AccFloat32VALVE);
    // Shuffle capability only when shfl is actually used (like
    // Float64 above: no needless device requirements).
    if (c->uses_shfl)
        emit1(c, &c->head, S_Capability, C_GroupShuffle);
    if (c->uses_ballot)
        emit1(c, &c->head, S_Capability, C_GroupBallot);
    if (c->uses_redux || c->uses_state_lockstep) {
        emit1(c, &c->head, S_Capability, C_GroupArith);
        if (c->uses_redux)
            emit1(c, &c->head, S_Capability, C_GroupClustered);
    }
    if (c->uses_image_query)
        emit1(c, &c->head, S_Capability, C_ImageQuery);
    if (c->surf_table) {
        emit1(c, &c->head, S_Capability, C_ShaderNonUniform);
        emit1(c, &c->head, S_Capability, C_RuntimeDescriptorArray);
        emit1(c, &c->head, S_Capability,
              C_StorageImageArrayNonUniformIndexing);
    }
    if (c->uses_sampled1d)
        emit1(c, &c->head, S_Capability, C_Sampled1D);
    if (c->uses_image1d)
        emit1(c, &c->head, S_Capability, C_Image1D);
    if (c->uses_cube_array)
        emit1(c, &c->head, S_Capability, C_SampledCubeArray);
    if (c->uses_gather_ext)
        emit1(c, &c->head, S_Capability, C_ImageGatherExtended);
    if (c->uses_i64_atomics || c->uses_int64_image_atomic)
        emit1(c, &c->head, S_Capability, C_Int64Atomics);
    if (c->uses_int64_image)
        emit1(c, &c->head, S_Capability, C_Int64ImageEXT);
    if (c->uses_atomic_f32_add)
        emit1(c, &c->head, S_Capability, C_AtomicFloat32AddEXT);
    if (c->uses_atomic_f64_add)
        emit1(c, &c->head, S_Capability, C_AtomicFloat64AddEXT);
    if (c->uses_atomic_f32_minmax)
        emit1(c, &c->head, S_Capability, C_AtomicFloat32MinMaxEXT);
    if (c->use_fma || c->use_fma_f64) {
        // Directed rounding takes the guaranteed-fused path only when
        // the device advertises VK_KHR_shader_fma.
        emit1(c, &c->head, S_Capability, C_FmaKHR);
    }
    if (c->uses_shader_clock)
        emit1(c, &c->head, S_Capability, C_ShaderClockKHR);
    if (c->use_dot_product) {
        // dp4a uses the packed 4x8-bit integer dot product. Both
        // instructions and capabilities are core in SPIR-V 1.6.
        emit1(c, &c->head, S_Capability, C_DotProduct);
        emit1(c, &c->head, S_Capability, C_DotProductInput4x8BitPacked);
    }
    if (c->use_ssbo8)
        emit1(c, &c->head, S_Capability, C_StorageBuffer8BitAccess);
    if (c->use_ssbo16) {
        emit1(c, &c->head, S_Capability, C_Int16);
        emit1(c, &c->head, S_Capability, C_StorageBuffer16BitAccess);
    }
    if (c->use_fc32 || c->use_fc64) {
        emit1(c, &c->head, S_Capability, C_DenormPreserve);
        emit1(c, &c->head, S_Capability, C_SignedZeroInfNanPreserve);
    }
    if (c->use_vmm) {
        emit1(c, &c->head, S_Capability, C_VulkanMemoryModel);
        emit1(c, &c->head, S_Capability, C_VulkanMemoryModelDeviceScope);
    }
    if (need_phys_addr(c))
        emit1(c, &c->head, S_Capability, C_PhysicalStorageBufferAddresses);

    if (c->uses_fdot2) {
        const char *fext = "SPV_VALVE_mixed_float_dot_product";
        size_t fwords = (strlen(fext) + 1 + 3) / 4;
        put(c, &c->head, (uint32_t)((1 + fwords) << 16) | S_Extension);
        emit_str(c, &c->head, fext);
    }
    if (c->caps.coopmat_f16 || c->caps.coopmat_bf16) {
        const char *cext = "SPV_KHR_cooperative_matrix";
        size_t cwords = (strlen(cext) + 1 + 3) / 4;
        put(c, &c->head, (uint32_t)((1 + cwords) << 16) | S_Extension);
        emit_str(c, &c->head, cext);
    }
    if (c->caps.coopmat_bf16) {
        const char *bext = "SPV_KHR_bfloat16";
        size_t bwords = (strlen(bext) + 1 + 3) / 4;
        put(c, &c->head, (uint32_t)((1 + bwords) << 16) | S_Extension);
        emit_str(c, &c->head, bext);
    }
    // Extension strings must follow every OpCapability (SPIR-V layout).
    if (c->surf_table) {
        const char *dext = "SPV_EXT_descriptor_indexing";
        size_t dwords = (strlen(dext) + 1 + 3) / 4;
        put(c, &c->head, (uint32_t)((1 + dwords) << 16) | S_Extension);
        emit_str(c, &c->head, dext);
    }
    if (c->use_fma || c->use_fma_f64) {
        const char *ename = "SPV_KHR_fma";
        size_t ewords = (strlen(ename) + 1 + 3) / 4;
        put(c, &c->head, (uint32_t)((1 + ewords) << 16) | S_Extension);
        emit_str(c, &c->head, ename);
    }
    if (c->uses_atomic_f32_add || c->uses_atomic_f64_add) {
        const char *aext = "SPV_EXT_shader_atomic_float_add";
        size_t awords = (strlen(aext) + 1 + 3) / 4;
        put(c, &c->head, (uint32_t)((1 + awords) << 16) | S_Extension);
        emit_str(c, &c->head, aext);
    }
    if (c->uses_atomic_f32_minmax) {
        const char *mext = "SPV_EXT_shader_atomic_float_min_max";
        size_t mwords = (strlen(mext) + 1 + 3) / 4;
        put(c, &c->head, (uint32_t)((1 + mwords) << 16) | S_Extension);
        emit_str(c, &c->head, mext);
    }
    if (c->uses_int64_image) {
        const char *iext = "SPV_EXT_shader_image_int64";
        size_t iwords = (strlen(iext) + 1 + 3) / 4;
        put(c, &c->head, (uint32_t)((1 + iwords) << 16) | S_Extension);
        emit_str(c, &c->head, iext);
    }
    if (c->uses_shader_clock) {
        const char *sck = "SPV_KHR_shader_clock";
        size_t skwords = (strlen(sck) + 1 + 3) / 4;
        put(c, &c->head, (uint32_t)((1 + skwords) << 16) | S_Extension);
        emit_str(c, &c->head, sck);
    }
    // GLSL.std.450 extended instruction set (transcendentals): always
    // imported, like glslc does (unused imports validate cleanly).
    c->glsl450 = nid(c);
    {
        const char *gname = "GLSL.std.450";
        size_t gwords = (strlen(gname) + 1 + 3) / 4;
        put(c, &c->head, (uint32_t)((2 + gwords) << 16) | S_ExtInstImport);
        put(c, &c->head, c->glsl450);
        emit_str(c, &c->head, gname);
    }
    put(c, &c->head, (3u << 16) | S_MemoryModel);
    // PhysicalStorageBuffer64 (5348) is required for OpConvertUToPtr and
    // PhysicalStorageBuffer pointers; descriptor/Workgroup/Private
    // pointers remain valid under it.
    put(c, &c->head, need_phys_addr(c) ? 5348u : 0u); // Logical / PhysicalStorageBuffer64
    put(c, &c->head, c->use_vmm ? 3u : 1u); // Vulkan / GLSL450

    c->t_void = nid(c);
    emit1(c, &c->types, S_TypeVoid, c->t_void);
    c->t_bool = nid(c);
    emit1(c, &c->types, S_TypeBool, c->t_bool);
    c->t_u32 = nid(c);
    put(c, &c->types, (4u << 16) | S_TypeInt);
    put(c, &c->types, c->t_u32);
    put(c, &c->types, 32);
    put(c, &c->types, 0);
    c->t_u64 = nid(c);
    put(c, &c->types, (4u << 16) | S_TypeInt);
    put(c, &c->types, c->t_u64);
    put(c, &c->types, 64);
    put(c, &c->types, 0);
    c->t_f32 = nid(c);
    put(c, &c->types, (3u << 16) | S_TypeFloat);
    put(c, &c->types, c->t_f32);
    put(c, &c->types, 32);
    if (c->uses_f64 && !c->emu_f64) {
        c->t_f64 = nid(c);
        put(c, &c->types, (3u << 16) | S_TypeFloat);
        put(c, &c->types, c->t_f64);
        put(c, &c->types, 64);
    }
    c->t_v3u32 = nid(c);
    put(c, &c->types, (4u << 16) | S_TypeVector);
    put(c, &c->types, c->t_v3u32);
    put(c, &c->types, c->t_u32);
    put(c, &c->types, 3);
    // Ballot result type (always a 4-word vector per the spec).
    c->t_uvec4 = nid(c);
    put(c, &c->types, (4u << 16) | S_TypeVector);
    put(c, &c->types, c->t_uvec4);
    put(c, &c->types, c->t_u32);
    put(c, &c->types, 4);
    c->t_ptr_in_v3 = nid(c);
    put(c, &c->types, (4u << 16) | S_TypePointer);
    put(c, &c->types, c->t_ptr_in_v3);
    put(c, &c->types, SC_Input);
    put(c, &c->types, c->t_v3u32);
}

// Geometry code -> SPIR-V image shape.
// 1D textures (1d/a1d) are represented as 2D tiled images so large
// linear texrefs can exceed maxImageDimension1D. The shader converts
// 1D coordinates using the tile stride in tex metadata (word 6).
int tex_geom_dim(int g) {
    switch (g) {
    case 1: case 4: return 1;   // 1d / a1d -> 2D tiled
    case 2: case 5: case 8: case 9: return 1; // 2d / a2d / 2dms / a2dms
    case 3: return 2;           // 3d
    case 6: case 7: return 3;   // cube / acube
    }
    return 1;
}

int tex_geom_arrayed(int g) {
    // a1d (4) is a tiled 2D array image (one tile grid per layer); the
    // array layer travels as the final coordinate component.
    return g == 4 || g == 5 || g == 7 || g == 9;
}

int tex_geom_ms(int g) {
    return g == 8 || g == 9;
}

// True when instruction `ti` names texture parameter `name`, either
// directly or through `mov.u64 %rd, name`.
int tex_use_matches(ctx_t *c, const ptx_instr_t *ti, const char *name) {
    if (strcmp(ti->tex_ref, name) == 0)
        return 1;
    for (size_t bi = 0; bi < c->cur_fn->nblocks; bi++)
        for (size_t ii = 0; ii < c->cur_fn->blocks[bi].count; ii++) {
            const ptx_instr_t *m = &c->cur_fn->blocks[bi].instrs[ii];
            if (m->op == POP_MOV && strcmp(m->dst, ti->tex_ref) == 0 &&
                m->nsrc > 0 && m->src[0].kind == OPD_PARAM &&
                strcmp(m->src[0].param, name) == 0)
                return 1;
        }
    return 0;
}

// Derive the image shape and sampled type from the parameter's uses.
int tex_shape_scan(ctx_t *c, const char *name, int *geom,
                          int *sty_int, int line) {
    *geom = 2; // default 2D
    *sty_int = 0;
    int seen = 0;
    for (size_t bi = 0; bi < c->cur_fn->nblocks; bi++)
        for (size_t ii = 0; ii < c->cur_fn->blocks[bi].count; ii++) {
            const ptx_instr_t *ti = &c->cur_fn->blocks[bi].instrs[ii];
            if ((ti->op != POP_TEX && ti->op != POP_TLD4) ||
                !tex_use_matches(c, ti, name))
                continue;
            if (!seen) {
                *geom = ti->tex_geom;
                *sty_int = ti->ty != PTY_F32;
                seen = 1;
            } else if (*geom != ti->tex_geom) {
                char msg[160];
                snprintf(msg, sizeof(msg),
                         "texture parameter %s is used with conflicting "
                         "geometries", name);
                failf(c, line, msg);
                return -1;
            } else if ((ti->ty != PTY_F32) != *sty_int) {
                char msg[160];
                snprintf(msg, sizeof(msg),
                         "texture parameter %s is sampled as both float "
                         "and integer", name);
                failf(c, line, msg);
                return -1;
            }
        }
    return 0;
}

int surf_dim_of(int g) {
    switch (g) {
    case 1: case 4: return 0;
    case 2: case 5: return 1;
    case 3: return 2;
    }
    return 1;
}

int surf_arrayed_of(int g) {
    return g == 4 || g == 5;
}

int surf_ref_from_param(ctx_t *c, const char *ref, const char *name,
                               int depth) {
    if (depth > 8)
        return 0;
    for (size_t bi = 0; bi < c->cur_fn->nblocks; bi++)
        for (size_t ii = 0; ii < c->cur_fn->blocks[bi].count; ii++) {
            const ptx_instr_t *m = &c->cur_fn->blocks[bi].instrs[ii];
            if (strcmp(m->dst, ref) != 0)
                continue;
            if (m->op == POP_LD && m->space == PSP_PARAM &&
                m->src[0].kind == OPD_ADDR && !m->src[0].addr_reg[0] &&
                strcmp(m->src[0].param, name) == 0)
                return 1;
            if (m->op == POP_MOV && m->nsrc > 0) {
                if (m->src[0].kind == OPD_PARAM &&
                    strcmp(m->src[0].param, name) == 0)
                    return 1;
                if (m->src[0].kind == OPD_REG &&
                    surf_ref_from_param(c, m->src[0].reg, name, depth + 1))
                    return 1;
            }
        }
    return 0;
}

int surf_use_matches(ctx_t *c, const ptx_instr_t *ti,
                            const char *name) {
    if (strcmp(ti->surf_ref, name) == 0)
        return 1;
    return surf_ref_from_param(c, ti->surf_ref, name, 0);
}

// True when `ref` is defined by exactly one load/move chain from a
// static .surfref/ .u64 surface parameter.  Anything else (memory load,
// selp, multiple assignments) uses the dynamic surface table.
void surf_scan_dynamic(ctx_t *c) {
    c->surf_table = 0;
    for (size_t bi = 0; bi < c->cur_fn->nblocks; bi++)
        for (size_t ii = 0; ii < c->cur_fn->blocks[bi].count; ii++) {
            const ptx_instr_t *ti = &c->cur_fn->blocks[bi].instrs[ii];
            if (ti->op == POP_ELECT)
                c->uses_ballot = 1;
            if (ti->op == POP_MBARRIER)
                c->uses_i64_atomics = 1;
            if (ti->op == POP_RED_ASYNC && ti->ty == PTY_U64)
                c->uses_i64_atomics = 1;
            if (ti->op != POP_SULD && ti->op != POP_SUST &&
                ti->op != POP_SURED && ti->op != POP_SUQ)
                continue;
            if (!surf_ref_is_static(c, ti->surf_ref, 0))
                c->surf_table = 1;
        }
}

int surf_ref_is_static(ctx_t *c, const char *ref, int depth) {    if (depth > 8)
        return 0;
    int nmatch = 0;
    for (int q = 0; q < c->ncparams; q++)
        if (c->cparams[q].is_tex == 3 &&
            strcmp(c->cparams[q].name, ref) == 0)
            nmatch++;
    if (nmatch == 1)
        return 1;
    if (nmatch > 1)
        return 0;
    const ptx_instr_t *def = NULL;
    int ndef = 0;
    for (size_t bi = 0; bi < c->cur_fn->nblocks; bi++)
        for (size_t ii = 0; ii < c->cur_fn->blocks[bi].count; ii++) {
            const ptx_instr_t *m = &c->cur_fn->blocks[bi].instrs[ii];
            if (strcmp(m->dst, ref) == 0) {
                def = m;
                ndef++;
            }
        }
    if (ndef != 1 || !def)
        return 0;
    const char *pn = NULL;
    if (def->op == POP_LD && def->space == PSP_PARAM &&
        def->src[0].kind == OPD_ADDR && !def->src[0].addr_reg[0])
        pn = def->src[0].param;
    else if (def->op == POP_MOV && def->nsrc > 0) {
        if (def->src[0].kind == OPD_PARAM)
            pn = def->src[0].param;
        else if (def->src[0].kind == OPD_REG)
            return surf_ref_is_static(c, def->src[0].reg, depth + 1);
    }
    if (!pn)
        return 0;
    for (int q = 0; q < c->ncparams; q++)
        if (c->cparams[q].is_tex == 3 &&
            strcmp(c->cparams[q].name, pn) == 0)
            return 1;
    return 0;
}

int surf_shape_scan(ctx_t *c, const char *name, int *geom,
                           int *formatted, int line) {
    *geom = 2;
    *formatted = 0;
    int seen = 0;
    for (size_t bi = 0; bi < c->cur_fn->nblocks; bi++)
        for (size_t ii = 0; ii < c->cur_fn->blocks[bi].count; ii++) {
            const ptx_instr_t *ti = &c->cur_fn->blocks[bi].instrs[ii];
            if ((ti->op != POP_SULD && ti->op != POP_SUST &&
                 ti->op != POP_SURED && ti->op != POP_SUQ) ||
                !surf_use_matches(c, ti, name))
                continue;
            if (ti->op == POP_SUQ)
                continue; // queries do not constrain the geometry
            int g = ti->surf_geom ? ti->surf_geom : 2;
            if (!seen) {
                *geom = g;
                *formatted = ti->surf_formatted;
                seen = 1;
            } else if (*geom != g) {
                char msg[160];
                snprintf(msg, sizeof(msg),
                         "surface parameter %s is used with conflicting "
                         "geometries", name);
                failf(c, line, msg);
                return -1;
            } else if (*formatted != ti->surf_formatted) {
                char msg[160];
                snprintf(msg, sizeof(msg),
                         "surface parameter %s mixes formatted and "
                         "unformatted access", name);
                failf(c, line, msg);
                return -1;
            }
        }
    return 0;
}

int surf_table_add(ctx_t *c, int dim, int arrayed, int isint,
                          int cas, int lock) {
    for (int i = 0; i < c->surf_table_count; i++)
        if (c->surf_tables[i].dim == dim &&
            c->surf_tables[i].arrayed == arrayed &&
            c->surf_tables[i].isint == isint &&
            c->surf_tables[i].cas == cas &&
            c->surf_tables[i].lock == lock)
            return i;
    if (c->surf_table_count >= ACUDA_SURF_TABLE_MAX_TYPES)
        return -1;
    int idx = c->surf_table_count++;
    surf_table_t *t = &c->surf_tables[idx];
    memset(t, 0, sizeof(*t));
    t->used = 1;
    t->dim = dim;
    t->arrayed = arrayed;
    t->isint = isint;
    t->cas = cas;
    t->lock = lock;
    t->binding = 1 + idx; // descriptor set number; binding is 0
    uint32_t sty = (isint || cas) ? c->t_u32 : c->t_f32;
    uint32_t img = nid(c);
    put(c, &c->types, (9u << 16) | S_TypeImage);
    put(c, &c->types, img);
    put(c, &c->types, sty);
    put(c, &c->types, (uint32_t)dim);
    put(c, &c->types, 0); // Depth
    put(c, &c->types, (uint32_t)arrayed);
    put(c, &c->types, 0); // MS
    put(c, &c->types, 2); // Sampled: storage image
    put(c, &c->types,
        (isint || cas) ? 33u : (lock ? 3u : 1u));
    uint32_t arr = nid(c);
    put(c, &c->types, (3u << 16) | S_TypeRuntimeArray);
    put(c, &c->types, arr);
    put(c, &c->types, img);
    uint32_t ptrty = nid(c);
    put(c, &c->types, (4u << 16) | S_TypePointer);
    put(c, &c->types, ptrty);
    put(c, &c->types, SC_UniformConstant);
    put(c, &c->types, arr);
    uint32_t var = nid(c);
    put(c, &c->types, (4u << 16) | S_Variable);
    put(c, &c->types, ptrty);
    put(c, &c->types, var);
    put(c, &c->types, SC_UniformConstant);
    emit_decorate1(c, var, D_DescriptorSet, (uint32_t)t->binding);
    emit_decorate1(c, var, D_Binding, 0);
    t->img_ty = img;
    t->var = var;
    t->elem_ptr_ty = nid(c);
    put(c, &c->types, (4u << 16) | S_TypePointer);
    put(c, &c->types, t->elem_ptr_ty);
    put(c, &c->types, SC_UniformConstant);
    put(c, &c->types, img);
    return idx;
}

void emit_prelude(ctx_t *c) {
    // Shared SSBO layout: struct { u32 words[] }.
    c->t_wordarr = nid(c);
    put(c, &c->types, (3u << 16) | S_TypeRuntimeArray);
    put(c, &c->types, c->t_wordarr);
    put(c, &c->types, c->t_u32);
    emit_decorate1(c, c->t_wordarr, D_ArrayStride, 4);
    c->t_ssbo = nid(c);
    put(c, &c->types, (3u << 16) | S_TypeStruct);
    put(c, &c->types, c->t_ssbo);
    put(c, &c->types, c->t_wordarr);
    emit_decorate(c, c->t_ssbo, D_Block);
    emit_member_decorate(c, c->t_ssbo, 0, D_Offset, 0);
    c->t_ptr_ssbo = nid(c);
    put(c, &c->types, (4u << 16) | S_TypePointer);
    put(c, &c->types, c->t_ptr_ssbo);
    put(c, &c->types, SC_StorageBuffer);
    put(c, &c->types, c->t_ssbo);
    c->t_ptr_u_ssbo = nid(c);
    put(c, &c->types, (4u << 16) | S_TypePointer);
    put(c, &c->types, c->t_ptr_u_ssbo);
    put(c, &c->types, SC_StorageBuffer);
    put(c, &c->types, c->t_u32);

    // Push-constant layout for scalar params: struct { u32 words[N] }.
    if (c->npush > 0) {
        c->t_pusharr = nid(c);
        put(c, &c->types, (4u << 16) | S_TypeArray);
        put(c, &c->types, c->t_pusharr);
        put(c, &c->types, c->t_u32);
        put(c, &c->types, c->push_len);
        emit_decorate1(c, c->t_pusharr, D_ArrayStride, 4);
        c->t_push = nid(c);
        put(c, &c->types, (3u << 16) | S_TypeStruct);
        put(c, &c->types, c->t_push);
        put(c, &c->types, c->t_pusharr);
        emit_decorate(c, c->t_push, D_Block);
        emit_member_decorate(c, c->t_push, 0, D_Offset, 0);
        c->t_ptr_push = nid(c);
        put(c, &c->types, (4u << 16) | S_TypePointer);
        put(c, &c->types, c->t_ptr_push);
        put(c, &c->types, SC_PushConstant);
        put(c, &c->types, c->t_push);
        c->t_ptr_u_push = nid(c);
        put(c, &c->types, (4u << 16) | S_TypePointer);
        put(c, &c->types, c->t_ptr_u_push);
        put(c, &c->types, SC_PushConstant);
        put(c, &c->types, c->t_u32);
        c->push_var = nid(c);
        put(c, &c->types, (4u << 16) | S_Variable);
        put(c, &c->types, c->t_ptr_push);
        put(c, &c->types, c->push_var);
        put(c, &c->types, SC_PushConstant);
    }

    if (need_phys_addr(c)) {
        c->t_ptr_phys_u32 = nid(c);
        put(c, &c->types, (4u << 16) | S_TypePointer);
        put(c, &c->types, c->t_ptr_phys_u32);
        put(c, &c->types, SC_PhysicalStorageBuffer);
        put(c, &c->types, c->t_u32);
        c->t_ptr_phys_u64 = nid(c);
        put(c, &c->types, (4u << 16) | S_TypePointer);
        put(c, &c->types, c->t_ptr_phys_u64);
        put(c, &c->types, SC_PhysicalStorageBuffer);
        put(c, &c->types, c->t_u64);
        c->t_ptr_phys_f32 = nid(c);
        put(c, &c->types, (4u << 16) | S_TypePointer);
        put(c, &c->types, c->t_ptr_phys_f32);
        put(c, &c->types, SC_PhysicalStorageBuffer);
        put(c, &c->types, c->t_f32);
        c->t_ptr_phys_f64 = 0;
        if (c->t_f64) {
            c->t_ptr_phys_f64 = nid(c);
            put(c, &c->types, (4u << 16) | S_TypePointer);
            put(c, &c->types, c->t_ptr_phys_f64);
            put(c, &c->types, SC_PhysicalStorageBuffer);
            put(c, &c->types, c->t_f64);
        }
    }

    c->t_v2f32 = nid(c);
    put(c, &c->types, (4u << 16) | S_TypeVector);
    put(c, &c->types, c->t_v2f32);
    put(c, &c->types, c->t_f32);
    put(c, &c->types, 2);
    c->t_v3f32 = nid(c);
    put(c, &c->types, (4u << 16) | S_TypeVector);
    put(c, &c->types, c->t_v3f32);
    put(c, &c->types, c->t_f32);
    put(c, &c->types, 3);
    c->t_v4f32 = nid(c);
    put(c, &c->types, (4u << 16) | S_TypeVector);
    put(c, &c->types, c->t_v4f32);
    put(c, &c->types, c->t_f32);
    put(c, &c->types, 4);
    c->t_v2u32 = nid(c);
    put(c, &c->types, (4u << 16) | S_TypeVector);
    put(c, &c->types, c->t_v2u32);
    put(c, &c->types, c->t_u32);
    put(c, &c->types, 2);
    c->t_sampler_ty = nid(c);
    put(c, &c->types, (2u << 16) | S_TypeSampler);
    put(c, &c->types, c->t_sampler_ty);

    // Function type: void().
    c->t_funty = nid(c);
    put(c, &c->types, (3u << 16) | S_TypeFunction);
    put(c, &c->types, c->t_funty);
    put(c, &c->types, c->t_void);

    // Descriptor variables, one per pointer param.  Image-backed
    // .texref parameters become combined image samplers (unified mode)
    // or plain images when the kernel also declares a .samplerref
    // (independent mode).  .samplerref parameters become samplers and
    // everything else is a storage buffer.
    c->tex_split = 0;
    for (int q = 0; q < c->ncparams; q++)
        if (c->cparams[q].is_tex == 2)
            c->tex_split = 1;
    for (int i = 0; i < c->nbindings; i++) {
        cparam_t *cp = NULL;
        for (int q = 0; q < c->ncparams; q++)
            if (c->cparams[q].binding == i) {
                cp = &c->cparams[q];
                break;
            }
        c->tex_vars[i] = 0;
        c->tex_image_ty[i] = 0;
        c->tex_sampled_ty[i] = 0;
        c->tex_sampler_var[i] = 0;
        c->tex_dim[i] = 1;
        c->tex_arrayed[i] = 0;
        c->tex_ms[i] = 0;
        if (cp && cp->is_tex == 3) {
            // .surfref -> storage image (Sampled=2).
            int geom = 2, fmt = 0;
            if (surf_shape_scan(c, cp->name, &geom, &fmt, 0) != 0)
                return;
            int dim = surf_dim_of(geom);
            c->tex_dim[i] = dim;
            c->tex_arrayed[i] = surf_arrayed_of(geom);
            c->tex_ms[i] = 0;
            c->tex_int[i] = fmt ? 0 : 1;
            uint32_t sty = (fmt && !c->surf_f32_cas[i]) ? c->t_f32
                                                        : c->t_u32;
            uint32_t ifmt = !fmt ? 33u
                            : c->surf_f32_cas[i] ? 33u
                            : c->surf_f32_atomic[i] ? 3u
                                                    : 1u;
            if (!fmt && c->surf_u64[i]) {
                sty = c->t_u64;
                ifmt = 40u; // R64ui
            }
            uint32_t img_ops[7] = {sty,
                                     (uint32_t)dim,
                                     0u, // Depth
                                     (uint32_t)c->tex_arrayed[i],
                                     0u, // MS
                                     2u, // Sampled: storage image
                                     // Format is fixed so no *WithoutFormat
                                     // capability is needed: unformatted .b
                                     // surfaces are R32ui (R64ui for 64-bit),
                                     // .p surfaces Rgba32f. A surface used by
                                     // sured.add.f32 must be R32f for
                                     // OpImageTexelPointer.
                                     ifmt};
            uint32_t img = type_intern(c, S_TypeImage, img_ops, 7);
            uint32_t img_ptr_ops[2] = {SC_UniformConstant, img};
            uint32_t ptrty =
                type_intern(c, S_TypePointer, img_ptr_ops, 2);
            uint32_t var = nid(c);
            put(c, &c->types, (4u << 16) | S_Variable);
            put(c, &c->types, ptrty);
            put(c, &c->types, var);
            put(c, &c->types, SC_UniformConstant);
            emit_decorate1(c, var, D_DescriptorSet, 0);
            emit_decorate1(c, var, D_Binding, (uint32_t)i);
            c->tex_vars[i] = var;
            c->tex_image_ty[i] = img;
            c->tex_sampled_ty[i] = 0;
            c->tex_sampler_var[i] = 0;
            c->ssbo_vars[i] = 0;
            continue;
        }
        if (cp && (cp->is_tex == 1 || cp->is_tex == 2)) {
            if (cp->is_tex == 2) {
                // Sampler object (one shared pointer type: all
                // samplers are identical).
                uint32_t samp_ptr_ops[2] = {SC_UniformConstant,
                                            c->t_sampler_ty};
                uint32_t ptrty =
                    type_intern(c, S_TypePointer, samp_ptr_ops, 2);
                uint32_t var = nid(c);
                put(c, &c->types, (4u << 16) | S_Variable);
                put(c, &c->types, ptrty);
                put(c, &c->types, var);
                put(c, &c->types, SC_UniformConstant);
                emit_decorate1(c, var, D_DescriptorSet, 0);
                emit_decorate1(c, var, D_Binding, (uint32_t)i);
                c->tex_sampler_var[i] = var;
                c->ssbo_vars[i] = 0;
                continue;
            }
            int geom = 2, sty_int = 0;
            if (tex_shape_scan(c, cp->name, &geom, &sty_int, 0) != 0)
                return;
            c->tex_int[i] = sty_int;
            uint32_t sty = sty_int ? c->t_u32 : c->t_f32;
            int dim = tex_geom_dim(geom);
            c->tex_dim[i] = dim;
            c->tex_arrayed[i] = tex_geom_arrayed(geom);
            c->tex_ms[i] = tex_geom_ms(geom);
            uint32_t img_ops[7] = {sty,
                                     (uint32_t)dim,
                                     // Depth 2 ("no indication") accepts both
                                     // colour sampling and depth-compare
                                     // sampling of float textures.
                                     sty_int ? 0u : 2u,
                                     (uint32_t)c->tex_arrayed[i],
                                     (uint32_t)c->tex_ms[i], // MS
                                     1u, // Sampled
                                     0u}; // Unknown format
            uint32_t img = type_intern(c, S_TypeImage, img_ops, 7);
            uint32_t simg_ops[1] = {img};
            uint32_t simg = type_intern(c, S_TypeSampledImage, simg_ops, 1);
            uint32_t pointee = c->tex_split ? img : simg;
            uint32_t ptr_ops[2] = {SC_UniformConstant, pointee};
            uint32_t ptrty = type_intern(c, S_TypePointer, ptr_ops, 2);
            uint32_t var = nid(c);
            put(c, &c->types, (4u << 16) | S_Variable);
            put(c, &c->types, ptrty);
            put(c, &c->types, var);
            put(c, &c->types, SC_UniformConstant);
            emit_decorate1(c, var, D_DescriptorSet, 0);
            emit_decorate1(c, var, D_Binding, (uint32_t)i);
            c->tex_vars[i] = var;
            c->tex_image_ty[i] = img;
            c->tex_sampled_ty[i] = simg;
            c->ssbo_vars[i] = 0;
        } else {
            c->ssbo_vars[i] = nid(c);
            put(c, &c->types, (4u << 16) | S_Variable);
            put(c, &c->types, c->t_ptr_ssbo);
            put(c, &c->types, c->ssbo_vars[i]);
            put(c, &c->types, SC_StorageBuffer);
            emit_decorate1(c, c->ssbo_vars[i], D_DescriptorSet, 0);
            emit_decorate1(c, c->ssbo_vars[i], D_Binding, (uint32_t)i);
        }
    }

    // Dynamic cudaSurfaceObject_t tables, one per distinct surface
    // type (dim/arrayed/format).  Handles index every table at a
    // common slot; a companion SSBO carries per-slot suq metadata.
    c->surf_table_count = 0;
    int need_meta = 0;
    for (size_t bi = 0; bi < c->cur_fn->nblocks; bi++)
        for (size_t ii = 0; ii < c->cur_fn->blocks[bi].count; ii++) {
            const ptx_instr_t *ti = &c->cur_fn->blocks[bi].instrs[ii];
            if (ti->op != POP_SULD && ti->op != POP_SUST &&
                ti->op != POP_SURED && ti->op != POP_SUQ)
                continue;
            if (surf_ref_is_static(c, ti->surf_ref, 0))
                continue;
            c->surf_table = 1;
            need_meta = 1;
            if (ti->op == POP_SUQ)
                continue;
            int g = ti->surf_geom ? ti->surf_geom : 2;
            int dim = surf_dim_of(g);
            int arrayed = surf_arrayed_of(g);
            int isint = ti->surf_formatted ? 0 : 1;
            int cas = 0, lock = 0;
            // Dynamic surface handles are indexed at runtime, so use
            // the load/CAS (or lock) table even when hardware image
            // atomics are available.
            int f32_fallback =
                ti->op == POP_SURED && ti->surf_ty == PTY_F32 &&
                (ti->surf_redop == 1 || ti->surf_redop == 2);
            int f32_add_fallback =
                ti->op == POP_SURED && ti->surf_ty == PTY_F32 &&
                ti->surf_redop == 0;
            if (f32_fallback || f32_add_fallback) {
                if (c->caps.surf_alias_ok)
                    cas = 1;
                else {
                    lock = 1;
                    c->surf_lock = 1;
                }
                c->surf_lock |= lock;
            }
            if (surf_table_add(c, dim, arrayed, isint, cas, lock) < 0) {
                failf(c, ti->line, "too many dynamic surface types");
                return;
            }
        }
    if (need_meta) {
        uint32_t marr = nid(c);
        put(c, &c->types, (3u << 16) | S_TypeRuntimeArray);
        put(c, &c->types, marr);
        put(c, &c->types, c->t_u32);
        emit_decorate1(c, marr, D_ArrayStride, 4);
        uint32_t mst = nid(c);
        put(c, &c->types, (3u << 16) | S_TypeStruct);
        put(c, &c->types, mst);
        put(c, &c->types, marr);
        emit_decorate(c, mst, D_Block);
        emit_member_decorate(c, mst, 0, D_Offset, 0);
        uint32_t mptr = nid(c);
        put(c, &c->types, (4u << 16) | S_TypePointer);
        put(c, &c->types, mptr);
        put(c, &c->types, SC_StorageBuffer);
        put(c, &c->types, mst);
        uint32_t mvar = nid(c);
        put(c, &c->types, (4u << 16) | S_Variable);
        put(c, &c->types, mptr);
        put(c, &c->types, mvar);
        put(c, &c->types, SC_StorageBuffer);
        emit_decorate1(c, mvar, D_DescriptorSet, 0);
        emit_decorate1(c, mvar, D_Binding, ACUDA_SURF_META_BINDING);
        uint32_t mu32 = nid(c);
        put(c, &c->types, (4u << 16) | S_TypePointer);
        put(c, &c->types, mu32);
        put(c, &c->types, SC_StorageBuffer);
        put(c, &c->types, c->t_u32);
        c->surf_meta_var = mvar;
        c->surf_meta_ptr_u32 = mu32;
    }
    if (c->surf_lock) {
        c->lock_var = nid(c);
        put(c, &c->types, (4u << 16) | S_Variable);
        put(c, &c->types, c->t_ptr_ssbo);
        put(c, &c->types, c->lock_var);
        put(c, &c->types, SC_StorageBuffer);
        emit_decorate1(c, c->lock_var, D_DescriptorSet, 0);
        emit_decorate1(c, c->lock_var, D_Binding,
                       ACUDA_SURF_LOCK_BINDING);
    }

    // Static .shared variables: one u32-word array per variable in
    // Workgroup storage (no descriptor, no push data; contents are
    // uninitialized, as in CUDA).
    c->t_ptr_w_u32 = nid(c);
    put(c, &c->types, (4u << 16) | S_TypePointer);
    put(c, &c->types, c->t_ptr_w_u32);
    put(c, &c->types, 4); // StorageClass Workgroup
    put(c, &c->types, c->t_u32);
    c->t_ptr_w_f32 = nid(c);
    put(c, &c->types, (4u << 16) | S_TypePointer);
    put(c, &c->types, c->t_ptr_w_f32);
    put(c, &c->types, 4); // StorageClass Workgroup
    put(c, &c->types, c->t_f32);
    c->t_ptr_w_u64 = nid(c);
    put(c, &c->types, (4u << 16) | S_TypePointer);
    put(c, &c->types, c->t_ptr_w_u64);
    put(c, &c->types, 4); // StorageClass Workgroup
    put(c, &c->types, c->t_u64);
    c->t_ptr_w_f64 = 0;
    if (c->t_f64) {
        c->t_ptr_w_f64 = nid(c);
        put(c, &c->types, (4u << 16) | S_TypePointer);
        put(c, &c->types, c->t_ptr_w_f64);
        put(c, &c->types, 4); // StorageClass Workgroup
        put(c, &c->types, c->t_f64);
    }
    c->t_ptr_img_u32 = nid(c);
    put(c, &c->types, (4u << 16) | S_TypePointer);
    put(c, &c->types, c->t_ptr_img_u32);
    put(c, &c->types, 11); // StorageClass Image
    put(c, &c->types, c->t_u32);
    c->t_ptr_img_f32 = nid(c);
    put(c, &c->types, (4u << 16) | S_TypePointer);
    put(c, &c->types, c->t_ptr_img_f32);
    put(c, &c->types, 11); // StorageClass Image
    put(c, &c->types, c->t_f32);
    c->t_ptr_img_u64 = nid(c);
    put(c, &c->types, (4u << 16) | S_TypePointer);
    put(c, &c->types, c->t_ptr_img_u64);
    put(c, &c->types, 11); // StorageClass Image
    put(c, &c->types, c->t_u64);
    if (c->uses_swbar) {
        // Software barrier state: one monotonic arrival counter per
        // PTX barrier id (16 * u32) plus one implicit cluster barrier
        // slot (index 16). Initialized by the entry prologue.
        uint32_t len = const_u32(c, 17);
        uint32_t arr = nid(c);
        put(c, &c->types, (4u << 16) | S_TypeArray);
        put(c, &c->types, arr);
        put(c, &c->types, c->t_u32);
        put(c, &c->types, len);
        uint32_t tpa = nid(c);
        put(c, &c->types, (4u << 16) | S_TypePointer);
        put(c, &c->types, tpa);
        put(c, &c->types, 4); // StorageClass Workgroup
        put(c, &c->types, arr);
        c->bar_var = nid(c);
        put(c, &c->types, (4u << 16) | S_Variable);
        put(c, &c->types, tpa);
        put(c, &c->types, c->bar_var);
        put(c, &c->types, 4); // StorageClass Workgroup
    }
    if (c->sh_coop_typed) {
        if (c->sh_coop_comp == 0) {
            if (!c->t_f16) {
                c->t_f16 = nid(c);
                put(c, &c->types, (3u << 16) | S_TypeFloat);
                put(c, &c->types, c->t_f16);
                put(c, &c->types, 16);
            }
            c->t_coop_elem = c->t_f16;
        } else {
            if (!c->t_bf16) {
                c->t_bf16 = nid(c);
                put(c, &c->types, (4u << 16) | S_TypeFloat);
                put(c, &c->types, c->t_bf16);
                put(c, &c->types, 16);
                put(c, &c->types, 0);
            }
            c->t_coop_elem = c->t_bf16;
        }
        if (!c->t_u16) {
            c->t_u16 = nid(c);
            put(c, &c->types, (4u << 16) | S_TypeInt);
            put(c, &c->types, c->t_u16);
            put(c, &c->types, 16);
            put(c, &c->types, 0);
        }
        c->t_ptr_w_coop = nid(c);
        put(c, &c->types, (4u << 16) | S_TypePointer);
        put(c, &c->types, c->t_ptr_w_coop);
        put(c, &c->types, 4);
        put(c, &c->types, c->t_coop_elem);
    }
    for (int i = 0; i < c->nshvars; i++) {
        // Extern (dynamic) variables share one launch-sized array
        // below; static ones keep per-variable arrays here.
        if (c->shvars[i].is_extern)
            continue;
        if (i >= ACUDA_MAX_SHVARS) {
            fail(c, "too many .shared variables");
            return;
        }
        int typed = c->sh_coop_typed;
        int sh64 = c->shvars[i].elem == 8;
        // CUDA rounds each static shared allocation up to a 16-byte
        // granularity; lifted kernels (and the PTX corpus) rely on the
        // padding by addressing past the declared scalar (e.g. `sbuf+4`
        // after `.shared .u32 sbuf`).  Keep the declared byte count for
        // type heuristics and size only the Workgroup array.
        size_t shbytes = c->shvars[i].bytes;
        shbytes = shbytes < 16 ? 16 : (shbytes + 15) & ~(size_t)15;
        size_t words = typed ? (shbytes / 2)
                             : sh64 ? (shbytes / 8)
                                    : (shbytes + 3) / 4;
        if (words == 0)
            words = 1;
        if (words > 0xffffffffu) {
            fail(c, "shared variable too large");
            return;
        }
        uint32_t len = const_u32(c, (uint32_t)words);
        uint32_t arr = nid(c);
        put(c, &c->types, (4u << 16) | S_TypeArray);
        put(c, &c->types, arr);
        put(c, &c->types,
            typed ? c->t_coop_elem
                  : sh64 ? (c->shared_f64_view ? c->t_f64 : c->t_u64)
                         : (c->shared_f32_view ? c->t_f32 : c->t_u32));
        put(c, &c->types, len);
        // No ArrayStride here: the Vulkan environment forbids explicit
        // layout decorations on Workgroup variables
        // (VUID-StandaloneSpirv-None-10684); indexing is by element
        // either way. (SSBO arrays keep theirs: host-visible layout.)
        uint32_t tpa = nid(c);
        put(c, &c->types, (4u << 16) | S_TypePointer);
        put(c, &c->types, tpa);
        put(c, &c->types, 4); // StorageClass Workgroup
        put(c, &c->types, arr);
        c->sh_vars[i] = nid(c);
        put(c, &c->types, (4u << 16) | S_Variable);
        put(c, &c->types, tpa);
        put(c, &c->types, c->sh_vars[i]);
        put(c, &c->types, 4); // StorageClass Workgroup
        if (c->failed)
            return;
    }
    // Spinlock for 64-bit shared atomics expressed on the u32-viewed
    // array (see emit_atom_shared64 in atom.c).  A Workgroup variable is
    // one per workgroup, which is exactly the atomic's scope.
    c->sh_lock_var = 0;
    {
        int needs_sh_lock = 0;
        for (int fi = 0; fi < c->nfns && !needs_sh_lock; fi++)
            for (size_t bi = 0;
                 bi < c->fns[fi].nblocks && !needs_sh_lock; bi++)
                for (size_t ii = 0; ii < c->fns[fi].blocks[bi].count; ii++) {
                    const ptx_instr_t *ti =
                        &c->fns[fi].blocks[bi].instrs[ii];
                    if ((ti->op == POP_ATOM || ti->op == POP_ATOMCAS) &&
                        ti->space == PSP_SHARED &&
                        (ti->ty == PTY_U64 || ti->ty == PTY_F64)) {
                        needs_sh_lock = 1;
                        break;
                    }
                }
        if (needs_sh_lock && c->nshvars > 0) {
            c->sh_lock_var = nid(c);
            put(c, &c->types, (4u << 16) | S_Variable);
            put(c, &c->types, c->t_ptr_w_u32);
            put(c, &c->types, c->sh_lock_var);
            put(c, &c->types, 4); // StorageClass Workgroup
        }
    }
    // Dynamic `.shared` (`.extern`): one u32-word Workgroup array
    // shared by all extern declarations (they alias the same launch-
    // sized base in PTX), sized by spec constant 3 (words; default 1
    // keeps the module valid before specialization). Static
    // variables keep their own arrays above; contents here arrive
    // per-launch (uninitialized, as in CUDA).
    {
        int any_extern = 0;
        for (int i = 0; i < c->nshvars; i++) {
            if (c->shvars[i].is_extern) {
                any_extern = 1;
                break;
            }
        }
        c->has_extern = any_extern;
        if (any_extern) {
            put(c, &c->types, (4u << 16) | S_SpecConstant);
            put(c, &c->types, c->t_u32);
            put(c, &c->types, c->sp_dyn);
            put(c, &c->types, 1);
            emit_decorate1(c, c->sp_dyn, D_SpecId, 3);
            uint32_t arr = nid(c);
            put(c, &c->types, (4u << 16) | S_TypeArray);
            put(c, &c->types, arr);
            put(c, &c->types, c->t_u32);
            put(c, &c->types, c->sp_dyn);
            uint32_t tpa = nid(c);
            put(c, &c->types, (4u << 16) | S_TypePointer);
            put(c, &c->types, tpa);
            put(c, &c->types, 4); // StorageClass Workgroup
            put(c, &c->types, arr);
            uint32_t dyn = nid(c);
            put(c, &c->types, (4u << 16) | S_Variable);
            put(c, &c->types, tpa);
            put(c, &c->types, dyn);
            put(c, &c->types, 4); // StorageClass Workgroup
            for (int i = 0; i < c->nshvars; i++) {
                if (c->shvars[i].is_extern)
                    c->sh_vars[i] = dyn;
            }
            if (c->failed)
                return;
        }
    }
    // Private .local variables: u32-word arrays at module scope (no
    // ArrayStride: explicit layout is for host-visible storage only).
    c->t_ptr_priv_u32 = nid(c);
    put(c, &c->types, (4u << 16) | S_TypePointer);
    put(c, &c->types, c->t_ptr_priv_u32);
    put(c, &c->types, 6); // StorageClass Private (grammar-verified)
    put(c, &c->types, c->t_u32);
    // Per-invocation loop counter for synchronous bulk copies.
    c->bulk_i_var = 0;
    for (size_t bi = 0; bi < c->cur_fn->nblocks && !c->bulk_i_var; bi++)
        for (size_t ii = 0; ii < c->cur_fn->blocks[bi].count; ii++)
            if (c->cur_fn->blocks[bi].instrs[ii].op == POP_CPASYNC_BULK) {
                uint32_t pty = nid(c);
                put(c, &c->types, (4u << 16) | S_TypePointer);
                put(c, &c->types, pty);
                put(c, &c->types, 6); // Private
                put(c, &c->types, c->t_u64);
                c->bulk_i_var = nid(c);
                put(c, &c->types, (4u << 16) | S_Variable);
                put(c, &c->types, pty);
                put(c, &c->types, c->bulk_i_var);
                put(c, &c->types, 6); // Private
                break;
            }
    if (c->host_machine || c->cfg_machine) {
        // Function-storage variables carry the per-invocation register
        // file and PC across blocks of the host-call state machine.
        struct { uint32_t *dst; uint32_t ty; } fns[5] = {
            { &c->t_ptr_fn_u32, c->t_u32 },
            { &c->t_ptr_fn_u64, c->t_u64 },
            { &c->t_ptr_fn_f32, c->t_f32 },
            { &c->t_ptr_fn_f64, c->t_f64 },
            { &c->t_ptr_fn_bool, c->t_bool },
        };
        for (int fi = 0; fi < 5; fi++) {
            if (fns[fi].ty == 0)
                continue; // f64 is declared only when used
            uint32_t id = nid(c);
            *fns[fi].dst = id;
            put(c, &c->types, (4u << 16) | S_TypePointer);
            put(c, &c->types, id);
            put(c, &c->types, 7); // StorageClass Function
            put(c, &c->types, fns[fi].ty);
        }
    }
    for (int i = 0; i < c->nlocals; i++) {
        if (i >= ACUDA_MAX_LOCALS) {
            fail(c, "too many .local variables");
            return;
        }
        size_t words = (c->locals[i].bytes + 3) / 4;
        if (words == 0)
            words = 1;
        if (words > 0xffffffffu) {
            fail(c, "local variable too large");
            return;
        }
        uint32_t len = const_u32(c, (uint32_t)words);
        uint32_t arr = nid(c);
        put(c, &c->types, (4u << 16) | S_TypeArray);
        put(c, &c->types, arr);
        put(c, &c->types, c->t_u32);
        put(c, &c->types, len);
        uint32_t tpa = nid(c);
        put(c, &c->types, (4u << 16) | S_TypePointer);
        put(c, &c->types, tpa);
        put(c, &c->types, 6); // StorageClass Private
        put(c, &c->types, arr);
        c->lo_vars[i] = nid(c);
        put(c, &c->types, (4u << 16) | S_Variable);
        put(c, &c->types, tpa);
        put(c, &c->types, c->lo_vars[i]);
        put(c, &c->types, 6); // StorageClass Private
        if (c->failed)
            return;
    }

    // Initialized .global images: one u32-word Private array per
    // variable, initialized from the declaration bytes (zero-padded
    // tail). Private scope keeps the read-only image per-thread with
    // no host-visible layout (no ArrayStride, same rule as .local).
    // Dynamic offsets index it at runtime; stores stay clean errors
    // (no device backing yet).
    for (int i = 0; i < c->nglobals; i++) {
        if (i >= ACUDA_MAX_GLOBALS) {
            fail(c, "too many .global variables");
            return;
        }
        size_t words = (c->globals[i].nbytes + 3) / 4;
        if (words == 0)
            words = 1;
        if (words > 0xffffffffu) {
            fail(c, "global variable too large");
            return;
        }
        uint32_t len = const_u32(c, (uint32_t)words);
        uint32_t arr = nid(c);
        put(c, &c->types, (4u << 16) | S_TypeArray);
        put(c, &c->types, arr);
        put(c, &c->types, c->t_u32);
        put(c, &c->types, len);
        // Word constants for the initializer (little-endian lanes).
        uint32_t *wids = malloc(words * sizeof(*wids));
        if (!wids) {
            fail(c, "out of memory");
            return;
        }
        for (size_t w = 0; w < words; w++) {
            uint32_t word = 0;
            for (size_t k = 0; k < 4; k++) {
                size_t idx = w * 4 + k;
                if (idx < c->globals[i].nbytes)
                    word |= (uint32_t)c->globals[i].bytes[idx] << (8 * k);
            }
            wids[w] = const_u32(c, word);
        }
        uint32_t init = nid(c);
        put(c, &c->types, (uint32_t)((3 + words) << 16) | S_ConstantComposite);
        put(c, &c->types, arr);
        put(c, &c->types, init);
        for (size_t w = 0; w < words; w++)
            put(c, &c->types, wids[w]);
        free(wids);
        uint32_t gtpa = nid(c);
        put(c, &c->types, (4u << 16) | S_TypePointer);
        put(c, &c->types, gtpa);
        put(c, &c->types, 6); // StorageClass Private
        put(c, &c->types, arr);
        c->gl_vars[i] = nid(c);
        put(c, &c->types, (5u << 16) | S_Variable);
        put(c, &c->types, gtpa);
        put(c, &c->types, c->gl_vars[i]);
        put(c, &c->types, 6); // StorageClass Private
        put(c, &c->types, init);
        if (c->failed)
            return;
    }

    // Dynamic .const image: a zero-filled Private byte bank with every
    // declared .const copied at its flattened offset. Register-indexed
    // ld.const reads this image; direct [name+imm] accesses still fold to
    // immediates. The 64 KiB floor matches the CUDA constant-bank size.
    if (c->has_const_bank) {
        size_t bank_size = c->const_bank_size;
        if (bank_size < 65536)
            bank_size = 65536;
        bank_size = (bank_size + 3u) & ~(size_t)3u;
        if (bank_size > 4u * 1024u * 1024u) {
            fail(c, "dynamic .const image too large");
            return;
        }
        uint8_t *bank = calloc(bank_size, 1);
        if (!bank) {
            fail(c, "out of memory");
            return;
        }
        for (int i = 0; i < c->nconsts; i++) {
            const ptx_const_t *pc = &c->consts[i];
            if (pc->bank_off + pc->nbytes > bank_size) {
                free(bank);
                fail(c, "dynamic .const image offset out of range");
                return;
            }
            memcpy(bank + pc->bank_off, pc->bytes, pc->nbytes);
        }
        size_t words = bank_size / 4;
        uint32_t len = const_u32(c, (uint32_t)words);
        uint32_t arr = nid(c);
        put(c, &c->types, (4u << 16) | S_TypeArray);
        put(c, &c->types, arr);
        put(c, &c->types, c->t_u32);
        put(c, &c->types, len);
        uint32_t *wids = malloc(words * sizeof(*wids));
        if (!wids) {
            free(bank);
            fail(c, "out of memory");
            return;
        }
        for (size_t w = 0; w < words; w++) {
            uint32_t word = 0;
            for (size_t k = 0; k < 4; k++)
                word |= (uint32_t)bank[w * 4 + k] << (8 * k);
            wids[w] = const_u32(c, word);
        }
        free(bank);
        uint32_t init = nid(c);
        put(c, &c->types, (uint32_t)((3 + words) << 16) | S_ConstantComposite);
        put(c, &c->types, arr);
        put(c, &c->types, init);
        for (size_t w = 0; w < words; w++)
            put(c, &c->types, wids[w]);
        free(wids);
        uint32_t btpa = nid(c);
        put(c, &c->types, (4u << 16) | S_TypePointer);
        put(c, &c->types, btpa);
        put(c, &c->types, 6); // StorageClass Private
        put(c, &c->types, arr);
        c->const_bank_var = nid(c);
        put(c, &c->types, (5u << 16) | S_Variable);
        put(c, &c->types, btpa);
        put(c, &c->types, c->const_bank_var);
        put(c, &c->types, 6); // StorageClass Private
        put(c, &c->types, init);
        if (c->failed)
            return;
    }

    // Builtin inputs (ids reserved early: entry point precedes decorations).
    // %ntid (WorkgroupSize) is a *specialization* composite so launches
    // may use any blockDimX in 1..1024 (specialized per pipeline).
    // Ids are reserved up front (entry references them); declared here.
    put(c, &c->types, (4u << 16) | S_SpecConstant);
    put(c, &c->types, c->t_u32);
    put(c, &c->types, c->sp_x);
    put(c, &c->types, 256);
    emit_decorate1(c, c->sp_x, D_SpecId, 0);
    put(c, &c->types, (4u << 16) | S_SpecConstant);
    put(c, &c->types, c->t_u32);
    put(c, &c->types, c->sp_y);
    put(c, &c->types, 1);
    emit_decorate1(c, c->sp_y, D_SpecId, 1);
    put(c, &c->types, (4u << 16) | S_SpecConstant);
    put(c, &c->types, c->t_u32);
    put(c, &c->types, c->sp_z);
    put(c, &c->types, 1);
    emit_decorate1(c, c->sp_z, D_SpecId, 2);
    c->v_ntid = nid(c);
    put(c, &c->types, (6u << 16) | S_SpecConstantComposite);
    put(c, &c->types, c->t_v3u32);
    put(c, &c->types, c->v_ntid);
    put(c, &c->types, c->sp_x);
    put(c, &c->types, c->sp_y);
    put(c, &c->types, c->sp_z);
    put(c, &c->types, (4u << 16) | S_Variable);
    put(c, &c->types, c->t_ptr_in_v3);
    put(c, &c->types, c->v_tid);
    put(c, &c->types, SC_Input);
    emit_decorate1(c, c->v_tid, D_BuiltIn, B_LocalInvocationId);
    put(c, &c->types, (4u << 16) | S_Variable);
    put(c, &c->types, c->t_ptr_in_v3);
    put(c, &c->types, c->v_ctaid);
    put(c, &c->types, SC_Input);
    emit_decorate1(c, c->v_ctaid, D_BuiltIn, B_WorkgroupId);
    put(c, &c->types, (4u << 16) | S_Variable);
    put(c, &c->types, c->t_ptr_in_v3);
    put(c, &c->types, c->v_nctaid);
    put(c, &c->types, SC_Input);
    emit_decorate1(c, c->v_nctaid, D_BuiltIn, B_NumWorkGroups);
    // Subgroup lane for shfl (scalar u32 input, BuiltIn 41).
    // Emitted only with shfl (an unused interface variable trips
    // Vulkan VUIDs under the strict env).
    if (c->uses_subid) {
        uint32_t tpi = nid(c);
        put(c, &c->types, (4u << 16) | S_TypePointer);
        put(c, &c->types, tpi);
        put(c, &c->types, SC_Input);
        put(c, &c->types, c->t_u32);
        put(c, &c->types, (4u << 16) | S_Variable);
        put(c, &c->types, tpi);
        put(c, &c->types, c->v_subid);
        put(c, &c->types, SC_Input);
        emit_decorate1(c, c->v_subid, D_BuiltIn, 41); // SubgroupLocalInvocationId
    }
}

// Entry point + execution mode + name. Must be emitted before any
// decorations (section order: entry, exec, debug, annotations).
// SPIR-V 1.4+ requires every global variable used by the entry point
// to be listed in its interface. Globals are created on demand while
// the prelude and body are emitted, so this runs at the very end and
// scans the types section (function-local variables live in the
// function section and are excluded by storage class).
size_t count_globals(ctx_t *c) {
    size_t n = 0;
    for (size_t i = 0; i < c->types.count;) {
        uint32_t wc = c->types.words[i] >> 16;
        if (wc == 0)
            break;
        if ((c->types.words[i] & 0xffffu) == S_Variable && wc >= 4 &&
            i + 3 < c->types.count && c->types.words[i + 3] != 7)
            n++;
        i += wc;
    }
    return n;
}

void emit_entry_ann(ctx_t *c, const ptx_func_t *fn) {
    size_t name_words = (strlen(fn->name) + 1 + 3) / 4;
    size_t niface = count_globals(c);
    put(c, &c->entry,
        (uint32_t)((3 + name_words + niface) << 16) | S_EntryPoint);
    put(c, &c->entry, 5); // GLCompute
    put(c, &c->entry, c->fn_id);
    emit_str(c, &c->entry, fn->name);
    for (size_t i = 0; i < c->types.count;) {
        uint32_t wc = c->types.words[i] >> 16;
        if (wc == 0)
            break;
        if ((c->types.words[i] & 0xffffu) == S_Variable && wc >= 4 &&
            i + 3 < c->types.count && c->types.words[i + 3] != 7)
            put(c, &c->entry, c->types.words[i + 2]);
        i += wc;
    }
    put(c, &c->entry, (6u << 16) | S_ExecutionModeId);
    put(c, &c->entry, c->fn_id);
    put(c, &c->entry, 38); // LocalSizeId (ids, not literals)
    put(c, &c->entry, c->sp_x);
    put(c, &c->entry, c->sp_y);
    put(c, &c->entry, c->sp_z);
    if (c->use_fc32 || c->use_fc64) {
        // DenormPreserve = 4459, SignedZeroInfNanPreserve = 4461.
        put(c, &c->entry, (4u << 16) | S_ExecutionMode);
        put(c, &c->entry, c->fn_id);
        put(c, &c->entry, 4459);
        put(c, &c->entry, 32);
        put(c, &c->entry, (4u << 16) | S_ExecutionMode);
        put(c, &c->entry, c->fn_id);
        put(c, &c->entry, 4461);
        put(c, &c->entry, 32);
        if (c->use_fc64) {
            put(c, &c->entry, (4u << 16) | S_ExecutionMode);
            put(c, &c->entry, c->fn_id);
            put(c, &c->entry, 4459);
            put(c, &c->entry, 64);
            put(c, &c->entry, (4u << 16) | S_ExecutionMode);
            put(c, &c->entry, c->fn_id);
            put(c, &c->entry, 4461);
            put(c, &c->entry, 64);
        }
    }
    // Debug subsection order: OpString precedes OpName.
    if (c->uses_brkpt) {
        const char *fname = "<ptx>";
        size_t fwords = (strlen(fname) + 1 + 3) / 4;
        put(c, &c->entry,
            (uint32_t)((2 + fwords) << 16) | S_String);
        put(c, &c->entry, c->v_file);
        emit_str(c, &c->entry, fname);
    }
    put(c, &c->entry, (uint32_t)((2 + name_words) << 16) | S_Name);
    put(c, &c->entry, c->fn_id);
    emit_str(c, &c->entry, fn->name);
}




uint32_t phys_align_of(ctx_t *c, uint32_t ptr);

void emit_load(ctx_t *c, spv_blob_t *b, uint32_t ty, uint32_t id,
                      uint32_t ptr) {
    uint32_t al = phys_align_of(c, ptr);
    if (al) {
        // PhysicalStorageBuffer accesses require an Aligned memory
        // operand (SPV_KHR_physical_storage_buffer).
        put(c, b, (6u << 16) | S_Load);
        put(c, b, ty);
        put(c, b, id);
        put(c, b, ptr);
        put(c, b, 0x2u); // Memory Operand: Aligned
        put(c, b, al);
        return;
    }
    put(c, b, (4u << 16) | S_Load);
    put(c, b, ty);
    put(c, b, id);
    put(c, b, ptr);
}

