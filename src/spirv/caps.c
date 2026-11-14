// PTX subset -> SPIR-V (Logical, GLSL450) emitter.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if !defined(_MSC_VER)
#include <strings.h>
#endif
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <pthread.h>
#endif
#include "ptx.h"
#include "acuda/compiler.h"
#include "spirv_internal.h"

void acuda_compiler_caps_init(acuda_compiler_caps_t *caps) {
    if (!caps)
        return;
    memset(caps, 0, sizeof(*caps));
    caps->surf_alias_ok = 1;
}

void acuda_compiler_set_surface_alias(acuda_compiler_caps_t *caps, int available) {
    if (caps) caps->surf_alias_ok = available != 0;
}
void acuda_compiler_set_shader_fma(acuda_compiler_caps_t *caps, int available) {
    if (caps) caps->shader_fma = available != 0;
}
void acuda_compiler_set_shader_fma_f64(acuda_compiler_caps_t *caps, int available) {
    if (caps) caps->shader_fma_f64 = available != 0;
}
void acuda_compiler_set_soft_fma(acuda_compiler_caps_t *caps, int force) {
    if (caps) caps->soft_fma = force != 0;
}
void acuda_compiler_set_f16_native(acuda_compiler_caps_t *caps, int available) {
    if (caps) caps->f16_native = available != 0;
}
void acuda_compiler_set_cooperative_matrix(acuda_compiler_caps_t *caps, int f16, int bf16) {
    if (caps) { caps->coopmat_f16 = f16 != 0; caps->coopmat_bf16 = bf16 != 0; }
}
void acuda_compiler_set_mixed_fdot(acuda_compiler_caps_t *caps, int available) {
    if (caps) caps->mixed_fdot = available != 0;
}
void acuda_compiler_set_dot_product(acuda_compiler_caps_t *caps, int available) {
    if (caps) caps->dot_product = available != 0;
}
void acuda_compiler_set_dot_product_forms(acuda_compiler_caps_t *caps,
                                          int signed_accel,
                                          int unsigned_accel,
                                          int mixed_accel) {
    if (!caps) return;
    caps->dot_signed = signed_accel != 0;
    caps->dot_unsigned = unsigned_accel != 0;
    caps->dot_mixed = mixed_accel != 0;
}
void acuda_compiler_get_dot_product_forms(const acuda_compiler_caps_t *caps,
                                          int *signed_accel,
                                          int *unsigned_accel,
                                          int *mixed_accel) {
    int s = caps ? caps->dot_signed : 0;
    int u = caps ? caps->dot_unsigned : 0;
    int m = caps ? caps->dot_mixed : 0;
    if (signed_accel) *signed_accel = s;
    if (unsigned_accel) *unsigned_accel = u;
    if (mixed_accel) *mixed_accel = m;
}
void acuda_compiler_set_zero_init_workgroup(acuda_compiler_caps_t *caps, int available) {
    if (caps) caps->zero_init_wg = available != 0;
}
void acuda_compiler_set_ssbo8(acuda_compiler_caps_t *caps, int available) {
    if (caps) caps->ssbo8 = available != 0;
}
void acuda_compiler_set_ssbo16(acuda_compiler_caps_t *caps, int available) {
    if (caps) caps->ssbo16 = available != 0;
}
void acuda_compiler_set_float_controls(acuda_compiler_caps_t *caps, int f32_ok, int f64_ok) {
    if (caps) { caps->fc32 = f32_ok != 0; caps->fc64 = f64_ok != 0; }
}
void acuda_compiler_set_vulkan_memory_model(acuda_compiler_caps_t *caps, int available) {
    if (caps) caps->vmm = available != 0;
}
void acuda_compiler_set_bda(acuda_compiler_caps_t *caps, int available) {
    if (caps) caps->bda = available != 0;
}
void acuda_compiler_set_atomic_float(acuda_compiler_caps_t *caps, unsigned features) {
    if (caps) caps->atomic_float = features;
}
unsigned acuda_compiler_get_atomic_float(const acuda_compiler_caps_t *caps) {
    return caps ? caps->atomic_float : 0u;
}
void acuda_compiler_set_int64_image(acuda_compiler_caps_t *caps, int available) {
    if (caps) caps->int64_image = available != 0;
}

