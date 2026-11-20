// rtxdiasm PTX-to-SPIR-V compiler. C ABI.
//
// Subset (see acuda docs/ptx-subset.md): integer/float arithmetic, loads,
// stores, 1D indexing via %tid/%ntid/%ctaid (.x), single forward guard
// branch, ret/exit. Everything else is a clean compile error.
#ifndef ACUDA_COMPILER_H
#define ACUDA_COMPILER_H

#include <stddef.h>
#include <stdint.h>
#include "acuda/runtime.h"
#include "acuda/hostcall.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ACUDA_ENTRY_NAME_MAX 1024
#define ACUDA_MAX_PARAMS 256
// Dynamic surface-object tables: one runtime-sized storage-image
// descriptor array per surface type, each in its own descriptor set.
// Binding 254 in set 0 carries per-slot metadata.
#define ACUDA_SURF_TABLE_MAX_TYPES 12
#define ACUDA_SURF_META_BINDING 254u
#define ACUDA_SURF_LOCK_BINDING 253u
#define ACUDA_SURF_LOCK_SLOTS 65536u
#define ACUDA_SURF_LOCK_BYTES (ACUDA_SURF_LOCK_SLOTS * 4u)
#define ACUDA_MAX_HOST_CALLS 32
// By-value scalar structs larger than this many 8-byte words (128 bytes)
// use a descriptor-bound storage buffer instead of push constants. This
// keeps the push block below the 256-byte limit most Vulkan devices
// advertise while preserving the existing ABI for small structs.
#define ACUDA_BYVAL_SSBO_WORDS 16u

typedef struct acuda_kernel_info {
    char entry[ACUDA_ENTRY_NAME_MAX];
    uint64_t serial;             // unique per compilation (cache key)
    uint32_t num_ptr_params; // descriptor bindings, in .param order
    uint32_t num_params;     // total params, in .param order (<= ACUDA_MAX_PARAMS)
    // Per-param kind, in .param order: 0 = 32-bit scalar (4B arg, 1 word),
    // 1 = pointer (1 binding; stride from param_stride), 2 = 64-bit
    // scalar (8B integer arg, 2 words).
    uint32_t is_ptr[ACUDA_MAX_PARAMS];
    // Parallel to is_ptr: declared argument stride in bytes (from the
    // .param type: .u64/.f64/.b64 = 8, by-value = wide_words * 8,
    // else 4; 0 for continuations, hidden texrefs and unused params).
    // cudart packs kernelParams at these strides; the driver must walk
    // with them, not with kind-implied sizes.
    uint32_t param_stride[ACUDA_MAX_PARAMS];
    // Parallel to is_ptr: number of 8-byte words for a kind-4
    // by-value struct (0 otherwise).
    uint32_t wide_words[ACUDA_MAX_PARAMS];
    // 1 when a kind-4 by-value struct is too large for the push-constant
    // block and is passed through a descriptor-bound storage buffer
    // instead. The host uploads wide_words * 8 argument bytes.
    uint32_t param_ssbo[ACUDA_MAX_PARAMS];
    // Parallel to is_ptr: 1 when the binding is a .texref/.samplerref
    // descriptor buffer (a buffer-backed texture).
    uint32_t is_tex[ACUDA_MAX_PARAMS];
    // Parallel to is_tex: module-scope texture references injected as
    // hidden parameters.  The driver supplies their bound CUtexObject
    // rather than reading kernelParams.
    uint32_t hidden_texref[ACUDA_MAX_PARAMS];
    char hidden_texref_name[ACUDA_MAX_PARAMS][ACUDA_ENTRY_NAME_MAX];
    // Module-scope .global/.const symbols injected as hidden pointer params.
    // The driver/runtime supplies their device buffer addresses by name.
    uint32_t hidden_symbol[ACUDA_MAX_PARAMS];
    char hidden_symbol_name[ACUDA_MAX_PARAMS][ACUDA_ENTRY_NAME_MAX];
    // Per descriptor binding (dense pointer order): 0 = storage buffer,
    // 1 = combined image sampler, 3 = standalone sampler.
    uint32_t binding_kind[ACUDA_MAX_PARAMS];
    // Binding uses an R32_UINT alias view for the portable f32 surface
    // min/max CAS fallback (native shaderImageFloat32AtomicMinMax absent).
    uint32_t surf_f32_cas[ACUDA_MAX_PARAMS];
    uint32_t surf_f32_lock[ACUDA_MAX_PARAMS];
    uint32_t surf_lock;
    // Dynamic surface-object tables (one per dim/arrayed/format type).
    uint32_t surf_table;
    uint32_t surf_table_count;
    uint32_t surf_table_binding[ACUDA_SURF_TABLE_MAX_TYPES];
    uint32_t surf_table_dim[ACUDA_SURF_TABLE_MAX_TYPES];
    uint32_t surf_table_arrayed[ACUDA_SURF_TABLE_MAX_TYPES];
    uint32_t surf_table_int[ACUDA_SURF_TABLE_MAX_TYPES];
    uint32_t surf_table_cas[ACUDA_SURF_TABLE_MAX_TYPES];
    uint32_t surf_table_lock[ACUDA_SURF_TABLE_MAX_TYPES];
    // Texture/sampler metadata words appended after the scalar push
    // constants (6 words per texture or sampler binding, binding order).
    uint32_t tex_meta_bytes;
    uint32_t push_bytes;     // scalar words total, in bytes
    uint32_t block_x;        // default LocalSize.x (256, the spec-constant
                             // default); launches accept 1..1024, each size
                             // getting its own specialized pipeline
    uint32_t num_regs;       // declared registers (func attribute estimate)
    uint32_t ptx_version;    // .version as major*10+minor (e.g. 88)
    uint32_t sm_version;     // .target sm (e.g. 80); 0 if absent
    uint32_t shared_bytes;   // static .shared total, in bytes (Workgroup)
    uint32_t has_extern;     // 1 when any `.extern .shared` is present
                             // (dynamic part sized per launch)
    // 1 when the kernel uses warp-level subgroup operations (shfl, vote,
    // redux, match, elect, warp barriers, MMA). The runtime requires a
    // 32-lane subgroup for these so PTX warp semantics stay exact.
    uint32_t uses_subgroup;
    // 1 when the kernel uses workgroup barriers (bar.sync/arrive,
    // mbarrier, bar.warp.sync); forced subgroup sizes are avoided there.
    uint32_t uses_barrier;
    // 0 = unknown (old blob, runtime should scan SPIR-V), 1 = no shaderFloat64
    // capability required, 2 = OpCapability Float64 required.
    uint32_t requires_float64;
    // 1 when kernel pointer params are passed as 64-bit device addresses
    // (PhysicalStorageBuffer) instead of descriptor bindings. The runtime
    // appends one address per entry in bda_param[]; addr_push_bytes is
    // the size of that region at the end of the push-constant block.
    uint32_t bda;
    // 1 when the shader resolves raw 64-bit device addresses
    // (PhysicalStorageBuffer). The runtime cannot track which buffers
    // such a kernel touches, so it inserts a global memory barrier
    // around the dispatch.
    uint32_t uses_bda_access;
    uint32_t addr_push_bytes;
    uint32_t bda_param[ACUDA_MAX_PARAMS]; // binding/user-pointer indices
    // 1 when the kernel may write through this pointer binding (dense
    // pointer order, including internal host-call buffers at the end).
    // Conservative: 1 whenever a store/atomic/reduction/host call or a
    // surface write could touch the parameter's memory.
    uint32_t param_written[ACUDA_MAX_PARAMS];

    // Host-call chain metadata. These counts describe user-visible
    // params; the two internal state/request buffers are not counted.
    uint32_t host_segment_count; // 0 = ordinary kernel (state-machine
                                 // kernels report 1)
    uint32_t host_state_machine; // 1 = relaunch until no request is pending
    uint32_t host_call_count;
    uint32_t host_nargs[ACUDA_MAX_HOST_CALLS];
    uint32_t host_state_stride;   // bytes per invocation
    uint32_t host_shared_stride;  // bytes per workgroup
    uint32_t host_request_stride; // bytes per invocation
    char host_call_names[ACUDA_MAX_HOST_CALLS][ACUDA_HOST_NAME_MAX];
    uint32_t host_arg_bytes[ACUDA_MAX_HOST_CALLS];
    uint32_t host_ret_bytes[ACUDA_MAX_HOST_CALLS];
    uint32_t host_ret_offsets[ACUDA_MAX_HOST_CALLS]; // in the state buffer
    uint32_t host_arg_offsets[ACUDA_MAX_HOST_CALLS][ACUDA_MAX_HOST_CALLS]; // in the request args

} acuda_kernel_info_t;

typedef struct acuda_kernel acuda_kernel_t;

// Per-compile capability/configuration (thread-safe: each compile uses its own
// caps; no global compiler state). Different GPUs/devices should use different
// caps. Initialize with acuda_compiler_caps_init() for safe defaults.
typedef struct acuda_compiler_caps {
    int shader_fma;
    int shader_fma_f64;
    int soft_fma;
    int f16_native;
    int coopmat_f16;
    int coopmat_bf16;
    int mixed_fdot;
    int dot_product;
    int dot_signed, dot_unsigned, dot_mixed;
    int zero_init_wg;
    int ssbo8;
    int ssbo16;
    int fc32, fc64;
    int vmm;
    int bda;
    unsigned atomic_float;
    int int64_image;
    int surf_alias_ok; // default 1
} acuda_compiler_caps_t;

void acuda_compiler_caps_init(acuda_compiler_caps_t *caps);

// Compiles one entry from PTX source text. entry_name NULL = first entry.
// caps NULL uses default-initialized caps. On failure returns != SUCCESS and
// writes a message to errbuf.
acuda_status_t acuda_compile_ptx(const char *ptx_text, const char *entry_name,
                                 acuda_kernel_t **out_kernel,
                                 char *errbuf, size_t errlen,
                                 const acuda_compiler_caps_t *caps);
// Fast structural validation (parse only, no SPIR-V emission).
acuda_status_t acuda_ptx_validate(const char *ptx_text,
                                  char *errbuf, size_t errlen);

// Capability setters operate only on the provided caps object (per-ctx).
void acuda_compiler_set_shader_fma(acuda_compiler_caps_t *caps, int available);
void acuda_compiler_set_soft_fma(acuda_compiler_caps_t *caps, int force);
void acuda_compiler_set_f16_native(acuda_compiler_caps_t *caps, int available);
void acuda_compiler_set_cooperative_matrix(acuda_compiler_caps_t *caps,
                                           int f16, int bf16);
void acuda_compiler_set_mixed_fdot(acuda_compiler_caps_t *caps, int available);
void acuda_compiler_set_shader_fma_f64(acuda_compiler_caps_t *caps, int available);
void acuda_compiler_set_dot_product(acuda_compiler_caps_t *caps, int available);
void acuda_compiler_set_dot_product_forms(acuda_compiler_caps_t *caps,
                                          int signed_accel,
                                          int unsigned_accel,
                                          int mixed_accel);
void acuda_compiler_get_dot_product_forms(const acuda_compiler_caps_t *caps,
                                          int *signed_accel,
                                          int *unsigned_accel,
                                          int *mixed_accel);
void acuda_compiler_set_zero_init_workgroup(acuda_compiler_caps_t *caps,
                                            int available);
void acuda_compiler_set_ssbo8(acuda_compiler_caps_t *caps, int available);
void acuda_compiler_set_ssbo16(acuda_compiler_caps_t *caps, int available);
void acuda_compiler_set_float_controls(acuda_compiler_caps_t *caps,
                                       int f32_ok, int f64_ok);
void acuda_compiler_set_vulkan_memory_model(acuda_compiler_caps_t *caps,
                                            int available);
void acuda_compiler_set_bda(acuda_compiler_caps_t *caps, int available);
// Native float atomics: each bit mirrors a VK_EXT_shader_atomic_float
// (or VK_EXT_shader_atomic_float2) feature.
enum {
    ACUDA_ATOMIC_F32_SHARED = 1u << 0,
    ACUDA_ATOMIC_F32_BUFFER = 1u << 1,
    ACUDA_ATOMIC_F64_SHARED = 1u << 2,
    ACUDA_ATOMIC_F64_BUFFER = 1u << 3,
    ACUDA_ATOMIC_IMAGE_F32_ADD = 1u << 4,
    ACUDA_ATOMIC_IMAGE_F32_MINMAX = 1u << 5,
};
void acuda_compiler_set_atomic_float(acuda_compiler_caps_t *caps,
                                     unsigned features);
unsigned acuda_compiler_get_atomic_float(const acuda_compiler_caps_t *caps);
void acuda_compiler_set_surface_alias(acuda_compiler_caps_t *caps, int available);
void acuda_compiler_set_int64_image(acuda_compiler_caps_t *caps, int available);

void acuda_kernel_free(acuda_kernel_t *kernel);
acuda_status_t acuda_kernel_info(const acuda_kernel_t *kernel,
                                 acuda_kernel_info_t *out_info);
const acuda_kernel_info_t *acuda_kernel_info_ptr(const acuda_kernel_t *kernel);
acuda_status_t acuda_kernel_spirv(const acuda_kernel_t *kernel,
                                  const uint32_t **out_words,
                                  size_t *out_word_count);
acuda_status_t acuda_kernel_release_spirv(acuda_kernel_t *kernel);
acuda_status_t acuda_kernel_serialize(const acuda_kernel_t *kernel,
                                      uint8_t **out_data, size_t *out_size);
acuda_status_t acuda_kernel_deserialize(const uint8_t *data, size_t size,
                                        acuda_kernel_t **out_kernel);
acuda_status_t acuda_kernel_deserialize_file(const char *path,
                                             acuda_kernel_t **out_kernel);
uint32_t acuda_kernel_segment_count(const acuda_kernel_t *kernel);
acuda_status_t acuda_kernel_segment(const acuda_kernel_t *kernel,
                                    uint32_t index,
                                    acuda_kernel_t **out_segment);
uint32_t acuda_kernel_launch_ptr_params(const acuda_kernel_t *kernel);

#ifdef __cplusplus
}
#endif

#endif
