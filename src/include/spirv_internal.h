// Internal shared declarations for the PTX -> SPIR-V emitter.
#ifndef ACUDA_SPIRV_INTERNAL_H
#define ACUDA_SPIRV_INTERNAL_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "ptx.h"
#include "acuda/compiler.h"

#ifndef ACUDA_DEBUG_PRINT
#define ACUDA_DEBUG_PRINT(...)                    \
    do {                                          \
        if (getenv("ACUDA_TRACE"))                \
            fprintf(stderr, __VA_ARGS__);         \
    } while (0)




#endif

#define MAX_PARAMS 256
// SASS-to-PTX materialises a temporary per unrolled operation.  PyTorch's
// vectorised elementwise kernels reach ~4200 declared registers, and the
// fully unrolled block radix sort (radixSortKVInPlace<...,32,32>) declares
// ~12,800 (mostly %rt/%rdt address temporaries).  Give the table enough
// headroom for those without changing the per-compile code paths.
#define MAX_REGS 32768
#define LOCAL_SIZE_X 256u

// ---- SPIR-V opcode numbers (verified against SPIRV-Headers grammar) ----
enum {
    SPV_NOP_NO = 0,
    S_Capability = 17, S_Extension = 10, S_MemoryModel = 14, S_EntryPoint = 15,
    S_String = 7, S_Line = 8,
    S_ExecutionMode = 16, S_Name = 5, S_Decorate = 71,
    S_MemberDecorate = 72, S_TypeVoid = 19, S_TypeBool = 20,
    S_TypeInt = 21, S_TypeFloat = 22, S_TypeVector = 23,
    S_TypeImage = 25, S_TypeSampler = 26, S_TypeSampledImage = 27,
    S_TypeArray = 28, S_TypeRuntimeArray = 29, S_TypeStruct = 30,
    S_TypePointer = 32, S_TypeFunction = 33, S_Constant = 43,
    S_ConstantTrue = 41, S_ConstantFalse = 42, // grammar-verified
    S_Variable = 59, S_Function = 54, S_Label = 248, S_Branch = 249,
    S_BranchCond = 250, S_SelectionMerge = 247, S_Return = 253,
    S_Switch = 251,
    S_ExtInstImport = 11, S_ExtInst = 12, // grammar-verified
    S_SampledImage = 86,
    S_ImageTexelPointer = 60, S_ImageRead = 98, S_ImageWrite = 99,
    S_ImageSampleImplicitLod = 87, S_ImageSampleExplicitLod = 88,
    S_ImageSampleDrefExplicitLod = 90, S_ImageFetch = 95,
    S_ImageGather = 96, S_ImageDrefGather = 97, S_Image = 100,
    S_ImageQuerySizeLod = 103, S_ImageQuerySize = 104,
    S_ImageQueryLevels = 106, S_ImageQuerySamples = 107,
    S_ReadClockKHR = 5056,
    S_FunctionEnd = 56, S_Load = 61, S_Store = 62, S_AccessChain = 65,
    S_PtrAccessChain = 67,
    S_CompositeConstruct = 80, S_CompositeExtract = 81, S_Bitcast = 124,
    S_UConvert = 113, S_FConvert = 115,
    S_ConvertFToU = 109, S_ConvertFToS = 110, // grammar-verified
    S_ConvertSToF = 111, S_ConvertUToF = 112, // grammar-verified
    S_SpecConstant = 50, S_SpecConstantComposite = 51,
    S_ExecutionModeId = 331,
    S_IAdd = 128, S_ISub = 130, S_IMul = 132, S_FAdd = 129,
    S_UMulExtended = 151, // low/high u64 product in a two-member struct
    S_FSub = 131, S_FMul = 133, S_ShrL = 194,
    S_UGreaterEqual = 174, S_ULess = 176, S_ULessEqual = 178,
    S_UGreater = 172, S_IEqual = 170, S_INotEqual = 171,
    S_ULessThan = 176, S_ULessThanEqual = 178,
    S_FOrdEqual = 180, S_FOrdNotEqual = 182, S_FOrdLess = 184,
    S_FOrdLessEqual = 188, S_FOrdGreater = 186, S_FOrdGreaterEqual = 190,
    S_FUnordNotEqual = 183, // grammar-verified
    S_FUnordEqual = 181, S_FUnordLessThan = 185, S_FUnordGreaterThan = 187,
    S_FUnordLessThanEqual = 189, S_FUnordGreaterThanEqual = 191, // grammar-verified
    S_ConstantComposite = 44,
    S_SConvert = 114, S_ShrA = 195, S_And = 199, S_Fma = 46,
    S_Shl = 196, S_Or = 197, S_Select = 169, S_LogicalOr = 166,
    S_LogicalAnd = 167, S_LogicalNot = 168, // grammar-verified
    S_LogicalNotEqual = 165, // bool XOR
    S_LoopMerge = 246, S_Phi = 245,
    S_SLess = 177, S_SLessEqual = 179, S_SGreater = 173,
    S_SGreaterEqual = 175, S_Not = 200, S_SNegate = 126, S_FNegate = 127,
    S_UDiv = 134, S_FDiv = 136, S_UMod = 137, S_Xor = 198,
    S_ControlBarrier = 224, S_MemoryBarrier = 225, // grammar-verified
    S_AtomicLoad = 227, // grammar-verified
    S_AtomicCAS = 230, S_AtomicIAdd = 234, // grammar-verified
    S_AtomicFAddEXT = 6035, // SPV_EXT_shader_atomic_float_add
    S_AtomicFMinEXT = 5614, // SPV_EXT_shader_atomic_float_min_max
    S_AtomicFMaxEXT = 5615, // SPV_EXT_shader_atomic_float_min_max
    S_AtomicExchange = 229, S_AtomicSMin = 236, S_AtomicUMin = 237,
    S_AtomicSMax = 238, S_AtomicUMax = 239, S_AtomicAnd = 240,
    S_AtomicOr = 241, S_AtomicXor = 242,
    S_FmaKHR = 4427, // SPV_KHR_fma OpFmaKHR (guaranteed fused)
    S_ConvertUToPtr = 120, // PhysicalStorageBuffer address -> pointer
    S_SDot = 4450, S_UDot = 4451, S_SUDot = 4452, // SPV_KHR_integer_dot_product
    S_SDotAccSat = 4453, S_UDotAccSat = 4454,
    S_SUDotAccSat = 4455, // SPV_KHR_integer_dot_product
    S_FDot2MixAcc32VALVE = 6916,                 // SPV_VALVE_mixed_float_dot_product
    S_GroupShuffle = 345, // OpGroupNonUniformShuffle, grammar-verified
    S_GroupBallot = 339, S_GroupBallotExtract = 341, // grammar-verified
    S_BitCount = 205, // grammar-verified
    S_GroupIAdd = 349, S_GroupSMin = 353, S_GroupUMin = 354, // grammar-verified
    S_GroupSMax = 356, S_GroupUMax = 357, // grammar-verified
    S_GroupBitwiseAnd = 359, S_GroupBitwiseOr = 360,
    S_GroupBitwiseXor = 361, // grammar-verified
};
enum {
    C_Shader = 1, C_Int64 = 11, C_Int64Atomics = 12, C_Float64 = 10,
    C_ImageGatherExtended = 25, C_ImageCubeArray = 34,
    C_Sampled1D = 43, C_Image1D = 44, C_SampledCubeArray = 45,
    C_ImageQuery = 50,
    C_FmaKHR = 6030,
    C_AtomicFloat32AddEXT = 6033, // SPV_EXT_shader_atomic_float_add
    C_AtomicFloat64AddEXT = 6034, // SPV_EXT_shader_atomic_float_add
    C_AtomicFloat32MinMaxEXT = 5612, // SPV_EXT_shader_atomic_float_min_max
    C_Int64ImageEXT = 5016, // SPV_EXT_shader_image_int64
    C_ShaderClockKHR = 5055, // SPV_KHR_shader_clock
    C_DotProduct = 6019, C_DotProductInput4x8BitPacked = 6018,
    C_DotProductFloat16AccFloat32VALVE = 6912,
    C_StorageBuffer8BitAccess = 4448, C_StorageBuffer16BitAccess = 4433,
    C_Int16 = 22, C_Float16 = 9,
    C_DenormPreserve = 4464, C_SignedZeroInfNanPreserve = 4466,
    C_VulkanMemoryModel = 5345, C_VulkanMemoryModelDeviceScope = 5346,
    C_PhysicalStorageBufferAddresses = 5347,
    C_GroupShuffle = 65, // grammar-verified
    C_GroupBallot = 64, // OpGroupNonUniformBallot, grammar-verified
    C_GroupArith = 63, C_GroupClustered = 67, // grammar-verified
    C_ShaderNonUniform = 5301, C_RuntimeDescriptorArray = 5302,
    C_StorageImageArrayNonUniformIndexing = 5309,
    SC_UniformConstant = 0, SC_Input = 1, SC_Uniform = 2, SC_PushConstant = 9, SC_StorageBuffer = 12,
    SC_PhysicalStorageBuffer = 5349,
    D_ArrayStride = 6, D_BuiltIn = 11, D_DescriptorSet = 34, D_Binding = 33,
    D_SpecId = 1,
    D_Block = 2, D_Offset = 35, D_NonUniform = 5300,
    B_WorkgroupSize = 25, B_WorkgroupId = 26, B_LocalInvocationId = 27,
    B_NumWorkGroups = 24,
};
typedef struct {
    char name[64];
    ptx_ty_t decl_ty;
    ptx_ty_t cur_ty; // effective type of `cur` (SSA id), may be an alias
    int defined;
    int is_coopmat;
    int cur_is_coopmat;
    int coopmat_comp;
    int coopmat_use;
    uint32_t coopmat_ty;
    uint32_t cur;
    int is_ptr;
    int binding;
    uint32_t off;
    int shvar; // >=0: pointer into Workgroup shared variable shvar
               // (binding unused); -1: SSBO binding or scalar
    int glvar; // >=0: pointer into .global image variable glvar
               // (read-only; stores stay clean errors)
    int paddr; // >=0: symbolic address of kernel parameter paddr
    int is_tex; // 1 texref, 2 samplerref, 0 ordinary pointer/value
    int lovar;  // >=0: backing .local variable index (binding == -2)
    int dyn_home; // 1: home resolved at runtime via dyn_tag
    uint32_t dyn_tag; // Function variable holding the home tag
               // (mov.u64 %rd, param); readable via ld.param [%rd]
} reg_t;

typedef struct {
    char name[512];
    int kind; // 0 = 32-bit scalar (1 push word), 1 = pointer (1 binding),
              // 2 = 64-bit scalar (2 push words)
    int is_tex; // 1 = buffer-backed .texref/.samplerref descriptor
    int is_hidden_symbol; // module-scope .global/.const hidden BDA pointer param
    int binding;
    int push_word;
    int push_words;
    ptx_ty_t ty;
} cparam_t;

typedef struct {
    uint64_t bits;
    uint32_t id;
} constent_t;

// Interned non-aggregate type declarations.  Identical OpTypeImage /
// OpTypeSampledImage / OpTypePointer spellings must share one id:
// the validator rejects duplicates.
typedef struct {
    uint32_t op, w0, w1, w2, w3, w4, w5, w6;
    uint32_t id;
} typent_t;

typedef struct {
    int used;
    int dim, arrayed, isint;
    int cas;  // R32ui alias + integer CAS fallback
    int lock; // R32f image protected by the global surface lock
    int binding;
    uint32_t img_ty;
    uint32_t elem_ptr_ty;
    uint32_t var;
} surf_table_t;

typedef struct {
    spv_blob_t head;  // capabilities, memory model
    spv_blob_t entry; // entry point, execution modes, names (emitted late)
    spv_blob_t ann;   // decorations
    spv_blob_t types; // types, constants, globals
    spv_blob_t fn;    // function body
    char *err;
    size_t errlen;
    int failed;
    uint32_t next_id;
    uint32_t t_void, t_bool, t_u32, t_u64, t_f32, t_f64, t_bf16, t_v3u32, t_uvec4;
    uint32_t t_u64x2; // lazily declared { u64, u64 } for OpUMulExtended
    uint32_t sp_x, sp_y, sp_z; // workgroup-size spec constants (0, 1, 2)
    uint32_t sp_dyn; // dynamic-.shared word count spec constant (3)
    // Per-arena CUdeviceptr translation table, specialization constants
    // 4.. : a {token_base, size, delta_lo, delta_hi} quad per arena slot.
    // A zero size marks the slot as unused, so the emitted chain degrades
    // to the identity when the driver supplies nothing.  The delta is
    // 64-bit: arena device addresses are not required to be reachable
    // within 32 bits of their host mapping.
    uint32_t sp_arena[8][4];
    int has_extern;  // any .extern .shared variable present
    uint32_t t_ptr_in_v3, t_wordarr, t_ssbo, t_ptr_ssbo, t_ptr_u_ssbo;
    uint32_t t_pusharr, t_push, t_ptr_push, t_ptr_u_push, t_funty;
    // Static .shared variables (Workgroup storage, one per ptx_shvar_t).
    const ptx_shvar_t *shvars;
    int nshvars;
    // Initialized .const arrays (loads materialize as immediates).
    const ptx_const_t *consts;
    int nconsts;
    uint32_t const_bank_var;
    uint32_t const_bank_bytes;
    uint32_t const_bank_size;
    int has_const_bank;
    // Initialized .global arrays (loads served from a read-only
    // per-thread Private image; stores stay clean errors).
    const ptx_const_t *globals;
    int nglobals;
    const ptx_extfunc_t *extfuncs;
    int nextfuncs;
#ifndef ACUDA_MAX_GLOBALS
#define ACUDA_MAX_GLOBALS 256
#endif
    uint32_t gl_vars[ACUDA_MAX_GLOBALS];
    // Entry params (for distinguishing unused-but-declared params,
    // whose dead loads define undef, from typos).
    const ptx_param_t *fnparams;
    int nfnparams;
    // Function table (module scope): callee names for function-address
    // moves (mov.<ty> %rd, <funcname> -> compile-time callee home).
    const ptx_func_t *fns;
    const ptx_func_t *cur_fn; // current function (host-call state masks)
    int host_machine; // emit the per-invocation PC state machine
    int cfg_machine;  // 1 = generic CFG state machine (kind 8)
    int host_home[MAX_REGS];     // HOME_* for each register
    int host_home_aux[MAX_REGS]; // binding/shvar/glvar/param index
    uint64_t host_home_mask[MAX_REGS]; // possible homes (0..31 global)
    uint32_t host_paddr_mask[MAX_REGS]; // possible parameter-address slots
    uint8_t host_dynamic[MAX_REGS];
    uint32_t host_tag_var[MAX_REGS];
    uint32_t host_reg_var[MAX_REGS]; // Function variable per register
    // Materialized 32-bit word of a dynamic pointer-word register (the
    // lifted 32-bit addressing idiom holds both a value word and an
    // address offset in the same register).  Zero when unused.
    uint32_t host_word_var[MAX_REGS];
    // 1 when the register is live across a block edge and therefore
    // needs a state slot (generic CFG state machine only).
    uint8_t host_need[MAX_REGS];
    uint32_t host_pc_var;
    uint32_t t_ptr_fn_u32, t_ptr_fn_u64, t_ptr_fn_f32, t_ptr_fn_f64,
             t_ptr_fn_bool;
    int host_call_count;
    uint32_t host_ret_off[32];
    uint32_t host_state_stride;
    uint32_t host_local_base;
    uint32_t host_local_off[32];
    int host_nlocals;
    uint32_t host_shared_stride;
    uint32_t host_shared_off[32];
    uint32_t host_global_base;
    uint32_t host_global_off[ACUDA_MAX_GLOBALS];
    int host_nglobals;
    int nfns;
    uint32_t t_ptr_w_u32; // pointer Workgroup u32 (shared element access)
    uint32_t t_ptr_w_f32; // pointer Workgroup f32 (native float atomics)
    uint32_t t_ptr_w_u64; // pointer Workgroup u64 (64-bit shared vars/atomics)
    uint32_t t_ptr_w_f64; // pointer Workgroup f64 (native f64 atomics)
    uint32_t t_v2f32, t_v3f32, t_v4f32, t_v2u32, t_v4u64;
    uint32_t t_sampler_ty; // OpTypeSampler
    uint32_t t_cm_f16[3];
    uint32_t t_cm_bf16[3];
    uint32_t t_cm_f32[3];
    int uses_coopmat;
    int sh_coop_comp;   // 0=f16, 1=bf16, -1=untyped shared
    int sh_coop_typed;  // all static shared vars are 16-bit WMMA tiles
    uint32_t t_coop_elem;
    uint32_t t_ptr_w_coop;
    uint32_t tex_vars[MAX_PARAMS]; // image/sampler descriptor variables
    uint32_t tex_image_ty[MAX_PARAMS]; // OpTypeImage for each binding
    uint32_t tex_sampled_ty[MAX_PARAMS]; // OpTypeSampledImage per binding
    // Per-binding image shape, derived from the first using instruction.
    int tex_dim[MAX_PARAMS];     // 0 1D, 1 2D, 2 3D, 3 Cube
    int tex_arrayed[MAX_PARAMS];
    int tex_ms[MAX_PARAMS];
    int tex_sampler_var[MAX_PARAMS]; // OpTypeSampler variable (samplerref)
    int tex_split; // kernel declares a .samplerref: texrefs are images
    int tex_meta_word[MAX_PARAMS]; // first push word of per-binding metadata
    int tex_int[MAX_PARAMS];       // sampled type is integer (for txq)
    int surf_f32_atomic[MAX_PARAMS]; // binding is used by native sured f32
    int surf_f32_cas[MAX_PARAMS];    // R32_UINT alias + CAS fallback
    int surf_f32_lock[MAX_PARAMS];   // R32f image protected by the lock
    int surf_u64[MAX_PARAMS]; // binding is an R64ui surface
    int surf_lock; // module uses the hidden global surface lock buffer
    int uses_int64_image; // Int64ImageEXT is used
    int uses_int64_image_atomic; // + Int64Atomics
    // Dynamic cudaSurfaceObject_t tables (one per surface type).
    int surf_table;
    int surf_table_count;
    surf_table_t surf_tables[ACUDA_SURF_TABLE_MAX_TYPES];
    uint32_t surf_meta_var;
    uint32_t surf_meta_ptr_u32;
    uint32_t t_ptr_img_u32;        // pointer Image u32 (storage-image atomics)
    uint32_t t_ptr_img_u64;        // pointer Image u64 (R64ui surfaces)
    uint32_t t_ptr_img_f32;        // pointer Image f32 (image float atomics)
    uint32_t lock_var;             // hidden StorageBuffer { u32 lock } variable
#define ACUDA_MAX_SHVARS 32
    uint32_t sh_vars[ACUDA_MAX_SHVARS];
    // One Workgroup u32 used as a spinlock when a 64-bit shared atomic
    // has to be expressed on the u32-viewed shared array (lifted SASS
    // declares __shared as bytes, so no u64 pointer exists).
    uint32_t sh_lock_var;
    // Function-scope .local variables (Private per-invocation storage,
    // one u32-word array per variable, uninitialized like CUDA).
    const ptx_shvar_t *locals;
    int nlocals;
    uint32_t t_ptr_priv_u32; // pointer Private u32 (local element access)
#define ACUDA_MAX_LOCALS 64
    uint32_t lo_vars[ACUDA_MAX_LOCALS];
    uint32_t fn_id;
    uint32_t v_tid, v_ntid, v_ctaid, v_nctaid, v_subid;
    int uses_frnd; // module has directed-rounding add/mul
    int use_fma;   // and the device supports VK_KHR_shader_fma
    int use_fma_f64; // f64 variant passed its own probe
    int use_dot_product; // module has dp4a and the device supports it
    int zero_init_wg; // driver zero-initializes Workgroup storage
    int use_ssbo8; // native 8-bit storage-buffer access
    int use_ssbo16; // native 16-bit storage-buffer access
    int use_fc32, use_fc64; // float-control execution modes
    int use_vmm; // Vulkan memory model capability
    int use_bda; // pointer params are PhysicalStorageBuffer addresses
    int uses_brkpt; // module has brkpt: emit OpString/OpLine
    uint32_t v_file; // OpString id for PTX source lines
    uint32_t bda_word[MAX_PARAMS]; // push-constant word per binding
    uint32_t t_ptr_phys_u32, t_ptr_phys_u64, t_ptr_phys_u16, t_ptr_phys_f32,
             t_ptr_phys_f64;
    uint32_t phys_ptrs[256];  // physical pointer ids + alignment
    uint32_t phys_align[256];
    int nphys;
    uint32_t t_u8; // OpTypeInt 8 0 (lazily created)
    uint32_t t_u16; // OpTypeInt 16 0 (lazily created)
    uint32_t ssbo8_vars[MAX_PARAMS]; // byte-view alias per binding
    uint32_t ssbo8_ptr_ty[MAX_PARAMS];
    uint32_t ssbo16_vars[MAX_PARAMS]; // halfword-view alias per binding
    uint32_t ssbo16_ptr_ty[MAX_PARAMS];
    int uses_i64_atomics; // module uses 64-bit atomic operations
    uint32_t c_tid, c_ntid, c_ctaid, c_nctaid;
    uint32_t ssbo_vars[MAX_PARAMS];
    // Lazily-created u64 alias views of the same storage-buffer
    // bindings (for 64-bit atomics; the u32 word view cannot be used).
    uint32_t ssbo64_vars[MAX_PARAMS];
    uint32_t ssbo64_ptr_ty[MAX_PARAMS];
    uint32_t push_len; // u32 const id holding npush (created before prelude)
    int nbindings;
    uint32_t push_var;
    int npush;
    constent_t cu32[8192];
    int nu32;
    constent_t cu64[2048];
    int nu64;
    constent_t cf32[1024];
    int nf32;
    constent_t cf64[1024];
    int nf64;
#define ACUDA_MAX_TYPEKEYS 256
    typent_t typekeys[ACUDA_MAX_TYPEKEYS];
    int ntypekeys;
    // f64 emulation (ACUDA_EMU_F64=1/true/yes): demote f64 to f32.
    // Explicit opt-in only; native path is untouched when unset.
    int emu_f64;
    int uses_f64; // any f64 reg decl or instruction type
    int uses_shfl; // any shfl/match/mma instruction (needs GroupNonUniformShuffle)
    int uses_ballot; // any vote/activemask (needs GroupNonUniformBallot)
    int uses_state_lockstep;    // kind-8 machine dispatches at min PC so
                                // divergent lanes reconverge
    int uses_state_lockstep_wg; // when set with lockstep, use workgroup
                                // scope min-PC (needed for bar.sync/red)
    int uses_barrier; // any workgroup/barrier primitive
    int uses_f16_native; // native Float16 arithmetic is used
    int uses_fdot2;      // OpFDot2MixAcc32VALVE is used
    int uses_atomic_f32_add; // OpAtomicFAddEXT on f32 is used
    int uses_atomic_f64_add; // OpAtomicFAddEXT on f64 is used
    int uses_atomic_f32_minmax; // OpAtomicFMin/MaxEXT on f32 is used
    int uses_shader_clock; // %clock / %clock64 (ShaderClockKHR)
    int shared_f32_view; // shared vars are f32 (scalar-f32 module only)
    int shared_f64_view; // 8-byte shared vars are f64 (scalar-f64 module only)
    uint32_t t_f16, t_v2f16;
    // Within-block CSE for f16/bf16 unpacking (cleared at OpLabel).
    uint32_t cse_gen;
    uint32_t cse_split_key[256], cse_split_hi[256], cse_split_lo[256];
    uint32_t cse_split_gen[256];
    uint32_t cse_cvt_key[256], cse_cvt_val[256], cse_cvt_gen[256];
    uint8_t cse_cvt_kind[256];
    // Within-block reuse of the sparse 2:4 A expansion for
    // OpFDot2MixAcc32VALVE (cleared at OpLabel via cse_gen).
    uint32_t sp_pack_gen;
    uint32_t sp_pack_key[6];
    uint32_t sp_pack_a[2][8][2];
    int uses_redux; // any redux (needs Arithmetic+Clustered)
    int uses_image_query; // any txq image query (needs ImageQuery)
    int uses_sampled1d;   // 1D image types
    int uses_image1d;     // 1D storage image access
    int uses_cube_array;  // cube-array image types
    int uses_gather_ext;  // runtime-offset gathers
    int uses_commons;     // .common globals need prologue zeroing
    int last_local;       // .local index from the last resolve_addr
     int addr_size;        // 32 or 64 (from .address_size)
    int uses_swbar;    // any counted/arrive barrier (software barrier state)
    int bar_sw_id[17]; // barrier ids that use the software counter (16 = cluster)
    uint32_t bar_var;  // Workgroup u32[16] counter array (software barriers)
    uint32_t lockstep_min_var; // Workgroup u32 for workgroup-scope min-PC dispatch
    int uses_subid; // any subgroup-lane use (shfl/vote/redux/match)
    reg_t regs[MAX_REGS];
    int nregs;
    cparam_t cparams[MAX_PARAMS];
    int ncparams;
    uint32_t *block_ids;
    int nblocks;
    uint32_t bulk_i_var; // Private u64 loop counter for bulk copies
    uint32_t cc_cf; // extended-arithmetic carry slot (CF bool SSA id)
    int cc_defined; // set by the first .cc writer; reads before fail honestly
    uint32_t glsl450; // GLSL.std.450 import id (transcendentals)
    char tex_dst_tmp[4][64]; // scratch: current texture destinations
    acuda_compiler_caps_t caps;
} ctx_t;

// A resolved SSA value. For pointer regs, id/off describe (binding, byte
// offset): id is unused, off is the u64 offset SSA id.
typedef struct {
    uint32_t id;
    ptx_ty_t ty;
    int is_ptr;
    int binding;
    uint32_t off;
} val_t;

// Per-register state snapshotted at block boundaries.
typedef struct {
    uint32_t id;
    uint32_t off; // pointer byte-offset SSA id
    ptx_ty_t cur_ty;
    int defined;
    int is_ptr;
    int binding;
    int shvar;
    int glvar;
    int paddr;
    int lovar;
} acuda_flow_snap_t;

// Control-flow shape and per-block snapshots.
typedef struct {
    int kind; // 0 = straight, 1 = guard, 2 = diamond, 3 = loop, 4 = flow
    int j;    // branch block (cbra)
    int t;    // then target (cbra); == J for guard
    int e;    // else target (diamond only)
    int J;    // join (diamond only)
    int pred_neg;
    char pred[64];
    // Loop (kind 3): header H with body H..L and merge M (all blocks
    // in between are straight-line; the parser splits at every branch
    // so cbras are always terminal).
    // Form (a): H ends with cbra exiting forward to M (M = final ret
    // block); body H+1..L is straight-line; L ends with bra.uni to H.
    // Form (b): H ends with cbra to ITSELF; H+1 is the exit path.
    // Exactly one cbra and one back-edge in the whole kernel either way.
    int loop_h;   // header block
    int loop_l;   // latch block (== H for form (b))
    int loop_M;   // merge block (final for (a), H+1 for (b))
    int loop_self; // 1 = form (b), 0 = form (a)
    // Nested loop (kind 9): outer H..L with one inner loop H2..L2
    // inside the outer body; both form (a), inner exit == outer latch.
    int loop_h2, loop_l2, loop_M2;
    int pred_neg2;
    char pred2[64];
    // Flow (kind 4): per-block tables, malloc'd (cfg_free releases).
    int *selfloop; // 1 = block ends with cbra to itself (loop header)
    int *fwdtgt;   // forward cbra target block, -1 when none
    // Join support: snapshot of every register at each block's end
    // (value id, or byte-offset id for pointers), plus live-in sets so
    // merge blocks can phi path-dependent values (OpUndef where a
    // path leaves the register undefined, matching PTX semantics).
    int flow_nregs;
    acuda_flow_snap_t *flow_snap; // [nblocks * flow_nregs]
    char *flow_live; // live-in [nblocks * flow_nregs]
} cfg_t;



// Helpers shared across emitter translation units.
void fail(ctx_t *c, const char *msg);
void failf(ctx_t *c, int line, const char *msg);
void put(ctx_t *c, spv_blob_t *b, uint32_t w);
uint32_t nid(ctx_t *c);
uint32_t const_u32(ctx_t *c, uint32_t v);
int check_dst(ctx_t *c, const char *dst, ptx_ty_t ty, int line);
val_t resolve(ctx_t *c, spv_blob_t *b, const ptx_opd_t *op, ptx_ty_t want,
              int line);
void def_val(ctx_t *c, const char *dst, ptx_ty_t ty, uint32_t id);
int resolve_addr(ctx_t *c, spv_blob_t *b, const ptx_opd_t *op,
                 int *binding, uint32_t *off, int *shvar, int *glvar,
                 int line);
int resolve_addr_ex(ctx_t *c, spv_blob_t *b, const ptx_opd_t *op,
                    int *binding, uint32_t *off, int *shvar, int *glvar,
                    int64_t *sh_imm, int line);
uint32_t sh_addr_ptr(ctx_t *c, spv_blob_t *b, int var, uint32_t off);
uint32_t sh_addr_ptr_imm(ctx_t *c, spv_blob_t *b, int var, uint32_t off,
                         int64_t imm);
uint32_t sh_addr_ptr64(ctx_t *c, spv_blob_t *b, int var, uint32_t off);
uint32_t bda_convert(ctx_t *c, spv_blob_t *b, uint32_t pty, uint32_t off);
// 32-bit CUdeviceptr token -> 64-bit device address (arena token table).
uint32_t token_to_addr(ctx_t *c, spv_blob_t *b, uint32_t token);
uint32_t pack_lane_get(ctx_t *c, spv_blob_t *b, uint32_t v, int lane,
                       int width, int sign);
uint32_t swbar_ptr(ctx_t *c, spv_blob_t *b, uint32_t idx);
void emit_swbar_init(ctx_t *c, spv_blob_t *b);
void emit_common_init(ctx_t *c, spv_blob_t *b);
uint32_t addr_ptr(ctx_t *c, spv_blob_t *b, int binding, uint32_t off);
uint32_t atom_wrap_loop(ctx_t *c, spv_blob_t *b, uint32_t ptr,
                        int is_shared, uint32_t bound);
uint32_t atom_dec_loop(ctx_t *c, spv_blob_t *b, uint32_t ptr,
                       int is_shared, uint32_t bound);
uint32_t atom_fadd64_loop(ctx_t *c, spv_blob_t *b, uint32_t ptr,
                          uint32_t add, int is_shared);
uint32_t atom_fadd_loop(ctx_t *c, spv_blob_t *b, uint32_t ptr,
                        int is_shared, uint32_t add);
uint32_t ssbo64_ptr(ctx_t *c, spv_blob_t *b, int binding, uint32_t off);

// Opcode-family emitters (one instruction; caller has already decoded it).
void emit_atom(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
uint32_t const_u64(ctx_t *c, uint64_t v);
reg_t *find_reg(ctx_t *c, const char *name);
void emit_load(ctx_t *c, spv_blob_t *b, uint32_t ty, uint32_t id,
               uint32_t ptr);
void emit_arith(ctx_t *c, spv_blob_t *b, uint32_t op, uint32_t ty,
                uint32_t id, uint32_t a, uint32_t c2);
uint32_t f16_to_f32val(ctx_t *c, spv_blob_t *b, uint32_t h, int line);
uint32_t sat_f32(ctx_t *c, spv_blob_t *b, uint32_t x, int line);
uint32_t bf16_to_f32val(ctx_t *c, spv_blob_t *b, uint32_t h, int line);
uint32_t pred_val(ctx_t *c, spv_blob_t *b, const ptx_opd_t *op, int neg,
                  int line);
uint32_t ballot_warp_word(ctx_t *c, spv_blob_t *b, uint32_t bal,
                          uint32_t W);
void emit_group_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
uint32_t acuda_coopmat_type(ctx_t *c, int comp, int use);
int emit_texture_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
int emit_surface_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
int tex_ref_binding(ctx_t *c, const char *name, int *binding,
                    uint32_t *off, int *is_tex, int line);
void emit_mem_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void ensure_f16_types(ctx_t *c);
void emit_fma_into(ctx_t *c, spv_blob_t *b, uint32_t ty, uint32_t dst,
                   uint32_t x, uint32_t y, uint32_t z);
uint32_t emit_f64_fma_rnd(ctx_t *c, spv_blob_t *b, uint32_t x, uint32_t y,
                          uint32_t z, int mode, int line);
// dot(f16x2 a, f16x2 b) + f32 accumulator (OpFDot2MixAcc32VALVE).
uint32_t emit_fdot2_f32(ctx_t *c, spv_blob_t *b, uint32_t a_bits,
                        uint32_t b_bits, uint32_t acc);
void emit_host_request(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_host_reply(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_host_spill(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_host_restore(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
void emit_host_result(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
int build_host_homes(ctx_t *c, const ptx_func_t *fn, char *err,
                     size_t errlen);
void emit_host_machine(ctx_t *c, const ptx_func_t *fn);
void emit_cfg_machine(ctx_t *c, const ptx_func_t *fn);
uint32_t host_mem_off(ctx_t *c, spv_blob_t *b, int line, uint32_t field);
uint32_t host_shared_off_bytes(ctx_t *c, spv_blob_t *b, int var,
                               uint32_t byteoff);
uint32_t host_global_off_bytes(ctx_t *c, spv_blob_t *b, int var,
                               uint32_t byteoff);
uint32_t host_local_off_bytes(ctx_t *c, spv_blob_t *b, int var,
                              uint32_t byteoff);
uint32_t sh_word_ptr(ctx_t *c, spv_blob_t *b, int var, uint32_t word);
void emit_instr(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
int block_index_of(ctx_t *c, const ptx_func_t *fn, const char *label);
int block_terminator(const ptx_block_t *b);
void emit_block_body(ctx_t *c, spv_blob_t *b, const ptx_block_t *blk,
                     int skip_last);
void emit_branch(ctx_t *c, spv_blob_t *b, uint32_t target);
int analyze_cfg(ctx_t *c, const ptx_func_t *fn, cfg_t *cfg);
void cfg_free(cfg_t *cfg);
void emit_flow_block(ctx_t *c, const ptx_func_t *fn, const cfg_t *cfg,
                     int bi, const uint32_t *tramp,
                     const uint32_t *exitcopy, uint32_t *flow_cur);
void emit_loop(ctx_t *c, const ptx_func_t *fn, const cfg_t *cfg);
void emit_nested_loop(ctx_t *c, const ptx_func_t *fn, const cfg_t *cfg);
void emit_head(ctx_t *c);
void emit_prelude(ctx_t *c);
void emit_entry_ann(ctx_t *c, const ptx_func_t *fn);
void emit_arith_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
uint32_t clz32(ctx_t *c, spv_blob_t *b, uint32_t x, int line);
uint32_t clz64_val(ctx_t *c, spv_blob_t *b, uint32_t v, int line);
uint32_t brev32_val(ctx_t *c, spv_blob_t *b, uint32_t v);
uint32_t bfe_value(ctx_t *c, spv_blob_t *b, uint32_t a64, uint32_t bpos, uint32_t clen, int w64, int sign, int line);
uint32_t bfi_value(ctx_t *c, spv_blob_t *b, uint32_t a64, uint32_t b64, uint32_t cpos, uint32_t dlen, int w64, int line);
uint32_t half_ftz_bits(ctx_t *c, spv_blob_t *b, int is_bf16, uint32_t bits, int line);
uint32_t half_minnan_lane(ctx_t *c, spv_blob_t *b, int is_bf16, int is_max, uint32_t av, uint32_t bv, int line);
uint32_t mul_hi_value(ctx_t *c, spv_blob_t *b, ptx_ty_t ty,
                      int sign, uint32_t a, uint32_t bval, int line);
uint32_t f16_satfinite_bits(ctx_t *c, spv_blob_t *b, uint32_t bits, int line);
uint32_t pack_lane_put(ctx_t *c, spv_blob_t *b, uint32_t out, uint32_t lane, int idx, int width, uint32_t mask);
uint32_t f32_directed_div_exact(ctx_t *c, spv_blob_t *b, uint32_t fa, uint32_t fb, int mode, int line);
uint32_t f64_directed_div_exact(ctx_t *c, spv_blob_t *b, uint32_t fa, uint32_t fb, int mode, int line);
uint32_t flush_f16_bits(ctx_t *c, spv_blob_t *b, uint32_t bits, int line);
uint32_t set_bop_join(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in, uint32_t t, int line);
uint32_t ptr_token_u64(ctx_t *c, spv_blob_t *b, const reg_t *r, int line);
uint32_t const_f32(ctx_t *c, uint32_t bits);
uint32_t const_f64(ctx_t *c, uint64_t bits);
uint32_t cvt_pack_lane(ctx_t *c, spv_blob_t *b, uint32_t v, int sign, int bits, int line);
void def_global(ctx_t *c, const char *dst, int var, uint32_t off);
void def_shared(ctx_t *c, const char *dst, int var, uint32_t off);
uint32_t do_cmp(ctx_t *c, spv_blob_t *b, ptx_cmp_t cmp, ptx_ty_t ty, int sign, uint32_t a, uint32_t b2, int line);
uint32_t f16_from_fp8byte(ctx_t *c, spv_blob_t *b, uint32_t p, int e4m3, int line);
uint32_t f16_to_f32val(ctx_t *c, spv_blob_t *b, uint32_t h, int line);
uint32_t sat_f32(ctx_t *c, spv_blob_t *b, uint32_t x, int line);
uint32_t f32bits_to_f64bits(ctx_t *c, spv_blob_t *b, uint32_t f);
uint32_t f32_to_bf16bits(ctx_t *c, spv_blob_t *b, uint32_t f, int line);
uint32_t f32_to_f16bits(ctx_t *c, spv_blob_t *b, uint32_t f, int line);
uint32_t f64bits_to_f32bits(ctx_t *c, spv_blob_t *b, uint32_t d);
uint32_t f64_to_f16bits(ctx_t *c, spv_blob_t *b, uint32_t f, int line);
uint32_t flush_f32_denorm(ctx_t *c, spv_blob_t *b, uint32_t f, int line);
uint32_t fp8byte_from_f32(ctx_t *c, spv_blob_t *b, uint32_t f, int e4m3, int line);

int emit_cvt_micro(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
int emit_elect_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
int emit_mbarrier_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
int emit_predicated_instr(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in,
                          int (*fn)(ctx_t *, spv_blob_t *,
                                    const ptx_instr_t *));
void emit_mbarrier_complete_tx(ctx_t *c, spv_blob_t *b, uint32_t ptr,
                               uint32_t bytes);
void emit_red_async(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in);
uint32_t lane_id(ctx_t *c, spv_blob_t *b, int line);
int surf_ref_is_static(ctx_t *c, const char *ref, int depth);
void surf_scan_dynamic(ctx_t *c);
uint32_t pack2(ctx_t *c, spv_blob_t *b, uint32_t hi, uint32_t lo);
uint32_t pack_lane_fma(ctx_t *c, spv_blob_t *b, int is_bf16, uint32_t av, uint32_t bv, uint32_t cv, int ftz, int sat, int relu, int line);
uint32_t relu_f32(ctx_t *c, spv_blob_t *b, uint32_t x, int line);
uint32_t popc32(ctx_t *c, spv_blob_t *b, uint32_t x);
val_t resolve_f32_bits(ctx_t *c, spv_blob_t *b, const ptx_opd_t *op, int line);
uint32_t round_f32_directed(ctx_t *c, spv_blob_t *b, uint32_t x, int rnd, int line);
uint32_t round_f64_directed(ctx_t *c, spv_blob_t *b, uint32_t x, int rnd, int line);
uint32_t sel_bool(ctx_t *c, spv_blob_t *b, uint32_t cond, uint32_t t,
                  uint32_t f);
uint32_t sel_u32(ctx_t *c, spv_blob_t *b, uint32_t cond, uint32_t t, uint32_t f);
uint32_t sel_u64(ctx_t *c, spv_blob_t *b, uint32_t cond, uint32_t t, uint32_t f);
void split_pack(ctx_t *c, spv_blob_t *b, uint32_t p, uint32_t *hi, uint32_t *lo);
int analyze_params(const ptx_unit_t *unit, const ptx_func_t *fn,
                   int *kinds_out, char *err, size_t errlen);
int decl_int_width(ptx_ty_t ty);
ptx_ty_t eff_ty(ctx_t *c, ptx_ty_t ty);
uint32_t ty_of(ctx_t *c, ptx_ty_t ty);
void def_ptr(ctx_t *c, const char *dst, int binding, uint32_t off);
void emit_store(ctx_t *c, spv_blob_t *b, uint32_t ptr, uint32_t val);
void emit_push_load(ctx_t *c, spv_blob_t *b, uint32_t *id, ptx_ty_t ty,
                    int word, int line);
uint32_t f32bits_to_f64bits(ctx_t *c, spv_blob_t *b, uint32_t f);
uint32_t f64bits_to_f32bits(ctx_t *c, spv_blob_t *b, uint32_t d);
int find_const(ctx_t *c, const char *name);
cparam_t *find_cparam(ctx_t *c, const char *name);
int find_local(ctx_t *c, const char *name);
uint32_t gl_addr_ptr(ctx_t *c, spv_blob_t *b, int var, uint32_t off);
uint32_t lo_addr_ptr(ctx_t *c, spv_blob_t *b, int var, uint32_t off);
uint32_t gl_load_u64(ctx_t *c, spv_blob_t *b, int var, uint32_t off);
uint32_t const_bank_load_u32(ctx_t *c, spv_blob_t *b, uint32_t off);
uint32_t const_bank_load_u64(ctx_t *c, spv_blob_t *b, uint32_t off);
uint32_t const_bank_load_narrow(ctx_t *c, spv_blob_t *b, uint32_t off,
                                int width);
uint32_t load_u64(ctx_t *c, spv_blob_t *b, int binding, uint32_t off);
uint32_t lo_load_u64(ctx_t *c, spv_blob_t *b, int var, uint32_t off);
void lo_store_u64(ctx_t *c, spv_blob_t *b, int var, uint32_t off,
                  uint32_t val);
uint32_t narrow_ld_global(ctx_t *c, spv_blob_t *b, int binding,
                          uint32_t off, int width);
uint32_t narrow_ld_glvar(ctx_t *c, spv_blob_t *b, int var, uint32_t off,
                         int width);
uint32_t narrow_ld_shared(ctx_t *c, spv_blob_t *b, int var, uint32_t off,
                          int width);
void narrow_st_global(ctx_t *c, spv_blob_t *b, int binding, uint32_t off,
                      uint32_t val, int width);
void narrow_st_shared(ctx_t *c, spv_blob_t *b, int var, uint32_t off,
                      uint32_t val, int width);
uint32_t sh_load_u64(ctx_t *c, spv_blob_t *b, int var, uint32_t off);
void sh_store_u64(ctx_t *c, spv_blob_t *b, int var, uint32_t off,
                  uint32_t val);
void store_u64(ctx_t *c, spv_blob_t *b, int binding, uint32_t off,
               uint32_t val);

uint32_t half_arith_lane(ctx_t *c, spv_blob_t *b, int is_bf16,
                         uint32_t fop, int ftz, int sat,
                         uint32_t av, uint32_t bv, int line);
void f32x2_split(ctx_t *c, spv_blob_t *b, uint32_t u, uint32_t *lo, uint32_t *hi);
uint32_t f32x2_pack(ctx_t *c, spv_blob_t *b, uint32_t lo, uint32_t hi);
uint32_t f32_addsub_lane(ctx_t *c, spv_blob_t *b, int is_add,
                         uint32_t fa, uint32_t fb, int frnd,
                         int ftz, int sat, int line);
uint32_t add_carry_u64(ctx_t *c, spv_blob_t *b, int w64,
                       uint32_t a, uint32_t b64, uint32_t cin,
                       uint32_t *sum_out, int line);
uint32_t sub_borrow_u64(ctx_t *c, spv_blob_t *b, int w64,
                        uint32_t a, uint32_t b64, uint32_t cin,
                        uint32_t *diff_out, int line);
uint32_t f32_soft_addsub(ctx_t *c, spv_blob_t *b, uint32_t fa,
                         uint32_t fb, int is_sub, int mode, int line);
uint32_t f64_soft_addsub(ctx_t *c, spv_blob_t *b, uint32_t fa,
                         uint32_t fb, int is_sub, int mode, int line);
uint32_t f32_mul_lane(ctx_t *c, spv_blob_t *b, uint32_t fa,
                      uint32_t fb, int frnd, int ftz, int sat, int line);
uint32_t f32_directed_mul_soft(ctx_t *c, spv_blob_t *b, uint32_t fa,
                               uint32_t fb, int mode, int line);
uint32_t f64_directed_mul_soft(ctx_t *c, spv_blob_t *b, uint32_t fa,
                               uint32_t fb, int mode, int line);
uint32_t f32_fma_lane(ctx_t *c, spv_blob_t *b, uint32_t fa,
                      uint32_t fb, uint32_t fc, int ftz, int sat, int line);
uint32_t f32_fma3(ctx_t *c, spv_blob_t *b, uint32_t x, uint32_t y, uint32_t z);
uint32_t f32_glsl_fma3(ctx_t *c, spv_blob_t *b, uint32_t x, uint32_t y, uint32_t z);
uint32_t f32_soft_fma(ctx_t *c, spv_blob_t *b, uint32_t fa,
                      uint32_t fb, uint32_t fc, int mode, int line,
                      uint32_t *exact_sign, uint32_t *exact_nz);
uint32_t f64_soft_fma(ctx_t *c, spv_blob_t *b, uint32_t fa,
                      uint32_t fb, uint32_t fc, int mode, int line,
                      uint32_t *exact_sign, uint32_t *exact_nz);
uint32_t bool_bin(ctx_t *c, spv_blob_t *b, uint32_t op, uint32_t x, uint32_t y);
uint32_t cmp_u32(ctx_t *c, spv_blob_t *b, ptx_cmp_t cmp, int sign,
                 uint32_t x, uint32_t y, int line);
uint32_t f32_relu_nonneg(ctx_t *c, spv_blob_t *b, uint32_t v, int line);
uint32_t f32_to_bits(ctx_t *c, spv_blob_t *b, uint32_t f);
uint32_t int_to_float_dir(ctx_t *c, spv_blob_t *b, uint32_t v,
                          int sign, int wsrc, ptx_ty_t ty,
                          int mode, int line);
uint32_t u32_bin(ctx_t *c, spv_blob_t *b, uint32_t op, uint32_t x, uint32_t y);
void bulk_copy_loop(ctx_t *c, spv_blob_t *b, int dst_sh,
                    int dst_gl, int dst_binding, uint32_t dst_off,
                    int src_sh, int src_gl, int src_binding,
                    uint32_t src_off, uint32_t size, int line);
uint32_t fold_sh_imm(ctx_t *c, spv_blob_t *b, uint32_t off, int64_t imm);
uint32_t swbar_add(ctx_t *c, spv_blob_t *b, uint32_t ptr, uint32_t val);
void swbar_wait(ctx_t *c, spv_blob_t *b, uint32_t ptr, uint32_t count);
uint32_t bool_const(ctx_t *c, int v);
uint32_t wmma_subid(ctx_t *c, spv_blob_t *b);
uint32_t wmma_shfl(ctx_t *c, spv_blob_t *b, uint32_t val, uint32_t lane, uint32_t base);
uint32_t sparse_pick_b(ctx_t *c, spv_blob_t *b, const uint32_t raw[2][2], int cc, uint32_t idx, int line);
uint32_t wmma_index(ctx_t *c, spv_blob_t *b, uint32_t row, uint32_t col, uint32_t stride, int rowmajor);
uint32_t wmma_ld_half(ctx_t *c, spv_blob_t *b, int shvar, int glvar, int binding, uint32_t off);
uint32_t wmma_ld_bytes4(ctx_t *c, spv_blob_t *b, int shvar, int glvar, int binding, uint32_t off);
uint32_t wmma_ld_f32(ctx_t *c, spv_blob_t *b, int shvar, int glvar, int binding, uint32_t off);
void wmma_st_f32(ctx_t *c, spv_blob_t *b, int shvar, int glvar, int binding, uint32_t off, uint32_t v);
uint32_t wmma_byte_off(ctx_t *c, spv_blob_t *b, uint32_t base_off, uint32_t idx, int eshift);
uint32_t dot4_packed(ctx_t *c, spv_blob_t *b, int asign, int bsign, uint32_t a, uint32_t bv);
void unpack4_to_bytes(ctx_t *c, spv_blob_t *b, uint32_t w, int is_signed, uint32_t out[2]);

// Internal SPIR-V emitter helpers (split across spirv/*.c)
void spv_blob_free(spv_blob_t *blob);
void fail(ctx_t *c, const char *msg);
void failf(ctx_t *c, int line, const char *msg);
void put(ctx_t *c, spv_blob_t *b, uint32_t w);
uint32_t nid(ctx_t *c);
void emit1(ctx_t *c, spv_blob_t *b, uint32_t op, uint32_t w0);
void emit_str(ctx_t *c, spv_blob_t *b, const char *s);
ptx_ty_t eff_ty(ctx_t *c, ptx_ty_t ty);
uint32_t ty_of(ctx_t *c, ptx_ty_t ty);
uint32_t const_u32(ctx_t *c, uint32_t v);
uint32_t const_u64(ctx_t *c, uint64_t v);
uint32_t const_f32(ctx_t *c, uint32_t bits);
uint32_t const_f64(ctx_t *c, uint64_t bits);
uint32_t type_intern(ctx_t *c, uint32_t op, const uint32_t *ops, uint32_t nops);
reg_t *find_reg(ctx_t *c, const char *name);
int decl_int_width(ptx_ty_t ty);
cparam_t *find_cparam(ctx_t *c, const char *name);
int find_shvar(ctx_t *c, const char *name);
int find_local(ctx_t *c, const char *name);
int find_const(ctx_t *c, const char *name);
int find_global(ctx_t *c, const char *name);
int block_index_of(ctx_t *c, const ptx_func_t *fn, const char *label);
void emit_decorate(ctx_t *c, uint32_t target, uint32_t deco);
void emit_decorate1(ctx_t *c, uint32_t target, uint32_t deco, uint32_t lit);
void emit_member_decorate(ctx_t *c, uint32_t target, uint32_t member, uint32_t deco, uint32_t lit);
int need_phys_addr(ctx_t *c);
void emit_head(ctx_t *c);
void surf_scan_dynamic(ctx_t *c);
int surf_ref_is_static(ctx_t *c, const char *ref, int depth);
void emit_prelude(ctx_t *c);
void emit_entry_ann(ctx_t *c, const ptx_func_t *fn);
void emit_store(ctx_t *c, spv_blob_t *b, uint32_t ptr, uint32_t val);
uint32_t ptr_token_u64(ctx_t *c, spv_blob_t *b, const reg_t *r, int line);
int check_dst(ctx_t *c, const char *dst, ptx_ty_t ty, int line);
void def_val(ctx_t *c, const char *dst, ptx_ty_t ty, uint32_t id);
void def_ptr(ctx_t *c, const char *dst, int binding, uint32_t off);
void def_paddr(ctx_t *c, const char *dst, int pidx);
void def_shared(ctx_t *c, const char *dst, int var, uint32_t off);
void def_global(ctx_t *c, const char *dst, int var, uint32_t off);
uint32_t phys_align_of(ctx_t *c, uint32_t ptr);
uint32_t load_u64(ctx_t *c, spv_blob_t *b, int binding, uint32_t off);
uint32_t f32bits_to_f64bits(ctx_t *c, spv_blob_t *b, uint32_t f);
uint32_t f64bits_to_f32bits(ctx_t *c, spv_blob_t *b, uint32_t d);
uint32_t lane_id(ctx_t *c, spv_blob_t *b, int line);
uint32_t pack2(ctx_t *c, spv_blob_t *b, uint32_t hi, uint32_t lo);
uint32_t addr_ptr(ctx_t *c, spv_blob_t *b, int binding, uint32_t off);
uint32_t gl_addr_ptr(ctx_t *c, spv_blob_t *b, int var, uint32_t off);
uint32_t gl_load_u64(ctx_t *c, spv_blob_t *b, int var, uint32_t off);
uint32_t const_bank_load_u32(ctx_t *c, spv_blob_t *b, uint32_t off);
uint32_t const_bank_load_u64(ctx_t *c, spv_blob_t *b, uint32_t off);
uint32_t sh_addr_ptr(ctx_t *c, spv_blob_t *b, int var, uint32_t off);
uint32_t sh_addr_ptr64(ctx_t *c, spv_blob_t *b, int var, uint32_t off);
uint32_t sh_word_ptr(ctx_t *c, spv_blob_t *b, int var, uint32_t word);
uint32_t swbar_ptr(ctx_t *c, spv_blob_t *b, uint32_t idx);
void emit_swbar_init(ctx_t *c, spv_blob_t *b);
void emit_common_init(ctx_t *c, spv_blob_t *b);
uint32_t sh_load_u64(ctx_t *c, spv_blob_t *b, int var, uint32_t off);
uint32_t lo_addr_ptr(ctx_t *c, spv_blob_t *b, int var, uint32_t off);
uint32_t lo_load_u64(ctx_t *c, spv_blob_t *b, int var, uint32_t off);
uint32_t ssbo64_ptr(ctx_t *c, spv_blob_t *b, int binding, uint32_t off);
uint32_t flush_f32_denorm(ctx_t *c, spv_blob_t *b, uint32_t f, int line);
uint32_t popc32(ctx_t *c, spv_blob_t *b, uint32_t x);
uint32_t clz32(ctx_t *c, spv_blob_t *b, uint32_t x, int line);
uint32_t special_x(ctx_t *c, spv_blob_t *b, ptx_special_t sp, int line);
uint32_t special_comp(ctx_t *c, spv_blob_t *b, ptx_special_t sp,
                      int comp, int line);
uint32_t bda_convert(ctx_t *c, spv_blob_t *b, uint32_t pty, uint32_t addr);
uint32_t u16_type(ctx_t *c);
uint32_t phys_u16_ty(ctx_t *c);
uint32_t bda_word_ptr(ctx_t *c, spv_blob_t *b, uint32_t addr, uint32_t *sh);
uint32_t f32_neg(ctx_t *c, spv_blob_t *b, uint32_t f);
uint32_t f64_neg(ctx_t *c, spv_blob_t *b, uint32_t f);
uint32_t bool_not(ctx_t *c, spv_blob_t *b, uint32_t x);
uint32_t u64_bin(ctx_t *c, spv_blob_t *b, uint32_t op, uint32_t x, uint32_t y);
uint32_t trunc_u64_u32(ctx_t *c, spv_blob_t *b, uint32_t v);
uint32_t zext_u32_u64(ctx_t *c, spv_blob_t *b, uint32_t v);
uint32_t bits_to_f32(ctx_t *c, spv_blob_t *b, uint32_t v);
uint32_t bits_to_f64(ctx_t *c, spv_blob_t *b, uint32_t v);
uint32_t f64_to_bits(ctx_t *c, spv_blob_t *b, uint32_t f);
uint32_t f32_round_pack(ctx_t *c, spv_blob_t *b, uint32_t total, uint32_t extra, uint32_t eVal, uint32_t sRes, int mode, int line);
uint32_t f64_round_pack(ctx_t *c, spv_blob_t *b, uint32_t total, uint32_t extra, uint32_t eVal, uint32_t sRes, int mode, int line);
uint32_t f16_round_pack(ctx_t *c, spv_blob_t *b, uint32_t total, uint32_t extra, uint32_t eVal, uint32_t sRes, int mode, int line);



uint32_t arith_cmp_u64(ctx_t *c, spv_blob_t *b, ptx_cmp_t cmp, uint32_t x, uint32_t y, int line);


void emit_range(ctx_t *c, spv_blob_t *b, const ptx_block_t *blk, size_t from, size_t to_excl);
uint32_t last_emitted_label(const spv_blob_t *blob);

#endif
