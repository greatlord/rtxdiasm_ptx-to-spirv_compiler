// Internal PTX lexer + IR + SPIR-V emitter declarations.
#ifndef ACUDA_PTX_H
#define ACUDA_PTX_H

#include <stddef.h>
#include <stdint.h>
#include "acuda/compiler.h"

// Upper bound on module-scope `.global` images (read-only data tables
// and assertion strings emitted by the SASS lifter).
#ifndef ACUDA_MAX_GLOBALS
#define ACUDA_MAX_GLOBALS 256
#endif

typedef enum {
    T_EOF, T_IDENT, T_INT, T_FLOAT, T_BITS, T_PUNCT, T_STR
} ptx_tok_kind_t;

typedef struct {
    ptx_tok_kind_t kind;
    char text[512];
    uint64_t imm;   // T_INT value, T_FLOAT/T_BITS raw bits
    int is_float;   // T_INT that is actually 0F hex-float bits
    char punct;     // T_PUNCT character
    int line;
} ptx_tok_t;

typedef struct {
    const char *src;
    size_t pos;
    int line;
    ptx_tok_t saved;
    int has_saved;
} ptx_lexer_t;

void ptx_lex_init(ptx_lexer_t *lx, const char *src);
int ptx_next(ptx_lexer_t *lx, ptx_tok_t *out, char *err, size_t errlen);
void ptx_unget(ptx_lexer_t *lx, const ptx_tok_t *tok);

// ---- IR ----
typedef enum {
    PTY_U32, PTY_U64, PTY_F32, PTY_F64, PTY_PRED,
    PTY_U8, PTY_U16, PTY_F16,
    PTY_BF16, PTY_F16X2, PTY_BF16X2
    // Narrow integer regs hold zero-extended canonical values (low
    // bits valid); F16 regs hold the exact f32 value (rounded at
    // every def). BF16 regs hold 16-bit brain-float patterns in u32
    // SSA (like F16 halves-as-bits, but widened/narrowed with bf16
    // bias/widths); F16X2/BF16X2 regs hold two 16-bit patterns in
    // one u32 (HIGH half = first source). No narrow SSA types exist:
    // U8/U16 lower to u32, F16 lowers to f32, packed/bf16 lower to u32.
} ptx_ty_t;
typedef enum { PSP_PARAM, PSP_GLOBAL, PSP_GENERIC, PSP_SHARED, PSP_LOCAL, PSP_CONST } ptx_space_t;

// Symbolic address token for a module-scope .const symbol. Consts are
// inlined as immediates, so these tokens exist only for address
// identity (mov.u64 %rd, c; compare), never for dereference.
#define PTX_CONST_ADDR_BASE 0x8000000000000000ull
typedef enum {
    POP_MOV, POP_ADD, POP_SUB, POP_MULLO, POP_MADLO, POP_MULWIDE, POP_CVT,
    POP_CVTPACK8,
    POP_SETP, POP_LD, POP_ST, POP_CPASYNC, POP_CPASYNC_BULK, POP_BRA, POP_CBRA, POP_RET, POP_EXIT, POP_FMA,
    POP_SHR, POP_SHL, POP_VSHR, POP_AND, POP_SELP, POP_NOT, POP_NEG, POP_MINMAX,
    POP_DIV, POP_REM, POP_OR, POP_XOR, POP_BAR, POP_BARRED, POP_MUL24, POP_RND,
    POP_SHF, POP_ABS, POP_POPC, POP_CLZ, POP_BREV,
    POP_BFE, POP_BFI, POP_BMSK, POP_BFIND,
    POP_MADWIDE, POP_MULHI, POP_DP4A, POP_DP2A, POP_SAD, POP_PRMT,
    POP_DOT2, // ACuda extension: f16x2 dot with f32 accumulator
    POP_GLSL, POP_ADDSAT, POP_COPYSIGN, POP_MINNAN,
    POP_ATOM, POP_ATOMCAS, POP_CALL, POP_SHFL, POP_VOTE, POP_REDUX, POP_MATCH,
    POP_MEMBAR, POP_NANOSLEEP, POP_CVTPACK, POP_SET, POP_MMA, POP_LDMATRIX,
    POP_FP8_TO_F16X2, POP_F32_TO_FP8X2, // FP8_TO_F16X2: 1 b16 src, dst b32,
    POP_BRKPT, // brkpt: emitted as an OpLine source marker
    POP_ELECT, POP_MBARRIER, // elect.sync / mbarrier family
    POP_RED_ASYNC, // red.async / multimem.red.async
    POP_CVT_MICRO, // e2m1/e2m3/e3m2/ue8m0/s2f6 conversions (see cvt_* fields)
    POP_FNS, POP_SZEXT, POP_CLMAD, POP_MAD24, // extra integer families
    POP_TESTP, // floating-point property test
    POP_RCP,    // reciprocal (with optional directed rounding)
    POP_CNOT, POP_LOP3, POP_SLCT, POP_PREFETCH, POP_HINT, POP_BARWARP,
    POP_ISSPACEP,
    POP_TEX, POP_TXQ, POP_ISTYPEP, POP_TLD4,
    POP_SULD, POP_SUST, POP_SURED, POP_SUQ,
    POP_BRX, // brx.idx: lowered to a compare/branch chain
    POP_HOSTCALL, POP_HOST_REQUEST, POP_HOST_REPLY,
    POP_HOST_SPILL, POP_HOST_RESTORE, POP_HOST_RESULT,
    POP_WMMA_LOAD, POP_WMMA_STORE, POP_WMMA_MMA
                         // fp8 selects e4m3/e5m2; F32_TO_FP8X2: 2 f32
                         // srcs (first -> high byte), dst b16
                         // MEMBAR: arg = 0 cta / 1 gl / 2 sys;
                         // NANOSLEEP: hint only, operand type-checked
                         // CVTPACK: ty = pack (F16X2/BF16X2), 2 f32 srcs;
                         // SET: u32-dest compare, ty = dst flavor
                         // (U32 scalar-full or packed u16 masks, F16X2
                         // packed f16 1.0/0.0), ty2 = cmp type
                         // (F16/F16X2), cmp = GT/GTU
                         // MMA: m16n8k16 row.col f32.f16/bf16 and
                         // m16n8k32 row.col s32.s8, regs in mregs
                         // (4 dst + 4 A + 2 B + 4 C)
                         // LDMATRIX: .m8n8.x{1,2,4}[.trans].b16, dst
                         // regs in mregs[0..ldm_num), src = shared addr
} ptx_op_t;
typedef enum { PCM_GE, PCM_LT, PCM_LE, PCM_GT, PCM_EQ, PCM_NE, PCM_NAN, PCM_NUM, PCM_EQU, PCM_NEU, PCM_LTU, PCM_LEU, PCM_GTU, PCM_GEU } ptx_cmp_t;

typedef enum {
    OPD_REG, OPD_IMM, OPD_SPECIAL, OPD_PARAM, OPD_ADDR, OPD_REGOFF,
    OPD_UNDEF // synthesized unknown value (predicated-desugar false arm)
} ptx_opd_kind_t;

typedef enum { SPC_TID, SPC_NTID, SPC_CTAID, SPC_NCTAID, SPC_LANEID, SPC_LANEMASKLT } ptx_special_t;

typedef struct {
    ptx_opd_kind_t kind;
    char reg[64];    // OPD_REG dst or src register name
    uint64_t imm;    // OPD_IMM raw bits (int value or f32 bits)
    int imm_is_float;
    // Exact source text of a float literal (for f64: strtod at resolve,
    // so inexact decimals keep full precision; "" when N/A).
    char imm_text[128];
    ptx_special_t special; // OPD_SPECIAL
    int comp;              // vector component 0..2 (x/y/z)
    char param[256];       // OPD_PARAM / OPD_ADDR base param name
    char addr_reg[64];     // OPD_ADDR address register
    int64_t addr_imm;      // OPD_ADDR constant byte offset
    uint64_t regoff_imm;   // OPD_REGOFF constant added to reg[]
} ptx_opd_t;

typedef struct {
    ptx_op_t op;
    ptx_ty_t ty;    // primary type (.u32/.u64/.f32), PTY_PRED unused
    ptx_ty_t ty2;   // CVT source type
    ptx_space_t space;
    ptx_cmp_t cmp;
    int sign; // signed variant (.s32/.s64)
    int pack; // packed-integer lane count: 2 (.u16x2/.s16x2) or 4 (.u8x4/.s8x4)
    int sat_sub; // POP_ADDSAT: 1 = saturating subtract
    int sz_clamp; // POP_SZEXT: 1 = .clamp, 0 = .wrap
    int is_max; // POP_MINMAX: 1 = max, 0 = min
    int is_hi; // POP_MUL24: 1 = hi (bits 47..16), 0 = lo
    int rnd; // POP_RND f32 rounding mode: 1 = rni (nearest-even), 2 = rzi (toward-zero)
    int sat; // POP_CVT integer saturation (sat.u32.s32: clamp negatives to 0)
    int packbits; // POP_CVTPACK8: cvt.pack lane width in bits
    int dsign; // POP_CVT integer destination signedness (wide-dst fill)
    int shf_dir; // POP_SHF: 1 = left (high-32 of V<<c), 0 = right (low-32 of V>>c)
    int shf_clamp; // POP_SHF: 1 = clamp (amounts >= 32 yield a/b), 0 = wrap
    int vs_clamp; // POP_VSHR: 1 = .clamp (shift >= width saturates), 0 = mask
    int vs_add;   // POP_VSHR: 1 = .add (add the third source after shifting)
    int cp_size;  // POP_CPASYNC: bytes to write to the shared destination
    int src_size; // POP_CPASYNC: source bytes read (rest zero-filled)
    int cp_bulk;  // POP_CPASYNC_BULK: 1 global->shared, 2 shared->global
    // POP_MBARRIER: operation code and modifiers.
    int mb_op;
    int mb_nocomplete, mb_expect, mb_parity, mb_layout;
    int cc_in; // extended arithmetic (addc/subc/madc): reads the CF slot
    int cc_out; // extended arithmetic (add.cc/sub.cc/addc.cc/...): writes the CF slot
    int ftz; // flush denormals to zero (inputs and results)
    int frnd; // directed rounding: 0 rn, 1 rz, 2 rm, 3 rp
    int approx_div; // POP_DIV: .approx/.full (hardware fdiv is allowed)
    int f32x2; // packed 2x f32 in a 64-bit holder (PTX 8.6, sm_100)
    int glop; // POP_GLSL: GLSL.std.450 extended opcode (grammar-verified numbers)
    int arg; // POP_BAR: barrier id (0..15), -1 for a dynamic full barrier
    int fence_sem;   // POP_MEMBAR: 0 = acq_rel/sc, 1 = acquire, 2 = release
    int fence_scope; // POP_MEMBAR: 0 cta, 1 cluster, 2 gpu, 3 sys
    int lop3_boolop; // POP_LOP3: 0 plain, 1 .or, 2 .and predicate output
    int tf32; // POP_CVT: cvt.{rn,rna}.tf32.f32
    int bar_arrive; // POP_BAR: 1 = bar.arrive (non-blocking arrival)
    int bar_cluster; // POP_BAR: 1 = barrier.cluster.arrive/wait
    int atom_op; // POP_ATOM: 0 = add (d = old, mem += v), 1 = inc-wrap (d = old, mem = old >= b ? 0 : old+1)
    // POP_RED_ASYNC: 0 = relaxed shared::cluster form with mbarrier
    // completion, 1 = release global form (also multimem single-device
    // fallback).  atom_op/space/ty/sign carry the reduction details.
    int rda_release, rda_scope;
    int shfl_mode; // POP_SHFL: 0 = up, 1 = down, 2 = bfly, 3 = idx
    int vote_mode; // POP_VOTE: 0 = all, 1 = any, 2 = ballot, 3 = activemask
    int redux_op; // POP_REDUX: 0 = add, 1 = min, 2 = max (sign in `sign`)
    int relu; // POP_CVTPACK: 1 = max(+0) each lane before narrowing
    int fp8; // POP_*FP8*: 1 = e4m3, 2 = e5m2
    // POP_CVT_MICRO: micro-float conversion.  cvt_dst/cvt_src are the
    // CVTMF_* codes below; sources live in src[0..3].
    int cvt_dst, cvt_src;
    int cvt_scaled, cvt_relu, cvt_satfinite, cvt_frnd, cvt_rs;
    char mregs[20][64]; // POP_MMA fragments (m8n8k4 f16 uses 8 dst +
                        // 2 A + 2 B + 8 C)
    char smreg[64];     // POP_MMA sparse: metadata register
    int mma_sel;        // POP_MMA sparse: sparsity selector (0..3)
    int mma_sparse;     // POP_MMA: 1 = mma.sp form
    char wregs[4][8][64]; // POP_WMMA: D/A/B/C fragment registers
    int wmma_kind; // 0 = A, 1 = B, 2 = C, 3 = D
    int wmma_shape; // 0 = m16n16k16, 1 = m8n32k16, 2 = m32n8k16
    int wmma_row;  // 1 = row-major, 0 = column-major
    int mma_elem; // POP_MMA: 0 = f16 lanes, 1 = bf16 lanes, 2 = s8 bytes,
                  // 3 = tf32 bits
    int mma_k;    // POP_MMA: K dimension (4, 8, 16, or 32)
    int mma_asign; // POP_MMA integer: 1 = s8 A, 0 = u8 A
    int mma_bsign; // POP_MMA integer: 1 = s8 B, 0 = u8 B
    int mma_b1_xor; // WMMA single-bit: 1 = xor, 0 = and
    int mma_f16acc; // POP_MMA: 1 = D/C are .f16x2 (f16 accumulation)
    int mma_m8f16;  // POP_MMA: 1 = the four-computation m8n8k4 f16 form
    int mma_m8s8;   // POP_MMA: 1 = m8n8k16 s8/u8 (two accumulators)
    int mma_f64_m16; // POP_MMA: 1 = f64 m16n8k4/k8/k16 (sm_90)
    int isp_space;   // POP_ISSPACEP: 0 shared, 1 global, 2 local,
                     // 3 const, 4 param
    // Geometry: 1 1d, 2 2d, 3 3d, 4 a1d, 5 a2d, 6 cube, 7 acube,
    // 8 2dms, 9 a2dms.
    int tex_geom;     // POP_TEX/TLD4: geometry code
    int tex_mode;     // POP_TEX: 0 base, 1 level, 2 grad
    int tex_ncoord;   // POP_TEX: coordinate components supplied
    int tex_nchan;    // POP_TEX: destination/result components (4 or 2)
    int tex_f16;      // POP_TEX: 1 = .v4.f16 dests, 2 = .v2.f16x2 dests
    int tex_query;    // POP_TXQ: 0 width, 1 height, 2 depth, ...
    int tex_comp;     // POP_TLD4: 0 r, 1 g, 2 b, 3 a
    int tex_compare;  // POP_TEX/TLD4: depth-compare operand present
    int tex_csign;    // POP_TEX: .s32 coordinate type
    int tex_noff;     // POP_TEX/TLD4: offset vector components (0 absent)
    int istypep_kind; // POP_ISTYPEP: 0 texref, 1 samplerref, 2 surfref
    char tex_ref[64]; // POP_TEX/TXQ: texture/texref register or param
    char tex_sampler[64]; // POP_TEX/TLD4: explicit sampler name, "" for none
    char tex_dst[4][64]; // POP_TEX: destination components
    char tex_coord[4][64]; // POP_TEX: coordinate registers
    char tex_off[4][64];   // POP_TEX: offset operand registers
    char tex_lod[64];      // POP_TEX.level: lod register
    char tex_dpdx[4][64];  // POP_TEX.grad: dPdx components
    char tex_dpdy[4][64];  // POP_TEX.grad: dPdy components
    char tex_cmp[64];      // POP_TEX: depth-compare value register
    char tex_dstp[64];     // optional destination predicate (d|p)
    // Surface instructions.
    int surf_geom;    // 1 1d, 2 2d, 3 3d, 4 a1d, 5 a2d
    int surf_vec;     // 1, 2 or 4
    int surf_ty;      // PTY_U8/U16/U32/U64, or PTY_F32 for sust.p
    int surf_formatted; // 1 for sust.p
    int surf_sign;    // .s32/.s64 reduction type
    int surf_redop;   // 0 add, 1 min, 2 max, 3 and, 4 or
    int surf_query;   // suq: 0 width ... 6 memory_layout
    int surf_ncoord;  // coordinate elements supplied
    char surf_ref[64];
    char surf_dst[4][64];
    char surf_coord[4][64];
    char surf_src[4][64];
    ptx_opd_t tex_coord_op[4]; // POP_TEX: parsed coordinate operands
    int ldm_num;   // POP_LDMATRIX: matrices loaded (1, 2 or 4)
    int ldm_trans; // POP_LDMATRIX: 1 = column-major (.trans)
    int host_call_id; // POP_HOSTCALL/REQUEST/RESULT: extfunc index
    int host_cont;    // POP_HOSTCALL: continuation block index
    int host_arg_off; // POP_HOST_RESULT: byte offset into the ret slot
    char pred[64];  // guard predicate, "" if none
    int pred_neg;   // @! form
    char dst[64];   // destination register, "" if none
    char dst2[64];  // setp second predicate (complement), "" if none
    int bop; // setp boolean operator: 0 = none, 1 = and, 2 = or, 3 = xor
    int bneg; // setp boolean source is negated (!pred)
    ptx_opd_t src[8]; // bfi takes 4
    ptx_opd_t *call_args; // POP_CALL/REQUEST: dynamically sized argument list
    int nsrc;
    char target[64]; // branch label; call callee name for POP_CALL
    char target_list[64]; // POP_CALL flist/fproto or POP_BRX tlist
    char call_cand[8][128]; // indirect dispatch: candidate callees
    int ncall_cand;
    int dead; // front-end scratch: drop after a peephole rewrite
    char rets[4][64]; // POP_CALL return slot/reg names
    int nrets;
    int line;
} ptx_instr_t;

// Micro-float (f4/f6/fp8/scale/s2f6) conversion format codes.
enum {
    CVTMF_E4M3X2 = 1, CVTMF_E5M2X2, CVTMF_E2M1X2, CVTMF_E2M3X2,
    CVTMF_E3M2X2, CVTMF_E4M3X4, CVTMF_E5M2X4, CVTMF_E2M1X4,
    CVTMF_E2M3X4, CVTMF_E3M2X4, CVTMF_UE8M0X2, CVTMF_S2F6X2,
    CVTMF_F16X2, CVTMF_BF16X2,
    CVTMF_SRC_F32, CVTMF_SRC_F16X2, CVTMF_SRC_BF16X2,
};

// POP_MBARRIER operations.
enum {
    MBAR_INIT = 1, MBAR_INVAL, MBAR_EXPECT_TX, MBAR_COMPLETE_TX,
    MBAR_ARRIVE, MBAR_ARRIVE_DROP, MBAR_TEST_WAIT, MBAR_TRY_WAIT,
    MBAR_PENDING, MBAR_CHECK_LAYOUT, MBAR_CP_ARRIVE,
};

typedef struct {
    ptx_instr_t *instrs;
    size_t count, cap;
    char label[64]; // "" for entry block
} ptx_block_t;

typedef struct {
    char name[256];
    ptx_ty_t ty;
} ptx_regdecl_t;

typedef struct {
    char name[256];
    int n;          // 2/4 components
    ptx_ty_t elem;  // U32/U16/U8 element type
} ptx_vecdecl_t; // vector register (scalarized to name_x/_y/... at parse)

typedef struct {
    char name[256];
    ptx_ty_t ty;
    int is_retval; // .func return slot (writes dropped, see st.param)
    int is_internal; // compiler-synthesized host-call state/request param
    int is_tex;      // .param .texref/.samplerref (buffer-backed descriptor)
    int is_hidden_texref; // module-scope .texref injected as a hidden param
    int wide_words;  // .param .u64xN: N-word by-value argument (0 = no)
    int wide_bytes;  // exact client byte size of a wide by-value blob
                     // (.b8[M] -> M, .u64xN -> N*8; 0 = not a blob)
} ptx_param_t;

typedef struct {
    char name[256];
    size_t elem;  // declared element size in bytes (1/2/4/8)
    size_t bytes; // total allocation, = dim * elemsize (0 = extern dynamic)
    int align;    // recorded .align (each var gets its own allocation,
                  // so alignment is trivially satisfied)
    int is_extern; // dynamic .shared: sized at launch, not here
    int is_common; // .common global: zero-initialized Private storage
} ptx_shvar_t;

typedef struct {
    uint32_t off;
    char name[128];
} ptx_sym_t;

typedef struct {
    char name[256];
    size_t elem;    // element bytes (1/2/4/8)
    size_t nbytes;  // dim * elem
    size_t bank_off; // byte offset in the flattened dynamic .const image
    uint8_t *bytes; // little-endian initialized data
    // Symbolic initializer lanes (function names), resolved once all
    // functions are known. A function pointer is emitted as its
    // compile-time 0x1000+index address.
    ptx_sym_t syms[16];
    int nsyms;
} ptx_const_t;

typedef struct {
    char name[256];
    ptx_ty_t ty;
    size_t bytes; // array size in bytes (scalar slots: element size)
    int align;
    int host_call_id; // >=0: return slot written by a host call
} ptx_paramslot_t; // in-function `.param` cell (call argument/return slot)

typedef struct {
    char name[256];
    size_t ret_bytes;
    size_t arg_bytes;
    int nargs;
    ptx_ty_t arg_ty[32];
    size_t arg_off[32];
} ptx_extfunc_t; // top-level `.extern .func` declaration

typedef struct {
    char name[256];
    int is_entry; // .entry (launchable) vs .func (call-only unless solo)
    ptx_param_t *params;    size_t nparams, params_cap;
    ptx_regdecl_t *regs;
    size_t nregs, regs_cap;
    ptx_block_t *blocks;
    size_t nblocks, blocks_cap;
    ptx_shvar_t *locals; // function-scope .local variables (Private storage)
    size_t nlocals, locals_cap;
    ptx_paramslot_t *slots; // in-function .param call slots (desugared pre-emit)
    size_t nslots, slots_cap;
    ptx_param_t *rparams; // `.reg`-form signature names (call-only ABI, never launch args)
    size_t nrparams, rparams_cap;
    ptx_vecdecl_t *vecs; // vector regs (scalarized at parse, never emitted)
    size_t nvecs, vecs_cap;
    uint8_t *host_skip; // per-reg: do not spill/restore (legacy chain)
    int *param_kinds_override; // per-param: fixed ABI (legacy chain)
    struct { char label[64]; char funcs[8][128]; int n; } calltargets[8];
    int ncalltargets;
    struct { char label[64]; char targets[64][64]; int n; } branchtargets[8];
    int nbranchtargets;
    int host_calls;       // 1 = emit the host state machine
    size_t host_ret_off[32]; // state-buffer offset of each call's return bytes
    uint32_t host_state_stride;
    uint32_t host_ret_total;
    uint32_t host_local_base;      // absolute state offset of the local area
    uint32_t host_local_total;
    uint32_t host_local_off[32];   // per-.local-variable offsets
    uint32_t host_shared_stride;   // per-workgroup shared area
    uint32_t host_shared_off[32];  // per-.shared-variable offsets
    uint32_t host_global_base;     // per-invocation .global image area
    uint32_t host_global_total;
    uint32_t host_global_off[ACUDA_MAX_GLOBALS];
    int uses_dynamic_const; // any ld.const with a register address
} ptx_func_t;

typedef struct {
    char name[256];
} ptx_texref_t;

typedef struct {
    ptx_func_t *funcs;
    size_t nfuncs;
    ptx_shvar_t *shvars; // module-scope .shared variables
    ptx_shvar_t *module_locals; // module-scope .local (Private) variables
    size_t nmodule_locals, module_locals_cap;
    size_t nshvars;
    ptx_const_t *consts; // module-scope initialized .const arrays
    size_t nconsts;
    size_t const_bank_bytes; // flattened dynamic-const image size
    ptx_const_t *globals; // module-scope initialized .global arrays
    size_t nglobals;      // (read-only image; stores stay clean errors)
    ptx_texref_t *texrefs; // module-scope .texref declarations
    size_t ntexrefs;
    ptx_extfunc_t *extfuncs;
    size_t nextfuncs, extfuncs_cap;
    int version_major, version_minor;
    int sm;
    char sm_name[16]; // full .target token (sm_90a, sm_120f, ...)
    int addr_size;
} ptx_unit_t;

void ptx_unit_free(ptx_unit_t *unit);

// Parses a full PTX unit. Returns 0 on success, -1 on error (err set).
int ptx_parse(const char *src, ptx_unit_t *unit, char *err, size_t errlen);

// ---- SPIR-V emitter ----

typedef struct {
    uint32_t *words;
    size_t count, cap;
    uint32_t next_id;
} spv_blob_t;

// Dynamic surface-object table metadata filled by the emitter.
typedef struct {
    int surf_table;
    int surf_table_count;
    int surf_dim[ACUDA_SURF_TABLE_MAX_TYPES];
    int surf_arrayed[ACUDA_SURF_TABLE_MAX_TYPES];
    int surf_int[ACUDA_SURF_TABLE_MAX_TYPES];
    int surf_binding[ACUDA_SURF_TABLE_MAX_TYPES];
    int surf_table_cas[ACUDA_SURF_TABLE_MAX_TYPES];
    int surf_table_lock[ACUDA_SURF_TABLE_MAX_TYPES];
    // Binding uses the R32_UINT alias view + integer CAS fallback for
    // sured.min/max.f32 when the native float image atomic is unavailable.
    int surf_f32_cas[ACUDA_MAX_PARAMS];
    int surf_f32_lock[ACUDA_MAX_PARAMS];
    int surf_lock;
    int uses_subgroup; // kernel relies on 32-lane subgroup semantics
    int uses_barrier;  // kernel uses workgroup/barrier primitives
    int requires_float64; // 0 unknown, 1 no, 2 requires shaderFloat64
    int bda;           // kernel uses PhysicalStorageBuffer addresses
    int uses_bda_access; // kernel resolves raw device addresses (global
                         // memory access not tracked per buffer)
    int n_bda;         // number of pointer params passed as addresses
    int bda_param[ACUDA_MAX_PARAMS]; // binding indices, address slot order
    int addr_base_word;              // first address push word
} ptx_emit_meta_t;

// Lowers one function to SPIR-V words. Returns 0 ok, -1 unsupported (err).
// param_kinds[i] (0=u32 scalar, 1=pointer, 2=64-bit scalar) is filled for
// the caller (skips retval params: kinds entry stays 0, unused).
// shared_bytes is filled with the total static .shared allocation.
int ptx_emit_spv(const ptx_unit_t *unit, const ptx_func_t *fn,
                 spv_blob_t *blob, int *param_kinds, size_t *shared_bytes,
                 ptx_emit_meta_t *meta, char *err, size_t errlen,
                 const acuda_compiler_caps_t *caps);

void spv_blob_free(spv_blob_t *blob);

#endif
