// Parameter ABI classification: pointer vs scalar vs unused, with
// lineage tracking through mov/add/sub/cvta and predicated updates.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "spirv_internal.h"

// ---- Phase 0: pointer-vs-value classification for .u64/.b64 params ----

// Lineage of each u64 register: source param index, -1 plain value,
// -2 ambiguous (mirrors emitter propagation rules exactly).
typedef struct {
    char name[512];
    int lin;
} linent_t;

// One entry per distinct register.  Large unrolled reduction kernels
// (SASS-lifted PyTorch) declare several thousand registers; 2048 made
// lin_set silently drop later entries, so a shared-base pointer lost
// its lineage half-way through the kernel.
#define MAX_LIN 16384

// Lineage codes below LIN_SHBASE mean "shared variable -code":
// LIN_SHBASE-var (never a param index, never -1/-2).
#define LIN_SHBASE -100
// Lineage codes at or below LIN_GLBSE mean ".global image -code":
// LIN_GLBSE-var. The shared range is effectively (-1000000, -100];
// more shared variables than the gap would collide, so analyze
// rejects absurd counts honestly (millions of decls).
#define LIN_GLBSE -1000000
// Pointer halves loaded back from a shared pointer table and repacked
// into a dynamic address: LIN_DYNP-var.
#define LIN_DYNP -2000000

// Half-word lineage: mov.b64 {lo,hi}, ptr is expanded by the parser
// into cvt/shr/cvt, losing the pointer classification of the source.
// Track which halves came from a pointer parameter so a store/reload
// through shared memory can still identify the pointer table.
#define HALF_LO 1
#define HALF_HI 2
#define HALF_SH 3
// Scalar (32-bit) parameter values use LIN_VALBASE - param_index.  They
// are values, never pointers, but still need to mark the parameter as
// used when the register is read.  The range is above LIN_SHBASE
// (-100); ACUDA_MAX_PARAMS is 64, so the ranges cannot collide.
#define LIN_VALBASE -3
typedef struct {
    char name[512];
    int kind;
    int param;
    int shvar;
} halfent_t;

static int half_find(halfent_t *tab, int n, const char *reg) {
    for (int i = 0; i < n; i++)
        if (strcmp(tab[i].name, reg) == 0)
            return i;
    return -1;
}

static void half_set(halfent_t *tab, int *n, const char *reg, int kind,
                     int param, int shvar) {
    int i = half_find(tab, *n, reg);
    if (i < 0) {
        if (*n >= MAX_LIN)
            return;
        i = (*n)++;
        snprintf(tab[i].name, sizeof(tab[i].name), "%s", reg);
    }
    tab[i].kind = kind;
    tab[i].param = param;
    tab[i].shvar = shvar;
}

static int lin_find(linent_t *tab, int n, const char *reg) {
    for (int i = 0; i < n; i++)
        if (strcmp(tab[i].name, reg) == 0)
            return tab[i].lin;
    return -1;
}

// Reconcile incoming lineage with the tracked old one across a
// predicated redefinition: the old value survives on the false arm,
// so agreement (or an untracked old) keeps incoming, while genuine
// divergence (two live lineages, or clobbering a live one) is -2 —
// an honest ambiguity error at next use, never a silent reclass.
// Unpredicated definitions return incoming unchanged.
static int pred_join(linent_t *tab, int n, const ptx_instr_t *in, int incoming) {
    if (!in->pred[0])
        return incoming;
    int old = lin_find(tab, n, in->dst);
    if (old == -2)
        return -2;
    int old_tracked = old >= 0 || old <= LIN_SHBASE;
    if (old_tracked && old != incoming)
        return -2;
    return incoming;
}

static void lin_set(linent_t *tab, int *n, const char *reg, int lin) {
    for (int i = 0; i < *n; i++) {
        if (strcmp(tab[i].name, reg) == 0) {
            tab[i].lin = lin;
            return;
        }
    }
    if (*n < MAX_LIN) {
        snprintf(tab[*n].name, sizeof(tab[*n].name), "%s", reg);
        tab[*n].lin = lin;
        (*n)++;
    }
}

static int opd_lin(linent_t *tab, int n, const ptx_opd_t *op) {
    if (op->kind == OPD_REG)
        return lin_find(tab, n, op->reg);
    if (op->kind == OPD_REGOFF)
        return lin_find(tab, n, op->reg); // base + const preserves lineage
    return -1; // immediates, specials, params(?): plain values
}

// Convert a scalar-parameter lineage code back to its parameter index,
// or -1 when the lineage is not a scalar parameter value.
static int lin_value_param(int lin) {
    if (lin <= LIN_VALBASE && lin > LIN_SHBASE)
        return LIN_VALBASE - lin;
    return -1;
}

// Record a scalar use of a parameter-backed register.  Scalar parameter
// values use their own lineage code, so they never become addresses but
// still count as a use of the parameter.
static void mark_scalar_use(int *as_val, int lin) {
    int vp = lin_value_param(lin);
    if (vp >= 0)
        as_val[vp] = 1;
    else if (lin >= 0)
        as_val[lin] = 1;
}

// kinds_out[i]: 0 = 32-bit scalar, 1 = pointer, 2 = 64-bit scalar.
int analyze_params(const ptx_unit_t *unit, const ptx_func_t *fn,
                          int *kinds_out, char *err, size_t errlen) {
    // Lineage partition safety: shared codes (LIN_SHBASE-var) must
    // not reach the .global range (LIN_GLBSE and below).
    if (unit->nshvars > (size_t)(LIN_SHBASE - LIN_GLBSE - 1)) {
        snprintf(err, errlen, "too many .shared variables");
        return -1;
    }
    int nparams = (int)fn->nparams;
    // LIN_VALBASE - i must stay above LIN_SHBASE (the shared-code
    // range); that permits at most 97 parameters.
    if (nparams > 97) {
        snprintf(err, errlen, "too many parameters");
        return -1;
    }
    int *as_addr = calloc((size_t)nparams, sizeof(int));
    int *as_val = calloc((size_t)nparams, sizeof(int));
    linent_t *lin = malloc(sizeof(*lin) * MAX_LIN);
    halfent_t *half = malloc(sizeof(*half) * MAX_LIN);
    uint64_t *shptr = calloc(unit->nshvars ? unit->nshvars : 1,
                             sizeof(*shptr));
    int nlin = 0, nhalf = 0;
    int rc = -1;
    if (!as_addr || !as_val || !lin || !half || !shptr) {
        snprintf(err, errlen, "out of memory");
        goto out;
    }
    for (int i = 0; i < nparams; i++)
        kinds_out[i] = 0;
    for (size_t bi = 0; bi < fn->nblocks; bi++) {
        const ptx_block_t *blk = &fn->blocks[bi];
        for (size_t ii = 0; ii < blk->count; ii++) {
            const ptx_instr_t *in = &blk->instrs[ii];
            int dst_pi = -1; // param defined via ld.param
            if (in->op == POP_LD && in->space == PSP_PARAM &&
                in->src[0].kind == OPD_ADDR && !in->src[0].addr_reg[0]) {
                for (int i = 0; i < nparams; i++) {
                    if (strcmp(fn->params[i].name, in->src[0].param) == 0) {
                        dst_pi = i;
                        break;
                    }
                }
                if (dst_pi >= 0 && in->dst[0]) {
                    // A large by-value struct is opaque metadata
                    // carried in a storage buffer; its words are plain
                    // scalars even when a field happens to hold an
                    // address.  Address use then goes through the
                    // scalar BDA path.
                    if ((uint32_t)fn->params[dst_pi].wide_words >
                        ACUDA_BYVAL_SSBO_WORDS) {
                        lin_set(lin, &nlin, in->dst, -1);
                        as_val[dst_pi] = 1;
} else if (fn->params[dst_pi].wide_words &&
           in->ty == PTY_U32 &&
           unit->addr_size != 32) {
    // A 32-bit word read from a by-value struct is
    // a scalar field, never a pointer (64-bit PTX).
    lin_set(lin, &nlin, in->dst, -1);
    as_val[dst_pi] = 1;
} else if (fn->params[dst_pi].wide_words &&
           in->ty == PTY_U32 &&
           unit->addr_size == 32) {
    // 32-bit PTX: a word of a by-value struct may be a
    // device pointer (e.g. PxgGpuArray.ptr). Track it as
    // the param so address use promotes to a pointer kind.
    lin_set(lin, &nlin, in->dst, dst_pi);
    as_val[dst_pi] = 1;
                    } else if (in->ty == PTY_U32) {
                        // In 32-bit address-space modules, a plain u32
                        // kernel parameter is pointer-sized (device pointers
                        // are 32-bit). Track it like a u64 pointer param so
                        // address use promotes it to a descriptor binding;
                        // pure value use still classifies as scalar.
                        if (unit->addr_size == 32 &&
                            !fn->params[dst_pi].wide_words) {
                            lin_set(lin, &nlin, in->dst, dst_pi);
                        } else {
                            // A 32-bit parameter is a scalar value, never a
                            // pointer.  Keep a per-parameter value code so
                            // scalar uses still count as a use, but do not
                            // let it participate in pointer arithmetic.
                            lin_set(lin, &nlin, in->dst, LIN_VALBASE - dst_pi);
                            // A 32-bit word read from a plain .u64 pointer
                            // parameter is its low/high half (the rtxdisam
                            // lift idiom for 32-bit SASS addressing):
                            // track the half so a later zero-extension +
                            // global access promotes the parameter to a
                            // pointer kind instead of binding a foreign
                            // descriptor with a garbage offset.
                            if (!fn->params[dst_pi].wide_words &&
                                fn->params[dst_pi].ty == PTY_U64 &&
                                !in->pred[0] &&
                                (in->src[0].addr_imm == 0 ||
                                 in->src[0].addr_imm == 4))
                                half_set(half, &nhalf, in->dst,
                                         in->src[0].addr_imm == 0 ? HALF_LO
                                                                  : HALF_HI,
                                         dst_pi, -1);
                        }
                    } else {
                        lin_set(lin, &nlin, in->dst, dst_pi);
                    }
                }
                continue;
            }
            if (in->op == POP_LD && in->space == PSP_PARAM &&
                in->src[0].kind == OPD_ADDR &&
                in->src[0].addr_reg[0]) {
                // ld.param [%rd] through a parameter address: the
                // loaded value has the addressed parameter's lineage
                // (for a pointer parameter, a pointer value).
                int l = lin_find(lin, nlin, in->src[0].addr_reg);
                if (in->dst[0])
                    lin_set(lin, &nlin, in->dst, l);
                continue;
            }
            // Pointer-word extraction.  The parser lowers
            // `mov.b64 {lo,hi}, ptr` to cvt.u32.u64 / shr.u64 32 /
            // cvt.u32.u64; remember which half came from which
            // pointer parameter.
            if (in->op == POP_CVT && in->nsrc == 1 &&
                in->src[0].kind == OPD_REG && in->dst[0] &&
                in->ty == PTY_U32 && in->ty2 == PTY_U64) {
                int sl = lin_find(lin, nlin, in->src[0].reg);
                int sh = half_find(half, nhalf, in->src[0].reg);
                if (sl >= 0) {
                    // Extracting a pointer's words reads its value.
                    // Without this a parameter that is only ever split
                    // into halves and repacked (never used as an
                    // address itself) looks unused, and the parameter
                    // load is lowered to an undefined value.
                    if (sl < nparams)
                        as_val[sl] = 1;
                    half_set(half, &nhalf, in->dst, HALF_LO, sl, -1);
                    lin_set(lin, &nlin, in->dst, -1);
                    continue;
                }
                if (sh >= 0 && half[sh].kind == HALF_HI) {
                    half_set(half, &nhalf, in->dst, HALF_HI,
                             half[sh].param, -1);
                    lin_set(lin, &nlin, in->dst, -1);
                    continue;
                }
                if (sh >= 0 && half[sh].kind == HALF_SH) {
                    half_set(half, &nhalf, in->dst, HALF_SH, -1,
                             half[sh].shvar);
                    lin_set(lin, &nlin, in->dst, -1);
                    continue;
                }
            }
            if (in->op == POP_SHR && in->ty == PTY_U64 && in->nsrc == 2 &&
                in->src[0].kind == OPD_REG &&
                in->src[1].kind == OPD_IMM && in->src[1].imm == 32 &&
                in->dst[0]) {
                int sl = lin_find(lin, nlin, in->src[0].reg);
                if (sl >= 0) {
                    // Taking the high word of a pointer reads its value
                    // (same reasoning as the cvt.u32.u64 arm above).
                    if (sl < nparams)
                        as_val[sl] = 1;
                    half_set(half, &nhalf, in->dst, HALF_HI, sl, -1);
                    lin_set(lin, &nlin, in->dst, -1);
                    continue;
                }
            }
            // Zero-extension of a pointer low half (lifted 32-bit
            // addressing: addr = zext(lo + off)): the result addresses
            // the half's parameter.  A widened high half is not a
            // usable address; leave it untracked (honest error at use).
            if (in->op == POP_CVT && in->ty == PTY_U64 &&
                in->ty2 == PTY_U32 && in->nsrc == 1 &&
                in->src[0].kind == OPD_REG && in->dst[0] &&
                !in->pred[0]) {
                int hf = half_find(half, nhalf, in->src[0].reg);
                if (hf >= 0 && half[hf].kind == HALF_LO &&
                    half[hf].param >= 0) {
                    lin_set(lin, &nlin, in->dst, half[hf].param);
                    continue;
                }
                // else fall through (generic arms kill lineage below)
            }
            // Reassemble a pointer from two shared-memory halves: the
            // result is a dynamic pointer token for that shared table.
            if (in->op == POP_MOV && in->ty == PTY_U64 && in->nsrc == 2 &&
                in->src[0].kind == OPD_REG && in->src[1].kind == OPD_REG &&
                in->dst[0]) {
                int h0 = half_find(half, nhalf, in->src[0].reg);
                int h1 = half_find(half, nhalf, in->src[1].reg);
                if (h0 >= 0 && h1 >= 0 && half[h0].kind == HALF_SH &&
                    half[h1].kind == HALF_SH &&
                    half[h0].shvar == half[h1].shvar && shptr &&
                    half[h0].shvar >= 0) {
                    lin_set(lin, &nlin, in->dst,
                            LIN_DYNP - half[h0].shvar);
                    half_set(half, &nhalf, in->dst, 0, -1, -1);
                    continue;
                }
            }
            // Store a pointer half into shared: remember which pointer
            // parameters feed this shared variable's table.
            if (in->op == POP_ST && in->space == PSP_SHARED &&
                in->nsrc >= 2 && in->src[0].kind == OPD_ADDR &&
                in->src[0].param[0] && in->src[1].kind == OPD_REG) {
                int sv = -1;
                for (size_t vi = 0; vi < unit->nshvars; vi++) {
                    if (strcmp(unit->shvars[vi].name,
                               in->src[0].param) == 0) {
                        sv = (int)vi;
                        break;
                    }
                }
                int h = half_find(half, nhalf, in->src[1].reg);
                if (sv >= 0 && h >= 0 &&
                    (half[h].kind == HALF_LO || half[h].kind == HALF_HI) &&
                    half[h].param >= 0 && half[h].param < 64)
                    shptr[sv] |= 1ull << (unsigned)half[h].param;
            }
            // Load from a shared pointer table: the loaded half may be
            // one of the stored pointer halves.
            if (in->op == POP_LD && in->space == PSP_SHARED &&
                in->src[0].kind == OPD_ADDR && in->dst[0]) {
                int sv = -1;
                if (in->src[0].param[0]) {
                    for (size_t vi = 0; vi < unit->nshvars; vi++) {
                        if (strcmp(unit->shvars[vi].name,
                                   in->src[0].param) == 0) {
                            sv = (int)vi;
                            break;
                        }
                    }
                } else if (in->src[0].addr_reg[0]) {
                    int l = lin_find(lin, nlin, in->src[0].addr_reg);
                    if (l <= LIN_SHBASE && l > LIN_GLBSE)
                        sv = LIN_SHBASE - l;
                }
                if (sv >= 0 && shptr[sv] && in->ty == PTY_U32) {
                    half_set(half, &nhalf, in->dst, HALF_SH, -1, sv);
                    continue;
                }
            }
            // Shared-variable base: mov.u64/u32 dst, svarname. Lineage
            // codes LIN_SHBASE-var (negative, never a param index).
            // .global image bases map LIN_GLBSE-var the same way.
            if (in->op == POP_MOV &&
                (in->ty == PTY_U64 || in->ty == PTY_U32) && in->dst[0] &&
                in->nsrc == 1 && in->src[0].kind == OPD_PARAM) {
                int sv = -1;
                for (size_t vi = 0; vi < unit->nshvars; vi++) {
                    if (strcmp(unit->shvars[vi].name, in->src[0].param) ==
                        0) {
                        sv = (int)vi;
                        break;
                    }
                }
                if (sv >= 0) {
                    lin_set(lin, &nlin, in->dst, LIN_SHBASE - sv);
                    continue;
                }
                int gv = -1;
                for (size_t vi = 0; vi < unit->nglobals; vi++) {
                    if (strcmp(unit->globals[vi].name, in->src[0].param) ==
                        0) {
                        gv = (int)vi;
                        break;
                    }
                }
                if (gv >= 0) {
                    lin_set(lin, &nlin, in->dst, LIN_GLBSE - gv);
                    continue;
                }
                // Address of a kernel parameter (mov.u64 %rd, param):
                // parameter storage must be pointer-backed, so mark the
                // parameter as address-used and carry its lineage.
                int pi = -1;
                for (int q = 0; q < (int)fn->nparams; q++) {
                    if (!fn->params[q].is_retval &&
                        strcmp(fn->params[q].name, in->src[0].param) == 0) {
                        pi = q;
                        break;
                    }
                }
                if (pi >= 0) {
                    as_addr[pi] = 1;
                    lin_set(lin, &nlin, in->dst, pi);
                    continue;
                }
                // Non-variable bare name: the emitter rejects it
                // honestly; clear any stale lineage meanwhile.
                lin_set(lin, &nlin, in->dst, -1);
                continue;
            }
            // 32-bit pointer-half arithmetic (lifted 32-bit SASS
            // addressing): mov/add/sub combining exactly one pointer
            // half with plain values stays that half, so a later
            // zero-extension + global access promotes the parameter.
            // Anything else (two different halves, ambiguous sides,
            // predicated redefinition) falls through and dies below.
            if ((in->op == POP_MOV || in->op == POP_ADD ||
                 in->op == POP_SUB) &&
                in->dst[0] && !in->pred[0] && in->ty == PTY_U32 &&
                in->nsrc >= 1 && in->nsrc <= 2) {
                int hkind = 0, hparam = -1, ok = 1;
                for (int s = 0; s < in->nsrc && ok; s++) {
                    if (in->src[s].kind == OPD_IMM)
                        continue;
                    if (in->src[s].kind != OPD_REG) {
                        ok = 0;
                        break;
                    }
                    int hf = half_find(half, nhalf, in->src[s].reg);
                    if (hf >= 0 &&
                        (half[hf].kind == HALF_LO ||
                         half[hf].kind == HALF_HI) &&
                        half[hf].param >= 0) {
                        if (in->op == POP_SUB && s == 1) {
                            ok = 0; // value minus half is not the half
                            break;
                        }
                        if (hkind == 0) {
                            hkind = half[hf].kind;
                            hparam = half[hf].param;
                        } else if (hkind != half[hf].kind ||
                                   hparam != half[hf].param) {
                            ok = 0; // two different halves: not a half
                            break;
                        }
                    } else {
                        int l = lin_find(lin, nlin, in->src[s].reg);
                        if (l == -2) {
                            ok = 0; // ambiguous: honest error below
                            break;
                        }
                        // A full parameter or shared/global/dynamic
                        // lineage on the other side defers to the
                        // general mov/add/sub rules below (which know
                        // about address-used bases); only plain values
                        // ride along with the half.
                        if (l >= 0 || l <= LIN_SHBASE) {
                            ok = 0;
                            break;
                        }
                        mark_scalar_use(as_val, l);
                    }
                }
                if (ok && hkind != 0) {
                    half_set(half, &nhalf, in->dst, hkind, hparam, -1);
                    lin_set(lin, &nlin, in->dst, -1);
                    continue;
                }
                // else fall through (generic arms kill lineage below)
            }
            // Propagate through mov.u64 / add.u64 / sub.u64 (incl. cvta
            // which the parser desugared to mov.u64). When both sides
            // track params, attribute to the side already used as an
            // address (base + small offset, the numba idiom); genuine
            // ptr+ptr stays ambiguous and errors at use. Shared-variable
            // lineage propagates the same way through exactly one
            // shared side; anything else involving shared codes is
            // ambiguous.  In 32-bit address-space modules, u32 mov/add/sub
            // also carry pointer arithmetic.
            if ((in->op == POP_MOV || in->op == POP_ADD ||
                 in->op == POP_SUB || in->op == POP_MADLO ||
                 in->op == POP_MADWIDE) &&
                in->dst[0] &&
                (in->ty == PTY_U64 || in->op == POP_MADLO ||
                 in->op == POP_MADWIDE ||
                 (in->ty == PTY_U32 && unit->addr_size == 32) ||
                 // A 32-bit shared byte offset (mov.u32 %r, smem;
                 // add.u32 %r, %r, idx) stays shared-derived.
                 (in->ty == PTY_U32 &&
                   (in->op == POP_ADD || in->op == POP_SUB ||
                    in->op == POP_MOV) &&
                   ((in->nsrc > 0 &&
                     (opd_lin(lin, nlin, &in->src[0]) <= LIN_SHBASE ||
                      opd_lin(lin, nlin, &in->src[0]) >= 0)) ||
                    (in->nsrc > 1 &&
                     (opd_lin(lin, nlin, &in->src[1]) <= LIN_SHBASE ||
                       opd_lin(lin, nlin, &in->src[1]) >= 0)))))) {
                int la = in->nsrc > 0 ? opd_lin(lin, nlin, &in->src[0]) : -1;
                int lb = in->nsrc > 1 ? opd_lin(lin, nlin, &in->src[1]) : -1;
                mark_scalar_use(as_val, la);
                mark_scalar_use(as_val, lb);
                if ((in->op == POP_MADLO || in->op == POP_MADWIDE) &&
                    in->nsrc == 3) {
                    // d = base + a*b: exactly one pointer source decides
                    // the result's lineage (same rule as the emitter).
                    int lc = opd_lin(lin, nlin, &in->src[2]);
                    mark_scalar_use(as_val, lc);
                    int cand[3] = { la, lb, lc };
                    int chosen = -1, nchosen = 0;
                    for (int q = 0; q < 3; q++) {
                        if (cand[q] >= 0 || cand[q] <= LIN_SHBASE) {
                            chosen = cand[q];
                            nchosen++;
                        }
                    }
                    if (nchosen == 1)
                        lin_set(lin, &nlin, in->dst,
                                pred_join(lin, nlin, in, chosen));
                    else if (nchosen == 0)
                        lin_set(lin, &nlin, in->dst,
                                pred_join(lin, nlin, in, -1));
                    else
                        lin_set(lin, &nlin, in->dst,
                                pred_join(lin, nlin, in, -2));
                    continue;
                }
                int sa = la <= LIN_SHBASE ? la : -1;
                int sb = lb <= LIN_SHBASE ? lb : -1;
                // A scalar value (LIN_VALBASE - param) used as the
                // offset must not break pointer lineage: shared_base +
                // scalar is still a shared pointer.  Only -2 means the
                // other operand is itself an ambiguous pointer.
                int a_value = la > LIN_SHBASE && la != -2;
                int b_value = lb > LIN_SHBASE && lb != -2;
                if (sa <= LIN_SHBASE || sb <= LIN_SHBASE) {
                    if (sa <= LIN_SHBASE && b_value)
                        lin_set(lin, &nlin, in->dst, pred_join(lin, nlin, in, sa));
                    else if (sb <= LIN_SHBASE && a_value)
                        lin_set(lin, &nlin, in->dst, pred_join(lin, nlin, in, sb));
                    else
                        lin_set(lin, &nlin, in->dst, pred_join(lin, nlin, in, -2));
                } else if (la >= 0 && lb >= 0) {
                    int au = as_addr[la], bu = as_addr[lb];
                    if (au && !bu)
                        lin_set(lin, &nlin, in->dst, pred_join(lin, nlin, in, la));
                    else if (!au && bu)
                        lin_set(lin, &nlin, in->dst, pred_join(lin, nlin, in, lb));
                    else
                        // Both params, both-or-neither used as an
                        // address: u64 add/sub of two addresses is
                        // plain integer arithmetic and the result is a
                        // value, not a pointer (ptxas parity;
                        // stateful_neg_offset). The emitter agrees.
                        lin_set(lin, &nlin, in->dst, pred_join(lin, nlin, in, -1));
                } else {
                    int one = la >= 0 ? la : (lb >= 0 ? lb : -1);
                    lin_set(lin, &nlin, in->dst, pred_join(lin, nlin, in, one));
                }
                continue;
            }
            // Address uses: ld/st/atom/red.async address register.
            if ((in->op == POP_LD || in->op == POP_ST || in->op == POP_ATOM ||
                 in->op == POP_ATOMCAS || in->op == POP_RED_ASYNC ||
                 in->op == POP_WMMA_LOAD || in->op == POP_WMMA_STORE) &&
                (in->op == POP_WMMA_LOAD || in->op == POP_WMMA_STORE ||
                 in->op == POP_RED_ASYNC ||
                 in->space == PSP_GLOBAL || in->space == PSP_GENERIC ||
                 in->space == PSP_SHARED)) {
                const ptx_opd_t *ad = &in->src[0];
                if (ad->kind == OPD_ADDR && ad->addr_reg[0]) {
                    int l = lin_find(lin, nlin, ad->addr_reg);
                    if (l == -2 && !fn->host_calls) {
                        snprintf(err, errlen, "line %d: ambiguous address",
                                 in->line);
                        goto out;
                    }
                    int is_shared_access = (in->space == PSP_SHARED) ||
                        ((in->op == POP_WMMA_LOAD ||
                          in->op == POP_WMMA_STORE) &&
                         l <= LIN_SHBASE);
                    if (is_shared_access) {
                        // Host-call dynamic pointers may mix spaces;
                        // the state machine dispatches on the tag.
                        if (!fn->host_calls && l >= 0 &&
                            unit->addr_size != 32) {
                            snprintf(err, errlen, "line %d: global address used with shared access",
                                     in->line);
                            goto out;
                        }
                        if (!fn->host_calls && l > LIN_SHBASE &&
                            unit->addr_size != 32) {
                            snprintf(err, errlen, "line %d: shared access needs a .shared-derived address (mov/cvta from the variable)",
                                     in->line);
                            goto out;
                        }
                        if (!fn->host_calls && l <= LIN_GLBSE) {
                            snprintf(err, errlen, "line %d: shared access to a .global address (space mismatch)",
                                     in->line);
                            goto out;
                        }
                    } else if (l >= 0) {
                        as_addr[l] = 1;
                    } else if (l <= LIN_DYNP &&
                               l > LIN_DYNP - (int)unit->nshvars) {
                        // Dynamic pointer reassembled from a shared
                        // pointer table: every parameter stored into
                        // that table is pointer-valued.
                        int sv = LIN_DYNP - l;
                        for (int p = 0; p < nparams && p < 64; p++)
                            if (shptr[sv] & (1ull << (unsigned)p))
                                as_addr[p] = 1;
                    } else if (l <= LIN_GLBSE) {
                        // .global-derived address in global/generic
                        // space: served from the image at emission,
                        // not a kernel param (nothing to mark).
                    } else if (l <= LIN_SHBASE && !fn->host_calls) {
                        // A register whose lineage was inferred as shared is
                        // being used as a global/generic address.  This can
                        // be a false positive when PTX reuses a register name
                        // across scopes (the linear lineage tracker does not
                        // model liveness), which previously made the compiler
                        // reject valid kernels at large sizes.  Treat it as a
                        // value (no address-param marking) instead of
                        // erroring out, and let emission handle it.
                        // (No as_addr marking; lineage stays tracked.)
                    }
                }
            }
            // Scalar uses: every register operand elsewhere.
            // mma fragments live in mregs (not src): all fourteen are
            // scalar value uses (fragment regs are never addresses);
            // the four destinations kill pointer lineage below.
            if (in->op == POP_MMA) {
                for (int m = 0; m < 20; m++) {
                    if (!in->mregs[m][0])
                        continue;
                    int l = lin_find(lin, nlin, in->mregs[m]);
                    if (l == -2) {
                        snprintf(err, errlen,
                                 "line %d: ambiguous pointer arithmetic",
                                 in->line);
                        goto out;
                    }
                    mark_scalar_use(as_val, l);
                }
                if (in->mma_sparse && in->smreg[0]) {
                    int l = lin_find(lin, nlin, in->smreg);
                    if (l == -2) {
                        snprintf(err, errlen,
                                 "line %d: ambiguous pointer arithmetic",
                                 in->line);
                        goto out;
                    }
                    mark_scalar_use(as_val, l);
                }
                for (int m = 0; m < 4; m++)
                    lin_set(lin, &nlin, in->mregs[m], -1);
                continue;
            }
            if (in->op == POP_WMMA_MMA || in->op == POP_WMMA_STORE ||
                in->op == POP_WMMA_LOAD) {
                int first = in->op == POP_WMMA_STORE ? 0 : 1;
                int last = in->op == POP_WMMA_MMA ? 4 : 1;
                for (int r = first; r < last; r++)
                    for (int j = 0; j < 8; j++) {
                        int l = lin_find(lin, nlin, in->wregs[r][j]);
                        if (l == -2) {
                            snprintf(err, errlen,
                                     "line %d: ambiguous pointer arithmetic",
                                     in->line);
                            goto out;
                        }
                        mark_scalar_use(as_val, l);
                    }
                if (in->op != POP_WMMA_STORE)
                    for (int j = 0; j < 8; j++)
                        lin_set(lin, &nlin, in->wregs[0][j], -1);
                if (in->op == POP_WMMA_MMA)
                    continue;
            }
            const ptx_opd_t *host_ops =
                (in->op == POP_HOSTCALL && in->call_args) ? in->call_args
                                                          : in->src;
            for (int s = 0; s < in->nsrc; s++) {
                const ptx_opd_t *op = &host_ops[s];
                if (in->op == POP_LD && in->space == PSP_PARAM)
                    continue; // handled above (param name, not reg use)
                if (in->op == POP_ISSPACEP)
                    continue; // source is an address, not a scalar use
                if ((in->op == POP_CPASYNC ||
                     in->op == POP_CPASYNC_BULK) && s < 2) {
                    // Address operands, not scalar uses.
                    if (op->kind == OPD_ADDR) {
                        if (op->addr_reg[0]) {
                            int l = lin_find(lin, nlin, op->addr_reg);
                            if (l >= 0)
                                as_addr[l] = 1;
                        } else if (op->param[0]) {
                            for (int pi = 0; pi < nparams; pi++)
                                if (strcmp(fn->params[pi].name,
                                           op->param) == 0)
                                    as_addr[pi] = 1;
                        }
                    }
                    continue;
                }
                if ((in->op == POP_LD || in->op == POP_ST ||
                     in->op == POP_ATOM || in->op == POP_ATOMCAS ||
                     in->op == POP_WMMA_LOAD || in->op == POP_WMMA_STORE) &&
                    s == 0 &&
                    (in->op == POP_WMMA_LOAD || in->op == POP_WMMA_STORE ||
                     in->space == PSP_GLOBAL || in->space == PSP_GENERIC ||
                     in->space == PSP_SHARED))
                    continue; // address position, handled above
                if (op->kind != OPD_REG)
                    continue;
                // Predicate regs and .pred dsts are not tracked.
                int l = lin_find(lin, nlin, op->reg);
                if (l == -2) {
                    snprintf(err, errlen,
                             "line %d: ambiguous pointer arithmetic", in->line);
                    goto out;
                }
                // A kernel pointer passed directly (or through
                // mov/add) to a host callback is an address use. The
                // host ABI then substitutes a real host pointer for
                // the 64-bit argument at service time.
                if ((in->op == POP_HOSTCALL ||
                     (fn->host_calls && in->op == POP_SELP &&
                      in->ty == PTY_U64) ||
                     (fn->host_calls && in->op == POP_MADWIDE &&
                      s == 2)) &&
                    l >= 0) {
                    as_addr[l] = 1;
                    continue;
                }
                mark_scalar_use(as_val, l);
            }
            // Any other definition kills pointer lineage: without
            // this, reusing a tracked register for a plain value
            // leaves stale lineage behind (false ambiguities or, worse,
            // misclassified params). ld.param and mov/add set their
            // own lineage above and `continue` past here. Predicated
            // definitions keep old lineage (the old value survives on
            // the false arm; the incoming side was already attributed
            // as a scalar use above) — only mov may be predicated.
            if (in->dst[0] && !in->pred[0]) {
                lin_set(lin, &nlin, in->dst, -1);
                half_set(half, &nhalf, in->dst, 0, -1, -1);
            }
        }
    }
    for (int i = 0; i < nparams; i++) {
        if (fn->params[i].is_retval)
            continue; // excluded from counting (no arg slot)
        // Module-scope symbol buffers injected as hidden BDA pointer params
        // are addressed by symbol name in codegen, not ld.param; treat as
        // pointer bindings so they get descriptor/BDA plumbing and launch args.
        if (fn->params[i].is_hidden_symbol) {
            kinds_out[i] = 1;
            continue;
        }
        // Compiler-synthesized host-call buffers are always pointer
        // bindings, even in segments that never read them.
        if (fn->params[i].is_internal || fn->params[i].is_tex) {
            kinds_out[i] = 1;
            continue;
        }
        // A 16-byte by-value struct argument is either two pointer
        // halves (two bindings from one host argument) or opaque
        // scalar words (four push words from one host argument).
        if (fn->params[i].wide_words) {
            // A struct too large for the push-constant block is opaque
            // metadata passed through a storage buffer, even when some
            // of its words look like addresses.
            if ((uint32_t)fn->params[i].wide_words >
                ACUDA_BYVAL_SSBO_WORDS)
                kinds_out[i] = 4;
            else if (as_addr[i] && as_val[i])
                kinds_out[i] = 6; // mixed blob: word0 ptr, rest scalar
            else
                kinds_out[i] = as_addr[i] ? 5 : (as_val[i] ? 4 : 3);
            continue;
        }
        // A .u64 parameter is pointer-typed in NVIDIA's parameter
        // space: reading its words or passing it through mov/cvta does
        // not make it a scalar. Prefer the pointer classification so
        // address and word-value uses can coexist (the emitter lowers
        // pointer-word reads to defined constants).
        if (as_addr[i])
            kinds_out[i] = 1;
        else if (!as_val[i])
            kinds_out[i] = 3; // unused: no binding, no push words
        else if (fn->params[i].ty == PTY_U64 ||
                 fn->params[i].ty == PTY_F64)
            kinds_out[i] = 2;
        else
            kinds_out[i] = 0;
    }
    rc = 0;
out:
    free(as_addr);
    free(as_val);
    free(lin);
    free(half);
    free(shptr);
    return rc;
}
