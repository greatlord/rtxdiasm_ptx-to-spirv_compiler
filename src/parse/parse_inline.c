// Post-parse call desugaring and direct-call inlining.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ptx.h"
#include "parse/parse_internal.h"

#if defined(_MSC_VER)
#define ACUDA_NOINLINE __declspec(noinline)
#else
#define ACUDA_NOINLINE __attribute__((noinline))
#endif

// ---- Post-parse call inlining ----
//
// Direct `call` (ZLUDA corpus) lowers by splicing the callee body at
// the call site with fresh register names (single SPIR-V function,
// like all our entries). In-function `.param` slots become movs in
// the same pass: `st.param [slot], v` defines the cell,
// `ld.param dst, [slot]` reads it (same-block only, else an honest
// error). Supported: `.param`-slot callees with 0/1 return slots,
// straight-line bodies, calls in straight-line caller blocks.
// Honest errors: undefined/extern/recursive callees, multi-block
// callees, `.local`/slot decls in callees, arg/return count
// mismatches, `.reg`-form results, cross-block slot traffic.

static int inl_fail(char *err, size_t errlen, int line, const char *msg) {
    snprintf(err, errlen, "line %d: %s", line, msg);
    return -1;
}

static int inl_find_slot(const ptx_func_t *fn, const char *name) {
    for (size_t i = 0; i < fn->nslots; i++)
        if (strcmp(fn->slots[i].name, name) == 0)
            return (int)i;
    return -1;
}

static int extfunc_find(const ptx_unit_t *u, const char *name) {
    for (size_t i = 0; i < u->nextfuncs; i++)
        if (strcmp(u->extfuncs[i].name, name) == 0)
            return (int)i;
    return -1;
}

static int inl_find_param(const ptx_func_t *fn, const char *name) {
    for (size_t i = 0; i < fn->nparams; i++)
        if (strcmp(fn->params[i].name, name) == 0)
            return (int)i;
    return -1;
}

static int inl_find_func(const ptx_unit_t *u, const char *name) {
    for (size_t i = 0; i < u->nfuncs; i++)
        if (strcmp(u->funcs[i].name, name) == 0)
            return (int)i;
    return -1;
}

static const ptx_regdecl_t *inl_caller_reg(const ptx_func_t *fn,
                                           const char *name) {
    for (size_t i = 0; i < fn->nregs; i++)
        if (strcmp(fn->regs[i].name, name) == 0)
            return &fn->regs[i];
    return NULL;
}

typedef struct {
    char old[64];
    char fresh[64];
} inl_ren_t;

typedef struct {
    char old[256];
    char fresh[256];
} inl_lmap_t;

typedef struct {
    char slot[256];
    ptx_opd_t val;
} inl_cell_t;

static void inl_lmap_apply(ptx_instr_t *in, const inl_lmap_t *map, int n) {
    for (int s = 0; s < in->nsrc; s++) {
        ptx_opd_t *op = &in->src[s];
        if (op->kind != OPD_ADDR || op->param[0] == '\0')
            continue;
        for (int i = 0; i < n; i++) {
            if (strcmp(op->param, map[i].old) == 0) {
                snprintf(op->param, sizeof(op->param), "%s", map[i].fresh);
                break;
            }
        }
    }
}

static int inl_add_callee_locals(ptx_func_t *root,
                                 const ptx_func_t *callee, int tag,
                                 inl_lmap_t *map, int *nmap,
                                 char *err, size_t errlen, int line) {
    if (callee->nlocals == 0)
        return 0;
    if (callee->nlocals > 64)
        return inl_fail(err, errlen, line, "too many callee locals");
    if (root->nlocals + callee->nlocals > 64)
        return inl_fail(err, errlen, line,
                        "too many caller locals after inlining");
    for (size_t i = 0; i < callee->nlocals; i++) {
        char fresh[256];
        int written = snprintf(fresh, sizeof(fresh), "__c%d_%s", tag,
                               callee->locals[i].name);
        if (written < 0 || (size_t)written >= sizeof(fresh))
            return inl_fail(err, errlen, line,
                            "call local rename too long");
        for (size_t j = 0; j < root->nlocals; j++)
            if (strcmp(root->locals[j].name, fresh) == 0)
                return inl_fail(err, errlen, line,
                                "call local rename collision");
        if (root->nlocals == root->locals_cap) {
            size_t cap = root->locals_cap ? root->locals_cap * 2 : 4;
            ptx_shvar_t *nl = realloc(root->locals, sizeof(*nl) * cap);
            if (!nl)
                return inl_fail(err, errlen, line, "out of memory");
            root->locals = nl;
            root->locals_cap = cap;
        }
        strcpy(map[*nmap].old, callee->locals[i].name);
        strcpy(map[*nmap].fresh, fresh);
        (*nmap)++;
        root->locals[root->nlocals] = callee->locals[i];
        snprintf(root->locals[root->nlocals].name,
                 sizeof(root->locals[root->nlocals].name), "%s", fresh);
        root->nlocals++;
    }
    return 0;
}

static const char *inl_ren_find(inl_ren_t *m, int n, const char *old) {
    for (int i = 0; i < n; i++)
        if (strcmp(m[i].old, old) == 0)
            return m[i].fresh;
    return NULL;
}

// Checked copy (honest "name too long", never truncation: silent
// truncation here would merge two registers).
static int inl_cpy(char *dst, size_t cap, const char *src, char *err,
                   size_t errlen, int line) {
    if (strlen(src) >= cap)
        return inl_fail(err, errlen, line, "name too long");
    strcpy(dst, src);
    return 0;
}

static int inl_ren_copy(char *dst, size_t dstlen, inl_ren_t *m, int n,
                        const char *name, char *err, size_t errlen,
                        int line) {
    const char *f = inl_ren_find(m, n, name);
    if (!f)
        return inl_fail(err, errlen, line, "callee register without rename");
    return inl_cpy(dst, dstlen, f, err, errlen, line);
}

// Copy one callee instruction into the caller: rename every register
// position (dst/dst2/pred/sources/address bases); param/special/
// immediate operands pass through by value.
static int inl_copy_instr(ptx_instr_t *dst, const ptx_instr_t *src,
                          inl_ren_t *m, int n, char *err, size_t errlen) {
    *dst = *src;
    if (dst->dst[0] &&
        inl_ren_copy(dst->dst, sizeof(dst->dst), m, n, src->dst, err,
                     errlen, src->line) != 0)
        return -1;
    if (dst->dst2[0] &&
        inl_ren_copy(dst->dst2, sizeof(dst->dst2), m, n, src->dst2, err,
                     errlen, src->line) != 0)
        return -1;
    if (dst->pred[0] &&
        inl_ren_copy(dst->pred, sizeof(dst->pred), m, n, src->pred, err,
                     errlen, src->line) != 0)
        return -1;
    for (int s = 0; s < dst->nsrc; s++) {
        ptx_opd_t *op = &dst->src[s];
        if (op->kind == OPD_REG || op->kind == OPD_REGOFF) {
            if (inl_ren_copy(op->reg, sizeof(op->reg), m, n,
                             src->src[s].reg, err, errlen, src->line) !=
                0)
                return -1;
        } else if (op->kind == OPD_ADDR && op->addr_reg[0]) {
            if (inl_ren_copy(op->addr_reg, sizeof(op->addr_reg), m, n,
                             src->src[s].addr_reg, err, errlen,
                             src->line) != 0)
                return -1;
        }
    }
    return 0;
}

static int inl_cell_get(inl_cell_t *cells, int n, const char *slot,
                        ptx_opd_t *out) {
    for (int i = 0; i < n; i++)
        if (strcmp(cells[i].slot, slot) == 0) {
            *out = cells[i].val;
            return 0;
        }
    return -1;
}

static void inl_cell_set(inl_cell_t *cells, int *n, const char *slot,
                         const ptx_opd_t *val) {
    for (int i = 0; i < *n; i++)
        if (strcmp(cells[i].slot, slot) == 0) {
            cells[i].val = *val;
            return;
        }
    // Keys are slot/param names (all < 256 by construction).
    strcpy(cells[*n].slot, slot);
    cells[*n].val = *val;
    (*n)++;
}

typedef struct {
    const char **stack; // function names on the expansion chain
    int depth;
    int *counter; // unit-shared fresh-name discriminator
} inl_ctx_t;

// Append one instruction to a growth buffer.
static int inl_push(ptx_instr_t **tmp, size_t *ntmp, size_t *tcap,
                    const ptx_instr_t *in, char *err, size_t errlen,
                    int line) {
    if (*ntmp + 1 > *tcap) {
        size_t ncap = *tcap ? *tcap * 2 : 8;
        ptx_instr_t *ni = malloc(sizeof(*ni) * ncap);
        if (!ni)
            return inl_fail(err, errlen, line, "out of memory");
        memcpy(ni, *tmp, sizeof(*ni) * *ntmp);
        free(*tmp);
        *tmp = ni;
        *tcap = ncap;
    }
    (*tmp)[(*ntmp)++] = *in;
    return 0;
}

// Resolution scope for a call site: the directly enclosing function
// plus the bindings that map its names to outermost (emitted)
// function operands. Top level: cells = block map, empty seed, no
// renames. Nested (inside a callee walk): no cells (callees declare
// no slots), seed = callee .param values, ren = callee renames.
typedef struct {
    ptx_func_t *fn;
    ptx_func_t *root;
    inl_cell_t *cells;
    int *ncells;
    inl_cell_t *seed;
    int nseed;
    inl_ren_t *ren;
    int nren;
} inl_scope_t;

// Resolve a call argument name in scope to an outer-scope value
// operand (slots via cells, .param params via seed, registers via
// rename or directly at top level).
static int inl_arg_val(inl_scope_t *scope, const char *name, ptx_opd_t *out,
                       char *err, size_t errlen, int line) {
    if (inl_find_slot(scope->fn, name) >= 0) {
        if (!scope->cells)
            return inl_fail(err, errlen, line,
                            "call argument slot not stored in block");
        return inl_cell_get(scope->cells, *scope->ncells, name, out) == 0
                   ? 0
                   : inl_fail(err, errlen, line,
                              "call argument slot not stored in block");
    }
    if (scope->seed) {
        for (int i = 0; i < scope->nseed; i++) {
            if (strcmp(scope->seed[i].slot, name) == 0) {
                *out = scope->seed[i].val;
                return 0;
            }
        }
    }
    {
        const ptx_regdecl_t *rd = inl_caller_reg(scope->fn, name);
        if (!rd) {
            if (inl_find_param(scope->fn, name) >= 0)
                return inl_fail(err, errlen, line,
                                "call argument is a parameter (load it first)");
            return inl_fail(err, errlen, line,
                            "call argument is not a slot or register");
        }
        memset(out, 0, sizeof(*out));
        out->kind = OPD_REG;
        if (scope->ren) {
            const char *f = inl_ren_find(scope->ren, scope->nren, name);
            if (!f)
                return inl_fail(err, errlen, line,
                                "call argument without rename");
            return inl_cpy(out->reg, sizeof(out->reg), f, err, errlen,
                           line);
        }
        return inl_cpy(out->reg, sizeof(out->reg), name, err, errlen,
                       line);
    }
}

// Resolve a call result name in scope to a definition target: a slot
// (cell, top level only) or an outer-scope register name.
static int inl_ret_tgt(inl_scope_t *scope, const char *name, int *is_slot,
                       char *tgt, size_t cap, char *err, size_t errlen,
                       int line) {
    if (inl_find_slot(scope->fn, name) >= 0) {
        *is_slot = 1;
        return inl_cpy(tgt, cap, name, err, errlen, line);
    }
    const ptx_regdecl_t *rd = inl_caller_reg(scope->fn, name);
    if (!rd) {
        if (inl_find_param(scope->fn, name) >= 0)
            return inl_fail(err, errlen, line, "call result is a parameter");
        return inl_fail(err, errlen, line,
                        "call result is not a slot or register");
    }
    *is_slot = 0;
    if (scope->ren) {
        const char *f = inl_ren_find(scope->ren, scope->nren, name);
        if (!f)
            return inl_fail(err, errlen, line, "call result without rename");
        return inl_cpy(tgt, cap, f, err, errlen, line);
    }
    return inl_cpy(tgt, cap, name, err, errlen, line);
}

// Declared type of a direct top-level argument (slot or register)
// for signature agreement. Renamed/seeded values were checked at
// their own level; immediates rely on the emission backstop
// (resolve checks operand types honestly).
static int inl_arg_ty(inl_scope_t *scope, const char *name, ptx_ty_t *ty) {
    if (scope->ren || scope->nseed > 0)
        return -1;
    int si = inl_find_slot(scope->fn, name);
    if (si >= 0) {
        *ty = scope->fn->slots[(size_t)si].ty;
        return 0;
    }
    const ptx_regdecl_t *rd = inl_caller_reg(scope->fn, name);
    if (!rd)
        return -1;
    *ty = rd->ty;
    return 0;
}

// Expand one POP_CALL into straight-line caller code appended to
// *out (realloc'd). *ncells/cells is the caller's current block map;
// return slots are defined there for later same-block loads.
static int inl_expand_call(ptx_unit_t *u, inl_scope_t *scope,
                           const ptx_instr_t *call,
                           ptx_instr_t **out, size_t *nout, size_t *ocap,
                           inl_ctx_t *cx, char *err, size_t errlen);

static int inl_walk_block(ptx_unit_t *u, ptx_func_t *scope,
                          ptx_instr_t *ins, size_t nin,
                          ptx_instr_t **out, size_t *nout, size_t *ocap,
                          inl_cell_t *cells, int *ncells, inl_ctx_t *cx,
                          char *err, size_t errlen) {
    // Resolve simple indirect calls: a register written by
    // `mov.u64 %rd, funcname` (or by another such mov) carries a
    // statically known callee. Anything loaded from memory or modified
    // by other arithmetic stays an honest error.
    {
        struct { char reg[64]; char fn[256]; } fmap[64];
        int nf = 0;
        for (size_t i = 0; i < nin; i++) {
            ptx_instr_t *in = &ins[i];
            if (in->op == POP_CALL &&
                inl_find_func(u, in->target) < 0 &&
                extfunc_find(u, in->target) < 0) {
                for (int j = 0; j < nf; j++)
                    if (strcmp(fmap[j].reg, in->target) == 0) {
                        snprintf(in->target, sizeof(in->target), "%s",
                                 fmap[j].fn);
                        break;
                    }
            }
            if (!in->dst[0])
                continue;
            const char *fn = NULL;
            if (in->op == POP_MOV && in->nsrc == 1) {
                if (in->src[0].kind == OPD_PARAM &&
                    (inl_find_func(u, in->src[0].param) >= 0 ||
                     extfunc_find(u, in->src[0].param) >= 0)) {
                    fn = in->src[0].param;
                } else if (in->src[0].kind == OPD_REG) {
                    for (int j = 0; j < nf; j++)
                        if (strcmp(fmap[j].reg, in->src[0].reg) == 0) {
                            fn = fmap[j].fn;
                            break;
                        }
                }
            }
            int slot = -1;
            for (int j = 0; j < nf; j++)
                if (strcmp(fmap[j].reg, in->dst) == 0) {
                    slot = j;
                    break;
                }
            if (fn) {
                if (slot < 0 && nf < 64) {
                    slot = nf++;
                    snprintf(fmap[slot].reg, sizeof(fmap[slot].reg), "%s",
                             in->dst);
                }
                if (slot >= 0)
                    snprintf(fmap[slot].fn, sizeof(fmap[slot].fn), "%s", fn);
            } else if (slot >= 0) {
                fmap[slot] = fmap[--nf];
            }
        }
    }
    for (size_t i = 0; i < nin; i++) {
        ptx_instr_t *in = &ins[i];
        if (in->op == POP_ST && in->space == PSP_PARAM) {
            // Own return-slot store (kernel ABI): dropped, not
            // modeled. Anything else here is a call-slot store:
            // defines the cell, emits nothing. Slot cells are
            // untyped bit-carriers (like mov.b32); agreement with
            // the callee signature is checked at call boundaries.
            int pi = inl_find_param(scope, in->src[0].param);
            if (pi >= 0 && scope->params[(size_t)pi].is_retval)
                continue;
            int si = inl_find_slot(scope, in->src[0].param);
            if (si < 0)
                return inl_fail(err, errlen, in->line,
                                "st.param to unknown slot");
            inl_cell_set(cells, ncells, in->src[0].param, &in->src[1]);
            continue;
        }
        if (in->op == POP_LD && in->space == PSP_PARAM &&
            in->src[0].kind == OPD_ADDR && !in->src[0].addr_reg[0] &&
            inl_find_param(scope, in->src[0].param) < 0) {
            // Slot load (entry-param loads keep their op): the slot
            // must be stored earlier in this block.
            int si = inl_find_slot(scope, in->src[0].param);
            if (si < 0)
                return inl_fail(err, errlen, in->line,
                                "ld.param from unknown slot");
            if (scope->slots[(size_t)si].host_call_id >= 0) {
                ptx_instr_t hr;
                memset(&hr, 0, sizeof(hr));
                hr.op = POP_HOST_RESULT;
                hr.host_call_id =
                    scope->slots[(size_t)si].host_call_id;
                hr.host_arg_off = (int)in->src[0].addr_imm;
                hr.ty = in->ty;
                hr.line = in->line;
                if (inl_cpy(hr.dst, sizeof(hr.dst), in->dst, err, errlen,
                            in->line) != 0)
                    return -1;
                if (inl_push(out, nout, ocap, &hr, err, errlen,
                             in->line) != 0)
                    return -1;
                continue;
            }
            if (in->src[0].addr_imm != 0)
                return inl_fail(err, errlen, in->line,
                                "param slot offset unsupported");
            ptx_opd_t v;
            if (inl_cell_get(cells, *ncells, in->src[0].param, &v) != 0)
                return inl_fail(err, errlen, in->line,
                                "param slot read before write in block");
            ptx_instr_t mv;
            memset(&mv, 0, sizeof(mv));
            mv.op = POP_MOV;
            mv.ty = in->ty;
            mv.line = in->line;
            if (inl_cpy(mv.dst, sizeof(mv.dst), in->dst, err, errlen,
                        in->line) != 0)
                return -1;
            mv.src[0] = v;
            mv.nsrc = 1;
            if (*nout + 1 > *ocap) {
                size_t ncap = *ocap ? *ocap * 2 : 8;
                ptx_instr_t *ni = malloc(sizeof(*ni) * ncap);
                if (!ni)
                    return inl_fail(err, errlen, in->line, "out of memory");
                memcpy(ni, *out, sizeof(*ni) * *nout);
                free(*out);
                *out = ni;
                *ocap = ncap;
            }
            (*out)[(*nout)++] = mv;
            continue;
        }
        if (in->op == POP_CALL &&
            strcmp(in->target, "__assertfail") == 0) {
            // CUDA device assert: report and abort the invocation. We
            // have no host-side assert sink, so terminate (trap) and
            // drop the rest of this block (unreachable by definition).
            ptx_instr_t ex;
            memset(&ex, 0, sizeof(ex));
            ex.op = POP_EXIT;
            ex.line = in->line;
            if (*nout + 1 > *ocap) {
                size_t ncap = *ocap ? *ocap * 2 : 8;
                ptx_instr_t *ni = malloc(sizeof(*ni) * ncap);
                if (!ni)
                    return inl_fail(err, errlen, in->line, "out of memory");
                memcpy(ni, *out, sizeof(*ni) * *nout);
                free(*out);
                *out = ni;
                *ocap = ncap;
            }
            (*out)[(*nout)++] = ex;
            return 0;
        }
        if (in->op == POP_CALL) {
            if (inl_find_func(u, in->target) < 0 &&
                extfunc_find(u, in->target) < 0 &&
                (in->target[0] == '%' ||
                 inl_caller_reg(scope, in->target)))
                return inl_fail(err, errlen, in->line,
                                "indirect call target is not a statically known function");
            int ei = extfunc_find(u, in->target);
            if (ei >= 0) {
                // Declared external function with no body: host call.
                // Resolve slot arguments now; the state-machine emitter
                // turns this into a request plus a continuation PC.
                ptx_extfunc_t *ef = &u->extfuncs[(size_t)ei];
                inl_scope_t sc;
                sc.fn = scope;
                sc.root = scope;
                sc.cells = cells;
                sc.ncells = ncells;
                sc.seed = NULL;
                sc.nseed = 0;
                sc.ren = NULL;
                sc.nren = 0;
                ptx_instr_t hc;
                memset(&hc, 0, sizeof(hc));
                hc.op = POP_HOSTCALL;
                hc.host_call_id = ei;
                hc.line = in->line;
                hc.ty = ef->ret_bytes > 4 ? PTY_U64 : PTY_U32;
                snprintf(hc.pred, sizeof(hc.pred), "%s", in->pred);
                hc.pred_neg = in->pred_neg;
                if (in->nsrc > 32)
                    return inl_fail(err, errlen, in->line,
                                    "too many host call arguments");
                hc.call_args = in->call_args;
                in->call_args = NULL;
                hc.nsrc = in->nsrc;
                for (int a = 0; a < in->nsrc; a++) {
                    if (inl_arg_val(&sc, hc.call_args[a].param,
                                    &hc.call_args[a], err, errlen,
                                    in->line) != 0)
                        return -1;
                }
                if (in->nrets > 1)
                    return inl_fail(err, errlen, in->line,
                                    "host call with multiple results unsupported");
                if (in->nrets == 1) {
                    int is_slot = 0;
                    char tgt[64];
                    if (inl_ret_tgt(&sc, in->rets[0], &is_slot, tgt,
                                    sizeof(tgt), err, errlen,
                                    in->line) != 0)
                        return -1;
                    if (is_slot) {
                        int si = inl_find_slot(scope, tgt);
                        if (si < 0)
                            return inl_fail(err, errlen, in->line,
                                            "host call result slot not found");
                        scope->slots[(size_t)si].host_call_id = ei;
                    } else {
                        ptx_instr_t hr;
                        memset(&hr, 0, sizeof(hr));
                        hr.op = POP_HOST_RESULT;
                        hr.host_call_id = ei;
                        hr.host_arg_off = 0;
                        hr.ty = hc.ty;
                        hr.line = in->line;
                        snprintf(hr.dst, sizeof(hr.dst), "%s", tgt);
                        if (inl_push(out, nout, ocap, &hr, err, errlen,
                                     in->line) != 0)
                            return -1;
                    }
                }
                if (inl_push(out, nout, ocap, &hc, err, errlen,
                             in->line) != 0)
                    return -1;
                continue;
            }
            if (in->nsrc > 8)
                return inl_fail(err, errlen, in->line,
                                "device call with more than 8 arguments unsupported");
            inl_scope_t sc;
            sc.fn = scope;
            sc.root = scope;
            sc.cells = cells;
            sc.ncells = ncells;
            sc.seed = NULL;
            sc.nseed = 0;
            sc.ren = NULL;
            sc.nren = 0;
            if (inl_expand_call(u, &sc, in, out, nout, ocap, cx, err,
                                errlen) != 0)
                return -1;
            continue;
        }
        if (*nout + 1 > *ocap) {
            size_t ncap = *ocap ? *ocap * 2 : 8;
            ptx_instr_t *ni = malloc(sizeof(*ni) * ncap);
            if (!ni)
                return inl_fail(err, errlen, in->line, "out of memory");
            memcpy(ni, *out, sizeof(*ni) * *nout);
            free(*out);
            *out = ni;
            *ocap = ncap;
        }
        (*out)[(*nout)++] = *in;
    }
    return 0;
}

static int inl_expand_call(ptx_unit_t *u, inl_scope_t *scope,
                           const ptx_instr_t *call,
                           ptx_instr_t **out, size_t *nout, size_t *ocap,
                           inl_ctx_t *cx, char *err, size_t errlen) {
    ptx_func_t *root = scope->root;
    for (int d = 0; d < cx->depth; d++) {
        if (strcmp(cx->stack[d], call->target) == 0)
            return inl_fail(err, errlen, call->line,
                            "recursive call unsupported");
    }
    if (cx->depth > 64)
        return inl_fail(err, errlen, call->line,
                        "call depth exceeded (possible recursion)");
    int ci = inl_find_func(u, call->target);
    if (ci < 0) {
        char m[640];
        snprintf(m, sizeof(m), "call to undefined function '%s'",
                 call->target);
        return inl_fail(err, errlen, call->line, m);
    }
    ptx_func_t *callee = &u->funcs[(size_t)ci];
    if (callee->nblocks != 1)
        return inl_fail(err, errlen, call->line,
                        "callee has multiple blocks (only straight-line callees supported)");
    if (callee->nslots > 0)
        return inl_fail(err, errlen, call->line,
                        "callee declares param slots (unsupported in calls)");
    // Positional params (excluding retval slots) <-> call arguments.
    // `.param` slots are cell-wired; `.reg` names are register-aliased.
    // A mixed signature may use `.reg` returns with `.param` arguments.
    int nparams = 0, nretvs = 0;
    for (size_t i = 0; i < callee->nparams; i++) {
        if (callee->params[i].is_retval)
            nretvs++;
        else
            nparams++;
    }
    int nrp = 0, nrr = 0;
    for (size_t i = 0; i < callee->nrparams; i++) {
        if (callee->rparams[i].is_retval)
            nrr++;
        else
            nrp++;
    }
    int argreg = nrp > 0;
    int retregflag = nrr > 0;
    int mixed = nparams > 0 && nretvs == 0 && nrp == 0 && nrr > 0;
    if ((nparams + nretvs > 0) && (nrp + nrr > 0) && !mixed)
        return inl_fail(err, errlen, call->line,
                        "callee mixes .param/.reg signature");
    int regform = retregflag;
    if (!argreg) {
        if (!retregflag && nretvs > 1)
            return inl_fail(err, errlen, call->line,
                            "callee has multiple return slots (only .reg form, unsupported)");
        if (call->nsrc != nparams) {
            char m[640];
            snprintf(m, sizeof(m),
                     "call passes %d args but '%s' takes %d params",
                     call->nsrc, callee->name, nparams);
            return inl_fail(err, errlen, call->line, m);
        }
        int expectedRets = retregflag ? nrr : nretvs;
        if (call->nrets != expectedRets) {
            char m[640];
            snprintf(m, sizeof(m),
                     "call returns %d values but '%s' returns %d",
                     call->nrets, callee->name, expectedRets);
            return inl_fail(err, errlen, call->line, m);
        }
    } else {
        if (call->nsrc != nrp) {
            char m[640];
            snprintf(m, sizeof(m),
                     "call passes %d args but '%s' takes %d params",
                     call->nsrc, callee->name, nrp);
            return inl_fail(err, errlen, call->line, m);
        }
        if (call->nrets != nrr) {
            char m[640];
            snprintf(m, sizeof(m),
                     "call returns %d values but '%s' returns %d",
                     call->nrets, callee->name, nrr);
            return inl_fail(err, errlen, call->line, m);
        }
    }
    // Resolve call arguments to outermost value operands. Direct
    // names carry a signature-agreement check; seeded values were
    // checked at their own level.
    ptx_opd_t argvals[4];
    {
        for (int a = 0; a < call->nsrc; a++) {
            if (inl_arg_val(scope, call->src[a].param, &argvals[a], err,
                            errlen, call->line) != 0)
                return -1;
            ptx_ty_t want = PTY_U32;
            {
                int k = -1;
                if (!argreg) {
                    for (size_t i = 0; i < callee->nparams; i++) {
                        if (callee->params[i].is_retval)
                            continue;
                        if (++k == a)
                            want = callee->params[i].ty;
                    }
                } else {
                    for (size_t i = 0; i < callee->nrparams; i++) {
                        if (callee->rparams[i].is_retval)
                            continue;
                        if (++k == a)
                            want = callee->rparams[i].ty;
                    }
                }
            }
            ptx_ty_t have;
            if (inl_arg_ty(scope, call->src[a].param, &have) == 0 &&
                have != want)
                return inl_fail(err, errlen, call->line,
                                "call argument type mismatch with callee param");
        }
    }
    // Resolve call results to definition targets: slots (cells) or
    // outermost register names.
    int ret_is_slot[4];
    char ret_tgt[4][64];
    for (int r = 0; r < call->nrets; r++) {
        if (inl_ret_tgt(scope, call->rets[r], &ret_is_slot[r], ret_tgt[r],
                        sizeof(ret_tgt[r]), err, errlen, call->line) != 0)
            return -1;
    }
    // Return agreement: register targets must match the callee
    // return type; preds cannot land in slots (no pred cells).
    {
        int k = 0;
        for (size_t i = 0; i < (retregflag ? callee->nrparams : callee->nparams); i++) {
            int isrv = retregflag ? callee->rparams[i].is_retval
                               : callee->params[i].is_retval;
            if (!isrv)
                continue;
            ptx_ty_t wt = retregflag ? callee->rparams[i].ty
                                  : callee->params[i].ty;
            if (ret_is_slot[k]) {
                if (wt == PTY_PRED)
                    return inl_fail(err, errlen, call->line,
                                    "pred result needs a register");
            } else {
                const ptx_regdecl_t *rd = NULL;
                for (size_t q = 0; q < root->nregs; q++) {
                    if (strcmp(root->regs[q].name, ret_tgt[k]) == 0) {
                        rd = &root->regs[q];
                        break;
                    }
                }
                if (!rd)
                    return inl_fail(err, errlen, call->line,
                                    "call result register not declared");
                if (rd->ty != wt)
                    return inl_fail(err, errlen, call->line,
                                    "call result type mismatch");
            }
            k++;
        }
    }
    int tag = (*cx->counter)++;
    inl_lmap_t lmap[64];
    int nlmap = 0;
    if (inl_add_callee_locals(root, callee, tag, lmap, &nlmap,
                              err, errlen, call->line) != 0)
        return -1;
    // Rename every callee register fresh, except pred-typed `.reg`
    // params/returns (no mov exists for preds): those alias the
    // resolved caller names. Fresh decls land in the outermost
    // function.
    inl_ren_t *ren = malloc(sizeof(*ren) *
                            (callee->nregs ? callee->nregs : 1));
    if (!ren)
        return inl_fail(err, errlen, call->line, "out of memory");
    int nren = 0;
    for (size_t i = 0; i < callee->nregs; i++) {
        const char *alias = NULL;
        if (regform) {
            int k = 0;
            for (size_t j = 0; j < callee->nrparams && !alias; j++) {
                if (callee->rparams[j].is_retval ||
                    callee->rparams[j].ty != PTY_PRED)
                    goto nextparam;
                if (strcmp(callee->rparams[j].name,
                           callee->regs[i].name) == 0) {
                    if (argvals[k].kind != OPD_REG) {
                        free(ren);
                        return inl_fail(err, errlen, call->line,
                                        "pred argument needs a register");
                    }
                    alias = argvals[k].reg;
                }
            nextparam:
                if (!callee->rparams[j].is_retval)
                    k++;
            }
            k = 0;
            for (size_t j = 0; j < callee->nrparams && !alias; j++) {
                if (!callee->rparams[j].is_retval ||
                    callee->rparams[j].ty != PTY_PRED)
                    goto nextret;
                if (strcmp(callee->rparams[j].name,
                           callee->regs[i].name) == 0) {
                    // Pred returns alias register targets (slot+pred
                    // errored during verification).
                    alias = ret_tgt[k];
                }
            nextret:
                if (callee->rparams[j].is_retval)
                    k++;
            }
        }
        if (alias) {
            if (inl_cpy(ren[nren].old, sizeof(ren[nren].old),
                        callee->regs[i].name, err, errlen,
                        call->line) != 0) {
                free(ren);
                return -1;
            }
            if (inl_cpy(ren[nren].fresh, sizeof(ren[nren].fresh), alias,
                        err, errlen, call->line) != 0) {
                free(ren);
                return -1;
            }
            nren++;
            continue;
        }
        char pre[32];
        snprintf(pre, sizeof(pre), "__c%d_", tag);
        if (inl_cpy(ren[nren].old, sizeof(ren[nren].old),
                    callee->regs[i].name, err, errlen, call->line) != 0) {
            free(ren);
            return -1;
        }
        if (strlen(pre) + strlen(callee->regs[i].name) >=
            sizeof(ren[nren].fresh)) {
            free(ren);
            return inl_fail(err, errlen, call->line, "call rename too long");
        }
        strcpy(ren[nren].fresh, pre);
        strcat(ren[nren].fresh, callee->regs[i].name);
        // Fresh names must not alias an existing outermost register
        // (else two regs would merge silently).
        for (size_t k = 0; k < root->nregs; k++) {
            if (strcmp(root->regs[k].name, ren[nren].fresh) == 0) {
                free(ren);
                return inl_fail(err, errlen, call->line,
                                "call rename collision");
            }
        }
        if (func_add_reg(root, ren[nren].fresh,
                         callee->regs[i].ty) != 0) {
            free(ren);
            return inl_fail(err, errlen, call->line, "out of memory");
        }
        nren++;
    }
    // Map callee param names (positional, non-retval) to arg values.
    inl_cell_t seed[64];
    int nseed = 0;
    if (nparams > 64)
        return inl_fail(err, errlen, call->line, "too many callee params");
    {
        int ai = 0;
        for (size_t i = 0; i < callee->nparams; i++) {
            if (callee->params[i].is_retval)
                continue;
            if (inl_cpy(seed[nseed].slot, sizeof(seed[nseed].slot),
                        callee->params[i].name, err, errlen,
                        call->line) != 0) {
                free(ren);
                return -1;
            }
            seed[nseed].val = argvals[ai++];
            nseed++;
        }
    }
    // Every callee return must be written in the body (else the
    // caller would read an unwritten value): `.param` returns via
    // st.param, `.reg` returns via dst/dst2.
    {
        ptx_block_t *cb0 = &callee->blocks[0];
        for (size_t i = 0; i < (retregflag ? callee->nrparams : callee->nparams); i++) {
            int isrv = retregflag ? callee->rparams[i].is_retval
                               : callee->params[i].is_retval;
            if (!isrv)
                continue;
            const char *rn = retregflag ? callee->rparams[i].name
                                     : callee->params[i].name;
            int found = 0;
            for (size_t b = 0; b < cb0->count && !found; b++) {
                const ptx_instr_t *in = &cb0->instrs[b];
                if (!retregflag && in->op == POP_ST &&
                    in->space == PSP_PARAM &&
                    in->src[0].kind == OPD_ADDR &&
                    !in->src[0].addr_reg[0] &&
                    strcmp(in->src[0].param, rn) == 0)
                    found = 1;
                if (retregflag &&
                    ((in->dst[0] && strcmp(in->dst, rn) == 0) ||
                     (in->dst2[0] && strcmp(in->dst2, rn) == 0)))
                    found = 1;
                // A nested call naming this return writes it (its own
                // write is verified recursively); bodies here are
                // straight-line, hence unconditional.
                if (in->op == POP_CALL) {
                    for (int r = 0; r < in->nrets && !found; r++) {
                        if (strcmp(in->rets[r], rn) == 0)
                            found = 1;
                    }
                }
            }
            if (!found) {
                free(ren);
                if (retregflag)
                    return inl_fail(err, errlen, call->line,
                                    "callee never writes its return");
                return inl_fail(err, errlen, call->line,
                                "callee never writes its return slot");
            }
        }
    }
    // Walk the callee body: params arrive via seed cells, the return
    // store becomes a mov to the precomputed target, `ret` is dropped
    // (control returns to the caller), nested calls recurse.
    char retreg[64] = "";
    ptx_block_t *cb = &callee->blocks[0];
    ptx_instr_t *tmp = NULL;
    size_t ntmp = 0, tcap = 0;
    // `.reg`-form non-pred params arrive via copy-in movs (preds
    // aliased in ren; `.param` params arrive via seed cells).
    if (argreg) {
        int k = 0;
        for (size_t i = 0; i < callee->nrparams; i++) {
            if (callee->rparams[i].is_retval)
                continue;
            if (callee->rparams[i].ty != PTY_PRED) {
                const char *dst =
                    inl_ren_find(ren, nren, callee->rparams[i].name);
                if (!dst) {
                    free(ren);
                    return inl_fail(err, errlen, call->line,
                                    "callee parameter without rename");
                }
                ptx_instr_t mv;
                memset(&mv, 0, sizeof(mv));
                mv.op = POP_MOV;
                mv.ty = callee->rparams[i].ty;
                mv.line = call->line;
                if (inl_cpy(mv.dst, sizeof(mv.dst), dst, err, errlen,
                            call->line) != 0) {
                    free(ren);
                    return -1;
                }
                mv.src[0] = argvals[k];
                mv.nsrc = 1;
                if (inl_push(&tmp, &ntmp, &tcap, &mv, err, errlen,
                             call->line) != 0) {
                    free(ren);
                    free(tmp);
                    return -1;
                }
            }
            k++;
        }
    }
    {
        // Manual walk (like inl_walk_block, but return stores map to
        // the fresh return register instead of a caller cell).
        for (size_t i = 0; i < cb->count; i++) {
            ptx_instr_t *in = &cb->instrs[i];
            if (in->op == POP_RET)
                continue; // return == continue after the call
            if (in->op == POP_BRA || in->op == POP_CBRA) {
                free(ren);
                free(tmp);
                return inl_fail(err, errlen, in->line,
                                "callee branch unsupported");
            }
            if (in->op == POP_ST && in->space == PSP_PARAM) {
                int pi = inl_find_param(callee, in->src[0].param);
                if (pi < 0 || !callee->params[(size_t)pi].is_retval) {
                    free(ren);
                    free(tmp);
                    return inl_fail(err, errlen, in->line,
                                    "callee stores unknown param");
                }
                if (callee->params[(size_t)pi].ty != in->ty) {
                    free(ren);
                    free(tmp);
                    return inl_fail(err, errlen, in->line,
                                    "callee return type mismatch");
                }
                // Return target: fresh cell register for slots (last
                // write wins), the resolved register directly.
                const char *dstname;
                if (ret_is_slot[0]) {
                    if (!retreg[0]) {
                        snprintf(retreg, sizeof(retreg), "__retr%d_", tag);
                        if (func_add_reg(root, retreg, in->ty) != 0) {
                            free(ren);
                            free(tmp);
                            return inl_fail(err, errlen, call->line,
                                            "out of memory");
                        }
                    }
                    dstname = retreg;
                } else {
                    dstname = ret_tgt[0];
                }
                ptx_instr_t mv;
                memset(&mv, 0, sizeof(mv));
                mv.op = POP_MOV;
                mv.ty = in->ty;
                mv.line = in->line;
                if (inl_cpy(mv.dst, sizeof(mv.dst), dstname, err, errlen,
                            in->line) != 0) {
                    free(ren);
                    free(tmp);
                    return -1;
                }
                mv.src[0] = in->src[1];
                // Rename a register value from callee scope.
                if (mv.src[0].kind == OPD_REG ||
                    mv.src[0].kind == OPD_REGOFF) {
                    if (inl_ren_copy(mv.src[0].reg, sizeof(mv.src[0].reg),
                                     ren, nren, in->src[1].reg, err, errlen,
                                     in->line) != 0) {
                        free(ren);
                        free(tmp);
                        return -1;
                    }
                } else if (mv.src[0].kind == OPD_ADDR &&
                           mv.src[0].addr_reg[0]) {
                    free(ren);
                    free(tmp);
                    return inl_fail(err, errlen, in->line,
                                    "callee returns an address");
                }
                if (ntmp + 1 > tcap) {
                    size_t ncap = tcap ? tcap * 2 : 8;
                    ptx_instr_t *ni = malloc(sizeof(*ni) * ncap);
                    if (!ni) {
                        free(ren);
                        free(tmp);
                        return inl_fail(err, errlen, in->line,
                                        "out of memory");
                    }
                    memcpy(ni, tmp, sizeof(*ni) * ntmp);
                    free(tmp);
                    tmp = ni;
                    tcap = ncap;
                }
                tmp[ntmp++] = mv;
                continue;
            }
            if (in->op == POP_LD && in->space == PSP_PARAM &&
                in->src[0].kind == OPD_ADDR && !in->src[0].addr_reg[0]) {
                int pi = inl_find_param(callee, in->src[0].param);
                if (pi < 0)
                    return inl_fail(err, errlen, in->line,
                                    "callee reads unknown param");
                if (callee->params[(size_t)pi].is_retval)
                    return inl_fail(err, errlen, in->line,
                                    "callee reads its return slot");
                ptx_opd_t v;
                if (inl_cell_get(seed, nseed, in->src[0].param, &v) != 0) {
                    free(ren);
                    free(tmp);
                    return inl_fail(err, errlen, in->line,
                                    "callee param without argument");
                }
                ptx_instr_t mv;
                memset(&mv, 0, sizeof(mv));
                mv.op = POP_MOV;
                mv.ty = in->ty;
                mv.line = in->line;
                if (inl_ren_copy(mv.dst, sizeof(mv.dst), ren, nren,
                                 in->dst, err, errlen, in->line) != 0) {
                    free(ren);
                    free(tmp);
                    return -1;
                }
                mv.src[0] = v;
                mv.nsrc = 1;
                if (ntmp + 1 > tcap) {
                    size_t ncap = tcap ? tcap * 2 : 8;
                    ptx_instr_t *ni = malloc(sizeof(*ni) * ncap);
                    if (!ni) {
                        free(ren);
                        free(tmp);
                        return inl_fail(err, errlen, in->line,
                                        "out of memory");
                    }
                    memcpy(ni, tmp, sizeof(*ni) * ntmp);
                    free(tmp);
                    tmp = ni;
                    tcap = ncap;
                }
                tmp[ntmp++] = mv;
                continue;
            }
            if (in->op == POP_CALL) {
                // Nested call: resolve in callee scope (slots none,
                // params via seed, regs via rename) into the same
                // outermost buffer.
                inl_scope_t sub;
                sub.fn = callee;
                sub.root = root;
                sub.cells = NULL;
                sub.ncells = NULL;
                sub.seed = seed;
                sub.nseed = nseed;
                sub.ren = ren;
                sub.nren = nren;
                const char **stack2 = malloc(sizeof(*stack2) *
                                            (size_t)(cx->depth + 1));
                if (!stack2) {
                    free(ren);
                    free(tmp);
                    return inl_fail(err, errlen, in->line,
                                    "out of memory");
                }
                memcpy(stack2, cx->stack,
                       sizeof(*stack2) * (size_t)cx->depth);
                stack2[cx->depth] = callee->name;
                inl_ctx_t subcx;
                subcx.stack = stack2;
                subcx.depth = cx->depth + 1;
                subcx.counter = cx->counter;
                int rc = inl_expand_call(u, &sub, in, &tmp, &ntmp, &tcap,
                                         &subcx, err, errlen);
                free(stack2);
                if (rc != 0) {
                    free(ren);
                    free(tmp);
                    return -1;
                }
                continue;
            }
            {
                ptx_instr_t cp;
                // OPD_PARAM naming a callee param would leak scope.
                for (int s = 0; s < in->nsrc; s++) {
                    if (in->src[s].kind == OPD_PARAM &&
                        inl_find_param(callee, in->src[s].param) >= 0) {
                        free(ren);
                        free(tmp);
                        return inl_fail(err, errlen, in->line,
                                        "callee moves a param address");
                    }
                }
                if (inl_copy_instr(&cp, in, ren, nren, err, errlen) !=
                    0) {
                    free(ren);
                    free(tmp);
                    return -1;
                }
                inl_lmap_apply(&cp, lmap, nlmap);
                if (ntmp + 1 > tcap) {
                    size_t ncap = tcap ? tcap * 2 : 8;
                    ptx_instr_t *ni = malloc(sizeof(*ni) * ncap);
                    if (!ni) {
                        free(ren);
                        free(tmp);
                        return inl_fail(err, errlen, in->line,
                                        "out of memory");
                    }
                    memcpy(ni, tmp, sizeof(*ni) * ntmp);
                    free(tmp);
                    tmp = ni;
                    tcap = ncap;
                }
                tmp[ntmp++] = cp;
            }
        }
    }
    // Wire returns per pair: slot targets define cells (top level
    // only — callees declare no slots); register targets take a post
    // mov unless already aliased (preds, which have no mov).
    {
        int k = 0;
        for (size_t i = 0; i < (retregflag ? callee->nrparams : callee->nparams); i++) {
            int isrv = retregflag ? callee->rparams[i].is_retval
                               : callee->params[i].is_retval;
            if (!isrv)
                continue;
            const char *rn = retregflag ? callee->rparams[i].name
                                     : callee->params[i].name;
            if (ret_is_slot[k]) {
                // Param `.param` store already targeted retreg; a
                // `.reg` return sits in its fresh register: move it.
                if (retregflag) {
                    const char *fr = inl_ren_find(ren, nren, rn);
                    if (!fr) {
                        free(ren);
                        free(tmp);
                        return inl_fail(err, errlen, call->line,
                                        "callee return without rename");
                    }
                    ptx_instr_t mv;
                    memset(&mv, 0, sizeof(mv));
                    mv.op = POP_MOV;
                    mv.ty = callee->rparams[i].ty;
                    mv.line = call->line;
                    if (inl_cpy(mv.dst, sizeof(mv.dst), retreg, err,
                                errlen, call->line) != 0) {
                        free(ren);
                        free(tmp);
                        return -1;
                    }
                    memset(&mv.src[0], 0, sizeof(mv.src[0]));
                    mv.src[0].kind = OPD_REG;
                    if (inl_cpy(mv.src[0].reg, sizeof(mv.src[0].reg), fr,
                                err, errlen, call->line) != 0) {
                        free(ren);
                        free(tmp);
                        return -1;
                    }
                    mv.nsrc = 1;
                    if (inl_push(&tmp, &ntmp, &tcap, &mv, err, errlen,
                                 call->line) != 0) {
                        free(ren);
                        free(tmp);
                        return -1;
                    }
                }
                if (!scope->cells) {
                    free(ren);
                    free(tmp);
                    return inl_fail(err, errlen, call->line,
                                    "call result slot not visible here");
                }
                ptx_opd_t v;
                memset(&v, 0, sizeof(v));
                v.kind = OPD_REG;
                if (inl_cpy(v.reg, sizeof(v.reg), retreg, err, errlen,
                            call->line) != 0) {
                    free(ren);
                    free(tmp);
                    return -1;
                }
                inl_cell_set(scope->cells, scope->ncells, call->rets[k],
                             &v);
            } else {
                // Register target: `.param` stores already targeted it
                // directly; `.reg` returns need a post mov unless the
                // pred alias already denotes it.
                const char *val = NULL;
                if (!retregflag) {
                    val = NULL; // targeted directly, nothing to do
                } else {
                    val = inl_ren_find(ren, nren, rn);
                    if (!val) {
                        free(ren);
                        free(tmp);
                        return inl_fail(err, errlen, call->line,
                                        "callee return without rename");
                    }
                }
                if (val && strcmp(val, ret_tgt[k]) != 0) {
                    ptx_instr_t mv;
                    memset(&mv, 0, sizeof(mv));
                    mv.op = POP_MOV;
                    mv.ty = retregflag ? callee->rparams[i].ty
                                    : callee->params[i].ty;
                    mv.line = call->line;
                    if (inl_cpy(mv.dst, sizeof(mv.dst), ret_tgt[k], err,
                                errlen, call->line) != 0) {
                        free(ren);
                        free(tmp);
                        return -1;
                    }
                    memset(&mv.src[0], 0, sizeof(mv.src[0]));
                    mv.src[0].kind = OPD_REG;
                    if (inl_cpy(mv.src[0].reg, sizeof(mv.src[0].reg), val,
                                err, errlen, call->line) != 0) {
                        free(ren);
                        free(tmp);
                        return -1;
                    }
                    mv.nsrc = 1;
                    if (inl_push(&tmp, &ntmp, &tcap, &mv, err, errlen,
                                 call->line) != 0) {
                        free(ren);
                        free(tmp);
                        return -1;
                    }
                }
            }
            k++;
        }
    }
    free(ren);
    // Splice the expansion; define return cells for later loads.
    for (size_t i = 0; i < ntmp; i++) {
        if (*nout + 1 > *ocap) {
            size_t ncap = *ocap ? *ocap * 2 : 8;
            ptx_instr_t *ni = malloc(sizeof(*ni) * ncap);
            if (!ni) {
                free(tmp);
                return inl_fail(err, errlen, call->line, "out of memory");
            }
            memcpy(ni, *out, sizeof(*ni) * *nout);
            free(*out);
            *out = ni;
            *ocap = ncap;
        }
        (*out)[(*nout)++] = tmp[i];
    }
    free(tmp);
    return 0;
}

static int inl_fn_has_call(const ptx_func_t *fn) {
    for (size_t b = 0; b < fn->nblocks; b++) {
        for (size_t i = 0; i < fn->blocks[b].count; i++) {
            ptx_op_t op = fn->blocks[b].instrs[i].op;
            if (op == POP_CALL)
                return 1;
            if ((op == POP_ST || op == POP_LD) &&
                fn->blocks[b].instrs[i].space == PSP_PARAM) {
                ptx_opd_t *ad = &fn->blocks[b].instrs[i].src[0];
                if (ad->kind == OPD_ADDR && !ad->addr_reg[0] &&
                    inl_find_param(fn, ad->param) < 0)
                    return 1; // slot traffic needs the pass too
            }
        }
    }
    return 0;
}

// Recover pointer provenance through an unpack/repack round trip:
//   mov.b64 {lo,hi}, src ; ... ; mov.b64 dst, {lo,hi}
// is lowered at parse time to cvt/shr/cvt, which loses the pointer
// classification of src. Rewriting the pack as a plain 64-bit copy of
// src restores it (and is always bit-equivalent). Same-block only.
typedef struct {
    char reg[64];
    char src[64];
    int half;   // 0 low, 1 high
    size_t up;  // index of the unpack sequence that produced it
} inl_half_t;

static void inl_half_set(inl_half_t *hs, int *nh, const char *reg,
                         const char *src, int half, size_t up) {
    for (int i = 0; i < *nh; i++)
        if (strcmp(hs[i].reg, reg) == 0) {
            snprintf(hs[i].src, sizeof(hs[i].src), "%s", src);
            hs[i].half = half;
            hs[i].up = up;
            return;
        }
    if (*nh >= 64)
        return;
    snprintf(hs[*nh].reg, sizeof(hs[*nh].reg), "%s", reg);
    snprintf(hs[*nh].src, sizeof(hs[*nh].src), "%s", src);
    hs[*nh].half = half;
    hs[*nh].up = up;
    (*nh)++;
}

static void inl_half_drop(inl_half_t *hs, int *nh, const char *name,
                          int match_src) {
    for (int i = 0; i < *nh;) {
        const char *v = match_src ? hs[i].src : hs[i].reg;
        if (strcmp(v, name) == 0) {
            hs[i] = hs[--(*nh)];
            continue;
        }
        i++;
    }
}

static int inl_unpack_seq(const ptx_block_t *blk, size_t i, char *lo,
                          char *hi, char *src) {
    if (i + 2 >= blk->count)
        return 0;
    const ptx_instr_t *a = &blk->instrs[i];
    const ptx_instr_t *b = &blk->instrs[i + 1];
    const ptx_instr_t *d = &blk->instrs[i + 2];
    if (a->op != POP_CVT || a->ty != PTY_U32 || a->ty2 != PTY_U64 ||
        a->nsrc != 1 || a->src[0].kind != OPD_REG || !a->dst[0])
        return 0;
    if (b->op != POP_SHR || b->ty != PTY_U64 || b->nsrc != 2 ||
        b->src[0].kind != OPD_REG || b->src[1].kind != OPD_IMM ||
        b->src[1].imm != 32 || !b->dst[0])
        return 0;
    if (d->op != POP_CVT || d->ty != PTY_U32 || d->ty2 != PTY_U64 ||
        d->nsrc != 1 || d->src[0].kind != OPD_REG || !d->dst[0])
        return 0;
    if (strcmp(a->src[0].reg, b->src[0].reg) != 0 ||
        strcmp(d->src[0].reg, b->dst) != 0)
        return 0;
    snprintf(lo, 64, "%s", a->dst);
    snprintf(hi, 64, "%s", d->dst);
    snprintf(src, 64, "%s", a->src[0].reg);
    return 1;
}

static int inl_reg_uses(const ptx_func_t *fn, const char *name) {
    int uses = 0;
    for (size_t b = 0; b < fn->nblocks; b++) {
        const ptx_block_t *blk = &fn->blocks[b];
        for (size_t i = 0; i < blk->count; i++) {
            const ptx_instr_t *in = &blk->instrs[i];
            if (in->pred[0] && strcmp(in->pred, name) == 0)
                uses++;
            for (int s = 0; s < in->nsrc; s++)
                if (in->src[s].kind == OPD_REG &&
                    strcmp(in->src[s].reg, name) == 0)
                    uses++;
            if (in->call_args)
                for (int s = 0; s < in->nsrc; s++)
                    if (in->call_args[s].kind == OPD_REG &&
                        strcmp(in->call_args[s].reg, name) == 0)
                        uses++;
        }
    }
    return uses;
}

static void inl_fix_packed_pointers(ptx_func_t *fn) {
    for (size_t bi = 0; bi < fn->nblocks; bi++) {
        ptx_block_t *blk = &fn->blocks[bi];
        inl_half_t hs[64];
        int nh = 0;
        for (size_t i = 0; i < blk->count; i++) {
            char lo[64], hi[64], src[64];
            if (inl_unpack_seq(blk, i, lo, hi, src)) {
                inl_half_set(hs, &nh, lo, src, 0, i);
                inl_half_set(hs, &nh, hi, src, 1, i);
                i += 2;
                continue;
            }
            ptx_instr_t *in = &blk->instrs[i];
            if (in->op == POP_MOV && in->nsrc == 2 &&
                in->src[0].kind == OPD_REG &&
                in->src[1].kind == OPD_REG) {
                const char *l = in->src[0].reg;
                const char *h = in->src[1].reg;
                const char *srcname = NULL;
                const char *loreg = NULL;
                const char *hireg = NULL;
                size_t up = (size_t)-1;
                int ok = 0;
                for (int a = 0; a < nh && !ok; a++) {
                    if (hs[a].half != 0 || strcmp(hs[a].reg, l) != 0)
                        continue;
                    for (int b = 0; b < nh; b++) {
                        if (hs[b].half != 1 || strcmp(hs[b].reg, h) != 0)
                            continue;
                        if (strcmp(hs[a].src, hs[b].src) == 0) {
                            srcname = hs[a].src;
                            loreg = hs[a].reg;
                            hireg = hs[b].reg;
                            up = hs[a].up;
                            ok = 1;
                            break;
                        }
                    }
                }
                if (ok) {
                    // A pure round trip: the two halves and the shift
                    // temp are used only by this pack and the hi cvt.
                    // Drop the unpack so the pointer source is never
                    // used as a scalar.
                    if (up != (size_t)-1 &&
                        inl_reg_uses(fn, loreg) == 1 &&
                        inl_reg_uses(fn, hireg) == 1 &&
                        inl_reg_uses(fn, blk->instrs[up + 1].dst) == 1) {
                        blk->instrs[up].dead = 1;
                        blk->instrs[up + 1].dead = 1;
                        blk->instrs[up + 2].dead = 1;
                    }
                    in->nsrc = 1;
                    memset(&in->src[1], 0, sizeof(in->src[1]));
                    in->src[0].kind = OPD_REG;
                    snprintf(in->src[0].reg, sizeof(in->src[0].reg), "%s",
                             srcname);
                }
            }
            if (in->dst[0])
                inl_half_drop(hs, &nh, in->dst, 0);
            if (in->dst2[0])
                inl_half_drop(hs, &nh, in->dst2, 0);
            if (in->dst[0]) {
                inl_half_drop(hs, &nh, in->dst, 1);
                inl_half_drop(hs, &nh, in->dst2[0] ? in->dst2 : "",
                              1);
            }
        }
        size_t w = 0;
        for (size_t i = 0; i < blk->count; i++)
            if (!blk->instrs[i].dead)
                blk->instrs[w++] = blk->instrs[i];
        blk->count = w;
    }
}

// brx.idx lowering: replace the indirect branch with one comparison +
// conditional-branch block per target. The check blocks are inserted
// immediately after the brx block so the false path falls through the
// chain to the original next block (the predicated-false path, or the
// undefined out-of-range path for unpredicated brx).
static int inl_block_label_index(const ptx_func_t *fn, const char *label) {
    for (size_t i = 0; i < fn->nblocks; i++)
        if (fn->blocks[i].label[0] &&
            strcmp(fn->blocks[i].label, label) == 0)
            return (int)i;
    return -1;
}

static int inl_lower_brx(ptx_unit_t *u, ptx_func_t *fn, char *err,
                         size_t errlen) {
    (void)u;
    static int seq;
    parser_t tp;
    memset(&tp, 0, sizeof(tp));
    tp.err = err;
    tp.errlen = errlen;
    for (size_t bi = 0; bi < fn->nblocks; bi++) {
        ptx_block_t *blk = &fn->blocks[bi];
        if (blk->count == 0)
            continue;
        ptx_instr_t brx = blk->instrs[blk->count - 1];
        if (brx.op != POP_BRX)
            continue;
        int li = -1;
        for (int l = 0; l < fn->nbranchtargets; l++)
            if (strcmp(fn->branchtargets[l].label, brx.target_list) == 0) {
                li = l;
                break;
            }
        if (li < 0)
            return inl_fail(err, errlen, brx.line,
                            "brx.idx target list not found");
        int n = fn->branchtargets[li].n;
        if (n <= 0)
            return inl_fail(err, errlen, brx.line,
                            "empty .branchtargets list");
        if (n > 64)
            return inl_fail(err, errlen, brx.line,
                            "too many .branchtargets entries");
        int tgt[64];
        // A predicated brx gets a guard block in front of the check
        // chain (the false path continues after the instruction).
        int guard = brx.pred[0] && (bi + 1 < fn->nblocks);
        for (int k = 0; k < n; k++) {
            tgt[k] = inl_block_label_index(
                fn, fn->branchtargets[li].targets[k]);
            if (tgt[k] < 0)
                return inl_fail(err, errlen, brx.line,
                                "brx.idx target label not found");
            int back = guard ? (tgt[k] <= (int)bi)
                             : (tgt[k] < (int)bi);
            if (back)
                return inl_fail(err, errlen, brx.line,
                                "backward brx.idx targets unsupported");
        }
        // Drop the indirect branch; it is the block terminator.
        blk->count--;

        // The check chain ends in a dedicated fall-through block that
        // branches to the function's final ret/exit. Without it the
        // last check's false edge would land on an outer selection's
        // merge block (invalid structured control flow).
        size_t oldn = fn->nblocks;
        size_t add = (size_t)(guard ? n : n - 1) + 1;
        ptx_block_t *nb = realloc(fn->blocks,
                                  sizeof(*nb) * (oldn + add));
        if (!nb)
            return inl_fail(err, errlen, brx.line, "out of memory");
        fn->blocks = nb;
        memmove(&fn->blocks[bi + 1 + add], &fn->blocks[bi + 1],
                sizeof(*nb) * (oldn - bi - 1));
        for (size_t k = 1; k <= add; k++)
            memset(&fn->blocks[bi + k], 0, sizeof(*nb));
        fn->nblocks = oldn + add;
        fn->blocks_cap = oldn + add;

        size_t cb0 = bi + (guard ? 1 : 0);
        size_t after = bi + add;
        blk = &fn->blocks[bi];

        if (guard) {
            // @!p bra <continuation> where continuation is the block
            // that followed the brx (shifted past the inserted chain).
            size_t ci = bi + 1 + add;
            if (fn->blocks[ci].label[0] == '\0') {
                snprintf(fn->blocks[ci].label,
                         sizeof(fn->blocks[ci].label), "__brx_c_%d", seq);
            }
            ptx_instr_t g;
            memset(&g, 0, sizeof(g));
            g.op = POP_CBRA;
            g.line = brx.line;
            snprintf(g.pred, sizeof(g.pred), "%s", brx.pred);
            g.pred_neg = !brx.pred_neg;
            snprintf(g.target, sizeof(g.target), "%s",
                     fn->blocks[ci].label);
            if (block_add_instr(fn, bi, &g, &tp, brx.line) != 0)
                return -1;
        }

        // A register selector is compared directly; an immediate is
        // materialized once in the first check block.
        char idxreg[64];
        if (brx.src[0].kind == OPD_REG) {
            snprintf(idxreg, sizeof(idxreg), "%s", brx.src[0].reg);
        } else {
            snprintf(idxreg, sizeof(idxreg), "__brx_i_%d", seq);
            if (func_add_reg(fn, idxreg, PTY_U32) != 0)
                return inl_fail(err, errlen, brx.line, "out of memory");
            ptx_instr_t mv;
            memset(&mv, 0, sizeof(mv));
            mv.op = POP_MOV;
            mv.ty = PTY_U32;
            mv.line = brx.line;
            snprintf(mv.dst, sizeof(mv.dst), "%s", idxreg);
            mv.src[0] = brx.src[0];
            mv.nsrc = 1;
            if (block_add_instr(fn, cb0, &mv, &tp, brx.line) != 0)
                return -1;
        }
        for (int k = 0; k < n; k++) {
            size_t cb = cb0 + (size_t)k;
            char preg[64];
            snprintf(preg, sizeof(preg), "__brx_p_%d_%d", seq, k);
            if (func_add_reg(fn, preg, PTY_PRED) != 0)
                return inl_fail(err, errlen, brx.line, "out of memory");
            ptx_instr_t sp;
            memset(&sp, 0, sizeof(sp));
            sp.op = POP_SETP;
            sp.ty = PTY_U32;
            sp.cmp = PCM_EQ;
            sp.line = brx.line;
            snprintf(sp.dst, sizeof(sp.dst), "%s", preg);
            sp.src[0].kind = OPD_REG;
            snprintf(sp.src[0].reg, sizeof(sp.src[0].reg), "%s", idxreg);
            sp.src[1].kind = OPD_IMM;
            sp.src[1].imm = (uint64_t)k;
            sp.nsrc = 2;
            if (brx.pred[0] && !guard) {
                // No continuation block to guard: fold the predicate
                // into each check.
                sp.bop = 1;
                sp.bneg = brx.pred_neg;
                sp.src[2].kind = OPD_REG;
                snprintf(sp.src[2].reg, sizeof(sp.src[2].reg), "%s",
                         brx.pred);
                sp.nsrc = 3;
            }
            if (block_add_instr(fn, cb, &sp, &tp, brx.line) != 0)
                return -1;
            ptx_instr_t br;
            memset(&br, 0, sizeof(br));
            br.op = POP_CBRA;
            br.line = brx.line;
            snprintf(br.pred, sizeof(br.pred), "%s", preg);
            snprintf(br.target, sizeof(br.target), "%s",
                     fn->branchtargets[li].targets[k]);
            if (block_add_instr(fn, cb, &br, &tp, brx.line) != 0)
                return -1;
        }

        // Fall-through block: branch to the function's final block.
        size_t fin = fn->nblocks - 1;
        int need_exit = fin <= after;
        if (!need_exit) {
            int term = fn->blocks[fin].count
                           ? (int)fn->blocks[fin]
                                 .instrs[fn->blocks[fin].count - 1]
                                 .op
                           : -1;
            if (term != POP_RET && term != POP_EXIT)
                need_exit = 1;
        }
        if (need_exit) {
            size_t xb = new_block(fn, &tp, NULL);
            if (xb == (size_t)-1 || tp.failed)
                return -1;
            snprintf(fn->blocks[xb].label,
                     sizeof(fn->blocks[xb].label), "__brx_f_%d", seq);
            ptx_instr_t ex;
            memset(&ex, 0, sizeof(ex));
            ex.op = POP_EXIT;
            ex.line = brx.line;
            if (block_add_instr(fn, xb, &ex, &tp, brx.line) != 0)
                return -1;
            fin = xb;
        } else if (fn->blocks[fin].label[0] == '\0') {
            snprintf(fn->blocks[fin].label,
                     sizeof(fn->blocks[fin].label), "__brx_f_%d", seq);
        }
        ptx_instr_t ab;
        memset(&ab, 0, sizeof(ab));
        ab.op = POP_BRA;
        ab.line = brx.line;
        snprintf(ab.target, sizeof(ab.target), "%s",
                 fn->blocks[fin].label);
        if (block_add_instr(fn, after, &ab, &tp, brx.line) != 0)
            return -1;

        bi += add;
        seq++;
    }
    return 0;
}

// Cross-block pass for indirect calls. A register whose value is a
// `mov %rd, funcname` (possibly through further movs/selp) carries a
// statically known candidate set. One candidate becomes a direct call;
// several candidates become a runtime comparison chain and one call
// block per candidate. A `.calltargets` list provides the authoritative
// set (needed for function pointers loaded from memory).
typedef struct {
    char reg[64];
    char funcs[8][256];
    int n;
    int poisoned; // redefined by something that is not a known function
    int overflow;
} inl_fset_t;

static inl_fset_t *inl_fset_lookup(inl_fset_t *m, int n, const char *reg) {
    for (int i = 0; i < n; i++)
        if (strcmp(m[i].reg, reg) == 0)
            return &m[i];
    return NULL;
}

static inl_fset_t *inl_fset_ensure(inl_fset_t *m, int *n, const char *reg) {
    inl_fset_t *e = inl_fset_lookup(m, *n, reg);
    if (e)
        return e;
    if (*n >= 64)
        return NULL;
    e = &m[(*n)++];
    memset(e, 0, sizeof(*e));
    snprintf(e->reg, sizeof(e->reg), "%s", reg);
    return e;
}

static void inl_fset_add(inl_fset_t *e, const char *fn) {
    for (int i = 0; i < e->n; i++)
        if (strcmp(e->funcs[i], fn) == 0)
            return;
    if (e->n >= 8) {
        e->overflow = 1;
        return;
    }
    snprintf(e->funcs[e->n++], sizeof(e->funcs[0]), "%s", fn);
}

// Deep-copy one instruction (call argument vectors are owned by each
// copy) and optionally rewrite the callee name.
static int inl_clone_instr(ptx_instr_t *dst, const ptx_instr_t *src,
                           const char *target, char *err, size_t errlen,
                           int line) {
    *dst = *src;
    dst->call_args = NULL;
    if (src->call_args && src->nsrc > 0) {
        size_t bytes = sizeof(*dst->call_args) * (size_t)src->nsrc;
        dst->call_args = malloc(bytes);
        if (!dst->call_args)
            return inl_fail(err, errlen, line, "out of memory");
        memcpy(dst->call_args, src->call_args, bytes);
    }
    if (target)
        snprintf(dst->target, sizeof(dst->target), "%s", target);
    dst->ncall_cand = 0;
    dst->target_list[0] = '\0';
    return 0;
}

static int inl_make_mov_func(ptx_instr_t *in, const char *dst,
                             const char *func, int line) {
    memset(in, 0, sizeof(*in));
    in->op = POP_MOV;
    in->ty = PTY_U64;
    in->line = line;
    snprintf(in->dst, sizeof(in->dst), "%s", dst);
    in->src[0].kind = OPD_PARAM;
    snprintf(in->src[0].param, sizeof(in->src[0].param), "%s", func);
    in->nsrc = 1;
    return 0;
}

// Replace a marked multi-target call with:
//   checks: setp.eq + @p bra call_i
//   calls:  call func_i; <tail copy>; bra ret
//   ret:    ret/exit
// The call must be in the final block with a straight-line tail.
static int inl_rewrite_dispatch(ptx_unit_t *u, ptx_func_t *fn, char *err,
                                size_t errlen) {
    (void)u;
    static int seq;
    parser_t tp;
    memset(&tp, 0, sizeof(tp));
    tp.err = err;
    tp.errlen = errlen;

    for (size_t bi = 0; bi < fn->nblocks; bi++) {
        ptx_block_t *blk = &fn->blocks[bi];
        size_t ii = 0;
        int found = 0;
        for (; ii < blk->count; ii++) {
            if (blk->instrs[ii].op == POP_CALL &&
                blk->instrs[ii].ncall_cand >= 2) {
                found = 1;
                break;
            }
        }
        if (!found)
            continue;
        ptx_instr_t call = blk->instrs[ii];
        int n = call.ncall_cand;
        if (n > 8)
            return inl_fail(err, errlen, call.line,
                            "too many indirect call targets");
        if (bi != fn->nblocks - 1)
            return inl_fail(err, errlen, call.line,
                            "multi-target indirect call must be in the final block");
        if (blk->count == 0)
            return inl_fail(err, errlen, call.line,
                            "multi-target indirect call without tail");
        int term = blk->instrs[blk->count - 1].op;
        if (term != POP_RET && term != POP_EXIT)
            return inl_fail(err, errlen, call.line,
                            "multi-target indirect call tail must end in ret/exit");
        for (size_t k = ii + 1; k + 1 < blk->count; k++) {
            int op = blk->instrs[k].op;
            if (op == POP_BRA || op == POP_CBRA || op == POP_RET ||
                op == POP_EXIT)
                return inl_fail(err, errlen, call.line,
                                "multi-target indirect call tail must be straight-line");
            if (op == POP_CALL)
                return inl_fail(err, errlen, call.line,
                                "multi-target indirect call before another call unsupported");
        }
        // Snapshot the tail (deep-copied per candidate below).
        size_t ntail = blk->count - ii - 1;
        ptx_instr_t *tail = NULL;
        if (ntail > 0) {
            tail = malloc(sizeof(*tail) * ntail);
            if (!tail)
                return inl_fail(err, errlen, call.line, "out of memory");
            for (size_t k = 0; k < ntail; k++)
                tail[k] = blk->instrs[ii + 1 + k];
        }
        // Truncate the original block to the pre-call instructions.
        blk->count = ii;

        char retlab[64];
        snprintf(retlab, sizeof(retlab), "__idc_r_%d", seq);

        // Comparison checks, one block each (a branch ends a block).
        for (int j = 0; j < n; j++) {
            char areg[64], preg[64], tlab[64];
            snprintf(areg, sizeof(areg), "__idc_a_%d_%d", seq, j);
            snprintf(preg, sizeof(preg), "__idc_p_%d_%d", seq, j);
            snprintf(tlab, sizeof(tlab), "__idc_t_%d_%d", seq, j);
            if (func_add_reg(fn, areg, PTY_U64) != 0 ||
                func_add_reg(fn, preg, PTY_PRED) != 0)
                goto oom_tail;
            size_t cb = (size_t)-1;
            if (j == 0) {
                cb = bi;
            } else {
                char clab[64];
                snprintf(clab, sizeof(clab), "__idc_c_%d_%d", seq, j);
                cb = new_block(fn, &tp, clab);
                if (cb == (size_t)-1 || tp.failed)
                    goto oom_tail;
            }
            ptx_instr_t mv, sp, br;
            inl_make_mov_func(&mv, areg, call.call_cand[j],
                              call.line);
            if (block_add_instr(fn, cb, &mv, &tp, call.line) != 0)
                goto oom_tail;
            memset(&sp, 0, sizeof(sp));
            sp.op = POP_SETP;
            sp.ty = PTY_U64;
            sp.cmp = PCM_EQ;
            sp.line = call.line;
            snprintf(sp.dst, sizeof(sp.dst), "%s", preg);
            sp.src[0].kind = OPD_REG;
            snprintf(sp.src[0].reg, sizeof(sp.src[0].reg), "%s",
                     call.target);
            sp.src[1].kind = OPD_REG;
            snprintf(sp.src[1].reg, sizeof(sp.src[1].reg), "%s", areg);
            sp.nsrc = 2;
            if (block_add_instr(fn, cb, &sp, &tp, call.line) != 0)
                goto oom_tail;
            memset(&br, 0, sizeof(br));
            br.op = POP_CBRA;
            br.line = call.line;
            snprintf(br.pred, sizeof(br.pred), "%s", preg);
            snprintf(br.target, sizeof(br.target), "%s", tlab);
            if (block_add_instr(fn, cb, &br, &tp, call.line) != 0)
                goto oom_tail;
            if (tp.failed)
                goto oom_tail;
        }
        // Unmatched fallthrough: no candidate matched the runtime
        // pointer (invalid PTX, but keep the CFG structured). Returning
        // here also keeps the last check's false edge out of the
        // selection merge block emitted below.
        {
            char ulab[64];
            snprintf(ulab, sizeof(ulab), "__idc_u_%d", seq);
            size_t ub = new_block(fn, &tp, ulab);
            if (ub == (size_t)-1 || tp.failed)
                goto oom_tail;
            ptx_instr_t br;
            memset(&br, 0, sizeof(br));
            br.op = POP_BRA;
            br.line = call.line;
            snprintf(br.target, sizeof(br.target), "%s", retlab);
            if (block_add_instr(fn, ub, &br, &tp, call.line) != 0)
                goto oom_tail;
        }
        // One call block per candidate, then a shared ret block.
        for (int j = 0; j < n; j++) {
            char tlab[64];
            snprintf(tlab, sizeof(tlab), "__idc_t_%d_%d", seq, j);
            size_t tb = new_block(fn, &tp, tlab);
            if (tb == (size_t)-1 || tp.failed)
                goto oom_tail;
            ptx_instr_t c;
            if (inl_clone_instr(&c, &call, call.call_cand[j], err, errlen,
                                call.line) != 0)
                goto oom_tail;
            if (block_add_instr(fn, tb, &c, &tp, call.line) != 0) {
                free(c.call_args);
                goto oom_tail;
            }
            for (size_t k = 0; k + 1 < ntail; k++) {
                ptx_instr_t tc;
                if (inl_clone_instr(&tc, &tail[k], NULL, err, errlen,
                                    call.line) != 0)
                    goto oom_tail;
                if (block_add_instr(fn, tb, &tc, &tp, call.line) != 0) {
                    free(tc.call_args);
                    goto oom_tail;
                }
            }
            ptx_instr_t br;
            memset(&br, 0, sizeof(br));
            br.op = POP_BRA;
            br.line = call.line;
            snprintf(br.target, sizeof(br.target), "%s", retlab);
            if (block_add_instr(fn, tb, &br, &tp, call.line) != 0)
                goto oom_tail;
        }
        {
            size_t rb = new_block(fn, &tp, retlab);
            if (rb == (size_t)-1 || tp.failed)
                goto oom_tail;
            ptx_instr_t r;
            memset(&r, 0, sizeof(r));
            r.op = (term == POP_EXIT) ? POP_EXIT : POP_RET;
            r.line = call.line;
            if (block_add_instr(fn, rb, &r, &tp, call.line) != 0)
                goto oom_tail;
        }
        free(call.call_args);
        free(tail);
        seq++;
        continue;

oom_tail:
        free(call.call_args);
        free(tail);
        if (tp.failed)
            return -1;
        return inl_fail(err, errlen, call.line, "out of memory");
    }
    return 0;
}

static ACUDA_NOINLINE int inl_resolve_indirect(
    ptx_unit_t *u, ptx_func_t *fn, char *err, size_t errlen) {
    inl_fset_t fmap[64];
    int nf = 0;
    for (size_t b = 0; b < fn->nblocks; b++) {
        for (size_t i = 0; i < fn->blocks[b].count; i++) {
            ptx_instr_t *in = &fn->blocks[b].instrs[i];
            if (in->op == POP_CALL &&
                inl_find_func(u, in->target) < 0 &&
                extfunc_find(u, in->target) < 0) {
                const char *cand[8];
                int nc = 0;
                int have_list = 0;
                if (in->target_list[0]) {
                    for (int ci = 0; ci < fn->ncalltargets; ci++) {
                        if (strcmp(fn->calltargets[ci].label,
                                   in->target_list) != 0)
                            continue;
                        have_list = 1;
                        for (int k = 0;
                             k < fn->calltargets[ci].n && nc < 8; k++)
                            cand[nc++] = fn->calltargets[ci].funcs[k];
                        break;
                    }
                }
                if (!have_list) {
                    inl_fset_t *e = inl_fset_lookup(fmap,
                                                    nf, in->target);
                    if (e && !e->poisoned && !e->overflow) {
                        for (int k = 0; k < e->n && nc < 8; k++)
                            cand[nc++] = e->funcs[k];
                    }
                }
                if (nc == 1) {
                    snprintf(in->target, sizeof(in->target), "%s",
                             cand[0]);
                } else if (nc > 1) {
                    in->ncall_cand = nc > 8 ? 8 : nc;
                    for (int k = 0; k < in->ncall_cand; k++)
                        snprintf(in->call_cand[k],
                                 sizeof(in->call_cand[0]), "%s",
                                 cand[k]);
                }
                for (int k = 0; k < nc && nc <= 8; k++) {
                    if (inl_find_func(u, cand[k]) < 0 &&
                        extfunc_find(u, cand[k]) < 0) {
                        snprintf(err, errlen,
                                 "line %d: unknown function '%s' in "
                                 "indirect call target list",
                                 in->line, cand[k]);
                        return -1;
                    }
                }
            }
            if (!in->dst[0])
                continue;
            const char *known = NULL;
            inl_fset_t *srcset = NULL;
            if (in->op == POP_MOV && in->nsrc == 1) {
                if (in->src[0].kind == OPD_PARAM &&
                    (inl_find_func(u, in->src[0].param) >= 0 ||
                     extfunc_find(u, in->src[0].param) >= 0)) {
                    known = in->src[0].param;
                } else if (in->src[0].kind == OPD_REG) {
                    srcset = inl_fset_lookup(fmap, nf,
                                             in->src[0].reg);
                }
            } else if (in->op == POP_SELP && in->nsrc >= 3 &&
                       in->src[0].kind == OPD_REG &&
                       in->src[1].kind == OPD_REG) {
                inl_fset_t *a = inl_fset_lookup(fmap, nf,
                                                in->src[0].reg);
                inl_fset_t *bb = inl_fset_lookup(fmap, nf,
                                                 in->src[1].reg);
                inl_fset_t *e = inl_fset_ensure(fmap, &nf, in->dst);
                if (e) {
                    if (a && bb && !a->poisoned && !bb->poisoned) {
                        for (int k = 0; k < a->n; k++)
                            inl_fset_add(e, a->funcs[k]);
                        for (int k = 0; k < bb->n; k++)
                            inl_fset_add(e, bb->funcs[k]);
                    } else {
                        e->n = 0;
                        e->poisoned = 1;
                    }
                }
                continue;
            }
            inl_fset_t *e = inl_fset_ensure(fmap, &nf, in->dst);
            if (!e)
                continue;
            if (known) {
                if (e->poisoned) {
                    e->n = 0;
                    e->overflow = 0;
                    e->poisoned = 0;
                }
                inl_fset_add(e, known);
            } else if (srcset && !srcset->poisoned) {
                if (e->poisoned) {
                    e->n = 0;
                    e->overflow = 0;
                    e->poisoned = 0;
                }
                for (int k = 0; k < srcset->n; k++)
                    inl_fset_add(e, srcset->funcs[k]);
            } else {
                e->n = 0;
                e->poisoned = 1;
            }
        }
    }
    return inl_rewrite_dispatch(u, fn, err, errlen);
}

// Function pointer tables in .const/.global initializers name functions.
// Once every function is known, replace the symbolic lane with the same
// compile-time address value that `mov.u64 %rd, func` materializes.
static int inl_resolve_symbols(ptx_unit_t *u, char *err, size_t errlen) {
    for (size_t ci = 0; ci < u->nconsts; ci++) {
        ptx_const_t *c = &u->consts[ci];
        for (int si = 0; si < c->nsyms; si++) {
            int fi = inl_find_func(u, c->syms[si].name);
            if (fi < 0)
                continue; // ordinary symbolic const reference: keep zero
            uint64_t addr = 0x1000u + (uint64_t)fi;
            if ((size_t)c->syms[si].off + 8 > c->nbytes) {
                return inl_fail(err, errlen, 0,
                                "function address lane outside .const");
            }
            for (int k = 0; k < 8; k++)
                c->bytes[c->syms[si].off + (uint32_t)k] =
                    (uint8_t)(addr >> (8 * k));
        }
    }
    for (size_t gi = 0; gi < u->nglobals; gi++) {
        ptx_const_t *g = &u->globals[gi];
        for (int si = 0; si < g->nsyms; si++) {
            int fi = inl_find_func(u, g->syms[si].name);
            if (fi < 0) {
                snprintf(err, errlen,
                         "line 0: unknown symbol '%s' in .global initializer",
                         g->syms[si].name);
                return -1;
            }
            uint64_t addr = 0x1000u + (uint64_t)fi;
            if ((size_t)g->syms[si].off + 8 > g->nbytes)
                return inl_fail(err, errlen, 0,
                                "function address lane outside .global");
            for (int k = 0; k < 8; k++)
                g->bytes[g->syms[si].off + (uint32_t)k] =
                    (uint8_t)(addr >> (8 * k));
        }
    }
    return 0;
}

static int inl_expand_multiblock(ptx_unit_t *u, char *err, size_t errlen);

// Inline calls + desugar slots in every function that needs it.
// Runs at the end of ptx_parse (post-DCE), so emission only ever
// sees straight-line code. Functions without calls/slots still get
// a sweep dropping own return-slot stores (kernel ABI, previously
// dropped at parse).
int inl_run(ptx_unit_t *u, char *err, size_t errlen) {
    if (inl_resolve_symbols(u, err, errlen) != 0)
        return -1;
    for (size_t f = 0; f < u->nfuncs; f++)
        inl_fix_packed_pointers(&u->funcs[f]);
    for (size_t f = 0; f < u->nfuncs; f++)
        if (inl_lower_brx(u, &u->funcs[f], err, errlen) != 0)
            return -1;
    for (size_t f = 0; f < u->nfuncs; f++)
        if (inl_resolve_indirect(u, &u->funcs[f], err, errlen) != 0)
            return -1;
    if (inl_expand_multiblock(u, err, errlen) != 0)
        return -1;
    int taggen = 0; // unit-shared: fresh names stay unique across funcs
    for (size_t f = 0; f < u->nfuncs; f++) {
        ptx_func_t *fn = &u->funcs[f];
        if (!inl_fn_has_call(fn)) {
            for (size_t b = 0; b < fn->nblocks; b++) {
                size_t w = 0;
                for (size_t i = 0; i < fn->blocks[b].count; i++) {
                    ptx_instr_t *in = &fn->blocks[b].instrs[i];
                    if (in->op == POP_ST && in->space == PSP_PARAM) {
                        int pi = inl_find_param(fn, in->src[0].param);
                        if (pi >= 0 &&
                            fn->params[(size_t)pi].is_retval)
                            continue; // ABI store: not modeled
                    }
                    if (w != i)
                        fn->blocks[b].instrs[w] = fn->blocks[b].instrs[i];
                    w++;
                }
                fn->blocks[b].count = w;
            }
            continue;
        }
        const char *stack[72];
        stack[0] = fn->name;
        inl_ctx_t cx;
        cx.stack = stack;
        cx.depth = 1;
        cx.counter = &taggen;
        for (size_t b = 0; b < fn->nblocks; b++) {
            inl_cell_t cells[256]; // keys are slot names (decl-capped)
            int ncells = 0;
            ptx_instr_t *out = NULL;
            size_t nout = 0, ocap = 0;
            if (inl_walk_block(u, fn, fn->blocks[b].instrs,
                               fn->blocks[b].count, &out, &nout, &ocap,
                               cells, &ncells, &cx, err, errlen) != 0) {
                free(out);
                return -1;
            }
            free(fn->blocks[b].instrs);
            fn->blocks[b].instrs = out;
            fn->blocks[b].count = nout;
            // (out may be NULL with nout 0: empty block is legal.)
        }
    }
    return 0;
}


// ---- Multi-block callee expansion (single-return .param form) ----
//
// The straight-line inliner below replaces a call in place. A callee
// with several blocks needs new caller blocks, so expand those first:
// splice the (renamed) callee CFG after the call block, turn the single
// ret block into the caller's continuation (the call tail must end in
// ret/exit), and turn the call into a branch to the callee entry.
// Constraints (honest errors otherwise): .param-form callee, one ret
// block, no nested calls/locals/slots, call in the final caller block,
// straight-line call tail.

static int inl_label_exists(const ptx_func_t *fn, const char *label) {
    for (size_t i = 0; i < fn->nblocks; i++)
        if (strcmp(fn->blocks[i].label, label) == 0)
            return 1;
    return 0;
}

static int inl_expand_multiblock(ptx_unit_t *u, char *err, size_t errlen) {
    int tag = 0;
    for (size_t f = 0; f < u->nfuncs; f++) {
        ptx_func_t *fn = &u->funcs[f];
        for (size_t bi = 0; bi < fn->nblocks; bi++) {
            ptx_block_t *blk = &fn->blocks[bi];
            for (size_t ii = 0; ii < blk->count; ii++) {
                ptx_instr_t *call = &blk->instrs[ii];
                if (call->op != POP_CALL)
                    continue;
                int ci = inl_find_func(u, call->target);
                if (ci < 0)
                    continue; // existing inliner reports it
                if (call->nsrc > 8)
                    return inl_fail(err, errlen, call->line,
                                    "device call with more than 8 arguments unsupported");
                ptx_func_t *callee = &u->funcs[(size_t)ci];
                if (callee->nblocks <= 1)
                    continue; // straight-line path
                if (callee->nlocals > 0 || callee->nslots > 0)
                    return inl_fail(err, errlen, call->line,
                                    "multi-block callee uses .local/.param slots");
                if (callee->nrparams > 0)
                    return inl_fail(err, errlen, call->line,
                                    "multi-block .reg-form callee unsupported");
                // Exactly one ret, and it must be the callee's tail.
                int retb = -1;
                for (size_t b = 0; b < callee->nblocks; b++) {
                    for (size_t k = 0; k < callee->blocks[b].count; k++) {
                        ptx_op_t op = callee->blocks[b].instrs[k].op;
                        if (op == POP_RET) {
                            if (retb >= 0)
                                return inl_fail(err, errlen, call->line,
                                                "multi-block callee has multiple returns");
                            retb = (int)b;
                        } else if (op == POP_EXIT) {
                            return inl_fail(err, errlen, call->line,
                                            "multi-block callee uses exit");
                        } else if (op == POP_CALL) {
                            return inl_fail(err, errlen, call->line,
                                            "multi-block callee contains a call");
                        }
                    }
                }
                if (retb < 0)
                    return inl_fail(err, errlen, call->line,
                                    "multi-block callee has no ret");
                if ((size_t)retb != callee->nblocks - 1)
                    return inl_fail(err, errlen, call->line,
                                    "multi-block callee ret must be last");
                // Call must be in the final caller block with a
                // straight-line tail ending in ret/exit.
                if (bi != fn->nblocks - 1)
                    return inl_fail(err, errlen, call->line,
                                    "multi-block call must be in the final block");
                if (blk->count == 0)
                    return inl_fail(err, errlen, call->line,
                                    "multi-block call without tail");
                int term = blk->instrs[blk->count - 1].op;
                if (term != POP_RET && term != POP_EXIT)
                    return inl_fail(err, errlen, call->line,
                                    "multi-block call tail must end in ret/exit");
                for (size_t k = ii + 1; k + 1 < blk->count; k++) {
                    ptx_op_t op = blk->instrs[k].op;
                    if (op == POP_BRA || op == POP_CBRA || op == POP_RET ||
                        op == POP_EXIT)
                        return inl_fail(err, errlen, call->line,
                                        "multi-block call tail must be straight-line");
                }
                // Resolve slot arguments defined before the call.
                inl_cell_t cells[256];
                int ncells = 0;
                for (size_t k = 0; k < ii; k++) {
                    ptx_instr_t *in = &blk->instrs[k];
                    if (in->op == POP_ST && in->space == PSP_PARAM &&
                        in->src[0].kind == OPD_ADDR &&
                        !in->src[0].addr_reg[0] &&
                        inl_find_param(fn, in->src[0].param) < 0)
                        inl_cell_set(cells, &ncells, in->src[0].param,
                                     &in->src[1]);
                }
                inl_scope_t sc;
                sc.fn = fn;
                sc.root = fn;
                sc.cells = cells;
                sc.ncells = &ncells;
                sc.seed = NULL;
                sc.nseed = 0;
                sc.ren = NULL;
                sc.nren = 0;
                ptx_opd_t argvals[64];
                int nargs = 0;
                for (int a = 0; a < call->nsrc; a++) {
                    if (a >= 64)
                        return inl_fail(err, errlen, call->line,
                                        "too many call arguments");
                    if (inl_arg_val(&sc, call->src[a].param,
                                    &argvals[nargs], err, errlen,
                                    call->line) != 0)
                        return -1;
                    nargs++;
                }
                int nparams = 0;
                for (size_t p = 0; p < callee->nparams; p++)
                    if (!callee->params[p].is_retval)
                        nparams++;
                if (nargs != nparams)
                    return inl_fail(err, errlen, call->line,
                                    "multi-block call argument count mismatch");
                // Fresh labels for every callee block.
                char (*lmap)[64] = calloc(callee->nblocks, 64);
                if (!lmap)
                    return inl_fail(err, errlen, call->line, "out of memory");
                for (size_t b = 0; b < callee->nblocks; b++) {
                    for (int tries = 0;; tries++) {
                        snprintf(lmap[b], 64, "__m%d_b%zu", ++tag, b);
                        if (!inl_label_exists(fn, lmap[b]))
                            break;
                        if (tries > 64) {
                            free(lmap);
                            return inl_fail(err, errlen, call->line,
                                            "cannot allocate inline label");
                        }
                    }
                }
                // Rename every callee register.
                inl_ren_t *ren = malloc(sizeof(*ren) * (callee->nregs ? callee->nregs : 1));
                int nren = 0;
                if (!ren) {
                    free(lmap);
                    return inl_fail(err, errlen, call->line, "out of memory");
                }
                for (size_t r = 0; r < callee->nregs; r++) {
                    snprintf(ren[nren].old, sizeof(ren[nren].old), "%.63s",
                             callee->regs[r].name);
                    snprintf(ren[nren].fresh, sizeof(ren[nren].fresh),
                             "__m%d_r%zu", tag, r);
                    for (size_t k = 0; k < fn->nregs; k++) {
                        if (strcmp(fn->regs[k].name, ren[nren].fresh) == 0) {
                            free(ren);
                            free(lmap);
                            return inl_fail(err, errlen, call->line,
                                            "inline rename collision");
                        }
                    }
                    if (func_add_reg(fn, ren[nren].fresh,
                                     callee->regs[r].ty) != 0) {
                        free(ren);
                        free(lmap);
                        return inl_fail(err, errlen, call->line,
                                        "out of memory");
                    }
                    nren++;
                }
                // Return register (slot return) is written by the callee.
                char retreg[64] = "";
                int ret_is_slot = 0;
                for (size_t p = 0; p < callee->nparams; p++)
                    if (callee->params[p].is_retval)
                        ret_is_slot = 1;
                if (ret_is_slot) {
                    snprintf(retreg, sizeof(retreg), "__m%d_ret", tag);
                    for (size_t k = 0; k < fn->nregs; k++)
                        if (strcmp(fn->regs[k].name, retreg) == 0)
                            snprintf(retreg, sizeof(retreg), "__m%d_retx", tag);
                    ptx_ty_t rty = PTY_U32;
                    for (size_t p = 0; p < callee->nparams; p++)
                        if (callee->params[p].is_retval)
                            rty = callee->params[p].ty;
                    if (func_add_reg(fn, retreg, rty) != 0) {
                        free(ren);
                        free(lmap);
                        return inl_fail(err, errlen, call->line,
                                        "out of memory");
                    }
                }
                // Build the copied blocks (insert after the call block).
                size_t oldn = fn->nblocks;
                ptx_block_t *nb = realloc(fn->blocks,
                                          sizeof(*nb) * (oldn + callee->nblocks));
                if (!nb) {
                    free(ren);
                    free(lmap);
                    return inl_fail(err, errlen, call->line, "out of memory");
                }
                fn->blocks = nb;
                blk = &fn->blocks[bi]; // realloc may have moved the array
                memset(&fn->blocks[oldn], 0,
                       sizeof(*nb) * callee->nblocks);
                for (size_t b = 0; b < callee->nblocks; b++) {
                    ptx_block_t *dst = &fn->blocks[oldn + b];
                    ptx_block_t *src = &callee->blocks[b];
                    snprintf(dst->label, sizeof(dst->label), "%s", lmap[b]);
                    dst->cap = src->count + 8;
                    dst->instrs = calloc(dst->cap, sizeof(*dst->instrs));
                    if (!dst->instrs) {
                        free(ren);
                        free(lmap);
                        return inl_fail(err, errlen, call->line,
                                        "out of memory");
                    }
                    for (size_t k = 0; k < src->count; k++) {
                        ptx_instr_t *in = &src->instrs[k];
                        ptx_instr_t out;
                        if (in->op == POP_RET)
                            break; // tail merged below
                        if (in->op == POP_LD && in->space == PSP_PARAM &&
                            in->src[0].kind == OPD_ADDR &&
                            !in->src[0].addr_reg[0]) {
                            int pi = inl_find_param(callee, in->src[0].param);
                            if (pi < 0 || callee->params[(size_t)pi].is_retval) {
                                free(ren);
                                free(lmap);
                                return inl_fail(err, errlen, call->line,
                                                "multi-block callee reads its return slot");
                            }
                            int ai = 0;
                            for (int p = 0; p < pi; p++)
                                if (!callee->params[p].is_retval)
                                    ai++;
                            memset(&out, 0, sizeof(out));
                            out.op = POP_MOV;
                            out.ty = in->ty;
                            out.line = in->line;
                            if (inl_ren_copy(out.dst, sizeof(out.dst), ren,
                                             nren, in->dst, err, errlen,
                                             in->line) != 0) {
                                free(ren);
                                free(lmap);
                                return -1;
                            }
                            out.src[0] = argvals[ai];
                            out.nsrc = 1;
                            dst->instrs[dst->count++] = out;
                            continue;
                        }
                        if (in->op == POP_ST && in->space == PSP_PARAM) {
                            int pi = inl_find_param(callee, in->src[0].param);
                            if (pi < 0 || !callee->params[(size_t)pi].is_retval) {
                                free(ren);
                                free(lmap);
                                return inl_fail(err, errlen, call->line,
                                                "multi-block callee stores a param slot");
                            }
                            if (!retreg[0]) {
                                free(ren);
                                free(lmap);
                                return inl_fail(err, errlen, call->line,
                                                "multi-block callee return without target");
                            }
                            memset(&out, 0, sizeof(out));
                            out.op = POP_MOV;
                            out.ty = in->ty;
                            out.line = in->line;
                            snprintf(out.dst, sizeof(out.dst), "%s", retreg);
                            out.src[0] = in->src[1];
                            if (out.src[0].kind == OPD_REG ||
                                out.src[0].kind == OPD_REGOFF) {
                                if (inl_ren_copy(out.src[0].reg,
                                                 sizeof(out.src[0].reg), ren,
                                                 nren, in->src[1].reg, err,
                                                 errlen, in->line) != 0) {
                                    free(ren);
                                    free(lmap);
                                    return -1;
                                }
                            }
                            out.nsrc = 1;
                            dst->instrs[dst->count++] = out;
                            continue;
                        }
                        if (inl_copy_instr(&out, in, ren, nren, err, errlen) != 0) {
                            free(ren);
                            free(lmap);
                            return -1;
                        }
                        if (out.op == POP_BRA || out.op == POP_CBRA) {
                            int tg = -1;
                            for (size_t q = 0; q < callee->nblocks; q++)
                                if (strcmp(callee->blocks[q].label,
                                           out.target) == 0) {
                                    tg = (int)q;
                                    break;
                                }
                            if (tg < 0) {
                                free(ren);
                                free(lmap);
                                return inl_fail(err, errlen, in->line,
                                                "multi-block callee branch to unknown label");
                            }
                            snprintf(out.target, sizeof(out.target), "%s",
                                     lmap[tg]);
                        }
                        dst->instrs[dst->count++] = out;
                    }
                    // Explicit fallthrough to the next callee block.
                    if ((int)b != retb) {
                        const ptx_instr_t *last =
                            dst->count ? &dst->instrs[dst->count - 1] : NULL;
                        if (!last || (last->op != POP_BRA &&
                                      last->op != POP_CBRA)) {
                            ptx_instr_t br;
                            memset(&br, 0, sizeof(br));
                            br.op = POP_BRA;
                            br.line = call->line;
                            snprintf(br.target, sizeof(br.target), "%s",
                                     lmap[b + 1]);
                            dst->instrs[dst->count++] = br;
                        }
                    }
                    if ((int)b == retb) {
                        // Merge the caller's straight-line tail, then its
                        // original terminator (ret/exit).
                        for (size_t k = ii + 1; k < blk->count; k++) {
                            ptx_instr_t out = blk->instrs[k];
                            if (out.op == POP_ST && out.space == PSP_PARAM)
                                continue; // slot stores already resolved
                            ptx_instr_t mv;
                            int rewritten = 0;
                            if (out.op == POP_LD && out.space == PSP_PARAM &&
                                out.src[0].kind == OPD_ADDR &&
                                !out.src[0].addr_reg[0]) {
                                int pi = inl_find_param(fn, out.src[0].param);
                                if (pi < 0 && retreg[0]) {
                                    // ld.param from the call's return slot
                                    memset(&mv, 0, sizeof(mv));
                                    mv.op = POP_MOV;
                                    mv.ty = out.ty;
                                    mv.line = out.line;
                                    snprintf(mv.dst, sizeof(mv.dst), "%s",
                                             out.dst);
                                    memset(&mv.src[0], 0, sizeof(mv.src[0]));
                                    mv.src[0].kind = OPD_REG;
                                    snprintf(mv.src[0].reg,
                                             sizeof(mv.src[0].reg), "%s",
                                             retreg);
                                    mv.nsrc = 1;
                                    rewritten = 1;
                                }
                            }
                            if (rewritten)
                                dst->instrs[dst->count++] = mv;
                            else if (out.op != POP_RET && out.op != POP_EXIT)
                                dst->instrs[dst->count++] = out;
                        }
                        ptx_instr_t ex;
                        memset(&ex, 0, sizeof(ex));
                        ex.op = POP_RET;
                        ex.line = call->line;
                        dst->instrs[dst->count++] = ex;
                    }
                }
                // Caller block: keep pre-call code, branch to the entry.
                blk->count = ii;
                ptx_instr_t br;
                memset(&br, 0, sizeof(br));
                br.op = POP_BRA;
                br.line = call->line;
                snprintf(br.target, sizeof(br.target), "%s", lmap[0]);
                if (blk->count + 1 > blk->cap) {
                    size_t ncap = blk->cap ? blk->cap * 2 : 8;
                    ptx_instr_t *ni = realloc(blk->instrs,
                                              sizeof(*ni) * ncap);
                    if (!ni) {
                        free(ren);
                        free(lmap);
                        return inl_fail(err, errlen, call->line,
                                        "out of memory");
                    }
                    blk->instrs = ni;
                    blk->cap = ncap;
                }
                blk->instrs[blk->count++] = br;
                fn->nblocks = oldn + callee->nblocks;
                free(ren);
                free(lmap);
                // Continue scanning from the new blocks.
                bi = oldn - 1; // loop ++ moves to the first spliced block
                break;
            }
        }
    }
    return 0;
}
