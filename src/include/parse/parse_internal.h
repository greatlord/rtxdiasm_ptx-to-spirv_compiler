// Internal shared declarations for the PTX parser.
#ifndef ACUDA_PARSE_INTERNAL_H
#define ACUDA_PARSE_INTERNAL_H

#include <stddef.h>
#include <stdint.h>
#include "ptx.h"

typedef struct {
    char name[64];
    char internal[64];
    int depth; // anonymous-brace depth where the shadow applies
} scope_bind_t;

typedef struct {
    ptx_lexer_t lx;
    char *err;
    size_t errlen;
    int failed;
    ptx_func_t *fn; // current function (for .reg declaration lookup)
    int predst_n;   // synthetic predicated-store label discriminator
    // Anonymous `{ ... }` scopes: a register re-declared in an inner
    // brace gets a fresh internal name, and references inside that
    // scope resolve to it (block.ptx shadowing).
    scope_bind_t *sbind;
    size_t nsbind, sbind_cap;
    int depth;
    int sm; // .target SM version (0 = unspecified)
    unsigned scope_seq;
    char last_label[64];
} parser_t;

void perr(parser_t *p, int line, const char *msg);
int copy_name(char *dst, size_t cap, const char *src, parser_t *p, int line);
int next(parser_t *p, ptx_tok_t *t);
int expect_punct(parser_t *p, char c, const char *what);
int expect_comma(parser_t *p);
int at_punct(parser_t *p, char c);
int map_ty(const char *s, ptx_ty_t *out);
int map_cmp(const char *s, ptx_cmp_t *out);
int split_op(const char *op, char parts[6][32], int *nparts);
int parse_operand(parser_t *p, ptx_opd_t *op);
int parse_dst(parser_t *p, char *name, size_t namelen, int want_pred);
int parse_brace_regs(parser_t *p, char names[][64], int n, int opline);
int parse_vec_operand(parser_t *p, char out[][64], int n, ptx_ty_t elem,
                      int opline);
int block_add_instr(ptx_func_t *fn, size_t bi, const ptx_instr_t *in,
                    parser_t *p, int line);
size_t new_block(ptx_func_t *fn, parser_t *p, const char *label);
int func_add_reg(ptx_func_t *fn, const char *name, ptx_ty_t ty);
ptx_vecdecl_t *vec_find(ptx_func_t *fn, const char *name);
int vec_elem_bytes(ptx_ty_t e);
int vec_elem_ty(const char *s, ptx_ty_t *out);
int vec_component_reg(parser_t *p, const char *text, char out[64], int line);
void ld_f32_bits_fixup(ptx_func_t *fn, ptx_instr_t *in);
void st_f32_bits_fixup(ptx_func_t *fn, ptx_instr_t *in);
int cvt_int_ty(const char *s, ptx_ty_t *ty, int *sign);
int setp_ty(parser_t *p, const char *s, ptx_ty_t *ty, int *sign,
            int line);
int func_add_vec(ptx_func_t *fn, const char *name, int n, ptx_ty_t elem);
int parse_shared_decl(parser_t *p, ptx_unit_t *unit);
int parse_extern_func(parser_t *p, ptx_unit_t *unit);
int parse_extern_shared(parser_t *p, ptx_unit_t *unit);
int parse_param_slot(parser_t *p, ptx_func_t *fn, const char *head,
                     int headline);
int parse_local_decl(parser_t *p, ptx_func_t *fn);
int parse_module_local_decl(parser_t *p, ptx_unit_t *unit);
int parse_common_decl(parser_t *p, ptx_unit_t *unit);
int parse_const_decl(parser_t *p, ptx_unit_t *unit);
int parse_global_decl(parser_t *p, ptx_unit_t *unit);
int parse_texref_decl(parser_t *p, ptx_unit_t *unit);
int parse_reg_decl(parser_t *p, ptx_func_t *fn, const char *decltok,
                   int dtline);
void scope_pop(parser_t *p);
int vec_type(const char *s, int *n, ptx_ty_t *elem);
int parse_entry(parser_t *p, ptx_unit_t *unit, int is_entry);
int inl_run(ptx_unit_t *u, char *err, size_t errlen);
int ptx_hostcall_prepare(ptx_unit_t *u, ptx_func_t *fn, char *err,
                         size_t errlen);
int parse_atom_instr(parser_t *p, ptx_func_t *fn, size_t bi,
                     const char *opname, int opline);
int parse_texture_instr(parser_t *p, ptx_func_t *fn, size_t bi,
                        const char *opname, int opline);
int parse_surface_instr(parser_t *p, ptx_func_t *fn, size_t bi,
                        const char *opname, int opline);
int parse_cvt_micro(parser_t *p, ptx_func_t *fn, size_t bi,
                    const char *opname, int opline);
int parse_mbarrier_instr(parser_t *p, ptx_func_t *fn, size_t bi,
                         const char *pred, int pred_neg,
                         const char *opname, int opline);
int parse_cluster_instr(parser_t *p, ptx_func_t *fn, size_t bi,
                        const char *pred, int pred_neg,
                        const char *opname, int opline);
int parse_multimem(parser_t *p, ptx_func_t *fn, size_t bi,
                   const char *pred, int pred_neg, const char *opname,
                   int opline);
int parse_red_async(parser_t *p, ptx_func_t *fn, size_t bi,
                    const char *pred, int pred_neg,
                    const char *opname, int opline);
int parse_matrix_instr(parser_t *p, ptx_func_t *fn, size_t bi,
                       const char *pred, int pred_neg, const char *opname,
                       int opline);
int parse_instr(parser_t *p, ptx_func_t *fn, size_t bi, const char *pred,
                int pred_neg, const char *opname, int opline);

#endif
