// Arithmetic, conversion, bit-manipulation, comparison and select
// helpers and opcode emission (split from arith.c).
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "spirv_internal.h"
#include "instructions/logic_shift/logic_shift.h"
#include "instructions/conversion/conversion.h"
#include "instructions/memory/memory.h"
#include "instructions/arith/arith.h"
#include "instructions/arith/integer/integer.h"
#include "instructions/data_movement/prmt.h"
#include "instructions/video/video_ops.h"
#include "instructions/compare/compare.h"
#include "instructions/arith/float/float.h"

void emit_arith_op(ctx_t *c, spv_blob_t *b, const ptx_instr_t *in) {
    int line = in->line;
    (void)line;
    switch (in->op) {
    case POP_ADD:
    case POP_SUB:
        emit_add_sub_op(c, b, in);
        return;
    case POP_MULLO:
        emit_mullo_op(c, b, in);
        return;
    case POP_MADLO:
        emit_madlo_op(c, b, in);
        return;
    case POP_FMA:
        emit_fma_op(c, b, in);
        return;
    case POP_MULWIDE:
        emit_mulwide_op(c, b, in);
        return;
    case POP_MADWIDE:
        emit_madwide_op(c, b, in);
        return;
    case POP_MULHI:
        emit_mulhi_op(c, b, in);
        return;
    case POP_DOT2:
        emit_dot2_op(c, b, in);
        return;
    case POP_DP4A:
        emit_dp4a_op(c, b, in);
        return;
    case POP_DP2A:
        emit_dp2a_op(c, b, in);
        return;
    case POP_SAD:
        emit_sad_op(c, b, in);
        return;
    case POP_PRMT:
        emit_prmt_op(c, b, in);
        return;
    case POP_GLSL:
        emit_glsl_op(c, b, in);
        return;
    case POP_ADDSAT:
        emit_addsat_op(c, b, in);
        return;
    case POP_TESTP:
        emit_testp_op(c, b, in);
        return;
    case POP_COPYSIGN:
        emit_copysign_op(c, b, in);
        return;
    case POP_MINNAN:
        emit_minnan_op(c, b, in);
        return;
    case POP_MUL24:
        emit_mul24_op(c, b, in);
        return;
    case POP_MAD24:
        emit_mad24_op(c, b, in);
        return;
    case POP_CLMAD:
        emit_clmad_op(c, b, in);
        return;
    case POP_FNS:
        emit_fns_op(c, b, in);
        return;
    case POP_SZEXT:
        emit_szext_op(c, b, in);
        return;
    case POP_SHF:
        emit_shf_op(c, b, in);
        return;
    case POP_ABS:
        emit_abs_op(c, b, in);
        return;
    case POP_CNOT:
        emit_cnot_op(c, b, in);
        return;
    case POP_LOP3:
        emit_lop3_op(c, b, in);
        return;
    case POP_SLCT:
        emit_slct_op(c, b, in);
        return;
    case POP_PREFETCH:
    case POP_HINT:
        emit_prefetch_hint_op(c, b, in);
        return;
    case POP_POPC:
        emit_popc_op(c, b, in);
        return;
    case POP_CLZ:
        emit_clz_op(c, b, in);
        return;
    case POP_BREV:
        emit_brev_op(c, b, in);
        return;
    case POP_BFE:
        emit_bfe_op(c, b, in);
        return;
    case POP_BFI:
        emit_bfi_op(c, b, in);
        return;
    case POP_BMSK:
        emit_bmsk_op(c, b, in);
        return;
    case POP_BFIND:
        emit_bfind_op(c, b, in);
        return;
    case POP_CVTPACK8:
        emit_cvtpack8_op(c, b, in);
        return;
    case POP_CVT:
        emit_cvt_op(c, b, in);
        return;
    case POP_CVTPACK:
        emit_cvtpack_op(c, b, in);
        return;
    case POP_FP8_TO_F16X2:
        emit_fp8_to_f16x2_op(c, b, in);
        return;
    case POP_F32_TO_FP8X2:
        emit_f32_to_fp8x2_op(c, b, in);
        return;
    case POP_RND:
        emit_rnd_op(c, b, in);
        return;
    case POP_SHR:
    case POP_SHL:
        emit_shift_op(c, b, in);
        return;
    case POP_VSHR:
        emit_vshr_op(c, b, in);
        return;
    case POP_AND:
    case POP_OR:
    case POP_XOR:
        emit_logic_bitwise(c, b, in);
        return;
    case POP_SELP:
        emit_selp_op(c, b, in);
        return;
    case POP_NOT:
        emit_not_op(c, b, in);
        return;
    case POP_NEG:
        emit_neg_op(c, b, in);
        return;
    case POP_MINMAX:
        emit_minmax_op(c, b, in);
        return;
    case POP_RCP:
        emit_rcp_op(c, b, in);
        return;
    case POP_DIV:
    case POP_REM:
        emit_div_rem_op(c, b, in);
        return;
    case POP_SETP:
        emit_setp_op(c, b, in);
        return;
    case POP_SET:
        emit_set_op(c, b, in);
        return;
    default:
        failf(c, line, "unhandled arithmetic op");
        return;
    }
}
