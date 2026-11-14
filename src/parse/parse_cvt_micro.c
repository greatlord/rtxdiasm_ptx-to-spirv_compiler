// cvt micro-float forms: e2m1/e2m3/e3m2 (f4/f6), e4m3/e5m2 x4,
// ue8m0 scale factors, s2f6 fixed point, and the .scaled::n2::ue8m0
// block-scaled conversions.
#include <stdio.h>
#include <string.h>
#include "ptx.h"
#include "parse/parse_internal.h"

#define CVT_MAX_PARTS 10

static int cvt_split(const char *op, char parts[CVT_MAX_PARTS][32],
                     int *np) {
    *np = 0;
    const char *s = op;
    while (*s) {
        if (*np >= CVT_MAX_PARTS)
            return -1;
        size_t n = 0;
        while (s[n] && s[n] != '.' && n + 1 < 32)
            n++;
        if (s[n] == '.' && n + 1 >= 32)
            return -1;
        memcpy(parts[*np], s, n);
        parts[*np][n] = '\0';
        (*np)++;
        s += n;
        if (*s == '.')
            s++;
        else if (*s)
            return -1;
    }
    return 0;
}

static int cvt_value_fmt(const char *s) {
    if (strcmp(s, "e4m3x2") == 0) return CVTMF_E4M3X2;
    if (strcmp(s, "e5m2x2") == 0) return CVTMF_E5M2X2;
    if (strcmp(s, "e2m1x2") == 0) return CVTMF_E2M1X2;
    if (strcmp(s, "e2m3x2") == 0) return CVTMF_E2M3X2;
    if (strcmp(s, "e3m2x2") == 0) return CVTMF_E3M2X2;
    if (strcmp(s, "e4m3x4") == 0) return CVTMF_E4M3X4;
    if (strcmp(s, "e5m2x4") == 0) return CVTMF_E5M2X4;
    if (strcmp(s, "e2m1x4") == 0) return CVTMF_E2M1X4;
    if (strcmp(s, "e2m3x4") == 0) return CVTMF_E2M3X4;
    if (strcmp(s, "e3m2x4") == 0) return CVTMF_E3M2X4;
    if (strcmp(s, "ue8m0x2") == 0) return CVTMF_UE8M0X2;
    if (strcmp(s, "s2f6x2") == 0) return CVTMF_S2F6X2;
    if (strcmp(s, "f16x2") == 0) return CVTMF_F16X2;
    if (strcmp(s, "bf16x2") == 0) return CVTMF_BF16X2;
    if (strcmp(s, "f32") == 0) return CVTMF_SRC_F32;
    return 0;
}

int parse_cvt_micro(parser_t *p, ptx_func_t *fn, size_t bi,
                    const char *opname, int opline) {
    char parts[CVT_MAX_PARTS][32];
    int np = 0;
    memset(parts, 0, sizeof(parts));
    if (cvt_split(opname, parts, &np) != 0) {
        perr(p, opline, "bad cvt opcode");
        return -1;
    }
    ptx_instr_t in;
    memset(&in, 0, sizeof(in));
    in.line = opline;
    in.op = POP_CVT_MICRO;
    int dtype = 0, atype = 0, nfmt = 0;
    for (int i = 1; i < np; i++) {
        const char *s = parts[i];
        if (strcmp(s, "rn") == 0) { in.cvt_frnd = 0; continue; }
        if (strcmp(s, "rz") == 0) { in.cvt_frnd = 1; continue; }
        if (strcmp(s, "rp") == 0) { in.cvt_frnd = 2; continue; }
        if (strcmp(s, "satfinite") == 0) { in.cvt_satfinite = 1; continue; }
        if (strcmp(s, "relu") == 0) { in.cvt_relu = 1; continue; }
        if (strcmp(s, "rs") == 0) { in.cvt_rs = 1; continue; }
        if (strcmp(s, "scaled::n2::ue8m0") == 0) {
            in.cvt_scaled = 1;
            continue;
        }
        if (strcmp(s, "ftz") == 0)
            continue;
        int f = cvt_value_fmt(s);
        if (f) {
            if (!nfmt)
                dtype = f;
            else if (nfmt == 1)
                atype = f;
            else {
                perr(p, opline, "too many cvt type qualifiers");
                return -1;
            }
            nfmt++;
            continue;
        }
        perr(p, opline, "bad cvt-mf qualifier");
        return -1;
    }
    if (nfmt != 2 || !dtype || !atype) {
        perr(p, opline, "cvt-mf needs destination and source types");
        return -1;
    }
    in.cvt_dst = dtype;
    if (atype == CVTMF_F16X2)
        in.cvt_src = CVTMF_SRC_F16X2;
    else if (atype == CVTMF_BF16X2)
        in.cvt_src = CVTMF_SRC_BF16X2;
    else
        in.cvt_src = atype;
    if (in.cvt_rs) {
        perr(p, opline, "cvt .rs stochastic rounding is not implemented");
        return -1;
    }
    // Every packed micro-float / s2f6 destination requires .satfinite;
    // the f16x2/bf16x2 unpack and ue8m0 forms do not.
    if (!in.cvt_satfinite && dtype != CVTMF_F16X2 &&
        dtype != CVTMF_BF16X2 && dtype != CVTMF_UE8M0X2) {
        perr(p, opline, "packed micro-float cvt requires .satfinite");
        return -1;
    }
    if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
        return -1;
    if (expect_comma(p) != 0)
        return -1;
    int x4 = dtype == CVTMF_E4M3X4 || dtype == CVTMF_E5M2X4 ||
             dtype == CVTMF_E2M1X4 || dtype == CVTMF_E2M3X4 ||
             dtype == CVTMF_E3M2X4;
    int f16dst = dtype == CVTMF_F16X2 || dtype == CVTMF_BF16X2;
    if (x4) {
        char regs[4][64];
        if (parse_vec_operand(p, regs, 4, PTY_F32, opline) != 0)
            return -1;
        for (int i = 0; i < 4; i++) {
            memset(&in.src[i], 0, sizeof(in.src[i]));
            in.src[i].kind = OPD_REG;
            snprintf(in.src[i].reg, sizeof(in.src[i].reg), "%.63s", regs[i]);
        }
        in.nsrc = 4;
        if (expect_comma(p) != 0 || parse_operand(p, &in.src[4]) != 0)
            return -1;
    } else if (f16dst) {
        // Unpack a packed micro pair into f16x2/bf16x2, optionally
        // scaled by ue8m0x2.
        if (parse_operand(p, &in.src[0]) != 0)
            return -1;
        in.nsrc = 1;
        if (at_punct(p, ',')) {
            if (!in.cvt_scaled || dtype != CVTMF_BF16X2) {
                perr(p, opline, "scale factor needs .scaled::n2::ue8m0 "
                                "and a .bf16x2 destination");
                return -1;
            }
            if (parse_operand(p, &in.src[1]) != 0)
                return -1;
            in.nsrc = 2;
        }
    } else if (in.cvt_src == CVTMF_SRC_F16X2 ||
               in.cvt_src == CVTMF_SRC_BF16X2) {
        if (parse_operand(p, &in.src[0]) != 0)
            return -1;
        in.nsrc = 1;
        if (at_punct(p, ',')) {
            if (!in.cvt_scaled || dtype != CVTMF_S2F6X2) {
                perr(p, opline, "scale factor needs .scaled::n2::ue8m0 "
                                "and an .s2f6x2 destination");
                return -1;
            }
            if (parse_operand(p, &in.src[1]) != 0)
                return -1;
            in.nsrc = 2;
        }
    } else if (in.cvt_src == CVTMF_SRC_F32) {
        if (parse_operand(p, &in.src[0]) != 0 || expect_comma(p) != 0 ||
            parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (at_punct(p, ',')) {
            if (!in.cvt_scaled || dtype != CVTMF_S2F6X2) {
                perr(p, opline, "scale factor needs .scaled::n2::ue8m0 "
                                "and an .s2f6x2 destination");
                return -1;
            }
            if (parse_operand(p, &in.src[2]) != 0)
                return -1;
            in.nsrc = 3;
        }
    } else {
        perr(p, opline, "unsupported cvt source type for micro destination");
        return -1;
    }
    if (expect_punct(p, ';', "after cvt") != 0)
        return -1;
    return block_add_instr(fn, bi, &in, p, opline);
}
