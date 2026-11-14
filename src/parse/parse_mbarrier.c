// elect.sync and mbarrier parsing.
#include <stdio.h>
#include <string.h>
#include "ptx.h"
#include "parse/parse_internal.h"

#define MB_MAX_PARTS 12

static int mb_split(const char *op, char parts[MB_MAX_PARTS][32],
                    int *np) {
    *np = 0;
    const char *s = op;
    while (*s) {
        if (*np >= MB_MAX_PARTS)
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

static int mb_has(char parts[MB_MAX_PARTS][32], int np, const char *s) {
    for (int i = 0; i < np; i++)
        if (strcmp(parts[i], s) == 0)
            return 1;
    return 0;
}

static int mb_addr(parser_t *p, ptx_opd_t *o, int line) {
    if (parse_operand(p, o) != 0)
        return -1;
    if (o->kind != OPD_ADDR) {
        perr(p, line, "mbarrier operand must be [address]");
        return -1;
    }
    return 0;
}

static int parse_elect(parser_t *p, ptx_func_t *fn, size_t bi,
                       const char *pred, int pred_neg, int opline) {
    ptx_instr_t in;
    memset(&in, 0, sizeof(in));
    in.line = opline;
    in.op = POP_ELECT;
    if (copy_name(in.pred, sizeof(in.pred), pred[0] ? pred : "", p,
                  opline) != 0)
        return -1;
    in.pred_neg = pred_neg;
    if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0)
        return -1;
    if (at_punct(p, '|')) {
        if (parse_dst(p, in.dst2, sizeof(in.dst2), 1) != 0)
            return -1;
    }
    if (expect_comma(p) != 0 || parse_operand(p, &in.src[0]) != 0)
        return -1;
    in.nsrc = 1;
    if (expect_punct(p, ';', "after elect.sync") != 0)
        return -1;
    return block_add_instr(fn, bi, &in, p, opline);
}

static int parse_mbarrier(parser_t *p, ptx_func_t *fn, size_t bi,
                          const char *pred, int pred_neg,
                          const char *opname, int opline) {
    char parts[MB_MAX_PARTS][32];
    int np = 0;
    memset(parts, 0, sizeof(parts));
    if (mb_split(opname, parts, &np) != 0) {
        perr(p, opline, "bad mbarrier opcode");
        return -1;
    }
    ptx_instr_t in;
    memset(&in, 0, sizeof(in));
    in.line = opline;
    in.op = POP_MBARRIER;
    if (copy_name(in.pred, sizeof(in.pred), pred[0] ? pred : "", p,
                  opline) != 0)
        return -1;
    in.pred_neg = pred_neg;
    in.mb_layout = mb_has(parts, np, "layout::v1") ? 1 : 0;
    if (strcmp(parts[0], "cp") == 0) {
        // cp.async.mbarrier.arrive[.noinc]{.shared...}.b64 [addr]
        in.mb_op = MBAR_CP_ARRIVE;
        in.mb_nocomplete = mb_has(parts, np, "noinc") ? 1 : 0;
        if (mb_addr(p, &in.src[0], opline) != 0)
            return -1;
        in.nsrc = 1;
    } else if (strcmp(parts[1], "init") == 0) {
        in.mb_op = MBAR_INIT;
        if (mb_addr(p, &in.src[0], opline) != 0 ||
            expect_comma(p) != 0 ||
            parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
    } else if (strcmp(parts[1], "inval") == 0) {
        in.mb_op = MBAR_INVAL;
        if (mb_addr(p, &in.src[0], opline) != 0)
            return -1;
        in.nsrc = 1;
    } else if (strcmp(parts[1], "expect_tx") == 0) {
        in.mb_op = MBAR_EXPECT_TX;
        if (mb_addr(p, &in.src[0], opline) != 0 ||
            expect_comma(p) != 0 ||
            parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
    } else if (strcmp(parts[1], "complete_tx") == 0) {
        in.mb_op = MBAR_COMPLETE_TX;
        if (mb_addr(p, &in.src[0], opline) != 0 ||
            expect_comma(p) != 0 ||
            parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
    } else if (strcmp(parts[1], "arrive") == 0 ||
               strcmp(parts[1], "arrive_drop") == 0) {
        in.mb_op = strcmp(parts[1], "arrive_drop") == 0
                       ? MBAR_ARRIVE_DROP : MBAR_ARRIVE;
        in.mb_nocomplete = mb_has(parts, np, "noComplete") ? 1 : 0;
        in.mb_expect = mb_has(parts, np, "expect_tx") ? 1 : 0;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0 ||
            expect_comma(p) != 0 ||
            mb_addr(p, &in.src[0], opline) != 0)
            return -1;
        in.nsrc = 1;
        if (at_punct(p, ',')) {
            if (parse_operand(p, &in.src[1]) != 0)
                return -1;
            in.nsrc = 2;
        }
    } else if (strcmp(parts[1], "test_wait") == 0 ||
               strcmp(parts[1], "try_wait") == 0) {
        in.mb_op = strcmp(parts[1], "test_wait") == 0
                       ? MBAR_TEST_WAIT : MBAR_TRY_WAIT;
        in.mb_parity = mb_has(parts, np, "parity") ? 1 : 0;
        if (parse_dst(p, in.dst, sizeof(in.dst), 1) != 0 ||
            expect_comma(p) != 0 ||
            mb_addr(p, &in.src[0], opline) != 0 ||
            expect_comma(p) != 0 ||
            parse_operand(p, &in.src[1]) != 0)
            return -1;
        in.nsrc = 2;
        if (at_punct(p, ',')) {
            if (parse_operand(p, &in.src[2]) != 0)
                return -1;
            in.nsrc = 3;
        }
    } else if (strcmp(parts[1], "pending_count") == 0) {
        in.mb_op = MBAR_PENDING;
        if (parse_dst(p, in.dst, sizeof(in.dst), 0) != 0 ||
            expect_comma(p) != 0 ||
            parse_operand(p, &in.src[0]) != 0)
            return -1;
        in.nsrc = 1;
    } else if (strcmp(parts[1], "check_layout") == 0) {
        in.mb_op = MBAR_CHECK_LAYOUT;
        if (parse_dst(p, in.dst, sizeof(in.dst), 1) != 0 ||
            expect_comma(p) != 0 ||
            mb_addr(p, &in.src[0], opline) != 0)
            return -1;
        in.nsrc = 1;
    } else {
        perr(p, opline, "unsupported mbarrier operation");
        return -1;
    }
    if (expect_punct(p, ';', "after mbarrier") != 0)
        return -1;
    return block_add_instr(fn, bi, &in, p, opline);
}

int parse_mbarrier_instr(parser_t *p, ptx_func_t *fn, size_t bi,
                         const char *pred, int pred_neg,
                         const char *opname, int opline) {
    if (strncmp(opname, "elect.sync", 10) == 0)
        return parse_elect(p, fn, bi, pred, pred_neg, opline);
    return parse_mbarrier(p, fn, bi, pred, pred_neg, opname, opline);
}
