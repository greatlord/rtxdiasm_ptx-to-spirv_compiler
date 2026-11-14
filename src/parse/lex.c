#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ptx.h"

void ptx_lex_init(ptx_lexer_t *lx, const char *src) {
    lx->src = src;
    lx->pos = 0;
    lx->line = 1;
    lx->has_saved = 0;
}

static char peek(ptx_lexer_t *lx) { return lx->src[lx->pos]; }

static void skip_space(ptx_lexer_t *lx) {
    for (;;) {
        char c = peek(lx);
        if (c == ' ' || c == '\t' || c == '\r') {
            lx->pos++;
        } else if (c == '\n') {
            lx->pos++;
            lx->line++;
        } else if (c == '/' && lx->src[lx->pos + 1] == '/') {
            while (peek(lx) && peek(lx) != '\n')
                lx->pos++;
        } else if (c == '/' && lx->src[lx->pos + 1] == '*') {
            lx->pos += 2;
            while (peek(lx) && !(peek(lx) == '*' && lx->src[lx->pos + 1] == '/')) {
                if (peek(lx) == '\n')
                    lx->line++;
                lx->pos++;
            }
            if (peek(lx))
                lx->pos += 2;
        } else if (c == '#') {
            // Preprocessor line: skip, harmless in hand-written PTX.
            while (peek(lx) && peek(lx) != '\n')
                lx->pos++;
        } else {
            break;
        }
    }
}

static int is_ident_start(char c) {
    return isalpha((unsigned char)c) || c == '_' || c == '%' || c == '$' || c == '.';
}

static int is_ident_char(char c) {
    return isalnum((unsigned char)c) || c == '_' || c == '%' || c == '$' || c == '.';
}

int ptx_next(ptx_lexer_t *lx, ptx_tok_t *out, char *err, size_t errlen) {
    if (lx->has_saved) {
        lx->has_saved = 0;
        *out = lx->saved;
        return 0;
    }
    skip_space(lx);
    memset(out, 0, sizeof(*out));
    out->line = lx->line;
    char c = peek(lx);
    if (c == '\0') {
        out->kind = T_EOF;
        return 0;
    }
    if (is_ident_start(c)) {
        size_t n = 0;
        // `::` inside an identifier is PTX's cache/space qualifier
        // syntax (`createpolicy.fractional.L2::evict_last`, cache
        // hints in opcodes). A single ':' stays punctuation so labels
        // (`done:`) keep working.
        while (n + 1 < sizeof(out->text) &&
               (is_ident_char(peek(lx)) ||
                (peek(lx) == ':' && lx->src[lx->pos + 1] == ':'))) {
            if (peek(lx) == ':') {
                out->text[n++] = lx->src[lx->pos++];
                if (n + 1 < sizeof(out->text))
                    out->text[n++] = lx->src[lx->pos++];
                continue;
            }
            out->text[n++] = lx->src[lx->pos++];
        }
        out->text[n] = '\0';
        if (is_ident_char(peek(lx))) {
            snprintf(err, errlen, "line %d: identifier too long", lx->line);
            return -1;
        }
        out->kind = T_IDENT;
        return 0;
    }
    if (isdigit((unsigned char)c)) {
        // 0x hex int, 0F hex-float bits, decimal int/float.
        size_t start = lx->pos;
        if (c == '0' && (lx->src[lx->pos + 1] == 'x' || lx->src[lx->pos + 1] == 'X')) {
            lx->pos += 2;
            uint64_t v = 0;
            int digits = 0;
            while (isxdigit((unsigned char)peek(lx))) {
                char d = lx->src[lx->pos++];
                v = v * 16 + (uint64_t)(isdigit((unsigned char)d)
                                        ? d - '0'
                                        : tolower((unsigned char)d) - 'a' + 10);
                digits++;
            }
            if (!digits) {
                snprintf(err, errlen, "line %d: bad hex constant", lx->line);
                return -1;
            }
            // PTX allows a C-style U suffix on integer constants (the
            // width comes from the instruction type, not the literal).
            if (peek(lx) == 'U' || peek(lx) == 'u')
                lx->pos++;
            out->kind = T_INT;
            out->imm = v;
            return 0;
        }
        // 0F<8 hex> = float bits, 0D<16 hex> = double bits. Scanned
        // eagerly: hex payloads contain A-F letters that the generic
        // number loop below would stop at (e.g. 0F3FC00000 for 1.5).
        if (c == '0' && (lx->src[lx->pos + 1] == 'F' ||
                         lx->src[lx->pos + 1] == 'f' ||
                         lx->src[lx->pos + 1] == 'D' ||
                         lx->src[lx->pos + 1] == 'd')) {
            char L = lx->src[lx->pos + 1];
            int want = (L == 'D' || L == 'd') ? 16 : 8;
            lx->pos += 2;
            uint64_t v = 0;
            int digits = 0;
            while (isxdigit((unsigned char)peek(lx)) && digits < want) {
                char d = lx->src[lx->pos++];
                v = v * 16 + (uint64_t)(isdigit((unsigned char)d)
                                        ? d - '0'
                                        : tolower((unsigned char)d) - 'a' + 10);
                digits++;
            }
            if (digits != want || isalnum((unsigned char)peek(lx))) {
                snprintf(err, errlen, "line %d: bad hex float constant",
                         lx->line);
                return -1;
            }
            size_t len = lx->pos - start;
            memcpy(out->text, lx->src + start, len);
            out->text[len] = '\0';
            out->kind = T_FLOAT;
            out->imm = v;
            return 0;
        }
        // 0H<4 hex> = half bits (f16 immediates, e.g. 0H3C00 for 1.0).
        if (c == '0' && (lx->src[lx->pos + 1] == 'H' ||
                         lx->src[lx->pos + 1] == 'h')) {
            lx->pos += 2;
            uint64_t v = 0;
            int digits = 0;
            while (isxdigit((unsigned char)peek(lx)) && digits < 4) {
                char d = lx->src[lx->pos++];
                v = v * 16 + (uint64_t)(isdigit((unsigned char)d)
                                        ? d - '0'
                                        : tolower((unsigned char)d) - 'a' + 10);
                digits++;
            }
            if (digits != 4 || isalnum((unsigned char)peek(lx))) {
                snprintf(err, errlen, "line %d: bad hex half constant",
                         lx->line);
                return -1;
            }
            size_t len = lx->pos - start;
            memcpy(out->text, lx->src + start, len);
            out->text[len] = '\0';
            out->kind = T_FLOAT;
            out->imm = v & 0xffffu;
            return 0;
        }
        while (isalnum((unsigned char)peek(lx)) || peek(lx) == '.' ||
               peek(lx) == '+' || peek(lx) == '-') {
            // Stop letter runs that are not float exponents: only e/E/p/P
            // continue; anything else ends the number.
            char d = peek(lx);
            if (isalpha((unsigned char)d) && d != 'e' && d != 'E') {
                if (!(lx->pos == start + 1 &&
                      (d == 'F' || d == 'f' || d == 'D' || d == 'd')))
                    break;
            }
            lx->pos++;
        }
        size_t len = lx->pos - start;
        if (len >= sizeof(out->text)) {
            snprintf(err, errlen, "line %d: numeric constant too long", lx->line);
            return -1;
        }
        memcpy(out->text, lx->src + start, len);
        out->text[len] = '\0';
        // NOTE: 0F/0D/0H hex floats are intercepted by the eager
        // scans above and never reach this generic loop.
        char *end = NULL;
        int is_flt = strchr(out->text, '.') || strchr(out->text, 'e') ||
                     strchr(out->text, 'E');
        if (is_flt) {
            float f = strtof(out->text, &end);
            if (!end || *end) {
                snprintf(err, errlen, "line %d: bad float '%s'", lx->line, out->text);
                return -1;
            }
            uint32_t bits;
            memcpy(&bits, &f, 4);
            out->kind = T_FLOAT;
            out->imm = bits;
        } else {
            // Full u64 range (strtoull): PTX immediates are bit
            // patterns, so 9223372036854775808 (2^63, i.e. INT64_MIN
            // negated, as numba emits) parses fine; unary minus wraps.
            errno = 0;
            unsigned long long v = strtoull(out->text, &end, 10);
            if (!end || *end || errno == ERANGE) {
                snprintf(err, errlen, "line %d: bad integer '%s'", lx->line,
                         out->text);
                return -1;
            }
            out->kind = T_INT;
            out->imm = (uint64_t)v;
        }
        // Trailing U suffix on decimal integers (same as hex above).
        if (out->kind == T_INT && (peek(lx) == 'U' || peek(lx) == 'u'))
            lx->pos++;
        return 0;
    }
    if (c == '"') {
        // Quoted string (only meaningful inside .pragma, which the
        // parser skips wholesale). Lexed as a token so other uses
        // fail with a precise message instead of a confusing one.
        lx->pos++;
        size_t n = 0;
        while (peek(lx) && peek(lx) != '"' && peek(lx) != '\n' &&
               n + 1 < sizeof(out->text))
            out->text[n++] = lx->src[lx->pos++];
        out->text[n] = '\0';
        if (peek(lx) != '"') {
            snprintf(err, errlen, "line %d: unterminated string literal",
                     lx->line);
            return -1;
        }
        lx->pos++;
        out->kind = T_STR;
        return 0;
    }
    out->kind = T_PUNCT;
    out->punct = c;
    out->text[0] = c;
    lx->pos++;
    return 0;
}

void ptx_unget(ptx_lexer_t *lx, const ptx_tok_t *tok) {
    lx->saved = *tok;
    lx->has_saved = 1;
}
