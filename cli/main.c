/* rtxdiasm_ptx-to-spirv_compiler
 * Usage:
 *   rtxdiasm_ptx-to-spirv_compiler <input.ptx> <output.blob> [entry_name] [--cap.<name>=<value> ...]
 *
 * Compiles PTX to a serialized kernel blob (AKB1). All compiler capabilities
 * must be provided explicitly by the caller; this tool performs no device
 * autodetection.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "acuda/compiler.h"

static unsigned char *read_all(const char *path, size_t *sz) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0) { fclose(f); return NULL; }
    rewind(f);
    unsigned char *b = (unsigned char *)malloc((size_t)n + 1);
    if (!b) { fclose(f); return NULL; }
    if (fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); fclose(f); return NULL; }
    b[n] = 0;
    fclose(f);
    if (sz) *sz = (size_t)n;
    return b;
}

static int parse_long(const char *s, long *out) {
    if (!s || !*s) return 0;
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 0);
    if (errno != 0 || end == s || *end != '\0') return 0;
    *out = v;
    return 1;
}

static int parse_ulong(const char *s, unsigned long *out) {
    if (!s || !*s) return 0;
    char *end = NULL;
    errno = 0;
    unsigned long v = strtoul(s, &end, 0);
    if (errno != 0 || end == s || *end != '\0') return 0;
    *out = v;
    return 1;
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "usage: %s <input.ptx> <output.blob> [entry_name] [--cap.<name>=<value> ...]\n"
        "\nCapabilities (all explicit; defaults match acuda_compiler_caps_init):\n"
        "  shader_fma shader_fma_f64 soft_fma f16_native coopmat_f16 coopmat_bf16\n"
        "  mixed_fdot dot_product dot_signed dot_unsigned dot_mixed zero_init_wg\n"
        "  ssbo8 ssbo16 fc32 fc64 vmm bda atomic_float int64_image surf_alias_ok\n",
        argv0);
}

static int apply_cap(acuda_compiler_caps_t *caps, const char *name, const char *val) {
    long iv = 0; unsigned long uv = 0;
    if (strcmp(name, "shader_fma") == 0) { if (!parse_long(val,&iv)) return 0; acuda_compiler_set_shader_fma(caps,(int)iv); return 1; }
    if (strcmp(name, "shader_fma_f64") == 0) { if (!parse_long(val,&iv)) return 0; acuda_compiler_set_shader_fma_f64(caps,(int)iv); return 1; }
    if (strcmp(name, "soft_fma") == 0) { if (!parse_long(val,&iv)) return 0; acuda_compiler_set_soft_fma(caps,(int)iv); return 1; }
    if (strcmp(name, "f16_native") == 0) { if (!parse_long(val,&iv)) return 0; acuda_compiler_set_f16_native(caps,(int)iv); return 1; }
    if (strcmp(name, "coopmat_f16") == 0) { if (!parse_long(val,&iv)) return 0; acuda_compiler_set_cooperative_matrix(caps,(int)iv,caps->coopmat_bf16); return 1; }
    if (strcmp(name, "coopmat_bf16") == 0) { if (!parse_long(val,&iv)) return 0; acuda_compiler_set_cooperative_matrix(caps,caps->coopmat_f16,(int)iv); return 1; }
    if (strcmp(name, "mixed_fdot") == 0) { if (!parse_long(val,&iv)) return 0; acuda_compiler_set_mixed_fdot(caps,(int)iv); return 1; }
    if (strcmp(name, "dot_product") == 0) { if (!parse_long(val,&iv)) return 0; acuda_compiler_set_dot_product(caps,(int)iv); return 1; }
    if (strcmp(name, "dot_signed") == 0) { if (!parse_long(val,&iv)) return 0; acuda_compiler_set_dot_product_forms(caps,(int)iv,caps->dot_unsigned,caps->dot_mixed); return 1; }
    if (strcmp(name, "dot_unsigned") == 0) { if (!parse_long(val,&iv)) return 0; acuda_compiler_set_dot_product_forms(caps,caps->dot_signed,(int)iv,caps->dot_mixed); return 1; }
    if (strcmp(name, "dot_mixed") == 0) { if (!parse_long(val,&iv)) return 0; acuda_compiler_set_dot_product_forms(caps,caps->dot_signed,caps->dot_unsigned,(int)iv); return 1; }
    if (strcmp(name, "zero_init_wg") == 0) { if (!parse_long(val,&iv)) return 0; acuda_compiler_set_zero_init_workgroup(caps,(int)iv); return 1; }
    if (strcmp(name, "ssbo8") == 0) { if (!parse_long(val,&iv)) return 0; acuda_compiler_set_ssbo8(caps,(int)iv); return 1; }
    if (strcmp(name, "ssbo16") == 0) { if (!parse_long(val,&iv)) return 0; acuda_compiler_set_ssbo16(caps,(int)iv); return 1; }
    if (strcmp(name, "fc32") == 0) { if (!parse_long(val,&iv)) return 0; acuda_compiler_set_float_controls(caps,(int)iv,caps->fc64); return 1; }
    if (strcmp(name, "fc64") == 0) { if (!parse_long(val,&iv)) return 0; acuda_compiler_set_float_controls(caps,caps->fc32,(int)iv); return 1; }
    if (strcmp(name, "vmm") == 0) { if (!parse_long(val,&iv)) return 0; acuda_compiler_set_vulkan_memory_model(caps,(int)iv); return 1; }
    if (strcmp(name, "bda") == 0) { if (!parse_long(val,&iv)) return 0; acuda_compiler_set_bda(caps,(int)iv); return 1; }
    if (strcmp(name, "atomic_float") == 0) { if (!parse_ulong(val,&uv)) return 0; acuda_compiler_set_atomic_float(caps,(unsigned)uv); return 1; }
    if (strcmp(name, "int64_image") == 0) { if (!parse_long(val,&iv)) return 0; acuda_compiler_set_int64_image(caps,(int)iv); return 1; }
    if (strcmp(name, "surf_alias_ok") == 0) { if (!parse_long(val,&iv)) return 0; acuda_compiler_set_surface_alias(caps,(int)iv); return 1; }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 1;
    }
    const char *in = argv[1];
    const char *out = argv[2];
    const char *entry = NULL;
    acuda_compiler_caps_t caps;
    acuda_compiler_caps_init(&caps);

    int argi = 3;
    if (argi < argc && strncmp(argv[argi], "--", 2) != 0) {
        if (strcmp(argv[argi], "-") != 0)
            entry = argv[argi];
        argi++;
    }
    for (; argi < argc; argi++) {
        const char *a = argv[argi];
        if (strncmp(a, "--entry=", 8) == 0) {
            entry = a + 8;
            if (entry[0] == '\0') entry = NULL;
            continue;
        }
        if (strncmp(a, "--cap.", 6) == 0) {
            const char *eq = strchr(a + 6, '=');
            if (!eq) {
                fprintf(stderr, "invalid capability argument: %s\n", a);
                return 1;
            }
            char name[64];
            size_t nlen = (size_t)(eq - (a + 6));
            if (nlen == 0 || nlen >= sizeof(name)) {
                fprintf(stderr, "invalid capability name: %s\n", a);
                return 1;
            }
            memcpy(name, a + 6, nlen);
            name[nlen] = '\0';
            if (!apply_cap(&caps, name, eq + 1)) {
                fprintf(stderr, "unknown or invalid capability: %s\n", a);
                return 1;
            }
            continue;
        }
        if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
            usage(argv[0]);
            return 0;
        }
        fprintf(stderr, "unexpected argument: %s\n", a);
        usage(argv[0]);
        return 1;
    }

    size_t sz = 0;
    unsigned char *ptx = read_all(in, &sz);
    if (!ptx) {
        fprintf(stderr, "rtxdiasm_ptx-to-spirv_compiler: cannot read %s\n", in);
        return 2;
    }

    acuda_kernel_t *k = NULL;
    char err[1024];
    err[0] = '\0';
    acuda_status_t st = acuda_compile_ptx((const char *)ptx, entry, &k, err, sizeof(err), &caps);
    free(ptx);
    if (st != ACUDA_SUCCESS || !k) {
        fprintf(stderr, "rtxdiasm_ptx-to-spirv_compiler: %s\n", err[0] ? err : "compile failed");
        return 6;
    }

    uint8_t *blob = NULL;
    size_t bsz = 0;
    st = acuda_kernel_serialize(k, &blob, &bsz);
    acuda_kernel_free(k);
    if (st != ACUDA_SUCCESS || !blob) {
        fprintf(stderr, "rtxdiasm_ptx-to-spirv_compiler: serialize failed\n");
        return 6;
    }

    FILE *fo = fopen(out, "wb");
    if (!fo || fwrite(blob, 1, bsz, fo) != bsz) {
        if (fo) fclose(fo);
        free(blob);
        fprintf(stderr, "rtxdiasm_ptx-to-spirv_compiler: cannot write %s\n", out);
        return 3;
    }
    fclose(fo);
    free(blob);
    return 0;
}
