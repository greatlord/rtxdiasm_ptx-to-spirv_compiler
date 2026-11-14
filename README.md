# rtxdiasm_ptx-to-spirv_compiler

Standalone PTX-to-SPIR-V compiler extracted from ACuda/rtxdisam.

## Outputs

- Static library: `rtxdiasm_ptx_to_spirv_compiler`
- Shared/dynamic library: `rtxdiasm_ptx_to_spirv_compiler`
- Executable: `rtxdiasm_ptx-to-spirv_compiler`

## CLI

```text
rtxdiasm_ptx-to-spirv_compiler <input.ptx> <output.blob> [entry_name] [--cap.<name>=<value> ...]
```

- Produces a serialized kernel blob (`AKB1`).
- **No device autodetection is performed.** All compiler capabilities must be supplied explicitly by the caller via `--cap.*` arguments (defaults match `acuda_compiler_caps_init()`).

Supported capability arguments:
`shader_fma`, `shader_fma_f64`, `soft_fma`, `f16_native`, `coopmat_f16`, `coopmat_bf16`,
`mixed_fdot`, `dot_product`, `dot_signed`, `dot_unsigned`, `dot_mixed`, `zero_init_wg`,
`ssbo8`, `ssbo16`, `fc32`, `fc64`, `vmm`, `bda`, `atomic_float`, `int64_image`, `surf_alias_ok`.

## Build

```sh
cmake -S . -B build
cmake --build build --config Release
```

## License

MIT (see `LICENSE`).
