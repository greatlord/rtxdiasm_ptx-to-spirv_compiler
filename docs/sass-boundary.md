# SASS boundary: no SASS→PTX, and how we prove our PTX is genuine

## There is no SASS→PTX (verified, not assumed)

"SASS to PTX" does not exist — not in NVIDIA's toolchain (`cuobjdump`
disassembles SASS→SASS text, never back to PTX) and not in ZLUDA either:
their `ptxas` crate consumes PTX *text* (`--gpu-name`, mimics NVIDIA's
CLI) and their test corpus is PTX files; no disassembler or decompiler
exists in their source. Recovering structured PTX ( let alone correct
kernels) from per-arch machine code for sm_50…sm_90 would be a
decompiler research project, and mistranslated SASS would violate our
core rule. So: SASS-only binaries (all of torch aten: 2931 cubins,
zero PTX members) stay unloadable with a clean `INVALID_IMAGE`, and
`cuModuleLoadFatBinary` extracts only validated PTX members.

## The reference loop we run instead (PTX → ptxas → cubin → SASS text)

`python-test/ptx_sass_ref.py` assembles every `tests/ptx/*.ptx` with
real NVIDIA `ptxas` (sm_80, from the `nvidia-cuda-nvcc` pip wheel) and
dumps SASS with `cuobjdump` (from the triton wheel). Ground truth lives
in `tests/sass/*.sm_80.sass.txt`. This already caught two divergences
where we were more lenient than NVIDIA (both fixed):
- `mad.lo` with a special-register source is rejected (`mov` to a plain
  register first); our parser enforces the same error now.
- `.reg %r<4>` is `%r0`–`%r3` (0-based, like ours — the shared
  misunderstanding is documented, not the semantics).

Notes: `.func`-only files (`nb_f32`, `vecadd_numba`) assemble fine but
contain no entry points, so their SASS dumps are header-only — expected,
not a failure. Tools are env-overridable (`ACUDA_PTXAS`,
`ACUDA_CUOBJDUMP`) and the script skips cleanly when absent.
