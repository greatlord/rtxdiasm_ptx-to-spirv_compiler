# Pinned CUDA/PTX baseline: CUDA 13.4

Do not forget: the PyTorch test env is **CUDA 13.4** (`torch 2.15.0+cu134`,
`torch.version.cuda == 13.4`, see `python-test/setup-example run with
zluda.txt`). All compatibility claims are measured against this.

- CUDA toolkit: **13.4** (venv
  `/home/magnusolsen/git/aicode-ptx/virtual python/ven-cu134`, cu134
  wheels; `libcudart.so.13`).
- Target PTX ISA: **9.4** (ships with CUDA 13.4; full table in
  `new-todo-PTX.md`).
- Observed PTX in-tree: the checked-in NVRTC fixture is `.version 8.5` /
  `.target sm_80` (`tests/ptx/vecadd_nvrtc.ptx`). CUDA 13.4's NVRTC emits
  `.version 9.x`; both are accepted (versions are recorded, not gated).
  Numba-harvested files are `.version 8.8`.
- Our parser: `.version` must name a released PTX ISA version (1.0–1.5,
  2.0–2.3, 3.0–3.2, 4.0–4.3, 5.0, 6.0–6.5, 7.0–7.8, 8.0–8.8,
  9.0–9.4); the default ceiling is 9.4 (`ACUDA_PTX` overrides).
  `.target sm_NN[a|f]` is checked against the release that introduced
  it: sm_10/11 (1.0), sm_12/13 (1.2), sm_20 (2.0), sm_30 (3.0),
  sm_35 (3.1), sm_32/50 (4.0), sm_37/52 (4.1), sm_53 (4.2),
  sm_60–62 (5.0), sm_70 (6.0), sm_72 (6.1), sm_75 (6.3), sm_80 (7.0),
  sm_86 (7.1), sm_87 (7.4), sm_89/90 (7.8), sm_90a (8.0), sm_100/101
  (8.6), sm_120 (8.7), sm_100f/101f/103/121 (8.8), sm_88/110 (9.0),
  sm_107 (9.4).  Unknown names and too-new pairs are clean errors.
- Pinned target: **sm_80** (Ampere baseline). Lowest common sm_8x —
  every CUDA 11+ toolchain emits it, it JITs everywhere sm_80+, and it
  avoids sm_90-only features (clusters, TMA, wgmma) we reject. Matches
  our reported cc 8.0 and all five `tests/ptx/*.ptx` (all `.target
  sm_80`). Harvest new PTX with sm_80 (e.g. numba `cc=(8, 0)`).
- Decided (2026-11-02, CUDA 13.4 switch): `cuDriverGetVersion` and
  `cudaRuntimeGetVersion` report **13040 (13.4)**. The CUDA 13.4 runtime
  requires a driver that reports the same version before it initializes,
  and PyTorch 2.15+cu134 checks that pair. The runtime library is now
  `libcudart.so.13.40` (`SOVERSION 13`). Pinned by `test_driver`
  (`ver == 13040`) and `test_cudart` (`rt == drv == 13040`).
- CUDA 13.4 runtime initialization also queries `cuGetExportTable` for
  the `CUDART_INTERFACE`, tools/TLS, context-storage, context-check,
  and integrity tables. Those are implemented in `src/driver/driver.c`
  (including the integrity checksum) so `cudaGetDeviceCount` and
  `cudaGetDeviceProperties` succeed; PyTorch's own kernels are still
  SASS-only and cannot be launched.
- Implemented: `ACUDA_DUMP_SPV_DIR` (per-kernel dumps), `ACUDA_PTX` /
  `ACUDA_SM` ceilings (clean `NOT_SUPPORTED` above cap; `.version`
  defaults to a 9.4 ceiling, unset/garbage = 9.4; pinned by `test_ptx`), `ACUDA_EMU_F64=1` demotion (f32-level
  approximate, loud warning, pinned by `python-test/nb_f64_emu.py`).
  No JSON config file exists — env vars are the mechanism.
