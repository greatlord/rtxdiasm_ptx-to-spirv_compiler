# Harvesting PTX from Numba (reference workflow, 2026-10-14)

Yes: `@cuda.jit` Python can be turned into PTX text on this AMD machine
with no NVIDIA GPU, and that PTX is future input for the ACuda compiler.
Proven with `tests/ptx/vecadd_numba.ptx` (`.version 8.8`, `.target sm_80`).

## Recipe (scratch env in /tmp, project venvs untouched)

```bash
python3.12 -m venv /tmp/opencode/numba-test
/tmp/opencode/numba-test/bin/pip install numpy numba \
  nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12
export NUMBA_CUDA_DRIVER="/home/magnusolsen/git/aicode-ptx/no git/zluda-linux-86ec0a93/zluda-bin/libcuda.so"
export CUDA_HOME=".../numba-test/lib/python3.12/site-packages/nvidia/cuda_nvcc"
export LD_LIBRARY_PATH=".../nvidia/cuda_runtime/lib:.../cuda_nvcc/nvvm/lib64:.../zluda-bin/"
```

Then bypass `dispatcher.compile()` (it calls `bind()` → `cuLinkCreate`,
which ZLUDA rejects with `CUDA_ERROR_NOT_SUPPORTED`) and use:

```python
from numba.cuda.compiler import compile_cuda
cres = compile_cuda(py_func, void, sig_tuple, cc=(8, 5))
llvm_text = cres.library.get_llvm_str()
ptx_text = cres.library.get_asm_str(cc=(8, 5))
```

ZLUDA is used ONLY as a driver shim for device query (reports
`AMD Radeon RX 6800 XT [ZLUDA]`, CC 8.5). llvmlite ships its own LLVM,
NVVM/libdevice come from the pip wheels — no system LLVM needed.

## Why this matters for ACuda

Numba/PyTorch/CuPy only need two things from us: the CUDA **driver API**
(Phase 5: context, module load, launch, memcpy) plus a **PTX consumer**
(Phase 4/6 compiler). Their Python→PTX frontends already run on AMD
today. ACuda never needs to reimplement Numba.

## Gap analysis: numba PTX vs Phase 4 subset

`vecadd_numba.ptx` is deliberately OUTSIDE the Phase 4 subset — keep it
as a Phase 6 target / negative test. It needs, at minimum:

- `.func` (+ `st.param` retval) in addition to `.entry`
- generic (unqualified) `ld.f64`/`st.f64` — no `.global` state space
- `.b32`/`.b64`/`.s32`/`.s64` types, `mul.lo.s64`, `cvt.s64.s32`
- `bra.uni` + multi-block diamond CFG (`$L__BB0_1/2/3`), not one guard
- 22 unpacked array-descriptor `.param .b64`s instead of raw pointers
- 64-bit float ops (`add.f64`)

None of this changes Phase 4: hand-written minimal kernels first,
numba-shaped PTX later.
