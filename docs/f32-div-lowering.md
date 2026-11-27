# f32 division lowering

`div.full.f32` / `div.rn.f32` (non-approximate, correctly rounded) are
lowered to SPIR-V `OpFDiv` by default.

## Why

The previous lowering used `f32_directed_div_exact`, a fully-unrolled
28-iteration 64-bit integer software division. It is bit-exact, but the
resulting shader is enormous and the NVIDIA Vulkan driver miscompiles it
into a **non-terminating dispatch** for some inputs. This was observed
with PhysX 2.8's `KernelFluidCalcHashPacked`: the launch never signals its
timeline semaphore (30 s timeout) even though every other kernel in the
same stream completes.

Lowering the same PTX to `OpFDiv` (or `div.approx`) makes the launch
return `SYNC_OK`, and SPIR-V `OpFDiv` is IEEE-754 correctly rounded, so
the result is still correct on conformant drivers.

## Behaviour

* Default: f32 `div.full` / `div.rn` -> `OpFDiv`.
* `ACUDA_SOFT_F32_DIV=1`: restore the old exact integer sequence (useful
  for drivers where `OpFDiv` is not correctly rounded, or for bisecting).
* f64 division is unchanged: it always uses `f64_directed_div_exact`
  unless `ACUDA_EMU_F64` selects the emulated path.

`div.approx` is unaffected (it already used `OpFDiv`).
