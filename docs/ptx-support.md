# PTX support overview



This is the reader-facing summary of what ACuda's PTX → SPIR-V compiler

supports and what it deliberately rejects. The exhaustive, comma-level

matrix lives in [`supported-isa.md`](supported-isa.md) and the

chapter-by-chapter audit in [`isa-coverage.md`](isa-coverage.md); this page explains

the shape of the subset, the runtime model, and the known divergences.



ACuda is a **PTX front end + SPIR-V backend**, not a CUDA compiler:

cubin/SASS is out of scope (see [`sass-boundary.md`](sass-boundary.md)).

Kernels execute through Vulkan compute, so the subset is deliberately

limited to what maps cleanly onto SPIR-V 1.6 / Vulkan 1.3. The runtime

requires a Vulkan 1.3 device with `maintenance4` (`LocalSizeId`) and

`synchronization2` (the barrier/copy/format 1.3 APIs are used

directly); optional features such as `shaderIntegerDotProduct` are

still probed at init and fall back cleanly.



## At a glance



| Area | Support |

|---|---|

| Input | `.version` released values up to 9.4, `.target sm_NN[a\|f]` validated against that release, `.address_size 64` required |

| Entries | `.entry`/`.visible .entry`, `.func` callees (inlined, including callee `.local` frames and mixed `.reg`/`.param` multi-return signatures), `.extern` prototype declarations |

| Kernel ABI | pointer params → storage-buffer bindings, scalar words → push constants, up to 64 params, 1D grid |

| Workgroup | any `blockDim.x` 1–1024, specialized per size (`LocalSizeId`) |

| Arithmetic | int 8/16/32/64 (8-bit ALU is an extension), f16/f16x2/bf16x2, f32, f64 (native where available) |

| Memory | `.param`, `.shared` (static + extern), `.local`, `.global` images, `.const`, generic `ld/st`; native byte SSBO access when `storageBuffer8BitAccess` is available; R64ui surfaces (`SPV_EXT_shader_image_int64`) for 64-bit `suld`/`sust`/`sured` when `shaderImageInt64Atomics` is available |

| Atomics | `.add/.inc/.dec/.and/.or/.xor/.exch/.min/.max/.cas`; 32-bit, 64-bit integer, `.add.f32`; global/shared/generic; `red.async`, `multimem.red.async` (single-device) |

| Warp/CTA | `shfl`/`shfl.sync`, `vote` (`all/any/uni/ballot`) / `vote.sync`, `activemask`, `redux.sync`, `match.sync`, `elect.sync`, counted `bar.sync`/`bar.arrive`, `bar.red`, `membar`, `mbarrier`, `red.async`; warp instructions require 32-lane subgroups (requested via `subgroupSizeControl`, 32/64 default accepted) |

| Matrix | `mma.sync` m16n8k16 f16/bf16 (f32 or f16 accumulate), m16n8k8 f16/bf16 (f32 or f16 accumulate), m16n8k4 tf32, m8n8k4 f16/f64, m16n8k4/k8/k16 f64 (sm_90), m8n8k16/m16n8k16/m16n8k32 s8/u8 (all A/B sign combinations); sparse `mma.sp` m16n8k16/m16n8k32 f16/bf16; `wmma` m16n16k16 f16/bf16->f32 and f16->f16, plus m8n32k16/m32n8k16 f16->f32/f16 (load/store/mma); `ldmatrix.sync` m8n8 x1/x2/x4 (+`.trans`) |

| Control flow | straight line, forward guards, diamonds, single-header loops, sequential multi-region flow |

| Coverage | ZLUDA corpus **196/198**; Windows/RTX3070 kit **36/36**; ctest **14/14** |



Anything outside the documented subset is rejected with a

`line N: ...` diagnostic and `ACUDA_ERROR_NOT_SUPPORTED`. The compiler

does not silently mistranslate unsupported forms.



## Compilation and launch pipeline



```text

PTX text

  → parser (attributes, declarations, instructions)

  → SPIR-V 1.6 module (Logical addressing, GLSL450, compute entry point)

  → Vulkan compute pipeline (specialized for the launched workgroup size)

  → device execution

```



There is no PTX-level linking step. Direct calls to `.func` definitions

are inlined at compile time; supported indirect calls and unresolved

`.extern` functions lower to split-dispatch host calls. The ACuda API

exposes `acuda_host_call_register()`; the CUDA driver shim installs an

identity stub for any host call that has no registered callback. A PTX

module must contain an `.entry` (or a `.func` that can be selected

explicitly); a declaration-only module such as an isolated

`.extern .global` has no launchable kernel and is rejected.



### Kernel ABI



- Pointer parameters become storage-buffer descriptor bindings, in

  declaration order.

- `.u32`/`.f32` parameters become push-constant words.

- 64-bit scalar parameters take two push words.

- By-value scalar structs (`.u64xN`, up to 256 words) fit in push

  constants when small. Structs larger than 128 bytes are passed through

  a descriptor-bound storage buffer instead, so kernels such as

  `TensorIterator`-shaped reductions no longer exceed the device's

  `maxPushConstantsSize`. The driver uploads exactly `N * 8` argument

  bytes and `ld.param` reads them by byte offset. Direct

  `acuda_launch_kernel*` callers must supply that storage buffer in the

  descriptor list (`acuda_kernel_info.param_ssbo` marks the slot).

- Unused parameters consume no descriptor/push slot; the driver skips

  them (the analyzer classifies each parameter as pointer, scalar, or

  unused).

- `blockDim.x` is specialized per launch; the default metadata is 256.

- The backend holds a global pointer as **(descriptor, byte offset)**,

  not a raw device address. This matters for pointer arithmetic and for

  `stateful_neg_offset`-style idioms (see `supported-isa.md` §4b).

- Addressable parameters (`mov.u64 %rd, param` + `ld.param.T [%rd]`) use

  a symbolic parameter address; the value still comes from push

  constants/descriptors.



## Instruction families



The table below is a map; exact type spellings, corner cases and pinned

tests are in [`supported-isa.md`](supported-isa.md) §6.



| Family | Highlights |

|---|---|

| Moves | `mov` over integer/float/predicate/vector forms; brace-vector moves; `mov.v8` with eight brace-listed registers |

| Integer arithmetic | `add/sub` (incl. packed `.u16x2/.s16x2/.u8x4/.s8x4` and saturating forms), `mul.lo/wide/hi`, `mad.lo/hi/wide`, `clmad`, `mul24`, `mad24`, `addc/subc/madc` carry chains, `sad`, `dp4a`, `dp2a`, `prmt`, `bfe/bfi/bmsk/bfind`, `fns`, `szext`, `popc/clz/brev`, `abs`, `neg`, `min/max` (incl. packed and `.relu`), shifts (`shl/shr/shr.s`, `shf`, `vshr`), `and/or/xor/not`, `selp` |

| Float arithmetic | `add/sub/mul` with `.rnd`/`.ftz`/`.sat`, `div.approx/full/rn` and `.ftz`, `fma`/`mad` with `.ftz`/`.sat`, `min/max` (`.ftz`/`.NaN`), `neg`, `abs`, `copysign` f32/f64, `testp` properties, `rcp` (incl. exact rn and directed f32/f64), `sqrt`/`rsqrt`, transcendentals via GLSL.std.450, directed `add/sub/mul/fma/mad/div` over f32 and f64 (integer software arithmetic); half-precision add/sub/mul/fma/neg/abs/min/max/tanh/ex2 over `.f16`/`.f16x2`/`.bf16`/`.bf16x2` (including fma `.relu`), packed `.f32x2` add/sub/mul/fma; `.f16`/`.f16x2` division (extension) |

| Conversion | integer width/sign matrix, f16/f32/f64 conversions, directed int→float (`cvt.{rn,rz,rm,rp}.{f16,f32,f64}.{u,s}{8,16,32,64}`), `cvt.{rn,rna}.tf32.f32`, `.satfinite` half narrowing, round-to-integer (`cvt.rni/rzi/rmi/rpi`), saturating and packed conversions (`cvt.pack` u16/s16 and u8/s8/u4/s4/u2/s2 lanes, `cvt.sat`), FP8 x2 (`e4m3/e5m2`), f16x2/bf16x2 packs, `cvt.relu` |

| Comparison | `setp` over integers/floats/predicates, ordered and unordered forms, boolean-combining forms, dual destinations, `set` masks |

| Memory | `ld/st` in all supported spaces; vector `.v2/.v4/.v8` widths (including `.v2.u64` and `.b128`); sub-word lanes (native byte SSBO when available); memory-order-qualified forms (`ld.acquire`/`st.release`); `ld.global.nc`; `cp.async`, `cp.async.mbarrier.arrive` |

| Atomics | `atom.add/inc/dec/and/or/xor/exch/min/max/cas` over 32-bit types; 64-bit add/and/or/xor/exch/min/max/cas; `.add.f32`; global/shared/generic, old-value return; `red.async` (relaxed shared reductions + release global add) and `multimem.red.async` (single-device fallback). `inc`/`dec` are u32-only like ptxas; `.b128`/vector atomics stay clean errors |

| Control flow | `bra`/`bra.uni`, predicated branches, `ret`, `exit`, `trap` |

| Warp/CTA | `shfl.sync` and legacy unsuffixed `shfl` (up/down/bfly/idx), `vote.sync.{all,any,uni,ballot}` and legacy unsuffixed `vote`, `activemask`, `redux.sync`, `match.sync`, `elect.sync`, `bar.sync` (bare/counted), `bar.arrive`, `bar.red.and/or.pred`, `membar`, `mbarrier.{init,inval,expect_tx,complete_tx,arrive,arrive_drop,test_wait,try_wait,pending_count,check_layout}`, `cp.async.mbarrier.arrive` |

| Matrix | `mma.sync.aligned.m16n8k16.row.col.f32.f16/bf16.f32`, `mma.sync.aligned.m16n8k8.row.col.f32.f16/bf16.f32`, `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`, `mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16`, `mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32`, `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32`, `mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64`, `mma.sync.aligned.m16n8k4/k8/k16.row.col.f64.f64.f64.f64` (sm_90), `mma.sync.aligned.m8n8k16/m16n8k16/m16n8k32.row.col.s32.{s8,u8}.{s8,u8}.s32`, `mma.sp[::ordered_metadata].sync.aligned.m16n8k16.row.col.f32.f16/bf16.f16/bf16.f32`, `mma.sp[::ordered_metadata].sync.aligned.m16n8k32.row.col.f32.f16/bf16.f16/bf16.f32`, `wmma` m16n16k16 f16/bf16->f32 and f16->f16, m8n32k16/m32n8k16 f16->f32/f16, m16n16k8 tf32->f32 (`load/mma/store`), `ldmatrix.sync.aligned.m8n8.x{1,2,4}[.trans].b16` |

| `.b128` | `.reg .b128` (two u64 halves), `ld/st[.space].b128`, `mov.b128` |

| Misc | `isspacep` (shared/global/local/param), `trap`, `nanosleep`/`prefetch`/`prefetchu`/`applypriority`/`discard`/`griddepcontrol` (hints or no-ops), `createpolicy` (cache hint, descriptor 0), `%laneid`, `%lanemask_lt`, `WARP_SZ` |



## Memory model



| Space | Model |

|---|---|

| `.param` | Read-only push constants/descriptors (large by-value structs use a storage buffer); `st.param` to the kernel return slot is dropped |

| `.shared` | Static Workgroup arrays (one per variable); unsized `.extern .shared` aliases a launch-sized spec-constant array |

| `.local` | Private per-invocation arrays; direct `[var]` addressing and callee-local frames (fresh caller-local arrays per inline) |

| `.global` (kernel pointers) | Plain pointer parameters use `PhysicalStorageBuffer` device addresses (BDA); texture/sampler/surface parameters and host-call internal state buffers keep descriptor bindings. 4-byte alignment for 32-bit, 8-byte for 64-bit |

| `.global` (module initializer) | Read-only per-thread image; stores/atomics stay errors |

| `.const` | Direct `ld.const [name+imm]` materializes immediates; register-indexed loads read a flattened Private constant image, with symbolic addresses carrying an offset token |

| Generic | Routed by lineage to shared vs. global; ambiguous addresses are errors |

| Warp width | PTX warp instructions require 32/64-lane subgroup hardware when a launch spans more lanes than the native subgroup; launches no larger than the native subgroup are exact on smaller devices |



Shared/global/local sub-word accesses use lane shift+mask for loads and

CAS read-modify-write for stores, so adjacent lanes remain race-free.



When the device supports `vulkanMemoryModel` + `vulkanMemoryModelDeviceScope`

(Vulkan 1.2 core, present on all supported 1.3 devices here), modules

declare `OpMemoryModel Logical Vulkan` and the

`VulkanMemoryModel`/`VulkanMemoryModelDeviceScope` capabilities. The

emitter already emits explicit scopes and memory semantics on atomics

and barriers, so this only formalizes the model (no instruction

changes). `ACUDA_PTX_NO_VMM=1` restores the GLSL450 module for A/B.



## Control flow subset



Supported:



- straight-line code, forward conditional guards, if/else diamonds;

- single-header loops with `OpPhi`-carried register values, including

  multi-block bodies with `break` (branch to the loop merge) and

  `continue` (branch to the latch);

- multi-block `.param`-form callees with a single return, inlined by

  splicing renamed blocks and merging the straight-line call tail into

  the ret block;

- sequential multi-region flow: forward conditional exits and

  single-block self-loops in sequence, with join phis and epilogue

  duplication for early returns.



Explicitly rejected:



- branches into a loop header;

- backward unconditional branches outside a single-header loop;

- nested loops, form-(a) loops mixed with other branches, branches into

  loop headers;

- predicated `ret`/`exit`/`bar`, predicated `ld`, predicated atomics and

  calls;

- path-dependent pointers at a join (recompute from indices instead).



## Runtime-specific behavior



These are semantic choices forced by the Vulkan/SPIR-V target:



- **`cp.async` / `cp.async.bulk`** are lowered as equivalent synchronous

  copies; `commit_group`/`wait_group`/`wait_all` are no-ops because the

  Vulkan compute model has no async-copy primitive. A short `src-size`

  or ignore-src predicate zero-fills the remainder. Bulk copies cover

  `shared::cta`↔`global`; tensor/reduce/multicast forms stay errors.

- **`createpolicy`** produces the default cache-policy descriptor (0);

  cache hints have no effect on results.

- **`mbarrier`** objects are packed 64-bit shared-memory words updated

  by an atomic CAS loop. `try_wait` is a non-blocking test (the caller's

  loop provides the wait), and because `cp.async`/bulk copies are

  synchronous their arrive/complete-tx effects happen immediately.

- **`red.async`** relaxed shared-cluster reductions run on the local

  CTA's shared memory (the backend has no cross-CTA cluster), and the

  completion is reported to the mbarrier as an immediate complete-tx.

  Release/global and `multimem.red.async` forms are release-scope global

  atomics; `multimem` is a single-device fallback. These forms are

  sm_100-gated.

- **`trap`** ends the invocation cleanly (no device launch error), and

  `call __assertfail` uses the same trap semantics; assert arguments are

  ignored because there is no host assert sink.

- **Counted barriers / `bar.arrive`** use a per-workgroup monotonic

  arrival counter in shared memory (16 ids). The entry prologue zeroes

  the counters behind a uniform full-CTA barrier. `bar.sync` waits for

  its round's `count` arrivals; `bar.arrive` only adds one. Counts must

  be nonzero warp-size multiples (register counts are accepted and are

  the caller's responsibility); an id used by a counted/arrive form

  routes a bare `bar.sync` on that id through the same software path

  with the full CTA count. Mismatched counts or divergent placement are

  UB as in CUDA. This path uses atomics and a spin loop rather than a

  hardware partial barrier.

- **`trap`** lowers to `OpReturn` (exit the invocation). SPIR-V's

  `OpTerminateInvocation` is a fragment-shader instruction, so it is not

  used in compute kernels.

- **`brkpt`** emits an `OpString "<ptx>"` and an `OpLine` with the PTX

  line number, which gives debuggers/tools a source-location marker.

  `OpLine` is metadata, not a breakpoint: the instruction remains a

  runtime no-op (Vulkan has no portable compute breakpoint).

- **Narrow arithmetic** stays on 32-bit SPIR-V ops with width masks:

  native `Int8`/`Int16` arithmetic would not reduce GPU work, and 8-bit

  scalar ALU is an ACuda extension (ptxas rejects it). The `Int16`

  capability is enabled only for 16-bit *storage*.

- **Host-side 1.3**: buffer/image barriers use

  `vkCmdPipelineBarrier2`, queue submission uses `vkQueueSubmit2` with

  timeline semaphores, copies use `vkCmd*2`, and format queries use

  `VkFormatProperties3` (`vkGetPhysicalDeviceFormatProperties2`).

  Optional 1.3 features are used when present: `dynamicRendering` for the

  MSAA fill pass (the render pass remains the fallback),

  `computeFullSubgroups` only for pipelines whose specialized workgroup

  size is a multiple of the subgroup size, and

  `VkPhysicalDeviceVulkan13Properties` for launch/allocation limits

  (`maxComputeWorkgroupSubgroups`, workgroup/grid/shared-memory limits,

  and `maxBufferSize`). Image uploads wait on a per-submission fence

  instead of draining the whole queue with `vkQueueWaitIdle`.

- **Batched submission**: launches and device copies are recorded into

  an open command buffer and submitted together (default 32 commands;

  `ACUDA_BATCH=<n>`, `ACUDA_NO_BATCH=1`). Stream sync, events, host

  transfers, image work, and buffer frees flush first. This amortizes

  `vkQueueSubmit2` (~22–27 µs) and makes small kernels

  launch-latency-bound instead of submit-bound. Within a batch, the

  runtime also skips redundant pipeline/descriptor/push-constant binds

  and caches up to eight descriptor sets for repeated/alternating

  buffer bindings.

- **Submission overlap and pool recycling**: buffer-only submissions no

  longer wait for full completion of the previous submission; the

  per-buffer barriers carry the real dependencies (writes track WAR

  ordering too). Image/texture work and event records keep the

  full-completion wait. Command buffers and descriptor pools are

  recycled after their timeline value completes (`ACUDA_FORCE_SERIAL=1`

  restores the old fully serialized behavior for A/B runs;

  `ACUDA_PROFILE_LAUNCH=1` prints per-phase submission timings).

- **Barrier elision**: the compiler reports per-pointer-parameter write

  usage (`acuda_kernel_info_t::param_written`), computed by a

  conservative taint pass over the PTX IR. The runtime tracks a

  per-buffer dirty bit and records the acquire barrier only when an

  earlier recorded write may still be pending; read-only input chains

  therefore skip redundant barriers. Ambiguous stores, surface writes,

  calls, bulk copies, and host-call kernels mark every parameter

  written. Transfers keep their explicit full barriers.

- **Timestamps**: `acuda_event_elapsed` uses `vkCmdWriteTimestamp2`

  query pools and `VkPhysicalDeviceLimits::timestampPeriod`; it returns

  `ACUDA_ERROR_NOT_SUPPORTED` on queues without timestamp bits.

- **ISA introspection**: with `ACUDA_DUMP_ISA=1`, compute pipelines are

  created with `VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR`

  and the driver's NIR/ACO/assembly representations are printed when

  `VK_KHR_pipeline_executable_properties` is available.

- **`nanosleep`** is dropped after type-checking (performance hint only).

- **`fma.rn.f32`/`fma.rn.f64`** use `OpFmaKHR` when the init probe

  proves that path correct, otherwise `GLSL.std.450 Fma` (specified

  fused, and reliable on RADV where the `OpFmaKHR` probe fails). Only

  **directed** `fma`/`mad`/`add`/`sub`/`mul` use the fully integer

  software FMA (exact product, normalized-window addend alignment,

  sticky bits, normalization, and our own `rn/rz/rm/rp` rounding).

  f32 uses 64-bit intermediates; f64 uses 128-bit significands.

  `ACUDA_PTX_NO_FMA=1` forces the software path. MMA/WMMA accumulation

  uses the same fused path, f16→f32 widening uses native `OpFConvert`

  (with f16 subnormals flushed to match the emulated path), and

  f16/bf16 unpack conversions are CSE'd within a basic block.

- **Int8 MMA**: the s8/u8 `mma.sync` inner loop combines each group of

  four K values into one packed `OpSDot`/`OpUDot`/`OpSUDot` when

  `shaderIntegerDotProduct` is available (~13x in `bench_math`); the

  exact scalar `IMul`/`IAdd` chain remains as the fallback

  (`ACUDA_PTX_NO_DOT_PRODUCT=1`).

- **Mixed float dot (VALVE)**: when

  `VK_VALVE_shader_mixed_float_dot_product` exposes

  `shaderMixedFloatDotProductFloat16AccFloat32`, dense and sparse f16

  `mma.sync`/`mma.sp` accumulation uses `OpFDot2MixAcc32VALVE` (the AMD

  `v_dot2_f32_f16` instruction): one instruction computes two f16

  products plus the f32 accumulator. This is ~6.4x faster than the

  widening + FMA sequence in `bench_math` (dense ~3.75 -> ~24 TFLOP/s,

  sparse ~2.2 -> ~14.3 TFLOP/s; see `tests/bench/README.md`). Sparse MMA packs each pair of dense K

  positions into one 32-bit word after expanding the 2:4 selector, so

  both halves of the fdot2 input carry live values. The packed A words

  are memoized per basic block and reused by later `mma.sp` instructions

  with the same fragments and selector, which shrinks the emitted module

  by ~27% (`test_ptx` covers two MMAs sharing one packing).

  The installed validation layer predates the extension, so the path is

  enabled only when validation is off; `ACUDA_FDOT2=1` forces it and

  `ACUDA_NO_FDOT2=1` disables it.

- **`acuda.dot2.f32.f16` intrinsic** (ACuda PTX extension):

  `acuda.dot2.f32.f16 d, a, b, c;` computes

  `d = c + dot(f16x2 a, f16x2 b)` with f32 accumulation. It lowers to

  `OpFDot2MixAcc32VALVE` when the VALVE feature is active, otherwise to

  four half-widening conversions and two FMAs. `bench_gemm` uses it to

  feed shared-memory operands directly, avoiding the cross-lane

  gathers that `mma.sync` fragment semantics require. Like the VALVE

  capability itself, the hardware form needs validation off (or

  `ACUDA_FDOT2=1`). The mnemonic is ACuda-specific and is not accepted

  by ptxas; all ptxas-validated tests use standard PTX.

- **Nested loops**: a restricted nested-loop shape is emitted as

  structured SPIR-V (kind 9): an outer form-(a) loop whose body holds one

  inner form-(a) loop with the inner merge doubling as the outer latch

  (the persistent-grid layout). Other nested CFGs still use the generic

  state-machine fallback. `test_ptx` covers the structured form with a

  two-level counter kernel.

- **Shared-memory addressing**: a constant byte offset on a

  shared-derived pointer is kept out of the 64-bit pointer chain and

  added to a 32-bit word index instead, so ACO can fold it into the LDS

  instruction offset (and pair adjacent 32-bit reads). Before this, every

  `ld.shared [reg+imm]` emitted a 64-bit add; the GEMM benchmark gained

  ~45% (10.0 -> 14.9 TFLOP/s at 2048^3), and the 32-wide K step took it

  to ~17.0, then tile/occupancy tuning to ~21.1 at 2048^3 and ~24.0 at

  4096x4096x2048.

- **BF16 conversions**: widening BF16 to F32 is a 16-bit shift (BF16 is

  the upper half of an F32); dense f16/bf16 MMA also processes both

  halves of each 32-bit fragment register per shuffle and blocks the

  four output accumulators.

- **Native f16**: `fma.rn.f16x2` uses native `Float16` arithmetic when

  `shaderFloat16` is available and the init probe measures it faster

  than the f32-emulated path (RADV: ~129x faster in `bench_math`; the

  init probe's smaller kernel sees >30x). Other

  f16 operations remain f32-emulated. `ACUDA_F16_EMULATED=1` forces the

  emulated path and `ACUDA_F16_PROBE_OFF=1` skips the timing probe

  (native when supported).

- **Subgroup control**: `subgroupSizeControl` requests 32 lanes for PTX

  warp kernels. `ACUDA_SUBGROUP_SIZE=<n>` forces a size for

  barrier-free kernels (occupancy experiments). Native dp4a gains

  ~2–3% at 32 on RADV, but register-heavy expansions can regress

  (the no-dot fallback is ~2x slower at 32), so it stays opt-in.

  Kernels using `bar.*`/`mbarrier` keep the device default so counted

  barriers retain their semantics.

- **32-lane warp semantics**: kernels that use shfl/vote/redux/match/elect

  or `bar.warp.sync` request `requiredSubgroupSize = 32` through

  `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` whenever

  `subgroupSizeControl` supports it for compute. Without the control,

  the compiler's 32-lane emitters are exact on 32- and 64-lane default

  subgroups (64-lane waves are split into warp segments); any other

  default size makes those kernels fail with `NOT_SUPPORTED` instead of

  silently corrupting lane masks and shuffles.

- **Workgroup zero-initialization**: with

  `shaderZeroInitializeWorkgroupMemory`, workgroup storage arrives

  zeroed, so the software-barrier counter prologue (17 stores plus a

  CTA barrier) is dropped. `.common` remains explicitly zeroed because

  it is Private storage. `ACUDA_PTX_NO_ZERO_INIT_WG=1` restores the

  prologue.

- **Float controls**: when the device reports

  `shaderDenormPreserveFloat32/64` and

  `shaderSignedZeroInfNanPreserveFloat32/64`, the module declares the

  corresponding execution modes (and capabilities), so non-`.ftz`

  denormal results and signed-zero/NaN behavior follow IEEE instead of

  the driver default. `ACUDA_PTX_NO_FLOAT_CONTROLS=1` disables them;

  `ACUDA_DEBUG_FLOAT=1` prints the choice.

- **Native 8/16-bit storage**: `.b8/.u8` global/generic accesses use a

  `RuntimeArray<uchar>` alias (`StorageBuffer8BitAccess`) when

  `storageBuffer8BitAccess` is available; `.b16/.u16` use a

  `RuntimeArray<ushort>` alias (`StorageBuffer16BitAccess` + `Int16`,

  with `shaderInt16` enabled) when `storageBuffer16BitAccess` is

  available. Byte/halfword addressing replaces the shift/mask loads and

  the u32-word CAS stores; shared and Private accesses keep the

  word-packed path. `ACUDA_PTX_NO_SSBO8=1` / `ACUDA_PTX_NO_SSBO16=1`

  force the old paths.

- **Buffer device addresses**: the runtime enables `bufferDeviceAddress`,

  allocates buffers with `SHADER_DEVICE_ADDRESS_BIT` +

  `MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT`, and exposes the value through

  `acuda_buffer_device_address()`. Every plain pointer parameter is

  passed as a 64-bit address in the push-constant block and accessed

  through `PhysicalStorageBuffer` (`OpConvertUToPtr` + aligned physical

  loads/stores/atomics, with the PhysicalStorageBuffer64 addressing

  model), while `.texref`/`.samplerref`/`.surfref` parameters and the

  host-call internal state buffers keep descriptor bindings in the same

  kernel. Host-call state machines carry physical pointers through

  their saved register file and dynamic-home dispatch (a dedicated

  physical tag), and matrix instructions use the same addressing.

  `ACUDA_PTX_NO_BDA=1` forces bindings everywhere.

  Textures/samplers/surfaces may be mixed with plain pointers in one

  kernel (the plain pointers use BDA, the descriptors keep bindings).

  Pointer halves and dynamic pointer tables round-trip real addresses,

  so `mov.b64 {lo,hi}`/reassembly works identically; pointer+pointer

  add/sub yields the raw address value under BDA (only differences are

  portable, as on real hardware).

- **`dp4a`** uses the native packed integer dot product

  (`OpSDot`/`OpUDot`/`OpSUDot` with `DotProduct`/

  `DotProductInput4x8BitPacked`) when the device reports

  `shaderIntegerDotProduct` **and the specific form is accelerated**

  (`VkPhysicalDeviceShaderIntegerDotProductProperties`). Both the

  instructions and capabilities are core in the SPIR-V 1.6 module.

  On RADV/RX 6800 XT the signed and unsigned packed forms are

  accelerated but the mixed-signedness form is not, so mixed forms keep

  the exact expansion. `acuda.dp4a.sat.*` (ACuda extension, not ptxas)

  selects the `OpSDotAccSat`/`OpUDotAccSat`/`OpSUDotAccSat` forms with

  an exact 64-bit saturating fallback. `ACUDA_PTX_NO_DOT_PRODUCT=1`

  forces the fallback; `ACUDA_PTX_NO_DOT_ACCEL=1` ignores the

  acceleration properties. On RADV/RX 6800 XT (which maps it to

  `v_dot4_u32_u8`) the kernel drops from 568 to 101 ISA instructions

  and 809 to 169 SPIR-V function-body instructions; the in-tree

  `bench_dp4a` (`tests/bench/`) measures 2.95x compute-bound (`reps=64`)

  and 1.03x launch-bound (`reps=1`); the launch-bound regime dropped from

  ~0.33 ms to ~0.004 ms per launch after command/descriptor-pool

  recycling, submission overlap, and batched submission.

  `dp2a` always uses the exact expansion: a `DotProductInputAll`

  `<2 x i16>` prototype reduced static ISA instructions from 320 to 181

  but was runtime-neutral (~1.0x) and needed `shaderInt16`/`Int16`, so

  it was not adopted.

- **`div.rn` and directed division** use exact integer long division,

  because desktop drivers (observed on RADV) are not correctly rounded

  for `OpFDiv`. `.approx`/`.full.f32` keep the hardware path.

- **64-bit atomics** use `Int64Atomics`; the runtime enables the

  buffer/shared 64-bit atomic features when available and rejects the

  launch cleanly otherwise. Global 64-bit atomics access a lazily

  created u64 alias view of the storage binding (the normal u32-word

  view cannot be atomically addressed as u64); shared 64-bit atomics

  stay a clean error.

- **f64** uses native `Float64` when the device supports it. With

  `ACUDA_EMU_F64=1`, f64 is demoted to f32 (approximate, warned once);

  f64 entry parameters are not supported.



## Known clean errors



The following are intentionally rejected rather than approximated:



- recursive calls, and external functions or indirect-call shapes outside the supported dispatcher;

- `.reg`-form multi-block callees, multiple-return callees, nested calls,

  or a multi-block call outside the final caller block;

- `.tex`, `.surf`, image/surface operations;

- `.global` stores and atomics on module-initialized `.global` images;

- 64-bit shared atomics, `.inc`/`.dec` on 64-bit types, and `.b128`

  atomics (`.cas.b128`);

- `.u8/.s8` `div/rem/mad` (the wrap-ALU forms are supported);

- `.bf16`/`.bf16x2` division;

- dynamic-id counted barriers;

- `.v8` anywhere except `ld/st.v8` and brace-list `mov.v8`;

- `.reg .v8` declarations, vector↔scalar `mov` shapes not listed;

- address-taken `.local` variables and parameter-address arithmetic;

- modules with no launchable `.entry`.



The ZLUDA corpus's two remaining rejections are examples:

`extern_func.ptx` (undefined external call) and `global_extern_array.ptx`

(no entry point).



## Future optimizations



- **Float-controls RTZ fast path for `.rz`.** Vulkan exposes only

  `RoundingModeRTE` (the default) and `RoundingModeRTZ`; there is no

  `RTN`/`RTP` execution mode and no `shaderRoundingModeRTN/RTGFloat*`

  property. Execution modes are also entry-point-wide, while PTX can mix

  rounding modes instruction by instruction. A kernel that uses only

  `.rz` could therefore replace the integer residual with

  `OpExecutionMode %main RoundingModeRTZ <width>` plus ordinary

  `OpFAdd`/`OpFMul`/`OpFDiv`/GLSL `Fma`, gated by an init probe that

  verifies the driver honours RTZ for f32/f64. This is an optimization

  only: `.rm`/`.rp` and mixed-mode kernels still need the software

  path, and it is not needed for correctness. Not implemented.



## Testing and coverage



```bash

/usr/bin/ctest --test-dir build/release --output-on-failure

```



`~/.local/bin/ctest` may be a broken shim; use `/usr/bin/ctest`.



| Test | What it covers |

|---|---|

| `test_ptx` | hand-written kernels, GPU launches, metadata, negative pins |

| `test_driver` / `test_cudart` | driver/runtime ABI and real kernels through `libcuda`/`libcudart` |

| `test_zluda_ptx` | optional local ZLUDA corpus sweep (allowlist + must-fail set) |

| `test_ptx_kit` | 36 imported Windows/RTX3070 truth cases: compile, `spirv-val`, launch, exact output |



Coverage snapshots (current tree):



```text

ZLUDA corpus:   196/198 compiled, 2 clean-rejected

Windows kit:     36/36 compile + validate + run

ctest:           10/10

```



Tests are in `tests/ptx/` (hand-written), `tests/zluda-ptx/` (optional

local ZLUDA corpus), and `tests/windows-kit/` (imported truth cases).

When adding support, update the relevant allowlist and mirrored negative

test, and update `supported-isa.md` only when the feature matrix changes.



## Environment knobs



| Variable | Effect |

|---|---|

| `ACUDA_PTX="9.4"` | `.version` ceiling (default 9.4; unset/garbage = 9.4) |

| `ACUDA_SM="sm_89"` | Reject newer `.target` values (unset = accept) |

| `ACUDA_EMU_F64=1` | Demote f64 to f32 (approximate) |

| `ACUDA_PTX_NO_FMA=1` | Force the integer directed-rounding fallback |

| `ACUDA_PTX_NO_DOT_PRODUCT=1` | Force the dp4a multiply-add fallback |

| `ACUDA_PTX_NO_SUBGROUP32=1` | Do not request 32-wide subgroups |

| `ACUDA_PTX_NO_ZERO_INIT_WG=1` | Keep the software-barrier prologue |

| `ACUDA_PTX_NO_SSBO8=1` | Force the word-packed byte path |

| `ACUDA_PTX_NO_SSBO16=1` | Force the word-packed halfword path |

| `ACUDA_PTX_NO_FLOAT_CONTROLS=1` | Keep the driver's float-control defaults |

| `ACUDA_PTX_NO_VMM=1` | Emit the GLSL450 memory model instead of Vulkan |

| `ACUDA_PTX_NO_BDA=1` | Use descriptor bindings instead of physical addresses |

| `ACUDA_DEBUG_FLOAT=1` | Print float-control choices |

| `ACUDA_DEBUG_SUBGROUP=1` | Print subgroup size/control choices |

| `ACUDA_DUMP_SPV_DIR=<dir>` | Dump each emitted module as `.spv` |

| `ACUDA_DEBUG_CLASSIFY=1` | Print parameter classification decisions |

| `ACUDA_NO_VALIDATION=1` | Do not enable the Vulkan validation layer |

| `ACUDA_FORCE_SERIAL=1` | Disable submission overlap (old serialized behavior) |

| `ACUDA_BATCH=<n>` | Commands per batched submission (default 32) |

| `ACUDA_NO_BATCH=1` | Flush after every command |

| `ACUDA_SUBGROUP_SIZE=<n>` | Force a subgroup size for barrier-free kernels (occupancy tuning) |

| `ACUDA_F16_EMULATED=1` | Force f16 arithmetic through the f32 path |

| `ACUDA_F16_PROBE_OFF=1` | Skip the native-vs-emulated f16 timing probe |

| `ACUDA_FDOT2=1` | Force `OpFDot2MixAcc32VALVE` even with validation enabled |

| `ACUDA_NO_FDOT2=1` | Disable the mixed float dot path |

| `ACUDA_DEBUG_F16=1` | Print the f16 probe timings and choice |

| `ACUDA_PROFILE_LAUNCH=1` | Print recording/submission timings at exit |



## Source layout



```text

src/compiler/

  lex.c                 tokenizer

  parse.c               shared parser helpers + ptx_parse

  parse_instr.c         general instruction decoding

  parse_atom.c          atom.* decoding

  parse_mbarrier.c      elect.sync / mbarrier decoding

  parse_red_async.c     red.async / multimem.red.async decoding

  parse_matrix.c        ldmatrix / mma decoding

  parse_entry.c         signatures + function body loop

  parse_decl.c          .shared/.local/.const/.global declarations

  parse_inline.c        call slots + inlining

  spirv.c               shared emitter helpers + MOV/dispatch

  spirv_module.c        module/capabilities/entry/prelude

  spirv_arith.c         arithmetic/conversions/comparisons

  spirv_mem.c           ld/st/cp.async

  spirv_control.c       CFG, guards, loops, flow/phi

  spirv_group.c         shfl/vote/redux/match/MMA/ldmatrix/barrier

  spirv_atom.c          atomics

  spirv_mbarrier.c      elect.sync / mbarrier lowering

  spirv_red_async.c     red.async / multimem.red.async lowering

  spirv_params.c        parameter ABI classification

```



For the full feature table, see [`supported-isa.md`](supported-isa.md).
