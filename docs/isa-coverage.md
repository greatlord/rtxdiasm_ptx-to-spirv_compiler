# PTX ISA 9.3 section coverage

Audit of ACuda's support against the PTX ISA 9.3 instruction table of
contents (sections 9.7.1–9.7.14). Status meanings:

- **Done** — the section's documented forms/types are implemented (or
  deliberately accepted as the documented no-op/hint), with clean
  errors for anything outside the subset.
- **Partly** — the instruction works for a documented subset, but some
  documented types, modifiers, or operand forms remain clean errors.
- **Todo** — not implemented; the parser/emitter rejects it with a
  clean `line N: ...` error.

This is a status map, not a second support matrix. Exact accepted
spellings and test references live in
[`supported-isa.md`](supported-isa.md); known omissions are also listed
in [`ptx-support.md`](ptx-support.md).

Legend: ✅ done · 🔶 partly · ❌ todo.

## 9.7.1 Integer arithmetic

| § | Instruction | Status | Notes |
|---|---|---|---|
| 9.7.1.1 | `add` | ✅ | All documented integer types, packed x2/x4, `.sat`; ptxas-validated. |
| 9.7.1.2 | `sub` | ✅ | All documented integer types, packed x2/x4, `.sat`. |
| 9.7.1.3 | `mul` | ✅ | `.lo/.hi/.wide` over 16/32/64-bit signed/unsigned. |
| 9.7.1.4 | `mad` | ✅ | `.lo/.hi/.wide`, 16/32/64-bit, `mad.hi.sat.s32`. |
| 9.7.1.5 | `clmad` | ✅ | `.lo/.hi.u64` (PTX 9.3). |
| 9.7.1.6 | `mul24` | ✅ | `.lo/.hi.{u32,s32}`. |
| 9.7.1.7 | `mad24` | ✅ | `.lo/.hi.{u32,s32}` and `.hi.sat.s32`. |
| 9.7.1.8 | `sad` | ✅ | `.u16/.s16/.u32/.s32/.u64/.s64`. |
| 9.7.1.9 | `div` | ✅ | `.u16/.s16/.u32/.s32/.u64/.s64`; signed C semantics. |
| 9.7.1.10 | `rem` | ✅ | Same widths; sign follows dividend. |
| 9.7.1.11 | `abs` | ✅ | `.s16/.s32/.s64` (integer). |
| 9.7.1.12 | `neg` | ✅ | `.s16/.s32/.s64`, `.s8x4`. |
| 9.7.1.13 | `min` | ✅ | Integer scalar + packed + `.relu` forms. |
| 9.7.1.14 | `max` | ✅ | Integer scalar + packed + `.relu` forms. |
| 9.7.1.15 | `popc` | ✅ | `.b32/.b64` (destination `.u32`). |
| 9.7.1.16 | `clz` | ✅ | `.b32/.b64`. |
| 9.7.1.17 | `bfind` | ✅ | `.u32/.s32/.u64/.s64`, `.shiftamt`. |
| 9.7.1.18 | `fns` | ✅ | `fns.b32`. |
| 9.7.1.19 | `brev` | ✅ | `.b32/.b64`. |
| 9.7.1.20 | `bfe` | ✅ | `.u32/.s32/.u64/.s64`. |
| 9.7.1.21 | `bfi` | ✅ | `.b32/.b64`. |
| 9.7.1.22 | `szext` | ✅ | `.clamp/.wrap.{u32,s32}`. |
| 9.7.1.23 | `bmsk` | ✅ | `.clamp/.wrap.b32`. |
| 9.7.1.24 | `dp4a` | ✅ | Every `u32`/`s32` operand mix. Uses native `OpSDot`/`OpUDot`/`OpSUDot` (packed 4x8-bit) when `shaderIntegerDotProduct` is available, else an exact expansion. |
| 9.7.1.25 | `dp2a` | ✅ | Every `u32`/`s32` mix, `.lo/.hi`; exact expansion (an experimental `DotProductInputAll` vector form measured runtime-neutral on RADV and needs `shaderInt16`). |

## 9.7.2 Extended-precision arithmetic

| § | Instruction | Status | Notes |
|---|---|---|---|
| 9.7.2.1 | `add.cc` | ✅ | `.u32/.s32/.u64/.s64`, PTX carry semantics. |
| 9.7.2.2 | `addc` | ✅ | Optional `.cc`. |
| 9.7.2.3 | `sub.cc` | ✅ | `.u32/.s32/.u64/.s64`, borrow semantics. |
| 9.7.2.4 | `subc` | ✅ | Optional `.cc`. |
| 9.7.2.5 | `mad.cc` | ✅ | `.hi/.lo`, 32/64-bit. |
| 9.7.2.6 | `madc` | ✅ | `.hi/.lo`, optional `.cc`, 32/64-bit. |

## 9.7.3 Floating point

| § | Instruction | Status | Notes |
|---|---|---|---|
| 9.7.3.1 | `testp` | ✅ | All six properties, `.f32/.f64`. |
| 9.7.3.2 | `copysign` | ✅ | `.f32/.f64`. |
| 9.7.3.3 | `add` | 🔶 | Scalar `.f32/.f64/.f16/.bf16`, packed half forms, and `.f32x2` (two scalar lanes; PTX 8.6/sm_100 shape emulated). Directed `.f32/.f64` uses the integer software FMA, which is exact for subnormal results too. |
| 9.7.3.4 | `sub` | 🔶 | Same subset as `add`. |
| 9.7.3.5 | `mul` | 🔶 | Same subset as `add`. |
| 9.7.3.6 | `fma` | 🔶 | `.rn/.rz/.rm/.rp` `.f32/.f64`, scalar/half/packed, `.f32x2` (two scalar lanes), half `.relu`, and half/bf16 `.oob`. Directed roundings and the f32/f64 fallback use a fully integer software FMA (exact product, alignment, normalization and rounding) that runs on the GPU. `.oob` is accepted and lowered as an ordinary fused multiply-add; the tensor OOB-NaN payload is undocumented, so the special OOB-NaN clamp is not modelled (exact for every non-OOB input). |
| 9.7.3.7 | `mad` | 🔶 | `mad.f32`/`.f64` with `.rn/.rz/.rm/.rp`; routed through the same software FMA as `fma` when the fused-path probe fails. |
| 9.7.3.8 | `div` | 🔶 | `rn` and directed `.rz/.rm/.rp` over `.f32/.f64` use exact integer long division (the driver's fdiv is not correctly rounded here); `.approx/.full.f32` keep the hardware fdiv; `.ftz` where documented. |
| 9.7.3.9 | `abs` | ✅ | `.f16/.f16x2/.bf16/.bf16x2/.f32/.f64`; `.ftz` where documented. |
| 9.7.3.10 | `neg` | ✅ | Same float set; `.ftz` where documented. |
| 9.7.3.11 | `min` | 🔶 | `ftz`/`NaN` and half forms; `.xorsign.abs` and 3-input `.abs` remain todo. |
| 9.7.3.12 | `max` | 🔶 | Same as `min`. |
| 9.7.3.13 | `rcp` | 🔶 | `approx/rn.f32`, `rn`/directed f32 and f64 (exact integer long division for rn/directed); the special `rcp.approx.ftz.f64` remains todo. |
| 9.7.3.14 | `rcp.approx.ftz.f64` | ❌ | Special truncated 1.11.20 form; not implemented. |
| 9.7.3.15 | `sqrt` | 🔶 | `approx/rn.f32` (+`.ftz`), `rn.f64`; directed roundings remain todo. |
| 9.7.3.16 | `rsqrt` | 🔶 | `approx.f32/.f64` (+`.ftz.f32`); special `.approx.ftz.f64` remains todo. |
| 9.7.3.17 | `rsqrt.approx.ftz.f64` | ❌ | Special truncated form; not implemented. |
| 9.7.3.18 | `sin` | ✅ | `.approx{.ftz}.f32`. |
| 9.7.3.19 | `cos` | ✅ | `.approx{.ftz}.f32`. |
| 9.7.3.20 | `lg2` | ✅ | `.approx{.ftz}.f32`. |
| 9.7.3.21 | `ex2` | ✅ | `.approx{.ftz}.f32` and half forms. |
| 9.7.3.22 | `tanh` | ✅ | `.approx` over f32 and half types. |

## 9.7.4 Half precision

| § | Instruction | Status | Notes |
|---|---|---|---|
| 9.7.4.1 | `add` | ✅ | `.f16/.f16x2/.bf16/.bf16x2`, `.rn/.ftz/.sat` as documented. |
| 9.7.4.2 | `sub` | ✅ | Same. |
| 9.7.4.3 | `mul` | ✅ | Same. |
| 9.7.4.4 | `fma` | 🔶 | `.f16/.f16x2/.bf16/.bf16x2`, `.ftz/.sat`; `.relu` remains todo. |
| 9.7.4.5 | `neg` | ✅ | All half types, `.ftz` where documented. |
| 9.7.4.6 | `abs` | ✅ | All half types, `.ftz` where documented. |
| 9.7.4.7 | `min` | 🔶 | Default/`NaN`/`ftz`; `.xorsign.abs` remains todo. |
| 9.7.4.8 | `max` | 🔶 | Same. |
| 9.7.4.9 | `tanh` | ✅ | `.f16/.f16x2/.bf16/.bf16x2`. |
| 9.7.4.10 | `ex2` | ✅ | `.f16/.f16x2` and `.ftz.bf16/.bf16x2`. |

## 9.7.5 Mixed precision

| § | Instruction | Status | Notes |
|---|---|---|---|
| 9.7.5.1 | `add` | ⛔ | `add.f32.{f16,bf16}` requires sm_100; explicit target-gated error. |
| 9.7.5.2 | `sub` | ⛔ | `sub.f32.{f16,bf16}` requires sm_100; explicit target-gated error. |
| 9.7.5.3 | `fma` | ⛔ | `fma.f32.{f16,bf16}` requires sm_100; explicit target-gated error. |

## 9.7.6 Comparison and selection

| § | Instruction | Status | Notes |
|---|---|---|---|
| 9.7.6.1 | `set` | 🔶 | Full `.f16/.f16x2` CmpOp matrix with `.ftz` and `and/or/xor` BoolOp, integer-mask/`.f16`/`.f16x2` destinations. `.bf16/.bf16x2` require sm_90 and are target-gated; other exotic destination types remain todo. |
| 9.7.6.2 | `setp` | ✅ | Full comparison/boolean/dual-destination subset. |
| 9.7.6.3 | `selp` | ✅ | Scalar select with predicate/immediate. |
| 9.7.6.4 | `slct` | ✅ | `.s32`/`.f32` compare, all documented destination widths, `.ftz.f32`. |

## 9.7.7 Half-precision comparison

| § | Instruction | Status | Notes |
|---|---|---|---|
| 9.7.7.1 | `set` | ✅ | Full `.f16/.f16x2` CmpOp matrix with `.ftz` and BoolOp; integer/`.f16`/`.f16x2` destinations. bf16 variants are sm_90-gated. |
| 9.7.7.2 | `setp` | ✅ | `.f16` and dual-destination `.f16x2` comparisons, all CmpOps, `.ftz`, and BoolOp forms. bf16 variants are sm_90-gated. |

## 9.7.8 Logic and shift

| § | Instruction | Status | Notes |
|---|---|---|---|
| 9.7.8.1 | `and` | ✅ | 32/64-bit (and narrow aliases). |
| 9.7.8.2 | `or` | ✅ | 32/64-bit. |
| 9.7.8.3 | `xor` | ✅ | 32/64-bit. |
| 9.7.8.4 | `not` | ✅ | 32/64-bit. |
| 9.7.8.5 | `cnot` | ✅ | `.b16/.b32/.b64`. |
| 9.7.8.6 | `lop3` | ✅ | Basic `lop3.b32` plus the `.or`/`.and` BoolOp forms (`d|p`/`_|p`, `p = (d != 0) BoolOp q`). |
| 9.7.8.7 | `shf` | ✅ | `.clamp/.wrap.l/r.b32`. |
| 9.7.8.8 | `shl` | ✅ | 8/16/32/64-bit (narrow forms are extensions). |
| 9.7.8.9 | `shr` | ✅ | Logical/arithmetic 8/16/32/64-bit. |

## 9.7.9 Data movement and conversion

| § | Instruction | Status | Notes |
|---|---|---|---|
| 9.7.9.1 | Cache operators | ✅ | Scalar/vector `ld` `.ca/.cg/.cs/.lu/.cv`, `st` `.wb/.cg/.cs/.wt`, `ld.global.nc`, `.volatile/.weak/.relaxed/.acquire/.release`, and `.L1::*`/`.L2::*` cache hints (including the `L2::cache_hint` policy operand) are accepted as hints on the coherent model. |
| 9.7.9.2 | Cache eviction priority hints | 🔶 | Parsed/tolerated where part of supported `ld/st`; no behavioural effect. |
| 9.7.9.3 | `mov` | ✅ | Integer/float/predicate/vector/brace forms, including packed brace moves (`mov.b32 {lo,hi},r`); `.u8`/`.f16` mov are not PTX (ptxas rejects them). |
| 9.7.9.4 | `mov` (second section) | ✅ | Same subset. |
| 9.7.9.5 | `shfl` (deprecated) | ✅ | `.up/.down/.bfly/.idx` with implicit full mask. |
| 9.7.9.6 | `shfl.sync` | ✅ | `.up/.down/.bfly/.idx`, `.b32/.f32`, optional predicate. |
| 9.7.9.7 | `prmt` | ✅ | `.b32/.u32/.s32`. |
| 9.7.9.8 | `ld` | ✅ | All ISA state spaces (`.const/.global/.local/.param{::entry,::func}/.shared{::cta,::cluster}`), scalar and `.v2/.v4/.v8` widths, `.b128`, `.weak/.volatile/.relaxed/.acquire/.mmio`, `cta/cluster/gpu/sys` scopes, `.unified` address suffix, `.L1::`/`.L2::` eviction priorities, `.L2::64B/128B/256B` prefetch hints, and `.L2::cache_hint` with its policy operand (scalar and vector); `.b8/.u8` global loads use a native byte view when `storageBuffer8BitAccess` is available, and `.b16/.u16` a native halfword view when `storageBuffer16BitAccess` is available. Plain-buffer kernels address globals through `PhysicalStorageBuffer` (BDA) when available. `.tex`/`.surf` are texture/surface instructions, not `ld` state spaces. |
| 9.7.9.9 | `ld.global.nc` | ✅ | Plain global load (coherency hint only). |
| 9.7.9.10 | `ldu` | ✅ | Scalar `.b8/.b16/.b32/.b64/.u*/.s*/.f32/.f64` and `.v2/.v4/.v8/.b128` forms (`.global`/generic); lowered through the normal load path. |
| 9.7.9.11 | `st` | 🔶 | Supported spaces/types/vector widths plus cache/ordering hints and `.shared::cta`/`.shared::cluster` aliases; global `.b8/.u8`/`.b16/.u16` stores use the native byte/halfword views when the storage features are available; plain-buffer kernels use `PhysicalStorageBuffer` (BDA) stores; bulk forms remain todo. |
| 9.7.9.12 | `st.async` | ⛔ | Requires sm_90; explicit target-gated error. |
| 9.7.9.13 | `multimem.st.async` | ✅ | Single-device fallback: a multimem address names the local GPU's memory, so the release store lowers to an ordinary global store. Scalar `.b32/.b64/.u*/s*/.f32/.f64`; vector forms remain clean errors. |
| 9.7.9.14 | `st.bulk` | ⛔ | Requires sm_90; explicit target-gated error. |
| 9.7.9.15 | `multimem.ld_reduce`/`multimem.st`/`multimem.red` | ✅ | Single-device fallback: `ld_reduce` is a load (reduction over one replica is the identity), `st` a store, and `red` the matching scalar atomic reduction. `.add/.min/.max/.and/.or/.xor` over `.b32/.b64/.u*/s*` plus `.f32/.f64` add/min/max; vector and 16-bit/8-bit float forms remain clean errors. |
| 9.7.9.16 | `prefetch`/`prefetchu` | 🔶 | Scalar cache prefetch/uniform-cache hint forms are accepted as no-ops; tensormap prefetch remains todo. |
| 9.7.9.17 | `applypriority` | ✅ | `.global.L2::evict_normal [a], 128` accepted as a hint. |
| 9.7.9.18 | `discard` | ✅ | `.global.L2 [a], 128` accepted as a hint (data not destructively invalidated). |
| 9.7.9.19 | `createpolicy` | ✅ | Accepted as the default cache-policy descriptor (0); hints have no effect. |
| 9.7.9.20 | `isspacep` | ✅ | `.shared{::cta,::cluster}`/`.global`/`.local`/`.const`/`.param{::entry}` over tracked pointer registers and `.const` identity tokens. |
| 9.7.9.21 | `cvta` | ✅ | Shared/generic/global/local address conversions in the supported address model. |
| 9.7.9.22 | `cvt` | 🔶 | Integer width/sign matrix, f16/f32/f64, packed/fp8/sat forms, and int↔float (`rn` plus directed `rz/rm/rp` int→f16/f32/f64 over s/u 8/16/32/64; `rni/rzi/rmi/rpi` f32/f64→int and `.f64.f64` rounding over 16/32/64-bit destinations, saturating via GLSL RoundEven/Trunc/Floor/Ceil for f64), `cvt.{rn,rna}.tf32.f32` (sm_90-gated), `.rn[.relu].satfinite.{f16,f16x2}.f32`, and the micro-float family: `cvt.rn.satfinite{.relu}.{e2m1x2,e2m3x2,e3m2x2}.f32` (pack two f32, RNE + saturation), the same formats from `.f16x2/.bf16x2`, `cvt.rn{.relu}.{f16x2,bf16x2}` unpacking those formats (optional `.scaled::n2::ue8m0` for bf16x2), `cvt.{rz,rp}{.satfinite}.ue8m0x2.{f32,bf16x2}` / `cvt.rn.bf16x2.ue8m0x2`, and `s2f6x2` fixed point both directions (f32/bf16x2, optional block scale). e4m3/e5m2 x2 also work to/from bf16x2 and with `.scaled::n2::ue8m0`. The x4 packing forms (`.e2m1x4/.e4m3x4/.e5m2x4/.e2m3x4/.e3m2x4` with `.rs` stochastic rounding) stay clean errors/todo. |
| 9.7.9.23 | `cvt.pack` | ✅ | `.sat.{u16,s16}.s32` (no `c`) and `.sat.{u8,s8,u4,s4,u2,s2}.s32.b32`, saturating each source then packing high/low lanes with `c` in the remaining bits. |
| 9.7.9.24 | `mapa` | 🔶 | Single-CTA cluster approximation: `.shared::cluster.u32/.u64` register sources map to the same local address (the target rank is ignored, like `barrier.cluster`). Shared-variable/`var+imm` sources remain clean errors. sm_90-gated. |
| 9.7.9.25 | `getctarank` | 🔶 | Returns 0 (a one-CTA cluster has one member) for `.u32/.u64` destinations; generic and `.shared::cluster` forms. sm_90-gated. |
| 9.7.9.26 | Asynchronous copy | 🔶 | Non-bulk `cp.async.{ca,cg}.shared[::cta].global` with 4/8/16-byte copies, immediate or register `src-size`, ignore-src predicates, cache/prefetch hints, and `commit_group`/`wait_group`/`wait_all`; bulk `cp.async.bulk` for `shared::cta`↔`global` with register/immediate size, `bulk_group`/mbarrier operands, and `cp.async.bulk.prefetch`/`commit_group`/`wait_group`. Copies are lowered synchronously (there is no Vulkan compute async-copy primitive), so completion is immediate and wait/group ops are no-ops by construction. Tensor, `cp.reduce.async.bulk`, multicast, and mbarrier-based waits remain todo (mbarrier is 9.7.14.16). |
| 9.7.9.27 | `tensormap.replace` | ⛔ | Requires sm_90a; explicit target-gated error. |

## 9.7.10 Fabric

| § | Instruction | Status | Notes |
|---|---|---|---|
| 9.7.10 | Fabric operations | ❌ | CFT handles / fabric operations not implemented. |

## 9.7.11 Texture

| § | Instruction | Status | Notes |
|---|---|---|---|
| 9.7.11 | `tex`/`tld4`/`txq`/`istypep` | ✅ | Vulkan-image-backed textures via `acuda_image_create` (VkImage) + `acuda_sampler_create` (VkSampler) + `acuda_texture_create_image`. Geometries: `.1d/.2d/.3d/.a1d/.a2d/.cube/.acube/.2dms/.a2dms`; `tex` with `.base/.level/.grad` LOD modes, `.v4.{f32,s32,u32,f16}`, `.v2.f16x2`, constant offsets (`ConstOffset`) and gather offsets (`Offset`), depth-compare operand, destination predicates, nearest/linear filtering and wrap/mirror/clamp/border address modes. Independent texturing: `.samplerref` parameters bind standalone samplers and `OpSampledImage` pairs them with `.texref` images. Multisample: real 2D/2D-array MSAA storage (single mip level, float formats) filled from the source image; `.2dms` fetches the selected sample. `tld4.{r,g,b,a}` works for `.2d/.a2d/.cube/.acube` (incl. depth-compare gather). `txq` answers width/height/depth/array_size/num_mipmap_levels/num_samples plus the real sampler filter/address modes from per-binding runtime metadata, so it also works for txq-only kernels of any geometry. `istypep.{texref,samplerref,surfref}` answers from the parameter/register type. |

## 9.7.12 Surface

| § | Instruction | Status | Notes |
|---|---|---|---|
| 9.7.12 | `suld`/`sust`/`sured`/`suq` | ✅ | Vulkan storage-image surfaces: `.param .surfref` or a `.u64` surface-object handle binds a storage image (`OpTypeImage` `Sampled=2`; `R32ui` for `.b`, `Rgba32f` for `.p`). Handles that are not statically tied to one parameter use runtime-sized per-type storage-image tables (one descriptor set per surface type), so a kernel can select among live surface objects at runtime, including mixed geometries and formats in the same kernel. Supported: `suld.b`/`sust.b` for `.1d/.2d/.3d/.a1d/.a2d` with `.b8/.b16/.b32/.b64` and scalar/`.v2`/`.v4`; `sust.p` formatted sample stores; `sured.b.{add,min,max,and,or}.{u32,s32,b32}` via `OpImageTexelPointer` + atomics; `suq` width/height/depth/channel_data_type/channel_order/array_size/memory_layout from runtime metadata. Byte-addressed `.b` forms address 4-byte texels; 64-bit reductions and `.trap` faults are not represented (in-bounds accesses expected). |

## 9.7.13 Control flow

| § | Instruction | Status | Notes |
|---|---|---|---|
| 9.7.13.1 | `{}` blocks | ✅ | Scoped declarations/braces. |
| 9.7.13.2 | `@` predication | 🔶 | Pure ALU and trailing stores; predicated `ld`, atomics, calls, `ret`/`exit`, `bar`, u64 destinations remain todo. |
| 9.7.13.3 | `bra` | ✅ | Dedicated structured emitters for guards, diamonds, single-header loops, multi-block breaks/continues and sequential regions. Any other control flow (nested/multi-header loops, branches into loop headers, irreducible backedges, non-final returns) falls back to a per-block dispatch state machine, so arbitrary CFGs compile. |
| 9.7.13.4 | `brx.idx` | 🔶 | Forward-target jump tables via `.branchtargets`: register or immediate `.u32` selector, predicated form (false path continues after the instruction), `.uni` accepted, up to 64 targets. Lowered to an index-compare/conditional-branch chain. Backward/self targets, compact label-range syntax, and targets that would branch into a loop header remain todo/clean errors. |
| 9.7.13.5 | `call` | 🔶 | Direct calls inlined (`.param`/`.reg` forms, single ret); unresolved `.extern .func` calls lower to split-dispatch host callbacks. Indirect calls resolve through cross-block `mov`/`selp` chains or `.calltargets` lists, with single- and multi-target runtime dispatch (straight-line callees). Recursive and multi-return shapes remain todo. |
| 9.7.13.6 | `ret` | ✅ | |
| 9.7.13.7 | `exit` | ✅ | |
| 9.7.20.4 | `trap` | ✅ | Ends the invocation via `OpReturn`; `OpTerminateInvocation` is fragment-only, so it is not used in compute. |
| 9.7.20.1 | `brkpt` | ✅ | Emits an `OpString "<ptx>"` + `OpLine` with the PTX line (a debugger source-location aid); it is a runtime no-op because SPIR-V/Vulkan has no portable compute breakpoint. |

Warp-level instructions below (shfl/vote/redux/match/elect/warp barriers)
assume PTX's 32-lane warp. The runtime requests a 32-wide subgroup via
`VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` when the device
supports `subgroupSizeControl`; without it, 32- and 64-lane defaults are
accepted (64-lane waves are split into warp segments) and other sizes
are rejected at launch.

## 9.7.14 Parallel synchronization and communication

| § | Instruction | Status | Notes |
|---|---|---|---|
| 9.7.14.1 | `bar`/`barrier` | ✅ | `bar`/`bar.cta`, `barrier`/`barrier.cta`, `.sync`/`.arrive`/`.red` with optional `.aligned`; full-CTA and counted sync, non-blocking arrive, and `bar.red.{and,or}.pred`/`.popc.u32` (warp-scoped approximation). `barrier.cluster` is covered by 9.7.14.3. |
| 9.7.14.2 | `bar.warp.sync` | 🔶 | Full-warp membermask (`0xffffffff`) lowers to a subgroup control barrier; partial masks remain todo. |
| 9.7.14.3 | `barrier.cluster` | 🔶 | `.arrive{.relaxed/.release}{.aligned}` and `.wait{.acquire}{.aligned}` lower to a dedicated CTA-wide software barrier (counter slot 16). This is a single-CTA cluster approximation: Vulkan/SPIR-V has no cluster execution scope, so cross-CTA synchronization is not modelled. |
| 9.7.14.4 | `membar`/`fence` | ✅ | `membar.cta/gl/sys`, `fence{.sc/.acq_rel/.acquire/.release}.{cta,cluster,gpu,sys}`, `fence.mbarrier_init`/`sync_restrict`/`op_restrict` forms, and `fence.proxy`/`membar.proxy` (`.alias`, `.async`, address/size acquire forms) mapped to `OpMemoryBarrier`. `.sc` uses AcquireRelease (Vulkan forbids SequentiallyConsistent); cluster scope is a Workgroup approximation; proxy fences use device scope. |
| 9.7.14.5 | `atom` | 🔶 | Scalar `.b16/.b32/.b64` bit ops (`.b16` CAS via a u32 word CAS; `.b32/.b64` `.and/.or/.xor/.exch/.cas`), integer `.add/.inc/.dec/.min/.max`, `.add.f32`, `.add.f64` (CAS loop), and `.add.noftz.{f16,f16x2,bf16,bf16x2}` (CAS loops), over global/shared/generic. `.shared::cta`/`.shared::cluster` (cluster treated as CTA), `.cluster` scope, and `.L2::cache_hint` (policy parsed, hint dropped). 64-bit shared atomics work on `.u64`-typed shared variables (native Workgroup u64 atomics); 64-bit atomics on `.u32`/byte shared buffers and `.b128`/vector atomics remain todo (no 128-bit SPIR-V atomics; vector atomics are not implemented). |
| 9.7.14.6 | `red` | 🔶 | Integer/f32 scalar subset matching `atom` (global/shared/generic); vector, f64, and half `add.noftz` forms remain todo. |
| 9.7.14.7 | `red.async` | 🔶 | `red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes.{inc,dec,min,max,and,or,xor,add}.{u32,s32,b32,u64}` and `red.async{.mmio}.release.{gpu,sys}.global.add.{u32,s32,u64,s64}`. The reduction runs as a shared/global atomic (u64 via Int64Atomics); completion is immediate because the operation is synchronous, so the mbarrier gets the corresponding complete-tx byte count (4/8). The relaxed form's destination/mbarrier must be local shared memory: there is no cross-CTA cluster support, so the remote-CTA form is a single-CTA approximation (like `barrier.cluster`). Release forms are sm_100-gated. |
| 9.7.14.8 | `multimem.red.async` | 🔶 | `multimem.red.async.release.{gpu,sys}.global.add.{u32,s32,u64}` lowered as a single-device global release add; with one GPU a multimem address refers to that GPU's own memory. sm_100-gated. |
| 9.7.14.9 | `vote` (deprecated) | ✅ | `.all/.any/.uni/.ballot`. |
| 9.7.14.10 | `vote.sync` | ✅ | `.all/.any/.uni/.ballot`. |
| 9.7.14.11 | `match.sync` | 🔶 | `.any` over `.b32/.f32`; `.all` todo. |
| 9.7.14.12 | `activemask` | ✅ | `.b32`. |
| 9.7.14.13 | `redux.sync` | 🔶 | `.add/.min/.max` over `.u32/.s32`; other ops todo. |
| 9.7.14.14 | `griddepcontrol` | 🔶 | `.launch_dependents`/`.wait` accepted as no-ops; dependent-grid ordering is not modeled. |
| 9.7.14.15 | `elect.sync` | ✅ | Elects the lowest active lane in `membermask` (ballot + lowest-set-bit); `d`/`p` and the `_` sink are supported, including predicated use. |
| 9.7.14.16 | `mbarrier` | 🔶 | `.shared{::cta}` object as a packed 64-bit word (pending/expected/tx-count/parity/layout/valid) updated by CAS: `init`/`inval` (`.layout::{v0,v1}`), `expect_tx`, `complete_tx`, `arrive`/`arrive.expect_tx`/`arrive.noComplete` (optional count), `arrive_drop`, `test_wait`/`try_wait` (`parity`/state forms), `pending_count` (`.layout`), `check_layout.layout::{v0,v1}`, and `cp.async.mbarrier.arrive{.noinc}`. `try_wait` is a non-blocking test (the caller loops); `cp.async` and bulk copies are synchronous, so their arrive/complete-tx effects are immediate. `test_wait`/`try_wait` `.phase_type::{primary,conditional}` and the PTX 9.3 `reportPredicate`/`reportValue` operands remain clean errors. |

## 9.7.15-9.7.17 matrix summary

The dense `mma.sync` subset, a minimal `wmma` path, and sparse
`mma.sp` m16n8k16 are implemented today. `wgmma` and `tcgen05` are
target-gated to Hopper/Blackwell and remain todo.

## Summary

- **Done:** the integer arithmetic and extended-precision chapters,
  most scalar floating-point and half-precision arithmetic,
  logic/shifts, warp shuffles/votes, the dense `mma.sync` subset,
  `wmma` m16n16k16 f16->f32, sparse `mma.sp` m16n8k16, textures
  (9.7.11), surfaces (9.7.12), cluster barriers, and the core
  control-flow/memory subset.
- **Partly:** float `add/sub/mul/fma/mad/div/min/max/rcp/sqrt/rsqrt`
  modifiers and packed `.f32x2`; half `fma` `.relu` and
  `min/max .xorsign.abs`; data movement/cache/cvt/call/barrier
  subforms, indirect calls, forward `brx.idx` tables, atomics/`red`
  coverage, and the remaining `wmma`/`mma`/`mma.sp` shapes.
- **Todo:** mixed-precision add/sub/fma, remaining async/bulk memory
  ops, fabric, `mapa`/`getctarank` variants, backward `brx.idx` targets,
  `.b128`/vector atomics, tensormap/TMA, and
  the matrix families beyond the current dense `mma.sync`, `wmma`,
  and `mma.sp` subsets (block scaling, remaining shapes,
  `wgmma`, `tcgen05`).

## 9.7.15 Warp-level matrix operations

Target availability matters here: RTX 3070 is `sm_86`, so `mma.sync`,
`wmma`, and `mma.sp` are the testable families. `wgmma` requires
`sm_90a` (Hopper) and TensorCore 5th-generation instructions require
`sm_100` (Blackwell), so they cannot be validated on an RTX 3070.

| § | Instruction | Status | Notes |
|---|---|---|---|
| 9.7.15.3 | Block scaling for `mma.sync` | ❌ | Block-scale (`mxf8f6f4`/scale-vector) forms are not implemented; they are Hopper/Blackwell-era and not testable on sm_86. |
| 9.7.15.4 | `wmma` | 🔶 | f16/f32 forms of `.m16n16k16`, `.m8n32k16`, and `.m32n8k16`, plus `.m16n16k8.f32.tf32.tf32.f32` (tf32 A/B): `wmma.load.{a,b}...{f16,bf16}` (`.m16n16k16` bf16), `wmma.load.c...{f16,f32}`, `wmma.mma...{f32.f32,f16.f16}` plus the `.m16n16k16` bf16 form, and `wmma.store.d...{f16,f32}` over global/shared, using an internally consistent private fragment layout (PTX leaves the fragment identity target-dependent). Other types (f64/integer/sub-byte), mixed dtype/ctype pairs, and bf16/f16 accumulate for the m8n32k16/m32n8k16 shapes remain todo. |
| 9.7.15.5 | `mma` / `mma.sync` | 🔶 | Dense register-fragment forms: `.m16n8k16.row.col.f32.f16/bf16.f32`, `.m16n8k16.row.col.f16.f16.f16.f16` (f16 accumulators), `.m16n8k8.row.col.f32.f16/bf16.f32`, `.m16n8k8.row.col.f16.f16.f16.f16`, `.m16n8k4.row.col.f32.tf32.tf32.f32`, `.m8n8k4.row.col.f32.f16.f16.f32` (four lane-group MMAs), `.m8n8k4.row.col.f64.f64.f64.f64` (default `.rn`), and target-gated `.m16n8k4/k8/k16.row.col.f64.f64.f64.f64` (sm_90), and `.m8n8k16`/`.m16n8k16`/`.m16n8k32.row.col.s32.{s8,u8}.{s8,u8}.s32` (all four signed/unsigned A/B combinations). f16 accumulation is computed in f32 and rounded once to f16 (PTX permits any precision at least half and leaves rounding/order unspecified). Other shapes/layouts, f64 accumulation variants, fp8/sub-byte types, directed-rounding/saturate qualifiers, and block-scaling forms remain todo. |
| 9.7.15.6 | `mma.sp` (sparse) | 🔶 | `mma.sp.sync` / `mma.sp::ordered_metadata.sync` `.aligned.m16n8k16` and `.m16n8k32` `.row.col.f32.f16/bf16.f16/bf16.f32` with 2:4 sparse A, 2 A regs, 2 B regs, 4 C/D regs, one metadata `.b32`, and a 0..3 selector. A is expanded to dense fragments using the metadata bit layout from the PTX ISA; other shapes/types (k64/k128, tf32/f64/fp8/int, block scaling) remain todo. |
| 9.7.16.5 | `wgmma.mma_async` | ❌ | Hopper (`sm_90a`) warpgroup MMA; not testable on RTX 3070. |
| 9.7.16.6 | `wgmma.mma_async.sp` | ❌ | Hopper sparse warpgroup MMA; not testable on RTX 3070. |
| 9.7.16.7 | `wgmma` asynchronous proxy ops | ❌ | Hopper async proxy fences/commit/wait forms; not testable on RTX 3070. |
| 9.7.17 | TensorCore 5th generation (`tcgen05`) | ❌ | Blackwell (`sm_100`) tensor-memory/MMA family; not testable on RTX 3070. |

### Matrix next steps for RTX 3070 validation

1. Additional dense `mma.sync` types (fp8/sub-byte).
2. Sparse `mma.sp` k64/k128 and tf32/fp8/int forms.
3. Remaining wmma types (f64, integer, sub-byte).

`wgmma`, `tcgen05`, and `mma.sync` block scaling are explicit
target-gated errors (`wgmma requires sm_90a`, `tcgen05 requires sm_100`,
`block scaling is not implemented`), not silent translations. They stay
backlog until an `sm_90a`/`sm_100` device is available for validation.
