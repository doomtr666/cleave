# Plan: fewer bits, by the program's choice — `bf16` first, then fp8 and quantized storage

Status: phase 0 done (GPU), phase 1 mostly done (2026-10-15); the rest proposed.

## The rule

Precision is the program's choice, never the backend's (`doc/plan-vulkan.md`, Decisions). `f32` means
real f32 everywhere: no TF32 behind the user's back, no silent downcast on the GPU. A program that
writes `Dense<bf16, In, Out>` asks for bf16, and then the backend picks the best instruction for it:
AVX-512 BF16 (`vdpbf16ps`) on the CPU, cooperative matrices (tensor cores) on Vulkan. Where a device
can't do what the program asks (bf16 cooperative matrices on a GPU without them), that is a reported
error or a reported, explicit fallback the user allowed, never a quiet change of precision.

## Why now

nanoLM's step is ~65-70% matmuls (profile of 2026-10-15), already at cuBLAS parity in f32: f32 has
nothing left to give there. bf16 is the lever: twice the compute density on tensor cores, half the
bytes everywhere, half the activation memory (room for larger batches or contexts).

Expected gain, honestly: on GeForce Ada, fp16/bf16 tensor throughput *with f32 accumulation* is half
the advertised (f16-accumulation) figure, so about **2x the f32 peak**, not 4-8x (to be measured,
phase 0). With ~1.6-1.8x reached on the matmuls and halved traffic elsewhere, **~1.3-1.5x on the
whole step**.

## What the machine offers (vulkaninfo, 2026-10-15)

- **NVIDIA GeForce RTX 4070 SUPER**, driver 610.88: `VK_KHR_cooperative_matrix` (rev. 2),
  `VK_KHR_shader_bfloat16` with `shaderBFloat16Type` and **`shaderBFloat16CooperativeMatrix`**
  (`shaderBFloat16DotProduct` false), `VK_EXT_shader_float8` with **`shaderFloat8CooperativeMatrix`**,
  `VK_NV_cooperative_matrix2`, `VK_KHR_shader_float16_int8`, `VK_KHR_buffer_device_address`,
  `VK_EXT_device_generated_commands`.
- **AMD Radeon(TM) Graphics** (the Ryzen 9700X's iGPU), AMD proprietary driver 26.8.1:
  `VK_KHR_cooperative_matrix` too, no bf16 features listed. A second vendor on the same machine: the
  vendor-neutral claim can be shown, not only stated.
- **MLIR 22's SPIR-V dialect** knows `SPV_KHR_bfloat16` and the `BFloat16TypeKHR`,
  `BFloat16CooperativeMatrixKHR` capabilities, and the `KHR_cooperative_matrix` ops.
- **Zen 5**: AVX-512 BF16 (`vdpbf16ps`: pairs of bf16 multiplied, accumulated in f32).

## Phases

### 0. Probes (before building anything)

- The cooperative-matrix shapes and types each device supports
  (`vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`: M, N, K, A/B/C/result types, scope), added to
  `cleave-vk`'s device info and printed by a test.
- A hand-written MLIR kernel: a 16x16x16-fragment bf16 product with f32 accumulation through
  `gpu.subgroup_mma_*` (or the SPIR-V cooperative-matrix ops directly), lowered by our pipeline,
  accepted by the validation layer, correct against the CPU. What MLIR's `convert-gpu-to-spirv` does
  with bf16 fragments, and what has to be ours.
- Its throughput on a large product, against the f32 SIMT kernel: confirms (or not) the ~2x.
- On the CPU: `linalg.matmul` bf16 x bf16 -> f32, and what lowers to `vdpbf16ps` (`x86vector`'s
  bf16 dot op), with `target-cpu=native`.

Findings written here; they decide phases 3 and 4.

**Found (2026-10-15).** `cleave_vk::DeviceInfo::cooperative_matrices`, printed by
`vulkan.rs::cooperative_matrix_shapes_are_listed_per_device` (all subgroup scope):
- RTX 4070 SUPER, 15 products. **bf16: one, `16x16x16 bf16*bf16+f32 -> f32`** — exactly the
  accumulation the plan wants, a single fragment shape. f16: 16x16x16, 16x8x16, 16x8x8, accumulating
  in f16 or f32. fp8 (e4m3, e5m2): 16x16x32, accumulating in f16 or f32. int8/uint8: 16x16x32 and
  16x8x32 into 32-bit integers.
- Radeon iGPU, 11 products: f16 16x16x16 accumulating in f16 or f32 (one saturating variant), int8
  and uint8 in every signedness mix 16x16x16 into i32 (saturating or not). **No bf16, no fp8.**
- The product both vendors share: **`16x16x16 f16*f16+f32 -> f32`**, the vendor-neutral demo's
  format. A bf16 program on the iGPU gets the located error of phase 4.

(`ash` 0.38 predates the bf16 and fp8 component types; their values come from the Vulkan registry,
`cleave-vk`'s `component_type_name`.)

**A bf16 fragment product runs (2026-10-15).** `cleave-mlir/tests/data/mma_bf16.mlir`, serialized by
our `serialize_spirv`, run by `cleave-vk` (`tests/coopmat.rs::a_bf16_fragment_product_runs_on_tensor_cores`):
exact against the CPU on the RTX 4070 SUPER, validation layer silent, the iGPU skipped. What it took:
- MLIR 22's `!gpu.mma_matrix` rejects bf16 ("elements must be SI8, UI8, I32, F16, F32, or F64"),
  so the `gpu` dialect's subgroup-MMA path is closed for bf16: phase 4 emits the
  `spirv.KHR.CooperativeMatrix{Load,MulAdd,Store}` ops itself (the SPIR-V dialect accepts bf16 as
  `SPIRV_BFloat16KHR`).
- SPIR-V 1.6 with the Vulkan memory model (cooperative matrices in a shader require it, `spirv-val`
  says so), every interface variable listed on the entry point (1.4+); our other kernels stay 1.3.
- The device features enabled: `cleave-vk` now enables, where present, `cooperativeMatrix`,
  `shaderBFloat16Type`/`shaderBFloat16CooperativeMatrix` (`VK_KHR_shader_bfloat16`, its structure
  declared by hand: newer than `ash` 0.38), `storageBuffer16BitAccess`, `shaderFloat16`,
  `vulkanMemoryModel` (`OptionalFeatures`); existing kernels unaffected (`vulkan.rs` green).

**Throughput (2026-10-15, measured by the assistant, to confirm):** `mma_bf16_peak.mlir` (fragments in
registers, 4 independent accumulators per subgroup, no memory traffic), `tests/coopmat.rs::
bf16_tensor_core_throughput` (ignored by default): **69.0 TFLOPS** bf16 x bf16 + f32 on the RTX 4070
SUPER — the GeForce Ada figure for 16-bit inputs with f32 accumulation (half the f16-accumulation
one), ~1.95x the card's ~35.5 TFLOPS f32 peak. Against what our f32 matmuls reach (19.6 TFLOPS,
cuBLAS parity, ~55% of peak), the same fraction of the bf16 peak would be ~38-48 TFLOPS: ~2-2.4x on
the products, ~1.5-1.6x on nanoLM's step, if phase 4 feeds the tensor cores as well as the f32 kernel
feeds its units — the open question, since this peak has no memory traffic at all.

**A real GEMM through `VK_KHR_cooperative_matrix` (2026-10-15): ~59-63 TFLOPS, 3x our f32.** Probes
written in GLSL (`glslang` 16.2: `GL_KHR_cooperative_matrix` + `GL_EXT_bfloat16`), measured by
`tests/coopmat.rs::tiled_bf16_gemm_probe` (`CLEAVE_COOPMAT_SPV`; rectangular tiles via
`CLEAVE_COOPMAT_TILE_M`/`_N`; A, B and C in device memory, uploaded through `Gpu::copy`; sampled
entries checked exactly), 4096^3, RTX 4070 SUPER, every result exact:
- Ours (shared-memory tiles 128x128, BK 32, 8 subgroups of 64x32, register prefetch, loops unrolled):
  **58.7 TFLOPS**.
- The reference: NVIDIA's own `vk_cooperative_matrix_perf` shared-memory shader (Jeff Bolz, MIT,
  github.com/jeffbolznv/vk_cooperative_matrix_perf), adapted to the harness's bindings only:
  **62.7 TFLOPS** at 128x128x32 (51-56 at other tiles).
- `VK_NV_cooperative_matrix2` (its `workgroup.comp`: workgroup-scope matrices loaded through tensor
  views, the driver doing the tiling): **64.3 TFLOPS** at 64x128x64 with 128 threads, 54-58 elsewhere,
  2.1 at 256x256 (a cliff: tile sizes must be tuned). **The vendor-neutral KHR path is within a few
  percent of it**, so no NVIDIA-specific path is needed.
- So a bf16 GEMM reaches 80-90% of the measured peak (69-78 TFLOPS, boost-dependent), ~3x the 19.6 of
  our f32 kernels: better than this plan's estimate.
- A lesson, kept here because it cost an evening: the harness first allocated its matrices with
  `Gpu::buffer` (mapped, `HOST_ACCESS_RANDOM`: system memory), so every variant read A and B across
  PCIe and all plateaued at 10-18 TFLOPS; the dissections of that evening (barrier cost, an
  "unexplained" ceiling) measured PCIe, not the kernels, and are void. Calibrating against an external
  reference exposed it. `cleave-vk` gained `Gpu::copy` and, along the way, the driver's per-pipeline
  statistics (`Gpu::kernel_statistics`, `VK_KHR_pipeline_executable_properties`: registers, shared
  memory; "Local Memory Size" reads 2^36 for every kernel on NVIDIA, a sentinel) and the
  `VK_NV_cooperative_matrix2` features (enabled where present, for measurement).
- Not done: an instruction-level profile (Nsight Graphics' `ngfx` CLI crashed in its Qt front end).

Phase 4 therefore targets `VK_KHR_cooperative_matrix` with the structure of the shader above (the
reference's), generated by `cleave-tile-matmuls-for-gpu` as SPIR-V cooperative-matrix ops, tile
shapes searched by the autotuner. The f16 KHR path is the one to measure on the iGPU (both vendors).

### 1. `bf16` as a real type

**Status (2026-10-15): scalars and CPU tensors done; checkpoints, `Init` and Rust bindings left.**
- The compiler's scalar lists replaced by one, `cleave/src/scalars.rs` (`FLOAT_TYPES`: `bf16`,
  `f16`, `f32`, `f64`): inference, CPS, e-graph literals and seeds, monomorphization, the registry's
  literal parsing. `f16` is one stdlib file away.
- `stdlib/num`: `Float`, `Additive`, `Ring`, `Ord` on `bf16` in bf16 arithmetic (each result
  rounded); `Transcendental` computed in f32 and rounded. `stdlib/convert`: `bf16` to and from `f32`,
  `f64`, and from `i32`. `stdlib/io`: `Print<bf16>` (as the f32 it widens to).
- Products of 16-bit floats accumulate in f32, rounded once (phase 2's rule, settled as `matmul`'s
  semantics for `bf16`: decision of 2026-10-15, option A — `bf16 x bf16 -> bf16`, the PyTorch and
  hardware semantics; an f32 result is the program's explicit choice, through `Fma` with an f32
  accumulator, still to add): `mlir_lower.rs::widened_16_bit_float_tensor` computes the product in
  f32 and `arith.truncf`s it; `build_matmul_named_op` builds the mixed-precision body (`arith.extf`
  of narrower inputs); never the BLAS tier.
- The Vulkan backend refuses 16-bit float tensors with a message (`pipeline.rs`): its 32-bit tiling
  read a `bf16` product wrong without failing, before phase 4.
- Tests: `mlir_lower.rs::bf16_arithmetic_rounds_to_bf16`, `bf16_products_accumulate_in_f32` (259 ones
  sum to the bf16 260, where bf16 accumulation would give 256; a 512^3 product above
  `BLAS_MIN_WORK`), `vulkan.rs::bf16_tensors_are_refused_on_the_gpu`.

Original scope:

- `stdlib/num`: `#[mlir_type(bf16)] impl Float<bf16>`, `Additive`/`Ring`/`Transcendental` impls. Each
  bf16 operation rounds to bf16, as the type says; programs that want f32 arithmetic convert
  explicitly (`.to()`, `arith.extf`/`truncf`).
- The compiler's hard-coded scalar lists (`infer.rs`'s primitive names, `registry.rs`'s float names,
  `rust_bindings.rs`: bf16 crosses to Rust as its `u16` bits), literals (`1.5: bf16`), printing,
  checkpoints (save/restore bf16 tensors, a format bump if needed), `Init`.
- Tests on the CPU: arithmetic against f32 rounded to bf16, conversions both ways, round-to-nearest-
  even, a checkpoint round trip.

### 2. The accumulation type is part of the algebra

A product of bf16 tensors accumulates in f32: `MatMul<Tensor<bf16, M, K>, Tensor<bf16, K, N>>` yields
`Tensor<f32, M, N>`, exact to the instruction's semantics on every backend. Storing the result as bf16
is the program's explicit conversion. Same for reductions (`sum` of bf16 accumulates in f32) and the
norms. `nn`'s layers generic over the element type (`Dense<T, In, Out>`), so `Dense<bf16, ...>` is a
one-word change for a model.

### 3. CPU: AVX-512 BF16

The matmul schedule's inner contraction on bf16 inputs lowered to `vdpbf16ps` (phase 0 says how),
elementwise bf16 ops vectorized with conversions. Tests: products against an f32 reference within
bf16 tolerance, on shapes that hit the tails.

### 4. GPU: cooperative matrices

`cleave-tile-matmuls-for-gpu` gains fragment tiles: shared-memory tiles as today, then per subgroup a
grid of cooperative-matrix fragments (the device's M/N/K from phase 0), `MulAdd` with f32
accumulators, store. `ProductTiling` gets the fragment shape as a dimension; the autotuner
(`bench/gpu_matmul/tune.rs`) searches it, and `matmul_table.inc` gains the element type as a key. The
existing work skipping (dead output tiles, zero k-slices) applies unchanged, at tile granularity.
Requires the device features at pipeline creation; a device without bf16 cooperative matrices (the
iGPU, by phase 0's answer) gets a located error, or, only where the user allowed it, the SIMT kernel
on bf16 inputs. Tests: `vulkan.rs`'s product tests in bf16 against the CPU, on both devices.

### 5. Mixed-precision training, written by the user

bf16 has 8 bits of mantissa: an update smaller than 2^-8 of a weight vanishes. The standard recipe
keeps an f32 master copy in the optimizer and computes in bf16 (no loss scaling: bf16 has f32's
exponent range). In cleave the program says it: an optimizer wrapper (`Mixed<Muon>`) holding the f32
master and the bf16 working copy, or a parameter type distinguishing storage from compute. Muon's
Newton-Schulz in bf16 as the speedrun does, by the same explicit choice. Then nanoLM in bf16 against
f32 at equal tokens, judged in bits per character on the validation set (and the user's own timing),
with the PyTorch twin under `autocast(bfloat16)` as the reference.

### 6. Later: fp8 and quantized storage

- fp8 (E4M3/E5M2) cooperative matrices, which this GPU supports: same mechanism, with per-tensor or
  per-block scales (OCP Microscaling MXFP8), for inference first.
- Inference quantization: int8/fp8 with block scales, then the compressed-tensor storage algebra
  decompressed per tile, then codebooks (`doc/backlog.md`: "Compressed tensors, decompressed per
  tile", "Fewer bits per weight" — QuIP#'s Hadamard + E8, QTIP's trellis).
- Learned precision (`doc/backlog.md`, "Learned precision"): where the bits go, chosen by training
  within a budget the user sets.

## Risks

- MLIR's bf16 cooperative-matrix lowering may be partial (phase 0 says how much is ours to write).
- Accuracy: bf16 elementwise chains (norms, softmax) lose precision; the recipe keeps those in f32,
  and the program writes where it converts. `--explain precision` (backlog) would make this visible.
- The two devices differ: bf16 on NVIDIA only. The iGPU is the vendor-neutral check for f32 and fp16
  paths, not for bf16.
