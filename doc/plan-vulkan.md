# Plan: the Vulkan backend

A GPU backend for cleave on Vulkan Compute, SPIR-V and cooperative matrices: open, vendor-neutral, never
CUDA (`doc/hld.md`, project values). Target machine: an NVIDIA RTX 4070 SUPER (Ada, 12 GB). This file
fixes the architecture; each phase gets its details when it starts.

## Decisions

- **The CPU backend doesn't change.** Not one byte of its output: the 18 examples' LLVM dumps stay
  identical (the comparison script used for every compiler change), and a CPU-only project pulls in no
  Vulkan dependency. The GPU backend is added beside it, never woven through it.
- **Precision is the program's choice, never the backend's.** A kernel is written in the precision its
  computation needs (f64 is right for some physics). The first version is f32 throughout; f16 and bf16
  come after, as real types usable on CPU and GPU alike, not as a silent GPU-side downcast. Where an op
  runs is the compiler's choice; what it computes is not.
- **Placement is inferred** (no `.to(device)`), with the program able to see and constrain it: a
  placement report, and warnings for host reads that synchronize inside a hot loop.
- **A fixed machine first, a general machine description in the architecture.** The first version
  targets one CPU and one GPU, named `CPU0` and `GPU0`, and a program pins what it must with
  `#[CPU0]`/`#[GPU0]`. Underneath, the compiler works on a machine description from the start
  (processors, their memories, the links between them, capabilities and rates), of which this is the
  built-in two-node instance, so that the general cases come later without redesign (below, "Later").
- **Placement and ordering are a scheduling problem with transfer costs** (HEFT, below), not a rule
  per op. Some scalar code may run on a GPU (one thread) to avoid a round trip.
- **Several GPUs need ops split, not only placed**: a partitioning level (sharding specifications,
  propagated GSPMD-style) before scheduling.
- **The runtime is ours**, in Rust: `ash` for Vulkan, `vk-mem` (the native VMA, compiled from AMD's
  C++) for memory. MLIR's Vulkan runner wrappers are not in our LLVM build and are a test tool anyway.

## Structure

Two sides, as `doc/hld.md` already splits them (descriptor on the host, payload wherever the backend
puts heavy data):

- **Host: unchanged CPS code**, compiled for the CPU as today: control flow, scalars, refcounts, calls,
  `spawn`, differentiation, the e-graph. The GPU never runs CPS logic.
- **Device: kernels**, one per fused group of tensor ops, cut at the `linalg` level after fusion:
  tiled to `scf.forall` mapped to `#gpu.block`/`#gpu.thread` by a transform schedule (the GPU twin of
  `matmul_vectorize.transform.mlir`), outlined (`gpu.launch` -> `gpu.module`), converted
  (`convert-gpu-to-spirv`), serialized to a SPIR-V binary embedded in the object as data. What stays on
  the host side of a `gpu.launch_func` becomes calls into the runtime (`cleave_vk_dispatch`, ...).

Everything compiler-side exists in our LLVM 22 build: `GPUToSPIRV`, `SCFToGPU`, `MemRefToSPIRV`, the
SPIR-V serializer, the `KHR_cooperative_matrix` ops.

### Code layout

- `cleave/src/pipeline.rs`: `Backend::Vulkan` beside `Backend::Cpu`; `lower_to_llvm` keeps the CPU
  stages as they are and branches to the Vulkan stages (`--backend vulkan`).
- `cleave/src/machine.rs` (new): the machine description (processors, memories, links, capabilities,
  rates); one built-in instance, `CPU0` + `GPU0` (the first Vulkan device).
- `cleave/src/placement.rs` (new): the scheduler below; its output is an annotation on the module the
  lowering reads. Never called with `Backend::Cpu`.
- `cleave/mlir/*_gpu.transform.mlir` (new): the GPU schedules.
- Shim passes for whatever MLIR doesn't do (the same `cleave-*` mechanism as the CPU's).
- `cleave-vk` (new crate): the runtime. `cleave-rt` stays Vulkan-free; a program built for Vulkan links
  both.

## Memory

- A tensor's descriptor stays a host memref; on the device side its pointer becomes a `VkBuffer` plus
  offset (or a buffer device address). The refcount stays on the host, atomic, unchanged.
- Payloads come from VMA: its linear pools where the CPU uses region arenas, its default pools for
  refcounted payloads (`doc/hld.md`, "Generalizing to Vulkan").
- Transfers are explicit operations the scheduler places (an arc with a cost), never implicit on
  access. A host read of a device tensor (`c[0, 0]`, `print`, an `export fn` boundary) is a download of
  what is read, after a wait on the work producing it.

## Asynchrony

The one concern with no CPU analogue:

- Work is recorded into command buffers and submitted per sequence (a training step: one submission);
  barriers between dispatches come from the exact dependencies the compiler has.
- Freeing waits for the GPU: a `cleave_release` reaching zero on a device payload queues the buffer
  behind a timeline semaphore value, reclaimed when it signals (the "future" `doc/hld.md` anticipated).
- The host waits only when it reads.

## Scheduling: HEFT with transfer costs

The units are the tensor ops (fused groups, or their shards, below) and scalar computations of one
function body; the arcs their data dependencies. Every shape is static, so every cost is computable at
compile time, against the machine description:

- a node's cost on each processor of the description: FLOPs and bytes against a machine description (peak
  rates, bandwidths; f64 on Ada at 1/64 of f32 makes physics in f64 stay on the CPU without a special
  rule), plus the GPU's per-dispatch overhead (~10 us), which keeps small tensors on the CPU;
- an arc's cost when its ends are on different processors: bytes over the link between their memories
  (nothing within one memory, PCIe, NVLink, network), plus the synchronization when it ends on a
  processor that waits for it.

HEFT, as published, is exactly this: P heterogeneous processors, a communication cost per pair. It
ranks nodes by upward rank (longest remaining path, average costs), then assigns each, in rank order,
to the processor giving the earliest finish time, transfers included; every processor runs
concurrently. Adaptations cleave needs:

- **Loops.** A loop body is scheduled as a steady state: a loop-carried value (a model's weights) must
  be placed the same at the end of an iteration as at the start, a constraint on its producer and its
  consumers in the next iteration. Placing the weights on the device once, before the loop, is what
  this yields for training.
- **Capabilities.** A node is only eligible on processors that can run it: element-by-element host
  loops, I/O and `extern` calls need a CPU (the one whose machine holds the resource), a kernel needs a
  device the backend compiles for, fast f64 is a rate in the description. Its operands pay the transfer
  from wherever they were produced.
- **Pins.** `#[CPU0]`/`#[GPU0]` on a function or a binding fix its processor, a name from the
  machine description; the rest is scheduled around them.

## Later: general machines

Designed for, not built now:

- **General descriptions**: several CPUs, NUMA zones as separate memories with their own links,
  several GPUs, several machines, given to the compiler like a target triple (`--machine <file>`), and
  obtained from the job's allocation on a cluster (Slurm: the nodes and devices a job was granted).
- **Pins become constraints** a source can keep across machines: colocation, capabilities, a resource
  named in the description.
- **A first heterogeneous testbed on the development machine**: `CPU0`, the 4070 SUPER and the
  Ryzen's iGPU make a real three-node description with an uneven cost matrix: the iGPU shares system
  memory with the CPU (transfers nearly free), the 4070 is across PCIe, and the two GPUs can only
  exchange through the host (no peer-to-peer across vendors). With the iGPU two orders of magnitude
  slower than the 4070, a correct scheduler gives it only small work next to CPU-resident data: a test
  that it doesn't spread naively.
- **Several machines: an `mpirun`/`srun`-launched binary.** One SPMD executable per rank: the compiler
  schedules the whole program over the cluster's description once, each rank executes its projection
  of that schedule (its rank selects its nodes), and an arc between ranks becomes MPI communication.
  The schedule being static, every transfer is known at compile time (size, source, destination,
  order): buffers preallocated, persistent requests (`MPI_Send_init`/`MPI_Start`), sends and receives
  emitted in matching order, so no deadlock by construction; collectives as nodes with their topology
  cost. GPU payloads go through GPU-aware MPI where available, staged through the host otherwise.
  Linked against MPI's standard ABI (MPI 5.0) if the target implementations provide it, so one binary
  runs under Open MPI and MPICH alike: to check when we get there.
- **Profile-guided costs**: costs measured on a run feed the next compile's schedule (the e-graph's
  "plausible variants" for PGO, `doc/hld.md`, fit the same loop).

### Partitioning: several GPUs

HEFT places indivisible nodes; with M GPUs one matmul of nanoLM must be split, or M-1 of them wait.
Before scheduling, a partitioning level decides how tensors and ops are split: sharding specifications
(which axis of a tensor is split over which group of processors), propagated along the graph
GSPMD-style from a few seeds (the batch axis for data parallelism, a weight axis for tensor
parallelism). Splitting introduces collectives (all-reduce of gradients, all-gather) as ordinary nodes
whose costs come from the topology; HEFT then schedules the shards and collectives. Pipeline
parallelism falls out of it: successive layers on different GPUs, overlapping. Across machines, the
network links are arcs like the others, the latency hidden by the futures `doc/hld.md` designs for
distribution.

Phase 1 starts with the degenerate case (two processors, every tensor kernel on the GPU, CPU-only
nodes on the CPU, transfers at the cuts) to have the pipeline end to end; HEFT replaces that rule once the costs are
measured, not guessed.

## Kernels

- **Elementwise** (after fusion): one thread per element.
- **Reductions** (sum, norms, softmax): subgroup operations (`GroupNonUniform*`) and workgroup shared
  memory. The real kernel-design work of the backend.
- **Matmul**: cooperative matrices (`spirv.KHR.CooperativeMatrixMulAdd`): workgroup tiles, subgroup
  tiles, 16x16x16 fragments. In f32 first, through the same structure without cooperative matrices
  (Ada's tensor cores take f16/bf16/tf32 operands; tf32 is a precision choice like the others, not a
  silent one). Which formats the driver exposes: `vulkaninfo`, checked, not assumed.

## Interop with Rust

An `export fn` whose loop runs inside cleave (nanoLM's `train_lm(rounds, ...)`) keeps its state on the
device for the whole call. A Rust program that drives a step at a time and passes the state back and
forth would pay a round trip per call: device-resident state needs an opaque handle on the Rust side.
To decide before Phase 4, with the binding generator (`--emit-bindings`).

## Errors

Out of device memory (12 GB), device loss, a missing extension or format: reported with the source
position of what needed it, never a crash; the placement report says where every tensor lives.

## Testing

- Correctness without a GPU: lavapipe (Mesa's software Vulkan, Khronos-conformant) on the Linux CI
  runners, which have no GPU. Whether its Mesa version exposes cooperative matrices is to check; if
  not, the CI tests the non-coopmat matmul fallback.
- Locally, two vendors: the development machine has the 4070 SUPER and the Ryzen's AMD iGPU (also
  Vulkan 1.4 with cooperative matrices). A kernel right on one and wrong on the other is almost always
  our bug (a missing barrier, a race one driver tolerates). The runtime picks the device explicitly
  (discrete first), never by enumeration order.
- The Vulkan SDK's validation layers on in every debug and test run.
- Every GPU test has a CPU twin: same program, same result within the tolerance of a different
  summation order.
- Performance on the 4070 SUPER, measured by the user.

## Phases

0. **Runtime and one kernel end to end.** `cleave-vk`: instance, device, a buffer, a pipeline from
   SPIR-V, a dispatch, a fence. The compiler side for one elementwise op (`a + b` on `Tensor<f32, N>`):
   kernel outlined, SPIR-V embedded, the host calling the runtime. Tested under lavapipe and on the GPU.
   **Done (2026-10-11)**: `cleave-vk` (`Gpu`, buffers, kernels, synchronous dispatch;
   `cleave_vk_dispatch` for generated code, copying each argument in and out); `serialize_spirv` in
   `cleave-mlir`; `--backend vulkan`: a "GPU kernels" stage after deallocation (upstream passes plus
   `cleave-tile-parallel-loops-for-gpu`, `cleave-set-spirv-entry-abi`, `cleave-lower-gpu-launches`),
   the CPU's matmul schedule left out. Tested on the 4070 SUPER and the AMD iGPU, validation layer on
   (`cleave-vk/tests/dispatch.rs`, `cleave/tests/vulkan.rs`); not yet under lavapipe (CI). Every
   remaining `linalg` op becomes a kernel, matmul included (naively, a thread per output element).
1. **Every tensor op on the device** (the degenerate placement), transfers at host cuts, reductions.
   **Done (2026-10-11)**, reductions aside. Residency without moving the payloads: the host copy stays
   the refcounted one (the payload's `RcHeader` sits next to it, and CPU atomics on write-combined BAR
   memory aren't atomic), and the runtime keeps a device mirror of each heap payload a kernel uses,
   uploaded only when the host wrote it since, downloaded only when the host reads it.
   - The compiler proves which kernel arguments are heap payloads (`isHeapPayload`: a `memref.alloc`,
     through views, loop-carried values, branch results and private functions' parameters); the
     rest (stack, globals, struct fields, public functions' parameters) is copied per launch.
   - `cleave-sync-host-accesses` reports every host access (`cleave_vk_sync_host` before,
     `cleave_vk_host_wrote` after a write), hoisted out of loops and branches that launch nothing;
     an address taken for anything but the runtime counts as an access (`sgemm`).
   - `cleave-rt`'s free hook (`set_free_hook`: a payload reaching refcount zero, an arena range a
     region exit rewinds) retires mirrors before an address is reused.
   - Kernel arguments are bound from the start of the buffer they view (a kernel applies a static
     offset itself), with read/write flags; a write-only argument isn't uploaded when the kernel
     covers all of it.
   - Copies between heap payloads run as kernels (`cleave-gpu-copies`), scalars are kernel arguments,
     `math` ops go through `convert-math-to-spirv`.
   - Every example prints the same on the GPU as on the CPU, on both devices, validation on
     (`cleave/tests/vulkan.rs`); `CLEAVE_VK_STATS=1` counts transfers, `CLEAVE_VK_TRACE=1` traces.

   Left: reductions to a scalar have no parallel dimension and run on the host (correct, slow);
   real GPU reductions (subgroups, shared memory) go with the kernel work of Phase 3.

   Struct-field tensors (2026-10-11): resident too. `descriptorsAreHeapPayloads`, a closed-world
   check once per module: every memref descriptor built (`llvm.insertvalue` of its allocated or
   aligned pointer) holds `cleave_alloc_rc`'s result, a pointer read out of another descriptor, null,
   or a heap memref's address; none comes from outside (no public function takes, no external
   function but cleave's runtime returns, no LLVM global holds, a pointer). Then a memref read out of
   any descriptor is a heap payload.

   MNIST on the GPU (2026-10-11): `cargo run --release -p mnist-interop --features vulkan`
   (`cleave-build` links `cleave-vk`'s staticlib, which holds `cleave-rt`, in place of
   `cleave-rt`'s). Correct, test accuracy 0.9331 (0.9332 on the CPU), and 744 s against about 10 s on
   the CPU, one run: naive matmuls (a thread per output element), reductions on the host, a
   synchronization at every host read. Phase 3's work.
2. **Asynchrony**: one command buffer per sequence, barriers, deferred frees.
   **Done (2026-10-11)**, in `cleave-vk`'s runtime alone. Dispatches, uploads and downloads are
   recorded into a batch (a full memory barrier between any two commands), submitted when the host
   needs a result (a synchronized read, a retired mirror) or after 256 commands, without waiting
   then. Mirrors live in device-only memory and move through staging buffers: an upload is the
   host's bytes copied into a fresh staging buffer and a recorded copy, so it never waits for work
   in flight; a download is a recorded copy into host-cached memory, then one wait. Buffers and
   descriptor pools a batch uses are kept until its fence signals (a freed payload's mirror
   included). Only a copied (not mirrored) argument the kernel writes still waits at its dispatch:
   its memory may be the caller's stack. `x = x + w` ten times: 21 dispatches, 1 submit; `xor`
   (376 252 tiny kernels): about 60 s to about 10 s, one indicative run. Timeline semaphores and
   overlapping several batches in flight with the host's work come when a measurement asks for
   them.
3. **Matmul** with cooperative matrices, and its fallback.
   Started with measurement (2026-10-11): `CLEAVE_VK_PROFILE=1` prints at exit each kernel's
   dispatches and device time (Vulkan timestamps around each dispatch), the waits, the time in the
   runtime, the transfers and the host synchronization calls. On MNIST it led, in order, to:
   - arena payloads (`cleave_alloc_local`, freed in bulk at region exit, which the free hook
     reports) admitted by the descriptor check: one such allocation had failed it for the whole
     module, so no struct-field tensor was mirrored (92 GiB uploaded);
   - `cleave-parallel-iterators-first`: a `linalg.generic` reducing over rows (`reduce0`) put its
     reduction loop on the host, launching a kernel per row;
   - refcount calls (`cleave_retain`/`release`) not counted as escapes of the address they get;
   - a cache of recent ranges per thread, so a host synchronization the compiler couldn't hoist out
     of a loop costs a lookup, not the lock;
   - no host tasks on the Vulkan backend (`CodegenOptions::host_tasks`): spawned optimizer steps only
     contended to record into one queue, libomp's waiting took most of the host's time;
   - unfolded steps made `convert-parallel-loops-to-gpu` leave loops on the host silently
     (learning-rate fills, element by element): `cleave-tile-parallel-loops-for-gpu` now folds the
     steps and bounds it computes (`affine.min` and `arith` patterns on those ops only; a global
     `canonicalize`, tried first, merged the nested loops and mapped everything to blocks of one
     thread); a warning says so if a loop is ever left;
   - a workgroup's threads run along the dimension indexing its accesses' contiguous dimension.
   744 s became 20 s (the CPU: about 10 s), one submit per step, only the input batches uploaded,
   9.7 s of device time, the matmuls first (one thread per output element: 34 to 140 GFLOP/s).
4. **MNIST, then nanoLM on the GPU**; the PyTorch twin on CUDA as the speed reference only.
   nanoLM (2026-10-11): `--features vulkan` (checkpoints in a `-vulkan` directory, never the CPU
   run's) and `--features small` (d256, 4 layers, context 256), built in target directories of
   their own. Correct: the small model's losses equal the CPU's to the printed digit after 5 steps.
   Getting there fixed, among others: `--define` overrides applied after comprehensions were
   unrolled (a CPU bug too); views at run-time offsets sunk into kernels
   (`cleave-sink-views-into-launches`, `fold-memref-alias-ops`); "a kernel that doesn't read an
   argument needn't have it uploaded" replaced by `cleave_vk_discard`, emitted where an op
   overwrites a fresh, unaliased allocation it doesn't read (an in-place `o * w` had dropped the
   host's `o`: attention gradients 3 to 15 times off, found with a miniature transformer whose
   gradients are now a test); `sin`/`cos` in kernels as MLIR's polynomials (`GLSL.std.450`'s are
   relaxed precision); the copy path made mirror-aware. Slow: 15 s/step against 0.15 s on the CPU,
   most of it host loops over tensor elements (`causal_softmax`, `rope_rotate`, ... written as
   loops in the stdlib: billions of host synchronizations) and `sgemm` (BLAS, on the host).
   Then, still exact: `cleave-blas-to-linalg` (an `sgemm` call made `linalg` ops on the memrefs its
   pointers come from, a kernel like any other); `cleave-parallelize-loops` (cleave's counted loops
   raised to `scf.for` by MLIR's `UpliftWhileToFor`, the outermost of a nest with independent
   iterations made `scf.parallel`: nothing carried, no call, every written buffer a fresh unaliased
   allocation indexed by the induction variable at one position; the stack arrays cleave builds a
   multi-index access's indices in forwarded to their values first). 2.5 s/step;
   `CLEAVE_DEBUG_PARALLELIZE=1` says why a loop stays sequential. Left: reductions to a scalar
   (`squared_norm`, `sum`), the dispatch count (about 170 000 a step), the matmul kernels.
5. **HEFT placement** on the two-node machine, with measured costs, the placement report, the pins.
6. **f16/bf16** as types, CPU and GPU.
7. Later: general machine descriptions, partitioning over several GPUs and machines, cluster
   allocation, profile-guided costs.
