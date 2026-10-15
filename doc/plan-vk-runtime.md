# Plan: a simpler, predictable Vulkan runtime

Status: proposed, 2026-10-15. Nothing implemented yet.

The Vulkan backend works and trains nanoLM at about PyTorch-CUDA speed, but its recent optimizations
were driven by nanoLM's profile (matmul tables, attention folding). This plan goes the other way: the
runtime and the kernel boundary themselves, generic, measured on programs that are *not* nanoLM. The
target is an architecture that is simpler (less runtime code, fewer Vulkan concepts in play),
predictable (dispatches, barriers and transfers known from the program), and a better base for every
later optimization, ML or not. It converges the runtime toward what `doc/plan-vulkan.md` intended
("Asynchrony": barriers from the compiler's exact dependencies, timeline semaphores, transfers as
explicit operations) and which the first implementation approximated at runtime.

## Where the runtime stands

`cleave-vk/src/lib.rs` (2100 lines) emulates unified memory. Every payload is allocated and owned on
the host; the device holds a *mirror* of it, keyed by host address (`Runtime::mirrors`, a `BTreeMap`),
with `host_dirty`/`device_newer` flags, overlap queries for copied arguments, generations and
thread-local caches to avoid redundant synchronizations, LRU eviction when device memory runs out,
spare-buffer pools, and per-batch `HashSet`s of buffers read and written to place barriers. Each
`cleave_vk_dispatch` looks the kernel up by `(SPIR-V address, entry name)` in a `HashMap` keyed by a
`String`, resolves every argument to a mirror or a fresh copy, writes push descriptors (or allocates
a descriptor pool per dispatch when push descriptors don't apply), records, and submits every 256
commands behind a fence.

What a profiled nanoLM run (3 steps at `VOCAB` 8192 plus two evaluations, `CLEAVE_VK_PROFILE=1`,
2026-10-15) shows of it:
- 726,663 dispatches; 158,300 under 5 µs and 276,624 of 5-20 µs: 12% of device time in tiny kernels.
- ~4.2 µs of host time per dispatch outside waits: kernel lookup 800 ms, mirrors 540 ms, descriptors
  150 ms, recording 965 ms.
- 36% of commands after a barrier had no buffer hazard.
- 3.2 GB uploaded and 1.9 GB downloaded in 28 s; 2.8 GB of device memory in mirrors whose host copies
  also live in RAM.

nanoLM hides most of this behind large matmuls. A scientific code made of small stencils, particle
updates or solver iterations on modest grids would be dominated by it: dispatch overhead and
barriers, not arithmetic.

## Principles

- **Generic, measured on a suite that isn't ML** (below, "Measuring"). nanoLM stays a regression
  check, not the target.
- **Simplify before optimizing**: each step removes runtime machinery, or replaces it with a fact the
  compiler already has.
- **Predictable**: what can be known from the program (placement, transfers, dispatch order,
  barriers) is decided at compile time and reported; the runtime executes it.
- **Less noise in the system, not only less time.** Every dispatch avoided removes a barrier, its
  share of command traffic over PCIe, DMA fetches, driver and WDDM scheduler work, and lines in a
  trace. What it frees serves what matters (the bus for data, the copy engines for transfers, the CPU
  for host work), and a dataflow of thousands of dispatches is one a person can still read and debug;
  hundreds of thousands isn't. Phases are judged on these counts as much as on wall time.
- **Fewest dispatches first, then cheap ones, then GPU-generated ones.** Every technique that
  removes a dispatch comes first (fusion, folding loops and independent `spawn`s into grids); what
  remains is made cheap (recorded and replayed, device addresses, no descriptors); what remains and
  depends on data is generated on the device (indirect dispatch, device-generated commands: one host
  command for N dispatches). The compiler knows which sequences are fixed and which are data-dependent,
  chooses accordingly, and reports it.
- **No megakernel.** One persistent kernel with a work queue takes the register pressure of its
  greediest part, can't be preempted (a dispatch over ~2 s trips Windows' TDR and resets the GPU),
  and is opaque to profiling and to explanation. Recorded sequences of ordinary dispatches give the
  host-side savings without those costs.
- **The CPU backend doesn't change** (`doc/plan-vulkan.md`, Decisions).

## 1. Device-resident memory instead of mirrors

Invert the ownership: a payload the compiler places on the device is *allocated* on the device (VMA)
and lives only there; the host keeps the descriptor and the refcount, as today. Host access becomes
an explicit operation the compiler emits where it already knows a host read or write happens (the
`cleave_vk_sync_host` / `cleave_vk_host_wrote` sites): a download of what is read, an upload of what
is written, through a staging ring.

Removes: the mirror map and its overlap queries, the dirty/newer flags and generations, the
thread-local range caches, copied arguments, and the double residency (host RAM plus device memory)
of every device tensor. Eviction under memory pressure becomes the allocator's problem (an explicit
spill policy, reported), not a side effect of mirroring.

Needs: placement known per allocation site. The compiler already decides which tensor ops become
launches; allocations feeding only launches are device allocations, those read on the host are host
ones, and a tensor crossing gets an explicit transfer (`doc/plan-vulkan.md`, Memory: "transfers are
explicit operations the scheduler places, never implicit on access"). A first version can keep
mirrors as the fallback for what placement can't decide, and shrink their use step by step.

## 2. Buffer device addresses instead of descriptors

`VK_KHR_buffer_device_address` (core in Vulkan 1.2): a kernel receives 64-bit device pointers, in
push constants (128 bytes guaranteed: up to 16 pointers), or for more arguments in a small argument
buffer whose address is itself pushed. SPIR-V side: the `PhysicalStorageBuffer64` addressing model
and storage class, which MLIR's SPIR-V dialect supports; `cleave-lower-gpu-launches` passes pointers
instead of binding buffers.

Removes: descriptor set layouts, push descriptors, descriptor pools per dispatch, `VkBuffer`
bookkeeping per argument. A view or a sub-allocation is a pointer plus an offset, so `viewRoot` and
the cast look-through in `cleave-lower-gpu-launches` become pointer arithmetic, and arenas become
one buffer with offsets. Dispatch cost: bind pipeline, push constants, dispatch.

Risk to check first: alignment and aliasing rules of physical storage buffers (`Aliased`/`Restrict`
decorations: our tensors don't alias, which is the fast case), and SPIR-V/driver support on the
target devices (NVIDIA and RADV/AMD both support it on every GPU we care about).

## 3. Timeline semaphores instead of fences

One timeline semaphore per queue (core in 1.2): each submission signals the next value; garbage and
deferred frees carry the value after which they're free; the host waits for a value only when it
reads. Replaces per-batch fences and `retire_finished`'s polling with integer comparisons, and opens a
clean path to a separate transfer queue overlapping copies with compute (the staging ring of section
1), synchronized by values rather than by fences and binary semaphores.

## 4. Barriers from dependencies, not from a full barrier after any hazard

Today: a full memory barrier whenever the next command touches a buffer the batch read or wrote since
the last one (a conservative runtime approximation; 36% of barriers had no hazard at all). Steps:
- **Cheap exact runtime tracking first**: with device addresses and allocation ids, hazards per
  argument range (read-after-write, write-after-read, write-after-write), compact ids instead of
  `HashSet<vk::Buffer>`, a barrier only on a real hazard, scoped to what's needed.
- **Then compile time**: within a function, the compiler knows each launch's read and written
  arguments (`ARG_READ`/`ARG_WRITTEN` are already emitted) and whether two allocations can overlap;
  it can emit the barriers itself between launches, and the runtime only handles what crosses
  function boundaries.

## 5. Recorded dispatch sequences

A whole-program compiler knows the sequence of launches of a loop body with static shapes (a training
step, a time step of a simulation, a solver iteration): record its command buffer once, replay it
each iteration. Host cost per dispatch drops to nearly nothing; the device does exactly the same work
(this is not a megakernel). Addresses must be stable between replays, or indirected through an
argument table the host rewrites before each replay (a few kilobytes): the allocator's reuse pattern
decides which. Comes after sections 1-4, which make it possible (stable device addresses, no
per-dispatch descriptors, barriers known up front), and only where measurements show host time still
matters.

## 6. Kernel fusion as loop fusion

Fewer dispatches and more arithmetic per byte: two consecutive `scf.parallel` nests with compatible
iteration spaces, where no dependence forbids it, become one kernel. The legality question is the one
`cleaveParallelizeLoops` already answers with `AffineForms`/`AffineSystem` and Presburger emptiness
(`accessesAreDisjoint`, `mayReach`): one more pass before launches are formed, reusing it.
- First the simple cases: elementwise chains, elementwise into a row reduction, broadcasts; the
  norms, activations, residuals and optimizer steps of ML, and successive stencil sweeps or particle
  updates elsewhere.
- Then fusion across reductions (a reduction feeding a broadcast feeding a product), of which flash
  attention is the hardest instance: after the simple cases, not before
  (`doc/backlog.md`, "Polyhedral scheduling at the MLIR level: fusion across reductions").
- A diagnostic for each fusion done or refused, with the dependence that refused it.

## 7. Independent `spawn`s fold into grids

On the CPU, nanoLM splits each batch into `SPLIT` micro-batches so that `spawn` spreads them over
threads. The Vulkan backend has no host tasks, so the `spawn`s run one after another, each with its
own stream of small dispatches: the same work in `SPLIT` times more dispatches of `SPLIT` times
smaller kernels. A profiled step (2026-10-15): ~1,200 dispatches per micro-batch gradient, 64
micro-batches per optimizer step (`ACCUM` 8 × `SPLIT` 8), ~120,000 dispatches per step; at a few µs of
fixed device cost each, plus the drains at barriers, a few hundred ms of a 5.6 s step. On the GPU,
independent tasks of a `spawn` comprehension become one more grid dimension, as
`cleave-fold-loops-into-grids` already does for host loops, under the same memory cap (`foldedBytes`,
grouped by the largest divisor that fits): one dispatch for the group. Generic: any data-parallel
`spawn` written for CPU threads stops costing dispatches on the GPU.

## 8. Device-generated work, where the dispatch stream depends on data

- `vkCmdDispatchIndirect` (core): grid sizes read from a device buffer a previous kernel wrote; no
  host round trip for a computed size (stream compaction, adaptive meshes, active-element counts).
  Still one command per dispatch.
- `VK_EXT_device_generated_commands` (released late 2024; NVIDIA, RADV): a kernel writes a sequence
  of commands (dispatches, push constants, pipeline switches through execution sets) and one host
  command executes it, `vkCmdExecuteGeneratedCommandsEXT`. Control flow moves to the device while the
  work stays ordinary dispatches: preemptible, profilable, unlike a megakernel. Driver maturity and
  profiling support to check on the target devices first.
- Watch, don't build on yet: work graphs (true device-side dataflow, a kernel enqueuing work for other
  nodes; production in D3D12, only `VK_AMDX_shader_enqueue`, experimental, on Vulkan).

## 9. Small work stays on the CPU

A kernel too small to amortize a dispatch (and the transfers around it) runs on the CPU: a cost model
from shapes, the dispatch overhead measured once per device, and a placement report saying where each
op ran and why (`doc/plan-vulkan.md`, "Scheduling: HEFT with transfer costs").

## Measuring

A GPU performance suite next to `cleave/tests/examples.rs`, deliberately not ML-shaped, each program
reporting dispatches, host µs per dispatch, barriers, bytes up and down, device time and occupancy
(the profile's existing counters):
- a diffusion or Jacobi stencil on a modest grid (e.g. 512²), thousands of sweeps;
- an N-body step with a few thousand bodies;
- many small matrix products in a loop (e.g. 64×64);
- an iterative solver (CG) on a sparse or matrix-free operator;
- Mandelbrot, the convex hull, and nanoLM's small configuration as the ML regression check.

Each step above states its expected effect on these counters before it starts and is judged on them
(plus wall time, measured by the user on the real machine).

## Phases

1. **The suite and its baseline**, with today's runtime: the numbers every later step is judged on.
2. **Transfer audit**: account for the 3.2 GB up / 1.9 GB down on nanoLM and the suite; fix what's
   avoidable within the current design.
3. **Timeline semaphores** (section 3): small, self-contained.
4. **Device addresses** (section 2), kernels and runtime together, mirrors kept.
5. **Device-resident memory** (section 1), mirrors only as a fallback, then removed where placement
   decides.
6. **Exact barriers** (section 4), runtime first, then compile time.
7. **Loop fusion** (section 6), simple cases, and **`spawn`s folded into grids** (section 7).
8. **Recorded sequences** (section 5), **device-generated work** where the stream depends on data
   (section 8), and **small work on the CPU** (section 9), as the suite shows host time, data-dependent
   launches or tiny kernels dominating.

Each phase keeps `cleave/tests/vulkan.rs` green and nanoLM's losses equal to the CPU's to the printed
digit, and lands as its own change.
