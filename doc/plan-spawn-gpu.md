# Plan: what `spawn` means on the GPU

Status: proposed, 2026-10-15. Nothing implemented yet. Extends `doc/plan-spawn.md` ("GPU (not
implemented; the semantics must already allow it)") and `doc/plan-vk-runtime.md` (section 7); the
backlog entry "`spawn` on the GPU: the semantics aren't defined, and today's behavior is an accident"
is the problem statement.

## Where it stands

- **A `spawn` is a plain call on the Vulkan backend.** `CodegenOptions::host_tasks` is false there
  (`cleave/src/pipeline.rs`): host tasks only contended to record into one queue (measured on MNIST,
  libomp's waiting took most of the host's time), so `lower_spawn` (`cleave/src/mlir_lower.rs`) emits an
  ordinary, inlinable `func.call`, and the matching waits disappear. A deliberate choice, but reported
  nowhere, and the only mapping there is.
- **A `spawn` comprehension yields a tuple**: `[for j in 0..SPLIT: spawn f(...)]` is unrolled into
  `SPLIT` separate calls whose results form a `__TupleN` (arrays of structs aren't a comprehension
  default). There is no loop left for `cleave-fold-loops-into-grids` to fold. A `for` over the tuple
  by its variable unrolls (`doc/plan-compile-time-sequences.md`); that failed for a tuple a
  comprehension produced ("no `impl Index<__Tuple8<...>>`": the index was seen before the
  comprehension's type defaulted), fixed on 2026-10-15 (`check_pending_indices` asks for the
  unrolling; `mlir_lower.rs::a_loop_over_a_comprehension_of_structs_unrolls`), so a sum over the
  parts can now be written once for any `SPLIT`.
- **The existing fold's cap answers another question.** `foldedBytes` (64 MiB, `CLEAVE_GPU_FOLD_BYTES`)
  groups iterations so their peak-live allocations stay in the L2 cache: a locality objective. A
  micro-batch of a training step holds gigabytes, so under that cap it would never group.
- **The cost, measured** on nanoLM (2026-10-15, same model and gradient): `SPLIT` 8 → 2 is +2.4%
  tokens/s on the GPU (fewer, larger dispatches), but costs a CPU build its parallelism (one part per
  core). The kernel now says `SPLIT = 2`: a GPU tuning written into the program.

## The semantics

`spawn f(args)` means: *this call may run concurrently with the code around it; its result is a
future.* The language already guarantees what makes every mapping legal: no shared mutable state, so
running in place (serial elision), on another thread, on another queue or folded with its siblings
computes the same values. What a backend must preserve, whatever it chooses:
- **The values**, including where results are combined: a reduction over the parts has a fixed order
  (a fixed tree), never one that depends on the mapping or on timing.
- **The memory bound**: a mapping must fit the target's memory; folding more parts than fit is not a
  legal choice, whatever it would save.
- **Reporting**: each `spawn` site gets a diagnostic saying how it was mapped and why (a placement
  report, `doc/plan-vulkan.md`: "placement is inferred, with the program able to see and constrain
  it"). No silent default.

The mappings, in the order a GPU backend prefers them:
1. **Folded into a grid**: siblings that call the same function on independent inputs of the same
   shapes become one more grid dimension of that function's kernels. One stream of dispatches instead
   of N, each kernel N times larger.
2. **Async compute queues**: siblings that differ (a compute-bound product beside a bandwidth-bound
   norm) go to separate queues, event dependencies between them, stream-ordered allocation. Needs the
   timeline semaphores of `doc/plan-vk-runtime.md` (phase 3).
3. **Sequential**: the fallback, today's only behavior, now reported.
4. **Across processors**: a task on the CPU while another runs on the GPU, from the scheduler of
   `doc/plan-vulkan.md` ("HEFT with transfer costs"). Later.

## Steps

1. **Report what happens today.** A diagnostic per `spawn` site on the Vulkan backend ("run in place:
   no host tasks on this backend"), under the existing debug switch, then in the placement report. A
   test that checks it. Nothing else changes.
2. **A homogeneous `spawn` comprehension yields an array.** When every element has the same type (the
   same function), `[for j in 0..N: spawn f(...)]` is an `[T; N]` (arrays of structs exist already:
   `[for i in 0..LAYERS: new_block()]`, by context), and lowers as a loop over `j`, not N unrolled
   calls: the loop step 5 folds. With it, a stdlib deterministic tree reduction over an array (a fixed
   pairwise tree, so the order doesn't depend on the mapping). Settles `doc/plan-spawn.md`'s open
   question "arrays of futures in the first version or not". Tests: the CPU task path unchanged (same
   values, same thread use), the reduction's order fixed. (Indexing the parts by a loop variable
   already works on the tuple, see above: nanoLM's `parallel_grad` can be written once for any
   `SPLIT` before this step.)
3. **Per-backend defines from the build** (interim): `cleave-build` sets a `define` per backend
   (`Build::define` exists), so nanoLM says `SPLIT` 8 for the CPU and 2 for the GPU without the source
   choosing. Becomes unnecessary for nanoLM once step 5 lands, stays useful in general.
4. **Probe the fold on a spawn loop.** On a small transformer (nanoLM's `small` sizes), the loop of
   step 2 inlined and handed to `foldLoopIntoGrids`: what it accepts, what blocks it (the callee's
   nested host loops over layers and heads, its allocations and frees, arena descriptors, residue
   operations). Findings written here before building anything.
5. **Fold spawn loops under a device-memory objective.** A second objective beside the cache one:
   for a loop that comes from a `spawn` comprehension, the largest group whose peak-live allocations
   fit a fraction of the device memory budget (the runtime already tracks the budget, `make_room`),
   chosen to minimize dispatches; the diagnostic says which objective applied and the group chosen.
   Done when nanoLM at `SPLIT = 8` runs on the GPU at least as fast as at `SPLIT = 2` (11.48 vs 11.21
   k tok/s measured on 2026-10-15, same losses to four decimals) and step 3's interim define is gone.
6. **Async compute queues** for heterogeneous siblings (mapping 2), after `doc/plan-vk-runtime.md`'s
   timeline semaphores.

Each step keeps `cleave/tests/vulkan.rs` and the CPU `spawn` tests green, and lands as its own change.

## Open questions

- Whether folding should also apply to plain (non-`spawn`) loops of calls under the memory objective,
  or only where the program said `spawn` (an explicit "these are independent" from the programmer,
  where for plain loops the compiler must prove it).
- How the device-memory fraction is chosen (fixed, or from the measured peak of one iteration) and
  reported.
- Heterogeneous comprehensions (different functions) stay tuples; whether they deserve a mapping
  beyond async queues.
