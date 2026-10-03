# Plan: `spawn` — structured task parallelism, CPU tasks and GPU async compute

Status (2026-10-03): steps 1-2 done — the libomp probe and runtime audit (§3-§4), then the runtime itself: atomic refcounts, per-thread arenas, a per-thread pool, no measurable single-threaded cost; OpenBLAS rebuilt with `USE_LOCKING` (release `cleave-openblas-0.3.33-r3`, to adopt in `ci/openblas-version.txt`). Step 3 begun: `spawn`/`sync` parse, type and run end to end under serial elision (a spawned call runs in place, `sync` is a no-op; `cleave/tests/spawn.rs`). Next: the parallel lowering (§ "Step 3, the first vertical slice", points 2-5). Agreed with the user on 2026-10-03; supersedes the open
questions of the `doc/backlog.md` entry "An explicit, Cilk/Go-style `spawn`/`sync` concurrency
primitive", which it answers. Decided: libomp's tasks as the scheduler (§4), atomic refcounts and a
concurrent pool (§3).

Where this comes from: the runtime was first meant to stay single-threaded, with fine-grained
automatic parallelism inside operations (the matmul level) — not generic enough. Then a graph
scheduler, close to cleave's sibling project — but it gives up much of what the compiler can do.
Cilk's model, spawn/sync over work stealing, is the third idea: it copes with imbalance and nesting,
and `parallel for` and the like are built on top of it.

## 0. Why

Parallelism discovered by the compiler, one operation at a time, didn't pay on the CPU: a fork-join
per op costs ~12% on MNIST, and the automatic fusion of parallel regions was built, measured and
reverted (`doc/backlog.md`, the fork/join fusion entries). Single-thread, nanoLM's step is now 76%
`sgemm` at OpenBLAS's ceiling (`doc/backlog.md`, the nanoLM single-thread entries): what is left is
spreading independent work over cores, at a grain coarse enough that scheduling costs nothing.

The programmer knows that grain. nanoLM's step, data-parallel:

```
let g1 = spawn gpt_grad(x1, y1, p, m);
...
let g8 = spawn gpt_grad(x8, y8, p, m);
let g = g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8;    // each read waits for its task
```

Eight micro-batches of 4 sequences instead of one of 32: one fork-join per step, GEMMs of 512 rows
still efficient single-threaded.

## 1. The model

- **`spawn f(args)`** starts a call that *may* run in parallel with the rest of the function. Only a
  call: the task captures nothing but its arguments, already evaluated (Cilk's `cilk_spawn`), which
  sidesteps cleave's closure-capture gap entirely.
- **Its result is a future**, invisible in the types: `g1` has `gpt_grad`'s result type. **Reading it
  waits for that task, and only that one.** Passing it unread to another `spawn` makes a dependency,
  not a wait.
- **`sync`** waits for every task the current function started. **Every function ends with an
  implicit `sync`**: no task outlives the function that started it (structured concurrency, Swift's
  `async let` is the closest precedent). A future can't escape: storing one in a struct, a tuple or
  an array, or returning it, reads it.
- **No function colouring.** Any function can be spawned; the decision is made at the call site, not
  in the callee's signature (unlike `async`/`await` in JS, C#, Rust).
- **Serial elision is guaranteed.** Removing every `spawn` and `sync` gives a valid sequential program
  with the same result, bit for bit. Parallelism is permission, never obligation: a backend may run
  any spawned call inline. This is what makes one program correct on every target (§4).
- **Determinism.** Tasks share no mutable state (§2), and a reduction's order is the one written
  (`g1 + g2 + ...`), not the scheduler's: results are identical whatever the thread count — unlike a
  framework's parallel reductions.

Later, if needed, sugar for loops: `[for k in 0..8: spawn f(k)]` is an array of futures, each element
read waiting for its own task. Not in the first version unless nanoLM needs a variable task count.

## 2. Safety: no data race by construction

Tensors and structs are values; most functions (`gpt_grad`) are pure. A race needs shared mutable
state, of which cleave has three kinds, each ruled out statically:

1. **Mutable arrays (`mut` arrays, `DynArray`) passed to a task are frozen until it completes**: the
   parent, or another task, writing one in between is a compile error. Checked with the existing
   alias analysis (`alias_analysis.rs`).
2. **Global runtime state.** A spawned function must not reach, transitively, an `extern` that isn't
   `#[pure]`: the random generator (`rand_*`, one global PCG state), printing, checkpoint I/O. A
   compile error naming the path. (Per-task random streams can come later if a use appears.)
3. **The runtime's own bookkeeping**, §3 — the one real cost of this design.

## 3. The runtime: what shared-memory tasks require

`doc/hld.md` ("Threading", Memory management) fixed the runtime as single-threaded by design:
non-atomic refcounts, one global region arena (`ARENA_*`, `REGION_DEPTH`), a pool with global free
lists behind one spinlock (`POOL_LOCK`), and parallelism across CPS computations planned as
*share-nothing* contexts exchanging messages. It names shared-memory threads "the harder path
(real atomic refcounts, thread-confinement proofs to elide them)". `spawn` takes that path, deliberately:
eight tasks read the same model `m`, so they touch the same refcounted objects. `hld.md`'s threading
section gets rewritten to say so when the runtime changes (step 2).

What has to change (decided):

- **Atomic refcounts.** A task borrows its arguments: the parent holds a reference until the task
  completes, so a task never needs to keep an argument alive; but a task's code may still
  retain/release an argument's parts (a field stored into a new struct it returns). Retain/release
  become atomic. Their single-threaded cost is measured on MNIST and nanoLM in step 2; if it shows,
  the known refinements come after, not before: a "shared" bit in the header set at `spawn` on what a
  task can reach, atomics only when set (Swift's approach), and the affine/pool path, which already
  has no refcount, staying as it is.
- **A concurrent pool.** Today's free lists are global behind one spinlock (`POOL_LOCK`), which eight
  threads allocating per step would contend on. Per-thread free lists, with blocks freed by another
  thread than their allocator's going back through a remote-free list (mimalloc's scheme).
- **Region arenas per thread**: the arena (`ARENA_*`, `REGION_DEPTH`) is one global bump region today;
  a task opening regions needs its own.
- **Region analysis** (`region_analysis.rs`) stays per function: a task's regions are its own.

### Scaling: designed for many cores, measured on eight

The first target is an 8-core desktop, but a HPC node has up to ~192 cores: nothing below may be a
design that only works at 8. Contention points, and what keeps each one off the hot path:

- **Refcount cache-line bouncing.** The real scaling killer of atomic refcounting (Swift's ARC on
  many cores): N tasks retaining and releasing the same object (the model's weights) fight over one
  cache line. Atomicity is correct, not sufficient. A task must not touch the refcount of what it
  borrowed: arguments are borrowed for the task's whole life (the parent keeps them alive), so the
  compiler elides the retain/release pairs on them and on their parts inside spawned code — the
  borrow rule `refcount.rs` already applies to parameters, extended through field reads. A retain is
  left only where a task makes a borrowed value outlive the task (stores it in its result), once per
  object, not per use.
- **The allocator.** No global lock and no global list on the hot path: per-thread free lists, frees
  from another thread through a per-owner remote list (batched, not one atomic per block).
- **NUMA.** On a multi-socket node, memory is first-touch: a task's buffers allocated on the socket
  that runs it. Per-thread pools give this for free; nothing may pre-allocate everything from one
  thread.
- **False sharing.** Headers, task result slots and per-thread runtime state cache-line aligned (64
  bytes), never packed next to another thread's.
- **Reductions.** Summing N task results in program order is a chain of N adds, a sequential tail at
  192 tasks. Reductions are written as balanced trees (pairwise, each level spawned), still in a
  fixed, program-defined order, so still deterministic (§1); the stdlib gets a `reduce` built that way.
- **The scheduler.** libomp's per-thread task deques and stealing scale to many cores; the cost to
  watch is task grain: too many tiny tasks and the stealing itself dominates. Spawned work stays
  coarse (a whole micro-batch, a whole head), and a `parallel for` (later) cuts its range by a grain
  size, not one task per element.

Each point is checked at 8 threads with a contention-specific measurement (uProf: cache-line
contention, `LsBadStatus`/HITM-style events; libomp's own task statistics) — at 8 cores contention is
invisible in wall time long before it hurts at 192, so it has to be looked for, not waited for.

## 4. Lowering

### CPU

- **Scheduler: libomp's tasks** (decided). A good work-stealing runtime is months of work — the
  user has built several; libomp's is mature, already linked, and handles nesting and imbalance.
  Tasks are `omp.task` (with `depend` clauses for a future passed unread to another task) and
  `omp.taskwait`, through MLIR's OpenMP dialect, or straight calls to libomp's task entry points
  (`__kmpc_omp_task_alloc`/`__kmpc_omp_task`/`__kmpc_omp_taskwait`) from the generated code if the
  dialect's outlining fights the trampolines below — the probe of step 1 says which.
  Tasks run inside a parallel region: the outermost function that spawns opens one (`omp.parallel`
  around an `omp.single` running its body) when it isn't already inside one (`omp_in_parallel()`);
  a spawning function called from a task is already inside the team, so its tasks are ordinary
  deferred tasks, stealable by any thread. Outside any region, libomp runs a task inline — which
  is serial elision. A `parallel for` (later) is built on tasks (divide and conquer, `taskloop`),
  not on `omp.wsloop`, so it nests the same way.
- **Calls.** A spawned call is a `func.call` inside an `omp.task` region whose result is stored to a
  slot the parent owns; the region captures the parent's SSA values (the arguments) and MLIR outlines
  it itself (`--convert-openmp-to-llvm`, then the translation to `__kmpc_omp_task_alloc`/
  `__kmpc_omp_task`). No hand-written trampoline. With the out-params pass (`pipeline.rs`,
  `buffer-results-to-out-params`), the slot is the destination the caller already wants, no copy.
- **CPS.** `spawn f(args)` becomes a `LetSpawn { var, callee, args, cont }`; an await is inserted
  before the first read of `var` on every path (data-flow); `sync` and each function's exits get a
  `Sync`. Refcounting (`refcount.rs`) treats a spawned call's arguments as borrowed until the await.
- **BLAS** stays pinned to one thread (`cleave-rt`, `ensure_thread_count_pinned`): parallelism comes
  from the tasks, one pool for the whole process.

### GPU (not implemented; the semantics must already allow it)

A kernel launch is already asynchronous: it returns at once, completion signals an event. So
`spawn` = **async compute**: enqueue the call's kernels on a stream/queue of their own. Reading the
result maps two ways: **from another kernel**, an event dependency between streams, the host never
blocks; **from the host** (a loss printed, a token sampled), the host waits on the event. Different
kernels then overlap when the hardware has room (a compute-bound GEMM next to a bandwidth-bound
normalization), or run one after the other when it doesn't — serial elision makes both legal.
Allocation inside a task must be stream-ordered (`cudaMallocAsync` or the equivalent). Within one
operation, parallelism stays the compiler's job (`linalg` → `gpu` dialect): on a GPU, per-op
parallelism is the native model, not the overhead it was on the CPU.

Where to split is a per-target tuning, not a property of the program: eight micro-batches help a CPU
(one per core) and hurt a GPU (one large batch fills it better). The split factor is a parameter —
a `define` set per target, as nanoLM's sizes are today.

### Probe results (step 1, 2026-10-03)

A hand-written MLIR program (`omp.parallel` + `omp.single` + `omp.task`, MLIR 22's own
`--convert-openmp-to-llvm`, `mlir-translate`, `clang`, `libomp` from the LLVM prefix), a task being a
call to a function burning ~0.5 s:

| case | 1 thread | 8 threads | speedup | result |
|---|---|---|---|---|
| 8 independent tasks | 4.00 s | 0.51 s | 7.8x | identical |
| 2 tasks chained by `depend(out)`/`depend(in)` | 0.99 s | 1.02 s | 1x (expected) | identical |
| nested: 4 tasks, each calling a function that spawns 2 | 3.94 s | 0.51 s | 7.7x | identical |

The dialect route works as is: tasks deferred and stolen, nesting through a plain function call into a
spawning function costs nothing extra, dependencies via `__kmpc_omp_task_with_deps`. `--emit-object`'s
stub list (`pipeline.rs::register_openmp_stub_symbols`) needs the task entry points added
(`__kmpc_omp_task_alloc`, `__kmpc_omp_task`, `__kmpc_omp_task_with_deps`, `__kmpc_omp_taskwait`,
`__kmpc_single`, `__kmpc_end_single`); `--run` already loads the real `libomp.dll`, and `--emit-exe`
and `cleave-build` already link it.

### Runtime audit (step 1, 2026-10-03)

Every piece of `cleave-rt` state a task's code can reach, and what it needs. Generated code never
touches a refcount itself: retain/release are always calls (`mlir_lower.rs::lower_refcount_call`),
so making them atomic is a change to `cleave-rt` alone.

| state | today | under tasks | needs |
|---|---|---|---|
| refcount (`cleave_retain`/`cleave_release`, `_void`, `_tagged`) | plain `+= 1`/`-= 1` | races on shared objects | atomic `fetch_add`/`fetch_sub`, acquire fence on reaching zero; elided on borrowed arguments (§3, scaling) |
| pool and `cleave_alloc_rc` free lists (`FREE_LISTS`, `POOL_LOCK`) | global, one spinlock | correct, but one lock for every thread | per-thread lists, remote frees (§3) |
| region arena (`ARENA_CURSOR`, `REGION_DEPTH`) | global bump pointer, non-atomic load/store; `region_exit` resets the cursor (global LIFO) | **broken**: two threads bump the same cursor, one thread's exit frees another's allocations | per-thread arenas (`thread_local`); 22 enter/exit sites in nanoLM's kernel |
| random generator (`PCG_STATE`) | one global stream (atomic) | nondeterministic sequence | forbidden in spawned code (§2) |
| printing, checkpoints (`FILES`, a `Mutex`) | safe | ordering effects | forbidden in spawned code (§2) |
| OpenBLAS (`SINGLE_THREADED` build) | pinned to one thread (`Once`) | stays single-threaded (the parallelism is above the BLAS grain), but is now called from several threads at once: OpenBLAS documents concurrent calls into a single-threaded build as safe only with `USE_LOCKING=1` (its internal buffer allocator). 2400 concurrent calls (8 threads, distinct matrices) matched the serial results exactly and didn't crash — evidence, not proof | check `cleave-openblas-redist`'s build flags; add `USE_LOCKING=1` to the single-threaded build if absent (no OpenMP build: the existing one links the MSVC runtime) |
| `memrefCopy`, MLIR's `malloc`/`free`, debug instrumentation (`Mutex`es) | stateless or locked | fine | — |

## 5. First target and measurement

nanoLM, data-parallel over micro-batches (§0), gradients summed in program order. Baselines: cleave
single-thread (808 ms/step at 2026-10-03), PyTorch with `OMP_NUM_THREADS=8` on the same twin
(`bench/nanolm-pytorch/gpt.py`, to measure). The losses must stay identical to the single-threaded
run's: same micro-batch split, same summation order (§1, determinism).

## 6. Steps

1. **Runtime probe and audit** (no language change): a hand-written probe of libomp tasks from
   generated code (dialect `omp.task` versus direct `__kmpc_*` calls, nesting, `depend`); the list of
   every retain/release, allocation and region site a task's code can reach, and what each needs (§3).
2. **Runtime**: atomic retain/release, the concurrent pool, per-thread arenas, a single-threaded
   OpenBLAS built with `USE_LOCKING=1` (the audit above); `hld.md`'s threading section rewritten. Single-threaded speed
   measured before and after (MNIST, nanoLM).
3. **Language**: grammar (`spawn` before a call, `sync` statement), CPS `LetSpawn`/await/`Sync`,
   futures invisible in typing, the escape rule (a future stored or returned is read).
4. **Checks** (§2): frozen arguments, purity of spawned calls; located errors, tested.
5. **Lowering**: trampolines, argument blocks, result slots through the out-params destinations.
6. **nanoLM data-parallel**, measured against both baselines (§5); then `doc/user_guide.md`.
7. **Parallelism a scientist gets without asking, from the stdlib.** A scientist should never have to
   write `spawn` to use the cores: the stdlib does, the way it routes large matmuls to BLAS. Three
   layers: (1) user code is ordinary code; (2) the stdlib is written with `spawn` and a `parallel for`
   built on it — large elementwise and reduction operations split above a size threshold, `map`/`reduce`
   over collections, a data-parallel training step in `nn` — all through the algebra + `mlir::`
   mechanism, nothing hardwired; (3) the compiler parallelizes nothing on its own, it only lowers
   `spawn`. `spawn` stays available to whoever needs it (irregular tasks, tree searches), never
   required (Julia's model: few users write `@spawn`, everyone uses threaded libraries).
8. **Retire the compiler's automatic parallelization** once step 7 matches it: `--openmp`'s
   `affine-parallelize`, the matmul schedule's `scf.forall` → `omp.wsloop`, the region fusion stages
   hardwired to MNIST's network shape, the parallel-loop stub symbols. Not before: today it is
   MNIST's only multi-threaded path (6.4 s at 4 threads). The new path must at least match it on
   MNIST and nanoLM, measured, before the old one goes.

### Step 3, the first vertical slice

What makes it non-trivial: `mlir_lower.rs` works on tensors, before bufferization, and One-Shot
Bufferize can't see through an `omp.task` region. So the task is created *after* bufferization:

1. **Syntax**: `let x = spawn f(args);` (`ExprKind::Spawn`, the call's type) and a `sync;` statement.
2. **CPS**: three `PrimOp`s in ordinary `LetPrim`s — `Spawn { unit }` (a handle with the call's
   result type), `Await` (before the first statement reading `x`), `Sync` (the statement, and before
   a function's return when it spawned). Passes that don't know them stop on them, as on any
   unknown `PrimOp` (the e-graph's walk).
3. **`mlir_lower`**: a spawn is an ordinary `func.call` tagged `cleave.spawn`, `Await`/`Sync` are
   markers: bufferization and the out-params pass treat the call like any other (its result written
   straight into the caller's buffer).
4. **A post-bufferization pass** does the parallel part: each tagged call wrapped in an `omp.task`
   (a slot for a scalar result, loaded at its await); markers become `omp.taskwait`; each function
   that spawns gets a wrapper opening `omp.parallel` + `omp.single` when `omp_in_parallel()` is false.
5. **Refcounting**: a spawned call's arguments must stay alive until its await or the `sync`, not
   just past the call — the task may not have run yet. The delicate part.

First version: an await is a `taskwait` (waits for all children) — conservative, correct by serial
elision; per-future waits (`depend`) come after.

## 7. Open questions

- `omp.task` from MLIR's dialect or direct `__kmpc_*` calls (§4), settled by the probe of step 1.
- Whether atomic refcounts cost anything measurable single-threaded (§3, step 2), which decides if the
  shared-bit refinement is needed at all.
- Arrays of futures (`[for ..: spawn ..]`) in the first version or not.
- An explicit `await` marker, optional, for a reader who wants blocking points visible — not in the
  first version (minimal annotations).
