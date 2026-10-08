# Backlog

*Ordered — top to bottom is the order we work through it. Not a wishlist: each entry is a real, confirmed gap (found by testing or by direct inspection), not a guess about what might be missing.*

Completed items live in [backlog-done.md](backlog-done.md).

---

## The pool keeps every freed block, whatever its size class: freed memory is never given back

`cleave_release` parks every freed block in its size class's free list (`pool_push`), for the next
allocation of that class. A program whose phases use different sizes (a training step: activations in
the forward and backward passes, weight-sized buffers in the optimizer) holds the peak of each class
at once. Measured on nanoLM at 116M parameters (d768, 16 layers, context 1024, `CLEAVE_ALLOC_STATS`):
live 14.9 GiB at most in size classes, 22.1 GiB held (live plus parked), up to 14.5 GiB parked.
A cap on the parked bytes (a fixed budget, or a fraction of the live bytes), beyond which a large
block goes back to the system, would keep the pool's speed on the sizes every step reuses; to measure
in memory and in time (a block taken back from the system costs page faults).

## A `DynArray` of structs doesn't release its elements when it dies

Its slots hold references (`dynarray_set_ptr` retains, `dynarray_get_ptr` hands out a retained one,
an overwritten element is released), but the `DynArray` envelope has no release cascade into its
buffer: the elements it still holds when it dies leak. And an element released through
`dynarray_set_ptr` (overwritten) is released flat (`cleave_release`), without the cascade into its
own refcounted fields that the compiler generates per type.

## Debt: the test harnesses each carry their own copy of the pipeline

`cleave/tests/` has at least ten private copies of the compile sequence (CPS conversion, derivative
synthesis, e-graph, dead code, escape analysis, `insert_refcounting`, `lower_program`,
`lower_to_llvm`, JIT symbol table): `leaks.rs`, `checkpoint.rs`, `language_model_ops.rs`,
`affine_pool_alloc.rs`, `array_release_cascade.rs`, `extern_buffers.rs`, `refcount.rs`,
`spawn_leaks.rs`, ... Each drifts from `pipeline.rs::build_optimized_cps` on its own, and most run with
`tasks: false`: the leak of every array built by a function (2026-10-08, 3.8 GiB a nanoLM step) went
unseen by the leak suite, which never runs nanoLM's shape with tasks. `main.rs` has the same problem
inside the compiler (four copies of the optimize-and-refcount sequence, one per dump flag). One
harness in the crate (`cleave::testing`), built on the real pipeline function and taking
`CodegenOptions` (tasks on by default), with the allocation counter and the JIT symbol table; the
dump flags as taps on the one pipeline.

## Debt: a constraint on a never-generalized abstract variable is checked nowhere

`infer.rs`'s module comment, in its own words: a constraint on a variable still abstract and never
generalized (a `let mut`'s, say) "has nowhere further to travel once its enclosing scope finishes. It's
silently unchecked". Not in the backlog until the 2026-10-08 audit. The failure, if it comes, comes
later (an impl not found at monomorphization) without the `let`'s location. The same comment still
lists mutability checking as not done, which `check_mutability` has done for a while.

## Debt: compiler panics reachable from user programs

`mlir_lower.rs` has 191 `panic!`s and `cps.rs` 27; most are internal invariants, some are unsupported
programs (index-assignment into a `#[mlir_type]` value, a function-typed struct field, constructing
`[Tensor<...>; N]`, a struct array nested in an array, a multi-def loop condition, "CPS doesn't
support ... yet"). A user gets a compiler crash, not a located error. A corpus of small programs, one
per unsupported construct, run by a test that expects a diagnostic with a span; each panic it reaches
becomes a check before lowering (inference or a pre-lowering validation pass). Found and fixed
by the 2026-10-08 audit's probes: an integer literal that doesn't fit its type (past `i64` it panicked,
`let x: i8 = 300;` wrapped silently to `44`) is now a located error (`infer.rs::check_literal_ranges`);
a lambda literal passed straight to a call now works (`lower.rs::hoist_lambda_args`), one used as any
other value is a located error (`pipeline.rs::check_lambda_positions`). Left: the five unsupported
constructs above.

## Debt: `cleaveElideBlockCopies` relies on an assumption it can't check

`cleave-mlir-shim/cpp/shim.cpp`: eliding a block copy is sound only if no layout constant (a stride,
an offset) of the temporary was folded before bufferization, and the pass says so ("Assumed, not
checkable here"). True of its one user today (`Sgemm::sgemm`'s `to_buffer` with a dynamic layout); a
future extern taking a block through a plain-layout `to_buffer` and a cast would break it silently, a
wrong stride in a BLAS call. Either the producer marks the buffers it guarantees (an attribute the pass
requires), or the pass proves it from the IR it sees (no constant-folded `extract_strided_metadata` of
the temporary anywhere).

## Debt: stdlib names the compiler knows

Contrary to `feedback_extensibility_discipline`: `Complex` (`infer.rs`, `cps.rs`, `mlir_lower.rs`:
imaginary literals and their default type), `Tensor` (`egraph.rs`: the AD rebuilds `Tensor(data:
...)` by name), `Additive`/`add`/`zero` (`egraph.rs`, `monomorphize.rs`: the AD's accumulation and
zeros), `DynArray` (`infer.rs`, `mlir_lower.rs`). Each wants either an algebra the stdlib impls carry
(a literal algebra for imaginary literals; the AD's accumulation through whatever algebra declares
it) or, where the dependency is the language's own, a documented one in one place.

## Debt: declared axioms that can't be built are dropped without a word

`egraph.rs::axiom_to_rewrite` (and `derivative_rule_to_rewrite`) return `None` when a rule's body isn't
representable in the e-graph (`build_pattern`: a field access, a struct literal, ...) or `egg` refuses
the pattern; the caller skips it (`if let Some(...)`). A derivative rule missing that way surfaces later
as a clean "no rule reaches" error; an axiom just never fires, and the stdlib author who declared it
never learns. A warning (or an error) when the stdlib is loaded, naming the axiom and what in its body
isn't representable.

## Debt: code nothing uses: experimental passes, stdlib helpers

`unroll_jam.rs` (629 lines) and `chain_split.rs` (364) run only behind `CodegenOptions::unroll_jam`/
`chain_split`, both `false` by default; no example, build script or test turns them on, except
`cleave/tests/unroll_jam_probe.rs` exercising the transformation alone. Their history
(`backlog-done.md`: the native matmul's IPC investigation, "two dead ends and one real fix") says they
didn't pay, and the products that mattered moved to BLAS since. Removed with their options and probe,
or kept with a sentence saying what would make them worth turning on. `CodegenOptions::tag_releases`
belongs to the runtime-diagnostics entry below. The stdlib has the same kind of dead weight:
`blas_matmul_transpose_a`, `blas_fma`, `blas_fma_transpose_a` and `blas_fma_transpose_b`
(`stdlib/linalg/matrix.cleave`) are called and tested nowhere, left from before the BLAS dispatch moved
to the `blas` attribute; a test each, or gone.

## Debt: comments that tell the code's history instead of the code

The compiler is 32.8k lines of code and 18.5k of comments; `pipeline.rs::lower_to_llvm` has 622
comment lines for 319 of code. Many narrate how a line came to be ("found by direct testing", former
designs, superseded measurements) and some are now wrong: the production matmul schedule is still
labelled "TEMP EXPERIMENT" (`pipeline.rs`), `num.cleave` says the e-graph has no floats, stdlib
comments describe workarounds since removed. A pass per file: what the code does and why, now; the
history to `backlog-done.md` and git.

## Debt: the largest functions

`cps.rs::convert_expr` (726 code lines), `infer.rs::infer_expr_kind` (466), `main.rs::real_main`
(466, see the harness entry above), `pipeline.rs::lower_to_llvm` (319, plus 622 of comments),
`monomorphize.rs::monomorphize` (313) and `collect_instantiations_expr` (305), `refcount.rs::
rewrite_body` (267). Split along their own match arms where each arm is a self-contained rule.

## Debt: documentation hygiene

`backlog.md` was cleared of its finished entries on 2026-10-08 (72 moved to `backlog-done.md`); code
comments still cite some of them as "`doc/backlog.md`'s own ... item". Plan statuses are stale (`plan-nanolm.md`:
2026-10-02). `plan-affine-ownership.md`, `plan-region-arena.md` and `plan-blas-native.md` are partly in
French, where technical docs are English. Seventeen `CLEAVE_*` environment variables remain: the dump
and trace ones are tools, but behaviour switches (`CLEAVE_OPENMP`, `CLEAVE_NO_THREAD_CACHE`) belong in
`CodegenOptions` or the runtime's documented settings, and all of them in one list (`building.md`).

## A matmul whose row count isn't a multiple of the schedule's row tile (8) fails to compile

Found moving `examples/mnist-interop`'s `evaluate` from one image at a time to batches of 100: every
`forward<100>` layer is a `100 x K` matmul, and the transform-dialect schedule tiles rows by 8
(`matmul_vectorize.transform.mlir`, `tile_using_forall tile_sizes [8, 0]`). The remainder tile (4 rows)
has a dynamic size; that linalg op isn't vectorized, falls through to `--convert-linalg-to-affine-loops`,
and the pass fails: `'affine.for' op operand cannot be used as a dimension id` (the bound is an SSA
value defined inside the enclosing `scf` loop, not a valid affine dim/symbol). Batch 1 (the old
`forward<1>`) and multiples of 8/16 (32, 80) compile fine. `evaluate` uses 80 for now. The fix belongs
in the schedule (pad or peel the remainder tile, as the narrow-output `pad` path already does for N)
or in the fallback lowering, not in user code.

---

## A BLAS result passed to a function and read again later is copied at the call

Eight per nanoLM micro-batch (one per layer, 1.5 MB each, ~1% of the step's traffic, 2026-10-08):
the value projection, computed by `sgemm` into a buffer of dynamic layout (`cleaveLowerBlasMatmuls`:
the layout `cleaveElideBlockCopies`' soundness needs), passed to `causal_attention`, whose
parameter has the plain layout (`function-boundary-type-conversion=identity-layout-map`). One-Shot
Bufferize can't cast a dynamic layout to the plain one without a check, so it copies. The reverse
copy forwarding (`cleaveForwardCopiesToDestinations`) removes the copy when the source isn't used
afterwards; here the backward pass reads it again (`causal_attention_backward`). Removing it needs
to know the callee never writes that parameter: a read-only-parameter analysis over the call graph
(an argument only read by `linalg` inputs, loads, transfer reads, or passed to read-only parameters
of other functions), then the destination becomes the source when neither is written while the copy
lives. Or layouts at function boundaries taken from the type ("Views as first-class descriptors").

---

## Study: parallelism on a Rust runtime (Rayon) instead of libomp

Raised by the user (2026-10-08), after the run-to-run variance turned out to be libomp's worker
placement left to the OS (two workers on one core's SMT siblings: ~20% slower for the whole run,
drawn at process start). Today `spawn` lowers to `omp.task` and parallel loops to OpenMP through
MLIR's OpenMP dialect and LLVM's OpenMPIRBuilder (outlining, captures, `taskwait`), the runtime
being libomp, configured by environment variables read once at its initialization.

The idea: lower `spawn`/await and `scf.parallel` through MLIR's `async` dialect (`async.execute`,
`async.await`, `async-parallel-for`), which outlines the bodies and calls a C runtime API
(`mlirAsyncRuntimeExecute`, `...AwaitToken`, ...) that `cleave-rt` would implement on Rayon. Gains:
the pool entirely ours (thread count, placement in `ThreadPoolBuilder::start_handler`), no
libomp, no `__kmpc_*`, nested parallelism composing in one work-stealing pool, portable. Not a
home-made scheduler (`project_spawn_parallelism`'s rule): Rayon is the scheduler.

The risk to settle first: a blocking await inside a worker takes it out of the pool (lost
parallelism, or a deadlock if every task waits). Either `async`'s coroutine lowering (an await
suspends the task) or waits that help (run other tasks meanwhile, as `rayon::join` does).
Prototype on a recursive `spawn` and a parallel loop: per-task overhead, nanoLM's scaling under
the same placement, nested waits. Migrate only if it holds.

---

## A user `fn` named like an algebra method shadows it inside the stdlib

Found 2026-10-07: a program declaring its own `fn step(a, b)` (two parameters) fails with
`stdlib/nn/nn.cleave:1198:29: error: \`step\` expects 2 argument(s), found 4`: the stdlib's own call
to `Optimizer::step` (four arguments) resolved to the user's function. A user's top-level name must
not reach into another module's bodies; the stdlib's calls resolve in the stdlib's scope (and an
unqualified call in user code that matches both should be ambiguous or prefer the local `fn`, by a
stated rule, not by accident). Repro: any program with `use nn;` and a `fn step` of another arity.

---

## Views as first-class descriptors: a strided view that retains the refcounted tensor it looks into

Planned as Part 2 of `doc/plan-struct-arrays.md` (heap references: arrays of structs first, views on
the same base: retained references, copy on write).

`Slice::slice`/`update` (`stdlib/linalg/tensor.cleave`) give *ephemeral* views today: a slice becomes
a `memref.subview` inside one function, `Sgemm::sgemm` reads it with its real strides, and a block
written by an extern and put back where it was read is written in place
(`cleave_mlir_shim::elide_block_copies`). A view that leaves its function is copied: function
boundaries bufferize with `identity-layout-map`, so a view passed to a `#[no_inline]` function (the
`blas_*` helpers), returned, or stored in a struct field becomes a fresh contiguous buffer.

The model to build (raised by the user: a Fortran-style descriptor): a view is a value of its own,
`{storage, offset, sizes, strides}`, that *retains* the storage it looks into. MLIR's memref is that
descriptor already (allocated pointer, aligned pointer, offset, sizes, strides); the allocated pointer
is the base tensor's `cleave_alloc_rc` data, whose header `cleave_retain`/`cleave_release` find, so a
view retains its base when made and releases it when it dies. Value semantics hold through the
refcount: a live view keeps its base above 1, so an `update` of the base copies instead of writing in
place (copy on write, as Swift's arrays).

What makes it a language change, not a patch:
- layout as a type-level property, inferred, not written: a contiguous tensor keeps the identity
  layout (static strides, the fast loops, `--affine-super-vectorize`), a view carries a strided one
  (a phantom layout parameter, or a `View` type of its own). Functions monomorphize on it like on
  const generics: called with a contiguous tensor they compile as today, with a view as strided.
- function boundaries take the layout of the type instead of `identity-layout-map` everywhere.
- rank N: offsets as `[i32; Dims.len()]` (spread in lowering as `Index` does), the block's rank
  checked against the source's at type level (`Part.len() == Dims.len()`), not left to MLIR's
  verifier; `layout: "dynamic"` derived from the rank instead of a rank-2 literal.
- rank-reducing views (a head of `[B, T, H, DH]`, a row): partial indexing `x[i]` as a view, the
  natural spelling.
- interactions to settle: affine (headerless) structs, `spawn` (atomic refcounts), views of views.

The precondition `elide_block_copies` relies on (no layout fact of the copied block folded into a
constant before bufferization, `cpp/shim.cpp`) disappears with descriptors: the layout is a run-time
value everywhere a view goes.

---

## The `nn` library and the MNIST kernel are far harder to read and write than their PyTorch equivalent — a priority for adoption

Raised directly by the user, comparing `examples/mnist-interop/src/kernel.cleave` with
`bench/mnist-pytorch/mnist_bench.py` line for line: the cleave version makes its author fight the type
system (generics, const generics, explicit turbofish) for things that are not their job. Concrete
friction, all visible in today's kernel:
- a user-defined `Network` needs a hand-written `impl Optimizer<Opt, Network, NetworkState<StateL1,
  StateL2, StateL3, StateL4>>` plus a parallel `NetworkState<...>` struct, forwarding `init_state`/`step`
  field by field — pure structural boilerplate the compiler could derive (a "parameter tree", the way
  JAX pytrees or `nn.Module.parameters()` work);
- `Init::he()` / `Init::xavier()` need the target type spelled out at each use;
- the loss must be a concrete, non-generic `fn` (batch size fixed in its signature) for `grad()` to accept it;
- user code reaches for `mlir::memref::alloc()` and `Tensor::<f32, B, N>(data: ...)` to build a batch.

Success measure: the cleave kernel reads about as plainly as `mnist_bench.py`.

**Inventory, 2026-10-01** — each simplification tried alone on a copy of the kernel, compiled with the
real CLI. What already works today was applied: the kernel went from 249 to 132 lines (mostly stale
comments, plus annotations that were workarounds for since-fixed bugs), with identical generated code
and accuracy (0.9341). No longer needed: scalar type annotations (`let n: i32 = ...`), the const
generic and return type on `forward` (now plain `fn forward(x, net)`: field access on an unannotated
parameter is a deferred `FieldConstraint`, `doc/hld.md`),
intermediate annotated variables for `.to()` conversions, method-call syntax for plain fns
(`net.l1.dense_forward(x)`), expressions as loop bounds.

Fixed since: **two resolution paths for one algebra call** (bare `init_state(opt, net)` panicked in MLIR
lowering while `Optimizer::init_state(opt, net)` worked, and a top-level `fn relu(x: i32)` was
type-checked against `Activation::relu`). One lexical name-resolution pass now decides every call's
target before inference (`cleave/src/resolve.rs`, `doc/hld.md` "Name resolution"); the kernel calls
`init_state`/`step` unqualified.

What still fails, by nature:
- **Bug — invalid MLIR instead of a type error.** Passing a tensor's `.data` to an `extern` that writes
  into it (`train_batch_pixels(start, x.data)`) generates a module that fails verification ("operand type
  mismatch: expected `!llvm.ptr`, provided `tensor<...>`"). It should be rejected cleanly: a tensor is a
  value, and an extern writing into one breaks that.
- ~~**Language gap — tensor construction needs a turbofish.**~~ Fixed: `Tensor(data: pixels)` infers
  `Dims` from the value (`infer.rs::infer_struct_lit_pack_arity`: every arity tried, exactly one must fit
  the constraints, else an error). Values with no shape information (`Ring::zero()`) still need it.
- **Stdlib/design gap — no sanctioned way to build a tensor from host data.** User code has to write
  `mlir::memref::alloc()` and `Tensor::<f32, 32, 784>(data: ...)`. Needs a decision on how an extern hands
  data to a tensor (ties to the previous two points).
- ~~**Language gap — no tuple destructuring in `let`.**~~ Fixed: `let (a, mut b) = ...;` and `(net, state)
  = step(...);`, desugared by `lower.rs` (`doc/user_guide.md`, "Tuples").
- **Autodiff limitation — `grad()` needs a non-generic function with a declared return type.** The loss
  is pinned to batch size 32 in its signature, and an unannotated loss is refused. A derivative that is
  instantiated per call site, like any generic fn, would lift both.
- **Language gap, the biggest one — no structural derivation.** Training a user-defined `Network` needs a
  hand-written `impl Optimizer<...>` plus a parallel `NetworkState<...>` struct, forwarding every method
  field by field (35 lines here). What's needed is a way for an algebra to apply to any struct whose
  fields implement it ("parameter trees"), derived by the compiler — a real design decision (an explicit
  `derive`-like mechanism, or algebras declared as structurally recursive).
- **Stdlib gaps** — no `argmax`, `softmax`, cross-entropy or accuracy helpers (the evaluation loop is
  written by hand). Decided for `argmax`: first `argmax_rows(t: Tensor<T, R, C>) -> [i32; R]` in
  `linalg` (a plain array, since `Tensor<T: Float, ...>` can't hold `i32`), later `Tensor`s of
  non-float elements, which quantization will need anyway; `println(("Epoch=", epoch))` needs a tuple to print several values.
- Already tracked separately: a matmul whose row count isn't a multiple of 8 doesn't compile (evaluation
  uses batches of 80 because of it).

Suggested order: the two bugs first (clear fixes, and the first one is exactly the kind of structural
inconsistency that gets harder to fix later), then the small language gaps (pack-arity inference, tuple
destructuring), then the design questions with the user: structural derivation, generic `grad()`, and
how host data becomes a tensor.

---

## A call taking other calls' results as a tuple stops the e-graph pass from inlining those calls

Found on MNIST after `Optimizer` moved to `optim`'s generic impl for `Trainable` models (training about
1 s slower): `Collect::collect((step(...), step(...), ...))`, the identity a comprehension is wrapped in,
left all four `Optimizer::step<Sgd, Dense<...>>` calls un-inlined, while the same tuple consumed by
field projections inlines them (`egraph.rs`, `Forward::walk`/`is_transparent_chain`). Worked around
where it arose: a comprehension collected into its own tuple emits no call (`cps.rs`). The underlying
limitation is in the walk, and any other opaque call consuming call results will hit it.

## Pool allocator: heavy structs inside light containers

Adam on a two-layer `Trainable` model corrupted memory (2026-10-02): its state is tuples of tuples of
`AdamState` (tensors and scalars), pool-allocated (`alias_analysis.rs`, affine) inside light tuples.
Two causes found: `field_affine_positions` keyed fields by container *name*, so every `__Tuple2<..>`
instantiation shared one answer (fixed: keyed by full type); and something still unidentified in
training after restoring such a state (`cleave/tests/checkpoint.rs`'s resume test crashes without the
workaround). Workaround in place: a struct that can sit inside a light container is never pool-
allocated (`structs_inside_light_containers`). It is conservative (a plain light wrapper around a pool
struct was fine before) and costs the pool to such structs; find the remaining cause, then narrow it.
MNIST has no heavy struct and is unaffected.

## A nested comprehension allocates and copies each row

`[for b in 0..B: [for c in 0..C: e]]` into a matrix runs the inner comprehension once per row: each
row is its own freshly allocated `[T; C]` (`memref.alloc`, a real `malloc`/`free`), then copied into
the outer buffer through a dynamic-offset subview, which lowers to MLIR's generic runtime
`memrefCopy` (`mlir_lower.rs::copy_array_row`). Found on MNIST's cross-entropy gradient (32 rows of
10 per step, a measurable share of a ~1 s regression); `nn`'s `CrossEntropy` now fills its result in
place instead. The real fix is destination passing: the inner comprehension writes straight into the
outer row's slice, with no row array at all; at least, a row copy lowered as a plain loop or `memcpy`
rather than `memrefCopy`. Matters for nanoLM, where the loss covers every position of the batch over
the whole vocabulary.

## Comprehension expressiveness: iterating a collection, filters, several generators; `%`

Comprehensions take one range (`[for i in a..b: e]`); `e` is any expression, an `if`/`else` or a block
included, and nesting builds nested collections. Missing, in the order discussed (2026-10-02):
- `[for x in v: f(x)]`, iterating a collection by value: sugar for `[for i in 0..v.len(): f(v[i])]`,
  unrolled or a loop as usual. Also gives `for v in t` over a tuple.
- Filters, `[for i in 0..n if p(i): e]`: the length is no longer known up front, which `Generate`
  (indexed fill) can't serve. A second protocol, appending (an empty collection, one `push` per kept
  element), for `DynArray` and any growable collection; the compiler picks indexed when there is no
  filter and the bounds are rectangular, appending otherwise. A target that only fills by index
  (`[T; N]`, `Tensor`) with a filter is an error ("length unknown at compile time"), never a silent
  truncation. A filter whose condition folds on an unrolled comprehension already works through the
  pruning of copies, giving a shorter tuple (a form of slicing).
- Several generators flattened, `[for i in 0..3, for j in 0..4: e]`: rectangular bounds keep a known
  length (indexed, `k` split into `i = k / 4`, `j = k - i * 4`); bounds depending on an outer variable
  need appending.
- `%` as an operator: `Rem::rem`/`Rem::mod` exist (`stdlib/num`), the grammar has no `%` for them
  (`mul_op` is `*` and `/` only; a test wrote `i - (i / 7) * 7`).

## Arrays of tensors (and of structs, nested) as a comprehension's default target; `Generate` beyond two dimensions

A comprehension over same-typed elements fills an array only when the elements are scalars (or arrays
of them); tensors or structs with bounds known at compile time are unrolled into a tuple instead
(`infer.rs::decide_comprehensions`, `is_array_element`). `Generate` for `Tensor` covers one and two
dimensions (`stdlib/linalg/tensor.cleave`); more needs one impl per rank, or an impl over the
`Dims...` pack once packs can be taken apart (`doc/plan-compile-time-sequences.md`, step 6). The
fuller rule from
`doc/plan-compile-time-sequences.md` ("an array when they unify") needs the backend to represent
`[Tensor<f32, 1, 2>; 2]`: a memref can't hold tensors, and a struct-leaf array is single-dimension
only (`mlir_lower.rs::lower_array_construct`). Found with `Optimizer` on a `Dense<f32, 1, 2>`, whose
two fields share a type: the state came out as an array of tensors and failed MLIR verification.

Found again on 2026-10-04 (nanoLM v2), two ways. `is_array_element` took any `Ty::Con` for a scalar,
so a struct named without generics (`Net`, `Gpt`) made the comprehension an array of structs: nanoLM's
`[for j in 0..SPLIT: spawn gpt_grad_micro(..)]` collected eight micro-batch gradients into `[Gpt; 8]`,
which nothing releases (512 MB per training step). Fixed: numbers and booleans only, a struct of any
kind gives a tuple (`leaks.rs::a_comprehension_of_spawned_structs_leaves_nothing_behind`). Still open:
a comprehension of structs *without* `spawn` and with a runtime-shaped body goes through `Generate`
into `[Net; 4]`, whose `memref.alloc` fails verification ("result must be memref, got `!llvm.ptr`").

## Indexable collections as algebras: `x[i]` on structs and tuples, unrolled `for`, comprehensions, slices

Steps 1-3 done (`doc/plan-compile-time-sequences.md`, "Suggested order"); slices, packs in the same
style and structural autodiff remain. Originally: arrays, tensors, tuples and structs all
indexed through stdlib algebras (`Index` for a runtime index, `Field<S, I, F>` for a constant one,
synthesized per struct), under the same rule as const generics — legal wherever it folds at compile
time, no hint. What makes "apply to every layer" (training a user `Network` without a hand-written
`impl Optimizer`/`NetworkState`), the fifteen per-arity `Print` tuple impls in `stdlib/io/io.cleave`,
and slices (`t[1:]`, `t[:25]`) expressible.

## Literal suffixes, units of measure, and folding driven by the algebras (idea, 2026-10-01)

Three ideas that hold together, none started:

- **User-defined literal suffixes.** `2v`, `1.5ma`, `3db` desugar to an algebra call on the literal, the
  way `+` desugars to `add`. First client: the imaginary suffix `4i`, hard-coded in the compiler today
  (`ExprKind::ImaginaryLit`), which would move to `stdlib/complex` — one primitive fewer.
- **Units of measure as types** (F#'s units of measure): `Quantity<T, const M, const L, const S, const A>`,
  `mul`/`div` adding/subtracting exponents through const-generic folding, `add` requiring equal
  exponents, zero run-time cost. A strong argument for cleave-cast (a PINN whose loss is physical
  equations, checked dimensionally by the compiler).
- **Folding by evaluating the algebra's own definition.** `const_eval.rs` knows integer `add`/`sub`/
  `mul`/`div` in Rust, whatever the algebra: the one place the compiler "knows how to count", and wrong
  for any algebra redefining `+` (decibels add logarithmically: `3dB + 3dB` ≈ 6.02 dB). Folding should
  run the impl (`Ring<i32>::add`, `dB::add`, `Quantity::mul`) on constants instead — compile-time
  evaluation of pure cleave functions, possibly through the existing JIT. Closes the leak and leaves the
  compiler's core purely structural.
- **The same compile-time evaluator would open user-defined transformations** beyond `derivative`
  (PDF → CDF, interval arithmetic, uncertainty propagation): compositional ones are expressible today as
  a type plus its algebras (`Dual<T>`, `Interval<T>`), or as declared per-operation rules generalizing
  `derivative`; non-compositional ones (integration, inversion) only through rule tables and numerics.
  Fun, not planned.

## Generic bodies are checked per instantiation, not once with rigid type variables

A generic impl's body (`impl<T: Float> ...`) is inferred with `T` an ordinary, permissive type variable:
constraints on it are deferred and only really checked at each instantiation, and a generic impl whose
body fails generic inference is silently skipped (`monomorphize.rs::build_impl_templates`) until some
instantiation reports the error. So a body using an operation its bounds don't grant (`x + 1` under
`T: Show`) is only caught when instantiated at a type lacking it. Checking once, generically — `T`
rigid, satisfying exactly its declared bounds (and what they imply) — would report it at the definition
for every user. Applies to all generic code (impls over packs of types included), so it is its own
project, not part of the pack work.

## Retain/release are opaque calls, so LLVM can never fold them — emit the refcount fast path as ordinary IR

`cleave_retain`/`cleave_release` are external calls into `cleave-rt`: LLVM can't see that a retain
followed by a release of the same pointer is a no-op, can't drop the count updates of an object that
never escapes, and can't scalarize anything around them. Since 2026-09-30, `cleave-mlir-shim` annotates
the *allocator* entry points (`allockind`/`allocsize`/`noalias`, the subset that is sound — see
`annotateAllocators`), which already removed a few allocations and copies, but deliberately not the
refcounted release: it frees only at zero, so declaring it a `free` would be a miscompile.

The direction, raised in conversation: lower retain to an inline `load`/`add`/`store` of the header's
count (the header sits at `ptr - 16`, `cleave-rt`'s `rc_header`), and release to an inline decrement and
compare, calling into the runtime only on the slow path (count reaching zero: free + cascade). LLVM then
sees ordinary memory operations and can pair and remove them on its own. Needs care on two points before
building it: (1) atomicity — today's counts are only ever touched outside OpenMP regions (`cleave-rt`'s
own allocator-free-regions argument), which must stay true or the inline ops must become atomic; (2) the
GPU direction (handle separate from data, timeline-based availability) discussed the same day, which
argues for keeping the header's layout private to one place. Measure first: the dynamic count of
retain/release calls per training step bounds the possible gain.

---

## Ownership is classified by *type* (`is_rc(ty: &Ty)`) where it should be classified by *role* — the model that would be correct, why the code answers a coarser question, and what that costs

Worked out in discussion after the `8a748f8` reset, against the code rather than from first principles. The model below is the user's; the gaps between it and the implementation were each checked in the source.

**The model — two storage classes, decided by one question: does this value cross the jump?**

- **Slot** — parameters and return values, anything carried into the next continuation. Ownership transfers here. **Release rule: a value is released at the jump, if and only if it is not used in the continuation being jumped to** (not passed as an argument, and not free in its body). Local and syntactic — no fixpoint.
- **Temp** — everything that dies before the jump. Three sub-classes: the stack (tiny only), the arena (CPS-managed, for temps that provably don't escape the region), and buffers MLIR's own lowering introduces (a reduction accumulator being the hard case).

Aliasing is handled without changing the release rule: when a call hands back its own argument, a `retain` at the transfer means the count goes 2 → 1 at the jump and the object survives. That is not a special case bolted on — the existing test `a_struct_passed_to_a_genuinely_identity_shaped_function_is_released_only_once` already passes on this base, so the rule must preserve it and does.

Note that "slot" and "temp" are the same predicate read in two directions — escapes the continuation, or doesn't. There is one analysis to write, not two.

**Why the code has bugs even though the model holds — three measured divergences:**

1. **Classification is by type, not by role.** `RefcountCtx::is_rc` takes a `&Ty`. Two values of the same type, one escaping and one not, are indistinguishable. This is the deep one, and it explains the `8a748f8` disaster exactly: that commit flipped **one boolean** in `is_rc` (adding bare tensors) and thereby moved an entire population from "MLIR owns it" to "cleave owns it" in a single edit. Under role-based classification that edit could not be expressed — each value is classified by what it does, so there is no switch that reclassifies a population wholesale.
2. **Release fires at the function's `return`, not at the jump.** `refcount.rs`'s own module doc states this as a deliberate choice ("deliberately **not** a last-use/liveness analysis — it releases at the *latest* possible point"). The machinery to do better already exists: the pass already computes what is live at each jump (arguments passed plus free variables); it uses that to decide what to *keep*, never to decide what to release.
3. **Ownership does not cross a call.** `walk_var_info` populates `owned_origin` only from `LetPrim` — `PrimOp::Struct` is owned, `PrimOp::Field` inherits from its base, everything else is not. The `App` arm is literally empty (`CExpr::App { .. } => {}`). So a value returned from a call is never marked owned, and rule 2 then sweeps it up at the `return`. **This is the direct cause of the unbounded loop leak in the entry above** — and it is why `b = Boxed(...)` constructed inline does *not* leak (it goes through `PrimOp::Struct`) while `b = bump(b)` does.

**Explicitly ruled out — do not restart this.** A separate ANF/normalisation pass to name intermediates was considered and is **not needed**: cleave's CPS already names everything. `LetPrim { var, .. }` binds every primitive result, and a call's result arrives as the continuation's own parameter. The problem was never that values lack names; it is that ownership is not computed for them across a call. Checked in `cps.rs`/`refcount.rs` directly, not assumed.

**What the model buys downstream.** Every hack in the current pipeline exists to repair a decision taken at the wrong granularity: `dps_rewrite` decides buffer sharing at MLIR level after CPS has already committed its retain/release placement (and `Strategy::Passthrough` then has to emit a compensating `cleave_retain`); `compensate_refcounts.rs` existed to patch that in turn; `unify_alloc.rs` renames allocators wholesale at the end because nothing earlier can say who owns what. Deciding ownership per value, at the jump, removes the reason each of those exists — see the destination-passing entry below for the measured half of that argument.

**Order of work.** The missing `App` rule first: it is one rule in one file, it closes a measured leak, and it is the honest prerequisite for any last-use analysis. The role-based classification and the jump-time release rule after that. Destination-passing emission is a separate, larger piece.

---

## `--promote-buffers-to-stack` is not in the pipeline at all — the obvious lever for the MLIR-owned temporaries cleave can never own, never measured

Checked directly: `pipeline.rs` runs `one-shot-bufferize`, `ownership-based-buffer-deallocation`, `buffer-deallocation-simplification` and `lower-deallocations`, and **no** buffer-hoisting or stack-promotion pass of any kind.

That leaves the one storage class cleave has no lever on entirely at the mercy of the heap. Some of MLIR's own lowering-introduced buffers are short-lived and statically bounded — a reduction accumulator is the canonical example, and the one case destination-passing can never fix, since there is no pre-existing buffer of the right shape to hand it. `--promote-buffers-to-stack` would move exactly that population off the heap without cleave having to own it, which is the property that matters: the rule for this class has to stay "MLIR owns it, cleave never touches it" (violating that rule is precisely what `8a748f8` did).

Caveat, and the reason this needs measuring rather than just enabling: the project has a standing, hard-won rule that bounded-but-large never reaches the real call stack (see the struct-allocation-strategy entry below, and the AOT `--no-openmp` stack-overflow bug in [backlog-done.md](backlog-done.md)). MLIR's pass takes a size threshold; it must be set conservatively, and the AOT `--no-openmp` configuration — the one that already overflowed once — is the acceptance case, not the default one.

Cheap to try: add the pass, set a small threshold, measure `mnist-interop` under the usual protocol plus an explicit `--no-openmp` run.

---

## `mut` carries no real semantic weight for *most* scalar/struct/tuple bindings — likely removable for those, but `DynArray`'s own mutate-in-place `push` turns out to be a genuine, permanent exception, not a stopgap — raised and then corrected while designing the struct-allocation-strategy entry below

Surfaced directly while working out why `DynArray<T>`'s own envelope (`{buf: RawBuf, len: i32, cap: i32}`) seemed to need heap identity despite being exactly as small/pointer-shaped as `Dense` (see the entry below): the real reason wasn't the envelope's own size, it was `push`'s own API convention — mutates its `mut v: DynArray<T>` parameter in place, returns nothing, relies on the caller's binding already being a heap pointer under today's implementation for the mutation to be observed at all.

**A two-category model was proposed, tried, and found genuinely unsound — the attempt and why it fails matter as much as the conclusion.** The idea: **storage types** (`Array`, `Tensor`, `RawBuf` — anything reached through *indexed* access) are mutable by nature, no annotation needed; **value types** (scalars, structs, tuples — everything else) are never mutated in place, so a `mut` binding's own "reassignment" is semantically indistinguishable from plain shadowing (already compiles to pure SSA/phi-carried rebinding under the hood, confirmed via this session's own hand-written MLIR probes). Under this model, `DynArray`'s own envelope looked like an ordinary "value type" (small, pointer-shaped) that just needed `push` migrated to value-return style (`v = v.push(x)`, matching `Optimizer::step`'s own convention) to drop its last mutate-in-place case.

**Implemented directly, then caught by the user before landing, via a precise, concrete counter-example — a real aliasing hazard, not a style objection.** `push`'s own body, even rewritten to return a fresh envelope, still writes the new element *through the existing, shared `buf` pointer* in the common (no-resize) case: `raw_set(v.buf, v.len, x); DynArray(buf: v.buf, len: v.len+1, cap: v.cap)`. Given `let v1 = v0.push(x); let v2 = v0.push(y);` (both reading the *same*, unreassigned `v0`) — both writes land in the *identical* slot (`v0.buf[v0.len]`), and `v1`/`v2` end up as two silently-divergent envelopes sharing corrupted storage, each believing it owns the next free slot. Exactly the classic C "fat pointer copied by value" hazard (`struct { char* data; size_t len, cap; }`, trivially copyable, silently aliased). Cleave has no move/ownership/borrow-checking system (deliberately, per this whole project's own direction) to make `v0.push(x)` *consume* `v0` the way Rust's own `Vec::push(&mut self, ...)`/move semantics prevents this exact bug — so nothing catches a stale-binding reuse at all.

**Root cause, precisely: `DynArray` has no safe value-semantics representation at all, not just an inconvenient one.** The only two implementations of "copy the envelope" are: share the `buf` pointer (cheap, but exactly this aliasing hazard) or deep-copy the backing buffer on every envelope copy (safe, but destroys the entire point of an efficiently-growable, in-place-mutated collection). Unlike `Dense`/`Network`/tuples — which have *no* shared backing storage at all to alias, so two independent copies can never diverge dangerously — `DynArray` is inherently a single-owner, shared-mutable resource. Reverted immediately, cleanly (`git checkout --` on all five touched files, confirmed via `git status`/`git diff` — zero trace left): `push` stays exactly as it was, mutating its `mut v: DynArray<T>` parameter in place, relying on `DynArray` staying a stable, heap-boxed, single-identity value.

**The corrected model**: `mut` remains genuinely load-bearing for *two* structurally different reasons, not one — array/`Tensor`/`RawBuf` indexed-element assignment (as originally reasoned), **and** any struct type built around a single-owner, in-place-mutated shared resource (`DynArray`, confirmed the only real instance in the whole codebase, `doc/backlog.md`'s own struct-allocation-strategy entry below has the full survey) — the latter needing `mut`-parameter mutation specifically *because* cleave has no ownership system to make a safe value-return alternative exist at all. `mut` on an *ordinary* scalar/struct/tuple binding (never wrapping a `DynArray`, transitively) still reduces to pure shadowing, that part of the original reasoning holds — but "remove `mut` as a language concept, keep it only for arrays" was wrong as stated; the real boundary is narrower and shaped by this ownership gap, not by storage-vs-value alone. Not attempted further — the removal idea itself is now correctly understood to be more constrained than first thought, not abandoned outright for the cases where it *does* hold.

## A struct's own allocation strategy is hardcoded to "always heap, always refcount" regardless of shape — real hardware-profiled evidence it dominates the remaining single-thread gap, a three-axis design worked out and empirically validated; axis 1 (slot vs. pointer), including field-granularity release tracking, is now implemented, test-verified, and measured on the real kernel; axes 2/3's own embedded-array-threshold optimization is not

Raised chasing the `--no-openmp`-vs-PyTorch single-thread gap (`2.3×`, `doc/backlog.md`'s own thread-count-sweep entry) once the known matmul-codegen levers (FMA fusion, register-resident accumulation, the `128->10` padding fix, `linalg.matmul_transpose_a`/`_b`) were already exhausted — the question became "where do the remaining cycles actually go," answered with real hardware counters (AMD uProf, `collect --config hotspots -g`), not another guess.

**The profiler-confirmed finding, on the real `mnist-interop` kernel, `--no-openmp`, a 32,000-sample capped run**: `train_and_evaluate` itself (the one giant `--inline`d function, real FMA compute included) is only `50.6%` of process CPU time. The rest — `memrefCopy` (`7.2%`), the CRT/NT heap allocator chain (`vcruntime140.dll`'s own internal allocator wrapper, resolved via the DLL's own export table + image base to sit ~2KB past `__NLG_Return2`, i.e. an unexported helper between `std::alloc::alloc` and the real syscall; `ntdll.dll:RtlAllocateHeap`/`RtlFreeHeap`, resolved the identical way, confirmed *directly inside* those two functions, `9.3%`), and the arena's own `cleave_region_enter`/`cleave_region_exit` bookkeeping (`10.7%` combined) — is pure memory/allocation churn, zero floating-point work.

**Confirmed causally, not just correlated, by the user's own suggested experiment**: rebuilt the identical training loop with the network reduced to a single minimal `Dense<784,10>` layer (same FFI data loading, same `grad()`/`Optimizer::step` structure otherwise) — total CPU time barely moved (`11.85s` -> `12.83s`) despite the real compute (`train_and_evaluate`) collapsing from `50.6%` to `6.3%`. `ntdll.dll` (the heap allocator) went from `1.70s` to `6.13s` **in absolute terms**, becoming the single hottest module, ahead of the executable itself. The fixed, per-batch allocation/copy/bookkeeping cost is essentially independent of network size — on the real 4-layer network it was already comparable to the entire matmul workload; shrink the compute and it dominates completely.

**Root-caused, via static disassembly of the real object's own relocations (`llvm-objdump -r`, matching each `cleave_alloc_rc` call site to its own size argument)**: of 70 static `cleave_alloc_rc` (heap) call sites in the compiled kernel, the majority (11×112 bytes, 10×16 bytes, 4×104 bytes, ...) are far too small to be tensor payloads — they're the wrapper structs `Dense`/`Network`/`NetworkState`/`DenseState` and the tuples `Optimizer::step` returns, each just a handful of pointer fields, individually heap-allocated (`malloc` + refcount header) purely because `unify_alloc.rs`'s own policy heap-boxes *every* struct uniformly, regardless of shape.

**Two full days earlier, established while root-causing the AOT `--no-openmp` stack-overflow bug (`doc/backlog-done.md`, same session): `region_analysis.rs`'s own arena promotion (axis B below) structurally can never help this specific allocation class.** The wrapper structs are built inside `Optimizer::step`, which is *by definition* not region-local (its own output becomes `net`/`state` for the *next* iteration — exactly the condition the arena's own "can't free one object out of order" limit (`doc/hld.md`) correctly refuses to promote). Extending `region_analysis.rs`'s own reach further would never reclassify this — the wrapper struct's escaping nature is precisely what that analysis is designed to say no to. A genuinely different mechanism is needed, not a deeper version of the existing one.

**The design, worked out directly with the user and empirically validated at every step, not just reasoned about — three real, orthogonal axes, not one:**

1. **Slot vs. pointer (new axis)** — a struct is "light" (no heap identity at all — a genuine LLVM aggregate SSA value, threaded through a loop's own carried state exactly like a scalar accumulator already is via `mlir_lower.rs::lower_loop`'s `scf.while`) *iff* every field, after the rule in (2) below, is a scalar or a pointer. Cleave has no aliasing/reference model at all (already established, `stdlib/nn/nn.cleave`'s own `Init<T>` doc comment) — nothing ever observes two bindings of "the same" struct diverging, so there's no correctness reason every struct must have a stable heap address. `Dense`/`Network`/`NetworkState`/the `Optimizer::step` tuple all qualify unconditionally, at any layer count `N` (a struct of pointers is cheap regardless of how many, `N` is a compile-time constant either way) — validated directly, not assumed: a hand-written MLIR probe threading a 2-pointer struct through a 1,000-iteration `scf.while`, lowered all the way to real x86-64 (`mlir-opt` -> `mlir-translate` -> `llc -O2`), compiles to a *fixed*, one-time `push`/`sub rsp, 0x20` prologue and a pure register swap (`rdi`/`rsi`) inside the loop — zero calls to any allocator. Stress-tested at `N=64` fields (transformer-scale, several attention/FFN weight tensors per layer): still zero heap, a *single* one-time `sub rsp, 0x2c8` (712 bytes) for the fields that spill past available GPRs, a confirmed genuine loop (backward `jle`, not unrolled).

2. **Inline vs. indirected, for arrays specifically (the C-style pitfall, caught directly by the user)** — "bounded" (known size at compile time) is *not* sufficient to justify real stack space, any more than `int buf[1_000_000_000];` is safe in C just because its size is a compile-time constant; a struct field's own *direct* footprint matters, not just whether it's bounded. A pointer field is always cheap regardless of what it references (the pointee's own allocation is a completely separate question — axis 3, below); an *embedded* array field carries its own real `N × sizeof(T)` weight and needs a real, conservative, register/tiny-stack-spill-scale cutoff — past it, the array is converted to a pointer to a separately-allocated buffer (exactly how `Dense`'s own `w`/`b` `Tensor` fields already work today, confirmed directly: `w1`'s own `cleave_alloc_rc` call carries its *own*, separate, full tensor byte size — `0x188000` = `784×512×4` — never folded into `Dense`'s own tiny 16-byte allocation). This is what keeps axis 1 from ever reintroducing the C pitfall through the back door.

   Also validated directly, both directions of the realistic access pattern, not just the trivial swap case: a 64-`f32`-element array embedded alongside 2 pointers, genuinely initialized (not `undef` — ruled out a first, misleading probe where LLVM was silently inventing unconstrained values), updated at a **fixed** index every iteration — LLVM's own standard optimizer recognizes the repeated write as dead until the loop's last iteration and defers materialization entirely to the loop exit, `O(1)` per iteration, no cleave-side mechanism needed. Updated at a **dynamic** index (the realistic case, `arr[i % 64] = ...`) — `llvm.insertvalue` structurally cannot take a non-constant index at all, so LLVM correctly falls back to a real `alloca` + `getelementptr` + indexed `store` — one genuine store per iteration (necessary, not wasted), still against a *single*, one-time, bounded stack allocation (`sub rsp, 0x138`, never growing with iteration count), with the final struct-to-return-value copy itself auto-vectorized (`zmm`/`ymm` bulk moves). Both patterns: zero heap, one fixed allocation regardless of loop trip count.

3. **Bounded-but-large still never reaches the real call stack — arena or heap, never raw `llvm.alloca`, no matter how "bounded" it technically is.** The same C-style pitfall axis 2 closes for *embedded* arrays reappears one level up if a *large-but-bounded* buffer were ever allowed to claim "it's bounded, so `alloca` is fine" directly (exactly the class of bug the `--no-openmp` native-stack-leak fix, `doc/backlog-done.md`, already required real work to close, for per-loop-iteration allocation specifically). The resolution is structural, not a size heuristic to tune: once axis 2 has forced anything past the tiny inline threshold into a pointer, that pointer's own *target* allocation is *always* routed to the pre-existing axis B (`region_analysis.rs`'s arena when the construction site is provably region-local, real heap+refcount otherwise) — it never competes for axis-1 stack space at all. `Init<Tensor<T,In,Out>>::xavier`/`he`'s own `[T;In,Out]` scratch buffer (`stdlib/nn/nn.cleave`) already does exactly this today, via `mlir::memref::alloc()`, not `alloca` — consistent with this model already, not a new case to handle.

**Net result originally claimed for the three axes together — corrected below, this turned out wrong for the actual motivating structs**: `Dense`/`Network`/`NetworkState`/tuples move from "always heap+refcount" to "always a value, zero allocation of any kind" — eliminating the majority of the 70 static `cleave_alloc_rc` sites found on the real kernel. The genuinely large, genuinely escaping tensors (`w1..w4`, `net`/`state`'s own real weights) are completely unaffected, correctly staying on the existing refcount path (axis B already says "not region-local" for them, correctly, since they must survive past every iteration).

**Axis 1 is now implemented, in `mlir_lower.rs`, not `unify_alloc.rs`** — `is_light_struct`/`is_light_struct_rec`/`is_light_field_ty` (recursive, whole-program-informed, cycle-guarded via a `visiting` set) classify every struct name at every field-type site; `ty_to_mlir` picks a bare `!llvm.struct<(...)>` value instead of the usual opaque `!llvm.ptr` for a "light" one, and `lower_struct_construct`/`lower_field_access` gained parallel `llvm.mlir.undef`+`insertvalue`/`extractvalue`-based paths that never touch `alloc_llvm_value`/`cleave_alloc_rc` at all. **Five structural disqualifiers found necessary, four of them only by direct testing (real `STATUS_ACCESS_VIOLATION` crashes, MLIR-verifier failures, or an outright memory leak on a real training run — not one of them reasoned out in advance)**, each its own whole-program scan mirroring `refcount.rs`'s established `collect_constructed_struct_names` shape, except the last (a local, non-recursive re-check, deliberately *not* a call to `refcount::is_refcounted` itself — see below):
- A field that's array/tensor/vector-shaped (axis 2 isn't implemented yet — see below) — checked structurally, no scan needed.
- `DynArray` itself, by name — the mutate-in-place aliasing hazard the `mut`-removal entry above already covers in full. (No longer needs its own separate transitive check — see the fifth disqualifier below, which subsumes it: `DynArray` is always genuinely refcounted the moment it's constructed anywhere.)
- Any struct ever the target of a real `s.field = v;` (`refcount::collect_field_mutated_struct_names`) — `lower_field_store`'s own GEP-based mutation needs a stable address a light aggregate value structurally doesn't have. Caught by `tests/user_guide.rs::struct_field_mutation`'s own MLIR-verifier failure the moment the classifier stopped being a single-hardcoded-name proof-of-concept.
- Any struct ever named as an `extern fn`'s own parameter or return type (`refcount::collect_extern_boundary_struct_names`) — a real C-ABI symbol (`cleave-rt`) is written assuming cleave's fixed, uniform pointer-shaped struct representation regardless of field shape; `dynarray.cleave`'s own `RawBuf {}` (a zero-field struct, only ever produced by an `extern fn`, hence vacuously "light" under a naive empty-`all()` check) and the generic `RawBuffer<S: HeapStruct>`'s `_ptr`-suffixed impls (any concrete struct `S` used as a `DynArray<S>` element) both hit this directly — caught via a genuine `STATUS_ACCESS_VIOLATION` crash (`RawBuf`) and a `'func.call' op operand type mismatch` MLIR-verifier error (`DynArray<Point>` vs. `DynArray<Pair>` in the same program) the first time real stdlib code, not just a hand-written proof-of-concept struct, went through the classifier.
- **A fifth disqualifier — any struct with a genuinely-refcounted nested field — existed temporarily, then was removed and replaced with real field-granularity release tracking.** Found the hard way, against the real `mnist-interop` training loop, not a test or a probe: a severe, unbounded memory leak, every training iteration — `retain_targets` already retains every refcounted field argument of a `PrimOp::Struct` construction unconditionally, keyed on the field's own type, completely independent of whether the *containing* struct is itself refcounted; once the container is "light," nothing ever schedules a matching release for that extra retain, since the CPS-level refcounting pass had no notion of "end of life" for a bare aggregate value, only for a heap pointer with a real, trackable owning scope. First fixed by blanket-excluding any such struct from "light" at all (the fifth disqualifier) — closing the leak, but at the cost of `Network`/`NetworkState`/`Dense`/the `Optimizer::step` tuple, exactly the structs this whole mechanism was built for, never actually becoming light.

**Replaced with the real fix, directly in `refcount.rs`'s own core insertion algorithm (`insert_refcounting_fn`/`rewrite_body`), not a separate optimization layered on top** — following directly from the same "CPS already reifies every future use as an explicit argument, so ownership is a local question" discussion that produced the Tier 1 pair-cancellation pass above. `mlir_lower.rs` gained `LightLeafPath`/`light_struct_release_leaves` — a recursive walk (mirroring `is_light_struct_rec`'s own shape) finding every genuinely-refcounted field reachable from a light struct's own top level, transitively through further light fields, each as a `[(struct type, field name), ...]` hop sequence ending in the leaf's own type. `refcount.rs` threads this through three places `ctx.is_rc(ty)` used to gate alone, now `ctx.is_rc(ty) || !ctx.light_release_leaves(ty).is_empty()`: the ordinary owned-seeding condition, and — found only by hitting real, further bugs, not anticipated up front — the `Fix`-local continuation's own seeding (a loop's own carried `net`/`state` is exactly this shape) and `retain_targets` itself (embedding an *existing* light-with-leaves value into a fresh container, e.g. `Optimizer::step`'s own real `(Network, NetworkState)` return tuple, needs each of its own leaves retained, not the whole value). Release/retain emission for a light-with-leaves value (`wrap_releases`, `field_read_retain`, `retain_targets`'s own wrapping) now builds a real `PrimOp::Field` chain down to each leaf, terminating in an ordinary `Retain`/`Release` on it, via a small shared `build_leaf_chain` helper — reusing the exact same `PrimOp::Field`/`PrimOp::Retain`/`PrimOp::Release` primitives and their existing, already-correct MLIR lowering, no new primitive or lowering code needed at all.

**Two further real bugs found only by running the actual `digits-interop`/`mnist-interop` binaries, not caught by the full test suite** (a `STATUS_HEAP_CORRUPTION` on the very first run after removing the fifth disqualifier): (1) the `Fix`-local seeding gate above, missed on the first pass — a loop's own carried light-with-leaves state was silently never tracked, leaking its leaves every iteration; (2) `retain_targets`'s own gate, also missed initially — embedding an existing light-with-leaves value (not a fresh field-read) into a new container retained nothing, so the tuple case above corrupted memory the moment the caller's own `net`/`state` bindings were later (correctly) released, since nothing had protected the tuple's own copy of their leaves. Both are the exact same class of oversight as the original fifth-disqualifier leak — every `ctx.is_rc(ty)` gate in this file needed its own light-with-leaves counterpart, found one at a time by testing against the real, full-scale programs, not derivable by inspection alone.

**Consequence, corrected and now precisely measured, not estimated**: `Network`/`NetworkState`/the `Optimizer::step` tuple **do become light now** (each is built entirely from light/heavy-pointer fields once `Dense`/`DenseState` are excluded only for their own direct `Tensor` fields — axis 2, still not implemented). `Dense`/`DenseState` themselves still don't (their own fields are directly `Tensor`-tagged, the separate, still-real axis-2 gap documented below) — matching the honest scope this design always had once axis 2 was factored out.

**Measured directly, `--dump-mlir-lowered` on the real `mnist-interop` kernel, textual `call @cleave_alloc_rc` occurrences before any inlining collapses duplicate call sites (the same controlled comparison established earlier in this entry) — three points on the same axis, not just two**: `208` (pre-axis-1 baseline, commit `8924672`) `-> 207` (axis-1 with the fifth disqualifier — only `Sgd`) `-> 202` (axis-1 with field-granularity release tracking replacing it). Five more construction sites disappear once `Network`/`NetworkState`/the `Optimizer::step` tuple actually qualify as light — a real, if still modest relative to the original "majority of 70 sites" ambition, measured win, not the zero this entry's own fifth disqualifier had left it at. `call @cleave_retain`/`call @cleave_release` counts stay at `8`/`16` either way — the same logical retains/releases still happen, now distributed across individual leaves via `PrimOp::Field` chains instead of whole-struct pointers for the newly-light containers, not eliminated (that's Tier 1's own job, above, and it already found nothing further to cancel here).

Verified: full `cargo test -p cleave --release` suite green throughout, including two dedicated Tier 1 tests (above) and the pre-existing struct-allocation tests updated for the light-struct behavior change (not weakened — each got a genuinely heavy variant added alongside a new light-specific test). `digits-interop` re-run end to end after every step in this whole entry, accuracy unchanged at `0.94713414` — including the run that first caught the two field-granularity bugs above via a real `STATUS_HEAP_CORRUPTION` crash, and the clean run after fixing them. `mnist-interop`'s own real training loop re-run in full (10 epochs) after the field-granularity fix specifically: `test accuracy: 0.9342`, bit-identical to the established baseline, no crash — the real, full-scale confirmation this whole mechanism needed, not just the small-scale `digits-interop` one.

**A tensor field has the *exact same* unsolved release-tracking gap as a refcounted struct field — currently avoided only by accident, not by design, and axis 2 (below) must not be attempted without closing it first.** Raised directly by the user, verified precisely, not assumed: a `#[mlir_type(tensor)]` field's own payload *is* allocated via `cleave_alloc_rc` (`store_native_shape_field`), a real `RcHeader` exactly like any struct — `is_refcounted(Tensor)` only says `false` because a tensor's own aliasing story is simpler (copied fresh on every store, never independently shared), so the general CPS-level Retain/Release pass doesn't need to track a *bare* tensor variable. But its actual liberation has exactly one path in the entire codebase: `lower_release_cascade`'s own tensor-field branch, reached *only* when the tensor's containing struct's own `cleave_release` call reports `freed` (`mlir_lower.rs:3400-3414`) — piggybacked entirely on the container being a real, heap-allocated, `cleave_release`-calling struct. A "light" struct never makes that call (no pointer of its own to release) — so a tensor field embedded directly into one, the way axis 2 below casually proposed, would leak its payload unconditionally, on every single construction, with *nothing* in the current classifier (`field_struct_is_ever_refcounted` deliberately excludes tensor/vector-tagged names, matching `is_refcounted` itself) positioned to catch it. Today's separate "tensor/vector-tagged field — not yet implemented" exclusion in `is_light_field_ty` accidentally prevents this by blocking construction support entirely, not because anyone designed it as the safety mechanism — any future work on axis 2 must extend the *release*-tracking exclusion (the fifth disqualifier above), not just build the `insertvalue` machinery, or it reopens the identical `mnist-interop` leak on the single most expensive class of payload in the whole system.

**The tensor-leaf case (`Dense`/`DenseState` themselves) is now also implemented — see the dedicated entry below for the full design, a real bug found and fixed, and an empirical measurement.** Only the *embedded untagged array field* half of axis 2 (`[T;N]` declared straight on a struct, not a `Tensor`) remains genuinely not implemented — still correctly a hard disqualifier in `is_light_field_ty`, for the original, narrower reason (no `insertvalue`-based inline-array flattening built).

## Per-shape, compile-time OpenMP thread-count selection — a real, cleave-specific lever MKL/PyTorch can only approximate at runtime, not yet attempted

Raised directly by the user, following up on the accumulator-latency-chain fix (`doc/backlog-done.md`'s own "the `36.7%` accumulator-latency chain... real fix found and landed" entry): PyTorch/MKL choose their own thread count *dynamically*, per call, based on problem size — a small matmul often runs with fewer threads than a big one, since fork-join overhead can exceed the parallelism benefit for small work. Measured directly on the real kernel, by the user, across five thread counts (`1`/`2`/`4`/`8`/`16`): **`4` threads is the real optimum (`22.80s`), not `8` (physical cores, `24.56s`) or `16` (default, `30.26s`)** — a real, non-obvious, U-shaped curve, not "more threads always better" up to the physical core count.

**`OMP_DYNAMIC=true` (the zero-code, standard-OpenMP way to ask libomp itself to pick fewer threads when it judges the work too small) does not reproduce this** — tested directly: `29.82s`, essentially unchanged from the plain `16`-thread default, alongside a real runtime warning (`OMP: Warning #227: Cannot determine machine load balance - Using KMP_DYNAMIC_MODE=thread limit`) — this build's own libomp can't do the load-balance detection its own dynamic mode needs on this platform. Not a viable path as-is.

**The real, cleave-specific opportunity, not yet built**: every matmul shape in a cleave program is fully static (every dimension known at compile time, unlike MKL, which only learns shapes at runtime) — in principle, `mlir_lower.rs`/the transform-dialect schedule could compute a real, shape-appropriate `num_threads` per matmul (its own FLOP count, `M*N*K`, all compile-time constants) and bake it directly into that *specific* region's own `omp.parallel`, doing *better* than MKL's own runtime heuristic since the decision is made with perfect, static information, not an educated guess. **Two real, unresolved technical questions, found by checking before attempting, not assumed away**:
1. `--convert-scf-to-openmp`'s own `--num-threads=<uint>` option (confirmed real, `mlir-opt --help-hidden`) is **pass-wide**, applying uniformly to every `scf.parallel` region a single invocation converts — not a per-region override. Getting a genuinely different thread count per matmul shape would need either running the conversion multiple times, each scoped to a different subset of regions (partitioned by shape — real, fiddly schedule engineering, not attempted), or a transform-dialect mechanism that computes a thread count from a matched op's own static shape and attaches it as a real, per-op attribute `--convert-scf-to-openmp`'s own conversion pattern would need to consult instead of (or ahead of) the pass-wide option — not confirmed whether that pattern reads any such override at all.
2. Even if a per-region thread count can be set, the *aggregate* effect isn't obviously a pure win: `net_grad`/`Optimizer::step` run `18,750` times over a full training run, each doing 8 real matmuls — if several of those get their *own* independent thread-team fork/join per call (rather than sharing one already-open team), the *number* of fork/join events could go up even as *some* individual regions get cheaper, a real tradeoff not yet measured either way.

**Scoped as its own, separate chantier — not blocking the rest of the parallelization work**: a real, bounded next step (not yet started) would be confirming point 1 directly (read `SCFToOpenMP.cpp`'s own conversion pattern for whether it already supports a per-op override, before designing schedule-side arithmetic to produce one), then, if real, prototyping a two-tier split (e.g., "big" shapes at one thread count, "small" ones at another, chosen via a compile-time FLOP-count threshold) rather than a fully general per-shape scheme up front. **User's own explicit call**: find further *single-thread* wins first (a fixed thread count is trivial to change later and doesn't gate that work at all), then come back to find the real optimal thread count — dynamic or fixed — once the single-thread baseline stops moving.

**Re-measured after the native-stack-leak fix (`doc/backlog-done.md`'s own "AOT binary built with `--no-openmp`... genuinely crashed" entry), against a real PyTorch-CPU baseline for the first time on this exact sweep, by the user directly, on the same machine, same session — the U-shape holds, and a sharper, more important finding sits alongside it**:

| config | elapsed | vs PyTorch (1 thread) |
|---|---|---|
| PyTorch (`torch.set_num_threads(1)`, `mnist_bench.py --threads 1`) | **16.46s** (0.9303 acc) | reference |
| cleave, `OMP_NUM_THREADS=4` | **22.49s** | 1.37× slower |
| cleave, `OMP_NUM_THREADS=8` (physical core count) | 25.00s | 1.52× |
| cleave, `OMP_NUM_THREADS=2` | 27.38s | 1.66× |
| cleave, `--no-openmp` (genuinely no OpenMP compiled in) | 37.86s | 2.30× |
| cleave, `OMP_NUM_THREADS=1` (OpenMP build, forced to 1 thread) | 42.95s | 2.61× |
| cleave, `OMP_NUM_THREADS=16` (default today — logical/SMT count) | 45.60s | 2.77× |

(all cleave rows: `0.9342` accuracy, identical across every config, confirming none of this is a correctness difference.)

Two separate findings, not one:
1. **The U-shape is real and reproducible** across two independent measurement sessions (`22.80s`/`24.56s`/`30.26s` then vs `22.49s`/`25.00s`/`45.60s` now, at `4`/`8`/`16` threads respectively) — 4 threads beats both fewer and more, on an 8-physical-core/16-logical-thread machine. The `16`-thread number moved a fair amount between sessions (`30.26s` -> `45.60s`) — not chased further here, plausibly ordinary machine-load variance at this specific oversubscribed/SMT-contended setting rather than a regression from any change in between (every other row's own accuracy and rough magnitude stayed consistent).
2. **New this round, and the more consequential one**: `OMP_NUM_THREADS=1` on the OpenMP-compiled binary (`42.95s`) is measurably *slower* than `--no-openmp` (`37.86s`) — real, non-zero OpenMP fork/join machinery overhead exists even at one worker thread, on top of whatever the actual compute costs. And **even cleave's own best-tuned configuration (`22.49s` at 4 threads) is still `1.37×` slower than a single PyTorch thread (`16.46s`)** — the gap this project's whole parallelization effort is meant to close isn't purely a "not enough threads" question; it's already present at the 1-thread-vs-1-thread level, underneath whatever the thread-count tuning can still buy back. Closing *that* gap (single-thread codegen quality, `doc/backlog-done.md`'s own several already-fixed mechanisms — FMA fusion, vectorization, the accumulator-latency chain, `Broadcast0`/`reduce0` vectorization) remains the more fundamental lever than thread-count tuning alone, exactly the ordering the user's own earlier explicit call (just above) already anticipated.

**Re-measured again, much later, after the region-open-conditional fix and the `--cse` addition (both entries below) — the single-thread-codegen-quality lever named above just paid off directly, not a separate axis**: cleave's own `--no-openmp` serial time dropped to **`29.59s`** (user's own measurement, quiet machine) — against the same `16.46s` PyTorch-1-thread reference, that's **`1.80×`**, down from `2.30×` at the same config just above. Raised directly by the user, cross-checking the table here rather than trusting a vague impression: **"si tu regardes les baselines, je crois qu'en monothread PyTorch est vers 15, ça veut dire qu'on est plus qu'à x2 à peu près"** — confirmed precisely against this table's own stored `16.46s` reference, `1.80×`, close to but a bit better than the user's own quick "~2×" estimate. A real, quantified narrowing of exactly the gap this section's own finding #2 called "the more fundamental lever" — achieved with zero new threads, on the single-thread axis alone, ahead of the user's own planned change to the parallelization strategy itself (which is why single-thread, not the OpenMP rows above, is the reference this project now measures itself against — see the `--cse` entry below, raised directly by the user: *"non ce n'est pas openmp, il est prévu de changer le parallélisme, donc c'est bien monothread qui nous intéresse"*).

**Re-measured a third time, after the `dps_rewrite` `Strategy::Passthrough` fix (own entry below, `~8.4×` fewer bytes copied per sample) — user's own real measurement, four independent serial runs plus one 4-thread run, same quiet machine**: `--no-openmp` — `31.17s`, `26.87s`, `29.68s`, `28.10s` (average `~28.96s`) — against `16.46s`, **`~1.76×`**, a further real narrowing from `1.80×`, on the single-thread axis alone, exactly as this fix's own object-level measurement predicted (a real, if partial, reduction — the copy eliminated was a real cost, but compute already dominates `~69%` of total time per the VTune breakdown above, so a memcpy-only fix was never going to close the whole remaining gap by itself). **The 4-thread number is the real headline**: `OMP_NUM_THREADS=4` — **`16.59s`** — against PyTorch's own `16.46s` (1 thread), that's **`~1.01×`** — cleave's own current best configuration has, for the first time this whole project, essentially **matched** a single PyTorch thread on the one real workload this project measures itself against. Not yet a win, but the gap this project's own "parallelization is the viability test" framing exists to close is, right now, close to zero on this specific measurement.

---

## `mlir-sys`/`melior`'s own ceiling: binary distribution, and whether cleave (and a second, sibling language) could ever get real custom MLIR dialects — DECIDED, direction confirmed, not started

**Decided, 2026-09-17, directly by the user, following the same evening's own matmul-IPC investigation** (the roofline/`split_reduction`/`loop.unroll_and_jam` chain of entries above — the concrete, repeated experience of hitting melior/mlir-sys's own ceiling while chasing a real perf bug is what prompted the question). The framing that resolved it: the recurring friction is not "Rust was the wrong language" — any binding *other than* MLIR's own two first-party-supported consumers (native C++, and Python via its own upstream-maintained `pybind11` layer) hits the identical class of wall, since the C API every third-party binding (Rust included) goes through is a *deliberately* restricted subset, documented as such by the MLIR project itself, never intended as a full C++ re-exposure. The real question was never "Rust vs. some other host language" — it's "stay a permanent second-class consumer of a fast-moving upstream via community bindings, or stop."

**Decision: stop depending on `mlir-sys`/`melior`.** Not a full rewrite of cleave in C++ — the front-end (parsing, `infer.rs`, `monomorphize.rs`, CPS, `egraph.rs`'s own `egg`-based saturation, `refcount.rs`, `region_analysis.rs`) has zero MLIR dependency and stays exactly as it is, in Rust, unaffected. The replacement is scoped to exactly the layer that touches MLIR today (`mlir_lower.rs`, `pipeline.rs`, `dps_rewrite.rs`, `unify_alloc.rs`, the `ExecutionEngine`/`PassManager` wrappers) — a real, custom, narrow C++ shim, compiled from cleave's own build (the LLVM/MLIR C++ toolchain is already vendored and built from source for every developer today, `I:/Dev/llvm-project` — the marginal cost of also compiling a small, purpose-built shim against that same tree is far lower than it would be for a project with no existing C++ build step at all), exposing exactly what cleave needs as a real C API, called from Rust the same way `mlir-sys` is called today.

**Recommended sequencing, explicitly against a big-bang replacement**: start with the single narrowest, highest-value gap already measured this session — real `TargetMachine` construction for `mlirExecutionEngineCreate` (the `--target-cpu`/`--target-features` entry above, confirmed dead on arrival today) — as the first real shim function, proving the build/link/FFI shape on a minimal surface before deciding whether to extend to the Properties API gap (`build_matmul_transpose_no_seed`'s own workaround) or a real cleave-owned dialect (TableGen, now genuinely available once C++ is already in the build). Each subsequent capability added to the shim only when a concrete, already-measured cleave need demands it — not spec'd out fully in advance.

**Not started** — this entry records the decision and its scope, not an implementation. A real scoping/design pass, separate from this session, is the right next step before any shim code is written.

Raised directly, motivated by three separate, compounding pains: today's build depends on a hand-built local LLVM/MLIR/openmp toolchain (`I:/Dev/llvm-mlir-22`, no prebuilt Windows package exists anywhere, unlike Homebrew on Linux/Mac) *and* a locally-patched `mlir-sys` fork (`C:\dev\mlir-sys`, real Windows MSVC static-linking fixes, two upstream PRs open since June, neither addressed); `melior`/`mlir-sys` together expose no way to construct a custom `TargetMachine` (the `--target-cpu`/`--target-features` entry directly above) *or* to define a genuinely new MLIR dialect; and a second, sibling language is now planned, sharing enough of cleave's own architecture (and likely wanting its own domain-specific dialect) that this stops being a cleave-only concern.

**Is the C API's own dialect coverage exhaustive? Not the right question — the C API's design is the real ceiling, not `bindgen`'s coverage of it.** `mlir-sys` binds whatever `mlir-c/*.h` declares; if well maintained, that binding step itself loses nothing. What's genuinely incomplete is the C API surface itself (documented as such in MLIR's own `CAPI.md`: a deliberately curated subset for language bindings, never a full C++ re-exposure). Two clearly different capabilities live under "the C API," worth not conflating:
- **Building an operation of any dialect, including one the C API has no dialect-specific wrapper for at all** — exhaustive, structurally, because MLIR's own generic operation-construction path (`OperationState` + an op name string + generic operand/attribute/result lists) mirrors the generic textual assembly form (`"dialect.op"(...) : (...) -> (...)`) and needs no dialect-specific C API support. This is already the mechanism `melior`/`mlir-sys` lean on for most non-core ops.
- **Defining a brand-new dialect** (real types, verifiers, canonicalization/folding patterns, op interfaces the generic pass pipeline consults — `MemoryEffectOpInterface` for DCE/CSE, interfaces bufferization/inlining need, etc., custom assembly) — this is a C++/TableGen artifact today; the C API exposes no authoring surface for it at all, only consumption of a dialect already compiled into a C++ `Dialect` subclass. Not a `mlir-sys` gap — an MLIR architectural fact, unaffected by how thorough any binding layer is.
- Which specific dialect-scoped headers (`mlir-c/Dialect/*.h`) today's fork actually binds vs. omits is a real, separately checkable question (this fork's own `build.rs`/`wrapper.h` allowlist) — not answered here from memory, deliberately, rather than guessed.

**Can a new dialect be defined without TableGen at all? Partially, via a real, underused MLIR mechanism — IRDL — not via unregistered ops.** Two genuinely different paths exist, worth not conflating either:
- *Unregistered ops* (`allowUnregisteredDialects`) — emit `"mine.foo"(...)` with no real `mlir::Dialect` behind it at all. Works for construction/lowering, but zero verification, zero op interfaces (so invisible to every generic pass that keys off one), zero canonicalization.
- **IRDL** (the `irdl` dialect) — describes a *new* dialect's ops/types/constraints as MLIR IR itself (`irdl.dialect`/`irdl.operation` operations), loaded at runtime, no C++ compilation involved at all — and since IRDL's own defining ops are just ops, building them is fully reachable from the plain generic C-API construction path above. Gives a real, registered `Dialect` with structural (type/operand/result-constraint) verification. Real limitation, to verify against this project's own pinned MLIR 22 rather than assume from a possibly-stale general impression: last known state was structural-constraint-only — no arbitrary custom verifiers, no canonicalization/folding patterns, no custom assembly format. Whether that's still accurate for the exact pinned version needs checking directly against `I:/Dev/llvm-project`'s own IRDL docs/tests before this gets designed further, not assumed from a general sense of the feature.
- A Rust proc-macro dialect layer, concretely: generating ergonomic, typed Rust builder functions for a dialect's ops from a small Rust-side DSL (not TableGen `.td`) is real, buildable, bounded work — either calling the generic construction path directly, or emitting IRDL to get real registration+verification, both entirely C-API-only, no C++ required. **This full-Rust/IRDL shape is the recommended primary target, deliberately preferred over TableGen even where TableGen would reach further** — see the explicit comparison and reasoning below.

**TableGen automation via `build.rs` — real, genuinely better than a naive hand-invoked setup, but demoted to a documented, deliberately-not-built fallback, not the primary path.** Raised directly: since LLVM/MLIR gets rebuilt from source anyway (the redistribution track above), `mlir-tblgen` is already sitting in that same build tree — a `build.rs` step could invoke it directly (mechanical, the same shape as `prost-build`/`protoc`-style codegen-then-compile crates already use `cc::Build` for), generating the dialect's `.inc` fragments into a small, largely fixed hand-written C++ shell (the pattern every MLIR-internal dialect already uses), then compiling a thin `extern "C"` shim over it. The one genuinely strong point in its favor: doing this **at the same time, same toolchain, as the LLVM/MLIR core build itself** (whether producing the prebuilt tarball, or building locally from source) means the shim's C++ ABI is guaranteed to match what it links against — structurally avoiding the exact class of ABI/toolchain fragility (Windows MSVC static-linking bugs, already fought once in the current `mlir-sys` fork) that a separately-obtained `mlir-tblgen`/C++ toolchain could reintroduce. This reaches real TableGen-level power (real op traits/interfaces — `Pure`, `MemoryEffectOpInterface`, `InferTypeOpInterface` — wired into MLIR's own canonicalization/CSE/DCE/constant-folding, not just IRDL's structural verification), which IRDL, in its last known state, does not.

Despite that, **not the recommended primary path**: it reintroduces, at a smaller scale, exactly the class of fragility this whole exploration is trying to get away from — a real C++ toolchain becomes a hard, load-bearing requirement for anyone (not just an occasional contributor) adding or changing an op in the dialect, not merely an occasional convenience; the prebuilt tarball only hides this for consumers who never touch the dialect's own definition, and for an exploratory domain-specific dialect, that's likely to be often, by the people building it, not rarely. It also means running two parallel extension mechanisms side by side — TableGen `.td` for MLIR-side ops, cleave's own algebra+stdlib mechanism for everything else — where this project's own established discipline (one real extension mechanism, never a second bolted on alongside it) favors a single, uniform, Rust-native path instead, especially with a second, sibling language planned to reuse whichever story gets built here. Kept as a real, working, documented option — not built preemptively, triggered only by a concrete cleave-specific op that provably needs a capability IRDL structurally cannot provide (a real custom verifier beyond type/arity constraints, a canonicalization/folding pattern, custom assembly) — not built speculatively ahead of that need.

**Counter-consideration, raised directly and worth keeping on record rather than resolved away by the e-graph point below**: cleave already runs equational rewriting (its own `egg`-based e-graph saturation) *before* MLIR generation, which covers a real share of what dialect-level canonicalization would otherwise be doing — but a real custom dialect (even IRDL-only, structural-verification-level) would still be a genuine net gain beyond that, not a redundant layer: it buys native access to MLIR's own general-purpose framework facilities cleave gets none of today by lowering straight into existing dialects' own ops — real SSA-form guarantees and the analyses built on them (dominance, liveness, use-def chains), MLIR's own constant-folding and dataflow-analysis infrastructure, and a real place for cleave-specific op interfaces (e.g., a real `MemoryEffectOpInterface` on cleave's own ops) to make cleave-specific operations visible to MLIR's *existing* generic passes, not just to cleave's own hand-written lowering code. The e-graph phase and a real dialect are complementary, not substitutes — the e-graph handles cleave's own domain-specific equational rewriting pre-lowering; a real dialect would extend what MLIR's own general infrastructure can see and do *after* lowering, on cleave's own vocabulary rather than only on what it's already been rewritten down into.

**Prebuilt-LLVM-per-platform redistribution — real, but the cost is release engineering, not Rust plumbing.** The Rust side (a `build.rs` that downloads a hash-verified per-platform tarball from a hosting location like GitHub Releases, caches it locally) is small, maybe a day of work — crates.io's own size limits rule out embedding gigabyte binaries directly in a published crate, so this is the realistic shape, not an alternative choice. The real, recurring cost is producing and hosting the tarballs themselves: at minimum Windows MSVC / Linux glibc / macOS x86_64+arm64 (4-5 build-matrix legs), each a multi-hour from-scratch LLVM+MLIR(+openmp) build, **repeated on every LLVM/MLIR version bump, indefinitely** — a standing release-engineering commitment comparable in ongoing size to maintaining the `mlir-sys` fork itself, not a one-time project.

**Complexity of the three concrete asks, and a recommended sequencing:**
1. *Rework `mlir-sys` (redistribution + filling gaps)* — moderate, decomposable, low risk to cleave itself. Filling genuine `mlir-sys`-side binding gaps (headers not yet wired into `bindgen`) is mechanical. Filling a gap like the `TargetMachine`/`--target-cpu` one above is *not* a `mlir-sys`-side fix at all — the missing entry point doesn't exist in MLIR's own C API, so this specifically needs a patch one layer further up, against the vendored LLVM/MLIR C++ source itself (a real new C API function, then bound), not just against the binding crate. Redistribution is real, valuable, and separable from the binding-gap work — worth pursuing on its own track. Given two real fixes already sitting unaddressed upstream since June, owning this fork outright as a permanent, private dependency (rather than continuing to hope for eventual upstream merges) is the more honest posture already, independent of any of this.
2. *A new layer over/replacing `melior`, with proc-macro dialect support* — the most open-ended of the three; real, bounded value exists (ergonomic builders, IRDL-based dialect definition) at the primary, full-Rust/IRDL target — full TableGen-parity authoring is reachable too (via the `build.rs`-automated fallback above) but deliberately not the default path, for the reasons above. Verifying IRDL's actual current capability against the exact pinned MLIR 22 tree is the real, cheap first step, and now doubly load-bearing: the whole full-Rust-primary bet rides on IRDL being real/capable enough for cleave's own dialect, not merely "acceptable as a way to avoid C++."
3. *Retarget cleave onto the result* — near-zero cost if scope stays "swap what's underneath `mlir-sys`, keep `melior`" (the generic op-construction path `mlir_lower.rs` already uses doesn't change regardless of what's happening at the distribution/binding layer beneath it). A real, multi-week-scale port if `melior` itself gets replaced — `mlir_lower.rs` is already large (struct-field packing, tensor/memref/math/scf/openmp lowering) and would need full parity with today's coverage before cleave could build again at all.

Recommended order, not yet started: (1) alone first — it pays off immediately with zero retarget risk. Investigate IRDL's real current capability against the pinned MLIR 22 tree next, in isolation, before committing to any proc-macro dialect layer — full-Rust/IRDL is the target to build toward, with the `build.rs`-driven TableGen+shim path kept in reserve, documented, not built, for the specific day a concrete op needs a capability IRDL structurally lacks. Don't touch cleave's own retarget until the dialect layer (whichever of the two it ends up needing) has been validated standalone, independent of cleave's own progress.

**Sequencing refined by the user, same week — binary redistribution moves *into* the shim's own build, not the `mlir-sys` fork.** Investing real effort polishing `mlir-sys`'s own redistribution story doesn't make sense given the fork itself is the thing being replaced — that work would be thrown away the moment the shim lands. Revised near-term plan: (a) a real but deliberately cheap stopgap right now — smooth over the current per-developer `MLIR_SYS_220_PREFIX`/`TABLEGEN_220_PREFIX` env-var setup pain (today's actual friction point, `.cargo/config.toml`'s own doc comment already explains what to set but not how to make it easy) so `mlir-sys` stays usable day to day while the shim doesn't exist yet; (b) build the real target directly — the shim + a vendored/prebuilt LLVM, with binary redistribution designed as part of *its* own build from the start, `melior` and `mlir-sys` dropped entirely once it lands, not kept alongside it. Not yet started; the concrete shape of (a) hasn't been decided (a helper script vs. a `build.rs` auto-detection fallback vs. something else) as of this entry.

**(a)'s own shape decided, same conversation: a proc-macro for the ergonomic FFI surface, a genuinely minimal `build.rs` for the one thing only `build.rs` can do.** A proc-macro (pest's/`cxx`'s own pattern — reads a file/env at macro-expansion time, generates code inline, no `OUT_DIR`/`include!(concat!(...))` dance) can do all of the codegen side, but Cargo structurally only lets a `build.rs` emit `cargo:rustc-link-lib`/`cargo:rustc-link-search` — no proc-macro can. Real shape: a small `-sys`-style crate whose *only* `build.rs` job is locate/download the precompiled shim package and emit those two link directives (nothing else — no more figuring out where headers/paths are, that moves into the proc-macro side entirely); a proc-macro crate (or the same crate) does the actual ergonomic binding generation. `cxx` (Rust/C++ interop: proc-macro bridge declarations + `cxx_build` as the thin build.rs) is close-to-directly-applicable prior art for this exact split, worth evaluating before hand-rolling it. **`openblas-src` flagged as hitting the identical problem shape** (locate/vendor a precompiled external C library, minimal `build.rs`, everything else ergonomic) — a real, established crate (used by `ndarray-linalg`) that could plausibly serve as a second data point or even be reused directly for the OpenBLAS integration itself; its current maintenance state not yet checked.

**Toolchain choice for the shim's own C++ build, decided**: stay on Rust's default `x86_64-pc-windows-msvc` target rather than switching to `-gnu` (MinGW) to avoid MSVC — `clang-cl` (clang's own MSVC-ABI-compatible driver mode) plus `lld-link` (LLVM's own MSVC-compatible linker) let the shim's C++ side be built entirely with the LLVM toolchain while staying ABI-compatible with Rust's own default Windows target, avoiding the `-gnu` target's own rougher, less-tested path for an LLVM/MLIR build on Windows. Not "stuck with Microsoft's compiler" as first framed — the real constraint is ABI/CRT consistency, and `clang-cl` satisfies it without requiring `cl.exe`.

**Where the toolchain build+release itself should live, decided**: not a from-scratch repo — a genuine fork of `llvm/llvm-project` on GitHub, tentatively named `cleave-llvm-redist`, with `.github/workflows/ci.yml`'s own existing build steps (already proven, `ci/llvm-cmake-flags.txt`) moved there and a release-publishing step added (compress the install prefix, attach as a GitHub Release asset on a tag — independent of, and freely coexisting with, cleave's own separate `v*` releases in its own repo; a GitHub repo can hold as many independent tag-scoped releases as needed, no conflict either way). A fork rather than an unrelated new repo specifically because it keeps a real git relationship to upstream LLVM (trivial to track new tags), and forking costs nothing of consequence in practice (GitHub's own shared "fork network" storage; `llvm/llvm-project` already has thousands of forks) — the CI's own existing `--depth 1` shallow clone during a build is a separate, already-true-today concern (how the *runner* fetches source at build time) and doesn't bear on the fork's own storage cost either way. **The sibling language now has a tentative name, `cleave-gfx`** (graphics-domain, per the name) — first concrete naming signal for what it actually is, beyond "a second language sharing cleave's architecture."

## The conclusion of the exploration above: two concrete, genuinely shared dialects — CPS and an e-graph term vocabulary — not a general "mutualize everything" bet

Motivated by a second, sibling rendering/compute language now being designed alongside cleave, sharing enough of its architecture (MLIR backend, `egg`-based equational rewriting, a CPS-shaped IR) that a first instinct was to mutualize the whole backend wholesale. Narrowed down, directly, to two specific dialects worth building shared — each justified on its own concrete merits, not on general proximity between the two languages, and each doing double duty rather than serving only one side.

**A shared CPS dialect.** Real suspend/resume semantics for the rendering language's own `await`/`Promise<T>` (an `extern fn` returning `Promise<T>` rather than `T` marks a host-async call structurally, in its type, not via a separate declaration form or hidden convention — the CPS pass recognizes `Promise<T>` wherever it appears and treats `await` on it as the suspension operator) — but also, independently, a real and already-known cleave-side gap this same dialect would close: `Forward::walk` (`egraph.rs`) stops translating entirely the instant a CPS body meets a real `If` (confirmed directly by a standing test, `a_body_starting_with_if_translates_nothing` — `env` stays empty, the whole `If` untouched), and the existing "Multi-level call transparency in the e-graph forward translator" backlog-done entry already tried and confirmed, by direct testing (`examples/join_demo.cleave`), that a `phi(a,a)==a` rewrite rule alone doesn't fix this — nothing ever built the `phi` node for it to match against, since translation bails before reaching one. A CPS dialect built on real MLIR block arguments at join points supplies exactly the missing piece — a real, structurally-guaranteed join representation — for free, from MLIR's own SSA-region machinery, rather than needing a hand-rolled phi-construction step grafted onto cleave's own bespoke pre-MLIR CPS pass. Two genuinely separate problems (real external asynchrony for one language; a known equality-saturation gap in the other), same dialect, same underlying mechanism (suspend/join with correct SSA merge semantics) solving both.

**A shared e-graph term-vocabulary dialect** (the "formalize the expression language, not the live e-graph state" idea from this same exploration — the vocabulary is declared once, `egg`'s own solver stays an internal, opaque black-box pass consuming and producing instances of that vocabulary, never itself expressed as MLIR IR). Solves a real, concrete duplication risk: today, `egg`'s own term language (`CleaveLang`, `egraph.rs`) and what `mlir_lower.rs` knows how to lower are two independently hand-written, hand-synchronized things. A single declared schema generating both directions removes that sync burden. **And it depends on the CPS dialect above for its own biggest win**: only once CPS gives a real, join-aware IR does equality saturation have anything sound to saturate *across* a branch at all — the two dialects are complementary for this specific capability, neither alone gets there.

**A real asymmetry between the two languages, worth keeping on record precisely because it's a point *in favor* of sharing, not against it**: the *axiom set* fed into the e-graph dialect is expected to stay genuinely dynamic/open-ended for cleave — grown by whatever `derivative`/`adjoint`/algebra rules a given cleave program itself declares (`grad`/`derive`'s own extensibility, already real and stdlib-driven) — while the rendering language's own axiom set is more likely closed and static (a fixed, stdlib-authored set of hardware/rendering identities, not meant to grow per end-user program the same way). Conversely, **CPS usage is expected to be the more elaborate one on the rendering side** — real suspension across a genuine external asynchronous round-trip, live for real time, not just cleave's own current, more constrained use (structuring a derivative computation, no actual external asynchrony involved at all). Each language ends up stressing a different facet of the same two shared mechanisms rather than needing identical behavior from either — a real, concrete reason the sharing is sound, not just convenient because the two projects happen to be developed together.

Not built — gated on the same open validation step as the rest of this exploration (IRDL's real capability against the pinned MLIR 22 tree), and subject to the same documented, not-built TableGen+`build.rs` fallback for whichever specific op needs a capability IRDL structurally lacks (`cps.await` needing a real `MemoryEffectOpInterface` for correct behavior under MLIR's own generic passes is the concrete candidate already flagged above, worth watching for first).

## A real, separately-compiled, linkable *cleave-to-cleave* library format (metadata alongside compiled code)

Not "produce a real executable" — that shipped a while ago (`--emit-exe`/`cleave-build`, real and heavily used, see `doc/backlog-done.md`'s own corrected entry; this item's own title used to say otherwise and had gone stale). This is the narrower, still genuinely open question underneath it: does a "cleave library" — separately-compiled, linkable, with no source required downstream — make sense at all, given total monomorphization? A `.o` alone carries no metadata (declared signatures, generic templates, algebra impls) for a *downstream* cleave program to monomorphize *against* — unlike Rust's own `.rlib`/`.rmeta` split, which exists precisely to carry that metadata alongside compiled code. Without an equivalent, either (a) cross-crate cleave code always recompiles from source (the current `stdlib` model, extended — works, but risks "mega long" whole-program compile times, and doesn't produce a distributable library at all), or (b) a real compiled-library-plus-metadata format gets designed, a genuinely new, nontrivial piece of infrastructure. `extern fn`/`export fn` already covers the *other* interop story fine (calling into/out of compiled cleave across a language boundary to Rust, already built and working, `doc/backlog-done.md`) — this item is specifically about a *cleave-to-cleave* separately-compiled library, a different problem, needing a real design pass once the language itself is more stable. Not attempted.

## A lambda returned from a function, or stored in a struct/array field

The other two shapes originally bundled with "Calling a lambda literal directly" (now Done, see above) — a lambda value that outlives the single call site its own closure-conversion unit is specialized against. Unlike calling a lambda literal directly (pure desugaring, no new runtime representation needed), these two are a genuinely harder problem: `hld.md`'s own "no runtime closure ABI" design (see "Closure conversion (lambdas)" above) means a lambda never becomes a real runtime value today — every call to one is resolved to a concrete, statically-known unit at compile time. A lambda *returned* from a function (potentially different lambdas on different branches) or stored in a struct/array field (potentially read back and called from code with no static knowledge of which lambda it is) both need some real runtime representation to call through — a function pointer plus a captures record, at minimum — not just more of the "resolve by static name" machinery `monomorphize.rs`/`cps.rs` already have. Needs its own design pass, not attempted yet.

## No reserved-word list at all

A keyword (`let`, `if`, `and`, `not`, ...) is only special at its own specific grammar position — nothing stops it from *also* being used as an ordinary identifier elsewhere (`let and = 5;` presumably parses). Distinct from the word-boundary bug already fixed (see "Done" above, boolean logic): that was about a keyword wrongly matching as a *prefix* of a longer identifier; this is about a keyword being usable as a *complete* identifier at all. Not urgent, noted so it isn't lost.

## A size-class free-list pool for `cleave_alloc_rc` — landed; it surfaced three pre-existing `refcount.rs` bugs, two now fixed (including the crash), the third (a leak) documented and still open

Started from the user's own "pool for the CPS-continuation-carried slots" idea (a *former* version of this entry — see git history — sketched a bespoke, per-allocation-site, depth-bounded pool requiring a new static "is this loop-carried" classification). That design was **replaced before being built** by a simpler, general one, once re-examining the goal ("on veut juste payer le malloc qu'une fois") showed the per-site classification was solving the wrong problem: the real requirement is just "the same handful of sizes recur every iteration, so cache by size class" — which needs no new CPS-level analysis at all, and benefits every `cleave_alloc_rc` caller in the program, not just the one motivating loop.

**What actually landed, in `cleave-rt/src/lib.rs`**: `cleave_alloc_rc`/`cleave_release`'s own non-arena path now goes through a segregated free-list cache, one intrusive singly-linked list per power-of-two size class (`size_class`/`class_bytes`), guarded by a spinlock (`POOL_LOCK`) — a real, found-by-testing requirement, not caution for its own sake: `cleave_alloc_rc`/`cleave_release` calls exist *inside* several `..omp_par.N` OpenMP-outlined parallel regions (per-thread scratch tensors), genuinely reachable from multiple worker threads at once whenever `OMP_NUM_THREADS > 1` — confirmed directly against the real kernel's own disassembled `.o`, not assumed.

**Three real, confirmed-independently bugs in `cleave/src/refcount.rs` surfaced by the pool** (all pre-existing, unrelated to each other, all invisible under the *old* allocator because none manifests without the pool's own immediate, deterministic block reuse — the stray writes they caused landed in memory the old allocator had already handed back to the OS). Item 3 is the crash; item 1 was fixed en route; item 2 (a leak) had its fix attempt reverted and is still open:
1. **A genuine over-release for identity-preserving loop-carried values.** `Optimizer::step<Sgd, ...>` returns `state` unchanged (`Sgd` is a stateless optimizer — it re-wraps `state`'s own existing leaves, never computes a new value for them) — the training loop's own back-edge unconditionally releases the *old* carried state's own leaves every iteration, correct for `net` (genuinely recomputed each step) but one release too many for `state` specifically, since old-and-new are the exact same object. Root-caused precisely (not guessed) by tracing `--dump-cps-optimized` variable by variable. Fixed via a new `param_leaf_key` analysis (`refcount.rs`) that recognizes, at a function's own true return, when a returned leaf still traces back — through any number of transparent `Field`/`StructCtor` hops — to one of that function's own parameters, and skips exactly the first of the (pre-existing, harmless-until-now) redundant release occurrences for it, restoring the same balance a genuinely fresh value already had. Verified by direct count on the real kernel's own dump (`retains=2, releases=1` for `state`'s own leaves inside `Optimizer::step`, matching the `-1` its own caller already contributes).
2. **A real, unbounded leak** (confirmed, not just a corruption theory): a freshly-computed value passed as a literal argument to an ordinary function call (`Scale::scale(lr, grad)`'s own result, fed straight into `Ring::sub`, never read again) is never released by anyone — the caller's own bookkeeping "transfers" it to the call and stops tracking it, and the callee (an ordinary top-level function) never releases its own borrowed parameters, by this project's own established convention. **A fix was attempted and reverted** — see the comment left in place at `refcount.rs`'s own `rewrite_body::Fix` arm, right where the attempt lived: seeding the argument for release at the callee's own resumption, guarded on it not being one of the resumption's own free variables (protecting a genuinely-still-needed one) and on the enclosing function not being `region_analysis::find_region_local_functions`-local (arena-backed values are bulk-reclaimed regardless, so this fix's own motivating leak never applies there), still produced a real, reproducible `STATUS_ACCESS_VIOLATION` — confirmed, by disabling the fix outright, to be the fix's own doing. The real remaining flaw in that approach was not identified before the fix was reverted; the leak itself is real and still open.

3. **The crash — a compiler-emitted double `cleave_release` on the same allocation, in `cleave/src/refcount.rs`, now fixed.** The pool's free-list would acquire a garbage "next" link (a valid pointer ±1, so *misaligned* — ends in an odd nibble — not the `0xFFFF...` first guessed), crashing the *next* pop of that size class with `STATUS_ACCESS_VIOLATION`. Root-caused precisely this time, with real tooling: `lldb` now works (`winget install Python.Python.3.10` supplies the `python310.dll` `liblldb.dll` needs), and `CARGO_PROFILE_RELEASE_DEBUG=true` gives source-level symbols for `cleave-rt`. Temporary instrumentation in the pool (an alignment assert on every pop/push, a `!(base)` canary in bytes `[8..16]` of every parked block, and a `PARKED` set of currently-free block bases that `cleave_retain`/`cleave_release` consult) pinned it to *"`cleave_release` called on a block already freed and pooled"* — the canary intact but the next-pointer off by one, the signature of a stray `cleave_release` doing `refcount -= 1` on what is now the free-list link. `lldb` + disassembly of the faulting `train_and_evaluate` offset showed **two `call cleave_release` back-to-back on the same register**, once per epoch, right after the `Epoch=` print. Source: `println(("Epoch=", epoch))`. `stdlib/io`'s `Print::print<(A,B)>` and `println` are both `fn(x) -> x` — they return their argument unchanged — so the caller's `--dump-cps-optimized` showed `(release <println result>)` *and* `(release <the tuple it constructed>)`, two releases of one allocation. `refcount.rs`'s `rewrite_body::Fix` arm seeds every *transferred* literal call-argument into the call's own resumption for release; when that argument shares the resumption's return-value parameter's type, the callee may be handing that very allocation back (`Print::print`/`println` do), so the return-value parameter — already seeded, already released — covers it and the argument must not be seeded again. Guarded to only skip an argument the call is genuinely the *last* use of (not a free variable of the resumption): `net_grad: Network -> Network` also type-matches but returns a *fresh* gradient and its `net` argument is still needed by the following `Optimizer::step` in the same resumption, so it stays seeded. Verified by exact release-count diff on the real kernel's own dump (exactly one release removed, `v1791`, everything else identical), full `cargo test -p cleave` green, and real forced-clean AOT runs of both examples (`digits-interop` `0.94713414`; `mnist-interop` 10 epochs `0.9342`, no crash, OpenMP on). All the temporary pool instrumentation was removed afterward — `git diff` clean of everything but the `refcount.rs` fix. **A false lead worth recording**: the first cut of the `PARKED` set updated it *outside* `POOL_LOCK`, which raced concurrent OpenMP-worker pops/pushes and reported spurious "stale release"s in `digits-interop`; maintaining it inside the spinlock section, atomically with the `FREE_LISTS` mutation, made those vanish.

Item 2 above (the `Scale::scale` transferred-argument leak) is **unrelated to item 3 and still open** — item 3's fix *removes* a seed in the same arm, item 2's reverted attempt *added* one; they don't interact, and the leak is real.

**Net effect on the real kernel**: `RtlAllocateHeap`/`RtlFreeHeap` traffic for any repeated size is now paid at most once per process lifetime rather than every allocation.

### Reference profile — 2026-09-10, `mnist-interop`, post-pool + post-double-free-fix

Recorded as the baseline the next round of work (see below) is measured against. VTune 2025.8, **software** (user-mode) sampling — this machine is AMD, hardware event-based sampling is unavailable ("cannot recognize the processor"); sw-sampling inflates absolute CPU-seconds ~3–4× over a bare run, so read the **proportions**, not the wall-clock. Built `CARGO_PROFILE_RELEASE_DEBUG=true` (gives `cleave-rt` source symbols; `kernel.o` still has none, so the whole MLIR kernel shows as one flat `train_and_evaluate` symbol).

**Bare wall-clock** (`cargo run --release -p mnist-interop`, 10 epochs, accuracy `0.9342` in every configuration), a `OMP_NUM_THREADS` sweep on this 16-logical-CPU box:

| threads | elapsed | speedup vs 1T |
|--------:|--------:|--------------:|
| 1 (`CLEAVE_NO_OPENMP=1`) | ~18.8 s | 1.00× |
| 2 | 13.3 s | 1.43× |
| 4 | **12.1 s** | 1.56× |
| 8 | 18.0 s | 1.05× |
| 16 | 28.4 s | 0.66× |

Scaling peaks at ~1.6× on 4 threads and goes **negative** past that.

**VTune hotspots, 1 thread** (`train_and_evaluate` = the MLIR compute kernel; no spin time at all):

| symbol | self | share |
|---|--:|--:|
| `train_and_evaluate` | 17.06 s | 86.1 % |
| `memcpy`/`memmove` (`VCRUNTIME140`) + `cleave_rt::memrefCopy` | 1.19 s | 6.0 % |
| `RtlAllocateHeap` + `RtlFreeHeap` | 0.37 s | **1.9 %** (was ~12 % pre-pool) |
| `POOL_LOCK` (`AtomicBool::compare_exchange_weak`) | 0.21 s | ~1.0 % |
| host-side MNIST unpack (`data::train_pixel`) + driver-loop scaffolding | ~0.55 s | ~2.8 % |

The pool did its job: system-allocator cost fell from ~12 % to ~2 %, and the mono-thread kernel is now genuinely **compute-bound** (86 % in the kernel itself). `memcpy` (~6 %) is the remaining non-compute cost, consistent with the `dps_rewrite` `Strategy::Passthrough` fix already having landed.

**VTune hotspots, 4 threads** (267.9 s CPU / 67.5 s elapsed *under the profiler*; Effective Time 29.8 s = **11.1 %**, Spin 190.3 s = **71 %**, Overhead 47.8 s = **17.8 %**):

| symbol | CPU time | share |
|---|--:|--:|
| `libomp` spin-wait (`func@0x18001d04f`) | 121.0 s | 45.2 % |
| `_kmpc_barrier` | 98.4 s | 36.7 % |
| **`train_and_evaluate`** (real compute) | 16.5 s | **6.2 %** |
| `libomp` (`func@0x18001cb40`) | 9.4 s | 3.5 % |
| `_kmp_fork_call` (parallel-region creation) | 9.1 s | 3.4 % |

~85 % of CPU is pure OpenMP barrier + spin + fork/join; only ~6 % is compute. VTune's own diagnostic: *"CPU time spent on parallel work arrangement can be a result of too fine-grain parallelism. Try parallelizing outer loops rather than inner loops."* The pool spinlock is **not** the cause — `RtlAllocateHeap`/`compare_exchange_weak`/`cleave_alloc_rc` are all buried in `[Others]` at < 5 % combined; the earlier "`POOL_LOCK` contention is the parallel-scaling killer" hypothesis is **disproven** by this profile.

**The real finding, and the next work item**: cleave's `--convert-scf-to-openmp` parallelizes at far too fine a grain — the generated kernel forks a parallel region and hits a barrier around individual small ops inside the per-batch inner loop (1875 batches × 10 epochs × many parallelized ops = on the order of millions of fork/join + barrier round-trips), each doing microseconds of work on the tiny MLP layers at batch 32. This — not allocation, not `memcpy` — is the OpenMP-efficiency gap vs PyTorch (whose persistent threadpool dispatches coarse tasks). Fixing it means parallelizing an *outer* loop (the batch loop, or a tile band spanning many rows) once per region instead of the innermost linalg op, and/or keeping a worker pool alive across regions. Re-measure this exact sweep + both VTune profiles after any such change.

### Reference profile — 2026-09-17, `mnist-interop`, post-§11/§13/§14 (pool-cascade for never-mutated nested structs, `CLEAVE_AFFINE_STRUCTS` flipped on by default)

Mono-thread only (`CLEAVE_NO_OPENMP=1`, forced-clean rebuild) — the OpenMP sweep above is untouched by this round of work by design (memory-management and parallel scheduling are orthogonal axes; see `doc/plan-affine-ownership.md`). `vtune -collect hotspots`, same machine, same methodology as the 2026-09-10 reference above.

| symbol | CPU time | share |
|---|--:|--:|
| `train_and_evaluate` | 14.397 s | 78.1 % |
| `func@...` (`VCRUNTIME140.dll`, unresolved — `memcpy`-shaped, per this project's own established reading) | 1.276 s | 6.9 % |
| `memrefCopy` | 1.100 s | 6.0 % |
| `train_pixel` (host-side MNIST unpack) | 0.496 s | 2.7 % |
| `cleave_release` | 0.331 s | 1.8 % |
| `[Others]` | 0.842 s | 4.6 % |

Elapsed 18.58 s (CPU 18.44 s, 1 thread — no spin, matches elapsed almost exactly, unlike the inflated multi-thread CPU-seconds this file's own methodology note warns about).

**Real, absolute win inside `train_and_evaluate` itself**: `17.06 s -> 14.397 s`, **-15.6 %**, consistent with tonight's pool-cascade work removing header allocations and shortening release chains for the struct values this giant inlined function constructs and tears down every batch. But the *share* dropped (`86.1 % -> 78.1 %`) rather than held or grown, because the non-compute pieces did **not** shrink proportionally — `memrefCopy` is flat (`1.19 s -> ~1.10-1.28 s` combined with the VCRUNTIME frame), so as the compute numerator shrinks, this roughly-fixed copy cost eats a growing slice of a shrinking pie — the identical "shrinking base" dynamic this file's own OpenMP entries already named from the other side (`speedup ratio shrinks as the per-region cost shrinks`). Still, `78.1 % + 2.7 %` (`train_pixel`, also real work, not overhead) `= 80.8 %` in genuinely useful work — matching this round's own stated target ("le maximum de temps dans le code de calcul... on a déjà 80%+") almost exactly.

**Net reading**: the pool-cascade work is a real, measured, absolute single-thread win, not just a correctness/memory fix — but the next lever for *mono-thread* time is now unambiguously the `memcpy`/`memrefCopy` pair (`~12.9%` combined), not allocation (`cleave_release` alone is `1.8%`, already small). The OpenMP-scaling gap above remains completely untouched and is still the dominant, order-of-magnitude-larger opportunity once threads are involved.

## The generated kernel object now carries CodeView debug info, including real inline-site records for functions inlined away entirely — `train_and_evaluate` + every `..omp_par.N` region resolve by name, `matmul`/`net_grad`/... resolve by their own source line via `S_INLINESITE`, a real crash found and fixed along the way

Motivated directly by the profile above: every VTune/AMDuProf run so far attributed 100% of compute to one flat `train_and_evaluate` symbol with no line information, because `mlir_lower.rs` built every op with `Location::unknown` and the object went out with no `.debug$*` sections at all. Root-causing the `println` double-free earlier this session took ~30 tool calls of `llvm-objdump` + hand-correlation for want of exactly this.

**What landed, all in-process — no `mlir-opt` subprocess, no C-API DI-attr builder (this `mlir-sys` fork exposes none):**
- Every `Location::unknown` in `mlir_lower.rs`/`dps_rewrite.rs` replaced with a real `FileLineColLoc`. This is a hard prerequisite, not cosmetic: once a function carries a `DISubprogram`, the LLVM verifier rejects any inlinable call inside it whose location has no `!dbg`.
- `build_di_subprograms` (`mlir_lower.rs`) constructs one `#llvm.di_subprogram` per defined `llvm.func` from **text**, parsed as a single array attribute in one `Attribute::parse` call so the `distinct[0]<>` compile-unit id unifies to one shared `DICompileUnit` (separate parses would each mint their own — the LLVM verifier then rejects "multiple debug compile units"). `pipeline.rs::stamp_di_subprograms` attaches each via a `FusedLoc` on the `llvm.func`, run **last**, after every hand-walk (`unify_tensor_allocations`, `stamp_target_cpu`), so nothing inserts an un-located op into an already-`DISubprogram`'d function. This reimplements the one thing MLIR's `EnsureDebugInfoScopeOnLLVMFunc` pass does — that pass **segfaults** (`STATUS_ACCESS_VIOLATION`) invoked from this statically-linked `mlir-sys`, whether via melior's `create_di_scope_for_llvm_func_op_pass()` or `parse_pass_pipeline` after `register_all_passes()`; it runs fine as `mlir-opt --ensure-debug-info-scope-on-llvm-func` in a separate process, so the fault is almost certainly a dead-stripped `PassRegistration` static initializer in the MSVC static link, not the pass logic.
- An `llvm.module_flags` op with `CodeView = 1` + `Debug Info Version = 3`, built via `OperationBuilder` (no melior wrapper for it). Without the `CodeView` flag the backend emits DWARF `.debug_line`, which `lld-link` ignores for PDB purposes; with it, `.debug$S`/`.debug$T`, which `lld-link` folds straight into `mnist_interop.pdb` alongside the Rust `.debug$S` that `CARGO_PROFILE_RELEASE_DEBUG=true` already produces — zero extra linker config.
- `mlirExecutionEngineDumpToObjectFile` **does** emit the CodeView sections once the DI metadata + module flag are present (an earlier "it strips debug info" conclusion was a stale-object artifact).

**Source lines, two tiers:**
- **Per-function (robust, survives everything):** `CTopLevelFn::line`, set once in `convert_program` from the first AST span inside the body (the `fn` keyword has no node), threaded unchanged through `synthesize_derivatives`/`optimize_program`/`eliminate_dead_code`/`rc_opt`/`insert_refcounting` (all rebuild bodies but keep the struct). `train_and_evaluate` now resolves in `lldb` as `at kernel.cleave:218` — its first real statement.
- **Per-statement (best-effort):** `CpsProgram::op_lines: HashMap<CVar, u32>`, populated at the 13 `LetPrim` construction sites in `convert_program` (`convert_expr` tracks the innermost expression span in a `Cell`), carried through every pass, consumed by a process-global `GEN_LINE` that `mlir_lower.rs::lower_cexpr`'s `LetPrim` arm stamps before lowering each op. `train_and_evaluate`'s CodeView line table now spans `218, 223, 231, 233, 243, 244, 247` — its actual statements, not one flat line.

**The real root cause, found by direct measurement, and the fix — inlined functions now keep real, separately-attributable debug info:**

A `count_locs`/`has_real_line` walk inserted between every pass in `lower_to_llvm` (removed after use) first looked like it confirmed the obvious guess: `located` count crashes right after the very first pass (`--inline`). It didn't. A *shallow* `is_file_line_col_range()` check reports "unknown" for a `CallSiteLoc` even when the real line is sitting one level down in its own `callee` field — `--inline` wraps a cloned op's already-good location in `CallSiteLoc(calleeLoc, callerLoc)`, it never destroys it. `has_real_line` (recursing into `CallSiteLoc`/`FusedLoc`) showed the true picture: ~97% of ops resolve to a real line all the way through the pipeline. **The debug info was never lost. It had nowhere to attach a scope.** LLVM's translator needs the *callee* side of an inlined `CallSiteLoc` chain to already carry a `DISubprogram` to emit a real "inlined at" `DILocation` — and the previous design (`stamp_di_subprograms`, run once at the very end, on whichever `llvm.func`s survived) only ever gave a subprogram to `train_and_evaluate` + `dealloc_helper`. `matmul`/`net_grad`/`relu`/... never got one (they'd already been inlined away by then), so their otherwise-perfectly-good `CallSiteLoc` chains had no scope to resolve against, and the translator fell back to the one available scope: the enclosing (post-inline) function.

**The fix**: give *every* function its own `DISubprogram` **before** `--inline` runs, fused onto **every op** `gen_loc` stamps for it (not just the function's own top-level declaration op — that wrapper disappears once the function is inlined away, but each individual op's own location survives, wrapped in `CallSiteLoc`, and now carries its own scope from the start).
- `mlir_lower.rs::lower_program` now calls `build_di_subprograms` up front, once, for *all* of `program.funcs` (previously: only for the survivors, at the very end).
- `Attribute<'c>` can't live in a `static` (not `Send`, lifetime tied to `Context`) the way `GEN_LINE`/`GEN_FILE` already do, so `set_gen_subprogram`/`GEN_SUBPROGRAM` stash the attribute's own raw `MlirAttribute.ptr` as a `usize` instead — sound for the span of one `lower_program` call, the only place it's read back (`gen_loc`'s own doc comment has the full safety argument). `lower_program`'s per-function loop sets it before lowering each function's body; `gen_loc` fuses it onto the `FileLineColLoc` it would have returned anyway.
- This made the old post-hoc `stamp_di_subprograms`/`collect_defined_llvm_func_names` (attach a subprogram to whichever `llvm.func` survived) **entirely redundant** — deleted. What's left, `backfill_all_unknown_locs`, only does the one thing still needed: give a genuinely bare `UnknownLoc` op (one an MLIR lowering pass synthesized from scratch — the tiled/vectorized loop nests, mostly) the enclosing (already well-scoped) function's location as a floor.

**A real crash found and fixed along the way**: turning this on unconditionally first produced a reproducible `STATUS_ACCESS_VIOLATION` inside the `cleave-build` build script — not a graceful `Err`. The actual LLVM verifier complaint, buried in the raw stderr: `"function declaration may only have a unique !dbg attachment"`, once per `cleave-rt` extern (`cleave_alloc_rc`, `print_i32`, `train_pixel`, ...). `ensure_extern_declared` (`mlir_lower.rs`) builds these as real declarations (empty region, no body) but was calling the same `gen_loc()` as everything else — so the first cleave function to reference a given extern symbol fused *its own* `DISubprogram` onto that extern's declaration, which LLVM disallows (a declaration has no body to scope debug info to in the first place). Fixed by giving `ensure_extern_declared` a bare, un-fused `Location::new(...)` instead of `gen_loc()`.

**Verified, on the real, cleanly-rebuilt `mnist-interop` kernel.o** (not a synthetic probe):
- `llvm-readobj --codeview kernel.o` → **193 real `S_INLINESITE` records** (an isolated `emissionKind = Full` experiment on the same IR via `llc` directly showed 585 — `mlirExecutionEngineDumpToObjectFile`'s own codegen is a bit more conservative but demonstrably emits the same kind of record, not zero).
- `lldb`: `b kernel.cleave:158` (`MatMul::matmul`'s own real source line — a function with no standalone symbol left anywhere in the binary) → **`Breakpoint 1: 5 locations`** — 5 real, distinct addresses, one per generic instantiation actually inlined into the kernel. A breakpoint on an inlined-away function's own source line genuinely works.
- `image dump line-table kernel.cleave` (the *flat* PC→line view, not an inline-site query) still only shows `train_and_evaluate`'s own ~8 driver-loop lines — expected and not a regression: a flat line table is defined to report the outermost attribution; the per-inlined-function detail lives in the `S_INLINESITE` records instead, exactly what `b kernel.cleave:158` above reads. Checking for it via the wrong query (`image lookup -v -a <addr>`'s "Blocks:" listing, which lldb's own PDB/Windows inline-frame support doesn't populate) is what made this look broken longer than it needed to; VTune, a far more mature CodeView consumer, is expected to show real per-inlined-function source attribution now that the records genuinely exist in the object.
- Debugging the printer-vs-diagnostic distinction along the way: `Operation::to_string()` (melior's default) **never prints locations at all** (matches `mlir-opt`'s own default) — `count_locs`'s first, wildly-wrong readings on a *dumped* module were this, not real data loss. `OperationPrintingFlags::new().enable_debug_info(true, false)` + `to_string_with_flags` is required to see them.
- Full `cargo test -p cleave --release --no-fail-fast` green; `mnist-interop` 10 epochs `0.9342`, `digits-interop` `0.94713414`, both unchanged.

**Per-statement lines (`CpsProgram::op_lines`, `convert_program`'s 13 `LetPrim` sites) and the e-graph/derivative provenance foundation (`FoldData::line` on `ConstantFold`, `Forward::add_from_letprim`) from earlier in this same session are unaffected by the above and still exactly as good/limited as before**: real per-statement lines for anything that stays as CPS-emitted straight-line code (`train_and_evaluate`'s own driver loop: `218, 223, 231, 233, 243, 244, 247`), a coarse "whole rewritten segment / whole synthesized derivative → one line" floor for anything that goes through the e-graph or `synthesize_derivatives`, and `FoldData::line`'s own precise per-e-class line **still not wired through `rebuild`/`rebuild_segment`'s extraction** — the one piece of the per-*statement* (not per-*function*) story still open. This is now a smaller gap than it looked: even the coarse floor is a real, correctly-scoped `DISubprogram`'d location (net_grad's backward now genuinely attributes to `net_grad`'s own line via a real inline site, not just "somewhere in train_and_evaluate").

DI emission is unconditional (adds real weight to every `kernel.o` now — every function gets a subprogram, not just the survivors) — fine while profiling, worth a `--debug-symbols` gate later. The `--emit-object` CLI path still prints the pre-existing, unrelated `NYI: non-trivial layout map` diagnostic (confirmed non-fatal there both before and after everything above; the DISubprogram text now embedded in that diagnostic's own location dump is what made it readable enough to spot the "inlined at" chain in the first place).

## Push destination-passing detection down into CPS/the e-graph instead of an MLIR-level peephole — raised, real architectural tradeoff identified, not started

Raised directly by the user: `dps_rewrite.rs` is a post-hoc MLIR peephole; could the "does this computation's result flow straight into a struct-field store, in place" pattern be recognized earlier, at CPS/e-graph level, where the project's own extensibility discipline (algebra + `mlir::` stdlib mechanism) more naturally lives?

**The real constraint, checked directly, not assumed**: `egraph.rs::Forward::walk` treats `PrimOp::FieldStore` (alongside `Store`/`Extern`) as a hard stop — a real mutation effect, never looked at, the walk returns the remaining expression unchanged from that point on. This is by design (`hld.md`'s own v1 trust model: the e-graph only ever sees the pure world). So today, literally nothing in the e-graph ever observes a field-store site at all, let alone rewrites it.

**Why the MLIR-level placement isn't just historical accident, either**: `dps_rewrite` runs *after* `--inline`/`--linalg-fuse-elementwise-ops` have already fused what were several separate CPS-level calls (e.g. `matmul` then a bias `add`) into *one* `linalg.generic`. That fused shape — not the individual CPS-level calls — is what it redirects. Moving detection to CPS, before fusion, faces a real fork:
1. Handle only the simple case (one algebra call, single-use, stored directly) — decidable locally in CPS via the same "single consumer, no other use" analysis `rc_opt.rs` already does for retain/release pairs. Real, but misses exactly the case that measured `~8.4×` in this file's own `dps_rewrite` entry (a *fused* multi-op chain), since CPS-level single-call detection can't see what MLIR's own fusion pass will later combine.
2. Duplicate `--linalg-fuse-elementwise-ops`'s own fusion-eligibility reasoning at the CPS level to predict the fused shape ahead of time — real duplication of logic between two pipeline stages, a maintenance risk each time MLIR's own fusion rules change.

**A real precedent exists for encoding a genuine precondition into an e-graph rewrite here**: `IndependentZeroApplier` (`egraph.rs`) is already a custom `Applier` with an embedded side-condition (disjointness), not a pure unconditional rewrite — so "conditional rewrite, checked before firing" is an established, working pattern in this codebase's `egg` integration, not something to invent from scratch if this is pursued.

**Not started. The most promising direction, not yet validated**: rather than detecting fusion ahead of time, teach `Forward::walk` to cross a `FieldStore` when (and only when) the stored value has no other use — representing it in the e-graph language as a pure node parameterized by its own destination, rather than as an opaque effect. This reframes the problem as "when is a store observationally a pure construction" rather than "predict what MLIR will later fuse," and would need real design work (how a destination-parameterized node composes with the rest of `CleaveLang`, whether it still lets the fused-chain case fall out for free once downstream MLIR fusion runs on the resulting shape) before any code — a genuine next design session, not a quick follow-on.

## Two debug-inspection improvements, raised directly by the user while reading a real disassembly: a finer backfill floor for MLIR-synthesized ops, and a `CLEAVE_NO_INLINE` diagnostic gate to keep function boundaries intact

**1. `backfill_unknown_locs`/`backfill_all_unknown_locs` (`pipeline.rs`) now propagate the *nearest real location seen so far in program order*, not a single fixed fallback (the enclosing function's own declaration line) for every synthesized op in the whole body.** Found while the user was reading a real `mnist-interop` disassembly and unable to attribute large, unlabeled `mov mem, zmmN` sequences (plausibly a tiled matmul accumulator's own `linalg.fill` zero-seed) to anything — every MLIR-lowering-synthesized op (tiled loop scaffolding, vectorized epilogues — anything with no cleave-source counterpart to inherit a location from at all) used to collapse onto the *exact same* single line as literally everything else in the function, since the old backfill just stamped the function's own root location everywhere. Now each synthesized run inherits whichever real, cleave-emitted op it was generated closest to, until the next real one is seen — still a floor, not genuine provenance (structurally impossible for an op that never existed in cleave source), but one that separates "this sits near the matmul" from "this sits near the bias-add" instead of flattening a whole function to one address range.

**2. `CLEAVE_NO_INLINE=1`** (`pipeline.rs::lower_to_llvm`, same opt-out convention as `CLEAVE_NO_OPENMP`/`CLEAVE_NO_DPS`/`CLEAVE_NO_AFFINE_STRUCTS`) skips only `--inline` itself — `--convert-elementwise-to-linalg`/`--linalg-fuse-elementwise-ops` still run (harmless without cross-function inlining, nothing left for them to fuse), and every later stage degrades safely to its own always-correct, un-rewritten path. Diagnostic only, explicitly not for a perf build — it reopens exactly the double-scratch-buffer cost the inline+fuse comment above it measured and fixed. **Verified directly**: `CLEAVE_NO_INLINE=1`, forced-clean rebuild of `mnist-interop` — `llvm-nm` on the real `kernel.o` now shows `MatMul::matmul<...>` (every shape) as real, standalone, globally-defined symbols with their own `..omp_par.N` regions, instead of inlined away entirely; `mnist-interop` still runs correctly end to end, `0.9342`, no crash (predictably much slower — no fusion, no `dps_rewrite` candidates, exactly as expected for a diagnostic-only build).

**Verified**: full `cargo test -p cleave --release --no-fail-fast` green (one `fibonacci_example_runs_cleanly` failure on the first full-workspace run was a transient parallel-test-runner flake — passed cleanly both in isolation and on a clean re-run of the whole `examples` binary, not a real regression). Both examples re-run end to end with the default (inlining on) build after: `digits-interop` `0.94713414`, `mnist-interop` `0.9342`, unchanged.

## The `~6x` per-FLOP gap between long-K and short-K matmuls, mechanically confirmed by side-by-side disassembly — the fix direction was already identified (`llvm-mca` entry, above) but never implemented

Direct follow-on to the roofline entry above. Disassembled a matched pair, both real, separately-named symbols today thanks to `CLEAVE_NO_INLINE=1`: `MatMulTransposeA::matmul_transpose_a<Tensor<f32,32,784>,Tensor<f32,32,512>,Tensor<f32,784,512>>` (`dW1`, K=32 reduction, ~89-113 GFlop/s) against `MatMul::matmul<Tensor<f32,32,784>,Tensor<f32,784,512>,Tensor<f32,32,512>>` (L1's own forward matmul, K=784 reduction, ~14.7 GFlop/s) — identical total FLOPs (25.7M each), ~6x apart in wall time.

**Both loops use the identical narrow accumulator shape**: a serial chain rotating through only 3-4 independent `zmm` registers (`zmm1 = zmm1*mem+zmm0`, `zmm0 = zmm0*mem+zmm1`, `zmm2 = zmm2*mem+zmm1`, ... — each FMA's own output feeds the next FMA's own input, so the CPU cannot start FMA N+1 until FMA N's own latency has fully retired). This is exactly the pre-existing, already-documented `36.7%` accumulator-latency-chain finding (`vfmadd132ps` rotating through `zmm0-zmm3`).

**The mechanical difference, found by reading the actual loop structure, not assumed**: `dW1`'s own K=32 reduction fits *entirely* inside one unrolled, straight-line block — the latency stall is paid exactly **once**. L1-forward's own K=784 reduction wraps the *identical*-shaped 3-4-register chain inside a **real loop with a back-edge** (`jmp 0xc5e0`, `cmp r13, 0x2ff` — iterating roughly 24 times to cover 784 in chunks), paying the same per-chunk latency stall **~24 times**, because the accumulator set never widens with `K` — more reduction depth only ever means more trips through the same narrow, latency-bound chain, never more independent work to overlap it with.

**The fix direction is not new — it was already named, precisely, and never implemented**: this file's own `llvm-mca` entry (a real, separate analysis session, already in this backlog) concluded exactly this mechanism (`Data Dependencies 75.57%`, only 4 independent registers rotating) and named the fix: *"widen the reduction's own independent-accumulator count in `cleave/mlir/matmul_vectorize.transform.mlir`'s own schedule (more parallel partial sums per `k`-tile, combined only at the end) — 'unroll and jam' the reduction — to give the CPU more independent work to overlap while any one FMA's own latency is still in flight"*. The wider-`M`-tile fix that *did* land (`doc/backlog-done.md`, ~36% single-thread win) widened the *output* tile, which helps register/cache reuse across `M`/`N` — it never widened the `K`-reduction's own accumulator set, which is the specific, still-open gap this entry re-confirms mechanically, on the real kernel, with a real, GFlop/s-quantified before/after pair (`dW1` vs `L1-forward`) rather than a profiler percentage alone.

**Not attempted here** — real, scoped, but the transform-dialect schedule has a documented history of subtle regressions when touched (`doc/backlog-done.md`'s own pad-retry/stack-overflow story) — needs the same discipline: one isolated probe first, then the real kernel, full test suite, accuracy unchanged, before trusting any wall-clock win.

## Bug 4 — the flagged `collect_affine_carried_params` gap was real: a value threaded through an identity-shaped real call on every loop iteration hit a genuine mutual-dependency deadlock, not just a hypothetical risk

Picked up directly on the user's own request ("vas y pour carried params") rather than waiting for a real crash to surface it, following the exact same discipline as Bugs 1-3: reproduce the suspected gap in isolation first, via a dedicated debug probe, before writing any fix.

**Confirmed real, by direct construction, not assumed**: a minimal program mirroring `Display::display<Complex<T>>`'s own shape but inside a loop --

```
fn thread_through(cond: bool, a: Boxed) -> Boxed {
    if cond { opaque_sink(touch1(a)); a } else { opaque_sink(touch2(a)); a }
}
fn main() -> i32 {
    let mut b = Boxed(v: 0, tag: [0]);
    for i in 0..2000 { b = thread_through(i < 1000, b); };
    b.v
}
```

-- a debug probe on `affine_struct_vars`'s own output showed `entry affine = true`, `carried affine = false`, even though `identity_summary.returns_unchanged("thread_through", 1)` was already correctly `Some(true)`.

**Root cause: a genuine mutual dependency the existing single-pass-per-iteration fixed point can never seed on its own.** The loop's own back-edge argument (`b2`, `thread_through`'s own resumption parameter) can only be marked affine by `collect_affine_resumption_params` once the carried parameter itself (`thread_through`'s own argument at that call site) is *already* affine. The carried parameter, under the original rule, can only be marked affine once *every* one of its own sources -- including that same back-edge argument -- is *already* affine. Neither side has anywhere to seed from independently within the loop body itself; the true anchor (the entry argument, constructed *outside* the loop) is only ever checked by `collect_affine_carried_params`, which requires *every* source affine, back-edge included -- so the fixed point converges after one iteration having made no progress on this cycle at all, despite the real answer being sound.

**Fixed, in `alias_analysis.rs`, with a new mechanism structurally mirroring `tail_returns_var`/`IdentitySummary` exactly, retargeted**: `carried_param_flows_to_own_backedge` -- "does the carried parameter, traced forward through the loop's own body (through local join-point hops the same way, and through an identity-shaped real call's own resumption via `identity_summary`, chasing an unrelated intervening call's own trailing continuation exactly like `tail_returns_var` already had to), ever reach a tail-call back to the loop itself with the *same* value at the *same* position?" If so, the carried parameter is *structurally* guaranteed to denote the same allocation every iteration -- its own back-edge arguments need no independent proof of their own at all, and its affine-ness reduces to whether *any* recorded call site (in practice, the one real anchor: the entry argument) is already affine. `collect_affine_carried_params` now checks this as an additional, alternative path alongside the original "every source independently affine" rule (kept, since it's still what the *fresh-construction-per-iteration* shape, `b = bump(b)`, needs).

**One real bug found while building this, caught immediately by the same debug probe rather than assumed correct on the first attempt**: the first version, mirroring `tail_returns_var`'s *original* (single-hop) shape, didn't chase an unrelated intervening real call's own continuation -- every `for`/`while` loop's own bound check (`Ord::lt<i32>`) runs *before* the loop's real body on every iteration, and the carried value isn't one of *that* call's own arguments at all. Fixed identically to how `tail_returns_var` itself needed fixing for the same reason (this module's own established pattern, applied a second time to a new function rather than re-discovered from scratch).

**Verified**: a new test, `a_carried_parameter_threaded_through_an_identity_shaped_real_call_each_iteration_is_affine_too` (`cleave/tests/alias_analysis.rs`), the exact minimized repro from the debug probe, both facts asserted directly. `examples/complex.cleave --run`/`examples/convex_hull.cleave --run`: 0/100 each, re-confirmed unaffected. Full `cargo test --release --no-fail-fast` (`RUST_MIN_STACK=67108864` set, per the entry above) green -- 35 test binaries, run cleanly with no concurrent interference.

## `--target-cpu native` combined with an explicit `--target-features` delta is fatal for the AOT path (`--emit-object`/`--emit-exe`/`cleave-build`), even though each works fine alone — the JIT path's own `"native"` substitution never runs here

Found live, not hypothetical: `examples/mnist-interop/build.rs` already used `.target_cpu("native")` alone successfully (every build/run this session). Adding `.target_features("-avx512f")` alongside it — an attempt to get an AVX2+FMA build for comparison, the same technique `doc/user_guide.md`'s own `--target-features` section documents working for `--run` — instead aborted the AOT build outright: `'native' is not a recognized processor for this target (ignoring processor)` (repeated once per function) followed by a fatal `LLVM ERROR: 64-bit code requested on a subtarget that doesn't support it!`.

**Root cause, confirmed by reading, not yet fixed**: `"native"` is only ever resolved to a real CPU name (`llvm::sys::getHostCPUName()`/`getHostCPUFeatures()`) inside `cleave-mlir-shim/cpp/shim.cpp`'s own `cleaveExecutionEngineCreateWithTarget` — the JIT `ExecutionEngine`'s own target-machine construction, nothing else. `pipeline.rs::stamp_target_cpu` (the AOT path every `--emit-object`/`--emit-exe`/`cleave-build` build goes through instead) never calls into that shim at all — it just parses `options.target_cpu`/`target_features` into literal MLIR attributes (`target_cpu = "native"`, `#llvm.target_features<["-avx512f"]>`) and stamps them directly onto every `llvm.func`, verbatim, with no substitution and no "resolve host defaults, then layer the delta on top" step at all (that layering, per `doc/user_guide.md`'s own `--target-cpu`/`--target-features` section, is a real, separate step `resolve_codegen_options`/the JIT path performs *before* ever reaching a `TargetMachine` — `stamp_target_cpu` has no equivalent). LLVM's own AOT backend, given the literal string `"native"` as a function's `target-cpu` attribute, doesn't recognize it (real CPU codenames only), falls back to its own generic default, and then applies the *bare, unresolved* feature delta (`-avx512f`, no baseline features layered under it) on top of that fallback — producing an internally inconsistent subtarget the instruction selector can't handle for 64-bit code at all. `pipeline.rs`'s own doc comment on `target_cpu` (`Some("native".into())` "is also real and explicitly equivalent to `None` here") is accurate for the JIT path only — the doc comment doesn't distinguish, and should.

**The pre-existing `doc/user_guide.md` example this contradicts**: `--run --target-features -avx512f,-fma` (no explicit `--target-cpu`, so it's `None`, not `"native"`) — never hits this, since there's no `target_cpu` attribute stamped at all in that case, only the features delta; whether an *unset* `target_cpu` plus an explicit `target_features` is *also* broken on the AOT path specifically (as opposed to only the `Some("native")` combination actually hit here) hasn't been checked.

**Worked around, not fixed**: dropped `--target-features` entirely and used `--target-cpu x86-64-v3` alone instead (a real, LLVM-recognized CPU-level string in both the JIT and AOT paths, no substitution needed at all) — confirmed working end to end on `examples/mnist-interop`: accuracy unchanged (`0.9342`), `23.5s` vs. the AVX-512 build's own `~14s`, directionally consistent with the expected ~2x hit from halving vector width (512-bit `zmm` to 256-bit `ymm`).

**Not urgent, flagged as important**: `stamp_target_cpu` needs its own `"native"`-substitution step (or to reject the combination with a clean, located error instead of a raw LLVM abort) so the AOT path's `--target-cpu native` genuinely matches its own doc comment's claim under every combination, not just alone.

**Squarely the worst of this project's own three error tiers, not the middle one** — a real, useful distinction raised earlier this session for a different bug (`cps.rs`'s "unbound variable" panic, the entry above): a "clean" error names the actual problem at the actual location; a "confusing but located" one at least points somewhere real; this is the third, worst kind — a raw `LLVM ERROR:`/`abort()` with no cleave-level diagnostic, no span, no indication which `--target-cpu`/`--target-features` combination caused it, surfacing only as an opaque native crash on whatever `.compile("kernel")` call happened to trigger it. Any invalid `target_cpu`/`target_features` combination reaching this point deserves at minimum the middle tier (a located, if imperfect, cleave-level error) — reaching LLVM's own fatal abort path at all is the bug, independent of whatever the eventual real fix for the substitution gap itself turns out to be.

## `ConstValue` has no `Float` variant at all — a named `const`/`define` can't hold a float, at all, not even a simple physical constant

Found in conversation, not by testing a failure directly, but confirmed by reading: `ConstValue` (`infer.rs`) is exactly `Int(u64) | Bool(bool)` — no third variant. `const PI: f64 = 3.14159;` cannot fold today: `registry.rs::eval_const_expr` and `infer.rs::const_value_from_expr`'s own `ExprKind::NumberLit` arms both parse the literal's text via `.parse::<u64>()` only, so a float-shaped literal (anything with a `.`) never even reaches a `ConstValue` at all, let alone one of the right kind — it fails the same permissive-by-omission way any other unrecognized shape does (silently unresolved, `pipeline.rs::check_const_decl_errors` reports it at the point of use, not with a message pointing at "float unsupported" specifically).

**Squarely the right next investment in this exact area, not scope creep**: this whole `const`/`define` feature (this session's own work, `doc/hld.md`'s own "`const`/`define`: no `constexpr` sublanguage" section) is explicitly aimed at a scientist/HPC author, for whom a named float constant (a physical constant, a tolerance, a learning rate) is the single most obvious use case — arguably more obviously wanted than any integer/bool one already supported. Widening `const_eval.rs`'s own `ConstValue`/`eval_binop`/`eval_unop` to a `Float(f64)` variant (parsing `NumberLit`'s own text as a float when it contains a `.`, mirroring how ordinary literal defaulting already distinguishes `i32` vs `f32` shapes elsewhere in this codebase) is the natural, narrow, additive next step — same "grow what's foldable, don't invent a second mechanism" posture the whole feature was designed around, not a new design question.

**Deliberately not conflated with general constant folding for ordinary `let` bindings** — that's a separate, already-existing mechanism (`egraph.rs::Analysis::make`'s own `const_int` computation, `doc/hld.md`'s "Constant and copy propagation" section), running later in the pipeline (post-CPS, on already-typed code) for a different reason (an e-graph congruence-closure consequence, not a value `const`/`define` needs resolved before type inference can even run). Widening `const_eval.rs` doesn't touch that mechanism at all, and vice versa — the two stay independent on purpose.

## Interval/value-range analysis (a fixpoint-iterated abstract-interpretation lattice, not literal fixed-point *arithmetic*) — raised in conversation as a plausible future direction, not designed or scoped yet

The idea, as raised: track each variable's own possible value *range* (an interval, not just "known constant or not"), converged via the standard fixpoint-iteration abstract-interpretation technique (widening/narrowing across a loop's own back-edge, Cousot-style) rather than exact constant folding alone. The concrete motivating use case named directly: a `for` loop's own **trip count** becoming staticaly derivable when its bounds are provably within a known range — unlocking unroll/unroll-jam decisions, bounds-check elimination, and potentially const-generic-shaped dispatch (`doc/backlog.md`'s own already-closed "const generic compared via an operator" entry, and the still-open OpenBLAS-motivated size-threshold dispatch it unblocked) for cases that are range-bounded but not literally constant.

**A strict generalization of exact-value folding, not a separate, parallel mechanism sitting next to it** — the closing insight from the conversation that raised this: a proven interval `[x, x]` (lower bound equals upper bound) *is* a proven constant, exactly the degenerate, single-point case of the same lattice. `const_eval.rs`/`Analysis::make`'s exact folding only ever answers "is this value exactly known;" a real interval lattice answers the strictly more general "what range could this value fall in," with exact-known-constant as one specific, already-converged point in that range — so a full interval-analysis pass, if ever built, would properly *subsume* today's exact folding rather than duplicate it, the same way any abstract-interpretation lattice's most-precise element degenerates to exact evaluation. A real interval lattice needs its own join/widen operators and its own fixpoint loop over the CPS graph's own loop-carried arguments (`doc/hld.md`'s own "Memory management: a region/stack discipline derived from CPS" section already establishes that a loop's own carried state is syntactically explicit in this pipeline's CPS form, `carried_types`/`params` on `CFunDef` — the same structural fact that made copy-propagation's def-use question local rather than needing classical dominance-frontier computation likely applies here too, worth checking directly before assuming a full classical dataflow pass is needed). Not designed, not scoped, no repro/motivating failure yet — purely a direction flagged for later evaluation once the more concrete gaps above are closed.

## A statically-zero-trip-count loop's body is *not* eliminated the way a compile-time-constant `if`/`and`/`or` branch already is — "works" at `--opt-level 2` only by incidental LLVM backend DCE, not by anything cleave's own pipeline proves, and silently stops "working" at `--opt-level 0`

Raised in conversation, directly after the `if`/`and`/`or` dead-branch-elimination work above closed: does the identical guarantee hold for `for i in 0..0 { extern_call(i); }`, a loop whose own trip count is staticaly zero? Tested directly, not assumed — it does not, and the difference is real, not cosmetic.

**`if`/`and`/`or` eliminates the dead branch structurally, before LLVM's own backend ever runs** — confirmed by the entry above: `--dump-mlir-lowered` (printed straight after cleave's own CPS-optimize/MLIR-canonicalize stages, before any LLVM backend codegen) already shows zero trace of the untaken branch. **A zero-trip-count loop does not get this treatment at all** — `--dump-mlir-lowered` on `for i in 0..0 { acc = never_called(i); }` still shows `never_called` both declared and called, inside a real loop construct, completely unchanged from a loop whose bound is a genuine runtime value. The loop is never recognized as trivially dead at the CPS/MLIR level cleave itself controls.

**What actually removes it from the final object is a different, unrelated mechanism, and a much weaker guarantee**: at the default `--opt-level 2`, `never_called` is genuinely absent from the emitted object's own symbol table (confirmed via `llvm-objdump -t`) — but this is LLVM's own backend dead-code elimination, running during real machine-code generation, well after cleave's own IR has already committed to emitting the call. Proof this is incidental, not structural: **the identical program built with `--opt-level 0` has `never_called` right back in the symbol table**, unresolved. Anyone building for debugging (`--opt-level 0`, `--no-inline`, the exact combination this project's own `--no-inline` doc comment recommends for reading a clean disassembly) would need a symbol the source-level logic proves is never actually called — and a genuinely side-effecting `extern fn` (not provably pure) might not even be safe for LLVM's own backend to drop at any opt level, meaning this "works" only for the specific, effect-free probe tested here, not as a general guarantee at all.

**Not fixed, not scoped in depth** — the natural extension of the already-closed `if`/`and`/`or` work above (same underlying idea: a provably-unreachable branch/body should never require a symbol to exist), but genuinely more work: `if`'s condition is a single boolean value already reaching the const-fold machinery for free; a loop's own trip count first needs deriving from its bounds (`end - start` provably `<= 0` for the simplest case, `0..0` here) before the identical "this is unreachable, don't even lower it" treatment could apply. Plausibly connects to the interval/value-range analysis entry immediately above (a trip count is exactly the kind of fact that lattice would derive), but a narrower, `<=0`-specific special case might be enough for the immediate BLAS-adjacent motivation without waiting on the full lattice. No design attempted here.

## `matmul_transpose_b` (`A · Bᵀ`) is plausibly the one native-lowering variant a generic MLIR vectorizer can already handle well without packing — not measured, raised in conversation directly after the axiom work above landed

Raised in conversation: in row-major layout, `C[i,j] = Σ_k A[i,k]·B[j,k]` (exactly `matmul_transpose_b`'s own shape) is the one of the three matmul forms where *both* operands are read as a contiguous row along the reduction axis `k` — a plain vectorizable dot-product reduction, zero stride on either side. Contrast the other two: plain `matmul` (`NN`) has `B[k,j]` strided by `N` along the reduction axis (the classic reason BLAS packs `B` before its own hot loop); `matmul_transpose_a` (`TN`) has both operands contiguous per fixed `k`, but that's an outer-product/rank-1-update accumulation into `C`, a genuinely different loop structure, not the same loop merely relabeled.

**Why this might matter specifically for cleave, not BLAS in general**: cleave's native MLIR lowering has no packing/blocking microkernel of its own (unlike a real BLAS) — it leans on MLIR's own `linalg` vectorizer operating fairly directly on the `indexing_maps` the lowering emits. `matmul_transpose_b`'s zero-copy `indexing_maps` trick for the transposed operand is plausibly the one case among the three where that map stays affine and contiguous along the vectorized axis, i.e. the one case a generic vectorizer has a real shot at recognizing and turning into a proper vector reduction without any hand-written lowering. This also directly recoups the "NYI: non-trivial layout map" diagnostic already tracked elsewhere in this backlog (the struct-allocation-strategy entry's own native-lowering point of vigilance, and §4/§9 of `doc/plan-blas-native.md`) — same non-trivial-layout-map mechanism, this time asked from the angle of "does it actually vectorize well" rather than "does it crash."

**Not measured, not designed, nothing built** — purely a hypothesis worth checking empirically before doing anything: dump `--dump-mlir-lowered` for all three variants on the same shapes and read whether the vectorizer actually emits a vectorized reduction for `matmul_transpose_b` specifically (vs. a scalar loop for the other two, or for all three alike). If MLIR already handles it well, there's nothing to build. If not, this is a real candidate for a dedicated native lowering path for `MatMulTransposeB` specifically, distinct from `NN`/`TN` — but that's a follow-up decision, not this entry's own conclusion.

**Deliberately not touched**: four other test-harness files (`affine_pool_alloc.rs`, `alias_analysis.rs`, `array_release_cascade.rs`, `refcount.rs`) still run their own local copy of the old eliminate-before-optimize sequence — harmless for what they each test today (none of them exercise a never-directly-called cross-algebra reference), but now genuinely inconsistent with the real, fixed pipeline. Low-priority cleanup, not blocking anything.

## Checkpoints: make them a general building block for long computations, not an ML tool — to revisit once the language core is stable

The `stdlib/checkpoint` module (step 0 of `doc/plan-nanolm.md`) serves any long-running simulation: fault tolerance, resuming a run, splitting a computation into segments. Yet today it does `use linalg; use optim;` and holds the `AdamState` and `Trainable` impls itself, so a fluid simulation using it drags the optimizer in for nothing. Deliberately postponed: cleaning up the stdlib is easier once the language has settled.

What to do, by priority:
- **Invert the dependency.** `checkpoint` keeps only the algebra, scalars, tensors and tuples. The `Checkpoint<AdamState<…>>` and `Checkpoint<M: Trainable>` impls move to `optim`, next to their types: an impl lives with its type, not with the module that serializes it.
- **A file checksum.** The atomic rename protects against a stop mid-write, not against a file damaged afterwards (disk, network copy). Add a CRC32 or xxhash at the end of the file, checked by `restore`; it's a format change, so a version 2 of `CLVCKPT`.
- **Rotation.** Keep the last k checkpoints (`run.ckpt.1`, `.2`…): if the latest is corrupt, resume from the previous one.
- **One-line resume.** A `restore_or(path, init)` that resumes if the file exists and starts from `init` otherwise. Makes a program resumable without plumbing.
- **Later, asynchronous writes.** Copy the state, then write it on another thread while the computation continues. Only worth it when writing really weighs (a big simulation), not for nanoLM.
- **Generic file I/O in the runtime.** Once cleave has real strings: the format and its checks would be written in cleave on top of `open`/`write bytes`/`read bytes`/`rename`/`close` primitives, instead of living in `cleave-rt/src/checkpoint.rs`.

## nanoLM's kernel compile time: from ~3 minutes to ~30 s — measured, mostly fixed

Measured on `examples/nanolm/src/kernel.cleave` with the transformer (4 blocks, d = 128). Front end down to optimized CPS: ~5 s. CPS to MLIR: 1.5 s, 22 MB of MLIR. MLIR passes: ~23 s, 69 MB of LLVM dialect. LLVM optimization and code generation: ~90 s, plus ~24 s writing the object. So the cost was LLVM digesting a huge IR.

Why the IR was huge: everything inlined into the exported functions — `train_gpt` was one function of 704k lines out of 865k. Each heavy operation lowers to thousands of lines (attention forward ~5.7k, LayerNorm ~4.8k, a matmul ~1.7k: the matmul schedule's tiling, unrolled vector contractions, masks, memref descriptor plumbing), and full inlining copied them at every call site (LayerNorm 8 times forward plus its gradients, attention 4 + 12, …). `opt -O2 -time-passes` on that IR: `IndVarSimplify` 47%, then full unrolling, `LoopRotate`, `LICM`, all superlinear in function size; `llc` another 24 s, half of it register allocation.

What was done, and what each change did alone:
- `#[no_inline]` on `CausalAttention`'s and `LayerNorm`'s methods (`stdlib/nn`): each instance compiled once; IR down to 537k lines. **Alone it made things worse** (638 s): smaller functions let LLVM's `LoopUnrollPass` unroll far more (96 s of 174 in `opt`).
- `CodegenOptions::llvm_loop_unroll` (CLI `--no-llvm-unroll`, `cleave-build` `.llvm_loop_unroll(false)`), turning off LLVM's own loop unrolling in the execution engine's pipeline (`cleave-mlir-shim`'s `makeTransformer`; cleave's loops arrive already tiled and unrolled where it pays). **Alone, no gain** on the fully inlined kernel (168 s): there `IndVarSimplify` on the giant function dominates, not unrolling.
- **Both together: 80 s** through the CLI, `cargo build -p nanolm` 1 min 50; a training step's speed unchanged (50 steps: 42.7 s against 41.0 s for the old configuration, within noise; identical validation loss). `nanolm/build.rs` uses `LLVM_LOOP_UNROLL = false`; the default stays `true` until measured on MNIST and the rest.
- Also tried: `--no-inline` everywhere (12 s to compile, but 10× slower to run: fusion and vectorization need the small operations inlined), and rewriting Adam's leaf step as one loop (the IR grew: the ten tensor operations were already being fused well) — both reverted.

Still large; the next leads: why each operation lowers to thousands of lines (vectorized loop nests and descriptor `insertvalue`/`extractvalue` chains dominate the op histogram); `no_inline` for more of the heavy, repeated operations (the gradient bodies, Adam per leaf shape); and the 24 s of writing the object (maybe a second code generation).

Then the bigger model (6 blocks, d = 256), measured with `--no-openmp --no-llvm-unroll --target-cpu native --emit-object` and the phases timed separately (front end + CPS → MLIR ~3 s, MLIR passes, then the execution engine's optimization and the object's code generation, the bulk):
- **148 s** to start with (code generation 82 s, LLVM optimization 31 s), 385k lines of LLVM dialect: `train_gpt` 132k, `generate` 114k, the Adam step's `Trainable` glue 89k.
- `#[no_inline]` on Adam's leaf `step` (`stdlib/optim`): 112 s.
- The BLAS tier of the matmuls out of line (`stdlib/linalg/matrix.cleave`, `blas_*`): the `sgemm` call and its glue (destination allocation, descriptors) once per shape rather than at each of ~200 call sites; the `linalg` path stays inlined. 95 s. Putting *every* matmul out of line reached 63 s but made MNIST 22% slower (lost fusion), rejected.
- **The e-graph pass ignored `#[no_inline]`**: `optimize_program` walks a pure plain `fn` (a "transparent chain") as if inlined, so `block` disappeared into `gpt_logits` and `generate` before MLIR ever saw the attribute, and the Adam step's glue got the same treatment. It now keeps a `#[no_inline]` callee one opaque call (`Forward::honor_no_inline`), except while synthesizing a gradient, which has to see through a callee with no rule of its own. **36 s** (code generation 14.5 s, LLVM optimization 5 s), peak memory 4.2 → 2.7 GB; the Adam glue 89k → 5.7k lines, `generate` 114k → 55k.
- `#[no_inline]` on the checkpoint leaves (`stdlib/checkpoint`: a tensor's and an Adam state's `save_part`/`restore_part`), copied for each of the 102 leaves wherever the model is saved or restored: IR 232k → 181k lines.
- **`#[no_inline]` was read off the wrong impl**: `cps.rs::collect_units` took it from the impl being walked, while `specializations_of("Optimizer::step")` returns the specializations of *every* impl of the method (the trap `is_extern`/`is_pure` had already fallen into), so the last impl's attribute landed on all of them. Adam's spilled onto `Sgd`'s leaf step (MNIST ~1 s slower once the e-graph started honoring the attribute) and onto the generic `Trainable` glue of the optimizer and the checkpoints. Now carried by each specialization (`monomorphize.rs`, `Specialization::no_inline`); `language_model_ops.rs::no_inline_applies_to_the_declaring_method_only` covers this and the e-graph case. The glue, inlined again, puts nanoLM's IR at 207k lines (`train_gpt` 97k, `generate` 37k); **~30 s** to compile, peak memory 2.1 GB.

Left: `train_gpt`'s gradient, where everything is necessarily inlined (synthesis sees through `block`; a per-function gradient, the backward of `block` as its own function, would let it be compiled once instead of six times), and the refcounting around the model's construction and restore (most of `generate`'s remaining lines are `retain`/`release`, the "retain/release are opaque calls" entry).

## The in-process test harnesses don't run the matmul schedule: a test of a matmul there doesn't cover the real pipeline

Found while fixing matmuls whose column count isn't a multiple of 16: the test written in `cleave/tests/mlir_lower.rs` passed **even without the fix**, while the CLI (`--run`) failed. The harness pipeline in those files (`run_i32` and its relatives) doesn't apply `matmul_vectorize.transform.mlir`, so its matmuls take another lowering path. The regression test now goes through the CLI binary (`language_model_ops.rs`, `CARGO_BIN_EXE_cleave`). To fix: have the harnesses go through `pipeline.rs::lower_to_llvm` with the real options, or at least list which existing tests think they cover the schedule and don't.

## Under `--no-inline`, a matmul inside the BLAS-threshold `if` isn't vectorized

`matmul_transpose_b`'s body is `if P*Q*R > 100000000 { sgemm(...) } else { linalg matmul }`. Inlined, the `if` folds away (the product is a compile-time constant) and the matmul is vectorized like any other. Compiled as its own function (`--no-inline`), the `if` stays, the `linalg.matmul` sits inside an `scf.if`, and the schedule's matcher (`@match_matmul`) only accepts a matmul directly under a `func.func` or an `scf.while`: the matmul is left to scalar loops. Correct, just slow. Found writing `matmuls_with_a_partial_tile_are_vectorized`, which therefore runs inlined. To fix: fold a branch whose condition is a compile-time constant in the specialized function too (`doc/backlog.md` already has the zero-trip-count loop case of the same gap), or let the matcher accept an `scf.if` parent.

## Fewer bits per weight: `bf16` training, codebook quantization for inference (idea, 2026-10-04)

**`bf16` in training.** nanoLM's data-parallel scaling looks bounded by L3 traffic (each task reads every weight, `doc/plan-spawn.md`); `bf16` weights halve it. Zen 5 has AVX-512 BF16 (`vdpbf16ps`: pairs of `bf16` multiplied, accumulated in `f32`), about twice an `f32` FMA's throughput at the same width, and OpenBLAS has `sbgemm` (`BUILD_BFLOAT16`). The usual recipe: matmuls in `bf16` accumulating in `f32`, an `f32` master copy of the weights for the optimizer. In cleave: a `bf16` element type and a `Ring` instance accumulating in `f32`, in the stdlib — nothing in the compiler.

**Codebook (vector) quantization for inference.** Token-by-token generation is bound by memory bandwidth, not compute. Quantizing weights jointly in small blocks against a codebook beats scalar quantization at the same bit budget (packing gain, the reason QAM constellations exist): QuIP# (E8 lattice, 8-weight blocks), AQLM, VPTQ (learned codebooks). At 2-3 bits per weight a nanoLM fits in L2. Training-aware quantization goes through `grad` with a straight-through estimator (`floor` forward, identity backward). In cleave: a `Quantized<Codebook, N>` type behind an algebra, decoding fused into the matmul.

## Spectral layers: a differentiable FFT algebra, long convolutions for nanoLM, a Fourier neural operator for the PINN (idea, 2026-10-04)

Signal processing as a layer vocabulary, the direction the user likes best for a scientific language:

- **Structured matrices** instead of dense ones, O(n log n) instead of O(n²): circulant (FFT, pointwise product, inverse FFT), Fastfood (Hadamard, diagonals, permutations), ACDC (DCT), Butterfly/Kaleidoscope (the FFT's factorization, learned), Monarch (block-diagonal and permutation products, built to be fast on real hardware).
- **Layers that are filters**: FNet (attention replaced by a parameter-free 2D FFT, ~92% of BERT), S4 / Hyena / Mamba (each layer a linear system, a long convolution computed by FFT in O(T log T) where attention is O(T²); S4 parameterizes its kernel in the frequency domain). nanoLM's coherence is bounded by its context: a Hyena-style block against the attention block, same budget, same corpus.
- **Compressed transfer functions**: KAN (a learned activation per edge, splines or truncated Fourier series), SIREN and Fourier features.
- **Fourier neural operator** (Li et al.): the operator learned in Fourier space, high modes truncated — compression by construction, built for PDEs: a second scientific showcase next to cleave-cast's PINN.

The FFT differentiates cleanly (its adjoint is the conjugate inverse FFT), so a `Fourier` algebra over `stdlib/complex` makes all of these differentiable at no extra cost. Known caveat: on GPUs, FFT-structured layers often lose wall-clock to a tuned dense GEMM; on CPU, with our own compiler, it's a fight worth measuring.

## A scientific library beyond ML, up to converged hybrid solvers (direction, 2026-10-04)

ML is one client of the stdlib among many; the target is computational science and engineering. Numerical Recipes is a reference for algorithms, numerical pitfalls and test cases — its code is copyrighted under a restrictive license, never ported line by line.

**The building blocks**, roughly in order: FFT (radix-2 then mixed-radix, real and complex, n-D — also the base of the spectral layers above); direct solvers (LU, Cholesky, QR — our OpenBLAS is built `NOFORTRAN=1`, no LAPACK: blocked versions in cleave over `sgemm`, or a LAPACK build); iterative solvers (CG, BiCGSTAB, GMRES, preconditioners), generic over an operator `A·x` algebra so sparse and matrix-free operators fit; eigenvalues (power iteration, Lanczos, Jacobi); ODEs (RK4, adaptive RK45, symplectic Verlet); optimization (Newton, Brent, BFGS / L-BFGS, gradients from `grad`); quadrature and interpolation (Gauss-Legendre, Romberg, splines); geometry (convex hull, Delaunay).

**Implicit gradients, the piece that sets cleave apart.** Differentiating a converged solver by backpropagating through its iterations costs memory per iteration and differentiates the iteration, not the solution. At a fixed point `x* = F(x*, θ)`, the implicit function theorem gives `dx*/dθ = (I - ∂F/∂x)⁻¹ ∂F/∂θ`: one adjoint linear solve at `x*`, whatever the iteration count (Deep Equilibrium Models; JAX's `custom_root` / jaxopt). In cleave, a `FixedPoint` / `Solve` algebra whose gradient is declared (the mechanism `CrossEntropy` already uses): Newton, CG, coupling iterations and implicit time steps become differentiable by construction, at constant memory.

**Converged hybrid solvers** — the showcase this leads to, stronger than a PINN for engineers: the AI accelerates a real solver and keeps its structure, rather than replacing it (a PINN) or merely being fed better features. The classical loop iterates to tolerance, so the answer keeps its guarantees (residual, conservation, stability); a bad network slows convergence, it doesn't falsify the result. Forms: a learned preconditioner inside a Krylov solver; a learned initial guess (an FNO) for Newton / Jacobian-free Newton-Krylov; learned closures (turbulence, constitutive laws) inside a finite-volume or finite-element scheme, trained solver-in-the-loop (Um et al., 2020) through the implicit gradient; multiphysics coupling (partitioned fixed point with Aitken / Anderson acceleration, or monolithic Newton); adjoint-based design and shape optimization.

Progression: building blocks → `FixedPoint` / `Solve` with implicit gradients → discretizations (finite differences, finite volumes, then finite elements) → hybrids.

## Retain/release cascades are expanded inline at every site, per tensor leaf: nanoLM v2's `train_gpt` reached 98 MB of IR (found 2026-10-04)

A struct's retain or release is a cascade over every tensor leaf (`refcount.rs`'s field granularity,
`mlir_lower.rs::lower_release_cascade`), written out in full at each site, each leaf's call unpacking
its memref descriptor (~3.5 KB of IR per call). With nanoLM v2's eight micro-batch gradients held in
a tuple (correctly released now, see the comprehension entry above), `train_gpt` had 18,000
`cleave_release` and 9,400 `cleave_retain` calls, 98 MB of IR before bufferization, and the MLIR/LLVM
passes went past 10 GB. Fix in progress: one retain and one release function per struct type (Rust's
"drop glue"), called where the cascade used to be written out, so that code grows with the number of
types rather than sites times leaves.

## Light structs cross calls by value, decomposed into their scalars: a 58 KB argument area crashes past the stack's guard page, and the IR swells (found 2026-10-04)

nanoLM v2 segfaulted in its first training step, only with several threads and only at full size. The
Windows event log gave the faulting instruction (`nanolm.exe+0x2411a5`, symbolized with
`llvm-symbolizer` and the binary's PDB to `Optimizer::step` inlined in `train_gpt$tasks`): a store at
`0x27e8(%rsp)` right after `subq $0xe360, %rsp`, the argument area of a call to `Optimizer::step<..Gpt..>`
taking the model, its gradient and the optimizer state *by value*. A light struct (every field a
tensor descriptor, inline) is a first-class LLVM aggregate, which LLVM expands into its scalars at a
call: tens of thousands of bytes stored downward from the new `%rsp` with no probe, skipping Windows'
guard page, an access violation (not a stack overflow, so no handler runs on the faulting thread).
Latent until a call this large wasn't inlined. The same representation fills the IR with
`extractvalue`/`insertvalue` per descriptor wherever such a value is copied, a large share of
nanoLM v2's ~150 s of MLIR/LLVM passes. To fix at the source: pass a light struct above a size
threshold by pointer to a copy (the Windows x64 C ABI's rule for large aggregates), parameters,
calls and returns alike, `export fn`s and the Rust boundary included. Reproduction to write first: a
struct of tens of KB passed to a non-inlined function.

**Parameters fixed, same day.** A light struct of 128 bytes or more (`mlir_lower.rs::by_pointer`,
`BY_POINTER_MIN_BYTES`) is passed to an internal function as a pointer to a copy in the caller's
entry block (`call_arguments`, `entry_alloca`), loaded once in the callee; `main` and `export fn`s
keep their signature. Calls, loop conditions, leaf glue and spawned calls alike; a spawned call's task
captures only the pointer, the slot living until the caller's `sync`. nanoLM v2 trains (no crash), its
LLVM time 162 s -> 98 s. The crash itself isn't reproducible on demand in a small program (it depends
on the order of the stores and on how much stack the thread had committed), so the test checks the
ABI: `language_model_ops.rs::a_large_light_struct_crosses_a_call_by_pointer` (fails without it).
Still by value: **returned** light structs (LLVM demotes a large return to a hidden pointer, so no
crash, but the aggregate is still built with `insertvalue` per scalar), and light structs inside the
caller's own body.

**The slots themselves then overflowed the stack (2026-10-05).** The night run (`gpt 0 200 100`)
died with `STATUS_STACK_OVERFLOW` on the main thread. One slot per call site, alive for the whole
frame, and MLIR's inliner carrying a callee's slots into the caller wherever the call was (inside a
loop body, a dynamic allocation, bounded only by the loop's `stacksave`/`stackrestore`):
`train_gpt$tasks` held 272 KB of slots in its entry block and 1.6 MB elsewhere. Fixed once loops are
blocks (`cleave_mlir_shim::hoist_arg_slots`, end of `pipeline.rs::lower_to_llvm`): every slot goes
back to its function's entry block, and an ordinary call's (not a spawned one's) gets
`lifetime.start`/`lifetime.end` around its uses, so that LLVM's stack coloring shares storage
between slots. Moving the slots without the lifetimes made it worse (a 1.37 MB static frame).
`train_gpt$tasks`' frame is now 570 KB, static; a 100-step round trains (1429 ms/step, 10.9 GB).
Test: `language_model_ops.rs::argument_slots_sit_in_the_entry_block_with_bounded_lifetimes` (an IR
check; a runtime overflow in a small program depends on LLVM's argument promotion and memcpy
forwarding, which erase the slots of a callee reading few fields or of identical arguments).
Remaining: 570 KB is still half the main thread's stack. A slot whose uses span several blocks
gets no lifetime and keeps the whole frame, as does a spawned call's; and the copy itself is
redundant whenever the argument already lives in memory. Passing the caller's own storage, or
giving the main thread a larger stack (`/STACK` at link), would remove the margin question.

## Debug info attributes inlined stdlib code to the program's file
`llvm-symbolizer` on nanoLM v2 placed `Optimizer::step` at `kernel.cleave:386`, a line the kernel
doesn't have: the inlined function's `DISubprogram` names the right function but the program's file
(`stdlib/optim/optim.cleave` is the right one). Line numbers are then meaningless in a crash or a
profile. Found 2026-10-04.

## Two intermittent anomalies in `leaks.rs`, not reproduced (2026-10-04)

Once, `clipping_leaves_no_allocation_behind` failed in a full `leaks.rs` run (bytes left per clip
above `NOISE`); once, a run of `leaks.rs` produced no `test result` line at all (the process ended
early). Neither came back in 12 further runs. Both happened the day the bufferization (`allow-return-
allocs-from-loops`), refcounting (per-type glue functions) and ABI (large light structs by pointer)
changed; worth a loop of runs (`for i in $(seq 50)`) after the next change touching those, and a
capture of the failing output when it happens.
