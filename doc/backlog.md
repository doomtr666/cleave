# Backlog

*Ordered — top to bottom is the order we work through it. Not a wishlist: each entry is a real, confirmed gap (found by testing or by direct inspection), not a guess about what might be missing.*

Completed items live in [backlog-done.md](backlog-done.md).

---

## The pool keeps every freed block, whatever its size class: memory held well above the live peak

`cleave_release` parks every freed block in its size class's free list (`pool_push`), for the next
allocation of that class. A program whose phases use different sizes (a training step: activations in
the forward and backward passes, weight-sized buffers in the optimizer) holds the peak of each class
at once.

**A cap exists, off by default** (2026-10-10): `CLEAVE_POOL_PARK=<fraction>` parks a block too large for
the thread caches only while the parked ones stay under that fraction of the peak of live bytes, else
gives it back to the system (`cleave-rt::depot_push`; `CLEAVE_ALLOC_STATS` reports the large blocks'
peak held and live bytes; `cleave-rt/tests/pool_park.rs`). Measured on nanoLM 116M (`bench 0 1 10`, one
run each), live peak 11.1 GiB in every case:

| `CLEAVE_POOL_PARK` | held at most | ms/step |
|---|---|---|
| unset | 18.3 GiB | 11200 |
| 0.5 | 14.1 GiB | 11939 (+6.6%) |
| 0.25 | 13.0 GiB | 12071 (+7.8%) |
| 0.1 | 12.2 GiB | 12317 (+10%) |

A block given back is faulted in again when its size is needed: every step pays it. Kept off; for a run
that must fit in less memory. **The lead left**: let a parked block serve a request of a nearby smaller
class (up to 2x smaller, say), so the blocks parked in one class during a phase serve the next phase's
sizes instead of new ones being taken from the system: less held without giving anything back. To
measure the same way (held at most, ms/step), and the waste a block serving a smaller request carries.

## Intermittent test failures under load, never captured

- 2026-10-08: `scripts/test.ps1 smoke` failed once in `cleave --test language_model_ops` (no test named,
  the binary's exit only) while a nanoLM training took every core; rerun alone, then the whole suite
  again, it passed.
- 2026-10-04: once, `leaks.rs::clipping_leaves_no_allocation_behind` failed (bytes left per clip above
  `NOISE`); once, a `leaks.rs` run printed no `test result` line (the process ended early). Neither came
  back in 12 runs. That day changed bufferization (`allow-return-allocs-from-loops`), refcounting
  (per-type glue) and the ABI (large light structs by pointer).
- Earlier, closed as not reproduced (`backlog-done.md`: "`cargo test --workspace` intermittently fails
  one heavy JIT test under concurrent load").

Next time: keep the binary's whole output (`cargo test ... 2>&1 | tee`), the test name and the exit
code; after a change touching those areas, a loop of runs (`for i in $(seq 50)`).

## Debt: a constraint on a never-generalized abstract variable is checked nowhere

`infer.rs`'s module comment, in its own words: a constraint on a variable still abstract and never
generalized (a `let mut`'s, say) "has nowhere further to travel once its enclosing scope finishes. It's
silently unchecked". Not in the backlog until the 2026-10-08 audit. The failure, if it comes, comes
later (an impl not found at monomorphization) without the `let`'s location. The same comment still
lists mutability checking as not done, which `check_mutability` has done for a while.

## Debt: `cleaveElideBlockCopies` relies on an assumption it can't check

`cleave-mlir/cpp/shim.cpp`: eliding a block copy is sound only if no layout constant (a stride,
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

## Views as first-class descriptors: a strided view that retains the refcounted tensor it looks into

Planned as Part 2 of `doc/plan-struct-arrays.md` (heap references: arrays of structs first, views on
the same base: retained references, copy on write).

`Slice::slice`/`update` (`stdlib/linalg/tensor.cleave`) give *ephemeral* views today: a slice becomes
a `memref.subview` inside one function, `Sgemm::sgemm` reads it with its real strides, and a block
written by an extern and put back where it was read is written in place
(`cleave_mlir::elide_block_copies`). A view that leaves its function is copied: function
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

## `Generate` for tensors beyond two dimensions

A comprehension fills a tensor through `Generate`, implemented for one and two dimensions
(`stdlib/linalg/tensor.cleave`). More needs one impl per rank, or an impl over the `Dims...` pack once
packs can be taken apart (`doc/plan-compile-time-sequences.md`, step 6). Arrays of tensors and of
structs as a comprehension's target work (checked 2026-10-09: `[for j in 0..4: make(m, j)]` returning
a struct, runtime-shaped body, no `spawn`).

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
never escapes, and can't scalarize anything around them. Since 2026-09-30, `cleave-mlir` annotates
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

## cleave's own MLIR dialects: IRDL or TableGen (exploration)

cleave reaches MLIR through its own C API and C++ shim (`cleave-mlir`, `doc/plan-mlir-shim.md`), so a
C++ build is already part of building cleave. What that opens, not started: dialects of cleave's own,
for its vocabulary and for the sibling language (cleave-gfx), next entry.

- **Building an operation of any dialect** needs nothing new: the generic operation state cleave builds
  with reaches any registered op.
- **Defining a dialect** (types, verifiers, canonicalization and folding, interfaces the generic passes
  consult, `MemoryEffectOpInterface` above all) is C++/TableGen in MLIR. Two ways:
  - **IRDL**: the dialect described as MLIR IR (`irdl.dialect`/`irdl.operation`), loaded at run time,
    no C++ compiled; structural verification only, last checked (no custom verifier, folding or
    assembly format) — to re-check against the pinned MLIR 22 before relying on it.
  - **TableGen in the shim's build**: `mlir-tblgen` from the toolchain generating the `.inc` files the
    shim compiles; full power (traits, interfaces, folders), and the ABI matches by construction since
    the shim builds against the same install. The cost: a C++ toolchain for anyone changing the dialect,
    which building cleave already needs now.
- Unregistered ops (`allowUnregisteredDialects`) are not an option: no verification, no interfaces.

The e-graph (pre-MLIR rewriting) and a dialect (MLIR's analyses and generic passes on cleave's
vocabulary) complement each other rather than overlap.

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

## Interval/value-range analysis (a fixpoint-iterated abstract-interpretation lattice, not literal fixed-point *arithmetic*) — raised in conversation as a plausible future direction, not designed or scoped yet

The idea, as raised: track each variable's own possible value *range* (an interval, not just "known constant or not"), converged via the standard fixpoint-iteration abstract-interpretation technique (widening/narrowing across a loop's own back-edge, Cousot-style) rather than exact constant folding alone. The concrete motivating use case named directly: a `for` loop's own **trip count** becoming staticaly derivable when its bounds are provably within a known range — unlocking unroll/unroll-jam decisions, bounds-check elimination, and potentially const-generic-shaped dispatch (`doc/backlog.md`'s own already-closed "const generic compared via an operator" entry, and the still-open OpenBLAS-motivated size-threshold dispatch it unblocked) for cases that are range-bounded but not literally constant.

**A strict generalization of exact-value folding, not a separate, parallel mechanism sitting next to it** — the closing insight from the conversation that raised this: a proven interval `[x, x]` (lower bound equals upper bound) *is* a proven constant, exactly the degenerate, single-point case of the same lattice. `const_eval.rs`/`Analysis::make`'s exact folding only ever answers "is this value exactly known;" a real interval lattice answers the strictly more general "what range could this value fall in," with exact-known-constant as one specific, already-converged point in that range — so a full interval-analysis pass, if ever built, would properly *subsume* today's exact folding rather than duplicate it, the same way any abstract-interpretation lattice's most-precise element degenerates to exact evaluation. A real interval lattice needs its own join/widen operators and its own fixpoint loop over the CPS graph's own loop-carried arguments (`doc/hld.md`'s own "Memory management: a region/stack discipline derived from CPS" section already establishes that a loop's own carried state is syntactically explicit in this pipeline's CPS form, `carried_types`/`params` on `CFunDef` — the same structural fact that made copy-propagation's def-use question local rather than needing classical dominance-frontier computation likely applies here too, worth checking directly before assuming a full classical dataflow pass is needed). Not designed, not scoped, no repro/motivating failure yet — purely a direction flagged for later evaluation once the more concrete gaps above are closed.

## A statically-zero-trip-count loop's body is *not* eliminated the way a compile-time-constant `if`/`and`/`or` branch already is — "works" at `--opt-level 2` only by incidental LLVM backend DCE, not by anything cleave's own pipeline proves, and silently stops "working" at `--opt-level 0`

Raised in conversation, directly after the `if`/`and`/`or` dead-branch-elimination work above closed: does the identical guarantee hold for `for i in 0..0 { extern_call(i); }`, a loop whose own trip count is staticaly zero? Tested directly, not assumed — it does not, and the difference is real, not cosmetic.

**`if`/`and`/`or` eliminates the dead branch structurally, before LLVM's own backend ever runs** — confirmed by the entry above: `--dump-mlir-lowered` (printed straight after cleave's own CPS-optimize/MLIR-canonicalize stages, before any LLVM backend codegen) already shows zero trace of the untaken branch. **A zero-trip-count loop does not get this treatment at all** — `--dump-mlir-lowered` on `for i in 0..0 { acc = never_called(i); }` still shows `never_called` both declared and called, inside a real loop construct, completely unchanged from a loop whose bound is a genuine runtime value. The loop is never recognized as trivially dead at the CPS/MLIR level cleave itself controls.

**What actually removes it from the final object is a different, unrelated mechanism, and a much weaker guarantee**: at the default `--opt-level 2`, `never_called` is genuinely absent from the emitted object's own symbol table (confirmed via `llvm-objdump -t`) — but this is LLVM's own backend dead-code elimination, running during real machine-code generation, well after cleave's own IR has already committed to emitting the call. Proof this is incidental, not structural: **the identical program built with `--opt-level 0` has `never_called` right back in the symbol table**, unresolved. Anyone building for debugging (`--opt-level 0`, `--no-inline`, the exact combination this project's own `--no-inline` doc comment recommends for reading a clean disassembly) would need a symbol the source-level logic proves is never actually called — and a genuinely side-effecting `extern fn` (not provably pure) might not even be safe for LLVM's own backend to drop at any opt level, meaning this "works" only for the specific, effect-free probe tested here, not as a general guarantee at all.

**Not fixed, not scoped in depth** — the natural extension of the already-closed `if`/`and`/`or` work above (same underlying idea: a provably-unreachable branch/body should never require a symbol to exist), but genuinely more work: `if`'s condition is a single boolean value already reaching the const-fold machinery for free; a loop's own trip count first needs deriving from its bounds (`end - start` provably `<= 0` for the simplest case, `0..0` here) before the identical "this is unreachable, don't even lower it" treatment could apply. Plausibly connects to the interval/value-range analysis entry immediately above (a trip count is exactly the kind of fact that lattice would derive), but a narrower, `<=0`-specific special case might be enough for the immediate BLAS-adjacent motivation without waiting on the full lattice. No design attempted here.

## Checkpoints: make them a general building block for long computations, not an ML tool — to revisit once the language core is stable

The `stdlib/checkpoint` module (step 0 of `doc/plan-nanolm.md`) serves any long-running simulation: fault tolerance, resuming a run, splitting a computation into segments. Yet today it does `use linalg; use optim;` and holds the `AdamState` and `Trainable` impls itself, so a fluid simulation using it drags the optimizer in for nothing. Deliberately postponed: cleaning up the stdlib is easier once the language has settled.

What to do, by priority:
- **Invert the dependency.** `checkpoint` keeps only the algebra, scalars, tensors and tuples. The `Checkpoint<AdamState<…>>` and `Checkpoint<M: Trainable>` impls move to `optim`, next to their types: an impl lives with its type, not with the module that serializes it.
- **A file checksum.** The atomic rename protects against a stop mid-write, not against a file damaged afterwards (disk, network copy). Add a CRC32 or xxhash at the end of the file, checked by `restore`; it's a format change, so a version 2 of `CLVCKPT`.
- **Rotation.** Keep the last k checkpoints (`run.ckpt.1`, `.2`…): if the latest is corrupt, resume from the previous one.
- **One-line resume.** A `restore_or(path, init)` that resumes if the file exists and starts from `init` otherwise. Makes a program resumable without plumbing.
- **Later, asynchronous writes.** Copy the state, then write it on another thread while the computation continues. Only worth it when writing really weighs (a big simulation), not for nanoLM.
- **Generic file I/O in the runtime.** Once cleave has real strings: the format and its checks would be written in cleave on top of `open`/`write bytes`/`read bytes`/`rename`/`close` primitives, instead of living in `cleave-rt/src/checkpoint.rs`.

## The in-process test harnesses don't run the matmul schedule: a test of a matmul there doesn't cover the real pipeline

Found while fixing matmuls whose column count isn't a multiple of 16: the test written in `cleave/tests/mlir_lower.rs` passed **even without the fix**, while the CLI (`--run`) failed. The harness pipeline in those files (`run_i32` and its relatives) doesn't apply `matmul_vectorize.transform.mlir`, so its matmuls take another lowering path. The regression test now goes through the CLI binary (`language_model_ops.rs`, `CARGO_BIN_EXE_cleave`). To fix: have the harnesses go through `pipeline.rs::lower_to_llvm` with the real options, or at least list which existing tests think they cover the schedule and don't.

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

## The main thread's frame: 570 KB of argument slots in nanoLM's `train_gpt$tasks`

Half the main thread's stack (light structs of 128 bytes or more cross calls by pointer to a copy, the
slots hoisted to the entry block with `lifetime.start`/`lifetime.end`; `backlog-done.md`). A slot whose
uses span several blocks gets no lifetime and keeps the whole frame, as does a spawned call's; and the
copy is redundant whenever the argument already lives in memory. Passing the caller's own storage, or
a larger main-thread stack (`/STACK` at link, `-z stacksize`/a spawned main thread on Linux), would
remove the margin question.

## A gradient compiled once per function, not once per inlined call

nanoLM's `train_gpt` gradient is necessarily fully inlined: synthesis sees through `block`, so its
backward is emitted six times (once per layer) where a per-function gradient (the backward of `block`
as its own function) would compile it once. The largest remaining share of nanoLM's compile time.

## Consumed parameters (`own`): a move instead of a borrow

Every parameter is borrowed today: the caller keeps its reference and releases it. A consumed parameter
would hand the callee the caller's reference: retain/release pairs gone on every transfer, and `Buffer<T>`
able to reallocate its object itself (one allocation, no indirection to its slots) since `v.buf =
grow(v.buf, n)` would then give `grow` the only reference (`doc/plan-buffer.md`). Needs use-after-move
checking in inference and the transfer in `refcount.rs`.

 