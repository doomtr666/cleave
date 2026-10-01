# Indexable collections as algebras: arrays, tensors, tuples and structs

Status: design agreed (2026-10-01). Step 1 done (2026-10-02), see "Step 1 as built" below.

## The problem

Three things need "do this for every element" over values whose elements have **different types**:

- **Training a user model.** `step(opt, net, grad, state)` must apply to every layer of a `Network`, and
  the layers' types differ (`Dense<f32, 784, 512>`, `Dense<f32, 512, 256>`, ...). Today the user writes a
  35-line `impl Optimizer` plus a parallel `NetworkState` struct by hand
  (`examples/mnist-interop/src/kernel.cleave`).
- **Printing several values.** `stdlib/io/io.cleave` carries fifteen copy-pasted `impl Print<(A, B, ...)>`,
  one per tuple arity from 2 to 16.
- **Anything "per field" later:** `Init` (`let net: Network = he();`), saving/loading weights, gradient
  tangents.

Rejected along the way: a `derive` generator hard-coded in the compiler (against "everything through
algebras and the stdlib"); template-style recursion over `Cons<H, T>` lists (it already works in cleave
today, but the stdlib should not establish that practice); `t.i` field syntax; impl specialization ("the
most specific impl wins": a partial order, ambiguous as soon as two impls overlap without one containing
the other, and it lets a new impl silently change which code runs elsewhere).

## The idea: what `+` already does, applied to indexing

The compiler doesn't know how to add two `i32`: `a + b` desugars to `add(a, b)`, and `impl Ring<i32>`
in the stdlib says how, through an MLIR operation (`doc/hld.md`, "primitive types are algebras too").
Indexing gets the same treatment. Every collection — array, tensor, tuple, struct — is indexable through
algebras declared in the stdlib; the compiler keeps only what can't be expressed there.

The collections differ in one respect: **is the index a value or a constant?**

| collection | index | element type | algebra |
|---|---|---|---|
| array `[T; N]` | runtime value | always `T` | `Index<C, Elem, K>` |
| tensor `Tensor<T, Dims...>` | runtime value(s) | always `T` | `Index` (already today) |
| struct, tuple | constant (must fold) | depends on the index | `Field<S, I, F>` |

## The algebras

```
algebra Len<C> { fn len(c: C) -> i32; }                      // every collection

algebra Index<Container, Elem, const K: i32> {               // homogeneous, runtime index
    fn index(c: Container, idx: [i32; K]) -> Elem;
}

algebra Field<S, const I: i32, F> {                          // heterogeneous, constant index
    fn get(x: S) -> F;
}

algebra Collect<Target, Source> {                            // comprehensions, by target type
    fn collect(x: Source) -> Target;
}
```

`Index` exists today (`stdlib/linalg/tensor.cleave`, used by `Tensor` and `DynArray`); arrays would get
`impl<T, const N: i32> Index<[T; N], T, 1>` on top of the primitive array read (see "What stays primitive"), so generic code indexes arrays, tensors and structs alike. Writing (`a[i] = v`) gets the symmetric algebra (a `store`), slicing a
`Slice` algebra. `Index` moves from `linalg` to a prelude crate, since arrays are everywhere.

**`Field` is the heterogeneous case, and the hard part solves itself.** The type of `x[i]` depends on the
value of `i`. As an algebra, that is a multi-parameter algebra whose `F` appears only in the result —
exactly `Convert<From, To>`'s shape, which inference already resolves: dispatch finds the impl for
`I = 2`, which fixes `F`, and coherence (`check_no_overlapping_impls`, run on every compile) guarantees
only one exists. A type-level function, with no dependent types and no templates.

**The compiler synthesizes the struct impls, as data,** the way it already synthesizes `HeapStruct` for
every struct (`driver.rs::synthesize_heap_struct_marker_impls`):

```
impl Len<Network> { fn len(x) { 4 } }
impl Field<Network, 0, Dense<f32, 784, 512>> { fn get(x) { x.l1 } }   // the primitive projection
impl Field<Network, 1, Dense<f32, 512, 256>> { ... }
...
```

A tuple is already an anonymous struct (`(a, b, c)` desugars to `__Tuple3(0: a, 1: b, 2: c)`,
`driver.rs::synthesize_tuple_structs`), so tuples are covered with no extra work. An array satisfies
both algebras: its elements all share `T`, so `Field<[T; N], I, T>` holds for every `I`.

**`T: Struct`** becomes an ordinary bound (`algebra Struct<S> {}`, synthesized per struct), like
`T: Int`: "this type has fields, indexable at compile time".

## The rule: legal wherever it folds — the const-generics rule

A const generic accepts anything that folds at compile time; nothing marks it as special. That is what
makes cleave feel dynamic, and the same rule applies here, with no keyword or hint:

- **`x[i]`** desugars to `Index` (homogeneous collection, `i` any value) or `Field` (heterogeneous,
  `i` must fold — otherwise a located error, never a guess). `x.l1` desugars to `Field` too, the name
  turned into its position. `t.0` goes away in favor of `t[0]`.
- **`for i in 0..x.len { ... x[i] ... }`** stays a runtime loop when the indexed collections are
  homogeneous, and is **unrolled** by the compiler when one is heterogeneous: each copy has a constant
  `i`, so each `x[i]` is an ordinary `Field` dispatch. Zip is free: one index, several collections
  (`model[i]`, `grad[i]`, `state[i]`).
- **Comprehensions** `[for i in 0..n: f(x[i])]` are an algebra too, `Collect<Target, Source>`
  (`fn collect(x: Source) -> Target`, Rust's `collect()`/`FromIterator`), with the target taken from the
  expected type — the way `xavier()` already picks its impl. Like a number literal: the context decides
  (`let net: Network = [for ...]`, `let t: Tensor<f32, 4> = [for i in 0..4: ...]`), and when nothing
  does, a documented default applies: a tuple when element types differ (the length must fold), an array
  `[T; N]` when they unify and the length folds, a `DynArray<T>` when they unify and the length is only
  known at run time. The stdlib provides `Collect` for tensors (a way to initialize one by formula); the
  compiler synthesizes it for each struct, from its fields' values — which is how a comprehension
  rebuilds a `Network`.
- **Runtime-length collections** (`DynArray<T>`) are homogeneous, so nothing about them needs to fold:
  `v[i]` goes through `Index`, `v.len` through `Len` as a runtime value, and `for i in 0..v.len` stays a
  runtime loop. What must fold is decided by heterogeneity, not by the collection kind: an algebra over
  `DynArray<L>` (`Optimizer` for a model whose depth is a run-time parameter) is a plain loop.
- **Slices** `x[1:]`, `x[:25]`, `x[1:17]`: bounds must fold for a tuple (a shorter tuple); folding bounds
  on an array or tensor keep a static shape (`[T; 16]`); runtime bounds need a dynamically-sized type,
  out of scope here.
- **Packs** follow the same rule (`Dims.len`, `Dims[0]`, `for d in Dims`, comprehensions over them); a
  value typed by a pack (`args: Args...`) is already a tuple.

## What it buys

```
impl<Ts...> Print<(Ts...)> {                        // replaces the fifteen per-arity impls
    fn print(x) { for i in 0..x.len { print(x[i]); }; x }
}

impl<Opt, M: Trainable, S> Optimizer<Opt, M, S> {   // any model the user marked `Trainable`
    fn init_state(opt, model) { [for i in 0..model.len: init_state(opt, model[i])] }
    fn step(opt, model, grad, state) {
        let pairs = [for i in 0..model.len: step(opt, model[i], grad[i], state[i])];
        let new_model: M = [for i in 0..pairs.len: pairs[i][0]];     // Collect<M, ...>: rebuilds M
        (new_model, [for i in 0..pairs.len: pairs[i][1]])            // the state: a tuple, by default
    }
}
```

## No blanket impl over every struct

An impl "for every struct" (`impl<S: Struct> Print<S>`) overlaps any impl for one specific struct
(`impl Print<Complex<T>>`), and coherence rightly rejects it; resolving the overlap by specificity is the
rejected specialization. So `Struct` is a capability for generic code, and an algebra that should apply
to "all fields of these structs" uses its **own explicit marker**, added by the user to the structs it
concerns (`impl Trainable<Network> {}`): no struct becomes trainable by accident, and no overlap is
possible.

## A user-written impl replaces the synthesized one

The same rule as name resolution (`resolve.rs`: what the user writes shadows what the system provides),
applied to impls: if the program contains any user-written impl of `Field`, `Len` or `Collect` for a
struct, the compiler synthesizes none of that algebra for that struct. This is not specialization: both
candidates target exactly the same concrete struct, so there is no partial order to consult, and the
winner is visible in the user's own code. The grain is the whole algebra per struct (writing one `Field`
for `Network` means writing all of them), so synthesized and hand-written fields never mix. Two
hand-written impls that overlap, or a user *generic* impl covering a struct among others, remain real
overlaps and are rejected as today.

## What stays primitive in the compiler

Constructing and projecting a struct, and reading and writing an array element, stay primitive: they
are the boundary of the memory abstraction. Defining projection in an algebra would mean saying *where*
a field lives — and the compiler, not the stdlib, picks a struct's representation (a light struct is an
SSA `!llvm.struct`, projected with `extractvalue`; a heap one is a refcounted pointer, projected with an
address computation and a load), and switches it by analysis (mutation, crossing an `extern`, ...). An
algebra able to express that would expose addresses and pointer arithmetic, which cleave rules out by
design. So `Field`, `Index` and `Len` *name* these primitives for generic code — a synthesized `Field`
impl is the primitive projection — and never replace them. (A `Tensor` can define its own indexing in
the stdlib through `mlir::tensor` because an MLIR tensor is a value, not an address.)

- **Type constructors** and their memory layout (`[T; N]`, structs, `#[mlir_type]` types).
- **Synthesizing** `Len` / `Field` / `Struct` / `Collect` impls for each struct, as data — on demand rather than for
  every struct and arity up front, to keep compile time in check.
- **Unrolling** a `for` whose index must fold. In CPS the copies chain through their continuations
  (copy 0, then copy 1, ..., then what follows the loop), so `break` in copy *i* is a tail call to the
  continuation after the loop and `continue` one to copy *i + 1* — simpler than an ordinary loop's
  `__loop_running` flag, since unrolled code has no back edge. A `for` stays `()`-typed, so its `break`
  carries no value, as today.

Indexing, measuring, slicing and iterating all go through the stdlib.

## Risks and safeguards

- **Autodiff.** `grad` through a struct relies on the e-graph knowing field projection and struct
  construction (`egraph.rs::struct_projection_rewrites`, `construction_derivative_rewrites`). Keeping the
  projection primitive underneath `Field` keeps those rules valid; nothing here should route `x.l1`
  through an algebra call that the e-graph doesn't see through.
- **Memory analyses.** Refcounting, region and alias analyses reason on the array load/store primitives
  as explicit effects. Arrays therefore satisfy `Index` *on top of* their primitives (step 4), never by
  replacing them.
- **Performance.** Every step checks that the generated MLIR is unchanged where nothing new is used, and
  that MNIST keeps its accuracy and time.
- **Code size.** Unrolling copies the body once per element; the compiler picks loop or unrolling by
  homogeneity, invisibly in the source. Acceptable (a heterogeneous structure's size is written in its
  type), but worth watching.
- **Output-only generics.** `Field<S, I, F>` leans on the same inference path as `Convert<From, To>`,
  historically a fragile one; it will be exercised much harder.

## Structural autodiff moves to the stdlib too

`Index` already declares its derivative rule in the stdlib (`stdlib/linalg/tensor.cleave`: reading an
element and differentiating commute). With `Field` and `Collect` as algebras, the struct rules
hard-coded in the e-graph today (`egraph.rs::struct_projection_rewrites`,
`construction_derivative_rewrites`) become ordinary `derivative`/`adjoint` declarations:

- `Field::get`, forward: `get(d(x))` — the derivative of field *i* is field *i* of the derivative;
- `Collect`, forward: the collection of the derivatives;
- `Field::get`, reverse: a collection with the upstream adjoint at position *i* and zeros elsewhere,
  `adjoint get<I>(x), u: [for j in 0..x.len: if j == I { u } else { zero() }]`.

It also settles the tangent type of `grad` on a struct with non-differentiable fields: "the tangent of a
struct is the struct of its fields' tangents" is a comprehension, and a `Tangent` algebra can map `i32`
to `()`.

**Requirement: an `if` whose condition folds keeps only the branch taken.** In the adjoint above each
unrolled copy only type-checks one branch (`u` at position *i*, a zero of field *j*'s type elsewhere), so
a folding condition must select its branch at compile time and the other branch must not be typed — the
same "decided at compile time when it folds" rule.

What stays in the compiler for autodiff: the chain rule, differentiating through control flow (CPS), and
the forward/reverse algorithms in the e-graph. Every local rule — math functions, projection,
construction, indexing — lives in the stdlib.

## Tuples of any length

Tuples desugar to `__TupleN` structs declared up front for N = 2..16 (`driver.rs::MAX_TUPLE_ARITY`); a
comprehension or slice can produce any length, so these get synthesized on demand, per arity the program
actually uses, instead. Plumbing, not design. (Unrolling, by contrast, is not a size problem to work
around with a loop: heterogeneous elements need different code per element, so there is no loop to
emit; homogeneous ones belong in an array.)

## Error messages in unrolled code: best effort

An error inside an unrolled body points at the source line; naming the iteration too (`i = 2`, element
type `Dense<f32, 256, 128>`) is the goal, pursued best effort and improved over time, not a gate for any
step below.

## Writing a field by index; homogeneous structs

`x[i] = v` on a struct is the symmetric of reading: a write method beside `Field::get`, synthesized the
same way, under the same rules (`i` must fold, `v` must have field *i*'s type). Structs are already
references mutated in place (`v.x = 10.0` works today), so there are no new semantics to invent.

A struct whose fields all share one type (`struct Vec3 { x: f64, y: f64, z: f64 }`) is homogeneous like
an array, so it also gets `Index` synthesized: `v[i]` with a run-time `i`, and `for i in 0..3 { ...
v[i] ... }` stays a real loop. `v[0]` and `v.x` name the same component, as in GLSL/HLSL or Eigen — odd
for a general-purpose language, natural for a scientific one.

## Step 1 as built

Simpler than the `Field<S, I, F>` algebra sketched above, with the same effect:

- `x[k]` with a single folding index on a struct or tuple that has no `Index` impl
  (`infer.rs::is_positional_struct`) is the primitive projection of its `k`-th field: inference resolves
  it as a field access named `[k]` (`positional_field_name`), CPS emits the same `PrimOp::Field` as
  `x.name` (`ConcreteUnit::positional_fields`), so autodiff and the memory analyses see nothing new.
- In generic code (`fn first(t) { t[0] }`) the positional access is deferred and generalized through the
  existing `FieldConstraint` machinery, exactly like `t.name`, so "the `k`-th field's type" is resolved
  per instantiation without a type-level algebra.
- A type with an `Index` impl (`DynArray`, `Tensor`) keeps `Index`: the run-time path.
- `Len` lives in a new prelude crate, `stdlib/core`: arrays (`N`), `DynArray` (moved from a top-level
  `fn len`, which would otherwise shadow the algebra by name resolution), and one impl synthesized per
  struct and tuple (`driver.rs::synthesize_len_impls`), skipped when the program writes its own.
- `t.0` still works alongside `t[0]`.

`Field` as a user-visible algebra is therefore not needed yet; it would come back only if user code must
redefine positional access for a struct.

## Suggested order

1. ✅ `Len` synthesized for structs and tuples; `x[i]` with a folding index as positional projection;
   `t[0]` replaces `t.0`.
2. Unrolled `for`; collapse the fifteen `Print` impls as the first real use.
   - 2a ✅ (2026-10-02) concrete collections: `unroll.rs`, run from `driver::compile` after name
     resolution — a trial inference records unroll requests (`Infer::unroll_requests`: a `for` with
     folding bounds, `len(t)` included, whose body indexes a struct or tuple by its variable), the loop
     is rewritten into one copy per index (fresh node ids, the variable replaced by its value, wrapped in
     `loop { ...; break; }` when the body breaks), and the round repeats for nested loops. Found and
     fixed on the way: a `break` following an earlier statement that may break was lost (CPS
     `continue_after` didn't carry the loop's running flag), so the loop ran forever.
   - 2b ✅ (2026-10-02) impls over a pack of types: `impl<Ts...: Print> Print<Ts...>` replaces the
     fifteen per-arity tuple impls in `stdlib/io/io.cleave`. A type pack used as a whole type is the
     tuple of its elements (`infer.rs::TUPLE_OF_PACK`, unifying with any `__TupleN`), a bound on a pack
     holds element by element (`has_matching_impl`), and such an impl is a template: `unroll.rs` takes
     it out of the program and adds one concrete impl per tuple type the trial inference finds it used
     at, whose loop then unrolls like any other. The template's body is checked per instance, the same
     posture every generic impl body already has in cleave (checked permissively, real errors at
     instantiation); checking generic bodies once with rigid type variables would be new for all
     generic code, not specific to packs (`doc/backlog.md`).
   - Brought forward: an `if` whose condition folds should keep only its taken branch (listed under
     step 7) — the first natural unrolled loop (`if i == 1 { a = a + t[i] } else { b = b + t[i] }`)
     needs it, since every copy otherwise type-checks both branches.
3. Comprehensions through `Collect` (tensors, synthesized for structs); `Optimizer` for marked structs; the MNIST kernel without its hand-written
   `impl Optimizer` and `NetworkState`.
4. Arrays also through `Index` (and its writing counterpart) in the prelude, on top of the primitive
   array load/store, not replacing it.
5. Slices.
6. Packs in the same style.
7. Structural autodiff rules (`Field`, `Collect`) declared in the stdlib, replacing the e-graph's
   hard-coded struct rules; a `Tangent` algebra for non-differentiable fields; folding `if`s keeping only
   the branch taken.

Each step is useful alone and testable on its own.
