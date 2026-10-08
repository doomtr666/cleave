# Heap references: arrays of structs, then views

One model for every value that refers to refcounted storage it doesn't own alone: an array of
structs (or of tensors) holds a reference to each element; a view (`doc/backlog.md`, "Views as
first-class descriptors") holds a reference to a buffer with an offset, sizes and strides. Both need
the same three things: references retained and released, a write in place when the storage isn't
shared and a copy otherwise (copy on write), and a way across function boundaries that doesn't copy.
They differ in one each: an array owns its elements one by one (a release loop), a view has a
layout (strides, which code must be compiled for). Built in that order: the shared base and arrays
first, which have no layout question and unblock a model whose depth is a constant, then views on
the same base. They compose: a slice of an array of layers is a view over its handles.

# Part 1: arrays of structs, `[S; N]` as a real array

## The problem

A model's depth is its number of blocks; today it is written into the model's type. nanoLM's `Gpt`
has eight fields `b1`..`b8`, and `new_gpt`, `gpt_logits` and the PyTorch twin follow: twelve
layers means editing all of them. The natural spelling is a field `blocks: [Block; LAYERS]`, a
loop over it, and the depth a `define`:

```
struct Gpt { tok: Embedding<f32, VOCAB, WIDTH>, blocks: [Block; LAYERS], nf: Tensor<f32, 1, WIDTH> }
impl Trainable<Gpt> {}
fn gpt_logits<const N: i32>(x: [i32; N], m: Gpt) -> Tensor<f32, N, VOCAB> {
    let mut h = embedding_forward(m.tok, x);
    for i in 0..LAYERS { h = block(h, m.blocks[i]); };
    embedding_logits(m.tok, rms_norm(h, m.nf))
}
```

What any program with repeated parts (layers, particles, mesh cells) wants, not nanoLM only.

## What a probe showed (2026-10-08)

- A *tuple* of layers works end to end once the stdlib says a tuple of trainable parts is trainable
  (`impl<Ts...: Trainable> Trainable<Ts...>`, same for `Parameters`, `stdlib/optim`): a three-layer
  network built by a comprehension, an unrolled `for` in its forward, `grad`, `Sgd`, identical to the
  bit to the same network written with three named fields. Not kept: a tuple of `Parameters` then
  matches both `Checkpoint`'s impl for every tuple of checkpointable values (what saves `(model,
  state, seed)`) and its impl for `Parameters` structs, overlapping impls the language rejects.
  Arrays get their own `Parameters`/`Trainable` impls instead, which overlap nothing.
- `[Layer; N]` as a field type fails: no `Trainable` for arrays, and the unresolved impl ends in a
  panic in CPS (`call_names resolved Optimizer::init_state to ...'t1031, but no such unit exists`)
  instead of a located error.
- Reverse-mode AD goes through a loop with constant bounds (the e-graph unrolls it): `grad` through
  `for i in 0..3 { h = tanh(h * w) }` matches finite differences. A loop over `0..LAYERS` is
  differentiable as is; a truly dynamic bound is not (a separate, later question).
- MLIR refuses `memref<4x!llvm.ptr>` and `memref<4x!llvm.struct<...>>`: a memref's elements are
  builtin types only (integers, floats, `index`, vectors).
- `grad` requires explicitly typed parameters, and positional access on an unannotated generic
  parameter (`net.layers[i].d`) fails inference: two inference gaps met on the way, independent of
  arrays.

## Representation

Two kinds of element, as for struct fields today.

**Heavy structs** (refcounted, reached through a pointer: `Block`, `Dense`, anything holding a
tensor): the array holds element *handles*, each the element's pointer as an `i64`
(`memref<N x i64>`, `llvm.ptrtoint`/`llvm.inttoptr` at the boundary). A builtin element type, so the
memref machinery (allocation, loads and stores, loops, bufferization) applies unchanged. The array
holds one reference to each element: storing retains, overwriting or dropping releases.

**Light structs** (by value: `{x: f64, y: f64}`): no pointer to hold. Split field by field, one
memref per field (struct of arrays: no indirection, vectorizable, the layout a numerical code wants
for particles or mesh cells), rather than boxed (each element a heap copy behind a handle: uniform,
an indirection per access). Decided 2026-10-08: struct of arrays.

**Tensors** are refcounted buffers too: `[Tensor<f32, R, C>; N]` is an array of handles like an
array of heavy structs, built with it (step 2).

**The array itself** is a value like any array today, but one that owns references: its
deallocation must release its elements first, which MLIR's buffer deallocation knows nothing about.
So an array of heavy structs is itself a heavy, refcounted object (`cleave_alloc_rc`), released by
a loop over its handles when its count reaches zero, as `__release_leaves` does for a struct's
fields. It can then be a struct's field (`Gpt.blocks`), shared, retained, released like any
heavy value.

## Semantics

- `N` folds at compile time (a literal, a `define`, a const generic): the array's type, its size.
- Reading `a[i]`: `i` may be dynamic; the element comes out retained (a heavy value read from a
  container, as a field read does).
- Building: a comprehension `[for i in 0..N: e]` whose target is `[S; N]` (from the context: a
  field's type, a `let` annotation) fills an array instead of unrolling into a tuple; `Generate`
  and `Collect` gain the array-of-structs impls.
- Writing `a[i] = v` on a `let mut` array: in place, the old element released, `v` retained.
  Arrays have reference semantics, whatever their element type: `let b = a` names the same array,
  and `b` sees a later `a[i] = v`, as arrays of numbers always have (`[1, 2, 3]` included; the
  "storage types" of `doc/backlog.md`'s `mut` entry), as in Julia, numpy or Fortran. A copy is
  written explicitly (`[for i in 0..N: a[i]]`). Decided 2026-10-08, rather than copy on write for
  arrays of structs only, which would have made `b = a` mean a different thing depending on the
  element type.
- `len(a)` is `N` (`Len`, as for arrays today).

## Algebras (stdlib, no compiler special case)

- `Index<[S; N], S>` and `Len` for arrays of structs.
- `impl<T: Parameters, const N: i32> Parameters<[T; N]>` and the same for `Trainable`: the generic
  walkers (`Optimizer`, `Accumulate`, `GradNorm`, checkpoints) iterate `0..len(m)`, which folds,
  so they keep working element by element; their results (a state per element, a gradient per
  element) are arrays of the matching element types, the comprehension's target pinned by the
  signature as for tuples today.
- Autodiff: an element read is a projection, like a field read. With constant indices (the loop
  unrolled by the e-graph) it reuses the field machinery directly: the adjoint of `a[k]` adds into
  element `k` of the array's adjoint. A dynamic index needs an `Index` adjoint rule
  (`update(zero, i, u)`, `Slice`'s pattern) — only once differentiating a dynamic loop is on the
  table.
- `spawn` over elements (the optimizer's per-field tasks) as over tuple elements.

## Steps, each with its tests

1. The CPS panic on an unresolved impl becomes a located error (independent, first).
2. Types: `[S; N]` with a struct `S` accepted in signatures and fields; heavy elements as
   `memref<N x i64>` handles inside a refcounted array object; construction from a comprehension
   and from an explicit literal; reading `a[i]` with a dynamic `i`; release of the elements. Tests:
   values, and `leaks.rs`-style bytes per iteration for building, reading and dropping arrays.
3. `a[i] = v` in place, the overwritten element released. Test: `leaks.rs`, the write seen through
   an alias, no bytes left per store, for a light and a heavy element. Done.
4. `Parameters`/`Trainable` for arrays; `grad` through a loop over an array of layers; `Sgd`,
   `AdamW`, `Muon` steps; checkpoints. The probe's network rewritten with `layers: [Layer; 3]`,
   identical to the bit to the named-field version. Done: `language_model_ops.rs` (the gradient,
   bit-exact), `leaks.rs` (Sgd, AdamW, Muon), `checkpoint.rs` (a resumed run, bit-exact), the
   generic `Parameters`/`Checkpoint` impls covering arrays with no new code.
5. Light structs as struct of arrays (or boxed, if the decision above changes).
6. nanoLM: `blocks: [Block; LAYERS]`, `LAYERS` a `define`, the twin reading it too. Same losses to
   the bit at eight layers; then the larger model (`d768`, twelve layers, context 512), first an
   intermediate size under a memory watchdog.
7. The two inference gaps (`grad` of an unannotated parameter, positional access on a generic
   parameter), which the array version no longer needs but other programs do.

## Open questions

- Nested arrays of structs (`[[S; M]; N]`): handles to arrays, the same mechanism one level up;
  not needed for the first use.

# Part 2: views on the same base

The backlog entry "Views as first-class descriptors" has the design: a view is a value
`{storage, offset, sizes, strides}` retaining the buffer it looks into (the MLIR memref descriptor,
its allocated pointer the buffer's `cleave_alloc_rc` data), value semantics through the refcount
(an `update` of a shared buffer copies), layout as a type-level property inferred and monomorphized
on (contiguous code stays as fast as today), function boundaries taking the layout of the type
instead of `identity-layout-map` everywhere, offsets as `[i32; Dims.len()]` for rank N, and
rank-reducing views by partial indexing. What Part 1 leaves for it: retaining and releasing a
referenced buffer, and copy on write, built and tested on arrays first. The precondition
`cleave_mlir_shim::elide_block_copies` relies on disappears with descriptors (the layout is a run
time value wherever a view goes).
