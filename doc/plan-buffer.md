# Plan: `Buffer<T>`, the one primitive dynamic containers are built on

Status: done (2026-10-10, `backlog-done.md`: "`Buffer<T>`, and `DynArray` built on it", "`take`, `pop` and `HashMap`"). Left: consumed parameters (`doc/backlog.md`) for the one-allocation variant.

## Why

`DynArray<T>` (`stdlib/dynarray`) is built on `RawBuf<T>`: a `realloc`-backed pointer from `cleave-rt`
with no refcount header, one `extern` impl per element width (26 runtime symbols), and element
refcounting done in Rust, which doesn't know the element's type. Consequences: a `DynArray`'s storage is
never freed, whatever `T`; the elements it holds when it dies are never released; an overwritten
element is released flat, without its type's cascade; tensors and light structs can't be elements.
And every other dynamic container (a hash map, a deque, a heap) would need the same machinery again.

## Design

One compiler-known type, `Buffer<T>` (`stdlib/buffer`, tagged `#[mlir_type(buffer)]`): a refcounted
object (`cleave_alloc_rc`) holding `{data, cap, slot_bytes}`; `data` is a separate, zeroed allocation of
`cap` slots, each laid out as an element of `[T; N]` is (a scalar inline, a heavy struct's pointer, a
light struct inline, a tensor's descriptor).

- **Identity is stable** (`doc/backlog.md`, the B variant needs consumed parameters): `grow` reallocates
  `data` inside the object, so a buffer is a reference like any heavy struct, and every holder sees it
  grow. Elements move bitwise on a realloc: no retain, no release.
- **An all-zero slot is empty**: allocation and growth zero the new slots; the buffer's release skips
  them. The buffer never needs to know which slots are live, so sparse containers (open addressing)
  work as well as a prefix (`DynArray`'s `len`).
- **Reading and writing a slot are `PrimOp::Load`/`Store` with `array_ty = Buffer<T>`**: every pass
  already handles them (retain on store, release of the overwritten element in lowering, retain on an
  owned read, aliasing commitments, region escapes). Only their lowering learns the buffer's address
  computation (`data + i * slot`). The e-graph treats a buffer `Store` as an effect.
- **Allocation is `PrimOp::BufferAlloc`** (`mlir::cleave::buffer_alloc(cap)`): only lowering knows the
  slot's size. **Growth and capacity are runtime calls** (`cleave_buffer_grow`, `cleave_buffer_capacity`)
  reading the slot size from the object.
- **Release**: when the count reaches zero, a loop over the `cap` slots releases each non-empty one
  through `T`'s cascade (`push_cascade_leaf`, as an array of structs does, but a `scf.for` rather than
  unrolled), then frees `data`.

`DynArray<T>` becomes `{buf: Buffer<T>, len: i32}` in plain stdlib; `RawBuf`, `RawBuffer`, `HeapStruct`,
the 26 `dynarray_*` symbols and `refcount.rs`'s "never constructed" exclusion go.

## Steps

1. `Buffer<T>`: runtime (`cleave_buffer_alloc`/`grow`/`capacity`/`free_data`), type recognition
   (`ty_to_mlir`, `is_refcounted`, not light), `BufferAlloc`, `Load`/`Store` lowering, the release loop.
   Tests: scalar and struct elements read back, overwrite releases, growth keeps elements, a dying
   buffer releases its elements and its storage (`leaks.rs`).
2. `DynArray` on `Buffer`; remove `RawBuf` and the `dynarray_*` runtime. Tests: existing `DynArray`
   tests, `examples/convex_hull`, a `DynArray<Point>` leak test, a `DynArray` of tensors.
3. `take` (read a slot and leave it empty), for `pop`/`remove`.
4. A `HashMap<K, V>` (open addressing) in stdlib, to validate the abstraction.

Later, separately: a consumed-parameter mode (`own`) would let `grow` realloc the object itself (one
allocation, no indirection).
