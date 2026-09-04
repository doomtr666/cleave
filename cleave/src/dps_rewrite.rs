//! Destination-passing rewrite: eliminates the redundant scratch-buffer
//! copy `mlir_lower.rs::store_native_shape_field` emits on every struct-
//! field `Tensor` *write* -- found dominant directly, not assumed (VTune,
//! `examples/mnist-interop`, a real training run: ~38% of wall time in
//! unresolved `VCRUNTIME140.dll` frames -- confirmed by their size pattern
//! and position in the call tree to be `memcpy`, not compute -- plus
//! another ~12% in `malloc`/`free`).
//!
//! **The redundant copy, precisely**: `store_native_shape_field` computes
//! the value to store (`%value`, a `tensor<...>`), wraps it via
//! `bufferization.to_buffer` to get a raw pointer, allocates a *separate*,
//! fresh `cleave_alloc_rc`'d buffer, and `llvm.intr.memcpy`s `%value`'s own
//! data into it -- see that function's own doc comment for exactly why the
//! copy exists (the MLIR-tracked buffer backing `%value` would otherwise be
//! freed out from under the struct's own field by `--buffer-deallocation-
//! pipeline`, confirmed by direct testing). The copy is real, unavoidable
//! *as that function is written* -- but not unavoidable in general: if
//! `%value`'s own producer (after `--inline`/`--linalg-fuse-elementwise-
//! ops` have already collapsed cross-function calls into one body, `pipeline
//! .rs`'s own earlier stage) is a single-result `linalg.generic`/`linalg.
//! matmul` seeded by a plain `tensor.empty()`, that seed can be replaced
//! with the *destination* buffer instead -- letting One-Shot Bufferize
//! write the result directly where it needs to end up, no scratch buffer,
//! no copy at all. Confirmed directly against this toolchain on a minimal
//! probe before writing any of this: a `linalg.generic` with `outs(%dest :
//! memref<...>)` where `%dest` comes from `bufferization.to_tensor %ptr
//! restrict writable` writes straight into `%ptr`'s own backing memory,
//! post-bufferize -- exactly the promise `load_native_shape_field`'s own
//! read-side fix already relies on, applied the other direction.
//!
//! **Why this is a separate module, not folded into `mlir_lower.rs`**: it
//! operates on the module *after* `--inline`/`--linalg-fuse-elementwise-
//! ops` have already run (`pipeline.rs`'s own stage ordering) -- a genuine
//! second pass over already-lowered, already-lightly-optimized IR, not
//! part of the original `lower_program` construction at all. Conservative
//! by construction, matching `doc/hld.md`'s own escape-analysis posture
//! ("only needs to be conservative, not perfect"): every precondition below
//! is checked structurally before any mutation happens, and a single
//! mismatch anywhere in the chain leaves that one struct-field write
//! completely untouched, falling back to the always-correct copy-based
//! path `store_native_shape_field` already emits -- this rewrite can only
//! ever make a store *cheaper*, never *different*.

use melior::Context;
use melior::ir::attribute::{
    Attribute, DenseI32ArrayAttribute, DenseI64ArrayAttribute, FlatSymbolRefAttribute,
    IntegerAttribute, TypeAttribute,
};
use melior::ir::operation::{
    OperationBuilder, OperationLike, OperationMutLike, OperationRef, OperationResult,
};
use melior::ir::r#type::{IntegerType, MemRefType, RankedTensorType};
use melior::ir::{
    BlockLike, Identifier, Location, Module, Region, RegionLike, ShapedTypeLike, Type, Value,
    ValueLike,
};
use melior::dialect::{arith, func, llvm};

/// `OperationMutLike` (needed for `move_before`/`set_operand`/`remove_from_
/// parent`) is implemented for `OperationRefMut`, not the plain `Operation
/// Ref` every walk/match helper below returns -- the same conversion melior's
/// own `walk_mut` uses internally (`operation.rs`'s own test code, `unsafe {
/// OperationRefMut::from_raw(operation.to_raw()) }`), reused here rather than
/// invented: both ref kinds wrap the identical raw `MlirOperation` handle,
/// this only widens which methods are callable on it.
fn as_mut<'c, 'a>(op: OperationRef<'c, 'a>) -> melior::ir::operation::OperationRefMut<'c, 'a> {
    unsafe { melior::ir::operation::OperationRefMut::from_raw(op.to_raw()) }
}

use crate::mlir_lower::memref_descriptor_llvm_type;

/// Runs the rewrite over every candidate in `module`, in place. Safe to run
/// on a module that has none (a program with no struct-field `Tensor`
/// writes at all) -- finds nothing, changes nothing.
pub fn eliminate_redundant_field_store_copies<'c>(context: &'c Context, module: &mut Module<'c>) {
    let candidates = find_candidates(module);
    for candidate in candidates {
        rewrite_one(context, module.body(), candidate);
    }
}

/// Which of the two shapes `match_candidate` found, and the extra bit each
/// one needs `rewrite_one` to know beyond `Candidate::producer` itself. `Copy`
/// (holds nothing but a `usize`) so `rewrite_one` can match on `candidate.
/// strategy` more than once without fighting the borrow checker over a field
/// of a struct that also holds non-`Copy` op/value handles.
#[derive(Clone, Copy)]
enum Strategy<'c, 'a> {
    /// `linalg.generic` (`linalg.transpose`/plain elementwise ops build this
    /// shape -- `match_candidate`'s own doc comment on this variant's match
    /// site has the full story) -- a fresh computation, needs a real
    /// destination buffer, no re-seed: whatever `outs` holds is already
    /// provably don't-care, either because the region never reads it at
    /// all, or because it was seeded from `tensor.empty()` in the first
    /// place. `redirect_op` is `producer` itself here (`rewrite_one`
    /// redirects `outs_index` directly on it).
    ///
    /// `linalg.matmul` (the *named* op, `build_matmul_no_seed`'s own doc
    /// comment: a real `linalg.fill` seed, not the old no-seed trick) needs
    /// the *same* real destination, but can't have its own `outs` redirected
    /// directly the way `linalg.generic` can -- a real named-op reduction
    /// genuinely *reads* its accumulator, so redirecting `producer`'s own
    /// `outs` to the struct's fresh, uninitialized field storage (skipping
    /// the zero-fill it depended on) would silently corrupt the reduction
    /// with garbage -- confirmed the hard way, in this exact rewrite's own
    /// earlier history (see `match_candidate`'s own doc comment). Instead,
    /// `redirect_op` here is the *`linalg.fill`* feeding `producer`'s own
    /// `outs` -- redirecting *its* `outs` is exactly as safe as the plain
    /// `linalg.generic` case above (`linalg.fill` unconditionally discards
    /// whatever was there, `tensor.empty()`-seeded or not), and `producer`'s
    /// own operand needs no change at all: it already points at `linalg.
    /// fill`'s own result *value*, whose identity doesn't change just
    /// because *its own* inputs did.
    Overwrite {
        redirect_op: OperationRef<'c, 'a>,
        outs_index: usize,
    },
    /// `bufferization.to_tensor`, feeding straight from `load_native_shape_
    /// field`'s own emission shape -- not a computation at all, `value` is an
    /// unmodified read of some *other* struct's own tensor field. No fresh
    /// destination, no linalg op to redirect: `rewrite_one` retains `Candidate
    /// ::src_ptr` and reuses it directly as the new destination pointer.
    Passthrough,
}

/// One matched occurrence of the pattern -- every op in the chain, already
/// verified to exist and to have the right shape, plus everything the
/// rewrite needs to build the replacement.
struct Candidate<'c, 'a> {
    /// The op that produced the stored value -- `linalg.generic` for
    /// `Strategy::Overwrite` (the op `rewrite_one` redirects `outs` on), or
    /// the `bufferization.to_tensor` itself for
    /// `Strategy::Passthrough` (used there only as an anchor: the position
    /// everything new gets inserted before). Always has exactly one result,
    /// checked in `match_candidate`.
    producer: OperationRef<'c, 'a>,
    strategy: Strategy<'c, 'a>,
    /// The pointer `memcpy` copies *from* -- for `Strategy::Passthrough`,
    /// this already points at a real, live `cleave_alloc_rc`'d payload (`load
    /// _native_shape_field`'s own read never copies -- see that function's
    /// own doc comment), so `rewrite_one` reuses it directly as the new
    /// destination pointer instead of building anything fresh, retaining it
    /// once to reflect the new shared owner. Unused by the other two
    /// strategies (they build a brand new destination instead).
    src_ptr: Value<'c, 'a>,
    /// Neutered (its own size operand zeroed), not erased -- see `rewrite_
    /// one`'s own doc comment on why.
    memcpy: OperationRef<'c, 'a>,
    /// The *existing* `call @cleave_alloc_rc(...)`'s own result -- **not**
    /// relocated (an earlier version of this rewrite tried `move_before`
    /// on the existing alloc/size chain directly and hit real, reproducible
    /// dominance verification failures on the actual `examples/mnist-
    /// interop` kernel, absent from every hand-written test: `--inline`
    /// canonicalizes/CSEs identically-shaped size computations across
    /// *independent* struct-field stores, so the "existing" chain for one
    /// candidate can genuinely be shared with another candidate located
    /// elsewhere -- moving it breaks dominance for whichever use ends up
    /// on the wrong side of the move). Kept only so every other, already-
    /// existing use of it (the neutered memcpy's own dest operand, and the
    /// struct field's own descriptor-build sequence downstream) can be
    /// redirected, by value, to the *new* pointer this rewrite builds
    /// fresh instead -- see `rewrite_one`'s own doc comment.
    dest_ptr: Value<'c, 'a>,
    /// The old `call @cleave_alloc_rc(...)` itself -- kept so `rewrite_one`
    /// can (a) zero *its own* size operand (`set_operand`, no relocation,
    /// so none of the CSE/dominance risk `dest_ptr`'s own doc comment
    /// describes applies here) and (b) genuinely `cleave_release` it right
    /// afterward. Both found necessary the hard way, in sequence: once
    /// `dest_ptr` (its result) has zero remaining uses (every one of them
    /// redirected to the fresh pointer), the call *itself* still executes
    /// at runtime if left alone -- a real, measured regression (`examples/
    /// digits-interop`, 161ms -> 206ms) traced to a genuine, permanent
    /// per-rewritten-store memory leak (a real allocation, its own pointer
    /// orphaned, never reaching `cleave_release`). Zeroing the size alone
    /// only *bounds* that leak (a tiny, near-free block still allocated and
    /// still never freed); the `cleave_release` call closes it completely
    /// -- confirmed directly on `examples/mnist-interop`'s own real kernel
    /// (every one of its 9 real struct-field stores takes this exact path,
    /// zero taking the fast/relocated one): resident memory measured flat
    /// end to end, no drift, over a full real training epoch.
    alloc_call: OperationRef<'c, 'a>,
    /// `Some((size_zero, size_gep, size_ptrtoint))` when the whole chain is
    /// exclusively this store's own -- `rewrite_one` relocates it directly
    /// in that case, no fresh allocation, no leak, no extra runtime cost at
    /// all. `None` (the CSE-shared case) falls back to `build_fresh_alloc`
    /// plus neutering the old call -- see `Candidate::alloc_call`'s own doc
    /// comment.
    private_size_chain: Option<(OperationRef<'c, 'a>, OperationRef<'c, 'a>, OperationRef<'c, 'a>)>,
    elem_type: Type<'c>,
    dims: Vec<i64>,
}

fn op_name_is<'c, 'a>(op: OperationRef<'c, 'a>, name: &str) -> bool {
    op.name().as_string_ref().as_str() == Ok(name)
}

/// The operation that produced `value`, if `value` is an operation result
/// at all (never a block argument -- a function parameter can't be the
/// tensor this pattern is about, since `store_native_shape_field`'s own
/// value always comes from a local computation).
fn defining_op<'c, 'a>(value: Value<'c, 'a>) -> Option<OperationRef<'c, 'a>> {
    OperationResult::try_from(value).ok().map(|r| r.owner())
}

fn find_candidates<'c, 'a>(module: &'a Module<'c>) -> Vec<Candidate<'c, 'a>> {
    let mut memcpy_ops = Vec::new();
    collect_ops_named(module.body(), "llvm.intr.memcpy", &mut memcpy_ops);

    let mut candidates = Vec::new();
    for memcpy in memcpy_ops {
        if let Some(candidate) = match_candidate(module, memcpy) {
            candidates.push(candidate);
        }
    }
    candidates
}

fn collect_ops_named<'c, 'a>(
    block: melior::ir::BlockRef<'c, 'a>,
    name: &str,
    out: &mut Vec<OperationRef<'c, 'a>>,
) {
    let mut next = block.first_operation();
    while let Some(op) = next {
        if op_name_is(op, name) {
            out.push(op);
        }
        for region in op.regions() {
            let mut next_block = region.first_block();
            while let Some(b) = next_block {
                collect_ops_named(b, name, out);
                next_block = b.next_in_region();
            }
        }
        next = op.next_in_block();
    }
}

fn match_candidate<'c, 'a>(
    module: &'a Module<'c>,
    memcpy: OperationRef<'c, 'a>,
) -> Option<Candidate<'c, 'a>> {
    if memcpy.operand_count() < 3 {
        return None;
    }
    let dest_ptr = memcpy.operand(0).ok()?;
    let src_ptr = memcpy.operand(1).ok()?;

    let alloc_call = defining_op(dest_ptr)?;
    if !op_name_is(alloc_call, "func.call") {
        return None;
    }
    let callee = alloc_call.attribute("callee").ok()?;
    if callee.to_string() != "@cleave_alloc_rc" {
        return None;
    }
    let size_ptrtoint = defining_op(alloc_call.operand(0).ok()?)?;
    if !op_name_is(size_ptrtoint, "llvm.ptrtoint") {
        return None;
    }
    let size_gep = defining_op(size_ptrtoint.operand(0).ok()?)?;
    if !op_name_is(size_gep, "llvm.getelementptr") {
        return None;
    }
    let size_zero = defining_op(size_gep.operand(0).ok()?)?;
    if !op_name_is(size_zero, "llvm.mlir.zero") {
        return None;
    }
    // Whether the whole size/alloc chain is *exclusively* this store's own
    // -- if so, `rewrite_one` can just relocate it (`move_before`, zero
    // extra cost, zero leak) instead of building a fresh one and leaving
    // the old allocation call to still execute, orphaned (`Candidate::
    // alloc_call`'s own doc comment has the real, measured leak this
    // avoids in the common case). `--inline` can genuinely CSE the same
    // size computation across *independent* struct-field stores of the
    // same shape (confirmed directly, `examples/mnist-interop`'s own real
    // kernel) -- when that's happened, any one of these has more than the
    // one, single expected use, and relocating would break dominance for
    // whichever other use ends up on the wrong side of the move.
    let alloc_chain_is_private = count_uses(module.body(), size_zero.result(0).ok()?.into()) == 1
        && count_uses(module.body(), size_gep.result(0).ok()?.into()) == 1
        && count_uses(module.body(), size_ptrtoint.result(0).ok()?.into()) == 1;

    let inttoptr = defining_op(src_ptr)?;
    if !op_name_is(inttoptr, "llvm.inttoptr") {
        return None;
    }
    let index_cast = defining_op(inttoptr.operand(0).ok()?)?;
    if !op_name_is(index_cast, "arith.index_cast") {
        return None;
    }
    let extract_ptr = defining_op(index_cast.operand(0).ok()?)?;
    if !op_name_is(extract_ptr, "memref.extract_aligned_pointer_as_index") {
        return None;
    }
    // What `extract_ptr` operates on tells apart the two shapes.
    // `Overwrite` goes through `bufferization.to_buffer` -- a fresh
    // `tensor<...>`-typed computation genuinely needs bufferizing to get a
    // pointer at all. `Strategy::Passthrough` does
    // *not*: `store_native_shape_field` always builds `to_buffer(value)`
    // unconditionally (`mlir_lower.rs`'s own code, checked directly, has no
    // special case for this at all) -- but when `value` is itself `load_
    // native_shape_field`'s own `to_tensor(cast(load(...)))` result, `--
    // inline`'s own post-inline cleanup (confirmed directly: this project
    // runs no separate `--canonicalize` stage at all, so this fold is
    // coming from the inliner's own simplification pass, not from anything
    // this rewrite or `pipeline.rs` added on purpose) folds the trivial `to
    // _buffer(to_tensor(x)) -> x` round trip away before this rewrite ever
    // runs -- confirmed directly, not assumed, the hard way: an earlier
    // version of this matcher looked for `bufferization.to_tensor` as the
    // producer feeding a `to_buffer` exactly the way `Overwrite`'s own
    // shape works, and it silently never fired at all on this exact,
    // structurally real case (`Sgd`'s own state passthrough, `stdlib/optim
    // /optim.cleave`) -- `extract_ptr`'s own operand traced straight to the
    // `unrealized_conversion_cast` underneath the vanished `to_tensor`,
    // with no `to_buffer` anywhere in between to match against.
    let extract_src = defining_op(extract_ptr.operand(0).ok()?)?;

    let (strategy, producer, elem_type, dims) = if op_name_is(extract_src, "bufferization.to_buffer")
    {
        let to_buffer = extract_src;
        let value = to_buffer.operand(0).ok()?;
        let producer = defining_op(value)?;
        let result = OperationResult::try_from(value).ok()?;
        if result.result_number() != 0 || producer.result_count() != 1 {
            return None;
        }

        // `linalg.generic` (`linalg.matmul`/`linalg.transpose` both build
        // this now -- `build_matmul_no_seed`/`build_transpose_no_seed`'s
        // own doc comments -- as does the common elementwise shape,
        // `--convert-elementwise-to-linalg`'s own output for `Ring::sub`/
        // `Scale::scale` and friends): safe whenever its own `outs` block
        // argument is either
        //
        // - *provably unused* inside the region body at all (`transpose`,
        //   plain elementwise ops -- a pure overwrite, any seed value is
        //   fine, don't-care by construction), or
        // - referenced, but only ever fed by `tensor.empty()` (`matmul`) --
        //   `tensor.empty()` is MLIR's own "genuinely uninitialized, don't-
        //   care" placeholder; a producer that seeds its *own* accumulator
        //   from one has *already* committed to tolerating arbitrary
        //   garbage there (confirmed directly for matmul specifically,
        //   `build_matmul_no_seed`'s own doc comment: its region only ever
        //   reads `outs`'s value on the one reduction-index branch it never
        //   *selects*) -- so redirecting *which* arbitrary garbage sits
        //   there (a struct's own freshly allocated field storage, instead
        //   of a separate scratch `tensor.empty()`) changes nothing about
        //   the result.
        //
        // A version of this matcher used to also accept `linalg.matmul`
        // (the *named* op) seeded by a real zero-splat constant, re-
        // establishing that same zero-fill explicitly (`linalg.fill`) on
        // the new destination -- dead for a while, since `build_matmul_no_
        // seed` stopped needing a seed at all (a real, measured `memset`
        // eliminated, ~9.7% of wall time on `examples/mnist-interop`), then
        // real again once `build_matmul_no_seed` itself moved to the real
        // named `linalg.matmul` op (`vector.contract`/`vector.outerproduct`
        // -- MLIR's own dedicated, FMA-friendly contraction vectorization
        // path, confirmed directly to need a real named op, `Vectorization.
        // cpp`'s own "Generic op is ignored" -- `pipeline.rs`'s own matmul-
        // lowering stage has the fuller story), which pays for a real
        // `linalg.fill` again. Re-added below, this time by redirecting the
        // *fill's* own `outs`, not the matmul's -- see `Strategy::
        // Overwrite`'s own doc comment for exactly why those two are not
        // interchangeable.
        let strategy = if op_name_is(producer, "linalg.generic") {
            let outs_index = producer.operand_count().checked_sub(1)?;
            let region = producer.region(0).ok()?;
            let body = region.first_block()?;
            let outs_arg: melior::ir::Value = body.argument(outs_index).ok()?.into();
            if count_uses(body, outs_arg) != 0 {
                let outs_operand = producer.operand(outs_index).ok()?;
                let seed = defining_op(outs_operand)?;
                if !op_name_is(seed, "tensor.empty") {
                    return None;
                }
            }
            Strategy::Overwrite {
                redirect_op: producer,
                outs_index,
            }
        } else if op_name_is(producer, "linalg.matmul") {
            let outs_index = producer.operand_count().checked_sub(1)?;
            let outs_operand = producer.operand(outs_index).ok()?;
            let seed = defining_op(outs_operand)?;
            if !op_name_is(seed, "linalg.fill") {
                return None;
            }
            // `linalg.fill ins(%cst) outs(%old_dest)` -- its own `outs` is
            // exactly as safe to redirect as `linalg.generic`'s above
            // (unconditionally discarded, `tensor.empty()`-seeded or not);
            // no need to check what feeds *it*.
            let fill_outs_index = seed.operand_count().checked_sub(1)?;
            Strategy::Overwrite {
                redirect_op: seed,
                outs_index: fill_outs_index,
            }
        } else {
            return None;
        };

        // `%value` must have exactly one real use in the whole module --
        // the `to_buffer` above -- for this rewrite to be sound at all (see
        // this module's own doc comment: redirecting the producer's own
        // destination changes nothing for a use count of 1, but a second,
        // unrelated reader of `%value` would silently start reading
        // through a `cleave_alloc_rc`'d buffer with no retain of its own, a
        // real, if narrow, correctness risk). No `getUses()`-equivalent is
        // exposed by melior's own bindings (checked directly) -- counted by
        // hand instead, the same way `--symbol-dce`-style analyses would.
        if count_uses(module.body(), value) != 1 {
            return None;
        }

        // Still a `tensor<...>` here -- this rewrite runs *before* One-Shot
        // Bufferize (`dps_rewrite.rs`'s own module doc comment) --
        // `MemRefType` doesn't apply yet; found directly (`MemRefType::
        // try_from` on a real `tensor<4x4xf32>` fails silently via `.ok()?`
        // , the exact reason an early version of this matcher never fired
        // on a genuinely matching case at all).
        let tensor_ty = RankedTensorType::try_from(result.r#type()).ok()?;
        let elem_type = tensor_ty.element();
        let rank = tensor_ty.rank();
        let mut dims = Vec::with_capacity(rank);
        for i in 0..rank {
            // Cleave tensors are always statically, fully shaped --
            // `Dynamic` should never occur; bail rather than guess if it
            // somehow does.
            match tensor_ty.dim_size(i).ok()? {
                melior::ir::r#type::DimSize::Static(size) => dims.push(size as i64),
                melior::ir::r#type::DimSize::Dynamic => return None,
            }
        }
        (strategy, producer, elem_type, dims)
    } else if op_name_is(extract_src, "builtin.unrealized_conversion_cast") {
        // `Strategy::Passthrough` -- not a computation at all. `extract_src`
        // is itself `load_native_shape_field`'s own memref-materializing
        // cast, feeding `extract_ptr` directly (the `to_tensor`/`to_buffer`
        // pair that would normally sit here already folded away -- this
        // block's own doc comment above has the story). `memref_val` is the
        // exact same live payload some *other* struct's field already
        // reads from -- `Candidate::src_ptr` (computed at the very top of
        // this function, from `memcpy`'s own operand(1)) already *is* the
        // raw pointer extracted from it, so nothing further needs building
        // here beyond confirming the shape and reading its type.
        let memref_val = extract_ptr.operand(0).ok()?;
        let read_descriptor = defining_op(extract_src.operand(0).ok()?)?;
        if !op_name_is(read_descriptor, "llvm.load") {
            return None;
        }
        // Same "exactly one real use" safety posture as the other two
        // strategies -- see the doc comment on that same check above.
        if count_uses(module.body(), memref_val) != 1 {
            return None;
        }
        let memref_ty = MemRefType::try_from(memref_val.r#type()).ok()?;
        let elem_type = memref_ty.element();
        let rank = memref_ty.rank();
        let mut dims = Vec::with_capacity(rank);
        for i in 0..rank {
            match memref_ty.dim_size(i).ok()? {
                melior::ir::r#type::DimSize::Static(size) => dims.push(size as i64),
                melior::ir::r#type::DimSize::Dynamic => return None,
            }
        }
        (Strategy::Passthrough, extract_src, elem_type, dims)
    } else {
        return None;
    };

    let private_size_chain =
        alloc_chain_is_private.then_some((size_zero, size_gep, size_ptrtoint));

    Some(Candidate {
        producer,
        strategy,
        src_ptr,
        memcpy,
        dest_ptr,
        alloc_call,
        private_size_chain,
        elem_type,
        dims,
    })
}

/// Counts how many operand slots, anywhere in `block` (recursively, through
/// nested regions), refer to `target` -- melior exposes no `getUses()`
/// equivalent, so this walks by hand; acceptable here because it only ever
/// runs once per *candidate* (already a small, pre-filtered set), not once
/// per op in the module.
fn count_uses<'c, 'a>(block: melior::ir::BlockRef<'c, 'a>, target: Value<'c, 'a>) -> usize {
    let mut count = 0;
    let mut next = block.first_operation();
    while let Some(op) = next {
        for operand in op.operands() {
            if operand == target {
                count += 1;
            }
        }
        for region in op.regions() {
            let mut next_block = region.first_block();
            while let Some(b) = next_block {
                count += count_uses(b, target);
                next_block = b.next_in_region();
            }
        }
        next = op.next_in_block();
    }
    count
}

/// Rewrites every operand slot, anywhere in `block` (recursively, through
/// nested regions), that refers to `old` so it refers to `new` instead --
/// melior exposes no `replaceAllUsesWith` equivalent (checked directly,
/// same as `count_uses`'s own doc comment), so this walks by hand too.
fn replace_all_uses<'c, 'a>(
    block: melior::ir::BlockRef<'c, 'a>,
    old: Value<'c, 'a>,
    new: Value<'c, 'a>,
) {
    let mut next = block.first_operation();
    while let Some(op) = next {
        for i in 0..op.operand_count() {
            if op.operand(i).ok() == Some(old) {
                as_mut(op).set_operand(i, new);
            }
        }
        for region in op.regions() {
            let mut next_block = region.first_block();
            while let Some(b) = next_block {
                replace_all_uses(b, old, new);
                next_block = b.next_in_region();
            }
        }
        next = op.next_in_block();
    }
}

/// `llvm.getelementptr <base>[<indices>]`, typed as `pointee_ty` -- the
/// standalone equivalent of `mlir_lower.rs::gep` (that one takes a private
/// `&LowerCtx` this module has no access to; duplicated rather than
/// refactoring a helper 9 call sites elsewhere already share).
fn gep<'c>(
    context: &'c Context,
    block: melior::ir::BlockRef<'c, 'c>,
    before: OperationRef<'c, 'c>,
    base: Value<'c, '_>,
    indices: &[i64],
    pointee_ty: Type<'c>,
) -> Value<'c, 'c> {
    let location = Location::unknown(context);
    let ptr_ty = llvm::r#type::pointer(context, 0);
    let raw: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
    let built = OperationBuilder::new("llvm.getelementptr", location)
        .add_attributes(&[
            (
                Identifier::new(context, "rawConstantIndices"),
                DenseI32ArrayAttribute::new(context, &raw).into(),
            ),
            (
                Identifier::new(context, "elem_type"),
                TypeAttribute::new(pointee_ty).into(),
            ),
        ])
        .add_operands(&[base])
        .add_results(&[ptr_ty])
        .build()
        .unwrap_or_else(|e| panic!("dps_rewrite: failed to build llvm.getelementptr: {e}"));
    block
        .insert_operation_before(before, built)
        .result(0)
        .unwrap()
        .into()
}

/// Builds a fresh `cleave_alloc_rc(sizeof(flat array of `elem_type` x
/// `dims`))` call, positioned right before `before` -- the standalone
/// equivalent of `mlir_lower.rs::alloc_llvm_value`/`llvm_type_size_bytes`,
/// duplicated for the same reason `gep` above is. `cleave_alloc_rc` is
/// already declared in this module (the candidate's own *existing* alloc
/// call already references it) -- no fresh declaration needed.
fn build_fresh_alloc<'c>(
    context: &'c Context,
    block: melior::ir::BlockRef<'c, 'c>,
    before: OperationRef<'c, 'c>,
    elem_type: Type<'c>,
    dims: &[i64],
) -> Value<'c, 'c> {
    let location = Location::unknown(context);
    let ptr_ty = llvm::r#type::pointer(context, 0);
    let total_elems: u32 = dims.iter().product::<i64>() as u32;
    let flat_array_ty = llvm::r#type::array(elem_type, total_elems);
    let null: Value = block
        .insert_operation_before(before, llvm::zero(ptr_ty, location))
        .result(0)
        .unwrap()
        .into();
    let one_past = gep(context, block, before, null, &[1], flat_array_ty);
    let i64_ty: Type = IntegerType::new(context, 64).into();
    let size: Value = block
        .insert_operation_before(
            before,
            OperationBuilder::new("llvm.ptrtoint", location)
                .add_operands(&[one_past])
                .add_results(&[i64_ty])
                .build()
                .unwrap_or_else(|e| panic!("dps_rewrite: failed to build llvm.ptrtoint: {e}")),
        )
        .result(0)
        .unwrap()
        .into();
    block
        .insert_operation_before(
            before,
            func::call(
                context,
                FlatSymbolRefAttribute::new(context, "cleave_alloc_rc"),
                &[size],
                &[ptr_ty],
                location,
            ),
        )
        .result(0)
        .unwrap()
        .into()
}

fn rewrite_one<'c>(
    context: &'c Context,
    module_body: melior::ir::BlockRef<'c, '_>,
    candidate: Candidate<'c, '_>,
) {
    let location = Location::unknown(context);
    let block = candidate
        .producer
        .block()
        .expect("candidate ops were just found live in a block");
    let i64_ty: Type = IntegerType::new(context, 64).into();

    // `relocated` mirrors `Candidate::private_size_chain`'s own doc comment
    // for `Strategy::Overwrite`: `true` only when the
    // *existing* alloc/size chain was relocated in place rather than left
    // behind for the tail below to neuter+release. `Strategy::Passthrough`
    // never relocates -- it needs no allocation at all, fresh or relocated
    // (`Candidate::src_ptr`'s own doc comment), so its old `alloc_call` is
    // always genuinely dead and must always take the neuter+release path,
    // exactly like the slow (fresh-build) path of the other two strategies.
    let (new_dest_ptr, relocated) = match candidate.strategy {
        Strategy::Passthrough => {
            // Not a fresh computation at all -- `src_ptr` already points at
            // a real, live `cleave_alloc_rc`'d payload (`load_native_shape_
            // field`'s own read never copies). Share it directly: one more
            // `cleave_retain` reflects the new second owner, no allocation,
            // no descriptor build, no linalg op to redirect at all.
            //
            // Inserted right before `memcpy`, *not* before `candidate.
            // producer` -- `producer` here is the `unrealized_conversion_
            // cast` that `src_ptr`'s own extraction chain (`extract_ptr`/
            // `index_cast`/`inttoptr`) is built *from*, so it comes
            // *earlier* in program order than `src_ptr` itself. Inserting a
            // use of `src_ptr` before its own definition is a real
            // dominance violation (confirmed directly, not assumed: this
            // exact mistake, tried first, failed verification -- "operand
            // #0 does not dominate this use"). `memcpy` already uses `src_
            // ptr` as its own second operand, so it's guaranteed to be
            // dominated by it.
            ensure_cleave_retain_declared(context, module_body);
            block.insert_operation_before(
                candidate.memcpy,
                func::call(
                    context,
                    FlatSymbolRefAttribute::new(context, "cleave_retain"),
                    &[candidate.src_ptr],
                    &[],
                    location,
                ),
            );
            (candidate.src_ptr, false)
        }
        Strategy::Overwrite {
            redirect_op,
            outs_index,
        } => {
            // Fast path: the size/alloc chain is exclusively this store's
            // own (`Candidate::private_size_chain`'s own doc comment) --
            // just relocate it, zero extra allocation, zero leak. Slow path
            // (CSE-shared): build a fresh one and neuter the old call's own
            // size instead, accepting a small, bounded, but real per-store
            // cost rather than risk the dominance failures relocating a
            // *shared* chain caused directly, reproducibly, on `examples/
            // mnist-interop`'s own real kernel (absent from every hand-
            // written test) -- `Candidate::alloc_call`'s own doc comment has
            // the measured regression neutering exists to fix.
            //
            // Every insertion/move below anchors on `redirect_op`, not
            // `candidate.producer` -- identical for the `linalg.generic`
            // case (`redirect_op == producer`), but genuinely different for
            // the `linalg.matmul`+`linalg.fill` case (`redirect_op` is the
            // *fill*, which sits *before* `producer` in program order --
            // see `Strategy::Overwrite`'s own doc comment). Anchoring on
            // `producer` there would insert the new value's own definition
            // *after* the fill whose operand it's about to replace, a
            // dominance violation. Anchoring on `redirect_op` is correct in
            // both cases: whatever is valid immediately before it is
            // trivially also valid before anything after it.
            let relocated = candidate.private_size_chain.is_some();
            let new_dest_ptr = if let Some((size_zero, size_gep, size_ptrtoint)) =
                candidate.private_size_chain
            {
                as_mut(size_zero).move_before(redirect_op);
                as_mut(size_gep).move_before(redirect_op);
                as_mut(size_ptrtoint).move_before(redirect_op);
                as_mut(candidate.alloc_call).move_before(redirect_op);
                candidate.dest_ptr
            } else {
                build_fresh_alloc(context, block, redirect_op, candidate.elem_type, &candidate.dims)
            };

            // Build a real `memref<dims x elem>` view of `new_dest_ptr` --
            // the same hand-built-descriptor-plus-`unrealized_conversion_
            // cast` trick `load_native_shape_field` already relies on, in
            // reverse: there, struct bits become a memref to read; here, a
            // raw pointer becomes a memref to write into. Identity layout,
            // no strides needed -- `new_dest_ptr` is a brand new, densely
            // packed allocation, never aliased or reshaped.
            let rank = candidate.dims.len();
            let descriptor_ty = memref_descriptor_llvm_type(context, rank);
            let zero_i64 = block.insert_operation_before(
                redirect_op,
                arith::constant(context, IntegerAttribute::new(i64_ty, 0).into(), location),
            );
            let poison =
                block.insert_operation_before(redirect_op, llvm::poison(descriptor_ty, location));
            let mut descriptor_val: Value = poison.result(0).unwrap().into();
            for pos in [0i64, 1] {
                let inserted = block.insert_operation_before(
                    redirect_op,
                    llvm::insert_value(
                        context,
                        descriptor_val,
                        DenseI64ArrayAttribute::new(context, &[pos]),
                        new_dest_ptr,
                        location,
                    ),
                );
                descriptor_val = inserted.result(0).unwrap().into();
            }
            let zero_offset: Value = zero_i64.result(0).unwrap().into();
            let inserted = block.insert_operation_before(
                redirect_op,
                llvm::insert_value(
                    context,
                    descriptor_val,
                    DenseI64ArrayAttribute::new(context, &[2]),
                    zero_offset,
                    location,
                ),
            );
            descriptor_val = inserted.result(0).unwrap().into();
            // Row-major sizes/strides -- cleave's tensors are always
            // statically, fully shaped, matching `store_native_shape_field`
            // 's own identical descriptor-build (mirrored here, not
            // reinvented).
            let mut stride = 1i64;
            let mut strides = vec![0i64; rank];
            for i in (0..rank).rev() {
                strides[i] = stride;
                stride *= candidate.dims[i];
            }
            for i in 0..rank {
                let size_const = block.insert_operation_before(
                    redirect_op,
                    arith::constant(
                        context,
                        IntegerAttribute::new(i64_ty, candidate.dims[i]).into(),
                        location,
                    ),
                );
                let inserted = block.insert_operation_before(
                    redirect_op,
                    llvm::insert_value(
                        context,
                        descriptor_val,
                        DenseI64ArrayAttribute::new(context, &[3, i as i64]),
                        size_const.result(0).unwrap().into(),
                        location,
                    ),
                );
                descriptor_val = inserted.result(0).unwrap().into();
                let stride_const = block.insert_operation_before(
                    redirect_op,
                    arith::constant(
                        context,
                        IntegerAttribute::new(i64_ty, strides[i]).into(),
                        location,
                    ),
                );
                let inserted = block.insert_operation_before(
                    redirect_op,
                    llvm::insert_value(
                        context,
                        descriptor_val,
                        DenseI64ArrayAttribute::new(context, &[4, i as i64]),
                        stride_const.result(0).unwrap().into(),
                        location,
                    ),
                );
                descriptor_val = inserted.result(0).unwrap().into();
            }

            let memref_ty: Type =
                MemRefType::new(candidate.elem_type, &candidate.dims, None, None).into();
            let cast = block.insert_operation_before(
                redirect_op,
                OperationBuilder::new("builtin.unrealized_conversion_cast", location)
                    .add_operands(&[descriptor_val])
                    .add_results(&[memref_ty])
                    .build()
                    .unwrap_or_else(|e| {
                        panic!("dps_rewrite: failed to build unrealized_conversion_cast: {e}")
                    }),
            );
            let dest_memref: Value = cast.result(0).unwrap().into();

            // `candidate.producer`'s own result type here, deliberately --
            // not `redirect_op`'s: for the `linalg.fill` case they're the
            // same shape either way (fill's own result feeds straight into
            // `producer`'s `outs`), but `producer`'s is the type this whole
            // rewrite is ultimately about (`candidate.elem_type`/`dims` were
            // already derived from it in `match_candidate`, for the same
            // reason).
            let tensor_ty = candidate.producer.result(0).unwrap().r#type();
            let restrict = Attribute::parse(context, "unit")
                .unwrap_or_else(|| panic!("dps_rewrite: failed to parse `unit` attribute"));
            let writable = Attribute::parse(context, "unit")
                .unwrap_or_else(|| panic!("dps_rewrite: failed to parse `unit` attribute"));
            let to_tensor = block.insert_operation_before(
                redirect_op,
                OperationBuilder::new("bufferization.to_tensor", location)
                    .add_operands(&[dest_memref])
                    .add_results(&[tensor_ty])
                    .add_attributes(&[
                        (Identifier::new(context, "restrict"), restrict),
                        (Identifier::new(context, "writable"), writable),
                    ])
                    .build()
                    .unwrap_or_else(|e| {
                        panic!("dps_rewrite: failed to build bufferization.to_tensor: {e}")
                    }),
            );
            let dest_tensor: Value = to_tensor.result(0).unwrap().into();

            // No re-seed needed for `linalg.generic` (`redirect_op ==
            // producer` there: whatever `outs` held before -- unused
            // entirely, or `tensor.empty()`'s own genuine garbage --
            // redirecting it to `dest_tensor` is exactly as safe as the
            // value it replaces). For `linalg.matmul`, `redirect_op` is the
            // `linalg.fill` instead, and this redirects *its* `outs` --
            // `producer` (the matmul) needs no operand change at all: it
            // already reads `redirect_op`'s own result value, whose
            // identity is unchanged by what now feeds *it*.
            as_mut(redirect_op).set_operand(outs_index, dest_tensor);
            (new_dest_ptr, relocated)
        }
    };

    // The whole copy tail is dead weight now -- `%value` (the producer's
    // own result) still exists and is still what the struct's field
    // descriptor conceptually holds, but nothing *reads* it through `to_
    // buffer` for any reason that matters any more.
    //
    // Not actually *erased*, on two separate findings, each real, neither
    // assumed:
    //
    // 1. `OperationMutLike::remove_from_parent` (melior 0.27.4) segfaults
    //    on this exact, real op -- confirmed directly, isolated to a single
    //    `remove_from_parent()` call on `memcpy` alone, nothing else
    //    touched (`STATUS_ACCESS_VIOLATION`, reproducible every time on
    //    this toolchain). A real melior/MLIR-C-API gap, not something this
    //    rewrite can safely work around by erasing anyway.
    // 2. The natural fallback -- leave the chain in place and let One-Shot
    //    Bufferize itself recognize `%value` as equivalent to `dest_tensor`
    //    (the same buffer, no real copy needed) -- was *tried* and
    //    disproved directly (`cleave/tests/dps_rewrite.rs`'s own `the_copy_
    //    becomes_a_same_address_no_op...` test, first written expecting
    //    this and failing): bufferization's own equivalence analysis does
    //    *not* recognize the roundtrip through a second `to_buffer` as
    //    trivial here, and still allocates a genuinely separate scratch
    //    buffer, meaning the copy would still be a real, full-size memcpy
    //    at runtime if left completely alone.
    //
    // The fix that's actually load-bearing: force the memcpy's own *size*
    // operand to a compile-time zero. Whatever buffer `to_buffer` resolves
    // to (the same one, or a fresh, now-pointlessly-allocated scratch one),
    // the byte-for-byte copy itself -- the dominant real cost this whole
    // rewrite exists to remove (VTune, `examples/mnist-interop`: ~38% of
    // wall time in unresolved `memcpy`-shaped frames) -- genuinely does not
    // happen: a zero-length `memcpy` touches no memory, regardless of its
    // own pointer operands. `dest_ptr`'s own descriptor build (already in
    // the IR, untouched) still correctly describes where the real data
    // lives, since that *is* where the producer now writes it directly.
    let zero_size = block.insert_operation_before(
        candidate.producer,
        arith::constant(context, IntegerAttribute::new(i64_ty, 0).into(), location),
    );
    let zero_size: Value = zero_size.result(0).unwrap().into();
    as_mut(candidate.memcpy).set_operand(2, zero_size);
    // The old `cleave_alloc_rc` call itself would still execute (and its
    // result, orphaned, never reach `cleave_release`, a real leak found the
    // hard way -- `Candidate::alloc_call`'s own doc comment) *if* it were a
    // genuinely separate call left behind -- only true on the slow
    // (fresh-build) path; on the fast (relocated) path, `candidate.
    // alloc_call` *is* the one and only allocation, already correctly
    // sized and already wired everywhere it needs to be, so it must be
    // left completely alone.
    if !relocated {
        as_mut(candidate.alloc_call).set_operand(0, zero_size);
    }

    // Redirect every other pre-existing use of the *old* alloc's own
    // pointer (the struct field's own descriptor-build sequence,
    // downstream, plus the now-neutered memcpy's own dest operand) to the
    // fresh one built above -- see `Candidate::dest_ptr`'s own doc comment
    // for why this is a `replace_all_uses`, not a relocation, of the old
    // chain.
    replace_all_uses(module_body, candidate.dest_ptr, new_dest_ptr);

    // Genuinely free the old (slow-path only, now zero-sized but still
    // real) allocation, rather than leave it bounded-but-leaked -- inserted
    // *after* `replace_all_uses` specifically so this new call's own
    // operand isn't itself caught and redirected by it (it must keep
    // pointing at the *old* pointer, the one actually being freed). Placed
    // right after `candidate.alloc_call` itself, not before `candidate.
    // producer` -- `alloc_call` is *not* relocated on this (slow) path, and
    // `store_native_shape_field`'s own emission order always puts it
    // *after* the producer (compute value -> to_buffer/extract chain ->
    // size/alloc chain -> memcpy -> descriptor build); inserting any use of
    // its own result any earlier than that is a real dominance violation
    // (confirmed directly, not assumed: `--inline`-heavy code, this exact
    // ordering mistake tried first, "operand #0 does not dominate this
    // use"). `dead_code_result` -- `cleave_release`'s own `bool` ("did this
    // actually free the block") return value, unused here on purpose: this
    // rewrite already knows, structurally, that nothing else can hold a
    // second reference to an allocation it just proved has zero remaining
    // uses in the whole module.
    if !relocated {
        ensure_cleave_release_declared(context, module_body);
        let dead_code_result: Value = block
            .insert_operation_after(
                candidate.alloc_call,
                func::call(
                    context,
                    FlatSymbolRefAttribute::new(context, "cleave_release"),
                    &[candidate.dest_ptr],
                    &[IntegerType::new(context, 1).into()],
                    location,
                ),
            )
            .result(0)
            .unwrap()
            .into();
        let _ = dead_code_result;
    }
}

/// Declares `cleave_retain` if this module doesn't already have it -- the
/// `Strategy::Passthrough` mirror of `ensure_cleave_release_declared` right
/// below (same reasoning: cheap to check, a real verification failure to
/// declare the same symbol twice).
fn ensure_cleave_retain_declared<'c>(context: &'c Context, module_body: melior::ir::BlockRef<'c, '_>) {
    let mut next = module_body.first_operation();
    while let Some(op) = next {
        if op_name_is(op, "func.func")
            && op
                .attribute("sym_name")
                .map(|a| a.to_string() == "\"cleave_retain\"")
                .unwrap_or(false)
        {
            return;
        }
        next = op.next_in_block();
    }
    let location = Location::unknown(context);
    let ptr_ty = llvm::r#type::pointer(context, 0);
    let decl = func::func(
        context,
        melior::ir::attribute::StringAttribute::new(context, "cleave_retain"),
        melior::ir::attribute::TypeAttribute::new(
            melior::ir::r#type::FunctionType::new(context, &[ptr_ty], &[]).into(),
        ),
        Region::new(),
        &[(
            Identifier::new(context, "sym_visibility"),
            melior::ir::attribute::StringAttribute::new(context, "private").into(),
        )],
        location,
    );
    module_body.append_operation(decl);
}

/// Declares `cleave_release` if this module doesn't already have it --
/// virtually always true already in practice (any real program with a
/// struct-field `Tensor` write, the precondition for this whole rewrite to
/// find anything at all, also reassigns/releases *some* struct somewhere),
/// but not something to assume without checking: declaring the *same*
/// symbol name twice is a real verification failure, not a harmless no-op.
fn ensure_cleave_release_declared<'c>(context: &'c Context, module_body: melior::ir::BlockRef<'c, '_>) {
    let mut next = module_body.first_operation();
    while let Some(op) = next {
        if op_name_is(op, "func.func")
            && op
                .attribute("sym_name")
                .map(|a| a.to_string() == "\"cleave_release\"")
                .unwrap_or(false)
        {
            return;
        }
        next = op.next_in_block();
    }
    let location = Location::unknown(context);
    let ptr_ty = llvm::r#type::pointer(context, 0);
    let bool_ty: Type = IntegerType::new(context, 1).into();
    let decl = func::func(
        context,
        melior::ir::attribute::StringAttribute::new(context, "cleave_release"),
        melior::ir::attribute::TypeAttribute::new(
            melior::ir::r#type::FunctionType::new(context, &[ptr_ty], &[bool_ty]).into(),
        ),
        Region::new(),
        &[(
            Identifier::new(context, "sym_visibility"),
            melior::ir::attribute::StringAttribute::new(context, "private").into(),
        )],
        location,
    );
    module_body.append_operation(decl);
}
