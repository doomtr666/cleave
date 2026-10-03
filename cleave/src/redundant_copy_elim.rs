//! Eliminates a real, self-inflicted `memref.copy %x, %x` (source and
//! destination the identical SSA value) left behind by One-Shot Bufferize's
//! own materialization of a tiled `scf.forall`/`tensor.parallel_insert_slice`
//! write-back (`pipeline.rs`'s own matmul-tiling stage,
//! `transform.structured.tile_using_forall` + `--loop-invariant-subset-
//! hoisting`) -- found directly, on a minimal isolated probe reproducing
//! that exact schedule (an 8-row-tiled matmul + bias-add epilogue, the same
//! shape `examples/mnist-interop`'s own real kernel uses for every layer),
//! not assumed: after hoisting + `one-shot-bufferize`, a tile's own private
//! accumulator is correctly copied *back* into the shared output's own
//! subview (a real, necessary write, `memref.copy %alloc_6, %subview_5`),
//! immediately followed by a second, entirely redundant `memref.copy
//! %subview_5, %subview_5` -- copying that exact same subview into *itself*,
//! right after it was just written. Neither `--canonicalize` nor `--cse`
//! folds this away (confirmed directly against this exact toolchain, both
//! applied to the same probe) -- `memref.copy` apparently has no such fold
//! registered at all.
//!
//! **Unconditionally safe, no analysis needed at all -- a real, deliberate
//! contrast with the considerably more involved reasoning** a structurally
//! different problem needs (redirecting a
//! *computation's own destination*, which needs real precondition checks to
//! stay sound). A `memref.copy` from a memref to itself is a no-op by
//! definition, in every case, for every shape -- `source == destination`
//! (SSA value identity) is the *entire* correctness argument.
//!
//! **Must run *after* `--cse` specifically** (`pipeline.rs`'s own stage
//! ordering) -- without it, two structurally identical `memref.subview`s
//! feeding the same copy are two *different* SSA values, and this pass's
//! own identity check would miss them entirely; confirmed directly on the
//! same probe: zero self-copies matched until `--cse` ran first, merging the
//! duplicate `memref.subview` computations into one shared value.

use melior::Context;
use melior::ir::operation::{Operation, OperationBuilder, OperationLike, OperationMutLike, OperationRef};
use melior::ir::{BlockLike, Module, RegionLike, TypeLike, ValueLike};

/// `OperationMutLike` (needed for `remove_from_parent`) is
/// implemented for `OperationRefMut`, not the plain `OperationRef` a walk
/// returns; both wrap the same raw `MlirOperation` handle, this only widens
/// which methods are callable on it.
fn as_mut<'c, 'a>(op: OperationRef<'c, 'a>) -> melior::ir::operation::OperationRefMut<'c, 'a> {
    unsafe { melior::ir::operation::OperationRefMut::from_raw(op.to_raw()) }
}

/// Genuinely deletes `op` -- **not** a bare `remove_from_parent()` call left
/// to dangle, a real bug found and fixed the hard way here, not designed in
/// from the start: `mlirOperationRemoveFromParent`'s own documented contract
/// (`mlir_lower.rs::build_matmul_transpose_no_seed`'s own doc comment quotes
/// it directly) is "not destroyed, only unlinked" -- the caller owns it
/// afterward, and *must* either re-insert it (that function's own case) or
/// genuinely free it. Leaving it merely detached, neither, crashed for real
/// (`STATUS_STACK_BUFFER_OVERRUN`, a real MLIR-internal assertion --
/// `"expected that op has no uses"` -- hit later, in a subsequent pass, not
/// at the call site itself), confirmed directly on this exact toolchain,
/// building the real `examples/digits-interop` kernel -- not a hypothetical:
/// exactly the same failure mode first found on `llvm.intr.memcpy`, and `unify_alloc.rs`'s own doc comment
/// independently reconfirms for `memref.alloc`/`memref.dealloc` ("erasure
/// 'succeeds' at the call site itself, then crashes later, at module
/// teardown") -- a general melior/MLIR-C-API hazard across *every* op kind
/// tried so far, not specific to any one of them. The fix: detach, then
/// immediately reclaim real ownership (`Operation::from_raw`, the same
/// `unsafe` escape hatch `build_matmul_transpose_no_seed` already uses for
/// the opposite purpose, re-insertion) and `drop` it explicitly -- `Operation`
/// 's own `Drop` impl calls `mlirOperationDestroy`, the genuine free
/// `remove_from_parent` alone never performs.
fn erase<'c, 'a>(op: OperationRef<'c, 'a>) {
    as_mut(op).remove_from_parent();
    drop(unsafe { Operation::from_raw(op.to_raw()) });
}

/// Runs the elimination over every `memref.copy` in `module`, in place. Safe
/// to run on a module with none at all -- finds nothing, changes nothing.
pub fn eliminate_self_copies<'c>(_context: &'c Context, module: &mut Module<'c>) {
    let mut dead = Vec::new();
    collect_self_copies(module.body(), &mut dead);
    for op in dead {
        erase(op);
    }
}

fn collect_self_copies<'c, 'a>(block: melior::ir::BlockRef<'c, 'a>, out: &mut Vec<OperationRef<'c, 'a>>) {
    let mut next = block.first_operation();
    while let Some(op) = next {
        if op.name().as_string_ref().as_str() == Ok("memref.copy") {
            // `memref.copy` is always exactly `(source, destination)`, two
            // operands, no results (`MemRefOps.td`'s own signature) -- a
            // genuine self-copy is simply operand(0) and operand(1) naming
            // the identical SSA value.
            if let (Ok(src), Ok(dst)) = (op.operand(0), op.operand(1)) {
                if src == dst {
                    out.push(op);
                }
            }
        }
        for region in op.regions() {
            let mut next_block = region.first_block();
            while let Some(b) = next_block {
                collect_self_copies(b, out);
                next_block = b.next_in_region();
            }
        }
        next = op.next_in_block();
    }
}

/// Turns every `linalg.copy` between memrefs of a *dynamic* size into the
/// `memref.copy` it is, before `--convert-linalg-to-affine-loops`
/// (`pipeline.rs`). Such a copy is the write-back of a partial tile: a
/// matmul whose column count isn't a multiple of the schedule's 16
/// (`matmul_vectorize.transform.mlir`, `structured.pad`'s `copy_back_op`),
/// the GPT's `128 -> 104` output layer. Its size is an `affine.min` of an
/// `scf.for` induction variable, which the affine pass rejects as a
/// dimension ("operand cannot be used as a dimension id"), failing the
/// whole compilation. `memref.copy` needs no loops of its own. Copies of a
/// static size, every one the schedule produced before this, are left as
/// they are.
pub fn lower_dynamic_copies<'c>(_context: &'c Context, module: &mut Module<'c>) {
    let mut copies = Vec::new();
    collect_dynamic_copies(module.body(), &mut copies);
    for op in copies {
        let (Ok(src), Ok(dst)) = (op.operand(0), op.operand(1)) else { continue };
        let block = op.block().expect("a linalg.copy has a parent block");
        let copy = OperationBuilder::new("memref.copy", op.location())
            .add_operands(&[src, dst])
            .build()
            .expect("failed to build memref.copy");
        block.insert_operation_before(op, copy);
        erase(op);
    }
}

fn collect_dynamic_copies<'c, 'a>(block: melior::ir::BlockRef<'c, 'a>, out: &mut Vec<OperationRef<'c, 'a>>) {
    let mut next = block.first_operation();
    while let Some(op) = next {
        if op.name().as_string_ref().as_str() == Ok("linalg.copy") && op.operand_count() == 2 && op.result_count() == 0 {
            let dynamic = (0..2).any(|i| {
                op.operand(i).is_ok_and(|v| {
                    let ty = v.r#type();
                    ty.is_mem_ref() && ty.to_string().contains('?')
                })
            });
            if dynamic {
                out.push(op);
            }
        }
        for region in op.regions() {
            let mut next_block = region.first_block();
            while let Some(b) = next_block {
                collect_dynamic_copies(b, out);
                next_block = b.next_in_region();
            }
        }
        next = op.next_in_block();
    }
}

/// Hands a call its final destination directly when the buffer it wrote its
/// result into was only copied there. After `buffer-results-to-out-params`
/// (`pipeline.rs`), a call whose result goes into a struct field or a tuple
/// element looks like
///
/// ```text
/// %out = memref.alloc() : memref<256x256xf32>
/// call @f(%a, %out)
/// ...
/// memref.copy %out, %field : memref<256x256xf32> to memref<256x256xf32>
/// ```
///
/// where `%field` is the field's storage (`mlir_lower.rs::
/// build_tensor_descriptor_value`, written by `materialize_in_destination`
/// and bufferized to that copy). Rewritten to `call @f(%a, %field)`, the copy
/// and the allocation gone: a training step's weight gradients and every
/// tensor a function hands back inside an aggregate are no longer copied.
///
/// Conservative, every condition local and checked: `%out` is a
/// `memref.alloc` used exactly twice, as an operand of one `func.call` and as
/// the copy's source, all three in one block in that order; `%field` has the
/// same type, is defined earlier in that block (so it dominates the call),
/// isn't an operand of the call and has no use before the copy. A field
/// buffer is written only by its materialization, so nothing between the call
/// and the copy can have observed it.
pub fn forward_out_param_copies<'c>(_context: &'c Context, module: &mut Module<'c>) {
    forward_dead_source_copies(module);
    let mut index = OpIndex::default();
    index_block(module.body(), &mut index);
    let mut rewrites = Vec::new();
    for &op in &index.copies {
        if let Some(rewrite) = match_out_param_copy(&index, op) {
            rewrites.push(rewrite);
        }
    }
    for (call, operand, dst, copy, alloc) in rewrites {
        as_mut(call).set_operand(operand, dst);
        erase(copy);
        erase(alloc);
    }
}

/// Drops the copy of a buffer that is never used again into a fresh one: the
/// fresh buffer becomes the old one. `Tensor(data: buf)` copies `buf` before
/// marking the tensor `restrict` (`mlir_lower.rs::lower_tagged_struct_
/// construct`), since `buf` could in general still be written afterwards;
/// when it isn't (an array filled by loops and handed over, as the causal
/// attention's per-head blocks and gradients are), the copy is pure cost.
///
/// `memref.copy %a, %b` with `%a` and `%b` both `memref.alloc`s of the same
/// type in the copy's block, `%a` with no use after the copy (nested uses
/// counted at their enclosing operation in that block), `%b` with none
/// before it: every use of `%b` becomes `%a`, the copy and `%b` go. `%a` is
/// defined earlier in the block, so it dominates everything `%b` did.
pub fn forward_dead_source_copies(module: &mut Module) {
    loop {
        let mut index = OpIndex::default();
        index_block(module.body(), &mut index);
        let mut done = std::collections::HashSet::new();
        let mut rewrites = Vec::new();
        let mut hoists = Vec::new();
        for &copy in &index.copies {
            if let Some((a, b, b_alloc)) = match_dead_source_copy(&index, copy) {
                // One rewrite per buffer per round: a later one could involve
                // a value an earlier one replaces.
                if done.insert(value_key(a)) && done.insert(value_key(b)) {
                    rewrites.push((copy, a, b, b_alloc));
                }
            } else if let Some((a_alloc, a, b, slice)) = match_hoistable_destination(&index, copy) {
                if done.insert(value_key(a)) && done.insert(value_key(b)) {
                    hoists.push((copy, a_alloc, a, b, slice));
                }
            }
        }
        if rewrites.is_empty() && hoists.is_empty() {
            return;
        }
        for (copy, a, b, b_alloc) in rewrites {
            for &(user, operand) in index.uses.get(&value_key(b)).map(Vec::as_slice).unwrap_or(&[]) {
                if op_key(user) != op_key(copy) {
                    as_mut(user).set_operand(operand, a);
                }
            }
            erase(copy);
            erase(b_alloc);
        }
        for (copy, a_alloc, a, b, slice) in hoists {
            // The destination's definition, moved right after `%a`'s
            // allocation in its original order; then `%b` replaces `%a`.
            let anchor = a_alloc.next_in_block().expect("an allocation is followed by its uses");
            for op in slice {
                as_mut(op).move_before(anchor);
            }
            for &(user, operand) in index.uses.get(&value_key(a)).map(Vec::as_slice).unwrap_or(&[]) {
                if op_key(user) != op_key(copy) {
                    as_mut(user).set_operand(operand, b);
                }
            }
            erase(copy);
            erase(a_alloc);
        }
    }
}

#[allow(clippy::type_complexity)]
fn match_dead_source_copy<'c, 'a>(
    index: &OpIndex<'c, 'a>,
    copy: OperationRef<'c, 'a>,
) -> Option<(melior::ir::Value<'c, 'a>, melior::ir::Value<'c, 'a>, OperationRef<'c, 'a>)> {
    let (a, b) = (copy.operand(0).ok()?, copy.operand(1).ok()?);
    if a == b || a.r#type() != b.r#type() {
        return None;
    }
    let &(block, copy_pos) = index.position.get(&op_key(copy))?;
    let alloc_of = |v: melior::ir::Value<'c, 'a>| -> Option<(OperationRef<'c, 'a>, usize)> {
        let op = melior::ir::operation::OperationResult::try_from(v).ok()?.owner();
        if op.name().as_string_ref().as_str() != Ok("memref.alloc") || op.operand_count() != 0 {
            return None;
        }
        let &(b, pos) = index.position.get(&op_key(op))?;
        (b == block).then_some((op, pos))
    };
    alloc_of(a)?;
    let (b_alloc, _) = alloc_of(b)?;
    let position_in_block = |user: OperationRef<'c, 'a>| -> Option<usize> {
        let ancestor = ancestor_in_block(user, block)?;
        index.position.get(&op_key(ancestor)).map(|&(_, pos)| pos)
    };
    for &(user, _) in index.uses.get(&value_key(a))? {
        if op_key(user) != op_key(copy) && position_in_block(user)? >= copy_pos {
            return None;
        }
    }
    for &(user, _) in index.uses.get(&value_key(b))? {
        if op_key(user) != op_key(copy) && position_in_block(user)? <= copy_pos {
            return None;
        }
    }
    Some((a, b, b_alloc))
}

#[derive(Default)]
struct OpIndex<'c, 'a> {
    /// Each operation's block and position in it.
    position: std::collections::HashMap<usize, (usize, usize)>,
    /// Each value's uses: the using operation and the operand index.
    uses: std::collections::HashMap<usize, Vec<(OperationRef<'c, 'a>, usize)>>,
    copies: Vec<OperationRef<'c, 'a>>,
}

fn op_key(op: OperationRef) -> usize {
    op.to_raw().ptr as usize
}

fn value_key<'c, 'a>(v: melior::ir::Value<'c, 'a>) -> usize {
    v.to_raw().ptr as usize
}

fn index_block<'c, 'a>(block: melior::ir::BlockRef<'c, 'a>, index: &mut OpIndex<'c, 'a>) {
    let block_key = block.to_raw().ptr as usize;
    let mut next = block.first_operation();
    let mut pos = 0;
    while let Some(op) = next {
        index.position.insert(op_key(op), (block_key, pos));
        for i in 0..op.operand_count() {
            if let Ok(v) = op.operand(i) {
                index.uses.entry(value_key(v)).or_default().push((op, i));
            }
        }
        if op.name().as_string_ref().as_str() == Ok("memref.copy") {
            index.copies.push(op);
        }
        for region in op.regions() {
            let mut next_block = region.first_block();
            while let Some(b) = next_block {
                index_block(b, index);
                next_block = b.next_in_region();
            }
        }
        pos += 1;
        next = op.next_in_block();
    }
}

/// The ancestor of `op` (itself included) directly in `block`, if any.
fn ancestor_in_block<'c, 'a>(mut op: OperationRef<'c, 'a>, block: usize) -> Option<OperationRef<'c, 'a>> {
    loop {
        let b = op.block()?;
        if b.to_raw().ptr as usize == block {
            return Some(op);
        }
        op = b.parent_operation()?;
    }
}

#[allow(clippy::type_complexity)]
fn match_out_param_copy<'c, 'a>(
    index: &OpIndex<'c, 'a>,
    copy: OperationRef<'c, 'a>,
) -> Option<(OperationRef<'c, 'a>, usize, melior::ir::Value<'c, 'a>, OperationRef<'c, 'a>, OperationRef<'c, 'a>)> {
    let (src, dst) = (copy.operand(0).ok()?, copy.operand(1).ok()?);
    if src == dst || src.r#type() != dst.r#type() {
        return None;
    }
    let &(block, copy_pos) = index.position.get(&op_key(copy))?;
    let alloc = melior::ir::operation::OperationResult::try_from(src).ok()?.owner();
    if alloc.name().as_string_ref().as_str() != Ok("memref.alloc") {
        return None;
    }
    let &(alloc_block, alloc_pos) = index.position.get(&op_key(alloc))?;
    let src_uses = index.uses.get(&value_key(src))?;
    if alloc_block != block || src_uses.len() != 2 {
        return None;
    }
    let &(call, operand) = src_uses.iter().find(|(u, _)| op_key(*u) != op_key(copy))?;
    if call.name().as_string_ref().as_str() != Ok("func.call") {
        return None;
    }
    let &(call_block, call_pos) = index.position.get(&op_key(call))?;
    if call_block != block || !(alloc_pos < call_pos && call_pos < copy_pos) {
        return None;
    }
    // `dst` defined earlier in the block, by an operation.
    let def = melior::ir::operation::OperationResult::try_from(dst).ok()?.owner();
    let &(def_block, def_pos) = index.position.get(&op_key(def))?;
    if def_block != block || def_pos >= call_pos {
        return None;
    }
    // No use of `dst` before the copy, the call's own operands included.
    for &(user, _) in index.uses.get(&value_key(dst)).map(Vec::as_slice).unwrap_or(&[]) {
        if op_key(user) == op_key(copy) {
            continue;
        }
        let ancestor = ancestor_in_block(user, block)?;
        let &(_, pos) = index.position.get(&op_key(ancestor))?;
        if pos <= copy_pos {
            return None;
        }
    }
    Some((call, operand, dst, copy, alloc))
}

/// The other direction: `memref.copy %a, %b` where `%a` is a fresh
/// `memref.alloc` filled earlier and dead after the copy, and `%b` a
/// destination defined *later* (a tuple element's or struct field's storage,
/// `mlir_lower.rs::build_tensor_descriptor_value`, built right before its
/// materialization). When `%b`'s definition is a chain of side-effect-free
/// operations plus the field's `cleave_alloc_rc`, depending only on values
/// available before `%a`'s allocation, that chain moves up next to it and
/// `%a`'s writers write `%b` directly: `causal_attention_backward`'s `dq`,
/// `dk`, `dv` are filled in place in the returned tuple. Allocating the field
/// earlier changes nothing observable. Returns `%a`'s allocation, `%a`, `%b`
/// and the chain to move, in block order.
#[allow(clippy::type_complexity)]
fn match_hoistable_destination<'c, 'a>(
    index: &OpIndex<'c, 'a>,
    copy: OperationRef<'c, 'a>,
) -> Option<(
    OperationRef<'c, 'a>,
    melior::ir::Value<'c, 'a>,
    melior::ir::Value<'c, 'a>,
    Vec<OperationRef<'c, 'a>>,
)> {
    let (a, b) = (copy.operand(0).ok()?, copy.operand(1).ok()?);
    if a == b || a.r#type() != b.r#type() {
        return None;
    }
    let &(block, copy_pos) = index.position.get(&op_key(copy))?;
    let a_alloc = melior::ir::operation::OperationResult::try_from(a).ok()?.owner();
    if a_alloc.name().as_string_ref().as_str() != Ok("memref.alloc") || a_alloc.operand_count() != 0 {
        return None;
    }
    let &(a_block, a_pos) = index.position.get(&op_key(a_alloc))?;
    if a_block != block {
        return None;
    }
    let position_in_block = |user: OperationRef<'c, 'a>| -> Option<usize> {
        let ancestor = ancestor_in_block(user, block)?;
        index.position.get(&op_key(ancestor)).map(|&(_, pos)| pos)
    };
    for &(user, _) in index.uses.get(&value_key(a))? {
        if op_key(user) != op_key(copy) && position_in_block(user)? >= copy_pos {
            return None;
        }
    }
    let b_def = melior::ir::operation::OperationResult::try_from(b).ok()?.owner();
    let &(b_block, b_pos) = index.position.get(&op_key(b_def))?;
    if b_block != block || b_pos <= a_pos || b_pos >= copy_pos {
        return None;
    }
    for &(user, _) in index.uses.get(&value_key(b)).map(Vec::as_slice).unwrap_or(&[]) {
        if op_key(user) != op_key(copy) && position_in_block(user)? <= copy_pos {
            return None;
        }
    }
    // The backward slice of `%b` after `%a`'s allocation, every op movable;
    // operands defined before `%a`'s allocation, or block arguments of this
    // block or an enclosing one, are available at the new position already.
    let mut slice: Vec<(usize, OperationRef<'c, 'a>)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut work = vec![b_def];
    while let Some(op) = work.pop() {
        if !seen.insert(op_key(op)) {
            continue;
        }
        let &(op_block, pos) = index.position.get(&op_key(op))?;
        if op_block != block || pos <= a_pos {
            continue;
        }
        if !is_movable(op) {
            return None;
        }
        slice.push((pos, op));
        for i in 0..op.operand_count() {
            if let Ok(result) = melior::ir::operation::OperationResult::try_from(op.operand(i).ok()?) {
                work.push(result.owner());
            }
        }
    }
    // Nothing outside the slice may use its results before the copy: moving
    // them earlier must not change what an earlier operation sees.
    let slice_keys: std::collections::HashSet<usize> = slice.iter().map(|(_, op)| op_key(*op)).collect();
    for &(_, op) in &slice {
        for r in 0..op.result_count() {
            let v: melior::ir::Value = op.result(r).ok()?.into();
            for &(user, _) in index.uses.get(&value_key(v)).map(Vec::as_slice).unwrap_or(&[]) {
                if slice_keys.contains(&op_key(user)) || op_key(user) == op_key(copy) {
                    continue;
                }
                if position_in_block(user)? <= copy_pos {
                    return None;
                }
            }
        }
    }
    slice.sort_by_key(|(pos, _)| *pos);
    Some((a_alloc, a, b, slice.into_iter().map(|(_, op)| op).collect()))
}

/// What `match_hoistable_destination` may move earlier: no side effect but
/// allocating, no region.
fn is_movable(op: OperationRef) -> bool {
    if op.region_count() != 0 {
        return false;
    }
    match op.name().as_string_ref().as_str() {
        Ok(
            "llvm.getelementptr" | "llvm.ptrtoint" | "llvm.insertvalue" | "llvm.extractvalue" | "llvm.mlir.poison"
            | "llvm.mlir.undef" | "llvm.mlir.zero" | "llvm.mlir.constant" | "arith.constant"
            | "builtin.unrealized_conversion_cast",
        ) => true,
        Ok("func.call") => op
            .attribute("callee")
            .is_ok_and(|callee| callee.to_string() == "@cleave_alloc_rc"),
        _ => false,
    }
}
