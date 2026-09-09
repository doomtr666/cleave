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
//! contrast with `dps_rewrite.rs`'s own considerably more involved
//! reasoning** for a structurally different problem (redirecting a
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
use melior::ir::operation::{Operation, OperationLike, OperationMutLike, OperationRef};
use melior::ir::{BlockLike, Module, RegionLike};

/// Mirrors `dps_rewrite.rs`'s own identical helper and identical doc-comment
/// reasoning -- `OperationMutLike` (needed for `remove_from_parent`) is
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
/// exactly the same failure mode `dps_rewrite.rs`'s own doc comment already
/// documents for `llvm.intr.memcpy`, and `unify_alloc.rs`'s own doc comment
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
