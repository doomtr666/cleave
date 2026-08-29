//! Gives cleave sole ownership of every tensor payload's own physical
//! memory — the "free-standing" half of unifying tensor and struct
//! allocation onto one mechanism (`dps_rewrite.rs`'s own module doc
//! comment already handles the other half: a tensor value that gets
//! written into a struct field, redirected to `cleave_alloc_rc` directly,
//! before bufferization ever runs). Tensor arithmetic that never touches a
//! struct field is, today, left entirely to MLIR's own bufferization-
//! inserted `memref.alloc`/`--buffer-deallocation-pipeline` — plain
//! `malloc`/`free`, a completely different allocator from every other heap
//! value cleave's own code ever produces (confirmed directly: a program
//! computing purely with free-standing tensors, zero structs, emits zero
//! `cleave_alloc_rc` calls and only ever calls `malloc`).
//!
//! **Runs at the very end of the pipeline**, once every `memref.alloc`/
//! `memref.dealloc` has already been lowered all the way to `llvm.call
//! @malloc`/`llvm.call @free` (`--convert-to-llvm`, `pipeline.rs`'s own
//! final lowering stage) — not before. Two real, independent reasons this
//! is the right point, not just a convenient one:
//!
//! - `llvm.call @malloc(size) -> ptr` and `llvm.call @cleave_alloc_rc(size)
//!   -> ptr` share the *exact* same signature — swapping the callee is a
//!   single in-place attribute edit, no operand/result rebuilding, no
//!   erasure of anything. Earlier in the pipeline, this same allocation is
//!   still a `memref.alloc()` with no operands at all (its own size is
//!   baked into its *type*, not an SSA value) — there would be nothing to
//!   "rename" a callee *of* at that stage, only a whole op to rebuild in
//!   place, with a different result type — exactly the kind of erase-and-
//!   replace this module exists to avoid.
//! - `melior`'s own `remove_from_parent` is confirmed unsafe to call on
//!   real ops from this pipeline (`dps_rewrite.rs`'s own doc comment, on
//!   `llvm.intr.memcpy`; checked again directly here, on `memref.alloc`/
//!   `memref.dealloc` specifically, since one op kind erasing safely never
//!   implies another does — same result: erasure "succeeds" at the call
//!   site itself, then crashes later, at module teardown). A callee rename
//!   needs no erasure at all, sidestepping the whole question for both
//!   halves of this rewrite.
//!
//! `llvm.call @free(ptr) -> ()` doesn't share `cleave_release`'s own
//! signature (`(ptr) -> i1`) — `cleave-rt::cleave_release_void` exists
//! purely to close that one gap, matching `free`'s own void-returning shape
//! exactly, so this rewrite stays a pure rename for both halves, not a
//! rename-plus-rebuild for one of them.
//!
//! **Why a blanket, whole-module rename is correct here, not just
//! convenient**: by the time this runs, cleave's own code has never
//! emitted a `malloc`/`free` call of its own — every struct, and every
//! struct-crossing tensor, already goes through `cleave_alloc_rc` directly
//! (`mlir_lower.rs::alloc_llvm_value`, `dps_rewrite.rs`) — so *every*
//! `llvm.call @malloc`/`@free` left in the module by this point is,
//! unconditionally, one MLIR's own bufferization pipeline inserted for a
//! tensor payload. No pattern-matching on shape, no per-site safety proof
//! needed the way `dps_rewrite.rs`'s own `Strategy` enum requires:
//! `--ownership-based-buffer-deallocation`'s own alias/liveness analysis
//! has *already* decided, ahead of this rewrite, exactly which allocation
//! belongs to which deallocation, and exactly when each one fires
//! (including across function boundaries). This rewrite only ever changes
//! *which allocator* physically backs a decision that analysis already
//! made correctly — it is not a second liveness analysis of its own, and
//! doesn't need to be one.

use melior::Context;
use melior::ir::attribute::{FlatSymbolRefAttribute, StringAttribute};
use melior::ir::operation::{OperationLike, OperationMutLike, OperationRef};
use melior::ir::{BlockLike, Module, RegionLike};

fn as_mut<'c, 'a>(op: OperationRef<'c, 'a>) -> melior::ir::operation::OperationRefMut<'c, 'a> {
    unsafe { melior::ir::operation::OperationRefMut::from_raw(op.to_raw()) }
}

fn op_name_is<'c, 'a>(op: OperationRef<'c, 'a>, name: &str) -> bool {
    op.name().as_string_ref().as_str() == Ok(name)
}

/// Renames every `llvm.call @malloc`/`llvm.call @free` in `module` to
/// `@cleave_alloc_rc`/`@cleave_release_void` — see this module's own doc
/// comment for why this is sound, general, and needs no erasure of
/// anything.
pub fn unify_tensor_allocations<'c>(context: &'c Context, module: &mut Module<'c>) {
    retarget_calls(context, module.body(), "malloc", "cleave_alloc_rc");
    retarget_calls(context, module.body(), "free", "cleave_release_void");
}

/// Renames every `llvm.call @old(...)` in `block` (recursively, through
/// nested regions) to `llvm.call @new(...)`, then makes sure `@new` is
/// actually declared: if some *other* part of the program already
/// declares it (the overwhelmingly common case — any real program with a
/// struct already declares `cleave_alloc_rc`), the now-unreferenced `@old`
/// declaration is simply left behind, dead but harmless; only if `@new`
/// has no declaration anywhere yet does `@old`'s own declaration get
/// renamed in place to become it (identical signature by construction —
/// see this module's own doc comment).
fn retarget_calls<'c, 'a>(
    context: &'c Context,
    body: melior::ir::BlockRef<'c, 'a>,
    old: &str,
    new: &str,
) {
    let old_symbol = format!("@{old}");
    let new_already_declared = find_top_level_func_decl(body, new).is_some();

    let renamed_any = rename_call_sites(context, body, &old_symbol, new);
    if !renamed_any {
        // Nothing called `@old` at all in this program (a struct-only
        // program with no free-standing tensor arithmetic, say) — no
        // declaration bookkeeping needed either.
        return;
    }

    if !new_already_declared {
        if let Some(decl) = find_top_level_func_decl(body, old) {
            as_mut(decl).set_attribute(
                "sym_name",
                StringAttribute::new(context, new).into(),
            );
        }
    }
}

/// Renames every `llvm.call @old(...)`'s own callee to `@new` (recursively,
/// through nested regions) — returns whether at least one call site was
/// found and renamed.
fn rename_call_sites<'c, 'a>(
    context: &'c Context,
    block: melior::ir::BlockRef<'c, 'a>,
    old_symbol: &str,
    new: &str,
) -> bool {
    let mut renamed = false;
    let mut next = block.first_operation();
    while let Some(op) = next {
        if op_name_is(op, "llvm.call") {
            if let Ok(callee) = op.attribute("callee") {
                if callee.to_string() == old_symbol {
                    as_mut(op).set_attribute(
                        "callee",
                        FlatSymbolRefAttribute::new(context, new).into(),
                    );
                    renamed = true;
                }
            }
        }
        for region in op.regions() {
            let mut next_block = region.first_block();
            while let Some(b) = next_block {
                renamed |= rename_call_sites(context, b, old_symbol, new);
                next_block = b.next_in_region();
            }
        }
        next = op.next_in_block();
    }
    renamed
}

/// Finds a top-level `llvm.func @name` (a declaration or a definition —
/// this module only ever expects a bare declaration for `malloc`/`free`,
/// but doesn't assume it) directly in `body`, not recursing into any
/// function's own inner blocks — `llvm.func` ops are always direct
/// children of the module body, never nested.
fn find_top_level_func_decl<'c, 'a>(
    body: melior::ir::BlockRef<'c, 'a>,
    name: &str,
) -> Option<OperationRef<'c, 'a>> {
    let mut next = body.first_operation();
    while let Some(op) = next {
        if op_name_is(op, "llvm.func") {
            if let Ok(sym_name) = op.attribute("sym_name") {
                if sym_name.to_string() == format!("\"{name}\"") {
                    return Some(op);
                }
            }
        }
        next = op.next_in_block();
    }
    None
}
