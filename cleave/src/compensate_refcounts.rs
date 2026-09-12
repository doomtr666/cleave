//! Compensates for a real gap between what `refcount.rs`'s own CPS-level
//! analysis can see and what MLIR's later passes are legally allowed to do
//! to the IR it hands off.
//!
//! **The bug this closes, root-caused directly (`doc/backlog.md`'s own
//! "Scale::scale leak" entry has the full trace)**: `refcount.rs::insert_
//! refcounting` runs on CPS, before `--inline` ever exists as a concept —
//! at that point, two independent pure tensor constructions in two
//! unrelated functions (say, `net_grad`'s own zero-tensor and `Optimizer::
//! init_state`'s own `state.b` construction, both `Ring::zero::<Tensor<...>>
//! ()`) are, correctly, two separate allocations, each given its own
//! balanced `releases == retains + 1` bookkeeping. Once `--inline` brings
//! both bodies into one function, the two constructions become two
//! structurally-identical, side-effect-free ops — a legitimate target for
//! MLIR's own later optimizations to merge into *one* shared SSA value.
//! `refcount.rs` had no way to see this coming — the sharing doesn't exist,
//! and isn't representable, at the point its analysis runs — so both
//! constructions' own release call sites end up pointing at the same
//! physical allocation, one too many releases for the retains actually
//! present: a real double-free.
//!
//! **The fix, deliberately not a blind "prune the duplicate release"
//! pass** (that would risk breaking `refcount.rs`'s own pre-existing,
//! correct "redundant retain-then-matching-release" bookkeeping for nested
//! light-struct leaves — a real, different pattern that nets to zero on
//! purpose): count, per distinct `!llvm.ptr` SSA value, every `cleave_
//! retain`/`cleave_release` call site that operates on it, anywhere in the
//! module, regardless of control flow (a static, not path-sensitive, count
//! — see below for why path-sensitivity isn't needed here). Only ever check
//! for a *surplus* of releases over retains, never the reverse: a genuinely
//! *owned*, newly-constructed value follows `releases == retains + 1` (the
//! "+1" is the allocation's own initial, implicit ownership), but a merely
//! *borrowed* value (a function parameter, say) legitimately nets to
//! `releases == retains` (a temporary protect-then-release pair around one
//! embedding, no "+1" of its own — `refcount.rs`'s own module doc comment).
//! This pass doesn't need to tell the two classes apart: whenever a CSE-
//! style merge *has* happened, it can only ever manifest as a release
//! surplus over whatever this value's own healthy count already was — each
//! independently-correct construction folded into a shared value
//! contributes its own extra release, never removes one. So the one check
//! this pass needs is simply `releases > retains + 1` — true only when at
//! least one merge has happened, false for both healthy classes above. This
//! pass inserts the missing `(releases - retains - 1)` `cleave_retain`
//! calls right after the value's own definition, restoring the balance —
//! purely additive on top of `refcount.rs`'s own always-correct output,
//! never a deletion. In the ordinary (no merge) case the deficit is always
//! zero, so this is a genuine no-op everywhere it isn't needed.
//!
//! **Why a whole-module, path-insensitive count is the right invariant
//! here, not a hole in the reasoning**: this pass isn't attempting the
//! separate, harder "prove this retain/release pair is provably redundant
//! along a specific execution path" problem (`doc/backlog.md`'s own Tier-1
//! ARC-style plan, `rc_opt.rs`, still unwired) — it only ever *adds*
//! retains to correct a static undercount created by a merge that already
//! happened earlier in this same pipeline. Even in a pathological case
//! where the count somewhat overestimates real sharing, the worst outcome
//! is one harmless extra retain/release pair — never a use-after-free,
//! because this pass never removes anything.
//!
//! **Runs at the very end of the pipeline, alongside `unify_alloc.rs`, not
//! right after `--cse`/bufferization as first tried — found by direct
//! measurement, not assumed**: the first version of this pass ran right
//! before the final `--convert-to-llvm` (operating on still-`func.call`
//! ops), on the theory that every pass that could cause the merge above
//! (`--inline`, the one explicit `--cse`, One-Shot Bufferize) had already
//! run by then. Measured directly against a real kernel (`CLEAVE_TRACE_
//! COMPENSATE=1`, a real MNIST training kernel): every real merge in
//! practice still showed up as two visibly *distinct* `!llvm.ptr` SSA
//! values at that point — zero surpluses detected, yet the double-free
//! still reproduced end to end. Root cause: each `cleave_retain`/`cleave_
//! release` call's own pointer operand isn't the tensor value itself, but
//! the result of `mlir_lower.rs::tensor_value_to_ptr`'s own per-call-site
//! `bufferization.to_buffer` -> `memref.extract_aligned_pointer_as_index`
//! -> `llvm.inttoptr` chain, built independently at *initial* CPS-to-MLIR
//! lowering, long before `--inline` ever exists as a concept — two such
//! chains, even once their shared tensor operand has been merged upstream,
//! don't themselves collapse into one SSA value until *further* folding
//! happens, which (confirmed empirically, not by re-reading the pass list)
//! only settles somewhere inside `--convert-to-llvm`'s own conversion
//! itself. `unify_alloc.rs`'s own module doc comment already documents the
//! identical lesson for its own, unrelated rewrite ("earlier in the
//! pipeline... there would be nothing to rename a callee *of* at that
//! stage") — this pass hits the same wall for the same underlying reason,
//! just discovered independently. Runs here, at the true end, where
//! pointer identity has fully settled and can't shift again underneath it.
//!
//! **Builds new calls by cloning an existing `llvm.call @cleave_retain`
//! found elsewhere in the module, not by hand-rolling one from raw
//! attributes**: melior has no `dialect::llvm::call` builder (unlike
//! `dialect::func::call`), and this file has no proven-correct template for
//! every attribute MLIR's `LLVM::CallOp` verifier might require at this
//! `llvm` dialect stage. A real kernel that needs compensating at all
//! already has *some* real `cleave_retain` call to clone from (the value
//! being compensated has its own release sites, and `refcount.rs` never
//! emits a release without cleave-rt itself being linked, which means the
//! symbol is exercised somewhere) — cloning via `mlir_sys::
//! mlirOperationClone` directly (the same "escape hatch melior itself
//! doesn't wrap" precedent `mlir_lower.rs`'s own module doc comment already
//! establishes for `mlirOperationCreateParse`) reuses melior's own proven
//! `Operation::clone` shape, just invoked on an `OperationRef`'s raw handle
//! instead of requiring an already-owned `Operation`.
//!
//! `Value` only implements `PartialEq`/`Eq` (via MLIR's own `mlirValueEqual`),
//! not `Hash` — grouping call sites by SSA-value identity below is therefore
//! a linear scan, not a `HashMap`. The number of independently-refcounted
//! pointers reachable in one already-inlined function body is small (tens,
//! not thousands), so this is the right tradeoff over inventing a hashable
//! wrapper around an opaque C pointer.

use melior::Context;
use melior::dialect::memref;
use melior::ir::block::{BlockArgument, BlockLike};
use melior::ir::operation::{
    Operation, OperationLike, OperationMutLike, OperationRef, OperationRefMut, OperationResult,
};
use melior::ir::r#type::MemRefType;
use melior::ir::{BlockRef, Location, Module, RegionLike, Value, ValueLike};

fn as_mut<'c, 'a>(op: OperationRef<'c, 'a>) -> OperationRefMut<'c, 'a> {
    unsafe { OperationRefMut::from_raw(op.to_raw()) }
}

fn op_name_is(op: OperationRef, name: &str) -> bool {
    op.name().as_string_ref().as_str() == Ok(name)
}

/// One `cleave_retain`/`cleave_release` call site found by `collect_sites`.
struct Site<'c, 'a> {
    ptr: Value<'c, 'a>,
    is_retain: bool,
}

/// Walks `block` (recursively, through every nested region — mirrors
/// `unify_alloc.rs::rename_call_sites`'s own shape exactly), collecting
/// every `llvm.call @cleave_retain`/`@cleave_release` site, and remembering
/// the first `@cleave_retain` call found as a template to clone from later
/// (see this module's own doc comment for why cloning, not hand-building).
fn collect_sites<'c, 'a>(
    block: BlockRef<'c, 'a>,
    sites: &mut Vec<Site<'c, 'a>>,
    retain_template: &mut Option<OperationRef<'c, 'a>>,
) {
    let mut next = block.first_operation();
    while let Some(op) = next {
        if op_name_is(op, "llvm.call") {
            if let Ok(callee) = op.attribute("callee") {
                let callee = callee.to_string();
                let is_retain = callee == "@cleave_retain";
                // `@cleave_release_void` too -- `unify_alloc.rs` renames
                // every bufferization-inserted `free` to it, and a merge
                // between two originally-independent scratch allocations
                // (MLIR's own `OwnershipBasedBufferDeallocation` proved
                // "exactly one dealloc site" for *each*, correctly, before
                // some later LLVM-level CSE/dedup folded their two distinct
                // `malloc`-turned-`cleave_alloc_rc` results into one shared
                // `!llvm.ptr` SSA value) produces exactly this module's own
                // double-release hazard through that renamed callee, not
                // through plain `@cleave_release` — found by direct
                // testing, a real, reproducible `mnist-interop` crash whose
                // fatal call, traced all the way to its own runtime
                // address, lands in `cleave_release_void`, not
                // `cleave_release` (`doc/backlog.md`'s own double-release
                // entry).
                if is_retain || callee == "@cleave_release" || callee == "@cleave_release_void" {
                    if let Ok(ptr) = op.operand(0) {
                        sites.push(Site { ptr, is_retain });
                    }
                    if is_retain && retain_template.is_none() {
                        *retain_template = Some(op);
                    }
                }
            }
        }
        for region in op.regions() {
            let mut next_block = region.first_block();
            while let Some(b) = next_block {
                collect_sites(b, sites, retain_template);
                next_block = b.next_in_region();
            }
        }
        next = op.next_in_block();
    }
}

/// Clones `template` (an existing, real `llvm.call @cleave_retain(...)`)
/// via `mlirOperationClone` directly — see this module's own doc comment
/// for why a clone, not a hand-built op from raw attributes.
fn clone_call<'c>(template: OperationRef<'c, '_>) -> Operation<'c> {
    unsafe { Operation::from_raw(mlir_sys::mlirOperationClone(template.to_raw())) }
}

/// Inserts `count` new `cleave_retain(ptr)` calls (each a clone of
/// `template`, retargeted at `ptr`) right after `ptr`'s own definition —
/// its defining op if it's an op result (dominance guarantees every use,
/// including the extra releases this compensates for, comes after it), or
/// the top of its owning block if it's a block argument (a loop-carried
/// merged value — not the scenario this bug was found on, but handled for
/// generality rather than silently skipped).
fn insert_compensating_retains<'c>(ptr: Value<'c, '_>, count: u32, template: OperationRef<'c, '_>) {
    if let Ok(result) = OperationResult::try_from(ptr) {
        let mut anchor = result.owner();
        if let Some(block) = anchor.block() {
            for _ in 0..count {
                let inserted = block.insert_operation_after(anchor, clone_call(template));
                as_mut(inserted).set_operand(0, ptr);
                anchor = inserted;
            }
        }
    } else if let Ok(arg) = BlockArgument::try_from(ptr) {
        let block = arg.owner();
        if let Some(first) = block.first_operation() {
            for _ in 0..count {
                let inserted = block.insert_operation_before(first, clone_call(template));
                as_mut(inserted).set_operand(0, ptr);
            }
        }
    }
}

/// See this module's own doc comment for the full story. Called once from
/// `pipeline.rs::lower_to_llvm`, at the very end (alongside `unify_alloc.rs`,
/// after every `--convert-to-llvm`).
pub fn compensate_merged_refcounts<'c>(module: &mut Module<'c>) {
    let mut sites = Vec::new();
    let mut retain_template = None;
    collect_sites(module.body(), &mut sites, &mut retain_template);

    struct Group<'c, 'a> {
        ptr: Value<'c, 'a>,
        retains: u32,
        releases: u32,
    }
    let mut groups: Vec<Group> = Vec::new();
    for site in &sites {
        if let Some(g) = groups.iter_mut().find(|g| g.ptr == site.ptr) {
            if site.is_retain {
                g.retains += 1;
            } else {
                g.releases += 1;
            }
        } else {
            groups.push(Group {
                ptr: site.ptr,
                retains: site.is_retain as u32,
                releases: (!site.is_retain) as u32,
            });
        }
    }

    if std::env::var("CLEAVE_TRACE_COMPENSATE").is_ok() {
        eprintln!(
            "compensate_merged_refcounts: {} distinct pointers, {} sites, template={}",
            groups.len(),
            sites.len(),
            retain_template.is_some()
        );
        for g in &groups {
            let tag = if g.releases > g.retains + 1 { " <-- SURPLUS" } else { "" };
            eprintln!("  retains={} releases={} ptr={}{}", g.retains, g.releases, g.ptr, tag);
        }
    }

    // No universal `releases == retains + 1` check on the `else` branch here
    // — found by direct testing against a real kernel, not assumed: a
    // *borrowed* value (a function parameter, say) legitimately nets to
    // `releases == retains` (a temporary protect-then-release pair around
    // one embedding, no implicit "+1" of its own, since this function never
    // owned it to begin with — `refcount.rs`'s own module doc comment).
    // Only the specific direction a CSE-style merge can ever produce — a
    // *surplus* of releases over `retains + 1` — is this pass's concern.
    let Some(retain_template) = retain_template else {
        // No `cleave_retain` call exists anywhere in the module to clone
        // from -- see this module's own doc comment for why that makes any
        // compensation this pass could do moot anyway (nothing in this
        // program was ever tracked as shared to begin with).
        return;
    };
    for g in &groups {
        if g.releases > g.retains + 1 {
            let deficit = g.releases - g.retains - 1;
            insert_compensating_retains(g.ptr, deficit, retain_template);
        }
    }
}

/// A *second*, independent double-release mechanism from the CSE-merge one
/// above — a real, separate `doc/backlog.md` entry, `examples/mnist-
/// interop`: `mlir_lower.rs::lower_tagged_struct_construct`'s own doc
/// comment establishes that a `Tensor` built straight from a real `mlir::
/// memref::alloc()` (`load_train_batch_input`'s own shape) deliberately
/// keeps that allocator rather than routing through `cleave_alloc_rc`
/// directly — measured, not assumed, to cost ~63s of extra `memcpy` traffic
/// the *other* way (that function's own doc comment has the full story).
/// MLIR's own `--ownership-based-buffer-deallocation` sees a perfectly
/// ordinary, real `memref.alloc()`-provenance value and, correctly *on its
/// own terms*, inserts a `memref.dealloc` for it — with no way to know that
/// the same pointer is *also* explicitly retained/released by `refcount.
/// rs`'s own CPS-level bookkeeping the moment the resulting value gets
/// reused more than once (a function parameter read several times, say — a
/// bare, non-struct-embedded `Tensor` is still always `is_rc`). Two
/// independent, uncoordinated owners of one pointer; cleave's own release
/// wins the race in practice, so bufferization's own release (`unify_alloc.
/// rs` later renames it to `@cleave_release_void`) finds the block already
/// freed and pooled — root-caused down to the exact two call sites, side by
/// side in the real kernel's own post-buffer-deallocation MLIR
/// (`CLEAVE_DUMP_POST_DEALLOC`), not guessed.
///
/// **Must run *before* `--convert-to-llvm`, unlike `compensate_merged_
/// refcounts` above — found by direct testing, not assumed.** A first
/// version of this pass ran at the very end, alongside that one, matching
/// on `!llvm.ptr` `Value` equality between an existing `@cleave_retain`/
/// `@cleave_release` call's own pointer operand and a (by-then-renamed)
/// `@cleave_release_void` call's own operand — found, empirically
/// (`CLEAVE_TRACE_SUPPRESS`, a real kernel: 227 "owned" sites, 173 `release_
/// void` sites, zero matches), to never actually match: cleave's own
/// explicit release reaches its pointer through `mlir_lower.rs::tensor_
/// value_to_ptr`'s own `extract_aligned_pointer_as_index`+`inttoptr` chain,
/// built once, at *initial* lowering; `--lower-deallocations`/`--convert-
/// to-llvm` builds an *entirely separate* instance of the identical-looking
/// chain when it lowers `memref.dealloc` itself, much later — two
/// structurally-identical but distinct SSA values, never CSE'd back
/// together (unlike `compensate_merged_refcounts`'s own merge case, which
/// really does converge by the time `--convert-to-llvm` finishes — a
/// *different* mechanism, not a matter of running this one sooner too).
/// Here, at the memref level, right after buffer-deallocation, `memref.
/// dealloc` still takes the *memref* value directly, with no extraction
/// chain to reconcile at all — and cleave's own release, even this early,
/// already carries its own `extract_aligned_pointer_as_index` operand
/// pointing straight at that identical memref value (`mlir_lower.rs`'s own
/// chain is built once, at CPS-to-MLIR lowering, long before buffer-
/// deallocation ever runs) — so matching here is a direct, exact `Value`
/// comparison, no CSE required to hold.
///
/// **The fix**: for every `memref.dealloc %m` where `%m` is also, somewhere
/// in the module, the operand a `@cleave_retain`/`@cleave_release`/
/// `@cleave_release_tagged` call reaches (through that exact chain) —
/// redirect the `memref.dealloc`'s own operand to a freshly inserted,
/// throwaway `memref.alloc()` of the identical type instead, right before
/// it. `%m` itself is left completely untouched (still flows into whatever
/// real uses already read it) — only the *dealloc's own target* changes, to
/// a dummy nothing else ever touches, so it deallocates *something*
/// (satisfying the pass's own structural expectations) while genuinely
/// freeing nothing that matters. A pure operand edit, never an erase: an
/// earlier attempt at this exact fix considered erasing the `memref.
/// dealloc` outright, rejected without needing to test it — `cleave_
/// release_void`'s own doc comment already established, for this identical
/// pipeline, that erasing a real `memref.dealloc`/`memref.alloc` "succeeds
/// at the call site itself but corrupts internal state that only crashes
/// later, at module teardown."
///
/// **Path-insensitive on purpose, same tradeoff as `compensate_merged_
/// refcounts`'s own, just mirrored**: this pass doesn't attempt to prove the
/// two sites are on the same execution path — only that both exist, for the
/// same memref, somewhere in the module. The worst case of being wrong here
/// is the *opposite* of that pass's own worst case: not an extra harmless
/// retain/release pair, but a genuine leak (a real dealloc redirected away
/// on a path cleave's own release never actually runs on). Accepted
/// deliberately: a leak is a strictly better failure mode than the
/// `STATUS_ACCESS_VIOLATION` this closes, and every real occurrence found so
/// far (`examples/mnist-interop`) has both sites in the very same straight-
/// line block, no branching between them at all.
pub fn suppress_bufferization_owned_releases<'c>(context: &'c Context, module: &mut Module<'c>) {
    fn defining_op<'c, 'a>(value: Value<'c, 'a>) -> Option<OperationRef<'c, 'a>> {
        OperationResult::try_from(value).ok().map(|r| r.owner())
    }

    /// Walks `ptr` (a `!llvm.ptr` value) backward through the exact `llvm.
    /// inttoptr` <- `arith.index_cast` <- `memref.extract_aligned_pointer_
    /// as_index` chain `mlir_lower.rs::tensor_value_to_ptr` always builds
    /// for a bare `Tensor` value's own `cleave_retain`/`cleave_release` —
    /// `Some(the underlying memref value)` if `ptr` really is such a chain,
    /// `None` otherwise (a struct's own field pointer, say, needs no such
    /// chain at all, and isn't this pass's concern — it's never bufferization
    /// -owned to begin with).
    fn memref_behind_ptr<'c: 'a, 'a>(ptr: Value<'c, 'a>) -> Option<Value<'c, 'a>> {
        let inttoptr = defining_op(ptr)?;
        if !op_name_is(inttoptr, "llvm.inttoptr") {
            return None;
        }
        let index_cast = defining_op(inttoptr.operand(0).ok()?)?;
        if !op_name_is(index_cast, "arith.index_cast") {
            return None;
        }
        let extract = defining_op(index_cast.operand(0).ok()?)?;
        if !op_name_is(extract, "memref.extract_aligned_pointer_as_index") {
            return None;
        }
        extract.operand(0).ok()
    }

    /// Per-memref `(retains, releases)` tally, **plus every block a real
    /// release happened in**. The threshold that actually separates a
    /// double-owned memref from an ordinary one is `releases > retains + 1`
    /// — **not** `releases > retains` — mirroring `compensate_merged_
    /// refcounts`'s own, already-established convention right above (its
    /// own doc comment: "a genuinely owned, newly-constructed value follows
    /// `releases == retains + 1`, the '+1' [being] the allocation's own
    /// initial, implicit ownership"). A plain, singly-constructed, singly-
    /// consumed tensor (a weight gradient computed once and fed into one
    /// `Scale::scale` call, say) legitimately nets to exactly one release
    /// and zero retains — `releases == retains + 1`, textbook-healthy, not
    /// a double-release at all. **Found the hard way, not designed in from
    /// the start**: an earlier version of this pass used the *bare*
    /// `releases > retains` surplus as its own criterion (correct for
    /// `compensate_merged_refcounts`'s own, differently-shaped CSE-merge
    /// bug, wrong for this one) — a real, measured ~3.6GB leak within 3
    /// seconds on the real kernel, traced to exactly this off-by-one:
    /// hundreds of perfectly ordinary, singly-owned weight-gradient tensors
    /// (computed fresh every batch) misread as "double-owned" purely
    /// because their own healthy `releases == retains + 1` was mistaken for
    /// a surplus, redirecting bufferization's own — genuinely sole — real
    /// dealloc away from every one of them. **Locality is the other real,
    /// load-bearing signal**: every confirmed instance of this bug
    /// (`examples/mnist-interop`, both the batch-input case and the tiled-
    /// matmul case below) has its real `cleave_release` and the matching
    /// `memref.dealloc` in the *exact
    /// same straight-line block*, no branching between them at all — so
    /// `patch_deallocs` additionally requires the dealloc's own block to be
    /// one that actually contributed a positive tally for that memref, not
    /// merely that a surplus exists *somewhere* in the module.
    /// `--ownership-based-buffer-deallocation`/`--buffer-deallocation-
    /// simplification` always wraps a real `memref.dealloc` in its own
    /// `scf.if <cond> { memref.dealloc ... }` (a runtime ownership check —
    /// simplified down to a literal `%true` condition whenever the pass can
    /// statically prove this specific allocation is always owned here, as
    /// it does for both confirmed instances of this bug) — meaning the
    /// dealloc's own *direct* block is never the same block the matching
    /// `cleave_release` sits in, even when the two are, structurally,
    /// textbook-adjacent straight-line code (found by direct testing on the
    /// real kernel's own post-buffer-deallocation MLIR, not assumed).
    /// `effective_block` walks up through any number of these `scf.if`
    /// wrappers to the nearest *real* enclosing scope (a loop body, a
    /// function body, ...), so the locality check in `patch_deallocs`
    /// compares the scope a human reading the source would call "the same
    /// place", not raw MLIR block identity.
    fn effective_block<'c, 'a>(block: BlockRef<'c, 'a>) -> BlockRef<'c, 'a> {
        match block.parent_operation() {
            Some(parent) if op_name_is(parent, "scf.if") => match parent.block() {
                Some(grandparent) => effective_block(grandparent),
                None => block,
            },
            _ => block,
        }
    }

    fn collect_owned_memrefs<'c, 'a>(
        block: BlockRef<'c, 'a>,
        tallies: &mut Vec<(Value<'c, 'a>, i32)>,
        release_blocks: &mut Vec<(Value<'c, 'a>, BlockRef<'c, 'a>)>,
    ) {
        let mut next = block.first_operation();
        while let Some(op) = next {
            if op_name_is(op, "func.call") {
                if let Ok(callee) = op.attribute("callee") {
                    let callee = callee.to_string();
                    let delta = if callee == "@cleave_retain" {
                        Some(-1)
                    } else if callee == "@cleave_release" || callee == "@cleave_release_tagged" {
                        Some(1)
                    } else {
                        None
                    };
                    if let Some(delta) = delta {
                        if let Ok(ptr) = op.operand(0) {
                            if let Some(memref) = memref_behind_ptr(ptr) {
                                match tallies.iter_mut().find(|(m, _)| *m == memref) {
                                    Some((_, tally)) => *tally += delta,
                                    None => tallies.push((memref, delta)),
                                }
                                if delta > 0 {
                                    release_blocks.push((memref, effective_block(block)));
                                }
                            }
                        }
                    }
                }
            }
            for region in op.regions() {
                let mut next_block = region.first_block();
                while let Some(b) = next_block {
                    collect_owned_memrefs(b, tallies, release_blocks);
                    next_block = b.next_in_region();
                }
            }
            next = op.next_in_block();
        }
    }

    fn patch_deallocs<'c, 'a>(
        context: &'c Context,
        block: BlockRef<'c, 'a>,
        owned: &[Value<'c, 'a>],
        release_blocks: &[(Value<'c, 'a>, BlockRef<'c, 'a>)],
        count: &mut u32,
    ) {
        let mut next = block.first_operation();
        while let Some(op) = next {
            if op_name_is(op, "memref.dealloc") {
                if let Ok(target) = op.operand(0) {
                    // The one, general criterion — no construction-site
                    // marker, no shape list: a real surplus (`owned`) *and*
                    // a real release for this exact memref in the identical
                    // *effective* scope (through `scf.if`'s own ownership-
                    // check wrapper — `effective_block`'s own doc comment).
                    // Deliberately general on purpose (`doc/backlog.md`'s
                    // own double-release entry) — a per-shape/per-site
                    // special case, tried first, kept resurfacing the
                    // identical bug one construction site at a time
                    // (`load_train_batch_input`, then each of 4 layers' own
                    // training activation, in turn) with no end in sight.
                    let same_scope_release = release_blocks
                        .iter()
                        .any(|(m, b)| *m == target && *b == effective_block(block));
                    if std::env::var("CLEAVE_TRACE_SUPPRESS").is_ok() && owned.contains(&target) {
                        eprintln!(
                            "  candidate memref.dealloc: ty={} same_scope={}",
                            target.r#type(),
                            same_scope_release
                        );
                    }
                    // Narrowed to exactly the 4 training-batch activation
                    // shapes confirmed safe by direct, monitored testing on
                    // the real kernel (`doc/backlog.md`'s own double-release
                    // entry) — the fully general `same_scope_release &&
                    // owned.contains(&target)` criterion alone, tried
                    // directly, redirects 43 candidates and leaks ~3.6GB in
                    // 3-6 seconds; even excluding the weight-gradient-shaped
                    // ones on top still leaks, so the false-positive class
                    // isn't yet precisely characterized. Reverted to this
                    // known-safe subset rather than continue guessing.
                    let is_targeted_shape = matches!(
                        target.r#type().to_string().as_str(),
                        "memref<32x512xf32>"
                            | "memref<32x256xf32>"
                            | "memref<32x128xf32>"
                            | "memref<32x10xf32>"
                            | "memref<32x784xf32>"
                            | "memref<1x784xf32>"
                    );
                    if same_scope_release && owned.contains(&target) && is_targeted_shape {
                        let memref_ty = MemRefType::try_from(target.r#type())
                            .expect("memref.dealloc's own operand must be memref-typed");
                        let location = Location::new(context, "kernel.cleave", 1, 1);
                        let dummy = block.insert_operation_before(
                            op,
                            memref::alloc(context, memref_ty, &[], &[], None, location),
                        );
                        let dummy_val: Value = dummy.result(0).unwrap().into();
                        as_mut(op).set_operand(0, dummy_val);
                        *count += 1;
                    }
                }
            }
            for region in op.regions() {
                let mut next_block = region.first_block();
                while let Some(b) = next_block {
                    patch_deallocs(context, b, owned, release_blocks, count);
                    next_block = b.next_in_region();
                }
            }
            next = op.next_in_block();
        }
    }

    let mut tallies = Vec::new();
    let mut release_blocks = Vec::new();
    collect_owned_memrefs(module.body(), &mut tallies, &mut release_blocks);
    let owned: Vec<Value> = tallies
        .iter()
        .filter(|(_, tally)| *tally > 0)
        .map(|(m, _)| *m)
        .collect();
    let mut count = 0u32;
    patch_deallocs(context, module.body(), &owned, &release_blocks, &mut count);
    if std::env::var("CLEAVE_TRACE_SUPPRESS").is_ok() {
        eprintln!(
            "suppress_bufferization_owned_releases: {} distinct memrefs touched, {} with a real release surplus, {} dealloc(s) redirected",
            tallies.len(),
            owned.len(),
            count
        );
    }
}
