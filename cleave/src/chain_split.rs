//! Splits the long, strictly sequential `vector.outerproduct` chain that
//! `apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"`
//! (`cleave/mlir/matmul_vectorize.transform.mlir`) leaves behind for every
//! vectorized K-tile contraction into `factor` independent, shorter chains,
//! combined with `factor - 1` `arith.addf`s at the end — a real, measured
//! fix for the same `~6x` per-FLOP IPC gap `unroll_jam.rs` also targets, but
//! at the *right* granularity, found only after that first attempt's own
//! real-kernel measurement (`doc/backlog.md`) showed it wasn't.
//!
//! **Why this exists alongside `unroll_jam.rs`, not instead of it, at
//! first — then why it turned out to be the one that actually works.**
//! `unroll_jam.rs` widens the *outer* K-tile `scf.for` loop (trip count
//! `K/16`) into `factor` independent copies. Wired into the real pipeline
//! and measured (AMD uProf), it moved *nothing*: `MatMul::matmul`'s own
//! K=784 reduction stayed at IPC `0.227`, identical to the `0.226` baseline.
//! Disassembling the result showed why: every one of those "independent"
//! copies loads its *own* `16x16` operand tile (`vector.transfer_read` into
//! `vector<16x16xf32>`) -- a real, unavoidable `16`-register cost per copy,
//! before a single accumulator register is even counted. `factor=7` demanded
//! `7 * 16 = 112` `zmm` registers against a `32`-register file; the
//! compiler's register allocator answered with constant spill/reload that
//! ate exactly the cycles the extra ILP was supposed to save.
//!
//! Dumping the real IR right after the schedule runs (`CLEAVE_DUMP_PRE_
//! BUFFERIZE`) showed *why* that tile load is so large, and where the real
//! narrow chain actually lives: `vectorize {create_named_contraction}`
//! produces one `vector.contract` per `16`-wide K-tile, and lowering it to
//! `vector.outerproduct` unrolls the contraction's own reduction dimension
//! into `16` *sequential* `vector.outerproduct`s, each one's own accumulator
//! operand being the *previous* one's single result -- a `16`-deep
//! dependency chain **inside a single K-tile iteration**, not across them.
//! The outer `scf.for`'s own trip count (`K/16`) only multiplies this
//! further: the *real* critical path for a `K=784` reduction is `784`
//! sequential `vfmadd`-latency steps end to end, which is why `unroll_jam.rs`
//! measured such a catastrophic IPC in the first place -- and why widening
//! the *outer* loop, which duplicates the `16`-register tile load per copy,
//! was the wrong lever: the chain to break is *inside* the tile, not across
//! tiles.
//!
//! **A real, existing MLIR transform-dialect knob was checked first and
//! measured to be a worse fix, not assumed to be irrelevant.** `apply_
//! patterns.vector.lower_contraction` takes a `lowering_strategy` other than
//! `"outerproduct"`: `"dot"` computes each output lane via an elementwise
//! `arith.mulf` plus a single `vector.reduction<add>` (a hardware, `log2(16)
//! = 4`-deep shuffle-reduce tree) -- independent per lane, confirmed via an
//! isolated `mlir-opt --transform-interpreter` probe on the exact same
//! `1x16 . 16x16 -> 1x16` shape before touching the real schedule. Measured
//! on the real kernel (same AMD uProf protocol): `MatMul::matmul`'s own IPC
//! *did* jump from `0.226` to `2.574` (confirmed independently twice, once
//! via this project's own DuckDB query against `cpu.db`, once via the AMD
//! uProf GUI's own per-function `CYCLES_NOT_IN_HALT` view) -- but real
//! wall-clock more than doubled (`68s -> ~200s`, confirmed directly by hand,
//! not just this session's own noisier bash-tool timings). Summing
//! `CYCLES_NOT_IN_HALT` across every hot function in the uProf GUI's own
//! summary view against the real clock (`~313e9` cycles at `4.5GHz = ~70s`
//! of *active* compute inside a `281s` profile) showed roughly `75%` of
//! total wall-clock time was spent *halted*, not computing -- `"dot"`
//! trades a shorter dependency chain for enough extra `vector.extract`/
//! `vector.insert`/`vector.reduction` instruction volume (and, going by the
//! ballooned halted-time share, probably larger generated code / more
//! `_chkstk` stack-probing overhead too) that it is a real, measured net
//! loss despite the IPC win. Reverted; `matmul_vectorize.transform.mlir`
//! is back to `"outerproduct"`.
//!
//! **This pass is the third attempt: keep `"outerproduct"` (cheap, no extra
//! instruction volume) but restructure *which* accumulator each step feeds,
//! entirely by rewiring existing operands** -- no new loop, no cloning, no
//! extra operand loads. The `16x16` tile (`vector.transfer_read`) and the
//! transposed `1x16` row it's paired against are read *once*, exactly as
//! before; every one of the original `16` `vector.extract`/`vector.
//! outerproduct` pairs stays exactly where it already is. The only changes:
//! at each of `factor - 1` split points, the group's first `vector.
//! outerproduct` has its own accumulator operand redirected from "the
//! previous group's last op's result" to a fresh `vector.broadcast` of
//! `0.0`, and `factor - 1` new `arith.addf`s are inserted after the whole
//! original chain to combine the `factor` independent partial sums into one.
//! Register cost: the same unavoidable `16` for the tile, plus `factor`
//! (one live accumulator per independent partial chain) instead of `1` --
//! `factor=4` costs `20` registers total, comfortably under `32`, unlike
//! `unroll_jam.rs`'s own `112`.

use melior::Context;
use melior::ir::attribute::FloatAttribute;
use melior::ir::block::BlockLike;
use melior::ir::operation::{
    OperationBuilder, OperationLike, OperationMutLike, OperationRef, OperationRefMut,
};
use melior::ir::{Block, Identifier, Module, RegionLike, Type, TypeLike, Value, ValueLike};

/// Candidate split factors, tried largest first, only ever chosen if it
/// evenly divides the real chain length (`pick_split_factor`) -- this
/// project's own K-tile width is fixed at `16` (`matmul_vectorize.transform
/// .mlir`'s own `tile_using_for tile_sizes [0, 0, 16]`), so `4` is expected
/// to be the answer in practice, but nothing here hardcodes `16`.
const CANDIDATE_SPLIT_FACTORS: &[i64] = &[4, 3, 2];

/// Below this many chained `vector.outerproduct`s, leave the chain alone --
/// splitting a short chain barely shortens the critical path and isn't
/// worth even the small fixed cost of the combining `addf`s. Also what
/// keeps this pass from re-splitting its own output: `split_outerproduct_
/// chains`'s own driving loop re-scans the whole module after every split,
/// and each of the `factor` groups a split produces is itself still a
/// shorter chained run of `vector.outerproduct`s (`chains_from` doesn't
/// distinguish "original" from "already split"). With `CANDIDATE_SPLIT_
/// FACTORS` capped at `4`, any chain long enough to be split at all
/// (`>= 8`) produces groups no longer than `len / 2`, always `< 8` --
/// found directly, not assumed: an earlier version of this pass with the
/// threshold at `4` recursively flattened every 16-chain down to 16 fully
/// independent single-`outerproduct` "chains" combined by a much deeper
/// tree of `addf`s than intended, visible immediately as repeated `len=4
/// factor=4` splits in `CLEAVE_TRACE_CHAIN_SPLIT=1` right after each real
/// `len=16 factor=4` one.
const MIN_CHAIN_LEN_TO_SPLIT: i64 = 8;

/// Runs the pass over every `vector.outerproduct` chain anywhere in
/// `module`, splitting each one whose length is both long enough
/// (`MIN_CHAIN_LEN_TO_SPLIT`) and evenly divisible by some candidate factor.
/// Conservative by construction, matching `unroll_jam.rs`'s own posture: a
/// chain this pass doesn't recognize, or can't find a factor for, is left
/// byte-for-byte as the schedule already built it.
///
/// **Off by default** (`CodegenOptions::chain_split`, `--chain-split` on the
/// CLI) — same posture and same reason as `unroll_jam.rs`'s own doc comment
/// on `unroll_jam`: real, disassembly-verified to produce exactly the
/// intended independent-chain structure, but measured on the real kernel
/// (clean rebuild, AMD uProf) at IPC `0.197` — *worse* than the `0.226`
/// unsplit baseline, not better (`doc/backlog.md`'s own full writeup). The
/// bottleneck this pass targets (FMA dependency-chain latency) was never the
/// real one for this kernel — a cache-locality fix elsewhere closed the
/// actual gap. Kept for the same reason `unroll_jam.rs` is: a correct,
/// working mechanism that simply wasn't the fix for this specific shape.
pub fn split_outerproduct_chains<'c>(context: &'c Context, module: &mut Module<'c>, enabled: bool) {
    if !enabled {
        return;
    }
    let trace = std::env::var("CLEAVE_TRACE_CHAIN_SPLIT").is_ok();
    let mut applied = 0usize;
    loop {
        let op = module.as_operation_mut();
        let Some(chain) = find_next_chain(op) else {
            break;
        };
        let len = chain.len() as i64;
        let Some(factor) = pick_split_factor(len) else {
            // Should not happen -- `find_next_chain` already only returns
            // chains long enough to have *some* candidate factor -- but
            // conservatively bail rather than panic if that ever changes.
            break;
        };
        if trace {
            eprintln!("CLEAVE_TRACE_CHAIN_SPLIT: splitting chain len={len} factor={factor}");
        }
        split_chain(context, chain, factor);
        applied += 1;
    }
    if trace {
        eprintln!("CLEAVE_TRACE_CHAIN_SPLIT: {applied} chain(s) split");
    }
}

/// A real, evenly-divisible split factor for a chain of length `len`, or
/// `None` if the chain is too short or no candidate divides it evenly.
fn pick_split_factor(len: i64) -> Option<i64> {
    if len < MIN_CHAIN_LEN_TO_SPLIT {
        return None;
    }
    CANDIDATE_SPLIT_FACTORS
        .iter()
        .copied()
        .find(|factor| len % factor == 0)
}

fn is_outerproduct(op: &OperationRefMut) -> bool {
    matches!(op.name().as_string_ref().as_str(), Ok("vector.outerproduct"))
}

/// Finds the next maximal run of chained `vector.outerproduct`s anywhere in
/// `op`'s own subtree with a real, usable split factor -- searches each
/// block's own top-level op list first (`find_chain_in_block`), then
/// recurses into every child op's own sub-regions, mirroring `unroll_jam.rs`
/// 's own `find_next_candidate` walk order.
fn find_next_chain<'c, 'a>(op: OperationRefMut<'c, 'a>) -> Option<Vec<OperationRefMut<'c, 'a>>> {
    for region in op.regions() {
        let mut next_block = region.first_block();
        while let Some(block) = next_block {
            if let Some(chain) = find_chain_in_block(&block) {
                return Some(chain);
            }
            let mut next_op = block.first_operation_mut();
            while let Some(child) = next_op {
                next_op = child.next_in_block_mut();
                if let Some(chain) = find_next_chain(child) {
                    return Some(chain);
                }
            }
            next_block = block.next_in_region();
        }
    }
    None
}

/// Scans `block`'s own top-level op list (not descending into any op's own
/// sub-regions -- the caller, `find_next_chain`, already does that
/// separately) for a maximal run of `vector.outerproduct`s where each one
/// after the first consumes the *previous* one's single result as its own
/// accumulator operand -- exactly the shape `lower_contraction lowering_
/// strategy = "outerproduct"` produces for one K-tile's own reduction
/// dimension. Returns the first such run at least `MIN_CHAIN_LEN_TO_SPLIT`
/// long with a real split factor, or `None`.
fn find_chain_in_block<'c, 'a>(block: &Block<'c>) -> Option<Vec<OperationRefMut<'c, 'a>>> {
    let mut ops = Vec::new();
    let mut next = block.first_operation_mut();
    while let Some(op) = next {
        next = op.next_in_block_mut();
        ops.push(op);
    }

    let mut i = 0;
    while i < ops.len() {
        if !is_outerproduct(&ops[i]) {
            i += 1;
            continue;
        }
        // The chain is a *data*-dependency chain, not a physically
        // contiguous run of ops -- the real schedule interleaves two
        // `vector.extract`s (one per `vector.outerproduct` operand) between
        // consecutive links (see this module's own doc comment). Search
        // forward for the next `vector.outerproduct` that actually consumes
        // the current end's own result, skipping over whatever sits between
        // them untouched.
        let mut chain_indices = vec![i];
        let mut cursor = i;
        for (j, op) in ops.iter().enumerate().skip(i + 1) {
            if is_outerproduct(op) && chains_from(&ops[cursor], op) {
                chain_indices.push(j);
                cursor = j;
            }
        }
        let len = chain_indices.len() as i64;
        if pick_split_factor(len).is_some() {
            let mut ops: Vec<Option<OperationRefMut<'c, 'a>>> =
                ops.into_iter().map(Some).collect();
            return Some(
                chain_indices
                    .into_iter()
                    .map(|idx| ops[idx].take().unwrap())
                    .collect(),
            );
        }
        i = cursor + 1;
    }
    None
}

/// Does `next` consume `prev`'s own single result as its own accumulator
/// operand (operand index `2`: `vector.outerproduct`'s own `$lhs, $rhs,
/// $acc` operand order)?
fn chains_from(prev: &OperationRefMut, next: &OperationRefMut) -> bool {
    if !is_outerproduct(next) || next.operand_count() != 3 {
        return false;
    }
    let Ok(acc) = next.operand(2) else { return false };
    let Ok(prev_result) = prev.result(0) else {
        return false;
    };
    acc == Value::from(prev_result)
}

/// Every real use of `value` anywhere in the module, as `(owning operation,
/// operand index)` pairs -- collected via raw FFI (`mlirValueGetFirstUse`/
/// `mlirOpOperandGetOwner`/`mlirOpOperandGetOperandNumber`/`mlirOpOperandGet
/// NextUse`), since melior 0.27.4 has no safe wrapper for a value's own
/// use-list (its own `value_like.rs` marks this a `TODO`). Collected as a
/// snapshot *before* building anything new that might also use `value` --
/// `split_chain`'s own combining `addf` chain legitimately needs to consume
/// the original chain's final result as one of its real inputs, and walking
/// a *live* use-list while adding to it would incorrectly rewrite that new
/// use right back onto itself.
fn collect_uses(value: Value) -> Vec<(mlir_sys::MlirOperation, u32)> {
    let mut uses = Vec::new();
    let mut use_ = unsafe { mlir_sys::mlirValueGetFirstUse(value.to_raw()) };
    while !unsafe { mlir_sys::mlirOpOperandIsNull(use_) } {
        let owner = unsafe { mlir_sys::mlirOpOperandGetOwner(use_) };
        let index = unsafe { mlir_sys::mlirOpOperandGetOperandNumber(use_) };
        uses.push((owner, index));
        use_ = unsafe { mlir_sys::mlirOpOperandGetNextUse(use_) };
    }
    uses
}

/// Rewrites `chain` (a run of `factor`-divisible chained `vector.
/// outerproduct`s, all still exactly where the schedule built them) into
/// `factor` independent partial chains combined by `factor - 1` `arith.
/// addf`s -- see this module's own doc comment for the full mechanism and
/// why it's safe to do purely by rewiring existing operands.
fn split_chain<'c>(context: &'c Context, mut chain: Vec<OperationRefMut<'c, '_>>, factor: i64) {
    let len = chain.len();
    let group_size = len / factor as usize;
    let loc = chain[0].location();

    let acc_ty = chain[0].operand(2).unwrap().r#type();
    let elem_ty = unsafe { Type::from_raw(mlir_sys::mlirShapedTypeGetElementType(acc_ty.to_raw())) };

    // Cut the chain at each of the `factor - 1` internal group boundaries --
    // group `0` is left completely untouched, starting from the loop's own
    // real accumulator exactly as before.
    for g in 1..factor as usize {
        let boundary = g * group_size;
        let parent_block = chain[boundary].block().expect("op has a parent block");
        let boundary_ref: OperationRef = unsafe { OperationRef::from_raw(chain[boundary].to_raw()) };

        let zero_attr = FloatAttribute::new(context, elem_ty, 0.0);
        let zero_scalar = OperationBuilder::new("arith.constant", loc)
            .add_attributes(&[(Identifier::new(context, "value"), zero_attr.into())])
            .add_results(&[elem_ty])
            .build()
            .expect("failed to build zero scalar constant");
        let zero_scalar = parent_block.insert_operation_before(boundary_ref, zero_scalar);
        let zero_scalar_val: Value = zero_scalar.result(0).unwrap().into();

        let broadcast = OperationBuilder::new("vector.broadcast", loc)
            .add_operands(&[zero_scalar_val])
            .add_results(&[acc_ty])
            .build()
            .expect("failed to build zero broadcast");
        let broadcast = parent_block.insert_operation_before(boundary_ref, broadcast);
        let broadcast_val: Value = broadcast.result(0).unwrap().into();

        chain[boundary].set_operand(2, broadcast_val);
    }

    // Snapshot every real use of the *original* chain's own final result
    // before building the combining `addf`s -- see `collect_uses`'s own doc
    // comment for why the ordering here matters.
    let old_final: Value = chain[len - 1].result(0).unwrap().into();
    let uses = collect_uses(old_final);

    // Combine the `factor` independent partial sums, inserted right after
    // the whole original chain -- physically *after* every op any of them
    // reference, however early in the chain that op sits.
    let parent_block = chain[len - 1].block().expect("op has a parent block");
    let mut insert_after: OperationRef = unsafe { OperationRef::from_raw(chain[len - 1].to_raw()) };
    let mut combined: Value = chain[group_size - 1].result(0).unwrap().into();
    for g in 1..factor as usize {
        let group_last = (g + 1) * group_size - 1;
        let rhs: Value = chain[group_last].result(0).unwrap().into();
        let add_op = OperationBuilder::new("arith.addf", loc)
            .add_operands(&[combined, rhs])
            .add_results(&[acc_ty])
            .build()
            .expect("failed to build combining addf");
        let add_op = parent_block.insert_operation_after(insert_after, add_op);
        combined = add_op.result(0).unwrap().into();
        insert_after = add_op;
    }

    // Rewire every real use captured *before* the combining `addf`s existed
    // onto the new, combined value -- the `addf`s' own use of `old_final`
    // (built afterward) is correctly left alone.
    for (owner_raw, index) in uses {
        let mut owner = unsafe { OperationRefMut::from_raw(owner_raw) };
        owner.set_operand(index as usize, combined);
    }
}
