//! Unroll-and-jam for a vectorized `scf.for` reduction — widens the narrow
//! (3-4 register) accumulator chain `transform.structured.vectorize
//! {create_named_contraction}` leaves behind (`cleave/mlir/matmul_vectorize
//! .transform.mlir`'s own `tile_using_for tile_sizes [0, 0, 16]` K-tiling)
//! into `factor` genuinely independent accumulators, combined once at the
//! end — a real, measured fix for the `~6x` per-FLOP gap a long-K matmul
//! pays relative to a short-K one (`doc/backlog.md`'s own AMD uProf/IPC
//! investigation: `MatMul::matmul`'s own K=784 forward reduction measured at
//! IPC `0.226`, against `MatMulTransposeA`'s K=32 reduction at IPC `1.7`,
//! identical FLOPs).
//!
//! **Why this is hand-written Rust, not a `transform.loop.unroll_and_jam`
//! call**: MLIR's own SCF-dialect implementation of that utility
//! (`mlir/lib/Dialect/SCF/Utils/Utils.cpp::loopUnrollJamByFactor`) rejects
//! any loop with results outright — confirmed by reading it directly, not
//! assumed. Only the *Affine* dialect's own version (`mlir/lib/Dialect/
//! Affine/Utils/LoopUtils.cpp`) handles a top-level reduction loop,
//! including the final combining step this module also needs — and the K-
//! reduction loop this project's own schedule produces is `scf.for`, not
//! `affine.for`. This is that same algorithm, ported by hand, proven first
//! on an isolated, JIT-executed probe (`cleave/tests/unroll_jam_probe.rs`)
//! before landing here — see that file for the three generalizations
//! (a real `vector.contract` combine, a scalar accumulator, a loop nested
//! inside a non-reducing outer loop) each verified independently there.
//!
//! **The one structural difference from a straight C++ port**: an MLIR op
//! has a fixed result count once built — there is no "add a result to this
//! existing op" operation (`replaceWithAdditionalYields`, the real C++
//! utility this needs, has no melior equivalent). So this builds a brand
//! *new* `scf.for` with `factor` iter_args/results from scratch (a new body
//! block, the old body's ops cloned `factor` times inside it), rather than
//! mutating the original op in place, then rewires every real use of the old
//! loop's one result onto the new, combined value and erases the old op.
//!
//! **The factor is capped by real register pressure, not just trip-count
//! divisibility — found the hard way, not designed in up front.** The first
//! working version of this pass picked the largest `CANDIDATE_FACTORS` entry
//! that evenly divided the trip count and nothing else; wired into the real
//! pipeline, it chose `factor=7` for the real K=784 matmul and measured
//! *zero* IPC improvement (AMD uProf: `0.227`, same as the unwidened
//! baseline's `0.226`). Disassembling the result (`llvm-objdump` on
//! `mnist-interop.exe`) showed why: this project's real accumulator is
//! `vector<8x16xf32>` — `512` bytes, a full `8` AVX-512 `zmm` registers on
//! its own — so `factor=7` demanded `56` architectural registers against a
//! `32`-register file, and the compiler's register allocator answered with
//! constant spill/reload (`vmovaps` to a stack slot immediately followed by
//! reloading a *different* value into the same register, repeated
//! throughout the loop body) that ate exactly the cycles the wider
//! accumulator chain was supposed to save. `vector_register_count`/
//! `RESERVED_VECTOR_REGISTERS` below cap the factor search by the real
//! register budget for this reason — not a hypothetical concern, a directly
//! observed failure mode.

use melior::Context;
use melior::ir::attribute::IntegerAttribute;
use melior::ir::block::BlockLike;
use melior::ir::operation::{
    OperationBuilder, OperationLike, OperationMutLike, OperationRef, OperationRefMut,
    OperationResult,
};
use melior::ir::{Block, Identifier, Module, Region, RegionLike, Type, TypeLike, Value, ValueLike};

/// Candidate unroll factors, tried largest first — chosen once we know the
/// real trip count (see `pick_factor`), never blindly fixed: `784/16 = 49`
/// (`7 * 7`, only `7`/`49` divide it) and `256/16 = 16` (divisible by every
/// power of two up to `16`) need genuinely different factors — there is no
/// single value that evenly divides every real matmul shape's own K-tile
/// trip count. Also capped, separately, by real register pressure -- see
/// `vector_register_count`/`pick_factor`'s own doc comments for why `8`
/// here is an upper bound `pick_factor` will in practice almost never reach
/// for this project's actual `vector<8x16xf32>` accumulators.
const CANDIDATE_FACTORS: &[i64] = &[8, 7, 6, 5, 4, 3, 2];

/// Bytes in one AVX-512 architectural vector register (`zmm0`-`zmm31`).
const VECTOR_REGISTER_BYTES: i64 = 64;

/// The hard architectural limit on this project's own target ISA
/// (`fma_roofline.rs`'s own roofline measurement is itself AVX-512-specific)
/// -- `zmm0` through `zmm31`, fixed by the ISA, not a tuning knob.
const TOTAL_VECTOR_REGISTERS: i64 = 32;

/// Registers left over for everything in the loop body that ISN'T one of
/// the widened accumulators -- the two operand loads/broadcasts every
/// `vector.outerproduct` iteration needs, plus whatever else the vectorized
/// body still keeps live. Found by disassembling the real, un-widened K=784
/// loop directly (`llvm-objdump` on `mnist-interop.exe`) rather than
/// guessed: its own steady-state body already keeps a handful of `zmm`
/// registers busy for operand traffic before this pass ever touches it.
const RESERVED_VECTOR_REGISTERS: i64 = 8;

/// How many architectural AVX-512 `zmm` registers a single copy of `ty`
/// needs to live in -- `1` for anything that isn't a vector type at all
/// (the isolated probe's own scalar `f32` accumulator), the real number of
/// 64-byte lanes otherwise (`vector<8x16xf32>`, this project's own real
/// accumulator shape: `8*16*4 = 512` bytes `= 8` registers, found directly
/// via disassembly -- see this module's own doc comment on the register-
/// spilling bug this constant exists to prevent).
///
/// Reaches past melior via raw FFI (`mlir_sys::mlirTypeIsAVector`/
/// `mlirShapedTypeGetRank`/`mlirShapedTypeGetDimSize`) the same way
/// `mlir_lower.rs::build_matmul_transpose_no_seed` already does elsewhere in
/// this project -- melior 0.27.4 has no safe `VectorType` wrapper to
/// introspect a vector's own shape with.
fn vector_register_count(ty: Type) -> i64 {
    let raw = ty.to_raw();
    if !unsafe { mlir_sys::mlirTypeIsAVector(raw) } {
        return 1;
    }
    let rank = unsafe { mlir_sys::mlirShapedTypeGetRank(raw) };
    let mut elements: i64 = 1;
    for dim in 0..rank {
        elements *= unsafe { mlir_sys::mlirShapedTypeGetDimSize(raw, dim as isize) };
    }
    // This project's own `mlir::` stdlib is f32-only throughout -- 4
    // bytes/element is not an assumption made only here.
    let bytes = elements * 4;
    (bytes + VECTOR_REGISTER_BYTES - 1) / VECTOR_REGISTER_BYTES
}

/// Below this many tile-iterations, leave the loop alone entirely — matches
/// this project's own already-measured finding that a short reduction (`K
/// <= ~64`, e.g. the batch-sized `K=32` weight-gradient matmuls) is already
/// efficient (IPC `1.7`-`3.4`, `doc/backlog.md`) without any of this; only a
/// long reduction (`K` in the hundreds) pays the narrow-accumulator tax
/// enough to be worth restructuring.
const MIN_TRIP_COUNT_TO_UNROLL: i64 = 8;

/// Runs the pass over every `scf.for` reduction anywhere in `module`,
/// unrolling-and-jamming each one whose own trip count is both large enough
/// (`MIN_TRIP_COUNT_TO_UNROLL`) and evenly divisible by some candidate
/// factor. Conservative by construction, matching this project's own
/// established posture for exactly this class of transform (`dps_rewrite.rs`
/// 's own module doc comment: "a single mismatch anywhere in the chain
/// leaves that one... completely untouched, falling back to the always-
/// correct path"): a loop this pass doesn't recognize, or can't find a
/// factor for, is left byte-for-byte as the existing schedule already built
/// it — this pass can only ever make a long reduction *faster*, never
/// *different*.
///
/// **Off by default, opt-in via `CLEAVE_UNROLL_JAM=1`** (`CLEAVE_AFFINE_
/// STRUCTS`/`CLEAVE_TAG_RELEASES`'s own established convention for a real,
/// working, but not-default-on mechanism) — real, JIT-proven correct
/// (`cleave/tests/unroll_jam_probe.rs`), but measured on the real kernel and
/// found not to be the actual fix for the long-K matmul IPC gap it was built
/// for (`doc/backlog.md`'s own "the long-K matmul IPC gap was never a
/// latency-chain problem, it was cache locality" entry has the full
/// measured story: widening the *outer* K-tile loop duplicates a real,
/// unavoidable `16`-register operand-tile load per copy, `factor=7` demands
/// `112` architectural registers against `32`, and the real fix turned out
/// to be a cache-locality one, an `M`-tile-size change elsewhere). Kept in
/// the tree rather than deleted: a correct, generalizable mechanism (real
/// multi-`iter_arg` handling, a real `vector.outerproduct` combine-op
/// recognition) that may be the right tool for a *different* shape later.
pub fn unroll_and_jam_reductions<'c>(context: &'c Context, module: &mut Module<'c>) {
    if std::env::var("CLEAVE_UNROLL_JAM").is_err() {
        return;
    }
    let trace = std::env::var("CLEAVE_TRACE_UNROLL_JAM").is_ok();
    let mut applied = 0usize;
    loop {
        let op = module.as_operation_mut();
        let Some((for_op, factor, reduction_idx)) = find_next_candidate(op) else {
            break;
        };
        if trace {
            eprintln!("CLEAVE_TRACE_UNROLL_JAM: applying factor={factor} reduction_idx={reduction_idx}");
        }
        unroll_and_jam_reduction(context, for_op, factor, reduction_idx);
        applied += 1;
    }
    if trace {
        eprintln!("CLEAVE_TRACE_UNROLL_JAM: {applied} reduction(s) rewritten");
    }
}

/// Finds the next `scf.for` anywhere in `op`'s own subtree that is both a
/// recognized reduction (`is_recognized_reduction`) and has a real, usable
/// unroll factor (`pick_factor`) — pre-order, innermost-first is not
/// required here the way the isolated probe's own `find_scf_for` needed it
/// (a non-reducing M/N-tiling loop is simply never a match at all, so the
/// walk never needs to distinguish "found the wrong scf.for" from "found
/// nothing yet" the way the probe's synthetic nesting test did).
///
/// One candidate at a time, not a single upfront collect-then-apply pass:
/// `unroll_and_jam_reduction` inserts new ops and erases the old one, which
/// would invalidate a `Vec` of `OperationRefMut`s collected before any of
/// them ran. `unroll_and_jam_reductions`'s own driving loop re-walks from
/// the module's own root after each one instead — a real cost (re-scanning
/// already-handled subtrees), accepted deliberately: this runs once, on a
/// module with at most a few dozen real matmul-family reductions, not
/// somewhere latency-sensitive.
fn find_next_candidate<'c, 'a>(
    op: OperationRefMut<'c, 'a>,
) -> Option<(OperationRefMut<'c, 'a>, i64, usize)> {
    if matches!(op.name().as_string_ref().as_str(), Ok("scf.for")) {
        let trace = std::env::var("CLEAVE_TRACE_UNROLL_JAM").is_ok();
        let reduction_idx = find_reduction_iter_arg(&op);
        if trace {
            let detail = op
                .region(0)
                .ok()
                .and_then(|r| r.first_block())
                .and_then(|b| b.terminator())
                .map(|t| t.to_string())
                .unwrap_or_default();
            eprintln!(
                "CLEAVE_TRACE_UNROLL_JAM: scf.for operand_count={} reduction_idx={reduction_idx:?} terminator={detail}",
                op.operand_count(),
            );
        }
        if let Some(reduction_idx) = reduction_idx {
            let acc_registers = op
                .operand(3 + reduction_idx)
                .map(|v| vector_register_count(v.r#type()))
                .unwrap_or(1);
            let factor = pick_factor(&op, acc_registers);
            if trace {
                eprintln!(
                    "CLEAVE_TRACE_UNROLL_JAM:   acc_registers={acc_registers} factor={factor:?}"
                );
            }
            if let Some(factor) = factor {
                return Some((op, factor, reduction_idx));
            }
        }
    }
    for region in op.regions() {
        let mut next_block = region.first_block();
        while let Some(block) = next_block {
            let mut next_op = block.first_operation_mut();
            while let Some(child) = next_op {
                next_op = child.next_in_block_mut();
                if let Some(found) = find_next_candidate(child) {
                    return Some(found);
                }
            }
            next_block = block.next_in_region();
        }
    }
    None
}

/// Which `iter_arg` position (0-based, among iter_args only -- not counting
/// `scf.for`'s own leading lower/upper/step operands) is the *real*
/// reduction — its own yielded value produced directly by `vector.
/// outerproduct` (found by tracing the real schedule, not assumed: `vectorize
/// {create_named_contraction}` itself produces `vector.contract`, but the
/// real schedule's own later contraction-lowering step, run as part of the
/// same transform sequence, already rewrites that into the `vfmadd132ps`-
/// producing `vector.outerproduct` chain before this pass ever runs -- so
/// `vector.contract` is checked too, for robustness against a future
/// schedule change, but never actually seen here), or a plain `arith.addf`
/// (kept for the isolated probe's own simpler shapes)? `None` if no position
/// matches, or more than one does (this first version only ever widens a
/// loop with exactly one genuine reduction; a loop with two would need two
/// independently-sized factor choices, not attempted here).
///
/// **Why a loop can have more than one `iter_arg` at all, found directly,
/// not assumed going in**: `--loop-invariant-subset-hoisting` (`pipeline.rs`
/// , runs right before this pass) turns the K-reduction's own accumulator
/// from a tensor round-tripped through `vector.transfer_read`/`write` every
/// iteration into a real, register-resident `vector<...>` value — but it
/// does this by *adding* a second `iter_arg` for the hoisted vector,
/// keeping the *original* tensor `iter_arg` as a now-trivial, unchanged
/// passthrough (still needed after the loop, for the real `vector.transfer_
/// write` that stores the final result back into the destination tensor).
/// Every other `iter_arg` position, therefore, is expected to be exactly
/// this kind of loop-invariant passthrough — carried through unchanged by
/// every unrolled copy alike, never duplicated per copy the way the real
/// accumulator is.
fn find_reduction_iter_arg(op: &OperationRefMut) -> Option<usize> {
    let num_iter_args = op.operand_count().checked_sub(3)?;
    if num_iter_args == 0 {
        return None;
    }
    let region = op.region(0).ok()?;
    let block = region.first_block()?;
    let terminator = block.terminator()?;
    if !matches!(terminator.name().as_string_ref().as_str(), Ok("scf.yield")) {
        return None;
    }
    let mut found = None;
    for i in 0..num_iter_args {
        let yielded = terminator.operand(i).ok()?;
        let Ok(result) = OperationResult::try_from(yielded) else {
            continue;
        };
        if matches!(
            result.owner().name().as_string_ref().as_str(),
            Ok("arith.addf") | Ok("vector.contract") | Ok("vector.outerproduct")
        ) {
            if found.is_some() {
                // More than one real reduction in the same loop -- not
                // attempted, conservatively skipped rather than guessed at.
                return None;
            }
            found = Some(i);
        }
    }
    found
}

/// A real, constant trip count, at least `MIN_TRIP_COUNT_TO_UNROLL`, evenly
/// divisible by some `CANDIDATE_FACTORS` entry that ALSO fits the real
/// register budget for an accumulator needing `acc_registers` registers each
/// (see this module's own doc comment for the measured spill/reload failure
/// this second cap exists to prevent) — the largest such factor, or `None`
/// if the loop is too short, has a non-constant bound, or no candidate both
/// divides it evenly and fits the register budget.
fn pick_factor(op: &OperationRefMut, acc_registers: i64) -> Option<i64> {
    let lower = const_index_value(op.operand(0).ok()?)?;
    let upper = const_index_value(op.operand(1).ok()?)?;
    let step = const_index_value(op.operand(2).ok()?)?;
    if step <= 0 {
        return None;
    }
    let trip_count = (upper - lower) / step;
    if trip_count < MIN_TRIP_COUNT_TO_UNROLL {
        return None;
    }
    let acc_registers = acc_registers.max(1);
    let budget = (TOTAL_VECTOR_REGISTERS - RESERVED_VECTOR_REGISTERS).max(acc_registers);
    let max_factor_by_registers = (budget / acc_registers).max(1);
    CANDIDATE_FACTORS
        .iter()
        .copied()
        .filter(|&factor| factor <= max_factor_by_registers)
        .find(|factor| trip_count % factor == 0)
}

fn const_index_value(v: Value) -> Option<i64> {
    let result = OperationResult::try_from(v).ok()?;
    let op = result.owner();
    if !matches!(op.name().as_string_ref().as_str(), Ok("arith.constant")) {
        return None;
    }
    let attr = op.attribute("value").ok()?;
    IntegerAttribute::try_from(attr).ok().map(|a| a.value())
}

/// Replaces every real use of `old` with `new`, anywhere in `root`'s own
/// subtree — a full-module walk, not just the immediately-following op
/// (the isolated probe's own simplification, safe there because its test
/// modules only ever had one consumer; the real kernel's K-reduction result
/// can be consumed anywhere downstream, including inside a different,
/// later-tiled loop entirely). Sound for plain SSA dominance reasons: `old`
/// can only ever appear as an operand of an op that comes *after* its own
/// definition, so walking the whole module and checking every op's own
/// operands is always safe, just not maximally cheap.
fn replace_all_uses<'c, 'a>(root: OperationRefMut<'c, 'a>, old: Value<'c, 'a>, new: Value<'c, 'a>) {
    fn walk<'c, 'a>(mut op: OperationRefMut<'c, 'a>, old: Value<'c, 'a>, new: Value<'c, 'a>) {
        if op
            .operands()
            .any(|operand| operand == old)
        {
            op.replace_uses_of_with(old, new);
        }
        for region in op.regions() {
            let mut next_block = region.first_block();
            while let Some(block) = next_block {
                let mut next_op = block.first_operation_mut();
                while let Some(child) = next_op {
                    next_op = child.next_in_block_mut();
                    walk(child, old, new);
                }
                next_block = block.next_in_region();
            }
        }
    }
    walk(root, old, new);
}

/// Unroll-and-jam `for_op` by `factor` -- `factor` independent accumulators,
/// combined into one value with `factor - 1` combine ops right after the
/// (rebuilt) loop. See this module's own doc comment for why this builds a
/// whole new `scf.for` rather than mutating the old one in place, and
/// `cleave/tests/unroll_jam_probe.rs` for the isolated, JIT-verified proof
/// this exact algorithm produces a numerically identical answer.
fn unroll_and_jam_reduction<'c>(
    context: &'c Context,
    mut for_op: OperationRefMut<'c, '_>,
    factor: i64,
    reduction_idx: usize,
) {
    let region = for_op.region(0).expect("scf.for has a body region");
    let old_block = region.first_block().expect("scf.for body has a block");
    let loc = for_op.location();

    let lower = for_op.operand(0).unwrap();
    let upper = for_op.operand(1).unwrap();
    let step = for_op.operand(2).unwrap();
    let step_c = const_index_value(step).expect("pick_factor already required a constant step");

    let num_iter_args = for_op.operand_count() - 3;
    let old_inits: Vec<Value> = (0..num_iter_args)
        .map(|i| for_op.operand(3 + i).unwrap())
        .collect();
    let old_iter_args: Vec<Value> = (0..num_iter_args)
        .map(|i| old_block.argument(1 + i).unwrap().into())
        .collect();
    let old_iv: Value = old_block.argument(0).unwrap().into();
    let old_terminator = old_block.terminator().expect("scf.for body has a terminator");
    let old_yields: Vec<Value> = (0..num_iter_args)
        .map(|i| old_terminator.operand(i).unwrap())
        .collect();

    // Every `iter_arg` position other than the real reduction is expected to
    // be a pure, loop-invariant passthrough (`find_reduction_iter_arg`'s own
    // doc comment) -- carried through as one shared new iter_arg, never
    // duplicated per unrolled copy the way the real accumulator is.
    let passthrough_indices: Vec<usize> =
        (0..num_iter_args).filter(|&i| i != reduction_idx).collect();
    let passthrough_count = passthrough_indices.len();

    let old_acc = old_iter_args[reduction_idx];
    let old_yielded = old_yields[reduction_idx];

    // Every real op in the old body, in order, excluding the terminator --
    // the block's own last op is always the terminator (`scf.for`'s body
    // always ends in `scf.yield`), so collecting everything and dropping
    // the last entry avoids needing an identity comparison on
    // `OperationRef` at all.
    let mut old_body_ops = Vec::new();
    let mut next_op = old_block.first_operation();
    while let Some(op) = next_op {
        next_op = op.next_in_block();
        old_body_ops.push(op);
    }
    old_body_ops.pop();

    let index_ty = old_iv.r#type();
    let acc_ty = old_acc.r#type();
    let passthrough_tys: Vec<_> = passthrough_indices
        .iter()
        .map(|&i| old_iter_args[i].r#type())
        .collect();

    let mut arg_types = vec![(index_ty, loc)];
    for ty in &passthrough_tys {
        arg_types.push((*ty, loc));
    }
    for _ in 0..factor {
        arg_types.push((acc_ty, loc));
    }
    let new_block = Block::new(&arg_types);
    let new_iv: Value = new_block.argument(0).unwrap().into();
    let new_passthroughs: Vec<Value> = (0..passthrough_count)
        .map(|i| new_block.argument(1 + i).unwrap().into())
        .collect();
    let new_accs: Vec<Value> = (0..factor)
        .map(|i| new_block.argument(1 + passthrough_count + i as usize).unwrap().into())
        .collect();

    let mut final_acc_updates = Vec::new();
    let mut final_passthrough_updates: Vec<Option<Value>> = vec![None; passthrough_count];
    for copy in 0..factor {
        let iv_copy = if copy == 0 {
            new_iv
        } else {
            let offset_attr = IntegerAttribute::new(index_ty, copy * step_c);
            let offset_op = OperationBuilder::new("arith.constant", loc)
                .add_attributes(&[(Identifier::new(context, "value"), offset_attr.into())])
                .add_results(&[index_ty])
                .build()
                .expect("failed to build offset constant");
            let offset_op = new_block.append_operation(offset_op);
            let offset_val: Value = offset_op.result(0).unwrap().into();
            let add_op = OperationBuilder::new("arith.addi", loc)
                .add_operands(&[new_iv, offset_val])
                .add_results(&[index_ty])
                .build()
                .expect("failed to build shifted iv");
            let add_op = new_block.append_operation(add_op);
            add_op.result(0).unwrap().into()
        };

        let mut remap: Vec<(Value, Value)> =
            vec![(old_iv, iv_copy), (old_acc, new_accs[copy as usize])];
        for (j, &idx) in passthrough_indices.iter().enumerate() {
            remap.push((old_iter_args[idx], new_passthroughs[j]));
        }
        for op in &old_body_ops {
            // A real, independent `mlirOperationClone` -- `OperationRef`'s
            // own derived `Clone` would just copy the *reference* (the same
            // handle), not the operation itself. Calling the C API directly
            // mirrors `mlir_lower.rs::build_matmul_transpose_no_seed`'s own
            // precedent for reaching past melior when it has no safe
            // wrapper for something the C API itself does support.
            let cloned_raw = unsafe { mlir_sys::mlirOperationClone(op.to_raw()) };
            let cloned: melior::ir::operation::Operation =
                unsafe { melior::ir::operation::Operation::from_raw(cloned_raw) };
            for i in 0..cloned.operand_count() {
                let operand = cloned.operand(i).unwrap();
                if let Some((_, new_val)) = remap.iter().find(|(old, _)| *old == operand) {
                    let raw = cloned.to_raw();
                    let mut m = unsafe { OperationRefMut::from_raw(raw) };
                    m.set_operand(i, *new_val);
                }
            }
            let had_result = cloned.result_count() > 0;
            let old_result: Option<Value> = if had_result {
                Some(op.result(0).unwrap().into())
            } else {
                None
            };
            let inserted = new_block.append_operation(cloned);
            if let Some(old_result) = old_result {
                let new_result: Value = inserted.result(0).unwrap().into();
                remap.push((old_result, new_result));
            }
        }

        let mapped_acc_yield = remap
            .iter()
            .rev()
            .find(|(old, _)| *old == old_yielded)
            .map(|(_, new)| *new)
            .expect("the yielded reduction value must have been produced by a cloned op");
        final_acc_updates.push(mapped_acc_yield);

        // Every copy computes the identical passthrough value (it is, by
        // construction, loop-invariant) -- resolve each one only once, from
        // whichever copy happens to produce it first.
        for (j, &idx) in passthrough_indices.iter().enumerate() {
            if final_passthrough_updates[j].is_none() {
                let mapped = remap
                    .iter()
                    .rev()
                    .find(|(old, _)| *old == old_yields[idx])
                    .map(|(_, new)| *new)
                    .expect("the yielded passthrough value must have been produced or remapped");
                final_passthrough_updates[j] = Some(mapped);
            }
        }
    }
    let final_passthrough_updates: Vec<Value> = final_passthrough_updates
        .into_iter()
        .map(|v| v.expect("every passthrough is resolved by the first unrolled copy"))
        .collect();

    let mut yield_operands = final_passthrough_updates.clone();
    yield_operands.extend(final_acc_updates.iter().copied());
    let new_yield = OperationBuilder::new("scf.yield", loc)
        .add_operands(&yield_operands)
        .build()
        .expect("failed to build the new scf.yield");
    new_block.append_operation(new_yield);

    let new_region = Region::new();
    new_region.append_block(new_block);

    let new_step_attr = IntegerAttribute::new(index_ty, step_c * factor);
    let new_step_op = OperationBuilder::new("arith.constant", loc)
        .add_attributes(&[(Identifier::new(context, "value"), new_step_attr.into())])
        .add_results(&[index_ty])
        .build()
        .expect("failed to build the new step constant");

    let parent_block = for_op.block().expect("scf.for has a parent block");
    let for_op_ref: OperationRef = unsafe { OperationRef::from_raw(for_op.to_raw()) };
    let new_step_op = parent_block.insert_operation_before(for_op_ref, new_step_op);
    let new_step_val: Value = new_step_op.result(0).unwrap().into();

    let mut new_for_operands = vec![lower, upper, new_step_val];
    for &idx in &passthrough_indices {
        new_for_operands.push(old_inits[idx]);
    }
    let acc_init = old_inits[reduction_idx];
    for _ in 0..factor {
        new_for_operands.push(acc_init);
    }

    let mut result_types = passthrough_tys.clone();
    result_types.extend(std::iter::repeat(acc_ty).take(factor as usize));

    let new_for = OperationBuilder::new("scf.for", loc)
        .add_operands(&new_for_operands)
        .add_results(&result_types)
        .add_regions_vec(vec![new_region])
        .build()
        .expect("failed to build the new scf.for");
    let new_for = parent_block.insert_operation_before(for_op_ref, new_for);

    let mut combined: Value = new_for.result(passthrough_count).unwrap().into();
    let mut insert_after: OperationRef = new_for;
    for k in 1..factor {
        let rhs: Value = new_for.result(passthrough_count + k as usize).unwrap().into();
        let add_op = OperationBuilder::new("arith.addf", loc)
            .add_operands(&[combined, rhs])
            .add_results(&[acc_ty])
            .build()
            .expect("failed to build combining addf");
        let add_op = parent_block.insert_operation_after(insert_after, add_op);
        combined = add_op.result(0).unwrap().into();
        insert_after = add_op;
    }

    // Rewire every real use of each of the old loop's own results -- a
    // full-module walk per result (`replace_all_uses`), not just the
    // isolated probe's own "the very next op" simplification, since the real
    // kernel's own consumers can sit anywhere downstream. Each passthrough
    // result maps onto its own single new result unchanged; the real
    // reduction's own result maps onto the newly combined value.
    let module_root_raw = {
        let mut current = unsafe { OperationRefMut::from_raw(for_op.to_raw()) };
        loop {
            match current.parent_operation_mut() {
                Some(parent) => current = parent,
                None => break current.to_raw(),
            }
        }
    };
    for (j, &idx) in passthrough_indices.iter().enumerate() {
        let old_result: Value = for_op.result(idx).unwrap().into();
        let new_result: Value = new_for.result(j).unwrap().into();
        let root = unsafe { OperationRefMut::from_raw(module_root_raw) };
        replace_all_uses(root, old_result, new_result);
    }
    let old_reduction_result: Value = for_op.result(reduction_idx).unwrap().into();
    let root = unsafe { OperationRefMut::from_raw(module_root_raw) };
    replace_all_uses(root, old_reduction_result, combined);

    // `remove_from_parent` only *detaches* -- it does not destroy (melior's
    // own doc comment on it: "not destroyed, only unlinked"). Left alone,
    // `for_op` would leak: a real, live, orphaned operation, which a later
    // pass (canonicalize/CSE) can trip over the exact way this project's
    // own `redundant_copy_elim.rs` already did once (`doc/backlog.md`'s own
    // `--cse` entry) -- reclaim real ownership immediately via
    // `Operation::from_raw` and let it drop, which is what actually calls
    // `mlirOperationDestroy`.
    for_op.remove_from_parent();
    let raw = for_op.to_raw();
    drop(unsafe { melior::ir::operation::Operation::from_raw(raw) });
}
