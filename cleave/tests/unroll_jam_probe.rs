//! Isolated probe, deliberately outside `mlir_lower.rs`/`pipeline.rs` --
//! proves the "unroll a vectorized reduction `scf.for` into N independent
//! accumulators, combine once at the end" mechanism (`doc/backlog.md`'s own
//! matmul-IPC investigation) works via melior's real API, on a minimal,
//! hand-built module, before touching the real matmul pipeline at all.
//! MLIR's own `transform.loop.unroll_and_jam` can't do this directly for
//! `scf.for` (confirmed by reading `mlir/lib/Dialect/SCF/Utils/Utils.cpp`
//! directly: it rejects any loop with results) -- this is the hand-written
//! Rust equivalent of the *Affine* dialect's own complete version of the
//! same algorithm (`mlir/lib/Dialect/Affine/Utils/LoopUtils.cpp`), which
//! does handle a top-level reduction loop, including the final combining
//! step -- ported by hand since melior has no `IRMapping`/`replaceWith
//! AdditionalYields` equivalent.
//!
//! **The one structural difference from a straight port**: `scf.for` (like
//! every MLIR op) has a fixed result count once built -- there is no way to
//! "add a result" to the existing op in place. So instead of growing the
//! *same* op (what `replaceWithAdditionalYields` does in C++, via a real
//! MLIR-internal utility with no melior equivalent), this builds a brand
//! *new* `scf.for` with `factor` iter_args/results from scratch, with a new
//! body block cloned `factor` times inside, then replaces the one real
//! external use of the old loop's single result and erases the old op.

use melior::Context;
use melior::dialect::DialectRegistry;
use melior::ir::block::BlockLike;
use melior::ir::operation::{OperationBuilder, OperationLike, OperationMutLike, OperationRef, OperationResult};
use melior::ir::{Block, Identifier, Module, Region, RegionLike, Value, ValueLike};
use melior::ir::attribute::IntegerAttribute;
use melior::pass;
use melior::utility::register_all_dialects;

fn context() -> Context {
    let dialect_registry = DialectRegistry::new();
    register_all_dialects(&dialect_registry);
    let context = Context::new();
    context.append_dialect_registry(&dialect_registry);
    context.load_all_available_dialects();
    context
}

/// `%i_f32 = index_cast+sitofp %i; %b = broadcast %i_f32; %acc = addf %acc, %b`
/// per iteration, tiled by 16, over `0..256` -- accumulated into a
/// `vector<16xf32>`, horizontally summed to one `f32` at the end. A real,
/// induction-variable-dependent computation (not a constant), specifically
/// so a bug in remapping the IV for each unrolled copy would change the
/// answer, not silently disappear.
const SOURCE: &str = r#"
module {
  func.func @accumulate() -> f32 attributes { llvm.emit_c_interface } {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %zero = arith.constant dense<0.0> : vector<16xf32>
    %result = scf.for %i = %c0 to %c256 step %c16 iter_args(%acc = %zero) -> (vector<16xf32>) {
      %i_i32 = arith.index_cast %i : index to i32
      %i_f32 = arith.sitofp %i_i32 : i32 to f32
      %b = vector.broadcast %i_f32 : f32 to vector<16xf32>
      %new_acc = arith.addf %acc, %b : vector<16xf32>
      scf.yield %new_acc : vector<16xf32>
    }
    %sum = vector.reduction <add>, %result : vector<16xf32> into f32
    return %sum : f32
  }
}
"#;

/// Expected result computed independently in Rust, not by hand: 16
/// iterations (`256 / 16`), `i = 0, 16, 32, ..., 240`, each contributing
/// `16 * i` to the horizontal sum (broadcast to all 16 lanes, then reduced).
fn expected_sum() -> f32 {
    let mut acc = 0.0f32;
    let mut i = 0;
    while i < 256 {
        acc += 16.0 * (i as f32);
        i += 16;
    }
    acc
}

fn lower_to_llvm_and_run(context: &Context, module: &mut Module) -> f32 {
    assert!(module.as_operation().verify(), "module failed verification");

    let pm = pass::PassManager::new(context);
    pm.add_pass(pass::conversion::create_scf_to_control_flow());
    pm.add_pass(pass::conversion::create_vector_to_llvm());
    pm.add_pass(pass::conversion::create_to_llvm());
    pm.add_pass(pass::conversion::create_finalize_mem_ref_to_llvm());
    pm.add_pass(pass::conversion::create_to_llvm());
    pm.add_pass(pass::conversion::create_reconcile_unrealized_casts());
    // Real, found-by-testing requirement, not copied defensively: without
    // this, three dead `builtin.unrealized_conversion_cast`s (an `i64`-typed
    // constant materialized back to `index`, then never used again once its
    // *other* uses had already been converted directly) survive all the way
    // to the final module -- `reconcile-unrealized-casts` only ever merges
    // round-trip cast *pairs*, never removes a one-way, zero-use cast, and
    // `ExecutionEngine`'s own LLVM translation refuses the module outright
    // the moment even one such dead op remains (`builtin.unrealized_
    // conversion_cast` has no real LLVM lowering at all).
    pm.add_pass(pass::transform::create_canonicalizer());
    pm.run(module).expect("lowering to llvm must succeed");
    assert!(
        module.as_operation().verify(),
        "module failed verification after lowering:\n{}",
        module.as_operation()
    );

    let engine = melior::ExecutionEngine::new(module, 2, &[], false, false);
    let mut result: f32 = -1.0;
    unsafe {
        engine
            .invoke_packed("accumulate", &mut [&mut result as *mut f32 as *mut ()])
            .unwrap_or_else(|e| panic!("JIT invocation failed: {e:?}"));
    }
    result
}

#[test]
fn baseline_serial_reduction_computes_the_right_value() {
    let context = context();
    let mut module = Module::parse(&context, SOURCE).expect("failed to parse probe module");
    let got = lower_to_llvm_and_run(&context, &mut module);
    assert_eq!(got, expected_sum());
}

/// Does `op` (an `scf.for`) have exactly the shape `unroll_and_jam_reduction`
/// itself already requires -- a real terminator yielding a value produced by
/// a recognized combine op? Kept in sync with that function's own two
/// combine-op names deliberately, by literal duplication rather than a
/// shared helper -- this is a probe, and the two call sites checking the
/// same two strings is a feature here (a real drift between them would fail
/// loudly, as a lookup miss, not silently).
fn is_recognized_reduction(op: &melior::ir::operation::OperationRefMut) -> bool {
    let Some(region) = op.region(0).ok() else {
        return false;
    };
    let Some(block) = region.first_block() else {
        return false;
    };
    let Some(terminator) = block.terminator() else {
        return false;
    };
    if !matches!(terminator.name().as_string_ref().as_str(), Ok("scf.yield")) {
        return false;
    }
    let Ok(yielded) = terminator.operand(0) else {
        return false;
    };
    let Ok(result) = OperationResult::try_from(yielded) else {
        return false;
    };
    matches!(
        result.owner().name().as_string_ref().as_str(),
        Ok("arith.addf") | Ok("vector.contract")
    )
}

/// Finds the *innermost* `scf.for` whose own body is a recognized reduction
/// (its terminator yields a value produced by `arith.addf`/`vector.contract`)
/// -- not just the first `scf.for` encountered. Needed the moment a real M/N
/// tiling loop (itself a plain, non-reducing `scf.for`/`scf.forall`, exactly
/// what wraps the real K-reduction in the actual schedule) sits *outside*
/// the loop this pass actually wants -- a real nesting shape this probe's
/// original, simpler walk never had to distinguish.
fn find_scf_for<'c, 'a>(
    op: melior::ir::operation::OperationRefMut<'c, 'a>,
) -> Option<melior::ir::operation::OperationRefMut<'c, 'a>> {
    if matches!(op.name().as_string_ref().as_str(), Ok("scf.for")) && is_recognized_reduction(&op) {
        return Some(op);
    }
    for region in op.regions() {
        let mut next_block = region.first_block();
        while let Some(block) = next_block {
            let mut next_op = block.first_operation_mut();
            while let Some(child) = next_op {
                next_op = child.next_in_block_mut();
                if let Some(found) = find_scf_for(child) {
                    return Some(found);
                }
            }
            next_block = block.next_in_region();
        }
    }
    None
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

/// Unroll-and-jam `for_op` (an `scf.for` with exactly one `iter_arg`, a real
/// `arith.addf` reduction) by `factor`: `factor` independent accumulators,
/// combined into one value with `factor - 1` `arith.addf`s right after the
/// (rebuilt) loop. See this file's own module doc comment for why this
/// builds a whole new `scf.for` rather than mutating the old one in place.
fn unroll_and_jam_reduction<'c>(
    context: &'c Context,
    mut for_op: melior::ir::operation::OperationRefMut<'c, '_>,
    factor: i64,
) {
    let region = for_op.region(0).expect("scf.for has a body region");
    let old_block = region.first_block().expect("scf.for body has a block");
    let loc = for_op.location();

    let lower = for_op.operand(0).unwrap();
    let upper = for_op.operand(1).unwrap();
    let step = for_op.operand(2).unwrap();
    let init = for_op.operand(3).unwrap();

    let step_c = const_index_value(step).expect("probe only handles a constant step");
    let lower_c = const_index_value(lower).expect("probe only handles a constant lower bound");
    let upper_c = const_index_value(upper).expect("probe only handles a constant upper bound");
    let trip_count = (upper_c - lower_c) / step_c;
    assert!(
        trip_count % factor == 0,
        "probe only handles a trip count evenly divisible by the unroll factor"
    );

    let old_iv: Value = old_block.argument(0).unwrap().into();
    let old_acc: Value = old_block.argument(1).unwrap().into();
    let old_terminator = old_block.terminator().expect("scf.for body has a terminator");
    assert!(
        matches!(old_terminator.name().as_string_ref().as_str(), Ok("scf.yield")),
        "expected the body's own terminator to be scf.yield"
    );
    let old_yielded = old_terminator.operand(0).unwrap();
    let combine_owner = OperationResult::try_from(old_yielded)
        .expect("the yielded value must be a real op's result")
        .owner();
    assert!(
        matches!(
            combine_owner.name().as_string_ref().as_str(),
            Ok("arith.addf") | Ok("vector.contract")
        ),
        "probe only recognizes a plain arith.addf or vector.contract combine"
    );

    // Every real op in the old body, in order, excluding the terminator --
    // what gets cloned `factor` times into the new body (once per
    // accumulator copy, including copy 0). The block's own last op is
    // always the terminator (`scf.for`'s body always ends in `scf.yield`),
    // so collecting everything and dropping the last entry avoids needing
    // an identity comparison on `OperationRef` at all.
    let mut old_body_ops = Vec::new();
    let mut next_op = old_block.first_operation();
    while let Some(op) = next_op {
        next_op = op.next_in_block();
        old_body_ops.push(op);
    }
    let popped = old_body_ops.pop();
    debug_assert!(
        popped.map(|op| matches!(op.name().as_string_ref().as_str(), Ok("scf.yield"))) == Some(true)
    );

    let index_ty = old_iv.r#type();
    let acc_ty = old_acc.r#type();

    // New body block: `iv, acc_0, acc_1, ..., acc_{factor-1}`.
    let mut arg_types = vec![(index_ty, loc)];
    for _ in 0..factor {
        arg_types.push((acc_ty, loc));
    }
    let new_block = Block::new(&arg_types);
    let new_iv: Value = new_block.argument(0).unwrap().into();
    let new_accs: Vec<Value> = (0..factor)
        .map(|i| new_block.argument(1 + i as usize).unwrap().into())
        .collect();

    let mut final_updates = Vec::new();
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

        // old value -> new value, seeded with this copy's own iv/accumulator.
        let mut remap: Vec<(Value, Value)> = vec![(old_iv, iv_copy), (old_acc, new_accs[copy as usize])];
        for op in &old_body_ops {
            // A real, independent `mlirOperationClone` -- `OperationRef`'s
            // own derived `Clone` would just copy the *reference* (the same
            // handle), not the operation itself; `Operation`'s own `Clone`
            // impl exists for exactly this but needs an owned `Operation`
            // to start from, which a mere reference into someone else's
            // block never is. Calling the C API directly here mirrors
            // `build_matmul_transpose_no_seed`'s own precedent
            // (`mlir_lower.rs`) for reaching past melior when it has no
            // safe wrapper for something the C API itself does support.
            let cloned_raw = unsafe { mlir_sys::mlirOperationClone(op.to_raw()) };
            let cloned: melior::ir::operation::Operation =
                unsafe { melior::ir::operation::Operation::from_raw(cloned_raw) };
            for i in 0..cloned.operand_count() {
                let operand = cloned.operand(i).unwrap();
                if let Some((_, new_val)) = remap.iter().find(|(old, _)| *old == operand) {
                    // `set_operand` needs `OperationMutLike`, reachable via a
                    // raw round-trip since `cloned` is a plain owned
                    // `Operation` here, not yet inserted into any block.
                    let raw = cloned.to_raw();
                    let mut m = unsafe { melior::ir::operation::OperationRefMut::from_raw(raw) };
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

        let mapped_yield = remap
            .iter()
            .rev()
            .find(|(old, _)| *old == old_yielded)
            .map(|(_, new)| *new)
            .expect("the yielded value must have been produced by a cloned op");
        final_updates.push(mapped_yield);
    }

    let new_yield = OperationBuilder::new("scf.yield", loc)
        .add_operands(&final_updates)
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
    for _ in 0..factor {
        new_for_operands.push(init);
    }
    let new_for = OperationBuilder::new("scf.for", loc)
        .add_operands(&new_for_operands)
        .add_results(&vec![acc_ty; factor as usize])
        .add_regions_vec(vec![new_region])
        .build()
        .expect("failed to build the new scf.for");
    let new_for = parent_block.insert_operation_before(for_op_ref, new_for);

    let mut combined: Value = new_for.result(0).unwrap().into();
    let mut insert_after: melior::ir::operation::OperationRef = new_for;
    for k in 1..factor {
        let rhs: Value = new_for.result(k as usize).unwrap().into();
        let add_op = OperationBuilder::new("arith.addf", loc)
            .add_operands(&[combined, rhs])
            .add_results(&[acc_ty])
            .build()
            .expect("failed to build combining addf");
        let add_op = parent_block.insert_operation_after(insert_after, add_op);
        combined = add_op.result(0).unwrap().into();
        insert_after = add_op;
    }

    // The old loop's single result has exactly one real consumer in this
    // probe (`vector.reduction`, the very next op) -- rewire it directly,
    // matching this probe's own known, minimal shape rather than a fully
    // general def-use walk.
    let old_result: Value = for_op.result(0).unwrap().into();
    let mut consumer = for_op
        .next_in_block_mut()
        .expect("the loop's own result must have a real consumer immediately after it");
    consumer.replace_uses_of_with(old_result, combined);

    // `remove_from_parent` only *detaches* -- it does not destroy (melior's
    // own doc comment on it says so explicitly: "not destroyed, only
    // unlinked", matching `mlirOperationRemoveFromParent`'s own C API
    // contract). Left alone, `for_op` would leak: a real, live, orphaned
    // operation, holding onto its own operands' uses forever, never
    // reclaimed by anything -- exactly the class of bug this project's own
    // `redundant_copy_elim.rs` already hit and fixed once (`doc/backlog.md`,
    // "`--cse` was missing..." entry). The fix is the same: reclaim real
    // ownership immediately via `Operation::from_raw` and let it drop --
    // `Operation`'s own `Drop` impl is what actually calls
    // `mlirOperationDestroy`.
    for_op.remove_from_parent();
    let raw = for_op.to_raw();
    drop(unsafe { melior::ir::operation::Operation::from_raw(raw) });
}

#[test]
fn unroll_jam_by_factor_4_gives_the_same_answer() {
    let context = context();
    let mut module = Module::parse(&context, SOURCE).expect("failed to parse probe module");
    {
        let op = module.as_operation_mut();
        let for_op = find_scf_for(op).expect("probe module must contain a scf.for");
        unroll_and_jam_reduction(&context, for_op, 4);
    }
    assert!(
        module.as_operation().verify(),
        "module failed verification after unroll-and-jam:\n{}",
        module.as_operation()
    );
    let got = lower_to_llvm_and_run(&context, &mut module);
    assert_eq!(got, expected_sum());
}

/// Widened, closer to the *real* matmul K-reduction shape: the per-iteration
/// combine is a real `vector.contract` (`acc = sum_j(lhs[j]*rhs[j]) + acc`,
/// the exact op `vectorize {create_named_contraction}` produces in the real
/// schedule, `cleave/mlir/matmul_vectorize.transform.mlir`) rather than this
/// file's own earlier, simpler `arith.addf`+`vector.broadcast` stand-in --
/// and the accumulator itself is a bare scalar `f32`, not a `vector<16xf32>`,
/// proving the pass doesn't quietly depend on the accumulator being a vector
/// type either. `%rhs` is a constant `dense<1.0>`, so each iteration
/// contributes `16 * i_f32` to the running sum -- the identical
/// `expected_sum()` this file already established for the simpler case,
/// reused as-is rather than re-derived.
const SOURCE_CONTRACT: &str = r#"
module {
  func.func @accumulate() -> f32 attributes { llvm.emit_c_interface } {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %zero = arith.constant 0.0 : f32
    %ones = arith.constant dense<1.0> : vector<16xf32>
    %result = scf.for %i = %c0 to %c256 step %c16 iter_args(%acc = %zero) -> (f32) {
      %i_i32 = arith.index_cast %i : index to i32
      %i_f32 = arith.sitofp %i_i32 : i32 to f32
      %lhs = vector.broadcast %i_f32 : f32 to vector<16xf32>
      %new_acc = vector.contract {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
        iterator_types = ["reduction"],
        kind = #vector.kind<add>
      } %lhs, %ones, %acc : vector<16xf32>, vector<16xf32> into f32
      scf.yield %new_acc : f32
    }
    return %result : f32
  }
}
"#;

#[test]
fn baseline_vector_contract_reduction_computes_the_right_value() {
    let context = context();
    let mut module = Module::parse(&context, SOURCE_CONTRACT).expect("failed to parse probe module");
    let got = lower_to_llvm_and_run(&context, &mut module);
    assert_eq!(got, expected_sum());
}

#[test]
fn unroll_jam_a_vector_contract_reduction_by_factor_4_gives_the_same_answer() {
    let context = context();
    let mut module = Module::parse(&context, SOURCE_CONTRACT).expect("failed to parse probe module");
    {
        let op = module.as_operation_mut();
        let for_op = find_scf_for(op).expect("probe module must contain a scf.for");
        unroll_and_jam_reduction(&context, for_op, 4);
    }
    assert!(
        module.as_operation().verify(),
        "module failed verification after unroll-and-jam:\n{}",
        module.as_operation()
    );
    let got = lower_to_llvm_and_run(&context, &mut module);
    assert_eq!(got, expected_sum());
}

/// Widened again: the real K-reduction never sits at a function's own top
/// level -- it's nested inside the M/N tiling loops the schedule already
/// built (`transform.structured.tile_using_forall`/`tile_using_for`,
/// `cleave/mlir/matmul_vectorize.transform.mlir`), themselves plain,
/// non-reducing loops. This wraps the exact same K-reduction body in a
/// trivial (trip-count-1) outer `scf.for` -- enough to prove `for_op.block()`
/// correctly resolves to the *inner* loop's own enclosing block (the outer
/// loop's own body block, not the function's top-level entry block) rather
/// than silently assuming one, without needing to also model a real,
/// multi-iteration outer tiling dimension to prove the point.
const SOURCE_NESTED: &str = r#"
module {
  func.func @accumulate() -> f32 attributes { llvm.emit_c_interface } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant dense<0.0> : vector<16xf32>
    %m_result = scf.for %m = %c0 to %c1 step %c1 iter_args(%outer_acc = %zero) -> (vector<16xf32>) {
      %c16 = arith.constant 16 : index
      %c256 = arith.constant 256 : index
      %inner_result = scf.for %i = %c0 to %c256 step %c16 iter_args(%acc = %zero) -> (vector<16xf32>) {
        %i_i32 = arith.index_cast %i : index to i32
        %i_f32 = arith.sitofp %i_i32 : i32 to f32
        %b = vector.broadcast %i_f32 : f32 to vector<16xf32>
        %new_acc = arith.addf %acc, %b : vector<16xf32>
        scf.yield %new_acc : vector<16xf32>
      }
      scf.yield %inner_result : vector<16xf32>
    }
    %sum = vector.reduction <add>, %m_result : vector<16xf32> into f32
    return %sum : f32
  }
}
"#;

#[test]
fn baseline_nested_reduction_computes_the_right_value() {
    let context = context();
    let mut module = Module::parse(&context, SOURCE_NESTED).expect("failed to parse probe module");
    let got = lower_to_llvm_and_run(&context, &mut module);
    assert_eq!(got, expected_sum());
}

#[test]
fn unroll_jam_the_inner_loop_of_a_nested_reduction_gives_the_same_answer() {
    let context = context();
    let mut module = Module::parse(&context, SOURCE_NESTED).expect("failed to parse probe module");
    {
        let op = module.as_operation_mut();
        let for_op = find_scf_for(op).expect("probe module must contain a recognized reduction scf.for");
        // Confirm this really did find the *inner* loop, not the trivial
        // outer one -- both are named `scf.for`, only one has the right
        // shape, and a regression here (matching the outer loop instead)
        // would otherwise fail much later, confusingly, inside the pass
        // itself rather than right where the real mistake would be.
        assert_eq!(
            const_index_value(for_op.operand(1).unwrap()),
            Some(256),
            "expected to find the inner (upper bound 256) loop, not the outer trivial one"
        );
        unroll_and_jam_reduction(&context, for_op, 4);
    }
    assert!(
        module.as_operation().verify(),
        "module failed verification after unroll-and-jam:\n{}",
        module.as_operation()
    );
    let got = lower_to_llvm_and_run(&context, &mut module);
    assert_eq!(got, expected_sum());
}
