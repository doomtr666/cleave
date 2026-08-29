//! Real, JIT-executed tests for `cleave::dps_rewrite::
//! eliminate_redundant_field_store_copies` -- deliberately *not* sharing
//! `mlir_lower.rs`'s own `run_i32_from_cps` helper (that one's pipeline
//! predates `--inline`/`--linalg-fuse-elementwise-ops` entirely, and 125+
//! existing tests already depend on it staying exactly as it is) -- this
//! file builds its own small pipeline instead, matching `pipeline.rs::
//! emit_object`'s real stage order up through the point this rewrite runs,
//! then continues to a real JIT invocation so every test here checks an
//! *actual computed value*, not just "the verifier didn't complain."
//!
//! Every test here uses a plain elementwise op (`Ring::sub`), never
//! `matmul` -- found the hard way, not by design up front: `linalg.matmul`
//! genuinely *reads* its own `outs` operand as a reduction accumulator, so
//! it's deliberately excluded from the rewrite entirely (`dps_rewrite.rs`'s
//! own doc comment on that exact check has the full story).

use cleave::cps::{collect_mlir_types, collect_struct_schemas, collect_units, convert_program};
use cleave::driver::compile;
use cleave::dps_rewrite::eliminate_redundant_field_store_copies;
use cleave::mlir_lower::lower_program;
use cleave::pipeline::check_type_errors;
use cleave::registry::Registry;
use melior::Context;
use melior::dialect::DialectRegistry;
use melior::ir::operation::OperationLike;
use melior::pass;
use melior::utility::{parse_pass_pipeline, register_all_dialects};

fn context() -> Context {
    let dialect_registry = DialectRegistry::new();
    register_all_dialects(&dialect_registry);
    let context = Context::new();
    context.append_dialect_registry(&dialect_registry);
    context.load_all_available_dialects();
    context
}

/// Lowers `src`, runs `--inline`/`--convert-elementwise-to-linalg`/
/// `--linalg-fuse-elementwise-ops` (the real pipeline's own prerequisite for
/// this rewrite -- see `dps_rewrite.rs`'s own module doc comment) and the
/// rewrite itself, then continues through bufferization and buffer-
/// deallocation, returning the printed text at that point -- for checking
/// what the rewrite's *effect after bufferization* actually looks like (in
/// particular, that the still-present-but-unerased copy tail's own size
/// operand becomes a compile-time zero -- `dps_rewrite
/// .rs`'s own module doc comment on why erasure is skipped has the story).
fn bufferized_text(context: &Context, src: &str) -> String {
    let (result, _sources) = compile(vec![("test.cleave".to_string(), src.to_string())], &[]);
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    if let Err(diags) = check_type_errors(&program, &registry) {
        panic!("type check failed: {diags:?}");
    }
    let units = collect_units(&program, &registry);
    let cps_program = convert_program(units);
    let mlir_types = collect_mlir_types(&program);
    let struct_schemas = collect_struct_schemas(&program);
    let mut module = lower_program(context, &cps_program, &mlir_types, struct_schemas);
    assert!(module.as_operation().verify());

    let pass_manager = pass::PassManager::new(context);
    pass_manager.add_pass(pass::transform::create_inliner());
    pass_manager.add_pass(pass::linalg::create_convert_elementwise_to_linalg_pass());
    pass_manager.add_pass(pass::linalg::create_linalg_elementwise_op_fusion_pass());
    pass_manager
        .run(&mut module)
        .expect("inline/elementwise-to-linalg/fuse must succeed");

    eliminate_redundant_field_store_copies(context, &mut module);

    let pass_manager = pass::PassManager::new(context);
    pass::bufferization::register_one_shot_bufferize_pass();
    parse_pass_pipeline(
        pass_manager.as_operation_pass_manager(),
        "builtin.module(one-shot-bufferize{bufferize-function-boundaries=true})",
    )
    .expect("failed to parse the one-shot-bufferize pass pipeline");
    pass_manager
        .run(&mut module)
        .expect("one-shot-bufferize must succeed");

    let pass_manager = pass::PassManager::new(context);
    pass_manager.add_pass(pass::bufferization::create_ownership_based_buffer_deallocation_pass());
    pass_manager.add_pass(pass::bufferization::create_buffer_deallocation_simplification_pass());
    pass_manager.add_pass(pass::bufferization::create_lower_deallocations_pass());
    pass_manager.add_pass(pass::conversion::create_bufferization_to_mem_ref());
    pass_manager
        .run(&mut module)
        .expect("buffer-deallocation must succeed");

    module.as_operation().to_string()
}

/// Like `lower_and_rewrite`, but continues all the way to a real JIT
/// invocation of `main() -> f32` -- the decisive check: the rewrite must
/// not just leave a verifiable module behind, it must compute the *same
/// value* the unrewritten path already does.
fn run_f32_with_rewrite(context: &Context, src: &str) -> f32 {
    let (result, _sources) = compile(vec![("test.cleave".to_string(), src.to_string())], &[]);
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    if let Err(diags) = check_type_errors(&program, &registry) {
        panic!("type check failed: {diags:?}");
    }
    let units = collect_units(&program, &registry);
    let cps_program = convert_program(units);
    let mlir_types = collect_mlir_types(&program);
    let struct_schemas = collect_struct_schemas(&program);
    let mut module = lower_program(context, &cps_program, &mlir_types, struct_schemas);
    assert!(module.as_operation().verify());

    let pass_manager = pass::PassManager::new(context);
    pass_manager.add_pass(pass::transform::create_inliner());
    pass_manager.add_pass(pass::linalg::create_convert_elementwise_to_linalg_pass());
    pass_manager.add_pass(pass::linalg::create_linalg_elementwise_op_fusion_pass());
    pass_manager
        .run(&mut module)
        .expect("inline/elementwise-to-linalg/fuse must succeed");

    eliminate_redundant_field_store_copies(context, &mut module);
    assert!(
        module.as_operation().verify(),
        "module failed verification after the rewrite\n{}",
        module.as_operation()
    );

    let pass_manager = pass::PassManager::new(context);
    pass::bufferization::register_one_shot_bufferize_pass();
    parse_pass_pipeline(
        pass_manager.as_operation_pass_manager(),
        "builtin.module(one-shot-bufferize{bufferize-function-boundaries=true})",
    )
    .expect("failed to parse the one-shot-bufferize pass pipeline");
    pass_manager
        .run(&mut module)
        .expect("one-shot-bufferize must succeed");

    let pass_manager = pass::PassManager::new(context);
    pass_manager.add_pass(pass::bufferization::create_ownership_based_buffer_deallocation_pass());
    pass_manager.add_pass(pass::bufferization::create_buffer_deallocation_simplification_pass());
    pass_manager.add_pass(pass::bufferization::create_lower_deallocations_pass());
    pass_manager.add_pass(pass::conversion::create_bufferization_to_mem_ref());
    pass_manager
        .run(&mut module)
        .expect("buffer-deallocation must succeed");

    let pass_manager = pass::PassManager::new(context);
    pass_manager.add_pass(pass::linalg::create_convert_linalg_to_loops_pass());
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
    pass_manager
        .run(&mut module)
        .expect("lowering to the llvm dialect must succeed");

    let engine = melior::ExecutionEngine::new(&module, 2, &[], false, false);
    unsafe {
        engine.register_symbol("cleave_alloc", cleave_rt::cleave_alloc as *mut ());
        engine.register_symbol("cleave_alloc_rc", cleave_rt::cleave_alloc_rc as *mut ());
        engine.register_symbol("cleave_retain", cleave_rt::cleave_retain as *mut ());
        engine.register_symbol("cleave_release", cleave_rt::cleave_release as *mut ());
        engine.register_symbol("memrefCopy", cleave_rt::memrefCopy as *mut ());
    }
    let mut result: f32 = -1.0;
    unsafe {
        engine
            .invoke_packed("main", &mut [&mut result as *mut f32 as *mut ()])
            .unwrap_or_else(|e| panic!("JIT invocation failed: {e:?}"));
    }
    result
}

const SOURCE_PRELUDE: &str = r#"
        use linalg;

        struct Boxed { v: Tensor<f32, 4, 4> }

        fn build(a: Tensor<f32, 4, 4>, b: Tensor<f32, 4, 4>) -> Boxed {
            Boxed(v: Ring::sub(a, b))
        }
"#;

fn tensor_literal(base: f32) -> String {
    let mut rows = Vec::new();
    for r in 0..4 {
        let mut cols = Vec::new();
        for c in 0..4 {
            cols.push(format!("{:.1}", base + (r * 4 + c) as f32));
        }
        rows.push(format!("[{}]", cols.join(", ")));
    }
    format!("Tensor::<f32, 4, 4>(data: [{}])", rows.join(", "))
}

/// The dominant real-world shape (`Optimizer::step`'s own SGD update,
/// `stdlib/optim/optim.cleave`): a value computed by a *separate,
/// now-inlined* function, then stored into a struct field.
#[test]
fn a_struct_field_written_from_an_inlined_elementwise_op_computes_the_right_value() {
    let context = context();
    let a = tensor_literal(100.0);
    let b = tensor_literal(1.0);
    let src = format!(
        r#"
        {SOURCE_PRELUDE}
        fn main() -> f32 {{
            let a: Tensor<f32, 4, 4> = {a};
            let b: Tensor<f32, 4, 4> = {b};
            let boxed = build(a, b);
            boxed.v[1, 2]
        }}
        "#
    );
    let value = run_f32_with_rewrite(&context, &src);
    // a[1,2] = 100 + 6 = 106, b[1,2] = 1 + 6 = 7 -> 99.
    assert_eq!(value, 99.0, "a[1,2] - b[1,2] should be 99.0");
}

/// Structural check that the rewrite actually *fired* (not just that it was
/// safely skipped): the copy this rewrite leaves behind (`dps_rewrite.rs`'s
/// own doc comment explains why it's left rather than erased, and why its
/// own *size* operand -- not its addresses -- is what actually gets
/// neutered) must have a compile-time-zero size operand once bufferization
/// runs, proving the real byte-for-byte copy genuinely does not happen at
/// runtime, not just that a copy of *some* kind still exists (which would
/// also be true of the always-correct, unrewritten fallback path).
#[test]
fn the_copy_is_neutered_to_zero_bytes_for_the_matching_shape() {
    let context = context();
    let a = tensor_literal(100.0);
    let b = tensor_literal(1.0);
    let src = format!(
        r#"
        {SOURCE_PRELUDE}
        fn main() -> f32 {{
            let a: Tensor<f32, 4, 4> = {a};
            let b: Tensor<f32, 4, 4> = {b};
            let boxed = build(a, b);
            boxed.v[0, 0]
        }}
        "#
    );
    let text = bufferized_text(&context, &src);
    let memcpy_line = text
        .lines()
        .find(|line| line.contains("llvm.intr.memcpy"))
        .unwrap_or_else(|| panic!("expected a `llvm.intr.memcpy` line, got:\n{text}"));
    // `"llvm.intr.memcpy"(%DEST, %SRC, %SIZE) <{isVolatile = false}> : ...`
    let size_operand = memcpy_line
        .split_once('(')
        .and_then(|(_, rest)| rest.split_once(')'))
        .map(|(args, _)| args)
        .unwrap_or_else(|| panic!("could not parse memcpy operands from:\n{memcpy_line}"))
        .split(',')
        .nth(2)
        .map(str::trim)
        .unwrap_or_else(|| panic!("expected a third (size) operand in:\n{memcpy_line}"));
    // The size operand is an SSA name (e.g. `%84`); confirm *that* SSA value
    // is bound to a real `arith.constant 0 : i64` somewhere in the text --
    // not any old zero-looking substring match.
    let zero_def = format!("{size_operand} = arith.constant 0 : i64");
    assert!(
        text.contains(&zero_def),
        "expected {size_operand} to be defined as a zero i64 constant, got:\n{text}"
    );
}

/// A value that's read *again* elsewhere (here, stored into a second field
/// too) must **not** be rewritten -- the safety precondition this whole
/// module's own doc comment calls out (`count_uses(value) == 1`) -- still
/// correct, just via the unrewritten (copy-based) fallback path, since
/// redirecting *either* store's own destination would leave the other one
/// silently reading through the wrong (or already-freed) buffer.
#[test]
fn a_multiply_used_value_is_left_on_the_safe_fallback_path() {
    let context = context();
    let a = tensor_literal(100.0);
    let b = tensor_literal(1.0);
    let src = format!(
        r#"
        use linalg;

        struct Pair {{ v: Tensor<f32, 4, 4>, w: Tensor<f32, 4, 4> }}

        fn build(a: Tensor<f32, 4, 4>, b: Tensor<f32, 4, 4>) -> Pair {{
            let m = Ring::sub(a, b);
            Pair(v: m, w: m)
        }}

        fn main() -> f32 {{
            let a: Tensor<f32, 4, 4> = {a};
            let b: Tensor<f32, 4, 4> = {b};
            let p = build(a, b);
            p.w[2, 3]
        }}
        "#
    );
    let value = run_f32_with_rewrite(&context, &src);
    // a[2,3] = 100 + 11 = 111, b[2,3] = 1 + 11 = 12 -> 99.
    assert_eq!(value, 99.0, "a[2,3] - b[2,3] should be 99.0");
}

/// `linalg.matmul` must never match at all -- it genuinely reads its own
/// `outs` operand as a reduction accumulator (`Ring::zero()`-seeded), so
/// redirecting it to an uninitialized destination would silently corrupt
/// the result. A real `matmul`-into-field store must still compute the
/// right answer, on the always-correct fallback path.
#[test]
fn matmul_into_a_struct_field_is_never_rewritten_and_still_computes_correctly() {
    let context = context();
    let src = r#"
        use linalg;

        struct Boxed { v: Tensor<f32, 4, 4> }

        fn build(a: Tensor<f32, 4, 4>, b: Tensor<f32, 4, 4>) -> Boxed {
            Boxed(v: matmul(a, b))
        }

        fn main() -> f32 {
            let a: Tensor<f32, 4, 4> = Tensor::<f32, 4, 4>(data: [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0]
            ]);
            let identity: Tensor<f32, 4, 4> = Tensor::<f32, 4, 4>(data: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]);
            let boxed = build(a, identity);
            boxed.v[1, 2]
        }
        "#;
    let value = run_f32_with_rewrite(&context, src);
    assert_eq!(value, 7.0, "matmul(a, identity)[1,2] should equal a[1,2]");
}
