//! Stage 6's own end-to-end proof: a real `axiom`, parsed through the real
//! compiler front end, actually changes a real function's own compiled CPS
//! (folding a redundant call away entirely) while the whole program still
//! JIT-executes to the identical runtime value — not just that
//! `optimize_program` compiles or that its own text output looks plausible.

use cleave::cps::{
    collect_mlir_types, collect_struct_schemas, collect_units, convert_program, dump_cps_program,
};
use cleave::driver::compile;
use cleave::egraph::optimize_program;
use cleave::mlir_lower::lower_program;
use cleave::registry::Registry;
use melior::Context;
use melior::dialect::DialectRegistry;
use melior::ir::operation::OperationLike;
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

/// Lowers an already-built `cps_program` to the `llvm` dialect and JIT-
/// invokes its own `main`, returning the result — mirrors `mlir_lower.rs`'s
/// own `run_i32` exactly, except it takes a `CpsProgram` directly rather
/// than a source string, so the same `mlir_types`/`struct_schemas` (derived
/// from the unoptimized AST, never touched by `optimize_program`) can be
/// reused to run both the naive and the axiom-optimized form of the
/// identical program.
fn run(
    context: &Context,
    program: &cleave::ast::Program,
    cps_program: &cleave::cps::CpsProgram,
) -> i32 {
    let mlir_types = collect_mlir_types(program);
    let struct_schemas = collect_struct_schemas(program);
    let mut module = lower_program(context, cps_program, &mlir_types, struct_schemas);
    assert!(
        module.as_operation().verify(),
        "generated MLIR module failed verification"
    );

    let pass_manager = pass::PassManager::new(context);
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    pass_manager
        .run(&mut module)
        .expect("lowering to the llvm dialect must succeed");

    let engine = melior::ExecutionEngine::new(&module, 2, &[], false, false);
    // SAFETY: a real, valid `extern "C" fn`, live for the process's whole
    // lifetime — mirrors `main.rs`'s own registration exactly.
    unsafe {
        engine.register_symbol("cleave_alloc", cleave_rt::cleave_alloc as *mut ());
        engine.register_symbol("cleave_alloc_rc", cleave_rt::cleave_alloc_rc as *mut ());
        engine.register_symbol("cleave_retain", cleave_rt::cleave_retain as *mut ());
        engine.register_symbol("cleave_release", cleave_rt::cleave_release as *mut ());
    }
    let mut out: i32 = -1;
    // SAFETY: `out` is a live, correctly-aligned `i32` on the stack for the
    // duration of this call, matching the verified `i32`-returning `main`.
    unsafe {
        engine
            .invoke_packed("main", &mut [&mut out as *mut i32 as *mut ()])
            .expect("JIT invocation must succeed");
    }
    out
}

/// The full proof: `helper`'s own parameter `x` is a genuine free variable
/// at the e-graph level (never a literal, so this can't be mistaken for
/// plain constant folding, already covered by `egraph.rs`'s own Stage 2
/// tests) — `x + 0` compiles to a real call to `Ring::add<i32>`, which the
/// `add_zero` axiom folds away entirely, leaving `helper` a bare `return x`.
/// `main` itself is untouched (it calls `helper`, a unit whose own body
/// contains a real `Fix` — not straight-line — so `Forward::walk` correctly
/// stops at that call rather than inlining it; see `egraph.rs`'s own
/// `is_straight_line` doc comment).
#[test]
fn an_axiom_folds_a_real_call_away_and_the_optimized_program_still_executes_to_the_same_value() {
    let context = context();
    // `add_zero` is a real axiom on the shipped `Ring<T>` now
    // (`stdlib/num/num.cleave`) -- no local `algebra Ring<T>` fragment
    // needed here (redeclaring it would collide with the real one, "same
    // parameter types", found by direct testing once the axiom actually
    // shipped).
    let src = "fn helper(x: i32) -> i32 {
        x + 0
    }

    fn main() -> i32 {
        helper(21)
    }";
    let (result, _sources) = compile(vec![("test.cleave".to_string(), src.to_string())], &[]);
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);

    let naive_units = collect_units(&program, &registry);
    let naive_cps = convert_program(naive_units);
    let naive_dump = dump_cps_program(&naive_cps);
    let naive_helper = naive_dump
        .split("(fn helper")
        .nth(1)
        .expect("`helper` must be in the naive dump");
    assert!(
        naive_helper.contains("Ring::add<i32>"),
        "the naive form must still call `Ring::add<i32>`, got:\n{naive_helper}"
    );

    let optimize_units = collect_units(&program, &registry);
    let optimize_cps = convert_program(optimize_units);
    let (optimized, explanations) = optimize_program(optimize_cps, &registry, true);
    assert!(
        !explanations.is_empty(),
        "the axiom must have fired on at least one function"
    );
    assert!(
        explanations.iter().any(|e| e.contains("helper")),
        "expected an explanation naming `helper` specifically, got {explanations:?}"
    );

    let optimized_dump = dump_cps_program(&optimized);
    let optimized_helper = optimized_dump
        .split("(fn helper")
        .nth(1)
        .expect("`helper` must be in the optimized dump");
    assert!(
        !optimized_helper.contains("Ring::add<i32>"),
        "the axiom should have folded the call away entirely, got:\n{optimized_helper}"
    );
    assert_ne!(
        naive_helper, optimized_helper,
        "`helper`'s own CPS must be observably different after optimization"
    );

    // The real proof, not just that the CPS text changed: both forms still
    // execute to the identical runtime value.
    let naive_units_for_run = collect_units(&program, &registry);
    let naive_cps_for_run = convert_program(naive_units_for_run);
    assert_eq!(
        run(&context, &program, &naive_cps_for_run),
        21,
        "the naive (unoptimized) program's own result"
    );
    assert_eq!(
        run(&context, &program, &optimized),
        21,
        "the axiom-optimized program must produce the identical result"
    );
}

/// The Struct/Field extension's own end-to-end proof — deliberately *not*
/// `examples/complex.cleave` (its own float literals and multi-level call
/// nesting put it out of reach of this increment entirely, see the plan
/// this was built against). `i32` struct fields sidestep the float wall on
/// purpose: `p.y` is a real struct-field read, never a literal, so nothing
/// here could be mistaken for plain constant folding. `helper`'s own body
/// only becomes fully optimizable once the built-in struct-projection
/// rewrite (`struct_projection_rewrites`) unions `p.y` with the literal `0`
/// it was constructed from — only then does `add_zero`'s own pattern have
/// anything to match, folding `p.x + p.y` down to a bare `p.x` read.
#[test]
fn a_struct_field_read_lets_add_zero_fold_a_real_call_away_and_the_optimized_program_still_executes_to_the_same_value()
 {
    let context = context();
    let src = "struct Pair { x: i32, y: i32 }

    fn helper() -> i32 {
        let p = Pair(x: 21, y: 0);
        p.x + p.y
    }

    fn main() -> i32 {
        helper()
    }";
    let (result, _sources) = compile(vec![("test.cleave".to_string(), src.to_string())], &[]);
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);

    let naive_units = collect_units(&program, &registry);
    let naive_cps = convert_program(naive_units);
    let naive_dump = dump_cps_program(&naive_cps);
    let naive_helper = naive_dump
        .split("(fn helper")
        .nth(1)
        .expect("`helper` must be in the naive dump");
    assert!(
        naive_helper.contains("Ring::add<i32>"),
        "the naive form must still call `Ring::add<i32>`, got:\n{naive_helper}"
    );

    let optimize_units = collect_units(&program, &registry);
    let optimize_cps = convert_program(optimize_units);
    let (optimized, explanations) = optimize_program(optimize_cps, &registry, true);
    assert!(
        !explanations.is_empty(),
        "the struct-projection rewrite plus add_zero must have fired on at least one function"
    );
    assert!(
        explanations.iter().any(|e| e.contains("helper")),
        "expected an explanation naming `helper` specifically, got {explanations:?}"
    );

    let optimized_dump = dump_cps_program(&optimized);
    let optimized_helper = optimized_dump
        .split("(fn helper")
        .nth(1)
        .expect("`helper` must be in the optimized dump");
    assert!(
        !optimized_helper.contains("Ring::add<i32>"),
        "the axiom should have folded the call away entirely, got:\n{optimized_helper}"
    );
    assert_ne!(
        naive_helper, optimized_helper,
        "`helper`'s own CPS must be observably different after optimization"
    );

    let naive_units_for_run = collect_units(&program, &registry);
    let naive_cps_for_run = convert_program(naive_units_for_run);
    assert_eq!(
        run(&context, &program, &naive_cps_for_run),
        21,
        "the naive (unoptimized) program's own result"
    );
    assert_eq!(
        run(&context, &program, &optimized),
        21,
        "the axiom-optimized program must produce the identical result"
    );
}
