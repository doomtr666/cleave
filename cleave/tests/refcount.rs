//! Real, end-to-end proof for `cleave::refcount::insert_refcounting`
//! (`doc/hld.md`'s own "Memory management" section, Phase 0's struct/
//! descriptor half) — each test JIT-executes a real compiled program and
//! checks its actual numeric result, not just that the generated CPS/MLIR
//! *looks* plausible. Several of these are direct regression tests for
//! real bugs found via `examples/digits-interop` (a genuine
//! `STATUS_HEAP_CORRUPTION`/`STATUS_ACCESS_VIOLATION`, not a hypothetical)
//! — see `refcount.rs`'s own module doc comment for the full story behind
//! each one; a bare "doesn't crash" is itself the meaningful assertion for
//! those (a premature or missing release manifests as corruption, not a
//! wrong-but-plausible number, especially at the tiny scale these tests
//! run at — real correctness needs `examples/digits-interop`'s own actual
//! training run, already verified separately, not repeatable here).

use cleave::cps::{collect_mlir_types, collect_struct_schemas, collect_units, convert_program};
use cleave::driver::compile;
use cleave::egraph::optimize_program;
use cleave::mlir_lower::lower_program;
use cleave::pipeline::check_type_errors;
use cleave::refcount::insert_refcounting;
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

/// Compiles `src` through the *real* pipeline (`pipeline.rs::
/// build_optimized_cps`'s own exact sequence: CPS conversion, dead-code
/// elimination, e-graph optimization, a second dead-code sweep, then
/// `insert_refcounting`), lowers to the `llvm` dialect, and JIT-invokes
/// `main`, returning its own `i32` result.
fn run_i32(src: &str) -> i32 {
    let context = context();
    let (result, _sources) = compile(vec![("test.cleave".to_string(), src.to_string())], &[]);
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    if let Err(diags) = check_type_errors(&program, &registry) {
        panic!("type check failed: {diags:?}");
    }
    let units = collect_units(&program, &registry);
    let cps_program = convert_program(units);
    let cps_program = cleave::cps::eliminate_dead_code(cps_program);
    let (cps_program, _) = optimize_program(cps_program, &registry, false);
    let cps_program = cleave::cps::eliminate_dead_code(cps_program);
    let struct_schemas = collect_struct_schemas(&program);
    let mlir_types = collect_mlir_types(&program);
    let cps_program = insert_refcounting(cps_program, &struct_schemas, &mlir_types);

    let mlir_types = collect_mlir_types(&program);
    let struct_schemas = collect_struct_schemas(&program);
    let mut module = lower_program(&context, &cps_program, &mlir_types, struct_schemas);
    assert!(
        module.as_operation().verify(),
        "generated MLIR module failed verification"
    );

    let pass_manager = pass::PassManager::new(&context);
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
    pass_manager
        .run(&mut module)
        .expect("lowering to the llvm dialect must succeed");

    let engine = melior::ExecutionEngine::new(&module, 2, &[], false, false);
    // SAFETY: each `cleave_rt::*` pointer is a real, valid `extern "C" fn`,
    // live for the process's whole lifetime — mirrors `pipeline.rs::
    // register_cleave_rt_symbols`'s own registration.
    unsafe {
        engine.register_symbol("cleave_alloc", cleave_rt::cleave_alloc as *mut ());
        engine.register_symbol("cleave_alloc_rc", cleave_rt::cleave_alloc_rc as *mut ());
        engine.register_symbol("cleave_retain", cleave_rt::cleave_retain as *mut ());
        engine.register_symbol("cleave_release", cleave_rt::cleave_release as *mut ());
    }
    let mut out: i32 = -1;
    // SAFETY: `out` is a live, correctly-aligned `i32` on the stack for the
    // duration of this call.
    unsafe {
        engine
            .invoke_packed("main", &mut [&mut out as *mut i32 as *mut ()])
            .expect("JIT invocation must succeed — a real crash here (STATUS_HEAP_CORRUPTION/STATUS_ACCESS_VIOLATION) is exactly the failure mode a wrong retain/release insertion produces");
    }
    out
}

/// The simplest case: a struct constructed and never used again must be
/// released without disturbing anything else in scope.
#[test]
fn an_unused_locally_constructed_struct_is_released_cleanly() {
    let src = "struct Point { x: i32, y: i32 }
    fn make_unused() -> i32 {
        let p = Point(x: 1, y: 2);
        42
    }
    fn main() -> i32 { make_unused() }";
    assert_eq!(run_i32(src), 42);
}

/// A struct returned from a function must *not* be released before
/// returning — the caller reads its fields afterward.
#[test]
fn a_returned_struct_is_not_released_and_its_fields_read_correctly_afterward() {
    let src = "struct Point { x: i32, y: i32 }
    fn make_point() -> Point {
        let p = Point(x: 1, y: 2);
        p
    }
    fn main() -> i32 {
        let p = make_point();
        p.x + p.y
    }";
    assert_eq!(run_i32(src), 3);
}

/// A struct passed as a *borrowed* parameter to two separate calls must
/// remain valid for both — parameters are never released by the callee
/// (`insert_refcounting_fn`'s own doc comment).
#[test]
fn a_struct_borrowed_by_two_separate_calls_stays_valid_for_both() {
    let src = "struct Point { x: i32, y: i32 }
    fn sum(p: Point) -> i32 { p.x + p.y }
    fn main() -> i32 {
        let p = Point(x: 3, y: 4);
        let a = sum(p);
        let b = sum(p);
        a + b
    }";
    assert_eq!(run_i32(src), 14);
}

/// A loop constructing a fresh struct every iteration, reassigned via
/// `let mut` — each iteration's own struct must be released once replaced,
/// without corrupting the next one.
#[test]
fn a_freshly_constructed_struct_reassigned_every_loop_iteration_is_released_correctly() {
    let src = "struct Point { x: i32, y: i32 }
    fn main() -> i32 {
        let mut total = 0;
        for i in 0..100 {
            let p = Point(x: i, y: i);
            total = total + p.x + p.y;
        };
        total
    }";
    assert_eq!(run_i32(src), 9900);
}

/// Storing an *existing* struct-typed value into another struct's own
/// field (construction-time embedding, `Line(a: p1, b: p2)`) creates a
/// second, independent reference — both the original bindings (`p1`/`p2`)
/// and the new container (`l`) must remain independently valid and
/// readable. A missing retain here is exactly the bug `rewrite_body`'s own
/// "retain-on-construction" logic exists to prevent (a dangling read that
/// may not manifest as a wrong value immediately, since freed memory isn't
/// always overwritten right away — the real assertion here is `main`
/// returning without corruption at all).
#[test]
fn embedding_an_existing_struct_into_a_new_ones_own_field_retains_it_correctly() {
    let src = "struct Point { x: i32, y: i32 }
    struct Line { a: Point, b: Point }
    fn make_line() -> Line {
        let p1 = Point(x: 1, y: 2);
        let p2 = Point(x: 3, y: 4);
        Line(a: p1, b: p2)
    }
    fn main() -> i32 {
        let l = make_line();
        l.a.x + l.a.y + l.b.x + l.b.y
    }";
    assert_eq!(run_i32(src), 10);
}

/// A struct value carried across a loop (`let mut`, reassigned inside)
/// *through a real function call* every iteration — not just a local
/// reconstruction (already covered above). Regression test for the first
/// real `STATUS_HEAP_CORRUPTION` found via `examples/digits-interop`: the
/// value returned by the call and the loop's own previous carried value
/// must not collide, and the carried value must survive across many
/// iterations without being released early.
#[test]
fn a_struct_carried_through_a_loop_via_a_real_function_call_stays_correct_across_iterations() {
    let src = "struct Point { x: i32, y: i32 }
    fn bump(p: Point) -> Point {
        Point(x: p.x + 1, y: p.y + 1)
    }
    fn main() -> i32 {
        let mut p = Point(x: 0, y: 0);
        for i in 0..20 {
            p = bump(p);
        };
        p.x + p.y
    }";
    assert_eq!(run_i32(src), 40);
}

/// Regression test for the second and third real `STATUS_HEAP_CORRUPTION`
/// bugs found via `examples/digits-interop` (`train_and_evaluate`'s own
/// `opt`): a fresh, owned struct constructed *outside* two nested loops,
/// referenced only *inside* the innermost one (through an intervening real
/// call's own resumption continuation, exactly like `Optimizer::init_state`
/// sitting between `opt`'s own construction and the training loop) — must
/// survive every iteration of *both* loops and be released exactly once,
/// after the outer loop is genuinely done. A premature release (the
/// second bug: the inner loop releasing it at the end of its own first
/// invocation) or a "trampoline" resumption hiding it from `live_set` (the
/// third bug) each reproduced a real crash at this exact shape; a
/// correctness assertion on the final value, not just "doesn't crash",
/// confirms the shared struct's own fields weren't corrupted along the way
/// either.
#[test]
fn a_struct_referenced_only_through_an_intervening_call_survives_two_nested_loops() {
    let src = "struct Config { step: i32 }
    fn advance(c: Config, p: i32) -> i32 { p + c.step }
    fn make_config() -> Config { Config(step: 1) }
    fn main() -> i32 {
        let cfg = make_config();
        let mut total = 0;
        for outer in 0..3 {
            for inner in 0..4 {
                total = advance(cfg, total);
            };
        };
        total
    }";
    assert_eq!(run_i32(src), 12);
}
