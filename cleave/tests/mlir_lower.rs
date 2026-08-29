use cleave::cps::{
    UnitBody, collect_mlir_types, collect_struct_schemas, collect_units, convert_program,
    eliminate_dead_code,
};
use cleave::driver::compile;
use cleave::egraph::{DerivativeRequest, optimize_program, synthesize_derivatives};
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

/// Registers every `stdlib/dynarray/dynarray.cleave` + `stdlib/display/
/// display.cleave` + `stdlib/io/io.cleave` runtime symbol this test file's
/// own several independent, hand-built `ExecutionEngine` harnesses might
/// need, unconditionally, harmless if unused. Once `stdlib/io/io.cleave`
/// started `use dynarray;`/`use display;` for its own new `Display<T>`-
/// backed `Print<T>` impls, `use io;` alone (already used by the
/// overwhelming majority of tests in this file) started transitively
/// pulling in *every* `RawBuffer<T>` impl for *every* scalar width
/// (`stdlib/dynarray/dynarray.cleave`'s own six `impl RawBuffer<...>`
/// blocks, all non-generic, hence eagerly compiled into any program
/// reaching them at all — `cps.rs::collect_units`'s own "non-generic impl"
/// branch, unconditional, regardless of whether `DynArray<i16>` etc. is
/// ever actually *constructed*) — not just the `i8`-width one `Display<T>`
/// itself actually uses. A single shared helper here, instead of hand-
/// listing the same ~30 symbols at each of this file's own several
/// `ExecutionEngine::new` call sites, keeps them from drifting out of sync.
fn register_io_symbols(engine: &melior::ExecutionEngine) {
    unsafe {
        engine.register_symbol("print_i8", cleave_rt::print_i8 as *mut ());
        engine.register_symbol("print_i16", cleave_rt::print_i16 as *mut ());
        engine.register_symbol("print_i32", cleave_rt::print_i32 as *mut ());
        engine.register_symbol("print_i64", cleave_rt::print_i64 as *mut ());
        engine.register_symbol("print_f32", cleave_rt::print_f32 as *mut ());
        engine.register_symbol("print_f64", cleave_rt::print_f64 as *mut ());
        engine.register_symbol("print_bytes", cleave_rt::print_bytes as *mut ());
        engine.register_symbol(
            "print_dynarray_bytes",
            cleave_rt::print_dynarray_bytes as *mut (),
        );
        engine.register_symbol("format_f32", cleave_rt::format_f32 as *mut ());
        engine.register_symbol("format_f64", cleave_rt::format_f64 as *mut ());
        engine.register_symbol("dynarray_alloc_i8", cleave_rt::dynarray_alloc_i8 as *mut ());
        engine.register_symbol("dynarray_grow_i8", cleave_rt::dynarray_grow_i8 as *mut ());
        engine.register_symbol("dynarray_get_i8", cleave_rt::dynarray_get_i8 as *mut ());
        engine.register_symbol("dynarray_set_i8", cleave_rt::dynarray_set_i8 as *mut ());
        engine.register_symbol(
            "dynarray_alloc_i16",
            cleave_rt::dynarray_alloc_i16 as *mut (),
        );
        engine.register_symbol("dynarray_grow_i16", cleave_rt::dynarray_grow_i16 as *mut ());
        engine.register_symbol("dynarray_get_i16", cleave_rt::dynarray_get_i16 as *mut ());
        engine.register_symbol("dynarray_set_i16", cleave_rt::dynarray_set_i16 as *mut ());
        engine.register_symbol(
            "dynarray_alloc_i32",
            cleave_rt::dynarray_alloc_i32 as *mut (),
        );
        engine.register_symbol("dynarray_grow_i32", cleave_rt::dynarray_grow_i32 as *mut ());
        engine.register_symbol("dynarray_get_i32", cleave_rt::dynarray_get_i32 as *mut ());
        engine.register_symbol("dynarray_set_i32", cleave_rt::dynarray_set_i32 as *mut ());
        engine.register_symbol(
            "dynarray_alloc_i64",
            cleave_rt::dynarray_alloc_i64 as *mut (),
        );
        engine.register_symbol("dynarray_grow_i64", cleave_rt::dynarray_grow_i64 as *mut ());
        engine.register_symbol("dynarray_get_i64", cleave_rt::dynarray_get_i64 as *mut ());
        engine.register_symbol("dynarray_set_i64", cleave_rt::dynarray_set_i64 as *mut ());
        engine.register_symbol(
            "dynarray_alloc_f32",
            cleave_rt::dynarray_alloc_f32 as *mut (),
        );
        engine.register_symbol("dynarray_grow_f32", cleave_rt::dynarray_grow_f32 as *mut ());
        engine.register_symbol("dynarray_get_f32", cleave_rt::dynarray_get_f32 as *mut ());
        engine.register_symbol("dynarray_set_f32", cleave_rt::dynarray_set_f32 as *mut ());
        engine.register_symbol(
            "dynarray_alloc_f64",
            cleave_rt::dynarray_alloc_f64 as *mut (),
        );
        engine.register_symbol("dynarray_grow_f64", cleave_rt::dynarray_grow_f64 as *mut ());
        engine.register_symbol("dynarray_get_f64", cleave_rt::dynarray_get_f64 as *mut ());
        engine.register_symbol("dynarray_set_f64", cleave_rt::dynarray_set_f64 as *mut ());
    }
}

/// Compiles `src` all the way through CPS conversion and MLIR lowering,
/// returning the verified module's own printed text.
fn lower(context: &Context, src: &str) -> String {
    let (result, _sources) = compile(vec![("test.cleave".to_string(), src.to_string())], &[]);
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    let units = collect_units(&program, &registry);
    let cps_program = convert_program(units);
    let mlir_types = collect_mlir_types(&program);
    let struct_schemas = collect_struct_schemas(&program);
    let module = lower_program(context, &cps_program, &mlir_types, struct_schemas);
    assert!(
        module.as_operation().verify(),
        "generated MLIR module failed verification"
    );
    module.as_operation().to_string()
}

#[test]
fn a_function_returning_a_bare_literal_lowers_to_a_constant_plus_return() {
    let context = context();
    let text = lower(&context, "fn main() -> i32 { 0 }");
    assert!(text.contains("func.func @main() -> i32"), "got:\n{text}");
    assert!(text.contains("arith.constant 0 : i32"), "got:\n{text}");
    assert!(
        text.contains("return %c0_i32 : i32") || text.contains("return %"),
        "got:\n{text}"
    );
}

/// A real bug, found by direct testing (`examples/axiom_demo.cleave`):
/// `llvm.emit_c_interface` (and the extra `_mlir_ciface_<name>` wrapper it
/// generates) used to get attached to *every* top-level fn unconditionally
/// — needless bloat, since the only function this project ever invokes via
/// `ExecutionEngine::invoke_packed` is `main` (confirmed: every call site,
/// `main.rs`'s own `--run` and every test, hardcodes `"main"`); an ordinary
/// internal call already goes through a plain `call @name`, which never
/// needs the C-interface wrapper at all. Scoped to `main` alone now.
#[test]
fn only_main_gets_the_c_interface_export_attribute() {
    let context = context();
    let text = lower(
        &context,
        "fn helper(x: i32) -> i32 { x }\nfn main() -> i32 { helper(1) }",
    );
    assert!(
        text.contains("func.func @main() -> i32 attributes {llvm.emit_c_interface}"),
        "main must still carry the export attribute, got:\n{text}"
    );
    assert!(
        !text.contains("func.func @helper(%arg0: i32) -> i32 attributes"),
        "an ordinary internal function must not carry it, got:\n{text}"
    );
}

#[test]
fn different_integer_widths_lower_to_their_own_mlir_type() {
    let context = context();
    let text = lower(&context, "fn main() -> i64 { 42 }");
    assert!(text.contains("func.func @main() -> i64"), "got:\n{text}");
    assert!(text.contains("arith.constant 42 : i64"), "got:\n{text}");
}

/// The real end-to-end proof: not just that the MLIR *looks* right, but
/// that it actually executes (via a JIT `ExecutionEngine`, after lowering
/// to the `llvm` dialect) and produces the *correct*, non-coincidental
/// value -- `17`, not `0`, specifically so a bug that always returns zero
/// wouldn't silently pass.
#[test]
fn a_compiled_program_actually_runs_and_returns_the_right_value() {
    let context = context();
    let (result, _sources) = compile(
        vec![(
            "test.cleave".to_string(),
            "fn main() -> i32 { 17 }".to_string(),
        )],
        &[],
    );
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    let units = collect_units(&program, &registry);
    let cps_program = convert_program(units);
    let mlir_types = collect_mlir_types(&program);
    let struct_schemas = collect_struct_schemas(&program);
    let mut module = lower_program(&context, &cps_program, &mlir_types, struct_schemas);
    assert!(module.as_operation().verify());

    let pass_manager = pass::PassManager::new(&context);
    // `num`'s own `Rem::mod` (prelude, unconditionally compiled into every
    // module regardless of whether the program actually calls it — see
    // `doc/backlog.md`'s own "dead-code elimination" item) has a real `if`/
    // `else` body, so `scf.if` is now present in *every* compiled module,
    // not just ones whose own source uses `if` — `create_to_llvm` alone
    // can't translate it (needs `create_scf_to_control_flow` first, same
    // reasoning as `run_i32`'s own doc comment below), found by direct
    // testing the moment `mod`/`rem` landed in the stdlib.
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    // Multiple independent conversion passes can each leave `builtin.
    // unrealized_conversion_cast` bridge ops between their own intermediate
    // representations behind -- found by direct testing, kept defensively:
    // real LLVM-IR translation can't handle a bare `unrealized_conversion_
    // cast` at all (`LLVM Translation failed for operation: builtin.
    // unrealized_conversion_cast`, a hard native crash). This is the
    // standard MLIR cleanup for exactly that situation: folds/cancels
    // chains of these casts away (`cast(cast(x, A->B), B->A) == x`), not a
    // real lowering step of its own.
    pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
    pass_manager
        .run(&mut module)
        .expect("lowering to the llvm dialect must succeed");

    let engine = melior::ExecutionEngine::new(&module, 2, &[], false, false);
    // Registered unconditionally, harmless if unused -- any struct
    // construction anywhere in the program (not just a top-level return)
    // needs `cleave_alloc` (see `mlir_lower.rs::alloc_struct`'s own doc
    // comment).
    unsafe {
        engine.register_symbol("cleave_alloc", cleave_rt::cleave_alloc as *mut ());
        engine.register_symbol("cleave_alloc_rc", cleave_rt::cleave_alloc_rc as *mut ());
        engine.register_symbol("cleave_retain", cleave_rt::cleave_retain as *mut ());
        engine.register_symbol("cleave_release", cleave_rt::cleave_release as *mut ());
        engine.register_symbol("cleave_release_void", cleave_rt::cleave_release_void as *mut ());
        engine.register_symbol("cleave_alloc_local", cleave_rt::cleave_alloc_local as *mut ());
        engine.register_symbol("cleave_region_enter", cleave_rt::cleave_region_enter as *mut ());
        engine.register_symbol("cleave_region_exit", cleave_rt::cleave_region_exit as *mut ());
    }
    let mut out: i32 = -1;
    unsafe {
        engine
            .invoke_packed("main", &mut [&mut out as *mut i32 as *mut ()])
            .expect("JIT invocation must succeed");
    }
    assert_eq!(out, 17);
}

/// Lowers `src` to the `llvm` dialect and JIT-invokes its `main`, returning
/// the result. `scf.if` (and any other structured-control-flow op) has no
/// direct LLVM IR translation of its own -- `create_scf_to_control_flow`
/// lowers it to the `cf` dialect's ordinary branches first, which `create_
/// to_llvm` *does* know how to translate. Skipping this produced a hard
/// native crash (`STATUS_ACCESS_VIOLATION`), not a clean Rust-level error --
/// found by direct testing, same fix applied in `main.rs`'s own `--run`.
///
/// `check_type_errors` runs first, exactly like every one of `main.rs`'s own
/// `--run`/`--emit-*` call sites already do -- `doc/backlog.md`'s own former
/// "`run_i32` skips `check_type_errors`" entry: without it, a genuinely
/// invalid test source (one whose own error a per-function inference pass
/// doesn't happen to catch -- `TypeErrorKind::VariadicStructNeedsTurbofish`
/// was the one found by direct testing) reached CPS/MLIR lowering anyway and
/// crashed the *whole test binary* natively (`Symbols not found`,
/// `STATUS_STACK_BUFFER_OVERRUN`) instead of failing cleanly as just that one
/// test. A clean `panic!` here reports the real diagnostic and fails only
/// the calling test, matching every other assertion failure in this file.
fn run_i32(context: &Context, src: &str) -> i32 {
    let (result, _sources) = compile(vec![("test.cleave".to_string(), src.to_string())], &[]);
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    if let Err(diags) = check_type_errors(&program, &registry) {
        panic!("type check failed: {diags:?}");
    }
    let units = collect_units(&program, &registry);
    // `fprime = derive(f);` (`doc/backlog.md`'s own auto-diff item) needs
    // `synthesize_derivatives` run right after `convert_program` -- mirrors
    // `main.rs`'s own `build_cps_program` exactly (that one's private to
    // the binary, so this test harness's own shared JIT helper needs the
    // identical three-step sequence spelled out here instead of reusing it).
    let requests: Vec<DerivativeRequest> = units
        .iter()
        .filter_map(|u| match &u.body {
            UnitBody::Derivative(of, is_grad) => Some(DerivativeRequest {
                name: u.name.clone(),
                of: of.clone(),
                is_grad: *is_grad,
            }),
            _ => None,
        })
        .collect();
    let cps_program = convert_program(units);
    let struct_schemas = collect_struct_schemas(&program);
    let cps_program = synthesize_derivatives(cps_program, &requests, &registry, &struct_schemas)
        .unwrap_or_else(|e| panic!("cannot derive: {e:?}"));
    run_i32_from_cps(context, &program, cps_program)
}

/// Like `run_i32`, but also runs `optimize_program` (+ the surrounding
/// `eliminate_dead_code` sweeps) — matching `main.rs`'s own real `--run`
/// pipeline, which `run_i32` itself does not (`doc/backlog.md`'s own
/// "`run_i32` skips `check_type_errors`, unlike `main.rs`" entry already
/// flags the same class of pre-existing test-harness/CLI divergence).
/// Needed for real, not speculatively: `linalg::MatMul`'s own product-rule-
/// synthesized derivative tree, freshly extracted and *un*-simplified,
/// panics MLIR lowering under plain `run_i32` while the exact same source
/// runs correctly through the real CLI — the only difference between the
/// two pipelines being this step. Kept as a *separate* function rather than
/// added to `run_i32` itself — tried first, reverted: it broke five other,
/// already-passing tests elsewhere in this file, too invasive a change to
/// a helper 125+ tests already share for a gap only this one test hits.
fn run_i32_with_optimization_pass(context: &Context, src: &str) -> i32 {
    let (result, _sources) = compile(vec![("test.cleave".to_string(), src.to_string())], &[]);
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    let units = collect_units(&program, &registry);
    let requests: Vec<DerivativeRequest> = units
        .iter()
        .filter_map(|u| match &u.body {
            UnitBody::Derivative(of, is_grad) => Some(DerivativeRequest {
                name: u.name.clone(),
                of: of.clone(),
                is_grad: *is_grad,
            }),
            _ => None,
        })
        .collect();
    let cps_program = convert_program(units);
    let struct_schemas = collect_struct_schemas(&program);
    let cps_program = synthesize_derivatives(cps_program, &requests, &registry, &struct_schemas)
        .unwrap_or_else(|e| panic!("cannot derive: {e:?}"));
    let cps_program = eliminate_dead_code(cps_program);
    let (cps_program, _) = optimize_program(cps_program, &registry, false);
    let cps_program = eliminate_dead_code(cps_program);
    run_i32_from_cps(context, &program, cps_program)
}

fn run_i32_from_cps(
    context: &Context,
    program: &cleave::ast::Program,
    cps_program: cleave::cps::CpsProgram,
) -> i32 {
    let mlir_types = collect_mlir_types(program);
    let struct_schemas = collect_struct_schemas(program);
    let mut module = lower_program(context, &cps_program, &mlir_types, struct_schemas);
    assert!(
        module.as_operation().verify(),
        "generated MLIR module failed verification"
    );

    // Staged as three *separate* `PassManager`s, each run to completion
    // before the next is even built — found by direct testing to matter,
    // not just tidier: the identical passes combined into one single
    // `PassManager`/one `run()` call failed partway (`op was not
    // bufferized`) even though every individual stage, run to completion
    // first, succeeds cleanly. Not fully root-caused beyond that (melior's
    // own pass-manager nesting/ordering semantics across `add_pass` and a
    // `parse_pass_pipeline`-populated nested nest — see the bufferize stage
    // below — most likely), but empirically robust, so kept as the working
    // shape rather than chased further.
    //
    // Stage 1: a bare `arith.addf` (etc.) on `tensor`-typed operands
    // (`Ring<Tensor<T,Dims...>>`'s own elementwise impls, `stdlib/linalg/
    // tensor.cleave`) has no `BufferizableOpInterface` implementation of
    // its own — only a real structured/named op does — so one-shot-
    // bufferize (stage 2) can't handle it directly without this first.
    let pass_manager = pass::PassManager::new(context);
    pass_manager.add_pass(pass::linalg::create_convert_elementwise_to_linalg_pass());
    pass_manager
        .run(&mut module)
        .expect("convert-elementwise-to-linalg must succeed");

    // Stage 2: `bufferize-function-boundaries=true` — melior's own
    // generated `create_one_shot_bufferize_pass()` binding takes no
    // options at all (the underlying C API constructor is zero-argument),
    // so the option has to go in via a real textual pass-pipeline string
    // instead (`melior::utility::parse_pass_pipeline`) — without it, a
    // `tensor`-typed function parameter/return (any cross-function call
    // involving a `Vector`/`Matrix`) is left bridged by a `bufferization.
    // to_buffer`/`to_tensor` pair at the function boundary that nothing
    // later in this pipeline can legalize (`failed to legalize operation
    // 'bufferization.to_buffer'`, a real pass failure, found by direct
    // testing) — this option makes one-shot-bufferize rewrite the
    // function's own signature directly instead, eliminating the bridge
    // entirely. The pass must be registered by name first — textual
    // pipeline parsing looks it up by its own registered name, unlike
    // `add_pass`, which already has the concrete `Pass` object in hand.
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

    // Stage 3: ordinary lowering to the `llvm` dialect — everything past
    // this point is plain `memref`/`arith`/`scf`, already fully handled by
    // this project's own pre-existing pipeline, unchanged.
    let pass_manager = pass::PassManager::new(context);
    pass_manager.add_pass(pass::linalg::create_convert_linalg_to_loops_pass());
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    // Multiple independent conversion passes can each leave `builtin.
    // unrealized_conversion_cast` bridge ops between their own intermediate
    // representations behind -- found by direct testing, kept defensively:
    // real LLVM-IR translation can't handle a bare `unrealized_conversion_
    // cast` at all (`LLVM Translation failed for operation: builtin.
    // unrealized_conversion_cast`, a hard native crash). This is the
    // standard MLIR cleanup for exactly that situation: folds/cancels
    // chains of these casts away (`cast(cast(x, A->B), B->A) == x`), not a
    // real lowering step of its own.
    pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
    pass_manager
        .run(&mut module)
        .expect("lowering to the llvm dialect must succeed");

    let engine = melior::ExecutionEngine::new(&module, 2, &[], false, false);
    // Registered unconditionally, harmless if unused -- any struct
    // construction anywhere in the program (not just a top-level return)
    // needs `cleave_alloc` (see `mlir_lower.rs::alloc_struct`'s own doc
    // comment). `memrefCopy` the same way -- `cleave_rt::memrefCopy`'s own
    // doc comment has the full story (`one-shot-bufferize`'s own generated
    // `memref.copy` calls need it once a tensor value is big enough to need
    // a real defensive copy, first hit by `derive_through_dense_forward_
    // computes_the_right_gradient`, just below).
    unsafe {
        engine.register_symbol("cleave_alloc", cleave_rt::cleave_alloc as *mut ());
        engine.register_symbol("cleave_alloc_rc", cleave_rt::cleave_alloc_rc as *mut ());
        engine.register_symbol("cleave_retain", cleave_rt::cleave_retain as *mut ());
        engine.register_symbol("cleave_release", cleave_rt::cleave_release as *mut ());
        engine.register_symbol("cleave_release_void", cleave_rt::cleave_release_void as *mut ());
        engine.register_symbol("cleave_alloc_local", cleave_rt::cleave_alloc_local as *mut ());
        engine.register_symbol("cleave_region_enter", cleave_rt::cleave_region_enter as *mut ());
        engine.register_symbol("cleave_region_exit", cleave_rt::cleave_region_exit as *mut ());
        engine.register_symbol("memrefCopy", cleave_rt::memrefCopy as *mut ());
        engine.register_symbol("rand_seed", cleave_rt::rand_seed as *mut ());
        engine.register_symbol("rand_uniform_f32", cleave_rt::rand_uniform_f32 as *mut ());
        engine.register_symbol("rand_uniform_f64", cleave_rt::rand_uniform_f64 as *mut ());
        engine.register_symbol("rand_normal_f32", cleave_rt::rand_normal_f32 as *mut ());
        engine.register_symbol("rand_normal_f64", cleave_rt::rand_normal_f64 as *mut ());
    }
    // `use io;` now transitively pulls in `stdlib/display/display.cleave`
    // and `stdlib/dynarray/dynarray.cleave` -- see `register_io_symbols`'s
    // own doc comment.
    register_io_symbols(&engine);
    let mut out: i32 = -1;
    unsafe {
        engine
            .invoke_packed("main", &mut [&mut out as *mut i32 as *mut ()])
            .expect("JIT invocation must succeed");
    }
    out
}

#[test]
fn an_arithmetic_intrinsic_lowers_to_the_right_arith_op() {
    let context = context();
    let text = lower(&context, "fn main() -> i32 { 2 + 3 }");
    assert!(text.contains("arith.addi"), "got:\n{text}");
}

#[test]
fn an_arithmetic_intrinsic_actually_computes_the_right_value() {
    let context = context();
    assert_eq!(run_i32(&context, "fn main() -> i32 { 2 + 3 * 4 }"), 14);
}

#[test]
fn an_if_expression_lowers_to_scf_if() {
    let context = context();
    let text = lower(&context, "fn main() -> i32 { if 1 < 2 { 10 } else { 20 } }");
    assert!(text.contains("scf.if"), "got:\n{text}");
    assert!(text.contains("arith.cmpi"), "got:\n{text}");
}

#[test]
fn an_if_expression_actually_picks_the_right_branch() {
    let context = context();
    assert_eq!(
        run_i32(&context, "fn main() -> i32 { if 1 < 2 { 10 } else { 20 } }"),
        10
    );
    assert_eq!(
        run_i32(&context, "fn main() -> i32 { if 2 < 1 { 10 } else { 20 } }"),
        20
    );
}

/// The real end-to-end proof for recursion: a genuinely recursive function
/// (not just a self-tail-call trampoline) compiles, lowers `scf.if` and two
/// real `func.call @fib` sites correctly, and JIT-executes to the
/// mathematically correct, non-trivial value -- `fib(10) == 55`, not a
/// value any single one of the individual pieces (arithmetic, `if`, a call)
/// could produce by coincidence on its own.
#[test]
fn a_recursive_function_actually_computes_fibonacci() {
    let context = context();
    let src = "fn fib(n: i32) -> i32 { if n < 2 { n } else { fib(n - 1) + fib(n - 2) } } fn main() -> i32 { fib(10) }";
    assert_eq!(run_i32(&context, src), 55);
}

/// A `()`-returning function called from another function -- not just as
/// the program's own entry point -- used to fail MLIR verification
/// outright (`'func.call' op incorrect number of results for callee`):
/// `lower_top_level_fn` already declares a `()`-returning callee with
/// *zero* MLIR results (`is_unit_ty`, applied uniformly to every top-level
/// fn, not scoped to `main`), but `lower_real_call`'s own call site used to
/// unconditionally request exactly one result regardless of the callee's
/// own return type. Found via `examples/vector.cleave`'s own `print_vec`
/// helper -- an entirely ordinary pattern (a small `()`-returning helper
/// called from `main`), not a contrived case.
#[test]
fn a_unit_returning_function_can_be_called_from_another_function() {
    let context = context();
    let src = "
        fn touch(x: i32) {
            x + 1;
        }
        fn main() -> i32 {
            touch(1);
            touch(2);
            42
        }
    ";
    assert_eq!(run_i32(&context, src), 42);
}

/// The generic mechanism itself (`PrimOp::RawMlirOp`, `mlir_lower.rs::
/// lower_raw_mlir_op`), independent of `stdlib/num`'s own use of it --
/// `doc/hld.md`'s "one generic 'emit this named MLIR op' primitive": a
/// direct `mlir::dialect::op(...)` call needs no per-op Rust knowledge at
/// all, confirmed here without going through `+`/`<` sugar.
#[test]
fn a_direct_mlir_call_lowers_to_the_named_op_with_no_declaration() {
    let context = context();
    let text = lower(&context, "fn main() -> i32 { mlir::arith::addi(2, 3) }");
    assert!(text.contains("arith.addi"), "got:\n{text}");
    // Unlike `extern`, a raw MLIR op call needs no `func.func` declaration
    // at all -- it's not a symbol to resolve, just an op to emit directly.
    // Checked precisely by symbol name, not by a blanket "no `func.func
    // private` anywhere in the dump" -- `lower_program` now stamps every
    // *other*, genuinely internal function (`Ring::zero<...>` and friends,
    // pulled in by `use num;` regardless of whether this program's own
    // `main` ever calls them) `sym_visibility = "private"` too (`pipeline.
    // rs`'s own structured-vectorization stage needs `--symbol-dce` to see
    // them as removable), which MLIR's own pretty-printer renders as this
    // same `func.func private` prefix -- unrelated to this test's actual
    // claim, which is specifically about `arith.addi` never needing a
    // declared symbol of its own.
    assert!(
        !text.contains("func.func private @\"arith.addi\"")
            && !text.contains("func.func private @arith.addi"),
        "got:\n{text}"
    );
}

#[test]
fn a_direct_mlir_call_actually_computes_the_right_value() {
    let context = context();
    assert_eq!(
        run_i32(&context, "fn main() -> i32 { mlir::arith::addi(2, 3) }"),
        5
    );
}

/// The predicate attribute is the one genuinely non-uniform case (`arith.
/// cmpi`/`cmpf` need a static attribute beyond bare operands) -- confirmed
/// directly against real melior/MLIR behavior before committing `stdlib/
/// num`'s own 36 bodies to it (see `stdlib/num/num.cleave`'s own comment on
/// this): the predicate is the *raw integer* ordinal (`"2 : i64"` for
/// `slt`), not the symbolic name (`Attribute::parse` rejects a bare `"slt"`
/// -- that's custom per-op pretty-printer syntax, not generic attribute
/// text).
#[test]
fn a_comparison_with_a_named_predicate_attribute_lowers_and_executes_correctly() {
    let context = context();
    let text = lower(
        &context,
        "fn main() -> i32 { if mlir::arith::cmpi(1, 2, predicate: \"2 : i64\") { 10 } else { 20 } }",
    );
    assert!(
        text.contains("arith.cmpi") && text.contains("slt"),
        "got:\n{text}"
    );
    assert_eq!(
        run_i32(
            &context,
            "fn main() -> i32 { if mlir::arith::cmpi(1, 2, predicate: \"2 : i64\") { 10 } else { 20 } }"
        ),
        10
    );
    assert_eq!(
        run_i32(
            &context,
            "fn main() -> i32 { if mlir::arith::cmpi(2, 1, predicate: \"2 : i64\") { 10 } else { 20 } }"
        ),
        20
    );
}

/// A composite, multi-step primitive (the case that motivated moving to
/// this mechanism at all -- see the session's own design discussion: not
/// every useful primitive maps to exactly one MLIR op) needs *no new
/// sequencing machinery* -- it's just an ordinary cleave function body
/// (`let` chain + tail expression), each step an `mlir::...` call. `abs`
/// here chains `subi` + `cmpi` + `select` -- three real ops, zero Rust code
/// added to support it beyond what a single-op call already needed.
///
/// One real, deliberate limit this surfaces (not a bug): an `mlir::...`
/// call's own result is a fresh, otherwise-unconstrained type variable (see
/// `infer.rs::infer_expr`'s own `mlir::` branch) -- fine when it flows
/// straight into something that *does* constrain it (a function's own
/// declared return type, an ordinary algebra-dispatched operation), but an
/// intermediate `let` feeding *only* into further `mlir::` calls has
/// nothing to pin its type down at all and needs an explicit annotation,
/// same as any other genuinely unconstrained `let` in ML-style inference.
#[test]
fn a_composite_multi_step_mlir_body_executes_correctly() {
    let context = context();
    let src = "
        fn abs(a: i32) -> i32 {
            let neg: i32 = mlir::arith::subi(0, a);
            let is_neg: bool = mlir::arith::cmpi(a, 0, predicate: \"2 : i64\");
            mlir::arith::select(is_neg, neg, a)
        }
        fn main() -> i32 { abs(x) }
    ";
    assert_eq!(run_i32(&context, &src.replace("abs(x)", "abs(3 - 8)")), 5);
    assert_eq!(run_i32(&context, &src.replace("abs(x)", "abs(8 - 3)")), 5);
}

/// Named-argument syntax (`predicate: "..."`) is grammatically legal on any
/// call (see `grammar.pest`'s own `mlir_attr` comment: checked in `infer.rs`,
/// not blocked structurally) -- but semantically only ever means something
/// on a reserved `mlir::...` call. An ordinary call using it must still be
/// rejected, confirming the grammar addition didn't silently widen what
/// ordinary calls accept. `compile()` itself only surfaces parse/use-
/// resolution errors, not type errors (the same gap `main.rs`'s own
/// `check_type_errors` gate exists to close) -- `dump_program` is used here
/// instead specifically because it actually runs inference.
#[test]
fn named_arguments_on_an_ordinary_call_are_rejected() {
    let (result, _sources) = compile(
        vec![(
            "test.cleave".to_string(),
            "fn foo(x: i32) -> i32 { x } fn main() -> i32 { foo(x: \"5\") }".to_string(),
        )],
        &[],
    );
    let program = result.unwrap_or_else(|e| {
        panic!("expected a parse/use-resolution success (the type error is caught later): {e:?}")
    });
    let registry = Registry::build(&program);
    let (_, errs) = cleave::dump::dump_program(&program, &registry);
    assert!(
        !errs.is_empty(),
        "`foo(x: \"5\")` supplies zero real positional arguments to a one-parameter fn and must be rejected"
    );
}

const EXTERN_PRINT_SRC: &str =
    "extern fn print_i32(x: i32) -> i32; fn main() -> i32 { print_i32(42) }";

#[test]
fn an_extern_fn_call_lowers_to_a_private_declaration_plus_a_real_call() {
    let context = context();
    let text = lower(&context, EXTERN_PRINT_SRC);
    assert!(
        text.contains("func.func private @print_i32(i32) -> i32"),
        "got:\n{text}"
    );
    assert!(text.contains("call @print_i32"), "got:\n{text}");
}

/// The real end-to-end proof for `extern fn`: declares, calls, lowers to
/// the `llvm` dialect, registers the real Rust function pointer backing the
/// symbol (the same `ExecutionEngine::register_symbol` mechanism `main.rs`'s
/// own `--run` path uses), and JIT-invokes it -- asserting the JIT-returned
/// value is exactly `42` (not a coincidental default) proves the whole path
/// end to end without needing to capture stdout.
#[test]
fn an_extern_fn_call_actually_executes_through_a_registered_symbol() {
    let context = context();
    let (result, _sources) = compile(
        vec![("test.cleave".to_string(), EXTERN_PRINT_SRC.to_string())],
        &[],
    );
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    let units = collect_units(&program, &registry);
    let cps_program = convert_program(units);
    let mlir_types = collect_mlir_types(&program);
    let struct_schemas = collect_struct_schemas(&program);
    let mut module = lower_program(&context, &cps_program, &mlir_types, struct_schemas);
    assert!(module.as_operation().verify());

    let pass_manager = pass::PassManager::new(&context);
    // `num`'s own `Rem::mod` (prelude, unconditionally compiled into every
    // module regardless of whether the program actually calls it — see
    // `doc/backlog.md`'s own "dead-code elimination" item) has a real `if`/
    // `else` body, so `scf.if` is now present in *every* compiled module,
    // not just ones whose own source uses `if` — `create_to_llvm` alone
    // can't translate it (needs `create_scf_to_control_flow` first, same
    // reasoning as `run_i32`'s own doc comment below), found by direct
    // testing the moment `mod`/`rem` landed in the stdlib.
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    // Multiple independent conversion passes can each leave `builtin.
    // unrealized_conversion_cast` bridge ops between their own intermediate
    // representations behind -- found by direct testing, kept defensively:
    // real LLVM-IR translation can't handle a bare `unrealized_conversion_
    // cast` at all (`LLVM Translation failed for operation: builtin.
    // unrealized_conversion_cast`, a hard native crash). This is the
    // standard MLIR cleanup for exactly that situation: folds/cancels
    // chains of these casts away (`cast(cast(x, A->B), B->A) == x`), not a
    // real lowering step of its own.
    pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
    pass_manager
        .run(&mut module)
        .expect("lowering to the llvm dialect must succeed");

    let engine = melior::ExecutionEngine::new(&module, 2, &[], false, false);
    unsafe {
        engine.register_symbol("print_i32", cleave_rt::print_i32 as *mut ());
        engine.register_symbol("cleave_alloc", cleave_rt::cleave_alloc as *mut ());
        engine.register_symbol("cleave_alloc_rc", cleave_rt::cleave_alloc_rc as *mut ());
        engine.register_symbol("cleave_retain", cleave_rt::cleave_retain as *mut ());
        engine.register_symbol("cleave_release", cleave_rt::cleave_release as *mut ());
    }
    let mut out: i32 = -1;
    unsafe {
        engine
            .invoke_packed("main", &mut [&mut out as *mut i32 as *mut ()])
            .expect("JIT invocation must succeed");
    }
    assert_eq!(out, 42);
}

/// `export fn` lowers to an ordinary public `func.func` under its own
/// cleave name -- no `sym_visibility: "private"` (that's `extern`'s own
/// declaration-only case, `mlir_lower.rs:2137` at the time of writing), and
/// (deliberately, for a scalar signature -- see `lower_top_level_fn`'s own
/// doc comment) no `llvm.emit_c_interface` either, unlike `main`.
#[test]
fn an_export_fn_lowers_to_an_ordinary_public_func_under_its_own_name() {
    let context = context();
    let text = lower(
        &context,
        "export fn kernel(x: i32) -> i32 { x }\nfn main() -> i32 { 1 }",
    );
    assert!(
        text.contains("func.func @kernel(%arg0: i32) -> i32"),
        "got:\n{text}"
    );
    assert!(
        !text.contains("private @kernel"),
        "an export fn must stay publicly visible, got:\n{text}"
    );
}

/// `export(symbol)`'s own parenthesized override renames the real emitted
/// LLVM symbol, mirroring `extern(symbol)`'s identical shape in the
/// opposite direction.
#[test]
fn an_export_fn_with_a_symbol_override_lowers_under_the_overridden_name() {
    let context = context();
    let text = lower(
        &context,
        "export(real_kernel_symbol) fn kernel(x: i32) -> i32 { x }\nfn main() -> i32 { 1 }",
    );
    assert!(
        text.contains("func.func @real_kernel_symbol"),
        "got:\n{text}"
    );
    assert!(
        !text.contains("@kernel("),
        "the plain cleave name must not appear as its own symbol once overridden, got:\n{text}"
    );
}

/// `extern(...)`'s own parenthesized override -- the case a bare `extern fn`
/// can't cover: two algebra-impl methods sharing the *same* cleave-level
/// name (`print`) but each binding a genuinely different real C symbol
/// (`print_i32`/`print_i64`), the way `stdlib/io/io.cleave`'s own `Print<T>`
/// is actually built. Calling `print` as an ordinary, unqualified function
/// call (not an operator) dispatches through algebra resolution exactly
/// like `+`/`<` do -- confirmed directly (`gt(3, 2)` behaves identically to
/// `3 > 2`) before relying on it here.
const EXTERN_IMPL_PRINT_SRC: &str = "
algebra Print<T> { fn print(x: T) -> T; }
impl Print<i32> { extern(print_i32) fn print(x: i32) -> i32; }
impl Print<i64> { extern(print_i64) fn print(x: i64) -> i64; }
fn main() -> i32 { print(42); print(66:i64); 0 }
";

#[test]
fn an_extern_impl_methods_override_binds_a_distinct_symbol_per_impl() {
    let context = context();
    let text = lower(&context, EXTERN_IMPL_PRINT_SRC);
    assert!(
        text.contains("func.func private @print_i32(i32) -> i32"),
        "got:\n{text}"
    );
    assert!(
        text.contains("func.func private @print_i64(i64) -> i64"),
        "got:\n{text}"
    );
    assert!(text.contains("call @print_i32"), "got:\n{text}");
    assert!(text.contains("call @print_i64"), "got:\n{text}");
}

/// The real end-to-end proof: both impls' own `extern`-backed symbols
/// actually get called through, at their own correct (different) widths --
/// `print(42)`'s own return value flowing all the way back out to `main`'s
/// own JIT-observed result proves the `i32` call site genuinely executed,
/// not just that it verified.
#[test]
fn an_extern_impl_method_actually_executes_the_right_symbol_at_each_call_site() {
    let context = context();
    let (result, _sources) = compile(
        vec![("test.cleave".to_string(), EXTERN_IMPL_PRINT_SRC.to_string())],
        &[],
    );
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    let units = collect_units(&program, &registry);
    let cps_program = convert_program(units);
    let mlir_types = collect_mlir_types(&program);
    let struct_schemas = collect_struct_schemas(&program);
    let mut module = lower_program(&context, &cps_program, &mlir_types, struct_schemas);
    assert!(module.as_operation().verify());

    let pass_manager = pass::PassManager::new(&context);
    // `num`'s own `Rem::mod` (prelude, unconditionally compiled into every
    // module regardless of whether the program actually calls it — see
    // `doc/backlog.md`'s own "dead-code elimination" item) has a real `if`/
    // `else` body, so `scf.if` is now present in *every* compiled module,
    // not just ones whose own source uses `if` — `create_to_llvm` alone
    // can't translate it (needs `create_scf_to_control_flow` first, same
    // reasoning as `run_i32`'s own doc comment below), found by direct
    // testing the moment `mod`/`rem` landed in the stdlib.
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    // Multiple independent conversion passes can each leave `builtin.
    // unrealized_conversion_cast` bridge ops between their own intermediate
    // representations behind -- found by direct testing, kept defensively:
    // real LLVM-IR translation can't handle a bare `unrealized_conversion_
    // cast` at all (`LLVM Translation failed for operation: builtin.
    // unrealized_conversion_cast`, a hard native crash). This is the
    // standard MLIR cleanup for exactly that situation: folds/cancels
    // chains of these casts away (`cast(cast(x, A->B), B->A) == x`), not a
    // real lowering step of its own.
    pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
    pass_manager
        .run(&mut module)
        .expect("lowering to the llvm dialect must succeed");

    let engine = melior::ExecutionEngine::new(&module, 2, &[], false, false);
    unsafe {
        engine.register_symbol("print_i32", cleave_rt::print_i32 as *mut ());
        engine.register_symbol("print_i64", cleave_rt::print_i64 as *mut ());
        engine.register_symbol("cleave_alloc", cleave_rt::cleave_alloc as *mut ());
        engine.register_symbol("cleave_alloc_rc", cleave_rt::cleave_alloc_rc as *mut ());
        engine.register_symbol("cleave_retain", cleave_rt::cleave_retain as *mut ());
        engine.register_symbol("cleave_release", cleave_rt::cleave_release as *mut ());
    }
    let mut out: i32 = -1;
    unsafe {
        engine
            .invoke_packed("main", &mut [&mut out as *mut i32 as *mut ()])
            .expect("JIT invocation must succeed");
    }
    assert_eq!(out, 0);
}

/// De-risks the extern/array ABI boundary (see the plan this was built
/// against, in `doc/backlog.md`'s own "No string support at all" item's
/// eventual "Done" write-up) *before* string support depends on it: an
/// `extern fn`'s own declared `Ty::Array` parameter must not be lowered as
/// an ordinary `memref` argument — MLIR's default `convert-to-llvm`
/// conversion turns a `memref` crossing any `func.call` boundary into a
/// descriptor *struct*, not a bare pointer, which no hand-written
/// `cleave-rt` extern fn could plausibly match. `mlir_lower.rs`'s own
/// array-aware extern-call lowering extracts a raw pointer + a compile-time
/// known length explicitly instead, passing `(ptr, len)` as two ordinary
/// scalar arguments — this is the real, load-bearing proof that mechanism
/// works, not just that the module verifies.
#[test]
fn an_array_argument_crosses_an_extern_call_boundary_correctly() {
    let context = context();
    let (result, _sources) = compile(
        vec![(
            "test.cleave".to_string(),
            "extern fn sum_bytes(x: [i8; 3]) -> i32; fn main() -> i32 { sum_bytes([1, 2, 3]) }"
                .to_string(),
        )],
        &[],
    );
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    let units = collect_units(&program, &registry);
    let cps_program = convert_program(units);
    let mlir_types = collect_mlir_types(&program);
    let struct_schemas = collect_struct_schemas(&program);
    let mut module = lower_program(&context, &cps_program, &mlir_types, struct_schemas);
    assert!(module.as_operation().verify());

    let pass_manager = pass::PassManager::new(&context);
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    // Multiple independent conversion passes can each leave `builtin.
    // unrealized_conversion_cast` bridge ops between their own intermediate
    // representations behind -- found by direct testing, kept defensively:
    // real LLVM-IR translation can't handle a bare `unrealized_conversion_
    // cast` at all (`LLVM Translation failed for operation: builtin.
    // unrealized_conversion_cast`, a hard native crash). This is the
    // standard MLIR cleanup for exactly that situation: folds/cancels
    // chains of these casts away (`cast(cast(x, A->B), B->A) == x`), not a
    // real lowering step of its own.
    pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
    pass_manager
        .run(&mut module)
        .expect("lowering to the llvm dialect must succeed");

    let engine = melior::ExecutionEngine::new(&module, 2, &[], false, false);
    unsafe {
        engine.register_symbol("sum_bytes", cleave_rt::sum_bytes as *mut ());
        engine.register_symbol("cleave_alloc", cleave_rt::cleave_alloc as *mut ());
        engine.register_symbol("cleave_alloc_rc", cleave_rt::cleave_alloc_rc as *mut ());
        engine.register_symbol("cleave_retain", cleave_rt::cleave_retain as *mut ());
        engine.register_symbol("cleave_release", cleave_rt::cleave_release as *mut ());
    }
    let mut out: i32 = -1;
    unsafe {
        engine
            .invoke_packed("main", &mut [&mut out as *mut i32 as *mut ()])
            .expect("JIT invocation must succeed");
    }
    assert_eq!(out, 6, "1 + 2 + 3");
}

/// A `()`-returning `extern fn` (`extern fn touch_i32(x: i32);`, no `->`
/// clause) — a real C ABI shape never previously exercised by any real
/// example: every other `extern fn` in this codebase returns a real value.
/// `PrimOp::Extern`'s own lowering used to declare/call with `ty_to_mlir(ctx,
/// ty)` unconditionally, which has no real case for `()` either (falls
/// through to the generic-struct arm, `!llvm.ptr`) — requesting one bogus
/// pointer-typed result against a real C symbol that returns nothing at all.
/// Fixed the same way `lower_real_call`/`lower_top_level_fn` already handle
/// a unit-returning top-level fn: zero declared/requested MLIR results, and
/// the cleave-level result is never bound into `env` (nothing downstream can
/// look up a unit value anyway).
#[test]
fn a_unit_returning_extern_fn_can_be_called_correctly() {
    let context = context();
    let (result, _sources) = compile(
        vec![(
            "test.cleave".to_string(),
            "extern fn touch_i32(x: i32);\nfn main() -> i32 { touch_i32(5); 42 }".to_string(),
        )],
        &[],
    );
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    let units = collect_units(&program, &registry);
    let cps_program = convert_program(units);
    let mlir_types = collect_mlir_types(&program);
    let struct_schemas = collect_struct_schemas(&program);
    let mut module = lower_program(&context, &cps_program, &mlir_types, struct_schemas);
    assert!(module.as_operation().verify());
    let text = module.as_operation().to_string();
    // The real, structural proof: the declared extern signature (and its
    // own call site) must request *zero* results, not a bogus `!llvm.ptr`
    // (`ty_to_mlir`'s own generic-struct fallback for `()`, before the fix)
    // that happens not to crash here only because nothing ever reads it.
    assert!(
        text.contains("func.func private @touch_i32(i32)") && !text.contains("touch_i32(i32) -> "),
        "got:\n{text}"
    );
    assert!(
        text.contains("call @touch_i32(%c5_i32) : (i32) -> ()"),
        "got:\n{text}"
    );

    let pass_manager = pass::PassManager::new(&context);
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    // Multiple independent conversion passes can each leave `builtin.
    // unrealized_conversion_cast` bridge ops between their own intermediate
    // representations behind -- found by direct testing, kept defensively:
    // real LLVM-IR translation can't handle a bare `unrealized_conversion_
    // cast` at all (`LLVM Translation failed for operation: builtin.
    // unrealized_conversion_cast`, a hard native crash). This is the
    // standard MLIR cleanup for exactly that situation: folds/cancels
    // chains of these casts away (`cast(cast(x, A->B), B->A) == x`), not a
    // real lowering step of its own.
    pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
    pass_manager
        .run(&mut module)
        .expect("lowering to the llvm dialect must succeed");

    let engine = melior::ExecutionEngine::new(&module, 2, &[], false, false);
    unsafe {
        engine.register_symbol("touch_i32", cleave_rt::touch_i32 as *mut ());
        engine.register_symbol("cleave_alloc", cleave_rt::cleave_alloc as *mut ());
        engine.register_symbol("cleave_alloc_rc", cleave_rt::cleave_alloc_rc as *mut ());
        engine.register_symbol("cleave_retain", cleave_rt::cleave_retain as *mut ());
        engine.register_symbol("cleave_release", cleave_rt::cleave_release as *mut ());
    }
    let mut out: i32 = -1;
    unsafe {
        engine
            .invoke_packed("main", &mut [&mut out as *mut i32 as *mut ()])
            .expect("JIT invocation must succeed");
    }
    assert_eq!(out, 42);
}

/// The real end-to-end proof for string support: `print("hi")` — a string
/// literal (desugared to an `[i8; 2]` array literal, `lower.rs::
/// lower_string_lit`) dispatched through the new `impl<const N: i32>
/// Print<[i8; N]>` (`stdlib/io/io.cleave`), crossing the extern boundary via
/// the same array-aware lowering `an_array_argument_crosses_an_extern_call_
/// boundary_correctly` above already proved. `Print<[i8;N]>::print`'s own
/// declared return type is `[i8; N]` (identity, matching every other
/// `Print<T>` impl's "prints and returns unchanged" contract) even though
/// the real `print_bytes` extern symbol returns a plain `i64` byte count —
/// `mlir_lower.rs`'s own array-return reconciliation must discard that
/// scalar and thread the original array through, not try to reconstruct a
/// `[i8;N]` from an `i64`.
#[test]
fn a_string_literal_printed_via_print_writes_the_right_bytes_to_stdout() {
    let context = context();
    let (result, _sources) = compile(
        vec![(
            "test.cleave".to_string(),
            "use io;\nfn main() -> i32 { print(\"hi\"); 0 }".to_string(),
        )],
        &[],
    );
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    let units = collect_units(&program, &registry);
    let cps_program = convert_program(units);
    let mlir_types = collect_mlir_types(&program);
    let struct_schemas = collect_struct_schemas(&program);
    let mut module = lower_program(&context, &cps_program, &mlir_types, struct_schemas);
    assert!(module.as_operation().verify());
    let text = module.as_operation().to_string();
    assert!(
        text.contains("call @print_bytes"),
        "expected a real call to print_bytes, got:\n{text}"
    );
    assert!(
        text.contains("llvm.inttoptr"),
        "expected the array-aware pointer extraction, got:\n{text}"
    );

    let pass_manager = pass::PassManager::new(&context);
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    // Multiple independent conversion passes can each leave `builtin.
    // unrealized_conversion_cast` bridge ops between their own intermediate
    // representations behind -- found by direct testing, kept defensively:
    // real LLVM-IR translation can't handle a bare `unrealized_conversion_
    // cast` at all (`LLVM Translation failed for operation: builtin.
    // unrealized_conversion_cast`, a hard native crash). This is the
    // standard MLIR cleanup for exactly that situation: folds/cancels
    // chains of these casts away (`cast(cast(x, A->B), B->A) == x`), not a
    // real lowering step of its own.
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
    }
    register_io_symbols(&engine);
    let mut out: i32 = -1;
    unsafe {
        engine
            .invoke_packed("main", &mut [&mut out as *mut i32 as *mut ()])
            .expect("JIT invocation must succeed");
    }
    assert_eq!(out, 0);
}

/// A generic top-level `fn` with *no call site anywhere in the program*
/// used to crash CPS conversion outright (`could not resolve call to ...`,
/// an unresolved-type-variable key never matching anything in `call_index`)
/// -- found for real adding `println<T: Print>` (`stdlib/io/io.cleave`) to
/// the prelude-adjacent `io` crate: no top-level generic `fn` in this
/// codebase had ever gone genuinely uncalled before (every prior one was
/// always reached from *some* test/example), so `cps.rs::collect_units`'s
/// own `keys.is_empty()` check -- meant to detect "non-generic, no
/// instantiations to iterate, build one direct unit" -- had never been
/// exercised against a function that was *also* generic, where
/// `keys.is_empty()` is equally true for the completely different reason
/// "genuinely generic, simply never called." Fixed by additionally checking
/// `fn_result`'s own concreteness before treating a `keys.is_empty()` fn as
/// buildable directly. This test's own `unused<T: Print>` is never called
/// anywhere in `main` -- its only job is to exist, uncalled, alongside a
/// program that otherwise runs fine.
#[test]
fn an_uncalled_generic_top_level_fn_does_not_crash_compilation() {
    let context = context();
    let src = "
        use io;
        fn unused<T: Print>(x: T) -> T {
            print(x)
        }
        fn main() -> i32 {
            print(1);
            0
        }";
    assert_eq!(run_i32(&context, src), 0);
}

/// `'x'`/`'\n'` -- a char literal is full erasure at lowering time (`lower.
/// rs::lower_char_lit`), sugar for a plain `i8`-suffixed `NumberLit`; end to
/// end, it must behave *exactly* like writing the byte value out by hand.
#[test]
fn a_char_literal_evaluates_to_its_ascii_byte_value() {
    let context = context();
    let src = "
        fn main() -> i32 {
            let a: i8 = 'x';
            let n: i8 = '\\n';
            let bs: i8 = '\\\\';
            let q: i8 = '\\'';
            if a == 120:i8 and n == 10:i8 and bs == 92:i8 and q == 39:i8 { 1 } else { 0 }
        }";
    assert_eq!(run_i32(&context, src), 1);
}

/// A string literal's own escapes (`\n`/`\t`/`\\`/`\"`) decode to their real
/// byte values end to end, not just at the AST level (`tests/lower.rs`'s own
/// `string_literal_escapes_decode_to_their_real_byte_values` covers that
/// half) -- indexed directly as the `[i8; N]` array a string literal always
/// desugars to.
#[test]
fn a_string_literal_with_escapes_indexes_to_the_right_decoded_bytes() {
    let context = context();
    let src = r#"
        fn main() -> i32 {
            let s: [i8; 4] = "a\tb\n";
            if s[0] == 97:i8 and s[1] == 9:i8 and s[2] == 98:i8 and s[3] == 10:i8 { 1 } else { 0 }
        }"#;
    assert_eq!(run_i32(&context, src), 1);
}

#[test]
fn a_while_loop_lowers_to_scf_while() {
    let context = context();
    let text = lower(
        &context,
        "fn main() -> i32 { let mut acc = 0; while acc < 5 { acc = acc + 1; }; acc }",
    );
    assert!(text.contains("scf.while"), "got:\n{text}");
    assert!(text.contains("scf.condition"), "got:\n{text}");
    assert!(text.contains("scf.yield"), "got:\n{text}");
}

#[test]
fn a_while_loop_actually_computes_the_right_value() {
    let context = context();
    assert_eq!(
        run_i32(
            &context,
            "fn main() -> i32 { let mut acc = 0; while acc < 5 { acc = acc + 1; }; acc }"
        ),
        5
    );
}

/// The real end-to-end proof for loops carrying *more than one* value at
/// once (the loop's own implicit index *and* a user-declared accumulator,
/// see `lower_loop`'s own doc comment on why each carried value's own type
/// isn't necessarily `result_type`) -- `sum(0..9) == 45`, not a value
/// either the loop mechanism or the arithmetic alone could produce by
/// coincidence.
#[test]
fn a_for_loop_with_an_accumulator_actually_computes_the_right_value() {
    let context = context();
    assert_eq!(
        run_i32(
            &context,
            "fn main() -> i32 { let mut acc = 0; for i in 0..10 { acc = acc + i; }; acc }"
        ),
        45
    );
}

/// `doc/backlog-done.md`'s own "`for x in array`" item — element-based
/// iteration, the real end-to-end proof: `cps.rs`'s new `ExprKind::ForIn`
/// arm correctly derives the array's own size from its type (`ctx.node_
/// types`, not a user-written bound) and binds `x` to the *loaded element*
/// (via one extra `LetPrim{Load}`), not the index — `1+2+3+4 = 10`, not `10`
/// (the sum of indices `0+1+2+3`) either, the real discriminator between
/// "iterating elements" and "iterating indices by coincidence".
#[test]
fn a_for_in_loop_over_an_array_sums_its_own_elements_not_its_indices() {
    let context = context();
    let src = "fn main() -> i32 { let arr = [1, 2, 3, 4]; let mut total = 0; for x in arr { total = total + x; }; total }";
    assert_eq!(run_i32(&context, src), 10);
}

/// A `for x in arr` loop's own carried-state threading (an *outer* mutated
/// variable, not just the loop-local accumulator) — mirrors `a_loop_
/// carrying_two_different_types_computes_the_right_value`'s own precedent,
/// proving `mutated_free_vars`'s new `ForIn` arm actually threads a second,
/// independently-typed carried value correctly, not just that translation
/// doesn't crash.
#[test]
fn a_for_in_loop_carries_an_outer_mutated_variable_correctly() {
    let context = context();
    let src = "
        fn main() -> i32 {
            let arr = [1, 2, 3, 4];
            let mut total = 0;
            let mut count = 0.0;
            for x in arr {
                total = total + x;
                count = count + 1.0;
            };
            if total == 10 and count == 4.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// A 1-D array literal (`PrimOp::Array`) lowers to `memref.alloc` plus one
/// `memref.store` per element -- indices go through `arith.index_cast`
/// first (`memref.load`/`store` require `index`-typed operands specifically,
/// not the ordinary `i32` cleave indices are otherwise typed as -- see
/// `lower_array_load`'s own doc comment).
#[test]
fn an_array_literal_lowers_to_alloc_plus_stores() {
    let context = context();
    let text = lower(&context, "fn main() -> i32 { let a = [1, 2, 3]; a[0] }");
    assert!(
        text.contains("memref.alloc() : memref<3xi32>"),
        "got:\n{text}"
    );
    assert!(text.contains("memref.store"), "got:\n{text}");
    assert!(text.contains("memref.load"), "got:\n{text}");
    assert!(text.contains("arith.index_cast"), "got:\n{text}");
}

/// The real end-to-end proof: a mutated element (`a[0] = 10`, a real
/// `memref.store` effect -- see `cps.rs`'s own "an array is a stable
/// reference, mutated in place" design) is actually visible on a later read
/// through the *same* memref, not a stale/copied value -- `15`, not `6`
/// (`1+2+3`), so a silent copy-instead-of-mutate bug wouldn't pass.
#[test]
fn an_array_literal_write_then_read_computes_the_right_value() {
    let context = context();
    let src = "fn main() -> i32 { let mut a = [1, 2, 3]; a[0] = 10; a[0] + a[1] + a[2] }";
    assert_eq!(run_i32(&context, src), 15);
}

/// A nested array literal (`[[1,2,3],[4,5,6]]`) -- `ExprKind::ArrayLit`'s own
/// CPS conversion is fully generic over each element, so the inner rows are
/// each their own, separately-built `PrimOp::Array` value; this file's own
/// representation is one *flat* memref, never memref-of-memrefs (see
/// `array_memref_type`'s own doc comment), so the outer `Array` copies each
/// inner row's contents into the right slice instead of referencing it
/// (`copy_nested_array`) -- proven here by a real multi-index write (`a[1,
/// 2] = 60`) landing in the right flat position and every other element
/// surviving the copy unchanged: `1+2+3+4+5+60 == 75`.
#[test]
fn a_nested_array_literal_lowers_and_computes_the_right_value() {
    let context = context();
    let text = lower(
        &context,
        "fn main() -> i32 { let a = [[1, 2, 3], [4, 5, 6]]; a[0, 0] }",
    );
    assert!(
        text.contains("memref.alloc() : memref<2x3xi32>"),
        "got:\n{text}"
    );
    let src = "
        fn main() -> i32 {
            let mut a = [[1, 2, 3], [4, 5, 6]];
            a[1, 2] = 60;
            a[0, 0] + a[0, 1] + a[0, 2] + a[1, 0] + a[1, 1] + a[1, 2]
        }
    ";
    assert_eq!(run_i32(&context, src), 75);
}

/// `[value; N]` (`PrimOp::ArrayRepeat`) -- `N` comes from the `LetPrim`'s own
/// declared array type, not from re-evaluating a separate count operand each
/// time (see `lower_array_repeat`'s own doc comment); nested (`[[0; K]; N]`,
/// `examples/matmul.cleave`'s own accumulator-init shape) goes through the
/// same elementwise-copy path a nested literal does.
#[test]
fn array_repeat_including_nested_computes_the_right_value() {
    let context = context();
    let src = "
        fn main() -> i32 {
            let mut a = [7; 4];
            let mut b = [[0; 3]; 2];
            b[1, 2] = 99;
            a[0] + a[1] + a[2] + a[3] + b[0, 0] + b[1, 2]
        }
    ";
    assert_eq!(run_i32(&context, src), 127);
}

/// A `struct` is a stable reference (`llvm.alloca` — via a real heap-backed
/// `cleave_alloc` call, see `mlir_lower.rs::alloc_struct`'s own doc comment
/// for why not literal `llvm.alloca` — plus `llvm.getelementptr`/`llvm.
/// store`/`llvm.load`), not an `!llvm.struct` SSA value built via `undef`+
/// `insertvalue` — found necessary by direct testing: a struct returned
/// from one function and read by its caller came back reading garbage once
/// its storage lived in the *constructing* function's own stack frame.
#[test]
fn a_struct_literal_lowers_to_a_heap_alloc_plus_field_gep_and_stores() {
    let context = context();
    let text = lower(
        &context,
        "struct Pair { a: i32, b: i32 } fn main() -> i32 { let p = Pair(a: 1, b: 2); p.a + p.b }",
    );
    assert!(text.contains("call @cleave_alloc"), "got:\n{text}");
    assert!(text.contains("llvm.getelementptr"), "got:\n{text}");
    assert!(text.contains("llvm.store"), "got:\n{text}");
    assert!(text.contains("llvm.load"), "got:\n{text}");
}

/// The real end-to-end proof for scalar fields: construction (`llvm.
/// alloca`-via-`cleave_alloc` + `llvm.store` per field) followed by field
/// reads (`llvm.getelementptr` + `llvm.load`) round-trips the exact values
/// written -- `12` (`5+7`), not a coincidental default.
#[test]
fn a_struct_literal_construction_and_field_read_computes_the_right_value() {
    let context = context();
    let src = "
        struct Pair { a: i32, b: i32 }
        fn main() -> i32 {
            let p = Pair(a: 5, b: 7);
            p.a + p.b
        }
    ";
    assert_eq!(run_i32(&context, src), 12);
}

/// Direct field-mutation assignment (`p.a = 10`, `PrimOp::FieldStore`) --
/// mirrors `an_array_literal_write_then_read_computes_the_right_value`'s own
/// shape for arrays: a real effect through the struct's own stable pointer,
/// visible on a later read through the *same* reference -- `12` (`10+2`),
/// not `3` (`1+2`), so a silent copy-instead-of-mutate bug wouldn't pass.
#[test]
fn a_struct_field_mutation_write_then_read_computes_the_right_value() {
    let context = context();
    let src = "
        struct Pair { a: i32, b: i32 }
        fn main() -> i32 {
            let mut p = Pair(a: 1, b: 2);
            p.a = 10;
            p.a + p.b
        }
    ";
    assert_eq!(run_i32(&context, src), 12);
}

/// A struct field that's itself array-typed can't be an `!llvm.struct`
/// field directly (confirmed empirically: `Type::parse`+`module.verify()`
/// rejects a `memref` operand to `llvm.insertvalue`, "operand #1 must be
/// primitive LLVM type") -- it's embedded *inline* as an `!llvm.array`
/// instead, addressed via `llvm.getelementptr` walking straight through the
/// struct *and* the array in one instruction (see `mlir_lower.rs::
/// struct_llvm_type`'s own doc comment). `p.values[0] = 10` proves a real,
/// runtime-indexed write through this path lands correctly: `12` (`10+2`),
/// not `3` (`1+2`).
#[test]
fn a_struct_with_an_array_field_computes_the_right_value() {
    let context = context();
    let src = "
        struct Pair { values: [i32; 2] }
        fn main() -> i32 {
            let mut p = Pair(values: [1, 2]);
            p.values[0] = 10;
            p.values[0] + p.values[1]
        }
    ";
    assert_eq!(run_i32(&context, src), 12);
}

/// `doc/backlog.md`'s own "Nested struct-as-field, unverified" item —
/// "designed for but never actually exercised end to end." A struct field
/// whose own type is *another* struct: `ty_to_llvm_field_type`'s own
/// fallback treats it identically to any other struct value (an opaque
/// `!llvm.ptr`, reference semantics), so a `Triangle`'s own `a`/`b`/`c`
/// fields are each a real, independent heap-allocated `Point` — proven by
/// reading through *two* levels of field access (`t.a.x`) and confirming
/// the three points stay independently addressable, not aliased.
#[test]
fn a_struct_field_whose_own_type_is_another_struct_computes_the_right_value() {
    let context = context();
    let src = "
        struct Point { x: i32, y: i32 }
        struct Triangle { a: Point, b: Point, c: Point }
        fn main() -> i32 {
            let t = Triangle(a: Point(x: 1, y: 2), b: Point(x: 10, y: 20), c: Point(x: 100, y: 200));
            t.a.x + t.b.y + t.c.x
        }
    ";
    assert_eq!(run_i32(&context, src), 1 + 20 + 100);
}

/// The mutation counterpart — a *nested* struct field mutated through two
/// levels of field access (`t.a.x = ...`), mirrors `a_struct_field_
/// mutation_write_then_read_computes_the_right_value`'s own "write then
/// read through the same reference" shape, one level deeper.
#[test]
fn a_nested_struct_fields_own_field_mutation_write_then_read_computes_the_right_value() {
    let context = context();
    let src = "
        struct Point { x: i32, y: i32 }
        struct Triangle { a: Point, b: Point, c: Point }
        fn main() -> i32 {
            let mut t = Triangle(a: Point(x: 1, y: 2), b: Point(x: 10, y: 20), c: Point(x: 100, y: 200));
            t.a.x = 1000;
            t.a.x + t.b.y + t.c.x
        }
    ";
    assert_eq!(run_i32(&context, src), 1000 + 20 + 100);
}

/// An array whose own *element* type is a struct, not a scalar — used to be
/// a hard native MLIR crash (`invalid memref element type`, a
/// `STATUS_STACK_BUFFER_OVERRUN` abort, since `array_memref_type` built a
/// `memref` unconditionally even for a struct-typed leaf's own opaque
/// `!llvm.ptr` element type — see `doc/backlog.md`'s own former "Array of
/// struct elements" item). Fixed: a struct-leaf array now gets a real heap
/// allocation shaped as an inline `!llvm.array` instead (`array_leaf_is_
/// struct`, `lower_array_construct`'s own struct-leaf branch).
#[test]
fn an_array_of_structs_computes_the_right_value() {
    let context = context();
    let src = "
        struct Point { x: i32, y: i32 }
        fn main() -> i32 {
            let pts = [Point(x: 1, y: 2), Point(x: 10, y: 20), Point(x: 100, y: 200)];
            pts[0].x + pts[1].y + pts[2].x
        }
    ";
    assert_eq!(run_i32(&context, src), 1 + 20 + 100);
}

/// The struct-*field* variant — same root cause as the test above, not a
/// separate bug: the array literal `[Point(...), Point(...)]` is built via
/// the ordinary `PrimOp::Array` path regardless of its eventual destination.
/// Exercises `copy_array_into_llvm_field`'s own pointer-source branch (Stage
/// 5): the source array is itself `!llvm.ptr`-backed, copied element-by-
/// element into the struct's own embedded `!llvm.array` field.
#[test]
fn a_struct_field_whose_own_type_is_an_array_of_structs_computes_the_right_value() {
    let context = context();
    let src = "
        struct Point { x: i32, y: i32 }
        struct Path { pts: [Point; 2] }
        fn main() -> i32 {
            let p = Path(pts: [Point(x: 1, y: 2), Point(x: 10, y: 20)]);
            p.pts[0].x + p.pts[1].y
        }
    ";
    assert_eq!(run_i32(&context, src), 1 + 20);
}

/// A loop carrying two *differently*-typed values at once (an `i32` counter
/// and an `f32` accumulator), inside a function whose own return type
/// matches *neither* directly (only through a comparison) -- found by
/// direct testing: `lower_loop` used to materialize every carried value's
/// own *initial* literal using one shared expected type (the enclosing
/// function's own return type), silently miscompiling whichever carried
/// position didn't happen to match it (masked by every earlier loop test,
/// which only ever carried one type, always equal to the function's own
/// return type). `total`'s own accumulation only reaches `5.0` if every one
/// of its five `+ 1.0` additions actually ran as a real `f32` op.
#[test]
fn a_loop_carrying_two_different_types_computes_the_right_value() {
    let context = context();
    let src = "
        fn main() -> i32 {
            let mut i = 0;
            let mut total = 0.0;
            while i < 5 {
                total = total + 1.0;
                i = i + 1;
            };
            if total > 4.5 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// A bodyless-`else` `if` used as a discarded statement, reassigning an
/// outer variable, *inside* a loop body — `mlir_lower.rs::lower_if` used to
/// support only a single-parameter join (the `if`'s own value); this needs
/// two positions at once (the `if`'s own unit value, discarded, plus the
/// reassigned `saw_three`), confirmed broken before `lower_if` was
/// generalized to `CFunDef::carried_types` the same way `lower_loop` already
/// was. `saw_three` only reaches `1` if the write actually happened on
/// exactly the iteration where `i == 3`.
#[test]
fn an_if_with_no_else_reassigning_an_outer_variable_inside_a_loop_computes_the_right_value() {
    let context = context();
    let src = "
        fn main() -> i32 {
            let mut i = 0;
            let mut saw_three = 0;
            while i < 5 {
                if i == 3 {
                    saw_three = 1;
                };
                i = i + 1;
            };
            saw_three
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// The same shape, but with a real (non-unit) `else` too, and *two* outer
/// variables reassigned across the branches — `evens`/`odds` counted by
/// which arm actually ran each iteration.
#[test]
fn an_if_else_reassigning_two_outer_variables_inside_a_loop_computes_the_right_value() {
    let context = context();
    let src = "
        fn main() -> i32 {
            let mut i = 0;
            let mut evens = 0;
            let mut odds = 0;
            while i < 6 {
                if i == 0 or i == 2 or i == 4 {
                    evens = evens + 1;
                } else {
                    odds = odds + 1;
                };
                i = i + 1;
            };
            evens * 10 + odds
        }
    ";
    assert_eq!(run_i32(&context, src), 33);
}

/// An `if` carrying *both* a real (used-elsewhere) value *and* a reassigned
/// outer variable at once — the join's own first position (the `if`'s
/// value, `-1`/`1`) is deliberately never read by the caller (`classify`'s
/// own tail is `label`, not the `if` itself), proving the multi-position
/// join binds each position to the *right* `scf.if` result independently,
/// not just "whichever one happens to be used."
#[test]
fn an_if_with_both_a_value_and_a_carried_variable_computes_the_right_value() {
    let context = context();
    let src = "
        fn classify(n: i32) -> i32 {
            let mut label = 0;
            if n < 0 {
                label = 1;
                -1
            } else {
                label = 2;
                1
            };
            label
        }
        fn main() -> i32 { classify(-5) }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

// ------------------------------------------------------------ closure conversion

/// Stage A: a `let`-bound lambda with no captures, called directly by name —
/// the simplest possible slice of closure conversion, verified end-to-end.
#[test]
fn a_lambda_with_no_captures_actually_computes_the_right_value() {
    let context = context();
    let src = "fn main() -> i32 { let add_captured = fn(x) { x + 10 }; add_captured(5) }";
    assert_eq!(run_i32(&context, src), 15);
}

/// Stage A: a lambda referencing an enclosing-scope variable — its own
/// generated unit must carry `base` as a leading, spliced-in argument at
/// every call site (`ConcreteUnit`'s own widened `params`/`param_types`).
#[test]
fn a_lambda_that_captures_an_enclosing_variable_actually_computes_the_right_value() {
    let context = context();
    let src = "fn main() -> i32 { let base = 100; let add_base = fn(x) { x + base }; add_base(5) }";
    assert_eq!(run_i32(&context, src), 105);
}

/// Stage A: a `let`-bound lambda is a syntactic value, so it gets
/// generalized (real Hindley-Milner let-polymorphism) exactly like a
/// top-level generic `fn` — `id` gets called at two different concrete
/// types from two different call sites, each independently specialized
/// (`LambdaScheme`/`monomorphize.rs`'s own lambda worklist).
#[test]
fn a_generic_lambda_is_let_polymorphic_like_a_top_level_generic_fn() {
    let context = context();
    let src = "
        fn main() -> i32 {
            let id = fn(x) { x };
            let a = id(1);
            let b = id(1.5);
            if b > 1.0 { a } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// A capture is snapshotted at the lambda's own `let` (its current `CVal`
/// gathered from `env` right then), not captured by reference — a `let mut`
/// reassignment *after* the closure is built must not be visible to it. This
/// is one of the two open risks the closure-conversion plan flagged
/// explicitly as needing its own stated decision plus a test, not an
/// accident of implementation order (see `CVal::Closure`'s own doc comment).
#[test]
fn a_captured_variable_is_snapshotted_at_the_lambda_not_captured_by_reference() {
    let context = context();
    let src = "
        fn main() -> i32 {
            let mut x = 1;
            let f = fn() { x };
            x = 2;
            f()
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// Stage B: `apply`'s own `f: (i32) -> i32` parameter is fully concrete
/// (`(i32) -> i32` is just an ordinary type, no generic involved at all) —
/// so nothing about `apply` itself is generic in `monomorphize.rs`'s sense;
/// the higher-order specialization (`apply[f=<lambda...>]`, `collect_units`'s
/// own `build_higher_order_specializations`) is the entire reason this can
/// resolve at all.
#[test]
fn passing_a_lambda_to_a_higher_order_parameter_actually_computes_the_right_value() {
    let context = context();
    let src = "
        fn apply(f: (i32) -> i32, x: i32) -> i32 { f(x) }
        fn main() -> i32 {
            let inc = fn(x) { x + 1 };
            apply(inc, 5)
        }
    ";
    assert_eq!(run_i32(&context, src), 6);
}

/// Two distinct callables passed to the *same* higher-order callee at two
/// different call sites must each get their own independent specialization
/// (`(callee, [erased position -> resolved callable])`, memoized) — proven
/// by the two results actually differing (`6`, not e.g. `6 + 6` from a
/// wrongly-shared one).
#[test]
fn two_distinct_callables_passed_to_the_same_higher_order_callee_get_separate_specializations() {
    let context = context();
    let src = "
        fn apply(f: (i32) -> i32, x: i32) -> i32 { f(x) }
        fn main() -> i32 {
            let inc = fn(x) { x + 1 };
            let dec = fn(x) { x - 1 };
            apply(inc, 5) + apply(dec, 5)
        }
    ";
    assert_eq!(run_i32(&context, src), 10);
}

/// A higher-order call reached indirectly (through an intermediate,
/// ordinary top-level `fn`, not the same function the lambda was `let`-bound
/// in) — `doc/user_guide.md`'s own existing higher-order-functions example,
/// run for real for the first time (previously caveated as "type-checks but
/// can't run yet").
#[test]
fn user_guide_higher_order_example_actually_runs() {
    let context = context();
    let src = "
        fn apply(f: (i32) -> i32, x: i32) -> i32 { f(x) }
        fn g() -> i32 {
            let inc = fn(x) { x + 1 };
            apply(inc, 5)
        }
        fn main() -> i32 { g() }
    ";
    assert_eq!(run_i32(&context, src), 6);
}

// ------------------------------------------------------------ div/mod/rem/bitwise (stdlib/num)

#[test]
fn integer_division_truncates_toward_zero() {
    let context = context();
    assert_eq!(run_i32(&context, "fn main() -> i32 { 17 / 5 }"), 3);
    assert_eq!(run_i32(&context, "fn main() -> i32 { -17 / 5 }"), -3);
}

#[test]
fn float_division_computes_the_right_value() {
    let context = context();
    let src = "fn main() -> i32 { let d: f64 = 10.0 / 4.0; if d == 2.5 { 1 } else { 0 } }";
    assert_eq!(run_i32(&context, src), 1);
}

/// `div` works for every declared `Ring<T>` width, not just `i32` — each
/// one needed its own `mlir::arith::divsi`/`divf` impl (see `stdlib/num/
/// num.cleave`), not just the algebra's own declaration.
#[test]
fn division_works_across_every_ring_width() {
    let context = context();
    let src = "
        fn main() -> i32 {
            let a8: i8 = 20;
            let b8: i8 = 3;
            let d8: i8 = a8 / b8;
            let a16: i16 = 20;
            let b16: i16 = 3;
            let d16: i16 = a16 / b16;
            let a64: i64 = 20;
            let b64: i64 = 3;
            let d64: i64 = a64 / b64;
            let ok8 = if d8 == 6 { 1 } else { 0 };
            let ok16 = if d16 == 6 { 1 } else { 0 };
            let ok64 = if d64 == 6 { 1 } else { 0 };
            ok8 + ok16 + ok64
        }
    ";
    assert_eq!(run_i32(&context, src), 3);
}

/// `rem` is truncated-division remainder (sign follows the dividend, same
/// as MLIR's own `arith.remsi`/C's `%`) — `mod` is Euclidean/floored
/// (sign follows the divisor, always non-negative for a positive divisor).
/// The two must actually differ on a negative operand, or the whole reason
/// `grammar.md` insists on two distinct names would be moot.
#[test]
fn mod_and_rem_differ_correctly_on_negative_operands() {
    let context = context();
    assert_eq!(run_i32(&context, "fn main() -> i32 { rem(-17, 5) }"), -2);
    assert_eq!(run_i32(&context, "fn main() -> i32 { mod(-17, 5) }"), 3);
    // Both operands positive -- rem and mod must agree.
    assert_eq!(run_i32(&context, "fn main() -> i32 { rem(17, 5) }"), 2);
    assert_eq!(run_i32(&context, "fn main() -> i32 { mod(17, 5) }"), 2);
}

#[test]
fn bitwise_operators_compute_the_right_values() {
    let context = context();
    assert_eq!(run_i32(&context, "fn main() -> i32 { bitand(12, 10) }"), 8);
    assert_eq!(run_i32(&context, "fn main() -> i32 { bitor(12, 10) }"), 14);
    assert_eq!(run_i32(&context, "fn main() -> i32 { bitxor(12, 10) }"), 6);
    assert_eq!(run_i32(&context, "fn main() -> i32 { bitnot(0) }"), -1);
    assert_eq!(run_i32(&context, "fn main() -> i32 { shl(1, 4) }"), 16);
    assert_eq!(run_i32(&context, "fn main() -> i32 { shr(16, 2) }"), 4);
    // `shr` is the arithmetic (sign-extending) right shift -- a negative
    // input must stay negative, not fill with zeros.
    assert_eq!(run_i32(&context, "fn main() -> i32 { shr(-16, 2) }"), -4);
}

/// `bitnot` specifically needed a real fix mid-session: an inline `-1`
/// directly inside the `mlir::arith::xori(a, -1)` call independently
/// defaults to `i32` (an `mlir::` call's own arguments don't cross-unify —
/// no real declared signature to unify against), mismatching `a`'s own
/// width for every `Bitwise<T>` impl but `i32` itself. Fixed with an
/// explicit `let all_bits: i8 = -1;` intermediate (same idiom as `Logic::
/// implies`'s own `let not_a: bool = ...`) — verified here across every
/// non-`i32` width specifically, so a regression to the inline form would
/// be caught immediately instead of only failing silently for `i8`/`i16`/
/// `i64` callers nobody happened to test.
#[test]
fn bitnot_works_across_every_non_i32_width() {
    let context = context();
    let src = "
        fn main() -> i32 {
            let a8: i8 = 5;
            let n8: i8 = bitnot(a8);
            let a16: i16 = 5;
            let n16: i16 = bitnot(a16);
            let a64: i64 = 5;
            let n64: i64 = bitnot(a64);
            let ok8 = if n8 == -6 { 1 } else { 0 };
            let ok16 = if n16 == -6 { 1 } else { 0 };
            let ok64 = if n64 == -6 { 1 } else { 0 };
            ok8 + ok16 + ok64
        }
    ";
    assert_eq!(run_i32(&context, src), 3);
}

// ------------------------------------------------------------ inherent impls

/// `doc/user_guide.md`'s own existing "Inherent impls" example, run for
/// real via dot-call syntax for the first time (previously caveated as
/// "type-checks but can't be JIT-executed yet").
#[test]
fn a_non_generic_inherent_method_called_via_dot_syntax_actually_runs() {
    let context = context();
    let src = "
        struct Vec2 { x: f64, y: f64 }
        impl struct Vec2 {
            fn magnitude_sq(v) -> f64 { v.x * v.x + v.y * v.y }
        }
        fn main() -> i32 {
            let v = Vec2(x: 1.0, y: 2.0);
            if v.magnitude_sq() == 5.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// Mutual recursion between two sibling inherent methods on the *same*
/// struct — `infer_inherent_impl_block`'s own stated reason for existing
/// (both methods inferred together, sharing one `Infer`), now proven all
/// the way through real execution: `w.dec().is_odd()` calling back into a
/// sibling `is_even`, and vice versa.
#[test]
fn mutually_recursive_inherent_methods_on_the_same_struct_actually_run() {
    let context = context();
    let src = "
        struct Wrapped { n: i32 }
        impl struct Wrapped {
            fn dec(w) -> Wrapped { Wrapped(n: w.n - 1) }
            fn is_even(w) -> bool { if w.n == 0 { true } else { w.dec().is_odd() } }
            fn is_odd(w) -> bool { if w.n == 0 { false } else { w.dec().is_even() } }
        }
        fn main() -> i32 {
            let w = Wrapped(n: 7);
            if w.is_odd() { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// A *generic* inherent impl (`impl<T> Boxed<T> { ... }`) — `cps.rs::
/// collect_units`'s own `InherentImpl` branch reads specializations back
/// from `monomorphize.rs`'s own inherent-method worklist, mirroring the
/// generic-algebra-impl case exactly but through `InherentTemplate`/
/// `derive_inherent_instantiation` instead.
#[test]
fn a_generic_inherent_method_called_via_dot_syntax_actually_runs() {
    let context = context();
    let src = "
        struct Boxed<T> { value: T }
        impl<T: Ring> struct Boxed<T> {
            fn doubled(b) -> T { add(b.value, b.value) }
        }
        fn main() -> i32 {
            let b = Boxed(value: 21);
            b.doubled()
        }
    ";
    assert_eq!(run_i32(&context, src), 42);
}

/// The same generic inherent method, called at *two different* concrete
/// types from two different call sites — each needs its own independent
/// specialization (mirrors `a_generic_function_called_at_two_types_
/// converts_to_two_separate_specializations` in `cleave/tests/cps.rs`, one
/// level up for inherent impls).
#[test]
fn a_generic_inherent_method_called_at_two_types_computes_the_right_value() {
    let context = context();
    let src = "
        struct Boxed<T> { value: T }
        impl<T: Ring> struct Boxed<T> {
            fn doubled(b) -> T { add(b.value, b.value) }
        }
        fn main() -> i32 {
            let bi = Boxed(value: 21);
            let bf = Boxed(value: 1.5);
            if bf.doubled() == 3.0 { bi.doubled() } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 42);
}

/// A self-recursive `let`-bound lambda (`doc/backlog.md`'s own "Self-
/// recursive `let`-bound lambda" item) — a genuinely separate gap from the
/// type-inference-layer one already fixed: even once `infer.rs` correctly
/// types `fact`'s own self-call, `monomorphize.rs`'s call-resolution scope
/// never saw the self-binding (`StmtKind::Let` only inserted `name ->
/// value.id` *after* walking the lambda's own body) and the lambda-
/// worklist's own drain loop re-walked that body from a totally empty
/// scope regardless — both needed fixing before `cps.rs`'s `resolve_call`
/// could ever find a unit to call. `5! = 120`, not a panic.
#[test]
fn a_self_recursive_let_bound_lambda_actually_runs() {
    let context = context();
    let src = "
        fn main() -> i32 {
            let fact = fn(n: i32) -> i32 { if n <= 1 { 1 } else { n * fact(n - 1) } };
            fact(5)
        }
    ";
    assert_eq!(run_i32(&context, src), 120);
}

/// The same self-recursive lambda, but *unannotated* -- `fact` is generalized
/// (`is_syntactic_value`) with a genuinely open type variable in its own
/// scheme, so its own body's `node_types` (as recorded by ordinary, once-
/// only whole-program inference) are still generic. A real second bug, found
/// by direct testing right after the annotated case above started passing:
/// discovering the self-call while walking that still-generic copy of the
/// body (from `main`'s own initial, "seed scan" walk) reverse-unifies
/// against open type variables and "succeeds" trivially (unifying `Ty::Var`
/// against anything always does), producing a bogus, never-actually-
/// reachable specialization whose own inner calls (`n <= 1`, here) then fail
/// to resolve against any concrete algebra impl. Guarded by requiring a
/// `derive_instantiation` result to be fully concrete (no leftover `Ty::
/// Var`) before ever recording it — the *real* concrete instantiation is
/// still found separately, from `fact(5)`'s own outer, already-concrete
/// call site in `main`.
#[test]
fn an_unannotated_self_recursive_lambda_still_runs_despite_being_generalized() {
    let context = context();
    let src = "
        fn main() -> i32 {
            let fact = fn(n) { if n <= 1 { 1 } else { n * fact(n - 1) } };
            fact(5)
        }
    ";
    assert_eq!(run_i32(&context, src), 120);
}

/// Same gap, but the self-recursive lambda also *captures* an outer
/// variable (`step`) -- proves the second, independent half of the fix:
/// even once `resolve_call` finds the right unit, the self-call must also
/// re-supply that unit's own captures (its leading params), since the
/// ordinary "already-bound `CVal::Closure`" fast path in `cps.rs`'s `Call`
/// arm never fires for a self-call (that binding only ever exists in the
/// *enclosing* scope's own environment, never inside the lambda's own
/// separately-converted unit body). A wrong/missing splice here fails loud
/// (wrong arity/argument types), not silently.
#[test]
fn a_capturing_self_recursive_lambda_re_supplies_its_own_captures() {
    let context = context();
    let src = "
        fn main() -> i32 {
            let step = 2;
            let count_by = fn(n: i32) -> i32 { if n <= 0 { 0 } else { step + count_by(n - step) } };
            count_by(10)
        }
    ";
    assert_eq!(run_i32(&context, src), 10);
}

/// `doc/backlog.md`'s own "Calling a lambda literal directly" item —
/// `(fn(a, b) { a + b })(1, 2)` used to have no grammar production to parse
/// at all (`call_expr` only ever accepts a bare `path` callee, and
/// `postfix_op` had no "just call whatever's on the left" alternative).
/// Fixed as pure syntactic sugar in `lower.rs`: desugars to `{ let
/// <synthetic> = <base>; <synthetic>(<args>) }`, reusing the *existing*
/// let-bound-lambda pipeline wholesale (`infer.rs`'s `lambda_schemes`,
/// `monomorphize.rs`'s lambda worklist, `cps.rs`'s closure conversion) —
/// nothing downstream of `lower.rs` needed to change at all.
#[test]
fn a_lambda_literal_called_directly_actually_runs() {
    let context = context();
    let src = "
        fn main() -> i32 {
            (fn(a, b) { a + b })(1, 2)
        }
    ";
    assert_eq!(run_i32(&context, src), 3);
}

/// Same desugaring, but used mid-expression rather than alone in tail
/// position — proves the synthesized `Block` composes as an ordinary
/// sub-expression, not just as a statement/tail on its own.
#[test]
fn a_lambda_literal_called_directly_composes_inside_a_larger_expression() {
    let context = context();
    let src = "
        fn main() -> i32 {
            1 + (fn(x: i32) -> i32 { x * 2 })(5)
        }
    ";
    assert_eq!(run_i32(&context, src), 11);
}

/// `doc/backlog.md`'s own "Const-generic algebra parameters" item —
/// `algebra Sum<T, const N: i32> { fn total(x: [T; N]) -> T; }`'s own `N`
/// used to be silently dropped both at a call site (`infer_algebra_call`'s
/// own generics mapping) and while conformance-checking an impl's own
/// method body against the algebra's declared signature (`infer_impl_fn_
/// generic_with_env`'s identical, separate gap, found by direct testing
/// once this end-to-end case exercised it) — both fixed the same way,
/// giving the const generic an ordinary fresh var instead of `None`. No
/// existing stdlib algebra declares its own const generic, so this
/// combination was never exercised before; run for real here rather than
/// just at the type-inference layer, to confirm `monomorphize.rs`/`cps.rs`
/// also carry it through correctly.
#[test]
fn an_algebras_own_const_generic_actually_runs() {
    let context = context();
    let src = "
        algebra Sum<T, const N: i32> {
            fn total(x: [T; N]) -> T;
        }
        impl Sum<i32> {
            fn total(x) -> i32 { x[0] + x[1] + x[2] }
        }
        fn main() -> i32 {
            total([10, 20, 12])
        }
    ";
    assert_eq!(run_i32(&context, src), 42);
}

/// `doc/backlog.md`'s own "Explicit turbofish on a const generic, for a
/// plain top-level `fn`" item — the exact repro. Root cause, confirmed by
/// direct testing, is *not* a turbofish-arity bug at all: `N`, referenced
/// as an ordinary body *value* (not a type-position use like `[T; N]`),
/// shares its own single fresh type-var between two different questions
/// ("N's own declared type" and "N's own generic identity") -- checking it
/// against its own declared return type (`-> i32`) used to permanently
/// collapse that var to `Con("i32")`, destroying its own identity before
/// `rep`'s own scheme was ever built, leaving turbofish with 0 declared
/// generics to match against.
#[test]
fn explicit_turbofish_on_a_const_generic_actually_runs() {
    let context = context();
    let src = "
        fn rep<const N: i32>(x: i32) -> i32 { N }
        fn main() -> i32 { rep::<3>(5) }
    ";
    assert_eq!(run_i32(&context, src), 3);
}

/// Same root cause, but with *no* turbofish anywhere -- proves the fix is
/// general, not specific to the turbofish call site: an inherent method
/// reading its own struct's const generic as a body value, pinned purely by
/// the constructed array's own size, the same way an ordinary generic
/// struct field already works. Found, by direct testing, to already fail
/// this same way (`CPS: could not resolve method call`) before this fix,
/// confirming the turbofish repro above was only the easiest way to
/// *reach* this bug, not its actual cause.
#[test]
fn a_const_generic_read_as_a_body_value_with_no_turbofish_actually_runs() {
    let context = context();
    let src = "
        struct Box<T, const N: i32> { data: [T; N] }
        impl<T, const N: i32> struct Box<T, N> {
            fn size(b) -> i32 { N }
        }
        fn main() -> i32 {
            Box(data: [1, 2, 3]).size()
        }
    ";
    assert_eq!(run_i32(&context, src), 3);
}

/// `doc/backlog.md`'s own "Complex literals" item — `2i + 4` used to be
/// unrepresentable (`ImaginaryLit` always inferred to a placeholder). Fixed
/// via literal-shape widening, not a general numeric-conversion mechanism
/// (see this session's own extensive design discussion, now also recorded
/// in `doc/backlog.md`'s new "Explicit conversion between numeric types"
/// item): both `2i` and `4` are still-elastic literals here, not already-
/// concrete values, so this is purely a defaulting question — `4`'s own
/// `Int` shape constraint is satisfied once it resolves to `Complex<f64>`
/// (ℤ ⊂ ℂ, unlike `Int` vs `Float`, which stay mutually exclusive exactly
/// as before). `stdlib/complex/complex.cleave`'s own `Ring<Complex<T>>` is
/// `examples/complex.cleave`'s own already-proven arithmetic, moved
/// verbatim into the stdlib. `use complex;` explicitly — not prelude.
///
/// `z`'s own `let` carries an explicit `Complex<f64>` annotation — found by
/// direct testing to be *necessary*, not decorative, and a real, separate,
/// documented limitation (see `doc/backlog.md`'s own "Complex literals"
/// Done entry): `z.real`/`z.imag`'s own field-access type resolution
/// (`infer.rs`'s `ExprKind::FieldAccess`) runs immediately, permanently
/// recording `<not-yet-inferred>` if `z`'s own base type is still a bare
/// `Ty::Var` at that point — which it is here, since `2i + 4`'s own
/// `Complex` defaulting doesn't happen until `apply_defaults` runs, at the
/// very end of the whole function's inference, strictly after every
/// statement (including this one) has already been walked. An explicit
/// annotation pins `z`'s type immediately at its own `let`, sidestepping
/// the ordering gap entirely — unannotated field access on a value whose
/// own type comes purely from deferred literal-defaulting is a known,
/// separate, not-yet-fixed limitation, not something this item's own
/// literal-widening fix could address without a deeper change to when/how
/// defaulting runs relative to field-access resolution.
#[test]
fn a_complex_literal_added_to_a_plain_int_literal_actually_runs() {
    let context = context();
    let src = "
        use complex;
        fn main() -> i32 {
            let z: Complex<f64> = 2i + 4;
            if z.real == 4.0 and z.imag == 2.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// Same as above, operands reversed (`4 + 2i`) — proves the widening isn't
/// order-dependent (`add`'s own two arguments are inferred independently,
/// each carrying its own shape constraint, before dispatch ever runs).
#[test]
fn a_plain_int_literal_added_to_a_complex_literal_actually_runs() {
    let context = context();
    let src = "
        use complex;
        fn main() -> i32 {
            let z: Complex<f64> = 4 + 2i;
            if z.real == 4.0 and z.imag == 2.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// `Complex<T>`'s own `magnitude_sq`/`magnitude` inherent methods
/// (`stdlib/complex/complex.cleave`) — `3 + 4i` has magnitude `5` (the
/// classic 3-4-5 right triangle), checked both ways: `magnitude_sq`
/// (`real*real + imag*imag`, no `sqrt`, exact) and `magnitude` itself
/// (`mlir::math::sqrt` on top of it).
#[test]
fn complex_magnitude_and_magnitude_sq_compute_correctly() {
    let context = context();
    let src = "
        use complex;
        fn main() -> i32 {
            let z: Complex<f64> = 3.0 + 4i;
            if z.magnitude_sq() == 25.0 and z.magnitude() == 5.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// Real bug, found by direct testing while extending `examples/complex.
/// cleave`: `apply_defaults` processed `pending_defaults` in plain source
/// order — when a bare `Float` literal and an `Imaginary` literal end up
/// sharing the same merged variable (via `+`) with *no* enclosing
/// annotation to pin the result early (every other complex-literal test
/// above always has one), whichever literal happened to be written *first*
/// won the defaulting race. `5.0 + 7.5i` (float literal first) collapsed
/// the shared variable to a bare `f32`, discarding the imaginary part
/// entirely and failing with `no impl Complex<f32>`; `7.5i + 5.0`
/// (imaginary literal first) happened to work purely by accident of
/// iteration order. Fixed: `Complex` defaults are now always applied
/// before `Int`/`Float` ones, order-independently (ℤ, ℝ ⊂ ℂ — a bare
/// shape only ever widens *into* `Complex`, never the reverse).
///
/// Neither `z1` nor `z2` is consumed afterward (no field access, no method
/// call, no annotation) — any of those would themselves force early
/// unification and mask the exact bug this proves fixed; the real
/// assertion here is that this compiles and runs at all; before the fix,
/// `compile(...)` itself failed.
#[test]
fn a_float_literal_and_an_imaginary_literal_combine_regardless_of_operand_order() {
    let context = context();
    let src = "
        use complex;
        fn main() -> i32 {
            let z1 = 5.0 + 7.5i;
            let z2 = 7.5i + 5.0;
            1
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// Real bug, found while building `Convert<From, To>`: a bare numeric
/// literal inside a generic impl's own body (`Pair`'s `b` field, defaulting
/// to `f32` rather than staying `T`) silently collapsed the impl's own
/// generic `T` to a concrete `f32` at *declaration* time — corrupting the
/// whole template, so a *second*, differently-typed call site (`f64`) could
/// never reverse-unify against it (`monomorphize.rs`'s own `derive_impl_
/// instantiation`, `NoneMatched` → `MonomorphizationFailed`). A single-
/// instantiation test wouldn't catch this at all (an `f32`-only program
/// happened to work by accident, since that's exactly what the corrupted
/// template got stuck at) — this specifically invokes the same generic
/// impl at *two* different concrete types in one program, proving the
/// template itself stayed properly generic. `Float` deliberately isn't
/// redeclared here — it's already a real, `#[mlir_type(...)]`-tagged
/// prelude algebra (`stdlib/num/num.cleave`); a second, local, untagged
/// `algebra Float<T> {}` was tried first and silently corrupted `ty_to_
/// mlir`'s own `f64` lowering (a real, separate program-merge gap, found
/// by direct testing and worth its own future look, not chased further
/// here) — every existing test in this file already avoids this by relying
/// on the prelude's own `Float`/`Int` instead of redeclaring either.
#[test]
fn a_generic_impls_own_generic_stays_generic_across_multiple_concrete_instantiations() {
    let context = context();
    let src = "
        struct Pair<T> { a: T, b: T }
        algebra Widen<From, To> { fn widen(x: From) -> To; }
        impl<T: Float> Widen<T, Pair<T>> {
            fn widen(x) { Pair(a: x, b: 0.0) }
        }
        fn main() -> i32 {
            let x: f32 = 3.0;
            let y: f64 = 4.0;
            let p1: Pair<f32> = widen(x);
            let p2: Pair<f64> = widen(y);
            if p1.a == 3.0 and p1.b == 0.0 and p2.a == 4.0 and p2.b == 0.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// `doc/backlog.md`'s own "`Convert`/output-only-generic premature-commit
/// bug" item — the mirror image of `a_generic_impls_own_generic_stays_
/// generic_across_multiple_concrete_instantiations` just above, but for a
/// call *inside* the generic impl's own body dispatching an algebra whose
/// gating generic (`Convert<From,To>`'s own `From`) is concrete
/// *independently* of the enclosing impl's still-abstract `T`, rather than
/// tied to it the way `Ring<Complex<T>>::add`'s own internal dispatch
/// already is. `Boxed<T,N>::widen_n`'s own body calls `n_i.to()` — with
/// *two* real `Convert<i32,_>` impls declared (`stdlib/convert/
/// convert.cleave`, `f32` and `f64`), this used to either commit `T`
/// permanently to whichever candidate happened to be tried first (surfacing
/// as a spurious type mismatch the moment the *other* instantiation's own
/// concrete argument type was reconciled against it) or a flat-out
/// `AmbiguousDispatch` — even though neither instantiation's own real call
/// site is ever actually ambiguous. `N` is routed through `let n_i: i32 =
/// mlir::arith::addi(N, 0);` before `.to()`, not passed bare — a bare const
/// generic used directly as a call argument hits a separate, already-
/// documented `Ty::Const`-vs-`Ty::Con` gap (`doc/backlog.md`), unrelated to
/// the bug this test is actually about.
#[test]
fn an_algebra_call_inside_a_generic_impl_body_resolves_independently_per_instantiation() {
    let context = context();
    let src = "
        use convert;
        struct Boxed<T, const N: i32> { x: T }
        algebra WidenN<Container, T> { fn widen_n(c: Container) -> T; }
        impl<T: Float, const N: i32> WidenN<Boxed<T, N>, T> {
            fn widen_n(b) {
                let n_i: i32 = mlir::arith::addi(N, 0);
                b.x + n_i.to()
            }
        }
        fn main() -> i32 {
            let b1: Boxed<f32, 3> = Boxed(x: 1.0);
            let b2: Boxed<f64, 3> = Boxed(x: 1.0);
            let r1: f32 = widen_n(b1);
            let r2: f64 = widen_n(b2);
            if r1 == 4.0 and r2 == 4.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

// ---------------------------------------------------------------------
// `doc/backlog.md`'s "Explicit conversion between numeric types" item —
// `stdlib/convert/convert.cleave`'s `algebra Convert<From, To>` and `.to()`
// (`lower.rs`'s own postfix desugaring, `x.to()` -> `convert(x)`).
// ---------------------------------------------------------------------

/// `.to()` on an already-concrete value, with exactly one candidate impl
/// for that `From` — the ergonomic, common case: no turbofish needed, real
/// `arith::sitofp` runs under the hood.
#[test]
fn to_sugar_converts_an_int_to_a_float_end_to_end() {
    let context = context();
    let src = "
        use convert;
        fn main() -> i32 {
            let n: i32 = 7;
            let f: f64 = n.to();
            if f == 7.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// The same sugar on a bare literal, not an already-annotated value —
/// `4.to()` is the far more natural spelling in practice, and exercises
/// `infer_algebra_call`'s *deferred* path (the literal's own shape var
/// isn't concrete until `apply_defaults` runs, well after the call itself
/// was type-checked — see `check_pending_constraints`).
#[test]
fn to_sugar_on_a_bare_literal_converts_correctly() {
    let context = context();
    let src = "
        use convert;
        fn main() -> i32 {
            let f: f64 = 4.to();
            if f == 4.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// A second, competing `Convert<i32, _>` impl (declared right here, proving
/// `Convert` is an ordinary, user-extensible algebra, not a closed set)
/// makes a bare, turbofish-free `n.to()` genuinely ambiguous — `.to()`
/// itself has no turbofish grammar (postfix method syntax doesn't carry
/// one), so disambiguating means dropping to the explicit `convert::<From,
/// To>(x)` call form instead. Proves that end to end: it dispatches
/// cleanly and runs the *right* impl body (`i32 -> f64`'s real `sitofp`),
/// not the competing `i32 -> Widened` one — a real runtime distinction, not
/// just two differently-typed declarations.
#[test]
fn turbofish_on_convert_disambiguates_between_two_real_competing_impls() {
    let context = context();
    let src = "
        use convert;
        struct Widened { value: i32 }
        impl Convert<i32, Widened> { fn convert(x) { Widened(value: x) } }
        fn main() -> i32 {
            let f: f64 = convert::<i32, f64>(9);
            if f == 9.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

// ---------------------------------------------------------------------
// `doc/backlog.md`'s "Deferred/symbolic constant folding" item — an array
// size *computed* from two of a generic fn's own const generics (`[i32;
// N+M]`), staying symbolic (`Ty::ConstExpr`) through declaration-time
// inference and only folding into a real concrete size once monomorphized
// at a real, turbofish-pinned call site.
// ---------------------------------------------------------------------

/// The real payoff, end to end: `f`'s own generic signature (`c: [i32;
/// N+M]`) is meaningless until monomorphized. `N`/`M` are each individually
/// recovered the ordinary way, first — `monomorphize.rs`'s own `derive_
/// instantiation` reverse-unifies a generic call's own concrete argument
/// types against the scheme's declared shape, and `a`'s/`b`'s own real
/// array lengths (2, 3) pin `N`/`M` directly, exactly like any other
/// generic parameter. *Then* `c`'s own declared `N+M` gets checked: by the
/// time `unify` reaches `c`'s position (`Ty::Fn`'s own arm unifies
/// parameters strictly left to right, mutating one shared `Subst`), `N`
/// and `M` are already bound from `a`/`b`, so `Subst::apply`'s own fold
/// (called at the top of every `unify`) collapses `N+M` into a real
/// `Const(5)` *before* the match ever runs — matching `c`'s own actual
/// 5-element argument cleanly, no special-casing needed. The JIT-computed
/// sum proves the values themselves, not just the types, came through
/// correctly.
///
/// Deliberately *not* the shape where `N`/`M` appear *only* combined inside
/// `N+M` (`fn f<const N,M>(x: [i32; N+M]) -> ...`, called via explicit
/// turbofish alone) — confirmed by direct testing that `derive_
/// instantiation` never consults a call's own explicit turbofish at all
/// (`collect_instantiations_expr`'s own `ExprKind::Call` arm discards it,
/// `(path, _, args, ..)`), only ever reverse-unifying from argument/return
/// *value* types — so a const generic that never appears on its own,
/// anywhere in the signature, can't be recovered for monomorphization
/// purposes this way. A real, separate, narrower gap, flagged in
/// `doc/backlog.md` rather than fixed here — this test's own shape (each
/// const generic *also* appearing directly somewhere) is the realistic,
/// working case this item's own motivating example (`[T; N+M]`) is
/// actually about.
#[test]
fn a_generic_fns_computed_array_size_folds_correctly_at_monomorphization() {
    let context = context();
    let src = "
        fn f<const N: i32, const M: i32>(a: [i32; N], b: [i32; M], c: [i32; N+M]) -> i32 {
            c[0] + c[1]
        }
        fn main() -> i32 {
            f([1, 2], [3, 4, 5], [10, 20, 30, 40, 50])
        }
    ";
    assert_eq!(run_i32(&context, src), 30);
}

// ---------------------------------------------------------------------
// Deferred field-access/method-call resolution — a value whose only
// concreteness comes from `apply_defaults` used to permanently lock a
// field access or method call on it to `<not-yet-inferred>` (see
// `infer.rs`'s own `pending_field_accesses`/`pending_method_calls`).
// ---------------------------------------------------------------------

/// The real, user-reported repro, end to end: `examples/complex.cleave`'s
/// own exact shape — `let z7 = 5.0 + 7.5i;` with *no* annotation, then
/// `.magnitude()` directly. `z7`'s own type only becomes concrete via
/// `apply_defaults`, well after `.magnitude()`'s own call-site resolution
/// used to already have given up.
#[test]
fn an_unannotated_complex_literals_method_call_actually_runs() {
    let context = context();
    let src = "
        use complex;
        fn main() -> i32 {
            let z7 = 3.0 + 4.0i;
            if z7.magnitude() == 5.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// Same, for plain field access instead of a method call.
#[test]
fn an_unannotated_complex_literals_field_access_actually_runs() {
    let context = context();
    let src = "
        use complex;
        fn main() -> i32 {
            let z7 = 3.0 + 4.0i;
            if z7.real == 3.0 and z7.imag == 4.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

// ---------------------------------------------------------------------
// `Tensor<T, const Dims...: i32>` (`stdlib/linalg/tensor.cleave`): one
// pack-generic struct, `#[mlir_type(tensor)]`-tagged — its real
// representation is a native MLIR `tensor<...>` value, never the ordinary
// heap-allocated struct reference. See `doc/backlog.md`'s own entry for
// the design (supersedes an earlier, reverted `Ty::Vector` hardcoded `Ty`
// variant, and — later — three separately-declared concretely-ranked
// structs, before `Ty::Pack`/`Ty::PackLen` existed). Every construction
// below needs an explicit turbofish: a pack's own arity is never inferred
// from a field literal's shape alone.
// ---------------------------------------------------------------------

/// The smallest possible vertical slice: an ordinary named-field struct
/// literal (`Tensor::<f32,3>(data: [...])`) really builds a real MLIR
/// `tensor<3xf32>` value (`tagged_struct_native_type`/`lower_tagged_struct_
/// construct`), and `v[i]` — dispatched through the new `Index<Container,
/// Elem>` algebra fallback, not a field — reads a real element back out of
/// it.
#[test]
fn a_tagged_struct_constructs_as_a_real_tensor_and_index_reads_the_right_element() {
    let context = context();
    let src = r#"
        use linalg;
        fn main() -> i32 {
            let v = Tensor::<f32, 3>(data: [1.0, 2.0, 3.0]);
            let x0 = v[0];
            let x1 = v[1];
            let x2 = v[2];
            if x0 == 1.0 and x1 == 2.0 and x2 == 3.0 { 1 } else { 0 }
        }
    "#;
    assert_eq!(run_i32(&context, src), 1);
}

/// `Ring<Tensor<T,Dims...>>` (`stdlib/linalg/tensor.cleave`) — ordinary
/// elementwise arithmetic (`+` desugars to `Ring::add` the same way it does
/// for any other `Ring`-impl'd type), `arith.addf` broadcasting structurally
/// over the `tensor<3xf32>`-typed operands — no new mechanism, and no
/// awareness anywhere in `stdlib/linalg/tensor.cleave` that this needs
/// anything beyond the one-line-per-op shape every other `Ring` impl uses.
#[test]
fn elementwise_ring_add_on_tagged_vectors_computes_the_right_values() {
    let context = context();
    let src = r#"
        use linalg;
        fn main() -> i32 {
            let a = Tensor::<f32, 3>(data: [1.0, 2.0, 3.0]);
            let b = Tensor::<f32, 3>(data: [10.0, 20.0, 30.0]);
            let c = a + b;
            let x0 = c[0];
            let x1 = c[1];
            let x2 = c[2];
            if x0 == 11.0 and x1 == 22.0 and x2 == 33.0 { 1 } else { 0 }
        }
    "#;
    assert_eq!(run_i32(&context, src), 1);
}

/// A real, structural `linalg.matmul`-backed matmul on a nested rank-2
/// `Tensor<T,R,C>` — an ordinary stdlib type (`stdlib/linalg/
/// tensor.cleave`) — the flagship goal of this whole item (`doc/hld.md`'s
/// own "don't lower prematurely" thesis, worked example: `matmul` as one
/// opaque, reassociation-eligible node, not a hand-written triple-nested
/// loop). 2x3 times 3x2 -> 2x2, the exact same hand-computed expected
/// values `examples/matmul.cleave` already uses. Read back via the real
/// `mc[i,j]` sugar directly -- multi-dimensional `Index` dispatch (`doc/
/// backlog.md`'s own former "not attempted" item), not a raw `mlir::tensor
/// ::extract` escape hatch: `ast.rs::ExprKind::Index` now carries a whole
/// bracket group's indices on one node, and `stdlib/linalg/tensor.cleave`'s
/// own `Index<Tensor<T,Dims...>, T>` impl (`idx: [i32; Dims.len()]`)
/// dispatches the whole group as one call. No `let zero: i32 = ...`
/// workaround needed at the call site any more either -- the `arith.
/// index_cast`-on-a-bare-literal rough edge (`doc/backlog.md`) lives
/// entirely *inside* the impl body now (via `mlir::tensor::extract`'s own
/// variadic-index-array form, `lower_tensor_extract_spread`), invisible to
/// a caller.
///
/// `mc` needs no explicit type annotation, unlike an earlier version of
/// this test: `MatMul<A,B,C>`'s own `C` is exactly the same kind of
/// output-only generic `Index<Container,Elem,K>`'s own `Elem` is
/// (`doc/backlog.md`'s own "`check_pending_constraints`'s output-only-
/// generic gate" item, plus `ExprKind::Index`'s own missing deferred-
/// resolution path) — both real, fixed bugs now, not workarounds.
#[test]
fn a_structural_linalg_matmul_on_tagged_matrices_computes_the_right_values() {
    let context = context();
    let src = r##"
        use linalg;
        fn main() -> i32 {
            let ma = Tensor::<f32, 2, 3>(data: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
            let mb = Tensor::<f32, 3, 2>(data: [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);
            let mc = matmul(ma, mb);
            if mc[0,0] == 58.0 and mc[0,1] == 64.0 and mc[1,0] == 139.0 and mc[1,1] == 154.0 { 1 } else { 0 }
        }
    "##;
    assert_eq!(run_i32(&context, src), 1);
}

/// `Ring<Tensor<T,Dims...>>` at rank 2 (`stdlib/linalg/tensor.cleave`) —
/// mirrors `elementwise_ring_add_on_tagged_vectors_computes_the_right_
/// values`'s own shape one rank up: `arith.addf` broadcasting structurally
/// over a `tensor<2x2xf32>`-typed operand, confirmed directly rather than
/// assumed to just work at rank 2 — the *same* impl as the rank-1 test
/// above, not a separate one.
#[test]
fn elementwise_ring_add_on_tagged_matrices_computes_the_right_values() {
    let context = context();
    let src = r#"
        use linalg;
        fn main() -> i32 {
            let a = Tensor::<f32, 2, 2>(data: [[1.0, 2.0], [3.0, 4.0]]);
            let b = Tensor::<f32, 2, 2>(data: [[10.0, 20.0], [30.0, 40.0]]);
            let c = a + b;
            if c[0,0] == 11.0 and c[1,1] == 44.0 { 1 } else { 0 }
        }
    "#;
    assert_eq!(run_i32(&context, src), 1);
}

/// `Tensor<T, const Dims...: i32>` (`stdlib/linalg/tensor.cleave`) at rank
/// 3 — confirms the same generalized `#[mlir_type(tensor)]` mechanism
/// (`tagged_struct_native_type`/`lower_tagged_struct_construct`, both fully
/// shape-generic already, driven by `flatten_array_dims` over however many
/// dims the field's own array type actually has) really does scale past 2
/// dims with no new Rust code needed — construction plus a real `t[i,j,k]`
/// read back, through the *same* pack-generic `Index<Tensor<T,Dims...>, T>`
/// impl the rank-1/rank-2 tests above also exercise, not a separate one.
///
/// Found a real, separate bug while fixing this test up for the pack
/// migration (not a pack bug itself): `run_i32` never calls `check_type_
/// errors` the way `main.rs` does before codegen, so a genuinely ill-typed
/// program (the old, pre-migration `Tensor(data: ...)` with no turbofish,
/// now correctly rejected — `TypeErrorKind::VariadicStructNeedsTurbofish`)
/// sails straight through into a malformed module instead of failing this
/// one test cleanly, crashing the *entire* test binary (`Symbols not found:
/// [ _mlir__mlir_ciface_main ]`, `STATUS_STACK_BUFFER_OVERRUN`) — confirmed
/// directly, not guessed: the exact same source cleanly reports `` `Tensor`
/// has a variadic generic ... `` through the real CLI (`main.rs`'s own
/// `--run`, which *does* call `check_type_errors`). Noted in `doc/
/// backlog.md` rather than fixed here — a `run_i32` robustness gap, a
/// different concern from this item.
#[test]
fn a_rank_3_tensor_constructs_and_reads_back_the_right_value() {
    let context = context();
    let src = r#"
        use linalg;
        fn main() -> i32 {
            let t = Tensor::<f32, 2, 2, 2>(data: [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]]
            ]);
            if t[0,1,0] == 3.0 and t[1,1,1] == 8.0 { 1 } else { 0 }
        }
    "#;
    assert_eq!(run_i32(&context, src), 1);
}

/// `doc/backlog.md`'s own "Real pack-generic `[value; Dims...]` array-
/// repeat" item — `[value; Dims...]` (a whole *pack* reference as an
/// array-repeat's own count, not one named const generic) now builds a
/// real, same-shaped, repeated-value construction from ordinary cleave
/// source — the missing piece `derivative-independent-zero`'s own `build_
/// zero` (`egraph.rs`) had to hand-recognize structurally instead, before
/// this. Checked at three different ranks through the same pack-generic
/// impl, not just one shape happening to work: rank 1 (`Tensor<f32,3>`),
/// rank 2 (`Tensor<f32,2,2>`), rank 3 (`Tensor<f32,2,2,2>`), reading back
/// elements at both extremes (`[0,...]` and the last valid index) of each.
#[test]
fn array_repeat_over_a_whole_pack_builds_a_same_shaped_tensor_at_every_rank() {
    let context = context();
    let src = r#"
        use linalg;
        algebra Filled<T> {
            fn filled() -> T;
        }
        impl<T: Float, const Dims...: i32> Filled<Tensor<T, Dims...>> {
            fn filled() -> Tensor<T, Dims...> {
                Tensor::<T, Dims...>(data: [7.0; Dims...])
            }
        }
        fn main() -> i32 {
            let v: Tensor<f32, 3> = Filled::filled();
            let m: Tensor<f32, 2, 2> = Filled::filled();
            let t: Tensor<f32, 2, 2, 2> = Filled::filled();
            if v[0] == 7.0 and v[2] == 7.0
                and m[0,0] == 7.0 and m[1,1] == 7.0
                and t[0,0,0] == 7.0 and t[1,1,1] == 7.0
            { 1 } else { 0 }
        }
    "#;
    assert_eq!(run_i32(&context, src), 1);
}

/// The negative counterpart — `m[i,j,k]` (3 indices) on a rank-2 `Tensor`
/// must be rejected, not silently accepted with the extra index ignored:
/// `Index<Tensor<T,Dims...>,T>`'s own `idx: [i32; Dims.len()]` pins `K` to
/// `m`'s own real rank (2) the moment it's monomorphized, so a 3-element
/// bracket group builds a `[i32;3]` array whose length doesn't structurally
/// match.
///
/// `K` is deliberately excluded from `infer_algebra_call`'s own `resolved_
/// generics` (only an algebra's *type* generics — `Container`, `Elem` — ever
/// feed `dispatch_algebra_call`'s target-matching; a *const* generic like
/// `K` is bound separately, by ordinary `unify_at` against the impl's own
/// declared `idx` parameter type) — so ordinary immediate dispatch accepts
/// `m[0,0,0]` freely (nothing about `K` blocks it there), and the real
/// rejection only surfaces one pass later, at monomorphization
/// (`derive_impl_instantiation`'s own full signature match, which *does*
/// see `m`'s own real rank and fails to unify it against a `[i32;3]`
/// query) — `cleave::monomorphize::dump_monomorphized` runs that pass;
/// `cleave::dump::dump_program` (ordinary inference alone, as used by e.g.
/// `named_arguments_on_an_ordinary_call_are_rejected` above) would wrongly
/// report this program as accepted. `m`'s own explicit `Tensor<f32,2,2>`
/// annotation sidesteps the separate, already-documented `doc/backlog.md`
/// inference gap (`check_pending_constraints`'s output-only-generic gate)
/// entirely, so this test isolates exactly the one thing it means to check.
#[test]
fn over_indexing_a_matrix_is_rejected() {
    let (result, _sources) = compile(
        vec![(
            "test.cleave".to_string(),
            "use linalg;\nfn main() -> i32 { let m: Tensor<f32,2,2> = Tensor::<f32,2,2>(data: [[1.0, 2.0], [3.0, 4.0]]); if m[0,0,0] == 1.0 { 1 } else { 0 } }"
                .to_string(),
        )],
        &[],
    );
    let program = result.unwrap_or_else(|e| {
        panic!("expected a parse/use-resolution success (the type error is caught later): {e:?}")
    });
    let registry = Registry::build(&program);
    let (_, errs) = cleave::monomorphize::dump_monomorphized(&program, &registry);
    assert!(
        !errs.is_empty(),
        "`m[0,0,0]` supplies 3 indices to a rank-2 `Tensor` and must be rejected"
    );
}

/// The array-side counterpart -- over-indexing a real, plain 2D array
/// (`a[i,j,k]`, one more index than its own declared rank) must still be
/// cleanly rejected, exactly as before this session's `Vec<Expr>` rework of
/// `ExprKind::Index` (recursion-per-level -> one explicit peel-loop, see
/// `infer.rs`'s own doc comment) -- confirms that rewrite didn't loosen the
/// existing array-arity check.
#[test]
fn over_indexing_a_plain_array_is_rejected() {
    let (result, _sources) = compile(
        vec![(
            "test.cleave".to_string(),
            "fn main() -> i32 { let a = [[1, 2], [3, 4]]; a[0,0,0] }".to_string(),
        )],
        &[],
    );
    let program = result.unwrap_or_else(|e| {
        panic!("expected a parse/use-resolution success (the type error is caught later): {e:?}")
    });
    let registry = Registry::build(&program);
    let (_, errs) = cleave::dump::dump_program(&program, &registry);
    assert!(
        !errs.is_empty(),
        "`a[0,0,0]` supplies one more index than this 2D array's own rank and must be rejected"
    );
}

// ------------------------------------------------------------ check_pending_constraints's output-only-generic gate

/// The real end-to-end proof for `doc/backlog.md`'s own "`check_pending_
/// constraints`'s output-only-generic gate" item: `print(v[0])`, with *no*
/// intervening `let x: f32 = v[0];` annotation, used to panic in `cps.rs`'s
/// own `resolve_call` ("could not resolve call to `print`") rather than
/// compile and run. `v`'s own element type is still an undefaulted `Ty::Var`
/// at the point `v[0]` is first seen (defaulting only runs once, at the very
/// end of `main`'s own inference), so `Index`'s own dispatch defers -- and
/// its output-only `Elem` generic was never independently pinned by
/// anything else the way a `==` comparison against a literal would (a bare
/// `print(...)` call forces nothing on its own). `print_f32` is registered
/// for real, not just verified structurally -- a wrong dispatch would show
/// up as a genuine JIT symbol-resolution failure, not just a silently wrong
/// value.
#[test]
fn print_of_an_unannotated_index_result_no_longer_panics() {
    let context = context();
    let (result, _sources) = compile(
        vec![(
            "test.cleave".to_string(),
            "use io;\nuse linalg;\nfn main() -> i32 { let v = Tensor::<f32,3>(data: [1.0, 2.0, 3.0]); print(v[0]); 0 }".to_string(),
        )],
        &[],
    );
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    let units = collect_units(&program, &registry);
    let cps_program = convert_program(units);
    let mlir_types = collect_mlir_types(&program);
    let struct_schemas = collect_struct_schemas(&program);
    let mut module = lower_program(&context, &cps_program, &mlir_types, struct_schemas);
    assert!(
        module.as_operation().verify(),
        "generated MLIR module failed verification"
    );

    let pass_manager = pass::PassManager::new(&context);
    pass_manager.add_pass(pass::linalg::create_convert_elementwise_to_linalg_pass());
    pass_manager
        .run(&mut module)
        .expect("convert-elementwise-to-linalg must succeed");

    let pass_manager = pass::PassManager::new(&context);
    pass::bufferization::register_one_shot_bufferize_pass();
    parse_pass_pipeline(
        pass_manager.as_operation_pass_manager(),
        "builtin.module(one-shot-bufferize{bufferize-function-boundaries=true})",
    )
    .expect("failed to parse the one-shot-bufferize pass pipeline");
    pass_manager
        .run(&mut module)
        .expect("one-shot-bufferize must succeed");

    let pass_manager = pass::PassManager::new(&context);
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
    }
    register_io_symbols(&engine);
    let mut out: i32 = -1;
    unsafe {
        engine
            .invoke_packed("main", &mut [&mut out as *mut i32 as *mut ()])
            .expect("JIT invocation must succeed");
    }
    assert_eq!(out, 0);
}

/// The `MatMul<A,B,C>` counterpart -- `C` is exactly the same kind of
/// output-only generic `Index`'s own `Elem` is (confirmed directly this
/// session: the original version of the matmul JIT test above only avoided
/// this by reading `mc` back through a raw `mlir::tensor::extract` call,
/// which never needed `mc`'s own cleave-level type resolved at all). `mc`
/// has *no* explicit `Tensor<f32,2,2>` annotation here, unlike the matmul
/// test above -- `print(mc[0,0])` alone must now be enough.
#[test]
fn print_of_an_unannotated_matmul_index_result_no_longer_panics() {
    let context = context();
    let (result, _sources) = compile(
        vec![(
            "test.cleave".to_string(),
            "use io;\nuse linalg;\nfn main() -> i32 {\n\
             let ma = Tensor::<f32,2,3>(data: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);\n\
             let mb = Tensor::<f32,3,2>(data: [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);\n\
             let mc = matmul(ma, mb);\n\
             print(mc[0,0]);\n\
             0\n\
             }"
            .to_string(),
        )],
        &[],
    );
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    let units = collect_units(&program, &registry);
    let cps_program = convert_program(units);
    let mlir_types = collect_mlir_types(&program);
    let struct_schemas = collect_struct_schemas(&program);
    let mut module = lower_program(&context, &cps_program, &mlir_types, struct_schemas);
    assert!(
        module.as_operation().verify(),
        "generated MLIR module failed verification"
    );

    let pass_manager = pass::PassManager::new(&context);
    pass_manager.add_pass(pass::linalg::create_convert_elementwise_to_linalg_pass());
    pass_manager
        .run(&mut module)
        .expect("convert-elementwise-to-linalg must succeed");

    let pass_manager = pass::PassManager::new(&context);
    pass::bufferization::register_one_shot_bufferize_pass();
    parse_pass_pipeline(
        pass_manager.as_operation_pass_manager(),
        "builtin.module(one-shot-bufferize{bufferize-function-boundaries=true})",
    )
    .expect("failed to parse the one-shot-bufferize pass pipeline");
    pass_manager
        .run(&mut module)
        .expect("one-shot-bufferize must succeed");

    let pass_manager = pass::PassManager::new(&context);
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
    }
    register_io_symbols(&engine);
    let mut out: i32 = -1;
    unsafe {
        engine
            .invoke_packed("main", &mut [&mut out as *mut i32 as *mut ()])
            .expect("JIT invocation must succeed");
    }
    assert_eq!(out, 0);
}

// ------------------------------------------------------------ stdlib/nn -- activations and reductions

/// `Activation<T>` (`stdlib/nn/nn.cleave`) on plain scalars -- the base case
/// before extending to `Tensor<T,N>` below. `relu`/`sigmoid`/`tanh`, each an
/// ordinary one-or-two-line `mlir::...` body, no different in kind from any
/// existing `Ring`/`Ord` impl. `sigmoid(0.0) == 0.5` is exact in IEEE 754
/// (`exp(0)==1.0`, `1+1==2`, `1/2==0.5`), safe to check with `==`.
#[test]
fn relu_sigmoid_tanh_compute_correctly_on_scalars() {
    let context = context();
    // No `use linalg;` here, deliberately — real proof that `driver.rs`'s
    // own transitive `use` resolution actually works: `nn.cleave`'s own
    // internal `use linalg;` is followed on its own now, even though this
    // test itself never touches a `Tensor` value (nn.cleave's own `Tensor`-
    // based `Activation` impl is merged into any program using `nn` at all,
    // whole-crate, no per-symbol filtering — it needs `vector` loaded
    // regardless of whether *this* file ever names it).
    // `Activation::tanh` — qualified, not bare `tanh(0.0)` — ambiguous
    // between `Activation<f32>` and `Transcendental<f32>` (`stdlib/num/
    // num.cleave`) now that both declare a 1-arg `tanh`.
    let src = r#"
        use nn;
        fn main() -> i32 {
            if relu(-2.0) == 0.0 and relu(3.0) == 3.0 and sigmoid(0.0) == 0.5 and Activation::tanh(0.0) == 0.0 { 1 } else { 0 }
        }
    "#;
    assert_eq!(run_i32(&context, src), 1);
}

/// The `Tensor<T,N>` counterpart -- confirms `math.exp`/`math.tanh`/`arith.
/// maximumf` really do apply directly to a `tensor<Nxf32>`-typed operand the
/// same way `Ring<Tensor<T,Dims...>>::add`'s own `arith.addf` already does
/// (no per-element loop), the one genuinely unverified-until-tested claim
/// this item's own plan made. `relu`/`sigmoid` each need a same-shape
/// constant vector to compare/combine against (`arith.maximumf`/`arith.addf`
/// don't broadcast a bare scalar into a tensor operand) -- built via the
/// already-working `[value; N]` array-repeat + `Tensor::<T,N>(data: ...)`
/// construction, exactly like `stdlib/nn/nn.cleave`'s own impl bodies do.
#[test]
fn relu_sigmoid_tanh_compute_correctly_on_vectors() {
    let context = context();
    let src = r#"
        use nn;
        fn main() -> i32 {
            let v = Tensor::<f32, 3>(data: [-2.0, 3.0, 0.0]);
            let r = relu(v);
            let s = sigmoid(Tensor::<f32, 3>(data: [0.0, 0.0, 0.0]));
            let t = Activation::tanh(Tensor::<f32, 3>(data: [0.0, 0.0, 0.0]));
            if r[0] == 0.0 and r[1] == 3.0 and r[2] == 0.0
                and s[0] == 0.5 and s[1] == 0.5 and s[2] == 0.5
                and t[0] == 0.0 and t[1] == 0.0 and t[2] == 0.0
            { 1 } else { 0 }
        }
    "#;
    assert_eq!(run_i32(&context, src), 1);
}

/// `Activation<Tensor<T,Dims...>>` is pack-generic now (was `Tensor<T,N>`,
/// rank-1 only) -- this is the actual, previously-unverified claim: a
/// genuinely rank-2 tensor (`Tensor<f32,1,3>`, the shape `Dense::forward`'s
/// own output takes) works through the identical impl body with no special
/// casing, confirming the `N` -> `Dims...` generalization is as mechanical
/// as it looked (reusing the same pack-generic array-repeat `Ring<Tensor<T,
/// Dims...>>::zero()` already proved for `[zero(); Dims...]`).
#[test]
fn relu_sigmoid_tanh_compute_correctly_on_a_rank_2_tensor() {
    let context = context();
    let src = r#"
        use nn;
        fn main() -> i32 {
            let v = Tensor::<f32, 1, 3>(data: [[-2.0, 3.0, 0.0]]);
            let r = relu(v);
            let s = sigmoid(Tensor::<f32, 1, 3>(data: [[0.0, 0.0, 0.0]]));
            let t = Activation::tanh(Tensor::<f32, 1, 3>(data: [[0.0, 0.0, 0.0]]));
            if r[0, 0] == 0.0 and r[0, 1] == 3.0 and r[0, 2] == 0.0
                and s[0, 0] == 0.5 and s[0, 1] == 0.5 and s[0, 2] == 0.5
                and t[0, 0] == 0.0 and t[0, 1] == 0.0 and t[0, 2] == 0.0
            { 1 } else { 0 }
        }
    "#;
    assert_eq!(run_i32(&context, src), 1);
}

/// `Sum`/`Mean`/`Max` (`stdlib/nn/nn.cleave`) on `Tensor<T,N>` -- plain
/// cleave source (a `for` loop plus this session's own real multi-index
/// `Index` reads), no MLIR-level reduction op needed at all. `[1.0, 5.0,
/// 3.0]`: sum 9.0, mean 3.0, max 5.0 -- chosen so no two are equal by
/// coincidence.
#[test]
fn sum_mean_max_of_a_vector_compute_correctly() {
    let context = context();
    let src = r#"
        use nn;
        fn main() -> i32 {
            let v = Tensor::<f32, 3>(data: [1.0, 5.0, 3.0]);
            if sum(v) == 9.0 and mean(v) == 3.0 and max_of(v) == 5.0 { 1 } else { 0 }
        }
    "#;
    assert_eq!(run_i32(&context, src), 1);
}

/// `Dense<T,In,Out>` (`stdlib/nn/nn.cleave`) -- the fully-connected-layer
/// helper the session's own `Network`/`xor_tensor.cleave` rewrite bundles
/// multiple of. `layer.forward(x)` needs `matmul` and `+` (`Ring::add`) to
/// resolve correctly from *inside* a generic inherent-impl method body -- a
/// real, found-by-testing monomorphization bug, not anticipated by the plan
/// that added `Dense`: `monomorphize`'s own four worklists (`fn`/`impl`/
/// `lambda`/`inherent`) used to drain in one fixed sequential pass, each
/// fully emptied before the next started; `Dense::forward` (an `inherent_
/// worklist` entry, drained *last*) calling `matmul` (a still-generic
/// `impl_worklist` entry, drained *third*) pushed new work onto a worklist
/// whose own draining loop had already finished, silently dropping it --
/// `MatMul::matmul<...>` got a real, correctly-mangled `call_names` entry
/// but no actual specialization, surfacing only as `cps.rs`'s own "call_
/// names resolved ... but no such unit exists" panic. No prior generic
/// inherent method's own body ever called a still-generic *algebra* method
/// (`Complex`/`DynArray`'s own inherent methods only ever call concrete/
/// extern ops), so this never came up before `Dense`. Fixed generally in
/// `monomorphize.rs`: the four worklists now redrain to a real fixed point
/// (loop until all four are empty at once), not a single one-shot pass.
///
/// `Dense::forward` deliberately applies *no* activation of its own (`nn.
/// cleave`'s own doc comment on it) -- an earlier version fused `sigmoid`
/// into the end of it, inherited unquestioned from `examples/xor_tensor.
/// cleave`'s own pre-`Dense` `sigmoid2` helper; raised directly and
/// reverted, since a real reusable `Dense` needs to stay just the affine
/// transform, the caller composing whichever activation it wants on top.
/// `x = [0, 1]`, `w = [[1,0],[0,-1]]`, `b = [0,0]` -- `matmul(x,w)+b =
/// [0,-1]` exactly, checked directly (both representable, no tolerance
/// needed).
#[test]
fn dense_layer_forward_computes_the_right_values() {
    let context = context();
    let src = "
        use nn;
        use linalg;
        fn main() -> i32 {
            let layer: Dense<f32, 2, 2> = Dense(
                w: Tensor::<f32, 2, 2>(data: [[1.0, 0.0], [0.0, -1.0]]),
                b: Tensor::<f32, 1, 2>(data: [[0.0, 0.0]])
            );
            let x: Tensor<f32, 1, 2> = Tensor::<f32, 1, 2>(data: [[0.0, 1.0]]);
            let h = layer.forward(x);
            if h[0, 0] == 0.0 and h[0, 1] == -1.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// `derive()` through `Dense::forward` -- the actual point of `Dense`, and a
/// genuinely deeper composition than `dense_layer_forward_computes_the_
/// right_values` just above needed to check (that one never differentiates
/// at all). Three more real, separate bugs were found and fixed getting
/// *here*, none anticipated by the plan that scoped `Dense`:
///
/// 1. `Activation<Tensor<T,Dims...>>::sigmoid`'s own body used raw `mlir::
///    arith::negf`/`addf`/`divf` calls directly -- a *purely* pure-primop
///    chain (`egraph.rs::is_straight_line`), so `Forward::walk` treated it
///    as one opaque, unnamed node, the same treatment a genuinely atomic op
///    like `matmul` correctly gets (it has a declared `derivative` rule to
///    match against). `sigmoid` has none (by design -- composes from its
///    own primitives instead), so an opaque `sigmoid` node could never
///    differentiate at all. Fixed by rewriting its body to compose from
///    real, named, already-differentiable calls instead (`Ring::neg`/`add`/
///    `div`, a new `Transcendental<Tensor<T,Dims...>>::exp`), the exact
///    shape its own scalar sibling already used -- see `stdlib/nn/nn.cleave`
///    and `stdlib/linalg/tensor.cleave`'s own doc comments for the fuller
///    story.
/// 2. Cross-leaf independence (`egraph.rs::inject_cross_leaf_independence`)
///    used to be injected only *within* one `Tensor` field's own leaves --
///    correct for a struct with a single `Tensor` field, but `Dense` has
///    two (`w`, `b`), and nothing told the solver `net.w`'s leaves and `net.
///    b`'s leaves are independent of *each other*. Fixed by injecting it
///    once, centrally, over a whole top-level parameter's *entire* leaf set.
/// 3. `Ring<T>`'s own `derivative div(a,b): div(sub(...),...)` references
///    `sub`/`mul` -- *same* algebra as `div` itself, which `monomorphize.rs`
///    ::`resolve_derivative_rule_expr_ty` used to treat as "nothing to
///    seed, `t`'s own specialization already covers it" — true only for a
///    literally self-recursive call (`MatMul::matmul` calling itself), not
///    for a *different* method of the same algebra. `Ring::sub<f32>`/`Ring::
///    mul<f32>` happen to always be called directly elsewhere in every
///    scalar-only test that exists, coincidentally masking this; `Ring::
///    sub<Tensor<f32,1,2>>` is never called anywhere in ordinary source
///    here, so it was never monomorphized, surfacing as `cps.rs`'s own
///    "call_names resolved ... but no such unit exists" panic.
///
/// `w = identity`, `b = 0` -- gradients computed independently in cleave
/// source itself (via the same stdlib `sigmoid`, not hand-copied constants),
/// checked with a tolerance, same posture as every other `derive()`
/// numeric-correctness test in this file.
/// `main.rs`'s own `fn main()` runs the whole pipeline on a dedicated 64MB-
/// stack worker thread, precisely because CPS conversion recurses once per
/// AST node and a big-enough program exceeds the OS default (1MB on Windows)
/// well before anything is actually wrong -- found necessary here for the
/// identical reason, empirically: this test's own synthesized derivative
/// (through `Dense::forward`'s inlined `matmul`/`add` and a separate,
/// sibling `sigmoid` call, times 9 leaves) genuinely overflows an ordinary
/// `#[test]` thread's own smaller
/// default stack (`STATUS_STACK_OVERFLOW`), even though the identical source
/// runs fine through the real CLI. Scoped to just this one test, not `run_
/// i32` itself -- `run_i32_with_optimization_pass`'s own doc comment already
/// documents why touching that shared 130+-test helper for one outlier's own
/// need was tried and reverted before.
#[test]
fn derive_through_dense_forward_computes_the_right_gradient() {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(derive_through_dense_forward_computes_the_right_gradient_body)
        .unwrap()
        .join()
        .unwrap();
}

fn derive_through_dense_forward_computes_the_right_gradient_body() {
    let context = context();
    let src = "
        use nn;
        use linalg;
        fn loss(x1: f32, x2: f32, y: f32, layer: Dense<f32, 2, 2>) -> f32 {
            let x = Tensor::<f32, 1, 2>(data: [[x1, x2]]);
            let h = sigmoid(layer.forward(x));
            let pred = h[0, 0] + h[0, 1];
            let err = pred - y;
            err * err
        }
        grad = derive(loss);
        fn main() -> i32 {
            let x1: f32 = 3.0;
            let x2: f32 = 4.0;
            let y: f32 = 0.0;
            let layer: Dense<f32, 2, 2> = Dense(
                w: Tensor::<f32, 2, 2>(data: [[1.0, 0.0], [0.0, 1.0]]),
                b: Tensor::<f32, 1, 2>(data: [[0.0, 0.0]])
            );
            let s0: f32 = sigmoid(x1);
            let s1: f32 = sigmoid(x2);
            let err: f32 = s0 + s1 - y;
            let d0: f32 = s0 * (1.0 - s0);
            let d1: f32 = s1 * (1.0 - s1);
            let expected_w00: f32 = 2.0 * err * d0 * x1;
            let expected_w01: f32 = 2.0 * err * d1 * x1;
            let expected_w10: f32 = 2.0 * err * d0 * x2;
            let expected_w11: f32 = 2.0 * err * d1 * x2;
            let expected_b0: f32 = 2.0 * err * d0;
            let expected_b1: f32 = 2.0 * err * d1;
            let g = grad(x1, x2, y, layer);
            let diff_w00: f32 = g.3.w[0, 0] - expected_w00;
            let diff_w01: f32 = g.3.w[0, 1] - expected_w01;
            let diff_w10: f32 = g.3.w[1, 0] - expected_w10;
            let diff_w11: f32 = g.3.w[1, 1] - expected_w11;
            let diff_b0: f32 = g.3.b[0, 0] - expected_b0;
            let diff_b1: f32 = g.3.b[0, 1] - expected_b1;
            let abs_w00: f32 = if diff_w00 < 0.0 { 0.0 - diff_w00 } else { diff_w00 };
            let abs_w01: f32 = if diff_w01 < 0.0 { 0.0 - diff_w01 } else { diff_w01 };
            let abs_w10: f32 = if diff_w10 < 0.0 { 0.0 - diff_w10 } else { diff_w10 };
            let abs_w11: f32 = if diff_w11 < 0.0 { 0.0 - diff_w11 } else { diff_w11 };
            let abs_b0: f32 = if diff_b0 < 0.0 { 0.0 - diff_b0 } else { diff_b0 };
            let abs_b1: f32 = if diff_b1 < 0.0 { 0.0 - diff_b1 } else { diff_b1 };
            if abs_w00 < 0.0001 and abs_w01 < 0.0001 and abs_w10 < 0.0001 and abs_w11 < 0.0001
                and abs_b0 < 0.0001 and abs_b1 < 0.0001
            { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// The `grad()` counterpart of `derive_through_dense_forward_computes_the_
/// right_gradient` just above -- the same hand-derived expected values and
/// numeric tolerance, `grad()` instead of `derive()`, `x` a direct `Tensor`
/// parameter rather than built from two scalars inside `loss` (`grad()`'s
/// own reverse-mode walk doesn't yet have an adjoint for a `Tensor`
/// *construction* from scalars — a real, separate, `derive()`-only-for-now
/// gap, out of scope for this test's own real point: `synthesize_one_
/// gradient`'s own struct-parameter support, `backward_walk`'s own `field_
/// ops` routing, `synthesize_struct_adjoint`/`build_zero_recursive`).
/// `layer: Dense<f32,2,2>` is read through `layer.forward(x)` (an
/// *inherent* method, transparently inlined, exposing `matmul(x, layer.w) +
/// layer.b` directly -- two nested field reads, `layer.w`/`layer.b`, both
/// routed into `field_contributions` and recombined into one `Dense(w:...,
/// b:...)` adjoint value once the walk reaches `layer`'s own e-class). Also
/// the first real exercise of the non-square `matmul` adjoint fix (`build_
/// pattern::resolve_multi_target_call_ty`) through a genuine struct field,
/// not just a bare parameter.
#[test]
fn grad_through_dense_forward_computes_the_right_gradient() {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(grad_through_dense_forward_computes_the_right_gradient_body)
        .unwrap()
        .join()
        .unwrap();
}

fn grad_through_dense_forward_computes_the_right_gradient_body() {
    let context = context();
    let src = "
        use nn;
        use linalg;
        fn loss(x: Tensor<f32, 1, 2>, y: f32, layer: Dense<f32, 2, 2>) -> f32 {
            let h = sigmoid(layer.forward(x));
            let pred = sum(h);
            let err = pred - y;
            err * err
        }
        gw = grad(loss);
        fn main() -> i32 {
            let x1: f32 = 3.0;
            let x2: f32 = 4.0;
            let x: Tensor<f32, 1, 2> = Tensor::<f32, 1, 2>(data: [[x1, x2]]);
            let y: f32 = 0.0;
            let layer: Dense<f32, 2, 2> = Dense(
                w: Tensor::<f32, 2, 2>(data: [[1.0, 0.0], [0.0, 1.0]]),
                b: Tensor::<f32, 1, 2>(data: [[0.0, 0.0]])
            );
            let s0: f32 = sigmoid(x1);
            let s1: f32 = sigmoid(x2);
            let err: f32 = s0 + s1 - y;
            let d0: f32 = s0 * (1.0 - s0);
            let d1: f32 = s1 * (1.0 - s1);
            let expected_w00: f32 = 2.0 * err * d0 * x1;
            let expected_w01: f32 = 2.0 * err * d1 * x1;
            let expected_w10: f32 = 2.0 * err * d0 * x2;
            let expected_w11: f32 = 2.0 * err * d1 * x2;
            let expected_b0: f32 = 2.0 * err * d0;
            let expected_b1: f32 = 2.0 * err * d1;
            let g = gw(x, y, layer);
            let diff_w00: f32 = g.2.w[0, 0] - expected_w00;
            let diff_w01: f32 = g.2.w[0, 1] - expected_w01;
            let diff_w10: f32 = g.2.w[1, 0] - expected_w10;
            let diff_w11: f32 = g.2.w[1, 1] - expected_w11;
            let diff_b0: f32 = g.2.b[0, 0] - expected_b0;
            let diff_b1: f32 = g.2.b[0, 1] - expected_b1;
            let abs_w00: f32 = if diff_w00 < 0.0 { 0.0 - diff_w00 } else { diff_w00 };
            let abs_w01: f32 = if diff_w01 < 0.0 { 0.0 - diff_w01 } else { diff_w01 };
            let abs_w10: f32 = if diff_w10 < 0.0 { 0.0 - diff_w10 } else { diff_w10 };
            let abs_w11: f32 = if diff_w11 < 0.0 { 0.0 - diff_w11 } else { diff_w11 };
            let abs_b0: f32 = if diff_b0 < 0.0 { 0.0 - diff_b0 } else { diff_b0 };
            let abs_b1: f32 = if diff_b1 < 0.0 { 0.0 - diff_b1 } else { diff_b1 };
            if abs_w00 < 0.0001 and abs_w01 < 0.0001 and abs_w10 < 0.0001 and abs_w11 < 0.0001
                and abs_b0 < 0.0001 and abs_b1 < 0.0001
            { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// `stdlib/rand/rand.cleave` -- confirms the central risk that plan was
/// actually written to de-risk: does the e-graph-based optimizer (`Forward::
/// walk`, run on *every* program's own body via `optimize_program`, not just
/// a `derive()`d one) ever fold two *separate* source-level `rand_uniform_
/// f32()` calls into one (same name, same empty argument list -- exactly
/// the shape ordinary hashconsing would treat as one shared node)? It
/// doesn't: `PrimOp::Extern` is one of the handful of primops `Forward::
/// walk` treats as a real effect and stops translating at (never reordered/
/// folded, unlike a pure arithmetic op), so two calls stay two independent
/// draws -- checked here directly, not just reasoned about. Also confirms
/// `rand_seed` makes the stream genuinely reproducible.
#[test]
fn rand_uniform_gives_independent_reproducible_draws() {
    let context = context();
    let src = "
        use rand;
        fn main() -> i32 {
            rand_seed(42);
            let a: f32 = rand_uniform_f32();
            let b: f32 = rand_uniform_f32();
            rand_seed(42);
            let c: f32 = rand_uniform_f32();
            let in_range = a >= 0.0 and a < 1.0 and b >= 0.0 and b < 1.0;
            if a != b and a == c and in_range { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// `Rand<T>::uniform`/`normal` (the actual cleave-level API `rand.cleave`
/// exposes, not the raw `cleave-rt` primitives just above) compose their own
/// affine transform correctly, dispatched via `Ring<T>` like every other
/// numeric op in this stdlib.
#[test]
fn rand_uniform_and_normal_algebra_methods_compute_correctly() {
    let context = context();
    let src = "
        use rand;
        fn main() -> i32 {
            rand_seed(7);
            let u: f32 = Rand::uniform(-2.0, 2.0);
            let in_range = u >= -2.0 and u < 2.0;
            let n: f32 = Rand::normal(10.0, 0.001);
            // `std = 0.001` -- a normal draw this tight around its own mean
            // is, for any practical purpose, guaranteed within +/-0.1.
            let diff: f32 = n - 10.0;
            let abs_diff: f32 = if diff < 0.0 { 0.0 - diff } else { diff };
            if in_range and abs_diff < 0.1 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// Stage 3 of the plan this feature was built from: not a real init helper
/// (Xavier/He are explicitly future work, `doc/backlog.md`), just enough to
/// prove the *mechanism* a real one will need actually works end to end --
/// a genuine nested loop with indexed writes (`buf[i][j] = ...`, `cps.rs`'s
/// own multi-index `Store`, already proven elsewhere, not a new mechanism)
/// filling a real mutable `[[T;C];R]` array one independent draw at a time,
/// then handed to `Tensor::<T,R,C>(data: buf)`. `T: Float + Rand` (two
/// simultaneous bounds on the same generic) is the one genuinely unverified
/// risk the plan flagged going in -- `infer.rs`'s own bound-checking code
/// iterates every bound in a `T: A + B` list, not just the first, so this
/// was expected to work, and does.
///
/// `random_matrix(-1.0, 1.0)` here, deliberately *no* explicit turbofish --
/// a real, separate, found-by-testing bug: `random_matrix::<f32, 2, 2>(...)`
/// (three generics -- one type, two const -- on an ordinary top-level fn
/// call, not a variadic struct construction) mis-binds the const args,
/// surfacing as a genuine type error one level in (`expected f32, found 2`)
/// and, without it, a downstream CPS failure resolving `zero()`'s own call
/// (`could not resolve call to zero`) that looked at first like an array-
/// repeat/generic-`zero()` gap but wasn't -- `--dump-inference-pass`
/// confirmed `zero()`'s own declaration-time type is the correct symbolic
/// `T` either way, turbofish present or not. Not investigated further here,
/// out of scope for this plan -- flagged in `doc/backlog.md` instead. Every
/// generic here (`T`, `R`, `C`) is already fully determined by `main`'s own
/// `let m: Tensor<f32, 2, 2> = ...` annotation, so no turbofish is actually
/// needed at this call site at all -- ordinary output-context inference,
/// the usual path, sidesteps the bug entirely rather than working around it.
#[test]
fn a_tensor_can_be_filled_with_independent_random_draws_via_nested_indexed_writes() {
    let context = context();
    let src = "
        use rand;
        use linalg;
        fn random_matrix<T: Float + Rand + Ring, const R: i32, const C: i32>(lo: T, hi: T) -> Tensor<T, R, C> {
            let mut buf: [[T; C]; R] = [[zero(); C]; R];
            for i in 0..R {
                for j in 0..C {
                    buf[i][j] = uniform(lo, hi);
                };
            };
            Tensor::<T, R, C>(data: buf)
        }
        fn main() -> i32 {
            rand_seed(3);
            let m: Tensor<f32, 2, 2> = random_matrix(-1.0, 1.0);
            let all_in_range = m[0, 0] >= -1.0 and m[0, 0] < 1.0
                and m[0, 1] >= -1.0 and m[0, 1] < 1.0
                and m[1, 0] >= -1.0 and m[1, 0] < 1.0
                and m[1, 1] >= -1.0 and m[1, 1] < 1.0;
            // Not *every* entry identical -- the exact failure mode `[value;
            // Dims...]` (broadcast-one-draw) would have produced.
            let not_degenerate = m[0, 0] != m[0, 1] or m[0, 0] != m[1, 0] or m[0, 0] != m[1, 1];
            if all_in_range and not_degenerate { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// The real payoff of the `cps.rs` array-repeat fix, over the previous
/// test's own hand-written nested loop: `[uniform(lo, hi); Dims...]`
/// directly, inside an impl generic over a whole pack (`Dims...`, not one
/// fixed rank like `Dense`'s own `In`/`Out`) -- one expression fills a
/// tensor of *any* rank with independent draws, mirroring `array_repeat_
/// over_a_whole_pack_builds_a_same_shaped_tensor_at_every_rank`'s own
/// multi-rank shape but for an impure per-leaf value instead of a pure
/// constant, which is exactly the case `[value; Dims...]` used to get wrong
/// (evaluate once, broadcast). Checked at rank 1 and rank 3 through the
/// *same* algebra impl, not two separate ones -- and non-degeneracy checked
/// at rank 3 specifically, since that's the shape most likely to expose the
/// old double-broadcast bug (more independent leaves for a
/// broadcast-one-draw regression to collide on by chance).
///
/// An algebra impl on purpose, not a plain top-level `fn` -- found, while
/// writing this test, that a top-level fn generic over a whole pack
/// (`fn random_fill<T, const Dims...: i32>(...)`) fails to monomorphize
/// correctly when called at *two different ranks* from the same program
/// (`expected Tensor<f32,2,2,2>, found Tensor<'t9,'t11>` -- a real, separate
/// bug, logged in `doc/backlog.md`, out of scope for this fix). An algebra
/// impl generic over the same pack, the pattern `Ring<Tensor<T,Dims...>>`
/// and `Filled<Tensor<T,Dims...>>` above already use, has no such problem
/// -- *unless* its own body also calls into a second, unrelated algebra
/// (here, `uniform`'s own `Rand<T>` dispatch): that combination, instantiated
/// at two ranks in one program, hangs on a real infinite-recursion stack
/// overflow -- a second, deeper, and also separate bug, also logged in
/// `doc/backlog.md`, also out of scope here. So each rank below is checked
/// via its *own*, separately compiled program instead of two calls sharing
/// one -- sidesteps both bugs while still proving the actual claim this
/// test exists for (`[uniform(...); Dims...]` truly fills independently, at
/// more than one rank, through one pack-generic impl).
#[test]
fn array_repeat_over_a_whole_pack_fills_a_tensor_with_independent_random_draws_at_any_rank() {
    let context = context();
    let rand_fill_src = |ty: &str, checks: &str| -> String {
        format!(
            "
            use rand;
            use linalg;
            algebra RandomFilled<T> {{
                fn random_fill() -> T;
            }}
            impl<T: Float + Rand + Ring, const Dims...: i32> RandomFilled<Tensor<T, Dims...>> {{
                fn random_fill() {{
                    Tensor::<T, Dims...>(data: [uniform(-1.0, 1.0); Dims...])
                }}
            }}
            fn main() -> i32 {{
                rand_seed(7);
                let v: {ty} = RandomFilled::random_fill();
                {checks}
            }}
            "
        )
    };
    let rank1 = rand_fill_src(
        "Tensor<f32, 4>",
        "let in_range = v[0] >= -1.0 and v[0] < 1.0
                and v[1] >= -1.0 and v[1] < 1.0
                and v[2] >= -1.0 and v[2] < 1.0
                and v[3] >= -1.0 and v[3] < 1.0;
            let not_degenerate = v[0] != v[1] or v[1] != v[2] or v[2] != v[3];
            if in_range and not_degenerate { 1 } else { 0 }",
    );
    let rank3 = rand_fill_src(
        "Tensor<f32, 2, 2, 2>",
        "let in_range = v[0,0,0] >= -1.0 and v[0,0,0] < 1.0
                and v[1,1,1] >= -1.0 and v[1,1,1] < 1.0;
            let not_degenerate = v[0,0,0] != v[0,0,1] or v[0,0,0] != v[0,1,0]
                or v[0,0,0] != v[1,0,0] or v[0,0,0] != v[1,1,1];
            if in_range and not_degenerate { 1 } else { 0 }",
    );
    assert_eq!(run_i32(&context, &rank1), 1);
    assert_eq!(run_i32(&context, &rank3), 1);
}

/// The turbofish-ordering bug the test above's own doc comment flagged and
/// worked around (dropping the turbofish entirely) -- now fixed for real,
/// tested here with the turbofish restored. Root cause (`Infer::generalize`'s
/// own doc comment, `infer.rs`, has the full story): `scheme.vars` used to
/// be built by scanning every free variable in the function's own fully-
/// substituted type and sorting the ids *numerically*, on the assumption
/// that a declared generic's own originally-minted `TyVar` always stays the
/// lowest-numbered (hence first-sorted) representative of whatever it later
/// gets unified with. `Tensor::<T,R,C>(data:...)` inside `random_matrix`'s
/// own body reminted its *own* fresh vars for `Tensor`'s own generics and
/// unified them against the caller's `T`/`R`/`C` -- `unify`'s own `(Ty::Var
/// (v), _) => bind(v, _)` binds whichever side is the var *to* the other
/// with no id-ordering preference at all, so `R`'s own final representative
/// could easily end up being one of `Tensor`'s own *later*-minted vars,
/// numerically higher than `T`'s -- scrambling `scheme.vars`'s order
/// relative to the declaration order `explicit_generics` (the turbofish
/// itself) is zipped against in `infer_call`. Surfaced as a nonsensical
/// `expected f32, found 2` type mismatch. Fixed by building `scheme.vars`
/// from the function's own *declared* generics list directly (resolving
/// each one's own original var through the current substitution, in the
/// order it was actually written) instead of recovering order via a
/// numeric-sort heuristic after the fact.
#[test]
fn explicit_turbofish_on_a_generic_function_whose_body_shares_generic_names_with_another_construction_resolves_in_declaration_order()
 {
    let context = context();
    let src = "
        use rand;
        use linalg;
        fn random_matrix<T: Float + Rand + Ring, const R: i32, const C: i32>(lo: T, hi: T) -> Tensor<T, R, C> {
            let mut buf: [[T; C]; R] = [[zero(); C]; R];
            for i in 0..R {
                for j in 0..C {
                    buf[i][j] = uniform(lo, hi);
                };
            };
            Tensor::<T, R, C>(data: buf)
        }
        fn main() -> i32 {
            rand_seed(3);
            let m: Tensor<f32, 2, 2> = random_matrix::<f32, 2, 2>(-1.0, 1.0);
            let all_in_range = m[0, 0] >= -1.0 and m[0, 0] < 1.0
                and m[0, 1] >= -1.0 and m[0, 1] < 1.0
                and m[1, 0] >= -1.0 and m[1, 0] < 1.0
                and m[1, 1] >= -1.0 and m[1, 1] < 1.0;
            let not_degenerate = m[0, 0] != m[0, 1] or m[0, 0] != m[1, 0] or m[0, 0] != m[1, 1];
            if all_in_range and not_degenerate { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// The counterpart bug (zero-*value*-parameter generic function, `stdlib/nn/
/// nn.cleave`'s `Init<T>::xavier`/`he`'s own doc comment has the full story)
/// -- also now fixed by the same `Infer::generalize` change, confirmed here
/// with an explicit turbofish specifically (the shape that used to be
/// misdiagnosed as a real *argument*, `"expects 0 argument(s), found 1"`).
#[test]
fn explicit_turbofish_on_a_zero_value_parameter_generic_function_resolves_correctly() {
    let context = context();
    let src = "
        use linalg;
        fn make_zero_matrix<T: Float, const R: i32, const C: i32>() -> Tensor<T, R, C> {
            Tensor::<T, R, C>(data: [[zero(); C]; R])
        }
        fn main() -> i32 {
            let m = make_zero_matrix::<f32, 2, 2>();
            if m[0, 0] == 0.0 and m[1, 1] == 0.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// `Init<T>::xavier`/`he` (`stdlib/nn/nn.cleave`) -- the real weight-init
/// helpers `doc/backlog.md`'s own "No weight-initialization helpers" item
/// asked for, now that `rand` exists. Deliberately an `algebra` (`Init<T>`),
/// not the plain top-level generic function `random_matrix` above used --
/// two real, separate, found-by-testing bugs specific to a *zero-value-
/// argument* generic call (`fn dense_xavier<T,In,Out>() -> Dense<T,In,Out>`,
/// every generic determined purely by the caller's own expected-type
/// context): the call site's own type never propagated into the body at all
/// (an internal `.to()` came back "ambiguous dispatch"), and an explicit
/// turbofish to work around that was itself misdiagnosed as a real argument
/// ("expects 0 argument(s), found 1"). `Ring::zero()` -- also zero-arg, also
/// purely output-type-determined -- already works everywhere, so routing
/// through the identical algebra-dispatch mechanism (`Init<Dense<T,In,Out>>
/// ::xavier()`) sidesteps both bugs at once rather than fixing either
/// (flagged in `doc/backlog.md`, not investigated further).
///
/// `Dense<f32,2,3>` -- `limit = sqrt(6/(2+3))`, computed independently here
/// (not a hand-copied constant), checked against every one of the six
/// generated weights, plus non-degeneracy (not every entry identical -- the
/// `[value; Dims...]` broadcast failure mode this whole design avoids).
#[test]
fn init_xavier_generates_weights_within_the_correct_glorot_bound() {
    let context = context();
    let src = "
        use nn;
        use linalg;
        fn main() -> i32 {
            rand_seed(11);
            let d: Dense<f32, 2, 3> = Init::xavier();
            let limit: f32 = mlir::math::sqrt(6.0 / 5.0);
            let neg_limit: f32 = 0.0 - limit;
            let in_range = d.w[0, 0] >= neg_limit and d.w[0, 0] < limit
                and d.w[0, 1] >= neg_limit and d.w[0, 1] < limit
                and d.w[0, 2] >= neg_limit and d.w[0, 2] < limit
                and d.w[1, 0] >= neg_limit and d.w[1, 0] < limit
                and d.w[1, 1] >= neg_limit and d.w[1, 1] < limit
                and d.w[1, 2] >= neg_limit and d.w[1, 2] < limit;
            let not_degenerate = d.w[0, 0] != d.w[0, 1] or d.w[0, 0] != d.w[1, 2];
            let bias_is_zero = d.b[0, 0] == 0.0 and d.b[0, 1] == 0.0 and d.b[0, 2] == 0.0;
            if in_range and not_degenerate and bias_is_zero { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// `Init<T>::he` -- statistical sanity over many independent `Dense`
/// constructions rather than an exact-bound check (a normal distribution has
/// no hard cutoff, unlike Xavier's uniform range above): sample mean near 0,
/// standard deviation in the right order of magnitude for `std = sqrt(2/4) =
/// 0.7071...` (`Dense<f32,4,1>`, `fan_in = 4`), same loose-bound posture as
/// `cleave-rt`'s own `rand_normal_f32_is_not_degenerate` test.
#[test]
fn init_he_generates_weights_with_the_right_statistics() {
    let context = context();
    let src = "
        use nn;
        use linalg;
        fn main() -> i32 {
            rand_seed(5);
            let mut sum: f32 = 0.0;
            let mut sum_sq: f32 = 0.0;
            let mut any_far: i32 = 0;
            for i in 0..200 {
                let d: Dense<f32, 4, 1> = Init::he();
                sum = sum + d.w[0, 0] + d.w[1, 0] + d.w[2, 0] + d.w[3, 0];
                sum_sq = sum_sq + d.w[0, 0] * d.w[0, 0] + d.w[1, 0] * d.w[1, 0]
                    + d.w[2, 0] * d.w[2, 0] + d.w[3, 0] * d.w[3, 0];
                let abs0: f32 = if d.w[0, 0] < 0.0 { 0.0 - d.w[0, 0] } else { d.w[0, 0] };
                if abs0 > 0.3 { any_far = 1; };
            };
            let n: f32 = 800.0;
            let mean: f32 = sum / n;
            let mean_abs: f32 = if mean < 0.0 { 0.0 - mean } else { mean };
            let var: f32 = sum_sq / n;
            // `std = sqrt(2/4) ~= 0.7071`, `var ~= 0.5` -- a loose band
            // around that, not exact-value tolerance.
            if mean_abs < 0.1 and var > 0.2 and var < 1.0 and any_far == 1 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

// ---------------------------------------------------------------------
// `doc/backlog.md`'s own "`CVal::Float` in the e-graph" item — a real
// float literal is now representable inside `egraph.rs`'s own `Forward`/
// `rebuild` (`CleaveLang::Float`, wrapping `ordered_float::OrderedFloat<f64>`)
// instead of silently stopping translation the moment it's touched.
// ---------------------------------------------------------------------

/// A real function whose own body computes over a float literal, run
/// through the *full* `--run` pipeline (`collect_units` -> `convert_program`
/// -> `optimize_program` -> `eliminate_dead_code` -> `lower_program` -> JIT)
/// -- proves `optimize_program` now translates the segment fully (instead of
/// stopping dead at the literal, per this module's own former doc comment)
/// and still reconstructs/codegens the exact right numeric result, not just
/// that construction alone doesn't panic (`egraph.rs`'s own unit tests
/// already prove that in isolation).
#[test]
fn a_float_literal_inside_a_straight_line_segment_still_computes_correctly_through_the_full_pipeline()
 {
    let context = context();
    let src = "
        fn main() -> i32 {
            let x: f32 = 3.0;
            let y: f32 = x * 2.0;
            if y == 6.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

// ---------------------------------------------------------------------
// `doc/backlog.md`'s own auto-diff item — `fprime = derive(f);`, a new
// top-level item mechanically synthesizing `fprime`'s own body via e-graph
// rewriting (`egraph.rs::synthesize_derivatives`/`derivative_rewrites`) --
// `f` itself must be an existing, non-generic top-level `fn` with every
// parameter (and its own return type) the same concrete numeric type.
// ---------------------------------------------------------------------

/// The single-parameter, scalar-derivative case: `d(x^2)/dx = 2x`, checked
/// numerically at `x = 3.0` (`fprime(3.0) == 6.0`) through the real `--run`
/// pipeline, not a synthetic e-graph-only test — proves signature
/// synthesis (`driver.rs`), `UnitBody::Derivative` (`cps.rs`), and the real
/// e-graph-based synthesis all work together end to end.
#[test]
fn derive_of_a_single_parameter_function_computes_the_scalar_derivative() {
    let context = context();
    let src = "
        fn f(x: f32) -> f32 { x * x }
        fprime = derive(f);
        fn main() -> i32 {
            let d: f32 = fprime(3.0);
            if d == 6.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// The multi-parameter, Jacobian case: `f(x,y) = x*y + x`, `df/dx = y+1`,
/// `df/dy = x` -- at `(x,y) = (3.0, 5.0)`, `[6.0, 3.0]`, chosen so neither
/// component is right by coincidence and neither equals the other. Proves
/// the `N > 1` tuple-wrapping path (`__Tuple2`, ordinary struct construction
/// -- `driver.rs::synthesize_derive_signatures`'s own doc comment) together
/// with the "different free variable -> 0" base rule (needed for `y`'s own
/// term to correctly vanish from `df/dx`, and `x`'s own from `df/dy`).
#[test]
fn derive_of_a_two_parameter_function_computes_the_jacobian_as_a_tuple() {
    let context = context();
    let src = "
        fn f(x: f32, y: f32) -> f32 { x * y + x }
        fprime = derive(f);
        fn main() -> i32 {
            let g: (f32, f32) = fprime(3.0, 5.0);
            if g.0 == 6.0 and g.1 == 3.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

// ---------------------------------------------------------------------
// `gw = grad(f);` -- Phase 3's own real reverse-mode backend
// (`egraph.rs::synthesize_one_gradient`/`backward_walk`), generalizing the
// Phase 0 spike's two hardcoded Rust match arms into real, `stdlib`-
// declared `adjoint` rules (`Ring<T>`/`Transcendental<T>`, `stdlib/num/
// num.cleave`). Same real `--run` pipeline as the `derive()` tests just
// above -- `run_i32` already threads `UnitBody::Derivative`'s own `is_grad`
// flag through unchanged, so `grad()` source syntax needed no test-harness
// changes at all.
// ---------------------------------------------------------------------

/// The exact single-parameter case `derive_of_a_single_parameter_function_
/// computes_the_scalar_derivative` already checks, just via `grad()`
/// instead -- `d(x^2)/dx = 2x` at `x = 3.0` is `6.0` under either mode,
/// numerically confirming the two backends agree on the simplest possible
/// case (one leaf, one contribution, no accumulation).
#[test]
fn grad_of_a_single_parameter_function_computes_the_scalar_derivative() {
    let context = context();
    let src = "
        fn f(x: f32) -> f32 { x * x }
        gw = grad(f);
        fn main() -> i32 {
            let d: f32 = gw(3.0);
            if d == 6.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// `grad()`'s own `N > 1` case -- the identical `f(x,y) = x*y + x` `derive_
/// of_a_two_parameter_function_computes_the_jacobian_as_a_tuple` already
/// checks, confirming `finish_derivative_body`'s own tuple-wrapping path is
/// shared correctly between both backends.
#[test]
fn grad_of_a_two_parameter_function_computes_the_gradient_as_a_tuple() {
    let context = context();
    let src = "
        fn f(x: f32, y: f32) -> f32 { x * y + x }
        gw = grad(f);
        fn main() -> i32 {
            let g: (f32, f32) = gw(3.0, 5.0);
            if g.0 == 6.0 and g.1 == 3.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// `f(a,b) = a*b + a` -- the exact worked example from the reverse-mode
/// design discussion (`egraph.rs::tests::spike_reverse_mode_matches_the_
/// hand_derived_gradient`'s own doc comment), now proven through the real
/// `grad()` source syntax end to end rather than only the isolated,
/// hand-built spike e-graph: `a` has two consumers (`mul` and `add`), so
/// this is the first real exercise of `accumulate_adjoint`'s own "second
/// contribution" branch (`Ring::add`-summing two independently-arrived
/// contributions to the same leaf) through the full compiler pipeline.
#[test]
fn grad_accumulates_multiple_contributions_to_the_same_parameter() {
    let context = context();
    let src = "
        fn f(a: f32, b: f32) -> f32 { a * b + a }
        gw = grad(f);
        fn main() -> i32 {
            let g: (f32, f32) = gw(3.0, 4.0);
            if g.0 == 5.0 and g.1 == 3.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// A composed, multi-level case exercising every one of `Transcendental<T>`'s
/// own declared `adjoint` rules (`exp`) alongside `Ring<T>`'s (`neg`, `add`,
/// `div`) through `sigmoid`'s own real, multi-op body (`stdlib/num/
/// num.cleave`), not just the two `Ring::add`/`Ring::mul` rules the simpler
/// cases above already cover -- `d(sigmoid(x))/dx = sigmoid(x)*(1-sigmoid(x))`,
/// checked with a tolerance (a real `exp` is involved, unlike every case
/// above) the same way `derive()`'s own transcendental-function tests
/// already do elsewhere in this file.
#[test]
fn grad_through_a_composed_function_using_transcendental_and_ring_adjoint_rules() {
    let context = context();
    let src = "
        use nn;
        fn loss(x: f32) -> f32 { sigmoid(x) }
        gw = grad(loss);
        fn main() -> i32 {
            let d: f32 = gw(0.0);
            // sigmoid(0) = 0.5, sigmoid'(0) = 0.5 * (1 - 0.5) = 0.25
            let lo: f32 = 0.24;
            let hi: f32 = 0.26;
            if d > lo and d < hi { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// `synthesize_one_gradient`'s own parameter scope check (`is_grad_
/// supported_ty`: scalar, `Tensor`, or a struct recursively built from
/// either -- a bare array is none of those) -- a real, located error
/// rather than a panic deeper in the pipeline or a silently wrong
/// gradient. `xs: [f32; 3]` (not a `Tensor`) passes `driver.rs::
/// synthesize_derive_signatures`'s own checks unchanged (identical to what
/// `derive()` already accepts for the same parameter shape) -- this is
/// `grad()`'s own, additional, narrower restriction.
#[test]
#[should_panic(
    expected = "reverse-mode differentiation only supports scalar, `Tensor`, and struct"
)]
fn grad_on_a_non_scalar_parameter_is_a_clean_error_not_a_panic_elsewhere() {
    let context = context();
    let src = "
        fn f(xs: [f32; 3]) -> f32 { xs[0] }
        gw = grad(f);
        fn main() -> i32 { 0 }
    ";
    run_i32(&context, src);
}

/// The real point of reverse-mode over `derive()`'s own `ParamShape::
/// Tensor` (`doc/backlog.md`'s own "digits-interop" item): a whole `Tensor`
/// parameter differentiates as *one* opaque leaf, not one marker per flat
/// element. `loss(w) = sum(w)` -- `d(sum(w))/dw = broadcast(1)`, a tensor of
/// all-ones, exercising `Sum<Tensor<T,N,M>,T>`'s own new `adjoint` rule
/// (`stdlib/nn/nn.cleave`) through the real, end-to-end `grad()` pipeline.
///
/// Getting here needed two further, real fixes beyond the `Tensor`-
/// parameter relaxation itself (`egraph.rs::synthesize_one_gradient`),
/// found chasing exactly this test:
/// - `Forward::walk` used to represent a call only when its own callee's
///   body was straight-line or a transparent chain of further calls
///   (`is_straight_line`/`is_transparent_chain`) -- `Sum::sum`'s own real
///   body (a nested `for` loop, needed since a straight-line, tensor-native
///   full reduction genuinely produces a 0-rank tensor this codebase has no
///   representation for) fit neither, so `sum(w)` wasn't representable at
///   all. Fixed: an algebra-dispatched method (`callee.origin.is_some()`)
///   is now *always* one opaque `Op` node regardless of its own body's
///   shape -- differentiation never looks inside one anyway, it consults
///   the method's own declared `derivative`/`adjoint` rule instead, so a
///   loop/branch inside a callee's own body is no more relevant to
///   representability than it already wasn't relevant to correctness.
/// - `monomorphize.rs::find_impl_for_target`'s own resolution used to be
///   *positional* -- it assumed the one concrete type an adjoint rule's own
///   argument reveals always corresponds to the target algebra's own
///   *first* declared generic. True by coincidence for `Transpose<A,B>`
///   (`stdlib/linalg/matrix.cleave`) but false for `Sum<Container,T>::
///   broadcast(u: T) -> Container` (`u`'s own type is `T`, the *second*
///   generic) -- worse, for a *same*-algebra call (`broadcast`, declared on
///   `Sum` itself, called from `Sum`'s own `adjoint sum` rule), the
///   argument alone can never reveal `Container`'s own `N`/`M` at all.
///   Fixed by resolving a same-algebra call's own targets directly from the
///   *enclosing* rule's own already-fully-resolved `type_env` instead of
///   re-deriving them from argument types.
#[test]
fn grad_of_a_tensor_parameter_computes_the_gradient_as_one_opaque_tensor_leaf() {
    let context = context();
    let src = "
        use nn;
        fn loss(w: Tensor<f32,2,2>) -> f32 { sum(w) }
        gw = grad(loss);
        fn main() -> i32 {
            let w = Tensor::<f32,2,2>(data:[[1.0,2.0],[3.0,4.0]]);
            let g = gw(w);
            if g[0,0] == 1.0 and g[0,1] == 1.0 and g[1,0] == 1.0 and g[1,1] == 1.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// Phase 5: `synthesize_one_gradient` now accepts a struct-typed parameter
/// whose own fields are all scalar/`Tensor`/struct (`is_grad_supported_ty`,
/// checked recursively) -- `backward_walk`'s own `field_ops` routing
/// (`accumulate_field_contribution`) sums each field's own contributions
/// independently and `synthesize_struct_adjoint` recombines them into one
/// real `Pair(a:...,b:...)` value once the walk reaches `p`'s own e-class,
/// zero-filling any field never actually read (`build_zero_recursive`).
/// `d(p.a+p.b)/dp = Pair(a:1,b:1)`, independent of `p`'s own values --
/// checked at two different `p`, confirming it's a real per-field
/// derivative, not a coincidence.
#[test]
fn grad_on_a_struct_parameter_with_only_scalar_fields_computes_the_right_gradient() {
    let context = context();
    let src = "
        struct Pair { a: f32, b: f32 }
        fn f(p: Pair) -> f32 { p.a + p.b }
        gw = grad(f);
        fn main() -> i32 {
            let p = Pair(a: 3.0, b: 4.0);
            let g = gw(p);
            if g.a == 1.0 and g.b == 1.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// `is_grad_supported_ty`'s own recursion still rejects a struct with a
/// genuinely unsupported field (`xs: [f32; 3]`, a bare array, not a
/// `Tensor`) -- a real, located error at the *outer* struct parameter,
/// not a guess or a panic three stages downstream.
#[test]
#[should_panic(
    expected = "reverse-mode differentiation only supports scalar, `Tensor`, and struct"
)]
fn grad_on_a_struct_parameter_with_an_unsupported_field_is_still_a_clean_error() {
    let context = context();
    let src = "
        struct HasArray { xs: [f32; 3] }
        fn f(p: HasArray) -> f32 { p.xs[0] }
        gw = grad(f);
        fn main() -> i32 { 0 }
    ";
    run_i32(&context, src);
}

/// `Forward::try_unroll_for_loop` (`egraph.rs`) -- a literal-bounded `for`
/// loop written directly in a derived function's own body now unrolls
/// instead of stopping `Forward::walk` dead. `loss(w) = sum_i(w * data[i])`
/// over a small local array, summed via a hand-written accumulation loop (no
/// `sum()` -- calling a *separate* unit whose own body loops is still out of
/// scope, blocked by the unrelated "multi-level call transparency" gap) --
/// `d(loss)/dw = sum_i(data[i]) = 1.0 + 2.0 + 3.0 = 6.0`. Also the real
/// proof `derivative-independent-zero`'s own occurs-check generalization
/// (`egraph.rs::depends_on_eclass`) is load-bearing: `data[i]` is a compound
/// `Load` node, not leaf-shaped, so the *old*, leaf-shape-only base rule
/// would have left `derivative(data[i], w)` permanently stuck.
#[test]
fn derive_of_a_function_containing_a_statically_bounded_for_loop_computes_the_correct_derivative() {
    let context = context();
    let src = "
        fn loss(w: f32) -> f32 {
            let data: [f32; 3] = [1.0, 2.0, 3.0];
            let mut total = 0.0;
            for i in 0..3 {
                total = total + w * data[i];
            };
            total
        }
        grad = derive(loss);
        fn main() -> i32 {
            let d: f32 = grad(10.0);
            if d == 6.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// The real proof `build_pattern`'s own cross-algebra resolution and
/// `NumberLit`-type fixes are load-bearing, not speculative: `stdlib/nn`'s
/// own `Activation::tanh` derivative rule (`1 - tanh(x)^2`) needs `Ring`'s
/// own `sub`/`mul` (a *different* algebra than the one the rule is declared
/// on) and a real `Float` literal `1` (matching a real runtime `f32`, not
/// the `Int`-shaped literal every rule used to build unconditionally).
/// `expected` is computed independently, via ordinary cleave arithmetic
/// calling the exact same underlying `Ring`/`Activation` units the
/// synthesized `fprime` itself calls -- bit-identical IEEE754 results,
/// checked with a plain `==`, no hand-computed literal to get subtly wrong.
#[test]
fn derive_of_tanh_uses_stdlib_nns_own_declared_derivative_rule() {
    let context = context();
    // `Transcendental::tanh` — qualified, not bare `tanh(x)` — `Activation
    // <T>` and `Transcendental<T>` (`stdlib/num/num.cleave`) both declare a
    // 1-arg `tanh` now (`tanh`'s own derivative rule moved to `Transcendental
    // <T>`, the math function's real home — `doc/backlog.md`'s own note),
    // so an unqualified call is genuinely ambiguous.
    let src = "
        use nn;
        fn f(x: f32) -> f32 { Transcendental::tanh(x) }
        fprime = derive(f);
        fn main() -> i32 {
            let x: f32 = 0.5;
            let t: f32 = Transcendental::tanh(x);
            let expected: f32 = 1.0 - t * t;
            let got: f32 = fprime(x);
            if got == expected { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// A real, general, pre-existing bug in `derivative_rule_to_rewrite`
/// (`egraph.rs`), found while giving `stdlib/num/num.cleave`'s new
/// `Transcendental<T>` its own `exp`/`tanh` derivative rules: the
/// differentiation variable's own internal pattern symbol used to be the
/// literal `?x` — silently *colliding* with a declared parameter genuinely
/// named `x` (`fn exp(x: T) -> T; derivative exp(x): mul(exp(x), d(x));`).
/// The rule then only ever matched when the method was applied *directly*
/// to the exact variable being differentiated (`exp(w)` w.r.t. `w`), never
/// through any real composition — invisible on `Ring<T>` (whose own params
/// are always named `a`/`b`) purely by naming coincidence. `exp(exp(w))`
/// is the minimal repro: `exp` composed with *itself*, no other algebra
/// involved at all, so this isn't about cross-algebra resolution — it's
/// specifically the `?x` collision. `d/dw[exp(exp(w))] = exp(exp(w)) *
/// exp(w)`, computed independently the same way the sibling test above
/// does.
#[test]
fn derive_through_a_composed_transcendental_call_whose_own_param_is_named_x() {
    let context = context();
    let src = "
        fn f(w: f32) -> f32 { exp(exp(w)) }
        fprime = derive(f);
        fn main() -> i32 {
            let w: f32 = 1.0;
            let inner: f32 = exp(w);
            let outer: f32 = exp(inner);
            let expected: f32 = outer * inner;
            let got: f32 = fprime(w);
            if got == expected { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// `Ring<T>`'s own `div`/`neg` never had a `derivative` rule at all before
/// this — only `add`/`sub`/`mul` did — a real, found-by-testing gap (not
/// speculative): `stdlib/nn/nn.cleave`'s own `sigmoid` needs the quotient
/// rule to be differentiable through `1.0 / (1.0 + exp(-x))` at all. `d/dw
/// [1/w] = -1/w^2`, computed independently.
#[test]
fn derive_through_ring_div_uses_the_quotient_rule() {
    let context = context();
    let src = "
        fn f(w: f32) -> f32 { 1.0 / w }
        fprime = derive(f);
        fn main() -> i32 {
            let w: f32 = 2.0;
            let expected: f32 = -1.0 / (w * w);
            let got: f32 = fprime(w);
            if got == expected { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// `doc/backlog.md`'s own "Multi-level call transparency" item — done, not
/// just discussed: `Forward::walk`'s own `Fix` arm now recognizes a callee
/// whose own body is itself a *chain* of further pure calls (`is_
/// transparent_chain`), not just a single primop (`is_straight_line`), and
/// walks straight *into* it instead of stopping — `sigmoid(w)`, a named
/// call to `stdlib/nn/nn.cleave`'s own `Activation<f32>::sigmoid` (whose
/// own body chains `neg`/`exp`/`add`/`div`), differentiates correctly now,
/// where it used to fail with "unsupported control flow" even though the
/// exact same body written *inline* already worked. `d/dw[sigmoid(w)^2] =
/// 2*sigmoid(w)^2*(1-sigmoid(w))`, computed independently the same way the
/// sibling tests above do.
#[test]
fn derive_through_a_named_call_to_a_multi_primitive_stdlib_function() {
    let context = context();
    let src = "
        use nn;
        fn loss(w: f32) -> f32 {
            let s = sigmoid(w);
            s * s
        }
        fprime = derive(loss);
        fn main() -> i32 {
            let w: f32 = 0.5;
            let s: f32 = sigmoid(w);
            let expected: f32 = 2.0 * s * s * (1.0 - s);
            let got: f32 = fprime(w);
            if got == expected { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// The real reason multi-level transparency needed alpha-renaming, not
/// just recursion: `sigmoid` called *twice* in the same function, with
/// *different* arguments (`sigmoid(w1)`, `sigmoid(w2)`) — a real, found-by-
/// testing bug in an early version of `is_transparent_chain`/`Forward::
/// walk`'s own inlining: without renaming each inlined copy's own internal
/// temporaries, the second call's own translation would silently alias the
/// first's in `self.env` (`try_unroll_for_loop`'s own identical concern for
/// repeated loop iterations). `d(s1*s2)/dw1 = s1*(1-s1)*s2`, `d(s1*s2)/dw2
/// = s1*s2*(1-s2)`, computed independently — checked with a tolerance, not
/// `==`: the two sides reach the same value through a genuinely different
/// sequence of multiplications (the synthesized gradient's own operand
/// order comes from the product rule's own expansion, not this test's
/// hand-written one), so an exact bit-for-bit match isn't guaranteed by
/// IEEE 754 even when both are mathematically correct — confirmed directly
/// (`0.19213304` vs `0.19213302`), not assumed.
#[test]
fn derive_through_two_separate_calls_to_the_same_multi_primitive_function() {
    let context = context();
    let src = "
        use nn;
        fn loss(w1: f32, w2: f32) -> f32 {
            let s1 = sigmoid(w1);
            let s2 = sigmoid(w2);
            s1 * s2
        }
        fprime = derive(loss);
        fn main() -> i32 {
            let w1: f32 = 0.5;
            let w2: f32 = 1.5;
            let s1: f32 = sigmoid(w1);
            let s2: f32 = sigmoid(w2);
            let expected0: f32 = s1 * (1.0 - s1) * s2;
            let expected1: f32 = s1 * s2 * (1.0 - s2);
            let got = fprime(w1, w2);
            let diff0: f32 = got.0 - expected0;
            let diff1: f32 = got.1 - expected1;
            let abs0: f32 = if diff0 < 0.0 { 0.0 - diff0 } else { diff0 };
            let abs1: f32 = if diff1 < 0.0 { 0.0 - diff1 } else { diff1 };
            if abs0 < 0.0001 and abs1 < 0.0001 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// Differentiating a *construction* itself — `egraph.rs::construction_
/// derivative_rewrites`, the built-in, non-declarable rule (`PrimOp::Struct`/
/// `PrimOp::Array` aren't algebra methods, so no `derivative` item in cleave
/// source could ever attach to them) distributing `d(...)` component-wise
/// onto a construction's own arguments: `d(Struct(f1:e1,...)) = Struct(f1:
/// d(e1),...)`. Found missing while reaching for a tensorial reformulation of
/// `examples/xor.cleave` (packing scalar weights into a `linalg::Tensor`
/// before using them) — `d(w[0])` where `w = Tensor::<f32,2>(data:[w1,w2])`
/// had no path to a derivative at all before this, confirmed directly via a
/// minimal probe (indexing alone, no arithmetic) before implementing the
/// fix. `f(w1,w2) = w[0] + w[1]*w[1]`, `df/dw1 = 1`, `df/dw2 = 2*w2` — proves
/// the rule composes correctly with `Index`'s own derivative rule (`stdlib/
/// linalg/tensor.cleave`) and `Ring`'s product rule, not just that
/// construction alone differentiates in isolation.
#[test]
fn derive_through_a_tagged_tensor_constructed_from_parameters_and_indexed_back() {
    let context = context();
    let src = "
        use linalg;
        fn f(w1: f32, w2: f32) -> f32 {
            let w = Tensor::<f32,2>(data:[w1, w2]);
            w[0] + w[1] * w[1]
        }
        fprime = derive(f);
        fn main() -> i32 {
            let w1: f32 = 3.0;
            let w2: f32 = 5.0;
            let expected0: f32 = 1.0;
            let expected1: f32 = 2.0 * w2;
            let got = fprime(w1, w2);
            if got.0 == expected0 and got.1 == expected1 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// Same `construction_derivative_rewrites` rule, exercised through an
/// ordinary user-declared struct (`PrimOp::Struct`, the `struct_projection_
/// rewrites` field-read path) rather than a `#[mlir_type(tensor)]`-tagged
/// one — proves the rule is genuinely generic over *any* reached struct
/// shape, not something that only happens to work for `linalg::Tensor`.
/// `p = Point(x: x, y: y*y)`, `d(p.x + p.y)/dx = 1`, `d(...)/dy = 2*y`.
#[test]
fn derive_through_a_plain_struct_constructed_from_parameters_and_read_back() {
    let context = context();
    let src = "
        struct Point { x: f32, y: f32 }
        fn f(x: f32, y: f32) -> f32 {
            let p = Point(x: x, y: y * y);
            p.x + p.y
        }
        fprime = derive(f);
        fn main() -> i32 {
            let x: f32 = 3.0;
            let y: f32 = 5.0;
            let expected0: f32 = 1.0;
            let expected1: f32 = 2.0 * y;
            let got = fprime(x, y);
            if got.0 == expected0 and got.1 == expected1 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// `doc/backlog.md`'s own "Toward a matmul-based tensorial XOR" item —
/// three real, separate bugs, all fixed here:
///
/// 1. `linalg::MatMul<A,B,C>`'s own new `derivative matmul(a, b): add(matmul
///    (d(a), b), matmul(a, d(b)));` needs its outer `add` call resolved at
///    `C` alone (a single concrete `Tensor` type), not the whole "A,B,C"
///    combined string `MatMul`'s own multi-target algebra reports —
///    `build_pattern`'s own per-parameter type environment (`egraph.rs`)
///    is what makes that resolve correctly now.
/// 2. `d(b)/da*` (the identity matrix `b` doesn't depend on any of the
///    `a*` parameters `a` is built from) needs a *tensor-shaped* zero, not
///    a bare scalar `0.0` — `derivative-independent-zero`'s own new
///    `IndependentZeroApplier`/`build_zero` (`egraph.rs`).
/// 3. `Ring<Tensor<f32,2,2>>::add` (the sum in `MatMul`'s own product rule
///    above) is a *generic* impl — nothing in this source ever calls it
///    directly (only `matmul` itself is called), so without `monomorphize.
///    rs`'s own new `seed_derivative_rule_references` it would never get a
///    real concrete unit built at all (`doc/backlog.md`'s own "Bug 3" entry
///    — this test used to need a throwaway direct `a + b` call added purely
///    to force monomorphization; no longer necessary).
///
/// `c = matmul(a, identity)`, `f = c[0,0] + c[1,1] = trace(a) = a1 + a4` —
/// `df/da1 = 1`, `df/da2 = 0`, `df/da3 = 0`, `df/da4 = 1`.
#[test]
fn derive_through_matmul_against_a_constant_identity_matrix_uses_the_product_rule_and_a_typed_zero()
{
    let context = context();
    let src = "
        use linalg;
        fn f(a1: f32, a2: f32, a3: f32, a4: f32) -> f32 {
            let a = Tensor::<f32,2,2>(data:[[a1, a2], [a3, a4]]);
            let b = Tensor::<f32,2,2>(data:[[1.0, 0.0], [0.0, 1.0]]);
            let c = matmul(a, b);
            c[0,0] + c[1,1]
        }
        fprime = derive(f);
        fn main() -> i32 {
            let g = fprime(1.0, 2.0, 3.0, 4.0);
            if g.0 == 1.0 and g.1 == 0.0 and g.2 == 0.0 and g.3 == 1.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32_with_optimization_pass(&context, src), 1);
}

/// `doc/backlog.md`'s own "gradient w.r.t. a struct parameter" item —
/// `derive()` on a function taking an ordinary, non-`Tensor` struct
/// parameter, recursing into its own field via a plain `Field` projection
/// (no eta-expansion needed at all — see `egraph.rs::build_param_shape`'s
/// own doc comment for why that's specific to `Tensor` leaves). `loss = (w*x
/// - y)^2`, `w=2, x=3, y=1` -> `pred=6, err=5, d(loss)/dw = 2*err*x = 30`.
#[test]
fn derive_of_a_loss_taking_an_ordinary_struct_parameter_computes_the_right_gradient() {
    let context = context();
    let src = "
        struct Weight { w: f32 }
        fn loss(x: f32, y: f32, p: Weight) -> f32 {
            let pred = p.w * x;
            let err = pred - y;
            err * err
        }
        grad = derive(loss);
        fn main() -> i32 {
            let p = Weight(w: 2.0);
            let g = grad(3.0, 1.0, p);
            if g.2.w == 30.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// The real target: `derive()` on a function taking a struct parameter
/// whose own field is `Tensor`-shaped — needs every piece together (`Index::
/// index` eta-expansion + reconstruction union so the chain rule can
/// differentiate *through* an operand that's otherwise opaque; the cross-
/// leaf independence fix, since `p.w[0,0]`/`p.w[0,1]` both report the same
/// `free_deps = {p}` and are otherwise indistinguishable to the general
/// `derivative-independent-zero` rule; and `Forward::walk`'s own `known_
/// types` fix for a real call's own return type, without which saturation
/// never terminates cleanly — see `egraph.rs`'s own doc comments on each).
/// Also the real end-to-end proof that a `Tensor`-typed *struct field*
/// actually lowers and runs (`mlir_lower.rs::store_native_shape_field`/
/// `load_native_shape_field`, needed once this got as far as real MLIR
/// codegen — found directly: `'llvm.store' op operand #0 must be LLVM type
/// with size, but got 'tensor<1x2xf32>'`, before those existed).
///
/// `loss = (w[0,0]*x + w[0,1] - y)^2`, `w=[2,3], x=4, y=5` -> `pred=11,
/// err=6, d/dw00 = 2*err*x = 48`, `d/dw01 = 2*err = 12`.
#[test]
fn derive_of_a_loss_taking_a_struct_parameter_with_a_tensor_field_computes_the_right_gradient() {
    let context = context();
    let src = "
        use linalg;
        struct Pair { w: Tensor<f32, 1, 2> }
        fn loss(x: f32, y: f32, p: Pair) -> f32 {
            let pred = p.w[0, 0] * x + p.w[0, 1];
            let err = pred - y;
            err * err
        }
        grad = derive(loss);
        fn main() -> i32 {
            let p = Pair(w: Tensor::<f32, 1, 2>(data: [[2.0, 3.0]]));
            let g = grad(4.0, 5.0, p);
            if g.2.w[0, 0] == 48.0 and g.2.w[0, 1] == 12.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// The cross-leaf independence fix specifically, isolated: `p.w[0,1]` is
/// never referenced anywhere in `loss`'s own body at all — its own gradient
/// must still resolve to a clean `0.0`, not get stuck (`egraph.rs::build_
/// param_shape`'s own doc comment on why the general `derivative-
/// independent-zero` rule can't discover this on its own for two leaves of
/// the same tensor).
#[test]
fn derive_of_a_loss_ignoring_one_tensor_field_element_gives_it_a_clean_zero_gradient() {
    let context = context();
    let src = "
        use linalg;
        struct Pair { w: Tensor<f32, 1, 2> }
        fn loss(x: f32, p: Pair) -> f32 {
            p.w[0, 0] * x
        }
        grad = derive(loss);
        fn main() -> i32 {
            let p = Pair(w: Tensor::<f32, 1, 2>(data: [[2.0, 3.0]]));
            let g = grad(4.0, p);
            if g.1.w[0, 0] == 4.0 and g.1.w[0, 1] == 0.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// `doc/backlog.md`'s own "`Ring<T>` gained a real `zero()`" item — a real,
/// long-latent `linalg::matmul` bug, found integrating `Ring::zero()` into
/// `derivative-independent-zero`: `matmul`'s own accumulator used to be
/// `mlir::tensor::empty()`, genuinely *uninitialized* memory, on the
/// strength of a comment claiming `linalg.matmul` "computes the entire
/// result itself, never accumulates" — wrong (`linalg.matmul` is BLAS-GEMM
/// semantics, `C := A@B + C`; confirmed directly via `--dump-mlir-lowered`,
/// past `convert-linalg-to-loops`, which genuinely loads the accumulator's
/// own current value before adding into it). Every prior test happened to
/// get zero-initialized memory back from the allocator anyway, purely by
/// allocator behavior, never guaranteed — masking this for as long as
/// `matmul` existed. `matmul(a, Ring::zero())` is the minimal, direct
/// reproduction: the result must be *exactly* the zero matrix regardless of
/// `a`'s own values — a wrong, uninitialized accumulator would instead
/// leak whatever garbage previously occupied that memory into the result.
#[test]
fn matmul_against_a_real_zero_tensor_gives_exactly_zero_not_uninitialized_garbage() {
    let context = context();
    let src = "
        use linalg;
        fn main() -> i32 {
            let a = Tensor::<f32,2,2>(data:[[1.0,2.0],[3.0,4.0]]);
            let z: Tensor<f32,2,2> = Ring::zero();
            let c = matmul(a, z);
            if c[0,0] == 0.0 and c[0,1] == 0.0 and c[1,0] == 0.0 and c[1,1] == 0.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// `Transpose<A,B>` (`stdlib/linalg/matrix.cleave`) — a real, non-square
/// case (`2x3 -> 3x2`), not just a square self-check, so a genuine index
/// permutation bug (as opposed to a no-op identity) couldn't hide. Reuses
/// `matmul`'s own established "zero-initialized `linalg.` accumulator"
/// fix, verified numerically here the same direct way `matmul_against_a_
/// real_zero_tensor_...` already verifies matmul's own version of it.
#[test]
fn transpose_of_a_non_square_tensor_permutes_both_axes_correctly() {
    let context = context();
    let src = "
        use linalg;
        fn main() -> i32 {
            let a = Tensor::<f32,2,3>(data:[[1.0,2.0,3.0],[4.0,5.0,6.0]]);
            let b = transpose(a);
            if b[0,0] == 1.0 and b[0,1] == 4.0 and b[1,0] == 2.0 and b[1,1] == 5.0 and b[2,0] == 3.0 and b[2,1] == 6.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// `Sum<Tensor<T,N,M>,T>`'s own new rank-2 impl (`stdlib/nn/nn.cleave`) — a
/// real, non-trivial 2x2 case (`1+2+3+4=10`), numerically verified through
/// the actual JIT, not just checked to compile.
#[test]
fn sum_of_a_rank_2_tensor_computes_the_right_total() {
    let context = context();
    let src = "
        use nn;
        fn main() -> i32 {
            let a = Tensor::<f32,2,2>(data:[[1.0,2.0],[3.0,4.0]]);
            let total = sum(a);
            if total == 10.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// Genuinely non-square (`3x2`, not `2x2`) — the real regression case for
/// the compiler bug `Sum::sum`'s own body doc comment describes (`stdlib/
/// nn/nn.cleave`): combining two different nested `for`-loop variables
/// bound by *different* const generics (`N`/`M`) into one array literal
/// silently conflates them during this generic impl's own body type-check,
/// invisible for a square `N == M` case.
#[test]
fn sum_of_a_non_square_rank_2_tensor_computes_the_right_total() {
    let context = context();
    let src = "
        use nn;
        fn main() -> i32 {
            let a = Tensor::<f32,3,2>(data:[[1.0,2.0],[3.0,4.0],[5.0,6.0]]);
            let total = sum(a);
            if total == 21.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

// The real target this whole item was chasing (`doc/backlog.md`'s own
// "toward simple ML" roadmap item) — a genuine 2-input, 2-hidden-neuron,
// 1-output network, differentiated end to end through two levels of named
// function calls (`loss` -> `forward` -> `sigmoid`, `forward` itself calling
// `sigmoid` three times) — is `examples/xor.cleave`, not a test here: a
// *single* forward pass through this exact shape (no `for` loop at all)
// already takes over 30 real seconds to saturate (confirmed directly, not
// guessed — the same `derive()`-saturation-cost item tracked below, but
// specifically about *nesting depth*, not batch size or parameter count: a
// *minimal* two-level-nested case, four parameters total, was equally slow).
// Too slow and too close to `synthesize_derivatives`'s own hardcoded 30-
// second wall-clock limit to belong in the ordinary, fast-running test
// suite — `examples/xor.cleave`'s own real, end-to-end training run (via the
// examples sweep, not `cargo test`) is where this shape is actually
// exercised and verified (a real, correctly-trained XOR truth table, not
// just a single gradient check).

/// A real, separate bug found in the same investigation: a *qualified*
/// call (`Transcendental::tanh(x)`) inside a fully-concrete, non-generic
/// impl's own body (`Activation<f32>::tanh`, `stdlib/nn/nn.cleave`) used
/// to have no path to its own `call_names` entry at all — `monomorphize.rs`
/// `ImplMatch::FoundConcrete` recorded the *outer* call site correctly but
/// never enqueued the matched template's own body onto `impl_worklist`
/// (unlike `ImplMatch::Found`), so nothing ever walked *its* body looking
/// for the nested qualified call; separately, `cps.rs::collect_units`'s
/// own non-generic-impl branch (which compiles *every* concrete impl
/// unconditionally, regardless of reachability) hardcoded `call_names:
/// HashMap::new()`, needing the identical discovery run directly against
/// its own re-inferred body. Both fixed together — this test specifically
/// exercises the second path (`Activation<f64>::tanh`, never reached from
/// `main` at all, only present because `use nn;` merges the whole crate),
/// confirmed by direct testing to crash the JIT natively before the fix
/// (`could not resolve call to Transcendental::tanh`), not just fail
/// cleanly.
#[test]
fn a_qualified_call_inside_a_concrete_impls_own_body_resolves_correctly() {
    let context = context();
    let src = "
        use nn;
        fn main() -> i32 {
            let t: f32 = Activation::tanh(0.5);
            let expected: f32 = Transcendental::tanh(0.5);
            if t == expected { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

// ---------------------------------------------------------------------
// `doc/backlog.md`'s "Qualified-call syntax" item (`Ring::mul(a, b)`).
// ---------------------------------------------------------------------

/// The real proof, not just that it type-checks: `Foo` and `Bar` both
/// declare `foo(x: i32) -> i32`, and both have an `impl ... <i32>` with a
/// genuinely different body. Before this feature, `foo(5)` alone would be
/// rejected as `AmbiguousOperator`; this exercises the *qualified* form for
/// each one in the same program. This is also the exact collision shape
/// `cps.rs::build_call_index`'s own key (method name + concrete arg/ret
/// types, no algebra) can't distinguish — proving `Foo::foo(5)` and
/// `Bar::foo(5)` actually run their own, different bodies (not silently the
/// same one, e.g. from a `HashMap::insert` overwrite) is the direct
/// end-to-end check that `monomorphize.rs` is routing each qualified call
/// through `call_names` instead.
#[test]
fn qualified_calls_to_two_algebras_colliding_on_the_same_method_and_type_run_their_own_bodies() {
    let context = context();
    let src = "
        algebra Foo<T> { fn foo(x: T) -> T; }
        algebra Bar<T> { fn foo(x: T) -> T; }
        impl Foo<i32> { fn foo(x) { x + 1 } }
        impl Bar<i32> { fn foo(x) { x * 2 } }
        fn main() -> i32 {
            let a: i32 = Foo::foo(5);
            let b: i32 = Bar::foo(5);
            if a == 6 and b == 10 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

// ---------------------------------------------------------------------
// `doc/backlog.md`'s "No dynamic-size collection" item — `DynArray<T>`
// (`stdlib/dynarray/dynarray.cleave`), a real, growable collection built
// entirely as an ordinary stdlib struct + algebra impls, no `Ty`/grammar
// changes.
// ---------------------------------------------------------------------

/// Runs `src` the same way `an_extern_fn_call_actually_executes_through_a_
/// registered_symbol` does (the simple scf/llvm pipeline — no tensor types
/// involved here, so `run_i32`'s own fuller bufferize pipeline isn't
/// needed). Registers every scalar-width `dynarray_*` symbol unconditionally
/// (`doc/backlog.md`'s own "no dead-code elimination" item — `use dynarray;`
/// compiles all six `RawBuffer<T>` impls in `stdlib/dynarray/dynarray.cleave`
/// regardless of which width the test's own program actually calls, exactly
/// the same reason `num`'s own `Rem::mod` is always present too), plus
/// `cleave_alloc` (every struct construction needs it) and whichever extra
/// symbols the caller passes in (the `_ptr` width, only ever declared
/// locally by a specific test's own `impl RawBuffer<SomeStruct>`, never
/// unconditionally compiled).
fn run_i32_with_dynarray_symbols(
    context: &Context,
    src: &str,
    extra_symbols: &[(&str, *mut ())],
) -> i32 {
    let (result, _sources) = compile(vec![("test.cleave".to_string(), src.to_string())], &[]);
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    let units = collect_units(&program, &registry);
    let cps_program = convert_program(units);
    let mlir_types = collect_mlir_types(&program);
    let struct_schemas = collect_struct_schemas(&program);
    let mut module = lower_program(context, &cps_program, &mlir_types, struct_schemas);
    assert!(
        module.as_operation().verify(),
        "generated MLIR module failed verification"
    );

    let pass_manager = pass::PassManager::new(context);
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
    }
    register_io_symbols(&engine);
    unsafe {
        for (name, ptr) in extra_symbols {
            engine.register_symbol(name, *ptr);
        }
    }
    let mut out: i32 = -1;
    unsafe {
        engine
            .invoke_packed("main", &mut [&mut out as *mut i32 as *mut ()])
            .expect("JIT invocation must succeed");
    }
    out
}

/// Pushes past the initial capacity (4), forcing at least one real
/// `RawBuffer<i32>::grow`, then reads every element back both via `.get(i)`
/// and via `v[i]` (the `Index<DynArray<T>,T>` fallback) — proving growth
/// preserves the earlier elements correctly, not just that the *last* push
/// landed right. `.len()` folded in here too rather than its own separate
/// test — a direct, minimal check against the known push count.
#[test]
fn a_dynarray_grows_past_its_initial_capacity_and_reads_back_correct_values() {
    let context = context();
    let src = "
        use dynarray;
        fn main() -> i32 {
            let v: DynArray<i32> = dynarray_new(4);
            v.push(10);
            v.push(20);
            v.push(30);
            v.push(40);
            v.push(50);
            v.push(60);
            if v.len() == 6
                and v.get(0) == 10 and v[0] == 10
                and v.get(3) == 40 and v[3] == 40
                and v.get(5) == 60 and v[5] == 60
            { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32_with_dynarray_symbols(&context, src, &[]), 1);
}

/// The struct-element proof (`doc/backlog-done.md`'s own note on why this
/// was scoped into v1 rather than deferred): every cleave struct value is
/// already an opaque pointer, so `DynArray<Point>` reuses the *exact same*
/// `_ptr`-suffixed `cleave-rt` functions the scalar-width impls use, via one
/// small `impl RawBuffer<Point>` written here (mechanical, zero new Rust
/// code) — direct end-to-end evidence a real heap-referenced struct
/// round-trips correctly through the raw pointer-width buffer, not just a
/// scalar.
#[test]
fn a_dynarray_of_structs_grows_and_reads_back_correct_field_values() {
    let context = context();
    let src = "
        use dynarray;
        struct Point { x: f64, y: f64 }
        impl RawBuffer<Point> {
            extern(dynarray_alloc_ptr) fn alloc(cap: i32) -> RawBuf;
            extern(dynarray_grow_ptr) fn grow(buf: RawBuf, old_cap: i32, new_cap: i32) -> RawBuf;
            extern(dynarray_get_ptr) fn get(buf: RawBuf, i: i32) -> Point;
            extern(dynarray_set_ptr) fn set(buf: RawBuf, i: i32, x: Point);
        }
        fn main() -> i32 {
            let v: DynArray<Point> = dynarray_new(4);
            v.push(Point(x: 1.0, y: 2.0));
            v.push(Point(x: 3.0, y: 4.0));
            v.push(Point(x: 5.0, y: 6.0));
            v.push(Point(x: 7.0, y: 8.0));
            v.push(Point(x: 9.0, y: 10.0));
            let p0: Point = v.get(0);
            let p4: Point = v.get(4);
            if v.len() == 5 and p0.x == 1.0 and p0.y == 2.0 and p4.x == 9.0 and p4.y == 10.0 { 1 } else { 0 }
        }
    ";
    let symbols: &[(&str, *mut ())] = &[
        (
            "dynarray_alloc_ptr",
            cleave_rt::dynarray_alloc_ptr as *mut (),
        ),
        ("dynarray_grow_ptr", cleave_rt::dynarray_grow_ptr as *mut ()),
        ("dynarray_get_ptr", cleave_rt::dynarray_get_ptr as *mut ()),
        ("dynarray_set_ptr", cleave_rt::dynarray_set_ptr as *mut ()),
    ];
    assert_eq!(run_i32_with_dynarray_symbols(&context, src, symbols), 1);
}

/// A real, previously-latent bug, found directly while writing `examples/
/// convex_hull.cleave` — `callgraph.rs`'s own whole-program pass applies a
/// Haskell-style Monomorphism Restriction to *any* zero-parameter top-level
/// fn (`f.params.is_empty()`), regardless of whether it's otherwise generic
/// — `dynarray_new<T>()`'s own original, argument-less signature hit this
/// exactly: its `T` was never generalized at all, staying a single, shared,
/// monomorphic type variable for the *entire compiled program*. Invisible in
/// every earlier test (each only ever used `DynArray` at one concrete `T`
/// per compiled program) — this is the direct regression proof: `DynArray<
/// i32>` and `DynArray<Point>` constructed and used *simultaneously* in the
/// same program, both resolving to their own correct, independent element
/// type. Fixed by giving `dynarray_new` a real parameter (`initial_cap: i32`
/// — genuinely useful on its own, not just a workaround) so it no longer
/// hits the nullary gate at all.
#[test]
fn dynarray_generalizes_correctly_across_two_different_concrete_types_in_one_program() {
    let context = context();
    let src = "
        use dynarray;
        struct Point { x: f64, y: f64 }
        impl RawBuffer<Point> {
            extern(dynarray_alloc_ptr) fn alloc(cap: i32) -> RawBuf;
            extern(dynarray_grow_ptr) fn grow(buf: RawBuf, old_cap: i32, new_cap: i32) -> RawBuf;
            extern(dynarray_get_ptr) fn get(buf: RawBuf, i: i32) -> Point;
            extern(dynarray_set_ptr) fn set(buf: RawBuf, i: i32, x: Point);
        }
        fn main() -> i32 {
            let ints: DynArray<i32> = dynarray_new(4);
            ints.push(1);
            ints.push(2);
            let points: DynArray<Point> = dynarray_new(4);
            points.push(Point(x: 5.0, y: 6.0));
            let p0: Point = points.get(0);
            if ints.len() == 2 and ints.get(0) == 1 and ints.get(1) == 2
                and points.len() == 1 and p0.x == 5.0 and p0.y == 6.0
            { 1 } else { 0 }
        }
    ";
    let symbols: &[(&str, *mut ())] = &[
        (
            "dynarray_alloc_ptr",
            cleave_rt::dynarray_alloc_ptr as *mut (),
        ),
        ("dynarray_grow_ptr", cleave_rt::dynarray_grow_ptr as *mut ()),
        ("dynarray_get_ptr", cleave_rt::dynarray_get_ptr as *mut ()),
        ("dynarray_set_ptr", cleave_rt::dynarray_set_ptr as *mut ()),
    ];
    assert_eq!(run_i32_with_dynarray_symbols(&context, src, symbols), 1);
}

// ---------------------------------------------------------------------
// `doc/backlog.md`'s "a while-loop condition needing more than one chained
// real call breaks `lower_loop`" item.
// ---------------------------------------------------------------------

/// `i < hull.len()` needs *two* sequential real calls before the branch
/// (`DynArray::len<...>`, then `Ord::lt<i32>` against its result) — found
/// directly while writing `examples/convex_hull.cleave`, where
/// `mlir_lower.rs::lower_loop` used to panic (`"a loop's own condition-
/// continuation must be a bare \`if\`"`), since it only ever handled a
/// condition CPS-converting to *exactly one* call. Real end-to-end proof,
/// not just that it lowers cleanly: pushes 3 elements, loops while `i` is
/// less than the *live* `hull.len()` (not a value cached before the loop),
/// printing each one — a wrong loop bound would either skip elements or
/// read out of bounds, not just fail to compile.
#[test]
fn a_while_condition_needing_two_chained_calls_lowers_and_runs_correctly() {
    let context = context();
    let src = "
        use dynarray;
        fn main() -> i32 {
            let hull: DynArray<i32> = dynarray_new(4);
            hull.push(10);
            hull.push(20);
            hull.push(30);
            let mut i: i32 = 0;
            let mut sum: i32 = 0;
            while i < hull.len() {
                sum = sum + hull.get(i);
                i = i + 1;
            };
            if i == 3 and sum == 60 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32_with_dynarray_symbols(&context, src, &[]), 1);
}

// ---------------------------------------------------------------------
// `doc/backlog-done.md`'s own "break value" item (`break`/`loop { }`).
// ---------------------------------------------------------------------

/// Sums until a threshold, `break;` instead of relying purely on the
/// condition — asserts the early-exit sum, not just "didn't crash": a
/// wrong guard (e.g. one that doesn't actually stop `sum = sum + i;` from
/// running once more) would silently produce `10` (`0+1+2+3+4`) instead of
/// the correct `10` too by coincidence at `i==5`, so the real proof is at
/// `i==3` instead, where a bug would show `6` (correct) vs `9` (one extra
/// iteration snuck through) — deliberately not the boundary an accidental
/// off-by-one could hide behind.
#[test]
fn a_while_loop_exits_early_via_a_bare_break() {
    let context = context();
    let src = "
        fn main() -> i32 {
            let mut i: i32 = 0;
            let mut sum: i32 = 0;
            while i < 100 {
                if i == 3 {
                    break;
                };
                sum = sum + i;
                i = i + 1;
            };
            sum
        }
    ";
    assert_eq!(run_i32(&context, src), 3);
}

/// `break` several stack frames deep (`if` inside `while`) — proves it
/// isn't limited to a top-level statement in the loop body, and that
/// nothing textually *after* the `if` (in the same loop-body block) runs
/// once broken (`doc/backlog-done.md`'s own note on the real bug this
/// caught directly: an earlier version of this fix let `sum = sum + i;`
/// keep executing once more after the break, with stale values).
#[test]
fn a_break_inside_a_nested_if_inside_a_while_exits_the_correct_loop() {
    let context = context();
    let src = "
        fn main() -> i32 {
            let mut i: i32 = 0;
            let mut sum: i32 = 0;
            while i < 100 {
                if i == 5 {
                    break;
                };
                sum = sum + i;
                i = i + 1;
            };
            sum
        }
    ";
    // 0+1+2+3+4 -- `i == 5` itself never contributes.
    assert_eq!(run_i32(&context, src), 10);
}

#[test]
fn a_for_loop_can_break_early() {
    let context = context();
    let src = "
        fn main() -> i32 {
            let mut sum: i32 = 0;
            for i in 0..100 {
                if i == 5 {
                    break;
                };
                sum = sum + i;
            };
            sum
        }
    ";
    assert_eq!(run_i32(&context, src), 10);
}

#[test]
fn a_for_in_loop_can_break_early() {
    let context = context();
    let src = "
        fn main() -> i32 {
            let arr: [i32; 6] = [10, 20, 30, 40, 50, 60];
            let mut sum: i32 = 0;
            for x in arr {
                if x == 40 {
                    break;
                };
                sum = sum + x;
            };
            sum
        }
    ";
    assert_eq!(run_i32(&context, src), 60);
}

/// `let x = loop { ... break 5; };` — the only loop kind that can produce a
/// real value via `break`.
#[test]
fn a_loop_expr_returns_the_value_from_break() {
    let context = context();
    let src = "
        fn main() -> i32 {
            let mut n: i32 = 0;
            let x: i32 = loop {
                n = n + 1;
                if n == 7 {
                    break n * 10;
                };
            };
            x
        }
    ";
    assert_eq!(run_i32(&context, src), 70);
}

/// A break inside an *inner* loop must not affect the *outer* loop's own
/// iteration count — each `break` targets its nearest enclosing loop only.
#[test]
fn nested_loops_break_only_the_innermost_one() {
    let context = context();
    let src = "
        fn main() -> i32 {
            let mut outer_count: i32 = 0;
            let mut inner_total: i32 = 0;
            for o in 0..3 {
                outer_count = outer_count + 1;
                let mut i: i32 = 0;
                while i < 100 {
                    if i == 4 {
                        break;
                    };
                    inner_total = inner_total + 1;
                    i = i + 1;
                };
            };
            if outer_count == 3 and inner_total == 12 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

/// The "zero cost when unused" requirement (`cps.rs::loop_contains_break`'s
/// own doc comment), checked directly, not just assumed: an ordinary loop
/// containing no `break` anywhere must not gain any of the guard machinery
/// (no `__loop_running` carried slot, no extra `Logic::and` call folded into
/// its own condition) — the generated MLIR text has no trace of it. A
/// silent regression here (every existing loop suddenly guarded) would be
/// the single most damaging possible mistake in this feature.
#[test]
fn a_loop_with_no_break_produces_no_guard_machinery() {
    let context = context();
    let src = "
        fn main() -> i32 {
            let mut i: i32 = 0;
            let mut sum: i32 = 0;
            while i < 10 {
                sum = sum + i;
                i = i + 1;
            };
            sum
        }
    ";
    let text = lower(&context, src);
    // `Logic::and<...>` itself is *always* present in some form -- not just
    // its own declaration (the prelude compiles every function in
    // unconditionally, `doc/backlog.md`'s own "no dead-code elimination"
    // item), but even a real *call* to it, from `Rem::mod`'s own
    // pre-existing, unrelated body (integer modulo sign-correction) —
    // checking the whole module's own text isn't precise enough. `main`
    // sorts last among this module's own functions (`convert_program`'s own
    // alphabetical ordering — every algebra-qualified name starts
    // uppercase, sorting before lowercase `main` in ASCII), so everything
    // after `func.func @main` is `main`'s own body specifically.
    let main_text = text
        .split("func.func @main")
        .nth(1)
        .unwrap_or_else(|| panic!("no `@main` found in:\n{text}"));
    assert!(
        !main_text.contains("Logic::and"),
        "an unguarded loop's own `main` must not call `Logic::and` at all, got:\n{main_text}"
    );
    assert_eq!(run_i32(&context, src), 45);
}

// `(T1,T2)`/`(a,b)`/`t.0` — `doc/backlog.md`'s own former "Tuples as a
// language feature" item, desugared entirely at `lower.rs` time into
// ordinary generic-struct syntax naming a synthesized `__TupleN` struct
// (`driver.rs::synthesize_tuple_structs`, injected into every compiled
// program automatically — see its own doc comment) — no new `Ty`/CPS/MLIR
// primitive anywhere, these tests exercise the *existing* struct pipeline
// through the tuple sugar, not a new lowering path.

#[test]
fn a_tuple_constructs_and_reads_both_fields_back_correctly() {
    let context = context();
    let out = run_i32(
        &context,
        "fn main() -> i32 { let t: (i32, i32) = (3, 4); t.0 * 10 + t.1 }",
    );
    assert_eq!(out, 34);
}

#[test]
fn a_heterogeneous_tuple_reads_a_non_i32_field_back_correctly() {
    let context = context();
    // `t.1` stays `f64`-typed all the way through (no `i32`-narrowing
    // `Convert` impl exists, or is needed) -- folded into the `i32` return
    // via a comparison instead, matching `run_i32`'s own return type.
    let out = run_i32(
        &context,
        "fn main() -> i32 { let t: (i32, f64) = (3, 4.5); if t.1 == 4.5 { t.0 } else { -1 } }",
    );
    assert_eq!(out, 3);
}

#[test]
fn a_nested_tuple_reads_back_correctly_through_a_chained_field_access() {
    let context = context();
    let out = run_i32(
        &context,
        "fn main() -> i32 { let t: ((i32, i32), i32) = ((1, 2), 3); t.0.0 * 100 + t.0.1 * 10 + t.1 }",
    );
    assert_eq!(out, 123);
}

#[test]
fn a_tuple_field_is_mutable_through_a_let_mut_binding() {
    let context = context();
    let out = run_i32(
        &context,
        "fn main() -> i32 { let mut t: (i32, i32) = (10, 20); t.0 = 99; t.0 + t.1 }",
    );
    assert_eq!(out, 119);
}

#[test]
fn a_function_over_an_explicit_tuple_parameter_type_returns_the_right_element() {
    let context = context();
    let out = run_i32(
        &context,
        "fn first(t: (i32, bool)) -> i32 { t.0 }\nfn main() -> i32 { first((7, true)) }",
    );
    assert_eq!(out, 7);
}

/// The real end-to-end validation this whole feature was built to satisfy —
/// `doc/backlog.md`'s own former "Multi-argument, heterogeneous `print`"
/// item: `print(("x=", x, "y=", y))`, a genuinely heterogeneous tuple
/// (two string literals, an `i32`, an `f64`) dispatched through `stdlib/io/
/// io.cleave`'s own `impl<A: Print, ...> Print<(A, ...)>` — each element
/// independently dispatched to whichever `Print<T>` impl matches its own
/// concrete type, unrolled by hand (no variadic generics — deliberately
/// deferred, see `doc/backlog.md`'s own "variadic generics" item) across
/// four separately-written impls (arity 2 through 4). Also the real
/// end-to-end proof for two separate, pre-existing bugs found *while*
/// building this validation (both fixed, not tuple-specific): a string
/// literal is itself `[i8;N]`, an array-typed field once it's read back out
/// of the tuple's own storage (`x.0`) — crossing `Print<[i8;N]>`'s own
/// extern-call boundary needs `mlir_lower.rs::array_ptr_and_len`'s own
/// `!llvm.ptr`-vs-`memref` branch (an array field is never a real `memref`,
/// only a standalone one is); and reaching `Print<[i8;N]>::print` at all,
/// once a *second* generic `Print<...>` impl exists in the same crate,
/// needed `monomorphize.rs`/`cps.rs`'s own per-specialization extern-ness
/// fix (a real, previously-latent duplicate-specialization bug, invisible
/// until two generic impls of the same algebra/method coexisted for the
/// first time).
#[test]
fn a_multi_argument_heterogeneous_print_call_prints_every_element_in_order() {
    let context = context();
    // `run_i32_with_dynarray_symbols`'s own base set (`register_io_symbols`)
    // already covers every `print_*`/`Display<T>` symbol this needs --
    // `extra_symbols` is for anything genuinely *outside* that (none here).
    let out = run_i32_with_dynarray_symbols(
        &context,
        "use io;\nfn main() -> i32 { let x: i32 = 3; let y: f64 = 4.5; print((\"x=\", x, \"y=\", y)); 0 }",
        &[],
    );
    assert_eq!(out, 0);
}

/// `doc/backlog.md`'s own "Variadic generics" item — a const-generic pack
/// (`struct Tensor<T, const Dims...: i32> { data: [T; Dims...] }`),
/// resolved via an explicit turbofish at a construction site, constructs
/// and reads its own field back correctly for an ordinary (heap-allocated)
/// struct — the real end-to-end proof, not just a type-checking one. See
/// `an_algebra_impl_generic_over_a_pack_computes_through_a_tagged_native_
/// tensor_correctly`, below, for the `#[mlir_type(tensor)]`-tagged
/// native-value case (`Vector`/`Matrix`/`Tensor`'s own real representation)
/// generic over a pack — a separate, deeper mechanism, now also working.
#[test]
fn a_const_generic_pack_struct_constructs_and_reads_its_field_back_correctly() {
    let context = context();
    let src = "struct Tensor<T, const Dims...: i32> { data: [T; Dims...] }
        fn main() -> i32 {
            let t = Tensor::<i32, 2, 3>(data: [[1, 2, 3], [4, 5, 6]]);
            t.data[1, 2]
        }";
    assert_eq!(run_i32(&context, src), 6);
}

#[test]
fn a_const_generic_pack_struct_supports_a_different_rank_at_a_different_call_site() {
    // Two separate concrete instantiations of the *same* pack-generic
    // struct, at genuinely different ranks (2 dims vs. 3 dims) — proving
    // the pack's own arity is resolved independently per construction
    // site, not baked in once globally.
    let context = context();
    let src = "struct Tensor<T, const Dims...: i32> { data: [T; Dims...] }
        fn main() -> i32 {
            let a = Tensor::<i32, 4>(data: [10, 20, 30, 40]);
            let b = Tensor::<i32, 2, 2, 2>(data: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
            a.data[2] + b.data[1, 0, 1]
        }";
    assert_eq!(run_i32(&context, src), 36);
}

/// The deeper half of the pack mechanism: a `#[mlir_type(tensor)]`-tagged
/// struct (`stdlib/linalg/tensor.cleave`'s own `Tensor`, a native MLIR
/// `tensor<...>` SSA value, never a heap-allocated reference) generic over
/// a pack, *and* a real algebra impl (`Ring`, real bodies — `mlir::arith::
/// addf`, not an empty tag impl) generic over the same pack — one impl
/// covering every rank, the mechanism `stdlib/linalg/tensor.cleave`'s own
/// `Ring<Tensor<T,Dims...>>` now actually uses. Needed `Ty::Pack`/`Ty::
/// PackResolved` (symbolic packs
/// surviving declaration-time inference, only resolving once a real call
/// site unifies against them) plus two real, separately-fixed bugs: `unify`
/// treating two *identical* open pack variables meeting each other as a
/// mismatch (the common case at declaration time — an algebra's own
/// expected type and an impl's own annotated declaration share the same
/// pack `TyVar`, from the same `impl_mapping`), and `lower_field_access`/
/// `lower_array_load` never having a case for a tagged struct's own native
/// tensor representation at all (always assumed a `memref` or `!llvm.ptr`,
/// silently never exercised by the existing stdlib because `Vector`/
/// `Matrix`/`Tensor`'s own `Index` impls read through the reserved `mlir::
/// tensor::extract` intrinsic instead of ordinary field-access syntax) — see
/// `doc/backlog.md`'s own note for both.
#[test]
fn an_algebra_impl_generic_over_a_pack_computes_through_a_tagged_native_tensor_correctly() {
    let context = context();
    let src = "
        struct Box3<T: Float, const Dims...: i32> { data: [T; Dims...] }

        algebra NativeShape<T> {}
        #[mlir_type(tensor)]
        impl<T: Float, const Dims...: i32> NativeShape<Box3<T, Dims...>> {}

        impl<T: Float, const Dims...: i32> Ring<Box3<T, Dims...>> {
            fn add(a: Box3<T, Dims...>, b: Box3<T, Dims...>) -> Box3<T, Dims...> { mlir::arith::addf(a, b) }
            fn sub(a: Box3<T, Dims...>, b: Box3<T, Dims...>) -> Box3<T, Dims...> { mlir::arith::subf(a, b) }
            fn mul(a: Box3<T, Dims...>, b: Box3<T, Dims...>) -> Box3<T, Dims...> { mlir::arith::mulf(a, b) }
            fn div(a: Box3<T, Dims...>, b: Box3<T, Dims...>) -> Box3<T, Dims...> { mlir::arith::divf(a, b) }
            fn neg(a: Box3<T, Dims...>) -> Box3<T, Dims...> { mlir::arith::negf(a) }
        }

        fn main() -> i32 {
            let a = Box3::<f64, 2, 2>(data: [[1.0, 2.0], [3.0, 4.0]]);
            let b = Box3::<f64, 2, 2>(data: [[10.0, 10.0], [10.0, 10.0]]);
            let c = a + b;
            if c.data[0, 0] == 11.0 and c.data[1, 1] == 14.0 { 1 } else { 0 }
        }";
    assert_eq!(run_i32(&context, src), 1);
}

/// `mlir::tensor::extract`'s own variadic-index-array form
/// (`lower_tensor_extract_spread`, `mlir_lower.rs`) — a pack-generic
/// `Index<Box3<T,Dims...>,T>` impl passes a single `idx: [i32; 2]` array
/// straight through to `mlir::tensor::extract(t, idx)`, instead of
/// spreading it into separate scalar arguments at the cleave call site (no
/// such spread syntax exists — see `doc/backlog.md`'s own note). Both the
/// raw intrinsic call *and* the real `t[i,j]` sugar (dispatched through
/// `Index`, exactly like `Vector`/`Matrix`/`Tensor`'s own existing,
/// fixed-arity impls, `stdlib/linalg/tensor.cleave`) are checked — the
/// second is the one that actually matters for the migration, the first
/// isolates the new lowering mechanism on its own.
#[test]
fn a_pack_generic_index_impl_extracts_through_a_variadic_index_array() {
    let context = context();
    let src = "
        use linalg;
        struct Box3<T: Float, const Dims...: i32> { data: [T; Dims...] }
        #[mlir_type(tensor)]
        impl<T: Float, const Dims...: i32> NativeShape<Box3<T, Dims...>> {}
        impl<T: Float, const Dims...: i32> Index<Box3<T, Dims...>, T> {
            fn index(t: Box3<T, Dims...>, idx: [i32; 2]) -> T { mlir::tensor::extract(t, idx) }
        }
        fn main() -> i32 {
            let a = Box3::<f64, 2, 2>(data: [[1.0, 2.0], [3.0, 4.0]]);
            let idx = [1, 0];
            let direct: f64 = mlir::tensor::extract(a, idx);
            let sugared = a[1, 0];
            if direct == 3.0 and sugared == 3.0 { 1 } else { 0 }
        }";
    assert_eq!(run_i32(&context, src), 1);
}

/// `Ty::PackLen` (`Dims.len()`) — the other half needed for a *single*
/// `Index<Tensor<T,Dims...>,T>` impl to cover every rank (matching how
/// `Ring<Tensor<T,Dims...>>` already does, see the test above): the impl's
/// own `idx: [i32; Dims.len()]` parameter no longer hardcodes a literal
/// arity, so this one impl is exercised at *two* genuinely different ranks
/// (1 and 2) in the same program — proving `Dims.len()` resolves
/// independently per call site, the same regression this whole pack
/// mechanism already guards for ordinary construction (`a_const_generic_
/// pack_struct_supports_a_different_rank_at_a_different_call_site`).
#[test]
fn a_pack_generic_index_impl_using_dims_len_covers_every_rank_in_one_impl() {
    let context = context();
    let src = "
        use linalg;
        struct Box3<T: Float, const Dims...: i32> { data: [T; Dims...] }
        #[mlir_type(tensor)]
        impl<T: Float, const Dims...: i32> NativeShape<Box3<T, Dims...>> {}
        impl<T: Float, const Dims...: i32> Index<Box3<T, Dims...>, T> {
            fn index(t: Box3<T, Dims...>, idx: [i32; Dims.len()]) -> T { mlir::tensor::extract(t, idx) }
        }
        fn main() -> i32 {
            let v = Box3::<f64, 4>(data: [10.0, 20.0, 30.0, 40.0]);
            let m = Box3::<f64, 2, 2>(data: [[1.0, 2.0], [3.0, 4.0]]);
            let x = v[2] + m[1, 0];
            if x == 33.0 { 1 } else { 0 }
        }";
    assert_eq!(run_i32(&context, src), 1);
}

/// A real, pre-existing bug found migrating `stdlib/linalg/tensor.cleave`
/// onto one pack-generic `Tensor` (not itself about packs): a generic
/// impl's own body constructing a struct via an explicit turbofish that
/// names the *impl's own* enclosing generics (`Tensor::<T,N>(...)` inside
/// `impl<T,const N:i32> Zeroed<Tensor<T,N>>`) used to resolve `T`/`N`
/// against an *empty* generics mapping (`generic_arg_to_ty`'s own doc
/// comment used to describe this as a known, deliberately-unfixed gap) —
/// `T` became a bogus literal type genuinely spelled `"T"`, failing its own
/// `Float` bound with `no impl Float<T>`. Needed for `stdlib/nn/nn.cleave`'s
/// own `Activation<Tensor<T,N>>::relu`, which builds a same-shape zero/one
/// constant this same way. Fixed by resolving through `self.active_
/// generics` instead — the same lookup table `ty_from_ast_mapped`'s other
/// callers already use for "a bare name referencing the enclosing
/// declaration's own generic."
#[test]
fn a_turbofish_inside_a_generic_impl_body_resolves_the_impls_own_enclosing_generics() {
    let context = context();
    let src = "
        struct Box3<T: Float, const N: i32> { data: [T; N] }
        algebra NativeShape<T> {}
        #[mlir_type(tensor)]
        impl<T: Float, const N: i32> NativeShape<Box3<T, N>> {}
        algebra Zeroed<T> { fn zeroed() -> T; }
        impl<T: Float, const N: i32> Zeroed<Box3<T, N>> {
            fn zeroed() -> Box3<T, N> { Box3::<T, N>(data: [0.0; N]) }
        }
        fn main() -> i32 {
            let z: Box3<f64, 3> = zeroed();
            if z.data[0] == 0.0 and z.data[1] == 0.0 and z.data[2] == 0.0 { 1 } else { 0 }
        }";
    assert_eq!(run_i32(&context, src), 1);
}

/// A real, previously-latent dispatch bug, found and fixed building a
/// genuinely composing `Optimizer<Opt, Model>` algebra (`doc/backlog.md`'s
/// own "Optimizer" item): `impl<T: Float+Ring> Optimizer<Sgd, T>` and
/// `impl<Opt> Optimizer<Opt, Pair>` both unify *structurally* against a
/// call needing `Optimizer::step(Sgd_value, Pair_value, ...)` (`T:=Pair`,
/// `Opt:=Sgd`) — only `T`'s own declared `Float+Ring` bound (which `Pair`
/// satisfies neither of) tells them apart. Two genuinely separate call-
/// resolution functions in `monomorphize.rs` shared this exact flaw —
/// `derive_impl_instantiation` (drives this test's own top-level `main()`
/// call) and `find_impl_for_target` (drives the narrower `derivative`-
/// rule/`Ring::zero`/`Index`-seeding paths, not exercised by this
/// particular program) — both used to pick the first *structurally*
/// matching template, in whatever order `templates` happened to iterate,
/// with no regard for whether the match's own bounds actually held; fixed
/// by having both verify each candidate's own `ImplTemplate::generic_
/// bounds` before accepting it, falling through to the next candidate on a
/// bound failure exactly like a structural mismatch already did. Before the
/// fix, this exact program failed deep inside the *wrong* impl's own body
/// (`Ring::sub`/`Scale::scale` asked to specialize for `(Pair,Pair)` —
/// `Pair` was never meant to unify with `Optimizer<Sgd, T>`'s own `T` at
/// all) rather than resolving to the correct, structurally-more-specific
/// `Optimizer<Opt, Pair>`.
#[test]
fn dispatch_between_a_bound_generic_impl_and_an_unbound_but_structurally_specific_one_picks_the_bound_satisfying_candidate()
 {
    let context = context();
    let src = "
        use linalg;
        struct Sgd { lr: f32 }
        impl Scale<f32, f32> { fn scale(c, s) { c * s } }
        algebra Optimizer<Opt, Model> {
            fn step(opt: Opt, model: Model, grad: Model) -> Model;
        }
        impl<T: Float + Ring> Optimizer<Sgd, T> {
            fn step(opt, model, grad) { Ring::sub(model, Scale::scale(grad, opt.lr)) }
        }
        struct Pair { a: f32, b: f32 }
        impl<Opt> Optimizer<Opt, Pair> {
            fn step(opt, model, grad) {
                Pair(a: Optimizer::step(opt, model.a, grad.a), b: Optimizer::step(opt, model.b, grad.b))
            }
        }
        fn main() -> i32 {
            let opt = Sgd(lr: 0.5);
            let p = Pair(a: 1.0, b: 2.0);
            let g = Pair(a: 0.1, b: 0.2);
            let p2: Pair = Optimizer::step(opt, p, g);
            if p2.a > 0.94 and p2.a < 0.96 and p2.b > 1.89 and p2.b < 1.91 { 1 } else { 0 }
        }";
    assert_eq!(run_i32(&context, src), 1);
}

/// `Optimizer<Sgd, Tensor<T,Dims...>>::step` (`stdlib/optim/optim.cleave`)
/// -- one step, exact expected value: `model - lr*grad`. `state` is
/// carried through completely unchanged for `Sgd` (it has none of its own
/// to update) -- checked too, since a `Ring::zero()`-initialized state
/// silently becoming something *else* would be its own real bug.
#[test]
fn optimizer_sgd_step_computes_the_exact_expected_value() {
    let context = context();
    let src = "
        use optim;
        use linalg;
        fn main() -> i32 {
            let opt = Sgd(lr: 0.5);
            let model: Tensor<f32, 3> = Tensor::<f32, 3>(data: [1.0, 2.0, 3.0]);
            let grad: Tensor<f32, 3> = Tensor::<f32, 3>(data: [0.1, 0.2, 0.3]);
            let state: Tensor<f32, 3> = Optimizer::init_state(opt, model);
            let r = Optimizer::step(opt, model, grad, state);
            let m: Tensor<f32, 3> = r.0;
            let s: Tensor<f32, 3> = r.1;
            if m[0] == 0.95 and m[1] == 1.9 and m[2] == 2.85
                and s[0] == 0.0 and s[1] == 0.0 and s[2] == 0.0
            { 1 } else { 0 }
        }";
    assert_eq!(run_i32(&context, src), 1);
}

/// `Optimizer<Momentum, Tensor<T,Dims...>>::step` -- *two* consecutive
/// steps with two different gradients, exact expected values -- the real
/// point being the second step's own expected value only comes out right
/// if `state` (the velocity `v`) was genuinely threaded through from the
/// first step, not silently reset. `v1 = (1-beta)*grad1` (`state0` is
/// zero), `model1 = model0 - lr*v1`; `v2 = beta*v1 + (1-beta)*grad2`,
/// `model2 = model1 - lr*v2` -- computed independently here (`beta=0.9,
/// lr=0.5`): `v1 = 0.01`, `model1 = 0.995`; `v2 = 0.029`, `model2 =
/// 0.9805`. A state-reset bug would instead give `v2 = 0.02`, `model2 =
/// 0.985` -- a different, wrong value this test would catch.
#[test]
fn optimizer_momentum_carries_its_own_state_correctly_across_two_steps() {
    let context = context();
    let src = "
        use optim;
        use linalg;
        fn main() -> i32 {
            let opt = Momentum(lr: 0.5, beta: 0.9);
            let model0: Tensor<f32, 1> = Tensor::<f32, 1>(data: [1.0]);
            let state0: Tensor<f32, 1> = Optimizer::init_state(opt, model0);
            let grad1: Tensor<f32, 1> = Tensor::<f32, 1>(data: [0.1]);
            let r1 = Optimizer::step(opt, model0, grad1, state0);
            let model1: Tensor<f32, 1> = r1.0;
            let state1: Tensor<f32, 1> = r1.1;
            let grad2: Tensor<f32, 1> = Tensor::<f32, 1>(data: [0.2]);
            let r2 = Optimizer::step(opt, model1, grad2, state1);
            let model2: Tensor<f32, 1> = r2.0;
            let ok1 = model1[0] > 0.9949 and model1[0] < 0.9951;
            let ok2 = model2[0] > 0.9804 and model2[0] < 0.9806;
            if ok1 and ok2 { 1 } else { 0 }
        }";
    assert_eq!(run_i32(&context, src), 1);
}

/// `Optimizer<Opt, Dense<T,In,Out>>` (`stdlib/nn/nn.cleave`) -- confirms the
/// one composing impl, generic over `Opt` itself, actually recurses
/// correctly into `.w`/`.b` (through `Sgd`'s own leaf-level `Tensor` impl,
/// already checked in isolation above) -- `lr: 1.0` so `new = old - grad`
/// exactly, easy to check by hand for every element of both fields.
#[test]
fn optimizer_composes_correctly_one_level_up_through_dense() {
    let context = context();
    let src = "
        use nn;
        fn main() -> i32 {
            let opt = Sgd(lr: 1.0);
            let w: Tensor<f32, 2, 2> = Tensor::<f32, 2, 2>(data: [[1.0, 2.0], [3.0, 4.0]]);
            let b: Tensor<f32, 1, 2> = Tensor::<f32, 1, 2>(data: [[5.0, 6.0]]);
            let model: Dense<f32, 2, 2> = Dense(w: w, b: b);
            let gw: Tensor<f32, 2, 2> = Tensor::<f32, 2, 2>(data: [[0.1, 0.1], [0.1, 0.1]]);
            let gb: Tensor<f32, 1, 2> = Tensor::<f32, 1, 2>(data: [[0.1, 0.1]]);
            let grad: Dense<f32, 2, 2> = Dense(w: gw, b: gb);
            let state: Dense<f32, 2, 2> = Optimizer::init_state(opt, model);
            let r = Optimizer::step(opt, model, grad, state);
            let m: Dense<f32, 2, 2> = r.0;
            if m.w[0, 0] > 0.899 and m.w[0, 0] < 0.901
                and m.w[1, 1] > 3.899 and m.w[1, 1] < 3.901
                and m.b[0, 0] > 4.899 and m.b[0, 0] < 4.901
                and m.b[0, 1] > 5.899 and m.b[0, 1] < 5.901
            { 1 } else { 0 }
        }";
    assert_eq!(run_i32(&context, src), 1);
}

/// The real, end-to-end proof: the exact same `Network`/`Dense` shape and
/// training loop `examples/xor_tensor.cleave` itself uses (`Optimizer::
/// step`/`init_state`, one call, no per-field lines) but trained with
/// `Momentum` instead of `Sgd` -- confirms the composing `Dense`/`Network`
/// impls genuinely work for a *second* optimizer with no changes of their
/// own, the actual point of making them generic over `Opt` rather than
/// duplicating one impl per optimizer. Not checked against an exact
/// reference output (unlike `examples/xor_tensor.cleave`'s own `Sgd` run --
/// different optimizer, different trajectory) -- checked for genuine
/// convergence instead, the same `~0,1,1,0` shape, within the same step
/// budget.
#[test]
fn optimizer_momentum_trains_a_real_network_to_convergence() {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(optimizer_momentum_trains_a_real_network_to_convergence_body)
        .unwrap()
        .join()
        .unwrap();
}

fn optimizer_momentum_trains_a_real_network_to_convergence_body() {
    let context = context();
    let src = "
        use io;
        use nn;
        use linalg;
        struct Network { l1: Dense<f32, 2, 2>, l2: Dense<f32, 2, 1> }
        impl<Opt> Optimizer<Opt, Network> {
            fn init_state(opt, model) {
                Network(l1: Optimizer::init_state(opt, model.l1), l2: Optimizer::init_state(opt, model.l2))
            }
            fn step(opt, model, grad, state) {
                let r1 = Optimizer::step(opt, model.l1, grad.l1, state.l1);
                let r2 = Optimizer::step(opt, model.l2, grad.l2, state.l2);
                (Network(l1: r1.0, l2: r2.0), Network(l1: r1.1, l2: r2.1))
            }
        }
        fn forward(x1: f32, x2: f32, net: Network) -> f32 {
            let x = Tensor::<f32, 1, 2>(data: [[x1, x2]]);
            let h = sigmoid(net.l1.forward(x));
            let out = sigmoid(net.l2.forward(h));
            out[0, 0]
        }
        fn loss(x1: f32, x2: f32, y: f32, net: Network) -> f32 {
            let pred = forward(x1, x2, net);
            let err = pred - y;
            err * err
        }
        grad = derive(loss);
        fn main() -> i32 {
            rand_seed(2);
            let mut net: Network = Network(l1: Init::xavier(), l2: Init::xavier());
            let opt = Momentum(lr: 0.5, beta: 0.9);
            let mut state: Network = Optimizer::init_state(opt, net);
            let x1s: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
            let x2s: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
            let ys: [f32; 4] = [0.0, 1.0, 1.0, 0.0];
            for step in 0..4000 {
                let i: i32 = step - (step / 4) * 4;
                let g = grad(x1s[i], x2s[i], ys[i], net);
                let r = Optimizer::step(opt, net, g.3, state);
                net = r.0;
                state = r.1;
            };
            let p00 = forward(0.0, 0.0, net);
            let p01 = forward(0.0, 1.0, net);
            let p10 = forward(1.0, 0.0, net);
            let p11 = forward(1.0, 1.0, net);
            if p00 < 0.5 and p01 > 0.5 and p10 > 0.5 and p11 < 0.5 { 1 } else { 0 }
        }";
    assert_eq!(run_i32(&context, src), 1);
}

/// A real, previously-latent infinite-recursion bug, found and fixed
/// building `stdlib/display/display.cleave`'s own `Display<T>` (`doc/
/// backlog.md`'s own "print générique" item): once *two* structurally-
/// different generic impls of the same algebra both bound generically on
/// that *same* algebra (`Wrap<[T;N]>` and `Wrap<Box3<T>>` below, both
/// `T: Wrap`), calling `wrap` on a value whose own element type is still an
/// unconstrained type variable at declaration-check time hung the compiler
/// forever, a real native stack overflow, not a controlled type error.
///
/// Root cause: `Infer::matching_impls`'s own bound-satisfaction check
/// (`bounds_satisfied`) asked `has_matching_impl` whether a *still fully
/// unconstrained* type variable satisfies a candidate's own declared bound
/// — for a bound naming an algebra with its own further generic, self-
/// bounding impl, that recurses back into `matching_impls` with *another*
/// fresh, equally unconstrained variable, forever. Fixed by treating a
/// not-yet-concrete bound argument as *permissively* satisfied (skip the
/// check entirely, the same "nothing to check yet isn't the same as a
/// check that failed" posture `check_pending_constraints` already
/// establishes one level up) rather than actually searching for a match —
/// `has_matching_impl` is never even called on it.
#[test]
fn two_overlapping_generic_impls_each_bounded_on_their_own_algebra_dont_hang_the_compiler() {
    let context = context();
    let src = "
        algebra Wrap<T> {
            fn wrap(x: T) -> i32;
        }
        impl Wrap<i32> {
            fn wrap(x) { x }
        }
        impl<T: Wrap, const N: i32> Wrap<[T; N]> {
            fn wrap(a) { wrap(a[0]) }
        }
        struct Box3<T> { v: T }
        impl<T: Wrap> Wrap<Box3<T>> {
            fn wrap(b) { wrap(b.v) }
        }
        fn main() -> i32 {
            let b: Box3<i32> = Box3(v: 5);
            wrap(b)
        }";
    assert_eq!(run_i32(&context, src), 5);
}

/// `Display<i32>` (`stdlib/display/display.cleave`) -- exact byte-by-byte
/// check, negative and positive, the two real ways a hand-rolled digit
/// formatter can go wrong (sign handling, digit order). `Display::display`
/// returns the `DynArray<i8>` directly, inspectable in cleave itself before
/// any `print`/stdout involvement -- a more precise assertion than
/// capturing stdout would give.
#[test]
fn display_i32_produces_the_exact_expected_bytes() {
    let context = context();
    let src = "
        use display;
        fn main() -> i32 {
            let mut out: DynArray<i8> = dynarray_new(8);
            out = Display::display(-42, out);
            if out.len() == 3 and out.get(0) == 45 and out.get(1) == 52 and out.get(2) == 50 { 1 } else { 0 }
        }";
    assert_eq!(run_i32(&context, src), 1);
}

/// `Display<f32>` -- exact byte-by-byte check, via the new `format_f32`
/// extern (`cleave-rt`), same `format!(\"{x}\")` `print_f32` itself already
/// uses.
#[test]
fn display_f32_produces_the_exact_expected_bytes() {
    let context = context();
    let src = "
        use display;
        fn main() -> i32 {
            let mut out: DynArray<i8> = dynarray_new(8);
            out = Display::display(3.5, out);
            if out.len() == 3 and out.get(0) == 51 and out.get(1) == 46 and out.get(2) == 53 { 1 } else { 0 }
        }";
    assert_eq!(run_i32(&context, src), 1);
}

/// `Display<[T;N]>` -- exact bracketed, comma-separated rendering,
/// `\"[1, 2, 3]\"` byte for byte, confirming the array-of-scalars
/// composition (`Display::display` recursing into `Display<i32>` per
/// element) works correctly -- the actual real-world shape `print([1,2,3])`
/// uses.
#[test]
fn display_of_an_int_array_renders_brackets_and_commas_correctly() {
    let context = context();
    let src = r#"
        use display;
        fn main() -> i32 {
            let mut out: DynArray<i8> = dynarray_new(16);
            out = Display::display([1, 2, 3], out);
            let expected: [i8; 9] = [91, 49, 44, 32, 50, 44, 32, 51, 93]; // "[1, 2, 3]"
            let mut ok: i32 = 1;
            if out.len() != 9 { ok = 0; };
            for i in 0..9 {
                if out.get(i) != expected[i] { ok = 0; };
            };
            ok
        }"#;
    assert_eq!(run_i32(&context, src), 1);
}

/// A real, previously-latent type-inference bug, found and fixed building
/// `Display<Tensor<T,Dims...>>` (`stdlib/display/display.cleave`, `doc/
/// backlog.md`'s own "print générique" item): `t[0]` (`Index<Tensor<T,N>,
/// T>`, an *algebra* dispatch, unlike an ordinary array's direct `PrimOp::
/// Load`) correctly *defers* while `T` is still abstract mid-declaration
/// of an enclosing generic impl -- but deferring used to leave `Index`'s
/// own output-only `Elem` generic (`t[0]`'s own type) as a genuinely
/// orphaned fresh variable, never unified with `T` at all, silently
/// defaulted to `i32` by the time a *second* dispatch (`wrap` again, here)
/// needed a concrete answer -- confirmed directly via `--dump-cps-
/// optimized`: `Wrap::wrap<Tensor<f32,3>>` itself calling `Wrap::
/// wrap<i32>`, not `<f32>`, before the fix.
///
/// Fixed in `Infer::infer_algebra_call` (`cleave/src/infer.rs`): committing
/// a deferred dispatch immediately is still sound whenever exactly one impl
/// matches *and* every position it would newly pin down resolves only to
/// variables `self.active_generics` already owns (`Elem := T`, the same
/// openness carried forward under a new name, no new guess) -- see
/// `Infer::unambiguous_and_preserves_openness`'s own doc comment for the
/// full reasoning, including why this must *not* fire for `Convert<From,
/// To>`-shaped cases (`stdlib/nn/nn.cleave`'s own `mean`, `N.to()`), which
/// still correctly defer.
#[test]
fn tensor_index_result_dispatches_to_the_correct_algebra_impl_even_while_still_generic() {
    let context = context();
    let src = "
        use linalg;
        algebra Wrap<T> {
            fn wrap(x: T) -> i32;
        }
        impl Wrap<i32> {
            fn wrap(x) { x }
        }
        impl Wrap<f32> {
            fn wrap(x) { 999 }
        }
        impl<T: Wrap, const N: i32> Wrap<Tensor<T, N>> {
            fn wrap(t) { Wrap::wrap(t[0]) }
        }
        fn main() -> i32 {
            let v: Tensor<f32, 3> = Tensor::<f32, 3>(data: [1.0, 2.0, 3.0]);
            Wrap::wrap(v)
        }";
    assert_eq!(run_i32(&context, src), 999);
}

/// A self-recursive generic top-level `fn`, called at two different
/// concrete types in the same program, alongside an algebra `impl` whose
/// own body triggers a *deferred-but-immediately-committed* dispatch (the
/// same `Wrap<Tensor<T,N>>` shape as the test just above). Each top-level
/// `fn` is inferred in its own group by `callgraph::infer_program`
/// (`cleave/src/callgraph.rs`), one `Infer` instance per group, each
/// restarting its own `TyVar` numbering at 0 -- so a later group's brand
/// new generic parameter can numerically collide with some unrelated, still
/// -free variable left over in an earlier group's already-finished
/// `scheme.ty`. `Infer::generalize`'s own `env_fv` check (run against
/// `global_env`, accumulated across groups) doesn't know these two `0`s
/// aren't the same variable -- found for real via `examples/fibonacci.
/// cleave`: it silently generalized `fibonacci<T: Int>` as monomorphic
/// (locked to whichever concrete type its *first* call site used) instead
/// of polymorphic, so the *second* call site failed to unify. Fixed by
/// threading one monotonic `TyVar` counter across the whole group loop
/// (`Infer::with_var_counter_starting_at`/`var_counter`) instead of letting
/// each group's own `Infer::new` restart at 0. This test doesn't even need
/// `Wrap`'s own dispatch to *matter* semantically -- its only job is to
/// shift enough extra top-level declarations/groups through the loop
/// before `fibonacci`'s own group to make a collision likely, the same way
/// `use display;`/`use linalg;` incidentally did in the original report.
#[test]
fn a_self_recursive_generic_fn_called_at_two_concrete_types_stays_polymorphic_even_with_other_algebra_impls_present()
 {
    let context = context();
    let src = "
        use display;
        algebra Wrap<T> {
            fn wrap(x: T) -> i32;
        }
        impl Wrap<i32> {
            fn wrap(x) { x }
        }
        impl Wrap<f32> {
            fn wrap(x) { 999 }
        }
        impl<T: Wrap, const N: i32> Wrap<Tensor<T, N>> {
            fn wrap(t) { Wrap::wrap(t[0]) }
        }
        fn fibonacci<T: Int>(x: T) -> T {
            if x <= 1 { x } else { fibonacci(x-1) + fibonacci(x-2) }
        }
        fn main() -> i32 {
            let a: i32 = fibonacci(16:i32);
            let b: i64 = fibonacci(42:i64);
            if b == 267914296:i64 { a } else { -1 }
        }";
    assert_eq!(run_i32(&context, src), 987);
}

/// A pack-generic top-level `fn` (`const Dims...: i32`), called at two
/// different *ranks* in the same program — `random_fill::<f32,[4]>` then
/// `random_fill::<f32,[2,2,2]>`. Broke the same way the fibonacci case just
/// above did, but for a genuinely different underlying reason: `Infer::
/// instantiate_with_mapping` (`cleave/src/infer.rs`) minted *every* fresh
/// replacement for a generalized scheme's own `scheme.vars` as a bare `Ty::
/// Var` (`TyVarGen::fresh` always returns one) — silently downgrading a
/// pack-kind generic (`Ty::Pack`, what `const Dims...: i32` actually mints,
/// `Infer::fresh_vars_for_generics`) into an ordinary scalar var on *every*
/// re-instantiation. A single call site still happened to work (nothing yet
/// forced the corrupted var back into pack shape until unification against
/// a concrete `Tensor<_, N>` did it structurally), but a second call site
/// at a different rank got its own fresh-but-still-scalar var with no
/// pack-ness left to recover, escaping monomorphization as a bare
/// unresolved type variable. Fixed by recovering each `scheme.vars` entry's
/// real kind from `scheme.ty` itself (`collect_pack_vars`, scanning for
/// `Ty::Pack(v)` occurrences) before minting its fresh replacement.
#[test]
fn a_pack_generic_top_level_fn_called_at_two_different_ranks_monomorphizes_both() {
    let context = context();
    let src = "
        use linalg;
        fn fill<T: Ring, const Dims...: i32>(v: T) -> Tensor<T, Dims...> {
            Tensor::<T, Dims...>(data: [v; Dims...])
        }
        fn main() -> i32 {
            let a: Tensor<f32, 4> = fill(2.0);
            let b: Tensor<f32, 2, 2, 2> = fill(3.0);
            let mut ok: i32 = 1;
            for i in 0..4 { if a[i] != 2.0 { ok = 0; }; };
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        if b[i, j, k] != 3.0 { ok = 0; };
                    };
                };
            };
            ok
        }";
    assert_eq!(run_i32(&context, src), 1);
}

/// The algebra-impl counterpart of the test just above — same pack-generic-
/// called-at-two-ranks shape, but through `impl<T, const Dims...: i32>
/// SomeAlgebra<Tensor<T,Dims...>>` rather than a plain top-level `fn`, and
/// with the impl's own body calling into a *second* algebra (`Rand::
/// uniform`, via `Rand<T>`) rather than just constructing directly.
/// Previously didn't fail cleanly — it hung the compiler outright (an
/// unbounded stack overflow), isolated at the time to needing *both* two
/// ranks in one program *and* a nested cross-algebra call inside the pack-
/// generic impl body (dropping either condition alone worked fine). Fixed
/// by the exact same root-cause fix as the top-level-`fn` case above — this
/// path also goes through `Infer::instantiate_with_mapping` (an algebra
/// impl's own generalized method scheme gets re-instantiated per call site
/// exactly like a top-level `fn`'s does), so the corrupted-pack-var
/// mechanism was identical underneath, just reached through a different
/// front door. This test's own real job is simply completing at all,
/// within the harness's ordinary timeout, rather than hanging.
#[test]
fn an_algebra_impl_generic_over_a_pack_calling_a_second_algebra_works_at_two_ranks_in_one_program() {
    let context = context();
    let src = "
        use linalg;
        use rand;
        algebra RandomFilled<T> { fn random_fill() -> T; }
        impl<T: Float + Rand + Ring, const Dims...: i32> RandomFilled<Tensor<T, Dims...>> {
            fn random_fill() { Tensor::<T, Dims...>(data: [Rand::uniform(-1.0, 1.0); Dims...]) }
        }
        fn main() -> i32 {
            let v: Tensor<f32, 4> = RandomFilled::random_fill();
            let t: Tensor<f32, 2, 2, 2> = RandomFilled::random_fill();
            0
        }";
    assert_eq!(run_i32(&context, src), 0);
}

/// `Display<Tensor<T,N>>` (rank 1) -- exact byte-for-byte rendering,
/// fractional values specifically (an integer-valued `f32` like `1.0`
/// prints as `"1"`, not `"1.0"` -- Rust's own `Display` for floats omits a
/// trailing `.0`, the *same* `print_f32` already does -- so this wouldn't
/// actually distinguish correct `f32` dispatch from the old, wrong `i32`
/// default; a fractional value does).
#[test]
fn display_of_a_rank_1_tensor_renders_correctly() {
    let context = context();
    let src = r#"
        use display;
        use linalg;
        fn main() -> i32 {
            let t: Tensor<f32, 2> = Tensor::<f32, 2>(data: [1.5, 2.5]);
            let mut out: DynArray<i8> = dynarray_new(16);
            out = Display::display(t, out);
            let expected: [i8; 10] = [91, 49, 46, 53, 44, 32, 50, 46, 53, 93]; // "[1.5, 2.5]"
            let mut ok: i32 = 1;
            if out.len() != 10 { ok = 0; };
            for i in 0..10 {
                if out.get(i) != expected[i] { ok = 0; };
            };
            ok
        }"#;
    assert_eq!(run_i32(&context, src), 1);
}

/// `Display<Tensor<T,R,C>>` (rank 2) -- exact byte-for-byte rendering,
/// nested brackets -- `print(net.l1.w)`, this whole item's own real
/// motivating case.
#[test]
fn display_of_a_rank_2_tensor_renders_correctly() {
    let context = context();
    let src = r#"
        use display;
        use linalg;
        fn main() -> i32 {
            let t: Tensor<f32, 2, 2> = Tensor::<f32, 2, 2>(data: [[1.5, 2.5], [3.5, 4.5]]);
            let mut out: DynArray<i8> = dynarray_new(32);
            out = Display::display(t, out);
            // "[[1.5, 2.5], [3.5, 4.5]]"
            let expected: [i8; 24] = [
                91, 91, 49, 46, 53, 44, 32, 50, 46, 53, 93, 44, 32,
                91, 51, 46, 53, 44, 32, 52, 46, 53, 93, 93
            ];
            let mut ok: i32 = 1;
            if out.len() != 24 { ok = 0; };
            for i in 0..24 {
                if out.get(i) != expected[i] { ok = 0; };
            };
            ok
        }"#;
    assert_eq!(run_i32(&context, src), 1);
}

/// Real, end-to-end exercise of `region_analysis`/`cleave_alloc_local`/
/// `cleave_region_enter`/`cleave_region_exit` wired together
/// (`mlir_lower.rs::alloc_llvm_value`/`lower_loop`) -- `make_boxed`'s own
/// result (`b`) is read once, through a field projection (`b.v`), by an
/// ordinary arithmetic op feeding the loop's own carried `acc`; `b` itself
/// (and the `Boxed` struct it names) never reaches the carried state, so
/// `region_analysis::find_region_local_functions` should mark `make_boxed`
/// region-local -- meaning every one of its own 5 iterations' worth of
/// `Boxed` construction draws from the *same* arena slot, reused after each
/// `cleave_region_exit` rewinds the cursor back. A wrong wiring here (a
/// dominance bug, a premature rewind, a wrong allocator choice) would show
/// up as a wrong sum, not just a crash -- the decisive check.
#[test]
fn a_loop_calling_a_region_local_struct_returning_function_computes_correctly() {
    let context = context();
    let src = r#"
        struct Boxed { v: i32 }

        fn make_boxed(x: i32) -> Boxed { Boxed(v: x * 2) }

        fn main() -> i32 {
            let mut acc: i32 = 0;
            for i in 0..5 {
                let b = make_boxed(i);
                acc = acc + b.v;
            };
            acc
        }
        "#;
    // sum of 2*i for i in 0..5 = 2*(0+1+2+3+4) = 20.
    assert_eq!(run_i32(&context, src), 20);
}

/// The same shape as above, but with a *nested* struct field (`b.inner.v`)
/// and a *nested* loop (an outer loop running the inner one twice) -- two
/// real generalizations worth checking directly, not assumed to follow from
/// the flat, single-loop case: field-projection tracing through more than
/// one hop, and `REGION_DEPTH` genuinely nesting (an inner loop's own
/// `region_exit` must not close the *outer* loop's own still-open region).
#[test]
fn region_locality_holds_with_nested_fields_and_a_nested_loop() {
    let context = context();
    let src = r#"
        struct Inner { v: i32 }
        struct Outer { inner: Inner }

        fn make_outer(x: i32) -> Outer { Outer(inner: Inner(v: x * 3)) }

        fn main() -> i32 {
            let mut total: i32 = 0;
            for _pass in 0..2 {
                let mut acc: i32 = 0;
                for i in 0..4 {
                    let o = make_outer(i);
                    acc = acc + o.inner.v;
                };
                total = total + acc;
            };
            total
        }
        "#;
    // Inner sum: 3*(0+1+2+3) = 18, run twice by the outer loop -> 36.
    assert_eq!(run_i32(&context, src), 36);
}

// A region-local branch for `Tensor`-typed construction (`lower_tagged_
// struct_construct`, closing the real gap that `load_train_input`/`load_
// train_target` were marked region-local but never actually reached
// `cleave_alloc_local`) was tried here and reverted, not kept -- measured
// directly (VTune, `examples/mnist-interop`) to reintroduce ~63s of real
// `memcpy` traffic: the hand-built memref it produced was opaque to One-
// Shot Bufferize's own alias analysis in a way `tensor.from_elements`
// itself isn't, so bufferization could no longer prove downstream reads
// were copy-free the way it can for the ordinary path. The struct-boundary
// half of the region-local mechanism (the two tests above) still stands.
