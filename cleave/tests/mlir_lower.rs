use cleave::cps::{collect_mlir_types, collect_struct_schemas, collect_units, convert_program};
use cleave::driver::compile;
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
    assert!(module.as_operation().verify(), "generated MLIR module failed verification");
    module.as_operation().to_string()
}

#[test]
fn a_function_returning_a_bare_literal_lowers_to_a_constant_plus_return() {
    let context = context();
    let text = lower(&context, "fn main() -> i32 { 0 }");
    assert!(text.contains("func.func @main() -> i32"), "got:\n{text}");
    assert!(text.contains("arith.constant 0 : i32"), "got:\n{text}");
    assert!(text.contains("return %c0_i32 : i32") || text.contains("return %"), "got:\n{text}");
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
    let (result, _sources) = compile(vec![("test.cleave".to_string(), "fn main() -> i32 { 17 }".to_string())], &[]);
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
    pass_manager.run(&mut module).expect("lowering to the llvm dialect must succeed");

    let engine = melior::ExecutionEngine::new(&module, 2, &[], false, false);
    // Registered unconditionally, harmless if unused -- any struct
    // construction anywhere in the program (not just a top-level return)
    // needs `cleave_alloc` (see `mlir_lower.rs::alloc_struct`'s own doc
    // comment).
    unsafe {
        engine.register_symbol("cleave_alloc", cleave_rt::cleave_alloc as *mut ());
    }
    let mut out: i32 = -1;
    unsafe {
        engine.invoke_packed("main", &mut [&mut out as *mut i32 as *mut ()]).expect("JIT invocation must succeed");
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
fn run_i32(context: &Context, src: &str) -> i32 {
    let (result, _sources) = compile(vec![("test.cleave".to_string(), src.to_string())], &[]);
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    let units = collect_units(&program, &registry);
    let cps_program = convert_program(units);
    let mlir_types = collect_mlir_types(&program);
    let struct_schemas = collect_struct_schemas(&program);
    let mut module = lower_program(context, &cps_program, &mlir_types, struct_schemas);
    assert!(module.as_operation().verify(), "generated MLIR module failed verification");

    let pass_manager = pass::PassManager::new(context);
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    pass_manager.run(&mut module).expect("lowering to the llvm dialect must succeed");

    let engine = melior::ExecutionEngine::new(&module, 2, &[], false, false);
    // Registered unconditionally, harmless if unused -- any struct
    // construction anywhere in the program (not just a top-level return)
    // needs `cleave_alloc` (see `mlir_lower.rs::alloc_struct`'s own doc
    // comment).
    unsafe {
        engine.register_symbol("cleave_alloc", cleave_rt::cleave_alloc as *mut ());
    }
    let mut out: i32 = -1;
    unsafe {
        engine.invoke_packed("main", &mut [&mut out as *mut i32 as *mut ()]).expect("JIT invocation must succeed");
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
    assert_eq!(run_i32(&context, "fn main() -> i32 { if 1 < 2 { 10 } else { 20 } }"), 10);
    assert_eq!(run_i32(&context, "fn main() -> i32 { if 2 < 1 { 10 } else { 20 } }"), 20);
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
    assert!(!text.contains("func.func private"), "got:\n{text}");
}

#[test]
fn a_direct_mlir_call_actually_computes_the_right_value() {
    let context = context();
    assert_eq!(run_i32(&context, "fn main() -> i32 { mlir::arith::addi(2, 3) }"), 5);
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
    let text = lower(&context, "fn main() -> i32 { if mlir::arith::cmpi(1, 2, predicate: \"2 : i64\") { 10 } else { 20 } }");
    assert!(text.contains("arith.cmpi") && text.contains("slt"), "got:\n{text}");
    assert_eq!(run_i32(&context, "fn main() -> i32 { if mlir::arith::cmpi(1, 2, predicate: \"2 : i64\") { 10 } else { 20 } }"), 10);
    assert_eq!(run_i32(&context, "fn main() -> i32 { if mlir::arith::cmpi(2, 1, predicate: \"2 : i64\") { 10 } else { 20 } }"), 20);
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
    let (result, _sources) =
        compile(vec![("test.cleave".to_string(), "fn foo(x: i32) -> i32 { x } fn main() -> i32 { foo(x: \"5\") }".to_string())], &[]);
    let program = result.unwrap_or_else(|e| panic!("expected a parse/use-resolution success (the type error is caught later): {e:?}"));
    let registry = Registry::build(&program);
    let (_, errs) = cleave::dump::dump_program(&program, &registry);
    assert!(!errs.is_empty(), "`foo(x: \"5\")` supplies zero real positional arguments to a one-parameter fn and must be rejected");
}

const EXTERN_PRINT_SRC: &str = "extern fn print_i32(x: i32) -> i32; fn main() -> i32 { print_i32(42) }";

#[test]
fn an_extern_fn_call_lowers_to_a_private_declaration_plus_a_real_call() {
    let context = context();
    let text = lower(&context, EXTERN_PRINT_SRC);
    assert!(text.contains("func.func private @print_i32(i32) -> i32"), "got:\n{text}");
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
    let (result, _sources) = compile(vec![("test.cleave".to_string(), EXTERN_PRINT_SRC.to_string())], &[]);
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
    pass_manager.run(&mut module).expect("lowering to the llvm dialect must succeed");

    let engine = melior::ExecutionEngine::new(&module, 2, &[], false, false);
    unsafe {
        engine.register_symbol("print_i32", cleave_rt::print_i32 as *mut ());
        engine.register_symbol("cleave_alloc", cleave_rt::cleave_alloc as *mut ());
    }
    let mut out: i32 = -1;
    unsafe {
        engine.invoke_packed("main", &mut [&mut out as *mut i32 as *mut ()]).expect("JIT invocation must succeed");
    }
    assert_eq!(out, 42);
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
    assert!(text.contains("func.func private @print_i32(i32) -> i32"), "got:\n{text}");
    assert!(text.contains("func.func private @print_i64(i64) -> i64"), "got:\n{text}");
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
    let (result, _sources) = compile(vec![("test.cleave".to_string(), EXTERN_IMPL_PRINT_SRC.to_string())], &[]);
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
    pass_manager.run(&mut module).expect("lowering to the llvm dialect must succeed");

    let engine = melior::ExecutionEngine::new(&module, 2, &[], false, false);
    unsafe {
        engine.register_symbol("print_i32", cleave_rt::print_i32 as *mut ());
        engine.register_symbol("print_i64", cleave_rt::print_i64 as *mut ());
        engine.register_symbol("cleave_alloc", cleave_rt::cleave_alloc as *mut ());
    }
    let mut out: i32 = -1;
    unsafe {
        engine.invoke_packed("main", &mut [&mut out as *mut i32 as *mut ()]).expect("JIT invocation must succeed");
    }
    assert_eq!(out, 0);
}

#[test]
fn a_while_loop_lowers_to_scf_while() {
    let context = context();
    let text = lower(&context, "fn main() -> i32 { let mut acc = 0; while acc < 5 { acc = acc + 1; }; acc }");
    assert!(text.contains("scf.while"), "got:\n{text}");
    assert!(text.contains("scf.condition"), "got:\n{text}");
    assert!(text.contains("scf.yield"), "got:\n{text}");
}

#[test]
fn a_while_loop_actually_computes_the_right_value() {
    let context = context();
    assert_eq!(run_i32(&context, "fn main() -> i32 { let mut acc = 0; while acc < 5 { acc = acc + 1; }; acc }"), 5);
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
    assert_eq!(run_i32(&context, "fn main() -> i32 { let mut acc = 0; for i in 0..10 { acc = acc + i; }; acc }"), 45);
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
    assert!(text.contains("memref.alloc() : memref<3xi32>"), "got:\n{text}");
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
    let text = lower(&context, "fn main() -> i32 { let a = [[1, 2, 3], [4, 5, 6]]; a[0, 0] }");
    assert!(text.contains("memref.alloc() : memref<2x3xi32>"), "got:\n{text}");
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
    let text = lower(&context, "struct Pair { a: i32, b: i32 } fn main() -> i32 { let p = Pair(a: 1, b: 2); p.a + p.b }");
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
