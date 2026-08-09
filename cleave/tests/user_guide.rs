//! One test per runnable example in `doc/user_guide.md` — not a substitute
//! for the doc's own prose (code is deliberately duplicated here rather than
//! extracted from the markdown: the guide optimizes for pedagogical clarity,
//! this file for precise assertions, and there's no markdown-to-cleave-test
//! extraction mechanism the way `rustdoc --test` exists for Rust code in doc
//! comments). Serves two purposes: fast, `cargo test`-speed verification
//! while the guide is being written/edited (rather than a `cargo run`
//! subprocess per example), and a lasting regression suite — if a language
//! change breaks one of the guide's own examples, this catches it.
//!
//! Every test actually JIT-executes (`run_i32`, mirroring `cleave/tests/
//! mlir_lower.rs`'s own helper) and asserts a real, non-coincidental return
//! value — matching this whole project's own "verified by running it, not
//! just by type-checking" discipline.

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

fn build_module<'c>(context: &'c Context, src: &str) -> melior::ir::Module<'c> {
    let (result, _sources) = compile(vec![("test.cleave".to_string(), src.to_string())], &[]);
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    let units = collect_units(&program, &registry);
    let cps_program = convert_program(units);
    let mlir_types = collect_mlir_types(&program);
    let struct_schemas = collect_struct_schemas(&program);
    let module = lower_program(context, &cps_program, &mlir_types, struct_schemas);
    assert!(module.as_operation().verify(), "generated MLIR module failed verification");
    module
}

/// Lowers `src` to the `llvm` dialect and JIT-invokes its `main`, returning
/// the result. `scf.if` (and any other structured-control-flow op) has no
/// direct LLVM IR translation of its own -- `create_scf_to_control_flow`
/// lowers it to the `cf` dialect's ordinary branches first, which `create_
/// to_llvm` *does* know how to translate.
fn run_i32(context: &Context, src: &str) -> i32 {
    let mut module = build_module(context, src);

    let pass_manager = pass::PassManager::new(context);
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    pass_manager.run(&mut module).expect("lowering to the llvm dialect must succeed");

    let engine = melior::ExecutionEngine::new(&module, 2, &[], false, false);
    // Registered unconditionally, harmless if unused -- any struct
    // construction anywhere in the program needs `cleave_alloc` (see
    // `mlir_lower.rs::alloc_struct`'s own doc comment).
    unsafe {
        engine.register_symbol("cleave_alloc", cleave_rt::cleave_alloc as *mut ());
        engine.register_symbol("print_i8", cleave_rt::print_i8 as *mut ());
        engine.register_symbol("print_i16", cleave_rt::print_i16 as *mut ());
        engine.register_symbol("print_i32", cleave_rt::print_i32 as *mut ());
        engine.register_symbol("print_i64", cleave_rt::print_i64 as *mut ());
        engine.register_symbol("print_f32", cleave_rt::print_f32 as *mut ());
        engine.register_symbol("print_f64", cleave_rt::print_f64 as *mut ());
    }
    let mut out: i32 = -1;
    unsafe {
        engine.invoke_packed("main", &mut [&mut out as *mut i32 as *mut ()]).expect("JIT invocation must succeed");
    }
    out
}

// ---------------------------------------------------------------- Hello, cleave

#[test]
fn hello_cleave() {
    let context = context();
    assert_eq!(run_i32(&context, "fn main() -> i32 { 42 }"), 42);
}

// ---------------------------------------------------------------- Arithmetic

#[test]
fn arithmetic_on_primitive_types_just_works() {
    let context = context();
    assert_eq!(run_i32(&context, "fn add_one(x: i32) -> i32 { x + 1 } fn main() -> i32 { add_one(5) }"), 6);
}

// ---------------------------------------------------------------- Bindings

#[test]
fn let_and_let_mut() {
    let context = context();
    let src = "
        fn f() -> i32 {
            let a = 1;
            let mut b = 2;
            b = b + a;
            b
        }
        fn main() -> i32 { f() }
    ";
    assert_eq!(run_i32(&context, src), 3);
}

// ---------------------------------------------------------------- Functions

#[test]
fn unannotated_function_infers_a_polymorphic_type() {
    let context = context();
    assert_eq!(run_i32(&context, "fn add_one(x) { x + 1 } fn main() -> i32 { add_one(5) }"), 6);
}

#[test]
fn annotated_function_signature() {
    let context = context();
    assert_eq!(run_i32(&context, "fn add_one(x: i32) -> i32 { x + 1 } fn main() -> i32 { add_one(5) }"), 6);
}

#[test]
fn mutual_recursion_any_declaration_order() {
    let context = context();
    let src = "
        fn is_even(n: i32) -> bool {
            if n == 0 { true } else { is_odd(n - 1) }
        }
        fn is_odd(n: i32) -> bool {
            if n == 0 { false } else { is_even(n - 1) }
        }
        fn main() -> i32 { if is_even(4) { 1 } else { 0 } }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

// ---------------------------------------------------------------- Control flow

#[test]
fn if_else_is_an_expression() {
    let context = context();
    let src = "
        fn abs(x: i32) -> i32 {
            if x < 0 { -x } else { x }
        }
        fn main() -> i32 { abs(-3) }
    ";
    assert_eq!(run_i32(&context, src), 3);
}

#[test]
fn for_loop_accumulator() {
    let context = context();
    let src = "
        fn sum_to(n: i32) -> i32 {
            let mut total = 0;
            for i in 0..n {
                total = total + i;
            };
            total
        }
        fn main() -> i32 { sum_to(5) }
    ";
    assert_eq!(run_i32(&context, src), 10);
}

#[test]
fn boolean_logic_and_or_xor_implies_not() {
    let context = context();
    let src = "
        fn main() -> i32 {
            let a = true;
            let b = false;
            if (a and not b) implies (a or b) { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

// ---------------------------------------------------------------- Structs

#[test]
fn struct_construction_and_field_access() {
    let context = context();
    let src = "
        struct Vec2 { x: f64, y: f64 }
        fn magnitude_sq(v: Vec2) -> f64 { v.x * v.x + v.y * v.y }
        fn main() -> i32 {
            if magnitude_sq(Vec2(x: 3.0, y: 4.0)) == 25.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

#[test]
fn struct_field_mutation() {
    let context = context();
    let src = "
        struct Vec2 { x: f64, y: f64 }
        fn main() -> i32 {
            let mut v = Vec2(x: 1.0, y: 2.0);
            v.x = 10.0;
            if v.x + v.y == 12.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

// ---------------------------------------------------------------- Arrays

#[test]
fn array_literal_index_and_mutation() {
    let context = context();
    let src = "
        fn main() -> i32 {
            let mut a = [1, 2, 3];
            a[0] = 10;
            a[0] + a[1] + a[2]
        }
    ";
    assert_eq!(run_i32(&context, src), 15);
}

#[test]
fn multi_dimensional_array_indexing() {
    let context = context();
    let src = "
        fn main() -> i32 {
            let mut grid = [[1, 2, 3], [4, 5, 6]];
            grid[1, 2] = 60;
            grid[0, 0] + grid[1, 2]
        }
    ";
    assert_eq!(run_i32(&context, src), 61);
}

// ---------------------------------------------------------------- Algebras

#[test]
fn algebras_how_operators_actually_work() {
    let context = context();
    let src = "
        struct Vec2 { x: f64, y: f64 }
        impl Ring<Vec2> {
            fn add(a, b) { Vec2(x: a.x + b.x, y: a.y + b.y) }
        }
        fn translate(a: Vec2, b: Vec2) -> Vec2 {
            a + b
        }
        fn main() -> i32 {
            let r = translate(Vec2(x: 1.0, y: 2.0), Vec2(x: 3.0, y: 4.0));
            if r.x == 4.0 and r.y == 6.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

// ---------------------------------------------------------------- Inherent impls

/// `doc/user_guide.md`'s own "Inherent impls" example, run for real via
/// dot-call syntax (`v.magnitude_sq()`) -- previously caveated as "type-
/// checks but can't be JIT-executed yet" (`cps.rs` had no `ExprKind::
/// MethodCall` conversion arm at all; `doc/backlog.md`'s own item 7).
#[test]
fn inherent_impl_method_computes_the_right_value_via_dot_syntax() {
    let context = context();
    let src = "
        struct Vec2 { x: f64, y: f64 }
        impl struct Vec2 {
            fn magnitude_sq(v) -> f64 { v.x * v.x + v.y * v.y }
        }
        fn main() -> i32 {
            if Vec2(x: 1.0, y: 2.0).magnitude_sq() == 5.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

// ---------------------------------------------------------------- Generics

#[test]
fn generic_struct_field_type_inferred() {
    let context = context();
    let src = "
        struct Pair<T> { a: T, b: T }
        fn f() -> Pair<f64> {
            Pair(a: 1.0, b: 2.0)
        }
        fn main() -> i32 {
            if f().a == 1.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

#[test]
fn let_polymorphism_reuses_a_generic_function_at_two_types() {
    let context = context();
    let src = "
        fn identity(x) { x }
        fn g() -> i32 {
            let a = identity(1);
            let b = identity(1.5);
            if b > 1.0 { a } else { 0 }
        }
        fn main() -> i32 { g() }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

#[test]
fn bounds_restrict_a_generic_to_types_with_the_right_algebra() {
    let context = context();
    let src = "
        fn smaller<T: Ord>(a: T, b: T) -> T {
            if a < b { a } else { b }
        }
        fn main() -> i32 { smaller(1, 2) }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

// ---------------------------------------------------------------- Const-generics

#[test]
fn const_generic_array_field() {
    let context = context();
    let src = "
        struct Vector<T, const N: i32> { data: [T; N] }
        fn f() -> f64 {
            let v = Vector::<f64, 3>(data: [1.0, 2.0, 3.0]);
            v.data[0] + v.data[1] + v.data[2]
        }
        fn main() -> i32 {
            if f() == 6.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

// ---------------------------------------------------------------- Turbofish

#[test]
fn turbofish_pins_an_otherwise_uninferrable_generic_argument() {
    let context = context();
    let src = "
        struct Vector<T, const N: i32> { data: [T; N] }
        fn f() -> f64 {
            let v = Vector::<f64, 3>(data: [1.0, 2.0, 3.0]);
            v.data[0]
        }
        fn main() -> i32 {
            if f() == 1.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

// ---------------------------------------------------------------- Heterogeneous algebras / matmul

#[test]
fn heterogeneous_algebra_matrix_multiplication() {
    let context = context();
    let src = "
        algebra MatMul<A, B, C> {
            fn matmul(a: A, b: B) -> C;
        }

        struct Matrix<T: Float, const R: i32, const C: i32> {
            values: [T; R, C],
        }

        impl<T: Float, const N: i32, const M: i32, const K: i32>
            MatMul<Matrix<T,N,M>, Matrix<T,M,K>, Matrix<T,N,K>> {
            fn matmul(a, b) {
                let mut result = Matrix(values: [[0.0; K]; N]);
                for i in 0..N {
                    for j in 0..K {
                        let mut sum = 0.0;
                        for k in 0..M {
                            sum = sum + a.values[i,k] * b.values[k,j];
                        };
                        result.values[i,j] = sum;
                    };
                };
                result
            }
        }

        fn main() -> i32 {
            let a = Matrix::<f32, 2, 2>(values: [[1.0, 2.0], [3.0, 4.0]]);
            let b = Matrix::<f32, 2, 2>(values: [[5.0, 6.0], [7.0, 8.0]]);
            let c = matmul(a, b);
            if c.values[0,0] == 19.0 and c.values[0,1] == 22.0 and c.values[1,0] == 43.0 and c.values[1,1] == 50.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

// ---------------------------------------------------------------- Higher-order functions

#[test]
fn higher_order_function_computes_the_right_value() {
    let context = context();
    let src = "
        fn apply(f: (i32) -> i32, x: i32) -> i32 {
            f(x)
        }
        fn g() -> i32 {
            let inc = fn(x) { x + 1 };
            apply(inc, 5)
        }
        fn main() -> i32 { g() }
    ";
    assert_eq!(run_i32(&context, src), 6);
}

// ---------------------------------------------------------------- extern fn / print

#[test]
fn extern_fn_print_returns_its_argument_unchanged() {
    let context = context();
    let src = "
        use io;
        fn main() -> i32 {
            print(42)
        }
    ";
    assert_eq!(run_i32(&context, src), 42);
}

// ---------------------------------------------------------------- Type inference and defaulting

#[test]
fn unsuffixed_literal_defaults_to_i32() {
    let context = context();
    assert_eq!(run_i32(&context, "fn f() -> i32 { 1 } fn main() -> i32 { f() }"), 1);
}

#[test]
fn float_literal_needs_a_dot() {
    let context = context();
    let src = "fn h() -> f64 { 1.0 } fn main() -> i32 { if h() == 1.0 { 1 } else { 0 } }";
    assert_eq!(run_i32(&context, src), 1);
}

// ---------------------------------------------------------------- Putting it together

#[test]
fn putting_it_together_worked_example() {
    let context = context();
    let src = "
        struct Vec2 {
            x: f64,
            y: f64,
        }

        impl Ring<Vec2> {
            fn add(a, b) { Vec2(x: a.x + b.x, y: a.y + b.y) }
        }

        fn magnitude_sq(v: Vec2) -> f64 { v.x * v.x + v.y * v.y }

        fn combine<T: Ring>(a: T, b: T) -> T {
            a + b
        }

        fn main() -> i32 {
            let a = Vec2(x: 1.0, y: 2.0);
            let b = Vec2(x: 3.0, y: 4.0);
            let c = combine(a, b);
            if magnitude_sq(c) == 52.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}
