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

use cleave::pipeline::CodegenOptions;
use cleave_mlir_shim::mlir::Context;
use cleave_mlir_shim::mlir::dialect::DialectRegistry;
use cleave_mlir_shim::mlir::utility::register_all_dialects;

fn context() -> Context {
    let dialect_registry = DialectRegistry::new();
    register_all_dialects(&dialect_registry);
    let context = Context::new();
    context.append_dialect_registry(&dialect_registry);
    context.load_all_available_dialects();
    context
}

/// Compiles and runs `src`'s `main` through the real pipeline
/// (`cleave::run`, what `--run` uses): the guide's examples as a user runs
/// them. `_context` is unused, kept for the call sites.
fn run_i32(_context: &Context, src: &str) -> i32 {
    let options = CodegenOptions { openmp: false, tasks: false, ..Default::default() };
    cleave::run::run_source("test.cleave", src, &options).unwrap_or_else(|e| panic!("{}", e.join("
")))
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
    assert_eq!(
        run_i32(
            &context,
            "fn add_one(x: i32) -> i32 { x + 1 } fn main() -> i32 { add_one(5) }"
        ),
        6
    );
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
    assert_eq!(
        run_i32(
            &context,
            "fn add_one(x) { x + 1 } fn main() -> i32 { add_one(5) }"
        ),
        6
    );
}

#[test]
fn annotated_function_signature() {
    let context = context();
    assert_eq!(
        run_i32(
            &context,
            "fn add_one(x: i32) -> i32 { x + 1 } fn main() -> i32 { add_one(5) }"
        ),
        6
    );
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
        impl Additive<Vec2> {
            fn add(a, b) { Vec2(x: a.x + b.x, y: a.y + b.y) }
            fn sub(a, b) { Vec2(x: a.x - b.x, y: a.y - b.y) }
            fn neg(a) { Vec2(x: -a.x, y: -a.y) }
            fn zero() { Vec2(x: 0.0, y: 0.0) }
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

#[test]
fn field_access_on_an_unannotated_parameter() {
    let context = context();
    let src = "
        struct Vec2 { x: f64, y: f64 }
        struct Pixel { x: i32, y: i32, color: i32 }
        fn first(v) { v.x }
        fn main() -> i32 {
            if first(Vec2(x: 3.0, y: 4.0)) == 3.0 and first(Pixel(x: 7, y: 0, color: 1)) == 7 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

#[test]
fn tuples_destructured_by_let_and_assignment() {
    let context = context();
    let src = "
        fn pair(n: i32) -> (i32, i32) { (n, n * 10) }
        fn main() -> i32 {
            let (a, mut b) = pair(2);
            b = b + 1;
            let ((c, d), e) = ((3, 4), 5);
            let mut x = 0;
            let mut y = 0;
            (x, y) = pair(6);
            (x, y) = (y, x);
            if a == 2 and b == 21 and c + d + e == 12 and x == 60 and y == 6 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

#[test]
fn structs_and_tuples_indexed_by_position() {
    let context = context();
    let src = "
        struct Vec3 { x: f64, y: f64, z: f64 }
        fn first(t) { t[0] }
        fn main() -> i32 {
            let v = Vec3(x: 1.0, y: 2.0, z: 3.0);
            if v[2] == 3.0 and v[0] == v.x and first((7, 2.5)) == 7 and len(v) == 3 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

#[test]
fn for_over_a_tuple_is_unrolled() {
    let context = context();
    let src = "
        fn main() -> i32 {
            let t = (7, 2.5, 9);
            let mut same = 0;
            for i in 0..t.len() { if t[i] == t[i] { same = same + 1; }; };
            same
        }
    ";
    assert_eq!(run_i32(&context, src), 3);
}

#[test]
fn an_impl_over_a_pack_of_types() {
    let context = context();
    let src = "
        algebra Show<T> { fn show(x: T) -> i32; }
        impl Show<i32> { fn show(x) { 1 } }
        impl Show<f64> { fn show(x) { 10 } }
        impl<Ts...: Show> Show<Ts...> {
            fn show(x) {
                let mut s = 0;
                for i in 0..x.len() { s = s + show(x[i]); };
                s
            }
        }
        fn main() -> i32 { show((1, 2.5:f64, 3)) }
    ";
    assert_eq!(run_i32(&context, src), 12);
}

#[test]
fn a_comprehension_seeds_any_collection() {
    let context = context();
    let src = "
        use dynarray;
        fn main() -> i32 {
            let big = [for i in 0..10000: i * 2];
            let n = 5;
            let v = [for i in 0..n: i * i];
            let grid = [for i in 0..2: [for j in 0..3: i * 10 + j]];
            if big[9999] == 19998 and v[4] == 16 and grid[1][2] == 12 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

#[test]
fn a_comprehension_takes_its_type_from_context() {
    let context = context();
    let src = "
        struct Pair { a: i32, b: f64 }
        fn twice(x) { x + x }
        fn main() -> i32 {
            let squares = [for i in 0..4: i * i];
            let p = Pair(a: 1, b: 2.5);
            let t = [for i in 0..p.len(): twice(p[i])];
            let q: Pair = [for i in 0..p.len(): twice(p[i])];
            if squares[3] == 9 and t[1] == 5.0 and q.b == 5.0 { 1 } else { 0 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1);
}

#[test]
fn how_a_call_finds_what_it_calls() {
    let context = context();
    let src = "
        algebra Doubling<T> { fn twice(x: T) -> T; }
        impl Doubling<i32> { fn twice(x) { x * 2 } }
        fn twice(x: i32) -> i32 { x * 10 }
        fn main() -> i32 {
            twice(5) + Doubling::twice(5)
        }
    ";
    assert_eq!(run_i32(&context, src), 60);
}

// ---------------------------------------------------------------- Inherent impls

/// Inherent impls are gone as a language concept -- `v.method(args)` is now
/// pure sugar for `method(v, args)`, resolved through the same call-site
/// machinery as any other top-level function. This is the direct successor
/// of the old "Inherent impls" example: dot-call syntax on an ordinary
/// top-level function still works exactly the same way.
#[test]
fn dot_call_on_a_top_level_function_computes_the_right_value() {
    let context = context();
    let src = "
        struct Vec2 { x: f64, y: f64 }
        fn magnitude_sq(v: Vec2) -> f64 { v.x * v.x + v.y * v.y }
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

// ---------------------------------------------------------------- const/define

#[test]
fn a_named_const_is_usable_as_an_ordinary_value() {
    let context = context();
    let src = "
        const SEUIL: i32 = 100;
        fn main() -> i32 { SEUIL + 1 }
    ";
    assert_eq!(run_i32(&context, src), 101);
}

#[test]
fn arbitrary_foldable_arithmetic_on_named_consts_works_in_generic_argument_position() {
    let context = context();
    let src = "
        const A: i32 = 3;
        const B: i32 = 5;
        const C: i32 = 2;
        fn probe<const N: i32>() -> i32 { N }
        fn main() -> i32 { probe::<(A + B) * C - 1>() }
    ";
    // (3 + 5) * 2 - 1 = 15.
    assert_eq!(run_i32(&context, src), 15);
}

#[test]
fn a_define_left_unoverridden_uses_its_own_default() {
    let context = context();
    let src = "
        define FLAG: bool = false;
        fn main() -> i32 { if FLAG { 1 } else { 0 } }
    ";
    assert_eq!(run_i32(&context, src), 0);
}

#[test]
fn cleave_openmp_is_a_usable_compiler_injected_define() {
    let context = context();
    let src = "fn main() -> i32 { if CLEAVE_OPENMP { 1 } else { 0 } }";
    // It says what the run's options say: this file's `run_i32` runs without
    // OpenMP (`cleave/tests/const_decl.rs`'s
    // `cleave_openmp_reflects_the_resolved_openmp_option` checks both values).
    assert_eq!(run_i32(&context, src), 0);
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
    assert_eq!(
        run_i32(&context, "fn f() -> i32 { 1 } fn main() -> i32 { f() }"),
        1
    );
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

        impl Additive<Vec2> {
            fn add(a, b) { Vec2(x: a.x + b.x, y: a.y + b.y) }
            fn sub(a, b) { Vec2(x: a.x - b.x, y: a.y - b.y) }
            fn neg(a) { Vec2(x: -a.x, y: -a.y) }
            fn zero() { Vec2(x: 0.0, y: 0.0) }
        }

        fn magnitude_sq(v: Vec2) -> f64 { v.x * v.x + v.y * v.y }

        fn combine<T: Additive>(a: T, b: T) -> T {
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
