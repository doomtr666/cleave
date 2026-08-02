use cleave::driver::compile;
use cleave::dump::dump_program;
use cleave::registry::Registry;

fn dump(src: &str) -> (String, usize) {
    let (result, _sources) = compile(vec![("test.cleave".to_string(), src.to_string())], &[]);
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    let (out, errors) = dump_program(&program, &registry);
    (out, errors.len())
}

#[test]
fn dumps_resolved_param_and_tail_types_for_unannotated_params() {
    // `add` needs a real declared `Ring` now — `infer_call` no longer has a
    // permissive built-in fallback for operator names (see its doc comment).
    let src = "algebra Ring<T> { fn add(a: T, b: T) -> T; }\n\
               impl Ring<f64> { fn add(a: f64, b: f64) -> f64 { a } }\n\
               fn f(a, b) -> f64 { a + b }";
    let (out, errs) = dump(src);
    assert_eq!(errs, 0, "got:\n{out}");
    assert!(out.contains("fn f(a: f64, b: f64) -> f64"), "got:\n{out}");
    // Every sub-expression is annotated, not just the outermost one per
    // line (`a`/`b` each carry their own `:f64` alongside the call's own).
    assert!(out.contains("add(a:f64, b:f64):f64"), "got:\n{out}");
}

#[test]
fn dumps_let_and_tail_statements_with_their_own_types() {
    // `x` is generalized at the `let` (a syntactic value, immutable) — its
    // own definition site correctly shows a still-open type variable, *not*
    // a misleadingly concrete default (`apply_defaults` must never bind a
    // variable `generalize` already quantified, see `Infer::quantified`'s
    // doc comment — found necessary by a real, reproducible case: a
    // self-recursive top-level `fn` whose own body showed a hardcoded `i32`
    // for a variable its own signature reported as still-generic). The
    // *tail*'s own reference is a fresh instantiation, independently pinned
    // to `i32` by `f`'s declared return type — genuinely concrete, not a
    // default.
    let (out, errs) = dump("fn f() -> i32 { let x = 1; x }");
    assert_eq!(errs, 0, "got:\n{out}");
    assert!(out.contains("let x = 1:'a"), "got:\n{out}");
    assert!(out.contains("x:i32"), "got:\n{out}");
}

#[test]
fn a_type_error_in_one_function_does_not_stop_the_others() {
    let src = "algebra Ring<T> { fn add(a: T, b: T) -> T; }\n\
               fn bad(a: f64, b: i32) -> f64 { a + b }\n\
               fn good() -> i32 { 1 }";
    let (out, errs) = dump(src);
    assert_eq!(errs, 1, "exactly the one broken function should error, got:\n{out}");
    assert!(out.contains("type error"), "got:\n{out}");
    assert!(out.contains("fn good() -> i32"), "the working function must still be dumped, got:\n{out}");
    assert!(out.contains("1:i32"), "got:\n{out}");
}

#[test]
fn nested_sub_expressions_are_each_annotated_with_their_own_type_not_just_the_outermost() {
    // The whole point of `--dump-inference-pass`: debugging a deeply nested
    // expression needs to see every sub-expression's own type, not just the
    // one type reported for the entire statement/tail line.
    let src = "algebra Ring<T> { fn add(a: T, b: T) -> T; fn sub(a: T, b: T) -> T; }
        impl Ring<i32> { fn add(a: i32, b: i32) -> i32 { a } fn sub(a: i32, b: i32) -> i32 { a } }
        fn f(x: i32) -> i32 { add(sub(x, 1), sub(x, 2)) }";
    let (out, errs) = dump(src);
    assert_eq!(errs, 0, "got:\n{out}");
    assert!(out.contains("sub(x:i32, 1:i32):i32"), "the inner `sub` calls must show their own type, got:\n{out}");
    assert!(
        out.contains("add(sub(x:i32, 1:i32):i32, sub(x:i32, 2:i32):i32):i32"),
        "got:\n{out}"
    );
}

#[test]
fn a_generalized_self_recursive_function_does_not_show_a_contradictory_default_in_its_own_body() {
    // The exact scenario reported directly: `fibonacci`, generic over any
    // type implementing `TestAlg` (declared here for both `i32` and `i64` —
    // `fibonacci`'s own body uses bare, undotted literals throughout, which
    // now genuinely constrains it to `Int`-family types specifically, not
    // just any `Num`; see `cleave/tests/callgraph.rs`'s
    // `fibonacci_with_bare_int_literals_in_its_own_body_cannot_be_called_with_a_float`
    // for the `Float` counterpart), called with an `i64` argument — a
    // *different* concrete type than its own internal defaulting would ever
    // pick on its own (`i32`), still proving real polymorphism. Before the
    // `Infer::quantified` fix, `fibonacci`'s own body hardcoded `x:i32`
    // throughout (an arbitrary default), directly contradicting its own
    // signature (`(x: 't..) -> 't..`, correctly still-generic).
    let src = "algebra TestAlg<T> {
            fn add(x: T, y: T) -> T;
            fn sub(x: T, y: T) -> T;
            fn gt(x: T, y: T) -> bool;
        }
        impl TestAlg<i32> { fn add(x, y) { x } fn sub(x, y) { x } fn gt(x, y) { true } }
        impl TestAlg<i64> { fn add(x, y) { x } fn sub(x, y) { x } fn gt(x, y) { true } }
        fn fibonacci(x) {
            if x > 2 { fibonacci(x - 1) + fibonacci(x - 2) } else { x }
        }
        fn main() -> i64 { fibonacci(42:i64) }";
    let (out, errs) = dump(src);
    assert_eq!(errs, 0, "got:\n{out}");
    assert!(out.contains("main() -> i64"), "got:\n{out}");
    assert!(out.contains("fibonacci(42:i64):i64"), "got:\n{out}");

    // Scoped to `fibonacci`'s own rendered block specifically — the `impl
    // TestAlg<i32>`/`<i64>` blocks legitimately show `x:i32`/`x:i64` (a
    // genuinely concrete impl), which a whole-output check would wrongly
    // trip on.
    let fib_start = out.find("fn fibonacci").expect("fibonacci must be dumped");
    let fib_end = fib_start + out[fib_start..].find("\n}\n").unwrap();
    let fib_block = &out[fib_start..fib_end];
    assert!(
        !fib_block.contains(":i32") && !fib_block.contains(":i64") && !fib_block.contains(":f32"),
        "fibonacci's own body must not hardcode a default that contradicts its still-generic signature, got:\n{fib_block}"
    );
}

#[test]
fn an_impl_method_can_call_an_ordinary_top_level_function() {
    // A real bug, found by direct testing: an `impl` method's own `env` was
    // always empty (`Env::new()`, never connected to `callgraph.rs`'s
    // `global_env`) -- calling *any* top-level `fn`, even a wholly ordinary
    // non-generic one, silently fell through to `infer_call`'s
    // `<unresolved-call:...>` placeholder, with no error at all (the
    // placeholder never happened to reach the impl method's own exposed
    // signature). Fixed by seeding the method body's `env` with
    // `ProgramInference::global_env` (`Infer::infer_impl_fn_generic_with_env`).
    let src = "algebra TestAlg<T> { fn gt(x: T, y: T) -> bool; }
        fn helper(x: i32) -> i32 { x }
        impl TestAlg<i32> { fn gt(x, y) { helper(x); true } }
        fn main() -> i32 { 0 }";
    let (out, errs) = dump(src);
    assert_eq!(errs, 0, "got:\n{out}");
    assert!(out.contains("helper(x:i32):i32"), "got:\n{out}");
}

#[test]
fn structs_and_algebras_are_shown_as_markers_not_omitted() {
    let (out, errs) = dump("struct Vec2 { x: f64, y: f64 }\nfn f() -> i32 { 0 }");
    assert_eq!(errs, 0, "got:\n{out}");
    assert!(out.contains("struct Vec2"), "got:\n{out}");
}
