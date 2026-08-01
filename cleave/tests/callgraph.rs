use cleave::ast::{FileId, Program};
use cleave::callgraph::infer_program;
use cleave::infer::Ty;
use cleave::lower::Lowerer;
use cleave::parser::{CleaveParser, Rule};
use cleave::registry::Registry;
use pest::Parser;

fn lower_program(src: &str) -> Program {
    let pair = CleaveParser::parse(Rule::program, src)
        .unwrap_or_else(|e| panic!("parse failed for {src:?}: {e}"))
        .next()
        .unwrap();
    Lowerer::new(FileId(0)).lower_program(pair)
}

fn registry_from(src: &str) -> Registry {
    Registry::build(&lower_program(src))
}

/// Same test-only stand-in-for-a-real-stdlib fixture as `tests/infer.rs` —
/// see that file's own doc comment for why this isn't the final stdlib
/// design.
fn builtin_registry() -> Registry {
    registry_from(
        "algebra Ring<T> {
            fn add(a: T, b: T) -> T;
            fn sub(a: T, b: T) -> T;
        }
        algebra Ord<T> {
            fn gt(a: T, b: T) -> bool;
            fn eq(a: T, b: T) -> bool;
        }
        impl Ring<i32> { fn add(a: i32, b: i32) -> i32 { a } fn sub(a: i32, b: i32) -> i32 { a } }
        impl Ord<i32> { fn gt(a: i32, b: i32) -> bool { true } fn eq(a: i32, b: i32) -> bool { true } }",
    )
}

/// Like `builtin_registry`, but `Ring`/`Ord` are implemented for **both**
/// `i32` and `f32` — needed for the generalize×defaulting×merge matrix
/// below, which specifically needs two *different* concrete numeric types
/// both to be legitimate, so a conflict between them is a genuine ambiguity
/// and not just a missing `impl`. Also declares `Int`/`Float` (mirroring
/// `stdlib/num/num.cleave`) — the matrix's whole point is checking that a
/// numeric literal's own shape is a real, checked constraint, so the
/// registry it runs against needs to actually have something to check it
/// against, exactly like a program that writes `use num;` would.
fn dual_type_registry() -> Registry {
    registry_from(
        "algebra Ring<T> {
            fn add(a: T, b: T) -> T;
            fn sub(a: T, b: T) -> T;
        }
        algebra Ord<T> {
            fn gt(a: T, b: T) -> bool;
        }
        algebra Int<T> {}
        algebra Float<T> {}
        impl Ring<i32> { fn add(a: i32, b: i32) -> i32 { a } fn sub(a: i32, b: i32) -> i32 { a } }
        impl Ring<f32> { fn add(a: f32, b: f32) -> f32 { a } fn sub(a: f32, b: f32) -> f32 { a } }
        impl Ord<i32> { fn gt(a: i32, b: i32) -> bool { true } }
        impl Int<i32> {}
        impl Float<f32> {}
        impl Ord<f32> { fn gt(a: f32, b: f32) -> bool { true } }",
    )
}

fn ok_result(results: &cleave::callgraph::ProgramInference, name: &str) -> Ty {
    results
        .results
        .get(name)
        .unwrap_or_else(|| panic!("no result recorded for `{name}`"))
        .as_ref()
        .unwrap_or_else(|e| panic!("inference failed for `{name}`: {e:?}"))
        .result
        .clone()
}

fn err_result(results: &cleave::callgraph::ProgramInference, name: &str) -> cleave::infer::TypeError {
    results
        .results
        .get(name)
        .unwrap_or_else(|| panic!("no result recorded for `{name}`"))
        .as_ref()
        .err()
        .unwrap_or_else(|| panic!("expected `{name}` to fail inference, but it succeeded"))
        .clone()
}

#[test]
fn self_recursive_function_resolves_via_its_own_placeholder() {
    // The motivating example from the actual CLI session: a self-recursive
    // top-level `fn`, previously falling through to an `<unresolved-call>`
    // placeholder on its own parameter because `fibonacci` wasn't in `env`
    // while its own body was being inferred. `fibonacci` itself takes a
    // parameter, so it's generalized (see
    // `a_generalizable_top_level_function_is_not_defaulted_before_it_can_generalize`)
    // — its *own* reported type is a still-generic variable, exactly like a
    // polymorphic `let`-bound lambda's would be; what actually matters here
    // is that a real caller instantiating it resolves cleanly.
    let registry = builtin_registry();
    let program = lower_program(
        "fn fibonacci(x) {
            if x > 2 { fibonacci(x - 1) + fibonacci(x - 2) } else { x }
        }
        fn use_fibonacci() -> i32 { fibonacci(42) }",
    );
    let result = infer_program(&program, &registry);
    assert!(result.results.get("fibonacci").unwrap().is_ok(), "{:?}", result.results.get("fibonacci"));
    assert_eq!(ok_result(&result, "use_fibonacci"), Ty::Con("i32".to_string()));
}

#[test]
fn mutual_recursion_between_two_functions_resolves() {
    // `is_even`/`is_odd` — the canonical two-function mutual-recursion
    // example: neither can be inferred in isolation (each calls the other,
    // declared in either order), only as a group.
    let registry = builtin_registry();
    let program = lower_program(
        "fn is_even(n) {
            if n == 0 { true } else { is_odd(n - 1) }
        }
        fn is_odd(n) {
            if n == 0 { false } else { is_even(n - 1) }
        }",
    );
    let result = infer_program(&program, &registry);
    assert_eq!(ok_result(&result, "is_even"), Ty::Con("bool".to_string()));
    assert_eq!(ok_result(&result, "is_odd"), Ty::Con("bool".to_string()));
}

#[test]
fn declaration_order_does_not_matter_for_mutual_recursion() {
    // Same pair as above, declared in the opposite order — Tarjan's
    // algorithm (and the grouping it drives) must not be sensitive to which
    // one happens to appear first in the source file.
    let registry = builtin_registry();
    let program = lower_program(
        "fn is_odd(n) {
            if n == 0 { false } else { is_even(n - 1) }
        }
        fn is_even(n) {
            if n == 0 { true } else { is_odd(n - 1) }
        }",
    );
    let result = infer_program(&program, &registry);
    assert_eq!(ok_result(&result, "is_even"), Ty::Con("bool".to_string()));
    assert_eq!(ok_result(&result, "is_odd"), Ty::Con("bool".to_string()));
}

#[test]
fn top_level_functions_are_generalized_and_usable_at_different_types_by_different_callers() {
    // Once a function's own group is fully inferred it's generalized just
    // like a `let`-bound lambda — a later, unrelated caller (its own,
    // separate group, processed afterwards) can instantiate it fresh at a
    // different concrete type.
    let registry = builtin_registry();
    let program = lower_program(
        "fn identity(x) { x }
        fn use_i32() -> i32 { identity(1) }
        fn use_f64() -> f64 { identity(1.5) }",
    );
    let result = infer_program(&program, &registry);
    assert_eq!(ok_result(&result, "use_i32"), Ty::Con("i32".to_string()));
    assert_eq!(ok_result(&result, "use_f64"), Ty::Con("f64".to_string()));
}

#[test]
fn defaulting_is_deferred_until_the_whole_mutually_recursive_group_is_known() {
    // `f` and `g` each bottom out on a bare numeric literal — if defaulting
    // ran right after inferring `f`'s own body (rather than once for the
    // whole group), it would risk pinning `f`'s literal to a concrete type
    // before `g`'s own body — which `f`'s literal is unified with, via the
    // mutual calls — had contributed anything. Both must still agree.
    let registry = builtin_registry();
    let program = lower_program(
        "fn f(n) {
            if n == 0 { 0 } else { g(n - 1) }
        }
        fn g(n) {
            if n == 0 { 1 } else { f(n - 1) }
        }
        fn use_f() -> i32 { f(5) }",
    );
    let result = infer_program(&program, &registry);
    assert!(result.results.get("f").unwrap().is_ok(), "{:?}", result.results.get("f"));
    assert!(result.results.get("g").unwrap().is_ok(), "{:?}", result.results.get("g"));
    assert_eq!(ok_result(&result, "use_f"), Ty::Con("i32".to_string()));
}

#[test]
fn a_type_error_in_one_group_does_not_corrupt_an_unrelated_group() {
    let registry = builtin_registry();
    let program = lower_program(
        "fn broken(x) {
            if broken(x) > 2 { true } else { 1 }
        }
        fn fine() -> i32 { 42 }",
    );
    let result = infer_program(&program, &registry);
    assert!(result.results.get("broken").unwrap().is_err(), "`broken`'s if-branches disagree (bool vs i32)");
    assert_eq!(ok_result(&result, "fine"), Ty::Con("i32".to_string()));
}

#[test]
fn a_generalizable_top_level_function_is_not_defaulted_before_it_can_generalize() {
    // Regression test for a real ordering bug found while building this:
    // `add_one`, processed alone (nothing else forces its parameter
    // concrete), must still generalize its literal's `Num`-constrained
    // variable rather than have `apply_defaults` permanently pin it to
    // `i32` first — otherwise a *later*, differently-typed caller
    // (`use_f64`, its own separate group, processed afterwards) could never
    // unify with it again.
    let registry = registry_from(
        "algebra Ring<T> { fn add(a: T, b: T) -> T; }
        algebra Num<T> {}
        impl Ring<i32> { fn add(a: i32, b: i32) -> i32 { a } }
        impl Ring<f64> { fn add(a: f64, b: f64) -> f64 { a } }
        impl Num<i32> {}
        impl Num<f64> {}",
    );
    let program = lower_program(
        "fn add_one(x) { x + 1 }
        fn use_i32() -> i32 { add_one(1) }
        fn use_f64() -> f64 { add_one(1.5) }",
    );
    let result = infer_program(&program, &registry);
    assert_eq!(ok_result(&result, "use_i32"), Ty::Con("i32".to_string()));
    assert_eq!(ok_result(&result, "use_f64"), Ty::Con("f64".to_string()));
}

#[test]
fn a_nullary_function_is_not_generalized_monomorphism_restriction() {
    // Haskell's Monomorphism Restriction, applied here: a zero-parameter
    // top-level `fn` can't accept any caller-supplied type information at
    // its call sites the way a parameterized one can, so its own bare
    // literal must default directly to a concrete type rather than being
    // reported as spuriously "generic".
    let registry = registry_from(
        "algebra Ring<T> { fn add(a: T, b: T) -> T; }
        algebra Num<T> {}
        impl Ring<i32> { fn add(a: i32, b: i32) -> i32 { a } }
        impl Num<i32> {}",
    );
    let program = lower_program("fn one() { 1 }");
    let result = infer_program(&program, &registry);
    assert_eq!(ok_result(&result, "one"), Ty::Con("i32".to_string()));
}

// ---------------------------------------------------------------------
// generalize × apply_defaults × union-find merging matrix
//
// All three bugs found while building this pass lived at this one
// intersection: a variable shared between two things that *look*
// independent (two separate calls to the same generalized function, two
// literals, two members of a mutually-recursive group) via unification
// through a shared generic algebra call. A single-function, single-call-site
// test can't reach this territory at all — it takes a deliberate merge to
// get there. These are grouped together, not scattered, so the coverage
// gap this class of bug lives in stays visible as one thing.
// ---------------------------------------------------------------------

#[test]
fn two_calls_to_a_generalized_function_merged_via_an_operator_with_conflicting_types_are_rejected() {
    // The exact real-world repro: `fibonacci`, self-recursive and
    // generalized (see `self_recursive_function_resolves_via_its_own_placeholder`),
    // called twice with *different* concrete types and then forced together
    // by `add`'s own shared generic `T`. Each call gets its own fresh
    // instantiation (so this is *not* the same bug as
    // `defaulting_is_deferred_until_the_whole_mutually_recursive_group_is_known`),
    // but `add(fibonacci(42), fibonacci(42.0))` still requires both
    // instantiations to resolve to the *same* type — `i32` and `f32` can't
    // both win.
    let registry = dual_type_registry();
    let program = lower_program(
        "fn fibonacci(x) {
            if x > 2 { fibonacci(x - 1) + fibonacci(x - 2) } else { x }
        }
        fn use_both() -> i32 { fibonacci(42) + fibonacci(42.0) }",
    );
    let result = infer_program(&program, &registry);
    assert!(result.results.get("fibonacci").unwrap().is_ok(), "{:?}", result.results.get("fibonacci"));
    let err = err_result(&result, "use_both");
    // `MissingImpl`, not `Unify` — the conflict is caught by the literal's
    // own `Int`/`Float` constraint failing against whichever concrete type
    // the merge settled on (see `infer.rs`'s `NumberLit` handling), not by
    // `apply_defaults` comparing the two literals against each other
    // directly.
    assert!(matches!(err.kind, cleave::infer::TypeErrorKind::MissingImpl { .. }), "got: {:?}", err.kind);
}

#[test]
fn two_calls_to_a_generalized_function_merged_via_an_operator_with_the_same_type_succeed() {
    let registry = dual_type_registry();
    let program = lower_program(
        "fn fibonacci(x) {
            if x > 2 { fibonacci(x - 1) + fibonacci(x - 2) } else { x }
        }
        fn use_both() -> i32 { fibonacci(1) + fibonacci(2) }",
    );
    let result = infer_program(&program, &registry);
    assert_eq!(ok_result(&result, "use_both"), Ty::Con("i32".to_string()));
}

#[test]
fn two_independent_instantiations_of_a_generalized_function_at_different_types_do_not_conflict() {
    // The counterpart to the two tests above: binding two differently-typed
    // instantiations to two *separate*, never-combined `let`s must not be
    // treated as a conflict — nothing ever asks the two to agree. Uses
    // `identity`, not `fibonacci`: `fibonacci`'s own body contains bare,
    // int-shaped literals (`x - 1`, `x - 2`), which — now that a literal's
    // shape is a real constraint, not just a defaulting hint — genuinely
    // does restrict it to `Int` types specifically (see
    // `fibonacci_with_bare_int_literals_in_its_own_body_cannot_be_called_with_a_float`
    // below); `identity` has no literals in its body at all, so it's
    // actually `Num`-unconstrained and a fair test of "two independent
    // instantiations don't conflict" on its own.
    let registry = dual_type_registry();
    let program = lower_program(
        "fn identity(x) { x }
        fn use_both() -> i32 {
            let a = identity(42);
            let b = identity(42.0);
            0
        }",
    );
    let result = infer_program(&program, &registry);
    assert_eq!(ok_result(&result, "use_both"), Ty::Con("i32".to_string()));
}

#[test]
fn fibonacci_with_bare_int_literals_in_its_own_body_cannot_be_called_with_a_float() {
    // A direct, surprising-at-first consequence of literal shape becoming a
    // real constraint, worth its own test: `fibonacci`, written with bare
    // `x - 1`/`x - 2` (no `.`), is not actually `Num`-polymorphic — its own
    // body requires `Int` specifically, since there's no implicit
    // int/float conversion anywhere in the language. Calling it with a
    // float argument is now correctly rejected — this *used* to silently
    // "work" (report `f32`) before literal shape had any real enforcement.
    let registry = dual_type_registry();
    let program = lower_program(
        "fn fibonacci(x) {
            if x > 2 { fibonacci(x - 1) + fibonacci(x - 2) } else { x }
        }
        fn use_as_float() -> f32 { fibonacci(42.0) }",
    );
    let result = infer_program(&program, &registry);
    assert!(result.results.get("fibonacci").unwrap().is_ok(), "{:?}", result.results.get("fibonacci"));
    let err = err_result(&result, "use_as_float");
    assert!(matches!(err.kind, cleave::infer::TypeErrorKind::MissingImpl { .. }), "got: {:?}", err.kind);
}

#[test]
fn fibonacci_with_bare_int_literals_still_works_when_called_with_an_int() {
    let registry = dual_type_registry();
    let program = lower_program(
        "fn fibonacci(x) {
            if x > 2 { fibonacci(x - 1) + fibonacci(x - 2) } else { x }
        }
        fn use_as_int() -> i32 { fibonacci(42) }",
    );
    let result = infer_program(&program, &registry);
    assert_eq!(ok_result(&result, "use_as_int"), Ty::Con("i32".to_string()));
}

#[test]
fn conflicting_literal_shapes_across_a_mutually_recursive_group_are_rejected() {
    // `f`'s base case (`1`, int-shaped) and `g`'s base case (`2.0`,
    // float-shaped) end up sharing a type through the mutual calls (`f`'s
    // own return feeds `g`'s if-branch and vice versa) even though neither
    // literal is textually anywhere near the other. Both `f` and `g` take a
    // parameter, so they're generalized — `n`'s shared, conflicting-shaped
    // variable is quantified into their scheme rather than defaulted on the
    // spot (see `Infer::quantified`), so the conflict only actually
    // surfaces once *something* instantiates that scheme — here, `use_f`.
    // A never-called mutually-recursive pair with an unsatisfiable (`Int`
    // *and* `Float` on the same variable) scheme is a real, separate,
    // deliberately-not-caught gap: nothing checks a scheme's own
    // constraints for satisfiability at generalization time, only at
    // instantiation — matching how nothing here proactively checks dead
    // code for soundness in general.
    let registry = dual_type_registry();
    let program = lower_program(
        "fn f(n) {
            if n > 0 { g(n - 1) } else { 1 }
        }
        fn g(n) {
            if n > 0 { f(n - 1) } else { 2.0 }
        }
        fn use_f() -> i32 { f(5) }",
    );
    let result = infer_program(&program, &registry);
    let err = err_result(&result, "use_f");
    assert!(matches!(err.kind, cleave::infer::TypeErrorKind::MissingImpl { .. }), "got: {:?}", err.kind);
}

#[test]
fn same_literal_shape_across_a_mutually_recursive_group_succeeds() {
    let registry = dual_type_registry();
    let program = lower_program(
        "fn f(n) {
            if n > 0 { g(n - 1) } else { 1 }
        }
        fn g(n) {
            if n > 0 { f(n - 1) } else { 2 }
        }",
    );
    let result = infer_program(&program, &registry);
    assert!(result.results.get("f").unwrap().is_ok(), "{:?}", result.results.get("f"));
    assert!(result.results.get("g").unwrap().is_ok(), "{:?}", result.results.get("g"));
}

#[test]
fn a_nullary_function_is_callable_from_another_function() {
    // A real bug, found by testing (unrelated to the shape-conflict matrix
    // above, surfaced by the struct-construction work): a nullary member
    // never went through the generalize loop (Monomorphism Restriction) and
    // so was never inserted into `global_env` *at all* — meaning calling it
    // from any other function fell straight through to `infer_call`'s
    // unresolved-call placeholder. Not generalizing a nullary binding only
    // ever meant "don't quantify its free variables"; it never meant "don't
    // expose it to callers".
    let registry = builtin_registry();
    let program = lower_program(
        "fn constant() -> i32 { 42 }
        fn use_constant() -> i32 { constant() }",
    );
    let result = infer_program(&program, &registry);
    assert_eq!(ok_result(&result, "constant"), Ty::Con("i32".to_string()));
    assert_eq!(ok_result(&result, "use_constant"), Ty::Con("i32".to_string()));
}

#[test]
fn a_bad_const_generic_type_name_is_caught_by_the_whole_program_pass_too() {
    // A real bug, found by direct user testing via the actual CLI: `Infer::
    // check_pending_type_names` (see `infer.rs`) was wired into `finish_fn`,
    // but `callgraph.rs`'s whole-program pass calls `infer_fn_raw` directly
    // and never went through `finish_fn` at all -- so `const R: Int` was
    // rejected by `infer_fn`/`infer_impl_fn_generic` (exercised by
    // `tests/infer.rs`) but sailed through silently for any ordinary
    // top-level `fn`, which is every function `--dump-inference-pass` (and
    // every real `.cleave` file) actually goes through. `main`, unaffected
    // and in a separate SCC group, must still succeed -- confirms the fix
    // is attributed per-function, not a group-wide false positive.
    let src = "algebra Int<T> {}
        impl Int<i32> {}
        algebra Float<T> {}
        impl Float<f64> {}
        struct Matrix<T : Float, const R : Int, const C : Int> {
            values : [T; R, C]
        }
        fn f() -> Matrix<f64, 2, 3> {
            Matrix(values: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        }
        fn main() -> i32 {
            42
        }";
    let program = lower_program(src);
    let registry = registry_from(src);
    let result = infer_program(&program, &registry);
    let err = err_result(&result, "f");
    assert!(
        matches!(&err.kind, cleave::infer::TypeErrorKind::TypeNameIsAnAlgebra { name } if name == "Int"),
        "got: {:?}",
        err.kind
    );
    assert_eq!(ok_result(&result, "main"), Ty::Con("i32".to_string()));
}
