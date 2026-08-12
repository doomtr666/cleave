use cleave::ast::{FileId, Program};
use cleave::callgraph::infer_program;
use cleave::infer::{ConstValue, Ty, TypeErrorKind};
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
fn a_bodyless_top_level_fn_is_rejected() {
    // Legal grammatically (see `grammar.pest`'s own `fn_decl` comment) but
    // never legal for a top-level `fn` specifically -- `infer_program`
    // itself is the one real validation point with an enclosing `Item`'s
    // own span to report against (`FnDecl` carries none).
    let registry = builtin_registry();
    let program = lower_program("fn f(x: i32) -> i32;");
    let result = infer_program(&program, &registry);
    let err = result.results.get("f").unwrap().as_ref().unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::MissingFnBody { .. }), "got: {:?}", err.kind);
}

#[test]
fn an_extern_fn_is_accepted_bodyless_with_its_declared_signature() {
    let registry = builtin_registry();
    let program = lower_program("extern fn f(x: i32) -> i32;");
    let result = infer_program(&program, &registry);
    assert_eq!(ok_result(&result, "f"), Ty::Con("i32".to_string()));
}

#[test]
fn an_extern_fn_cannot_be_generic() {
    let registry = builtin_registry();
    let program = lower_program("extern fn f<T>(x: T) -> T;");
    let result = infer_program(&program, &registry);
    let err = result.results.get("f").unwrap().as_ref().unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::ExternFnCannotBeGeneric { .. }), "got: {:?}", err.kind);
}

#[test]
fn a_const_generic_bounded_for_loop_correctly_dispatches_once_a_caller_instantiates_it() {
    // Regression test: `N` (a const generic) referenced as an ordinary value
    // (`for i in 0..N`) merges with `0`'s own `Int`-constrained literal var —
    // `generalize` sweeps that shared constraint into `fill`'s own `Scheme`
    // (it shares a free variable with `N`, one of the just-quantified vars),
    // and `use_it`'s own call site (`fill::<i32, 4>(1)`) re-queues it against
    // a fresh copy of `N`'s own var, now concretely `Ty::Const(Int(4))`.
    // Without `Scheme`/`Infer` tracking each const generic's own declared
    // width across that whole `generalize`/`instantiate` journey (see
    // `Scheme::const_widths`, threaded exactly like `constraints` already
    // is), this incorrectly rejects with `no impl Int<4>` — checking the
    // *value* `4` against `Int` instead of `N`'s own declared width `i32`.
    let registry = registry_from(
        "algebra Ring<T> { fn add(a: T, b: T) -> T; }
        algebra Int<T> {}
        impl Ring<i32> { fn add(a: i32, b: i32) -> i32 { a } }
        impl Int<i32> {}",
    );
    let program = lower_program(
        "fn fill<T: Int, const N: i32>(v: T) -> [T; N] {
            let mut arr = [v; N];
            for i in 0..N {
                arr[i] = v;
            };
            arr
        }
        fn use_it() -> [i32; 4] { fill::<i32, 4>(1) }",
    );
    let result = infer_program(&program, &registry);
    assert_eq!(
        ok_result(&result, "use_it"),
        Ty::Array(Box::new(Ty::Con("i32".to_string())), Box::new(Ty::Const(ConstValue::Int(4))))
    );
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
    // spot (see `Infer::quantified`).
    //
    // Originally written (and named) to document a real, separate gap: a
    // never-called mutually-recursive pair with an unsatisfiable scheme
    // used to generalize completely silently, the conflict only ever
    // surfacing if *something* later instantiated it — see `doc/backlog.md`'s
    // own "Scheme satisfiability at generalization time" item, now fixed
    // (`Infer::generalize`'s own doc comment has the full story). Updated
    // to assert the *new* behavior directly on `f` — the conflict is now
    // caught immediately, with no external caller needed at all (unlike
    // `a_mutually_recursive_groups_own_conflicting_shape_constraints_are_
    // rejected_even_with_no_external_caller` below, which exercises the
    // same fix through a single shared expression rather than two
    // recursive `if`/`else` base cases feeding each other through mutual
    // calls — kept as a distinct, genuinely different shape).
    let registry = dual_type_registry();
    let program = lower_program(
        "fn f(n) {
            if n > 0 { g(n - 1) } else { 1 }
        }
        fn g(n) {
            if n > 0 { f(n - 1) } else { 2.0 }
        }",
    );
    let result = infer_program(&program, &registry);
    let err = err_result(&result, "f");
    assert!(matches!(err.kind, cleave::infer::TypeErrorKind::UnsatisfiableScheme { .. }), "got: {:?}", err.kind);
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

#[test]
fn a_higher_order_function_accepts_a_matching_lambda_from_a_caller() {
    let src = "fn apply(f: (i32) -> i32, x: i32) -> i32 { f(x) }
        fn g() -> i32 { let inc = fn(x) { x }; apply(inc, 5) }";
    let program = lower_program(src);
    let registry = registry_from(src);
    let result = infer_program(&program, &registry);
    assert_eq!(ok_result(&result, "g"), Ty::Con("i32".to_string()));
}

#[test]
fn a_higher_order_function_rejects_a_lambda_with_the_wrong_return_type() {
    let src = "fn apply(f: (i32) -> bool, x: i32) -> bool { f(x) }
        fn g() -> bool { let bad = fn(x: i32) -> i32 { x }; apply(bad, 5) }";
    let program = lower_program(src);
    let registry = registry_from(src);
    let result = infer_program(&program, &registry);
    let err = err_result(&result, "g");
    assert!(matches!(err.kind, cleave::infer::TypeErrorKind::Unify(_)), "got: {:?}", err.kind);
}

// ---------------------------------------------------------------------
// a nullary top-level `fn`'s own returned type can still carry a free
// variable with a real pending constraint on it -- the Monomorphism
// Restriction means it's never `generalize`d, but the constraint must not
// be lost just because of that (see `Infer::constraints_touching`'s own
// doc comment for the bug this closes: a real, un-satisfiable program used
// to silently type-check).
// ---------------------------------------------------------------------

#[test]
fn a_nullary_functions_returned_closure_still_enforces_its_own_constraint() {
    let registry = builtin_registry();
    let program = lower_program(
        "fn make_adder() { fn(x) { add(x, x) } }
        fn use_i32() -> i32 { let f = make_adder(); f(5) }",
    );
    let result = infer_program(&program, &registry);
    assert!(result.results.get("make_adder").unwrap().is_ok(), "{:?}", result.results.get("make_adder"));
    assert_eq!(ok_result(&result, "use_i32"), Ty::Con("i32".to_string()));
}

#[test]
fn a_nullary_functions_returned_closure_rejects_a_type_with_no_matching_impl() {
    // Same `make_adder` as above, but instantiated at `bool` -- `Ring` has
    // no `impl Ring<bool>` anywhere in `builtin_registry`. Before this fix,
    // `make_adder`'s own `Ring` constraint on its closure's parameter
    // vanished the moment `make_adder`'s own group (and its `Infer`
    // instance) finished, since `Scheme::mono` carried zero constraints —
    // this type-checked successfully with no error at all.
    let registry = builtin_registry();
    let program = lower_program(
        "fn make_adder() { fn(x) { add(x, x) } }
        fn use_bool() -> bool { let f = make_adder(); f(true) }",
    );
    let result = infer_program(&program, &registry);
    let err = err_result(&result, "use_bool");
    assert!(
        matches!(&err.kind, cleave::infer::TypeErrorKind::MissingImpl { algebra, .. } if algebra == "Ring"),
        "got: {:?}",
        err.kind
    );
}

/// Real bug found by direct testing (`for i in 0..N { arr[i] = v; }`, called
/// as `fill::<f32, 4>(...)`): a bare numeric literal (`0`, the loop's own
/// start bound) carries a real `Num`/`Int` constraint; `for`'s own inference
/// unifies that literal's var with `N`'s (the loop's end bound) — so `N`'s
/// var ends up carrying that constraint too, generalized into `fill`'s own
/// `Scheme` (it shares a free variable with `N`, one of `fill`'s quantified
/// vars) and re-checked once instantiated at `main`'s own call site, where
/// the turbofish pins `N` straight to `Ty::Const(Int(4))`. Before the
/// `Scheme::const_widths` fix, that check asked `has_matching_impl("Num",
/// Ty::Const(Int(4)))` directly — nonsensical (impls are declared for real
/// types, never one specific constant value) — and always failed with `no
/// impl Num<4>`, regardless of the call.
#[test]
fn a_const_generic_used_as_a_for_loop_bound_is_checked_against_its_own_declared_width_at_the_call_site() {
    let src = "algebra Num<T> {}
        algebra Int<T> : Num {}
        algebra Float<T> : Num {}
        impl Int<i32> {}
        impl Float<f32> {}
        fn fill<T: Float, const N: i32>(v: T) -> [T; N] {
            let mut arr = [v; N];
            for i in 0..N {
                arr[i] = v;
            };
            arr
        }
        fn main() -> [f32; 4] {
            fill::<f32, 4>(1.0)
        }";
    let program = lower_program(src);
    let registry = registry_from(src);
    let result = infer_program(&program, &registry);
    assert!(result.results.get("fill").unwrap().is_ok(), "{:?}", result.results.get("fill"));
    assert_eq!(ok_result(&result, "main"), Ty::Array(Box::new(Ty::Con("f32".to_string())), Box::new(Ty::Const(ConstValue::Int(4)))));
}

/// Proves the width bridge checks the *real* declared width, not a blanket
/// "assume it's fine" (e.g. always default to `i32`) — `const M: i64` used
/// as a `for` loop bound demands `Int`, and this registry only declares
/// `impl Int<i32>` (deliberately no `impl Int<i64>`), so `g::<4>(1)` must
/// still fail on the *real* width. Were the fix instead falling back to a
/// flat "assume i32" default, this would wrongly succeed.
#[test]
fn a_const_generic_used_as_a_for_loop_bound_is_checked_against_its_real_width_not_a_default() {
    let src = "algebra Num<T> {}
        algebra Int<T> : Num {}
        impl Int<i32> {}
        fn g<const M: i64>(v: i32) -> [i32; M] {
            let mut arr = [v; M];
            for i in 0..M {
                arr[i] = v;
            };
            arr
        }
        fn main() -> [i32; 4] {
            g::<4>(1)
        }";
    let program = lower_program(src);
    let registry = registry_from(src);
    let result = infer_program(&program, &registry);
    assert!(result.results.get("g").unwrap().is_ok(), "{:?}", result.results.get("g"));
    let err = err_result(&result, "main");
    // Whichever of the merged var's own constraints (`Num` and `Int` are
    // both pushed for a bare numeric literal, see `NumberLit`'s own
    // inference arm) happens to be checked first — either is equally proof
    // that the *real* `i64` width was checked, not a default.
    assert!(
        matches!(&err.kind, cleave::infer::TypeErrorKind::MissingImpl { algebra, ty } if (algebra == "Int" || algebra == "Num") && ty == "i64"),
        "got: {:?}",
        err.kind
    );
}

// ---------------------------------------------------------------------
// An inherent method's own inferred return type reaching an external
// caller (`infer_inherent_impls_early`) -- previously, a *different*
// function calling an unannotated inherent method only ever saw the
// `<not-yet-inferred>` placeholder, regardless of what the method's own
// body actually computed.
// ---------------------------------------------------------------------

/// Deliberately *one* combined source, unlike most tests above, which can
/// freely split the caller and the registry's own fixture into two separate
/// strings: `Registry::build` scans `program.items` directly for `struct`/
/// `impl struct` declarations, so the struct and its own inherent impl must
/// actually be part of the `Program` being inferred, not just known to a
/// separately-built `Registry` (same reasoning `tests/monomorphize.rs`'s own
/// `a_generic_algebra_impl_method_is_specialized_at_a_concrete_call_site`
/// already documents for algebra impls).
#[test]
fn an_external_callers_own_return_type_resolves_through_an_unannotated_inherent_method() {
    let src = "algebra Ring<T> { fn add(a: T, b: T) -> T; }
        impl Ring<i32> { fn add(a: i32, b: i32) -> i32 { a } }
        struct Vec2 { x: i32, y: i32 }
        impl struct Vec2 {
            fn sum_fields(v) { v.x + v.y }
        }
        fn use_it(v: Vec2) -> i32 { v.sum_fields() }";
    let registry = registry_from(src);
    let program = lower_program(src);
    let result = infer_program(&program, &registry);
    assert_eq!(ok_result(&result, "use_it"), Ty::Con("i32".to_string()));
}

#[test]
fn a_generic_inherent_methods_own_return_type_resolves_through_the_call_sites_own_concrete_generic() {
    let src = "struct Boxed<T> { value: T }
        impl<T> struct Boxed<T> {
            fn get(b) { b.value }
        }
        fn use_it(b: Boxed<i32>) -> i32 { b.get() }";
    let registry = registry_from(src);
    let program = lower_program(src);
    let result = infer_program(&program, &registry);
    assert_eq!(ok_result(&result, "use_it"), Ty::Con("i32".to_string()));
}

/// Graceful degradation, not a regression: an inherent method whose own
/// return type genuinely depends on a top-level `fn` (not yet visible to
/// the early pass, since `global_env` doesn't exist until *after* this
/// pass runs -- see this feature's own design notes) still correctly
/// defers to a placeholder for an *external* caller, exactly like any
/// other not-yet-resolvable reference elsewhere in this compiler -- no
/// crash, no silently wrong type.
#[test]
fn an_inherent_methods_own_dependency_on_a_top_level_fn_gracefully_defers_for_an_external_caller() {
    let src = "fn helper() -> i32 { 42 }
        struct Vec2 { x: i32, y: i32 }
        impl struct Vec2 {
            fn compute(v) { helper() }
        }
        fn use_it(v: Vec2) { v.compute() }";
    let registry = registry_from(src);
    let program = lower_program(src);
    let result = infer_program(&program, &registry);
    // A placeholder surviving all the way to a function's own *exposed*
    // result type is a real, reported error (`check_no_placeholder`,
    // matching every other case where that's true throughout this
    // compiler) -- not a silent success, and specifically not a confusing
    // type mismatch (which is what happens instead if the placeholder gets
    // unified against an incompatible concrete type elsewhere, a *worse*
    // failure mode this test is deliberately not exercising).
    let err = err_result(&result, "use_it");
    assert!(matches!(&err.kind, TypeErrorKind::Unresolved(s) if s.starts_with('<')), "got: {:?}", err.kind);
}

/// `doc/backlog.md`'s own "Explicit turbofish on a const generic, for a
/// plain top-level `fn`" item. Root cause, confirmed by direct testing, is
/// not a turbofish-arity bug at all: `N`, referenced as an ordinary body
/// *value* (`{ N }`, not a type-position use like `[T; N]`), used to share
/// its own single fresh type-var between "N's own declared type" (`i32`)
/// and "N's own generic identity" -- checking it against `rep`'s own
/// declared return type permanently collapsed that var to `Con("i32")`,
/// destroying its own identity before `rep`'s scheme was ever built, so
/// `rep` was reported with *zero* declared generics -- turbofish had
/// nothing to match `::<3>` against. `f`'s own resolved type comes out as
/// `Ty::Const(ConstValue::Int(3))`, not a bare `Ty::Con("i32")` -- once `N`
/// is correctly pinned to `3` by the turbofish call, that's genuinely what
/// `rep::<3>(5)`'s own call-site type *is* (mirrors `--dump-inference-pass`
/// on this exact source, confirmed directly during development).
#[test]
fn explicit_turbofish_on_a_const_generic_resolves_the_calls_own_type() {
    let registry = Registry::default();
    let program = lower_program(
        "fn rep<const N: i32>(x: i32) -> i32 { N }
         fn f() -> i32 { rep::<3>(5) }",
    );
    let result = infer_program(&program, &registry);
    assert!(result.results.get("rep").unwrap().is_ok(), "{:?}", result.results.get("rep"));
    assert_eq!(ok_result(&result, "f"), Ty::Const(ConstValue::Int(3)));
}

/// `doc/backlog.md`'s own "Scheme satisfiability at generalization time"
/// item — `f`/`g`'s shared quantified `t` carries both an `Int` shape
/// constraint (from `f`'s `x + 1`) and a `Float` one (from `g`'s own `x +
/// 1.0`), which can never both hold for any single concrete type — yet
/// nothing external ever calls into this group, so nothing used to notice.
/// Confirmed directly by testing before writing the fix: both `f`/`g`
/// generalized cleanly, reported as `fn f(x: 'a) -> 'a`, no error anywhere
/// — the fix must fire *here*, at generalization time, with no caller
/// needed at all, not just "eventually, once someone instantiates it"
/// (that half already worked, confirmed separately).
#[test]
fn a_mutually_recursive_groups_own_conflicting_shape_constraints_are_rejected_even_with_no_external_caller() {
    let registry = dual_type_registry();
    let program = lower_program(
        "fn f(x) { g(x); x + 1 }
         fn g(x) { f(x); x + 1.0 }",
    );
    let result = infer_program(&program, &registry);
    let err = err_result(&result, "f");
    assert!(matches!(err.kind, TypeErrorKind::UnsatisfiableScheme { .. }), "got: {:?}", err.kind);
}

/// Regression guard: two *compatible* single-target constraints sharing one
/// quantified variable (`Int` and `Ord`, both satisfied by `i32`) must keep
/// generalizing cleanly — the fix only rejects a genuinely empty
/// intersection, not "more than one constraint" in general.
#[test]
fn compatible_shape_constraints_on_the_same_variable_still_generalize() {
    let registry = dual_type_registry();
    let program = lower_program("fn is_positive(x) { gt(x, 0) }");
    let result = infer_program(&program, &registry);
    assert!(result.results.get("is_positive").unwrap().is_ok(), "{:?}", result.results.get("is_positive"));
}
