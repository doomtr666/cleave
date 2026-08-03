use cleave::ast::{FileId, FnDecl, GenericParam, ItemKind, Program, Span, StmtKind, Type};
use cleave::infer::{ConstValue, Infer, Ty, TypeErrorKind};
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

fn lower_one_fn(src: &str) -> FnDecl {
    let program = lower_program(src);
    assert_eq!(program.items.len(), 1, "expected exactly one item in {src:?}");
    match program.items.into_iter().next().unwrap().kind {
        ItemKind::Fn(f) => f,
        other => panic!("expected a fn item, got {other:?}"),
    }
}

/// Finds the first `impl` item in `src` and returns (algebra name, target
/// type, its first method, the enclosing `impl` item's own span).
fn lower_one_impl(src: &str) -> (String, Type, FnDecl, Span) {
    let program = lower_program(src);
    for item in program.items {
        if let ItemKind::Impl(d) = item.kind {
            let f = d.fns.into_iter().next().expect("expected at least one fn in the impl");
            return (d.algebra, d.target, f, item.span);
        }
    }
    panic!("no impl item found in {src:?}");
}

/// Finds the first *inherent* `impl` item in `src` and returns (target
/// type, its impl-level generics, its first method, the enclosing `impl`
/// item's own span).
fn lower_one_inherent_impl(src: &str) -> (Type, Vec<GenericParam>, FnDecl, Span) {
    let program = lower_program(src);
    for item in program.items {
        if let ItemKind::InherentImpl(d) = item.kind {
            let f = d.fns.into_iter().next().expect("expected at least one fn in the impl");
            return (d.target, d.generics, f, item.span);
        }
    }
    panic!("no inherent impl item found in {src:?}");
}

/// Like `lower_one_inherent_impl`, but returns *every* method of the impl
/// block, for `Infer::infer_inherent_impl_block` — needed for mutual-
/// recursion tests, which require at least two methods sharing one `Infer`.
fn lower_inherent_impl(src: &str) -> (Type, Vec<GenericParam>, Vec<FnDecl>, Span) {
    let program = lower_program(src);
    for item in program.items {
        if let ItemKind::InherentImpl(d) = item.kind {
            return (d.target, d.generics, d.fns, item.span);
        }
    }
    panic!("no inherent impl item found in {src:?}");
}

/// A **test-only** fixture standing in for a real stdlib â€” `infer_call` no
/// longer special-cases operator names at all (see its own doc comment), so
/// most of these tests need *some* declared algebra backing `add`/`lt`/`and`
/// to keep testing what they were actually written to test (unification,
/// generalization, scoping â€” not "is there a registry entry"). Deliberately
/// not the real stdlib design: the user flagged that whether primitives
/// should decompose along strict mathematical lines (`Semigroup`/`Monoid`/
/// `Group`/`Ring`/...) or a simpler grouping is still an open question to
/// come back to â€” this is just enough surface to keep the rest of the test
/// suite meaningful in the meantime.
fn builtin_registry() -> Registry {
    registry_from(
        "algebra Ring<T> {
            fn add(a: T, b: T) -> T;
            fn sub(a: T, b: T) -> T;
            fn mul(a: T, b: T) -> T;
            fn div(a: T, b: T) -> T;
            fn neg(a: T) -> T;
        }
        algebra Ord<T> {
            fn lt(a: T, b: T) -> bool;
            fn le(a: T, b: T) -> bool;
            fn gt(a: T, b: T) -> bool;
            fn ge(a: T, b: T) -> bool;
            fn eq(a: T, b: T) -> bool;
            fn neq(a: T, b: T) -> bool;
        }
        algebra Bool<T> {
            fn and(a: T, b: T) -> T;
            fn or(a: T, b: T) -> T;
            fn xor(a: T, b: T) -> T;
            fn implies(a: T, b: T) -> T;
        }
        algebra Int<T> {}
        algebra Float<T> {}
        impl Ring<i32> { fn add(a: i32, b: i32) -> i32 { a } fn sub(a: i32, b: i32) -> i32 { a } fn mul(a: i32, b: i32) -> i32 { a } fn div(a: i32, b: i32) -> i32 { a } fn neg(a: i32) -> i32 { a } }
        impl Ring<i64> { fn add(a: i64, b: i64) -> i64 { a } fn sub(a: i64, b: i64) -> i64 { a } fn mul(a: i64, b: i64) -> i64 { a } fn div(a: i64, b: i64) -> i64 { a } fn neg(a: i64) -> i64 { a } }
        impl Ring<f64> { fn add(a: f64, b: f64) -> f64 { a } fn sub(a: f64, b: f64) -> f64 { a } fn mul(a: f64, b: f64) -> f64 { a } fn div(a: f64, b: f64) -> f64 { a } fn neg(a: f64) -> f64 { a } }
        impl Ord<i32> { fn lt(a: i32, b: i32) -> bool { true } fn le(a: i32, b: i32) -> bool { true } fn gt(a: i32, b: i32) -> bool { true } fn ge(a: i32, b: i32) -> bool { true } fn eq(a: i32, b: i32) -> bool { true } fn neq(a: i32, b: i32) -> bool { true } }
        impl Ord<f64> { fn lt(a: f64, b: f64) -> bool { true } fn le(a: f64, b: f64) -> bool { true } fn gt(a: f64, b: f64) -> bool { true } fn ge(a: f64, b: f64) -> bool { true } fn eq(a: f64, b: f64) -> bool { true } fn neq(a: f64, b: f64) -> bool { true } }
        impl Bool<bool> { fn and(a: bool, b: bool) -> bool { a } fn or(a: bool, b: bool) -> bool { a } fn xor(a: bool, b: bool) -> bool { a } fn implies(a: bool, b: bool) -> bool { a } }
        impl Int<i32> {}
        impl Int<i64> {}
        impl Float<f32> {}
        impl Float<f64> {}",
    )
}

fn infer_src(src: &str) -> Ty {
    let f = lower_one_fn(src);
    let registry = builtin_registry();
    let mut infer = Infer::new(&registry);
    infer.infer_fn(&f).unwrap_or_else(|e| panic!("inference failed for {src:?}: {e:?}"))
}

#[test]
fn annotated_params_pin_the_body_type() {
    // `add` is a built-in-operator stand-in (see infer.rs module docs) that
    // requires both operands to unify to the same type.
    let ty = infer_src("fn f(a: f64, b: f64) -> f64 { a + b }");
    assert_eq!(ty, Ty::Con("f64".to_string()));
}

#[test]
fn mismatched_annotated_params_are_rejected() {
    let f = lower_one_fn("fn f(a: f64, b: i32) -> f64 { a + b }");
    let registry = Registry::default();
    let mut infer = Infer::new(&registry);
    assert!(infer.infer_fn(&f).is_err(), "f64 and i32 should not unify");
}

#[test]
fn unannotated_params_are_inferred_from_a_concrete_sibling() {
    // Neither `a` nor `b` is annotated, but the declared return type pins
    // the whole chain down through unification.
    let ty = infer_src("fn f(a, b) -> f64 { a + b }");
    assert_eq!(ty, Ty::Con("f64".to_string()));
}

#[test]
fn unconstrained_int_literal_defaults_to_i32() {
    let ty = infer_src("fn f() -> i32 { 1 }");
    assert_eq!(ty, Ty::Con("i32".to_string()));
}

#[test]
fn two_literals_with_conflicting_shapes_merged_by_a_shared_generic_are_rejected() {
    // A real bug, found by testing (reported directly): `1 + 2.0` â€” `add`'s
    // generic `T` forces both operands to the same type, merging the two
    // literals' own type variables into one. The old `apply_defaults`
    // resolved each `pending_defaults` entry independently, in encounter
    // order â€” `1`'s `Int` preference bound the shared variable to `i32`
    // first, and `2.0`'s own `Float` preference, checked *after* that bind
    // already made the variable concrete, was silently discarded rather
    // than compared against it at all. `2.0` becoming `i32` must be an
    // error, not a silent, order-dependent choice.
    let f = lower_one_fn("fn f() { 1 + 2.0 }");
    let registry = builtin_registry();
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    // `MissingImpl`, not `Unify` â€” caught by `2.0`'s own `Float` constraint
    // failing against whichever concrete type the merge settled on (`i32`,
    // from `1`'s own default), not by `apply_defaults` comparing the two
    // literals against each other directly â€” see `infer.rs`'s `NumberLit`
    // handling and `stdlib/num/num.cleave`.
    assert!(matches!(err.kind, TypeErrorKind::MissingImpl { .. }), "got: {:?}", err.kind);
}

#[test]
fn two_literals_with_the_same_shape_merged_by_a_shared_generic_still_default_fine() {
    // The non-conflicting counterpart â€” both literals prefer `Int`, so
    // merging them via `add`'s shared generic `T` must still succeed.
    let ty = infer_src("fn f() -> i32 { 1 + 2 }");
    assert_eq!(ty, Ty::Con("i32".to_string()));
}

#[test]
fn a_bare_int_shaped_literal_forced_into_a_float_context_is_now_rejected() {
    // This premise flipped once a literal's shape became a real constraint
    // (`Int`/`Float`, see `stdlib/num/num.cleave`): the language has no
    // implicit int/float conversions anywhere, so an undotted `1` genuinely
    // cannot become `f64` just because the context asks for it â€” unlike a
    // real *unification*-only fact (a declared type), which a literal's
    // mere shape-based *default preference* used to always lose to
    // silently. Now it's `Int t` vs. `no impl Int<f64>` â€” a real,
    // registry-checked conflict, not a defaulting nicety.
    let f = lower_one_fn("fn f() -> f64 { 1 }");
    let registry = builtin_registry();
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::MissingImpl { .. }), "got: {:?}", err.kind);
}

#[test]
fn a_dotted_literal_in_a_float_context_still_works() {
    // The counterpart proving the rejection above is about shape, not about
    // literals-in-general: `1.0` (dotted, `Float`-shaped) in the exact same
    // `f64` context is fine.
    let ty = infer_src("fn f() -> f64 { 1.0 }");
    assert_eq!(ty, Ty::Con("f64".to_string()));
}

#[test]
fn a_suffixed_literal_bypasses_the_shape_constraint_entirely() {
    // `1:f64` is pinned directly to `Con("f64")` at the AST level (see
    // `NumberLit`'s `Some(suffix)` branch) â€” no fresh variable, no `Int`
    // constraint ever generated for it, so there's nothing to conflict with
    // regardless of the surrounding context's own shape expectations.
    let ty = infer_src("fn f() -> f64 { 1:f64 }");
    assert_eq!(ty, Ty::Con("f64".to_string()));
}

#[test]
fn unconstrained_literal_with_no_annotation_defaults_by_shape() {
    // No return-type annotation at all â€” still defaults via the literal's
    // own text (no '.' => int default), exactly like Haskell's defaulting.
    let f = lower_one_fn("fn f() { 1 }");
    let registry = Registry::default();
    let mut infer = Infer::new(&registry);
    let ty = infer.infer_fn(&f).unwrap();
    assert_eq!(ty, Ty::Con("i32".to_string()));

    let f2 = lower_one_fn("fn f() { 1.0 }");
    let registry2 = Registry::default();
    let mut infer2 = Infer::new(&registry2);
    let ty2 = infer2.infer_fn(&f2).unwrap();
    assert_eq!(ty2, Ty::Con("f32".to_string()));
}

#[test]
fn suffixed_literal_is_pinned_directly() {
    let ty = infer_src("fn f() { 1:i64 }");
    assert_eq!(ty, Ty::Con("i64".to_string()));
}

#[test]
fn comparison_ops_produce_bool_and_require_matching_operands() {
    let ty = infer_src("fn f(a: i32, b: i32) -> bool { a < b }");
    assert_eq!(ty, Ty::Con("bool".to_string()));
}

#[test]
fn if_branches_must_unify() {
    let ty = infer_src("fn f(c: bool) -> i32 { if c { 1 } else { 2 } }");
    assert_eq!(ty, Ty::Con("i32".to_string()));
}

#[test]
fn if_branches_with_mismatched_types_are_rejected() {
    let f = lower_one_fn("fn f(c: bool) { if c { 1:i32 } else { 1.0:f64 } }");
    let registry = Registry::default();
    let mut infer = Infer::new(&registry);
    assert!(infer.infer_fn(&f).is_err(), "i32 and f64 branches should not unify");
}

#[test]
fn let_binding_propagates_the_value_type() {
    let ty = infer_src("fn f() -> f64 { let x = 1.0; x }");
    assert_eq!(ty, Ty::Con("f64".to_string()));
}

#[test]
fn logical_ops_require_bool_operands() {
    let ty = infer_src("fn f(a: bool, b: bool) -> bool { a and b }");
    assert_eq!(ty, Ty::Con("bool".to_string()));
}

#[test]
fn lambda_infers_a_function_type() {
    let ty = infer_src("fn f() { fn(a: f64, b: f64) { a + b } }");
    assert_eq!(ty, Ty::Fn(vec![Ty::Con("f64".to_string()), Ty::Con("f64".to_string())], Box::new(Ty::Con("f64".to_string()))));
}

#[test]
fn explicit_generic_bound_constrains_a_parameter_with_no_other_evidence() {
    // `x` is never used in any operation that would otherwise imply a
    // constraint (bare `{ x }`, no arithmetic, no literal) â€” `<T: Int>` is
    // the *only* source of the `Int` requirement here. Before
    // `FnDecl::generics` was wired into inference, this bound parsed but
    // was read nowhere: `T` resolved as a bogus concrete type literally
    // named `"T"`, and the bound was silently dropped.
    let f = lower_one_fn("fn f<T: Int>(x: T) -> f32 { x }");
    let registry = builtin_registry();
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    assert!(
        matches!(&err.kind, TypeErrorKind::MissingImpl { algebra, .. } if algebra == "Int"),
        "got: {:?}",
        err.kind
    );
}

#[test]
fn explicit_generic_bound_accepts_a_satisfying_type() {
    let ty = infer_src("fn f<T: Int>(x: T) -> i32 { x }");
    assert_eq!(ty, Ty::Con("i32".to_string()));
}

#[test]
fn a_declared_generic_type_parameter_without_a_bound_stays_unconstrained() {
    // Contrast with the two tests above: `<T>` with no `: Bound` at all
    // still resolves `T` to a real fresh variable (not the bogus literal
    // `Con("T")` from before this was wired up), but pushes no constraint â€”
    // `f32`, which does not implement `Int`, is accepted fine.
    let ty = infer_src("fn f<T>(x: T) -> f32 { x }");
    assert_eq!(ty, Ty::Con("f32".to_string()));
}

// ---------------------------------------------------------------------
// algebra-bound inheritance (`algebra Int<T> : Num`): an impl of the
// *narrower* algebra alone satisfies a bound on the *wider* one too, with
// no separate impl needed -- mirrors `stdlib/num/num.cleave`'s own design.
// ---------------------------------------------------------------------

#[test]
fn algebra_bound_inheritance_lets_a_narrower_impl_satisfy_a_wider_bound() {
    let registry = registry_from(
        "algebra Num<T> {}
         algebra Int<T> : Num {}
         impl Int<i32> {}",
    );
    let f = lower_one_fn("fn f<T: Num>(x: T) -> i32 { x }");
    let mut infer = Infer::new(&registry);
    let ty = infer.infer_fn(&f).unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(ty, Ty::Con("i32".to_string()));
}

#[test]
fn algebra_bound_inheritance_does_not_apply_to_an_unrelated_type() {
    // `f64` has no `impl Int<f64>` and no `impl Num<f64>` either -- the
    // bound must not accept it just because *some* type satisfies `Num`
    // through inheritance.
    let registry = registry_from(
        "algebra Num<T> {}
         algebra Int<T> : Num {}
         impl Int<i32> {}",
    );
    let f = lower_one_fn("fn f<T: Num>(x: T) -> f64 { x }");
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    assert!(matches!(&err.kind, TypeErrorKind::MissingImpl { algebra, .. } if algebra == "Num"), "got: {:?}", err.kind);
}

#[test]
fn cyclic_algebra_bounds_reject_cleanly_instead_of_looping_forever() {
    let registry = registry_from(
        "algebra A<T> : B {}
         algebra B<T> : A {}",
    );
    let f = lower_one_fn("fn f<T: A>(x: T) -> i32 { x }");
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    assert!(matches!(&err.kind, TypeErrorKind::MissingImpl { algebra, .. } if algebra == "A"), "got: {:?}", err.kind);
}

#[test]
fn a_two_ingredient_algebra_is_satisfied_by_impls_of_both_ingredients_alone() {
    // The mirror image of `algebra_bound_inheritance_lets_a_narrower_impl_
    // satisfy_a_wider_bound` above: there, the concrete impl sits on the
    // *more specific* side (`Int<i32>` witnessing `Num<i32>`). Here it sits
    // on the *more general* side(s) instead -- `AdditiveMonoid<i32>` and
    // `MultiplicativeMonoid<i32>` together witness `Semiring<i32>`, with no
    // separate (even empty) `impl Semiring<i32>` anywhere.
    let registry = registry_from(
        "algebra AdditiveMonoid<T> { fn add(a: T, b: T) -> T; }
         algebra MultiplicativeMonoid<T> { fn mul(a: T, b: T) -> T; }
         algebra Semiring<T> : AdditiveMonoid + MultiplicativeMonoid {}
         impl AdditiveMonoid<i32> { fn add(a, b) { a } }
         impl MultiplicativeMonoid<i32> { fn mul(a, b) { a } }",
    );
    let f = lower_one_fn("fn f<T: Semiring>(x: T) -> i32 { x }");
    let mut infer = Infer::new(&registry);
    let ty = infer.infer_fn(&f).unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(ty, Ty::Con("i32".to_string()));
}

#[test]
fn a_single_bound_algebra_does_not_aggregate_and_cannot_be_satisfied_by_its_own_parent() {
    // Same shape as the two-ingredient case above, minus one ingredient --
    // deliberately must *not* work the same way. A single bound is a
    // rename/specialization relationship (already covered by the reverse-
    // witness direction when checking the *parent*), not a composition; the
    // forward-aggregate direction is gated on 2+ bounds specifically to
    // avoid exactly this collapsing into "checking `Semiring<i32>` is the
    // same as checking `AdditiveMonoid<i32>`" if it fired for one bound too.
    let registry = registry_from(
        "algebra AdditiveMonoid<T> { fn add(a: T, b: T) -> T; }
         algebra Semiring<T> : AdditiveMonoid {}
         impl AdditiveMonoid<i32> { fn add(a, b) { a } }",
    );
    let f = lower_one_fn("fn f<T: Semiring>(x: T) -> i32 { x }");
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    assert!(
        matches!(&err.kind, TypeErrorKind::MissingImpl { algebra, .. } if algebra == "Semiring"),
        "got: {:?}",
        err.kind
    );
}

#[test]
fn sibling_algebras_sharing_one_parent_bound_do_not_leak_into_each_other() {
    // The actual regression, found by testing right after the naive
    // "non-empty bounds -> aggregate" version of this fix landed: `Int` and
    // `Float` are siblings, each with a *single* bound on `Num`. A version
    // that aggregated through any non-empty bound list let `i8` (a real
    // `Int`) also satisfy a `Float` query, since `Num<i8>` holds (via
    // `Int`) and a single-bound aggregate for `Float` couldn't distinguish
    // that from `Num<i8>` holding via `Float` itself. Mirrors `stdlib/num/
    // num.cleave`'s own real shape exactly (see `tests/stdlib.rs`'s own
    // equivalent check against the real stdlib file).
    let registry = registry_from(
        "algebra Num<T> {}
         algebra Int<T> : Num {}
         algebra Float<T> : Num {}
         impl Int<i8> {}",
    );
    let f = lower_one_fn("fn f<T: Float>(x: T) -> i8 { x }");
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    assert!(
        matches!(&err.kind, TypeErrorKind::MissingImpl { algebra, .. } if algebra == "Float"),
        "got: {:?}",
        err.kind
    );
}

#[test]
fn lambda_declared_return_type_constrains_the_body() {
    let f = lower_one_fn("fn f() { fn(a: f64, b: i32) -> f64 { a } }");
    let registry = Registry::default();
    let mut infer = Infer::new(&registry);
    let ty = infer.infer_fn(&f).unwrap();
    match ty {
        Ty::Fn(params, ret) => {
            assert_eq!(params, vec![Ty::Con("f64".to_string()), Ty::Con("i32".to_string())]);
            assert_eq!(*ret, Ty::Con("f64".to_string()));
        }
        other => panic!("expected a Ty::Fn, got {other:?}"),
    }
}

#[test]
fn let_bound_lambda_is_callable_by_name() {
    let ty = infer_src("fn f() -> f64 { let g = fn(a: f64, b: f64) { a + b }; g(1.0, 2.0) }");
    assert_eq!(ty, Ty::Con("f64".to_string()));
}

#[test]
fn calling_a_bound_lambda_with_wrong_arity_is_rejected() {
    let f = lower_one_fn("fn f() { let g = fn(a: f64, b: f64) { a }; g(1.0) }");
    let registry = Registry::default();
    let mut infer = Infer::new(&registry);
    assert!(infer.infer_fn(&f).is_err(), "calling a 2-arg lambda with 1 arg must fail");
}

#[test]
fn calling_a_bound_lambda_with_wrong_arg_type_is_rejected() {
    let f = lower_one_fn("fn f() { let g = fn(a: f64, b: f64) { a }; g(1.0, true) }");
    let registry = Registry::default();
    let mut infer = Infer::new(&registry);
    assert!(infer.infer_fn(&f).is_err());
}

#[test]
fn lambda_params_do_not_leak_into_the_enclosing_scope() {
    // `a` inside the lambda must not become visible after it â€” a lambda
    // introduces its own scope, same as any other block.
    let f = lower_one_fn("fn f() { let g = fn(a: f64) { a }; a }");
    let registry = Registry::default();
    let mut infer = Infer::new(&registry);
    assert!(infer.infer_fn(&f).is_err(), "`a` must be unbound outside the lambda");
}

#[test]
fn let_inside_if_branch_does_not_leak_into_enclosing_scope() {
    // Regression test for the env-scoping bug found while adding lambdas:
    // a `let` inside an `if`-branch must not remain visible afterward.
    let f = lower_one_fn("fn f(c: bool) { if c { let x = 1; x } else { 0 }; x }");
    let registry = Registry::default();
    let mut infer = Infer::new(&registry);
    assert!(infer.infer_fn(&f).is_err(), "`x` must be unbound outside the if-branch that declared it");
}

#[test]
fn let_bound_lambda_is_generalized_and_usable_at_two_types() {
    // The textbook example: without generalization, `id`'s single type
    // variable gets pinned by the first call and the second call fails.
    let ty = infer_src("fn f() -> bool { let id = fn(x) { x }; let _a = id(1.0); id(true) }");
    assert_eq!(ty, Ty::Con("bool".to_string()));
}

#[test]
fn generalization_does_not_depend_on_call_order() {
    // Same program, calls in the other order â€” must behave identically,
    // since global-scope-style independence from order was exactly the
    // point of building real schemes instead of a single shared type var.
    let ty = infer_src("fn f() -> f64 { let id = fn(x) { x }; let _a = id(true); id(1.0) }");
    assert_eq!(ty, Ty::Con("f64".to_string()));
}

#[test]
fn let_mut_lambda_is_not_generalized() {
    // This checks the *syntactic rule* fires (declared `mut` => trivial
    // scheme) â€” `id` is never reassigned here, so this does not exercise
    // reassignment itself; see `let_mut_reassignment_is_checked_against_the_original_type`
    // for the actual scenario the rule protects against. Uses an explicit
    // `:f64` suffix on the first call deliberately â€” an unsuffixed literal's
    // type variable is its own separate, unrelated gap (see `infer_call`'s
    // doc comment) that would otherwise mask what this test is checking.
    let f = lower_one_fn("fn f() { let mut id = fn(x) { x }; let _a = id(1.0:f64); id(true) }");
    let registry = Registry::default();
    let mut infer = Infer::new(&registry);
    assert!(infer.infer_fn(&f).is_err(), "a `let mut` lambda must not be usable at two different types");
}

#[test]
fn let_mut_reassignment_is_checked_against_the_original_type() {
    // The actual scenario the "never generalize `mut`" rule exists for:
    // reassigning `id` to a lambda with an incompatible shape must be
    // rejected against the *original* (necessarily monomorphic) type â€” if
    // `id` had been generalized instead, `instantiate` would hand this
    // `Assign` a fresh, disconnected copy of the scheme and the mismatch
    // below would go undetected.
    let f = lower_one_fn(
        "fn f() { let mut id = fn(x: f64) { x }; id = fn(x: bool, y: bool) { x }; id(1.0) }",
    );
    let registry = Registry::default();
    let mut infer = Infer::new(&registry);
    assert!(infer.infer_fn(&f).is_err(), "reassigning `id` to an incompatible function type must be rejected");
}

fn registry_from(src: &str) -> Registry {
    Registry::build(&lower_program(src))
}

#[test]
fn operator_resolves_against_a_single_declared_algebra() {
    let registry = registry_from(
        "algebra Ring<T> { fn add(a: T, b: T) -> T; }
         impl Ring<i32> { fn add(a: i32, b: i32) -> i32 { a } }",
    );
    let f = lower_one_fn("fn f(a: i32, b: i32) -> i32 { a + b }");
    let mut infer = Infer::new(&registry);
    let ty = infer.infer_fn(&f).unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(ty, Ty::Con("i32".to_string()));
}

#[test]
fn operator_call_without_a_matching_impl_is_rejected() {
    // `Ring` is declared and owns `add`, but no `impl Ring<bool>` exists.
    let registry = registry_from(
        "algebra Ring<T> { fn add(a: T, b: T) -> T; }
         impl Ring<i32> { fn add(a: i32, b: i32) -> i32 { a } }",
    );
    let f = lower_one_fn("fn f(a: bool, b: bool) -> bool { a + b }");
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::MissingImpl { .. }), "got: {:?}", err.kind);
}

#[test]
fn missing_impl_against_unit_hints_at_a_discarded_tail() {
    // The exact real-world shape (the actual bug found while building
    // whole-program recursion): a self-recursive `fn` whose `if`-expression
    // is followed by a stray `;`, discarding its value â€” `f`'s own inferred
    // return type collapses to `()` (via the self-reference's `ret_var`
    // tie-back, unified with the block's own "no tail" result), and that
    // `()` then fails the `Num` constraint a bare literal generated
    // elsewhere in the same, now-merged, equivalence class. The error should
    // point at the actual cause, not just report a bare, opaque
    // `no impl Num<()>`.
    let registry = registry_from(
        "algebra Ring<T> { fn add(a: T, b: T) -> T; }
         algebra Ord<T> { fn gt(a: T, b: T) -> bool; }
         algebra Num<T> {}
         impl Ring<i32> { fn add(a: i32, b: i32) -> i32 { a } }
         impl Ord<i32> { fn gt(a: i32, b: i32) -> bool { true } }
         impl Num<i32> {}",
    );
    let f = lower_one_fn("fn f(x) { if x > 0 { f(x) + 1 } else { x }; }");
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    let message = err.kind.to_string();
    assert!(message.contains("no `impl") && message.contains("<()>`"), "got: {message}");
    assert!(message.contains("discarding its value"), "got: {message}");
}

#[test]
fn unify_mismatch_against_a_found_unit_hints_at_a_discarded_tail() {
    let f = lower_one_fn("fn f() -> i32 { let x = 1; }");
    let registry = Registry::default();
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    let message = err.kind.to_string();
    assert!(message.contains("expected `i32`, found `()`"), "got: {message}");
    assert!(message.contains("discarding its value"), "got: {message}");
}

#[test]
fn unify_mismatch_against_an_expected_unit_does_not_hint() {
    // The reverse direction â€” `()` is what the *context* required (an `if`
    // with no `else`), not what a discarded tail produced. Not the situation
    // the hint is about.
    // A suffixed literal (`1:i32`) is pinned directly to `Con("i32")` (no
    // intermediate type variable) â€” needed for a genuine `Mismatch`, since
    // unifying `()` against a bare, still-unbound variable just binds it
    // (no error at all).
    let f = lower_one_fn("fn f(c: bool) { if c { 1:i32 } }");
    let registry = Registry::default();
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    let message = err.kind.to_string();
    assert!(message.contains("expected `()`, found `i32`"), "got: {message}");
    assert!(!message.contains("discarding its value"), "got: {message}");
}

#[test]
fn operator_ambiguous_between_two_algebras_is_rejected() {
    // Two independent, legitimately-scoped algebras both declaring `add` â€”
    // not a "someone's overriding" signal, an ordinary name collision (see
    // conversation notes / `TypeErrorKind::AmbiguousOperator`'s doc comment).
    let registry = registry_from(
        "algebra Ring<T> { fn add(a: T, b: T) -> T; }
         algebra Tropical<T> { fn add(a: T, b: T) -> T; }",
    );
    let f = lower_one_fn("fn f(a: i32, b: i32) -> i32 { a + b }");
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    match err.kind {
        TypeErrorKind::AmbiguousOperator { candidates, .. } => {
            let mut c = candidates;
            c.sort();
            assert_eq!(c, vec!["Ring".to_string(), "Tropical".to_string()]);
        }
        other => panic!("expected AmbiguousOperator, got {other:?}"),
    }
}

#[test]
fn algebra_generic_parameter_is_instantiated_fresh_not_treated_as_concrete() {
    // `T` in `algebra Ring<T> { fn add(a: T, b: T) -> T; }` must become a
    // fresh type variable per call, not a literal concrete type named "T" â€”
    // otherwise this would reject `f64` args as "expected T, found f64".
    let registry = registry_from(
        "algebra Ring<T> { fn add(a: T, b: T) -> T; }
         impl Ring<f64> { fn add(a: f64, b: f64) -> f64 { a } }",
    );
    let f = lower_one_fn("fn f(a: f64, b: f64) -> f64 { a + b }");
    let mut infer = Infer::new(&registry);
    let ty = infer.infer_fn(&f).unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(ty, Ty::Con("f64".to_string()));
}

#[test]
fn numeric_literal_rejected_when_unified_with_a_non_numeric_type_and_num_is_registered() {
    // The original motivating example: without a real `Num` constraint,
    // `1`'s type variable happily unifies with `bool` and nothing ever
    // catches it (see `infer_call`'s doc comment on the built-in fallback).
    // With `Num` actually registered, this must now be rejected. `Ring<bool>`
    // is declared too so `add` itself resolves fine â€” the rejection here
    // must come specifically from `Num`, not from `Ring` also lacking `bool`.
    let registry = registry_from(
        "algebra Num<T> {} impl Num<i32> {}
         algebra Ring<T> { fn add(a: T, b: T) -> T; }
         impl Ring<bool> { fn add(a: bool, b: bool) -> bool { a } }",
    );
    let f = lower_one_fn("fn f(a: bool) -> bool { let x = 1; x + a }");
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    assert!(matches!(&err.kind, TypeErrorKind::MissingImpl { algebra, .. } if algebra == "Num"), "got: {:?}", err.kind);
}

#[test]
fn numeric_literal_accepted_when_unified_with_a_registered_numeric_type() {
    let registry = registry_from(
        "algebra Num<T> {} impl Num<i32> {}
         algebra Ring<T> { fn add(a: T, b: T) -> T; }
         impl Ring<i32> { fn add(a: i32, b: i32) -> i32 { a } }",
    );
    let f = lower_one_fn("fn f(a: i32) -> i32 { let x = 1; x + a }");
    let mut infer = Infer::new(&registry);
    let ty = infer.infer_fn(&f).unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(ty, Ty::Con("i32".to_string()));
}

#[test]
fn generalized_lambda_carries_its_algebra_constraint_to_call_sites() {
    // The actual "qualified types" payoff: `x`'s own constraint (`T: Ring`,
    // generated from `x + x` inside the lambda) travels into `g`'s scheme
    // via `generalize`, then re-attaches fresh at each call site via
    // `instantiate` â€” so calling `g` with a type that has no `impl Ring`
    // is caught here, not silently accepted.
    let registry = registry_from(
        "algebra Ring<T> { fn add(a: T, b: T) -> T; }
         impl Ring<i32> { fn add(a: i32, b: i32) -> i32 { a } }",
    );
    let f = lower_one_fn("fn f() { let g = fn(x) { x + x }; g(true) }");
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::MissingImpl { .. }), "got: {:?}", err.kind);
}

#[test]
fn generalized_lambda_constraint_allows_a_type_with_a_real_impl() {
    let registry = registry_from(
        "algebra Ring<T> { fn add(a: T, b: T) -> T; }
         impl Ring<i32> { fn add(a: i32, b: i32) -> i32 { a } }",
    );
    let f = lower_one_fn("fn f() -> i32 { let g = fn(x) { x + x }; g(1:i32) }");
    let mut infer = Infer::new(&registry);
    let ty = infer.infer_fn(&f).unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(ty, Ty::Con("i32".to_string()));
}

#[test]
fn operator_with_zero_candidates_is_rejected_not_permissive() {
    // No algebra at all declares `add` here. It resolves internally to an
    // honest `<unresolved-call:add>` placeholder rather than silently
    // succeeding via the old permissive built-in stand-in (see `infer_call`'s
    // doc comment) â€” but that placeholder surviving all the way to `f`'s own
    // exposed return type must itself be an error, not a quietly "successful"
    // inference (see `infer_fn`'s check, found by running the CLI on a real
    // file where exactly this slipped through silently).
    let registry = registry_from("algebra Unrelated<T> { fn neg(a: T) -> T; }");
    let f = lower_one_fn("fn f(a: f64, b: f64) { a + b }");
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::Unresolved(ref p) if p == "<unresolved-call:add>"), "got: {:?}", err.kind);
}

#[test]
fn unresolved_call_surviving_to_the_final_type_is_rejected() {
    // The exact scenario found by hand: a lambda calling another top-level
    // `fn` that isn't itself inferred yet (no cross-function inference, see
    // module docs) produces `(t) -> <unresolved-call:add>` as the lambda's
    // own type â€” this used to be returned as if inference had succeeded.
    let f = lower_one_fn("fn f() { let g = fn(x) { add(x, 1) }; g(1) }");
    let registry = Registry::default();
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::Unresolved(_)), "got: {:?}", err.kind);
}

#[test]
fn unresolved_call_that_gets_overridden_by_a_conflict_is_still_an_error() {
    // If something concrete conflicts with the placeholder before `infer_fn`
    // returns, that conflict is reported as an ordinary `Unify` mismatch â€”
    // the new end-of-function check only ever *adds* a safety net for the
    // case nothing else already caught, it doesn't change this pre-existing
    // behavior.
    let f = lower_one_fn("fn f() -> i32 { let a = undeclared_fn(1); a + 1.0:f64 }");
    let registry = builtin_registry();
    let mut infer = Infer::new(&registry);
    assert!(infer.infer_fn(&f).is_err());
}

#[test]
fn number_literal_defaulting_still_works_once_generalized() {
    // `1`'s variable is now generalized right alongside `x`'s (see
    // `generalize`'s doc comment) â€” this confirms defaulting the resulting
    // orphaned scheme-internal variable afterward is still harmless and
    // this still resolves to i32 via the declared return type, without a
    // `Num` algebra registered to check against (empty registry here).
    let ty = infer_src("fn f() -> i32 { let f2 = fn(x) { x + 1 }; f2(1) }");
    assert_eq!(ty, Ty::Con("i32".to_string()));
}

#[test]
fn function_with_a_bare_literal_is_usable_at_multiple_numeric_types() {
    // The bug an earlier version of `generalize` had: excluding a number
    // literal's variable from generalization also excluded `x`'s own
    // genericity, since `x + 1` unifies them into the same variable â€”
    // forcing `add_one` monomorphic the instant a bare literal appeared in
    // its body. This is exactly the C++/Rust generic-numeric-literal pain
    // (`num_traits::One`-style boilerplate) â€” fixed by generalizing the
    // literal's variable too, with its `Num` constraint riding along.
    let registry = registry_from(
        "algebra Ring<T> { fn add(a: T, b: T) -> T; }
         impl Ring<i32> { fn add(a: i32, b: i32) -> i32 { a } }
         impl Ring<f64> { fn add(a: f64, b: f64) -> f64 { a } }",
    );
    let f = lower_one_fn(
        "fn f() { let add_one = fn(x) { x + 1 }; let a = add_one(1.0:f64); add_one(1:i32) }",
    );
    let mut infer = Infer::new(&registry);
    let ty = infer.infer_fn(&f).unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(ty, Ty::Con("i32".to_string()));
}

#[test]
fn calling_a_non_function_binding_is_not_callable() {
    let f = lower_one_fn("fn f() { let x = 1.0; x(2.0) }");
    let registry = Registry::default();
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::NotCallable(_)), "got: {:?}", err.kind);
}

#[test]
fn unknown_identifier_is_reported() {
    let f = lower_one_fn("fn f() { y }");
    let registry = Registry::default();
    let mut infer = Infer::new(&registry);
    assert!(infer.infer_fn(&f).is_err(), "unbound identifier `y` must be rejected");
}

#[test]
fn param_types_are_resolved_for_unannotated_params() {
    let f = lower_one_fn("fn f(a, b) -> f64 { a + b }");
    let registry = builtin_registry();
    let mut infer = Infer::new(&registry);
    infer.infer_fn(&f).unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(infer.param_types, vec![Ty::Con("f64".to_string()), Ty::Con("f64".to_string())]);
}

#[test]
fn node_types_records_every_subexpression_fully_resolved() {
    // `a`'s own `Path` node isn't annotated (unannotated param), so its
    // recorded type only becomes concrete via the *later* unification
    // inside `add` â€” confirms `infer_fn` re-resolves `node_types` through
    // the final `subst` rather than handing back whatever was captured at
    // the moment each node was first visited.
    let f = lower_one_fn("fn f(a) -> f64 { a }");
    let registry = Registry::default();
    let mut infer = Infer::new(&registry);
    infer.infer_fn(&f).unwrap_or_else(|e| panic!("{e:?}"));
    let tail = f.body.as_ref().unwrap().tail.as_deref().expect("expected a tail expression");
    assert_eq!(infer.node_types.get(&tail.id), Some(&Ty::Con("f64".to_string())));
}

#[test]
fn impl_method_unannotated_params_are_seeded_from_the_algebra_signature() {
    // The exact bug found by hand: `impl TestAlg<i32> { fn add(x, y) {...} }`
    // with no annotations at all used to infer `x`/`y` as bare, disconnected
    // type variables â€” nothing tied them to `i32` even though the enclosing
    // `impl` unambiguously determines them via the algebra's own signature.
    let registry = registry_from("algebra TestAlg<T> { fn add(x: T, y: T) -> T; }");
    let (algebra, target, f, span) = lower_one_impl(
        "algebra TestAlg<T> { fn add(x: T, y: T) -> T; }
         impl TestAlg<i32> { fn add(x, y) { x } }",
    );
    let mut infer = Infer::new(&registry);
    let ret = infer.infer_impl_fn(&algebra, &target, &f, span).unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(infer.param_types, vec![Ty::Con("i32".to_string()), Ty::Con("i32".to_string())]);
    assert_eq!(ret, Ty::Con("i32".to_string()));
}

#[test]
fn impl_method_annotation_conflicting_with_the_algebra_is_rejected() {
    // The impl annotates its own params â€” but the algebra (via this impl's
    // target type) expects `i32`, not `f64`. The algebra is the single
    // source of truth; an explicit annotation is checked against it, never
    // trusted as an independent second truth that could silently diverge.
    let registry = registry_from("algebra TestAlg<T> { fn add(x: T, y: T) -> T; }");
    let (algebra, target, f, span) = lower_one_impl(
        "algebra TestAlg<T> { fn add(x: T, y: T) -> T; }
         impl TestAlg<i32> { fn add(x: f64, y: f64) -> f64 { x } }",
    );
    let mut infer = Infer::new(&registry);
    assert!(infer.infer_impl_fn(&algebra, &target, &f, span).is_err());
}

#[test]
fn impl_method_not_declared_by_the_algebra_is_rejected() {
    let registry = registry_from("algebra TestAlg<T> { fn add(x: T, y: T) -> T; }");
    let (algebra, target, f, span) = lower_one_impl(
        "algebra TestAlg<T> { fn add(x: T, y: T) -> T; }
         impl TestAlg<i32> { fn multiply(x: i32, y: i32) -> i32 { x } }",
    );
    let mut infer = Infer::new(&registry);
    let err = infer.infer_impl_fn(&algebra, &target, &f, span).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::NotDeclaredByAlgebra { .. }), "got: {:?}", err.kind);
}

#[test]
fn impl_method_arity_mismatch_against_the_algebra_is_rejected() {
    let registry = registry_from("algebra TestAlg<T> { fn add(x: T, y: T) -> T; }");
    let (algebra, target, f, span) = lower_one_impl(
        "algebra TestAlg<T> { fn add(x: T, y: T) -> T; }
         impl TestAlg<i32> { fn add(x: i32, y: i32, z: i32) -> i32 { x } }",
    );
    let mut infer = Infer::new(&registry);
    let err = infer.infer_impl_fn(&algebra, &target, &f, span).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::ArityMismatch { .. }), "got: {:?}", err.kind);
}

// ---------------------------------------------------------------------
// attributes (`#[mlir(...)]`) and bodyless `fn`s
// ---------------------------------------------------------------------

// A bodyless top-level `fn` is validated by `callgraph::infer_program`
// itself, not by `infer_fn` directly (which has no `Span` to report against
// other than the body it doesn't have — see `infer_fn_raw`'s own doc
// comment) — see `tests/callgraph.rs`'s
// `a_bodyless_top_level_fn_is_rejected`.

#[test]
fn a_bodyless_inherent_method_is_rejected() {
    let (target, generics, f, span) =
        lower_one_inherent_impl("struct Vec2 { x: f64, y: f64 } impl struct Vec2 { fn len(v) -> f64; }");
    let registry = builtin_registry();
    let mut infer = Infer::new(&registry);
    let err = infer.infer_inherent_impl_fn_generic(&cleave::infer::Env::new(), &generics, &target, &f, span).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::MissingFnBody { .. }), "got: {:?}", err.kind);
}

#[test]
fn a_bodyless_algebra_impl_method_with_no_attribute_is_rejected() {
    let registry = registry_from("algebra TestAlg<T> { fn add(x: T, y: T) -> T; }");
    let (algebra, target, f, span) = lower_one_impl(
        "algebra TestAlg<T> { fn add(x: T, y: T) -> T; }
         impl TestAlg<i32> { fn add(x: i32, y: i32) -> i32; }",
    );
    let mut infer = Infer::new(&registry);
    let err = infer.infer_impl_fn(&algebra, &target, &f, span).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::MissingIntrinsicAttribute { .. }), "got: {:?}", err.kind);
}

#[test]
fn a_bodyless_algebra_impl_method_with_mlir_attribute_type_checks() {
    let registry = registry_from("algebra TestAlg<T> { fn add(x: T, y: T) -> T; }");
    let (algebra, target, f, span) = lower_one_impl(
        "algebra TestAlg<T> { fn add(x: T, y: T) -> T; }
         impl TestAlg<i32> {
             #[mlir(mlir_i32_add)]
             fn add(x: i32, y: i32) -> i32;
         }",
    );
    let mut infer = Infer::new(&registry);
    let ty = infer.infer_impl_fn(&algebra, &target, &f, span).unwrap();
    assert_eq!(ty, Ty::Con("i32".to_string()));
}

// ---------------------------------------------------------------------
// structs: construction (`Vec2(x: 1.0, y: 2.0)`) and field access (`v.x`)
// ---------------------------------------------------------------------

/// Lowers a whole program (struct/algebra/impl declarations plus a `fn`
/// named `f`), builds a `Registry` from it, and infers `f` in isolation â€”
/// needed here (rather than `lower_one_fn`/`infer_src`) since these tests
/// need the registry to actually know about a declared `struct`, not just an
/// `algebra`.
fn infer_fn_named(src: &str, name: &str) -> Result<Ty, cleave::infer::TypeError> {
    let program = lower_program(src);
    let registry = Registry::build(&program);
    let f = program
        .items
        .iter()
        .find_map(|item| match &item.kind {
            ItemKind::Fn(f) if f.name == name => Some(f.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no fn named `{name}` in {src:?}"));
    let mut infer = Infer::new(&registry);
    infer.infer_fn(&f)
}

#[test]
fn struct_lit_constructs_and_returns_the_struct_type() {
    let ty = infer_fn_named(
        "struct Vec2 { x: f64, y: f64 }
         fn f() -> Vec2 { Vec2(x: 1.0, y: 2.0) }",
        "f",
    )
    .unwrap();
    assert_eq!(ty, Ty::Con("Vec2".to_string()));
}

#[test]
fn struct_lit_field_order_does_not_need_to_match_declaration_order() {
    let ty = infer_fn_named(
        "struct Vec2 { x: f64, y: f64 }
         fn f() -> Vec2 { Vec2(y: 2.0, x: 1.0) }",
        "f",
    )
    .unwrap();
    assert_eq!(ty, Ty::Con("Vec2".to_string()));
}

#[test]
fn struct_lit_field_value_is_checked_against_its_declared_type() {
    let err = infer_fn_named(
        "struct Vec2 { x: f64, y: f64 }
         fn f() -> Vec2 { Vec2(x: true, y: 2.0) }",
        "f",
    )
    .unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::Unify(_)), "got: {:?}", err.kind);
}

#[test]
fn struct_lit_naming_an_unknown_struct_is_rejected() {
    let err = infer_fn_named("fn f() { Vec2(x: 1.0, y: 2.0) }", "f").unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::UnknownStruct(_)), "got: {:?}", err.kind);
}

#[test]
fn struct_lit_missing_a_declared_field_is_rejected() {
    let err = infer_fn_named(
        "struct Vec2 { x: f64, y: f64 }
         fn f() -> Vec2 { Vec2(x: 1.0) }",
        "f",
    )
    .unwrap_err();
    assert!(
        matches!(&err.kind, TypeErrorKind::MissingField { field, .. } if field == "y"),
        "got: {:?}",
        err.kind
    );
}

#[test]
fn struct_lit_with_an_undeclared_field_name_is_rejected() {
    let err = infer_fn_named(
        "struct Vec2 { x: f64, y: f64 }
         fn f() -> Vec2 { Vec2(x: 1.0, y: 2.0, z: 3.0) }",
        "f",
    )
    .unwrap_err();
    assert!(
        matches!(&err.kind, TypeErrorKind::NoSuchField { field, .. } if field == "z"),
        "got: {:?}",
        err.kind
    );
}

#[test]
fn struct_lit_with_a_duplicate_field_is_rejected() {
    let err = infer_fn_named(
        "struct Vec2 { x: f64, y: f64 }
         fn f() -> Vec2 { Vec2(x: 1.0, x: 2.0, y: 3.0) }",
        "f",
    )
    .unwrap_err();
    assert!(
        matches!(&err.kind, TypeErrorKind::DuplicateField { field, .. } if field == "x"),
        "got: {:?}",
        err.kind
    );
}

#[test]
fn generic_struct_lit_infers_its_own_type_argument_from_field_values() {
    // No explicit `<f64>` written anywhere at the construction site â€” `T`'s
    // concrete type is inferred from `a`/`b`'s own values, exactly like an
    // algebra call's own generic parameter is inferred from its arguments.
    let ty = infer_fn_named(
        "struct Pair<T> { a: T, b: T }
         fn f() -> Pair<f64> { Pair(a: 1.0, b: 2.0) }",
        "f",
    )
    .unwrap();
    assert_eq!(ty, Ty::App("Pair".to_string(), vec![Ty::Con("f64".to_string())]));
}

#[test]
fn generic_struct_lits_two_fields_of_the_same_type_parameter_must_agree() {
    // `1.0` (a bare, unsuffixed literal) doesn't become `Con("f64")`
    // immediately â€” it stays an abstract, deferred-defaulting variable
    // carrying a real `Float` constraint (see `stdlib/num/num.cleave`) until
    // `apply_defaults` runs. `true` is immediately concrete (`Con("bool")`),
    // so unifying `T` against it right away succeeds with nothing (yet) to
    // conflict with â€” the actual conflict only surfaces once `1.0`'s own
    // `Float` constraint is checked against the `bool` `T` ended up as,
    // hence `MissingImpl`, not a direct `Unify` mismatch. Needs `Float`
    // actually registered (`registry_from`, not the bare `Registry::default`
    // this test previously used) or there's nothing to check the constraint
    // against at all.
    let err = infer_fn_named(
        "algebra Float<T> {}
         struct Pair<T> { a: T, b: T }
         fn f() { Pair(a: 1.0, b: true) }",
        "f",
    )
    .unwrap_err();
    assert!(
        matches!(&err.kind, TypeErrorKind::MissingImpl { algebra, .. } if algebra == "Float"),
        "got: {:?}",
        err.kind
    );
}

#[test]
fn declaring_a_generic_struct_return_type_without_its_own_type_argument_is_a_real_mismatch() {
    // `-> Pair` (bare, no `<f64>`) is a *different* type than `Pair<f64>` â€”
    // matches ordinary generic-type rules (Rust has the same expectation);
    // this isn't a "generic structs are unsupported" gap, it's an honest
    // `Con("Pair")` vs. `App("Pair", [f64])` mismatch.
    let err = infer_fn_named(
        "struct Pair<T> { a: T, b: T }
         fn f() -> Pair { Pair(a: 1.0, b: 2.0) }",
        "f",
    )
    .unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::Unify(_)), "got: {:?}", err.kind);
}

#[test]
fn field_access_on_a_known_struct_resolves_the_declared_field_type() {
    let ty = infer_fn_named(
        "struct Vec2 { x: f64, y: f64 }
         fn f(v: Vec2) -> f64 { v.x }",
        "f",
    )
    .unwrap();
    assert_eq!(ty, Ty::Con("f64".to_string()));
}

#[test]
fn field_access_on_a_generic_struct_resolves_the_type_argument_not_the_bare_parameter_name() {
    // `a`'s declared field type is the literal parameter name `T` â€” must
    // read back as `f64` (this specific value's own type argument), not the
    // meaningless bare `Con("T")`.
    let ty = infer_fn_named(
        "struct Pair<T> { a: T, b: T }
         fn f(p: Pair<f64>) -> f64 { p.a }",
        "f",
    )
    .unwrap();
    assert_eq!(ty, Ty::Con("f64".to_string()));
}

#[test]
fn field_access_on_a_generic_struct_still_abstract_type_argument_stays_that_variable() {
    // `p`'s own type argument is never pinned to anything concrete here â€”
    // `p.a`'s type should be *exactly* that same still-open variable, not a
    // placeholder and not a wrongly-defaulted concrete type.
    let f = lower_one_fn("fn f(p) { p.a }");
    let registry = registry_from("struct Pair<T> { a: T, b: T }");
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    // `p`'s own type is never pinned to `Pair<...>` at all here (nothing in
    // the body forces it), so `p.a` stays `<not-yet-inferred>` and the
    // function's own exposed type correctly gets rejected as unresolved â€”
    // matches `field_access_on_a_still_abstract_base_stays_a_placeholder_not_an_error`'s
    // own reasoning for the non-generic case.
    assert!(matches!(err.kind, TypeErrorKind::Unresolved(_)), "got: {:?}", err.kind);
}

#[test]
fn field_access_naming_an_undeclared_field_is_rejected() {
    let err = infer_fn_named(
        "struct Vec2 { x: f64, y: f64 }
         fn f(v: Vec2) -> f64 { v.z }",
        "f",
    )
    .unwrap_err();
    assert!(
        matches!(&err.kind, TypeErrorKind::NoSuchField { field, .. } if field == "z"),
        "got: {:?}",
        err.kind
    );
}

#[test]
fn field_access_on_a_non_struct_concrete_type_is_rejected() {
    let err = infer_fn_named("fn f(x: i32) -> i32 { x.foo }", "f").unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::NoSuchField { .. }), "got: {:?}", err.kind);
}

#[test]
fn field_access_on_a_still_abstract_base_stays_a_placeholder_not_an_error() {
    // `x`'s own type is never pinned to anything â€” matches the same "we
    // don't know yet, not a failure" posture as any other not-yet-inferred
    // construct. The field access is a discarded *statement*, not the
    // function's own exposed tail, so the placeholder never has to survive
    // to the final type (which would correctly be rejected â€” see
    // `TypeErrorKind::Unresolved` â€” this test is about the field-access
    // node's own recorded type specifically, via `node_types`).
    let f = lower_one_fn("fn f(x) -> i32 { x.foo; 1 }");
    let registry = Registry::default();
    let mut infer = Infer::new(&registry);
    infer.infer_fn(&f).unwrap_or_else(|e| panic!("{e:?}"));
    let field_access = &f.body.as_ref().unwrap().stmts[0];
    let StmtKind::Expr(e) = &field_access.kind else { panic!("expected an Expr statement") };
    assert_eq!(infer.node_types.get(&e.id), Some(&Ty::Con("<not-yet-inferred>".to_string())));
}

#[test]
fn zero_field_struct_via_a_zero_arg_call_resolves_to_the_struct_type() {
    // The one remaining parse-level ambiguity (see `grammar.pest`'s
    // `primary` comment) â€” `Empty()` lowers as an ordinary zero-arg `Call`,
    // not a `StructLit`; `infer_call`'s own fallback recognizes it anyway.
    let ty = infer_fn_named(
        "struct Empty {}
         fn f() -> Empty { Empty() }",
        "f",
    )
    .unwrap();
    assert_eq!(ty, Ty::Con("Empty".to_string()));
}

#[test]
fn an_impl_target_generic_left_undeclared_on_the_impl_itself_is_treated_as_a_bogus_concrete_type() {
    // `impl TestAlg<Complex<T>>` (no `<T: Float>` of its own) has nowhere
    // for `T` to be bound -- `ty_from_ast_mapped` falls back to treating the
    // bare name as a literal concrete type `Con("T")`, not a generic. This
    // isn't a defaulting/inference bug: it's a real, if confusingly
    // silent, consequence of the impl never declaring `T` as its own
    // generic parameter. The fix on the user's side is always
    // `impl<T: Float> TestAlg<Complex<T>>`; this test pins the current
    // (silent, not-a-nice-error) behavior so a future improvement to
    // diagnose it explicitly has a regression test to update.
    let src = "algebra TestAlg<T> {
            fn gt(x : T, y : T) -> bool;
        }
        struct Complex<T> {
            real : T,
            imag : T,
        }
        impl TestAlg<Complex<T>> {
            fn gt(x , y) { true }
        }";
    let registry = registry_from(src);
    let program = lower_program(src);
    let (algebra, generics, target, f, span) = {
        let item = program
            .items
            .into_iter()
            .find(|item| matches!(&item.kind, ItemKind::Impl(d) if matches!(&d.target.kind, cleave::ast::TypeKind::Path(p, _) if p.segments == ["Complex"])))
            .unwrap();
        match item.kind {
            ItemKind::Impl(d) => {
                let f = d.fns.into_iter().next().unwrap();
                (d.algebra, d.generics, d.target, f, item.span)
            }
            other => panic!("expected impl, got {other:?}"),
        }
    };
    let mut infer = Infer::new(&registry);
    infer.infer_impl_fn_generic(&algebra, &generics, &target, &f, span).unwrap();
    assert_eq!(infer.param_types, vec![Ty::App("Complex".to_string(), vec![Ty::Con("T".to_string())]); 2]);
}

#[test]
fn an_impl_declaring_its_own_generic_resolves_the_target_generic_to_a_fresh_variable() {
    // The correctly-spelled counterpart to the test above: `impl<T: Float>
    // TestAlg<Complex<T>>` declares `T` as its own generic, so it resolves
    // through `impl_mapping` to a fresh, unbound `Ty::Var` -- not a literal
    // `Con("T")`.
    let src = "algebra TestAlg<T> {
            fn gt(x : T, y : T) -> bool;
        }
        struct Complex<T> {
            real : T,
            imag : T,
        }
        impl<T: Float> TestAlg<Complex<T>> {
            fn gt(x , y) { true }
        }
        algebra Float<T> {}
        impl Float<f64> {}";
    let registry = registry_from(src);
    let program = lower_program(src);
    let (algebra, generics, target, f, span) = {
        let item = program
            .items
            .into_iter()
            .find(|item| matches!(&item.kind, ItemKind::Impl(d) if matches!(&d.target.kind, cleave::ast::TypeKind::Path(p, _) if p.segments == ["Complex"])))
            .unwrap();
        match item.kind {
            ItemKind::Impl(d) => {
                let f = d.fns.into_iter().next().unwrap();
                (d.algebra, d.generics, d.target, f, item.span)
            }
            other => panic!("expected impl, got {other:?}"),
        }
    };
    let mut infer = Infer::new(&registry);
    infer.infer_impl_fn_generic(&algebra, &generics, &target, &f, span).unwrap();
    match &infer.param_types[..] {
        [Ty::App(name, args), Ty::App(name2, args2)] => {
            assert_eq!(name, "Complex");
            assert_eq!(name2, "Complex");
            assert!(matches!(args.as_slice(), [Ty::Var(_)]), "expected a fresh var, got {args:?}");
            assert_eq!(args, args2, "both params should share the same fresh var");
        }
        other => panic!("expected two App(\"Complex\", [Var(_)]) param types, got {other:?}"),
    }
}

// ---------------------------------------------------------------------
// arrays: literals, indexing, and `[T; N]` type annotations
// ---------------------------------------------------------------------

#[test]
fn array_literal_infers_element_type_and_literal_size() {
    let ty = infer_src("fn f() -> [i32; 3] { [1, 2, 3] }");
    assert_eq!(ty, Ty::Array(Box::new(Ty::Con("i32".to_string())), Box::new(Ty::Const(ConstValue::Int(3)))));
}

#[test]
fn empty_array_literal_has_size_zero_and_an_unconstrained_element_type() {
    let ty = infer_src("fn f() { [] }");
    match ty {
        Ty::Array(elem, size) => {
            assert!(matches!(*elem, Ty::Var(_)), "expected an unconstrained element var, got {elem:?}");
            assert_eq!(*size, Ty::Const(ConstValue::Int(0)));
        }
        other => panic!("expected an Array, got {other:?}"),
    }
}

#[test]
fn array_literal_with_mismatched_element_types_is_rejected() {
    let f = lower_one_fn("fn f() { [1, true] }");
    let registry = builtin_registry();
    let mut infer = Infer::new(&registry);
    assert!(infer.infer_fn(&f).is_err(), "an int and a bool must not unify as one array's element type");
}

#[test]
fn array_index_returns_the_element_type() {
    let ty = infer_src("fn f(a: [i32; 3]) -> i32 { a[0] }");
    assert_eq!(ty, Ty::Con("i32".to_string()));
}

#[test]
fn indexing_a_non_array_is_rejected() {
    let f = lower_one_fn("fn f(a: i32) { a[0] }");
    let registry = builtin_registry();
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::Unify(_)), "got: {:?}", err.kind);
}

#[test]
fn indexing_with_a_non_integer_index_is_rejected() {
    // The index is `Int`-constrained the same way a bare literal is (see
    // `infer.rs`'s `ExprKind::Index` handling) -- `1.5` is `Float`-shaped,
    // which conflicts with that constraint once defaulted, the same
    // mechanism `two_literals_with_conflicting_shapes_merged_by_a_shared_generic_are_rejected`
    // exercises for `+`.
    let f = lower_one_fn("fn f(a: [i32; 3]) { a[1.5] }");
    let registry = builtin_registry();
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::MissingImpl { .. }), "got: {:?}", err.kind);
}

#[test]
fn indexing_an_unresolved_base_defers_as_not_yet_inferred() {
    // `a`'s type is never pinned down by anything in `f`'s own body --
    // `Index` itself is permissive about it ("we don't know yet", matching
    // `FieldAccess`'s identical posture for the same situation), but the
    // placeholder it produces still isn't allowed to survive all the way to
    // `f`'s own exposed return type (`check_no_placeholder` -- see
    // `unresolved_call_surviving_to_the_final_type_is_rejected` for the same
    // check exercised via a cross-function call instead of indexing).
    let f = lower_one_fn("fn f(a) { a[0] }");
    let registry = builtin_registry();
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::Unresolved(_)), "got: {:?}", err.kind);
}

#[test]
fn fortran_style_multi_index_sugar_indexes_a_nested_array() {
    // `a[0, 1]` desugars (in `lower.rs`) to `Index(Index(a, 0), 1)` --
    // exercised here at the type level: `[[i32; 4]; 3]` indexed twice peels
    // one dimension per `Index`, landing on the scalar element type.
    let ty = infer_src("fn f(a: [[i32; 4]; 3]) -> i32 { a[0, 1] }");
    assert_eq!(ty, Ty::Con("i32".to_string()));
}

#[test]
fn arrays_of_different_literal_sizes_do_not_unify() {
    // The actual point of tracking size as part of the type at all: two
    // branches of an `if` returning differently-sized arrays is a genuine,
    // statically-caught shape mismatch, not silently accepted the way it
    // would be if `Array`'s size were erased (or arrays stayed the old
    // `<array-type-not-yet-inferred>` placeholder).
    let f = lower_one_fn("fn f(c: bool) { if c { [1, 2, 3] } else { [1, 2] } }");
    let registry = builtin_registry();
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::Unify(_)), "got: {:?}", err.kind);
}

#[test]
fn a_declared_array_return_type_checks_the_literals_actual_size() {
    // `-> [i32; 4]` declared, but the body's literal only has 3 elements --
    // must be rejected at the declared-type-vs-body-result check, same as
    // any other declared-vs-inferred mismatch.
    let f = lower_one_fn("fn f() -> [i32; 4] { [1, 2, 3] }");
    let registry = builtin_registry();
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::Unify(_)), "got: {:?}", err.kind);
}

#[test]
fn an_array_type_annotation_with_a_literal_size_round_trips() {
    let ty = infer_src("fn f(a: [f64; 4]) -> [f64; 4] { a }");
    assert_eq!(ty, Ty::Array(Box::new(Ty::Con("f64".to_string())), Box::new(Ty::Const(ConstValue::Int(4)))));
}

// ---------------------------------------------------------------------
// constant folding (`const_eval`): pure literal arithmetic in a shape
// position, general — not tied to any const generic being involved.
// ---------------------------------------------------------------------

#[test]
fn a_literal_arithmetic_array_size_folds_to_a_concrete_const() {
    let ty = infer_src("fn f(a: [f64; 4+3]) -> [f64; 7] { a }");
    assert_eq!(ty, Ty::Array(Box::new(Ty::Con("f64".to_string())), Box::new(Ty::Const(ConstValue::Int(7)))));
}

#[test]
fn a_folded_literal_array_size_still_rejects_a_real_mismatch() {
    // Proves folding is real (not a no-op that just gives up permissively):
    // `4+3` must resolve to exactly `7`, not silently accept a `3`-element
    // literal.
    let f = lower_one_fn("fn f() -> [i32; 4+3] { [1, 2, 3] }");
    let registry = builtin_registry();
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::Unify(_)), "got: {:?}", err.kind);
}

#[test]
fn an_unsupported_operator_in_shape_position_stays_permissively_unconstrained() {
    // `const_eval::eval_binop` only knows `add`/`mul` so far -- `4-3`
    // (`sub`) must fall through to the existing "not evaluated" placeholder
    // (same as any other not-yet-inferred array type) rather than crash or
    // be treated as a hard parse/evaluation error. That placeholder can
    // never be *exposed* in a function's own signature though (same rule
    // any other still-unresolved type follows, `check_no_placeholder`) --
    // this is what proves the fallback path was actually reached, not
    // skipped some other way.
    let f = lower_one_fn("fn f(a: [i32; 4-3]) -> [i32; 4-3] { a }");
    let registry = builtin_registry();
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    assert!(
        matches!(&err.kind, TypeErrorKind::Unresolved(name) if name == "<array-type-not-yet-inferred>"),
        "got: {:?}",
        err.kind
    );
}

// ---------------------------------------------------------------------
// const-generics (`Matrix<f64, 3, 3>`): positional `Ty::App` args, mixing
// type and const generics on the same struct
// ---------------------------------------------------------------------

#[test]
fn a_const_generic_type_annotation_builds_positional_app_args() {
    // No `struct Matrix` declaration needed here -- a bare type annotation's
    // `Ty::App` doesn't check the registry at all (only construction/field
    // access do); this just pins down the *shape* `ty_from_ast_mapped`
    // builds for a path mixing type and const generic arguments.
    let ty = infer_fn_named("fn f(m: Matrix<f64, 3, 4>) -> Matrix<f64, 3, 4> { m }", "f").unwrap();
    assert_eq!(
        ty,
        Ty::App(
            "Matrix".to_string(),
            vec![Ty::Con("f64".to_string()), Ty::Const(ConstValue::Int(3)), Ty::Const(ConstValue::Int(4))],
        )
    );
}

#[test]
fn a_structs_const_generic_is_inferred_from_the_constructed_arrays_actual_size() {
    // `Vec<T, const N: i32>` declares no explicit way to *supply* `N` at
    // a construction site (there's no turbofish syntax) -- it's inferred
    // purely from the field value, the same "infer everything from usage"
    // stance the struct's own type generics already use. Here, `data`'s
    // declared type `[T; N]` unifies against the literal `[1, 2, 3]`'s own
    // inferred `[i32; 3]`, pinning `N` to `Const(3)` for free.
    let ty = infer_fn_named(
        "struct Vec<T, const N: i32> { data: [T; N] }
         fn f() -> Vec<i32, 3> { Vec(data: [1, 2, 3]) }",
        "f",
    )
    .unwrap();
    assert_eq!(ty, Ty::App("Vec".to_string(), vec![Ty::Con("i32".to_string()), Ty::Const(ConstValue::Int(3))]));
}

#[test]
fn a_structs_const_generic_mismatch_against_the_declared_return_type_is_rejected() {
    let err = infer_fn_named(
        "struct Vec<T, const N: i32> { data: [T; N] }
         fn f() -> Vec<i32, 4> { Vec(data: [1, 2, 3]) }",
        "f",
    )
    .unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::Unify(_)), "got: {:?}", err.kind);
}

#[test]
fn field_access_on_a_struct_with_a_const_generic_resolves_the_type_generic_field() {
    // `zip_struct_generics` must line up `T`/`N` positionally against
    // `App`'s two argument slots -- a regression guard for the "skip Const,
    // don't consume a `type_args` entry for it" bug the old filter-based
    // zip would have (silently pairing `T` with `N`'s own slot instead of
    // `T`'s).
    let ty = infer_fn_named(
        "struct Vec<T, const N: i32> { data: [T; N] }
         fn f(v: Vec<i32, 3>) -> [i32; 3] { v.data }",
        "f",
    )
    .unwrap();
    assert_eq!(ty, Ty::Array(Box::new(Ty::Con("i32".to_string())), Box::new(Ty::Const(ConstValue::Int(3)))));
}

#[test]
fn a_const_generic_is_not_forced_to_be_int_just_by_being_declared() {
    // `const B: bool` is legitimate entirely on its own -- `Int` is an
    // algebra over *types* (i8/i32/...), which has nothing to say about a
    // bool-typed const-generic that's never used as an array size.
    // Supplied here via a bare type annotation (`Flagged<i32, true>`) since
    // there's no way to *infer* a bool const-generic from a field value the
    // way an array-sized field infers an integer one.
    let ty = infer_fn_named(
        "struct Flagged<T, const B: bool> { value: T }
         fn f(v: Flagged<i32, true>) -> Flagged<i32, true> { v }",
        "f",
    )
    .unwrap();
    assert_eq!(
        ty,
        Ty::App("Flagged".to_string(), vec![Ty::Con("i32".to_string()), Ty::Const(ConstValue::Bool(true))])
    );
}

#[test]
fn a_bool_literal_used_directly_as_an_array_size_is_rejected() {
    // The one real, structural check: an array's own size slot can't hold
    // an outright bool literal. Deferred as the usual "not yet inferred"
    // placeholder rather than a dedicated error kind, then caught the same
    // way any placeholder surviving to `f`'s own exposed signature already
    // is (see `check_no_placeholder`).
    let f = lower_one_fn("fn f(a: [i32; true]) { a }");
    let registry = builtin_registry();
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::Unresolved(_)), "got: {:?}", err.kind);
}

#[test]
fn a_const_generics_declared_type_satisfying_int_is_accepted() {
    let ty = infer_fn_named(
        "algebra Int<T> {}
         impl Int<i32> {}
         struct Vec<T, const N: i32> { data: [T; N] }
         fn f() -> Vec<i32, 3> { Vec(data: [1, 2, 3]) }",
        "f",
    )
    .unwrap();
    assert_eq!(ty, Ty::App("Vec".to_string(), vec![Ty::Con("i32".to_string()), Ty::Const(ConstValue::Int(3))]));
}

// ---------------------------------------------------------------------
// overlapping generic impls: two `impl`s of the same algebra whose own
// target patterns could both match a common instantiation
// ---------------------------------------------------------------------

#[test]
fn overlapping_generic_impls_of_the_same_algebra_are_rejected() {
    // `Complex<T>` with `T: Float` and `Complex<T>` with `T: Ord` -- both
    // patterns are shape-identical (`Complex<_>`), so some hypothetical
    // `Complex<X>` satisfying both bounds would have no principled impl to
    // dispatch to. Caught purely by shape, without asking whether `Float`
    // and `Ord` could ever really share a concrete type (see the method's
    // own doc comment on why that's deliberately not attempted).
    let src = "algebra TestAlg<T> {
            fn gt(x : T, y : T) -> bool;
        }
        struct Complex<T> {
            real : T,
            imag : T,
        }
        algebra Float<T> {}
        impl Float<f64> {}
        algebra Ord2<T> {}
        impl Ord2<f64> {}
        impl<T: Float> TestAlg<Complex<T>> {
            fn gt(x , y) { true }
        }
        impl<T: Ord2> TestAlg<Complex<T>> {
            fn gt(x , y) { true }
        }";
    let registry = registry_from(src);
    let mut infer = Infer::new(&registry);
    let errors = infer.check_no_overlapping_impls();
    assert_eq!(errors.len(), 1, "got: {errors:?}");
    assert!(matches!(errors[0].kind, TypeErrorKind::OverlappingImpls { .. }), "got: {:?}", errors[0].kind);
}

#[test]
fn non_overlapping_generic_impls_of_different_target_shapes_are_accepted() {
    // `Complex<T>` and `Quaternion<T>` can never unify against each other
    // (different struct names) -- two entirely legitimate, independent
    // generic impls of the same algebra, not a coherence problem.
    let src = "algebra TestAlg<T> {
            fn gt(x : T, y : T) -> bool;
        }
        struct Complex<T> { real : T, imag : T }
        struct Quaternion<T> { a : T, b : T, c : T, d : T }
        algebra Float<T> {}
        impl Float<f64> {}
        impl<T: Float> TestAlg<Complex<T>> {
            fn gt(x , y) { true }
        }
        impl<T: Float> TestAlg<Quaternion<T>> {
            fn gt(x , y) { true }
        }";
    let registry = registry_from(src);
    let mut infer = Infer::new(&registry);
    let errors = infer.check_no_overlapping_impls();
    assert!(errors.is_empty(), "got: {errors:?}");
}

#[test]
fn a_single_generic_impl_never_overlaps_with_itself() {
    let src = "algebra TestAlg<T> {
            fn gt(x : T, y : T) -> bool;
        }
        struct Complex<T> { real : T, imag : T }
        algebra Float<T> {}
        impl Float<f64> {}
        impl<T: Float> TestAlg<Complex<T>> {
            fn gt(x , y) { true }
        }";
    let registry = registry_from(src);
    let mut infer = Infer::new(&registry);
    let errors = infer.check_no_overlapping_impls();
    assert!(errors.is_empty(), "got: {errors:?}");
}

#[test]
fn a_const_generics_own_declared_type_naming_an_algebra_instead_of_a_type_is_rejected() {
    // A real bug, found by direct user testing: `const R: Int` -- `Int` is
    // the *algebra* that governs which types are legal integers (`i32`,
    // `i64`, ...), not a type itself. Nothing previously checked that a type
    // annotation's bare name actually names a type at all, so this passed
    // silently. `Matrix<3>` at the use site still resolves structurally
    // (`ty_from_ast_mapped` doesn't refuse to build the `Ty` — it just also
    // queues the diagnostic), which is why the failure surfaces from `f`'s
    // own construction, not from parsing the struct declaration itself.
    let src = "algebra Int<T> {}
        impl Int<i32> {}
        struct Matrix<const R: Int> { x: i32 }
        fn f() -> Matrix<3> { Matrix(x: 1) }";
    let err = infer_fn_named(src, "f").unwrap_err();
    assert!(
        matches!(&err.kind, TypeErrorKind::TypeNameIsAnAlgebra { name } if name == "Int"),
        "got: {:?}",
        err.kind
    );
}

#[test]
fn a_const_generics_own_declared_type_naming_a_real_type_is_accepted() {
    let src = "algebra Int<T> {}
        impl Int<i32> {}
        struct Matrix<const R: i32> { x: i32 }
        fn f() -> Matrix<3> { Matrix(x: 1) }";
    infer_fn_named(src, "f").unwrap();
}

// ---------------------------------------------------------------------
// turbofish: explicit `::<...>` generic arguments on a call or struct
// construction, for when nothing about the arguments/field values
// themselves would pin the instantiation down
// ---------------------------------------------------------------------

#[test]
fn turbofish_on_a_let_bound_generic_lambda_pins_the_instantiation() {
    let ty = infer_src("fn f() -> f64 { let id = fn(x) { x }; id::<f64>(1.0) }");
    assert_eq!(ty, Ty::Con("f64".to_string()));
}

#[test]
fn turbofish_conflicting_with_the_arguments_own_type_is_rejected() {
    let f = lower_one_fn("fn f() { let id = fn(x) { x }; id::<f64>(1:i32) }");
    let registry = builtin_registry();
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::Unify(_)), "got: {:?}", err.kind);
}

#[test]
fn turbofish_arity_mismatch_against_a_generic_lambda_is_rejected() {
    let f = lower_one_fn("fn f() { let id = fn(x) { x }; id::<f64, i32>(1.0) }");
    let registry = builtin_registry();
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::ArityMismatch { .. }), "got: {:?}", err.kind);
}

#[test]
fn turbofish_on_struct_construction_pins_type_and_const_generics() {
    // The actual reported motivation: forcing `f64` (and the size) directly,
    // rather than relying on an incidentally-suffixed literal to propagate
    // the right type through unification.
    let ty = infer_fn_named(
        "struct Vec<T, const N: i32> { data: [T; N] }
         fn f() -> Vec<f64, 3> { Vec::<f64, 3>(data: [1.0, 2.0, 3.0]) }",
        "f",
    )
    .unwrap();
    assert_eq!(ty, Ty::App("Vec".to_string(), vec![Ty::Con("f64".to_string()), Ty::Const(ConstValue::Int(3))]));
}

#[test]
fn turbofish_on_struct_construction_forcing_a_bare_int_literal_into_float_is_still_rejected() {
    // `[1, 2, 3]`'s bare literals are `Int`-shaped (no `.`) -- turbofish
    // pinning `T` to `f64` conflicts with that real, checked constraint the
    // exact same way any other int-literal-forced-into-a-float-context
    // does (see `a_bare_int_shaped_literal_forced_into_a_float_context_is_
    // now_rejected`); turbofish doesn't bypass the shape check.
    let err = infer_fn_named(
        "algebra Int<T> {}
         impl Int<i32> {}
         algebra Float<T> {}
         impl Float<f64> {}
         struct Vec<T, const N: i32> { data: [T; N] }
         fn f() { Vec::<f64, 3>(data: [1, 2, 3]) }",
        "f",
    )
    .unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::MissingImpl { .. }), "got: {:?}", err.kind);
}

#[test]
fn turbofish_on_struct_construction_arity_mismatch_is_rejected() {
    let err = infer_fn_named(
        "struct Vec<T, const N: i32> { data: [T; N] }
         fn f() -> Vec<f64, 3> { Vec::<f64>(data: [1.0, 2.0, 3.0]) }",
        "f",
    )
    .unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::ArityMismatch { .. }), "got: {:?}", err.kind);
}

#[test]
fn turbofish_on_struct_construction_conflicting_with_the_field_values_size_is_rejected() {
    let err = infer_fn_named(
        "struct Vec<T, const N: i32> { data: [T; N] }
         fn f() { Vec::<f64, 4>(data: [1.0, 2.0, 3.0]) }",
        "f",
    )
    .unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::Unify(_)), "got: {:?}", err.kind);
}

// ---------------------------------------------------------------------
// control flow: while/for
// ---------------------------------------------------------------------

#[test]
fn a_while_loop_is_always_unit_typed() {
    // Same reasoning as an `if` with no `else`: the body might run zero
    // times, and there's no `break value` mechanism, so nothing meaningful
    // could ever come out of evaluating one as an expression.
    let ty = infer_src("fn f() -> bool { let mut i = 0; while i > 0 { i }; true }");
    assert_eq!(ty, Ty::Con("bool".to_string()));
}

#[test]
fn a_while_loops_condition_must_be_bool() {
    let f = lower_one_fn("fn f() { while 1 { 2 } }");
    let registry = builtin_registry();
    let mut infer = Infer::new(&registry);
    assert!(infer.infer_fn(&f).is_err(), "a non-bool condition must be rejected");
}

#[test]
fn a_for_loops_body_and_variable_type_check_normally() {
    let ty = infer_src("fn f() -> bool { for i in 0..10 { i > 0 }; true }");
    assert_eq!(ty, Ty::Con("bool".to_string()));
}

#[test]
fn a_for_loops_start_and_end_must_agree_in_type() {
    // `0` (Int-shaped) vs `3.0` (Float-shaped) -- the exact same real,
    // checked shape conflict a bare literal hits anywhere else in this
    // file, not a for-loop-specific special case.
    let f = lower_one_fn("fn f() { for i in 0..3.0 { i } }");
    let registry = builtin_registry();
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::MissingImpl { .. }), "got: {:?}", err.kind);
}

#[test]
fn a_for_loops_variable_does_not_leak_into_the_enclosing_scope() {
    let f = lower_one_fn("fn f() { for i in 0..10 { i }; i }");
    let registry = builtin_registry();
    let mut infer = Infer::new(&registry);
    assert!(infer.infer_fn(&f).is_err(), "`i` must be unbound outside the loop");
}

#[test]
fn a_for_loops_variable_is_int_constrained_not_forced_to_a_specific_width() {
    // No hardcoded width -- the same "constrained, not blessed" posture
    // `ExprKind::Index`'s own bound already uses. `i64` bounds must work
    // exactly as well as the default `i32`.
    let ty = infer_src("fn f() -> i64 { let mut acc = 0:i64; for i in 0:i64..10:i64 { acc = i; }; acc }");
    assert_eq!(ty, Ty::Con("i64".to_string()));
}

#[test]
fn a_const_generic_used_as_a_for_loop_bound_stays_a_shape_slot_not_a_defaulted_int() {
    // Regression test: `for i in 0..N` unifies `0`'s own (defaultable)
    // literal var with `N`'s own shape-slot var (`ExprKind::For`'s own
    // `unify_at(start_ty, end_ty)`) -- `apply_defaults` must not then
    // commit `N := Ty::Con("i32")` for real (a *type*), which would
    // permanently corrupt the array-size slot `N` is meant to stay usable
    // as (a `Ty::Var`, eventually a `Ty::Const` once a caller pins it
    // concretely) -- found by direct testing via `examples/matmul.cleave`,
    // whose own `N`/`M`/`K` bounds, defaulted this way, then failed to
    // monomorphize at all.
    let f = lower_one_fn(
        "fn fill<T: Int, const N: i32>(v: T) -> [T; N] {
            let mut arr = [v; N];
            for i in 0..N {
                arr[i] = v;
            };
            arr
        }",
    );
    let registry = builtin_registry();
    let mut infer = Infer::new(&registry);
    let ty = infer.infer_fn(&f).unwrap_or_else(|e| panic!("inference failed: {e:?}"));
    match ty {
        Ty::Array(_, size) => {
            assert!(matches!(*size, Ty::Var(_)), "expected `N` to stay an abstract shape slot, got {size:?}")
        }
        other => panic!("expected an array type, got {other:?}"),
    }
}

// ---------------------------------------------------------------------
// higher-order functions: an explicit function-type annotation
// (`(i32) -> i32`) on a parameter
// ---------------------------------------------------------------------

#[test]
fn a_function_typed_parameter_is_itself_usable_inside_the_body() {
    let ty = infer_src("fn apply(f: (i32) -> i32, x: i32) -> i32 { f(x) }");
    assert_eq!(ty, Ty::Con("i32".to_string()));
}

#[test]
fn a_function_typed_parameter_type_checks_in_isolation_even_with_no_caller() {
    let f = lower_one_fn("fn apply(f: (i32) -> bool, x: i32) -> bool { f(x) }");
    let registry = builtin_registry();
    let mut infer = Infer::new(&registry);
    assert!(infer.infer_fn(&f).is_ok());
    // See `tests/callgraph.rs` for the call-site rejection (a lambda
    // returning the wrong type passed to `apply`) -- that needs a real
    // caller, which only the whole-program pass resolves; `infer_fn`/
    // `infer_fn_named` only ever handle a function calling *itself*.
}

#[test]
fn a_function_type_with_multiple_params_round_trips() {
    let ty = infer_src("fn apply(f: (i32, f64) -> bool, x: i32, y: f64) -> bool { f(x, y) }");
    assert_eq!(ty, Ty::Con("bool".to_string()));
}

// ---------------------------------------------------------------------
// inherent impls (`impl struct Vec2 { ... }`, no algebra) and method-call
// (`v.foo(...)`) dispatch
// ---------------------------------------------------------------------

#[test]
fn an_inherent_methods_unannotated_first_parameter_defaults_to_the_impl_target() {
    let ty = infer_fn_named(
        "struct Vec2 { x: f64, y: f64 }
         impl struct Vec2 { fn get_x(v) -> f64 { v.x } }
         fn f() -> f64 { let p = Vec2(x: 1.0, y: 2.0); p.get_x() }",
        "f",
    )
    .unwrap();
    assert_eq!(ty, Ty::Con("f64".to_string()));
}

#[test]
fn method_call_dispatches_the_declared_return_type() {
    let ty = infer_fn_named(
        "struct Vec2 { x: f64, y: f64 }
         impl struct Vec2 { fn scale(v, s: f64) -> f64 { s } }
         fn f() -> f64 { let p = Vec2(x: 1.0, y: 2.0); p.scale(2.0) }",
        "f",
    )
    .unwrap();
    assert_eq!(ty, Ty::Con("f64".to_string()));
}

#[test]
fn method_call_with_a_wrong_argument_type_is_rejected() {
    let err = infer_fn_named(
        "struct Vec2 { x: f64, y: f64 }
         impl struct Vec2 { fn scale(v, s: f64) -> f64 { s } }
         fn f() { let p = Vec2(x: 1.0, y: 2.0); p.scale(true) }",
        "f",
    )
    .unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::Unify(_)), "got: {:?}", err.kind);
}

#[test]
fn method_call_with_wrong_arity_is_rejected() {
    let err = infer_fn_named(
        "struct Vec2 { x: f64, y: f64 }
         impl struct Vec2 { fn scale(v, s: f64) -> f64 { s } }
         fn f() { let p = Vec2(x: 1.0, y: 2.0); p.scale(1.0, 2.0) }",
        "f",
    )
    .unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::ArityMismatch { .. }), "got: {:?}", err.kind);
}

#[test]
fn method_call_on_an_unknown_method_is_rejected() {
    let err = infer_fn_named(
        "struct Vec2 { x: f64, y: f64 }
         fn f() { let p = Vec2(x: 1.0, y: 2.0); p.bogus() }",
        "f",
    )
    .unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::NoSuchMethod { .. }), "got: {:?}", err.kind);
}

#[test]
fn method_call_on_a_non_struct_base_is_rejected() {
    let f = lower_one_fn("fn f(x: i32) { x.foo() }");
    let registry = builtin_registry();
    let mut infer = Infer::new(&registry);
    let err = infer.infer_fn(&f).unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::NoSuchMethod { .. }), "got: {:?}", err.kind);
}

#[test]
fn a_method_call_with_no_declared_return_type_defers_permissively_when_unused() {
    // No `->` on `len` -- the call's own result has nowhere to report a
    // real type from (dispatch never re-runs the method's own body), so it
    // defers as a placeholder. As long as that placeholder never has to
    // reach the caller's *own* exposed signature (the call's result is
    // simply discarded here), the whole program still succeeds.
    let ty = infer_fn_named(
        "struct Vec2 { x: f64, y: f64 }
         impl struct Vec2 { fn len(v) { v.x } }
         fn f() -> i32 { let p = Vec2(x: 1.0, y: 2.0); p.len(); 42 }",
        "f",
    )
    .unwrap();
    assert_eq!(ty, Ty::Con("i32".to_string()));
}

#[test]
fn a_generic_structs_inherent_method_resolves_the_type_argument() {
    let ty = infer_fn_named(
        "struct Boxed<T> { value: T }
         impl<T> struct Boxed<T> { fn get(b) -> T { b.value } }
         fn f() -> f64 { let b = Boxed(value: 1.0); b.get() }",
        "f",
    )
    .unwrap();
    assert_eq!(ty, Ty::Con("f64".to_string()));
}

#[test]
fn a_self_recursive_unannotated_inherent_method_infers_its_own_return_type() {
    // Real bug, found by direct testing: `w.countdown()` (a recursive call
    // to the *same* method, via the ordinary `v.method()` syntax, not by
    // bare name) used to defer to `<not-yet-inferred>` even during the
    // method's own declaration -- nothing tied a recursive `MethodCall` back
    // to the enclosing invocation's own still-open return type the way a
    // self-recursive top-level `fn` already gets via `env` (`infer_fn`'s own
    // seeded placeholder never applies here: dispatch never consults `env`
    // for its callee). Fixed via `Infer::in_progress_methods`.
    let registry = registry_from(
        "algebra Ord<T> { fn eq(a: T, b: T) -> bool; }
         impl Ord<i32> { fn eq(a, b) { true } }
         struct Wrap { n : i32 }",
    );
    let (target, generics, f, span) = lower_one_inherent_impl(
        "impl struct Wrap {
            fn countdown(w) {
                if eq(w.n, 0) { 0 } else { w.countdown() }
            }
        }",
    );
    let mut infer = Infer::new(&registry);
    let ty = infer
        .infer_inherent_impl_fn_generic(&cleave::infer::Env::new(), &generics, &target, &f, span)
        .unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(ty, Ty::Con("i32".to_string()));
}

#[test]
fn two_mutually_recursive_inherent_methods_infer_correctly() {
    // The narrower gap `in_progress_methods` (single self-only slot)
    // couldn't close: `is_even`/`is_odd`, as *separately declared* methods
    // on the same struct, each calling the other -- neither self-recursive.
    // Real bug this closes: each method used to be inferred with its own
    // fresh `Infer`, in total isolation, so a recursive call to the *other*
    // method (still mid-inference, in a completely different `Infer`
    // instance) always deferred to `<not-yet-inferred>`, which then failed
    // to unify with the `if`'s other branch. Fixed by `Infer::
    // infer_inherent_impl_block`, which shares one `Infer` across every
    // method of one impl block and seeds all their placeholders up front,
    // mirroring `callgraph.rs`'s own group-based treatment of mutually
    // recursive top-level `fn`s.
    let registry = registry_from(
        "algebra Ord<T> { fn eq(a: T, b: T) -> bool; }
         impl Ord<i32> { fn eq(a, b) { true } }
         algebra Ring<T> { fn sub(a: T, b: T) -> T; }
         impl Ring<i32> { fn sub(a, b) { a } }
         struct Wrap { n : i32 }",
    );
    let (target, generics, fns, span) = lower_inherent_impl(
        "impl struct Wrap {
            fn dec(w) -> Wrap { Wrap(n: w.n - 1) }
            fn is_even(w) {
                if eq(w.n, 0) { true } else { w.dec().is_odd() }
            }
            fn is_odd(w) {
                if eq(w.n, 0) { false } else { w.dec().is_even() }
            }
        }",
    );
    let mut infer = Infer::new(&registry);
    let results = infer.infer_inherent_impl_block(&cleave::infer::Env::new(), &generics, &target, &fns, span);
    let is_even = results.get("is_even").unwrap().as_ref().unwrap_or_else(|e| panic!("{e:?}"));
    let is_odd = results.get("is_odd").unwrap().as_ref().unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(is_even.1, Ty::Con("bool".to_string()));
    assert_eq!(is_odd.1, Ty::Con("bool".to_string()));
}

#[test]
fn a_single_self_recursive_method_still_works_through_infer_inherent_impl_block() {
    // The block-level entry point, exercised with just one method -- a
    // regression guard for the refactor `infer_inherent_impl_block` was
    // built from (`infer_inherent_impl_fn_raw`, factored out of the
    // existing single-method `infer_inherent_impl_fn_generic`), not a new
    // capability on its own.
    let registry = registry_from(
        "algebra Ord<T> { fn eq(a: T, b: T) -> bool; }
         impl Ord<i32> { fn eq(a, b) { true } }
         struct Wrap { n : i32 }",
    );
    let (target, generics, fns, span) = lower_inherent_impl(
        "impl struct Wrap {
            fn countdown(w) {
                if eq(w.n, 0) { 0 } else { w.countdown() }
            }
        }",
    );
    let mut infer = Infer::new(&registry);
    let results = infer.infer_inherent_impl_block(&cleave::infer::Env::new(), &generics, &target, &fns, span);
    let countdown = results.get("countdown").unwrap().as_ref().unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(countdown.1, Ty::Con("i32".to_string()));
}

// ---------------------------------------------------------------------
// heterogeneous algebra dispatch: `algebra MatMul<A, B, C>`, a multi-target
// impl, and `a * b` resolving a genuinely different result type
// ---------------------------------------------------------------------

const MATMUL_SRC: &str = "
    algebra MatMul<A, B, C> { fn mul(a: A, b: B) -> C; }
    algebra Float<T> {}
    impl Float<f32> {}
    impl Float<f64> {}
    struct Matrix<T : Float, const R : i32, const C : i32> { values : [T; R, C] }
    impl<T: Float, const N: i32, const M: i32, const K: i32>
        MatMul<Matrix<T,N,M>, Matrix<T,M,K>, Matrix<T,N,K>> { fn mul(a, b) { a } }
";

#[test]
fn matmul_resolves_the_output_shape_from_the_two_input_shapes() {
    // The actual point: `C` (N,K) is neither `A`'s shape (N,M) nor `B`'s
    // (M,K) -- it only exists once dispatch itself determines it.
    let src = format!(
        "{MATMUL_SRC}
         fn f() -> Matrix<f32, 2, 5> {{
             let a = Matrix::<f32, 2, 3>(values: [[1.0,1.0,1.0],[1.0,1.0,1.0]]);
             let b = Matrix::<f32, 3, 5>(values: [[1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0]]);
             a * b
         }}"
    );
    let ty = infer_fn_named(&src, "f").unwrap();
    assert_eq!(
        ty,
        Ty::App("Matrix".to_string(), vec![Ty::Con("f32".to_string()), Ty::Const(ConstValue::Int(2)), Ty::Const(ConstValue::Int(5))])
    );
}

#[test]
fn matmul_rejects_a_mismatched_middle_dimension() {
    let src = format!(
        "{MATMUL_SRC}
         fn f() {{
             let a = Matrix::<f32, 2, 3>(values: [[1.0,1.0,1.0],[1.0,1.0,1.0]]);
             let b = Matrix::<f32, 4, 5>(values: [[1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0]]);
             let c = a * b;
             42
         }}"
    );
    let err = infer_fn_named(&src, "f").unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::MissingImpl { .. }), "got: {:?}", err.kind);
}

#[test]
fn matmul_element_type_must_also_agree() {
    // f32 vs f64 -- shapes line up (2x3 * 3x5), element type doesn't.
    let src = format!(
        "{MATMUL_SRC}
         fn f() {{
             let a = Matrix::<f32, 2, 3>(values: [[1.0,1.0,1.0],[1.0,1.0,1.0]]);
             let b = Matrix::<f64, 3, 5>(values: [[1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0]]);
             let c = a * b;
             42
         }}"
    );
    let err = infer_fn_named(&src, "f").unwrap_err();
    assert!(matches!(err.kind, TypeErrorKind::MissingImpl { .. }), "got: {:?}", err.kind);
}

#[test]
fn overlapping_multi_target_impls_of_the_same_algebra_are_rejected() {
    let src = "algebra MatMul<A, B, C> { fn mul(a: A, b: B) -> C; }
        struct Matrix<T> { data: T }
        algebra Float2<T> {}
        impl Float2<f64> {}
        algebra Ord2<T> {}
        impl Ord2<f64> {}
        impl<T: Float2> MatMul<Matrix<T>, Matrix<T>, Matrix<T>> { fn mul(a, b) { a } }
        impl<T: Ord2> MatMul<Matrix<T>, Matrix<T>, Matrix<T>> { fn mul(a, b) { a } }";
    let registry = registry_from(src);
    let mut infer = Infer::new(&registry);
    let errors = infer.check_no_overlapping_impls();
    assert_eq!(errors.len(), 1, "got: {errors:?}");
    assert!(matches!(errors[0].kind, TypeErrorKind::OverlappingImpls { .. }), "got: {:?}", errors[0].kind);
}

#[test]
fn non_overlapping_multi_target_impls_differing_in_a_later_target_are_accepted() {
    let src = "algebra MatMul<A, B, C> { fn mul(a: A, b: B) -> C; }
        struct Matrix<T> { data: T }
        struct Vector<T> { data: T }
        impl<T> MatMul<Matrix<T>, Matrix<T>, Matrix<T>> { fn mul(a, b) { a } }
        impl<T> MatMul<Matrix<T>, Vector<T>, Vector<T>> { fn mul(a, b) { b } }";
    let registry = registry_from(src);
    let mut infer = Infer::new(&registry);
    let errors = infer.check_no_overlapping_impls();
    assert!(errors.is_empty(), "got: {errors:?}");
}
