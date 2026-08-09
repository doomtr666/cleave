use cleave::ast::{FileId, Program};
use cleave::lower::Lowerer;
use cleave::parser::{CleaveParser, Rule};
use cleave::registry::Registry;
use pest::Parser;

fn program(src: &str) -> Program {
    let pair = CleaveParser::parse(Rule::program, src)
        .unwrap_or_else(|e| panic!("parse failed for {src:?}: {e}"))
        .next()
        .unwrap();
    Lowerer::new(FileId(0)).lower_program(pair)
}

#[test]
fn finds_impl_for_concrete_target() {
    let p = program(
        "algebra Ring<T> { fn add(a: T, b: T) -> T; }
         impl Ring<Vec2> { fn add(a: Vec2, b: Vec2) -> Vec2 { a } }",
    );
    let reg = Registry::build(&p);
    assert!(reg.has_impl("Ring", &vec2_type()));
}

#[test]
fn reports_no_impl_for_untargeted_type() {
    let p = program(
        "algebra Ring<T> { fn add(a: T, b: T) -> T; }
         impl Ring<Vec2> { fn add(a: Vec2, b: Vec2) -> Vec2 { a } }",
    );
    let reg = Registry::build(&p);
    assert!(!reg.has_impl("Ring", &bool_type()), "no `impl Ring<bool>` was ever declared");
}

#[test]
fn unknown_algebra_has_no_impls() {
    let p = program("struct Vec2 { x: f64, y: f64 }");
    let reg = Registry::build(&p);
    assert!(!reg.has_algebra("Ring"));
    assert!(!reg.has_impl("Ring", &vec2_type()));
}

#[test]
fn finds_candidate_algebras_by_name_and_arity() {
    let p = program(
        "algebra Ring<T> { fn add(a: T, b: T) -> T; }
         algebra Tropical<T> { fn add(a: T, b: T) -> T; }
         algebra Unary<T> { fn neg(a: T) -> T; }",
    );
    let reg = Registry::build(&p);
    let mut candidates = reg.algebras_with_fn("add", 2);
    candidates.sort();
    assert_eq!(candidates, vec!["Ring", "Tropical"], "both declare a 2-arg `add`");
}

#[test]
fn arity_mismatch_excludes_a_candidate() {
    let p = program("algebra Ring<T> { fn add(a: T, b: T) -> T; }");
    let reg = Registry::build(&p);
    assert!(reg.algebras_with_fn("add", 3).is_empty());
}

#[test]
fn fn_sig_lookup_returns_the_declared_signature() {
    let p = program("algebra Ring<T> { fn add(a: T, b: T) -> T; }");
    let reg = Registry::build(&p);
    let sig = reg.fn_sig("Ring", "add").expect("Ring declares add");
    assert_eq!(sig.params.len(), 2);
}

#[test]
fn axioms_are_not_counted_as_fn_signatures() {
    let p = program(
        "algebra Ring<T> { fn add(a: T, b: T) -> T; axiom comm(a: T, b: T): add(a,b) == add(b,a); }",
    );
    let reg = Registry::build(&p);
    assert!(reg.fn_sig("Ring", "comm").is_none(), "an axiom name isn't a callable signature");
    assert!(reg.algebras_with_fn("comm", 2).is_empty());
}

#[test]
fn builds_from_a_program_with_no_algebras_at_all() {
    let p = program("fn main() -> i32 { 0 }");
    let reg = Registry::build(&p);
    assert!(!reg.has_algebra("Ring"));
    assert!(reg.algebras_with_fn("add", 2).is_empty());
}

#[test]
fn generic_impls_with_the_same_bare_target_shape_but_different_bounds_both_survive() {
    // A real bug, found by testing: indexing generic impls purely by
    // `fmt_type(target)` collided two structurally-different impls whose
    // bare target both stringify as `Complex<T>` (bounds live on the
    // impl's own `generics`, not on `target`) -- the second silently
    // overwrote the first in the registry's own map before any later
    // overlap/coherence check could ever see both.
    let p = program(
        "algebra Ring<T> { fn add(a: T, b: T) -> T; }
         struct Complex<T> { real: T, imag: T }
         impl<T: Float> Ring<Complex<T>> { fn add(a: Complex<T>, b: Complex<T>) -> Complex<T> { a } }
         impl<T: Ord> Ring<Complex<T>> { fn add(a: Complex<T>, b: Complex<T>) -> Complex<T> { a } }",
    );
    let reg = Registry::build(&p);
    assert_eq!(reg.generic_impls("Ring").len(), 2, "both impls should be indexed, not just one");
}

#[test]
fn finds_an_inherent_method_by_struct_and_method_name() {
    let p = program("struct Vec2 { x: f64, y: f64 }\nimpl struct Vec2 {\n    fn len(v) { v.x }\n}");
    let reg = Registry::build(&p);
    let entry = reg.inherent_method("Vec2", "len").expect("Vec2 declares an inherent `len`");
    assert_eq!(entry.method.name, "len");
    assert!(entry.generics.is_empty());
}

#[test]
fn no_inherent_method_for_an_unknown_name_or_struct() {
    let p = program("struct Vec2 { x: f64, y: f64 }\nimpl struct Vec2 {\n    fn len(v) { v.x }\n}");
    let reg = Registry::build(&p);
    assert!(reg.inherent_method("Vec2", "bogus").is_none());
    assert!(reg.inherent_method("Bogus", "len").is_none());
}

#[test]
fn generic_inherent_impls_own_generics_are_indexed() {
    let p = program("struct Matrix<T> { data: T }\nimpl<T: Float> struct Matrix<T> {\n    fn get(m) { m }\n}");
    let reg = Registry::build(&p);
    let entry = reg.inherent_method("Matrix", "get").expect("Matrix declares an inherent `get`");
    assert_eq!(entry.generics.len(), 1);
}

#[test]
fn all_impls_returns_every_target_in_declaration_order() {
    let p = program(
        "algebra MatMul<A, B, C> { fn mul(a: A, b: B) -> C; }
         impl<T> MatMul<T, T, T> { fn mul(a, b) { a } }",
    );
    let reg = Registry::build(&p);
    let impls = reg.all_impls("MatMul");
    assert_eq!(impls.len(), 1);
    let (generics, targets) = &impls[0];
    assert_eq!(generics.len(), 1);
    assert_eq!(targets.len(), 3, "A, B, C -- three targets");
}

#[test]
fn all_impls_includes_single_target_impls_too() {
    let p = program("algebra Ring<T> { fn add(a: T, b: T) -> T; }\nimpl Ring<i32> { fn add(a, b) { a } }");
    let reg = Registry::build(&p);
    let impls = reg.all_impls("Ring");
    assert_eq!(impls.len(), 1);
    assert_eq!(impls[0].1.len(), 1);
}

/// An `axiom` declared inside an `algebra` block is retained by the
/// `Registry`, not silently discarded the way it used to be (`registry.rs`
/// previously filtered `AlgebraItemKind::Axiom(_)` straight to `None` while
/// building `sigs` — never stored anywhere at all) — the first prerequisite
/// for anything downstream (an e-graph pass) to eventually turn one into a
/// real rewrite rule.
#[test]
fn registry_retains_axioms_declared_on_an_algebra() {
    let p = program(
        "algebra Ring<T> {
            fn add(a: T, b: T) -> T;
            axiom add_commutative(a, b): add(a, b) == add(b, a);
         }",
    );
    let reg = Registry::build(&p);
    let axioms = reg.axioms("Ring");
    assert_eq!(axioms.len(), 1, "expected exactly one retained axiom");
    assert_eq!(axioms[0].name, "add_commutative");
    assert_eq!(axioms[0].params.len(), 2);
}

/// An algebra with no axioms at all still resolves (empty, not missing) --
/// same convention every other `Registry` accessor in this file already
/// follows for an absent entry.
#[test]
fn registry_axioms_is_empty_not_missing_when_none_are_declared() {
    let p = program("algebra Ring<T> { fn add(a: T, b: T) -> T; }");
    let reg = Registry::build(&p);
    assert!(reg.axioms("Ring").is_empty());
    assert!(reg.axioms("NoSuchAlgebra").is_empty());
}

fn vec2_type() -> cleave::ast::Type {
    type_from("Vec2")
}

fn bool_type() -> cleave::ast::Type {
    type_from("bool")
}

fn type_from(name: &str) -> cleave::ast::Type {
    let src = format!("fn f(x: {name}) {{ x }}");
    let p = program(&src);
    match &p.items[0].kind {
        cleave::ast::ItemKind::Fn(f) => f.params[0].ty.clone().unwrap(),
        other => panic!("expected fn item, got {other:?}"),
    }
}
