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
