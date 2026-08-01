use cleave::ast::{FileId, ItemKind, Type};
use cleave::driver::stdlib_path;
use cleave::lower::Lowerer;
use cleave::parser::{CleaveParser, Rule};
use cleave::registry::Registry;
use pest::Parser;
use std::fs;

fn type_from(name: &str) -> Type {
    let src = format!("fn f(x: {name}) {{ x }}");
    let pair = CleaveParser::parse(Rule::program, &src).unwrap().next().unwrap();
    let program = Lowerer::new(FileId(0)).lower_program(pair);
    match &program.items[0].kind {
        ItemKind::Fn(f) => f.params[0].ty.clone().unwrap(),
        other => panic!("expected fn item, got {other:?}"),
    }
}

#[test]
fn stdlib_path_resolves_to_a_real_directory() {
    let path = stdlib_path().expect("stdlib directory should be found relative to the test binary");
    assert!(path.is_dir(), "resolved path {path:?} is not a directory");
}

#[test]
fn stdlib_num_declares_impls_for_core_numeric_types() {
    // `num` is a *directory* (a crate), not a flat file — see grammar.md's
    // "a directory is a crate, strictly" rule.
    let path = stdlib_path().unwrap().join("num").join("num.cleave");
    let src = fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {path:?}: {e}"));

    let pair = CleaveParser::parse(Rule::program, &src)
        .unwrap_or_else(|e| panic!("failed to parse {path:?}: {e}"))
        .next()
        .unwrap();
    let program = Lowerer::new(FileId(0)).lower_program(pair);
    let registry = Registry::build(&program);

    assert!(registry.has_algebra("Num"));
    for ty in ["i8", "i16", "i32", "i64", "f32", "f64"] {
        assert!(registry.has_impl("Num", &type_from(ty)), "expected impl Num<{ty}>");
    }
    assert!(!registry.has_impl("Num", &type_from("bool")), "bool must not be considered Num");
}

#[test]
fn stdlib_num_splits_int_and_float_as_independent_algebras() {
    // A numeric literal's own shape (`.` or not) is a real, checked
    // constraint against these — see `infer.rs`'s `NumberLit` handling and
    // `stdlib/num/num.cleave`'s own doc comment on why `Int`/`Float` are
    // independent markers rather than `Int<T>: Num`/`Float<T>: Num`
    // (algebra-bound inheritance isn't implemented in the registry yet).
    let path = stdlib_path().unwrap().join("num").join("num.cleave");
    let src = fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {path:?}: {e}"));
    let pair = CleaveParser::parse(Rule::program, &src).unwrap().next().unwrap();
    let program = Lowerer::new(FileId(0)).lower_program(pair);
    let registry = Registry::build(&program);

    assert!(registry.has_algebra("Int"));
    assert!(registry.has_algebra("Float"));
    for ty in ["i8", "i16", "i32", "i64"] {
        assert!(registry.has_impl("Int", &type_from(ty)), "expected impl Int<{ty}>");
        assert!(!registry.has_impl("Float", &type_from(ty)), "{ty} must not be considered Float");
    }
    for ty in ["f32", "f64"] {
        assert!(registry.has_impl("Float", &type_from(ty)), "expected impl Float<{ty}>");
        assert!(!registry.has_impl("Int", &type_from(ty)), "{ty} must not be considered Int");
    }
}
