use cleave::ast::{FileId, ItemKind};
use cleave::driver::stdlib_path;
use cleave::lower::Lowerer;
use cleave::parser::{CleaveParser, Rule};
use cleave::registry::Registry;
use pest::Parser;
use std::fs;

#[test]
fn stdlib_path_resolves_to_a_real_directory() {
    let path = stdlib_path().expect("stdlib directory should be found relative to the test binary");
    assert!(path.is_dir(), "resolved path {path:?} is not a directory");
}

#[test]
fn stdlib_num_declares_impls_for_core_numeric_types() {
    // `num` is a *directory* (a crate), not a flat file — see grammar.md's
    // "a directory is a crate, strictly" rule.
    //
    // No type here has a *direct* `impl Num<...>` any more — each reaches
    // `Num` through `Int`/`Float`'s own bound on it instead (`algebra
    // Int<T> : Num`, see `num.cleave`'s own doc comment), so this checks
    // `Infer::has_matching_impl` (via a real bound on a generic parameter,
    // its only public-API entry point) rather than `Registry::has_impl`,
    // which only ever sees direct impls by design.
    let registry = load_num_registry();
    assert!(registry.has_algebra("Num"));
    for ty in ["i8", "i16", "i32", "i64", "f32", "f64"] {
        assert!(
            satisfies_bound(&registry, "Num", ty),
            "expected {ty} to satisfy a `Num` bound"
        );
    }
    assert!(
        !satisfies_bound(&registry, "Num", "bool"),
        "bool must not be considered Num"
    );
}

#[test]
fn stdlib_num_splits_int_and_float_as_independent_algebras() {
    // A numeric literal's own shape (`.` or not) is a real, checked
    // constraint against these — see `infer.rs`'s `NumberLit` handling.
    // `Int`/`Float` both bound `Num` (see `num.cleave`'s own doc comment)
    // but stay independent of *each other* — an integer type must still
    // never satisfy a `Float` bound and vice versa.
    let registry = load_num_registry();
    assert!(registry.has_algebra("Int"));
    assert!(registry.has_algebra("Float"));
    for ty in ["i8", "i16", "i32", "i64"] {
        assert!(
            satisfies_bound(&registry, "Int", ty),
            "expected {ty} to satisfy an `Int` bound"
        );
        assert!(
            !satisfies_bound(&registry, "Float", ty),
            "{ty} must not be considered Float"
        );
    }
    for ty in ["f32", "f64"] {
        assert!(
            satisfies_bound(&registry, "Float", ty),
            "expected {ty} to satisfy a `Float` bound"
        );
        assert!(
            !satisfies_bound(&registry, "Int", ty),
            "{ty} must not be considered Int"
        );
    }
}

fn load_num_registry() -> Registry {
    let path = stdlib_path().unwrap().join("num").join("num.cleave");
    let src = fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {path:?}: {e}"));
    let pair = CleaveParser::parse(Rule::program, &src)
        .unwrap_or_else(|e| panic!("failed to parse {path:?}: {e}"))
        .next()
        .unwrap();
    let program = Lowerer::new(FileId(0)).lower_program(pair);
    Registry::build(&program)
}

/// Whether `ty` satisfies a `bound` generic-parameter constraint under
/// `registry` — pins a generic parameter's declared return type to `ty`
/// concretely within one function's own inference, exactly the way
/// `tests/infer.rs`'s own bound-inheritance tests do, since `Infer::
/// has_matching_impl` (the method that actually walks algebra-bound
/// inheritance) isn't `pub`.
fn satisfies_bound(registry: &Registry, bound: &str, ty: &str) -> bool {
    let src = format!("fn f<T: {bound}>(x: T) -> {ty} {{ x }}");
    let pair = CleaveParser::parse(Rule::program, &src)
        .unwrap()
        .next()
        .unwrap();
    let program = Lowerer::new(FileId(0)).lower_program(pair);
    let f = match &program.items[0].kind {
        ItemKind::Fn(f) => f.clone(),
        other => panic!("expected fn item, got {other:?}"),
    };
    let mut infer = cleave::infer::Infer::new(registry);
    infer.infer_fn(&f).is_ok()
}
