use cleave::ast::{Block, Expr, ExprKind, FileId, ItemKind, Program, StmtKind};
use cleave::infer::{ConstValue, Ty};
use cleave::lower::Lowerer;
use cleave::monomorphize::monomorphize;
use cleave::parser::{CleaveParser, Rule};
use cleave::registry::Registry;
use pest::Parser;

/// Finds the first `Call` expression whose callee is named `callee`,
/// anywhere inside `fn_name`'s own body — just enough of a search to locate
/// a specific recursive call site for a test assertion, not a general
/// traversal like `monomorphize.rs`'s own internal ones.
fn find_call<'a>(program: &'a Program, fn_name: &str, callee: &str) -> &'a Expr {
    let f = program
        .items
        .iter()
        .find_map(|item| match &item.kind {
            ItemKind::Fn(f) if f.name == fn_name => Some(f),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no fn named `{fn_name}` in program"));
    find_call_in_block(&f.body, callee).unwrap_or_else(|| panic!("no call to `{callee}` found inside `{fn_name}`"))
}

fn find_call_in_block<'a>(block: &'a Block, callee: &str) -> Option<&'a Expr> {
    for stmt in &block.stmts {
        let value = match &stmt.kind {
            StmtKind::Let { value, .. } | StmtKind::Assign { value, .. } => value,
            StmtKind::Expr(e) => e,
        };
        if let Some(found) = find_call_in_expr(value, callee) {
            return Some(found);
        }
    }
    block.tail.as_deref().and_then(|t| find_call_in_expr(t, callee))
}

fn find_call_in_expr<'a>(expr: &'a Expr, callee: &str) -> Option<&'a Expr> {
    if let ExprKind::Call(path, _, args) = &expr.kind {
        if path.segments == [callee.to_string()] {
            return Some(expr);
        }
        for a in args {
            if let Some(found) = find_call_in_expr(a, callee) {
                return Some(found);
            }
        }
    }
    match &expr.kind {
        ExprKind::If { cond, then_branch, else_branch } => find_call_in_expr(cond, callee)
            .or_else(|| find_call_in_block(then_branch, callee))
            .or_else(|| else_branch.as_deref().and_then(|eb| match eb {
                cleave::ast::ElseBranch::If(e) => find_call_in_expr(e, callee),
                cleave::ast::ElseBranch::Block(b) => find_call_in_block(b, callee),
            })),
        _ => None,
    }
}

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

/// Same test-only stand-in-for-a-real-stdlib fixture as `tests/callgraph.rs`
/// — see that file's own doc comment for why this isn't the final stdlib
/// design.
fn builtin_registry() -> Registry {
    registry_from(
        "algebra Ring<T> {
            fn add(a: T, b: T) -> T;
            fn sub(a: T, b: T) -> T;
        }
        algebra Ord<T> {
            fn eq(a: T, b: T) -> bool;
        }
        impl Ring<i32> { fn add(a: i32, b: i32) -> i32 { a } fn sub(a: i32, b: i32) -> i32 { a } }
        impl Ord<i32> { fn eq(a: i32, b: i32) -> bool { true } }",
    )
}

/// No leftover `Ty::Var` anywhere in a `Ty` — the exact class of bug this
/// whole pass exists to prevent (a monomorphized specialization that's
/// still, in some corner, generic).
fn assert_fully_concrete(ty: &Ty) {
    fn walk(ty: &Ty, path: &str) {
        match ty {
            Ty::Var(v) => panic!("found a leftover Ty::Var({v:?}) at {path} in supposedly-monomorphized type {ty}"),
            Ty::Con(_) | Ty::Const(_) => {}
            Ty::App(_, args) => args.iter().for_each(|a| walk(a, path)),
            Ty::Fn(params, ret) => {
                params.iter().for_each(|p| walk(p, path));
                walk(ret, path);
            }
            Ty::Array(elem, size) => {
                walk(elem, path);
                walk(size, path);
            }
        }
    }
    walk(ty, "<top>");
}

#[test]
fn a_generic_function_called_at_two_types_produces_two_specializations() {
    let registry = builtin_registry();
    let program = lower_program(
        "fn identity(x) { x }
        fn use_i32() -> i32 { identity(1) }
        fn use_f64() -> f64 { identity(1.5) }",
    );
    let (mono, _) = monomorphize(&program, &registry);
    let mut keys = mono.specializations_of("identity").to_vec();
    keys.sort();
    assert_eq!(keys, vec!["identity<f64>".to_string(), "identity<i32>".to_string()]);

    for key in &keys {
        for t in mono.node_types(key).values() {
            assert_fully_concrete(t);
        }
        assert_fully_concrete(mono.result(key));
        for t in mono.param_types(key) {
            assert_fully_concrete(t);
        }
    }
    assert_eq!(mono.result("identity<i32>"), &Ty::Con("i32".to_string()));
    assert_eq!(mono.result("identity<f64>"), &Ty::Con("f64".to_string()));
}

#[test]
fn a_non_generic_function_produces_no_specializations_of_itself() {
    let registry = builtin_registry();
    let program = lower_program("fn add_one(x: i32) -> i32 { x }");
    let (mono, _) = monomorphize(&program, &registry);
    assert!(mono.specializations_of("add_one").is_empty());
}

#[test]
fn two_mutually_recursive_generic_functions_each_get_exactly_one_specialization() {
    let registry = builtin_registry();
    let program = lower_program(
        "fn is_even(n) {
            if n == 0 { true } else { is_odd(n - 1) }
        }
        fn is_odd(n) {
            if n == 0 { false } else { is_even(n - 1) }
        }
        fn main() -> bool { is_even(4) }",
    );
    let (mono, _) = monomorphize(&program, &registry);
    assert_eq!(mono.specializations_of("is_even"), &["is_even<i32>".to_string()]);
    assert_eq!(mono.specializations_of("is_odd"), &["is_odd<i32>".to_string()]);
    assert_eq!(mono.result("is_even<i32>"), &Ty::Con("bool".to_string()));
    assert_eq!(mono.result("is_odd<i32>"), &Ty::Con("bool".to_string()));
}

#[test]
fn a_generic_function_calling_another_generic_function_transitively_instantiates_it() {
    let registry = builtin_registry();
    let program = lower_program(
        "fn identity(x) { x }
        fn wrapper(x) { identity(x) }
        fn use_it() -> i32 { wrapper(1) }",
    );
    let (mono, _) = monomorphize(&program, &registry);
    assert_eq!(mono.specializations_of("wrapper"), &["wrapper<i32>".to_string()]);
    // Discovered transitively (from inside `wrapper<i32>`'s own body), not
    // directly from any concrete entry point calling `identity` itself.
    assert_eq!(mono.specializations_of("identity"), &["identity<i32>".to_string()]);
}

#[test]
fn calling_the_same_instantiation_from_two_call_sites_does_not_duplicate_it() {
    let registry = builtin_registry();
    let program = lower_program(
        "fn identity(x) { x }
        fn a() -> i32 { identity(1) }
        fn b() -> i32 { identity(2) }",
    );
    let (mono, _) = monomorphize(&program, &registry);
    assert_eq!(mono.specializations_of("identity"), &["identity<i32>".to_string()]);
}

#[test]
fn a_generic_function_never_called_from_a_concrete_entry_point_gets_no_specialization() {
    let registry = builtin_registry();
    let program = lower_program(
        "fn identity(x) { x }
        fn never_called(x) { identity(x) }
        fn main() -> i32 { 0 }",
    );
    let (mono, _) = monomorphize(&program, &registry);
    assert!(mono.specializations_of("never_called").is_empty());
    assert!(mono.specializations_of("identity").is_empty());
}

#[test]
fn a_self_recursive_generic_functions_own_call_names_do_not_leak_across_instantiations() {
    // Real bug, found by direct testing: every instantiation of the same
    // generic `fn` shares the *same* body, and therefore the *same*
    // `NodeId`s (see `monomorphize.rs`'s own "no AST cloning" doc comment)
    // -- a self-recursive call site's `NodeId` is identical across
    // `fibonacci<i32>` and `fibonacci<i64>`. A first version kept one
    // *shared* `NodeId -> mangled name` map across every specialization,
    // so whichever instantiation was processed *last* silently overwrote
    // the other's own recursive-call resolution: `fibonacci<i64>`'s own
    // body rendered its recursive call as `fibonacci<i32>(...)`, wrong.
    let registry = registry_from(
        "algebra Ring<T> { fn add(a: T, b: T) -> T; }
        algebra Ord<T> { fn eq(a: T, b: T) -> bool; }
        impl Ring<i32> { fn add(a: i32, b: i32) -> i32 { a } }
        impl Ord<i32> { fn eq(a: i32, b: i32) -> bool { true } }
        impl Ring<i64> { fn add(a: i64, b: i64) -> i64 { a } }
        impl Ord<i64> { fn eq(a: i64, b: i64) -> bool { true } }",
    );
    let program = lower_program(
        "fn fibonacci(x) {
            if x == 0 { x } else { add(fibonacci(x), fibonacci(x)) }
        }
        fn use_i32() -> i32 { fibonacci(1) }
        fn use_i64() -> i64 { fibonacci(1:i64) }",
    );
    let (mono, _) = monomorphize(&program, &registry);
    let mut keys = mono.specializations_of("fibonacci").to_vec();
    keys.sort();
    assert_eq!(keys, vec!["fibonacci<i32>".to_string(), "fibonacci<i64>".to_string()]);

    let call = find_call(&program, "fibonacci", "fibonacci");
    assert_eq!(mono.call_names("fibonacci<i32>").get(&call.id), Some(&"fibonacci<i32>".to_string()));
    assert_eq!(mono.call_names("fibonacci<i64>").get(&call.id), Some(&"fibonacci<i64>".to_string()));
}

#[test]
fn a_seed_functions_own_call_to_a_generic_callee_resolves_to_the_right_specialization() {
    let registry = builtin_registry();
    let program = lower_program(
        "fn identity(x) { x }
        fn main() -> i32 { identity(5) }",
    );
    let (mono, _) = monomorphize(&program, &registry);
    let call = find_call(&program, "main", "identity");
    assert_eq!(mono.seed_call_names().get(&call.id), Some(&"identity<i32>".to_string()));
}

// ---------------------------------------------------------------------
// generic algebra-impl methods (`MatMul`-style) -- the same worklist,
// discovered via structural dispatch instead of a bare name lookup, see
// `monomorphize.rs`'s own doc comment on the "one unified algorithm"
// design.
// ---------------------------------------------------------------------

#[test]
fn a_generic_algebra_impl_method_is_specialized_at_a_concrete_call_site() {
    // Deliberately *one* combined source, unlike the top-level-`fn`-only
    // tests above, which can freely split the caller and the registry's
    // own fixture into two separate strings: `build_impl_templates` scans
    // `program.items` directly for `ItemKind::Impl`, so the impl itself
    // must actually be part of the `Program` being monomorphized, not just
    // known to a separately-built `Registry`.
    let src = "algebra MatMul<A, B, C> { fn mul(a: A, b: B) -> C; }
         struct Matrix<T, const R: i32, const C: i32> { values: T }
         impl<T, const N: i32, const M: i32, const K: i32>
             MatMul<Matrix<T,N,M>, Matrix<T,M,K>, Matrix<T,N,K>> { fn mul(a, b) { a } }
         fn f() -> i32 {
            let a = Matrix::<f32, 2, 2>(values: 1.0);
            let b = Matrix::<f32, 2, 2>(values: 1.0);
            let c = a * b;
            0
        }";
    let registry = registry_from(src);
    let program = lower_program(src);
    let (mono, _) = monomorphize(&program, &registry);
    let keys = mono.specializations_of("MatMul::mul");
    assert_eq!(keys, &["MatMul::mul<Matrix<f32, 2, 2>, Matrix<f32, 2, 2>, Matrix<f32, 2, 2>>".to_string()]);
    assert_eq!(
        mono.result(&keys[0]),
        &Ty::App("Matrix".to_string(), vec![Ty::Con("f32".to_string()), Ty::Const(ConstValue::Int(2)), Ty::Const(ConstValue::Int(2))])
    );

    // The call site itself must show the mangled name too, not the bare
    // (ambiguous, once other instantiations exist) `mul`.
    let call = find_call(&program, "f", "mul");
    assert_eq!(mono.seed_call_names().get(&call.id), Some(&keys[0]));
}

#[test]
fn a_stub_bodys_declaration_time_type_error_is_a_real_unification_conflict_for_a_rectangular_instantiation() {
    // Not a monomorphization bug -- a real, pre-existing consequence of the
    // stub body `fn mul(a, b) { a }` (used throughout this project's own
    // examples/tests, standing in for a real matmul implementation): its
    // *declared* return type is `C` (`Matrix<T,N,K>`), but it actually
    // returns `a`'s own type (`A` = `Matrix<T,N,M>`) -- declaration-time
    // inference unifies these, forcing `M = K` in the impl's own template.
    // A *square* call site (M happens to equal K) unifies fine by
    // coincidence (see the test above); a genuinely *rectangular* one
    // (M != K) hits a real conflict the moment monomorphization tries to
    // unify the template's own (now M=K-merged) pattern against the
    // concrete, non-square query -- correctly discovered as "no matching
    // template", not silently accepted with a wrong result.
    let src = "algebra MatMul<A, B, C> { fn mul(a: A, b: B) -> C; }
         struct Matrix<T, const R: i32, const C: i32> { values: T }
         impl<T, const N: i32, const M: i32, const K: i32>
             MatMul<Matrix<T,N,M>, Matrix<T,M,K>, Matrix<T,N,K>> { fn mul(a, b) { a } }
         fn f() -> i32 {
            let a = Matrix::<f32, 2, 3>(values: 1.0);
            let b = Matrix::<f32, 3, 5>(values: 1.0);
            let c = a * b;
            0
        }";
    let registry = registry_from(src);
    let program = lower_program(src);
    let (mono, _) = monomorphize(&program, &registry);
    assert!(mono.specializations_of("MatMul::mul").is_empty(), "the M=K-merged template cannot satisfy a rectangular call");
}

#[test]
fn a_non_generic_algebra_impl_needs_no_specialization_and_is_unaffected() {
    let src = "algebra Ring<T> { fn add(a: T, b: T) -> T; }
         impl Ring<i32> { fn add(a: i32, b: i32) -> i32 { a } }
         fn f() -> i32 { 1 + 2 }";
    let registry = registry_from(src);
    let program = lower_program(src);
    let (mono, _) = monomorphize(&program, &registry);
    assert!(mono.specializations_of("Ring::add").is_empty());
}

#[test]
fn a_multi_target_algebras_own_concrete_impl_dispatches_correctly_from_inside_a_generic_sibling() {
    // Regression test: `MatMul` has *two* impls -- a concrete
    // `MatMul<f32,f32,f32>` (needed by the generic impl's own body below,
    // for its element-wise scalar multiply) and the generic
    // `MatMul<Matrix<T,N,M>,...>`. `a.values * b.values` (both `T`, still
    // abstract at the generic impl's own declaration time) must defer to a
    // single `Constraint` holding the *whole* `(A,B,C)` tuple together (see
    // `Constraint`'s own doc comment) -- checking `A`, `B`, `C` as three
    // independent, single-type constraints could never verify a combined
    // 3-target impl exists (`no impl MatMul<f32>`, checking one slot alone).
    // Separately, recognizing the concrete impl already covers this call
    // (so it needs no specialization of its own) requires a *combined*
    // structural match too -- `Registry`'s own multi-target key
    // (`"f32f32f32"`, the concatenation of all three targets) is invisible
    // to a per-type name lookup (see `ImplTemplate::is_generic`'s own doc
    // comment) -- found by direct testing via `examples/matmul.cleave`.
    let src = "algebra MatMul<A, B, C> { fn mul(a: A, b: B) -> C; }
         impl MatMul<f32, f32, f32> { fn mul(a, b) { a } }
         struct Matrix<T, const R: i32, const C: i32> { values: T }
         impl<T, const N: i32, const M: i32, const K: i32>
             MatMul<Matrix<T,N,M>, Matrix<T,M,K>, Matrix<T,N,K>> {
             fn mul(a, b) {
                 let x = a.values * b.values;
                 Matrix(values: x)
             }
         }
         fn f() -> i32 {
            let a = Matrix::<f32, 2, 2>(values: 1.0);
            let b = Matrix::<f32, 2, 2>(values: 1.0);
            let c = a * b;
            0
        }";
    let registry = registry_from(src);
    let program = lower_program(src);
    let (mono, _) = monomorphize(&program, &registry);
    assert!(mono.errors().is_empty(), "expected no monomorphization errors, got {:?}", mono.errors());
    let keys = mono.specializations_of("MatMul::mul");
    assert_eq!(keys, &["MatMul::mul<Matrix<f32, 2, 2>, Matrix<f32, 2, 2>, Matrix<f32, 2, 2>>".to_string()]);
}
