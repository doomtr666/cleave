use cleave::ast::{FileId, ItemKind};
use cleave::driver::{merge_programs, parse_file, FileSource};
use cleave::print::print_program;

fn file(id: u32, name: &str, text: &str) -> FileSource {
    FileSource { id: FileId(id), name: name.to_string(), text: text.to_string() }
}

#[test]
fn algebra_fragments_across_files_are_merged() {
    let f1 = file(0, "a.cleave", "algebra Ring<T> { fn add(a: T, b: T) -> T; }");
    let f2 = file(1, "b.cleave", "algebra Ring<T> { axiom assoc_add(a: T, b: T, c: T): add(add(a,b),c) == add(a,add(b,c)); }");

    let p1 = parse_file(&f1).unwrap();
    let p2 = parse_file(&f2).unwrap();
    let merged = merge_programs(vec![p1, p2]).unwrap();

    assert_eq!(merged.items.len(), 1, "the two fragments should merge into one algebra item");
    let out = print_program(&merged);
    assert!(out.contains("fn add(a: T, b: T) -> T;"), "got:\n{out}");
    assert!(out.contains("axiom assoc_add"), "got:\n{out}");
}

#[test]
fn merge_is_order_independent() {
    let f1 = file(0, "a.cleave", "algebra Ring<T> { fn add(a: T, b: T) -> T; }");
    let f2 = file(1, "b.cleave", "algebra Ring<T> { fn sub(a: T, b: T) -> T; }");

    let forward = merge_programs(vec![parse_file(&f1).unwrap(), parse_file(&f2).unwrap()]).unwrap();
    let backward = merge_programs(vec![parse_file(&f2).unwrap(), parse_file(&f1).unwrap()]).unwrap();

    let a = print_program(&forward);
    let b = print_program(&backward);
    assert!(a.contains("fn add") && a.contains("fn sub"), "got:\n{a}");
    assert!(b.contains("fn add") && b.contains("fn sub"), "got:\n{b}");
}

#[test]
fn impl_fragments_across_files_are_merged_by_algebra_and_target() {
    let f1 = file(0, "a.cleave", "impl Ring<Vec2> { fn add(a: Vec2, b: Vec2) -> Vec2 { a } }");
    let f2 = file(1, "b.cleave", "impl Ring<Vec2> { fn sub(a: Vec2, b: Vec2) -> Vec2 { a } }");

    let merged = merge_programs(vec![parse_file(&f1).unwrap(), parse_file(&f2).unwrap()]).unwrap();
    assert_eq!(merged.items.len(), 1);
    match &merged.items[0].kind {
        ItemKind::Impl(d) => assert_eq!(d.fns.len(), 2),
        other => panic!("expected an impl item, got {other:?}"),
    }
}

/// Real bug, found by direct testing while investigating a native MLIR
/// crash: `#[mlir_type(...)]` (`ImplDecl::attrs`) used to be silently
/// dropped when an attr-less fragment of the same `impl` (algebra + target
/// + generics) happened to be merged first — `merge_impl_fragment` only
/// ever merged `.fns`, never `.attrs`, so whichever fragment was processed
/// first silently determined the final, merged impl's own attrs, discarding
/// a *later* fragment's own tag entirely, with no diagnostic anywhere. A
/// local, unrelated `impl Float<f64> {}` (no tag) processed before
/// `stdlib/num/num.cleave`'s own `#[mlir_type(f64)] impl Float<f64> {}`
/// (the entry file's own items are always merged before any resolved
/// `use`/prelude crate's) reproduced this directly — `f64` silently lost
/// its own MLIR type text, later crashing `mlir_lower.rs::ty_to_mlir` with
/// a native assertion instead of a clean, attributable error.
#[test]
fn an_attrless_impl_fragment_adopts_a_sibling_fragments_attrs() {
    let f1 = file(0, "a.cleave", "impl Float<f64> {}");
    let f2 = file(1, "b.cleave", "#[mlir_type(f64)] impl Float<f64> {}");

    let merged = merge_programs(vec![parse_file(&f1).unwrap(), parse_file(&f2).unwrap()]).unwrap();
    assert_eq!(merged.items.len(), 1);
    match &merged.items[0].kind {
        ItemKind::Impl(d) => assert_eq!(d.attrs.len(), 1, "the tagged fragment's own attr must survive the merge"),
        other => panic!("expected an impl item, got {other:?}"),
    }
}

/// Same as above, fragments in the opposite order — the fix must not just
/// happen to work because the tagged fragment merges *into* the untagged
/// one; it has to work regardless of which side is the accumulator.
#[test]
fn an_attrless_impl_fragment_adopts_a_sibling_fragments_attrs_reversed_order() {
    let f1 = file(0, "a.cleave", "#[mlir_type(f64)] impl Float<f64> {}");
    let f2 = file(1, "b.cleave", "impl Float<f64> {}");

    let merged = merge_programs(vec![parse_file(&f1).unwrap(), parse_file(&f2).unwrap()]).unwrap();
    assert_eq!(merged.items.len(), 1);
    match &merged.items[0].kind {
        ItemKind::Impl(d) => assert_eq!(d.attrs.len(), 1, "the tagged fragment's own attr must survive the merge"),
        other => panic!("expected an impl item, got {other:?}"),
    }
}

#[test]
fn impl_fragments_disagreeing_on_attrs_is_a_conflict() {
    let f1 = file(0, "a.cleave", "#[mlir_type(f32)] impl Float<f64> {}");
    let f2 = file(1, "b.cleave", "#[mlir_type(f64)] impl Float<f64> {}");

    let errs = merge_programs(vec![parse_file(&f1).unwrap(), parse_file(&f2).unwrap()]).unwrap_err();
    assert_eq!(errs.len(), 1);
    assert!(errs[0].message.contains("disagree"), "got: {}", errs[0].message);
}

#[test]
fn impl_fragments_agreeing_on_attrs_are_not_a_conflict() {
    let f1 = file(0, "a.cleave", "#[mlir_type(f64)] impl Float<f64> {}");
    let f2 = file(1, "b.cleave", "#[mlir_type(f64)] impl Float<f64> { fn foo(x: f64) -> f64 { x } }");

    let merged = merge_programs(vec![parse_file(&f1).unwrap(), parse_file(&f2).unwrap()]).unwrap();
    match &merged.items[0].kind {
        ItemKind::Impl(d) => assert_eq!(d.attrs.len(), 1),
        other => panic!("expected an impl item, got {other:?}"),
    }
}

#[test]
fn overloads_with_different_param_types_coexist() {
    let f1 = file(0, "a.cleave", "algebra Ring<T> { fn add(a: T, b: T) -> T; }");
    let f2 = file(1, "b.cleave", "algebra Ring<T> { fn add(a: T, b: f64) -> T; }");

    let merged = merge_programs(vec![parse_file(&f1).unwrap(), parse_file(&f2).unwrap()]).unwrap();
    match &merged.items[0].kind {
        ItemKind::Algebra(d) => assert_eq!(d.items.len(), 2, "two distinct signatures should both survive"),
        other => panic!("expected an algebra item, got {other:?}"),
    }
}

#[test]
fn same_params_different_return_is_a_conflict() {
    // Nothing at a call site could ever disambiguate these by return type
    // alone — this must collide, not coexist as an "overload".
    let f1 = file(0, "a.cleave", "algebra Ring<T> { fn add(a: T, b: T) -> T; }");
    let f2 = file(1, "b.cleave", "algebra Ring<T> { fn add(a: T, b: T) -> bool; }");

    let errs = merge_programs(vec![parse_file(&f1).unwrap(), parse_file(&f2).unwrap()]).unwrap_err();
    assert_eq!(errs.len(), 1);
    assert!(errs[0].message.contains("add"), "got: {}", errs[0].message);
}

#[test]
fn true_duplicate_signature_is_a_conflict() {
    let f1 = file(0, "a.cleave", "algebra Ring<T> { fn add(a: T, b: T) -> T; }");
    let f2 = file(1, "b.cleave", "algebra Ring<T> { fn add(a: T, b: T) -> T; }");

    let errs = merge_programs(vec![parse_file(&f1).unwrap(), parse_file(&f2).unwrap()]).unwrap_err();
    assert_eq!(errs.len(), 1);
}

#[test]
fn duplicate_axiom_name_is_a_conflict() {
    let f1 = file(0, "a.cleave", "algebra Ring<T> { axiom comm(a: T, b: T): add(a,b) == add(b,a); }");
    let f2 = file(1, "b.cleave", "algebra Ring<T> { axiom comm(a: T, b: T): add(b,a) == add(a,b); }");

    let errs = merge_programs(vec![parse_file(&f1).unwrap(), parse_file(&f2).unwrap()]).unwrap_err();
    assert_eq!(errs.len(), 1);
    assert!(errs[0].message.contains("comm"), "got: {}", errs[0].message);
}

#[test]
fn incompatible_generics_across_fragments_is_a_conflict() {
    let f1 = file(0, "a.cleave", "algebra Ring<T> { fn add(a: T, b: T) -> T; }");
    let f2 = file(1, "b.cleave", "algebra Ring<T, U> { fn sub(a: T, b: U) -> T; }");

    let errs = merge_programs(vec![parse_file(&f1).unwrap(), parse_file(&f2).unwrap()]).unwrap_err();
    assert_eq!(errs.len(), 1);
}

#[test]
fn unannotated_params_skip_the_structural_conflict_check() {
    // Can't compare signatures structurally without types — deferred to
    // inference, not treated as an error here (see module docs).
    let f1 = file(0, "a.cleave", "algebra Ring<T> { fn add(a: T, b: T) -> T; }");
    let f2 = file(1, "b.cleave", "impl Ring<Vec2> { fn add(a, b) { a } }");

    let merged = merge_programs(vec![parse_file(&f1).unwrap(), parse_file(&f2).unwrap()]);
    assert!(merged.is_ok());
}

#[test]
fn duplicate_struct_across_files_is_a_conflict() {
    let f1 = file(0, "a.cleave", "struct Vec2 { x: f64, y: f64 }");
    let f2 = file(1, "b.cleave", "struct Vec2 { x: f64, y: f64 }");

    let errs = merge_programs(vec![parse_file(&f1).unwrap(), parse_file(&f2).unwrap()]).unwrap_err();
    assert_eq!(errs.len(), 1);
}

#[test]
fn duplicate_free_fn_across_files_is_a_conflict() {
    let f1 = file(0, "a.cleave", "fn main() -> i32 { 0 }");
    let f2 = file(1, "b.cleave", "fn main() -> i32 { 1 }");

    let errs = merge_programs(vec![parse_file(&f1).unwrap(), parse_file(&f2).unwrap()]).unwrap_err();
    assert_eq!(errs.len(), 1);
}

#[test]
fn unrelated_items_across_files_merge_cleanly() {
    let f1 = file(0, "a.cleave", "struct Vec2 { x: f64, y: f64 }");
    let f2 = file(1, "b.cleave", "fn main() -> i32 { 0 }");

    let merged = merge_programs(vec![parse_file(&f1).unwrap(), parse_file(&f2).unwrap()]).unwrap();
    assert_eq!(merged.items.len(), 2);
}

#[test]
fn generic_impls_sharing_a_bare_target_shape_but_different_bounds_stay_distinct() {
    // A real bug, found by testing: `fmt_type(target)` alone was used as the
    // "same impl fragment" key -- `impl<T: Float> Ring<Complex<T>>` and
    // `impl<T: Ord> Ring<Complex<T>>` both stringify their bare target as
    // `Complex<T>` (bounds live on the impl's own `generics`, not on
    // `target`), so the second was silently merged *into* the first's `fns`
    // as though they were the same impl block split in two, rather than
    // being kept as two independent impls. Two fns total, split 1/1 (not
    // combined into a single `d.fns.len() == 2`), is the tell.
    let f1 = file(
        0,
        "a.cleave",
        "struct Complex<T> { real: T, imag: T }
         impl<T: Float> Ring<Complex<T>> { fn add(a: Complex<T>, b: Complex<T>) -> Complex<T> { a } }",
    );
    let f2 = file(
        1,
        "b.cleave",
        "impl<T: Ord> Ring<Complex<T>> { fn lt(a: Complex<T>, b: Complex<T>) -> bool { true } }",
    );

    let merged = merge_programs(vec![parse_file(&f1).unwrap(), parse_file(&f2).unwrap()]).unwrap();
    let impls: Vec<_> = merged.items.iter().filter(|i| matches!(i.kind, ItemKind::Impl(_))).collect();
    assert_eq!(impls.len(), 2, "expected two distinct impls, got:\n{}", print_program(&merged));
    for item in impls {
        let ItemKind::Impl(d) = &item.kind else { unreachable!() };
        assert_eq!(d.fns.len(), 1, "each impl should keep only its own fn, got: {:?}", d.fns.iter().map(|f| &f.name).collect::<Vec<_>>());
    }
}

#[test]
fn inherent_impl_fragments_across_files_are_merged_by_target() {
    let f1 = file(0, "a.cleave", "struct Vec2 { x: f64, y: f64 }\nimpl struct Vec2 { fn len(v) { v.x } }");
    let f2 = file(1, "b.cleave", "impl struct Vec2 { fn scale(v) { v.x } }");

    let merged = merge_programs(vec![parse_file(&f1).unwrap(), parse_file(&f2).unwrap()]).unwrap();
    let inherent: Vec<_> = merged.items.iter().filter(|i| matches!(i.kind, ItemKind::InherentImpl(_))).collect();
    assert_eq!(inherent.len(), 1, "same target -- one merged fragment, got:\n{}", print_program(&merged));
    let ItemKind::InherentImpl(d) = &inherent[0].kind else { unreachable!() };
    assert_eq!(d.fns.len(), 2);
}

#[test]
fn duplicate_inherent_method_across_files_is_a_conflict() {
    let f1 = file(0, "a.cleave", "struct Vec2 { x: f64, y: f64 }\nimpl struct Vec2 { fn len(v) { v.x } }");
    let f2 = file(1, "b.cleave", "impl struct Vec2 { fn len(v) { v.y } }");

    let errs = merge_programs(vec![parse_file(&f1).unwrap(), parse_file(&f2).unwrap()]).unwrap_err();
    assert_eq!(errs.len(), 1);
    assert!(errs[0].message.contains("len"), "got: {}", errs[0].message);
}
