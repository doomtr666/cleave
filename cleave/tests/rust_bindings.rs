use cleave::cps::{collect_units, convert_program, eliminate_dead_code};
use cleave::driver::compile;
use cleave::registry::Registry;
use cleave::rust_bindings::generate_rust_bindings;

fn bindings(src: &str) -> Result<String, Vec<String>> {
    let (result, _sources) = compile(vec![("test.cleave".to_string(), src.to_string())], &[]);
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    let cps_program = convert_program(collect_units(&program, &registry));
    let cps_program = eliminate_dead_code(cps_program);
    generate_rust_bindings(&cps_program.funcs)
}

#[test]
fn a_scalar_export_fn_generates_a_matching_extern_c_declaration() {
    let out = bindings("export fn cleave_add(a: i32, b: i32) -> i32 { a + b }").expect("expected a successful generation");
    assert!(out.contains("extern \"C\""), "got:\n{out}");
    assert!(out.contains("pub fn cleave_add(a0: i32, a1: i32) -> i32;"), "got:\n{out}");
}

#[test]
fn a_symbol_override_uses_the_overridden_name_not_the_cleave_name() {
    let out = bindings("export(real_symbol) fn cleave_add(a: i32, b: i32) -> i32 { a + b }")
        .expect("expected a successful generation");
    assert!(out.contains("pub fn real_symbol(a0: i32, a1: i32) -> i32;"), "got:\n{out}");
    assert!(!out.contains("cleave_add"), "the plain cleave name must not leak into the binding once overridden, got:\n{out}");
}

#[test]
fn a_unit_returning_export_fn_omits_a_return_type() {
    let out = bindings("export fn touch(x: i32) { }").expect("expected a successful generation");
    assert!(out.contains("pub fn touch(a0: i32);"), "a unit return must have no `->` clause at all, got:\n{out}");
}

#[test]
fn a_non_exported_fn_is_not_included_at_all() {
    let out = bindings("fn helper(x: i32) -> i32 { x }\nexport fn kernel(x: i32) -> i32 { x }")
        .expect("expected a successful generation");
    assert!(!out.contains("helper"), "an ordinary, non-exported fn must not appear in the bindings, got:\n{out}");
    assert!(out.contains("kernel"), "got:\n{out}");
}

#[test]
fn multiple_exports_each_get_their_own_declaration_in_one_extern_c_block() {
    let out = bindings("export fn a(x: i32) -> i32 { x }\nexport fn b(x: f64) -> f64 { x }").expect("expected a successful generation");
    assert!(out.contains("pub fn a(a0: i32) -> i32;"), "got:\n{out}");
    assert!(out.contains("pub fn b(a0: f64) -> f64;"), "got:\n{out}");
    assert_eq!(out.matches("extern \"C\"").count(), 1, "expected a single shared extern block, got:\n{out}");
}

#[test]
fn an_array_argument_on_an_export_fn_is_a_reported_error_not_a_panic() {
    let err = bindings("export fn f(x: [i32; 3]) -> i32 { x[0] }").expect_err("an array-typed export fn signature isn't supported yet");
    assert!(!err.is_empty());
    assert!(err[0].contains("f"), "the error should name the offending fn, got: {err:?}");
}
