use cleave::parser::{CleaveParser, Rule};
use pest::Parser;

fn parses(rule: Rule, input: &str) {
    match CleaveParser::parse(rule, input) {
        Ok(mut pairs) => {
            let pair = pairs.next().unwrap();
            assert_eq!(pair.as_span().as_str(), input, "did not consume the whole input");
        }
        Err(e) => panic!("failed to parse {input:?} as {rule:?}: {e}"),
    }
}

#[test]
fn lambda_with_body() {
    parses(Rule::lambda_expr, "fn(a, b) { a + b }");
}

#[test]
fn lambda_no_params() {
    parses(Rule::lambda_expr, "fn() { 1 }");
}

#[test]
fn lambda_annotated_params_and_return() {
    parses(Rule::lambda_expr, "fn(a: f64, b: f64) -> f64 { a + b }");
}

#[test]
fn lambda_as_let_value() {
    parses(Rule::let_stmt, "let f = fn(a, b) { a + b };");
}

#[test]
fn lambda_literal_called_directly() {
    parses(Rule::expr, "(fn(a, b) { a + b })(1, 2)");
}

#[test]
fn let_binding_no_type() {
    parses(Rule::let_stmt, "let a = c + b;");
}

#[test]
fn let_binding_with_type() {
    parses(Rule::let_stmt, "let a : int32 = c + d;");
}

#[test]
fn let_mut_binding() {
    parses(Rule::let_stmt, "let mut acc = 0;");
}

#[test]
fn reassignment() {
    parses(Rule::assign_stmt, "acc = acc + a;");
}

#[test]
fn fn_no_types_inferred() {
    parses(Rule::fn_decl, "fn add(a, b) { a + b }");
}

#[test]
fn fn_with_types() {
    parses(Rule::fn_decl, "fn add(a: i32, b: i32) -> i32 { a + b }");
}

#[test]
fn a_bare_attribute_parses() {
    parses(Rule::attribute, "#[export]");
}

#[test]
fn an_attribute_with_an_ident_argument_parses() {
    parses(Rule::attribute, "#[mlir(mlir_f32_add_instruction)]");
}

#[test]
fn a_bodyless_fn_parses() {
    // Legal grammatically anywhere a `fn` appears (see `grammar.pest`'s own
    // `fn_decl` comment) -- the restriction to "only inside an algebra
    // impl, and only with a recognized attribute" is enforced later, by
    // `infer.rs`, not by the parser.
    parses(Rule::fn_decl, "fn add(a: i32, b: i32) -> i32;");
}

#[test]
fn an_attributed_bodyless_fn_parses() {
    parses(Rule::fn_decl, "#[mlir(mlir_i32_add)] fn add(a: i32, b: i32) -> i32;");
}

#[test]
fn main_entry_point() {
    parses(Rule::fn_decl, "fn main() -> i32 { 0 }");
}

#[test]
fn if_expression() {
    parses(Rule::fn_decl, "fn min(a, b) { if a < b { a } else { b } }");
}

#[test]
fn logical_and_or() {
    parses(Rule::expr, "x > 0 and x < 100 or y == 0");
}

#[test]
fn imaginary_literal() {
    parses(Rule::expr, "3 + 4i");
}

#[test]
fn struct_decl() {
    parses(Rule::struct_decl, "struct Vec2 { x: f64, y: f64 }");
}

#[test]
fn struct_lit_parses_as_struct_lit() {
    parses(Rule::struct_lit, "Vec2(x: 1.0, y: 2.0)");
    parses(Rule::expr, "Vec2(x: 1.0, y: 2.0)");
}

#[test]
fn struct_lit_with_a_single_field() {
    parses(Rule::expr, "Vec2(x: 1.0)");
}

#[test]
fn zero_arg_call_is_not_parsed_as_a_struct_literal() {
    // The ambiguity `grammar.pest`'s `primary` comment documents: `f()` and
    // a zero-*field* struct construction are syntactically identical, and
    // `call_expr` must win — checked here by inspecting *which* alternative
    // `primary` actually picked, not just that something parsed.
    let pair = CleaveParser::parse(Rule::primary, "f()").unwrap().next().unwrap();
    let inner = pair.into_inner().next().unwrap();
    assert_eq!(inner.as_rule(), Rule::call_expr, "got: {:?}", inner.as_rule());
}

#[test]
fn named_field_call_is_parsed_as_a_struct_literal() {
    let pair = CleaveParser::parse(Rule::primary, "Vec2(x: 1.0, y: 2.0)").unwrap().next().unwrap();
    let inner = pair.into_inner().next().unwrap();
    assert_eq!(inner.as_rule(), Rule::struct_lit, "got: {:?}", inner.as_rule());
}

#[test]
fn algebra_decl_with_axiom() {
    parses(
        Rule::algebra_decl,
        "algebra Ring<T> {\n\
         \x20   fn add(a: T, b: T) -> T;\n\
         \x20   axiom assoc_add(a: T, b: T, c: T): add(add(a,b),c) == add(a,add(b,c));\n\
         }",
    );
}

#[test]
fn impl_decl() {
    parses(
        Rule::impl_decl,
        "impl Ring<Vec2> {\n\
         \x20   fn add(a: Vec2, b: Vec2) -> Vec2 { Vec2(a.x + b.x, a.y + b.y) }\n\
         }",
    );
}

#[test]
fn generic_impl_decl_with_a_bounded_generic_target() {
    parses(
        Rule::impl_decl,
        "impl<T: Float> Ring<Complex<T>> {\n\
         \x20   fn add(a: Complex<T>, b: Complex<T>) -> Complex<T> { a }\n\
         }",
    );
}

#[test]
fn field_access_and_call() {
    parses(Rule::expr, "Vec2(a.x + b.x, a.y + b.y)");
}

#[test]
fn array_type_param() {
    parses(Rule::fn_decl, "fn sum(a: [f64; 4]) -> f64 { 0.0 }");
}

#[test]
fn nested_array_type_multidim() {
    // [[f64; 4]; 3] — a 3x4 matrix's worth of storage, via recursive nesting,
    // no dedicated multi-dim array syntax needed (array_type's element is `type_`,
    // which is itself recursively `array_type`-capable).
    parses(Rule::fn_decl, "fn m(a: [[f64; 4]; 3]) -> f64 { 0.0 }");
}

#[test]
fn fortran_style_array_type_sugar() {
    parses(Rule::fn_decl, "fn m(a: [f64; 3, 4]) -> f64 { 0.0 }");
}

#[test]
fn fortran_style_indexing_sugar() {
    parses(Rule::expr, "a[i, j] + a[i, j + 1]");
}

#[test]
fn array_literal() {
    parses(Rule::expr, "[1, 2, 3]");
}

#[test]
fn array_repeat_literal_parses() {
    parses(Rule::expr, "[0.0; 4]");
}

#[test]
fn array_repeat_literal_count_can_name_a_const_generic() {
    // `n` here names a const generic of the enclosing fn/impl -- resolved
    // through ordinary type inference (`infer.rs`), not expanded at lowering
    // time the way a literal count is (see `grammar.pest`'s own comment).
    parses(Rule::expr, "[0.0; n]");
}

#[test]
fn array_repeat_literal_count_is_not_a_general_expression() {
    // Deliberately restricted to `numeric_lit | ident` -- a compile-time
    // integer (a literal or a const-generic reference), not a computation.
    assert!(CleaveParser::parse(Rule::expr, "[0.0; n + 1]").is_err());
}

#[test]
fn array_indexing() {
    parses(Rule::expr, "a[0] + a[1]");
}

#[test]
fn array_literal_then_index() {
    parses(Rule::expr, "[1, 2, 3][0]");
}

#[test]
fn while_loop_parses() {
    parses(Rule::while_expr, "while a < 10 { a }");
}

#[test]
fn for_loop_parses() {
    parses(Rule::for_expr, "for i in 0..n { i }");
}

#[test]
fn implication() {
    parses(Rule::expr, "a and b implies c or d");
}

#[test]
fn unicode_identifier_greek() {
    parses(Rule::let_stmt, "let π = 3.14159;");
}

#[test]
fn unicode_identifier_in_expr() {
    parses(Rule::expr, "τ * r");
}

#[test]
fn use_declaration() {
    parses(Rule::use_decl, "use linalg::Matrix;");
}

#[test]
fn qualified_call() {
    parses(Rule::expr, "Ring::add(a, b)");
}

#[test]
fn qualified_type() {
    parses(Rule::fn_decl, "fn f(a: linalg::Matrix<f64, 3, 3>) -> f64 { 0.0 }");
}

#[test]
fn bool_const_generic_argument_parses() {
    parses(Rule::fn_decl, "fn f(a: Grid<f64, true>) -> f64 { 0.0 }");
}

#[test]
fn bool_const_generic_argument_is_not_parsed_as_a_bogus_type_path() {
    // `ident` has no keyword exclusion anywhere in this grammar (see
    // `grammar.pest`'s own `generic_arg` comment) -- `true`/`false` would
    // otherwise happily parse as an ordinary `path`, the same class of
    // ambiguity `zero_arg_call_is_not_parsed_as_a_struct_literal` guards for
    // calls vs. struct literals. `bool_lit` must win the race, not `type_`.
    let pair = CleaveParser::parse(Rule::generic_arg, "true").unwrap().next().unwrap();
    let inner = pair.into_inner().next().unwrap();
    assert_eq!(inner.as_rule(), Rule::bool_lit, "got: {:?}", inner.as_rule());
}

#[test]
fn two_files_same_crate_directory() {
    // A crate is its directory's `.cleave` files, collectively — not something a single
    // file declares about itself. Both "files" below just parse as ordinary top-level
    // items; which crate they belong to is the compiler driver's job (which directory
    // it found them in when resolving a `use`), not something expressed in the grammar.
    let file_a = "struct Vec2 { x: f64, y: f64 }";
    let file_b = "impl Ring<Vec2> { fn add(a: Vec2, b: Vec2) -> Vec2 { Vec2(a.x + b.x, a.y + b.y) } }";
    parses(Rule::program, file_a);
    parses(Rule::program, file_b);
}

#[test]
fn full_program() {
    let src = "\
struct Vec2 { x: f64, y: f64 }

algebra Ring<T> {
    fn add(a: T, b: T) -> T;
    axiom assoc_add(a: T, b: T, c: T): add(add(a,b),c) == add(a,add(b,c));
}

impl Ring<Vec2> {
    fn add(a: Vec2, b: Vec2) -> Vec2 { Vec2(a.x + b.x, a.y + b.y) }
}

fn main() -> i32 {
    let a = 1;
    let mut acc = 0;
    acc = acc + a;
    0
}
";
    parses(Rule::program, src);
}

#[test]
fn turbofish_on_a_call() {
    parses(Rule::expr, "fibonacci::<f64>(x)");
}

#[test]
fn turbofish_on_a_struct_construction() {
    parses(Rule::expr, "Matrix::<f64, 4, 4>(values: v)");
}

#[test]
fn turbofish_is_not_confused_with_comparison_chains() {
    // The whole reason `::<` exists rather than bare `<...>` in expression
    // position: `f < T > (a, b)` (three chained comparisons) is otherwise a
    // perfectly valid parse of the exact same characters. Checked here by
    // inspecting *which* alternative `primary` picked, not just that
    // something parsed — same style as `zero_arg_call_is_not_parsed_as_a_
    // struct_literal`.
    let pair = CleaveParser::parse(Rule::primary, "f::<i32>(x)").unwrap().next().unwrap();
    let inner = pair.into_inner().next().unwrap();
    assert_eq!(inner.as_rule(), Rule::call_expr, "got: {:?}", inner.as_rule());
}

#[test]
fn function_type_annotation_parses() {
    parses(Rule::fn_decl, "fn apply(f: (i32) -> i32, x: i32) -> i32 { f(x) }");
}

#[test]
fn function_type_with_no_params_parses() {
    parses(Rule::type_, "() -> i32");
}

#[test]
fn function_type_with_multiple_params_parses() {
    parses(Rule::type_, "(i32, f64) -> bool");
}

#[test]
fn inherent_impl_on_a_bare_struct_name_parses() {
    parses(Rule::impl_decl, "impl struct Vec2 {\n    fn len(v) { v.x }\n}");
}

#[test]
fn inherent_impl_on_a_generic_struct_parses() {
    parses(Rule::impl_decl, "impl<T> struct Matrix<T> {\n    fn get(m) { m }\n}");
}

#[test]
fn inherent_impl_is_distinguished_from_an_algebra_impl_by_the_struct_keyword() {
    let pair =
        CleaveParser::parse(Rule::impl_decl, "impl struct Vec2 { fn len(v) { v.x } }").unwrap().next().unwrap();
    let inner = pair.into_inner().next().unwrap();
    assert_eq!(inner.as_rule(), Rule::inherent_impl, "got: {:?}", inner.as_rule());
}

#[test]
fn algebra_impl_is_still_recognized_as_such() {
    let pair = CleaveParser::parse(Rule::impl_decl, "impl Ring<Vec2> { fn add(a, b) { a } }").unwrap().next().unwrap();
    let inner = pair.into_inner().next().unwrap();
    assert_eq!(inner.as_rule(), Rule::algebra_impl, "got: {:?}", inner.as_rule());
}

#[test]
fn a_single_generic_struct_target_is_no_longer_ambiguous_with_an_algebra_impl() {
    // The real bug the `struct` keyword fixes, found by testing:
    // `impl<T> Boxed<T> { ... }` (no keyword) parses *identically* either
    // way -- `algebra_impl` treats `Boxed` as an algebra name with `T` as
    // its own single-type target, and fully matches, wrongly, since
    // `algebra_impl` is tried first. The `struct` keyword makes the two
    // shapes unambiguous at the very first token that differs.
    let pair = CleaveParser::parse(Rule::impl_decl, "impl<T> struct Boxed<T> { fn get(b) { b } }")
        .unwrap()
        .next()
        .unwrap();
    let inner = pair.into_inner().next().unwrap();
    assert_eq!(inner.as_rule(), Rule::inherent_impl, "got: {:?}", inner.as_rule());
}

#[test]
fn multi_target_algebra_impl_parses() {
    parses(
        Rule::impl_decl,
        "impl<T: Float, const N: i32, const M: i32, const K: i32> MatMul<Matrix<T,N,M>, Matrix<T,M,K>, Matrix<T,N,K>> {\n    fn mul(a, b) { a }\n}",
    );
}

#[test]
fn single_target_algebra_impl_still_parses_unchanged() {
    parses(Rule::impl_decl, "impl Ring<Vec2> { fn add(a, b) { a } }");
}
