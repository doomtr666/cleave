use cleave::ast::FileId;
use cleave::lower::Lowerer;
use cleave::parser::{CleaveParser, Rule};
use cleave::print::print_program;
use pest::Parser;

fn print_src(src: &str) -> String {
    let pair = CleaveParser::parse(Rule::program, src)
        .unwrap_or_else(|e| panic!("parse failed for {src:?}: {e}"))
        .next()
        .unwrap();
    let program = Lowerer::new(FileId(0)).lower_program(pair);
    print_program(&program)
}

#[test]
fn prints_lambda_with_desugared_body() {
    let out = print_src("fn f() { fn(a, b) { a + b } }");
    assert!(out.contains("fn(a, b) { add(a, b) }"), "got:\n{out}");
}

#[test]
fn prints_struct_lit_with_named_fields() {
    let out = print_src("fn f() { Vec2(x: 1.0, y: 2.0) }");
    assert!(out.contains("Vec2(x: 1.0, y: 2.0)"), "got:\n{out}");
}

#[test]
fn prints_lambda_with_annotations() {
    let out = print_src("fn f() { fn(a: f64, b: f64) -> f64 { a } }");
    assert!(
        out.contains("fn(a: f64, b: f64) -> f64 { a }"),
        "got:\n{out}"
    );
}

#[test]
fn prints_desugared_operators_not_resugared() {
    // The whole point: show `add(a, b)`, not `a + b` — verifying what
    // lowering actually produced, not hiding it back behind sugar.
    let out = print_src("fn f(a, b) { a + b }");
    assert!(out.contains("add(a, b)"), "got:\n{out}");
    assert!(
        !out.contains(" + "),
        "should not re-sugar back to '+', got:\n{out}"
    );
}

#[test]
fn prints_let_mut_and_reassignment_distinctly() {
    let out = print_src("fn f() { let mut acc = 0; acc = acc + 1; acc }");
    assert!(out.contains("let mut acc = 0;"), "got:\n{out}");
    assert!(out.contains("acc = add(acc, 1);"), "got:\n{out}");
}

#[test]
fn prints_multidim_array_type_as_nested() {
    // Confirms the Fortran-sugar flattening is visible, not hidden.
    let out = print_src("fn f(a: [f64; 3, 4]) -> f64 { 0.0 }");
    assert!(out.contains("[[f64; 4]; 3]"), "got:\n{out}");
}

#[test]
fn prints_a_multi_index_bracket_group_directly() {
    // `a[i,j]` is now one `Index` node carrying both indices directly (see
    // `ast.rs::ExprKind::Index`'s own doc comment) -- printed back out the
    // same shape, not re-nested into `a[i][j]`.
    let out = print_src("fn f(a, i, j) { a[i, j] }");
    assert!(out.contains("a[i, j]"), "got:\n{out}");
}

#[test]
fn prints_full_program_readably() {
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
    let out = print_src(src);
    // Structural checks rather than a brittle full-string match — this is a
    // human-eyeballing tool first, exact formatting isn't load-bearing.
    assert!(out.contains("struct Vec2 {"));
    assert!(out.contains("x: f64,"));
    assert!(out.contains("algebra Ring<T> {"));
    assert!(
        out.contains(
            "axiom assoc_add(a: T, b: T, c: T): eq(add(add(a, b), c), add(a, add(b, c)));"
        )
    );
    assert!(out.contains("impl Ring<Vec2> {"));
    assert!(out.contains("fn main() -> i32 {"));
    assert!(out.contains("let mut acc = 0;"));
    assert!(out.contains("acc = add(acc, a);"));
    // No raw Debug noise (NodeId/Span) should leak into the pretty-printed output.
    assert!(!out.contains("NodeId"));
    assert!(!out.contains("Span"));
}
