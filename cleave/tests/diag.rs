use cleave::ast::FileId;
use cleave::diag::{Diagnostic, SourceMap};
use cleave::infer::Infer;
use cleave::lower::Lowerer;
use cleave::parser::{CleaveParser, Rule};
use cleave::ast::ItemKind;
use cleave::registry::Registry;
use pest::Parser;

#[test]
fn pest_error_renders_as_clickable_location() {
    let src = "fn f(a, b { a }";
    let err = CleaveParser::parse(Rule::program, src).unwrap_err();
    let mut sources = SourceMap::default();
    sources.add(FileId(0), "bad.cleave", src);
    let diag = Diagnostic::from_pest(&err, FileId(0));
    let rendered = sources.render(&diag);
    assert!(rendered.starts_with("bad.cleave:1:"), "got: {rendered}");
    assert!(rendered.contains("error:"), "got: {rendered}");
}

#[test]
fn type_error_locates_the_offending_line_not_just_line_one() {
    // The mismatch is on line 3 (`a + b`) — confirms `line_col` actually
    // counts newlines rather than only ever reporting line 1. Needs a real
    // `Ring` declared so `add` actually unifies `a`/`b` together (and
    // reveals their mismatch) instead of resolving as an unrelated
    // "unresolved call" — see `infer_call`'s doc comment: there's no
    // permissive built-in fallback left to paper over this.
    let algebra_pair = CleaveParser::parse(Rule::program, "algebra Ring<T> { fn add(a: T, b: T) -> T; }")
        .unwrap()
        .next()
        .unwrap();
    let algebra_program = Lowerer::new(FileId(1)).lower_program(algebra_pair);
    let registry = Registry::build(&algebra_program);

    let src = "fn f(a: f64, b: i32) -> f64 {\n    let _unused = 0;\n    a + b\n}";
    let pair = CleaveParser::parse(Rule::program, src).unwrap().next().unwrap();
    let program = Lowerer::new(FileId(0)).lower_program(pair);
    let f = match &program.items[0].kind {
        ItemKind::Fn(f) => f,
        other => panic!("expected fn, got {other:?}"),
    };
    let err = Infer::new(&registry).infer_fn(f).unwrap_err();

    let mut sources = SourceMap::default();
    sources.add(FileId(0), "bad_types.cleave", src);
    let rendered = sources.render(&Diagnostic::from(&err));
    assert!(rendered.starts_with("bad_types.cleave:3:"), "got: {rendered}");
    assert!(rendered.contains("f64"), "got: {rendered}");
    assert!(rendered.contains("i32"), "got: {rendered}");
}

#[test]
fn missing_file_falls_back_to_unknown_rather_than_panicking() {
    use cleave::ast::Span;
    let diag = Diagnostic::error("oops", Span { file: FileId(99), start: 0, end: 0 });
    let sources = SourceMap::default();
    assert_eq!(sources.render(&diag), "<unknown>: error: oops");
}
