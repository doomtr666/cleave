use cleave::ast::FileId;
use cleave::ast::ItemKind;
use cleave::diag::{Diagnostic, SourceMap};
use cleave::infer::Infer;
use cleave::lower::Lowerer;
use cleave::parser::{CleaveParser, Rule};
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
    let algebra_pair = CleaveParser::parse(
        Rule::program,
        "algebra Ring<T> { fn add(a: T, b: T) -> T; }",
    )
    .unwrap()
    .next()
    .unwrap();
    let algebra_program = Lowerer::new(FileId(1)).lower_program(algebra_pair);
    let registry = Registry::build(&algebra_program);

    let src = "fn f(a: f64, b: i32) -> f64 {\n    let _unused = 0;\n    a + b\n}";
    let pair = CleaveParser::parse(Rule::program, src)
        .unwrap()
        .next()
        .unwrap();
    let program = Lowerer::new(FileId(0)).lower_program(pair);
    let f = match &program.items[0].kind {
        ItemKind::Fn(f) => f,
        other => panic!("expected fn, got {other:?}"),
    };
    let err = Infer::new(&registry).infer_fn(f).unwrap_err();

    let mut sources = SourceMap::default();
    sources.add(FileId(0), "bad_types.cleave", src);
    let rendered = sources.render(&Diagnostic::from(&err));
    assert!(
        rendered.starts_with("bad_types.cleave:3:"),
        "got: {rendered}"
    );
    assert!(rendered.contains("f64"), "got: {rendered}");
    assert!(rendered.contains("i32"), "got: {rendered}");
}

#[test]
fn missing_file_falls_back_to_unknown_rather_than_panicking() {
    use cleave::ast::Span;
    let diag = Diagnostic::error(
        "oops",
        Span {
            file: FileId(99),
            start: 0,
            end: 0,
        },
    );
    let sources = SourceMap::default();
    assert_eq!(sources.render(&diag), "<unknown>: error: oops");
}

/// Compiles `src` with the real CLI, returning what it printed to stderr.
fn cli_stderr(name: &str, src: &str) -> String {
    let dir = std::env::temp_dir().join("cleave-diag-tests");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, src).unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cleave"))
        .args(["--no-openmp", "--no-tasks", "--no-debug-info", "--run"])
        .arg(&path)
        .output()
        .expect("cannot run cleave");
    String::from_utf8_lossy(&output.stderr).to_string() + &String::from_utf8_lossy(&output.stdout)
}

/// An integer literal that doesn't fit its type is a located error: past
/// `i64` (it used to panic the compiler), and past a narrower type (`300`
/// as an `i8` used to wrap silently to `44`). The extremes still fit,
/// `-128` included (`-` applied to `128`).
#[test]
fn an_integer_literal_that_does_not_fit_its_type_is_an_error() {
    let big = cli_stderr("big_literal.cleave", "fn main() -> i32 { let x: i64 = 99999999999999999999; if x > 0 { 1 } else { 0 } }\n");
    assert!(big.contains(":1:33: error: the literal `99999999999999999999` doesn't fit in `i64`"), "{big}");
    let narrow = cli_stderr("narrow_literal.cleave", "fn main() -> i32 { let x: i8 = 300; if x < 100 { 1 } else { 0 } }\n");
    assert!(narrow.contains(":1:32: error: the literal `300` doesn't fit in `i8`"), "{narrow}");
    let extremes = cli_stderr(
        "extreme_literals.cleave",
        "fn main() -> i32 { let x: i8 = -128; let y: i8 = 127; let z: i32 = 2147483647; let w: i64 = 9223372036854775807; if x < y and z > 0 and w > 0 { 1 } else { 0 } }\n",
    );
    assert!(extremes.contains("main returned: 1"), "{extremes}");
}

/// A lambda written in place as a call's argument is bound by name first and
/// the callee specialized for it, like a comprehension's function; a lambda
/// used as any other value (an array element) is a located error, where both
/// used to panic the compiler.
#[test]
fn a_lambda_passed_in_place_runs_and_one_stored_is_an_error() {
    let passed = cli_stderr(
        "lambda_argument.cleave",
        "fn apply(f: (i32) -> i32, x: i32) -> i32 { f(x) }\nfn main() -> i32 { apply(fn(x: i32) -> i32 { x + 1 }, 3) }\n",
    );
    assert!(passed.contains("main returned: 4"), "{passed}");
    let stored = cli_stderr("lambda_stored.cleave", "fn main() -> i32 { let g = fn(x: i32) -> i32 { x + 1 }; let h = [g]; 1 }\n");
    assert!(stored.contains(":1:66: error: `g` used as a value"), "{stored}");
}
