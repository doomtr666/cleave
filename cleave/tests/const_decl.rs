//! Real, JIT-executed tests for a whole-program `const NAME: T = expr;`
//! item -- `grammar.pest::const_decl`, `ast::ConstDecl`, `Registry::
//! eval_global_consts`, `Infer::infer_expr_kind`'s own `ExprKind::Path`
//! fallback (checked *after* the ordinary local `Env` lookup, so a
//! same-named local always wins -- ordinary shadowing, not a special
//! case), and `cps.rs::convert_expr`'s own matching `Path` fallback.
//!
//! A global const is *not* a mutable global (cleave has none) -- it's a
//! name for an already-known-at-compile-time value, evaluated once by
//! `Registry::build`, before inference ever runs.
//!
//! **Type and value travel two deliberately separate paths, not one**
//! (`registry.rs::Registry::global_consts`'s own doc comment has the full
//! reasoning): a reference resolves, for *type-checking*, to the const's
//! own *declared* type (an ordinary `Ty::Con`, via `Infer::ty_from_ast`) --
//! not `Ty::Const`, which would make two *different* global consts of the
//! same declared type fail to unify against each other through a shared
//! generic `T` (`add(A, B)`'s own `Ring::add<T>`) the same way two ordinary
//! `i32` values never would. The actual *value*, separately, is cloned
//! once from `Registry::global_consts()` into every `ConcreteUnit` (`cps.
//! rs`), read back by `convert_expr`'s own `Path` handling directly --
//! keyed by name, not `NodeId`, since a whole-program const's value never
//! varies by which unit references it (unlike a const generic's own
//! per-instantiation value, which genuinely does, and stays on the
//! existing `Ty::Const`/`node_types` path unchanged).
//!
//! Harness duplicated from `cleave/tests/user_guide.rs`'s own `run_i32`
//! (that file's own doc comment explains why: pedagogical-vs-precise, no
//! extraction mechanism between them) rather than shared, matching this
//! project's own established posture for small, focused test files.

use cleave::driver::compile;
use cleave::pipeline::CodegenOptions;
use cleave::registry::Registry;
use cleave_mlir_shim::mlir::Context;
use cleave_mlir_shim::mlir::dialect::DialectRegistry;
use cleave_mlir_shim::mlir::utility::register_all_dialects;

fn context() -> Context {
    let dialect_registry = DialectRegistry::new();
    register_all_dialects(&dialect_registry);
    let context = Context::new();
    context.append_dialect_registry(&dialect_registry);
    context.load_all_available_dialects();
    context
}

fn run_i32(context: &Context, src: &str) -> i32 {
    run_i32_with_defines(context, src, &[])
}

/// Same as `run_i32`, but threading `defines` through `Registry::
/// build_with_defines` instead of the plain, defines-free `Registry::
/// build` -- `main.rs`'s own `build_registry` helper does the identical
/// thing for the real CLI `--define` flag, just with `eprintln!`/`ExitCode`
/// instead of a panic for a config-level error.
fn run_i32_with_defines(context: &Context, src: &str, defines: &[(&str, &str)]) -> i32 {
    run_i32_with_defines_and_openmp(context, src, defines, true)
}

/// `run_i32_with_defines`'s own general form -- also threads a real `openmp`
/// value through to `Registry::build_with_defines`'s own `CLEAVE_OPENMP`
/// injection (`registry.rs`'s own doc comment), for tests that need to
/// observe *which* value `CLEAVE_OPENMP` actually resolves to at runtime,
/// not just that a `define`/`const` folds.
fn run_i32_with_defines_and_openmp(
    _context: &Context,
    src: &str,
    defines: &[(&str, &str)],
    openmp: bool,
) -> i32 {
    let defines: Vec<(String, String)> = defines.iter().map(|(n, v)| (n.to_string(), v.to_string())).collect();
    // `openmp` is what the registry's `CLEAVE_OPENMP` says; the code runs
    // without OpenMP either way.
    let (program, registry, sources) =
        cleave::run::check_sources(vec![("test.cleave".to_string(), src.to_string())], &defines, openmp)
            .unwrap_or_else(|e| panic!("{}", e.join("
")));
    let options = CodegenOptions { openmp: false, tasks: false, ..Default::default() };
    cleave::run::run_main(&program, &registry, Some(&sources), &options, &[]).unwrap_or_else(|e| panic!("{}", e.join("
")))
}

/// The baseline case: a const referenced as an ordinary value, combined
/// with ordinary arithmetic (`Ring::add<i32>`, the operator `+` desugars
/// to) -- not just read back unchanged.
#[test]
fn a_global_const_is_usable_as_an_ordinary_value() {
    let context = context();
    let src = "
        const SEUIL: i32 = 100;
        fn main() -> i32 { SEUIL + 1 }
    ";
    assert_eq!(run_i32(&context, src), 101);
}

/// A local binding of the same name must win -- ordinary lexical shadowing,
/// exactly like a local `let` shadowing an outer one. Proves the fallback
/// in `Infer::infer_expr_kind`'s `Path` arm really is checked *after* `Env`,
/// not instead of it.
#[test]
fn a_local_binding_shadows_a_global_const_of_the_same_name() {
    let context = context();
    let src = "
        const SEUIL: i32 = 100;
        fn main() -> i32 {
            let SEUIL: i32 = 5;
            SEUIL
        }
    ";
    assert_eq!(run_i32(&context, src), 5);
}

/// A function parameter shadows too -- the same rule, a different binder.
#[test]
fn a_parameter_shadows_a_global_const_of_the_same_name() {
    let context = context();
    let src = "
        const SEUIL: i32 = 100;
        fn identity(SEUIL: i32) -> i32 { SEUIL }
        fn main() -> i32 { identity(7) }
    ";
    assert_eq!(run_i32(&context, src), 7);
}

/// The case that used to be a known limitation: two *different* global
/// consts combined directly by the *same* shared generic `T`
/// (`Ring::add<T>`'s own dispatch), not just used independently. Closed by
/// splitting type (`Ty::Con`, unifies freely) from value (`Registry::
/// global_consts()` -> `ConcreteUnit`/`Ctx` -> `cps.rs`, bypassing `node_
/// types`/`Ty::Const` entirely for this case) -- this module's own doc
/// comment has the full mechanism.
#[test]
fn two_different_global_consts_combine_through_a_shared_operator() {
    let context = context();
    let src = "
        const A: i32 = 5;
        const B: i32 = 6;
        fn main() -> i32 { A + B }
    ";
    assert_eq!(run_i32(&context, src), 11);
}

/// A three-way chain (`A + B + C`, i.e. `add(add(A, B), C)`) -- proves the
/// fix isn't narrowly special-cased to exactly two operands: `add(A, B)`'s
/// own *result* (an ordinary `Ty::Con`, same as any other `Ring::add<i32>`
/// call) unifies against `C`'s declared type exactly the same way two
/// ordinary values would, no different from `SEUIL + 1` above.
#[test]
fn three_different_global_consts_chain_through_shared_operators() {
    let context = context();
    let src = "
        const A: i32 = 5;
        const B: i32 = 6;
        const C: i32 = 100;
        fn main() -> i32 { A + B + C }
    ";
    assert_eq!(run_i32(&context, src), 111);
}

/// A const referencing another const in its own initializer (`Registry::
/// eval_global_consts`'s own fixpoint loop) combined, afterward, with a
/// *third*, independent const through a shared operator -- both halves of
/// this feature exercised in one program: evaluation-time cross-references
/// between consts, and inference-time combination of their results.
#[test]
fn a_const_referencing_another_const_combines_correctly_with_a_third() {
    let context = context();
    let src = "
        const A: i32 = 5;
        const B: i32 = A + 1;
        const C: i32 = 100;
        fn main() -> i32 { B + C }
    ";
    // B = 6, so B + C = 106.
    assert_eq!(run_i32(&context, src), 106);
}

/// Declaration order doesn't matter -- `B` (declared *before* `A` in
/// source) still resolves correctly, proving `eval_global_consts`'s own
/// fixpoint loop, not a single top-to-bottom pass.
#[test]
fn a_const_can_reference_another_const_declared_later_in_the_file() {
    let context = context();
    let src = "
        const B: i32 = A + 1;
        const A: i32 = 5;
        fn main() -> i32 { B }
    ";
    assert_eq!(run_i32(&context, src), 6);
}

/// A genuinely nonsensical initializer (references a name that is neither
/// a local nor any real const) still reaches a real, located diagnostic --
/// not a panic, not a silent wrong answer -- `check_type_errors` (this
/// file's own `run_i32` calls it, unlike `user_guide.rs`'s identically-
/// named helper) is what actually catches it, the same path any other
/// unknown-name error already goes through.
#[test]
#[should_panic(expected = "error: `const BOGUS`'s own initializer is not a compile-time constant expression")]
fn an_unresolvable_const_initializer_is_a_located_error_not_a_silent_failure() {
    let context = context();
    let src = "
        const BOGUS: i32 = TOTALLY_UNDECLARED_NAME;
        fn main() -> i32 { BOGUS }
    ";
    run_i32(&context, src);
}

/// A global const as an **explicit turbofish argument** (`probe::<SEUIL>()`)
/// -- a *third*, independent gap from the one above: even after `Infer`'s
/// own resolution succeeds, `monomorphize.rs` re-derives each concrete
/// instantiation through its *own*, entirely separate, `Infer`-free
/// functions (`concrete_ty_from_ast`/`concrete_const_from_expr` --
/// `derive_instantiation`'s own doc comment explains why no `Infer`
/// instance is available there), which needed the identical registry
/// fallback threaded through independently.
#[test]
fn a_global_const_is_usable_as_an_explicit_turbofish_argument() {
    let context = context();
    let src = "
        const SEUIL: i32 = 4;
        fn probe<const N: i32>() -> i32 { N * 10 }
        fn main() -> i32 { probe::<SEUIL>() }
    ";
    assert_eq!(run_i32(&context, src), 40);
}

/// No separate "const expression" grammar category exists -- any
/// arithmetic combination of named consts and literals is legal wherever a
/// concrete const-generic value is needed, as long as it actually folds
/// (`grammar.pest`'s own `generic_arg` doc comment: `expr` is a real
/// fallback there now, tried only after `type_` already failed to parse a
/// bare name/real type). Three named consts, nested operators, both `+`
/// and `*`/`-`, as a turbofish argument -- not just a single name or a
/// two-operand combination.
#[test]
fn arbitrary_arithmetic_on_named_consts_folds_in_generic_argument_position() {
    let context = context();
    let src = "
        const A: i32 = 3;
        const B: i32 = 5;
        const C: i32 = 2;
        fn probe<const N: i32>() -> i32 { N }
        fn main() -> i32 { probe::<(A + B) * C - 1>() }
    ";
    // (3 + 5) * 2 - 1 = 15.
    assert_eq!(run_i32(&context, src), 15);
}

/// `define NAME: T = expr;`, no `--define` supplied at all -- behaves
/// exactly like a `const` with that same value, using its own declared
/// default (`registry.rs::eval_global_consts`'s own doc comment: an
/// external override is only ever *checked first*, never required).
#[test]
fn a_define_with_a_default_resolves_to_its_own_default_when_not_overridden() {
    let context = context();
    let src = "
        define SEUIL: i32 = 100;
        fn main() -> i32 { SEUIL + 1 }
    ";
    assert_eq!(run_i32_with_defines(&context, src, &[]), 101);
}

/// The whole point of `define` over `const`: an external `--define
/// SEUIL=7` wins over the file's own default of `100`, without ever
/// re-evaluating it (`eval_global_consts`'s fixpoint loop checks `defines`
/// *before* attempting `default` at all).
#[test]
fn a_define_is_overridden_by_a_matching_cli_define() {
    let context = context();
    let src = "
        define SEUIL: i32 = 100;
        fn main() -> i32 { SEUIL + 1 }
    ";
    assert_eq!(run_i32_with_defines(&context, src, &[("SEUIL", "7")]), 8);
}

/// `define NAME: T;`, no default at all (mirrors `extern fn`'s bodyless
/// pattern -- `grammar.pest::define_decl`'s own doc comment) -- resolves
/// purely from an external `--define`, with nothing in source to fold.
#[test]
fn a_define_with_no_default_resolves_from_a_cli_define() {
    let context = context();
    let src = "
        define SEUIL: i32;
        fn main() -> i32 { SEUIL * 2 }
    ";
    assert_eq!(run_i32_with_defines(&context, src, &[("SEUIL", "21")]), 42);
}

/// A `bool`-typed `define`, overridden -- `parse_define_value`'s own
/// `"true"`/`"false"` parsing, not just the `u64` fallback every other test
/// here exercises.
#[test]
fn a_bool_define_is_overridden_by_a_cli_define() {
    let context = context();
    let src = "
        define FLAG: bool = false;
        fn main() -> i32 { if FLAG { 1 } else { 0 } }
    ";
    assert_eq!(run_i32_with_defines(&context, src, &[("FLAG", "true")]), 1);
}

/// A `define` with no default and no matching `--define` reaches a real,
/// located diagnostic (`pipeline.rs::check_const_decl_errors`'s own
/// `ItemKind::Define` arm, the `None` branch) -- not a silent wrong value,
/// not a panic deep inside codegen once `cps.rs::convert_expr` fails to
/// find it in `ConcreteUnit::global_consts`.
#[test]
#[should_panic(expected = "error: `define SEUIL` has no default")]
fn a_define_with_no_default_and_no_override_is_a_located_error() {
    let context = context();
    let src = "
        define SEUIL: i32;
        fn main() -> i32 { SEUIL }
    ";
    run_i32_with_defines(&context, src, &[]);
}

/// `--define` targeting a name that isn't declared at all, anywhere in the
/// program -- a configuration-level error (`Registry::build_with_defines`'s
/// own `Vec<String>` return, not a `Diagnostic` -- there's no real `Span`
/// for CLI input to point at), checked directly rather than through
/// `run_i32_with_defines`'s panic wrapper.
#[test]
fn overriding_an_unknown_name_is_a_reported_error() {
    let (result, _sources) = compile(
        vec![(
            "test.cleave".to_string(),
            "define SEUIL: i32 = 100;\nfn main() -> i32 { SEUIL }".to_string(),
        )],
        &[],
    );
    let program = result.unwrap();
    let (_, errors) =
        Registry::build_with_defines(&program, &[("NOPE".to_string(), "1".to_string())], true);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].contains("no such const/define"), "{errors:?}");
}

/// `--define` targeting a real `const` -- rejected outright, "const" keeps
/// meaning what it says (`eval_global_consts`'s own doc comment: this is
/// exactly the design the user insisted on -- a `const` is never
/// overridable, a `define` is a genuinely distinct concept, not a const
/// with an escape hatch).
#[test]
fn overriding_a_real_const_is_a_reported_error() {
    let (result, _sources) = compile(
        vec![(
            "test.cleave".to_string(),
            "const SEUIL: i32 = 100;\nfn main() -> i32 { SEUIL }".to_string(),
        )],
        &[],
    );
    let program = result.unwrap();
    let (_, errors) =
        Registry::build_with_defines(&program, &[("SEUIL".to_string(), "7".to_string())], true);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(
        errors[0].contains("never overridable") && errors[0].contains("declare it `define`"),
        "{errors:?}"
    );
}

/// A `--define` value that doesn't parse against its own `define`'s
/// declared type (`"seven"` against `i32`) -- `parse_define_value`
/// returning `None` reaches a reported error, not a silent fallback to an
/// unresolved/zero value.
#[test]
fn a_badly_typed_cli_define_value_is_a_reported_error() {
    let (result, _sources) = compile(
        vec![(
            "test.cleave".to_string(),
            "define SEUIL: i32 = 100;\nfn main() -> i32 { SEUIL }".to_string(),
        )],
        &[],
    );
    let program = result.unwrap();
    let (_, errors) = Registry::build_with_defines(
        &program,
        &[("SEUIL".to_string(), "seven".to_string())],
        true,
    );
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].contains("not a valid"), "{errors:?}");
}

/// Unary negation of a named const, in an ordinary `const`'s own
/// initializer -- `const_eval.rs::eval_unop`'s own doc comment (`-a`/`not
/// a`, `lower.rs::lower_unary`'s desugaring to a one-argument `Call`) --
/// found missing live: `eval_const_expr`/`const_value_from_expr`/
/// `concrete_const_from_expr` all only ever matched a *two*-argument
/// `Call` before this, so `const NEG: i32 = -A;` used to reach the same
/// "not a compile-time constant expression" diagnostic as a genuinely
/// unresolvable initializer, rather than folding.
#[test]
fn unary_negation_of_a_named_const_folds_in_a_const_initializer() {
    let context = context();
    let src = "
        const A: i32 = 3;
        const NEG: i32 = -A;
        fn main() -> i32 { NEG }
    ";
    assert_eq!(run_i32(&context, src), -3);
}

/// The same unary negation, composed with binary operators, in
/// **generic-argument position** -- both evaluators (`Infer::const_value_
/// from_expr`, `monomorphize.rs::concrete_const_from_expr`) needed the
/// identical one-argument `Call` arm independently.
#[test]
fn unary_negation_composes_with_binary_operators_in_generic_argument_position() {
    let context = context();
    let src = "
        const A: i32 = 3;
        const B: i32 = 10;
        fn probe<const N: i32>() -> i32 { N }
        fn main() -> i32 { probe::<-(A * 2) + B>() }
    ";
    // -(3 * 2) + 10 = 4.
    assert_eq!(run_i32(&context, src), 4);
}

/// `CLEAVE_OPENMP` -- the first compiler-injected constant (`Registry::
/// inject_compiler_define`'s own doc comment, `doc/plan-blas-native.md`'s
/// own §2 "Constantes injectées par le compilateur") -- is usable as an
/// ordinary value, and genuinely reflects whichever `openmp` value this
/// exact compilation resolved, not a hardcoded default baked in once.
#[test]
fn cleave_openmp_reflects_the_resolved_openmp_option() {
    let context = context();
    let src = "fn main() -> i32 { if CLEAVE_OPENMP { 1 } else { 0 } }";
    assert_eq!(
        run_i32_with_defines_and_openmp(&context, src, &[], true),
        1
    );
    assert_eq!(
        run_i32_with_defines_and_openmp(&context, src, &[], false),
        0
    );
}

/// `Registry::list_defines` -- `--dump-defines`'s own one real consumer --
/// lists every currently-resolved `define`, source-declared or compiler-
/// injected alike, sorted by name, and excludes an ordinary `const`
/// entirely (a `const` is never a `define`, `is_define`'s own doc comment).
#[test]
fn list_defines_includes_declared_and_injected_defines_but_not_consts() {
    let (result, _sources) = compile(
        vec![(
            "test.cleave".to_string(),
            "const PI_APPROX: i32 = 3;\n\
             define SEUIL: i32 = 100;\n\
             define FLAG: bool = false;\n\
             fn main() -> i32 { 0 }"
                .to_string(),
        )],
        &[],
    );
    let program = result.unwrap();
    let (registry, errors) =
        Registry::build_with_defines(&program, &[("SEUIL".to_string(), "42".to_string())], true);
    assert!(errors.is_empty(), "{errors:?}");

    let defines = registry.list_defines();
    let names: Vec<&str> = defines.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, vec!["CLEAVE_OPENMP", "FLAG", "SEUIL"], "{names:?}");

    let seuil = defines.iter().find(|(n, _)| n == "SEUIL").unwrap().1;
    assert_eq!(seuil, cleave::infer::ConstValue::Int(42));
    let openmp = defines
        .iter()
        .find(|(n, _)| n == "CLEAVE_OPENMP")
        .unwrap()
        .1;
    assert_eq!(openmp, cleave::infer::ConstValue::Bool(true));
}

/// A floating-point `define` (`define MUON_LR: f32 = 0.02;`), folded through
/// an expression of other consts, and overridden with `--define` like an
/// integer one. Used to be rejected: a `define` was an integer or a bool.
#[test]
fn a_float_define_folds_and_can_be_overridden() {
    let context = context();
    let src = "
        define RATE: f32 = 0.25;
        define SCALE: f32 = RATE * 4.0 - 0.5;
        fn main() -> i32 {
            let x: f32 = SCALE * 10.0;
            if x > 4.99 { if x < 5.01 { 1 } else { 2 } } else { 3 }
        }
    ";
    assert_eq!(run_i32(&context, src), 1, "RATE * 4.0 - 0.5 = 0.5, times 10");
    // 0.5 * 4.0 - 0.5 = 1.5, times 10: above 5.01.
    assert_eq!(run_i32_with_defines(&context, src, &[("RATE", "0.5")]), 2);
}

/// A `define` naming a struct field's dimension (`t: [i32; W]`, as `w:
/// Tensor<f32, 1, W>` in nanoLM), overridden with the CLI's `--define`: the
/// field is built and stored at the overridden size. The struct schemas used
/// to be computed from the defaults, so the value was built at one size and
/// stored at another, failing MLIR verification (`Registry::apply_defines`);
/// and `[1; W]` took the *type* of `W` for its length. Through the CLI, the
/// real entry point (this file's own harness has no runtime for structs).
#[test]
fn an_overridden_define_sizes_struct_fields_too() {
    let dir = std::env::temp_dir().join("cleave-const-decl");
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("define_field.cleave");
    std::fs::write(
        &source,
        "
        define W: i32 = 4;
        struct S { t: [i32; W] }
        fn main() -> i32 {
            let s = S(t: [1; W]);
            let mut n = 0;
            for j in 0..W { n = n + s.t[j]; };
            n
        }
        ",
    )
    .unwrap();
    let run = |define: Option<&str>| {
        let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_cleave"));
        cmd.args(["--no-openmp", "--no-debug-info"]);
        if let Some(d) = define {
            cmd.args(["--define", d]);
        }
        let output = cmd.arg("--run").arg(&source).output().expect("cannot run cleave");
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        format!("{stdout}{stderr}")
    };
    let out = run(None);
    assert!(out.contains("main returned: 4"), "{out}");
    let out = run(Some("W=8"));
    assert!(out.contains("main returned: 8"), "{out}");
}
