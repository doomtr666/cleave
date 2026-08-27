use cleave::driver::compile;
use cleave::pipeline::{compile_and_emit, emit_exe};
use cleave::registry::Registry;
use std::process::Command;

/// Compiles `src` (through the real driver, so `sources`/`registry` match
/// exactly what `main.rs`'s own `--emit-exe` handler builds) and links it to
/// a real standalone `.exe` at `exe_path` via `emit_exe`.
fn build_exe(src: &str, exe_path: &std::path::Path) -> Result<(), Vec<String>> {
    let (result, sources) = compile(vec![("test.cleave".to_string(), src.to_string())], &[]);
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    emit_exe(&program, &registry, &sources, exe_path)
}

/// Regression test for a real, direct-testing-found crash: `emit_object`
/// (`pipeline.rs`) used to call `ExecutionEngine::new`/`dump_to_object_file`
/// *without* registering `cleave-rt`'s own symbols by pointer -- fine for
/// every earlier `--emit-object` test, which happened to only compile
/// `export fn`s with no `extern fn` call in their own body, but a real,
/// direct crash (`STATUS_STACK_BUFFER_OVERRUN`) the moment a compiled
/// program's body actually called a real `extern fn` (`print_i32`, here):
/// the JIT engine apparently still needs every externally-called symbol
/// resolvable at construction time, even for object-only emission with no
/// intent to ever invoke anything. A regression here would very likely
/// crash the whole test process, not just fail an assertion -- an
/// acceptable, visible way for this particular class of bug to surface in
/// `cargo test`, matching how `tests/mlir_lower.rs`'s own end-to-end
/// `extern fn` tests are already structured.
#[test]
fn emitting_an_object_for_a_program_that_calls_a_real_extern_fn_does_not_crash() {
    let dir = std::env::temp_dir().join(format!("cleave_pipeline_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let object_path = dir.join("out.o");

    let src = "extern fn print_i32(x: i32) -> i32;\nfn main() -> i32 { print_i32(42) }".to_string();
    let result = compile_and_emit(
        vec![("test.cleave".to_string(), src)],
        &[],
        Some(&object_path),
        None,
    );

    assert!(
        result.is_ok(),
        "expected a successful object emission, got: {result:?}"
    );
    assert!(
        object_path.exists(),
        "expected an object file to actually be written"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// Same shape as the test just above, but for a genuinely *custom* `extern
/// fn` -- one not in `cleave-rt`'s own fixed set at all, the real-Rust-
/// interop case (`examples/digits-interop`, a consuming crate providing its
/// own externs, linked in by an ordinary linker only *after* this object is
/// emitted). `register_cleave_rt_symbols` alone can't satisfy `melior::
/// ExecutionEngine::new`'s own "every external symbol resolvable at
/// construction time" requirement for a name it's never heard of -- found
/// for real building `digits-interop`'s own data-loading kernel, the exact
/// `STATUS_STACK_BUFFER_OVERRUN` crash `register_cleave_rt_symbols`'s own
/// doc comment already describes for the *known*-symbol case, this time for
/// an unknown one. Fixed by `register_unresolved_extern_stubs` (`pipeline.
/// rs`), registering an inert stub for anything not already known.
#[test]
fn emitting_an_object_for_a_program_with_a_genuinely_custom_extern_fn_does_not_crash() {
    let dir = std::env::temp_dir().join(format!(
        "cleave_pipeline_custom_extern_test_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let object_path = dir.join("out.o");

    let src = "extern fn totally_custom_host_fn(x: i32) -> i32;\n\
               fn main() -> i32 { totally_custom_host_fn(42) }"
        .to_string();
    let result = compile_and_emit(
        vec![("test.cleave".to_string(), src)],
        &[],
        Some(&object_path),
        None,
    );

    assert!(
        result.is_ok(),
        "expected a successful object emission, got: {result:?}"
    );
    assert!(
        object_path.exists(),
        "expected an object file to actually be written"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// The real end-to-end proof for `emit_exe`: an `i32`-returning `main` that
/// calls a real `extern fn` gets compiled all the way to a standalone
/// `.exe`, run as a genuinely separate process (not through cleave's own
/// JIT/`ExecutionEngine` at all), and both its stdout and its own process
/// exit code are checked against the actual expected values -- also the
/// regression test for the symbol collision this module's own doc comment
/// describes (`EXE_ENTRY_SYMBOL`): before that fix, linking failed outright
/// with `duplicate symbol: main`.
#[test]
fn an_i32_returning_main_compiles_links_and_runs_as_a_real_standalone_exe() {
    let dir = std::env::temp_dir().join(format!("cleave_pipeline_exe_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let exe_path = dir.join("prog.exe");

    let src = "extern fn print_i32(x: i32) -> i32;\nfn main() -> i32 { print_i32(1234); 7 }";
    let result = build_exe(src, &exe_path);
    assert!(
        result.is_ok(),
        "expected a successful exe build, got: {result:?}"
    );
    assert!(exe_path.exists(), "expected a real .exe to be written");

    let output = Command::new(&exe_path)
        .output()
        .unwrap_or_else(|e| panic!("failed to run the built exe: {e}"));
    assert_eq!(
        output.status.code(),
        Some(7),
        "expected the process exit code to be main's own return value"
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stdout).trim(),
        "1234",
        "expected print_i32's own stdout output"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// A unit-returning `main` (`fn main() { ... }`, no `->` clause) takes the
/// shim's other branch (`emit_exe`'s own doc comment) -- a plain call, no
/// `std::process::exit`, exit code defaults to the process's ordinary `0`.
#[test]
fn a_unit_returning_main_compiles_links_and_runs_as_a_real_standalone_exe() {
    let dir = std::env::temp_dir().join(format!(
        "cleave_pipeline_exe_unit_test_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let exe_path = dir.join("prog.exe");

    let src = "extern fn print_i32(x: i32) -> i32;\nfn main() { print_i32(9999); }";
    let result = build_exe(src, &exe_path);
    assert!(
        result.is_ok(),
        "expected a successful exe build, got: {result:?}"
    );

    let output = Command::new(&exe_path)
        .output()
        .unwrap_or_else(|e| panic!("failed to run the built exe: {e}"));
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "9999");

    let _ = std::fs::remove_dir_all(&dir);
}

/// `print`/`println` (`stdlib/io/io.cleave`) newline behavior, checked
/// against a real exe's own *exact* stdout bytes -- not `.trim()`'d, unlike
/// the test just above, since a stray/missing newline is exactly the bug
/// this guards: `print` used to hardcode a trailing `\n` for every scalar
/// (`cleave-rt`'s own `print_i32`/... used to call `println!`), silently
/// inconsistent with every string/array/tensor/tuple `Print<T>` impl (routed
/// through `print_bytes`, never added one) -- found directly
/// (`print("step "); print(step);` produced an invisible newline neither
/// call actually wrote). `print` now never appends one; `println` (a plain
/// `T: Print`-bound wrapper) always does.
#[test]
fn print_never_appends_a_newline_and_println_always_does() {
    let dir = std::env::temp_dir().join(format!(
        "cleave_pipeline_exe_println_test_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let exe_path = dir.join("prog.exe");

    let src = "use io;\nfn main() { print(1); print(2); println(3); print(4); }";
    let result = build_exe(src, &exe_path);
    assert!(
        result.is_ok(),
        "expected a successful exe build, got: {result:?}"
    );

    let output = Command::new(&exe_path)
        .output()
        .unwrap_or_else(|e| panic!("failed to run the built exe: {e}"));
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "123\n4",
        "print(1); print(2); println(3) run together with no separator until \
         println(3)'s own trailing newline; the final print(4) has none"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// A program with no `fn main()` at all (e.g. a pure `export fn` library
/// source, never meant to become a standalone exe) is a clean, reported
/// error -- not a panic, not a confusing downstream linker failure.
#[test]
fn emit_exe_without_a_main_fn_is_a_clean_error() {
    let dir = std::env::temp_dir().join(format!(
        "cleave_pipeline_exe_nomain_test_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let exe_path = dir.join("prog.exe");

    let result = build_exe(
        "export fn cleave_add(a: i32, b: i32) -> i32 { a + b }",
        &exe_path,
    );
    assert!(
        result.is_err(),
        "expected an error, no `main` exists to become the exe's own entry point"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
