//! `cleave <file.cleave> [--dump-ast] [--dump-inference-pass] [--dump-monomorphized]`
//! — compiles one file (with real `use` resolution: the file's own directory
//! as the project search path, the shipped stdlib always as a fallback —
//! see `driver::compile`). This is the front-end so far: parse, lower,
//! merge, resolve `use`, infer, monomorphize top-level `fn`s — nothing
//! downstream of that exists yet (no e-graph, no MLIR).
//!
//! Each compiler pass gets its own `--dump-<pass>` flag, printing exactly
//! that stage's output and nothing else — the same "see before and after,
//! don't guess" discipline `print.rs` was built for early on, extended to a
//! real multi-flag CLI instead of hand-editing this file per experiment.
//! Passing none defaults to `--dump-inference-pass` alone (today's most
//! commonly wanted pass); passing more than one prints each requested stage
//! under its own header, in pipeline order, so "before" and "after" a given
//! pass sit next to each other. More `--dump-*` flags arrive as more passes
//! do (CPS conversion, ...).

use cleave::cps::{
    collect_mlir_types, collect_struct_schemas, dump_cps_program, dump_cps_program_readable,
    eliminate_dead_code,
};
use cleave::diag::SourceMap;
use cleave::driver::compile;
use cleave::dump::dump_program;
use cleave::egraph::optimize_program;
use cleave::mlir_lower::lower_program;
use cleave::monomorphize::dump_monomorphized;
use cleave::pipeline::{
    Backend, CodegenOptions, build_cps_program, check_type_errors, lower_to_llvm,
};
use cleave::print::print_program;
use cleave::registry::Registry;
use melior::Context;
use melior::dialect::DialectRegistry;
use melior::ir::operation::OperationLike;
use melior::utility::register_all_dialects;
use std::path::PathBuf;
use std::process::ExitCode;

struct Args {
    path: PathBuf,
    dump_ast: bool,
    dump_inference_pass: bool,
    dump_monomorphized: bool,
    dump_cps: bool,
    dump_cps_optimized: bool,
    dump_cps_readable: bool,
    dump_cps_equivalences: bool,
    dump_mlir: bool,
    dump_mlir_lowered: bool,
    dump_defines: bool,
    run: bool,
    emit_object: Option<PathBuf>,
    emit_bindings: Option<PathBuf>,
    emit_exe: Option<PathBuf>,
    opt_level: u8,
    openmp: Option<bool>,
    target_cpu: Option<String>,
    target_features: Option<String>,
    backend: String,
    inline: Option<bool>,
    llvm_loop_unroll: Option<bool>,
    inline_threshold: Option<usize>,
    tasks: Option<bool>,
    affine_structs: Option<bool>,
    debug_info: Option<bool>,
    /// `--define NAME=VALUE`, repeatable -- `grammar.pest`'s own
    /// `define_decl` doc comment. Collected raw here; parsed/validated
    /// against the program's own `define` declarations by `Registry::
    /// build_with_defines`, the one real consumer.
    defines: Vec<(String, String)>,
}

fn parse_args() -> Result<Args, String> {
    let mut path = None;
    let mut dump_ast = false;
    let mut dump_inference_pass = false;
    let mut dump_monomorphized = false;
    let mut dump_cps = false;
    let mut dump_cps_optimized = false;
    let mut dump_cps_readable = false;
    let mut dump_cps_equivalences = false;
    let mut dump_mlir = false;
    let mut dump_mlir_lowered = false;
    let mut dump_defines = false;
    let mut run = false;
    let mut emit_object = None;
    let mut emit_bindings = None;
    let mut emit_exe = None;
    let mut opt_level: u8 = 2;
    let mut openmp: Option<bool> = None;
    let mut target_cpu = None;
    let mut target_features = None;
    let mut backend = "cpu".to_string();
    let mut inline: Option<bool> = None;
    let mut llvm_loop_unroll: Option<bool> = None;
    let mut inline_threshold: Option<usize> = None;
    let mut tasks: Option<bool> = None;
    let mut affine_structs: Option<bool> = None;
    let mut debug_info: Option<bool> = None;
    let mut defines: Vec<(String, String)> = Vec::new();

    let mut args_iter = std::env::args().skip(1);
    while let Some(arg) = args_iter.next() {
        match arg.as_str() {
            "--dump-ast" => dump_ast = true,
            "--dump-inference-pass" => dump_inference_pass = true,
            "--dump-monomorphized" => dump_monomorphized = true,
            "--dump-cps" => dump_cps = true,
            "--dump-cps-optimized" => dump_cps_optimized = true,
            "--dump-cps-readable" => dump_cps_readable = true,
            "--dump-cps-equivalences" => dump_cps_equivalences = true,
            "--dump-mlir" => dump_mlir = true,
            "--dump-mlir-lowered" => dump_mlir_lowered = true,
            "--dump-defines" => dump_defines = true,
            "--run" => run = true,
            "--emit-object" => {
                let value = args_iter
                    .next()
                    .ok_or_else(|| "--emit-object requires a path argument".to_string())?;
                emit_object = Some(PathBuf::from(value));
            }
            "--emit-bindings" => {
                let value = args_iter
                    .next()
                    .ok_or_else(|| "--emit-bindings requires a path argument".to_string())?;
                emit_bindings = Some(PathBuf::from(value));
            }
            "--emit-exe" => {
                let value = args_iter
                    .next()
                    .ok_or_else(|| "--emit-exe requires a path argument".to_string())?;
                emit_exe = Some(PathBuf::from(value));
            }
            "--opt-level" => {
                let value = args_iter
                    .next()
                    .ok_or_else(|| "--opt-level requires a value (0-3)".to_string())?;
                opt_level = value
                    .parse::<u8>()
                    .ok()
                    .filter(|n| *n <= 3)
                    .ok_or_else(|| format!("--opt-level must be 0-3, got {value:?}"))?;
            }
            "--openmp" => openmp = Some(true),
            "--no-openmp" => openmp = Some(false),
            "--target-cpu" => {
                let value = args_iter
                    .next()
                    .ok_or_else(|| "--target-cpu requires a value".to_string())?;
                target_cpu = Some(value);
            }
            "--target-features" => {
                let value = args_iter
                    .next()
                    .ok_or_else(|| "--target-features requires a value".to_string())?;
                target_features = Some(value);
            }
            "--backend" => {
                let value = args_iter
                    .next()
                    .ok_or_else(|| "--backend requires a value".to_string())?;
                backend = value;
            }
            // Every `--X`/`--no-X` pair below is a plain reassignment of its
            // own mutable local, processed by this same single forward loop
            // over `args_iter` -- the last occurrence of a given flag always
            // wins, matching gcc/clang's own `-f`/`-fno-` "last one wins"
            // convention (`-inline -no-inline` ends with inlining off), with
            // no extra bookkeeping needed for it. See `CodegenOptions`'s own
            // doc comments (`pipeline.rs`) for what each one actually gates.
            "--inline" => inline = Some(true),
            "--no-inline" => inline = Some(false),
            "--llvm-unroll" => llvm_loop_unroll = Some(true),
            "--no-llvm-unroll" => llvm_loop_unroll = Some(false),
            "--inline-threshold" => {
                let value = args_iter
                    .next()
                    .ok_or_else(|| "--inline-threshold requires a value (MLIR operations)".to_string())?;
                inline_threshold = Some(
                    value
                        .parse::<usize>()
                        .map_err(|_| format!("--inline-threshold must be a number of operations, got {value:?}"))?,
                );
            }
            "--tasks" => tasks = Some(true),
            "--no-tasks" => tasks = Some(false),
            "--affine-structs" => affine_structs = Some(true),
            "--no-affine-structs" => affine_structs = Some(false),
            "--debug-info" => debug_info = Some(true),
            "--no-debug-info" => debug_info = Some(false),
            "--define" => {
                let value = args_iter
                    .next()
                    .ok_or_else(|| "--define requires a NAME=VALUE argument".to_string())?;
                let (name, val) = value
                    .split_once('=')
                    .ok_or_else(|| format!("--define {value:?}: expected NAME=VALUE"))?;
                defines.push((name.to_string(), val.to_string()));
            }
            other if other.starts_with("--") => return Err(format!("unknown flag {other:?}")),
            other if path.is_none() => path = Some(PathBuf::from(other)),
            other => {
                return Err(format!(
                    "only one input file is supported, got a second argument {other:?}"
                ));
            }
        }
    }

    // No `--dump-*`/`--run`/`--emit-object` flag at all defaults to today's
    // one real pass, so the common case (`cleave file.cleave`) stays exactly
    // as terse as before these flags existed.
    if !dump_ast
        && !dump_inference_pass
        && !dump_monomorphized
        && !dump_cps
        && !dump_cps_optimized
        && !dump_cps_readable
        && !dump_cps_equivalences
        && !dump_mlir
        && !dump_mlir_lowered
        && !dump_defines
        && !run
        && emit_object.is_none()
        && emit_bindings.is_none()
        && emit_exe.is_none()
    {
        dump_inference_pass = true;
    }

    match path {
        Some(path) => Ok(Args {
            path,
            dump_ast,
            dump_inference_pass,
            dump_monomorphized,
            dump_cps,
            dump_cps_optimized,
            dump_cps_readable,
            dump_cps_equivalences,
            dump_mlir,
            dump_mlir_lowered,
            dump_defines,
            run,
            emit_object,
            emit_bindings,
            emit_exe,
            opt_level,
            openmp,
            target_cpu,
            target_features,
            backend,
            inline,
            llvm_loop_unroll,
            inline_threshold,
            tasks,
            affine_structs,
            debug_info,
            defines,
        }),
        None => Err(
            "usage: cleave <file.cleave> [--dump-ast] [--dump-inference-pass] [--dump-monomorphized] [--dump-cps] \
             [--dump-cps-optimized] [--dump-cps-readable] [--dump-cps-equivalences] [--dump-mlir] [--dump-mlir-lowered] [--dump-defines] [--run] \
             [--emit-object <path>] [--emit-bindings <path>] [--emit-exe <path>] \
             [--opt-level <0-3>] [--openmp | --no-openmp] [--target-cpu <name>] [--target-features <+f,-f,...>] \
             [--backend cpu] [--inline | --no-inline] [--affine-structs | --no-affine-structs] \
             [--debug-info | --no-debug-info] [--define NAME=VALUE]..."
                .to_string(),
        ),
    }
}

/// `Registry::build_with_defines`, plus reporting -- the one real consumer
/// of `args.defines` (`grammar.pest`'s own `define_decl` doc comment).
/// Every call site below used to read `Registry::build(&program)` directly
/// (no `--define` support at all, silently); this is that same call,
/// `--define`-aware, with its own errors (an unknown name, a real `const`
/// targeted, a value that doesn't parse) printed and turned into a real
/// exit code -- `Err`'s own `ExitCode` is already `FAILURE`, callers just
/// need to propagate it their own way (an early `return` for `--run`/
/// `--emit-*`, folded into the accumulating `exit` variable for `--dump-*`,
/// see this file's own module doc comment on why those differ).
fn build_registry(
    program: &cleave::ast::Program,
    defines: &[(String, String)],
    openmp: bool,
) -> Result<Registry, ExitCode> {
    let (registry, errors) = Registry::build_with_defines(program, defines, openmp);
    if errors.is_empty() {
        Ok(registry)
    } else {
        for e in &errors {
            eprintln!("error: {e}");
        }
        Err(ExitCode::FAILURE)
    }
}

/// Resolves `args`'s own codegen flags into a real `CodegenOptions` -- a
/// single, universal resolution, called once and reused (`real_main`'s own
/// `cleave_openmp` doc comment has the full reasoning for why `openmp` no
/// longer varies by call site the way it once did: `--dump-*`/`--run`/
/// `--emit-*` are additive, orthogonal choices, not different defaults for
/// the same underlying option).
fn resolve_codegen_options(args: &Args) -> Result<CodegenOptions, String> {
    let backend = match args.backend.as_str() {
        "cpu" => Backend::Cpu,
        other => return Err(format!("backend {other:?} is not implemented yet -- only \"cpu\" is supported today")),
    };
    let defaults = CodegenOptions::default();
    Ok(CodegenOptions {
        opt_level: args.opt_level,
        openmp: args.openmp.unwrap_or(true),
        target_cpu: args.target_cpu.clone(),
        target_features: args.target_features.clone(),
        backend,
        inline: args.inline.unwrap_or(defaults.inline),
        llvm_loop_unroll: args.llvm_loop_unroll.unwrap_or(defaults.llvm_loop_unroll),
        inline_threshold: args.inline_threshold.unwrap_or(defaults.inline_threshold),
        tasks: args.tasks.unwrap_or(defaults.tasks),
        affine_structs: args.affine_structs.unwrap_or(defaults.affine_structs),
        debug_info: args.debug_info.unwrap_or(defaults.debug_info),
    })
}

// CPS conversion (`cps.rs::convert_program`) recurses once per statement/
// subexpression in a unit's own body, so a Rust-level stack frame is spent
// per AST node converted -- for a large-enough `main` (`tensor_demo.cleave`,
// found by direct testing) this genuinely exceeds the OS's default main-
// thread stack (1MB on Windows, unless raised by the linker) well before
// anything is actually wrong with the program. Running the whole pipeline on
// a worker thread with a generous, fixed stack sidesteps that platform
// default entirely -- the same fix rustc's own driver uses for the identical
// reason, not a workaround for a logic bug.
fn main() -> ExitCode {
    std::thread::Builder::new()
        .stack_size(1024 * 1024 * 1024)
        .spawn(real_main)
        .unwrap()
        .join()
        .unwrap()
}

fn real_main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::FAILURE;
        }
    };

    if args.path.is_dir() {
        // Reading a directory as a file fails with a raw, unhelpful OS error
        // (Windows: "Access is denied", os error 5 — nothing about it says
        // "directory") — worth a clear message instead of passing that
        // straight through, since it's an easy mistake to make (e.g. typing
        // `cargo run cleave` from the workspace root, where `cleave` is also
        // the crate subdirectory's name, passes that literal path through
        // as the argument, not a package selector).
        eprintln!("error: {} is a directory, not a file", args.path.display());
        return ExitCode::FAILURE;
    }

    let text = match std::fs::read_to_string(&args.path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: failed to read {}: {e}", args.path.display());
            return ExitCode::FAILURE;
        }
    };

    // The file's own directory is a project search path — a sibling
    // directory next to it, named after a crate, resolves a `use` the same
    // way a real project root would (see `driver.rs`/`grammar.md`).
    let project_dir = args
        .path
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let file_name = args.path.display().to_string();

    let (result, sources) = compile(vec![(file_name, text)], &[project_dir]);
    let mut program = match result {
        Ok(p) => p,
        Err(errs) => {
            report(&errs, &sources);
            return ExitCode::FAILURE;
        }
    };
    // `--define` overrides become the program's own values, for every pass
    // that reads them (`Registry::apply_defines`).
    Registry::apply_defines(&mut program, &args.defines);

    // `openmp`'s own resolution is now a single, universal rule (`args.
    // openmp.unwrap_or(true)`) -- no more per-mode default (`resolve_
    // codegen_options`'s own doc comment used to vary this by call site;
    // found, in conversation, to have no real technical justification: the
    // `cleave-rt` arena allocator's own `Ordering::Relaxed` safety argument
    // -- "every OpenMP-parallelized region is provably allocator-free" --
    // is a property of the *generated code itself*, never of which engine
    // (JIT vs AOT) ends up running it, so there was never a real reason for
    // the two to differ). Computed once, here, reused by every mode below
    // (registry construction *and* `resolve_codegen_options`) and also fed
    // straight into `Registry::build_with_defines`'s own `CLEAVE_OPENMP`
    // injection (`registry.rs`'s own doc comment on that).
    let cleave_openmp = args.openmp.unwrap_or(true);

    // `args.defines`, validated exactly once, here -- every `Registry::
    // build_with_defines` call site below (one per `--dump-*`/`--run`/
    // `--emit-*` mode, `program`/`args.defines` both unchanged) re-derives
    // the identical, deterministic result, so a bad `--define` only ever
    // gets reported once, not once per requested mode.
    if let Err(code) = build_registry(&program, &args.defines, cleave_openmp) {
        return code;
    }

    // Only header-separate stages when more than one is being dumped at
    // once — no point labeling the single thing being shown in the common,
    // single-flag (or no-flag) case.
    let flags_set = [
        args.dump_ast,
        args.dump_inference_pass,
        args.dump_monomorphized,
        args.dump_cps,
        args.dump_cps_optimized,
        args.dump_cps_readable,
        args.dump_cps_equivalences,
        args.dump_mlir,
        args.dump_mlir_lowered,
        args.dump_defines,
    ]
    .iter()
    .filter(|b| **b)
    .count();
    let multiple = flags_set > 1;
    let mut exit = ExitCode::SUCCESS;

    if args.dump_ast {
        if multiple {
            println!("--- ast (pre-inference) ---\n");
        }
        print!("{}", print_program(&program));
        if multiple {
            println!();
        }
    }

    if args.dump_inference_pass {
        if multiple {
            println!("--- inference pass ---\n");
        }
        let registry = Registry::build_with_defines(&program, &args.defines, cleave_openmp).0;
        let (out, errs) = dump_program(&program, &registry);
        print!("{out}");
        if !errs.is_empty() {
            let diags: Vec<_> = errs.iter().map(cleave::diag::Diagnostic::from).collect();
            report(&diags, &sources);
            exit = ExitCode::FAILURE;
        }
        if multiple {
            println!();
        }
    }

    if args.dump_defines {
        if multiple {
            println!("--- defines ---\n");
        }
        let registry = Registry::build_with_defines(&program, &args.defines, cleave_openmp).0;
        for (name, value) in registry.list_defines() {
            println!("{name} = {value}");
        }
        if multiple {
            println!();
        }
    }

    if args.dump_monomorphized {
        if multiple {
            println!("--- monomorphized ---\n");
        }
        let registry = Registry::build_with_defines(&program, &args.defines, cleave_openmp).0;
        let (out, errs) = dump_monomorphized(&program, &registry);
        print!("{out}");
        if !errs.is_empty() {
            let diags: Vec<_> = errs.iter().map(cleave::diag::Diagnostic::from).collect();
            report(&diags, &sources);
            exit = ExitCode::FAILURE;
        }
    }

    if args.dump_cps {
        if multiple {
            println!("--- cps ---\n");
        }
        let registry = Registry::build_with_defines(&program, &args.defines, cleave_openmp).0;
        if let Err(diags) = check_type_errors(&program, &registry) {
            report(&diags, &sources);
            exit = ExitCode::FAILURE;
        } else {
            match build_cps_program(&program, &registry, None) {
                Ok(cps_program) => {
                    let cps_program = eliminate_dead_code(cps_program);
                    print!("{}", dump_cps_program(&cps_program));
                }
                Err(errs) => {
                    for e in &errs {
                        eprintln!("error: {e}");
                    }
                    exit = ExitCode::FAILURE;
                }
            }
        }
    }

    if args.dump_cps_optimized {
        if multiple {
            println!("--- cps (optimized) ---\n");
        }
        let registry = Registry::build_with_defines(&program, &args.defines, cleave_openmp).0;
        if let Err(diags) = check_type_errors(&program, &registry) {
            report(&diags, &sources);
            exit = ExitCode::FAILURE;
        } else {
            match cleave::pipeline::build_optimized_cps(&program, &registry, None) {
                Ok(optimized) => {
                    print!("{}", dump_cps_program(&optimized));
                }
                Err(errs) => {
                    for e in &errs {
                        eprintln!("error: {e}");
                    }
                    exit = ExitCode::FAILURE;
                }
            }
        }
    }

    if args.dump_cps_readable {
        if multiple {
            println!("--- cps (readable) ---\n");
        }
        // Identical pipeline to `--dump-cps-optimized` above, right down to
        // the two-sweep dead-code elimination and refcounting -- only the
        // final rendering differs (`dump_cps_program_readable`'s own doc
        // comment: a flattened, direct-style form of the exact same
        // optimized-and-refcounted program, not a different snapshot of it).
        let registry = Registry::build_with_defines(&program, &args.defines, cleave_openmp).0;
        if let Err(diags) = check_type_errors(&program, &registry) {
            report(&diags, &sources);
            exit = ExitCode::FAILURE;
        } else {
            match cleave::pipeline::build_optimized_cps(&program, &registry, None) {
                Ok(optimized) => {
                    print!("{}", dump_cps_program_readable(&optimized));
                }
                Err(errs) => {
                    for e in &errs {
                        eprintln!("error: {e}");
                    }
                    exit = ExitCode::FAILURE;
                }
            }
        }
    }

    if args.dump_cps_equivalences {
        if multiple {
            println!("--- cps equivalences ---\n");
        }
        let registry = Registry::build_with_defines(&program, &args.defines, cleave_openmp).0;
        if let Err(diags) = check_type_errors(&program, &registry) {
            report(&diags, &sources);
            exit = ExitCode::FAILURE;
        } else {
            match build_cps_program(&program, &registry, None) {
                Ok(cps_program) => {
                    // Not dead-code-eliminated first -- `--dump-cps-
                    // optimized`'s own identical comment, just above, has
                    // the full reasoning.
                    let (_, explanations) = optimize_program(cps_program, &registry, true);
                    if explanations.is_empty() {
                        println!("(no axiom rewrites fired)");
                    } else {
                        for e in &explanations {
                            println!("{e}");
                        }
                    }
                }
                Err(errs) => {
                    for e in &errs {
                        eprintln!("error: {e}");
                    }
                    exit = ExitCode::FAILURE;
                }
            }
        }
    }

    if args.dump_mlir {
        if multiple {
            println!("--- mlir ---\n");
        }
        let registry = Registry::build_with_defines(&program, &args.defines, cleave_openmp).0;
        if let Err(diags) = check_type_errors(&program, &registry) {
            report(&diags, &sources);
            exit = ExitCode::FAILURE;
        } else {
            match cleave::pipeline::build_optimized_cps(&program, &registry, None) {
                Ok(cps_program) => {
                    let dialect_registry = DialectRegistry::new();
                    register_all_dialects(&dialect_registry);
                    let context = Context::new();
                    context.append_dialect_registry(&dialect_registry);
                    context.load_all_available_dialects();

                    // The program `--run` lowers, releases included: its
                    // options set first (`lower_program` reads
                    // `affine_structs` from them).
                    match resolve_codegen_options(&args) {
                        Ok(options) => cleave::options::set(options),
                        Err(e) => {
                            eprintln!("error: {e}");
                            return ExitCode::FAILURE;
                        }
                    }
                    let mlir_types = collect_mlir_types(&program);
                    let struct_schemas = collect_struct_schemas(&program);
                    let module = lower_program(&context, &cps_program, &mlir_types, struct_schemas);
                    if !module.as_operation().verify() {
                        eprintln!("error: generated MLIR module failed verification");
                        exit = ExitCode::FAILURE;
                    } else {
                        print!("{}", module.as_operation());
                    }
                }
                Err(errs) => {
                    for e in &errs {
                        eprintln!("error: {e}");
                    }
                    exit = ExitCode::FAILURE;
                }
            }
        }
    }

    if args.dump_mlir_lowered {
        if multiple {
            println!("--- mlir (lowered) ---\n");
        }
        let registry = Registry::build_with_defines(&program, &args.defines, cleave_openmp).0;
        if let Err(diags) = check_type_errors(&program, &registry) {
            report(&diags, &sources);
            exit = ExitCode::FAILURE;
        } else {
            match cleave::pipeline::build_optimized_cps(&program, &registry, None) {
                Ok(cps_program) => {
                    let dialect_registry = DialectRegistry::new();
                    register_all_dialects(&dialect_registry);
                    let context = Context::new();
                    context.append_dialect_registry(&dialect_registry);
                    context.load_all_available_dialects();

                    // `lower_to_llvm` -- the shared pipeline `--run` below
                    // also uses, right up to (not including) JIT invocation
                    // -- this *is* the form that actually gets handed to the
                    // `ExecutionEngine`, `llvm.*` dialect ops standing in for
                    // real textual LLVM IR (melior/mlir-sys, as vendored,
                    // don't expose `mlirTranslateModuleToLLVMIR` at all --
                    // real `.ll` text isn't reachable without adding a raw
                    // FFI binding ourselves). OpenMP defaults *on* here too
                    // now, same universal default every mode uses (`real_
                    // main`'s own `cleave_openmp` doc comment) -- pass
                    // `--no-openmp` explicitly for a serial dump.
                    //
                    // Resolved *before* `lower_program` below, not after --
                    // `crate::options::current()` (`lower_program`'s own
                    // `affine_structs` gate reads it) must already reflect
                    // this run's real CLI flags by the time `lower_program`
                    // itself runs, not whatever the thread-local default
                    // happened to be beforehand.
                    let options = match resolve_codegen_options(&args) {
                        Ok(options) => options,
                        Err(e) => {
                            eprintln!("error: {e}");
                            std::process::exit(1);
                        }
                    };
                    cleave::options::set(options.clone());

                    let mlir_types = collect_mlir_types(&program);
                    let struct_schemas = collect_struct_schemas(&program);
                    let mut module =
                        lower_program(&context, &cps_program, &mlir_types, struct_schemas);
                    if !module.as_operation().verify() {
                        eprintln!("error: generated MLIR module failed verification");
                        exit = ExitCode::FAILURE;
                    } else {
                        match lower_to_llvm(&context, &mut module, &options) {
                            Ok(()) => print!("{}", module.as_operation()),
                            Err(errs) => {
                                for e in &errs {
                                    eprintln!("error: {e}");
                                }
                                exit = ExitCode::FAILURE;
                            }
                        }
                    }
                }
                Err(errs) => {
                    for e in &errs {
                        eprintln!("error: {e}");
                    }
                    exit = ExitCode::FAILURE;
                }
            }
        }
    }

    if args.run {
        let registry = Registry::build_with_defines(&program, &args.defines, cleave_openmp).0;
        // CPS conversion (`collect_units`/`convert_program`, shared by all
        // three blocks above and below) assumes every reachable unit's own
        // types are fully concrete -- a program with a real type error
        // elsewhere (e.g. a mismatched argument type in some *other*
        // function) can leave a generic function's call sites never seeded
        // for monomorphization at all, which used to reach CPS conversion
        // anyway and panic there with a confusing low-level message
        // (`resolve_call`'s own `could not resolve call to ...` panic,
        // found by direct testing) instead of this clean diagnostic.
        if let Err(diags) = check_type_errors(&program, &registry) {
            report(&diags, &sources);
            return ExitCode::FAILURE;
        }
        let options = match resolve_codegen_options(&args) {
            Ok(options) => options,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        };
        // The pipeline every test running a program uses too (`cleave::run`).
        match cleave::run::run_main(&program, &registry, None, &options, &[]) {
            Ok(result) => {
                println!("main returned: {result}");
                return ExitCode::from(result as u8);
            }
            Err(errs) => {
                for e in &errs {
                    eprintln!("error: {e}");
                }
                return ExitCode::FAILURE;
            }
        }
    }

    if args.emit_object.is_some() || args.emit_bindings.is_some() {
        let registry = Registry::build_with_defines(&program, &args.defines, cleave_openmp).0;
        let options = match resolve_codegen_options(&args) {
            Ok(options) => options,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        };
        cleave::options::set(options.clone());
        if let Err(errs) = cleave::pipeline::emit_from_program(
            &program,
            &registry,
            &sources,
            args.emit_object.as_deref(),
            args.emit_bindings.as_deref(),
            &options,
        ) {
            for e in &errs {
                eprintln!("error: {e}");
            }
            return ExitCode::FAILURE;
        }
        if let Some(p) = &args.emit_object {
            println!("wrote {}", p.display());
        }
        if let Some(p) = &args.emit_bindings {
            println!("wrote {}", p.display());
        }
    }

    if let Some(exe_path) = &args.emit_exe {
        let registry = Registry::build_with_defines(&program, &args.defines, cleave_openmp).0;
        let options = match resolve_codegen_options(&args) {
            Ok(options) => options,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        };
        cleave::options::set(options.clone());
        if let Err(errs) =
            cleave::pipeline::emit_exe(&program, &registry, &sources, exe_path, &options)
        {
            for e in &errs {
                eprintln!("error: {e}");
            }
            return ExitCode::FAILURE;
        }
        println!("wrote {}", exe_path.display());
    }

    exit
}

fn report(diags: &[cleave::diag::Diagnostic], sources: &SourceMap) {
    for d in diags {
        eprintln!("{}", sources.render(d));
    }
}

// `build_cps_program`/`check_type_errors` now live in `cleave::pipeline`
// (imported above) -- shared with `cleave-build`'s own in-process build-
// script API, which needs the identical pipeline glue outside this binary
// entirely. See that module's own doc comment.

#[cfg(test)]
mod stdlib_smoke {
    #[test]
    fn found_from_main_binary_layout() {
        assert!(cleave::driver::stdlib_path().is_some());
    }
}
