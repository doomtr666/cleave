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
    collect_mlir_types, collect_struct_schemas, dump_cps_program, eliminate_dead_code,
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
    dump_cps_equivalences: bool,
    dump_mlir: bool,
    dump_mlir_lowered: bool,
    run: bool,
    emit_object: Option<PathBuf>,
    emit_bindings: Option<PathBuf>,
    emit_exe: Option<PathBuf>,
    opt_level: u8,
    openmp: Option<bool>,
    target_cpu: Option<String>,
    target_features: Option<String>,
    backend: String,
}

fn parse_args() -> Result<Args, String> {
    let mut path = None;
    let mut dump_ast = false;
    let mut dump_inference_pass = false;
    let mut dump_monomorphized = false;
    let mut dump_cps = false;
    let mut dump_cps_optimized = false;
    let mut dump_cps_equivalences = false;
    let mut dump_mlir = false;
    let mut dump_mlir_lowered = false;
    let mut run = false;
    let mut emit_object = None;
    let mut emit_bindings = None;
    let mut emit_exe = None;
    let mut opt_level: u8 = 2;
    let mut openmp: Option<bool> = None;
    let mut target_cpu = None;
    let mut target_features = None;
    let mut backend = "cpu".to_string();

    let mut args_iter = std::env::args().skip(1);
    while let Some(arg) = args_iter.next() {
        match arg.as_str() {
            "--dump-ast" => dump_ast = true,
            "--dump-inference-pass" => dump_inference_pass = true,
            "--dump-monomorphized" => dump_monomorphized = true,
            "--dump-cps" => dump_cps = true,
            "--dump-cps-optimized" => dump_cps_optimized = true,
            "--dump-cps-equivalences" => dump_cps_equivalences = true,
            "--dump-mlir" => dump_mlir = true,
            "--dump-mlir-lowered" => dump_mlir_lowered = true,
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
        && !dump_cps_equivalences
        && !dump_mlir
        && !dump_mlir_lowered
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
            dump_cps_equivalences,
            dump_mlir,
            dump_mlir_lowered,
            run,
            emit_object,
            emit_bindings,
            emit_exe,
            opt_level,
            openmp,
            target_cpu,
            target_features,
            backend,
        }),
        None => Err(
            "usage: cleave <file.cleave> [--dump-ast] [--dump-inference-pass] [--dump-monomorphized] [--dump-cps] \
             [--dump-cps-optimized] [--dump-cps-equivalences] [--dump-mlir] [--dump-mlir-lowered] [--run] \
             [--emit-object <path>] [--emit-bindings <path>] [--emit-exe <path>] \
             [--opt-level <0-3>] [--openmp | --no-openmp] [--target-cpu <name>] [--target-features <+f,-f,...>] \
             [--backend cpu]"
                .to_string(),
        ),
    }
}

/// Resolves `args`'s own codegen flags into a real `CodegenOptions` --
/// `openmp_default` differs per call site (`true` for `--emit-object`/
/// `--emit-bindings`/`--emit-exe`, `false` for `--run`/`--dump-mlir-
/// lowered`, see `CodegenOptions::openmp`'s own doc comment for why),
/// overridden either way by an explicit `--openmp`/`--no-openmp`.
fn resolve_codegen_options(args: &Args, openmp_default: bool) -> Result<CodegenOptions, String> {
    let backend = match args.backend.as_str() {
        "cpu" => Backend::Cpu,
        other => return Err(format!("backend {other:?} is not implemented yet -- only \"cpu\" is supported today")),
    };
    Ok(CodegenOptions {
        opt_level: args.opt_level,
        openmp: args.openmp.unwrap_or(openmp_default),
        target_cpu: args.target_cpu.clone(),
        target_features: args.target_features.clone(),
        backend,
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
    let program = match result {
        Ok(p) => p,
        Err(errs) => {
            report(&errs, &sources);
            return ExitCode::FAILURE;
        }
    };

    // Only header-separate stages when more than one is being dumped at
    // once — no point labeling the single thing being shown in the common,
    // single-flag (or no-flag) case.
    let flags_set = [
        args.dump_ast,
        args.dump_inference_pass,
        args.dump_monomorphized,
        args.dump_cps,
        args.dump_cps_optimized,
        args.dump_cps_equivalences,
        args.dump_mlir,
        args.dump_mlir_lowered,
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
        let registry = Registry::build(&program);
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

    if args.dump_monomorphized {
        if multiple {
            println!("--- monomorphized ---\n");
        }
        let registry = Registry::build(&program);
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
        let registry = Registry::build(&program);
        if let Err(diags) = check_type_errors(&program, &registry) {
            report(&diags, &sources);
            exit = ExitCode::FAILURE;
        } else {
            match build_cps_program(&program, &registry) {
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
        let registry = Registry::build(&program);
        if let Err(diags) = check_type_errors(&program, &registry) {
            report(&diags, &sources);
            exit = ExitCode::FAILURE;
        } else {
            match build_cps_program(&program, &registry) {
                Ok(cps_program) => {
                    let cps_program = eliminate_dead_code(cps_program);
                    let (optimized, _) = optimize_program(cps_program, &registry, false);
                    // A second sweep: `optimize_program` can itself fold away
                    // every remaining call to a stdlib specialization (e.g.
                    // `10 + x - 10` reducing to `x` via axioms) — the first
                    // sweep, run *before* optimization, has no way to know
                    // that in advance, so a unit only unreachable *after*
                    // axiom rewriting would otherwise survive despite having
                    // zero real callers left. Found by direct testing
                    // (`examples/axiom_demo.cleave`): `Ring::add<i32>`/
                    // `Ring::sub<i32>` remained in `--dump-cps-optimized`'s
                    // own output even though `helper`'s optimized body no
                    // longer called either.
                    let optimized = eliminate_dead_code(optimized);
                    // Last CPS-to-CPS step, strictly after the e-graph pass
                    // -- see `cleave::refcount`'s own module doc comment
                    // and `pipeline.rs::build_optimized_cps`'s own
                    // identical step. Included here too so this flag
                    // actually shows the CPS `--emit-object`/`--run` lower,
                    // not an earlier, pre-refcounting snapshot of it.
                    let struct_schemas = collect_struct_schemas(&program);
                    let mlir_types = collect_mlir_types(&program);
                    let optimized = cleave::refcount::insert_refcounting(
                        optimized,
                        &struct_schemas,
                        &mlir_types,
                    );
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

    if args.dump_cps_equivalences {
        if multiple {
            println!("--- cps equivalences ---\n");
        }
        let registry = Registry::build(&program);
        if let Err(diags) = check_type_errors(&program, &registry) {
            report(&diags, &sources);
            exit = ExitCode::FAILURE;
        } else {
            match build_cps_program(&program, &registry) {
                Ok(cps_program) => {
                    let cps_program = eliminate_dead_code(cps_program);
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
        let registry = Registry::build(&program);
        if let Err(diags) = check_type_errors(&program, &registry) {
            report(&diags, &sources);
            exit = ExitCode::FAILURE;
        } else {
            match build_cps_program(&program, &registry) {
                Ok(cps_program) => {
                    let cps_program = eliminate_dead_code(cps_program);
                    let (cps_program, _) = optimize_program(cps_program, &registry, false);
                    // See `--dump-cps-optimized`'s own comment above: a
                    // second sweep is needed to catch a unit `optimize_
                    // program` itself made unreachable (e.g. an axiom
                    // folding away every remaining call to it), which the
                    // first sweep — run before optimization — has no way to
                    // anticipate.
                    let cps_program = eliminate_dead_code(cps_program);

                    let dialect_registry = DialectRegistry::new();
                    register_all_dialects(&dialect_registry);
                    let context = Context::new();
                    context.append_dialect_registry(&dialect_registry);
                    context.load_all_available_dialects();

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
        let registry = Registry::build(&program);
        if let Err(diags) = check_type_errors(&program, &registry) {
            report(&diags, &sources);
            exit = ExitCode::FAILURE;
        } else {
            match build_cps_program(&program, &registry) {
                Ok(cps_program) => {
                    let cps_program = eliminate_dead_code(cps_program);
                    let (cps_program, _) = optimize_program(cps_program, &registry, false);
                    // See `--dump-cps-optimized`'s own comment above: a
                    // second sweep is needed to catch a unit `optimize_
                    // program` itself made unreachable (e.g. an axiom
                    // folding away every remaining call to it), which the
                    // first sweep — run before optimization — has no way to
                    // anticipate.
                    let cps_program = eliminate_dead_code(cps_program);

                    let dialect_registry = DialectRegistry::new();
                    register_all_dialects(&dialect_registry);
                    let context = Context::new();
                    context.append_dialect_registry(&dialect_registry);
                    context.load_all_available_dialects();

                    let mlir_types = collect_mlir_types(&program);
                    let struct_schemas = collect_struct_schemas(&program);
                    let mut module =
                        lower_program(&context, &cps_program, &mlir_types, struct_schemas);
                    if !module.as_operation().verify() {
                        eprintln!("error: generated MLIR module failed verification");
                        exit = ExitCode::FAILURE;
                    } else {
                        // `lower_to_llvm` -- the shared pipeline `--run`
                        // below also uses, right up to (not including) JIT
                        // invocation -- this *is* the form that actually
                        // gets handed to the `ExecutionEngine`, `llvm.*`
                        // dialect ops standing in for real textual LLVM IR
                        // (melior/mlir-sys, as vendored, don't expose
                        // `mlirTranslateModuleToLLVMIR` at all -- real `.ll`
                        // text isn't reachable without adding a raw FFI
                        // binding ourselves). `--dump-mlir-lowered` defaults
                        // OpenMP *off* (see `resolve_codegen_options`'s own
                        // doc comment) -- pass `--openmp` explicitly to see
                        // the parallelized form.
                        let options = match resolve_codegen_options(&args, false) {
                            Ok(options) => options,
                            Err(e) => {
                                eprintln!("error: {e}");
                                std::process::exit(1);
                            }
                        };
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
        let registry = Registry::build(&program);
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
        let cps_program = match build_cps_program(&program, &registry) {
            Ok(p) => p,
            Err(errs) => {
                for e in &errs {
                    eprintln!("error: {e}");
                }
                return ExitCode::FAILURE;
            }
        };
        let cps_program = eliminate_dead_code(cps_program);
        let (cps_program, _) = optimize_program(cps_program, &registry, false);
        // See `--dump-cps-optimized`'s own comment above: a second sweep is
        // needed to catch a unit `optimize_program` itself made unreachable
        // (e.g. an axiom folding away every remaining call to it), which
        // the first sweep — run before optimization — has no way to
        // anticipate.
        let cps_program = eliminate_dead_code(cps_program);
        // Last CPS-to-CPS step, strictly after the e-graph pass -- see
        // `cleave::refcount`'s own module doc comment for why, and
        // `pipeline.rs::build_optimized_cps`'s own identical step.
        let mlir_types = collect_mlir_types(&program);
        let struct_schemas = collect_struct_schemas(&program);
        let cps_program = cleave::refcount::insert_refcounting(
            cps_program,
            &struct_schemas,
            &mlir_types,
        );

        let dialect_registry = DialectRegistry::new();
        register_all_dialects(&dialect_registry);
        let context = Context::new();
        context.append_dialect_registry(&dialect_registry);
        context.load_all_available_dialects();

        let mut module = lower_program(&context, &cps_program, &mlir_types, struct_schemas);
        if !module.as_operation().verify() {
            eprintln!("error: generated MLIR module failed verification");
            return ExitCode::FAILURE;
        }

        // `lower_to_llvm` -- shared with `--dump-mlir-lowered` above and
        // `emit_object` (`pipeline.rs`). `--run` defaults OpenMP *off*
        // (`resolve_codegen_options`'s own doc comment); pass `--openmp`
        // explicitly to invoke the parallelized form for real.
        let options = match resolve_codegen_options(&args, false) {
            Ok(options) => options,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        };
        if let Err(errs) = lower_to_llvm(&context, &mut module, &options) {
            for e in &errs {
                eprintln!("error: {e}");
            }
            return ExitCode::FAILURE;
        }

        // `shared_library_paths` -- unlike `emit_object` (which only ever
        // *dumps* an object file, never actually invokes anything through
        // its own engine, so a dummy stub pointer is sound for `__kmpc_*`
        // resolvability alone), `--run` really does execute the compiled
        // code -- if `options.openmp` is on, the emitted `omp.parallel`/
        // `__kmpc_fork_call` machinery genuinely runs, so it needs *real*
        // `libomp` symbols, not stubs. `ExecutionEngine::new`'s own third
        // parameter exists for exactly this: point it at the real `libomp.
        // dll` and let the JIT resolve `__kmpc_*` from it directly, no stub
        // registration needed at all.
        let mut shared_libs: Vec<String> = Vec::new();
        if options.openmp {
            match std::env::var("MLIR_SYS_220_PREFIX") {
                Ok(prefix) => shared_libs.push(format!("{prefix}/bin/libomp.dll")),
                Err(_) => {
                    eprintln!(
                        "error: MLIR_SYS_220_PREFIX must be set (see .cargo/config.toml) to run with --openmp"
                    );
                    return ExitCode::FAILURE;
                }
            }
        }
        let shared_lib_refs: Vec<&str> = shared_libs.iter().map(String::as_str).collect();
        let engine = melior::ExecutionEngine::new(
            &module,
            options.opt_level as usize,
            &shared_lib_refs,
            false,
            false,
        );
        // SAFETY: see `cleave::pipeline::register_cleave_rt_symbols`'s own
        // doc comment -- shared with `--emit-object`, which needs the
        // identical registration for a reason specific to it (see there).
        unsafe {
            cleave::pipeline::register_cleave_rt_symbols(&engine);
        }
        let mut result: i32 = -1;
        // SAFETY: `result` is a live, correctly-aligned `i32` on the stack
        // for the duration of this call, matching exactly what `main`'s own
        // (verified, i32-returning) MLIR signature writes into.
        match unsafe { engine.invoke_packed("main", &mut [&mut result as *mut i32 as *mut ()]) } {
            Ok(()) => {
                println!("main returned: {result}");
                return ExitCode::from(result as u8);
            }
            Err(error) => {
                eprintln!("error: failed to invoke `main`: {error}");
                return ExitCode::FAILURE;
            }
        }
    }

    if args.emit_object.is_some() || args.emit_bindings.is_some() {
        let registry = Registry::build(&program);
        let options = match resolve_codegen_options(&args, true) {
            Ok(options) => options,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        };
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
        let registry = Registry::build(&program);
        let options = match resolve_codegen_options(&args, true) {
            Ok(options) => options,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::FAILURE;
            }
        };
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
