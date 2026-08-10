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

use cleave::cps::{collect_mlir_types, collect_struct_schemas, collect_units, convert_program, dump_cps_program, eliminate_dead_code};
use cleave::diag::SourceMap;
use cleave::driver::compile;
use cleave::dump::dump_program;
use cleave::egraph::optimize_program;
use cleave::mlir_lower::lower_program;
use cleave::monomorphize::dump_monomorphized;
use cleave::print::print_program;
use cleave::registry::Registry;
use melior::Context;
use melior::dialect::DialectRegistry;
use melior::ir::operation::OperationLike;
use melior::pass;
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

    for arg in std::env::args().skip(1) {
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
            other if other.starts_with("--") => return Err(format!("unknown flag {other:?}")),
            other if path.is_none() => path = Some(PathBuf::from(other)),
            other => return Err(format!("only one input file is supported, got a second argument {other:?}")),
        }
    }

    // No `--dump-*`/`--run` flag at all defaults to today's one real pass,
    // so the common case (`cleave file.cleave`) stays exactly as terse as
    // before these flags existed.
    if !dump_ast
        && !dump_inference_pass
        && !dump_monomorphized
        && !dump_cps
        && !dump_cps_optimized
        && !dump_cps_equivalences
        && !dump_mlir
        && !dump_mlir_lowered
        && !run
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
        }),
        None => Err(
            "usage: cleave <file.cleave> [--dump-ast] [--dump-inference-pass] [--dump-monomorphized] [--dump-cps] \
             [--dump-cps-optimized] [--dump-cps-equivalences] [--dump-mlir] [--dump-mlir-lowered] [--run]"
                .to_string(),
        ),
    }
}

fn main() -> ExitCode {
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
    let project_dir = args.path.parent().map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
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
            let units = collect_units(&program, &registry);
            let cps_program = convert_program(units);
            let cps_program = eliminate_dead_code(cps_program);
            print!("{}", dump_cps_program(&cps_program));
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
            let units = collect_units(&program, &registry);
            let cps_program = convert_program(units);
            let cps_program = eliminate_dead_code(cps_program);
            let (optimized, _) = optimize_program(cps_program, &registry);
            // A second sweep: `optimize_program` can itself fold away every
            // remaining call to a stdlib specialization (e.g. `10 + x - 10`
            // reducing to `x` via axioms) — the first sweep, run *before*
            // optimization, has no way to know that in advance, so a unit
            // only unreachable *after* axiom rewriting would otherwise
            // survive despite having zero real callers left. Found by
            // direct testing (`examples/axiom_demo.cleave`): `Ring::add<i32>`/
            // `Ring::sub<i32>` remained in `--dump-cps-optimized`'s own
            // output even though `helper`'s optimized body no longer called
            // either.
            let optimized = eliminate_dead_code(optimized);
            print!("{}", dump_cps_program(&optimized));
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
            let units = collect_units(&program, &registry);
            let cps_program = convert_program(units);
            let cps_program = eliminate_dead_code(cps_program);
            let (_, explanations) = optimize_program(cps_program, &registry);
            if explanations.is_empty() {
                println!("(no axiom rewrites fired)");
            } else {
                for e in &explanations {
                    println!("{e}");
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
            let units = collect_units(&program, &registry);
            let cps_program = convert_program(units);
            let cps_program = eliminate_dead_code(cps_program);
            let (cps_program, _) = optimize_program(cps_program, &registry);
            // See `--dump-cps-optimized`'s own comment above: a second
            // sweep is needed to catch a unit `optimize_program` itself
            // made unreachable (e.g. an axiom folding away every remaining
            // call to it), which the first sweep — run before optimization
            // — has no way to anticipate.
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
            let units = collect_units(&program, &registry);
            let cps_program = convert_program(units);
            let cps_program = eliminate_dead_code(cps_program);
            let (cps_program, _) = optimize_program(cps_program, &registry);
            // See `--dump-cps-optimized`'s own comment above: a second
            // sweep is needed to catch a unit `optimize_program` itself
            // made unreachable (e.g. an axiom folding away every remaining
            // call to it), which the first sweep — run before optimization
            // — has no way to anticipate.
            let cps_program = eliminate_dead_code(cps_program);

            let dialect_registry = DialectRegistry::new();
            register_all_dialects(&dialect_registry);
            let context = Context::new();
            context.append_dialect_registry(&dialect_registry);
            context.load_all_available_dialects();

            let mlir_types = collect_mlir_types(&program);
            let struct_schemas = collect_struct_schemas(&program);
            let mut module = lower_program(&context, &cps_program, &mlir_types, struct_schemas);
            if !module.as_operation().verify() {
                eprintln!("error: generated MLIR module failed verification");
                exit = ExitCode::FAILURE;
            } else {
                // Mirrors `--run`'s own pass pipeline exactly, right up to
                // (not including) JIT invocation -- this *is* the form that
                // actually gets handed to the `ExecutionEngine`, `llvm.*`
                // dialect ops standing in for real textual LLVM IR (melior/
                // mlir-sys, as vendored, don't expose `mlirTranslateModule
                // ToLLVMIR` at all -- real `.ll` text isn't reachable
                // without adding a raw FFI binding ourselves).
                let pass_manager = pass::PassManager::new(&context);
                pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
                pass_manager.add_pass(pass::conversion::create_to_llvm());
                if pass_manager.run(&mut module).is_err() {
                    eprintln!("error: MLIR-to-LLVM lowering pass failed");
                    exit = ExitCode::FAILURE;
                } else {
                    print!("{}", module.as_operation());
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
        let units = collect_units(&program, &registry);
        let cps_program = convert_program(units);
        let cps_program = eliminate_dead_code(cps_program);
        let (cps_program, _) = optimize_program(cps_program, &registry);
        // See `--dump-cps-optimized`'s own comment above: a second sweep is
        // needed to catch a unit `optimize_program` itself made unreachable
        // (e.g. an axiom folding away every remaining call to it), which
        // the first sweep — run before optimization — has no way to
        // anticipate.
        let cps_program = eliminate_dead_code(cps_program);

        let dialect_registry = DialectRegistry::new();
        register_all_dialects(&dialect_registry);
        let context = Context::new();
        context.append_dialect_registry(&dialect_registry);
        context.load_all_available_dialects();

        let mlir_types = collect_mlir_types(&program);
        let struct_schemas = collect_struct_schemas(&program);
        let mut module = lower_program(&context, &cps_program, &mlir_types, struct_schemas);
        if !module.as_operation().verify() {
            eprintln!("error: generated MLIR module failed verification");
            return ExitCode::FAILURE;
        }

        let pass_manager = pass::PassManager::new(&context);
        // `scf.if` (and any other structured-control-flow op) has no direct
        // LLVM IR translation of its own -- `-convert-scf-to-cf` lowers it
        // to the `cf` dialect's ordinary branches first, which `-convert-
        // to-llvm` (below) *does* know how to translate. Skipping this
        // produced a hard native crash (STATUS_ACCESS_VIOLATION), not a
        // clean Rust-level error -- found by direct testing.
        pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
        pass_manager.add_pass(pass::conversion::create_to_llvm());
        if pass_manager.run(&mut module).is_err() {
            eprintln!("error: MLIR-to-LLVM lowering pass failed");
            return ExitCode::FAILURE;
        }

        let engine = melior::ExecutionEngine::new(&module, 2, &[], false, false);
        // A short, explicit, hardcoded list -- grows one line per `extern
        // fn` the Rust-implemented runtime (`cleave-rt`) actually provides.
        // Registering by real function pointer, not dynamic symbol lookup by
        // name, is what lets this sidestep the Windows/MSVC CRT-symbol-
        // visibility questions a raw libc binding would run into.
        // SAFETY: each `cleave_rt::*` pointer is a real, valid `extern "C"
        // fn`, live for the process's whole lifetime.
        unsafe {
            engine.register_symbol("print_i8", cleave_rt::print_i8 as *mut ());
            engine.register_symbol("print_i16", cleave_rt::print_i16 as *mut ());
            engine.register_symbol("print_i32", cleave_rt::print_i32 as *mut ());
            engine.register_symbol("print_i64", cleave_rt::print_i64 as *mut ());
            engine.register_symbol("print_f32", cleave_rt::print_f32 as *mut ());
            engine.register_symbol("print_f64", cleave_rt::print_f64 as *mut ());
            engine.register_symbol("print_bytes", cleave_rt::print_bytes as *mut ());
            engine.register_symbol("cleave_alloc", cleave_rt::cleave_alloc as *mut ());
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

    exit
}

fn report(diags: &[cleave::diag::Diagnostic], sources: &SourceMap) {
    for d in diags {
        eprintln!("{}", sources.render(d));
    }
}

/// Runs whole-program type inference and monomorphization purely to check
/// for errors, discarding `dump_monomorphized`'s own text output -- a
/// mandatory gate before CPS conversion (`collect_units`/`convert_program`),
/// which assumes every reachable unit's own types are already fully
/// concrete and has no error-reporting of its own. Without this, a type
/// error anywhere in the program (not necessarily in the function actually
/// being compiled) can leave an unrelated generic function's call sites
/// never seeded for monomorphization, which used to reach CPS conversion
/// anyway and panic there with a low-level, confusing message instead of
/// this clean diagnostic -- found by direct testing.
fn check_type_errors(program: &cleave::ast::Program, registry: &Registry) -> Result<(), Vec<cleave::diag::Diagnostic>> {
    let (_, errs) = cleave::monomorphize::dump_monomorphized(program, registry);
    let mut diags: Vec<cleave::diag::Diagnostic> = errs.iter().map(cleave::diag::Diagnostic::from).collect();
    diags.extend(check_mutability_errors(program));
    if diags.is_empty() { Ok(()) } else { Err(diags) }
}

/// A purely syntactic pass (`cleave::infer::check_mutability`, no type
/// information needed) run once per `fn` body anywhere in the program — a
/// top-level `fn`, and every method of every algebra/inherent `impl` — since
/// none of those go through `callgraph::infer_program`'s own whole-program
/// walk uniformly enough to hang this off of instead.
fn check_mutability_errors(program: &cleave::ast::Program) -> Vec<cleave::diag::Diagnostic> {
    let mut errors = Vec::new();
    for item in &program.items {
        let fns: Vec<&cleave::ast::FnDecl> = match &item.kind {
            cleave::ast::ItemKind::Fn(f) => vec![f],
            cleave::ast::ItemKind::Impl(d) => d.fns.iter().collect(),
            cleave::ast::ItemKind::InherentImpl(d) => d.fns.iter().collect(),
            _ => vec![],
        };
        for f in fns {
            if let Err(e) = cleave::infer::check_mutability(f) {
                errors.push(cleave::diag::Diagnostic::from(&e));
            }
        }
    }
    errors
}

#[cfg(test)]
mod stdlib_smoke {
    #[test]
    fn found_from_main_binary_layout() {
        assert!(cleave::driver::stdlib_path().is_some());
    }
}
