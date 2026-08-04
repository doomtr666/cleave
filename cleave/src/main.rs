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

use cleave::cps::{collect_units, convert_program, dump_cps_program};
use cleave::diag::SourceMap;
use cleave::driver::compile;
use cleave::dump::dump_program;
use cleave::monomorphize::dump_monomorphized;
use cleave::print::print_program;
use cleave::registry::Registry;
use std::path::PathBuf;
use std::process::ExitCode;

struct Args {
    path: PathBuf,
    dump_ast: bool,
    dump_inference_pass: bool,
    dump_monomorphized: bool,
    dump_cps: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut path = None;
    let mut dump_ast = false;
    let mut dump_inference_pass = false;
    let mut dump_monomorphized = false;
    let mut dump_cps = false;

    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--dump-ast" => dump_ast = true,
            "--dump-inference-pass" => dump_inference_pass = true,
            "--dump-monomorphized" => dump_monomorphized = true,
            "--dump-cps" => dump_cps = true,
            other if other.starts_with("--") => return Err(format!("unknown flag {other:?}")),
            other if path.is_none() => path = Some(PathBuf::from(other)),
            other => return Err(format!("only one input file is supported, got a second argument {other:?}")),
        }
    }

    // No `--dump-*` flag at all defaults to today's one real pass, so the
    // common case (`cleave file.cleave`) stays exactly as terse as before
    // these flags existed.
    if !dump_ast && !dump_inference_pass && !dump_monomorphized && !dump_cps {
        dump_inference_pass = true;
    }

    match path {
        Some(path) => Ok(Args { path, dump_ast, dump_inference_pass, dump_monomorphized, dump_cps }),
        None => Err(
            "usage: cleave <file.cleave> [--dump-ast] [--dump-inference-pass] [--dump-monomorphized] [--dump-cps]".to_string(),
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
    let flags_set =
        [args.dump_ast, args.dump_inference_pass, args.dump_monomorphized, args.dump_cps].iter().filter(|b| **b).count();
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
        let units = collect_units(&program, &registry);
        let cps_program = convert_program(units);
        print!("{}", dump_cps_program(&cps_program));
    }

    exit
}

fn report(diags: &[cleave::diag::Diagnostic], sources: &SourceMap) {
    for d in diags {
        eprintln!("{}", sources.render(d));
    }
}

#[cfg(test)]
mod stdlib_smoke {
    #[test]
    fn found_from_main_binary_layout() {
        assert!(cleave::driver::stdlib_path().is_some());
    }
}
