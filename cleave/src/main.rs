//! `cleave <file.cleave> [--dump-ast] [--dump-inference-pass]` — compiles one
//! file (with real `use` resolution: the file's own directory as the
//! project search path, the shipped stdlib always as a fallback — see
//! `driver::compile`). This is the front-end so far: parse, lower, merge,
//! resolve `use`, infer — nothing downstream of that exists yet (no
//! monomorphization, no e-graph, no MLIR).
//!
//! Each compiler pass gets its own `--dump-<pass>` flag, printing exactly
//! that stage's output and nothing else — the same "see before and after,
//! don't guess" discipline `print.rs` was built for early on, extended to a
//! real multi-flag CLI instead of hand-editing this file per experiment.
//! Passing none defaults to `--dump-inference-pass` alone (today's only
//! real pass); passing more than one prints each requested stage under its
//! own header, in pipeline order, so "before" and "after" a given pass sit
//! next to each other. More `--dump-*` flags arrive as more passes do (CPS
//! conversion, monomorphization, ...).

use cleave::diag::SourceMap;
use cleave::driver::compile;
use cleave::dump::dump_program;
use cleave::print::print_program;
use cleave::registry::Registry;
use std::path::PathBuf;
use std::process::ExitCode;

struct Args {
    path: PathBuf,
    dump_ast: bool,
    dump_inference_pass: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut path = None;
    let mut dump_ast = false;
    let mut dump_inference_pass = false;

    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--dump-ast" => dump_ast = true,
            "--dump-inference-pass" => dump_inference_pass = true,
            other if other.starts_with("--") => return Err(format!("unknown flag {other:?}")),
            other if path.is_none() => path = Some(PathBuf::from(other)),
            other => return Err(format!("only one input file is supported, got a second argument {other:?}")),
        }
    }

    // No `--dump-*` flag at all defaults to today's one real pass, so the
    // common case (`cleave file.cleave`) stays exactly as terse as before
    // these flags existed.
    if !dump_ast && !dump_inference_pass {
        dump_inference_pass = true;
    }

    match path {
        Some(path) => Ok(Args { path, dump_ast, dump_inference_pass }),
        None => Err("usage: cleave <file.cleave> [--dump-ast] [--dump-inference-pass]".to_string()),
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
    let multiple = args.dump_ast && args.dump_inference_pass;
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
