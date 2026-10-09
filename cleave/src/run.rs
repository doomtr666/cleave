//! Compiling a program and running its `main` in this process, through the
//! JIT: what `cleave --run` does, and what a test running a program uses, so
//! that the two run the same pipeline. Tests each carried their own copy of
//! it (CPS conversion, e-graph, refcounting, lowering, a JIT symbol table),
//! drifting apart and mostly without tasks: a leak of every array built by
//! a function went unseen by the leak tests (`doc/backlog-done.md`).

use crate::ast::Program;
use crate::cps::{collect_mlir_types, collect_struct_schemas};
use crate::diag::{Diagnostic, SourceMap};
use crate::mlir_lower::lower_program;
use crate::pipeline::{CodegenOptions, build_optimized_cps, check_type_errors, lower_to_llvm};
use crate::registry::Registry;
use melior::Context;
use melior::dialect::DialectRegistry;
use melior::ir::operation::OperationLike;
use melior::utility::register_all_dialects;

/// Parses `sources` (`(file name, text)` pairs, `use` resolved against the
/// shipped stdlib) and type-checks them: the program and its registry, or
/// every diagnostic rendered with its location.
pub fn check_sources(
    sources: Vec<(String, String)>,
    defines: &[(String, String)],
    openmp: bool,
) -> Result<(Program, Registry, SourceMap), Vec<String>> {
    let (result, map) = crate::driver::compile(sources, &[]);
    let render = |diags: &[Diagnostic], map: &SourceMap| diags.iter().map(|d| map.render(d)).collect::<Vec<_>>();
    let mut program = result.map_err(|diags| render(&diags, &map))?;
    Registry::apply_defines(&mut program, defines);
    let (registry, define_errors) = Registry::build_with_defines(&program, defines, openmp);
    if !define_errors.is_empty() {
        return Err(define_errors);
    }
    check_type_errors(&program, &registry).map_err(|diags| render(&diags, &map))?;
    Ok((program, registry, map))
}

/// Compiles a checked `program` with `options` and runs its `main` (`fn
/// main() -> i32`), returning what it returned. `extra_symbols` resolve the
/// `extern fn`s the program declares beyond `cleave-rt`'s
/// (`pipeline::register_cleave_rt_symbols`): a test's own host functions,
/// never one of the runtime's (the JIT aborts on a symbol defined twice).
/// `libomp` is loaded when the code needs it (OpenMP, or tasks a program
/// spawns), from `MLIR_SYS_220_PREFIX`.
pub fn run_main(
    program: &Program,
    registry: &Registry,
    sources: Option<&SourceMap>,
    options: &CodegenOptions,
    extra_symbols: &[(&str, *mut ())],
) -> Result<i32, Vec<String>> {
    run_main_with(program, registry, sources, options, extra_symbols, |invoke| invoke())?
}

/// `run_main`, handing `around` the function that runs `main` once compiled:
/// what a caller measuring the run alone (its allocations, its time) wraps,
/// compilation excluded.
pub fn run_main_with<T>(
    program: &Program,
    registry: &Registry,
    sources: Option<&SourceMap>,
    options: &CodegenOptions,
    extra_symbols: &[(&str, *mut ())],
    around: impl FnOnce(&dyn Fn() -> Result<i32, Vec<String>>) -> T,
) -> Result<T, Vec<String>> {
    crate::options::set(options.clone());
    let cps_program = build_optimized_cps(program, registry, sources)?;

    let dialect_registry = DialectRegistry::new();
    register_all_dialects(&dialect_registry);
    let context = Context::new();
    context.append_dialect_registry(&dialect_registry);
    context.load_all_available_dialects();

    let mlir_types = collect_mlir_types(program);
    let struct_schemas = collect_struct_schemas(program);
    let mut module = lower_program(&context, &cps_program, &mlir_types, struct_schemas);
    if !module.as_operation().verify() {
        return Err(vec!["generated MLIR module failed verification".to_string()]);
    }
    lower_to_llvm(&context, &mut module, options)?;

    let mut shared_libs: Vec<String> = Vec::new();
    if options.openmp || (options.tasks && crate::cps::uses_spawn(&cps_program)) {
        let why = if options.openmp {
            "with OpenMP (--no-openmp to run without)"
        } else {
            "a program using `spawn` (its tasks run on libomp, even under --no-openmp; --no-tasks to run them in place)"
        };
        let prefix = std::env::var("MLIR_SYS_220_PREFIX")
            .map_err(|_| vec![format!("MLIR_SYS_220_PREFIX must be set (see .cargo/config.toml) to run {why}")])?;
        shared_libs.push(format!("{prefix}/bin/libomp.dll"));
    }
    let shared_lib_refs: Vec<&str> = shared_libs.iter().map(String::as_str).collect();
    let target = crate::pipeline::target(options)?;
    // SAFETY: `module` is a valid module, owned here.
    let engine = unsafe { cleave_mlir_shim::ExecutionEngine::new(module.to_raw(), &target, &shared_lib_refs) }
        .map_err(|e| vec![format!("failed to compile the program: {e}")])?;
    // SAFETY: every symbol registered is a real `extern "C"` function live for
    // the whole process (`register_cleave_rt_symbols`'s doc comment); the
    // caller vouches for `extra_symbols` the same way.
    unsafe {
        crate::pipeline::register_cleave_rt_symbols(&engine);
        for (name, f) in extra_symbols {
            engine.register_symbol(name, *f);
        }
    }
    let invoke = || {
        let mut result: i32 = -1;
        // SAFETY: `result` is a live, aligned `i32`, what the verified `i32`-
        // returning `main` writes into.
        unsafe { engine.invoke_packed("main", &mut [&mut result as *mut i32 as *mut ()]) }
            .map_err(|e| vec![format!("failed to invoke `main`: {e}")])?;
        Ok(result)
    };
    Ok(around(&invoke))
}

/// `check_sources` then `run_main`, for one source file named `name`.
pub fn run_source(name: &str, src: &str, options: &CodegenOptions) -> Result<i32, Vec<String>> {
    run_source_with(name, src, options, &[])
}

/// `run_source` with host functions for the program's `extern fn`s
/// (`run_main`'s `extra_symbols`).
pub fn run_source_with(
    name: &str,
    src: &str,
    options: &CodegenOptions,
    extra_symbols: &[(&str, *mut ())],
) -> Result<i32, Vec<String>> {
    let (program, registry, map) = check_sources(vec![(name.to_string(), src.to_string())], &[], options.openmp)?;
    run_main(&program, &registry, Some(&map), options, extra_symbols)
}
