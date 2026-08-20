//! Shared high-level pipeline entry points, reused by both `main.rs`'s CLI
//! and `cleave-build`'s in-process build-script API (`cleave-build/src/
//! lib.rs`) -- extracted here specifically so the two never drift: "parse
//! this cleave source, type-check it, emit an object file and/or generated
//! Rust FFI bindings for its `export fn`s" needs to mean exactly the same
//! thing whether it's invoked from the command line or from someone else's
//! `build.rs`.

use crate::ast::Program;
use crate::cps::{
    CpsProgram, UnitBody, collect_mlir_types, collect_struct_schemas, collect_units, convert_program, eliminate_dead_code,
};
use crate::diag::{Diagnostic, SourceMap};
use crate::egraph::{DerivativeRequest, optimize_program, synthesize_derivatives};
use crate::mlir_lower::lower_program;
use crate::registry::Registry;
use melior::Context;
use melior::dialect::DialectRegistry;
use melior::ir::operation::OperationLike;
use melior::pass;
use melior::utility::{parse_pass_pipeline, register_all_dialects};
use std::path::{Path, PathBuf};

/// `collect_units` + `convert_program` + `synthesize_derivatives`, bundled
/// -- see the module's own doc comment for why every pipeline entry point
/// needs all three, in this exact order.
pub fn build_cps_program(program: &Program, registry: &Registry) -> Result<CpsProgram, Vec<String>> {
    let units = collect_units(program, registry);
    let requests: Vec<DerivativeRequest> = units
        .iter()
        .filter_map(|u| match &u.body {
            UnitBody::Derivative(of) => Some(DerivativeRequest { name: u.name.clone(), of: of.clone() }),
            _ => None,
        })
        .collect();
    let cps_program = convert_program(units);
    synthesize_derivatives(cps_program, &requests, registry)
}

/// Runs whole-program type inference and monomorphization purely to check
/// for errors -- a mandatory gate before CPS conversion, which assumes
/// every reachable unit's own types are already fully concrete and has no
/// error-reporting of its own.
pub fn check_type_errors(program: &Program, registry: &Registry) -> Result<(), Vec<Diagnostic>> {
    let (_, errs) = crate::monomorphize::dump_monomorphized(program, registry);
    let mut diags: Vec<Diagnostic> = errs.iter().map(Diagnostic::from).collect();
    diags.extend(check_mutability_errors(program));
    if diags.is_empty() { Ok(()) } else { Err(diags) }
}

/// A purely syntactic pass (`crate::infer::check_mutability`, no type
/// information needed), run once per `fn` body anywhere in the program.
fn check_mutability_errors(program: &Program) -> Vec<Diagnostic> {
    let mut errors = Vec::new();
    for item in &program.items {
        let fns: Vec<&crate::ast::FnDecl> = match &item.kind {
            crate::ast::ItemKind::Fn(f) => vec![f],
            crate::ast::ItemKind::Impl(d) => d.fns.iter().collect(),
            crate::ast::ItemKind::InherentImpl(d) => d.fns.iter().collect(),
            _ => vec![],
        };
        for f in fns {
            if let Err(e) = crate::infer::check_mutability(f) {
                errors.push(Diagnostic::from(&e));
            }
        }
    }
    errors
}

fn render_all(diags: &[Diagnostic], sources: &SourceMap) -> Vec<String> {
    diags.iter().map(|d| sources.render(d)).collect()
}

/// Runs an already-compiled, already-type-checked `program` through to an
/// object file and/or a generated Rust FFI binding file for every `export
/// fn` reachable in it -- the shared implementation behind `main.rs`'s
/// `--emit-object`/`--emit-bindings` flags and `compile_and_emit` below.
/// Takes `program`/`registry`/`sources` already built rather than raw
/// source text, so a caller that already has them on hand (`main.rs`, which
/// compiles once up front and reuses the result across every `--dump-*`
/// flag) never re-parses.
pub fn emit_from_program(
    program: &Program,
    registry: &Registry,
    sources: &SourceMap,
    object_path: Option<&Path>,
    bindings_path: Option<&Path>,
) -> Result<(), Vec<String>> {
    check_type_errors(program, registry).map_err(|errs| render_all(&errs, sources))?;

    let cps_program = build_cps_program(program, registry)?;
    let cps_program = eliminate_dead_code(cps_program);
    let (cps_program, _) = optimize_program(cps_program, registry);
    // See `--dump-cps-optimized`'s own comment in `main.rs`: a second sweep
    // is needed to catch a unit `optimize_program` itself made unreachable.
    let cps_program = eliminate_dead_code(cps_program);

    if let Some(bindings_path) = bindings_path {
        let bindings = crate::rust_bindings::generate_rust_bindings(&cps_program.funcs)?;
        std::fs::write(bindings_path, bindings)
            .map_err(|e| vec![format!("failed to write {}: {e}", bindings_path.display())])?;
    }

    if let Some(object_path) = object_path {
        emit_object(program, &cps_program, object_path)?;
    }

    Ok(())
}

/// Parses/merges/resolves `sources_in` from scratch (`driver::compile`'s
/// own shape: one or more `(file_name, text)` pairs) and runs the result
/// through `emit_from_program` -- the simple, one-call API `cleave-build`
/// actually wants: a build script has no pre-existing `Program` lying
/// around the way `main.rs` does.
pub fn compile_and_emit(
    sources_in: Vec<(String, String)>,
    project_dirs: &[PathBuf],
    object_path: Option<&Path>,
    bindings_path: Option<&Path>,
) -> Result<(), Vec<String>> {
    let (result, sources) = crate::driver::compile(sources_in, project_dirs);
    let program = result.map_err(|errs| render_all(&errs, &sources))?;
    let registry = Registry::build(&program);
    emit_from_program(&program, &registry, &sources, object_path, bindings_path)
}

/// The three-stage MLIR-to-`llvm`-dialect lowering pipeline `--run`/`--dump-
/// mlir-lowered` also use, ending in a real `.o` written to `object_path`
/// (`ExecutionEngine::dump_to_object_file`) rather than JIT invocation --
/// see `main.rs`'s own original version of this block for the full
/// reasoning behind each stage; kept identical here, just parameterized.
fn emit_object(program: &Program, cps_program: &CpsProgram, object_path: &Path) -> Result<(), Vec<String>> {
    let dialect_registry = DialectRegistry::new();
    register_all_dialects(&dialect_registry);
    let context = Context::new();
    context.append_dialect_registry(&dialect_registry);
    context.load_all_available_dialects();

    let mlir_types = collect_mlir_types(program);
    let struct_schemas = collect_struct_schemas(program);
    let mut module = lower_program(&context, cps_program, &mlir_types, struct_schemas);
    if !module.as_operation().verify() {
        return Err(vec!["generated MLIR module failed verification".to_string()]);
    }

    let pass_manager = pass::PassManager::new(&context);
    pass_manager.add_pass(pass::linalg::create_convert_elementwise_to_linalg_pass());
    if pass_manager.run(&mut module).is_err() {
        return Err(vec!["MLIR-to-LLVM lowering pass failed (elementwise-to-linalg)".to_string()]);
    }

    let pass_manager = pass::PassManager::new(&context);
    pass::bufferization::register_one_shot_bufferize_pass();
    if parse_pass_pipeline(
        pass_manager.as_operation_pass_manager(),
        "builtin.module(one-shot-bufferize{bufferize-function-boundaries=true})",
    )
    .is_err()
        || pass_manager.run(&mut module).is_err()
    {
        return Err(vec!["MLIR-to-LLVM lowering pass failed (one-shot-bufferize)".to_string()]);
    }

    let pass_manager = pass::PassManager::new(&context);
    pass_manager.add_pass(pass::linalg::create_convert_linalg_to_loops_pass());
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
    if pass_manager.run(&mut module).is_err() {
        return Err(vec!["MLIR-to-LLVM lowering pass failed (to-llvm)".to_string()]);
    }

    // `enable_object_dump = true` (the JIT path always passes `false`) --
    // no symbol registration (JIT-only), no `invoke_packed`: an unresolved
    // call into `cleave-rt` stays an ordinary external symbol reference in
    // the emitted object, resolved later by a real linker against
    // `cleave-rt`'s own staticlib.
    let engine = melior::ExecutionEngine::new(&module, 2, &[], true, false);
    let Some(object_path_str) = object_path.to_str() else {
        return Err(vec![format!("object path {object_path:?} is not valid UTF-8")]);
    };
    engine.dump_to_object_file(object_path_str);
    Ok(())
}

