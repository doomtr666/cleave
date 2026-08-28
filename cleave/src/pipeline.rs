//! Shared high-level pipeline entry points, reused by both `main.rs`'s CLI
//! and `cleave-build`'s in-process build-script API (`cleave-build/src/
//! lib.rs`) -- extracted here specifically so the two never drift: "parse
//! this cleave source, type-check it, emit an object file and/or generated
//! Rust FFI bindings for its `export fn`s" needs to mean exactly the same
//! thing whether it's invoked from the command line or from someone else's
//! `build.rs`.

use crate::ast::{ItemKind, Program};
use crate::cps::{
    CpsProgram, UnitBody, collect_mlir_types, collect_struct_schemas, collect_units,
    convert_program, eliminate_dead_code,
};
use crate::diag::{Diagnostic, SourceMap};
use crate::egraph::{DerivativeRequest, optimize_program, synthesize_derivatives};
use crate::mlir_lower::lower_program;
use crate::refcount::insert_refcounting;
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
pub fn build_cps_program(
    program: &Program,
    registry: &Registry,
) -> Result<CpsProgram, Vec<String>> {
    let units = collect_units(program, registry);
    let requests: Vec<DerivativeRequest> = units
        .iter()
        .filter_map(|u| match &u.body {
            UnitBody::Derivative(of, is_grad) => Some(DerivativeRequest {
                name: u.name.clone(),
                of: of.clone(),
                is_grad: *is_grad,
            }),
            _ => None,
        })
        .collect();
    let cps_program = convert_program(units);
    let struct_schemas = collect_struct_schemas(program);
    synthesize_derivatives(cps_program, &requests, registry, &struct_schemas)
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
    let cps_program = build_optimized_cps(program, registry)?;

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

/// `build_cps_program` + the standard `eliminate_dead_code` / `optimize_
/// program` / `eliminate_dead_code` sequencing every pipeline entry point
/// needs (see `--dump-cps-optimized`'s own comment in `main.rs` for why the
/// second sweep is needed) -- shared by `emit_from_program` and `emit_exe`.
fn build_optimized_cps(program: &Program, registry: &Registry) -> Result<CpsProgram, Vec<String>> {
    let cps_program = build_cps_program(program, registry)?;
    let cps_program = eliminate_dead_code(cps_program);
    let (cps_program, _) = optimize_program(cps_program, registry, false);
    let cps_program = eliminate_dead_code(cps_program);
    // Last CPS-to-CPS step, strictly after the e-graph pass -- see
    // `refcount`'s own module doc comment for why (it has no notion of
    // `Retain`/`Release`'s own effectful ordering, inserting them earlier
    // risks its own rewriting scrambling them).
    let struct_schemas = collect_struct_schemas(program);
    let mlir_types = collect_mlir_types(program);
    Ok(insert_refcounting(cps_program, &struct_schemas, &mlir_types))
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

/// Shared libraries the JIT's own `ExecutionEngine` needs loaded *alongside*
/// the lowered module, for symbols the lowered `llvm` dialect calls but never
/// defines itself -- `memrefCopy` (`mlir::ExecutionEngine::CRunnerUtils`),
/// the runtime helper `one-shot-bufferize` inserts a real call to once a
/// tensor value needs a defensive copy before a write it can't otherwise
/// prove safe. Every program up to and including this session's own `Dense`/
/// `Network` work happened to need no such copy at all (confirmed directly:
/// `--dump-mlir-lowered` on every prior example has zero `memrefCopy` calls),
/// so this was never missing *for those* -- but the gap was always there:
/// `ExecutionEngine::new`'s own `shared_library_paths` parameter was `&[]`
/// unconditionally, on both call sites, so a program needing it would always
/// have crashed exactly the way this one did (`JIT session error: Symbols
/// not found: [ memrefCopy ]`) the moment a big enough derivative expression
/// finally triggered a real defensive copy. `mlir_c_runner_utils.dll` (not
/// `mlir_runner_utils.dll`, its sibling — the latter is the *print*/timing
/// helper library, a separate concern) really does export it (confirmed
/// directly: `dumpbin /exports` on the actual `.dll`), built alongside this
/// project's own real, non-"compiler-only" MLIR 22 install (`.cargo/config.
/// Registers every real `cleave-rt` function by pointer against `engine` --
/// shared by `--run` (`main.rs`) and `emit_object` below. A short, explicit,
/// hardcoded list, growing one line per `extern fn` `cleave-rt` provides;
/// registering by real function pointer, not dynamic symbol lookup by name,
/// sidesteps the Windows/MSVC CRT-symbol-visibility questions a raw libc
/// binding would run into.
///
/// Needed for *object emission* too, not just real JIT invocation, found by
/// direct testing: `ExecutionEngine::new`/`dump_to_object_file` apparently
/// still needs every externally-called symbol resolvable at construction
/// time even though nothing is ever actually invoked through this engine
/// instance -- omitting registration crashed hard (`STATUS_STACK_BUFFER_
/// OVERRUN`) the moment a compiled program called a real `extern fn`
/// (`print_i32`, say), where every earlier `--emit-object` test happened to
/// only exercise `export fn`s with no `extern fn` calls in their own
/// bodies, so this went unnoticed until a program mixing both was tried.
/// The registered pointer only satisfies the engine's own internal
/// requirement, though -- confirmed directly (`llvm-nm` on the emitted
/// `.o`) that the real external symbol still comes out as an ordinary
/// undefined (`U`) relocation, not a baked-in address: the actual object
/// file is unaffected, still meant to be resolved later by a real linker
/// against `cleave-rt`'s own staticlib.
///
/// SAFETY: each `cleave_rt::*` pointer is a real, valid `extern "C" fn`,
/// live for the process's whole lifetime.
///
/// `memrefCopy` belongs here too, even though it isn't a cleave `extern fn`
/// any cleave source ever calls directly -- `cleave_rt::memrefCopy`'s own
/// doc comment has the full story: it's this project's own reimplementation
/// of an MLIR runtime helper (`mlir::ExecutionEngine::CRunnerUtils.h`),
/// needed because `one-shot-bufferize`'s own lowering calls it directly by
/// name whenever a tensor value needs a real defensive copy, and this
/// engine has no shared library loaded to satisfy that on its own.
pub unsafe fn register_cleave_rt_symbols(engine: &melior::ExecutionEngine) {
    unsafe {
        engine.register_symbol("memrefCopy", cleave_rt::memrefCopy as *mut ());
        engine.register_symbol("rand_seed", cleave_rt::rand_seed as *mut ());
        engine.register_symbol("rand_uniform_f32", cleave_rt::rand_uniform_f32 as *mut ());
        engine.register_symbol("rand_uniform_f64", cleave_rt::rand_uniform_f64 as *mut ());
        engine.register_symbol("rand_normal_f32", cleave_rt::rand_normal_f32 as *mut ());
        engine.register_symbol("rand_normal_f64", cleave_rt::rand_normal_f64 as *mut ());
        engine.register_symbol("print_i8", cleave_rt::print_i8 as *mut ());
        engine.register_symbol("print_i16", cleave_rt::print_i16 as *mut ());
        engine.register_symbol("print_i32", cleave_rt::print_i32 as *mut ());
        engine.register_symbol("print_i64", cleave_rt::print_i64 as *mut ());
        engine.register_symbol("print_f32", cleave_rt::print_f32 as *mut ());
        engine.register_symbol("print_f64", cleave_rt::print_f64 as *mut ());
        engine.register_symbol("print_bytes", cleave_rt::print_bytes as *mut ());
        engine.register_symbol(
            "print_dynarray_bytes",
            cleave_rt::print_dynarray_bytes as *mut (),
        );
        engine.register_symbol("format_f32", cleave_rt::format_f32 as *mut ());
        engine.register_symbol("format_f64", cleave_rt::format_f64 as *mut ());
        engine.register_symbol("cleave_alloc", cleave_rt::cleave_alloc as *mut ());
        engine.register_symbol("cleave_alloc_rc", cleave_rt::cleave_alloc_rc as *mut ());
        engine.register_symbol("cleave_retain", cleave_rt::cleave_retain as *mut ());
        engine.register_symbol("cleave_release", cleave_rt::cleave_release as *mut ());
        engine.register_symbol("dynarray_alloc_i8", cleave_rt::dynarray_alloc_i8 as *mut ());
        engine.register_symbol("dynarray_grow_i8", cleave_rt::dynarray_grow_i8 as *mut ());
        engine.register_symbol("dynarray_get_i8", cleave_rt::dynarray_get_i8 as *mut ());
        engine.register_symbol("dynarray_set_i8", cleave_rt::dynarray_set_i8 as *mut ());
        engine.register_symbol(
            "dynarray_alloc_i16",
            cleave_rt::dynarray_alloc_i16 as *mut (),
        );
        engine.register_symbol("dynarray_grow_i16", cleave_rt::dynarray_grow_i16 as *mut ());
        engine.register_symbol("dynarray_get_i16", cleave_rt::dynarray_get_i16 as *mut ());
        engine.register_symbol("dynarray_set_i16", cleave_rt::dynarray_set_i16 as *mut ());
        engine.register_symbol(
            "dynarray_alloc_i32",
            cleave_rt::dynarray_alloc_i32 as *mut (),
        );
        engine.register_symbol("dynarray_grow_i32", cleave_rt::dynarray_grow_i32 as *mut ());
        engine.register_symbol("dynarray_get_i32", cleave_rt::dynarray_get_i32 as *mut ());
        engine.register_symbol("dynarray_set_i32", cleave_rt::dynarray_set_i32 as *mut ());
        engine.register_symbol(
            "dynarray_alloc_i64",
            cleave_rt::dynarray_alloc_i64 as *mut (),
        );
        engine.register_symbol("dynarray_grow_i64", cleave_rt::dynarray_grow_i64 as *mut ());
        engine.register_symbol("dynarray_get_i64", cleave_rt::dynarray_get_i64 as *mut ());
        engine.register_symbol("dynarray_set_i64", cleave_rt::dynarray_set_i64 as *mut ());
        engine.register_symbol(
            "dynarray_alloc_f32",
            cleave_rt::dynarray_alloc_f32 as *mut (),
        );
        engine.register_symbol("dynarray_grow_f32", cleave_rt::dynarray_grow_f32 as *mut ());
        engine.register_symbol("dynarray_get_f32", cleave_rt::dynarray_get_f32 as *mut ());
        engine.register_symbol("dynarray_set_f32", cleave_rt::dynarray_set_f32 as *mut ());
        engine.register_symbol(
            "dynarray_alloc_f64",
            cleave_rt::dynarray_alloc_f64 as *mut (),
        );
        engine.register_symbol("dynarray_grow_f64", cleave_rt::dynarray_grow_f64 as *mut ());
        engine.register_symbol("dynarray_get_f64", cleave_rt::dynarray_get_f64 as *mut ());
        engine.register_symbol("dynarray_set_f64", cleave_rt::dynarray_set_f64 as *mut ());
        engine.register_symbol(
            "dynarray_alloc_ptr",
            cleave_rt::dynarray_alloc_ptr as *mut (),
        );
        engine.register_symbol("dynarray_grow_ptr", cleave_rt::dynarray_grow_ptr as *mut ());
        engine.register_symbol("dynarray_get_ptr", cleave_rt::dynarray_get_ptr as *mut ());
        engine.register_symbol("dynarray_set_ptr", cleave_rt::dynarray_set_ptr as *mut ());
    }
}

/// The three-stage MLIR-to-`llvm`-dialect lowering pipeline `--run`/`--dump-
/// mlir-lowered` also use, ending in a real `.o` written to `object_path`
/// (`ExecutionEngine::dump_to_object_file`) rather than JIT invocation --
/// see `main.rs`'s own original version of this block for the full
/// reasoning behind each stage; kept identical here, just parameterized.
fn emit_object(
    program: &Program,
    cps_program: &CpsProgram,
    object_path: &Path,
) -> Result<(), Vec<String>> {
    let dialect_registry = DialectRegistry::new();
    register_all_dialects(&dialect_registry);
    let context = Context::new();
    context.append_dialect_registry(&dialect_registry);
    context.load_all_available_dialects();

    let mlir_types = collect_mlir_types(program);
    let struct_schemas = collect_struct_schemas(program);
    let mut module = lower_program(&context, cps_program, &mlir_types, struct_schemas);
    if !module.as_operation().verify() {
        return Err(vec![
            "generated MLIR module failed verification".to_string(),
        ]);
    }

    let pass_manager = pass::PassManager::new(&context);
    pass_manager.add_pass(pass::linalg::create_convert_elementwise_to_linalg_pass());
    if pass_manager.run(&mut module).is_err() {
        return Err(vec![
            "MLIR-to-LLVM lowering pass failed (elementwise-to-linalg)".to_string(),
        ]);
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
        return Err(vec![
            "MLIR-to-LLVM lowering pass failed (one-shot-bufferize)".to_string(),
        ]);
    }

    // Tensor *payload* deallocation — tried here twice before and
    // reverted both times (`doc/backlog.md`, "MLIR's own buffer-
    // deallocation pipeline corrupts memory against cleave's current
    // struct/tensor-field ABI"); real end-to-end training crashed with
    // `STATUS_ACCESS_VIOLATION` (or, on the retest, silently trained to
    // random-guess accuracy — no crash, still wrong). Root-caused
    // precisely this time, by hand, against this exact toolchain: a
    // struct's own `Tensor` field, read via `load_native_shape_field`
    // (`mlir_lower.rs`), used to cast the field's own storage directly
    // into a `memref` and hand it to `bufferization.to_tensor ...
    // restrict` — `restrict` is a promise of *exclusive* ownership, a real
    // lie for a struct field (the struct itself still owns and reuses that
    // same storage) — confirmed directly to cause both silent in-place
    // corruption of the struct's own field (One-Shot Bufferize, trusting
    // the promise, computes straight back into it) and, once this pass
    // runs, premature deallocation of the struct's own storage (this
    // pass, trusting the same promise, frees it the moment the "exclusive"
    // reference's own last use passes). Fixed at the true source
    // (`load_native_shape_field`'s own doc comment has the full story): a
    // defensive `memref.alloc`+`memref.copy` before `to_tensor ... restrict
    // writable`, so the promise is genuinely true and this pass — reused
    // here as-is, no longer worked around — frees the *copy*, never the
    // struct's own storage.
    let pass_manager = pass::PassManager::new(&context);
    pass_manager.add_pass(pass::bufferization::create_ownership_based_buffer_deallocation_pass());
    pass_manager.add_pass(pass::bufferization::create_buffer_deallocation_simplification_pass());
    pass_manager.add_pass(pass::bufferization::create_lower_deallocations_pass());
    pass_manager.add_pass(pass::conversion::create_bufferization_to_mem_ref());
    if pass_manager.run(&mut module).is_err() {
        return Err(vec![
            "MLIR-to-LLVM lowering pass failed (buffer-deallocation)".to_string(),
        ]);
    }

    let pass_manager = pass::PassManager::new(&context);
    pass_manager.add_pass(pass::linalg::create_convert_linalg_to_loops_pass());
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
    if pass_manager.run(&mut module).is_err() {
        return Err(vec![
            "MLIR-to-LLVM lowering pass failed (to-llvm)".to_string(),
        ]);
    }

    let engine = melior::ExecutionEngine::new(&module, 2, &[], true, false);
    // SAFETY: see `register_cleave_rt_symbols`'s own doc comment.
    unsafe {
        register_cleave_rt_symbols(&engine);
        register_unresolved_extern_stubs(&engine, program);
    }
    let Some(object_path_str) = object_path.to_str() else {
        return Err(vec![format!(
            "object path {object_path:?} is not valid UTF-8"
        )]);
    };
    engine.dump_to_object_file(object_path_str);
    Ok(())
}

/// `register_cleave_rt_symbols`'s own doc comment establishes that `Execution
/// Engine::new`/`dump_to_object_file` needs *every* externally-called symbol
/// resolvable at construction time, even for object-only emission where
/// nothing is ever actually invoked through this engine instance — but that
/// registration only ever covers `cleave-rt`'s own fixed, known set. A
/// program declaring its *own* `extern fn` (real Rust interop, `examples/
/// digits-interop/src/kernel.cleave` — the whole point of `export fn`/`--
/// emit-object` existing at all: a consuming Rust crate provides its own
/// externs, compiled into the *same final binary* by an ordinary linker
/// afterward, not by this engine) has no way to satisfy that requirement at
/// this point in the pipeline — the real implementation lives in the
/// consuming crate, which hasn't even been compiled yet when this object is
/// being emitted. Found for real, not hypothetical: the very first program
/// with a genuinely custom `extern fn` (not one of `cleave-rt`'s own) hit
/// exactly the `STATUS_STACK_BUFFER_OVERRUN` crash `register_cleave_rt_
/// symbols`'s own doc comment already describes for the *known*-symbol case,
/// just for an *unknown* one instead.
///
/// The fix mirrors that same doc comment's own confirmed finding — "the
/// registered pointer only satisfies the engine's own internal requirement
/// ... the actual object file is unaffected, still meant to be resolved
/// later by a real linker" — so *what* gets registered here doesn't matter
/// at all, only *that* something does: every `extern fn` in `program` not
/// already in `KNOWN_CLEAVE_RT_SYMBOLS` gets the same inert stub pointer.
/// The emitted object's own undefined relocation for that symbol is
/// unaffected either way (confirmed the identical way that doc comment
/// already did, via `llvm-nm`), so this is sound regardless of the stub's
/// own signature mismatch against the real one.
///
/// SAFETY: `dummy_extern_stub`'s own address is a real, valid, live-for-the-
/// whole-process function pointer — its signature never has to match the
/// real extern's own, since it's provably never called through this engine.
unsafe fn register_unresolved_extern_stubs(engine: &melior::ExecutionEngine, program: &Program) {
    extern "C" fn dummy_extern_stub() {}
    for item in &program.items {
        let ItemKind::Fn(f) = &item.kind else { continue };
        if !f.is_extern {
            continue;
        }
        let symbol = f.extern_symbol.as_deref().unwrap_or(&f.name);
        if KNOWN_CLEAVE_RT_SYMBOLS.contains(&symbol) {
            continue;
        }
        // SAFETY: forwarded from this function's own contract.
        unsafe {
            engine.register_symbol(symbol, dummy_extern_stub as *mut ());
        }
    }
}

/// Every symbol `register_cleave_rt_symbols` registers, by name — kept as an
/// explicit, separate list (not derived from that function's own body)
/// purely because the real registration there is one hardcoded `register_
/// symbol` call per real function pointer, not a loop over data; a new
/// `cleave-rt` extern needs a line added in *both* places (the doc comment
/// on each cross-references the other).
const KNOWN_CLEAVE_RT_SYMBOLS: &[&str] = &[
    "memrefCopy",
    "rand_seed",
    "rand_uniform_f32",
    "rand_uniform_f64",
    "rand_normal_f32",
    "rand_normal_f64",
    "print_i8",
    "print_i16",
    "print_i32",
    "print_i64",
    "print_f32",
    "print_f64",
    "print_bytes",
    "print_dynarray_bytes",
    "format_f32",
    "format_f64",
    "cleave_alloc",
    "cleave_alloc_rc",
    "cleave_retain",
    "cleave_release",
    "dynarray_alloc_i8",
    "dynarray_grow_i8",
    "dynarray_get_i8",
    "dynarray_set_i8",
    "dynarray_alloc_i16",
    "dynarray_grow_i16",
    "dynarray_get_i16",
    "dynarray_set_i16",
    "dynarray_alloc_i32",
    "dynarray_grow_i32",
    "dynarray_get_i32",
    "dynarray_set_i32",
    "dynarray_alloc_i64",
    "dynarray_grow_i64",
    "dynarray_get_i64",
    "dynarray_set_i64",
    "dynarray_alloc_f32",
    "dynarray_grow_f32",
    "dynarray_get_f32",
    "dynarray_set_f32",
    "dynarray_alloc_f64",
    "dynarray_grow_f64",
    "dynarray_get_f64",
    "dynarray_set_f64",
    "dynarray_alloc_ptr",
    "dynarray_grow_ptr",
    "dynarray_get_ptr",
    "dynarray_set_ptr",
];

/// The fixed internal symbol cleave's own `fn main()` gets renamed to when
/// compiling a standalone executable (`emit_exe` below) -- never seen by a
/// cleave program's own author, purely an implementation detail of the
/// generated Rust shim's own linking. See `emit_exe`'s own doc comment for
/// why a rename is needed at all.
const EXE_ENTRY_SYMBOL: &str = "__cleave_program_main";

/// Compiles `program` all the way to a real, standalone `.exe` at
/// `exe_path` -- `emit_object` plus a real link step (`emit_object`/`--
/// emit-object` alone only ever produces a `.o`, still needing an external
/// linker to become anything runnable).
///
/// The real work here, beyond `emit_object`: cleave's own compiled `main`
/// gets the literal LLVM/object-file symbol name `main` (`mlir_lower.rs`'s
/// own `lower_top_level_fn`) -- which collides with a *real* Rust binary's
/// own `fn main()` (confirmed directly: linking two objects that both
/// define `main` fails with `duplicate symbol: main`, and an ordinary
/// `rustc`-compiled `fn main()` genuinely does emit a real, unmangled
/// `main` symbol of its own, needed by `std`'s own runtime-startup code to
/// call back into it). So this reuses the *already-existing* `export fn`
/// symbol-override mechanism (`ast.rs`'s own `FnDecl::is_export`/
/// `export_symbol`, `mlir_lower.rs`'s own resulting symbol-name logic) to
/// rename just `main`'s own emitted symbol to `EXE_ENTRY_SYMBOL` -- no new
/// MLIR-lowering code needed at all, just flipping the same two fields a
/// real `export fn` would set, directly on the `CTopLevelFn` found by name
/// after CPS conversion.
///
/// The actual link step shells out to `rustc` (first `std::process::
/// Command` use in this codebase) against a tiny, generated Rust "shim"
/// source (`fn main() { std::process::exit(unsafe { EXE_ENTRY_SYMBOL() })
/// }` for an `i32`-returning cleave `main`, just a bare call for a unit-
/// returning one) -- not a raw system linker (`clang`/`lld-link` directly),
/// found necessary by direct testing: `cleave-rt` links Rust's own `std`,
/// which needs a real, sizeable list of Windows system libraries
/// (`ws2_32`, `ntdll`, `userenv`, `bcrypt`, ...) that only `rustc`'s own
/// linker invocation knows how to supply correctly and keeps up to date --
/// a bare `clang -o exe kernel.o cleave_rt.lib` left ~30 unresolved
/// external symbols. `rustc` still only *drives* the link, though: nothing
/// here hardcodes MSVC's own `link.exe` as the actual linker backend --
/// `rustc`'s own default on this platform is used as-is (this project's
/// own MLIR dependency doesn't remove the need for a working Rust
/// toolchain to build `cleave` itself in the first place, so requiring one
/// again here adds no new prerequisite).
pub fn emit_exe(
    program: &Program,
    registry: &Registry,
    sources: &SourceMap,
    exe_path: &Path,
) -> Result<(), Vec<String>> {
    check_type_errors(program, registry).map_err(|errs| render_all(&errs, sources))?;
    let mut cps_program = build_optimized_cps(program, registry)?;

    let Some(main_fn) = cps_program.funcs.iter_mut().find(|f| f.def.name == "main") else {
        return Err(vec![
            "no `fn main()` found -- a standalone executable needs a real entry point".to_string(),
        ]);
    };
    main_fn.is_export = true;
    main_fn.export_symbol = Some(EXE_ENTRY_SYMBOL.to_string());
    let main_returns_i32 = !matches!(&main_fn.result, crate::infer::Ty::Con(name) if name == "()");

    // A process id alone isn't a unique enough work-dir name -- found by
    // direct testing (`cargo test`'s own default parallel test execution
    // runs multiple `emit_exe` calls concurrently, *within the same test
    // process*, so several calls sharing one PID clobbered each other's
    // `program.o`/`shim.rs` mid-flight, non-deterministically linking
    // whichever call's files happened to still be on disk). A monotonic
    // counter, unique per call within this process, added alongside the PID
    // (still useful for a human skimming `%TEMP%`) closes the gap.
    static WORK_DIR_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let call_id = WORK_DIR_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let work_dir =
        std::env::temp_dir().join(format!("cleave_emit_exe_{}_{call_id}", std::process::id()));
    std::fs::create_dir_all(&work_dir)
        .map_err(|e| vec![format!("failed to create a temp directory: {e}")])?;
    let object_path = work_dir.join("program.o");
    let shim_path = work_dir.join("shim.rs");

    emit_object(program, &cps_program, &object_path)?;

    let shim_src = if main_returns_i32 {
        format!(
            "unsafe extern \"C\" {{ fn {EXE_ENTRY_SYMBOL}() -> i32; }}\nfn main() {{ std::process::exit(unsafe {{ {EXE_ENTRY_SYMBOL}() }}); }}\n"
        )
    } else {
        format!(
            "unsafe extern \"C\" {{ fn {EXE_ENTRY_SYMBOL}(); }}\nfn main() {{ unsafe {{ {EXE_ENTRY_SYMBOL}() }}; }}\n"
        )
    };
    std::fs::write(&shim_path, shim_src)
        .map_err(|e| vec![format!("failed to write {}: {e}", shim_path.display())])?;

    let runtime_dir = cleave_rt_search_dir()?;
    let status = std::process::Command::new("rustc")
        .arg(&shim_path)
        .arg("-o")
        .arg(exe_path)
        .arg("-C")
        .arg(format!("link-arg={}", object_path.display()))
        .arg("-L")
        .arg(&runtime_dir)
        .arg("-l")
        .arg("cleave_rt")
        .status();
    let _ = std::fs::remove_dir_all(&work_dir);
    match status {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => Err(vec![format!(
            "rustc failed while linking the final executable (exit status: {s})"
        )]),
        Err(e) => Err(vec![format!("failed to run `rustc` (is it on PATH?): {e}")]),
    }
}

/// `cleave_rt.lib`/`libcleave_rt.a` sits alongside `cleave.exe` itself in an
/// ordinary `cargo build`/`cargo run` -- both land in the same
/// `target/<profile>/` directory. A `cargo test`-built test binary is one
/// level deeper (`target/<profile>/deps/`), where only a *hash-suffixed*
/// copy exists (Cargo's own convention for a dependency built once but
/// consumed by several test binaries) -- found directly testing this exact
/// code path from `tests/pipeline.rs`, so both layouts are checked here,
/// preferring the running executable's own immediate directory first.
/// Known, deliberate limitation beyond that: a `cleave` binary copied/
/// installed somewhere with neither layout nearby won't find one.
fn cleave_rt_search_dir() -> Result<PathBuf, Vec<String>> {
    let exe = std::env::current_exe().map_err(|e| {
        vec![format!(
            "failed to locate the running cleave executable: {e}"
        )]
    })?;
    let candidates = exe.ancestors().skip(1).take(2);
    for dir in candidates {
        if dir.join("cleave_rt.lib").exists() || dir.join("libcleave_rt.a").exists() {
            return Ok(dir.to_path_buf());
        }
    }
    exe.parent().map(|p| p.to_path_buf()).ok_or_else(|| {
        vec![format!(
            "cleave executable path {} has no parent directory",
            exe.display()
        )]
    })
}
