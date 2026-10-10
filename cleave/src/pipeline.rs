//! Shared high-level pipeline entry points, reused by both `main.rs`'s CLI
//! and `cleave-build`'s in-process build-script API (`cleave-build/src/
//! lib.rs`) -- extracted here specifically so the two never drift: "parse
//! this cleave source, type-check it, emit an object file and/or generated
//! Rust FFI bindings for its `export fn`s" needs to mean exactly the same
//! thing whether it's invoked from the command line or from someone else's
//! `build.rs`.

use crate::ast::Program;
use crate::cps::{
    CpsProgram, UnitBody, collect_mlir_types, collect_struct_schemas, collect_units,
    convert_program, eliminate_dead_code,
};
use crate::diag::{Diagnostic, SourceMap};
use crate::egraph::{DerivativeRequest, optimize_program, synthesize_derivatives};
use crate::escape::escaping_struct_vars;
use crate::mlir_lower::lower_program;
use crate::refcount::insert_refcounting;
use crate::registry::Registry;
use cleave_mlir::Context;
use cleave_mlir::dialect::DialectRegistry;
use cleave_mlir::ir::attribute::Attribute;
use cleave_mlir::ir::operation::{OperationBuilder};
use cleave_mlir::ir::{Identifier, Location, Module};
use cleave_mlir::utility::register_all_dialects;
use std::path::{Path, PathBuf};

/// The hardware a program is compiled for. The CPU only, today: a GPU backend
/// (Vulkan, through MLIR's `spirv` dialect) is planned, and every stage of
/// `lower_to_llvm` assumes this one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
}

/// How a program is compiled: by `lower_to_llvm` and code generation, for
/// every entry point (`--run`, `--emit-object`, `--emit-exe`, `cleave-build`).
#[derive(Debug, Clone)]
pub struct CodegenOptions {
    /// LLVM's optimization level, `0`-`3` (`target`).
    pub opt_level: u8,
    /// The OpenMP stage: the outer loops of `linalg` kernels parallelized
    /// (`affine-parallelize`, `convert-scf-to-openmp`). On by default, for
    /// the JIT as for objects. Sound with cleave's single-threaded arena
    /// (`cleave-rt`) because a parallelized loop never allocates.
    pub openmp: bool,
    /// The CPU code is generated for (`target`): an LLVM processor name, or
    /// `native` for the host's with every feature it has. `None`: the host's.
    pub target_cpu: Option<String>,
    /// Features added to or removed from the CPU's (`+avx2,-avx512f`).
    pub target_features: Option<String>,
    pub backend: Backend,
    /// MLIR's inliner (`lower_to_llvm`'s first stage). Off keeps every
    /// function boundary, to read one function's disassembly or profile;
    /// slower code, since producers and consumers then don't fuse.
    pub inline: bool,
    /// Structs whose lifetime the alias analysis proves affine come from the
    /// pool allocator without a refcount (`mlir_lower.rs::lower_program`).
    /// Off is the fallback for a shape the analysis gets wrong.
    pub affine_structs: bool,
    /// Debug info: a `DISubprogram` per function, every operation's location
    /// scoped by it (`mlir_lower.rs::build_di_subprograms`), and the module
    /// flags (CodeView on Windows, DWARF elsewhere). Off leaves plain
    /// locations, for dumps and profiles without the noise.
    pub debug_info: bool,
    /// LLVM's own loop unrolling, in the optimization pipeline
    /// (`cleave-mlir`'s `makeTransformer`). cleave's loops reach LLVM already
    /// tiled, vectorized and unrolled where it pays; LLVM unrolling them
    /// again was 55% of `opt -O2` on nanoLM's transformer kernel. `true`
    /// keeps the standard pipeline.
    pub llvm_loop_unroll: bool,
    /// Whether `spawn`ed calls run as tasks (`doc/plan-spawn.md`). `false`
    /// runs each one in place, where it's written, and drops the waits: serial
    /// elision, always a valid execution of the same program, same results,
    /// no OpenMP runtime involved. For the in-process test harnesses (their
    /// engines don't load libomp, and `leaks.rs` counts allocations per
    /// thread), and for comparing against a single-threaded run. On by
    /// default.
    pub tasks: bool,
    /// The size, in MLIR operations, past which a callee isn't inlined: its
    /// own body plus everything it would inline in turn
    /// (`cleave-limit-inlining`). Small functions, the elementwise
    /// operations fusion needs, stay inlined; a large one gains nothing from
    /// it (a call costs nothing next to thousands of operations) and costs a
    /// lot: LLVM's passes are superlinear in a function's size (nanoLM's
    /// `train_gpt` fully inlined grew from 225 operations to ~80,000, LICM
    /// alone taking 124 s of a 190 s compile).
    pub inline_threshold: usize,
}

/// `CodegenOptions::inline_threshold`'s default.
pub const DEFAULT_INLINE_THRESHOLD: usize = 1000;

impl Default for CodegenOptions {
    fn default() -> Self {
        Self {
            opt_level: 2,
            openmp: true,
            target_cpu: None,
            target_features: None,
            backend: Backend::Cpu,
            inline: true,
            affine_structs: true,
            debug_info: true,
            llvm_loop_unroll: true,
            tasks: true,
            inline_threshold: DEFAULT_INLINE_THRESHOLD,
        }
    }
}

/// `collect_units` + `convert_program` + `synthesize_derivatives`, bundled
/// -- see the module's own doc comment for why every pipeline entry point
/// needs all three, in this exact order.
pub fn build_cps_program(
    program: &Program,
    registry: &Registry,
    sources: Option<&SourceMap>,
) -> Result<CpsProgram, Vec<String>> {
    let units = collect_units(program, registry);
    let requests: Vec<DerivativeRequest> = units
        .iter()
        .filter_map(|u| match &u.body {
            UnitBody::Derivative(of, is_grad, grad_target_index) => Some(DerivativeRequest {
                name: u.name.clone(),
                of: of.clone(),
                is_grad: *is_grad,
                grad_target_index: *grad_target_index,
            }),
            _ => None,
        })
        .collect();
    let cps_program = convert_program(units, sources);
    let struct_schemas = collect_struct_schemas(program);
    let cps_program = synthesize_derivatives(cps_program, &requests, registry, &struct_schemas)?;
    crate::cps::check_spawn_purity(&cps_program, &reentrant_externs(program))?;
    Ok(cps_program)
}

/// Runs whole-program type inference and monomorphization purely to check
/// for errors -- a mandatory gate before CPS conversion, which assumes
/// every reachable unit's own types are already fully concrete and has no
/// error-reporting of its own.
/// The C symbols of every `#[reentrant]` extern: one that touches nothing but
/// its arguments (no global state), so a `spawn`ed task may call it even
/// though it isn't `#[pure]` (`cps::check_spawn_purity`). `sgemm` writes its
/// result through a pointer, which `#[pure]` must never claim, yet is
/// perfectly safe to call from several tasks at once.
fn reentrant_externs(program: &Program) -> crate::collections::HashSet<String> {
    let mut out = crate::collections::HashSet::default();
    let mut note = |f: &crate::ast::FnDecl| {
        if f.is_extern && f.attrs.iter().any(|a| a.name == "reentrant") {
            out.insert(f.extern_symbol.clone().unwrap_or_else(|| f.name.clone()));
        }
    };
    for item in &program.items {
        match &item.kind {
            crate::ast::ItemKind::Fn(f) => note(f),
            crate::ast::ItemKind::Impl(d) => d.fns.iter().for_each(&mut note),
            _ => {}
        }
    }
    out
}

pub fn check_type_errors(program: &Program, registry: &Registry) -> Result<(), Vec<Diagnostic>> {
    // Coherence first: two impls that could both apply to one type leave
    // dispatch with no principled way to choose, so every later choice
    // assumes this holds.
    let overlaps = crate::infer::Infer::new(registry).check_no_overlapping_impls();
    let mut diags: Vec<Diagnostic> = overlaps.iter().map(Diagnostic::from).collect();
    let (_, errs) = crate::monomorphize::dump_monomorphized(program, registry);
    diags.extend(errs.iter().map(Diagnostic::from));
    diags.extend(check_mutability_errors(program));
    diags.extend(check_const_decl_errors(program, registry));
    diags.extend(check_impl_completeness(program, registry));
    diags.extend(check_lambda_positions(program));
    if diags.is_empty() { Ok(()) } else { Err(diags) }
}

/// A lambda is supported bound by a `let`, then called by that name (`let f =
/// fn(x: i32) -> i32 { x + 1 }; f(2)`) or passed by that name to a call
/// (`apply(f, 2)`, written in place too: `apply(fn(x: i32) -> i32 { x + 1 },
/// 2)`); as any other value (an array element, a field, a result) it isn't
/// yet, and used to panic the compiler's CPS conversion: a located error
/// instead. `doc/backlog.md`: "A lambda returned from a function, or stored
/// in a struct/array field".
fn check_lambda_positions(program: &Program) -> Vec<Diagnostic> {
    use crate::ast::{Block, ElseBranch, Expr, ExprKind, ItemKind, StmtKind};

    fn unsupported(span: crate::ast::Span, what: &str) -> Diagnostic {
        Diagnostic {
            severity: crate::diag::Severity::Error,
            message: format!(
                "{what}: a function value can only be called or passed to a call for now"
            ),
            span: Some(span),
            notes: Vec::new(),
        }
    }

    // `lambdas`: the names in scope bound to a lambda.
    fn walk_block(block: &Block, lambdas: &mut Vec<String>, out: &mut Vec<Diagnostic>) {
        let depth = lambdas.len();
        for stmt in &block.stmts {
            match &stmt.kind {
                StmtKind::Let { name, value, .. } => {
                    if let ExprKind::Lambda { body, .. } = &value.kind {
                        walk_block(body, lambdas, out);
                        lambdas.push(name.clone());
                    } else {
                        walk_expr(value, lambdas, out);
                        // A later `let` of the same name shadows the lambda.
                        if let Some(i) = lambdas.iter().rposition(|n| n == name) {
                            lambdas.remove(i);
                        }
                    }
                }
                StmtKind::Assign { target, value } => {
                    walk_expr(target, lambdas, out);
                    walk_expr(value, lambdas, out);
                }
                StmtKind::Expr(e) => walk_expr(e, lambdas, out),
                StmtKind::Break(e) => {
                    if let Some(e) = e {
                        walk_expr(e, lambdas, out);
                    }
                }
                StmtKind::Sync => {}
            }
        }
        if let Some(tail) = &block.tail {
            walk_expr(tail, lambdas, out);
        }
        lambdas.truncate(depth);
    }

    fn walk_expr(expr: &Expr, lambdas: &mut Vec<String>, out: &mut Vec<Diagnostic>) {
        match &expr.kind {
            ExprKind::Match { .. } => unreachable!("a `match` is lowered by `driver::desugar_enums`"),
            ExprKind::Lambda { .. } => out.push(unsupported(expr.span, "a lambda used as a value")),
            ExprKind::Path(p) if p.segments.len() == 1 && lambdas.contains(&p.segments[0]) => {
                out.push(unsupported(expr.span, &format!("`{}` used as a value", p.segments[0])));
            }
            ExprKind::NumberLit { .. }
            | ExprKind::ImaginaryLit { .. }
            | ExprKind::BoolLit(_)
            | ExprKind::Path(_)
            | ExprKind::PackRef(_) => {}
            // The callee is a name, called. A lambda's name passed straight
            // to a call is supported too: the callee is specialized for it
            // (`cps.rs::build_higher_order_specializations`), as for a
            // comprehension's function or a lambda written in place
            // (`lower.rs::hoist_lambda_args`).
            ExprKind::Call(_, _, args, _) => {
                for a in args {
                    match &a.kind {
                        ExprKind::Path(p) if p.segments.len() == 1 && lambdas.contains(&p.segments[0]) => {}
                        // In place: a comprehension's function, until
                        // `unroll.rs` binds it (in generic code, per instance).
                        ExprKind::Lambda { body, .. } => walk_block(body, lambdas, out),
                        _ => walk_expr(a, lambdas, out),
                    }
                }
            }
            ExprKind::Spawn(e) | ExprKind::FieldAccess(e, _) => walk_expr(e, lambdas, out),
            ExprKind::Index(base, indices) => {
                walk_expr(base, lambdas, out);
                indices.iter().for_each(|i| walk_expr(i, lambdas, out));
            }
            ExprKind::ArrayLit(items) => items.iter().for_each(|i| walk_expr(i, lambdas, out)),
            ExprKind::ArrayRepeat { value, count } => {
                walk_expr(value, lambdas, out);
                walk_expr(count, lambdas, out);
            }
            ExprKind::StructLit(_, _, fields) => fields.iter().for_each(|(_, e)| walk_expr(e, lambdas, out)),
            ExprKind::If { cond, then_branch, else_branch } => {
                walk_expr(cond, lambdas, out);
                walk_block(then_branch, lambdas, out);
                match else_branch.as_deref() {
                    Some(ElseBranch::If(e)) => walk_expr(e, lambdas, out),
                    Some(ElseBranch::Block(b)) => walk_block(b, lambdas, out),
                    None => {}
                }
            }
            ExprKind::While { cond, body } => {
                walk_expr(cond, lambdas, out);
                walk_block(body, lambdas, out);
            }
            ExprKind::For { start, end, body, .. } => {
                walk_expr(start, lambdas, out);
                walk_expr(end, lambdas, out);
                walk_block(body, lambdas, out);
            }
            ExprKind::ForIn { iter, body, .. } => {
                walk_expr(iter, lambdas, out);
                walk_block(body, lambdas, out);
            }
            ExprKind::Loop { body } | ExprKind::Block(body) => walk_block(body, lambdas, out),
        }
    }

    let mut out = Vec::new();
    for item in &program.items {
        let fns: Vec<&crate::ast::FnDecl> = match &item.kind {
            ItemKind::Fn(f) => vec![f],
            ItemKind::Impl(d) => d.fns.iter().collect(),
            _ => vec![],
        };
        for f in fns {
            if let Some(body) = &f.body {
                walk_block(body, &mut Vec::new(), &mut out);
            }
        }
    }
    out
}

/// Every impl defines every function its algebra declares. A missing one
/// went unnoticed until something called it, possibly from a rule the
/// compiler applies itself: `Sum<Tensor<T, N>, T>` had no `broadcast`, which
/// `sum`'s adjoint calls, and differentiating a rank-1 `sum` panicked in the
/// e-graph ("extracted `Op` node `Sum::broadcast<..>` is in none of this
/// module's own lookup tables").
/// The targets of every impl of `algebra`, as written (`Tensor<T, Dims...>`).
fn impls_for(program: &Program, algebra: &str) -> Vec<String> {
    program
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            crate::ast::ItemKind::Impl(d) if d.algebra == algebra => Some(crate::print::fmt_type(&d.target)),
            _ => None,
        })
        .collect()
}

fn check_impl_completeness(program: &Program, registry: &Registry) -> Vec<Diagnostic> {
    program
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            crate::ast::ItemKind::Impl(d) => {
                let target = crate::print::fmt_type(&d.target);
                let missing: Vec<&str> = registry
                    .fn_names(&d.algebra)
                    .into_iter()
                    .filter(|name| !d.fns.iter().any(|f| f.name == *name))
                    .collect();
                if !missing.is_empty() {
                    return Some(Diagnostic::error(
                        format!(
                            "`impl {}<{target}>` doesn't define {}",
                            d.algebra,
                            missing.iter().map(|m| format!("`{m}`")).collect::<Vec<_>>().join(", ")
                        ),
                        item.span,
                    ));
                }
                // A super-algebra with functions of its own (`algebra Ring<T>
                // : Additive`) needs its own impl: a bound alone would let
                // `Additive<T>` hold with no `add` anywhere. A marker one
                // (`Int<T> : Num`, no functions) is implied, as before.
                let unmet: Vec<&String> = registry
                    .algebra_bounds(&d.algebra)
                    .iter()
                    .filter(|b| !registry.fn_names(b).is_empty() && !impls_for(program, b).contains(&target))
                    .collect();
                (!unmet.is_empty()).then(|| {
                    Diagnostic::error(
                        format!(
                            "`impl {}<{target}>` needs {}",
                            d.algebra,
                            unmet.iter().map(|b| format!("`impl {b}<{target}>`")).collect::<Vec<_>>().join(", ")
                        ),
                        item.span,
                    )
                })
            }
            _ => None,
        })
        .collect()
}

/// Every top-level `const NAME: T = expr;`'s own initializer must actually
/// have evaluated -- `registry.rs::Registry::eval_global_consts`'s own
/// evaluator is deliberately *permissive by omission* (an expression shape
/// it doesn't recognize, including a reference to an undeclared name, is
/// silently left unevaluated there, not reported -- that module's own doc
/// comment says this diagnostic is "`infer.rs`'s own job"). Without this
/// check, a const whose value never resolved would still type-check fine
/// at every *use* site (`Registry::global_const_type` is populated
/// unconditionally, straight from the AST's own declared type, regardless
/// of whether evaluation succeeded) and only fail much later, confusingly,
/// when `cps.rs::convert_expr` can't find a value for it either
/// (`ConcreteUnit::global_consts`, also empty for this name) -- a real,
/// found-by-testing gap, not a hypothetical: `const BOGUS: i32 =
/// TOTALLY_UNDECLARED_NAME;` used to reach `panic!("CPS: unbound variable
/// ...")` instead of a clean, located diagnostic.
fn check_const_decl_errors(program: &Program, registry: &Registry) -> Vec<Diagnostic> {
    program
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            crate::ast::ItemKind::Const(d) if !registry.has_global_const_value(&d.name) => {
                Some(Diagnostic::error(
                    format!(
                        "`const {}`'s own initializer is not a compile-time constant expression",
                        d.name
                    ),
                    d.value.span,
                ))
            }
            // A `define` can legitimately have nothing to evaluate at all
            // (`grammar.pest`'s own `define_decl` doc comment: `= expr` is
            // optional there, unlike `const_decl`'s own mandatory one) --
            // still a real error if it stays unresolved (neither a default
            // *nor* an external `--define` ever supplied a value), just a
            // different, clearer message than a `const`'s own ("not a
            // compile-time constant" would be actively misleading for a
            // `define` with no default at all -- there's no expression here
            // to have failed evaluating). `item.span` (the whole `define
            // NAME: T [= expr];` item), not `d.value`'s -- that's `None`
            // exactly in the case this message needs to cover.
            crate::ast::ItemKind::Define(d) if !registry.has_global_const_value(&d.name) => {
                Some(Diagnostic::error(
                    match &d.value {
                        Some(_) => format!(
                            "`define {}`'s own default is not a compile-time constant expression, \
                             and no `--define {}=...` provided a value",
                            d.name, d.name
                        ),
                        None => format!(
                            "`define {}` has no default and was not provided via `--define {}=...`",
                            d.name, d.name
                        ),
                    },
                    item.span,
                ))
            }
            _ => None,
        })
        .collect()
}

/// A purely syntactic pass (`crate::infer::check_mutability`, no type
/// information needed), run once per `fn` body anywhere in the program.
fn check_mutability_errors(program: &Program) -> Vec<Diagnostic> {
    let mut errors = Vec::new();
    for item in &program.items {
        let fns: Vec<&crate::ast::FnDecl> = match &item.kind {
            crate::ast::ItemKind::Fn(f) => vec![f],
            crate::ast::ItemKind::Impl(d) => d.fns.iter().collect(),
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
///
/// `Ok(true)` when the emitted code calls the OpenMP runtime — `--openmp`'s
/// parallel loops, or `spawn`'s tasks (`doc/plan-spawn.md`) — so a caller
/// linking the object (`cleave-build`) must link libomp too.
pub fn emit_from_program(
    program: &Program,
    registry: &Registry,
    sources: &SourceMap,
    object_path: Option<&Path>,
    bindings_path: Option<&Path>,
    options: &CodegenOptions,
) -> Result<bool, Vec<String>> {
    let start = std::time::Instant::now();
    check_type_errors(program, registry).map_err(|errs| render_all(&errs, sources))?;
    report_stage("type checking (with a monomorphization)", start);
    let cps_program = build_optimized_cps(program, registry, Some(sources))?;
    let needs_openmp = options.openmp || (options.tasks && crate::cps::uses_spawn(&cps_program));

    if let Some(bindings_path) = bindings_path {
        let bindings = crate::rust_bindings::generate_rust_bindings(&cps_program.funcs)?;
        std::fs::write(bindings_path, bindings)
            .map_err(|e| vec![format!("failed to write {}: {e}", bindings_path.display())])?;
    }

    if let Some(object_path) = object_path {
        emit_object(program, &cps_program, object_path, options, sources)?;
    }

    Ok(needs_openmp)
}

/// `build_cps_program` + the standard `optimize_program` / `eliminate_dead_
/// code` sequencing every pipeline entry point needs -- shared by `emit_
/// from_program` and `emit_exe`. Deliberately *not* dead-code-eliminated
/// before `optimize_program` runs (see `--dump-cps-optimized`'s own comment
/// in `main.rs` for the full reasoning: an axiom/`derivative`/`adjoint`
/// rule can reference a unit no ordinary call site reaches at all, which a
/// pre-optimization sweep would strip before `optimize_program` ever gets a
/// chance to need it) -- the single sweep *after* still catches the
/// opposite case, a unit `optimize_program` itself made unreachable.
pub fn build_optimized_cps(
    program: &Program,
    registry: &Registry,
    sources: Option<&SourceMap>,
) -> Result<CpsProgram, Vec<String>> {
    let start = std::time::Instant::now();
    let cps_program = build_cps_program(program, registry, sources)?;
    report_stage("CPS conversion (units, derivatives)", start);
    let start = std::time::Instant::now();
    let (cps_program, _) = optimize_program(cps_program, registry, false);
    report_stage("e-graph optimization", start);
    let start = std::time::Instant::now();
    let cps_program = eliminate_dead_code(cps_program);
    // Last CPS-to-CPS step, strictly after the e-graph pass -- see
    // `refcount`'s own module doc comment for why (it has no notion of
    // `Retain`/`Release`'s own effectful ordering, inserting them earlier
    // risks its own rewriting scrambling them).
    let struct_schemas = collect_struct_schemas(program);
    let mlir_types = collect_mlir_types(program);
    // Escape analysis: which `PrimOp::Struct`-bound vars cross a jump
    // (slots, need header + RC) vs die before any jump (arena temps).
    // Must run after e-graph optimisation — the optimised CPS is what
    // `insert_refcounting` and `lower_program` both operate on.
    let escaping = escaping_struct_vars(&cps_program);
    let refcounted = insert_refcounting(cps_program, &struct_schemas, &mlir_types, &escaping);
    report_stage("dead code, refcounting", start);
    Ok(refcounted)
}

/// Parses/merges/resolves `sources_in` from scratch (`driver::compile`'s
/// own shape: one or more `(file_name, text)` pairs) and runs the result
/// through `emit_from_program` -- the simple, one-call API `cleave-build`
/// actually wants: a build script has no pre-existing `Program` lying
/// around the way `main.rs` does.
///
/// `defines` mirrors the CLI `--define NAME=VALUE` flag (`main.rs`'s own
/// `build_registry` helper) -- `cleave-build::Build::define` is its
/// `build.rs`-facing counterpart. A `--define`-level config error (unknown
/// name, targets a real `const`, badly-typed value -- `Registry::build_
/// with_defines`'s own doc comment) is reported through the same `Vec
/// <String>` error channel every other failure here already uses, not a
/// separate return shape.
/// What `compile_and_emit` produced besides its files: whether the object
/// needs libomp, and every source file the compile read (the entry files and
/// each module `use` reached, the stdlib's included): what a build script
/// depends on.
#[derive(Debug)]
pub struct Emitted {
    pub needs_openmp: bool,
    pub loaded_files: Vec<String>,
}

pub fn compile_and_emit(
    sources_in: Vec<(String, String)>,
    project_dirs: &[PathBuf],
    object_path: Option<&Path>,
    bindings_path: Option<&Path>,
    options: &CodegenOptions,
    defines: &[(String, String)],
) -> Result<Emitted, Vec<String>> {
    let start = std::time::Instant::now();
    let (result, sources) = crate::driver::compile(sources_in, project_dirs);
    let mut program = result.map_err(|errs| render_all(&errs, &sources))?;
    report_stage("parsing, name resolution", start);
    Registry::apply_defines(&mut program, defines);
    let (registry, define_errors) = Registry::build_with_defines(&program, defines, options.openmp);
    if !define_errors.is_empty() {
        return Err(define_errors);
    }
    let needs_openmp = emit_from_program(&program, &registry, &sources, object_path, bindings_path, options)?;
    Ok(Emitted { needs_openmp, loaded_files: sources.file_names() })
}

/// Registers every `cleave-rt` function a program may call, by pointer,
/// with the JIT `engine` (`--run` and the in-process tests; an object file
/// leaves them for the linker). One line per `extern fn` `cleave-rt`
/// provides; by pointer rather than by name lookup, which sidesteps the
/// Windows/MSVC questions of which CRT exports what. `memrefCopy` is
/// `cleave-rt`'s own version of MLIR's runtime helper, which
/// `one-shot-bufferize` calls by name for a defensive copy.
///
/// SAFETY: each `cleave_rt::*` pointer is a real, valid `extern "C" fn`,
/// live for the process's whole lifetime.
pub unsafe fn register_cleave_rt_symbols(engine: &cleave_mlir::ExecutionEngine) {
    unsafe {
        engine.register_symbol("memrefCopy", cleave_rt::memrefCopy as *mut ());
        engine.register_symbol("cleave_parallel_threads", cleave_rt::cleave_parallel_threads as *mut ());
        engine.register_symbol("cleave_bind_worker", cleave_rt::cleave_bind_worker as *mut ());
        engine.register_symbol("rand_seed", cleave_rt::rand_seed as *mut ());
        engine.register_symbol("rand_state", cleave_rt::rand_state as *mut ());
        engine.register_symbol("cleave_ckpt_create", cleave_rt::checkpoint::cleave_ckpt_create as *mut ());
        engine.register_symbol("cleave_ckpt_open", cleave_rt::checkpoint::cleave_ckpt_open as *mut ());
        engine.register_symbol("cleave_ckpt_close", cleave_rt::checkpoint::cleave_ckpt_close as *mut ());
        engine.register_symbol("cleave_ckpt_write_f32s", cleave_rt::checkpoint::cleave_ckpt_write_f32s as *mut ());
        engine.register_symbol("cleave_ckpt_read_f32s", cleave_rt::checkpoint::cleave_ckpt_read_f32s as *mut ());
        engine.register_symbol("cleave_ckpt_write_f32", cleave_rt::checkpoint::cleave_ckpt_write_f32 as *mut ());
        engine.register_symbol("cleave_ckpt_read_f32", cleave_rt::checkpoint::cleave_ckpt_read_f32 as *mut ());
        engine.register_symbol("cleave_ckpt_write_f64", cleave_rt::checkpoint::cleave_ckpt_write_f64 as *mut ());
        engine.register_symbol("cleave_ckpt_read_f64", cleave_rt::checkpoint::cleave_ckpt_read_f64 as *mut ());
        engine.register_symbol("cleave_ckpt_write_i32", cleave_rt::checkpoint::cleave_ckpt_write_i32 as *mut ());
        engine.register_symbol("cleave_ckpt_read_i32", cleave_rt::checkpoint::cleave_ckpt_read_i32 as *mut ());
        engine.register_symbol("cleave_ckpt_write_i64", cleave_rt::checkpoint::cleave_ckpt_write_i64 as *mut ());
        engine.register_symbol("cleave_ckpt_read_i64", cleave_rt::checkpoint::cleave_ckpt_read_i64 as *mut ());
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
            "print_buffer_bytes",
            cleave_rt::print_buffer_bytes as *mut (),
        );
        engine.register_symbol("format_f32", cleave_rt::format_f32 as *mut ());
        engine.register_symbol("format_f64", cleave_rt::format_f64 as *mut ());
        engine.register_symbol("cleave_alloc", cleave_rt::cleave_alloc as *mut ());
        engine.register_symbol("cleave_alloc_rc", cleave_rt::cleave_alloc_rc as *mut ());
        engine.register_symbol("cleave_retain", cleave_rt::cleave_retain as *mut ());
        engine.register_symbol("cleave_release", cleave_rt::cleave_release as *mut ());
        // `doc/plan-affine-ownership.md`'s Stage 2 -- only ever called when
        // `CLEAVE_AFFINE_STRUCTS=1` (`mlir_lower.rs::lower_program`'s own
        // doc comment), registered unconditionally here regardless, same
        // as every other `cleave-rt` symbol on this list.
        engine.register_symbol("cleave_alloc_pool", cleave_rt::cleave_alloc_pool as *mut ());
        engine.register_symbol(
            "cleave_release_pool",
            cleave_rt::cleave_release_pool as *mut (),
        );
        engine.register_symbol(
            "cleave_release_void",
            cleave_rt::cleave_release_void as *mut (),
        );
        engine.register_symbol(
            "cleave_alloc_local",
            cleave_rt::cleave_alloc_local as *mut (),
        );
        engine.register_symbol(
            "cleave_region_enter",
            cleave_rt::cleave_region_enter as *mut (),
        );
        engine.register_symbol(
            "cleave_region_exit",
            cleave_rt::cleave_region_exit as *mut (),
        );
        engine.register_symbol("cleave_buffer_alloc", cleave_rt::cleave_buffer_alloc as *mut ());
        engine.register_symbol("cleave_buffer_grow", cleave_rt::cleave_buffer_grow as *mut ());
        engine.register_symbol("cleave_buffer_capacity", cleave_rt::cleave_buffer_capacity as *mut ());
        engine.register_symbol("cleave_buffer_free_data", cleave_rt::cleave_buffer_free_data as *mut ());
        // `stdlib/blas/blas.cleave`'s own `raw_sgemm` extern -- lazily loads
        // `openblas.dll` on first real call (`cleave_rt::blas_dynload`), so
        // registering it here unconditionally costs nothing for a program
        // that never calls into `blas`.
        engine.register_symbol("cleave_blas_sgemm", cleave_rt::cleave_blas_sgemm as *mut ());
    }
}

/// Registers, once per process, every pass a textual pipeline may name:
/// MLIR's and cleave's (`cleave_mlir::register_passes`). MLIR's pass
/// registry is a global, unsynchronized table: registering at each use wrote
/// to it while another thread compiling at the same time (the test harnesses
/// run compilations in parallel) read it, an intermittent
/// `STATUS_ACCESS_VIOLATION`. The shim registers inside a function-local
/// static's initialization, which other threads wait for, and runs it before
/// parsing any pipeline.
pub fn register_passes() {
    cleave_mlir::register_passes();
}

/// Rows of a BLAS product computed per tile when it is fused with its
/// elementwise consumer (`cleave-blas-tile-and-fuse`): a tile of a product
/// 1024 wide is 512 KB, within a core's 1 MB L2 alongside its slice of `A`;
/// fewer rows would call `sgemm` more often, each call packing `B` again.
const BLAS_TILE_ROWS: i64 = 128;

/// The matmul schedule's row tile (`tile_using_forall tile_sizes [8, ..]` in
/// `matmul_vectorize.transform.mlir`): a product whose rows aren't a multiple
/// of it is split first into whole tiles and a static remainder
/// (`cleave-split-row-remainders`).
const MATMUL_ROW_TILE: i64 = 8;

/// The matmul tile/vectorize schedule, compiled into the binary and loaded
/// into the context's transform library from memory
/// (`cleave_mlir::load_transform_library`).
const MATMUL_SCHEDULE: &str = include_str!("../mlir/matmul_vectorize.transform.mlir");

/// `CLEAVE_TIME_STAGES=1`: how long each stage of a compile takes, on stderr:
/// the front end's (parsing, type checking, CPS, e-graph, refcounting), each
/// pass pipeline of `lower_to_llvm` (named by its line here) and the rest of
/// `emit_object`.
/// What a compile spends its time on, for `doc/backlog.md`'s compile-time
/// entry.
fn time_stages() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CLEAVE_TIME_STAGES").is_ok_and(|v| v == "1"))
}

fn report_stage(what: &str, since: std::time::Instant) {
    if time_stages() {
        eprintln!("cleave stage: {what}: {:.2} s", since.elapsed().as_secs_f64());
    }
}

/// Runs one stage of `lower_to_llvm`: the textual pipeline `pipeline`
/// (`builtin.module(...)`), through the shim, timed and with cleave's pass
/// statistics printed under `CLEAVE_TIME_STAGES=1`. `what` names the stage in
/// an error.
fn run_stage(module: &mut Module, what: &str, pipeline: &str) -> Result<(), Vec<String>> {
    let start = std::time::Instant::now();
    // SAFETY: `module` is a valid module, borrowed mutably here.
    let result = unsafe { cleave_mlir::run_pipeline(module.to_raw(), pipeline, time_stages()) };
    report_stage(what, start);
    result.map_err(|e| {
        vec![if e.is_empty() {
            format!("MLIR-to-LLVM lowering pass failed ({what})")
        } else {
            format!("MLIR-to-LLVM lowering: invalid pipeline for {what}: {e}")
        }]
    })
}
/// Lowers `module`, as `mlir_lower.rs` built it, to the LLVM dialect: six
/// textual pipelines (`run_stage`), the passes in each run in order by one
/// pass manager. Shared by every entry point that compiles (`--run`,
/// `--dump-mlir-lowered`, `emit_object`), so they can't drift apart.
///
/// The stages, and why each pass is where it is:
///
/// 1. **Tensors.** Calls are inlined (`cleave-limit-inlining` first keeps the
///    largest callees, and `#[no_inline]` ones, out of line), so producers
///    and consumers meet in one function: elementwise operations become
///    `linalg.generic`s and fuse, one kernel writing one buffer instead of a
///    chain through intermediates. BLAS-marked products are tiled with their
///    elementwise consumer and lowered to `sgemm` calls; the other matmuls go
///    through the tile-and-vectorize schedule (`matmul_vectorize.transform.
///    mlir`, loaded into the context first), split first when their rows
///    aren't a multiple of its row tile. `loop-invariant-subset-hoisting`
///    then keeps a tile's accumulator in registers across the reduction,
///    which it can only do while the IR is still tensors.
/// 2. **Bufferization.** Empty tensors eliminated so producers write into
///    their destination (a struct field's buffer), then One-Shot Bufferize
///    across function boundaries. cleave's copy rewrites follow, each
///    removing a copy bufferization left: blocks written in place, a dead
///    buffer's copy into a fresh one, a call's result copied into its
///    destination (`buffer-results-to-out-params` between them turns returned
///    buffers into out-parameters, which the next rewrites forward).
/// 3. **Deallocation.** Ownership-based deallocation, simplified and lowered;
///    cleave's adoptions (buffers whose ownership the refcounted runtime
///    takes) lowered; frees moved to the last use; `spawn` markers lowered to
///    OpenMP tasks, or removed with tasks off.
/// 4. **Loops and vectors.** Dead symbols removed (the callees inlining left
///    behind, whose strided signatures the vectorizer would reject), then
///    `cse` (which the self-copy elimination after it needs: two identical
///    subviews must be one value). `linalg` becomes affine loops (`scf`
///    loops for an op whose bounds aren't affine);
///    with OpenMP, the outer loops are parallelized; `mulf`/`addf` may
///    contract into FMAs; the affine super-vectorizer vectorizes the loops
///    the schedule didn't (it reports, without failing, the strided loops it
///    can't handle; those diagnostics are dropped). Permuted transfers it
///    created, which LLVM can't lower, become scalar loops; transcendentals
///    become polynomials that vectorize; every loop body frees its stack at
///    each iteration; parallel loops become OpenMP regions whose threads are
///    placed on cores.
/// 5. **LLVM.** Vectors, OpenMP, then everything else to the LLVM dialect;
///    the C-interface wrappers' debug locations stripped (they reuse their
///    function's, which LLVM rejects); `malloc`/`free` made calls to cleave's
///    allocator. With debug info, the module flags (CodeView on Windows).
/// 6. **Final.** Synthesized operations given their nearest source line;
///    large aggregate copies made `memcpy`s; argument slots hoisted with
///    bounded lifetimes; functions kept out of line marked so for LLVM.
///
/// `CLEAVE_DUMP_PRE_BUFFERIZE`, `CLEAVE_DUMP_POST_OUT_PARAMS` and
/// `CLEAVE_DUMP_POST_DEALLOC` write the module between stages 1-2, inside 2
/// (after the out-parameter rewrites) and after 3: bufferization's decisions
/// leave no trace after the fact, the two sides of it are what to compare.
pub fn lower_to_llvm<'c>(
    context: &'c Context,
    module: &mut Module<'c>,
    options: &CodegenOptions,
) -> Result<(), Vec<String>> {
    let Backend::Cpu = options.backend;
    register_passes();
    let dump = |module: &Module, variable: &str| {
        if let Ok(path) = std::env::var(variable) {
            std::fs::write(&path, module.as_operation().to_string())
                .unwrap_or_else(|e| eprintln!("{variable}: failed to write {path}: {e}"));
        }
    };

    // SAFETY: `context` is the module's, used by this thread only.
    if !unsafe { cleave_mlir::load_transform_library(context.to_raw(), MATMUL_SCHEDULE, "matmul_vectorize.transform.mlir") } {
        return Err(vec!["failed to load the matmul schedule".to_string()]);
    }
    run_stage(
        module,
        "tensors",
        &format!(
            "builtin.module(cleave-limit-inlining{{threshold={threshold}}},{inline}\
             convert-elementwise-to-linalg,linalg-fuse-elementwise-ops,\
             cleave-blas-tile-and-fuse{{rows={BLAS_TILE_ROWS}}},cleave-lower-blas-matmuls,\
             cleave-split-row-remainders{{rows={MATMUL_ROW_TILE}}},transform-interpreter{{entry-point=__transform_main}},\
             loop-invariant-subset-hoisting,cleave-reuse-dying-inputs)",
            threshold = options.inline_threshold,
            inline = if options.inline { "inline," } else { "" },
        ),
    )?;
    dump(module, "CLEAVE_DUMP_PRE_BUFFERIZE");

    run_stage(
        module,
        "bufferization",
        "builtin.module(eliminate-empty-tensors,\
         one-shot-bufferize{bufferize-function-boundaries=true function-boundary-type-conversion=identity-layout-map allow-return-allocs-from-loops=true},\
         cleave-elide-block-copies,scf-forall-to-parallel,cleave-forward-dead-source-copies,\
         buffer-results-to-out-params{hoist-static-allocs=true},\
         cleave-forward-out-param-copies,cleave-forward-copies-to-destinations)",
    )?;
    dump(module, "CLEAVE_DUMP_POST_OUT_PARAMS");

    run_stage(
        module,
        "deallocation",
        &format!(
            "builtin.module(cleave-fold-passthrough-iter-args,ownership-based-buffer-deallocation,\
             buffer-deallocation-simplification,bufferization-lower-deallocations,\
             cleave-lower-adoptions,cleave-dealloc-at-last-use,convert-bufferization-to-memref,\
             cleave-lower-spawns{{tasks={}}})",
            options.tasks
        ),
    )?;
    dump(module, "CLEAVE_DUMP_POST_DEALLOC");

    let strided_loops_skipped = context.attach_diagnostic_handler(|diagnostic| {
        diagnostic.to_string().contains("NYI: non-trivial layout map")
    });
    let loops = run_stage(
        module,
        "loops and vectors",
        &format!(
            "builtin.module(symbol-dce,func.func(lower-vector-multi-reduction),cse,\
             cleave-eliminate-self-copies,cleave-lower-dynamic-copies,\
             canonicalize,expand-strided-metadata,lower-affine,canonicalize,\
             convert-vector-to-scf,cleave-lower-non-affine-linalg,convert-linalg-to-affine-loops,\
             func.func(affine-fold-memref-alias-ops),{parallelize}\
             cleave-mark-contract,func.func(affine-super-vectorize{{virtual-vector-size=16}}),\
             cleave-lower-permuted-transfers,cleave-approximate-math,lower-affine,\
             cleave-insert-stack-scopes,{openmp}cleave-bind-teams)",
            parallelize = if options.openmp { "func.func(affine-parallelize{max-nested=1})," } else { "" },
            openmp = if options.openmp { "convert-scf-to-openmp," } else { "" },
        ),
    );
    context.detach_diagnostic_handler(strided_loops_skipped);
    loops?;

    run_stage(
        module,
        "LLVM",
        &format!(
            "builtin.module(convert-vector-to-llvm,cleave-convert-openmp-if-used{{always={}}},\
             convert-to-llvm,convert-scf-to-cf,finalize-memref-to-llvm,convert-to-llvm,\
             reconcile-unrealized-casts,cleave-strip-ciface-debug-info,cleave-unify-tensor-allocations)",
            options.openmp
        ),
    )?;
    if options.debug_info {
        // CodeView (`.debug$S`, what a `.pdb` is built from) on Windows, DWARF
        // elsewhere, for the `DISubprogram`s `mlir_lower.rs` attached.
        let code_view = if cfg!(windows) { "#llvm.mlir.module_flag<warning, \"CodeView\", 1 : i32>, " } else { "" };
        let flags = Attribute::parse(
            context,
            &format!("[{code_view}#llvm.mlir.module_flag<max, \"Debug Info Version\", 3 : i32>]"),
        )
        .expect("pipeline: failed to parse llvm.module_flags attribute");
        let op = OperationBuilder::new("llvm.module_flags", Location::unknown(context))
            .add_attributes(&[(Identifier::new(context, "flags"), flags)])
            .build()
            .expect("pipeline: failed to build llvm.module_flags op");
        module.body().append_operation(op);
    }

    run_stage(
        module,
        "final",
        &format!(
            "builtin.module(cleave-backfill-locations,cleave-copy-aggregates-in-memory{{min-bytes={}}},\
             cleave-hoist-arg-slots,cleave-apply-no-inline)",
            crate::mlir_lower::BY_POINTER_MIN_BYTES
        ),
    )
}

/// Pre-order walk: finds every `_mlir_ciface_*` wrapper `llvm.func`
/// anywhere under `op` and gives its own location, and every op nested
/// inside it, a bare `Location::unknown` -- see `lower_to_llvm`'s own call-
/// site comment for why. Stops descending once a matching function is found
/// (its whole subtree is handled by `force_location` in one shot).
///
/// `pub`, not `pub(crate)`: every test file with its own bespoke, minimal
/// JIT pass pipeline (`tests/{mlir_lower,egraph,refcount,
/// unify_alloc,user_guide}.rs` -- found by direct testing, not assumed, when
/// the real `cargo test -p cleave` suite hard-crashed the LLVM verifier the
/// first time it ran after `mlir_lower.rs` started fusing a `DISubprogram`
/// onto every function, not just ones surviving `--inline`) needs to call
/// this too, right after its own `--convert-to-llvm` sequence -- none of
/// them go through `lower_to_llvm` itself.
pub fn strip_ciface_wrapper_debug_info(module: &mut Module) {
    // SAFETY: `module` is a valid module, borrowed mutably here.
    unsafe { cleave_mlir::run_pipeline(module.to_raw(), "builtin.module(cleave-strip-ciface-debug-info)", false) }
        .expect("cleave-strip-ciface-debug-info");
}

/// Runs the textual pipeline `pipeline` (`builtin.module(...)`) on `module`:
/// for harnesses running their own passes rather than `lower_to_llvm`.
pub fn run_passes(module: &mut Module, pipeline: &str) -> Result<(), String> {
    register_passes();
    // SAFETY: `module` is a valid module, borrowed mutably here.
    unsafe { cleave_mlir::run_pipeline(module.to_raw(), pipeline, false) }
}

/// A JIT for `module`, in the LLVM dialect, compiled for the host at
/// optimization level `opt_level`: for harnesses running their own passes.
pub fn jit(module: &Module, opt_level: usize) -> cleave_mlir::ExecutionEngine {
    let target = cleave_mlir::Target::new(None, None, opt_level, false, true).expect("the host target");
    // SAFETY: `module` is a valid module, borrowed for the call.
    unsafe { cleave_mlir::ExecutionEngine::new(module.to_raw(), &target, &[]) }
        .unwrap_or_else(|e| panic!("failed to compile the module: {e}"))
}

/// The target `options` ask for (`cleave_mlir::Target`): for objects
/// and JITs alike.
pub fn target(options: &CodegenOptions) -> Result<cleave_mlir::Target, Vec<String>> {
    cleave_mlir::Target::new(
        options.target_cpu.as_deref(),
        options.target_features.as_deref(),
        options.opt_level as usize,
        false,
        options.llvm_loop_unroll,
    )
    .map_err(|e| vec![format!("invalid target: {e}")])
}

/// Builds the module (`lower_program` + verify), runs it through `lower_
/// to_llvm`, and writes it as an object file at `object_path`, compiled for
/// the target `options` ask for -- the shared implementation behind
/// `--emit-object`/`--emit-bindings`/`--emit-exe`/`cleave-build`.
fn emit_object(
    program: &Program,
    cps_program: &CpsProgram,
    object_path: &Path,
    options: &CodegenOptions,
    sources: &SourceMap,
) -> Result<(), Vec<String>> {
    crate::mlir_lower::set_gen_file_table(sources.path_table());
    let dialect_registry = DialectRegistry::new();
    register_all_dialects(&dialect_registry);
    let context = Context::new();
    context.append_dialect_registry(&dialect_registry);
    context.load_all_available_dialects();
    // Needed for the OpenMP parallelization stage (`lower_to_llvm`) -- `omp.
    // parallel`/`omp.wsloop`/`omp.loop_nest` deliberately survive every
    // MLIR-level pass and are only ever turned into real LLVM IR (`__kmpc_
    // fork_call`/...) at final translation time, via a *separate*
    // registration (`mlirRegisterAllLLVMTranslations`, wrapped here) from
    // `register_all_dialects` above. Registered unconditionally, even when
    // `options.openmp` is `false` -- cheap, and simpler than threading the
    // option one layer further just to skip it.
    cleave_mlir::utility::register_all_llvm_translations(&context);

    let mlir_types = collect_mlir_types(program);
    let struct_schemas = collect_struct_schemas(program);
    let start = std::time::Instant::now();
    let mut module = lower_program(&context, cps_program, &mlir_types, struct_schemas);
    report_stage("MLIR lowering (mlir_lower)", start);
    if !module.as_operation().verify() {
        return Err(vec![
            "generated MLIR module failed verification".to_string(),
        ]);
    }

    let start = std::time::Instant::now();
    lower_to_llvm(&context, &mut module, options)?;
    report_stage("lower_to_llvm, in all", start);
    // `CLEAVE_DUMP_LLVM_DIALECT=<path>`: the module as it is handed to LLVM
    // (`mlir-translate --mlir-to-llvmir`, then `opt`/`llc -time-passes`, to
    // see where LLVM's own share of a compile goes).
    if let Ok(path) = std::env::var("CLEAVE_DUMP_LLVM_DIALECT") {
        std::fs::write(&path, module.as_operation().to_string())
            .unwrap_or_else(|e| eprintln!("CLEAVE_DUMP_LLVM_DIALECT: failed to write {path}: {e}"));
    }
    let start = std::time::Instant::now();

    let Some(object_path_str) = object_path.to_str() else {
        return Err(vec![format!(
            "object path {object_path:?} is not valid UTF-8"
        )]);
    };
    let target = target(options)?;
    // SAFETY: `module` is a valid module, owned here.
    unsafe { cleave_mlir::emit_object(module.to_raw(), &target, object_path_str) }
        .map_err(|e| vec![format!("failed to emit {object_path_str}: {e}")])?;
    report_stage("LLVM: translation, optimization, code generation", start);
    Ok(())
}

/// The fixed internal symbol cleave's own `fn main()` gets renamed to when
/// compiling a standalone executable (`emit_exe` below) -- never seen by a
/// cleave program's own author, purely an implementation detail of the
/// generated Rust shim's own linking. See `emit_exe`'s own doc comment for
/// why a rename is needed at all.
const EXE_ENTRY_SYMBOL: &str = "__cleave_program_main";

/// Compiles `program` to a standalone executable at `exe_path`: `emit_object`,
/// then a link driven by `rustc` against a generated Rust `main` that calls
/// cleave's (`fn main() { std::process::exit(unsafe { EXE_ENTRY_SYMBOL() }) }`
/// for an `i32` result). cleave's `main` is renamed `EXE_ENTRY_SYMBOL`, as an
/// `export fn` names its symbol, so it doesn't collide with the Rust one.
/// `rustc` rather than a bare linker: `cleave-rt` links Rust's `std`, whose
/// system libraries only `rustc` knows to pass (a bare `clang` link left ~30
/// symbols unresolved), and building cleave already needs a Rust toolchain.
pub fn emit_exe(
    program: &Program,
    registry: &Registry,
    sources: &SourceMap,
    exe_path: &Path,
    options: &CodegenOptions,
) -> Result<(), Vec<String>> {
    check_type_errors(program, registry).map_err(|errs| render_all(&errs, sources))?;
    let mut cps_program = build_optimized_cps(program, registry, Some(sources))?;

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

    emit_object(program, &cps_program, &object_path, options, sources)?;

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

    let runtime_lib = cleave_rt_library()?;
    let mut cmd = std::process::Command::new("rustc");
    cmd.arg(&shim_path)
        .arg("-o")
        .arg(exe_path)
        .arg("-C")
        .arg(format!("link-arg={}", object_path.display()))
        .arg("-C")
        .arg(format!("link-arg={}", runtime_lib.display()));
    if options.openmp || (options.tasks && crate::cps::uses_spawn(&cps_program)) {
        // `__kmpc_*` references, from `--openmp`'s parallel loops or `spawn`'s
        // tasks, resolved against the toolchain's `libomp`.
        let prefix = crate::toolchain::llvm_prefix().map_err(|e| vec![format!("{e} to link libomp")])?;
        cmd.arg("-L")
            .arg(crate::toolchain::libomp_link_dir(&prefix))
            .arg("-l")
            .arg(crate::toolchain::LIBOMP_LINK_NAME);
        if !cfg!(windows) {
            // Found at run time where it was linked from.
            cmd.arg("-C").arg(format!("link-arg=-Wl,-rpath,{}", crate::toolchain::libomp_link_dir(&prefix)));
        }
    }
    let status = cmd.status();
    let _ = std::fs::remove_dir_all(&work_dir);
    match status {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => Err(vec![format!(
            "rustc failed while linking the final executable (exit status: {s})"
        )]),
        Err(e) => Err(vec![format!("failed to run `rustc` (is it on PATH?): {e}")]),
    }
}

/// The `cleave-rt` static library to link a standalone executable with: the
/// most recently built of `cleave_rt.lib`/`libcleave_rt.a` beside the running
/// `cleave` (or one level up, for a test binary in `deps/`) and of Cargo's
/// hash-suffixed copies in `deps/`. Building `cleave` rebuilds `cleave-rt` as
/// a dependency into `deps/` only; the unsuffixed copy beside it is refreshed
/// only by building `cleave-rt` itself, so it goes stale as the runtime gains
/// symbols (`cleave_parallel_threads`, unresolved at link time, found with
/// `--emit-exe`). The newest one is the one built from the current sources.
/// A `cleave` installed with no runtime library near it finds none.
fn cleave_rt_library() -> Result<PathBuf, Vec<String>> {
    let exe = std::env::current_exe().map_err(|e| {
        vec![format!(
            "failed to locate the running cleave executable: {e}"
        )]
    })?;
    let is_runtime = |name: &str| {
        name == "cleave_rt.lib"
            || name == "libcleave_rt.a"
            || (name.starts_with("cleave_rt-") && name.ends_with(".lib"))
            || (name.starts_with("libcleave_rt-") && name.ends_with(".a"))
    };
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    for dir in exe.ancestors().skip(1).take(2) {
        for dir in [dir.to_path_buf(), dir.join("deps")] {
            let Ok(entries) = std::fs::read_dir(&dir) else { continue };
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if !is_runtime(&name) {
                    continue;
                }
                let Ok(modified) = entry.metadata().and_then(|m| m.modified()) else { continue };
                if newest.as_ref().is_none_or(|(t, _)| modified > *t) {
                    newest = Some((modified, entry.path()));
                }
            }
        }
    }
    newest.map(|(_, path)| path).ok_or_else(|| {
        vec![format!(
            "no cleave-rt static library (cleave_rt.lib / libcleave_rt.a) near {}",
            exe.display()
        )]
    })
}
