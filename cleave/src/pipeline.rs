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
use cleave_mlir_shim::mlir::Context;
use cleave_mlir_shim::mlir::dialect::DialectRegistry;
use cleave_mlir_shim::mlir::ir::attribute::Attribute;
use cleave_mlir_shim::mlir::ir::operation::{OperationBuilder};
use cleave_mlir_shim::mlir::ir::{Identifier, Location, Module};
use cleave_mlir_shim::mlir::utility::register_all_dialects;
use std::path::{Path, PathBuf};

/// The hardware target a program is being compiled for -- `doc/hld.md`'s
/// own introduction already commits the project to two reference targets
/// (CPU, and Vulkan Compute via the `spirv` dialect) as a stated project
/// value, framed there as "hardware targets fit the same plugin shape as
/// math algebras (name + cost function + MLIR lowering)". This enum is the
/// first concrete seam for that commitment -- deliberately a single variant
/// today (only `Cpu` is real; every pass in `lower_to_llvm` below assumes
/// it), not a speculative attempt at Vulkan support, which would need its
/// own real `spirv`-dialect lowering (a separate, much larger undertaking)
/// before a second variant could mean anything. `CodegenOptions::backend`
/// exists now specifically so the CLI/`Build` surface doesn't need a
/// breaking change the day a second variant *does* land.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
}

/// Real codegen configuration, threaded through every pipeline entry point
/// that lowers to `llvm`-dialect MLIR (`lower_to_llvm` below) or constructs
/// an `ExecutionEngine` (`emit_object`, and `main.rs`'s own `--run` block).
/// `Default` reproduces this project's own previously-hardcoded behavior
/// exactly (`opt_level: 2`, `openmp: true`, `target_cpu`/`target_features:
/// None` -- i.e. don't stamp anything, leaving whatever `ExecutionEngine`
/// already defaulted to before this struct existed, untouched) -- adding
/// this struct is additive, not a behavior change, for any caller that
/// doesn't ask for something different.
#[derive(Debug, Clone)]
pub struct CodegenOptions {
    /// LLVM's optimization level, `0`-`3` (`target`).
    pub opt_level: u8,
    /// Whether `lower_to_llvm` applies the OpenMP parallelization stage
    /// (`--affine-parallelize`/`--convert-scf-to-openmp`/`--convert-openmp-
    /// to-llvm`) at all. A single, universal default (`true`) since
    /// `main.rs`'s own `real_main`'s "`ExecutionPath`/`openmp`/`dump-X` are
    /// three independent, additive axes, not one flag whose default should
    /// depend on another" design conversation: JIT (`--run`/`--dump-mlir-
    /// lowered`) used to default this `false` at its own call sites, on the
    /// theory that `cleave-rt`'s own arena allocator globals (`ARENA_BASE`/
    /// `ARENA_CURSOR`/`REGION_DEPTH`, `cleave-rt/src/lib.rs` -- deliberately
    /// single-threaded `Ordering::Relaxed` atomics, sound only because every
    /// OpenMP-parallelized region is provably allocator-free) might behave
    /// differently under JIT than the already-stress-tested AOT path --
    /// found, on inspection, to have no real basis: that soundness argument
    /// is a property of the *generated code itself*, identical either way,
    /// never of which engine (JIT vs AOT) happens to execute it. `main.rs`'s
    /// own `resolve_codegen_options`/`Registry::build_with_defines` calls
    /// resolve this exact same `true` default once, universally, matching
    /// the real, measured 6.6x speedup this mechanism delivers on the AOT
    /// path that first proved it out.
    pub openmp: bool,
    /// The CPU code is generated for (`target`): an LLVM processor name, or
    /// `native` for the host's with every feature it has. `None`: the host's.
    pub target_cpu: Option<String>,
    /// Features added to or removed from the CPU's (`+avx2,-avx512f`).
    pub target_features: Option<String>,
    /// See `Backend`'s own doc comment -- `Cpu` is the only real value
    /// today; every stage `lower_to_llvm` runs assumes it.
    pub backend: Backend,
    /// Gates `lower_to_llvm`'s own MLIR-level inliner pass (`pass::transform
    /// ::create_inliner()`). Real, established default-on optimization --
    /// `false` was previously only reachable via `CLEAVE_NO_INLINE=1`, kept
    /// as a real, named opt-out (not removed) specifically for isolating a
    /// single function's own disassembly during profiling, this session's
    /// own established use for it (`doc/backlog.md`'s matmul-IPC
    /// investigation).
    pub inline: bool,
    /// Gates `alias_analysis`'s own affine-struct pool-allocation strategy
    /// (`mlir_lower.rs::lower_program`). **On by default** -- landed,
    /// measured, re-verified on both real kernels (`doc/plan-affine-
    /// ownership.md` §11-§14). `false` is the escape hatch for a structural
    /// shape neither real kernel nor the test suite happens to exercise --
    /// never remove this fallback casually (`lower_program`'s own doc
    /// comment has the full reasoning).
    pub affine_structs: bool,
    /// Gates `mlir_lower.rs::build_di_subprograms`/`set_gen_subprogram`
    /// (per-function `#llvm.di_subprogram` attributes, fused into every
    /// op's location via `gen_loc`) and `lower_to_llvm`'s own `!llvm.module
    /// .flags` emission of `CodeView`/`Debug Info Version`. **On by
    /// default**, matching the previously-unconditional behavior (there was
    /// no gate at all before this field existed). `false` skips both:
    /// `gen_loc` already falls back to a bare, subprogram-less `Location`
    /// whenever `GEN_SUBPROGRAM` is left at its default `None` (its own doc
    /// comment), so simply never calling `set_gen_subprogram` is sufficient
    /// -- no separate "strip DI" pass needed. A real, named opt-out for
    /// profiling/disassembly work that doesn't want `DISubprogram`/`!dbg`
    /// noise in dumped IR or symbolized profiles.
    pub debug_info: bool,
    /// LLVM's own loop unrolling, in the optimization pipeline the execution
    /// engine runs (`cleave-mlir-shim`'s `makeTransformer`). cleave's loops
    /// reach LLVM already tiled, vectorized and unrolled where it pays (the
    /// matmul schedule); LLVM unrolling them again was 55% of
    /// `opt -O2` on nanoLM's transformer kernel, most of a 3-minute compile.
    /// `true` keeps the standard pipeline.
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
    /// (`cleave_mlir_shim::limit_inlining`). Small functions, the elementwise
    /// operations fusion needs, stay inlined; a large one gains nothing from
    /// it (a call costs nothing next to thousands of operations) and costs a
    /// lot: LLVM's passes are superlinear in a function's size. nanoLM v2's
    /// `train_gpt` grew from 225 operations to ~80,000 by inlining the model's
    /// restore, initialization, clipping and checkpointing, walked leaf by
    /// leaf, and LICM alone then took 124 s of a 190 s compile.
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
pub unsafe fn register_cleave_rt_symbols(engine: &cleave_mlir_shim::ExecutionEngine) {
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
            "print_dynarray_bytes",
            cleave_rt::print_dynarray_bytes as *mut (),
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
        // `stdlib/blas/blas.cleave`'s own `raw_sgemm` extern -- lazily loads
        // `openblas.dll` on first real call (`cleave_rt::blas_dynload`), so
        // registering it here unconditionally costs nothing for a program
        // that never calls into `blas`.
        engine.register_symbol("cleave_blas_sgemm", cleave_rt::cleave_blas_sgemm as *mut ());
    }
}

/// Registers, once per process, every pass a textual pipeline may name:
/// MLIR's and cleave's (`cleave_mlir_shim::register_passes`). MLIR's pass
/// registry is a global, unsynchronized table: registering at each use wrote
/// to it while another thread compiling at the same time (the test harnesses
/// run compilations in parallel) read it, an intermittent
/// `STATUS_ACCESS_VIOLATION`. The shim registers inside a function-local
/// static's initialization, which other threads wait for, and runs it before
/// parsing any pipeline.
pub fn register_passes() {
    cleave_mlir_shim::register_passes();
}

/// Rows of a BLAS product computed per tile when it is fused with its
/// elementwise consumer (`cleave-blas-tile-and-fuse`): a tile of a product
/// 1024 wide is 512 KB, within a core's 1 MB L2 alongside its slice of `A`;
/// fewer rows would call `sgemm` more often, each call packing `B` again.
const BLAS_TILE_ROWS: i64 = 128;

/// The matmul tile/vectorize schedule, compiled into the binary and loaded
/// into the context's transform library from memory
/// (`cleave_mlir_shim::load_transform_library`).
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
    let result = unsafe { cleave_mlir_shim::run_pipeline(module.to_raw(), pipeline, time_stages()) };
    report_stage(what, start);
    result.map_err(|e| {
        vec![if e.is_empty() {
            format!("MLIR-to-LLVM lowering pass failed ({what})")
        } else {
            format!("MLIR-to-LLVM lowering: invalid pipeline for {what}: {e}")
        }]
    })
}

pub fn lower_to_llvm<'c>(
    context: &'c Context,
    module: &mut Module<'c>,
    options: &CodegenOptions,
) -> Result<(), Vec<String>> {
    let Backend::Cpu = options.backend;
    register_passes();

    // Inline, then fuse elementwise tensor ops, *before* bufferization --
    // found directly with VTune against a real training run
    // (`examples/mnist-interop`): 75% of wall time was inside
    // `Optimizer::step<Sgd,Network>`, and its own self time was near zero --
    // almost everything was in unlisted callees, i.e. allocation/copy/free
    // traffic, not compute. Root cause, confirmed by reading the raw
    // (pre-pipeline) MLIR directly: an SGD update (`stdlib/optim/optim.
    // cleave`'s own `Ring::sub(model, Scale::scale(grad, opt.lr))`) lowers
    // to *two separate* `func.func`s (`Ring::sub<...>`, `Scale::scale<...>`)
    // called via ordinary `call` ops -- cleave's own CPS-level inlining
    // (`doc/hld.md`, "Memory management") only ever collapses same-file,
    // non-algebra-dispatch calls; a cross-algebra dispatch like this one is
    // resolved to a concrete monomorphized function but stays a real CPS
    // `App`, never inlined. One-Shot Bufferize therefore materializes *two*
    // scratch buffers (one per function's own tensor result) instead of
    // one, every time this pattern appears -- and it's the dominant shape
    // in `forward`/`loss`/`Optimizer::step` alike, not a one-off.
    //
    // `--inline` (MLIR's own generic, dialect-agnostic inliner, `Transforms`
    // library) closes exactly this gap: flattening these calls into their
    // call sites puts producer and consumer in the same function body,
    // where `--linalg-fuse-elementwise-ops` (after `--convert-elementwise-
    // to-linalg` turns the now-inlined `arith.subf`/`arith.mulf` chain into
    // `linalg.generic` ops) can fuse them into *one* kernel writing *one*
    // buffer, instead of two kernels chained through an intermediate one --
    // confirmed directly against this exact toolchain (`mlir-opt --inline
    // --convert-elementwise-to-linalg --linalg-fuse-elementwise-ops` on a
    // minimal SGD-update probe): two `linalg.generic` ops with a
    // materialized intermediate collapse into one `linalg.generic` whose
    // body computes `mul` then `sub` directly, no intermediate at all.
    //
    // Verified safe at real scale before wiring in, not assumed: run
    // end-to-end (inline -> elementwise-to-linalg -> fuse -> one-shot-
    // bufferize -> buffer-deallocation -> linalg-to-loops -> to-llvm) on
    // `examples/mnist-interop`'s own real kernel -- zero verification
    // failures at any stage, `train_and_evaluate` itself (the `export fn`
    // Rust calls by symbol name) survives inlining intact (never a call
    // target of its own, so never itself inlined away), and MLIR-level
    // compile time stays sub-second (static IR size grows roughly 2x from
    // call-site duplication, same shape as any inliner, not the exponential
    // blowup a previous compile-time investigation hit and fixed
    // elsewhere -- `doc/backlog.md`'s own "real root cause of the 738s").
    // `CodegenOptions::inline = false` (`--no-inline` on the CLI) -- a
    // diagnostic-only knob, raised directly by the user to inspect the
    // generated code with real function boundaries kept
    // intact (`net_grad`, `matmul`, ... each stay their own `llvm.func`
    // instead of being flattened into `train_and_evaluate`), instead of
    // digging through `S_INLINESITE` records inside one giant disassembled
    // blob. Skips *only* the inliner itself -- `--convert-elementwise-to-
    // linalg`/`--linalg-fuse-elementwise-ops` still run (harmless without
    // inlining: there is nothing cross-function left for them to fuse, per
    // this pass's own comment above), and every later stage (the matmul
    // tiling/vectorization schedule below, `--eliminate-empty-tensors`)
    // degrades safely -- each one's own preconditions simply aren't met as
    // often, falling back to its own always-correct path (a struct field
    // store then costs a copy One-Shot inserts itself). Not meant for a
    // real perf build -- disabling inlining reopens exactly the double-
    // scratch-buffer cost this same pass's own comment above measured and
    // fixed.
    // Always, `--no-inline` included: it also marks the functions LLVM's own
    // inliner must leave alone (`apply_no_inline`, end of this function).
    run_stage(
        module,
        "inline/elementwise-to-linalg/fuse",
        &format!(
            "builtin.module(cleave-limit-inlining{{threshold={}}},{}convert-elementwise-to-linalg,linalg-fuse-elementwise-ops)",
            options.inline_threshold,
            if options.inline { "inline," } else { "" },
        ),
    )?;

    // TEMP EXPERIMENT (`doc/backlog.md`, register-residency + FMA + real
    // OpenMP parallelism, all three, on the matmul specifically): tile the
    // still-tensor-typed `linalg.matmul` two ways -- `tile_using_forall` on
    // the outer (`i`, genuinely parallel) dimension, `tile_using_for` on `j`
    // (vector-lane width) and `k` (the reduction) -- then vectorize with
    // `create_named_contraction` so the result is a real `vector.contract`,
    // not the generic transpose/extract shape a plain `linalg.generic`
    // (or `vectorize` without that attribute) would give -- confirmed
    // directly, both by MLIR's own source (`Vectorization.cpp`'s
    // `vectorizeAsLinalgContraction`: "Generic op is ignored as not every
    // arbitrary contraction body can be expressed by a vector.contract")
    // and by MLIR's own test suite (`contraction-interface.mlir`'s
    // `@negative_generic`, `CHECK-NOT: vector.contract`, the *exact*
    // indexing-map/body shape this project used to hand-build). `apply_
    // patterns.vector.lower_contraction{outerproduct}` then turns that
    // `vector.contract` into a `vector.outerproduct` chain -- confirmed via
    // real disassembly on an isolated probe: `vfmadd132ps`/`vbroadcastss`,
    // the textbook GEMM microkernel, not the `vpermt2ps`/`vinsertps`
    // shuffle pile the generic path produces. `--loop-invariant-subset-
    // hoisting` right after (safe here specifically because the IR is
    // still tensor-typed, not memref -- see the backlog's own long
    // derivation) makes the accumulator itself genuinely register-resident
    // across the `k` reduction, rather than round-tripping through memory
    // on every step. Schedule read from an external file (`cleave/mlir/
    // matmul_vectorize.transform.mlir`, bundled alongside this crate's own
    // source -- not `bench/mnist-pytorch/`, which is the PyTorch-comparison
    // benchmark's own directory, not a natural home for a real pipeline
    // dependency) rather than an embedded `transform.named_sequence`,
    // sidestepping a leftover-schedule translation error that only ever
    // affected standalone hand-written `.mlir` probes. Embedded in the
    // binary (`matmul_schedule_path`): it was read at run time from this
    // crate's source directory, so a `cleave` binary moved away from its
    // checkout couldn't compile a matmul.
    // The BLAS tier (`stdlib/linalg/matrix.cleave`, above `BLAS_MIN_WORK`):
    // each product's elementwise consumer tiled by rows with the product
    // fused in, then every marked product a `sgemm` call, before the
    // schedule vectorizes the `linalg` tier's.
    // SAFETY: `context` is the module's, used by this thread only.
    if !unsafe {
        cleave_mlir_shim::load_transform_library(context.to_raw(), MATMUL_SCHEDULE, "matmul_vectorize.transform.mlir")
    } {
        return Err(vec!["failed to load the matmul schedule".to_string()]);
    }
    run_stage(
        module,
        "BLAS tier, transform-dialect tile/vectorize",
        &format!(
            "builtin.module(cleave-blas-tile-and-fuse{{rows={BLAS_TILE_ROWS}}},cleave-lower-blas-matmuls,transform-interpreter{{entry-point=__transform_main}})"
        ),
    )?;

    run_stage(module, "loop-invariant-subset-hoisting", "builtin.module(loop-invariant-subset-hoisting)")?;

    // An elementwise op writing a fresh tensor while one of its operands, a
    // local result, dies there: it writes into that operand
    // (`cleave_mlir_shim::reuse_dying_inputs`), one buffer fewer alive.
    run_stage(module, "reuse dying inputs", "builtin.module(cleave-reuse-dying-inputs)")?;

    // `CLEAVE_DUMP_PRE_BUFFERIZE=<path>` -- the still-tensor-typed IR, as
    // `mlir_lower.rs` emitted it, immediately before one-shot-bufferize
    // runs. Pair it with `CLEAVE_DUMP_POST_DEALLOC` below: bufferization's
    // own equivalence/in-place-reuse decisions leave *no trace* after the
    // fact, so if two tensor-level values were given one buffer, diffing
    // the two sides of this boundary is the only way to see it. This must
    // stay *above* the pass manager below -- placed after it, it captures
    // memref-typed IR with every `bufferization.to_tensor` already folded
    // away, which silently answers a different question than the one the
    // name promises.
    if let Ok(path) = std::env::var("CLEAVE_DUMP_PRE_BUFFERIZE") {
        std::fs::write(&path, module.as_operation().to_string())
            .unwrap_or_else(|e| eprintln!("CLEAVE_DUMP_PRE_BUFFERIZE: failed to write {path}: {e}"));
    }

    // `eliminate-empty-tensors` first: it is what lets a struct field's own
    // `materialize_in_destination` (`mlir_lower.rs::build_tensor_descriptor_
    // value`) turn the tensor's producer into a direct write into the field's
    // buffer, instead of a scratch buffer plus a copy. The self-copy it
    // leaves behind (`memref.copy %b, %b`) is folded by the later
    // `--canonicalize` stage.
    //
    // `allow-return-allocs-from-loops`: a loop may carry a tensor whose new
    // value is a fresh buffer (`x = sgemm(w, x)`, any call returning a new
    // tensor) rather than an in-place update of the one it received. Without
    // it, One-Shot Bufferize rejects the loop outright ("Yield operand is not
    // equivalent to the corresponding iter bbArg"); with it, the loop's buffer
    // changes from one iteration to the next, and the ownership-based
    // deallocation below frees the one each iteration leaves behind.
    run_stage(
        module,
        "one-shot-bufferize",
        "builtin.module(eliminate-empty-tensors,one-shot-bufferize{bufferize-function-boundaries=true function-boundary-type-conversion=identity-layout-map allow-return-allocs-from-loops=true})",
    )?;

    // A block of a buffer given to an extern as its destination (`sgemm`
    // into `Slice::slice(out, ...)`, put back by `Slice::update`): written
    // in place instead of copied out and back in
    // (`cleave_mlir_shim::elide_block_copies`).
    run_stage(module, "elide block copies", "builtin.module(cleave-elide-block-copies)")?;

    // `--scf-forall-to-parallel` -- the *other* half of the real parallelism
    // fix, alongside `tile_using_forall` above: an `scf.forall` produced
    // pre-bufferize (still tensor-typed) carries `shared_outs`/`tensor.
    // parallel_insert_slice`, a shape this pass can't handle at all
    // (confirmed directly: silent exit 1, no diagnostic, on the tensor
    // form) -- `scf.parallel` has no equivalent of a shared, combined
    // result. Bufferized, the same `forall` has no results left at all
    // (direct in-place memory writes only), exactly the shape this pass
    // expects -- must run *here*, right after bufferize, not before.
    // Produces a real `scf.parallel`, which `--convert-scf-to-openmp`
    // (below, already the existing mechanism) picks up exactly like the
    // *old* path's `affine.parallel`-derived one.
    run_stage(module, "scf-forall-to-parallel", "builtin.module(scf-forall-to-parallel)")?;

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
    // A function's tensor results become out-parameters the caller allocates
    // (`hoist-static-allocs`: a result that was the callee's own fresh
    // `memref.alloc` is written straight into the caller's buffer, no copy in
    // the callee). Before the deallocation passes, which then see plain
    // caller-owned buffers. `cleave-forward-out-param-copies`
    // then hands the call the final destination directly when the result was
    // only copied there (a struct field, a tuple element).
    // First, so a function returning a filled array (`Tensor(data: buf)`)
    // returns that array's own allocation, which the out-params pass then
    // hoists to the caller.
    run_stage(module, "forward dead source copies", "builtin.module(cleave-forward-dead-source-copies)")?;
    run_stage(
        module,
        "buffer-results-to-out-params",
        "builtin.module(buffer-results-to-out-params{hoist-static-allocs=true})",
    )?;
    run_stage(module, "forward out-param copies", "builtin.module(cleave-forward-out-param-copies)")?;
    // A result written into a fresh buffer of the out-parameter's type and
    // then copied into it: written into the out-parameter directly
    // (`cleave_mlir_shim::forward_copies_to_destinations`).
    run_stage(module, "forward copies to destinations", "builtin.module(cleave-forward-copies-to-destinations)")?;
    if let Ok(path) = std::env::var("CLEAVE_DUMP_POST_OUT_PARAMS") {
        std::fs::write(&path, module.as_operation().to_string())
            .unwrap_or_else(|e| eprintln!("CLEAVE_DUMP_POST_OUT_PARAMS: failed to write {path}: {e}"));
    }

    // A loop yielding back the buffer it carries returns the buffer it was
    // given: said before the deallocation, whose alias analysis can't see
    // through a loop (`cleave_mlir_shim::fold_passthrough_iter_args`).
    run_stage(module, "fold passthrough iter args", "builtin.module(cleave-fold-passthrough-iter-args)")?;
    run_stage(
        module,
        "buffer-deallocation",
        "builtin.module(ownership-based-buffer-deallocation,buffer-deallocation-simplification,bufferization-lower-deallocations)",
    )?;
    // Adopted tensors (`PrimOp::Adopt`): the deallocation above took each for
    // a fresh buffer; it is a retain of the same one.
    run_stage(module, "lower adoptions", "builtin.module(cleave-lower-adoptions)")?;
    // Each buffer freed right after its last use, not at the end of its block
    // (`cleave_mlir_shim::dealloc_at_last_use`). Before the spawns are
    // lowered: a task's wait marker is a use of the buffers the task reads.
    run_stage(module, "dealloc at last use", "builtin.module(cleave-dealloc-at-last-use)")?;
    run_stage(module, "bufferization-to-memref", "builtin.module(convert-bufferization-to-memref)")?;

    // `CLEAVE_DUMP_POST_DEALLOC=<path>` -- the other side of the boundary
    // `CLEAVE_DUMP_PRE_BUFFERIZE` above opens: the module right after
    // `--ownership-based-buffer-deallocation`/`--lower-deallocations`,
    // still memref-level, so the real conditional-ownership `scf.if`/
    // `memref.dealloc` shape is visible before `--convert-to-llvm` turns it
    // into opaque `llvm.call @free`. This is the dump that made the
    // alloc-backed-with-a-dealloc population countable at all
    // (`doc/plan-region-arena.md` §8.3).
    // `spawn`'s markers become OpenMP tasks, now that buffers and their
    // deallocations are placed (`cleave_mlir_shim::lower_spawns`).
    run_stage(module, "spawn tasks", &format!("builtin.module(cleave-lower-spawns{{tasks={}}})", options.tasks))?;

    if let Ok(path) = std::env::var("CLEAVE_DUMP_POST_DEALLOC") {
        std::fs::write(&path, module.as_operation().to_string())
            .unwrap_or_else(|e| eprintln!("CLEAVE_DUMP_POST_DEALLOC: failed to write {path}: {e}"));
    }

    // `--symbol-dce`, right after `--inline` (above) made every inlined
    // call site's own original callee declaration dead weight -- only
    // actually removable now that `lower_program` (`mlir_lower.rs`) marks
    // every non-`main`/non-export function `sym_visibility = "private"`;
    // `--symbol-dce` only ever deletes a symbol that's *both* unreferenced
    // and private. Not just tidiness: leaving these dead, now-unreferenced
    // declarations in place is exactly what made the structured-
    // vectorization stage below hard-fail (see its own doc comment).
    run_stage(module, "symbol-dce", "builtin.module(symbol-dce)")?;

    // TEMP EXPERIMENT, continued: the transform-dialect-vectorized matmul
    // body is already fully vector-typed at this point (real `vector.
    // outerproduct` chains, from `apply_patterns.vector.lower_contraction`
    // above) -- lower it the rest of the way here, before the *old*
    // `linalg`-to-`affine`-to-`vector` path below ever runs (that path
    // never touches this op at all any more; it still runs for whatever
    // wasn't caught by the transform match, e.g. genuinely elementwise
    // ops). `--canonicalize` first, then `--lower-vector-multi-reduction`
    // for whatever reduction shape (if any) remains.
    // `lower-vector-multi-reduction` alone: a `canonicalize` was added to this
    // stage's pass manager before the pipeline was parsed into it, which
    // replaced it, so it never ran. Kept as it ran, the output unchanged.
    run_stage(
        module,
        "lower-vector-multi-reduction",
        "builtin.module(func.func(lower-vector-multi-reduction))",
    )?;

    // `--cse`, then `cleave-eliminate-self-copies` -- see that
    // module's own doc comment for the full story (a real, VTune-confirmed
    // cost: a genuine `memref.copy %x, %x` no-op, left behind by One-Shot
    // Bufferize's own materialization of the matmul-tiling stage's own
    // `scf.forall`/`tensor.parallel_insert_slice` write-back, above, that
    // neither `--canonicalize` nor `--cse` alone folds away). `--cse` must
    // run *first*, right here, specifically because two structurally
    // identical `memref.subview`s (the real, common shape this pattern
    // takes before CSE runs) are two *different* SSA values until CSE
    // merges them -- confirmed directly, on an isolated probe, that the
    // self-copy this pass targets doesn't even exist in the IR at all until
    // CSE has already run once.
    run_stage(module, "cse", "builtin.module(cse)")?;
    run_stage(module, "eliminate self copies", "builtin.module(cleave-eliminate-self-copies)")?;
    // Partial-tile write-backs, before the affine pass rejects them
    // (`cleave-lower-dynamic-copies`).
    run_stage(module, "lower dynamic copies", "builtin.module(cleave-lower-dynamic-copies)")?;

    // `--expand-strided-metadata` turns `memref.subview`'s own dynamic
    // offset/stride metadata (from the tiling above) into plain arithmetic
    // -- without it, `--finalize-memref-to-llvm` further below has nothing
    // it can convert. It itself emits `affine.apply`, hence `--lower-
    // affine` immediately after. A final `--canonicalize` cleans up the
    // arithmetic both passes leave behind.
    run_stage(
        module,
        "expand-strided-metadata/lower-affine",
        "builtin.module(canonicalize,expand-strided-metadata,lower-affine,canonicalize)",
    )?;

    // `--convert-vector-to-scf` -- the piece `--canonicalize` above never
    // touched: handles the genuinely N-D `vector<1x16x16xf32>` *load*
    // shapes (`vector.transfer_read`) directly, ahead of the ordinary
    // `--convert-vector-to-llvm` further below in the final lowering
    // stage.
    run_stage(module, "vector-to-scf", "builtin.module(convert-vector-to-scf)")?;

    // `--convert-linalg-to-affine-loops`, not the ordinary `-to-loops`
    // (`scf.for`) -- see the structured-vectorization stage right below for
    // why: `--affine-super-vectorize` only operates on `affine.for`.
    //
    // `--affine-fold-memref-alias-ops` then: a loop over a subview (a tile of a
    // larger buffer, `cleave_mlir_shim::blas_tile_and_fuse`'s consumer
    // writing its rows of the result; a `Slice::slice`) reads and writes the
    // buffer it is a view of instead, at offset indices, so that its memrefs
    // have the plain layout `--affine-super-vectorize` needs (it leaves a
    // loop over any other scalar).
    run_stage(module, "linalg-to-affine-loops", "builtin.module(convert-linalg-to-affine-loops)")?;
    run_stage(module, "affine-fold-memref-alias-ops", "builtin.module(func.func(affine-fold-memref-alias-ops))")?;

    // OpenMP parallelization -- marks every linalg-derived loop nest's own
    // *outermost* dimension `affine.parallel` when it's genuinely safe (no
    // loop-carried dependence), one thread's worth of work per outer-loop
    // iteration. Scoped deliberately narrow: only linalg-derived loops (the
    // ones this pass ever sees, `affine.for` from `--convert-linalg-to-
    // affine-loops` just above) -- never cleave's own hand-written `for`
    // loops (`Sum::sum`'s own manual-counter loop, the training loop, ...),
    // which lower via a completely different path (`mlir_lower.rs::lower_
    // loop`, `scf.while`, never `affine.for`) this pass never touches. This
    // scoping is *why* it's sound without any change to `cleave-rt`: a
    // linalg-derived kernel body (matmul's `mulf`/`addf`/`select`, an
    // elementwise op, ...) is pure arithmetic over already-materialized
    // memref slices -- it never calls `cleave_alloc_rc`/`cleave_alloc_
    // local`/`cleave_retain`/`cleave_release`, so running several of its
    // outer-loop iterations on different OS threads at once never touches
    // `cleave-rt`'s own arena globals (`ARENA_BASE`/`ARENA_CURSOR`/`REGION_
    // DEPTH`, deliberately single-threaded `Ordering::Relaxed` atomics --
    // see `cleave-rt/src/lib.rs`'s own doc comments). Parallelizing
    // anything *outside* this scope (the training loop itself, say) would
    // need that assumption revisited first -- not attempted here.
    //
    // `--affine-parallelize{max-nested=1}` must run *before* `--affine-
    // super-vectorize` below, not after -- found by direct `mlir-opt`
    // testing, not assumed: `affine-parallelize`'s own dependence analysis
    // only recognizes plain `affine.load`/`affine.store`, not the `vector.
    // transfer_read`/`write` pairs vectorization introduces, so run second
    // it silently parallelizes nothing at all (confirmed: identical output,
    // zero `affine.parallel` produced). `max-nested=1` deliberately caps
    // this to the *outermost* dimension only -- matmul's own loop nest is
    // `i (parallel, dim0) > j (parallel, dim1, the one `affine-super-
    // vectorize` packs into `vector<16xf32>` lanes) > k (reduction, the
    // accumulation)`; parallelizing `j` too would fight over the same
    // vectorization the FMA-fusion work earlier in this pipeline already
    // depends on, and `k` is never a parallelize candidate at all (`--
    // parallel-reductions` defaults `false`, correctly -- accumulation is
    // genuinely loop-carried). One thread per row of the output is coarse-
    // grained but real, sound, and doesn't disturb the SIMD lane packing.
    //
    // Gated on `options.openmp` -- see `CodegenOptions::openmp`'s own doc
    // comment for the default split (AOT `true`, JIT `false`) and why.
    if options.openmp {
        run_stage(module, "affine-parallelize", "builtin.module(func.func(affine-parallelize{max-nested=1}))")?;
    }

    // FMA fusion -- found by disassembling a real emitted object file
    // (`kernel.o`, `examples/mnist-interop`): the matmul accumulation loop
    // (`linalg.matmul`'s own lowering, right above, `acc = acc + a[i,k]*b
    // [k,j]`) already vectorizes to AVX-512 (`%zmm`) on this host by
    // default, but *never* to a fused multiply-add (`vfmaddXXXps`) --
    // always a separate `vmulps`+`vaddps` pair, even though the CPU
    // supports it. Root cause, confirmed directly: LLVM will not contract
    // a separate `fmul`+`fadd` into one `fma` unless explicitly permitted
    // (correct IEEE-754-strict default -- contraction changes the last
    // rounding step) -- `arith.mulf`/`arith.addf` need a `fastmath
    // <contract>` attribute for the LLVM dialect ops they lower to
    // (`llvm.fmul`/`llvm.fadd`) to carry the matching `fastmathFlags =
    // #llvm.fastmath<contract>`, confirmed to survive `--convert-to-llvm`
    // unchanged on a minimal handwritten case.
    //
    // MLIR's own `--math-uplift-to-fma` pass (rewrites the pair to an
    // explicit `math.fma` *before* this point) was tried first and
    // reverted: it does get a real `vfmadd`, but overwhelmingly the
    // *scalar* single-lane form (`vfmadd132ss`, 1222 occurrences vs. 132
    // packed `vfmadd132ps`, measured directly on `digits-interop`'s own
    // real object file) -- explicit `math.fma`/`llvm.intr.fma` calls,
    // materialized this early, are far harder for LLVM's own loop
    // vectorizer to recognize and pack across iterations than a plain
    // `fmul`+`fadd` pair is; a real, measured *regression* end-to-end on
    // `examples/mnist-interop` (294s fused-only -> 324s with the uplift
    // pass added) confirmed this wasn't just a disassembly curiosity.
    // Leaving the pair as plain `arith.mulf`/`arith.addf`, merely stamped
    // `contract`, defers the actual fuse-or-not decision to LLVM's own
    // backend, *after* its vectorizer has already run (exactly how `clang
    // -ffp-contract=fast` works, and why: the vectorizer sees an ordinary,
    // easy-to-pack `fmul`/`fadd` pair, and only the final instruction-
    // selection step -- which already knows the target has native packed
    // FMA -- decides to fuse, on the now-already-vectorized form).
    run_stage(module, "mark contract", "builtin.module(cleave-mark-contract)")?;

    // Structured vectorization -- found and verified directly against this
    // toolchain, not assumed: `--affine-super-vectorize` (MLIR's own
    // `affine.for`-level auto-vectorizer) genuinely packs the matmul
    // accumulation loop into real `vector.transfer_read`/`write` on
    // `vector<16xf32>` (16 = AVX-512's own width for `f32` on this host --
    // still correct, just needing more than one register, on a narrower
    // target) -- confirmed end-to-end down to clean `llvm.fmul`/`llvm.fadd`
    // on `vector<16xf32>`, zero leftover `unrealized_conversion_cast`,
    // `--convert-vector-to-llvm` below resolves it fully. `mark_mulf_addf_
    // contract` runs *before* this stage specifically so the `fastmath
    // <contract>` attribute survives vectorization onto the now-vector-
    // typed `arith.mulf`/`addf` pair -- letting LLVM's backend fuse them
    // into a *packed* `vfmaddXXXps` at instruction-selection time, the
    // fix for the scalar-FMA regression noted above, now compounded with
    // deterministic (not opportunistic) vectorization instead of hoping
    // LLVM's own loop vectorizer recognizes the pattern on its own.
    //
    // `--affine-super-vectorize` errors ("NYI: non-trivial layout map") on
    // any *cross-function-boundary* memref (the `strided<[?,?],offset:?>`
    // shape `bufferize-function-boundaries=true` gives every tensor-typed
    // function parameter/return) -- `--inline` (above) already moves every
    // *real* call site's own computation onto purely local, non-strided
    // memrefs, which vectorize cleanly, but the *original*, now-
    // unreferenced callee declarations were still left behind (`--inline`
    // never deletes them on its own) carrying that same strided shape --
    // confirmed directly (not assumed) that melior's own `PassManager::run`
    // treats this as a genuine failure (`Result::Err`), unlike raw `mlir-
    // opt`'s own exit code on the same input, which stayed 0 despite the
    // identical diagnostic -- a real, load-bearing discrepancy between the
    // two, found by testing both, not just one. Fixed at the source, not
    // worked around here: `--symbol-dce` above, now actually able to see
    // these declarations as private+unreferenced (`lower_program`'s own
    // `sym_visibility` stamping, `mlir_lower.rs`), removes them before this
    // stage ever runs.
    // `AffineVectorize` is restricted to run *on* `func.func`, not
    // `builtin.module` directly (found directly: melior's own returned
    // parse error names this exactly -- "restricted to 'func.func' ...
    // did you intend to nest?") -- needs the same `outer(inner(...))`
    // nesting `mlir-opt`'s own `--pass-pipeline=` flag would, unlike `one-
    // shot-bufferize` above, which really does run at the module level.
    //
    // A loop over a strided memref that isn't a function's parameter (a
    // `Slice::slice` view of a local tensor) remains: the pass leaves it
    // scalar, for LLVM's loop vectorizer (its strides are static), and
    // reports it as an `error:` diagnostic while succeeding. Not an error
    // here: that one diagnostic is dropped while the pass runs.
    let skipped_strided_loop = context.attach_diagnostic_handler(|diagnostic| {
        diagnostic.to_string().contains("NYI: non-trivial layout map")
    });
    let vectorized = run_stage(
        module,
        "affine-super-vectorize",
        "builtin.module(func.func(affine-super-vectorize{virtual-vector-size=16}))",
    );
    context.detach_diagnostic_handler(skipped_strided_loop);
    vectorized?;

    // `--convert-vector-to-scf` again, now on what `--affine-super-
    // vectorize` just produced (the first run, before `--convert-linalg-to-
    // affine-loops`, never saw it), **only when it produced a permuted
    // transfer** (a column read, `(d0, d1) -> (d0)`), which `--convert-
    // vector-to-llvm` can't lower; `target-rank=0` takes it down to scalar
    // loops (a permuted read is a gather anyway). Only then: the pass turns
    // *every* 1-D transfer into a scalar loop over a stack buffer, which on
    // nanoLM's kernel meant ~2400 more `llvm.alloca`s in loops (each a
    // `_chkstk` call per iteration on Windows, 9% of a training step) and a
    // fifth more IR. Inlined builds produce no permuted transfer; a
    // non-inlined `matmul_transpose_b` does. Before `--lower-affine`: the
    // loops it builds index through `affine.apply`. Left in place, they reached the execution
    // engine still in the `vector` dialect and crashed LLVM's translation
    // outright, no diagnostic (found compiling a non-inlined
    // `matmul_transpose_b<8x16, 8x16>`, i.e. any kernel built `--no-inline`;
    // inlining had happened to route those loops elsewhere).
    run_stage(module, "vector-to-scf after super-vectorize", "builtin.module(cleave-lower-permuted-transfers)")?;

    // `tanh`/`exp`/`log` as polynomial approximations, now that the element-
    // wise ops are vectors: left to `--convert-math-to-llvm`, a `math.tanh`
    // on a `vector<1024xf32>` became 1024 calls to libm's `tanhf` (LLVM has
    // no vector math library here), 26% of a nanoLM training step.
    // `cleave-mlir-shim`'s `cleaveApproximateMath` has the details.
    run_stage(module, "math approximation", "builtin.module(cleave-approximate-math)")?;

    // One shared scalar lowering pipeline from here on, `options.openmp`
    // only ever inserting the two genuinely OpenMP-specific pieces into it
    // -- not, as an earlier version of this function had it, two entirely
    // separate hand-maintained pipelines (one "with openmp", one "without")
    // sharing no code past this point. That earlier split is exactly what
    // let a real bug in here go unnoticed for as long as it did (`doc/
    // backlog.md`'s own "An AOT binary built with `--no-openmp`... genuinely
    // crashes with a native stack overflow" entry has the full story): the
    // `else` branch's own doc comment asserted "no `memref.alloca_scope`
    // complication to split around" on this path -- false, one-shot-
    // bufferize inserts it unconditionally, nothing to do with OpenMP at
    // all -- and nobody revisited that assumption when the `if` branch's
    // own two-phase `--convert-to-llvm` split was later found necessary to
    // handle exactly that construct correctly. A `false` assumption baked
    // into one of two diverging copies of "the same" logic is a maintenance
    // hazard by construction; one shared path with two small, clearly-
    // labeled insertions can't drift the same way.

    // `--lower-affine`, on its own -- turns any `affine.parallel` the
    // parallelize stage above produced into `scf.parallel`, and every
    // remaining `affine.for` (still scalar/vector but not, or not yet,
    // OS-thread-parallel) into ordinary `scf.for`. Needed unconditionally,
    // `options.openmp` or not -- `--affine-super-vectorize` above always
    // leaves *some* `affine.for` behind regardless. Run strictly before
    // `--convert-scf-to-openmp` below when that runs at all -- that pass's
    // own `scf.parallel` lowering needs to see it, not `affine.parallel`.
    run_stage(module, "lower-affine", "builtin.module(lower-affine)")?;

    // See `cleave-insert-stack-scopes` (`cleave-mlir-shim`) for the full
    // story -- the general insertion point, found after auditing every
    // loop-producing pass in this whole function rather than reacting to
    // individual crashes one at a time (`doc/backlog.md`'s own entry on
    // this: two real, structurally different crashing loop shapes were
    // already found this way, a strong sign the *reactive* approach doesn't
    // generalize on its own). Right here, and nowhere earlier or later, is
    // the one point where *every* structured loop shape this whole pipeline
    // can ever produce coexists simultaneously, still fully structured (no
    // multi-block CFG anywhere yet, `--convert-scf-to-cf` hasn't run) and
    // not yet consumed by anything OpenMP-specific (`--convert-scf-to-
    // openmp`, right below, only fires `if options.openmp`): `scf.while`
    // (cleave's own `for`/`while`, `mlir_lower.rs::lower_loop`, present
    // since initial construction), `scf.for` (from the transform-dialect
    // matmul schedule's own tiling, present since before bufferization --
    // `Problem B`'s own pad-retry path, `doc/backlog-done.md`, is exactly
    // this shape), and `scf.parallel` (from *two* separate sources that
    // both only reach this shape by this exact point: `--scf-forall-to-
    // parallel`'s own conversion of Stage 1's `tile_using_forall`, run
    // right after one-shot-bufferize, and `--lower-affine` right above
    // converting whatever `affine.parallel` the *old*, non-transform-
    // dialect fallback path -- `--convert-linalg-to-affine-loops` further
    // above, for any matmul the schedule's own `vectorize` step declined --
    // produced via `--affine-parallelize`). An earlier version of this call
    // ran right after buffer-deallocation instead (before any of `--
    // convert-linalg-to-affine-loops`/`--affine-parallelize`/this `--lower-
    // affine` had run at all) -- correct for `scf.while` and the transform-
    // dialect schedule's own `scf.for` (both already present that early),
    // but structurally blind to anything the *old* fallback path or Stage
    // 1's own parallel tiling would ever produce, neither of which exists
    // in loop form until later passes run -- a real, found-by-audit gap
    // this move closes, not (yet) a gap any specific crash had exposed.
    run_stage(module, "stack scopes in loops", "builtin.module(cleave-insert-stack-scopes)")?;

    if options.openmp {
        // `--convert-scf-to-openmp` -- the one pass in this whole stage that
        // actually turns `scf.parallel` into `omp.parallel`/`omp.wsloop`/`omp.
        // loop_nest`, one real OS thread per outer-loop chunk (`libomp`,
        // `cleave-rt`'s build now links, provides the actual thread pool at
        // runtime -- see `doc/backlog.md`'s own OpenMP entry). Load-bearing
        // that this runs on its own, in this exact spot, before anything else
        // touches the `scf.for` loops still nested inside that `scf.parallel`
        // (`j`/`k`, from `--lower-affine` just above) -- found by direct `mlir-
        // opt` testing, not assumed: this pass wraps the loop body in a `memref.
        // alloca_scope`, whose own verifier requires a *single-block* region.
        // That's still true right here (the nested `scf.for`s are each one
        // structured op, not yet an unrolled multi-block CFG) -- but running
        // `--convert-scf-to-cf` *before* this point (to "get it out of the
        // way" early) would have already broken it, and running this pass
        // *after* `--convert-scf-to-cf` runs anywhere near these loops breaks
        // it just the same, the moment that pass ever touches them. The actual
        // fix, verified end-to-end via `mlir-opt`/`mlir-translate` on a real
        // matmul kernel down to genuine `__kmpc_fork_call`/`__kmpc_for_static_
        // init_*`/`__kmpc_barrier` calls: `--convert-to-llvm` (below) lowers
        // `memref.alloca_scope` itself via its own dialect-interface mechanism,
        // *while the nested loops are still structured* -- only *after* that
        // succeeds is `--convert-scf-to-cf` safe to run on what's left (the
        // scoping constraint is gone with `memref.alloca_scope` itself), hence
        // the *second* `--convert-to-llvm` invocation further below to finish
        // the job. `scf` itself has no `ConvertToLLVMPatternInterface` of its
        // own (confirmed: a lone `--convert-to-llvm`, with no explicit `--
        // convert-scf-to-cf` anywhere, leaves every `scf.for` completely
        // untouched) -- the two-pass split below is the real fix, not a
        // roundabout way of doing one pass's job. **This is exactly why the
        // split can't be collapsed away even now that it's unconditional
        // below** -- `memref.alloca_scope` (from one-shot-bufferize, present
        // whether or not this branch ever runs) needs the identical two-phase
        // treatment either way; `options.openmp` only decides whether an
        // `omp.parallel` region also happens to sit inside it.
        run_stage(module, "scf-to-openmp", "builtin.module(convert-scf-to-openmp)")?;
    }

    // Every parallel region's members place themselves one per physical core
    // (`cleave_mlir_shim::bind_teams`, `cleave-rt`'s `cleave_bind_worker`).
    run_stage(module, "bind teams", "builtin.module(cleave-bind-teams)")?;

    // First `--convert-to-llvm`: alongside `--convert-vector-to-llvm`
    // (unchanged from before OpenMP support existed) and, only when
    // `options.openmp` inserted a real `omp.*` region above, `--convert-
    // openmp-to-llvm` (legalizes every operand/region type *inside* those
    // ops to the `llvm` dialect -- the ops themselves deliberately survive,
    // see `register_all_llvm_translations`'s own call-site comment above
    // for why) -- this is specifically what needs to see `memref.alloca_
    // scope` while it's still single-block, per the comment above, and
    // needs to run this early regardless of `options.openmp`: one-shot-
    // bufferize inserts that construct unconditionally, not only when an
    // `omp.parallel` region happens to sit inside it.
    // `--openmp`'s parallel loops, or `spawn`'s tasks (`lower_spawns`).
    run_stage(
        module,
        "vector/openmp/to-llvm",
        &format!(
            "builtin.module(convert-vector-to-llvm,cleave-convert-openmp-if-used{{always={}}},convert-to-llvm)",
            options.openmp
        ),
    )?;

    // `--convert-scf-to-cf` now finishes off the remaining nested `scf.for`
    // loops (`memref.alloca_scope` is gone, so its single-block constraint
    // no longer applies to them) -- and the second `--convert-to-llvm`
    // mops up the `cf.br`/`cf.cond_br` that produces (`cf`, unlike `scf`,
    // does have its own `ConvertToLLVMPatternInterface`, confirmed by this
    // exact sequence leaving zero leftover ops). Unconditional, identical
    // either way -- when `options.openmp` is set, `omp.parallel`/`omp.
    // wsloop`/`omp.loop_nest` themselves are still present in the module at
    // this point, expected, not a bug, see `register_all_llvm_
    // translations`'s own comment; when it isn't, there never were any.
    run_stage(
        module,
        "scf-to-cf/to-llvm/reconcile",
        "builtin.module(convert-scf-to-cf,finalize-memref-to-llvm,convert-to-llvm,reconcile-unrealized-casts)",
    )?;

    // `llvm.emit_c_interface` (`lower_top_level_fn`'s own comment on it, put
    // only on `main`) makes the `--convert-to-llvm` above mint a
    // `_mlir_ciface_main` trampoline -- and it does so by reusing `main`'s
    // own `Location` object unmodified for the new op and its body, found
    // directly: running the real test suite (most of which JITs and calls
    // `main`) hard-crashed the LLVM verifier with "DISubprogram attached to
    // more than one function" and "!dbg attachment points at wrong
    // subprogram for function". A `DISubprogram` describes exactly one
    // function; two `llvm.func`s both fused to the same distinct instance
    // is invalid. The wrapper is a one-line auto-generated ABI shim, not
    // real user code -- nobody will ever want to set a breakpoint inside
    // `_mlir_ciface_main` -- so the fix is simply to strip all debug info
    // from its whole subtree before it ever reaches the verifier.
    strip_ciface_wrapper_debug_info(module);

    // Gives cleave sole ownership of every tensor payload's own physical
    // memory -- see `cleave-unify-tensor-allocations` (`cleave-mlir-shim`) for why this
    // runs *here* specifically (right after `--convert-to-llvm`, not
    // before) and why a blanket rename is sound.
    run_stage(module, "unify tensor allocations", "builtin.module(cleave-unify-tensor-allocations)")?;

    // Emit `!llvm.module.flags` with `CodeView = 1` so the LLVM backend
    // writes CodeView (`.debug$S`/`.debug$T`, what a Windows `.pdb` is
    // built from) rather than DWARF for the `DISubprogram`s
    // `mlir_lower.rs::build_di_subprograms` attached. `Debug Info Version`
    // is set here too (the translation would add it anyway, but being
    // explicit keeps both flags in one place). Built via `OperationBuilder`
    // -- `llvm.module_flags` has no melior wrapper, and the `flags`
    // inherent attribute is parsed from its textual form. Gated on
    // `CodegenOptions::debug_info` -- with no `DISubprogram`s attached
    // (`mlir_lower.rs::lower_program`'s own gate on the same option), these
    // flags describe debug info that doesn't exist.
    if options.debug_info {
        let flags = Attribute::parse(
            context,
            "[#llvm.mlir.module_flag<warning, \"CodeView\", 1 : i32>, \
             #llvm.mlir.module_flag<max, \"Debug Info Version\", 3 : i32>]",
        )
        .expect("pipeline: failed to parse llvm.module_flags attribute");
        let op = OperationBuilder::new("llvm.module_flags", Location::unknown(context))
            .add_attributes(&[(Identifier::new(context, "flags"), flags)])
            .build()
            .expect("pipeline: failed to build llvm.module_flags op");
        module.body().append_operation(op);
    }

    // Every op `mlir_lower.rs::gen_loc` ever stamped already carries its own
    // function's `DISubprogram`, fused in from the start (`gen_loc`'s own
    // doc comment) -- including a surviving `llvm.func`'s own top-level
    // location, so there's nothing left to attach here. What *does* still
    // need a floor: an op an MLIR lowering pass rebuilt from scratch (the
    // tiled/vectorized loop nests, mostly), which comes out with a bare
    // `UnknownLoc` no translation can resolve a debug scope from. Back-fill
    // those, per surviving `llvm.func`, with that function's own (already
    // good) location -- a profiler then attributes that code to the
    // function rather than to an unnamed address range.
    run_stage(module, "backfill locations", "builtin.module(cleave-backfill-locations)")?;

    // Argument slots (`mlir_lower.rs::entry_alloca`): one the inliner carried
    // into a loop goes back to its function's entry block, and an ordinary
    // call's is bounded to its uses, so that LLVM can share storage between
    // them. Loops are blocks by now.
    // Large aggregates copied from memory to memory become `memcpy`s, not one
    // load and one store per scalar (`copy_aggregates_in_memory`); before
    // the slots get their lifetimes, which follow their uses.
    // Then the functions kept out of line by the inline threshold or
    // `#[no_inline]` stay out of line in LLVM too.
    run_stage(
        module,
        "aggregate copies, argument slots, no-inline",
        &format!(
            "builtin.module(cleave-copy-aggregates-in-memory{{min-bytes={}}},cleave-hoist-arg-slots,cleave-apply-no-inline)",
            crate::mlir_lower::BY_POINTER_MIN_BYTES
        ),
    )?;

    Ok(())
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
    unsafe { cleave_mlir_shim::run_pipeline(module.to_raw(), "builtin.module(cleave-strip-ciface-debug-info)", false) }
        .expect("cleave-strip-ciface-debug-info");
}

/// Runs the textual pipeline `pipeline` (`builtin.module(...)`) on `module`:
/// for harnesses running their own passes rather than `lower_to_llvm`.
pub fn run_passes(module: &mut Module, pipeline: &str) -> Result<(), String> {
    register_passes();
    // SAFETY: `module` is a valid module, borrowed mutably here.
    unsafe { cleave_mlir_shim::run_pipeline(module.to_raw(), pipeline, false) }
}

/// A JIT for `module`, in the LLVM dialect, compiled for the host at
/// optimization level `opt_level`: for harnesses running their own passes.
pub fn jit(module: &Module, opt_level: usize) -> cleave_mlir_shim::ExecutionEngine {
    let target = cleave_mlir_shim::Target::new(None, None, opt_level, false, true).expect("the host target");
    // SAFETY: `module` is a valid module, borrowed for the call.
    unsafe { cleave_mlir_shim::ExecutionEngine::new(module.to_raw(), &target, &[]) }
        .unwrap_or_else(|e| panic!("failed to compile the module: {e}"))
}

/// The target `options` ask for (`cleave_mlir_shim::Target`): for objects
/// and JITs alike.
pub fn target(options: &CodegenOptions) -> Result<cleave_mlir_shim::Target, Vec<String>> {
    cleave_mlir_shim::Target::new(
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
    cleave_mlir_shim::mlir::utility::register_all_llvm_translations(&context);

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
    unsafe { cleave_mlir_shim::emit_object(module.to_raw(), &target, object_path_str) }
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
        // `-l libomp` (not `-l omp` -- the real installed file is genuinely
        // named `libomp.lib`, the cross-platform LLVM convention, and `rustc`
        // on an `-msvc` target passes an `-l` name straight through to `link.
        // exe` as `NAME.lib` with no automatic prefix-stripping the way a GNU
        // linker would) -- needed whenever `emit_object`'s own OpenMP
        // parallelization stage or `spawn`'s tasks were used: the object then
        // carries unresolved `__kmpc_*` relocations. `CLEAVE_LLVM_PREFIX` (`.cargo/
        // config.toml`, the toolchain the shim builds against) is reused
        // here rather than a second, independently-maintained path -- `/lib` under it is exactly where
        // the real toolchain install puts `libomp.lib` (confirmed directly,
        // alongside every other real `.lib` this build already links
        // against).
        let mlir_prefix = std::env::var("CLEAVE_LLVM_PREFIX").map_err(|_| {
            vec![
                "CLEAVE_LLVM_PREFIX must be set (see .cargo/config.toml) to link libomp"
                    .to_string(),
            ]
        })?;
        cmd.arg("-L")
            .arg(format!("{mlir_prefix}/lib"))
            .arg("-l")
            .arg("libomp");
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
