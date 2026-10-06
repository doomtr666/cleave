//! Queryable index over a merged `Program`'s `algebra`/`impl`/`struct`
//! declarations — "does algebra `Ring` have an `impl` for concrete type
//! `Vec2`", "which algebras declare a `add(T, T) -> T`-shaped signature",
//! "what fields does `struct Vec2` declare". Built from the *already-merged*
//! output of `driver::merge_programs` (one logical `AlgebraDecl`/`ImplDecl`
//! per name — see `driver.rs`), not something that itself merges fragments.
//! Structs are never fragmented across files at all (`merge_programs`
//! rejects a duplicate struct name outright), so there's nothing to merge
//! for them — just an index over the already-unique declarations.
//!
//! This is deliberately just the data structure and the query surface — no
//! constraint generation/checking lives here (that's `infer.rs`'s job,
//! resolving an operator call against `algebras_with_fn`/`has_impl`, or a
//! struct literal/field access against `struct_fields`).

use crate::ast::*;
use crate::const_eval;
use crate::infer::ConstValue;
use crate::print::{fmt_generics, fmt_type};
use std::collections::{HashMap, HashSet};

#[derive(Default)]
pub struct Registry {
    algebras: HashMap<String, AlgebraEntry>,
    structs: HashMap<String, StructEntry>,
    /// Every top-level `const NAME: T = expr;`, evaluated once here, eagerly
    /// — see `Registry::eval_global_consts`'s own doc comment for the
    /// evaluator itself.
    ///
    /// **Deliberately keyed by *name*, not `NodeId`** — unlike a const
    /// generic's own per-call-site value (`Ty::Const`, resolved differently
    /// for every instantiation), a whole-program `const`'s value never
    /// varies by which function/specialization happens to reference it, so
    /// it needs no per-node/per-specialization tracking at all (the
    /// `node_types`-shaped machinery `Infer`/`monomorphize.rs`/`cps.rs`
    /// already thread everywhere for *that* purpose would be real, avoidable
    /// overkill here — checked directly: `node_types` alone is threaded
    /// through two dozen-plus function signatures in `monomorphize.rs`
    /// alone). Two separate, independently-consulted tables, not one:
    /// `Infer::infer_expr_kind`'s own `Path` arm reads `global_const_types`
    /// (the const's own *declared* type, e.g. `i32` — an ordinary type,
    /// freely unifiable with anything else of that type, unlike `Ty::Const`
    /// itself, which would make two *different* consts of the same
    /// declared type fail to unify against each other through a shared
    /// generic `T` — see `doc/plan-blas-native.md` §2's own "known
    /// limitation" entry, closed by this exact split); `cps.rs::collect_
    /// units` clones `global_consts` (the actual *value*) straight into
    /// every `ConcreteUnit`, read back by `convert_expr`'s own `Path`
    /// handling the same way `node_types` already is, just keyed by name
    /// instead of `NodeId`.
    global_consts: HashMap<String, ConstValue>,
    /// `ConstDecl::ty`, one per entry in `global_consts` — see that field's
    /// own doc comment for why this is a second, parallel table rather than
    /// folded into it.
    global_const_types: HashMap<String, Type>,
    /// Every `define` name (never a `const` one) — `pipeline.rs::check_
    /// const_decl_errors`'s own diagnostic wording is the one consumer,
    /// distinguishing "this `define` has no default and was never
    /// overridden" from a `const`'s own "initializer isn't a compile-time
    /// constant" (a real `const` always has an `Expr` to evaluate; only a
    /// `define` can legitimately have nothing at all to try).
    is_define: HashSet<String>,
}

struct StructEntry {
    generics: Vec<GenericParam>,
    fields: Vec<Field>,
}

struct AlgebraEntry {
    generics: Vec<GenericParam>,
    /// Other algebras this one requires (`algebra Int<T> : Num { ... }`) —
    /// see `Registry::algebra_bounds`'s own doc comment for what this means
    /// and where it's actually enforced (not here — `Registry` stays just
    /// data, see the module doc).
    bounds: Vec<String>,
    sigs: Vec<FnSig>,
    /// Every `axiom` declared directly on this algebra (`algebra Ring<T> {
    /// axiom add_commutative(a, b): add(a, b) == add(b, a); }`) — trusted,
    /// unverified assertions over the algebra's own generic `T` (`doc/
    /// hld.md`'s own "v1 trust model": no per-concrete-impl soundness gate,
    /// an axiom holds for every `T` a `Ring` impl exists for, exactly like
    /// a Rust trait impl is trusted rather than proven). Consumed by a later
    /// e-graph rewriting pass, once one exists; retained here purely as data
    /// — same "just data, no validation" stance the rest of this module
    /// already takes (see its own doc comment).
    axioms: Vec<AxiomDecl>,
    /// Every `derivative` rule declared directly on this algebra (`doc/
    /// backlog-done.md`'s own "Auto-diff v1" entry) — mirrors `axioms`
    /// exactly: trusted, unvalidated data, consumed by `egraph.rs::
    /// derivative_rule_rewrites` to build real e-graph `Rewrite`s per
    /// reached concrete type.
    derivative_rules: Vec<DerivativeRuleDecl>,
    /// Every `adjoint` rule declared directly on this algebra (`doc/
    /// backlog.md`'s own "reverse-mode differentiation" item) — mirrors
    /// `derivative_rules` exactly: trusted, unvalidated data, consumed by
    /// the reverse-mode backward pass to look up each reached (algebra,
    /// method, concrete type)'s own declared contribution rule.
    adjoint_rules: Vec<AdjointRuleDecl>,
    /// Keyed by the target type's canonical string (`fmt_type`) — same
    /// grouping key `driver.rs` uses to merge `impl` fragments.
    impls: HashMap<String, ImplEntry>,
}

struct ImplEntry {
    /// The impl's *own* generics (`impl<T: Float> Ring<Complex<T>>`) — empty
    /// for the overwhelmingly common non-generic case (`impl Ring<i32>`).
    /// Non-empty means the target below isn't a *concrete* type at all, just
    /// a *pattern*; matching a query type against it needs real unification
    /// (`Infer::has_matching_impl`), which is why this needs exposing at
    /// all — `has_impl_named`'s plain string-key lookup can only ever
    /// recognize an *exact* previously-declared spelling.
    generics: Vec<GenericParam>,
    target: Type,
    /// Second and later targets, for a heterogeneous algebra
    /// (`algebra MatMul<A, B, C>`) — empty for every single-generic algebra
    /// (i.e. almost always). See `ImplDecl::extra_targets`'s own doc
    /// comment for why this stays a separate field rather than folding
    /// `target` into a single always-a-`Vec` shape.
    extra_targets: Vec<Type>,
    fns: Vec<FnDecl>,
}

impl Registry {
    pub fn build(program: &Program) -> Self {
        let mut algebras: HashMap<String, AlgebraEntry> = HashMap::new();

        for item in &program.items {
            if let ItemKind::Algebra(d) = &item.kind {
                let sigs = d
                    .items
                    .iter()
                    .filter_map(|ai| match &ai.kind {
                        AlgebraItemKind::FnSig(sig) => Some(sig.clone()),
                        AlgebraItemKind::Axiom(_)
                        | AlgebraItemKind::DerivativeRule(_)
                        | AlgebraItemKind::AdjointRule(_) => None,
                    })
                    .collect();
                let axioms = d
                    .items
                    .iter()
                    .filter_map(|ai| match &ai.kind {
                        AlgebraItemKind::Axiom(axiom) => Some(axiom.clone()),
                        AlgebraItemKind::FnSig(_)
                        | AlgebraItemKind::DerivativeRule(_)
                        | AlgebraItemKind::AdjointRule(_) => None,
                    })
                    .collect();
                let derivative_rules = d
                    .items
                    .iter()
                    .filter_map(|ai| match &ai.kind {
                        AlgebraItemKind::DerivativeRule(dr) => Some(dr.clone()),
                        AlgebraItemKind::FnSig(_)
                        | AlgebraItemKind::Axiom(_)
                        | AlgebraItemKind::AdjointRule(_) => None,
                    })
                    .collect();
                let adjoint_rules = d
                    .items
                    .iter()
                    .filter_map(|ai| match &ai.kind {
                        AlgebraItemKind::AdjointRule(ar) => Some(ar.clone()),
                        AlgebraItemKind::FnSig(_)
                        | AlgebraItemKind::Axiom(_)
                        | AlgebraItemKind::DerivativeRule(_) => None,
                    })
                    .collect();
                algebras
                    .entry(d.name.clone())
                    .or_insert_with(|| AlgebraEntry {
                        generics: d.generics.clone(),
                        bounds: d.bounds.clone(),
                        sigs,
                        axioms,
                        derivative_rules,
                        adjoint_rules,
                        impls: HashMap::new(),
                    });
            }
        }

        for item in &program.items {
            if let ItemKind::Impl(d) = &item.kind {
                let entry = algebras
                    .entry(d.algebra.clone())
                    .or_insert_with(|| AlgebraEntry {
                        generics: Vec::new(),
                        bounds: Vec::new(),
                        sigs: Vec::new(),
                        axioms: Vec::new(),
                        derivative_rules: Vec::new(),
                        adjoint_rules: Vec::new(),
                        impls: HashMap::new(),
                    });
                // `fmt_type(target)` alone would collide two *different*
                // generic impls sharing the same bare target shape but
                // different bounds (`impl<T: Float> Ring<Complex<T>>` vs.
                // `impl<T: Ord> Ring<Complex<T>>` both stringify as
                // `Complex<T>`) — found via testing (an overlap-detection
                // test lost one of its two impls entirely, silently, before
                // the check ever ran). `fmt_generics` is empty for the
                // overwhelmingly common non-generic case, so this key is
                // identical to the old plain `fmt_type` one wherever
                // `has_impl_named`'s fast lookup actually depends on it.
                // `extra_targets` folded in the same way, for the same
                // reason — two heterogeneous impls sharing a first target
                // but differing afterward must stay distinct too.
                let extra_targets_key: String = d.extra_targets.iter().map(fmt_type).collect();
                let key = format!(
                    "{}{}{}",
                    fmt_type(&d.target),
                    extra_targets_key,
                    fmt_generics(&d.generics)
                );
                entry.impls.insert(
                    key,
                    ImplEntry {
                        generics: d.generics.clone(),
                        target: d.target.clone(),
                        extra_targets: d.extra_targets.clone(),
                        fns: d.fns.clone(),
                    },
                );
            }
        }

        // Structs are never fragmented across files the way `algebra`/`impl`
        // are (`driver::merge_programs` rejects a duplicate struct name
        // outright) — nothing to merge here, just index the already-unique
        // declarations.
        let mut structs: HashMap<String, StructEntry> = HashMap::new();
        for item in &program.items {
            if let ItemKind::Struct(d) = &item.kind {
                structs.insert(
                    d.name.clone(),
                    StructEntry {
                        generics: d.generics.clone(),
                        fields: d.fields.clone(),
                    },
                );
            }
        }

        let (global_consts, define_errors) = Self::eval_global_consts(program, &[]);
        debug_assert!(
            define_errors.is_empty(),
            "an empty `defines` list can never itself be invalid: {define_errors:?}"
        );
        let global_const_types: HashMap<String, Type> = program
            .items
            .iter()
            .filter_map(|item| match &item.kind {
                ItemKind::Const(d) => Some((d.name.clone(), d.ty.clone())),
                ItemKind::Define(d) => Some((d.name.clone(), d.ty.clone())),
                _ => None,
            })
            .collect();
        let is_define: HashSet<String> = program
            .items
            .iter()
            .filter_map(|item| match &item.kind {
                ItemKind::Define(d) => Some(d.name.clone()),
                _ => None,
            })
            .collect();

        let mut registry = Registry {
            algebras,
            structs,
            global_consts,
            global_const_types,
            is_define,
        };
        // No real CLI/`CodegenOptions` context here (`build`, unlike `build_
        // with_defines`, never receives one) -- `true`, the same universal
        // default `resolve_codegen_options` itself now resolves to absent an
        // explicit `--openmp`/`--no-openmp`, so `CLEAVE_OPENMP` is available
        // (and correct by default) even to a caller that never goes through
        // `build_with_defines` at all (every one of this project's own
        // library-level tests, `egraph.rs`'s own internal speculative-probe
        // registries, ...).
        registry.inject_compiler_define("CLEAVE_OPENMP", bool_type(), ConstValue::Bool(true));
        registry
    }

    /// Like `build`, but with `--define NAME=VALUE` (CLI) / `Build::define`
    /// (`cleave-build`) overrides applied — the *only* difference `define`
    /// (`grammar.pest`'s own `define_decl` doc comment) has from a plain
    /// `const`: its resolved value can come from here instead of its own
    /// declared default. Returns every problem with `defines` *itself* as
    /// plain strings, not `Diagnostic`s -- these are configuration errors
    /// (bad CLI input), not source-code ones, no real `Span` to point at;
    /// `pipeline.rs::check_const_decl_errors` is the separate, complementary
    /// check for a source-level problem (a `define` with neither a default
    /// nor any override at all), which *does* have a real declaration site
    /// to point at.
    ///
    /// `openmp` is the resolved `CodegenOptions::openmp` value this exact
    /// compilation is using (`main.rs`'s own single, universal `args.openmp.
    /// unwrap_or(true)` resolution, or `CodegenOptions::openmp` directly at
    /// `compile_and_emit`'s own call site) -- injected as `CLEAVE_OPENMP`,
    /// this project's first compiler-provided constant (`doc/plan-blas-
    /// native.md`'s own §2, "Constantes injectées par le compilateur").
    /// Writes every `--define NAME=VALUE` override into `program` itself, as
    /// the `define`'s own value, before anything reads it: every registry
    /// built from the program afterwards (`Registry::build` alone included,
    /// which `cps::collect_struct_schemas` and others call on their own)
    /// sees the same values. Overrides applied to a separately built
    /// registry only left a struct's field dimensions at their defaults
    /// (`w: Tensor<f32, 1, W>` built at the overridden `W`, stored at the
    /// default one, failing MLIR verification). A name that isn't a
    /// `define`, or a value that doesn't parse, is left for
    /// `build_with_defines` to report.
    pub fn apply_defines(program: &mut Program, defines: &[(String, String)]) {
        for (k, item) in program.items.iter_mut().enumerate() {
            let ItemKind::Define(d) = &mut item.kind else { continue };
            let Some((_, raw)) = defines.iter().find(|(n, _)| *n == d.name) else { continue };
            if Self::parse_define_value(raw, &d.ty).is_none() {
                continue;
            }
            let kind = match raw.as_str() {
                "true" => ExprKind::BoolLit(true),
                "false" => ExprKind::BoolLit(false),
                _ => ExprKind::NumberLit { text: raw.clone(), suffix: None },
            };
            match &mut d.value {
                Some(value) => value.kind = kind,
                None => {
                    d.value = Some(Expr {
                        id: crate::ast::NodeId(u32::MAX - k as u32),
                        span: item.span,
                        kind,
                    })
                }
            }
        }
    }

    pub fn build_with_defines(
        program: &Program,
        defines: &[(String, String)],
        openmp: bool,
    ) -> (Self, Vec<String>) {
        let mut registry = Self::build(program);
        let (global_consts, errors) = Self::eval_global_consts(program, defines);
        registry.global_consts = global_consts;
        registry.inject_compiler_define("CLEAVE_OPENMP", bool_type(), ConstValue::Bool(openmp));
        (registry, errors)
    }

    /// Writes a compiler-provided constant directly into the registry's own
    /// tables, bypassing source-level declaration entirely -- the `CLEAVE_*`
    /// namespace convention is just that, a convention (no enforcement, no
    /// collision check against a same-named real `const`/`define`): the
    /// injected value always wins, silently, if a program's own source ever
    /// picks the same name -- deliberately simple, the reserved-namespace
    /// discipline is entirely the program author's own responsibility to
    /// respect, not the compiler's to police. Treated as a `define` (`is_
    /// define` records it, so `--dump-defines` lists it and it behaves
    /// exactly like a source-level `define` everywhere else once resolved)
    /// even though it never went through `eval_global_consts` at all.
    fn inject_compiler_define(&mut self, name: &str, ty: Type, value: ConstValue) {
        self.global_const_types.insert(name.to_string(), ty);
        self.global_consts.insert(name.to_string(), value);
        self.is_define.insert(name.to_string());
    }

    /// Evaluates every top-level `const`/`define` to a concrete
    /// `ConstValue`, once, here — a small, self-contained, *permissive*
    /// evaluator (same posture `const_eval.rs`'s own module doc comment
    /// takes: an expression shape it doesn't recognize, or one that
    /// references an as-yet-unresolved const, is left unevaluated here
    /// rather than erroring — a real "this isn't a constant expression"
    /// diagnostic is `infer.rs`'s own job, once it actually tries to use
    /// the const and finds nothing here).
    ///
    /// A fixpoint loop, not a single top-to-bottom pass: `const B: i32 = A
    /// + 1;` declared *before* `const A: i32 = 5;` in source must still
    /// resolve (`grammar.pest`'s own `const_decl` doesn't order-restrict
    /// this any more than `fn`/`struct` declarations already don't) —
    /// repeating the sweep until a full pass makes no further progress
    /// handles any declaration order, and terminates in at most
    /// `program.items.len()` rounds (each round resolves at least one
    /// previously-stuck const, or the loop stops).
    ///
    /// `defines` is checked *before* a `define`'s own default expression is
    /// even attempted — an external override always wins, the same `-D`
    /// semantics `#ifndef X #define X 42 #endif` has (`X`, if already
    /// defined from outside, is never re-evaluated from its own file-local
    /// default at all). Every entry in `defines` is validated against the
    /// program's own declarations: naming a real `const` (never overridable
    /// — "const" keeps meaning what it says), naming nothing at all, or a
    /// value that doesn't parse against its `define`'s own declared type,
    /// are each a real, reported error here rather than silently ignored.
    fn eval_global_consts(
        program: &Program,
        defines: &[(String, String)],
    ) -> (HashMap<String, ConstValue>, Vec<String>) {
        enum Decl<'a> {
            Const(&'a str, &'a Expr),
            Define(&'a str, &'a Type, Option<&'a Expr>),
        }
        let decls: Vec<Decl> = program
            .items
            .iter()
            .filter_map(|item| match &item.kind {
                ItemKind::Const(d) => Some(Decl::Const(&d.name, &d.value)),
                ItemKind::Define(d) => Some(Decl::Define(&d.name, &d.ty, d.value.as_ref())),
                _ => None,
            })
            .collect();

        let mut errors = Vec::new();
        for (name, _) in defines {
            match decls.iter().find(|d| match d {
                Decl::Const(n, _) | Decl::Define(n, _, _) => *n == name,
            }) {
                Some(Decl::Define(..)) => {}
                Some(Decl::Const(..)) => errors.push(format!(
                    "--define: `{name}` is a `const`, never overridable -- declare it `define` instead"
                )),
                None => errors.push(format!("--define: no such const/define `{name}`")),
            }
        }

        let mut resolved: HashMap<String, ConstValue> = HashMap::new();
        loop {
            let mut progressed = false;
            for d in &decls {
                let (name, ty, default) = match d {
                    Decl::Const(name, expr) => (*name, None, Some(*expr)),
                    Decl::Define(name, ty, default) => (*name, Some(*ty), *default),
                };
                if resolved.contains_key(name) {
                    continue;
                }
                // `ty.is_some()` iff this decl is a `Decl::Define` -- a
                // `Decl::Const` matched by a `defines` entry is already a
                // reported error above, but that error doesn't stop this
                // loop from also reaching this decl; it must still fall
                // through to evaluating the const's own initializer below,
                // never treat `raw` as if `ty` were its declared type.
                if let (Some(ty), Some((_, raw))) =
                    (ty, defines.iter().find(|(n, _)| n == name))
                {
                    match Self::parse_define_value(raw, ty) {
                        Some(v) => {
                            resolved.insert(name.to_string(), v);
                            progressed = true;
                        }
                        None => errors.push(format!(
                            "--define {name}={raw:?}: not a valid `{}`",
                            fmt_type(ty)
                        )),
                    }
                    continue;
                }
                if let Some(expr) = default {
                    if let Some(v) = Self::eval_const_expr(expr, &resolved) {
                        resolved.insert(name.to_string(), v);
                        progressed = true;
                    }
                }
            }
            if !progressed {
                break;
            }
        }
        (resolved, errors)
    }

    /// Parses `raw` (an external `--define` value's own text) against
    /// `ty`'s own declared shape — the same two `ConstValue` variants
    /// everything else in this module already works with, nothing new:
    /// `bool` parses `"true"`/`"false"`, anything else attempts a plain
    /// `u64` (matching `ExprKind::NumberLit`'s own identical `.parse::
    /// <u64>()` elsewhere in this file — a literal integer in source has
    /// never been width/signedness-checked against its own declared type
    /// either).
    fn parse_define_value(raw: &str, ty: &Type) -> Option<ConstValue> {
        let TypeKind::Path(p, _) = &ty.kind else {
            return None;
        };
        if p.segments.len() == 1 && p.segments[0] == "bool" {
            match raw {
                "true" => Some(ConstValue::Bool(true)),
                "false" => Some(ConstValue::Bool(false)),
                _ => None,
            }
        } else if p.segments.len() == 1 && matches!(p.segments[0].as_str(), "f32" | "f64") {
            raw.parse::<f64>().ok().map(ConstValue::float)
        } else {
            raw.parse::<u64>().ok().map(ConstValue::Int)
        }
    }

    /// The expression shapes this evaluator understands — deliberately the
    /// same subset `infer.rs::const_value_from_expr` already recognizes for
    /// a const-generic's own value (that one walks `Ty`/`Subst`, needed
    /// there since a const generic's value may still be an unresolved
    /// variable at the point it's consulted; this one only ever sees
    /// already-fully-parsed source text, so it works directly on `Expr` and
    /// plain `ConstValue`, no `Infer` instance needed).
    fn eval_const_expr(expr: &Expr, known: &HashMap<String, ConstValue>) -> Option<ConstValue> {
        match &expr.kind {
            ExprKind::NumberLit { text, .. } => text
                .parse::<u64>()
                .ok()
                .map(ConstValue::Int)
                .or_else(|| text.parse::<f64>().ok().map(ConstValue::float)),
            ExprKind::BoolLit(b) => Some(ConstValue::Bool(*b)),
            ExprKind::Path(p) if p.segments.len() == 1 => known.get(&p.segments[0]).copied(),
            ExprKind::Call(path, _, args, _) if path.segments.len() == 1 && args.len() == 1 => {
                let a = Self::eval_const_expr(&args[0], known)?;
                const_eval::eval_unop(&path.segments[0], a)
            }
            ExprKind::Call(path, _, args, _) if path.segments.len() == 1 && args.len() == 2 => {
                let a = Self::eval_const_expr(&args[0], known)?;
                let b = Self::eval_const_expr(&args[1], known)?;
                const_eval::eval_binop(&path.segments[0], a, b)
            }
            _ => None,
        }
    }

    /// `SEUIL`'s own *declared* type, for `Infer::infer_expr_kind`'s own
    /// `ExprKind::Path` fallback — an ordinary `Ty` (via `Infer::ty_from_
    /// ast`), not `Ty::Const`, is what that fallback actually needs (see
    /// `global_consts`'s own doc comment for why). `None` for any name
    /// that isn't a top-level const at all.
    pub fn global_const_type(&self, name: &str) -> Option<&Type> {
        self.global_const_types.get(name)
    }

    /// `true` for a `define`, `false` for a `const` (or any other name) —
    /// `is_define`'s own doc comment on `Registry` has the one consumer.
    pub fn is_define(&self, name: &str) -> bool {
        self.is_define.contains(name)
    }

    /// Every currently-resolved `define` — source-declared (`grammar.pest::
    /// define_decl`) or compiler-injected (`inject_compiler_define`, the
    /// `CLEAVE_*` namespace convention) alike — name and value, sorted by
    /// name for a stable, diffable listing. `main.rs`'s own `--dump-defines`
    /// is the one real consumer. A source-declared `define` that never
    /// resolved at all (no default, no `--define` override) is never in
    /// `self.global_consts` in the first place (`eval_global_consts`'s own
    /// permissive-by-omission posture) — `pipeline.rs::check_const_decl_
    /// errors` is what catches that case, separately, as a real diagnostic;
    /// this only ever sees defines that already resolved.
    pub fn list_defines(&self) -> Vec<(String, ConstValue)> {
        let mut out: Vec<(String, ConstValue)> = self
            .is_define
            .iter()
            .filter_map(|name| self.global_consts.get(name).map(|v| (name.clone(), *v)))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// `SEUIL`'s own resolved *value* — unlike `global_const_type` (used for
    /// *ordinary* expression position, where the value must stay separate
    /// from the type to unify freely, `global_consts`'s own doc comment has
    /// the full story), this is exactly what a *const-generic* position
    /// (`Tensor<f32, SEUIL, SEUIL>`, `probe::<SEUIL>()`) needs directly:
    /// `Infer::const_value_from_expr`'s own `Path` fallback, the same
    /// function an ordinary const generic's own value already resolves
    /// through. `Ty::Const` genuinely is the right representation there —
    /// two *different* dimensions in the same slot really must be caught as
    /// distinct, the exact opposite of the ordinary-value-position case.
    pub fn global_const_value(&self, name: &str) -> Option<ConstValue> {
        self.global_consts.get(name).copied()
    }

    /// Did `name`'s own initializer actually evaluate? `pipeline.rs::check_
    /// const_decl_errors`'s one caller -- a name can be declared (`global_
    /// const_type` returns `Some`, straight off the AST, unconditionally)
    /// while still failing *this* check, if `eval_global_consts` couldn't
    /// resolve its own initializer expression.
    pub fn has_global_const_value(&self, name: &str) -> bool {
        self.global_consts.contains_key(name)
    }

    /// A snapshot of every top-level const's own resolved *value* —
    /// `cps.rs::collect_units`'s one real consumer, cloned once into every
    /// `ConcreteUnit` (see `global_consts`'s own doc comment for why a
    /// plain, name-keyed clone is enough, no per-node/per-specialization
    /// tracking needed). A name present here but absent (or present with a
    /// different, stale-looking value) is never possible by construction —
    /// this *is* the same table `global_const_type`'s own caller already
    /// trusted enough to type-check the reference against.
    pub fn global_consts(&self) -> HashMap<String, ConstValue> {
        self.global_consts.clone()
    }

    /// Does `algebra` have an `impl` for this concrete target type? String
    /// comparison against `fmt_type`, same canonicalization `driver.rs`
    /// already uses for merging — not a structural/generic-aware match.
    pub fn has_impl(&self, algebra: &str, target: &Type) -> bool {
        self.has_impl_named(algebra, &fmt_type(target))
    }

    /// Same check as `has_impl`, keyed directly by the type's canonical
    /// name string rather than an AST `Type` node — for callers (`infer.rs`)
    /// that only have their own internal type representation at hand, with
    /// no AST node (and no `NodeId`/`Span` to invent one from) to point to.
    pub fn has_impl_named(&self, algebra: &str, type_name: &str) -> bool {
        self.algebras
            .get(algebra)
            .is_some_and(|e| e.impls.contains_key(type_name))
    }

    /// Names of every declared algebra that has a `fn` signature named
    /// `fn_name` with exactly `arity` parameters — the candidate set an
    /// unqualified operator call (`add`) resolves against. More than one
    /// candidate is a real ambiguity (see conversation notes: this is not a
    /// "someone's trying to override an existing algebra" signal — two
    /// independent, legitimately-scoped algebras can both declare their own
    /// `add` — it's an ordinary name collision resolved like Rust's
    /// ambiguous trait methods: reject, ask for explicit qualification).
    /// Resolving that call is still the next increment's job, not this
    /// method's — this only reports the candidate set.
    pub fn algebras_with_fn(&self, fn_name: &str, arity: usize) -> Vec<&str> {
        self.algebras
            .iter()
            .filter(|(_, entry)| {
                entry
                    .sigs
                    .iter()
                    .any(|s| s.name == fn_name && s.params.len() == arity)
            })
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// The signature an algebra declares for `fn_name`, if any — for
    /// checking argument types against the algebra's own declared
    /// parameter/return types.
    /// The names of the functions `algebra` declares, in order; empty for an
    /// unknown algebra.
    pub fn fn_names(&self, algebra: &str) -> Vec<&str> {
        self.algebras
            .get(algebra)
            .map(|a| a.sigs.iter().map(|s| s.name.as_str()).collect())
            .unwrap_or_default()
    }

    pub fn fn_sig(&self, algebra: &str, fn_name: &str) -> Option<&FnSig> {
        self.algebras
            .get(algebra)?
            .sigs
            .iter()
            .find(|s| s.name == fn_name)
    }

    /// The algebra's own generic parameters (`<T>` in `algebra Ring<T>`) —
    /// needed to instantiate a declared signature's `T`-typed parameters
    /// with fresh inference type variables, rather than treating `T` as a
    /// bare concrete type named `"T"`.
    pub fn generics(&self, algebra: &str) -> &[GenericParam] {
        self.algebras
            .get(algebra)
            .map(|e| e.generics.as_slice())
            .unwrap_or(&[])
    }

    pub fn has_algebra(&self, algebra: &str) -> bool {
        self.algebras.contains_key(algebra)
    }

    /// Other algebras `algebra` itself requires (`algebra Int<T> : Num {
    /// ... }` — `Registry::generics(Int)` returns `bounds: ["Num"]`). Mirrors
    /// a `GenericParam::Type`'s own `bounds` field, one level up: instead of
    /// constraining a caller's generic parameter, this constrains *every*
    /// type any `impl` of `Int` is ever declared for — the actual check
    /// (`Int<i32>` existing implies `Num<i32>` must too) lives in
    /// `Infer::match_impl`, not here (`Registry` stays just data, see the
    /// module doc).
    pub fn algebra_bounds(&self, algebra: &str) -> &[String] {
        self.algebras
            .get(algebra)
            .map(|e| e.bounds.as_slice())
            .unwrap_or(&[])
    }

    /// Every `axiom` declared on `algebra` — see `AlgebraEntry::axioms`'s own
    /// doc comment. Empty, not missing, for an algebra with none (mirrors
    /// `algebra_bounds`'s own convention just above) and for an unknown
    /// algebra name entirely.
    pub fn axioms(&self, algebra: &str) -> &[AxiomDecl] {
        self.algebras
            .get(algebra)
            .map(|e| e.axioms.as_slice())
            .unwrap_or(&[])
    }

    /// Every `derivative` rule declared on `algebra` — see `AlgebraEntry::
    /// derivative_rules`'s own doc comment.
    pub fn derivative_rules(&self, algebra: &str) -> &[DerivativeRuleDecl] {
        self.algebras
            .get(algebra)
            .map(|e| e.derivative_rules.as_slice())
            .unwrap_or(&[])
    }

    /// Every `adjoint` rule declared on `algebra` — see `AlgebraEntry::
    /// adjoint_rules`'s own doc comment.
    pub fn adjoint_rules(&self, algebra: &str) -> &[AdjointRuleDecl] {
        self.algebras
            .get(algebra)
            .map(|e| e.adjoint_rules.as_slice())
            .unwrap_or(&[])
    }

    /// The reverse of `algebra_bounds`: every algebra that names `algebra`
    /// among its *own* bounds (`Int` and `Float` both, for `algebras_bounded_
    /// by("Num")`, once `algebra Int<T> : Num` / `algebra Float<T> : Num`
    /// exist) — what `Infer::has_matching_impl` walks to check "does some
    /// *other*, more specific algebra's impl count as this one too".
    pub fn algebras_bounded_by<'a>(
        &'a self,
        algebra: &'a str,
    ) -> impl Iterator<Item = &'a str> + 'a {
        self.algebras
            .iter()
            .filter(move |(_, e)| e.bounds.iter().any(|b| b == algebra))
            .map(|(name, _)| name.as_str())
    }

    /// Every declared algebra's name — used by `Infer::check_no_overlapping_impls`
    /// to sweep the whole registry once, rather than being told which
    /// algebras exist by some other, already-scoped caller.
    pub fn algebra_names(&self) -> impl Iterator<Item = &str> {
        self.algebras.keys().map(String::as_str)
    }

    pub fn has_struct(&self, name: &str) -> bool {
        self.structs.contains_key(name)
    }

    /// A declared struct's own fields, in declaration order — `None` if
    /// `name` doesn't name a known struct at all (distinct from `Some(&[])`,
    /// a genuinely empty struct).
    /// Every struct name this registry knows about — used by `egraph.rs`'s
    /// own `derivative-independent-zero` rule to snapshot, once (`Applier`
    /// implementations must be `Send + Sync + 'static`, so they can't hold
    /// a borrowed `&Registry` themselves), which single-field pack-generic
    /// structs (`linalg::Tensor`'s own shape) it knows how to build a
    /// same-shaped zero for.
    pub fn struct_names(&self) -> impl Iterator<Item = &str> {
        self.structs.keys().map(String::as_str)
    }

    pub fn struct_fields(&self, name: &str) -> Option<&[Field]> {
        self.structs.get(name).map(|e| e.fields.as_slice())
    }

    /// A declared struct's own generic parameters (`<T>` in `struct
    /// Vec2<T>`) — used to map a field's declared type (which may mention
    /// one of these names) to a real type, either fresh (construction) or
    /// the concrete argument a particular value was built with (field
    /// access) — see `infer.rs`'s `StructLit`/`FieldAccess` handling.
    pub fn struct_generics(&self, name: &str) -> &[GenericParam] {
        self.structs
            .get(name)
            .map(|e| e.generics.as_slice())
            .unwrap_or(&[])
    }

    /// Every *single-target* impl of `algebra` whose own target is a
    /// pattern rather than a concrete type (`impl<T: Float>
    /// Ring<Complex<T>>` — has generic parameters of its own, distinct from
    /// the algebra's own `<T>`), each as `(generics, target)` — for
    /// `Infer::has_matching_impl`'s real, unification-based matching.
    /// Excludes the common non-generic case entirely (still served by the
    /// plain, fast `has_impl_named`) *and* excludes every multi-target,
    /// heterogeneous impl (`extra_targets` non-empty) — `has_matching_impl`
    /// only ever checks *one* type against *one* pattern, so matching just
    /// a heterogeneous impl's own first target in isolation would be
    /// structurally meaningless (it says nothing about whether the *other*
    /// targets could also be satisfied); those go through
    /// `multi_target_impls`/`Infer::has_matching_impl_multi` instead, which
    /// checks every target together, coherently.
    pub fn generic_impls(&self, algebra: &str) -> Vec<(&[GenericParam], &Type)> {
        self.algebras
            .get(algebra)
            .map(|e| {
                e.impls
                    .values()
                    .filter(|i| !i.generics.is_empty() && i.extra_targets.is_empty())
                    .map(|i| (i.generics.as_slice(), &i.target))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// *Every* impl of `algebra`, uniformly, each as `(generics, all
    /// targets in declaration order)` — single-target or heterogeneous,
    /// generic or fully concrete alike, no filtering at all (unlike
    /// `generic_impls`, which deliberately excludes the cases already
    /// served by the fast string-keyed `has_impl_named`). For
    /// `Infer::dispatch_algebra_call`'s own real, *committing* dispatch:
    /// even a fully concrete, non-generic impl needs real unification
    /// (not just a string-equality check) whenever the *query* side still
    /// has an unresolved slot of its own — a generic appearing only in an
    /// algebra fn's return type, never independently pinned by any
    /// argument, is exactly that case; unifying a `Var` against a known
    /// concrete target is what actually binds it.
    pub fn all_impls(&self, algebra: &str) -> Vec<(&[GenericParam], Vec<&Type>)> {
        self.algebras
            .get(algebra)
            .map(|e| {
                e.impls
                    .values()
                    .map(|i| {
                        let targets: Vec<&Type> = std::iter::once(&i.target)
                            .chain(i.extra_targets.iter())
                            .collect();
                        (i.generics.as_slice(), targets)
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Every concrete type name known to satisfy `algebra` — direct impls,
    /// plus every type reachable transitively through `algebras_bounded_by`
    /// (the same bound-inheritance relationship `Infer::has_matching_impl`
    /// already walks for a *single* type probe, here built as a set instead
    /// — needed so `Infer::generalize`'s own scheme-satisfiability check can
    /// intersect two algebras' own candidate sets directly). `Num` itself
    /// has zero *direct* impls anywhere (`stdlib/num/num.cleave`'s own
    /// comment: every concrete numeric type reaches it only through `Int`/
    /// `Float`'s own bound) — without the recursive half here, every
    /// `Num`-constrained variable would come back with an empty candidate
    /// set and look falsely unsatisfiable.
    ///
    /// Deliberately narrow, matching `has_matching_impl`'s own documented
    /// gaps: only a non-generic, single-target, bare (no generic arguments
    /// of its own) impl counts — a generic impl (`impl<T> Ring<Complex<T>>`)
    /// contributes no *one* fixed concrete name, and an algebra satisfied
    /// only via its own *forward*-aggregate bounds (two or more bounds,
    /// neither individually implemented — see `has_matching_impl`'s own
    /// "forward aggregate" case) isn't attempted here either — not needed
    /// for `Int`/`Float`/`Num`, the only shapes that exist today.
    pub fn candidates_for(&self, algebra: &str) -> HashSet<String> {
        self.candidates_for_inner(algebra, &mut HashSet::new())
    }

    /// `visited` guards against a cyclic bound declaration (`algebra A : B`,
    /// `algebra B : A`) looping forever — same reasoning, same shape, as
    /// `Infer::has_matching_impl_inherited`'s own identical guard.
    fn candidates_for_inner<'a>(
        &'a self,
        algebra: &'a str,
        visited: &mut HashSet<&'a str>,
    ) -> HashSet<String> {
        let mut out = HashSet::new();
        if !visited.insert(algebra) {
            return out;
        }
        for (generics, targets) in self.all_impls(algebra) {
            let [target] = targets.as_slice() else {
                continue;
            };
            if !generics.is_empty() {
                continue;
            }
            if let TypeKind::Path(path, args) = &target.kind {
                if args.is_empty() {
                    out.insert(path.segments.join("::"));
                }
            }
        }
        for other in self.algebras_bounded_by(algebra) {
            out.extend(self.candidates_for_inner(other, visited));
        }
        out
    }

    /// Whether `candidates_for(algebra)` is only part of the story: some
    /// impl, of `algebra` or of an algebra bounded by it, is generic or has a
    /// parameterized target (`impl<T> Norm<Box<T>>`), whose types can't be
    /// listed. An empty candidate set then proves nothing.
    pub fn has_open_impls(&self, algebra: &str) -> bool {
        self.has_open_impls_inner(algebra, &mut HashSet::new())
    }

    fn has_open_impls_inner<'a>(&'a self, algebra: &'a str, visited: &mut HashSet<&'a str>) -> bool {
        if !visited.insert(algebra) {
            return false;
        }
        let open = self.all_impls(algebra).iter().any(|(generics, targets)| {
            !generics.is_empty()
                || !matches!(targets.as_slice(), [t] if matches!(&t.kind, TypeKind::Path(_, args) if args.is_empty()))
        });
        open || self
            .algebras_bounded_by(algebra)
            .collect::<Vec<_>>()
            .into_iter()
            .any(|other| self.has_open_impls_inner(other, visited))
    }

    /// Like `all_impls`, but also hands back each impl's own declared `fn`s
    /// — needed by `monomorphize.rs`, the one consumer that actually needs
    /// to specialize an impl method's own *body*, not just match its target
    /// pattern the way dispatch (`Infer::match_impl`) does. A separate
    /// method rather than widening `all_impls` itself: `match_impl` runs on
    /// every algebra-dispatched call during ordinary inference, and has no
    /// use for the extra `&[FnDecl]` slice it would otherwise have to carry
    /// around for nothing.
    pub fn all_impls_with_fns(
        &self,
        algebra: &str,
    ) -> Vec<(&[GenericParam], Vec<&Type>, &[FnDecl])> {
        self.algebras
            .get(algebra)
            .map(|e| {
                e.impls
                    .values()
                    .map(|i| {
                        let targets: Vec<&Type> = std::iter::once(&i.target)
                            .chain(i.extra_targets.iter())
                            .collect();
                        (i.generics.as_slice(), targets, i.fns.as_slice())
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// A synthetic `bool` type node, for `inject_compiler_define`'s own call
/// sites -- `NodeId(u32::MAX)`/a zero-length `Span` at file 0 are safe
/// sentinels here specifically because this node never enters `program.
/// items` at all (it lives only in the registry's own tables, read back by
/// `global_const_type`/`ty_from_ast`, neither of which inspects `id`), so
/// no real parsed node's own id can ever collide with it.
fn bool_type() -> Type {
    Node {
        id: NodeId(u32::MAX),
        span: Span {
            file: FileId(0),
            start: 0,
            end: 0,
        },
        kind: TypeKind::Path(Path::single("bool"), Vec::new()),
    }
}
