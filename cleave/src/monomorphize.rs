//! Monomorphization: for every generic top-level `fn` *and* every generic
//! `algebra`-impl method, generates one fully concrete specialization per
//! instantiation actually reachable from a concrete entry point — the
//! missing piece between real HM polymorphism/qualified dispatch and an
//! eventual backend that needs every type fully resolved, no free type
//! variables anywhere left (see `type_inference.md`'s own "Monomorphization"
//! section).
//!
//! Scoped to top-level `fn`s and *algebra*-impl methods — generic *inherent*
//! impl methods (`impl<T> struct Boxed<T> { ... }`) aren't attempted yet, a
//! smaller, structurally similar follow-up left for later (no algebra-level
//! dispatch-candidate search needed there — `Registry::inherent_method` is
//! already a direct, unambiguous lookup — but nothing here reuses the
//! algebra-specific matching machinery below for it yet).
//!
//! ## One unified algorithm underneath two different front-ends
//!
//! A top-level `fn` call resolves via a direct name lookup (`env.get(name)`
//! → one `Scheme`). An algebra-dispatched call (`a * b` → `mul(a, b)`)
//! resolves via a *structural* search instead — `check_no_overlapping_impls`
//! guarantees at most one impl of the matched algebra can coherently apply,
//! but *finding* it means trying candidates, not looking up a name. That
//! front-end difference is real and stays — but once a candidate is
//! selected, the core algorithm is identical: reverse-unify the candidate's
//! own pattern (a `Scheme`'s `ty`, or an impl's own target/signature
//! patterns) against a call site's already-concrete `node_types`, in a
//! throwaway `Subst`, to recover concrete bindings for every one of the
//! candidate's own generics — then substitute those bindings through the
//! candidate's own body to produce one specialization. Both worklists below
//! share this same reverse-unification shape; only how each one *finds* its
//! own candidate differs.
//!
//! ## No AST cloning needed
//!
//! `node_types: HashMap<NodeId, Ty>` is read as a parameter by every render
//! function (`dump_block` etc.), never baked into the AST itself. A generic
//! function's (or impl method's) body (`Block`) stays *one* shared,
//! unmodified reference across every instantiation of it — each concrete
//! specialization just gets its *own* separate `node_types` map, built by
//! substituting the *original* declaration's own (still-generic) node types
//! through that instantiation's own `TyVar -> Ty` mapping. No fresh
//! `NodeId`s, no deep-cloning `Expr`/`Block`/`Stmt`.
//!
//! ## No `Infer`/`TyVarGen` instance needed for the reverse-derivation step
//!
//! To recover *which* concrete types a call site instantiated a generic
//! callee at — never recorded anywhere by ordinary inference; both
//! `infer_call`'s own `instantiate_with_mapping` (for top-level `fn`s) and
//! `dispatch_algebra_call`'s own per-candidate `mapping` (for algebra impls)
//! build exactly this kind of mapping and then discard it on the spot —
//! unify the candidate's own pattern (using its own existing `TyVar`s
//! directly, no fresh re-instantiation) against a query built from the
//! *caller's* own already-concrete `node_types`, in a throwaway
//! `Subst::default()` used once and discarded. Each call site gets its own
//! fresh scratch `Subst` — no shared `TyVarGen`, so no risk from different
//! functions'/impls' own `TyVar` ids numerically colliding (each
//! `callgraph::infer_program` group, and each generic impl method's own
//! declaration-time inference below, mints its own fresh `Infer`, so raw
//! `TyVar` ids are *not* globally unique in the first place — this is fine
//! as long as a mapping built from one candidate's own pattern is only ever
//! applied to that same candidate's own body).
//!
//! This also handles self- and mutual-recursion for free, for both
//! worklists: a recursive/mutually-recursive call was already unified
//! against the same monomorphic self-placeholder (`infer_fn_raw`'s own
//! seeded placeholder for a top-level `fn`; dispatch's own signature-driven
//! resolution for an algebra impl, which never needed the callee's body
//! finished in the first place) during the candidate's own declaration-time
//! inference — so reverse-deriving its instantiation from `node_types`
//! naturally recovers the *same* concrete type as the enclosing
//! instantiation, no special-casing.
//!
//! Building a generic algebra impl method's own *template* (its param/
//! return/target patterns, and its body's own still-generic `node_types`)
//! reuses `Infer::infer_impl_fn_generic_with_env` directly — the exact same
//! entry point `dump.rs` already calls for `--dump-inference-pass` — and
//! reads back `Infer::target_types` (a field added specifically for this:
//! the impl's own resolved target pattern(s), through the *same* fresh
//! `impl_mapping` its `param_types`/`node_types` already used) alongside the
//! usual `param_types`/`node_types`. Without sharing that one `impl_mapping`
//! across all three, unifying `param_types` against a concrete call
//! wouldn't correctly pin an algebra generic that appears *only* in the
//! impl's own target pattern, never in any parameter (`C` in `fn mul(a: A,
//! b: B) -> C;`, exactly `MatMul`'s own shape).

use crate::ast::*;
use crate::callgraph::{self, ProgramInference};
use crate::dump::{dump_block_with_call_names, fmt_ty_named, TyVarNames};
use crate::infer::{free_vars, substitute, unify, Env, Infer, Scheme, Subst, Ty, TyVar, TypeError, TypeErrorKind};
use crate::registry::Registry;
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

/// One concrete instantiation — of a top-level `fn` or of a generic
/// algebra-impl method alike, the point where the two worklists below
/// converge back into one shared shape (see the module's own doc comment).
struct Specialization {
    params: Vec<Param>,
    body: Block,
    param_types: Vec<Ty>,
    result: Ty,
    node_types: HashMap<NodeId, Ty>,
    /// This specialization's *own* resolved mangled callee names — kept
    /// per-specialization, not one shared global map: every instantiation
    /// of the same generic candidate shares the *same* body, and therefore
    /// the *same* `NodeId`s (see the module's own "no AST cloning" doc
    /// comment). A self-recursive call site's `NodeId` is identical across
    /// `fibonacci<i32>` and `fibonacci<i64>` — it must resolve to
    /// `"fibonacci<i32>"` in the first and `"fibonacci<i64>"` in the
    /// second, so one global map keyed only by `NodeId` cannot represent
    /// both; a real bug, found by testing, that a single shared map
    /// produced (`fibonacci<i64>`'s own recursive call rendered as
    /// `fibonacci<i32>`, whichever specialization happened to be processed
    /// last).
    call_names: HashMap<NodeId, String>,
}

pub struct MonomorphizedProgram {
    /// Keyed by mangled display name (`"identity<i32>"`, `"MatMul::mul<
    /// Matrix<f32, 2, 3>, Matrix<f32, 3, 5>, Matrix<f32, 2, 5>>"`) — also
    /// the dedup key each worklist itself uses: two different
    /// instantiations always render to two different strings, and the same
    /// instantiation always renders to the same one.
    specializations: HashMap<String, Specialization>,
    /// Origin name (a bare top-level `fn` name, or `"Algebra::method"` for
    /// an impl method) -> its own specializations' display keys, in the
    /// order they were first discovered — `HashMap` iteration order isn't
    /// stable, and output should be deterministic run to run.
    by_origin: HashMap<String, Vec<String>>,
    /// Resolved mangled callee names for calls made from a *non-generic*
    /// ("seed") function's own body (`main`, say) — safe as a single global
    /// map, unlike `Specialization::call_names` above: a seed function is
    /// processed exactly once, its own body's `NodeId`s are never revisited
    /// under a different instantiation, so there's nothing for two entries
    /// to collide over.
    seed_call_names: HashMap<NodeId, String>,
    /// Every `MonomorphizationFailed` error found during either worklist —
    /// see `derive_impl_instantiation`'s own doc comment for exactly when
    /// this happens (candidates existed for a call's own method name, but
    /// none of them could actually be instantiated at its concrete types).
    errors: Vec<TypeError>,
}

impl MonomorphizedProgram {
    /// Every specialization discovered for `origin` (a top-level `fn`'s own
    /// bare name, or `"Algebra::method"`), in first-discovered order —
    /// empty (not missing) for a generic candidate that type-checked fine
    /// but was never actually called from any concrete entry point.
    pub fn specializations_of(&self, origin: &str) -> &[String] {
        self.by_origin.get(origin).map(Vec::as_slice).unwrap_or(&[])
    }

    pub fn params(&self, key: &str) -> &[Param] {
        &self.specializations[key].params
    }

    pub fn body(&self, key: &str) -> &Block {
        &self.specializations[key].body
    }

    pub fn param_types(&self, key: &str) -> &[Ty] {
        &self.specializations[key].param_types
    }

    pub fn result(&self, key: &str) -> &Ty {
        &self.specializations[key].result
    }

    pub fn node_types(&self, key: &str) -> &HashMap<NodeId, Ty> {
        &self.specializations[key].node_types
    }

    pub fn call_names(&self, key: &str) -> &HashMap<NodeId, String> {
        &self.specializations[key].call_names
    }

    pub fn seed_call_names(&self) -> &HashMap<NodeId, String> {
        &self.seed_call_names
    }

    pub fn errors(&self) -> &[TypeError] {
        &self.errors
    }
}

/// A generic algebra-impl method's own declaration-time "template" — built
/// once per (impl, method), before either worklist runs, exactly the way a
/// top-level `fn`'s own `Scheme` (from `callgraph::infer_program`) is
/// already built once up front. Everything here shares one `impl_mapping`
/// (via `Infer::infer_impl_fn_generic_with_env` + `Infer::target_types`),
/// so unifying `param_patterns`/`ret_pattern` against a concrete call and
/// reading back bindings for the free variables appearing *anywhere* here
/// (including `target_patterns`, which `param_patterns` alone doesn't
/// always cover — see the module's own doc comment) gives one consistent
/// answer.
struct ImplTemplate {
    algebra: String,
    method_name: String,
    params: Vec<Param>,
    body: Block,
    param_patterns: Vec<Ty>,
    ret_pattern: Ty,
    target_patterns: Vec<Ty>,
    node_types: HashMap<NodeId, Ty>,
    /// Whether the *impl* (not the algebra) this template came from declared
    /// any of its own generics — `false` for `impl MatMul<f32,f32,f32>`,
    /// `true` for `impl<T,N,M,K> MatMul<Matrix<T,N,M>,...>`. A concrete
    /// impl's own `param_patterns`/`ret_pattern` carry no free variables at
    /// all (already fully resolved against its own concrete targets) — its
    /// template exists purely so `derive_impl_instantiation` can recognize
    /// "a concrete impl already covers this call" *structurally*, checking
    /// the whole parameter/return shape together the same way a generic
    /// template's own match does, rather than a separate, single-type-at-a-
    /// time string lookup (`Registry::has_impl_named`) that can't recognize
    /// a multi-target algebra's own combined key (found by direct testing:
    /// `examples/matmul.cleave`'s own `impl MatMul<f32,f32,f32>` was invisible
    /// to a per-type `has_impl_named` check, since the registry's own key for
    /// it is the *concatenation* `"f32f32f32"`, never any individual `"f32"`
    /// alone).
    is_generic: bool,
}

/// Runs the whole-program inference pass (`callgraph::infer_program`) and
/// then both monomorphization worklists over its result — mirrors
/// `dump.rs`'s own `dump_program`, which runs the identical first step for
/// its own, separate purpose.
pub fn monomorphize(program: &Program, registry: &Registry) -> (MonomorphizedProgram, ProgramInference) {
    let program_inference = callgraph::infer_program(program, registry);
    let functions: HashMap<&str, &FnDecl> = program
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::Fn(f) => Some((f.name.as_str(), f)),
            _ => None,
        })
        .collect();

    let templates = build_impl_templates(program, registry, &program_inference.global_env);

    let mut mono = MonomorphizedProgram {
        specializations: HashMap::new(),
        by_origin: HashMap::new(),
        seed_call_names: HashMap::new(),
        errors: Vec::new(),
    };
    let mut fn_worklist: Vec<(String, Vec<Ty>)> = Vec::new();
    let mut impl_worklist: Vec<(usize, HashMap<TyVar, Ty>)> = Vec::new();

    // Seed: every function that itself type-checked to something *fully
    // concrete* (never generalized — a nullary member per the Monomorphism
    // Restriction, or a parameterized one whose own scheme just happens to
    // have no free variables left) has its own body's `node_types` already
    // fully resolved — scan it directly for calls into a generic callee,
    // whichever worklist it belongs to.
    for (name, f) in &functions {
        let Some(scheme) = program_inference.global_env.get(*name) else { continue };
        if !scheme.vars.is_empty() {
            continue;
        }
        // `None` for a top-level `fn` that `callgraph::infer_program` itself
        // already rejected (`MissingFnBody`) — such a function never makes
        // it into `global_env` at all, so the `scheme` lookup above would
        // already have skipped it -- *or* for an `extern fn` (see `ast.rs`'s
        // own `FnDecl::is_extern` doc comment), which `callgraph.rs` does
        // seed into `global_env` (so ordinary calls to it resolve), but
        // which has no body of its own to scan for further instantiations.
        let Some(body) = &f.body else { continue };
        collect_instantiations(
            body,
            &program_inference.node_types,
            &program_inference.global_env,
            &templates,
            &mut fn_worklist,
            &mut impl_worklist,
            &mut mono.seed_call_names,
            &mut mono.errors,
        );
    }

    while let Some((name, concrete_tys)) = fn_worklist.pop() {
        let display = display_instantiation(&name, &concrete_tys);
        if mono.specializations.contains_key(&display) {
            continue;
        }
        let (Some(&f), Some(scheme)) = (functions.get(name.as_str()), program_inference.global_env.get(&name)) else {
            continue;
        };
        let Ty::Fn(param_pattern, ret_pattern) = &scheme.ty else {
            continue; // a top-level fn's own scheme is always Ty::Fn — defensive, not expected
        };

        let mapping: HashMap<TyVar, Ty> = scheme.vars.iter().copied().zip(concrete_tys.iter().cloned()).collect();
        let param_types: Vec<Ty> = param_pattern.iter().map(|t| substitute(t, &mapping)).collect();
        let result = substitute(ret_pattern, &mapping);

        let body = f.body.as_ref().expect("a top-level fn with a global_env scheme always has a body");
        let mut exprs = Vec::new();
        collect_exprs_block(body, &mut exprs);
        let node_types: HashMap<NodeId, Ty> = exprs
            .iter()
            .filter_map(|e| program_inference.node_types.get(&e.id).map(|t| (e.id, substitute(t, &mapping))))
            .collect();

        // Scan *this specialization's own* (now fully concrete) node types
        // for further calls into another generic callee — transitive
        // instantiation, exactly like the seed step above. `call_names`
        // here is local to *this one* specialization — see
        // `Specialization::call_names`'s own doc comment for why that
        // matters specifically for a self-recursive call site.
        let mut call_names = HashMap::new();
        collect_instantiations(
            body,
            &node_types,
            &program_inference.global_env,
            &templates,
            &mut fn_worklist,
            &mut impl_worklist,
            &mut call_names,
            &mut mono.errors,
        );

        mono.by_origin.entry(name).or_default().push(display.clone());
        mono.specializations.insert(
            display,
            Specialization { params: f.params.clone(), body: body.clone(), param_types, result, node_types, call_names },
        );
    }

    while let Some((idx, mapping)) = impl_worklist.pop() {
        let t = &templates[idx];
        let display = display_impl_instantiation(t, &mapping);
        if mono.specializations.contains_key(&display) {
            continue;
        }

        let param_types: Vec<Ty> = t.param_patterns.iter().map(|p| substitute(p, &mapping)).collect();
        let result = substitute(&t.ret_pattern, &mapping);

        let mut exprs = Vec::new();
        collect_exprs_block(&t.body, &mut exprs);
        let node_types: HashMap<NodeId, Ty> =
            exprs.iter().filter_map(|e| t.node_types.get(&e.id).map(|ty| (e.id, substitute(ty, &mapping)))).collect();

        let mut call_names = HashMap::new();
        collect_instantiations(
            &t.body,
            &node_types,
            &program_inference.global_env,
            &templates,
            &mut fn_worklist,
            &mut impl_worklist,
            &mut call_names,
            &mut mono.errors,
        );

        let origin = format!("{}::{}", t.algebra, t.method_name);
        mono.by_origin.entry(origin).or_default().push(display.clone());
        mono.specializations.insert(
            display,
            Specialization { params: t.params.clone(), body: t.body.clone(), param_types, result, node_types, call_names },
        );
    }

    (mono, program_inference)
}

/// Builds one `ImplTemplate` per method of every *generic* algebra impl in
/// the program (a non-generic impl, e.g. `impl Ring<i32>`, needs no
/// template at all — it's already fully concrete, rendered unchanged by
/// `dump.rs`'s own existing `--dump-inference-pass`-style path, untouched
/// here). A method whose own declaration-time inference fails is silently
/// skipped — nothing to monomorphize for a method that doesn't type-check;
/// `--dump-inference-pass` is where that failure actually gets reported.
fn build_impl_templates(program: &Program, registry: &Registry, global_env: &Env) -> Vec<ImplTemplate> {
    let mut templates = Vec::new();
    for item in &program.items {
        let ItemKind::Impl(d) = &item.kind else { continue };
        let all_targets: Vec<Type> = std::iter::once(d.target.clone()).chain(d.extra_targets.iter().cloned()).collect();
        let is_generic = !d.generics.is_empty();
        for f in &d.fns {
            // A *generic* impl's own bodyless method (`#[mlir(...)]`-tagged,
            // or otherwise) has no body to substitute a specialization's
            // concrete types into — nothing to monomorphize, skipped.
            // A *concrete* impl's own bodyless method (the immediate,
            // real-primitive case — `impl Ring<f32> { #[mlir(...)] fn
            // add(...); }`) still needs a template built, even with no real
            // body: `derive_impl_instantiation`'s own structural match
            // against `is_generic == false` is what recognizes "a concrete
            // impl already covers this call, no specialization needed" (see
            // `ImplTemplate::is_generic`'s own doc comment) — skipping it
            // here entirely made that impl invisible to that check, so a
            // scalar call inside some *other* generic impl's own body (e.g.
            // `Complex<T>::add`'s own `x.real + y.real`, once monomorphized
            // at `T = f32`) wrongly fell through to `NoneMatched` against
            // the only *visible* (and structurally incompatible) candidate
            // — found by direct testing.
            if f.body.is_none() && is_generic {
                continue;
            }
            let mut infer = Infer::new(registry);
            let Ok(ret_pattern) = infer.infer_impl_fn_generic_with_env(global_env, &d.algebra, &d.generics, &all_targets, f, item.span)
            else {
                continue;
            };
            // Never read for a non-generic template — `derive_impl_
            // instantiation` returns `NoCandidates` the moment it sees
            // `is_generic == false`, before ever touching `body`.
            let body = f.body.clone().unwrap_or(Block { stmts: Vec::new(), tail: None });
            templates.push(ImplTemplate {
                algebra: d.algebra.clone(),
                method_name: f.name.clone(),
                params: f.params.clone(),
                body,
                param_patterns: infer.param_types.clone(),
                ret_pattern,
                target_patterns: infer.target_types.clone(),
                node_types: infer.node_types.clone(),
                is_generic,
            });
        }
    }
    templates
}

/// Walks every `Call` node in `body`; for each whose callee resolves, via
/// `global_env`, to a *generic* top-level `fn` (`scheme.vars` non-empty),
/// derives the concrete instantiation this call site actually used and
/// pushes it onto `fn_worklist`. For a name that *isn't* a top-level `fn`
/// at all, tries every `ImplTemplate` sharing that method name instead —
/// `check_no_overlapping_impls` guarantees at most one can coherently
/// unify against a given concrete query, so the first match found is
/// pushed onto `impl_worklist` and the search stops. If templates existed
/// for that name but *none* matched this specific call's own concrete
/// types, pushes a `MonomorphizationFailed` error instead of silently
/// dropping the call — see `derive_impl_instantiation`'s own doc comment
/// for why that's a real, worth-reporting outcome and not the same as "not
/// an algebra call at all". Either way (on success), records this specific
/// call node's own resolved mangled name into `call_names` (consulted
/// later by `dump_block_with_call_names`).
#[allow(clippy::too_many_arguments)]
fn collect_instantiations(
    body: &Block,
    node_types: &HashMap<NodeId, Ty>,
    global_env: &Env,
    templates: &[ImplTemplate],
    fn_worklist: &mut Vec<(String, Vec<Ty>)>,
    impl_worklist: &mut Vec<(usize, HashMap<TyVar, Ty>)>,
    call_names: &mut HashMap<NodeId, String>,
    errors: &mut Vec<TypeError>,
) {
    let mut exprs = Vec::new();
    collect_exprs_block(body, &mut exprs);
    for e in exprs {
        let ExprKind::Call(path, _, args, ..) = &e.kind else { continue };
        let name = path.segments.join("::");

        if let Some(scheme) = global_env.get(&name) {
            if scheme.vars.is_empty() {
                continue;
            }
            if let Some(concrete_tys) = derive_instantiation(scheme, e, args, node_types) {
                call_names.insert(e.id, display_instantiation(&name, &concrete_tys));
                fn_worklist.push((name, concrete_tys));
            }
            continue;
        }

        match derive_impl_instantiation(templates, &name, e, args, node_types) {
            ImplMatch::Found(idx, mapping) => {
                call_names.insert(e.id, display_impl_instantiation(&templates[idx], &mapping));
                impl_worklist.push((idx, mapping));
            }
            ImplMatch::NoCandidates => {} // not an algebra call, or a non-generic one -- nothing to do here
            ImplMatch::NoneMatched { algebra, tys } => {
                errors.push(TypeError {
                    span: e.span,
                    kind: TypeErrorKind::MonomorphizationFailed { algebra, method: name, tys },
                });
            }
        }
    }
}

/// Recovers the concrete type each of `scheme.vars` was instantiated to at
/// one specific call site — see the module's own doc comment for why this
/// needs no fresh `TyVarGen`/`Infer` at all. Returns `None` if `node_types`
/// is missing an entry it needs (a call inside a function that itself
/// failed to type-check, already excluded upstream — defensive here, not
/// expected to actually trigger) or the shapes genuinely don't unify (would
/// mean the whole-program pass itself was unsound — also not expected).
fn derive_instantiation(scheme: &Scheme, call: &Expr, args: &[Expr], node_types: &HashMap<NodeId, Ty>) -> Option<Vec<Ty>> {
    let arg_tys: Vec<Ty> = args.iter().map(|a| node_types.get(&a.id).cloned()).collect::<Option<_>>()?;
    let ret_ty = node_types.get(&call.id)?.clone();
    let query = Ty::Fn(arg_tys, Box::new(ret_ty));
    let mut trial = Subst::default();
    unify(&mut trial, &scheme.ty, &query).ok()?;
    Some(scheme.vars.iter().map(|v| trial.apply(&Ty::Var(*v))).collect())
}

enum ImplMatch {
    Found(usize, HashMap<TyVar, Ty>),
    /// No `ImplTemplate` shares this method name at all — not an algebra
    /// call in the first place, or a non-generic one (no template is ever
    /// built for those — see `build_impl_templates`) that dispatch will
    /// resolve normally, needing no specialization. Not an error.
    NoCandidates,
    /// At least one template shared this method name, but none of them
    /// unified against this call's own concrete types — a real, worth-
    /// reporting failure (see `derive_impl_instantiation`'s own doc
    /// comment), not silently treated the same as `NoCandidates`.
    NoneMatched { algebra: String, tys: String },
}

/// Like `derive_instantiation`, for the algebra-impl side: tries every
/// template named `method`, unifying its own `(param_patterns) -> ret_
/// pattern` against this call's own concrete `(arg_tys) -> ret_ty`, in one
/// shared trial `Subst` per candidate. On the first (and, per `check_no_
/// overlapping_impls`, only ever possible) match, reads back concrete
/// bindings for *every* free variable the template mentions anywhere —
/// `param_patterns`/`ret_pattern` *and* `target_patterns`, since an
/// algebra generic appearing only in the impl's own target (`C` in `fn
/// mul(a: A, b: B) -> C;`) would otherwise never get a binding at all, the
/// same reasoning `infer_algebra_call`'s own "input vs. output-only
/// generics" split exists for.
///
/// If candidates existed but none matched, that's *usually* a real,
/// surfaced failure (`ImplMatch::NoneMatched`) — found by direct testing
/// (and direct feedback: silently treating it the same as "never called"
/// hid a genuine, pre-existing bug in a stub impl body). Ordinary dispatch
/// (`Infer::dispatch_algebra_call`) only ever needs an impl's own *target*
/// pattern to match — it never re-checks the method's full parameter/
/// return shape the way this does, so a call that type-checks fine under
/// ordinary inference can still fail *here*, honestly, if the impl's own
/// declaration-time inference left its generics over-constrained (a stub
/// body silently merging two generics that should stay independent, say).
///
/// *Not* an error, though, if this call never needed a generic impl in the
/// first place: `TestAlg<i32>` (concrete) and `TestAlg<Complex<T>>`
/// (generic) both declare `add`; an ordinary `add(1, 2)` call correctly
/// dispatches to the *concrete* impl and needs no monomorphization
/// whatsoever. `build_impl_templates` builds a template for *every* impl,
/// concrete or generic (see `ImplTemplate::is_generic`'s own doc comment for
/// why a concrete impl needs one too, not just a name-based short-circuit) —
/// so `add(1, 2)`'s own query structurally matches `TestAlg<i32>`'s own
/// (already fully concrete) template first, and since that template isn't
/// generic, this returns `NoCandidates` rather than `Found`, exactly as if
/// no template existed for it at all.
fn derive_impl_instantiation(
    templates: &[ImplTemplate],
    method: &str,
    call: &Expr,
    args: &[Expr],
    node_types: &HashMap<NodeId, Ty>,
) -> ImplMatch {
    let candidates: Vec<&ImplTemplate> = templates.iter().filter(|t| t.method_name == method).collect();
    if candidates.is_empty() {
        return ImplMatch::NoCandidates;
    }
    let Some(arg_tys): Option<Vec<Ty>> = args.iter().map(|a| node_types.get(&a.id).cloned()).collect() else {
        return ImplMatch::NoCandidates;
    };
    let Some(ret_ty) = node_types.get(&call.id).cloned() else {
        return ImplMatch::NoCandidates;
    };
    let query = Ty::Fn(arg_tys, Box::new(ret_ty));

    for (idx, t) in templates.iter().enumerate() {
        if t.method_name != method || t.param_patterns.len() != args.len() {
            continue;
        }
        let pattern = Ty::Fn(t.param_patterns.clone(), Box::new(t.ret_pattern.clone()));
        let mut trial = Subst::default();
        if unify(&mut trial, &pattern, &query).is_err() {
            continue;
        }
        if !t.is_generic {
            return ImplMatch::NoCandidates;
        }
        let mut vars = HashSet::new();
        t.param_patterns.iter().for_each(|p| free_vars(p, &mut vars));
        free_vars(&t.ret_pattern, &mut vars);
        t.target_patterns.iter().for_each(|p| free_vars(p, &mut vars));
        let mapping: HashMap<TyVar, Ty> = vars.into_iter().map(|v| (v, trial.apply(&Ty::Var(v)))).collect();
        return ImplMatch::Found(idx, mapping);
    }
    ImplMatch::NoneMatched {
        algebra: candidates[0].algebra.clone(),
        tys: query.to_string(),
    }
}

fn display_instantiation(name: &str, tys: &[Ty]) -> String {
    if tys.is_empty() {
        name.to_string()
    } else {
        format!("{name}<{}>", tys.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "))
    }
}

/// The impl-side equivalent of `display_instantiation` — described by its
/// own *target* tuple (`Matrix<f32, 2, 3>, Matrix<f32, 3, 5>, Matrix<f32,
/// 2, 5>`), not by its internal generic names (`T, N, M, K`), since a
/// reader thinks of a `MatMul` call in terms of the operand/result shapes
/// actually involved, not the impl's own declaration.
fn display_impl_instantiation(t: &ImplTemplate, mapping: &HashMap<TyVar, Ty>) -> String {
    let targets = t.target_patterns.iter().map(|p| substitute(p, mapping).to_string()).collect::<Vec<_>>().join(", ");
    format!("{}::{}<{}>", t.algebra, t.method_name, targets)
}

/// Collects every sub-expression of `expr`, including `expr` itself, into
/// `out` — mirrors `callgraph.rs`'s own private `collect_calls_expr`
/// traversal shape (same exhaustive per-`ExprKind` structure), but collects
/// *every* node reference here, not just `Call`s by name, since this is
/// also used to build a specialization's own substituted `node_types`.
fn collect_exprs<'a>(expr: &'a Expr, out: &mut Vec<&'a Expr>) {
    out.push(expr);
    match &expr.kind {
        ExprKind::NumberLit { .. } | ExprKind::ImaginaryLit { .. } | ExprKind::BoolLit(_) | ExprKind::Path(_) => {}
        ExprKind::Call(_, _, args, ..) => args.iter().for_each(|a| collect_exprs(a, out)),
        ExprKind::FieldAccess(base, _) => collect_exprs(base, out),
        ExprKind::MethodCall(base, _, args) => {
            collect_exprs(base, out);
            args.iter().for_each(|a| collect_exprs(a, out));
        }
        ExprKind::Index(base, idx) => {
            collect_exprs(base, out);
            collect_exprs(idx, out);
        }
        ExprKind::ArrayLit(elems) => elems.iter().for_each(|e| collect_exprs(e, out)),
        ExprKind::ArrayRepeat { value, count } => {
            collect_exprs(value, out);
            collect_exprs(count, out);
        }
        ExprKind::StructLit(_, _, fields) => fields.iter().for_each(|(_, v)| collect_exprs(v, out)),
        ExprKind::If { cond, then_branch, else_branch } => {
            collect_exprs(cond, out);
            collect_exprs_block(then_branch, out);
            if let Some(eb) = else_branch {
                match &**eb {
                    ElseBranch::If(e) => collect_exprs(e, out),
                    ElseBranch::Block(b) => collect_exprs_block(b, out),
                }
            }
        }
        ExprKind::While { cond, body } => {
            collect_exprs(cond, out);
            collect_exprs_block(body, out);
        }
        ExprKind::For { start, end, body, .. } => {
            collect_exprs(start, out);
            collect_exprs(end, out);
            collect_exprs_block(body, out);
        }
        ExprKind::Block(b) => collect_exprs_block(b, out),
        ExprKind::Lambda { body, .. } => collect_exprs_block(body, out),
    }
}

fn collect_exprs_block<'a>(block: &'a Block, out: &mut Vec<&'a Expr>) {
    for stmt in &block.stmts {
        match &stmt.kind {
            StmtKind::Let { value, .. } => collect_exprs(value, out),
            StmtKind::Assign { target, value } => {
                collect_exprs(target, out);
                collect_exprs(value, out);
            }
            StmtKind::Expr(e) => collect_exprs(e, out),
        }
    }
    if let Some(tail) = &block.tail {
        collect_exprs(tail, out);
    }
}

// ------------------------------------------------------------ rendering

/// Renders the whole monomorphized program — every non-generic top-level
/// `fn` unchanged, every generic one replaced by *all* of its concrete
/// specializations actually reached (the generic declaration itself is
/// never shown standalone, mirroring real monomorphization: it isn't
/// directly callable once nothing consumes generics anymore). A generic
/// algebra impl gets the identical treatment, flattened out of its own
/// `impl` block into standalone, fully-qualified specializations — a
/// non-generic impl (`impl Ring<i32>`), needing no specialization at all,
/// still renders its own real, type-checked body inline, same as `--dump-
/// inference-pass`. `struct`/`algebra` items, and *inherent* impls (not
/// attempted this increment — see the module's own doc comment), still
/// render as bare markers.
pub fn dump_monomorphized(program: &Program, registry: &Registry) -> (String, Vec<TypeError>) {
    let (mono, program_inference) = monomorphize(program, registry);
    let mut out = String::new();
    let mut errors = Vec::new();

    for (i, item) in program.items.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        match &item.kind {
            ItemKind::Use(path) => {
                let _ = writeln!(out, "use {};", path.segments.join("::"));
            }
            ItemKind::Struct(d) => {
                let _ = writeln!(out, "struct {} {{ /* not type-inferred yet */ }}", d.name);
            }
            ItemKind::Algebra(d) => {
                let _ = writeln!(out, "algebra {} {{ /* not type-inferred yet */ }}", d.name);
            }
            ItemKind::Impl(d) => {
                if d.generics.is_empty() {
                    dump_concrete_impl(&mut out, &mut errors, d, item.span, registry, &program_inference.global_env);
                    continue;
                }
                for f in &d.fns {
                    let keys = mono.specializations_of(&format!("{}::{}", d.algebra, f.name));
                    if keys.is_empty() {
                        let _ = writeln!(
                            out,
                            "// `{}::{}` is generic but was never called from a concrete entry point -- no specialization to show",
                            d.algebra, f.name
                        );
                    }
                    for k in keys {
                        dump_one(
                            &mut out,
                            k,
                            mono.params(k),
                            mono.body(k),
                            mono.param_types(k),
                            mono.result(k),
                            mono.node_types(k),
                            mono.call_names(k),
                        );
                    }
                }
            }
            ItemKind::InherentImpl(d) => {
                let _ = writeln!(
                    out,
                    "impl {} {{ /* generic-impl monomorphization not attempted yet */ }}",
                    crate::print::fmt_type(&d.target)
                );
            }
            ItemKind::Fn(f) => match program_inference.results.get(&f.name) {
                Some(Err(e)) => {
                    let params: Vec<String> = f.params.iter().map(|p| p.name.clone()).collect();
                    let _ = writeln!(out, "fn {}({}) {{ /* type error, see diagnostics */ }}", f.name, params.join(", "));
                    errors.push(e.clone());
                }
                Some(Ok(fn_result)) => match program_inference.global_env.get(&f.name) {
                    Some(scheme) if scheme.vars.is_empty() => match &f.body {
                        Some(body) => {
                            dump_one(
                                &mut out,
                                &f.name,
                                &f.params,
                                body,
                                &fn_result.param_types,
                                &fn_result.result,
                                &program_inference.node_types,
                                mono.seed_call_names(),
                            );
                        }
                        // `extern fn` — no body to dump; render its resolved
                        // signature instead (see `ast.rs`'s own `FnDecl::
                        // is_extern` doc comment).
                        None => {
                            let params: Vec<String> = f.params.iter().map(|p| p.name.clone()).collect();
                            let _ = writeln!(out, "extern fn {}({}) -> {};", f.name, params.join(", "), fn_result.result);
                        }
                    },
                    Some(_) => {
                        let keys = mono.specializations_of(&f.name);
                        if keys.is_empty() {
                            let _ = writeln!(
                                out,
                                "// `{}` is generic but was never called from a concrete entry point -- no specialization to show",
                                f.name
                            );
                        }
                        for key in keys {
                            dump_one(
                                &mut out,
                                key,
                                mono.params(key),
                                mono.body(key),
                                mono.param_types(key),
                                mono.result(key),
                                mono.node_types(key),
                                mono.call_names(key),
                            );
                        }
                    }
                    None => unreachable!("`{}` type-checked successfully but has no scheme in global_env", f.name),
                },
                None => unreachable!("`{}` is a top-level `fn` item but callgraph::infer_program has no entry for it", f.name),
            },
        }
    }

    // `MonomorphizationFailed` errors found while walking either worklist
    // (a call site whose concrete types no candidate impl template could
    // actually be instantiated at) — see `derive_impl_instantiation`'s own
    // doc comment. Not tied to any one `program.items` entry the loop above
    // already visits, so appended here rather than folded into it.
    errors.extend(mono.errors().iter().cloned());

    (out, errors)
}

/// A non-generic algebra impl (`impl Ring<i32>`) needs no specialization at
/// all — rendered with its own real, type-checked body directly, the same
/// way `dump.rs`'s own `--dump-inference-pass` already does, since nothing
/// here can improve on an already-fully-concrete method.
fn dump_concrete_impl(out: &mut String, errors: &mut Vec<TypeError>, d: &ImplDecl, span: Span, registry: &Registry, global_env: &Env) {
    let targets: Vec<String> = std::iter::once(&d.target).chain(d.extra_targets.iter()).map(crate::print::fmt_type).collect();
    let _ = writeln!(out, "impl {}<{}> {{", d.algebra, targets.join(", "));
    let all_targets: Vec<Type> = std::iter::once(d.target.clone()).chain(d.extra_targets.iter().cloned()).collect();
    for f in &d.fns {
        let mut infer = Infer::new(registry);
        match infer.infer_impl_fn_generic_with_env(global_env, &d.algebra, &d.generics, &all_targets, f, span) {
            Ok(ret) => match &f.body {
                Some(body) => dump_one(out, &f.name, &f.params, body, &infer.param_types, &ret, &infer.node_types, &HashMap::new()),
                // A bodyless method (`#[mlir(...)]`-tagged) that type-checked
                // successfully — rendered as a bare signature, same as
                // `dump.rs`'s own `dump_impl_fn`.
                None => {
                    let mut names = TyVarNames::default();
                    let rendered_params: Vec<String> = f
                        .params
                        .iter()
                        .zip(infer.param_types.iter())
                        .map(|(p, t)| format!("{}: {}", p.name, fmt_ty_named(t, &mut names)))
                        .collect();
                    let ret = fmt_ty_named(&ret, &mut names);
                    for attr in &f.attrs {
                        let _ = writeln!(out, "#[{}({})]", attr.name, attr.args.join(", "));
                    }
                    let _ = writeln!(out, "fn {}({}) -> {ret};", f.name, rendered_params.join(", "));
                }
            },
            Err(e) => {
                let params: Vec<String> = f.params.iter().map(|p| p.name.clone()).collect();
                let _ = writeln!(out, "fn {}({}) {{ /* type error, see diagnostics */ }}", f.name, params.join(", "));
                errors.push(e);
            }
        }
    }
    let _ = writeln!(out, "}}");
}

#[allow(clippy::too_many_arguments)]
fn dump_one(
    out: &mut String,
    mangled_name: &str,
    params: &[Param],
    body: &Block,
    param_types: &[Ty],
    result: &Ty,
    node_types: &HashMap<NodeId, Ty>,
    call_names: &HashMap<NodeId, String>,
) {
    let mut names = TyVarNames::default();
    let rendered_params: Vec<String> =
        params.iter().zip(param_types.iter()).map(|(p, t)| format!("{}: {}", p.name, fmt_ty_named(t, &mut names))).collect();
    let ret = fmt_ty_named(result, &mut names);
    let _ = writeln!(out, "fn {}({}) -> {ret} {{", mangled_name, rendered_params.join(", "));
    dump_block_with_call_names(out, body, node_types, &mut names, 1, call_names);
    let _ = writeln!(out, "}}");
}
