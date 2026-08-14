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
use crate::infer::{find_placeholder_name, free_vars, substitute, unify, ConstValue, Env, Infer, Scheme, Subst, Ty, TyVar, TypeError, TypeErrorKind};
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
    /// Whether this template's own resolved `param_patterns`/`ret_pattern`/
    /// `target_patterns` carry any free variable at all — `false` for
    /// `impl MatMul<f32,f32,f32>`, `true` for `impl<T,N,M,K>
    /// MatMul<Matrix<T,N,M>,...>` (the impl's own declared generics, the
    /// overwhelmingly common source), but *also* `true` for a syntactically
    /// non-generic impl (`impl Sum<i32> { ... }`) whose method still
    /// inherits a free variable from the *algebra's* own const generic
    /// (`algebra Sum<T, const N: i32> { fn total(x: [T; N]) -> T; }` — `N`
    /// is never fixed by which impl matched, only by the call site) — found
    /// missing by direct testing, see `build_impl_templates`'s own computation
    /// of this field for the full story. A truly concrete impl's own
    /// `param_patterns`/`ret_pattern` carry no free variables at all
    /// (already fully resolved against its own concrete targets) — its
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

/// A generic inherent impl's own declaration-time "template" for one
/// method — the inherent-impl counterpart to `ImplTemplate`, simpler in one
/// respect: `Registry::inherent_method`'s own doc comment guarantees at
/// most one method of a given name exists per struct, so there's no
/// candidate *search* needed at a call site, only a direct `(struct_name,
/// method_name)` lookup — no `is_generic`/`ImplMatch`-style "found but none
/// matched" distinction either, a non-generic inherent impl never builds a
/// template at all (mirrors `build_impl_templates`'s own treatment of a
/// concrete algebra impl, one level simpler: nothing here needs to
/// structurally *recognize* "a concrete impl already covers this," since a
/// method name can only ever belong to the one impl block that declared
/// it). No separate `target_patterns` field either — a method's own first
/// parameter's pattern (`param_patterns[0]`) already *is* the impl's own
/// target pattern: `inherent_method_param_tys` sets an unannotated first
/// parameter to `target_ty` directly, and unifies an annotated one against
/// it, so the two are never independently free variables to track twice.
struct InherentTemplate {
    struct_name: String,
    method_name: String,
    params: Vec<Param>,
    body: Block,
    param_patterns: Vec<Ty>,
    ret_pattern: Ty,
    node_types: HashMap<NodeId, Ty>,
}

/// Builds one `InherentTemplate` per method of every *generic* inherent
/// impl in the program — mirrors `build_impl_templates`'s own doc comment,
/// one level simpler (see `InherentTemplate`'s own doc comment for why). A
/// whole impl block's own methods are inferred together, sharing one
/// `Infer` (`infer_inherent_impl_block`, real mutual recursion between
/// sibling methods) — each method's own template shares that same block's
/// `node_types`, filtered down to its own body's nodes only, same as any
/// other template/specialization here. A method whose own declaration-time
/// inference failed is silently skipped, same reasoning as `build_impl_
/// templates`.
fn build_inherent_templates(program: &Program, registry: &Registry, global_env: &Env) -> Vec<InherentTemplate> {
    let mut templates = Vec::new();
    for item in &program.items {
        let ItemKind::InherentImpl(d) = &item.kind else { continue };
        if d.generics.is_empty() {
            continue;
        }
        let TypeKind::Path(p, _) = &d.target.kind else { continue };
        let struct_name = p.segments.join("::");
        let mut infer = Infer::new(registry);
        let (_, results) = infer.infer_inherent_impl_block(global_env, &d.generics, &d.target, &d.fns, item.span);
        for f in &d.fns {
            let Some(Ok((param_patterns, ret_pattern))) = results.get(&f.name) else { continue };
            templates.push(InherentTemplate {
                struct_name: struct_name.clone(),
                method_name: f.name.clone(),
                params: f.params.clone(),
                body: f.body.clone().unwrap_or(Block { stmts: Vec::new(), tail: None }),
                param_patterns: param_patterns.clone(),
                ret_pattern: ret_pattern.clone(),
                node_types: infer.node_types.clone(),
            });
        }
    }
    templates
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
    let inherent_templates = build_inherent_templates(program, registry, &program_inference.global_env);
    let lambda_exprs = index_lambda_exprs(program, &program_inference.lambda_schemes);
    let duck_typed_fns = detect_duck_typed_fns(&functions, &program_inference);

    let mut mono = MonomorphizedProgram {
        specializations: HashMap::new(),
        by_origin: HashMap::new(),
        seed_call_names: HashMap::new(),
        errors: Vec::new(),
    };
    let mut fn_worklist: Vec<(String, Vec<Ty>)> = Vec::new();
    let mut impl_worklist: Vec<(usize, HashMap<TyVar, Ty>)> = Vec::new();
    let mut lambda_worklist: Vec<(NodeId, Vec<Ty>, String)> = Vec::new();
    let mut inherent_worklist: Vec<(usize, HashMap<TyVar, Ty>)> = Vec::new();

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
            &inherent_templates,
            &program_inference.lambda_schemes,
            HashMap::new(),
            &mut fn_worklist,
            &mut impl_worklist,
            &mut lambda_worklist,
            &mut inherent_worklist,
            &mut mono.seed_call_names,
            &mut mono.errors,
        );
    }

    while let Some((name, concrete_tys)) = fn_worklist.pop() {
        let display = display_instantiation(&name, &concrete_tys);
        if mono.specializations.contains_key(&display) {
            continue;
        }
        let Some(&f) = functions.get(name.as_str()) else { continue };
        let body = f.body.as_ref().expect("a top-level fn with a global_env scheme always has a body");

        let (param_types, result, node_types) = if duck_typed_fns.contains(&name) {
            // Duck-typed fallback (`detect_duck_typed_fns`'s own doc
            // comment) — a real, separate re-inference for this one
            // concrete call site, not substitution over the shared
            // declaration-time template: substitution can never resolve an
            // expression (field access, ...) whose own type genuinely
            // depends on this call site's own concrete argument types,
            // which the ordinary one-shot HM pass could never see.
            let mut infer = Infer::new(registry);
            match infer.infer_fn_with_concrete_params(f, concrete_tys.clone()) {
                Ok(result) => {
                    let mut exprs = Vec::new();
                    collect_exprs_block(body, &mut exprs);
                    let node_types: HashMap<NodeId, Ty> =
                        exprs.iter().filter_map(|e| infer.node_types.get(&e.id).map(|t| (e.id, t.clone()))).collect();
                    (infer.param_types.clone(), result, node_types)
                }
                Err(e) => {
                    let tys = concrete_tys.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", ");
                    mono.errors.push(TypeError {
                        span: e.span,
                        kind: TypeErrorKind::GenericFnInstantiationFailed { name: name.clone(), tys, inner: Box::new(e) },
                    });
                    continue;
                }
            }
        } else {
            let Some(scheme) = program_inference.global_env.get(&name) else { continue };
            let Ty::Fn(param_pattern, ret_pattern) = &scheme.ty else {
                continue; // a top-level fn's own scheme is always Ty::Fn — defensive, not expected
            };
            let mapping: HashMap<TyVar, Ty> = scheme.vars.iter().copied().zip(concrete_tys.iter().cloned()).collect();
            let param_types: Vec<Ty> = param_pattern.iter().map(|t| substitute(t, &mapping)).collect();
            let result = substitute(ret_pattern, &mapping);
            let mut exprs = Vec::new();
            collect_exprs_block(body, &mut exprs);
            let node_types: HashMap<NodeId, Ty> = exprs
                .iter()
                .filter_map(|e| program_inference.node_types.get(&e.id).map(|t| (e.id, substitute(t, &mapping))))
                .collect();
            (param_types, result, node_types)
        };

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
            &inherent_templates,
            &program_inference.lambda_schemes,
            HashMap::new(),
            &mut fn_worklist,
            &mut impl_worklist,
            &mut lambda_worklist,
            &mut inherent_worklist,
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
            &inherent_templates,
            &program_inference.lambda_schemes,
            HashMap::new(),
            &mut fn_worklist,
            &mut impl_worklist,
            &mut lambda_worklist,
            &mut inherent_worklist,
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

    // Lambda worklist -- structurally identical to the top-level-`fn`
    // worklist just above (same `Specialization` shape, same reverse-
    // unification via `derive_instantiation`), but a lambda has no top-
    // level `FnDecl`/`global_env` entry to read `params`/`body`/`scheme`
    // back from -- `lambda_exprs`/`program_inference.lambda_schemes`
    // (built/aggregated once, up front) stand in for those. Unlike a top-
    // level generic `fn`, a lambda's own body `node_types` were never given
    // a dedicated per-declaration template (no `ImplTemplate`-style struct
    // needed) -- they're read directly out of the *whole-program*
    // `program_inference.node_types`, exactly the same map (and the exact
    // same reasoning) the `fn_worklist` loop above already reads its own
    // generic pattern from, since ordinary inference records a lambda
    // body's node types there too, just still generic (pre-instantiation).
    while let Some((lambda_id, concrete_tys, self_name)) = lambda_worklist.pop() {
        let display = display_lambda_instantiation(lambda_id, &concrete_tys);
        if mono.specializations.contains_key(&display) {
            continue;
        }
        let (Some(scheme), Some(&lambda_expr)) =
            (program_inference.lambda_schemes.get(&lambda_id), lambda_exprs.get(&lambda_id))
        else {
            continue;
        };
        let ExprKind::Lambda { params, body, .. } = &lambda_expr.kind else {
            continue; // `lambda_exprs` only ever indexes `Lambda` nodes -- defensive, not expected
        };
        let Ty::Fn(param_pattern, ret_pattern) = &scheme.ty else {
            continue; // a lambda's own scheme is always Ty::Fn, mirroring a top-level fn's -- defensive
        };

        let mapping: HashMap<TyVar, Ty> = scheme.vars.iter().copied().zip(concrete_tys.iter().cloned()).collect();
        let param_types: Vec<Ty> = param_pattern.iter().map(|t| substitute(t, &mapping)).collect();
        let result = substitute(ret_pattern, &mapping);

        let mut exprs = Vec::new();
        collect_exprs_block(body, &mut exprs);
        let node_types: HashMap<NodeId, Ty> = exprs
            .iter()
            .filter_map(|e| program_inference.node_types.get(&e.id).map(|t| (e.id, substitute(t, &mapping))))
            .collect();

        let mut call_names = HashMap::new();
        // Seeded with this lambda's own canonical self-name (recovered
        // above from `scope`, at whichever call site originally discovered
        // this specialization -- see `collect_instantiations_expr`'s own
        // `ExprKind::Call` arm) -- otherwise this re-walk, starting fresh,
        // could never resolve a self-recursive call inside `body` at all.
        let mut initial_scope = HashMap::new();
        initial_scope.insert(self_name, lambda_id);
        collect_instantiations(
            body,
            &node_types,
            &program_inference.global_env,
            &templates,
            &inherent_templates,
            &program_inference.lambda_schemes,
            initial_scope,
            &mut fn_worklist,
            &mut impl_worklist,
            &mut lambda_worklist,
            &mut inherent_worklist,
            &mut call_names,
            &mut mono.errors,
        );

        let origin = format!("<lambda#{}>", lambda_id.0);
        mono.by_origin.entry(origin).or_default().push(display.clone());
        mono.specializations.insert(
            display,
            Specialization { params: params.clone(), body: body.clone(), param_types, result, node_types, call_names },
        );
    }

    // Inherent-method worklist -- structurally identical to the impl_
    // worklist loop above (same `Specialization` shape, same reverse-
    // unification via `derive_inherent_instantiation`), just reading back
    // from `InherentTemplate` instead of `ImplTemplate` and keying `by_
    // origin` as `"struct::method"` (matching `cps.rs::collect_units`'s own
    // `InherentImpl` branch, which reads specializations back by that exact
    // key).
    while let Some((idx, mapping)) = inherent_worklist.pop() {
        let t = &inherent_templates[idx];
        let display = display_inherent_instantiation(t, &mapping);
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
            &inherent_templates,
            &program_inference.lambda_schemes,
            HashMap::new(),
            &mut fn_worklist,
            &mut impl_worklist,
            &mut lambda_worklist,
            &mut inherent_worklist,
            &mut call_names,
            &mut mono.errors,
        );

        let origin = format!("{}::{}", t.struct_name, t.method_name);
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
/// Scans every generic top-level `fn` for placeholder residue anywhere in
/// its own body's declaration-time `node_types` — see `infer.rs`'s own
/// `is_placeholder` doc comment for what counts (`<not-yet-inferred>`,
/// `<unresolved-call:...>`, ...). A fn found here can never be correctly
/// specialized by the ordinary substitution path in the `fn_worklist` loop
/// below — substitution only ever replaces a `Ty::Var`, and a placeholder is
/// a `Ty::Con`, permanently baked in by the one-shot HM pass
/// (`callgraph::infer_program`) regardless of a later call site's own
/// concrete types. Instead, the `fn_worklist` loop routes anything found
/// here through `Infer::infer_fn_with_concrete_params` — a real, separate
/// re-inference per concrete call site, C++-templates-style, deliberately
/// *not* HM's own "checked once, sound everywhere" discipline (a second,
/// coexisting mechanism, not a replacement — see this feature's own commit/
/// discussion for why). Scoped to top-level `fn`s only: a generic algebra-
/// impl method's own signature is always fully, explicitly declared by its
/// `algebra`, so it never has an unconstrained parameter to trigger this in
/// the first place.
fn detect_duck_typed_fns(functions: &HashMap<&str, &FnDecl>, program_inference: &ProgramInference) -> HashSet<String> {
    let mut out = HashSet::new();
    for (name, f) in functions {
        let Some(scheme) = program_inference.global_env.get(*name) else { continue };
        if scheme.vars.is_empty() {
            continue;
        }
        let Some(body) = &f.body else { continue };
        let mut exprs = Vec::new();
        collect_exprs_block(body, &mut exprs);
        let has_placeholder =
            exprs.iter().any(|e| program_inference.node_types.get(&e.id).is_some_and(|t| find_placeholder_name(t).is_some()));
        if has_placeholder {
            out.insert((*name).to_string());
        }
    }
    out
}

fn build_impl_templates(program: &Program, registry: &Registry, global_env: &Env) -> Vec<ImplTemplate> {
    let mut templates = Vec::new();
    for item in &program.items {
        let ItemKind::Impl(d) = &item.kind else { continue };
        let all_targets: Vec<Type> = std::iter::once(d.target.clone()).chain(d.extra_targets.iter().cloned()).collect();
        let is_generic = !d.generics.is_empty();
        for f in &d.fns {
            // A bodyless method (extern-backed, or the old `#[mlir(...)]`
            // intrinsic tag) never needs body-substitution — nothing here
            // depends on that distinction, whether the impl itself is
            // generic or not: a template still gets built either way (see
            // `body` below, `unwrap_or`-defaulted to empty), only the
            // *body-substitution machinery* stays inert. A *generic*
            // extern-backed impl (`impl<const N: i32> Print<[i8; N]> {
            // extern(print_bytes) fn print(x: [i8; N]) -> [i8; N]; }`, the
            // first of its kind in this codebase — found missing by direct
            // testing, not by reading) still needs a real template: the
            // concrete `N` a given call site reaches is exactly what
            // `derive_impl_instantiation`/`call_names` exist to record, an
            // extern-backed method needs that as much as a real one does,
            // even though there's no cleave-level body to specialize.
            let mut infer = Infer::new(registry);
            let Ok(ret_pattern) = infer.infer_impl_fn_generic_with_env(global_env, &d.algebra, &d.generics, &all_targets, f, item.span)
            else {
                continue;
            };
            // Whether *this template's own resolved patterns* still carry a
            // free variable — not just whether the *impl* itself declared
            // generics (`!d.generics.is_empty()` alone, this method's own
            // original check): an impl with zero generics of its own
            // (`impl Sum<i32> { fn total(x) -> i32 { ... } }`) can still
            // inherit a free variable from the *algebra's* own const
            // generic (`algebra Sum<T, const N: i32> { fn total(x: [T; N])
            // -> T; }` — `N` maps to a fresh var in `infer_impl_fn_generic_
            // with_env`, never fixed by which impl matched, only by
            // whichever concrete call site this method's own specialization
            // is eventually built for) — found by direct testing once a
            // real const-generic-algebra call actually ran: treating this
            // template as non-generic left `N` permanently unresolved, and
            // `resolve_call` could never find a matching concrete unit for
            // it. `derive_impl_instantiation` already gathers free vars from
            // exactly these three patterns unconditionally once `is_generic`
            // is true (see its own doc comment) — no other change needed.
            let mut free = HashSet::new();
            infer.param_types.iter().for_each(|p| free_vars(p, &mut free));
            free_vars(&ret_pattern, &mut free);
            infer.target_types.iter().for_each(|p| free_vars(p, &mut free));
            let is_generic = is_generic || !free.is_empty();
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

/// Every `Lambda` expression, anywhere in any top-level `fn`'s body, that
/// `infer.rs` actually generalized (has a `LambdaScheme` entry) — indexed
/// once, up front, so the lambda-instantiation worklist (see `monomorphize`)
/// can read a lambda's own `params`/`body` back from just its `NodeId`, the
/// same way the fn/impl worklists already read theirs back from `functions`/
/// `templates`. A lambda with no scheme entry (`let mut`-bound, so never
/// generalized — see `Infer::lambda_schemes`'s own doc comment) is simply
/// invisible here: nothing can specialize it via this mechanism, and no
/// call to it is ever recognized as a lambda call either (`collect_
/// instantiations_block`'s own scope only ever admits scheme-bearing ones).
fn index_lambda_exprs<'a>(program: &'a Program, lambda_schemes: &HashMap<NodeId, Scheme>) -> HashMap<NodeId, &'a Expr> {
    let mut out = HashMap::new();
    for item in &program.items {
        let ItemKind::Fn(f) = &item.kind else { continue };
        let Some(body) = &f.body else { continue };
        let mut exprs = Vec::new();
        collect_exprs_block(body, &mut exprs);
        for e in exprs {
            if matches!(e.kind, ExprKind::Lambda { .. }) && lambda_schemes.contains_key(&e.id) {
                out.insert(e.id, e);
            }
        }
    }
    out
}

/// Walks every `Call` node in `body`, tracking a shadowing-aware scope of
/// which local names are currently `let`-bound to a (scheme-bearing) lambda
/// — same walk shape, same shadowing discipline, as `cps.rs`'s own
/// `mutated_free_vars`/`mutated_free_vars_expr` pair (see their doc
/// comments); the two problems are structurally identical ("which local
/// binding does this name currently refer to, respecting nested-scope
/// shadowing"), just answering a different question about it.
///
/// For each `Call`, checked in this order:
/// 1. Does the callee name resolve to a lambda currently in scope? If so,
///    and it unifies (`derive_instantiation`, the exact same reverse-
///    unification a top-level generic `fn` call already uses — a lambda's
///    own `Scheme` has the identical `Ty::Fn(params, ret)` shape), push
///    onto `lambda_worklist`. Checked *first*, deliberately: a local lambda
///    binding shadowing a same-named top-level `fn` must resolve to the
///    lambda, not the `fn` (`let f = fn(x){x}; f(5)` even when a top-level
///    `fn f` also exists).
/// 2. Otherwise, does it resolve, via `global_env`, to a *generic*
///    top-level `fn`? Push onto `fn_worklist`.
/// 3. Otherwise, try every `ImplTemplate` sharing that method name.
///    `check_no_overlapping_impls` guarantees at most one can coherently
///    unify against a given concrete query, so the first match found is
///    pushed onto `impl_worklist` and the search stops. If templates
///    existed for that name but *none* matched this specific call's own
///    concrete types, pushes a `MonomorphizationFailed` error instead of
///    silently dropping the call — see `derive_impl_instantiation`'s own
///    doc comment for why that's a real, worth-reporting outcome.
///
/// Any of the three, on success, records this specific call node's own
/// resolved mangled name into `call_names` (consulted later by both
/// `dump_block_with_call_names` and `cps.rs`'s own call resolution).
#[allow(clippy::too_many_arguments)]
fn collect_instantiations(
    body: &Block,
    node_types: &HashMap<NodeId, Ty>,
    global_env: &Env,
    templates: &[ImplTemplate],
    inherent_templates: &[InherentTemplate],
    lambda_schemes: &HashMap<NodeId, Scheme>,
    // Non-empty only when re-walking a lambda specialization's own body
    // from the `lambda_worklist` drain loop -- seeded with that lambda's
    // own canonical self-name, so a self-recursive call site inside it can
    // resolve the same way the initial (whole-function) scan already does.
    // Empty for every other caller (the seed scan, and the fn/impl/
    // inherent-worklist drain loops), matching this function's own prior
    // always-empty behavior for them.
    initial_scope: HashMap<String, NodeId>,
    fn_worklist: &mut Vec<(String, Vec<Ty>)>,
    impl_worklist: &mut Vec<(usize, HashMap<TyVar, Ty>)>,
    lambda_worklist: &mut Vec<(NodeId, Vec<Ty>, String)>,
    inherent_worklist: &mut Vec<(usize, HashMap<TyVar, Ty>)>,
    call_names: &mut HashMap<NodeId, String>,
    errors: &mut Vec<TypeError>,
) {
    let scope = initial_scope;
    collect_instantiations_block(
        body,
        node_types,
        global_env,
        templates,
        inherent_templates,
        lambda_schemes,
        &scope,
        fn_worklist,
        impl_worklist,
        lambda_worklist,
        inherent_worklist,
        call_names,
        errors,
    );
}

#[allow(clippy::too_many_arguments)]
fn collect_instantiations_block(
    block: &Block,
    node_types: &HashMap<NodeId, Ty>,
    global_env: &Env,
    templates: &[ImplTemplate],
    inherent_templates: &[InherentTemplate],
    lambda_schemes: &HashMap<NodeId, Scheme>,
    scope: &HashMap<String, NodeId>,
    fn_worklist: &mut Vec<(String, Vec<Ty>)>,
    impl_worklist: &mut Vec<(usize, HashMap<TyVar, Ty>)>,
    lambda_worklist: &mut Vec<(NodeId, Vec<Ty>, String)>,
    inherent_worklist: &mut Vec<(usize, HashMap<TyVar, Ty>)>,
    call_names: &mut HashMap<NodeId, String>,
    errors: &mut Vec<TypeError>,
) {
    let mut scope = scope.clone();
    for stmt in &block.stmts {
        match &stmt.kind {
            StmtKind::Let { name, value, .. } => {
                // Self-recursion (`let fact = fn(n) { ... fact(n - 1) ... };`)
                // -- seeded *before* walking `value`, not after, so a self-
                // call inside the lambda's own body (reached via this same
                // walk, through the `ExprKind::Lambda` arm below) can
                // already resolve `name` via `scope`. Only the *insert*
                // branch moves earlier: the *removal* branch (a non-lambda
                // rebinding) must stay after the walk, since `let f = f(1);`
                // legitimately means "call the outer `f`" and must keep
                // resolving that way.
                let is_lambda = lambda_schemes.contains_key(&value.id);
                if is_lambda {
                    scope.insert(name.clone(), value.id);
                }
                collect_instantiations_expr(
                    value,
                    node_types,
                    global_env,
                    templates,
                    inherent_templates,
                    lambda_schemes,
                    &scope,
                    fn_worklist,
                    impl_worklist,
                    lambda_worklist,
                    inherent_worklist,
                    call_names,
                    errors,
                );
                if !is_lambda {
                    // Re-`let`-bound to something else (or to an
                    // un-generalized lambda) -- shadows any outer lambda
                    // binding of the same name for the rest of this scope.
                    scope.remove(name);
                }
            }
            StmtKind::Assign { target, value } => {
                collect_instantiations_expr(
                    target,
                    node_types,
                    global_env,
                    templates,
                    inherent_templates,
                    lambda_schemes,
                    &scope,
                    fn_worklist,
                    impl_worklist,
                    lambda_worklist,
                    inherent_worklist,
                    call_names,
                    errors,
                );
                collect_instantiations_expr(
                    value,
                    node_types,
                    global_env,
                    templates,
                    inherent_templates,
                    lambda_schemes,
                    &scope,
                    fn_worklist,
                    impl_worklist,
                    lambda_worklist,
                    inherent_worklist,
                    call_names,
                    errors,
                );
            }
            StmtKind::Expr(e) => collect_instantiations_expr(
                e,
                node_types,
                global_env,
                templates,
                inherent_templates,
                lambda_schemes,
                &scope,
                fn_worklist,
                impl_worklist,
                lambda_worklist,
                inherent_worklist,
                call_names,
                errors,
            ),
        }
    }
    if let Some(tail) = &block.tail {
        collect_instantiations_expr(
            tail,
            node_types,
            global_env,
            templates,
            inherent_templates,
            lambda_schemes,
            &scope,
            fn_worklist,
            impl_worklist,
            lambda_worklist,
            inherent_worklist,
            call_names,
            errors,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_instantiations_expr(
    expr: &Expr,
    node_types: &HashMap<NodeId, Ty>,
    global_env: &Env,
    templates: &[ImplTemplate],
    inherent_templates: &[InherentTemplate],
    lambda_schemes: &HashMap<NodeId, Scheme>,
    scope: &HashMap<String, NodeId>,
    fn_worklist: &mut Vec<(String, Vec<Ty>)>,
    impl_worklist: &mut Vec<(usize, HashMap<TyVar, Ty>)>,
    lambda_worklist: &mut Vec<(NodeId, Vec<Ty>, String)>,
    inherent_worklist: &mut Vec<(usize, HashMap<TyVar, Ty>)>,
    call_names: &mut HashMap<NodeId, String>,
    errors: &mut Vec<TypeError>,
) {
    macro_rules! rec {
        ($e:expr) => {
            collect_instantiations_expr(
                $e,
                node_types,
                global_env,
                templates,
                inherent_templates,
                lambda_schemes,
                scope,
                fn_worklist,
                impl_worklist,
                lambda_worklist,
                inherent_worklist,
                call_names,
                errors,
            )
        };
    }
    macro_rules! rec_block {
        ($b:expr, $scope:expr) => {
            collect_instantiations_block(
                $b,
                node_types,
                global_env,
                templates,
                inherent_templates,
                lambda_schemes,
                $scope,
                fn_worklist,
                impl_worklist,
                lambda_worklist,
                inherent_worklist,
                call_names,
                errors,
            )
        };
    }
    match &expr.kind {
        ExprKind::NumberLit { .. } | ExprKind::ImaginaryLit { .. } | ExprKind::BoolLit(_) | ExprKind::Path(_) => {}
        ExprKind::Call(path, _, args, ..) => {
            // A callable passed as a bare argument (`apply(inc, 5)`) — see
            // `derive_value_instantiation`'s own doc comment for why this
            // can't just fall out of the ordinary recursive `rec!(a)` walk
            // just below (a `Path` argument is otherwise a structural
            // no-op). Checked for *every* argument, not just ones a later
            // pass (`cps.rs`'s own Stage B) will actually treat as higher-
            // order — cheap, and correctly a no-op (`scope.get` misses)
            // for an ordinary, non-lambda-bound argument.
            for a in args {
                let ExprKind::Path(p) = &a.kind else { continue };
                let arg_name = p.segments.join("::");
                let Some(&lambda_id) = scope.get(&arg_name) else { continue };
                let Some(scheme) = lambda_schemes.get(&lambda_id) else { continue };
                if let Some(concrete_tys) = derive_value_instantiation(scheme, node_types, a.id) {
                    if concrete_tys.iter().all(is_fully_concrete) {
                        call_names.insert(a.id, display_lambda_instantiation(lambda_id, &concrete_tys));
                        lambda_worklist.push((lambda_id, concrete_tys, arg_name));
                    }
                }
            }
            args.iter().for_each(|a| rec!(a));
            let name = path.segments.join("::");

            if let Some(&lambda_id) = scope.get(&name) {
                if let Some(scheme) = lambda_schemes.get(&lambda_id) {
                    if let Some(concrete_tys) = derive_instantiation(scheme, expr, args, node_types) {
                        // A self-recursive call site, reached while walking
                        // a still-*generic* copy of this lambda's own body
                        // (its own `node_types` not yet substituted for any
                        // particular concrete instantiation -- see `scope`'s
                        // own seeding in the `StmtKind::Let` arm above),
                        // reverse-unifies against types that are themselves
                        // still open type variables -- `derive_instantiation`
                        // happily "succeeds" against them (unifying a `Ty::
                        // Var` with anything always does), but the resulting
                        // `concrete_tys` isn't actually concrete at all.
                        // Recording it here would create a bogus, never-
                        // reachable specialization (its own body, if ever
                        // built, could go on to fail resolving *its own*
                        // calls against non-existent generic-type impls --
                        // found by direct testing on the unannotated CLI
                        // repro). Silently deferred instead, the same
                        // "not concrete yet" posture used everywhere else in
                        // this pass -- the *real*, concrete instantiation is
                        // still discovered separately, from whichever
                        // *external* call site actually pins this lambda's
                        // own generics down (`fact(5)`'s own outer call,
                        // here), and correctly re-resolves this exact same
                        // self-call site during its own drain-loop re-walk
                        // (`node_types` substituted there -- see `monomorphize`'s
                        // own lambda-worklist loop).
                        if concrete_tys.iter().all(is_fully_concrete) {
                            call_names.insert(expr.id, display_lambda_instantiation(lambda_id, &concrete_tys));
                            lambda_worklist.push((lambda_id, concrete_tys, name.clone()));
                        }
                    }
                }
                return;
            }

            if let Some(scheme) = global_env.get(&name) {
                if !scheme.vars.is_empty() {
                    if let Some(concrete_tys) = derive_instantiation(scheme, expr, args, node_types) {
                        call_names.insert(expr.id, display_instantiation(&name, &concrete_tys));
                        fn_worklist.push((name, concrete_tys));
                    }
                }
                return;
            }

            let Some(arg_tys): Option<Vec<Ty>> = args.iter().map(|a| node_types.get(&a.id).cloned()).collect() else {
                return;
            };
            match derive_impl_instantiation(templates, &name, expr.id, &arg_tys, node_types) {
                ImplMatch::Found(idx, mapping) => {
                    call_names.insert(expr.id, display_impl_instantiation(&templates[idx], &mapping));
                    impl_worklist.push((idx, mapping));
                }
                ImplMatch::NoCandidates => {} // not an algebra call, or a non-generic one -- nothing to do here
                ImplMatch::NoneMatched { algebra, tys } => {
                    errors.push(TypeError {
                        span: expr.span,
                        kind: TypeErrorKind::MonomorphizationFailed { algebra, method: name, tys },
                    });
                }
            }
        }
        ExprKind::FieldAccess(base, _) => rec!(base),
        // `v.method(args)` -- unlike an algebra call, an inherent method's
        // own struct is *already known* directly from `base`'s own resolved
        // type (`node_types`), so this is a plain `(struct_name, method_
        // name)` lookup among `inherent_templates`, not a structural
        // candidate search (see `InherentTemplate`'s own doc comment for
        // why there's at most one to find). A non-generic inherent method
        // needs no entry here at all -- `cps.rs`'s own `resolve_method_call`
        // falls back to its own bare `struct::method` unit name directly,
        // the same "no call_names entry needed" shape a non-generic
        // top-level `fn`/algebra impl already has.
        ExprKind::MethodCall(base, name, args) => {
            rec!(base);
            args.iter().for_each(|a| rec!(a));
            if let Some(struct_name) = node_types.get(&base.id).and_then(|t| match t {
                Ty::Con(n) | Ty::App(n, _) => Some(n.clone()),
                _ => None,
            }) {
                if let Some((idx, template)) =
                    inherent_templates.iter().enumerate().find(|(_, t)| t.struct_name == struct_name && t.method_name == *name)
                {
                    if let Some(mapping) = derive_inherent_instantiation(template, expr, base, args, node_types) {
                        call_names.insert(expr.id, display_inherent_instantiation(template, &mapping));
                        inherent_worklist.push((idx, mapping));
                    }
                }
            }
        }
        ExprKind::Index(base, indices) => {
            rec!(base);
            indices.iter().for_each(|i| rec!(i));
            // A non-array base -- `Index<Container, Elem, const K: i32>`
            // algebra dispatch (see `infer.rs`'s own `ExprKind::Index`
            // fallback doc comment) -- mirrors the bare-name-call handling
            // in `ExprKind::Call` above, structurally: a real array base
            // needs no entry here at all (`cps.rs`'s own `PrimOp::Load`
            // needs no `call_names` lookup), the same "no entry needed for
            // the non-generic/non-dispatched case" posture every other tier
            // here already has. The whole bracket group's own indices
            // become one synthetic `[i32;K]` array type here -- mirrors
            // `cps.rs`'s own identical construction for the real *value* at
            // CPS-conversion time, `K` known directly from `indices.len()`,
            // no real `Expr`/`NodeId` needed for "the idx array" at all.
            if let Some(base_ty) = node_types.get(&base.id).cloned() {
                if !matches!(base_ty, Ty::Array(..)) {
                    let idx_array_ty =
                        Ty::Array(Box::new(Ty::Con("i32".to_string())), Box::new(Ty::Const(ConstValue::Int(indices.len() as u64))));
                    match derive_impl_instantiation(templates, "index", expr.id, &[base_ty, idx_array_ty], node_types) {
                        ImplMatch::Found(tmpl_idx, mapping) => {
                            call_names.insert(expr.id, display_impl_instantiation(&templates[tmpl_idx], &mapping));
                            impl_worklist.push((tmpl_idx, mapping));
                        }
                        ImplMatch::NoCandidates => {}
                        ImplMatch::NoneMatched { algebra, tys } => {
                            errors.push(TypeError {
                                span: expr.span,
                                kind: TypeErrorKind::MonomorphizationFailed { algebra, method: "index".to_string(), tys },
                            });
                        }
                    }
                }
            }
        }
        ExprKind::ArrayLit(elems) => elems.iter().for_each(|e| rec!(e)),
        ExprKind::ArrayRepeat { value, count } => {
            rec!(value);
            rec!(count);
        }
        ExprKind::StructLit(_, _, fields) => fields.iter().for_each(|(_, v)| rec!(v)),
        ExprKind::If { cond, then_branch, else_branch } => {
            rec!(cond);
            rec_block!(then_branch, scope);
            if let Some(eb) = else_branch {
                match &**eb {
                    ElseBranch::If(e) => rec!(e),
                    ElseBranch::Block(b) => rec_block!(b, scope),
                }
            }
        }
        ExprKind::While { cond, body } => {
            rec!(cond);
            rec_block!(body, scope);
        }
        ExprKind::For { var, start, end, body } => {
            rec!(start);
            rec!(end);
            let mut inner = scope.clone();
            inner.remove(var);
            rec_block!(body, &inner);
        }
        ExprKind::Block(b) => rec_block!(b, scope),
        ExprKind::Lambda { params, body, .. } => {
            let mut inner = scope.clone();
            for p in params {
                inner.remove(&p.name);
            }
            rec_block!(body, &inner);
        }
    }
}

/// True iff `ty` (recursively) contains no leftover `Ty::Var` — used to
/// reject a `derive_instantiation`/`derive_value_instantiation` result
/// that "succeeded" only because it unified against a call site whose own
/// `node_types` are themselves still generic (a self-recursive call
/// discovered while walking a still-uninstantiated copy of a lambda's own
/// body — see the `ExprKind::Call` arm's own doc comment above). Unifying
/// a bare `Ty::Var` against anything always succeeds, so `derive_
/// instantiation` alone can't tell "genuinely concrete" from "still open"
/// apart on its own.
fn is_fully_concrete(ty: &Ty) -> bool {
    let mut vars = HashSet::new();
    free_vars(ty, &mut vars);
    vars.is_empty()
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
    call_id: NodeId,
    arg_tys: &[Ty],
    node_types: &HashMap<NodeId, Ty>,
) -> ImplMatch {
    let candidates: Vec<&ImplTemplate> = templates.iter().filter(|t| t.method_name == method).collect();
    if candidates.is_empty() {
        return ImplMatch::NoCandidates;
    }
    let Some(ret_ty) = node_types.get(&call_id).cloned() else {
        return ImplMatch::NoCandidates;
    };
    let query = Ty::Fn(arg_tys.to_vec(), Box::new(ret_ty));

    for (idx, t) in templates.iter().enumerate() {
        if t.method_name != method || t.param_patterns.len() != arg_tys.len() {
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

/// Like `derive_instantiation`, but for a lambda-bound name used as a bare
/// *value* (an argument to a higher-order call, `apply(inc, 5)`) rather than
/// itself being called at this site — unifies `scheme.ty` directly against
/// the reference's own already-resolved concrete type (`node_types
/// [node_id]`, pinned by ordinary unification against whatever position it
/// was passed into — e.g. a higher-order parameter's own declared `Ty::Fn`,
/// exactly the way an ordinary generic argument gets pinned by the
/// parameter it's passed to) instead of building a synthetic `Ty::Fn(args,
/// ret)` query the way a real call site needs to (there's no argument list/
/// return type of *this* reference's own to build one from — it's a bare
/// value, not a call). See `cps.rs`'s own closure-conversion module doc
/// comment ("higher-order calls") for why this needs to exist at all: a
/// callable passed as a bare argument never becomes a `Call` node of its
/// own, so `collect_instantiations_expr`'s ordinary per-`Call` detection
/// alone would never discover that it needs its own specialization built.
fn derive_value_instantiation(scheme: &Scheme, node_types: &HashMap<NodeId, Ty>, node_id: NodeId) -> Option<Vec<Ty>> {
    let ty = node_types.get(&node_id)?.clone();
    let mut trial = Subst::default();
    unify(&mut trial, &scheme.ty, &ty).ok()?;
    Some(scheme.vars.iter().map(|v| trial.apply(&Ty::Var(*v))).collect())
}

fn display_instantiation(name: &str, tys: &[Ty]) -> String {
    if tys.is_empty() {
        name.to_string()
    } else {
        format!("{name}<{}>", tys.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "))
    }
}

/// A lambda has no source-level name of its own to key its display string
/// off of (unlike `display_instantiation`) — its own `NodeId` is the only
/// thing that's actually unique to it (two lambdas bound to the same local
/// name `f` in two different functions are two different `NodeId`s, and
/// must render as two different mangled names). `#`, not `::`, deliberately
/// unparseable as an ordinary path segment — never meant to round-trip
/// through the grammar, only to be a unique `specializations`/`by_origin`
/// key and a readable `--dump-monomorphized` label.
fn display_lambda_instantiation(id: NodeId, tys: &[Ty]) -> String {
    display_instantiation(&format!("<lambda#{}>", id.0), tys)
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

/// The inherent-impl counterpart to `derive_instantiation`/`derive_impl_
/// instantiation` — no candidate search needed (see `InherentTemplate`'s
/// own doc comment), so this is called only once the caller already knows,
/// structurally, which single template applies (matched by `struct_name`/
/// `method_name` directly). Unifies the template's own `(param_patterns) ->
/// ret_pattern` against the call's own concrete `(base_ty, arg_tys...) ->
/// ret_ty` (`base` first, positionally — it fills the method's own first
/// parameter, an ordinary explicit slot, not a magic `self`), then reads
/// back bindings for every free variable the template mentions anywhere in
/// either.
fn derive_inherent_instantiation(
    template: &InherentTemplate,
    call: &Expr,
    base: &Expr,
    args: &[Expr],
    node_types: &HashMap<NodeId, Ty>,
) -> Option<HashMap<TyVar, Ty>> {
    let base_ty = node_types.get(&base.id)?.clone();
    let mut arg_tys = vec![base_ty];
    arg_tys.extend(args.iter().map(|a| node_types.get(&a.id).cloned()).collect::<Option<Vec<_>>>()?);
    let ret_ty = node_types.get(&call.id)?.clone();
    let query = Ty::Fn(arg_tys, Box::new(ret_ty));
    let pattern = Ty::Fn(template.param_patterns.clone(), Box::new(template.ret_pattern.clone()));
    let mut trial = Subst::default();
    unify(&mut trial, &pattern, &query).ok()?;
    let mut vars = HashSet::new();
    template.param_patterns.iter().for_each(|p| free_vars(p, &mut vars));
    free_vars(&template.ret_pattern, &mut vars);
    Some(vars.into_iter().map(|v| (v, trial.apply(&Ty::Var(v)))).collect())
}

/// The inherent-impl counterpart to `display_impl_instantiation` — described
/// by the concrete *receiver* type (`param_patterns[0]`, substituted; see
/// `InherentTemplate`'s own doc comment for why that's already the target
/// pattern, no separate field to read), the same "reader thinks in terms of
/// the operand shapes actually involved" reasoning.
fn display_inherent_instantiation(t: &InherentTemplate, mapping: &HashMap<TyVar, Ty>) -> String {
    let target = substitute(&t.param_patterns[0], mapping);
    format!("{}::{}<{}>", t.struct_name, t.method_name, target)
}

/// Collects every sub-expression of `expr`, including `expr` itself, into
/// `out` — mirrors `callgraph.rs`'s own private `collect_calls_expr`
/// traversal shape (same exhaustive per-`ExprKind` structure), but collects
/// *every* node reference here, not just `Call`s by name, since this is
/// also used to build a specialization's own substituted `node_types`.
pub(crate) fn collect_exprs<'a>(expr: &'a Expr, out: &mut Vec<&'a Expr>) {
    out.push(expr);
    match &expr.kind {
        ExprKind::NumberLit { .. } | ExprKind::ImaginaryLit { .. } | ExprKind::BoolLit(_) | ExprKind::Path(_) => {}
        ExprKind::Call(_, _, args, ..) => args.iter().for_each(|a| collect_exprs(a, out)),
        ExprKind::FieldAccess(base, _) => collect_exprs(base, out),
        ExprKind::MethodCall(base, _, args) => {
            collect_exprs(base, out);
            args.iter().for_each(|a| collect_exprs(a, out));
        }
        ExprKind::Index(base, indices) => {
            collect_exprs(base, out);
            indices.iter().for_each(|i| collect_exprs(i, out));
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

pub(crate) fn collect_exprs_block<'a>(block: &'a Block, out: &mut Vec<&'a Expr>) {
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
                // The algebra's own const generics are checked too, not just
                // the impl's — see `cps.rs::collect_units`'s identical guard
                // (and `ImplTemplate::is_generic`'s own doc comment) for why
                // `d.generics.is_empty()` alone isn't enough: an impl
                // declaring zero generics of its own can still inherit a
                // free variable from the algebra's own const generic.
                let algebra_has_const_generic =
                    registry.generics(&d.algebra).iter().any(|g| matches!(g, GenericParam::Const { .. }));
                if d.generics.is_empty() && !algebra_has_const_generic {
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
