//! Monomorphization: for every generic top-level `fn` *and* every generic
//! `algebra`-impl method, generates one fully concrete specialization per
//! instantiation actually reachable from a concrete entry point — the
//! missing piece between real HM polymorphism/qualified dispatch and an
//! eventual backend that needs every type fully resolved, no free type
//! variables anywhere left (see `type_inference.md`'s own "Monomorphization"
//! section).
//!
//! Scoped to top-level `fn`s and *algebra*-impl methods — the only two
//! things that can ever need monomorphizing. Inherent impls are not a
//! language concept here: `v.method(args)` is pure sugar for `method(v,
//! args)`, so a generic method dot-called on a struct is monomorphized as
//! an ordinary generic top-level `fn`, through the exact same path.
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
use crate::callgraph::{self, FnResult, ProgramInference};
use crate::cps::{StructSchema, collect_struct_schemas};
use crate::dump::{TyVarNames, dump_block_with_call_names, fmt_ty_named};
use crate::infer::{
    ConstValue, Env, Infer, InstanceOracle, Scheme, Subst, Ty, TyVar, TyVarGen, TypeError,
    TypeErrorKind, free_vars, substitute, unify,
};
use std::cell::{Cell, RefCell};
use crate::mlir_lower::struct_field_types;
use crate::registry::Registry;
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

/// One concrete instantiation — of a top-level `fn` or of a generic
/// algebra-impl method alike, the point where the two worklists below
/// converge back into one shared shape (see the module's own doc comment).
#[derive(Clone)]
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
    /// Mirrors `ast::FnDecl::is_extern`/`extern_symbol` — needed here, not
    /// just read off whichever `FnDecl` happens to be at hand, because
    /// `cps.rs::collect_units`'s own `ItemKind::Impl` branch iterates every
    /// impl of an algebra sharing a method name (`Print<i8>`, `Print<[i8;
    /// N]>`, `Print<Wrapper<A>>`, ... all named `print`) and, for *each*
    /// one, re-queries `specializations_of("Algebra::method")` — the full,
    /// shared list of every specialization under that origin, not just the
    /// ones *this* impl produced. A real bug, found by direct testing the
    /// first time two *generic* impls of the same algebra/method coexisted
    /// (`impl<const N: i32> Print<[i8; N]>` alongside a second generic
    /// `Print<...>` impl): deriving `UnitBody::Extern`/`UnitBody::Real` from
    /// *that* impl's own `FnDecl` rather than from the specialization
    /// actually being processed silently rebuilt an unrelated, already-
    /// correct specialization with the *wrong* impl's own body/extern-ness,
    /// and the resulting duplicate `ConcreteUnit` (same name, wrong body)
    /// silently overwrote the correct one in `convert_program`'s own
    /// `by_name` map. Recording each specialization's own extern-ness
    /// directly here — read back via `MonomorphizedProgram::is_extern`/
    /// `extern_symbol` — is the real fix: `collect_units` now asks the
    /// specialization itself, never the current outer-loop impl.
    is_extern: bool,
    extern_symbol: Option<String>,
    /// Mirrors `is_extern`/`extern_symbol`'s own reasoning exactly, for the
    /// identical reason (`FnDecl::attrs`, not read off whichever impl's own
    /// `f` happens to be at hand) — `#[pure]` on a generic algebra impl's
    /// own `extern` method (`stdlib/linalg/matrix.cleave`'s own
    /// `BlasSgemmRowMajor::blas_sgemm_rowmajor`). Read back via
    /// `MonomorphizedProgram::is_pure`.
    is_pure: bool,
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
    /// Exposed so `cps.rs::collect_units`'s own *non-generic*-impl branch
    /// (which re-infers each concrete impl method directly, rather than
    /// reusing a `Specialization` — see its own doc comment for why) can
    /// still run the identical qualified-call discovery `collect_
    /// instantiations_expr` already does for every *reachability-driven*
    /// specialization, instead of hardcoding `call_names: HashMap::new()`
    /// — a real, found-by-testing gap: a qualified call (`Transcendental::
    /// tanh(x)`) inside a fully-concrete impl's own body (`Activation<f64>
    /// ::tanh`, `stdlib/nn/nn.cleave`) was never discoverable at all
    /// through `collect_units`'s own independent, template-free path,
    /// panicking at CPS-lowering time (`could not resolve call`) rather
    /// than failing a clean type check, or — for the reachable case —
    /// working at all.
    templates: Vec<ImplTemplate>,
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

    pub fn is_extern(&self, key: &str) -> bool {
        self.specializations[key].is_extern
    }

    pub fn is_pure(&self, key: &str) -> bool {
        self.specializations[key].is_pure
    }

    pub fn extern_symbol(&self, key: &str) -> Option<&str> {
        self.specializations[key].extern_symbol.as_deref()
    }

    pub fn seed_call_names(&self) -> &HashMap<NodeId, String> {
        &self.seed_call_names
    }

    pub fn errors(&self) -> &[TypeError] {
        &self.errors
    }

    pub(crate) fn templates(&self) -> &[ImplTemplate] {
        &self.templates
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
#[derive(Clone)]
pub(crate) struct ImplTemplate {
    algebra: String,
    method_name: String,
    /// The impl's own declaration — its generics, targets and this method —
    /// for inferring an instance of it (`InstanceEngine::specialize_impl`).
    impl_generics: Vec<GenericParam>,
    impl_targets: Vec<Type>,
    decl: FnDecl,
    params: Vec<Param>,
    body: Block,
    param_patterns: Vec<Ty>,
    ret_pattern: Ty,
    target_patterns: Vec<Ty>,
    node_types: HashMap<NodeId, Ty>,
    /// Every one of this impl's own declared *type* generics that carries a
    /// non-empty bound list — the fresh `Ty` (`active_generics`'s own value,
    /// the exact same fresh var/pack already baked into `param_patterns`/
    /// `target_patterns`/`ret_pattern` above) paired with its own bound
    /// names. `find_impl_for_target`'s own bound-check reads this: a purely
    /// *structural* pattern match (its own original, and still primary,
    /// selection criterion) can't tell `impl<T: Float+Ring> Optimizer<Sgd,
    /// T>` apart from `impl<Opt> Optimizer<Opt, Pair>` for a query of
    /// `(Sgd, Pair)` — both unify structurally (`T:=Pair`, `Opt:=Sgd`) — only
    /// checking the bound (`Pair` doesn't implement `Float`/`Ring`) rules the
    /// first one out. See `find_impl_for_target`'s own doc comment for the
    /// full story (a real, previously-latent dispatch bug, found and fixed
    /// building a genuinely generic-over-its-own-optimizer `Optimizer<Opt,
    /// Model>` composing impl).
    generic_bounds: Vec<(Ty, Vec<String>)>,
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
    /// Mirrors `ast::FnDecl::is_extern`/`extern_symbol` — see
    /// `Specialization`'s own identical fields for why this needs to travel
    /// with the template, not be re-read from the impl currently being
    /// iterated.
    is_extern: bool,
    extern_symbol: Option<String>,
    /// Mirrors `Specialization`'s own identical field — see its own doc
    /// comment.
    is_pure: bool,
}

/// Runs the whole-program inference pass (`callgraph::infer_program`) and
/// then both monomorphization worklists over its result — mirrors
/// `dump.rs`'s own `dump_program`, which runs the identical first step for
/// its own, separate purpose.
pub fn monomorphize(
    program: &Program,
    registry: &Registry,
) -> (MonomorphizedProgram, ProgramInference) {
    let mut program_inference = callgraph::infer_program(program, registry);
    let functions: HashMap<&str, &FnDecl> = program
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::Fn(f) => Some((f.name.as_str(), f)),
            _ => None,
        })
        .collect();

    // Shared across every `Infer` session `build_impl_templates` and the
    // `impl_worklist` drain loop below each mint, one after another, for the
    // whole rest of this function — see `Infer::new_with_vars`'s own doc
    // comment for why: two of these short-lived sessions' own outputs get
    // unified directly against each other (the `impl_worklist` loop's own
    // re-inference fallback, `doc/backlog.md`'s own "composing algebra impl
    // generic over `Opt`..." entry), and `TyVar` numbers restarting at `0`
    // independently in each one is exactly what let that happen wrong once,
    // silently, before this existed.
    //
    // Seeded from `program_inference.next_var_id`, not `0` -- `derive_impl_
    // instantiation` (called from `collect_instantiations`, all over this
    // file) unifies a template's own pattern (built from one of *these*
    // shared-counter sessions) directly against a query built from
    // `program_inference.node_types` (`callgraph::infer_program`'s own,
    // entirely separate `Infer` sessions, already finished by the time this
    // function starts) -- the identical cross-session collision risk
    // `shared_vars` exists to close, just one session earlier than the two
    // this was first built for (`callgraph.rs`'s own `next_var_id`
    // chaining already prevents the identical collision *between* SCC
    // groups *within* that pass; `next_var_id` here is just that same
    // chain, continued one step further). Confirmed the hard way, by
    // direct testing: a template's own unrelated `TyVar` happening to
    // share a raw number with `state`'s own type in `program_inference`
    // silently aliased two completely independent struct fields together
    // (`NetworkState<S1,S2>`'s own `S2` ending up structurally equal to
    // `S1`) -- a genuine, non-deterministic (`HashMap`-iteration-order-
    // dependent) correctness bug, not just a missing-resolution one,
    // reproduced directly and root-caused precisely before this fix.
    let mut shared_vars = TyVarGen::starting_at(program_inference.next_var_id);
    let templates = build_impl_templates(
        program,
        registry,
        &program_inference.global_env,
        &mut shared_vars,
    );
    let lambda_exprs = index_lambda_exprs(program, &program_inference.lambda_schemes);

    let mut mono = MonomorphizedProgram {
        specializations: HashMap::new(),
        by_origin: HashMap::new(),
        seed_call_names: HashMap::new(),
        errors: Vec::new(),
        templates: templates.clone(),
    };
    let mut fn_worklist: Vec<(String, Vec<Ty>)> = Vec::new();
    let mut impl_worklist: Vec<(usize, HashMap<TyVar, Ty>)> = Vec::new();
    let mut lambda_worklist: Vec<(NodeId, Vec<Ty>, String)> = Vec::new();

    // The instance engine (`doc/plan-instance-inference.md`): every
    // specialization below is an instance it infers, generic functions and
    // impl methods alike. It owns copies of what it reads, so the passes
    // below can still update `program_inference`.
    let engine = InstanceEngine::new(
        registry,
        &functions,
        &templates,
        program_inference.global_env.clone(),
        program_inference.lambda_schemes.clone(),
        shared_vars,
    );

    // Non-generic functions (`main`, ...) inferred again as instances: every
    // call into a generic callee gets the result type of that callee's own
    // instance — what an output-only type (an optimizer's state, built by
    // `init_state`'s body) needs to reach its caller at all.
    for (name, f) in &functions {
        if f.is_extern || f.derivative_of.is_some() {
            continue;
        }
        let Some(body) = &f.body else { continue };
        // A fn whose scheme-level inference failed has no scheme: inferred
        // from its own annotations, and kept only if that comes out concrete.
        let params: Vec<Ty> = match program_inference.global_env.get(*name) {
            Some(scheme) if scheme.vars.is_empty() => {
                let Ty::Fn(params, _) = &scheme.ty else { continue };
                params.clone()
            }
            Some(_) => continue,
            None if f.generics.is_empty() => {
                let mut vars = engine.vars.get();
                let params = f.params.iter().map(|_| vars.fresh()).collect();
                engine.vars.set(vars);
                params
            }
            None => continue,
        };
        let mut infer = Infer::new_with_vars(registry, engine.vars.get()).with_oracle(&engine);
        let outcome =
            infer.infer_fn_with_concrete_params(f, params.clone(), None, &program_inference.global_env, None);
        engine.vars.set(infer.current_vars());
        let Ok(result) = outcome else { continue };
        let result = infer.subst.apply(&result);
        let params: Vec<Ty> = params.iter().map(|p| infer.subst.apply(p)).collect();
        if !is_fully_concrete(&result) || !params.iter().all(is_fully_concrete) {
            continue;
        }
        if infer.instance_call_names.is_empty() {
            continue;
        }
        // A root that only type-checks through its callees' instances (`let
        // a = f(t); a[1]`, `f`'s result unknown from its scheme alone): this
        // inference supersedes the scheme-level one's error.
        if let Some(entry @ Err(_)) = program_inference.results.get_mut(*name) {
            program_inference.global_env.insert(
                name.to_string(),
                Scheme::mono(Ty::Fn(params.clone(), Box::new(result.clone()))),
            );
            *entry = Ok(FnResult {
                param_types: params,
                result,
            });
        }
        let mut exprs = Vec::new();
        collect_exprs_block(body, &mut exprs);
        program_inference.node_types.extend(
            exprs
                .iter()
                .filter_map(|e| infer.node_types.get(&e.id).map(|t| (e.id, t.clone())))
                .filter(|(_, t)| is_fully_concrete(t)),
        );
        mono.seed_call_names.extend(infer.instance_call_names.clone());
    }

    let struct_schemas = collect_struct_schemas(program);
    for item in &program.items {
        let ItemKind::Fn(f) = &item.kind else {
            continue;
        };
        let Some(of_name) = &f.derivative_of else {
            continue;
        };
        let Some(scheme) = program_inference.global_env.get(of_name.as_str()) else {
            continue;
        };
        let Ty::Fn(param_tys, _) = &scheme.ty else {
            continue;
        };
        for param_ty in param_tys {
            seed_derive_tensor_field_indices(
                param_ty,
                &struct_schemas,
                &templates,
                registry,
                &mut impl_worklist,
            );
        }
    }

    // Seed: every function that itself type-checked to something *fully
    // concrete* (never generalized — a nullary member per the Monomorphism
    // Restriction, or a parameterized one whose own scheme just happens to
    // have no free variables left) has its own body's `node_types` already
    // fully resolved — scan it directly for calls into a generic callee,
    // whichever worklist it belongs to.
    for (name, f) in &functions {
        let Some(scheme) = program_inference.global_env.get(*name) else {
            continue;
        };
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
            &program_inference.lambda_schemes,
            HashMap::new(),
            &mut fn_worklist,
            &mut impl_worklist,
            &mut lambda_worklist,
            &mut mono.seed_call_names,
            &mut mono.errors,
            registry,
        );
    }

    // Seed: every *non-generic* algebra-impl method's own body also needs
    // scanning for further calls, the identical reason every non-generic
    // top-level `fn`'s body just got scanned above. `cps.rs::collect_units`
    // re-infers a non-generic impl's own body *itself* (a separate concern
    // -- it needs `Infer::infer_impl_fn_generic_with_env`'s own real
    // inference result to build the unit's `param_types`/`node_types`, not
    // just this template's already-resolved ones) but, until now, fed that
    // re-inference's own `collect_instantiations` call nothing but
    // throwaway worklists for `fn_worklist`/`impl_worklist`/`lambda_
    // worklist`/`inherent_worklist` -- only `call_names` was ever kept.
    // Its own doc comment already named the exact consequence, as an
    // accepted, narrow gap: "a call from *this* body into a still-*generic*
    // fn/impl needing its own further specialization isn't discovered this
    // way ... no known case needs it yet." A real case now does: `Display
    // <i32>::display` (`stdlib/display/display.cleave`, a non-generic
    // impl) calls `out.push(...)`, an inherent method on the *generic*
    // `DynArray<T>` (`stdlib/dynarray/dynarray.cleave`) -- needing its own
    // `DynArray::push<i8>` specialization built, exactly like `Dense::
    // forward` calling `matmul` needed one for `impl_worklist` (the
    // "backward push" fixed-point item just below this one). Scanning
    // *here*, with the real worklists, means `mono` already contains
    // whatever gets discovered by the time this function returns --
    // `cps.rs::collect_units`'s own generic-impl/generic-inherent-impl
    // branches already read `mono.specializations_of(...)` unconditionally,
    // so no change is needed there at all, only here. `templates` already
    // has one `ImplTemplate` per impl, concrete or generic alike (`build_
    // impl_templates`'s own doc comment) -- `!t.is_generic` alone identifies
    // the non-generic ones, the identical condition `cps.rs::collect_units`
    // itself gates its own separate re-inference branch on.
    for t in &templates {
        if t.is_generic {
            continue;
        }
        collect_instantiations(
            &t.body,
            &t.node_types,
            &program_inference.global_env,
            &templates,
            &program_inference.lambda_schemes,
            HashMap::new(),
            &mut fn_worklist,
            &mut impl_worklist,
            &mut lambda_worklist,
            &mut mono.seed_call_names,
            &mut mono.errors,
            registry,
        );
    }

    // Drained to a fixed point: an instance can discover more work for any
    // worklist (`collect_instantiations` over its body), and so can the
    // lambdas. Every function and impl-method instance is inferred by
    // `engine`; lambdas are still specialized by substitution.
    // Lambdas of instance bodies, absorbed as the engine reports them.
    let mut instance_lambda_exprs: HashMap<NodeId, Expr> = HashMap::new();
    let absorb = |produced: &mut Produced,
                  program_inference: &mut ProgramInference,
                  exprs: &mut HashMap<NodeId, Expr>| {
        for (id, scheme, expr, types) in std::mem::take(&mut produced.lambdas) {
            program_inference.lambda_schemes.insert(id, scheme);
            program_inference.node_types.extend(types);
            exprs.insert(id, expr);
        }
    };
    loop {
        let mut produced = engine.drain();
        absorb(&mut produced, &mut program_inference, &mut instance_lambda_exprs);
        merge_produced(&mut mono, produced, &templates, registry, &mut fn_worklist, &mut impl_worklist, &mut lambda_worklist);

        while let Some((name, concrete_tys)) = fn_worklist.pop() {
            let display = display_instantiation(&name, &concrete_tys);
            if mono.specializations.contains_key(&display) {
                continue;
            }
            let Some(Scheme { vars, ty: Ty::Fn(param_pattern, ret_pattern), .. }) = program_inference.global_env.get(&name) else {
                continue;
            };
            let mapping: HashMap<TyVar, Ty> = vars.iter().copied().zip(concrete_tys.iter().cloned()).collect();
            let args: Vec<Ty> = param_pattern.iter().map(|p| substitute(p, &mapping)).collect();
            if !args.iter().all(is_fully_concrete) {
                continue;
            }
            let ret = substitute(ret_pattern, &mapping);
            let scheme_args = concrete_tys.iter().all(is_fully_concrete).then_some(concrete_tys.as_slice());
            match engine.specialize_fn(&name, &args, is_fully_concrete(&ret).then_some(&ret), scheme_args) {
                Some((unit, _)) => alias_specialization(&mut mono, &engine, &unit, &display),
                None => {
                    if let Some(e) = engine.last_error.borrow_mut().take() {
                        mono.errors.push(TypeError {
                            span: e.span,
                            kind: TypeErrorKind::GenericFnInstantiationFailed {
                                name: name.clone(),
                                tys: concrete_tys.iter().map(Ty::to_string).collect::<Vec<_>>().join(", "),
                                inner: Box::new(e),
                            },
                        });
                    }
                }
            }
        }

        while let Some((idx, mapping)) = impl_worklist.pop() {
            let t = &templates[idx];
            let display = display_impl_instantiation(t, &mapping);
            if mono.specializations.contains_key(&display) {
                continue;
            }
            let args: Vec<Ty> = t.param_patterns.iter().map(|p| substitute(p, &mapping)).collect();
            if !args.iter().all(is_fully_concrete) {
                continue;
            }
            match engine.specialize_impl(idx, &mapping, &args) {
                Some((unit, _)) => alias_specialization(&mut mono, &engine, &unit, &display),
                None => {
                    if let Some(e) = engine.last_error.borrow_mut().take() {
                        mono.errors.push(e);
                    }
                }
            }
        }

        while let Some((lambda_id, concrete_tys, self_name)) = lambda_worklist.pop() {
            let display = display_lambda_instantiation(lambda_id, &concrete_tys);
            if mono.specializations.contains_key(&display) {
                continue;
            }
            let lambda_expr = lambda_exprs
                .get(&lambda_id)
                .copied()
                .or_else(|| instance_lambda_exprs.get(&lambda_id));
            let (Some(scheme), Some(lambda_expr)) =
                (program_inference.lambda_schemes.get(&lambda_id), lambda_expr)
            else {
                continue;
            };
            let ExprKind::Lambda { params, body, .. } = &lambda_expr.kind else {
                continue; // `lambda_exprs` only ever indexes `Lambda` nodes -- defensive, not expected
            };
            let Ty::Fn(param_pattern, ret_pattern) = &scheme.ty else {
                continue; // a lambda's own scheme is always Ty::Fn, mirroring a top-level fn's -- defensive
            };

            let mapping: HashMap<TyVar, Ty> = scheme
                .vars
                .iter()
                .copied()
                .zip(concrete_tys.iter().cloned())
                .collect();
            let param_types: Vec<Ty> = param_pattern
                .iter()
                .map(|t| substitute(t, &mapping))
                .collect();
            let result = substitute(ret_pattern, &mapping);

            let mut exprs = Vec::new();
            collect_exprs_block(body, &mut exprs);
            let node_types: HashMap<NodeId, Ty> = exprs
                .iter()
                .filter_map(|e| {
                    program_inference
                        .node_types
                        .get(&e.id)
                        .map(|t| (e.id, substitute(t, &mapping)))
                })
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
                &body,
                &node_types,
                &program_inference.global_env,
                &templates,
                &program_inference.lambda_schemes,
                initial_scope,
                &mut fn_worklist,
                &mut impl_worklist,
                &mut lambda_worklist,
                &mut call_names,
                &mut mono.errors,
                registry,
            );

            let origin = format!("<lambda#{}>", lambda_id.0);
            mono.by_origin
                .entry(origin)
                .or_default()
                .push(display.clone());
            mono.specializations.insert(
                display,
                Specialization {
                    params: params.clone(),
                    body: body.clone(),
                    param_types,
                    result,
                    node_types,
                    call_names,
                    is_extern: false,
                    extern_symbol: None,
                    is_pure: false,
                },
            );
        }

        let mut produced = engine.drain();
        let idle = produced.is_empty();
        absorb(&mut produced, &mut program_inference, &mut instance_lambda_exprs);
        merge_produced(&mut mono, produced, &templates, registry, &mut fn_worklist, &mut impl_worklist, &mut lambda_worklist);
        if idle && fn_worklist.is_empty() && impl_worklist.is_empty() && lambda_worklist.is_empty() {
            break;
        }
    }
    shared_vars = engine.finish().vars;
    let _ = shared_vars;

    (mono, program_inference)
}

fn build_impl_templates(
    program: &Program,
    registry: &Registry,
    global_env: &Env,
    shared_vars: &mut TyVarGen,
) -> Vec<ImplTemplate> {
    let mut templates = Vec::new();
    for item in &program.items {
        let ItemKind::Impl(d) = &item.kind else {
            continue;
        };
        let all_targets: Vec<Type> = std::iter::once(d.target.clone())
            .chain(d.extra_targets.iter().cloned())
            .collect();
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
            let mut infer = Infer::new_with_vars(registry, *shared_vars);
            let result = infer.infer_impl_fn_generic_with_env(
                global_env,
                &d.algebra,
                &d.generics,
                &all_targets,
                f,
                item.span,
            );
            // Written back either way -- even a *failed* attempt still
            // advanced `infer`'s own counter past whatever `TyVar`s it
            // minted along the way, and there's no reason to throw that
            // progress away (leaving `shared_vars` at its stale, pre-attempt
            // value here would let the *next* attempt mint `TyVar`s that
            // collide with this failed one's own already-abandoned-but-
            // still-numbered ones — harmless on their own since nothing
            // references them, but needless risk for zero benefit).
            *shared_vars = infer.current_vars();
            let Ok(ret_pattern) = result else {
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
            infer
                .param_types
                .iter()
                .for_each(|p| free_vars(p, &mut free));
            free_vars(&ret_pattern, &mut free);
            infer
                .target_types
                .iter()
                .for_each(|p| free_vars(p, &mut free));
            let is_generic = is_generic || !free.is_empty();
            // `active_generics` (set by `infer_impl_fn_generic_with_env`
            // just above, left populated after it returns — see that
            // field's own doc comment) is the exact fresh `Ty` each of
            // `d.generics`'s own declared names minted for *this* impl —
            // the same one already baked into `param_patterns`/`target_
            // patterns`/`ret_pattern` above. Paired with each declared
            // generic's own bound list (only `Type` generics carry bounds
            // at all; skipped when empty, matching `check_no_overlapping_
            // impls`'s own identical filter) — `find_impl_for_target`'s own
            // bound-check reads this back.
            let generic_bounds: Vec<(Ty, Vec<String>)> = d
                .generics
                .iter()
                .filter_map(|g| match g {
                    GenericParam::Type { name, bounds, .. } if !bounds.is_empty() => {
                        // Its representative, as the patterns name it: the
                        // generic may have been merged into another variable.
                        Some((infer.subst.apply(infer.active_generics.get(name)?), bounds.clone()))
                    }
                    _ => None,
                })
                .collect();
            // Never read for a non-generic template — `derive_impl_
            // instantiation` returns `NoCandidates` the moment it sees
            // `is_generic == false`, before ever touching `body`.
            let body = f.body.clone().unwrap_or(Block {
                stmts: Vec::new(),
                tail: None,
            });
            templates.push(ImplTemplate {
                algebra: d.algebra.clone(),
                method_name: f.name.clone(),
                impl_generics: d.generics.clone(),
                impl_targets: std::iter::once(d.target.clone()).chain(d.extra_targets.iter().cloned()).collect(),
                decl: f.clone(),
                params: f.params.clone(),
                body,
                param_patterns: infer.param_types.clone(),
                ret_pattern,
                target_patterns: infer.target_types.clone(),
                node_types: infer.node_types.clone(),
                generic_bounds,
                is_generic,
                is_extern: f.is_extern,
                extern_symbol: f.extern_symbol.clone(),
                is_pure: f.attrs.iter().any(|a| a.name == "pure"),
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
fn index_lambda_exprs<'a>(
    program: &'a Program,
    lambda_schemes: &HashMap<NodeId, Scheme>,
) -> HashMap<NodeId, &'a Expr> {
    let mut out = HashMap::new();
    for item in &program.items {
        let ItemKind::Fn(f) = &item.kind else {
            continue;
        };
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
pub(crate) fn collect_instantiations(
    body: &Block,
    node_types: &HashMap<NodeId, Ty>,
    global_env: &Env,
    templates: &[ImplTemplate],
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
    call_names: &mut HashMap<NodeId, String>,
    errors: &mut Vec<TypeError>,
    registry: &Registry,
) {
    let scope = initial_scope;
    collect_instantiations_block(
        body,
        node_types,
        global_env,
        templates,
        lambda_schemes,
        &scope,
        fn_worklist,
        impl_worklist,
        lambda_worklist,
        call_names,
        errors,
        registry,
    );
}

#[allow(clippy::too_many_arguments)]
fn collect_instantiations_block(
    block: &Block,
    node_types: &HashMap<NodeId, Ty>,
    global_env: &Env,
    templates: &[ImplTemplate],
    lambda_schemes: &HashMap<NodeId, Scheme>,
    scope: &HashMap<String, NodeId>,
    fn_worklist: &mut Vec<(String, Vec<Ty>)>,
    impl_worklist: &mut Vec<(usize, HashMap<TyVar, Ty>)>,
    lambda_worklist: &mut Vec<(NodeId, Vec<Ty>, String)>,
    call_names: &mut HashMap<NodeId, String>,
    errors: &mut Vec<TypeError>,
    registry: &Registry,
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
                    lambda_schemes,
                    &scope,
                    fn_worklist,
                    impl_worklist,
                    lambda_worklist,
                    call_names,
                    errors,
                    registry,
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
                    lambda_schemes,
                    &scope,
                    fn_worklist,
                    impl_worklist,
                    lambda_worklist,
                    call_names,
                    errors,
                    registry,
                );
                collect_instantiations_expr(
                    value,
                    node_types,
                    global_env,
                    templates,
                    lambda_schemes,
                    &scope,
                    fn_worklist,
                    impl_worklist,
                    lambda_worklist,
                    call_names,
                    errors,
                    registry,
                );
            }
            StmtKind::Expr(e) => collect_instantiations_expr(
                e,
                node_types,
                global_env,
                templates,
                lambda_schemes,
                &scope,
                fn_worklist,
                impl_worklist,
                lambda_worklist,
                call_names,
                errors,
                registry,
            ),
            StmtKind::Break(value) => {
                if let Some(v) = value {
                    collect_instantiations_expr(
                        v,
                        node_types,
                        global_env,
                        templates,
                        lambda_schemes,
                        &scope,
                        fn_worklist,
                        impl_worklist,
                        lambda_worklist,
                        call_names,
                        errors,
                        registry,
                    );
                }
            }
        }
    }
    if let Some(tail) = &block.tail {
        collect_instantiations_expr(
            tail,
            node_types,
            global_env,
            templates,
            lambda_schemes,
            &scope,
            fn_worklist,
            impl_worklist,
            lambda_worklist,
            call_names,
            errors,
            registry,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_instantiations_expr(
    expr: &Expr,
    node_types: &HashMap<NodeId, Ty>,
    global_env: &Env,
    templates: &[ImplTemplate],
    lambda_schemes: &HashMap<NodeId, Scheme>,
    scope: &HashMap<String, NodeId>,
    fn_worklist: &mut Vec<(String, Vec<Ty>)>,
    impl_worklist: &mut Vec<(usize, HashMap<TyVar, Ty>)>,
    lambda_worklist: &mut Vec<(NodeId, Vec<Ty>, String)>,
    call_names: &mut HashMap<NodeId, String>,
    errors: &mut Vec<TypeError>,
    registry: &Registry,
) {
    macro_rules! rec {
        ($e:expr) => {
            collect_instantiations_expr(
                $e,
                node_types,
                global_env,
                templates,
                lambda_schemes,
                scope,
                fn_worklist,
                impl_worklist,
                lambda_worklist,
                call_names,
                errors,
                registry,
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
                lambda_schemes,
                $scope,
                fn_worklist,
                impl_worklist,
                lambda_worklist,
                call_names,
                errors,
                registry,
            )
        };
    }
    match &expr.kind {
        ExprKind::NumberLit { .. }
        | ExprKind::ImaginaryLit { .. }
        | ExprKind::BoolLit(_)
        | ExprKind::Path(_)
        | ExprKind::PackRef(_) => {}
        ExprKind::Call(path, generics, args, ..) => {
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
                let Some(&lambda_id) = scope.get(&arg_name) else {
                    continue;
                };
                let Some(scheme) = lambda_schemes.get(&lambda_id) else {
                    continue;
                };
                if let Some(concrete_tys) = derive_value_instantiation(scheme, node_types, a.id) {
                    if concrete_tys.iter().all(is_fully_concrete) {
                        call_names
                            .insert(a.id, display_lambda_instantiation(lambda_id, &concrete_tys));
                        lambda_worklist.push((lambda_id, concrete_tys, arg_name));
                    }
                }
            }
            args.iter().for_each(|a| rec!(a));

            // Qualified call (`Ring::mul(a, b)`) — `doc/backlog-done.md`'s
            // own "qualified-call syntax" item, resolved here one step
            // earlier than the ordinary name-based tiers below, mirroring
            // `infer.rs::infer_call`'s own qualified-call check. A lambda
            // or top-level `fn` could never be bound under a name
            // containing `"::"` in the first place, so tiers 1/2 below are
            // structurally safe from this either way -- checking first just
            // keeps the two files' own control flow parallel, and avoids
            // tier 3's own broader, algebra-blind search (and the collision
            // it would reintroduce: `build_call_index`'s own bare-name-keyed
            // map has no way to tell two same-named, same-concrete-type
            // methods from two *different* algebras apart -- see that
            // function's own doc comment). `templates.iter().any(..)`, not a
            // `Registry` lookup (none in scope here) -- a safe proxy for "is
            // this genuinely a qualified algebra call" at this pipeline
            // stage: type-checking already ran and would have rejected a
            // qualified call naming a nonexistent algebra or undeclared
            // method, so a real algebra name here always has at least one
            // template by now.
            if let [algebra, method] = path.segments.as_slice() {
                if templates.iter().any(|t| &t.algebra == algebra) {
                    let Some(arg_tys): Option<Vec<Ty>> = args
                        .iter()
                        .map(|a| node_types.get(&a.id).cloned())
                        .collect()
                    else {
                        return;
                    };
                    match derive_impl_instantiation(
                        templates,
                        registry,
                        Some(algebra),
                        method,
                        expr.id,
                        &arg_tys,
                        node_types,
                    ) {
                        ImplMatch::Found(idx, mapping) => {
                            call_names.insert(
                                expr.id,
                                display_impl_instantiation(&templates[idx], &mapping),
                            );
                            impl_worklist.push((idx, mapping));
                        }
                        // Same as `Found` just above, but with an empty (no-op)
                        // mapping — a real, found-by-testing gap: `FoundConcrete`
                        // recorded *this* call site's own `call_names` entry
                        // correctly, but never enqueued the matched template's
                        // own body onto `impl_worklist` the way `Found` does,
                        // so nothing ever walked *its* body looking for further
                        // nested calls. Invisible until a fully-concrete impl's
                        // own body made a *qualified* call into a different
                        // algebra for the first time (`Activation<f32>::tanh`
                        // calling `Transcendental::tanh(x)`, `stdlib/nn/
                        // nn.cleave`) — an ordinary bare call from a concrete
                        // impl already resolves fine without this, through
                        // `cps.rs`'s own `call_index` fallback (see `ImplMatch::
                        // FoundConcrete`'s own doc comment), which is exactly
                        // why this went unnoticed until a *qualified* one
                        // needed `call_names` specifically.
                        ImplMatch::FoundConcrete(idx) => {
                            call_names.insert(
                                expr.id,
                                display_impl_instantiation(&templates[idx], &HashMap::new()),
                            );
                            impl_worklist.push((idx, HashMap::new()));
                        }
                        ImplMatch::NoCandidates => {} // type-checking already validated this qualified call; not expected, harmless if reached
                        ImplMatch::Ambiguous { algebra, candidates } => {
                            errors.push(TypeError {
                                span: expr.span,
                                kind: TypeErrorKind::AmbiguousDispatch { algebra, candidates },
                            });
                        }
                        ImplMatch::NoneMatched { algebra, tys } => {
                            errors.push(TypeError {
                                span: expr.span,
                                kind: TypeErrorKind::MonomorphizationFailed {
                                    algebra,
                                    method: method.clone(),
                                    tys,
                                },
                            });
                        }
                    }
                    return;
                }
            }

            let name = path.segments.join("::");

            if let Some(&lambda_id) = scope.get(&name) {
                if let Some(scheme) = lambda_schemes.get(&lambda_id) {
                    if let Some(concrete_tys) =
                        derive_instantiation(scheme, expr, generics, args, node_types, registry)
                    {
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
                            call_names.insert(
                                expr.id,
                                display_lambda_instantiation(lambda_id, &concrete_tys),
                            );
                            lambda_worklist.push((lambda_id, concrete_tys, name.clone()));
                        }
                    }
                }
                return;
            }

            if let Some(scheme) = global_env.get(&name) {
                if !scheme.vars.is_empty() {
                    if let Some(concrete_tys) =
                        derive_instantiation(scheme, expr, generics, args, node_types, registry)
                    {
                        call_names.insert(expr.id, display_instantiation(&name, &concrete_tys));
                        fn_worklist.push((name, concrete_tys));
                    }
                }
                return;
            }

            let Some(arg_tys): Option<Vec<Ty>> = args
                .iter()
                .map(|a| node_types.get(&a.id).cloned())
                .collect()
            else {
                return;
            };
            match derive_impl_instantiation(
                templates, registry, None, &name, expr.id, &arg_tys, node_types,
            ) {
                ImplMatch::Found(idx, mapping) => {
                    call_names.insert(
                        expr.id,
                        display_impl_instantiation(&templates[idx], &mapping),
                    );
                    impl_worklist.push((idx, mapping));
                }
                ImplMatch::FoundConcrete(_) => unreachable!(
                    "derive_impl_instantiation never returns FoundConcrete when algebra is None"
                ),
                ImplMatch::NoCandidates => {} // not an algebra call, or a non-generic one -- nothing to do here
                ImplMatch::Ambiguous { algebra, candidates } => {
                    errors.push(TypeError {
                        span: expr.span,
                        kind: TypeErrorKind::AmbiguousDispatch { algebra, candidates },
                    });
                }
                ImplMatch::NoneMatched { algebra, tys } => {
                    errors.push(TypeError {
                        span: expr.span,
                        kind: TypeErrorKind::MonomorphizationFailed {
                            algebra,
                            method: name,
                            tys,
                        },
                    });
                }
            }
        }
        ExprKind::FieldAccess(base, _) => rec!(base),
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
                // A struct or tuple indexed by position (`x[0]`) is a field
                // projection, not an `Index` dispatch (`Infer::is_positional_struct`).
                if !matches!(base_ty, Ty::Array(..))
                    && !Infer::new(registry).is_positional_struct(&base_ty)
                {
                    let idx_array_ty = Ty::Array(
                        Box::new(Ty::Con("i32".to_string())),
                        Box::new(Ty::Const(ConstValue::Int(indices.len() as u64))),
                    );
                    match derive_impl_instantiation(
                        templates,
                        registry,
                        None,
                        "index",
                        expr.id,
                        &[base_ty, idx_array_ty],
                        node_types,
                    ) {
                        ImplMatch::Found(tmpl_idx, mapping) => {
                            call_names.insert(
                                expr.id,
                                display_impl_instantiation(&templates[tmpl_idx], &mapping),
                            );
                            impl_worklist.push((tmpl_idx, mapping));
                        }
                        ImplMatch::FoundConcrete(_) => unreachable!(
                            "derive_impl_instantiation never returns FoundConcrete when algebra is None"
                        ),
                        ImplMatch::NoCandidates => {}
                        ImplMatch::Ambiguous { algebra, candidates } => {
                            errors.push(TypeError {
                                span: expr.span,
                                kind: TypeErrorKind::AmbiguousDispatch { algebra, candidates },
                            });
                        }
                        ImplMatch::NoneMatched { algebra, tys } => {
                            errors.push(TypeError {
                                span: expr.span,
                                kind: TypeErrorKind::MonomorphizationFailed {
                                    algebra,
                                    method: "index".to_string(),
                                    tys,
                                },
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
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
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
        ExprKind::For {
            var,
            start,
            end,
            body,
        } => {
            rec!(start);
            rec!(end);
            let mut inner = scope.clone();
            inner.remove(var);
            rec_block!(body, &inner);
        }
        ExprKind::ForIn { var, iter, body } => {
            rec!(iter);
            let mut inner = scope.clone();
            inner.remove(var);
            rec_block!(body, &inner);
        }
        ExprKind::Loop { body } => rec_block!(body, scope),
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
///
/// `explicit_generics` — the call's own turbofish (`f::<3, 4>(x)`), unified
/// against `scheme.vars` *first*, before the ordinary argument/return
/// reverse-unification below — found necessary (not just belt-and-suspenders)
/// by direct testing: `doc/backlog.md`'s own former "explicit turbofish
/// never consulted" item. A const generic that only ever appears *combined*
/// with another one in a parameter's own type (`fn f<const N, M>(x: [T; N +
/// M])`, no parameter mentioning `N`/`M` individually) can never be
/// recovered from `arg_tys`/`ret_ty` alone — unifying `scheme.ty`'s own
/// still-symbolic `ConstExpr("add", Var(N), Var(M))` against a concrete
/// array length like `7` is genuinely underdetermined (`unify`'s own
/// `ConstExpr`-against-`Const` arm has no rule for it, by design — see its
/// own doc comment) and fails outright. Binding `N`/`M` from the turbofish
/// into `trial` *before* that reverse-unification runs sidesteps the
/// problem entirely, with no arithmetic reasoning needed here at all: once
/// `Var(N)`/`Var(M)` are already bound, `Subst::apply`'s own constant-
/// folding (`fold_const_expr`) collapses `ConstExpr("add", Const(3),
/// Const(4))` to `Const(7)` before the two sides are ever compared, so the
/// ordinary `Array`-vs-`Array` unification just matches.
fn derive_instantiation(
    scheme: &Scheme,
    call: &Expr,
    explicit_generics: &[GenericArg],
    args: &[Expr],
    node_types: &HashMap<NodeId, Ty>,
    registry: &Registry,
) -> Option<Vec<Ty>> {
    let mut trial = Subst::default();
    // Arity is already validated by type-checking (`infer_call`'s own
    // `ArityMismatch`) whenever `explicit_generics` is non-empty at all — the
    // length check here is defensive, not expected to actually fail; a
    // mismatch just means this call site gets no turbofish-derived help,
    // falling back to plain reverse-unification exactly like before this fix.
    if explicit_generics.len() == scheme.vars.len() {
        for (v, g) in scheme.vars.iter().zip(explicit_generics) {
            if let Some(explicit_ty) = concrete_ty_from_generic_arg(g, registry) {
                unify(&mut trial, &Ty::Var(*v), &explicit_ty).ok()?;
            }
        }
    }
    let arg_tys: Vec<Ty> = args
        .iter()
        .map(|a| node_types.get(&a.id).cloned())
        .collect::<Option<_>>()?;
    let ret_ty = node_types.get(&call.id)?.clone();
    let query = Ty::Fn(arg_tys, Box::new(ret_ty));
    unify(&mut trial, &scheme.ty, &query).ok()?;
    resolve_field_constraints(scheme, &mut trial, registry);
    Some(
        scheme
            .vars
            .iter()
            .map(|v| trial.apply(&Ty::Var(*v)))
            .collect(),
    )
}

/// Pins the scheme variables a `FieldConstraint` alone determines (`n.l1`'s
/// type in `fn deep(n) { n.l1.x }`, absent from the signature) once the call
/// has made their base concrete — repeatedly, since one field's type is the
/// next one's base. Leaves a constraint whose base stays open alone.
fn resolve_field_constraints(scheme: &Scheme, trial: &mut Subst, registry: &Registry) {
    let mut infer = Infer::new(registry);
    loop {
        let mut progressed = false;
        for fc in &scheme.field_constraints {
            let result = trial.apply(&fc.result);
            if is_fully_concrete(&result) {
                continue;
            }
            let base = trial.apply(&fc.base);
            if !is_fully_concrete(&base) {
                continue;
            }
            let Ok(field_ty) = infer.resolve_field_access(&base, &fc.field, fc.span) else {
                continue;
            };
            if unify(trial, &result, &field_ty).is_ok() {
                progressed = true;
            }
        }
        if !progressed {
            return;
        }
    }
}

/// Converts one turbofish argument (`f::<i32, 3>`'s `i32`/`3`) to a `Ty` —
/// `infer.rs`'s own `Infer::generic_arg_to_ty`/`const_value_from_expr`, but
/// with no `Infer` instance available here (see the module's own "No
/// `Infer`/`TyVarGen` instance needed" doc comment) and no need for one: by
/// this pipeline stage type-checking has already run, so an explicit
/// turbofish argument is always either a literal or an arithmetic
/// combination of literals — never a still-open reference to some *other*
/// generic (that case, and any other shape `infer.rs`'s own richer version
/// handles, falls through to `None` here, same as if no turbofish were
/// given at all — `derive_instantiation`'s own ordinary reverse-unification
/// is the fallback, unaffected either way).
fn concrete_ty_from_generic_arg(g: &GenericArg, registry: &Registry) -> Option<Ty> {
    match g {
        GenericArg::Type(t) => concrete_ty_from_ast(t, registry),
        GenericArg::Const(e) => concrete_const_from_expr(e, registry),
    }
}

fn concrete_ty_from_ast(ty: &Type, registry: &Registry) -> Option<Ty> {
    match &ty.kind {
        TypeKind::Path(p, args) => {
            let name = p.segments.join("::");
            if args.is_empty() {
                // A bare name in this position is *always* parsed as a
                // type at the grammar level, even when the source actually
                // meant a whole-program `const`/`define` referenced by
                // name (`probe::<SEUIL>()` -- `grammar.pest`'s own
                // `generic_arg` doc comment, `infer.rs::ty_from_ast_mapped`'s
                // identical fallback has the fuller story) -- checked
                // *before* defaulting to `Ty::Con(name)`, for the same
                // reason `ty_from_ast_mapped` checks it before assuming a
                // bare name is a real type: a name can't be both.
                if let Some(v) = registry.global_const_value(&name) {
                    return Some(Ty::Const(v));
                }
                return Some(Ty::Con(name));
            }
            let type_args: Vec<Ty> = args
                .iter()
                .map(|a| concrete_ty_from_generic_arg(a, registry))
                .collect::<Option<_>>()?;
            Some(Ty::App(name, type_args))
        }
        TypeKind::Array(elem, size) => {
            let elem = concrete_ty_from_ast(elem, registry)?;
            let size = concrete_const_from_expr(size, registry)?;
            Some(Ty::Array(Box::new(elem), Box::new(size)))
        }
        TypeKind::Fn(params, ret) => {
            let params = params
                .iter()
                .map(|p| concrete_ty_from_ast(p, registry))
                .collect::<Option<_>>()?;
            let ret = concrete_ty_from_ast(ret, registry)?;
            Some(Ty::Fn(params, Box::new(ret)))
        }
        // `doc/backlog.md`'s own "Variadic generics" item -- grammar/AST
        // exist (Milestone 1), nothing resolves a pack yet -- `None`, the
        // same "can't resolve this turbofish argument, fall back to
        // ordinary reverse-unification" posture this function's own doc
        // comment already documents for any other not-yet-handled shape.
        TypeKind::PackRef(_) => None,
    }
}

fn concrete_const_from_expr(value: &Expr, registry: &Registry) -> Option<Ty> {
    match &value.kind {
        ExprKind::NumberLit { text, .. } => text
            .parse::<u64>()
            .ok()
            .map(|n| Ty::Const(ConstValue::Int(n))),
        ExprKind::BoolLit(b) => Some(Ty::Const(ConstValue::Bool(*b))),
        // A whole-program `const`/`define` referenced by name in a
        // const-generic-eligible position (`[T; SEUIL]`, an arithmetic
        // operand) -- `concrete_ty_from_ast`'s own identical fallback just
        // above has the fuller reasoning.
        ExprKind::Path(p) if p.segments.len() == 1 => {
            registry.global_const_value(&p.segments[0]).map(Ty::Const)
        }
        // The unary counterpart of the binary arm just below (`-N`, `lower.
        // rs::lower_unary`'s own desugaring) -- same reasoning.
        ExprKind::Call(path, _, args, ..) if path.segments.len() == 1 && args.len() == 1 => {
            let a = concrete_const_from_expr(&args[0], registry)?;
            let Ty::Const(av) = a else { return None };
            crate::const_eval::eval_unop(&path.segments[0], av).map(Ty::Const)
        }
        ExprKind::Call(path, _, args, ..) if path.segments.len() == 1 && args.len() == 2 => {
            let a = concrete_const_from_expr(&args[0], registry)?;
            let b = concrete_const_from_expr(&args[1], registry)?;
            let (Ty::Const(av), Ty::Const(bv)) = (&a, &b) else {
                return None;
            };
            crate::const_eval::eval_binop(&path.segments[0], *av, *bv).map(Ty::Const)
        }
        _ => None,
    }
}

enum ImplMatch {
    Found(usize, HashMap<TyVar, Ty>),
    /// A *non-generic* template matched, but only reached when `algebra`
    /// was `Some(_)` (a qualified call, `doc/backlog-done.md`'s own
    /// "qualified-call syntax" item) — an ordinary, unqualified call in
    /// this exact situation returns `NoCandidates` instead (see below),
    /// since `cps.rs`'s own bare-name `call_index` already resolves a
    /// genuinely unambiguous concrete impl fine on its own. A *qualified*
    /// call can't trust that: `call_index`'s own key has no algebra in it
    /// at all, so two different algebras implementing the same method for
    /// the same concrete types — exactly what a qualified call exists to
    /// pick between — would silently collide there. Carries the matched
    /// template's own index so the caller can write `call_names` directly,
    /// bypassing `call_index` entirely for this call.
    FoundConcrete(usize),
    /// No `ImplTemplate` shares this method name at all — not an algebra
    /// call in the first place, or a non-generic one (no template is ever
    /// built for those — see `build_impl_templates`) that dispatch will
    /// resolve normally, needing no specialization. Not an error.
    NoCandidates,
    /// At least one template shared this method name, but none of them
    /// unified against this call's own concrete types — a real, worth-
    /// reporting failure (see `derive_impl_instantiation`'s own doc
    /// comment), not silently treated the same as `NoCandidates`.
    NoneMatched {
        algebra: String,
        tys: String,
    },
    /// Several impls match this call's types: the call doesn't determine
    /// which one it means (`derive_impl_instantiation`'s own doc comment).
    Ambiguous {
        algebra: String,
        candidates: Vec<String>,
    },
}

/// Finds the `ImplTemplate` (if any) whose own `target_patterns` unify
/// against `target_tys`, *and* whose own declared bounds the resulting
/// substitution actually satisfies — the impl-side counterpart of `Infer::
/// dispatch_algebra_call`'s own simpler "target alone" matching, not
/// `derive_impl_instantiation`'s fuller param/return-shape matching just
/// below (which exists to *disambiguate* two algebras sharing one method
/// name). Here `algebra`/`method` are already known exactly — read directly
/// off a `derivative` rule's own declaration (`seed_derivative_rule_
/// references`, below) — so there's nothing to disambiguate *between
/// algebras*, just "does some generic impl of this exact algebra/method
/// cover this exact target." Mirrors `derive_impl_instantiation`'s own
/// free-var read-back exactly. `None` for a *non*-generic template too —
/// `collect_units` already includes those unconditionally, nothing to seed.
///
/// The bound-check is real, not defensive boilerplate — a genuine, previously-
/// latent bug, found and fixed building a composing `impl<Opt> Optimizer
/// <Opt, Dense<T,In,Out>>` (`doc/backlog.md`'s own "Optimizer" item):
/// `impl<T: Float+Ring> Optimizer<Sgd, T>` and `impl<Opt> Optimizer<Opt,
/// Pair>` both unify *structurally* against a query of `(Sgd, Pair)` (`T:=
/// Pair`, `Opt:=Sgd`) — this function's own loop, before this fix, returned
/// whichever came first in `templates`' own iteration order with no regard
/// for whether `T`'s own declared `Float+Ring` bound actually holds for the
/// type it just got unified against (`Pair`, which implements neither) —
/// silently handing a doomed specialization request to the caller, only
/// failing much later, confusingly, as `TypeErrorKind::MonomorphizationFailed`
/// deep inside the wrong impl's own body. `Infer::matching_impls` (this
/// function's own inference-side counterpart, driving ordinary call-site
/// dispatch) already checks this correctly — this function is a genuinely
/// separate, structural-only search over `ImplTemplate`s (built once, ahead
/// of time, by `build_impl_templates`) rather than the registry's own raw
/// declarations, and had simply never grown the same check. Each rejected-
/// on-bounds candidate now falls through to the next (a plain `continue`,
/// same shape the structural-mismatch check just above already uses) —
/// this module's own top-of-file doc comment's claim that `check_no_
/// overlapping_impls` alone "guarantees at most one impl ... can coherently
/// apply" turned out not to hold in practice (that check is never actually
/// invoked by the real compile pipeline — `doc/backlog.md` has the fuller
/// story, logged separately, not fixed here) — this function no longer
/// depends on that guarantee holding, it verifies bounds itself directly.
fn find_impl_for_target(
    templates: &[ImplTemplate],
    registry: &Registry,
    algebra: &str,
    method: &str,
    target_tys: &[Option<Ty>],
) -> Option<(usize, HashMap<TyVar, Ty>)> {
    for (idx, t) in templates.iter().enumerate() {
        if t.algebra != algebra
            || t.method_name != method
            || target_tys.len() > t.target_patterns.len()
            || !t.is_generic
        {
            continue;
        }
        let mut trial = Subst::default();
        // A `None` entry (a target position no caller could resolve, e.g.
        // `Transpose<A,B>`'s own `B` before this whole function's own
        // "fewer concrete targets" relaxation just below existed at all —
        // `find_impl_for_target`'s own callers only ever supply *some*
        // prefix positions, `resolve_derivative_rule_expr_ty`'s own doc
        // comment on why: only one concrete type is ever recoverable from a
        // call's own argument-type agreement) is simply skipped here, not
        // unified against at all — its own free variables get pinned later,
        // by *another* target pattern that shares them, or this candidate
        // is rejected below (`fully_resolved`) if nothing ever does.
        if t.target_patterns
            .iter()
            .zip(target_tys)
            .filter_map(|(pat, concrete)| concrete.as_ref().map(|c| (pat, c)))
            .any(|(pat, concrete)| unify(&mut trial, pat, concrete).is_err())
        {
            continue;
        }
        let bounds_satisfied = t.generic_bounds.iter().all(|(var, bounds)| {
            let resolved = trial.apply(var);
            let mut infer = Infer::new(registry);
            bounds
                .iter()
                .all(|bound| infer.has_matching_impl(bound, std::slice::from_ref(&resolved)))
        });
        if !bounds_satisfied {
            continue;
        }
        let mut vars = HashSet::new();
        t.param_patterns
            .iter()
            .for_each(|p| free_vars(p, &mut vars));
        free_vars(&t.ret_pattern, &mut vars);
        t.target_patterns
            .iter()
            .for_each(|p| free_vars(p, &mut vars));
        let mapping: HashMap<TyVar, Ty> = vars
            .into_iter()
            .map(|v| (v, trial.apply(&Ty::Var(v))))
            .collect();
        // Fewer concrete targets than this template declares (an output-
        // only trailing generic, e.g. `Transpose<A,B>`'s own `B` — nothing
        // ever calls `transpose` with `B` given explicitly, it falls out of
        // `A`'s own shape) is only a valid match when every *other* target
        // pattern's own free variables were already pinned by the ones
        // actually supplied above — found directly, needed for real:
        // `MatMul`'s own new `adjoint` rule (`stdlib/linalg/matrix.cleave`)
        // is the first rule anywhere in this codebase to reference a
        // genuinely multi-target *different* algebra cross-algebra
        // (`resolve_derivative_rule_expr_ty`'s own single-`agreed`-type
        // limitation only ever supplies `A`, never `B`) — `Transpose`'s own
        // `B` happens to reuse the exact same `T`/`N`/`M` vars `A` already
        // pins, so this still resolves soundly. A future, genuinely
        // independent trailing generic (unconstrained by any supplied
        // target) would leave a raw `Ty::Var` in `mapping` here — rejected,
        // not guessed, the same posture this whole function already takes
        // for a bounds mismatch just above.
        let fully_resolved = mapping.values().all(|resolved| {
            let mut free = HashSet::new();
            free_vars(resolved, &mut free);
            free.is_empty()
        });
        if !fully_resolved {
            continue;
        }
        return Some((idx, mapping));
    }
    None
}

/// For a `derivative` rule's own body, resolves each subexpression's own
/// concrete `Ty` bottom-up — mirrors `egraph.rs::build_pattern`'s own
/// identical cross-algebra resolution, duplicated here rather than shared,
/// since `egraph.rs` depends on `egg`, which this module must not. Whenever
/// a call into a genuinely *different* algebra is found, seeds that
/// algebra/method/target instantiation into `impl_worklist` (via `find_
/// impl_for_target`) if a template covers it — see `seed_derivative_rule_
/// references`'s own doc comment for why this needs to happen here, this
/// early, rather than relying on `synthesize_derivatives`'s own later
/// `referenced`-set mechanism.
fn resolve_derivative_rule_expr_ty(
    expr: &Expr,
    algebra: &str,
    type_env: &HashMap<String, Ty>,
    param_tys: &HashMap<&str, Ty>,
    registry: &Registry,
    infer: &mut Infer,
    templates: &[ImplTemplate],
    impl_worklist: &mut Vec<(usize, HashMap<TyVar, Ty>)>,
) -> Option<Ty> {
    match &expr.kind {
        ExprKind::Path(p) => param_tys.get(p.segments.join("::").as_str()).cloned(),
        ExprKind::NumberLit { .. } | ExprKind::BoolLit(_) => None,
        // `d(...)` sugar (`egraph.rs::build_pattern`'s own doc comment) --
        // differentiating distributes component-wise, so `d(inner)` always
        // has the exact same type as `inner` itself.
        ExprKind::Call(path, _, args, _) if path.segments.join("::") == "d" => {
            let [inner] = args.as_slice() else {
                return None;
            };
            resolve_derivative_rule_expr_ty(
                inner,
                algebra,
                type_env,
                param_tys,
                registry,
                infer,
                templates,
                impl_worklist,
            )
        }
        ExprKind::Call(path, _, args, _) => {
            let method = path.segments.join("::");
            let arg_tys: Vec<Ty> = args
                .iter()
                .filter_map(|a| {
                    resolve_derivative_rule_expr_ty(
                        a,
                        algebra,
                        type_env,
                        param_tys,
                        registry,
                        infer,
                        templates,
                        impl_worklist,
                    )
                })
                .collect();
            let owner = if registry
                .fn_sig(algebra, &method)
                .is_some_and(|s| s.params.len() == args.len())
            {
                algebra.to_string()
            } else {
                match registry.algebras_with_fn(&method, args.len()).as_slice() {
                    [only] => only.to_string(),
                    _ => return None,
                }
            };
            // The one concrete type this call's own arguments agree on --
            // every algebra ever called this way across the whole stdlib
            // today is single-generic (`Ring<T>`, `Transcendental<T>`), so
            // agreement alone pins its concrete value -- mirrors `egraph.rs
            // ::build_pattern`'s own identical reasoning and identical bail-
            // on-disagreement posture (`agreed` left `None` on disagreement,
            // handled below rather than an early `return None` directly in
            // the loop, so a same-algebra call — see below — can still
            // return its own declared type even when this doesn't apply).
            let mut agreed: Option<Ty> = None;
            for t in &arg_tys {
                match &agreed {
                    None => agreed = Some(t.clone()),
                    Some(a) if a == t => {}
                    Some(_) => {
                        agreed = None;
                        break;
                    }
                }
            }
            // Seeded regardless of same-algebra vs. cross-algebra --
            // `owner == algebra` used to skip this entirely ("`t`'s own
            // specialization already covers it"), which is only true when
            // the call is *literally* self-recursive (`MatMul::matmul`
            // calling itself). A *different* method of the *same* algebra
            // (`Ring<T>`'s own `derivative div(a,b): div(sub(...),...)`,
            // referencing `sub`/`mul` alongside `div`) is a separate `Impl
            // Template`, monomorphized independently -- found missing
            // directly, by testing `Ring<Tensor<f32,1,2>>::div` (`Dense::
            // forward`'s `sigmoid`, `stdlib/nn/nn.cleave`): `Ring::sub<
            // Tensor<f32,1,2>>`/`Ring::mul<Tensor<f32,1,2>>` are never
            // called from ordinary source anywhere in that program, unlike
            // the `f32` case, where `sub`/`mul` happen to *also* be called
            // directly elsewhere (`err*err`, `pred-y`), coincidentally
            // masking this exact gap in every scalar-only test until now.
            // Surfaced identically to the already-known cross-algebra gap
            // this function's own doc comment documents: `cps.rs`'s "call_
            // names resolved ... but no such unit exists"-style panic,
            // extracting a derivative that references a never-monomorphized
            // unit.
            // `owner`'s own declared generic parameters, in order -- needed
            // to place each argument's own resolved type at the *right*
            // target position, rather than assuming the one type a call's
            // own arguments agree on (`agreed`, just above) always
            // corresponds to `owner`'s own *first* declared generic.
            let owner_generics: Vec<&str> = registry
                .generics(&owner)
                .iter()
                .filter(|g| !matches!(g, GenericParam::Const { .. }))
                .map(|g| g.name())
                .collect();
            // Each argument whose own declared type (`sig.params[i].ty`,
            // `owner`'s own fn signature) is a bare, unqualified name
            // matching one of `owner`'s own generics pins that generic's
            // own real position — used for *any* multi-target `owner`
            // (same algebra or cross), not just cross-algebra: a same-
            // algebra recursive call's own arguments do *not* generally
            // share the *enclosing* rule's own instantiation the way a
            // `derivative` rule's own `d(a)`/`d(b)`-wrapped arguments do
            // (found directly, empirically, the exact same way `egraph.rs
            // ::build_pattern`'s own identical fix was: `MatMul`'s own
            // adjoint rule's `matmul(u, transpose(b))`/`matmul(transpose(a),
            // u)` are each a *third*, genuinely different instantiation
            // from both the forward call and each other — reusing `type_
            // env` unconditionally here seeded the *wrong* unit, or none
            // at all, exactly mirroring that earlier, now-fixed bug).
            let target_tys: Vec<Option<Ty>> = registry
                .fn_sig(&owner, &method)
                .into_iter()
                .flat_map(|sig| arg_tys.iter().zip(&sig.params))
                .fold(vec![None; owner_generics.len()], |mut acc, (arg_ty, sig_param)| {
                    if let Some(declared) = &sig_param.ty {
                        if let TypeKind::Path(p, gens) = &declared.kind {
                            if gens.is_empty() && p.segments.len() == 1 {
                                if let Some(idx) =
                                    owner_generics.iter().position(|g| *g == p.segments[0])
                                {
                                    acc[idx] = Some(arg_ty.clone());
                                }
                            }
                        }
                    }
                    acc
                });
            // Tried *unfilled* first, deliberately — a still-open position
            // (`C` in `matmul(transpose(a), u)`, `MatMul`'s own adjoint
            // rule: neither argument maps to it, it's the *return* type)
            // that unification can *itself* pin from the other, filled
            // ones (`A`/`B` alone determine `N`/`M`/`K`, hence `C`, for
            // `MatMul`) must be left to do so — pre-filling it from the
            // *enclosing* rule's own `type_env` would silently substitute
            // a *different* recursive instantiation's own value there
            // (found directly, empirically: `matmul(transpose(a), u)`'s
            // own real `C` is `Tensor<f32,2,2>`, the enclosing forward
            // call's own `C` is `Tensor<f32,1,2>` — genuinely different).
            // Retried *with* the `type_env` fallback only if the unfilled
            // attempt didn't fully resolve — `Sum::broadcast(u: T) ->
            // Container`'s own real, opposite case: `Container`'s own `N`/
            // `M` appear *nowhere else* in the template, so leaving it
            // `None` can never be pinned by unification at all, and the
            // enclosing rule's own `type_env` (the *only* other source
            // that could ever know it) is the correct, and only, fallback.
            let seed_result = if target_tys.iter().any(Option::is_some) {
                find_impl_for_target(templates, registry, &owner, &method, &target_tys)
            } else {
                None
            };
            let seed_result = seed_result.or_else(|| {
                if owner != algebra {
                    return None;
                }
                let mut filled = target_tys.clone();
                for (slot, name) in filled.iter_mut().zip(&owner_generics) {
                    if slot.is_none() {
                        *slot = type_env.get(*name).cloned();
                    }
                }
                find_impl_for_target(templates, registry, &owner, &method, &filled)
            });
            if let Some((idx, mapping)) = seed_result {
                    // This call's own *real*, resolved return type — needed
                    // whenever it's itself nested inside a further,
                    // enclosing call (`transpose(a)` inside `matmul
                    // (transpose(a), u)`, `MatMul`'s own adjoint rule): the
                    // matched template's own `target_patterns[ret_idx]`
                    // (still in terms of *its own* `TyVar`s), substituted
                    // through `mapping` (`find_impl_for_target`'s own
                    // return value, every one of those vars now fully
                    // resolved) — the identical value `egraph.rs::build_
                    // pattern`'s own `result_ty` computation reaches via
                    // `resolve_declared_type`/`generic_substitution`, just
                    // built from structured `Ty`s here instead of text.
                    // Returning `agreed`/`type_env`-based guesses here
                    // (this function's own previous behavior) is what
                    // silently seeded the *wrong* instantiation for a non-
                    // square `transpose` nested this way — found directly,
                    // empirically, chasing exactly this case.
                    let owner_ret_ty = registry
                        .fn_sig(&owner, &method)
                        .and_then(|sig| sig.ret.clone())
                        .and_then(|ret| match &ret.kind {
                            TypeKind::Path(p, gens)
                                if gens.is_empty() && p.segments.len() == 1 =>
                            {
                                owner_generics.iter().position(|g| *g == p.segments[0])
                            }
                            _ => None,
                        })
                        .map(|ret_idx| substitute(&templates[idx].target_patterns[ret_idx], &mapping));
                    impl_worklist.push((idx, mapping));
                    if let Some(ret_ty) = owner_ret_ty {
                        return Some(ret_ty);
                    }
            }
            if owner == algebra {
                // Same-algebra call — its own result type is this algebra's
                // own declared return type, substituted through the
                // *enclosing* instantiation's own `type_env`. Only reached
                // when the resolution just above couldn't determine a real
                // return type (e.g. nothing pinned at all) — a safe
                // fallback specifically because it's *this* rule's own
                // enclosing instantiation, not a guess about some other one.
                return registry
                    .fn_sig(&owner, &method)?
                    .ret
                    .as_ref()
                    .map(|ret| infer.ty_from_ast_mapped(ret, type_env));
            }
            agreed
        }
        _ => None,
    }
}

/// `doc/backlog.md`'s own "Toward a matmul-based tensorial XOR"/"Bug 3"
/// entry: a `derivative` rule's own synthesized reference to a *different*
/// algebra's generic-impl method (`MatMul`'s own product rule needing
/// `Ring::add<Tensor<f32,2,2>>`) is otherwise discovered far too late —
/// `synthesize_derivatives`/`derivative_rule_rewrites` run *after*
/// monomorphization has already finished, so nothing during the e-graph
/// rewriting stage can retroactively make `collect_units` build a concrete
/// unit for a generic impl that was never a real call site to begin with.
/// Confirmed directly, not guessed: a program calling `matmul` but never
/// calling `Ring::add` on a `Tensor` anywhere else used to panic extracting
/// the synthesized derivative (`egraph: extracted Op node "Ring::add<...>"
/// is in none of this module's own lookup tables`) — the identical program
/// with one throwaway direct `a + b` call added, purely to force
/// monomorphization, differentiated correctly.
///
/// Called from *inside* the `impl_worklist` drain loop, right alongside the
/// existing `collect_instantiations` call that discovers ordinary call-
/// based instantiations — deliberately, not as a separate outer fixed-point
/// pass: `impl_worklist` is drained with an ordinary `while let Some(...) =
/// impl_worklist.pop()`, so an entry *pushed* here, mid-loop, is picked up
/// naturally by that same loop's own later iterations, no extra plumbing
/// needed for the fixed point (a newly-seeded unit's own `derivative` rules,
/// if it has any, get the identical treatment in *its* own turn).
fn seed_derivative_rule_references(
    registry: &Registry,
    algebra: &str,
    method: &str,
    target_tys: &[Ty],
    templates: &[ImplTemplate],
    impl_worklist: &mut Vec<(usize, HashMap<TyVar, Ty>)>,
) {
    let Some(rule) = registry
        .derivative_rules(algebra)
        .iter()
        .find(|r| r.method == method)
    else {
        return;
    };
    let Some(sig) = registry.fn_sig(algebra, method) else {
        return;
    };
    let type_env: HashMap<String, Ty> = registry
        .generics(algebra)
        .iter()
        .filter(|g| !matches!(g, GenericParam::Const { .. }))
        .map(|g| g.name().to_string())
        .zip(target_tys.iter().cloned())
        .collect();
    let mut infer = Infer::new(registry);
    let param_tys: HashMap<&str, Ty> = rule
        .params
        .iter()
        .zip(&sig.params)
        .filter_map(|(rule_p, sig_p)| {
            Some((
                rule_p.name.as_str(),
                infer.ty_from_ast_mapped(sig_p.ty.as_ref()?, &type_env),
            ))
        })
        .collect();
    resolve_derivative_rule_expr_ty(
        &rule.body,
        algebra,
        &type_env,
        &param_tys,
        registry,
        &mut infer,
        templates,
        impl_worklist,
    );
}

/// A sibling of `seed_derivative_rule_references`, same call site, same
/// worklist-injection mechanism, but for a declared `adjoint` rule instead
/// of a `derivative` one — `MatMul`'s own adjoint rule needs `Transpose::
/// transpose<Tensor<...>>`, which no ordinary call site in a program that
/// only ever calls `matmul` directly (never `transpose`) reaches at all —
/// found directly, the identical "extracted expression references a
/// symbol `rebuild`/`collect_units` never actually monomorphized" class of
/// gap `seed_derivative_rule_references`'s own doc comment already
/// documents for the forward-mode path (`egraph.rs`'s own reverse-mode
/// tests confirmed this empirically: `Transpose::transpose<...>` simply
/// never appeared in `collect_units`'s own output for a program calling
/// only `matmul`).
///
/// `param_tys` gains one entry beyond `seed_derivative_rule_references`'s
/// own: `rule.upstream`'s own type, the algebra method's own declared
/// return type (`sig.ret`) resolved through the same `type_env` — an
/// `adjoint` rule's body references it as an ordinary bound variable, no
/// special `d(...)`-style syntax (`AdjointRuleDecl`'s own doc comment).
/// A tuple-shaped body (`rule.params.len() > 1`) is walked field by field —
/// `resolve_derivative_rule_expr_ty` itself has no `ExprKind::StructLit`
/// case (a `derivative` rule's own body is always a single expression,
/// never a tuple), so each contribution expression is resolved
/// independently instead of trying to teach that shared function a new
/// top-level shape it otherwise never needs.
fn seed_adjoint_rule_references(
    registry: &Registry,
    algebra: &str,
    method: &str,
    target_tys: &[Ty],
    templates: &[ImplTemplate],
    impl_worklist: &mut Vec<(usize, HashMap<TyVar, Ty>)>,
) {
    let Some(rule) = registry
        .adjoint_rules(algebra)
        .iter()
        .find(|r| r.method == method)
    else {
        return;
    };
    let Some(sig) = registry.fn_sig(algebra, method) else {
        return;
    };
    let type_env: HashMap<String, Ty> = registry
        .generics(algebra)
        .iter()
        .filter(|g| !matches!(g, GenericParam::Const { .. }))
        .map(|g| g.name().to_string())
        .zip(target_tys.iter().cloned())
        .collect();
    let mut infer = Infer::new(registry);
    let mut param_tys: HashMap<&str, Ty> = rule
        .params
        .iter()
        .zip(&sig.params)
        .filter_map(|(rule_p, sig_p)| {
            Some((
                rule_p.name.as_str(),
                infer.ty_from_ast_mapped(sig_p.ty.as_ref()?, &type_env),
            ))
        })
        .collect();
    if let Some(ret) = sig.ret.as_ref() {
        param_tys.insert(
            rule.upstream.as_str(),
            infer.ty_from_ast_mapped(ret, &type_env),
        );
    }
    let bodies: Vec<&Expr> = if rule.params.len() > 1 {
        match &rule.body.kind {
            ExprKind::StructLit(_, _, fields) => fields.iter().map(|(_, e)| e).collect(),
            _ => return,
        }
    } else {
        vec![&rule.body]
    };
    for body in bodies {
        resolve_derivative_rule_expr_ty(
            body,
            algebra,
            &type_env,
            &param_tys,
            registry,
            &mut infer,
            templates,
            impl_worklist,
        );
    }
}

/// A third sibling of `seed_derivative_rule_references`/`seed_adjoint_rule_
/// references`, same call site, same worklist-injection mechanism — for a
/// declared `axiom` instead. Genuinely different shape from those two,
/// though: a `derivative`/`adjoint` rule's own params always match `rule.
/// method`'s own signature one-to-one (`resolve_derivative_rule_expr_ty`'s
/// own `param_tys` is built by a direct sig-zip, no recursion needed to
/// seed it), but an `axiom`'s own params can appear *nested* inside a sub-
/// call (`matmul(transpose(a), b)`'s own `a`, buried inside `transpose
/// (a)`) with no method of its own to zip against directly — found live,
/// building the matmul/transpose rewrite rules this exists for: `egraph.
/// rs::seed_axiom_type_env` already had to solve the identical problem for
/// *building the rewrite rule itself* (a purely textual, `egg`-facing
/// concern); this is that same algorithm's `Ty`-based twin, needed because
/// this module must not depend on `egg` (`resolve_derivative_rule_expr_ty`'s
/// own doc comment already states the same constraint) and because a
/// *rewrite rule* firing and a *unit actually existing to call* are two
/// separate problems — `MatMulTransposeA::matmul_transpose_a<...>` appearing
/// in an extracted expression that `collect_units` never monomorphized in
/// the first place is exactly `seed_derivative_rule_references`'s own
/// documented failure mode, for axioms instead of derivative rules.
fn seed_axiom_references(
    registry: &Registry,
    algebra: &str,
    target_tys: &[Ty],
    templates: &[ImplTemplate],
    impl_worklist: &mut Vec<(usize, HashMap<TyVar, Ty>)>,
) {
    let type_env: HashMap<String, Ty> = registry
        .generics(algebra)
        .iter()
        .filter(|g| !matches!(g, GenericParam::Const { .. }))
        .map(|g| g.name().to_string())
        .zip(target_tys.iter().cloned())
        .collect();
    let mut infer = Infer::new(registry);
    for axiom in registry.axioms(algebra) {
        let ExprKind::Call(path, _, args, _) = &axiom.body.kind else {
            continue;
        };
        if path.segments.join("::") != "eq" {
            continue;
        }
        let [lhs, rhs] = args.as_slice() else {
            continue;
        };
        let params: HashSet<&str> = axiom.params.iter().map(|p| p.name.as_str()).collect();

        // Whichever side's own *outer* call belongs to `algebra` directly
        // (true for at least one side of every real axiom -- an axiom
        // declared under `MatMul` is a fact *about* `matmul`) resolves its
        // own overall type straight from `type_env`, no hint needed at
        // all. By the axiom's own `lhs == rhs` equality, that identical
        // type applies to the *other* side too, seeded as its own top-
        // level `expected_ty`.
        let mut overall_ty: Option<Ty> = None;
        for side in [lhs, rhs] {
            let ExprKind::Call(path, _, side_args, _) = &side.kind else {
                continue;
            };
            let method = path.segments.join("::");
            let Some(sig) = registry.fn_sig(algebra, &method) else {
                continue;
            };
            if sig.params.len() != side_args.len() {
                continue;
            }
            if let Some(ret) = &sig.ret {
                overall_ty = Some(infer.ty_from_ast_mapped(ret, &type_env));
                break;
            }
        }

        // Shared across both sides, and across the whole walk of each:
        // every axiom param's own real type, filled in as either side's
        // own traversal resolves it (a bare param can appear on *both*
        // sides -- `matmul_transpose_a_rewrite`'s own `a`/`b` do -- so
        // resolving it once, from whichever side makes it easiest, must
        // carry over to the other). This is the piece `overall_ty` alone
        // can't replace: pinning only a cross-algebra callee's own
        // *return*-type position (`MatMulTransposeA::matmul_transpose_a`'s
        // own `C`) leaves its `P` (its first target's own row count)
        // genuinely undetermined by `C` alone — the *same* shape `Trans
        // pose<A,B>`'s own `B` needs `A` for, not something a single
        // return-type pin can ever resolve on its own. `a`/`b` being
        // already-known bare params (from the *other* side's own
        // traversal) is what actually pins it — found live, exactly this
        // way: without this, `find_impl_for_target` correctly refused to
        // guess (`fully_resolved` rejecting a leftover free `P`), silently
        // contributing nothing, no different in outcome from the earlier,
        // even-more-wrong bug this replaced.
        let mut param_tys: HashMap<&str, Ty> = HashMap::new();
        for side in [lhs, rhs] {
            seed_axiom_expr_references(
                side,
                overall_ty.as_ref(),
                true,
                algebra,
                &type_env,
                &params,
                registry,
                &mut infer,
                templates,
                impl_worklist,
                &mut param_tys,
            );
        }
    }
}

/// The top-down walk `seed_axiom_references` drives — the mirror image of
/// `resolve_derivative_rule_expr_ty`'s own bottom-up one (which resolves a
/// call's own unpinned generic *from* its already-resolved arguments; this
/// instead resolves each argument's own type *from* the enclosing call's
/// already-known one, exactly `egraph.rs::seed_axiom_type_env`'s own
/// algorithm, restated with structured `Ty`/`find_impl_for_target` in place
/// of text/`resolve_multi_target_call_ty`). `expected_ty` is `None` only
/// ever legitimate at the very top (`seed_axiom_references`'s own two
/// top-level calls) — a `Call` there only makes progress at all when it
/// turns out to belong to `algebra` itself, whose own `type_env` already
/// resolves that case with no hint needed. `algebra`/`type_env` stay fixed
/// at the axiom's own enclosing values throughout the whole recursion, the
/// same invariant `egraph.rs::seed_axiom_type_env`'s own doc comment states
/// for its identical two parameters — never rebound to whichever `owner` a
/// nested call resolves to.
///
/// `param_tys` is filled in by the base case (a bare `Path` naming one of
/// `params`, recording whatever `expected_ty` this call site resolved for
/// it) and *read back* by a `Call`'s own target-type resolution below,
/// pinning any of `owner`'s targets whose corresponding argument is itself
/// an already-resolved bare param — the piece a single return-type pin
/// alone can't give (this function's own `seed_axiom_references` caller
/// has the concrete motivating case). Shared across both sides of the
/// axiom's own `==`, on purpose: a param resolved from one side is exactly
/// as real on the other.
#[allow(clippy::too_many_arguments)]
fn seed_axiom_expr_references<'p>(
    expr: &Expr,
    expected_ty: Option<&Ty>,
    trust_ty: bool,
    algebra: &str,
    type_env: &HashMap<String, Ty>,
    params: &HashSet<&'p str>,
    registry: &Registry,
    infer: &mut Infer,
    templates: &[ImplTemplate],
    impl_worklist: &mut Vec<(usize, HashMap<TyVar, Ty>)>,
    param_tys: &mut HashMap<&'p str, Ty>,
) {
    match &expr.kind {
        ExprKind::Path(p) if p.segments.len() == 1 => {
            if let (Some(&name), Some(t)) = (
                params.iter().find(|&&n| n == p.segments[0]),
                expected_ty,
            ) {
                param_tys.entry(name).or_insert_with(|| t.clone());
            }
        }
        ExprKind::Call(path, _, args, _) => {
            let method = path.segments.join("::");
            let owner = if registry
                .fn_sig(algebra, &method)
                .is_some_and(|s| s.params.len() == args.len())
            {
                algebra.to_string()
            } else {
                match registry.algebras_with_fn(&method, args.len()).as_slice() {
                    [only] => only.to_string(),
                    _ => return,
                }
            };
            let Some(sig) = registry.fn_sig(&owner, &method) else {
                return;
            };
            let owner_generics: Vec<&str> = registry
                .generics(&owner)
                .iter()
                .filter(|g| !matches!(g, GenericParam::Const { .. }))
                .map(|g| g.name())
                .collect();

            // Pin from already-resolved bare-param arguments first --
            // this function's own doc comment has the motivating case (`P`
            // in `MatMulTransposeA::matmul_transpose_a`, never determined
            // by its own return type `C` alone) -- and, just as important,
            // *before* ever considering `trust_ty && owner == algebra`
            // below: a sibling pin is grounded in these specific real args'
            // own already-confirmed identity, so it must always win over
            // blindly trusting `type_env`/`ty` for a *nested* same-algebra
            // call, which (unlike the truly-outermost case) is not
            // guaranteed to share the enclosing instantiation at all --
            // found live, in exactly this shape: `matmul_transpose_
            // distributes`'s own RHS, `transpose(matmul(a, b))`, reaches
            // this `matmul(a, b)` nested one level inside a *foreign*-owner
            // `transpose` (never itself the outermost call on this side),
            // where `a`/`b` are already known (from the *other* side's own
            // traversal) to be the plain, untransposed shapes -- genuinely
            // different from `ty`'s own (transposed-and-swapped) shape, so
            // trusting `ty` there produced a real, wrong `MatMul::matmul<
            // ...>` reference (one that skipped `find_impl_for_target`
            // entirely, and so was never actually monomorphized) instead of
            // the reachable rewrite it should have found.
            let mut target_tys: Vec<Option<Ty>> = vec![None; owner_generics.len()];
            for (arg, sig_param) in args.iter().zip(&sig.params) {
                let ExprKind::Path(p) = &arg.kind else { continue };
                if p.segments.len() != 1 {
                    continue;
                }
                let Some(known) = param_tys.get(p.segments[0].as_str()) else {
                    continue;
                };
                let Some(declared) = &sig_param.ty else { continue };
                let TypeKind::Path(dp, gens) = &declared.kind else {
                    continue;
                };
                if !gens.is_empty() || dp.segments.len() != 1 {
                    continue;
                }
                if let Some(idx) = owner_generics.iter().position(|g| *g == dp.segments[0]) {
                    target_tys[idx] = Some(known.clone());
                }
            }
            let sibling_pinned = target_tys.iter().any(Option::is_some);

            // `!sibling_pinned && trust_ty && owner == algebra`: this exact
            // instantiation's own `type_env` already resolves every one of
            // `owner`'s targets directly, no template lookup needed at all
            // -- but only reached now as a fallback, once the sibling pin
            // above has had first say, and only sound as long as no
            // *ancestor* call on this same path has already matched `owner
            // == algebra` once (`egraph.rs::seed_axiom_type_env`'s own
            // identical `trust_ty` doc comment has the full reasoning: a
            // *nested* same-algebra call, e.g. `transpose(transpose(a))`'s
            // own inner `transpose`, is not guaranteed to share the
            // enclosing instantiation at all — `trust_ty` becomes `false`
            // the moment one does, below, and is otherwise carried through
            // unchanged across a *foreign*-owner call, unlike this
            // function's own former `is_top`, which forced it `false`
            // unconditionally on every recursion).
            let owner_tys: Vec<Ty> = if !sibling_pinned && trust_ty && owner == algebra {
                owner_generics
                    .iter()
                    .filter_map(|n| type_env.get(*n).cloned())
                    .collect()
            } else {
                // Then the return-type pin from context, same as before --
                // never overriding an already-argument-pinned position.
                let ret_idx = sig.ret.as_ref().and_then(|ret| match &ret.kind {
                    TypeKind::Path(p, gens) if gens.is_empty() && p.segments.len() == 1 => {
                        owner_generics.iter().position(|g| *g == p.segments[0])
                    }
                    _ => None,
                });
                if let (Some(idx), Some(t)) = (ret_idx, expected_ty) {
                    if target_tys[idx].is_none() {
                        target_tys[idx] = Some(t.clone());
                    }
                }
                if target_tys.iter().all(Option::is_none) {
                    // Nothing pins any of `owner`'s own targets here at
                    // all -- still worth recursing into each argument raw
                    // (a nested same-algebra call can resolve itself
                    // directly from `type_env`, regardless of never having
                    // received an `expected_ty` at all) rather than giving
                    // up on this whole call outright. See `egraph.rs::
                    // seed_axiom_type_env`'s identical fix and its own doc
                    // comment for the full motivating case (`Fma`'s own
                    // fusion axioms, `stdlib/linalg/matrix.cleave`: `MatMul
                    // ::matmul` is never itself the outermost call on
                    // either side of `add(matmul(a,b), c) == fma(a,b,c)`,
                    // always nested one level inside `add`/`fma`).
                    for arg in args {
                        seed_axiom_expr_references(
                            arg, None, trust_ty, algebra, type_env, params, registry, infer,
                            templates, impl_worklist, param_tys,
                        );
                    }
                    // Sibling propagation, single-generic owners only
                    // (`Ring<T>`'s own `add(a:T,b:T)->T`): every one of
                    // `owner`'s own params sharing the *same* declared
                    // generic name must share the identical concrete type
                    // -- so once the raw recursion just above resolves
                    // *any* sibling, every other still-unresolved bare
                    // sibling declared with that same generic name gets it
                    // too. `c` in `add(matmul(a,b), c)` is exactly this:
                    // never itself nested, so nothing else here could ever
                    // reach it.
                    if owner_generics.len() == 1 {
                        seed_sibling_param_tys(owner_generics[0], args, &sig.params, param_tys, params);
                    }
                    return;
                }
                let Some((idx, mapping)) =
                    find_impl_for_target(templates, registry, &owner, &method, &target_tys)
                else {
                    return; // no covering impl -- a real gap elsewhere, not this function's job to guess past
                };
                let resolved: Vec<Ty> = templates[idx]
                    .target_patterns
                    .iter()
                    .map(|p| substitute(p, &mapping))
                    .collect();
                impl_worklist.push((idx, mapping));
                resolved
            };
            let owner_type_env: HashMap<String, Ty> = owner_generics
                .iter()
                .map(|s| s.to_string())
                .zip(owner_tys)
                .collect();
            // `false` only for a same-algebra match just taken -- any
            // *other* nested call (a foreign owner, resolved above via
            // `find_impl_for_target`/context) keeps `trust_ty` as it
            // already was, per this function's own doc comment above.
            let next_trust = trust_ty && owner != algebra;
            for (arg, sig_param) in args.iter().zip(&sig.params) {
                let arg_expected = sig_param
                    .ty
                    .as_ref()
                    .map(|d| infer.ty_from_ast_mapped(d, &owner_type_env));
                seed_axiom_expr_references(
                    arg,
                    arg_expected.as_ref(),
                    next_trust,
                    algebra,
                    type_env,
                    params,
                    registry,
                    infer,
                    templates,
                    impl_worklist,
                    param_tys,
                );
            }
        }
        _ => {}
    }
}

/// Mirrors `egraph.rs::resolve_sibling_expr_type`'s own single-generic
/// sibling-propagation step, in structured `Ty` form: once a bare pattern
/// variable's own real type has been independently discovered elsewhere in
/// this same walk (`param_tys`, filled in either by the raw recursion just
/// above in `seed_axiom_expr_references`, or by the *other* side of the
/// axiom's own `==` having already run), every other still-unresolved
/// sibling argument declared with the *same* bare generic name shares it
/// too — `Ring<T>`'s own `add(a:T,b:T)->T` is exactly this: `c` in `add(
/// matmul(a,b), c)` is never itself nested, so it can only ever be
/// discovered this way.
fn seed_sibling_param_tys<'p>(
    shared_name: &str,
    args: &[Expr],
    sig_params: &[Param],
    param_tys: &mut HashMap<&'p str, Ty>,
    params: &HashSet<&'p str>,
) {
    let is_shared = |p_ty: &Option<crate::ast::Type>| {
        p_ty.as_ref().is_some_and(|d| {
            matches!(&d.kind, TypeKind::Path(p, gens)
                if gens.is_empty() && p.segments.len() == 1 && p.segments[0] == shared_name)
        })
    };
    let known: Option<Ty> = args.iter().zip(sig_params).find_map(|(arg, sig_param)| {
        if !is_shared(&sig_param.ty) {
            return None;
        }
        let ExprKind::Path(p) = &arg.kind else { return None };
        if p.segments.len() != 1 {
            return None;
        }
        param_tys.get(p.segments[0].as_str()).cloned()
    });
    let Some(known) = known else { return };
    for (arg, sig_param) in args.iter().zip(sig_params) {
        if !is_shared(&sig_param.ty) {
            continue;
        }
        if let ExprKind::Path(p) = &arg.kind {
            if p.segments.len() == 1 {
                if let Some(&name) = params.iter().find(|&&n| n == p.segments[0]) {
                    param_tys.entry(name).or_insert_with(|| known.clone());
                }
            }
        }
    }
}

/// A sibling of `seed_derivative_rule_references`, same call site, same
/// worklist-injection mechanism — but a *different* trigger: not "some
/// `derivative` rule's own body references this," since nothing declared
/// anywhere ever references `Ring::zero` at all. `egraph.rs`'s own built-in
/// `derivative-independent-zero` rule can call `Ring::zero<X>` *dynamically*
/// for any type `X` reached inside a `derive()`d function (`doc/backlog.md`'s
/// own "Real pack-generic `[value; Dims...]` array-repeat" item's own
/// follow-on — `zero()`, a real `Ring<T>`-declared method now, replacing
/// `egraph.rs::build_zero`'s own hand-built construction), with no
/// `derivative` rule anywhere in the loop. Scoped to `Ring` specifically
/// (the one algebra `derivative-independent-zero` ever needs a zero from)
/// and to *generic* `Ring<X>` impls only — a non-generic one (`Ring<f32>`,
/// `Ring<f64>`) needs no seeding at all, `cps.rs::collect_units` already
/// includes every non-generic impl's own methods (`zero()` included)
/// unconditionally, the same reason `find_impl_for_target` itself already
/// only ever matches a generic template.
fn seed_ring_zero(
    algebra: &str,
    target_tys: &[Ty],
    templates: &[ImplTemplate],
    registry: &Registry,
    impl_worklist: &mut Vec<(usize, HashMap<TyVar, Ty>)>,
) {
    if algebra != "Ring" {
        return;
    }
    let target_tys: Vec<Option<Ty>> = target_tys.iter().cloned().map(Some).collect();
    if let Some((idx, mapping)) =
        find_impl_for_target(templates, registry, "Ring", "zero", &target_tys)
    {
        impl_worklist.push((idx, mapping));
    }
}

/// Seeds `Index<Tensor<T,Dims...>, T>::index` for every `Tensor`-typed field
/// reachable (recursively, through however many levels of nested struct)
/// from a `derive()`d function's own parameter types — `egraph.rs::build_
/// param_shape`'s own eta-expansion needs a real, already-monomorphized
/// `Index::index<Tensor<...>, ...>` unit for *every* such field, even one no
/// ordinary call site in the program ever indexes explicitly (`examples/
/// xor_tensor.cleave`'s own `net.l1.w` -- passed straight to `matmul`, never
/// written as `net.l1.w[i,j]` anywhere), so ordinary reachability-driven
/// monomorphization alone can never seed it. The exact same "referenced only
/// dynamically, from inside `derive()`'s own machinery, never from any real
/// call site" shape `seed_ring_zero` just above already handles for `Ring::
/// zero` -- found missing directly, by testing `Dense`/`Network` (the first
/// program to ever pass a struct with a *nested* struct's own `Tensor` field
/// into `derive()`), the same "not anticipated by the plan, found only once
/// something with real depth was tried" pattern as `Dense::forward`'s own
/// worklist-ordering bug just above.
fn seed_derive_tensor_field_indices(
    ty: &Ty,
    struct_schemas: &HashMap<String, StructSchema>,
    templates: &[ImplTemplate],
    registry: &Registry,
    impl_worklist: &mut Vec<(usize, HashMap<TyVar, Ty>)>,
) {
    // `Ty::Con(name)` (a non-generic struct, `Network`) and `Ty::App(name,
    // args)` (a generic one, `Dense<f32,2,2>`) both name a real struct here
    // -- `Ty::Con` is just `Ty::App` with zero args, the same collapse
    // `struct_field_types`'s own callers elsewhere already rely on.
    let (name, args): (&str, &[Ty]) = match ty {
        Ty::Con(name) => (name, &[]),
        Ty::App(name, args) => (name, args),
        _ => return,
    };
    if name == "Tensor" {
        let Some(elem_ty) = args.first() else { return };
        if let Some((idx, mapping)) = find_impl_for_target(
            templates,
            registry,
            "Index",
            "index",
            &[Some(ty.clone()), Some(elem_ty.clone())],
        ) {
            impl_worklist.push((idx, mapping));
        }
        return;
    }
    // A scalar (`f32`, `i32`, ...) or any other non-struct type name has no
    // fields to walk -- `struct_schemas` only ever indexes real `struct`
    // declarations, and `struct_field_types` panics on a miss (it assumes
    // its caller already knows `name` names a real struct, true everywhere
    // else it's called from), so this must be checked *before* calling it.
    if !struct_schemas.contains_key(name) {
        return;
    }
    for (_, field_ty) in struct_field_types(struct_schemas, name, args) {
        seed_derive_tensor_field_indices(
            &field_ty,
            struct_schemas,
            templates,
            registry,
            impl_worklist,
        );
    }
}

/// Like `derive_instantiation`, for the algebra-impl side: tries every
/// template named `method`, unifying its own `(param_patterns) -> ret_
/// pattern` against this call's own concrete `(arg_tys) -> ret_ty`, in one
/// shared trial `Subst` per candidate. On the first match whose own bounds
/// the resulting substitution actually satisfies, reads back concrete
/// bindings for *every* free variable the template mentions anywhere —
/// `param_patterns`/`ret_pattern` *and* `target_patterns`, since an
/// algebra generic appearing only in the impl's own target (`C` in `fn
/// mul(a: A, b: B) -> C;`) would otherwise never get a binding at all, the
/// same reasoning `infer_algebra_call`'s own "input vs. output-only
/// generics" split exists for.
///
/// The bound-check (`t.generic_bounds`, same field, same reasoning as
/// `find_impl_for_target`'s own — see that function's doc comment for the
/// full story) is what coherence relies on: a purely structural match can't
/// distinguish `impl<T: Float+Ring> Optimizer<Sgd, T>` from `impl<Opt>
/// Optimizer<Opt, Pair>` for a call site needing `Optimizer::step(Sgd_value,
/// Pair_value, ...)` — both unify (`T:=Pair`, `Opt:=Sgd`), only the first
/// one's own bound (`Pair` doesn't implement `Float`/`Ring`) rules it out.
/// Every candidate is then weighed, never the first taken: coherence
/// (`check_no_overlapping_impls`, run by `pipeline.rs::check_type_errors`)
/// leaves at most one for a concrete call, so several mean the call doesn't
/// determine its impl — `ImplMatch::Ambiguous`, an error.
///
/// If candidates existed but none matched (structurally, or on bounds),
/// that's *usually* a real, surfaced failure (`ImplMatch::NoneMatched`) —
/// found by direct testing (and direct feedback: silently treating it the
/// same as "never called" hid a genuine, pre-existing bug in a stub impl
/// body). Ordinary dispatch (`Infer::dispatch_algebra_call`) only ever
/// needs an impl's own *target* pattern to match — it never re-checks the
/// method's full parameter/return shape the way this does, so a call that
/// type-checks fine under ordinary inference can still fail *here*,
/// honestly, if the impl's own declaration-time inference left its
/// generics over-constrained (a stub body silently merging two generics
/// that should stay independent, say).
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
    registry: &Registry,
    // `Some(algebra)` for a qualified call (`doc/backlog-done.md`'s own
    // "qualified-call syntax" item) — restricts the whole search to that one
    // algebra's own templates, instead of searching by method name alone.
    // `None` (every existing caller, before this item) leaves every line
    // below byte-for-byte the same as before it existed.
    algebra: Option<&str>,
    method: &str,
    call_id: NodeId,
    arg_tys: &[Ty],
    node_types: &HashMap<NodeId, Ty>,
) -> ImplMatch {
    let Some(ret_ty) = node_types.get(&call_id).cloned() else {
        return ImplMatch::NoCandidates;
    };
    derive_impl_instantiation_for(templates, registry, algebra, method, arg_tys, &ret_ty)
}

/// `derive_impl_instantiation` given the call's result type directly — for
/// the instance oracle, which has no `node_types` entry for the call (its
/// result is what it is asked for).
fn derive_impl_instantiation_for(
    templates: &[ImplTemplate],
    registry: &Registry,
    algebra: Option<&str>,
    method: &str,
    arg_tys: &[Ty],
    ret_ty: &Ty,
) -> ImplMatch {
    let owned_by =
        |t: &&ImplTemplate| t.method_name == method && algebra.map_or(true, |a| t.algebra == a);
    let candidates: Vec<&ImplTemplate> = templates.iter().filter(owned_by).collect();
    if candidates.is_empty() {
        return ImplMatch::NoCandidates;
    }
    let ret_ty = ret_ty.clone();
    // A const generic's value (`N` in `N > 100`) is typed `Ty::Const`, which
    // unifies with every integer width: widened to its ordinary type first, the
    // same widening `cps.rs::dispatch_ty` applies to its own dispatch keys, or
    // the first integer impl declared (`Ord<i8>`) would be picked.
    let query = Ty::Fn(
        arg_tys.iter().map(crate::cps::dispatch_ty).collect(),
        Box::new(crate::cps::dispatch_ty(&ret_ty)),
    );

    // Every matching impl, not the first: with coherence checked
    // (`check_no_overlapping_impls`), a call whose types are concrete matches
    // at most one, so several means the call itself doesn't determine its
    // impl — an indeterminacy to report, never a choice to make.
    let mut matches: Vec<(usize, Option<HashMap<TyVar, Ty>>)> = Vec::new();
    for (idx, t) in templates.iter().enumerate() {
        if t.method_name != method
            || t.param_patterns.len() != arg_tys.len()
            || algebra.is_some_and(|a| t.algebra != a)
        {
            continue;
        }
        let pattern = Ty::Fn(t.param_patterns.clone(), Box::new(t.ret_pattern.clone()));
        let mut trial = Subst::default();
        if unify(&mut trial, &pattern, &query).is_err() {
            continue;
        }
        if !t.is_generic {
            matches.push((idx, None));
            continue;
        }
        let bounds_satisfied = t.generic_bounds.iter().all(|(var, bounds)| {
            let resolved = trial.apply(var);
            let mut infer = Infer::new(registry);
            bounds
                .iter()
                .all(|bound| infer.has_matching_impl(bound, std::slice::from_ref(&resolved)))
        });
        if !bounds_satisfied {
            continue;
        }
        let mut vars = HashSet::new();
        t.param_patterns
            .iter()
            .for_each(|p| free_vars(p, &mut vars));
        free_vars(&t.ret_pattern, &mut vars);
        t.target_patterns
            .iter()
            .for_each(|p| free_vars(p, &mut vars));
        let mapping: HashMap<TyVar, Ty> = vars
            .into_iter()
            .map(|v| (v, trial.apply(&Ty::Var(v))))
            .collect();
        matches.push((idx, Some(mapping)));
    }
    // A query still holding type variables (a generic lambda's body walked
    // before any instantiation) can match several impls without being
    // ambiguous: its concrete instantiation decides later, so nothing is
    // chosen now.
    if matches.len() > 1 && !is_fully_concrete(&query) {
        return ImplMatch::NoCandidates;
    }
    match matches.len() {
        0 => ImplMatch::NoneMatched {
            algebra: candidates[0].algebra.clone(),
            tys: query.to_string(),
        },
        1 => match matches.pop().expect("one match") {
            (idx, Some(mapping)) => ImplMatch::Found(idx, mapping),
            (idx, None) => match algebra {
                None => ImplMatch::NoCandidates,
                Some(_) => ImplMatch::FoundConcrete(idx),
            },
        },
        _ => ImplMatch::Ambiguous {
            algebra: templates[matches[0].0].algebra.clone(),
            candidates: matches
                .iter()
                .map(|(idx, _)| {
                    let t = &templates[*idx];
                    let targets: Vec<String> =
                        t.target_patterns.iter().map(ToString::to_string).collect();
                    format!("{}<{}>", t.algebra, targets.join(", "))
                })
                .collect(),
        },
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
fn derive_value_instantiation(
    scheme: &Scheme,
    node_types: &HashMap<NodeId, Ty>,
    node_id: NodeId,
) -> Option<Vec<Ty>> {
    let ty = node_types.get(&node_id)?.clone();
    let mut trial = Subst::default();
    unify(&mut trial, &scheme.ty, &ty).ok()?;
    Some(
        scheme
            .vars
            .iter()
            .map(|v| trial.apply(&Ty::Var(*v)))
            .collect(),
    )
}

fn display_instantiation(name: &str, tys: &[Ty]) -> String {
    if tys.is_empty() {
        name.to_string()
    } else {
        format!(
            "{name}<{}>",
            tys.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )
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
    let targets = t
        .target_patterns
        .iter()
        .map(|p| substitute(p, mapping).to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!("{}::{}<{}>", t.algebra, t.method_name, targets)
}

/// Collects every sub-expression of `expr`, including `expr` itself, into
/// `out` — mirrors `callgraph.rs`'s own private `collect_calls_expr`
/// traversal shape (same exhaustive per-`ExprKind` structure), but collects
/// *every* node reference here, not just `Call`s by name, since this is
/// also used to build a specialization's own substituted `node_types`.
pub(crate) fn collect_exprs<'a>(expr: &'a Expr, out: &mut Vec<&'a Expr>) {
    out.push(expr);
    match &expr.kind {
        ExprKind::NumberLit { .. }
        | ExprKind::ImaginaryLit { .. }
        | ExprKind::BoolLit(_)
        | ExprKind::Path(_)
        | ExprKind::PackRef(_) => {}
        ExprKind::Call(_, _, args, ..) => args.iter().for_each(|a| collect_exprs(a, out)),
        ExprKind::FieldAccess(base, _) => collect_exprs(base, out),
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
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
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
        ExprKind::For {
            start, end, body, ..
        } => {
            collect_exprs(start, out);
            collect_exprs(end, out);
            collect_exprs_block(body, out);
        }
        ExprKind::ForIn { iter, body, .. } => {
            collect_exprs(iter, out);
            collect_exprs_block(body, out);
        }
        ExprKind::Loop { body } => collect_exprs_block(body, out),
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
            StmtKind::Break(value) => {
                if let Some(v) = value {
                    collect_exprs(v, out);
                }
            }
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
            ItemKind::Const(d) => {
                let _ = writeln!(out, "const {} {{ /* not type-inferred yet */ }}", d.name);
            }
            ItemKind::Define(d) => {
                let _ = writeln!(out, "define {} {{ /* not type-inferred yet */ }}", d.name);
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
                let algebra_has_const_generic = registry
                    .generics(&d.algebra)
                    .iter()
                    .any(|g| matches!(g, GenericParam::Const { .. }));
                if d.generics.is_empty() && !algebra_has_const_generic {
                    dump_concrete_impl(
                        &mut out,
                        &mut errors,
                        d,
                        item.span,
                        registry,
                        &program_inference.global_env,
                    );
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
            ItemKind::Fn(f) => match program_inference.results.get(&f.name) {
                Some(Err(e)) => {
                    let params: Vec<String> = f.params.iter().map(|p| p.name.clone()).collect();
                    let _ = writeln!(
                        out,
                        "fn {}({}) {{ /* type error, see diagnostics */ }}",
                        f.name,
                        params.join(", ")
                    );
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
                            let params: Vec<String> =
                                f.params.iter().map(|p| p.name.clone()).collect();
                            let _ = writeln!(
                                out,
                                "extern fn {}({}) -> {};",
                                f.name,
                                params.join(", "),
                                fn_result.result
                            );
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
                    None => unreachable!(
                        "`{}` type-checked successfully but has no scheme in global_env",
                        f.name
                    ),
                },
                None => unreachable!(
                    "`{}` is a top-level `fn` item but callgraph::infer_program has no entry for it",
                    f.name
                ),
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
fn dump_concrete_impl(
    out: &mut String,
    errors: &mut Vec<TypeError>,
    d: &ImplDecl,
    span: Span,
    registry: &Registry,
    global_env: &Env,
) {
    let targets: Vec<String> = std::iter::once(&d.target)
        .chain(d.extra_targets.iter())
        .map(crate::print::fmt_type)
        .collect();
    let _ = writeln!(out, "impl {}<{}> {{", d.algebra, targets.join(", "));
    let all_targets: Vec<Type> = std::iter::once(d.target.clone())
        .chain(d.extra_targets.iter().cloned())
        .collect();
    for f in &d.fns {
        let mut infer = Infer::new(registry);
        match infer.infer_impl_fn_generic_with_env(
            global_env,
            &d.algebra,
            &d.generics,
            &all_targets,
            f,
            span,
        ) {
            Ok(ret) => match &f.body {
                Some(body) => dump_one(
                    out,
                    &f.name,
                    &f.params,
                    body,
                    &infer.param_types,
                    &ret,
                    &infer.node_types,
                    &HashMap::new(),
                ),
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
                    let _ = writeln!(
                        out,
                        "fn {}({}) -> {ret};",
                        f.name,
                        rendered_params.join(", ")
                    );
                }
            },
            Err(e) => {
                let params: Vec<String> = f.params.iter().map(|p| p.name.clone()).collect();
                let _ = writeln!(
                    out,
                    "fn {}({}) {{ /* type error, see diagnostics */ }}",
                    f.name,
                    params.join(", ")
                );
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
    let rendered_params: Vec<String> = params
        .iter()
        .zip(param_types.iter())
        .map(|(p, t)| format!("{}: {}", p.name, fmt_ty_named(t, &mut names)))
        .collect();
    let ret = fmt_ty_named(result, &mut names);
    let _ = writeln!(
        out,
        "fn {}({}) -> {ret} {{",
        mangled_name,
        rendered_params.join(", ")
    );
    dump_block_with_call_names(out, body, node_types, &mut names, 1, call_names);
    let _ = writeln!(out, "}}");
}

/// Moves what the instance engine built into `mono` — each impl-method
/// instance also seeding the derivative/axiom references of its algebra —
/// and its discovered work into the worklists.
fn merge_produced(
    mono: &mut MonomorphizedProgram,
    produced: Produced,
    templates: &[ImplTemplate],
    registry: &Registry,
    fn_worklist: &mut Vec<(String, Vec<Ty>)>,
    impl_worklist: &mut Vec<(usize, HashMap<TyVar, Ty>)>,
    lambda_worklist: &mut Vec<(NodeId, Vec<Ty>, String)>,
) {
    for (impl_of, origin, display, spec) in produced.specializations {
        if mono.specializations.contains_key(&display) {
            continue;
        }
        if let Some((idx, target_tys)) = impl_of {
            let t = &templates[idx];
            seed_derivative_rule_references(registry, &t.algebra, &t.method_name, &target_tys, templates, impl_worklist);
            seed_ring_zero(&t.algebra, &target_tys, templates, registry, impl_worklist);
            seed_adjoint_rule_references(registry, &t.algebra, &t.method_name, &target_tys, templates, impl_worklist);
            seed_axiom_references(registry, &t.algebra, &target_tys, templates, impl_worklist);
        }
        mono.by_origin.entry(origin).or_default().push(display.clone());
        mono.specializations.insert(display, spec);
    }
    fn_worklist.extend(produced.fn_worklist);
    impl_worklist.extend(produced.impl_worklist);
    lambda_worklist.extend(produced.lambda_worklist);
    mono.errors.extend(produced.errors);
}

/// A worklist item names its instance from the call site's view
/// (`requested`); the engine names it from the instance's own (`unit`). When
/// those differ, the instance is registered under both, so every call name
/// finds its unit.
fn alias_specialization(mono: &mut MonomorphizedProgram, engine: &InstanceEngine, unit: &str, requested: &str) {
    if unit == requested {
        return;
    }
    let produced = engine.produced.borrow();
    let found = produced
        .specializations
        .iter()
        .find(|(_, _, d, _)| d == unit)
        .map(|(_, origin, _, spec)| (origin.clone(), spec.clone()))
        .or_else(|| mono.specializations.get(unit).map(|s| (String::new(), s.clone())));
    drop(produced);
    if let Some((origin, spec)) = found {
        if !origin.is_empty() {
            mono.by_origin.entry(origin).or_default().push(requested.to_string());
        }
        mono.specializations.insert(requested.to_string(), spec);
    }
}

// ------------------------------------------------------------ instance oracle

/// The monomorphizer as `InstanceOracle` (`doc/plan-instance-inference.md`):
/// asked, from inside an instance's inference, for a generic callee's
/// instance at concrete argument types, it infers that instance right away
/// (recursively, memoized) and answers with its unit name and result type.
/// What the instances it builds discover in turn is collected for the
/// worklists of `monomorphize`.
struct InstanceEngine<'a> {
    registry: &'a Registry,
    functions: &'a HashMap<&'a str, &'a FnDecl>,
    templates: &'a [ImplTemplate],
    global_env: Env,
    lambda_schemes: HashMap<NodeId, Scheme>,
    vars: Cell<TyVarGen>,
    /// Why the last instance asked for failed, for the worklists to report.
    last_error: RefCell<Option<TypeError>>,
    /// Ids for the nodes of unrolled copies of instance bodies, in a range no
    /// parsed node reaches.
    next_node: Cell<u32>,
    in_progress: RefCell<HashSet<String>>,
    done: RefCell<HashMap<String, (String, Ty)>>,
    produced: RefCell<Produced>,
}

/// What `InstanceEngine` hands back to `monomorphize`.
#[derive(Default)]
struct Produced {
    vars: TyVarGen,
    /// (template index and resolved targets for an impl method, `None` for a
    /// top-level fn; origin; unit name; specialization).
    specializations: Vec<(Option<(usize, Vec<Ty>)>, String, String, Specialization)>,
    fn_worklist: Vec<(String, Vec<Ty>)>,
    impl_worklist: Vec<(usize, HashMap<TyVar, Ty>)>,
    lambda_worklist: Vec<(NodeId, Vec<Ty>, String)>,
    errors: Vec<TypeError>,
    /// Lambdas of instance bodies (`InstanceEngine::instance_lambdas`): the
    /// scheme, the expression and the node types the instance's inference
    /// gave them, which the program-wide inference never saw.
    lambdas: Vec<(NodeId, Scheme, Expr, HashMap<NodeId, Ty>)>,
}

impl Produced {
    fn is_empty(&self) -> bool {
        self.specializations.is_empty()
            && self.fn_worklist.is_empty()
            && self.impl_worklist.is_empty()
            && self.lambda_worklist.is_empty()
            && self.errors.is_empty()
            && self.lambdas.is_empty()
    }
}

impl<'a> InstanceEngine<'a> {
    /// Takes what instances built so far discovered.
    fn drain(&self) -> Produced {
        std::mem::take(&mut *self.produced.borrow_mut())
    }

    fn new(
        registry: &'a Registry,
        functions: &'a HashMap<&'a str, &'a FnDecl>,
        templates: &'a [ImplTemplate],
        global_env: Env,
        lambda_schemes: HashMap<NodeId, Scheme>,
        vars: TyVarGen,
    ) -> Self {
        InstanceEngine {
            registry,
            functions,
            templates,
            global_env,
            lambda_schemes,
            vars: Cell::new(vars),
            last_error: RefCell::new(None),
            next_node: Cell::new(1 << 30),
            in_progress: RefCell::new(HashSet::new()),
            done: RefCell::new(HashMap::new()),
            produced: RefCell::new(Produced::default()),
        }
    }

    fn finish(self) -> Produced {
        let mut produced = self.produced.into_inner();
        produced.vars = self.vars.get();
        produced
    }

    /// The instance of impl method `templates[idx]` at the concrete argument
    /// types `args` (`mapping`: the template's bindings the call determined;
    /// whatever it left open, an output-only target such as an optimizer's
    /// state, the body's inference decides).
    fn specialize_impl(&self, idx: usize, mapping: &HashMap<TyVar, Ty>, args: &[Ty]) -> Option<(String, Ty)> {
        *self.last_error.borrow_mut() = None;
        let t = &self.templates[idx];
        if t.is_extern {
            // No body to infer: the instance is the signature at these types.
            let result = substitute(&t.ret_pattern, mapping);
            let param_types: Vec<Ty> = t.param_patterns.iter().map(|p| substitute(p, mapping)).collect();
            if !is_fully_concrete(&result) || !param_types.iter().all(is_fully_concrete) {
                return None;
            }
            let display = display_impl_instantiation(t, mapping);
            let target_tys: Vec<Ty> = t.target_patterns.iter().map(|p| substitute(p, mapping)).collect();
            let mut produced = self.produced.borrow_mut();
            if !produced.specializations.iter().any(|(_, _, d, _)| *d == display) {
                produced.specializations.push((
                    Some((idx, target_tys)),
                    format!("{}::{}", t.algebra, t.method_name),
                    display.clone(),
                    Specialization {
                        params: t.params.clone(),
                        body: t.body.clone(),
                        param_types,
                        result: result.clone(),
                        node_types: HashMap::new(),
                        call_names: HashMap::new(),
                        is_extern: true,
                        extern_symbol: t.extern_symbol.clone(),
                        is_pure: t.is_pure,
                    },
                ));
            }
            return Some((display, result));
        }
        // Keyed by the targets the call determined as well as the arguments:
        // a method without arguments (`Init::xavier()`) has one instance per
        // target type. A target the call left open reads `?`.
        let targets_key: Vec<String> = t
            .target_patterns
            .iter()
            .map(|p| {
                let ty = substitute(p, mapping);
                if is_fully_concrete(&ty) { ty.to_string() } else { "?".to_string() }
            })
            .collect();
        let key = format!(
            "{}::{}<{}>({})",
            t.algebra,
            t.method_name,
            targets_key.join(", "),
            args.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ")
        );
        if let Some(found) = self.done.borrow().get(&key) {
            return Some(found.clone());
        }
        if !self.in_progress.borrow_mut().insert(key.clone()) {
            return None;
        }

        let span = t
            .body
            .tail
            .as_deref()
            .map(|e| e.span)
            .or_else(|| t.body.stmts.first().map(|s| s.span))
            .unwrap_or(Span {
                file: FileId(0),
                start: 0,
                end: 0,
            });
        let mut decl = t.decl.clone();
        let (infer, outcome) = self.infer_unrolling(&mut decl, |infer, decl| {
            // Targets the call left open become this session's own variables.
            let mut open: HashMap<TyVar, Ty> = HashMap::new();
            let target_tys: Vec<Ty> = t
                .target_patterns
                .iter()
                .map(|p| {
                    let ty = substitute(p, mapping);
                    let mut fv = HashSet::new();
                    free_vars(&ty, &mut fv);
                    for v in fv {
                        open.entry(v).or_insert_with(|| infer.fresh_var());
                    }
                    substitute(&ty, &open)
                })
                .collect();
            infer.infer_impl_fn_instance(
                &self.global_env,
                &t.algebra,
                &t.impl_generics,
                &t.impl_targets,
                decl,
                span,
                &target_tys,
                args,
            )
        });
        self.in_progress.borrow_mut().remove(&key);
        let result = match outcome {
            Ok(r) => r,
            Err(e) => {
                *self.last_error.borrow_mut() = Some(e);
                return None;
            }
        };
        let body = decl.body.clone().unwrap_or(Block { stmts: Vec::new(), tail: None });

        let mut exprs = Vec::new();
        collect_exprs_block(&body, &mut exprs);
        let mut node_types: HashMap<NodeId, Ty> = exprs
            .iter()
            .filter_map(|e| infer.node_types.get(&e.id).map(|ty| (e.id, ty.clone())))
            .collect();
        if !is_fully_concrete(&result)
            || infer.param_types.iter().any(|p| !is_fully_concrete(p))
            || node_types.values().any(|v| !is_fully_concrete(v))
        {
            return None;
        }
        let lambda_schemes = self.instance_lambdas(&infer, &exprs, &mut node_types);

        // The template's own bindings, now that the instance is concrete.
        let mut trial = Subst::default();
        unify(
            &mut trial,
            &Ty::Fn(t.param_patterns.clone(), Box::new(t.ret_pattern.clone())),
            &Ty::Fn(infer.param_types.clone(), Box::new(result.clone())),
        )
        .ok()?;
        let mut vars = HashSet::new();
        t.target_patterns.iter().for_each(|p| free_vars(p, &mut vars));
        let final_mapping: HashMap<TyVar, Ty> =
            vars.into_iter().map(|v| (v, trial.apply(&Ty::Var(v)))).collect();
        let target_tys: Vec<Ty> = t.target_patterns.iter().map(|p| substitute(p, &final_mapping)).collect();
        let display = display_impl_instantiation(t, &final_mapping);

        let mut call_names = HashMap::new();
        {
            let mut produced = self.produced.borrow_mut();
            let Produced {
                fn_worklist,
                impl_worklist,
                lambda_worklist,
                errors,
                ..
            } = &mut *produced;
            collect_instantiations(
                &body,
                &node_types,
                &self.global_env,
                self.templates,
                &lambda_schemes,
                HashMap::new(),
                fn_worklist,
                impl_worklist,
                lambda_worklist,
                &mut call_names,
                errors,
                self.registry,
            );
        }
        call_names.extend(infer.instance_call_names.clone());

        self.produced.borrow_mut().specializations.push((
            Some((idx, target_tys)),
            format!("{}::{}", t.algebra, t.method_name),
            display.clone(),
            Specialization {
                params: decl.params.clone(),
                body,
                param_types: infer.param_types.clone(),
                result: result.clone(),
                node_types,
                call_names,
                is_extern: t.is_extern,
                extern_symbol: t.extern_symbol.clone(),
                is_pure: t.is_pure,
            },
        ));
        self.done.borrow_mut().insert(key, (display.clone(), result.clone()));
        Some((display, result))
    }
}

impl InstanceEngine<'_> {
    /// What an instance's body needs beyond its node types: the value of
    /// each const generic it reads (`Infer::const_refs`, put back as
    /// `Ty::Const` for `cps.rs`), and its lambdas, specialized from this
    /// inference rather than the program-wide one (a comprehension's
    /// function exists only in the instance's own copy of the body). Returns
    /// the lambda schemes to resolve the body's calls with.
    fn instance_lambdas(
        &self,
        infer: &Infer,
        exprs: &[&Expr],
        node_types: &mut HashMap<NodeId, Ty>,
    ) -> HashMap<NodeId, Scheme> {
        for (id, value) in &infer.const_refs {
            if node_types.contains_key(id) {
                node_types.insert(*id, Ty::Const(value.clone()));
            }
        }
        let mut schemes = self.lambda_schemes.clone();
        for e in exprs {
            if !matches!(e.kind, ExprKind::Lambda { .. }) {
                continue;
            }
            let Some(scheme) = infer.lambda_schemes.get(&e.id) else { continue };
            let scheme = Scheme {
                ty: infer.subst.apply(&scheme.ty),
                ..scheme.clone()
            };
            let ExprKind::Lambda { body, .. } = &e.kind else { continue };
            let mut inner = Vec::new();
            collect_exprs_block(body, &mut inner);
            let types: HashMap<NodeId, Ty> = inner
                .iter()
                .filter_map(|x| node_types.get(&x.id).map(|t| (x.id, t.clone())))
                .collect();
            schemes.insert(e.id, scheme.clone());
            self.produced.borrow_mut().lambdas.push((e.id, scheme, (*e).clone(), types));
        }
        schemes
    }

    /// Infers an instance with `infer_one` (a fresh session each round), and
    /// while that inference asks for loops to be unrolled
    /// (`Infer::unroll_requests`: a loop over a collection now known to be
    /// heterogeneous), unrolls them in `decl` — this instance's own copy —
    /// prunes the `if`s that now fold, and infers again.
    fn infer_unrolling<'s>(
        &'s self,
        decl: &mut FnDecl,
        infer_one: impl Fn(&mut Infer<'s>, &FnDecl) -> Result<Ty, TypeError>,
    ) -> (Infer<'s>, Result<Ty, TypeError>) {
        const MAX_ROUNDS: usize = 16;
        let mut round = 0;
        loop {
            let mut infer = Infer::new_with_vars(self.registry, self.vars.get()).with_oracle(self);
            let outcome = infer_one(&mut infer, decl);
            self.vars.set(infer.current_vars());
            if infer.unroll_requests.is_empty() || round == MAX_ROUNDS {
                return (infer, outcome);
            }
            round += 1;
            let requests: HashMap<NodeId, (u64, u64)> = infer
                .unroll_requests
                .iter()
                .map(|(id, start, end)| (*id, (*start, *end)))
                .collect();
            let mut ids = NodeIdGen::starting_at(self.next_node.get());
            if let Some(body) = &mut decl.body {
                crate::unroll::unroll_block(body, &requests, &mut ids);
                crate::unroll::prune_block(body);
            }
            self.next_node.set(ids.current());
        }
    }

    /// The instance of generic top-level fn `name` at the concrete argument
    /// types `args`: its body inferred with them, its result whatever that
    /// body computes. Named after the scheme's own variables
    /// (`display_instantiation`), as every other path names it.
    fn specialize_fn(
        &self,
        name: &str,
        args: &[Ty],
        ret: Option<&Ty>,
        scheme_args: Option<&[Ty]>,
    ) -> Option<(String, Ty)> {
        *self.last_error.borrow_mut() = None;
        let f = *self.functions.get(name)?;
        let scheme = self.global_env.get(name)?.clone();
        if scheme.vars.is_empty() || f.is_extern || f.derivative_of.is_some() {
            return None;
        }
        f.body.as_ref()?;
        let key = format!(
            "{name}<{}>({}) -> {}",
            scheme_args
                .map(|v| v.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "))
                .unwrap_or_default(),
            args.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
            ret.map(ToString::to_string).unwrap_or_default()
        );
        // Declared generics come first among the scheme's variables.
        let declared: Option<Vec<Ty>> = scheme_args
            .filter(|v| v.len() >= f.generics.len())
            .map(|v| v[..f.generics.len()].to_vec());
        if let Some(found) = self.done.borrow().get(&key) {
            return Some(found.clone());
        }
        if !self.in_progress.borrow_mut().insert(key.clone()) {
            return None;
        }
        let mut decl = f.clone();
        let (infer, outcome) = self.infer_unrolling(&mut decl, |infer, decl| {
            infer.infer_fn_with_concrete_params(decl, args.to_vec(), ret.cloned(), &self.global_env, declared.as_deref())
        });
        self.in_progress.borrow_mut().remove(&key);
        let result = match outcome {
            Ok(r) => r,
            Err(e) => {
                *self.last_error.borrow_mut() = Some(e);
                return None;
            }
        };
        let body = decl.body.clone()?;

        let mut exprs = Vec::new();
        collect_exprs_block(&body, &mut exprs);
        let mut node_types: HashMap<NodeId, Ty> = exprs
            .iter()
            .filter_map(|e| infer.node_types.get(&e.id).map(|ty| (e.id, ty.clone())))
            .collect();
        if !is_fully_concrete(&result)
            || infer.param_types.iter().any(|p| !is_fully_concrete(p))
            || node_types.values().any(|v| !is_fully_concrete(v))
        {
            return None;
        }
        let lambda_schemes = self.instance_lambdas(&infer, &exprs, &mut node_types);
        // The call's own instantiation names the instance when it gave one: a
        // generic only a turbofish fixes (`N` in `fn probe<const N: i32>() ->
        // i32`) appears nowhere in the signature to be recovered from.
        let concrete_tys: Vec<Ty> = match scheme_args {
            Some(given) if given.len() == scheme.vars.len() => given.to_vec(),
            _ => {
                let mut trial = Subst::default();
                if unify(
                    &mut trial,
                    &scheme.ty,
                    &Ty::Fn(infer.param_types.clone(), Box::new(result.clone())),
                )
                .is_err()
                {
                    return None;
                }
                // Variables only field accesses determine (`n.l1`'s type),
                // absent from the signature.
                resolve_field_constraints(&scheme, &mut trial, self.registry);
                scheme.vars.iter().map(|v| trial.apply(&Ty::Var(*v))).collect()
            }
        };
        if !concrete_tys.iter().all(is_fully_concrete) {
            return None;
        }
        let display = display_instantiation(name, &concrete_tys);

        let mut call_names = HashMap::new();
        {
            let mut produced = self.produced.borrow_mut();
            let Produced {
                fn_worklist,
                impl_worklist,
                lambda_worklist,
                errors,
                ..
            } = &mut *produced;
            collect_instantiations(
                &body,
                &node_types,
                &self.global_env,
                self.templates,
                &lambda_schemes,
                HashMap::new(),
                fn_worklist,
                impl_worklist,
                lambda_worklist,
                &mut call_names,
                errors,
                self.registry,
            );
        }
        call_names.extend(infer.instance_call_names.clone());

        self.produced.borrow_mut().specializations.push((
            None,
            name.to_string(),
            display.clone(),
            Specialization {
                params: decl.params.clone(),
                body,
                param_types: infer.param_types.clone(),
                result: result.clone(),
                node_types,
                call_names,
                is_extern: false,
                extern_symbol: None,
                is_pure: f.attrs.iter().any(|a| a.name == "pure"),
            },
        ));
        self.done.borrow_mut().insert(key, (display.clone(), result.clone()));
        Some((display, result))
    }
}

impl InstanceOracle for InstanceEngine<'_> {
    fn fn_instance(
        &self,
        name: &str,
        args: &[Ty],
        ret: Option<&Ty>,
        scheme_args: Option<&[Ty]>,
    ) -> Option<(String, Ty)> {
        self.specialize_fn(name, args, ret, scheme_args)
    }

    fn impl_instance(&self, algebra: &str, method: &str, args: &[Ty], ret: Option<&Ty>) -> Option<(String, Ty)> {
        // The call's result, when the caller fixed it (an output-only target,
        // a broadcast's shape); otherwise an unbound variable no template can
        // share a number with.
        let ret = ret.cloned().unwrap_or(Ty::Var(TyVar(u32::MAX - 1)));
        match derive_impl_instantiation_for(self.templates, self.registry, Some(algebra), method, args, &ret) {
            ImplMatch::FoundConcrete(idx) => {
                let t = &self.templates[idx];
                Some((display_impl_instantiation(t, &HashMap::new()), t.ret_pattern.clone()))
            }
            ImplMatch::Found(idx, mapping) => self.specialize_impl(idx, &mapping, args),
            _ => None,
        }
    }
}
