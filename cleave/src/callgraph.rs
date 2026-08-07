//! Whole-program inference over top-level `fn`s: resolves calls between
//! *separately declared* functions properly — direct self-recursion (see
//! `infer.rs`'s `infer_fn` for the single-function case this builds on) and
//! arbitrary mutual recursion between two or more `fn`s, in any order of
//! declaration, "de façon définitive" as requested, not just deferred
//! further.
//!
//! ## Why this is a separate pass from `infer_fn` itself
//!
//! The ideal shape, as first sketched in conversation, is a single lazy
//! recursive walk: when inference hits a call to some other top-level `fn`,
//! it recurses *right there* into inferring that callee (memoized by name, a
//! Tarjan-style index/low-link per name detecting when a cycle closes), with
//! no separate graph-construction step at all. That shape doesn't survive
//! contact with Rust's ownership rules unchanged: the recursive trigger
//! would have to happen from *inside* `infer_call`, which only ever holds
//! `&mut self` on the *one* `Infer` doing the current function's own
//! inference — there's no way for it to also reach back into an external
//! table/stack of *other* functions' in-progress state without threading
//! that state through every single inference method (`infer_expr`,
//! `infer_block`, ... — all of them, since a call can be arbitrarily deeply
//! nested) just to serve this one caller.
//!
//! What's built instead keeps the substance of that design and only changes
//! *when* the graph is known: a cheap, type-free scan of each function's own
//! body (`collect_calls`, just "which other known `fn` names does this text
//! mention", no inference at all) builds the call graph up front, and
//! **Tarjan's SCC algorithm** (`Tarjan::run`, an ordinary, self-contained
//! graph algorithm — no `Infer` involved) determines both the grouping
//! (which functions are mutually recursive) and the processing order
//! (callees' groups before callers', which is exactly the pop order Tarjan's
//! own recursion produces). The actual type inference is then still done by
//! *one* mechanism, run once per group — `Infer::infer_fn_raw` — matching
//! the original intuition: real recursion in the inference itself, just with
//! the grouping worked out first rather than interleaved.
//!
//! ## Why a whole group shares one `Infer`
//!
//! Two mutually-recursive functions can end up sharing type variables
//! *through* their calls to each other (`is_even`/`is_odd`, each calling the
//! other with the other's own decremented parameter) even though they start
//! out as textually separate `Param` bindings — unifying those variables
//! together requires one shared `Subst`. Only *between* groups does the
//! boundary become "just a finished `Scheme` crosses over, fresh
//! variables via `instantiate` at every use" — the same boundary `let`
//! generalization already relies on.
//!
//! ## Why defaulting/constraint-checking happens once per *group*, not once
//! per function
//!
//! `apply_defaults`/`check_pending_constraints` must run only after *every*
//! member of a mutually-recursive group has had its body walked — running
//! them right after the first member (as an earlier draft of this module
//! did, calling `infer_fn`'s full sequence per member) can default a numeric
//! literal, or check a constraint, before a still-pending sibling has
//! contributed everything it might additionally unify against that same
//! variable. Hence `infer_fn_raw` (no defaulting/constraint-check) run once
//! per member, then the defaulting/constraint/placeholder steps run exactly
//! once for the whole group.
//!
//! ## Generalization
//!
//! Once a whole group's bodies are inferred, every member that didn't fail
//! is generalized — together, against `global_env` (the schemes already
//! finished from *earlier* groups), never against the group's own
//! in-progress `Env` (which still holds this group's own siblings as
//! monomorphic placeholders; counting those as "free in the environment"
//! would wrongly block generalization of a variable two siblings still
//! genuinely share). This is deliberately the same "top-level `fn`s are
//! `let`-polymorphic too" answer the pseudocode implied ("généralise ce
//! groupe entier ensemble, marque chacun Fini(scheme)") — every call site of
//! a finished function gets its own fresh instantiation, exactly like a
//! generalized `let`.
//!
//! `impl` methods are out of scope here — they dispatch through the algebra
//! registry (`infer_call`/`infer_algebra_call`), not by bare name lookup in
//! `env`, so mutual recursion between `impl` methods "resolves" already
//! (the registry doesn't care whether the impl's own body has finished being
//! inferred) — that's the separate, still-open circular-primitive-impl
//! problem, not this one.

use crate::ast::*;
use crate::infer::{check_no_placeholder, Constraint, Env, Infer, Scheme, Ty, TypeError, TypeErrorKind};
use crate::registry::Registry;
use std::collections::{HashMap, HashSet};

/// One top-level `fn`'s outcome from a whole-program inference run.
#[derive(Debug)]
pub struct FnResult {
    pub param_types: Vec<Ty>,
    pub result: Ty,
}

pub struct ProgramInference {
    /// One entry per top-level `fn` in the program, keyed by name.
    pub results: HashMap<String, Result<FnResult, TypeError>>,
    pub node_types: HashMap<NodeId, Ty>,
    /// Every top-level `fn`'s finished, generalized `Scheme`, by name — the
    /// same `Env` this pass builds up internally, exposed so `dump.rs` can
    /// seed an `impl` method's own body with it
    /// (`Infer::infer_impl_fn_generic_with_env`), letting it call an
    /// ordinary top-level function by name instead of always hitting
    /// `infer_call`'s unresolved-call placeholder (see that method's own
    /// doc comment for the bug this closes).
    pub global_env: Env,
}

/// Runs whole-program inference over every top-level `fn` in `program` (see
/// module docs). `impl` methods are untouched — callers still infer those
/// exactly as before (`Infer::infer_impl_fn`, one at a time).
pub fn infer_program(program: &Program, registry: &Registry) -> ProgramInference {
    let functions: HashMap<String, &FnDecl> = program
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::Fn(f) => Some((f.name.clone(), f)),
            _ => None,
        })
        .collect();
    // A top-level `fn`'s own enclosing `Item` span — `FnDecl` itself
    // carries none (see `ast.rs`) — needed only to report a bodyless one
    // (legal grammatically, see `grammar.pest`'s own `fn_decl` comment, but
    // never legal for a top-level `fn` specifically).
    let item_spans: HashMap<String, Span> = program
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::Fn(f) => Some((f.name.clone(), item.span)),
            _ => None,
        })
        .collect();

    let known: HashSet<&str> = functions.keys().map(String::as_str).collect();
    let mut graph: HashMap<String, Vec<String>> = HashMap::new();
    for (name, f) in &functions {
        let mut calls = Vec::new();
        if let Some(body) = &f.body {
            collect_calls_block(body, &known, &mut calls);
        }
        graph.entry(name.clone()).or_default().extend(calls);
    }

    let sccs = Tarjan::run(&graph);

    let mut global_env: Env = Env::new();
    let mut results: HashMap<String, Result<FnResult, TypeError>> = HashMap::new();
    let mut node_types: HashMap<NodeId, Ty> = HashMap::new();

    for group in &sccs {
        let mut infer = Infer::new(registry);

        // Seed every member's placeholder before inferring *any* of their
        // bodies — visible to every other member (mutual recursion) and to
        // itself (self-recursion).
        let mut placeholders: HashMap<String, (Vec<Ty>, Ty, HashMap<String, Ty>)> = HashMap::new();
        let mut group_env = global_env.clone();
        for name in group {
            let f = functions[name.as_str()];
            let (param_types, ret_var, generics) = infer.fresh_fn_shape(f);
            group_env.insert(name.clone(), Scheme::mono(Ty::Fn(param_types.clone(), Box::new(ret_var.clone()))));
            placeholders.insert(name.clone(), (param_types, ret_var, generics));
        }

        let mut raw_results: HashMap<String, Result<Ty, TypeError>> = HashMap::new();
        for name in group {
            let f = functions[name.as_str()];
            if f.body.is_none() {
                let span = item_spans[name.as_str()];
                // `extern fn foo(...) -> ...;` — legal without a body (see
                // `ast.rs`'s own `FnDecl::is_extern` doc comment): there's
                // no body to infer, so bind the placeholder's own `ret_var`
                // straight to the declared return type instead (mirroring
                // what body inference would otherwise do by unifying it
                // against the body's own result) and treat that as this
                // member's outcome, same shape any other successful member
                // produces below.
                if f.is_extern {
                    if !f.generics.is_empty() {
                        raw_results.insert(
                            name.clone(),
                            Err(TypeError { span, kind: TypeErrorKind::ExternFnCannotBeGeneric { name: f.name.clone() } }),
                        );
                        continue;
                    }
                    // Mirrors `infer_fn_raw`'s own two steps for a real
                    // body (infer the result, then tie `ret_var` to it) —
                    // here the declared return type stands in for "what the
                    // body computed", and the outcome stored is that same
                    // bare return `Ty`, not `Ty::Fn(params, ret)`, matching
                    // exactly what `infer_fn_raw` itself returns.
                    let (_, ret_var, _) = placeholders[name].clone();
                    let declared_ret =
                        f.ret.as_ref().map(|t| infer.ty_from_ast(t)).unwrap_or_else(|| Ty::Con("()".to_string()));
                    let outcome = infer
                        .unify_at(span, &ret_var, &declared_ret)
                        .and_then(|()| infer.check_pending_type_names())
                        .map(|()| declared_ret);
                    raw_results.insert(name.clone(), outcome);
                    continue;
                }
                raw_results.insert(
                    name.clone(),
                    Err(TypeError { span, kind: TypeErrorKind::MissingFnBody { name: f.name.clone() } }),
                );
                continue;
            }
            let (param_types, ret_var, generics) = placeholders[name].clone();
            // `check_pending_type_names` right here, per member, not folded
            // into the group-wide `check_pending_constraints` sweep below —
            // see that method's own doc comment for why: each entry is
            // anchored to the one function whose body produced it, not a
            // property of the group as a whole.
            let outcome = infer
                .infer_fn_raw(f, &group_env, param_types, ret_var, &generics)
                .and_then(|ty| infer.check_pending_type_names().map(|()| ty));
            raw_results.insert(name.clone(), outcome);
        }

        // Generalize every successfully-inferred member that takes at least
        // one parameter — *before* defaulting runs, mirroring how a `let`'s
        // own `generalize` call always happens during body inference, well
        // before `finish_fn`'s defaulting. Doing this the other way around
        // is a real bug, found by testing: a variable must be marked
        // quantified (`Infer::quantified`, populated by `generalize` itself)
        // *before* `apply_defaults` runs, or `apply_defaults` — which skips
        // any variable already in that set — never gets the chance to leave
        // it alone; a generalizable top-level `fn add_one(x) { x + 1 }`,
        // processed alone in its own singleton group with nothing yet
        // forcing `x` concrete, would otherwise get silently defaulted
        // straight to `i32` and permanently monomorphized before
        // `generalize` ever ran.
        //
        // A **nullary** member is deliberately *not* generalized at all —
        // this is Haskell's Monomorphism Restriction, applied for the same
        // reason it exists there: a binding with no parameters can't accept
        // any caller-supplied type information at its call sites the way a
        // parameterized one can, so "generalizing" it would only mean
        // silently recomputing a fresh, possibly differently-defaulted
        // result at every use — surprising, and pointless for something that
        // takes no input to begin with. It also keeps `main` itself (always
        // nullary) reporting a concrete signature rather than a spuriously
        // "generic" one.
        let mut nullary_constraints: HashMap<String, Vec<Constraint>> = HashMap::new();
        for name in group {
            let Ok(_) = &raw_results[name] else { continue };
            let f = functions[name.as_str()];
            let (param_types, ret_var, _) = &placeholders[name];
            let final_fn_ty = Ty::Fn(
                param_types.iter().map(|t| infer.subst.apply(t)).collect(),
                Box::new(infer.subst.apply(ret_var)),
            );
            if f.params.is_empty() {
                // Not generalized (Monomorphism Restriction — see below) —
                // but a pending constraint on a variable this nullary
                // member's own returned type still exposes must still
                // survive into its `Scheme::mono`, captured here (this
                // group's own `Infer`, and `self.constraints` along with
                // it, is discarded once this group finishes) for the
                // second loop below to attach. See `Infer::constraints_
                // touching`'s own doc comment for the bug this closes.
                nullary_constraints.insert(name.clone(), infer.constraints_touching(&final_fn_ty));
                continue;
            }
            // Generalized against `global_env` — the group's own siblings
            // (`group_env`) must not count as "free in the environment", see
            // module docs.
            let scheme = infer.generalize(&global_env, &final_fn_ty);
            global_env.insert(name.clone(), scheme);
        }

        infer.apply_defaults();
        // A constraint failure here is a property of the group's mutual
        // definition as a whole, not attributable to one specific member —
        // reported against every member whose own body-inference otherwise
        // succeeded (a raw-inference failure is already more specific and is
        // left alone).
        if let Err(e) = infer.check_pending_constraints() {
            for name in group {
                if let Some(r @ Ok(_)) = raw_results.get_mut(name) {
                    *r = Err(e.clone());
                }
            }
        }

        // Read uniformly *after* defaulting, for every member — safe now
        // that `apply_defaults` itself refuses to touch a variable
        // `generalize` already quantified above, so a generalized member's
        // own reported shape (and its body's `node_types`) still correctly
        // shows a still-open variable as generic, not a misleading default;
        // only a variable that ended up neither concretely pinned nor
        // quantified by anything gets defaulted here.
        for name in group {
            match raw_results.remove(name).unwrap() {
                Ok(result_ty) => {
                    let f = functions[name.as_str()];
                    let (param_types, _, _) = &placeholders[name];
                    let final_params: Vec<Ty> = param_types.iter().map(|t| infer.subst.apply(t)).collect();
                    let final_result = infer.subst.apply(&result_ty);

                    match check_no_placeholder(f, &final_result, &final_params) {
                        Ok(()) => {
                            // A nullary member never gets *generalized*
                            // above (Monomorphism Restriction — see the
                            // first loop's own comment) and so was never
                            // added to `global_env` at all until this point
                            // — a real bug, found by testing: any *other*
                            // function calling a nullary helper (`fn make()
                            // -> Vec2 { ... } fn main() { make() }`) silently
                            // fell through to `infer_call`'s unresolved-call
                            // placeholder, since `env.get("make")` never
                            // found anything. Not generalizing a nullary
                            // binding only ever meant "don't quantify its
                            // free variables" — it never meant "don't expose
                            // it to callers at all". Inserted with an empty
                            // `vars` (matching a parameterized member's own
                            // `Scheme`, minus any quantified vars) using the
                            // exact same fully-resolved type `FnResult`
                            // itself reports, so a later caller's
                            // `instantiate` (a no-op on an unquantified
                            // scheme) sees the real, concrete answer — but
                            // *with* `nullary_constraints`' own entry
                            // attached (not the true empty-`Scheme::mono` a
                            // parameterless binding might suggest — see
                            // `Infer::constraints_touching`'s own doc
                            // comment for why an empty constraint list here
                            // would silently lose a real, checkable
                            // requirement).
                            if f.params.is_empty() {
                                let ty = Ty::Fn(final_params.clone(), Box::new(final_result.clone()));
                                let constraints = nullary_constraints.remove(name).unwrap_or_default();
                                global_env.insert(
                                    name.clone(),
                                    Scheme { vars: Vec::new(), constraints, ty, const_widths: HashMap::new() },
                                );
                            }
                            results
                                .insert(name.clone(), Ok(FnResult { param_types: final_params, result: final_result }));
                        }
                        Err(e) => {
                            results.insert(name.clone(), Err(e));
                        }
                    }
                }
                Err(e) => {
                    results.insert(name.clone(), Err(e));
                }
            }
        }

        node_types.extend(infer.node_types.iter().map(|(id, t)| (*id, infer.subst.apply(t))));
    }

    ProgramInference { results, node_types, global_env }
}

// ------------------------------------------------------------ static call-graph scan

/// Collects, into `out`, the name of every `known` top-level `fn` referenced
/// anywhere in `block` — as a call (`Call`) or as a bare value (`Path`, e.g.
/// passing a function by name without calling it; still a genuine
/// type-dependency edge). Purely syntactic — no type information, so this
/// may (harmlessly) over-approximate the real call graph, e.g. a name that
/// happens to collide with a `let`-bound local never actually resolving to
/// the top-level `fn` at inference time. Grouping a few extra functions
/// together than strictly necessary just means generalizing them slightly
/// later than the ideal — still sound, never incorrect.
fn collect_calls_block(block: &Block, known: &HashSet<&str>, out: &mut Vec<String>) {
    for stmt in &block.stmts {
        match &stmt.kind {
            StmtKind::Let { value, .. } => collect_calls_expr(value, known, out),
            StmtKind::Assign { target, value } => {
                collect_calls_expr(target, known, out);
                collect_calls_expr(value, known, out);
            }
            StmtKind::Expr(e) => collect_calls_expr(e, known, out),
        }
    }
    if let Some(tail) = &block.tail {
        collect_calls_expr(tail, known, out);
    }
}

fn collect_calls_expr(expr: &Expr, known: &HashSet<&str>, out: &mut Vec<String>) {
    match &expr.kind {
        ExprKind::NumberLit { .. } | ExprKind::ImaginaryLit { .. } | ExprKind::BoolLit(_) => {}
        ExprKind::Path(p) => {
            let name = p.segments.join("::");
            if known.contains(name.as_str()) {
                out.push(name);
            }
        }
        ExprKind::Call(path, _, args, ..) => {
            let name = path.segments.join("::");
            if known.contains(name.as_str()) {
                out.push(name);
            }
            for a in args {
                collect_calls_expr(a, known, out);
            }
        }
        ExprKind::FieldAccess(b, _) => collect_calls_expr(b, known, out),
        ExprKind::MethodCall(b, _, args) => {
            collect_calls_expr(b, known, out);
            for a in args {
                collect_calls_expr(a, known, out);
            }
        }
        ExprKind::Index(b, i) => {
            collect_calls_expr(b, known, out);
            collect_calls_expr(i, known, out);
        }
        ExprKind::ArrayLit(elems) => {
            for e in elems {
                collect_calls_expr(e, known, out);
            }
        }
        ExprKind::ArrayRepeat { value, count } => {
            collect_calls_expr(value, known, out);
            collect_calls_expr(count, known, out);
        }
        ExprKind::StructLit(_, _, fields) => {
            for (_, v) in fields {
                collect_calls_expr(v, known, out);
            }
        }
        ExprKind::If { cond, then_branch, else_branch } => {
            collect_calls_expr(cond, known, out);
            collect_calls_block(then_branch, known, out);
            if let Some(eb) = else_branch {
                match &**eb {
                    ElseBranch::If(e) => collect_calls_expr(e, known, out),
                    ElseBranch::Block(b) => collect_calls_block(b, known, out),
                }
            }
        }
        ExprKind::While { cond, body } => {
            collect_calls_expr(cond, known, out);
            collect_calls_block(body, known, out);
        }
        ExprKind::For { start, end, body, .. } => {
            collect_calls_expr(start, known, out);
            collect_calls_expr(end, known, out);
            collect_calls_block(body, known, out);
        }
        ExprKind::Block(b) => collect_calls_block(b, known, out),
        ExprKind::Lambda { body, .. } => collect_calls_block(body, known, out),
    }
}

// ------------------------------------------------------------ Tarjan's SCC algorithm

/// Standard Tarjan's strongly-connected-components algorithm over the static
/// call graph. Produces groups in the order they're identified (popped) —
/// which, for a DFS following call edges, is always "callee's group before
/// caller's group": a group can only be popped once every function it calls
/// has itself already been assigned to some group, so a function's own
/// callees are always fully resolved (and, here, already generalized into
/// `global_env`) by the time its own group is processed.
struct Tarjan<'a> {
    graph: &'a HashMap<String, Vec<String>>,
    index_of: HashMap<String, usize>,
    low_link: HashMap<String, usize>,
    on_stack: HashSet<String>,
    stack: Vec<String>,
    next_index: usize,
    sccs: Vec<Vec<String>>,
}

impl<'a> Tarjan<'a> {
    fn run(graph: &'a HashMap<String, Vec<String>>) -> Vec<Vec<String>> {
        let mut names: Vec<&String> = graph.keys().collect();
        // Sorted purely for deterministic output across runs — Tarjan's
        // correctness doesn't depend on visit order, only reproducibility of
        // e.g. test assertions and CLI output does.
        names.sort();

        let mut t = Tarjan {
            graph,
            index_of: HashMap::new(),
            low_link: HashMap::new(),
            on_stack: HashSet::new(),
            stack: Vec::new(),
            next_index: 0,
            sccs: Vec::new(),
        };
        for name in names {
            if !t.index_of.contains_key(name) {
                t.strongconnect(name);
            }
        }
        t.sccs
    }

    fn strongconnect(&mut self, name: &str) {
        let index = self.next_index;
        self.next_index += 1;
        self.index_of.insert(name.to_string(), index);
        self.low_link.insert(name.to_string(), index);
        self.stack.push(name.to_string());
        self.on_stack.insert(name.to_string());

        if let Some(edges) = self.graph.get(name) {
            for callee in edges.clone() {
                if !self.index_of.contains_key(&callee) {
                    self.strongconnect(&callee);
                    let callee_low = self.low_link[&callee];
                    let entry = self.low_link.get_mut(name).unwrap();
                    *entry = (*entry).min(callee_low);
                } else if self.on_stack.contains(&callee) {
                    let callee_index = self.index_of[&callee];
                    let entry = self.low_link.get_mut(name).unwrap();
                    *entry = (*entry).min(callee_index);
                }
            }
        }

        if self.low_link[name] == self.index_of[name] {
            let mut group = Vec::new();
            loop {
                let w = self.stack.pop().expect("root's own strongconnect pushed it, so the stack can't be empty");
                self.on_stack.remove(&w);
                let is_root = w == name;
                group.push(w);
                if is_root {
                    break;
                }
            }
            self.sccs.push(group);
        }
    }
}
