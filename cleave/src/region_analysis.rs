//! Determines which top-level `fn`s are safe to lower with every heap
//! construction site inside their own body drawing from the *currently
//! open* region (`cleave_alloc_local`) instead of the ordinary heap
//! (`cleave_alloc_rc`) — `doc/hld.md`'s own "Memory management" section,
//! the concrete first application of the arena `cleave-rt` already builds
//! (`cleave-rt/src/lib.rs`'s own `cleave_region_enter`/`cleave_alloc_local`/
//! `cleave_region_exit`).
//!
//! **The real target, found directly, not assumed**: `examples/mnist-
//! interop`'s own training loop (`for s in 0..n { let x = ...; let y =
//! ...; let g = net_grad(x, y, net); let r = Optimizer::step(opt, net,
//! g.2, state); net = r.0; state = r.1; }`) — `net_grad`'s own result
//! (`g`) is read once, by `Optimizer::step`, and never carried past this
//! same iteration; `Optimizer::step`'s own result (`r.0`/`r.1`) *becomes*
//! `net`/`state`, carried into the next iteration. `net_grad` is safe to
//! mark region-local; `Optimizer::step` is not.
//!
//! **Why this needs to run *before* `--inline`, at the CPS level, not as
//! an MLIR-level pass over the fully-inlined kernel** — a real design
//! correction, not the first thing tried: a `net`/`state` struct value is
//! always a bare `!llvm.ptr` at the MLIR level, so a *backward* walk from
//! `scf.while`'s own `scf.yield` operands only ever discovers the
//! *outermost* allocation (`Network`'s own struct storage) directly —
//! nested fields (`Dense`'s own `w`/`b` tensors, reached through an
//! `llvm.store` into the outer struct's own field slot, not through any
//! SSA value flow) are invisible to that walk entirely, and would be
//! wrongly classified safe. At the CPS level, by contrast, `region_enter`/
//! `region_exit` only need to wrap the *call site* itself (`lower_real_
//! call`, `mlir_lower.rs`) — `--inline` (which runs afterward) splices the
//! callee's own body in place of that call, so *everything* the callee
//! itself allocates, at any depth, automatically ends up between the two
//! markers, with no separate tracing needed for nested allocations at all.
//!
//! **The real precondition this module checks, not just assumes**: marking
//! a function's own allocation sites region-local is only sound if *every*
//! call site targeting it, anywhere in the whole program, is itself proven
//! safe — a call found strictly inside some loop's own repeating body,
//! whose own result never reaches that loop's own carried (escaping)
//! state, checked structurally, the same "no dataflow fixpoint needed, the
//! structure already says so" argument `doc/hld.md`'s own "Memory
//! management" section makes for CPS-level lifetimes generally, applied
//! here to *which* top-level function a call targets rather than to one
//! local value's own liveness.
//!
//! **`doc/plan-region-arena.md`'s own "Step 2" — a real generalization of
//! this module, not the original shape.** The original version of this
//! analysis required a callee to have *exactly one* call site in the whole
//! program before marking it region-local — sufficient, but needlessly
//! strict: it also rejected a callee with two or more call sites even when
//! *every one* of them was individually proven just as safe (found by
//! direct testing against this module's own test suite — see `cleave/
//! tests/region_analysis.rs`'s own `a_function_called_from_more_than_one_
//! place_is_never_marked_local`, whose two call sites, both inside the same
//! loop body, both turn out to be individually non-escaping; that test was
//! updated alongside this change, not left encoding the old, narrower
//! rule). `analyze` below computes the sound, general condition directly:
//! a callee is region-local iff *every* call site targeting it, anywhere in
//! the program, is in `safe_sites` — the true special case of which the old
//! `call_counts == 1` rule was one easy, always-safe instance (a single
//! call site, itself already proven safe, trivially satisfies "every site
//! safe"). This generalization also *transitively* absorbs a shared helper
//! called from two or more *already-region-local* functions' own bodies —
//! previously excluded outright regardless of how safe each caller was
//! individually — using `HashSet<CVar>` membership (call-site *identity*,
//! `CVar`s already unique across the whole program, `refcount::insert_
//! refcounting`'s own `max_cvar_in_program` establishes this same fact)
//! rather than a numeric tally, precisely so that discovering the same call
//! site "safe" twice, through two different code paths, is a harmless,
//! idempotent no-op rather than a double-count that could ever inflate a
//! callee's own safe-occurrence count past its real, physical total.
//!
//! **What this generalization does *not* by itself fix — `region_
//! specialize.rs`'s own job**: a callee genuinely shared between a
//! safe context and an unsafe one (`doc/backlog.md`'s own real example:
//! a shared algebra function called from both the training loop, where its
//! result never escapes one iteration, and `evaluate()`, where it doesn't
//! run inside a loop at all) still, correctly, fails "every site safe" and
//! stays excluded here. `region_specialize::specialize_region_local_
//! functions` is the *other* half of Step 2: it runs earlier in the
//! pipeline, *before* this analysis is ever consulted, and duplicates such
//! a genuinely-mixed callee into two names — the original, still serving
//! its unsafe call sites, and a `{name}$region` copy, retargeted to serve
//! only the sites this same `analyze` function already proves safe. This
//! module needs no knowledge of that split at all: by the time `analyze`
//! runs, `{name}$region` is just another top-level function whose *every*
//! call site happens to be safe, by construction.

use crate::cps::{CExpr, CFunDef, CVal, CVar, CpsProgram, PrimOp};
use std::collections::{HashMap, HashSet};

/// The whole public surface: every top-level function name safe to lower
/// with `cleave_alloc_local` at each of its own construction sites. A thin
/// wrapper over `analyze` (`pub(crate)`, see its own doc comment) — kept as
/// a separate, stable, `pub` entry point since `mlir_lower.rs`'s own single
/// call site (`lower_program`) only ever needs the per-function verdict,
/// never the finer-grained `safe_sites`/`sites_by_callee` `region_
/// specialize.rs` additionally needs.
pub fn find_region_local_functions(program: &CpsProgram) -> HashSet<String> {
    analyze(program).region_local
}

/// The full result of this module's own whole-program fixed point — see
/// `analyze`'s own doc comment. `pub(crate)`, not `pub`: consumed by
/// `region_specialize.rs` (a sibling module, same crate) alongside `find_
/// region_local_functions`'s own narrower public contract; no external
/// caller has a legitimate use for `safe_sites`/`sites_by_callee` on their
/// own.
pub(crate) struct RegionAnalysis {
    pub(crate) region_local: HashSet<String>,
    /// Every individual call site — identified by its own bound `result_
    /// var`, a `CVar`, unique across the *whole* program — proven safe,
    /// Kind-1 (found directly inside some loop's own repeating body,
    /// individually non-escaping) or Kind-2 (found, transitively, directly
    /// inside an already-`region_local` function's own body — no separate
    /// escaping check needed there, `collect_direct_callees`'s own doc
    /// comment has the soundness argument, unchanged from the original
    /// version of this module).
    pub(crate) safe_sites: HashSet<CVar>,
    /// Every real top-level call site anywhere in the program, `result_
    /// var`-identified, grouped by callee name — built once, exhaustively
    /// (`collect_all_call_sites_in`, unconditional recursion, loops or not,
    /// exactly `count_calls_in`'s own old shape but keeping each site's own
    /// identity instead of only a running count). `region_specialize.rs`'s
    /// own "does this callee need a `$region` split" test is exactly "some,
    /// but not all, of `sites_by_callee[callee]` are in `safe_sites`".
    pub(crate) sites_by_callee: HashMap<String, Vec<CVar>>,
}

/// The shared whole-program fixed point behind both `find_region_local_
/// functions`'s own public, per-function verdict and `region_specialize::
/// specialize_region_local_functions`'s own need for individual call-site
/// identity (this module's own top doc comment has the full reasoning for
/// why a `HashSet<CVar>` of call-site identities, not a numeric tally, is
/// what keeps this sound under transitive propagation).
///
/// Bounded, like the original version's own transitive-descent worklist:
/// each outer iteration either adds at least one name to `region_local` (at
/// most `top_level_names.len()` times) or the loop ends, and each iteration
/// does at most one body walk per already-`region_local` name.
pub(crate) fn analyze(program: &CpsProgram) -> RegionAnalysis {
    let top_level_names: HashSet<String> = program.funcs.iter().map(|f| f.def.name.clone()).collect();
    let by_name: HashMap<&str, &CFunDef> = program
        .funcs
        .iter()
        .map(|f| (f.def.name.as_str(), &f.def))
        .collect();

    // Every real call site anywhere, by identity, grouped by callee —
    // built once, exhaustively, so neither the fixed point below nor `region_
    // specialize.rs` ever needs to re-walk the whole program per candidate.
    let mut sites_by_callee: HashMap<String, Vec<CVar>> = HashMap::new();
    for f in &program.funcs {
        let mut sites = Vec::new();
        collect_all_call_sites_in(&f.def.body, &top_level_names, &mut sites);
        for (callee, v) in sites {
            sites_by_callee.entry(callee).or_default().push(v);
        }
    }

    // Kind-1 seed: every call site found directly inside some loop's own
    // repeating body, individually proven non-escaping. `find_loops_and_
    // mark`/`analyze_loop_body` below are structurally identical to the
    // original version of this module — only *what* a proven-safe
    // occurrence does changes (recording its own site identity in `safe_
    // sites`, unconditionally, rather than gating on `call_counts == 1`
    // before inserting the *callee name* into a set).
    let mut safe_sites: HashSet<CVar> = HashSet::new();
    for f in &program.funcs {
        find_loops_and_mark(&f.def.body, &top_level_names, &mut safe_sites);
    }

    // Fixed point: a top-level function is wholesale region-local once
    // *every* one of its own call sites is in `safe_sites`; once it is,
    // every direct callee found anywhere in *its own* body is safe too (no
    // separate escaping check needed there — `collect_direct_callees`'s own
    // soundness argument, unchanged from the original version of this
    // module), which can in turn make some *other* function's own call
    // sites all-safe, and so on.
    let mut region_local: HashSet<String> = HashSet::new();
    loop {
        let mut changed = false;
        for name in &top_level_names {
            if region_local.contains(name) {
                continue;
            }
            let Some(sites) = sites_by_callee.get(name) else {
                continue;
            };
            if !sites.is_empty() && sites.iter().all(|v| safe_sites.contains(v)) {
                region_local.insert(name.clone());
                changed = true;
            }
        }
        if !changed {
            break;
        }
        // Propagate Kind-2 safety from *every* currently-region-local
        // function's own body -- re-walking already-processed names each
        // pass is redundant work (re-inserting an already-safe `CVar` is a
        // harmless no-op) but simpler than tracking a separate delta set,
        // and still bounded by `top_level_names.len()` outer iterations.
        for name in region_local.clone() {
            let Some(def) = by_name.get(name.as_str()) else {
                continue;
            };
            let mut inner_sites = Vec::new();
            collect_all_call_sites_in(&def.body, &top_level_names, &mut inner_sites);
            for (_, v) in inner_sites {
                safe_sites.insert(v);
            }
        }
    }

    RegionAnalysis {
        region_local,
        safe_sites,
        sites_by_callee,
    }
}

/// Every top-level function name called *directly* anywhere in `expr` (any
/// nesting of `If`/`Fix`) — a thin wrapper over `collect_direct_call_sites`
/// below (this module's own risk otherwise: re-implementing the identical
/// `Fix{[k], App(callee, args)}` call-shape match a second time, silently
/// drifting apart from it over time).
///
/// **A second caller, outside this module**: `mlir_lower.rs::lower_loop`
/// reuses this directly (`pub(crate)`) on one specific loop's own already-
/// extracted `then_branch`, intersecting the result against `region_local_
/// fns` to decide whether *this* loop's own iteration needs `cleave_region_
/// enter`/`cleave_region_exit` wrapped around it at all — `doc/backlog.md`'s
/// own "every loop iteration... unconditionally opens and closes a region"
/// finding (a real, VTune-confirmed `486`-million-call cost on the real
/// `mnist-interop` kernel, `cleave_region_enter` itself near-free per call
/// but never skipped even when nothing inside a given loop ever allocates
/// region-locally at all) is exactly what this fixes.
pub(crate) fn collect_direct_callees(expr: &CExpr, top_level_names: &HashSet<String>, out: &mut HashSet<String>) {
    let mut sites = Vec::new();
    collect_direct_call_sites(expr, top_level_names, &mut sites);
    out.extend(sites.into_iter().map(|(name, _)| name));
}

/// `collect_direct_callees`'s own identical call-shape recognition,
/// additionally recording each call site's own bound `result_var` — needed
/// by `analyze`'s own fixed point (Kind-2 propagation: a specific call
/// site's own identity, not just its callee's name, is what `safe_sites`
/// tracks).
fn collect_direct_call_sites(expr: &CExpr, top_level_names: &HashSet<String>, out: &mut Vec<(String, CVar)>) {
    match expr {
        CExpr::LetPrim { cont, .. } => collect_direct_call_sites(cont, top_level_names, out),
        CExpr::App { .. } => {}
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_direct_call_sites(then_branch, top_level_names, out);
            collect_direct_call_sites(else_branch, top_level_names, out);
        }
        CExpr::Fix { defs, body } => {
            if let [k] = defs.as_slice() {
                if let [result_var] = k.params[..] {
                    if let CExpr::App {
                        func: CVal::Label(callee),
                        args,
                    } = &**body
                    {
                        let targets_k = args
                            .last()
                            .map(|a| matches!(a, CVal::Label(n) if n == &k.name))
                            .unwrap_or(false);
                        if targets_k && top_level_names.contains(callee) {
                            out.push((callee.clone(), result_var));
                        }
                    }
                }
            }
            for d in defs {
                collect_direct_call_sites(&d.body, top_level_names, out);
            }
            collect_direct_call_sites(body, top_level_names, out);
        }
    }
}

/// Every real top-level call site anywhere in `expr`, `result_var`-
/// identified, grouped by callee name by `analyze`'s own caller —
/// unconditional recursion through every `Fix` def's own body regardless of
/// `carried_types` (loop or not), the same uniform shape the original
/// version of this module's own `count_calls_in` used for a plain running
/// count alone. Exhaustive by design: `analyze`'s own "every call site
/// safe?" test is meaningless without first knowing the true, complete
/// population of call sites to check that against.
fn collect_all_call_sites_in(expr: &CExpr, top_level_names: &HashSet<String>, out: &mut Vec<(String, CVar)>) {
    match expr {
        CExpr::LetPrim { cont, .. } => collect_all_call_sites_in(cont, top_level_names, out),
        CExpr::App { .. } => {}
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_all_call_sites_in(then_branch, top_level_names, out);
            collect_all_call_sites_in(else_branch, top_level_names, out);
        }
        CExpr::Fix { defs, body } => {
            if let [k] = defs.as_slice() {
                if let [result_var] = k.params[..] {
                    if let CExpr::App {
                        func: CVal::Label(callee),
                        args: call_args,
                    } = &**body
                    {
                        let targets_k = call_args
                            .last()
                            .map(|a| matches!(a, CVal::Label(n) if n == &k.name))
                            .unwrap_or(false);
                        if targets_k && top_level_names.contains(callee) {
                            out.push((callee.clone(), result_var));
                        }
                    }
                }
            }
            for d in defs {
                collect_all_call_sites_in(&d.body, top_level_names, out);
            }
            collect_all_call_sites_in(body, top_level_names, out);
        }
    }
}

/// Walks `expr` (a top-level function's own body, recursively through
/// every nested `Fix`/`If`) looking for a self-recursive `CFunDef` (a real
/// loop — `carried_types.is_some()`, `mlir_lower.rs::lower_loop`'s own
/// precondition) and, for each one found, analyzes it.
fn find_loops_and_mark(expr: &CExpr, top_level_names: &HashSet<String>, safe_sites: &mut HashSet<CVar>) {
    match expr {
        CExpr::LetPrim { cont, .. } => find_loops_and_mark(cont, top_level_names, safe_sites),
        CExpr::App { .. } => {}
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            find_loops_and_mark(then_branch, top_level_names, safe_sites);
            find_loops_and_mark(else_branch, top_level_names, safe_sites);
        }
        CExpr::Fix { defs, body } => {
            for d in defs {
                if d.carried_types.is_some() {
                    analyze_loop_body(d, top_level_names, safe_sites);
                }
                // Recurse into the def's own body too -- a nested loop (the
                // outer `epoch` loop containing the inner `s` loop, say),
                // or a real call's own resumption continuation, might
                // itself contain further loops.
                find_loops_and_mark(&d.body, top_level_names, safe_sites);
            }
            find_loops_and_mark(body, top_level_names, safe_sites);
        }
    }
}

/// The real analysis, for one loop's own body: which of its own direct
/// top-level calls are individually safe (non-escaping) — recorded by call-
/// site identity in `safe_sites`, unconditionally; `analyze`'s own fixed
/// point is what later decides, per callee *name*, whether *every* one of
/// its call sites (this loop's own occurrences, plus any found elsewhere in
/// the whole program) ended up safe.
///
/// **Scoped to `then_branch` alone, not `loop_def.body` as a whole — a
/// real, found-by-testing soundness bug, not a style choice.**
/// `loop_def.body` (a loop's own condition-chain-then-`If` shape, see
/// `loop_then_branch`'s own doc comment) has *three* distinct parts: the
/// condition chain (evaluated every iteration, but not analyzed here
/// either — see below), `then_branch` (the loop's own real, repeating
/// body — the *only* code that genuinely runs "inside" the loop, once per
/// iteration), and `else_branch`. `mlir_lower.rs::lower_loop`'s own doc
/// comment is explicit about `else_branch`: "runs in the *outer* scope,
/// ordinary flow" — it's whatever comes *after* the loop in source,
/// running exactly once, lowered entirely outside the `scf.while` op, not
/// per-iteration at all. Scanning the whole `loop_def.body` (the original,
/// broken version of this function) walked into `else_branch` too — since
/// a CPS-converted loop's own "exit" path structurally *contains* the rest
/// of the enclosing function as part of the same term, this doesn't just
/// miss an opportunity, it actively misclassifies real, one-shot,
/// after-the-loop calls (`dynarray_new`, say, called once before any loop
/// even starts but reached this way through an *earlier* loop's own exit
/// branch) as if they ran once per iteration — marking them region-local
/// even though their own call site is never wrapped in a `cleave_region_
/// enter`/`cleave_region_exit` pair at all (only `lower_loop`'s own
/// generated code ever emits one), so the very first allocation inside
/// such a function hits `cleave-rt::cleave_alloc_local`'s own `assert_
/// region_open` and aborts the process. Confirmed directly, not guessed:
/// `examples/convex_hull.cleave --run` (no `#[mlir_type(...)]`-tagged type
/// anywhere in it, so unrelated to any tensor-specific pipeline stage)
/// crashed exactly this way, `dynarray_new<Point>` wrongly in `region_
/// local_fns` — fixed by this restriction alone.
fn analyze_loop_body(loop_def: &CFunDef, top_level_names: &HashSet<String>, safe_sites: &mut HashSet<CVar>) {
    let Some(then_branch) = loop_then_branch(loop_def) else {
        // An unrecognized condition-chain shape -- `mlir_lower.rs::
        // lower_loop` is where a genuinely malformed loop panics loudly at
        // actual lowering time; this analysis just conservatively marks
        // nothing region-local for one it can't confidently parse this
        // way, exactly as safe as never having found the loop at all.
        return;
    };

    // The escaping set -- every `CVar` referenced in *any* tail-call back
    // to this same loop (`scf.yield`'s own operands, once lowered) --
    // these, and everything transitively derived from them, must survive
    // past this iteration.
    let mut escaping: HashSet<CVar> = HashSet::new();
    collect_escaping(then_branch, &loop_def.name, &mut escaping);

    // `children[base]` = every `CVar` bound via `PrimOp::Field` reading
    // straight out of `base` (`g.2`'s own `CVar` is a child of `g`'s) --
    // together with `calls` (every direct top-level call's own `(callee,
    // bound result CVar)` pair) found anywhere in this same loop body.
    let mut children: HashMap<CVar, Vec<CVar>> = HashMap::new();
    let mut calls: Vec<(String, CVar)> = Vec::new();
    collect_calls_and_derivations(then_branch, top_level_names, &mut children, &mut calls);

    for (_callee, result_var) in &calls {
        if !reaches_escaping(*result_var, &children, &escaping) {
            safe_sites.insert(*result_var);
        }
    }
}

/// Extracts `loop_def.body`'s own real, per-iteration body — the `then_
/// branch` of the terminal `If` at the end of its own condition chain
/// (zero or more sequential real calls, each a `Fix{[k], App(callee,
/// args)}` layer, `k`'s own body continuing the chain — the exact shape
/// `mlir_lower.rs::lower_loop`'s own identical walk parses; see that
/// function's own doc comment for the full story of why a loop's
/// condition can be more than one call). Returns `None` for a shape this
/// doesn't recognize — this analysis only ever gets more conservative by
/// bailing out, never wrong; `lower_loop` remains the sole authority that
/// panics on a genuinely malformed loop.
fn loop_then_branch(loop_def: &CFunDef) -> Option<&CExpr> {
    loop_then_and_else_branch(loop_def).map(|(then_branch, _)| then_branch)
}

/// `loop_then_branch`'s own condition-chain walk, generalized to hand back
/// `else_branch` too — code that runs once *this* loop exits, `mlir_lower.
/// rs::lower_loop`'s own doc comment: "runs in the *outer* scope, ordinary
/// flow". Used by `collect_escaping`/`collect_calls_and_derivations` below
/// to correctly treat a *nested* loop's own exit path as still belonging to
/// whichever *enclosing* loop's own analysis is currently walking through
/// it, while never re-scanning the nested loop's own repeating body (`then_
/// branch`) — that body already gets its own, separate, dedicated `analyze_
/// loop_body` call from `find_loops_and_mark`'s own top-level walk, which
/// alone has the right `escaping` set (relative to *that* loop's own
/// self-recursive tail-call) to judge it correctly.
pub(crate) fn loop_then_and_else_branch(loop_def: &CFunDef) -> Option<(&CExpr, &CExpr)> {
    let mut cursor: &CExpr = &loop_def.body;
    loop {
        match cursor {
            CExpr::If {
                then_branch,
                else_branch,
                ..
            } => return Some((then_branch, else_branch)),
            CExpr::Fix { defs, .. } => {
                let [cond_k] = &defs[..] else {
                    return None;
                };
                cursor = &cond_k.body;
            }
            _ => return None,
        }
    }
}

/// Every `CVar` referenced in the args of any `App` that tail-calls
/// `loop_name` (this loop's own recursive "continue" self-call) —
/// `mlir_lower.rs::lower_loop`'s own `then_branch`'s tail recursion becomes
/// exactly `scf.yield` on these same values.
pub(crate) fn collect_escaping(expr: &CExpr, loop_name: &str, escaping: &mut HashSet<CVar>) {
    match expr {
        CExpr::LetPrim { cont, .. } => collect_escaping(cont, loop_name, escaping),
        CExpr::App {
            func: CVal::Label(name),
            args,
        } if name == loop_name => {
            for a in args {
                if let CVal::Var(v) = a {
                    escaping.insert(*v);
                }
            }
        }
        CExpr::App { .. } => {}
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_escaping(then_branch, loop_name, escaping);
            collect_escaping(else_branch, loop_name, escaping);
        }
        CExpr::Fix { defs, body } => {
            for d in defs {
                // A *nested* loop definition (`d.carried_types.is_some()`,
                // `mlir_lower.rs::lower_loop`'s own precondition) — a real,
                // found-by-testing soundness bug, not a style choice, to
                // recurse into its own `then_branch` (its own repeating
                // body) here: this scan is relative to *this* (some
                // enclosing) loop's own `loop_name`, and the generic `App`
                // arm above never recognizes the *nested* loop's own
                // self-recursive tail-call (a *different* name) as an
                // escape at all — so anything that genuinely escapes only
                // as far as the *nested* loop's own carried state (`doc/
                // backlog.md`'s own real example: `Optimizer::step`'s own
                // result, carried into `net`/`state` by the *inner* batch
                // loop) looks, from this *outer* loop's own perspective,
                // like it never escapes anywhere — a false "safe" verdict
                // that `find_region_local_functions`'s own whole-program
                // `HashSet` (a union across every loop analyzed, never an
                // intersection) then keeps permanently, even though the
                // nested loop's own dedicated analysis already got it
                // right. Only the nested loop's own *else branch* (`loop_
                // then_and_else_branch`'s own doc comment: ordinary flow,
                // still part of *this* enclosing scope) is genuinely this
                // loop's own concern; an ordinary (non-loop) continuation's
                // own body is unaffected, and still fully recursed into.
                if d.carried_types.is_some() {
                    if let Some((_, else_branch)) = loop_then_and_else_branch(d) {
                        collect_escaping(else_branch, loop_name, escaping);
                    }
                } else {
                    collect_escaping(&d.body, loop_name, escaping);
                }
            }
            collect_escaping(body, loop_name, escaping);
        }
    }
}

/// Populates `children` (`PrimOp::Field` parent -> child edges) and `calls`
/// (every direct top-level call found, `collect_all_call_sites_in`'s own
/// identical shape).
pub(crate) fn collect_calls_and_derivations(
    expr: &CExpr,
    top_level_names: &HashSet<String>,
    children: &mut HashMap<CVar, Vec<CVar>>,
    calls: &mut Vec<(String, CVar)>,
) {
    match expr {
        CExpr::LetPrim { var, op, args, cont, .. } => {
            if matches!(op, PrimOp::Field { .. }) {
                if let [CVal::Var(base)] = args.as_slice() {
                    children.entry(*base).or_default().push(*var);
                }
            }
            collect_calls_and_derivations(cont, top_level_names, children, calls);
        }
        CExpr::App { .. } => {}
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_calls_and_derivations(then_branch, top_level_names, children, calls);
            collect_calls_and_derivations(else_branch, top_level_names, children, calls);
        }
        CExpr::Fix { defs, body } => {
            if let [k] = defs.as_slice() {
                if let [result_var] = k.params[..] {
                    if let CExpr::App {
                        func: CVal::Label(callee),
                        args: call_args,
                    } = &**body
                    {
                        let targets_k = call_args
                            .last()
                            .map(|a| matches!(a, CVal::Label(n) if n == &k.name))
                            .unwrap_or(false);
                        if targets_k && top_level_names.contains(callee) {
                            calls.push((callee.clone(), result_var));
                        }
                    }
                }
            }
            for d in defs {
                // Same nested-loop restriction as `collect_escaping`'s own
                // identical fix, and for the identical reason: a call
                // sitting inside a *nested* loop's own repeating body has
                // already been (or will be) collected by that loop's own
                // separate `analyze_loop_body` call, against *its* own,
                // correctly-scoped `escaping` set — collecting it *again*
                // here just hands the *same* call to a check that can only
                // ever get its own escaping status wrong (relative to the
                // wrong loop's own tail-call). Its own else branch (still
                // ordinary flow in this enclosing scope) is unaffected.
                if d.carried_types.is_some() {
                    if let Some((_, else_branch)) = loop_then_and_else_branch(d) {
                        collect_calls_and_derivations(else_branch, top_level_names, children, calls);
                    }
                } else {
                    collect_calls_and_derivations(&d.body, top_level_names, children, calls);
                }
            }
            collect_calls_and_derivations(body, top_level_names, children, calls);
        }
    }
}

/// Whether `start` (or anything transitively derived from it via
/// `PrimOp::Field` — `children`'s own edges) is in `escaping` — a plain
/// reachability search, no fixpoint needed (the graph is a finite,
/// acyclic set of field-projection edges over one loop body's own CPS
/// term, never larger).
pub(crate) fn reaches_escaping(start: CVar, children: &HashMap<CVar, Vec<CVar>>, escaping: &HashSet<CVar>) -> bool {
    let mut stack = vec![start];
    let mut seen = HashSet::new();
    while let Some(v) = stack.pop() {
        if !seen.insert(v) {
            continue;
        }
        if escaping.contains(&v) {
            return true;
        }
        if let Some(kids) = children.get(&v) {
            stack.extend(kids.iter().copied());
        }
    }
    false
}
