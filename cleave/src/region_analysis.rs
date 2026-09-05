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
//! a function's own allocation sites region-local is only sound if that
//! function has *exactly one* call site in the whole program, and that one
//! call's own result never reaches the enclosing loop's own carried
//! (escaping) state — checked structurally, the same "no dataflow fixpoint
//! needed, the structure already says so" argument `doc/hld.md`'s own
//! "Memory management" section makes for CPS-level lifetimes generally,
//! applied here to *which* top-level function a call targets rather than
//! to one local value's own liveness. A function called from more than one
//! place, or from a non-loop context, is conservatively left alone —
//! `cleave-rt::cleave_alloc_local`'s own `assert_region_open` would catch
//! a wrong classification loudly (a crash, not silent corruption) if this
//! analysis were ever wrong, but this module's own job is to not be wrong
//! in the first place, not to rely on that assertion as a safety net.

use crate::cps::{CExpr, CFunDef, CVal, CVar, CpsProgram, PrimOp};
use std::collections::{HashMap, HashSet};

/// The whole public surface: every top-level function name safe to lower
/// with `cleave_alloc_local` at each of its own construction sites.
pub fn find_region_local_functions(program: &CpsProgram) -> HashSet<String> {
    let top_level_names: HashSet<String> = program.funcs.iter().map(|f| f.def.name.clone()).collect();
    let call_counts = count_call_sites(program, &top_level_names);

    let mut region_local = HashSet::new();
    for f in &program.funcs {
        find_loops_and_mark(&f.def.body, &top_level_names, &call_counts, &mut region_local);
    }
    region_local
}

/// How many real, top-level-call-shaped `App`s target each top-level
/// function name, across the *whole* program — `find_loops_and_mark`'s own
/// safety precondition (a region-local candidate must have exactly one).
fn count_call_sites(program: &CpsProgram, top_level_names: &HashSet<String>) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for f in &program.funcs {
        count_calls_in(&f.def.body, top_level_names, &mut counts);
    }
    counts
}

fn count_calls_in(expr: &CExpr, top_level_names: &HashSet<String>, counts: &mut HashMap<String, usize>) {
    match expr {
        CExpr::LetPrim { cont, .. } => count_calls_in(cont, top_level_names, counts),
        CExpr::App { .. } => {}
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            count_calls_in(then_branch, top_level_names, counts);
            count_calls_in(else_branch, top_level_names, counts);
        }
        CExpr::Fix { defs, body } => {
            // `lower_real_call`'s own exact shape (`mlir_lower.rs`'s own
            // doc comment on that function): a single-def `Fix` whose own
            // body is a real call targeting that one def's own name as its
            // trailing continuation argument.
            if let [k] = defs.as_slice() {
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
                        *counts.entry(callee.clone()).or_insert(0) += 1;
                    }
                }
            }
            for d in defs {
                count_calls_in(&d.body, top_level_names, counts);
            }
            count_calls_in(body, top_level_names, counts);
        }
    }
}

/// Walks `expr` (a top-level function's own body, recursively through
/// every nested `Fix`/`If`) looking for a self-recursive `CFunDef` (a real
/// loop — `carried_types.is_some()`, `mlir_lower.rs::lower_loop`'s own
/// precondition) and, for each one found, analyzes it.
fn find_loops_and_mark(
    expr: &CExpr,
    top_level_names: &HashSet<String>,
    call_counts: &HashMap<String, usize>,
    region_local: &mut HashSet<String>,
) {
    match expr {
        CExpr::LetPrim { cont, .. } => find_loops_and_mark(cont, top_level_names, call_counts, region_local),
        CExpr::App { .. } => {}
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            find_loops_and_mark(then_branch, top_level_names, call_counts, region_local);
            find_loops_and_mark(else_branch, top_level_names, call_counts, region_local);
        }
        CExpr::Fix { defs, body } => {
            for d in defs {
                if d.carried_types.is_some() {
                    analyze_loop_body(d, top_level_names, call_counts, region_local);
                }
                // Recurse into the def's own body too -- a nested loop (the
                // outer `epoch` loop containing the inner `s` loop, say),
                // or a real call's own resumption continuation, might
                // itself contain further loops.
                find_loops_and_mark(&d.body, top_level_names, call_counts, region_local);
            }
            find_loops_and_mark(body, top_level_names, call_counts, region_local);
        }
    }
}

/// The real analysis, for one loop's own body: which of its own direct
/// top-level calls are safe to mark region-local.
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
fn analyze_loop_body(
    loop_def: &CFunDef,
    top_level_names: &HashSet<String>,
    call_counts: &HashMap<String, usize>,
    region_local: &mut HashSet<String>,
) {
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

    for (callee, result_var) in &calls {
        // Exactly one call site in the *whole* program -- this function's
        // own module doc comment has the real reasoning for why that's
        // load-bearing, not just a nicety.
        if call_counts.get(callee).copied().unwrap_or(0) != 1 {
            continue;
        }
        if !reaches_escaping(*result_var, &children, &escaping) {
            region_local.insert(callee.clone());
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
fn loop_then_and_else_branch(loop_def: &CFunDef) -> Option<(&CExpr, &CExpr)> {
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
fn collect_escaping(expr: &CExpr, loop_name: &str, escaping: &mut HashSet<CVar>) {
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
/// (every direct top-level call found, `lower_real_call`'s own exact shape
/// — see `count_calls_in`'s own doc comment for that same shape).
fn collect_calls_and_derivations(
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
fn reaches_escaping(start: CVar, children: &HashMap<CVar, Vec<CVar>>, escaping: &HashSet<CVar>) -> bool {
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
