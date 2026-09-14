
//! Escape analysis for CPS-level struct allocations — determines, for every
//! `PrimOp::Struct`-bound `CVar` in a function body, whether it is a *slot*
//! (crosses at least one continuation jump and therefore needs a refcount
//! header + CPS retain/release) or a *temporary* (dies before any jump and
//! can be allocated from the arena without a header).
//!
//! ## The model
//!
//! A value bound by `PrimOp::Struct` is a **slot** if and only if it appears
//! in the `live_set` of at least one `App` within the scope that follows its
//! own binding. In CPS, `live_set(App{func, args})` is:
//!   - every `CVal::Var` literally passed as `func` or an argument, plus
//!   - for every `CVal::Label(name)` appearing anywhere in `func`/`args` that
//!     names a known `Fix`-local continuation, that continuation's own free
//!     variables.
//!
//! If the variable never appears in any `live_set` within its own scope, it
//! dies strictly before the first jump it could reach — a pure temporary,
//! allocatable from the arena, no header needed.
//!
//! ## Relationship to `region_analysis.rs`
//!
//! `region_analysis` works at function granularity: which *top-level
//! functions* are safe to lower with `cleave_alloc_local`. This module works
//! at variable granularity: which *specific bindings* within any function
//! need a refcount header. The two are complementary.
//!
//! ## Relationship to `refcount.rs`
//!
//! `refcount.rs` already computes `live_set` at every `App` to decide which
//! owned values to release. This module asks the symmetric question before
//! refcounting runs: which owned values ever survive a jump at all. The
//! `live_set` definition here mirrors `refcount.rs::live_set` exactly.

use crate::cps::{CExpr, CFunDef, CVal, CVar, CpsProgram, PrimOp};
use std::collections::{HashMap, HashSet};

/// For every top-level function in `program`, returns the set of `CVar`s
/// bound by `PrimOp::Struct` within its body (at any nesting depth) that
/// are *escaping* — appear in the `live_set` of at least one `App` reachable
/// from their binding. These are slot-class values that require
/// `cleave_alloc_rc` + a refcount header. Non-escaping struct vars are arena
/// temporaries (`cleave_alloc_local`, no header).
///
/// `CVar`s are globally unique so the result is a single flat `HashSet`
/// over the whole program with no collision risk across functions.
pub fn escaping_struct_vars(program: &CpsProgram) -> HashSet<CVar> {
    let mut escaping = HashSet::new();
    for f in &program.funcs {
        let local_free_vars = collect_local_free_vars_for(&f.def);
        find_escaping_in(&f.def.body, &local_free_vars, &mut escaping);
    }
    escaping
}

/// Free-variable sets for every `Fix`-local def in `top`. Mirrors
/// `refcount::collect_local_free_vars` — any change to the live-set rule
/// must be applied to both.
fn collect_local_free_vars_for(top: &CFunDef) -> HashMap<String, HashSet<CVar>> {
    let mut out: HashMap<String, HashSet<CVar>> = HashMap::new();
    let mut func_labels: HashMap<String, HashSet<String>> = HashMap::new();
    walk_local_free_vars_in(&top.body, &mut out, &mut func_labels);
    // Fixpoint: propagate transitively through bare tail-calls.
    loop {
        let mut changed = false;
        let names: Vec<String> = out.keys().cloned().collect();
        for name in names {
            let deps = func_labels.get(&name).cloned().unwrap_or_default();
            let mut additions: Vec<CVar> = Vec::new();
            for dep in &deps {
                if let Some(dep_vars) = out.get(dep) {
                    additions.extend(dep_vars.iter().copied());
                }
            }
            let entry = out.get_mut(&name).unwrap();
            for v in additions {
                changed |= entry.insert(v);
            }
        }
        if !changed {
            break;
        }
    }
    out
}

fn walk_local_free_vars_in(
    expr: &CExpr,
    out: &mut HashMap<String, HashSet<CVar>>,
    func_labels: &mut HashMap<String, HashSet<String>>,
) {
    match expr {
        CExpr::LetPrim { cont, .. } => walk_local_free_vars_in(cont, out, func_labels),
        CExpr::App { .. } => {}
        CExpr::If { then_branch, else_branch, .. } => {
            walk_local_free_vars_in(then_branch, out, func_labels);
            walk_local_free_vars_in(else_branch, out, func_labels);
        }
        CExpr::Fix { defs, body } => {
            for d in defs {
                let mut bound: HashSet<CVar> = d.params.iter().copied().collect();
                let mut referenced: HashSet<CVar> = HashSet::new();
                let mut inner_func_labels: HashSet<String> = HashSet::new();
                collect_bound_and_referenced_in(&d.body, &mut bound, &mut referenced, &mut inner_func_labels);
                let free: HashSet<CVar> = referenced.difference(&bound).copied().collect();
                out.insert(d.name.clone(), free);
                func_labels.insert(d.name.clone(), inner_func_labels);
                walk_local_free_vars_in(&d.body, out, func_labels);
            }
            walk_local_free_vars_in(body, out, func_labels);
        }
    }
}

fn collect_bound_and_referenced_in(
    expr: &CExpr,
    bound: &mut HashSet<CVar>,
    referenced: &mut HashSet<CVar>,
    func_labels: &mut HashSet<String>,
) {
    match expr {
        CExpr::LetPrim { var, args, cont, .. } => {
            bound.insert(*var);
            for a in args { if let CVal::Var(v) = a { referenced.insert(*v); } }
            collect_bound_and_referenced_in(cont, bound, referenced, func_labels);
        }
        CExpr::App { func, args } => {
            match func {
                CVal::Var(v) => { referenced.insert(*v); }
                CVal::Label(name) => { func_labels.insert(name.clone()); }
                _ => {}
            }
            for a in args { if let CVal::Var(v) = a { referenced.insert(*v); } }
        }
        CExpr::If { cond, then_branch, else_branch } => {
            if let CVal::Var(v) = cond { referenced.insert(*v); }
            collect_bound_and_referenced_in(then_branch, bound, referenced, func_labels);
            collect_bound_and_referenced_in(else_branch, bound, referenced, func_labels);
        }
        CExpr::Fix { defs, body } => {
            for d in defs {
                bound.extend(d.params.iter().copied());
                collect_bound_and_referenced_in(&d.body, bound, referenced, func_labels);
            }
            collect_bound_and_referenced_in(body, bound, referenced, func_labels);
        }
    }
}

fn live_set_at(
    func: &CVal,
    args: &[CVal],
    local_free_vars: &HashMap<String, HashSet<CVar>>,
) -> HashSet<CVar> {
    let mut live: HashSet<CVar> = HashSet::new();
    for v in std::iter::once(func).chain(args.iter()) {
        match v {
            CVal::Var(cv) => { live.insert(*cv); }
            CVal::Label(name) => {
                if let Some(fv) = local_free_vars.get(name) {
                    live.extend(fv.iter().copied());
                }
            }
            _ => {}
        }
    }
    live
}

fn find_escaping_in(
    expr: &CExpr,
    local_free_vars: &HashMap<String, HashSet<CVar>>,
    escaping: &mut HashSet<CVar>,
) {
    match expr {
        CExpr::LetPrim { var, op, cont, .. } => {
            if matches!(op, PrimOp::Struct(..)) {
                if is_live_at_any_app(*var, cont, local_free_vars) {
                    escaping.insert(*var);
                }
            }
            find_escaping_in(cont, local_free_vars, escaping);
        }
        CExpr::App { .. } => {}
        CExpr::If { then_branch, else_branch, .. } => {
            find_escaping_in(then_branch, local_free_vars, escaping);
            find_escaping_in(else_branch, local_free_vars, escaping);
        }
        CExpr::Fix { defs, body } => {
            for d in defs { find_escaping_in(&d.body, local_free_vars, escaping); }
            find_escaping_in(body, local_free_vars, escaping);
        }
    }
}

/// Returns `true` if `var` is live at any `App` in `expr`. Terminates early.
fn is_live_at_any_app(
    var: CVar,
    expr: &CExpr,
    local_free_vars: &HashMap<String, HashSet<CVar>>,
) -> bool {
    match expr {
        CExpr::App { func, args } => live_set_at(func, args, local_free_vars).contains(&var),
        CExpr::LetPrim { cont, .. } => is_live_at_any_app(var, cont, local_free_vars),
        CExpr::If { then_branch, else_branch, .. } => {
            is_live_at_any_app(var, then_branch, local_free_vars)
                || is_live_at_any_app(var, else_branch, local_free_vars)
        }
        CExpr::Fix { defs, body } => {
            defs.iter().any(|d| is_live_at_any_app(var, &d.body, local_free_vars))
                || is_live_at_any_app(var, body, local_free_vars)
        }
    }
}
