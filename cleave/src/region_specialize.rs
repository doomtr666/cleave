//! `doc/plan-region-arena.md`'s own "Step 2", second half — see `region_
//! analysis.rs`'s own module doc comment ("What this generalization does
//! *not* by itself fix") for the full context. Lifts the one case `region_
//! analysis::analyze`'s own relaxed "every call site is safe" rule still,
//! correctly, cannot admit on its own: a callee genuinely shared between a
//! safe context and an unsafe one (`doc/backlog.md`'s own real example: a
//! shared algebra function called from both the training loop, where its
//! result never escapes one iteration, and `evaluate()`, where it doesn't
//! run inside a loop at all, or does but escapes there).
//!
//! **"Duplicate, don't contextualize"** (`doc/plan-region-arena.md`'s own
//! words): for each such genuinely-mixed callee, clone its `CFunDef` under
//! a fresh `{name}$region` name and retarget *only* the call sites `region_
//! analysis::analyze` already proves safe to the copy, leaving every other
//! call site pointing at the untouched original. `region_analysis.rs`'s own
//! per-function analysis and `mlir_lower.rs`'s own per-function lowering
//! flag (`currently_region_local`, set once per top-level function) then
//! both stay exactly as they are — only the *population* of top-level
//! functions changes, and `region_analysis::analyze`, run again afterward
//! (`mlir_lower.rs::lower_program`'s own existing call, unmodified), simply
//! discovers that `{name}$region`'s own call sites are, by construction,
//! *all* safe.
//!
//! **Placement — must run before `region_analysis::find_region_local_
//! functions` is ever consulted**, i.e. as a CPS→CPS pass in the CPS-
//! building stage, alongside `refcount::insert_refcounting` (`pipeline.rs::
//! build_optimized_cps`'s own doc comment has the full placement argument:
//! `mlir_lower.rs::lower_program` receives an already-final `&CpsProgram`
//! and calls `find_region_local_functions` on it internally, so any pass
//! that changes *which* top-level functions exist has to run strictly
//! before that point).
//!
//! **`$region` as the naming separator, not a hack**: cleave's own lexer
//! grammar has no `$` in a valid identifier, so `{name}$region` can never
//! collide with real user source, and this project's own CPS-level local
//! continuation names already use the identical convention (`FreshVars::
//! label`, e.g. `"k$0"`) — see `mlir_lower.rs`'s own resolution of `CVal::
//! Label` for why a *local* Fix-continuation's own name never needs to be
//! globally unique in the first place (lexically scoped to the one top-
//! level function that defines it), unlike a `CVar`, which *is* meaningful
//! program-wide (`op_lines: HashMap<CVar, SrcLoc>`) and is why this module
//! renumbers a clone's own `CVar`s (`clone_and_renumber` below) rather than
//! copying them verbatim.

use crate::cps::{CExpr, CTopLevelFn, CVal, CVar, CpsProgram, FreshVars, SrcLoc};
use crate::region_analysis::analyze;
use std::collections::{HashMap, HashSet};

/// The whole public surface. A no-op (returns `program` untouched, no new
/// functions, no renaming) when nothing in the whole program is genuinely
/// mixed — the common case for a program with no shared-and-split-needed
/// helper at all, and exactly what every existing test other than this
/// module's own exercises implicitly by never triggering it.
pub fn specialize_region_local_functions(mut program: CpsProgram) -> CpsProgram {
    let analysis = analyze(&program);

    // A callee needs a split iff *some*, but not *all*, of its own call
    // sites are already proven safe -- one proven-safe occurrence with
    // nothing else to split off needs no new name at all (`region_
    // analysis::analyze`'s own relaxed rule already marks the untouched
    // original region-local directly); zero proven-safe occurrences means
    // there is nothing here for this pass to usefully retarget.
    let needs_split: HashSet<String> = analysis
        .sites_by_callee
        .iter()
        .filter(|(name, sites)| {
            !analysis.region_local.contains(name.as_str())
                && sites.iter().any(|v| analysis.safe_sites.contains(v))
        })
        .map(|(name, _)| name.clone())
        .collect();

    if needs_split.is_empty() {
        return program;
    }

    // Retarget every already-proven-safe occurrence of a to-be-split callee
    // to its own `$region` copy, in place -- every other occurrence (an
    // unsafe one, or one of an untouched callee) is left pointing at the
    // original name unchanged. Done *before* the clones themselves are
    // built: the clone is a snapshot of what the *original* looks like once
    // its own safe call sites have already been carved off elsewhere in the
    // program -- the clone's own body is an exact copy of that (same-named)
    // original body, not of some half-retargeted intermediate state, so the
    // retargeting order relative to cloning doesn't actually matter here;
    // done first purely so the loop below borrows `program.funcs` only
    // once, mutably, before the (separate) push loop borrows it again.
    for f in &mut program.funcs {
        retarget_safe_call_sites(&mut f.def.body, &needs_split, &analysis.safe_sites);
    }

    // Build each `$region` copy: a full clone of the *original* definition
    // (as it existed before retargeting above touched anything -- the
    // clones are taken from `program.funcs` fresh, after retargeting, but
    // retargeting only ever rewrites a *callee label* at a call site, never
    // a `CFunDef`'s own name/params/body shape, so "the original" is stable
    // to clone from at this point), renamed, and renumbered.
    let mut fresh = FreshVars::starting_at(crate::egraph::max_cvar_in_program(&program) + 1);
    let clones: Vec<CTopLevelFn> = program
        .funcs
        .iter()
        .filter(|f| needs_split.contains(&f.def.name))
        .map(|f| CTopLevelFn {
            def: f.def.clone(),
            param_types: f.param_types.clone(),
            result: f.result.clone(),
            k_ret: f.k_ret,
            origin: f.origin.clone(),
            no_inline: f.no_inline,
            // A `$region` copy is an internal implementation detail of this
            // one split, never independently reachable by name from
            // outside cleave -- never exported, regardless of whether the
            // original was.
            is_export: false,
            export_symbol: None,
            loc: f.loc,
        })
        .collect();

    for top in clones {
        let renamed = clone_and_renumber(top, &mut fresh, &mut program.op_lines);
        program.funcs.push(renamed);
    }

    program
}

/// Rewrites every `CVal::Label(callee)` at a call site whose own bound
/// `result_var` is in `safe_sites`, for any `callee` in `needs_split`, to
/// `"{callee}$region"` -- leaves every other call site (a call to a callee
/// not being split, or an unsafe occurrence of one that is) untouched.
/// Mirrors `region_analysis::collect_all_call_sites_in`'s own exact call-
/// shape recognition (unconditional recursion, loops or not) since it must
/// find *every* call site that walk does, not a subset -- a call this
/// function failed to visit could be a safe occurrence left silently
/// pointing at the wrong (original, now `$region`-sibling) name.
fn retarget_safe_call_sites(expr: &mut CExpr, needs_split: &HashSet<String>, safe_sites: &HashSet<CVar>) {
    match expr {
        CExpr::LetPrim { cont, .. } => retarget_safe_call_sites(cont, needs_split, safe_sites),
        CExpr::App { .. } => {}
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            retarget_safe_call_sites(then_branch, needs_split, safe_sites);
            retarget_safe_call_sites(else_branch, needs_split, safe_sites);
        }
        CExpr::Fix { defs, body } => {
            if let [k] = defs.as_slice() {
                if let [result_var] = k.params[..] {
                    if let CExpr::App {
                        func: CVal::Label(callee),
                        args,
                    } = &mut **body
                    {
                        let targets_k = args
                            .last()
                            .map(|a| matches!(a, CVal::Label(n) if n == &k.name))
                            .unwrap_or(false);
                        if targets_k && needs_split.contains(callee.as_str()) && safe_sites.contains(&result_var) {
                            callee.push_str("$region");
                        }
                    }
                }
            }
            for d in defs {
                retarget_safe_call_sites(&mut d.body, needs_split, safe_sites);
            }
            retarget_safe_call_sites(body, needs_split, safe_sites);
        }
    }
}

/// Clones `top`'s own `def.name` to `{original}$region` and gives its own
/// body/params/`k_ret` a fresh, non-colliding `CVar` numbering -- `CVar`s
/// are meaningful across the *whole* program (`op_lines: HashMap<CVar,
/// SrcLoc>`, `refcount::insert_refcounting`'s own identical `FreshVars::
/// starting_at(max_cvar_in_program(...) + 1)` idiom), so a verbatim clone
/// would silently alias the original's own numbering instead of genuinely
/// duplicating it -- two semantically distinct bindings, in the original
/// and its copy, sharing one `CVar` id would corrupt `op_lines` (whichever
/// entry was inserted second wins, misattributing a debug-info position)
/// and leave any future whole-program, `CVar`-keyed pass with an unsound
/// assumption to trip over. Local Fix-continuation *names* (`CVal::Label`,
/// a `String`, e.g. `"k$0"`) are deliberately left untouched -- see this
/// module's own top doc comment for why they never need renaming.
fn clone_and_renumber(mut top: CTopLevelFn, fresh: &mut FreshVars, op_lines: &mut HashMap<CVar, SrcLoc>) -> CTopLevelFn {
    top.def.name.push_str("$region");

    let mut remap: HashMap<CVar, CVar> = HashMap::new();
    for p in &mut top.def.params {
        let new = fresh.var();
        remap.insert(*p, new);
        *p = new;
    }
    remap_expr(&mut top.def.body, fresh, &mut remap);
    top.k_ret = *remap
        .get(&top.k_ret)
        .expect("k_ret must be one of def.params, and every param was just remapped above");

    for (&old, &new) in &remap {
        if let Some(loc) = op_lines.get(&old).copied() {
            op_lines.insert(new, loc);
        }
    }

    top
}

/// Recursively remaps every `CVar` bound (`LetPrim.var`, a nested `Fix`
/// def's own `params`) or referenced (`CVal::Var`) in `expr`, minting a
/// fresh replacement, via `fresh`, the *first* time a given old `CVar` is
/// seen -- safe in one top-down pass because CPS is define-before-use: a
/// `CVal::Var(v)` reference is only ever well-formed after `v`'s own
/// binding site has already been visited earlier in the *same* traversal
/// (an outer function's own `params`, seeded into `map` by `clone_and_
/// renumber` before this is ever called; a `LetPrim`'s own `var`, remapped
/// after its own `args` — which reference *earlier* bindings — are
/// remapped; a `Fix` def's own `params`, remapped before *any* sibling
/// def's own body is visited, so one def's own body can forward-reference
/// a sibling's params — a shape this project's own CPS conversion doesn't
/// currently produce, but a cheap, safe generality to keep here regardless).
fn remap_expr(expr: &mut CExpr, fresh: &mut FreshVars, map: &mut HashMap<CVar, CVar>) {
    match expr {
        CExpr::LetPrim { var, args, cont, .. } => {
            for a in args.iter_mut() {
                remap_val(a, map);
            }
            let new_var = fresh.var();
            map.insert(*var, new_var);
            *var = new_var;
            remap_expr(cont, fresh, map);
        }
        CExpr::App { func, args } => {
            remap_val(func, map);
            for a in args.iter_mut() {
                remap_val(a, map);
            }
        }
        CExpr::If {
            cond,
            then_branch,
            else_branch,
        } => {
            remap_val(cond, map);
            remap_expr(then_branch, fresh, map);
            remap_expr(else_branch, fresh, map);
        }
        CExpr::Fix { defs, body } => {
            for d in defs.iter_mut() {
                for p in d.params.iter_mut() {
                    let new = fresh.var();
                    map.insert(*p, new);
                    *p = new;
                }
            }
            for d in defs.iter_mut() {
                remap_expr(&mut d.body, fresh, map);
            }
            remap_expr(body, fresh, map);
        }
    }
}

/// Rewrites `v` in place via `map` if it's a `CVar` reference -- every other
/// `CVal` variant (a literal, a `Label`, `Unit`, ...) carries no `CVar` of
/// its own and is left untouched. `CVal::Closure` (own doc comment: "fully
/// gone by the time a `CpsProgram` is built") never reaches this function.
fn remap_val(v: &mut CVal, map: &HashMap<CVar, CVar>) {
    if let CVal::Var(x) = v {
        *x = map
            .get(x)
            .copied()
            .unwrap_or_else(|| panic!("remap_val: CVar {x} referenced before its own binding site was remapped"));
    }
}
