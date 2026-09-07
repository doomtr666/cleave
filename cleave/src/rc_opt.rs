//! Tier 1 of `doc/backlog.md`'s own struct-allocation-strategy entry: a
//! retain/release *pair-cancellation* pass, run strictly after
//! `refcount::insert_refcounting` — never replaces it, never changes what
//! it decides is *correct* (this pass can only remove work already proven
//! redundant, exactly the shape Swift/Objective-C's own ARC optimizer
//! uses: the naive baseline insertion stays the one source of truth for
//! correctness, this pass only elides calls it can *prove* are no-ops).
//! Worst case — nothing provable anywhere — this pass is the identity
//! function; it can never make a correct program incorrect.
//!
//! ## The criterion
//!
//! `refcount::insert_refcounting`'s own `retain_targets` mechanism retains
//! an *existing* value unconditionally whenever it's embedded into a fresh
//! `PrimOp::Struct`/`Array`/`FieldStore`/`Store` (or read back out via
//! `PrimOp::Field`, retain-on-read) — regardless of whether the original
//! binding is ever used again. If it *isn't* — nothing after this point
//! reads the retained `CVar` again, on *every* reachable path — then the
//! retain (count `1 -> 2`) is exactly cancelled by whatever `Release`
//! `refcount.rs`'s own `releases_for_app` already inserted for that same,
//! now-dead binding (count `2 -> 1`): net effect, the count stays at `1`,
//! identical to never having retained at all (a plain transfer into the
//! container, not a duplicate).
//!
//! **Why this is safe by construction for a value that escapes into
//! unbounded storage (`DynArray::push`), with no special-cased detection
//! needed at all — the exact concern raised while designing this pass, not
//! an afterthought**: `push<T>(mut v: DynArray<T>, x: T)`'s own `x` is a
//! *borrowed* function parameter — `refcount.rs`'s own module doc comment
//! is explicit that a parameter is never seeded into `owned`, hence never
//! given a matching `Release` inside the callee's own body at all. This
//! pass only ever eliminates a retain when it can find and remove an
//! *actual*, textually-present `Release` for the exact same `CVar`, on
//! every reachable path — a borrowed value's retain (protecting a brand
//! new, independent reference the callee's own storage now holds, exactly
//! `push`'s own real need) simply never has one to find, so this pass
//! leaves it alone automatically, without knowing anything about
//! `DynArray` specifically. The same reasoning covers any other case where
//! `x`'s own eventual fate isn't resolved by a textually-nearby `Release`
//! at all — conservative by construction, not by an enumerated exception
//! list.
//!
//! ## Scope, conservative by design
//!
//! A forward scan from a `Retain(x)` stops at any `Fix` boundary it can't
//! specifically recognize as an `if`'s own join (`Fix { defs: [join],
//! body: If { .. } }` — `cps.rs`'s own `ExprKind::If` conversion shape) —
//! recursing separately into `then_branch`/`else_branch` from there (each
//! decides `x`'s own fate independently — no merge/dynamic flag needed,
//! since a CPS branch is its own separate straight-line code, not a shared
//! basic block with phi nodes), but never descending into the join
//! continuation's own body itself: by the time either branch's own tail
//! call reaches it, `x`'s fate was already fully decided one way or the
//! other. Any other `Fix` (a loop, or a real call's own resumption) is a
//! hard stop for this first version — elimination for a retain whose scan
//! would need to cross it is simply not attempted; conservative, never
//! unsound, and cheap to extend later if it turns out to matter (measured,
//! not assumed — see `doc/backlog.md`'s own entry for the real kernel
//! measurement this pass's own impact was checked against).

use crate::cps::{CExpr, CVal, CVar, CpsProgram, PrimOp};

/// Runs once, after `refcount::insert_refcounting`, over every top-level
/// function's own body — see the module's own doc comment for the full
/// design and its own safety argument.
pub(crate) fn eliminate_redundant_retain_release(program: CpsProgram) -> CpsProgram {
    let funcs = program
        .funcs
        .into_iter()
        .map(|mut top| {
            top.def.body = optimize_body(top.def.body);
            top
        })
        .collect();
    CpsProgram { funcs }
}

/// The main recursive walk — applies retain/release pair elimination
/// throughout `expr`, and recurses into every `Fix`'s own `defs[*].body`
/// as its own independent scope (`refcount.rs`'s own `local_free_vars`/
/// `local_claim_vars` seed each of these the same way, for the identical
/// reason). `Fix`'s own trailing `body` field is *not* a fresh scope
/// (`refcount.rs::rewrite_body`'s own `CExpr::Fix` arm threads the *same*
/// `owned` set into it, whether it's an `if`'s own two branches or a
/// loop/call's own initial jump) — walked as part of the very same pass,
/// not recursed into separately.
fn optimize_body(expr: CExpr) -> CExpr {
    match expr {
        CExpr::LetPrim {
            var,
            ty,
            op,
            args,
            cont,
        } => {
            if let PrimOp::Retain(_) = &op {
                if let [CVal::Var(x)] = args.as_slice() {
                    let x = *x;
                    let result = if starts_protected_embedding(&cont) {
                        try_eliminate_for_retain(&cont, x)
                    } else {
                        // The retain-on-read shape (`refcount.rs::rewrite_
                        // body`'s own `field_read_retain`) — `x` is simply
                        // an ordinarily-owned value from here on, no
                        // protected first occurrence to skip past.
                        try_eliminate_for(&cont, x)
                    };
                    if let Some(new_cont) = result {
                        // The retain itself is dropped entirely (not
                        // re-emitted) — keep optimizing what's left, more
                        // eliminable retains may follow.
                        return optimize_body(new_cont);
                    }
                }
            }
            CExpr::LetPrim {
                var,
                ty,
                op,
                args,
                cont: Box::new(optimize_body(*cont)),
            }
        }
        CExpr::App { .. } => expr,
        CExpr::If {
            cond,
            then_branch,
            else_branch,
        } => CExpr::If {
            cond,
            then_branch: Box::new(optimize_body(*then_branch)),
            else_branch: Box::new(optimize_body(*else_branch)),
        },
        CExpr::Fix { defs, body } => {
            let defs = defs
                .into_iter()
                .map(|mut d| {
                    d.body = optimize_body(d.body);
                    d
                })
                .collect();
            CExpr::Fix {
                defs,
                body: Box::new(optimize_body(*body)),
            }
        }
    }
}

/// Whether `expr` is exactly the shape `refcount.rs::rewrite_body` always
/// builds right after a `retain_targets`-originated `Retain` — another
/// stacked `Retain` (several refcounted fields in the same construction
/// each get their own), or the real embedding op itself
/// (`Struct`/`Array`/`FieldStore`/`Store`). If neither, the retain being
/// examined must be the *other* shape (retain-on-read), handled directly
/// by `try_eliminate_for` instead.
fn starts_protected_embedding(expr: &CExpr) -> bool {
    matches!(
        expr,
        CExpr::LetPrim { op, .. }
            if matches!(
                op,
                PrimOp::Retain(_)
                    | PrimOp::Struct(..)
                    | PrimOp::Array
                    | PrimOp::FieldStore { .. }
                    | PrimOp::Store { .. }
            )
    )
}

/// Walks past the protected embedding prefix (one or more stacked
/// `Retain`s, then the real embedding op) without treating `x`'s own
/// expected occurrence there as a disqualifying reuse, then switches to
/// `try_eliminate_for` for everything genuinely afterward. `expr` is
/// assumed (by `starts_protected_embedding`, checked at the only call
/// site) to actually have this shape — falls back to giving up, not to
/// guessing, if it somehow doesn't.
fn try_eliminate_for_retain(expr: &CExpr, x: CVar) -> Option<CExpr> {
    let CExpr::LetPrim {
        var,
        ty,
        op,
        args,
        cont,
    } = expr
    else {
        return None;
    };
    match op {
        PrimOp::Retain(_) => {
            let new_cont = try_eliminate_for_retain(cont, x)?;
            Some(CExpr::LetPrim {
                var: *var,
                ty: ty.clone(),
                op: op.clone(),
                args: args.clone(),
                cont: Box::new(new_cont),
            })
        }
        PrimOp::Struct(..) | PrimOp::Array | PrimOp::FieldStore { .. } | PrimOp::Store { .. } => {
            let new_cont = try_eliminate_for(cont, x)?;
            Some(CExpr::LetPrim {
                var: *var,
                ty: ty.clone(),
                op: op.clone(),
                args: args.clone(),
                cont: Box::new(new_cont),
            })
        }
        _ => None,
    }
}

/// The real forward scan: does `x` get read again anywhere reachable from
/// `expr`, on *any* path (disqualifying — the retain is genuinely needed,
/// give up), or does *every* reachable path resolve to an actual, matching
/// `Release(x)` (eliminable — remove it)? See the module's own doc comment
/// for the full reasoning, including why a path that resolves to neither
/// (a loop/call `Fix`, or a terminal `App` with `x` simply absent, meaning
/// its own fate isn't tracked by this local mechanism at all) must also
/// give up rather than guess.
fn try_eliminate_for(expr: &CExpr, x: CVar) -> Option<CExpr> {
    match expr {
        CExpr::LetPrim {
            var,
            ty,
            op,
            args,
            cont,
        } => {
            if let PrimOp::Release(_) = op {
                if let [CVal::Var(rx)] = args.as_slice() {
                    if *rx == x {
                        // The match — `x` is dead from here on by
                        // definition, nothing past its own release needs
                        // checking. Drop this `Release` LetPrim entirely.
                        return Some((**cont).clone());
                    }
                }
            }
            // A `Field` read *through* `x` (reading some other value stored
            // inside it) is transparent — it doesn't create any new,
            // independent reference to `x`, just one more dereference of
            // the exact same pointer that's about to die anyway, no less
            // valid than the release itself needing `x` to still be a live
            // pointer right up to (but not including) that point. Found by
            // direct testing: without this exemption, `o.x.a` (a field
            // read off a field read) wrongly disqualified `o.x`'s own
            // retain-on-read from ever being eliminated, since reading
            // `.a` off it looks, structurally, exactly like "using `x`
            // again" otherwise. Only `x` appearing anywhere else (handed
            // to something that could store or hand off a *new* reference
            // to it) is a genuine, disqualifying second use.
            let transparent_field_read = matches!(op, PrimOp::Field { .. })
                && matches!(args.as_slice(), [CVal::Var(base)] if *base == x);
            if !transparent_field_read && references(args, x) {
                return None; // a genuine second use -- the retain stays
            }
            let new_cont = try_eliminate_for(cont, x)?;
            Some(CExpr::LetPrim {
                var: *var,
                ty: ty.clone(),
                op: op.clone(),
                args: args.clone(),
                cont: Box::new(new_cont),
            })
        }
        // A terminal jump — whether or not `x` appears in it, there is no
        // `cont` past it for this scan to keep looking in. `x` present
        // here is a real, disqualifying use (still needed at this exact
        // jump); `x` absent means the scan ran off the end without ever
        // finding a `Release` for it — equally unresolved, equally "give
        // up" (never "assume it's fine").
        CExpr::App { func, args } => {
            let _ = (func, args, x);
            None
        }
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            let new_then = try_eliminate_for(then_branch, x)?;
            let new_else = try_eliminate_for(else_branch, x)?;
            let CExpr::If { cond, .. } = expr else {
                unreachable!()
            };
            Some(CExpr::If {
                cond: cond.clone(),
                then_branch: Box::new(new_then),
                else_branch: Box::new(new_else),
            })
        }
        CExpr::Fix { defs, body } => {
            // Recognize exactly the `if`'s own join shape (`cps.rs::
            // ExprKind::If`'s own conversion) -- pass through into both
            // branches transparently; any other `Fix` (a loop, or a real
            // call's own resumption) is a hard stop for this first
            // version, per the module's own doc comment.
            if let ([_join], CExpr::If { .. }) = (defs.as_slice(), body.as_ref()) {
                let new_body = try_eliminate_for(body, x)?;
                Some(CExpr::Fix {
                    defs: defs.clone(),
                    body: Box::new(new_body),
                })
            } else {
                None
            }
        }
    }
}

fn references(args: &[CVal], x: CVar) -> bool {
    args.iter().any(|v| matches!(v, CVal::Var(v) if *v == x))
}
