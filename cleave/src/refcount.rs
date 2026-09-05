//! Phase 0's general release/retain insertion — `doc/hld.md`'s own "Memory
//! management" section, the struct/descriptor half (the tensor-payload half
//! was tried via MLIR's own `--buffer-deallocation-pipeline` and reverted,
//! see `doc/backlog.md`; this is a separate, hand-rolled mechanism that
//! doesn't depend on MLIR's own ownership inference at all).
//!
//! ## The rule: ownership, not last-use
//!
//! A function is responsible for releasing exactly the struct-typed values
//! *it itself* freshly constructed (`PrimOp::Struct`) within its own body,
//! minus whichever one it returns (ownership transfers to the caller on
//! return — never released before the `return` itself). A function's own
//! *parameters* are never released by the callee — borrowed, not owned
//! (cleave lets a caller keep using its own binding after passing it to a
//! function, so any other convention would be unsound). Storing an
//! *existing* struct-typed value into another struct's own field
//! (`PrimOp::FieldStore`) or an array slot (`PrimOp::Store`) creates a
//! second, independent reference that will eventually be released on its
//! own — `Retain` before the store keeps the refcount honest.
//!
//! This is deliberately **not** a last-use/liveness analysis — it releases
//! at the *latest* possible point (a function's own return, or wherever a
//! value stops being passed forward to whatever it still needs to reach),
//! not the earliest. That's a real, accepted performance cost (holding
//! memory slightly longer than optimal) in exchange for soundness that
//! needs no dataflow fixpoint at all — matching `doc/hld.md`'s own explicit
//! design philosophy ("naive refcounting alone is already sound without
//! any static analysis... the rest is just optimization").
//!
//! ## Why this is sound without a liveness fixpoint
//!
//! `mlir_lower.rs::lower_cexpr` already establishes the load-bearing
//! invariant this leans on directly: a `Fix` always introduces **exactly
//! one** local continuation (`let [def] = &defs[..] else { panic!(...) }`),
//! and `Fix.body` is always one of exactly three recognized shapes — an
//! `If` (the continuation is a join, reached from both branches), a self-
//! call to the same def (a loop's own entry, reached from itself plus its
//! own back-edge), or a call to a genuinely different unit with the def as
//! the trailing continuation argument (a real call's own resumption,
//! reached from exactly one place). Every continuation's own set of
//! callers is therefore small and structurally fixed by *this* Fix node
//! alone — never an arbitrary graph needing a cross-caller merge.
//!
//! Release insertion happens **at every `App`** (a tail jump, to `k_ret`,
//! to a local continuation, or to a real function) — never deferred into a
//! continuation's own body — based on what's live *at that jump*: every
//! `CVal::Var` literally passed as an argument, plus, for every
//! `CVal::Label` appearing anywhere in the callee/args (a local
//! continuation's own name, wherever it's referenced — as the direct
//! callee for a join/loop jump, or buried in a real call's trailing
//! continuation argument), that continuation's own *free variables*
//! (`local_free_vars`) — struct-typed values it references directly rather
//! than receiving as an explicit argument, which would otherwise be
//! released prematurely by an *outer* scope that doesn't realize the
//! continuation still needs them (found by direct construction, not
//! guessed: `let x = Struct(..); let y = foo(); bar(x, y);` — the call to
//! `foo` doesn't mention `x` in its own arguments at all, only `bar`,
//! inside `foo`'s own resumption continuation, does).
//!
//! A local continuation's own body is then processed **once**, independent
//! of every caller, seeded with exactly the struct-typed values it's
//! responsible for: its own free variables (protected at every call site
//! above, so ownership correctly falls to it) plus its own declared
//! parameters that happen to be struct-typed (a join/loop's own carried
//! value, or a real call's own freshly-returned result — each a *fresh*
//! reference the receiving continuation now owns). Each continuation's
//! parameter types come from `CFunDef::carried_types` (populated for both
//! if-joins and loops, per `mlir_lower.rs`'s own `lower_if`/`lower_loop`)
//! or, for a real call's single-parameter resumption, the callee's own
//! declared return type (`CpsProgram`'s own top-level signatures) — never
//! guessed.
//!
//! ## Scope
//!
//! Non-cascading: `cleave_release` (`cleave-rt`) decrements a refcount and
//! frees a flat block, without recursing into struct-typed *fields* of the
//! value being released — a struct with only primitive/tensor fields is
//! freed correctly and completely; a struct with nested struct-typed
//! fields (`Network` containing `Dense`) has its own top-level allocation
//! freed, but a nested `Dense` whose refcount hasn't independently reached
//! zero elsewhere still leaks — strictly no worse than today's "everything
//! leaks forever," and a real, deliberate, separately-scoped follow-up
//! (type-specific, per-monomorphized-struct-type cascading release
//! functions), not attempted here. Tensors themselves are never
//! refcounted by this pass at all (see `is_refcounted`'s own doc comment)
//! — that's the separate tensor-*payload* problem `doc/backlog.md` already
//! documents.

use crate::cps::{CExpr, CFunDef, CTopLevelFn, CVal, CVar, CpsProgram, FreshVars, PrimOp};
use crate::egraph::max_cvar_in_program;
use crate::infer::Ty;
use std::collections::{HashMap, HashSet};

/// Whether `ty` needs refcounting at all — an ordinary, non-generic-or-
/// instantiated struct (`Ty::Con`/`Ty::App` naming a real `struct`
/// declaration) that construction actually heap-allocates via
/// `cleave_alloc_rc` (`mlir_lower.rs::lower_struct_construct`). A
/// `#[mlir_type(tensor)]`/`#[mlir_type(vector)]`-tagged struct (`Tensor`
/// itself) is excluded — `mlir_lower.rs::lower_tagged_struct_construct`'s
/// own doc comment confirms it never goes through `alloc_llvm_value` at
/// all, producing a bare native SSA value with no refcount header to act
/// on. A primitive/array/unit type is excluded structurally (neither
/// `Ty::Con` nor `Ty::App` naming a declared struct).
///
/// **A third exclusion, found by direct testing against a real, intermittent
/// memory-corruption bug, not assumed**: `name` must also have at least one
/// real `PrimOp::Struct` construction site somewhere in the *whole compiled
/// program* (`constructed`, below) — `stdlib/dynarray/dynarray.cleave`'s own
/// `RawBuf {}` (an ordinary, untagged, zero-field struct declaration, so the
/// first two checks alone don't exclude it) is the motivating case: its own
/// doc comment is explicit that it's "never constructed via `RawBuf(...)`
/// anywhere in this module, only ever produced/consumed by the `RawBuffer<T>`
/// impls below" — every real value of this type comes from an `extern fn`
/// return (`dynarray_alloc_ptr`/`dynarray_alloc_i32`/...), a plain
/// `realloc`-backed pointer from `cleave-rt`'s own internal allocator, with
/// *no* `RcHeader` in front of it at all. Before this exclusion, `is_
/// refcounted` was purely type-based, blind to *origin* — a `RawBuf`-typed
/// field (`DynArray.buf`) still got ordinary `Retain`/`Release` calls
/// inserted around it, each one reading/writing an `RcHeader` that was never
/// really there, off whatever bytes happened to sit just before that
/// pointer — real, silent, non-deterministic corruption (confirmed directly:
/// `cleave_release`'s own `Layout::from_size_align` panicking with
/// `LayoutError` roughly a third of the time, on a minimal `let h: DynArray<
/// i32> = dynarray_new(4);` alone, no `Point`/no `HeapStruct` involved at
/// all — varying run to run because the garbage byte pattern in the memory
/// immediately preceding a fresh allocation is itself unspecified). Every
/// *genuinely* refcounted struct in this codebase (`DynArray` itself
/// included) has a real construction site somewhere reachable — this
/// exclusion only ever fires for the `RawBuf`-shaped "opaque FFI handle,
/// produced solely by `extern fn`s" idiom, structurally, with no hardcoded
/// name anywhere.
pub(crate) fn is_refcounted(
    ty: &Ty,
    struct_schemas: &HashMap<String, crate::cps::StructSchema>,
    mlir_types: &HashMap<String, String>,
    constructed: &HashSet<String>,
) -> bool {
    let name = match ty {
        Ty::Con(name) | Ty::App(name, _) => name,
        _ => return false,
    };
    struct_schemas.contains_key(name)
        && !matches!(
            mlir_types.get(name).map(String::as_str),
            Some("tensor") | Some("vector")
        )
        && constructed.contains(name)
}

/// Every struct name with at least one real `PrimOp::Struct` construction
/// site anywhere in `program` — see `is_refcounted`'s own doc comment for
/// why this matters: a struct type with *no* real construction site at all
/// is only ever produced by an `extern fn` (the `RawBuf`-shaped "opaque FFI
/// handle" idiom), never by `cleave_alloc_rc`, so it must never be retained/
/// released. Walks every top-level function's own body, recursively through
/// every nested `Fix`/`If` — mirrors `region_analysis.rs`'s own established
/// "plain recursive `CExpr` walk, no fixpoint needed" shape for this same
/// kind of whole-program structural fact.
pub(crate) fn collect_constructed_struct_names(program: &CpsProgram) -> HashSet<String> {
    let mut names = HashSet::new();
    for f in &program.funcs {
        collect_constructed_in(&f.def.body, &mut names);
    }
    names
}

fn collect_constructed_in(expr: &CExpr, names: &mut HashSet<String>) {
    match expr {
        CExpr::LetPrim { op, cont, .. } => {
            if let PrimOp::Struct(name, _) = op {
                names.insert(name.clone());
            }
            collect_constructed_in(cont, names);
        }
        CExpr::App { .. } => {}
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_constructed_in(then_branch, names);
            collect_constructed_in(else_branch, names);
        }
        CExpr::Fix { defs, body } => {
            for d in defs {
                collect_constructed_in(&d.body, names);
            }
            collect_constructed_in(body, names);
        }
    }
}

/// The unit `()` type, used for `Retain`/`Release`'s own bound `LetPrim`
/// var — never read, same convention `Store`/`FieldStore` already use.
fn unit_ty() -> Ty {
    Ty::Con("()".to_string())
}

fn as_var(v: &CVal) -> Option<CVar> {
    match v {
        CVal::Var(v) => Some(*v),
        _ => None,
    }
}

/// Recursively collects every `CVar` this `CFunDef`'s own body — including
/// every nested `Fix`-local def's own body, at any depth — either *binds*
/// (a `LetPrim`'s own `var`, or a nested `Fix`-local def's own `params`)
/// or *references* (any `CVal::Var` anywhere). `def.params` themselves are
/// seeded into `bound` up front. Free variables are `referenced - bound` —
/// sound without tracking scope order at all, since every `CVar` in this
/// IR is minted once, globally unique (`FreshVars::var`), so "referenced
/// but never bound anywhere in this subtree" unambiguously means "must
/// come from an enclosing scope" — see the module's own doc comment for
/// why this is exactly the set that needs protecting at every call site
/// that might jump here.
/// `(free vars, func-position label deps)` — the second component is every
/// `Label` this def's own subtree ever tail-calls *as a callee* (`App::
/// func`, never merely passed along as a trailing-continuation argument —
/// `live_set` already resolves *that* case directly, per-App, by reading
/// the referenced label's own free variables at the exact call site that
/// passes it) — used by `collect_local_free_vars`'s own fixpoint, see that
/// function's own doc comment for why a def can't just use its own direct
/// references alone.
fn local_free_vars(def: &CFunDef) -> (HashSet<CVar>, HashSet<String>) {
    let mut bound: HashSet<CVar> = def.params.iter().copied().collect();
    let mut referenced: HashSet<CVar> = HashSet::new();
    let mut func_labels: HashSet<String> = HashSet::new();
    collect_bound_and_referenced(&def.body, &mut bound, &mut referenced, &mut func_labels);
    (
        referenced.difference(&bound).copied().collect(),
        func_labels,
    )
}

fn collect_bound_and_referenced(
    expr: &CExpr,
    bound: &mut HashSet<CVar>,
    referenced: &mut HashSet<CVar>,
    func_labels: &mut HashSet<String>,
) {
    match expr {
        CExpr::LetPrim {
            var, args, cont, ..
        } => {
            bound.insert(*var);
            for a in args {
                if let Some(v) = as_var(a) {
                    referenced.insert(v);
                }
            }
            collect_bound_and_referenced(cont, bound, referenced, func_labels);
        }
        CExpr::App { func, args } => {
            match func {
                CVal::Var(v) => {
                    referenced.insert(*v);
                }
                CVal::Label(name) => {
                    func_labels.insert(name.clone());
                }
                _ => {}
            }
            for a in args {
                if let Some(v) = as_var(a) {
                    referenced.insert(v);
                }
            }
        }
        CExpr::If {
            cond,
            then_branch,
            else_branch,
        } => {
            if let Some(v) = as_var(cond) {
                referenced.insert(v);
            }
            collect_bound_and_referenced(then_branch, bound, referenced, func_labels);
            collect_bound_and_referenced(else_branch, bound, referenced, func_labels);
        }
        CExpr::Fix { defs, body } => {
            for d in defs {
                bound.extend(d.params.iter().copied());
                collect_bound_and_referenced(&d.body, bound, referenced, func_labels);
            }
            collect_bound_and_referenced(body, bound, referenced, func_labels);
        }
    }
}

/// Every `CVar` this function's own nested `Fix` structure ever binds,
/// resolved to its own concrete `Ty` — a `LetPrim`'s own `var` (from its
/// own declared `ty`), a join/loop `Fix`-local def's own `params` (from
/// `CFunDef::carried_types`, populated for both — see `mlir_lower.rs`'s
/// own `lower_if`/`lower_loop`), and a real call's own single-parameter
/// resumption def (from the callee's own declared return type,
/// `signatures` — resolved the identical way `mlir_lower.rs::LowerCtx::
/// signatures` already does, just before MLIR types exist to look up
/// against). The function's own top-level `params` are *not* included —
/// callers never seed `owned` from this map for those (see
/// `insert_refcounting_fn`'s own doc comment: borrowed, never released).
///
/// Alongside every `CVar`'s own type, this also computes its own
/// *ownership provenance* — whether releasing it is ever this program's
/// own responsibility at all, as opposed to a borrowed alias into memory
/// someone else manages. **This distinction is load-bearing, not an
/// optimization**: found by direct testing (a real `STATUS_HEAP_
/// CORRUPTION` crash, `examples/digits-interop`, bisected down to a 5-line
/// repro) — a naive "every struct-typed free variable a continuation
/// still needs must be *that continuation's own* responsibility to
/// release" rule (this module's own first version) is *unsound*: `Dense::
/// forward`-style code reads `model.l1` (`PrimOp::Field` on a *borrowed*
/// function parameter) and references it again inside a later
/// continuation (after an intervening real call) — that field read is
/// never independently owned, it's an alias into `model`'s own storage,
/// which its *caller* still needs after this function returns; releasing
/// it corrupts the caller's own object.
///
/// The rule: a value is owned exactly when it's a fresh allocation
/// (`PrimOp::Struct`) or a real call's own freshly-returned result
/// (a resumption def's single param) — *or* a `PrimOp::Field` read whose
/// own *base* is itself owned (propagated, since embedding an owned value
/// into a fresh struct and reading it back out doesn't change who's
/// responsible for it — see `rewrite_body`'s own retain-on-construction
/// logic, which is what keeps a value's refcount honest through exactly
/// this kind of embed-then-read round trip). Everything else (a function
/// parameter itself, a `Field` read whose base isn't owned, any other
/// `PrimOp`) is conservatively *not* owned — matching `doc/hld.md`'s own
/// "unprovable -> conservative" default, just applied one level deeper
/// than the mut-vs-plain-let split it originally described.
///
/// A join/loop `Fix`-local def's own *carried* param's ownership is
/// resolved from what's actually passed for that position at its own
/// entry call — a loop's own `Fix.body` directly (its self-recursive
/// back-edge, found deeper inside its own body, is expected to agree,
/// same underlying value each iteration in every case this module has
/// been tested against); a join's own two branches (`find_call_args`,
/// both are expected to tail-call the same join, `mlir_lower.rs::lower_
/// if`'s own doc comment) — owned only if *both* agree.
fn collect_var_info(
    top: &CTopLevelFn,
    signatures: &HashMap<String, Ty>,
    var_types: &mut HashMap<CVar, Ty>,
    owned_origin: &mut HashMap<CVar, bool>,
) {
    walk_var_info(&top.def.body, signatures, var_types, owned_origin);
}

fn is_owned_val(v: &CVal, owned_origin: &HashMap<CVar, bool>) -> bool {
    match v {
        CVal::Var(cv) => owned_origin.get(cv).copied().unwrap_or(false),
        _ => false,
    }
}

/// Finds the argument list of the (unique, by this IR's own convention)
/// `App` tail-calling `target` anywhere within `expr` — used to resolve a
/// join's own carried-param ownership from what each of its two branches
/// actually passes. Searches through every `LetPrim`/`If`/`Fix` (including
/// a nested `Fix`-local def's own body — a nested real call's own
/// resumption can itself end by tail-calling an *outer* join, e.g. `if c1
/// { let y = foo(); y } else { 0 }`).
fn find_call_args<'a>(expr: &'a CExpr, target: &str) -> Option<&'a [CVal]> {
    match expr {
        CExpr::LetPrim { cont, .. } => find_call_args(cont, target),
        CExpr::App {
            func: CVal::Label(name),
            args,
        } if name == target => Some(args),
        CExpr::App { .. } => None,
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => find_call_args(then_branch, target).or_else(|| find_call_args(else_branch, target)),
        CExpr::Fix { defs, body } => {
            for d in defs {
                if let Some(a) = find_call_args(&d.body, target) {
                    return Some(a);
                }
            }
            find_call_args(body, target)
        }
    }
}

fn walk_var_info(
    expr: &CExpr,
    signatures: &HashMap<String, Ty>,
    var_types: &mut HashMap<CVar, Ty>,
    owned_origin: &mut HashMap<CVar, bool>,
) {
    match expr {
        CExpr::LetPrim {
            var,
            ty,
            op,
            args,
            cont,
        } => {
            var_types.insert(*var, ty.clone());
            let is_owned = match op {
                PrimOp::Struct(..) => true,
                PrimOp::Field { .. } => args
                    .first()
                    .is_some_and(|base| is_owned_val(base, owned_origin)),
                _ => false,
            };
            owned_origin.insert(*var, is_owned);
            walk_var_info(cont, signatures, var_types, owned_origin);
        }
        CExpr::App { .. } => {}
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            walk_var_info(then_branch, signatures, var_types, owned_origin);
            walk_var_info(else_branch, signatures, var_types, owned_origin);
        }
        CExpr::Fix { defs, body } => {
            let [def] = &defs[..] else {
                // Multi-def `Fix` isn't lowered anywhere yet either
                // (`mlir_lower.rs::lower_cexpr`'s own identical panic) —
                // nothing to type here until that's real.
                walk_var_info(body, signatures, var_types, owned_origin);
                return;
            };
            // `body` (both join branches, or nothing new for a bare loop-
            // entry `App`) is walked *first* — a carried param's own
            // ownership is resolved from arguments bound *inside* it, so
            // those need their own provenance settled before `def`'s own
            // params can be.
            walk_var_info(body, signatures, var_types, owned_origin);
            if let Some(carried) = &def.carried_types {
                match body.as_ref() {
                    CExpr::If { .. } => {
                        for (i, (p, t)) in def.params.iter().zip(carried).enumerate() {
                            var_types.insert(*p, t.clone());
                            let then_owned = find_call_args(then_branch_of(body), &def.name)
                                .and_then(|a| a.get(i))
                                .is_some_and(|v| is_owned_val(v, owned_origin));
                            let else_owned = find_call_args(else_branch_of(body), &def.name)
                                .and_then(|a| a.get(i))
                                .is_some_and(|v| is_owned_val(v, owned_origin));
                            owned_origin.insert(*p, then_owned && else_owned);
                        }
                    }
                    CExpr::App { args, .. } => {
                        for (i, (p, t)) in def.params.iter().zip(carried).enumerate() {
                            var_types.insert(*p, t.clone());
                            let owned = args.get(i).is_some_and(|v| is_owned_val(v, owned_origin));
                            owned_origin.insert(*p, owned);
                        }
                    }
                    _ => {}
                }
            } else if let CExpr::App {
                func: CVal::Label(callee),
                ..
            } = body.as_ref()
            {
                // A real call's own resumption: exactly one param, typed
                // by the callee's own declared return type — mirrors
                // `mlir_lower.rs::lower_real_call`'s identical lookup.
                // (The loop-entry shape also matches this same `App`
                // pattern but always carries `carried_types`, handled by
                // the branch above — this arm is only reached when it
                // doesn't, i.e. a genuine real call.) Always owned — a
                // real call always either constructs a fresh value or
                // forwards ownership of one it already held (its own
                // `k_ret` never fires on a merely-borrowed value it
                // didn't itself return).
                if let ([p], Some(result_ty)) = (&def.params[..], signatures.get(callee)) {
                    var_types.insert(*p, result_ty.clone());
                    owned_origin.insert(*p, true);
                }
            }
            walk_var_info(&def.body, signatures, var_types, owned_origin);
        }
    }
}

fn then_branch_of(body: &CExpr) -> &CExpr {
    match body {
        CExpr::If { then_branch, .. } => then_branch,
        _ => unreachable!("only called when `body` is already known to be `CExpr::If`"),
    }
}

fn else_branch_of(body: &CExpr) -> &CExpr {
    match body {
        CExpr::If { else_branch, .. } => else_branch,
        _ => unreachable!("only called when `body` is already known to be `CExpr::If`"),
    }
}

/// Every `Fix`-local def's own name mapped to its own free variables —
/// computed once per function, consulted at every `App` site to decide
/// what's still needed by whatever continuation it might jump to (see the
/// module's own doc comment). Unfiltered — a variable free in *both* an
/// outer def and a def nested inside it (a value needed throughout an
/// outer loop, referenced again deep inside an inner one nested within
/// it) appears in *both* entries here, exactly matching what each one's
/// own body genuinely still needs. Contrast with `collect_local_claim_
/// vars`, which is *not* like this, deliberately — see that function's
/// own doc comment for why the two must stay separate.
///
/// **Transitive through a bare tail-call to another local label, via a
/// fixpoint over every def's own `func_labels` (`local_free_vars`'s own
/// doc comment) — found necessary by direct testing (a third distinct
/// `STATUS_HEAP_CORRUPTION`, same repro, surviving the first two fixes).**
/// A "trampoline" def — one whose own body, or some tail of it, is just
/// `App{Label(other), args}` with no computation of its own (a loop's own
/// back-edge resumption after incrementing its index: `Ring::add(i, 1,
/// k)`, `k`'s own body being nothing but `loop(i2, ...carried)`) —
/// doesn't *directly* reference anything the loop itself still needs
/// (it only forwards the already-carried values) — but the loop it
/// tail-calls does, every subsequent iteration. Without this, `live_set`
/// (built directly from this map) sees the trampoline as not needing a
/// value the loop genuinely still does, releasing it one hop too early.
/// Bounded and guaranteed to terminate: values only ever grow (a pure
/// union each round), and there are finitely many `(def, CVar)` pairs.
fn collect_local_free_vars(top: &CTopLevelFn, out: &mut HashMap<String, HashSet<CVar>>) {
    let mut func_labels: HashMap<String, HashSet<String>> = HashMap::new();
    walk_local_free_vars(&top.def.body, out, &mut func_labels);
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
}

fn walk_local_free_vars(
    expr: &CExpr,
    out: &mut HashMap<String, HashSet<CVar>>,
    func_labels: &mut HashMap<String, HashSet<String>>,
) {
    match expr {
        CExpr::LetPrim { cont, .. } => walk_local_free_vars(cont, out, func_labels),
        CExpr::App { .. } => {}
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            walk_local_free_vars(then_branch, out, func_labels);
            walk_local_free_vars(else_branch, out, func_labels);
        }
        CExpr::Fix { defs, body } => {
            for d in defs {
                let (free_vars, deps) = local_free_vars(d);
                out.insert(d.name.clone(), free_vars);
                func_labels.insert(d.name.clone(), deps);
                walk_local_free_vars(&d.body, out, func_labels);
            }
            walk_local_free_vars(body, out, func_labels);
        }
    }
}

/// Every `Fix`-local def's own name mapped to the *subset* of its own free
/// variables it's actually responsible for eventually releasing — used
/// only for seeding (`rewrite_body`'s own `Fix` arm), never for `live_set`
/// (which needs the *unfiltered* `local_free_vars` above: a value still
/// has to be protected at every call site that might reach a continuation
/// still using it, regardless of who ultimately owns releasing it).
///
/// **Why this can't just be `local_free_vars` itself, found by direct
/// testing (a second real `STATUS_HEAP_CORRUPTION` in the same repro,
/// surviving the first fix): a value can be a genuine free variable of
/// *both* an outer, recursively-re-entered def and an inner one nested
/// inside it** — `train_and_evaluate`'s own `opt` (the `Sgd` optimizer, a
/// fresh, owned struct, constructed once) is referenced throughout the
/// *whole* outer epoch loop, including deep inside the *inner* per-sample
/// loop nested within each epoch. Seeding *both* loops independently from
/// their own (unfiltered) free variables — this module's own first
/// version — makes the *inner* loop release it the moment its own current
/// invocation no longer needs it (at the end of *one* epoch) — correct
/// only for the very last epoch; every earlier one leaves the *outer*
/// loop's own next iteration reading a freed pointer the moment it
/// re-enters the inner loop again.
///
/// The fix: a free variable claimed by an *enclosing* def is never also
/// claimed by a def nested inside it — ownership of releasing it belongs
/// to the *shallowest* def that references it (the one whose own exit is
/// genuinely final, not just "this particular invocation, among possibly
/// many, is done"), computed top-down, propagating each def's own already-
/// claimed set down into whatever's nested inside it.
fn collect_local_claim_vars(top: &CTopLevelFn, out: &mut HashMap<String, HashSet<CVar>>) {
    walk_local_claim_vars(&top.def.body, &HashSet::new(), out);
}

fn walk_local_claim_vars(
    expr: &CExpr,
    claimed_by_ancestors: &HashSet<CVar>,
    out: &mut HashMap<String, HashSet<CVar>>,
) {
    match expr {
        CExpr::LetPrim { cont, .. } => walk_local_claim_vars(cont, claimed_by_ancestors, out),
        CExpr::App { .. } => {}
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            walk_local_claim_vars(then_branch, claimed_by_ancestors, out);
            walk_local_claim_vars(else_branch, claimed_by_ancestors, out);
        }
        CExpr::Fix { defs, body } => {
            for d in defs {
                let own_claim: HashSet<CVar> = local_free_vars(d)
                    .0
                    .difference(claimed_by_ancestors)
                    .copied()
                    .collect();
                let claimed_including_this: HashSet<CVar> = claimed_by_ancestors
                    .union(&own_claim)
                    .copied()
                    .collect();
                out.insert(d.name.clone(), own_claim);
                walk_local_claim_vars(&d.body, &claimed_including_this, out);
            }
            walk_local_claim_vars(body, claimed_by_ancestors, out);
        }
    }
}

struct RefcountCtx<'a> {
    struct_schemas: &'a HashMap<String, crate::cps::StructSchema>,
    mlir_types: &'a HashMap<String, String>,
    /// See `is_refcounted`'s own doc comment for why this exists — a struct
    /// name absent here is only ever produced by an `extern fn` (`RawBuf`'s
    /// own "opaque FFI handle" idiom), never by `cleave_alloc_rc`, so it
    /// must never be retained/released.
    constructed_structs: &'a HashSet<String>,
    var_types: &'a HashMap<CVar, Ty>,
    /// Whether releasing a given `CVar` is ever this program's own
    /// responsibility at all — see `collect_var_info`'s own doc comment
    /// for the full rule and the real corruption bug this exists to
    /// prevent. Consulted everywhere a value is considered for seeding
    /// into a scope's own `owned` set (never for `Retain`, which stays
    /// unconditional on type alone — see that same doc comment).
    owned_origin: &'a HashMap<CVar, bool>,
    local_free_vars: &'a HashMap<String, HashSet<CVar>>,
    local_claim_vars: &'a HashMap<String, HashSet<CVar>>,
    fresh: &'a FreshVars,
}

impl RefcountCtx<'_> {
    fn is_rc(&self, ty: &Ty) -> bool {
        is_refcounted(
            ty,
            self.struct_schemas,
            self.mlir_types,
            self.constructed_structs,
        )
    }
}

/// Runs the whole pass — see the module's own doc comment for the design.
/// Applied once per top-level function, on the *final* optimized
/// `CpsProgram`, right before MLIR lowering (`pipeline.rs`) — after the
/// e-graph pass, never before it: `egraph.rs` has no notion of `Retain`/
/// `Release`'s own effectful ordering requirements, and inserting them
/// earlier risks the e-graph's own rewriting scrambling them.
pub fn insert_refcounting(
    program: CpsProgram,
    struct_schemas: &HashMap<String, crate::cps::StructSchema>,
    mlir_types: &HashMap<String, String>,
) -> CpsProgram {
    let fresh = FreshVars::starting_at(max_cvar_in_program(&program) + 1);
    let signatures: HashMap<String, Ty> = program
        .funcs
        .iter()
        .map(|f| (f.def.name.clone(), f.result.clone()))
        .collect();
    let constructed_structs = collect_constructed_struct_names(&program);
    let funcs = program
        .funcs
        .into_iter()
        .map(|top| {
            let mut var_types = HashMap::new();
            let mut owned_origin = HashMap::new();
            collect_var_info(&top, &signatures, &mut var_types, &mut owned_origin);
            let mut local_free_vars = HashMap::new();
            collect_local_free_vars(&top, &mut local_free_vars);
            let mut local_claim_vars = HashMap::new();
            collect_local_claim_vars(&top, &mut local_claim_vars);
            let ctx = RefcountCtx {
                struct_schemas,
                mlir_types,
                constructed_structs: &constructed_structs,
                var_types: &var_types,
                owned_origin: &owned_origin,
                local_free_vars: &local_free_vars,
                local_claim_vars: &local_claim_vars,
                fresh: &fresh,
            };
            insert_refcounting_fn(top, &ctx)
        })
        .collect();
    CpsProgram { funcs }
}

/// The function's own top-level `params` are deliberately never seeded
/// into `owned` — they're this function's real ABI parameters, borrowed
/// from the caller (see the module's own doc comment for why that's the
/// only sound convention), never this function's own responsibility to
/// release.
fn insert_refcounting_fn(top: CTopLevelFn, ctx: &RefcountCtx) -> CTopLevelFn {
    let k_ret = top.k_ret;
    let new_body = rewrite_body(top.def.body, Vec::new(), k_ret, ctx);
    CTopLevelFn {
        def: CFunDef {
            name: top.def.name,
            params: top.def.params,
            body: new_body,
            carried_types: top.def.carried_types,
        },
        param_types: top.param_types,
        result: top.result,
        k_ret: top.k_ret,
        origin: top.origin,
        is_export: top.is_export,
        export_symbol: top.export_symbol,
    }
}

/// The core rewrite — see the module's own doc comment for the design;
/// `owned` is this scope's own struct-typed values not yet accounted for,
/// grown by every `PrimOp::Struct` construction seen along the way (and
/// seeded, for a `Fix`-local def, before this is ever called for its own
/// body — see the `Fix` arm below).
fn rewrite_body(
    expr: CExpr,
    mut owned: Vec<(CVar, Ty)>,
    k_ret: CVar,
    ctx: &RefcountCtx,
) -> CExpr {
    match expr {
        CExpr::LetPrim {
            var,
            ty,
            op,
            args,
            cont,
        } => {
            // A fresh construction is always this scope's own responsibility
            // to release. A `Field` read is too, but only conditionally —
            // exactly when `collect_var_info` already determined it's
            // `owned` (propagated from an owned base, `collect_var_info`'s
            // own doc comment) — a read off a *borrowed* base must never be
            // pushed here (nothing owns it, nothing should ever release
            // it). Without this, a value the retain-on-read fix just below
            // correctly protects from a container's own cascading release
            // is never actually tracked for release *itself* — retained
            // once, released never, a real per-call leak found by direct
            // testing (`examples/mnist-interop`, real training: `net_grad`'s
            // own returned gradient `Network`, extracted via `g.2`,
            // consumed only as a *borrowed* argument to `Optimizer::step`
            // and never referenced again — correctly retained, but with no
            // scope ever seeded to release it, leaking one full network's
            // worth of memory *every single training sample*; invisible on
            // `digits-interop`'s own small network, ~9KB/sample, only
            // large enough to matter at real MNIST scale, ~2MB/sample).
            let field_read_owned = matches!(&op, PrimOp::Field { .. })
                && ctx.owned_origin.get(&var).copied().unwrap_or(false);
            if (matches!(&op, PrimOp::Struct(..)) || field_read_owned) && ctx.is_rc(&ty) {
                owned.push((var, ty.clone()));
            }
            // Retain-on-store: an *existing* struct-typed value written
            // into another struct's own storage creates a second,
            // independent owner — see the module's own doc comment.
            // `FieldStore`/`Store` (`s.field = v` / `a[i] = v`, a real
            // *mutation* of already-existing storage) alias exactly their
            // own trailing `value` operand; `Struct` construction itself
            // aliases *every* argument that's its own field values (`Line
            // (a: p1, b: p2)` embeds `p1`'s/`p2`'s own pointers directly,
            // found by direct testing — a value returned this way, still
            // separately owned and released by *this* function too,
            // otherwise goes dangling the moment this function's own
            // `owned`-tracking releases `p1`/`p2` at its own return,
            // silently — no immediate crash, since `cleave_release`'s own
            // `dealloc` doesn't have to scribble over freed memory right
            // away, exactly the kind of bug that only manifests later,
            // once something else reuses the address).
            //
            // **`Array` needs the identical treatment, found by direct
            // testing against a real, intermittent memory-corruption bug,
            // not assumed**: `[p1, p2, p3]` (`ExprKind::ArrayLit`) embeds
            // `p1`'s/`p2`'s/`p3`'s own pointers directly, the exact same
            // "aliases every argument" shape `Struct` construction already
            // gets — but was missing here entirely, so an array literal of
            // struct-typed elements got *no* retain at all for any of them.
            // Root-caused against `examples/convex_hull.cleave --run`'s own
            // intermittent corruption by dumping `--dump-cps-optimized`
            // directly (the *actual* generated code, not a guess): `let
            // points: [Point; 3] = [Point(...), Point(...), Point(...)];`
            // released all three freshly-constructed `Point`s **immediately
            // after** building the array containing them — `points[i]`'s
            // own reads, every one of them, for the rest of `main`, read
            // back through an already-released (and potentially already-
            // reused) pointer. `ArrayRepeat` (`[v; N]`) is a plausible,
            // structurally similar risk — not confirmed by a real failing
            // case the way `Array` was, and deliberately not touched here;
            // flagged in `doc/backlog.md` instead of guessed at.
            let retain_targets: Vec<CVal> = match &op {
                PrimOp::FieldStore { .. } | PrimOp::Store { .. } => {
                    args.last().cloned().into_iter().collect()
                }
                PrimOp::Struct(..) | PrimOp::Array => args.clone(),
                _ => Vec::new(),
            };
            let mut retains: Vec<(CVal, Ty)> = Vec::new();
            for target in retain_targets {
                if let CVal::Var(cv) = &target {
                    if let Some(rty) = ctx.var_types.get(cv) {
                        if ctx.is_rc(rty) {
                            retains.push((target, rty.clone()));
                        }
                    }
                }
            }

            // Retain-on-read: `PrimOp::Field` reading a refcounted value
            // *out* of a container aliases that container's own copy —
            // the mirror image of retain-on-store above, needed for the
            // identical reason once `lower_release_cascade` (`mlir_lower.
            // rs`) exists: the read result is tracked as its own,
            // independently-owned value going forward (`collect_var_info`'s
            // own `PrimOp::Field` rule already propagates ownership from
            // the base — this is that same decision's other half), and
            // without a retain here, the container's own *eventual*
            // cascading release decrements the exact same underlying
            // refcount the read result's *own* later release also expects
            // to — found by direct testing, a real `STATUS_ACCESS_
            // VIOLATION`: `Optimizer::step`'s own returned tuple, once
            // released by its caller, cascades into freeing the very
            // `Network` that same caller just extracted and is about to
            // carry into the next training iteration.
            let field_read_retain: Option<Ty> = if let PrimOp::Field { .. } = &op {
                if ctx.is_rc(&ty) && ctx.owned_origin.get(&var).copied().unwrap_or(false) {
                    Some(ty.clone())
                } else {
                    None
                }
            } else {
                None
            };

            let new_cont = rewrite_body(*cont, owned, k_ret, ctx);
            let new_cont = if let Some(rty) = field_read_retain {
                let rvar = ctx.fresh.var();
                CExpr::LetPrim {
                    var: rvar,
                    ty: unit_ty(),
                    op: PrimOp::Retain(rty),
                    args: vec![CVal::Var(var)],
                    cont: Box::new(new_cont),
                }
            } else {
                new_cont
            };
            let mut result = CExpr::LetPrim {
                var,
                ty,
                op,
                args,
                cont: Box::new(new_cont),
            };
            for (target, rty) in retains {
                let rvar = ctx.fresh.var();
                result = CExpr::LetPrim {
                    var: rvar,
                    ty: unit_ty(),
                    op: PrimOp::Retain(rty),
                    args: vec![target],
                    cont: Box::new(result),
                };
            }
            result
        }
        CExpr::App { func, args } => {
            let to_release = releases_for_app(&func, &args, owned, ctx);
            wrap_releases(to_release, CExpr::App { func, args }, ctx)
        }
        CExpr::If {
            cond,
            then_branch,
            else_branch,
        } => {
            let new_then = rewrite_body(*then_branch, owned.clone(), k_ret, ctx);
            let new_else = rewrite_body(*else_branch, owned, k_ret, ctx);
            CExpr::If {
                cond,
                then_branch: Box::new(new_then),
                else_branch: Box::new(new_else),
            }
        }
        CExpr::Fix { defs, body } => {
            // A real call's own resumption def is structurally distinct
            // from a join/loop def (see `mlir_lower.rs::lower_cexpr`'s own
            // identical dispatch): its `carried_types` is `None`, and
            // `Fix.body` targets a genuinely different unit, not itself.
            // Whether `Fix.body` is a bare `App` — a loop's own entry *or*
            // a real call's own resumption dispatch (`mlir_lower.rs::
            // lower_cexpr`'s own identical two-way split within the `App`
            // case) — as opposed to an `If` (a join). Both `App` shapes
            // need `transferred` seeded into the sole def below; the `If`
            // shape doesn't — see `transferred`'s own doc comment.
            let body_is_app = matches!(&defs[..], [_]) && matches!(body.as_ref(), CExpr::App { .. });

            // What's live at `Fix.body`'s own jump, computed *before*
            // building `new_defs` below (both `App` shapes need it to seed
            // correctly) — `transferred` is exactly `releases_for_app`'s
            // own complement: every value the calling scope still owned
            // that's protected here (a literal argument, or needed by the
            // callee's own free variables) rather than released outright.
            //
            // **Why this is needed at all, found by direct testing (a real
            // `STATUS_HEAP_CORRUPTION`, `examples/digits-interop`,
            // bisected to exactly 2 real epochs — 1 worked, 2 didn't,
            // confirming a genuine per-iteration leak, not a one-off).** A
            // value the calling scope still owns, when passed as a
            // *literal argument* to the call/jump itself, is correctly
            // protected here (still needed *for* it) — but if it's never
            // referenced again afterward (not one of the target's own free
            // variables — the calling scope's own processing ends at this
            // exact `App`, it has no "later" of its own to release it in),
            // nothing else ever gets a chance to release it. For a real
            // call specifically: the callee never releases a borrowed
            // parameter, and the resumption's own free-variable-only seed
            // has no way to know about a value entirely consumed *by the
            // call*, never carried past it. The fix: the target inherits
            // the calling scope's *whole* remaining ownership, not just
            // its own free variables — CPS's own "before" and "after" a
            // call/jump are one and the same logical scope, just split
            // there.
            //
            // **A loop's own entry needs the identical fix, found by
            // direct testing too (a second, distinct `STATUS_HEAP_
            // CORRUPTION` surviving the first fix, same repro): a value
            // can be a free variable of an *intermediate* ancestor
            // scope — a real call's own resumption sitting between where
            // the value is constructed and the loop itself (`train_and_
            // evaluate`'s own `opt`, constructed once, then threaded
            // through `Optimizer::init_state`'s own resumption before
            // ever reaching the training loop) — which `local_claim_vars`
            // correctly assigns to that intermediate scope (the shallowest
            // *syntactic* free-variable owner), but which the loop itself
            // also needs, every iteration. Without this, the intermediate
            // scope protects it at its own single `App` (correctly, it's
            // still needed) but never gets *another* chance to release it
            // (nothing follows that `App` in its own body), and the loop
            // never claims it either (excluded from `local_claim_vars` by
            // the intermediate ancestor already claiming it) — the exact
            // same "no scope ever gets a turn" gap, one level removed.
            let (to_release, transferred, entry_arg_vars): (
                Vec<(CVar, Ty)>,
                Vec<(CVar, Ty)>,
                HashSet<CVar>,
            ) = match body.as_ref() {
                CExpr::App { func, args } => {
                    let live: HashSet<CVar> = live_set(func, args, ctx);
                    let arg_vars: HashSet<CVar> = args.iter().filter_map(as_var).collect();
                    let (transferred, to_release) =
                        owned.into_iter().partition(|(v, _)| live.contains(v));
                    (to_release, transferred, arg_vars)
                }
                _ => (Vec::new(), owned, HashSet::new()),
            };

            let new_defs = defs
                .into_iter()
                .map(|def| {
                    let mut seed: Vec<(CVar, Ty)> = Vec::new();
                    let mut seen: HashSet<CVar> = HashSet::new();
                    let is_owned = |v: &CVar| ctx.owned_origin.get(v).copied().unwrap_or(false);
                    if let Some(fv) = ctx.local_claim_vars.get(&def.name) {
                        for v in fv {
                            if let Some(ty) = ctx.var_types.get(v) {
                                if ctx.is_rc(ty) && is_owned(v) && seen.insert(*v) {
                                    seed.push((*v, ty.clone()));
                                }
                            }
                        }
                    }
                    for p in &def.params {
                        if let Some(ty) = ctx.var_types.get(p) {
                            if ctx.is_rc(ty) && is_owned(p) && seen.insert(*p) {
                                seed.push((*p, ty.clone()));
                            }
                        }
                    }
                    if body_is_app {
                        // A loop's own entry call passes its own initial
                        // carried values as literal arguments -- the exact
                        // same underlying references as `def.params`
                        // above, just under the *caller's* own CVar names
                        // rather than the loop's own. Excluded here to
                        // avoid seeding the identical reference twice
                        // under two different names (a real double-
                        // release, not just redundant bookkeeping — found
                        // by direct testing: this exact case broke `mlir_
                        // lower.rs::lower_loop`'s own strict "condition
                        // chain has no intervening `LetPrim`" shape
                        // requirement the moment it fired). A real call's
                        // own resumption has no such overlap — its sole
                        // param is the *call's own return value*, never
                        // one of its own arguments — so this exclusion is
                        // a no-op there.
                        // `transferred` genuinely does hand ownership all
                        // the way down through a whole chain of App-shaped
                        // `Fix`es (a value can be re-transferred many times
                        // over, once per hop, before finally landing where
                        // it's released) — this is *not* double-counting
                        // with `local_claim_vars` above, even when the
                        // same variable appears in both: `owned` here is a
                        // strictly local, per-scope list, `partition`ed
                        // between `to_release` and `transferred` — once a
                        // value moves into `transferred`, *this* def no
                        // longer tracks it at all, so there's nothing left
                        // for it to double-release. Requires `local_free_
                        // vars` to be genuinely transitive through a bare
                        // tail-call to another local label (`collect_
                        // local_free_vars`'s own doc comment) — without
                        // that, a "trampoline" def (one whose own body is
                        // just `App{Label(other), args}`, e.g. a loop's
                        // own back-edge resumption after incrementing its
                        // index) looks like it doesn't need a value at
                        // all, and `transferred` releases it there
                        // instead of continuing to hand it down to `other`
                        // — found by direct testing, the third distinct
                        // `STATUS_HEAP_CORRUPTION` in the same repro.
                        let is_loop = def.carried_types.is_some();
                        for (v, ty) in &transferred {
                            if (!is_loop || !entry_arg_vars.contains(v)) && seen.insert(*v) {
                                seed.push((*v, ty.clone()));
                            }
                        }
                    }
                    // A loop def's own body needs the same shape-preserving
                    // care as `Fix.body` itself, one level further —
                    // `rewrite_loop_condition_chain`'s own doc comment has
                    // the full story (a fourth distinct real crash this
                    // exact repro produced, found by direct testing).
                    let is_loop_def =
                        def.carried_types.is_some() && matches!(body.as_ref(), CExpr::App { .. });
                    let new_def_body = if is_loop_def {
                        rewrite_loop_condition_chain(def.body, seed, k_ret, ctx)
                    } else {
                        rewrite_body(def.body, seed, k_ret, ctx)
                    };
                    CFunDef {
                        name: def.name,
                        params: def.params,
                        body: new_def_body,
                        carried_types: def.carried_types,
                    }
                })
                .collect();
            // `Fix.body` must stay exactly one of the shapes `mlir_lower.rs::
            // lower_cexpr` pattern-matches (a bare `If`, or a bare `App` for
            // a loop's own entry / a real call) -- it can't be wrapped in a
            // `Release` `LetPrim` the way an ordinary `App` reachable from
            // anywhere else can. The `If` case never needed a wrapper to
            // begin with (both branches inherit `owned` unchanged and each
            // end their own path with their own `App`, already handled by
            // the recursive call below). The `App` case does need one --
            // computed above (`to_release`) exactly like the ordinary
            // `CExpr::App` arm below, but placed *before the whole `Fix`
            // node* instead of around the bare `App` itself: semantically
            // identical (`Fix` has no runtime effect of its own, its defs
            // are just local labels), and preserves the exact shape
            // `lower_cexpr` requires.
            let new_outer_body = match *body {
                CExpr::App { func, args } => CExpr::App { func, args },
                other => rewrite_body(other, transferred, k_ret, ctx),
            };
            let fix_node = CExpr::Fix {
                defs: new_defs,
                body: Box::new(new_outer_body),
            };
            wrap_releases(to_release, fix_node, ctx)
        }
    }
}

/// A loop def's own body is a strict special case, one level stricter than
/// `Fix.body`'s own shape requirement above — `mlir_lower.rs::lower_loop`
/// requires it to be *exactly* a chain of `Fix{defs:[k], body:App{Label(
/// callee), args}}` nodes (each a real call, its own resumption `k`
/// continuing the chain — `doc`'s own "a while-loop condition needing more
/// than one chained real call" case) ending in a bare `If` — no
/// intervening `LetPrim` anywhere in that *whole chain*, not even wrapping
/// one of its own interior `Fix` nodes the way the ordinary `Fix` arm
/// above safely does for a ordinary, standalone `Fix`. Found by direct
/// testing, a fourth distinct real crash in the same repro (`examples/
/// mnist-interop`, surviving all three earlier fixes): a loop carrying a
/// struct value that's never read inside the loop body itself (only
/// replaced) is correctly judged dead the moment the loop is entered —
/// but the ordinary `Fix` arm's own wrapping, applied to the condition
/// chain's *first* link, put that release *before* `loop_def.body` itself,
/// which is exactly the shape `lower_loop` forbids (its own walk starts
/// at `&loop_def.body` expecting `Fix`/`If` directly, no `LetPrim`).
///
/// The fix: nothing in `owned` is released anywhere in this chain — held
/// artificially live all the way through, deferred — until the terminal
/// `If` is reached, where ordinary `rewrite_body` processing resumes (an
/// `If`'s own branches carry no such constraint, each ends in its own
/// ordinary `App` to the join, already handled correctly).
fn rewrite_loop_condition_chain(
    expr: CExpr,
    owned: Vec<(CVar, Ty)>,
    k_ret: CVar,
    ctx: &RefcountCtx,
) -> CExpr {
    match expr {
        CExpr::Fix { defs, body } if matches!(body.as_ref(), CExpr::App { .. }) => {
            let new_defs = defs
                .into_iter()
                .map(|def| CFunDef {
                    name: def.name,
                    params: def.params,
                    body: rewrite_loop_condition_chain(def.body, owned.clone(), k_ret, ctx),
                    carried_types: def.carried_types,
                })
                .collect();
            CExpr::Fix {
                defs: new_defs,
                body,
            }
        }
        other => rewrite_body(other, owned, k_ret, ctx),
    }
}

/// The live set at a jump (`App{func, args}`) — every `CVal::Var` literally
/// passed, plus, for every `CVal::Label` appearing anywhere in `func`/
/// `args` that names a known `Fix`-local continuation, that continuation's
/// own free variables — see the module's own doc comment for why both are
/// needed.
fn live_set(func: &CVal, args: &[CVal], ctx: &RefcountCtx) -> HashSet<CVar> {
    let mut live: HashSet<CVar> = HashSet::new();
    for v in std::iter::once(func).chain(args.iter()) {
        match v {
            CVal::Var(cv) => {
                live.insert(*cv);
            }
            CVal::Label(name) => {
                if let Some(fv) = ctx.local_free_vars.get(name) {
                    live.extend(fv.iter().copied());
                }
            }
            _ => {}
        }
    }
    live
}

/// Exactly the entries of `owned` *not* covered by `live_set` — what's safe
/// to release right before this jump.
fn releases_for_app(
    func: &CVal,
    args: &[CVal],
    owned: Vec<(CVar, Ty)>,
    ctx: &RefcountCtx,
) -> Vec<(CVar, Ty)> {
    let live = live_set(func, args, ctx);
    owned.into_iter().filter(|(v, _)| !live.contains(v)).collect()
}

fn wrap_releases(to_release: Vec<(CVar, Ty)>, inner: CExpr, ctx: &RefcountCtx) -> CExpr {
    let mut result = inner;
    for (var, ty) in to_release.into_iter().rev() {
        let rvar = ctx.fresh.var();
        result = CExpr::LetPrim {
            var: rvar,
            ty: unit_ty(),
            op: PrimOp::Release(ty),
            args: vec![CVal::Var(var)],
            cont: Box::new(result),
        };
    }
    result
}
