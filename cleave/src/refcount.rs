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
/// Whether `ty` is itself a bare `#[mlir_type(tensor)]`-tagged type (like
/// `Tensor<T, Dims...>`) — deliberately *not* covered by `is_refcounted`
/// (its own doc comment excludes it on purpose, matching `mlir_lower.rs::
/// lower_field_access`'s own "no `!llvm.struct` storage at all" native-
/// shape handling), yet a `Tensor` value genuinely *is* heap-backed
/// (`cleave_alloc_rc`, at bufferization) and does need a real `Retain`/
/// `Release` when it's ever independently, directly owned — matching
/// `mlir_lower.rs::collect_light_leaves`'s own dedicated "a Tensor field is
/// always its own leaf" rule, the actual authority for this, reused here
/// directly for the one case that rule doesn't itself reach: a *bare*
/// Tensor value, never wrapped in any struct at all (`Scale::scale`'s own
/// real return shape — see `rewrite_body`'s own `Fix` arm, the transferred-
/// argument seeding this exists for).
fn is_bare_tensor_ty(ty: &Ty, mlir_types: &HashMap<String, String>) -> bool {
    let name = match ty {
        Ty::Con(n) => n.as_str(),
        Ty::App(n, _) => n.as_str(),
        _ => return false,
    };
    mlir_types.get(name).map(String::as_str) == Some("tensor")
}

pub(crate) fn is_refcounted(
    ty: &Ty,
    struct_schemas: &HashMap<String, crate::cps::StructSchema>,
    mlir_types: &HashMap<String, String>,
    constructed: &HashSet<String>,
    field_mutated: &HashSet<String>,
    extern_boundary: &HashSet<String>,
) -> bool {
    let (name, type_args): (&String, &[Ty]) = match ty {
        Ty::Con(name) => (name, &[]),
        Ty::App(name, args) => (name, args),
        _ => return false,
    };
    struct_schemas.contains_key(name)
        && !matches!(
            mlir_types.get(name).map(String::as_str),
            Some("tensor") | Some("vector")
        )
        && constructed.contains(name)
        // A fourth exclusion (`mlir_lower.rs::is_light_struct`'s own doc
        // comment, `doc/backlog.md`'s own struct-allocation-strategy
        // entry): a "light" struct is a bare `!llvm.struct<(...)>` SSA
        // value, never a `cleave_alloc_rc`-backed pointer -- no refcount
        // header exists for any `Retain`/`Release` call to act on, same
        // reasoning `RawBuf`'s own exclusion above already establishes for
        // a different reason (no construction site vs. no heap identity at
        // all).
        && !crate::mlir_lower::is_light_struct(
            name,
            type_args,
            struct_schemas,
            mlir_types,
            field_mutated,
            extern_boundary,
            constructed,
        )
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

/// Every struct name that's ever the target of a real `PrimOp::FieldStore`
/// (`s.field = v;`, a direct in-place field mutation) anywhere in `program`
/// — `mlir_lower.rs::is_light_struct`'s own doc comment has the full
/// reasoning for why this disqualifies a struct from the "light" (bare
/// `!llvm.struct` SSA value) representation: `lower_field_store`'s own GEP-
/// based mutation assumes a stable address to write through, which a light
/// struct — an immutable-once-constructed aggregate value, no identity of
/// its own — structurally doesn't have. Mirrors `collect_constructed_
/// struct_names`'s own identical walk shape exactly, just watching for a
/// different `PrimOp` variant. Deliberately *not* transitive the way
/// `DynArray`'s own exclusion is (`contains_dynarray_transitively`): a
/// struct merely *containing*, as one of its own fields, some other struct
/// that happens to be field-mutated *elsewhere*, on its own separate
/// binding, is completely unaffected — only the field-mutated struct's own
/// name needs excluding, not everything that ever references it.
pub(crate) fn collect_field_mutated_struct_names(program: &CpsProgram) -> HashSet<String> {
    let mut names = HashSet::new();
    for f in &program.funcs {
        collect_field_mutated_in(&f.def.body, &mut names);
    }
    names
}

fn collect_field_mutated_in(expr: &CExpr, names: &mut HashSet<String>) {
    match expr {
        CExpr::LetPrim { op, cont, .. } => {
            if let PrimOp::FieldStore { struct_ty, .. } = op {
                let name = match struct_ty {
                    Ty::Con(name) | Ty::App(name, _) => name.clone(),
                    _ => unreachable!("MLIR lowering: `FieldStore`'s own `struct_ty` is always a declared struct type"),
                };
                names.insert(name);
            }
            collect_field_mutated_in(cont, names);
        }
        CExpr::App { .. } => {}
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_field_mutated_in(then_branch, names);
            collect_field_mutated_in(else_branch, names);
        }
        CExpr::Fix { defs, body } => {
            for d in defs {
                collect_field_mutated_in(&d.body, names);
            }
            collect_field_mutated_in(body, names);
        }
    }
}

/// Every struct name that ever crosses an `extern fn` boundary — named as
/// its own `LetPrim`'s declared result type, or as one of `PrimOp::Extern`'s
/// own `param_types`, anywhere in `program`. `mlir_lower.rs::is_light_
/// struct`'s own doc comment has the full reasoning: a real, separately-
/// compiled C-ABI symbol (`cleave-rt`) is written assuming cleave's fixed,
/// uniform pointer-shaped struct representation, regardless of the struct's
/// own field shape — flattening such a struct to a bare `!llvm.struct<
/// (...)>` SSA value would silently change that extern call's own declared
/// MLIR signature out from under the real native function on the other
/// side, which still only ever takes/returns a plain pointer. A generic
/// algebra impl backed by an extern (`RawBuffer<S: HeapStruct>`'s own
/// `_ptr`-suffixed impl, `dynarray.cleave`) is exactly this shape once
/// monomorphized to a concrete struct `S` — found by direct testing (a real
/// `'func.call' op operand type mismatch` MLIR verification failure on
/// `DynArray<Point>`, not a hypothetical concern). Not transitive, same
/// reasoning as `collect_field_mutated_struct_names`'s own doc comment —
/// only the struct actually named at the boundary itself is excluded, not
/// everything that references it.
pub(crate) fn collect_extern_boundary_struct_names(program: &CpsProgram) -> HashSet<String> {
    let mut names = HashSet::new();
    for f in &program.funcs {
        collect_extern_boundary_in(&f.def.body, &mut names);
    }
    names
}

fn note_struct_boundary_ty(ty: &Ty, names: &mut HashSet<String>) {
    if let Ty::Con(name) | Ty::App(name, _) = ty {
        names.insert(name.clone());
    }
}

fn collect_extern_boundary_in(expr: &CExpr, names: &mut HashSet<String>) {
    match expr {
        CExpr::LetPrim {
            op, ty, cont, ..
        } => {
            if let PrimOp::Extern { param_types, .. } = op {
                note_struct_boundary_ty(ty, names);
                for pt in param_types {
                    note_struct_boundary_ty(pt, names);
                }
            }
            collect_extern_boundary_in(cont, names);
        }
        CExpr::App { .. } => {}
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_extern_boundary_in(then_branch, names);
            collect_extern_boundary_in(else_branch, names);
        }
        CExpr::Fix { defs, body } => {
            for d in defs {
                collect_extern_boundary_in(&d.body, names);
            }
            collect_extern_boundary_in(body, names);
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

/// A `CVar`'s own defining shape, tracked only for the two forms
/// `param_leaf_key` needs to see *through* — a plain `field.F(base)` read
/// (transparent: the result denotes exactly whatever `base`'s own `F` field
/// already denotes, no new value), or a `struct.Name[f1,...](a1,...)`
/// construction (also transparent, *per field* — reading `.f_i` back off
/// this exact construction denotes exactly `a_i` again, nothing copied).
/// Anything else (a real call's own result, an arithmetic `PrimOp`, ...) is
/// a genuine fresh value and gets no entry at all.
enum ValueDef {
    Field(CVar, String),
    StructCtor(HashMap<String, CVar>),
}

/// Every `CVar` this function's own body defines via `Field`/`Struct`,
/// resolved to its own `ValueDef` — see `param_leaf_key`'s own doc comment
/// for what this is *for*. A single forward walk suffices (CPS is SSA — a
/// `CVar` is bound at most once, always *before* any later reference to it,
/// so nothing here needs a fixpoint).
fn collect_value_defs(top: &CTopLevelFn) -> HashMap<CVar, ValueDef> {
    let mut defs = HashMap::new();
    walk_value_defs(&top.def.body, &mut defs);
    defs
}

fn walk_value_defs(expr: &CExpr, defs: &mut HashMap<CVar, ValueDef>) {
    match expr {
        CExpr::LetPrim {
            var, op, args, cont, ..
        } => {
            match op {
                PrimOp::Field { field, .. } => {
                    if let [CVal::Var(base)] = args.as_slice() {
                        defs.insert(*var, ValueDef::Field(*base, field.clone()));
                    }
                }
                PrimOp::Struct(_name, field_names) => {
                    let fields: HashMap<String, CVar> = field_names
                        .iter()
                        .zip(args.iter())
                        .filter_map(|(f, a)| match a {
                            CVal::Var(v) => Some((f.clone(), *v)),
                            _ => None,
                        })
                        .collect();
                    defs.insert(*var, ValueDef::StructCtor(fields));
                }
                _ => {}
            }
            walk_value_defs(cont, defs);
        }
        CExpr::App { .. } => {}
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            walk_value_defs(then_branch, defs);
            walk_value_defs(else_branch, defs);
        }
        CExpr::Fix {
            defs: fdefs, body, ..
        } => {
            for d in fdefs {
                walk_value_defs(&d.body, defs);
            }
            walk_value_defs(body, defs);
        }
    }
}

/// What `v`'s own value, fully resolved through any number of transparent
/// `Field`/`StructCtor` hops (`ValueDef`'s own doc comment), ultimately
/// turns out to be: still exactly one of this function's own parameters
/// (`Param` — propagated through *every* further `Field` hop on top of it
/// too, since a field read off an unresolved parameter is just as much
/// "the same value the caller already holds" as the parameter itself —
/// there's no `StructCtor` entry for a parameter to peel a field off *of*,
/// unlike a locally-constructed value), a locally-constructed struct whose
/// own field->argument map is known (`Struct`, letting a *further* `Field`
/// hop on top resolve transparently too — `param_leaf_key`'s own real
/// need: `v2265.l1.w` must resolve exactly as far as `v2265.l1` alone
/// (itself `Field`, resolving to `v2252`, itself a fresh `StructCtor`)
/// already does, peeling `.w` off *that* struct's own field map, not off
/// `v2265.l1`'s own (nonexistent) one), or genuinely `Opaque` (a real call
/// result, arithmetic, or any other `PrimOp` — a real fresh value, full
/// stop).
enum Resolved<'a> {
    /// Still exactly parameter `.0`'s own value — `.1` is the sequence of
    /// field names taken *from that parameter itself* to reach here (empty
    /// for the parameter's own bare `CVar`), not from wherever the walk
    /// happened to start — the whole point: `v2431` (`v2265.l1.w`, one
    /// re-derivation) and `v2250` (`v749.l1.w` via a completely different,
    /// earlier chain) must resolve to the *identical* `(v749, ["l1","w"])`
    /// key despite being different `CVar`s, different starting points, and
    /// different numbers of hops — `wrap_releases`'s own dedup (below)
    /// depends on that.
    Param(CVar, Vec<String>),
    Struct(&'a HashMap<String, CVar>),
    Opaque,
}

fn resolve<'a>(v: CVar, params: &HashSet<CVar>, defs: &'a HashMap<CVar, ValueDef>) -> Resolved<'a> {
    if params.contains(&v) {
        return Resolved::Param(v, Vec::new());
    }
    match defs.get(&v) {
        Some(ValueDef::StructCtor(fields)) => Resolved::Struct(fields),
        Some(ValueDef::Field(base, field)) => match resolve(*base, params, defs) {
            Resolved::Struct(fields) => match fields.get(field) {
                Some(arg) => resolve(*arg, params, defs),
                None => Resolved::Opaque,
            },
            Resolved::Param(p, mut path) => {
                path.push(field.clone());
                Resolved::Param(p, path)
            }
            Resolved::Opaque => Resolved::Opaque,
        },
        None => Resolved::Opaque,
    }
}

/// `resolve`, continued past `var`'s own resolution through `steps` (a
/// `LightLeafPath`'s own field-name path, `wrap_releases`'s real caller) —
/// the shared engine behind both `wrap_releases`'s heavy-`is_rc` branch
/// (`steps` empty, resolving `var` alone) and its light-struct-leaf branch
/// (`steps` non-empty).
fn resolve_leaf<'a>(
    var: CVar,
    steps: &[(Ty, String)],
    params: &HashSet<CVar>,
    defs: &'a HashMap<CVar, ValueDef>,
) -> Resolved<'a> {
    let mut current = resolve(var, params, defs);
    for (_, field) in steps {
        current = match current {
            Resolved::Struct(fields) => match fields.get(field) {
                Some(arg) => resolve(*arg, params, defs),
                None => Resolved::Opaque,
            },
            Resolved::Param(p, mut path) => {
                path.push(field.clone());
                Resolved::Param(p, path)
            }
            Resolved::Opaque => Resolved::Opaque,
        };
    }
    current
}

/// Whether `v`'s own value is *exactly* (never a copy of) something this
/// function received as one of its own borrowed formal parameters —
/// resolved transitively through any number of `Field`/`Struct`
/// reconstructions in between (`ValueDef`'s own doc comment: both are
/// transparent, denote the identical underlying value, never a new one).
///
/// **What this exists to fix, found by direct testing against the real
/// `examples/mnist-interop` kernel once the pool allocator (`cleave-rt`'s
/// own size-class free-list cache) started reusing freed blocks
/// immediately instead of relying on the OS heap's own lazy reuse — a real
/// `STATUS_ACCESS_VIOLATION`, root-caused precisely (not guessed) by
/// tracing `Optimizer::step<Sgd, Network, NetworkState<...>>`'s own
/// `--dump-cps-optimized` body variable by variable**: `Sgd`'s own `state`
/// genuinely never changes (a stateless optimizer — `Optimizer::step` for
/// `Sgd` just re-wraps `state`'s own existing leaves, unchanged, into a
/// fresh `NetworkState`); this function's own true return value therefore
/// embeds tensors that are *also* still reachable through its own `state`
/// parameter. Before this fix, this module's own generic "release whatever
/// this scope still owns that the return call's own *literal* arguments
/// don't cover" step (`releases_for_app`, this function's own caller)
/// couldn't tell that apart from the ordinary case (`net`'s own leaves,
/// genuinely freshly computed by the very same function) — it released
/// *every* such leaf's own locally-tracked copy unconditionally, correct
/// for the fresh case (a completely independent object, its own count
/// starting fresh) but one release too many for the identity-preserving
/// case: the loop's own next iteration still reaches the exact same tensor
/// through its own newly-returned `state`, whose count that extra release
/// already brought to zero — a real, silent premature free, invisible
/// until *something else* later reused the same freed block and the
/// dangling reference read or wrote through it.
///
/// Deliberately narrow — only ever consulted at a function's own **true**
/// return (`rewrite_body`'s own `CExpr::App` arm, guarded on `func ==
/// k_ret`), never at an ordinary call/loop tail-call, where the existing
/// `live_set`-based protection is already exactly right. A leaf that
/// *isn't* provably param-traced (any real computation anywhere in its own
/// derivation chain) resolves to `None` here, exactly like today —
/// strictly additive, never loosens an existing, already-correct release.
///
/// **Returns the dedup key itself (`Some((param, field_path))`), not just a
/// bool — found necessary, not a nicety, by a second real bug this fix's
/// own first version introduced**: `owned`'s own pre-existing redundancy
/// (a light struct nested inside another gets tracked as *two* separate
/// `to_release` entries — its own, and again transitively through the
/// outer one — true for `net`'s leaves too, symmetric there since *both*
/// copies get an equally redundant retain) means a single param-traced leaf
/// can appear here more than once in the very same `wrap_releases` call.
/// Skipping *every* occurrence (this fix's own first version) leaves the
/// matching retains uncompensated — a real leak (`v749.l1.w`'s own count
/// growing by one every single training-loop iteration, unbounded).
/// Skipping only the *first* occurrence per key (`wrap_releases`'s own
/// `skip_once` set) restores exactly the same "one retain answered by one
/// release" balance the ordinary (non-param-traced) case already has,
/// letting the *other*, redundant occurrence release normally, same as
/// before this fix existed at all.
fn param_leaf_key(
    var: CVar,
    steps: &[(Ty, String)],
    params: &HashSet<CVar>,
    defs: &HashMap<CVar, ValueDef>,
) -> Option<(CVar, Vec<String>)> {
    match resolve_leaf(var, steps, params, defs) {
        Resolved::Param(p, path) => Some((p, path)),
        _ => None,
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
    /// See `is_refcounted`'s own doc comment — passed through to `is_light_
    /// struct` as one of its own disqualifiers.
    field_mutated_structs: &'a HashSet<String>,
    /// See `is_refcounted`'s own doc comment — passed through to `is_light_
    /// struct` as its third disqualifier alongside `constructed_structs`/
    /// `field_mutated_structs`.
    extern_boundary_structs: &'a HashSet<String>,
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
    /// This one top-level function's own formal parameters — `param_leaf_
    /// key`'s own base case. Per-function (not whole-program), matching
    /// `RefcountCtx` itself being rebuilt fresh for each `top` in `insert_
    /// refcounting`'s own loop.
    params: &'a HashSet<CVar>,
    /// `param_leaf_key`'s own `Field`/`StructCtor` lookup table for this
    /// one function's own body — see that function's own doc comment.
    value_defs: &'a HashMap<CVar, ValueDef>,
}

impl RefcountCtx<'_> {
    fn is_rc(&self, ty: &Ty) -> bool {
        is_refcounted(
            ty,
            self.struct_schemas,
            self.mlir_types,
            self.constructed_structs,
            self.field_mutated_structs,
            self.extern_boundary_structs,
        )
    }

    /// Every genuinely-refcounted field reachable from `ty`'s own top
    /// level, transitively through any further light fields — empty for
    /// anything that isn't itself a light struct (`mlir_lower::light_
    /// struct_release_leaves` checks that first). A light struct's own
    /// binding is never `is_rc` (correctly — it has no heap identity of
    /// its own to release), so nothing else in this module would ever
    /// visit its fields on its behalf; this is what lets `rewrite_body`
    /// seed a light-but-leaf-bearing value into `owned` anyway, and
    /// `wrap_releases` emit a real `Release` for each of its own leaves
    /// (via a `PrimOp::Field` chain) at its own true last-use point,
    /// instead of the single flat `Release` a heavy struct's own binding
    /// gets — closing the exact leak `mlir_lower.rs::is_light_field_ty`'s
    /// own doc comment describes (a struct like `Network` holding real
    /// `Dense` pointers, retained on embedding but never released, since
    /// nothing tracked its own light container at all).
    fn light_release_leaves(&self, ty: &Ty) -> Vec<crate::mlir_lower::LightLeafPath> {
        let (name, type_args): (&str, &[Ty]) = match ty {
            Ty::Con(name) => (name.as_str(), &[]),
            Ty::App(name, args) => (name.as_str(), args.as_slice()),
            _ => return Vec::new(),
        };
        crate::mlir_lower::light_struct_release_leaves(
            name,
            type_args,
            self.struct_schemas,
            self.mlir_types,
            self.field_mutated_structs,
            self.extern_boundary_structs,
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
    let field_mutated_structs = collect_field_mutated_struct_names(&program);
    let extern_boundary_structs = collect_extern_boundary_struct_names(&program);
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
            let params: HashSet<CVar> = top.def.params.iter().copied().collect();
            let value_defs = collect_value_defs(&top);
            let ctx = RefcountCtx {
                struct_schemas,
                mlir_types,
                constructed_structs: &constructed_structs,
                field_mutated_structs: &field_mutated_structs,
                extern_boundary_structs: &extern_boundary_structs,
                var_types: &var_types,
                owned_origin: &owned_origin,
                local_free_vars: &local_free_vars,
                local_claim_vars: &local_claim_vars,
                fresh: &fresh,
                params: &params,
                value_defs: &value_defs,
            };
            insert_refcounting_fn(top, &ctx)
        })
        .collect();
    // Tier 1 of `doc/backlog.md`'s own struct-allocation-strategy entry —
    // a retain/release pair-cancellation optimization, strictly additive
    // on top of the naive, always-correct insertion above (never changes
    // *what* is correct, only elides calls proven redundant) — see
    // `rc_opt`'s own module doc comment for the full design. Applied here,
    // inside the one true entry point, rather than at each of this
    // function's own three call sites, so every one of them benefits
    // automatically (`--dump-cps-optimized`, `--run`, and the real AOT
    // pipeline) without needing to remember to call it separately.
    crate::rc_opt::eliminate_redundant_retain_release(CpsProgram { funcs })
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
        no_inline: top.no_inline,
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
            // A light struct with at least one genuinely-refcounted field
            // reachable through it (`light_release_leaves`) is seeded here
            // too, alongside an ordinary heavy (`is_rc`) value — its own
            // binding has no heap identity of its own to release, but its
            // leaves do, and `wrap_releases` below is what actually knows
            // the difference (a real `Release` on the pointer for the
            // heavy case, a `PrimOp::Field` chain ending in `Release` for
            // each leaf otherwise).
            if (matches!(&op, PrimOp::Struct(..)) || field_read_owned)
                && (ctx.is_rc(&ty) || !ctx.light_release_leaves(&ty).is_empty())
            {
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
            // Embedding an *existing* light-with-leaves value (`Network`,
            // once it has genuinely-refcounted fields of its own) into a
            // fresh container is the identical hazard `retain_targets`
            // already exists to protect against for an ordinary heavy
            // value — except there's no single pointer of the light
            // value's own to retain; each of *its* own leaves needs
            // retaining individually instead. Missing this let a freshly
            // built `(Network, NetworkState)` tuple (`Optimizer::step`'s
            // own real return shape) embed a `Network` whose own `Dense`
            // leaves were never protected from that same `Network`
            // binding's own later release — a real `STATUS_HEAP_
            // CORRUPTION`, found directly against `examples/digits-
            // interop`'s own real training run, not hypothetical.
            let mut retains: Vec<(CVal, Ty)> = Vec::new();
            let mut light_leaf_retains: Vec<(CVar, crate::mlir_lower::LightLeafPath)> = Vec::new();
            for target in retain_targets {
                if let CVal::Var(cv) = &target {
                    if let Some(rty) = ctx.var_types.get(cv) {
                        if ctx.is_rc(rty) {
                            retains.push((target, rty.clone()));
                        } else {
                            for leaf in ctx.light_release_leaves(rty) {
                                light_leaf_retains.push((*cv, leaf));
                            }
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
            //
            // A *light* `Network` (once it has genuinely-refcounted leaves
            // of its own, `light_release_leaves`) hits the identical
            // aliasing hazard one level down: extracting it out of a
            // container copies its own field bytes — including its own
            // `Dense` pointers — so the container's own eventual release
            // (or cascade) and this read result's own eventual leaf
            // releases would otherwise decrement the exact same `Dense`
            // refcounts, once each, unless *this* copy's own leaves are
            // independently retained too. There's no single pointer to
            // retain for the light value itself (nothing to increment) —
            // instead, retain each of its own leaves directly, mirroring
            // exactly how `wrap_releases` below releases them.
            enum FieldReadProtect {
                Whole(Ty),
                LightLeaves(Vec<crate::mlir_lower::LightLeafPath>),
            }
            let field_read_retain: Option<FieldReadProtect> = if let PrimOp::Field { .. } = &op {
                if ctx.owned_origin.get(&var).copied().unwrap_or(false) {
                    if ctx.is_rc(&ty) {
                        Some(FieldReadProtect::Whole(ty.clone()))
                    } else {
                        let leaves = ctx.light_release_leaves(&ty);
                        if leaves.is_empty() {
                            None
                        } else {
                            Some(FieldReadProtect::LightLeaves(leaves))
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            };

            let new_cont = rewrite_body(*cont, owned, k_ret, ctx);
            let new_cont = match field_read_retain {
                Some(FieldReadProtect::Whole(rty)) => {
                    let rvar = ctx.fresh.var();
                    CExpr::LetPrim {
                        var: rvar,
                        ty: unit_ty(),
                        op: PrimOp::Retain(rty),
                        args: vec![CVal::Var(var)],
                        cont: Box::new(new_cont),
                    }
                }
                Some(FieldReadProtect::LightLeaves(leaves)) => {
                    wrap_light_leaves(ctx, var, &leaves, PrimOp::Retain, new_cont)
                }
                None => new_cont,
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
            for (base, leaf) in light_leaf_retains {
                result = build_leaf_chain(
                    ctx,
                    CVal::Var(base),
                    &leaf.steps,
                    &leaf.leaf_ty,
                    PrimOp::Retain,
                    result,
                );
            }
            result
        }
        CExpr::App { func, args } => {
            let to_release = releases_for_app(&func, &args, owned, ctx);
            // The function's own *true* return (as opposed to a real
            // call's own dispatch or a loop's own back-edge, both still
            // handled exactly as before) — `param_leaf_key`'s own doc
            // comment has the full story: a leaf embedded in this exact
            // return value that's provably still the identical object one
            // of this function's own parameters already denotes was never
            // this function's own to give away a second time here (its own
            // *caller*, symmetrically, already assumes a real call's
            // result is always a genuinely fresh, independently-owned
            // reference — `collect_var_info`'s own doc comment on `PrimOp
            // ::Field`'s ownership rule says so explicitly).
            // At this function's own *true* return (as opposed to a real
            // call's own dispatch or a loop's own back-edge, both still
            // released exactly as before) — `param_leaf_key`'s own
            // doc comment has the full story: a leaf still reachable
            // through one of this function's own parameters was never
            // this function's own to give away a second time here (its
            // own *caller*, symmetrically, already assumes a real call's
            // result is always a genuinely fresh, independently-owned
            // reference — `collect_var_info`'s own doc comment on `PrimOp
            // ::Field`'s ownership rule says so explicitly).
            let at_true_return = matches!(&func, CVal::Var(v) if *v == k_ret);
            wrap_releases(to_release, CExpr::App { func, args }, ctx, at_true_return)
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
                    // A light struct with its own genuinely-refcounted
                    // leaves needs seeding here exactly like an ordinary
                    // heavy (`is_rc`) value does — its own binding still
                    // has no heap identity to release directly, but a
                    // loop's own carried `net`/`state` (this whole
                    // mechanism's own motivating case) is *exactly* this
                    // shape: never seeding it here would silently leak its
                    // own leaves every single iteration, the identical
                    // "no scope ever gets a turn" gap this function's own
                    // doc comment already describes for a different case.
                    let needs_seed =
                        |ty: &Ty| ctx.is_rc(ty) || !ctx.light_release_leaves(ty).is_empty();
                    if let Some(fv) = ctx.local_claim_vars.get(&def.name) {
                        for v in fv {
                            if let Some(ty) = ctx.var_types.get(v) {
                                if needs_seed(ty) && is_owned(v) && seen.insert(*v) {
                                    seed.push((*v, ty.clone()));
                                }
                            }
                        }
                    }
                    for p in &def.params {
                        if let Some(ty) = ctx.var_types.get(p) {
                            if needs_seed(ty) && is_owned(p) && seen.insert(*p) {
                                seed.push((*p, ty.clone()));
                            }
                        }
                    }
                    // A *real* call's own resumption, `def.carried_types
                    // .is_none()` (a loop's own entry/back-edge is excluded
                    // below instead, `carried_types.is_some()`), consuming
                    // an *owned* value as a literal argument that's never
                    // referenced again anywhere leaves that value with no
                    // scope left to release it at all: `live_set`'s own
                    // "any literal argument is live" rule (needed so the
                    // callee itself still sees a valid pointer while it
                    // runs) only protects the *call's own duration*, and an
                    // ordinary top-level function (`Ring::sub`, `Scale::
                    // scale`, ...) never releases its own borrowed
                    // parameters (this module's own documented convention)
                    // — a real, confirmed, unbounded per-training-step leak
                    // (`Optimizer::step`'s own Sgd update: `Scale::scale
                    // (lr, grad)`'s own result, fed straight into `Ring::
                    // sub` and never touched again). **A fix for this was
                    // attempted and reverted** — seeding the argument into
                    // `seed` here, gated on it not appearing in `def`'s own
                    // `local_free_vars` (protecting a genuinely-still-
                    // needed one), still produced a real, reproducible
                    // `STATUS_ACCESS_VIOLATION` (`cleave-rt`'s own temporary
                    // `CLEAVE_DEBUG_POOL`/`RtlCaptureStackBackTrace`
                    // instrumentation traced it to `cleave_release` being
                    // called on a header that was never a valid allocation
                    // at all — `data_size=0`, refcount already negative on
                    // the very first tracked event for that address) —
                    // confirmed, by disabling the fix outright, to be the
                    // fix's own doing, not a pre-existing issue it merely
                    // exposed; excluding the enclosing function's own body
                    // when it's `region_analysis::find_region_local_
                    // functions` (arena-backed, bulk-reclaimed regardless)
                    // did *not* resolve it either, so the real remaining
                    // flaw in this approach is still unidentified. Left
                    // here as a precise, validated write-up rather than a
                    // silently-reintroduced bug — `doc/backlog.md`'s own
                    // struct-allocation-strategy entry has the same account
                    // for anyone picking this back up.
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
                        // A real call's own resumption takes the call's
                        // *return value* as its sole parameter, and
                        // `collect_var_info` already seeded that param above
                        // (`owned_origin.insert(*p, true)` for the real-call
                        // shape). When a transferred *literal argument*
                        // shares that parameter's type, the callee may
                        // forward that very allocation straight back out as
                        // its result rather than build a fresh one —
                        // `stdlib/io`'s own `Print::print`/`println` are
                        // literally `fn(x) -> x`. Seeding the argument *as
                        // well as* the return-value param then releases one
                        // allocation twice: a real double-free, found via
                        // the size-class pool as two back-to-back `cleave_
                        // release` calls on the same 12-byte tuple from
                        // `println(("Epoch=", epoch))` in `mnist-interop`'s
                        // own training loop (`FREE_LISTS` next-pointer
                        // corrupted, misaligned-pointer crash one pop
                        // later). The return-value param already owns that
                        // resource, so the aliasing argument is skipped
                        // here. A callee that instead genuinely consumes
                        // such an argument and returns a *fresh* value of
                        // the same type would now leak it — accepted:
                        // strictly better than the use-after-free, and that
                        // shape (take `T` by value, ignore it, return a new
                        // `T`) is not one this stdlib actually has.
                        let resumption_ret_tys: Vec<&Ty> = if is_loop {
                            Vec::new()
                        } else {
                            def.params
                                .iter()
                                .filter_map(|p| ctx.var_types.get(p))
                                .collect()
                        };
                        for (v, ty) in &transferred {
                            let is_entry_arg = entry_arg_vars.contains(v);
                            // Only an argument this call is the *last* use of
                            // can be the one the callee hands straight back:
                            // if the resumption still needs it (a free
                            // variable of `def` -- e.g. `net`, passed to
                            // `net_grad` here and then again to `Optimizer::
                            // step` in the same resumption), the type match
                            // is a coincidence (`net_grad: Network ->
                            // Network` returns a *fresh* gradient), and it
                            // must still be seeded.
                            let arg_needed_later = ctx
                                .local_free_vars
                                .get(&def.name)
                                .is_some_and(|fv| fv.contains(v));
                            let aliases_ret_param = !is_loop
                                && is_entry_arg
                                && !arg_needed_later
                                && resumption_ret_tys.iter().any(|rt| *rt == ty);
                            if (!is_loop || !is_entry_arg)
                                && !aliases_ret_param
                                && seen.insert(*v)
                            {
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
            // Never the function's own true return — that's always the
            // *bare* `CExpr::App` arm above (a `Fix` wrapping an `App` as
            // its own trailing `body` is either a real call's own
            // resumption or a loop's own back-edge/entry, both still
            // released exactly as before `param_leaf_key` existed).
            wrap_releases(to_release, fix_node, ctx, false)
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

/// For each `(var, ty)` no longer live: a plain `Release(ty)` for an
/// ordinary heavy (`is_rc`) value, exactly as before — or, for a light
/// struct with genuinely-refcounted leaves of its own (`light_release_
/// leaves`), a `PrimOp::Field` chain down to each leaf followed by a real
/// `Release` on it, since there's no single pointer of `var`'s own to
/// release at all. See `mlir_lower.rs::is_light_field_ty`'s own doc
/// comment for why this exists: without it, embedding a heavy field
/// (`Dense`) into a light container (`Network`) retains it unconditionally
/// (`retain_targets` above, keyed on the *field's* own type) but never
/// released it anywhere, since a light value's own binding was never
/// tracked at all — a real, per-construction leak, not hypothetical.
/// `at_true_return`: whether `inner` is this function's own real `k_ret`
/// dispatch (as opposed to a real call's own resumption or a loop's own
/// back-edge/entry) — see `param_leaf_key`'s own doc comment for why
/// that's the *only* place a light struct's own leaf can be safely skipped
/// here: everywhere else, `owned`/`to_release` (this function's own callers,
/// `releases_for_app`) already correctly means "this scope is genuinely
/// done with it", true release-target aliasing or not.
///
/// `skip_once` (`param_leaf_key`'s own doc comment has the full story) is
/// scoped to this one call — a fresh, empty set per `wrap_releases` call,
/// not threaded in from outside — since it exists only to de-duplicate
/// `to_release`'s own pre-existing redundancy *within* this exact release
/// point, never across two different ones.
fn wrap_releases(to_release: Vec<(CVar, Ty)>, inner: CExpr, ctx: &RefcountCtx, at_true_return: bool) -> CExpr {
    let mut result = inner;
    let mut skip_once: HashSet<(CVar, Vec<String>)> = HashSet::new();
    for (var, ty) in to_release.into_iter().rev() {
        // `is_bare_tensor_ty` alongside `is_rc` here — a bare `Tensor`
        // (never wrapped in any struct) needs the identical plain `Release`
        // this branch already emits, not the light-struct-leaf-chain one
        // below (`ctx.light_release_leaves(&ty)` is *always* empty for a
        // `Tensor` itself, `is_bare_tensor_ty`'s own doc comment) — without
        // this, a value seeded here by the transferred-argument fix
        // (`rewrite_body`'s own `Fix` arm) would silently route into the
        // light branch and get *zero* releases emitted for it at all.
        if ctx.is_rc(&ty) || is_bare_tensor_ty(&ty, ctx.mlir_types) {
            if at_true_return {
                if let Some(key) = param_leaf_key(var, &[], ctx.params, ctx.value_defs) {
                    // First redundant occurrence of this exact param-
                    // traced value: skip it, matching the one still-
                    // outstanding compensating retain (`param_leaf_key`'s
                    // own doc comment). Any *further* occurrence releases
                    // normally, exactly as before this fix existed.
                    if skip_once.insert(key) {
                        continue;
                    }
                }
            }
            let rvar = ctx.fresh.var();
            result = CExpr::LetPrim {
                var: rvar,
                ty: unit_ty(),
                op: PrimOp::Release(ty),
                args: vec![CVal::Var(var)],
                cont: Box::new(result),
            };
        } else {
            let mut leaves = ctx.light_release_leaves(&ty);
            if at_true_return {
                leaves.retain(|leaf| {
                    match param_leaf_key(var, &leaf.steps, ctx.params, ctx.value_defs) {
                        Some(key) => !skip_once.insert(key),
                        None => true,
                    }
                });
            }
            result = wrap_light_leaves(ctx, var, &leaves, PrimOp::Release, result);
        }
    }
    result
}

/// Builds a `PrimOp::Field` chain from `base` down to `leaf.leaf_ty`, then
/// wraps `inner` with `terminal(leaf_ty)` (a `Retain`/`Release` `PrimOp`,
/// passed as a bare tuple-variant constructor) applied to the leaf value —
/// shared by both the retain-on-read and release-on-scope-exit sides of a
/// light struct's own leaf tracking (`field_read_retain`/`wrap_releases`
/// above). `leaves` is walked in reverse so the resulting `CExpr` reads,
/// top to bottom, in the same order `leaves` was given.
fn wrap_light_leaves(
    ctx: &RefcountCtx,
    base: CVar,
    leaves: &[crate::mlir_lower::LightLeafPath],
    terminal: fn(Ty) -> PrimOp,
    inner: CExpr,
) -> CExpr {
    let mut result = inner;
    for leaf in leaves.iter().rev() {
        result = build_leaf_chain(ctx, CVal::Var(base), &leaf.steps, &leaf.leaf_ty, terminal, result);
    }
    result
}

fn build_leaf_chain(
    ctx: &RefcountCtx,
    base: CVal,
    steps: &[(Ty, String)],
    leaf_ty: &Ty,
    terminal: fn(Ty) -> PrimOp,
    inner: CExpr,
) -> CExpr {
    match steps {
        [] => {
            let rvar = ctx.fresh.var();
            CExpr::LetPrim {
                var: rvar,
                ty: unit_ty(),
                op: terminal(leaf_ty.clone()),
                args: vec![base],
                cont: Box::new(inner),
            }
        }
        [(struct_ty, field), rest @ ..] => {
            let next_var = ctx.fresh.var();
            let next_ty = rest
                .first()
                .map(|(t, _)| t.clone())
                .unwrap_or_else(|| leaf_ty.clone());
            let rest_chain = build_leaf_chain(ctx, CVal::Var(next_var), rest, leaf_ty, terminal, inner);
            CExpr::LetPrim {
                var: next_var,
                ty: next_ty,
                op: PrimOp::Field {
                    struct_ty: struct_ty.clone(),
                    field: field.clone(),
                },
                args: vec![base],
                cont: Box::new(rest_chain),
            }
        }
    }
}
