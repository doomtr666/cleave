//! Whole-program fixed point deciding, for every top-level function's own
//! *ordinary* parameter (never the trailing `k_ret` continuation — see
//! `CTopLevelFn::k_ret`'s own doc comment), whether calling that function
//! can ever result in **more than one live reference** to the value passed
//! in that position existing at once, anywhere the program reaches.
//!
//! This is the actual precondition `doc/plan-affine-ownership.md` ("piste
//! 1") needs: a value that is *never* aliased needs no refcount header at
//! all — a deterministic alloc, and a free at its own provable last use,
//! no runtime retain/release call either. A value that *can* be aliased
//! keeps today's mechanism, unchanged.
//!
//! ## Computed once, consulted in O(1) — never re-walked per query
//!
//! Same discipline as `region_analysis::analyze`: [`analyze`] runs exactly
//! once per compilation, over the whole program, and hands back an
//! [`AliasSummary`] — a flat lookup table. Every later consumer queries
//! [`AliasSummary::is_aliased`], a single `HashMap` lookup, never
//! re-walking any CPS tree.
//!
//! Internally this is itself split into two distinct passes, deliberately
//! kept apart so the *expensive* one only ever runs once:
//!
//! 1. **One real walk of the whole program** (`collect_facts`) — O(program
//!    size), touches every `CExpr` node exactly once. Extracts two small,
//!    already-summarized artifacts: `seed` (parameter positions *directly*
//!    proven aliased by a local, syntactic fact — see rule 1 below) and
//!    `edges` (a dependency graph: "if the callee's own position ends up
//!    aliased, so does mine").
//! 2. **A worklist propagation over that graph alone** (`propagate`) —
//!    never touches the CPS again, only `seed`/`edges`, the small
//!    artifacts step 1 already produced. This is where recursion and
//!    mutual recursion between top-level functions get resolved to a
//!    correct answer, by iterating to a fixed point — the same reason a
//!    loop's own back-edge no longer needs special-casing once liveness is
//!    computed this way (`doc/plan-affine-ownership.md` §2.1) — except
//!    here the cycle lives in the *call graph*, not inside one function's
//!    own control flow.
//!
//! ## The rules, and why they're sound
//!
//! **Every rule reduces to one question: does the same `CVar` show up as a
//! live operand at more than one "point of commitment"** — a *commitment*
//! being anything that hands a reference somewhere it can independently
//! outlive this exact point: a `Struct`/`Array` construction (builds new,
//! permanent storage), a `FieldStore`/`Store` (mutates *existing* storage
//! the same permanent way — `s.field = p`/`arr[i] = p`), or a real function
//! call whose own callee might do either of those with what it's handed.
//! An *ordinary* borrow — read a field off it, pass it to a call that
//! never stores it anywhere — is not a commitment; the value comes back
//! out of that use with exactly the same one owner it had going in.
//!
//! 1. **The same parameter appears more than once within one commitment's
//!    own argument list** (`Struct(a: p, b: p)` — two fields of the *same*
//!    fresh struct now hold the identical pointer) — this alone is
//!    already two independent references to the one allocation, no later
//!    use needed at all.
//! 2. **A parameter is embedded into a commitment, and is *also* still
//!    referenced afterward** (`let s = Struct(field: p); read(p);` — a
//!    plain occurs-check on the rest of the function from that point on,
//!    no liveness subtlety needed: this only asks "does `p` ever get used
//!    again", not "on which specific path"). Computed directly here, by a
//!    real walk of the CPS — **deliberately not** by consulting
//!    `refcount.rs`'s own `Retain` markers, even though they mark exactly
//!    this same hazard: those don't exist yet at the point in the pipeline
//!    this analysis is meant to run (`insert_refcounting` is what
//!    *decides* whether to emit a header at all, informed by this
//!    analysis — the dependency only ever goes one way).
//! 3. **Passed as a literal argument to another function**, at the exact
//!    position that function's own summary marks aliased — propagated
//!    transitively, resolved by the fixed point in step 2 (of the module's
//!    own two-pass split above, not this list).
//! 4. **Passed to an `extern fn`** (no body in `program.funcs` — no way to
//!    see what a C-ABI callee does with the pointer) is seeded aliased
//!    directly, unconditionally — the same "no evidence, assume the
//!    dangerous case" discipline already established in `refcount.rs`
//!    (`collect_identity_param_positions`'s own doc comment has the twin
//!    of this exact reasoning).
//! 4b. **Embedded into an `Array`/`ArrayRepeat` construction** is seeded
//!    aliased directly, unconditionally too — the same "no evidence, no
//!    assumption" reasoning as rule 4, for a different reason: a `Load`
//!    reading back out of that array uses a genuinely runtime index (`arr
//!    [i]`, `i` a loop variable, not a compile-time constant in general),
//!    so *which* originally-embedded element comes back out at any later
//!    load site is not statically knowable here — unlike a `Struct`'s own
//!    fields, only ever reached back out via a named, fully-visible
//!    `Field` projection this analysis already understands completely.
//!    Confirmed a real, not hypothetical, gap: `examples/convex_hull
//!    .cleave`'s `[Point; 8]` array literal, each `Point` embedded once and
//!    never referenced again *by that same name* (rule 2's own occurs-check
//!    is blind to this — the array is what survives, not the original
//!    binding), classified every one of the 8 as pool-eligible even though
//!    `points[current]` (a `Load` off that exact array, at a runtime index)
//!    is later pushed into a `DynArray<Point>` — a real, non-deterministic
//!    use-after-free once the pool block backing whichever element was
//!    loaded got reused for something else, `doc/backlog.md`'s own entry on
//!    this has the full failure signature and reproduction.
//! 5. **Returned unchanged (an identity-shaped function) does NOT, on its
//!    own, mark the parameter aliased.** A pure pass-through is not a
//!    commitment — it creates no *new* reference by itself. The aliasing
//!    hazard in that shape exists only if the *caller* also keeps using
//!    its own pre-call binding afterward, which is already exactly rule 2,
//!    evaluated at the *caller's* own call site (its own occurs-check
//!    already catches "used again after the call" the same way it catches
//!    "used again after a construction") — and is additionally already
//!    handled, independently, by `refcount.rs::collect_identity_param_
//!    positions` (a deliberately separate, narrower, already-landed fix
//!    for precisely locating *where* the resulting release goes, not
//!    *whether* a header is needed at all — that function's own doc
//!    comment). Marking this rule true from the callee's own side too
//!    would double-book the same fact from two directions.
//!
//! Field-level aliasing (does *one field* of a struct end up independently
//! referenced) is explicitly out of scope here, matching the original
//! Tier-1 plan's own scoping: this analysis answers "is the parameter's
//! own top-level pointer identity ever duplicated", at variable
//! granularity, not field granularity.

use crate::cps::{CExpr, CFunDef, CVal, CVar, CpsProgram, PrimOp};
use crate::infer::Ty;
use std::collections::{HashMap, HashSet};

/// The result of [`analyze`] — a flat, precomputed table. Every query is
/// O(1); nothing here ever re-walks a CPS tree.
pub struct AliasSummary {
    /// Function name -> set of its own ordinary parameter *indices*
    /// (0-based, `k_ret` excluded) that can be aliased.
    aliased: HashMap<String, HashSet<usize>>,
}

impl AliasSummary {
    /// Whether `fn_name`'s own parameter at `param_index` can ever be
    /// aliased when the function is called. `true` for any function this
    /// analysis has no information about (an `extern fn`, or a name not
    /// found at all) — the same "no evidence, assume the dangerous case"
    /// rule `collect_facts`'s own rule 4 applies during analysis, kept
    /// consistent at the query boundary too.
    pub fn is_aliased(&self, fn_name: &str, param_index: usize) -> bool {
        self.aliased
            .get(fn_name)
            .map(|set| set.contains(&param_index))
            .unwrap_or(true)
    }
}

/// A dependency edge: if `depends_on` (callee, its own parameter index)
/// ends up aliased, so must `dependent` (the caller, its own parameter
/// index that was passed at that position).
type Edge = ((String, usize), (String, usize));

/// The result of [`analyze_identity`] — whether a top-level function
/// provably hands one of its own parameters back **completely unchanged**
/// as its own return value, possibly through a chain of other functions
/// doing the identical thing.
///
/// Consumed by `refcount.rs::rewrite_body`'s own `Fix` arm to decide
/// whether a transferred literal argument might be the *exact same
/// allocation* the resumption's own parameter already denotes — the real,
/// confirmed hazard this exists to prevent (`println((..))` in a real
/// training loop: `println` calls `Print::print`, which is genuinely
/// `fn(x) -> x`, and hands its own result straight back unchanged —
/// `println` is therefore identity-shaped too, *transitively*, even
/// though nothing in its own body directly returns its own parameter by
/// name). Found by direct testing on `examples/mnist-interop`: an earlier
/// version of this check only recognized a *direct* return (`fn(x) -> x`
/// literally), which correctly protects `Print::print` itself but not
/// `println`'s own wrapping of it — a real `STATUS_HEAP_CORRUPTION`
/// (`CLEAVE_DEBUG_POOL`: "release on parked block", `data_size=12`,
/// exactly the `(&str, i32)` tuple `println(("Epoch=", epoch))` builds).
pub struct IdentitySummary {
    /// Function name -> set of its own ordinary parameter indices proven
    /// to be returned unchanged on at least one reachable path. Every
    /// function this program defines a body for gets an entry (possibly
    /// empty) — see [`AliasSummary`]'s own identical discipline and
    /// [`analyze_identity`]'s own doc comment for why the distinction
    /// from "no entry at all" (an `extern fn`) matters to this struct's
    /// own consumer.
    identity: HashMap<String, HashSet<usize>>,
}

impl IdentitySummary {
    /// `Some(true)`/`Some(false)` for a function this analysis has a real
    /// body for; `None` for one it doesn't (an `extern fn`) — deliberately
    /// *not* collapsed to a single bool default, unlike [`AliasSummary::
    /// is_aliased`]: the caller (`refcount.rs`) has its own, different
    /// fallback for the "no evidence" case (a coarser type-coincidence
    /// check, preserved there rather than duplicated here), and needs to
    /// tell that case apart from "checked, and it's genuinely not
    /// identity-shaped".
    pub fn returns_unchanged(&self, fn_name: &str, param_index: usize) -> Option<bool> {
        self.identity
            .get(fn_name)
            .map(|set| set.contains(&param_index))
    }
}

/// Runs the identity analysis — see [`IdentitySummary`]'s own doc comment
/// for what it computes and why, and the module's own top-level doc
/// comment for the two-pass design ([`tail_returns_var`] once per
/// parameter, [`propagate`] resolves the transitive chains) this shares
/// verbatim with [`analyze`].
///
/// One generalized forward trace per ordinary parameter, not two separate
/// pattern-matched rules the way this used to be split (a *direct* return
/// vs. a *single-hop* resumption-forwards-a-real-call shape): found, by a
/// real, deterministic `examples/complex.cleave --run` segfault (`doc/
/// backlog.md`'s own entry has the full story), that splitting it that way
/// left an entire class of shape undetected — a value threaded through an
/// `if`/`else`'s own local join continuation (`Display<Complex<T>>`'s own
/// sign-of-imaginary-part branch, `stdlib/display/display.cleave`) before
/// reaching either `k_ret` or a further real call never matched *either*
/// rule, since neither the old rule 1 (only a *literal* parameter reaching
/// `k_ret` directly) nor the old rule 2 (only a *single* real-call
/// resumption, `tail_returns_var` explicitly refusing to recurse any
/// deeper) ever looked *through* a plain local join point at all. A join
/// point is not a real function — it never needs its own entry in this
/// module's own `known_functions`/interprocedural fixed point — so tracing
/// through one is a purely local, single-function question, resolved once
/// here rather than deferred.
pub fn analyze_identity(program: &CpsProgram) -> IdentitySummary {
    let known_functions: HashSet<&str> = program
        .funcs
        .iter()
        .map(|f| f.def.name.as_str())
        .collect();

    let mut seed: HashSet<(String, usize)> = HashSet::new();
    let mut edges: Vec<Edge> = Vec::new();

    for f in &program.funcs {
        let Some((&k_ret, ordinary_params)) = f.def.params.split_last() else {
            continue;
        };
        let mut defs_by_name: HashMap<&str, &CFunDef> = HashMap::new();
        collect_local_defs_by_name(&f.def.body, &mut defs_by_name);
        for (i, &param) in ordinary_params.iter().enumerate() {
            let mut visiting = HashSet::new();
            if tail_returns_var(
                &f.def.body,
                &f.def.name,
                i,
                k_ret,
                param,
                &defs_by_name,
                &known_functions,
                &mut visiting,
                &mut edges,
            ) {
                seed.insert((f.def.name.clone(), i));
            }
        }
    }

    let resolved = propagate(seed, edges);

    let mut identity: HashMap<String, HashSet<usize>> = HashMap::new();
    for f in &program.funcs {
        identity.entry(f.def.name.clone()).or_default();
    }
    for (name, idx) in resolved {
        identity.entry(name).or_default().insert(idx);
    }
    IdentitySummary { identity }
}

/// Every local `Fix`-def anywhere in `expr` (any nesting depth), keyed by
/// its own label — local defs share one flat name space across their
/// whole enclosing top-level function (mutual recursion between them, a
/// loop tail-calling itself, is ordinary), so [`tail_returns_var`] needs
/// to look one up by name regardless of *where* under the function's body
/// it happens to be declared relative to whichever call site is tracing
/// through it. Mirrors [`collect_local_defs`]'s own identical walk (a
/// different module-internal purpose, `affine_struct_vars`'s own), kept
/// separate rather than shared since that one also seeds/edges as it
/// walks and this one only ever collects names.
fn collect_local_defs_by_name<'a>(expr: &'a CExpr, out: &mut HashMap<&'a str, &'a CFunDef>) {
    match expr {
        CExpr::LetPrim { cont, .. } => collect_local_defs_by_name(cont, out),
        CExpr::App { .. } => {}
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_local_defs_by_name(then_branch, out);
            collect_local_defs_by_name(else_branch, out);
        }
        CExpr::Fix { defs, body } => {
            for d in defs {
                out.insert(d.name.as_str(), d);
                collect_local_defs_by_name(&d.body, out);
            }
            collect_local_defs_by_name(body, out);
        }
    }
}

/// Whether `var`, starting from `expr` (`f_name`'s own body, or a local
/// def's body reached while tracing through one), ever flows *unchanged*
/// all the way to `f_name`'s own `k_ret` — the module's own doc comment on
/// [`analyze_identity`] has the full motivation for why this replaced the
/// old two-rule split. Three real cases where the trace can end:
///
/// 1. **Reaches `k_ret` directly, as the literal argument** — proven,
///    return `true` right here, no further work needed.
/// 2. **Tail-calls a local `Fix`-def** (a join point, a resumption, a
///    loop's own self-call, found in `defs_by_name`) — trace *through* it:
///    find which of *its own* parameter positions received `var` at this
///    call site, then re-ask the identical question about that parameter
///    inside that def's own body. Chains of any length and shape resolve
///    this way, one recursive call per hop, not by growing this function's
///    own case list.
/// 3. **Tail-calls a real, whole-program function** (found in
///    `known_functions`) — this is the interprocedural half, resolved by
///    [`propagate`]'s own fixed point exactly like rule 3 elsewhere in this
///    module: push an edge from `(f_name, param_idx)` (the parameter this
///    *entire* trace was launched to answer, threaded through unchanged
///    across every local hop above — not the local `var` at this specific
///    point, which belongs only to the current def) to `(callee, pos)`,
///    and answer `false` for *this* call directly — [`analyze_identity`]'s
///    own `seed`/`edges` split, plus [`propagate`], resolves it from there.
///
/// `defs_by_name`/`known_functions` distinguish cases 2 and 3 — a name
/// never appears in both, since a top-level function's own name and a
/// `Fix`-local label are drawn from disjoint namespaces (confirmed by
/// every other whole-program collector in this module already relying on
/// the same distinction).
///
/// `visiting`: labels already on the current recursion path — purely to
/// stay terminating on a genuine loop back-edge (a `loop$N` def tail-
/// calling itself). A revisited label conservatively answers "no
/// evidence" (`false`) for that one edge, the same posture every other
/// "can't decide" case in this module already takes — a loop's own real
/// exit is a separate, textually distinct tail call (its `if`'s other
/// branch) this same walk still reaches independently, never through the
/// revisited label itself.
///
/// `If`'s own two branches are combined with `||` (either proving it is
/// enough), matching this module's own established, deliberate bias
/// (`propagate`'s own doc comment, `is_committed_at`'s "over-classify as
/// shared rather than risk corruption" posture): `refcount.rs`'s own
/// consumer of this fact only ever uses `true` to *skip* an otherwise-real
/// release (protecting against a possible double-release), never to skip
/// a needed *retain* — the failure mode on a branch that doesn't actually
/// preserve identity is a conservative extra reference kept alive
/// (`refcount.rs`'s own existing release-tracking on that value's *other*
/// name still fires normally), never a use-after-free.
#[allow(clippy::too_many_arguments)]
fn tail_returns_var(
    expr: &CExpr,
    f_name: &str,
    param_idx: usize,
    k_ret: CVar,
    var: CVar,
    defs_by_name: &HashMap<&str, &CFunDef>,
    known_functions: &HashSet<&str>,
    visiting: &mut HashSet<String>,
    edges: &mut Vec<Edge>,
) -> bool {
    match expr {
        CExpr::LetPrim { cont, .. } => tail_returns_var(
            cont, f_name, param_idx, k_ret, var, defs_by_name, known_functions, visiting, edges,
        ),
        CExpr::App { func, args } => match func {
            CVal::Var(v) if *v == k_ret => {
                matches!(args.as_slice(), [CVal::Var(v)] if *v == var)
            }
            CVal::Label(callee) if known_functions.contains(callee.as_str()) => {
                if let Some(pos) = args.iter().position(|a| matches!(a, CVal::Var(v) if *v == var)) {
                    edges.push(((f_name.to_string(), param_idx), (callee.clone(), pos)));
                }
                // `var` might simply not be one of this real call's own
                // arguments at all (an unrelated comparison/computation
                // happening in between, e.g. `Ord::lt<i32>` inside
                // `Display::display<f64>`'s own formatting loop, found by
                // direct testing the first version of this fix still
                // missed) -- if so it survives completely untouched into
                // whatever this call's own trailing continuation is, once
                // the call itself returns. Every real call in this CPS
                // passes that continuation as one of its own `args`, as a
                // `CVal::Label` naming a local `Fix`-def (never anything
                // else `CVal::Label` is used for) -- trace on through it,
                // `var` unchanged, exactly like the local-callee arm below
                // already does when `var` isn't one of *its* own params
                // either.
                args.iter().any(|a| {
                    if let CVal::Label(cont) = a {
                        if let Some(def) = defs_by_name.get(cont.as_str()) {
                            if visiting.insert(cont.clone()) {
                                let result = tail_returns_var(
                                    &def.body, f_name, param_idx, k_ret, var, defs_by_name,
                                    known_functions, visiting, edges,
                                );
                                visiting.remove(cont.as_str());
                                return result;
                            }
                        }
                    }
                    false
                })
            }
            CVal::Label(callee) => {
                let Some(def) = defs_by_name.get(callee.as_str()) else {
                    return false;
                };
                // `var` continues into `def`'s own body either re-bound to
                // whichever of its own parameters received it *as an
                // argument* at this call site, or -- if it's simply never
                // passed at all -- completely unchanged: CPS `CVar` ids are
                // globally unique (no shadowing anywhere in this codebase's
                // own numbering scheme), so a local `Fix`-def's body can
                // (and very often does, `Display::display<f64>`'s own
                // `out` parameter closed over by its own formatting loop
                // being the exact case found here) reference an enclosing-
                // scope variable *directly*, never threading it through its
                // own explicit `params` at all. Requiring an explicit
                // argument-position match unconditionally (the first
                // version of this fix) silently broke this closure-capture
                // case, wrongly returning `false` even for `Display::
                // display<f64>`'s own dead-simple, genuinely direct
                // `(k_ret out)` return -- found immediately by a dedicated
                // debug probe once the fix's own first attempt still didn't
                // clear `examples/complex.cleave --run`'s crash.
                let inner_var = match args.iter().position(|a| matches!(a, CVal::Var(v) if *v == var)) {
                    Some(pos) => match def.params.get(pos) {
                        Some(&p) => p,
                        None => return false,
                    },
                    None => var,
                };
                if !visiting.insert(callee.clone()) {
                    return false;
                }
                let result = tail_returns_var(
                    &def.body,
                    f_name,
                    param_idx,
                    k_ret,
                    inner_var,
                    defs_by_name,
                    known_functions,
                    visiting,
                    edges,
                );
                visiting.remove(callee.as_str());
                result
            }
            _ => false,
        },
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            let then_result = tail_returns_var(
                then_branch, f_name, param_idx, k_ret, var, defs_by_name, known_functions, visiting, edges,
            );
            let else_result = tail_returns_var(
                else_branch, f_name, param_idx, k_ret, var, defs_by_name, known_functions, visiting, edges,
            );
            then_result || else_result
        }
        CExpr::Fix { body, .. } => tail_returns_var(
            body, f_name, param_idx, k_ret, var, defs_by_name, known_functions, visiting, edges,
        ),
    }
}

/// The other half of what `doc/plan-affine-ownership.md`'s Stage 2 needs,
/// alongside [`analyze`] itself: [`analyze`] answers "is this *parameter
/// position*, of this function, ever aliased" — a whole-program summary.
/// The header/no-header decision at an actual construction site needs a
/// different, narrower question: "does *this one local value*, from where
/// it's built to every place it's used, ever get committed twice, reused
/// after a commitment, or handed to a parameter position `analyze` already
/// marked aliased (or unknown)".
///
/// Deliberately **not** a second whole-program fixed point — there's
/// nothing left to iterate: [`AliasSummary`] is already fully computed by
/// the time this runs, so every call site this value reaches can be
/// answered by a single `O(1)` lookup into it. This function's own walk of
/// `body` is the *only* new work, and it's linear in the size of `body`
/// alone — reuses [`is_committed_at`] (the same per-op commitment check
/// [`collect_facts`] itself is built from) and [`occurs_in`], rather than
/// re-deriving either.
pub fn value_is_ever_aliased(var: CVar, body: &CExpr, summary: &AliasSummary) -> bool {
    match body {
        CExpr::LetPrim {
            op, args, cont, ..
        } => {
            let occurrences = is_committed_at(op, args, var);
            if occurrences >= 2 || (occurrences == 1 && occurs_in(var, cont)) {
                return true;
            }
            if let PrimOp::Extern { .. } = op {
                if args.iter().any(|a| matches!(a, CVal::Var(v) if *v == var)) {
                    return true;
                }
            }
            if is_array_embedded(op, args, var) {
                return true;
            }
            value_is_ever_aliased(var, cont, summary)
        }
        CExpr::App { func, args } => {
            if let CVal::Label(callee) = func {
                for (i, arg) in args.iter().enumerate() {
                    if matches!(arg, CVal::Var(v) if *v == var) && summary.is_aliased(callee, i) {
                        return true;
                    }
                }
            }
            false
        }
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            value_is_ever_aliased(var, then_branch, summary)
                || value_is_ever_aliased(var, else_branch, summary)
        }
        CExpr::Fix { defs, body } => {
            defs.iter()
                .any(|d| value_is_ever_aliased(var, &d.body, summary))
                || value_is_ever_aliased(var, body, summary)
        }
    }
}

/// How many times `var` appears among `op`'s own *commitment* arguments —
/// the ones that hand a reference into storage able to independently
/// outlive this exact point. Shared by [`collect_facts`] (the whole-
/// program summary builder) and [`value_is_ever_aliased`] (a one-off
/// per-value query) so the definition of "commitment" only ever lives in
/// one place — see the module's own doc comment for what counts and why.
fn is_committed_at(op: &PrimOp, args: &[CVal], var: CVar) -> usize {
    let commitment_args: &[CVal] = match op {
        PrimOp::Struct(..) | PrimOp::Array => args,
        PrimOp::FieldStore { .. } | PrimOp::Store { .. } => {
            args.last().map(std::slice::from_ref).unwrap_or(&[])
        }
        _ => &[],
    };
    commitment_args
        .iter()
        .filter(|a| matches!(a, CVal::Var(v) if *v == var))
        .count()
}

/// Whether `var` is one of the elements an `Array` construction embeds, or
/// the replicated value an `ArrayRepeat` one does — rule 4b, the module's
/// own doc comment has the full reasoning (a `Load`'s own index is runtime,
/// so unlike a `Struct`'s named fields, this analysis can never see which
/// originally-embedded element comes back out at a later load site, hence
/// treating this the same as rule 4's "no evidence, assume the dangerous
/// case" extern-fn treatment rather than [`is_committed_at`]'s own occurs-
/// check-gated rule 1/2). `Array` commits every one of its own arguments
/// (each becomes one element); `ArrayRepeat` only its first (`args =
/// [value, count]` — `count` is a size, never itself a commitment).
fn is_array_embedded(op: &PrimOp, args: &[CVal], var: CVar) -> bool {
    let commitment_args: &[CVal] = match op {
        PrimOp::Array => args,
        PrimOp::ArrayRepeat => args.first().map(std::slice::from_ref).unwrap_or(&[]),
        _ => &[],
    };
    commitment_args
        .iter()
        .any(|a| matches!(a, CVal::Var(v) if *v == var))
}

/// Runs the whole analysis — see the module's own doc comment for the
/// two-pass design and why it stays cheap regardless of how much the call
/// graph recurses.
///
/// **Every named `CFunDef` in the program is its own analyzed unit here —
/// a top-level function exactly like a local `Fix`-def (a loop, an `if`-
/// join, a real call's own resumption), no distinction between them.** A
/// `Fix`-local def has a name (its own `CVal::Label`), a parameter list,
/// and a body — structurally identical to a top-level `CTopLevelFn` (the
/// two-parameter `carried_types.is_some()` shape a loop or `if`-join uses,
/// or the one-parameter `carried_types: None` shape a real call's
/// resumption uses; `doc/plan-affine-ownership.md` §2.1 already
/// established this: "a loop, in CPS, is already a real function with an
/// explicit interface"). The *only* thing that used to set them apart here
/// was this analysis's own bookkeeping — `collect_facts`'s `App` arm
/// consulted a `known_functions` set containing only `program.funcs`'
/// names, and treated a call to anything else (any local label) as "no
/// evidence, assume aliased", the same fallback rule 4 correctly uses for
/// a genuine `extern fn` (a callee with *no* body to inspect at all). A
/// local `Fix`-def has a perfectly inspectable body sitting right there —
/// there was never a real "no evidence" case for it, just an analysis that
/// hadn't been taught to look.
///
/// Found to matter for real, not just as a cleanup: `doc/plan-affine-
/// ownership.md` §11's own loop-carried pool-allocator corruption traces
/// back to exactly this gap — a value passed as a loop's own *entry*
/// argument was unconditionally "aliased" by the old rule (it's handed to
/// `App{Label(some_loop), ..}`, and `some_loop` was never in `known_
/// functions`), regardless of what that loop's own body actually does with
/// it — closing off the one case (`affine_struct_vars`'s carried-parameter
/// propagation, further down this file) that most needed a real answer
/// instead of a reflexive "yes".
pub fn analyze(program: &CpsProgram) -> AliasSummary {
    let mut seed: HashSet<(String, usize)> = HashSet::new();
    let mut edges: Vec<Edge> = Vec::new();
    let mut all_names: HashSet<String> = HashSet::new();

    for f in &program.funcs {
        // The trailing continuation parameter is never itself a candidate
        // (`CTopLevelFn::k_ret`'s own doc comment) — only the ordinary,
        // cleave-level parameters that precede it. A local `Fix`-def has
        // no such trailing slot at all (`CFunDef::params`'s own doc
        // comment) — every one of its own params is ordinary, handled by
        // `collect_local_defs` below instead.
        let ordinary_params = &f.def.params[..f.def.params.len().saturating_sub(1)];
        all_names.insert(f.def.name.clone());
        collect_facts(&f.def.name, ordinary_params, &f.def.body, &mut seed, &mut edges);
        collect_local_defs(&f.def.body, &mut seed, &mut edges, &mut all_names);
    }

    let aliased_pairs = propagate(seed, edges);

    // Every function *and every local `Fix`-def* this program actually
    // defines a body for gets an entry, even an empty one — deliberately,
    // mirroring `refcount.rs::collect_identity_param_positions`'s own
    // identical discipline (that function's own doc comment has the full
    // reasoning). Without this, a def that's known and fully analyzed, but
    // never aliased for any parameter, would end up with *no* entry at
    // all — indistinguishable from a def this analysis has never seen (an
    // `extern fn`), and `is_aliased`'s own "no entry means no evidence,
    // assume aliased" fallback would then wrongly mark every one of its
    // parameters aliased too. Found by direct testing, not assumed:
    // `ping`/`pong` (mutual recursion, neither ever commits its own
    // parameter anywhere) both came back aliased before this fix, for
    // exactly this reason — and the identical hazard applies to a local
    // loop/if-join/resumption def now that each gets its own entry too.
    let mut aliased: HashMap<String, HashSet<usize>> = HashMap::new();
    for name in &all_names {
        aliased.entry(name.clone()).or_default();
    }
    for (name, idx) in aliased_pairs {
        aliased.entry(name).or_default().insert(idx);
    }
    AliasSummary { aliased }
}

/// Discovers every local `Fix`-def anywhere in `expr` (at any nesting
/// depth — a loop inside a loop, a real call's own resumption inside an
/// `if`-join, any combination), runs [`collect_facts`] on each one's own
/// body with *its own* name and full parameter list (never a trailing
/// continuation to exclude — see [`analyze`]'s own doc comment), and
/// registers its name in `all_names` for the "every def gets an entry"
/// step. A separate walk from [`collect_facts`]'s own `Fix` arm on
/// purpose: that one keeps checking the *enclosing* function's own
/// parameters for hazards inside nested code (a loop body reading a
/// struct captured from its own enclosing scope, say) — this one asks the
/// entirely different question "what does *this* def do with *its own*
/// parameters", the same question [`analyze`]'s own top-level loop already
/// asks for every `CTopLevelFn`.
fn collect_local_defs(
    expr: &CExpr,
    seed: &mut HashSet<(String, usize)>,
    edges: &mut Vec<Edge>,
    all_names: &mut HashSet<String>,
) {
    match expr {
        CExpr::LetPrim { cont, .. } => collect_local_defs(cont, seed, edges, all_names),
        CExpr::App { .. } => {}
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_local_defs(then_branch, seed, edges, all_names);
            collect_local_defs(else_branch, seed, edges, all_names);
        }
        CExpr::Fix { defs, body } => {
            for d in defs {
                all_names.insert(d.name.clone());
                collect_facts(&d.name, &d.params, &d.body, seed, edges);
                collect_local_defs(&d.body, seed, edges, all_names);
            }
            collect_local_defs(body, seed, edges, all_names);
        }
    }
}

/// One real walk of `body` (`f_name`'s own body — a top-level function's or
/// a local `Fix`-def's, [`analyze`]'s own doc comment — recursively
/// through every nested `Fix`/`If` — the same "plain recursive `CExpr`
/// walk, no fixpoint needed *at this stage*" shape `region_analysis.rs`/
/// `refcount.rs`'s own whole-program structural collectors already
/// establish). Populates `seed`/`edges` — see the module's own doc comment
/// for what each rule means and why it's sound.
fn collect_facts(
    f_name: &str,
    ordinary_params: &[crate::cps::CVar],
    expr: &CExpr,
    seed: &mut HashSet<(String, usize)>,
    edges: &mut Vec<Edge>,
) {
    match expr {
        CExpr::LetPrim {
            op, args, cont, ..
        } => {
            // The commitment arguments for this op -- the ones that hand a
            // reference into storage able to independently outlive this
            // exact point. `Struct`/`Array` commit *every* argument (each
            // one becomes a field/element of the fresh container);
            // `FieldStore`/`Store` commit only their own trailing *value*
            // operand (`s.field = v`/`arr[i] = v` -- the base/index
            // operands are read, never stored). Anything else (`Field`,
            // `Load`, `RawMlirOp`, ...) is an ordinary borrow, no
            // commitment at all. `Extern` is handled separately, just
            // below -- not a "commitment" in this local sense, but rule 4
            // (no visible body at all) applies to it unconditionally.
            // Rule 4, the `LetPrim`-shaped half: a call to an `extern fn`
            // is `PrimOp::Extern` here, **never** `CExpr::App` -- confirmed
            // directly (`--dump-cps`: `extern.opaque_sink v458` is a
            // `LetPrim`, not a jump at all) -- so the `App` arm below,
            // which only ever sees a *real*, CPS-visible callee, can never
            // reach an extern call to seed it. Every ordinary parameter
            // passed to one is seeded aliased unconditionally, the same
            // "no evidence, assume the dangerous case" reasoning as the
            // `App` arm's own rule 4.
            if let PrimOp::Extern { .. } = op {
                for (i, param) in ordinary_params.iter().enumerate() {
                    if args.iter().any(|a| matches!(a, CVal::Var(v) if v == param)) {
                        seed.insert((f_name.to_string(), i));
                    }
                }
            }
            // Rule 4b: embedded into an `Array`/`ArrayRepeat` construction
            // -- the module's own doc comment has the full reasoning (a
            // later `Load`'s own runtime index makes it impossible to know
            // here which originally-embedded element comes back out), same
            // unconditional "no evidence, assume the dangerous case"
            // treatment as the `Extern` case just above, not gated by
            // `occurs_in` the way rule 1/2 below are.
            for (i, param) in ordinary_params.iter().enumerate() {
                if is_array_embedded(op, args, *param) {
                    seed.insert((f_name.to_string(), i));
                }
            }
            for (i, param) in ordinary_params.iter().enumerate() {
                let occurrences = is_committed_at(op, args, *param);
                // Rule 1: committed more than once in this *same* op's own
                // argument list (`Struct(a: p, b: p)`) -- two independent
                // references to one allocation, no later use needed.
                let committed_twice_here = occurrences >= 2;
                // Rule 2: committed once here, and still referenced
                // *somewhere* later in this same function (a plain
                // occurs-check on the rest of the body from this point on
                // -- see the module's own doc comment for why this is
                // computed directly rather than via `refcount.rs`'s own
                // `Retain` markers).
                let committed_and_reused_later = occurrences == 1 && occurs_in(*param, cont);
                if committed_twice_here || committed_and_reused_later {
                    seed.insert((f_name.to_string(), i));
                }
            }
            collect_facts(f_name, ordinary_params, cont, seed, edges);
        }
        CExpr::App { func, args } => {
            // Rules 2/3: a real call (not a tail-call to a local `Fix`
            // label — `CVal::Label` for a genuine callee, matching every
            // other whole-program collector in this codebase's own
            // established "real callee vs. local continuation" test).
            // Every named `CVal::Label` target — a top-level function *or*
            // a local `Fix`-def — has a real, inspectable body somewhere
            // in this program (`analyze`'s own doc comment: an `extern
            // fn` is never called this way, it's always `PrimOp::Extern`,
            // handled above): always an edge, resolved by `propagate`
            // once that callee's own summary is known, never a local-vs-
            // top-level seed distinction. Rule 3, generalized.
            if let CVal::Label(callee) = func {
                for (i, arg) in args.iter().enumerate() {
                    if let CVal::Var(v) = arg {
                        if let Some(caller_idx) = ordinary_params.iter().position(|p| p == v) {
                            edges.push((
                                (f_name.to_string(), caller_idx),
                                (callee.clone(), i),
                            ));
                        }
                    }
                }
            }
            // Rule 5 (deliberately absent): a bare `App{func: Var(k_ret),
            // args: [Var(p)]}` — this function's own true return, handing
            // a parameter straight back unchanged — adds nothing here. See
            // the module's own doc comment for why that's correct, not a
            // gap.
        }
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_facts(f_name, ordinary_params, then_branch, seed, edges);
            collect_facts(f_name, ordinary_params, else_branch, seed, edges);
        }
        CExpr::Fix { defs, body } => {
            for d in defs {
                collect_facts(f_name, ordinary_params, &d.body, seed, edges);
            }
            collect_facts(f_name, ordinary_params, body, seed, edges);
        }
    }
}

/// Whether `v` is referenced anywhere in `expr` -- any operand of any
/// `LetPrim`/`App`, any `If`'s own condition, at any nesting depth. A
/// plain occurs-check: this only asks "does `v` ever get used again from
/// this point on", not "on which specific path" -- matching rule 2's own
/// conservative "when in doubt, aliased" stance (`doc/plan-affine-
/// ownership.md`'s own established discipline: over-classify as shared
/// rather than risk the corruption class of bug an under-classification
/// would produce).
///
/// **`Retain(v)`/`Release(v)` are deliberately never counted as an
/// occurrence of `v`**, even though their own `args` names it directly.
/// This is what makes this whole module safe to run on CPS `refcount.rs`
/// has *already* instrumented (not just the pre-refcounting snapshot
/// [`analyze`]/[`value_is_ever_aliased`] were designed against) — needed
/// so [`affine_struct_vars`] can be computed the same way `mlir_lower.rs::
/// lower_program` already computes `region_local_fns`, internally, from
/// the one `CpsProgram` it actually receives, with no new parameter
/// threaded through the pipeline at all. A `Release(v)` is the *opposite*
/// of a further use — it is refcount.rs's own declaration that `v`'s
/// lifetime ends *here*, so counting it as "occurs again" would produce
/// exactly the false-positive this comment warns about: every value
/// refcount.rs tracks gets a `Release` somewhere in its own continuation
/// by construction, which would make this occurs-check fire on
/// *everything*. A `Retain(v)` needs no special credit either: it only
/// ever fires at a site this module's own rule 1/2 already detects
/// independently, from the *same*, refcounting-untouched `Struct`/`Array`/
/// `FieldStore`/`Store` argument lists — so skipping it costs no real
/// detection power, only removes a redundant, contamination-prone path.
fn occurs_in(v: CVar, expr: &CExpr) -> bool {
    match expr {
        CExpr::LetPrim { op, args, cont, .. } => {
            let counts = !matches!(op, PrimOp::Retain(_) | PrimOp::Release(_));
            (counts && args.iter().any(|a| matches!(a, CVal::Var(vv) if *vv == v)))
                || occurs_in(v, cont)
        }
        CExpr::App { func, args } => {
            matches!(func, CVal::Var(vv) if *vv == v)
                || args.iter().any(|a| matches!(a, CVal::Var(vv) if *vv == v))
        }
        CExpr::If {
            cond,
            then_branch,
            else_branch,
        } => {
            matches!(cond, CVal::Var(vv) if *vv == v)
                || occurs_in(v, then_branch)
                || occurs_in(v, else_branch)
        }
        CExpr::Fix { defs, body } => {
            defs.iter().any(|d| occurs_in(v, &d.body)) || occurs_in(v, body)
        }
    }
}

/// The `PrimOp::Struct`-bound `CVar`s eligible for `doc/plan-affine-
/// ownership.md`'s Stage 2 — a real heap allocation, but with **no
/// `RcHeader`, no retain/release runtime call at all**: `cleave_alloc_
/// pool`/`cleave_release_pool` (`cleave-rt`'s own doc comment) directly,
/// exactly one deterministic free at the value's own provably-last use.
///
/// Two conditions, both required:
/// 1. **[`value_is_ever_aliased`] says no.** The actual precondition —
///    never more than one live reference, anywhere this program reaches.
/// 2. **The struct's own declared fields hold nothing that would need
///    cascading into at release time** — no ordinary refcounted nested
///    struct field, no tensor-tagged field. This is *this first landing's
///    own* restriction, not a fundamental limit of the analysis: a
///    struct embedding another affine-eligible value could, in principle,
///    also go through this path with its own nested release folded in —
///    but that needs a real cascade story worked out for a headerless
///    container first (mirroring `mlir_lower.rs::lower_release_cascade`,
///    which only knows how to cascade into a *headered* release today).
///    Restricting to "no refcounted/tensor field at all" sidesteps that
///    entirely for now: such a struct's own release is *always* the
///    whole story, nothing left to recurse into.
pub fn affine_struct_vars(
    program: &CpsProgram,
    summary: &AliasSummary,
    identity_summary: &IdentitySummary,
    struct_schemas: &HashMap<String, crate::cps::StructSchema>,
    mlir_types: &HashMap<String, String>,
    constructed_structs: &HashSet<String>,
    field_mutated_structs: &HashSet<String>,
    extern_boundary_structs: &HashSet<String>,
    region_local_fns: &HashSet<String>,
) -> HashSet<CVar> {
    // A function `region_analysis::find_region_local_functions` already
    // proved has exactly one call site, itself inside a loop, whose own
    // result never escapes that one iteration — `mlir_lower.rs::
    // alloc_llvm_value` routes *every* construction inside such a function
    // to the arena (`cleave_alloc_local`) unconditionally, checked *before*
    // `ctx.affine_structs` and taking priority over it (that function's own
    // doc comment: "bulk reclaim at region exit beats a per-value pool
    // round-trip whenever it's already available"). Every one of this
    // function's three propagation passes below skips such a function's own
    // body entirely — not just declining to mark its own constructions
    // affine, but never even considering them as a *source* for a
    // resumption or carried parameter elsewhere either — keeping that
    // single existing mechanism the *only* one deciding this function's own
    // allocator, with nothing here able to disagree with it.
    //
    // Found necessary by direct execution, not reasoned in advance:
    // `CLEAVE_DEBUG_POOL=1` on `many_short_lived_affine_constructions_run_
    // correctly` (`cleave/tests/affine_pool_alloc.rs`) reported real pool
    // corruption *even after* this module's own resumption-parameter
    // propagation (below) was already restricted to a callee's return being
    // affine only via this same, single `affine` set — `bump` there has
    // exactly one call site confined to one loop iteration, so `region_
    // analysis` marks it region-local, its own internal construction is
    // arena-backed, yet this analysis (with no visibility into that fact)
    // still called it "affine", making the real call's own resumption
    // parameter affine too, and its release then tried `cleave_release_
    // pool` on a pointer that was never a pool block at all — corrupting
    // the arena's own live buffer. `cleave_release_pool` has no `is_in_
    // arena` safety net the way the ordinary `cleave_release` does (its own
    // doc comment: "Unconditional... just push the block back") — this
    // exclusion is the only thing standing between a region-local
    // function's own arena-backed construction and exactly that hazard.
    let non_region_local: Vec<&crate::cps::CTopLevelFn> = program
        .funcs
        .iter()
        .filter(|f| !region_local_fns.contains(&f.def.name))
        .collect();

    let mut affine = HashSet::new();
    for f in &non_region_local {
        collect_affine_candidates(
            &f.def.body,
            summary,
            struct_schemas,
            mlir_types,
            constructed_structs,
            field_mutated_structs,
            extern_boundary_structs,
            &f.def.body,
            &mut affine,
        );
    }

    // `doc/plan-affine-ownership.md` §11's first confirmed corruption, the
    // narrower half: a real call's own *resumption* parameter (`Fix{defs:
    // [def with one param, carried_types: None], body: App{Label(callee),
    // ..}}` — the exact shape `alias_analysis::analyze_identity`'s own
    // `tail_returns_var` already recognizes) is never itself a
    // `PrimOp::Struct` site, so `collect_affine_candidates` above never
    // considers it — yet `refcount::insert_refcounting` releases it
    // *directly*, by this exact `CVar`, whenever the call's result isn't
    // threaded any further (no loop/if-join re-binding on top). If the
    // callee always hands back an allocation this analysis already knows
    // is pool-eligible, this resumption parameter denotes the *same*
    // allocation under a new name — its own release must go through the
    // pool too, or it reads an `RcHeader` that was never written
    // (confirmed directly: `CLEAVE_DEBUG_POOL=1` on `many_short_lived_
    // affine_constructions_run_correctly`, `cleave/tests/affine_pool_
    // alloc.rs`, reported "popped block ... was not marked parked (pool
    // corruption)" *even though* the test's own numeric assertion still
    // happened to pass — a silent corruption, not a crash, on that
    // particular size/reuse pattern).
    //
    // §11.4's extension, landed the same way: a **loop/if-join's own
    // carried parameter** (`def.carried_types.is_some()`) is eligible the
    // identical way, but needs *every* one of its sources — the entry
    // argument, and every back-edge argument fed to it from anywhere within
    // the same enclosing top-level function — to already be affine, not
    // just one caller like a resumption parameter has. Only became
    // reachable at all once `analyze()` itself stopped treating a call to a
    // local `Fix`-label as automatically aliased (`analyze`'s own doc
    // comment) — before that, a loop's own entry value could never be
    // proven affine no matter what this function did with it.
    //
    // Both are one shared fixed point, not two separate ones — mirrors
    // this whole module's own established discipline (`propagate()`'s own
    // doc comment): function `f` might tail-forward another function `g`'s
    // own resumption result completely unchanged (`fn f(..) { g(..) }`),
    // and a loop's own back-edge might itself be fed by a resumption
    // parameter becoming affine only in this very pass — chains of any
    // length and either shape resolve together, by iterating until nothing
    // new is found anywhere, never by recursing into a chain by hand.
    let known_functions: HashSet<&str> = program.funcs.iter().map(|f| f.def.name.as_str()).collect();
    let per_fn_carried_facts: Vec<CarriedParamFacts> = non_region_local
        .iter()
        .map(|f| {
            let mut facts = CarriedParamFacts {
                calls: HashMap::new(),
                carried_defs: Vec::new(),
            };
            collect_carried_param_facts(&f.def.body, &mut facts);
            facts
        })
        .collect();
    // [`carried_param_flows_to_own_backedge`]'s own local-def lookup --
    // one map per top-level function, the same "every local `Fix`-def
    // anywhere in this one function, keyed by its own label" shape
    // [`analyze_identity`] already builds for the identical reason.
    let per_fn_defs_by_name: Vec<HashMap<&str, &CFunDef>> = non_region_local
        .iter()
        .map(|f| {
            let mut defs_by_name = HashMap::new();
            collect_local_defs_by_name(&f.def.body, &mut defs_by_name);
            defs_by_name
        })
        .collect();
    loop {
        let mut fn_return_affine: HashMap<&str, bool> = HashMap::new();
        for f in &non_region_local {
            let mut return_vars = Vec::new();
            let clean = collect_return_vars(f.k_ret, &f.def.body, &mut return_vars);
            fn_return_affine.insert(
                f.def.name.as_str(),
                clean && !return_vars.is_empty() && return_vars.iter().all(|v| affine.contains(v)),
            );
        }
        let mut changed = false;
        for f in &non_region_local {
            changed |= collect_affine_resumption_params(
                &f.def.body,
                &known_functions,
                &fn_return_affine,
                identity_summary,
                &mut affine,
            );
        }
        for (facts, defs_by_name) in per_fn_carried_facts.iter().zip(&per_fn_defs_by_name) {
            changed |= collect_affine_carried_params(
                facts,
                defs_by_name,
                &known_functions,
                identity_summary,
                &mut affine,
            );
        }
        // §13/§14's own extension: `refcount::insert_refcounting` sometimes
        // releases a field *read back out* of a never-mutated struct
        // directly (`(field.inner v466)`/`(release v477)`), rather than the
        // containing struct's own `CVar` — the exact same physical
        // allocation as whatever was embedded at that field's own
        // construction site, under a third name. If that field position is
        // already proven always-affine (`field_affine_positions`, computed
        // fresh each iteration from the current `affine` — cheap, and
        // needed since a field's own eligibility can itself only become
        // known partway through this very fixed point), the field-read
        // result needs the identical treatment or its own release
        // disagrees with whatever allocator actually backed it — confirmed
        // directly, a real `STATUS_ACCESS_VIOLATION` on the very first
        // nested-struct pool test written for this extension.
        let field_affine = field_affine_positions(program, &affine);
        for f in &non_region_local {
            changed |= collect_affine_field_reads(
                &f.def.body,
                struct_schemas,
                field_mutated_structs,
                &field_affine,
                &mut affine,
            );
        }
        if !changed {
            break;
        }
    }

    affine
}

/// See [`affine_struct_vars`]'s own doc comment on the loop this is called
/// from for why this exists. Walks the whole body once, adding to `affine`
/// every `PrimOp::Field`-bound `CVar` reading an already-proven-affine
/// field off a never-field-mutated struct. `PrimOp::Load` (array indexing)
/// is deliberately not covered — `field_affine_positions` never marks an
/// array-typed field position affine in the first place (an array's own
/// `CVar` is never itself a member of `affine`), so this would find nothing
/// to do for one regardless; not worth the extra code until a real case
/// needs it.
fn collect_affine_field_reads(
    expr: &CExpr,
    struct_schemas: &HashMap<String, crate::cps::StructSchema>,
    field_mutated_structs: &HashSet<String>,
    field_affine: &HashMap<(String, usize), bool>,
    affine: &mut HashSet<CVar>,
) -> bool {
    match expr {
        CExpr::LetPrim { var, op, cont, .. } => {
            let mut changed = false;
            if let PrimOp::Field { struct_ty, field } = op {
                if !affine.contains(var) {
                    let (name, type_args) = crate::mlir_lower::struct_name_and_args(struct_ty);
                    if !field_mutated_structs.contains(name) {
                        let fields = crate::mlir_lower::struct_field_types(struct_schemas, name, type_args);
                        if let Some(pos) = fields.iter().position(|(n, _)| n == field) {
                            if field_affine.get(&(name.to_string(), pos)) == Some(&true) {
                                affine.insert(*var);
                                changed = true;
                            }
                        }
                    }
                }
            }
            changed
                | collect_affine_field_reads(cont, struct_schemas, field_mutated_structs, field_affine, affine)
        }
        CExpr::App { .. } => false,
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            let a = collect_affine_field_reads(then_branch, struct_schemas, field_mutated_structs, field_affine, affine);
            let b = collect_affine_field_reads(else_branch, struct_schemas, field_mutated_structs, field_affine, affine);
            a || b
        }
        CExpr::Fix { defs, body } => {
            let mut changed = false;
            for d in defs {
                changed |=
                    collect_affine_field_reads(&d.body, struct_schemas, field_mutated_structs, field_affine, affine);
            }
            changed |= collect_affine_field_reads(body, struct_schemas, field_mutated_structs, field_affine, affine);
            changed
        }
    }
}

/// One top-level function's own body, reduced to exactly what [`collect_
/// affine_carried_params`] needs: every call made anywhere to a local
/// `Fix`-label (keyed by that label's own name, each call's own args
/// reduced to `Some(v)` for a plain `CVal::Var(v)` or `None` for anything
/// else — a literal, `CVal::Label`, `CVal::Unit`/`CVal::Int`/... is never
/// attributable to a single origin `CVar`, so the AND-join downstream must
/// treat it as an unresolved source, never silently skip it), and every
/// loop/if-join-shaped local def's own name and parameter list (`def.
/// carried_types.is_some()` — a real call's own resumption, `carried_
/// types: None`, is [`collect_affine_resumption_params`]'s own separate
/// concern, not this one's). A local `Fix`-def's own name never escapes
/// its enclosing top-level function, so one call per top-level body is
/// always enough — never needs to look further.
struct CarriedParamFacts {
    calls: HashMap<String, Vec<Vec<Option<CVar>>>>,
    carried_defs: Vec<(String, Vec<CVar>)>,
}

fn collect_carried_param_facts(expr: &CExpr, out: &mut CarriedParamFacts) {
    match expr {
        CExpr::LetPrim { cont, .. } => collect_carried_param_facts(cont, out),
        CExpr::App { func, args } => {
            if let CVal::Label(name) = func {
                let call_args = args
                    .iter()
                    .map(|a| match a {
                        CVal::Var(v) => Some(*v),
                        _ => None,
                    })
                    .collect();
                out.calls.entry(name.clone()).or_default().push(call_args);
            }
        }
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_carried_param_facts(then_branch, out);
            collect_carried_param_facts(else_branch, out);
        }
        CExpr::Fix { defs, body } => {
            for d in defs {
                if d.carried_types.is_some() {
                    out.carried_defs.push((d.name.clone(), d.params.clone()));
                }
                collect_carried_param_facts(&d.body, out);
            }
            collect_carried_param_facts(body, out);
        }
    }
}

/// Adds a loop/if-join's own carried parameter to `affine` once *every*
/// call this program makes to its own name — the entry, and every
/// back-edge, wherever they textually sit — either (a) passes an already-
/// affine `CVar` at that same position (the original rule), or (b) is
/// itself provably the *same allocation* as the carried parameter, traced
/// all the way through the loop's own body via [`carried_param_flows_to_
/// own_backedge`] (the fix below). Returns whether anything new was added,
/// exactly like [`collect_affine_resumption_params`]'s own identical
/// contract.
///
/// **Why (b) is needed, not just (a) — a real, confirmed gap, not a
/// hypothetical one**: a carried value threaded each iteration through an
/// identity-shaped real call (`b = display_and_return(cond, b);`,
/// `Display::display<Complex<T>>`'s own real shape, `doc/backlog.md`'s
/// "examples/complex.cleave" entry) creates a genuine mutual dependency
/// rule (a) alone can never resolve: the back-edge argument is that call's
/// own *resumption* parameter, which [`collect_affine_resumption_params`]
/// can only mark affine once the carried parameter *itself* is already
/// affine (it's the call's own argument) — but the carried parameter can
/// only become affine, under rule (a), once that same resumption parameter
/// already is. Neither side has any way to seed first; the fixed point
/// converges after exactly one iteration with nothing added, even though
/// the true answer (both affine, anchored by the entry argument from
/// *outside* the loop) is real and sound. Confirmed directly with a
/// dedicated probe before writing this fix: `entry affine = true`,
/// `carried affine = false` even though the identity fact itself
/// (`IdentitySummary::returns_unchanged`) was already correctly `true`.
///
/// Rule (b) sidesteps the mutual dependency entirely: if the carried
/// parameter is *structurally* guaranteed to reach every one of its own
/// back-edges unchanged (no need for those *specific* arguments to be
/// independently affine at all — they denote the exact same allocation by
/// construction), then its own affine-ness reduces to whether *any* call
/// site (in practice, the one real anchor: the entry argument from outside
/// the loop, itself never traced by [`carried_param_flows_to_own_
/// backedge`] since it's a different `CVar` the loop's own body never
/// references) is already affine — rule (a), unconditionally, for that one
/// site.
#[allow(clippy::too_many_arguments)]
fn collect_affine_carried_params(
    facts: &CarriedParamFacts,
    defs_by_name: &HashMap<&str, &CFunDef>,
    known_functions: &HashSet<&str>,
    identity_summary: &IdentitySummary,
    affine: &mut HashSet<CVar>,
) -> bool {
    let mut changed = false;
    for (name, params) in &facts.carried_defs {
        // A def this program never actually calls (dead code, or one this
        // walk simply hasn't found a call site for) has no sources at all
        // — "vacuously true" would be unsound here (nothing ever proved it
        // safe), unlike a plain boolean AND over a genuinely non-empty set.
        let Some(calls) = facts.calls.get(name).filter(|c| !c.is_empty()) else {
            continue;
        };
        let Some(def) = defs_by_name.get(name.as_str()) else {
            continue;
        };
        for (i, param) in params.iter().enumerate() {
            if affine.contains(param) {
                continue;
            }
            // Rule (a): every recorded call site (entry + every back-edge)
            // independently passes an already-affine value.
            let every_source_affine = calls
                .iter()
                .all(|call_args| matches!(call_args.get(i), Some(Some(v)) if affine.contains(v)));
            // Rule (b): the carried parameter is structurally guaranteed to
            // reach every one of its own back-edges unchanged -- those
            // specific back-edge arguments need no independent proof at
            // all, so a *single* already-affine call site (in practice,
            // the one real anchor: the entry argument from outside the
            // loop) is enough.
            let anchored_by_any_affine_source = || {
                carried_param_flows_to_own_backedge(
                    &def.body, name, i, *param, defs_by_name, known_functions, identity_summary,
                    &mut HashSet::new(),
                ) && calls
                    .iter()
                    .any(|call_args| matches!(call_args.get(i), Some(Some(v)) if affine.contains(v)))
            };
            if every_source_affine || anchored_by_any_affine_source() {
                affine.insert(*param);
                changed = true;
            }
        }
    }
    changed
}

/// Whether `var`, starting from `expr` (part of `loop_name`'s own body, or
/// a local def's body reached while tracing through one), ever flows
/// *unchanged* all the way back into a tail-call to `loop_name` itself,
/// with `var` as the literal argument at `position` — the loop-carried-
/// parameter analogue of [`tail_returns_var`] (that function's own doc
/// comment, and [`analyze_identity`]'s, have the shared motivation: local
/// join-point hops and identity-shaped real-call hops are traced through
/// identically here), just targeting "reaches my own back-edge unchanged"
/// instead of "reaches `k_ret` unchanged". [`collect_affine_carried_
/// params`]'s own doc comment has the full story on why this exists.
///
/// Three real cases, mirroring [`tail_returns_var`]'s exactly:
/// 1. **Tail-calls `loop_name` itself** with `var` at `position` — proven,
///    `true` right here.
/// 2. **Tail-calls a local `Fix`-def** (a join point, a nested resumption)
///    — trace through it exactly like [`tail_returns_var`] does: re-bind
///    to whichever of its own parameters received `var`, or leave it
///    unchanged if it's simply captured rather than passed.
/// 3. **Tail-calls a real, whole-program function** with `var` at position
///    `j` — unlike [`tail_returns_var`] (which defers this case to the
///    interprocedural fixed point via an edge), `identity_summary` is
///    already fully resolved by the time this runs (`affine_struct_vars`'s
///    own call site computes it first) — consult it directly: if `Some
///    (true)`, trace on through that call's own resumption (found the same
///    way a local join point is, via `defs_by_name`), with `var` re-bound
///    to the resumption's own single parameter. Otherwise, dead end,
///    `false`.
#[allow(clippy::too_many_arguments)]
fn carried_param_flows_to_own_backedge(
    expr: &CExpr,
    loop_name: &str,
    position: usize,
    var: CVar,
    defs_by_name: &HashMap<&str, &CFunDef>,
    known_functions: &HashSet<&str>,
    identity_summary: &IdentitySummary,
    visiting: &mut HashSet<String>,
) -> bool {
    match expr {
        CExpr::LetPrim { cont, .. } => carried_param_flows_to_own_backedge(
            cont, loop_name, position, var, defs_by_name, known_functions, identity_summary, visiting,
        ),
        CExpr::App { func, args } => match func {
            CVal::Label(callee) if callee == loop_name => {
                matches!(args.get(position), Some(CVal::Var(v)) if *v == var)
            }
            CVal::Label(callee) if known_functions.contains(callee.as_str()) => {
                // `var` might not be one of *this* call's own arguments at
                // all (an unrelated computation happening in between --
                // exactly the loop-bound check, `Ord::lt<i32>`, every
                // `for`/`while` loop's own condition test runs before ever
                // reaching its real body; found directly, the first
                // version of this function returned `false` here
                // unconditionally and never got past a single loop
                // iteration's own bound check as a result). If so, it
                // survives unchanged into this call's own trailing
                // continuation once the call returns -- trace on through
                // that, `var` unchanged, mirroring `tail_returns_var`'s own
                // identical fix. Only when `var` *is* one of the real
                // arguments does the callee's own identity fact apply.
                let resumption_var = match args.iter().position(|a| matches!(a, CVal::Var(v) if *v == var)) {
                    Some(j) if identity_summary.returns_unchanged(callee, j) == Some(true) => None,
                    Some(_) => return false,
                    None => Some(var),
                };
                args.iter().any(|a| {
                    if let CVal::Label(cont) = a {
                        if let Some(def) = defs_by_name.get(cont.as_str()) {
                            let next_var = match resumption_var {
                                Some(v) => Some(v),
                                None => def.params.first().copied(),
                            };
                            if let Some(next_var) = next_var {
                                if visiting.insert(cont.clone()) {
                                    let result = carried_param_flows_to_own_backedge(
                                        &def.body, loop_name, position, next_var, defs_by_name,
                                        known_functions, identity_summary, visiting,
                                    );
                                    visiting.remove(cont.as_str());
                                    return result;
                                }
                            }
                        }
                    }
                    false
                })
            }
            CVal::Label(callee) => {
                let Some(def) = defs_by_name.get(callee.as_str()) else {
                    return false;
                };
                let inner_var = match args.iter().position(|a| matches!(a, CVal::Var(v) if *v == var)) {
                    Some(pos) => match def.params.get(pos) {
                        Some(&p) => p,
                        None => return false,
                    },
                    None => var,
                };
                if !visiting.insert(callee.clone()) {
                    return false;
                }
                let result = carried_param_flows_to_own_backedge(
                    &def.body, loop_name, position, inner_var, defs_by_name, known_functions,
                    identity_summary, visiting,
                );
                visiting.remove(callee.as_str());
                result
            }
            _ => false,
        },
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            let t = carried_param_flows_to_own_backedge(
                then_branch, loop_name, position, var, defs_by_name, known_functions, identity_summary,
                visiting,
            );
            let e = carried_param_flows_to_own_backedge(
                else_branch, loop_name, position, var, defs_by_name, known_functions, identity_summary,
                visiting,
            );
            t || e
        }
        CExpr::Fix { body, .. } => carried_param_flows_to_own_backedge(
            body, loop_name, position, var, defs_by_name, known_functions, identity_summary, visiting,
        ),
    }
}

/// Every `CVar` a tail return (`App{Var(k_ret), [v]}`, at any nesting
/// depth) of this one function hands back, pushed into `out`. Returns
/// `false` — poisoning the whole fact for the caller, the same "no
/// evidence, assume unsafe" discipline this module already applies
/// everywhere else — the moment a return site doesn't decompose into
/// exactly zero or one plain `CVal::Var` (a tuple/multi-value return, or a
/// literal): this function's own return can't be attributed to a single
/// traceable origin, so it must never be treated as "always affine".
fn collect_return_vars(k_ret: CVar, expr: &CExpr, out: &mut Vec<CVar>) -> bool {
    match expr {
        CExpr::LetPrim { cont, .. } => collect_return_vars(k_ret, cont, out),
        CExpr::App { func, args } => {
            if matches!(func, CVal::Var(v) if *v == k_ret) {
                match args.as_slice() {
                    [] => true,
                    [CVal::Var(v)] => {
                        out.push(*v);
                        true
                    }
                    _ => false,
                }
            } else {
                // Not a return at all — a local jump (loop/if-join/another
                // real call's own resumption), nothing to attribute here.
                true
            }
        }
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => collect_return_vars(k_ret, then_branch, out) && collect_return_vars(k_ret, else_branch, out),
        CExpr::Fix { defs, body } => {
            defs.iter().all(|d| collect_return_vars(k_ret, &d.body, out))
                && collect_return_vars(k_ret, body, out)
        }
    }
}

/// Walks one top-level function's own body, adding to `affine` the
/// resumption parameter of every real call whose callee's own return is
/// already known `fn_return_affine`. Returns whether anything new was
/// added — the fixed-point loop in [`affine_struct_vars`] stops once every
/// call in the program reports `false`.
fn collect_affine_resumption_params(
    expr: &CExpr,
    known_functions: &HashSet<&str>,
    fn_return_affine: &HashMap<&str, bool>,
    identity_summary: &IdentitySummary,
    affine: &mut HashSet<CVar>,
) -> bool {
    match expr {
        CExpr::LetPrim { cont, .. } => collect_affine_resumption_params(
            cont, known_functions, fn_return_affine, identity_summary, affine,
        ),
        CExpr::App { .. } => false,
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            let a = collect_affine_resumption_params(
                then_branch, known_functions, fn_return_affine, identity_summary, affine,
            );
            let b = collect_affine_resumption_params(
                else_branch, known_functions, fn_return_affine, identity_summary, affine,
            );
            a || b
        }
        CExpr::Fix { defs, body } => {
            let mut changed = false;
            if let ([def], CExpr::App {
                func: CVal::Label(callee),
                args: call_args,
            }) = (defs.as_slice(), body.as_ref())
            {
                if def.carried_types.is_none() && known_functions.contains(callee.as_str()) {
                    if let [p] = def.params.as_slice() {
                        // Two, independent reasons this call's own
                        // resumption parameter can denote an already-
                        // affine allocation under a new name:
                        let callee_constructs_and_returns_affine =
                            fn_return_affine.get(callee.as_str()).copied().unwrap_or(false);
                        // `Display::display<Complex<T>>`'s own shape,
                        // `doc/backlog.md`'s "examples/complex.cleave --
                        // run" entry has the full story: `callee` never
                        // constructs anything of its own at all -- it just
                        // hands back one of its *own* parameters unchanged
                        // (proven by `analyze_identity`, generalized to
                        // trace through however many local join points it
                        // takes to get there). If the caller's own
                        // argument at that exact position is *itself*
                        // already known-affine, the resumption denotes
                        // that same allocation, not a fresh one --
                        // `fn_return_affine` alone can never see this,
                        // since it only asks "is the literal returned
                        // `CVar` a `PrimOp::Struct` site in `affine`",
                        // never "does it trace back to a *caller-supplied*
                        // one through an identity-shaped parameter".
                        let callee_forwards_an_affine_argument = call_args.iter().enumerate().any(
                            |(j, a)| {
                                matches!(a, CVal::Var(v) if affine.contains(v))
                                    && identity_summary.returns_unchanged(callee, j) == Some(true)
                            },
                        );
                        if !affine.contains(p)
                            && (callee_constructs_and_returns_affine || callee_forwards_an_affine_argument)
                        {
                            affine.insert(*p);
                            changed = true;
                        }
                    }
                }
            }
            for d in defs {
                changed |= collect_affine_resumption_params(
                    &d.body, known_functions, fn_return_affine, identity_summary, affine,
                );
            }
            changed |= collect_affine_resumption_params(
                body, known_functions, fn_return_affine, identity_summary, affine,
            );
            changed
        }
    }
}

fn collect_affine_candidates(
    expr: &CExpr,
    summary: &AliasSummary,
    struct_schemas: &HashMap<String, crate::cps::StructSchema>,
    mlir_types: &HashMap<String, String>,
    constructed_structs: &HashSet<String>,
    field_mutated_structs: &HashSet<String>,
    extern_boundary_structs: &HashSet<String>,
    fn_body: &CExpr,
    affine: &mut HashSet<CVar>,
) {
    match expr {
        CExpr::LetPrim {
            var, op, ty, cont, ..
        } => {
            if matches!(op, PrimOp::Struct(..))
                && struct_cascade_is_viable(
                    ty,
                    struct_schemas,
                    mlir_types,
                    constructed_structs,
                    field_mutated_structs,
                    extern_boundary_structs,
                )
                && !value_is_ever_aliased(*var, fn_body, summary)
            {
                affine.insert(*var);
            }
            collect_affine_candidates(
                cont,
                summary,
                struct_schemas,
                mlir_types,
                constructed_structs,
                field_mutated_structs,
                extern_boundary_structs,
                fn_body,
                affine,
            );
        }
        CExpr::App { .. } => {}
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_affine_candidates(
                then_branch,
                summary,
                struct_schemas,
                mlir_types,
                constructed_structs,
                field_mutated_structs,
                extern_boundary_structs,
                fn_body,
                affine,
            );
            collect_affine_candidates(
                else_branch,
                summary,
                struct_schemas,
                mlir_types,
                constructed_structs,
                field_mutated_structs,
                extern_boundary_structs,
                fn_body,
                affine,
            );
        }
        CExpr::Fix { defs, body } => {
            for d in defs {
                collect_affine_candidates(
                    &d.body,
                    summary,
                    struct_schemas,
                    mlir_types,
                    constructed_structs,
                    field_mutated_structs,
                    extern_boundary_structs,
                    fn_body,
                    affine,
                );
            }
            collect_affine_candidates(
                body,
                summary,
                struct_schemas,
                mlir_types,
                constructed_structs,
                field_mutated_structs,
                extern_boundary_structs,
                fn_body,
                affine,
            );
        }
    }
}

/// Whether `ty` (a struct type) has zero fields that would need
/// `mlir_lower.rs::lower_release_cascade`-style recursion at release time
/// — see [`affine_struct_vars`]'s own doc comment for why this is this
/// first landing's own restriction, not a fundamental one.
/// Whether `field_ty` (a field's own declared type — the true leaf, after
/// flattening any array nesting, exactly like `mlir_lower.rs::push_cascade_
/// children`'s own recursion) would need `lower_release_cascade`-style
/// recursion at release time — a tensor-tagged type, or an ordinary
/// refcounted nested struct. Shared by both directions
/// [`struct_cascade_is_viable`] needs: "does this struct have any cascade
/// work at all" and "is this specific cascade-worthy field's own struct
/// type itself mutation-free".
fn field_needs_cascade(
    field_ty: &Ty,
    struct_schemas: &HashMap<String, crate::cps::StructSchema>,
    mlir_types: &HashMap<String, String>,
    constructed_structs: &HashSet<String>,
    field_mutated_structs: &HashSet<String>,
    extern_boundary_structs: &HashSet<String>,
) -> bool {
    let (_, leaf_ty) = crate::mlir_lower::flatten_array_dims(field_ty);
    let (field_name, _): (&str, &[Ty]) = match leaf_ty {
        Ty::Con(n) => (n.as_str(), &[]),
        Ty::App(n, args) => (n.as_str(), args.as_slice()),
        // A genuine primitive leaf (after flattening): never a cascade
        // target.
        _ => return false,
    };
    if matches!(mlir_types.get(field_name).map(String::as_str), Some("tensor") | Some("vector")) {
        return true;
    }
    struct_schemas.contains_key(field_name)
        && crate::refcount::is_refcounted(
            leaf_ty,
            struct_schemas,
            mlir_types,
            constructed_structs,
            field_mutated_structs,
            extern_boundary_structs,
        )
}

/// Whether `ty` (a struct type) can have its `Release` cascaded into at
/// release time at all — either because it has zero fields that would need
/// `mlir_lower.rs::lower_release_cascade`-style recursion
/// ([`field_needs_cascade`]), or because *every* cascade-worthy field's own
/// occupant is fixed forever from its own single construction site: `ty`
/// itself, and every cascade-worthy *struct* field's own type (a tensor
/// field is always a leaf — cleave has no way to field-mutate one of those
/// independently of the container holding it), is never field-mutated
/// anywhere in the whole program (`field_mutated_structs`). Decidable
/// statically, with no flow-sensitive points-to needed at all — `doc/plan-
/// affine-ownership.md` §14.4's own "Phase B" scoping, confirmed directly
/// on the real target (`examples/mnist-interop`): zero `PrimOp::FieldStore`
/// anywhere in the whole compiled kernel.
///
/// A struct that clears this bar is *not yet* itself affine-eligible on its
/// own — [`affine_struct_vars`]'s own "never aliased" condition (rule 1)
/// still has to hold too, exactly as before. What clearing it *does* mean:
/// [`field_affine_positions`] may go on to decide, per field, whether
/// `mlir_lower.rs::lower_release_pool_cascade` (a headerless cascade) or
/// the ordinary headered one is the right generated code for that specific
/// field, once and for all — the allocator choice for a *field* is a
/// property of the *type*, not of any one instance (that generated cascade
/// runs for every value of this struct type, not once per construction
/// site), which is exactly why "is it ever field-mutated" (a whole-program,
/// type-level fact) is the right question here, not a per-instance one.
fn struct_cascade_is_viable(
    ty: &Ty,
    struct_schemas: &HashMap<String, crate::cps::StructSchema>,
    mlir_types: &HashMap<String, String>,
    constructed_structs: &HashSet<String>,
    field_mutated_structs: &HashSet<String>,
    extern_boundary_structs: &HashSet<String>,
) -> bool {
    let (name, type_args): (&str, &[Ty]) = match ty {
        Ty::Con(name) => (name.as_str(), &[]),
        Ty::App(name, args) => (name.as_str(), args.as_slice()),
        _ => return false,
    };
    // `ty` itself must not be tensor-tagged (`#[mlir_type(tensor)]`/
    // `vector`) -- such a construction never reaches `alloc_llvm_value`'s
    // own new dispatch branch at all today (`lower_prim_op`'s `PrimOp::
    // Struct` arm routes it to `lower_tagged_struct_construct` instead,
    // which never even looks at `ctx.affine_structs`), so this check is
    // currently redundant with that call-site structure -- kept anyway,
    // deliberately, so this function stays correct on its own rather than
    // relying on an unrelated caller to keep it safe. A tensor's own
    // representation is a memref descriptor, not the plain heap layout
    // `cleave_alloc_pool`/`cleave_release_pool` assume.
    if matches!(mlir_types.get(name).map(String::as_str), Some("tensor") | Some("vector")) {
        return false;
    }
    let fields = crate::mlir_lower::struct_field_types(struct_schemas, name, type_args);
    let has_cascade_field = fields.iter().any(|(_, field_ty)| {
        field_needs_cascade(
            field_ty,
            struct_schemas,
            mlir_types,
            constructed_structs,
            field_mutated_structs,
            extern_boundary_structs,
        )
    });
    if !has_cascade_field {
        return true;
    }
    if field_mutated_structs.contains(name) {
        return false;
    }
    fields.iter().all(|(_, field_ty)| {
        let (_, leaf_ty) = crate::mlir_lower::flatten_array_dims(field_ty);
        let (field_name, _): (&str, &[Ty]) = match leaf_ty {
            Ty::Con(n) => (n.as_str(), &[]),
            Ty::App(n, args) => (n.as_str(), args.as_slice()),
            _ => return true,
        };
        matches!(mlir_types.get(field_name).map(String::as_str), Some("tensor") | Some("vector"))
            || !field_mutated_structs.contains(field_name)
    })
}

/// For every `(struct type name, field position)` whose own type is
/// [`field_needs_cascade`]-worthy, whether *every* construction site of
/// that struct type anywhere in the program passes an already-[`affine_
/// struct_vars`]-eligible argument at that position. A whole-program AND,
/// not an OR: `mlir_lower.rs::lower_release_pool_cascade` is generated once
/// per *type*, never once per instance, so the allocator choice for a
/// given field must hold for every instance of that type — one instance
/// backed by an ordinary header is enough to require the header-based
/// cascade for every one of them (`doc/plan-affine-ownership.md` §7's own
/// "when in doubt, header" discipline).
pub fn field_affine_positions(program: &CpsProgram, affine: &HashSet<CVar>) -> HashMap<(String, usize), bool> {
    let mut result: HashMap<(String, usize), bool> = HashMap::new();
    for f in &program.funcs {
        collect_field_affine_facts(&f.def.body, affine, &mut result);
    }
    result
}

fn collect_field_affine_facts(
    expr: &CExpr,
    affine: &HashSet<CVar>,
    result: &mut HashMap<(String, usize), bool>,
) {
    match expr {
        CExpr::LetPrim { op, args, cont, .. } => {
            if let PrimOp::Struct(name, _) = op {
                for (i, arg) in args.iter().enumerate() {
                    let this_site_affine = matches!(arg, CVal::Var(v) if affine.contains(v));
                    result
                        .entry((name.clone(), i))
                        .and_modify(|ok| *ok = *ok && this_site_affine)
                        .or_insert(this_site_affine);
                }
            }
            collect_field_affine_facts(cont, affine, result);
        }
        CExpr::App { .. } => {}
        CExpr::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_field_affine_facts(then_branch, affine, result);
            collect_field_affine_facts(else_branch, affine, result);
        }
        CExpr::Fix { defs, body } => {
            for d in defs {
                collect_field_affine_facts(&d.body, affine, result);
            }
            collect_field_affine_facts(body, affine, result);
        }
    }
}

/// Standard worklist propagation over a small, already-extracted
/// dependency graph — never touches the CPS itself. `seed` are the
/// positions already known aliased (rules 1/3); `edges` are "if the
/// second one is aliased, so is the first" dependencies (rule 2). Runs
/// until nothing new is discovered — the whole reason recursive and
/// mutually-recursive functions resolve to a correct answer without
/// needing any special-case recognition of "this is a cycle": the
/// worklist simply keeps going until it has nothing left to propagate,
/// exactly like `refcount.rs::collect_local_free_vars`'s own established
/// fixed point does for free variables.
fn propagate(seed: HashSet<(String, usize)>, edges: Vec<Edge>) -> HashSet<(String, usize)> {
    // Index edges by their own dependency (the RHS) so a newly-discovered
    // aliased pair can cheaply find everything waiting on it, instead of
    // re-scanning the whole edge list on every step.
    let mut waiting_on: HashMap<(String, usize), Vec<(String, usize)>> = HashMap::new();
    for (dependent, depends_on) in edges {
        waiting_on.entry(depends_on).or_default().push(dependent);
    }

    let mut aliased: HashSet<(String, usize)> = HashSet::new();
    let mut worklist: Vec<(String, usize)> = Vec::new();
    for pair in seed {
        if aliased.insert(pair.clone()) {
            worklist.push(pair);
        }
    }

    while let Some(pair) = worklist.pop() {
        if let Some(dependents) = waiting_on.get(&pair) {
            for dependent in dependents {
                if aliased.insert(dependent.clone()) {
                    worklist.push(dependent.clone());
                }
            }
        }
    }

    aliased
}
