//! First integration of `egg` (equality saturation / e-graphs) into the
//! compiler pipeline, driven by a user-declared `axiom` (`registry.rs::
//! Registry::axioms`). See the plan this module was built against for the
//! full design reasoning (post-CPS, no binders, copy-propagation for free,
//! `hld.md`'s own v1 trust model) — summarized here as the module grows.
//!
//! ## Scope of this module, today
//!
//! Only the `Language`/`Analysis` scaffolding (`CleaveLang`/`ConstantFold`)
//! exists so far — no translator to or from `cps.rs`'s own `CExpr` yet, no
//! axiom-to-`Rewrite` translation. Both are separate, later additions.
//!
//! `CleaveLang` deliberately carries **one** `Op(Symbol, Vec<Id>)` variant
//! for *every* operation node, not a separate `AlgebraOp`/`RawOp` pair as
//! first sketched (let alone the `(algebra, method, type)`-triple sketched
//! before that) — `egg::define_language!` only lets a *string-tagged*
//! variant (`"+" = Add([Id; 2])`) carry a compile-time-fixed tag, the same
//! literal for every instance; it's a *bare data* variant (`Other(Symbol,
//! Vec<Id>)`, no leading string) that carries a genuinely per-instance
//! runtime symbol — and a bare data variant's own `FromOp`/parsing tries
//! `op.parse::<Symbol>()`, which never fails, so a *second* same-shaped
//! bare variant would simply be unreachable by parsing, shadowed by
//! whichever comes first in the enum. Both findings came from directly
//! reading `egg-0.11.0`'s own macro source and iterating against real
//! compile errors, not from the (partly stale) doc comment alone.
//!
//! `Op`'s own `Symbol` reuses a `ConcreteUnit`'s already-unique `name`
//! directly for an algebra-dispatched operation (e.g. `"Ring::add<i32>"`),
//! or a combined `"mlir-op:type"` string for a raw `mlir::...` call with no
//! `ConcreteUnit` of its own to borrow a name from (e.g. `"arith.addi:i32"`)
//! — two concrete instantiations of the same abstract op are simply two
//! different discriminants, never structurally unified by accident,
//! achieving the type-safety goal from this project's own design discussion
//! with no separate `Analysis`-based gate needed, and without a second Rust
//! variant to disambiguate: the two naming conventions (`"::"` present vs.
//! absent) never collide as strings, and nothing downstream needs a
//! type-level distinction between them, only the text itself.
//!
//! `CVal::Float` is representable now (`CleaveLang::Float`, wrapping
//! `ordered_float::OrderedFloat<f64>` — bare `f64` has neither `Ord` nor
//! `Hash`, both required by `define_language!`'s own derives) — added once
//! `doc/backlog.md`'s own auto-diff work actually needed a real float leaf
//! (`derivative(x,x) → 1.0`, `f32`/`f64`-typed). Deliberately narrow still:
//! `ConstantFold` stays `Int`-only (`Data = Option<u64>`) — a `Float` leaf
//! is representable and passes through a segment untouched, but arithmetic
//! over one (`2.0 * 3.0`) doesn't itself fold to `6.0` yet, a real, separate,
//! smaller follow-up once something actually needs it.

use egg::{Analysis, DidMerge, Id, Symbol};

// `define_language!` doesn't accept a doc comment on an individual variant
// (only the enum itself) — see the module's own doc comment above for what
// each of these actually means: `Op` is either an algebra-dispatched
// operation (`Symbol` = a `ConcreteUnit`'s own name, e.g. `"Ring::add<i32>"`)
// or a raw `mlir::...` call (`Symbol` = `"mlir-op:type"`); `Free` is a true
// free variable from outside the segment being translated, a synthetic
// per-segment symbol reverse-mapped back to its own original `CVal`
// elsewhere, never parsed.
egg::define_language! {
    pub enum CleaveLang {
        Op(Symbol, Vec<Id>),
        Int(u64), // matches `cps::CVal::Int(u64)`'s own representation exactly
        Float(ordered_float::OrderedFloat<f64>), // matches `cps::CVal::Float(f64)` -- wrapped for `Ord`/`Hash`, both required by `define_language!`'s own derives, bare `f64` has neither
        Bool(bool),
        Free(Symbol),
    }
}

/// Recovers the abstract, desugared operator name (`"add"`, `"mul"`, ...) —
/// `const_eval::eval_binop`'s own expected `op` argument — from a `Op`
/// node's own combined symbol, whichever of the two conventions produced it
/// (see the module's own doc comment on `CleaveLang::Op`): an algebra-
/// dispatched unit name's own method segment (`"Ring::add<i32>"` -> `"add"`
/// — a `ConcreteUnit`'s own method name, threaded straight through from
/// `origin`, already *is* the exact name `eval_binop` expects, since both
/// ultimately come from the same source-level desugaring, `ast.rs`'s own
/// `Call` doc comment), or a small, explicit lookup for the raw mlir op
/// names this module's own translator can actually produce today
/// (`stdlib/num`'s own bodies never use anything else) — grown one entry at
/// a time, the same discipline `eval_binop`'s own match already follows.
fn abstract_op_name(symbol: &str) -> Option<&str> {
    if let Some(method) = symbol.split("::").nth(1) {
        return method.split('<').next();
    }
    match symbol.split(':').next()? {
        "arith.addi" | "arith.addf" => Some("add"),
        "arith.muli" | "arith.mulf" => Some("mul"),
        _ => None,
    }
}

/// Folds a node whose own children are all already-known constants into a
/// literal `Int`, and propagates it — mirrors `examples/egg_toy.rs`'s own
/// `ConstantFold` in shape (the one piece of that toy worth reusing
/// directly), narrowed to `Int` only per this module's own current scope,
/// but delegates the actual arithmetic to `const_eval::eval_binop` — the
/// same evaluator `infer.rs` already uses to fold a *type-level* const
/// generic expression (`[T; N+M]`), reused here for a *runtime* value
/// instead of writing a second one from scratch (per this module's own
/// earlier note that this was worth checking before doing that). Constant
/// *propagation* to every later use needs nothing further here: once
/// `modify` unions a folded `Int` into an e-class, every reference to that
/// e-class already resolves to it through ordinary e-class sharing — the
/// ANF/CPS-variable-reuse mechanism the forward translator relies on for
/// copy-propagation is the exact same mechanism, not a separate pass.
/// `known_types` -- a `Free`/`Struct`/`Array` node's own concrete `Ty`,
/// keyed by the exact same `Symbol` its own `CleaveLang::Op`/`Free` node
/// uses, populated by `Forward` (a top-level function's own declared
/// parameter types for `Free`; a `LetPrim`'s own declared type for a
/// construction) *before* the corresponding node is ever added to the
/// e-graph -- read back by `make`'s own `Free`/`Op` arms below to seed
/// `FoldData::own_ty`. Lives here (an `Analysis` field, reachable from
/// `make` via `egraph.analysis`, since `Analysis::make` itself takes no
/// `&self`) rather than as a `build_zero`-local parameter, since `make`
/// needs it too. Deliberately *not* populated for every other node kind
/// (a raw `mlir::...` op, an algebra-dispatched call's own result, ...) --
/// `derivative-independent-zero`'s own custom `Applier` only ever *needs*
/// a real `Ty` for the two shapes it actually knows how to build a
/// same-shaped zero for (`build_zero`'s own doc comment); anywhere else,
/// `own_ty` staying `None` correctly means "this rule doesn't try to
/// shortcut here," falling back to full recursive chain-rule expansion
/// elsewhere in the same e-class -- a missed optimization, never unsound.
#[derive(Default, Clone)]
pub struct ConstantFold {
    known_types: HashMap<Symbol, Ty>,
}

/// `ConstantFold`'s own `Analysis::Data` — the existing int-fold value
/// (`const_int`) plus, since `depends_on_eclass`'s own bug (see `derivative-
/// independent-zero`'s doc comment), `free_deps`: the set of `Free`
/// variable symbols this e-class's value is *provably* bounded by. An
/// upper bound per representation (`make`'s own `Op` arm: the union of its
/// children's own bounds — a value built from `a`/`b` can only ever depend
/// on what `a`/`b` themselves depend on, nothing more), *narrowed by
/// intersection* whenever two representations of the *same* e-class get
/// merged (`merge`, below): if even one representation of an e-class proves
/// independence from `w` (an empty bound), the whole e-class, being value-
/// equal to every one of its own representations, provably is too —
/// regardless of how many *other*, less-reduced representations happen to
/// still mention `w` syntactically (`w * 0.0`, e.g. — value-independent of
/// `w`, but its own naive bound still contains `w`). This is exactly what a
/// *live* e-graph traversal (`depends_on_eclass`'s own former self) cannot
/// give soundly: the more saturation progresses, the more such "mentions
/// `w`, but doesn't truly depend on it" representations accumulate in
/// *every* e-class, making a live "does any representation mention `w`"
/// search increasingly, silently wrong. An ordinary `Analysis` field
/// avoids this the same way `const_int` already does for constant folding:
/// computed once per e-node, merged compositionally as e-classes combine,
/// never re-derived by walking the live, ever-growing graph.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FoldData {
    pub const_int: Option<u64>,
    pub free_deps: HashSet<Symbol>,
    /// This e-class's own concrete cleave `Ty`, when known (`ConstantFold::
    /// known_types`'s own doc comment) -- used by `derivative-independent-
    /// zero`'s own custom `Applier` (`build_zero`) to build a same-*shaped*
    /// zero rather than always a bare scalar one. `merge`'s own policy on
    /// disagreement (below) drops it to `None` rather than trusting either
    /// side or panicking -- a real, if never-yet-observed, safety margin:
    /// unlike `const_int`, two representations of one e-class disagreeing
    /// here would be a genuine type-soundness violation, not just a
    /// tolerable "haven't proven convergence yet" state, so this analysis
    /// doesn't assert it away.
    pub own_ty: Option<Ty>,
}

impl Analysis<CleaveLang> for ConstantFold {
    type Data = FoldData;

    fn make(egraph: &mut egg::EGraph<CleaveLang, Self>, enode: &CleaveLang, _id: Id) -> Self::Data {
        let const_int = (|| match enode {
            CleaveLang::Int(n) => Some(*n),
            CleaveLang::Op(op, args) => {
                let name = abstract_op_name(op.as_str())?;
                match args.as_slice() {
                    // `neg` is unary — folds via `eval_binop("sub", 0, a)`
                    // directly (`Ring<T>::neg`'s own real body already
                    // computes exactly this, `mlir::arith::subi(0, a)`, for
                    // every integer width) rather than inventing a separate
                    // `eval_unop` just for one operator. Safe the same way
                    // `add`/`mul`/`sub` already are: wrapping subtraction
                    // gives the identical `u64` bit pattern regardless of
                    // whether the operand is "meant" to be signed or
                    // unsigned.
                    [a] if name == "neg" => {
                        let a = crate::infer::ConstValue::Int(egraph[*a].data.const_int?);
                        match crate::const_eval::eval_binop("sub", crate::infer::ConstValue::Int(0), a)? {
                            crate::infer::ConstValue::Int(n) => Some(n),
                            crate::infer::ConstValue::Bool(_) => None,
                        }
                    }
                    // `div` is deliberately *not* folded here, unlike every
                    // other recognized operator — see `const_eval::eval_
                    // binop`'s own `"div"` arm doc comment: this analysis
                    // has no width/signedness tag to tell whether a real
                    // `Ring::div<T>` node's operands are meant as signed or
                    // unsigned, and division (unlike add/mul/sub/neg)
                    // genuinely differs between the two for a negative
                    // operand — folding it here regardless would risk a
                    // silent wrong answer, not just a missed optimization.
                    // `eval_binop`'s own `"div"` support stays real and
                    // used, just only by `infer.rs`'s const-generic
                    // evaluator, where an array-size operand is always
                    // non-negative so the ambiguity can't arise.
                    [_, _] if name == "div" => None,
                    [a, b] => {
                        let a = crate::infer::ConstValue::Int(egraph[*a].data.const_int?);
                        let b = crate::infer::ConstValue::Int(egraph[*b].data.const_int?);
                        match crate::const_eval::eval_binop(name, a, b)? {
                            crate::infer::ConstValue::Int(n) => Some(n),
                            crate::infer::ConstValue::Bool(_) => None,
                        }
                    }
                    _ => None,
                }
            }
            // `Float` deliberately doesn't fold, even over two known-Float
            // operands -- `const_int` stays `Int`-only, a real, separate
            // follow-up once something actually needs it (module's own doc
            // comment).
            CleaveLang::Float(_) | CleaveLang::Bool(_) | CleaveLang::Free(_) => None,
        })();
        let free_deps = match enode {
            CleaveLang::Free(sym) => std::iter::once(*sym).collect(),
            // `derivative(inner, x)` gets no special case here — `x` (a
            // `Free`) contributing itself, and `inner`'s own bound, to this
            // node's own naive bound is sound (an *unreduced* marker's own
            // eventual value can only depend on what `inner`/`x` already
            // do); the `merge` below is what actually keeps this from
            // polluting an e-class that *also* has a genuinely independent
            // representation.
            CleaveLang::Op(_, args) => args.iter().flat_map(|id| egraph[*id].data.free_deps.iter().copied()).collect(),
            CleaveLang::Int(_) | CleaveLang::Float(_) | CleaveLang::Bool(_) => HashSet::new(),
        };
        // `Free`'s own declared type (`Forward`'s own `param_types`, a
        // top-level function's own parameter types) and a construction's
        // own declared type (`Forward`'s own `LetPrim`-declared `ty`) both
        // reach here purely via `egraph.analysis.known_types`, keyed by
        // this exact node's own symbol -- populated *before* the node was
        // added (`ConstantFold`'s own doc comment). Every other node kind
        // (a raw `mlir::...` op, an algebra-dispatched call, ...)
        // deliberately stays `None` here -- see that same doc comment for
        // why that's sufficient, not a gap.
        let own_ty = match enode {
            CleaveLang::Free(sym) => egraph.analysis.known_types.get(sym).cloned(),
            CleaveLang::Op(sym, _) => egraph.analysis.known_types.get(sym).cloned(),
            CleaveLang::Int(_) | CleaveLang::Float(_) | CleaveLang::Bool(_) => None,
        };
        FoldData { const_int, free_deps, own_ty }
    }

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        let int_merge = egg::merge_option(&mut to.const_int, from.const_int, |a, b| {
            assert_eq!(*a, b, "constant-fold analysis disagreed with itself on the same e-class's own value");
            DidMerge(false, false)
        });
        let to_len = to.free_deps.len();
        let from_len = from.free_deps.len();
        let intersected: HashSet<Symbol> = to.free_deps.intersection(&from.free_deps).copied().collect();
        let new_len = intersected.len();
        to.free_deps = intersected;
        let ty_merge = match (&to.own_ty, from.own_ty) {
            (Some(t1), Some(t2)) if *t1 == t2 => DidMerge(false, false),
            (Some(_), Some(_)) => {
                to.own_ty = None; // a real type-soundness violation if this ever fires -- safer to drop than to trust either side
                DidMerge(true, true)
            }
            (Some(_), None) => DidMerge(false, false),
            (None, t2 @ Some(_)) => {
                to.own_ty = t2;
                DidMerge(true, false)
            }
            (None, None) => DidMerge(false, false),
        };
        int_merge | DidMerge(new_len != to_len, new_len != from_len) | ty_merge
    }

    fn modify(egraph: &mut egg::EGraph<CleaveLang, Self>, id: Id) {
        if let Some(n) = egraph[id].data.const_int {
            let added = egraph.add(CleaveLang::Int(n));
            egraph.union(id, added);
        }
    }
}

// ---------------------------------------------------------------- CPS -> e-graph (forward)

use crate::cps::{CExpr, CFunDef, CTopLevelFn, CVal, CVar, PrimOp};
use crate::infer::{ConstValue, Ty};
use egg::EGraph;
use std::collections::HashMap;

/// Whether `op` has no observable side effect of its own — safe to treat a
/// call whose own body is made only of ops like this as one opaque, freely
/// reorderable/CSE-able node (`is_straight_line` below). An *allowlist*, not
/// a denylist over `FieldStore`/`Store`/`Extern`: a future `PrimOp` variant
/// this doesn't yet know about defaults to `false` (a missed optimization)
/// rather than silently becoming unsoundly reorderable the moment someone
/// adds one and forgets to list it here. `Array`/`ArrayRepeat`/`Load` are
/// listed as pure even though `Forward::walk` doesn't yet translate them
/// directly (they aren't in *its* own match) — this only gates whether a
/// *call* to something containing them is safe to treat as one opaque node,
/// never requires descending into that callee's own body.
fn is_pure_prim_op(op: &PrimOp) -> bool {
    matches!(op, PrimOp::RawMlirOp { .. } | PrimOp::Field { .. } | PrimOp::Struct(..) | PrimOp::Array | PrimOp::ArrayRepeat | PrimOp::Load { .. })
}

/// Whether `expr`'s own body contains no real control flow (`Fix`/`If`) and
/// no `LetPrim` carrying a real effect (`is_pure_prim_op`) — a call to a
/// unit whose own body has this shape can be treated *transparently* by
/// `Forward::walk` below: calling it is semantically indistinguishable from
/// inlining its own one computed value directly, so translation can keep
/// building the *same* e-graph straight through the call, rather than
/// treating it as a segment boundary. Deliberately conservative in one
/// direction: a unit that itself calls *another* pure unit is rejected too
/// (its own body contains a `Fix`, from that nested call), even though it's
/// also "really" pure — multi-level transparency is a natural follow-up,
/// not needed for this module's own first target (`Ring::add<i32>`, a
/// single raw `mlir::...` call, no nested calls at all).
///
/// The effect check is not optional: a callee shaped `LetPrim{op: Extern
/// {..}} -> App` (exactly `Print<T>::print`'s own real body — no `Fix`/`If`
/// at all) would otherwise be judged straight-line, and `Forward::walk`
/// would fold calling it into one opaque, freely-reorderable `Op` node —
/// genuinely unsound, printing has an observable, order-dependent effect.
/// Found by direct testing while extending this module to translate
/// `Struct`/`Field` further, which makes translation reach a real `print`
/// call for the first time.
fn is_straight_line(expr: &CExpr) -> bool {
    match expr {
        CExpr::LetPrim { op, cont, .. } => is_pure_prim_op(op) && is_straight_line(cont),
        CExpr::App { .. } => true,
        CExpr::Fix { .. } | CExpr::If { .. } => false,
    }
}

/// Multi-level call transparency (`doc/backlog.md`'s own item, the "natural
/// follow-up" `is_straight_line`'s own doc comment above already names) —
/// whether `expr` is a *chain* of real calls to other units, each of which
/// is itself either `is_straight_line` (a single primop — `Ring::add<f32>`)
/// or (recursively) another such chain, with no loop/branch/effect
/// anywhere. `Forward::walk`'s own `Fix` arm uses this to decide whether to
/// walk *into* a callee's own body (true multi-level inlining) rather than
/// collapsing it into one opaque `Op` node (`is_straight_line`'s own,
/// single-level case — still the right call for a genuinely atomic unit:
/// its own identity as a *named* op is exactly what a declared axiom/
/// derivative rule needs to match against, which is why the two cases stay
/// distinct rather than merging into one check). `visiting` guards against
/// two units transitively calling each other (mutual recursion would
/// otherwise recurse forever here, before `Forward::walk` ever gets a
/// chance to see real recursion and reject it the ordinary way).
fn is_transparent_chain(expr: &CExpr, units: &HashMap<String, &CTopLevelFn>, visiting: &mut HashSet<String>) -> bool {
    match expr {
        CExpr::LetPrim { op, cont, .. } => is_pure_prim_op(op) && is_transparent_chain(cont, units, visiting),
        CExpr::App { .. } => true,
        CExpr::If { .. } => false,
        CExpr::Fix { defs, body } => {
            let Some((_, unit_name, _, rest)) = recognize_real_call(defs, body) else { return false };
            let Some(callee) = units.get(unit_name) else { return false };
            if is_straight_line(&callee.def.body) {
                return is_transparent_chain(rest, units, visiting);
            }
            if !visiting.insert(unit_name.to_string()) {
                return false; // already being checked higher up this same call chain -- mutually recursive, reject
            }
            // `unit_name` comes back out of `visiting` *before* checking
            // `rest` — real, found-by-testing bug in an earlier version:
            // keeping it in scope across both checks made a second,
            // sequential *sibling* call to the very same unit (`sigmoid`
            // called twice inside one `forward`, neither call nested inside
            // the other) look identical to genuine mutual recursion and get
            // wrongly rejected — `visiting` must only ever track units
            // actually on the current *ancestor* chain, not ones already
            // fully checked and returned from.
            let body_ok = is_transparent_chain(&callee.def.body, units, visiting);
            visiting.remove(unit_name);
            body_ok && is_transparent_chain(rest, units, visiting)
        }
    }
}

/// Forward CPS-segment-to-e-graph translation state — one instance per
/// segment being translated. `env` records, for every `LetPrim`-bound
/// `CVar` successfully translated so far, which e-class it now corresponds
/// to (the mechanism this module's own doc comment calls "copy-propagation/
/// CSE for free": a *later* reference to that same `CVar` reuses this same
/// `Id` directly, rather than being treated as a fresh, opaque value).
pub struct Forward {
    pub egraph: EGraph<CleaveLang, ConstantFold>,
    pub env: HashMap<CVar, egg::Id>,
    /// A true free variable's own synthetic symbol -> its original `CVal`,
    /// so a later reconstruction stage can put the *original* value back
    /// (not the symbol text, which was never meant to be parsed) — see the
    /// module's own doc comment on `Free`.
    pub free_vars: HashMap<Symbol, CVal>,
    /// The reverse half of `free_vars`, for the one case a caller needs to
    /// go the other way: given a *specific* external `CVar` (a function's
    /// own parameter, say — never itself `LetPrim`-bound, so never an `env`
    /// key — see that field's own doc comment), which e-class does its
    /// `Free` node live in, if it was ever actually referenced at all.
    /// Populated alongside `free_vars`, only for `CVal::Var` references
    /// (the only `CVal` shape with a `CVar` identity to key on at all) —
    /// found necessary directly: `synthesize_derivatives` needs exactly
    /// this to build one `derivative(root, param)` node per one of `f`'s
    /// own parameters, and reading `env` instead (an earlier attempt)
    /// silently produced `None` for every parameter, since a parameter is
    /// never `LetPrim`-bound.
    pub external_vars: HashMap<CVar, egg::Id>,
    /// Every algebra-dispatched unit actually inlined transparently while
    /// building this e-graph (unit name -> its own `origin`) — a later
    /// axiom-to-`Rewrite` translation stage (Stage 4) needs exactly this
    /// set: "which concrete instantiations were actually reached" is what
    /// tells it which concrete `Rewrite`s are even worth building (see the
    /// module's own doc comment on why one generic rewrite per axiom isn't
    /// possible at all). A plain top-level `fn` (`origin: None`) is never
    /// recorded here — no axiom could ever reference one.
    pub reached: HashMap<String, (String, String)>,
    /// Every unit name used via the *real-call* path (`recognize_real_call`)
    /// — a superset of `reached`'s own keys (this includes a transparently-
    /// inlined plain top-level `fn` too, `origin: None`, which `reached`
    /// itself deliberately excludes — see its own doc comment). A later
    /// backward-reconstruction stage (Stage 5) needs this specifically to
    /// know, for a given `Op` node's own symbol, whether to rebuild it as a
    /// real call (`Fix`/`App`) or a raw `mlir::...` op (`LetPrim`) —
    /// without it, telling the two apart would mean guessing from the
    /// symbol's own text shape, exactly the kind of parsing this module has
    /// deliberately avoided everywhere else.
    pub call_units: std::collections::HashSet<String>,
    /// A raw `mlir::...` op's own combined symbol -> its original `(mlir op
    /// name, concrete type, attrs)` — mirrors `free_vars`'s own reasoning
    /// exactly (a later reconstruction stage needs the *original* pieces
    /// back, not the combined text, which was never meant to be parsed).
    pub raw_ops: HashMap<Symbol, (String, Ty, Vec<(String, String)>)>,
    /// A struct-construction `Op`'s own combined symbol -> its original
    /// `(struct name, field names in the order this construction's own args
    /// are in, the constructed value's own concrete `Ty`)` — mirrors
    /// `raw_ops`'s own reasoning exactly.
    pub struct_ops: HashMap<Symbol, (String, Vec<String>, Ty)>,
    /// A field-read `Op`'s own combined symbol -> its original `(the base's
    /// own concrete struct `Ty`, the field name, the read value's own
    /// concrete `Ty`)` — mirrors `raw_ops`'s own reasoning exactly.
    pub field_ops: HashMap<Symbol, (Ty, String, Ty)>,
    /// An array-literal `Op`'s own combined symbol -> the constructed
    /// array's own concrete `Ty` — mirrors `struct_ops`'s own reasoning
    /// (no name/field-list of its own to carry, an array has neither).
    pub array_ops: HashMap<Symbol, Ty>,
    /// An `[value; count]` `Op`'s own combined symbol -> the constructed
    /// array's own concrete `Ty` — mirrors `array_ops`'s own reasoning.
    pub array_repeat_ops: HashMap<Symbol, Ty>,
    /// A (possibly multi-index) array-read `Op`'s own combined symbol ->
    /// its original `(the base's own concrete array `Ty`, the read value's
    /// own concrete `Ty`)` — mirrors `field_ops`'s own reasoning exactly.
    pub load_ops: HashMap<Symbol, (Ty, Ty)>,
    /// A top-level function's own real parameters (`CTopLevelFn::def.
    /// params`, excluding the trailing `k_ret`) -> their own declared `Ty`
    /// (`CTopLevelFn::param_types`, same order) -- set by the caller
    /// (`synthesize_derivatives`/`optimize_program`) right after `Forward::
    /// default()`, before `walk` ever runs. The *only* source `cval_to_id`'s
    /// own `Free`-minting branch has for a free variable's own type
    /// (`ConstantFold::known_types`'s own doc comment) -- every `Free` node
    /// this translation ever mints traces back to one of `f`'s own
    /// parameters (nothing else is free at this level, `walk` only ever
    /// being called on one whole top-level function's own body), so this
    /// one map is sufficient, no per-callee threading needed even through
    /// multi-level call transparency's own inlining (an inlined callee's
    /// own parameters are bound via `self.env`, never minted as `Free`).
    pub param_types: HashMap<CVar, Ty>,
    next_free: u32,
}

impl Default for Forward {
    fn default() -> Self {
        Self {
            egraph: EGraph::default(),
            env: HashMap::new(),
            free_vars: HashMap::new(),
            external_vars: HashMap::new(),
            reached: HashMap::new(),
            call_units: std::collections::HashSet::new(),
            raw_ops: HashMap::new(),
            struct_ops: HashMap::new(),
            field_ops: HashMap::new(),
            array_ops: HashMap::new(),
            array_repeat_ops: HashMap::new(),
            load_ops: HashMap::new(),
            param_types: HashMap::new(),
            next_free: 0,
        }
    }
}

impl Forward {
    /// Translates as much of `expr`'s own straight-line prefix as possible,
    /// returning the `CExpr` where it had to stop — the "boundary" this
    /// module's own doc comment (and the plan it was built against) both
    /// name. Unchanged (a clone of `expr` itself, or a sub-expression of
    /// it) if nothing could be translated at all — never partially rewrites
    /// what it stops at. Every `CVar` successfully translated along the way
    /// is recorded in `self.env`; the boundary's own remaining `CVal::Var`
    /// references (into the segment, or genuinely free) are resolved the
    /// identical way a *later* reference inside the segment would be.
    pub(crate) fn walk(&mut self, expr: &CExpr, units: &HashMap<String, &CTopLevelFn>, fresh: &FreshVars) -> CExpr {
        match expr {
            CExpr::LetPrim { var, ty, op: PrimOp::RawMlirOp { op, attrs }, args, cont } => {
                let Some(arg_ids) = self.cvals_to_ids(args) else {
                    return expr.clone(); // an unrepresentable argument (Unit/Float/Label/Closure) -- stop here, unchanged
                };
                // `attrs` matters to the op's own real meaning (`cmpi`'s own
                // `lt` vs `gt` predicate, say) -- folded into the same
                // combined symbol as the mlir op name/type, or two
                // semantically different ops would wrongly share one node
                // identity.
                let symbol = format!("{op}:{ty}:{attrs:?}");
                let sym = Symbol::from(symbol);
                self.raw_ops.entry(sym).or_insert_with(|| (op.clone(), ty.clone(), attrs.clone()));
                self.egraph.analysis.known_types.entry(sym).or_insert_with(|| ty.clone());
                let id = self.egraph.add(CleaveLang::Op(sym, arg_ids));
                self.env.insert(*var, id);
                self.walk(cont, units, fresh)
            }
            // Construction and a field read are both pure (no aliasing
            // effect from either alone -- see `mlir_lower.rs::alloc_struct`'s
            // own doc comment: a real heap allocation, but nothing yet reads
            // or writes through it here) -- translated the identical way
            // `RawMlirOp` is above. `ty`/`struct_ty` are folded into each
            // symbol for the same reason `RawMlirOp`'s own `ty` is: two
            // structurally-different-typed struct ops must never wrongly
            // share one node identity.
            CExpr::LetPrim { var, ty, op: PrimOp::Struct(struct_name, field_names), args, cont } => {
                let Some(arg_ids) = self.cvals_to_ids(args) else {
                    return expr.clone();
                };
                let symbol = format!("struct:{ty}:{}", field_names.join(","));
                let sym = Symbol::from(symbol);
                self.struct_ops.entry(sym).or_insert_with(|| (struct_name.clone(), field_names.clone(), ty.clone()));
                self.egraph.analysis.known_types.entry(sym).or_insert_with(|| ty.clone());
                let id = self.egraph.add(CleaveLang::Op(sym, arg_ids));
                self.env.insert(*var, id);
                self.walk(cont, units, fresh)
            }
            CExpr::LetPrim { var, ty, op: PrimOp::Field { struct_ty, field }, args, cont } => {
                let Some(arg_ids) = self.cvals_to_ids(args) else {
                    return expr.clone();
                };
                let symbol = format!("field:{struct_ty}:{field}");
                let sym = Symbol::from(symbol);
                self.field_ops.entry(sym).or_insert_with(|| (struct_ty.clone(), field.clone(), ty.clone()));
                self.egraph.analysis.known_types.entry(sym).or_insert_with(|| ty.clone());
                let id = self.egraph.add(CleaveLang::Op(sym, arg_ids));
                self.env.insert(*var, id);
                self.walk(cont, units, fresh)
            }
            // Construction/read, mirroring `Struct`/`Field` above exactly --
            // an array is also a stable heap reference (`cps.rs::PrimOp::
            // Store`'s own doc comment: "a stable array reference sidesteps
            // Stage 4's own mutation-threading"), so the identical
            // "construction/read alone has no aliasing effect, only a real
            // mutation (`Store`, excluded) does" reasoning applies -- the
            // `segment_root_var`'s own "exactly one referenced segment var"
            // rule already closes the aliasing question generically, not
            // just for structs.
            CExpr::LetPrim { var, ty, op: PrimOp::Array, args, cont } => {
                let Some(arg_ids) = self.cvals_to_ids(args) else {
                    return expr.clone();
                };
                let sym = Symbol::from(format!("array:{ty}:{}", args.len()));
                self.array_ops.entry(sym).or_insert_with(|| ty.clone());
                self.egraph.analysis.known_types.entry(sym).or_insert_with(|| ty.clone());
                let id = self.egraph.add(CleaveLang::Op(sym, arg_ids));
                self.env.insert(*var, id);
                self.walk(cont, units, fresh)
            }
            CExpr::LetPrim { var, ty, op: PrimOp::ArrayRepeat, args, cont } => {
                let Some(arg_ids) = self.cvals_to_ids(args) else {
                    return expr.clone();
                };
                let sym = Symbol::from(format!("array-repeat:{ty}"));
                self.array_repeat_ops.entry(sym).or_insert_with(|| ty.clone());
                self.egraph.analysis.known_types.entry(sym).or_insert_with(|| ty.clone());
                let id = self.egraph.add(CleaveLang::Op(sym, arg_ids));
                self.env.insert(*var, id);
                self.walk(cont, units, fresh)
            }
            CExpr::LetPrim { var, ty, op: PrimOp::Load { array_ty }, args, cont } => {
                let Some(arg_ids) = self.cvals_to_ids(args) else {
                    return expr.clone();
                };
                // Index count is folded in too -- `a[i]` and `a[i,j]` on the
                // *same* array type are genuinely different reads (`args.len()`
                // already reflects this in the node's own children count, but
                // the symbol carries it explicitly too, matching every other
                // op in this module: never rely on children arity alone to
                // disambiguate what a symbol means).
                let sym = Symbol::from(format!("load:{array_ty}:{ty}:{}", args.len()));
                self.load_ops.entry(sym).or_insert_with(|| (array_ty.clone(), ty.clone()));
                self.egraph.analysis.known_types.entry(sym).or_insert_with(|| ty.clone());
                let id = self.egraph.add(CleaveLang::Op(sym, arg_ids));
                self.env.insert(*var, id);
                self.walk(cont, units, fresh)
            }
            // Any other `PrimOp` (`FieldStore`/`Store`/`Extern`) is a real
            // mutation effect or an external call, never freely reorderable/
            // foldable the way pure arithmetic is -- stop, unchanged, same
            // as any other unrecognized shape.
            CExpr::Fix { defs, body } => {
                if let Some(unrolled) = self.try_unroll_for_loop(defs, body, units, fresh) {
                    return unrolled;
                }
                if let Some((result_var, unit_name, real_args, rest)) = recognize_real_call(defs, body) {
                    if let Some(callee) = units.get(unit_name) {
                        if is_straight_line(&callee.def.body) {
                            if let Some(arg_ids) = self.cvals_to_ids(real_args) {
                                let id = self.egraph.add(CleaveLang::Op(unit_name.into(), arg_ids));
                                self.env.insert(result_var, id);
                                self.call_units.insert(unit_name.to_string());
                                if let Some(origin) = &callee.origin {
                                    self.reached.insert(unit_name.to_string(), origin.clone());
                                }
                                return self.walk(rest, units, fresh);
                            }
                        } else if is_transparent_chain(&callee.def.body, units, &mut HashSet::new()) {
                            // Multi-level transparency (`doc/backlog.md`'s
                            // own item) — `callee`'s own body isn't a single
                            // primop (so it can't become one opaque `Op`
                            // node the way `Ring::add<f32>` does — nothing
                            // declares an axiom/derivative rule keyed by
                            // *its* own name), but it's still a pure chain
                            // of further real calls, no loop/branch/effect
                            // anywhere in it — walking straight through it
                            // is semantically identical to inlining its own
                            // body at this call site. `callee`'s own body is
                            // alpha-renamed first (`fresh`, shared across
                            // this *whole* translation, never a fresh-per-
                            // call-site instance) — the *same* callee
                            // inlined more than once (`sigmoid(a)` and
                            // `sigmoid(b)` in the same loss function) must
                            // not let its own internal temporaries alias
                            // across the two call sites in `self.env`,
                            // exactly the reason `try_unroll_for_loop`
                            // above needs identical treatment for repeated
                            // loop iterations.
                            if let Some(arg_ids) = self.cvals_to_ids(real_args) {
                                let ordinary_params = &callee.def.params[..callee.def.params.len().saturating_sub(1)];
                                let mut map: HashMap<CVar, CVar> = HashMap::new();
                                let renamed_params: Vec<CVar> = ordinary_params.iter().map(|_| fresh.var()).collect();
                                for (&orig, &renamed) in ordinary_params.iter().zip(&renamed_params) {
                                    map.insert(orig, renamed);
                                }
                                let renamed_body = Self::alpha_rename(&callee.def.body, &mut map, fresh);
                                for (&renamed, &id) in renamed_params.iter().zip(&arg_ids) {
                                    self.env.insert(renamed, id);
                                }
                                if let CExpr::App { args: ret_args, .. } = self.walk(&renamed_body, units, fresh) {
                                    if let Some((ret_val, _)) = ret_args.split_last() {
                                        if let Some(ret_id) = self.cval_to_id(ret_val) {
                                            self.env.insert(result_var, ret_id);
                                            return self.walk(rest, units, fresh);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                expr.clone() // not `emit_call`'s own real-call shape, or the callee isn't pure/known -- stop, unchanged
            }
            // Any other `PrimOp` (`FieldStore`/`Store`/`Extern`) — a real
            // mutation effect or an external call, never freely
            // reorderable/foldable the way pure arithmetic is. `App`/`If` —
            // a plain tail-return or real control flow. All stop here,
            // unchanged.
            CExpr::LetPrim { .. } | CExpr::App { .. } | CExpr::If { .. } => expr.clone(),
        }
    }

    fn cvals_to_ids(&mut self, vals: &[CVal]) -> Option<Vec<egg::Id>> {
        vals.iter().map(|v| self.cval_to_id(v)).collect()
    }

    fn cval_to_id(&mut self, v: &CVal) -> Option<egg::Id> {
        match v {
            // `env` caches a `LetPrim`-bound var's own e-class; `external_
            // vars` caches an *external* one's (a function's own
            // parameter, say) — checked here too, not just exposed as a
            // side table, or the *same* external `CVar` referenced more
            // than once within one segment would mint a *different* `Free`
            // node each time (a distinct synthetic symbol per occurrence,
            // no natural congruence between them) — silently modeling `x`
            // used twice as two unrelated values. Found directly, by
            // testing `derive`: `x * x`'s own two (identical) argument
            // references used to become `Free(fv0)`/`Free(fv1)`, two
            // different e-classes, which `derivative(?x,?x) -> 1`'s own
            // pattern match (requiring the *same* e-class in both
            // positions) could then only ever satisfy for *one* of the two
            // occurrences a rule like the product rule's own RHS
            // introduces — silently computing the wrong derivative instead
            // of erroring. A pre-existing gap in this module's own design,
            // not specific to auto-diff: nothing in the axiom/constant-fold
            // rule set that existed before this needed "the same external
            // var, referenced twice, really is one value" to hold.
            CVal::Var(cv) => Some(match self.env.get(cv).or_else(|| self.external_vars.get(cv)) {
                Some(&id) => id,
                None => {
                    let sym = Symbol::from(format!("fv{}", self.next_free));
                    self.next_free += 1;
                    self.free_vars.insert(sym, v.clone());
                    // `own_ty` (`ConstantFold::known_types`'s own doc
                    // comment) needs this recorded *before* `egraph.add`
                    // below -- `Analysis::make` reads it back while the
                    // node is being inserted.
                    if let Some(ty) = self.param_types.get(cv) {
                        self.egraph.analysis.known_types.insert(sym, ty.clone());
                    }
                    let id = self.egraph.add(CleaveLang::Free(sym));
                    self.external_vars.insert(*cv, id);
                    id
                }
            }),
            CVal::Int(n) => Some(self.egraph.add(CleaveLang::Int(*n))),
            CVal::Float(f) => Some(self.egraph.add(CleaveLang::Float((*f).into()))),
            CVal::Bool(b) => Some(self.egraph.add(CleaveLang::Bool(*b))),
            // `Unit` has no e-graph representation (nothing to compute);
            // `Label`/`Closure` never appear as an ordinary computed
            // argument. Neither is translatable here.
            CVal::Unit | CVal::Label(_) | CVal::Closure { .. } => None,
        }
    }

    /// Recognizes `cps.rs`'s own `ExprKind::For` lowering shape (`cps.rs`'s
    /// own `for` arm, ground-truthed via `--dump-cps` on `for i in 0..3 {
    /// acc = acc + i; }`: a self-recursive `Fix` whose one `CFunDef` carries
    /// `[i_var, ...carried_params]`, `carried_types: Some(_)` — never
    /// `recognize_real_call`-shaped, which is exactly why every `for`/`while`
    /// loop stops `Forward::walk` dead today, unconditionally) and, only
    /// when both bounds are literal `CVal::Int`s within `MAX_UNROLL_
    /// ITERATIONS`, mechanically unrolls it: each iteration is walked with
    /// `i_var` bound to that iteration's own literal e-class and the carried
    /// vars bound to whatever e-classes the *previous* iteration actually
    /// produced, splicing straight into the next iteration's own copy of the
    /// body (or the loop's own exit continuation, on the last one) instead
    /// of stopping at the recursive tail call. Returns `None` — bail,
    /// caller falls back to today's "stop, unchanged" behavior — the moment
    /// any expected piece of this shape doesn't hold; never partially
    /// unrolls. A non-loop `Fix` (`carried_types: None`) is explicitly
    /// rejected here so `recognize_real_call`'s own real-call path is
    /// untouched.
    ///
    /// Alpha-renames every *internally* bound `CVar` (a `LetPrim`'s own
    /// `var`, a nested `Fix`'s own `CFunDef::params`) throughout `expr` to a
    /// fresh one from `fresh`, threading the growing old-to-new map through
    /// recursively so every reference sees its own binder's fresh name — a
    /// reference to a var *not* found in `map` (the loop's own `i_var`/
    /// carried params, a function parameter, anything bound *outside*
    /// `expr`) is left untouched, exactly `substitute_var`'s own fallback.
    /// Needed because `try_unroll_for_loop` walks the *same* `then_branch`
    /// once per iteration: without this, two different iterations' own
    /// internal computations (a product-rule/sum-rule expansion's own
    /// intermediate `derivative(...)` sub-nodes, specifically) can end up
    /// hash-consed onto each other through nothing but coincidentally
    /// reused `CVar` numbers, unioning genuinely different values —
    /// confirmed directly, empirically: without this, a loop mixing two
    /// different kinds of per-iteration ops (`add` then `load`, `derive_of_
    /// a_function_containing_a_statically_bounded_for_loop_computes_the_
    /// correct_derivative`'s own regression shape) produced a real,
    /// self-referential cycle in the saturated e-graph (an e-class unioned
    /// with `add(itself, something)`), permanently stuck.
    fn alpha_rename(expr: &CExpr, map: &mut HashMap<CVar, CVar>, fresh: &FreshVars) -> CExpr {
        let rename = |v: &CVal, map: &HashMap<CVar, CVar>| match v {
            CVal::Var(cv) => CVal::Var(*map.get(cv).unwrap_or(cv)),
            other => other.clone(),
        };
        match expr {
            CExpr::LetPrim { var, ty, op, args, cont } => {
                let new_args = args.iter().map(|a| rename(a, map)).collect();
                let new_var = fresh.var();
                map.insert(*var, new_var);
                CExpr::LetPrim { var: new_var, ty: ty.clone(), op: op.clone(), args: new_args, cont: Box::new(Self::alpha_rename(cont, map, fresh)) }
            }
            CExpr::App { func, args } => CExpr::App { func: rename(func, map), args: args.iter().map(|a| rename(a, map)).collect() },
            CExpr::Fix { defs, body } => {
                let new_defs = defs
                    .iter()
                    .map(|d| {
                        let mut inner_map = map.clone();
                        let new_params: Vec<CVar> = d
                            .params
                            .iter()
                            .map(|&p| {
                                let np = fresh.var();
                                inner_map.insert(p, np);
                                np
                            })
                            .collect();
                        CFunDef { name: d.name.clone(), params: new_params, body: Self::alpha_rename(&d.body, &mut inner_map, fresh), carried_types: d.carried_types.clone() }
                    })
                    .collect();
                CExpr::Fix { defs: new_defs, body: Box::new(Self::alpha_rename(body, map, fresh)) }
            }
            CExpr::If { cond, then_branch, else_branch } => {
                CExpr::If { cond: rename(cond, map), then_branch: Box::new(Self::alpha_rename(then_branch, map, fresh)), else_branch: Box::new(Self::alpha_rename(else_branch, map, fresh)) }
            }
        }
    }

    fn try_unroll_for_loop(&mut self, defs: &[CFunDef], body: &CExpr, units: &HashMap<String, &CTopLevelFn>, fresh: &FreshVars) -> Option<CExpr> {
        let [loop_def] = defs else { return None };
        loop_def.carried_types.as_ref()?;
        let CExpr::App { func: CVal::Label(call_label), args: init_args } = body else { return None };
        if call_label != &loop_def.name {
            return None;
        }
        let (&i_var, carried_params) = loop_def.params.split_first()?;
        let (start_val, carried_init) = init_args.split_first()?;
        let &CVal::Int(start) = start_val else { return None };

        let CExpr::Fix { defs: cond_defs, body: cond_body } = &loop_def.body else { return None };
        let (_, _, cmp_args, if_expr) = recognize_real_call(cond_defs, cond_body)?;
        let [_, CVal::Int(end)] = cmp_args else { return None };
        let end = *end;
        let CExpr::If { then_branch, else_branch, .. } = if_expr else { return None };
        if end.saturating_sub(start) > MAX_UNROLL_ITERATIONS {
            return None;
        }

        let mut carried_ids = self.cvals_to_ids(carried_init)?;
        for idx in start..end {
            let i_id = self.egraph.add(CleaveLang::Int(idx));
            self.env.insert(i_var, i_id);
            for (&p, &id) in carried_params.iter().zip(&carried_ids) {
                self.env.insert(p, id);
            }
            let renamed = Self::alpha_rename(then_branch, &mut HashMap::new(), fresh);
            let CExpr::App { func: CVal::Label(l), args } = self.walk(&renamed, units, fresh) else { return None };
            if l != loop_def.name {
                return None;
            }
            carried_ids = self.cvals_to_ids(&args[1..])?;
        }
        for (&p, &id) in carried_params.iter().zip(&carried_ids) {
            self.env.insert(p, id);
        }
        Some(self.walk(else_branch, units, fresh))
    }
}

/// Cap on how many concrete copies `Forward::try_unroll_for_loop` will ever
/// generate for one loop — a deliberately conservative constant, easy to
/// raise once real-world use justifies it: past this, unrolling would trade
/// a clean "stop, unchanged" bail for an e-graph blow-up (equality
/// saturation's own memory/time cost grows with node count), which is worse
/// than just not unrolling.
const MAX_UNROLL_ITERATIONS: u64 = 1024;

/// Recognizes `cps.rs::emit_call`'s own exact `UnitBody::Real` shape --
/// `Fix{defs: [k], body: App{func: Label(unit_name), args}}` where `k`'s
/// own single param is the call's result and `args`'s own last entry names
/// that same `k` as the continuation to resume -- returning `(result_var,
/// unit_name, real_args, k's own body)` on a match. `carried_types: None`
/// on `k` is part of the match: an `if`-join/loop's own synthesized
/// continuation always sets `Some(_)` there instead (see `CFunDef::
/// carried_types`'s own doc comment), so this can't be confused with either.
fn recognize_real_call<'a>(defs: &'a [CFunDef], body: &'a CExpr) -> Option<(CVar, &'a str, &'a [CVal], &'a CExpr)> {
    let [CFunDef { name: k_name, params, body: rest, carried_types: None }] = defs else { return None };
    let [result_var] = params.as_slice() else { return None };
    let CExpr::App { func: CVal::Label(unit_name), args } = body else { return None };
    let (last, real_args) = args.split_last()?;
    if !matches!(last, CVal::Label(l) if l == k_name) {
        return None;
    }
    Some((*result_var, unit_name.as_str(), real_args, rest))
}

// ---------------------------------------------------------------- axiom -> Rewrite

use crate::ast::{AxiomDecl, DerivativeRuleDecl, Expr, ExprKind};
use crate::registry::Registry;
use egg::{ENodeOrVar, PatternAst, Rewrite, Var};
use std::collections::HashSet;

/// Builds one concrete `Rewrite` per `(axiom, reached concrete type)` pair,
/// for every axiom declared on whichever algebra each `reached` unit
/// belongs to — see the module's own doc comment on `CleaveLang::Op` for
/// why a single *generic* rewrite per axiom can't be built at all: egg's
/// own pattern matching is exact-string on a node's own discriminant, and
/// `Op`'s discriminant is always a concrete unit name, never an abstract
/// one. `reached` — `Forward::reached` — is exactly "every algebra-
/// dispatched unit actually inlined while building the e-graph being
/// optimized"; only concrete types actually present are worth building a
/// rule for.
///
/// **A stated scope limit for this first increment**, not an oversight:
/// this assumes an axiom's own body only ever calls methods of the *same*
/// algebra it's declared on (true of every axiom this project has written
/// so far, e.g. `Ring<T>`'s own `add_commutative`) — a call inside the
/// axiom body is substituted as `"{that algebra}::{method}<{reached
/// type}>"` unconditionally, no cross-algebra resolution attempted. A
/// substitution that doesn't correspond to any unit actually reached is
/// harmless, not unsound: the resulting rule simply never matches anything
/// in this particular e-graph.
pub fn axiom_rewrites(registry: &Registry, reached: &HashMap<String, (String, String)>) -> Vec<Rewrite<CleaveLang, ConstantFold>> {
    let mut reached_types: HashMap<&str, HashSet<&str>> = HashMap::new();
    for (unit_name, (algebra, _method)) in reached {
        if let Some(ty) = concrete_type_of(unit_name) {
            reached_types.entry(algebra.as_str()).or_default().insert(ty);
        }
    }

    let mut rules = Vec::new();
    for (algebra, types) in &reached_types {
        for axiom in registry.axioms(algebra) {
            for ty in types {
                if let Some(rw) = axiom_to_rewrite(algebra, ty, axiom, registry) {
                    rules.push(rw);
                }
            }
        }
    }
    rules
}

/// Extracts the bracketed type argument from a unit's own display name
/// (`"Ring::add<i32>"` -> `"i32"`) — a real, if narrow, parse (unlike every
/// other place in this module, which deliberately avoids parsing a unit
/// name at all — see `ConcreteUnit::origin`'s own doc comment). Nothing
/// structurally carries "the concrete type" on its own the way `origin`
/// carries `(algebra, method)`, and building that out for the general,
/// possibly-multi-generic case (`MatMul<A, B, C>`) is more machinery than
/// this first increment (scoped to single-generic algebras like `Ring<T>`)
/// needs — a mismatch here (nested generics, several type arguments) just
/// means no rewrite gets built for that unit, a missed optimization, never
/// a wrong one.
fn concrete_type_of(unit_name: &str) -> Option<&str> {
    let start = unit_name.find('<')? + 1;
    let end = unit_name.rfind('>')?;
    (start < end).then(|| &unit_name[start..end])
}

/// Splits a multi-target algebra's own combined type string (`concrete_
/// type_of`'s own doc comment: `"Tensor<f32, 2>, f32"` for `Index<Container,
/// Elem, K>`) back into its individual pieces, on *top-level* commas only —
/// a bare `.split(", ")` would wrongly cut inside a nested generic's own
/// argument list (`Tensor<f32, 2>` itself contains a `, `). Mirrors exactly
/// how the combined string was built in the first place (`cps.rs`'s own
/// `targets_str = infer.target_types.iter().map(Ty::to_string).collect::
/// <Vec<_>>().join(", ")`), just run in reverse.
fn split_top_level_commas(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, c) in s.char_indices() {
        match c {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(s[start..i].trim());
                start = i + c.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(s[start..].trim());
    parts
}

/// A multi-target algebra's own declared generic *names*, in order
/// (`MatMul<A, B, C>` -> `["A", "B", "C"]`), matched positionally against
/// `split_top_level_commas(ty)` -- the substitution a `derivative`/`axiom`
/// rule's own body needs to know which of its own params (declared, in the
/// algebra's own `fn` signature, against generic names like `A`/`B`/`C`)
/// resolves to which *concrete* type at this particular reached
/// instantiation. Only `GenericParam::Type` entries participate -- a
/// `const` generic (`Index<Container, Elem, const K: i32>`'s own `K`) is
/// never part of the combined type-target string in the first place (`cps.
/// rs`'s own `target_types`, built separately from const-generic
/// substitution). Collapses to one trivial entry for the overwhelmingly
/// common single-generic algebra (`Ring<T>`) -- `ty` unchanged, exactly the
/// flat string every axiom/derivative rule already assumed before this.
fn generic_substitution(algebra: &str, ty: &str, registry: &Registry) -> HashMap<String, String> {
    let names = registry.generics(algebra).iter().filter(|g| !matches!(g, crate::ast::GenericParam::Const { .. })).map(|g| g.name());
    names.zip(split_top_level_commas(ty)).map(|(name, part)| (name.to_string(), part.to_string())).collect()
}

/// Resolves one *declared* type (from an algebra's own `fn` signature --
/// always written against that algebra's own generic names, e.g. `fn matmul
/// (a: A, b: B) -> C;`) to its concrete value at one reached instantiation,
/// via `subst` (`generic_substitution`'s own output). Only ever succeeds
/// when the declared type is *exactly* one bare generic name (every
/// existing algebra signature in this codebase's own stdlib is written this
/// way) -- a more structured declared type (`Index<Container,Elem,K>`'s own
/// `idx: [i32; K]`) isn't resolved here, `None` rather than a guess, the
/// same conservative posture `build_pattern` already takes throughout.
fn resolve_declared_type(declared: &crate::ast::Type, subst: &HashMap<String, String>) -> Option<String> {
    subst.get(&crate::print::fmt_type(declared)).cloned()
}

fn axiom_to_rewrite(algebra: &str, ty: &str, axiom: &AxiomDecl, registry: &Registry) -> Option<Rewrite<CleaveLang, ConstantFold>> {
    // Every axiom in this codebase is declared on a single-generic algebra
    // (`Ring<T>`'s own `add_commutative`, ...) -- every param shares the
    // one flat `ty` uniformly, the same assumption this function has always
    // made (`build_pattern`'s own doc comment: "no cross-algebra resolution
    // attempted" for axioms specifically). `derivative_rule_type_env`
    // builds the *real*, per-param substitution `derivative` rules need
    // instead (multi-target algebras, e.g. `MatMul<A,B,C>`).
    let type_env: HashMap<&str, String> = axiom.params.iter().map(|p| (p.name.as_str(), ty.to_string())).collect();
    let ExprKind::Call(path, _, args, _) = &axiom.body.kind else { return None };
    let [lhs, rhs] = args.as_slice() else { return None };
    if path.segments.join("::") != "eq" {
        return None; // an axiom body that isn't `lhs == rhs` isn't representable yet
    }
    // No `d(...)` sugar, no referenced-unit bookkeeping -- both are a
    // `derivative`-rule-only concern (`build_pattern`'s own doc comment).
    let mut referenced = HashSet::new();
    let mut lhs_ast = PatternAst::default();
    build_pattern(lhs, algebra, ty, &type_env, None, &mut referenced, registry, &mut lhs_ast)?;
    let mut rhs_ast = PatternAst::default();
    build_pattern(rhs, algebra, ty, &type_env, None, &mut referenced, registry, &mut rhs_ast)?;
    let name = format!("{}@{algebra}<{ty}>", axiom.name);
    Rewrite::new(name, egg::Pattern::new(lhs_ast), egg::Pattern::new(rhs_ast)).ok()
}

/// Walks one side of an axiom's (or a `derivative` rule's) own body,
/// building it up as a `PatternAst` node by node (never through string
/// parsing — this module has already hit real ambiguities doing that twice
/// for `CleaveLang` itself, see its own doc comment; building
/// programmatically sidesteps the entire class of problem). A bare `Path`
/// matching one of the declared `params` becomes a pattern variable
/// (`?name`); a bare integer/bool literal becomes the matching `CleaveLang`
/// leaf directly (axiom/derivative-rule bodies are never type-checked —
/// `registry.rs` retains them as pure, unvalidated data — so a literal's
/// own text is parsed directly, no real type inference). Anything else (a
/// field access, a struct literal, ...) isn't representable yet — returns
/// `None`, rejecting the whole rule rather than guessing.
///
/// A number literal is `CleaveLang::Float` when `ty` is `f32`/`f64`, `Int`
/// otherwise — found necessary, not assumed: parsing every literal as `u64`
/// unconditionally (this function's own earlier behavior) meant an axiom
/// like `Ring<T>`'s own `add_zero(a): add(a, 0) == a;`, instantiated for a
/// float `T`, built a rule whose own "zero" position could never match a
/// real runtime `CleaveLang::Float(0.0)` — a different `enode` variant
/// entirely — so the rule was silently inert for every float instantiation
/// (`doc/backlog-done.md`'s own `CVal::Float` entry documents this exactly;
/// fixed here since `Activation::tanh`'s own migrated `derivative` rule,
/// `1 - tanh(u)²`, needs its literal `1` to actually match).
///
/// `d_var: Some(x)` — only ever set while building a `derivative` rule's
/// own body, `None` for an ordinary axiom — recognizes `d(inner)` (a `Call`
/// whose path is literally `"d"`, exactly one argument) as sugar for "the
/// derivative of `inner` with respect to this rule's own implicit
/// differentiation variable," compiling to `Op("derivative", [inner, x])`
/// instead of resolving `"d"` as an ordinary algebra method — the *only*
/// new case; every other `Call` shape is unaffected, and `d_var: None`
/// (every axiom) never takes this branch at all, so a real algebra method
/// genuinely named `d` still resolves normally there.
///
/// `referenced` collects the concrete unit name (`"{algebra}::{method}
/// <{ty}>"`) of every ordinary `Call` visited — a `derivative` rule's own
/// body can reference a unit the function actually being differentiated
/// never itself called (the product rule always needs `add`, even
/// differentiating a body that only ever multiplies) — `synthesize_
/// derivatives` needs this set to know such a reference is valid and to
/// let `rebuild` recognize it afterward. Unused by `axiom_to_rewrite`
/// (passed a throwaway set), which has never needed this.
///
/// Returns the built node's own `Id` alongside its own resolved concrete
/// type, when known (`None` for a bare literal, which carries no type of
/// its own) — needed so an *enclosing* call to a genuinely different
/// algebra (`Ring::add`, called from inside `MatMul<A,B,C>`'s own product
/// rule) can work out which single concrete type to instantiate `Ring` at,
/// rather than reusing whatever multi-target combined string the
/// *enclosing* rule's own `ty` happens to be (`MatMul::matmul<A,B,C>`'s own
/// combined "A,B,C" string is not a real `Ring<T>` instantiation at all).
/// `type_env` (param name -> concrete type, built once by the caller —
/// `derivative_rule_type_env` for a real `derivative` rule, a trivial
/// flat map in `axiom_to_rewrite`) is what makes a `Path` leaf's own type
/// known in the first place.
fn build_pattern(
    expr: &Expr,
    algebra: &str,
    ty: &str,
    type_env: &HashMap<&str, String>,
    d_var: Option<Var>,
    referenced: &mut HashSet<String>,
    registry: &Registry,
    ast: &mut PatternAst<CleaveLang>,
) -> Option<(egg::Id, Option<String>)> {
    match &expr.kind {
        ExprKind::Path(p) => {
            let name = p.segments.join("::");
            let Some(resolved_ty) = type_env.get(name.as_str()) else {
                return None; // a bare name that isn't one of the rule's own params -- not representable
            };
            let var = Var::from(Symbol::from(format!("?{name}")));
            Some((ast.add(ENodeOrVar::Var(var)), Some(resolved_ty.clone())))
        }
        ExprKind::NumberLit { text, .. } if matches!(ty, "f32" | "f64") => {
            let n: f64 = text.parse().ok()?;
            Some((ast.add(ENodeOrVar::ENode(CleaveLang::Float(n.into()))), None))
        }
        ExprKind::NumberLit { text, .. } => {
            let n: u64 = text.parse().ok()?;
            Some((ast.add(ENodeOrVar::ENode(CleaveLang::Int(n))), None))
        }
        ExprKind::BoolLit(b) => Some((ast.add(ENodeOrVar::ENode(CleaveLang::Bool(*b))), None)),
        ExprKind::Call(path, _, call_args, _) if path.segments.join("::") == "d" && d_var.is_some() => {
            let [inner] = call_args.as_slice() else { return None }; // `d(...)` always takes exactly one argument
            let (inner_id, inner_ty) = build_pattern(inner, algebra, ty, type_env, d_var, referenced, registry, ast)?;
            let x_id = ast.add(ENodeOrVar::Var(d_var.unwrap()));
            // Differentiating distributes component-wise (`construction_
            // derivative_rewrites`'s own doc comment) -- `d(inner)` always
            // has the exact same shape/type as `inner` itself.
            Some((ast.add(ENodeOrVar::ENode(CleaveLang::Op("derivative".into(), vec![inner_id, x_id]))), inner_ty))
        }
        ExprKind::Call(path, _, call_args, _) => {
            let method = path.segments.join("::");
            // Which algebra actually owns `method` — almost always the
            // *enclosing* one (`Ring<T>`'s own axioms/derivative rules
            // calling `add`/`mul`/..., all declared right there), checked
            // first so every existing single-algebra rule keeps resolving
            // exactly as before. Falls back to a real registry search
            // (`Registry::algebras_with_fn`) only when the enclosing
            // algebra doesn't declare it — needed for real:
            // `Activation::tanh`'s own derivative rule (`1 - tanh(u)²`)
            // needs `Ring`'s own `sub`/`mul`, not `Activation`'s (which
            // doesn't have either). Ambiguous (more than one algebra
            // declares it) or simply unknown — rejected, not guessed,
            // same posture as everywhere else `build_pattern` returns
            // `None`.
            let owner = if registry.fn_sig(algebra, &method).is_some_and(|sig| sig.params.len() == call_args.len()) {
                algebra.to_string()
            } else {
                match registry.algebras_with_fn(&method, call_args.len()).as_slice() {
                    [only] => only.to_string(),
                    _ => return None,
                }
            };
            let mut ids = Vec::with_capacity(call_args.len());
            let mut arg_types: Vec<Option<String>> = Vec::with_capacity(call_args.len());
            for a in call_args {
                let (id, arg_ty) = build_pattern(a, algebra, ty, type_env, d_var, referenced, registry, ast)?;
                ids.push(id);
                arg_types.push(arg_ty);
            }
            // Same algebra as the one this whole rule is declared on --
            // reuse the outer, possibly multi-target combined `ty` string
            // unchanged, exactly like every rule already did before this:
            // a recursive self-call (`MatMul::matmul` calling itself inside
            // its own product rule, `Index::index` likewise) is inherently
            // the *same* multi-target instantiation as the enclosing rule,
            // not something to re-derive from its own arguments' types.
            // Otherwise (a genuinely *different* algebra, e.g. `Ring::add`
            // called from inside `MatMul`'s own rule): every such target in
            // this codebase's stdlib is single-generic (`Ring<T>`,
            // `Transcendental<T>`), so the one concrete type its own
            // (type-bearing) arguments agree on *is* that generic's own
            // concrete value -- a genuinely multi-generic *different*-
            // algebra callee bails (`None`) rather than guessing, matching
            // this function's posture everywhere else.
            let call_ty = if owner == algebra {
                ty.to_string()
            } else {
                let mut agreed: Option<&str> = None;
                for t in arg_types.iter().flatten() {
                    match agreed {
                        None => agreed = Some(t.as_str()),
                        Some(a) if a == t.as_str() => {}
                        Some(_) => return None,
                    }
                }
                agreed?.to_string()
            };
            let unit_name = format!("{owner}::{method}<{call_ty}>");
            referenced.insert(unit_name.clone());
            // This call's own result type, for whoever (if anyone) embeds
            // it as an argument to a further, enclosing call -- `owner`'s
            // own declared return type (e.g. `MatMul::matmul`'s own `C`),
            // substituted through `owner`'s own generic-name mapping at
            // `call_ty`. `None` (rather than falling back to `call_ty`
            // itself) when the declared return type isn't a bare generic
            // name `resolve_declared_type` can resolve -- an enclosing
            // different-algebra call needing it then correctly bails too,
            // rather than silently building a wrong unit name from it.
            let result_ty = registry
                .fn_sig(&owner, &method)
                .and_then(|sig| sig.ret.as_ref())
                .and_then(|ret| resolve_declared_type(ret, &generic_substitution(&owner, &call_ty, registry)));
            let call_id = ast.add(ENodeOrVar::ENode(CleaveLang::Op(unit_name.into(), ids)));
            Some((call_id, result_ty))
        }
        _ => None,
    }
}

/// Built-in, not user-declared (unlike `axiom_rewrites`) — "reading a field
/// straight back out of the struct that just set it" is definitionally
/// true for every struct, the same way constant folding is definitionally
/// true for every `+`/`*`, and no `algebra`/`axiom` should have to spell it
/// out. Driven by `Forward`'s own `struct_ops`/`field_ops` (every struct-
/// construction/field-read symbol actually reached while building the
/// e-graph being optimized), not `registry.axioms` — this has nothing to do
/// with user-declared algebra laws. For each reached construction, for each
/// of its own fields that was also actually read somewhere reached, builds
/// `field(struct(?f0, ..., ?fN), field_i) -> ?f_i` programmatically via
/// `PatternAst`/`ENodeOrVar` — the identical technique `build_pattern`
/// already uses, not string parsing.
pub fn struct_projection_rewrites(
    struct_ops: &HashMap<Symbol, (String, Vec<String>, Ty)>,
    field_ops: &HashMap<Symbol, (Ty, String, Ty)>,
) -> Vec<Rewrite<CleaveLang, ConstantFold>> {
    let mut rules = Vec::new();
    for (struct_sym, (struct_name, field_names, struct_ty)) in struct_ops {
        for field_name in field_names {
            let field_sym = Symbol::from(format!("field:{struct_ty}:{field_name}"));
            if !field_ops.contains_key(&field_sym) {
                continue; // constructed but never actually read anywhere reached -- no rewrite worth building
            }

            let mut lhs_ast = PatternAst::default();
            let struct_children: Vec<egg::Id> =
                field_names.iter().map(|f| lhs_ast.add(ENodeOrVar::Var(Var::from(Symbol::from(format!("?{f}")))))).collect();
            let struct_id = lhs_ast.add(ENodeOrVar::ENode(CleaveLang::Op(*struct_sym, struct_children)));
            lhs_ast.add(ENodeOrVar::ENode(CleaveLang::Op(field_sym, vec![struct_id])));

            let mut rhs_ast = PatternAst::default();
            rhs_ast.add(ENodeOrVar::Var(Var::from(Symbol::from(format!("?{field_name}")))));

            let name = format!("struct-projection:{struct_name}.{field_name}");
            if let Ok(rw) = Rewrite::new(name, egg::Pattern::new(lhs_ast), egg::Pattern::new(rhs_ast)) {
                rules.push(rw);
            }
        }
    }
    rules
}

/// `d(Struct(f1:e1, ..., fn:en)) = Struct(f1:d(e1), ..., fn:d(en))` and
/// `d([e1, ..., en]) = [d(e1), ..., d(en)]` -- differentiating a
/// construction distributes component-wise onto its own arguments: the
/// derivative of "build a value out of these pieces" is "build the same
/// shape out of the pieces' own derivatives," the direct algebraic
/// meaning agreed on before writing this (see `doc/backlog.md`'s own
/// "toward tensorial ML" item). Built-in, not declarable via cleave
/// source -- same reason `derivative-self`/`derivative-independent-zero`
/// above are hardcoded rather than registry-declared: `PrimOp::Struct`/
/// `PrimOp::Array` aren't algebra methods with a fixed, nameable
/// signature a `derivative` item could ever attach to. One concrete rule
/// per actually-reached struct/array shape, mirroring `struct_projection_
/// rewrites`'s own "only build what this translation actually touched"
/// discipline, just above -- harmless, not unsound, if a rule never
/// matches anything in a given e-graph.
fn construction_derivative_rewrites(
    struct_ops: &HashMap<Symbol, (String, Vec<String>, Ty)>,
    array_ops: &HashMap<Symbol, Ty>,
) -> Vec<Rewrite<CleaveLang, ConstantFold>> {
    let mut rules = Vec::new();
    let x = Var::from(Symbol::from("?__diff_x"));

    for (struct_sym, (_struct_name, field_names, struct_ty)) in struct_ops {
        let field_vars: Vec<Var> = field_names.iter().map(|f| Var::from(Symbol::from(format!("?{f}")))).collect();

        let mut lhs = PatternAst::default();
        let field_ids: Vec<egg::Id> = field_vars.iter().map(|&v| lhs.add(ENodeOrVar::Var(v))).collect();
        let struct_id = lhs.add(ENodeOrVar::ENode(CleaveLang::Op(*struct_sym, field_ids)));
        let x_id_lhs = lhs.add(ENodeOrVar::Var(x));
        lhs.add(ENodeOrVar::ENode(CleaveLang::Op("derivative".into(), vec![struct_id, x_id_lhs])));

        let mut rhs = PatternAst::default();
        let x_id_rhs = rhs.add(ENodeOrVar::Var(x));
        let d_field_ids: Vec<egg::Id> = field_vars
            .iter()
            .map(|&v| {
                let f_id = rhs.add(ENodeOrVar::Var(v));
                rhs.add(ENodeOrVar::ENode(CleaveLang::Op("derivative".into(), vec![f_id, x_id_rhs])))
            })
            .collect();
        rhs.add(ENodeOrVar::ENode(CleaveLang::Op(*struct_sym, d_field_ids)));

        // `struct_ty` (the full, concrete type text -- e.g. `Tensor<f32,
        // 1, 2>`), not the bare `struct_name` (`Tensor`) alone -- otherwise
        // two differently-shaped instantiations of the same pack-generic
        // struct (`Tensor<f32,1,2>` and `Tensor<f32,2,2>`, say, both
        // reached in one translation) built two *different* rules sharing
        // one *name*, a real, found-by-testing egg warning ("Duplicated
        // rule names may affect rule reporting and scheduling") — harmless
        // functionally (each rule's own pattern is still distinct,
        // correctly matched independently), but not a name collision this
        // module should actually produce.
        let name = format!("derivative-construction:{struct_ty}");
        if let Ok(rw) = Rewrite::new(name, egg::Pattern::new(lhs), egg::Pattern::new(rhs)) {
            rules.push(rw);
        }
    }

    for (array_sym, ty) in array_ops {
        // The element count lives only inside the symbol string itself
        // (`array:{ty}:{count}`, `Forward::walk`'s own `PrimOp::Array` arm)
        // -- `array_ops`'s value is the element type alone, so the arity
        // has to be parsed back out here rather than read off a field.
        let Some(count) = array_sym.as_str().rsplit(':').next().and_then(|n| n.parse::<usize>().ok()) else { continue };
        let elem_vars: Vec<Var> = (0..count).map(|i| Var::from(Symbol::from(format!("?e{i}")))).collect();

        let mut lhs = PatternAst::default();
        let elem_ids: Vec<egg::Id> = elem_vars.iter().map(|&v| lhs.add(ENodeOrVar::Var(v))).collect();
        let array_id = lhs.add(ENodeOrVar::ENode(CleaveLang::Op(*array_sym, elem_ids)));
        let x_id_lhs = lhs.add(ENodeOrVar::Var(x));
        lhs.add(ENodeOrVar::ENode(CleaveLang::Op("derivative".into(), vec![array_id, x_id_lhs])));

        let mut rhs = PatternAst::default();
        let x_id_rhs = rhs.add(ENodeOrVar::Var(x));
        let d_elem_ids: Vec<egg::Id> = elem_vars
            .iter()
            .map(|&v| {
                let e_id = rhs.add(ENodeOrVar::Var(v));
                rhs.add(ENodeOrVar::ENode(CleaveLang::Op("derivative".into(), vec![e_id, x_id_rhs])))
            })
            .collect();
        rhs.add(ENodeOrVar::ENode(CleaveLang::Op(*array_sym, d_elem_ids)));

        let name = format!("derivative-construction:array:{ty}:{count}");
        if let Ok(rw) = Rewrite::new(name, egg::Pattern::new(lhs), egg::Pattern::new(rhs)) {
            rules.push(rw);
        }
    }

    rules
}

// ---------------------------------------------------------------- derivative (auto-diff)

use egg::{Applier, Language, Subst};

/// `Op("derivative", [expr_id, wrt_id])` — introduced by `synthesize_
/// derivatives` as a synthetic marker (not a real cleave call — no algebra
/// named "derivative" exists) wrapping the value being differentiated and
/// the variable being differentiated with respect to. These rewrite rules
/// progressively eliminate it during the very same `egg::Runner::run`/
/// saturation pass that already runs axioms and constant-folding — toward
/// `doc/backlog.md`'s own auto-diff item, not a bespoke, separate
/// differentiation engine: chain/sum/product rule plus the two base cases
/// (`derivative(x,x) → 1`, `derivative(anything-that-doesn't-depend-on-x,
/// x) → 0`).
///
/// `ty` is the concrete numeric type being differentiated over (known
/// directly from the function being derived, not recovered from `reached`
/// the way `axiom_rewrites` recovers its own type set) — the base rules
/// below fire even when *no* algebra is reached at all (`fn f(x: f32) ->
/// f32 { x }`, the identity function, needs only `derivative(x,x) → 1`, no
/// `Ring` op in sight). Every other rule (sum/sub/product, the elementary-
/// function table) is built only for whichever `Ring<ty>`/elementary-
/// function unit is actually present in `reached` — same "only build what's
/// actually reached" discipline `axiom_rewrites` already uses; harmless,
/// not unsound, if a rule never matches anything in a given e-graph.
pub fn derivative_rewrites(ty: &str, reached: &HashMap<String, (String, String)>, registry: &Registry) -> (Vec<Rewrite<CleaveLang, ConstantFold>>, HashSet<String>) {
    let one = if matches!(ty, "f32" | "f64") { CleaveLang::Float(1.0.into()) } else { CleaveLang::Int(1) };

    let mut rules = Vec::new();

    // `derivative(?x, ?x) -> 1` — an ordinary declarative rule: egg's own
    // pattern semantics already require two occurrences of the same
    // pattern var to match the same e-class, no custom code needed.
    // Base cases, both built-in (not declarable — see the module's own
    // doc comment on why): they're facts about what a variable/a constant
    // *is*, not an algebraic law belonging to any one algebra.
    {
        let mut lhs = PatternAst::default();
        let x = Var::from(Symbol::from("?x"));
        let x1 = lhs.add(ENodeOrVar::Var(x));
        let x2 = lhs.add(ENodeOrVar::Var(x));
        lhs.add(ENodeOrVar::ENode(CleaveLang::Op("derivative".into(), vec![x1, x2])));
        let mut rhs = PatternAst::default();
        rhs.add(ENodeOrVar::ENode(one.clone()));
        if let Ok(rw) = Rewrite::new("derivative-self", egg::Pattern::new(lhs), egg::Pattern::new(rhs)) {
            rules.push(rw);
        }
    }

    // `derivative(?a, ?x) -> 0`, conditioned on `?a`'s own e-class genuinely
    // differing from `?x`'s *and* `?x` never actually occurring anywhere
    // inside `?a`'s own subtree (a real, recursive occurs-check over the
    // e-graph — `depends_on_eclass`, below — not just "is `?a` shaped like a
    // leaf"). Originally restricted to leaf shapes (`Free`/`Int`/`Float`/
    // `Bool`) only; broadened after finding, directly, that a compound-but-
    // still-`?x`-independent subexpression — `data[i]` (`PrimOp::Load`)
    // reading from a constant array inside a function being differentiated
    // w.r.t. a *different* parameter — has no leaf shape at all, so the old
    // condition left its own `derivative(load(...), ?x)` permanently stuck
    // (no declared rule applies to `Load` either — it isn't an algebra
    // method). A bare e-class-inequality check alone would still be
    // unsound, unchanged from before: `derivative(x*y, x)`'s own `x*y`
    // subexpression has a *different* e-class than `x` too, but obviously
    // still depends on it — the occurs-check is exactly what tells the two
    // cases apart correctly, for *any* compound shape, not just the leaf
    // ones.
    //
    // The "0" itself is no longer a single fixed literal (`IndependentZero
    // Applier`/`build_zero`, below) — found directly, not anticipated,
    // building the tensorial `examples/xor.cleave` reformulation: `?a`
    // isn't always scalar (`d(x)/dw` where `x` is itself a whole
    // independent `linalg::Tensor` input) — a bare scalar `0.0` unioned
    // into that e-class is a real type-soundness violation (a scalar
    // masquerading as a tensor), silently *cheaper* than the correctly-
    // shaped answer the chain rule separately builds in the very same
    // e-class, so extraction preferred the wrong one. Confirmed via a
    // minimal probe: an identity-matrix literal fed into `matmul`, once
    // `MatMul::matmul`'s own new product rule needed differentiating
    // through it, crashed MLIR lowering ("a nested array's own element
    // must be an already-built array value, not a bare literal"). Needs
    // `?a`'s own concrete `Ty` (`ConstantFold::known_types`'s own doc
    // comment) to build a properly-shaped zero; when that `Ty` isn't known,
    // or isn't a shape `build_zero` knows how to decompose, the rule simply
    // doesn't fire for that e-class — sound, not a guess: the full
    // recursive chain-rule expansion (already present in the same e-class
    // regardless) is what supplies the correct answer there instead, this
    // rule staying a pure optimization shortcut, never the sole source of
    // truth.
    {
        let a = Var::from(Symbol::from("?a"));
        let x = Var::from(Symbol::from("?x"));
        let mut lhs = PatternAst::default();
        let a_id = lhs.add(ENodeOrVar::Var(a));
        let x_id = lhs.add(ENodeOrVar::Var(x));
        lhs.add(ENodeOrVar::ENode(CleaveLang::Op("derivative".into(), vec![a_id, x_id])));
        let tensor_like: HashMap<String, String> = registry
            .struct_names()
            .filter_map(|name| {
                let [field] = registry.struct_fields(name)? else { return None };
                if !registry.struct_generics(name).last()?.is_variadic() {
                    return None;
                }
                Some((name.to_string(), field.name.clone()))
            })
            .collect();
        let applier = IndependentZeroApplier { a, x, tensor_like };
        if let Ok(rw) = Rewrite::new("derivative-independent-zero", egg::Pattern::new(lhs), applier) {
            rules.push(rw);
        }
    }

    // Everything else — sum/product/chain rule, for whichever algebra/
    // method — comes entirely from `derivative` rules declared in cleave
    // source (`stdlib/num`, `stdlib/nn`, or a user's own algebra), not
    // hard-coded here (`doc/backlog-done.md`'s own "Auto-diff v1 ->
    // algebra-declared rules" entry).
    let (declared_rules, referenced) = derivative_rule_rewrites(registry, reached);
    rules.extend(declared_rules);

    (rules, referenced)
}

/// `derivative-independent-zero`'s own custom `Applier` (`derivative_
/// rewrites`'s own doc comment on why the "0" has to be dynamically
/// shaped, not a fixed literal) — a genuinely dynamic decision (which e-
/// class matched `?a`, and what `Ty` it turns out to have, are only known
/// at rewrite-*application* time, not when this rule is built), so a plain
/// declarative `egg::Pattern` RHS can't express it at all. Folds the old
/// `ConditionalApplier`'s own disjointness condition directly into `apply_
/// one` (rather than keeping a separate `Condition`) since building the
/// zero and checking whether it's even possible are the same lookup
/// (`egraph[a_class].data.own_ty`) — no reason to compute it twice.
struct IndependentZeroApplier {
    a: Var,
    x: Var,
    /// Single-field, pack-generic struct name -> that sole field's own
    /// declared name (`linalg::Tensor`'s own `data`) — `build_zero`'s own
    /// doc comment. Snapshotted once, owned, since `Applier` implementations
    /// must be `Send + Sync + 'static` (`Rewrite::new`'s own bound) and so
    /// can't hold a borrowed `&Registry`.
    tensor_like: HashMap<String, String>,
}

impl Applier<CleaveLang, ConstantFold> for IndependentZeroApplier {
    fn apply_one(
        &self,
        egraph: &mut egg::EGraph<CleaveLang, ConstantFold>,
        eclass: Id,
        subst: &Subst,
        _searcher_ast: Option<&PatternAst<CleaveLang>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        let a_class = egraph.find(subst[self.a]);
        let x_class = egraph.find(subst[self.x]);
        if a_class == x_class {
            return vec![];
        }
        // `x_class`'s own `free_deps` is exactly `{x}` (a bare `Free` node,
        // nothing else could have narrowed it further) — checking
        // disjointness rather than a bare `.contains` reads the same but
        // stays correct even if `x` were ever something richer than a
        // single free variable.
        if !egraph[x_class].data.free_deps.is_disjoint(&egraph[a_class].data.free_deps) {
            return vec![];
        }
        // A bare literal's own *kind* (`CleaveLang::Float`/`Int`) already
        // tells us it's scalar, unambiguously, with no need for `own_ty` at
        // all -- checked first, directly off `a_class`'s own e-nodes, so
        // `derivative(3.0, x) -> 0` keeps working even where nothing (a
        // hand-built test e-graph, say, bypassing `Forward` entirely) ever
        // populated `ConstantFold::known_types`.
        let is_float = egraph[a_class].nodes.iter().any(|n| matches!(n, CleaveLang::Float(_)));
        let is_int = egraph[a_class].nodes.iter().any(|n| matches!(n, CleaveLang::Int(_)));
        if is_float || is_int {
            let zero_id = if is_float { egraph.add(CleaveLang::Float(0.0.into())) } else { egraph.add(CleaveLang::Int(0)) };
            egraph.union(eclass, zero_id);
            return vec![eclass];
        }
        let Some(ty) = egraph[a_class].data.own_ty.clone() else { return vec![] }; // shape unknown -- don't guess, let the chain rule supply the answer elsewhere
        let Some(zero_id) = build_zero(egraph, &ty, &self.tensor_like) else { return vec![] };
        egraph.union(eclass, zero_id);
        vec![eclass]
    }

    fn vars(&self) -> Vec<Var> {
        vec![self.a, self.x]
    }
}

/// Builds a same-*shaped* zero e-node for `ty`, recursively — the direct
/// fix for `derivative_rewrites`'s own found bug (a bare scalar `0.0`
/// silently standing in for a whole independent tensor/array). Three shapes
/// recognized, matching exactly what's reachable through `linalg::Tensor`
/// plus ordinary arrays; anything else (a genuinely multi-field struct, an
/// algebra-`Op` node whose own result type isn't tracked at all —
/// `ConstantFold::known_types`'s own doc comment) returns `None` rather
/// than guessing, same posture as every other "build what's reachable, bail
/// otherwise" function in this module:
/// - A scalar numeric `Con` — the original literal `0`/`0.0`.
/// - `Array(elem, size)` — a same-size array of recursively-built zero
///   elements, symbol-for-symbol matching what `Forward::walk`'s own
///   `PrimOp::Array` arm would have built for a real literal of this shape
///   (`format!("array:{ty}:{n}")`, `ty` here *being* this exact `Array`
///   value, not reconstructed by hand).
/// - `App(struct_name, args)` where `struct_name` is single-field and
///   pack-generic (`tensor_like`, `IndependentZeroApplier`'s own doc
///   comment) — `linalg::Tensor<T, Dims...>`'s own shape specifically,
///   recognized *structurally* (a variadic trailing generic, one field),
///   not by hardcoding the name "Tensor": `args`' own first element is
///   taken as the element type, every remaining element must already be a
///   resolved `Ty::Const` (the pack's own concrete dimensions) — bails if
///   not, rather than trusting a struct that merely *looks* similar. The
///   field's own type is rebuilt as nested nested `Array`s around those
///   dims (`[ElemTy; Dims...]`'s own real monomorphized shape), then this
///   same function recurses on it.
fn build_zero(egraph: &mut egg::EGraph<CleaveLang, ConstantFold>, ty: &Ty, tensor_like: &HashMap<String, String>) -> Option<egg::Id> {
    match ty {
        Ty::Con(name) if matches!(name.as_str(), "f32" | "f64") => Some(egraph.add(CleaveLang::Float(0.0.into()))),
        Ty::Con(_) => Some(egraph.add(CleaveLang::Int(0))),
        Ty::Array(elem, size) => {
            let Ty::Const(ConstValue::Int(n)) = **size else { return None };
            let elem_id = build_zero(egraph, elem, tensor_like)?;
            let sym = Symbol::from(format!("array:{ty}:{n}"));
            egraph.analysis.known_types.entry(sym).or_insert_with(|| ty.clone());
            Some(egraph.add(CleaveLang::Op(sym, vec![elem_id; n as usize])))
        }
        Ty::App(struct_name, args) => {
            let field_name = tensor_like.get(struct_name)?;
            let (elem_ty, dims) = args.split_first()?;
            if dims.is_empty() || !dims.iter().all(|d| matches!(d, Ty::Const(_))) {
                return None; // not shaped like `[ElemTy; Dims...]` -- bail rather than guess
            }
            let field_ty = dims.iter().rev().fold(elem_ty.clone(), |acc, d| Ty::Array(Box::new(acc), Box::new(d.clone())));
            let data_id = build_zero(egraph, &field_ty, tensor_like)?;
            let struct_sym = Symbol::from(format!("struct:{ty}:{field_name}"));
            egraph.analysis.known_types.entry(struct_sym).or_insert_with(|| ty.clone());
            Some(egraph.add(CleaveLang::Op(struct_sym, vec![data_id])))
        }
        _ => None,
    }
}

/// Builds one concrete `Rewrite` per `(derivative rule, reached concrete
/// type)` pair — the `derivative`-rule counterpart of `axiom_rewrites`,
/// same `reached_types` grouping, same "only build what's actually
/// reached" discipline (harmless, not unsound, if a rule never matches
/// anything in a given e-graph). Also returns the union of every built
/// rule's own referenced-unit set (`build_pattern`'s own doc comment) —
/// `synthesize_derivatives` needs it to let `rebuild` recognize a unit a
/// fired rule references that the function actually being differentiated
/// never itself called.
fn derivative_rule_rewrites(registry: &Registry, reached: &HashMap<String, (String, String)>) -> (Vec<Rewrite<CleaveLang, ConstantFold>>, HashSet<String>) {
    let mut reached_types: HashMap<&str, HashSet<&str>> = HashMap::new();
    for (unit_name, (algebra, _method)) in reached {
        if let Some(ty) = concrete_type_of(unit_name) {
            reached_types.entry(algebra.as_str()).or_default().insert(ty);
        }
    }

    let mut rules = Vec::new();
    let mut referenced = HashSet::new();
    for (algebra, types) in &reached_types {
        for rule in registry.derivative_rules(algebra) {
            for ty in types {
                if let Some((rw, refs)) = derivative_rule_to_rewrite(algebra, ty, rule, registry) {
                    rules.push(rw);
                    referenced.extend(refs);
                }
            }
        }
    }
    (rules, referenced)
}

/// `derivative` counterpart of `axiom_to_rewrite`: `derivative mul(a, b):
/// add(mul(a, d(b)), mul(d(a), b));` becomes `derivative(Ring::mul<ty>(?a,
/// ?b), ?x) -> Ring::add<ty>(Ring::mul<ty>(?a, derivative(?b,?x)), Ring::
/// mul<ty>(derivative(?a,?x), ?b))` — the LHS built by hand (the outer
/// `derivative(method(...), ?__diff_x)` wrapper has no source-level `Expr`
/// of its own to walk), the RHS via `build_pattern` with `d_var: Some(?
/// __diff_x)` so every `d(...)` in the declared body compiles to a nested
/// `derivative` node sharing the identical `?__diff_x`.
///
/// The differentiation variable's own pattern symbol is deliberately
/// `__diff_x`, not a bare `x` — a real, found-by-testing bug: an earlier
/// version used literal `?x`, which silently *collided* with a declared
/// parameter genuinely named `x` (`fn exp(x: T) -> T; derivative exp(x):
/// mul(exp(x), d(x));`, `stdlib/num/num.cleave`'s own `Transcendental<T>`)
/// — `param_ids` below, keyed off each parameter's own real name, would
/// then bind *the same* egg pattern variable `?x` the rule's own LHS
/// already uses for "the differentiation variable," forcing them to match
/// the *same* e-class: the rule then only ever fired when the method was
/// applied *directly* to the exact variable being differentiated (`exp(w)`
/// w.r.t. `w`), never through any composition (`exp(exp(w))`, `exp(-w)`,
/// ...), silently leaving the derivative marker permanently stuck —
/// invisible on `Ring<T>` (whose own params are always named `a`/`b`,
/// never `x`) purely by naming coincidence, not because the bug didn't
/// apply there too.
///
/// Each param's own concrete type (`type_env` below, `build_pattern`'s own
/// doc comment) comes from the algebra's own declared signature for
/// `rule.method` (`fn matmul(a: A, b: B) -> C;`), substituted positionally
/// through `generic_substitution` -- needed for real once a multi-target
/// algebra's own product rule (`MatMul<A,B,C>`'s `derivative matmul(a, b):
/// add(matmul(d(a), b), matmul(a, d(b)));`) has to call a genuinely
/// *different*, single-target algebra (`Ring::add`): the outer `add` call's
/// own operands are both type `C` specifically, not the whole "A,B,C"
/// combined `ty` string the enclosing `matmul` rule was built for — found
/// directly, not anticipated, via a minimal probe that panicked building an
/// `Op` node literally named `Ring::add<Tensor<f32,2,2>, Tensor<f32,2,2>,
/// Tensor<f32,2,2>>`, a unit that could never actually exist. Falls back to
/// the flat `ty` for any param whose own declared type isn't a bare generic
/// name (`resolve_declared_type`'s own doc comment) — harmless: such a
/// param only feeds a *same*-algebra recursive call in practice (`Index`'s
/// own `idx: [i32; K]`), which ignores `type_env` entirely and reuses `ty`
/// unchanged regardless.
fn derivative_rule_to_rewrite(algebra: &str, ty: &str, rule: &DerivativeRuleDecl, registry: &Registry) -> Option<(Rewrite<CleaveLang, ConstantFold>, HashSet<String>)> {
    let sig = registry.fn_sig(algebra, &rule.method)?;
    let subst = generic_substitution(algebra, ty, registry);
    let type_env: HashMap<&str, String> = rule
        .params
        .iter()
        .zip(&sig.params)
        .map(|(rule_p, sig_p)| {
            let resolved = sig_p.ty.as_ref().and_then(|declared| resolve_declared_type(declared, &subst)).unwrap_or_else(|| ty.to_string());
            (rule_p.name.as_str(), resolved)
        })
        .collect();
    let x = Var::from(Symbol::from("?__diff_x"));

    let mut lhs = PatternAst::default();
    let mut param_ids = Vec::with_capacity(rule.params.len());
    for p in &rule.params {
        param_ids.push(lhs.add(ENodeOrVar::Var(Var::from(Symbol::from(format!("?{}", p.name))))));
    }
    let method_id = lhs.add(ENodeOrVar::ENode(CleaveLang::Op(format!("{algebra}::{}<{ty}>", rule.method).into(), param_ids)));
    let x_id_lhs = lhs.add(ENodeOrVar::Var(x));
    lhs.add(ENodeOrVar::ENode(CleaveLang::Op("derivative".into(), vec![method_id, x_id_lhs])));

    let mut referenced = HashSet::new();
    let mut rhs = PatternAst::default();
    build_pattern(&rule.body, algebra, ty, &type_env, Some(x), &mut referenced, registry, &mut rhs)?;

    let name = format!("derivative-{algebra}::{}<{ty}>", rule.method);
    let rw = Rewrite::new(name, egg::Pattern::new(lhs), egg::Pattern::new(rhs)).ok()?;
    Some((rw, referenced))
}

/// The cost function extraction must use whenever a `derivative` node might
/// still be present — plain `AstSize` is the *wrong* tool here: a still-
/// unreduced `derivative(mul(x,y), x)` (3 nodes) is structurally *smaller*
/// than its own fully-reduced expansion (`add(mul(x,0), mul(1,y))`, 7+
/// nodes), so `AstSize` alone would happily keep the tiny, unhelpful
/// original — the exact opposite of this whole feature's own point (found
/// directly, by testing: `derivative_of_x_times_y_with_respect_to_x_
/// eliminates_the_derivative_marker` failed against plain `AstSize` before
/// this existed). Any live `derivative` node is penalized enormously (not
/// simply rejected outright — a saturation pass that genuinely can't
/// eliminate every last one, e.g. an elementary function with no known
/// derivative rule, should still extract *something* rather than fail to
/// extract at all) — ordinary `AstSize`-style counting otherwise, so a
/// fully `derivative`-free result still picks the cheapest *among* those.
pub struct DerivativeFreeCost;

impl egg::CostFunction<CleaveLang> for DerivativeFreeCost {
    type Cost = usize;

    fn cost<C>(&mut self, enode: &CleaveLang, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let penalty = if matches!(enode, CleaveLang::Op(sym, _) if sym.as_str() == "derivative") { 1_000_000 } else { 0 };
        enode.fold(1 + penalty, |sum, id| sum.saturating_add(costs(id)))
    }
}

// ---------------------------------------------------------------- e-graph -> CPS (backward)

use crate::cps::FreshVars;
use egg::RecExpr;
use std::cell::RefCell;

/// Every lookup table the backward translator needs, bundled into one
/// value passed by reference — `rebuild`/`rebuild_args` would otherwise
/// gain one more positional parameter per new op kind forever (already at
/// 8 distinct tables once `Array`/`ArrayRepeat`/`Load` join `raw_ops`/
/// `struct_ops`/`field_ops`/`call_units`/`free_vars` — past the point
/// where separate positional arguments stay readable). Built once per
/// segment from `Forward`'s own fields (see `optimize_program`'s and
/// `rebuild_segment`'s own callers).
struct OpTables<'a> {
    free_vars: &'a HashMap<Symbol, CVal>,
    raw_ops: &'a HashMap<Symbol, (String, Ty, Vec<(String, String)>)>,
    call_units: &'a HashSet<String>,
    struct_ops: &'a HashMap<Symbol, (String, Vec<String>, Ty)>,
    field_ops: &'a HashMap<Symbol, (Ty, String, Ty)>,
    array_ops: &'a HashMap<Symbol, Ty>,
    array_repeat_ops: &'a HashMap<Symbol, Ty>,
    load_ops: &'a HashMap<Symbol, (Ty, Ty)>,
    /// A reconstructed `Free` node's own original `CVal` (`free_vars`
    /// above) rebinds to a *different* `CVar` before it's used, if it's a
    /// key here — empty for every existing caller (`optimize_program`'s own
    /// `rebuild_segment`, splicing an optimized segment back into the
    /// *same* function it came from, needs no such rebinding). `synthesize_
    /// derivatives` is the one real user: `fprime`'s own body is built from
    /// `f`'s own already-translated e-graph, whose `Free` nodes reference
    /// *`f`'s* own parameter `CVar`s — reused directly, those would mean
    /// two different top-level functions each declaring a formal parameter
    /// under the same numeric identity while `f` itself may still be live
    /// elsewhere in the program (a real correctness issue, found during
    /// design, not debugging — `max_cvar_in_program`'s own doc comment
    /// already treats `CVar` uniqueness across the whole program as
    /// required). This maps each of `f`'s own parameter `CVar`s to a fresh
    /// one minted for `fprime` instead.
    param_substitution: &'a HashMap<CVar, CVar>,
}

/// The mechanical inverse of `Forward::walk` — turns an extracted
/// `RecExpr<CleaveLang>` (`Extractor::find_best`'s own second return value)
/// back into a `CExpr` fragment, calling `k` with the `CVal` naming the
/// reconstructed root value once done. Mirrors `cps.rs::convert_expr`'s own
/// continuation-passing shape exactly (build inner-to-outer, `k` names
/// "what happens next"), for the identical reason: a real call reconstructs
/// to `Fix`/`App`, which needs to *wrap* everything that comes after it,
/// not just sit inline the way a `LetPrim` can.
///
/// `memo` — `Extractor::find_best`'s own `RecExpr` legitimately DAG-shares
/// one `Id` between multiple parents once two constructions hashcons
/// together (see the module's own note on `Struct`/`Field` CSE); without a
/// memo, a shared `Id` reached from two different parents would get
/// rebuilt twice, with two different fresh `CVar`s, silently throwing away
/// that exact sharing (e.g. two separate `cleave_alloc` calls at runtime
/// for what should be one). `RefCell`, not a plain `&mut`, for the same
/// reason `FreshVars` itself uses `Cell` internally — every continuation
/// closure in this file is `&dyn Fn`, not `FnMut`, and a `RefCell` lets a
/// *shared* reference be threaded through them while still mutating the
/// map, each borrow scoped to one statement so runtime borrow conflicts
/// can't arise (nothing here ever holds a `Ref`/`RefMut` across a nested
/// call).
fn rebuild(recexpr: &RecExpr<CleaveLang>, id: egg::Id, fresh: &FreshVars, tables: &OpTables, memo: &RefCell<HashMap<egg::Id, CVal>>, k: &dyn Fn(CVal) -> CExpr) -> CExpr {
    // `.cloned()` right here, not `if let Some(v) = memo.borrow().get(&id)`
    // -- the latter keeps the `Ref` guard alive for the whole `if let`
    // *body* (`v` borrows from it), so a `k` that recurses into another
    // `memo.borrow_mut()` elsewhere would panic ("already borrowed") the
    // moment the memoized path is actually taken. Found by direct testing.
    let cached = memo.borrow().get(&id).cloned();
    if let Some(v) = cached {
        return k(v);
    }
    match &recexpr[id] {
        CleaveLang::Int(n) => k(CVal::Int(*n)),
        CleaveLang::Float(f) => k(CVal::Float((*f).into())),
        CleaveLang::Bool(b) => k(CVal::Bool(*b)),
        CleaveLang::Free(sym) => {
            let v = tables.free_vars.get(sym).unwrap_or_else(|| panic!("egraph: no original CVal recorded for free symbol `{sym}`"));
            let v = match v {
                CVal::Var(cv) => tables.param_substitution.get(cv).map_or_else(|| v.clone(), |&new_cv| CVal::Var(new_cv)),
                other => other.clone(),
            };
            k(v)
        }
        CleaveLang::Op(sym, children) => {
            rebuild_args(recexpr, children, fresh, tables, memo, &|arg_vals| {
                let name = sym.as_str();
                if let Some((op, ty, attrs)) = tables.raw_ops.get(sym) {
                    let var = fresh.var();
                    memo.borrow_mut().insert(id, CVal::Var(var));
                    CExpr::LetPrim {
                        var,
                        ty: ty.clone(),
                        op: PrimOp::RawMlirOp { op: op.clone(), attrs: attrs.clone() },
                        args: arg_vals,
                        cont: Box::new(k(CVal::Var(var))),
                    }
                } else if let Some((struct_name, field_names, ty)) = tables.struct_ops.get(sym) {
                    let var = fresh.var();
                    memo.borrow_mut().insert(id, CVal::Var(var));
                    CExpr::LetPrim {
                        var,
                        ty: ty.clone(),
                        op: PrimOp::Struct(struct_name.clone(), field_names.clone()),
                        args: arg_vals,
                        cont: Box::new(k(CVal::Var(var))),
                    }
                } else if let Some((struct_ty, field, ty)) = tables.field_ops.get(sym) {
                    let var = fresh.var();
                    memo.borrow_mut().insert(id, CVal::Var(var));
                    CExpr::LetPrim {
                        var,
                        ty: ty.clone(),
                        op: PrimOp::Field { struct_ty: struct_ty.clone(), field: field.clone() },
                        args: arg_vals,
                        cont: Box::new(k(CVal::Var(var))),
                    }
                } else if let Some(ty) = tables.array_ops.get(sym) {
                    let var = fresh.var();
                    memo.borrow_mut().insert(id, CVal::Var(var));
                    CExpr::LetPrim { var, ty: ty.clone(), op: PrimOp::Array, args: arg_vals, cont: Box::new(k(CVal::Var(var))) }
                } else if let Some(ty) = tables.array_repeat_ops.get(sym) {
                    let var = fresh.var();
                    memo.borrow_mut().insert(id, CVal::Var(var));
                    CExpr::LetPrim { var, ty: ty.clone(), op: PrimOp::ArrayRepeat, args: arg_vals, cont: Box::new(k(CVal::Var(var))) }
                } else if let Some((array_ty, ty)) = tables.load_ops.get(sym) {
                    let var = fresh.var();
                    memo.borrow_mut().insert(id, CVal::Var(var));
                    CExpr::LetPrim {
                        var,
                        ty: ty.clone(),
                        op: PrimOp::Load { array_ty: array_ty.clone() },
                        args: arg_vals,
                        cont: Box::new(k(CVal::Var(var))),
                    }
                } else if tables.call_units.contains(name) {
                    let result_var = fresh.var();
                    memo.borrow_mut().insert(id, CVal::Var(result_var));
                    let k_label = fresh.label("k");
                    let mut call_args = arg_vals;
                    call_args.push(CVal::Label(k_label.clone()));
                    CExpr::Fix {
                        defs: vec![CFunDef { name: k_label, params: vec![result_var], body: k(CVal::Var(result_var)), carried_types: None }],
                        body: Box::new(CExpr::App { func: CVal::Label(name.to_string()), args: call_args }),
                    }
                } else {
                    // A rewrite rule (an axiom) introduced a symbol neither
                    // the forward translator nor any known call ever
                    // produced -- can't happen from this module's own
                    // axiom/struct-projection translation, but a real,
                    // worth-panicking-on bug if it ever did: silently
                    // guessing how to lower an unknown op would be far
                    // worse.
                    panic!("egraph: extracted `Op` node `{name}` is in none of this module's own lookup tables, nor a known real call");
                }
            })
        }
    }
}

fn rebuild_args(recexpr: &RecExpr<CleaveLang>, ids: &[egg::Id], fresh: &FreshVars, tables: &OpTables, memo: &RefCell<HashMap<egg::Id, CVal>>, k: &dyn Fn(Vec<CVal>) -> CExpr) -> CExpr {
    fn go(recexpr: &RecExpr<CleaveLang>, ids: &[egg::Id], fresh: &FreshVars, tables: &OpTables, memo: &RefCell<HashMap<egg::Id, CVal>>, acc: Vec<CVal>, k: &dyn Fn(Vec<CVal>) -> CExpr) -> CExpr {
        let Some((first, rest)) = ids.split_first() else { return k(acc) };
        rebuild(recexpr, *first, fresh, tables, memo, &|v| {
            let mut acc2 = acc.clone();
            acc2.push(v);
            go(recexpr, rest, fresh, tables, memo, acc2, k)
        })
    }
    go(recexpr, ids, fresh, tables, memo, Vec::new(), k)
}

/// Rebuilds an optimized segment back into CPS form and splices it in place
/// of the original — every reference to `old_root_var` (the `CVar` the
/// segment originally bound its own final value to, before optimization)
/// inside `boundary` (`Forward::walk`'s own returned, untouched tail) gets
/// substituted with whatever `CVal` the reconstruction actually produces.
/// Takes `Forward`'s own lookup tables bundled as `&OpTables` rather than
/// `&Forward` as a whole — `fwd.egraph` itself is never needed here (only
/// `Runner`/`Extractor` need it, upstream of this call), and taking it
/// anyway would force a caller to keep the *whole* `Forward` borrowed even
/// after moving its own `egraph` field into a `Runner`. Owns the
/// reconstruction's own `memo` (one per segment, never shared across calls
/// — see `rebuild`'s own doc comment).
#[allow(clippy::too_many_arguments)]
pub(crate) fn rebuild_segment(
    recexpr: &RecExpr<CleaveLang>,
    root: egg::Id,
    old_root_var: CVar,
    boundary: &CExpr,
    free_vars: &HashMap<Symbol, CVal>,
    raw_ops: &HashMap<Symbol, (String, Ty, Vec<(String, String)>)>,
    call_units: &HashSet<String>,
    struct_ops: &HashMap<Symbol, (String, Vec<String>, Ty)>,
    field_ops: &HashMap<Symbol, (Ty, String, Ty)>,
    array_ops: &HashMap<Symbol, Ty>,
    array_repeat_ops: &HashMap<Symbol, Ty>,
    load_ops: &HashMap<Symbol, (Ty, Ty)>,
    fresh: &FreshVars,
) -> CExpr {
    let no_substitution = HashMap::new();
    let tables = OpTables { free_vars, raw_ops, call_units, struct_ops, field_ops, array_ops, array_repeat_ops, load_ops, param_substitution: &no_substitution };
    let memo = RefCell::new(HashMap::new());
    rebuild(recexpr, root, fresh, &tables, &memo, &|final_val| substitute_var(boundary, old_root_var, &final_val))
}

/// Replaces every occurrence of `CVal::Var(from)` with `to`, throughout
/// `expr` — used by `rebuild_segment` to patch the boundary's own
/// reference to the segment's original final value.
fn substitute_var(expr: &CExpr, from: CVar, to: &CVal) -> CExpr {
    let sub = |v: &CVal| if matches!(v, CVal::Var(cv) if *cv == from) { to.clone() } else { v.clone() };
    match expr {
        CExpr::LetPrim { var, ty, op, args, cont } => CExpr::LetPrim {
            var: *var,
            ty: ty.clone(),
            op: op.clone(),
            args: args.iter().map(sub).collect(),
            cont: Box::new(substitute_var(cont, from, to)),
        },
        CExpr::App { func, args } => CExpr::App { func: sub(func), args: args.iter().map(sub).collect() },
        CExpr::Fix { defs, body } => CExpr::Fix {
            defs: defs
                .iter()
                .map(|d| CFunDef {
                    name: d.name.clone(),
                    params: d.params.clone(),
                    body: substitute_var(&d.body, from, to),
                    carried_types: d.carried_types.clone(),
                })
                .collect(),
            body: Box::new(substitute_var(body, from, to)),
        },
        CExpr::If { cond, then_branch, else_branch } => CExpr::If {
            cond: sub(cond),
            then_branch: Box::new(substitute_var(then_branch, from, to)),
            else_branch: Box::new(substitute_var(else_branch, from, to)),
        },
    }
}

// ---------------------------------------------------------------- whole-program optimization pass (Stage 6)

use crate::cps::CpsProgram;
use egg::{AstSize, Extractor, Runner};

/// Every `CVal::Var` occurrence anywhere in `expr` — a binder itself
/// (`LetPrim`'s own `var`, a `Fix`-local `CFunDef`'s own `params`) is never
/// collected, only actual references, the same distinction `substitute_var`
/// already draws implicitly.
fn collect_var_refs(expr: &CExpr, out: &mut HashSet<CVar>) {
    fn note(v: &CVal, out: &mut HashSet<CVar>) {
        if let CVal::Var(cv) = v {
            out.insert(*cv);
        }
    }
    match expr {
        CExpr::LetPrim { args, cont, .. } => {
            for a in args {
                note(a, out);
            }
            collect_var_refs(cont, out);
        }
        CExpr::App { func, args } => {
            note(func, out);
            for a in args {
                note(a, out);
            }
        }
        CExpr::Fix { defs, body } => {
            for d in defs {
                collect_var_refs(&d.body, out);
            }
            collect_var_refs(body, out);
        }
        CExpr::If { cond, then_branch, else_branch } => {
            note(cond, out);
            collect_var_refs(then_branch, out);
            collect_var_refs(else_branch, out);
        }
    }
}

/// The one segment-bound `CVar` `boundary` needs reconstructed, or `None`
/// if there isn't exactly one — either `boundary` doesn't reference the
/// segment at all (nothing worth optimizing), or it references *more than
/// one* segment-bound var, which `rebuild_segment` has no mechanism to
/// re-bind (extraction only ever reconstructs the one DAG needed to produce
/// a single root's value, under fresh `CVar` numbering — any other original
/// binding the boundary still needs would go dangling once the segment is
/// replaced). Skipping in that case is a correctness requirement, not just
/// a missed optimization — it's also what closes a struct-aliasing question
/// raised while extending this module to translate `Struct`/`Field`:
/// `PrimOp::FieldStore`'s own doc comment ("a struct is a stable reference")
/// plus `mlir_lower.rs::alloc_struct`'s real heap allocation mean two
/// hashconsed-together `Struct` constructions really could differ in
/// identity if something later `FieldStore`s into one and reads back
/// through the other — but the *only* way that divergence could ever be
/// observed is if the boundary still holds live references to *both* of
/// them, which "exactly one referenced segment var" already forecloses
/// outright. No separate aliasing guard needed.
///
/// Subsumes the old, narrower rule (the boundary being exactly `App{args:
/// [Var(v)]}` — a whole function body translating all the way to its own
/// final return) as one instance of this: that shape always references
/// exactly one var too, so every previously-optimizable function still is.
fn segment_root_var(boundary: &CExpr, env: &HashMap<CVar, egg::Id>) -> Option<CVar> {
    let mut refs = HashSet::new();
    collect_var_refs(boundary, &mut refs);
    let mut in_segment = refs.into_iter().filter(|v| env.contains_key(v));
    let root = in_segment.next()?;
    in_segment.next().is_none().then_some(root)
}

fn max_cvar_in_cexpr(expr: &CExpr, max: &mut CVar) {
    match expr {
        CExpr::LetPrim { var, cont, .. } => {
            *max = (*max).max(*var);
            max_cvar_in_cexpr(cont, max);
        }
        CExpr::App { .. } => {}
        CExpr::Fix { defs, body } => {
            for d in defs {
                for p in &d.params {
                    *max = (*max).max(*p);
                }
                max_cvar_in_cexpr(&d.body, max);
            }
            max_cvar_in_cexpr(body, max);
        }
        CExpr::If { then_branch, else_branch, .. } => {
            max_cvar_in_cexpr(then_branch, max);
            max_cvar_in_cexpr(else_branch, max);
        }
    }
}

/// The highest `CVar` used anywhere in `program` — every parameter and
/// every `LetPrim`/`Fix`-bound variable, across every function.
/// `optimize_program` seeds its own shared `FreshVars` one above this, so a
/// rebuilt segment's own freshly minted variables can never collide with
/// any variable already live anywhere else in the program — necessary
/// because reconstruction happens *after* `convert_program`'s own
/// `FreshVars` (internal to that pass, unreachable from here) has already
/// been discarded, its own count lost with it.
fn max_cvar_in_program(program: &CpsProgram) -> CVar {
    let mut max = 0;
    for f in &program.funcs {
        for p in &f.def.params {
            max = max.max(*p);
        }
        max_cvar_in_cexpr(&f.def.body, &mut max);
    }
    max
}

/// Runs one straight-line-segment optimization pass over every function in
/// `program`: for each function whose own body translates fully up to its
/// final return (`segment_root_var`), builds whatever axiom `Rewrite`s
/// apply to what was actually reached (`axiom_rewrites`), saturates
/// (`egg::Runner`), extracts the cheapest equivalent form (`AstSize`), and
/// rebuilds it back into CPS (`rebuild_segment`) — see the module's own doc
/// comment, and the plan this was built against, for the full design. A
/// function that doesn't translate that far (calls something this module
/// doesn't recognize partway through, branches immediately, has no axioms
/// covering whatever it reached, ...) is left completely unchanged, never
/// partially rewritten.
///
/// Returns the optimized program alongside one flat equivalence-explanation
/// string (`egg::Explanation::get_flat_string`) per function whose own body
/// actually changed — `--dump-cps-equivalences`'s own data source; `hld.md`
/// already names `explain_equivalence` this whole mechanism's primary
/// debugging tool, given a real CLI surface here rather than later.
pub fn optimize_program(program: CpsProgram, registry: &Registry) -> (CpsProgram, Vec<String>) {
    let fresh = FreshVars::starting_at(max_cvar_in_program(&program) + 1);
    let units: HashMap<String, &CTopLevelFn> = program.funcs.iter().map(|f| (f.def.name.clone(), f)).collect();

    let mut new_bodies: HashMap<String, CExpr> = HashMap::new();
    let mut explanations = Vec::new();

    for f in &program.funcs {
        let mut fwd = Forward::default();
        // Must happen before anything is added to `fwd.egraph` (`walk`,
        // right below) -- `EGraph::with_explanations_enabled` itself panics
        // otherwise (found by direct testing), and `explain_equivalence`
        // further down needs it on the exact egraph nodes were added to,
        // not one enabled only after the fact on a `Runner`'s own copy (see
        // this same function's own note on `with_egraph` below).
        fwd.egraph = fwd.egraph.with_explanations_enabled();
        let real_params = &f.def.params[..f.def.params.len() - 1];
        fwd.param_types = real_params.iter().copied().zip(f.param_types.iter().cloned()).collect();
        let boundary = fwd.walk(&f.def.body, &units, &fresh);
        let Some(root_var) = segment_root_var(&boundary, &fwd.env) else { continue };
        let Some(&root_id) = fwd.env.get(&root_var) else { continue };

        let mut rules = axiom_rewrites(registry, &fwd.reached);
        rules.extend(struct_projection_rewrites(&fwd.struct_ops, &fwd.field_ops));
        if rules.is_empty() {
            continue; // nothing this pass knows how to apply to what this function reached
        }

        // Captured from the *pre-run* e-graph, before any rewrite has had a
        // chance to union anything -- capturing it from `runner.egraph`
        // *after* `run` (tried first, found wrong by direct testing) reads
        // back an arbitrary member of the now-merged equivalence class, not
        // the original pre-rewrite tree, silently defeating the "did
        // anything actually change" comparison below.
        let original = fwd.egraph.id_to_expr(root_id);
        let Forward { egraph, free_vars, raw_ops, call_units, struct_ops, field_ops, array_ops, array_repeat_ops, load_ops, .. } = fwd;
        // `with_egraph` *replaces* the runner's own `egraph` field wholesale
        // -- calling it before `with_explanations_enabled` (which mutates
        // the runner's current egraph in place) silently discards the flag,
        // found by direct testing (`explain_equivalence` panicking with
        // "explanations not enabled" despite the call sitting right above
        // it). Order matters here specifically because of that replace-vs-
        // mutate asymmetry between the two builder methods.
        let mut runner = Runner::default().with_egraph(egraph).with_explanations_enabled().run(&rules);
        let root_id = runner.egraph.find(root_id);
        let extractor = Extractor::new(&runner.egraph, AstSize);
        let (_, best) = extractor.find_best(root_id);
        // Compared by `Display` text, not `RecExpr`'s own derived
        // `PartialEq` -- two `RecExpr`s representing the identical tree
        // (even printing identically) can carry a different internal id
        // layout and compare unequal via the derived `PartialEq`, found by
        // direct testing (a `Bitwise::bitnot` segment that never even
        // touches `Ring::add` was showing up as "changed"). The same
        // reasoning `egraph.rs`'s own Stage 2 tests already apply to
        // avoiding parsing a `RecExpr` back from text -- text is this
        // module's own reliable comparison surface, not the derived struct
        // equality.
        if best.to_string() == original.to_string() {
            continue; // saturation ran, but extraction picked the exact original form back -- nothing to report or rebuild
        }

        explanations.push(format!("{}: {}", f.def.name, runner.explain_equivalence(&original, &best).get_flat_string()));
        let rebuilt = rebuild_segment(&best, best.root(), root_var, &boundary, &free_vars, &raw_ops, &call_units, &struct_ops, &field_ops, &array_ops, &array_repeat_ops, &load_ops, &fresh);
        new_bodies.insert(f.def.name.clone(), rebuilt);
    }
    drop(units);

    let funcs = program
        .funcs
        .into_iter()
        .map(|mut f| {
            if let Some(body) = new_bodies.remove(&f.def.name) {
                f.def.body = body;
            }
            f
        })
        .collect();
    (CpsProgram { funcs }, explanations)
}

// ---------------------------------------------------------------- derivative synthesis (Stage 7)

/// `fprime = derive(f);` (`ast.rs`'s own `FnDecl::derivative_of`) — one
/// entry per such declaration, built by the caller from `collect_units`'s
/// own `Vec<ConcreteUnit>` (`u.body`'s own `UnitBody::Derivative(of)`)
/// *before* `convert_program` consumes it — see `main.rs`'s own call site.
pub struct DerivativeRequest {
    pub name: String,
    pub of: String,
}

/// The real body-synthesis half of `doc/backlog.md`'s own auto-diff item —
/// runs once, right after `convert_program` (so every ordinary unit,
/// `f` included, already has its own real, CPS-converted body to build
/// from) and before the first `eliminate_dead_code` (so a synthesized
/// `fprime`'s own real calls are already visible to it). For each request:
/// walks `f`'s own already-converted body into a fresh e-graph exactly the
/// way `optimize_program` already does for an ordinary straight-line
/// segment, adds one `Op("derivative", [root, param])` node per one of
/// `f`'s own real parameters (skipped — a literal zero built directly,
/// no e-graph node needed — for a parameter `f`'s own body never actually
/// references), saturates with `axiom_rewrites`/`struct_projection_
/// rewrites`/`derivative_rewrites` together (the *same* saturation pass a
/// real algebraic identity and a derivative rule can both fire within —
/// this is the actual point of building `derivative` as ordinary e-graph
/// rewriting rather than a separate engine), extracts each with
/// `DerivativeFreeCost`, and rebuilds — `N == 1` becomes `fprime`'s whole
/// body directly, `N > 1` wraps all `N` into one `[T; N]` array (`PrimOp::
/// Array`, the same construction `[a,b,c]` already lowers to).
///
/// A request whose own `of` never became a real unit is silently skipped —
/// `driver.rs`'s own signature-synthesis pass already rejects that case
/// earlier, with a real diagnostic; this only guards against being handed a
/// request malformed in some way that pass didn't anticipate. A request
/// whose body doesn't translate as one clean straight-line segment reaching
/// its own return (`Forward::try_unroll_for_loop`'s own cap exceeded, a
/// branch, an unresolvable loop bound) is *not* silently skipped — it's a
/// real, collected error below, matching the "no rule reaches it" case.
/// Used to `continue` silently instead (`req.name`'s own function simply
/// never added to `new_funcs`, the real failure surfacing three stages
/// downstream in `mlir_lower.rs` as a confusing "call to unknown top-level
/// fn" panic) — found directly, empirically, before this was fixed.
///
/// Returns `Err` — one message per request that couldn't be fully
/// differentiated, not the first found (this project's own "see every
/// conflict, not just the first" posture, `driver.rs::merge_programs`'
/// own doc comment) — whenever saturation still leaves a live `derivative`
/// node in a parameter's own best extraction: no rule (built-in base case
/// or declared `derivative` rule) reaches it, so it's genuinely
/// undifferentiable with what's in scope, not a bug to guess past. Used to
/// panic instead (`rebuild` choking on an `Op` symbol none of its own
/// lookup tables recognized) — real, found directly while building this.
pub fn synthesize_derivatives(program: CpsProgram, requests: &[DerivativeRequest], registry: &Registry) -> Result<CpsProgram, Vec<String>> {
    let fresh = FreshVars::starting_at(max_cvar_in_program(&program) + 1);
    let units: HashMap<String, &CTopLevelFn> = program.funcs.iter().map(|f| (f.def.name.clone(), f)).collect();

    let mut new_funcs = Vec::new();
    let mut errors = Vec::new();

    for req in requests {
        let Some(&of_unit) = units.get(req.of.as_str()) else { continue };

        let mut fwd = Forward::default();
        let real_params = &of_unit.def.params[..of_unit.def.params.len() - 1];
        fwd.param_types = real_params.iter().copied().zip(of_unit.param_types.iter().cloned()).collect();
        let boundary = fwd.walk(&of_unit.def.body, &units, &fresh);
        // Both of these used to `continue` silently (no error pushed) —
        // found directly, empirically, before this fix existed: `derive()`
        // on a function whose own body wasn't fully representable (any
        // `for`/`while` loop, before `Forward::try_unroll_for_loop`; still
        // true today for a too-large or branching loop, or a call to a unit
        // whose own body has a loop) meant `req.name`'s own function was
        // simply never added to `new_funcs`, with the *real* failure
        // surfacing three stages downstream, confusingly, in
        // `mlir_lower.rs` (`panicked ... call to unknown top-level fn
        // \`grad\``) the moment something else tried to call it. Reusing
        // the exact same `errors`/`Result` mechanism the `missing.is_empty()`
        // check just below already established, rather than inventing a
        // second one.
        let Some(root_var) = segment_root_var(&boundary, &fwd.env) else {
            errors.push(format!("cannot derive `{}`: function body is not fully representable (unsupported control flow, e.g. a loop that could not be unrolled, or a branch)", req.name));
            continue;
        };
        let Some(&root_id) = fwd.env.get(&root_var) else {
            errors.push(format!("cannot derive `{}`: function body is not fully representable (unsupported control flow, e.g. a loop that could not be unrolled, or a branch)", req.name));
            continue;
        };

        // `of_unit.def.params`' own trailing entry is `f`'s own `k_ret` —
        // real parameters are everything before it (`CTopLevelFn::k_ret`'s
        // own doc comment).
        let f_params = &of_unit.def.params[..of_unit.def.params.len() - 1];

        // Fresh params for `fprime` itself — never `f`'s own reused (see
        // `OpTables::param_substitution`'s own doc comment for why).
        let mut param_substitution = HashMap::new();
        let mut new_params = Vec::with_capacity(f_params.len() + 1);
        for &p in f_params {
            let np = fresh.var();
            param_substitution.insert(p, np);
            new_params.push(np);
        }
        let k_ret = fresh.var();
        new_params.push(k_ret);

        // One `derivative` node per `f`'s own parameter, in declared order
        // — `None` for a parameter `f`'s own body never actually
        // references anywhere (no `Free` node was ever minted for it). A
        // parameter is a *free* variable from `Forward::walk`'s own
        // perspective (never `LetPrim`-bound, so never an `env` key —
        // that field's own doc comment) — `external_vars` (not `env`) is
        // the table that actually answers "which e-class does this
        // specific external `CVar` correspond to" — found directly: an
        // earlier version of this read `env` instead, which silently
        // produced `None` for every parameter, since `env` only ever
        // tracks a segment's own internally-computed values.
        let derivative_ids: Vec<Option<egg::Id>> =
            f_params.iter().map(|p| fwd.external_vars.get(p).map(|&param_id| fwd.egraph.add(CleaveLang::Op("derivative".into(), vec![root_id, param_id])))).collect();

        let ty_text = of_unit.result.to_string();

        let mut rules = axiom_rewrites(registry, &fwd.reached);
        rules.extend(struct_projection_rewrites(&fwd.struct_ops, &fwd.field_ops));
        rules.extend(construction_derivative_rewrites(&fwd.struct_ops, &fwd.array_ops));
        let (derivative_rules, referenced) = derivative_rewrites(&ty_text, &fwd.reached, registry);
        rules.extend(derivative_rules);

        let Forward { egraph, free_vars, raw_ops, mut call_units, struct_ops, field_ops, array_ops, array_repeat_ops, load_ops, .. } = fwd;
        // A fired `derivative` rule's own RHS can reference a unit `f`'s
        // own body never itself called (the product rule always needs
        // `Ring::add<ty>`, even differentiating a body that only ever
        // multiplies) — `referenced` (`derivative_rewrites`'s own return
        // value, collected while building the rules actually in play)
        // names every one; `rebuild`'s own `Op` handling needs each,
        // confirmed to actually exist somewhere in the whole program
        // (`units`, built once above — `collect_units` already builds a
        // `ConcreteUnit` for *every* non-generic algebra-impl method
        // unconditionally, regardless of whether anything calls it,
        // mirrors `f` itself always getting a unit), inserted into `call_
        // units` to recognize it as a real call rather than panicking on
        // an unrecognized symbol. Found directly, not anticipated: `fn f
        // (x: f32) -> f32 { x * x }` has no `add` call anywhere, yet its
        // own derivative's product-rule expansion needs one.
        call_units.extend(referenced.iter().filter(|name| units.contains_key(name.as_str())).cloned());
        // `Runner::default()`'s own `iter_limit` (30) is tuned for `optimize_
        // program`'s ordinary axiom/constant-fold segments, not this pass:
        // a declared `derivative` rule only ever peels *one* level of
        // nesting per firing (each application still needs its *own*
        // saturation iteration to become visible to the next), so a chain
        // deep enough to need it (e.g. `Forward::try_unroll_for_loop`
        // unrolling a real, multi-iteration loop before differentiating
        // through it) can need more than 30 rounds even though it's
        // otherwise a small, ordinary saturation — found directly: a two-
        // parameter loss function differentiated through just a 2-iteration
        // unrolled loop already hit `IterationLimit(30)` (confirmed via
        // `runner.stop_reason`, not guessed), stopping with `derivative`
        // markers genuinely still reducible, one more round would have
        // continued eliminating them. `node_limit`/`time_limit` raised
        // alongside it for the same reason — `iter_limit` alone would just
        // trade one silent stop for another.
        let runner = Runner::default().with_iter_limit(1000).with_node_limit(1_000_000).with_time_limit(std::time::Duration::from_secs(30)).with_egraph(egraph).run(&rules);
        let tables =
            OpTables { free_vars: &free_vars, raw_ops: &raw_ops, call_units: &call_units, struct_ops: &struct_ops, field_ops: &field_ops, array_ops: &array_ops, array_repeat_ops: &array_repeat_ops, load_ops: &load_ops, param_substitution: &param_substitution };
        let extractor = Extractor::new(&runner.egraph, DerivativeFreeCost);

        // A `derivative` node surviving into the *best* extraction means
        // no rule — built-in base case or declared `derivative` rule —
        // ever reached it: genuinely undifferentiable with what's in
        // scope. Checked *before* `rebuild` (which has no lookup-table
        // entry for a raw `derivative` symbol and would otherwise panic on
        // exactly this) — one clean, real error instead.
        let mut missing: Vec<String> = derivative_ids
            .iter()
            .flatten()
            .filter_map(|&id| undifferentiable_unit(&runner.egraph, &extractor, id))
            .collect();
        if !missing.is_empty() {
            missing.sort();
            missing.dedup();
            errors.push(format!("cannot derive `{}`: no derivative rule for `{}`", req.name, missing.join("`, `")));
            continue;
        }

        let result_ty = of_unit.result.clone();
        let body = build_param_derivatives(&derivative_ids, &of_unit.param_types, &runner.egraph, &extractor, &fresh, &tables, Vec::new(), &|values| {
            finish_derivative_body(values, k_ret, &result_ty, &fresh)
        });

        let n = f_params.len();
        let result = if n == 1 { of_unit.result.clone() } else { Ty::Array(Box::new(of_unit.result.clone()), Box::new(Ty::Const(ConstValue::Int(n as u64)))) };

        new_funcs.push(CTopLevelFn {
            def: CFunDef { name: req.name.clone(), params: new_params, body, carried_types: None },
            param_types: of_unit.param_types.clone(),
            result,
            k_ret,
            origin: None,
        });
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    let mut funcs = program.funcs;
    funcs.extend(new_funcs);
    Ok(CpsProgram { funcs })
}

/// Checks one parameter's own best extraction for a surviving `derivative`
/// node — `None` if it's fully eliminated, `Some(unit)` naming the inner
/// op it's still wrapped around (or a generic fallback if the inner
/// position isn't itself a real op, e.g. a bare `Free`/literal) otherwise.
fn undifferentiable_unit(egraph: &egg::EGraph<CleaveLang, ConstantFold>, extractor: &Extractor<DerivativeFreeCost, CleaveLang, ConstantFold>, id: egg::Id) -> Option<String> {
    let (_, best) = extractor.find_best(egraph.find(id));
    for node in best.as_ref() {
        let CleaveLang::Op(sym, children) = node else { continue };
        if sym.as_str() != "derivative" {
            continue;
        }
        let name = children.first().and_then(|&inner| match &best[inner] {
            CleaveLang::Op(inner_sym, _) => Some(inner_sym.to_string()),
            _ => None,
        });
        return Some(name.unwrap_or_else(|| "<unknown>".to_string()));
    }
    None
}

/// The literal zero of `ty` — `f`'s own parameter `p` was never referenced
/// anywhere in its body, so `d(f)/dp` is trivially this, no e-graph node
/// needed at all.
fn zero_value(ty: &Ty) -> CVal {
    if matches!(ty.to_string().as_str(), "f32" | "f64") {
        CVal::Float(0.0)
    } else {
        CVal::Int(0)
    }
}

/// Sequences one reconstruction per parameter — mirrors `rebuild_args`'s
/// own shape exactly (build inner-to-outer, `k` names "what happens next
/// once every value is collected"), just over `derivative_ids`/`param_
/// types` in lockstep instead of one shared `RecExpr`'s own children:
/// each parameter's own derivative was extracted as an *independent*
/// `RecExpr` (a separate `Extractor::find_best` call per `Id`, not one
/// shared tree), so this can't reuse `rebuild_args` directly.
#[allow(clippy::too_many_arguments)]
fn build_param_derivatives(
    derivative_ids: &[Option<egg::Id>],
    param_types: &[Ty],
    egraph: &egg::EGraph<CleaveLang, ConstantFold>,
    extractor: &Extractor<DerivativeFreeCost, CleaveLang, ConstantFold>,
    fresh: &FreshVars,
    tables: &OpTables,
    acc: Vec<CVal>,
    k: &dyn Fn(Vec<CVal>) -> CExpr,
) -> CExpr {
    let Some((first, rest_ids)) = derivative_ids.split_first() else { return k(acc) };
    let (first_ty, rest_types) = param_types.split_first().expect("derivative_ids and param_types must be the same length");
    match first {
        None => {
            let mut acc2 = acc.clone();
            acc2.push(zero_value(first_ty));
            build_param_derivatives(rest_ids, rest_types, egraph, extractor, fresh, tables, acc2, k)
        }
        Some(id) => {
            let (_, best) = extractor.find_best(egraph.find(*id));
            // A *fresh* memo per parameter, not one shared across all of
            // them — a real bug, found by direct testing (the Jacobian
            // case): each parameter's own derivative is extracted as an
            // *independent* `RecExpr` (a separate `Extractor::find_best`
            // call per `Id`), and different `RecExpr`s' own internal ids
            // are small integers starting fresh from 0 each time, with no
            // relationship to one another — a memo shared across more than
            // one `RecExpr` silently reused the *first* parameter's own
            // cached reconstructions while rebuilding the *second*,
            // collapsing two genuinely different derivatives (`y+1` and
            // `x`) down to the same value in both array slots.
            let memo = RefCell::new(HashMap::new());
            rebuild(&best, best.root(), fresh, tables, &memo, &|v| {
                let mut acc2 = acc.clone();
                acc2.push(v);
                build_param_derivatives(rest_ids, rest_types, egraph, extractor, fresh, tables, acc2, k)
            })
        }
    }
}

/// The tail of a synthesized `fprime`'s own body — `N == 1`'s single
/// reconstructed value tail-calls `k_ret` directly (an ordinary `return`,
/// same idiom every real top-level fn's own body already ends in); `N > 1`
/// wraps every value in one `PrimOp::Array` construction first (the
/// gradient/Jacobian row) before doing the same.
fn finish_derivative_body(values: Vec<CVal>, k_ret: CVar, result_ty: &Ty, fresh: &FreshVars) -> CExpr {
    if let [only] = values.as_slice() {
        return CExpr::App { func: CVal::Var(k_ret), args: vec![only.clone()] };
    }
    let n = values.len();
    let array_ty = Ty::Array(Box::new(result_ty.clone()), Box::new(Ty::Const(ConstValue::Int(n as u64))));
    let arr_var = fresh.var();
    CExpr::LetPrim {
        var: arr_var,
        ty: array_ty,
        op: PrimOp::Array,
        args: values,
        cont: Box::new(CExpr::App { func: CVal::Var(k_ret), args: vec![CVal::Var(arr_var)] }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egg::{EGraph, Extractor, AstSize};

    /// Proves both the `Language` shape and the folding `Analysis` are
    /// usable in isolation, before anything CPS-shaped touches either:
    /// `AlgebraOp("Ring::add<i32>", [Int(2), Int(3)])` folds to `Int(5)`
    /// during construction (`Analysis::modify` fires the moment the e-class
    /// is created, no explicit `rebuild()` needed for a plain `add`-only
    /// graph with no rules run over it), and extraction picks the folded
    /// literal over the original (unfolded) computed form, since a bare
    /// `Int` is strictly cheaper under `AstSize` than an `AlgebraOp` node
    /// with two children.
    #[test]
    fn algebra_op_over_two_int_literals_folds_and_extracts_as_the_literal() {
        let mut egraph: EGraph<CleaveLang, ConstantFold> = EGraph::default();
        let two = egraph.add(CleaveLang::Int(2));
        let three = egraph.add(CleaveLang::Int(3));
        let add = egraph.add(CleaveLang::Op("Ring::add<i32>".into(), vec![two, three]));

        // Compared by `Display` text, not by parsing an expected string back
        // into a `RecExpr` for equality — `CleaveLang`'s own bare-data
        // variants (`Op`/`Int`/`Free`) are ambiguous to parse from a zero-
        // child string (`Op`'s own `Vec<Id>` accepts an empty child list,
        // so a plain string like `"5"` always parses as `Op("5", [])`
        // first, declaration order winning, never reaching `Int`) — found
        // directly by testing. Parsing only matters for `Pattern<L>`, not
        // for this module's own unit tests, so this sidesteps it entirely.
        let extractor = Extractor::new(&egraph, AstSize);
        let (_, best) = extractor.find_best(add);
        assert_eq!(best.to_string(), "5", "expected the folded literal, got {best}");
    }

    /// A node whose own children aren't both known constants doesn't fold
    /// at all — `Analysis::Data` stays `None`, `modify` never fires, and
    /// extraction returns the original (unfoldable) structure unchanged.
    #[test]
    fn algebra_op_over_a_free_variable_does_not_fold() {
        let mut egraph: EGraph<CleaveLang, ConstantFold> = EGraph::default();
        let a = egraph.add(CleaveLang::Free("a".into()));
        let two = egraph.add(CleaveLang::Int(2));
        let add = egraph.add(CleaveLang::Op("Ring::add<i32>".into(), vec![a, two]));

        assert_eq!(egraph[add].data.const_int, None);
        let extractor = Extractor::new(&egraph, AstSize);
        let (_, best) = extractor.find_best(add);
        assert_eq!(best.to_string(), "(Ring::add<i32> a 2)", "got {best}");
    }

    /// `neg` folds via `eval_binop("sub", 0, a)` (`ConstantFold::make`'s own
    /// unary-arity arm) — `Ring::neg<i32>(5)` should constant-fold to the
    /// same `u64` bit pattern `0u64.wrapping_sub(5)` gives, matching `Ring<T>
    /// ::neg`'s own real runtime body (`mlir::arith::subi(0, a)`).
    #[test]
    fn neg_folds_a_single_int_literal() {
        let mut egraph: EGraph<CleaveLang, ConstantFold> = EGraph::default();
        let five = egraph.add(CleaveLang::Int(5));
        let neg = egraph.add(CleaveLang::Op("Ring::neg<i32>".into(), vec![five]));
        assert_eq!(egraph[neg].data.const_int, Some(0u64.wrapping_sub(5)));
    }

    /// `div` is deliberately *not* folded by `ConstantFold` (unlike every
    /// other recognized operator) — see `ConstantFold::make`'s own `"div"`
    /// doc comment: no width/signedness tag exists to tell whether the
    /// operands are meant as signed or unsigned, and division (unlike add/
    /// mul/sub/neg) genuinely differs between the two for a negative
    /// operand. `Ring::div<i32>(6, 3)` must stay unfolded here even though
    /// `eval_binop`'s own `"div"` arm (used by `infer.rs`'s const-generic
    /// evaluator) *can* compute `6/3` — the exclusion is specifically about
    /// this runtime-facing analysis, not about `eval_binop` itself.
    #[test]
    fn div_is_never_folded_by_constant_fold() {
        let mut egraph: EGraph<CleaveLang, ConstantFold> = EGraph::default();
        let six = egraph.add(CleaveLang::Int(6));
        let three = egraph.add(CleaveLang::Int(3));
        let div = egraph.add(CleaveLang::Op("Ring::div<i32>".into(), vec![six, three]));
        assert_eq!(egraph[div].data.const_int, None);
    }

    // -------------------------------------------------- axiom -> Rewrite (Stage 4)

    /// A real axiom, parsed through the real compiler front end (mirrors
    /// `registry.rs`'s own test style) — proves the whole chain end to end:
    /// `axiom_rewrites` builds a real `Rewrite` from it, and *running* that
    /// rewrite over a small hand-built e-graph (`Op("TestRing::add<i32>", [a,
    /// b])`) actually unions it with its own commuted form
    /// (`Op("TestRing::add<i32>", [b, a])`) — not just that a `Rewrite` value
    /// got constructed without error.
    #[test]
    fn a_real_axiom_builds_a_rewrite_that_actually_fires() {
        let (result, _sources) = crate::driver::compile(
            vec![(
                "test.cleave".to_string(),
                "algebra TestRing<T> {
                    fn add(a: T, b: T) -> T;
                    axiom add_commutative(a, b): add(a, b) == add(b, a);
                 }"
                .to_string(),
            )],
            &[],
        );
        let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
        let registry = Registry::build(&program);

        let mut reached = HashMap::new();
        reached.insert("TestRing::add<i32>".to_string(), ("TestRing".to_string(), "add".to_string()));
        let rules = axiom_rewrites(&registry, &reached);
        assert_eq!(rules.len(), 1, "expected exactly one rewrite, for the one reached type");

        let mut egraph: EGraph<CleaveLang, ConstantFold> = EGraph::default();
        let a = egraph.add(CleaveLang::Free("a".into()));
        let b = egraph.add(CleaveLang::Free("b".into()));
        let ab = egraph.add(CleaveLang::Op("TestRing::add<i32>".into(), vec![a, b]));
        let ba = egraph.add(CleaveLang::Op("TestRing::add<i32>".into(), vec![b, a]));
        assert_ne!(egraph.find(ab), egraph.find(ba), "must not be equivalent *before* the rule runs");

        let runner = egg::Runner::default().with_egraph(egraph).run(&rules);
        assert_eq!(runner.egraph.find(ab), runner.egraph.find(ba), "commutativity must make the two forms equivalent");
    }

    /// An axiom belonging to an algebra with no reached instantiation at
    /// all produces no rewrites — nothing to build a rule *for*.
    #[test]
    fn no_rewrites_are_built_for_an_unreached_algebra() {
        let (result, _sources) = crate::driver::compile(
            vec![(
                "test.cleave".to_string(),
                "algebra TestRing<T> {
                    fn add(a: T, b: T) -> T;
                    axiom add_commutative(a, b): add(a, b) == add(b, a);
                 }"
                .to_string(),
            )],
            &[],
        );
        let program = result.unwrap();
        let registry = Registry::build(&program);
        let rules = axiom_rewrites(&registry, &HashMap::new());
        assert!(rules.is_empty());
    }

    // -------------------------------------------------- rebuild (Stage 5)

    /// The full round trip: forward-translate a caller's own real call to
    /// `TestRing::add<i32>(x, 0)`, saturate with an identity axiom
    /// (`add(a, 0) == a`), extract, and rebuild back into CPS — the
    /// reconstructed segment must be `x` *directly* (a `Free` `CVal`, no
    /// `LetPrim`/`Fix` left at all), not the original computed call: the
    /// identity axiom makes `a` strictly cheaper under `AstSize` than
    /// `add(a, 0)`, so extraction picks it deterministically (unlike
    /// commutativity, which doesn't change cost either way — `Stage 4`'s
    /// own test only proves the two forms became *equivalent*, not which
    /// one extraction prefers).
    #[test]
    fn a_fired_rewrite_reconstructs_the_cheaper_form() {
        let (result, _sources) = crate::driver::compile(
            vec![(
                "test.cleave".to_string(),
                "algebra TestRing<T> {
                    fn add(a: T, b: T) -> T;
                    axiom add_zero(a): add(a, 0) == a;
                 }"
                .to_string(),
            )],
            &[],
        );
        let program = result.unwrap();
        let registry = Registry::build(&program);

        let callee = CTopLevelFn {
            def: CFunDef {
                name: "TestRing::add<i32>".to_string(),
                params: vec![10, 11, 12],
                body: CExpr::LetPrim {
                    var: 20,
                    ty: i32_ty(),
                    op: PrimOp::RawMlirOp { op: "arith.addi".to_string(), attrs: vec![] },
                    args: vec![CVal::Var(10), CVal::Var(11)],
                    cont: Box::new(CExpr::App { func: CVal::Var(12), args: vec![CVal::Var(20)] }),
                },
                carried_types: None,
            },
            param_types: vec![i32_ty(), i32_ty()],
            result: i32_ty(),
            k_ret: 12,
            origin: Some(("TestRing".to_string(), "add".to_string())),
        };
        let mut units: HashMap<String, &CTopLevelFn> = HashMap::new();
        units.insert("TestRing::add<i32>".to_string(), &callee);

        // `let v = TestRing::add<i32>(x, 0); <boundary referencing v>` --
        // `x` (CVar 0) is a genuine free variable, from outside this segment.
        let expr = CExpr::Fix {
            defs: vec![CFunDef {
                name: "k$0".to_string(),
                params: vec![5],
                body: CExpr::App { func: CVal::Var(99), args: vec![CVal::Var(5)] },
                carried_types: None,
            }],
            body: Box::new(CExpr::App {
                func: CVal::Label("TestRing::add<i32>".to_string()),
                args: vec![CVal::Var(0), CVal::Int(0), CVal::Label("k$0".to_string())],
            }),
        };

        let mut fwd = Forward::default();
        let boundary = fwd.walk(&expr, &units, &FreshVars::new());
        assert!(matches!(boundary, CExpr::App { .. }));
        let root_var: CVar = 5;
        let root_id = fwd.env[&root_var];

        let rules = axiom_rewrites(&registry, &fwd.reached);
        assert_eq!(rules.len(), 1);

        let Forward { egraph, free_vars, raw_ops, call_units, struct_ops, field_ops, array_ops, array_repeat_ops, load_ops, .. } = fwd;
        let runner = egg::Runner::default().with_egraph(egraph).run(&rules);
        let extractor = Extractor::new(&runner.egraph, AstSize);
        let (_, best) = extractor.find_best(runner.egraph.find(root_id));

        let fresh = FreshVars::new();
        let rebuilt = rebuild_segment(&best, best.root(), root_var, &boundary, &free_vars, &raw_ops, &call_units, &struct_ops, &field_ops, &array_ops, &array_repeat_ops, &load_ops, &fresh);

        // Everything folded away: the reconstructed form is a bare `App`
        // (the boundary itself), its own reference to `root_var` patched
        // directly to `x`'s own original `CVal` (`Var(0)`) -- no `LetPrim`,
        // no `Fix`, the call to `TestRing::add<i32>` never happens at all.
        match &rebuilt {
            CExpr::App { args, .. } => {
                assert!(matches!(args.as_slice(), [CVal::Var(0)]), "expected the boundary patched straight to `x` (Var(0)), got {rebuilt:?}");
            }
            other => panic!("expected a bare App, the whole computation having folded away -- got {other:?}"),
        }
    }

    // -------------------------------------------------- Forward (Stage 3)

    use crate::infer::Ty;

    fn i32_ty() -> Ty {
        Ty::Con("i32".to_string())
    }

    /// A `LetPrim` chain of two raw `mlir::...` ops, ending in a bare tail
    /// `App` (the shape a function's own `return` compiles to) — both ops
    /// translate, `env` ends up with an e-class for each of the two bound
    /// `CVar`s, and the boundary is exactly the trailing `App` (control
    /// transfer to the return continuation, not itself decomposable into an
    /// e-graph node).
    #[test]
    fn a_straight_line_letprim_chain_translates_fully_leaving_only_the_tail_app_as_boundary() {
        let tail = CExpr::App { func: CVal::Var(99), args: vec![CVal::Var(1)] };
        let expr = CExpr::LetPrim {
            var: 0,
            ty: i32_ty(),
            op: PrimOp::RawMlirOp { op: "arith.addi".to_string(), attrs: vec![] },
            args: vec![CVal::Int(2), CVal::Int(3)],
            cont: Box::new(CExpr::LetPrim {
                var: 1,
                ty: i32_ty(),
                op: PrimOp::RawMlirOp { op: "arith.muli".to_string(), attrs: vec![] },
                args: vec![CVal::Var(0), CVal::Int(10)],
                cont: Box::new(tail.clone()),
            }),
        };
        let mut fwd = Forward::default();
        let boundary = fwd.walk(&expr, &HashMap::new(), &FreshVars::new());
        assert!(matches!(boundary, CExpr::App { .. }), "expected the bare tail App as the boundary, got {boundary:?}");
        assert!(fwd.env.contains_key(&0) && fwd.env.contains_key(&1), "both LetPrim-bound vars must have their own e-class");
        // (2 + 3) folds to 5, then 5 * 10 folds to 50 -- constant folding
        // firing automatically as each node is added, not a separate step.
        assert_eq!(fwd.egraph[fwd.env[&1]].data.const_int, Some(50));
    }

    /// `doc/backlog.md`'s own "`CVal::Float` in the e-graph" item — a real
    /// float literal, as an argument to a raw `mlir::...` op, must not stop
    /// translation dead the way it used to (`cval_to_id` returning `None`
    /// for `CVal::Float`, per this module's own former doc comment). Mirrors
    /// `a_straight_line_letprim_chain_translates_fully_leaving_only_the_tail_
    /// app_as_boundary` exactly, just with one operand being a real `f32`
    /// float literal instead of an `Int`.
    #[test]
    fn a_letprim_with_a_float_literal_argument_translates_instead_of_stopping() {
        let f32_ty = || Ty::Con("f32".to_string());
        let tail = CExpr::App { func: CVal::Var(99), args: vec![CVal::Var(0)] };
        let expr = CExpr::LetPrim {
            var: 0,
            ty: f32_ty(),
            op: PrimOp::RawMlirOp { op: "arith.mulf".to_string(), attrs: vec![] },
            args: vec![CVal::Var(10), CVal::Float(2.0)],
            cont: Box::new(tail.clone()),
        };
        let mut fwd = Forward::default();
        let boundary = fwd.walk(&expr, &HashMap::new(), &FreshVars::new());
        assert!(matches!(boundary, CExpr::App { .. }), "expected the bare tail App as the boundary, got {boundary:?}");
        assert!(fwd.env.contains_key(&0), "the LetPrim-bound var must have its own e-class -- the float argument must not have stopped translation");
    }

    /// The other half of `CVal::Float` representability: not just building a
    /// `CleaveLang::Float` node (proven above), but reconstructing one back
    /// into a real `CVal::Float` with the *exact* original value — no
    /// rewrite needs to actually fire for this (unlike `a_fired_rewrite_
    /// reconstructs_the_cheaper_form`'s own Stage-5 precedent): `rebuild`'s
    /// own `CleaveLang::Float` leaf arm is exercised identically whether the
    /// extracted `RecExpr` came from an untouched e-graph or a saturated
    /// one, so extracting straight from `fwd.egraph` with zero rules run is
    /// already the exact same code path `optimize_program` uses whenever a
    /// rewrite *does* fire.
    #[test]
    fn a_float_leaf_round_trips_through_extraction_and_rebuild_unchanged() {
        let f32_ty = || Ty::Con("f32".to_string());
        let tail = CExpr::App { func: CVal::Var(99), args: vec![CVal::Var(0)] };
        let expr = CExpr::LetPrim {
            var: 0,
            ty: f32_ty(),
            op: PrimOp::RawMlirOp { op: "arith.mulf".to_string(), attrs: vec![] },
            args: vec![CVal::Var(10), CVal::Float(2.5)],
            cont: Box::new(tail.clone()),
        };
        let mut fwd = Forward::default();
        let boundary = fwd.walk(&expr, &HashMap::new(), &FreshVars::new());
        let root_var: CVar = 0;
        let root_id = fwd.env[&root_var];

        let extractor = Extractor::new(&fwd.egraph, AstSize);
        let (_, best) = extractor.find_best(root_id);

        let Forward { free_vars, raw_ops, call_units, struct_ops, field_ops, array_ops, array_repeat_ops, load_ops, .. } = fwd;
        let fresh = FreshVars::new();
        let rebuilt = rebuild_segment(&best, best.root(), root_var, &boundary, &free_vars, &raw_ops, &call_units, &struct_ops, &field_ops, &array_ops, &array_repeat_ops, &load_ops, &fresh);

        match &rebuilt {
            CExpr::LetPrim { args, .. } => {
                let has_float = args.iter().any(|a| matches!(a, CVal::Float(f) if *f == 2.5));
                assert!(has_float, "expected the reconstructed args to still carry the exact float literal, got {rebuilt:?}");
            }
            other => panic!("expected a rebuilt LetPrim carrying the raw mlir op, got {other:?}"),
        }
    }

    /// A real call (`emit_call`'s own `Fix`/`App` shape) to a unit whose own
    /// body is straight-line gets recognized transparently — the call
    /// becomes one `Op` node (tagged with the callee's own unit name), and
    /// translation continues straight through into the call's own
    /// continuation, rather than stopping at the `Fix`.
    #[test]
    fn a_real_call_to_a_straight_line_unit_is_transparent() {
        let callee = CTopLevelFn {
            def: CFunDef {
                name: "Ring::add<i32>".to_string(),
                params: vec![10, 11, 12],
                body: CExpr::LetPrim {
                    var: 20,
                    ty: i32_ty(),
                    op: PrimOp::RawMlirOp { op: "arith.addi".to_string(), attrs: vec![] },
                    args: vec![CVal::Var(10), CVal::Var(11)],
                    cont: Box::new(CExpr::App { func: CVal::Var(12), args: vec![CVal::Var(20)] }),
                },
                carried_types: None,
            },
            param_types: vec![i32_ty(), i32_ty()],
            result: i32_ty(),
            k_ret: 12,
            origin: Some(("Ring".to_string(), "add".to_string())),
        };
        let mut units: HashMap<String, &CTopLevelFn> = HashMap::new();
        units.insert("Ring::add<i32>".to_string(), &callee);

        // The *caller's* own shape: `Fix{ k(result) { App(k_ret_of_caller, [result]) } , App(Ring::add<i32>, [a, b, k]) }`.
        let expr = CExpr::Fix {
            defs: vec![CFunDef {
                name: "k$0".to_string(),
                params: vec![5],
                body: CExpr::App { func: CVal::Var(99), args: vec![CVal::Var(5)] },
                carried_types: None,
            }],
            body: Box::new(CExpr::App {
                func: CVal::Label("Ring::add<i32>".to_string()),
                args: vec![CVal::Int(2), CVal::Int(3), CVal::Label("k$0".to_string())],
            }),
        };
        let mut fwd = Forward::default();
        let boundary = fwd.walk(&expr, &units, &FreshVars::new());
        assert!(matches!(boundary, CExpr::App { .. }), "translation must continue past the Fix into k$0's own body, got {boundary:?}");
        assert!(fwd.env.contains_key(&5), "the call's own result var must have its own e-class");
        // 2 + 3 folds to 5 through the *callee's* own translated op.
        assert_eq!(fwd.egraph[fwd.env[&5]].data.const_int, Some(5));
        assert_eq!(
            fwd.reached.get("Ring::add<i32>"),
            Some(&("Ring".to_string(), "add".to_string())),
            "the inlined call's own algebra origin must be recorded for a later axiom-matching stage"
        );
    }

    /// A real call to a unit whose own body is *not* straight-line (it has
    /// a real `If` inside) is correctly rejected — the `Fix` is returned as
    /// the boundary, unchanged, rather than being incorrectly inlined.
    #[test]
    fn a_real_call_to_a_non_straight_line_unit_stops_at_the_fix() {
        let callee = CTopLevelFn {
            def: CFunDef {
                name: "branchy".to_string(),
                params: vec![10, 11],
                body: CExpr::If {
                    cond: CVal::Bool(true),
                    then_branch: Box::new(CExpr::App { func: CVal::Var(11), args: vec![CVal::Int(1)] }),
                    else_branch: Box::new(CExpr::App { func: CVal::Var(11), args: vec![CVal::Int(2)] }),
                },
                carried_types: None,
            },
            param_types: vec![i32_ty()],
            result: i32_ty(),
            k_ret: 11,
            origin: None,
        };
        let mut units: HashMap<String, &CTopLevelFn> = HashMap::new();
        units.insert("branchy".to_string(), &callee);

        let expr = CExpr::Fix {
            defs: vec![CFunDef {
                name: "k$0".to_string(),
                params: vec![5],
                body: CExpr::App { func: CVal::Var(99), args: vec![CVal::Var(5)] },
                carried_types: None,
            }],
            body: Box::new(CExpr::App {
                func: CVal::Label("branchy".to_string()),
                args: vec![CVal::Int(1), CVal::Label("k$0".to_string())],
            }),
        };
        let mut fwd = Forward::default();
        let boundary = fwd.walk(&expr, &units, &FreshVars::new());
        assert!(matches!(boundary, CExpr::Fix { .. }), "a non-straight-line callee must stop translation at the Fix, got {boundary:?}");
        assert!(fwd.env.is_empty(), "nothing should have been translated at all");
    }

    // -------------------------------------------------- for-loop unrolling (Forward::try_unroll_for_loop)

    /// Builds the exact CPS shape `cps.rs`'s own `for` lowering produces for
    /// `for i in 0..end { acc = acc + i; }` — ground-truthed via `--dump-cps`
    /// on that exact source, not guessed — parameterized only over `end`'s
    /// own bound `CVal` so the three tests below can each plug in a
    /// literal-in-range, non-literal, or too-large bound without repeating
    /// the whole shape. `Ring::add<i32>` is registered as a real,
    /// straight-line callee (mirrors `a_real_call_to_a_straight_line_unit_
    /// is_transparent`'s own callee exactly) since the loop body's own two
    /// additions (`acc + i`, `i + 1`) need to resolve as real calls during
    /// unrolling, unlike the loop's own comparison (`Ord::lt<i32>`, never
    /// itself translated — `try_unroll_for_loop` only ever reads its two
    /// operands via `recognize_real_call`, it doesn't need a real callee).
    fn for_loop_fix(end: CVal) -> CExpr {
        const I: CVar = 1; // v436 in the real dump
        const ACC: CVar = 0; // v435
        const COND: CVar = 2; // v437
        const ACC2: CVar = 3; // v438 = acc + i
        const I2: CVar = 4; // v439 = i + 1
        const K_RET: CVar = 99; // f's own outer continuation (v434)

        let then_branch = CExpr::Fix {
            defs: vec![CFunDef {
                name: "k1".to_string(),
                params: vec![ACC2],
                body: CExpr::Fix {
                    defs: vec![CFunDef {
                        name: "k2".to_string(),
                        params: vec![I2],
                        body: CExpr::App { func: CVal::Label("loop$0".to_string()), args: vec![CVal::Var(I2), CVal::Var(ACC2)] },
                        carried_types: None,
                    }],
                    body: Box::new(CExpr::App { func: CVal::Label("Ring::add<i32>".to_string()), args: vec![CVal::Var(I), CVal::Int(1), CVal::Label("k2".to_string())] }),
                },
                carried_types: None,
            }],
            body: Box::new(CExpr::App { func: CVal::Label("Ring::add<i32>".to_string()), args: vec![CVal::Var(ACC), CVal::Var(I), CVal::Label("k1".to_string())] }),
        };
        let else_branch = CExpr::App { func: CVal::Var(K_RET), args: vec![CVal::Var(ACC)] };
        let cond_fix = CExpr::Fix {
            defs: vec![CFunDef {
                name: "k_cond".to_string(),
                params: vec![COND],
                body: CExpr::If { cond: CVal::Var(COND), then_branch: Box::new(then_branch), else_branch: Box::new(else_branch) },
                carried_types: None,
            }],
            body: Box::new(CExpr::App { func: CVal::Label("Ord::lt<i32>".to_string()), args: vec![CVal::Var(I), end, CVal::Label("k_cond".to_string())] }),
        };
        CExpr::Fix {
            defs: vec![CFunDef { name: "loop$0".to_string(), params: vec![I, ACC], body: cond_fix, carried_types: Some(vec![i32_ty(), i32_ty()]) }],
            body: Box::new(CExpr::App { func: CVal::Label("loop$0".to_string()), args: vec![CVal::Int(0), CVal::Int(0)] }),
        }
    }

    fn ring_add_i32_callee() -> CTopLevelFn {
        CTopLevelFn {
            def: CFunDef {
                name: "Ring::add<i32>".to_string(),
                params: vec![10, 11, 12],
                body: CExpr::LetPrim {
                    var: 20,
                    ty: i32_ty(),
                    op: PrimOp::RawMlirOp { op: "arith.addi".to_string(), attrs: vec![] },
                    args: vec![CVal::Var(10), CVal::Var(11)],
                    cont: Box::new(CExpr::App { func: CVal::Var(12), args: vec![CVal::Var(20)] }),
                },
                carried_types: None,
            },
            param_types: vec![i32_ty(), i32_ty()],
            result: i32_ty(),
            k_ret: 12,
            origin: Some(("Ring".to_string(), "add".to_string())),
        }
    }

    /// A literal-bounded `for` loop unrolls fully: translation continues
    /// straight through the loop into the exit continuation (the boundary is
    /// the `App` calling `f`'s own outer `k_ret`, not the `Fix` itself), and
    /// the carried variable (`acc`) ends up bound to the *correctly folded*
    /// final value (`0+0 -> 0`, `0+1 -> 1`, `1+2 -> 3`) — proof the carried
    /// state actually threads iteration-to-iteration rather than each
    /// iteration reusing the same initial binding.
    #[test]
    fn a_literal_bounded_for_loop_unrolls_and_carries_state_correctly() {
        let callee = ring_add_i32_callee();
        let mut units: HashMap<String, &CTopLevelFn> = HashMap::new();
        units.insert("Ring::add<i32>".to_string(), &callee);

        let expr = for_loop_fix(CVal::Int(3));
        let mut fwd = Forward::default();
        let boundary = fwd.walk(&expr, &units, &FreshVars::new());

        assert!(matches!(boundary, CExpr::App { func: CVal::Var(99), .. }), "expected unrolling to continue straight into the loop's own exit continuation, got {boundary:?}");
        const ACC: CVar = 0;
        let acc_id = *fwd.env.get(&ACC).expect("the carried `acc` var must have its own e-class after unrolling");
        assert_eq!(fwd.egraph[acc_id].data.const_int, Some(3), "0+0, then +1, then +2 must fold to 3 -- carried state must thread across iterations, not restart each time");
    }

    /// A non-literal bound (`end` is a free variable, not a `CVal::Int`)
    /// must leave translation exactly as conservative as before this
    /// feature existed: stop at the `Fix`, unchanged, nothing translated.
    #[test]
    fn a_for_loop_with_a_non_literal_bound_is_not_unrolled() {
        let callee = ring_add_i32_callee();
        let mut units: HashMap<String, &CTopLevelFn> = HashMap::new();
        units.insert("Ring::add<i32>".to_string(), &callee);

        let expr = for_loop_fix(CVal::Var(50));
        let mut fwd = Forward::default();
        let boundary = fwd.walk(&expr, &units, &FreshVars::new());

        assert!(matches!(boundary, CExpr::Fix { .. }), "a non-literal bound must bail, leaving the original Fix as the boundary, got {boundary:?}");
        assert!(fwd.env.is_empty(), "nothing should have been translated at all");
    }

    /// A bound exceeding `MAX_UNROLL_ITERATIONS` also bails, exactly like a
    /// non-literal one — trading a clean "stop, unchanged" for an e-graph
    /// blow-up would be worse than not unrolling.
    #[test]
    fn a_for_loop_exceeding_the_unroll_cap_is_not_unrolled() {
        let callee = ring_add_i32_callee();
        let mut units: HashMap<String, &CTopLevelFn> = HashMap::new();
        units.insert("Ring::add<i32>".to_string(), &callee);

        let expr = for_loop_fix(CVal::Int(MAX_UNROLL_ITERATIONS + 1));
        let mut fwd = Forward::default();
        let boundary = fwd.walk(&expr, &units, &FreshVars::new());

        assert!(matches!(boundary, CExpr::Fix { .. }), "a too-large bound must bail, leaving the original Fix as the boundary, got {boundary:?}");
        assert!(fwd.env.is_empty(), "nothing should have been translated at all");
    }

    // -------------------------------------------------- is_straight_line effectfulness (Stage A)

    /// A `LetPrim` chain with no `Fix`/`If` anywhere -- the only thing the
    /// *old* `is_straight_line` ever checked -- must still be rejected if
    /// any of its own `PrimOp`s has a real effect (`Extern`, here: a call to
    /// a separately-compiled C-ABI symbol, e.g. `print`). No `Fix`/`If` is
    /// needed to make this unsound: a callee shaped exactly like this is
    /// `Print<f32>::print`'s own real body.
    #[test]
    fn is_straight_line_rejects_a_letprim_chain_containing_a_real_effect() {
        let body = CExpr::LetPrim {
            var: 20,
            ty: i32_ty(),
            op: PrimOp::Extern { symbol: "print_i32".to_string(), param_types: vec![i32_ty()] },
            args: vec![CVal::Var(10)],
            cont: Box::new(CExpr::App { func: CVal::Var(11), args: vec![CVal::Var(20)] }),
        };
        assert!(!is_straight_line(&body), "an Extern effect must not be judged straight-line, no Fix/If needed to reject it");
    }

    /// The integration-level counterpart: a real call to a unit whose own
    /// impurity comes from an `Extern` op (not an `If`, unlike the sibling
    /// test above) must still stop translation at the `Fix` -- proving the
    /// fix actually reaches `Forward::walk` via `is_straight_line`, not just
    /// the standalone function.
    #[test]
    fn a_real_call_to_an_effectful_unit_stops_at_the_fix() {
        let callee = CTopLevelFn {
            def: CFunDef {
                name: "Print<i32>::print".to_string(),
                params: vec![10, 11],
                body: CExpr::LetPrim {
                    var: 20,
                    ty: i32_ty(),
                    op: PrimOp::Extern { symbol: "print_i32".to_string(), param_types: vec![i32_ty()] },
                    args: vec![CVal::Var(10)],
                    cont: Box::new(CExpr::App { func: CVal::Var(11), args: vec![CVal::Var(20)] }),
                },
                carried_types: None,
            },
            param_types: vec![i32_ty()],
            result: i32_ty(),
            k_ret: 11,
            origin: Some(("Print".to_string(), "print".to_string())),
        };
        let mut units: HashMap<String, &CTopLevelFn> = HashMap::new();
        units.insert("Print<i32>::print".to_string(), &callee);

        let expr = CExpr::Fix {
            defs: vec![CFunDef {
                name: "k$0".to_string(),
                params: vec![5],
                body: CExpr::App { func: CVal::Var(99), args: vec![CVal::Var(5)] },
                carried_types: None,
            }],
            body: Box::new(CExpr::App {
                func: CVal::Label("Print<i32>::print".to_string()),
                args: vec![CVal::Int(1), CVal::Label("k$0".to_string())],
            }),
        };
        let mut fwd = Forward::default();
        let boundary = fwd.walk(&expr, &units, &FreshVars::new());
        assert!(matches!(boundary, CExpr::Fix { .. }), "an effectful callee must stop translation at the Fix, got {boundary:?}");
        assert!(fwd.env.is_empty(), "nothing should have been translated at all");
    }

    /// A body that's real control flow from the very start (`If`) never
    /// gets translated at all — the boundary is the whole original
    /// expression, unchanged, `env` stays empty.
    #[test]
    fn a_body_starting_with_if_translates_nothing() {
        let expr = CExpr::If {
            cond: CVal::Bool(true),
            then_branch: Box::new(CExpr::App { func: CVal::Var(9), args: vec![CVal::Int(1)] }),
            else_branch: Box::new(CExpr::App { func: CVal::Var(9), args: vec![CVal::Int(2)] }),
        };
        let mut fwd = Forward::default();
        let boundary = fwd.walk(&expr, &HashMap::new(), &FreshVars::new());
        assert!(matches!(boundary, CExpr::If { .. }));
        assert!(fwd.env.is_empty());
        assert!(fwd.free_vars.is_empty(), "nothing was translated, so nothing should have been treated as free either");
    }

    // -------------------------------------------------- Struct/Field forward translation (Stage B)

    fn pair_ty() -> Ty {
        Ty::Con("Pair".to_string())
    }

    /// A bare struct construction, no field read afterward -- translates
    /// exactly like a `RawMlirOp` `LetPrim` does: one `Op` node, the bound
    /// var lands in `env`, translation continues into `cont`.
    #[test]
    fn a_struct_construction_translates_and_the_boundary_is_the_tail_app() {
        let expr = CExpr::LetPrim {
            var: 0,
            ty: pair_ty(),
            op: PrimOp::Struct("Pair".to_string(), vec!["x".to_string(), "y".to_string()]),
            args: vec![CVal::Int(21), CVal::Int(0)],
            cont: Box::new(CExpr::App { func: CVal::Var(99), args: vec![CVal::Var(0)] }),
        };
        let mut fwd = Forward::default();
        let boundary = fwd.walk(&expr, &HashMap::new(), &FreshVars::new());
        assert!(matches!(boundary, CExpr::App { .. }), "expected the bare tail App as the boundary, got {boundary:?}");
        assert!(fwd.env.contains_key(&0), "the struct-construction-bound var must have its own e-class");
    }

    /// A field read right after the struct construction that produced it --
    /// both translate, both land in `env`. The read does *not* fold to its
    /// own source literal (`0`, here) -- `ConstantFold` has no
    /// struct-projection knowledge yet, that's a separate, later mechanism
    /// (`struct_projection_rewrites`, Stage F), not something this stage's
    /// own forward translation is expected to do on its own.
    #[test]
    fn a_field_read_after_a_struct_construction_translates_transparently() {
        let expr = CExpr::LetPrim {
            var: 0,
            ty: pair_ty(),
            op: PrimOp::Struct("Pair".to_string(), vec!["x".to_string(), "y".to_string()]),
            args: vec![CVal::Int(21), CVal::Int(0)],
            cont: Box::new(CExpr::LetPrim {
                var: 1,
                ty: i32_ty(),
                op: PrimOp::Field { struct_ty: pair_ty(), field: "y".to_string() },
                args: vec![CVal::Var(0)],
                cont: Box::new(CExpr::App { func: CVal::Var(99), args: vec![CVal::Var(1)] }),
            }),
        };
        let mut fwd = Forward::default();
        let boundary = fwd.walk(&expr, &HashMap::new(), &FreshVars::new());
        assert!(matches!(boundary, CExpr::App { .. }), "expected the bare tail App as the boundary, got {boundary:?}");
        assert!(fwd.env.contains_key(&0) && fwd.env.contains_key(&1), "both the construction and the field read must have their own e-class");
        assert_eq!(
            fwd.egraph[fwd.env[&1]].data.const_int, None,
            "a field read must not constant-fold to its own source literal yet -- no struct-projection knowledge this stage"
        );
    }

    /// Two `Struct` constructions with literally-equal args hashcons to the
    /// *same* e-class automatically -- ordinary e-graph behavior, no rule
    /// needed -- the load-bearing fact behind `segment_root_var`'s own
    /// safety argument (Stage D): merging two constructions is only ever
    /// observable if something later can tell them apart via a live
    /// reference to *both*, which that stage's own "exactly one referenced
    /// segment var" rule forecloses.
    #[test]
    fn two_structurally_identical_struct_constructions_hashcons_to_the_same_eclass() {
        let build = || CExpr::LetPrim {
            var: 1,
            ty: pair_ty(),
            op: PrimOp::Struct("Pair".to_string(), vec!["x".to_string(), "y".to_string()]),
            args: vec![CVal::Int(1), CVal::Int(2)],
            cont: Box::new(CExpr::App { func: CVal::Var(99), args: vec![CVal::Var(1)] }),
        };
        let expr = CExpr::LetPrim {
            var: 0,
            ty: pair_ty(),
            op: PrimOp::Struct("Pair".to_string(), vec!["x".to_string(), "y".to_string()]),
            args: vec![CVal::Int(1), CVal::Int(2)],
            cont: Box::new(build()),
        };
        let mut fwd = Forward::default();
        let _boundary = fwd.walk(&expr, &HashMap::new(), &FreshVars::new());
        assert_eq!(fwd.env[&0], fwd.env[&1], "two structurally identical constructions must hashcons to the same e-class");
    }

    // -------------------------------------------------- Array/ArrayRepeat/Load forward translation

    fn array_ty() -> Ty {
        Ty::Array(Box::new(i32_ty()), Box::new(Ty::Const(crate::infer::ConstValue::Int(2))))
    }

    /// An array literal (`[1, 2]`), mirrors the struct-construction test
    /// exactly -- one `Op` node, the bound var lands in `env`.
    #[test]
    fn an_array_literal_translates_and_the_boundary_is_the_tail_app() {
        let expr = CExpr::LetPrim {
            var: 0,
            ty: array_ty(),
            op: PrimOp::Array,
            args: vec![CVal::Int(1), CVal::Int(2)],
            cont: Box::new(CExpr::App { func: CVal::Var(99), args: vec![CVal::Var(0)] }),
        };
        let mut fwd = Forward::default();
        let boundary = fwd.walk(&expr, &HashMap::new(), &FreshVars::new());
        assert!(matches!(boundary, CExpr::App { .. }), "expected the bare tail App as the boundary, got {boundary:?}");
        assert!(fwd.env.contains_key(&0), "the array-literal-bound var must have its own e-class");
    }

    /// `[0; 2]`, mirrors the array-literal test.
    #[test]
    fn an_array_repeat_translates_and_the_boundary_is_the_tail_app() {
        let expr = CExpr::LetPrim {
            var: 0,
            ty: array_ty(),
            op: PrimOp::ArrayRepeat,
            args: vec![CVal::Int(0), CVal::Int(2)],
            cont: Box::new(CExpr::App { func: CVal::Var(99), args: vec![CVal::Var(0)] }),
        };
        let mut fwd = Forward::default();
        let boundary = fwd.walk(&expr, &HashMap::new(), &FreshVars::new());
        assert!(matches!(boundary, CExpr::App { .. }), "expected the bare tail App as the boundary, got {boundary:?}");
        assert!(fwd.env.contains_key(&0), "the array-repeat-bound var must have its own e-class");
    }

    /// A single-index read right after the array literal that produced it
    /// -- mirrors `a_field_read_after_a_struct_construction_translates_
    /// transparently` exactly, both land in `env`.
    #[test]
    fn a_load_after_an_array_literal_translates_transparently() {
        let expr = CExpr::LetPrim {
            var: 0,
            ty: array_ty(),
            op: PrimOp::Array,
            args: vec![CVal::Int(1), CVal::Int(2)],
            cont: Box::new(CExpr::LetPrim {
                var: 1,
                ty: i32_ty(),
                op: PrimOp::Load { array_ty: array_ty() },
                args: vec![CVal::Var(0), CVal::Int(0)],
                cont: Box::new(CExpr::App { func: CVal::Var(99), args: vec![CVal::Var(1)] }),
            }),
        };
        let mut fwd = Forward::default();
        let boundary = fwd.walk(&expr, &HashMap::new(), &FreshVars::new());
        assert!(matches!(boundary, CExpr::App { .. }), "expected the bare tail App as the boundary, got {boundary:?}");
        assert!(fwd.env.contains_key(&0) && fwd.env.contains_key(&1), "both the array literal and the load must have their own e-class");
    }

    // -------------------------------------------------- Struct/Field backward reconstruction (Stage C)

    /// A struct construction followed by a field read, round-tripped
    /// through extract+`rebuild_segment` with no rewrite rules run at all —
    /// a faithful round trip, nothing foldable yet (no struct-projection
    /// rewrite ran, that's Stage F): the rebuilt segment must still contain
    /// both the construction and the read, in order.
    #[test]
    fn a_struct_field_segment_round_trips_through_extract_and_rebuild() {
        let expr = CExpr::LetPrim {
            var: 0,
            ty: pair_ty(),
            op: PrimOp::Struct("Pair".to_string(), vec!["x".to_string(), "y".to_string()]),
            args: vec![CVal::Int(21), CVal::Int(0)],
            cont: Box::new(CExpr::LetPrim {
                var: 1,
                ty: i32_ty(),
                op: PrimOp::Field { struct_ty: pair_ty(), field: "x".to_string() },
                args: vec![CVal::Var(0)],
                cont: Box::new(CExpr::App { func: CVal::Var(99), args: vec![CVal::Var(1)] }),
            }),
        };
        let mut fwd = Forward::default();
        let boundary = fwd.walk(&expr, &HashMap::new(), &FreshVars::new());
        let root_var: CVar = 1;
        let root_id = fwd.env[&root_var];

        let Forward { egraph, free_vars, raw_ops, call_units, struct_ops, field_ops, array_ops, array_repeat_ops, load_ops, .. } = fwd;
        let extractor = Extractor::new(&egraph, AstSize);
        let (_, best) = extractor.find_best(root_id);

        let fresh = FreshVars::new();
        let rebuilt = rebuild_segment(&best, best.root(), root_var, &boundary, &free_vars, &raw_ops, &call_units, &struct_ops, &field_ops, &array_ops, &array_repeat_ops, &load_ops, &fresh);

        match &rebuilt {
            CExpr::LetPrim { op: PrimOp::Struct(name, fields), cont, .. } => {
                assert_eq!(name, "Pair");
                assert_eq!(fields, &vec!["x".to_string(), "y".to_string()]);
                assert!(
                    matches!(cont.as_ref(), CExpr::LetPrim { op: PrimOp::Field { field, .. }, .. } if field == "x"),
                    "expected the field read to follow the construction, got {cont:?}"
                );
            }
            other => panic!("expected a LetPrim{{Struct}} at the root of the rebuilt segment, got {other:?}"),
        }
    }

    /// The array-typed mirror of `a_struct_field_segment_round_trips_
    /// through_extract_and_rebuild` — same round trip (an array literal
    /// followed by a `Load`), proving the backward translation added for
    /// `Array`/`Load` reconstructs the identical shape.
    #[test]
    fn an_array_load_segment_round_trips_through_extract_and_rebuild() {
        let expr = CExpr::LetPrim {
            var: 0,
            ty: array_ty(),
            op: PrimOp::Array,
            args: vec![CVal::Int(21), CVal::Int(0)],
            cont: Box::new(CExpr::LetPrim {
                var: 1,
                ty: i32_ty(),
                op: PrimOp::Load { array_ty: array_ty() },
                args: vec![CVal::Var(0), CVal::Int(0)],
                cont: Box::new(CExpr::App { func: CVal::Var(99), args: vec![CVal::Var(1)] }),
            }),
        };
        let mut fwd = Forward::default();
        let boundary = fwd.walk(&expr, &HashMap::new(), &FreshVars::new());
        let root_var: CVar = 1;
        let root_id = fwd.env[&root_var];

        let Forward { egraph, free_vars, raw_ops, call_units, struct_ops, field_ops, array_ops, array_repeat_ops, load_ops, .. } = fwd;
        let extractor = Extractor::new(&egraph, AstSize);
        let (_, best) = extractor.find_best(root_id);

        let fresh = FreshVars::new();
        let rebuilt = rebuild_segment(&best, best.root(), root_var, &boundary, &free_vars, &raw_ops, &call_units, &struct_ops, &field_ops, &array_ops, &array_repeat_ops, &load_ops, &fresh);

        match &rebuilt {
            CExpr::LetPrim { op: PrimOp::Array, args, cont, .. } => {
                assert_eq!(args.len(), 2);
                assert!(matches!(cont.as_ref(), CExpr::LetPrim { op: PrimOp::Load { .. }, .. }), "expected the load to follow the construction, got {cont:?}");
            }
            other => panic!("expected a LetPrim{{Array}} at the root of the rebuilt segment, got {other:?}"),
        }
    }

    /// The same struct field read twice (`p.x + p.x`, hand-built directly
    /// as two references to the same bound `CVar`) hashconses to one
    /// shared `Field` e-class with two parent edges into the `addi` node —
    /// without `rebuild`'s own memoization, each parent edge would
    /// independently re-emit a full `LetPrim{Struct}`/`LetPrim{Field}`
    /// chain; with it, the shared e-class is rebuilt exactly once and every
    /// other reference reuses the same fresh `CVar`.
    #[test]
    fn a_shared_field_read_rebuilds_its_own_struct_construction_exactly_once() {
        let expr = CExpr::LetPrim {
            var: 0,
            ty: pair_ty(),
            op: PrimOp::Struct("Pair".to_string(), vec!["x".to_string(), "y".to_string()]),
            args: vec![CVal::Int(1), CVal::Int(2)],
            cont: Box::new(CExpr::LetPrim {
                var: 1,
                ty: i32_ty(),
                op: PrimOp::Field { struct_ty: pair_ty(), field: "x".to_string() },
                args: vec![CVal::Var(0)],
                cont: Box::new(CExpr::LetPrim {
                    var: 2,
                    ty: i32_ty(),
                    op: PrimOp::RawMlirOp { op: "arith.addi".to_string(), attrs: vec![] },
                    args: vec![CVal::Var(1), CVal::Var(1)],
                    cont: Box::new(CExpr::App { func: CVal::Var(99), args: vec![CVal::Var(2)] }),
                }),
            }),
        };
        let mut fwd = Forward::default();
        let boundary = fwd.walk(&expr, &HashMap::new(), &FreshVars::new());
        let root_var: CVar = 2;
        let root_id = fwd.env[&root_var];

        let Forward { egraph, free_vars, raw_ops, call_units, struct_ops, field_ops, array_ops, array_repeat_ops, load_ops, .. } = fwd;
        let extractor = Extractor::new(&egraph, AstSize);
        let (_, best) = extractor.find_best(root_id);

        let fresh = FreshVars::new();
        let rebuilt = rebuild_segment(&best, best.root(), root_var, &boundary, &free_vars, &raw_ops, &call_units, &struct_ops, &field_ops, &array_ops, &array_repeat_ops, &load_ops, &fresh);

        fn count_struct_letprims(expr: &CExpr) -> usize {
            match expr {
                CExpr::LetPrim { op, cont, .. } => (matches!(op, PrimOp::Struct(..)) as usize) + count_struct_letprims(cont),
                CExpr::App { .. } => 0,
                CExpr::Fix { defs, body } => defs.iter().map(|d| count_struct_letprims(&d.body)).sum::<usize>() + count_struct_letprims(body),
                CExpr::If { then_branch, else_branch, .. } => count_struct_letprims(then_branch) + count_struct_letprims(else_branch),
            }
        }
        assert_eq!(
            count_struct_letprims(&rebuilt),
            1,
            "the shared struct construction (read twice via the same field, combined into one addi) must be rebuilt exactly once, not once per reference -- got {rebuilt:?}"
        );
    }

    // -------------------------------------------------- segment_root_var generalization (Stage D)

    /// A boundary that's a `Fix` (not the old, narrower bare-tail-`App`
    /// shape) wrapping a real call, referencing the segment's own bound var
    /// partway through — mirrors exactly what a function's own boundary
    /// looks like once translation stops at an effectful call (e.g.
    /// `print`), the case the old shape-matching `segment_root_var`
    /// couldn't see past at all.
    #[test]
    fn segment_root_var_picks_the_boundarys_own_referenced_var_even_mid_segment() {
        let expr = CExpr::LetPrim {
            var: 0,
            ty: i32_ty(),
            op: PrimOp::RawMlirOp { op: "arith.addi".to_string(), attrs: vec![] },
            args: vec![CVal::Int(2), CVal::Int(3)],
            cont: Box::new(CExpr::App { func: CVal::Var(99), args: vec![CVal::Var(0)] }),
        };
        let mut fwd = Forward::default();
        let _boundary_ignored = fwd.walk(&expr, &HashMap::new(), &FreshVars::new());

        let boundary = CExpr::Fix {
            defs: vec![CFunDef {
                name: "k$0".to_string(),
                params: vec![50],
                body: CExpr::App { func: CVal::Var(51), args: vec![CVal::Var(50)] },
                carried_types: None,
            }],
            body: Box::new(CExpr::App {
                func: CVal::Label("Print<i32>::print".to_string()),
                args: vec![CVal::Var(0), CVal::Label("k$0".to_string())],
            }),
        };
        assert_eq!(segment_root_var(&boundary, &fwd.env), Some(0));
    }

    /// The required safety property: a boundary referencing *two* distinct
    /// segment-bound vars must be refused, not guessed at — `rebuild_segment`
    /// has no mechanism to re-bind more than one original var (extraction
    /// only ever reconstructs the DAG needed for a single root, under fresh
    /// `CVar` numbering), so a second live reference would go dangling once
    /// the segment is replaced.
    #[test]
    fn segment_root_var_refuses_a_boundary_referencing_two_segment_vars() {
        let expr = CExpr::LetPrim {
            var: 0,
            ty: i32_ty(),
            op: PrimOp::RawMlirOp { op: "arith.addi".to_string(), attrs: vec![] },
            args: vec![CVal::Int(2), CVal::Int(3)],
            cont: Box::new(CExpr::LetPrim {
                var: 1,
                ty: i32_ty(),
                op: PrimOp::RawMlirOp { op: "arith.muli".to_string(), attrs: vec![] },
                args: vec![CVal::Var(0), CVal::Int(10)],
                cont: Box::new(CExpr::App { func: CVal::Var(99), args: vec![CVal::Var(1)] }),
            }),
        };
        let mut fwd = Forward::default();
        let _boundary_ignored = fwd.walk(&expr, &HashMap::new(), &FreshVars::new());

        let boundary = CExpr::App { func: CVal::Var(99), args: vec![CVal::Var(0), CVal::Var(1)] };
        assert_eq!(segment_root_var(&boundary, &fwd.env), None);
    }

    // -------------------------------------------------- struct-projection rewrite (Stage F)

    /// A hand-built `struct_ops`/`field_ops` pair (no CPS/`Forward::walk`
    /// involved — this is the rewrite-building function in isolation,
    /// mirroring `axiom_rewrites`'s own Stage 4 test style): the generated
    /// rule must actually union `field(struct(a,b), "y")` with `b` when run
    /// — not just that a `Rewrite` value got constructed without error.
    #[test]
    fn struct_projection_rewrites_unions_a_field_read_with_its_own_constructor_arg() {
        let mut struct_ops = HashMap::new();
        let struct_sym = Symbol::from("struct:Pair:x,y");
        struct_ops.insert(struct_sym, ("Pair".to_string(), vec!["x".to_string(), "y".to_string()], pair_ty()));

        let mut field_ops = HashMap::new();
        let field_sym = Symbol::from("field:Pair:y");
        field_ops.insert(field_sym, (pair_ty(), "y".to_string(), i32_ty()));

        let rules = struct_projection_rewrites(&struct_ops, &field_ops);
        assert_eq!(rules.len(), 1, "expected exactly one rule, for the one field actually read");

        let mut egraph: EGraph<CleaveLang, ConstantFold> = EGraph::default();
        let a = egraph.add(CleaveLang::Free("a".into()));
        let b = egraph.add(CleaveLang::Free("b".into()));
        let s = egraph.add(CleaveLang::Op(struct_sym, vec![a, b]));
        let field_y = egraph.add(CleaveLang::Op(field_sym, vec![s]));
        assert_ne!(egraph.find(field_y), egraph.find(b), "must not be equivalent before the rule runs");

        let runner = egg::Runner::default().with_egraph(egraph).run(&rules);
        assert_eq!(runner.egraph.find(field_y), runner.egraph.find(b), "field(struct(a,b), y) must union with b");
    }

    /// A field that was constructed but never actually read anywhere
    /// reached gets no rewrite — nothing to build a rule *for*, mirrors
    /// `axiom_rewrites`'s own `no_rewrites_are_built_for_an_unreached_algebra`.
    #[test]
    fn no_projection_rewrite_is_built_for_a_field_never_read() {
        let mut struct_ops = HashMap::new();
        struct_ops.insert(Symbol::from("struct:Pair:x,y"), ("Pair".to_string(), vec!["x".to_string(), "y".to_string()], pair_ty()));
        let rules = struct_projection_rewrites(&struct_ops, &HashMap::new());
        assert!(rules.is_empty());
    }

    // -------------------------------------------------- derivative (auto-diff)

    fn extract_best(egraph: &EGraph<CleaveLang, ConstantFold>, id: Id) -> String {
        let extractor = Extractor::new(egraph, DerivativeFreeCost);
        extractor.find_best(id).1.to_string()
    }

    /// An empty (but real) `Registry` -- the three base-case tests below
    /// need no declared `derivative` rule at all, only `derivative_
    /// rewrites`'s own signature (which now always takes a real `Registry`).
    fn empty_registry() -> Registry {
        Registry::build(&crate::ast::Program { items: Vec::new() })
    }

    /// `derivative(x,x) -> 1` -- the trivial base case: no declared rule
    /// needed at all (the identity function's own derivative is exactly
    /// this, with nothing else in sight).
    #[test]
    fn derivative_of_a_variable_with_respect_to_itself_is_one() {
        let reg = empty_registry();
        let (rules, _) = derivative_rewrites("f32", &HashMap::new(), &reg);
        let mut egraph: EGraph<CleaveLang, ConstantFold> = EGraph::default();
        let x = egraph.add(CleaveLang::Free("x".into()));
        let d = egraph.add(CleaveLang::Op("derivative".into(), vec![x, x]));

        let runner = egg::Runner::default().with_egraph(egraph).run(&rules);
        assert_eq!(extract_best(&runner.egraph, runner.egraph.find(d)), "1");
    }

    /// `derivative(y, x) -> 0` -- a *different* free variable is a leaf
    /// that doesn't depend on `x`.
    #[test]
    fn derivative_of_a_different_free_variable_is_zero() {
        let reg = empty_registry();
        let (rules, _) = derivative_rewrites("f32", &HashMap::new(), &reg);
        let mut egraph: EGraph<CleaveLang, ConstantFold> = EGraph::default();
        let x = egraph.add(CleaveLang::Free("x".into()));
        // `own_ty` (`ConstantFold::known_types`'s own doc comment) — real
        // code populates this via `Forward` before ever adding the `Free`
        // node; a hand-built test e-graph has to do the same for `build_
        // zero` to know `y` is scalar rather than bailing.
        egraph.analysis.known_types.insert(Symbol::from("y"), Ty::Con("f32".to_string()));
        let y = egraph.add(CleaveLang::Free("y".into()));
        let d = egraph.add(CleaveLang::Op("derivative".into(), vec![y, x]));

        let runner = egg::Runner::default().with_egraph(egraph).run(&rules);
        assert_eq!(extract_best(&runner.egraph, runner.egraph.find(d)), "0");
    }

    /// `derivative(3.0, x) -> 0` -- a literal constant is a leaf that
    /// doesn't depend on `x` either.
    #[test]
    fn derivative_of_a_float_literal_is_zero() {
        let reg = empty_registry();
        let (rules, _) = derivative_rewrites("f32", &HashMap::new(), &reg);
        let mut egraph: EGraph<CleaveLang, ConstantFold> = EGraph::default();
        let x = egraph.add(CleaveLang::Free("x".into()));
        let three = egraph.add(CleaveLang::Float(3.0.into()));
        let d = egraph.add(CleaveLang::Op("derivative".into(), vec![three, x]));

        let runner = egg::Runner::default().with_egraph(egraph).run(&rules);
        assert_eq!(extract_best(&runner.egraph, runner.egraph.find(d)), "0");
    }

    /// `derivative-independent-zero`'s own real, empirically-found bug (see
    /// `ConstantFold::Data`'s own `free_deps` doc comment): a *still-
    /// unreduced* `derivative(inner, x)` marker sitting in some e-class (the
    /// rule hasn't fired on that specific occurrence *yet* — normal,
    /// expected mid-saturation state) must never make the analysis think
    /// that e-class "depends on" `x`, just because the marker's own
    /// *second* child happens to *be* `x` by construction — that's the
    /// marker's own syntax, not a real value-level dependency. Reproduces
    /// the exact mechanism directly (no loop/unrolling needed at all):
    /// unions a literal `0.0`'s own e-class with an unreduced
    /// `derivative(0.0, w)` node targeting *the same* e-class — simulating
    /// exactly what a partially-saturated real e-graph looks like whenever
    /// `derivative-independent-zero` hasn't reduced every occurrence yet.
    #[test]
    fn an_unreduced_derivative_markers_own_children_do_not_pollute_free_deps() {
        let mut egraph: EGraph<CleaveLang, ConstantFold> = EGraph::default();
        let w = egraph.add(CleaveLang::Free("w".into()));
        let zero = egraph.add(CleaveLang::Float(0.0.into()));
        let deriv_marker = egraph.add(CleaveLang::Op("derivative".into(), vec![zero, w]));
        egraph.union(zero, deriv_marker);
        egraph.rebuild();
        assert!(
            egraph[zero].data.free_deps.is_empty(),
            "an unreduced `derivative(_, w)` marker merely sitting in `zero`'s own e-class must not be read as `zero` itself depending on `w`, got {:?}",
            egraph[zero].data.free_deps
        );
    }

    /// The real, deeper generalization of the bug above (found *after* the
    /// fix above, while testing a second, independent parameter): a live
    /// e-graph traversal can never be sound here at all, not just "buggy on
    /// derivative markers specifically" — equality saturation itself
    /// routinely unions a value-independent expression (`w * 0.0`) into the
    /// *same* e-class as a value it happens to equal (`0.0`) *regardless of
    /// `w`*, and the more saturation progresses, the more such "mentions
    /// `w`, but doesn't truly depend on it" alternatives accumulate in
    /// *every* e-class — a live "does *any* representation mention `w`"
    /// search becomes wrong more and more often as saturation runs longer.
    /// The `Analysis`-based fix (`ConstantFold::Data::free_deps`, merged by
    /// *intersection*) sidesteps this structurally: as long as *one*
    /// representation (here, the literal `Float(0.0)` added first) proves
    /// independence, later unioning in a `w`-mentioning alternative can
    /// only ever *narrow* the recorded bound, never widen it back.
    #[test]
    fn a_value_equal_but_w_mentioning_representation_does_not_pollute_free_deps() {
        let mut egraph: EGraph<CleaveLang, ConstantFold> = EGraph::default();
        let w = egraph.add(CleaveLang::Free("w".into()));
        let zero = egraph.add(CleaveLang::Float(0.0.into()));
        assert!(egraph[zero].data.free_deps.is_empty(), "a bare literal must start with no recorded dependencies");
        let w_times_zero = egraph.add(CleaveLang::Op("Ring::mul<f32>".into(), vec![w, zero]));
        assert_eq!(egraph[w_times_zero].data.free_deps, HashSet::from([Symbol::from("w")]), "w*0.0's own naive bound must mention w before it's unioned with anything");
        egraph.union(zero, w_times_zero);
        egraph.rebuild();
        assert!(
            egraph[zero].data.free_deps.is_empty(),
            "unioning a value-equal-but-w-mentioning representation must narrow (intersect), not widen, the e-class's own recorded dependency bound, got {:?}",
            egraph[zero].data.free_deps
        );
    }

    /// A real `Ring<f32>` with declared `derivative` rules for `add`/`mul`
    /// (mirrors `a_real_axiom_builds_a_rewrite_that_actually_fires`'s own
    /// precedent: the real front end, not a hand-built `reached` map with
    /// no registry behind it) -- `doc/backlog-done.md`'s own "algebra-
    /// declared rules" item, proven through the actual declared path.
    // `TestRing`, not `Ring` -- `Ring` is the *real* prelude algebra
    // (`num`/`logic` are always folded in, `driver.rs`'s own `PRELUDE_
    // CRATES`, regardless of whether this source `use`s them), so
    // redeclaring its own `add`/`mul` here would collide as a genuine
    // duplicate signature -- same precedent `a_real_axiom_builds_a_
    // rewrite_that_actually_fires` already established.
    fn ring_f32_with_derivative_rules() -> Registry {
        let (result, _sources) = crate::driver::compile(
            vec![(
                "test.cleave".to_string(),
                "algebra TestRing<T> {
                    fn add(a: T, b: T) -> T;
                    fn mul(a: T, b: T) -> T;
                    derivative add(a, b): add(d(a), d(b));
                    derivative mul(a, b): add(mul(a, d(b)), mul(d(a), b));
                 }"
                .to_string(),
            )],
            &[],
        );
        let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
        Registry::build(&program)
    }

    fn ring_f32_reached() -> HashMap<String, (String, String)> {
        let mut reached = HashMap::new();
        reached.insert("TestRing::mul<f32>".to_string(), ("TestRing".to_string(), "mul".to_string()));
        reached.insert("TestRing::add<f32>".to_string(), ("TestRing".to_string(), "add".to_string()));
        reached
    }

    /// The leaf-vs-compound distinction, specifically: `derivative(x*y, x)`
    /// must *not* fall for the "not literally the same e-class as x" trap
    /// -- `x*y`'s own e-class genuinely differs from `x`'s, but the whole
    /// point of the declared product rule is that this still depends on
    /// `x` (reduces to `1*y + 0*x`, i.e. `y` once `Ring::mul<f32>`'s own
    /// identity/zero axioms are in play too -- here, with no axioms
    /// loaded, just confirms it does *not* collapse to the literal `0`
    /// leaf-zero rule would wrongly produce if it weren't leaf-restricted).
    #[test]
    fn derivative_of_a_product_involving_x_does_not_wrongly_collapse_to_zero() {
        let reg = ring_f32_with_derivative_rules();
        let reached = ring_f32_reached();
        let (rules, _) = derivative_rewrites("f32", &reached, &reg);

        let mut egraph: EGraph<CleaveLang, ConstantFold> = EGraph::default();
        let x = egraph.add(CleaveLang::Free("x".into()));
        let y = egraph.add(CleaveLang::Free("y".into()));
        let xy = egraph.add(CleaveLang::Op("TestRing::mul<f32>".into(), vec![x, y]));
        let d = egraph.add(CleaveLang::Op("derivative".into(), vec![xy, x]));

        let runner = egg::Runner::default().with_egraph(egraph).run(&rules);
        let best = extract_best(&runner.egraph, runner.egraph.find(d));
        assert_ne!(best, "0", "declared product rule must not be short-circuited by the leaf-zero rule, got {best}");
    }

    /// The declared product rule fires, and — with the two base cases
    /// folding the inner `derivative(x,x)`/`derivative(y,x)` sub-terms away
    /// for real — the extracted form no longer mentions `derivative` *at
    /// all*: the core claim of this whole feature, the e-graph
    /// progressively eliminating the marker node entirely. `mul(x,0) +
    /// mul(1,y)` is the mathematically correct (if not maximally
    /// simplified) result — folding it further to bare `y` needs `mul_
    /// zero`/`mul_one`-shaped identity axioms `stdlib/num` doesn't declare
    /// today (only `add_zero`) — a real, natural follow-up, not attempted
    /// by this rule set.
    #[test]
    fn derivative_of_x_times_y_with_respect_to_x_eliminates_the_derivative_marker() {
        let reg = ring_f32_with_derivative_rules();
        let reached = ring_f32_reached();
        let (rules, _) = derivative_rewrites("f32", &reached, &reg);

        let mut egraph: EGraph<CleaveLang, ConstantFold> = EGraph::default();
        let x = egraph.add(CleaveLang::Free("x".into()));
        // `own_ty` (`ConstantFold::known_types`'s own doc comment) -- `y`'s
        // own independence from `x` needs it to build a same-shaped zero
        // (`build_zero`), for the product rule's own `d(y)` sub-term to
        // fully reduce.
        egraph.analysis.known_types.insert(Symbol::from("y"), Ty::Con("f32".to_string()));
        let y = egraph.add(CleaveLang::Free("y".into()));
        let xy = egraph.add(CleaveLang::Op("TestRing::mul<f32>".into(), vec![x, y]));
        let d = egraph.add(CleaveLang::Op("derivative".into(), vec![xy, x]));

        let runner = egg::Runner::default().with_egraph(egraph).run(&rules);
        let best = extract_best(&runner.egraph, runner.egraph.find(d));
        assert!(!best.contains("derivative"), "expected every `derivative` marker eliminated, got {best}");
        assert!(best.contains('y'), "expected the surviving expression to still reference y, got {best}");
    }

    /// The declared product rule's own referenced-unit set names `add` —
    /// `f`'s own `x*y` never calls it, but the rule's own RHS needs it.
    #[test]
    fn derivative_rewrites_reports_units_the_declared_rules_reference_but_reached_does_not_include() {
        let reg = ring_f32_with_derivative_rules();
        let mut reached = HashMap::new();
        reached.insert("TestRing::mul<f32>".to_string(), ("TestRing".to_string(), "mul".to_string()));
        let (_, referenced) = derivative_rewrites("f32", &reached, &reg);
        assert!(referenced.contains("TestRing::add<f32>"), "expected the product rule's own referenced-unit set to name add, got {referenced:?}");
    }

    /// No declared `derivative` rule at all -- only the two base rules get
    /// built.
    #[test]
    fn only_base_rules_are_built_when_nothing_is_declared() {
        let reg = empty_registry();
        let (rules, _) = derivative_rewrites("f32", &HashMap::new(), &reg);
        assert_eq!(rules.len(), 2, "expected exactly the two base rules (self, leaf-zero)");
    }

    /// The real, clean "cannot derive" error path — a method with no
    /// `derivative` rule at all, and no fallback (a bare `Op` node, not a
    /// leaf, not reducible by anything): `synthesize_derivatives` used to
    /// panic inside `rebuild` on exactly this shape (an unrecognized
    /// `derivative` symbol surviving extraction); it now returns a real
    /// `Err` naming the specific unit, through the *full* real pipeline
    /// (`crate::driver::compile` -> `collect_units` -> `convert_program`),
    /// not a synthetic e-graph-only setup.
    #[test]
    fn synthesize_derivatives_reports_a_clean_error_instead_of_panicking_when_no_rule_reaches_a_node() {
        let src = "
            algebra Foo<T> { fn bar(x: T) -> T; }
            impl Foo<f32> { fn bar(x) { x } }
            fn f(x: f32) -> f32 { bar(x) }
            fprime = derive(f);
        ";
        let (result, _sources) = crate::driver::compile(vec![("test.cleave".to_string(), src.to_string())], &[]);
        let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
        let registry = Registry::build(&program);
        let units = crate::cps::collect_units(&program, &registry);
        let requests: Vec<DerivativeRequest> = units
            .iter()
            .filter_map(|u| match &u.body {
                crate::cps::UnitBody::Derivative(of) => Some(DerivativeRequest { name: u.name.clone(), of: of.clone() }),
                _ => None,
            })
            .collect();
        let cps_program = crate::cps::convert_program(units);
        let errs = match synthesize_derivatives(cps_program, &requests, &registry) {
            Err(errs) => errs,
            Ok(_) => panic!("expected a real error, not a successfully synthesized fprime"),
        };
        assert_eq!(errs.len(), 1);
        assert!(errs[0].contains("Foo::bar<f32>"), "expected the error to name the undifferentiable unit, got {errs:?}");
    }

    /// The empirically-found bug this whole item's second half fixes: before
    /// it, `derive()`-ing a function whose body contains an untranslatable
    /// `for` loop (here: one exceeding `MAX_UNROLL_ITERATIONS`, still
    /// unrollable-in-principle but deliberately not attempted -- see that
    /// constant's own doc comment) didn't produce an error here at all: the
    /// `segment_root_var`/`env.get` lookups silently `continue`d, `fprime`
    /// was never added to `new_funcs`, and the real crash only surfaced
    /// three stages downstream in `mlir_lower.rs` ("call to unknown
    /// top-level fn `fprime`") the moment something else called it --
    /// reproduced directly, via a real `--run`, before this fix existed.
    #[test]
    fn synthesize_derivatives_reports_a_clean_error_instead_of_silently_dropping_the_function_when_the_body_is_not_fully_representable() {
        let src = "
            fn f(x: f32) -> f32 {
                let mut total = 0.0;
                for i in 0..2000 {
                    total = total + x;
                };
                total
            }
            fprime = derive(f);
        ";
        let (result, _sources) = crate::driver::compile(vec![("test.cleave".to_string(), src.to_string())], &[]);
        let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
        let registry = Registry::build(&program);
        let units = crate::cps::collect_units(&program, &registry);
        let requests: Vec<DerivativeRequest> = units
            .iter()
            .filter_map(|u| match &u.body {
                crate::cps::UnitBody::Derivative(of) => Some(DerivativeRequest { name: u.name.clone(), of: of.clone() }),
                _ => None,
            })
            .collect();
        let cps_program = crate::cps::convert_program(units);
        let errs = match synthesize_derivatives(cps_program, &requests, &registry) {
            Err(errs) => errs,
            Ok(_) => panic!("expected a real error, not a successfully synthesized fprime"),
        };
        assert_eq!(errs.len(), 1);
        assert!(errs[0].contains("cannot derive `fprime`"), "expected the error to name the request, got {errs:?}");
        assert!(errs[0].contains("not fully representable"), "expected the error to explain why, got {errs:?}");
    }
}
