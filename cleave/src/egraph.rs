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
//! `CVal::Float` is out of scope for this first increment — bare `f64` has
//! neither `Ord` nor `Hash` (both required by `define_language!`'s own
//! derives), and adding a dependency (`ordered-float`) isn't worth it before
//! the integer-only path is even proven. A later translator stage should
//! panic clearly on encountering one, not silently mishandle it.

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
#[derive(Default, Clone)]
pub struct ConstantFold;

impl Analysis<CleaveLang> for ConstantFold {
    type Data = Option<u64>;

    fn make(egraph: &mut egg::EGraph<CleaveLang, Self>, enode: &CleaveLang, _id: Id) -> Self::Data {
        let value_of = |id: &Id| egraph[*id].data;
        match enode {
            CleaveLang::Int(n) => Some(*n),
            CleaveLang::Op(op, args) => {
                let name = abstract_op_name(op.as_str())?;
                let [a, b] = args.as_slice() else { return None };
                let a = crate::infer::ConstValue::Int(value_of(a)?);
                let b = crate::infer::ConstValue::Int(value_of(b)?);
                match crate::const_eval::eval_binop(name, a, b)? {
                    crate::infer::ConstValue::Int(n) => Some(n),
                    crate::infer::ConstValue::Bool(_) => None,
                }
            }
            CleaveLang::Bool(_) | CleaveLang::Free(_) => None,
        }
    }

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        egg::merge_option(to, from, |a, b| {
            assert_eq!(*a, b, "constant-fold analysis disagreed with itself on the same e-class's own value");
            DidMerge(false, false)
        })
    }

    fn modify(egraph: &mut egg::EGraph<CleaveLang, Self>, id: Id) {
        if let Some(n) = egraph[id].data {
            let added = egraph.add(CleaveLang::Int(n));
            egraph.union(id, added);
        }
    }
}

// ---------------------------------------------------------------- CPS -> e-graph (forward)

use crate::cps::{CExpr, CFunDef, CTopLevelFn, CVal, CVar, PrimOp};
use crate::infer::Ty;
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
    next_free: u32,
}

impl Default for Forward {
    fn default() -> Self {
        Self {
            egraph: EGraph::default(),
            env: HashMap::new(),
            free_vars: HashMap::new(),
            reached: HashMap::new(),
            call_units: std::collections::HashSet::new(),
            raw_ops: HashMap::new(),
            struct_ops: HashMap::new(),
            field_ops: HashMap::new(),
            array_ops: HashMap::new(),
            array_repeat_ops: HashMap::new(),
            load_ops: HashMap::new(),
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
    pub fn walk(&mut self, expr: &CExpr, units: &HashMap<String, &CTopLevelFn>) -> CExpr {
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
                let id = self.egraph.add(CleaveLang::Op(sym, arg_ids));
                self.env.insert(*var, id);
                self.walk(cont, units)
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
                let id = self.egraph.add(CleaveLang::Op(sym, arg_ids));
                self.env.insert(*var, id);
                self.walk(cont, units)
            }
            CExpr::LetPrim { var, ty, op: PrimOp::Field { struct_ty, field }, args, cont } => {
                let Some(arg_ids) = self.cvals_to_ids(args) else {
                    return expr.clone();
                };
                let symbol = format!("field:{struct_ty}:{field}");
                let sym = Symbol::from(symbol);
                self.field_ops.entry(sym).or_insert_with(|| (struct_ty.clone(), field.clone(), ty.clone()));
                let id = self.egraph.add(CleaveLang::Op(sym, arg_ids));
                self.env.insert(*var, id);
                self.walk(cont, units)
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
                let id = self.egraph.add(CleaveLang::Op(sym, arg_ids));
                self.env.insert(*var, id);
                self.walk(cont, units)
            }
            CExpr::LetPrim { var, ty, op: PrimOp::ArrayRepeat, args, cont } => {
                let Some(arg_ids) = self.cvals_to_ids(args) else {
                    return expr.clone();
                };
                let sym = Symbol::from(format!("array-repeat:{ty}"));
                self.array_repeat_ops.entry(sym).or_insert_with(|| ty.clone());
                let id = self.egraph.add(CleaveLang::Op(sym, arg_ids));
                self.env.insert(*var, id);
                self.walk(cont, units)
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
                let id = self.egraph.add(CleaveLang::Op(sym, arg_ids));
                self.env.insert(*var, id);
                self.walk(cont, units)
            }
            // Any other `PrimOp` (`FieldStore`/`Store`/`Extern`) is a real
            // mutation effect or an external call, never freely reorderable/
            // foldable the way pure arithmetic is -- stop, unchanged, same
            // as any other unrecognized shape.
            CExpr::Fix { defs, body } => {
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
                                return self.walk(rest, units);
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
            CVal::Var(cv) => Some(match self.env.get(cv) {
                Some(&id) => id,
                None => {
                    let sym = Symbol::from(format!("fv{}", self.next_free));
                    self.next_free += 1;
                    self.free_vars.insert(sym, v.clone());
                    self.egraph.add(CleaveLang::Free(sym))
                }
            }),
            CVal::Int(n) => Some(self.egraph.add(CleaveLang::Int(*n))),
            CVal::Bool(b) => Some(self.egraph.add(CleaveLang::Bool(*b))),
            // `Unit` has no e-graph representation (nothing to compute);
            // `Float` is out of scope for this increment (module doc
            // comment); `Label`/`Closure` never appear as an ordinary
            // computed argument. All four simply aren't translatable here.
            CVal::Unit | CVal::Float(_) | CVal::Label(_) | CVal::Closure { .. } => None,
        }
    }
}

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

use crate::ast::{AxiomDecl, Expr, ExprKind};
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
                if let Some(rw) = axiom_to_rewrite(algebra, ty, axiom) {
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

fn axiom_to_rewrite(algebra: &str, ty: &str, axiom: &AxiomDecl) -> Option<Rewrite<CleaveLang, ConstantFold>> {
    let params: HashSet<&str> = axiom.params.iter().map(|p| p.name.as_str()).collect();
    let ExprKind::Call(path, _, args, _) = &axiom.body.kind else { return None };
    let [lhs, rhs] = args.as_slice() else { return None };
    if path.segments.join("::") != "eq" {
        return None; // an axiom body that isn't `lhs == rhs` isn't representable yet
    }
    let mut lhs_ast = PatternAst::default();
    build_pattern(lhs, algebra, ty, &params, &mut lhs_ast)?;
    let mut rhs_ast = PatternAst::default();
    build_pattern(rhs, algebra, ty, &params, &mut rhs_ast)?;
    let name = format!("{}@{algebra}<{ty}>", axiom.name);
    Rewrite::new(name, egg::Pattern::new(lhs_ast), egg::Pattern::new(rhs_ast)).ok()
}

/// Walks one side of an axiom's own equality, building it up as a
/// `PatternAst` node by node (never through string parsing — this module
/// has already hit real ambiguities doing that twice for `CleaveLang`
/// itself, see its own doc comment; building programmatically sidesteps the
/// entire class of problem). A bare `Path` matching one of the axiom's own
/// declared `params` becomes a pattern variable (`?name`); a `Call` becomes
/// an `Op` node tagged with the substituted concrete unit name, its own
/// arguments recursively built the same way; a bare integer/bool literal
/// becomes the matching `CleaveLang` leaf directly (axiom bodies are never
/// type-checked — `registry.rs` retains them as pure, unvalidated data, see
/// its own doc comment — so a literal's own text is parsed directly,
/// `u64`/`bool` only, the same narrowed-to-`Int` scope the rest of this
/// module already has). Anything else (a field access, a struct literal,
/// ...) isn't representable in an axiom body yet — returns `None`,
/// rejecting the whole axiom rather than guessing.
fn build_pattern(expr: &Expr, algebra: &str, ty: &str, params: &HashSet<&str>, ast: &mut PatternAst<CleaveLang>) -> Option<egg::Id> {
    match &expr.kind {
        ExprKind::Path(p) => {
            let name = p.segments.join("::");
            if !params.contains(name.as_str()) {
                return None; // a bare name that isn't one of the axiom's own params -- not representable
            }
            let var = Var::from(Symbol::from(format!("?{name}")));
            Some(ast.add(ENodeOrVar::Var(var)))
        }
        ExprKind::NumberLit { text, .. } => {
            let n: u64 = text.parse().ok()?;
            Some(ast.add(ENodeOrVar::ENode(CleaveLang::Int(n))))
        }
        ExprKind::BoolLit(b) => Some(ast.add(ENodeOrVar::ENode(CleaveLang::Bool(*b)))),
        ExprKind::Call(path, _, call_args, _) => {
            let method = path.segments.join("::");
            let unit_name = format!("{algebra}::{method}<{ty}>");
            let mut ids = Vec::with_capacity(call_args.len());
            for a in call_args {
                ids.push(build_pattern(a, algebra, ty, params, ast)?);
            }
            Some(ast.add(ENodeOrVar::ENode(CleaveLang::Op(unit_name.into(), ids))))
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
        CleaveLang::Bool(b) => k(CVal::Bool(*b)),
        CleaveLang::Free(sym) => {
            let v = tables.free_vars.get(sym).unwrap_or_else(|| panic!("egraph: no original CVal recorded for free symbol `{sym}`"));
            k(v.clone())
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
    let tables = OpTables { free_vars, raw_ops, call_units, struct_ops, field_ops, array_ops, array_repeat_ops, load_ops };
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
        let boundary = fwd.walk(&f.def.body, &units);
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

        assert_eq!(egraph[add].data, None);
        let extractor = Extractor::new(&egraph, AstSize);
        let (_, best) = extractor.find_best(add);
        assert_eq!(best.to_string(), "(Ring::add<i32> a 2)", "got {best}");
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
        let boundary = fwd.walk(&expr, &units);
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
        let boundary = fwd.walk(&expr, &HashMap::new());
        assert!(matches!(boundary, CExpr::App { .. }), "expected the bare tail App as the boundary, got {boundary:?}");
        assert!(fwd.env.contains_key(&0) && fwd.env.contains_key(&1), "both LetPrim-bound vars must have their own e-class");
        // (2 + 3) folds to 5, then 5 * 10 folds to 50 -- constant folding
        // firing automatically as each node is added, not a separate step.
        assert_eq!(fwd.egraph[fwd.env[&1]].data, Some(50));
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
        let boundary = fwd.walk(&expr, &units);
        assert!(matches!(boundary, CExpr::App { .. }), "translation must continue past the Fix into k$0's own body, got {boundary:?}");
        assert!(fwd.env.contains_key(&5), "the call's own result var must have its own e-class");
        // 2 + 3 folds to 5 through the *callee's* own translated op.
        assert_eq!(fwd.egraph[fwd.env[&5]].data, Some(5));
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
        let boundary = fwd.walk(&expr, &units);
        assert!(matches!(boundary, CExpr::Fix { .. }), "a non-straight-line callee must stop translation at the Fix, got {boundary:?}");
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
        let boundary = fwd.walk(&expr, &units);
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
        let boundary = fwd.walk(&expr, &HashMap::new());
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
        let boundary = fwd.walk(&expr, &HashMap::new());
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
        let boundary = fwd.walk(&expr, &HashMap::new());
        assert!(matches!(boundary, CExpr::App { .. }), "expected the bare tail App as the boundary, got {boundary:?}");
        assert!(fwd.env.contains_key(&0) && fwd.env.contains_key(&1), "both the construction and the field read must have their own e-class");
        assert_eq!(
            fwd.egraph[fwd.env[&1]].data, None,
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
        let _boundary = fwd.walk(&expr, &HashMap::new());
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
        let boundary = fwd.walk(&expr, &HashMap::new());
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
        let boundary = fwd.walk(&expr, &HashMap::new());
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
        let boundary = fwd.walk(&expr, &HashMap::new());
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
        let boundary = fwd.walk(&expr, &HashMap::new());
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
        let boundary = fwd.walk(&expr, &HashMap::new());
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
        let boundary = fwd.walk(&expr, &HashMap::new());
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
        let _boundary_ignored = fwd.walk(&expr, &HashMap::new());

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
        let _boundary_ignored = fwd.walk(&expr, &HashMap::new());

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
}
