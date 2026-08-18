//! Type inference — Hindley-Milner Algorithm W, extended to "HM + qualified
//! types" (Mark Jones' *Theory of Qualified Types*): [`Scheme`] is
//! `∀vars. constraints ⇒ ty`, not a bare `∀vars. ty`. This is what lets
//! `fn add(a, b) { a + b }` infer `T: Ring` from nothing but how `a`/`b` are
//! used, with no annotation and no special-casing of `add` — the constraint
//! `add`'s resolution generates for `T` is just an ordinary [`Constraint`]
//! that [`Infer::generalize`] sweeps into `add`'s own scheme alongside the
//! variable itself, exactly like any other free variable.
//!
//! Operator calls (`add`, `sub`, `eq`, ...) resolve against a real
//! [`crate::registry::Registry`] of declared `algebra`/`impl`s
//! (`infer_call`/`infer_algebra_call`) — ambiguous (2+ candidate algebras)
//! is a hard error, a concrete type with no matching `impl` is rejected
//! ([`TypeErrorKind::MissingImpl`]), and *zero* algebras declaring a name
//! (today's reality for most operators, absent a full stdlib) makes the
//! call exactly as "unresolved" as calling any other undeclared name — see
//! `infer_call`'s own doc comment for why an earlier permissive fallback
//! here was removed rather than kept as a bridge.
//!
//! `let`-generalization (real HM polymorphism, `∀a. ...` schemes) *is*
//! built: `Env` maps names to a [`Scheme`], not a bare [`Ty`]. A plain,
//! immutable `let` whose right-hand side is a *syntactic value* (a bare
//! `bool` literal, variable reference, or lambda — see
//! [`is_syntactic_value`]) gets generalized at the binding site
//! ([`Infer::generalize`]); every later reference instantiates a fresh copy
//! ([`Infer::instantiate`]), so e.g. `let id = fn(x) { x }; id(1); id(true);`
//! type-checks even though `id`'s parameter is never annotated. Deliberate
//! restrictions, not oversights:
//! - **`let mut` is never generalized**, regardless of what its value looks
//!   like — a mutable binding can be reassigned at one instantiation's type
//!   and read back at another's, which is exactly the classical ML
//!   ref-cell-polymorphism unsoundness. Simpler and safer than trying to
//!   reason about aliasing through the value's shape.
//! - **A bare number/imaginary literal `let` binding (`let a = 16;`) is
//!   *not* generalized either**, despite being a syntactic value in every
//!   other sense — see [`is_syntactic_value`]'s own doc comment for the real
//!   bug this closes (a literal's own numeric-defaulting eligibility,
//!   `pending_defaults`, is registered once, at the literal itself, never
//!   re-registered for a fresh `instantiate`-produced copy at a later use
//!   site). This is narrower than it sounds: a *function* whose body merely
//!   contains a bare literal (`fn add_one(x) { x + 1 }`) still generalizes
//!   fine, with its `Num` constraint riding along (`generalize`'s own doc
//!   comment) — that's the function's own whole-signature scheme, built by
//!   `callgraph.rs`, an entirely different generalization site than a
//!   `let`-bound *value* directly aliasing a literal's own variable.
//!
//! Also deliberately not done yet, and not hidden:
//! - Mutability checking (`let mut` vs. plain `let`, see `grammar.md`) — the
//!   environment here is types only; mutability is used above only to gate
//!   generalization, never checked for illegal reassignment of a plain `let`.
//! - Cross-function inference for `impl` methods specifically — an `impl`
//!   method body is still inferred in isolation from every *other* `impl`
//!   method or top-level `fn` (though it already resolves its own name via
//!   the ordinary algebra-dispatch mechanism in `infer_call`, so a
//!   self-recursive `impl` method "resolves" today — rightly or wrongly;
//!   see `hld.md`'s circular-primitive-impl discussion for why that's a
//!   separate, still-open problem, not a solved one).
//! - A constraint on a variable that's still abstract *and* never
//!   generalized (e.g. it belongs to a `let mut`) has nowhere further to
//!   travel once its enclosing scope finishes. It's silently unchecked, not
//!   because that's correct, but because there's no further propagation for
//!   it to travel *into* — a real gap, tracked here rather than hidden.
//!
//! Cross-function inference **for top-level `fn`s** — direct self-recursion
//! and arbitrary mutual recursion between separately-declared `fn`s — *is*
//! built, in two layers: `infer_fn` alone (used directly by tests, and by
//! `infer_impl_fn`'s sibling for `impl` methods) handles a function calling
//! *only itself*, by binding its own name to a monomorphic placeholder
//! before inferring its body (see `infer_fn`'s doc comment). The full,
//! "définitif" whole-program case — mutual recursion between two or more
//! distinct `fn`s, in any order of declaration — is `callgraph.rs`, which
//! drives many functions through the *same* underlying mechanism
//! (`infer_fn_raw`, `generalize`) a whole strongly-connected group at a
//! time. See that module's own doc comment for the algorithm.

use crate::ast::*;
use crate::const_eval;
use crate::print::fmt_type;
use crate::registry::Registry;
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------- types

/// Internal type representation used during inference — distinct from
/// `ast::TypeKind` (surface syntax). `Con` is a bare named type (`"i32"`,
/// `"bool"`, a non-generic `struct`'s own name, ...); `App` is `Con`'s
/// generic counterpart (`Complex<f64>`, `Complex<T>` with `T` still a type
/// variable) — kept as a *separate* variant rather than folding into `Con`
/// with an always-present (possibly-empty) argument list, so `unify`'s
/// `Con == Con` case (still the overwhelming common case — every primitive
/// type, every non-generic `struct`) stays a plain string comparison with no
/// argument-list handling to skip past. `App`'s argument list holds *every*
/// argument — type and const alike — positionally, in the generic
/// struct/algebra's own declaration order: `Matrix<f64, 3, 3>` is
/// `App("Matrix", [Con("f64"), Const(3), Const(3)])`, a const-generic
/// argument resolving through the exact same `Ty::Var`/`Ty::Const` machinery
/// an array's own size does (see `Ty::Const`, `ty_from_ast_mapped`'s
/// `GenericArg::Const` arm). No separate list, no filtering — a struct/impl
/// mixing type and const generics gets one uniform slot per declared
/// parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ty {
    Var(TyVar),
    /// A still-*symbolic* pack (`doc/backlog.md`'s own "Variadic generics"
    /// item) — the "however many" counterpart to `Var`'s own "one, not yet
    /// known": `Tensor<T, const Dims...: i32>`'s own declaration-time
    /// target pattern is `App("Tensor", [Var(t), Pack(dims)])`, meant to
    /// unify against a call site's `App("Tensor", [Con("f64"), Const(3),
    /// Const(4), Const(5)])` by matching `t` normally and letting `dims`
    /// absorb *everything remaining* (`[3,4,5]`), whatever that count turns
    /// out to be — `unify`'s own `(App, App)` arm is the only place that
    /// actually does this. Resolves through `Subst`'s own *ordinary*
    /// `bindings` table, exactly like `Var` does (`Subst::bind_pack` just
    /// binds `v -> PackResolved(elems)`, see that variant's own doc
    /// comment for why this — not a separate `TyVar -> Vec<Ty>` table —
    /// turned out to be the right call: it's what lets `monomorphize.rs`'s
    /// own already-existing "read every free var back via `apply(&Ty::
    /// Var(v))`" machinery, and `substitute`'s own 17 existing call sites,
    /// need zero signature changes to become pack-aware). Embedded as an
    /// ordinary node inside `App`'s own args list (never a separate field
    /// on `Scheme`/`ImplTemplate`) so the *existing* `free_vars`-driven
    /// `generalize`/`instantiate` machinery quantifies a pack var for free,
    /// the same way it already does for an ordinary nested `Var`.
    Pack(TyVar),
    /// A *resolved* pack's own concrete elements (`Tensor<f64,3,4,5>`'s own
    /// `Dims` resolving to `[Const(3),Const(4),Const(5)]`) — never built
    /// directly by ordinary code, only ever produced by `unify`'s own
    /// pack-aware `(App, App)` arm binding a `Ty::Pack(v)` (`Subst::
    /// bind_pack`), and consumed by `Subst::apply`/`substitute`'s own
    /// `App` arms, which *splice* it into the enclosing `App`'s own args
    /// list rather than keeping it as one nested element — the same
    /// "however many, flattened in place" shape `Ty::App`'s own arg list
    /// already has for an *ordinary*, non-pack instantiation (`Tensor<f64,
    /// 3,4,5>`'s own `type_args` is just `[f64,3,4,5]`, four flat entries,
    /// no marker anywhere saying "the last three came from a pack" — this
    /// variant only exists transiently, while a pack var's own binding is
    /// being read back or spliced, never as part of a "finished," fully-
    /// substituted `Ty::App`'s own args). A bare, unspliced `PackResolved`
    /// reaching anywhere else (MLIR lowering, `Display`, ...) is a real
    /// bug, not a shape those consumers need to understand — every one of
    /// them treats it as "not fully concrete yet"/an unreachable case,
    /// mirroring how a stray `Ty::Var` reaching codegen already is.
    PackResolved(Vec<Ty>),
    /// A pack's own *length*, as a const-generic value (`Dims.len()`,
    /// `doc/backlog.md`'s own "Variadic generics" item — needed to declare
    /// e.g. `Index<Tensor<T,Dims...>,T>::index`'s own `idx: [i32;
    /// Dims.len()]` parameter, since `K` there has to match whatever rank
    /// `Dims` turns out to be, not a fixed literal). Symbolic exactly like
    /// `Ty::ConstExpr` — folds to a real `Ty::Const(Int(n))` the moment the
    /// underlying pack var resolves (`Subst::apply`/`substitute`, mirroring
    /// `fold_const_expr`'s own "fold when concrete, stay symbolic
    /// otherwise" shape), never resolved eagerly here. Built only by
    /// recognizing the surface shape `<pack-name>.len()` — an ordinary
    /// zero-arg `ExprKind::MethodCall` whose base names an in-scope pack
    /// generic — no new grammar or AST node needed at all: `[i32; Dims.len()]`
    /// already parses as an ordinary array-dimension `expr` today.
    PackLen(TyVar),
    Con(String),
    /// A generic type applied to its own type arguments, *in the generic
    /// struct/algebra's own declaration order* — `Complex<T>` is
    /// `App("Complex", [Var(t)])`, matching `struct Complex<T> { ... }`'s
    /// own single type parameter. Whoever builds one and whoever reads one
    /// back (struct construction and field access, respectively — see
    /// `infer_expr_kind`) must agree on this ordering convention; nothing in
    /// the type itself records which name each argument corresponds to.
    App(String, Vec<Ty>),
    /// A lambda's type — parameter types plus a return type. No currying,
    /// no partial application: matches the surface syntax (`|a, b| ...`
    /// always takes its full argument list at once).
    Fn(Vec<Ty>, Box<Ty>),
    /// A fixed-size array — element type plus size, the size itself a `Ty`
    /// (either a resolved `Const(n)`, or a `Var` while still unknown/being
    /// unified against another array's size, or a mismatch — `[f64; 3]` and
    /// `[f64; 4]` are meant to be a genuine, static type error, not silently
    /// compatible). Kept as its own variant rather than folded into `App`
    /// (`Array("Array", [elem, size])`) so `unify`'s dedicated arm can pair
    /// element-with-element and size-with-size without `App`'s generic
    /// same-length-zip logic accidentally treating a size mismatch as just
    /// another arg mismatch (same message either way today, but element vs.
    /// size are conceptually different failures worth keeping distinguishable
    /// at the `Ty` level even if `Display` doesn't yet say more).
    Array(Box<Ty>, Box<Ty>),
    /// A resolved const-generic value — an array's own size (`[f64; 4]`'s
    /// `4`), but also, generally, *any* const-generic (`const B: bool`'s
    /// `true`/`false`): a const-generic isn't inherently integer-shaped,
    /// only an array's own size slot demands that specifically (see
    /// `ty_from_ast_mapped`'s `TypeKind::Array` arm, which is where the
    /// `Int` constraint actually lives — not on the const-generic mechanism
    /// itself). Lives in the exact same `Ty`/`Subst`/`unify` universe as
    /// ordinary types rather than a separate const-evaluator: a not-yet-known
    /// value is just an ordinary `Ty::Var`, resolved by the same unification
    /// `[f64; N]` unified against `[f64; 4]` already needs to perform anyway.
    /// Only ever meant to appear inside `Array`'s size slot or a
    /// const-generic's own mapped slot — nothing in `Ty` itself enforces
    /// that (see module docs: no kind-tagging on `TyVar`, deliberately;
    /// mixing a const value into a type-shaped position fails structurally
    /// the moment it's unified against anything but another `Var`/`Const`,
    /// which is the only gap tests need to guard, not a run-time tag).
    Const(ConstValue),
    /// A const-generic expression (`N+M`, the desugared shape `+` already
    /// produces everywhere else — see `ast.rs`'s own `Call` doc comment)
    /// whose operands aren't *both* concrete yet — `doc/backlog.md`'s own
    /// "Deferred/symbolic constant folding" item. Operator name matches
    /// `const_eval::eval_binop`'s own desugared names (`"add"`/`"sub"`/
    /// `"mul"`); operands are always const-shaped (`Var`, `Const`, or a
    /// nested `ConstExpr`) — not enforced by the type system, same
    /// permissive-but-structurally-safe posture `Const`'s own doc comment
    /// above already documents for that variant.
    ///
    /// A bare `Ty::Var` const-generic already survives a still-generic
    /// declaration untouched, resolved later by `substitute` once
    /// monomorphization supplies a real concrete value — `ConstExpr` needs
    /// the identical treatment, plus an actual fold step once *both*
    /// operands get there: `Subst::apply` and `substitute` are the only two
    /// places that ever need to attempt it (`unify` calls `Subst::apply` on
    /// both sides before its own match ever runs, so a `ConstExpr` that's
    /// already resolvable collapses to a real `Const` before `unify` needs
    /// to know anything special happened). Deliberately *not* eagerly
    /// "solved" against a single concrete total the way `N+M` against a
    /// literal array's own length might tempt — `N+M=5` alone never
    /// uniquely determines `N`/`M`, and guessing would contradict this
    /// project's own "never silently guessed, always checked" posture
    /// (matches the recent `AmbiguousDispatch` work) — `unify`'s existing
    /// catch-all already rejects `ConstExpr` against a bare `Const` cleanly
    /// once both operands are confirmed still unresolvable, with no new
    /// code needed for that case specifically.
    ConstExpr(String, Box<Ty>, Box<Ty>),
}

/// The value side of `Ty::Const` — a const-generic isn't inherently
/// integer-shaped (see `Ty::Const`'s own doc comment), so this is an enum
/// rather than a bare `u64`, open to whatever concrete literal kinds this
/// language's grammar can spell in const-generic position. `PartialEq` (not
/// a numeric comparison) is what `unify`'s `(Const(a), Const(b))` arm uses —
/// `Int(3)` and `Bool(true)` (or `Int(1)`) must never compare equal just
/// because some numeric encoding would coincide; they're different kinds of
/// value entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstValue {
    Int(u64),
    Bool(bool),
}

impl std::fmt::Display for ConstValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstValue::Int(n) => write!(f, "{n}"),
            ConstValue::Bool(b) => write!(f, "{b}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TyVar(pub u32);

#[derive(Default)]
pub struct TyVarGen(u32);

impl TyVarGen {
    pub fn fresh(&mut self) -> Ty {
        let v = TyVar(self.0);
        self.0 += 1;
        Ty::Var(v)
    }
}

// ---------------------------------------------------------------- substitution

#[derive(Debug, Default, Clone)]
pub struct Subst {
    bindings: HashMap<TyVar, Ty>,
    /// A const generic's own declared width (e.g. `i64` for `const N:
    /// i64`), keyed by whichever variable currently stands for its value —
    /// see `bind`'s own doc comment for why this lives here, alongside the
    /// ordinary bindings, rather than as a separate side-table on `Infer`
    /// (the shape this used to be, before a real bug was found: a variable
    /// merged *into* another one via `bind` used to silently lose its own
    /// entry, since nothing re-keyed it — see `doc/backlog.md`'s own "The
    /// `Ty::Const`/`Ty::Con` architectural tension" entry for the full
    /// story, including the exact reproduction that motivated this).
    const_widths: HashMap<TyVar, Ty>,
}

impl Subst {
    /// Follows variable chains to the current representative type,
    /// recursing into `Fn`'s parameter/return types so a partially-resolved
    /// function type reflects every binding made so far, not just its
    /// outermost shape.
    pub fn apply(&self, ty: &Ty) -> Ty {
        match ty {
            Ty::Var(v) => match self.bindings.get(v) {
                Some(next) => self.apply(next),
                None => ty.clone(),
            },
            // Resolves through the same `bindings` table an ordinary `Var`
            // does (`Subst::bind_pack` binds `v -> PackResolved(elems)`) —
            // still symbolic (unbound) is left as-is, same idle state a
            // still-open `Var` has.
            Ty::Pack(v) => match self.bindings.get(v) {
                Some(next) => self.apply(next),
                None => ty.clone(),
            },
            // Only ever meaningful *inside* an enclosing `App`'s own args
            // list, which splices these elements in place rather than
            // keeping this as one nested element (see the `App` arm just
            // below, and this variant's own doc comment) — re-applies each
            // element for freshness, same reasoning as `Var`'s own chain-
            // following. Reachable bare here only via `apply`'s own direct
            // recursion from that `App` arm, never as a genuinely top-level
            // query.
            Ty::PackResolved(elems) => Ty::PackResolved(elems.iter().map(|e| self.apply(e)).collect()),
            // Folds the moment the underlying pack var resolves — reuses
            // `Ty::Pack`'s own chase above rather than looking `v` up in
            // `bindings` a second, separate way, so it agrees exactly with
            // what `apply(&Ty::Pack(v))` would report.
            Ty::PackLen(v) => match self.apply(&Ty::Pack(*v)) {
                Ty::PackResolved(elems) => Ty::Const(ConstValue::Int(elems.len() as u64)),
                _ => ty.clone(),
            },
            Ty::Con(_) | Ty::Const(_) => ty.clone(),
            Ty::App(name, args) => {
                let mut out = Vec::with_capacity(args.len());
                for a in args {
                    match self.apply(a) {
                        Ty::PackResolved(elems) => out.extend(elems),
                        other => out.push(other),
                    }
                }
                Ty::App(name.clone(), out)
            }
            Ty::Fn(params, ret) => {
                Ty::Fn(params.iter().map(|p| self.apply(p)).collect(), Box::new(self.apply(ret)))
            }
            // `size` resolving to a whole `PackResolved` list means this one
            // syntactic level stands for however many real nesting levels
            // the pack has — expanded into the real nested chain, mirroring
            // `substitute`'s own identical `Ty::Array` fix, needed for the
            // exact same reason (`[value; Dims...]`, `ExprKind::ArrayRepeat`'s
            // own inference arm).
            Ty::Array(elem, size) => {
                let elem = self.apply(elem);
                match self.apply(size) {
                    Ty::PackResolved(dims) => dims.into_iter().rev().fold(elem, |acc, dim| Ty::Array(Box::new(acc), Box::new(dim))),
                    size => Ty::Array(Box::new(elem), Box::new(size)),
                }
            }
            // Fold eagerly the moment both operands resolve all the way to a
            // concrete `Const` — this runs at the top of every `unify` call
            // (see its own doc comment), so a `ConstExpr` pinned concrete by,
            // say, an explicit turbofish earlier in the same inference
            // collapses to a real `Const` before `unify`'s own match ever
            // sees it, with no special-casing needed there. Stays symbolic,
            // with its own operands resolved as far as they currently go,
            // when either one isn't concrete yet — exactly like a bare
            // `Ty::Var` already does.
            Ty::ConstExpr(op, a, b) => fold_const_expr(op, self.apply(a), self.apply(b)),
        }
    }

    /// A const generic's own value var (`v`) can get merged into *another*
    /// variable here (e.g. `for i in N..5` unifies `N`'s own var with `5`'s
    /// literal var) — real bug, found by direct testing: `N`'s own declared
    /// width used to live in a side-table keyed only by `N`'s *original*
    /// var, on `Infer` rather than here, so once `v` stopped being the
    /// reachable root (whichever operand `unify` happens to bind *from*,
    /// not *to* — order-dependent, not something callers control), every
    /// later lookup silently missed it, permanently defaulting the merged
    /// variable to a bare `Ty::Con` instead of leaving it open for
    /// monomorphization's own reverse-unification. Propagating the width
    /// forward here, at the one place a merge actually happens, means
    /// whichever variable the chain eventually resolves through always has
    /// its own entry — no caller needs to know or care which one that ends
    /// up being. `.entry().or_insert()`, not an unconditional overwrite: if
    /// `ty`'s own var is *also* already separately const-tainted (two
    /// distinct const generics unified together, a narrow edge case), the
    /// first one's width wins here, and a real mismatch between the two
    /// still surfaces normally wherever it's actually checked — not
    /// silently overwritten by whichever happened to be on which side.
    fn bind(&mut self, v: TyVar, ty: Ty) {
        if let (Some(width), Ty::Var(v2)) = (self.const_widths.get(&v).cloned(), &ty) {
            self.const_widths.entry(*v2).or_insert(width);
        }
        // A const generic's own value-var, checked against its own declared
        // *type* (an ordinary `Ty::Con`, e.g. `const N: i32` referenced as a
        // bare body value `{ N }`, unified against a caller's own declared
        // `-> i32`) — deliberately never bound here, unlike every other
        // bind. Binding it would permanently collapse its own identity to
        // `Con("i32")`, indistinguishable from any other `i32` value and
        // unrecoverable as "the function's own const generic N" by the time
        // its scheme is built — found by direct testing (`doc/backlog.md`'s
        // own "Explicit turbofish on a const generic" item): `N` never
        // showed up anywhere in the exposed `Ty::Fn(params, ret)` shape
        // `generalize`'s own free-var scan reads, so it was never quantified
        // at all, leaving turbofish with zero declared generics to match
        // `::<3>` against. Leaving `v` unbound here instead means the
        // *value* is still checked for compatibility (`unify`'s own new
        // `Ty::Const`/`Ty::Con` arms handle that once `v` later resolves to
        // a real `Ty::Const`), while `v`'s own identity survives — recovered
        // for real the same way an array's own size slot already is: via
        // `unify`'s ordinary var-to-`Ty::Const` binding, whenever a genuinely
        // concrete value (a turbofish argument, an inferable call-site
        // argument) actually pins it.
        if self.const_widths.contains_key(&v) && matches!(ty, Ty::Con(_)) {
            return;
        }
        self.bindings.insert(v, ty);
    }

    /// `v`'s own declared width, if it's (or has since been merged into) a
    /// const generic's own value slot — `None` for an ordinary type
    /// variable. Looks up `v` directly, not through `apply` first: `bind`'s
    /// own forward propagation already guarantees whichever variable a
    /// caller happens to hold a reference to has its own entry, regardless
    /// of how many further merges it's since been through.
    fn const_width(&self, v: TyVar) -> Option<Ty> {
        self.const_widths.get(&v).cloned()
    }

    fn set_const_width(&mut self, v: TyVar, width: Ty) {
        self.const_widths.insert(v, width);
    }

    /// Binds `v` to its own resolved pack elements — through the *ordinary*
    /// `bindings` table (`Ty::Pack`'s own doc comment explains why this,
    /// not a separate `TyVar -> Vec<Ty>` side table, turned out to be the
    /// right call). `apply(&Ty::Var(v))`/`apply(&Ty::Pack(v))` both read it
    /// straight back via the same lookup an ordinary binding already uses.
    fn bind_pack(&mut self, v: TyVar, elems: Vec<Ty>) {
        self.bind(v, Ty::PackResolved(elems));
    }

    /// Must recurse into `Fn`'s components — a variable can occur *inside* a
    /// function type (`'a = ('a) -> Int`) just as easily as anywhere else;
    /// missing that recursion here would silently defeat the whole point of
    /// having an occurs check once lambdas are in the mix.
    fn occurs(&self, v: TyVar, ty: &Ty) -> bool {
        match self.apply(ty) {
            Ty::Var(v2) => v == v2,
            // A pack var can't occur *inside* another type the way an
            // ordinary var can (it only ever appears as a direct `App`
            // argument, never nested one level deeper by construction) —
            // equality is the whole check, mirroring `Var`'s own arm.
            Ty::Pack(v2) => v == v2,
            Ty::PackResolved(elems) => elems.iter().any(|e| self.occurs(v, e)),
            Ty::PackLen(v2) => v == v2,
            Ty::Con(_) | Ty::Const(_) => false,
            Ty::App(_, args) => args.iter().any(|a| self.occurs(v, a)),
            Ty::Fn(params, ret) => params.iter().any(|p| self.occurs(v, p)) || self.occurs(v, &ret),
            Ty::Array(elem, size) => self.occurs(v, &elem) || self.occurs(v, &size),
            Ty::ConstExpr(_, a, b) => self.occurs(v, &a) || self.occurs(v, &b),
        }
    }
}

/// Failure from `unify` alone — no source location, since unification is a
/// pure type-level operation with no notion of "which expression". Callers
/// (inside `Infer`) attach the location, producing a `TypeError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnifyError {
    Mismatch(Ty, Ty),
    Occurs(TyVar, Ty),
}

impl std::fmt::Display for Ty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ty::Con(name) => write!(f, "{name}"),
            Ty::Var(TyVar(id)) => write!(f, "'t{id}"),
            Ty::Pack(TyVar(id)) => write!(f, "'t{id}..."),
            Ty::PackResolved(elems) => {
                write!(f, "{}", elems.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "))
            }
            Ty::PackLen(TyVar(id)) => write!(f, "'t{id}...len()"),
            Ty::App(name, args) => {
                let args = args.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(", ");
                write!(f, "{name}<{args}>")
            }
            Ty::Fn(params, ret) => {
                let params = params.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(", ");
                write!(f, "({params}) -> {ret}")
            }
            Ty::Array(elem, size) => write!(f, "[{elem}; {size}]"),
            Ty::Const(v) => write!(f, "{v}"),
            Ty::ConstExpr(op, a, b) => write!(f, "{op}({a}, {b})"),
        }
    }
}

/// `()` showing up somewhere it plainly doesn't belong (a mismatch against
/// it, an algebra with no sensible `impl Algebra<()>`) is, in practice,
/// always the same real cause: a block's would-be tail expression was
/// followed by a trailing `;`, silently discarding its value and making the
/// block evaluate to unit instead — see `grammar.md`, "Blocks are
/// expressions". Appended to the relevant `Display` impls below rather than
/// left for the reader to have to already know the rule and reconstruct it
/// from a bare `no impl Num<()>`/`found ()`.
const UNIT_DISCARD_HINT: &str =
    "(note: `()` usually means a block's last expression was followed by a `;`, discarding its value — check whether that `;` should be removed so it becomes the block's tail instead)";

fn is_unit(ty: &Ty) -> bool {
    matches!(ty, Ty::Con(name) if name == "()")
}

impl std::fmt::Display for UnifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnifyError::Mismatch(a, b) => {
                write!(f, "type mismatch: expected `{a}`, found `{b}`")?;
                // Only the "found" side, not "expected" — an *expected* `()`
                // (an `if` with no `else`, a declared `-> ()`) is ordinary
                // and not the discarded-tail situation the hint is about.
                if is_unit(b) && !is_unit(a) {
                    write!(f, " {UNIT_DISCARD_HINT}")?;
                }
                Ok(())
            }
            UnifyError::Occurs(v, t) => write!(f, "infinite type: `'t{}` occurs in `{t}`", v.0),
        }
    }
}

/// Unifies `a` and `b` under `subst`, extending it in place — the standard
/// Robinson unification algorithm, with the occurs check (rejecting e.g.
/// `unify('a, 'a -> Int)`, which would otherwise produce an infinite type).
pub fn unify(subst: &mut Subst, a: &Ty, b: &Ty) -> Result<(), UnifyError> {
    let a = subst.apply(a);
    let b = subst.apply(b);
    match (&a, &b) {
        (Ty::Con(x), Ty::Con(y)) if x == y => Ok(()),
        (Ty::Const(x), Ty::Const(y)) if x == y => Ok(()),
        // A resolved const-generic value against an ordinary type name —
        // e.g. a turbofish-pinned `const N: i32`'s own call-site type,
        // `Ty::Const(Int(3))`, checked against a caller's own declared
        // `-> i32`. Loose compatibility only, not exact-width-checked —
        // matches this codebase's own already-documented limitation
        // elsewhere (`fresh_vars_for_generics`'s own doc comment: nothing
        // here yet distinguishes `const N: i32` from `const M: i64` beyond
        // both being `Int`-typed). Without this, `Subst::bind`'s own
        // identity-preserving skip (see its doc comment) would still let a
        // const generic's own value survive type inference, but comparing
        // that value against any ordinary type it flows into (a caller's
        // own return-type check, an argument position, ...) would hard-fail
        // with no rule to reconcile the two shapes at all.
        (Ty::Const(ConstValue::Int(_)), Ty::Con(n)) | (Ty::Con(n), Ty::Const(ConstValue::Int(_)))
            if matches!(n.as_str(), "i8" | "i16" | "i32" | "i64") =>
        {
            Ok(())
        }
        (Ty::Const(ConstValue::Bool(_)), Ty::Con(n)) | (Ty::Con(n), Ty::Const(ConstValue::Bool(_))) if n == "bool" => {
            Ok(())
        }
        (Ty::Var(v1), Ty::Var(v2)) if v1 == v2 => Ok(()),
        // `Ty::Pack` reaching `unify` bare (not nested inside an enclosing
        // `App`'s own args list — that case is handled entirely separately,
        // below) was never expected before `[value; Dims...]` (`ExprKind::
        // ArrayRepeat`'s own pack-count support, `doc/backlog.md`'s own
        // "Toward a matmul-based tensorial XOR" follow-on): a `Ty::Array`'s
        // own `size` field can now genuinely be a still-open `Ty::Pack`
        // (`[T; Dims...]`'s own inferred type before `Dims` resolves), and
        // the very same array-repeat expression's own declared-vs-inferred
        // check (`infer_struct_lit_with_pack`) unifies *two* independently-
        // built `Ty::Array(_, Pack(v))` values against each other — both
        // reading the *same* `self.active_generics["Dims"]` var, so always
        // the *same* `v` in practice. Only that narrow, safe case is handled
        // here (mirroring `Ty::Var`'s own identical same-var shortcut just
        // above) — deliberately not a general "bind a Pack against anything"
        // arm the way `Ty::Var`'s own below is: a `Pack` conceptually stands
        // for a *list* of types, so binding it against an arbitrary
        // non-list `Ty` the way a plain type variable can would be
        // unsound, not just unneeded. Two *different*, still-open packs
        // meeting here falls through to the ordinary `Mismatch` below, a
        // real, flagged gap (never reached by anything currently reachable)
        // rather than a guess.
        (Ty::Pack(v1), Ty::Pack(v2)) if v1 == v2 => Ok(()),
        (Ty::Var(v), _) => {
            if subst.occurs(*v, &b) {
                return Err(UnifyError::Occurs(*v, b));
            }
            subst.bind(*v, b);
            Ok(())
        }
        (_, Ty::Var(v)) => {
            if subst.occurs(*v, &a) {
                return Err(UnifyError::Occurs(*v, a));
            }
            subst.bind(*v, a);
            Ok(())
        }
        (Ty::Fn(p1, r1), Ty::Fn(p2, r2)) if p1.len() == p2.len() => {
            for (x, y) in p1.iter().zip(p2) {
                unify(subst, x, y)?;
            }
            unify(subst, r1, r2)
        }
        // `doc/backlog.md`'s own "Variadic generics" item: same name
        // required exactly as before, but arity is no longer required to
        // match exactly — if one side's own args end in a still-open
        // `Ty::Pack` (already `subst.apply`'d above, so a *resolved* pack
        // would already have been spliced flat into this very list, never
        // reaching this match as a bare `Pack` at all), the non-pack
        // prefix unifies pairwise as usual and the pack absorbs whatever
        // the *other* side has left over, however many that turns out to
        // be. Neither side ending in an open pack is byte-for-byte the
        // original behavior (guarded, not just reasoned about — see the
        // `a1.len() != a2.len()` check inside the `(None, None)` arm,
        // still a real, immediate `Mismatch`). Both sides ending in an
        // open pack (two still-symbolic declarations meeting each other)
        // is deliberately out of scope for this pass — falls through to
        // `Mismatch`, a known, flagged gap, not silently wrong.
        (Ty::App(n1, a1), Ty::App(n2, a2)) if n1 == n2 => {
            let trailing_pack = |args: &[Ty]| match args.last() {
                Some(Ty::Pack(v)) => Some(*v),
                _ => None,
            };
            match (trailing_pack(a1), trailing_pack(a2)) {
                (None, None) => {
                    if a1.len() != a2.len() {
                        return Err(UnifyError::Mismatch(a.clone(), b.clone()));
                    }
                    for (x, y) in a1.iter().zip(a2) {
                        unify(subst, x, y)?;
                    }
                    Ok(())
                }
                (Some(v1), Some(v2)) if v1 == v2 => {
                    if a1.len() != a2.len() {
                        return Err(UnifyError::Mismatch(a.clone(), b.clone()));
                    }
                    for (x, y) in a1[..a1.len() - 1].iter().zip(&a2[..a2.len() - 1]) {
                        unify(subst, x, y)?;
                    }
                    Ok(())
                }
                (Some(_), Some(_)) => Err(UnifyError::Mismatch(a.clone(), b.clone())),
                (Some(v), None) => {
                    let prefix_len = a1.len() - 1;
                    if a2.len() < prefix_len {
                        return Err(UnifyError::Mismatch(a.clone(), b.clone()));
                    }
                    for (x, y) in a1[..prefix_len].iter().zip(&a2[..prefix_len]) {
                        unify(subst, x, y)?;
                    }
                    subst.bind_pack(v, a2[prefix_len..].to_vec());
                    Ok(())
                }
                (None, Some(v)) => {
                    let prefix_len = a2.len() - 1;
                    if a1.len() < prefix_len {
                        return Err(UnifyError::Mismatch(a.clone(), b.clone()));
                    }
                    for (x, y) in a1[..prefix_len].iter().zip(&a2[..prefix_len]) {
                        unify(subst, x, y)?;
                    }
                    subst.bind_pack(v, a1[prefix_len..].to_vec());
                    Ok(())
                }
            }
        }
        (Ty::Array(e1, s1), Ty::Array(e2, s2)) => {
            unify(subst, e1, e2)?;
            unify(subst, s1, s2)
        }
        // Two still-symbolic expressions with the same operator — mirrors
        // `Ty::App`'s own same-name-then-zip-args arm just above. Both sides
        // already went through `subst.apply` above, so if either one were
        // actually resolvable it would already be a plain `Ty::Const` by
        // this point (`fold_const_expr`, called from `Subst::apply`) — this
        // arm only ever runs when both genuinely still have a free operand.
        // A `ConstExpr` unified directly against a concrete `Const` (or
        // against a *different* operator) deliberately has no arm here —
        // falls through to the ordinary `Mismatch` below, exactly the
        // "don't guess" behavior wanted for something like `N+M` against a
        // literal array length alone (`N`/`M` individually still
        // undetermined — see `Ty::ConstExpr`'s own doc comment).
        (Ty::ConstExpr(op1, x1, y1), Ty::ConstExpr(op2, x2, y2)) if op1 == op2 => {
            unify(subst, x1, x2)?;
            unify(subst, y1, y2)
        }
        _ => Err(UnifyError::Mismatch(a, b)),
    }
}

/// A located inference failure — the unified error shape for this pass, so
/// every failure carries a `Span` and can be turned into a `Diagnostic`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeError {
    pub span: Span,
    pub kind: TypeErrorKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeErrorKind {
    Unify(UnifyError),
    UnknownName(String),
    ArityMismatch { name: String, expected: usize, found: usize },
    NotCallable(Ty),
    /// More than one declared algebra owns this name+arity — not a
    /// "someone's overriding an existing algebra" signal (root-ownership
    /// scoping, a separate check, already rules that out); an ordinary name
    /// collision between two independently-legitimate algebras, resolved
    /// like Rust's ambiguous trait methods: reject, require explicit
    /// qualification (not implemented yet — there's no qualified-call syntax
    /// wired to bypass this today, so this is currently a hard stop).
    AmbiguousOperator { name: String, candidates: Vec<String> },
    /// The one candidate algebra exists and its signature matched, but no
    /// `impl <algebra><ty>` was ever declared.
    MissingImpl { algebra: String, ty: String },
    /// A placeholder (`<unresolved-call:...>`, ...) survived all the way to
    /// this function's own exposed return or parameter type — see
    /// `infer_fn`'s check. Carries the placeholder string itself for the
    /// message (e.g. `<unresolved-call:add>`).
    Unresolved(String),
    /// An `impl <algebra><target>` provides a method the algebra never
    /// declared a signature for at all — see `infer_impl_fn`.
    NotDeclaredByAlgebra { algebra: String, name: String },
    /// A struct literal (`Vec2(x: 1.0, y: 2.0)`) whose path doesn't name any
    /// declared `struct`.
    UnknownStruct(String),
    /// A struct literal supplied a field name the struct never declared, or
    /// a field-access expression (`a.foo`) named a field that `a`'s own
    /// (concrete, known) struct type doesn't have.
    NoSuchField { struct_name: String, field: String },
    /// A struct literal never supplied a value for one of the struct's own
    /// declared fields — every field is required, there's no partial
    /// construction/defaulting.
    MissingField { struct_name: String, field: String },
    /// A struct literal supplied the same field name more than once.
    DuplicateField { struct_name: String, field: String },
    /// A method-call expression (`v.foo(...)`) whose base is a known,
    /// concrete type with no inherent method by that name — either no
    /// `impl` on that struct declares it, or the base isn't a struct at all
    /// (a function value, array, or const — none of which have methods).
    NoSuchMethod { struct_name: String, method: String },
    /// Two `impl`s of the same algebra declare their own generic target
    /// patterns in a way that could both match a common instantiation
    /// (`impl<T: Float> Ring<Complex<T>>` and `impl<T: Ord> Ring<Complex<T>>`
    /// — some `Complex<X>` satisfying both bounds would have no principled
    /// way to pick one) — see `Infer::check_no_overlapping_impls`.
    OverlappingImpls { algebra: String, a: String, b: String },
    /// A type annotation named a declared `algebra`, not a type (`const R:
    /// Int` — `Int` is what *governs* legal types there, `i32`/`i64` are the
    /// actual types) — see `Infer::pending_type_name_checks`.
    TypeNameIsAnAlgebra { name: String },
    /// A generic algebra-impl method's call site whose own concrete
    /// argument/return types don't unify against *any* candidate impl's own
    /// declaration-time pattern — found only by `monomorphize.rs`, never by
    /// ordinary dispatch (`Infer::dispatch_algebra_call`), since dispatch
    /// only ever needs the impl's own *target* pattern to match, never the
    /// method's full parameter/return shape armed with whatever an impl's
    /// own (possibly unsound — e.g. a stub body silently merging two
    /// generics that should stay independent) declaration-time inference
    /// happened to produce. See `monomorphize.rs`'s own doc comment for a
    /// concrete example (a stub matmul body accidentally requiring a square
    /// shape).
    MonomorphizationFailed { algebra: String, method: String, tys: String },
    /// A duck-typed fallback specialization (`monomorphize.rs`'s own
    /// `detect_duck_typed_fns` -- a generic top-level fn whose body
    /// couldn't be fully resolved by the ordinary one-shot HM pass, e.g.
    /// field access on an unannotated parameter) genuinely failed to
    /// type-check for one specific concrete call site. Unlike
    /// `MonomorphizationFailed` (a blind "no impl candidate unified", no
    /// inner detail available), this carries the real `TypeError` a full
    /// re-inference produced, span and all, pointing at the actual
    /// offending expression inside the fn's own body.
    GenericFnInstantiationFailed { name: String, tys: String, inner: Box<TypeError> },
    /// A top-level `fn` or inherent-impl method declared with no body
    /// (`fn foo(x: i32);`) — legal grammatically anywhere a `fn` appears
    /// (see `grammar.pest`'s own `fn_decl` comment), but only ever actually
    /// meaningful for an algebra-impl method (see `MissingIntrinsicAttribute`
    /// below) or a top-level `fn` marked `extern` (see `ExternFnCannotBe
    /// Generic` below) — there's nothing else a top-level `fn`/inherent
    /// method could possibly mean by omitting its body otherwise.
    MissingFnBody { name: String },
    /// An algebra-impl method declared with no body and not `extern` — the
    /// one case a bodyless `fn` *is* legal, but only for a real external C
    /// symbol; an intrinsic operation gets a real body (a reserved
    /// `mlir::...` call) instead, see `mlir_lower.rs`'s own module doc
    /// comment.
    MissingIntrinsicAttribute { name: String },
    /// `extern fn foo<T>(x: T) -> T;` — a real C-ABI boundary only ever
    /// crosses concrete, monomorphized signatures (see `doc/hld.md`'s own
    /// note on this), so an `extern fn` can't be generic the way an
    /// ordinary top-level `fn` can; there's no monomorphization pass for it
    /// to go through in the first place.
    ExternFnCannotBeGeneric { name: String },
    /// `x = v` (or `arr[i] = v`/`s.x = v`, resolved down to `x`'s own root
    /// binding — see `check_mutability`) where `x` was declared with a
    /// plain `let`, never `let mut` — a purely syntactic check, no type
    /// information involved at all.
    AssignToImmutable { name: String },
    /// A scheme's own quantified variable carries two (or more) single-
    /// target shape constraints whose candidate concrete types (`Registry::
    /// candidates_for`) share no common type at all — e.g. `Int t` and
    /// `Float t` on the same `t`, provably never satisfiable by *any*
    /// concrete type, caught at `generalize`'s own time rather than only
    /// once (if ever) something later instantiates the scheme — see
    /// `Infer::generalize`'s own doc comment.
    UnsatisfiableScheme { algebras: Vec<String> },
    /// A committing dispatch (`dispatch_algebra_call`) found more than one
    /// impl whose target pattern matches the call's own query tuple, and
    /// they don't all agree on how they resolve whatever position(s) were
    /// still an unresolved variable in that query — an algebra generic that
    /// only ever appears in a *return* type (`To` in `Convert<From, To>`)
    /// is never gated the way a parameter-appearing one is (see
    /// `infer_algebra_call`'s own `gating` comment), so nothing forces it
    /// concrete before dispatch commits. `check_no_overlapping_impls`
    /// doesn't catch this ahead of time either — it checks two
    /// *declarations'* own patterns against each other, and fully concrete,
    /// differently-shaped targets (`Convert<i32, f64>` vs. `Convert<i32,
    /// Complex<f64>>`) never unify against one another, so they're never
    /// flagged as overlapping. See `dispatch_algebra_call`'s own doc
    /// comment for the fix and `match_impl`'s for why the two are split.
    AmbiguousDispatch { algebra: String, candidates: Vec<String> },
    /// A qualified call (`Ring::mul(a, b)`) named a real, declared algebra,
    /// but that algebra doesn't declare a method by this name (or not at
    /// this arity) — `infer_call`'s own qualified-call path, checked
    /// *before* `infer_algebra_call` (which assumes its own caller already
    /// confirmed this, see its own `unreachable!` on this exact
    /// precondition).
    UnknownAlgebraMethod { algebra: String, method: String },
    /// `break;`/`break value;` with no enclosing loop at all (`Infer::
    /// loop_stack` empty at the statement) — includes a `break` lexically
    /// inside a loop but *through* an intervening lambda body (`ExprKind::
    /// Lambda` temporarily empties the stack while checking one) — a break
    /// must not escape a closure boundary, mirroring Rust's identical rule.
    BreakOutsideLoop,
    /// A const-generic division (`[T; N/M]`, or an explicit turbofish const
    /// arg) whose divisor is already, concretely, zero — see `Infer::
    /// pending_div_by_zero_checks`'s own doc comment for why this is
    /// deferred rather than an immediate error at `const_value_from_expr`'s
    /// own call site.
    ConstDivByZero { dividend: u64 },
    /// Constructing a struct whose own last declared generic is a *pack*
    /// (`doc/backlog.md`'s own "Variadic generics" item, `Tensor<T, const
    /// Dims: i32...>`) without an explicit turbofish supplying at least as
    /// many arguments as there are non-pack generics — a genuine, deliberate
    /// v1 restriction: inferring a pack's own arity purely from field values
    /// (the way an ordinary, single-slot generic already can) isn't
    /// supported yet, only turbofish-driven resolution is.
    VariadicStructNeedsTurbofish { struct_name: String, min_generics: usize },
}

impl std::fmt::Display for TypeErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TypeErrorKind::Unify(e) => write!(f, "{e}"),
            TypeErrorKind::UnknownName(name) => write!(f, "unknown identifier `{name}`"),
            TypeErrorKind::ArityMismatch { name, expected, found } => {
                write!(f, "`{name}` expects {expected} argument(s), found {found}")
            }
            TypeErrorKind::NotCallable(ty) => write!(f, "`{ty}` is not callable"),
            TypeErrorKind::AmbiguousOperator { name, candidates } => {
                write!(f, "`{name}` is ambiguous between: {}", candidates.join(", "))
            }
            TypeErrorKind::MissingImpl { algebra, ty } => {
                write!(f, "no `impl {algebra}<{ty}>`")?;
                if ty == "()" {
                    write!(f, " {UNIT_DISCARD_HINT}")?;
                }
                Ok(())
            }
            TypeErrorKind::Unresolved(placeholder) => {
                write!(f, "type could not be fully determined ({placeholder})")
            }
            TypeErrorKind::NotDeclaredByAlgebra { algebra, name } => {
                write!(f, "`{name}` is not declared by `algebra {algebra}`")
            }
            TypeErrorKind::UnknownStruct(name) => write!(f, "unknown struct `{name}`"),
            TypeErrorKind::NoSuchField { struct_name, field } => {
                write!(f, "no field `{field}` on `{struct_name}`")
            }
            TypeErrorKind::MissingField { struct_name, field } => {
                write!(f, "`struct {struct_name}` construction is missing field `{field}`")
            }
            TypeErrorKind::NoSuchMethod { struct_name, method } => {
                write!(f, "no method `{method}` on `{struct_name}`")
            }
            TypeErrorKind::DuplicateField { struct_name, field } => {
                write!(f, "field `{field}` given more than once constructing `struct {struct_name}`")
            }
            TypeErrorKind::OverlappingImpls { algebra, a, b } => {
                write!(
                    f,
                    "overlapping impls: `impl {algebra}<{a}>` and `impl {algebra}<{b}>` could both match the same type"
                )
            }
            TypeErrorKind::TypeNameIsAnAlgebra { name } => {
                write!(f, "`{name}` is an algebra, not a type — did you mean a concrete type it governs?")
            }
            TypeErrorKind::MonomorphizationFailed { algebra, method, tys } => {
                write!(f, "`{algebra}::{method}` cannot be specialized for ({tys}): its generic impl body doesn't type-check at this instantiation")
            }
            TypeErrorKind::GenericFnInstantiationFailed { name, tys, inner } => {
                write!(f, "`{name}` cannot be specialized for ({tys}): {}", inner.kind)
            }
            TypeErrorKind::MissingFnBody { name } => {
                write!(f, "`{name}` has no body — a `fn` may only omit one if it's `extern`")
            }
            TypeErrorKind::MissingIntrinsicAttribute { name } => {
                write!(f, "`{name}` has no body and isn't `extern` — an algebra-impl method without a body must be")
            }
            TypeErrorKind::ExternFnCannotBeGeneric { name } => {
                write!(f, "`extern fn {name}` cannot be generic — only concrete, monomorphized signatures can cross a C-ABI boundary")
            }
            TypeErrorKind::AssignToImmutable { name } => {
                write!(f, "cannot assign to `{name}` — declared with `let`, not `let mut`")
            }
            TypeErrorKind::UnsatisfiableScheme { algebras } => {
                write!(f, "no single type can ever satisfy all of: {} — this generic can never be called", algebras.join(", "))
            }
            TypeErrorKind::AmbiguousDispatch { algebra, candidates } => {
                write!(
                    f,
                    "ambiguous dispatch for `algebra {algebra}`: could resolve to any of {} — pin the remaining generic(s) explicitly (e.g. `name::<...>(...)`)",
                    candidates.join(", ")
                )
            }
            TypeErrorKind::ConstDivByZero { dividend } => {
                write!(f, "division by zero in a const-generic expression: `{dividend} / 0`")
            }
            TypeErrorKind::UnknownAlgebraMethod { algebra, method } => {
                write!(f, "`algebra {algebra}` has no method `{method}` at this arity")
            }
            TypeErrorKind::BreakOutsideLoop => write!(f, "`break` outside a loop"),
            TypeErrorKind::VariadicStructNeedsTurbofish { struct_name, min_generics } => {
                write!(
                    f,
                    "`{struct_name}` has a variadic generic — constructing it needs an explicit turbofish with at least {min_generics} argument(s) (e.g. `{struct_name}::<...>(...)`); inferring a pack's own arity from field values isn't supported yet"
                )
            }
        }
    }
}

impl From<&TypeError> for crate::diag::Diagnostic {
    fn from(err: &TypeError) -> Self {
        crate::diag::Diagnostic::error(err.kind.to_string(), err.span)
    }
}

// ---------------------------------------------------------------- inference

pub type Env = HashMap<String, Scheme>;

/// "`tys` together must satisfy `algebra`" — generated wherever a type (or,
/// for a multi-generic algebra dispatched with some argument still abstract,
/// the *whole tuple* of that algebra's own generics — see
/// `infer_algebra_call`'s deferred branches) is used in a way that requires
/// some algebra (an arithmetic operator call, a numeric literal's implicit
/// `Num` requirement), then either checked immediately (if every one of
/// `tys` is already concrete) or carried along until it can be — including
/// into an enclosing `let`'s [`Scheme`], via [`Infer::generalize`], which is
/// what makes `fn add(a, b) { a + b }` able to infer its own `T: Ring` bound
/// from nothing but usage. Almost always a single-element `Vec` (an ordinary
/// bound/shape check); more than one element only for a multi-generic
/// algebra's own deferred dispatch, checked together via `has_matching_impl`
/// exactly like `match_impl`'s own immediate-dispatch path already does —
/// checking each element independently could never verify a combined impl
/// like `MatMul<f32,f32,f32>` exists (found by direct testing:
/// `examples/matmul.cleave`'s own scalar multiply, deferred because its
/// enclosing generic `T` is still abstract at declaration time, wrongly
/// rejected as `no impl MatMul<f32>` under the old per-generic scheme).
/// `span` is where the constraint *originated* (kept through renaming at
/// `instantiate` time), so a violation caught later still points somewhere
/// meaningful.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Constraint {
    pub algebra: String,
    pub tys: Vec<Ty>,
    /// Positions in `tys` that must already be concrete before dispatch is
    /// even *attempted* — mirrors `infer_algebra_call`'s own `gating`
    /// computation (an algebra generic that appears in at least one
    /// parameter type). A position *not* listed here is output-only (an
    /// algebra generic appearing only in the return type, `C` in `fn
    /// mul(a:A,b:B)->C;`) — never independently pinned by anything else, so
    /// `check_pending_constraints` must not wait for it to already be
    /// concrete the way it waits for a gating position; it's the committing
    /// dispatch itself that's supposed to bind it. Ordinary single-target
    /// bound checks (`Int`/`Float`/`Num`/a declared `T: Bound`) are always
    /// fully gating by construction — see `all_gating`.
    pub gating_indices: Vec<usize>,
    pub span: Span,
}

impl Constraint {
    /// Every position in `tys` is gating — the ordinary case for a single-
    /// target bound/shape check (`Int`/`Float`/`Num`, a declared `T: Bound`),
    /// which has no output-only position to speak of at all. Distinct from
    /// a real multi-generic algebra-call constraint (`infer_algebra_call`'s
    /// own deferred pushes), which computes its own, real `gating_indices`
    /// directly instead of using this constructor.
    fn all_gating(algebra: String, tys: Vec<Ty>, span: Span) -> Self {
        let gating_indices = (0..tys.len()).collect();
        Constraint { algebra, tys, gating_indices, span }
    }
}

/// `∀vars. constraints ⇒ ty` — a qualified type scheme (Mark Jones' "Theory
/// of Qualified Types", the HM extension this project has been aiming at
/// since the type-inference discussion began). A plain monomorphic binding
/// is just `Scheme { vars: vec![], constraints: vec![], ty }`, the trivial
/// case; no separate representation needed for "not generalized" or "not
/// constrained".
#[derive(Debug, Clone)]
pub struct Scheme {
    pub vars: Vec<TyVar>,
    pub constraints: Vec<Constraint>,
    pub ty: Ty,
    /// For whichever of `vars` are const generics (`const N: i32`), their
    /// own declared type — has to travel through `generalize`/`instantiate`
    /// exactly like `constraints` does (see those functions' own doc
    /// comments): a bound carried into this scheme can end up checked
    /// against one of these vars once resolved to a bare `Ty::Const`, and
    /// that check needs the *real* declared width, not a guess (see
    /// `check_pending_constraints`'s own `Ty::Const` bridge).
    pub const_widths: HashMap<TyVar, Ty>,
}

impl Scheme {
    pub(crate) fn mono(ty: Ty) -> Self {
        Scheme { vars: Vec::new(), constraints: Vec::new(), ty, const_widths: HashMap::new() }
    }
}

/// Shared by `Subst::apply` and `substitute` — the only two places a
/// `Ty::ConstExpr` ever needs folding. `a`/`b` are already resolved as far
/// as the caller's own machinery (subst-chasing or generic-instantiation
/// substitution) can take them; if both landed on a concrete `Ty::Const`,
/// delegates to `const_eval::eval_binop` (already designed for exactly this
/// reuse — see its own module doc comment) and returns the folded result.
/// Otherwise rebuilds a `ConstExpr` with the partially-resolved operands —
/// still symbolic, exactly like an unresolved `Ty::Var` would stay.
fn fold_const_expr(op: &str, a: Ty, b: Ty) -> Ty {
    if let (Ty::Const(av), Ty::Const(bv)) = (&a, &b) {
        if let Some(result) = const_eval::eval_binop(op, *av, *bv) {
            return Ty::Const(result);
        }
    }
    Ty::ConstExpr(op.to_string(), Box::new(a), Box::new(b))
}

pub(crate) fn free_vars(ty: &Ty, out: &mut HashSet<TyVar>) {
    match ty {
        Ty::Var(v) => {
            out.insert(*v);
        }
        // Quantified exactly like an ordinary `Var` — this is the whole
        // reason `Ty::Pack` is embedded as an ordinary `App` argument
        // rather than a separate field on `Scheme`: `generalize`'s own
        // free-var scan (which calls this) picks a pack var up for free the
        // moment it appears anywhere inside a declaration's own pattern, no
        // separate pack-tracking mechanism needed on `Scheme` itself.
        Ty::Pack(v) => {
            out.insert(*v);
        }
        Ty::PackResolved(elems) => {
            for e in elems {
                free_vars(e, out);
            }
        }
        Ty::PackLen(v) => {
            out.insert(*v);
        }
        Ty::Con(_) | Ty::Const(_) => {}
        Ty::App(_, args) => {
            for a in args {
                free_vars(a, out);
            }
        }
        Ty::Fn(params, ret) => {
            for p in params {
                free_vars(p, out);
            }
            free_vars(ret, out);
        }
        Ty::Array(elem, size) => {
            free_vars(elem, out);
            free_vars(size, out);
        }
        Ty::ConstExpr(_, a, b) => {
            free_vars(a, out);
            free_vars(b, out);
        }
    }
}

/// The `monomorphize.rs` counterpart to `Subst::apply` — a standalone
/// function rather than a `Subst` method since its callers hold a bare
/// `HashMap<TyVar, Ty>` (a template's own concrete instantiation values),
/// not a full `Subst`. A pack var's own resolved binding lives in `mapping`
/// the exact same way an ordinary var's does (`Ty::PackResolved(elems)`,
/// not a separate list-valued table — see `Ty::Pack`'s own doc comment for
/// why), so this lookup needs no special-casing to find it; only the
/// *consumer* (`App`'s own arm, just below) needs to know to splice a
/// `PackResolved` result instead of nesting it.
pub(crate) fn substitute(ty: &Ty, mapping: &HashMap<TyVar, Ty>) -> Ty {
    match ty {
        Ty::Var(v) | Ty::Pack(v) => mapping.get(v).cloned().unwrap_or_else(|| ty.clone()),
        // Only ever meaningful spliced into an enclosing `App`'s own args
        // list (the arm just below) — reachable bare here only via this
        // function's own direct recursion from that arm.
        Ty::PackResolved(elems) => Ty::PackResolved(elems.iter().map(|e| substitute(e, mapping)).collect()),
        // Mirrors `Subst::apply`'s own `Ty::PackLen` arm — folds to a real
        // `Const` the moment `mapping` has the underlying pack var's own
        // resolved binding, stays symbolic otherwise.
        Ty::PackLen(v) => match mapping.get(v).cloned().map(|t| substitute(&t, mapping)) {
            Some(Ty::PackResolved(elems)) => Ty::Const(ConstValue::Int(elems.len() as u64)),
            _ => ty.clone(),
        },
        Ty::Con(_) | Ty::Const(_) => ty.clone(),
        Ty::App(name, args) => {
            let mut out = Vec::with_capacity(args.len());
            for a in args {
                match substitute(a, mapping) {
                    Ty::PackResolved(elems) => out.extend(elems),
                    other => out.push(other),
                }
            }
            Ty::App(name.clone(), out)
        }
        Ty::Fn(params, ret) => {
            Ty::Fn(params.iter().map(|p| substitute(p, mapping)).collect(), Box::new(substitute(ret, mapping)))
        }
        // `size` resolving to a whole `PackResolved` list (`[value; Dims...]`
        // — `ExprKind::ArrayRepeat`'s own inference arm builds exactly this
        // shape, `Ty::Array(elem, Ty::Pack(v))`, mirroring how a struct
        // field's own `[T; Dims...]` declared type already does) means this
        // one syntactic level actually stands for *however many* real
        // nesting levels the pack turns out to have — expanded here into the
        // real nested `Ty::Array` chain, the exact same `rev().fold` shape
        // `mlir_lower.rs::resolve_struct_field_ty_with_pack` already uses
        // for the identical surface shape in a struct field's own type.
        // Every other case (`size` resolving to an ordinary `Const`/`Var`,
        // the overwhelmingly common one) is unchanged, one level, byte-for-
        // byte what this arm already did before.
        Ty::Array(elem, size) => {
            let elem = substitute(elem, mapping);
            match substitute(size, mapping) {
                Ty::PackResolved(dims) => dims.into_iter().rev().fold(elem, |acc, dim| Ty::Array(Box::new(acc), Box::new(dim))),
                size => Ty::Array(Box::new(elem), Box::new(size)),
            }
        }
        // The monomorphization-time fold: `mapping` carries a generic
        // template's own real, concrete instantiation values (see this
        // function's own callers in `monomorphize.rs`), so this is where
        // `N+M` actually turns into a real number once a template gets
        // specialized for a concrete call site — same `fold_const_expr`
        // helper `Subst::apply` uses, see its own doc comment.
        Ty::ConstExpr(op, a, b) => fold_const_expr(op, substitute(a, mapping), substitute(b, mapping)),
    }
}

/// A `let` (never `let mut` — see module docs) is only generalizable when
/// its right-hand side is a syntactic *value*: nothing that could be an
/// aliased, later-mutated reference. Deliberately conservative — an ordinary
/// function call could in principle return something safely generalizable
/// too, but distinguishing that from one that can't requires effect
/// tracking this pass doesn't have.
///
/// **Deliberately excludes `NumberLit`/`ImaginaryLit`** — a real bug, found
/// by direct testing (`examples/test_loup.cleave`): `pending_defaults` (see
/// `apply_defaults`) registers a defaulting preference exactly once, for a
/// literal's own *original* type variable, at the point the literal itself
/// is inferred. Generalizing a bare-literal `let` binding (`let a = 16;`)
/// means every later reference to `a` instantiates its own *fresh*,
/// independent variable (`instantiate`) — never registered in `pending_
/// defaults`, since nothing re-runs literal inference at a use site — while
/// the *original* variable gets skipped by `apply_defaults` precisely
/// because it's quantified. Neither ever gets defaulted. If nothing else
/// later pins that fresh use-site variable to something concrete (an
/// explicit annotation, an operation with an already-concrete signature),
/// inference silently "succeeds" — no error anywhere — leaving a real,
/// unresolved `Ty::Var` that only surfaces as a Rust panic much later, deep
/// in CPS conversion. `BoolLit`/`Path`/`Lambda` don't have this gap:
/// `BoolLit`'s own type is already fully concrete (nothing to generalize
/// over), and `Path`/`Lambda` generalization doesn't depend on `pending_
/// defaults` at all (a lambda's own body gets fully, separately
/// re-specialized per instantiation via `monomorphize.rs`'s own lambda
/// worklist). A bare-literal `let` now binds monomorphically instead — same
/// as `let mut` already did — so every later use shares the exact same
/// variable `pending_defaults` already tracks.
fn is_syntactic_value(expr: &Expr) -> bool {
    matches!(expr.kind, ExprKind::BoolLit(_) | ExprKind::Path(_) | ExprKind::Lambda { .. })
}

/// A `Ty::Con` standing in for "not actually known" — `<unresolved-call:...>`
/// (cross-function calls, see module docs), `<not-yet-inferred>` (method
/// calls, indexing, array literals, and field access specifically when the
/// base's own type is still abstract — a *concrete* base resolves a field
/// access for real, or rejects it, see `ExprKind::FieldAccess`),
/// `<loop-not-yet-inferred>`,
/// `<complex-not-yet-inferred>`, `<array-type-not-yet-inferred>`. These must
/// never be checked against the registry as if they were real concrete
/// types — found via a real CLI test run: `let a = some_undeclared_fn(x);
/// acc + a` produced `no impl Ring<<unresolved-call:some_undeclared_fn>>`,
/// treating the placeholder's own marker string as though it were a genuine
/// type name. "We don't know" must stay permissive, not become a false
/// rejection dressed up as a real one.
fn is_placeholder(ty: &Ty) -> bool {
    matches!(ty, Ty::Con(name) if name.starts_with('<'))
}

/// No free type variable anywhere in `ty`, recursively — `Ty::Con` (always
/// trivially true) generalized to `Ty::App`: `Complex<f64>` is fully
/// concrete, `Complex<'t9>` isn't (yet), same "defer as a `Constraint`
/// rather than guess" treatment that a bare abstract variable already got
/// before `App` existed. Membership-checking (`has_matching_impl`) only
/// makes sense once this holds — matching a *still-open* query against a
/// generic impl's own pattern would either wrongly commit `self.subst` to
/// an arbitrary candidate or (via the throwaway `trial` clone) answer a
/// question that isn't actually being asked yet.
fn is_fully_concrete(ty: &Ty) -> bool {
    match ty {
        Ty::Var(_) => false,
        // A still-unresolved pack is exactly as "not concrete yet" as an
        // ordinary open `Var` — a bound one never reaches here as `Pack`
        // at all (`Subst::apply`/`substitute` already splice a resolved
        // pack's own elements into the enclosing `App` before this ever
        // sees it).
        Ty::Pack(_) => false,
        // Concrete iff every one of its own elements is — mirrors `App`'s
        // own identical arm just below.
        Ty::PackResolved(elems) => elems.iter().all(is_fully_concrete),
        // Same "not concrete yet" treatment as `Ty::Pack` — a resolved one
        // already folded to `Ty::Const` before reaching here.
        Ty::PackLen(_) => false,
        Ty::Con(_) | Ty::Const(_) => true,
        Ty::App(_, args) => args.iter().all(is_fully_concrete),
        Ty::Fn(params, ret) => params.iter().all(is_fully_concrete) && is_fully_concrete(ret),
        Ty::Array(elem, size) => is_fully_concrete(elem) && is_fully_concrete(size),
        // Not concrete unless *both* operands are — a `ConstExpr` reaching
        // this point at all already means `Subst::apply`/`substitute`
        // couldn't fold it (see `fold_const_expr`), so it's never
        // spuriously "concrete" by construction.
        Ty::ConstExpr(_, a, b) => is_fully_concrete(a) && is_fully_concrete(b),
    }
}

/// Like `is_placeholder`, but recurses into `Ty::Fn` — a function type whose
/// parameter or return type is itself a placeholder is just as unresolved as
/// a bare one, and this is exactly the shape a lambda calling an undeclared
/// cross-function `fn` produces (`(t) -> <unresolved-call:add>`).
pub(crate) fn find_placeholder_name(ty: &Ty) -> Option<String> {
    match ty {
        Ty::Con(name) if name.starts_with('<') => Some(name.clone()),
        Ty::Con(_) | Ty::Var(_) | Ty::Const(_) | Ty::Pack(_) | Ty::PackLen(_) => None,
        Ty::PackResolved(elems) => elems.iter().find_map(find_placeholder_name),
        Ty::App(_, args) => args.iter().find_map(find_placeholder_name),
        Ty::Fn(params, ret) => params.iter().find_map(find_placeholder_name).or_else(|| find_placeholder_name(ret)),
        Ty::Array(elem, size) => find_placeholder_name(elem).or_else(|| find_placeholder_name(size)),
        Ty::ConstExpr(_, a, b) => find_placeholder_name(a).or_else(|| find_placeholder_name(b)),
    }
}

/// Checks a function's *fully-resolved* (post-defaulting) return and
/// parameter types for a surviving placeholder — factored out of `finish_fn`
/// so the whole-program orchestrator (`callgraph.rs`) can run the exact same
/// check once per group member, after resolving through the group's final
/// substitution, instead of duplicating this logic.
pub(crate) fn check_no_placeholder(f: &FnDecl, final_result: &Ty, param_types: &[Ty]) -> Result<(), TypeError> {
    let unresolved = find_placeholder_name(final_result).or_else(|| param_types.iter().find_map(find_placeholder_name));
    if let Some(placeholder) = unresolved {
        if let Some(span) = f
            .body
            .as_ref()
            .and_then(|b| b.tail.as_deref().map(|t| t.span).or_else(|| b.stmts.last().map(|s| s.span)))
        {
            return Err(TypeError { span, kind: TypeErrorKind::Unresolved(placeholder) });
        }
    }
    Ok(())
}

/// Static, purely syntactic check (no type information needed at all) that
/// a plain (non-`mut`) `let` binding is never reassigned — `mutable` was
/// otherwise only ever consulted to decide whether to generalize
/// (`is_syntactic_value`'s own call site, `infer_block`'s `StmtKind::Let`
/// arm), nowhere else. Applies uniformly to a bare-name target (`x = v`)
/// and an indexed/field chain rooted at one (`arr[i] = v`, `s.x = v`,
/// `s.arr[i].y = v`) via `assign_target_root` — mutating *through* a stable
/// reference is still mutating the reference's own binding (matching
/// `cps.rs`'s own "a struct/array is a stable reference" design). A
/// function's own parameters are always immutable — `grammar.pest`'s own
/// `param` rule has no `mut` at all, no way to opt in — seeded as such here
/// since they never go through `StmtKind::Let`.
pub fn check_mutability(f: &FnDecl) -> Result<(), TypeError> {
    let Some(body) = &f.body else { return Ok(()) };
    let scope: HashMap<String, bool> = f.params.iter().map(|p| (p.name.clone(), p.mutable)).collect();
    check_mutability_block(body, &scope)
}

fn check_mutability_block(block: &Block, scope: &HashMap<String, bool>) -> Result<(), TypeError> {
    let mut scope = scope.clone();
    for stmt in &block.stmts {
        match &stmt.kind {
            StmtKind::Let { mutable, name, value, .. } => {
                check_mutability_expr(value, &scope)?;
                scope.insert(name.clone(), *mutable);
            }
            StmtKind::Assign { target, value } => {
                check_mutability_expr(target, &scope)?;
                check_mutability_expr(value, &scope)?;
                if let Some(root) = assign_target_root(target) {
                    if scope.get(root) == Some(&false) {
                        return Err(TypeError { span: stmt.span, kind: TypeErrorKind::AssignToImmutable { name: root.to_string() } });
                    }
                }
            }
            StmtKind::Expr(e) => check_mutability_expr(e, &scope)?,
            StmtKind::Break(value) => {
                if let Some(v) = value {
                    check_mutability_expr(v, &scope)?;
                }
            }
        }
    }
    if let Some(tail) = &block.tail {
        check_mutability_expr(tail, &scope)?;
    }
    Ok(())
}

fn check_mutability_expr(expr: &Expr, scope: &HashMap<String, bool>) -> Result<(), TypeError> {
    match &expr.kind {
        ExprKind::NumberLit { .. } | ExprKind::ImaginaryLit { .. } | ExprKind::BoolLit(_) | ExprKind::Path(_) | ExprKind::PackRef(_) => Ok(()),
        ExprKind::Call(_, _, args, ..) => args.iter().try_for_each(|a| check_mutability_expr(a, scope)),
        ExprKind::FieldAccess(base, _) => check_mutability_expr(base, scope),
        ExprKind::MethodCall(base, _, args) => {
            check_mutability_expr(base, scope)?;
            args.iter().try_for_each(|a| check_mutability_expr(a, scope))
        }
        ExprKind::Index(base, indices) => {
            check_mutability_expr(base, scope)?;
            indices.iter().try_for_each(|i| check_mutability_expr(i, scope))
        }
        ExprKind::ArrayLit(elems) => elems.iter().try_for_each(|e| check_mutability_expr(e, scope)),
        ExprKind::ArrayRepeat { value, count } => {
            check_mutability_expr(value, scope)?;
            check_mutability_expr(count, scope)
        }
        ExprKind::StructLit(_, _, fields) => fields.iter().try_for_each(|(_, v)| check_mutability_expr(v, scope)),
        ExprKind::If { cond, then_branch, else_branch } => {
            check_mutability_expr(cond, scope)?;
            check_mutability_block(then_branch, scope)?;
            match else_branch.as_deref() {
                Some(ElseBranch::If(e)) => check_mutability_expr(e, scope),
                Some(ElseBranch::Block(b)) => check_mutability_block(b, scope),
                None => Ok(()),
            }
        }
        ExprKind::While { cond, body } => {
            check_mutability_expr(cond, scope)?;
            check_mutability_block(body, scope)
        }
        ExprKind::For { var, start, end, body } => {
            check_mutability_expr(start, scope)?;
            check_mutability_expr(end, scope)?;
            let mut inner = scope.clone();
            inner.insert(var.clone(), false);
            check_mutability_block(body, &inner)
        }
        ExprKind::ForIn { var, iter, body } => {
            check_mutability_expr(iter, scope)?;
            let mut inner = scope.clone();
            inner.insert(var.clone(), false);
            check_mutability_block(body, &inner)
        }
        ExprKind::Loop { body } => check_mutability_block(body, scope),
        ExprKind::Block(b) => check_mutability_block(b, scope),
        // A lambda's own params shadow the outer scope for the duration of
        // its own body -- still walked, and still against the *outer*
        // scope layered underneath, so reassigning a captured outer `mut`
        // variable from inside a lambda is checked exactly like anywhere
        // else (a real, ordinary violation if that outer binding isn't
        // `mut`, independent of whether the lambda itself ever gets called
        // at this point at all).
        ExprKind::Lambda { params, body, .. } => {
            let mut inner = scope.clone();
            for p in params {
                inner.insert(p.name.clone(), false);
            }
            check_mutability_block(body, &inner)
        }
    }
}

/// Walks an assignment target (`x`, or a field/index chain into one —
/// `arr[i]`, `s.x`, `s.arr[i].y`) down to its own root name — the local
/// binding whose own mutability actually governs whether the assignment is
/// legal, regardless of how many `.field`/`[index]` steps sit on top of it.
/// `None` for anything that isn't a legal assignment-target shape at all
/// (defensive — the grammar already restricts `StmtKind::Assign`'s own
/// `target` to these three shapes).
fn assign_target_root(expr: &Expr) -> Option<&str> {
    match &expr.kind {
        ExprKind::Path(p) if p.segments.len() == 1 => Some(&p.segments[0]),
        ExprKind::FieldAccess(base, _) | ExprKind::Index(base, _) => assign_target_root(base),
        _ => None,
    }
}

pub struct Infer<'r> {
    pub subst: Subst,
    vars: TyVarGen,
    /// Number-literal type variables not yet pinned to a concrete type by
    /// unification with anything — defaulted (i32, or f64 if the literal's
    /// text has a `.`/exponent) once the whole body has been inferred, the
    /// same mechanism as Haskell's numeric-literal defaulting (see
    /// `grammar.md`, "Type conversion" — literal contextual inference). A
    /// literal's own shape (`.` or not) is *also* pushed as a real
    /// `Int`/`Float` constraint (see `NumberLit` below) — that's what
    /// actually enforces it; this list is purely the "what would this
    /// default to if truly nothing else ever decides" fallback.
    pending_defaults: Vec<(TyVar, NumberDefault)>,
    /// Constraints not yet resolved one way or the other — either checked
    /// against the registry once their type becomes concrete
    /// (`check_pending_constraints`, at the end of `infer_fn`) or migrated
    /// into an enclosing `let`'s `Scheme` by `generalize` first.
    constraints: Vec<Constraint>,
    /// Shared, read-only — built once from a whole compilation's merged
    /// `Program` (see `registry.rs`), reused across every `Infer::infer_fn`
    /// call rather than rebuilt per function.
    registry: &'r Registry,
    /// Every expression's inferred type, keyed by its own `NodeId` — the
    /// side-table `ast.rs` documents this pass as using, populated by
    /// `infer_expr` as it goes and fully re-resolved through `self.subst`
    /// (see `infer_fn`) before being handed back, so a caller never sees a
    /// stale, not-yet-unified value here. Relies on `NodeId`s being unique
    /// across a whole compilation, not just one file — see `driver.rs`'s
    /// `NodeIdGen` threading.
    pub node_types: HashMap<NodeId, Ty>,
    /// Each parameter's type, in `FnDecl::params` order, resolved the same
    /// way as `node_types` — exposed separately since `Param` itself has no
    /// `NodeId` to key a side-table entry by (see `grammar.md`/`ast.rs`;
    /// not every syntactic node got the uniform `Node<T>` treatment).
    pub param_types: Vec<Ty>,
    /// Set only by `infer_impl_fn_generic_with_env` — this impl's own
    /// target(s) (`Matrix<T,N,M>`, `Matrix<T,M,K>`, `Matrix<T,N,K>` for a
    /// `MatMul` impl), resolved through the *same* fresh `impl_mapping*`
    /// `param_types`/the method's own body used, so a caller (`monomorphize.rs`)
    /// can unify these against a call site's own concrete types and, in the
    /// very same `Subst`, recover consistent bindings for every one of this
    /// impl's own generics — `param_types` alone isn't enough for that,
    /// since an algebra's own return-type-only generic (`C` in `fn mul(a:
    /// A, b: B) -> C;`) never appears there at all. Empty for every other
    /// entry point (`infer_fn`, `infer_inherent_impl_fn_generic`, ...) —
    /// there's no equivalent "target pattern" for a top-level `fn` or an
    /// inherent method to expose here.
    pub target_types: Vec<Ty>,
    /// The currently-being-inferred top-level `fn`/impl-method's own
    /// generic-parameter-name -> `Ty` mapping (`fresh_fn_shape`'s own
    /// `generics` for a plain `fn`, `impl_mapping` for an impl method) — set
    /// once, at the top of whichever of `infer_fn_raw`/`infer_impl_fn_
    /// generic_with_env` is currently running, consulted by `ty_from_ast`'s
    /// two generics-aware callers (a `let`'s own type annotation, a nested
    /// lambda's own parameter/return annotations) instead of resolving
    /// against an empty map. A real, previously-silent bug, found by direct
    /// testing while building a generic `matmul` impl (`let acc: Boxed<T> =
    /// ...;`, referencing the enclosing impl's own generic `T` from inside
    /// its method body — never exercised before this): every *signature*-
    /// level reference to an
    /// enclosing generic (`fn f<T>(x: T)`, an impl's own declared target/
    /// param/return types) already threads a real mapping through
    /// `ty_from_ast_mapped` correctly, but any annotation syntactically
    /// *nested inside the body* used the always-empty `ty_from_ast` instead
    /// — `T` silently fell through to a bogus, permanently-unmatchable
    /// `Ty::Con("T")` rather than erroring or resolving correctly, only
    /// surfacing much later as a confusing `no impl Float<T>`. Deliberately
    /// *not* threaded as an explicit parameter through `infer_expr`/
    /// `infer_block`'s entire call graph — every recursive call already
    /// shares the one enclosing `fn`/impl-method's own body, so a single
    /// field set once at the top covers the whole body without an invasive,
    /// cross-cutting signature change everywhere. Left populated (never
    /// cleared) once a generic body finishes — harmless: nothing reads it
    /// outside of another `fn`/impl-method's own body inference, which
    /// always resets it first.
    active_generics: HashMap<String, Ty>,
    /// Every type variable `generalize` has ever quantified into some
    /// binding's `Scheme` — `apply_defaults` must never bind one of these.
    /// Found necessary by testing, not by design up front: a self-recursive
    /// top-level `fn` like `fibonacci(x)` generalizes to `∀t. ... => (t) ->
    /// t`, with `x`'s own type variable *literally* being `t` — not a copy,
    /// the same `TyVar`, since `infer_fn_raw` binds a parameter via
    /// `Scheme::mono` (no fresh-variable renaming at all). Without this,
    /// `apply_defaults` would still force that same variable concrete
    /// (`i32`, from whichever bare-integer literal it happened to touch
    /// inside the function's own body) purely for `node_types`'s sake — so
    /// `--dump-inference-pass` showed `fn fibonacci(x: 't6) -> 't6 { ... if
    /// ... { x:i32 ... } ... }`, the exact same variable reported as both
    /// still-generic (correct — a later caller genuinely does instantiate it
    /// at `f32` without error) *and* concretely `i32` (wrong — nothing pins
    /// it to `i32` specifically; that was always just an arbitrary default,
    /// not a real constraint) in the same breath. Harmless for actually
    /// *using* the scheme (`instantiate` substitutes over `scheme.ty`
    /// directly and never consults `self.subst`) — the bug was purely that
    /// `self.subst` is also what `node_types`/`param_types` read through,
    /// with nothing to stop that read from disagreeing with what the
    /// `Scheme` itself promises.
    quantified: HashSet<TyVar>,
    /// One entry per currently-open loop, its own **accumulator type** — the
    /// type every `break`/`break value` inside it (directly, not through a
    /// nested loop) unifies against. `While`/`For`/`ForIn` pin this to
    /// `Ty::Con("()")` immediately on push (their own natural, non-break
    /// exit has no value to reconcile a `break value` against — mirrors
    /// Rust's identical restriction); `Loop` seeds a fresh `Ty::Var`, later
    /// read back as the whole expression's own type, exactly like a `let`'s
    /// own scheme is read back after `generalize`. A `break` outside any
    /// loop (`loop_stack.is_empty()`) is `TypeErrorKind::BreakOutsideLoop`.
    /// `ExprKind::Lambda` temporarily swaps this for an empty `Vec` while
    /// checking its own body — a `break` must not escape through a closure
    /// boundary (the same rule Rust enforces, for the same soundness
    /// reason: a lambda can be called later, outside the loop's own frame).
    loop_stack: Vec<Ty>,
    /// A bare name in *type* position that resolves to a known `algebra`
    /// (and isn't also a known `struct`) — `(name, span)`, checked once at
    /// `finish_fn` time, same deferred shape as `constraints`/
    /// `pending_defaults`. Found via direct user testing: `const R: Int`
    /// (`Int` being the algebra, not a type — `i32`/`i64` are the actual
    /// types it governs) passed silently, because nothing anywhere checks
    /// that a type annotation's name refers to an actual type at all.
    /// Pushed from `ty_from_ast_mapped` — the one universal funnel every
    /// type annotation resolves through — rather than special-cased at each
    /// call site, so a parameter/return/field/const-generic type all get
    /// the same coverage for free. Deferred rather than an immediate
    /// `Result`-returning check because `ty_from_ast_mapped` is called from
    /// dozens of sites (many inside `map`/`filter_map` closures with no
    /// natural `?`-propagation path), *and* because `has_matching_impl`'s
    /// own speculative probing calls it too, where a hard, immediate error
    /// would be wrong (a probe that doesn't pan out must stay silent, not
    /// fail the whole enclosing inference) — deferring means a probe that's
    /// abandoned just leaves an unread, harmless queue entry behind, exactly
    /// the same "never commit to a speculative outcome" posture
    /// `has_matching_impl`'s own doc comment already establishes for
    /// `self.subst`.
    pending_type_name_checks: Vec<(String, Span)>,
    /// A const-generic division whose own divisor is *already* known,
    /// concretely, to be zero (`dividend`, `span`) — `(u64, Span)`, checked
    /// once at `finish_fn` time, the exact same deferred shape as
    /// `pending_type_name_checks` right above (same reasoning: `const_
    /// value_from_expr` has no natural `?`-propagation path either, called
    /// from `ty_from_ast_mapped`'s `Array` arm and `generic_arg_to_ty`).
    /// Pushed from `const_value_from_expr` itself: `const_eval::eval_binop`
    /// only ever returns `None` for `"div"` when the divisor is zero (every
    /// other unrecognized-operator/shape case is handled before reaching
    /// `"div"` at all), so a zero-divisor is the *only* way a `div` call
    /// with two already-`Ty::Const` operands can fail there — found
    /// directly, empirically, while adding `div`/`neg` const-eval support:
    /// left unchecked, this would otherwise build a `Ty::ConstExpr("div",
    /// N, 0)` that can never resolve further (both operands are already
    /// concrete) and silently survive all the way to `mlir_lower.rs`'s own
    /// codegen-time panic for an unresolved deferred const expression — a
    /// confusing message, not a real, located compile error, for something
    /// 100% certain and detectable right here.
    pending_div_by_zero_checks: Vec<(u64, Span)>,
    /// (struct, method) -> (param types, return-type placeholder) for
    /// whichever *inherent* method is currently having its own body
    /// inferred (`infer_inherent_impl_fn_generic`) — the impl-method
    /// equivalent of a top-level `fn`'s own self-reference seeded into
    /// `env` (see `infer_fn`). Consulted by `ExprKind::MethodCall`'s own
    /// dispatch *before* falling back to the registry's static, declared
    /// signature: an inherent method with no `->` annotation has nothing
    /// else to offer a recursive call site (dispatch never re-runs a
    /// callee's own body — see that arm's own doc comment on the narrower,
    /// still-open gap this doesn't close: a call to a *different*,
    /// similarly-unannotated method that hasn't started inferring yet still
    /// falls through to the placeholder).
    ///
    /// Only ever holds at most one entry *per nesting level* in practice
    /// (inherent methods are inferred one at a time, `dump.rs`'s own loop,
    /// never concurrently) — a plain `HashMap` rather than a stack because
    /// insert/remove already brackets each method's own body inference
    /// correctly regardless of whether an outer entry happens to still be
    /// present (a method calling a *different* struct's method while that
    /// other struct's own method is *also* mid-inference — not reachable
    /// today, since nothing infers two methods' bodies inside one another,
    /// but harmless either way: each key is its own (struct, name) pair).
    in_progress_methods: HashMap<(String, String), (Vec<Ty>, Ty)>,
    /// Every `let`-bound lambda's own generalized `Scheme`, keyed by the
    /// `Lambda` expression's own `NodeId` (not the `let`'s) — `node_types`
    /// alone isn't enough for a lambda: `ExprKind::Lambda` is a syntactic
    /// value (`is_syntactic_value`), so `let f = fn(x) { x + 1 };` gets
    /// generalized exactly like a top-level generic `fn` (real Hindley-
    /// Milner let-polymorphism), and each call site re-instantiates it fresh
    /// (`infer_call`'s own `env.get(&name)` arm) — meaning the lambda body's
    /// own `node_types` entries never get pinned to any one concrete type at
    /// all, they stay unresolved `Ty::Var`s forever. A later pass
    /// (`monomorphize.rs`) needs the *scheme itself* to reverse-derive each
    /// concrete instantiation actually used, the same way it already does
    /// for a generic top-level `fn` — this is what makes that possible.
    /// Populated at the same `StmtKind::Let` site `node_types`/`env` already
    /// are, re-resolved through `self.subst` at the same points `node_types`
    /// is (see `finish_fn`/`infer_impl_fn_generic_with_env`).
    pub lambda_schemes: HashMap<NodeId, Scheme>,
    /// Every inherent method's own early-inferred return-type pattern,
    /// reached anywhere in the whole program — see `callgraph::infer_
    /// inherent_impls_early`'s own doc comment for why this exists and what
    /// it does/doesn't cover. `None` (the default, `Infer::new`) for every
    /// existing caller that doesn't opt in — an ordinary `MethodCall` still
    /// falls back to the `<not-yet-inferred>` placeholder exactly like
    /// before, no behavior change for anything that doesn't call
    /// `with_inherent_patterns`.
    inherent_patterns: Option<&'r HashMap<(String, String), crate::callgraph::InherentMethodPattern>>,
    /// A field access (`v.foo`) whose own base was still a bare `Ty::Var`
    /// at the point it was written — e.g. `let z = 4i; z.real`, where `z`'s
    /// own type only becomes concrete once `apply_defaults` runs, at the
    /// very end of the whole function's inference. Deferred the same way
    /// `pending_type_name_checks` already defers a different "not knowable
    /// yet" question — `check_pending_field_accesses` drains this, once,
    /// right after `apply_defaults`. Never populated for a base that's
    /// *already* an unresolved placeholder (`<unresolved-call:...>` and
    /// friends) — that case genuinely never resolves no matter how long
    /// this waits, so it keeps returning the placeholder immediately,
    /// unchanged.
    pending_field_accesses: Vec<PendingFieldAccess>,
    /// The `MethodCall` counterpart to `pending_field_accesses` — see its
    /// own doc comment. `arg_tys`/`arg_spans` are captured already-resolved
    /// (arguments never depend on the base's own resolution, so they're
    /// still inferred immediately, same as today) — only the base-dependent
    /// half (which method, on which struct, with what return type) is
    /// deferred.
    pending_method_calls: Vec<PendingMethodCall>,
    /// The `Index` (`base[i,j,...]`) counterpart to `pending_field_accesses`
    /// — same reasoning, same shape: `mc[0,0]` right after `let mc =
    /// matmul(ma, mb);` has `mc`'s own type as a bare `Ty::Var` at the point
    /// it's written (an algebra call's output-only generic, `C` here, is
    /// never independently concrete until `MatMul`'s own dispatch actually
    /// runs — itself deferred until `apply_defaults`/`check_pending_
    /// constraints` — see `doc/backlog.md`'s own "`check_pending_
    /// constraints`'s output-only-generic gate" item). `index_tys`/`index_
    /// spans` are captured already-resolved, mirroring `pending_method_
    /// calls`'s own identical split — an index's own type never depends on
    /// the base's, so it's inferred (and `Int`-constrained) immediately
    /// either way; only "does this end up peeling an array dimension or
    /// dispatching `Index<Container,Elem,K>`" is what's actually deferred.
    pending_indices: Vec<PendingIndex>,
}

/// See `Infer::pending_field_accesses`'s own doc comment.
struct PendingFieldAccess {
    base: Ty,
    field: String,
    result: TyVar,
    span: Span,
}

/// See `Infer::pending_method_calls`'s own doc comment.
struct PendingMethodCall {
    base: Ty,
    method: String,
    arg_tys: Vec<Ty>,
    arg_spans: Vec<Span>,
    base_span: Span,
    result: TyVar,
    call_span: Span,
}

/// See `Infer::pending_indices`'s own doc comment.
struct PendingIndex {
    base: Ty,
    base_span: Span,
    index_tys: Vec<Ty>,
    index_spans: Vec<Span>,
    result: TyVar,
    span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumberDefault {
    Int,
    Float,
    /// A bare imaginary literal (`4i`) — see `ExprKind::ImaginaryLit`'s own
    /// handling. Unlike `Int`/`Float`, its default isn't a bare `Ty::Con`
    /// name — `apply_defaults` builds `App("Complex", [Con("f32")])`
    /// directly for this variant, matching `Float`'s own default width.
    Complex,
}

impl<'r> Infer<'r> {
    pub fn new(registry: &'r Registry) -> Self {
        Infer {
            subst: Subst::default(),
            vars: TyVarGen::default(),
            pending_defaults: Vec::new(),
            constraints: Vec::new(),
            registry,
            node_types: HashMap::new(),
            param_types: Vec::new(),
            target_types: Vec::new(),
            active_generics: HashMap::new(),
            quantified: HashSet::new(),
            loop_stack: Vec::new(),
            pending_type_name_checks: Vec::new(),
            pending_div_by_zero_checks: Vec::new(),
            in_progress_methods: HashMap::new(),
            lambda_schemes: HashMap::new(),
            inherent_patterns: None,
            pending_field_accesses: Vec::new(),
            pending_method_calls: Vec::new(),
            pending_indices: Vec::new(),
        }
    }

    /// Opts this instance into consulting `patterns` (`callgraph::infer_
    /// inherent_impls_early`'s own output) when `ExprKind::MethodCall`'s own
    /// dispatch hits an unannotated inherent method — builder-style, so
    /// `callgraph::infer_program` can chain it directly onto `Infer::new`.
    pub fn with_inherent_patterns(mut self, patterns: &'r HashMap<(String, String), crate::callgraph::InherentMethodPattern>) -> Self {
        self.inherent_patterns = Some(patterns);
        self
    }

    /// Maps `f`'s own declared generic type parameters (`fn f<T: Int>(x: T)
    /// -> T`) to fresh type variables, pushing a `Constraint` for each bound
    /// — the same treatment an algebra's own generic parameter already gets
    /// when instantiated for a call (`infer_algebra_call`), just sourced
    /// from a top-level `fn`'s own declaration instead of an `algebra`'s.
    /// Until this existed, `FnDecl::generics` parsed (`fn_decl`'s grammar
    /// already accepted `<T: Bound>`) but was read *nowhere* in this file —
    /// `T` resolved as a literal, nonsensical concrete type named `"T"` via
    /// the same `ty_from_ast_mapped` fallback any unrecognized bare path
    /// hits, and any bound was silently dropped, never even parsed into a
    /// `Constraint`.
    ///
    /// `span` is used for every bound's `Constraint` — `GenericParam` itself
    /// carries no `Span` of its own (see `ast.rs`), so there's nothing more
    /// precise available; callers pass a fallback (the body's own tail/last
    /// statement span, matching the same fallback `infer_fn_raw`'s own
    /// ret-var tie-back already uses) rather than changing `infer_fn`'s
    /// widely-used public signature just to thread a more exact one through.
    fn fn_generics_mapping(&mut self, f: &FnDecl, span: Span) -> HashMap<String, Ty> {
        self.fresh_generics_mapping(&f.generics, span)
    }

    /// Maps *any* declared generic parameter list (a top-level `fn`'s own, a
    /// `struct`'s own, ...) — type *and* const generics alike — to fresh
    /// vars, pushing a `Constraint` for each type-generic's own bound
    /// (`bound_list`, only ever legal after the plain-`ident` alternative).
    /// A const-generic (`const N: T`) gets no *algebra* constraint pushed
    /// *here* — it isn't inherently integer-shaped (`const B: bool` is
    /// perfectly legitimate on its own), so nothing about its own
    /// declaration demands `Int`; whichever *use site* actually needs an
    /// integer is responsible for saying so itself (`ty_from_ast_mapped`'s
    /// `TypeKind::Array` arm does exactly that, for array sizes
    /// specifically — the one real consumer today). Found via direct user
    /// feedback on an earlier version of this method, which pushed `Int`
    /// globally right here: that conflated "this project won't default to
    /// one blessed integer width" (still true — see `ExprKind::Index`'s own
    /// unconstrained-width `Int` requirement) with "every const-generic must
    /// be an integer" (false — the two are independent questions). Its
    /// declared type (`T` in `const N: T`) *is* still resolved, via
    /// `ty_from_ast_mapped` — originally only for the side effect of queuing
    /// a `pending_type_name_checks` entry if `T` turns out to actually name
    /// an `algebra` (`const R: Int`, a real bug found by direct user
    /// testing — `Int` is what constrains a type, not a type itself) rather
    /// than a real type; also recorded now, in `self.subst`'s own const-
    /// width tracking (see `Subst::bind`'s own doc comment), since a const
    /// generic referenced as an ordinary value (a `for` loop bound, say)
    /// needs its own real declared width recoverable later, once resolved
    /// to a bare `Ty::Const` — see `check_pending_constraints`'s own
    /// `Ty::Const` bridge. The shared core `fn_generics_mapping`
    /// delegates to, also used directly by struct construction/field access
    /// (`ExprKind::StructLit`/`FieldAccess`), where there's no `FnDecl` to
    /// read a generics list off of, just `Registry::struct_generics` — a
    /// struct's own const generics get the same width tracking for free.
    fn fresh_generics_mapping(&mut self, generics: &[GenericParam], span: Span) -> HashMap<String, Ty> {
        let mapping = self.fresh_vars_for_generics(generics);
        for g in generics {
            match g {
                GenericParam::Type { name, bounds, .. } => {
                    let ty = mapping[name].clone();
                    for bound in bounds {
                        self.constraints.push(Constraint::all_gating(bound.clone(), vec![ty.clone()], span));
                    }
                }
                GenericParam::Const { name, ty, .. } => {
                    let width = self.ty_from_ast_mapped(ty, &mapping);
                    if let Ty::Var(v) = &mapping[name] {
                        self.subst.set_const_width(*v, width);
                    }
                }
            }
        }
        mapping
    }

    /// The pack-aware counterpart to `ExprKind::StructLit`'s own ordinary
    /// construction path, taken when `struct_generics`'s own last entry is
    /// a pack (`doc/backlog.md`'s own "Variadic generics" item —
    /// `Tensor<T, const Dims: i32...>`'s own motivating case). A pack can't
    /// be squeezed into the ordinary `HashMap<String, Ty>` mapping every
    /// other generic-resolution path here uses (one name resolves to
    /// *several* types, not one) — resolved as a separate `Vec<Ty>`
    /// instead, positional against the turbofish's own trailing arguments.
    ///
    /// **Turbofish-driven only, deliberately, for this first cut**: a
    /// pack's own arity has no other source to infer it from — an ordinary
    /// generic can fall back on "whatever type the field value turns out to
    /// be," but nothing about a field's own *value* tells you how many
    /// *dimensions* a pack expanded to (`[T; Dims...]`'s own field value is
    /// one already-flat, already-nested array, not obviously "3 dims" vs.
    /// "a differently-shaped 2 dims" from its type alone without deeper
    /// unification machinery this cut doesn't build). `TypeErrorKind::
    /// VariadicStructNeedsTurbofish` if the turbofish is missing or too
    /// short.
    #[allow(clippy::too_many_arguments)]
    fn infer_struct_lit_with_pack(
        &mut self,
        env: &Env,
        span: Span,
        struct_name: &str,
        struct_generics: &[GenericParam],
        explicit_generics: &[GenericArg],
        fields: &[(String, Expr)],
        declared_fields: &[Field],
    ) -> Result<Ty, TypeError> {
        let non_pack = &struct_generics[..struct_generics.len() - 1];
        let pack_generic = struct_generics.last().expect("checked non-empty by the caller");
        if explicit_generics.len() < non_pack.len() {
            return Err(TypeError {
                span,
                kind: TypeErrorKind::VariadicStructNeedsTurbofish { struct_name: struct_name.to_string(), min_generics: non_pack.len() },
            });
        }

        // Non-pack generics resolve exactly as the ordinary path does —
        // fresh vars, unified against their own turbofish slot.
        let mapping = self.fresh_generics_mapping(non_pack, span);
        for (g, explicit) in non_pack.iter().zip(explicit_generics) {
            let fresh = mapping[g.name()].clone();
            let explicit_ty = self.generic_arg_to_ty(explicit);
            self.unify_at(span, &fresh, &explicit_ty)?;
        }
        // The pack itself: every remaining turbofish argument, in order —
        // resolved directly (not through a fresh var first), since there's
        // no field-value-driven inference to reconcile against here, unlike
        // the non-pack case above.
        let pack_tys: Vec<Ty> = explicit_generics[non_pack.len()..].iter().map(|g| self.generic_arg_to_ty(g)).collect();

        let mut seen: HashSet<String> = HashSet::new();
        for (name, value) in fields {
            let Some(decl_field) = declared_fields.iter().find(|f| &f.name == name).cloned() else {
                return Err(TypeError { span: value.span, kind: TypeErrorKind::NoSuchField { struct_name: struct_name.to_string(), field: name.clone() } });
            };
            if !seen.insert(name.clone()) {
                return Err(TypeError { span: value.span, kind: TypeErrorKind::DuplicateField { struct_name: struct_name.to_string(), field: name.clone() } });
            }
            let value_ty = self.infer_expr(env, value)?;
            let declared_ty = self.ty_from_ast_mapped_with_pack(&decl_field.ty, &mapping, pack_generic.name(), &pack_tys);
            self.unify_at(value.span, &declared_ty, &value_ty)?;
        }
        if let Some(missing) = declared_fields.iter().find(|f| !seen.contains(&f.name)) {
            return Err(TypeError { span, kind: TypeErrorKind::MissingField { struct_name: struct_name.to_string(), field: missing.name.clone() } });
        }

        let mut type_args: Vec<Ty> = non_pack.iter().map(|g| mapping[g.name()].clone()).collect();
        type_args.extend(pack_tys);
        Ok(Ty::App(struct_name.to_string(), type_args))
    }

    /// Resolves a struct field's own declared type at a pack-generic
    /// construction site — identical to `ty_from_ast_mapped` for every
    /// ordinary shape (delegated to directly), plus the two positions a
    /// pack can appear in (`TypeKind::Array`'s own doc comment): a whole
    /// array-dimension list (`[T; Dims...]`, expands to nested `Ty::Array`
    /// levels, one per `pack_tys` element) or a whole field type
    /// (`Args...`, becomes the tuple formed from `pack_tys`, reusing
    /// `ast::tuple_struct_name` directly). Shallow, deliberately — only
    /// the field's own *top-level* shape is checked for a pack reference,
    /// not arbitrary nested positions (`Tensor`/a future variadic `print`
    /// alike only ever need a pack at the top level of one field/parameter,
    /// see `doc/backlog.md`'s own "Variadic generics" item for the fuller
    /// scope this deliberately doesn't attempt yet).
    fn ty_from_ast_mapped_with_pack(&mut self, ty: &Type, mapping: &HashMap<String, Ty>, pack_name: &str, pack_tys: &[Ty]) -> Ty {
        match &ty.kind {
            TypeKind::Array(elem, size) if matches!(&size.kind, ExprKind::PackRef(name) if name == pack_name) => {
                let elem_ty = self.ty_from_ast_mapped(elem, mapping);
                pack_tys.iter().rev().fold(elem_ty, |acc, dim| Ty::Array(Box::new(acc), Box::new(dim.clone())))
            }
            TypeKind::PackRef(name) if name == pack_name => {
                let name = tuple_struct_name(pack_tys.len());
                if pack_tys.is_empty() { Ty::Con(name) } else { Ty::App(name, pack_tys.to_vec()) }
            }
            _ => self.ty_from_ast_mapped(ty, mapping),
        }
    }

    /// Protects an impl's own generic parameters (`T` in `impl<T: Float>
    /// ...`) from `apply_defaults` — the same protection `generalize`
    /// already gives a `let`-bound/top-level generic fn's own free type
    /// variables (`self.quantified`). A top-level generic fn gets that
    /// protection "for free": `callgraph.rs`'s whole-program pass calls
    /// `generalize` — which populates `quantified` — *before* `apply_
    /// defaults` ever runs. An impl's own generics go through no analogous
    /// step at all, so nothing previously stopped a bare numeric literal
    /// inside the method body from getting unified with the impl's own
    /// generic (e.g. an unannotated struct field value, defaulting instead
    /// of staying generic) and silently collapsing it to a concrete
    /// `f32`/`i32` at template-build time — real bug, found by direct
    /// testing while designing `Convert<From, To>`: `impl<T: Float>
    /// Widen<T, Pair<T>> { fn widen(x) { Pair(a: x, b: 0.0) } }` dumped as
    /// `fn widen(x: f32) -> Pair<f32>`, `T` gone entirely, which then made
    /// `monomorphize.rs`'s own `derive_impl_instantiation` unable to
    /// reverse-unify a *different* concrete call site (`f64`) against this
    /// same template ever again.
    ///
    /// Must be called with `mapping` resolved through the *current*
    /// `self.subst` — i.e. right before this template's own `finish_fn`/
    /// defaulting runs, after the method body has already been inferred —
    /// not at the point `mapping` was first built (`fresh_generics_
    /// mapping`, before any of the body's own inference has run). Real bug
    /// in an earlier version of this fix, found by direct testing:
    /// registering `mapping`'s own *original* fresh vars up front missed
    /// exactly the case that matters, since `Subst::bind`'s own merge
    /// direction (`unify` binds *from* one operand *to* the other, order-
    /// dependent, not something callers control) can make the impl's own
    /// generic var become a mere *alias* of some other variable (here,
    /// `Pair`'s own field-declared generic, itself merged with `0.0`'s own
    /// literal shape-var) during body inference — resolving *now*, through
    /// `self.subst.apply`, finds whichever variable actually survived that
    /// chain, exactly mirroring `apply_defaults`'s own resolution of the
    /// literal's var (see its own doc comment).
    ///
    /// Doesn't weaken anything: `check_pending_constraints` already skips a
    /// not-fully-concrete constraint regardless of `quantified` (the well-
    /// behaved case was already deferred this way), and an impl's own
    /// declared bound (`T: Float`) is independently, authoritatively re-
    /// checked at real dispatch time anyway (`matching_impls`'s own
    /// `bounds_satisfied`, against the call site's real concrete types) —
    /// this was never the load-bearing check for an impl's own generics.
    ///
    /// Deliberately not folded into `fresh_generics_mapping` itself, whose
    /// other callers shouldn't get this treatment: an ordinary call site
    /// re-deriving an inherent method's shape against an already-*concrete*
    /// receiver pins the impl's generic via real, immediate unification,
    /// never left for `apply_defaults` to guess at; a struct literal's own
    /// generic instantiation is an ordinary use site, not a declaration,
    /// and should stay normally defaultable.
    fn quantify_impl_generics(&mut self, mapping: &HashMap<String, Ty>) {
        for ty in mapping.values() {
            if let Ty::Var(v) = self.subst.apply(ty) {
                self.quantified.insert(v);
            }
        }
    }

    /// Binds each `const N: T` generic's own name into `env` as an ordinary
    /// value (`Scheme::mono`, the same fresh `Ty::Var` `fresh_generics_mapping`
    /// already created for it) — without this, `N` only exists in the
    /// *type*-position mapping (`ty_from_ast_mapped`'s own `HashMap<String,
    /// Ty>`), so a body referencing `N` as a value (`[0.0; N]`'s own count,
    /// `ExprKind::ArrayRepeat`) would see an ordinary unbound-name error.
    /// Called at every real "about to infer a body" entry point (top-level
    /// `fn`, algebra impl method, inherent impl method) — deliberately not
    /// folded into `fresh_generics_mapping` itself, which has other callers
    /// (`has_matching_impl`'s speculative probe, struct-literal generics)
    /// with no body being inferred and no `env` to seed.
    fn seed_const_generics(&self, generics: &[GenericParam], mapping: &HashMap<String, Ty>, env: &mut Env) {
        for g in generics {
            if let GenericParam::Const { name, .. } = g {
                env.insert(name.clone(), Scheme::mono(mapping[name].clone()));
            }
        }
    }

    /// Just the fresh-variable half of `fresh_generics_mapping`, with no
    /// `Constraint` pushed for any bound — used by `has_matching_impl`'s own
    /// speculative structural probe against a generic impl's target
    /// pattern, which must never leave real, persistent constraints behind
    /// for a match that might not even pan out (bounds are checked directly
    /// there instead, against whatever the probe's *trial* substitution
    /// resolved each parameter to).
    fn fresh_vars_for_generics(&mut self, generics: &[GenericParam]) -> HashMap<String, Ty> {
        // A `variadic` generic (`doc/backlog.md`'s own "Variadic generics"
        // item, `const Dims...: i32`/`Args...`) mints a *symbolic pack*
        // (`Ty::Pack`) instead of an ordinary `Ty::Var` — this one change
        // is what makes every other site that reads a name back out of the
        // resulting mapping (`ty_from_ast_mapped`'s existing bare-name
        // lookup chief among them) pick the pack up for free, with no
        // further special-casing needed there at all.
        let fresh = |v: &mut Self, variadic: bool| {
            let Ty::Var(id) = v.vars.fresh() else { unreachable!("TyVarGen::fresh always returns Ty::Var") };
            if variadic { Ty::Pack(id) } else { Ty::Var(id) }
        };
        generics
            .iter()
            .map(|g| match g {
                GenericParam::Type { name, variadic, .. } => (name.clone(), fresh(self, *variadic)),
                // A const-generic (`const N: i32`) maps to a fresh var
                // exactly like a type-generic does -- there's no separate
                // "const" unification universe, just the same `Ty::Var`,
                // resolved to a `Ty::Const` wherever it ends up used (an
                // array's size slot, today's only consumer). This is only
                // the fresh-var half -- `fresh_generics_mapping` (the other
                // caller of this method) is what actually checks the const's
                // own declared type against `Int`; this speculative variant
                // skips it same as it skips a type-generic's own bounds, for
                // the same reason (see its own doc comment). Note this fresh
                // var stands for the const's *value*, not its declared width
                // -- a value var only ever unifies against `Ty::Const(n)` or
                // another such var, so nothing here yet distinguishes `const
                // N: i32` from `const M: i64` beyond both being `Int`-typed;
                // catching a *width* mismatch between two differently-typed
                // consts would need `Ty::Const` to carry its own type too,
                // not attempted in this increment.
                GenericParam::Const { name, variadic, .. } => (name.clone(), fresh(self, *variadic)),
            })
            .collect()
    }

    /// The inverse of the type-argument list `StructLit` builds: given
    /// `struct_name`'s own declared generics and a *concrete* (or
    /// still-abstract, doesn't matter) `Ty::App`'s own `type_args`, zips
    /// them back together into a `name -> Ty` mapping — `Complex<f64>`'s
    /// `[Con("f64")]` zipped against `struct Complex<T> { ... }`'s own `[T]`
    /// gives `{"T": Con("f64")}`, so a field declared `real: T` resolves to
    /// `f64` for *this* value specifically. Purely positional, 1:1 — every
    /// declared generic (type *or* const) consumes exactly one `type_args`
    /// slot, matching `Ty::App`'s own documented convention (no filtering:
    /// `Matrix<f64, 3, 3>`'s `N`/`M` const-generics zip against `Const`
    /// slots the exact same way a type-generic zips against a `Con`/`App`
    /// one).
    fn zip_struct_generics(&self, struct_name: &str, type_args: &[Ty]) -> HashMap<String, Ty> {
        let struct_generics = self.registry.struct_generics(struct_name);
        struct_generics
            .iter()
            .zip(type_args)
            .map(|(g, t)| {
                let name = match g {
                    GenericParam::Type { name, .. } => name,
                    GenericParam::Const { name, .. } => name,
                };
                (name.clone(), t.clone())
            })
            .collect()
    }

    /// The core of `ExprKind::FieldAccess` — given an *already concrete*
    /// (or otherwise not-worth-deferring: a function/array/const value with
    /// no fields at all) base type, finds `name`'s own declared field type.
    /// Extracted so `check_pending_field_accesses` can run the exact same
    /// logic once a deferred base finally becomes concrete, instead of
    /// duplicating it — see `pending_field_accesses`'s own doc comment for
    /// why a field access needs deferring at all.
    fn resolve_field_access(&mut self, resolved: &Ty, name: &str, span: Span) -> Result<Ty, TypeError> {
        match resolved {
            // Non-generic struct (or any other bare concrete type — see the
            // `None` arm below) — field's declared type needs no further
            // mapping, it can't mention a generic parameter this struct
            // doesn't have.
            Ty::Con(struct_name) => match self.registry.struct_fields(struct_name) {
                Some(fields) => match fields.iter().find(|f| f.name == name) {
                    Some(field) => Ok(self.ty_from_ast(&field.ty)),
                    None => Err(TypeError {
                        span,
                        kind: TypeErrorKind::NoSuchField { struct_name: struct_name.clone(), field: name.to_string() },
                    }),
                },
                // A concrete, known type that simply isn't a struct at all
                // (`(1).foo`) — genuinely has no fields, rejected the same
                // way as a struct missing this specific one.
                None => Err(TypeError {
                    span,
                    kind: TypeErrorKind::NoSuchField { struct_name: struct_name.clone(), field: name.to_string() },
                }),
            },
            // A generic struct, already instantiated at some concrete set
            // of type arguments — map the struct's own declared generic
            // parameter names to *these* arguments (positionally, `App`'s
            // own established convention) before resolving the field's
            // declared type, so `real: T` on `Complex<f64>` reads back as
            // `f64`, not the literal, meaningless name `T`.
            Ty::App(struct_name, type_args) => match self.registry.struct_fields(struct_name) {
                Some(fields) => match fields.iter().find(|f| f.name == name).cloned() {
                    Some(field) => {
                        // A pack-generic struct (`doc/backlog.md`'s own
                        // "Variadic generics" item) needs the same pack-aware
                        // resolution `infer_struct_lit_with_pack`'s own
                        // construction-site path already uses — `type_args`
                        // is already fully flat either way (`Box3<f64,2,2,
                        // 2>`'s own `[f64,2,2,2]`), just longer than
                        // `struct_generics`' own declared-name count once a
                        // pack is involved; `zip_struct_generics`'s own pure
                        // 1:1 zip would otherwise silently drop everything
                        // past the pack's own first slot.
                        let struct_generics = self.registry.struct_generics(struct_name).to_vec();
                        if struct_generics.last().is_some_and(GenericParam::is_variadic) {
                            let non_pack = &struct_generics[..struct_generics.len() - 1];
                            let pack_generic = struct_generics.last().expect("checked non-empty above");
                            let mapping: HashMap<String, Ty> =
                                non_pack.iter().zip(type_args).map(|(g, t)| (g.name().to_string(), t.clone())).collect();
                            let pack_tys = &type_args[non_pack.len().min(type_args.len())..];
                            Ok(self.ty_from_ast_mapped_with_pack(&field.ty, &mapping, pack_generic.name(), pack_tys))
                        } else {
                            let mapping = self.zip_struct_generics(struct_name, type_args);
                            Ok(self.ty_from_ast_mapped(&field.ty, &mapping))
                        }
                    }
                    None => Err(TypeError {
                        span,
                        kind: TypeErrorKind::NoSuchField { struct_name: struct_name.clone(), field: name.to_string() },
                    }),
                },
                None => Err(TypeError {
                    span,
                    kind: TypeErrorKind::NoSuchField { struct_name: struct_name.clone(), field: name.to_string() },
                }),
            },
            // Neither has fields — a function value or an array (indexing,
            // not field access, is how you reach into an array) rejected
            // the same way as any other fieldless concrete type. `Const`/
            // `ConstExpr` can't actually reach here in practice (nothing
            // produces one as an *expression's* type, only inside another
            // type's size slot), but handled the same way for
            // exhaustiveness rather than a `todo!()`/panic waiting to be
            // hit by a future caller. `Var` only ever reaches here via the
            // deferred path once `check_pending_field_accesses` has already
            // confirmed it's still unresolved — same "genuinely fieldless"
            // treatment, not a distinct case.
            Ty::Fn(..) | Ty::Array(..) | Ty::Const(_) | Ty::ConstExpr(..) | Ty::Var(_) | Ty::Pack(_) | Ty::PackResolved(_) | Ty::PackLen(_) => Err(TypeError {
                span,
                kind: TypeErrorKind::NoSuchField { struct_name: resolved.to_string(), field: name.to_string() },
            }),
        }
    }

    /// The `MethodCall` counterpart to `resolve_field_access` — given an
    /// *already concrete* base type and the call's own already-inferred
    /// argument types (arguments never depend on the base's own
    /// resolution, so both the immediate and deferred callers infer them
    /// up front, unchanged), finds and dispatches `name`. Deliberately
    /// excludes the `in_progress_methods` self-recursion branch: those
    /// entries only ever exist *during* a method body's own in-flight
    /// inference and are always removed before that same call's
    /// `finish_fn`/defaulting phase runs — by the time a deferred call
    /// reaches this helper, `in_progress_methods` is structurally
    /// guaranteed not to contain it, so the immediate call site keeps that
    /// check inline, before ever calling this. See
    /// `pending_method_calls`'s own doc comment for why a method call
    /// needs deferring at all.
    fn resolve_method_call(
        &mut self,
        resolved_base: &Ty,
        name: &str,
        arg_tys: &[Ty],
        arg_spans: &[Span],
        base_span: Span,
        call_span: Span,
    ) -> Result<Ty, TypeError> {
        let struct_name = match resolved_base {
            Ty::Con(n) => n.clone(),
            Ty::App(n, _) => n.clone(),
            // A function value, array, or const-value — none of these have
            // methods, rejected the same way `resolve_field_access` rejects
            // them for fields. `Ty::Var` is included only for
            // exhaustiveness — the deferred path only ever calls this once
            // `check_pending_method_calls` has already confirmed the base
            // is concrete; the immediate path already returned earlier for
            // a bare `Ty::Var`.
            Ty::Fn(..) | Ty::Array(..) | Ty::Const(_) | Ty::Var(_) | Ty::ConstExpr(..) | Ty::Pack(_) | Ty::PackResolved(_) | Ty::PackLen(_) => {
                return Err(TypeError {
                    span: call_span,
                    kind: TypeErrorKind::NoSuchMethod { struct_name: resolved_base.to_string(), method: name.to_string() },
                });
            }
        };
        let Some(entry) = self.registry.inherent_method(&struct_name, name).cloned() else {
            return Err(TypeError {
                span: call_span,
                kind: TypeErrorKind::NoSuchMethod { struct_name, method: name.to_string() },
            });
        };
        // `base` fills the method's own first parameter — an ordinary,
        // explicit positional argument, not a magic `self` (see
        // `grammar.pest`'s `inherent_impl` comment for why: this project
        // doesn't have implicit-anything elsewhere, no reason to invent one
        // here).
        if entry.method.params.is_empty() || entry.method.params.len() != arg_tys.len() + 1 {
            return Err(TypeError {
                span: call_span,
                kind: TypeErrorKind::ArityMismatch {
                    name: name.to_string(),
                    expected: entry.method.params.len(),
                    found: arg_tys.len() + 1,
                },
            });
        }
        // The impl block's own generics (`impl<T: Float> Vec2<T>`) — fresh
        // per call, bounds pushed as real `Constraint`s exactly like an
        // algebra impl's own (`fresh_generics_mapping`). `target_ty` —
        // built *from* `impl_mapping`, so `Boxed<T>` becomes
        // `App("Boxed", [impl_mapping["T"]])` — is what actually pins those
        // generics down once unified against `resolved_base` below; a bare
        // fresh var for `param_tys[0]` would *also* end up correctly
        // unified with `resolved_base`, but as its own, disconnected
        // variable, never actually feeding back into `impl_mapping` at all
        // — a real bug, found by testing: a generic inherent method's own
        // return type (`T`, resolved through this same `impl_mapping`)
        // came back as a bare, still-unconstrained variable instead of the
        // concrete type `base` actually has. Mirrors
        // `infer_inherent_impl_fn_generic`'s own identical fix for the
        // exact same reason, on the declaration side.
        let impl_mapping = self.fresh_generics_mapping(&entry.generics, call_span);
        let target_ty = self.ty_from_ast_mapped(&entry.target, &impl_mapping);
        let param_tys = self.inherent_method_param_tys(&entry.method.params, &impl_mapping, &target_ty);
        self.unify_at(base_span, &param_tys[0], resolved_base)?;
        for (pt, (at, sp)) in param_tys[1..].iter().zip(arg_tys.iter().zip(arg_spans.iter())) {
            self.unify_at(*sp, pt, at)?;
        }
        // Unlike an algebra call (which always has a real declared
        // signature, return type included, to fall back on) or a
        // top-level `fn` call (whose return type was already *inferred*,
        // once, by the whole-program pass, then generalized into a
        // reusable `Scheme` — see `callgraph.rs`) — dispatch here never
        // re-runs the method's own body at the call site (except for the
        // recursive, `in_progress_methods` case handled by the immediate
        // caller before ever reaching here), so an inherent method with no
        // explicit `->` annotation has no return type available *anywhere*
        // else to report... unless `callgraph::infer_inherent_impls_early`
        // already ran and published one (`self.inherent_patterns`, opted
        // into via `with_inherent_patterns` — only
        // `callgraph::infer_program` itself does today). Its own pattern's
        // free vars are named by generic-parameter *name*
        // (`generics_mapping`, built by a *different* `Infer` instance than
        // this call site's own `impl_mapping`), so reusing it means
        // cross-referencing by name and remapping through *this* call
        // site's own fresh vars, not substituting it in directly. Falls
        // back to the placeholder, same posture as every other
        // genuinely-unresolved case in this file, for a method whose own
        // return type couldn't be determined even by that early pass (e.g.
        // it depends on a top-level `fn`, not yet visible to it — see that
        // pass's own doc comment for why this is a deliberate, graceful
        // deferral, not a bug).
        Ok(entry
            .method
            .ret
            .as_ref()
            .map(|t| self.ty_from_ast_mapped(t, &impl_mapping))
            .or_else(|| {
                let pattern = self.inherent_patterns?.get(&(struct_name.clone(), name.to_string()))?;
                let remap: HashMap<TyVar, Ty> = pattern
                    .generics_mapping
                    .iter()
                    .filter_map(|(n, t)| match t {
                        Ty::Var(v) => Some((*v, impl_mapping[n].clone())),
                        _ => None,
                    })
                    .collect();
                Some(substitute(&pattern.ret_pattern, &remap))
            })
            .unwrap_or_else(|| Ty::Con("<not-yet-inferred>".to_string())))
    }

    /// Fresh parameter types (annotated → concrete, else a fresh variable)
    /// plus a fresh return-type variable for `f` — the shape a caller must
    /// know about *before* `f`'s own body has been inferred, so it can be
    /// published as a placeholder for recursive/mutually-recursive calls to
    /// resolve against. Shared by `infer_fn` (seeding its own self-reference)
    /// and the whole-program orchestrator (`callgraph.rs`, seeding an entire
    /// mutually-recursive group's placeholders before inferring any of their
    /// bodies). Also returns `f`'s own generics mapping (`fn_generics_mapping`)
    /// — an annotation naming one of `f`'s own declared type parameters (`x:
    /// T`) resolves through it instead of through the always-empty mapping
    /// `ty_from_ast` alone would use, and callers need the same mapping again
    /// for `f`'s declared return type (`infer_fn_raw`), which must resolve
    /// `T` to the exact same fresh variable, not a second, unrelated one.
    pub(crate) fn fresh_fn_shape(&mut self, f: &FnDecl) -> (Vec<Ty>, Ty, HashMap<String, Ty>) {
        let span =
            f.body.as_ref().and_then(|b| b.tail.as_deref().map(|t| t.span).or_else(|| b.stmts.last().map(|s| s.span)));
        let generics = span.map(|span| self.fn_generics_mapping(f, span)).unwrap_or_default();
        let param_types = f
            .params
            .iter()
            .map(|p| match &p.ty {
                Some(t) => self.ty_from_ast_mapped(t, &generics),
                None => self.vars.fresh(),
            })
            .collect();
        (param_types, self.vars.fresh(), generics)
    }

    /// Infers a whole function body in isolation (see module docs — no
    /// cross-function signature lookup yet, beyond the function's own name,
    /// see below). Parameters without an explicit annotation get a fresh
    /// type variable, exactly like an unannotated `fn add(a, b) { a + b }` —
    /// see `grammar.md`'s implicit-monomorphization discussion.
    ///
    /// `f`'s own name is bound into its body's environment as a monomorphic
    /// placeholder (`fresh_fn_shape`) *before* the body is inferred — a
    /// top-level `fn` can therefore always call itself. Deliberately
    /// monomorphic, not generalized, within its own body: this is the
    /// classical ML `let rec` restriction, not a limitation specific to this
    /// pass — a recursive definition can't be polymorphic *at its own
    /// recursive call sites*, only for callers outside it once it's fully
    /// defined (which is exactly what the whole-program pass in
    /// `callgraph.rs` provides for cross-function/mutual recursion; this
    /// method alone only ever sees itself, never a sibling).
    pub fn infer_fn(&mut self, f: &FnDecl) -> Result<Ty, TypeError> {
        let (param_types, ret_var, generics) = self.fresh_fn_shape(f);
        let mut outer = Env::new();
        outer.insert(f.name.clone(), Scheme::mono(Ty::Fn(param_types.clone(), Box::new(ret_var.clone()))));
        let result = self.infer_fn_raw(f, &outer, param_types.clone(), ret_var, &generics)?;
        self.finish_fn(f, param_types, result)
    }

    /// Like `infer_fn`, but for `monomorphize.rs`'s own duck-typed
    /// fallback (`detect_duck_typed_fns`): `param_types` are a call site's
    /// own already-concrete argument types, substituted in from the very
    /// start, instead of `fresh_fn_shape`'s own fresh `Ty::Var`s. This is
    /// what lets a nominally-typed expression that depends on a parameter's
    /// own concrete shape (field access on what would otherwise be an
    /// unconstrained generic parameter) resolve for real, instead of
    /// deferring to a `<not-yet-inferred>` placeholder that ordinary
    /// generalize-then-substitute monomorphization can never repair
    /// afterward (see `is_placeholder`'s own doc comment). Self-recursion,
    /// and any explicitly-annotated parameter alongside the unannotated
    /// one, both fall out of the exact same mechanism `infer_fn`/`infer_fn_
    /// raw` already provide -- nothing extra needed here.
    pub(crate) fn infer_fn_with_concrete_params(&mut self, f: &FnDecl, param_types: Vec<Ty>) -> Result<Ty, TypeError> {
        let (_, ret_var, generics) = self.fresh_fn_shape(f);
        let mut outer = Env::new();
        outer.insert(f.name.clone(), Scheme::mono(Ty::Fn(param_types.clone(), Box::new(ret_var.clone()))));
        let result = self.infer_fn_raw(f, &outer, param_types.clone(), ret_var, &generics)?;
        self.finish_fn(f, param_types, result)
    }

    /// The body-inference core shared by `infer_fn` and the whole-program
    /// orchestrator: binds `param_types` (already-computed — see
    /// `fresh_fn_shape`) on top of `outer` (the self-reference alone for
    /// `infer_fn`; a whole mutually-recursive group's siblings, layered over
    /// every earlier group's finished schemes, for `callgraph.rs`), infers
    /// the body, and ties `ret_var` — whatever was published to callers
    /// *before* this body was known — to what the body actually produced.
    ///
    /// Deliberately stops short of `finish_fn`'s defaulting/constraint-check/
    /// placeholder-check: those consult `self.pending_defaults`/
    /// `self.constraints`, which for a mutually-recursive group are only
    /// complete once *every* member's body has been walked — running them
    /// after just one member (as an earlier version of this refactor did)
    /// could default a numeric literal, or check a constraint, before a
    /// sibling's still-pending inference had contributed everything it might
    /// additionally unify against that same variable. `infer_fn` calls
    /// `finish_fn` itself, immediately after, since it only ever has one
    /// member; `callgraph.rs` calls this once per group member first, then
    /// runs the defaulting/constraint/placeholder steps exactly once for the
    /// whole group.
    pub(crate) fn infer_fn_raw(
        &mut self,
        f: &FnDecl,
        outer: &Env,
        param_types: Vec<Ty>,
        ret_var: Ty,
        generics: &HashMap<String, Ty>,
    ) -> Result<Ty, TypeError> {
        // A bodyless top-level `fn` is rejected before this is ever reached
        // (`callgraph::infer_program`, the one real caller that has an
        // enclosing `Item`'s own span to report against — `FnDecl` itself
        // carries none, see `ast.rs`) — `infer_fn` (this method's other,
        // test-only caller) never constructs one either.
        let body = f.body.as_ref().expect("infer_fn_raw requires a body; caller must validate first");
        self.active_generics = generics.clone();
        let mut env = outer.clone();
        for (p, ty) in f.params.iter().zip(&param_types) {
            env.insert(p.name.clone(), Scheme::mono(ty.clone()));
        }
        self.seed_const_generics(&f.generics, generics, &mut env);
        let result = self.infer_block(&env, body)?;
        if let Some(ret) = &f.ret {
            let declared = self.ty_from_ast_mapped(ret, generics);
            self.unify_at(ret.span, &declared, &result)?;
        }
        // Tie the placeholder promised to (possibly recursive) callers back
        // to what the body actually computed — not automatic: if a
        // recursive call's result is simply discarded (`fn f(x) { f(x-1); 0
        // }`), nothing else would ever connect `ret_var` to the body's real
        // result, silently leaving a self-call checked against a type the
        // function might not actually return.
        if let Some(span) = body.tail.as_deref().map(|t| t.span).or_else(|| body.stmts.last().map(|s| s.span)) {
            self.unify_at(span, &ret_var, &result)?;
        }
        Ok(result)
    }

    /// Infers an `impl <algebra><target>`'s own method **against the
    /// algebra's declared signature for it** — the conformance check that
    /// `infer_fn` alone can't provide, found missing by hand: an unannotated
    /// impl method (`impl TestAlg<i32> { fn add(x, y) { x + y } }`) used to
    /// infer `x`/`y` as bare, disconnected type variables, with nothing
    /// tying them to `i32` even though the enclosing `impl` unambiguously
    /// determines them. The algebra's declaration is the single source of
    /// truth here: an annotation on the impl method itself is optional, and
    /// when present is *checked* against what the algebra expects (mapped
    /// through the impl's own target type), never trusted independently —
    /// same reasoning as everywhere else this project avoids two sources of
    /// truth that could silently drift apart.
    ///
    /// `fallback_span` is used wherever there's no more specific span to
    /// point at — `FnDecl` itself carries no `Span` (only the enclosing
    /// `Item` does, see `ast.rs`), so a whole-signature problem (wrong
    /// arity, a name the algebra never declared) has nowhere more precise to
    /// point; callers should pass the enclosing `impl` `Item`'s own span.
    ///
    /// A thin wrapper over `infer_impl_fn_generic` with an empty impl-level
    /// generics list — the overwhelmingly common case (`impl TestAlg<i32>`,
    /// no `<T>` of its own) — kept as a separate, stable entry point so the
    /// many existing direct callers (tests especially) don't need to thread
    /// an empty slice through by hand.
    pub fn infer_impl_fn(
        &mut self,
        algebra: &str,
        target: &Type,
        f: &FnDecl,
        fallback_span: Span,
    ) -> Result<Ty, TypeError> {
        self.infer_impl_fn_generic(algebra, &[], target, f, fallback_span)
    }

    /// Like `infer_impl_fn`, but for an impl that declares its *own* generic
    /// parameters (`impl<T: Float> TestAlg<Complex<T>> { ... }`) — distinct
    /// from the algebra's own `<T>` (`algebra TestAlg<T> { ... }`, already
    /// handled below). Without this, `target`'s own generic argument (`T` in
    /// `Complex<T>`) would resolve through `ty_from_ast`'s always-empty
    /// mapping — the same bogus-literal-concrete-type-named-`"T"` bug
    /// `fn_generics_mapping` fixed for top-level `fn`s, here for `impl`
    /// targets instead.
    ///
    /// A thin wrapper over `infer_impl_fn_generic_with_env` with an empty
    /// outer `env` — kept as a separate, stable entry point for the same
    /// reason `infer_impl_fn` is: many existing direct callers (tests
    /// especially) have no whole-program `global_env` to hand it and don't
    /// need one.
    pub fn infer_impl_fn_generic(
        &mut self,
        algebra: &str,
        impl_generics: &[GenericParam],
        target: &Type,
        f: &FnDecl,
        fallback_span: Span,
    ) -> Result<Ty, TypeError> {
        self.infer_impl_fn_generic_with_env(
            &Env::new(),
            algebra,
            impl_generics,
            std::slice::from_ref(target),
            f,
            fallback_span,
        )
    }

    /// Like `infer_impl_fn_generic`, but seeds the method body's own `env`
    /// with `outer` first — every top-level `fn`'s finished, generalized
    /// `Scheme` (`callgraph::infer_program`'s own `global_env`), so an impl
    /// method's body can call an ordinary top-level function by name, not
    /// just dispatch other algebra operators through the registry. Without
    /// this, `env` was always empty (`Env::new()`), and *any* call to a
    /// top-level `fn` from inside an impl method — even a wholly ordinary,
    /// non-generic one — silently fell through to `infer_call`'s
    /// `<unresolved-call:...>` placeholder, found by direct testing:
    /// `impl TestAlg<i32> { fn gt(x, y) { helper(x); true } }` reported
    /// `helper(x):<unresolved-call:helper>` with *no error at all*, since
    /// the placeholder never happened to reach `gt`'s own exposed
    /// signature. This closes that specific gap — not the harder, still-open
    /// one: genuine mutual recursion *spanning* an impl method and a
    /// top-level `fn` (`fn helper(x) { add(x, x) }` where `add` dispatches
    /// into an impl method that itself calls `helper`) still isn't handled,
    /// since the static call-graph scan `callgraph.rs` builds its SCC
    /// grouping from has no way to know, without type information, *which*
    /// impl an algebra-dispatched call like `add(x, y)` will actually target
    /// — unlike an ordinary top-level call, resolved by name alone with no
    /// ambiguity. `outer` here is always already a *finished* `global_env`
    /// (built entirely before any impl method is ever inferred — see
    /// `dump.rs`'s own ordering), so this doesn't need to solve that harder
    /// problem to still be a real, sound improvement over an always-empty
    /// `env`.
    pub fn infer_impl_fn_generic_with_env(
        &mut self,
        outer: &Env,
        algebra: &str,
        impl_generics: &[GenericParam],
        targets: &[Type],
        f: &FnDecl,
        fallback_span: Span,
    ) -> Result<Ty, TypeError> {
        let Some(sig) = self.registry.fn_sig(algebra, &f.name).cloned() else {
            return Err(TypeError {
                span: fallback_span,
                kind: TypeErrorKind::NotDeclaredByAlgebra { algebra: algebra.to_string(), name: f.name.clone() },
            });
        };
        if sig.params.len() != f.params.len() {
            return Err(TypeError {
                span: fallback_span,
                kind: TypeErrorKind::ArityMismatch {
                    name: f.name.clone(),
                    expected: sig.params.len(),
                    found: f.params.len(),
                },
            });
        }

        // The impl's *own* generics (`T` in `impl<T: Float> ...`) map to
        // fresh variables (with their own bounds pushed as real
        // `Constraint`s, exactly like a top-level `fn`'s own — see
        // `fresh_generics_mapping`) *before* the target types themselves are
        // resolved, so `Complex<T>` becomes `App("Complex", [fresh])`, not a
        // bogus `App("Complex", [Con("T")])`.
        let impl_mapping = self.fresh_generics_mapping(impl_generics, fallback_span);
        self.active_generics = impl_mapping.clone();
        let target_tys: Vec<Ty> = targets.iter().map(|t| self.ty_from_ast_mapped(t, &impl_mapping)).collect();
        self.target_types = target_tys.clone();

        // The algebra's own generic parameters bind, *positionally*, to
        // this impl's own targets — `T` (the algebra's own, first generic)
        // is literally `Complex<fresh>` throughout this whole method's
        // inference, not a fresh variable disconnected from it.
        let generics = self.registry.generics(algebra).to_vec();
        let mut target_ty_iter = target_tys.iter();
        let mapping: HashMap<String, Ty> = generics
            .iter()
            .filter_map(|g| match g {
                // `.next()` per `Type` generic, in declaration order — the
                // *positional* correspondence `targets` (and `target_tys`,
                // built from it in the same order) already establishes.
                GenericParam::Type { name, .. } => target_ty_iter.next().map(|ty| (name.clone(), ty.clone())),
                // A `Const` generic on the algebra's own side consumes no
                // target slot — unlike `T`, it's never fixed by which impl
                // matched, only by whichever concrete call site this
                // method's own specialization is eventually built for
                // (`monomorphize.rs`'s own `ImplTemplate` worklist) — an
                // ordinary fresh var here, exactly like an unannotated
                // top-level `fn` parameter, not a value read off `targets`.
                // Previously omitted entirely (`=> None`), so a signature
                // referencing it (`x: [T; N]`) could never resolve `N` at
                // all here — found by direct testing, the same root cause
                // as `infer_algebra_call`'s own identical gap.
                GenericParam::Const { name, .. } => Some((name.clone(), self.vars.fresh())),
            })
            .collect();

        let mut env = outer.clone();
        let mut param_types = Vec::with_capacity(f.params.len());
        for (p, sig_p) in f.params.iter().zip(&sig.params) {
            let expected = match &sig_p.ty {
                Some(t) => self.ty_from_ast_mapped(t, &mapping),
                None => self.vars.fresh(),
            };
            let ty = match &p.ty {
                // Annotated anyway — checked against the algebra's own
                // expectation, not trusted as an independent second truth.
                // Resolved through the *impl's* own generics mapping (not
                // the algebra's) — an explicit annotation here would
                // naturally reference `T` meaning "this impl's own type
                // parameter" (`x: Complex<T>`), not the algebra's.
                Some(t) => {
                    let annotated = self.ty_from_ast_mapped(t, &impl_mapping);
                    self.unify_at(t.span, &expected, &annotated)?;
                    annotated
                }
                None => expected,
            };
            param_types.push(ty.clone());
            env.insert(p.name.clone(), Scheme::mono(ty));
        }
        self.seed_const_generics(impl_generics, &impl_mapping, &mut env);

        let expected_ret =
            sig.ret.as_ref().map(|t| self.ty_from_ast_mapped(t, &mapping)).unwrap_or_else(|| Ty::Con("()".to_string()));
        if let Some(ret) = &f.ret {
            let declared = self.ty_from_ast_mapped(ret, &impl_mapping);
            self.unify_at(ret.span, &expected_ret, &declared)?;
        }

        // A bodyless algebra-impl method (`fn add(x: f32, y: f32) -> f32;`)
        // is legal only when `extern` justifies it (`ast.rs`'s own `FnDecl::
        // is_extern`/`extern_symbol` — the same body-justifying case a
        // top-level `fn` gets, legal here too, for a real external C
        // symbol). An eventual codegen intrinsic no longer needs this at
        // all: it gets a real body containing a reserved `mlir::...` call
        // instead (`infer_expr`'s own `mlir::`-recognizing branch) — see
        // `mlir_lower.rs`'s own module doc comment. The declared return
        // type *is* the result in the `extern` case — there's no body to
        // compute one, and nothing else here needs a body: `param_types`/
        // `target_types` are already fully resolved from the signature/impl
        // targets alone.
        let result = match &f.body {
            Some(body) => {
                let result = self.infer_block(&env, body)?;
                let result_span = body.tail.as_deref().map(|t| t.span).unwrap_or(fallback_span);
                self.unify_at(result_span, &expected_ret, &result)?;
                result
            }
            None => {
                if !f.is_extern {
                    return Err(TypeError {
                        span: fallback_span,
                        kind: TypeErrorKind::MissingIntrinsicAttribute { name: f.name.clone() },
                    });
                }
                expected_ret.clone()
            }
        };

        self.quantify_impl_generics(&impl_mapping);
        let final_result = self.finish_fn(f, param_types, result)?;
        // `finish_fn` already re-resolves `self.param_types`/`node_types`
        // through the final substitution before returning — `target_types`
        // needs the identical treatment, since it was set (see above)
        // before body inference (and therefore defaulting/constraint-
        // checking) had a chance to pin anything down further.
        self.target_types = self.target_types.iter().map(|t| self.subst.apply(t)).collect();
        Ok(final_result)
    }

    /// Infers one method of an *inherent* impl (`impl<T> Vec2<T> { fn
    /// len(v) { ... } }`) — much simpler than `infer_impl_fn_generic_with_env`:
    /// no algebra, so no declared `fn_sig` to conform to, no target-pattern
    /// existence/coherence checking. Params are typed exactly like an
    /// ordinary top-level `fn`'s own (annotated → resolved through
    /// `impl_mapping`, so a `T`/`R`/`C` reference resolves to the impl's own
    /// fresh generic instead of a bogus literal `Con("T")`; unannotated →
    /// fresh var) — **except the first parameter**, which defaults to the
    /// impl's own `target` type when left unannotated, exactly the same
    /// "fall back to what the enclosing impl already declares" treatment an
    /// *algebra* impl's own unannotated params already get from that
    /// algebra's declared `fn_sig` (an inherent impl has no separate
    /// signature to fall back to at all — the target itself is the closest
    /// equivalent, and specifically *the first parameter's* role, "the
    /// value this method belongs to", is already established by the impl
    /// block itself: `impl Vec2 { fn len(v) { v.x } }` already says this
    /// method is about `Vec2` values, so `v` defaulting to `Vec2` isn't new
    /// magic, just the same "unannotated infers from context" this project
    /// already does everywhere else). Found necessary by direct testing:
    /// without this, `v.x` inside an unannotated `fn len(v) { v.x }` sees
    /// `v` as a bare, totally unconstrained fresh var (nothing at
    /// *declaration* time — as opposed to a *call site*, which does supply
    /// a concrete `resolved_base` — ties it to `Vec2` at all), so
    /// `FieldAccess` defers it as `<not-yet-inferred>`, which then survives
    /// to the method's own exposed signature and fails
    /// `check_no_placeholder` — a method that never mentions its own
    /// receiver's type anywhere explicit could never type-check standalone
    /// otherwise. If explicitly annotated anyway, checked against `target`
    /// like anything else in this file that could carry two independent
    /// truths, not trusted blindly. The body is inferred normally, the
    /// declared return type (if any) checked against the body's own result
    /// — then the same shared `finish_fn` tail every other inference entry
    /// point uses. `outer` is `global_env`, same reasoning as
    /// `infer_impl_fn_generic_with_env`'s own (an inherent method can call
    /// an ordinary top-level `fn` too).
    pub fn infer_inherent_impl_fn_generic(
        &mut self,
        outer: &Env,
        impl_generics: &[GenericParam],
        target: &Type,
        f: &FnDecl,
        fallback_span: Span,
    ) -> Result<Ty, TypeError> {
        let impl_mapping = self.fresh_generics_mapping(impl_generics, fallback_span);
        self.active_generics = impl_mapping.clone();
        let target_ty = self.ty_from_ast_mapped(target, &impl_mapping);
        let param_types = self.inherent_method_param_tys(&f.params, &impl_mapping, &target_ty);

        // Self-reference, the impl-method equivalent of `infer_fn`'s own
        // seeded placeholder — see `in_progress_methods`'s own doc comment
        // for why this lives there rather than in `env`: a recursive call
        // goes through `ExprKind::MethodCall`'s own dispatch, which never
        // consults `env` for its callee at all.
        let self_key =
            if let TypeKind::Path(p, _) = &target.kind { Some((p.segments.join("::"), f.name.clone())) } else { None };
        let ret_var = self.vars.fresh();
        if let Some(key) = &self_key {
            self.in_progress_methods.insert(key.clone(), (param_types.clone(), ret_var.clone()));
        }

        let result = self.infer_inherent_impl_fn_raw(
            outer,
            impl_generics,
            &impl_mapping,
            &target_ty,
            param_types.clone(),
            ret_var,
            f,
            fallback_span,
        );

        if let Some(key) = &self_key {
            self.in_progress_methods.remove(key);
        }

        let result = result?;
        self.quantify_impl_generics(&impl_mapping);
        self.finish_fn(f, param_types, result)
    }

    /// The body-inference core `infer_inherent_impl_fn_generic` (single
    /// method, self-recursion only) and `infer_inherent_impl_block` (every
    /// method of one impl block, real mutual recursion) both build on —
    /// stops short of `finish_fn`'s defaulting/constraint-check/placeholder-
    /// check, for the identical reason `infer_fn_raw` stops short of it for
    /// a mutually-recursive *group* of top-level `fn`s (see `callgraph.rs`'s
    /// own doc comment): that sequence must run once, after every method
    /// sharing this `Infer` has had its body walked, not per-method.
    fn infer_inherent_impl_fn_raw(
        &mut self,
        outer: &Env,
        impl_generics: &[GenericParam],
        impl_mapping: &HashMap<String, Ty>,
        target_ty: &Ty,
        param_types: Vec<Ty>,
        ret_var: Ty,
        f: &FnDecl,
        fallback_span: Span,
    ) -> Result<Ty, TypeError> {
        // Only meaningful when the first parameter is annotated: a call
        // site has no "annotation" of its own to double-check, just a
        // concrete `resolved_base` unified against `param_tys[0]` directly
        // (see `ExprKind::MethodCall`'s own handling) — but a declaration
        // whose first parameter is annotated (`fn len(v: Vec2) { ... }`)
        // must still agree with the impl's own target, not silently accept
        // a second, independent truth.
        let Some(body) = &f.body else {
            return Err(TypeError { span: fallback_span, kind: TypeErrorKind::MissingFnBody { name: f.name.clone() } });
        };
        if let Some(t) = f.params.first().and_then(|p| p.ty.as_ref()) {
            self.unify_at(t.span, target_ty, &param_types[0])?;
        }

        let mut env = outer.clone();
        for (p, ty) in f.params.iter().zip(&param_types) {
            env.insert(p.name.clone(), Scheme::mono(ty.clone()));
        }
        self.seed_const_generics(impl_generics, impl_mapping, &mut env);

        let result = self.infer_block(&env, body)?;
        if let Some(ret) = &f.ret {
            let declared = self.ty_from_ast_mapped(ret, impl_mapping);
            self.unify_at(ret.span, &declared, &result)?;
        }
        let result_span = body.tail.as_deref().map(|t| t.span).unwrap_or(fallback_span);
        self.unify_at(result_span, &ret_var, &result)?;
        Ok(result)
    }

    /// Like `infer_inherent_impl_fn_generic`, but for *every* method of one
    /// inherent impl block together, sharing one `Infer`/`Subst` — real
    /// mutual recursion between two separately-declared methods on the
    /// *same* struct (`fn is_even(w) { ... w.dec().is_odd() ... } fn
    /// is_odd(w) { ... w.dec().is_even() ... }`), which `in_progress_
    /// methods`'s single self-only slot can't express (it only ever holds
    /// an entry for whichever *one* method is currently being inferred).
    /// The same reasoning `callgraph::infer_program` already applies to
    /// top-level `fn`s, scoped down to one impl block instead of the whole
    /// program.
    ///
    /// Doesn't build a real call graph or run Tarjan's algorithm the way
    /// `infer_program` does — there's no *generalization* to order
    /// correctly here: an inherent method's own generics are re-
    /// instantiated fresh at every call site via `impl_mapping`, dispatch-
    /// style, never through a reusable `Scheme` the way a top-level `fn`'s
    /// own scheme is. Simply seeding every member's self/mutual-reference
    /// placeholder before inferring *any* of their bodies, then inferring
    /// all of them against one shared `Infer`, is already correct on its
    /// own — there's no ordering to get right, only "everyone sees everyone
    /// else" to arrange.
    ///
    /// Mirrors `callgraph.rs`'s own "defaulting/constraint-checking run once
    /// per group, not once per member" rule, for the identical reason:
    /// `finish_fn`'s sequence can't run per-method here, since `apply_
    /// defaults`/`check_pending_constraints` would then drain state a
    /// not-yet-inferred sibling still needed to contribute to.
    ///
    /// Returns `(impl_mapping, per-method results)`: `impl_mapping` is this
    /// block's own generics-name-to-fresh-var mapping (already built
    /// internally, previously discarded after use) — exposed so a caller
    /// that stores one of these methods' own pattern for *later*, cross-
    /// call-site reuse (`callgraph::infer_inherent_impls_early`) can remap
    /// its free vars through a *different* call site's own fresh generics
    /// by cross-referencing generic-parameter *name*, the same trick this
    /// file's own template-building code already relies on elsewhere. The
    /// second element is one entry per `fns`, each either the method's own
    /// final `(param_types, result)` or the `TypeError` that rejected it —
    /// the caller (`dump.rs`) reads `self.node_types` afterward for
    /// rendering, same as any other inference entry point; unlike
    /// `param_types` (a single, last-write-wins field on `Infer`, unsuited
    /// to more than one method sharing an instance), `node_types` is keyed
    /// by `NodeId` and already accumulates correctly across however many
    /// bodies this one `Infer` instance ends up walking.
    pub fn infer_inherent_impl_block(
        &mut self,
        outer: &Env,
        impl_generics: &[GenericParam],
        target: &Type,
        fns: &[FnDecl],
        fallback_span: Span,
    ) -> (HashMap<String, Ty>, HashMap<String, Result<(Vec<Ty>, Ty), TypeError>>) {
        let impl_mapping = self.fresh_generics_mapping(impl_generics, fallback_span);
        self.active_generics = impl_mapping.clone();
        let target_ty = self.ty_from_ast_mapped(target, &impl_mapping);
        let struct_name = match &target.kind {
            TypeKind::Path(p, _) => Some(p.segments.join("::")),
            _ => None,
        };

        // Seed every member's placeholder before inferring *any* of their
        // bodies — visible to every other member (mutual recursion) and to
        // itself (self-recursion).
        let mut placeholders: HashMap<String, (Vec<Ty>, Ty)> = HashMap::new();
        for f in fns {
            let param_types = self.inherent_method_param_tys(&f.params, &impl_mapping, &target_ty);
            let ret_var = self.vars.fresh();
            if let Some(name) = &struct_name {
                self.in_progress_methods.insert((name.clone(), f.name.clone()), (param_types.clone(), ret_var.clone()));
            }
            placeholders.insert(f.name.clone(), (param_types, ret_var));
        }

        let mut raw_results: HashMap<String, Result<Ty, TypeError>> = HashMap::new();
        for f in fns {
            let (param_types, ret_var) = placeholders[&f.name].clone();
            // `check_pending_type_names` right here, per member, not folded
            // into a group-wide sweep — see `Infer::check_pending_type_
            // names`'s own doc comment for why: each entry belongs to
            // whichever one member's body produced it.
            let outcome = self
                .infer_inherent_impl_fn_raw(outer, impl_generics, &impl_mapping, &target_ty, param_types, ret_var, f, fallback_span)
                .and_then(|ty| self.check_pending_type_names().map(|()| ty))
                .and_then(|ty| self.check_pending_div_by_zero().map(|()| ty));
            raw_results.insert(f.name.clone(), outcome);
        }

        if let Some(name) = &struct_name {
            for f in fns {
                self.in_progress_methods.remove(&(name.clone(), f.name.clone()));
            }
        }

        self.quantify_impl_generics(&impl_mapping);
        self.apply_defaults();
        // A constraint failure here is a property of the block's mutual
        // definition as a whole, not attributable to one specific member —
        // reported against every member whose own raw inference otherwise
        // succeeded (a raw-inference failure is already more specific and
        // is left alone), mirroring `callgraph.rs`'s identical choice.
        if let Err(e) = self.check_pending_constraints_and_indices() {
            for outcome in raw_results.values_mut() {
                if outcome.is_ok() {
                    *outcome = Err(e.clone());
                }
            }
        }
        // Same "resolve now that defaulting has run, attribute a failure to
        // the whole block" posture as `check_pending_constraints` just
        // above — see `finish_fn`'s identical pairing for the single-
        // function path.
        if let Err(e) = self.check_pending_field_accesses().and_then(|()| self.check_pending_method_calls()) {
            for outcome in raw_results.values_mut() {
                if outcome.is_ok() {
                    *outcome = Err(e.clone());
                }
            }
        }

        // Re-resolve `node_types` through the final substitution before any
        // caller (`dump.rs`) reads it — mirrors `finish_fn`'s identical
        // step. Skipping this is a real bug, found by testing: a recursive
        // call site's own node (`w.dec().is_odd()`, inside `is_even`'s own
        // body) is recorded *while* `is_odd`'s own `ret_var` is still a bare
        // variable — by the time this method returns, `check_pending_
        // constraints`/unification elsewhere may have pinned it fully
        // concrete, but nothing had gone back to update the already-recorded
        // node entry to match.
        let resolved_nodes: Vec<(NodeId, Ty)> = self.node_types.iter().map(|(id, t)| (*id, self.subst.apply(t))).collect();
        self.node_types = resolved_nodes.into_iter().collect();
        self.resolve_lambda_schemes();

        let mut results = HashMap::new();
        for f in fns {
            let (param_types, _) = &placeholders[&f.name];
            let outcome = raw_results.remove(&f.name).unwrap();
            let resolved = outcome.map(|result_ty| {
                let final_params: Vec<Ty> = param_types.iter().map(|t| self.subst.apply(t)).collect();
                let final_result = self.subst.apply(&result_ty);
                (final_params, final_result)
            });
            let checked = resolved.and_then(|(final_params, final_result)| {
                check_no_placeholder(f, &final_result, &final_params)?;
                Ok((final_params, final_result))
            });
            results.insert(f.name.clone(), checked);
        }
        (impl_mapping, results)
    }

    /// The "first parameter defaults to the impl's own target type when left
    /// unannotated" convention shared by an inherent method's own
    /// declaration (`infer_inherent_impl_fn_generic`) and a method call's
    /// own dispatch (`ExprKind::MethodCall`) — see either call site's own
    /// doc comment for why specifically the *first* parameter gets this
    /// treatment (there's no magic `self`, just an ordinary positional
    /// parameter whose role — "the value this method belongs to" — the
    /// enclosing `impl` block already establishes). An explicitly annotated
    /// parameter (any position) is resolved through `impl_mapping` like any
    /// other type reference; deliberately doesn't unify or check anything
    /// here — callers do that with whatever they specifically have on hand
    /// (a concrete `resolved_base` at a call site, only `target_ty` itself
    /// at declaration time).
    fn inherent_method_param_tys(&mut self, params: &[Param], impl_mapping: &HashMap<String, Ty>, target_ty: &Ty) -> Vec<Ty> {
        params
            .iter()
            .enumerate()
            .map(|(i, p)| match &p.ty {
                Some(t) => self.ty_from_ast_mapped(t, impl_mapping),
                None if i == 0 => target_ty.clone(),
                None => self.vars.fresh(),
            })
            .collect()
    }

    /// Shared tail of `infer_fn`/`infer_impl_fn`: defaulting, the qualified-
    /// constraint sweep, finalizing `node_types`/`param_types` through the
    /// last substitution, and the unresolved-placeholder safety net.
    fn finish_fn(&mut self, f: &FnDecl, param_types: Vec<Ty>, result: Ty) -> Result<Ty, TypeError> {
        self.apply_defaults();
        // After defaulting, since defaulting can turn an abstract
        // `Num`-constrained variable concrete — check it against that
        // default, don't just assume defaulting made it automatically fine.
        self.check_pending_constraints_and_indices()?;
        self.check_pending_type_names()?;
        self.check_pending_div_by_zero()?;
        // Same reasoning as `check_pending_constraints` above — a field
        // access/method call deferred because its base was still a bare
        // `Ty::Var` at the point it was written now has its answer, one way
        // or the other, now that defaulting has run.
        self.check_pending_field_accesses()?;
        self.check_pending_method_calls()?;

        // Fully re-resolve everything through the final substitution before
        // handing it back — `node_types`/`param_types` may have captured a
        // type before some later unification (or defaulting) pinned it down
        // further; callers should never need to know that and re-apply
        // `subst` themselves.
        self.param_types = param_types.iter().map(|t| self.subst.apply(t)).collect();
        let resolved_nodes: Vec<(NodeId, Ty)> =
            self.node_types.iter().map(|(id, t)| (*id, self.subst.apply(t))).collect();
        self.node_types = resolved_nodes.into_iter().collect();
        self.resolve_lambda_schemes();

        let final_result = self.subst.apply(&result);

        // A placeholder (`<unresolved-call:...>`, ...) surviving all the way
        // to the function's own exposed return/parameter types must not be
        // reported as a successful inference — found by actually running
        // the CLI on a real file: `fn main() { let f = fn(x){x+1}; f(1) }`
        // with no `Ring` registered silently "succeeded", reporting `main`'s
        // own type as `<unresolved-call:add>`, no error anywhere. A
        // placeholder is explicitly "we don't know" (see `is_placeholder`) —
        // letting it flow through unresolved to become the final, reported
        // answer is not the same as things actually working.
        check_no_placeholder(f, &final_result, &self.param_types)?;

        Ok(final_result)
    }

    /// Unifies and, on failure, attaches `span` — the one place a raw
    /// `UnifyError` becomes a located `TypeError`. `pub(crate)` for the same
    /// reason `ty_from_ast` is: `callgraph.rs` needs it to bind an `extern
    /// fn`'s fresh `ret_var` (from `fresh_fn_shape`) to its declared return
    /// type, with no body to infer one from instead.
    pub(crate) fn unify_at(&mut self, span: Span, a: &Ty, b: &Ty) -> Result<(), TypeError> {
        unify(&mut self.subst, a, b).map_err(|e| TypeError { span, kind: TypeErrorKind::Unify(e) })
    }

    /// `∀vars. ty` where `vars` = free variables of `ty` that are free
    /// *nowhere* in `env` — the standard HM generalization rule, computed
    /// against the substitution's current state (`self.subst.apply`), since
    /// `env`'s stored types may reference variables unified since they were
    /// inserted.
    ///
    /// A number literal's own type variable is deliberately **not** excluded
    /// here, even though it's tracked in `self.pending_defaults` — an
    /// earlier version of this function did exclude it, reasoning that
    /// `apply_defaults` would otherwise force it concrete "after the fact"
    /// and silently collapse whatever had just been generalized. That
    /// reasoning was wrong, and excluding it broke something real: it
    /// blocked `fn add_one(x) { x + 1 }` from ever being usable at more than
    /// one numeric type, because unifying `x` with the literal `1` merges
    /// their variables into one, and excluding that one representative
    /// excluded `x`'s own genericity right along with it — precisely the
    /// C++/Rust generic-numeric-literal pain (`num_traits::One`-style
    /// boilerplate) this project doesn't have to inherit. Every variable
    /// quantified here is recorded in `self.quantified`, which
    /// `apply_defaults` consults and refuses to touch — not merely "it would
    /// be harmless if it did" (an earlier version relied on that argument
    /// alone: `instantiate`/`substitute` never consult `self.subst` for a
    /// scheme's own quantified variables, so a *future* instantiation was
    /// never actually at risk) but a real requirement, found by testing:
    /// `node_types` (what `--dump-inference-pass` renders) *does* read
    /// through `self.subst`, so a defaulted-but-still-quantified variable
    /// showed a concrete type in a function's own body that directly
    /// contradicted the still-generic signature reported for the very same
    /// variable — see `doc/type_inference.md`'s "must never be defaulted"
    /// section for the exact repro. The literal's `Constraint` (`Num`,
    /// generated at the literal itself — see `infer_expr`) travels into the
    /// scheme right alongside the variable, which is what keeps this sound:
    /// `add_one`'s scheme becomes `∀t. Num t => (t) -> t`, not the unsound
    /// fully-unconstrained `∀t. (t) -> t`.
    ///
    /// Returns `Err` for a provably-never-satisfiable scheme — `doc/backlog.
    /// md`'s own "Scheme satisfiability at generalization time" item: a
    /// mutually-recursive group whose members disagree on shape (`Int t`
    /// and `Float t` on the same quantified `t`, found by direct testing to
    /// generalize completely silently before this fix) used to slip through
    /// entirely if nothing outside the group ever called into it —
    /// `check_pending_constraints` deliberately *skips* any constraint
    /// whose type is already quantified (correct in general: nothing to
    /// check against until instantiation), but that also means a scheme's
    /// own *internal* consistency was never checked at the one point it's
    /// actually knowable: right here. See the satisfiability check just
    /// below `constraints`' own construction for the (deliberately narrow —
    /// single-target constraints only) check itself.
    pub(crate) fn generalize(&mut self, env: &Env, ty: &Ty) -> Result<Scheme, TypeError> {
        let ty = self.subst.apply(ty);
        let mut ty_fv = HashSet::new();
        free_vars(&ty, &mut ty_fv);

        let mut env_fv = HashSet::new();
        for scheme in env.values() {
            let resolved = self.subst.apply(&scheme.ty);
            let mut fv = HashSet::new();
            free_vars(&resolved, &mut fv);
            env_fv.extend(fv.into_iter().filter(|v| !scheme.vars.contains(v)));
        }

        let mut vars: Vec<TyVar> = ty_fv.into_iter().filter(|v| !env_fv.contains(v)).collect();
        vars.sort();
        // Recorded so `apply_defaults` never binds one of these afterward —
        // see `quantified`'s own doc comment.
        self.quantified.extend(vars.iter().copied());

        // Any pending constraint mentioning one of the variables just
        // quantified travels into this scheme — that's the actual
        // "qualified types" step: `fn add(a, b) { a + b }` ends up with
        // `T: Ring` in *its own* scheme not because anything special-cases
        // `add`, but because the constraint `add`'s resolution generated
        // for `T` is still open when `add`'s own `let`/`fn` boundary is
        // reached, and gets swept up here like any other free variable.
        //
        // Copied, not moved out of `self.constraints` — a real bug, found by
        // testing: for an ordinary `let`, no other binding ever shares its
        // quantified variables (any later use goes through `instantiate`,
        // which always mints fresh ones), so removing a matched constraint
        // here was never observably different from copying it. That
        // assumption breaks for `callgraph.rs`'s whole-program pass, which
        // calls `generalize` once per member of a mutually-recursive group
        // *before* any of them are instantiated — two members can
        // legitimately still share the very same raw, not-yet-instantiated
        // quantified variable (their two `if`/`else` branches both feeding a
        // value into the same mutually-recursive return type, say). Moving
        // the constraint meant whichever member's `generalize` call ran
        // first "stole" it, leaving the other member's own scheme with an
        // empty constraint list for a variable it was just as responsible
        // for. `check_pending_constraints` now skips any constraint whose
        // type resolves to a variable in `self.quantified` (mirroring
        // `apply_defaults`'s own guard) instead of relying on this sweep to
        // have removed it — every scheme that quantifies a shared variable
        // gets its own correct copy, and the original stays behind, inert
        // (checked properly at each of *its own* future instantiation sites
        // instead, exactly like any other constraint).
        let var_set: HashSet<TyVar> = vars.iter().copied().collect();
        let mut constraints = Vec::new();
        for c in &self.constraints {
            let resolved_tys: Vec<Ty> = c.tys.iter().map(|t| self.subst.apply(t)).collect();
            let mut fv = HashSet::new();
            for t in &resolved_tys {
                free_vars(t, &mut fv);
            }
            if fv.iter().any(|v| var_set.contains(v)) {
                constraints.push(Constraint {
                    algebra: c.algebra.clone(),
                    tys: resolved_tys,
                    gating_indices: c.gating_indices.clone(),
                    span: c.span,
                });
            }
        }

        // Satisfiability check — see this method's own doc comment. Only
        // single-target constraints (`tys == [Ty::Var(v)]`, the shape-
        // constraint case this item is actually about — `Int`/`Float`/`Num`/
        // a declared `T: Bound`) are considered; a multi-target constraint
        // (a heterogeneous algebra call spanning several quantified
        // variables at once, e.g. `MatMul<A,B,C>`) is a structural-match
        // question, not a per-variable satisfiability one — not attempted
        // here, flagged not hidden, matching this project's own posture for
        // a real-but-not-yet-attempted gap. A constraint whose own algebra
        // isn't registered at all is skipped entirely (mirrors `check_
        // pending_constraints`'s own identical guard) — this check must stay
        // a complete no-op for a registry that never declares `Int`/`Float`/
        // `Num` in the first place, exactly like every other consumer of
        // those shape constraints already is.
        let mut by_var: HashMap<TyVar, Vec<&Constraint>> = HashMap::new();
        for c in &constraints {
            if let [Ty::Var(v)] = c.tys.as_slice() {
                if self.registry.has_algebra(&c.algebra) {
                    by_var.entry(*v).or_default().push(c);
                }
            }
        }
        for cs in by_var.values() {
            if cs.len() < 2 {
                continue;
            }
            if let Some(algebras) = self.unsatisfiable_bounds(cs.iter().map(|c| c.algebra.as_str())) {
                return Err(TypeError { span: cs[0].span, kind: TypeErrorKind::UnsatisfiableScheme { algebras } });
            }
        }

        // Any of the just-quantified vars that are const generics carry
        // their own declared width along too — same reasoning as the
        // constraint sweep just above (a bound checked later, once this
        // scheme is instantiated, needs the *real* width, not a guess; see
        // `check_pending_constraints`'s own `Ty::Const` bridge).
        let const_widths: HashMap<TyVar, Ty> =
            self.subst.const_widths.iter().filter(|(v, _)| var_set.contains(v)).map(|(v, t)| (*v, t.clone())).collect();

        Ok(Scheme { vars, constraints, ty, const_widths })
    }

    /// Re-resolves every `lambda_schemes` entry's own `ty`/`const_widths`
    /// through the current `self.subst` — mirrors `node_types`'s own final
    /// resolution sweep (see `finish_fn`/`infer_impl_fn_generic_with_env`)
    /// and for the identical reason: `generalize` snapshots `self.subst` at
    /// the moment a `let`-bound lambda is generalized, but a variable free
    /// in the *enclosing* scope (kept monomorphic on purpose, so excluded
    /// from `vars`) can still get pinned down further by unification later
    /// in the same function body. `vars` themselves are never touched here
    /// (`self.subst` has nothing to say about a variable that's already
    /// been quantified away) — only the non-quantified remainder can move.
    fn resolve_lambda_schemes(&mut self) {
        let resolved: Vec<(NodeId, Scheme)> = self
            .lambda_schemes
            .iter()
            .map(|(id, scheme)| {
                let mut scheme = scheme.clone();
                scheme.ty = self.subst.apply(&scheme.ty);
                scheme.const_widths = scheme.const_widths.iter().map(|(v, t)| (*v, self.subst.apply(t))).collect();
                (*id, scheme)
            })
            .collect();
        self.lambda_schemes = resolved.into_iter().collect();
    }

    /// Every currently-pending constraint that shares at least one free
    /// variable with `ty` (after `self.subst`) — copied, not removed, same
    /// reasoning as `generalize`'s own sweep (a shared variable must stay
    /// checkable from every type that exposes it, not just whichever call
    /// claims it first). Unlike `generalize`'s own sweep, this doesn't
    /// filter by "free in `ty` but not in some enclosing `env`" — there's no
    /// quantification happening here, just forwarding whatever's still
    /// unresolved so a *later* caller, once it pins the variable concrete,
    /// can still check it.
    ///
    /// `callgraph.rs`'s own use: a *nullary* top-level `fn`'s returned type
    /// can still carry a free variable with a real pending constraint on it
    /// (`fn make_adder() { fn(x) { add(x, x) } }`'s returned closure's own
    /// `T: Ring`) even though the Monomorphism Restriction means it's never
    /// `generalize`d (see that module's own doc comment on why nullary
    /// bindings skip generalization) — without this, found by direct
    /// testing, the constraint silently vanished the moment that function's
    /// own `Infer` instance (and its `self.constraints`) was discarded at
    /// the end of its own group: `let f = make_adder(); f(true)` type-check
    /// *succeeded*, with no `Ring<bool>` impl anywhere, because
    /// `Scheme::mono` — the only thing crossing that group boundary for a
    /// nullary binding — carries zero constraints by construction.
    pub(crate) fn constraints_touching(&self, ty: &Ty) -> Vec<Constraint> {
        let mut ty_fv = HashSet::new();
        free_vars(&self.subst.apply(ty), &mut ty_fv);
        self.constraints
            .iter()
            .filter_map(|c| {
                let resolved_tys: Vec<Ty> = c.tys.iter().map(|t| self.subst.apply(t)).collect();
                let mut fv = HashSet::new();
                for t in &resolved_tys {
                    free_vars(t, &mut fv);
                }
                fv.iter().any(|v| ty_fv.contains(v)).then(|| Constraint {
                    algebra: c.algebra.clone(),
                    tys: resolved_tys,
                    gating_indices: c.gating_indices.clone(),
                    span: c.span,
                })
            })
            .collect()
    }

    /// Instantiates a scheme with fresh type variables — every reference to
    /// a generalized binding gets its own independent copy, which is what
    /// lets e.g. `id(1)` and `id(true)` coexist against the same `∀a. a->a`.
    /// The scheme's own constraints are renamed by the same fresh mapping
    /// and re-queued (`self.constraints`) exactly as if freshly generated at
    /// this call site — each instantiation gets its own copy of "T: Ring",
    /// checked/propagated independently, same as the type variable itself.
    fn instantiate(&mut self, scheme: &Scheme) -> Ty {
        self.instantiate_with_mapping(scheme).0
    }

    /// Like `instantiate`, but also hands back the fresh-variable mapping it
    /// built — needed by `infer_call`'s explicit-turbofish handling, which
    /// must unify each `f::<...>` argument against the *specific* fresh
    /// variable standing in for one of `scheme.vars`, not just against
    /// whatever position it happens to occupy in the substituted `Ty`
    /// itself (a variable can appear in multiple places, or nowhere
    /// syntactically obvious to zip against). `scheme.vars` is `generalize`'s
    /// own numeric-sort of the quantified set — since a function's own
    /// declared generics are minted as a single contiguous batch, in
    /// declaration order, *before* anything else in its body (see
    /// `fresh_fn_shape`), their `TyVar` ids stay lowest and in that same
    /// order, so `scheme.vars`'s order reliably matches declaration order
    /// for the common case. It does *not* if a declared generic never ends
    /// up free in the function's own signature at all (an unused type
    /// parameter) — a real, narrow edge case, not attempted here; a mismatch
    /// there would show up as `generic_arg_to_ty`'s unification landing on
    /// the wrong variable rather than a clean rejection.
    fn instantiate_with_mapping(&mut self, scheme: &Scheme) -> (Ty, HashMap<TyVar, Ty>) {
        let mapping: HashMap<TyVar, Ty> = scheme.vars.iter().map(|v| (*v, self.vars.fresh())).collect();
        for c in &scheme.constraints {
            self.constraints.push(Constraint {
                algebra: c.algebra.clone(),
                tys: c.tys.iter().map(|t| substitute(t, &mapping)).collect(),
                gating_indices: c.gating_indices.clone(),
                span: c.span,
            });
        }
        // Re-key `scheme.const_widths` through the same fresh mapping —
        // without this, a constraint re-queued just above (against a *fresh*
        // copy of a const generic's own var) would have no way to recover
        // that var's declared width once `check_pending_constraints` runs
        // again for *this* instantiation.
        for (v, width) in &scheme.const_widths {
            if let Some(Ty::Var(fresh)) = mapping.get(v) {
                self.subst.set_const_width(*fresh, width.clone());
            }
        }
        (substitute(&scheme.ty, &mapping), mapping)
    }

    /// Whether a group of bounds — all understood to constrain the *same*
    /// type variable — could ever be jointly satisfied by one shared
    /// concrete type. Returns the deduplicated, sorted list of conflicting
    /// algebra names if their `Registry::candidates_for` sets have an empty
    /// intersection (e.g. `Int` and `Float`, satisfied only by `i32` and
    /// `f64` respectively), or `None` if they're satisfiable — including the
    /// trivial case of fewer than two *registered* bounds, since there's
    /// nothing to conflict with. An algebra this registry never declares is
    /// skipped entirely rather than counted, mirroring `check_pending_
    /// constraints`'s own identical guard: this check must stay a no-op for
    /// a registry that never declares `Int`/`Float`/`Num` in the first
    /// place. Shared between `generalize`'s own scheme-satisfiability check
    /// and `check_no_overlapping_impls`'s bound-satisfiability gate — both
    /// are really asking the identical question over two different sources
    /// of "which bounds land on the same variable."
    fn unsatisfiable_bounds<'a>(&self, algebras: impl Iterator<Item = &'a str>) -> Option<Vec<String>> {
        let mut names: Vec<&str> = Vec::new();
        let mut candidates: Option<HashSet<String>> = None;
        for algebra in algebras {
            if !self.registry.has_algebra(algebra) {
                continue;
            }
            names.push(algebra);
            let this = self.registry.candidates_for(algebra);
            candidates = Some(match candidates {
                None => this,
                Some(prev) => prev.intersection(&this).cloned().collect(),
            });
        }
        if names.len() < 2 || !candidates.is_some_and(|s| s.is_empty()) {
            return None;
        }
        // Deduplicated and sorted — the same algebra can land in `names`
        // more than once (e.g. `Num` pushed alongside both `Int` and
        // `Float`), and the message should read as a clear, deterministic
        // list of what's actually in conflict.
        let mut algebras: Vec<String> = names.iter().map(|s| s.to_string()).collect();
        algebras.sort();
        algebras.dedup();
        Some(algebras)
    }

    /// Rejects two `impl`s of the same algebra whose own generic target
    /// *patterns* can structurally unify against a common instantiation —
    /// `impl<T: Float> Ring<Complex<T>>` and `impl<T: Ord> Ring<Complex<T>>`
    /// could both apply to some hypothetical `Complex<X>` satisfying both
    /// bounds, and nothing else in this file would ever notice: neither
    /// impl is ever compared against the other directly.
    /// `has_matching_impl`'s own speculative per-*query* probe can't catch
    /// this either — it only ever asks "does *some* impl apply" for one
    /// concrete type at a time, correctly stopping at the first match; nor
    /// would it be told a query type that only checking every possible
    /// concrete instantiation could reveal. This runs once, over the whole
    /// registry, independent of any particular call site — the same
    /// "whole program, all at once" scope `driver::merge_programs`'s own
    /// duplicate-name detection already uses for an analogous problem.
    ///
    /// Shape overlap alone isn't enough to report, though: `impl<T: Int>
    /// Ring<Box<T>>` and `impl<T: Float> Ring<Box<T>>` are shape-identical
    /// (`Box<_>`) but can never collide at any real call site, since no
    /// concrete type is ever both `Int` and `Float` — a real false positive,
    /// found by direct testing. Once `all_overlap` confirms the shapes
    /// coincide, `trial` (already populated by the very `unify` calls that
    /// proved it) tells us exactly how each side's own generics got merged;
    /// every `GenericParam::Type`'s bounds are grouped by their merged
    /// variable's resolved type, and `unsatisfiable_bounds` — the same
    /// question `generalize`'s own satisfiability check asks, just fed a
    /// different source of "bounds on one variable" — decides whether each
    /// group could really share a concrete type. A `Const` generic has no
    /// bounds at all, so nothing to check there. This narrows, rather than
    /// removes, the shape check's own deliberate conservatism (still Rust's
    /// own default-coherence posture otherwise: two *unbounded* overlapping
    /// patterns, or two whose bounds *do* share a candidate, are still
    /// rejected — no attempt at a fully general bound-satisfiability
    /// solver). Unifies against a throwaway `Subst`, same reasoning as
    /// `has_matching_impl`'s own `trial` — a shape collision here must never
    /// leave partial bindings behind for the next pair to trip over, and
    /// must never touch `self.subst` at all (there's no real query type
    /// involved, just two patterns compared to each other).
    pub fn check_no_overlapping_impls(&mut self) -> Vec<TypeError> {
        let mut errors = Vec::new();
        for algebra in self.registry.algebra_names().map(str::to_string).collect::<Vec<_>>() {
            // `all_impls` (every target, single or multi, generic or fully
            // concrete alike) rather than `generic_impls` (single-target
            // generic only) — a heterogeneous algebra's own impls need the
            // *whole* target tuple compared together: two impls overlap
            // only if *every* position could coincide simultaneously, not
            // just one. Concrete/non-generic impls ride along harmlessly
            // (two differently-concrete targets simply never unify), no
            // separate fast path needed for a check that only ever runs
            // once, over the whole registry.
            let impls = self.registry.all_impls(&algebra);
            for i in 0..impls.len() {
                for j in (i + 1)..impls.len() {
                    let (generics_a, targets_a) = &impls[i];
                    let (generics_b, targets_b) = &impls[j];
                    if targets_a.len() != targets_b.len() {
                        continue;
                    }
                    let mapping_a = self.fresh_vars_for_generics(generics_a);
                    let mapping_b = self.fresh_vars_for_generics(generics_b);
                    let mut trial = Subst::default();
                    let all_overlap = targets_a.iter().zip(targets_b.iter()).all(|(ta, tb)| {
                        let pattern_a = self.ty_from_ast_mapped(ta, &mapping_a);
                        let pattern_b = self.ty_from_ast_mapped(tb, &mapping_b);
                        unify(&mut trial, &pattern_a, &pattern_b).is_ok()
                    });
                    if all_overlap {
                        // Group every `Type` generic's own bounds — from
                        // *both* impls — by the resolved type its fresh var
                        // now shares in `trial`; two bounds land in the same
                        // group exactly when the shape match just above
                        // actually merged their variables together.
                        let mut groups: Vec<(Ty, Vec<&str>)> = Vec::new();
                        for (generics, mapping) in [(*generics_a, &mapping_a), (*generics_b, &mapping_b)] {
                            for g in generics {
                                if let GenericParam::Type { name, bounds, .. } = g {
                                    if bounds.is_empty() {
                                        continue;
                                    }
                                    let root = trial.apply(&mapping[name]);
                                    match groups.iter_mut().find(|(r, _)| *r == root) {
                                        Some((_, bs)) => bs.extend(bounds.iter().map(String::as_str)),
                                        None => groups.push((root, bounds.iter().map(String::as_str).collect())),
                                    }
                                }
                            }
                        }
                        let bounds_admit_a_shared_type =
                            groups.iter().all(|(_, bs)| self.unsatisfiable_bounds(bs.iter().copied()).is_none());
                        if bounds_admit_a_shared_type {
                            let fmt_targets =
                                |ts: &[&Type]| ts.iter().map(|t| fmt_type(t)).collect::<Vec<_>>().join(", ");
                            errors.push(TypeError {
                                span: targets_b[0].span,
                                kind: TypeErrorKind::OverlappingImpls {
                                    algebra: algebra.clone(),
                                    a: fmt_targets(targets_a),
                                    b: fmt_targets(targets_b),
                                },
                            });
                        }
                    }
                }
            }
        }
        errors
    }

    /// Every impl of `algebra` whose target pattern(s) unify, positionally,
    /// against `query`, *and* whose own generic bounds are satisfied — the
    /// shared candidate search underneath both `match_impl` (existence
    /// probe: does at least one exist) and `dispatch_algebra_call`
    /// (committing dispatch: does *exactly* one exist, or do several exist
    /// but agree anyway). Candidates come from `Registry::all_impls` (every
    /// impl of `algebra`, single- or multi-target, generic or fully
    /// concrete alike — the piece `Registry` itself can't do this matching,
    /// it's deliberately just data, see its own module docs); a mismatched
    /// target-tuple length skips the candidate outright.
    ///
    /// Each candidate's own unification runs against a *cloned* `Subst`
    /// (`trial`), never `self.subst` directly — a rejected candidate must
    /// never leave partial bindings behind for the next one to trip over,
    /// and this method itself never commits any candidate's `trial` back
    /// into `self.subst` (that's each caller's own decision — `match_impl`
    /// never needs to; `dispatch_algebra_call` always does, to *one*
    /// specific winner). Fresh variables for the candidate's own generics
    /// are minted via `fresh_vars_for_generics` specifically (not
    /// `fresh_generics_mapping`) for the same reason: no bound should
    /// become a *real*, persistent `Constraint` in `self.constraints` just
    /// because one candidate was tried and rejected — bounds are instead
    /// checked directly, recursively (via `has_matching_impl`): a
    /// structural match against `Complex<T>` alone isn't enough if `T`'s
    /// own resolved argument doesn't actually satisfy `T: Float`.
    fn matching_impls(&mut self, algebra: &str, query: &[Ty]) -> Vec<Subst> {
        let mut out = Vec::new();
        for (generics, targets) in self.registry.all_impls(algebra) {
            if targets.len() != query.len() {
                continue;
            }
            let mapping = self.fresh_vars_for_generics(generics);
            let mut trial = self.subst.clone();
            let all_matched = targets.iter().zip(query).all(|(target, q)| {
                let pattern_ty = self.ty_from_ast_mapped(target, &mapping);
                unify(&mut trial, &pattern_ty, q).is_ok()
            });
            if !all_matched {
                continue;
            }
            let bounds_satisfied = generics.iter().all(|g| match g {
                GenericParam::Type { name, bounds, .. } => {
                    let Some(arg_ty) = mapping.get(name) else { return true };
                    let resolved_arg = trial.apply(arg_ty);
                    bounds.iter().all(|bound| self.has_matching_impl(bound, std::slice::from_ref(&resolved_arg)))
                }
                GenericParam::Const { .. } => true,
            });
            if bounds_satisfied {
                out.push(trial);
            }
        }
        out
    }

    /// Non-committing existence probe: does `algebra` have an `impl` whose
    /// target pattern(s) unify against `query` — used both directly (as
    /// `has_matching_impl`'s own inner check) and via bound-satisfaction
    /// checks inside `matching_impls` itself. Never commits to `self.subst`
    /// — a match here is only ever "does *some* impl apply", not a
    /// commitment to *which* one, so permanently binding `self.subst` to
    /// whichever candidate `matching_impls` happens to return first would
    /// be arbitrary (see `dispatch_algebra_call`'s own doc comment for why
    /// a *committing* dispatch needs its own, more careful logic instead of
    /// just taking `matching_impls`'s first result the same way).
    ///
    /// For the overwhelmingly common single-target, fully-concrete case
    /// (`add(1, 2)` against `impl Ring<i32>`), `Registry::has_impl_named`'s
    /// plain `HashMap` lookup is tried first, before ever touching
    /// unification.
    fn match_impl(&mut self, algebra: &str, query: &[Ty]) -> bool {
        if let [ty] = query {
            if self.registry.has_impl_named(algebra, &ty.to_string()) {
                return true;
            }
        }
        !self.matching_impls(algebra, query).is_empty()
    }

    /// Non-committing existence probe: does *some* impl of `algebra` apply
    /// to the fully concrete `ty` — directly, or transitively through
    /// algebra-bound inheritance, checked in **both** directions along the
    /// bound graph (`algebra X : Y` never says which side ends up with the
    /// real `impl` — either can, and both are legitimate):
    ///
    /// - **Reverse witness**: some *more specific* algebra that itself
    ///   bounds on `algebra` has a real impl. `algebra Int<T> : Num { }`: an
    ///   `impl Int<i32>` alone, with no separate `impl Num<i32>` anywhere,
    ///   still satisfies a `Num` bound — exactly what `stdlib/num/num.cleave`
    ///   wants (see `Registry::algebra_bounds`'s own doc comment).
    /// - **Forward aggregate**: `algebra` itself bounds on one or more other
    ///   algebras, and *every one* of them is independently satisfied.
    ///   `algebra Semiring<T> : AdditiveMonoid + MultiplicativeMonoid { }`:
    ///   `impl AdditiveMonoid<i32>` and `impl MultiplicativeMonoid<i32>`
    ///   together satisfy a `Semiring` bound, with no separate (even empty)
    ///   `impl Semiring<i32>` needed. The mirror image of the reverse case —
    ///   there, the concrete impl sits on the more-specific side; here, it
    ///   sits on the more-general side(s) instead — found missing by direct
    ///   testing: only the reverse direction was originally built, since
    ///   `Int : Num` was the sole real motivating example at the time, and
    ///   `Semiring : AdditiveMonoid + MultiplicativeMonoid` (structurally
    ///   the *same* single-bound shape as `Int : Num`, once written with
    ///   only one bound — this was never actually about "+" or multiple
    ///   bounds at all) turned out to need the opposite walk.
    ///
    /// See `match_impl`'s own doc comment for the shared engine underneath
    /// the direct-impl check and exactly what non-committing means here.
    ///
    /// Deliberately *not* threaded through `match_impl`/`dispatch_algebra_
    /// call` itself — inheritance only makes sense for a marker/existence
    /// check like this one. Dispatching an actual operator call needs a
    /// *real* impl with a real fn body to run; you can't "borrow" `Ring::
    /// mul`'s implementation from some other algebra `Ring` merely bounds
    /// itself on (or aggregates from), the way a bare existence check can
    /// borrow `Num`'s.
    fn has_matching_impl(&mut self, algebra: &str, tys: &[Ty]) -> bool {
        self.has_matching_impl_inherited(algebra, tys, &mut HashSet::new())
    }

    fn has_matching_impl_inherited(&mut self, algebra: &str, tys: &[Ty], visited: &mut HashSet<String>) -> bool {
        // Guards against a cyclic bound declaration (`algebra A : B` and
        // `algebra B : A`) looping forever — a malformed program, but one
        // that must fail cleanly (no matching impl found) rather than
        // overflow the stack. Shared across *both* directions below, on
        // purpose: a cycle mixing reverse and forward hops (`A : B`, `B`
        // aggregating something that circles back to `A`) must terminate
        // just as cleanly as a same-direction one.
        if !visited.insert(algebra.to_string()) {
            return false;
        }
        if self.match_impl(algebra, tys) {
            return true;
        }
        let reverse_witness = self
            .registry
            .algebras_bounded_by(algebra)
            .collect::<Vec<_>>()
            .into_iter()
            .any(|other| self.has_matching_impl_inherited(other, tys, visited));
        if reverse_witness {
            return true;
        }
        // Forward aggregate — see this method's own doc comment. Gated on
        // *at least two* own bounds, not just "non-empty" — a real
        // regression, found by testing immediately after the naive
        // non-empty-only version landed: `Int<T> : Num` and `Float<T> :
        // Num` are *siblings* sharing one parent, each with a single bound.
        // Aggregating through a lone bound is exactly equivalent to asking
        // "is `Num` satisfied *by any means at all*" — which let `i8`
        // (a real `Int`) satisfy a `Float` query too, since `Num<i8>` holds
        // via `Int`, and a single-bound aggregate for `Float` couldn't tell
        // that apart from `Num<i8>` holding via `Float` itself. A single
        // bound is a rename/specialization relationship, already fully
        // covered by the reverse-witness direction above when checking the
        // *parent* — it never legitimately needs its own forward direction.
        // Two or more bounds is different in kind, not just degree: no
        // single sibling can satisfy a multi-ingredient conjunction by
        // accident the way one sibling can satisfy a shared single parent.
        let own_bounds = self.registry.algebra_bounds(algebra).to_vec();
        own_bounds.len() >= 2 && own_bounds.iter().all(|b| self.has_matching_impl_inherited(b, tys, visited))
    }

    /// Real, committing dispatch for a (possibly heterogeneous,
    /// possibly-multi-target) algebra call — checks `tys` together,
    /// coherently, against `algebra`'s own impls, and on success writes the
    /// winning match's bindings into `self.subst` for real. Committing is
    /// required specifically because a generic appearing only in an algebra
    /// fn's own *return* type (`C` in `fn mul(a: A, b: B) -> C;`, `To` in
    /// `Convert<From, To>`) is never independently constrained by a call's
    /// own arguments the way a parameter-appearing generic is — the only
    /// way to ever learn its concrete value is from a successful match's
    /// own bindings, and `infer_algebra_call`'s own `gating` never blocks
    /// dispatch on it being concrete first.
    ///
    /// `check_no_overlapping_impls` does *not*, on its own, make that safe
    /// in general — a real gap, found by direct testing while designing
    /// `Convert<From, To>`: it checks two *declarations'* own target
    /// patterns against each other, and two fully concrete, differently-
    /// shaped targets (`Convert<i32, f64>` vs. `Convert<i32, Complex<f64>>`)
    /// never unify against one another, so neither declaration is ever
    /// flagged as overlapping — yet both match a call site whose own `tys`
    /// still has `To` as a free `Ty::Var`. So this method doesn't just take
    /// `matching_impls`'s first result: every candidate's own resolution of
    /// `tys` is compared. If they all agree (including on whatever was
    /// free going in — the ordinary, overwhelmingly common case, and the
    /// only way more than one candidate can ever arise for `MatMul`-shaped
    /// algebras today), picking any one of them is equally correct, so the
    /// first is committed. If they genuinely disagree on how to resolve a
    /// still-free position, that's a real ambiguity — reported as
    /// `TypeErrorKind::AmbiguousDispatch` rather than silently committing
    /// to whichever candidate `Registry::all_impls` happened to iterate
    /// first. The caller can disambiguate with an explicit turbofish
    /// (`infer_algebra_call`'s own `explicit_generics` handling), which
    /// pins the free position *before* this is ever reached, collapsing
    /// `matching_impls` back down to a single candidate.
    fn dispatch_algebra_call(&mut self, algebra: &str, tys: &[Ty], span: Span) -> Result<bool, TypeError> {
        if let [ty] = tys {
            if self.registry.has_impl_named(algebra, &ty.to_string()) {
                return Ok(true);
            }
        }
        let matches = self.matching_impls(algebra, tys);
        if matches.is_empty() {
            return Ok(false);
        }
        if matches.len() > 1 {
            let resolved: Vec<Vec<Ty>> =
                matches.iter().map(|trial| tys.iter().map(|q| trial.apply(q)).collect()).collect();
            if resolved[1..].iter().any(|r| r != &resolved[0]) {
                let candidates = resolved
                    .iter()
                    .map(|r| format!("{algebra}<{}>", r.iter().map(Ty::to_string).collect::<Vec<_>>().join(", ")))
                    .collect();
                return Err(TypeError { span, kind: TypeErrorKind::AmbiguousDispatch { algebra: algebra.to_string(), candidates } });
            }
        }
        self.subst = matches.into_iter().next().unwrap();
        Ok(true)
    }

    /// Checks every constraint still pending against the registry, once —
    /// called at the end of `infer_fn`, after defaulting. A constraint whose
    /// type is *still* abstract at this point (never pinned to anything
    /// concrete, and not migrated into a `Scheme` along the way) is left
    /// unchecked: there's no cross-function propagation yet (see module
    /// docs) for it to travel further into, so it's permissive by omission,
    /// not by design — a real gap, not a silent decision to ignore it.
    ///
    /// A constraint whose type resolves to a variable in `self.quantified`
    /// is *also* skipped, deliberately: `generalize` now copies (doesn't
    /// remove) a matching constraint into every scheme that quantifies it
    /// (see `generalize`'s own doc comment for why moving it used to lose
    /// one when two schemes shared a variable) — the original, un-consumed
    /// entry left behind here is that same variable's constraint, still
    /// abstract and never touched again directly; it's genuinely owned by
    /// whichever scheme(s) carried it onward, each checked properly at its
    /// *own* future instantiation site, not here a second time.
    ///
    /// A constraint naming an algebra the registry has never heard of
    /// (`"Num"` when no stdlib was loaded, most tests today) is skipped
    /// outright, same reasoning as `infer_call`'s 0-candidates fallback:
    /// nothing to check against yet isn't the same as a check that failed.
    pub(crate) fn check_pending_constraints(&mut self) -> Result<(), TypeError> {
        for c in std::mem::take(&mut self.constraints) {
            if !self.registry.has_algebra(&c.algebra) {
                continue;
            }
            let resolved: Vec<Ty> = c.tys.iter().map(|t| self.subst.apply(t)).collect();
            if resolved.iter().any(|t| matches!(t, Ty::Var(v) if self.quantified.contains(v))) {
                continue;
            }
            if resolved.iter().any(is_placeholder) {
                continue;
            }
            // Only the *gating* (input-appearing) positions must already be
            // concrete before dispatch is even attempted — mirrors `infer_
            // algebra_call`'s own gating check for the immediate-dispatch
            // path exactly (`Constraint::gating_indices`'s own doc comment).
            // An output-only position (`Index<Container,Elem,K>`'s own
            // `Elem`, `MatMul<A,B,C>`'s own `C`) is *not* required concrete
            // here — it's the committing dispatch below that's supposed to
            // bind it, not a precondition for attempting one. Still abstract
            // somewhere among the *gating* positions — unlike the two
            // genuinely-dead-end cases just above, this one really can
            // still become ready later: e.g. `print(matmul(ma,mb)[0,0])`'s
            // own deferred `Print` constraint (`gating_indices` covering its
            // sole generic, tied to `mc[0,0]`'s own result) can't be
            // satisfied *within this same pass* — it needs `check_pending_
            // indices` to run first, which itself needs *this* method's own
            // earlier entry in the same queue (`MatMul`'s own deferred `C`)
            // already committed. Re-queued, not dropped, so `Infer::check_
            // pending_constraints_and_indices`'s own outer fixpoint gets
            // another chance at it once more of the picture resolves.
            if c.gating_indices.iter().any(|&i| !is_fully_concrete(&resolved[i])) {
                self.constraints.push(c);
                continue;
            }

            let all_concrete = resolved.iter().all(is_fully_concrete);
            if !all_concrete {
                // Gating is satisfied, but some *output-only* position is
                // still a bare `Ty::Var` — this can only be a real multi-
                // generic algebra-call constraint (an ordinary single-
                // target bound check, `Constraint::all_gating`, is always
                // fully gating by construction, so it always takes the
                // `all_concrete` branch below instead). `has_matching_impl`
                // (non-committing — only ever confirms *some* impl exists,
                // never binds anything) can't resolve this position; only a
                // real, committing dispatch can, exactly the same one
                // `infer_algebra_call`'s own immediate path already calls
                // once *its* gating is satisfied. This is the actual fix for
                // `doc/backlog.md`'s own "`check_pending_constraints`'s
                // output-only-generic gate" item — before this, such a
                // position was silently never bound, discarded forever along
                // with the rest of `self.constraints` at the top of this
                // loop's own `std::mem::take`.
                //
                // Same guard as `infer_algebra_call`'s own `generic_
                // context_pending` check, and for the identical reason:
                // `self.active_generics` stays set through `finish_fn`
                // (called right after body-checking, before it's ever
                // reset), so a constraint deferred *out of* `infer_algebra_
                // call` for still being inside a generic fn/impl body would
                // otherwise land right back here and commit anyway, one
                // phase later — the exact same premature-bind bug, not
                // actually fixed, just moved. Re-queued instead, same as
                // every other still-not-ready case in this loop — real
                // dispatch waits for `monomorphize.rs` to re-check this per
                // concrete instantiation instead. (`all_concrete` is already
                // known false here — the enclosing `if` — so this reduces to
                // just "are we inside a generic body at all".)
                if !self.active_generics.is_empty() {
                    self.constraints.push(c);
                    continue;
                }
                if !self.dispatch_algebra_call(&c.algebra, &resolved, c.span)? {
                    let ty = resolved.iter().map(Ty::to_string).collect::<Vec<_>>().join(", ");
                    return Err(TypeError { span: c.span, kind: TypeErrorKind::MissingImpl { algebra: c.algebra, ty } });
                }
                continue;
            }
            // A const generic referenced as an ordinary value (`for i in
            // 0..N`, say) can resolve here to a bare `Ty::Const` rather than
            // an ordinary `Ty::Con` — `has_matching_impl` has nothing to
            // check that against directly (impls are declared for real
            // types, never for one specific constant value). Bridge each
            // such element to its own const generic's declared width,
            // recovered from `self.subst`'s own const-width tracking via
            // `c.tys`' own *original*, pre-resolution var (see `Subst::
            // bind`'s own doc comment — threaded through `generalize`/`instantiate_with_
            // mapping` exactly like this constraint itself was, so it's
            // still there for a constraint re-checked at instantiation time,
            // not just at the original declaration site). Falls back to
            // this project's own conventional default representation when
            // untracked (an ordinary array-literal-length `Ty::Const`, e.g.
            // `[1,2,3]`'s own size, never tied to a declared const generic).
            // Checked *together*, one call — see `Constraint`'s own doc
            // comment for why a multi-generic algebra can't be verified any
            // other way.
            let checkable: Vec<Ty> = resolved
                .iter()
                .zip(&c.tys)
                .map(|(r, orig)| match r {
                    Ty::Const(cv) => {
                        let width = match orig {
                            Ty::Var(v) => self.subst.const_width(*v),
                            _ => None,
                        };
                        width.unwrap_or_else(|| match cv {
                            ConstValue::Int(_) => Ty::Con("i32".to_string()),
                            ConstValue::Bool(_) => Ty::Con("bool".to_string()),
                        })
                    }
                    _ => r.clone(),
                })
                .collect();
            // An `Int`/`Float`-shaped literal that ended up resolved to
            // `Complex<T>` (`4 + 2i` — see `ExprKind::ImaginaryLit`'s own
            // doc comment): satisfied directly, never consulting `has_
            // matching_impl` at all. Deliberately *not* expressed as an
            // algebra bound (`algebra Int<T> : Complex` would be backwards
            // — bound-inheritance answers "does concrete type X itself
            // satisfy algebra Y", never "can a literal of shape X widen
            // into a structurally different type Z"; `has_matching_impl`'s
            // own reverse-witness/forward-aggregate walk has no mechanism
            // for that second question at all). `Int`/`Float` stay mutually
            // exclusive with *each other* exactly as before — `1 + 2.0` is
            // untouched by this — this only ever fires when the resolved
            // type is genuinely `Complex<T>`, matching ℤ, ℝ ⊂ ℂ.
            let widened_to_complex =
                matches!(c.algebra.as_str(), "Int" | "Float") && matches!(&checkable[..], [Ty::App(name, _)] if name == "Complex");
            // No separate ambiguity check needed here the way `dispatch_
            // algebra_call` needs one: by this point `checkable` is already
            // fully concrete in *every* position (the `is_fully_concrete`
            // gate just above guarantees it, for every element, including
            // whatever was still an output-only generic's free var at the
            // original, deferred call site). Unifying an impl's own pattern
            // against an already-fully-concrete query can only ever bind
            // that pattern's own free vars *to* the query's own values —
            // `Subst::apply`-ing the (unchanged) query back through any
            // resulting trial always yields the query itself, identical
            // across every matching candidate. Ambiguity (two candidates
            // resolving a free position two different ways) is only even
            // possible while a position is still free — which, here,
            // structurally never happens: if `check_no_overlapping_impls`
            // already rejects two impls whose bound-satisfiable patterns
            // could coincide on one shared instantiation, no two impls can
            // ever *both* match one single fully concrete tuple in the
            // first place. `has_matching_impl` stays the right tool.
            if !widened_to_complex && !self.has_matching_impl(&c.algebra, &checkable) {
                let ty = checkable.iter().map(Ty::to_string).collect::<Vec<_>>().join(", ");
                return Err(TypeError { span: c.span, kind: TypeErrorKind::MissingImpl { algebra: c.algebra, ty } });
            }
        }
        Ok(())
    }

    /// Drains `pending_type_name_checks`, failing on the first entry — see
    /// that field's own doc comment for why this is deferred rather than an
    /// immediate error at the point `ty_from_ast_mapped` notices it. Called
    /// from `finish_fn` for the single-function paths (`infer_fn`,
    /// `infer_impl_fn_generic`); `callgraph.rs`'s whole-program pass calls
    /// `infer_fn_raw` directly instead (bypassing `finish_fn` — it
    /// reimplements an equivalent tail itself, since a *group's* defaulting/
    /// constraint-checking genuinely differs from a single function's, see
    /// that module's own doc comment) and must call this itself too — found
    /// missing by direct testing: `const R: Int` was rejected via
    /// `infer_fn`/`infer_impl_fn_generic` but sailed through silently for
    /// any ordinary top-level `fn`, which is every function `--dump-
    /// inference-pass` (and thus every real `.cleave` file) actually goes
    /// through. Unlike `check_pending_constraints` (checked once for a whole
    /// mutually-recursive group, deliberately — see `callgraph.rs`'s own
    /// comment on why a constraint failure there is the group's property,
    /// not one member's), each entry here is anchored to the exact call
    /// site that produced it and belongs to whichever *one* member was being
    /// inferred at the time — `callgraph.rs` accordingly drains this once
    /// per member, right after that member's own `infer_fn_raw` call, not
    /// once for the whole group.
    pub(crate) fn check_pending_type_names(&mut self) -> Result<(), TypeError> {
        match std::mem::take(&mut self.pending_type_name_checks).into_iter().next() {
            Some((name, span)) => Err(TypeError { span, kind: TypeErrorKind::TypeNameIsAnAlgebra { name } }),
            None => Ok(()),
        }
    }

    /// Drains `pending_div_by_zero_checks`, failing on the first entry —
    /// same deferred shape and same reasoning as `check_pending_type_names`
    /// right above (see that field's own doc comment); called from the
    /// identical sites for the identical reason.
    pub(crate) fn check_pending_div_by_zero(&mut self) -> Result<(), TypeError> {
        match std::mem::take(&mut self.pending_div_by_zero_checks).into_iter().next() {
            Some((dividend, span)) => Err(TypeError { span, kind: TypeErrorKind::ConstDivByZero { dividend } }),
            None => Ok(()),
        }
    }

    /// Drains `pending_field_accesses`, called from the same three sites as
    /// `check_pending_constraints`, right after `apply_defaults` — by then,
    /// any base whose only concreteness came from literal-defaulting has
    /// it; anything still a bare `Ty::Var` (or, transitively, a placeholder
    /// itself — e.g. chained off another entry that never resolved) never
    /// will. That case unifies `result` against the same
    /// `<not-yet-inferred>` placeholder the immediate path already returns
    /// for a genuinely-unresolvable base, so `check_no_placeholder`'s
    /// existing safety net still catches it downstream, unchanged. A
    /// resolved base runs the real lookup via `resolve_field_access`, the
    /// same helper `ExprKind::FieldAccess`'s own immediate path uses.
    pub(crate) fn check_pending_field_accesses(&mut self) -> Result<(), TypeError> {
        for pending in std::mem::take(&mut self.pending_field_accesses) {
            let resolved = self.subst.apply(&pending.base);
            let field_ty = if matches!(resolved, Ty::Var(_)) || is_placeholder(&resolved) {
                Ty::Con("<not-yet-inferred>".to_string())
            } else {
                self.resolve_field_access(&resolved, &pending.field, pending.span)?
            };
            self.unify_at(pending.span, &Ty::Var(pending.result), &field_ty)?;
        }
        Ok(())
    }

    /// The `MethodCall` counterpart to `check_pending_field_accesses` — see
    /// its own doc comment, and `pending_method_calls`'s, for the shared
    /// reasoning. `arg_tys` are passed through unresolved (as captured at
    /// the deferred call site): `unify`/`unify_at` already resolves both
    /// sides via `self.subst` internally, so there's no need to re-apply
    /// here first.
    pub(crate) fn check_pending_method_calls(&mut self) -> Result<(), TypeError> {
        for pending in std::mem::take(&mut self.pending_method_calls) {
            let resolved = self.subst.apply(&pending.base);
            let ret_ty = if matches!(resolved, Ty::Var(_)) || is_placeholder(&resolved) {
                Ty::Con("<not-yet-inferred>".to_string())
            } else {
                self.resolve_method_call(
                    &resolved,
                    &pending.method,
                    &pending.arg_tys,
                    &pending.arg_spans,
                    pending.base_span,
                    pending.call_span,
                )?
            };
            self.unify_at(pending.call_span, &Ty::Var(pending.result), &ret_ty)?;
        }
        Ok(())
    }

    /// Shared tail of `ExprKind::Index`'s own immediate path and `check_
    /// pending_indices`'s deferred one — extracted so the deferred path can
    /// reuse it without duplicating it, exactly like `resolve_field_access`'s
    /// own identical "extracted for the same reason" precedent. `base_ty`
    /// must already be concrete and not a placeholder (the caller's
    /// responsibility, same contract `resolve_field_access` already has);
    /// `index_tys`/`index_spans` are each already-inferred, already-`Int`-
    /// constrained.
    fn resolve_index(&mut self, base_ty: Ty, index_tys: &[Ty], index_spans: &[Span], base_span: Span, expr_span: Span) -> Result<Ty, TypeError> {
        match &base_ty {
            // A real array: peel one dimension per index, in order — the
            // direct generalization of what nested single-index `Index`
            // nodes used to achieve through recursion, now done in one
            // node/one loop instead (`a[i,j]` and the two-separate-brackets
            // `a[i][j]` stay equivalent for a real array either way — a
            // well-defined sub-array type exists at every step). Running out
            // of array dimensions with indices still remaining (`a[i,j,k]`
            // on a 2D array) is a direct `Mismatch` here — the same
            // rejection an over-indexed array always got, just reached
            // directly instead of indirectly through a failed `Index`-
            // algebra lookup on whatever scalar leaf type was left over.
            Ty::Array(..) => {
                let mut current = base_ty.clone();
                for span in index_spans {
                    current = match self.subst.apply(&current) {
                        Ty::Array(elem, _) => *elem,
                        Ty::Var(_) => return Ok(Ty::Con("<not-yet-inferred>".to_string())),
                        other if is_placeholder(&other) => return Ok(Ty::Con("<not-yet-inferred>".to_string())),
                        other => {
                            return Err(TypeError {
                                span: *span,
                                kind: TypeErrorKind::Unify(UnifyError::Mismatch(
                                    Ty::Array(Box::new(self.vars.fresh()), Box::new(self.vars.fresh())),
                                    other,
                                )),
                            });
                        }
                    };
                }
                Ok(self.subst.apply(&current))
            }
            // Concrete, resolved, and definitely not an array — not an
            // immediate error anymore: a `#[mlir_type(...)]`-tagged struct
            // (`Tensor<T, const Dims...: i32>`, `stdlib/linalg/tensor.
            // cleave`) has no array of its own to index into
            // directly, so `v[i]`/`m[i,j]` only makes sense through a real
            // declared `Index<Container, Elem, const K: i32>` impl —
            // dispatched exactly the way an ordinary bare-name operator call
            // already falls back through `infer_call`'s own `registry.
            // algebras_with_fn` lookup (mirrored here, not reused directly,
            // since there's no real `Expr` call node to hand `infer_call` —
            // `ExprKind::Index` stays its own dedicated AST shape, for the
            // mutability-checking `a[i] = x` needs, see
            // `check_mutability_expr`). The whole bracket group dispatches
            // as *one* call — every index unified against one shared fresh
            // `elem_ty` (mirrors `ExprKind::ArrayLit`'s own pairwise-unify
            // loop exactly, since this literally becomes a real `[i32;K]`
            // array value at CPS time, `cps.rs`'s own `ExprKind::Index` doc
            // comment) rather than dispatched index-by-index — unifying
            // that synthesized array type against the algebra's own
            // declared `[i32;K]` signature is what forces `elem_ty := i32`
            // and pins `K := index_tys.len()`, no special-casing needed
            // here at all. No candidate at all (the overwhelmingly common
            // case — indexing an `i32`, say) falls through to the same
            // `Mismatch` as before, unchanged.
            other => {
                let elem_ty = self.vars.fresh();
                for (index_ty, span) in index_tys.iter().zip(index_spans) {
                    self.unify_at(*span, &elem_ty, index_ty)?;
                }
                let idx_array_ty = Ty::Array(Box::new(elem_ty), Box::new(Ty::Const(ConstValue::Int(index_tys.len() as u64))));
                let candidates = self.registry.algebras_with_fn("index", 2);
                if candidates.len() > 1 {
                    return Err(TypeError {
                        span: expr_span,
                        kind: TypeErrorKind::AmbiguousOperator {
                            name: "index".to_string(),
                            candidates: candidates.into_iter().map(String::from).collect(),
                        },
                    });
                }
                if let Some(&algebra) = candidates.first() {
                    self.infer_algebra_call(expr_span, algebra, "index", &[other.clone(), idx_array_ty], &[base_span, expr_span], &[])
                } else {
                    Err(TypeError {
                        span: expr_span,
                        kind: TypeErrorKind::Unify(UnifyError::Mismatch(
                            Ty::Array(Box::new(self.vars.fresh()), Box::new(self.vars.fresh())),
                            other.clone(),
                        )),
                    })
                }
            }
        }
    }

    /// Drains `pending_indices` — see `Infer::pending_indices`'s own doc
    /// comment. Called (via `check_pending_constraints_and_indices`) right
    /// after `check_pending_constraints`, so a base type that only became
    /// concrete *because of* that method's own committing-dispatch fix
    /// (`mc`'s own `MatMul`-derived type, say) is already resolved by the
    /// time this runs. A base that's an *already*-unresolved placeholder is
    /// a genuine dead end (same reasoning as `PendingFieldAccess`'s own
    /// identical case) and resolves to the same dead `<not-yet-inferred>`
    /// placeholder the immediate path already returns — `check_no_
    /// placeholder`'s existing safety net still catches it downstream,
    /// unchanged. A base that's *still* a bare `Ty::Var` even now, though,
    /// is re-queued rather than given up on — it can still resolve in a
    /// later round of `check_pending_constraints_and_indices`'s own outer
    /// fixpoint (a chain of two-or-more deferred steps, e.g. `matmul(matmul
    /// (a,b),c)[0,0]`, needs more than one round to fully unwind).
    pub(crate) fn check_pending_indices(&mut self) -> Result<(), TypeError> {
        for pending in std::mem::take(&mut self.pending_indices) {
            let resolved = self.subst.apply(&pending.base);
            if matches!(resolved, Ty::Var(_)) {
                self.pending_indices.push(pending);
                continue;
            }
            let result_ty = if is_placeholder(&resolved) {
                Ty::Con("<not-yet-inferred>".to_string())
            } else {
                self.resolve_index(resolved, &pending.index_tys, &pending.index_spans, pending.base_span, pending.span)?
            };
            self.unify_at(pending.span, &Ty::Var(pending.result), &result_ty)?;
        }
        Ok(())
    }

    /// Runs `check_pending_constraints`/`check_pending_indices` to a
    /// fixpoint — needed together, in a loop, not just once each back-to-
    /// back: found directly while testing `print(matmul(ma,mb)[0,0])` with
    /// no intervening `let` at all. `check_pending_constraints`'s own single
    /// pass over `self.constraints` processes `MatMul`'s own deferred `C`
    /// *and* `Print`'s own deferred constraint (queued after it, sharing the
    /// same `Vec`) in one linear walk — but `Print`'s own gating depends on
    /// `mc[0,0]`'s own result, which only `check_pending_indices` resolves,
    /// and that method hasn't even run yet partway through `check_pending_
    /// constraints`'s own single pass. One call each, back-to-back, isn't
    /// enough — `Print`'s own entry needs a *second* look at `check_pending_
    /// constraints`, after `check_pending_indices` has had its own turn.
    /// Both methods now re-queue (not drop) whatever isn't ready yet
    /// specifically so a later round here can pick it back up. Terminates
    /// because each round either strictly shrinks the combined pending count
    /// (real progress) or leaves it unchanged — the latter genuinely stuck
    /// (a base nothing in this body ever pins down, e.g. `fn f(a) { a[0] }`
    /// with `a` never otherwise constrained): on that final, no-progress
    /// round, any leftover `pending_indices` entry is collapsed to the same
    /// dead `<not-yet-inferred>` placeholder `check_pending_field_accesses`'s
    /// own identical "give up" case already collapses to (`PendingIndex`'s
    /// own `result` was a real `Ty::Var`, not a placeholder, specifically so
    /// a *resolvable* chain could still be pinned down across rounds — but
    /// once no more progress is possible, leaving it a bare, permanently
    /// free variable would let it silently escape `check_no_placeholder`'s
    /// own downstream safety net instead of being caught by it, a real
    /// regression found directly by this project's own existing `indexing_
    /// an_unresolved_base_defers_as_not_yet_inferred` test). Leftover
    /// `self.constraints` entries need no equivalent final step — that's
    /// already `check_pending_constraints`'s own documented "permissive by
    /// omission" posture for a bound nothing ever pins down, unchanged.
    pub(crate) fn check_pending_constraints_and_indices(&mut self) -> Result<(), TypeError> {
        loop {
            let before = self.constraints.len() + self.pending_indices.len();
            self.check_pending_constraints()?;
            self.check_pending_indices()?;
            let after = self.constraints.len() + self.pending_indices.len();
            if after == 0 {
                return Ok(());
            }
            if after >= before {
                for pending in std::mem::take(&mut self.pending_indices) {
                    self.unify_at(pending.span, &Ty::Var(pending.result), &Ty::Con("<not-yet-inferred>".to_string()))?;
                }
                return Ok(());
            }
        }
    }

    /// Exposed `pub(crate)` (unlike `ty_from_ast_mapped` below) specifically
    /// so `callgraph.rs` can resolve an `extern fn`'s declared return type
    /// directly, the same way it already reuses `fresh_fn_shape` for that
    /// case's param types — no generics are ever in scope for an `extern
    /// fn` (rejected outright, see `TypeErrorKind::ExternFnCannotBeGeneric`),
    /// so the unmapped form is exactly what's needed there.
    pub(crate) fn ty_from_ast(&mut self, ty: &Type) -> Ty {
        self.ty_from_ast_mapped(ty, &HashMap::new())
    }

    /// Like `ty_from_ast`, but a bare path matching a key in `mapping`
    /// resolves to that (fresh, per-call-site) type variable instead of a
    /// literal `Ty::Con("T")` — used to instantiate an algebra's own generic
    /// parameter (`T` in `algebra Ring<T> { fn add(a: T, b: T) -> T; }`) with
    /// a fresh variable per call, rather than treating `T` as if it were a
    /// concrete type spelled "T".
    ///
    /// `pub(crate)` (unlike most of `Infer`'s own internals) — `monomorphize.
    /// rs`'s own preemptive-monomorphization pass (`doc/backlog.md`'s own
    /// "Toward a matmul-based tensorial XOR"'s "Bug 3" entry) needs exactly
    /// this: resolving a `derivative`-rule-referenced method's own declared
    /// parameter/return types (e.g. `Ring<T>::add`'s own `T`) against a
    /// concrete, already-fully-resolved substitution, with `mapping` this
    /// time holding real concrete `Ty`s rather than fresh vars.
    pub(crate) fn ty_from_ast_mapped(&mut self, ty: &Type, mapping: &HashMap<String, Ty>) -> Ty {
        match &ty.kind {
            TypeKind::Path(p, args) => {
                let name = p.segments.join("::");
                // A bare path matching a key in `mapping` resolves to that
                // variable regardless of whether it *also* carries its own
                // generic arguments — a mapped name is always a generic
                // parameter's own bare reference (`T`, never `T<U>`; nothing
                // in this language lets a type parameter itself be generic),
                // so `args` is always empty whenever this branch applies.
                if let Some(mapped) = mapping.get(&name) {
                    return mapped.clone();
                }
                // A bare name that resolves to a declared `algebra` (and
                // isn't *also* a `struct` of the same name — legal, if
                // confusing) is a real, common category error: an algebra
                // constrains which types are legal, it isn't itself one
                // (`Int` vs. `i32`/`i64`). Queued rather than an immediate
                // error — see `pending_type_name_checks`'s own doc comment
                // for why (dozens of infallible call sites, plus
                // `has_matching_impl`'s speculative reuse of this method).
                if self.registry.has_algebra(&name) && !self.registry.has_struct(&name) {
                    self.pending_type_name_checks.push((name.clone(), ty.span));
                }
                // Every argument — type *and* const — becomes one `Ty` slot
                // in `App`'s list, positionally, matching `Matrix<f64, 3,
                // 3>`'s own source order exactly; there's no more filtering
                // of const args out (see `Ty::App`'s own doc comment for why
                // that used to be deferred, and why const-generics now go
                // through the same `Ty::Var`/`Ty::Const` machinery arrays
                // already use). A const arg that isn't a bare integer
                // literal or a mapped const-generic name (an arbitrary
                // computed expression — not evaluated) becomes a fresh,
                // unconstrained var rather than poisoning the whole `App`
                // into a placeholder — permissive by omission, same posture
                // as everywhere else "not evaluated yet" shows up.
                let type_args: Vec<Ty> = args
                    .iter()
                    .map(|a| match a {
                        GenericArg::Type(t) => self.ty_from_ast_mapped(t, mapping),
                        GenericArg::Const(e) => {
                            self.const_value_from_expr(e, mapping).unwrap_or_else(|| self.vars.fresh())
                        }
                    })
                    .collect();
                if type_args.is_empty() {
                    Ty::Con(name)
                } else {
                    Ty::App(name, type_args)
                }
            }
            TypeKind::Array(elem, size) => {
                let elem = self.ty_from_ast_mapped(elem, mapping);
                match self.const_value_from_expr(size, mapping) {
                    // An array's own size demands an integer — but this is
                    // a *value*, not a type, so `Int` (an algebra over
                    // types — i8/i32/...) has nothing to check here; there
                    // is no `impl Int<3>` and there never would be (found by
                    // testing: pushing that constraint here rejected every
                    // valid `[i32; 3]` with a nonsensical `no impl Int<3>`).
                    // A structural check instead: an outright bool literal
                    // (`[T; true]`) is rejected on sight; a still-unresolved
                    // `Var` (a const-generic reference whose own value isn't
                    // known yet) is deferred, same "we don't know yet, not a
                    // failure" posture as everywhere else in this file —
                    // *if* that var later resolves to a `Bool`, nothing
                    // catches it today (see `ConstValue`'s own doc comment:
                    // a const-generic's declared type isn't tracked past its
                    // own declaration site), a known, flagged gap rather
                    // than a silent one.
                    Some(size_ty) if !matches!(size_ty, Ty::Const(ConstValue::Bool(_))) => {
                        Ty::Array(Box::new(elem), Box::new(size_ty))
                    }
                    // Also covers `const_value_from_expr` returning `None`
                    // outright — an operator `const_eval` doesn't know yet
                    // (`[T; N-1]`, today), or one it does but an operand is
                    // still abstract (`[T; N+M]` before either is concrete);
                    // `is_placeholder` keeps this out of every registry check.
                    _ => Ty::Con("<array-type-not-yet-inferred>".to_string()),
                }
            }
            TypeKind::Fn(params, ret) => Ty::Fn(
                params.iter().map(|p| self.ty_from_ast_mapped(p, mapping)).collect(),
                Box::new(self.ty_from_ast_mapped(ret, mapping)),
            ),
            // `Args...` used as a whole type (`doc/backlog.md`'s own
            // "Variadic generics" item) -- an ordinary bare-name lookup
            // against `mapping`, exactly like a non-variadic generic
            // reference just above (`TypeKind::Path`'s own identical
            // lookup) -- `fresh_vars_for_generics` is what actually makes
            // this resolve to a real `Ty::Pack`/`Ty::PackResolved` rather
            // than an ordinary `Ty::Var`/concrete type, not this site; this
            // is just the read. A name genuinely missing from `mapping`
            // (referencing a pack that was never declared, or a typo) is a
            // real bug either way -- panics with the same clarity the old,
            // unconditional version did.
            TypeKind::PackRef(name) => mapping
                .get(name)
                .cloned()
                .unwrap_or_else(|| panic!("type inference: pack reference `{name}...` has no matching declared generic in scope")),
        }
    }

    /// Recognizes `<pack-name>.len()` — a bare zero-arg method call whose
    /// receiver is a single-segment path naming an in-scope *pack* generic
    /// (`mapping.get(name)` resolving to `Ty::Pack`/`Ty::PackResolved`, not
    /// an ordinary type/const) — and returns its own const-generic length,
    /// symbolic (`Ty::PackLen`) or already-resolved (`Ty::Const`) as
    /// appropriate. Deliberately *not* a new grammar rule or AST node: `.len()`
    /// parses as an ordinary `ExprKind::MethodCall` already, ambiguity-free
    /// with a genuine method call since no non-pack generic name is ever a
    /// legal receiver for one at this same syntactic position (an ordinary
    /// value never appears in an array-dimension slot, and — in expression
    /// position — a real variable named the same as a pack generic can't
    /// coexist, generics and `let`-bindings share no namespace). `None` for
    /// anything else (a genuine method call, or `.len()` on something that
    /// isn't a pack at all) — the caller falls back to its own ordinary
    /// handling unchanged.
    fn pack_len_from_method_call(
        &self,
        mapping: &HashMap<String, Ty>,
        base: &Expr,
        method: &str,
        args: &[Expr],
    ) -> Option<Ty> {
        if method != "len" || !args.is_empty() {
            return None;
        }
        let ExprKind::Path(p) = &base.kind else { return None };
        if p.segments.len() != 1 {
            return None;
        }
        match self.subst.apply(mapping.get(&p.segments[0])?) {
            Ty::Pack(v) => Some(Ty::PackLen(v)),
            Ty::PackResolved(elems) => Some(Ty::Const(ConstValue::Int(elems.len() as u64))),
            _ => None,
        }
    }

    /// Resolves a const-value expression — an array type's size (`[f64;
    /// 4]`'s `4`, or `[T; N]`'s `N`) or a `<...>`-position const-generic
    /// argument (`Matrix<f64, 3, 3>`'s `3`, or a bool-valued one like
    /// `Grid<true>`'s `true`) alike, both the same shape of problem — to a
    /// `Ty`. A bare integer or bool literal becomes a resolved `Ty::Const`,
    /// a single-segment path matching a key in `mapping` becomes that
    /// const-generic's own (fresh, per-call-site) variable, resolved through
    /// `self.subst` first — a real bug, found by direct testing: an explicit
    /// turbofish (`Buf::<2, 3>(...)`) already unifies `mapping`'s own fresh
    /// vars against concrete values *earlier* in the same `StructLit`
    /// inference, but `mapping` itself is never mutated by that (only
    /// `self.subst` is), so a bare `mapping.get(name)` used to see the
    /// original, still-unresolved var and never noticed. An operator call
    /// (`4+3`, `N*2`, ...) — the same desugared shape ordinary `+`/`*`
    /// already produce, see `ast.rs`'s own `Call` doc comment — recurses on
    /// both operands and, if *both* resolve to a `Ty::Const`, delegates the
    /// actual arithmetic to the standalone `const_eval` module (deliberately
    /// isolated there — see its own module doc comment — rather than folded
    /// in here, so it stays reusable wherever else constant folding is
    /// needed later); an operator `const_eval` doesn't know yet, given two
    /// already-concrete operands, is `None`, same as before this comment was
    /// written. If either operand *isn't* concrete yet (an unresolved const
    /// generic — `doc/backlog.md`'s own "Deferred/symbolic constant
    /// folding" item), builds a real `Ty::ConstExpr` instead of giving up —
    /// stays symbolic through a still-generic declaration, exactly like a
    /// bare `Ty::Var` already does, folded later by `Subst::apply`/
    /// `substitute` once real values arrive. Deliberately *not* integer-
    /// only: whether the result actually needs to be an integer is up to the
    /// caller (`TypeKind::Array`'s own arm above pushes that constraint
    /// itself, since it's the one actual consumer that cares).
    fn const_value_from_expr(&mut self, value: &Expr, mapping: &HashMap<String, Ty>) -> Option<Ty> {
        match &value.kind {
            ExprKind::NumberLit { text, .. } => text.parse::<u64>().ok().map(|n| Ty::Const(ConstValue::Int(n))),
            ExprKind::BoolLit(b) => Some(Ty::Const(ConstValue::Bool(*b))),
            ExprKind::Path(p) if p.segments.len() == 1 => mapping.get(&p.segments[0]).map(|t| self.subst.apply(t)),
            // `Dims...` in an array-dimension position (`[T; Dims...]`,
            // `doc/backlog.md`'s own "Variadic generics" item) — the exact
            // same bare-name-against-`mapping` lookup the `Path` arm just
            // above already does for a non-pack const-generic reference;
            // `fresh_vars_for_generics` is what makes this resolve to a
            // real pack rather than an ordinary value, not this site.
            ExprKind::PackRef(name) => mapping.get(name).map(|t| self.subst.apply(t)),
            // `Dims.len()` in an array-dimension position (`[i32;
            // Dims.len()]`, `doc/backlog.md`'s own "Variadic generics" item)
            // — see `pack_len_from_method_call`'s own doc comment.
            ExprKind::MethodCall(base, method, args) => self.pack_len_from_method_call(mapping, base, method, args),
            ExprKind::Call(path, _, args, ..) if path.segments.len() == 1 && args.len() == 2 => {
                let a = self.const_value_from_expr(&args[0], mapping)?;
                let b = self.const_value_from_expr(&args[1], mapping)?;
                if let (Ty::Const(av), Ty::Const(bv)) = (&a, &b) {
                    // The *only* way `eval_binop("div", ...)` returns `None`
                    // for two already-concrete operands is a zero divisor
                    // (see `pending_div_by_zero_checks`'s own doc comment) —
                    // queue a real, located error rather than letting the
                    // fallback below build a `ConstExpr` that can never
                    // resolve further.
                    if path.segments[0] == "div" {
                        if let (ConstValue::Int(dividend), ConstValue::Int(0)) = (*av, *bv) {
                            self.pending_div_by_zero_checks.push((dividend, value.span));
                        }
                    }
                    return const_eval::eval_binop(&path.segments[0], *av, *bv).map(Ty::Const);
                }
                Some(Ty::ConstExpr(path.segments[0].clone(), Box::new(a), Box::new(b)))
            }
            _ => None,
        }
    }

    /// Resolves one explicit turbofish argument (`Matrix::<f64, 4, 4>`'s
    /// `f64`/`4`, `fibonacci::<f64>`'s `f64`) to a `Ty` — against `self.
    /// active_generics`, so a turbofish argument naming one of the
    /// *enclosing* fn/impl's own generic parameters by name (`fn g<U>() {
    /// f::<U>(x) }`, or `Tensor::<T,N>(...)` inside `impl<T,const N:i32>
    /// Zeroed<Tensor<T,N>>`'s own body) resolves to that enclosing generic,
    /// not a bogus literal type named "T"/const named "N" — a real,
    /// previously-documented gap (this comment used to describe it as
    /// unfixed), found blocking exactly the second case directly: `no impl
    /// Float<T>` from treating a turbofish `T` as a literal type name
    /// instead of the impl's own `T: Float`. Safe for every other caller
    /// too — `active_generics` is empty whenever none of this matters (a
    /// top-level, non-generic call site), so the overwhelmingly common case
    /// (a concrete type/const, `f64`/`4`/`true`) is unaffected either way.
    fn generic_arg_to_ty(&mut self, g: &GenericArg) -> Ty {
        match g {
            GenericArg::Type(t) => self.ty_from_ast_mapped(t, &self.active_generics.clone()),
            GenericArg::Const(e) => {
                self.const_value_from_expr(e, &self.active_generics.clone()).unwrap_or_else(|| self.vars.fresh())
            }
        }
    }

    /// Takes `env` by shared reference and clones it locally — a block (or
    /// an `if`-branch, which is just a block) introduces its own scope that
    /// must not leak `let` bindings back out to whatever comes after it in
    /// the enclosing scope. `Env` is cheap enough at this scale that cloning
    /// per nested scope is the simplest correct thing to do; a real
    /// scope-chain isn't worth it until this ever shows up in a profile.
    fn infer_block(&mut self, env: &Env, block: &Block) -> Result<Ty, TypeError> {
        let mut env = env.clone();
        for stmt in &block.stmts {
            match &stmt.kind {
                StmtKind::Let { mutable, name, ty, value } => {
                    // A lambda's own body may reference `name` itself
                    // (self-recursion, `let g = fn(n) { ... g(n) ... };`) —
                    // real bug, found by direct testing (`error: type
                    // mismatch: expected \`i32\`, found \`<unresolved-call:
                    // fact>\``): `name` is otherwise only inserted into
                    // `env` *after* the value has already been fully
                    // inferred, so a self-call inside the body fell through
                    // to the same "unresolved call" placeholder an
                    // undeclared name would. Fixed the same way `infer_fn_
                    // raw` already seeds a top-level `fn`'s own self-
                    // reference: a monomorphic placeholder (classical ML
                    // `let rec` restriction — never generalized here, even
                    // though `name`'s own outer binding may be, just below),
                    // scoped to only this one `let`'s own seeded env clone
                    // so it can't leak into a sibling statement or shadow
                    // anything beyond this lambda's own body. Harmless when
                    // the lambda never actually references itself — the
                    // placeholder simply goes unused.
                    let value_ty = if let ExprKind::Lambda { params, .. } = &value.kind {
                        let mut seeded_env = env.clone();
                        let param_vars: Vec<Ty> = params.iter().map(|_| self.vars.fresh()).collect();
                        let ret_var = self.vars.fresh();
                        let self_ty = Ty::Fn(param_vars, Box::new(ret_var));
                        seeded_env.insert(name.clone(), Scheme::mono(self_ty.clone()));
                        let inferred = self.infer_expr(&seeded_env, value)?;
                        // Tie the placeholder back to what the body actually
                        // computed — not automatic: if a recursive call's
                        // own result is simply discarded (`let g = fn(n) {
                        // g(n - 1); n };`), nothing else would ever connect
                        // `self_ty`'s own `ret_var` to the body's real
                        // result, silently leaving that one call site's own
                        // recorded node type an unresolved leftover variable
                        // — mirrors `infer_fn_raw`'s own identical fix for a
                        // top-level `fn`'s self-reference, for the identical
                        // reason.
                        self.unify_at(value.span, &self_ty, &inferred)?;
                        inferred
                    } else {
                        self.infer_expr(&env, value)?
                    };
                    if let Some(annotated) = ty {
                        // `self.active_generics`, not `ty_from_ast`'s always-
                        // empty map — a `let`'s own annotation may reference
                        // the enclosing `fn`/impl-method's own generic
                        // parameter — see `active_generics`'s own doc
                        // comment for the real bug this closes.
                        let declared = self.ty_from_ast_mapped(annotated, &self.active_generics.clone());
                        self.unify_at(annotated.span, &declared, &value_ty)?;
                    }
                    // `let mut` is never generalized — see module docs (the
                    // ref-cell-polymorphism unsoundness this avoids).
                    let scheme = if !mutable && is_syntactic_value(value) {
                        self.generalize(&env, &value_ty)?
                    } else {
                        Scheme::mono(value_ty)
                    };
                    if matches!(value.kind, ExprKind::Lambda { .. }) {
                        self.lambda_schemes.insert(value.id, scheme.clone());
                    }
                    env.insert(name.clone(), scheme);
                }
                StmtKind::Assign { target, value } => {
                    // `target` is a plain name, or a field/index chain into
                    // one — ordinary `infer_expr` handles all three shapes
                    // uniformly (`Path`, `FieldAccess`, `Index` already have
                    // their own inference arms; `Index`'s already extracts
                    // the correct array element type). For the plain-name
                    // case this is exactly what the old direct `env.get`
                    // did: the scheme is always trivial (never-generalized —
                    // `mut` bindings are excluded from generalization above),
                    // so `instantiate`-ing it is a no-op equivalent to
                    // reading `.ty` directly.
                    let target_ty = self.infer_expr(&env, target)?;
                    // Mutation stays exclusively for a real array — unlike
                    // an ordinary read, `ExprKind::Index`'s own `Index`-
                    // algebra fallback (see its doc comment) must *not*
                    // apply to an assignment *target*: `Store` is a real
                    // effect on a stable reference, but a `#[mlir_type(
                    // ...)]`-tagged struct's real representation is an
                    // immutable SSA value, nothing to mutate in place —
                    // `cps.rs`'s own assignment-target conversion is (and
                    // stays) hardcoded to a real array's `PrimOp::Store`,
                    // so this must reject cleanly here rather than let a
                    // tagged-struct target silently reach it. `base`'s own
                    // type is already resolved (`infer_expr(target)` just
                    // above walked into it) — re-read rather than re-infer.
                    if let ExprKind::Index(base, _) = &target.kind {
                        let base_ty = self.subst.apply(&self.node_types[&base.id].clone());
                        if !matches!(base_ty, Ty::Array(..) | Ty::Var(_)) && !is_placeholder(&base_ty) {
                            return Err(TypeError {
                                span: target.span,
                                kind: TypeErrorKind::Unify(UnifyError::Mismatch(
                                    Ty::Array(Box::new(self.vars.fresh()), Box::new(self.vars.fresh())),
                                    base_ty,
                                )),
                            });
                        }
                    }
                    let value_ty = self.infer_expr(&env, value)?;
                    self.unify_at(value.span, &target_ty, &value_ty)?;
                }
                StmtKind::Expr(e) => {
                    self.infer_expr(&env, e)?;
                }
                StmtKind::Break(value) => {
                    let Some(accumulator) = self.loop_stack.last().cloned() else {
                        return Err(TypeError { span: stmt.span, kind: TypeErrorKind::BreakOutsideLoop });
                    };
                    // A bare `break;` is `break ();` for typing purposes —
                    // legal in *any* loop kind, since `While`/`For`/`ForIn`
                    // pin their own accumulator to `Ty::Con("()")` on push
                    // (see `loop_stack`'s own doc comment): a `break value;`
                    // whose value isn't `()` naturally, mechanically fails to
                    // unify against it — an ordinary `Unify` mismatch, not a
                    // bespoke diagnostic, the same way `if`/`else` branch
                    // mismatches already work.
                    match value {
                        Some(v) => {
                            let value_ty = self.infer_expr(&env, v)?;
                            self.unify_at(v.span, &accumulator, &value_ty)?;
                        }
                        None => {
                            self.unify_at(stmt.span, &accumulator, &Ty::Con("()".to_string()))?;
                        }
                    }
                }
            }
        }
        match &block.tail {
            Some(tail) => self.infer_expr(&env, tail),
            None => Ok(Ty::Con("()".to_string())),
        }
    }

    /// Records every expression's type against its `NodeId` (`node_types`)
    /// before returning it — the recursive calls inside `infer_expr_kind`
    /// call back into *this* wrapper for their sub-expressions, so every
    /// node in the tree gets recorded, not just top-level ones.
    fn infer_expr(&mut self, env: &Env, expr: &Expr) -> Result<Ty, TypeError> {
        let ty = self.infer_expr_kind(env, expr)?;
        self.node_types.insert(expr.id, ty.clone());
        Ok(ty)
    }

    fn infer_expr_kind(&mut self, env: &Env, expr: &Expr) -> Result<Ty, TypeError> {
        match &expr.kind {
            ExprKind::NumberLit { suffix, text } => match suffix {
                Some(s) => Ok(Ty::Con(s.clone())),
                None => {
                    let v = self.vars.fresh();
                    if let Ty::Var(id) = v {
                        let is_float = text.contains('.') || text.contains('e') || text.contains('E');
                        let default = if is_float { NumberDefault::Float } else { NumberDefault::Int };
                        self.pending_defaults.push((id, default));
                        // "Num" *and* "Int"/"Float", explicitly, even though
                        // `algebra Int<T> : Num`/`Float<T> : Num` (see
                        // `stdlib/num/num.cleave`) now makes the `Num` half
                        // derivable from either one alone (`has_matching_
                        // impl`'s own bound-inheritance walk) — kept explicit
                        // anyway so this still works unchanged against *any*
                        // stdlib a program might load, not just one that
                        // happens to declare that particular bound. The
                        // shape-specific constraint is what makes a literal's
                        // `.` a real, checkable requirement rather than a
                        // defaulting-only hint that any later, unrelated
                        // unification could silently override: two literals
                        // of different shapes forced to the same type by a
                        // shared generic call (`add(1, 2.0)`,
                        // `add(fibonacci(1), fibonacci(2.0))`) or pinned by a
                        // declared type that doesn't match either shape both
                        // get caught by `check_pending_constraints` exactly
                        // like any other constraint — no special-casing
                        // needed in `apply_defaults` for it. Checked against
                        // the registry once concrete (or skipped entirely if
                        // no stdlib declaring these was ever loaded — see
                        // `check_pending_constraints`), so this is additive,
                        // not a behavior change for a program that never
                        // `use`s `num`.
                        self.constraints.push(Constraint::all_gating("Num".to_string(), vec![v.clone()], expr.span));
                        let shape_algebra = if is_float { "Float" } else { "Int" };
                        self.constraints.push(Constraint::all_gating(shape_algebra.to_string(), vec![v.clone()], expr.span));
                    }
                    Ok(v)
                }
            },
            // `doc/backlog.md`'s own "Complex literals" item — same shape as
            // `NumberLit`'s own unsuffixed-literal handling just above: a
            // fresh var, a `Num` constraint, a `Complex` shape constraint
            // (`stdlib/complex/complex.cleave`'s own marker algebra — same
            // "additive, no-op if no stdlib declares it" posture `Int`/
            // `Float` already have), and a `pending_defaults` fallback to
            // `Complex<f32>`, matching `Float`'s own default width.
            // `suffix` is always `None` here today (`lower.
            // rs` never constructs one — `grammar.pest`'s own `imaginary_lit`
            // rule has no suffix syntax at all, unlike `numeric_lit`'s own
            // `type_suffix?`) — not handled, matching what's actually
            // reachable.
            ExprKind::ImaginaryLit { .. } => {
                let v = self.vars.fresh();
                if let Ty::Var(id) = v {
                    self.pending_defaults.push((id, NumberDefault::Complex));
                    self.constraints.push(Constraint::all_gating("Num".to_string(), vec![v.clone()], expr.span));
                    self.constraints.push(Constraint::all_gating("Complex".to_string(), vec![v.clone()], expr.span));
                }
                Ok(v)
            }
            ExprKind::BoolLit(_) => Ok(Ty::Con("bool".to_string())),
            ExprKind::Path(p) => {
                let name = p.segments.join("::");
                let scheme = env
                    .get(&name)
                    .cloned()
                    .ok_or(TypeError { span: expr.span, kind: TypeErrorKind::UnknownName(name) })?;
                Ok(self.instantiate(&scheme))
            }
            // A reserved raw-MLIR-op call (`mlir::arith::addi(a, b)`) --
            // skips algebra/top-level-fn resolution entirely: type-check
            // each positional arg normally (needed so their own types are
            // known and their CPS conversion is ordinary), and return a
            // *fresh unification variable* as this call's own result type,
            // pinned down normally by whatever context needs it (the
            // enclosing fn's declared return type, a `let`'s own use, ...) --
            // deliberately not the `<unresolved-call:...>` placeholder
            // mechanism (`is_placeholder` above), which exists to *error* if
            // nothing else resolves it; this is meant to resolve the same
            // unremarkable way any other ordinarily-inferred expression
            // does. Operand types are never cross-checked against each
            // other or against the op's own real requirements here -- that
            // safety net is MLIR's own verifier, not this pass. A deliberate
            // trade-off, not an oversight: see `mlir_lower.rs`'s own module
            // doc comment.
            ExprKind::Call(path, _, args, _) if path.segments.first().map(String::as_str) == Some("mlir") => {
                for arg in args {
                    self.infer_expr(env, arg)?;
                }
                Ok(self.vars.fresh())
            }
            ExprKind::Call(path, generics, args, _) => self.infer_call(env, expr.span, path, generics, args),
            ExprKind::Block(b) => self.infer_block(env, b),
            ExprKind::If { cond, then_branch, else_branch } => {
                let cond_ty = self.infer_expr(env, cond)?;
                self.unify_at(cond.span, &Ty::Con("bool".to_string()), &cond_ty)?;
                let then_ty = self.infer_block(env, then_branch)?;
                match else_branch {
                    Some(eb) => {
                        let else_ty = match &**eb {
                            ElseBranch::If(e) => self.infer_expr(env, e)?,
                            ElseBranch::Block(b) => self.infer_block(env, b)?,
                        };
                        self.unify_at(expr.span, &then_ty, &else_ty)?;
                        Ok(then_ty)
                    }
                    // No `else` — both branches must agree with `()`, matching
                    // ordinary "if as expression" semantics (see `grammar.md`).
                    None => {
                        self.unify_at(expr.span, &Ty::Con("()".to_string()), &then_ty)?;
                        Ok(Ty::Con("()".to_string()))
                    }
                }
            }
            // A `while`/`for`/`for-in` loop's own value is always `()`,
            // unconditionally — same reasoning as an `if` with no `else`
            // (see that arm's own comment just above): the body might run
            // zero times, or its per-iteration result is discarded either
            // way, and a `break value;` inside one is rejected (an ordinary
            // `Unify` mismatch against the pinned `()` accumulator pushed
            // here) — only `ExprKind::Loop`, below, can produce a real
            // value. `loop_stack` is pushed/popped around the body's own
            // inference regardless of whether it actually contains a
            // `break` — cheap, and lets a `break` inside a *nested* loop
            // correctly see its own nearest enclosing entry.
            ExprKind::While { cond, body } => {
                let cond_ty = self.infer_expr(env, cond)?;
                self.unify_at(cond.span, &Ty::Con("bool".to_string()), &cond_ty)?;
                self.loop_stack.push(Ty::Con("()".to_string()));
                let result = self.infer_block(env, body);
                self.loop_stack.pop();
                result?;
                Ok(Ty::Con("()".to_string()))
            }
            ExprKind::For { var, start, end, body } => {
                let start_ty = self.infer_expr(env, start)?;
                let end_ty = self.infer_expr(env, end)?;
                self.unify_at(end.span, &start_ty, &end_ty)?;
                // The loop variable's own type — not necessarily `i32`
                // specifically (no hardcoded width, same "Int, unconstrained
                // width" posture `ExprKind::Index`'s own bound already
                // uses), just some real `Int`-impl'd type.
                self.constraints.push(Constraint::all_gating("Int".to_string(), vec![start_ty.clone()], start.span));
                let mut inner_env = env.clone();
                inner_env.insert(var.clone(), Scheme::mono(start_ty));
                // `infer_block` clones `inner_env` again internally — the
                // same cheap-clone tradeoff `Lambda`'s own handling above
                // already makes.
                self.loop_stack.push(Ty::Con("()".to_string()));
                let result = self.infer_block(&inner_env, body);
                self.loop_stack.pop();
                result?;
                Ok(Ty::Con("()".to_string()))
            }
            // `iter` must be a real, homogeneous array — unifying against a
            // fresh `Ty::Array(elem, size)` directly (not routed through
            // `resolve_index`'s own dual-path array/algebra-dispatch
            // machinery, which exists for `arr[i]`'s own `Index<...>`-trait
            // fallback — irrelevant here, this is specifically about a real
            // array) gets the ordinary `Mismatch` diagnostic for a non-array
            // `iter` for free, no new `TypeErrorKind` needed.
            ExprKind::ForIn { var, iter, body } => {
                let iter_ty = self.infer_expr(env, iter)?;
                let elem_ty = self.vars.fresh();
                let size_ty = self.vars.fresh();
                self.unify_at(iter.span, &Ty::Array(Box::new(elem_ty.clone()), Box::new(size_ty)), &iter_ty)?;
                let mut inner_env = env.clone();
                inner_env.insert(var.clone(), Scheme::mono(elem_ty));
                self.loop_stack.push(Ty::Con("()".to_string()));
                let result = self.infer_block(&inner_env, body);
                self.loop_stack.pop();
                result?;
                Ok(Ty::Con("()".to_string()))
            }
            // `loop { ... break value; ... }` — unconditional, the only loop
            // kind that can produce a real value: a fresh accumulator var,
            // unified against by every `break` directly inside it (see
            // `loop_stack`'s own doc comment), read back here exactly like a
            // `let`'s own scheme is read back after `generalize`. A `loop`
            // with no `break` anywhere inside it simply never gets its
            // accumulator unified against anything — stays a bare, permissive
            // `Ty::Var`, the same "under-determined stays permissive" posture
            // an empty array literal's own element type already has.
            ExprKind::Loop { body } => {
                let accumulator = self.vars.fresh();
                self.loop_stack.push(accumulator.clone());
                let result = self.infer_block(env, body);
                self.loop_stack.pop();
                result?;
                Ok(self.subst.apply(&accumulator))
            }
            ExprKind::MethodCall(base, name, args) => {
                // `Dims.len()` — checked *before* `base` is inferred as an
                // ordinary expression: a pack generic's own name is never
                // inserted into `env` (it's tracked via `self.active_
                // generics`, a type-level mapping, not a value one), so
                // `infer_expr` on a bare `Dims` would otherwise fail with
                // `UnknownName` before this ever gets a chance to run. See
                // `pack_len_from_method_call`'s own doc comment for why this
                // can't collide with a genuine method call.
                if let Some(pack_len) = self.pack_len_from_method_call(&self.active_generics.clone(), base, name, args) {
                    return Ok(pack_len);
                }
                let base_ty = self.infer_expr(env, base)?;
                let resolved_base = self.subst.apply(&base_ty);
                match &resolved_base {
                    // Still abstract — nothing pinned the base's type down
                    // *yet*, but it still might (e.g. `apply_defaults`,
                    // which hasn't run yet at this point in an ordinary
                    // top-to-bottom pass) — deferred exactly like
                    // `FieldAccess` defers the same "not knowable yet"
                    // question, resolved for real once
                    // `check_pending_method_calls` runs, after defaulting.
                    // Arguments never depend on the base's own resolution,
                    // so they're still inferred immediately, right here.
                    Ty::Var(_) => {
                        let mut arg_tys = Vec::with_capacity(args.len());
                        let mut arg_spans = Vec::with_capacity(args.len());
                        for a in args {
                            arg_tys.push(self.infer_expr(env, a)?);
                            arg_spans.push(a.span);
                        }
                        let Ty::Var(result) = self.vars.fresh() else { unreachable!("fresh() always returns Ty::Var") };
                        self.pending_method_calls.push(PendingMethodCall {
                            base: resolved_base,
                            method: name.clone(),
                            arg_tys,
                            arg_spans,
                            base_span: base.span,
                            result,
                            call_span: expr.span,
                        });
                        Ok(Ty::Var(result))
                    }
                    // An *already*-unresolved placeholder (a method call
                    // chained off another not-yet-inferred expression) —
                    // genuinely never resolves no matter how long this
                    // waits, so it keeps returning the placeholder
                    // immediately, unchanged.
                    Ty::Con(name2) if is_placeholder(&resolved_base) => {
                        let _ = name2;
                        Ok(Ty::Con("<not-yet-inferred>".to_string()))
                    }
                    _ => {
                        // A (self- or, less commonly, sibling-triggered)
                        // recursive call back into the method *currently*
                        // having its own body inferred — see
                        // `in_progress_methods`'s own doc comment. Reuses
                        // that enclosing invocation's own already-resolved
                        // param types and return-type placeholder directly,
                        // instead of re-deriving a fresh instantiation from
                        // the registry: there's exactly one in-flight
                        // instantiation to recurse into, the same one this
                        // call is already nested inside. Handled here,
                        // inline, rather than inside `resolve_method_call`:
                        // `in_progress_methods` entries are always removed
                        // before that same call's own `finish_fn`/defaulting
                        // phase runs, so this branch is structurally
                        // guaranteed irrelevant to the deferred path above.
                        let struct_name = match &resolved_base {
                            Ty::Con(n) => Some(n.clone()),
                            Ty::App(n, _) => Some(n.clone()),
                            _ => None,
                        };
                        if let Some(struct_name) = &struct_name {
                            if let Some((param_tys, ret_ty)) =
                                self.in_progress_methods.get(&(struct_name.clone(), name.clone())).cloned()
                            {
                                self.unify_at(base.span, &param_tys[0], &resolved_base)?;
                                for (pt, a) in param_tys[1..].iter().zip(args) {
                                    let at = self.infer_expr(env, a)?;
                                    self.unify_at(a.span, pt, &at)?;
                                }
                                return Ok(self.subst.apply(&ret_ty));
                            }
                        }
                        let mut arg_tys = Vec::with_capacity(args.len());
                        let mut arg_spans = Vec::with_capacity(args.len());
                        for a in args {
                            arg_tys.push(self.infer_expr(env, a)?);
                            arg_spans.push(a.span);
                        }
                        self.resolve_method_call(&resolved_base, name, &arg_tys, &arg_spans, base.span, expr.span)
                    }
                }
            }
            ExprKind::ArrayLit(elems) => {
                // Every element must agree on one type — checked pairwise
                // against a single shared fresh var, same "unify against a
                // running accumulator" shape `infer_block`'s if/else already
                // uses, so `[1, true]` is a real type error (mismatch
                // against whichever element resolved first), not silently
                // accepted. The empty array `[]` leaves `elem_ty` an
                // unconstrained `Var` — permissive, same as any other
                // under-determined type elsewhere in this file.
                let elem_ty = self.vars.fresh();
                for e in elems {
                    let t = self.infer_expr(env, e)?;
                    self.unify_at(e.span, &elem_ty, &t)?;
                }
                Ok(Ty::Array(Box::new(elem_ty), Box::new(Ty::Const(ConstValue::Int(elems.len() as u64)))))
            }
            // `[value; N]`, `N` naming a const generic (the literal-count
            // case desugars to `ArrayLit` at lowering time instead — see
            // `grammar.pest`'s `array_repeat`). `count` is an ordinary
            // `Path` node, resolved through `env` exactly like any other
            // value reference — it reaches the *same* `Ty::Var` already
            // seeded for that const generic (see the `fresh_generics_mapping`
            // callers, which also insert each const generic into `env`).
            //
            // `[value; Dims...]` — a whole *pack* reference — can't go
            // through `env`/`infer_expr` the same way: `Dims` alone is never
            // an ordinary bound value (only `Dims.len()` is recognized, via
            // `pack_len_from_method_call`), so `infer_expr` on a bare
            // `ExprKind::PackRef` panics outright (it's not meant to be
            // reached as an *ordinary* expression). Resolved instead exactly
            // the way a struct field's own `[T; Dims...]` declared type is,
            // at the *type* level — `self.active_generics` (the enclosing
            // impl's own generic-name -> `Ty` map, `generic_arg_to_ty`'s own
            // doc comment) — giving `Ty::Array(elem, Ty::Pack(v))`, one
            // level, symbolic (mirrors `ty_from_ast_mapped`'s own `TypeKind::
            // Array` handling of the identical surface shape in type
            // position) until monomorphization resolves `v` to a real list
            // of dims and `substitute`'s own new `Ty::Array` pack-expansion
            // (below) turns it into the real nested chain.
            ExprKind::ArrayRepeat { value, count } => {
                let elem_ty = self.infer_expr(env, value)?;
                let count_ty = if let ExprKind::PackRef(name) = &count.kind {
                    self.active_generics.get(name).cloned().unwrap_or_else(|| self.vars.fresh())
                } else {
                    self.infer_expr(env, count)?
                };
                Ok(Ty::Array(Box::new(elem_ty), Box::new(count_ty)))
            }
            // One bracket group, `a[i]` or `a[i,j,...]` — `indices` is
            // never empty (grammar requires at least one `expr` inside
            // `[...]`). `base`'s own type decides everything:
            ExprKind::Index(base, indices) => {
                let base_ty = self.infer_expr(env, base)?;
                let resolved_base = self.subst.apply(&base_ty);
                // Indices never depend on the base's own resolution —
                // inferred (and `Int`-constrained) immediately either way,
                // mirroring `PendingMethodCall`'s own "arguments captured
                // already-resolved" split (see `Infer::pending_indices`'s
                // own doc comment) — only what indexing actually *means*
                // (peel an array dimension, or dispatch `Index<Container,
                // Elem,K>`) is what's deferred below, when it can't be
                // decided yet.
                let mut index_tys: Vec<Ty> = Vec::with_capacity(indices.len());
                let mut index_spans: Vec<Span> = Vec::with_capacity(indices.len());
                for idx in indices {
                    let idx_ty = self.infer_expr(env, idx)?;
                    self.constraints.push(Constraint::all_gating("Int".to_string(), vec![idx_ty.clone()], idx.span));
                    index_tys.push(idx_ty);
                    index_spans.push(idx.span);
                }
                match &resolved_base {
                    // Still abstract — but *not* a dead end the way an
                    // already-unresolved placeholder is (see the guard just
                    // below): `mc[0,0]` right after `let mc = matmul(ma,
                    // mb);` has `mc`'s own type as a bare `Ty::Var` at this
                    // exact point (an algebra call's output-only generic
                    // isn't independently concrete until its own dispatch
                    // runs, itself possibly still deferred — see `doc/
                    // backlog.md`'s own "`check_pending_constraints`'s
                    // output-only-generic gate" item), but it *will*
                    // resolve, later, once `apply_defaults`/`check_pending_
                    // constraints` have run. Mirrors `FieldAccess`'s own
                    // identical `Ty::Var(_)` arm exactly: a fresh real
                    // `Ty::Var` (not a dead `Ty::Con` placeholder) is
                    // returned so this node can still be pinned down by a
                    // later unification, and the real resolution is
                    // deferred to `check_pending_indices`.
                    Ty::Var(_) => {
                        let Ty::Var(result) = self.vars.fresh() else { unreachable!("fresh() always returns Ty::Var") };
                        self.pending_indices.push(PendingIndex {
                            base: resolved_base,
                            base_span: base.span,
                            index_tys,
                            index_spans,
                            result,
                            span: expr.span,
                        });
                        Ok(Ty::Var(result))
                    }
                    // An *already*-unresolved placeholder (chained off
                    // another not-yet-inferred expression, e.g. an
                    // undeclared cross-function call) — genuinely never
                    // resolves no matter how long this waits, so it keeps
                    // returning the placeholder immediately, unchanged.
                    _ if is_placeholder(&resolved_base) => Ok(Ty::Con("<not-yet-inferred>".to_string())),
                    _ => self.resolve_index(resolved_base, &index_tys, &index_spans, base.span, expr.span),
                }
            }
            ExprKind::FieldAccess(base, name) => {
                let base_ty = self.infer_expr(env, base)?;
                let resolved = self.subst.apply(&base_ty);
                match &resolved {
                    // Still abstract — nothing pinned the base's type down
                    // *yet*, but it still might (e.g. `apply_defaults`,
                    // which hasn't run yet at this point in an ordinary
                    // top-to-bottom pass) — deferred exactly like
                    // `pending_type_name_checks` defers a different "not
                    // knowable yet" question, resolved for real once
                    // `check_pending_field_accesses` runs, after defaulting.
                    Ty::Var(_) => {
                        let Ty::Var(result) = self.vars.fresh() else { unreachable!("fresh() always returns Ty::Var") };
                        self.pending_field_accesses.push(PendingFieldAccess {
                            base: resolved,
                            field: name.clone(),
                            result,
                            span: expr.span,
                        });
                        Ok(Ty::Var(result))
                    }
                    // An *already*-unresolved placeholder (a field access
                    // chained off another not-yet-inferred expression, e.g.
                    // an undeclared cross-function call) — genuinely never
                    // resolves no matter how long this waits, so it keeps
                    // returning the placeholder immediately, unchanged.
                    Ty::Con(name2) if is_placeholder(&resolved) => {
                        let _ = name2;
                        Ok(Ty::Con("<not-yet-inferred>".to_string()))
                    }
                    _ => self.resolve_field_access(&resolved, name, expr.span),
                }
            }
            ExprKind::StructLit(path, explicit_generics, fields) => {
                let struct_name = path.segments.join("::");
                let Some(declared_fields) = self.registry.struct_fields(&struct_name).map(<[Field]>::to_vec) else {
                    return Err(TypeError { span: expr.span, kind: TypeErrorKind::UnknownStruct(struct_name) });
                };
                // Fresh variables for the struct's own generic parameters
                // (if any), one per construction site — exactly like an
                // algebra call instantiating the algebra's own `T` fresh
                // per call (`infer_algebra_call`). `T`'s eventual concrete
                // type is normally *inferred* from the field values below —
                // matches this project's "infer everything from usage"
                // stance everywhere else — but can also be pinned explicitly
                // via a turbofish (`Matrix::<f64, 4, 4>(values: ...)`),
                // needed whenever nothing about the field values themselves
                // would otherwise determine it (a real, reported ergonomics
                // gap: relying on an incidentally-suffixed literal to force
                // the right type through unification). Positional against
                // `struct_generics`' own declaration order — the same
                // convention `Ty::App`'s argument list, and `zip_struct_
                // generics`, already use.
                let struct_generics = self.registry.struct_generics(&struct_name).to_vec();

                // A struct whose own last declared generic is a *pack*
                // (`doc/backlog.md`'s own "Variadic generics" item) needs a
                // genuinely different construction path — a pack can't be
                // represented as one more `name -> Ty` entry in an ordinary
                // `HashMap<String, Ty>` mapping (it resolves to *several*
                // types, not one), and its own arity isn't knowable from the
                // struct's own declaration at all, only from a concrete
                // construction site. See `Self::infer_struct_lit_with_pack`'s
                // own doc comment for the full design.
                if struct_generics.last().is_some_and(GenericParam::is_variadic) {
                    return self.infer_struct_lit_with_pack(env, expr.span, &struct_name, &struct_generics, explicit_generics, fields, &declared_fields);
                }

                let generics_mapping = self.fresh_generics_mapping(&struct_generics, expr.span);
                if !explicit_generics.is_empty() {
                    if explicit_generics.len() != struct_generics.len() {
                        return Err(TypeError {
                            span: expr.span,
                            kind: TypeErrorKind::ArityMismatch {
                                name: format!("{struct_name}::<...>"),
                                expected: struct_generics.len(),
                                found: explicit_generics.len(),
                            },
                        });
                    }
                    for (g, explicit) in struct_generics.iter().zip(explicit_generics) {
                        let name = match g {
                            GenericParam::Type { name, .. } => name,
                            GenericParam::Const { name, .. } => name,
                        };
                        let fresh = generics_mapping[name].clone();
                        let explicit_ty = self.generic_arg_to_ty(explicit);
                        self.unify_at(expr.span, &fresh, &explicit_ty)?;
                    }
                }

                let mut seen: HashSet<String> = HashSet::new();
                for (name, value) in fields {
                    let Some(decl_field) = declared_fields.iter().find(|f| &f.name == name).cloned() else {
                        return Err(TypeError {
                            span: value.span,
                            kind: TypeErrorKind::NoSuchField { struct_name: struct_name.clone(), field: name.clone() },
                        });
                    };
                    if !seen.insert(name.clone()) {
                        return Err(TypeError {
                            span: value.span,
                            kind: TypeErrorKind::DuplicateField {
                                struct_name: struct_name.clone(),
                                field: name.clone(),
                            },
                        });
                    }
                    let value_ty = self.infer_expr(env, value)?;
                    let declared_ty = self.ty_from_ast_mapped(&decl_field.ty, &generics_mapping);
                    self.unify_at(value.span, &declared_ty, &value_ty)?;
                }
                if let Some(missing) = declared_fields.iter().find(|f| !seen.contains(&f.name)) {
                    return Err(TypeError {
                        span: expr.span,
                        kind: TypeErrorKind::MissingField {
                            struct_name: struct_name.clone(),
                            field: missing.name.clone(),
                        },
                    });
                }

                // Every one of the struct's own generic parameters — type
                // *and* const — becomes one `App` argument, positionally,
                // matching `Ty::App`'s own documented convention and what
                // `zip_struct_generics` expects to zip back apart at a
                // future field access. A const-generic gets no explicit
                // value at the construction site (there's no syntax for
                // one, unlike a type annotation's `Matrix<f64, 3, 3>`) — its
                // fresh var is pinned the same way any other field-inferred
                // type is: by `unify_at` above, against whatever the actual
                // field *value* turned out to be (an `[f64; 3]`-typed field
                // value unifies a `[T; N]`-declared field's `N` to `Const(3)`
                // for free, no separate mechanism needed).
                let type_args: Vec<Ty> = struct_generics
                    .iter()
                    .filter_map(|g| {
                        let name = match g {
                            GenericParam::Type { name, .. } => name,
                            GenericParam::Const { name, .. } => name,
                        };
                        generics_mapping.get(name).cloned()
                    })
                    .collect();
                if type_args.is_empty() {
                    Ok(Ty::Con(struct_name))
                } else {
                    Ok(Ty::App(struct_name, type_args))
                }
            }
            ExprKind::Lambda { params, ret, body } => {
                let mut inner_env = env.clone();
                let mut param_tys = Vec::with_capacity(params.len());
                for p in params {
                    // `active_generics`, not `ty_from_ast`'s always-empty
                    // map — see that field's own doc comment: a nested
                    // lambda's own parameter annotation can reference the
                    // enclosing `fn`/impl-method's own generic just as
                    // easily as a `let`'s own can.
                    let ty = match &p.ty {
                        Some(t) => self.ty_from_ast_mapped(t, &self.active_generics.clone()),
                        None => self.vars.fresh(),
                    };
                    inner_env.insert(p.name.clone(), Scheme::mono(ty.clone()));
                    param_tys.push(ty);
                }
                // `infer_block` clones `inner_env` again internally — fine,
                // it's the same cheap-clone tradeoff already made everywhere
                // else (see `infer_block`'s own doc comment).
                //
                // `loop_stack` swapped for empty while checking the body —
                // a `break` must not escape through a closure boundary (see
                // `loop_stack`'s own doc comment), even when the lambda is
                // lexically written inside a loop.
                let outer_loop_stack = std::mem::take(&mut self.loop_stack);
                let body_ty = self.infer_block(&inner_env, body);
                self.loop_stack = outer_loop_stack;
                let body_ty = body_ty?;
                if let Some(r) = ret {
                    let declared = self.ty_from_ast_mapped(r, &self.active_generics.clone());
                    self.unify_at(r.span, &declared, &body_ty)?;
                }
                Ok(Ty::Fn(param_tys, Box::new(body_ty)))
            }
            // `Dims...` reached as an ordinary expression -- only ever
            // meaningful as an array dimension's own size expression
            // (`TypeKind::Array`), resolved through `ty_from_ast_mapped`'s
            // own dedicated arm, never through general expression
            // inference. Grammar/AST exist (Milestone 1 of `doc/backlog.md`'s
            // own "Variadic generics" item); nothing resolves a pack yet.
            ExprKind::PackRef(name) => panic!("type inference: pack reference `{name}...` reached ordinary expression inference -- variadic generics aren't semantically supported yet (only the grammar/AST exist so far)"),
        }
    }

    /// No special case for operator names anywhere in here — `add`, `lt`,
    /// `and`, a `let`-bound lambda, and a genuinely unknown name all resolve
    /// through the exact same path: a real declared `algebra` (checked
    /// first, against the registry), then a bound lambda, then an honest
    /// "unresolved" placeholder. Matches `grammar.md`'s founding principle —
    /// "an operator is not a distinct language concept... dispatched by the
    /// same algebra mechanism as any other call" — literally, now, not just
    /// in the surface desugaring.
    ///
    /// An earlier version special-cased `add`/`sub`/`lt`/`and`/... with a
    /// permissive built-in fallback (unify any single type, no membership
    /// check) whenever zero algebras matched — meant as a temporary bridge
    /// before a real registry existed. Keeping it *after* the registry
    /// existed was actively harmful: it meant ordinary arithmetic
    /// (`x + 1`), completely unchanged, silently accepted absolutely any
    /// type combination as long as nothing algebra-based existed to check
    /// against — which, absent a full stdlib, was every program. `let x =
    /// fn(a) { a + a }; x(true)` type-checked. Removed — `add` with zero
    /// registered candidates is now exactly as "unresolved" as calling an
    /// undeclared function, no quieter.
    fn infer_call(
        &mut self,
        env: &Env,
        call_span: Span,
        path: &Path,
        explicit_generics: &[GenericArg],
        args: &[Expr],
    ) -> Result<Ty, TypeError> {
        // Qualified call (`Ring::mul(a, b)`) — `doc/backlog-done.md`'s own
        // "qualified-call syntax" item: disambiguates when more than one
        // algebra would otherwise own this name+arity (`AmbiguousOperator`,
        // below) by naming the intended algebra explicitly, `doc/grammar.md`'s
        // own already-documented design for this. Checked *before* the
        // ordinary unqualified path: a 2-segment `path` whose first segment
        // names a real, declared algebra is unambiguously a qualified call —
        // operator sugar (`a * b`) can never accidentally reach here, `lower.
        // rs`'s own `fold_binary` always builds a single-segment `Path::
        // single(name)`. Any other 2-segment path (naming something that
        // isn't a declared algebra) falls through to the unqualified path
        // below, unchanged.
        if let [algebra, method] = path.segments.as_slice() {
            if self.registry.has_algebra(algebra) {
                let arg_tys: Vec<Ty> = args.iter().map(|a| self.infer_expr(env, a)).collect::<Result<_, _>>()?;
                // `infer_algebra_call` itself `unreachable!()`s if `fn_sig`
                // misses — it assumes its own caller already confirmed the
                // method exists (its one existing caller only ever reaches it
                // via an already-`algebras_with_fn`-confirmed candidate) — so
                // this check has to happen here, not there.
                if !self.registry.fn_sig(algebra, method).is_some_and(|sig| sig.params.len() == args.len()) {
                    return Err(TypeError {
                        span: call_span,
                        kind: TypeErrorKind::UnknownAlgebraMethod { algebra: algebra.clone(), method: method.clone() },
                    });
                }
                let arg_spans: Vec<Span> = args.iter().map(|a| a.span).collect();
                return self.infer_algebra_call(call_span, algebra, method, &arg_tys, &arg_spans, explicit_generics);
            }
        }

        let name = path.segments.join("::");
        let arg_tys: Vec<Ty> = args
            .iter()
            .map(|a| self.infer_expr(env, a))
            .collect::<Result<_, _>>()?;

        let candidates = self.registry.algebras_with_fn(&name, args.len());
        if candidates.len() > 1 {
            return Err(TypeError {
                span: call_span,
                kind: TypeErrorKind::AmbiguousOperator {
                    name,
                    candidates: candidates.into_iter().map(String::from).collect(),
                },
            });
        }
        if let Some(&algebra) = candidates.first() {
            // Explicit turbofish on an algebra-dispatched operator call
            // (`add::<f64>(a, b)`, `convert::<i32, f64>(x)`) — threaded
            // straight through to `infer_algebra_call`, which pins each
            // named generic's own fresh var before dispatch ever runs (see
            // its own doc comment). Needed for real by `Convert<From, To>`:
            // an output-only generic like `To` is never independently
            // constrained by the call's own arguments, so an ambiguous
            // dispatch (`dispatch_algebra_call`'s own `AmbiguousDispatch`)
            // has no other way to be resolved.
            let arg_spans: Vec<Span> = args.iter().map(|a| a.span).collect();
            return self.infer_algebra_call(call_span, algebra, &name, &arg_tys, &arg_spans, explicit_generics);
        }

        if let Some(scheme) = env.get(&name).cloned() {
            // Calling a `let`-bound lambda by name (`let f = fn(a,b){a+b}; f(1,2)`),
            // instantiated fresh per call site so a generalized `f` can be
            // used at different types across different calls. Calling a
            // lambda *literal* directly isn't representable yet — `Call`'s
            // callee is a `Path`, not an arbitrary `Expr` (see
            // `grammar.pest`'s `lambda_expr` note) — deliberately deferred.
            let (instantiated, mapping) = self.instantiate_with_mapping(&scheme);
            if !explicit_generics.is_empty() {
                if explicit_generics.len() != scheme.vars.len() {
                    return Err(TypeError {
                        span: call_span,
                        kind: TypeErrorKind::ArityMismatch {
                            name: format!("{name}::<...>"),
                            expected: scheme.vars.len(),
                            found: explicit_generics.len(),
                        },
                    });
                }
                // `scheme.vars`' own order — see `instantiate_with_mapping`'s
                // doc comment for why this reliably matches declaration
                // order for the common case.
                for (v, g) in scheme.vars.iter().zip(explicit_generics) {
                    let fresh = mapping[v].clone();
                    let explicit_ty = self.generic_arg_to_ty(g);
                    self.unify_at(call_span, &fresh, &explicit_ty)?;
                }
            }
            match instantiated {
                Ty::Fn(param_tys, ret_ty) => {
                    if param_tys.len() != args.len() {
                        return Err(TypeError {
                            span: call_span,
                            kind: TypeErrorKind::ArityMismatch { name, expected: param_tys.len(), found: args.len() },
                        });
                    }
                    for (pt, (t, a)) in param_tys.iter().zip(arg_tys.iter().zip(args)) {
                        self.unify_at(a.span, pt, t)?;
                    }
                    Ok(*ret_ty)
                }
                other => Err(TypeError { span: call_span, kind: TypeErrorKind::NotCallable(other) }),
            }
        } else if args.is_empty() && self.registry.struct_fields(&name).is_some_and(<[Field]>::is_empty) {
            // `Empty()` — a zero-*field* struct's own construction syntax is
            // grammatically identical to a zero-arg call (see
            // `grammar.pest`'s `primary` comment: `call_expr` is tried
            // first, so it always wins the ambiguity) — special-cased here,
            // the one remaining gap that ordering doesn't resolve on its
            // own, rather than left to fall through to the unresolved
            // placeholder below.
            Ok(Ty::Con(name))
        } else {
            // No declared algebra owns this name (checked above, via the
            // registry) and it's not a `let`-bound lambda either. Left as an
            // explicit unknown rather than silently guessing, matching the
            // "never silent" principle elsewhere in this project — this is
            // still reachable for any name with zero registered candidates,
            // which today is most operators (no stdlib declares `Ring` yet).
            Ok(Ty::Con(format!("<unresolved-call:{name}>")))
        }
    }

    /// Resolves a call against exactly one candidate algebra (ambiguity —
    /// more than one candidate — is handled by the caller, `infer_call`,
    /// before this is ever reached). Instantiates the algebra's own generic
    /// parameters with fresh variables, unifies the declared signature's
    /// parameter types against the actual arguments, and — the actual
    /// payoff of having a registry at all — checks that a matching `impl`
    /// exists for every parameter type that ended up concrete. A parameter
    /// type still abstract (a generic caller) is deferred as a `Constraint`
    /// instead — see `generalize`/`check_pending_constraints`.
    fn infer_algebra_call(
        &mut self,
        call_span: Span,
        algebra: &str,
        name: &str,
        arg_tys: &[Ty],
        arg_spans: &[Span],
        explicit_generics: &[GenericArg],
    ) -> Result<Ty, TypeError> {
        let sig = self
            .registry
            .fn_sig(algebra, name)
            .cloned()
            .unwrap_or_else(|| unreachable!("registry reported `{algebra}` declares `{name}`, but fn_sig lookup failed"));

        if sig.params.len() != arg_spans.len() {
            return Err(TypeError {
                span: call_span,
                kind: TypeErrorKind::ArityMismatch { name: name.to_string(), expected: sig.params.len(), found: arg_spans.len() },
            });
        }

        let generics = self.registry.generics(algebra).to_vec();
        // Both `Type` *and* `Const` generics need a fresh var here — `Const`
        // used to be filtered out, so a signature referencing the algebra's
        // own const generic (an array size, `[T; N]`) could never resolve
        // `N` at all (`ty_from_ast_mapped`'s own `TypeKind::Array` arm falls
        // back to `<array-type-not-yet-inferred>` when its size expression's
        // mapped name is missing) — found by direct testing. Reuses the same
        // helper `has_matching_impl`'s own speculative probe already relies
        // on for the identical "just fresh vars, no side effects" need.
        let mapping = self.fresh_vars_for_generics(&generics);

        // Explicit turbofish (`convert::<i32, f64>(x)`) — same convention
        // as the `let`-bound-lambda/top-level-fn call path just above this
        // one: every declared generic, positionally, no partial turbofish.
        // Unifying here, before `param_tys`/`ret_ty`/`resolved_generics` are
        // built below, means an output-only generic (`To` in `Convert<From,
        // To>`, never gated the way a parameter-appearing one is) can be
        // pinned *before* `dispatch_algebra_call` ever runs — the only way
        // to resolve a real `AmbiguousDispatch` between two impls that
        // agree on every parameter-appearing generic but disagree on an
        // output-only one.
        if !explicit_generics.is_empty() {
            if explicit_generics.len() != generics.len() {
                return Err(TypeError {
                    span: call_span,
                    kind: TypeErrorKind::ArityMismatch {
                        name: format!("{name}::<...>"),
                        expected: generics.len(),
                        found: explicit_generics.len(),
                    },
                });
            }
            for (param, explicit) in generics.iter().zip(explicit_generics) {
                let param_name = match param {
                    GenericParam::Type { name, .. } => name,
                    GenericParam::Const { name, .. } => name,
                };
                let fresh = mapping[param_name].clone();
                let explicit_ty = self.generic_arg_to_ty(explicit);
                self.unify_at(call_span, &fresh, &explicit_ty)?;
            }
        }

        let param_tys: Vec<Ty> = sig
            .params
            .iter()
            .map(|p| match &p.ty {
                Some(t) => self.ty_from_ast_mapped(t, &mapping),
                None => self.vars.fresh(),
            })
            .collect();
        let ret_ty =
            sig.ret.as_ref().map(|t| self.ty_from_ast_mapped(t, &mapping)).unwrap_or_else(|| Ty::Con("()".to_string()));

        for (pt, (at, span)) in param_tys.iter().zip(arg_tys.iter().zip(arg_spans)) {
            self.unify_at(*span, pt, at)?;
        }

        // Checked against the *algebra's own* generics (positionally
        // resolved through `mapping`, e.g. `[A, B, C]` for a heterogeneous
        // `algebra MatMul<A, B, C>`), together, in one coherent dispatch —
        // not `param_tys` independently, one at a time. A per-parameter
        // loop over `param_tys` (an earlier version of this method's own
        // shape) can't express "these two parameters must resolve to types
        // that share a common impl instantiation": nothing would connect
        // the *trial* substitution used to check the first parameter to the
        // one used for the second, so a shape-mismatched
        // `Matrix<f32,2,3> * Matrix<f32,4,5>` could pass both checks
        // independently while the call's own result type stayed a bare,
        // totally unconstrained variable — found by direct testing. For a
        // single-generic algebra (`Ring<T>`, the overwhelming common case),
        // this resolves to exactly one type and behaves identically to the
        // old per-parameter check.
        //
        // `C` (a generic appearing *only* in the return type, never in any
        // parameter — `fn mul(a: A, b: B) -> C;`) is never independently
        // constrained by the call's own arguments the way `A`/`B` are —
        // the *only* way to ever learn its concrete value is from a
        // successful dispatch match itself. Gating readiness on "every
        // generic is concrete" (an earlier version of this method did
        // exactly that) can never work for such a generic: `C` starts and
        // stays an ordinary fresh, totally free `Ty::Var` right up until
        // dispatch itself resolves it, so the gate would never open —
        // found by direct testing (`no impl MatMul<Matrix<f32,2,3>>`, a
        // single-type message, instead of ever attempting a real
        // three-target match at all). Only the generics that actually
        // appear in at least one *parameter* — found by checking whether
        // each generic's own fresh var occurs free in `param_tys` — gate
        // readiness; a return-type-only generic rides along into the match
        // itself and gets bound *by* it.
        let mut param_free_vars: HashSet<TyVar> = HashSet::new();
        for pt in &param_tys {
            free_vars(pt, &mut param_free_vars);
        }
        let mut resolved_generics: Vec<Ty> = Vec::new();
        let mut gating: Vec<Ty> = Vec::new();
        // Positions in `resolved_generics` that are gating — threaded into
        // a deferred `Constraint` below (`Constraint::gating_indices`'s own
        // doc comment) so `check_pending_constraints`, whenever it re-checks
        // this exact tuple later, knows which position(s) it's still
        // waiting on and which are output-only and should instead be bound
        // *by* a real, committing dispatch, not required concrete before
        // one is even attempted — the fix for `doc/backlog.md`'s own
        // "`check_pending_constraints`'s output-only-generic gate" item.
        let mut gating_indices: Vec<usize> = Vec::new();
        for g in &generics {
            let GenericParam::Type { name, .. } = g else { continue };
            let fresh = &mapping[name];
            let resolved = self.subst.apply(fresh);
            if matches!(fresh, Ty::Var(v) if param_free_vars.contains(v)) {
                gating.push(resolved.clone());
                gating_indices.push(resolved_generics.len());
            }
            resolved_generics.push(resolved);
        }

        // Committing immediately, right here mid-body, the moment *gating*
        // alone is concrete (this method's own original design, restored
        // below) is right for the overwhelmingly common case — it's what
        // lets an output-only generic like `MatMul<A,B,C>`'s own `C` end up
        // resolved *at all* when nothing downstream ever annotates it
        // explicitly (`let c = matmul(a,b); c.values[0,0]` — deferring here
        // would leave `c`'s own type an unresolved var for the rest of the
        // body, well before `check_pending_constraints` ever gets a chance
        // to fix it, breaking every subsequent field/index access on it).
        // Two narrower cases still need to defer instead of committing
        // outright, layered on top of that general rule:
        let active_vars_pending = !self.active_generics.is_empty() && !resolved_generics.iter().all(is_fully_concrete);

        if gating.iter().any(is_placeholder) || active_vars_pending {
            // Either an *input* generic is an outright "unknown" (never
            // type-inferred), or — `active_vars_pending` — we're still
            // checking a still-*generic* fn/impl's own declaration (`self.
            // active_generics` non-empty, set for the whole duration of
            // that check) and some position, gating or output-only, isn't
            // concrete yet. The second case matters even with only *one*
            // matching candidate, where `dispatch_algebra_call` below would
            // otherwise commit without any ambiguity ever being raised:
            // found directly, `stdlib/nn/nn.cleave`'s own `mean` calling
            // `N.to()` (`Convert<i32,T>`, only `Convert<i32,f64>` declared
            // at the time) — `From` (`i32`) is concrete *independent* of
            // the enclosing impl's still-abstract `T`, so gating alone said
            // "ready" and permanently committed `self.subst`'s `T` to
            // `f64` during mere *declaration*-checking, before any real
            // instantiation ever ran — wrong the moment an `f32`
            // instantiation needed a different `T`. Deferred below, same as
            // the "still abstract" case — one constraint holding the whole
            // tuple together (`Constraint`'s own doc comment) — resolved
            // per real instantiation instead, once `quantify_impl_generics`
            // (called right after this body finishes) marks `T` quantified
            // and `check_pending_constraints`'s own leading guard skips it
            // silently, exactly like any other still-open generic bound.
            self.constraints.push(Constraint { algebra: algebra.to_string(), tys: resolved_generics, gating_indices, span: call_span });
        } else if gating.iter().all(is_fully_concrete) {
            // Ready: every *input* generic is known, and we're not
            // mid-declaration of a still-open generic either, so dispatch
            // can actually run — commits the match's own bindings for real
            // (see `dispatch_algebra_call`'s own doc comment for why that's
            // sound here specifically), which is what lets an output-only
            // generic like `C` end up resolved at all.
            match self.dispatch_algebra_call(algebra, &resolved_generics, call_span) {
                Ok(true) => {}
                Ok(false) => {
                    let ty = resolved_generics.iter().map(Ty::to_string).collect::<Vec<_>>().join(", ");
                    return Err(TypeError { span: call_span, kind: TypeErrorKind::MissingImpl { algebra: algebra.to_string(), ty } });
                }
                // More than one candidate structurally matches, and they
                // disagree on some still-open position — by construction
                // this can only happen when that position's own query type
                // was itself still a bare `Ty::Var` going in (two
                // *concrete* target patterns can never both unify against
                // one already-concrete query — `check_no_overlapping_impls`
                // already rules out two impls agreeing on a shared shape).
                // So "ambiguous right now" doesn't mean "ambiguous forever"
                // — the very next statement (an enclosing `let`'s own
                // annotation, a sibling operand in a binary op, ...) may
                // still pin that position down through ordinary
                // unification, entirely outside this dispatch. Found
                // directly: `let f: f64 = n.to();`, an *ordinary top-level*
                // call with no enclosing generic body at all, hit a false
                // `AmbiguousDispatch` the moment a second `Convert<i32,_>`
                // candidate (`Convert<i32,f32>`) existed, even though this
                // call's own real target is never actually ambiguous — the
                // annotation just hadn't been consulted yet. Deferred here
                // exactly like the cases above; `check_pending_constraints`
                // gets the *final* say once the whole body (hence every
                // local unification that could possibly disambiguate it)
                // has been walked — if it's still ambiguous *then*, that
                // failure is real and is allowed to propagate.
                Err(TypeError { kind: TypeErrorKind::AmbiguousDispatch { .. }, .. }) => {
                    self.constraints.push(Constraint {
                        algebra: algebra.to_string(),
                        tys: resolved_generics,
                        gating_indices,
                        span: call_span,
                    });
                }
                Err(e) => return Err(e),
            }
        } else {
            // Still abstract somewhere among the *input* generics (a
            // generic caller, or a `Complex<'t9>` whose own argument isn't
            // pinned down yet) — defer, one constraint for the whole tuple:
            // either `generalize` migrates it into an enclosing `let`'s
            // scheme, or `check_pending_constraints` catches it once
            // `infer_fn` finishes, whichever comes first. The output-only
            // positions (not in `gating_indices`) stay open `Ty::Var`s in
            // `resolved_generics` here, on purpose — `check_pending_
            // constraints` binds them later, it doesn't require them
            // concrete first.
            self.constraints.push(Constraint { algebra: algebra.to_string(), tys: resolved_generics, gating_indices, span: call_span });
        }

        Ok(self.subst.apply(&ret_ty))
    }

    /// Defaults any number-literal type variable never pinned to a concrete
    /// type by unification — mirrors Haskell's numeric-literal defaulting.
    ///
    /// Groups pending entries by their *current* union-find root before
    /// applying anything, rather than defaulting each one independently as
    /// it's encountered — a real bug, found by testing:
    /// `add(fibonacci(42), fibonacci(42.0))` (two calls whose results get
    /// forced to the same type by a shared generic `T`) used to silently
    /// resolve to `i32`, discarding `42.0`'s own `Float` preference purely
    /// because `42`'s `Int` preference happened to be processed first and
    /// bind the shared variable before `42.0`'s own entry was even looked
    /// at — by the time the second entry was checked, `subst` already
    /// reported the variable as concrete, indistinguishable from "pinned by
    /// a real declared type" (which correctly *should* silently win over a
    /// literal's own shape-based preference — see below). Grouping first
    /// means every literal sharing one still-open variable is compared
    /// against the others *before* any of them win, so a genuine conflict
    /// between two differently-shaped literals is caught instead of
    /// resolved by processing order.
    ///
    /// Deliberately does *not* try to detect a literal-shape conflict here
    /// (e.g. `add(1, 2.0)`, both defaulted through the same shared, merged
    /// variable) — an earlier version of this function grouped pending
    /// entries by their union-find root and compared preferences directly,
    /// entirely inside `apply_defaults`. That approach worked for two
    /// literals merged with *nothing else* deciding anything, but missed
    /// every other way the same conflict shows up: a variable pinned
    /// concrete by a *real* external unification (a declared return type)
    /// silently overrides a literal's own preference with no check at all
    /// (correctly, for `fn f() -> f64 { 1 }` — but that same silent-override
    /// path was indistinguishable from a variable a *sibling literal's own
    /// default* had just concretized, which should **not** be silently
    /// overridden), and a variable that ends up quantified — generalized,
    /// never itself defaulted — skipped the grouping check entirely by
    /// design, so two differently-shaped literals feeding a mutually
    /// recursive group's shared, generalized return type went unchecked no
    /// matter how the group was later (or never) instantiated. Real bugs,
    /// both found by testing after the fact.
    ///
    /// The actual fix doesn't live here at all: a literal's shape now
    /// generates its own real, independent constraint (`Int`/`Float`
    /// alongside `Num`, `stdlib/num/num.cleave`, pushed in the `NumberLit`
    /// branch above) instead of being consulted only as a defaulting
    /// preference. `check_pending_constraints` already checks *any* named
    /// algebra uniformly, regardless of how a variable ended up concrete
    /// (default, real unification, or — once instantiated — a scheme's own
    /// quantified variable) — so `Int`/`Float` get exactly the same
    /// treatment `Num` already had, with zero special-casing needed here.
    pub(crate) fn apply_defaults(&mut self) {
        let mut defaults = std::mem::take(&mut self.pending_defaults);
        // A `Complex` default must win over a merged `Int`/`Float` sibling
        // regardless of source order — `5.0 + 7.5i` and `7.5i + 5.0` must
        // resolve identically (ℤ, ℝ ⊂ ℂ: a bare `Int`/`Float` shape widens
        // into `Complex`, never the reverse). Processing every `Complex`
        // entry first means whichever `Int`/`Float` sibling shares that
        // same merged variable always finds it already concrete below
        // (skipped, same as any other already-resolved sibling) instead of
        // racing to default first — real bug, found by direct testing:
        // `5.0 + 7.5i` (the plain float literal written first) resolved
        // the shared variable to a bare `f32`, silently discarding the
        // imaginary part, while `7.5i + 5.0` happened to work purely by
        // accident of iteration order. `sort_by_key` is stable, so this
        // only ever reorders `Complex` ahead of non-`Complex` — it never
        // disturbs relative order within either group.
        defaults.sort_by_key(|(_, default)| !matches!(default, NumberDefault::Complex));
        for (var, default) in defaults {
            // Resolve to the *current* union-find root before checking
            // `quantified` — `var` here is the literal's own original
            // `TyVar`, which may since have become a mere alias (`subst`
            // chains it to some other, still-unbound variable) rather than
            // the root `generalize` actually recorded as quantified;
            // checking membership on the raw, pre-resolution `var` would
            // miss exactly the cases this guard exists for.
            let Ty::Var(root) = self.subst.apply(&Ty::Var(var)) else {
                // Already concrete — via a real unification, or via a
                // sibling literal's own default just now. Either way,
                // nothing left to default; a genuine shape conflict is
                // `check_pending_constraints`'s job now, not this one's.
                continue;
            };
            if self.quantified.contains(&root) {
                continue;
            }
            // A const generic's own shape-slot var can end up sharing this
            // same root — merged in by ordinary unification, e.g. `for i in
            // 0..N` unifying `0`'s own (defaultable) literal var with `N`'s
            // (see `ExprKind::For`'s own inference). Defaulting it here
            // would commit `N := Ty::Con("i32")` for real — a *type*,
            // permanently overwriting the `Ty::Var` this slot must stay as
            // until monomorphization resolves it to a concrete `Ty::Const`
            // (a *value*) — found by direct testing: `examples/matmul.cleave`'s
            // own `N`/`M`/`K` bounds, defaulted this way, then failed to
            // unify against a real call site's `Ty::Const(2)` during
            // monomorphization, structurally incompatible with the `Con`
            // shape defaulting had already locked in. Skipped exactly like
            // `quantified` above — this variable belongs to const-generic
            // resolution (`check_pending_constraints`'s own `Ty::Const`
            // bridge, or monomorphization's reverse-unification), not to
            // ordinary numeric-literal defaulting. `root` is exactly the
            // right variable to check here, not just a convenient one:
            // `Subst::bind`'s own forward propagation (see its doc comment)
            // guarantees `root`'s own entry exists whenever `var`'s chain
            // ever passed through a const generic's own value slot, however
            // many merges deep and regardless of which direction each merge
            // happened to bind — real bug, found and fixed by direct
            // testing (`for i in N..5`, `N` as the range's own *start*, used
            // to silently defeat this exact guard before `const_width`
            // moved into `Subst` itself).
            if self.subst.const_width(root).is_some() {
                continue;
            }
            let default_ty = match default {
                NumberDefault::Int => Ty::Con("i32".to_string()),
                NumberDefault::Float => Ty::Con("f32".to_string()),
                // Matches `Float`'s own default width — no principled reason
                // for a bare `4i` to default to a wider real/imaginary
                // component than a bare `4.0` would (real, found by direct
                // testing: the two used to disagree).
                NumberDefault::Complex => Ty::App("Complex".to_string(), vec![Ty::Con("f32".to_string())]),
            };
            unify(&mut self.subst, &Ty::Var(root), &default_ty)
                .expect("defaulting an unbound, non-quantified variable can't fail");
        }
    }
}

