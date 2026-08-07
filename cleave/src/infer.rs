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
//! immutable `let` whose right-hand side is a *syntactic value* (literal,
//! variable reference, or lambda — see [`is_syntactic_value`]) gets
//! generalized at the binding site ([`Infer::generalize`]); every later
//! reference instantiates a fresh copy ([`Infer::instantiate`]), so e.g.
//! `let id = fn(x) { x }; id(1); id(true);` type-checks even though `id`'s
//! parameter is never annotated. Two deliberate restrictions, not oversights:
//! - **`let mut` is never generalized**, regardless of what its value looks
//!   like — a mutable binding can be reassigned at one instantiation's type
//!   and read back at another's, which is exactly the classical ML
//!   ref-cell-polymorphism unsoundness. Simpler and safer than trying to
//!   reason about aliasing through the value's shape.
//!
//! A number literal's own type variable *is* eligible for generalization
//! (see `generalize`'s doc comment for why an earlier version wrongly
//! excluded it) — with its `Num` constraint riding along, so
//! `fn add_one(x) { x + 1 }` correctly generalizes to `∀t. Num t => (t)->t`
//! and is usable at every numeric type with a registered `Num` impl, not
//! forced monomorphic the moment a bare literal appears in its body.
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
//! - A self-recursive **`let`-bound lambda** (`let g = fn(n) { g(n) };`) —
//!   `g` isn't in `env` yet while its own body is being inferred, so its
//!   self-call falls through to the "unresolved call" placeholder like any
//!   undeclared name. Only top-level `fn`s get the self/mutual-recursion
//!   treatment (`infer_fn`'s self-reference binding; `callgraph.rs`'s
//!   whole-program pass) — a lambda has no name of its own to publish a
//!   placeholder under until it's *already* bound, which is exactly the
//!   chicken-and-egg this restriction reflects.
//! - Calling a lambda *literal* directly (`(fn(a, b) { a + b })(1, 2)`) —
//!   only a named binding holding one can be called; see `grammar.pest`.
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
pub struct Subst(HashMap<TyVar, Ty>);

impl Subst {
    /// Follows variable chains to the current representative type,
    /// recursing into `Fn`'s parameter/return types so a partially-resolved
    /// function type reflects every binding made so far, not just its
    /// outermost shape.
    pub fn apply(&self, ty: &Ty) -> Ty {
        match ty {
            Ty::Var(v) => match self.0.get(v) {
                Some(next) => self.apply(next),
                None => ty.clone(),
            },
            Ty::Con(_) | Ty::Const(_) => ty.clone(),
            Ty::App(name, args) => Ty::App(name.clone(), args.iter().map(|a| self.apply(a)).collect()),
            Ty::Fn(params, ret) => {
                Ty::Fn(params.iter().map(|p| self.apply(p)).collect(), Box::new(self.apply(ret)))
            }
            Ty::Array(elem, size) => Ty::Array(Box::new(self.apply(elem)), Box::new(self.apply(size))),
        }
    }

    fn bind(&mut self, v: TyVar, ty: Ty) {
        self.0.insert(v, ty);
    }

    /// Must recurse into `Fn`'s components — a variable can occur *inside* a
    /// function type (`'a = ('a) -> Int`) just as easily as anywhere else;
    /// missing that recursion here would silently defeat the whole point of
    /// having an occurs check once lambdas are in the mix.
    fn occurs(&self, v: TyVar, ty: &Ty) -> bool {
        match self.apply(ty) {
            Ty::Var(v2) => v == v2,
            Ty::Con(_) | Ty::Const(_) => false,
            Ty::App(_, args) => args.iter().any(|a| self.occurs(v, a)),
            Ty::Fn(params, ret) => params.iter().any(|p| self.occurs(v, p)) || self.occurs(v, &ret),
            Ty::Array(elem, size) => self.occurs(v, &elem) || self.occurs(v, &size),
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
        (Ty::Var(v1), Ty::Var(v2)) if v1 == v2 => Ok(()),
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
        (Ty::App(n1, a1), Ty::App(n2, a2)) if n1 == n2 && a1.len() == a2.len() => {
            for (x, y) in a1.iter().zip(a2) {
                unify(subst, x, y)?;
            }
            Ok(())
        }
        (Ty::Array(e1, s1), Ty::Array(e2, s2)) => {
            unify(subst, e1, e2)?;
            unify(subst, s1, s2)
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
            TypeErrorKind::MissingFnBody { name } => {
                write!(f, "`{name}` has no body — a `fn` may only omit one if it's `extern`")
            }
            TypeErrorKind::MissingIntrinsicAttribute { name } => {
                write!(f, "`{name}` has no body and isn't `extern` — an algebra-impl method without a body must be")
            }
            TypeErrorKind::ExternFnCannotBeGeneric { name } => {
                write!(f, "`extern fn {name}` cannot be generic — only concrete, monomorphized signatures can cross a C-ABI boundary")
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
    pub span: Span,
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

pub(crate) fn free_vars(ty: &Ty, out: &mut HashSet<TyVar>) {
    match ty {
        Ty::Var(v) => {
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
    }
}

pub(crate) fn substitute(ty: &Ty, mapping: &HashMap<TyVar, Ty>) -> Ty {
    match ty {
        Ty::Var(v) => mapping.get(v).cloned().unwrap_or_else(|| ty.clone()),
        Ty::Con(_) | Ty::Const(_) => ty.clone(),
        Ty::App(name, args) => Ty::App(name.clone(), args.iter().map(|a| substitute(a, mapping)).collect()),
        Ty::Fn(params, ret) => {
            Ty::Fn(params.iter().map(|p| substitute(p, mapping)).collect(), Box::new(substitute(ret, mapping)))
        }
        Ty::Array(elem, size) => {
            Ty::Array(Box::new(substitute(elem, mapping)), Box::new(substitute(size, mapping)))
        }
    }
}

/// A `let` (never `let mut` — see module docs) is only generalizable when
/// its right-hand side is a syntactic *value*: nothing that could be an
/// aliased, later-mutated reference. Deliberately conservative — an ordinary
/// function call could in principle return something safely generalizable
/// too, but distinguishing that from one that can't requires effect
/// tracking this pass doesn't have.
fn is_syntactic_value(expr: &Expr) -> bool {
    matches!(
        expr.kind,
        ExprKind::NumberLit { .. } | ExprKind::ImaginaryLit { .. } | ExprKind::BoolLit(_) | ExprKind::Path(_) | ExprKind::Lambda { .. }
    )
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
        Ty::Con(_) | Ty::Const(_) => true,
        Ty::App(_, args) => args.iter().all(is_fully_concrete),
        Ty::Fn(params, ret) => params.iter().all(is_fully_concrete) && is_fully_concrete(ret),
        Ty::Array(elem, size) => is_fully_concrete(elem) && is_fully_concrete(size),
    }
}

/// Like `is_placeholder`, but recurses into `Ty::Fn` — a function type whose
/// parameter or return type is itself a placeholder is just as unresolved as
/// a bare one, and this is exactly the shape a lambda calling an undeclared
/// cross-function `fn` produces (`(t) -> <unresolved-call:add>`).
fn find_placeholder_name(ty: &Ty) -> Option<String> {
    match ty {
        Ty::Con(name) if name.starts_with('<') => Some(name.clone()),
        Ty::Con(_) | Ty::Var(_) | Ty::Const(_) => None,
        Ty::App(_, args) => args.iter().find_map(find_placeholder_name),
        Ty::Fn(params, ret) => params.iter().find_map(find_placeholder_name).or_else(|| find_placeholder_name(ret)),
        Ty::Array(elem, size) => find_placeholder_name(elem).or_else(|| find_placeholder_name(size)),
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
    /// Each const generic's own declared type (`i32`, `i64`, `bool`, ...),
    /// keyed by its own fresh var — populated once, in `fresh_generics_
    /// mapping`'s `Const` arm, for every const generic this `Infer` instance
    /// ever mints a fresh var for. Consulted by `check_pending_constraints`
    /// when a constraint's own type resolves to a bare `Ty::Const` (a const
    /// generic referenced as an ordinary value, e.g. a `for` loop bound) —
    /// without this, nothing could recover which *specific* declared width
    /// produced that value, since `Ty::Const` itself carries none. Threaded
    /// through `generalize`/`instantiate_with_mapping` exactly like
    /// `constraints` is, for the identical reason: a constraint carried into
    /// a `Scheme` and re-checked at some later instantiation needs this same
    /// width information to still be available then, not just at the
    /// original declaration site.
    const_widths: HashMap<TyVar, Ty>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumberDefault {
    Int,
    Float,
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
            quantified: HashSet::new(),
            pending_type_name_checks: Vec::new(),
            in_progress_methods: HashMap::new(),
            const_widths: HashMap::new(),
        }
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
    /// than a real type; also recorded now, in `self.const_widths`, since a
    /// const generic referenced as an ordinary value (a `for` loop bound,
    /// say) needs its own real declared width recoverable later, once
    /// resolved to a bare `Ty::Const` — see `check_pending_constraints`'s
    /// own `Ty::Const` bridge. The shared core `fn_generics_mapping`
    /// delegates to, also used directly by struct construction/field access
    /// (`ExprKind::StructLit`/`FieldAccess`), where there's no `FnDecl` to
    /// read a generics list off of, just `Registry::struct_generics` — a
    /// struct's own const generics get the same width tracking for free.
    fn fresh_generics_mapping(&mut self, generics: &[GenericParam], span: Span) -> HashMap<String, Ty> {
        let mapping = self.fresh_vars_for_generics(generics);
        for g in generics {
            match g {
                GenericParam::Type { name, bounds } => {
                    let ty = mapping[name].clone();
                    for bound in bounds {
                        self.constraints.push(Constraint { algebra: bound.clone(), tys: vec![ty.clone()], span });
                    }
                }
                GenericParam::Const { name, ty } => {
                    let width = self.ty_from_ast_mapped(ty, &mapping);
                    if let Ty::Var(v) = &mapping[name] {
                        self.const_widths.insert(*v, width);
                    }
                }
            }
        }
        mapping
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
        generics
            .iter()
            .map(|g| match g {
                GenericParam::Type { name, .. } => (name.clone(), self.vars.fresh()),
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
                GenericParam::Const { name, .. } => (name.clone(), self.vars.fresh()),
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
                // built from it in the same order) already establishes; a
                // `Const` generic on the algebra's own side consumes no
                // target slot (const-generic algebra parameters aren't
                // attempted — same scope this project's const-generics work
                // has everywhere else it hasn't reached yet).
                GenericParam::Type { name, .. } => target_ty_iter.next().map(|ty| (name.clone(), ty.clone())),
                GenericParam::Const { .. } => None,
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

        self.finish_fn(f, param_types, result?)
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
    /// Returns one entry per `fns`, each either the method's own final
    /// `(param_types, result)` or the `TypeError` that rejected it — the
    /// caller (`dump.rs`) reads `self.node_types` afterward for rendering,
    /// same as any other inference entry point; unlike `param_types` (a
    /// single, last-write-wins field on `Infer`, unsuited to more than one
    /// method sharing an instance), `node_types` is keyed by `NodeId` and
    /// already accumulates correctly across however many bodies this one
    /// `Infer` instance ends up walking.
    pub fn infer_inherent_impl_block(
        &mut self,
        outer: &Env,
        impl_generics: &[GenericParam],
        target: &Type,
        fns: &[FnDecl],
        fallback_span: Span,
    ) -> HashMap<String, Result<(Vec<Ty>, Ty), TypeError>> {
        let impl_mapping = self.fresh_generics_mapping(impl_generics, fallback_span);
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
                .and_then(|ty| self.check_pending_type_names().map(|()| ty));
            raw_results.insert(f.name.clone(), outcome);
        }

        if let Some(name) = &struct_name {
            for f in fns {
                self.in_progress_methods.remove(&(name.clone(), f.name.clone()));
            }
        }

        self.apply_defaults();
        // A constraint failure here is a property of the block's mutual
        // definition as a whole, not attributable to one specific member —
        // reported against every member whose own raw inference otherwise
        // succeeded (a raw-inference failure is already more specific and
        // is left alone), mirroring `callgraph.rs`'s identical choice.
        if let Err(e) = self.check_pending_constraints() {
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
        results
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
        self.check_pending_constraints()?;
        self.check_pending_type_names()?;

        // Fully re-resolve everything through the final substitution before
        // handing it back — `node_types`/`param_types` may have captured a
        // type before some later unification (or defaulting) pinned it down
        // further; callers should never need to know that and re-apply
        // `subst` themselves.
        self.param_types = param_types.iter().map(|t| self.subst.apply(t)).collect();
        let resolved_nodes: Vec<(NodeId, Ty)> =
            self.node_types.iter().map(|(id, t)| (*id, self.subst.apply(t))).collect();
        self.node_types = resolved_nodes.into_iter().collect();

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
    pub(crate) fn generalize(&mut self, env: &Env, ty: &Ty) -> Scheme {
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
                constraints.push(Constraint { algebra: c.algebra.clone(), tys: resolved_tys, span: c.span });
            }
        }

        // Any of the just-quantified vars that are const generics carry
        // their own declared width along too — same reasoning as the
        // constraint sweep just above (a bound checked later, once this
        // scheme is instantiated, needs the *real* width, not a guess; see
        // `check_pending_constraints`'s own `Ty::Const` bridge).
        let const_widths: HashMap<TyVar, Ty> =
            self.const_widths.iter().filter(|(v, _)| var_set.contains(v)).map(|(v, t)| (*v, t.clone())).collect();

        Scheme { vars, constraints, ty, const_widths }
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
                fv.iter()
                    .any(|v| ty_fv.contains(v))
                    .then(|| Constraint { algebra: c.algebra.clone(), tys: resolved_tys, span: c.span })
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
                self.const_widths.insert(*fresh, width.clone());
            }
        }
        (substitute(&scheme.ty, &mapping), mapping)
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
    /// Deliberately conservative, same as Rust's own default coherence
    /// check: two patterns count as overlapping purely by *shape* — no
    /// attempt at proving two impls' bound sets could never both be
    /// satisfied by the same concrete type (Rust doesn't attempt that in
    /// general either, short of specialization/negative-reasoning features
    /// this language doesn't have). Unifies against a throwaway `Subst`,
    /// same reasoning as `has_matching_impl`'s own `trial` — a shape
    /// collision here must never leave partial bindings behind for the next
    /// pair to trip over, and must never touch `self.subst` at all (there's
    /// no real query type involved, just two patterns compared to each
    /// other).
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
                        let fmt_targets = |ts: &[&Type]| ts.iter().map(|t| fmt_type(t)).collect::<Vec<_>>().join(", ");
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
        errors
    }

    /// Checks whether `algebra` has an `impl` whose target pattern(s) unify,
    /// positionally, against `query` — the shared engine behind
    /// `has_matching_impl` (single-type existence probe, `commit = false`)
    /// and `dispatch_algebra_call` (multi-type, `commit = true`); see each
    /// wrapper's own doc comment for what distinguishes the two postures and
    /// why both are needed. Candidates come from `Registry::all_impls`
    /// (every impl of `algebra`, single- or multi-target, generic or fully
    /// concrete alike — the piece `Registry` itself can't do this matching,
    /// it's deliberately just data, see its own module docs); a mismatched
    /// target-tuple length skips the candidate outright.
    ///
    /// For the overwhelmingly common single-target, fully-concrete case
    /// (`add(1, 2)` against `impl Ring<i32>`), `Registry::has_impl_named`'s
    /// plain `HashMap` lookup is tried first, before ever touching
    /// unification.
    ///
    /// Each candidate's own unification runs against a *cloned* `Subst`
    /// (`trial`), never `self.subst` directly — a rejected candidate must
    /// never leave partial bindings behind for the next one to trip over.
    /// Fresh variables for the candidate's own generics are minted via
    /// `fresh_vars_for_generics` specifically (not `fresh_generics_mapping`)
    /// for the same reason: no bound should become a *real*, persistent
    /// `Constraint` in `self.constraints` just because one candidate was
    /// tried and rejected — bounds are instead checked directly, recursively
    /// (via `has_matching_impl`): a structural match against `Complex<T>`
    /// alone isn't enough if `T`'s own resolved argument doesn't actually
    /// satisfy `T: Float`.
    ///
    /// `commit`, once a candidate's target(s) unify *and* its bounds are
    /// satisfied, decides whether that winning `trial` gets written back
    /// into `self.subst` for real. `has_matching_impl` never commits — a
    /// match there is only ever "does *some* impl apply", not a commitment
    /// to *which* one, so permanently binding `self.subst` to whichever
    /// candidate happened to be tried first would be arbitrary.
    /// `dispatch_algebra_call` *must* commit: a generic appearing only in an
    /// algebra fn's own *return* type (`C` in `fn mul(a: A, b: B) -> C;`) is
    /// never independently constrained by a call's own arguments the way
    /// `A`/`B` are — the only way to ever learn its concrete value is from a
    /// successful match's own bindings. Safe specifically because
    /// `check_no_overlapping_impls` already guarantees at most one impl's
    /// target pattern can ever coherently match a given query tuple — no
    /// "arbitrary pick among several" the way an uncommitted probe has to
    /// stay neutral about. Checking the whole tuple together, positionally,
    /// in one shared `trial`, also matters on its own (independent of
    /// commit): a per-parameter loop calling `has_matching_impl`
    /// independently, once per algebra generic, can't express "these two
    /// parameters must resolve to types sharing one common impl
    /// instantiation" — a shared impl-generic (`M` in `impl<T,N,M,K>
    /// MatMul<Matrix<T,N,M>, Matrix<T,M,K>, Matrix<T,N,K>>`) could get bound
    /// to two *different* concrete values across two independent checks and
    /// neither call would ever notice — found by direct testing: `a * b` on
    /// two mismatched-shape matrices type-checked with the call's own result
    /// left as a bare, totally unconstrained variable.
    fn match_impl(&mut self, algebra: &str, query: &[Ty], commit: bool) -> bool {
        if let [ty] = query {
            if self.registry.has_impl_named(algebra, &ty.to_string()) {
                return true;
            }
        }
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
                GenericParam::Type { name, bounds } => {
                    let Some(arg_ty) = mapping.get(name) else { return true };
                    let resolved_arg = trial.apply(arg_ty);
                    bounds.iter().all(|bound| self.has_matching_impl(bound, std::slice::from_ref(&resolved_arg)))
                }
                GenericParam::Const { .. } => true,
            });
            if bounds_satisfied {
                if commit {
                    self.subst = trial;
                }
                return true;
            }
        }
        false
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
        if self.match_impl(algebra, tys, false) {
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
    /// winning match's bindings into `self.subst` for real. See
    /// `match_impl`'s own doc comment for why commit is required here
    /// specifically (an output-only generic can only ever be learned this
    /// way) and why it's sound (`check_no_overlapping_impls` guarantees no
    /// more than one impl could ever coherently match).
    fn dispatch_algebra_call(&mut self, algebra: &str, tys: &[Ty]) -> bool {
        self.match_impl(algebra, tys, true)
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
            // Still abstract somewhere in the tuple — not everything is
            // known yet, so there's nothing coherent to check (mirrors
            // `infer_algebra_call`'s own "still abstract somewhere" gate for
            // the immediate-dispatch path — same posture, deferred further
            // rather than guessed at).
            if !resolved.iter().all(is_fully_concrete) {
                continue;
            }
            // A const generic referenced as an ordinary value (`for i in
            // 0..N`, say) can resolve here to a bare `Ty::Const` rather than
            // an ordinary `Ty::Con` — `has_matching_impl` has nothing to
            // check that against directly (impls are declared for real
            // types, never for one specific constant value). Bridge each
            // such element to its own const generic's declared width,
            // recovered from `self.const_widths` via `c.tys`' own
            // *original*, pre-resolution var (see that field's own doc
            // comment — threaded through `generalize`/`instantiate_with_
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
                            Ty::Var(v) => self.const_widths.get(v).cloned(),
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
            if !self.has_matching_impl(&c.algebra, &checkable) {
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
    fn ty_from_ast_mapped(&mut self, ty: &Type, mapping: &HashMap<String, Ty>) -> Ty {
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
        }
    }

    /// Resolves a const-value expression — an array type's size (`[f64;
    /// 4]`'s `4`, or `[T; N]`'s `N`) or a `<...>`-position const-generic
    /// argument (`Matrix<f64, 3, 3>`'s `3`, or a bool-valued one like
    /// `Grid<true>`'s `true`) alike, both the same shape of problem — to a
    /// `Ty`. A bare integer or bool literal becomes a resolved `Ty::Const`,
    /// a single-segment path matching a key in `mapping` becomes that
    /// const-generic's own (fresh, per-call-site) variable, exactly like a
    /// type-generic's bare reference resolves in `ty_from_ast_mapped` above.
    /// An operator call (`4+3`, `N*2`, ...) — the same desugared shape
    /// ordinary `+`/`*` already produce, see `ast.rs`'s own `Call` doc
    /// comment — recurses on both operands and, if *both* resolve to a
    /// `Ty::Const`, delegates the actual arithmetic to the standalone
    /// `const_eval` module (deliberately isolated there — see its own module
    /// doc comment — rather than folded in here, so it stays reusable
    /// wherever else constant folding is needed later). Anything else (an
    /// operand still abstract, or an operator `const_eval` doesn't know yet)
    /// is `None`, deferred as a placeholder by the caller rather than
    /// guessed at or hard-rejected. Deliberately *not* integer-only:
    /// whether the result actually needs to be an integer is up to the
    /// caller (`TypeKind::Array`'s own arm above pushes that constraint
    /// itself, since it's the one actual consumer that cares).
    fn const_value_from_expr(&mut self, value: &Expr, mapping: &HashMap<String, Ty>) -> Option<Ty> {
        match &value.kind {
            ExprKind::NumberLit { text, .. } => text.parse::<u64>().ok().map(|n| Ty::Const(ConstValue::Int(n))),
            ExprKind::BoolLit(b) => Some(Ty::Const(ConstValue::Bool(*b))),
            ExprKind::Path(p) if p.segments.len() == 1 => mapping.get(&p.segments[0]).cloned(),
            ExprKind::Call(path, _, args, ..) if path.segments.len() == 1 && args.len() == 2 => {
                let (Ty::Const(a), Ty::Const(b)) =
                    (self.const_value_from_expr(&args[0], mapping)?, self.const_value_from_expr(&args[1], mapping)?)
                else {
                    return None;
                };
                const_eval::eval_binop(&path.segments[0], a, b).map(Ty::Const)
            }
            _ => None,
        }
    }

    /// Resolves one explicit turbofish argument (`Matrix::<f64, 4, 4>`'s
    /// `f64`/`4`, `fibonacci::<f64>`'s `f64`) to a `Ty` — always against an
    /// *empty* generics mapping, unlike `ty_from_ast_mapped`'s other
    /// callers: a turbofish argument naming one of the *enclosing*
    /// function's own generic parameters by name (`fn g<U>() { f::<U>(x) }`)
    /// isn't resolved correctly by this — it isn't threaded through
    /// `infer_call`'s own signature — same "not attempted this increment,
    /// flagged rather than silently wrong" posture as everywhere else a real
    /// gap shows up in this file. The overwhelmingly common case (a concrete
    /// type/const, `f64`/`4`/`true`) is unaffected by this limitation.
    fn generic_arg_to_ty(&mut self, g: &GenericArg) -> Ty {
        match g {
            GenericArg::Type(t) => self.ty_from_ast(t),
            GenericArg::Const(e) => {
                self.const_value_from_expr(e, &HashMap::new()).unwrap_or_else(|| self.vars.fresh())
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
                    let value_ty = self.infer_expr(&env, value)?;
                    if let Some(annotated) = ty {
                        let declared = self.ty_from_ast(annotated);
                        self.unify_at(annotated.span, &declared, &value_ty)?;
                    }
                    // `let mut` is never generalized — see module docs (the
                    // ref-cell-polymorphism unsoundness this avoids).
                    let scheme = if !mutable && is_syntactic_value(value) {
                        self.generalize(&env, &value_ty)
                    } else {
                        Scheme::mono(value_ty)
                    };
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
                    let value_ty = self.infer_expr(&env, value)?;
                    self.unify_at(value.span, &target_ty, &value_ty)?;
                }
                StmtKind::Expr(e) => {
                    self.infer_expr(&env, e)?;
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
                        self.constraints.push(Constraint {
                            algebra: "Num".to_string(),
                            tys: vec![v.clone()],
                            span: expr.span,
                        });
                        let shape_algebra = if is_float { "Float" } else { "Int" };
                        self.constraints.push(Constraint {
                            algebra: shape_algebra.to_string(),
                            tys: vec![v.clone()],
                            span: expr.span,
                        });
                    }
                    Ok(v)
                }
            },
            // Complex literals aren't inferable yet (no `Complex<T>` in the
            // built-in signature table below) — deferred with everything else.
            ExprKind::ImaginaryLit { .. } => Ok(Ty::Con("<complex-not-yet-inferred>".to_string())),
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
            // A `while`/`for` loop's own value is always `()`, unconditionally
            // — same reasoning as an `if` with no `else` (see that arm's own
            // comment just above): the body might run zero times (`while`)
            // or its per-iteration result is discarded either way (there's
            // no `break value` mechanism), so nothing meaningful could ever
            // come out of evaluating one as an expression.
            ExprKind::While { cond, body } => {
                let cond_ty = self.infer_expr(env, cond)?;
                self.unify_at(cond.span, &Ty::Con("bool".to_string()), &cond_ty)?;
                self.infer_block(env, body)?;
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
                self.constraints.push(Constraint { algebra: "Int".to_string(), tys: vec![start_ty.clone()], span: start.span });
                let mut inner_env = env.clone();
                inner_env.insert(var.clone(), Scheme::mono(start_ty));
                // `infer_block` clones `inner_env` again internally — the
                // same cheap-clone tradeoff `Lambda`'s own handling above
                // already makes.
                self.infer_block(&inner_env, body)?;
                Ok(Ty::Con("()".to_string()))
            }
            ExprKind::MethodCall(base, name, args) => {
                let base_ty = self.infer_expr(env, base)?;
                let resolved_base = self.subst.apply(&base_ty);
                // Still abstract, or an already-unresolved placeholder
                // (chained off another not-yet-inferred expression) — same
                // "we don't know yet" posture `FieldAccess` already uses.
                if matches!(resolved_base, Ty::Var(_)) || is_placeholder(&resolved_base) {
                    return Ok(Ty::Con("<not-yet-inferred>".to_string()));
                }
                let struct_name = match &resolved_base {
                    Ty::Con(n) => n.clone(),
                    Ty::App(n, _) => n.clone(),
                    // A function value, array, or const-value — none of
                    // these have methods, rejected the same way
                    // `FieldAccess` rejects them for fields. `Ty::Var` is
                    // included only for exhaustiveness — already returned
                    // above, never actually reached here.
                    Ty::Fn(..) | Ty::Array(..) | Ty::Const(_) | Ty::Var(_) => {
                        return Err(TypeError {
                            span: expr.span,
                            kind: TypeErrorKind::NoSuchMethod {
                                struct_name: resolved_base.to_string(),
                                method: name.clone(),
                            },
                        });
                    }
                };
                // A (self- or, less commonly, sibling-triggered) recursive
                // call back into the method *currently* having its own body
                // inferred — see `in_progress_methods`'s own doc comment.
                // Reuses that enclosing invocation's own already-resolved
                // param types and return-type placeholder directly, instead
                // of re-deriving a fresh instantiation from the registry:
                // there's exactly one in-flight instantiation to recurse
                // into, the same one this call is already nested inside.
                if let Some((param_tys, ret_ty)) = self.in_progress_methods.get(&(struct_name.clone(), name.clone())).cloned() {
                    self.unify_at(base.span, &param_tys[0], &resolved_base)?;
                    for (pt, a) in param_tys[1..].iter().zip(args) {
                        let at = self.infer_expr(env, a)?;
                        self.unify_at(a.span, pt, &at)?;
                    }
                    return Ok(self.subst.apply(&ret_ty));
                }
                let Some(entry) = self.registry.inherent_method(&struct_name, name).cloned() else {
                    return Err(TypeError {
                        span: expr.span,
                        kind: TypeErrorKind::NoSuchMethod { struct_name, method: name.clone() },
                    });
                };
                // `base` fills the method's own first parameter — an
                // ordinary, explicit positional argument, not a magic
                // `self` (see `grammar.pest`'s `inherent_impl` comment for
                // why: this project doesn't have implicit-anything
                // elsewhere, no reason to invent one here).
                if entry.method.params.is_empty() || entry.method.params.len() != args.len() + 1 {
                    return Err(TypeError {
                        span: expr.span,
                        kind: TypeErrorKind::ArityMismatch {
                            name: name.clone(),
                            expected: entry.method.params.len(),
                            found: args.len() + 1,
                        },
                    });
                }
                // The impl block's own generics (`impl<T: Float> Vec2<T>`)
                // — fresh per call, bounds pushed as real `Constraint`s
                // exactly like an algebra impl's own (`fresh_generics_
                // mapping`). `target_ty` — built *from* `impl_mapping`, so
                // `Boxed<T>` becomes `App("Boxed", [impl_mapping["T"]])` —
                // is what actually pins those generics down once unified
                // against `resolved_base` below; a bare fresh var for
                // `param_tys[0]` would *also* end up correctly unified with
                // `resolved_base`, but as its own, disconnected variable,
                // never actually feeding back into `impl_mapping` at all —
                // a real bug, found by testing: a generic inherent method's
                // own return type (`T`, resolved through this same
                // `impl_mapping`) came back as a bare, still-unconstrained
                // variable instead of the concrete type `base` actually
                // has. Mirrors `infer_inherent_impl_fn_generic`'s own
                // identical fix for the exact same reason, on the
                // declaration side.
                let impl_mapping = self.fresh_generics_mapping(&entry.generics, expr.span);
                let target_ty = self.ty_from_ast_mapped(&entry.target, &impl_mapping);
                let param_tys = self.inherent_method_param_tys(&entry.method.params, &impl_mapping, &target_ty);
                self.unify_at(base.span, &param_tys[0], &resolved_base)?;
                for (pt, a) in param_tys[1..].iter().zip(args) {
                    let at = self.infer_expr(env, a)?;
                    self.unify_at(a.span, pt, &at)?;
                }
                // Unlike an algebra call (which always has a real declared
                // signature, return type included, to fall back on) or a
                // top-level `fn` call (whose return type was already
                // *inferred*, once, by the whole-program pass, then
                // generalized into a reusable `Scheme` — see `callgraph.rs`)
                // — dispatch here never re-runs the method's own body at the
                // call site (except for the recursive, `in_progress_methods`
                // case just above), so an inherent method with no explicit
                // `->` annotation has no return type available *anywhere*
                // else to report. Defaulting that to `()` would be silently
                // *wrong*, not just imprecise (the method might return
                // anything) — deferred as the usual "we don't know yet"
                // placeholder instead, same posture as every other
                // genuinely-unresolved case in this file. A method whose
                // return type actually matters at a call site needs an
                // explicit annotation; genuinely inferring it from the body
                // the way a top-level `fn` does — for a call to some *other*
                // unannotated method not already mid-inference (mutual
                // recursion between two separately-declared inherent
                // methods, neither self-recursive) — is a further increment
                // (would need inherent methods to go through the same
                // whole-program, generalize-once treatment `callgraph.rs`
                // gives top-level `fn`s — not attempted here).
                Ok(entry
                    .method
                    .ret
                    .as_ref()
                    .map(|t| self.ty_from_ast_mapped(t, &impl_mapping))
                    .unwrap_or_else(|| Ty::Con("<not-yet-inferred>".to_string())))
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
            ExprKind::ArrayRepeat { value, count } => {
                let elem_ty = self.infer_expr(env, value)?;
                let count_ty = self.infer_expr(env, count)?;
                Ok(Ty::Array(Box::new(elem_ty), Box::new(count_ty)))
            }
            ExprKind::Index(base, idx) => {
                let base_ty = self.infer_expr(env, base)?;
                let idx_ty = self.infer_expr(env, idx)?;
                // The index itself must be some integer type — doesn't need
                // to be a literal, just `Int`-constrained like any other
                // integer-typed value (`fibonacci`'s own `T: Int` bound uses
                // the exact same mechanism).
                self.constraints.push(Constraint { algebra: "Int".to_string(), tys: vec![idx_ty], span: idx.span });
                let resolved_base = self.subst.apply(&base_ty);
                match resolved_base {
                    // Still abstract, or already an unresolved placeholder
                    // (chained off another not-yet-inferred expression) —
                    // same "we don't know yet" posture as `FieldAccess`.
                    Ty::Var(_) => Ok(Ty::Con("<not-yet-inferred>".to_string())),
                    _ if is_placeholder(&resolved_base) => Ok(Ty::Con("<not-yet-inferred>".to_string())),
                    Ty::Array(elem, _) => Ok(self.subst.apply(&elem)),
                    // Concrete, resolved, and definitely not an array — a
                    // real type error (indexing an `i32`, a struct, ...),
                    // not something to defer. Built directly rather than via
                    // `unify_at` since we already know it can't match: an
                    // `Array` can only ever structurally unify with another
                    // `Array` or an unresolved `Var`, both already excluded
                    // above.
                    other => Err(TypeError {
                        span: expr.span,
                        kind: TypeErrorKind::Unify(UnifyError::Mismatch(
                            Ty::Array(Box::new(self.vars.fresh()), Box::new(self.vars.fresh())),
                            other,
                        )),
                    }),
                }
            }
            ExprKind::FieldAccess(base, name) => {
                let base_ty = self.infer_expr(env, base)?;
                let resolved = self.subst.apply(&base_ty);
                match &resolved {
                    // Still abstract (nothing pinned the base's type down
                    // yet) — same "we don't know, not yet a failure"
                    // posture as any other not-yet-inferred construct;
                    // `is_placeholder` covers an *already*-unresolved base
                    // (a field access chained off another not-yet-inferred
                    // expression) the same way.
                    Ty::Var(_) => Ok(Ty::Con("<not-yet-inferred>".to_string())),
                    Ty::Con(name2) if is_placeholder(&resolved) => {
                        let _ = name2;
                        Ok(Ty::Con("<not-yet-inferred>".to_string()))
                    }
                    // Non-generic struct (or any other bare concrete type —
                    // see the `None` arm below) — field's declared type
                    // needs no further mapping, it can't mention a generic
                    // parameter this struct doesn't have.
                    Ty::Con(struct_name) => match self.registry.struct_fields(struct_name) {
                        Some(fields) => match fields.iter().find(|f| &f.name == name) {
                            Some(field) => Ok(self.ty_from_ast(&field.ty)),
                            None => Err(TypeError {
                                span: expr.span,
                                kind: TypeErrorKind::NoSuchField {
                                    struct_name: struct_name.clone(),
                                    field: name.clone(),
                                },
                            }),
                        },
                        // A concrete, known type that simply isn't a struct
                        // at all (`(1).foo`) — genuinely has no fields,
                        // rejected the same way as a struct missing this
                        // specific one.
                        None => Err(TypeError {
                            span: expr.span,
                            kind: TypeErrorKind::NoSuchField { struct_name: struct_name.clone(), field: name.clone() },
                        }),
                    },
                    // A generic struct, already instantiated at some
                    // concrete (or still-abstract, doesn't matter — could be
                    // `Complex<'t9>`) set of type arguments — map the
                    // struct's own declared generic parameter names to
                    // *these* arguments (positionally, `App`'s own
                    // established convention) before resolving the field's
                    // declared type, so `real: T` on `Complex<f64>` reads
                    // back as `f64`, not the literal, meaningless name `T`.
                    Ty::App(struct_name, type_args) => match self.registry.struct_fields(struct_name) {
                        Some(fields) => match fields.iter().find(|f| &f.name == name).cloned() {
                            Some(field) => {
                                let mapping = self.zip_struct_generics(struct_name, type_args);
                                Ok(self.ty_from_ast_mapped(&field.ty, &mapping))
                            }
                            None => Err(TypeError {
                                span: expr.span,
                                kind: TypeErrorKind::NoSuchField {
                                    struct_name: struct_name.clone(),
                                    field: name.clone(),
                                },
                            }),
                        },
                        None => Err(TypeError {
                            span: expr.span,
                            kind: TypeErrorKind::NoSuchField { struct_name: struct_name.clone(), field: name.clone() },
                        }),
                    },
                    // Neither has fields — a function value or an array
                    // (indexing, not field access, is how you reach into an
                    // array) rejected the same way as any other fieldless
                    // concrete type. `Const` can't actually reach here in
                    // practice (nothing produces one as an *expression's*
                    // type, only inside another type's size slot), but is
                    // handled the same way for exhaustiveness rather than a
                    // `todo!()`/panic waiting to be hit by a future caller.
                    Ty::Fn(..) | Ty::Array(..) | Ty::Const(_) => Err(TypeError {
                        span: expr.span,
                        kind: TypeErrorKind::NoSuchField { struct_name: resolved.to_string(), field: name.clone() },
                    }),
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
                    let ty = match &p.ty {
                        Some(t) => self.ty_from_ast(t),
                        None => self.vars.fresh(),
                    };
                    inner_env.insert(p.name.clone(), Scheme::mono(ty.clone()));
                    param_tys.push(ty);
                }
                // `infer_block` clones `inner_env` again internally — fine,
                // it's the same cheap-clone tradeoff already made everywhere
                // else (see `infer_block`'s own doc comment).
                let body_ty = self.infer_block(&inner_env, body)?;
                if let Some(r) = ret {
                    let declared = self.ty_from_ast(r);
                    self.unify_at(r.span, &declared, &body_ty)?;
                }
                Ok(Ty::Fn(param_tys, Box::new(body_ty)))
            }
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
            // (`add::<f64>(a, b)`) isn't wired up — algebra calls resolve
            // their own generic entirely from the argument types
            // (`infer_algebra_call`), with no `Scheme`/`instantiate` step
            // for a turbofish to hook into the same way a `let`-bound/
            // top-level fn call has below. Silently ignored rather than
            // rejected, matching this file's permissive-by-omission posture
            // for real gaps — not attempted this increment.
            return self.infer_algebra_call(call_span, algebra, &name, &arg_tys, args);
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
        args: &[Expr],
    ) -> Result<Ty, TypeError> {
        let sig = self
            .registry
            .fn_sig(algebra, name)
            .cloned()
            .unwrap_or_else(|| unreachable!("registry reported `{algebra}` declares `{name}`, but fn_sig lookup failed"));

        if sig.params.len() != args.len() {
            return Err(TypeError {
                span: call_span,
                kind: TypeErrorKind::ArityMismatch { name: name.to_string(), expected: sig.params.len(), found: args.len() },
            });
        }

        let generics = self.registry.generics(algebra).to_vec();
        let mapping: HashMap<String, Ty> = generics
            .iter()
            .filter_map(|g| match g {
                GenericParam::Type { name, .. } => Some((name.clone(), self.vars.fresh())),
                GenericParam::Const { .. } => None,
            })
            .collect();

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

        for (pt, (at, a)) in param_tys.iter().zip(arg_tys.iter().zip(args)) {
            self.unify_at(a.span, pt, at)?;
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
        for g in &generics {
            let GenericParam::Type { name, .. } = g else { continue };
            let fresh = &mapping[name];
            let resolved = self.subst.apply(fresh);
            if matches!(fresh, Ty::Var(v) if param_free_vars.contains(v)) {
                gating.push(resolved.clone());
            }
            resolved_generics.push(resolved);
        }

        if gating.iter().any(is_placeholder) {
            // At least one *input* generic is an "unknown" — a placeholder
            // for something not type-inferred yet. Deferred below, same as
            // the "still abstract" case — one constraint holding the *whole*
            // tuple together (see `Constraint`'s own doc comment), checked
            // coherently by `check_pending_constraints` once every element
            // is concrete, same engine `match_impl`'s immediate-dispatch
            // path already uses.
            self.constraints.push(Constraint { algebra: algebra.to_string(), tys: resolved_generics, span: call_span });
        } else if gating.iter().all(is_fully_concrete) {
            // Ready: every *input* generic is known, so dispatch can
            // actually run — commits the match's own bindings for real
            // (see `dispatch_algebra_call`'s own doc comment for why that's
            // sound here specifically), which is what lets an
            // output-only generic like `C` end up resolved at all.
            if !self.dispatch_algebra_call(algebra, &resolved_generics) {
                let ty = resolved_generics.iter().map(Ty::to_string).collect::<Vec<_>>().join(", ");
                return Err(TypeError { span: call_span, kind: TypeErrorKind::MissingImpl { algebra: algebra.to_string(), ty } });
            }
        } else {
            // Still abstract somewhere among the *input* generics (a
            // generic caller, or a `Complex<'t9>` whose own argument isn't
            // pinned down yet) — defer, one constraint for the whole tuple:
            // either `generalize` migrates it into an enclosing `let`'s
            // scheme, or `check_pending_constraints` catches it once
            // `infer_fn` finishes, whichever comes first.
            self.constraints.push(Constraint { algebra: algebra.to_string(), tys: resolved_generics, span: call_span });
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
        for (var, default) in std::mem::take(&mut self.pending_defaults) {
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
            // ordinary numeric-literal defaulting.
            if self.const_widths.contains_key(&root) {
                continue;
            }
            let concrete = match default {
                NumberDefault::Int => "i32",
                NumberDefault::Float => "f32",
            };
            unify(&mut self.subst, &Ty::Var(root), &Ty::Con(concrete.to_string()))
                .expect("defaulting an unbound, non-quantified variable can't fail");
        }
    }
}

