//! Queryable index over a merged `Program`'s `algebra`/`impl`/`struct`
//! declarations — "does algebra `Ring` have an `impl` for concrete type
//! `Vec2`", "which algebras declare a `add(T, T) -> T`-shaped signature",
//! "what fields does `struct Vec2` declare". Built from the *already-merged*
//! output of `driver::merge_programs` (one logical `AlgebraDecl`/`ImplDecl`
//! per name — see `driver.rs`), not something that itself merges fragments.
//! Structs are never fragmented across files at all (`merge_programs`
//! rejects a duplicate struct name outright), so there's nothing to merge
//! for them — just an index over the already-unique declarations.
//!
//! This is deliberately just the data structure and the query surface — no
//! constraint generation/checking lives here (that's `infer.rs`'s job,
//! resolving an operator call against `algebras_with_fn`/`has_impl`, or a
//! struct literal/field access against `struct_fields`).

use crate::ast::*;
use crate::print::{fmt_generics, fmt_type};
use std::collections::{HashMap, HashSet};

#[derive(Default)]
pub struct Registry {
    algebras: HashMap<String, AlgebraEntry>,
    structs: HashMap<String, StructEntry>,
    /// Struct name -> method name -> entry. No candidate/ambiguity handling
    /// needed the way algebra dispatch has (`algebras_with_fn`,
    /// `AmbiguousOperator`): `driver::merge_programs`' own duplicate-method
    /// detection already guarantees at most one method of a given name
    /// exists per struct, so a lookup here is either "the" method or none —
    /// see `Infer`'s own method-call handling.
    inherent_methods: HashMap<String, HashMap<String, InherentMethodEntry>>,
}

struct StructEntry {
    generics: Vec<GenericParam>,
    fields: Vec<Field>,
}

#[derive(Clone)]
pub struct InherentMethodEntry {
    /// The *impl block's* own generics (`impl<T> Vec2<T> { ... }`) — not
    /// necessarily spelled the same as the struct's own declared generic
    /// names; the impl's own `target` (below) is what actually establishes
    /// the correspondence, positionally, the same way an algebra impl's
    /// target already does for the algebra's own generics.
    pub generics: Vec<GenericParam>,
    pub target: Type,
    pub method: FnDecl,
}

struct AlgebraEntry {
    generics: Vec<GenericParam>,
    /// Other algebras this one requires (`algebra Int<T> : Num { ... }`) —
    /// see `Registry::algebra_bounds`'s own doc comment for what this means
    /// and where it's actually enforced (not here — `Registry` stays just
    /// data, see the module doc).
    bounds: Vec<String>,
    sigs: Vec<FnSig>,
    /// Every `axiom` declared directly on this algebra (`algebra Ring<T> {
    /// axiom add_commutative(a, b): add(a, b) == add(b, a); }`) — trusted,
    /// unverified assertions over the algebra's own generic `T` (`doc/
    /// hld.md`'s own "v1 trust model": no per-concrete-impl soundness gate,
    /// an axiom holds for every `T` a `Ring` impl exists for, exactly like
    /// a Rust trait impl is trusted rather than proven). Consumed by a later
    /// e-graph rewriting pass, once one exists; retained here purely as data
    /// — same "just data, no validation" stance the rest of this module
    /// already takes (see its own doc comment).
    axioms: Vec<AxiomDecl>,
    /// Every `derivative` rule declared directly on this algebra (`doc/
    /// backlog-done.md`'s own "Auto-diff v1" entry) — mirrors `axioms`
    /// exactly: trusted, unvalidated data, consumed by `egraph.rs::
    /// derivative_rule_rewrites` to build real e-graph `Rewrite`s per
    /// reached concrete type.
    derivative_rules: Vec<DerivativeRuleDecl>,
    /// Keyed by the target type's canonical string (`fmt_type`) — same
    /// grouping key `driver.rs` uses to merge `impl` fragments.
    impls: HashMap<String, ImplEntry>,
}

struct ImplEntry {
    /// The impl's *own* generics (`impl<T: Float> Ring<Complex<T>>`) — empty
    /// for the overwhelmingly common non-generic case (`impl Ring<i32>`).
    /// Non-empty means the target below isn't a *concrete* type at all, just
    /// a *pattern*; matching a query type against it needs real unification
    /// (`Infer::has_matching_impl`), which is why this needs exposing at
    /// all — `has_impl_named`'s plain string-key lookup can only ever
    /// recognize an *exact* previously-declared spelling.
    generics: Vec<GenericParam>,
    target: Type,
    /// Second and later targets, for a heterogeneous algebra
    /// (`algebra MatMul<A, B, C>`) — empty for every single-generic algebra
    /// (i.e. almost always). See `ImplDecl::extra_targets`'s own doc
    /// comment for why this stays a separate field rather than folding
    /// `target` into a single always-a-`Vec` shape.
    extra_targets: Vec<Type>,
    fns: Vec<FnDecl>,
}

impl Registry {
    pub fn build(program: &Program) -> Self {
        let mut algebras: HashMap<String, AlgebraEntry> = HashMap::new();

        for item in &program.items {
            if let ItemKind::Algebra(d) = &item.kind {
                let sigs = d
                    .items
                    .iter()
                    .filter_map(|ai| match &ai.kind {
                        AlgebraItemKind::FnSig(sig) => Some(sig.clone()),
                        AlgebraItemKind::Axiom(_) | AlgebraItemKind::DerivativeRule(_) => None,
                    })
                    .collect();
                let axioms = d
                    .items
                    .iter()
                    .filter_map(|ai| match &ai.kind {
                        AlgebraItemKind::Axiom(axiom) => Some(axiom.clone()),
                        AlgebraItemKind::FnSig(_) | AlgebraItemKind::DerivativeRule(_) => None,
                    })
                    .collect();
                let derivative_rules = d
                    .items
                    .iter()
                    .filter_map(|ai| match &ai.kind {
                        AlgebraItemKind::DerivativeRule(dr) => Some(dr.clone()),
                        AlgebraItemKind::FnSig(_) | AlgebraItemKind::Axiom(_) => None,
                    })
                    .collect();
                algebras.entry(d.name.clone()).or_insert_with(|| AlgebraEntry {
                    generics: d.generics.clone(),
                    bounds: d.bounds.clone(),
                    sigs,
                    axioms,
                    derivative_rules,
                    impls: HashMap::new(),
                });
            }
        }

        for item in &program.items {
            if let ItemKind::Impl(d) = &item.kind {
                let entry = algebras.entry(d.algebra.clone()).or_insert_with(|| AlgebraEntry {
                    generics: Vec::new(),
                    bounds: Vec::new(),
                    sigs: Vec::new(),
                    axioms: Vec::new(),
                    derivative_rules: Vec::new(),
                    impls: HashMap::new(),
                });
                // `fmt_type(target)` alone would collide two *different*
                // generic impls sharing the same bare target shape but
                // different bounds (`impl<T: Float> Ring<Complex<T>>` vs.
                // `impl<T: Ord> Ring<Complex<T>>` both stringify as
                // `Complex<T>`) — found via testing (an overlap-detection
                // test lost one of its two impls entirely, silently, before
                // the check ever ran). `fmt_generics` is empty for the
                // overwhelmingly common non-generic case, so this key is
                // identical to the old plain `fmt_type` one wherever
                // `has_impl_named`'s fast lookup actually depends on it.
                // `extra_targets` folded in the same way, for the same
                // reason — two heterogeneous impls sharing a first target
                // but differing afterward must stay distinct too.
                let extra_targets_key: String = d.extra_targets.iter().map(fmt_type).collect();
                let key = format!("{}{}{}", fmt_type(&d.target), extra_targets_key, fmt_generics(&d.generics));
                entry.impls.insert(
                    key,
                    ImplEntry {
                        generics: d.generics.clone(),
                        target: d.target.clone(),
                        extra_targets: d.extra_targets.clone(),
                        fns: d.fns.clone(),
                    },
                );
            }
        }

        // Structs are never fragmented across files the way `algebra`/`impl`
        // are (`driver::merge_programs` rejects a duplicate struct name
        // outright) — nothing to merge here, just index the already-unique
        // declarations.
        let mut structs: HashMap<String, StructEntry> = HashMap::new();
        for item in &program.items {
            if let ItemKind::Struct(d) = &item.kind {
                structs.insert(d.name.clone(), StructEntry { generics: d.generics.clone(), fields: d.fields.clone() });
            }
        }

        let mut inherent_methods: HashMap<String, HashMap<String, InherentMethodEntry>> = HashMap::new();
        for item in &program.items {
            if let ItemKind::InherentImpl(d) = &item.kind {
                // Only a bare/generic *path* target names a struct at all —
                // an inherent impl on, say, an array type has nowhere to
                // register itself usefully (nothing could ever dispatch a
                // method call to it, since dispatch always starts from a
                // resolved struct name); silently unindexed rather than a
                // hard error, matching this file's "just data, no
                // validation" stance (a real diagnostic for this, if
                // wanted, belongs in `infer.rs` alongside its other
                // `pending_type_name_checks`-style checks, not here).
                let TypeKind::Path(p, _) = &d.target.kind else { continue };
                let struct_name = p.segments.join("::");
                for f in &d.fns {
                    inherent_methods.entry(struct_name.clone()).or_default().insert(
                        f.name.clone(),
                        InherentMethodEntry { generics: d.generics.clone(), target: d.target.clone(), method: f.clone() },
                    );
                }
            }
        }

        Registry { algebras, structs, inherent_methods }
    }

    /// Does `algebra` have an `impl` for this concrete target type? String
    /// comparison against `fmt_type`, same canonicalization `driver.rs`
    /// already uses for merging — not a structural/generic-aware match.
    pub fn has_impl(&self, algebra: &str, target: &Type) -> bool {
        self.has_impl_named(algebra, &fmt_type(target))
    }

    /// Same check as `has_impl`, keyed directly by the type's canonical
    /// name string rather than an AST `Type` node — for callers (`infer.rs`)
    /// that only have their own internal type representation at hand, with
    /// no AST node (and no `NodeId`/`Span` to invent one from) to point to.
    pub fn has_impl_named(&self, algebra: &str, type_name: &str) -> bool {
        self.algebras.get(algebra).is_some_and(|e| e.impls.contains_key(type_name))
    }

    /// Names of every declared algebra that has a `fn` signature named
    /// `fn_name` with exactly `arity` parameters — the candidate set an
    /// unqualified operator call (`add`) resolves against. More than one
    /// candidate is a real ambiguity (see conversation notes: this is not a
    /// "someone's trying to override an existing algebra" signal — two
    /// independent, legitimately-scoped algebras can both declare their own
    /// `add` — it's an ordinary name collision resolved like Rust's
    /// ambiguous trait methods: reject, ask for explicit qualification).
    /// Resolving that call is still the next increment's job, not this
    /// method's — this only reports the candidate set.
    pub fn algebras_with_fn(&self, fn_name: &str, arity: usize) -> Vec<&str> {
        self.algebras
            .iter()
            .filter(|(_, entry)| entry.sigs.iter().any(|s| s.name == fn_name && s.params.len() == arity))
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// The signature an algebra declares for `fn_name`, if any — for
    /// checking argument types against the algebra's own declared
    /// parameter/return types.
    pub fn fn_sig(&self, algebra: &str, fn_name: &str) -> Option<&FnSig> {
        self.algebras.get(algebra)?.sigs.iter().find(|s| s.name == fn_name)
    }

    /// The algebra's own generic parameters (`<T>` in `algebra Ring<T>`) —
    /// needed to instantiate a declared signature's `T`-typed parameters
    /// with fresh inference type variables, rather than treating `T` as a
    /// bare concrete type named `"T"`.
    pub fn generics(&self, algebra: &str) -> &[GenericParam] {
        self.algebras.get(algebra).map(|e| e.generics.as_slice()).unwrap_or(&[])
    }

    pub fn has_algebra(&self, algebra: &str) -> bool {
        self.algebras.contains_key(algebra)
    }

    /// Other algebras `algebra` itself requires (`algebra Int<T> : Num {
    /// ... }` — `Registry::generics(Int)` returns `bounds: ["Num"]`). Mirrors
    /// a `GenericParam::Type`'s own `bounds` field, one level up: instead of
    /// constraining a caller's generic parameter, this constrains *every*
    /// type any `impl` of `Int` is ever declared for — the actual check
    /// (`Int<i32>` existing implies `Num<i32>` must too) lives in
    /// `Infer::match_impl`, not here (`Registry` stays just data, see the
    /// module doc).
    pub fn algebra_bounds(&self, algebra: &str) -> &[String] {
        self.algebras.get(algebra).map(|e| e.bounds.as_slice()).unwrap_or(&[])
    }

    /// Every `axiom` declared on `algebra` — see `AlgebraEntry::axioms`'s own
    /// doc comment. Empty, not missing, for an algebra with none (mirrors
    /// `algebra_bounds`'s own convention just above) and for an unknown
    /// algebra name entirely.
    pub fn axioms(&self, algebra: &str) -> &[AxiomDecl] {
        self.algebras.get(algebra).map(|e| e.axioms.as_slice()).unwrap_or(&[])
    }

    /// Every `derivative` rule declared on `algebra` — see `AlgebraEntry::
    /// derivative_rules`'s own doc comment.
    pub fn derivative_rules(&self, algebra: &str) -> &[DerivativeRuleDecl] {
        self.algebras.get(algebra).map(|e| e.derivative_rules.as_slice()).unwrap_or(&[])
    }

    /// The reverse of `algebra_bounds`: every algebra that names `algebra`
    /// among its *own* bounds (`Int` and `Float` both, for `algebras_bounded_
    /// by("Num")`, once `algebra Int<T> : Num` / `algebra Float<T> : Num`
    /// exist) — what `Infer::has_matching_impl` walks to check "does some
    /// *other*, more specific algebra's impl count as this one too".
    pub fn algebras_bounded_by<'a>(&'a self, algebra: &'a str) -> impl Iterator<Item = &'a str> + 'a {
        self.algebras.iter().filter(move |(_, e)| e.bounds.iter().any(|b| b == algebra)).map(|(name, _)| name.as_str())
    }

    /// Every declared algebra's name — used by `Infer::check_no_overlapping_impls`
    /// to sweep the whole registry once, rather than being told which
    /// algebras exist by some other, already-scoped caller.
    pub fn algebra_names(&self) -> impl Iterator<Item = &str> {
        self.algebras.keys().map(String::as_str)
    }

    pub fn has_struct(&self, name: &str) -> bool {
        self.structs.contains_key(name)
    }

    /// A declared struct's own fields, in declaration order — `None` if
    /// `name` doesn't name a known struct at all (distinct from `Some(&[])`,
    /// a genuinely empty struct).
    /// Every struct name this registry knows about — used by `egraph.rs`'s
    /// own `derivative-independent-zero` rule to snapshot, once (`Applier`
    /// implementations must be `Send + Sync + 'static`, so they can't hold
    /// a borrowed `&Registry` themselves), which single-field pack-generic
    /// structs (`linalg::Tensor`'s own shape) it knows how to build a
    /// same-shaped zero for.
    pub fn struct_names(&self) -> impl Iterator<Item = &str> {
        self.structs.keys().map(String::as_str)
    }

    pub fn struct_fields(&self, name: &str) -> Option<&[Field]> {
        self.structs.get(name).map(|e| e.fields.as_slice())
    }

    /// A declared struct's own generic parameters (`<T>` in `struct
    /// Vec2<T>`) — used to map a field's declared type (which may mention
    /// one of these names) to a real type, either fresh (construction) or
    /// the concrete argument a particular value was built with (field
    /// access) — see `infer.rs`'s `StructLit`/`FieldAccess` handling.
    pub fn struct_generics(&self, name: &str) -> &[GenericParam] {
        self.structs.get(name).map(|e| e.generics.as_slice()).unwrap_or(&[])
    }

    /// The struct's own inherent method named `method_name`, if any — see
    /// `InherentMethodEntry`'s own doc comment for why this is a plain
    /// lookup, no ambiguity/candidate handling.
    pub fn inherent_method(&self, struct_name: &str, method_name: &str) -> Option<&InherentMethodEntry> {
        self.inherent_methods.get(struct_name)?.get(method_name)
    }

    /// Every *single-target* impl of `algebra` whose own target is a
    /// pattern rather than a concrete type (`impl<T: Float>
    /// Ring<Complex<T>>` — has generic parameters of its own, distinct from
    /// the algebra's own `<T>`), each as `(generics, target)` — for
    /// `Infer::has_matching_impl`'s real, unification-based matching.
    /// Excludes the common non-generic case entirely (still served by the
    /// plain, fast `has_impl_named`) *and* excludes every multi-target,
    /// heterogeneous impl (`extra_targets` non-empty) — `has_matching_impl`
    /// only ever checks *one* type against *one* pattern, so matching just
    /// a heterogeneous impl's own first target in isolation would be
    /// structurally meaningless (it says nothing about whether the *other*
    /// targets could also be satisfied); those go through
    /// `multi_target_impls`/`Infer::has_matching_impl_multi` instead, which
    /// checks every target together, coherently.
    pub fn generic_impls(&self, algebra: &str) -> Vec<(&[GenericParam], &Type)> {
        self.algebras
            .get(algebra)
            .map(|e| {
                e.impls
                    .values()
                    .filter(|i| !i.generics.is_empty() && i.extra_targets.is_empty())
                    .map(|i| (i.generics.as_slice(), &i.target))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// *Every* impl of `algebra`, uniformly, each as `(generics, all
    /// targets in declaration order)` — single-target or heterogeneous,
    /// generic or fully concrete alike, no filtering at all (unlike
    /// `generic_impls`, which deliberately excludes the cases already
    /// served by the fast string-keyed `has_impl_named`). For
    /// `Infer::dispatch_algebra_call`'s own real, *committing* dispatch:
    /// even a fully concrete, non-generic impl needs real unification
    /// (not just a string-equality check) whenever the *query* side still
    /// has an unresolved slot of its own — a generic appearing only in an
    /// algebra fn's return type, never independently pinned by any
    /// argument, is exactly that case; unifying a `Var` against a known
    /// concrete target is what actually binds it.
    pub fn all_impls(&self, algebra: &str) -> Vec<(&[GenericParam], Vec<&Type>)> {
        self.algebras
            .get(algebra)
            .map(|e| {
                e.impls
                    .values()
                    .map(|i| {
                        let targets: Vec<&Type> =
                            std::iter::once(&i.target).chain(i.extra_targets.iter()).collect();
                        (i.generics.as_slice(), targets)
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Every concrete type name known to satisfy `algebra` — direct impls,
    /// plus every type reachable transitively through `algebras_bounded_by`
    /// (the same bound-inheritance relationship `Infer::has_matching_impl`
    /// already walks for a *single* type probe, here built as a set instead
    /// — needed so `Infer::generalize`'s own scheme-satisfiability check can
    /// intersect two algebras' own candidate sets directly). `Num` itself
    /// has zero *direct* impls anywhere (`stdlib/num/num.cleave`'s own
    /// comment: every concrete numeric type reaches it only through `Int`/
    /// `Float`'s own bound) — without the recursive half here, every
    /// `Num`-constrained variable would come back with an empty candidate
    /// set and look falsely unsatisfiable.
    ///
    /// Deliberately narrow, matching `has_matching_impl`'s own documented
    /// gaps: only a non-generic, single-target, bare (no generic arguments
    /// of its own) impl counts — a generic impl (`impl<T> Ring<Complex<T>>`)
    /// contributes no *one* fixed concrete name, and an algebra satisfied
    /// only via its own *forward*-aggregate bounds (two or more bounds,
    /// neither individually implemented — see `has_matching_impl`'s own
    /// "forward aggregate" case) isn't attempted here either — not needed
    /// for `Int`/`Float`/`Num`, the only shapes that exist today.
    pub fn candidates_for(&self, algebra: &str) -> HashSet<String> {
        self.candidates_for_inner(algebra, &mut HashSet::new())
    }

    /// `visited` guards against a cyclic bound declaration (`algebra A : B`,
    /// `algebra B : A`) looping forever — same reasoning, same shape, as
    /// `Infer::has_matching_impl_inherited`'s own identical guard.
    fn candidates_for_inner<'a>(&'a self, algebra: &'a str, visited: &mut HashSet<&'a str>) -> HashSet<String> {
        let mut out = HashSet::new();
        if !visited.insert(algebra) {
            return out;
        }
        for (generics, targets) in self.all_impls(algebra) {
            let [target] = targets.as_slice() else { continue };
            if !generics.is_empty() {
                continue;
            }
            if let TypeKind::Path(path, args) = &target.kind {
                if args.is_empty() {
                    out.insert(path.segments.join("::"));
                }
            }
        }
        for other in self.algebras_bounded_by(algebra) {
            out.extend(self.candidates_for_inner(other, visited));
        }
        out
    }

    /// Like `all_impls`, but also hands back each impl's own declared `fn`s
    /// — needed by `monomorphize.rs`, the one consumer that actually needs
    /// to specialize an impl method's own *body*, not just match its target
    /// pattern the way dispatch (`Infer::match_impl`) does. A separate
    /// method rather than widening `all_impls` itself: `match_impl` runs on
    /// every algebra-dispatched call during ordinary inference, and has no
    /// use for the extra `&[FnDecl]` slice it would otherwise have to carry
    /// around for nothing.
    pub fn all_impls_with_fns(&self, algebra: &str) -> Vec<(&[GenericParam], Vec<&Type>, &[FnDecl])> {
        self.algebras
            .get(algebra)
            .map(|e| {
                e.impls
                    .values()
                    .map(|i| {
                        let targets: Vec<&Type> =
                            std::iter::once(&i.target).chain(i.extra_targets.iter()).collect();
                        (i.generics.as_slice(), targets, i.fns.as_slice())
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}
