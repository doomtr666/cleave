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
use std::collections::HashMap;

#[derive(Default)]
pub struct Registry {
    algebras: HashMap<String, AlgebraEntry>,
    structs: HashMap<String, StructEntry>,
}

struct StructEntry {
    generics: Vec<GenericParam>,
    fields: Vec<Field>,
}

struct AlgebraEntry {
    generics: Vec<GenericParam>,
    sigs: Vec<FnSig>,
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
    #[allow(dead_code)]
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
                        AlgebraItemKind::Axiom(_) => None,
                    })
                    .collect();
                algebras.entry(d.name.clone()).or_insert_with(|| AlgebraEntry {
                    generics: d.generics.clone(),
                    sigs,
                    impls: HashMap::new(),
                });
            }
        }

        for item in &program.items {
            if let ItemKind::Impl(d) = &item.kind {
                let entry = algebras.entry(d.algebra.clone()).or_insert_with(|| AlgebraEntry {
                    generics: Vec::new(),
                    sigs: Vec::new(),
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
                let key = format!("{}{}", fmt_type(&d.target), fmt_generics(&d.generics));
                entry.impls.insert(
                    key,
                    ImplEntry { generics: d.generics.clone(), target: d.target.clone(), fns: d.fns.clone() },
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

        Registry { algebras, structs }
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

    /// Every impl of `algebra` whose *own* target is a pattern rather than a
    /// concrete type (`impl<T: Float> Ring<Complex<T>>` — has generic
    /// parameters of its own, distinct from the algebra's own `<T>`), each
    /// as `(generics, target)` — for `Infer::has_matching_impl`'s real,
    /// unification-based matching. Excludes the common non-generic case
    /// entirely; those are still served by the plain, fast `has_impl_named`.
    pub fn generic_impls(&self, algebra: &str) -> Vec<(&[GenericParam], &Type)> {
        self.algebras
            .get(algebra)
            .map(|e| {
                e.impls
                    .values()
                    .filter(|i| !i.generics.is_empty())
                    .map(|i| (i.generics.as_slice(), &i.target))
                    .collect()
            })
            .unwrap_or_default()
    }
}
