//! Multi-file crate assembly: parses each file into its own `Program`, then
//! merges them into one logical `Program`. `algebra`/`impl` fragments spread
//! across multiple files (partial-declaration style — think C#'s `partial
//! class`, or Rust's own inherent `impl` blocks, which are already merged
//! this way) are unioned into a single declaration, regardless of which
//! file is processed first: global scope is not order-dependent.
//!
//! `struct` and free `fn` items are *not* partial — exactly one declaration
//! per name, anywhere in the crate. Only `algebra`/`impl` were called out as
//! needing the "extremely heavy, one file per axiom" workflow.
//!
//! Conflict identity is `(name, parameter types)` — ordinary overload
//! resolution: two `add`s with different parameter types coexist; two
//! `add`s with identical parameter types but different return types
//! collide (return type is deliberately excluded from the key — nothing at
//! a call site could ever disambiguate two candidates by return type
//! alone). When any parameter lacks a type annotation, the signature can't
//! be compared structurally at this stage — that becomes an inference-time
//! concern, not a parse/merge-time one — so such items skip this check.

use crate::ast::*;
use crate::diag::Diagnostic;
use crate::lower::Lowerer;
use crate::parser::{CleaveParser, Rule};
use crate::print::{fmt_generics, fmt_type};
use pest::Parser;
use std::collections::HashMap;
use std::path::PathBuf;

/// Locates the shipped standard library directory relative to the running
/// `cleave` binary itself — a real installed toolchain finds its bundled
/// resources next to wherever *it* lives, not relative to a Cargo dev
/// environment (see `grammar.md`'s crate-resolution notes: for now this
/// *is* "the shipped standard library" location, full stop — no
/// manifest/registry involved, that's real distribution, deferred).
///
/// Walks upward from the executable's own path until a sibling directory
/// literally named `stdlib` turns up, rather than hardcoding an exact
/// ancestor count — `cargo run`'s binary and `cargo test`'s (one level
/// deeper, under `target/debug/deps/`) sit at different depths, and this
/// finds either without special-casing which one is running. A real
/// installed/packaged build will need a less improvised discovery
/// mechanism eventually; this is a deliberate bootstrap, not it.
pub fn stdlib_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    exe.ancestors().find_map(|dir| {
        let candidate = dir.join("stdlib");
        candidate.is_dir().then_some(candidate)
    })
}

pub struct FileSource {
    pub id: FileId,
    pub name: String,
    pub text: String,
}

/// Parses and lowers a single file. Does not merge — see `merge_programs`.
/// `NodeId`s start fresh at 0 — fine for single-file use (tests, isolated
/// `infer_fn` callers), but never use this across more than one file in the
/// same compilation; see `parse_file_with_ids`.
pub fn parse_file(src: &FileSource) -> Result<Program, Diagnostic> {
    let pair = CleaveParser::parse(Rule::program, &src.text)
        .map_err(|e| Diagnostic::from_pest(&e, src.id))?
        .next()
        .unwrap();
    Ok(Lowerer::new(src.id).lower_program(pair))
}

/// Like `parse_file`, but continuing a shared `NodeIdGen` across files —
/// what `compile`/`load_crate_dir` actually use, so that a `NodeId` (used as
/// a side-table key for e.g. inferred types, see `infer.rs`) uniquely
/// identifies a node across an *entire* multi-file compilation, not just
/// within whichever single file happened to produce it.
pub fn parse_file_with_ids(src: &FileSource, ids: NodeIdGen) -> Result<(Program, NodeIdGen), Diagnostic> {
    let pair = CleaveParser::parse(Rule::program, &src.text)
        .map_err(|e| Diagnostic::from_pest(&e, src.id))?
        .next()
        .unwrap();
    let mut lowerer = Lowerer::with_ids(src.id, ids);
    let program = lowerer.lower_program(pair);
    Ok((program, lowerer.into_ids()))
}

/// Allocates `FileId`s across an entire compilation — entry files and every
/// crate directory discovered while resolving `use`s alike, so no two files
/// anywhere in one `compile` call collide.
#[derive(Default)]
pub struct FileIdGen(u32);

impl FileIdGen {
    pub fn next(&mut self) -> FileId {
        let id = FileId(self.0);
        self.0 += 1;
        id
    }
}

/// Crates always available without an explicit `use` — a "prelude", the
/// same concept as Rust's `std::prelude` (`Vec`/`String`/`Option`/... always
/// in scope without writing `use std::vec::Vec;` in every file). `num` is
/// foundational enough (bare numeric literals need it to be checked against
/// at all — see `infer.rs`'s qualified-types section) that requiring an
/// explicit `use num;` in every single file would be pure ceremony. Loading
/// is best-effort in one specific sense: a prelude crate that can't be
/// *found* is silently skipped (no worse than before this existed, for an
/// environment where the stdlib isn't shipped) — but one that's found and
/// genuinely broken (a real parse/merge error) still surfaces normally,
/// same as any other crate; "missing" and "broken" are different problems.
/// More names join this list as more of the stdlib gets built.
const PRELUDE_CRATES: &[&str] = &["num"];

/// Finds a directory literally named `name` among `search_paths`, in order —
/// "project root(s), then the shipped stdlib" per `grammar.md`. First match
/// wins; callers are expected to pass `stdlib_path()` last in the list.
pub fn resolve_crate_dir(name: &str, search_paths: &[PathBuf]) -> Option<PathBuf> {
    search_paths.iter().map(|root| root.join(name)).find(|p| p.is_dir())
}

/// Reads every `*.cleave` file directly inside `dir` (not recursive — a
/// crate's files aren't nested in subdirectories, at least not yet), parses
/// and lowers each, and merges them via `merge_programs` — the same
/// partial-declaration merge used for a single compilation's own entry
/// files, since a crate's internal files follow the identical rule (a
/// directory *is* one crate, its `algebra`/`impl` fragments merge the same
/// way regardless of which side of a `use` they're on).
///
/// Sorted by path before parsing so which file "arrives first" doesn't
/// depend on the OS's directory-listing order — `merge_programs` doesn't
/// care about order itself, but `FileId` assignment (and therefore
/// diagnostic ordering) would otherwise be nondeterministic across runs.
pub fn load_crate_dir(
    dir: &std::path::Path,
    ids: &mut FileIdGen,
    node_ids: &mut NodeIdGen,
    sources: &mut crate::diag::SourceMap,
) -> Result<Program, Vec<Diagnostic>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("failed to read crate directory {dir:?}: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "cleave"))
        .collect();
    paths.sort();

    let mut errors = Vec::new();
    let mut programs = Vec::new();
    for path in paths {
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {path:?}: {e}"));
        let id = ids.next();
        let name = path.display().to_string();
        sources.add(id, name.clone(), text.clone());
        let taken_ids = std::mem::take(node_ids);
        match parse_file_with_ids(&FileSource { id, name, text }, taken_ids) {
            Ok((p, returned_ids)) => {
                *node_ids = returned_ids;
                programs.push(p);
            }
            Err(e) => errors.push(e),
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }
    merge_programs(programs)
}

/// The top-level entry point: parses `entry_sources`, resolves every
/// top-level `use` found in them against `extra_search_paths` (with
/// `stdlib_path()` always appended last), loads and merges each resolved
/// crate's own files, and merges everything — entries and imported crates
/// alike — into one final `Program`.
///
/// Deliberately one level deep only: a *loaded crate's own* `use`
/// statements are not themselves followed. Real transitive resolution needs
/// cycle detection (crate A using crate B using crate A) that this doesn't
/// attempt yet — fine for now since `stdlib/num` doesn't use anything
/// itself, not fine as a permanent limitation.
///
/// Only a `use` path's *first* segment is resolved to a crate directory —
/// `use linalg::Matrix;` and `use linalg;` behave identically, pulling in
/// the whole crate either way. Per-symbol import filtering was never the
/// intended semantics (a directory *is* the crate, no finer-grained
/// visibility exists yet), not a feature trimmed down to this.
///
/// Always returns the `SourceMap` it built, even on failure — every file
/// successfully parsed before the error is in it, which is exactly what's
/// needed to render whatever diagnostics came back (a CLI that only got a
/// `SourceMap` on success could never render its own compile errors).
pub fn compile(
    entry_sources: Vec<(String, String)>,
    extra_search_paths: &[PathBuf],
) -> (Result<Program, Vec<Diagnostic>>, crate::diag::SourceMap) {
    let mut ids = FileIdGen::default();
    let mut node_ids = NodeIdGen::default();
    let mut sources = crate::diag::SourceMap::default();
    let mut errors = Vec::new();
    let mut programs = Vec::new();

    for (name, text) in entry_sources {
        let id = ids.next();
        sources.add(id, name.clone(), text.clone());
        let taken_ids = std::mem::take(&mut node_ids);
        match parse_file_with_ids(&FileSource { id, name, text }, taken_ids) {
            Ok((p, returned_ids)) => {
                node_ids = returned_ids;
                programs.push(p);
            }
            Err(e) => errors.push(e),
        }
    }
    if !errors.is_empty() {
        return (Err(errors), sources);
    }

    let mut seen_crates = std::collections::HashSet::new();
    let mut wanted: Vec<(String, Span)> = Vec::new();
    for program in &programs {
        for item in &program.items {
            if let ItemKind::Use(path) = &item.kind {
                let name = path.segments[0].clone();
                if seen_crates.insert(name.clone()) {
                    wanted.push((name, item.span));
                }
            }
        }
    }

    let mut search_paths = extra_search_paths.to_vec();
    if let Some(std) = stdlib_path() {
        search_paths.push(std);
    }

    for (name, span) in wanted {
        match resolve_crate_dir(&name, &search_paths) {
            Some(dir) => match load_crate_dir(&dir, &mut ids, &mut node_ids, &mut sources) {
                Ok(p) => programs.push(p),
                Err(mut e) => errors.append(&mut e),
            },
            None => errors.push(Diagnostic::error(format!("cannot find crate `{name}`"), span)),
        }
    }

    // The prelude — see `PRELUDE_CRATES`'s doc comment for why a missing one
    // is silently skipped here while an explicit `use` above is not.
    for name in PRELUDE_CRATES {
        if seen_crates.contains(*name) {
            continue;
        }
        if let Some(dir) = resolve_crate_dir(name, &search_paths) {
            match load_crate_dir(&dir, &mut ids, &mut node_ids, &mut sources) {
                Ok(p) => programs.push(p),
                Err(mut e) => errors.append(&mut e),
            }
        }
    }

    if !errors.is_empty() {
        return (Err(errors), sources);
    }

    (merge_programs(programs), sources)
}

/// `(name, param-type strings)` — `None` if any parameter lacks an
/// annotation (signature not structurally comparable yet).
fn sig_key(name: &str, params: &[Param]) -> Option<(String, Vec<String>)> {
    let types: Option<Vec<String>> = params.iter().map(|p| p.ty.as_ref().map(fmt_type)).collect();
    types.map(|t| (name.to_string(), t))
}

fn generics_match(a: &[GenericParam], b: &[GenericParam]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).all(|(x, y)| match (x, y) {
        (GenericParam::Type { name: n1, bounds: b1 }, GenericParam::Type { name: n2, bounds: b2 }) => {
            n1 == n2 && b1 == b2
        }
        (GenericParam::Const { name: n1, ty: t1 }, GenericParam::Const { name: n2, ty: t2 }) => {
            n1 == n2 && fmt_type(t1) == fmt_type(t2)
        }
        _ => false,
    })
}

struct AlgebraAcc {
    anchor_id: NodeId,
    anchor_span: Span,
    decl: AlgebraDecl,
    seen_fn_sigs: HashMap<(String, Vec<String>), Span>,
    seen_axioms: HashMap<String, Span>,
}

struct ImplAcc {
    anchor_id: NodeId,
    anchor_span: Span,
    decl: ImplDecl,
    seen_fns: HashMap<(String, Vec<String>), Span>,
}

/// Merges every parsed file's items into one logical `Program`. On any
/// conflict, returns *all* conflicts found (not just the first) — a merge
/// pass sees the whole crate at once, so there's no reason to stop early.
pub fn merge_programs(programs: Vec<Program>) -> Result<Program, Vec<Diagnostic>> {
    let mut errors = Vec::new();
    let mut items: Vec<Item> = Vec::new();
    let mut struct_names: Vec<String> = Vec::new();
    let mut fn_names: Vec<String> = Vec::new();
    let mut algebras: Vec<AlgebraAcc> = Vec::new();
    let mut impls: Vec<ImplAcc> = Vec::new();

    for program in programs {
        for item in program.items {
            match &item.kind {
                ItemKind::Use(_) => items.push(item),
                ItemKind::Struct(d) => {
                    if struct_names.contains(&d.name) {
                        errors.push(Diagnostic::error(format!("duplicate struct `{}`", d.name), item.span));
                        continue;
                    }
                    struct_names.push(d.name.clone());
                    items.push(item);
                }
                ItemKind::Fn(d) => {
                    if fn_names.contains(&d.name) {
                        errors.push(Diagnostic::error(format!("duplicate fn `{}`", d.name), item.span));
                        continue;
                    }
                    fn_names.push(d.name.clone());
                    items.push(item);
                }
                ItemKind::Algebra(_) => merge_algebra_fragment(item, &mut algebras, &mut errors),
                ItemKind::Impl(_) => merge_impl_fragment(item, &mut impls, &mut errors),
            }
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    for acc in algebras {
        items.push(Node { id: acc.anchor_id, span: acc.anchor_span, kind: ItemKind::Algebra(acc.decl) });
    }
    for acc in impls {
        items.push(Node { id: acc.anchor_id, span: acc.anchor_span, kind: ItemKind::Impl(acc.decl) });
    }
    Ok(Program { items })
}

fn merge_algebra_fragment(item: Item, algebras: &mut Vec<AlgebraAcc>, errors: &mut Vec<Diagnostic>) {
    let ItemKind::Algebra(d) = item.kind else { unreachable!() };

    let Some(acc) = algebras.iter_mut().find(|a| a.decl.name == d.name) else {
        let mut seen_fn_sigs = HashMap::new();
        let mut seen_axioms = HashMap::new();
        for it in &d.items {
            match &it.kind {
                AlgebraItemKind::FnSig(sig) => {
                    if let Some(key) = sig_key(&sig.name, &sig.params) {
                        seen_fn_sigs.insert(key, it.span);
                    }
                }
                AlgebraItemKind::Axiom(ax) => {
                    seen_axioms.insert(ax.name.clone(), it.span);
                }
            }
        }
        algebras.push(AlgebraAcc { anchor_id: item.id, anchor_span: item.span, decl: d, seen_fn_sigs, seen_axioms });
        return;
    };

    if !generics_match(&acc.decl.generics, &d.generics) {
        errors.push(Diagnostic::error(
            format!("`algebra {}` fragment has generic parameters incompatible with a previous fragment", d.name),
            item.span,
        ));
        return;
    }
    for b in d.bounds {
        if !acc.decl.bounds.contains(&b) {
            acc.decl.bounds.push(b);
        }
    }
    for new_item in d.items {
        let conflict = match &new_item.kind {
            AlgebraItemKind::FnSig(sig) => sig_key(&sig.name, &sig.params).is_some_and(|key| {
                let dup = acc.seen_fn_sigs.contains_key(&key);
                if !dup {
                    acc.seen_fn_sigs.insert(key, new_item.span);
                }
                dup
            }),
            AlgebraItemKind::Axiom(ax) => {
                let dup = acc.seen_axioms.contains_key(&ax.name);
                if !dup {
                    acc.seen_axioms.insert(ax.name.clone(), new_item.span);
                }
                dup
            }
        };
        if conflict {
            let name = match &new_item.kind {
                AlgebraItemKind::FnSig(sig) => &sig.name,
                AlgebraItemKind::Axiom(ax) => &ax.name,
            };
            errors.push(Diagnostic::error(
                format!("`{name}` is declared more than once in `algebra {}` (same parameter types)", d.name),
                new_item.span,
            ));
            continue;
        }
        acc.decl.items.push(new_item);
    }
}

fn merge_impl_fragment(item: Item, impls: &mut Vec<ImplAcc>, errors: &mut Vec<Diagnostic>) {
    let ItemKind::Impl(d) = item.kind else { unreachable!() };
    let target_key = fmt_type(&d.target);
    // The impl's own generics (bounds included) are part of its identity,
    // not just the bare target shape — `impl<T: Float> Ring<Complex<T>>`
    // and `impl<T: Ord> Ring<Complex<T>>` must never be merged into one
    // fragment just because they share the same bare `Complex<T>` target
    // string; `fmt_generics` is empty for the overwhelmingly common
    // non-generic case, so this changes nothing there.
    let generics_key = fmt_generics(&d.generics);

    let Some(acc) = impls.iter_mut().find(|a| {
        a.decl.algebra == d.algebra
            && fmt_type(&a.decl.target) == target_key
            && fmt_generics(&a.decl.generics) == generics_key
    }) else {
        let mut seen_fns = HashMap::new();
        for f in &d.fns {
            if let Some(key) = sig_key(&f.name, &f.params) {
                seen_fns.insert(key, item.span);
            }
        }
        impls.push(ImplAcc { anchor_id: item.id, anchor_span: item.span, decl: d, seen_fns });
        return;
    };

    for f in d.fns {
        let conflict = sig_key(&f.name, &f.params).is_some_and(|key| {
            let dup = acc.seen_fns.contains_key(&key);
            if !dup {
                acc.seen_fns.insert(key, item.span);
            }
            dup
        });
        if conflict {
            // `FnDecl` itself carries no span (only the enclosing `Item`
            // does — see `ast.rs`), so this necessarily points at the whole
            // `impl` fragment currently being merged in, not the specific
            // conflicting method. A known, documented granularity gap.
            errors.push(Diagnostic::error(
                format!(
                    "`{}` is implemented more than once in `impl {}<{target_key}>` (same parameter types)",
                    f.name, d.algebra
                ),
                item.span,
            ));
            continue;
        }
        acc.decl.fns.push(f);
    }
}
