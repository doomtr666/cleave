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
//! Conflict identity for `algebra`/`impl` methods is `(name, parameter
//! types)` — ordinary overload resolution: two `add`s with different
//! parameter types coexist; two `add`s with identical parameter types but
//! different return types collide (return type is deliberately excluded
//! from the key — nothing at a call site could ever disambiguate two
//! candidates by return type alone). When any parameter lacks a type
//! annotation, the signature can't be compared structurally at this stage —
//! that becomes an inference-time concern, not a parse/merge-time one — so
//! such items skip this check. *Inherent* impl methods (`impl struct Vec2 {
//! ... }`) are the one exception: no overloading concept exists for them at
//! all (`Registry::inherent_method` stores exactly one entry per method
//! name, full stop), so conflict identity there is the bare method name —
//! see `merge_inherent_impl_fragment`'s own doc comment.

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
/// More names join this list as more of the stdlib gets built. `logic`
/// (`and`/`or`/`xor`/`implies`/`not` on `bool`) joined for the identical
/// reason `num` did: `if a and b { ... }` should just work, the same as
/// `a + b` does, not require an explicit `use logic;` for something this
/// basic.
const PRELUDE_CRATES: &[&str] = &["num", "logic"];

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
/// top-level `use` found in them (and, transitively, every `use` found in
/// each crate thereby loaded, and so on) against `extra_search_paths` (with
/// `stdlib_path()` always appended last), loads and merges each resolved
/// crate's own files, and merges everything — entries and imported crates
/// alike — into one final `Program`.
///
/// Real transitive resolution, not one level deep: `visit_crate`'s own DFS
/// scans a *newly loaded* crate's own merged `Program` for further
/// `ItemKind::Use` items exactly the same way this function's own top-level
/// scan already does for the entry files, recursing — with real cycle
/// detection (`stack`, the current DFS path — not just `loaded`, which only
/// ever dedups) reporting a clean, located diagnostic naming the actual
/// cycle rather than looping forever or silently truncating. Pushing each
/// crate's own `Program` in DFS *post-order* (after every one of its own
/// dependencies has already been visited) means `programs` ends up in a
/// real topological order, dependencies before dependents — `merge_programs`
/// itself doesn't strictly require this (a flat, order-independent union),
/// but it keeps `FileId` assignment deterministic, the same concern
/// `load_crate_dir`'s own path-sorting already exists for.
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

    let mut wanted: Vec<(String, Span)> = Vec::new();
    let mut wanted_names = std::collections::HashSet::new();
    for program in &programs {
        for item in &program.items {
            if let ItemKind::Use(path) = &item.kind {
                let name = path.segments[0].clone();
                if wanted_names.insert(name.clone()) {
                    wanted.push((name, item.span));
                }
            }
        }
    }

    let mut search_paths = extra_search_paths.to_vec();
    if let Some(std) = stdlib_path() {
        search_paths.push(std);
    }

    let mut loaded = std::collections::HashSet::new();
    let mut stack = Vec::new();
    for (name, span) in wanted {
        visit_crate(
            &name,
            span,
            true,
            &search_paths,
            &mut ids,
            &mut node_ids,
            &mut sources,
            &mut programs,
            &mut errors,
            &mut loaded,
            &mut stack,
        );
    }

    // The prelude — see `PRELUDE_CRATES`'s own doc comment for why a
    // missing one is silently skipped here (`required: false`, and no real
    // use-site span to report against even if it weren't) while an explicit
    // `use` above is not — joins the exact same transitive-discovery
    // worklist otherwise, a prelude crate's own `use`s followed exactly
    // like any other's.
    for name in PRELUDE_CRATES {
        if !wanted_names.contains(*name) {
            let synthetic_span = Span { file: FileId(0), start: 0, end: 0 };
            visit_crate(
                name,
                synthetic_span,
                false,
                &search_paths,
                &mut ids,
                &mut node_ids,
                &mut sources,
                &mut programs,
                &mut errors,
                &mut loaded,
                &mut stack,
            );
        }
    }

    if !errors.is_empty() {
        return (Err(errors), sources);
    }

    let result = merge_programs(programs).and_then(|program| synthesize_derive_signatures(program, &mut node_ids));
    (result, sources)
}

/// `fprime = derive(f);` (`grammar.pest`'s own `derive_decl`, lowered by
/// `lower.rs` to an ordinary bodyless `ItemKind::Fn` with `derivative_of:
/// Some("f")`) has no `params`/`ret` of its own yet at this point —
/// `lower.rs` has no way to know `f`'s own signature, possibly declared in
/// a different file entirely, only merged into one `Program` by
/// `merge_programs`, right before this runs. Mechanically derived, never
/// written by hand: `f` itself must already be a real, non-generic
/// function with every parameter *and* its own return type explicitly
/// declared as the exact same type `T` (checked by comparing each one's
/// own formatted text — `print.rs::fmt_type` — not full type inference,
/// which hasn't run yet at this point in the pipeline) — `N == 1`
/// parameter synthesizes `fn fprime(x: T) -> T` (an ordinary scalar
/// derivative); `N > 1` synthesizes `fn fprime(x1: T, ..., xN: T) -> [T;
/// N]` (the gradient/Jacobian row, reusing the array type that already
/// exists rather than needing tuples, which aren't implemented).
/// Heterogeneous parameter types, a missing/generic/nullary `f`, or any
/// unannotated parameter on `f` — every one a real, located `Diagnostic`,
/// never guessed.
fn synthesize_derive_signatures(mut program: Program, node_ids: &mut NodeIdGen) -> Result<Program, Vec<Diagnostic>> {
    let fns: HashMap<String, FnDecl> = program
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::Fn(f) => Some((f.name.clone(), f.clone())),
            _ => None,
        })
        .collect();

    let mut errors = Vec::new();
    for item in &mut program.items {
        let ItemKind::Fn(fprime) = &mut item.kind else { continue };
        let Some(of) = fprime.derivative_of.clone() else { continue };

        let Some(f) = fns.get(&of) else {
            errors.push(Diagnostic::error(format!("`derive({of})`: no function named `{of}`"), item.span));
            continue;
        };
        if !f.generics.is_empty() {
            errors.push(Diagnostic::error(format!("`derive({of})`: `{of}` is generic — only a non-generic function can be derived"), item.span));
            continue;
        }
        if f.params.is_empty() {
            errors.push(Diagnostic::error(format!("`derive({of})`: `{of}` takes no parameters — nothing to differentiate with respect to"), item.span));
            continue;
        }
        let Some(ret) = &f.ret else {
            errors.push(Diagnostic::error(format!("`derive({of})`: `{of}` has no declared return type"), item.span));
            continue;
        };
        let Some(param_tys): Option<Vec<&Type>> = f.params.iter().map(|p| p.ty.as_ref()).collect() else {
            errors.push(Diagnostic::error(format!("`derive({of})`: every parameter of `{of}` must have an explicit type"), item.span));
            continue;
        };
        let ret_text = fmt_type(ret);
        if param_tys.iter().any(|t| fmt_type(t) != ret_text) {
            errors.push(Diagnostic::error(
                format!("`derive({of})`: every parameter of `{of}`, and its own return type, must be the same type `{ret_text}` — heterogeneous types aren't supported yet"),
                item.span,
            ));
            continue;
        }

        fprime.params = f.params.clone();
        fprime.ret = Some(if f.params.len() == 1 {
            ret.clone()
        } else {
            let size = Node {
                id: node_ids.next(),
                span: ret.span,
                kind: ExprKind::NumberLit { text: f.params.len().to_string(), suffix: None },
            };
            Node { id: node_ids.next(), span: ret.span, kind: TypeKind::Array(Box::new(ret.clone()), Box::new(size)) }
        });
    }

    if errors.is_empty() { Ok(program) } else { Err(errors) }
}

/// One node of `compile`'s own transitive crate-resolution DFS — see that
/// function's own doc comment for the overall shape. `required` mirrors
/// `PRELUDE_CRATES`'s own "missing is silently fine, broken is not"
/// distinction: `false` only for a top-level prelude entry point that
/// simply isn't present; every crate discovered *because* something else
/// really does `use` it (an entry file, or another crate's own `use`) is
/// always `required: true`, including a prelude crate's own further
/// dependencies once the prelude crate itself is found.
#[allow(clippy::too_many_arguments)]
fn visit_crate(
    name: &str,
    span: Span,
    required: bool,
    search_paths: &[PathBuf],
    ids: &mut FileIdGen,
    node_ids: &mut NodeIdGen,
    sources: &mut crate::diag::SourceMap,
    programs: &mut Vec<Program>,
    errors: &mut Vec<Diagnostic>,
    loaded: &mut std::collections::HashSet<String>,
    stack: &mut Vec<String>,
) {
    if loaded.contains(name) {
        return;
    }
    if let Some(pos) = stack.iter().position(|n| n == name) {
        let cycle = stack[pos..].iter().cloned().chain(std::iter::once(name.to_string())).collect::<Vec<_>>().join(" -> ");
        errors.push(Diagnostic::error(format!("circular crate dependency: {cycle}"), span));
        loaded.insert(name.to_string());
        return;
    }

    let Some(dir) = resolve_crate_dir(name, search_paths) else {
        if required {
            errors.push(Diagnostic::error(format!("cannot find crate `{name}`"), span));
        }
        loaded.insert(name.to_string());
        return;
    };

    stack.push(name.to_string());
    match load_crate_dir(dir.as_path(), ids, node_ids, sources) {
        Ok(program) => {
            let mut deps: Vec<(String, Span)> = Vec::new();
            let mut dep_names = std::collections::HashSet::new();
            for item in &program.items {
                if let ItemKind::Use(path) = &item.kind {
                    let dep_name = path.segments[0].clone();
                    if dep_names.insert(dep_name.clone()) {
                        deps.push((dep_name, item.span));
                    }
                }
            }
            for (dep_name, dep_span) in deps {
                visit_crate(&dep_name, dep_span, true, search_paths, ids, node_ids, sources, programs, errors, loaded, stack);
            }
            // Post-order: this crate's own `Program` is only pushed *after*
            // every one of its own dependencies has been — see this
            // function's own doc comment for why (a real topological order).
            programs.push(program);
        }
        Err(mut e) => errors.append(&mut e),
    }
    stack.pop();
    loaded.insert(name.to_string());
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
    /// Keyed by the method name being differentiated (`DerivativeRuleDecl`
    /// has no separate rule name of its own, unlike `AxiomDecl` — the
    /// method already is the identity, mirrors `seen_axioms` exactly
    /// otherwise).
    seen_derivative_rules: HashMap<String, Span>,
}

struct ImplAcc {
    anchor_id: NodeId,
    anchor_span: Span,
    decl: ImplDecl,
    seen_fns: HashMap<(String, Vec<String>), Span>,
}

struct InherentImplAcc {
    anchor_id: NodeId,
    anchor_span: Span,
    decl: InherentImplDecl,
    /// Keyed by method *name* alone, unlike `ImplAcc::seen_fns` (which keys
    /// by `sig_key`, name **and** parameter types, to allow legitimate
    /// operator overloading — two different `add` signatures coexisting).
    /// Inherent methods have no such overloading concept: `Registry`'s own
    /// `inherent_methods` lookup stores exactly one entry per (struct,
    /// method name), full stop, so two same-named methods would silently
    /// clobber each other there regardless of what merge-time allowed
    /// through — this must reject on name alone to actually match what
    /// dispatch can support. Also sidesteps a real gap `sig_key` has for
    /// this specific case: it returns `None` (skipping the check
    /// entirely) whenever *any* parameter lacks a type annotation — fine
    /// for an algebra impl (an unannotated param falls back to the
    /// algebra's own declared signature) but wrong here, where an
    /// unannotated *first* parameter defaulting to the impl's own target
    /// is the common case, not the exception (see
    /// `Infer::infer_inherent_impl_fn_generic`).
    seen_fns: HashMap<String, Span>,
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
    let mut inherent_impls: Vec<InherentImplAcc> = Vec::new();

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
                ItemKind::InherentImpl(_) => merge_inherent_impl_fragment(item, &mut inherent_impls, &mut errors),
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
    for acc in inherent_impls {
        items.push(Node { id: acc.anchor_id, span: acc.anchor_span, kind: ItemKind::InherentImpl(acc.decl) });
    }
    Ok(Program { items })
}

fn merge_algebra_fragment(item: Item, algebras: &mut Vec<AlgebraAcc>, errors: &mut Vec<Diagnostic>) {
    let ItemKind::Algebra(d) = item.kind else { unreachable!() };

    let Some(acc) = algebras.iter_mut().find(|a| a.decl.name == d.name) else {
        let mut seen_fn_sigs = HashMap::new();
        let mut seen_axioms = HashMap::new();
        let mut seen_derivative_rules = HashMap::new();
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
                AlgebraItemKind::DerivativeRule(dr) => {
                    seen_derivative_rules.insert(dr.method.clone(), it.span);
                }
            }
        }
        algebras.push(AlgebraAcc { anchor_id: item.id, anchor_span: item.span, decl: d, seen_fn_sigs, seen_axioms, seen_derivative_rules });
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
            AlgebraItemKind::DerivativeRule(dr) => {
                let dup = acc.seen_derivative_rules.contains_key(&dr.method);
                if !dup {
                    acc.seen_derivative_rules.insert(dr.method.clone(), new_item.span);
                }
                dup
            }
        };
        if conflict {
            let name = match &new_item.kind {
                AlgebraItemKind::FnSig(sig) => &sig.name,
                AlgebraItemKind::Axiom(ax) => &ax.name,
                AlgebraItemKind::DerivativeRule(dr) => &dr.method,
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

/// Whether two `impl`s' own attribute lists carry the same semantic content
/// (name + args, positionally) — `Attribute` derives no `PartialEq` of its
/// own, and wouldn't want a blind one here anyway: `span` differs between
/// any two fragments by construction, but that's never a real disagreement.
fn attrs_match(a: &[Attribute], b: &[Attribute]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.name == y.name && x.args == y.args)
}

fn merge_impl_fragment(item: Item, impls: &mut Vec<ImplAcc>, errors: &mut Vec<Diagnostic>) {
    let ItemKind::Impl(d) = item.kind else { unreachable!() };
    // Every target folded into one key, comma-joined, not just the first —
    // two heterogeneous impls sharing a first target but differing in a
    // later one (`impl<...> MatMul<Matrix<T,N,M>, X, Y>` vs. `..., Z, W>`)
    // must stay distinct fragments, same reasoning the registry's own
    // equivalent key gets in `Registry::build`. Joined with a separator
    // (not bare-concatenated) both to avoid a pathological cross-target
    // string collision and because this same key doubles as the conflict
    // error message's own rendering below.
    let target_key_of =
        |t: &ImplDecl| -> String { std::iter::once(&t.target).chain(t.extra_targets.iter()).map(fmt_type).collect::<Vec<_>>().join(", ") };
    let target_key = target_key_of(&d);
    // The impl's own generics (bounds included) are part of its identity,
    // not just the bare target shape — `impl<T: Float> Ring<Complex<T>>`
    // and `impl<T: Ord> Ring<Complex<T>>` must never be merged into one
    // fragment just because they share the same bare `Complex<T>` target
    // string; `fmt_generics` is empty for the overwhelmingly common
    // non-generic case, so this changes nothing there.
    let generics_key = fmt_generics(&d.generics);

    let Some(acc) = impls.iter_mut().find(|a| {
        a.decl.algebra == d.algebra && target_key_of(&a.decl) == target_key && fmt_generics(&a.decl.generics) == generics_key
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

    // `#[mlir_type(...)]` (the only attribute an `impl` can carry — see
    // `ImplDecl::attrs`'s own doc comment) is a whole-program, type-level
    // fact about the target's own MLIR representation, not a per-fragment
    // detail that could ever legitimately vary between two fragments of
    // "the same" impl — unlike `.fns` (genuinely split across files on
    // purpose) merging never touched `.attrs` at all, so whichever fragment
    // happened to be processed *first* silently determined it, with the
    // second fragment's own attrs discarded with no diagnostic. Real bug,
    // found by direct testing: a local, unrelated `impl Float<f64> {}`
    // (e.g. an ad hoc test fixture, unaware `Float` is already a real
    // prelude algebra) processed before `stdlib/num/num.cleave`'s own
    // `#[mlir_type(f64)] impl Float<f64> {}` (the entry file's own items are
    // always merged before any resolved `use`/prelude crate's — see
    // `compile`'s own doc comment) silently won, discarding the prelude's
    // own tag — `mlir_lower.rs::ty_to_mlir` then had nothing to look `f64`
    // up by, and treated it as an opaque struct pointer instead of a real
    // float type, all the way down to a native MLIR assertion at codegen
    // time with no compile-time diagnostic anywhere pointing at the cause.
    // Fixed the same way a duplicate method (`.fns`, just below) already
    // is: adopt an empty accumulator's own attrs from whichever fragment
    // declares them, but a real disagreement between two fragments that
    // both declare attrs is reported, not silently resolved by fragment
    // order.
    if !d.attrs.is_empty() {
        if acc.decl.attrs.is_empty() {
            acc.decl.attrs.clone_from(&d.attrs);
        } else if !attrs_match(&acc.decl.attrs, &d.attrs) {
            errors.push(Diagnostic::error(
                format!(
                    "`impl {}<{target_key}>` fragments disagree on their own attributes (e.g. `#[mlir_type(...)]`) — an earlier fragment already declared different ones",
                    d.algebra
                ),
                item.span,
            ));
            return;
        }
    }

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

fn merge_inherent_impl_fragment(item: Item, impls: &mut Vec<InherentImplAcc>, errors: &mut Vec<Diagnostic>) {
    let ItemKind::InherentImpl(d) = item.kind else { unreachable!() };
    let target_key = fmt_type(&d.target);
    // Same reasoning as `merge_impl_fragment`'s own `generics_key` — the
    // impl's own generics are part of its identity, not just the bare
    // target shape.
    let generics_key = fmt_generics(&d.generics);

    let Some(acc) = impls
        .iter_mut()
        .find(|a| fmt_type(&a.decl.target) == target_key && fmt_generics(&a.decl.generics) == generics_key)
    else {
        let mut seen_fns = HashMap::new();
        for f in &d.fns {
            seen_fns.insert(f.name.clone(), item.span);
        }
        impls.push(InherentImplAcc { anchor_id: item.id, anchor_span: item.span, decl: d, seen_fns });
        return;
    };

    for f in d.fns {
        if acc.seen_fns.contains_key(&f.name) {
            errors.push(Diagnostic::error(
                format!("`{}` is implemented more than once in `impl {target_key}`", f.name),
                item.span,
            ));
            continue;
        }
        acc.seen_fns.insert(f.name.clone(), item.span);
        acc.decl.fns.push(f);
    }
}
