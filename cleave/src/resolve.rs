//! Name resolution: decides, once and purely lexically, what every name in a
//! function body refers to, and records the answer in the AST itself.
//!
//! **Calls.** The rule, in priority order, for a bare (single-segment) callee
//! name `f`:
//!
//! 1. an operator call (`a + b`, `-a`, `not a`; `Path::operator`) always means the
//!    algebra method of that name — a local or a top-level `fn` named `add` never
//!    captures `+`;
//! 2. a local binding in scope (a parameter, a `let`, a lambda parameter, a `for`
//!    variable) named `f`;
//! 3. a top-level `fn` named `f` (including `extern fn` and `grad`/`derive`
//!    declarations) that the body's crate sees: one of its own, or of a crate
//!    it uses, directly or not (`CrateScopes`). A program's functions never
//!    reach into the stdlib's bodies: a program's `fn step` doesn't capture
//!    `optim`'s calls to `Optimizer::step`;
//! 4. the method `f` of the one algebra declaring it with that arity.
//!
//! A qualified call (`Algebra::method`) always means that algebra's method, and
//! `mlir::...` calls are raw operations; both are left as they are.
//!
//! When rule 1 or 4 applies and exactly one algebra declares the method, the
//! callee path is rewritten to its qualified form `Algebra::method`. After this
//! pass, a single-segment callee is therefore always a local or a top-level `fn`
//! for every later pass (inference, monomorphization, CPS conversion), and an
//! algebra call always carries its algebra: there is no second, name-based
//! lookup that could disagree with this one. A name that no algebra declares, or
//! that several algebras declare (an ambiguity), stays bare: inference reports it
//! (`TypeErrorKind::AmbiguousOperator`) with the call's own span.
//!
//! **Shadowing `let`s.** A `let` whose name is already visible (a local in an
//! enclosing or the same scope, or a top-level `fn`) is renamed to a name unique
//! within its function (`x` => `x#1`), together with every reference in its
//! scope. Later passes key variables by name; without this, `{ x = 1; let x = 2;
//! }` made CPS conversion carry the inner `x` out of the block as the outer one.
//! `source_name` recovers the written name for diagnostics.
//!
//! Expressions that are evaluated at compile time as constant arithmetic rather
//! than dispatched (types, turbofish arguments, `[v; N]` repeat counts, `const`
//! and `define` values) are not rewritten: their operators stay bare names that
//! `const_eval` folds directly.

use std::cell::Cell;
use crate::collections::{HashMap, HashSet};

use crate::ast::{
    AlgebraItemKind, Block, ElseBranch, Expr, ExprKind, FileId, FnDecl, ItemKind, Path, Program,
    StmtKind,
};

/// Which crate each source file belongs to, and which crates each crate sees:
/// itself and the crates it uses, directly or not (the prelude's included).
/// A file it doesn't know (a synthesized item's) sees every crate, and is
/// seen by every crate.
#[derive(Default)]
pub struct CrateScopes {
    pub crate_of_file: HashMap<FileId, usize>,
    pub sees: Vec<HashSet<usize>>,
}

impl CrateScopes {
    fn sees(&self, from: Option<usize>, of: Option<usize>) -> bool {
        match (from, of) {
            (Some(from), Some(of)) => from == of || self.sees.get(from).is_some_and(|s| s.contains(&of)),
            _ => true,
        }
    }
}

/// Rewrites every function and impl body in `program` per the module doc
/// comment: algebra-targeted calls to `Algebra::method`, shadowing `let`s to
/// unique names.
pub fn resolve_calls(mut program: Program, scopes: &CrateScopes) -> Program {
    let crate_of = |file: FileId| scopes.crate_of_file.get(&file).copied();
    let mut top_level_fns: HashSet<String> = HashSet::default();
    let mut fn_crates: HashMap<String, Vec<Option<usize>>> = HashMap::default();
    let mut fieldless_structs: HashSet<String> = HashSet::default();
    let mut algebra_methods: HashMap<(String, usize), Vec<String>> = HashMap::default();
    let mut algebra_bounds: HashMap<String, Vec<String>> = HashMap::default();
    for item in &program.items {
        match &item.kind {
            ItemKind::Fn(f) => {
                top_level_fns.insert(f.name.clone());
                fn_crates.entry(f.name.clone()).or_default().push(crate_of(item.span.file));
            }
            ItemKind::Struct(d) if d.fields.is_empty() => {
                fieldless_structs.insert(d.name.clone());
            }
            ItemKind::Algebra(a) => {
                algebra_bounds.insert(a.name.clone(), a.bounds.clone());
                for ai in &a.items {
                    if let AlgebraItemKind::FnSig(sig) = &ai.kind {
                        algebra_methods
                            .entry((sig.name.clone(), sig.params.len()))
                            .or_default()
                            .push(a.name.clone());
                    }
                }
            }
            _ => {}
        }
    }
    let resolver = Resolver {
        top_level_fns,
        fn_crates,
        scopes,
        current_crate: Cell::new(None),
        fieldless_structs,
        algebra_methods,
        algebra_bounds,
        renamed: Cell::new(0),
    };

    for item in &mut program.items {
        resolver.current_crate.set(crate_of(item.span.file));
        match &mut item.kind {
            ItemKind::Fn(f) => resolver.resolve_fn(f),
            ItemKind::Impl(i) => i.fns.iter_mut().for_each(|f| resolver.resolve_fn(f)),
            _ => {}
        }
    }
    program
}

/// The name a binding was written with: `x` for `x#1`, a shadowing `let`
/// renamed by `resolve_calls`. Compiler-synthesized names (`<iife#3>`) are
/// returned unchanged.
pub fn source_name(name: &str) -> &str {
    match name.rsplit_once('#') {
        Some((source, n))
            if !source.starts_with('<') && !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) =>
        {
            source
        }
        _ => name,
    }
}

/// A local in scope: the name it was written with, and the name it has after
/// renaming (the same unless it is a shadowing `let`).
struct Local {
    source: String,
    actual: String,
}

struct Resolver<'s> {
    top_level_fns: HashSet<String>,
    /// The crate of each top-level `fn` of that name.
    fn_crates: HashMap<String, Vec<Option<usize>>>,
    scopes: &'s CrateScopes,
    /// The crate of the item whose body is being resolved.
    current_crate: Cell<Option<usize>>,
    /// Structs without fields: `Name()` (or `Name::<...>()`) constructs one,
    /// though it parses as a call (`grammar.pest`, `primary`).
    fieldless_structs: HashSet<String>,
    algebra_methods: HashMap<(String, usize), Vec<String>>,
    /// Each algebra's super-algebras (`algebra Ring<T> : Additive`).
    algebra_bounds: HashMap<String, Vec<String>>,
    /// Shadowing `let`s renamed so far in the current function.
    renamed: Cell<u32>,
}

impl Resolver<'_> {
    fn resolve_fn(&self, f: &mut FnDecl) {
        let Some(body) = &mut f.body else { return };
        self.renamed.set(0);
        let mut scope: Vec<Local> = f.params.iter().map(|p| unrenamed(&p.name)).collect();
        self.block(body, &mut scope);
    }

    /// The algebra a call to `name` with `arity` arguments targets, when exactly
    /// one algebra declares it.
    fn unique_algebra(&self, name: &str, arity: usize) -> Option<&str> {
        match self.algebra_methods.get(&(name.to_string(), arity))?.as_slice() {
            [algebra] => Some(algebra),
            _ => None,
        }
    }

    /// The algebra among `algebra` and its super-algebras, nearest first,
    /// that declares `name` with `arity` arguments: `Ring::add` is
    /// `Additive::add` (`algebra Ring<T> : Additive`), as a trait's method
    /// is reached through a subtrait in Rust.
    fn declaring_algebra(&self, algebra: &str, name: &str, arity: usize) -> Option<String> {
        let mut queue = vec![algebra.to_string()];
        let mut seen = HashSet::default();
        while let Some(a) = queue.pop() {
            if !seen.insert(a.clone()) {
                continue;
            }
            if self.algebra_methods.get(&(name.to_string(), arity)).is_some_and(|owners| owners.contains(&a)) {
                return Some(a);
            }
            if let Some(bounds) = self.algebra_bounds.get(&a) {
                queue.extend(bounds.iter().rev().cloned());
            }
        }
        None
    }

    /// Binds a `let` named `name`, renaming it when it shadows a visible name.
    fn bind_let(&self, name: &mut String, scope: &mut Vec<Local>) {
        let source = name.clone();
        if lookup(scope, &source).is_some() || self.top_level_fns.contains(&source) {
            self.renamed.set(self.renamed.get() + 1);
            *name = format!("{source}#{}", self.renamed.get());
        }
        scope.push(Local {
            source,
            actual: name.clone(),
        });
    }

    fn block(&self, block: &mut Block, scope: &mut Vec<Local>) {
        let depth = scope.len();
        for stmt in &mut block.stmts {
            match &mut stmt.kind {
                StmtKind::Sync => {}
                // The value is resolved before `name` enters scope: `let f = f(x);`
                // calls the outer `f`.
                StmtKind::Let { name, value, .. } => {
                    self.expr(value, scope);
                    self.bind_let(name, scope);
                }
                StmtKind::Assign { target, value } => {
                    self.expr(target, scope);
                    self.expr(value, scope);
                }
                StmtKind::Expr(e) => self.expr(e, scope),
                StmtKind::Break(value) => {
                    if let Some(v) = value {
                        self.expr(v, scope);
                    }
                }
            }
        }
        if let Some(tail) = &mut block.tail {
            self.expr(tail, scope);
        }
        scope.truncate(depth);
    }

    fn scoped_block(&self, block: &mut Block, scope: &mut Vec<Local>, binders: &[&str]) {
        let depth = scope.len();
        scope.extend(binders.iter().map(|b| unrenamed(b)));
        self.block(block, scope);
        scope.truncate(depth);
    }

    fn expr(&self, expr: &mut Expr, scope: &mut Vec<Local>) {
        match &mut expr.kind {
            ExprKind::Match { .. } => unreachable!("a `match` is lowered by `driver::desugar_enums`"),
            ExprKind::Spawn(call) => self.expr(call, scope),
            ExprKind::NumberLit { .. }
            | ExprKind::ImaginaryLit { .. }
            | ExprKind::BoolLit(_)
            | ExprKind::PackRef(_) => {}
            ExprKind::Path(path) => {
                if let [name] = path.segments.as_mut_slice() {
                    if let Some(actual) = lookup(scope, name) {
                        *name = actual.to_string();
                    }
                }
            }
            // `Name()` naming a struct without fields, and no function or
            // local of that name: a construction, rewritten into the struct
            // literal it is so that its generics (a marker struct has only
            // those: `AttentionShape<L, H>`, `stdlib/nn`) and its turbofish
            // are inferred like any struct literal's.
            ExprKind::Call(path, generics, args, _)
                if args.is_empty()
                    && path.segments.len() == 1
                    && self.fieldless_structs.contains(&path.segments[0])
                    && !self.top_level_fns.contains(&path.segments[0])
                    && lookup(scope, &path.segments[0]).is_none() =>
            {
                let (path, generics) = (path.clone(), std::mem::take(generics));
                expr.kind = ExprKind::StructLit(path, generics, Vec::new());
            }
            ExprKind::Call(path, _, args, _) => {
                args.iter_mut().for_each(|a| self.expr(a, scope));
                self.call_target(path, args.len(), scope);
            }
            ExprKind::FieldAccess(base, _) => self.expr(base, scope),
            ExprKind::Index(base, indices) => {
                self.expr(base, scope);
                indices.iter_mut().for_each(|i| self.expr(i, scope));
            }
            ExprKind::ArrayLit(elems) => elems.iter_mut().for_each(|e| self.expr(e, scope)),
            // `count` is a compile-time constant expression, not a dispatched call.
            ExprKind::ArrayRepeat { value, .. } => self.expr(value, scope),
            ExprKind::StructLit(_, _, fields) => {
                fields.iter_mut().for_each(|(_, v)| self.expr(v, scope))
            }
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.expr(cond, scope);
                self.block(then_branch, scope);
                match else_branch.as_deref_mut() {
                    Some(ElseBranch::If(e)) => self.expr(e, scope),
                    Some(ElseBranch::Block(b)) => self.block(b, scope),
                    None => {}
                }
            }
            ExprKind::While { cond, body } => {
                self.expr(cond, scope);
                self.block(body, scope);
            }
            ExprKind::For {
                var,
                start,
                end,
                body,
            } => {
                self.expr(start, scope);
                self.expr(end, scope);
                self.scoped_block(body, scope, &[var.as_str()]);
            }
            ExprKind::ForIn { var, iter, body } => {
                self.expr(iter, scope);
                self.scoped_block(body, scope, &[var.as_str()]);
            }
            ExprKind::Loop { body } | ExprKind::Block(body) => self.block(body, scope),
            ExprKind::Lambda { params, body, .. } => {
                let names: Vec<&str> = params.iter().map(|p| p.name.as_str()).collect();
                self.scoped_block(body, scope, &names);
            }
        }
    }

    fn call_target(&self, path: &mut Path, arity: usize, scope: &[Local]) {
        if let [algebra, name] = path.segments.as_mut_slice() {
            if let Some(owner) = self.declaring_algebra(algebra, name, arity) {
                *algebra = owner;
            }
            return;
        }
        let [name] = path.segments.as_mut_slice() else {
            return;
        };
        if path.operator {
            if let Some(algebra) = self.unique_algebra(name, arity) {
                path.segments = vec![algebra.to_string(), name.clone()];
            }
            return;
        }
        if let Some(actual) = lookup(scope, name) {
            *name = actual.to_string();
            return;
        }
        let from = self.current_crate.get();
        if self
            .fn_crates
            .get(name.as_str())
            .is_some_and(|crates| crates.iter().any(|&of| self.scopes.sees(from, of)))
        {
            return;
        }
        if let Some(algebra) = self.unique_algebra(name, arity) {
            path.segments = vec![algebra.to_string(), name.clone()];
        }
    }
}

fn unrenamed(name: &str) -> Local {
    Local {
        source: name.to_string(),
        actual: name.to_string(),
    }
}

/// The innermost local written `source`, by its name after renaming.
fn lookup<'s>(scope: &'s [Local], source: &str) -> Option<&'s str> {
    scope
        .iter()
        .rev()
        .find(|l| l.source == source)
        .map(|l| l.actual.as_str())
}
