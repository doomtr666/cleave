//! Whole-program "type-annotated AST" dump — runs inference over every
//! `fn`/`impl` method in a compiled `Program` and renders each statement/tail
//! alongside its resolved type. Top-level `fn`s go through
//! `callgraph::infer_program` (real self/mutual-recursion support, one pass
//! over the whole program); `impl` methods are still each inferred with
//! their own fresh `Infer`, in isolation from every other `impl` method or
//! `fn` (see `infer.rs`'s module docs on why that's a separate, still-open
//! gap, not an oversight here).

use crate::ast::*;
use crate::callgraph::{self, ProgramInference};
use crate::infer::{Infer, Ty, TyVar, TypeError};
use crate::print::{fmt_params, fmt_type};
use crate::registry::Registry;
use std::collections::HashMap;
use std::fmt::Write as _;

type NodeTypes = HashMap<NodeId, Ty>;

/// Runs inference over every `fn`/`impl` method in `program` and renders the
/// result as text. `struct`/`algebra`/`use` items are rendered as a bare
/// marker (not type-inferred at all, see `infer.rs`'s module docs) rather
/// than omitted, so the output still reflects the whole program's shape.
/// Collects *every* function's error rather than stopping at the first,
/// matching `driver::merge_programs`'s "see the whole picture at once"
/// stance — one broken function doesn't hide problems in the others.
pub fn dump_program(program: &Program, registry: &Registry) -> (String, Vec<TypeError>) {
    let mut out = String::new();
    let mut errors = Vec::new();
    // Whole-registry coherence check, independent of any particular
    // function's own inference below — two overlapping generic impls are a
    // problem with the `impl`s themselves, not something any one call site
    // would surface on its own (see `Infer::check_no_overlapping_impls`).
    errors.append(&mut Infer::new(registry).check_no_overlapping_impls());
    let program_inference = callgraph::infer_program(program, registry);

    for (i, item) in program.items.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        match &item.kind {
            ItemKind::Use(path) => {
                let _ = writeln!(out, "use {};", path.segments.join("::"));
            }
            ItemKind::Struct(d) => {
                let _ = writeln!(out, "struct {} {{ /* not type-inferred yet */ }}", d.name);
            }
            ItemKind::Algebra(d) => {
                let _ = writeln!(out, "algebra {} {{ /* not type-inferred yet */ }}", d.name);
            }
            ItemKind::Fn(f) => dump_program_fn(&mut out, &mut errors, f, &program_inference),
            ItemKind::Impl(d) => {
                let _ = writeln!(out, "impl {}<{}> {{", d.algebra, fmt_type(&d.target));
                for f in &d.fns {
                    dump_impl_fn(&mut out, &mut errors, f, registry, &d.algebra, &d.generics, &d.target, item.span);
                }
                let _ = writeln!(out, "}}");
            }
        }
    }

    (out, errors)
}

/// Assigns short, stable, per-function letters (`'a`, `'b`, ... `'z`, then
/// `'a1`, `'b1`, ...) to type variables the first time each is seen —
/// standard ML/Haskell-style pretty-printing (`val id : 'a -> 'a`), rather
/// than exposing the raw internal `TyVar` index (`'t6`), which encodes
/// nothing but allocation order and carries no meaning to a reader — flagged
/// directly as unhelpful for actually debugging with. One instance is shared
/// across a whole function's signature *and* body, so the same variable gets
/// the same letter everywhere it appears (`fn fibonacci(x: 'a) -> 'a { ...
/// x:'a ... }`) — two entirely unrelated functions each reusing `'a` for
/// their own first free variable is correct, not a collision; they really
/// are independent.
#[derive(Default)]
struct TyVarNames {
    names: HashMap<TyVar, String>,
}

impl TyVarNames {
    fn get(&mut self, v: TyVar) -> String {
        let next = self.names.len();
        self.names
            .entry(v)
            .or_insert_with(|| {
                let letter = (b'a' + (next % 26) as u8) as char;
                let suffix = next / 26;
                if suffix == 0 { letter.to_string() } else { format!("{letter}{suffix}") }
            })
            .clone()
    }
}

fn fmt_ty_named(ty: &Ty, names: &mut TyVarNames) -> String {
    match ty {
        Ty::Var(v) => format!("'{}", names.get(*v)),
        Ty::Con(name) => name.clone(),
        Ty::App(name, args) => {
            let args = args.iter().map(|a| fmt_ty_named(a, names)).collect::<Vec<_>>().join(", ");
            format!("{name}<{args}>")
        }
        Ty::Fn(params, ret) => {
            let params = params.iter().map(|p| fmt_ty_named(p, names)).collect::<Vec<_>>().join(", ");
            format!("({params}) -> {}", fmt_ty_named(ret, names))
        }
        Ty::Array(elem, size) => format!("[{}; {}]", fmt_ty_named(elem, names), fmt_ty_named(size, names)),
        Ty::Const(n) => n.to_string(),
    }
}

/// Renders one top-level `fn`, using the already-computed whole-program
/// result (`callgraph::infer_program`) rather than inferring it again here.
fn dump_program_fn(out: &mut String, errors: &mut Vec<TypeError>, f: &FnDecl, program_inference: &ProgramInference) {
    match program_inference.results.get(&f.name) {
        Some(Ok(fn_result)) => {
            let mut names = TyVarNames::default();
            let params: Vec<String> = f
                .params
                .iter()
                .zip(fn_result.param_types.iter())
                .map(|(p, t)| format!("{}: {}", p.name, fmt_ty_named(t, &mut names)))
                .collect();
            let ret = fmt_ty_named(&fn_result.result, &mut names);
            let _ = writeln!(out, "fn {}({}) -> {ret} {{", f.name, params.join(", "));
            dump_block(out, &f.body, &program_inference.node_types, &mut names, 1);
            let _ = writeln!(out, "}}");
        }
        Some(Err(e)) => {
            let params: Vec<String> = f.params.iter().map(|p| p.name.clone()).collect();
            let _ = writeln!(out, "fn {}({}) {{ /* type error, see diagnostics */ }}", f.name, params.join(", "));
            errors.push(e.clone());
        }
        None => unreachable!("`{}` is a top-level `fn` item but callgraph::infer_program has no entry for it", f.name),
    }
}

fn dump_impl_fn(
    out: &mut String,
    errors: &mut Vec<TypeError>,
    f: &FnDecl,
    registry: &Registry,
    algebra: &str,
    impl_generics: &[GenericParam],
    target: &Type,
    fallback_span: Span,
) {
    let mut infer = Infer::new(registry);
    match infer.infer_impl_fn_generic(algebra, impl_generics, target, f, fallback_span) {
        Ok(ret) => {
            let mut names = TyVarNames::default();
            let params: Vec<String> = f
                .params
                .iter()
                .zip(infer.param_types.iter())
                .map(|(p, t)| format!("{}: {}", p.name, fmt_ty_named(t, &mut names)))
                .collect();
            let ret = fmt_ty_named(&ret, &mut names);
            let _ = writeln!(out, "fn {}({}) -> {ret} {{", f.name, params.join(", "));
            dump_block(out, &f.body, &infer.node_types, &mut names, 1);
            let _ = writeln!(out, "}}");
        }
        Err(e) => {
            let params: Vec<String> = f.params.iter().map(|p| p.name.clone()).collect();
            let _ = writeln!(out, "fn {}({}) {{ /* type error, see diagnostics */ }}", f.name, params.join(", "));
            errors.push(e);
        }
    }
}

fn dump_block(out: &mut String, block: &Block, node_types: &NodeTypes, names: &mut TyVarNames, indent: usize) {
    let pad = "    ".repeat(indent);
    for stmt in &block.stmts {
        match &stmt.kind {
            StmtKind::Let { mutable, name, value, .. } => {
                let mut_kw = if *mutable { "mut " } else { "" };
                let _ = writeln!(out, "{pad}let {mut_kw}{name} = {};", fmt_expr_typed(value, node_types, names));
            }
            StmtKind::Assign { name, value } => {
                let _ = writeln!(out, "{pad}{name} = {};", fmt_expr_typed(value, node_types, names));
            }
            StmtKind::Expr(e) => {
                let _ = writeln!(out, "{pad}{};", fmt_expr_typed(e, node_types, names));
            }
        }
    }
    if let Some(tail) = &block.tail {
        let _ = writeln!(out, "{pad}{}", fmt_expr_typed(tail, node_types, names));
    }
}

/// Like `print::fmt_expr`, but every sub-expression — not just the outermost
/// one per statement/tail line — is annotated with its own resolved type
/// (`expr:type`, the same no-space convention an already-suffixed numeric
/// literal uses, e.g. `1:i32`), recursing all the way down, with any type
/// variable rendered via `names` (`'a`, `'b`, ... — see `TyVarNames`) rather
/// than its raw internal index. `print::fmt_expr` itself is left alone (used
/// by the plain, type-free `--dump-ast` output and by `print.rs`'s own
/// tests) — this is a separate renderer specifically for
/// `--dump-inference-pass`, where seeing only the outermost type per line
/// isn't enough to actually debug a deeply nested expression.
fn fmt_expr_typed(e: &Expr, node_types: &NodeTypes, names: &mut TyVarNames) -> String {
    // A suffixed literal is already fully annotated by its own suffix
    // (`1:i32`) — looking `node_types` up too would just repeat the same
    // information a second time.
    if let ExprKind::NumberLit { text, suffix: Some(s) } = &e.kind {
        return format!("{text}:{s}");
    }

    let base = match &e.kind {
        ExprKind::NumberLit { text, .. } => text.clone(),
        ExprKind::ImaginaryLit { text, .. } => format!("{text}i"),
        ExprKind::BoolLit(b) => b.to_string(),
        ExprKind::Path(p) => p.segments.join("::"),
        ExprKind::Call(path, args) => {
            format!("{}({})", path.segments.join("::"), fmt_expr_list_typed(args, node_types, names))
        }
        ExprKind::FieldAccess(base, name) => format!("{}.{name}", fmt_expr_typed(base, node_types, names)),
        ExprKind::MethodCall(base, name, args) => {
            format!(
                "{}.{name}({})",
                fmt_expr_typed(base, node_types, names),
                fmt_expr_list_typed(args, node_types, names)
            )
        }
        ExprKind::Index(base, idx) => {
            format!("{}[{}]", fmt_expr_typed(base, node_types, names), fmt_expr_typed(idx, node_types, names))
        }
        ExprKind::ArrayLit(elems) => format!("[{}]", fmt_expr_list_typed(elems, node_types, names)),
        ExprKind::StructLit(path, fields) => format!(
            "{}({})",
            path.segments.join("::"),
            fields
                .iter()
                .map(|(name, v)| format!("{name}: {}", fmt_expr_typed(v, node_types, names)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        ExprKind::If { cond, then_branch, else_branch } => {
            let mut s = format!(
                "if {} {}",
                fmt_expr_typed(cond, node_types, names),
                fmt_block_inline_typed(then_branch, node_types, names)
            );
            if let Some(eb) = else_branch {
                let _ = write!(s, " else {}", match &**eb {
                    ElseBranch::If(i) => fmt_expr_typed(i, node_types, names),
                    ElseBranch::Block(b) => fmt_block_inline_typed(b, node_types, names),
                });
            }
            s
        }
        ExprKind::While { cond, body } => {
            format!(
                "while {} {}",
                fmt_expr_typed(cond, node_types, names),
                fmt_block_inline_typed(body, node_types, names)
            )
        }
        ExprKind::For { var, start, end, body } => format!(
            "for {var} in {}..{} {}",
            fmt_expr_typed(start, node_types, names),
            fmt_expr_typed(end, node_types, names),
            fmt_block_inline_typed(body, node_types, names)
        ),
        ExprKind::Block(b) => fmt_block_inline_typed(b, node_types, names),
        ExprKind::Lambda { params, ret, body } => {
            let ret_ann = ret.as_ref().map(|t| format!(" -> {}", fmt_type(t))).unwrap_or_default();
            format!("fn({}){ret_ann} {}", fmt_params(params), fmt_block_inline_typed(body, node_types, names))
        }
    };
    let ty = node_types.get(&e.id).map(|t| fmt_ty_named(t, names)).unwrap_or_else(|| "?".to_string());
    format!("{base}:{ty}")
}

fn fmt_expr_list_typed(exprs: &[Expr], node_types: &NodeTypes, names: &mut TyVarNames) -> String {
    exprs.iter().map(|e| fmt_expr_typed(e, node_types, names)).collect::<Vec<_>>().join(", ")
}

/// Like `print.rs`'s own `fmt_block_inline`, but every statement/tail inside
/// is rendered via `fmt_expr_typed`.
fn fmt_block_inline_typed(b: &Block, node_types: &NodeTypes, names: &mut TyVarNames) -> String {
    let mut parts: Vec<String> = b
        .stmts
        .iter()
        .map(|s| match &s.kind {
            StmtKind::Let { mutable, name, value, .. } => {
                let mut_kw = if *mutable { "mut " } else { "" };
                format!("let {mut_kw}{name} = {};", fmt_expr_typed(value, node_types, names))
            }
            StmtKind::Assign { name, value } => format!("{name} = {};", fmt_expr_typed(value, node_types, names)),
            StmtKind::Expr(e) => format!("{};", fmt_expr_typed(e, node_types, names)),
        })
        .collect();
    if let Some(tail) = &b.tail {
        parts.push(fmt_expr_typed(tail, node_types, names));
    }
    if parts.is_empty() {
        "{}".to_string()
    } else {
        format!("{{ {} }}", parts.join(" "))
    }
}
