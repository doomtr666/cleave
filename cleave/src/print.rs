//! Pretty-printer for the AST — deliberately built *before* the next
//! transformation pass (type inference), so "before" and "after" ASTs can be
//! eyeballed and compared, rather than relying on raw `#[derive(Debug)]`
//! output (which drowns the actual structure under every `NodeId`/`Span`).
//!
//! Prints the AST's *actual* (already-desugared) shape — `add(a, b)`, not
//! `a + b` — on purpose: the whole point is to verify what `lower.rs`
//! produced, and re-sugaring would hide exactly what needs checking.

use crate::ast::*;
use std::fmt::Write as _;

pub fn print_program(program: &Program) -> String {
    let mut p = Printer::default();
    for (i, item) in program.items.iter().enumerate() {
        if i > 0 {
            p.blank_line();
        }
        p.print_item(item);
    }
    p.out
}

#[derive(Default)]
struct Printer {
    out: String,
    indent: usize,
}

impl Printer {
    fn line(&mut self, s: impl AsRef<str>) {
        for _ in 0..self.indent {
            self.out.push_str("    ");
        }
        self.out.push_str(s.as_ref());
        self.out.push('\n');
    }

    fn blank_line(&mut self) {
        self.out.push('\n');
    }

    fn indented(&mut self, f: impl FnOnce(&mut Self)) {
        self.indent += 1;
        f(self);
        self.indent -= 1;
    }

    // ---------------------------------------------------------------- items

    fn print_item(&mut self, item: &Item) {
        match &item.kind {
            ItemKind::Use(path) => self.line(format!("use {};", fmt_path(path))),
            ItemKind::Struct(d) => self.print_struct_decl(d),
            ItemKind::Algebra(d) => self.print_algebra_decl(d),
            ItemKind::Impl(d) => self.print_impl_decl(d),
            ItemKind::InherentImpl(d) => self.print_inherent_impl_decl(d),
            ItemKind::Fn(d) => self.print_fn_decl(d),
        }
    }

    fn print_struct_decl(&mut self, d: &StructDecl) {
        self.line(format!("struct {}{} {{", d.name, fmt_generics(&d.generics)));
        self.indented(|p| {
            for f in &d.fields {
                p.line(format!("{}: {},", f.name, fmt_type(&f.ty)));
            }
        });
        self.line("}");
    }

    fn print_algebra_decl(&mut self, d: &AlgebraDecl) {
        let bounds = if d.bounds.is_empty() { String::new() } else { format!(": {}", d.bounds.join(" + ")) };
        self.line(format!("algebra {}{}{} {{", d.name, fmt_generics(&d.generics), bounds));
        self.indented(|p| {
            for item in &d.items {
                match &item.kind {
                    AlgebraItemKind::FnSig(sig) => {
                        let ret = sig.ret.as_ref().map(|t| format!(" -> {}", fmt_type(t))).unwrap_or_default();
                        p.line(format!("fn {}({}){};", sig.name, fmt_params(&sig.params), ret));
                    }
                    AlgebraItemKind::Axiom(ax) => {
                        p.line(format!("axiom {}({}): {};", ax.name, fmt_params(&ax.params), fmt_expr(&ax.body)));
                    }
                }
            }
        });
        self.line("}");
    }

    fn print_impl_decl(&mut self, d: &ImplDecl) {
        let targets: Vec<String> = std::iter::once(&d.target).chain(d.extra_targets.iter()).map(fmt_type).collect();
        self.line(format!("impl{} {}<{}> {{", fmt_generics(&d.generics), d.algebra, targets.join(", ")));
        self.indented(|p| {
            for f in &d.fns {
                p.print_fn_decl(f);
            }
        });
        self.line("}");
    }

    fn print_inherent_impl_decl(&mut self, d: &InherentImplDecl) {
        self.line(format!("impl{} {} {{", fmt_generics(&d.generics), fmt_type(&d.target)));
        self.indented(|p| {
            for f in &d.fns {
                p.print_fn_decl(f);
            }
        });
        self.line("}");
    }

    fn print_fn_decl(&mut self, d: &FnDecl) {
        for attr in &d.attrs {
            self.line(format!("#[{}({})]", attr.name, attr.args.join(", ")));
        }
        let ret = d.ret.as_ref().map(|t| format!(" -> {}", fmt_type(t))).unwrap_or_default();
        match &d.body {
            Some(body) => {
                self.line(format!("fn {}{}({}){} {{", d.name, fmt_generics(&d.generics), fmt_params(&d.params), ret));
                self.indented(|p| p.print_block_contents(body));
                self.line("}");
            }
            // A bodyless `fn` (only ever legal for an algebra-impl method
            // with a recognized attribute, checked at inference time, not
            // here — see `grammar.pest`'s own `fn_decl` comment) — same
            // semicolon-terminated shape `print_algebra_decl` already uses
            // for an algebra's own `fn_sig`.
            None => {
                self.line(format!("fn {}{}({}){};", d.name, fmt_generics(&d.generics), fmt_params(&d.params), ret));
            }
        }
    }

    fn print_block_contents(&mut self, b: &Block) {
        for stmt in &b.stmts {
            self.print_stmt(stmt);
        }
        if let Some(tail) = &b.tail {
            self.line(fmt_expr(tail));
        }
    }

    fn print_stmt(&mut self, s: &Stmt) {
        match &s.kind {
            StmtKind::Let { mutable, name, ty, value } => {
                let mut_kw = if *mutable { "mut " } else { "" };
                let ty_ann = ty.as_ref().map(|t| format!(": {}", fmt_type(t))).unwrap_or_default();
                self.line(format!("let {mut_kw}{name}{ty_ann} = {};", fmt_expr(value)));
            }
            StmtKind::Assign { target, value } => self.line(format!("{} = {};", fmt_expr(target), fmt_expr(value))),
            StmtKind::Expr(e) => self.line(format!("{};", fmt_expr(e))),
        }
    }
}

// ---------------------------------------------------------------- formatting helpers
// (expressions/types are rendered inline via `fmt_*` — only statement/item/block
// *structure* needs indentation-aware printing above.)

/// `pub(crate)`: reused by `driver.rs`/`registry.rs` as part of the
/// structural key that tells two `impl` blocks apart — `fmt_type(target)`
/// alone can't distinguish `impl<T: Float> Ring<Complex<T>>` from
/// `impl<T: Ord> Ring<Complex<T>>` (both stringify their bare target as
/// `Complex<T>`; the bounds live on the impl's own `generics`, not on
/// `target` itself), so callers needing a true impl identity combine both.
pub(crate) fn fmt_generics(generics: &[GenericParam]) -> String {
    if generics.is_empty() {
        return String::new();
    }
    let parts: Vec<String> = generics
        .iter()
        .map(|g| match g {
            GenericParam::Type { name, bounds } if bounds.is_empty() => name.clone(),
            GenericParam::Type { name, bounds } => format!("{name}: {}", bounds.join(" + ")),
            GenericParam::Const { name, ty } => format!("const {name}: {}", fmt_type(ty)),
        })
        .collect();
    format!("<{}>", parts.join(", "))
}

pub(crate) fn fmt_params(params: &[Param]) -> String {
    params
        .iter()
        .map(|p| match &p.ty {
            Some(t) => format!("{}: {}", p.name, fmt_type(t)),
            None => p.name.clone(),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn fmt_path(path: &Path) -> String {
    path.segments.join("::")
}

/// `pub(crate)`: reused by `driver.rs` to build a structural key for
/// grouping multi-file `impl` fragments by target type, and for comparing
/// parameter types when detecting duplicate/overloaded signatures.
pub(crate) fn fmt_type(ty: &Type) -> String {
    match &ty.kind {
        TypeKind::Path(path, args) => {
            if args.is_empty() {
                fmt_path(path)
            } else {
                let args: Vec<String> = args
                    .iter()
                    .map(|a| match a {
                        GenericArg::Type(t) => fmt_type(t),
                        GenericArg::Const(e) => fmt_expr(e),
                    })
                    .collect();
                format!("{}<{}>", fmt_path(path), args.join(", "))
            }
        }
        TypeKind::Array(elem, dim) => format!("[{}; {}]", fmt_type(elem), fmt_expr(dim)),
        TypeKind::Fn(params, ret) => {
            format!("({}) -> {}", params.iter().map(fmt_type).collect::<Vec<_>>().join(", "), fmt_type(ret))
        }
    }
}

/// `pub(crate)`: reused by `dump.rs`. Renders an explicit turbofish
/// (`::<f64, 4, 4>`), or `""` if `args` is empty — the overwhelmingly common
/// case, since a struct/call's generics are normally inferred rather than
/// spelled out. Shared by `Call`/`StructLit`'s own `fmt_expr` arms below.
pub(crate) fn fmt_turbofish(args: &[GenericArg]) -> String {
    if args.is_empty() {
        return String::new();
    }
    let args: Vec<String> =
        args.iter().map(|a| match a { GenericArg::Type(t) => fmt_type(t), GenericArg::Const(e) => fmt_expr(e) }).collect();
    format!("::<{}>", args.join(", "))
}

/// `pub(crate)`: reused by `dump.rs` to render an expression's surface form
/// alongside its inferred type.
pub(crate) fn fmt_expr(e: &Expr) -> String {
    match &e.kind {
        ExprKind::NumberLit { text, suffix } => match suffix {
            Some(s) => format!("{text}:{s}"),
            None => text.clone(),
        },
        ExprKind::ImaginaryLit { text, .. } => format!("{text}i"),
        ExprKind::BoolLit(b) => b.to_string(),
        ExprKind::Path(p) => fmt_path(p),
        ExprKind::Call(path, generics, args, mlir_attrs) => {
            let mut parts: Vec<String> = args.iter().map(fmt_expr).collect();
            parts.extend(mlir_attrs.iter().map(|(name, text)| format!("{name}: {text:?}")));
            format!("{}{}({})", fmt_path(path), fmt_turbofish(generics), parts.join(", "))
        }
        ExprKind::FieldAccess(base, name) => format!("{}.{name}", fmt_expr(base)),
        ExprKind::MethodCall(base, name, args) => {
            format!("{}.{name}({})", fmt_expr(base), args.iter().map(fmt_expr).collect::<Vec<_>>().join(", "))
        }
        ExprKind::Index(base, idx) => format!("{}[{}]", fmt_expr(base), fmt_expr(idx)),
        ExprKind::ArrayLit(elems) => format!("[{}]", elems.iter().map(fmt_expr).collect::<Vec<_>>().join(", ")),
        ExprKind::ArrayRepeat { value, count } => format!("[{}; {}]", fmt_expr(value), fmt_expr(count)),
        ExprKind::StructLit(path, generics, fields) => format!(
            "{}{}({})",
            fmt_path(path),
            fmt_turbofish(generics),
            fields.iter().map(|(name, v)| format!("{name}: {}", fmt_expr(v))).collect::<Vec<_>>().join(", ")
        ),
        ExprKind::If { cond, then_branch, else_branch } => {
            let mut s = format!("if {} {}", fmt_expr(cond), fmt_block_inline(then_branch));
            if let Some(eb) = else_branch {
                let _ = write!(s, " else {}", match &**eb {
                    ElseBranch::If(i) => fmt_expr(i),
                    ElseBranch::Block(b) => fmt_block_inline(b),
                });
            }
            s
        }
        ExprKind::While { cond, body } => format!("while {} {}", fmt_expr(cond), fmt_block_inline(body)),
        ExprKind::For { var, start, end, body } => {
            format!("for {var} in {}..{} {}", fmt_expr(start), fmt_expr(end), fmt_block_inline(body))
        }
        ExprKind::Block(b) => fmt_block_inline(b),
        ExprKind::Lambda { params, ret, body } => {
            let ret_ann = ret.as_ref().map(|t| format!(" -> {}", fmt_type(t))).unwrap_or_default();
            format!("fn({}){ret_ann} {}", fmt_params(params), fmt_block_inline(body))
        }
    }
}

/// A block rendered as a single-line `{ ... }` — used wherever a block appears
/// *inside* an expression (if/while/for bodies). Top-level fn/impl bodies use
/// `Printer::print_block_contents` instead, for proper multi-line indentation.
fn fmt_block_inline(b: &Block) -> String {
    let mut parts: Vec<String> = b
        .stmts
        .iter()
        .map(|s| match &s.kind {
            StmtKind::Let { mutable, name, ty, value } => {
                let mut_kw = if *mutable { "mut " } else { "" };
                let ty_ann = ty.as_ref().map(|t| format!(": {}", fmt_type(t))).unwrap_or_default();
                format!("let {mut_kw}{name}{ty_ann} = {};", fmt_expr(value))
            }
            StmtKind::Assign { target, value } => format!("{} = {};", fmt_expr(target), fmt_expr(value)),
            StmtKind::Expr(e) => format!("{};", fmt_expr(e)),
        })
        .collect();
    if let Some(tail) = &b.tail {
        parts.push(fmt_expr(tail));
    }
    if parts.is_empty() {
        "{}".to_string()
    } else {
        format!("{{ {} }}", parts.join(" "))
    }
}
