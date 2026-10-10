//! Enums and `match` (`doc/plan-sum-types.md`), lowered right after the
//! crates are merged, before any other pass: the rest of the compiler sees
//! structs, functions, field reads and `if`s.
//!
//! - `enum E<G> { V(T0, T1), W }` becomes `struct E<G> { tag: i32, V.0: T0,
//!   V.1: T1 }` (a sum of fields, not a union) flagged `zero_fill`, and one
//!   constructor per variant, `fn V<G>(x0: T0, x1: T1) -> E<G>`, which builds
//!   it with its tag and its own fields: every other field is left all zero.
//!   An inactive variant's field releases nothing (a null pointer, a null
//!   tensor descriptor), so the struct's ordinary release cascade is right.
//! - `match e { V(a, _) => x, _ => y }` becomes `{ let <match#n> = e; if
//!   <match#n>.tag == k { let a = <match#n>.V.0; x } else { y } }`, the last
//!   arm the final `else`.
//! - A variant without data written as a bare name (`None`) or qualified
//!   (`Option::None`) is a call to its constructor; a qualified call
//!   (`Option::Some(x)`) a call to the bare one.
//!
//! Errors: a variant name two enums share (or one twice), an unknown variant
//! in a pattern, patterns of two enums in one `match`, a wrong number of
//! bindings, a variant matched twice, an arm after `_`, and a `match` that
//! doesn't cover every variant, naming the missing ones.

use crate::ast::*;
use crate::collections::HashMap;
use crate::diag::Diagnostic;

/// The tag field: which variant a value is.
pub const TAG_FIELD: &str = "tag";

/// The field holding the `index`th piece of data of `variant`.
pub fn variant_field(variant: &str, index: usize) -> String {
    format!("{variant}.{index}")
}

struct VariantInfo {
    enum_name: String,
    tag: usize,
    arity: usize,
}

struct Enums {
    variants: HashMap<String, VariantInfo>,
    /// Each enum's variant names, in declaration order.
    order: HashMap<String, Vec<String>>,
}

/// Lowers every `enum` and `match` of `program` (module doc comment).
pub fn desugar_enums(program: Program, node_ids: &mut NodeIdGen) -> Result<Program, Vec<Diagnostic>> {
    let mut errors = Vec::new();
    let mut enums = Enums { variants: HashMap::default(), order: HashMap::default() };
    let mut items = Vec::new();
    let mut lowered = Vec::new();
    for item in program.items {
        let ItemKind::Enum(decl) = &item.kind else {
            items.push(item);
            continue;
        };
        if decl.generics.iter().any(GenericParam::is_variadic) {
            errors.push(Diagnostic::error(format!("enum `{}`: a variadic generic isn't supported", decl.name), item.span));
            continue;
        }
        let mut names = Vec::new();
        for (tag, variant) in decl.variants.iter().enumerate() {
            if let Some(other) = enums.variants.get(&variant.name) {
                errors.push(Diagnostic::error(
                    format!("variant `{}` is declared by `{}` already: variant names are unique", variant.name, other.enum_name),
                    item.span,
                ));
                continue;
            }
            enums.variants.insert(
                variant.name.clone(),
                VariantInfo { enum_name: decl.name.clone(), tag, arity: variant.fields.len() },
            );
            names.push(variant.name.clone());
        }
        enums.order.insert(decl.name.clone(), names);
        lowered.push((item.span, decl.clone()));
    }
    for (span, decl) in &lowered {
        items.push(enum_struct(decl, *span, node_ids));
        for (tag, variant) in decl.variants.iter().enumerate() {
            items.push(constructor(decl, tag, variant, *span, node_ids));
        }
    }
    let mut counter = 0;
    let mut rewrite = |expr: &mut Expr| rewrite_expr(expr, &enums, node_ids, &mut counter, &mut errors);
    for item in &mut items {
        match &mut item.kind {
            ItemKind::Fn(f) => {
                if let Some(body) = &mut f.body {
                    for_each_expr_in_block_mut(body, &mut rewrite);
                }
            }
            ItemKind::Impl(i) => {
                for f in &mut i.fns {
                    if let Some(body) = &mut f.body {
                        for_each_expr_in_block_mut(body, &mut rewrite);
                    }
                }
            }
            _ => {}
        }
    }
    if errors.is_empty() { Ok(Program { items }) } else { Err(errors) }
}

fn node<K>(node_ids: &mut NodeIdGen, span: Span, kind: K) -> Node<K> {
    Node { id: node_ids.next(), span, kind }
}

fn path_type(node_ids: &mut NodeIdGen, span: Span, name: &str, args: Vec<GenericArg>) -> Type {
    node(node_ids, span, TypeKind::Path(Path::single(name), args))
}

/// `E<G>` as written in the enum's own scope: its generics as arguments.
fn enum_type(decl: &EnumDecl, span: Span, node_ids: &mut NodeIdGen) -> Type {
    let args = decl
        .generics
        .iter()
        .map(|g| match g {
            GenericParam::Type { name, .. } => GenericArg::Type(path_type(node_ids, span, name, Vec::new())),
            GenericParam::Const { name, .. } => GenericArg::Const(node(node_ids, span, ExprKind::Path(Path::single(name.clone())))),
        })
        .collect();
    path_type(node_ids, span, &decl.name, args)
}

fn enum_struct(decl: &EnumDecl, span: Span, node_ids: &mut NodeIdGen) -> Item {
    let mut fields = vec![Field { name: TAG_FIELD.to_string(), ty: path_type(node_ids, span, "i32", Vec::new()) }];
    for variant in &decl.variants {
        for (i, ty) in variant.fields.iter().enumerate() {
            fields.push(Field { name: variant_field(&variant.name, i), ty: ty.clone() });
        }
    }
    node(
        node_ids,
        span,
        ItemKind::Struct(StructDecl { name: decl.name.clone(), generics: decl.generics.clone(), fields, zero_fill: true }),
    )
}

fn constructor(decl: &EnumDecl, tag: usize, variant: &Variant, span: Span, node_ids: &mut NodeIdGen) -> Item {
    let params: Vec<Param> = variant
        .fields
        .iter()
        .enumerate()
        .map(|(i, ty)| Param { name: format!("x{i}"), ty: Some(ty.clone()), mutable: false })
        .collect();
    let mut fields = vec![(
        TAG_FIELD.to_string(),
        node(node_ids, span, ExprKind::NumberLit { text: tag.to_string(), suffix: None }),
    )];
    for (i, p) in params.iter().enumerate() {
        fields.push((variant_field(&variant.name, i), node(node_ids, span, ExprKind::Path(Path::single(p.name.clone())))));
    }
    let literal = node(node_ids, span, ExprKind::StructLit(Path::single(decl.name.clone()), Vec::new(), fields));
    let ret = enum_type(decl, span, node_ids);
    node(
        node_ids,
        span,
        ItemKind::Fn(FnDecl {
            name: variant.name.clone(),
            attrs: Vec::new(),
            is_extern: false,
            extern_symbol: None,
            is_export: false,
            export_symbol: None,
            generics: decl.generics.clone(),
            params,
            ret: Some(ret),
            body: Some(Block { stmts: Vec::new(), tail: Some(Box::new(literal)) }),
            derivative_of: None,
            is_grad: false,
            grad_target_param: None,
            grad_target_index: None,
        }),
    )
}

/// The variant `path` names, bare (`Some`) or qualified (`Option::Some`).
fn variant_of<'e>(path: &Path, enums: &'e Enums) -> Option<(&'e str, &'e VariantInfo)> {
    let (name, info) = match path.segments.as_slice() {
        [name] => (name, enums.variants.get(name)?),
        [enum_name, name] => {
            let info = enums.variants.get(name)?;
            if &info.enum_name != enum_name {
                return None;
            }
            (name, info)
        }
        _ => return None,
    };
    let (key, _) = enums.variants.get_key_value(name)?;
    Some((key.as_str(), info))
}

fn rewrite_expr(expr: &mut Expr, enums: &Enums, node_ids: &mut NodeIdGen, counter: &mut usize, errors: &mut Vec<Diagnostic>) {
    let span = expr.span;
    match &mut expr.kind {
        ExprKind::Path(path) if !path.operator => {
            if let Some((name, info)) = variant_of(path, enums) {
                if info.arity == 0 {
                    expr.kind = ExprKind::Call(Path::single(name), Vec::new(), Vec::new(), Vec::new());
                }
            }
        }
        ExprKind::Call(path, ..) if path.segments.len() == 2 => {
            if let Some((name, _)) = variant_of(path, enums) {
                *path = Path::single(name);
            }
        }
        ExprKind::Match { .. } => {
            let ExprKind::Match { scrutinee, arms } = std::mem::replace(&mut expr.kind, ExprKind::BoolLit(false)) else {
                unreachable!()
            };
            match lower_match(*scrutinee, arms, span, enums, node_ids, counter) {
                Ok(kind) => expr.kind = kind,
                Err(mut e) => errors.append(&mut e),
            }
        }
        _ => {}
    }
}

/// `match` as `{ let <match#n> = scrutinee; if .. }` (module doc comment).
fn lower_match(
    scrutinee: Expr,
    arms: Vec<MatchArm>,
    span: Span,
    enums: &Enums,
    node_ids: &mut NodeIdGen,
    counter: &mut usize,
) -> Result<ExprKind, Vec<Diagnostic>> {
    let mut errors = Vec::new();
    // Which enum: the one the patterns' variants belong to.
    let mut enum_name: Option<&str> = None;
    let mut covered: Vec<&str> = Vec::new();
    let mut wildcard_at: Option<usize> = None;
    let mut resolved: Vec<Option<(&str, &VariantInfo)>> = Vec::new();
    for (i, arm) in arms.iter().enumerate() {
        if wildcard_at.is_some() {
            errors.push(Diagnostic::error("this arm is never reached: an arm before it is `_`", arm.span));
        }
        match &arm.pattern {
            Pattern::Wildcard => {
                wildcard_at.get_or_insert(i);
                resolved.push(None);
            }
            Pattern::Variant { path, bindings } => {
                let Some((name, info)) = variant_of(path, enums) else {
                    errors.push(Diagnostic::error(format!("no variant `{}`", path.segments.join("::")), arm.span));
                    resolved.push(None);
                    continue;
                };
                match enum_name {
                    None => enum_name = Some(&info.enum_name),
                    Some(e) if e != info.enum_name => errors.push(Diagnostic::error(
                        format!("`{name}` is a variant of `{}`, not of `{e}`", info.enum_name),
                        arm.span,
                    )),
                    Some(_) => {}
                }
                if bindings.len() != info.arity {
                    errors.push(Diagnostic::error(
                        format!("`{name}` holds {} value(s), the pattern binds {}", info.arity, bindings.len()),
                        arm.span,
                    ));
                }
                if covered.contains(&name) {
                    errors.push(Diagnostic::error(format!("`{name}` is matched twice"), arm.span));
                }
                covered.push(name);
                resolved.push(Some((name, info)));
            }
        }
    }
    let exhaustive = wildcard_at.is_some()
        || enum_name.is_some_and(|e| enums.order[e].iter().all(|v| covered.contains(&v.as_str())));
    if !exhaustive {
        let missing: Vec<String> = match enum_name {
            Some(e) => enums.order[e].iter().filter(|v| !covered.contains(&v.as_str())).map(|v| format!("`{v}`")).collect(),
            None => Vec::new(),
        };
        errors.push(Diagnostic::error(format!("`match` doesn't cover {}", missing.join(", ")), span));
    }
    if !errors.is_empty() {
        return Err(errors);
    }

    *counter += 1;
    let scrutinee_name = format!("<match#{counter}>");
    let reachable = wildcard_at.map_or(arms.len(), |w| w + 1);
    let mut chain: Option<Expr> = None;
    for (index, (arm, variant)) in arms.into_iter().zip(resolved).take(reachable).enumerate().rev() {
        let mut stmts = Vec::new();
        if let (Pattern::Variant { bindings, .. }, Some((name, _))) = (&arm.pattern, variant) {
            for (i, binding) in bindings.iter().enumerate() {
                if binding == "_" {
                    continue;
                }
                let base = node(node_ids, arm.span, ExprKind::Path(Path::single(scrutinee_name.clone())));
                let value = node(node_ids, arm.span, ExprKind::FieldAccess(Box::new(base), variant_field(name, i)));
                stmts.push(node(
                    node_ids,
                    arm.span,
                    StmtKind::Let { mutable: false, name: binding.clone(), ty: None, value },
                ));
            }
        }
        let body = Block { stmts, tail: Some(Box::new(arm.body)) };
        let last = index + 1 == reachable;
        chain = Some(match (variant, last) {
            // The last arm, or `_`: whatever is left.
            (_, true) | (None, _) => node(node_ids, arm.span, ExprKind::Block(body)),
            (Some((_, info)), false) => {
                let base = node(node_ids, arm.span, ExprKind::Path(Path::single(scrutinee_name.clone())));
                let tag = node(node_ids, arm.span, ExprKind::FieldAccess(Box::new(base), TAG_FIELD.to_string()));
                let k = node(node_ids, arm.span, ExprKind::NumberLit { text: info.tag.to_string(), suffix: None });
                let cond = node(node_ids, arm.span, ExprKind::Call(Path::operator("eq"), Vec::new(), vec![tag, k], Vec::new()));
                let else_branch = chain.take().map(|next| {
                    Box::new(match next.kind {
                        ExprKind::Block(block) => ElseBranch::Block(block),
                        _ => ElseBranch::If(next),
                    })
                });
                node(node_ids, arm.span, ExprKind::If { cond: Box::new(cond), then_branch: body, else_branch })
            }
        });
    }
    let bind = node(
        node_ids,
        span,
        StmtKind::Let { mutable: false, name: scrutinee_name, ty: None, value: scrutinee },
    );
    Ok(ExprKind::Block(Block { stmts: vec![bind], tail: chain.map(Box::new) }))
}
