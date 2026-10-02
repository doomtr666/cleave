//! Unrolls `for` loops over heterogeneous collections
//! (`doc/plan-compile-time-sequences.md`, step 2).
//!
//! `for i in 0..t.len() { print(t[i]); }` with `t` a tuple or struct has a
//! different type at each `t[i]`, so the loop can't be one loop: it becomes one
//! copy of the body per value of `i`, with `i` replaced by that value. Each
//! copy is then an ordinary positional access (`t[0]`, `t[1]`, ...).
//!
//! Which loops, and how many copies, depends on types: inference records an
//! unroll request for a loop whose body indexes a struct or tuple by the loop
//! variable while its bounds fold (`Infer::unroll_requests`). This pass runs a
//! trial inference, rewrites the requested loops, and repeats until no request
//! is left (an inner loop can only be seen once its outer one is unrolled).
//! Programs with no loop indexing anything by its own variable skip it entirely.
//!
//! A copy that `break`s must leave the whole unrolled loop, not just its copy:
//! when the body breaks, the copies are wrapped in `loop { copy0; copy1; ...;
//! break; }`, so `break` keeps its meaning without any new construct.

use crate::ast::*;
use crate::callgraph;
use crate::registry::Registry;
use std::collections::HashMap;

/// More rounds than any real nesting of unrolled loops needs; a guard against
/// a request that would somehow reappear forever.
const MAX_ROUNDS: usize = 16;

pub fn unroll_heterogeneous_loops(mut program: Program, node_ids: &mut NodeIdGen) -> Program {
    for _ in 0..MAX_ROUNDS {
        if !program.items.iter().any(item_has_candidate_loop) {
            return program;
        }
        let registry = Registry::build(&program);
        let mut inference = callgraph::infer_program(&program, &registry);
        // `infer_program` covers top-level fns; a concrete impl's methods are
        // inferred here, for their loops. (Generic code unrolls per instance,
        // in `monomorphize.rs`.)
        for item in &program.items {
            let ItemKind::Impl(d) = &item.kind else { continue };
            if !d.generics.is_empty() {
                continue;
            }
            let mut targets = vec![d.target.clone()];
            targets.extend(d.extra_targets.iter().cloned());
            for f in &d.fns {
                if !f.body.as_ref().is_some_and(|b| any_expr_in_block(b, &mut is_candidate_loop)) {
                    continue;
                }
                let mut infer = crate::infer::Infer::new(&registry);
                let _ = infer.infer_impl_fn_generic_with_env(
                    &inference.global_env,
                    &d.algebra,
                    &[],
                    &targets,
                    f,
                    item.span,
                );
                inference.unroll_requests.append(&mut infer.unroll_requests);
            }
        }

        let requests: HashMap<NodeId, (u64, u64)> = inference
            .unroll_requests
            .iter()
            .map(|(id, start, end)| (*id, (*start, *end)))
            .collect();
        if requests.is_empty() {
            return program;
        }
        for item in &mut program.items {
            match &mut item.kind {
                ItemKind::Fn(f) => {
                    if let Some(body) = &mut f.body {
                        unroll_block(body, &requests, node_ids);
                        prune_block(body);
                    }
                }
                ItemKind::Impl(i) => {
                    for f in &mut i.fns {
                        if let Some(body) = &mut f.body {
                            unroll_block(body, &requests, node_ids);
                            prune_block(body);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    program
}

// ------------------------------------------------------------ folding `if`s

/// Replaces every `if` whose condition folds without any type information —
/// literals and operators on them, which every unrolled copy's index is —
/// by its taken branch, before inference ever sees the other one. A branch
/// that can never run then never has to type-check: in a loop unrolled over
/// a tuple, `if i == 1 { a = a + t[i] } else { b = b + t[i] }` only keeps,
/// in each copy, the branch that fits that copy's element type.
pub fn prune_constant_ifs(program: &mut Program) {
    for item in &mut program.items {
        match &mut item.kind {
            ItemKind::Fn(f) => {
                if let Some(body) = &mut f.body {
                    prune_block(body);
                }
            }
            ItemKind::Impl(i) => {
                for f in &mut i.fns {
                    if let Some(body) = &mut f.body {
                        prune_block(body);
                    }
                }
            }
            _ => {}
        }
    }
}

pub(crate) fn prune_block(b: &mut Block) {
    for s in &mut b.stmts {
        match &mut s.kind {
            StmtKind::Let { value, .. } => prune_expr(value),
            StmtKind::Assign { target, value } => {
                prune_expr(target);
                prune_expr(value);
            }
            StmtKind::Expr(e) => prune_expr(e),
            StmtKind::Break(v) => {
                if let Some(v) = v {
                    prune_expr(v);
                }
            }
        }
    }
    if let Some(t) = &mut b.tail {
        prune_expr(t);
    }
}

fn prune_expr(e: &mut Expr) {
    if let ExprKind::If {
        cond,
        then_branch,
        else_branch,
    } = &mut e.kind
    {
        if let Some(crate::infer::ConstValue::Bool(taken)) = fold_syntactic(cond) {
            let replacement = if taken {
                ExprKind::Block(std::mem::replace(then_branch, Block { stmts: Vec::new(), tail: None }))
            } else {
                match else_branch.take().map(|b| *b) {
                    Some(ElseBranch::Block(b)) => ExprKind::Block(b),
                    Some(ElseBranch::If(inner)) => inner.kind,
                    None => ExprKind::Block(Block {
                        stmts: Vec::new(),
                        tail: None,
                    }),
                }
            };
            e.kind = replacement;
            prune_expr(e);
            return;
        }
    }
    match &mut e.kind {
        ExprKind::NumberLit { .. }
        | ExprKind::ImaginaryLit { .. }
        | ExprKind::BoolLit(_)
        | ExprKind::Path(_)
        | ExprKind::PackRef(_) => {}
        ExprKind::Call(_, _, args, _) => args.iter_mut().for_each(prune_expr),
        ExprKind::FieldAccess(b, _) => prune_expr(b),
        ExprKind::Index(b, idx) => {
            prune_expr(b);
            idx.iter_mut().for_each(prune_expr);
        }
        ExprKind::ArrayLit(es) => es.iter_mut().for_each(prune_expr),
        ExprKind::ArrayRepeat { value, .. } => prune_expr(value),
        ExprKind::StructLit(_, _, fields) => fields.iter_mut().for_each(|(_, v)| prune_expr(v)),
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            prune_expr(cond);
            prune_block(then_branch);
            match else_branch.as_deref_mut() {
                Some(ElseBranch::If(x)) => prune_expr(x),
                Some(ElseBranch::Block(b)) => prune_block(b),
                None => {}
            }
        }
        ExprKind::While { cond, body } => {
            prune_expr(cond);
            prune_block(body);
        }
        ExprKind::For {
            start, end, body, ..
        } => {
            prune_expr(start);
            prune_expr(end);
            prune_block(body);
        }
        ExprKind::ForIn { iter, body, .. } => {
            prune_expr(iter);
            prune_block(body);
        }
        ExprKind::Loop { body } | ExprKind::Block(body) | ExprKind::Lambda { body, .. } => {
            prune_block(body)
        }
    }
}

/// A constant read off the syntax alone: integer and boolean literals, and
/// operators (`resolve.rs` marks them, `Path::operator`) applied to them.
fn fold_syntactic(e: &Expr) -> Option<crate::infer::ConstValue> {
    use crate::infer::ConstValue;
    match &e.kind {
        ExprKind::BoolLit(b) => Some(ConstValue::Bool(*b)),
        ExprKind::NumberLit { text, .. } => text.parse::<u64>().ok().map(ConstValue::Int),
        ExprKind::Call(path, _, args, _) if path.operator => {
            let op = path.segments.last()?;
            let values: Vec<ConstValue> = args.iter().map(fold_syntactic).collect::<Option<_>>()?;
            match values.as_slice() {
                [a] => crate::const_eval::eval_unop(op, *a),
                [a, b] => crate::const_eval::eval_binop(op, *a, *b),
                _ => None,
            }
        }
        _ => None,
    }
}

// ------------------------------------------------------------ candidates

fn item_has_candidate_loop(item: &Item) -> bool {
    let bodies: Vec<&Block> = match &item.kind {
        ItemKind::Fn(f) => f.body.iter().collect(),
        ItemKind::Impl(i) => i.fns.iter().filter_map(|f| f.body.as_ref()).collect(),
        _ => Vec::new(),
    };
    bodies.into_iter().any(|b| any_expr_in_block(b, &mut is_candidate_loop))
}

/// A `for` whose body indexes something by the loop variable alone (`x[i]`),
/// or a comprehension (always rewritten, once its bounds fold).
fn is_candidate_loop(e: &Expr) -> bool {
    if comprehension_parts(e).is_some() {
        return true;
    }
    let ExprKind::For { var, body, .. } = &e.kind else {
        return false;
    };
    any_expr_in_block(body, &mut |inner| match &inner.kind {
        ExprKind::Index(_, indices) => matches!(
            indices.as_slice(),
            [Node { kind: ExprKind::Path(p), .. }] if p.segments.len() == 1 && &p.segments[0] == var
        ),
        _ => false,
    })
}

fn any_expr_in_block(block: &Block, pred: &mut dyn FnMut(&Expr) -> bool) -> bool {
    block.stmts.iter().any(|s| match &s.kind {
        StmtKind::Let { value, .. } => any_expr(value, pred),
        StmtKind::Assign { target, value } => any_expr(target, pred) || any_expr(value, pred),
        StmtKind::Expr(e) => any_expr(e, pred),
        StmtKind::Break(v) => v.as_ref().is_some_and(|v| any_expr(v, pred)),
    }) || block.tail.as_deref().is_some_and(|t| any_expr(t, pred))
}

fn any_expr(e: &Expr, pred: &mut dyn FnMut(&Expr) -> bool) -> bool {
    if pred(e) {
        return true;
    }
    match &e.kind {
        ExprKind::NumberLit { .. }
        | ExprKind::ImaginaryLit { .. }
        | ExprKind::BoolLit(_)
        | ExprKind::Path(_)
        | ExprKind::PackRef(_) => false,
        ExprKind::Call(_, _, args, _) => args.iter().any(|a| any_expr(a, pred)),
        ExprKind::FieldAccess(b, _) => any_expr(b, pred),
        ExprKind::Index(b, idx) => any_expr(b, pred) || idx.iter().any(|i| any_expr(i, pred)),
        ExprKind::ArrayLit(es) => es.iter().any(|x| any_expr(x, pred)),
        ExprKind::ArrayRepeat { value, count } => any_expr(value, pred) || any_expr(count, pred),
        ExprKind::StructLit(_, _, fields) => fields.iter().any(|(_, v)| any_expr(v, pred)),
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            any_expr(cond, pred)
                || any_expr_in_block(then_branch, pred)
                || match else_branch.as_deref() {
                    Some(ElseBranch::If(x)) => any_expr(x, pred),
                    Some(ElseBranch::Block(b)) => any_expr_in_block(b, pred),
                    None => false,
                }
        }
        ExprKind::While { cond, body } => any_expr(cond, pred) || any_expr_in_block(body, pred),
        ExprKind::For {
            start, end, body, ..
        } => any_expr(start, pred) || any_expr(end, pred) || any_expr_in_block(body, pred),
        ExprKind::ForIn { iter, body, .. } => any_expr(iter, pred) || any_expr_in_block(body, pred),
        ExprKind::Loop { body } | ExprKind::Block(body) => any_expr_in_block(body, pred),
        ExprKind::Lambda { body, .. } => any_expr_in_block(body, pred),
    }
}

// ------------------------------------------------------------ rewriting

pub(crate) fn unroll_block(block: &mut Block, requests: &HashMap<NodeId, (u64, u64)>, ids: &mut NodeIdGen) {
    for stmt in &mut block.stmts {
        match &mut stmt.kind {
            StmtKind::Let { value, .. } => unroll_expr(value, requests, ids),
            StmtKind::Assign { target, value } => {
                unroll_expr(target, requests, ids);
                unroll_expr(value, requests, ids);
            }
            StmtKind::Expr(e) => unroll_expr(e, requests, ids),
            StmtKind::Break(v) => {
                if let Some(v) = v {
                    unroll_expr(v, requests, ids);
                }
            }
        }
    }
    if let Some(tail) = &mut block.tail {
        unroll_expr(tail, requests, ids);
    }
}

fn unroll_expr(e: &mut Expr, requests: &HashMap<NodeId, (u64, u64)>, ids: &mut NodeIdGen) {
    if let ExprKind::For { var, body, .. } = &e.kind {
        if let Some(&(start, end)) = requests.get(&e.id) {
            let (var, body) = (var.clone(), body.clone());
            e.kind = unrolled(&var, &body, start, end, e.span, ids);
            return;
        }
    }
    if let Some((var, body)) = comprehension_parts(e) {
        if let Some(&(start, end)) = requests.get(&e.id) {
            let (var, body) = (var.to_string(), body.clone());
            e.kind = collected(&var, &body, start, end, e.span, ids);
            return;
        }
    }
    match &mut e.kind {
        ExprKind::NumberLit { .. }
        | ExprKind::ImaginaryLit { .. }
        | ExprKind::BoolLit(_)
        | ExprKind::Path(_)
        | ExprKind::PackRef(_) => {}
        ExprKind::Call(_, _, args, _) => args.iter_mut().for_each(|a| unroll_expr(a, requests, ids)),
        ExprKind::FieldAccess(b, _) => unroll_expr(b, requests, ids),
        ExprKind::Index(b, idx) => {
            unroll_expr(b, requests, ids);
            idx.iter_mut().for_each(|i| unroll_expr(i, requests, ids));
        }
        ExprKind::ArrayLit(es) => es.iter_mut().for_each(|x| unroll_expr(x, requests, ids)),
        ExprKind::ArrayRepeat { value, .. } => unroll_expr(value, requests, ids),
        ExprKind::StructLit(_, _, fields) => {
            fields.iter_mut().for_each(|(_, v)| unroll_expr(v, requests, ids))
        }
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            unroll_expr(cond, requests, ids);
            unroll_block(then_branch, requests, ids);
            match else_branch.as_deref_mut() {
                Some(ElseBranch::If(x)) => unroll_expr(x, requests, ids),
                Some(ElseBranch::Block(b)) => unroll_block(b, requests, ids),
                None => {}
            }
        }
        ExprKind::While { cond, body } => {
            unroll_expr(cond, requests, ids);
            unroll_block(body, requests, ids);
        }
        ExprKind::For {
            start, end, body, ..
        } => {
            unroll_expr(start, requests, ids);
            unroll_expr(end, requests, ids);
            unroll_block(body, requests, ids);
        }
        ExprKind::ForIn { iter, body, .. } => {
            unroll_expr(iter, requests, ids);
            unroll_block(body, requests, ids);
        }
        ExprKind::Loop { body } | ExprKind::Block(body) | ExprKind::Lambda { body, .. } => {
            unroll_block(body, requests, ids)
        }
    }
}

/// A comprehension's variable and body (`ast::COMPREHENSION`).
fn comprehension_parts(e: &Expr) -> Option<(&str, &Block)> {
    let ExprKind::Call(path, _, args, _) = &e.kind else { return None };
    if path.segments != [COMPREHENSION] {
        return None;
    }
    let [_, _, lambda] = args.as_slice() else { return None };
    let ExprKind::Lambda { params, body, .. } = &lambda.kind else { return None };
    let [param] = params.as_slice() else { return None };
    Some((param.name.as_str(), body))
}

/// The replacement of `[for var in start..end: body]`: the tuple of the body's
/// copies, one per index (the copy itself when there is only one), handed to
/// `Collect::collect` for whatever type the context asks of it
/// (`ast::comprehension_collect`); `()` when there is none.
fn collected(var: &str, body: &Block, start: u64, end: u64, span: Span, ids: &mut NodeIdGen) -> ExprKind {
    let mut copies: Vec<Expr> = (start..end)
        .map(|k| {
            let mut copy = body.clone();
            renumber_block(&mut copy, ids);
            substitute_block(&mut copy, var, k);
            Node {
                id: ids.next(),
                span,
                kind: ExprKind::Block(copy),
            }
        })
        .collect();
    let source = match copies.len() {
        0 => {
            return ExprKind::Block(Block {
                stmts: Vec::new(),
                tail: None,
            });
        }
        1 => copies.pop().unwrap(),
        n => Node {
            id: ids.next(),
            span,
            kind: ExprKind::StructLit(
                Path::single(tuple_struct_name(n)),
                Vec::new(),
                copies.into_iter().enumerate().map(|(i, c)| (i.to_string(), c)).collect(),
            ),
        },
    };
    ExprKind::Call(comprehension_collect(), Vec::new(), vec![source], Vec::new())
}

/// The replacement of an unrolled `for var in start..end { body }`.
fn unrolled(var: &str, body: &Block, start: u64, end: u64, span: Span, ids: &mut NodeIdGen) -> ExprKind {
    let mut stmts: Vec<Stmt> = (start..end)
        .map(|k| {
            let mut copy = body.clone();
            renumber_block(&mut copy, ids);
            substitute_block(&mut copy, var, k);
            let copy_expr = Node {
                id: ids.next(),
                span,
                kind: ExprKind::Block(copy),
            };
            Node {
                id: ids.next(),
                span,
                kind: StmtKind::Expr(copy_expr),
            }
        })
        .collect();
    if !breaks_own_loop(body) {
        return ExprKind::Block(Block { stmts, tail: None });
    }
    stmts.push(Node {
        id: ids.next(),
        span,
        kind: StmtKind::Break(None),
    });
    let inner = Node {
        id: ids.next(),
        span,
        kind: ExprKind::Loop {
            body: Block { stmts, tail: None },
        },
    };
    ExprKind::Block(Block {
        stmts: vec![Node {
            id: ids.next(),
            span,
            kind: StmtKind::Expr(inner),
        }],
        tail: None,
    })
}

/// Whether `body` contains a `break` aimed at the loop it belongs to — one not
/// inside a nested loop or lambda of its own.
fn breaks_own_loop(body: &Block) -> bool {
    fn block(b: &Block) -> bool {
        b.stmts.iter().any(|s| match &s.kind {
            StmtKind::Break(_) => true,
            StmtKind::Let { value, .. } => expr(value),
            StmtKind::Assign { value, .. } => expr(value),
            StmtKind::Expr(e) => expr(e),
        }) || b.tail.as_deref().is_some_and(expr)
    }
    fn expr(e: &Expr) -> bool {
        match &e.kind {
            ExprKind::If {
                then_branch,
                else_branch,
                ..
            } => {
                block(then_branch)
                    || match else_branch.as_deref() {
                        Some(ElseBranch::If(x)) => expr(x),
                        Some(ElseBranch::Block(b)) => block(b),
                        None => false,
                    }
            }
            ExprKind::Block(b) => block(b),
            // A nested loop's or lambda's own `break` is its own.
            _ => false,
        }
    }
    block(body)
}

// ------------------------------------------------------------ fresh node ids

fn renumber_block(b: &mut Block, ids: &mut NodeIdGen) {
    for s in &mut b.stmts {
        s.id = ids.next();
        match &mut s.kind {
            StmtKind::Let { value, ty, .. } => {
                if let Some(t) = ty {
                    renumber_type(t, ids);
                }
                renumber_expr(value, ids);
            }
            StmtKind::Assign { target, value } => {
                renumber_expr(target, ids);
                renumber_expr(value, ids);
            }
            StmtKind::Expr(e) => renumber_expr(e, ids),
            StmtKind::Break(v) => {
                if let Some(v) = v {
                    renumber_expr(v, ids);
                }
            }
        }
    }
    if let Some(t) = &mut b.tail {
        renumber_expr(t, ids);
    }
}

fn renumber_generic_args(args: &mut [GenericArg], ids: &mut NodeIdGen) {
    for g in args {
        match g {
            GenericArg::Type(t) => renumber_type(t, ids),
            GenericArg::Const(e) => renumber_expr(e, ids),
        }
    }
}

pub(crate) fn renumber_type(t: &mut Type, ids: &mut NodeIdGen) {
    t.id = ids.next();
    match &mut t.kind {
        TypeKind::Path(_, args) => renumber_generic_args(args, ids),
        TypeKind::Array(elem, size) => {
            renumber_type(elem, ids);
            renumber_expr(size, ids);
        }
        TypeKind::Fn(params, ret) => {
            params.iter_mut().for_each(|p| renumber_type(p, ids));
            renumber_type(ret, ids);
        }
        _ => {}
    }
}

fn renumber_expr(e: &mut Expr, ids: &mut NodeIdGen) {
    e.id = ids.next();
    match &mut e.kind {
        ExprKind::NumberLit { .. }
        | ExprKind::ImaginaryLit { .. }
        | ExprKind::BoolLit(_)
        | ExprKind::Path(_)
        | ExprKind::PackRef(_) => {}
        ExprKind::Call(_, generics, args, _) => {
            renumber_generic_args(generics, ids);
            args.iter_mut().for_each(|a| renumber_expr(a, ids));
        }
        ExprKind::FieldAccess(b, _) => renumber_expr(b, ids),
        ExprKind::Index(b, idx) => {
            renumber_expr(b, ids);
            idx.iter_mut().for_each(|i| renumber_expr(i, ids));
        }
        ExprKind::ArrayLit(es) => es.iter_mut().for_each(|x| renumber_expr(x, ids)),
        ExprKind::ArrayRepeat { value, count } => {
            renumber_expr(value, ids);
            renumber_expr(count, ids);
        }
        ExprKind::StructLit(_, generics, fields) => {
            renumber_generic_args(generics, ids);
            fields.iter_mut().for_each(|(_, v)| renumber_expr(v, ids));
        }
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            renumber_expr(cond, ids);
            renumber_block(then_branch, ids);
            match else_branch.as_deref_mut() {
                Some(ElseBranch::If(x)) => renumber_expr(x, ids),
                Some(ElseBranch::Block(b)) => renumber_block(b, ids),
                None => {}
            }
        }
        ExprKind::While { cond, body } => {
            renumber_expr(cond, ids);
            renumber_block(body, ids);
        }
        ExprKind::For {
            start, end, body, ..
        } => {
            renumber_expr(start, ids);
            renumber_expr(end, ids);
            renumber_block(body, ids);
        }
        ExprKind::ForIn { iter, body, .. } => {
            renumber_expr(iter, ids);
            renumber_block(body, ids);
        }
        ExprKind::Loop { body } | ExprKind::Block(body) => renumber_block(body, ids),
        ExprKind::Lambda { params, ret, body } => {
            for p in params {
                if let Some(t) = &mut p.ty {
                    renumber_type(t, ids);
                }
            }
            if let Some(r) = ret {
                renumber_type(r, ids);
            }
            renumber_block(body, ids);
        }
    }
}

// ------------------------------------------------------------ loop variable -> constant

/// Replaces every read of `var` in `b` by the literal `k`, stopping where a
/// binding of the same name shadows it.
fn substitute_block(b: &mut Block, var: &str, k: u64) {
    for s in &mut b.stmts {
        match &mut s.kind {
            StmtKind::Let { name, value, .. } => {
                substitute_expr(value, var, k);
                if name == var {
                    return;
                }
            }
            StmtKind::Assign { target, value } => {
                substitute_expr(target, var, k);
                substitute_expr(value, var, k);
            }
            StmtKind::Expr(e) => substitute_expr(e, var, k),
            StmtKind::Break(v) => {
                if let Some(v) = v {
                    substitute_expr(v, var, k);
                }
            }
        }
    }
    if let Some(t) = &mut b.tail {
        substitute_expr(t, var, k);
    }
}

fn substitute_expr(e: &mut Expr, var: &str, k: u64) {
    match &mut e.kind {
        ExprKind::Path(p) if p.segments.len() == 1 && p.segments[0] == var => {
            e.kind = ExprKind::NumberLit {
                text: k.to_string(),
                suffix: None,
            };
        }
        ExprKind::NumberLit { .. }
        | ExprKind::ImaginaryLit { .. }
        | ExprKind::BoolLit(_)
        | ExprKind::Path(_)
        | ExprKind::PackRef(_) => {}
        ExprKind::Call(_, _, args, _) => args.iter_mut().for_each(|a| substitute_expr(a, var, k)),
        ExprKind::FieldAccess(b, _) => substitute_expr(b, var, k),
        ExprKind::Index(b, idx) => {
            substitute_expr(b, var, k);
            idx.iter_mut().for_each(|i| substitute_expr(i, var, k));
        }
        ExprKind::ArrayLit(es) => es.iter_mut().for_each(|x| substitute_expr(x, var, k)),
        ExprKind::ArrayRepeat { value, .. } => substitute_expr(value, var, k),
        ExprKind::StructLit(_, _, fields) => {
            fields.iter_mut().for_each(|(_, v)| substitute_expr(v, var, k))
        }
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            substitute_expr(cond, var, k);
            substitute_block(then_branch, var, k);
            match else_branch.as_deref_mut() {
                Some(ElseBranch::If(x)) => substitute_expr(x, var, k),
                Some(ElseBranch::Block(b)) => substitute_block(b, var, k),
                None => {}
            }
        }
        ExprKind::While { cond, body } => {
            substitute_expr(cond, var, k);
            substitute_block(body, var, k);
        }
        ExprKind::For {
            var: inner,
            start,
            end,
            body,
        } => {
            substitute_expr(start, var, k);
            substitute_expr(end, var, k);
            if inner != var {
                substitute_block(body, var, k);
            }
        }
        ExprKind::ForIn {
            var: inner,
            iter,
            body,
        } => {
            substitute_expr(iter, var, k);
            if inner != var {
                substitute_block(body, var, k);
            }
        }
        ExprKind::Loop { body } | ExprKind::Block(body) => substitute_block(body, var, k),
        ExprKind::Lambda { params, body, .. } => {
            if !params.iter().any(|p| p.name == var) {
                substitute_block(body, var, k);
            }
        }
    }
}
