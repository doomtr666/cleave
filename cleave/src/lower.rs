//! Converts Pest's parse tree into the typed [`crate::ast`] — the only place
//! that ever touches a `pest::iterators::Pair` directly. Everything downstream
//! (type inference, monomorphization, ...) works on `ast` types only.
//!
//! Operator desugaring (`a + b` => `Call(add, [a, b])`) and the Fortran-style
//! multi-dim sugar (`[f64; 3, 4]`, `a[i, j]`) happen here, at construction
//! time, per `grammar.md` — nothing downstream needs to know these sugars
//! ever existed.

use crate::ast::*;
use crate::parser::Rule;
use pest::iterators::{Pair, Pairs};

pub struct Lowerer {
    file: FileId,
    ids: NodeIdGen,
}

impl Lowerer {
    pub fn new(file: FileId) -> Self {
        Lowerer { file, ids: NodeIdGen::default() }
    }

    /// Like `new`, but continuing an existing `NodeIdGen` rather than
    /// starting a fresh one at 0 — needed so `NodeId`s stay unique across an
    /// entire multi-file compilation, not just within one file. `new` still
    /// starts fresh, for every single-file caller (tests, isolated
    /// `infer_fn`-only usage) that never needs cross-file uniqueness.
    pub fn with_ids(file: FileId, ids: NodeIdGen) -> Self {
        Lowerer { file, ids }
    }

    /// Hands back the (possibly advanced) generator so the next file in the
    /// same compilation can continue from where this one left off.
    pub fn into_ids(self) -> NodeIdGen {
        self.ids
    }

    fn span_of(&self, pair: &Pair<Rule>) -> Span {
        let s = pair.as_span();
        Span { file: self.file, start: s.start(), end: s.end() }
    }

    fn join(&self, a: Span, b: Span) -> Span {
        Span { file: self.file, start: a.start, end: b.end }
    }

    fn wrap<T>(&mut self, span: Span, kind: T) -> Node<T> {
        Node { id: self.ids.next(), span, kind }
    }

    // ---------------------------------------------------------------- program / items

    pub fn lower_program(&mut self, pair: Pair<Rule>) -> Program {
        debug_assert_eq!(pair.as_rule(), Rule::program);
        let mut items = Vec::new();
        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::item {
                items.push(self.lower_item(inner));
            }
        }
        Program { items }
    }

    fn lower_item(&mut self, pair: Pair<Rule>) -> Item {
        let span = self.span_of(&pair);
        let inner = pair.into_inner().next().unwrap();
        let kind = match inner.as_rule() {
            Rule::use_decl => ItemKind::Use(self.lower_use_decl(inner)),
            Rule::struct_decl => ItemKind::Struct(self.lower_struct_decl(inner)),
            Rule::algebra_decl => ItemKind::Algebra(self.lower_algebra_decl(inner)),
            Rule::impl_decl => {
                let variant = inner.into_inner().next().unwrap();
                match variant.as_rule() {
                    Rule::algebra_impl => ItemKind::Impl(self.lower_algebra_impl(variant)),
                    Rule::inherent_impl => ItemKind::InherentImpl(self.lower_inherent_impl(variant)),
                    r => unreachable!("impl_decl: unexpected rule {r:?}"),
                }
            }
            Rule::fn_decl => ItemKind::Fn(self.lower_fn_decl(inner)),
            r => unreachable!("item: unexpected rule {r:?}"),
        };
        self.wrap(span, kind)
    }

    fn lower_use_decl(&mut self, pair: Pair<Rule>) -> Path {
        let path_pair = pair.into_inner().next().unwrap();
        self.lower_path(path_pair)
    }

    fn lower_path(&mut self, pair: Pair<Rule>) -> Path {
        let segments = pair.into_inner().map(|p| p.as_str().to_string()).collect();
        Path { segments }
    }

    // ---------------------------------------------------------------- fn

    fn lower_fn_decl(&mut self, pair: Pair<Rule>) -> FnDecl {
        let mut inner = pair.into_inner().peekable();
        let name = inner.next().unwrap().as_str().to_string();

        let generics = if matches!(inner.peek().map(|p| p.as_rule()), Some(Rule::generic_params)) {
            self.lower_generic_params(inner.next().unwrap())
        } else {
            Vec::new()
        };

        let params = if matches!(inner.peek().map(|p| p.as_rule()), Some(Rule::param_list)) {
            self.lower_param_list(inner.next().unwrap())
        } else {
            Vec::new()
        };

        let ret = if matches!(inner.peek().map(|p| p.as_rule()), Some(Rule::type_)) {
            Some(self.lower_type(inner.next().unwrap()))
        } else {
            None
        };

        let body = self.lower_block(inner.next().unwrap());
        FnDecl { name, generics, params, ret, body }
    }

    fn lower_generic_params(&mut self, pair: Pair<Rule>) -> Vec<GenericParam> {
        pair.into_inner().map(|p| self.lower_generic_param(p)).collect()
    }

    fn lower_generic_param(&mut self, pair: Pair<Rule>) -> GenericParam {
        let inner: Vec<_> = pair.into_inner().collect();
        // `"const" ~ ident ~ ":" ~ type_` yields [ident, type_]; a bare/bounded
        // type param yields [ident] or [ident, bound_list] — disambiguated by
        // the second pair's rule, since "const" itself is a bare literal token
        // (no pair of its own).
        match inner.len() {
            2 if inner[1].as_rule() == Rule::type_ => {
                let name = inner[0].as_str().to_string();
                let ty = self.lower_type(inner[1].clone());
                GenericParam::Const { name, ty }
            }
            2 => {
                let name = inner[0].as_str().to_string();
                let bounds = self.lower_bound_list(inner[1].clone());
                GenericParam::Type { name, bounds }
            }
            1 => GenericParam::Type { name: inner[0].as_str().to_string(), bounds: Vec::new() },
            n => unreachable!("generic_param: unexpected arity {n}"),
        }
    }

    fn lower_bound_list(&mut self, pair: Pair<Rule>) -> Vec<String> {
        pair.into_inner().map(|p| p.as_str().to_string()).collect()
    }

    fn lower_param_list(&mut self, pair: Pair<Rule>) -> Vec<Param> {
        pair.into_inner().map(|p| self.lower_param(p)).collect()
    }

    fn lower_param(&mut self, pair: Pair<Rule>) -> Param {
        let mut inner = pair.into_inner();
        let name = inner.next().unwrap().as_str().to_string();
        let ty = inner.next().map(|p| self.lower_type(p));
        Param { name, ty }
    }

    // ---------------------------------------------------------------- struct

    fn lower_struct_decl(&mut self, pair: Pair<Rule>) -> StructDecl {
        let mut inner = pair.into_inner().peekable();
        let name = inner.next().unwrap().as_str().to_string();
        let generics = if matches!(inner.peek().map(|p| p.as_rule()), Some(Rule::generic_params)) {
            self.lower_generic_params(inner.next().unwrap())
        } else {
            Vec::new()
        };
        let fields = match inner.next() {
            Some(p) => self.lower_field_list(p),
            None => Vec::new(),
        };
        StructDecl { name, generics, fields }
    }

    fn lower_field_list(&mut self, pair: Pair<Rule>) -> Vec<Field> {
        pair.into_inner().map(|p| self.lower_field(p)).collect()
    }

    fn lower_field(&mut self, pair: Pair<Rule>) -> Field {
        let mut inner = pair.into_inner();
        let name = inner.next().unwrap().as_str().to_string();
        let ty = self.lower_type(inner.next().unwrap());
        Field { name, ty }
    }

    // ---------------------------------------------------------------- algebra

    fn lower_algebra_decl(&mut self, pair: Pair<Rule>) -> AlgebraDecl {
        let mut inner = pair.into_inner().peekable();
        let name = inner.next().unwrap().as_str().to_string();
        let generics = if matches!(inner.peek().map(|p| p.as_rule()), Some(Rule::generic_params)) {
            self.lower_generic_params(inner.next().unwrap())
        } else {
            Vec::new()
        };
        let bounds = if matches!(inner.peek().map(|p| p.as_rule()), Some(Rule::bound_list)) {
            self.lower_bound_list(inner.next().unwrap())
        } else {
            Vec::new()
        };
        let items = inner.map(|p| self.lower_algebra_item(p)).collect();
        AlgebraDecl { name, generics, bounds, items }
    }

    fn lower_algebra_item(&mut self, pair: Pair<Rule>) -> AlgebraItem {
        let span = self.span_of(&pair);
        let inner = pair.into_inner().next().unwrap();
        let kind = match inner.as_rule() {
            Rule::fn_sig => AlgebraItemKind::FnSig(self.lower_fn_sig(inner)),
            Rule::axiom_decl => AlgebraItemKind::Axiom(self.lower_axiom_decl(inner)),
            r => unreachable!("algebra_item: unexpected rule {r:?}"),
        };
        self.wrap(span, kind)
    }

    fn lower_fn_sig(&mut self, pair: Pair<Rule>) -> FnSig {
        let mut inner = pair.into_inner().peekable();
        let name = inner.next().unwrap().as_str().to_string();
        let params = if matches!(inner.peek().map(|p| p.as_rule()), Some(Rule::param_list)) {
            self.lower_param_list(inner.next().unwrap())
        } else {
            Vec::new()
        };
        let ret = inner.next().map(|p| self.lower_type(p));
        FnSig { name, params, ret }
    }

    fn lower_axiom_decl(&mut self, pair: Pair<Rule>) -> AxiomDecl {
        let mut inner = pair.into_inner().peekable();
        let name = inner.next().unwrap().as_str().to_string();
        let params = if matches!(inner.peek().map(|p| p.as_rule()), Some(Rule::param_list)) {
            self.lower_param_list(inner.next().unwrap())
        } else {
            Vec::new()
        };
        let body = self.lower_expr(inner.next().unwrap());
        AxiomDecl { name, params, body }
    }

    // ---------------------------------------------------------------- impl

    fn lower_algebra_impl(&mut self, pair: Pair<Rule>) -> ImplDecl {
        let mut inner = pair.into_inner().peekable();

        let generics = if matches!(inner.peek().map(|p| p.as_rule()), Some(Rule::generic_params)) {
            self.lower_generic_params(inner.next().unwrap())
        } else {
            Vec::new()
        };

        let algebra = inner.next().unwrap().as_str().to_string();
        let target = self.lower_type(inner.next().unwrap());
        // Zero or more additional `type_` pairs precede the `fn_decl*` tail
        // — same "peel off while the rule keeps matching" pattern used
        // everywhere else an optional repetition is flattened alongside
        // fixed leading fields in this same pair (see `generic_params?`
        // just above).
        let mut extra_targets = Vec::new();
        while matches!(inner.peek().map(|p| p.as_rule()), Some(Rule::type_)) {
            extra_targets.push(self.lower_type(inner.next().unwrap()));
        }
        let fns = inner.map(|p| self.lower_fn_decl(p)).collect();
        ImplDecl { algebra, generics, target, extra_targets, fns }
    }

    fn lower_inherent_impl(&mut self, pair: Pair<Rule>) -> InherentImplDecl {
        let mut inner = pair.into_inner().peekable();

        let generics = if matches!(inner.peek().map(|p| p.as_rule()), Some(Rule::generic_params)) {
            self.lower_generic_params(inner.next().unwrap())
        } else {
            Vec::new()
        };

        let target = self.lower_type(inner.next().unwrap());
        let fns = inner.map(|p| self.lower_fn_decl(p)).collect();
        InherentImplDecl { generics, target, fns }
    }

    // ---------------------------------------------------------------- types

    fn lower_type(&mut self, pair: Pair<Rule>) -> Type {
        let span = self.span_of(&pair);
        let mut inner = pair.into_inner();
        let first = inner.next().unwrap();
        match first.as_rule() {
            Rule::array_type => self.lower_array_type(first),
            Rule::fn_type => self.lower_fn_type(first),
            // `path`'s optional generic-arg list is flattened as trailing
            // siblings of `path` within this same `type_` pair, not nested
            // under a separate sub-rule — collect whatever remains.
            Rule::path => {
                let path = self.lower_path(first);
                let args = inner.map(|p| self.lower_generic_arg(p)).collect();
                self.wrap(span, TypeKind::Path(path, args))
            }
            r => unreachable!("type_: unexpected rule {r:?}"),
        }
    }

    /// `(T1, T2) -> R` — every inner `type_` pair is a parameter *except*
    /// the last, which is the (always-present, see `grammar.pest`'s own
    /// comment) return type.
    fn lower_fn_type(&mut self, pair: Pair<Rule>) -> Type {
        let span = self.span_of(&pair);
        let mut types: Vec<Type> = pair.into_inner().map(|p| self.lower_type(p)).collect();
        let ret = Box::new(types.pop().expect("fn_type always has at least a return type"));
        self.wrap(span, TypeKind::Fn(types, ret))
    }

    fn lower_generic_arg(&mut self, pair: Pair<Rule>) -> GenericArg {
        let inner = pair.into_inner().next().unwrap();
        match inner.as_rule() {
            Rule::type_ => GenericArg::Type(self.lower_type(inner)),
            Rule::numeric_lit => GenericArg::Const(self.lower_numeric_lit(inner)),
            Rule::bool_lit => GenericArg::Const(self.lower_bool_lit(inner)),
            r => unreachable!("generic_arg: unexpected rule {r:?}"),
        }
    }

    /// `[T; d0, d1, ...]` folds (right to left) into nested `[[T; d1]; d0]` —
    /// see `grammar.md`, the Fortran-style multi-dim sugar note.
    fn lower_array_type(&mut self, pair: Pair<Rule>) -> Type {
        let span = self.span_of(&pair);
        let mut inner: Vec<_> = pair.into_inner().collect();
        let dims: Vec<Pair<Rule>> = inner.split_off(1);
        let elem = self.lower_type(inner.remove(0));
        let dims: Vec<Expr> = dims.into_iter().map(|p| self.lower_expr(p)).collect();
        let mut acc = elem;
        for dim in dims.into_iter().rev() {
            acc = self.wrap(span, TypeKind::Array(Box::new(acc), Box::new(dim)));
        }
        acc
    }

    // ---------------------------------------------------------------- blocks / statements

    fn lower_block(&mut self, pair: Pair<Rule>) -> Block {
        let mut stmts = Vec::new();
        let mut tail = None;
        for inner in pair.into_inner() {
            match inner.as_rule() {
                Rule::stmt => stmts.push(self.lower_stmt(inner)),
                Rule::expr => tail = Some(Box::new(self.lower_expr(inner))),
                r => unreachable!("block: unexpected rule {r:?}"),
            }
        }
        Block { stmts, tail }
    }

    fn lower_stmt(&mut self, pair: Pair<Rule>) -> Stmt {
        let span = self.span_of(&pair);
        let inner = pair.into_inner().next().unwrap();
        let kind = match inner.as_rule() {
            Rule::let_stmt => self.lower_let_stmt(inner),
            Rule::assign_stmt => self.lower_assign_stmt(inner),
            Rule::expr_stmt => {
                let e = self.lower_expr(inner.into_inner().next().unwrap());
                StmtKind::Expr(e)
            }
            r => unreachable!("stmt: unexpected rule {r:?}"),
        };
        self.wrap(span, kind)
    }

    fn lower_let_stmt(&mut self, pair: Pair<Rule>) -> StmtKind {
        let mut mutable = false;
        let mut name = None;
        let mut ty = None;
        let mut value = None;
        for p in pair.into_inner() {
            match p.as_rule() {
                Rule::mut_kw => mutable = true,
                Rule::ident if name.is_none() => name = Some(p.as_str().to_string()),
                Rule::type_ => ty = Some(self.lower_type(p)),
                Rule::expr => value = Some(self.lower_expr(p)),
                _ => {}
            }
        }
        StmtKind::Let { mutable, name: name.unwrap(), ty, value: value.unwrap() }
    }

    fn lower_assign_stmt(&mut self, pair: Pair<Rule>) -> StmtKind {
        let mut inner = pair.into_inner();
        let name = inner.next().unwrap().as_str().to_string();
        let value = self.lower_expr(inner.next().unwrap());
        StmtKind::Assign { name, value }
    }

    // ---------------------------------------------------------------- expressions

    fn lower_expr(&mut self, pair: Pair<Rule>) -> Expr {
        debug_assert_eq!(pair.as_rule(), Rule::expr);
        self.lower_implication(pair.into_inner().next().unwrap())
    }

    fn lower_implication(&mut self, pair: Pair<Rule>) -> Expr {
        self.fold_binary(pair, Self::lower_logical, |op| match op {
            "implies" => "implies",
            _ => unreachable!(),
        })
    }

    fn lower_logical(&mut self, pair: Pair<Rule>) -> Expr {
        self.fold_binary(pair, Self::lower_comparison, |op| match op {
            "and" => "and",
            "or" => "or",
            "xor" => "xor",
            _ => unreachable!(),
        })
    }

    fn lower_comparison(&mut self, pair: Pair<Rule>) -> Expr {
        self.fold_binary(pair, Self::lower_additive, |op| match op {
            "<=" => "le",
            ">=" => "ge",
            "==" => "eq",
            "!=" => "neq",
            "<" => "lt",
            ">" => "gt",
            _ => unreachable!(),
        })
    }

    fn lower_additive(&mut self, pair: Pair<Rule>) -> Expr {
        self.fold_binary(pair, Self::lower_multiplicative, |op| match op {
            "+" => "add",
            "-" => "sub",
            _ => unreachable!(),
        })
    }

    fn lower_multiplicative(&mut self, pair: Pair<Rule>) -> Expr {
        self.fold_binary(pair, Self::lower_unary, |op| match op {
            "*" => "mul",
            "/" => "div",
            _ => unreachable!(),
        })
    }

    /// Shared left-associative fold for `child ~ (op ~ child)*` — every binary
    /// precedence level has this exact shape (see `grammar.pest`). Desugars
    /// directly to `Call(op_name, [lhs, rhs])`, per `grammar.md`'s "operators
    /// are sugar for named algebra functions" — algebra-qualifying the call
    /// target (`Ring::add` vs. bare `add`) is a later, type-checking concern.
    fn fold_binary(
        &mut self,
        pair: Pair<Rule>,
        lower_operand: fn(&mut Self, Pair<Rule>) -> Expr,
        op_name: fn(&str) -> &'static str,
    ) -> Expr {
        let mut inner = pair.into_inner();
        let mut acc = lower_operand(self, inner.next().unwrap());
        while let Some(op_pair) = inner.next() {
            let rhs_pair = inner.next().unwrap();
            let rhs = lower_operand(self, rhs_pair);
            let name = op_name(op_pair.as_str());
            let span = self.join(acc.span, rhs.span);
            acc = self.wrap(span, ExprKind::Call(Path::single(name), Vec::new(), vec![acc, rhs]));
        }
        acc
    }

    fn lower_unary(&mut self, pair: Pair<Rule>) -> Expr {
        let span = self.span_of(&pair);
        let mut inner = pair.into_inner();
        let first = inner.next().unwrap();
        match first.as_rule() {
            Rule::unary => {
                // "-" ~ unary — the "-" itself is a bare token, not a pair.
                let operand = self.lower_unary(first);
                self.wrap(span, ExprKind::Call(Path::single("neg"), Vec::new(), vec![operand]))
            }
            Rule::postfix => self.lower_postfix(first),
            r => unreachable!("unary: unexpected rule {r:?}"),
        }
    }

    fn lower_postfix(&mut self, pair: Pair<Rule>) -> Expr {
        let mut inner = pair.into_inner();
        let mut acc = self.lower_primary(inner.next().unwrap());
        for op in inner {
            acc = self.lower_postfix_op(acc, op);
        }
        acc
    }

    fn lower_postfix_op(&mut self, base: Expr, pair: Pair<Rule>) -> Expr {
        let op_span = self.span_of(&pair);
        let span = self.join(base.span, op_span);
        let mut inner = pair.into_inner();
        let first = inner.next().unwrap();
        match first.as_rule() {
            Rule::ident => {
                let name = first.as_str().to_string();
                // `call_args`'s mere presence (even with zero arguments, `.f()`)
                // distinguishes a method call from plain field access (`.x`) —
                // an absent vs. an empty `arg_list` are otherwise indistinguishable.
                match inner.next() {
                    Some(call_args) => {
                        let args = call_args
                            .into_inner()
                            .next()
                            .map(|al| self.lower_arg_list(al))
                            .unwrap_or_default();
                        self.wrap(span, ExprKind::MethodCall(Box::new(base), name, args))
                    }
                    None => self.wrap(span, ExprKind::FieldAccess(Box::new(base), name)),
                }
            }
            Rule::expr => {
                // one or more comma-separated indices — fold left, `a[i,j]` => `a[i][j]`
                let mut indices = vec![self.lower_expr(first)];
                indices.extend(inner.map(|p| self.lower_expr(p)));
                let mut acc = base;
                for idx in indices {
                    let s = self.join(acc.span, idx.span);
                    acc = self.wrap(s, ExprKind::Index(Box::new(acc), Box::new(idx)));
                }
                acc
            }
            r => unreachable!("postfix_op: unexpected rule {r:?}"),
        }
    }

    fn lower_arg_list(&mut self, pair: Pair<Rule>) -> Vec<Expr> {
        pair.into_inner().map(|p| self.lower_expr(p)).collect()
    }

    fn lower_primary(&mut self, pair: Pair<Rule>) -> Expr {
        let span = self.span_of(&pair);
        let inner = pair.into_inner().next().unwrap();
        match inner.as_rule() {
            Rule::if_expr => self.lower_if_expr(inner),
            Rule::while_expr => self.lower_while_expr(inner),
            Rule::for_expr => self.lower_for_expr(inner),
            Rule::array_lit => self.lower_array_lit(inner),
            Rule::lambda_expr => self.lower_lambda_expr(inner),
            Rule::struct_lit => self.lower_struct_lit(inner),
            Rule::call_expr => self.lower_call_expr(inner),
            Rule::literal => self.lower_literal(inner),
            Rule::path => {
                let path = self.lower_path(inner);
                self.wrap(span, ExprKind::Path(path))
            }
            Rule::expr => self.lower_expr(inner), // parenthesized — parens don't survive as a node
            Rule::block => {
                let b = self.lower_block(inner);
                self.wrap(span, ExprKind::Block(b))
            }
            r => unreachable!("primary: unexpected rule {r:?}"),
        }
    }

    fn lower_lambda_expr(&mut self, pair: Pair<Rule>) -> Expr {
        let span = self.span_of(&pair);
        let mut inner = pair.into_inner().peekable();

        let params = if matches!(inner.peek().map(|p| p.as_rule()), Some(Rule::param_list)) {
            self.lower_param_list(inner.next().unwrap())
        } else {
            Vec::new()
        };

        let ret = if matches!(inner.peek().map(|p| p.as_rule()), Some(Rule::type_)) {
            Some(self.lower_type(inner.next().unwrap()))
        } else {
            None
        };

        let body = self.lower_block(inner.next().unwrap());
        self.wrap(span, ExprKind::Lambda { params, ret, body })
    }

    fn lower_call_expr(&mut self, pair: Pair<Rule>) -> Expr {
        let span = self.span_of(&pair);
        let mut inner = pair.into_inner().peekable();
        let path = self.lower_path(inner.next().unwrap());
        let generics = self.lower_optional_turbofish(&mut inner);
        let args = inner.next().map(|p| self.lower_arg_list(p)).unwrap_or_default();
        self.wrap(span, ExprKind::Call(path, generics, args))
    }

    fn lower_struct_lit(&mut self, pair: Pair<Rule>) -> Expr {
        let span = self.span_of(&pair);
        let mut inner = pair.into_inner().peekable();
        let path = self.lower_path(inner.next().unwrap());
        let generics = self.lower_optional_turbofish(&mut inner);
        let fields = inner.next().map(|p| self.lower_field_init_list(p)).unwrap_or_default();
        self.wrap(span, ExprKind::StructLit(path, generics, fields))
    }

    /// Consumes a leading `Rule::turbofish` pair off `inner` if present,
    /// returning its generic arguments — shared by `lower_call_expr`/
    /// `lower_struct_lit`, the two constructs that can carry one.
    fn lower_optional_turbofish(&mut self, inner: &mut std::iter::Peekable<Pairs<Rule>>) -> Vec<GenericArg> {
        if matches!(inner.peek().map(|p| p.as_rule()), Some(Rule::turbofish)) {
            inner.next().unwrap().into_inner().map(|p| self.lower_generic_arg(p)).collect()
        } else {
            Vec::new()
        }
    }

    fn lower_field_init_list(&mut self, pair: Pair<Rule>) -> Vec<(String, Expr)> {
        pair.into_inner()
            .map(|field_init| {
                let mut inner = field_init.into_inner();
                let name = inner.next().unwrap().as_str().to_string();
                let value = self.lower_expr(inner.next().unwrap());
                (name, value)
            })
            .collect()
    }

    fn lower_if_expr(&mut self, pair: Pair<Rule>) -> Expr {
        let span = self.span_of(&pair);
        let mut inner = pair.into_inner();
        let cond = Box::new(self.lower_expr(inner.next().unwrap()));
        let then_branch = self.lower_block(inner.next().unwrap());
        let else_branch = inner.next().map(|p| {
            Box::new(match p.as_rule() {
                Rule::if_expr => ElseBranch::If(self.lower_if_expr(p)),
                Rule::block => ElseBranch::Block(self.lower_block(p)),
                r => unreachable!("if_expr else: unexpected rule {r:?}"),
            })
        });
        self.wrap(span, ExprKind::If { cond, then_branch, else_branch })
    }

    fn lower_while_expr(&mut self, pair: Pair<Rule>) -> Expr {
        let span = self.span_of(&pair);
        let mut inner = pair.into_inner();
        let cond = Box::new(self.lower_expr(inner.next().unwrap()));
        let body = self.lower_block(inner.next().unwrap());
        self.wrap(span, ExprKind::While { cond, body })
    }

    fn lower_for_expr(&mut self, pair: Pair<Rule>) -> Expr {
        let span = self.span_of(&pair);
        let mut inner = pair.into_inner();
        let var = inner.next().unwrap().as_str().to_string();
        let start = Box::new(self.lower_additive(inner.next().unwrap()));
        let end = Box::new(self.lower_additive(inner.next().unwrap()));
        let body = self.lower_block(inner.next().unwrap());
        self.wrap(span, ExprKind::For { var, start, end, body })
    }

    fn lower_array_lit(&mut self, pair: Pair<Rule>) -> Expr {
        let span = self.span_of(&pair);
        let Some(body) = pair.into_inner().next() else {
            return self.wrap(span, ExprKind::ArrayLit(Vec::new()));
        };
        let elems = match body.as_rule() {
            // `[value; N]` -- re-lowers the *same* parsed `value` pair `N`
            // times (cheap: `Pair` is a reference into the token stream, not
            // an owned deep copy) rather than lowering once and cloning the
            // resulting `Expr`, so each copy gets its own distinct `NodeId`
            // — every other node in this AST is unique per occurrence (see
            // `ast.rs`'s own doc comment on `NodeId`), and `node_types`
            // (keyed by `NodeId`) would silently collapse all `N` copies
            // into one entry otherwise.
            Rule::array_repeat => {
                let mut inner = body.into_inner();
                let value = inner.next().unwrap();
                let count_text = inner.next().unwrap().as_str();
                // `numeric_lit`'s own text, possibly `:suffix`-terminated —
                // a repeat count is never suffixed in practice, but strip it
                // defensively rather than let `.parse` reject it outright.
                let count: usize = count_text.split(':').next().unwrap().parse().unwrap_or_else(|e| {
                    panic!("array-repeat count {count_text:?} is not a valid array size: {e}")
                });
                (0..count).map(|_| self.lower_expr(value.clone())).collect()
            }
            Rule::array_list => body.into_inner().map(|p| self.lower_expr(p)).collect(),
            other => unreachable!("array_lit's own body must be array_repeat or array_list, got {other:?}"),
        };
        self.wrap(span, ExprKind::ArrayLit(elems))
    }

    fn lower_literal(&mut self, pair: Pair<Rule>) -> Expr {
        let inner = pair.into_inner().next().unwrap();
        match inner.as_rule() {
            Rule::imaginary_lit => self.lower_imaginary_lit(inner),
            Rule::numeric_lit => self.lower_numeric_lit(inner),
            Rule::bool_lit => self.lower_bool_lit(inner),
            r => unreachable!("literal: unexpected rule {r:?}"),
        }
    }

    fn lower_bool_lit(&mut self, pair: Pair<Rule>) -> Expr {
        let span = self.span_of(&pair);
        let b = pair.as_str() == "true";
        self.wrap(span, ExprKind::BoolLit(b))
    }

    /// `numeric_lit`/`imaginary_lit` are atomic (`@{}`) in the grammar, so pest
    /// gives us raw text with no `number_body`/`type_suffix` sub-pairs — split
    /// on `:` ourselves. Final numeric type (default/override/inferred) is a
    /// type-inference concern, not resolved here (see `grammar.md`).
    fn lower_numeric_lit(&mut self, pair: Pair<Rule>) -> Expr {
        let span = self.span_of(&pair);
        let (text, suffix) = split_type_suffix(pair.as_str());
        self.wrap(span, ExprKind::NumberLit { text: text.to_string(), suffix })
    }

    fn lower_imaginary_lit(&mut self, pair: Pair<Rule>) -> Expr {
        let span = self.span_of(&pair);
        let text = pair.as_str();
        let number_part = &text[..text.len() - 1]; // strip trailing "i"
        self.wrap(span, ExprKind::ImaginaryLit { text: number_part.to_string(), suffix: None })
    }
}

fn split_type_suffix(text: &str) -> (&str, Option<String>) {
    match text.find(':') {
        Some(idx) => (&text[..idx], Some(text[idx + 1..].to_string())),
        None => (text, None),
    }
}
