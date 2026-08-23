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
        Lowerer {
            file,
            ids: NodeIdGen::default(),
        }
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
        Span {
            file: self.file,
            start: s.start(),
            end: s.end(),
        }
    }

    fn join(&self, a: Span, b: Span) -> Span {
        Span {
            file: self.file,
            start: a.start,
            end: b.end,
        }
    }

    fn wrap<T>(&mut self, span: Span, kind: T) -> Node<T> {
        Node {
            id: self.ids.next(),
            span,
            kind,
        }
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
                    Rule::inherent_impl => {
                        ItemKind::InherentImpl(self.lower_inherent_impl(variant))
                    }
                    r => unreachable!("impl_decl: unexpected rule {r:?}"),
                }
            }
            Rule::fn_decl => ItemKind::Fn(self.lower_fn_decl(inner)),
            Rule::derive_decl => ItemKind::Fn(self.lower_derive_decl(inner)),
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

        let mut attrs = Vec::new();
        while matches!(inner.peek().map(|p| p.as_rule()), Some(Rule::attribute)) {
            attrs.push(self.lower_attribute(inner.next().unwrap()));
        }

        let is_extern = matches!(inner.peek().map(|p| p.as_rule()), Some(Rule::extern_kw));
        let extern_symbol = if is_extern {
            // `extern_kw`'s own optional `(symbol)` -- an `ident` pair
            // inside it if present, nothing if it's a bare `extern`.
            inner
                .next()
                .unwrap()
                .into_inner()
                .next()
                .map(|p| p.as_str().to_string())
        } else {
            None
        };

        let is_export = matches!(inner.peek().map(|p| p.as_rule()), Some(Rule::export_kw));
        let export_symbol = if is_export {
            inner
                .next()
                .unwrap()
                .into_inner()
                .next()
                .map(|p| p.as_str().to_string())
        } else {
            None
        };

        let name = inner.next().unwrap().as_str().to_string();

        let generics = if matches!(
            inner.peek().map(|p| p.as_rule()),
            Some(Rule::generic_params)
        ) {
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

        // `(block | ";")` -- a bare `;` is a literal token, producing no
        // pair of its own, so `None` here is exactly "no `block` pair was
        // left to consume", not a parse failure.
        let body = inner.next().map(|p| self.lower_block(p));
        FnDecl {
            name,
            attrs,
            is_extern,
            extern_symbol,
            is_export,
            export_symbol,
            generics,
            params,
            ret,
            body,
            derivative_of: None,
        }
    }

    /// `fprime = derive(f);` (`grammar.pest`'s own `derive_decl`) — lowers
    /// straight to an ordinary `FnDecl`, deliberately reusing the same
    /// `ItemKind::Fn` shape every other top-level `fn` uses (see `ast.rs`'s
    /// own `FnDecl::derivative_of` doc comment for why: every existing pass
    /// that already handles `ItemKind::Fn` uniformly — registry, print.rs,
    /// dump.rs, driver.rs's own merge logic — needs no change at all).
    /// `params`/`ret` are left empty here on purpose: this lowering step has
    /// no way to know `f`'s own signature (possibly declared in a different
    /// file entirely) — filled in by a dedicated later pass, once every
    /// crate's own items are merged (`driver.rs::compile`).
    fn lower_derive_decl(&mut self, pair: Pair<Rule>) -> FnDecl {
        let mut inner = pair.into_inner();
        let name = inner.next().unwrap().as_str().to_string();
        let of = inner.next().unwrap().as_str().to_string();
        FnDecl {
            name,
            attrs: Vec::new(),
            is_extern: false,
            extern_symbol: None,
            is_export: false,
            export_symbol: None,
            generics: Vec::new(),
            params: Vec::new(),
            ret: None,
            body: None,
            derivative_of: Some(of),
        }
    }

    fn lower_attribute(&mut self, pair: Pair<Rule>) -> Attribute {
        let span = self.span_of(&pair);
        let mut inner = pair.into_inner();
        let name = inner.next().unwrap().as_str().to_string();
        let args = inner.map(|p| p.as_str().to_string()).collect();
        Attribute { name, args, span }
    }

    fn lower_generic_params(&mut self, pair: Pair<Rule>) -> Vec<GenericParam> {
        pair.into_inner()
            .map(|p| self.lower_generic_param(p))
            .collect()
    }

    fn lower_generic_param(&mut self, pair: Pair<Rule>) -> GenericParam {
        let mut inner: Vec<_> = pair.into_inner().collect();
        // A `pack_marker` pair, right after the leading `ident`
        // (`Args...`/`const Dims...: i32`) -- stripped out first, its own
        // mere presence recorded as `variadic`, so the rest of this
        // function's own dispatch-by-shape logic stays exactly as it was
        // before packs existed (`doc/backlog.md`'s own "Variadic generics"
        // item). Placed right after the name, not at the end -- see
        // `grammar.pest`'s own `generic_param` doc comment for the real
        // ambiguity that ordering avoids.
        let variadic = matches!(inner.get(1).map(Pair::as_rule), Some(Rule::pack_marker));
        if variadic {
            inner.remove(1);
        }
        // `"const" ~ ident ~ ":" ~ type_` yields [ident, type_]; a bare/bounded
        // type param yields [ident] or [ident, bound_list] — disambiguated by
        // the second pair's rule, since "const" itself is a bare literal token
        // (no pair of its own).
        match inner.len() {
            2 if inner[1].as_rule() == Rule::type_ => {
                let name = inner[0].as_str().to_string();
                let ty = self.lower_type(inner[1].clone());
                GenericParam::Const { name, ty, variadic }
            }
            2 => {
                let name = inner[0].as_str().to_string();
                let bounds = self.lower_bound_list(inner[1].clone());
                GenericParam::Type {
                    name,
                    bounds,
                    variadic,
                }
            }
            1 => GenericParam::Type {
                name: inner[0].as_str().to_string(),
                bounds: Vec::new(),
                variadic,
            },
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
        let mut mutable = false;
        let mut name = None;
        let mut ty = None;
        for p in pair.into_inner() {
            match p.as_rule() {
                Rule::mut_kw => mutable = true,
                Rule::ident if name.is_none() => name = Some(p.as_str().to_string()),
                Rule::type_ => ty = Some(self.lower_type(p)),
                _ => {}
            }
        }
        Param {
            name: name.unwrap(),
            ty,
            mutable,
        }
    }

    // ---------------------------------------------------------------- struct

    fn lower_struct_decl(&mut self, pair: Pair<Rule>) -> StructDecl {
        let mut inner = pair.into_inner().peekable();
        let name = inner.next().unwrap().as_str().to_string();
        let generics = if matches!(
            inner.peek().map(|p| p.as_rule()),
            Some(Rule::generic_params)
        ) {
            self.lower_generic_params(inner.next().unwrap())
        } else {
            Vec::new()
        };
        let fields = match inner.next() {
            Some(p) => self.lower_field_list(p),
            None => Vec::new(),
        };
        StructDecl {
            name,
            generics,
            fields,
        }
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
        let generics = if matches!(
            inner.peek().map(|p| p.as_rule()),
            Some(Rule::generic_params)
        ) {
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
        AlgebraDecl {
            name,
            generics,
            bounds,
            items,
        }
    }

    fn lower_algebra_item(&mut self, pair: Pair<Rule>) -> AlgebraItem {
        let span = self.span_of(&pair);
        let inner = pair.into_inner().next().unwrap();
        let kind = match inner.as_rule() {
            Rule::fn_sig => AlgebraItemKind::FnSig(self.lower_fn_sig(inner)),
            Rule::axiom_decl => AlgebraItemKind::Axiom(self.lower_axiom_decl(inner)),
            Rule::derivative_rule_decl => {
                AlgebraItemKind::DerivativeRule(self.lower_derivative_rule_decl(inner))
            }
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

    fn lower_derivative_rule_decl(&mut self, pair: Pair<Rule>) -> DerivativeRuleDecl {
        let mut inner = pair.into_inner().peekable();
        let method = inner.next().unwrap().as_str().to_string();
        let params = if matches!(inner.peek().map(|p| p.as_rule()), Some(Rule::param_list)) {
            self.lower_param_list(inner.next().unwrap())
        } else {
            Vec::new()
        };
        let body = self.lower_expr(inner.next().unwrap());
        DerivativeRuleDecl {
            method,
            params,
            body,
        }
    }

    // ---------------------------------------------------------------- impl

    fn lower_algebra_impl(&mut self, pair: Pair<Rule>) -> ImplDecl {
        let mut inner = pair.into_inner().peekable();

        let mut attrs = Vec::new();
        while matches!(inner.peek().map(|p| p.as_rule()), Some(Rule::attribute)) {
            attrs.push(self.lower_attribute(inner.next().unwrap()));
        }

        let generics = if matches!(
            inner.peek().map(|p| p.as_rule()),
            Some(Rule::generic_params)
        ) {
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
        ImplDecl {
            attrs,
            algebra,
            generics,
            target,
            extra_targets,
            fns,
        }
    }

    fn lower_inherent_impl(&mut self, pair: Pair<Rule>) -> InherentImplDecl {
        let mut inner = pair.into_inner().peekable();

        let generics = if matches!(
            inner.peek().map(|p| p.as_rule()),
            Some(Rule::generic_params)
        ) {
            self.lower_generic_params(inner.next().unwrap())
        } else {
            Vec::new()
        };

        let target = self.lower_type(inner.next().unwrap());
        let fns = inner.map(|p| self.lower_fn_decl(p)).collect();
        InherentImplDecl {
            generics,
            target,
            fns,
        }
    }

    // ---------------------------------------------------------------- types

    fn lower_type(&mut self, pair: Pair<Rule>) -> Type {
        let span = self.span_of(&pair);
        let mut inner = pair.into_inner();
        let first = inner.next().unwrap();
        match first.as_rule() {
            Rule::array_type => self.lower_array_type(first),
            Rule::fn_type => self.lower_fn_type(first),
            Rule::tuple_type => self.lower_tuple_type(first),
            Rule::pack_ref => {
                let name = first.into_inner().next().unwrap().as_str().to_string();
                self.wrap(span, TypeKind::PackRef(name))
            }
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
        let ret = Box::new(
            types
                .pop()
                .expect("fn_type always has at least a return type"),
        );
        self.wrap(span, TypeKind::Fn(types, ret))
    }

    /// `(T1, T2)` desugars straight to the ordinary generic-struct-type
    /// syntax `__Tuple2<T1, T2>` already produces — see `ast::tuple_struct_
    /// name`'s own doc comment for why this needs no new `TypeKind` variant.
    fn lower_tuple_type(&mut self, pair: Pair<Rule>) -> Type {
        let span = self.span_of(&pair);
        let elems: Vec<Type> = pair.into_inner().map(|p| self.lower_type(p)).collect();
        let name = tuple_struct_name(elems.len());
        let args = elems.into_iter().map(GenericArg::Type).collect();
        self.wrap(span, TypeKind::Path(Path::single(name), args))
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
        let dims: Vec<Expr> = dims.into_iter().map(|p| self.lower_array_dim(p)).collect();
        let mut acc = elem;
        for dim in dims.into_iter().rev() {
            acc = self.wrap(span, TypeKind::Array(Box::new(acc), Box::new(dim)));
        }
        acc
    }

    /// One `array_dim` — an ordinary `expr`, or (`doc/backlog.md`'s own
    /// "Variadic generics" item) a pack reference (`Dims...`): the pack's
    /// own bare name, discarding the ordinary `Path` shape `lower_expr`
    /// would otherwise build for it, since `ExprKind::PackRef` only ever
    /// needs the raw name — see `TypeKind::Array`'s own doc comment for how
    /// this size slot later expands once `Dims` resolves to a concrete list.
    fn lower_array_dim(&mut self, pair: Pair<Rule>) -> Expr {
        let span = self.span_of(&pair);
        let mut inner = pair.into_inner();
        let expr = self.lower_expr(inner.next().unwrap());
        if matches!(inner.next().map(|p| p.as_rule()), Some(Rule::pack_marker)) {
            let ExprKind::Path(path) = &expr.kind else {
                panic!(
                    "array_dim: a pack marker's own preceding expr must be a bare path, got {:?}",
                    expr.kind
                );
            };
            self.wrap(span, ExprKind::PackRef(path.segments.join("::")))
        } else {
            expr
        }
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
            Rule::break_stmt => {
                let value = inner.into_inner().next().map(|p| self.lower_expr(p));
                StmtKind::Break(value)
            }
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
        StmtKind::Let {
            mutable,
            name: name.unwrap(),
            ty,
            value: value.unwrap(),
        }
    }

    fn lower_assign_stmt(&mut self, pair: Pair<Rule>) -> StmtKind {
        let mut inner = pair.into_inner();
        let target = self.lower_assign_target(inner.next().unwrap());
        let value = self.lower_expr(inner.next().unwrap());
        StmtKind::Assign { target, value }
    }

    /// `assign_target = { ident ~ assign_suffix* }` — same shape `postfix`
    /// builds for an ordinary expression (base + folded `[...]`/`.field`
    /// chain), so the suffix folding reuses `lower_postfix_op` directly
    /// (`assign_suffix`'s two alternatives are exactly `postfix_op`'s first
    /// two, minus the call-args/method-call one).
    fn lower_assign_target(&mut self, pair: Pair<Rule>) -> Expr {
        let mut inner = pair.into_inner();
        let ident = inner.next().unwrap();
        let ident_span = self.span_of(&ident);
        let mut acc = self.wrap(
            ident_span,
            ExprKind::Path(Path {
                segments: vec![ident.as_str().to_string()],
            }),
        );
        for suffix in inner {
            acc = self.lower_postfix_op(acc, suffix);
        }
        acc
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
            acc = self.wrap(
                span,
                ExprKind::Call(Path::single(name), Vec::new(), vec![acc, rhs], Vec::new()),
            );
        }
        acc
    }

    fn lower_unary(&mut self, pair: Pair<Rule>) -> Expr {
        let span = self.span_of(&pair);
        let mut inner = pair.into_inner();
        let first = inner.next().unwrap();
        match first.as_rule() {
            Rule::unary_op => {
                let call_name = match first.as_str() {
                    "-" => "neg",
                    "not" => "not",
                    op => unreachable!("unary_op: unexpected operator {op:?}"),
                };
                let operand = self.lower_unary(inner.next().unwrap());
                self.wrap(
                    span,
                    ExprKind::Call(
                        Path::single(call_name),
                        Vec::new(),
                        vec![operand],
                        Vec::new(),
                    ),
                )
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
            // `t.0`/`t.1` — a tuple field access reuses the exact same
            // `FieldAccess`/`MethodCall` branching an ordinary `.ident` does,
            // just with the raw digit text as the field/method name (see
            // `grammar.pest`'s own `tuple_index` doc comment for why no
            // separate AST shape is needed at all).
            Rule::ident | Rule::tuple_index => {
                let name = first.as_str().to_string();
                // `call_args`'s mere presence (even with zero arguments, `.f()`)
                // distinguishes a method call from plain field access (`.x`) —
                // an absent vs. an empty `arg_list` are otherwise indistinguishable.
                match inner.next() {
                    Some(call_args) => {
                        // Named (`mlir_attr`) arguments are only meaningful on
                        // a reserved `mlir::...` *call*, never a method call —
                        // silently dropped here, same as `MethodCall` itself
                        // never getting a `mlir::`-recognizing path segment.
                        let (args, _mlir_attrs) = call_args
                            .into_inner()
                            .next()
                            .map(|al| self.lower_arg_list(al))
                            .unwrap_or_default();
                        // `.to()` — sugar for `convert(x)`, exactly the same
                        // shape `fold_binary` already uses to desugar `a + b`
                        // to `add(a, b)` — not a new `MethodCall`-dispatch
                        // path. `ExprKind::MethodCall` resolves purely via
                        // inherent-method lookup (`registry.inherent_method`,
                        // see `infer.rs`'s own handling), a wholly separate
                        // mechanism from algebra dispatch; `Convert<From,
                        // To>`'s own `To` is an output-only generic that only
                        // ever resolves through a real algebra-call dispatch,
                        // so `.to()` has to reach `infer_call` the same way
                        // any other operator does. Only the zero-argument
                        // spelling counts — `convert`'s own declared
                        // signature takes exactly one argument, the receiver
                        // itself.
                        if name == "to" && args.is_empty() {
                            self.wrap(
                                span,
                                ExprKind::Call(
                                    Path::single("convert"),
                                    Vec::new(),
                                    vec![base],
                                    Vec::new(),
                                ),
                            )
                        } else {
                            self.wrap(span, ExprKind::MethodCall(Box::new(base), name, args))
                        }
                    }
                    None => self.wrap(span, ExprKind::FieldAccess(Box::new(base), name)),
                }
            }
            Rule::expr => {
                // One or more comma-separated indices, collected directly
                // into one `Index` node — *not* folded into nested single-
                // index nodes (an earlier version of this did, `a[i,j]` =>
                // `a[i][j]`): a real array still gets the identical
                // semantics either way (`infer.rs` peels one dimension per
                // index), but a `#[mlir_type(...)]`-tagged struct needs the
                // whole group intact to dispatch one real multi-index
                // `Index<Container,Elem,K>` call — seeing `m[i,j]` as two
                // separate single-index steps has nothing sensible to
                // dispatch the first step to (see `ast.rs`'s own `Index`
                // doc comment).
                let indices: Vec<Expr> = std::iter::once(self.lower_expr(first))
                    .chain(inner.map(|p| self.lower_expr(p)))
                    .collect();
                self.wrap(span, ExprKind::Index(Box::new(base), indices))
            }
            // A direct call on whatever `base` is (`(fn(a,b){a+b})(1,2)`) --
            // `Call`'s own callee is a bare `Path`, so this can't become a
            // `Call` node directly; desugared instead to `{ let <synthetic>
            // = <base>; <synthetic>(<args>) }`, reusing the *existing* let-
            // bound-lambda pipeline wholesale (`infer.rs`'s `lambda_schemes`,
            // `monomorphize.rs`'s lambda worklist, `cps.rs`'s closure
            // conversion all key off this exact `StmtKind::Let` shape,
            // whether hand-written or synthesized here). Deliberately not
            // restricted to a literal `Lambda` base -- this file never
            // reports a semantic diagnostic of its own (see the module's own
            // doc comment, "checked later, not here"); a non-lambda base
            // (e.g. a redundant `(some_path)(args)`) simply falls through to
            // the *already-existing* "unresolved call" placeholder in
            // `infer.rs`, since `lambda_schemes` is only ever populated for a
            // `let` whose value is syntactically `ExprKind::Lambda`. `base.
            // id.0` (globally unique) makes the synthetic name collision-
            // free, and `<`/`#`/`>` can never appear in a real `ident`.
            Rule::call_args => {
                let (args, mlir_attrs) = first
                    .into_inner()
                    .next()
                    .map(|al| self.lower_arg_list(al))
                    .unwrap_or_default();
                let base_span = base.span;
                let synthetic = format!("<iife#{}>", base.id.0);
                let let_stmt = self.wrap(
                    base_span,
                    StmtKind::Let {
                        mutable: false,
                        name: synthetic.clone(),
                        ty: None,
                        value: base,
                    },
                );
                let call = self.wrap(
                    span,
                    ExprKind::Call(Path::single(synthetic), Vec::new(), args, mlir_attrs),
                );
                self.wrap(
                    span,
                    ExprKind::Block(Block {
                        stmts: vec![let_stmt],
                        tail: Some(Box::new(call)),
                    }),
                )
            }
            r => unreachable!("postfix_op: unexpected rule {r:?}"),
        }
    }

    /// Positional args and `mlir_attr` pairs (`predicate: "slt"`) may appear
    /// in any order/interleaving at the grammar level (each `call_arg` tries
    /// `mlir_attr` before `expr`, see that rule's own comment) but are split
    /// apart here — see `ast.rs`'s own `ExprKind::Call::mlir_attrs` doc
    /// comment for why they're not unified into one representation.
    fn lower_arg_list(&mut self, pair: Pair<Rule>) -> (Vec<Expr>, Vec<(String, String)>) {
        let mut args = Vec::new();
        let mut mlir_attrs = Vec::new();
        for call_arg in pair.into_inner() {
            let p = call_arg.into_inner().next().unwrap();
            match p.as_rule() {
                Rule::mlir_attr => {
                    let mut inner = p.into_inner();
                    let name = inner.next().unwrap().as_str().to_string();
                    // `string_lit` is atomic (`@{...}`), so `.as_str()` includes
                    // its own surrounding quotes -- stripped here, once, so
                    // every consumer downstream (`infer.rs`, `mlir_lower.rs`)
                    // sees just the raw attribute text.
                    let text = inner.next().unwrap().as_str();
                    mlir_attrs.push((name, text[1..text.len() - 1].to_string()));
                }
                _ => args.push(self.lower_expr(p)),
            }
        }
        (args, mlir_attrs)
    }

    fn lower_primary(&mut self, pair: Pair<Rule>) -> Expr {
        let span = self.span_of(&pair);
        let inner = pair.into_inner().next().unwrap();
        match inner.as_rule() {
            Rule::if_expr => self.lower_if_expr(inner),
            Rule::while_expr => self.lower_while_expr(inner),
            Rule::for_expr => self.lower_for_expr(inner),
            Rule::loop_expr => self.lower_loop_expr(inner),
            Rule::array_lit => self.lower_array_lit(inner),
            Rule::lambda_expr => self.lower_lambda_expr(inner),
            Rule::struct_lit => self.lower_struct_lit(inner),
            Rule::tuple_lit => self.lower_tuple_lit(inner),
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
        let (args, mlir_attrs) = inner
            .next()
            .map(|p| self.lower_arg_list(p))
            .unwrap_or_default();
        self.wrap(span, ExprKind::Call(path, generics, args, mlir_attrs))
    }

    fn lower_struct_lit(&mut self, pair: Pair<Rule>) -> Expr {
        let span = self.span_of(&pair);
        let mut inner = pair.into_inner().peekable();
        let path = self.lower_path(inner.next().unwrap());
        let generics = self.lower_optional_turbofish(&mut inner);
        let fields = inner
            .next()
            .map(|p| self.lower_field_init_list(p))
            .unwrap_or_default();
        self.wrap(span, ExprKind::StructLit(path, generics, fields))
    }

    /// `(a, b)` desugars straight to `__Tuple2(0: a, 1: b)` — the ordinary
    /// struct-construction shape `__Tuple2`'s own synthesized declaration
    /// (`driver.rs::synthesize_tuple_structs`) already expects, field types
    /// inferred purely from `a`/`b`'s own types exactly like any other
    /// struct literal's generic fields already are (no turbofish needed).
    fn lower_tuple_lit(&mut self, pair: Pair<Rule>) -> Expr {
        let span = self.span_of(&pair);
        let elems: Vec<Expr> = pair.into_inner().map(|p| self.lower_expr(p)).collect();
        let name = tuple_struct_name(elems.len());
        let fields = elems
            .into_iter()
            .enumerate()
            .map(|(i, e)| (i.to_string(), e))
            .collect();
        self.wrap(
            span,
            ExprKind::StructLit(Path::single(name), Vec::new(), fields),
        )
    }

    /// Consumes a leading `Rule::turbofish` pair off `inner` if present,
    /// returning its generic arguments — shared by `lower_call_expr`/
    /// `lower_struct_lit`, the two constructs that can carry one.
    fn lower_optional_turbofish(
        &mut self,
        inner: &mut std::iter::Peekable<Pairs<Rule>>,
    ) -> Vec<GenericArg> {
        if matches!(inner.peek().map(|p| p.as_rule()), Some(Rule::turbofish)) {
            inner
                .next()
                .unwrap()
                .into_inner()
                .map(|p| self.lower_generic_arg(p))
                .collect()
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
        self.wrap(
            span,
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            },
        )
    }

    fn lower_while_expr(&mut self, pair: Pair<Rule>) -> Expr {
        let span = self.span_of(&pair);
        let mut inner = pair.into_inner();
        let cond = Box::new(self.lower_expr(inner.next().unwrap()));
        let body = self.lower_block(inner.next().unwrap());
        self.wrap(span, ExprKind::While { cond, body })
    }

    fn lower_loop_expr(&mut self, pair: Pair<Rule>) -> Expr {
        let span = self.span_of(&pair);
        let body = self.lower_block(pair.into_inner().next().unwrap());
        self.wrap(span, ExprKind::Loop { body })
    }

    /// `grammar.pest`'s own `for_expr` funnels *two* real syntactic shapes
    /// through one rule (`(".." ~ additive)?` is optional) — disambiguated
    /// here, not in the grammar itself, by checking whether the second child
    /// is another `additive` (the range form, `ExprKind::For`) or the `block`
    /// directly (the new element-based form, `ExprKind::ForIn` — `doc/
    /// backlog-done.md`'s own "`for x in array`" item).
    fn lower_for_expr(&mut self, pair: Pair<Rule>) -> Expr {
        let span = self.span_of(&pair);
        let mut inner = pair.into_inner();
        let var = inner.next().unwrap().as_str().to_string();
        let first = self.lower_additive(inner.next().unwrap());
        let next = inner.next().unwrap();
        if next.as_rule() == Rule::additive {
            let end = Box::new(self.lower_additive(next));
            let body = self.lower_block(inner.next().unwrap());
            self.wrap(
                span,
                ExprKind::For {
                    var,
                    start: Box::new(first),
                    end,
                    body,
                },
            )
        } else {
            let body = self.lower_block(next);
            self.wrap(
                span,
                ExprKind::ForIn {
                    var,
                    iter: Box::new(first),
                    body,
                },
            )
        }
    }

    fn lower_array_lit(&mut self, pair: Pair<Rule>) -> Expr {
        let span = self.span_of(&pair);
        let Some(body) = pair.into_inner().next() else {
            return self.wrap(span, ExprKind::ArrayLit(Vec::new()));
        };
        match body.as_rule() {
            Rule::array_repeat => {
                let mut inner = body.into_inner();
                let value = inner.next().unwrap();
                let count = inner.next().unwrap();
                match count.as_rule() {
                    // `[value; N]`, `N` a literal — re-lowers the *same*
                    // parsed `value` pair `N` times (cheap: `Pair` is a
                    // reference into the token stream, not an owned deep
                    // copy) rather than lowering once and cloning the
                    // resulting `Expr`, so each copy gets its own distinct
                    // `NodeId` — every other node in this AST is unique per
                    // occurrence (see `ast.rs`'s own doc comment on
                    // `NodeId`), and `node_types` (keyed by `NodeId`) would
                    // silently collapse all `N` copies into one entry
                    // otherwise.
                    Rule::numeric_lit => {
                        let count_text = count.as_str();
                        // `numeric_lit`'s own text, possibly `:suffix`-
                        // terminated — a repeat count is never suffixed in
                        // practice, but strip it defensively rather than let
                        // `.parse` reject it outright.
                        let n: usize = count_text.split(':').next().unwrap().parse().unwrap_or_else(|e| {
                            panic!("array-repeat count {count_text:?} is not a valid array size: {e}")
                        });
                        let elems = (0..n).map(|_| self.lower_expr(value.clone())).collect();
                        self.wrap(span, ExprKind::ArrayLit(elems))
                    }
                    // `[value; N]`, `N` naming a const generic — its value
                    // isn't known until monomorphization, so this can't be
                    // expanded here; kept as a real node, resolved through
                    // ordinary type inference instead (see `infer.rs`).
                    Rule::ident => {
                        let count_span = self.span_of(&count);
                        let value = Box::new(self.lower_expr(value));
                        let count = Box::new(self.wrap(
                            count_span,
                            ExprKind::Path(Path {
                                segments: vec![count.as_str().to_string()],
                            }),
                        ));
                        self.wrap(span, ExprKind::ArrayRepeat { value, count })
                    }
                    // `[value; Dims...]` — a whole *pack* reference, not one
                    // named const generic (`doc/backlog.md`'s own "Toward a
                    // matmul-based tensorial XOR" follow-on) — mirrors `lower_
                    // array_dim`'s own identical extraction (the pack's own
                    // bare name, discarding the ordinary `Path` shape),
                    // duplicated rather than shared since `pack_ref`'s own
                    // pest shape here (`ident ~ pack_marker`, matched whole)
                    // differs from `array_dim`'s (`expr ~ pack_marker?`,
                    // matched via its own inner `expr` first).
                    Rule::pack_ref => {
                        let count_span = self.span_of(&count);
                        let value = Box::new(self.lower_expr(value));
                        let name = count.into_inner().next().unwrap().as_str().to_string();
                        let count = Box::new(self.wrap(count_span, ExprKind::PackRef(name)));
                        self.wrap(span, ExprKind::ArrayRepeat { value, count })
                    }
                    r => unreachable!(
                        "array_repeat's own count must be numeric_lit, pack_ref, or ident, got {r:?}"
                    ),
                }
            }
            Rule::array_list => {
                let elems = body.into_inner().map(|p| self.lower_expr(p)).collect();
                self.wrap(span, ExprKind::ArrayLit(elems))
            }
            other => unreachable!(
                "array_lit's own body must be array_repeat or array_list, got {other:?}"
            ),
        }
    }

    fn lower_literal(&mut self, pair: Pair<Rule>) -> Expr {
        let inner = pair.into_inner().next().unwrap();
        match inner.as_rule() {
            Rule::imaginary_lit => self.lower_imaginary_lit(inner),
            Rule::numeric_lit => self.lower_numeric_lit(inner),
            Rule::bool_lit => self.lower_bool_lit(inner),
            Rule::string_lit => self.lower_string_lit(inner),
            r => unreachable!("literal: unexpected rule {r:?}"),
        }
    }

    /// Full erasure at lowering time, no `ExprKind` of its own — mirrors
    /// `lower_array_lit`'s own literal-`N` `array_repeat` case exactly: a
    /// string literal becomes an ordinary `ArrayLit` of `i8`-suffixed
    /// `NumberLit`s, one per UTF-8 byte (not `char` — nothing elsewhere in
    /// this compiler has any text-encoding stance to contradict). Each
    /// byte gets its own fresh `Expr`/`NodeId` via `self.wrap`, the same
    /// "every node is unique per occurrence" reasoning `array_repeat`'s own
    /// copies already need (`node_types` is keyed by `NodeId`).
    fn lower_string_lit(&mut self, pair: Pair<Rule>) -> Expr {
        let span = self.span_of(&pair);
        let text = pair.as_str();
        let content = &text[1..text.len() - 1]; // strip the surrounding quotes
        let elems = content
            .bytes()
            .map(|b| {
                self.wrap(
                    span,
                    ExprKind::NumberLit {
                        text: b.to_string(),
                        suffix: Some("i8".to_string()),
                    },
                )
            })
            .collect();
        self.wrap(span, ExprKind::ArrayLit(elems))
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
        self.wrap(
            span,
            ExprKind::NumberLit {
                text: text.to_string(),
                suffix,
            },
        )
    }

    fn lower_imaginary_lit(&mut self, pair: Pair<Rule>) -> Expr {
        let span = self.span_of(&pair);
        let text = pair.as_str();
        let number_part = &text[..text.len() - 1]; // strip trailing "i"
        self.wrap(
            span,
            ExprKind::ImaginaryLit {
                text: number_part.to_string(),
                suffix: None,
            },
        )
    }
}

fn split_type_suffix(text: &str) -> (&str, Option<String>) {
    match text.find(':') {
        Some(idx) => (&text[..idx], Some(text[idx + 1..].to_string())),
        None => (text, None),
    }
}
