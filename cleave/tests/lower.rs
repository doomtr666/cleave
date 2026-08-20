use cleave::ast::*;
use cleave::lower::Lowerer;
use cleave::parser::{CleaveParser, Rule};
use pest::Parser;

fn lower_program(src: &str) -> Program {
    let pair = CleaveParser::parse(Rule::program, src)
        .unwrap_or_else(|e| panic!("parse failed for {src:?}: {e}"))
        .next()
        .unwrap();
    Lowerer::new(FileId(0)).lower_program(pair)
}

fn lower_one_fn(src: &str) -> FnDecl {
    let program = lower_program(src);
    assert_eq!(program.items.len(), 1, "expected exactly one item in {src:?}");
    match program.items.into_iter().next().unwrap().kind {
        ItemKind::Fn(f) => f,
        other => panic!("expected a fn item, got {other:?}"),
    }
}

fn only_stmt_expr(body: &Option<Block>) -> &Expr {
    // helper for tests that just want the tail expression
    body.as_ref().expect("expected a body").tail.as_deref().expect("expected a tail expression")
}

#[test]
fn array_repeat_literal_desugars_to_n_copies() {
    let f = lower_one_fn("fn f() { [0.0; 4] }");
    match &only_stmt_expr(&f.body).kind {
        ExprKind::ArrayLit(elems) => {
            assert_eq!(elems.len(), 4);
            for e in elems {
                assert!(matches!(&e.kind, ExprKind::NumberLit { text, .. } if text == "0.0"));
            }
        }
        other => panic!("expected ArrayLit, got {other:?}"),
    }
}

#[test]
fn array_repeat_literals_own_copies_each_get_a_distinct_node_id() {
    // Every other node in the AST is unique per occurrence (see `ast.rs`'s
    // own `NodeId` doc comment) -- a repeat literal's own desugared copies
    // must be too, or `node_types` (keyed by `NodeId`) would silently
    // collapse them into one entry.
    let f = lower_one_fn("fn f() { [1; 3] }");
    match &only_stmt_expr(&f.body).kind {
        ExprKind::ArrayLit(elems) => {
            let ids: std::collections::HashSet<NodeId> = elems.iter().map(|e| e.id).collect();
            assert_eq!(ids.len(), 3, "expected 3 distinct NodeIds, got {ids:?}");
        }
        other => panic!("expected ArrayLit, got {other:?}"),
    }
}

/// A string literal has no representation of its own — full erasure at
/// lowering time, mirroring `array_repeat`'s own literal-`N` case exactly:
/// `"hi"` becomes an ordinary `ArrayLit` of `i8`-suffixed `NumberLit`s, one
/// per UTF-8 byte of the string's own text.
#[test]
fn string_literal_desugars_to_an_i8_array_literal() {
    let f = lower_one_fn("fn f() { \"hi\" }");
    match &only_stmt_expr(&f.body).kind {
        ExprKind::ArrayLit(elems) => {
            let bytes: Vec<&str> = elems
                .iter()
                .map(|e| match &e.kind {
                    ExprKind::NumberLit { text, suffix } => {
                        assert_eq!(suffix.as_deref(), Some("i8"), "expected every byte to be i8-suffixed");
                        text.as_str()
                    }
                    other => panic!("expected NumberLit, got {other:?}"),
                })
                .collect();
            assert_eq!(bytes, vec!["104", "105"], "'h' = 104, 'i' = 105");
        }
        other => panic!("expected ArrayLit, got {other:?}"),
    }
}

/// Same reasoning as `array_repeat_literals_own_copies_each_get_a_distinct_
/// node_id` above — `node_types` is keyed by `NodeId`, reusing one across
/// bytes would silently collapse them.
#[test]
fn string_literal_bytes_each_get_a_distinct_node_id() {
    let f = lower_one_fn("fn f() { \"abc\" }");
    match &only_stmt_expr(&f.body).kind {
        ExprKind::ArrayLit(elems) => {
            let ids: std::collections::HashSet<NodeId> = elems.iter().map(|e| e.id).collect();
            assert_eq!(ids.len(), 3, "expected 3 distinct NodeIds, got {ids:?}");
        }
        other => panic!("expected ArrayLit, got {other:?}"),
    }
}

#[test]
fn array_repeat_literal_with_zero_count_is_an_empty_array() {
    let f = lower_one_fn("fn f() { [1; 0] }");
    match &only_stmt_expr(&f.body).kind {
        ExprKind::ArrayLit(elems) => assert!(elems.is_empty()),
        other => panic!("expected ArrayLit, got {other:?}"),
    }
}

#[test]
fn an_ordinary_comma_separated_array_literal_still_lowers_unaffected() {
    let f = lower_one_fn("fn f() { [1, 2, 3] }");
    match &only_stmt_expr(&f.body).kind {
        ExprKind::ArrayLit(elems) => assert_eq!(elems.len(), 3),
        other => panic!("expected ArrayLit, got {other:?}"),
    }
}

#[test]
fn lambda_lowers_with_params_and_body() {
    let f = lower_one_fn("fn f() { fn(a, b) { a + b } }");
    match &only_stmt_expr(&f.body).kind {
        ExprKind::Lambda { params, ret, body } => {
            assert_eq!(params.len(), 2);
            assert_eq!(params[0].name, "a");
            assert!(ret.is_none());
            match &body.tail.as_deref().unwrap().kind {
                ExprKind::Call(path, _, _, ..) => assert_eq!(path.segments, vec!["add".to_string()]),
                other => panic!("expected desugared add, got {other:?}"),
            }
        }
        other => panic!("expected a Lambda, got {other:?}"),
    }
}

#[test]
fn lambda_with_annotated_params_and_return() {
    let f = lower_one_fn("fn f() { fn(a: f64, b: f64) -> f64 { a } }");
    match &only_stmt_expr(&f.body).kind {
        ExprKind::Lambda { params, ret, .. } => {
            assert!(params[0].ty.is_some());
            assert!(ret.is_some());
        }
        other => panic!("expected a Lambda, got {other:?}"),
    }
}

#[test]
fn struct_lit_lowers_with_named_fields() {
    let f = lower_one_fn("fn f() { Vec2(x: 1.0, y: 2.0) }");
    match &only_stmt_expr(&f.body).kind {
        ExprKind::StructLit(path, _, fields) => {
            assert_eq!(path.segments, vec!["Vec2".to_string()]);
            assert_eq!(fields.len(), 2);
            assert_eq!(fields[0].0, "x");
            assert_eq!(fields[1].0, "y");
        }
        other => panic!("expected a StructLit, got {other:?}"),
    }
}

#[test]
fn zero_field_struct_lit_lowers_to_a_call_not_a_struct_lit() {
    // The one remaining parse-level ambiguity — see `grammar.pest`'s
    // `primary` comment — a zero-arg call and a zero-field struct
    // construction are indistinguishable at parse time; `infer.rs`'s
    // `infer_call` closes the gap for real structs, not `lower.rs`.
    let f = lower_one_fn("fn f() { Empty() }");
    match &only_stmt_expr(&f.body).kind {
        ExprKind::Call(path, _, args, ..) => {
            assert_eq!(path.segments, vec!["Empty".to_string()]);
            assert!(args.is_empty());
        }
        other => panic!("expected a Call, got {other:?}"),
    }
}

#[test]
fn operator_desugars_to_call() {
    let f = lower_one_fn("fn add(a, b) { a + b }");
    match &only_stmt_expr(&f.body).kind {
        ExprKind::Call(path, _, args, ..) => {
            assert_eq!(path.segments, vec!["add".to_string()]);
            assert_eq!(args.len(), 2);
        }
        other => panic!("expected a Call, got {other:?}"),
    }
}

#[test]
fn operator_chain_is_left_associative() {
    // a - b - c => sub(sub(a, b), c)
    let f = lower_one_fn("fn f(a, b, c) { a - b - c }");
    match &only_stmt_expr(&f.body).kind {
        ExprKind::Call(outer_path, _, outer_args, ..) => {
            assert_eq!(outer_path.segments, vec!["sub".to_string()]);
            match &outer_args[0].kind {
                ExprKind::Call(inner_path, _, inner_args, ..) => {
                    assert_eq!(inner_path.segments, vec!["sub".to_string()]);
                    assert_eq!(inner_args.len(), 2);
                }
                other => panic!("expected nested sub Call, got {other:?}"),
            }
        }
        other => panic!("expected a Call, got {other:?}"),
    }
}

#[test]
fn comparison_and_logical_and_implication_desugar() {
    let f = lower_one_fn("fn f(a, b) { a and b implies a or b }");
    match &only_stmt_expr(&f.body).kind {
        ExprKind::Call(path, _, _, ..) => assert_eq!(path.segments, vec!["implies".to_string()]),
        other => panic!("expected implies Call, got {other:?}"),
    }
}

#[test]
fn unary_minus_desugars_to_neg() {
    let f = lower_one_fn("fn f(a) { -a }");
    match &only_stmt_expr(&f.body).kind {
        ExprKind::Call(path, _, args, ..) => {
            assert_eq!(path.segments, vec!["neg".to_string()]);
            assert_eq!(args.len(), 1);
        }
        other => panic!("expected neg Call, got {other:?}"),
    }
}

/// `grammar.pest`'s own `for_expr` funnels both shapes through one rule
/// (`(".." ~ additive)?`) — `lower_for_expr` is what actually tells them
/// apart. The range form must keep lowering to `ExprKind::For` exactly as
/// before (a direct regression guard on the grammar's new optional group).
#[test]
fn for_range_still_lowers_to_for() {
    let f = lower_one_fn("fn f(arr) { for i in 0..5 { i } }");
    match &only_stmt_expr(&f.body).kind {
        ExprKind::For { var, .. } => assert_eq!(var, "i"),
        other => panic!("expected ExprKind::For, got {other:?}"),
    }
}

/// The new element-based form (`doc/backlog-done.md`'s own "`for x in
/// array`" item) lowers to the new `ExprKind::ForIn` instead.
#[test]
fn for_in_array_lowers_to_for_in() {
    let f = lower_one_fn("fn f(arr) { for x in arr { x } }");
    match &only_stmt_expr(&f.body).kind {
        ExprKind::ForIn { var, .. } => assert_eq!(var, "x"),
        other => panic!("expected ExprKind::ForIn, got {other:?}"),
    }
}

#[test]
fn field_access_vs_zero_arg_method_call_are_distinguishable() {
    let f_field = lower_one_fn("fn f(v) { v.x }");
    match &only_stmt_expr(&f_field.body).kind {
        ExprKind::FieldAccess(_, name) => assert_eq!(name, "x"),
        other => panic!("expected FieldAccess, got {other:?}"),
    }

    let f_call = lower_one_fn("fn f(v) { v.x() }");
    match &only_stmt_expr(&f_call.body).kind {
        ExprKind::MethodCall(_, name, args) => {
            assert_eq!(name, "x");
            assert!(args.is_empty());
        }
        other => panic!("expected zero-arg MethodCall, got {other:?}"),
    }
}

/// `a[i,j]` — one bracket group — collects both indices directly into *one*
/// `Index` node (`indices.len() == 2`), not folded into nested single-index
/// nodes the way an earlier version of this desugaring did: a tagged
/// (`#[mlir_type(...)]`) struct's own multi-index `Index<Container,Elem,K>`
/// dispatch needs the whole group intact (see `ast.rs`'s own `Index` doc
/// comment) — nesting would leave nothing sensible for an inner single-index
/// step to dispatch to.
#[test]
fn multi_index_collects_into_one_index_node() {
    let f = lower_one_fn("fn f(a, i, j) { a[i, j] }");
    match &only_stmt_expr(&f.body).kind {
        ExprKind::Index(base, indices) => {
            assert!(matches!(&base.kind, ExprKind::Path(p) if p.segments == ["a"]));
            assert_eq!(indices.len(), 2);
            assert!(matches!(&indices[0].kind, ExprKind::Path(p) if p.segments == ["i"]));
            assert!(matches!(&indices[1].kind, ExprKind::Path(p) if p.segments == ["j"]));
        }
        other => panic!("expected Index, got {other:?}"),
    }
}

/// `a[i][j]` — two *separate* bracket pairs, unlike `a[i,j]` above — still
/// nests: each bracket is its own `Index` node, one index each. For a real
/// array the two spellings stay semantically equivalent either way
/// (`infer.rs` peels one dimension per index regardless of how many arrive
/// per node); a tagged struct's own `Index` dispatch only ever recognizes
/// the single-node, whole-group form.
#[test]
fn separate_bracket_pairs_still_nest() {
    let f = lower_one_fn("fn f(a, i, j) { a[i][j] }");
    match &only_stmt_expr(&f.body).kind {
        ExprKind::Index(inner, outer_indices) => {
            assert_eq!(outer_indices.len(), 1);
            assert!(matches!(&outer_indices[0].kind, ExprKind::Path(p) if p.segments == ["j"]));
            match &inner.kind {
                ExprKind::Index(base, inner_indices) => {
                    assert!(matches!(&base.kind, ExprKind::Path(p) if p.segments == ["a"]));
                    assert_eq!(inner_indices.len(), 1);
                    assert!(matches!(&inner_indices[0].kind, ExprKind::Path(p) if p.segments == ["i"]));
                }
                other => panic!("expected nested Index, got {other:?}"),
            }
        }
        other => panic!("expected Index, got {other:?}"),
    }
}

#[test]
fn function_type_annotation_lowers_to_typekind_fn() {
    let f = lower_one_fn("fn apply(f: (i32, f64) -> bool) { f }");
    let ty = f.params[0].ty.as_ref().unwrap();
    match &ty.kind {
        TypeKind::Fn(params, ret) => {
            assert_eq!(params.len(), 2);
            match &params[0].kind {
                TypeKind::Path(p, _) => assert_eq!(p.segments, vec!["i32".to_string()]),
                other => panic!("expected Path(i32), got {other:?}"),
            }
            match &params[1].kind {
                TypeKind::Path(p, _) => assert_eq!(p.segments, vec!["f64".to_string()]),
                other => panic!("expected Path(f64), got {other:?}"),
            }
            match &ret.kind {
                TypeKind::Path(p, _) => assert_eq!(p.segments, vec!["bool".to_string()]),
                other => panic!("expected Path(bool), got {other:?}"),
            }
        }
        other => panic!("expected Fn, got {other:?}"),
    }
}

#[test]
fn multidim_array_type_desugars_to_nested_array() {
    let f = lower_one_fn("fn f(a: [f64; 3, 4]) -> f64 { 0.0 }");
    let ty = f.params[0].ty.as_ref().unwrap();
    match &ty.kind {
        TypeKind::Array(elem, _outer_dim) => match &elem.kind {
            TypeKind::Array(inner_elem, _inner_dim) => match &inner_elem.kind {
                TypeKind::Path(p, _) => assert_eq!(p.segments, vec!["f64".to_string()]),
                other => panic!("expected Path(f64), got {other:?}"),
            },
            other => panic!("expected nested Array, got {other:?}"),
        },
        other => panic!("expected Array, got {other:?}"),
    }
}

#[test]
fn let_mut_is_detected() {
    let f = lower_one_fn("fn f() { let mut acc = 0; acc }");
    match &f.body.as_ref().unwrap().stmts[0].kind {
        StmtKind::Let { mutable, name, .. } => {
            assert!(*mutable);
            assert_eq!(name, "acc");
        }
        other => panic!("expected Let, got {other:?}"),
    }
}

#[test]
fn plain_let_is_not_mutable() {
    let f = lower_one_fn("fn f() { let a = 0; a }");
    match &f.body.as_ref().unwrap().stmts[0].kind {
        StmtKind::Let { mutable, .. } => assert!(!mutable),
        other => panic!("expected Let, got {other:?}"),
    }
}

#[test]
fn reassignment_is_a_distinct_statement_kind() {
    let f = lower_one_fn("fn f() { let mut a = 0; a = a + 1; a }");
    match &f.body.as_ref().unwrap().stmts[1].kind {
        StmtKind::Assign { target, .. } => match &target.kind {
            ExprKind::Path(p) => assert_eq!(p.segments, vec!["a".to_string()]),
            other => panic!("expected Path target, got {other:?}"),
        },
        other => panic!("expected Assign, got {other:?}"),
    }
}

#[test]
fn numeric_literal_type_suffix_is_split_out() {
    let f = lower_one_fn("fn f() { 1.25:f64 }");
    match &only_stmt_expr(&f.body).kind {
        ExprKind::NumberLit { text, suffix } => {
            assert_eq!(text, "1.25");
            assert_eq!(suffix.as_deref(), Some("f64"));
        }
        other => panic!("expected NumberLit, got {other:?}"),
    }
}

#[test]
fn imaginary_literal_strips_trailing_i() {
    let f = lower_one_fn("fn f() { 3 + 4i }");
    match &only_stmt_expr(&f.body).kind {
        ExprKind::Call(_, _, args, ..) => match &args[1].kind {
            ExprKind::ImaginaryLit { text, .. } => assert_eq!(text, "4"),
            other => panic!("expected ImaginaryLit, got {other:?}"),
        },
        other => panic!("expected Call, got {other:?}"),
    }
}

#[test]
fn qualified_call_path_has_two_segments() {
    let f = lower_one_fn("fn f(a, b) { Ring::add(a, b) }");
    match &only_stmt_expr(&f.body).kind {
        ExprKind::Call(path, _, _, ..) => {
            assert_eq!(path.segments, vec!["Ring".to_string(), "add".to_string()]);
        }
        other => panic!("expected Call, got {other:?}"),
    }
}

#[test]
fn turbofish_on_a_call_lowers_to_explicit_generic_args() {
    let f = lower_one_fn("fn f() { fibonacci::<f64>(1.0) }");
    match &only_stmt_expr(&f.body).kind {
        ExprKind::Call(path, generics, args, ..) => {
            assert_eq!(path.segments, vec!["fibonacci".to_string()]);
            assert_eq!(args.len(), 1);
            match &generics[..] {
                [GenericArg::Type(t)] => match &t.kind {
                    TypeKind::Path(p, _) => assert_eq!(p.segments, vec!["f64".to_string()]),
                    other => panic!("expected Path(f64), got {other:?}"),
                },
                other => panic!("expected one type generic arg, got {other:?}"),
            }
        }
        other => panic!("expected a Call, got {other:?}"),
    }
}

#[test]
fn an_ordinary_call_with_no_turbofish_has_empty_explicit_generics() {
    let f = lower_one_fn("fn f() { fibonacci(1) }");
    match &only_stmt_expr(&f.body).kind {
        ExprKind::Call(_, generics, _, ..) => assert!(generics.is_empty()),
        other => panic!("expected a Call, got {other:?}"),
    }
}

#[test]
fn turbofish_on_a_struct_construction_lowers_to_explicit_generic_args() {
    let f = lower_one_fn("fn f() { Matrix::<f64, 4, 4>(values: v) }");
    match &only_stmt_expr(&f.body).kind {
        ExprKind::StructLit(path, generics, fields) => {
            assert_eq!(path.segments, vec!["Matrix".to_string()]);
            assert_eq!(fields.len(), 1);
            assert_eq!(generics.len(), 3);
            assert!(matches!(&generics[0], GenericArg::Type(_)));
            assert!(matches!(&generics[1], GenericArg::Const(_)));
            assert!(matches!(&generics[2], GenericArg::Const(_)));
        }
        other => panic!("expected a StructLit, got {other:?}"),
    }
}

#[test]
fn every_node_gets_a_span_within_source_bounds() {
    let src = "fn add(a, b) { a + b }";
    let f = lower_one_fn(src);
    let e = only_stmt_expr(&f.body);
    assert!(e.span.start < e.span.end);
    assert!(e.span.end <= src.len());
    assert_eq!(e.span.file, FileId(0));
}

#[test]
fn full_program_lowers_end_to_end() {
    let src = "\
struct Vec2 { x: f64, y: f64 }

algebra Ring<T> {
    fn add(a: T, b: T) -> T;
    axiom assoc_add(a: T, b: T, c: T): add(add(a,b),c) == add(a,add(b,c));
}

impl Ring<Vec2> {
    fn add(a: Vec2, b: Vec2) -> Vec2 { Vec2(a.x + b.x, a.y + b.y) }
}

fn main() -> i32 {
    let a = 1;
    let mut acc = 0;
    acc = acc + a;
    0
}
";
    let program = lower_program(src);
    assert_eq!(program.items.len(), 4);
    assert!(matches!(program.items[0].kind, ItemKind::Struct(_)));
    assert!(matches!(program.items[1].kind, ItemKind::Algebra(_)));
    assert!(matches!(program.items[2].kind, ItemKind::Impl(_)));
    assert!(matches!(program.items[3].kind, ItemKind::Fn(_)));
}

#[test]
fn bool_const_generic_argument_lowers_to_a_bool_lit_generic_arg() {
    let f = lower_one_fn("fn f(a: Grid<f64, true>) { a }");
    let ty = f.params[0].ty.as_ref().expect("annotated param");
    match &ty.kind {
        TypeKind::Path(path, args) => {
            assert_eq!(path.segments, vec!["Grid".to_string()]);
            assert_eq!(args.len(), 2);
            match &args[1] {
                GenericArg::Const(e) => assert!(matches!(e.kind, ExprKind::BoolLit(true)), "got: {:?}", e.kind),
                other => panic!("expected a const generic arg, got {other:?}"),
            }
        }
        other => panic!("expected a path type, got {other:?}"),
    }
}

#[test]
fn inherent_impl_lowers_to_its_own_item_kind() {
    let program = lower_program("struct Vec2 { x: f64 }\nimpl struct Vec2 {\n    fn len(v) { v.x }\n}");
    assert_eq!(program.items.len(), 2);
    match &program.items[1].kind {
        ItemKind::InherentImpl(d) => {
            assert!(d.generics.is_empty());
            assert_eq!(d.fns.len(), 1);
            assert_eq!(d.fns[0].name, "len");
            match &d.target.kind {
                TypeKind::Path(p, args) => {
                    assert_eq!(p.segments, vec!["Vec2".to_string()]);
                    assert!(args.is_empty());
                }
                other => panic!("expected Path(Vec2), got {other:?}"),
            }
        }
        other => panic!("expected InherentImpl, got {other:?}"),
    }
}

#[test]
fn generic_inherent_impl_carries_its_own_generics_and_target_args() {
    let program = lower_program("struct Matrix<T> { data: T }\nimpl<T: Float> struct Matrix<T> {\n    fn get(m) { m }\n}");
    match &program.items[1].kind {
        ItemKind::InherentImpl(d) => {
            assert_eq!(d.generics.len(), 1);
            match &d.target.kind {
                TypeKind::Path(p, args) => {
                    assert_eq!(p.segments, vec!["Matrix".to_string()]);
                    assert_eq!(args.len(), 1);
                }
                other => panic!("expected Path(Matrix<T>), got {other:?}"),
            }
        }
        other => panic!("expected InherentImpl, got {other:?}"),
    }
}

#[test]
fn multi_target_algebra_impl_populates_extra_targets() {
    let program = lower_program(
        "algebra MatMul<A, B, C> { fn mul(a: A, b: B) -> C; }
         impl<T> MatMul<T, T, T> {\n    fn mul(a, b) { a }\n}",
    );
    match &program.items[1].kind {
        ItemKind::Impl(d) => {
            assert_eq!(d.algebra, "MatMul");
            assert_eq!(d.generics.len(), 1);
            assert_eq!(d.extra_targets.len(), 2, "T, T -- two targets beyond the first");
        }
        other => panic!("expected Impl, got {other:?}"),
    }
}

#[test]
fn single_target_algebra_impl_has_empty_extra_targets() {
    let program = lower_program(
        "algebra Ring<T> { fn add(a: T, b: T) -> T; }
         impl Ring<Vec2> { fn add(a, b) { a } }",
    );
    match &program.items[1].kind {
        ItemKind::Impl(d) => assert!(d.extra_targets.is_empty()),
        other => panic!("expected Impl, got {other:?}"),
    }
}

#[test]
fn export_fn_sets_is_export_and_leaves_symbol_none_without_an_override() {
    let f = lower_one_fn("export fn f(x: i32) -> i32 { x }");
    assert!(f.is_export);
    assert!(f.export_symbol.is_none());
    assert!(!f.is_extern);
    assert!(f.body.is_some(), "export fn keeps its real body");
}

#[test]
fn export_fn_with_a_parenthesized_symbol_sets_export_symbol() {
    let f = lower_one_fn("export(my_symbol) fn f(x: i32) -> i32 { x }");
    assert!(f.is_export);
    assert_eq!(f.export_symbol.as_deref(), Some("my_symbol"));
}

#[test]
fn extern_and_export_are_mutually_exclusive_at_the_grammar_level() {
    // `fn_decl`'s own `(extern_kw | export_kw)?` alternation admits at most
    // one -- `extern export fn`/`export extern fn` should fail to parse,
    // not silently pick one.
    let result = CleaveParser::parse(Rule::program, "extern export fn f(x: i32) -> i32;");
    assert!(result.is_err(), "expected a parse error for `extern export fn`");
}
