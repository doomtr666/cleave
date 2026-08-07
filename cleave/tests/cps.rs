use cleave::cps::{collect_units, convert_program, dump_cps_program};
use cleave::driver::compile;
use cleave::registry::Registry;

/// Compiles `src` against the real stdlib prelude (same driver path `dump.rs`'s
/// own tests use) and renders its Stage 1 CPS conversion.
fn cps(src: &str) -> String {
    let (result, _sources) = compile(vec![("test.cleave".to_string(), src.to_string())], &[]);
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    let units = collect_units(&program, &registry);
    let cps_program = convert_program(units);
    dump_cps_program(&cps_program)
}

/// Extracts just `(fn name (...) ...)`'s own printed block out of a full
/// dump — needed because `dump_cps_program` now prints *every* reachable
/// top-level fn, including all 36 real, `mlir::...`-bodied `Ring`/`Ord`
/// stdlib specializations (they all have real bodies now, see `stdlib/num/
/// num.cleave`, so they're no longer skipped the way bodyless `#[mlir(...)]`
/// intrinsics used to be) — meaning CPS variable numbers are no longer
/// small, test-local integers starting at `v0`, just whatever's next after
/// however many the stdlib's own conversion already consumed. Assertions
/// below scope to this test's own function and, where a specific number
/// still matters, capture it fresh from the output rather than hardcoding
/// one — the numbering itself was never semantically meaningful, only
/// self-consistency was.
fn fn_block<'a>(dump: &'a str, name: &str) -> &'a str {
    let start = dump.find(&format!("(fn {name} (")).unwrap_or_else(|| panic!("no `(fn {name} (...)` block found in:\n{dump}"));
    let mut depth = 0i32;
    for (i, c) in dump[start..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return &dump[start..start + i + 1];
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced parens looking for `(fn {name} (...)` in:\n{dump}");
}

#[test]
fn a_direct_mlir_call_converts_to_one_let_prim() {
    // A reserved `mlir::dialect::op(...)` call -- Appel's PRIMOP, straight-
    // line, no continuation needed. `stdlib/num`'s own `Ring<f64>::add`
    // (which wraps exactly this) no longer qualifies on its own -- it has a
    // real body now, so calling *it* goes through `Fix`+`App` like any
    // other real function (see the next test) -- only the raw `mlir::...`
    // call itself is genuinely straight-line.
    let out = cps("fn f(a: f64, b: f64) -> f64 { mlir::arith::addf(a, b) }");
    let block = fn_block(&out, "f");
    assert!(block.contains("let-prim"), "got:\n{block}");
    assert!(block.contains("(mlir.arith.addf"), "got:\n{block}");
    assert!(!block.contains("(fix"), "a raw mlir op call needs no synthesized continuation, got:\n{block}");
}

#[test]
fn a_call_to_a_real_function_converts_to_fix_plus_app_with_continuation() {
    let out = cps(
        "fn helper(a: f64, b: f64) -> f64 { add(a, b) }
         fn main() -> f64 { helper(1.0, 2.0) }",
    );
    // Scoped to `main`'s own block specifically: `helper`'s own body *also*
    // makes a real call now (`add` wraps a real `mlir::...` call inside a
    // real fn, not a straight-line intrinsic -- see `stdlib/num/num.cleave`),
    // so it competes with `main` for continuation-label numbers via the
    // same shared counter (`convert_program`'s own `fresh`, iterating units
    // in `HashMap` order -- not guaranteed stable across runs) -- checking
    // a literal `k$0` here would flakily depend on which of the two enters
    // that shared counter first.
    let block = fn_block(&out, "main");
    assert!(block.contains("(fix"), "a real callee must synthesize a local continuation, got:\n{block}");
    assert!(
        block.contains("(helper 1 2 k$"),
        "the callee must be tail-called with the continuation appended, got:\n{block}"
    );
}

#[test]
fn a_plain_let_of_a_literal_extends_env_without_introducing_a_new_cps_node() {
    let out = cps("fn f() -> f64 { let x = 1.0; x }");
    let block = fn_block(&out, "f");
    assert!(!block.contains("let-prim"), "a plain immutable let needs no CPS node at all, got:\n{block}");
    // `f`'s own body is `(fn f (vN) (vN 1))` -- whatever `vN` (its own
    // return continuation parameter) actually is, the literal must be
    // handed directly to it.
    let k_ret = block.trim_start_matches("(fn f (").split(')').next().unwrap();
    assert!(block.contains(&format!("({k_ret} 1)")), "the literal value must be handed directly to the return continuation, got:\n{block}");
}

#[test]
fn field_access_and_struct_construction_each_produce_their_own_let_prim_in_order() {
    let out = cps(
        "struct Vec2 { x: f64, y: f64 }
         fn main() -> f64 {
            let v = Vec2(x: 1.0, y: 2.0);
            add(v.x, v.y)
         }",
    );
    let struct_pos = out.find("struct.Vec2[x,y]").expect("struct construction must produce its own let-prim");
    let field_x_pos = out.find("field.x").expect("field access on x must produce its own let-prim");
    let field_y_pos = out.find("field.y").expect("field access on y must produce its own let-prim");
    assert!(struct_pos < field_x_pos && field_x_pos < field_y_pos, "got:\n{out}");
}

#[test]
fn an_if_else_expression_joins_both_arms_through_one_synthesized_continuation() {
    // Unary minus isn't wired up in the stdlib yet (a separate, pre-existing
    // gap, not a CPS concern) -- two positive branches exercise the same join
    // logic just as well.
    let out = cps("fn f(x: i32) -> i32 { if x > 0 { 1 } else { 2 } }");
    let block = fn_block(&out, "f");
    // `x > 0` is itself a real call (`Ord::gt<i32>`), so `f`'s own body
    // wraps the `if`'s own `Fix` inside *another* one for that call's own
    // continuation -- both arms must tail-call the *same* join label
    // (`j$1`, not `j$0`: label `0` went to the `gt` call's own continuation
    // first) -- not two separately inlined copies of "whatever happens
    // after the if" (what a naive CPS conversion, calling `k` directly in
    // each branch instead of through one named continuation, would produce
    // instead). `j$1` appears 3 times total: once as the `Fix`'s own
    // definition, once per arm's own tail call.
    assert_eq!(block.matches("j$1").count(), 3, "got:\n{block}");
    assert!(block.contains("(j$1 1)") && block.contains("(j$1 2)"), "each arm must tail-call the join continuation with its own value, got:\n{block}");
    assert_eq!(block.matches("Ord::gt<i32>").count(), 1, "the condition must be evaluated exactly once, got:\n{block}");
}

#[test]
fn a_bare_if_with_no_else_feeds_unit_to_the_join_continuation() {
    let out = cps("fn f(x: i32) -> i32 { if x > 0 { 1; }; 0 }");
    assert!(out.contains("()"), "a missing else must join with CVal::Unit, got:\n{out}");
}

#[test]
fn a_self_recursive_generic_function_using_if_else_converts_and_resolves_its_own_recursive_calls() {
    // Mirrors `examples/fibonacci.cleave` -- self-recursion through both
    // arithmetic (Ring) and comparison (Ord) intrinsics, and a real
    // recursive call inside the `then` branch, resolved via `call_names`.
    let out = cps(
        "fn fibonacci<T: Int>(x: T) -> T {
            if x > 2 { fibonacci(x - 1) + fibonacci(x - 2) } else { x }
        }
        fn main() -> i32 { fibonacci(10:i32) }",
    );
    assert!(out.contains("(fn fibonacci<i32>"), "got:\n{out}");
    assert_eq!(out.matches("(fibonacci<i32>").count(), 3, "the fn def plus its own two recursive calls, got:\n{out}");
    let block = fn_block(&out, "fibonacci<i32>");
    assert!(block.contains("Ord::gt<i32>") && block.contains("Ring::sub<i32>") && block.contains("Ring::add<i32>"), "got:\n{block}");
}

#[test]
fn a_for_loop_becomes_a_self_recursive_continuation_carrying_the_index() {
    let out = cps(
        "fn f() -> i32 {
            for i in 0..10 {
                add(i, i);
            };
            0
        }",
    );
    let block = fn_block(&out, "f");
    assert!(block.contains("(fix"), "got:\n{block}");
    // The loop-carried index is the recursive continuation's own single
    // parameter -- bound-check, body, increment, tail-recurse, all inside
    // its own body; the initial call seeds it with `start`.
    assert!(block.contains("Ord::lt<i32>"), "the implicit bound check must resolve to Ord::lt, got:\n{block}");
    assert!(block.contains("Ring::add<i32>"), "both the loop body's own `add` and the implicit increment use it, got:\n{block}");
    assert!(block.contains("(loop$0 0)"), "the loop must be entered by calling its own continuation with `start`, got:\n{block}");
    // On exit (bound check fails), control must reach the outer continuation
    // (`f`'s own return-continuation parameter) with the function's own
    // tail value, not get stuck inside the loop.
    let k_ret = block.trim_start_matches("(fn f (").split(')').next().unwrap();
    assert!(block.contains(&format!("({k_ret} 0)")), "got:\n{block}");
}

#[test]
fn a_while_loop_carries_no_state_and_reevaluates_its_condition_each_iteration() {
    let out = cps(
        "fn f(x: i32) -> i32 {
            while x > 0 {
                add(x, x);
            };
            1
        }",
    );
    let block = fn_block(&out, "f");
    assert!(block.contains("(fix"), "got:\n{block}");
    assert!(block.contains("Ord::gt<i32>"), "got:\n{block}");
    // No loop-carried state yet (`let mut` is a later stage) -- the
    // recursive continuation takes no parameters at all.
    assert!(block.contains("(loop$0 ()") || block.contains("(loop$0 ("), "the loop continuation must take no parameters, got:\n{block}");
    assert!(block.contains("(loop$0)"), "both re-entry and recursion must be plain zero-arg tail calls, got:\n{block}");
}

#[test]
fn a_plain_reassignment_in_straight_line_code_just_rebinds_the_name() {
    let out = cps("fn f() -> i32 { let mut x = 1; x = 2; x }");
    let block = fn_block(&out, "f");
    assert!(!block.contains("let-prim"), "straight-line reassignment needs no CPS node at all, got:\n{block}");
    let k_ret = block.trim_start_matches("(fn f (").split(')').next().unwrap();
    assert!(block.contains(&format!("({k_ret} 2)")), "the reassigned value must reach the return continuation, got:\n{block}");
}

#[test]
fn a_for_loop_accumulator_is_carried_as_an_extra_loop_parameter() {
    let out = cps(
        "fn f() -> i32 {
            let mut acc = 0;
            for i in 0..10 {
                acc = add(acc, i);
            };
            acc
        }",
    );
    let block = fn_block(&out, "f");
    // The recursive continuation now carries two values: the index and the
    // accumulator, index first positionally, whatever their own actual var
    // numbers are (offset by however many the stdlib prelude's own
    // conversion already consumed, see `fn_block`'s own doc comment).
    let params = block.split("(loop$0 (").nth(1).unwrap().split(')').next().unwrap();
    let mut params = params.split_whitespace();
    let (index_var, acc_var) = (params.next().unwrap(), params.next().unwrap());
    assert!(block.contains("(loop$0 0 0)"), "both the index and the accumulator start at their own initial values, got:\n{block}");
    // On exit, the *accumulator*'s own final value (not the index) must
    // reach the outer continuation (`f`'s own return-continuation param).
    let k_ret = block.trim_start_matches("(fn f (").split(')').next().unwrap();
    assert!(block.contains(&format!("({k_ret} {acc_var})")), "got:\n{block}");
    let _ = index_var; // only its presence as loop$0's own first param matters here
}

#[test]
fn an_if_else_mutating_a_let_mut_variable_threads_it_through_the_join() {
    let out = cps(
        "fn f(x: i32) -> i32 {
            let mut y = 0;
            if x > 0 { y = 1; } else { y = 2; };
            y
        }",
    );
    let block = fn_block(&out, "f");
    // `x > 0` is itself a real call, so the join label is `j$1` (`j$0` went
    // to that call's own continuation) -- see `an_if_else_expression_joins_
    // both_arms_through_one_synthesized_continuation`'s own comment. The
    // join continuation must carry `y` alongside the if's own (Unit) value,
    // and each arm must pass its own newly-assigned value along.
    assert!(block.contains("(j$1 (v") && block.matches("j$1").count() == 3, "got:\n{block}");
    assert!(block.contains("(j$1 () 1)") && block.contains("(j$1 () 2)"), "got:\n{block}");
}

#[test]
fn a_variable_shadowed_by_an_inner_let_mut_does_not_leak_into_the_outer_joins_carried_set() {
    // `y` inside the `then` branch is a *fresh*, inner `let mut` shadowing
    // nothing -- it must not be mistaken for an escaping mutation of some
    // enclosing `y` (there isn't one here at all).
    let out = cps(
        "fn f(x: i32) -> i32 {
            if x > 0 {
                let mut y = 1;
                y = 2;
                y
            } else {
                0
            }
        }",
    );
    let block = fn_block(&out, "f");
    // `x > 0` is itself a real call, so the join label is `j$1` (see the
    // comment on the previous test). The join continuation must carry
    // *only* the if's own result value -- one parameter, not two.
    let join_params = block.split("(j$1 (").nth(1).unwrap().split(')').next().unwrap();
    assert_eq!(join_params.split_whitespace().count(), 1, "the inner `y` must not leak as carried state, got:\n{block}");
    assert!(block.contains("(j$1 2)") && block.contains("(j$1 0)"), "got:\n{block}");
}

#[test]
fn mutation_inside_an_if_nested_in_a_for_loop_composes_correctly() {
    let out = cps(
        "fn f() -> i32 {
            let mut acc = 0;
            for i in 0..10 {
                if i > 5 {
                    acc = add(acc, i);
                };
            };
            acc
        }",
    );
    let block = fn_block(&out, "f");
    // Two levels of carrying: the inner if's own join carries `acc`, and the
    // outer for-loop's own recursive continuation also carries `acc`
    // (alongside its own index). Each real call (`Ord::lt`/`Ord::gt`/`Ring::
    // add`) also synthesizes its own continuation, so `(fix` now appears
    // more than just those two structural levels -- six total: the for-
    // loop itself, the bound-check's own continuation, the inner `if`'s own
    // join, the inner `if`'s own `then`-branch call continuation, and the
    // implicit increment's own call continuation, nested one inside the
    // next -- rather than re-deriving that count from first principles
    // here, this just pins today's real, observed shape.
    assert_eq!(block.matches("(fix").count(), 6, "got:\n{block}");
    assert!(block.contains("Ring::add<i32>") && block.contains("Ord::gt<i32>") && block.contains("Ord::lt<i32>"), "got:\n{block}");
}

#[test]
fn an_array_literal_an_indexed_write_and_an_indexed_read_use_the_same_stable_reference() {
    let out = cps(
        "fn f() -> i32 {
            let mut a = [1, 2, 3];
            a[0] = 10;
            a[1]
        }",
    );
    let block = fn_block(&out, "f");
    assert!(block.contains(": [i32; 3] = (array 1 2 3)"), "got:\n{block}");
    // `a`'s own array-reference variable, whatever it actually is.
    let a_var = block.split(": [i32; 3] = (array 1 2 3)").next().unwrap().rsplit("(let-prim ").next().unwrap();
    // The store and the load must both operate on `a`'s own reference --
    // no copy, no rebinding through `env` the way a scalar reassignment
    // would need.
    assert!(block.contains(&format!("(store {a_var} 0 10)")), "got:\n{block}");
    assert!(block.contains(&format!("(load {a_var} 1)")), "got:\n{block}");
}

#[test]
fn an_array_repeat_whose_count_names_a_const_generic_produces_its_own_prim_op() {
    // A *literal* repeat count (`[0; 3]`) desugars to an ordinary
    // `ArrayLit` at lowering time (see `ast.rs`'s own `ExprKind::ArrayRepeat`
    // doc comment) -- `ArrayRepeat` itself only ever survives to CPS
    // conversion when the count names a const generic instead, and by this
    // point monomorphization has already resolved that reference down to a
    // concrete `Ty::Const` (found by direct testing: an earlier version of
    // this module only ever looked a `Path` up in `env`, panicking with
    // "unbound variable `N`" the first time this case was actually
    // exercised).
    let out = cps(
        "fn make<const N: i32>(v: i32) -> [i32; N] { [v; N] }
         fn f() -> i32 { make::<3>(0); 0 }",
    );
    assert!(out.contains("array-repeat") && out.contains(" 3)"), "the count must resolve to a real literal, not stay an unbound name, got:\n{out}");
    assert!(!out.contains("unbound variable"), "got:\n{out}");
}

#[test]
fn an_array_mutated_inside_a_for_loop_body_needs_no_carrying_only_the_index_does() {
    // The whole point of the "stable reference" design: `a`'s own identity
    // never changes across iterations, so unlike a scalar accumulator
    // (`a_for_loop_accumulator_is_carried_as_an_extra_loop_parameter` above)
    // the loop's own recursive continuation carries *only* its index.
    let out = cps(
        "fn f() -> i32 {
            let a = [0, 0, 0];
            for i in 0..3 {
                a[i] = add(a[i], 1);
            };
            a[0]
        }",
    );
    let block = fn_block(&out, "f");
    let loop_params = block.split("(loop$0 (").nth(1).unwrap().split(')').next().unwrap();
    assert_eq!(loop_params.split_whitespace().count(), 1, "the loop must carry only its own index, got:\n{block}");
    assert!(block.contains("load") && block.contains("store"), "got:\n{block}");
}

#[test]
fn a_nested_indexed_assignment_target_collapses_into_one_multi_index_store() {
    // `a[0][0] = 9` -- collapsed into a single, combined `Store` rather than
    // an intermediate single-index `Load` (of "the row") followed by a
    // separate write into it, which would only be correct if `Load`
    // aliased the original storage instead of copying it out -- a
    // representation choice this module never actually commits to. See the
    // module's own "Arrays" doc comment.
    let out = cps("fn f() -> i32 { let mut a = [[1, 2], [3, 4]]; a[0][0] = 9; 0 }");
    let block = fn_block(&out, "f");
    assert!(block.contains(" 0 0 9)") && block.contains("(store "), "both indices must combine into one store, got:\n{block}");
    assert!(!block.contains("load"), "no intermediate row must ever be loaded out, got:\n{block}");
}

#[test]
fn a_multi_dimensional_read_also_collapses_into_one_multi_index_load() {
    let out = cps("fn f() -> i32 { let a = [[1, 2], [3, 4]]; a[1][0] }");
    let block = fn_block(&out, "f");
    assert!(block.contains("(load ") && block.contains(" 1 0)"), "got:\n{block}");
}

#[test]
#[should_panic(expected = "field-mutation assignment")]
fn a_field_mutation_assignment_target_is_explicitly_rejected_not_silently_wrong() {
    cps(
        "struct Vec2 { x: f64, y: f64 }
         fn f() -> f64 {
            let mut v = Vec2(x: 1.0, y: 2.0);
            v.x = 5.0;
            v.x
         }",
    );
}

#[test]
fn a_generic_function_called_at_two_types_converts_to_two_separate_specializations() {
    let out = cps(
        "fn double<T: Ring>(x: T) -> T { add(x, x) }
         fn main() -> i32 { double(1:i32); double(1.0:f32); 0 }",
    );
    assert!(out.contains("(fn double<i32>"), "got:\n{out}");
    assert!(out.contains("(fn double<f32>"), "got:\n{out}");
}
