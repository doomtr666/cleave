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

#[test]
fn a_call_to_a_stdlib_intrinsic_converts_to_one_let_prim() {
    // `add(a, b)` dispatches to the real stdlib `Ring<f64>::add`, a bodyless
    // `#[mlir(mlir_f64_add)]`-tagged concrete impl -- Appel's PRIMOP,
    // straight-line, no continuation needed.
    let out = cps("fn f(a: f64, b: f64) -> f64 { add(a, b) }");
    assert!(out.contains("(fn f (v0 v1 v2)"), "got:\n{out}");
    assert!(out.contains("(let-prim v3: f64 = (mlir_f64_add v0 v1)"), "got:\n{out}");
    assert!(out.contains("(v2 v3)"), "the result must be handed to the return continuation, got:\n{out}");
    assert!(!out.contains("(fix"), "a primitive call needs no synthesized continuation, got:\n{out}");
}

#[test]
fn a_call_to_a_real_function_converts_to_fix_plus_app_with_continuation() {
    let out = cps(
        "fn helper(a: f64, b: f64) -> f64 { add(a, b) }
         fn main() -> f64 { helper(1.0, 2.0) }",
    );
    assert!(out.contains("(fix"), "a real callee must synthesize a local continuation, got:\n{out}");
    assert!(out.contains("(helper 1 2 k$0)"), "the callee must be tail-called with the continuation appended, got:\n{out}");
}

#[test]
fn a_plain_let_of_a_literal_extends_env_without_introducing_a_new_cps_node() {
    let out = cps("fn f() -> f64 { let x = 1.0; x }");
    assert!(!out.contains("let-prim"), "a plain immutable let needs no CPS node at all, got:\n{out}");
    assert!(out.contains("(v0 1)"), "the literal value must be handed directly to the return continuation, got:\n{out}");
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
    // Both arms must tail-call the *same* join label -- not two separately
    // inlined copies of "whatever happens after the if" (which is what a
    // naive CPS conversion, calling `k` directly in each branch instead of
    // through one named continuation, would produce instead). `j$0` appears
    // 3 times total: once as the `Fix`'s own definition, once per arm's own
    // tail call.
    assert_eq!(out.matches("j$0").count(), 3, "got:\n{out}");
    assert!(out.contains("(j$0 1)") && out.contains("(j$0 2)"), "each arm must tail-call the join continuation with its own value, got:\n{out}");
    assert_eq!(out.matches("(mlir_i32_gt").count(), 1, "the condition must be evaluated exactly once, got:\n{out}");
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
    assert!(out.contains("mlir_i32_gt") && out.contains("mlir_i32_sub") && out.contains("mlir_i32_add"), "got:\n{out}");
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
    assert!(out.contains("(fix"), "got:\n{out}");
    // The loop-carried index is the recursive continuation's own single
    // parameter -- bound-check, body, increment, tail-recurse, all inside
    // its own body; the initial call seeds it with `start`.
    assert!(out.contains("mlir_i32_lt"), "the implicit bound check must resolve to Ord::lt, got:\n{out}");
    assert!(out.contains("mlir_i32_add"), "both the loop body's own `add` and the implicit increment use it, got:\n{out}");
    assert!(out.contains("(loop$0 0)"), "the loop must be entered by calling its own continuation with `start`, got:\n{out}");
    // On exit (bound check fails), control must reach the outer continuation
    // with the function's own tail value, not get stuck inside the loop.
    assert!(out.contains("(v0 0)"), "got:\n{out}");
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
    assert!(out.contains("(fix"), "got:\n{out}");
    assert!(out.contains("mlir_i32_gt"), "got:\n{out}");
    // No loop-carried state yet (`let mut` is a later stage) -- the
    // recursive continuation takes no parameters at all.
    assert!(out.contains("(loop$0 ()") || out.contains("(loop$0 ("), "the loop continuation must take no parameters, got:\n{out}");
    assert!(out.contains("(loop$0)"), "both re-entry and recursion must be plain zero-arg tail calls, got:\n{out}");
}

#[test]
fn a_plain_reassignment_in_straight_line_code_just_rebinds_the_name() {
    let out = cps("fn f() -> i32 { let mut x = 1; x = 2; x }");
    assert!(!out.contains("let-prim"), "straight-line reassignment needs no CPS node at all, got:\n{out}");
    assert!(out.contains("(v0 2)"), "the reassigned value must reach the return continuation, got:\n{out}");
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
    // The recursive continuation now carries two values: the index and the
    // accumulator (sorted by name: "acc" before "i").
    assert!(out.contains("(loop$0 (v2 v1)") || out.contains("(loop$0 (v1 v2)"), "got:\n{out}");
    assert!(out.contains("(loop$0 0 0)"), "both the index and the accumulator start at their own initial values, got:\n{out}");
    // On exit, the *accumulator*'s own final value (not the index) must
    // reach the outer continuation.
    assert!(out.contains("(v0 v1)"), "the accumulator's final value must reach the outer continuation, got:\n{out}");
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
    // The join continuation must carry `y` alongside the if's own (Unit)
    // value, and each arm must pass its own newly-assigned value along.
    assert!(out.contains("(j$0 (v4 v3)") || out.contains("(j$0 (v3 v4)"), "got:\n{out}");
    assert!(out.contains("(j$0 () 1)") && out.contains("(j$0 () 2)"), "got:\n{out}");
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
    // The join continuation must carry *only* the if's own result value --
    // one parameter, not two.
    assert!(out.contains("(j$0 (v3)"), "the inner `y` must not leak as carried state, got:\n{out}");
    assert!(out.contains("(j$0 2)") && out.contains("(j$0 0)"), "got:\n{out}");
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
    // Two levels of carrying: the inner if's own join carries `acc`, and the
    // outer for-loop's own recursive continuation also carries `acc`
    // (alongside its own index) -- both `fix`es must appear, nested.
    assert_eq!(out.matches("(fix").count(), 2, "got:\n{out}");
    assert!(out.contains("mlir_i32_add") && out.contains("mlir_i32_gt") && out.contains("mlir_i32_lt"), "got:\n{out}");
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
    assert!(out.contains("(let-prim v1: [i32; 3] = (array 1 2 3)"), "got:\n{out}");
    // The store and the load must both operate on `a`'s own reference (v1)
    // -- no copy, no rebinding through `env` the way a scalar reassignment
    // would need.
    assert!(out.contains("(store v1 0 10)"), "got:\n{out}");
    assert!(out.contains("(load v1 1)"), "got:\n{out}");
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
    assert!(out.contains("(loop$0 (v2)"), "the loop must carry only its own index, got:\n{out}");
    assert!(out.contains("load") && out.contains("store"), "got:\n{out}");
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
    assert!(out.contains("(store v3 0 0 9)"), "both indices must combine into one store, got:\n{out}");
    assert!(!out.contains("load"), "no intermediate row must ever be loaded out, got:\n{out}");
}

#[test]
fn a_multi_dimensional_read_also_collapses_into_one_multi_index_load() {
    let out = cps("fn f() -> i32 { let a = [[1, 2], [3, 4]]; a[1][0] }");
    assert!(out.contains("(load v3 1 0)"), "got:\n{out}");
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
