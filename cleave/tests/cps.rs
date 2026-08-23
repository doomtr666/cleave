use cleave::cps::{
    ConcreteUnit, collect_units, convert_program, dump_cps_program, eliminate_dead_code,
};
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

/// Like `cps`, but returns the raw `ConcreteUnit`s themselves rather than
/// the converted-and-dumped `CpsProgram` — for inspecting a unit's own
/// fields directly (`origin`, etc.), not just its converted body's text.
fn units(src: &str) -> Vec<ConcreteUnit> {
    let (result, _sources) = compile(vec![("test.cleave".to_string(), src.to_string())], &[]);
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    collect_units(&program, &registry)
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
    let start = dump
        .find(&format!("(fn {name} ("))
        .unwrap_or_else(|| panic!("no `(fn {name} (...)` block found in:\n{dump}"));
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

/// Extracts the actual `loop$N`/`j$N`/`k$N` label CPS conversion happened
/// to mint for this block's own `Fix` — like `fn_block`'s own doc comment
/// says of CPS variable numbers, the label *number* itself was never
/// semantically meaningful, only self-consistency was, so it's read fresh
/// from the output here rather than hardcoded. A new stdlib algebra impl
/// with its own `if`/loop (processed, alphabetically, before a lowercase-
/// named user function like `f`/`main` — `convert_program`'s own `fresh`
/// counter is shared across every unit) shifts every later unit's own
/// starting label upward; a test asserting a literal `"loop$0"`/`"j$1"`
/// would silently break the moment the stdlib grows, exactly as it did the
/// day `div`/`mod`/bitwise ops were added (found by direct testing, not
/// hypothetical) — this is the fix, not a one-off patch to the hardcoded
/// numbers themselves.
fn label<'a>(block: &'a str, prefix: &str) -> &'a str {
    let needle = format!("({prefix}$");
    let start = block
        .find(&needle)
        .unwrap_or_else(|| panic!("no `{prefix}$N` label found in:\n{block}"));
    let rest = &block[start + 1..];
    let end = rest.find([' ', ')']).unwrap_or(rest.len());
    &rest[..end]
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
    assert!(
        !block.contains("(fix"),
        "a raw mlir op call needs no synthesized continuation, got:\n{block}"
    );
}

#[test]
fn a_call_to_a_real_function_converts_to_fix_plus_app_with_continuation() {
    let out = cps("fn helper(a: f64, b: f64) -> f64 { add(a, b) }
         fn main() -> f64 { helper(1.0, 2.0) }");
    // Scoped to `main`'s own block specifically: `helper`'s own body *also*
    // makes a real call now (`add` wraps a real `mlir::...` call inside a
    // real fn, not a straight-line intrinsic -- see `stdlib/num/num.cleave`),
    // so it competes with `main` for continuation-label numbers via the
    // same shared counter (`convert_program`'s own `fresh`, iterating units
    // in `HashMap` order -- not guaranteed stable across runs) -- checking
    // a literal `k$0` here would flakily depend on which of the two enters
    // that shared counter first.
    let block = fn_block(&out, "main");
    assert!(
        block.contains("(fix"),
        "a real callee must synthesize a local continuation, got:\n{block}"
    );
    assert!(
        block.contains("(helper 1 2 k$"),
        "the callee must be tail-called with the continuation appended, got:\n{block}"
    );
}

#[test]
fn a_plain_let_of_a_literal_extends_env_without_introducing_a_new_cps_node() {
    let out = cps("fn f() -> f64 { let x = 1.0; x }");
    let block = fn_block(&out, "f");
    assert!(
        !block.contains("let-prim"),
        "a plain immutable let needs no CPS node at all, got:\n{block}"
    );
    // `f`'s own body is `(fn f (vN) (vN 1))` -- whatever `vN` (its own
    // return continuation parameter) actually is, the literal must be
    // handed directly to it.
    let k_ret = block
        .trim_start_matches("(fn f (")
        .split(')')
        .next()
        .unwrap();
    assert!(
        block.contains(&format!("({k_ret} 1)")),
        "the literal value must be handed directly to the return continuation, got:\n{block}"
    );
}

#[test]
fn field_access_and_struct_construction_each_produce_their_own_let_prim_in_order() {
    let out = cps("struct Vec2 { x: f64, y: f64 }
         fn main() -> f64 {
            let v = Vec2(x: 1.0, y: 2.0);
            add(v.x, v.y)
         }");
    let struct_pos = out
        .find("struct.Vec2[x,y]")
        .expect("struct construction must produce its own let-prim");
    let field_x_pos = out
        .find("field.x")
        .expect("field access on x must produce its own let-prim");
    let field_y_pos = out
        .find("field.y")
        .expect("field access on y must produce its own let-prim");
    assert!(
        struct_pos < field_x_pos && field_x_pos < field_y_pos,
        "got:\n{out}"
    );
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
    // continuation -- both arms must tail-call the *same* join label, not
    // two separately inlined copies of "whatever happens after the if"
    // (what a naive CPS conversion, calling `k` directly in each branch
    // instead of through one named continuation, would produce instead).
    // The join label appears 3 times total: once as the `Fix`'s own
    // definition, once per arm's own tail call.
    let j = label(block, "j");
    assert_eq!(block.matches(j).count(), 3, "got:\n{block}");
    assert!(
        block.contains(&format!("({j} 1)")) && block.contains(&format!("({j} 2)")),
        "each arm must tail-call the join continuation with its own value, got:\n{block}"
    );
    assert_eq!(
        block.matches("Ord::gt<i32>").count(),
        1,
        "the condition must be evaluated exactly once, got:\n{block}"
    );
}

#[test]
fn a_bare_if_with_no_else_feeds_unit_to_the_join_continuation() {
    let out = cps("fn f(x: i32) -> i32 { if x > 0 { 1; }; 0 }");
    assert!(
        out.contains("()"),
        "a missing else must join with CVal::Unit, got:\n{out}"
    );
}

#[test]
fn a_self_recursive_generic_function_using_if_else_converts_and_resolves_its_own_recursive_calls() {
    // Mirrors `examples/fibonacci.cleave` -- self-recursion through both
    // arithmetic (Ring) and comparison (Ord) intrinsics, and a real
    // recursive call inside the `then` branch, resolved via `call_names`.
    let out = cps("fn fibonacci<T: Int>(x: T) -> T {
            if x > 2 { fibonacci(x - 1) + fibonacci(x - 2) } else { x }
        }
        fn main() -> i32 { fibonacci(10:i32) }");
    assert!(out.contains("(fn fibonacci<i32>"), "got:\n{out}");
    assert_eq!(
        out.matches("(fibonacci<i32>").count(),
        3,
        "the fn def plus its own two recursive calls, got:\n{out}"
    );
    let block = fn_block(&out, "fibonacci<i32>");
    assert!(
        block.contains("Ord::gt<i32>")
            && block.contains("Ring::sub<i32>")
            && block.contains("Ring::add<i32>"),
        "got:\n{block}"
    );
}

#[test]
fn a_for_loop_becomes_a_self_recursive_continuation_carrying_the_index() {
    let out = cps("fn f() -> i32 {
            for i in 0..10 {
                add(i, i);
            };
            0
        }");
    let block = fn_block(&out, "f");
    assert!(block.contains("(fix"), "got:\n{block}");
    // The loop-carried index is the recursive continuation's own single
    // parameter -- bound-check, body, increment, tail-recurse, all inside
    // its own body; the initial call seeds it with `start`.
    assert!(
        block.contains("Ord::lt<i32>"),
        "the implicit bound check must resolve to Ord::lt, got:\n{block}"
    );
    assert!(
        block.contains("Ring::add<i32>"),
        "both the loop body's own `add` and the implicit increment use it, got:\n{block}"
    );
    let l = label(block, "loop");
    assert!(
        block.contains(&format!("({l} 0)")),
        "the loop must be entered by calling its own continuation with `start`, got:\n{block}"
    );
    // On exit (bound check fails), control must reach the outer continuation
    // (`f`'s own return-continuation parameter) with the function's own
    // tail value, not get stuck inside the loop.
    let k_ret = block
        .trim_start_matches("(fn f (")
        .split(')')
        .next()
        .unwrap();
    assert!(block.contains(&format!("({k_ret} 0)")), "got:\n{block}");
}

#[test]
fn a_while_loop_carries_no_state_and_reevaluates_its_condition_each_iteration() {
    let out = cps("fn f(x: i32) -> i32 {
            while x > 0 {
                add(x, x);
            };
            1
        }");
    let block = fn_block(&out, "f");
    assert!(block.contains("(fix"), "got:\n{block}");
    assert!(block.contains("Ord::gt<i32>"), "got:\n{block}");
    // No loop-carried state yet (`let mut` is a later stage) -- the
    // recursive continuation takes no parameters at all.
    let l = label(block, "loop");
    assert!(
        block.contains(&format!("({l} ()")) || block.contains(&format!("({l} (")),
        "the loop continuation must take no parameters, got:\n{block}"
    );
    assert!(
        block.contains(&format!("({l})")),
        "both re-entry and recursion must be plain zero-arg tail calls, got:\n{block}"
    );
}

#[test]
fn a_plain_reassignment_in_straight_line_code_just_rebinds_the_name() {
    let out = cps("fn f() -> i32 { let mut x = 1; x = 2; x }");
    let block = fn_block(&out, "f");
    assert!(
        !block.contains("let-prim"),
        "straight-line reassignment needs no CPS node at all, got:\n{block}"
    );
    let k_ret = block
        .trim_start_matches("(fn f (")
        .split(')')
        .next()
        .unwrap();
    assert!(
        block.contains(&format!("({k_ret} 2)")),
        "the reassigned value must reach the return continuation, got:\n{block}"
    );
}

#[test]
fn a_for_loop_accumulator_is_carried_as_an_extra_loop_parameter() {
    let out = cps("fn f() -> i32 {
            let mut acc = 0;
            for i in 0..10 {
                acc = add(acc, i);
            };
            acc
        }");
    let block = fn_block(&out, "f");
    // The recursive continuation now carries two values: the index and the
    // accumulator, index first positionally, whatever their own actual var
    // numbers are (offset by however many the stdlib prelude's own
    // conversion already consumed, see `fn_block`'s own doc comment).
    let l = label(block, "loop");
    let params = block
        .split(&format!("({l} ("))
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap();
    let mut params = params.split_whitespace();
    let (index_var, acc_var) = (params.next().unwrap(), params.next().unwrap());
    assert!(
        block.contains(&format!("({l} 0 0)")),
        "both the index and the accumulator start at their own initial values, got:\n{block}"
    );
    // On exit, the *accumulator*'s own final value (not the index) must
    // reach the outer continuation (`f`'s own return-continuation param).
    let k_ret = block
        .trim_start_matches("(fn f (")
        .split(')')
        .next()
        .unwrap();
    assert!(
        block.contains(&format!("({k_ret} {acc_var})")),
        "got:\n{block}"
    );
    let _ = index_var; // only its presence as the loop's own first param matters here
}

#[test]
fn an_if_else_mutating_a_let_mut_variable_threads_it_through_the_join() {
    let out = cps("fn f(x: i32) -> i32 {
            let mut y = 0;
            if x > 0 { y = 1; } else { y = 2; };
            y
        }");
    let block = fn_block(&out, "f");
    // `x > 0` is itself a real call, so the join label isn't necessarily
    // `j$0` -- see `an_if_else_expression_joins_both_arms_through_one_
    // synthesized_continuation`'s own comment (and `label`'s own doc
    // comment for why this reads the actual label fresh from the output).
    // The join continuation must carry `y` alongside the if's own (Unit)
    // value, and each arm must pass its own newly-assigned value along.
    let j = label(block, "j");
    assert!(
        block.contains(&format!("({j} (v")) && block.matches(j).count() == 3,
        "got:\n{block}"
    );
    assert!(
        block.contains(&format!("({j} () 1)")) && block.contains(&format!("({j} () 2)")),
        "got:\n{block}"
    );
}

#[test]
fn a_variable_shadowed_by_an_inner_let_mut_does_not_leak_into_the_outer_joins_carried_set() {
    // `y` inside the `then` branch is a *fresh*, inner `let mut` shadowing
    // nothing -- it must not be mistaken for an escaping mutation of some
    // enclosing `y` (there isn't one here at all).
    let out = cps("fn f(x: i32) -> i32 {
            if x > 0 {
                let mut y = 1;
                y = 2;
                y
            } else {
                0
            }
        }");
    let block = fn_block(&out, "f");
    // `x > 0` is itself a real call, so the join label isn't necessarily
    // `j$0` (see the comment on the previous test / `label`'s own doc
    // comment). The join continuation must carry *only* the if's own
    // result value -- one parameter, not two.
    let j = label(block, "j");
    let join_params = block
        .split(&format!("({j} ("))
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap();
    assert_eq!(
        join_params.split_whitespace().count(),
        1,
        "the inner `y` must not leak as carried state, got:\n{block}"
    );
    assert!(
        block.contains(&format!("({j} 2)")) && block.contains(&format!("({j} 0)")),
        "got:\n{block}"
    );
}

#[test]
fn mutation_inside_an_if_nested_in_a_for_loop_composes_correctly() {
    let out = cps("fn f() -> i32 {
            let mut acc = 0;
            for i in 0..10 {
                if i > 5 {
                    acc = add(acc, i);
                };
            };
            acc
        }");
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
    assert!(
        block.contains("Ring::add<i32>")
            && block.contains("Ord::gt<i32>")
            && block.contains("Ord::lt<i32>"),
        "got:\n{block}"
    );
}

#[test]
fn an_array_literal_an_indexed_write_and_an_indexed_read_use_the_same_stable_reference() {
    let out = cps("fn f() -> i32 {
            let mut a = [1, 2, 3];
            a[0] = 10;
            a[1]
        }");
    let block = fn_block(&out, "f");
    assert!(
        block.contains(": [i32; 3] = (array 1 2 3)"),
        "got:\n{block}"
    );
    // `a`'s own array-reference variable, whatever it actually is.
    let a_var = block
        .split(": [i32; 3] = (array 1 2 3)")
        .next()
        .unwrap()
        .rsplit("(let-prim ")
        .next()
        .unwrap();
    // The store and the load must both operate on `a`'s own reference --
    // no copy, no rebinding through `env` the way a scalar reassignment
    // would need.
    assert!(
        block.contains(&format!("(store {a_var} 0 10)")),
        "got:\n{block}"
    );
    assert!(
        block.contains(&format!("(load {a_var} 1)")),
        "got:\n{block}"
    );
}

#[test]
fn an_array_repeat_whose_count_names_a_const_generic_resolves_to_three_independent_elements() {
    // A *literal* repeat count (`[0; 3]`) desugars to an ordinary
    // `ArrayLit` at lowering time (see `ast.rs`'s own `ExprKind::ArrayRepeat`
    // doc comment) -- `ArrayRepeat` itself only ever survives to CPS
    // conversion when the count names a const generic instead, and by this
    // point monomorphization has already resolved that reference down to a
    // concrete `Ty::Const` (found by direct testing: an earlier version of
    // this module only ever looked a `Path` up in `env`, panicking with
    // "unbound variable `N`" the first time this case was actually
    // exercised).
    //
    // Converts to a plain `PrimOp::Array` with three (independently
    // converted) elements now, not `PrimOp::ArrayRepeat` -- `cps.rs::
    // convert_array_repeat_over_resolved_dims`'s own doc comment has the
    // full story: evaluating `value` once and broadcasting it, the former
    // behavior, is only correct for a referentially transparent `value`;
    // `v` here (a bare parameter reference) is exactly that, so the three
    // elements are identical (`v454 v454 v454`, all the same variable) --
    // a real, distinct call per element (`rand.cleave`'s own `uniform`, its
    // real motivating case) would instead produce three independent calls.
    let out = cps("fn make<const N: i32>(v: i32) -> [i32; N] { [v; N] }
         fn f() -> i32 { make::<3>(0); 0 }");
    let block = fn_block(&out, "make<3>");
    assert!(
        block.contains("(array "),
        "the count must resolve to a real 3-element array, not stay an unbound name, got:\n{block}"
    );
    assert!(!out.contains("unbound variable"), "got:\n{out}");
}

#[test]
fn an_array_mutated_inside_a_for_loop_body_needs_no_carrying_only_the_index_does() {
    // The whole point of the "stable reference" design: `a`'s own identity
    // never changes across iterations, so unlike a scalar accumulator
    // (`a_for_loop_accumulator_is_carried_as_an_extra_loop_parameter` above)
    // the loop's own recursive continuation carries *only* its index.
    let out = cps("fn f() -> i32 {
            let a = [0, 0, 0];
            for i in 0..3 {
                a[i] = add(a[i], 1);
            };
            a[0]
        }");
    let block = fn_block(&out, "f");
    let l = label(block, "loop");
    let loop_params = block
        .split(&format!("({l} ("))
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap();
    assert_eq!(
        loop_params.split_whitespace().count(),
        1,
        "the loop must carry only its own index, got:\n{block}"
    );
    assert!(
        block.contains("load") && block.contains("store"),
        "got:\n{block}"
    );
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
    assert!(
        block.contains(" 0 0 9)") && block.contains("(store "),
        "both indices must combine into one store, got:\n{block}"
    );
    assert!(
        !block.contains("load"),
        "no intermediate row must ever be loaded out, got:\n{block}"
    );
}

#[test]
fn a_multi_dimensional_read_also_collapses_into_one_multi_index_load() {
    let out = cps("fn f() -> i32 { let a = [[1, 2], [3, 4]]; a[1][0] }");
    let block = fn_block(&out, "f");
    assert!(
        block.contains("(load ") && block.contains(" 1 0)"),
        "got:\n{block}"
    );
}

/// A direct field-mutation assignment (`v.x = 5.0`) converts to a single
/// `PrimOp::FieldStore` — a real effect through `v`'s own existing pointer
/// (a struct is a stable reference, mutated in place, same as an array —
/// see `cps.rs`'s own "Arrays" doc comment), not a functional rebuild of
/// `v` itself, so no join/carried-state threading is involved.
#[test]
fn a_field_mutation_assignment_converts_to_one_field_store() {
    let out = cps("struct Vec2 { x: f64, y: f64 }
         fn f() -> f64 {
            let mut v = Vec2(x: 1.0, y: 2.0);
            v.x = 5.0;
            v.x
         }");
    let block = fn_block(&out, "f");
    assert!(block.contains("field-store.x"), "got:\n{block}");
}

#[test]
fn a_generic_function_called_at_two_types_converts_to_two_separate_specializations() {
    let out = cps("fn double<T: Ring>(x: T) -> T { add(x, x) }
         fn main() -> i32 { double(1:i32); double(1.0:f32); 0 }");
    assert!(out.contains("(fn double<i32>"), "got:\n{out}");
    assert!(out.contains("(fn double<f32>"), "got:\n{out}");
}

// ------------------------------------------------------------ closure conversion

/// A `let`-bound lambda with a captured variable converts to a call whose
/// own leading argument is that capture's `CVal`, gathered at the `let`
/// itself — `add_base(5)`'s own call site must pass `base` *and* `5`, in
/// that order, to `<lambda...>`'s own generated unit (see `ConcreteUnit`'s
/// own widened `params`, `collect_units`'s "Every `let`-bound lambda's own
/// specialization" doc comment).
#[test]
fn a_lambda_call_with_a_captured_variable_passes_the_capture_as_a_leading_argument() {
    let out =
        cps("fn main() -> i32 { let base = 100; let add_base = fn(x) { x + base }; add_base(5) }");
    let block = fn_block(&out, "main");
    // The `fix`'s own continuation body tail-calls the lambda's own
    // generated unit (`<lambda#N><i32>`) with two arguments, `base`'s own
    // value first -- `100` -- then the real argument, `5`.
    assert!(
        block.contains("<lambda") && block.contains("100 5"),
        "got:\n{block}"
    );
}

/// A lambda that captures *nothing* generates a unit with exactly one
/// (non-continuation) declared parameter -- no leading capture arguments at
/// all, distinguishing it structurally from the captured-variable case
/// above.
#[test]
fn a_lambda_with_no_captures_generates_a_unit_with_no_leading_capture_params() {
    let out = cps("fn main() -> i32 { let f = fn(x) { x + 1 }; f(5) }");
    let lambda_block_start = out
        .find("(fn <lambda")
        .unwrap_or_else(|| panic!("no lambda unit found in:\n{out}"));
    let params_line = &out[lambda_block_start..].lines().next().unwrap();
    // `(fn <lambda#N><i32> (vA vB)` -- vA is the lambda's own `x`, vB is
    // `k_ret`; a captured variable would insert a third leading `v` before
    // both.
    let param_count = params_line
        .split('(')
        .nth(2)
        .unwrap()
        .split(')')
        .next()
        .unwrap()
        .split_whitespace()
        .count();
    assert_eq!(
        param_count, 2,
        "expected exactly [x, k_ret], got:\n{params_line}"
    );
}

/// Calling a `let`-bound lambda's own name resolves via `CVal::Closure`
/// (Stage A), independent of whether a same-named top-level `fn` also
/// exists — the local binding must shadow it, mirroring ordinary lexical
/// scoping for any other name.
#[test]
fn a_lambda_bound_name_shadows_a_same_named_top_level_fn() {
    let out = cps("fn shadowed(x: i32) -> i32 { x + 1000 }
         fn main() -> i32 { let shadowed = fn(x) { x + 1 }; shadowed(5) }");
    let block = fn_block(&out, "main");
    // Resolves to the lambda's own generated unit, never the top-level
    // `shadowed` -- if it fell through to the top-level `fn` instead, this
    // call would read `(shadowed 5 ...)`, not a `<lambda...>` unit name.
    assert!(block.contains("<lambda"), "got:\n{block}");
    assert!(
        !block.contains("(shadowed "),
        "must not call the shadowed top-level fn, got:\n{block}"
    );
}

/// A generic `let`-bound lambda (`id`), called at two different concrete
/// types from two different call sites, gets two separate specializations —
/// real Hindley-Milner let-polymorphism for a lambda, exactly mirroring
/// `a_generic_function_called_at_two_types_converts_to_two_separate_
/// specializations` above for a top-level `fn`.
#[test]
fn a_generic_lambda_called_at_two_types_converts_to_two_separate_specializations() {
    let out = cps("fn main() -> i32 {
            let id = fn(x) { x };
            let a = id(1:i32);
            let b = id(1.0:f32);
            0
         }");
    assert!(
        out.contains("(fn <lambda") && out.matches("(fn <lambda").count() == 2,
        "got:\n{out}"
    );
}

/// Stage B: `apply`'s own body (`f(x)`) is never itself converted — a unit
/// with a `Ty::Fn`-typed parameter has no runtime representation to call it
/// through, so only its per-callable specializations (`apply[f=...]`) are
/// ever emitted. Confirms `convert_program`'s own skip (see its doc comment
/// on the `Ty::Fn` parameter check) rather than a stray `(fn apply (...)`
/// with an unresolved call inside it.
#[test]
fn the_original_unspecialized_higher_order_callee_is_never_itself_converted() {
    let out = cps("fn apply(f: (i32) -> i32, x: i32) -> i32 { f(x) }
         fn main() -> i32 { let inc = fn(x) { x + 1 }; apply(inc, 5) }");
    assert!(
        !out.contains("(fn apply ("),
        "the un-specialized `apply` must never be emitted, got:\n{out}"
    );
    assert!(out.contains("(fn apply[f="), "got:\n{out}");
}

/// The two open risks the closure-conversion plan flagged explicitly as
/// needing an explicit guard, not a silent misconversion — a bare lambda
/// *literal* (no prior `let`) used directly as a call argument. Neither
/// Stage A's own lambda-call resolution (needs a `let`-bound name) nor
/// Stage B's own higher-order-argument detection (needs a `Path` argument)
/// recognizes this shape at all — it falls through to `convert_expr`'s own
/// `Lambda` catch-all, which panics clearly rather than silently producing
/// a wrong (or missing) conversion.
#[test]
#[should_panic(expected = "CPS doesn't support")]
fn a_bare_lambda_literal_passed_directly_as_an_argument_panics_cleanly() {
    cps("fn apply(f: (i32) -> i32, x: i32) -> i32 { f(x) }
         fn main() -> i32 { apply(fn(x) { x + 1 }, 5) }");
}

// ------------------------------------------------------------ egg integration (Stage 1)

/// A `ConcreteUnit` built from an algebra impl's own method carries its own
/// `(algebra, method)` origin, structurally — not something a later pass
/// has to parse back out of the unit's own display name (`"Ring::add<i32>"`,
/// a one-way `format!`, see `monomorphize.rs::display_impl_instantiation`).
#[test]
fn an_algebra_impl_units_own_origin_names_its_algebra_and_method() {
    let all = units("fn main() -> i32 { 1 + 2 }");
    let add = all
        .iter()
        .find(|u| u.name == "Ring::add<i32>")
        .unwrap_or_else(|| {
            panic!(
                "no `Ring::add<i32>` unit found among: {:?}",
                all.iter().map(|u| &u.name).collect::<Vec<_>>()
            )
        });
    assert_eq!(add.origin, Some(("Ring".to_string(), "add".to_string())));
}

/// An ordinary top-level `fn`'s own unit has no algebra origin at all.
#[test]
fn a_top_level_fns_own_unit_has_no_origin() {
    let all = units("fn f() -> i32 { 1 } fn main() -> i32 { f() }");
    let f = all.iter().find(|u| u.name == "f").unwrap();
    assert_eq!(f.origin, None);
}

/// `CTopLevelFn` carries its own `origin` too, threaded straight through
/// from `ConcreteUnit::origin` (Stage 1) — a later e-graph pass, operating
/// on the *converted* `CpsProgram` rather than the pre-conversion
/// `ConcreteUnit`s, needs it at that point instead, and it's still not
/// something worth re-deriving by parsing a unit's own display name.
#[test]
fn a_top_level_fns_own_origin_survives_cps_conversion() {
    let (result, _sources) = compile(
        vec![(
            "test.cleave".to_string(),
            "fn main() -> i32 { 1 + 2 }".to_string(),
        )],
        &[],
    );
    let program = result.unwrap();
    let registry = Registry::build(&program);
    let all_units = collect_units(&program, &registry);
    let cps_program = convert_program(all_units);
    let add = cps_program
        .funcs
        .iter()
        .find(|f| f.def.name == "Ring::add<i32>")
        .unwrap_or_else(|| {
            panic!(
                "no `Ring::add<i32>` unit found among: {:?}",
                cps_program
                    .funcs
                    .iter()
                    .map(|f| &f.def.name)
                    .collect::<Vec<_>>()
            )
        });
    assert_eq!(add.origin, Some(("Ring".to_string(), "add".to_string())));
}

// -------------------------------------------------- dead-code elimination

/// A top-level `fn` never called from `main` (directly or transitively)
/// gets dropped; one that *is* reachable, and `main` itself, survive.
#[test]
fn dead_code_elimination_drops_an_unused_top_level_fn_but_keeps_reachable_ones() {
    let (result, _sources) = compile(
        vec![("test.cleave".to_string(), "fn unused(x: i32) -> i32 { x }\nfn used(x: i32) -> i32 { x }\nfn main() -> i32 { used(1) }".to_string())],
        &[],
    );
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    let cps_program = convert_program(collect_units(&program, &registry));
    let cps_program = eliminate_dead_code(cps_program);
    let names: Vec<&str> = cps_program
        .funcs
        .iter()
        .map(|f| f.def.name.as_str())
        .collect();
    assert!(names.contains(&"main"), "got: {names:?}");
    assert!(names.contains(&"used"), "got: {names:?}");
    assert!(!names.contains(&"unused"), "got: {names:?}");
}

/// An `export fn` is, by its own definition, a second root alongside
/// `main` -- an external host calls it directly, `main` itself may never
/// call it at all. `eliminate_dead_code`'s own worklist must seed every
/// exported unit's name, not just `"main"`, or a real, intentional export
/// with no cleave-side caller would be silently deleted before
/// `mlir_lower.rs` ever saw it.
#[test]
fn dead_code_elimination_keeps_an_export_fn_unreachable_from_main() {
    let (result, _sources) = compile(
        vec![(
            "test.cleave".to_string(),
            "export fn kernel(x: i32) -> i32 { x }\nfn main() -> i32 { 1 }".to_string(),
        )],
        &[],
    );
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    let cps_program = convert_program(collect_units(&program, &registry));
    let cps_program = eliminate_dead_code(cps_program);
    let names: Vec<&str> = cps_program
        .funcs
        .iter()
        .map(|f| f.def.name.as_str())
        .collect();
    assert!(names.contains(&"main"), "got: {names:?}");
    assert!(
        names.contains(&"kernel"),
        "export fn `kernel` has no cleave-side caller and must still survive DCE, got: {names:?}"
    );
}

/// The real motivating case (`doc/backlog.md`'s own "Dead-code elimination
/// for unused stdlib specializations" item): `stdlib/num/num.cleave`'s
/// width specializations (`Ring`/`Ord` × 6 widths) are unconditionally
/// collected by `collect_units` regardless of what the program actually
/// uses — DCE must remove every one the program never reaches, keeping
/// only the ones it does.
#[test]
fn dead_code_elimination_drops_unreached_stdlib_specializations() {
    let (result, _sources) = compile(
        vec![(
            "test.cleave".to_string(),
            "fn main() -> i32 { 1 + 2 }".to_string(),
        )],
        &[],
    );
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    let cps_program = convert_program(collect_units(&program, &registry));
    let before = cps_program.funcs.len();
    let cps_program = eliminate_dead_code(cps_program);
    let after = cps_program.funcs.len();
    assert!(
        after < before,
        "expected DCE to remove unreached stdlib specializations, before={before} after={after}"
    );
    let names: Vec<&str> = cps_program
        .funcs
        .iter()
        .map(|f| f.def.name.as_str())
        .collect();
    assert!(names.contains(&"Ring::add<i32>"), "got: {names:?}");
    assert!(
        !names.contains(&"Ring::add<f64>"),
        "the program never touches f64, got: {names:?}"
    );
}

/// A second real bug, found by direct testing (`examples/axiom_demo.cleave`):
/// `main.rs`'s own pipeline used to run DCE only *once*, before `egraph::
/// optimize_program` — but the axiom pass can itself fold away every
/// remaining call to a stdlib specialization (`10 + x - 10` reducing to `x`
/// via `add_commutative`/`add_sub_assoc`/constant-fold/`add_zero`), which
/// the first sweep has no way to anticipate, since it runs *before*
/// optimization ever happens. `Ring::add<i32>`/`Ring::sub<i32>` survived a
/// single DCE pass despite ending up with zero real callers — a second
/// sweep, run *after* `optimize_program`, is needed to actually remove them.
#[test]
fn dead_code_elimination_after_optimization_drops_specializations_the_axioms_folded_away() {
    let (result, _sources) = compile(
        vec![(
            "test.cleave".to_string(),
            "fn helper(x: i32) -> i32 { let y = 10; 10 + x - y }\nfn main() -> i32 { helper(21) }"
                .to_string(),
        )],
        &[],
    );
    let program = result.unwrap_or_else(|e| panic!("compile failed: {e:?}"));
    let registry = Registry::build(&program);
    let cps_program = convert_program(collect_units(&program, &registry));
    let cps_program = eliminate_dead_code(cps_program);
    let names_before: Vec<&str> = cps_program
        .funcs
        .iter()
        .map(|f| f.def.name.as_str())
        .collect();
    assert!(
        names_before.contains(&"Ring::add<i32>"),
        "a single DCE pass, before optimization, can't know these are about to become dead: {names_before:?}"
    );
    assert!(
        names_before.contains(&"Ring::sub<i32>"),
        "got: {names_before:?}"
    );

    let (optimized, _) = cleave::egraph::optimize_program(cps_program, &registry);
    let optimized = eliminate_dead_code(optimized);
    let names_after: Vec<&str> = optimized
        .funcs
        .iter()
        .map(|f| f.def.name.as_str())
        .collect();
    assert!(
        !names_after.contains(&"Ring::add<i32>"),
        "the axioms folded away every real call to add<i32>, got: {names_after:?}"
    );
    assert!(
        !names_after.contains(&"Ring::sub<i32>"),
        "the axioms folded away every real call to sub<i32>, got: {names_after:?}"
    );
    assert!(names_after.contains(&"helper"), "got: {names_after:?}");
    assert!(names_after.contains(&"main"), "got: {names_after:?}");
}
